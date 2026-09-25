"""Measure matched client-only and two-worker offset baselines on tiny W8A8.

This is not a BenchmarkResult: the reference can run in one process or as two
co-located loopback children. Neither establishes independent operators or
meters complete wire/setup/aggregate compute.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

import pllm
from pllm.roles import client_only_reference_graph, two_online_reference_graph
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_cluster import LocalOffsetCluster
from pllm.runtime.offset_reference import TwoOnlineOffsetReference, TwoOnlineOffsetTransport
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

_INPUT_IDS = (0, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("qwen2", "qwen3"), default="qwen2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--offset-backend", choices=("in-process", "loopback"),
                        default="in-process")
    parser.add_argument("--cohort", choices=("raw-two-token", "chat"),
                        default="raw-two-token")
    parser.add_argument("--include-prepared", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    if args.include_prepared and (args.cohort != "chat" or args.offset_backend != "loopback"):
        parser.error("--include-prepared requires --cohort chat --offset-backend loopback")

    with tempfile.TemporaryDirectory(prefix="pllm-offset-reference-") as root:
        checkpoint = create_tiny_llama_checkpoint(
            Path(root) / "model", num_hidden_layers=1,
            model_type=args.model_type, with_qkv_bias=args.model_type == "qwen2",
            qk_norm=args.model_type == "qwen3",
        )
        model_id = "tiny-offset-reference"
        checkpoint_artifact_bytes = sum(
            path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
        )
        manifest = load_hf_directory(checkpoint, model_id=model_id)
        first = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
        second = (
            MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
            if args.offset_backend == "in-process" else None
        )
        asyncio.run(first.load(manifest))
        if second is not None:
            asyncio.run(second.load(manifest))
        bundle_payload = first.client_bundle(model_id)
        bundle = ClientBundle.unpack(bundle_payload)
        input_ids = (
            tuple(bundle.tokenizer().encode(
                bundle.render_prompt([{"role": "user", "content": "A"}]),
                add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True)),
            ))
            if args.cohort == "chat" else _INPUT_IDS
        )
        maximum_input_tokens = 64 if args.cohort == "chat" else 4
        if not 1 <= len(input_ids) <= maximum_input_tokens:
            raise ValueError("matched chat cohort exceeds the bounded decoder plan")
        quantized_weight_bytes = sum(
            stage.weight.values.nbytes + stage.weight.scales.nbytes
            + (0 if stage.bias is None else stage.bias.nbytes)
            for stage in first.models[model_id].stages.values()
        )
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        plan = pllm.lower_model(
            config, batch=1, max_input_tokens=maximum_input_tokens, max_new_tokens=2,
        )
        compiled = compile_runtime_model(plan, bundle)

        def exchange(worker: MaskedTransformerEngine):
            def call(stage_id: str, payloads: list[bytes]) -> list[bytes]:
                stage = worker.models[model_id].stages[stage_id].spec
                return asyncio.run(worker.execute_stage(model_id, stage, payloads))
            return call

        client_stage_work = 0

        def client_only(stage_id: str, activation: np.ndarray) -> np.ndarray:
            nonlocal client_stage_work
            stage = first.models[model_id].stages[stage_id]
            quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
            integer = stage.compiled_weight.clear(quantized.values)
            client_stage_work += (
                quantized.rows * stage.spec.in_features * stage.spec.out_features
            )
            output = dequantize_matmul(
                integer, quantized.scales, stage.weight.scales,
                output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
            )
            if stage.bias is not None:
                output += stage.bias
            return np.ascontiguousarray(output, dtype=np.float32)

        def measure(index: int, cluster: LocalOffsetCluster | None) -> dict[str, Any]:
            nonlocal client_stage_work
            outcomes: dict[str, tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
            sample: dict[str, Any] = {}
            for method in (
                ("client_only", "offset") if index % 2 == 0 else ("offset", "client_only")
            ):
                offset: TwoOnlineOffsetReference | TwoOnlineOffsetTransport | None = None
                if method == "client_only":
                    client_stage_work = 0
                    remote = client_only
                elif cluster is not None:
                    offset = TwoOnlineOffsetTransport(
                        compiled, model_id=model_id,
                        worker_a=cluster.clients[0], worker_b=cluster.clients[1],
                        api_key_a=cluster.keys[0], api_key_b=cluster.keys[1],
                    )
                    remote = offset
                else:
                    assert second is not None
                    offset = TwoOnlineOffsetReference(
                        compiled, first, second, model_id=model_id,
                        exchange_a=exchange(first), exchange_b=exchange(second),
                    )
                    remote = offset
                before_workers = (
                    cluster.snapshot_process_metrics()
                    if method == "offset" and cluster is not None else None
                )
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                try:
                    session = compiled.session(remote)
                    session.prefill_ids(input_ids)
                    prefill = session.logits
                    selected = session.select_next()
                    session.decode_selected()
                    decode = session.logits
                    following = session.select_next()
                    session.finish()
                    if isinstance(offset, TwoOnlineOffsetTransport):
                        offset.complete()
                except BaseException:
                    if isinstance(offset, TwoOnlineOffsetTransport):
                        offset.abort()
                    raise
                wall_elapsed = time.perf_counter_ns() - wall_start
                cpu_elapsed = time.process_time_ns() - cpu_start
                sample[f"{method}_online_wall_ms"] = wall_elapsed / 1_000_000
                sample[f"{method}_online_cpu_ms"] = cpu_elapsed / 1_000_000
                if before_workers is not None:
                    assert cluster is not None
                    after_workers = cluster.snapshot_process_metrics()
                    cpu_by_role = {"client": cpu_elapsed / 1_000_000_000}
                    for role in ("worker_a", "worker_b"):
                        previous = before_workers[role]["cpu_ns"]
                        current = after_workers[role]["cpu_ns"]
                        assert previous is not None and current is not None
                        assert current >= previous
                        cpu_by_role[role] = (current - previous) / 1_000_000_000
                    sample["offset_online_cpu_seconds_by_role"] = cpu_by_role
                    sample["offset_aggregate_online_cpu_seconds"] = sum(cpu_by_role.values())
                    sample["offset_worker_lifetime_peak_rss_bytes"] = {
                        role: after_workers[role]["peak_rss_bytes"]
                        for role in ("worker_a", "worker_b")
                    }
                outcomes[method] = ((selected, following), (prefill, decode))
                if method == "client_only":
                    sample["client_only_body_integer_macs"] = client_stage_work
                    sample["client_only_online_network_bytes"] = 0
                if offset is not None:
                    cost = offset.costs
                    sample["offset_stage_calls"] = cost.stages
                    sample["offset_integer_macs"] = cost.total_integer_macs
                    sample["offset_stage_body_bytes"] = cost.total_stage_body_bytes
                    sample["offset_per_edge_bytes"] = {
                        "client_to_worker_a": cost.client_to_worker_a_bytes,
                        "worker_a_to_client": cost.worker_a_to_client_bytes,
                        "client_to_worker_b": cost.client_to_worker_b_bytes,
                        "worker_b_to_client": cost.worker_b_to_client_bytes,
                    }
                    sample["offset_worker_stage_ns"] = {
                        "worker_a": cost.worker_a_stage_ns,
                        "worker_b": cost.worker_b_stage_ns,
                    }
                    if isinstance(offset, TwoOnlineOffsetTransport):
                        link_bodies = offset.http_body_costs
                        sample["offset_http_body_bytes_by_worker"] = link_bodies
                        sample["offset_http_body_bytes_all_links"] = sum(
                            sum(values.values()) for values in link_bodies.values()
                        )
                        assert sum(
                            values["online_upload_bytes"] + values["online_download_bytes"]
                            for values in link_bodies.values()
                        ) == cost.total_stage_body_bytes
            sample["same_selected_tokens"] = (
                outcomes["client_only"][0] == outcomes["offset"][0]
            )
            sample["worst_logit_difference"] = max(
                float(np.max(np.abs(reference - candidate)))
                for reference, candidate in zip(
                    outcomes["client_only"][1], outcomes["offset"][1], strict=True,
                )
            )
            return sample

        cluster_context = (
            LocalOffsetCluster(
                checkpoint, model_id=model_id, weight_bits=8, activation_bits=8,
            )
            if args.offset_backend == "loopback" else contextlib.nullcontext(None)
        )
        samples: list[dict[str, Any]] = []
        with cluster_context as cluster:
            for index in range(args.repeats):
                samples.append(measure(index, cluster))

        prepared: dict[str, Any] | None = None
        if args.include_prepared:
            from pllm import Deployment, ExecutionBudget, Experiment, Model
            from pllm.profiles import MaskedLinearCpu
            from pllm.runtime.benchmark_cli import run_loopback_benchmark

            experiment = Experiment(
                name="matched prepared baseline",
                pipeline=MaskedLinearCpu(Model.path(str(checkpoint), model_id=model_id)),
                deployment=Deployment.local(root=str(Path(root) / "roles")),
                budget=ExecutionBudget(
                    requests=args.repeats, max_input_tokens=64, max_new_tokens=2,
                ),
            )
            control = run_loopback_benchmark(
                model=str(checkpoint), model_id=model_id, tiny=False,
                prompt="A", max_output_tokens=2, warmups=0,
                repetitions=args.repeats, timeout_seconds=180.0,
                experiment=experiment,
            )
            expected_body = bundle.manifest["metadata"]["body_fingerprint"]
            runs = control["runs"]
            accounting = control["topology_accounting"]["runs"]
            initial = control["topology_accounting"]["startup"]
            matched = [
                (run["model_fingerprint"] == expected_body,
                 run["tokens"]["input_tokens"] == len(input_ids),
                 run["tokens"]["output_tokens"] == 2)
                for run in runs
            ]
            if (
                not control["checks"]["passed"] or len(runs) != args.repeats
                or len(accounting) != args.repeats or not all(all(item) for item in matched)
                or initial is None or not initial["tracked_body_counter_set_present"]
                or any(
                    entry is None or not entry["tracked_body_counter_set_present"]
                    for entry in accounting
                )
            ):
                raise RuntimeError(
                    "prepared baseline cohort mismatch: "
                    f"runs={len(runs)}, records={len(accounting)}, "
                    f"checks={control['checks']['passed']}, "
                    f"body/input/output_matches={matched}, "
                    f"compiled_body={expected_body}, "
                    f"prepared_body={runs[0]['model_fingerprint'] if runs else None}, "
                    f"source_manifest={first.models[model_id].manifest.fingerprint}"
                )
            prepared = {
                "configuration_digest": experiment.configuration_digest(),
                "model_fingerprint": expected_body,
                "input_token_count": len(input_ids),
                "generated_token_count": 2,
                "completed_runs": len(runs),
                "body_counter_set_present": all(
                    entry is not None and entry["tracked_body_counter_set_present"]
                    for entry in accounting
                ),
                "initial_inventory_body_bytes_by_edge": initial["body_bytes_by_edge"],
                "initial_inventory_all_link_body_bytes": initial["all_link_serialized_body_bytes"],
                "recorded_body_bytes_including_initial_inventory": (
                    initial["all_link_serialized_body_bytes"]
                    + sum(entry["all_link_serialized_body_bytes"] for entry in accounting)
                ),
                "recorded_offline_body_bytes_during_runs": sum(
                    entry["all_link_serialized_body_bytes"]
                    - entry["online_all_link_serialized_body_bytes"]
                    for entry in accounting
                ),
                "median_online_client_body_bytes": statistics.median(
                    entry["online_client_serialized_body_bytes"] for entry in accounting
                ),
                "median_online_all_link_body_bytes": statistics.median(
                    entry["online_all_link_serialized_body_bytes"] for entry in accounting
                ),
                "median_online_aggregate_cpu_seconds": (
                    statistics.median(
                        entry["aggregate_run_window_cpu_seconds"] for entry in accounting
                    ) if all(
                        entry["aggregate_run_window_cpu_seconds"] is not None
                        for entry in accounting
                    ) else None
                ),
                "offline_inventory_before_online_capture": True,
                "generated_selection_compared": False,
            }

        print(json.dumps({
            "schema": "pllm.topology_reference_benchmark.v4",
            "scope": (
                "client_only_and_two_co_located_loopback_workers; stage_bodies_not_total_wire"
                if args.offset_backend == "loopback"
                else "client_only_and_in_process_offset; not_deployed_network"
            ),
            "offset_backend": args.offset_backend,
            "prepared": prepared,
            "offset_cpu_scope": (
                "coordinator_only; worker_CPU_reported_separately"
                if args.offset_backend == "loopback"
                else "same_process_including_both_worker_stages"
            ),
            "total_wire_bytes": None,
            "full_response_compute_cap_checked": False,
            "online_cpu_comparator_measured": args.offset_backend == "loopback",
            "unmeasured": [
                "HTTP/TLS framing and headers, health/startup, and failed transport attempts",
                "checkpoint transfer and client peak memory",
                "worker CPU before session admission or after session completion",
            ],
            "client_only_topology_digest": client_only_reference_graph().digest(),
            "two_online_topology_digest": two_online_reference_graph().digest(),
            "client_only_checkpoint_artifact_bytes": checkpoint_artifact_bytes,
            "client_only_quantized_weight_bytes": quantized_weight_bytes,
            "client_only_compiled_bundle_bytes": len(bundle_payload),
            "client_only_cold_checkpoint_transfer_bytes": None,
            "client_only_peak_memory_bytes": None,
            "model_type": args.model_type,
            "cohort": args.cohort,
            "plan_digest": compiled.model_plan_digest,
            "compiled_digest": compiled.digest,
            "model_body_fingerprint": bundle.manifest["metadata"]["body_fingerprint"],
            "python": platform.python_version(),
            "input_token_count": len(input_ids),
            "generated_token_count": 2,
            "repeats": args.repeats,
            "all_selected_tokens_match": all(s["same_selected_tokens"] for s in samples),
            "median_client_only_online_cpu_ms": statistics.median(
                s["client_only_online_cpu_ms"] for s in samples
            ),
            "median_offset_coordinator_cpu_ms": statistics.median(
                s["offset_online_cpu_ms"] for s in samples
            ),
            "median_offset_aggregate_online_cpu_seconds": (
                statistics.median(s["offset_aggregate_online_cpu_seconds"] for s in samples)
                if args.offset_backend == "loopback" else None
            ),
            "worst_logit_difference": max(s["worst_logit_difference"] for s in samples),
            "samples": samples,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
