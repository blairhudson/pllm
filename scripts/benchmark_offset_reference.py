"""Measure matched client-only and two-worker offset baselines on tiny W8A8.

This is not a BenchmarkResult: workers share one process and the report excludes
deployment traffic, operator independence, setup and full-response compute.
"""

from __future__ import annotations

import argparse
import asyncio
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
from pllm.runtime.offset_reference import TwoOnlineOffsetReference
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

_INPUT_IDS = (0, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("qwen2", "qwen3"), default="qwen2")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")

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
        first = MaskedTransformerEngine(threads=1)
        second = MaskedTransformerEngine(threads=1)
        asyncio.run(first.load(manifest))
        asyncio.run(second.load(manifest))
        bundle_payload = first.client_bundle(model_id)
        bundle = ClientBundle.unpack(bundle_payload)
        quantized_weight_bytes = sum(
            stage.weight.values.nbytes + stage.weight.scales.nbytes
            + (0 if stage.bias is None else stage.bias.nbytes)
            for stage in first.models[model_id].stages.values()
        )
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        plan = pllm.lower_model(
            config, batch=1, max_input_tokens=4, max_new_tokens=2,
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

        samples: list[dict[str, Any]] = []
        for index in range(args.repeats):
            outcomes: dict[str, tuple[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = {}
            sample: dict[str, Any] = {}
            for method in (
                ("client_only", "offset") if index % 2 == 0 else ("offset", "client_only")
            ):
                offset: TwoOnlineOffsetReference | None = None
                if method == "client_only":
                    client_stage_work = 0
                    remote = client_only
                else:
                    offset = TwoOnlineOffsetReference(
                        compiled, first, second, model_id=model_id,
                        exchange_a=exchange(first), exchange_b=exchange(second),
                    )
                    remote = offset
                wall_start = time.perf_counter_ns()
                cpu_start = time.process_time_ns()
                session = compiled.session(remote)
                session.prefill_ids(_INPUT_IDS)
                prefill = session.logits
                selected = session.select_next()
                session.decode_selected()
                decode = session.logits
                following = session.select_next()
                sample[f"{method}_online_wall_ms"] = (
                    time.perf_counter_ns() - wall_start
                ) / 1_000_000
                sample[f"{method}_online_cpu_ms"] = (
                    time.process_time_ns() - cpu_start
                ) / 1_000_000
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
            sample["same_selected_tokens"] = (
                outcomes["client_only"][0] == outcomes["offset"][0]
            )
            sample["worst_logit_difference"] = max(
                float(np.max(np.abs(reference - candidate)))
                for reference, candidate in zip(
                    outcomes["client_only"][1], outcomes["offset"][1], strict=True,
                )
            )
            samples.append(sample)

        print(json.dumps({
            "schema": "pllm.topology_reference_benchmark.v1",
            "scope": "client_only_and_in_process_offset; not_deployed_network",
            "client_only_topology_digest": client_only_reference_graph().digest(),
            "two_online_topology_digest": two_online_reference_graph().digest(),
            "client_only_checkpoint_artifact_bytes": checkpoint_artifact_bytes,
            "client_only_quantized_weight_bytes": quantized_weight_bytes,
            "client_only_compiled_bundle_bytes": len(bundle_payload),
            "client_only_cold_checkpoint_transfer_bytes": None,
            "client_only_peak_memory_bytes": None,
            "model_type": args.model_type,
            "plan_digest": compiled.model_plan_digest,
            "compiled_digest": compiled.digest,
            "python": platform.python_version(),
            "input_token_count": len(_INPUT_IDS),
            "generated_token_count": 2,
            "repeats": args.repeats,
            "all_selected_tokens_match": all(s["same_selected_tokens"] for s in samples),
            "median_client_only_online_cpu_ms": statistics.median(
                s["client_only_online_cpu_ms"] for s in samples
            ),
            "median_offset_online_cpu_ms": statistics.median(
                s["offset_online_cpu_ms"] for s in samples
            ),
            "worst_logit_difference": max(s["worst_logit_difference"] for s in samples),
            "samples": samples,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
