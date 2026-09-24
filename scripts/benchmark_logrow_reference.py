"""Measure a bounded, in-process LogRow research run against its clear baseline.

This is deliberately not a PLLM BenchmarkResult: no provider transport,
cryptographic assurance, or Experiment-selectable LogRow component exists.
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

import numpy as np

import pllm
from pllm.nonlinear import (
    create_logrow_scaled_silu_q7_reference,
    fit_compact_silu_q7_reference,
)
from pllm.protocols import prepare_logrow_q7_session_reference
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

INPUT_IDS = [0, 2]
MAX_NEW = 2
INTERMEDIATE = 64
LAYERS = 2
MATERIAL_BYTES = (len(INPUT_IDS) + MAX_NEW - 1) * INTERMEDIATE * LAYERS * 2144


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model-type", choices=("qwen2", "qwen3"), default="qwen2")
    parser.add_argument("--numeric", choices=("compact", "scaled"), default="compact")
    parser.add_argument("--public-range", type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.repeats <= 20:
        parser.error("--repeats must be between 1 and 20")
    if arguments.numeric == "scaled" and not 1 <= arguments.public_range <= 16:
        parser.error("--public-range must be an integer from 1 through 16")
    gate_weight_scale = 0.008 if arguments.numeric == "compact" else 0.08

    with tempfile.TemporaryDirectory(prefix="pllm-logrow-reference-") as root:
        checkpoint = create_tiny_llama_checkpoint(
            Path(root) / "model",
            num_hidden_layers=LAYERS,
            model_type=arguments.model_type,
            with_qkv_bias=arguments.model_type == "qwen2",
            qk_norm=arguments.model_type == "qwen3",
            gate_weight_scale=gate_weight_scale,
        )
        model_id = "tiny-logrow-reference"
        manifest = load_hf_directory(checkpoint, model_id=model_id)
        engine = MaskedTransformerEngine(threads=1)
        asyncio.run(engine.load(manifest))
        bundle = ClientBundle.unpack(engine.client_bundle(model_id))
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        plan = pllm.lower_model(
            config, batch=1, max_input_tokens=len(INPUT_IDS), max_new_tokens=MAX_NEW
        )
        compiled = compile_runtime_model(plan, bundle)
        profile = (
            fit_compact_silu_q7_reference(bytes(257 * 4))
            if arguments.numeric == "compact"
            else create_logrow_scaled_silu_q7_reference(arguments.public_range)
        )
        timing: dict[str, list[float]] = {
            "baseline_setup_wall_ms": [],
            "baseline_online_wall_ms": [],
            "logrow_offline_wall_ms": [],
            "logrow_online_wall_ms": [],
            "baseline_online_cpu_ms": [],
            "logrow_offline_cpu_ms": [],
            "logrow_online_cpu_ms": [],
        }
        matched: list[bool] = []
        prefill_deltas: list[float] = []
        stage_payload: dict[str, list[int]] = {"baseline": [], "logrow": []}
        issuance_digests: list[str] = []
        estimate_digests: list[str] = []

        for index in range(arguments.repeats):
            results = {}
            # Alternate order to avoid treating an always-first method as warm.
            for method in ("baseline", "logrow") if index % 2 == 0 else ("logrow", "baseline"):
                bytes_sent = 0

                def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
                    nonlocal bytes_sent
                    stage = bundle.stages[stage_id]
                    kernel = engine.models[model_id].stages[stage_id]
                    value = np.asarray(activation, dtype=np.float32)
                    output = value @ kernel.weight.dequantize().T
                    if stage.bias is not None:
                        output = output + stage.bias
                    output = np.ascontiguousarray(output, dtype=np.float32)
                    bytes_sent += value.nbytes + output.nbytes
                    return output

                setup_start = time.perf_counter_ns()
                setup_cpu = time.process_time_ns()
                if method == "baseline":
                    session = compiled.session(remote)
                else:
                    material = prepare_logrow_q7_session_reference(
                        plan,
                        profile,
                        max_elements=len(INPUT_IDS) * INTERMEDIATE,
                        max_evaluator_material_bytes=len(INPUT_IDS) * INTERMEDIATE * 2144,
                        max_decode_steps=MAX_NEW - 1,
                        max_session_evaluator_material_bytes=MATERIAL_BYTES,
                    )
                    issuance_digests.append(material.issuance_digest)
                    estimate_digests.append(json.loads(material.estimate())["estimate_digest"])
                    session = compiled.session(remote, research_logrow_material=material)
                setup_wall_ms = (time.perf_counter_ns() - setup_start) / 1_000_000
                setup_cpu_ms = (time.process_time_ns() - setup_cpu) / 1_000_000
                timing[f"{method}_{'setup' if method == 'baseline' else 'offline'}_wall_ms"].append(
                    setup_wall_ms
                )
                if method == "logrow":
                    timing["logrow_offline_cpu_ms"].append(setup_cpu_ms)

                start = time.perf_counter_ns()
                cpu = time.process_time_ns()
                prefill = session.prefill_ids(INPUT_IDS)
                first = session.select_next()
                session.decode_selected()
                second = session.select_next()
                timing[f"{method}_online_wall_ms"].append(
                    (time.perf_counter_ns() - start) / 1_000_000
                )
                timing[f"{method}_online_cpu_ms"].append((time.process_time_ns() - cpu) / 1_000_000)
                stage_payload[method].append(bytes_sent)
                results[method] = ((first, second), prefill.copy())
            matched.append(results["baseline"][0] == results["logrow"][0])
            prefill_deltas.append(
                float(np.max(np.abs(results["baseline"][1] - results["logrow"][1])))
            )

        record = {
            "schema_version": "pllm.logrow_local_reference_measurement.v1",
            "execution_scope": "in_process_synthetic_research_only",
            "paper": "https://pllm.run/research/papers/logrow/",
            "model_family": arguments.model_type,
            "checkpoint_body_fingerprint": compiled.bundle_fingerprint,
            "plan_digest": plan.digest,
            "baseline_binding_digest": compiled.digest,
            "method": "pllm/logrow-q7-local-reference/v1",
            "profile_digest": profile.digest.hex(),
            "numeric_profile": arguments.numeric,
            "public_range": arguments.public_range if arguments.numeric == "scaled" else 1,
            "gate_weight_scale": gate_weight_scale,
            "input_tokens": len(INPUT_IDS),
            "output_tokens": MAX_NEW,
            "repeats": arguments.repeats,
            "token_sequences_match_each_pair": all(matched),
            "prefill_max_absolute_logit_delta": max(prefill_deltas),
            "evaluator_body_bytes_per_response": MATERIAL_BYTES,
            "unique_issuance_per_response": len(set(issuance_digests)) == arguments.repeats,
            "session_estimate_digest": estimate_digests[0],
            "session_estimate_matches_all_repeats": len(set(estimate_digests)) == 1,
            "inprocess_stage_tensor_bytes_median": {
                method: statistics.median(values) for method, values in stage_payload.items()
            },
            "wall_and_cpu_ms_median": {
                field: statistics.median(values) for field, values in timing.items()
            },
            "network_bytes_measured": False,
            "peak_memory_measured": False,
            "disk_bytes_measured": False,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
        }
        print(json.dumps(record, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
