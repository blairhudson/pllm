"""Screen five communication ideas without admitting an inference component.

Pinned config/schedule and synthetic checkpoint-sized tables only. Measures
party-local sum-of-squares and compact lookup, conservative exact-fusion
opportunities, same-client CKKS SIMD, and supported serialization. No real
checkpoint nonlinear quality, protected decoder, full wire or CPU cap claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.interaction_reduction_reference import (
    fresh_shares,
    issue_broadcast,
    issue_lookup,
    issue_sum_squares,
)

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_CONTROL = _ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json"
_HUB_CACHE = os.environ.get("HF_HUB_CACHE") or str(
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
)


def fusion_audit(plan: Any, composition: Any) -> dict[str, Any]:
    """Only direct sequential linear edges are candidates; no nonlinear bypass.

    Even direct edges are blocked for exact W8A8 by intermediate activation
    quantization/rounding. Fan-out grouping is already implemented, not new gain.
    """
    schedule = plan.runtime_schedule(composition)
    if not schedule.complete or schedule.protected_execution:
        raise ValueError("fusion audit requires a complete baseline semantic schedule")
    result = {}
    for phase in ("prefill", "decode"):
        operations = plan.to_dict()[phase]["operations"]
        indexed = {row["id"]: row for row in operations}
        candidates = []
        blockers: Counter[str] = Counter()
        for row in operations:
            if row["operator"] not in ("linear", "output_head"):
                continue
            source = indexed[row["inputs"][0]]
            # Reshape has no numeric effect; allow reach-through for discovery
            # only. Layout/permutation and rounding still require a proof.
            while source["operator"] == "reshape":
                source = indexed[source["inputs"][0]]
            if source["operator"] == "linear":
                candidates.append([source["id"], row["id"]])
            else:
                blockers[source["operator"]] += 1
        steps = schedule.to_dict()[phase]["steps"]
        groups = [
            row
            for row in steps
            if row.get("executor") == "remote_stage"
            and len(row.get("operation_ids", [])) > 1
            and set(row.get("operators", [])) == {"linear"}
        ]
        result[phase] = {
            "body_norms_per_executed_row": sum(
                row["operator"] == "rms_norm" and row.get("layer") is not None for row in operations
            ),
            "serial_linear_candidate_edges": candidates,
            "source_operator_barriers": dict(sorted(blockers.items())),
            "existing_parallel_linear_groups": len(groups),
            "existing_duplicate_input_sends_avoided": sum(
                len(row["operation_ids"]) - 1 for row in groups
            ),
            "new_exact_w8a8_fusions": 0,
            "additional_remote_messages_eliminated": 0,
            "rounding_barrier": "intermediate dynamic activation quantization and float32 edges cannot be removed algebraically",
        }
    return {
        "plan_digest": plan.digest,
        "schedule_digest": schedule.digest,
        "scope": "conservative serial-linear discovery, not a graph optimizer or exhaustive algebra search",
        "phases": result,
    }


def probe_reduction(width: int) -> dict[str, Any]:
    values = np.arange(width, dtype=np.int64) % 255 - 127
    shares = fresh_shares(values.astype(np.uint32))
    started = time.process_time()
    materials = issue_sum_squares("synthetic", "norm-statistic", width)
    issuance_cpu = time.process_time() - started
    started = time.process_time()
    frames = [material.open(share) for material, share in zip(materials, shares, strict=True)]
    outputs = [materials[0].finish(frames[1]), materials[1].finish(frames[0])]
    expected = int(np.dot(values, values))
    if sum(outputs) % (1 << 32) != expected:
        raise RuntimeError("specialized sum-of-squares misses independent integer oracle")
    online_cpu = time.process_time() - started
    broadcast = issue_broadcast("synthetic", "normalize-product", width)
    scalar = np.asarray([23], dtype=np.uint32)
    scalar_shares = fresh_shares(scalar)
    broadcasts = [
        material.open(vector, scale)
        for material, vector, scale in zip(broadcast, shares, scalar_shares, strict=True)
    ]
    products = [broadcast[0].finish(broadcasts[1]), broadcast[1].finish(broadcasts[0])]
    if not np.array_equal(np.add(*products, dtype=np.uint32), values.astype(np.uint32) * scalar):
        raise RuntimeError("broadcast scalar multiply misses independent integer oracle")
    return {
        "width": width,
        "ring_bits": 32,
        "synthetic_signed_input_bound": 127,
        "exact_modular_statistic_parity": True,
        "sum_of_squares_body_bytes_online_both_directions": sum(len(f.body) for f in frames),
        "sum_of_squares_dealer_body_bytes_both_parties": sum(m.body_bytes for m in materials),
        "generic_elementwise_beaver_square_online_bytes": 4 * width * 4,
        "generic_elementwise_beaver_square_dealer_bytes": 6 * width * 4,
        "synthetic_issuance_cpu_seconds": issuance_cpu,
        "synthetic_online_local_cpu_seconds": online_cpu,
        "full_rmsnorm_known_online_floor_including_broadcast_scalar_multiply_bytes": (4 * width + 2)
        * 4,
        "full_rmsnorm_known_dealer_floor_including_broadcast_scalar_multiply_bytes": (6 * width + 4)
        * 4,
        "generic_square_plus_normalize_online_bytes": 8 * width * 4,
        "generic_square_plus_normalize_dealer_bytes": 12 * width * 4,
        "broadcast_multiply_exact_modular_parity": True,
        "broadcast_multiply_online_bytes": sum(len(frame.body) for frame in broadcasts),
        "broadcast_multiply_dealer_bytes": sum(material.body_bytes for material in broadcast),
        "broadcast_inverse_is_synthetic_not_rmsnorm_inverse": True,
        "missing_full_rmsnorm_work": [
            "private inverse square root and division by public width",
            "exact rescaling/output quantization and per-edge rounding",
            "weights/scales, authentication, distributed dealer and full wire",
        ],
        "numeric_order_counterexample": {
            "input": [1, 1],
            "divisor": 2,
            "sum_of_per_element_ties_even_rounded_squares": 0,
            "ties_even_rounded_sum_of_squares": 1,
        },
        "whole_rmsnorm_executable": False,
        "checkpoint_numeric_fidelity": False,
    }


def probe_lookup(vocabulary: int, width: int) -> dict[str, Any]:
    # Complete random public table at pinned dimensions, not a compressible
    # structured shortcut, and not a real checkpoint or private user token.
    table = np.random.default_rng(20260930).integers(
        -128, 128, size=(vocabulary, width), dtype=np.int8
    )
    table.setflags(write=False)
    fingerprint = hashlib.sha256(memoryview(table)).digest()
    token = vocabulary // 2 + 17
    started = time.process_time()
    keys = issue_lookup("synthetic", "lookup", token, vocabulary, fingerprint)
    generation_cpu = time.process_time() - started
    bodies = [len(key.serialize()) for key in keys]
    outputs, samples = [], []
    for key in keys:
        started = time.process_time()
        output = key.evaluate(table, session="synthetic", query="lookup", fingerprint=fingerprint)
        samples.append(time.process_time() - started)
        outputs.append(output)
    reconstructed = np.add(*outputs, dtype=np.uint32)
    if not np.array_equal(reconstructed, table[token].astype(np.uint32)):
        raise RuntimeError("private lookup misses exact signed-i8 table row")
    return {
        "vocabulary": vocabulary,
        "width": width,
        "table_kind": "public synthetic signed-i8, no per-token float scales",
        "table_sha256": fingerprint.hex(),
        "table_bytes_each_worker": table.nbytes,
        "query_body_bytes_both_workers": sum(bodies),
        "query_body_bytes_per_worker": bodies,
        "direct_client_embedding_u32_shares_bytes_both_workers": 2 * width * 4,
        "dense_u32_selector_shares_bytes_both_workers": 2 * vocabulary * 4,
        "result_share_bytes_both_workers_if_returned_to_client": 2 * width * 4,
        "results_can_stay_worker_local_without_this_return": True,
        "modular_table_macs_per_worker_per_token": vocabulary * width,
        "table_read_bytes_per_worker_per_token_at_least": table.nbytes,
        "selector_storage_bytes_each_worker": vocabulary * 4,
        "maximum_conversion_window_bytes": min(vocabulary, 1024) * width * 4,
        "key_generation_cpu_seconds": generation_cpu,
        "key_expansion_hash_and_table_scan_cpu_seconds_each_worker": samples,
        "exact_quantized_code_lookup_parity": True,
        "checkpoint_embedding_scale_and_numeric_conversion_validated": False,
        "he_embedding_conversion_available": False,
        "security_scope": "bounded local DPF research construction, no security review or authenticated independent-worker transport",
    }


def probe_he() -> dict[str, Any]:
    import tenseal as ts

    client = ts.context(
        ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60], n_threads=1
    )
    client.global_scale = 2**40
    public = client.serialize(save_secret_key=False, save_galois_keys=False, save_relin_keys=True)
    provider = ts.context_from(public, n_threads=1)
    if provider.has_secret_key():
        raise RuntimeError("CKKS evaluator contains client secret")
    batches = []
    width = 896
    for count in (1, 4, 16):
        # Separate ciphertext chunks beyond capacity; never claim 16 fit once.
        slots, chunk_capacity = 4096, 4096 // width
        values = [(0.25 + index / 32) for index in range(count)]
        started = time.process_time()
        requests = []
        lengths = []
        for first in range(0, count, chunk_capacity):
            rows = values[first : first + chunk_capacity]
            requests.append(ts.ckks_vector(client, np.repeat(rows, width).tolist()).serialize())
            lengths.append(len(rows) * width)
        encryption_cpu = time.process_time() - started
        started = time.process_time()
        responses = []
        for request in requests:
            encrypted = ts.ckks_vector_from(provider, request)
            responses.append(encrypted.square().serialize())
        evaluation_cpu = time.process_time() - started
        recovered = np.concatenate(
            [np.asarray(ts.ckks_vector_from(client, response).decrypt()) for response in responses]
        )
        expected = np.repeat(np.square(values), width)
        error = float(np.max(np.abs(recovered - expected)))
        if error > 1e-3:
            raise RuntimeError("same-client SIMD does not match independent-row square oracle")
        batches.append(
            {
                "independent_same_client_sequences": count,
                "hidden_width": width,
                "slots_per_ciphertext": slots,
                "ciphertexts_each_direction": len(requests),
                "values_per_ciphertext": lengths,
                "upload_body_bytes": sum(map(len, requests)),
                "download_body_bytes": sum(map(len, responses)),
                "both_direction_body_bytes_per_sequence": sum(map(len, requests + responses))
                / count,
                "encryption_cpu_seconds": encryption_cpu,
                "provider_square_cpu_seconds": evaluation_cpu,
                "provider_square_cpu_seconds_per_sequence": evaluation_cpu / count,
                "max_absolute_error": error,
            }
        )
    del provider
    gc.collect()

    serialization = []
    failure = None
    for mode in (ts.ENCRYPTION_TYPE.ASYMMETRIC, ts.ENCRYPTION_TYPE.SYMMETRIC):
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
            encryption_type=mode,
            n_threads=1,
        )
        context.global_scale = 2**40
        samples = []
        public_bytes = context.serialize(save_secret_key=False, save_galois_keys=False)
        evaluator = ts.context_from(public_bytes, n_threads=1)
        if evaluator.has_secret_key():
            raise RuntimeError("serialization evaluator contains a secret")
        for _ in range(3):
            request = ts.ckks_vector(context, [0.75] * width).serialize()
            response = ts.ckks_vector_from(evaluator, request).square().serialize()
            decrypted = np.asarray(ts.ckks_vector_from(context, response).decrypt())
            if np.max(np.abs(decrypted - 0.75**2)) > 1e-3:
                raise RuntimeError("serialization arithmetic parity failed")
            samples.append(
                {"request_bytes": len(request), "evaluated_response_bytes": len(response)}
            )
        if mode == ts.ENCRYPTION_TYPE.SYMMETRIC:
            try:
                context.encryptor().data.encrypt_zero_symmetric()
            except TypeError as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)}
            else:
                raise RuntimeError(
                    "seeded-return capability changed; explicitly inspect before claiming support"
                )
        serialization.append(
            {
                "encryption_mode": mode.name,
                "samples": samples,
                "provider_has_secret_key": False,
            }
        )
        del context, evaluator
        gc.collect()
    return {
        "tenseal_version": ts.__version__,
        "poly_modulus_degree": 8192,
        "coeff_modulus_bits": [60, 40, 40, 60],
        "scale_bits": 40,
        "provider_has_secret_key": False,
        "batching": {
            "samples": batches,
            "scope": "one synthetic ciphertext square only, no attention/rotations/KV or decoder",
            "mixed_client_keys_supported": False,
            "single_sequence_latency_improvement_established": False,
        },
        "seeded_serialization": {
            "samples": serialization,
            "ckks_vector_serialize_signature": str(inspect.signature(ts.CKKSVector.serialize)),
            "seeded_low_level_return_failure": failure,
            "seeded_ciphertext_transport_supported_by_tested_python_api": False,
            "symmetric_encryption_is_not_seeded_serialization": True,
            "secret_key_never_sent_to_provider": True,
        },
    }


def run(*, sample_he: bool = True, sample_lookup: bool = True) -> dict[str, Any]:
    config_path = Path(
        hf_hub_download(
            _MODEL, "config.json", revision=_REVISION, local_files_only=True, cache_dir=_HUB_CACHE
        )
    )
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    control = json.loads(_CONTROL.read_text())
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    reports = {}
    for tokens in (8, 32):
        cohort = next(row for row in control["cohorts"] if row["output_tokens"] == tokens)
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=tokens)
        audit = fusion_audit(plan, composition)
        if (
            plan.digest != cohort["official_plan_digest"]
            or audit["schedule_digest"] != cohort["official_schedule_digest"]
        ):
            raise ValueError("interaction screen is not bound to prepared control")
        if (
            plan.runtime_schedule(composition).composition_digest
            != control["source"]["pipeline_digest"]
        ):
            raise ValueError("interaction screen composition differs from prepared control")
        reports[str(tokens)] = {
            "input_tokens": 39,
            "output_tokens": tokens,
            "executed_rows": 39 + tokens - 1,
            "online_tenfold_budget_bytes": cohort["prepared_control"][
                "tenfold_online_budget_bytes"
            ],
            "all_link_tenfold_budget_bytes": cohort["prepared_control"][
                "tenfold_covered_budget_bytes"
            ],
            "fusion_audit": audit,
        }
    width, vocabulary = config["hidden_size"], config["vocab_size"]
    reduction = probe_reduction(width)
    lookup = probe_lookup(vocabulary, width) if sample_lookup else None
    for report in reports.values():
        norm_count = (
            report["fusion_audit"]["phases"]["prefill"]["body_norms_per_executed_row"]
            * report["executed_rows"]
        )
        report["rmsnorm_projection"] = {
            "body_norm_invocations": norm_count,
            "known_online_floor_bytes": norm_count
            * reduction[
                "full_rmsnorm_known_online_floor_including_broadcast_scalar_multiply_bytes"
            ],
            "known_dealer_floor_bytes": norm_count
            * reduction[
                "full_rmsnorm_known_dealer_floor_including_broadcast_scalar_multiply_bytes"
            ],
            "inverse_sqrt_rounding_attention_mlp_unpriced": True,
        }
        if lookup:
            rows = report["executed_rows"]
            report["lookup_projection"] = {
                "all_token_query_bytes_both_workers": rows
                * lookup["query_body_bytes_both_workers"],
                "direct_embedding_u32_shares_bytes_both_workers": rows * 2 * width * 4,
                "saved_boundary_upload_body_bytes": rows
                * (2 * width * 4 - lookup["query_body_bytes_both_workers"]),
                "additional_lookup_table_macs_both_workers": rows * 2 * width * vocabulary,
                "amortized_checkpoint_table_distribution_not_included": True,
            }
    return {
        "schema": "pllm.interaction_reduction_screen.v1",
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "ckks_context_threads": 1,
            "cpu_samples": "local process_time; synthetic primitives only, not complete-response CPU",
        },
        "source": {
            "checkpoint": f"{_MODEL}@{_REVISION}",
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "control_body_fingerprint": control["source"]["body_fingerprint"],
            "pipeline_digest": control["source"]["pipeline_digest"],
        },
        "cohorts": reports,
        "secure_reduction": reduction,
        "private_lookup": lookup,
        "he": probe_he() if sample_he else None,
        "whole_decoder_executable": False,
        "checkpoint_quality_validated": False,
        "full_wire_measured": False,
        "aggregate_compute_cap_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-he", action="store_true")
    parser.add_argument("--no-lookup", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(sample_he=not args.no_he, sample_lookup=not args.no_lookup),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
