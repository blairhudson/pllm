#!/usr/bin/env python3
"""Ten first-principles screens: exact algebra, pinned public traces, cost gates.

uv run --no-sync python scripts/probe_network_first_principles.py --output PATH
Use --algebra-only for bounded checkpoint-free counterexamples. No live protocol,
key issuance, private prompt, external implementation, or SDK option is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

from network_first_principles import (
    algebra_report,
    rational_silu,
    rational_silu_tails,
    silu32,
    tropical_silu,
    wallace_counts,
)

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PROMPTS = (
    "Compare a lighthouse with a satellite as a guide for a lost sailor.",
    "Why does a ceramic cup stay warm after the tea has been poured out?",
    "Give a short explanation of how a checksum differs from a secret password.",
    "A garden has three rows with seven plants each. Explain the total.",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def gf2_rank(matrix):
    """Exact minor rank: a LOWER bound on integer and 2-adic row independence."""
    packed = np.packbits(np.asarray(matrix, dtype=np.int64) & 1, axis=1)
    pivots = {}
    for row in packed:
        word = int.from_bytes(row.tobytes(), "little")
        while word:
            pivot = word.bit_length()
            if pivot not in pivots:
                pivots[pivot] = word
                break
            word ^= pivots[pivot]
    return len(pivots)


def active_kv_digest(runtime):
    digest = hashlib.sha256()
    for layer, cache in enumerate(runtime.caches):
        digest.update(str((layer, cache.length)).encode())
        for array in (cache.key, cache.value):
            if array is None:
                raise ValueError("missing full-KV public fixture state")
            active = np.ascontiguousarray(array[: cache.length])
            digest.update(str((active.shape, str(active.dtype))).encode())
            digest.update(active.tobytes())
    return digest.hexdigest()


def prior_budget():
    path = ROOT / "docs/evidence/network-algebra-sdk-qwen25-2026-10-06.json"
    payload = path.read_bytes()
    control = next(
        row["report"]
        for row in json.loads(payload)["candidates"]
        if row["name"] == "algebra-stage-packed"
    )
    summary = control["summary"]
    total = summary["accounted_setup_through_first_response_body_bytes"]
    return {
        "artifact": str(path.relative_to(ROOT)),
        "sha256": sha(payload),
        "input_tokens": 39,
        "generated_outputs": 8,
        "covered_setup_inclusive_bytes": total,
        "tenfold_body_target_bytes": total / 10,
        "hundredfold_body_target_bytes": total / 100,
        "historical_cold_cpu_seconds": control["process_cpu_accounting"][
            "aggregate_cold_first_response_cpu_seconds"
        ],
        "scope": "historical budget only; new screens are not matched transport/CPU measurements",
    }


def checkpoint_screens(result, guard):
    from decoder_probe_support import PROMPT, DecoderFixture
    from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

    guard()
    fixture = DecoderFixture(inputs=64, outputs=8)
    result["source"] = fixture.lock()
    records = {}
    sampled_inputs = {}
    total_silu, tropical_equal, rational_equal, outside = 0, 0, 0, 0
    worst_rational = 0.0

    def remote(stage_id, activation):
        guard()
        stage = fixture.model.stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=8)
        integer = stage.compiled_weight.clear(quantized.values)
        flat = np.asarray(integer).reshape(-1)
        blocks = flat[: len(flat) // 32 * 32].reshape(-1, 32)
        record = records.setdefault(
            stage_id,
            {
                "input_width": stage.spec.in_features,
                "output_width": stage.spec.out_features,
                "executed_rows": 0,
                "integer_outputs": 0,
                "nonzero_outputs": 0,
                "blocks32": 0,
                "blocks_with_at_most_one_nonzero": 0,
                "row_arithmetic_bits": stage.spec.in_features * max(stage.output_residue_bits)
                + 2 * sum(stage.output_residue_bits),
            },
        )
        record["executed_rows"] += len(quantized.values)
        record["integer_outputs"] += integer.size
        record["nonzero_outputs"] += int(np.count_nonzero(integer))
        record["blocks32"] += len(blocks)
        record["blocks_with_at_most_one_nonzero"] += int(
            np.count_nonzero(np.count_nonzero(blocks, axis=1) <= 1)
        )
        rows = sampled_inputs.setdefault(stage_id, [])
        retained = sum(len(v) for v in rows)
        if retained < 64:
            rows.append(quantized.values[: 64 - retained, :2048].copy())
        output = dequantize_matmul(
            integer,
            quantized.scales,
            stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    runtime = fixture.compiled.runtime(remote)
    original_local = runtime._local

    def observe_local(operation, values, *state):
        nonlocal total_silu, tropical_equal, rational_equal, worst_rational, outside
        # Observe the public fixture; execution remains the original implementation.
        answer = original_local(operation, values, *state)
        if operation["operator"] == "silu":
            x = values[operation["inputs"][0]]
            expected = silu32(x)
            actual = answer
            np.testing.assert_array_equal(expected.view(np.uint32), actual.view(np.uint32))
            total_silu += x.size
            tropical_equal += int(
                np.count_nonzero(expected.view(np.uint32) == tropical_silu(x).view(np.uint32))
            )
            bounded = np.isfinite(x) & (np.abs(x) <= 32)
            outside += int(x.size - np.count_nonzero(bounded))
            candidate = rational_silu(x[bounded])
            rational_equal += int(
                np.count_nonzero(candidate.view(np.uint32) == expected[bounded].view(np.uint32))
            )
            if candidate.size:
                worst_rational = max(
                    worst_rational,
                    float(np.max(np.abs(candidate.astype(np.float64) - expected[bounded]))),
                )
        return answer

    runtime._local = observe_local
    ids = fixture.tokens(PROMPT)
    if len(ids) != 39:
        raise ValueError("public control token count changed")
    _, logits, _ = runtime.prepare_ids(ids)
    for _ in range(7):
        logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
    assert total_silu == 24 * 4864 * 46
    for key, row in records.items():
        sample = np.concatenate(sampled_inputs.pop(key))
        row["rank_minor_rows"], row["rank_minor_columns"] = sample.shape
        row["independent_source_rows_lower_bound"] = gf2_rank(sample)
        row["possible_dependent_rows_upper_bound"] = (
            len(sample) - row["independent_source_rows_lower_bound"]
        )
        row["full_input_output_basis_storage_bytes_at_lower_rank"] = row[
            "independent_source_rows_lower_bound"
        ] * (row["input_width"] + row["output_width"] * 4)
    aggregate = {
        key: sum(row[key] for row in records.values())
        for key in (
            "executed_rows",
            "integer_outputs",
            "nonzero_outputs",
            "blocks32",
            "blocks_with_at_most_one_nonzero",
            "independent_source_rows_lower_bound",
            "possible_dependent_rows_upper_bound",
            "full_input_output_basis_storage_bytes_at_lower_rank",
        )
    }
    arithmetic_bits = sum(
        row["row_arithmetic_bits"] * row["executed_rows"] for row in records.values()
    )
    removable_bits = sum(
        row["row_arithmetic_bits"] * row["possible_dependent_rows_upper_bound"]
        for row in records.values()
    )
    aggregate["optimistic_span_arithmetic_reduction_fraction_upper_bound"] = (
        removable_bits / arithmetic_bits
    )
    result["public_trace"] = {
        "input_tokens": len(ids),
        "generated_outputs": 8,
        "public_token_cohort_sha256": sha(np.asarray(ids, dtype="<i8").tobytes()),
        "stages": records,
        "aggregate": aggregate,
        "silu_elements": total_silu,
        "tropical_bit_exact_elements": tropical_equal,
        "rational24_bit_exact_elements": rational_equal,
        "rational24_outside_public_domain": outside,
        "rational24_maximum_in_domain_absolute_error": worst_rational,
        "scope": "local clear-kernel public trajectory, counts only; source values not archived",
    }
    sampled_inputs.clear()
    del runtime

    # Exact public bitheap geometry, priced over the response's actual stage rows.
    total_ands, total_zech = 0, 0
    bitheaps = []
    for index, (key, stage) in enumerate(fixture.body.items()):
        guard()
        w = stage.weight.values
        width = stage.seeded_profile.wire_bits
        count = wallace_counts(w, width)
        per_row = int(count["and_gates_per_row"].sum())
        additions = int(np.maximum(np.count_nonzero(w, axis=1) - 1, 0).sum())
        executed = records[key]["executed_rows"]
        total_ands += per_row * executed
        total_zech += additions * executed
        bitheaps.append(
            {
                "stage": key,
                "ring_bits": width,
                "and_gates_per_activation_row": per_row,
                "zech_additions_per_activation_row": additions,
                "executed_rows": executed,
            }
        )
        if (index + 1) % 24 == 0:
            print(f"Priced {index + 1}/{len(fixture.body)} public stages", flush=True)
    target = result["historical_budget"]["hundredfold_body_target_bytes"]
    cpu = result["historical_budget"]["historical_cold_cpu_seconds"]
    result["encrypted_carry_save"] = {
        "stages": bitheaps,
        "body_linear_and_gates": total_ands,
        "all_other_work_free_cpu_nanoseconds_per_and_budget": cpu * 1e9 / total_ands,
        "two_16byte_half_gate_ciphertexts_bytes": 32 * total_ands,
        "fhe_backend_cpu_seconds": None,
        "fhe_key_distribution_bytes": None,
        "scope": "specified unsimplified bitheap, not minimum circuit size; FHE is a conditional CPU budget only",
    }
    result["zech_cost"] = {
        "body_linear_private_addition_lookups": total_zech,
        "hundredfold_all_link_bytes_per_lookup_if_everything_else_free": target / total_zech,
        "scope": "fully log-domain dot products; omits zero tests, conversion, nonlinear work and transport",
    }

    # Four fresh public cases, fixed range/depth, teacher-forced against original W8A8.
    quality = []
    for text in PUBLIC_PROMPTS:
        guard()
        tokens = fixture.tokens(text)
        reference = fixture.compiled.runtime(fixture.remote)
        _, original, _ = reference.prepare_ids(tokens)
        scores, chosen, states = (
            [original.copy()],
            [int(np.argmax(original))],
            [active_kv_digest(reference)],
        )
        for _ in range(2):
            original = reference.forward_ids([chosen[-1]])[-1]
            scores.append(original.copy())
            chosen.append(int(np.argmax(original)))
            states.append(active_kv_digest(reference))
        del reference
        record = {
            "public_cohort_sha256": sha(np.asarray(tokens, dtype="<i8").tobytes()),
            "input_tokens": len(tokens),
            "selected_positions": 3,
            "candidates": [],
        }
        for terms, tails in ((16, False), (24, False), (24, True)):
            started = time.process_time()
            evaluator = rational_silu_tails if tails else rational_silu
            trial = fixture.compiled.runtime(
                fixture.remote, nonlinear_evaluator=lambda _i, x, n=terms, fn=evaluator: fn(x, n)
            )
            candidate = {
                "terms": terms,
                "core_public_range": [-32, 32],
                "complete": False,
                "outside_core": "explicit_identity_signed_zero_tails" if tails else "reject",
                "execution_identity": sha(
                    json.dumps(
                        {
                            "binding": fixture.compiled.digest,
                            "probe_sources": result["code_sha256"],
                            "terms": terms,
                            "range": [-32, 32],
                            "tails": tails,
                        },
                        sort_keys=True,
                    ).encode()
                ),
            }
            try:
                _, score, _ = trial.prepare_ids(tokens)
                equal_logits, equal_kv, top1, maximum = 0, 0, 0, 0.0
                for step in range(3):
                    if step:
                        score = trial.forward_ids([chosen[step - 1]])[-1]
                    equal_logits += int(
                        np.array_equal(score.view(np.uint32), scores[step].view(np.uint32))
                    )
                    equal_kv += int(active_kv_digest(trial) == states[step])
                    top1 += int(int(np.argmax(score)) == chosen[step])
                    maximum = max(
                        maximum, float(np.max(np.abs(score.astype(np.float64) - scores[step])))
                    )
                candidate.update(
                    complete=True,
                    exact_logit_positions=equal_logits,
                    exact_kv_positions=equal_kv,
                    top1_agreements=top1,
                    worst_absolute_logit_error=maximum,
                )
            except ValueError as error:
                if "outside public rational domain" not in str(error):
                    raise
                candidate["rejection"] = str(error)
            candidate["research_local_cpu_seconds"] = time.process_time() - started
            record["candidates"].append(candidate)
            del trial
        quality.append(record)
    result["rational_numeric_diagnostic"] = {
        "reference": "same pinned W8A8 compiled runtime; not upstream float32 or task quality",
        "cases": quality,
        "protected_execution": False,
    }


def run(output, algebra_only=False):
    if not algebra_only and psutil.virtual_memory().available < 4 << 30:
        raise RuntimeError("checkpoint screen requires 4 GiB physical headroom")
    before_swap = psutil.swap_memory().used

    def guard():
        if (
            psutil.virtual_memory().available < 2 << 30
            or psutil.swap_memory().used > before_swap + (64 << 20)
        ):
            raise RuntimeError("screen stopped on host pressure or swap growth")

    result = {
        "schema": "pllm.network_first_principles.v1",
        "complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "code_sha256": {
            path.name: sha(path.read_bytes())
            for path in (
                Path(__file__),
                Path(__file__).with_name("network_first_principles.py"),
                Path(__file__).with_name("decoder_probe_support.py"),
            )
        },
        "historical_budget": prior_budget(),
        "algebra": algebra_report(24 * 4864 * 46),
        "full_wire_bytes": None,
        "whole_client_peak_rss_bytes": None,
        "sdk_promotion": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not algebra_only:
        checkpoint_screens(result, guard)
    result.update(
        complete=True,
        checkpoint_screen=not algebra_only,
        new_swap_bytes=max(0, psutil.swap_memory().used - before_swap),
        research_process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if platform.system() == "Darwin" else 1024),
    )
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"complete": True, "output": str(output), "new_swap_bytes": result["new_swap_bytes"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algebra-only", action="store_true")
    args = parser.parse_args()
    run(args.output, args.algebra_only)
