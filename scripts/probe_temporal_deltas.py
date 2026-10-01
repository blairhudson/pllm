"""Bounded, cache-only temporal W8A8 screen; no selectable protocol or private trace.

Run dense baseline, then a same-token replay using exact integer temporal updates.
Prefill comparisons are adjacent causal positions, not repeated prompt executions.
Support-dependent encodings below are deliberately non-oblivious oracle estimates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from unittest.mock import patch

import numpy as np

import decoder_probe_support as support
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.preparation_protocol import seeded_ring_profile

COHORTS = {
    "screen": (
        "Explain why neither server can see the prompt.",
        "What causes ocean tides? Give a short explanation.",
        "Write a Python function that adds two integers.",
    ),
    "confirmation": (
        "Describe how a bicycle brake works.",
        "Compare renewable solar power with wind power.",
        "Translate good morning into French and Spanish.",
    ),
}
SDK_REMOTE_ROLES = {"mlp_gate_up", "mlp_down"}
SDK_CLIENT_ATTENTION_ROLES = {"qkv_projection", "attention_output"}
DECODE_STEPS = 4
MAX_INPUTS = 64


def temporal_delta(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Subtract after widening: W8A8 differences occupy [-254, 254], not i8."""
    if previous.dtype != np.int8 or current.dtype != np.int8:
        raise ValueError("temporal inputs must be int8")
    if previous.ndim != 1 or previous.shape != current.shape:
        raise ValueError("temporal inputs must be equal-width vectors")
    if np.any(previous == -128) or np.any(current == -128):
        raise ValueError("symmetric W8A8 domain excludes -128")
    return current.astype(np.int16) - previous.astype(np.int16)


def exact_sparse_update(
    weight: np.ndarray, previous_product: np.ndarray, delta: np.ndarray
) -> np.ndarray:
    """Independent diagnostic reference; plaintext support is NOT a protocol."""
    if weight.dtype != np.int8 or weight.ndim != 2:
        raise ValueError("weight must be an int8 matrix")
    if delta.ndim != 1 or delta.shape != (weight.shape[1],):
        raise ValueError("delta width mismatch")
    if delta.dtype.kind != "i" or np.any(np.abs(delta.astype(np.int64)) > 254):
        raise ValueError("delta must be widened signed W8A8 differences")
    if previous_product.shape != (weight.shape[0],) or previous_product.dtype.kind != "i":
        raise ValueError("previous product must be an integer output vector")
    indices = np.flatnonzero(delta)
    correction = weight[:, indices].astype(np.int64) @ delta[indices].astype(np.int64)
    return previous_product.astype(np.int64) + correction


def encoding_estimates(width: int, changed: int, out: int, wire_bits: int) -> dict:
    """Fixed-alphabet oracle bounds only; no entropy or oblivious-code claim.

    Nine bits cover 508 possible nonzero differences. A combinatorial support
    code assumes the private count is known for free. Fresh masked values need
    ring-width slots, not nine-bit plaintext slots. Outputs stay dense.
    """
    if any(type(v) is not int for v in (width, changed, out, wire_bits)):
        raise ValueError("encoding dimensions must be integers")
    if width < 1 or not 0 <= changed <= width or out < 1 or wire_bits not in (16, 24, 32):
        raise ValueError("invalid encoding dimensions")
    packed = lambda bits: (bits + 7) // 8
    index_bits = (width - 1).bit_length()
    support_bits = (math.comb(width, changed) - 1).bit_length()
    dense_input = packed(width * wire_bits)
    output = packed(out * wire_bits)
    masked_oracle_input = packed(support_bits + changed * wire_bits)
    return {
        "dense_exact_ring_input_bytes": dense_input,
        "dense_online_bytes": dense_input + output,
        "dense_all_link_bytes": dense_input + 2 * output,
        "known_support_9bit_plaintext_bytes": packed(changed * 9),
        "enumerative_support_9bit_plaintext_bytes": packed(support_bits + changed * 9),
        "explicit_index_9bit_plaintext_bytes": packed(changed * (index_bits + 9)),
        "enumerative_support_masked_ring_oracle_input_bytes": masked_oracle_input,
        "masked_ring_oracle_online_bytes": masked_oracle_input + output,
        "masked_ring_oracle_all_link_bytes": masked_oracle_input + 2 * output,
        "public_full_width_padded_index_ring_input_bytes": packed(width * (index_bits + wire_bits)),
    }


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    return dict(
        zip(
            ("min", "p10", "median", "p90", "max"),
            map(float, np.quantile(values, (0, 0.1, 0.5, 0.9, 1))),
        )
    ) | {"count": len(values), "mean": float(np.mean(values))}


class Measurements:
    def __init__(self):
        self.counts = Counter()
        self.changed_fractions = []
        self.raw_float_changed_fractions = []
        self.relative_float_l2 = []
        self.changed_histogram = Counter()

    def add(
        self,
        previous,
        current,
        previous_scale,
        current_scale,
        previous_float,
        current_float,
        out,
        wire_bits,
    ):
        delta = temporal_delta(previous, current)
        changed = int(np.count_nonzero(delta))
        width = delta.size
        self.counts["transitions"] += 1
        self.counts["coordinates"] += width
        self.counts["changed_coordinates"] += changed
        self.counts["identical_quantized_rows"] += changed == 0
        self.counts["activation_scale_bit_changes"] += (
            np.float32(previous_scale).tobytes() != np.float32(current_scale).tobytes()
        )
        self.counts["delta_coordinates_outside_signed_i8"] += int(
            np.count_nonzero(np.abs(delta) > 127)
        )
        self.counts["max_abs_delta"] = max(self.counts["max_abs_delta"], int(np.max(np.abs(delta))))
        self.changed_fractions.append(changed / width)
        self.changed_histogram[changed] += 1
        self.raw_float_changed_fractions.append(float(np.mean(previous_float != current_float)))
        norm = np.linalg.norm(previous_float.astype(np.float64))
        self.relative_float_l2.append(
            float(
                np.linalg.norm(current_float.astype(np.float64) - previous_float)
                / max(float(norm), 1e-30)
            )
        )
        self.counts.update(encoding_estimates(width, changed, out, wire_bits))
        self.counts["dense_remote_macs"] += width * out
        self.counts["plaintext_support_oracle_remote_macs"] += changed * out
        self.counts["client_delta_subtractions"] += width
        self.counts["client_accumulator_additions"] += out

    def summary(self):
        counts: dict[str, int | float] = dict(self.counts)
        if counts.get("coordinates"):
            counts["unchanged_coordinate_fraction"] = 1 - (
                counts["changed_coordinates"] / counts["coordinates"]
            )
            counts["scale_change_fraction"] = (
                counts["activation_scale_bit_changes"] / counts["transitions"]
            )
        return counts | {
            "changed_fraction_distribution": distribution(self.changed_fractions),
            "raw_float_changed_fraction_distribution": distribution(
                self.raw_float_changed_fractions
            ),
            "relative_float_l2_distribution_not_exactness": distribution(self.relative_float_l2),
            "changed_coordinate_histogram": {
                str(k): v for k, v in sorted(self.changed_histogram.items())
            },
        }


def limited_fixture():
    # Reuse the fixture's binding/lock/tokenization; only its hard-coded thread
    # count needs a scoped constructor override. No shared helper edits.
    original = support.MaskedTransformerEngine

    def single_thread_engine(*args, **kwargs):
        kwargs["threads"] = 1
        return original(*args, **kwargs)

    with patch.object(support, "MaskedTransformerEngine", single_thread_engine):
        fixture = support.DecoderFixture(inputs=MAX_INPUTS, outputs=DECODE_STEPS + 1)
    if not fixture.engine.kernel.native or fixture.engine.kernel.threads != 1:
        raise RuntimeError("probe requires one-thread native checkpoint execution")
    schedule = fixture.plan.runtime_schedule(fixture.composition)
    if not schedule.complete or schedule.digest != fixture.compiled.runtime_schedule_digest:
        raise RuntimeError("checkpoint must bind a complete native semantic schedule")
    return fixture


def sdk_derived_scope(stage_metadata: dict) -> dict:
    """Project baseline semantic roles; never pretend candidate placement ran."""
    selected = {key: row for key, row in stage_metadata.items() if row["role"] in SDK_REMOTE_ROLES}
    counts = Counter()
    by_role = Counter()
    for row in selected.values():
        by_role[row["role"]] += 1
        counts["retained_client_previous_q_i8_bytes"] += row["retained_client_previous_q_i8_bytes"]
        counts["retained_client_exact_product_i32_bytes"] += row[
            "retained_client_exact_product_i32_bytes"
        ]
        counts["retained_client_activation_scale_f32_bytes"] += row[
            "retained_client_activation_scale_f32_bytes"
        ]
        counts["dense_initialization_remote_macs_per_sequence"] += (
            row["input_width"] * row["output_width"]
        )
        for metric, value in encoding_estimates(
            row["input_width"],
            row["input_width"],
            row["output_width"],
            row["current_exact_ring_bits"],
        ).items():
            if metric in {
                "dense_all_link_bytes",
                "dense_online_bytes",
                "dense_exact_ring_input_bytes",
            }:
                counts[f"{metric}_per_initialization_sequence"] += value
    counts["retained_client_temporal_state_bytes"] = sum(
        counts[key]
        for key in (
            "retained_client_previous_q_i8_bytes",
            "retained_client_exact_product_i32_bytes",
            "retained_client_activation_scale_f32_bytes",
        )
    )
    return {
        "execution_composition": "baseline MaskedLinearCpu",
        "derivation": "semantic-role subset of baseline checkpoint diagnostics",
        "distributed_candidate_placement_executed": False,
        "client_attention_roles": sorted(SDK_CLIENT_ATTENTION_ROLES),
        "remote_roles": sorted(SDK_REMOTE_ROLES),
        "remote_stage_count": len(selected),
        "remote_stage_count_by_role": dict(by_role),
        "remote_stage_ids": sorted(selected),
        "resources_per_live_sequence": dict(counts),
    }


def sdk_phase_arithmetic(groups: dict, cohort: str, phase: str) -> dict:
    counts = Counter()
    for (name, role, group_phase), measurement in groups.items():
        if name == cohort and role in SDK_REMOTE_ROLES and group_phase == phase:
            counts.update(
                {k: v for k, v in measurement.counts.items() if "bytes" in k or "macs" in k}
            )
    return dict(counts)


def run() -> dict:
    # Set BEFORE resolving/importing huggingface_hub. Never permit a download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[name] = "1"
    fixture = limited_fixture()
    stats = {}
    groups = {}
    stage_metadata = {}
    execution = Counter()
    workload = []
    cohort_locks = {}
    all_phase_bytes = {}

    for cohort, prompts in COHORTS.items():
        cohort_hasher = hashlib.sha256(b"pllm.temporal_delta.tokens.v1\0")
        cohort_counts = Counter()
        for prompt_index, prompt in enumerate(prompts):
            codec = fixture.compiled.runtime(fixture.remote)
            rendered = fixture.bundle.render_prompt([{"role": "user", "content": prompt}])
            ids = codec.encode_prompt(rendered)
            if not 2 <= len(ids) <= MAX_INPUTS:
                raise RuntimeError("public prompt outside untruncated probe bound")
            dense_scores, forcing = support.trajectory(
                fixture, ids, fixture.remote, count=DECODE_STEPS + 1
            )
            previous = {}
            stage_calls = Counter()

            def remote(key, activation):
                stage = fixture.body[key]
                spec = stage.spec
                q = quantize_activation_per_row(activation, bits=8)
                integer = np.asarray(stage.compiled_weight.clear(q.values), dtype=np.int64)
                integer = integer.reshape(q.rows, spec.out_features)
                phase = (
                    "prefill_adjacent_positions" if stage_calls[key] == 0 else "same_token_decode"
                )
                stage_calls[key] += 1
                stage_metadata[key] = {
                    "role": spec.role,
                    "layer": spec.layer_index,
                    "input_width": spec.in_features,
                    "output_width": spec.out_features,
                    "current_exact_ring_bits": stage.seeded_profile.wire_bits,
                    "independently_lifted_delta_ring_bits": seeded_ring_profile(
                        2 * stage.signed_output_bound
                    ).wire_bits,
                    "sdk_remote_under_attention_candidate_client_placement": spec.role
                    in SDK_REMOTE_ROLES,
                    "retained_client_previous_q_i8_bytes": spec.in_features,
                    "retained_client_exact_product_i32_bytes": 4 * spec.out_features,
                    "retained_client_activation_scale_f32_bytes": 4,
                    "remote_plaintext_previous_state_bytes_admitted": 0,
                }
                group = groups.setdefault((cohort, spec.role, phase), Measurements())
                stat = stats.setdefault((cohort, key, phase), Measurements())
                raw = np.asarray(activation, dtype=np.float32).reshape(q.rows, spec.in_features)
                if phase == "prefill_adjacent_positions":
                    pairs = (
                        (
                            q.values[i - 1],
                            q.values[i],
                            q.scales[i - 1],
                            q.scales[i],
                            raw[i - 1],
                            raw[i],
                        )
                        for i in range(1, q.rows)
                    )
                    execution["dense_initialization_rows"] += 1
                    initial = encoding_estimates(
                        spec.in_features,
                        spec.in_features,
                        spec.out_features,
                        stage.seeded_profile.wire_bits,
                    )
                    cohort_counts["dense_initialization_all_link_bytes"] += initial[
                        "dense_all_link_bytes"
                    ]
                    cohort_counts["dense_initialization_online_bytes"] += initial[
                        "dense_online_bytes"
                    ]
                    cohort_counts["dense_initialization_remote_macs"] += (
                        spec.in_features * spec.out_features
                    )
                else:
                    if q.rows != 1:
                        raise RuntimeError("decode must have exactly one stage input row")
                    prev_q, prev_s, prev_f, prev_product = previous[key]
                    delta = temporal_delta(prev_q, q.values[0])
                    # Native clear accepts i8 only. Two exact limbs cover the
                    # widened delta alphabet without i8 overflow or requantizing.
                    limb1 = np.clip(delta, -127, 127).astype(np.int8)
                    limb2 = (delta - limb1.astype(np.int16)).astype(np.int8)
                    correction = np.asarray(
                        stage.compiled_weight.clear(limb1[None]), dtype=np.int64
                    )
                    correction += np.asarray(
                        stage.compiled_weight.clear(limb2[None]), dtype=np.int64
                    )
                    updated = prev_product + correction[0]
                    if not np.array_equal(updated, integer[0]):
                        raise AssertionError(
                            "temporal native update differs from signed integer baseline"
                        )
                    modulus = stage.seeded_profile.modulus
                    if not np.array_equal(
                        (prev_product % modulus + correction[0] % modulus) % modulus,
                        integer[0] % modulus,
                    ):
                        raise AssertionError("exact-ring update differs from baseline")
                    execution["exact_native_signed_integer_decode_checks"] += 1
                    execution["exact_ring_decode_checks"] += 1
                    execution["native_delta_limb_macs"] += 2 * spec.in_features * spec.out_features
                    if prompt_index == 0 and stage_calls[key] == 2:
                        sparse = exact_sparse_update(stage.weight.values, prev_product, delta)
                        if not np.array_equal(sparse, integer[0]):
                            raise AssertionError(
                                "independent sparse update differs from native baseline"
                            )
                        execution["independent_sparse_integer_checks"] += 1
                        execution["independent_sparse_reference_macs"] += (
                            int(np.count_nonzero(delta)) * spec.out_features
                        )
                    integer[0] = updated
                    pairs = ((prev_q, q.values[0], prev_s, q.scales[0], prev_f, raw[0]),)
                for pair in pairs:
                    for target in (stat, group):
                        target.add(*pair, spec.out_features, stage.seeded_profile.wire_bits)
                previous[key] = (
                    q.values[-1].copy(),
                    q.scales[-1],
                    raw[-1].copy(),
                    integer[-1].copy(),
                )
                cohort_counts["executed_stage_rows"] += q.rows
                cohort_counts["replay_dense_baseline_macs"] += (
                    q.rows * spec.in_features * spec.out_features
                )
                out = dequantize_matmul(
                    integer,
                    q.scales,
                    stage.weight.scales,
                    output_shape=q.original_shape[:-1] + (spec.out_features,),
                )
                if stage.bias is not None:
                    out = out + stage.bias
                return np.ascontiguousarray(out, dtype=np.float32)

            scores, selected = support.trajectory(
                fixture, ids, remote, forcing=forcing, count=DECODE_STEPS + 1
            )
            if selected != forcing or any(
                a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes()
                for a, b in zip(scores, dense_scores)
            ):
                raise AssertionError(
                    "same-token replay changed exact float32 logits or greedy output"
                )
            if set(stage_calls) != set(fixture.body) or set(stage_calls.values()) != {
                DECODE_STEPS + 1
            }:
                raise RuntimeError("probe did not cover each native checkpoint body stage")
            execution["exact_float32_logit_rows"] += len(scores)
            cohort_hasher.update(np.asarray([len(ids), *ids, *forcing], dtype="<u4").tobytes())
            workload.append(
                {
                    "cohort": cohort,
                    "prompt_index": prompt_index,
                    "prefill_tokens": len(ids),
                    "same_token_decode_steps": DECODE_STEPS,
                }
            )
        cohort_locks[cohort] = {
            "token_trajectory_sha256": cohort_hasher.hexdigest(),
            **cohort_counts,
        }
        for phase in ("prefill_adjacent_positions", "same_token_decode"):
            all_phase_bytes[f"{cohort}/{phase}"] = sdk_phase_arithmetic(groups, cohort, phase)

    return {
        "schema": "pllm.temporal_delta_screen.v1",
        "source": fixture.lock(),
        "native_backend": fixture.engine.kernel.backend,
        "model_threads": 1,
        "workload": workload,
        "cohorts": cohort_locks,
        "verification": dict(execution),
        "prompts_sha256": support.digest(COHORTS),
        "controls_39_plus_32_context_only": {
            "sdk_attention_placement_warm_all_link_bytes": 148_297_986,
            "sdk_attention_placement_online_bytes": 93_221_048,
            "original_prepared_all_link_bytes": 178_970_558,
            "original_prepared_online_bytes": 113_545_024,
            "matched_to_this_short_cohort": False,
        },
        "sdk_remote_transition_arithmetic_only": all_phase_bytes,
        "sdk_derived_scope": sdk_derived_scope(stage_metadata),
        "by_role_phase": {"/".join(key): value.summary() for key, value in sorted(groups.items())},
        "stage_metadata": stage_metadata,
        "by_stage_phase": {"/".join(key): value.summary() for key, value in sorted(stats.items())},
        "state_scope": "One previous row per stage per live sequence; reset between prompts. Client: previous i8 input, exact i32 product, f32 scale. Diagnostic also retains floats and i64 products; not charged as a production layout. Private remote prior inputs/products require secret-shared or masked state and fresh correlations; none admitted.",
        "work_scope": "Single-process diagnostic emulates provider W products. Actual diagnostic repeats dense baseline, then dense replay plus two native delta limbs on decode and sampled NumPy sparse references. Total dense native MACs are twice the summed replay_dense_baseline_macs. Sparse oracle MACs assume plaintext support known remotely, not a private implementation. A hypothetical client does subtraction, accumulator addition, current-scale dequantization; no client dense W product proposed. No distributed performance measurement.",
        "privacy_scope": "No raw prompts, tokens, activations, scales, supports, logits, or weights serialized. Aggregate diagnostics are offline research output, not allowed private telemetry. Fresh dense masking destroys visible zero support; count, support, fallback and packet length cannot depend on private activity. Reused masks forbidden.",
        "encoding_scope": "Known-count support, plaintext nine-bit alphabet, and ring-slot sparse estimates are non-oblivious optimistic bounds. Count/framing/index-hiding, dense first rows, state distribution, masks, correlation generation, output refresh, HTTP, and other model operators are excluded from transition estimates. Fully dense outputs/corrections retained. No per-stage estimate authorizes a whole-model saving.",
        "exactness_scope": "Every decode stage verifies full signed integer equality and original-ring residues. Independent sparse NumPy reference checks first decode of first prompt in each cohort. Current activation scale and unchanged weight scale are reapplied; raw float proximity is never equality. Prefill delta sparsity measured, not incrementally executed.",
        "protocol_admission": False,
        "decision": "Temporal coordinate sparsity is only a first kill gate; no oblivious sparse protocol or runtime selection is admitted.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="omit per-stage phase histograms")
    args = parser.parse_args()
    result = run()
    if args.summary:
        result.pop("by_stage_phase")
        metadata = result.pop("stage_metadata")
        retained = {}
        for key, row in sorted(metadata.items()):
            role = row["role"]
            group = retained.setdefault(
                role,
                {
                    "stage_ids": [],
                    "input_width": row["input_width"],
                    "output_width": row["output_width"],
                    "previous_q_i8_bytes_per_stage": row["retained_client_previous_q_i8_bytes"],
                    "exact_product_i32_bytes_per_stage": row[
                        "retained_client_exact_product_i32_bytes"
                    ],
                    "activation_scale_f32_bytes_per_stage": 4,
                    "sdk_remote_under_attention_candidate_client_placement": row[
                        "sdk_remote_under_attention_candidate_client_placement"
                    ],
                    "current_exact_ring_bits_by_stage": {},
                    "independently_lifted_delta_ring_bits_by_stage": {},
                },
            )
            if (group["input_width"], group["output_width"]) != (
                row["input_width"],
                row["output_width"],
            ):
                raise RuntimeError("summary needs uniform role widths")
            group["stage_ids"].append(key)
            group["current_exact_ring_bits_by_stage"][key] = row["current_exact_ring_bits"]
            group["independently_lifted_delta_ring_bits_by_stage"][key] = row[
                "independently_lifted_delta_ring_bits"
            ]
        result["retained_state_by_role"] = retained
        for row in result["by_role_phase"].values():
            row.pop("changed_coordinate_histogram")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
