"""Select mixed numeric overrides by exact wire thresholds, then test held-out decoding.

This uses the compiled clear-kernel diagnostic, not a selectable mixed-precision
Experiment. Arithmetic body savings are projected; HTTP, setup and keys excluded.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter

import numpy as np

from pllm.runtime.preparation_protocol import seeded_ring_profile
from pllm.runtime.quantization import signed_qmax

from decoder_probe_support import DecoderFixture, PROMPT, ROOT, digest, stage_output, trajectory


def wire_bits(weight: np.ndarray, activation_bits: int) -> int:
    l1 = int(np.max(np.sum(np.abs(weight.astype(np.int16)), axis=1, dtype=np.int64)))
    return seeded_ring_profile(l1 * signed_qmax(activation_bits)).wire_bits


def run(*, structure_only=False) -> dict:
    fixture = DecoderFixture()
    lower = {}
    for key, stage in fixture.body.items():
        lower[key] = fixture.engine._load_stage(
            fixture.model.store,
            dataclasses.replace(stage.spec, weight_bits=4, activation_bits=8),
            fixture.model.manifest,
        )
    candidates = {}
    for role in sorted({stage.spec.role for stage in fixture.body.values()}):
        choices = []
        stages = [(key, stage) for key, stage in fixture.body.items() if stage.spec.role == role]
        for wb in (8, 4):
            for ab in (8, 7, 6, 4):
                if (wb, ab) == (8, 8):
                    continue
                saved = sum(
                    (
                        stage.seeded_profile.wire_bits
                        - wire_bits((stage if wb == 8 else lower[key]).weight.values, ab)
                    )
                    // 8
                    * 70
                    * (stage.spec.in_features + 2 * stage.spec.out_features)
                    for key, stage in stages
                )
                if saved > 0:
                    choices.append((wb + ab, wb, ab, saved))
        if choices:
            _, wb, ab, saved = max(choices)
            candidates[role] = {
                "role": role,
                "weight_bits": wb,
                "activation_bits": ab,
                "saved_bytes": saved,
            }
    if "mlp_down" in candidates:
        layers = sorted(
            {
                stage.spec.layer_index
                for stage in fixture.body.values()
                if stage.spec.role == "mlp_down"
            }
        )
        for name, selected in (
            ("every4", layers[::4]),
            ("last4", layers[-4:]),
            ("last1", layers[-1:]),
        ):
            numeric = dict(candidates["mlp_down"], layers=selected)
            numeric["saved_bytes"] = sum(
                (
                    stage.seeded_profile.wire_bits
                    - wire_bits(
                        (stage if numeric["weight_bits"] == 8 else lower[key]).weight.values,
                        numeric["activation_bits"],
                    )
                )
                // 8
                * 70
                * (stage.spec.in_features + 2 * stage.spec.out_features)
                for key, stage in fixture.body.items()
                if stage.spec.role == "mlp_down" and stage.spec.layer_index in selected
            )
            candidates[f"mlp_down_{name}"] = numeric

    def callback(roles, counts=None):
        def remote(key, activation):
            stage = fixture.body[key]
            numeric = roles.get(stage.spec.role, {"weight_bits": 8, "activation_bits": 8})
            if "layers" in numeric and stage.spec.layer_index not in numeric["layers"]:
                numeric = {"weight_bits": 8, "activation_bits": 8}
            chosen = stage if numeric["weight_bits"] == 8 else lower[key]
            if counts is not None:
                rows = int(activation.size // stage.spec.in_features)
                bits = wire_bits(chosen.weight.values, numeric["activation_bits"])
                counts["online"] += (
                    rows * (stage.spec.in_features + stage.spec.out_features) * bits // 8
                )
                counts["offline_correction"] += rows * stage.spec.out_features * bits // 8
                counts["body_integer_macs"] += (
                    rows * stage.spec.in_features * stage.spec.out_features
                )
            return stage_output(chosen, activation, numeric["activation_bits"])

        return remote

    result = {
        "schema": "pllm.wire_precision_screen.v1",
        "source": fixture.lock(),
        "non_selectable_numeric_override": True,
        "candidates": candidates,
        "projection_scope": "integer input/output/correction bodies only; no protocol framing, setup, full wire or distribution",
    }
    if structure_only:
        return result
    path = ROOT / "examples/benchmarks/wire_precision_prompts.json"
    cohorts = json.loads(path.read_text())
    texts = [text for items in cohorts.values() for text in items]
    if len(set(texts)) != len(texts) or set(cohorts) != {"tuning", "heldout", "confirmation"}:
        raise ValueError("public tuning and held-out cohorts must be disjoint")
    ids = {name: [fixture.tokens(text) for text in items] for name, items in cohorts.items()}
    if len({tuple(row) for rows in ids.values() for row in rows}) != len(texts):
        raise ValueError("different prompts collapse after bounded tokenization")
    result["public_cohort_digest"] = digest(cohorts)
    result["token_cohort_digest"] = digest(ids)

    def evaluate(rows, roles):
        matches = [0, 0]
        worst = 0.0
        for tokens in rows:
            expected, selected = trajectory(fixture, tokens, fixture.remote)
            actual, _ = trajectory(fixture, tokens, callback(roles), forcing=selected)
            for index in range(2):
                matches[index] += int(np.argmax(actual[index])) == selected[index]
                worst = max(worst, float(np.max(np.abs(actual[index] - expected[index]))))
        return {
            "samples": len(rows),
            "prefill_matches": int(matches[0]),
            "same_token_decode_matches": int(matches[1]),
            "worst_logit_error": worst,
        }

    tuning = {
        name: evaluate(ids["tuning"], {numeric["role"]: numeric})
        for name, numeric in candidates.items()
    }
    eligible = [
        role
        for role, score in tuning.items()
        if score["prefill_matches"] == score["same_token_decode_matches"] == score["samples"]
    ]
    winner = max(eligible, key=lambda role: candidates[role]["saved_bytes"], default=None)
    result["tuning"] = tuning
    result["selected_candidate"] = winner
    if winner is None:
        result["decision"] = "no candidate preserves all tuning prefill/decode selections"
        return result
    roles = {candidates[winner]["role"]: candidates[winner]}
    result["override_digest"] = digest(roles)
    result["heldout"] = {name: evaluate(ids[name], roles) for name in ("heldout", "confirmation")}
    matched = {}
    tokens = fixture.tokens(PROMPT)
    if len(tokens) != 39:
        raise ValueError("locked prompt no longer contains 39 tokens")
    for count in (8, 32):
        base_counts, mixed_counts = Counter(), Counter()
        expected, base_tokens = trajectory(fixture, tokens, callback({}, base_counts), count=count)
        actual, mixed_tokens = trajectory(
            fixture, tokens, callback(roles, mixed_counts), count=count
        )
        matched[str(count)] = {
            "baseline_arithmetic": dict(base_counts),
            "mixed_arithmetic": dict(mixed_counts),
            "free_generation_matches": sum(
                a == b for a, b in zip(base_tokens, mixed_tokens, strict=True)
            ),
            "output_tokens": count,
            "baseline_output_digest": digest(base_tokens),
            "mixed_output_digest": digest(mixed_tokens),
            "trajectory_logit_error": max(
                float(np.max(np.abs(a - b))) for a, b in zip(expected, actual, strict=True)
            ),
        }
    result["matched_39_token_requests"] = matched
    result["decision"] = (
        "held-out diagnostic only; mixed-stage commitments and live protocol remain unimplemented"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure-only", action="store_true")
    print(
        json.dumps(run(structure_only=parser.parse_args().structure_only), indent=2, sort_keys=True)
    )
