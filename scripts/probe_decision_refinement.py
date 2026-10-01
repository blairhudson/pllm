"""Sound correlated logit-difference bounds on an already-exact Qwen head.

This isolates certificate utility. All body operations and KV run unmodified;
no body traffic is avoided and no empirical error bar is treated as a proof.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from pllm.runtime.decision_refinement_reference import refine_linear_choice
from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle

from decoder_probe_support import DecoderFixture, ROOT, digest, trajectory


def run() -> dict:
    fixture = DecoderFixture()
    cohorts = json.loads((ROOT / "examples/benchmarks/wire_precision_prompts.json").read_text())
    original = ClientBundle.local_linear
    captured = []

    def observe(bundle, stage_id, activation):
        result = original(bundle, stage_id, activation)
        if stage_id == "lm_head":
            quantized = quantize_activation_per_row(activation, bits=8)
            captured.append(
                (
                    quantized.values[-1].copy(),
                    float(quantized.scales[-1]),
                    int(np.argmax(result[-1])),
                )
            )
        return result

    token_cohort = []
    with patch.object(ClientBundle, "local_linear", observe):
        for name in ("heldout", "confirmation"):
            for text in cohorts[name][:3]:
                ids = fixture.tokens(text)
                token_cohort.append(ids)
                trajectory(fixture, ids, fixture.remote)
    head = fixture.bundle.stages["lm_head"]
    results = []
    for x, scale, expected in captured:
        row = refine_linear_choice(head.client_weight, x, head.client_weight_scales, scale)
        if row.pop("selected") != expected:
            raise AssertionError("certificate/fallback changed rounded W8A8 head decision")
        results.append(row)
    return {
        "schema": "pllm.decision_refinement_gate.v1",
        "source": fixture.lock(),
        "token_cohort_digest": digest(token_cohort),
        "public_cohort_digest": digest(cohorts),
        "checked_prefill_and_decode_heads": len(results),
        "rounded_head_parity": True,
        "certified_before_full_head": sum(row["certified_before_full_head"] for row in results),
        "resolved_feature_counts": [row["resolved_features"] for row in results],
        "head_width": head.client_weight.shape[1],
        "body_work_avoided": 0,
        "provider_bytes_avoided": 0,
        "cost_caveat": "certificate scans unresolved public coefficients; fewer dot products is not measured CPU saving",
        "state_scope": "exact body/KV retained; no certificate for omitted body computation",
        "decision": "linear-head certificate feasibility only; complete decoder bounds and private refinement unavailable",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
