"""Honest-majority replicated-sharing cost gate; no executable decoder claimed."""

from __future__ import annotations

import json

import numpy as np

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_stages import scheduled_stage_specs

from decoder_probe_support import MODEL, REVISION, ROOT


def replicated_linear_oracle() -> None:
    rng = np.random.default_rng(20261002)
    modulus = 1 << 32
    for _ in range(50):
        x = rng.integers(-127, 128, (2, 7), dtype=np.int64)
        w = rng.integers(-127, 128, (5, 7), dtype=np.int64)
        a, b = rng.integers(0, modulus, (2, 2, 7), dtype=np.int64)
        c = (x - a - b) % modulus
        shares = [a, b, c]
        outputs = [(share @ w.T) % modulus for share in shares]
        np.testing.assert_array_equal(sum(outputs) % modulus, (x @ w.T) % modulus)
        # Replicated party i owns (share_i,share_i+1); linear outputs have same layout.
        for i in range(3):
            np.testing.assert_array_equal(outputs[i], (shares[i] @ w.T) % modulus)
        y = rng.integers(-127, 128, x.shape, dtype=np.int64)
        u, v = rng.integers(0, modulus, (2, *x.shape), dtype=np.int64)
        other = [u, v, (y - u - v) % modulus]
        masks = rng.integers(0, modulus, (3, *x.shape), dtype=np.int64)
        # Each party computes with only its adjacent pairs, then sends one share.
        # NumPy overflow retains low 32 bits, matching this exact modular oracle.
        products = [
            (
                shares[i] * other[i]
                + shares[i] * other[(i + 1) % 3]
                + shares[(i + 1) % 3] * other[i]
                + masks[i]
                - masks[(i - 1) % 3]
            )
            % modulus
            for i in range(3)
        ]
        np.testing.assert_array_equal(sum(products) % modulus, (x * y) % modulus)


def run() -> dict:
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    replicated_linear_oracle()
    config = Path(
        hf_hub_download(MODEL, "config.json", revision=REVISION, local_files_only=True)
    ).read_bytes()
    composition = MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    controls = json.loads(
        (ROOT / "docs/evidence/latent-response-network-qwen25-2026-09-28.json").read_text()
    )
    reports = {}
    for outputs in (8, 32):
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        schedule = plan.runtime_schedule(composition)
        if not schedule.complete:
            raise ValueError("three-party screen requires complete semantic control")
        stages = [
            s
            for s in scheduled_stage_specs(plan, composition)
            if s.role not in {"token_lookup", "lm_head"}
        ]
        rows = 39 + outputs - 1
        linear = rows * sum(s.in_features * s.out_features for s in stages)
        gates = sum(
            int(op["output_shape"][-1])
            for op in plan.prefill["operations"]
            if op["operator"] == "multiply" and op.get("layer") is not None
        )
        reports[str(outputs)] = {
            "plan_digest": plan.digest,
            "schedule_digest": schedule.digest,
            "executed_rows": rows,
            "two_offset_linear_macs": 2 * linear,
            "ordinary_replicated_linear_macs": 6 * linear,
            "unique_compute_then_reshare_linear_macs": 3 * linear,
            "unique_compute_linear_peer_bytes_32bit": 3
            * 4
            * rows
            * sum(s.out_features for s in stages),
            "gated_product_peer_bytes_32bit": 3 * 4 * rows * gates,
            "linear_mac_multiplier_ordinary_vs_offset": 3.0,
            "linear_mac_multiplier_reshare_vs_offset": 1.5,
            "online_target_bytes": (7_383_174 if outputs == 8 else 11_354_502),
        }
    return {
        "schema": "pllm.three_party_cost_gate.v1",
        "control_evidence_schema": controls["schema"],
        "source": {"model": MODEL, "revision": REVISION},
        "cohorts": reports,
        "protocol_source": "https://eprint.iacr.org/2018/403",
        "privacy": "three noncolluding operators; one semi-honest corruption; each holds two adjacent shares, never all three",
        "scope": "specified replicated arithmetic layouts, 32-bit residues; not exact float decoder or measured CPU/wire",
        "unknown": [
            "private quantization",
            "exact truncation",
            "softmax/SiLU",
            "state lifecycle",
            "full wire",
            "operator independence",
        ],
        "decision": "tested gated multiplication alone exceeds online target; 3/6 public-linear copies also exceed 2-copy arithmetic comparator",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
