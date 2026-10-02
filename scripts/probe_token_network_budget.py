"""Pinned 100x necessities and falsification of two new compression hypotheses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from pllm import Model, lower_model
from pllm.assurance import PublicPolynomialShiftRegression
from pllm.metrics import TokenNetworkBudgetProbe
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.semantic_stages import scheduled_stage_specs
from probe_region_contract_cost import _CONTROL, _MODEL, _REVISION, _ROOT, _SHARED_HUB_CACHE


def _shift_witnesses():
    rng = random.Random(17491)  # Public attack fixture, not protocol randomness.
    results = []
    for bits in (8, 16, 24, 32, 64):
        regression = PublicPolynomialShiftRegression(bits)
        q = 1 << bits
        recovered = 0
        for _ in range(64):
            g, u, a, b = [rng.getrandbits(bits) for _ in range(4)]
            witness = regression.evaluate(
                masked_gate=(g + a) % q,
                masked_up=(u + b) % q,
                coefficient_gu=(256 - 2 * a) % q,
                coefficient_g2=(-b) % q,
                linear_coefficient=256 % q,
            )
            recovered += witness.up_value == u and witness.gate_residue == g % (q // 2)
        results.append(
            {
                "ring_bits": bits,
                "cases": 64,
                "exact_leakage_witnesses": recovered,
                "up_bits_exposed": bits,
                "gate_bits_exposed": bits - 1,
            }
        )
    return results


def run():
    source = Path(
        hf_hub_download(
            _MODEL,
            "config.json",
            revision=_REVISION,
            cache_dir=_SHARED_HUB_CACHE,
            local_files_only=True,
        )
    )
    config = json.loads(source.read_text())
    composition = MaskedLinearCpu(
        Model.hf(_MODEL, revision=_REVISION),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    evidence = json.loads(_CONTROL.read_text())
    reports = {}
    plans = {}
    controls = {}
    for outputs in (8, 32):
        plan = lower_model(config, batch=1, max_input_tokens=39, max_new_tokens=outputs)
        record = next(row for row in evidence["cohorts"] if row["output_tokens"] == outputs)
        if plan.digest != record["official_plan_digest"]:
            raise ValueError("network budget source differs from the locked plan")
        control = record["prepared_control"]
        report = TokenNetworkBudgetProbe().project(
            plan,
            composition,
            response_new_tokens=outputs,
            baseline_online_body_bytes=control["online_all_link_body_bytes"],
            baseline_total_body_bytes=control["covered_all_link_body_bytes"],
        )
        if (
            report["schedule_digest"] != record["official_schedule_digest"]
            or report["composition_digest"] != evidence["source"]["pipeline_digest"]
        ):
            raise ValueError("network budget pipeline/schedule differs from baseline")
        reports[str(outputs)], plans[outputs], controls[outputs] = report, plan, control

    quality_path = _ROOT / "docs/evidence/projected-polynomial-quality-qwen25-2026-10-02.json"
    quality = json.loads(quality_path.read_text())
    if quality["source"]["revision"] != _REVISION or quality["source"]["model"] != _MODEL:
        raise ValueError("residual evidence differs from the target checkpoint")
    stages = scheduled_stage_specs(plans[32], composition)
    mlps = [stage for stage in stages if stage.role == "mlp_down"]
    if {(s.in_features, s.out_features) for s in mlps} != {(4864, 896)}:
        raise ValueError("residual gate requires its measured uniform source dimensions")
    r = reports["32"]
    row_count = r["executed_rows"] * len(mlps)
    issuance_count = (math.ceil(r["input_tokens"] / 16) + 31) * len(mlps)
    residual_hypotheses = {}
    for degree in (1, 2, 4):
        observed = quality["results"][f"public_degree_{degree}"]["optimistic_quantized_residual"]
        mean_count = math.ceil(observed["nonzero_code_corrections"] / observed["rows"])
        worst_count = observed["maximum_active_count"]
        # 13 index bits + 9 signed difference bits, delivered to two parties.
        # This is an ideal hidden-scatter oracle, not a privacy-preserving codec.
        tuple_bits = math.ceil(math.log2(4864)) + 9
        mean_body = row_count * math.ceil(2 * mean_count * tuple_bits / 8)
        worst_body = row_count * math.ceil(2 * worst_count * tuple_bits / 8)
        # Generalized shifted degree-d polynomial × up: two linear coefficients
        # derive from masks; 2(d-1) channel coefficients remain plus one output.
        # This formula is an optimistic extension, not an implemented evaluator.
        core_material = row_count * (2 * (degree - 1) * 4864 + 896) * 3 + issuance_count * 178
        residual_hypotheses[str(degree)] = {
            "source_scope": "public reference trajectories with free true scale; different prompt cohort from cost controls",
            "observed_mean_active_fraction": observed["mean_active_fraction"],
            "observed_maximum_active_count": worst_count,
            "hypothetical_padding_capacity": worst_count,
            "position_bits": 13,
            "signed_delta_bits": 9,
            "mean_rate_ideal_two_party_tuple_body_bytes": mean_body,
            "observed_worst_row_rate_ideal_two_party_tuple_body_bytes": worst_body,
            "optimistic_seeded_polynomial_core_material_bytes": core_material,
            "known_combined_body_floor_at_mean_rate": core_material + mean_body,
            "all_link_100x_budget_bytes": r["response_all_link_budget_bytes"],
            "decision_with_this_correlation_core": "veto"
            if core_material + mean_body > r["response_all_link_budget_bytes"]
            else "inconclusive",
            "public_worst_case_capacity_certificate": False,
            "missing": [
                "private residual discovery and oblivious scatter",
                "certified fixed traffic capacity",
                "true dynamic scale and range computation",
                "complete attention/norm/KV work",
                "opaque cross-layer state, token feedback, control and framing",
            ],
        }
    finite_difference = {}
    for name, field in (
        ("online", "online_all_link_body_bytes"),
        ("all_link", "covered_all_link_body_bytes"),
    ):
        delta = controls[32][field] - controls[8][field]
        finite_difference[name] = {
            "extra_recorded_bytes": delta,
            "extra_output_tokens": 24,
            "recorded_bytes_per_extra_output_token": delta / 24,
            "hundredfold_target_bytes_per_extra_output_token": delta / 2400,
        }
    result = {
        "schema": "pllm.hundredfold_hypothesis_screen.v1",
        "source": evidence["source"],
        "budgets": reports,
        "finite_difference_between_39_plus_8_and_39_plus_32": {
            **finite_difference,
            "scope": "finite difference of matched source/body response cohorts; not an isolated marginal-decode measurement",
        },
        "sparse_exact_residual_hypothesis": residual_hypotheses,
        "public_polynomial_mask_handoff_hypothesis": {
            "native_witnesses": _shift_witnesses(),
            "decision": "privacy_veto",
            "scope": "exposed full shifted coefficients leak inputs; no attack on opaque party-local shares",
        },
        "code_sha256": {
            path: hashlib.sha256((_ROOT / path).read_bytes()).hexdigest()
            for path in (
                "scripts/probe_token_network_budget.py",
                "python/pllm/metrics/token_budget.py",
                "crates/pllm-assurance/src/polynomial_shift.rs",
            )
        },
        "quality_evidence_sha256": hashlib.sha256(quality_path.read_bytes()).hexdigest(),
        "hundredfold_whole_decoder_admitted": False,
    }
    print(
        f"39+32: {r['response_online_budget_bytes']:,} online / {r['response_all_link_budget_bytes']:,} all-link bytes"
    )
    print(
        f"At most {r['maximum_full_width_openings_per_row_if_everything_else_free']} full-width openings/row "
        f"if everything else free; {r['full_width_client_feedback_scenario']['maximum_openings_per_row_after_feedback']} after full-width feedback"
    )
    quartic = residual_hypotheses["4"]
    print(
        f"Quartic ideal residual tuples: {quartic['mean_rate_ideal_two_party_tuple_body_bytes']:,} mean-rate / "
        f"{quartic['observed_worst_row_rate_ideal_two_party_tuple_body_bytes']:,} observed-worst-rate bytes; "
        f"core material {quartic['optimistic_seeded_polynomial_core_material_bytes']:,}"
    )
    witnesses = result["public_polynomial_mask_handoff_hypothesis"]["native_witnesses"]
    print(
        f"Exposed polynomial handoff: {sum(v['exact_leakage_witnesses'] for v in witnesses)}/"
        f"{sum(v['cases'] for v in witnesses)} public cases reveal up and all but one gate bit"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
