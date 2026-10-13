"""Independent lifecycle checks; no protected execution or timing claims."""
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/probe_bootstrap_budget.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_budget_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_both_candidates_pay_one_shared_public_delivery():
    model = probe.LifecycleBudget(125, 369, 105, 106)
    assert model.totals(1) == (494, 493)
    assert model.totals(5) == (1970, 1545)
    assert model.first_horizon(Fraction(1, 5)) == 5
    # Counting the control's first-delivery bytes again would inflate savings.
    assert model.saving(5) == Fraction(425, 1970)


def test_finite_mask_tape_has_no_horizon_saving():
    for horizon in (1, 2, 17, 4096):
        model = probe.LifecycleBudget(125, 369, horizon * 139, 139)
        assert model.totals(horizon)[0] == model.totals(horizon)[1]


def test_renewals_charge_initial_setup_and_break_monotonicity():
    model = probe.LifecycleBudget(10, 100, 0, 50, renewal_bytes=60, renewal_pairs=2)
    control, candidate = 10, 10
    for pair in range(1, 13):
        control += 100
        candidate += 50 + (60 if pair % 2 else 0)
        assert model.totals(pair) == (control, candidate)
    assert model.saving(3) < model.saving(2)
    assert model.first_horizon(Fraction(1, 10)) == 2


def test_exact_target_boundary_and_unattainable_ceiling():
    assert probe.LifecycleBudget(0, 100, 0, 20).first_horizon(Fraction(1, 5)) == 1
    assert probe.LifecycleBudget(1, 100, 0, 20).first_horizon(Fraction(1, 5)) is None
    assert probe.LifecycleBudget(0, 100, 1, 20).first_horizon(Fraction(1, 5)) is None
    model = probe.LifecycleBudget(0, 100, 10, 30)
    assert model.first_horizon(Fraction(1, 5)) == 1
    assert model.additional_helper_budget(1, Fraction(1, 5)) == 0
    assert model.additional_helper_budget(2, Fraction(1, 5)) == 5


@pytest.mark.parametrize("values", [
    (True, 100, 0, 20), (-1, 100, 0, 20), (0, 0, 0, 0),
    (0, 100, 0, 101), (0, 100, 0, 20, -1),
])
def test_reject_bad_byte_owners(values):
    with pytest.raises(ValueError):
        probe.LifecycleBudget(*values)


def test_reject_unbounded_or_unpriced_renewal():
    with pytest.raises(ValueError):
        probe.LifecycleBudget(0, 100, 0, 20, renewal_bytes=10)
    for interval in (0, -1, True):
        with pytest.raises(ValueError):
            probe.LifecycleBudget(0, 100, 0, 20, renewal_pairs=interval)
    model = probe.LifecycleBudget(0, 100, 0, 20)
    for count in (0, True, probe.MAX_HORIZON + 1):
        with pytest.raises(ValueError):
            model.totals(count)


def test_control_rejects_conserving_category_substitution(tmp_path):
    document = json.loads(probe.CONTROL.read_text())
    document["control"]["classes"]["corrections"] += 1
    document["control"]["classes"]["outputs"] -= 1
    target = tmp_path / "forged.json"
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="directed evidence"):
        probe.load_control(target)


def test_reference_limits_and_unknown_metrics_remain_explicit():
    report = probe.sensitivity_report()
    assert report["control_pair_bytes"] == 493575247
    assert report["arithmetic_pair_bytes"] == 368627357
    assert report["low_tier_qualified"] is None
    assert report["complete_per_generated_token_bytes"] is None
    for scenario in report["scenarios"]:
        assert scenario["horizons"][0]["control_bytes"] == report["control_pair_bytes"]
        assert scenario["first_target_horizon_pairs"]["50"] is None


def test_prime_codec_composition_charges_full_32_bit_ingress():
    report = probe.prime_field_payload_screen(probe.load_control())
    assert report["prime"] > 2**31
    assert report["input_bits"] == 32
    assert report["stage_count"] == 96
    assert report["executed_stage_rows"] == 15297
    assert report["old_input_payload_bytes"] == 90_104_112
    assert report["prime_input_payload_bytes"] == 114_931_712
    assert report["added_input_payload_bytes"] == 24_827_600
    assert report["unchanged_output_payload_bytes"] == 138_864_059
    assert report["optimistic_removed_pair_bytes_after_input_expansion"] == 114_233_089
    cartridge = next(c for c in report["alternatives"] if c["artifact"] == "two_accumulator_checkpoint_d64")
    assert cartridge["first_target_horizon_pairs"] == {"20": 19, "25": 34, "30": 207, "35": None, "50": None}


def test_prime_width_bound_agrees_with_actual_codec_and_extreme_recovery():
    spec = importlib.util.spec_from_file_location("budget_codec_oracle", PATH.with_name("probe_cyclic_mask_codec.py"))
    codec = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = codec
    spec.loader.exec_module(codec)
    report = probe.prime_field_payload_screen(probe.load_control())
    for width in report["recorded_row_widths"]:
        bound = (1 << (width - 1)) - 1
        g = codec.Geometry.symmetric(report["prime"], bound)
        assert g.bits == width
        for x in (-bound, -1, 0, 1, bound):
            for mask in (0, bound, g.q // 2, g.q - 1):
                assert g.decode(g.encode((x + mask) % g.q), mask) == x


def test_prime_screen_rejects_mismatched_source_rows(monkeypatch, tmp_path):
    forged = json.loads(probe.ROW_DEMAND.read_text())
    forged["budget"]["stages"][0]["weight_sha256"] = "0" * 64
    path = tmp_path / "forged-rows.json"
    path.write_text(json.dumps(forged))
    monkeypatch.setattr(probe, "ROW_DEMAND", path)
    with pytest.raises(ValueError, match="weight mismatch"):
        probe.prime_field_payload_screen(probe.load_control())
