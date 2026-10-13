"""Independent boundary and accounting controls; no model or crypto imports."""
import importlib.util
from fractions import Fraction
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_transciphered_linear_budget.py"
SPEC = importlib.util.spec_from_file_location("transciphered_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def test_signed_bytes_and_lift():
    assert [p.signed_byte(x) for x in (0, 127, 128, 255)] == [0, 127, -128, -1]
    assert p.prime_lift(p.PLAIN - 65536, p.PLAIN, 65536, 16) == 0
    assert p.prime_lift(p.PLAIN - 1, p.PLAIN, 1, 24) == (1 << 24) - 1
    with pytest.raises(ValueError):
        p.prime_lift(1, 257, 129, 16)
    with pytest.raises(ValueError):
        p.prime_lift(9, 257, 8, 16)
    with pytest.raises(ValueError):
        p.signed_byte(-1)


def test_padding_and_dependency_underfill():
    assert p.packing_count(896, 150, 14, 8192) == 17 + 14
    assert p.packing_count(9728, 1, 14, 8192) == 2 + 28
    assert p.packing_count(8192, 1, 1, 8192) == 2
    assert p.packing_count(8193, 1, 1, 8192) == 4
    assert p.packing_count(896, 150, 14, 8192) > p.ceildiv(896 * 164, 8192)
    with pytest.raises(ValueError):
        p.ceildiv(1, 0)


def test_conservation_and_common_public_reuse():
    for reuse in (True, False):
        row = p.ledger(1, 1, public_reuse=reuse)
        assert row["control"] == 493575247
        assert row["candidate"] == 293143374
        assert row["saving"] == 200431873
    warm = p.ledger(2, 1)
    cold = p.ledger(2, 1, public_reuse=False)
    assert cold["control"] - warm["control"] == p.C0
    assert cold["candidate"] - warm["candidate"] == p.C0


def test_expansion_exact_boundary():
    for pairs in (1, 4, 100):
        r = p.expansion_limit(Fraction(1, 5), pairs)
        assert p.ledger(pairs, r)["saving_fraction"] == Fraction(1, 5)
        assert p.ledger(pairs, r + Fraction(1, p.OUTPUT))["saving_fraction"] < Fraction(1, 5)
    assert float(p.expansion_limit(Fraction(1, 5))) == pytest.approx(1.9123875669146606)


def test_setup_renewal_failure_not_double_counted():
    base = p.ledger(3, Fraction(3, 2))
    charged = p.ledger(3, Fraction(3, 2), setup=100, renewal=70, lifetime=2,
                       recurring_extra=11, failures=13)
    assert charged["candidate"] - base["candidate"] == 100 + 70 + 33 + 13
    assert p.ledger(2, 1, setup=100, renewal=70, lifetime=2)["issuances"] == 1
    assert p.ledger(3, 1, setup=100, renewal=70, lifetime=2)["issuances"] == 2


def test_bootstrap_break_even():
    target, r, setup = Fraction(1, 5), Fraction(3, 2), 100000000
    first = p.first_horizon(target, r, setup)
    assert p.ledger(first, r, setup=setup)["saving_fraction"] >= target
    assert p.ledger(first - 1, r, setup=setup)["saving_fraction"] < target
    assert p.first_horizon(target, 9, 0) is None


def test_hidden_chosen_input_work_rejected():
    assert p.admit_placement(0, 1000)
    assert p.admit_placement(99, 1000)
    assert p.admit_placement(100, 1000)
    assert not p.admit_placement(101, 1000)
    assert not p.admit_placement(1000, 1000)  # client W*x/repair is not remote FHE


def test_exhaustive_algebra():
    assert p.algebra_control()["outputs_checked"] == 4608


def test_saved_geometry_and_serialization_rejection():
    geometry = p.geometry_screen({"executed": False, "levels": [
        {"data_primes": 1, "noise_bits": 19, "exact": True, "serialized_bytes": 123532}]})
    assert geometry["executed_rows"] == 15297
    assert [layout["ciphertexts"] for layout in geometry["layouts"]] == [12782, 7111]
    best = geometry["layouts"][1]["measured_serialization_projection"][0]
    assert best["hypothetical_output_bytes"] == 879118708
    assert Fraction(best["optimistic_recurring_saving_fraction"]) < 0
    assert not best["full_width_noise_certified"]


@pytest.mark.parametrize("exact,noise", [(False, 19), (True, 0), (True, -1)])
def test_invalid_cheap_reply_excluded(exact, noise):
    geometry = p.geometry_screen({"executed": False, "levels": [
        {"data_primes": 1, "noise_bits": noise, "exact": exact, "serialized_bytes": 1}]})
    assert len(geometry["rejected_modulus_levels"]) == 1
    assert all(not layout["measured_serialization_projection"] for layout in geometry["layouts"])


def test_stage_identity_and_original_ring_validation():
    stage = {"stage": "toy", "shape": [1, 1], "weight_sha256": "public",
             "original_ring_bits": 24, "stage_bits": 23}
    narrow = {**stage}
    saved = {**stage, "ring_bits": 24}
    p.validate_stage_contracts([stage], [narrow], [saved])
    with pytest.raises(ValueError, match="duplicate"):
        p.validate_stage_contracts([stage, stage], [narrow], [saved])
    with pytest.raises(ValueError, match="duplicate"):
        p.validate_stage_contracts([stage], [narrow, narrow], [saved])
    with pytest.raises(ValueError, match="sets differ"):
        p.validate_stage_contracts([stage], [narrow], [{**saved, "stage": "foreign"}])
    with pytest.raises(ValueError, match="ring/residue"):
        p.validate_stage_contracts([{**stage, "original_ring_bits": 32}], [narrow], [saved])
    with pytest.raises(ValueError, match="ring/residue"):
        p.validate_stage_contracts([stage], [{**narrow, "stage_bits": 24}], [saved])
