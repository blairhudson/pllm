"""Independent bounded oracles; no installed SDK imports or crypto execution."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_bootstrap_ingress.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_probe", PATH)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_all_byte_identities():
    assert probe.algebra()["exhaustive_byte_pairs"] == 65536
    assert probe.algebra()["independent_ring_outputs"] == 1152


@pytest.mark.parametrize("x,r,d", [(-128, 0, -1), (-1, 255, 0), (127, 255, 1), (0, 0, 0)])
def test_signed_wrap_endpoints(x, r, d):
    z, carry, sign, actual, biased, t = probe.lift(x, r)
    assert actual == d == carry - sign
    assert z - r + 256 * actual == x
    assert biased - r - 128 + 256 * t == x


@pytest.mark.parametrize("x,r", [(-129, 0), (128, 0), (0, -1), (0, 256)])
def test_outside_byte_contract_rejects(x, r):
    with pytest.raises(ValueError):
        probe.lift(x, r)


def test_complete_carry_and_lookup_leakage():
    result = probe.leakage()
    assert result["byte_only_max_tv"] == "0"
    assert result["byte_plus_Wd_max_tv"] == "7/8"
    assert result["shared_pad_table_distinct_patterns"] == 8
    assert result["prime_field_substitution"]["field257_residue"] != 32258


def test_ideal_ot_matches_clear_signed_products():
    assert probe.ot_checks()["outputs"] == 252
    assert probe.ot_checks()["optimized_biased_byte_outputs"] == 252


def test_independent_source_frame_enumeration():
    assert probe.request_frame_size(1, 896, 24, 24) == 2745
    assert probe.request_frame_size(1, 896, 8, 24) == 953
    assert probe.request_frame_size(150, 896, 24, 24) == 405644
    assert probe.request_frame_size(150, 896, 8, 24) == 136844


def test_exact_control_rows_cost_and_cold_warm_gap():
    result = probe.cost_screen()
    assert result["control_pair_bytes"] == 493575247
    assert result["arithmetic_subtotal"] == 368627357
    assert result["totals"]["byte_input_payload"] == 28732928
    assert result["totals"]["old_input_payload"] == 90104112
    assert result["input_frame_transport_overhead_retained"] == 339406
    assert sum(s["rows"] for s in result["stages"]) == 15297
    cold = result["horizons"][0]["helper_and_amortized_bootstrap_budget_per_pair"]
    assert cold["2/5"] == "15008871/5"
    assert cold["1/2"] == "-92711501/2"
    assert result["totals"]["selected_vector_ot_floor"] > 1000 * 493575247
    assert result["totals"]["selected_Wr_and_carry_ot_floor"] == 1242344364912
    assert result["horizons"][-1]["helper_and_amortized_bootstrap_budget_per_pair"]["1/2"][0] != "-"


def test_source_binding_and_no_claimed_runtime():
    report = probe.build_report()
    assert report["source_hashes"]["scripts/probe_bootstrap_ingress.py"] == probe.digest(PATH)
    assert report["protected_primitives_executed"] == 0
    assert report["client_peak_rss_bytes"] is None
    assert not report["measured_low_memory_claim"]
