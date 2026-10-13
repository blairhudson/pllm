"""Independent bounded matrix-triple regression tests; no SDK imports."""

import copy
import importlib.util
import json
import random
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/probe_matrix_triple_prefill.py"
SPEC = importlib.util.spec_from_file_location("matrix_triple_probe", PATH)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def test_exact_oracle_count_and_domains():
    result = p.bounded_algebra()
    assert result["independent_output_values"] == 3024
    assert result["fresh_a_cases"] == 48
    assert set(result["fresh_a_output_values_by_modulus"].values()) == {750}
    assert not result["pcg_implemented"]


@pytest.mark.parametrize("q", [2**16, 2**24, 2**32, p.FIELD])
def test_random_to_fixed_role_equations(q):
    rng = random.Random(91)
    triple = p.ToyTriple(rng, 2, 2, 4, q, p.UseRegistry(), "unique")
    w, x = [[127, -128], [-1, 3]], [[-128, 127, 0, -1], [127, -128, 1, 42]]
    got, view = triple.consume(w, x)
    assert got == [[v % q for v in row] for row in p.clear_signed(w, x)]
    assert p.add(view["hc"], view["hi"], q) == p.multiply(w, triple.b, q)
    assert view["d"] == p.add(w, triple.a, q, -1)


def test_replay_clone_cancel_malformed_burn_and_rollback_limit():
    rng, registry = random.Random(8), p.UseRegistry()
    triple = p.ToyTriple(rng, 1, 1, 1, 257, registry, "use")
    cloned = copy.copy(triple)
    triple.consume([[1]], [[4]])
    with pytest.raises(ValueError, match="spent"):
        cloned.consume([[1]], [[5]])
    cancelled = p.ToyTriple(rng, 1, 1, 1, 257, registry, "cancel")
    cancelled.cancel()
    with pytest.raises(ValueError, match="spent"):
        cancelled.consume([[1]], [[5]])
    bad = p.ToyTriple(rng, 1, 1, 1, 257, registry, "bad")
    with pytest.raises(ValueError, match="shape"):
        bad.consume([[1, 2]], [[3]])
    with pytest.raises(ValueError, match="spent"):
        bad.consume([[1]], [[3]])
    # Independent deep copy has independent registry: demonstrate NOT durable ownership.
    fresh = p.ToyTriple(rng, 1, 1, 1, 257, p.UseRegistry(), "rollback")
    rolled_back = copy.deepcopy(fresh)
    _, v1 = fresh.consume([[1]], [[2]])
    _, v2 = rolled_back.consume([[1]], [[9]])
    assert (v2["e"][0][0] - v1["e"][0][0]) % 257 == 7


def test_seed_reuse_and_small_field_counterexamples():
    witness = p.leakage_witnesses()
    assert witness["reused_B_opening_difference"] == 59
    assert witness["public_A_plus_total_C_recovers_B"]
    assert witness["small_field257_integer"] == 32258
    assert witness["small_field257_residue"] == 133


def test_fresh_A_requires_fresh_dense_D_reusable_A_requires_fresh_B():
    rng = random.Random(7)
    a = p.sample(rng, 2, 2, 257)
    t1 = p.ToyTriple(rng, 2, 2, 1, 257, p.UseRegistry(), 1, a)
    t2 = p.ToyTriple(rng, 2, 2, 1, 257, p.UseRegistry(), 2, a)
    w = [[1, 3], [-2, 5]]
    _, v1 = t1.consume(w, [[4], [7]])
    _, v2 = t2.consume(w, [[4], [7]])
    assert v1["d"] == v2["d"]
    assert t1.b != t2.b
    t3 = p.ToyTriple(rng, 2, 2, 1, 257, p.UseRegistry(), 3)
    _, v3 = t3.consume(w, [[4], [7]])
    assert v3["d"] != v1["d"]


def test_public_A_single_share_view_does_not_reveal_B_exhaustive():
    # Independent exhaustive ideal-dealer scalar view. Fixed A,B_i,C_i;
    # every candidate B occurs once as other B share varies. Other C share adjusts.
    q, a, bi, ci = 7, 3, 4, 5
    candidates = [(bi + bc) % q for bc in range(q)]
    assert sorted(candidates) == list(range(q))
    for b in candidates:
        cj = (a * b - ci) % q
        assert (ci + cj) % q == a * b % q


def test_source_geometry_exact_rows_pruning_and_ledger():
    control, stages = p.geometry()
    assert control["pair_bytes"] - control["classes"]["public_bundle"] == 368670156
    assert sum(s["rows"] == 15 for s in stages) == 3
    assert sum(s["rows"] for s in stages) == 15297


@pytest.mark.parametrize("tile", p.TILES)
def test_complete_remainders_decode_underfill_and_negative_headroom(tile):
    _, stages = p.geometry()
    for field in (False, True):
        result = p.project(stages, tile, field)
        assert result["net_saved_before_setup_helpers"] < 0
        assert result["D_two_direction_bytes"] > 139060689
        assert result["extra_client_dense_crypto_macs"] == sum(s["body_macs"] for s in stages)
        assert result["remote_original_body_fraction_prefill"] == 1
        assert result["remote_original_body_fraction_decode"] == 1
        for row in result["stages"]:
            assert row["batches"] == (row["prefill_rows"] + tile - 1) // tile + 14
            assert row["decode_unused_columns_each"] == tile - 1
            assert row["complete_remainder"] == row["prefill_rows"] % tile


def test_field_bounded_lift_not_free_ring_homomorphism():
    assert p.FIELD > 2 * 4864 * 128 * 128
    # Odd-prime wrap is not ring2^k wrap: reducing shares separately is incorrect.
    left, right = p.FIELD - 2, 3
    assert (left + right) % p.FIELD == 1
    assert (left % 65536 + right % 65536) % 65536 != 1


def test_matched_lifetime_budget_exact():
    budgets = p.headroom(139060689)
    assert budgets[0]["matched_control"] == 493575247
    assert budgets[0]["additional_setup_and_helper_budget_per_pair"]["1/2"] == "-215453869/2"
    assert budgets[-1]["additional_setup_and_helper_budget_per_pair"]["1/5"] == "326633289/5"


@pytest.fixture
def saved_sources():
    # Mutable independent loads exercise validation directly without altering files.
    return [json.loads((p.ROOT / path).read_text()) for path in p.SOURCES]


def source_stages(sources, index):
    if index == 0:
        return sources[0]["control"]["stages"]
    if index == 1:
        return sources[1]["checkpoint_geometry"]["stages"]
    if index == 2:
        return sources[2]["cost"]["stages"]
    return sources[3]["stages"]


@pytest.mark.parametrize("source", [1, 2, 3])
def test_duplicate_stage_rejected_before_indexing(saved_sources, source):
    stages = source_stages(saved_sources, source)
    # Replace, rather than append: preserve length to defeat count-only checks.
    stages[-1] = copy.deepcopy(stages[0])
    with pytest.raises(AssertionError, match="duplicate stage"):
        p.validate_geometry(*saved_sources)


@pytest.mark.parametrize("source", [0, 1, 2, 3])
def test_missing_stage_rejected_in_each_source(saved_sources, source):
    stages = source_stages(saved_sources, source)
    if source == 0:
        del stages[next(iter(stages))]
    else:
        stages.pop()
    with pytest.raises(AssertionError, match="stage set mismatch"):
        p.validate_geometry(*saved_sources)


@pytest.mark.parametrize("source", [0, 1, 2, 3])
def test_changed_stage_name_rejected_despite_equal_counts(saved_sources, source):
    stages = source_stages(saved_sources, source)
    if source == 0:
        stages["invented.stage"] = stages.pop(next(iter(stages)))
    else:
        stages[0]["stage"] = "invented.stage"
    with pytest.raises(AssertionError, match="stage set mismatch"):
        p.validate_geometry(*saved_sources)


@pytest.mark.parametrize("source", [2, 3])
def test_ring_mismatch_rejected_even_when_supported(saved_sources, source):
    row = source_stages(saved_sources, source)[0]
    row["original_ring_bits" if source == 2 else "ring_bits"] = 32
    with pytest.raises(AssertionError, match="original ring mismatch"):
        p.validate_geometry(*saved_sources)


@pytest.mark.parametrize("source", [1, 2, 3])
def test_encoded_width_mismatch_against_source_lock(saved_sources, source):
    row = source_stages(saved_sources, source)[0]
    if source == 3:
        row["annihilator"]["uniform_input_bits"] += 1
    else:
        row["stage_bits"] += 1
    with pytest.raises(AssertionError, match="encoded width mismatch"):
        p.validate_geometry(*saved_sources)


@pytest.mark.parametrize("mutation", ["count", "width", "zero"])
def test_residue_histogram_mismatch_rejected(saved_sources, mutation):
    hist = source_stages(saved_sources, 1)[0]["row_residue_width_histogram"]
    if mutation == "count":
        hist["24"] += 1
    elif mutation == "width":
        hist["25"] = hist.pop("24")
    else:
        hist["24"] = 0
    with pytest.raises(AssertionError, match="width"):
        p.validate_geometry(*saved_sources)


def test_duplicate_control_json_keys_cannot_be_silently_normalized():
    raw = '{"control":{"stages":{"same.stage":{},"same.stage":{}}}}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads(raw, object_pairs_hook=p.unique_json_object)


@pytest.mark.parametrize("source", [1, 3])
def test_source_lock_mismatch_rejected(saved_sources, source):
    lock = (saved_sources[1]["checkpoint_geometry"]["lock"] if source == 1
            else saved_sources[3]["source"])
    lock["source_lock_digest"] = "changed"
    with pytest.raises(AssertionError, match="source lock mismatch"):
        p.validate_geometry(*saved_sources)
