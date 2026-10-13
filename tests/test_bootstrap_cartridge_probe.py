"""Focused stdlib algebra/cost checks; no model, native timing, or downloads."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bootstrap_probe", Path(__file__).resolve().parents[1] / "scripts/probe_bootstrap_cartridge.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("stride", [1, 4, 16, 64, 1024])
def test_checkpoint_all_columns_and_union_reuse(stride):
    code = probe.Code(7, bytes(32))
    weights = [[-128, 127, 0, 1, -1, 33, -99], [3] * 7]
    cartridge = probe.CheckpointCartridge(weights, code, stride)
    columns, counts = cartridge.selected(range(code.length))
    for j, col in columns.items():
        assert col == [code.encode(row)[j] for row in weights]
    assert cartridge.selected([2, 2, 4])[1]["selected_columns"] == 2
    assert counts["first_mul_add_lanes"] + counts["second_mul_add_lanes"] <= 2 * code.length * (stride**2 + stride)
    assert len(cartridge.boundaries) == (code.length - 1) // stride


def test_exact_mask_from_independent_generator_and_full_i8_boundaries():
    code = probe.Code(17, bytes([7]) * 32)
    sparse = [(0, 1), (19, probe.FIELD - 1), (135, 99)]
    mask = code.syndrome(sparse)
    basis = [code.encode([int(i == j) for i in range(17)]) for j in range(17)]
    assert mask == [sum(row[j] * v for j, v in sparse) % probe.FIELD for row in basis]
    weights = [[-128] * 17, [127] * 17]
    for x in ([-128] * 17, [127] * 17):
        masked = [(a + b) % probe.FIELD for a, b in zip(x, mask)]
        for row, exact in zip(weights, probe.exact_oracle(weights, x)):
            p = code.encode(row)
            correction = sum(p[j] * v for j, v in sparse) % probe.FIELD
            assert (probe.dot(row, masked) - correction) % probe.FIELD == exact % probe.FIELD


def test_public_seed_is_not_fresh_noise():
    code = probe.Code(64, bytes([5]) * 32)
    assert code.encode(list(range(64))) == probe.Code(64, bytes([5]) * 32).encode(list(range(64)))
    a, b = probe.fresh_sparse(code, 160), probe.fresh_sparse(code, 160)
    assert len({j for j, _ in a}) == 160
    assert all(0 < v < probe.FIELD for _, v in a)
    assert a != b
    assert code.syndrome(a) != code.syndrome(b)


@pytest.mark.parametrize("sparse", [[], [(0, 0)], [(0, probe.FIELD)], [(8, 1)], [(1, 2), (1, 3)]])
def test_bad_sparse_rejected(sparse):
    with pytest.raises(ValueError):
        probe.Code(1, bytes(32)).syndrome(sparse)


def test_warm_control_denominator_and_impossible_targets():
    # (R*40-100)/(100+R*100) >= .2 iff R>=6, not >=3.
    assert probe.horizon(100, 100, 100, 40, 0, .2) == 6
    assert probe.horizon(100, 100, 100, 40, 0, .5) is None
    assert probe.horizon(100, 100, 0, 40, 20, .2) is None


def test_native_bounds_not_relaxed_by_p_only_accounting():
    square = probe.geometry(896, 896, 64)
    gate = probe.geometry(9728, 896, 64)
    down = probe.geometry(896, 4864, 64)
    assert square["reference_pq_payload_passes"]
    assert not gate["reference_pq_payload_passes"]
    assert not down["reference_pq_payload_passes"]
    assert gate["p_only_stage_payload_exceeds_low"]
    assert probe.MAX_ATTEMPTS == 65536 and probe.MAX_LIVE == 8


def test_symbolic_projection_and_exact_horizon_ceiling():
    report = probe.symbolic_report()
    assert report["remote_stages"] == 96
    assert report["public_weight_bytes"] == 357826560
    assert report["warm_recurring_bytes"] == 368670156
    assert report["arithmetic_subtotal_bytes"] == 368627357
    for candidate in report["candidates"]:
        for case in candidate["network_sensitivities"]:
            assert case["horizons_pairs"]["50"] is None
        assert candidate["network_sensitivities"][0]["horizons_pairs"]["35"] is not None
        for case in candidate["network_sensitivities"][1:]:
            assert case["horizons_pairs"]["35"] is None
        for case in candidate["network_sensitivities"][2:]:
            assert case["horizons_pairs"]["20"] is None


def test_report_refuses_replacement(tmp_path, monkeypatch):
    path = tmp_path / "evidence.json"
    path.write_text("do not replace")
    monkeypatch.setattr("sys.argv", ["probe", "--output", str(path)])
    with pytest.raises(SystemExit):
        probe.main()
    assert path.read_text() == "do not replace"
