"""Clear-only layout and finite-state functional checks; no crypto issuance."""
import importlib.util
import json
import math
from pathlib import Path

import pytest


path = Path(__file__).parents[1] / "scripts" / "probe_encrypted_residency.py"
spec = importlib.util.spec_from_file_location("encrypted_residency_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ClearArithmetic:
    """Slot oracle emulating only layout, never crypto or its error/cost claims."""
    def __init__(self, slots):
        self.encoder = self
        self.slots = slots
        self.rotations = []

    def slot_count(self):
        return self.slots

    def rotate(self, x, steps):
        self.rotations.append(steps)
        return x[steps:] + x[:steps]

    def multiply_plain(self, x, y):
        return [a * b for a, b in zip(x, y, strict=True)]

    multiply = multiply_plain

    def add(self, x, y):
        return [a + b for a, b in zip(x, y, strict=True)]


@pytest.mark.parametrize("lanes", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_resident_interleave_matches_independent_dense_recurrence(lanes, steps):
    weights, head, inputs = probe.fixture(lanes)
    expected = probe.oracle(weights, head, inputs, steps)
    arithmetic = ClearArithmetic(256)
    output = probe.resident_boundary(arithmetic, probe.pack(inputs, 256),
        probe.diagonals(weights), probe.diagonals(head), steps=steps, lanes=lanes)
    for slot, actual in enumerate(output):
        row, lane = (slot % (probe.WIDTH * lanes)) // lanes, slot % lanes
        target = expected[lane][row] if row < probe.ROWS else 0.0
        assert math.isclose(actual, target, rel_tol=0, abs_tol=1e-14)
    assert arithmetic.rotations == [lanes] * ((steps + 1) * (probe.WIDTH - 1))


@pytest.mark.parametrize("steps,lanes", [(0, 8), (4, 8), (True, 8), (2, 3), (2, 32), (2, True)])
def test_schedule_rejects_unbounded_or_malformed_geometry(steps, lanes):
    with pytest.raises(ValueError):
        probe.schedule(steps, lanes)


def test_period_and_matrix_shape_rejection():
    with pytest.raises(ValueError):
        probe.pack(probe.fixture(8)[2], 65)
    with pytest.raises(ValueError):
        probe.diagonals([[1.0]])
    with pytest.raises(ValueError):
        probe.diagonals([[float("nan")] * 8])


def test_zero_state_and_signed_lanes_do_not_cross_contaminate():
    weights, head, inputs = probe.fixture(8)
    inputs[3] = [0.0] * 8
    a = ClearArithmetic(256)
    result = probe.resident_boundary(a, probe.pack(inputs, 256), probe.diagonals(weights),
        probe.diagonals(head), steps=2, lanes=8)
    assert all(result[row * 8 + 3] == 0.0 for row in range(8))


def test_schedule_depth_charges_every_transition_and_final_head():
    assert probe.schedule(2, 8)["depth"] == 5
    assert probe.schedule(3, 16)["depth"] == 7


def test_resource_gate_charges_setup_and_fixed_final_ciphertext():
    gate = probe.resource_gate(probe.schedule(2, 8), {
        "degree": 16384, "public_key_frame_bytes": 1839104,
        "switching_key_frame_bytes": 11038720, "ciphertext_frame_bytes": 1576960})
    assert gate["setup_delivery_bytes_each_mode"] == 23925248
    assert gate["packed_online_frame_bytes"] == 1843362
    assert gate["serial_online_frame_bytes"] == 14746896
    assert gate["online_ratio"] == 8
    assert 1.49 < gate["setup_inclusive_ratio"] < 1.51
    assert gate["physical_wire_bytes"] is None


@pytest.mark.parametrize("platform,expected", [("darwin", 123), ("linux", 125952), ("freebsd", 125952)])
def test_peak_rss_platform_units(platform, expected):
    assert probe.peak_rss_bytes(123, platform) == expected


def test_source_snapshot_detects_changes_and_keeps_module_identity(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("x = 1\n")
    second.write_text("x = 1\n")
    paths = {"probe": first, "runtime": second}
    frozen = probe.source_snapshot(paths)
    assert frozen["probe"]["path"] != frozen["runtime"]["path"]
    assert frozen["probe"]["sha256"] == frozen["runtime"]["sha256"]
    probe.assert_sources_unchanged(frozen, probe.source_snapshot(paths))
    second.write_text("x = 2\n")
    with pytest.raises(AssertionError, match="sources changed"):
        probe.assert_sources_unchanged(frozen, probe.source_snapshot(paths))


def test_source_snapshot_detects_imported_path_changes(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("same\n")
    second.write_text("same\n")
    with pytest.raises(AssertionError, match="sources changed"):
        probe.assert_sources_unchanged(probe.source_snapshot({"runtime": first}),
                                       probe.source_snapshot({"runtime": second}))


def test_serial_evidence_frame_depth_numeric_and_identity_conservation():
    evidence = path.parents[1] / "docs" / "evidence" / "research-encrypted-residency-cost.json"
    if not evidence.exists():
        pytest.skip("serial crypto cohort not executed yet")
    report = json.loads(evidence.read_text())
    assert len(report["samples"]) == 3
    assert report["maximum_swap_growth_bytes"] == 0
    assert report["source_identity_verified_unchanged_after_workers"] is True
    checked, padding, logical = 0, 0, 0
    for pair in report["samples"]:
        for mode, rotations, queries in (("serial", [1], 8), ("packed", [8], 1)):
            sample = pair[mode]
            quality = sample["quality"]
            gate = probe.resource_gate(sample["schedule"], sample["parameters"])
            assert sample["source_identity"] == report["source_identity_frozen_before_workers"]
            assert sample["public_execution"]["rotation_steps"] == rotations
            assert sample["public_execution"]["ciphertext_queries"] == queries
            assert sample["public_execution"]["client_key_contexts"] == 1
            assert sample["public_execution"]["cross_client_multikey_batching"] is False
            assert len(sample["frames"]) == queries
            assert sample["setup_bytes"] == sum(sample["setup_frames"].values()) == gate["setup_delivery_bytes_each_mode"]
            assert sample["online_bytes"] == sum(sum(f.values()) for f in sample["frames"]) == gate[f"{mode}_online_frame_bytes"]
            assert sample["total_delivery_bytes"] == sample["setup_bytes"] + sample["online_bytes"]
            assert quality["maximum_absolute_error"] < quality["tolerance"] == probe.TOLERANCE
            assert quality["maximum_zero_padding_error"] < quality["tolerance"]
            assert quality["exact_integer_outputs"] == 0
            assert sample["remaining_depth"] == 0
            assert sample["parameters"]["depth"] == 5
            assert sample["operations"]["rescales"] == 26 * queries
            assert sample["operations"]["relinearizations"] == 2 * queries
            assert all(v is None for v in sample["whole_response_metrics"].values())
            checked += quality["checked_output_slots"]
            padding += quality["checked_zero_padding_slots"]
            logical += quality["unique_output_values"]
    assert (checked, padding, logical) == (55296, 165888, 96)
