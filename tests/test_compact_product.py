"""Independent complete arithmetic, source/profile commitments and SDK replay."""
import importlib.util
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from pllm import _native
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.metrics.compact_product import _profile, _oracle
from pllm.nonlinear import CompactGatedMultiplyQ12


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "benchmarks/research/compact_product.py"
    spec = importlib.util.spec_from_file_location("compact_product_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pieces,degree", [(1, 2), (8, 3), (16, 3)])
def test_replay_private_boundaries_material_ownership_and_numeric_identity(pieces, degree):
    component = CompactGatedMultiplyQ12(max_pieces=pieces, degree=degree)
    report = benchmark_operator(component, lanes=11, samples=1)
    replay = replay_operator_benchmark(report.configuration)
    assert report.resources == replay.resources
    assert report.quality == replay.quality
    assert report.quality.checked_outputs == report.quality.exact_integer_outputs == 11
    r = report.resources
    assert r.role_names == ("garbler_client", "evaluator")
    assert r.peer_frame_bytes == r.peer_rounds == 0
    assert r.client_entry_exit_payload_estimate_bytes == 11 * ((25 + 27 + 26) * 16 + 64)
    assert r.protocol_allocation_estimate_bytes == 64 << 20
    assert tuple(sum(row[i] for row in r.key_payload_bytes_by_operation) for i in (1, 2)) == r.raw_key_payload_bytes_per_party
    assert r.evaluator_ciphertext_bytes + r.evaluator_constant_label_bytes == r.raw_key_payload_bytes_per_party[1]
    assert report.to_dict()["full_wire_bytes"] is None
    assert not report.to_dict()["whole_model_executable"]
    assert not report.to_dict()["independent_role_transport"]
    assert "gate_q18" not in report.to_json()


@pytest.mark.parametrize("degree", [2, 3])
def test_private_piece_edges_and_signed_half_ties_match_independent_integer_oracle(degree):
    counts = [0] * 1024
    counts[577] = 100_000
    component = CompactGatedMultiplyQ12(max_pieces=16, degree=degree, public_calibration_counts=counts)
    native, profile = _profile(component)
    values = [-48 << 18, 48 << 18, -33, -32, -31, 0, 31, 32, 33]
    for piece in profile["pieces"]:
        for edge in (piece["lower"], piece["upper"]):
            values.extend((edge << 6) + delta for delta in (-33, -32, -31, 0, 31, 32, 33))
    up = [(-128 << 18, 128 << 18, -97, -96, -95, 95, 96, 97)[i % 8] for i in range(len(values))]
    assert len(values) <= 256
    actual, metadata = native.benchmark(struct.pack(f"<{len(values)}i", *values), struct.pack(f"<{len(up)}i", *up))
    assert list(struct.unpack(f"<{len(values)}i", actual)) == [_oracle(profile, g, u) for g, u in zip(values, up)]
    assert json.loads(metadata)["wires_per_lane"] * 384 + (2 << 20) < 64 << 20
    uniform, uniform_profile = _profile(CompactGatedMultiplyQ12(max_pieces=16, degree=degree))
    assert profile["profile_digest"] != uniform_profile["profile_digest"]


def test_input_and_calibration_admission_precedes_native_material(monkeypatch):
    monkeypatch.setattr(_native, "CompactProductReference", lambda *args: pytest.fail("unexpected garbling"))
    for kwargs in ({"max_pieces": 17}, {"max_pieces": True}, {"degree": 1}, {"degree": 2.0},
                   {"public_calibration_counts": [0]}, {"public_calibration_counts": [True] * 1024},
                   {"public_calibration_counts": [1_000_000] * 1024}):
        with pytest.raises(ConfigurationError):
            CompactGatedMultiplyQ12(**kwargs)
    for gate, up in (([True], [0]), ([0.0], [0]), ([(48 << 18) + 1], [0]), ([0], [-(128 << 18) - 1]), ([], [0])):
        with pytest.raises(ValueError):
            benchmark_operator(CompactGatedMultiplyQ12(), lanes=1, inputs={"gate_q18": gate, "up_q18": up})


def test_native_malformed_frames_and_resource_limits():
    native, _ = _profile(CompactGatedMultiplyQ12())
    word = bytes(4)
    for gate, up in ((b"", word), (word, word + b"\0"), (word * 257, word * 257),
                     (struct.pack("<i", (48 << 18) + 1), word)):
        with pytest.raises(ValueError):
            native.benchmark(gate, up)
    for counts in (bytes(4095), struct.pack("<1024I", *([1_000_000] * 1024))):
        with pytest.raises(ValueError):
            _native.CompactProductReference(counts, 8, 3)


@pytest.mark.parametrize("pieces,degree", [(4, 2), (8, 3), (16, 3)])
def test_clear_checkpoint_mapping_exhausts_q12_coordinates_and_float_boundaries(pieces, degree):
    bridge = benchmark_module().NativeMapping(pieces, degree)
    knots = np.arange(-32768, 32769, dtype=np.float32) / 4096
    gate = np.concatenate([knots, np.nextafter(knots, np.float32(-np.inf)),
                           np.nextafter(knots, np.float32(np.inf))]).reshape(1, 1, -1)
    up = np.linspace(-128, 128, gate.size, dtype=np.float32).reshape(gate.shape)
    assert bridge.evaluate(gate, up).dtype == np.float32
    assert bridge.checked == gate.size
    for invalid in (np.float32(48.01), np.float32(-48.01), np.float32(np.nan)):
        with pytest.raises(ValueError, match="profile domain rejection"):
            bridge.evaluate(np.full((1, 1, 1), invalid, np.float32), np.zeros((1, 1, 1), np.float32))


def test_retained_evidence_binds_frozen_profiles_complete_costs_and_quality_scope():
    root = Path(__file__).resolve().parents[1]
    quality = json.loads((root / "docs/evidence/research-compact-product-quality-qwen25.json").read_text())
    cost = json.loads((root / "docs/evidence/research-compact-product-cost.json").read_text())
    for record in (quality, cost):
        for path, expected in record["source_sha256"].items():
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected
    assert not quality["protected_checkpoint_execution"]
    control = quality["teacher_forced"]["summary"]["control"]
    assert control["bit_exact_positions"] == control["exact_kv_positions"] == 48
    assert quality["greedy_generation"]["summary"]["control"]["exact_trajectories"] == 12
    assert cost["fresh_processes"] == 9
    assert cost["checked_outputs"] == 2889
    for row in cost["reports"]:
        params = row["configuration"]["component"]["params"]
        name = f"pieces-{params['max_pieces']}-degree-{params['degree']}"
        assert row["numeric_contract_digest"] == quality["numeric_contract_digests"][name]
        numeric = quality["teacher_forced"]["summary"][name]
        assert numeric["rejected_prompts"] == 0
        assert numeric["bit_exact_positions"] == numeric["exact_kv_positions"] == 0
        assert not numeric["screen_pass"]
        assert row["quality"]["checked_outputs"] == row["quality"]["exact_integer_outputs"]
    assert len({r["numeric_contract_digest"] for r in cost["reports"]}) == 3
    for projection in cost["projections"]:
        assert projection["full_wire_bytes"] is None
        assert not projection["executable_decoder"]
        assert projection["role_names"] == ["garbler_client", "evaluator"]
        assert projection["peer_frame_bytes"] == 0
        assert projection["client_boundary_payload_estimate_bytes"] > 0
        assert [sum(row[i] for row in projection["keys_by_operation_and_role"]) for i in (1, 2)] == projection["retained_key_bytes_by_role"]
