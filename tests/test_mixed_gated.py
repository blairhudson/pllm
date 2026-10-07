"""Fixed-profile parity, independent boundary oracles and complete mixed owners."""
import json
import struct

import pytest

from pllm import _native
from pllm.components import create_component
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9


@pytest.mark.parametrize("component", [
    CurlWaveletGatedMultiplyQ9(), CurlWaveletGatedMultiplyQ9(levels=4), CurlWaveletGatedMultiplyQ9(levels=8),
    LogRowGatedMultiplyQ9(), CompactGatedMultiplyQ12(max_pieces=4, degree=2),
    CompactGatedMultiplyQ12(max_pieces=16, degree=3),
])
def test_default_and_mixed_share_numeric_profile_but_not_execution_contract(component):
    mixed = component.with_params(backend="mixed")
    assert mixed != component
    assert create_component(mixed.component, mixed.params) == mixed
    assert type(mixed).from_params(mixed.get_params()) == mixed
    old = benchmark_operator(component, lanes=5, samples=1)
    new = benchmark_operator(mixed, lanes=5, samples=1)
    again = replay_operator_benchmark(new.configuration)
    assert new.quality == old.quality == again.quality
    assert new.numeric_contract_digest == old.numeric_contract_digest
    assert new.configuration_digest != old.configuration_digest
    assert new.resources == again.resources
    r = new.resources
    assert r.role_names == ("garbler_worker", "evaluator_worker")
    assert r.peer_rounds == len(new.samples[0].peer_frame_bytes_by_phase)
    assert r.peer_frame_bytes_by_direction == tuple(sum(row[i] for row in new.samples[0].peer_frame_bytes_by_phase) for i in (0, 1))
    assert 0 < r.peer_payload_bits < r.peer_frame_bytes * 8
    assert tuple(sum(row[i] for row in r.key_payload_bytes_by_operation) for i in (1, 2)) == r.raw_key_payload_bytes_per_party
    assert r.protocol_allocation_estimate_bytes <= 64 << 20
    assert not new.to_dict()["whole_model_executable"]
    assert not new.to_dict()["privacy_reviewed"]
    assert new.to_dict()["full_wire_bytes"] is None


@pytest.mark.parametrize("degree", [2, 3])
def test_nonuniform_compact_private_piece_edges_and_rounding_ties(degree):
    from pllm.metrics.compact_product import _profile
    counts = [0] * 1024
    counts[451], counts[883] = 100_000, 500_000
    component = CompactGatedMultiplyQ12(max_pieces=16, degree=degree,
        public_calibration_counts=counts, backend="mixed")
    _, profile = _profile(component)
    gates = [-48 << 18, 48 << 18]
    for edge in sorted({p[key] for p in profile["pieces"] for key in ("lower", "upper")}):
        gates.extend((edge << 6) + delta for delta in (-33, -32, -31, 0, 31, 32, 33))
    ups = [(-128 << 18, 128 << 18, -97, -96, -95, 95, 96, 97)[i % 8] for i in range(len(gates))]
    report = benchmark_operator(component, lanes=len(gates), samples=1,
        inputs={"gate_q18": gates, "up_q18": ups})
    assert report.quality.exact_integer_outputs == len(gates)
    assert report.input_source.startswith("caller-owned")
    assert '"gate_q18"' not in report.to_json()


@pytest.mark.parametrize("kind", [CompactGatedMultiplyQ12, CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9])
def test_invalid_backend_inputs_reject_before_material(kind, monkeypatch):
    monkeypatch.setattr(_native, "mixed_gated_benchmark", lambda *a: pytest.fail("unexpected issuance"))
    for backend in (False, 3, {}, "bad"):
        with pytest.raises(ConfigurationError):
            kind(backend=backend)
    for gate, up in (([True], [0]), ([0.0], [0]), ([(48 << 18) + 1], [0]), ([0], [-(128 << 18) - 1])):
        with pytest.raises(ValueError):
            benchmark_operator(kind(backend="mixed"), lanes=1, inputs={"gate_q18": gate, "up_q18": up})


def test_native_admission_rejects_oversize_profile_and_frames():
    config = dict(method="logrow", levels=0, max_pieces=0, degree=0, counts=[])
    word = bytes(4)
    for changed in ({"extra": 0}, {"counts": [1]}, {"levels": 1}, {"method": "foreign"}, {"degree": True}):
        with pytest.raises(ValueError):
            _native.mixed_gated_benchmark(json.dumps(config | changed), word, word, bytes(32), bytes(32))
    for g, u, plan, session in ((b"", b"", bytes(32), bytes(32)), (word * 257, word * 257, bytes(32), bytes(32)),
            (word, word, bytes(31), bytes(32)), (word, word, bytes(32), bytes(33)),
            (struct.pack("<i", (48 << 18) + 1), word, bytes(32), bytes(32))):
        with pytest.raises(ValueError):
            _native.mixed_gated_benchmark(json.dumps(config), g, u, plan, session)


def test_default_serialized_identities_preserved():
    assert LogRowGatedMultiplyQ9().params == {}
    assert CurlWaveletGatedMultiplyQ9().params == {"levels": 0}
    assert "backend" not in CompactGatedMultiplyQ12().params
