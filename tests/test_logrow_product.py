"""Complete garbled block: numeric identity, ownership, replay and boundaries."""
import json
import hashlib
from pathlib import Path
import struct

import pytest

from pllm import _native
from pllm.configuration import ConfigurationError
from pllm.metrics import benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import CurlWaveletGatedMultiplyQ9, LogRowGatedMultiplyQ9


def test_replay_complete_accounting_and_same_numeric_profile_as_curl():
    report = benchmark_operator(LogRowGatedMultiplyQ9(), lanes=19, samples=2)
    replay = replay_operator_benchmark(report.configuration)
    curl = benchmark_operator(CurlWaveletGatedMultiplyQ9(), lanes=19, samples=2)
    assert report.resources == replay.resources
    assert report.quality == replay.quality == curl.quality
    assert report.numeric_contract_digest == curl.numeric_contract_digest
    r = report.resources
    assert r.role_names == ("garbler_client", "evaluator")
    assert r.peer_rounds == r.peer_frame_bytes == 0
    assert r.client_entry_exit_payload_estimate_bytes == 19 * ((25 + 27 + 23) * 16 + 64)
    assert r.protocol_allocation_estimate_bytes == 16 << 20
    parts = r.key_payload_bytes_by_operation
    assert tuple(sum(row[i] for row in parts) for i in (1, 2)) == r.raw_key_payload_bytes_per_party
    assert dict((name, b) for name, _, b in parts)["logrow_lookup"] == 19 * 11632
    assert sum(b for name, _, b in parts if name in {"up_rescale", "multiply", "output_rescale_sign"}) == r.evaluator_ciphertext_bytes
    assert report.to_dict()["full_wire_bytes"] is None
    assert not report.to_dict()["whole_model_executable"]
    assert not report.to_dict()["independent_role_transport"]
    assert "gate_q18" not in report.to_json()


def test_invalid_inputs_reject_before_any_garbling(monkeypatch):
    monkeypatch.setattr(_native, "logrow_product_benchmark", lambda *args: pytest.fail("unexpected issuance"))
    for inputs in ({}, {"gate_q18": [0]}, {"gate_q18": [0], "up_q18": None}):
        with pytest.raises(ConfigurationError):
            benchmark_operator(LogRowGatedMultiplyQ9(), lanes=1, inputs=inputs)
    for gate, up in (([True], [0]), ([0.0], [0]), ([(48 << 18) + 1], [0]), ([0], [-(128 << 18) - 1]), ([], [0])):
        with pytest.raises(ValueError):
            benchmark_operator(LogRowGatedMultiplyQ9(), lanes=1, inputs={"gate_q18": gate, "up_q18": up})


def test_native_frame_and_range_validation():
    word = struct.pack("<i", 0)
    for gate, up in ((b"", word), (word, word + b"\0"), (word * 257, word * 257),
                     (struct.pack("<i", (48 << 18) + 1), word), (word, struct.pack("<i", -(128 << 18) - 1))):
        with pytest.raises(ValueError):
            _native.logrow_product_benchmark(gate, up)


def test_clear_numeric_boundary_matches_garbled_labels_at_every_index():
    # Exhaust all 3,073 admitted table indices with alternating extreme up,
    # negative ties and nonintegral Q18 values. No client intermediate decode.
    from pllm.metrics.wavelet_gated import _oracle, _profile
    table, _ = _profile(0)
    values = list(range(-48 * 32, 48 * 32 + 1))
    for start in range(0, len(values), 256):
        gate = [v * 8192 for v in values[start:start + 256]]
        up = [[-128 << 18, -768, -256, 256, 768, 128 << 18][i % 6] for i in range(len(gate))]
        raw, metadata = _native.logrow_product_benchmark(
            struct.pack(f"<{len(gate)}i", *gate), struct.pack(f"<{len(up)}i", *up))
        assert list(struct.unpack(f"<{len(gate)}i", raw)) == [_oracle(table, 0, g, u) for g, u in zip(gate, up)]
        assert json.loads(metadata)["lanes"] == len(gate)


def test_retained_cost_cohort_conserves_roles_and_existing_quality_scope():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "docs/evidence/research-logrow-product-cost.json").read_text())
    assert raw["fresh_processes"] == 6
    assert raw["checked_outputs"] == 1926
    numeric = {row["numeric_contract_digest"] for row in raw["reports"]}
    assert numeric == {raw["quality_reference"]["numeric_contract_digest"]}
    reference = raw["quality_reference"]
    assert hashlib.sha256((root / reference["path"]).read_bytes()).hexdigest() == reference["sha256"]
    assert reference["teacher_forced"]["screen_pass"] is False
    assert reference["greedy_generation"]["exact_trajectories"] == 10
    for projection in raw["projections"]:
        assert projection["full_wire_bytes"] is None
        assert not projection["executable_decoder"]
        assert [sum(row[i] for row in projection["keys_by_operation_and_role"]) for i in (1, 2)] == projection["retained_key_bytes_by_role"]
    logrow = raw["projections"][1]
    assert logrow["role_names"] == ["garbler_client", "evaluator"]
    assert logrow["peer_frame_bytes"] == 0
    assert logrow["client_boundary_payload_estimate_bytes"] > 0
    assert logrow["retained_key_bytes_by_role"][1] > raw["projections"][0]["retained_key_bytes_by_role"][0] * 2
