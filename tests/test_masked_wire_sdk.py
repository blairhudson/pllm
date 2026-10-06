"""Installed boundary, independent integer oracle and complete frame accounting."""

from dataclasses import FrozenInstanceError
import json
import math
import struct

import numpy as np
import pytest

from pllm import _native
from pllm.metrics import MaskedWireGatedBlockProbe, OperatorBenchmarkReport


def test_sdk_public_fixture_accounting_and_replay():
    probe = MaskedWireGatedBlockProbe(lanes=64, samples=2)
    report = probe.run()
    assert isinstance(report, OperatorBenchmarkReport)
    assert report.quality.checked_outputs == report.quality.exact_integer_outputs == 128
    assert report.quality.maximum_absolute_error > 0  # correct approximation is not exact SiLU
    assert report.resources.raw_key_payload_bytes_per_party == (41_296, 41_296)
    assert report.resources.combined_raw_key_payload_bytes == 82_592
    assert report.resources.peer_frame_bytes_by_direction == (3010, 3010)
    assert report.resources.peer_frame_bytes == 6020
    assert report.resources.peer_rounds == 11
    assert report.resources.public_table_bytes_per_party == 8192
    assert report.resources.client_mask_payload_estimate_bytes == 960
    assert report.resources.client_entry_exit_payload_estimate_bytes == 1600
    assert report.resources.protocol_allocation_estimate_bytes < 64 << 20
    for sample in report.samples:
        assert sum(sum(row) for row in sample.peer_frame_bytes_by_phase) == 6020
        assert sample.issuance_seconds > 0 and sample.online_seconds > 0
        assert sample.native_call_process_cpu_seconds > 0
    value = json.loads(report.to_json())
    assert value["sdk_availability"] == "operator-benchmark"
    assert value["benchmark_status"] == "completed"
    assert value["resources_scope"] == "per-sample"
    assert value["paper_coverage"] == "component-reference"
    assert value["whole_model_executable"] is False
    assert value["full_wire_bytes"] is value["peak_memory_bytes"] is None
    assert {x["paper_id"] for x in value["citations"]} == {"sigma", "grotto", "pika"}
    replay = MaskedWireGatedBlockProbe.from_dict(value["configuration"])
    assert replay == probe and replay.digest == report.configuration_digest
    replay_report = replay.run()
    assert replay_report.resources == report.resources
    assert replay_report.quality == report.quality
    assert replay_report.numeric_contract_digest == report.numeric_contract_digest
    assert replay_report.native_implementation_digest == report.native_implementation_digest
    assert replay_report.sdk_implementation_digest == report.sdk_implementation_digest
    # Both records are immutable and callers cannot edit their configuration through an alias.
    with pytest.raises(FrozenInstanceError):
        probe.lanes = 1
    with pytest.raises(FrozenInstanceError):
        report.resources.peer_rounds = 0
    value["configuration"]["lanes"] = 1
    assert report.configuration["lanes"] == 64


def test_native_boundary_exact_floor_tails_and_maximum_batch():
    rng = np.random.default_rng(4281)
    gates = rng.integers(-(48 << 24), (48 << 24) + 1, size=256, dtype=np.int64)
    ups = rng.integers(-(128 << 24), (128 << 24) + 1, size=256, dtype=np.int64)
    gates[:12] = [
        -(48 << 24),
        -(16 << 24) - 1,
        -(16 << 24),
        -4097,
        -4096,
        -1,
        0,
        1,
        4095,
        4096,
        (16 << 24) - 1,
        48 << 24,
    ]
    ups[:12] = [128 << 24, -(128 << 24)] * 6
    raw, metadata = _native.masked_wire_benchmark(
        gates.astype("<i8").tobytes(),
        ups.astype("<i8").tobytes(),
        b"c" * 32,
        b"s" * 32,
    )
    table = [math.floor(4096 * (i / 64) / (1 + math.exp(i / 64))) for i in range(1024)]
    expected = []
    for gate, up in zip(gates, ups, strict=True):
        g, u = int(gate) // 4096, int(up) // 4096
        residual = table[min(abs(g // 64), 1023)]
        expected.append(((max(g, 0) - residual) * u) // 4096)
    assert list(struct.unpack("<256q", raw)) == expected
    record = json.loads(metadata)
    assert record["peer_frame_bytes"] == sum(sum(x) for x in record["phase_frame_bytes"])
    report = MaskedWireGatedBlockProbe(lanes=256, samples=1).run(gate_q24=gates, up_q24=ups)
    assert report.quality.exact_integer_outputs == 256
    assert report.input_source.startswith("caller-owned")
    assert "gate_q24" not in report.to_dict()
    assert "input_digest" not in report.to_json()


@pytest.mark.parametrize(
    "change",
    [
        {"lanes": 0},
        {"lanes": 257},
        {"lanes": True},
        {"samples": 0},
        {"samples": 17},
        {"samples": 1.5},
    ],
)
def test_config_rejects_unbounded_or_ambiguous_work(change):
    with pytest.raises(ValueError):
        MaskedWireGatedBlockProbe(**change)


@pytest.mark.parametrize(
    "inputs",
    [
        {"gate_q24": [0]},
        {"gate_q24": [0], "up_q24": []},
        {"gate_q24": [True], "up_q24": [0]},
        {"gate_q24": [0.0], "up_q24": [0]},
        {"gate_q24": [(48 << 24) + 1], "up_q24": [0]},
        {"gate_q24": [0], "up_q24": [-(128 << 24) - 1]},
        {"gate_q24": np.array([-(2**63)]), "up_q24": [0]},
    ],
)
def test_bad_inputs_reject_before_native_issuance(monkeypatch, inputs):
    def forbidden(*_):
        pytest.fail("native issuance reached on invalid input")

    monkeypatch.setattr(_native, "masked_wire_benchmark", forbidden)
    with pytest.raises(ValueError):
        MaskedWireGatedBlockProbe(lanes=1, samples=1).run(**inputs)


def test_native_boundary_rejects_malformed_and_outside_inputs():
    good = struct.pack("<q", 0)
    for g, u, config, session in (
        (b"", b"", b"c" * 32, b"s" * 32),
        (b"x", good, b"c" * 32, b"s" * 32),
        (good * 257, good * 257, b"c" * 32, b"s" * 32),
        (good, good, b"c", b"s" * 32),
        (good, good, b"c" * 32, b"s"),
        (struct.pack("<q", (48 << 24) + 1), good, b"c" * 32, b"s" * 32),
        (good, struct.pack("<q", -(128 << 24) - 1), b"c" * 32, b"s" * 32),
    ):
        with pytest.raises(ValueError):
            _native.masked_wire_benchmark(g, u, config, session)
    assert MaskedWireGatedBlockProbe(lanes=1, samples=1).run().quality.exact_integer_outputs == 1


def test_fail_closed_if_binding_disagrees_and_fresh_session_per_sample(monkeypatch):
    call = _native.masked_wire_benchmark
    sessions = []

    def capture(gate, up, config, session):
        sessions.append(session)
        return call(gate, up, config, session)

    monkeypatch.setattr(_native, "masked_wire_benchmark", capture)
    MaskedWireGatedBlockProbe(lanes=1, samples=3).run()
    assert len(set(sessions)) == 3

    def corrupt(*args):
        _, metadata = call(*args)
        return struct.pack("<q", 123456789), metadata

    monkeypatch.setattr(_native, "masked_wire_benchmark", corrupt)
    with pytest.raises(RuntimeError, match="integer oracle"):
        MaskedWireGatedBlockProbe(lanes=1, samples=1).run()


def test_replay_rejects_changed_or_extra_contract_fields():
    probe = MaskedWireGatedBlockProbe()
    for change in (
        {"numeric_profile": "ties-even"},
        {"schema": "new-schema"},
        {"unbounded": True},
        {"lanes": False},
    ):
        with pytest.raises(ValueError):
            probe.from_dict(probe.to_dict() | change)
