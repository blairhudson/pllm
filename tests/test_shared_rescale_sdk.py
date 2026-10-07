"""Signed boundaries, matching controls, installed lifecycle and resource accounting."""

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct

import pytest

from pllm import _native
from pllm.metrics import OperatorBenchmarkReport, benchmark_operator, replay_operator_benchmark
from pllm.nonlinear import SharedRescale


@pytest.mark.parametrize("bits,shift", [(2, 1), (8, 3), (16, 7), (32, 1), (32, 31)])
@pytest.mark.parametrize("rounding", ["floor", "ties_even"])
def test_all_backends_and_layouts_preserve_signed_contract(bits, shift, rounding):
    bound, divisor = 1 << (bits - 1), 1 << shift
    values = sorted(
        {
            x
            for x in (
                -bound,
                -bound + 1,
                -divisor - 1,
                -divisor // 2,
                -1,
                0,
                1,
                divisor // 2,
                divisor // 2 + 1,
                bound - 1,
            )
            if -bound <= x < bound
        }
    )
    reports = [
        benchmark_operator(
            SharedRescale(bits=bits, shift=shift, rounding=rounding, layout=layout, backend=backend),
            lanes=len(values), samples=1, inputs={"values": values},
        )
        for backend in ("prefix_dpf", "compact_dcf", "interval_dcf")
        for layout in ("unfused", "fused")
    ]
    assert len({r.numeric_contract_digest for r in reports}) == 1
    assert len({r.configuration_digest for r in reports}) == 6
    assert all(r.quality == reports[0].quality for r in reports)
    assert all(
        r.quality.checked_outputs == r.quality.exact_integer_outputs == 2 * len(values)
        for r in reports
    )
    assert all(r.resources.peer_rounds == (2 if i % 2 == 0 else 1) for i, r in enumerate(reports))
    assert all(
        r.quality.maximum_absolute_error <= (0.5 if rounding == "ties_even" else 1) for r in reports
    )


def test_replay_coverage_and_historical_material_contract():
    component = SharedRescale(rounding="ties_even")
    report = benchmark_operator(component, samples=2)
    assert isinstance(report, OperatorBenchmarkReport)
    assert report.quality.exact_integer_outputs == 256
    assert report.resources.raw_key_payload_bytes_per_party == (758 * 64,) * 2
    assert report.resources.peer_frame_bytes_by_direction == (334, 334)
    assert report.resources.peer_payload_bits == 4096
    assert report.resources.public_table_bytes_per_party == 0
    assert report.resources.client_mask_payload_estimate_bytes == 256
    assert report.resources.client_entry_exit_payload_estimate_bytes == 1536
    assert report.resources.protocol_allocation_estimate_bytes < 64 << 20
    unfused = benchmark_operator(component.with_params(layout="unfused"), samples=2)
    assert unfused.resources.raw_key_payload_bytes_per_party == (620 * 64,) * 2
    assert unfused.resources.peer_frame_bytes == 1336
    assert unfused.numeric_contract_digest == report.numeric_contract_digest
    assert unfused.quality == report.quality
    replay = replay_operator_benchmark(report.configuration)
    assert replay.configuration_digest == report.configuration_digest
    assert replay.native_implementation_digest == report.native_implementation_digest
    assert replay.sdk_implementation_digest == report.sdk_implementation_digest
    assert replay.resources == report.resources
    assert replay.quality == report.quality
    for sample in report.samples:
        assert sample.issuance_seconds > 0 and sample.online_seconds > 0
        assert sum(sum(row) for row in sample.peer_frame_bytes_by_phase) == 668
    serialized = json.loads(report.to_json())
    assert serialized["whole_model_executable"] is False
    assert serialized["sdk_availability"] == "operator-benchmark"
    assert serialized["full_wire_bytes"] is serialized["peak_memory_bytes"] is None
    assert "input_digest" not in serialized
    assert "values" not in serialized
    with pytest.raises(FrozenInstanceError):
        report.quality.checked_outputs = 0
    with pytest.raises(FrozenInstanceError):
        component.params = {}
    assert (
        benchmark_operator(component.with_params(rounding="floor"), samples=2).numeric_contract_digest
        != report.numeric_contract_digest
    )


@pytest.mark.parametrize(
    "change",
    [
        {"bits": 1},
        {"bits": 33},
        {"bits": True},
        {"shift": 0},
        {"shift": 16},
        {"shift": False},
        {"backend": "boolean_dpf"},
        {"rounding": "nearest"},
        {"layout": []},
    ],
)
def test_reject_unbounded_or_ambiguous_configuration(change):
    with pytest.raises(ValueError):
        SharedRescale(**change)


@pytest.mark.parametrize("values", [[], [0, 1], [True], [1.0], [32768], [-32769], "x"])
def test_reject_inputs_before_issuance(monkeypatch, values):
    def forbidden(*_):
        pytest.fail("native issuance reached on invalid input")

    monkeypatch.setattr(_native, "shared_rescale_benchmark", forbidden)
    with pytest.raises(ValueError):
        benchmark_operator(SharedRescale(), lanes=1, inputs={"values": values})


def test_boundary_maximum_lanes_and_fresh_sample_sessions(monkeypatch):
    run = _native.shared_rescale_benchmark
    sessions = []

    def capture(values, settings, config, session):
        sessions.append(session)
        return run(values, settings, config, session)

    monkeypatch.setattr(_native, "shared_rescale_benchmark", capture)
    report = benchmark_operator(SharedRescale(bits=32, shift=12), lanes=256, samples=3)
    assert report.quality.exact_integer_outputs == 1536
    assert len(set(sessions)) == 3
    settings = (16, 7, "floor", "fused", "interval_dcf")
    good = struct.pack("<q", 0)
    for data, cfg, session in (
        (b"", b"c" * 32, b"s" * 32),
        (b"x", b"c" * 32, b"s" * 32),
        (good * 257, b"c" * 32, b"s" * 32),
        (struct.pack("<q", 32768), b"c" * 32, b"s" * 32),
        (struct.pack("<q", -32769), b"c" * 32, b"s" * 32),
        (good, b"c", b"s" * 32),
        (good, b"c" * 32, b"s"),
    ):
        with pytest.raises(ValueError):
            run(data, settings, cfg, session)
    for bad in ((1, 0, "floor", "fused", "interval_dcf"), (16, 7, "floor", "fused", "bad")):
        with pytest.raises(ValueError):
            run(good, bad, b"c" * 32, b"s" * 32)


def test_corrupt_oracle_and_frames_fail_closed(monkeypatch):
    run = _native.shared_rescale_benchmark

    def corrupt(*args):
        _, metadata = run(*args)
        return struct.pack("<II", 123456, 0), metadata

    monkeypatch.setattr(_native, "shared_rescale_benchmark", corrupt)
    with pytest.raises(RuntimeError, match="integer oracle"):
        benchmark_operator(SharedRescale(), lanes=1, samples=1)

    def corrupt_frames(*args):
        output, metadata = run(*args)
        value = json.loads(metadata)
        value["phase_frame_bytes"][0][1] -= 1
        return output, json.dumps(value)

    monkeypatch.setattr(_native, "shared_rescale_benchmark", corrupt_frames)
    with pytest.raises(RuntimeError, match="phase accounting"):
        benchmark_operator(SharedRescale(), lanes=1, samples=1)


def test_citations_bind_actual_pinned_papers_and_replay_rejects_edits():
    root = Path(__file__).resolve().parents[1]
    library = json.loads((root / "docs/data/research/paper-library.json").read_text())
    papers = {p["id"]: p for p in library["papers"]}
    report = benchmark_operator(SharedRescale(), lanes=1, samples=1)
    for citation in report.citations:
        paper = papers[citation.paper_id]
        assert citation.source_sha256 == paper["sha256"]
        assert citation.title == paper["title"]
        assert citation.url == paper["source_url"]
        pdf = root / paper["file"]
        if pdf.exists():
            assert hashlib.sha256(pdf.read_bytes()).hexdigest() == citation.source_sha256
    for change in (
        {"schema": "wrong"},
        {"numeric_profile": "wrong"},
        {"extra": 1},
        {"lanes": False},
    ):
        with pytest.raises(ValueError):
            replay_operator_benchmark(report.configuration | change)


def test_retained_sdk_collection_preserves_rounding_groups_and_replays():
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / "docs/evidence/research-shared-rescale-sdk.json").read_text())
    assert document["schema"] == "pllm.operator_benchmark_collection.v1"
    reports = document["reports"]
    assert len(reports) == 12
    assert len({r["configuration_digest"] for r in reports}) == 12
    assert sum(r["quality"]["exact_integer_outputs"] for r in reports) == 4608
    for rounding in ("floor", "ties_even"):
        group = [r for r in reports if r["configuration"]["rounding"] == rounding]
        assert len(group) == 6
        assert len({r["numeric_contract_digest"] for r in group}) == 1
        assert all(r["quality"] == group[0]["quality"] for r in group)
    assert len({r["numeric_contract_digest"] for r in reports}) == 2
    for report in reports:
        configuration = report["configuration"]
        assert hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() == report["configuration_digest"]
        assert report["benchmark_status"] == "completed"
        assert report["whole_model_executable"] is False
        assert report["resources_scope"] == "per-sample"
        assert report["full_wire_bytes"] is report["peak_memory_bytes"] is None
        if (
            configuration["backend"] == "interval_dcf"
            and configuration["rounding"] == "floor"
            and configuration["layout"] == "fused"
        ):
            replay = replay_operator_benchmark(configuration).to_dict()
            assert replay["configuration_digest"] != report["configuration_digest"]
            assert replay["configuration"]["component"] == SharedRescale().to_spec()
            for key in (
                "native_implementation_digest",
                "numeric_contract_digest",
                "resources",
                "quality",
            ):
                actual = replay[key]
                if key in {"resources", "quality"}:
                    # New optional accounting fields do not rewrite archived
                    # measurements or turn unknown historical costs into zero.
                    actual = {field: actual[field] for field in report[key]}
                assert actual == report[key]
            assert replay["resources"]["role_names"] == ["party_0", "party_1"]
            for field in (
                "public_metadata_bytes_estimate", "public_profile_bytes_estimate",
                "evaluator_ciphertext_bytes", "evaluator_constant_label_bytes",
            ):
                assert replay["resources"][field] is None
            assert replay["quality"]["base_profile_exact_outputs"] is None
            assert replay["quality"]["maximum_encoded_profile_error"] is None
