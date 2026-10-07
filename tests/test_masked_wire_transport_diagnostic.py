"""Audit retained diagnostic scope, byte conservation and the independent oracle."""
import json
from pathlib import Path

import numpy as np
import pytest

@pytest.mark.parametrize("filename,case_count,output_count", [
    ("research-masked-wire-transport-qwen25.json", 6, 87552),
    ("research-masked-wire-overlap-qwen25.json", 12, 175104),
    ("research-masked-wire-seeded-qwen25.json", 12, 175104),
])
def test_transport_diagnostic_retains_exact_rows_and_conserved_costs(filename, case_count, output_count):
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence" / filename).read_text())
    assert report["status"] == "completed"
    assert report["source"]["width"] == 4864
    assert report["source"]["gated_regions"] == 24
    assert len(report["cases"]) == case_count
    assert len({c["native_resources"]["profile_digest"] for c in report["cases"]}) == 1
    assert sum(s["exact_outputs"] for c in report["cases"] for s in c["samples"]) == output_count
    for layout, requests, keys, bodies in (("unfused", 456, 6279880, 7151942),
                                         ("fused", 342, 11097672, 11951836)):
        cases = [c for c in report["cases"] if c["layout"] == layout]
        assert {c["transport"] for c in cases} == {"in-process", "http-loopback", "native-100/40"}
        for case in cases:
            assert len(case["samples"]) == 3
            selected_keys, selected_bodies = keys, bodies
            if case.get("material_encoding") == "seeded":
                selected_keys = 19 * (2 * 76 + sum(case["native_resources"]["material_payload_bytes_per_party"]))
                selected_bodies -= keys - selected_keys
            for sample in case["samples"]:
                stats = sample["stats"]
                assert stats["chunks"] == 19 and stats["elements"] == 4864
                assert stats["party_key_bytes"] == selected_keys
                assert sum(a["requests"] for a in sample["actions"].values()) == requests
                assert sum(a["upload_bytes"] + a["download_bytes"] for a in sample["actions"].values()) == selected_bodies
                assert sample["application_body_bytes"] == selected_bodies
                if case["transport"] != "in-process":
                    assert case["memory_admission"]["admitted"]
                    assert case["maximum_swap_growth_bytes"] == 0
                    assert sample["aggregate_process_cpu_seconds"] == pytest.approx(
                        sample["process_cpu_seconds"] + sum(sample["worker_cpu_seconds"]))
                if case["transport"] == "native-100/40":
                    assert sum(sample["directed_tcp_stream_bytes"].values()) >= selected_bodies


def test_seeded_reduction_is_key_only_and_retains_expanded_memory():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence/research-masked-wire-seeded-qwen25.json").read_text())
    assert report["dispatches"] == ["overlap"]
    assert report["material_encodings"] == ["full", "seeded"]
    cases = {(c["layout"], c["material_encoding"], c["transport"]): c for c in report["cases"]}
    for layout in ("unfused", "fused"):
        for transport in ("in-process", "http-loopback", "native-100/40"):
            full, seeded = (cases[layout, encoding, transport] for encoding in report["material_encodings"])
            for key in ("protocol_allocation_estimate_bytes", "party_key_bits", "peer_frame_bytes"):
                assert full["native_resources"][key] == seeded["native_resources"][key]
            for a, b in zip(full["samples"], seeded["samples"], strict=True):
                saving = a["stats"]["party_key_bytes"] - b["stats"]["party_key_bytes"]
                assert saving > 0
                assert a["application_body_bytes"] - b["application_body_bytes"] == saving
                assert a["stats"]["peer_frame_bytes"] == b["stats"]["peer_frame_bytes"]
                if transport == "native-100/40":
                    assert sum(a["directed_tcp_stream_bytes"].values()) - sum(b["directed_tcp_stream_bytes"].values()) == saving
            if transport != "in-process":
                a, b = (dict(c["memory_admission"]["estimate"]) for c in (full, seeded))
                assert a.pop("pipeline_digest") != b.pop("pipeline_digest")
                assert a == b


def test_overlap_changes_timing_and_admission_without_discarding_traffic():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence/research-masked-wire-overlap-qwen25.json").read_text())
    assert report["dispatches"] == ["serial", "overlap"]
    assert "sum request durations" in report["timing_scope"]
    assert "issuing thread" in report["timing_scope"]
    cases = {(c["layout"], c["dispatch"], c["transport"]): c for c in report["cases"]}
    for layout in ("unfused", "fused"):
        for transport in ("in-process", "http-loopback", "native-100/40"):
            serial, overlap = (cases[layout, dispatch, transport] for dispatch in report["dispatches"])
            assert serial["native_resources"] == overlap["native_resources"]
            for first, second in zip(serial["samples"], overlap["samples"], strict=True):
                assert first["exact_outputs"] == second["exact_outputs"]
                assert first["application_body_bytes"] == second["application_body_bytes"]
                for key in ("party_key_bytes", "peer_frame_bytes", "chunks", "elements"):
                    assert first["stats"][key] == second["stats"][key]
                if transport == "native-100/40":
                    assert first["directed_tcp_stream_bytes"] == second["directed_tcp_stream_bytes"]
            if transport != "in-process":
                a, b = (c["memory_admission"]["estimate"]["components"] for c in (serial, overlap))
                assert b["client_nonlinear_work_bytes"] == 2 * a["client_nonlinear_work_bytes"]


def test_diagnostic_fixture_agrees_with_native_mapping_at_signed_and_tail_boundaries(monkeypatch):
    from pllm import _native
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.research.masked_wire_transport import fixture

    gate, up, expected = fixture(16)
    g = np.floor(gate.astype(np.float64) * (1 << 24)).astype("<i8")
    u = np.floor(up.astype(np.float64) * (1 << 24)).astype("<i8")
    result = np.frombuffer(_native._masked_wire_clear(g.tobytes(), u.tobytes()), dtype="<i8")
    np.testing.assert_array_equal(result.astype(np.float32).reshape(expected.shape) / 4096, expected)


def test_diagnostic_keeps_finished_cases_when_next_case_fails(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from benchmarks.research import masked_wire_transport

    output = tmp_path / "diagnostic.json"
    completed = {"layout": "unfused", "transport": "http-loopback", "samples": [{"exact_outputs": 16}]}

    def fail(samples, *, record, compare_dispatch, compare_material):
        document = {"status": "running", "cases": [completed], "active_case": None}
        record(document)
        assert json.loads(output.read_text())["cases"] == [completed]
        document["active_case"] = {"layout": "fused", "transport": "http-loopback"}
        record(document)
        raise RuntimeError("private exception detail")

    monkeypatch.setattr(masked_wire_transport, "run", fail)
    monkeypatch.setattr("sys.argv", ["diagnostic", "--samples", "1", "--output", str(output)])
    with pytest.raises(RuntimeError):
        masked_wire_transport.main()
    saved = json.loads(output.read_text())
    assert saved["status"] == "failed" and saved["cases"] == [completed]
    assert saved["active_case"]["layout"] == "fused"
    assert saved["failure"] == {"type": "RuntimeError"}
    assert "private exception detail" not in output.read_text()
    with pytest.raises(FileExistsError):
        masked_wire_transport.main()
    assert json.loads(output.read_text()) == saved
