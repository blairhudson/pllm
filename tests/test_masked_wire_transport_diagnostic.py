"""Audit retained diagnostic scope, byte conservation and the independent oracle."""
import json
from pathlib import Path

import numpy as np
import pytest

def test_transport_diagnostic_retains_exact_rows_and_conserved_costs():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "docs/evidence/research-masked-wire-transport-qwen25.json").read_text())
    assert report["status"] == "completed"
    assert report["source"]["width"] == 4864
    assert report["source"]["gated_regions"] == 24
    assert len(report["cases"]) == 6
    assert len({c["native_resources"]["profile_digest"] for c in report["cases"]}) == 1
    assert sum(s["exact_outputs"] for c in report["cases"] for s in c["samples"]) == 87552
    for layout, requests, keys, bodies in (("unfused", 456, 6279880, 7151942),
                                         ("fused", 342, 11097672, 11951836)):
        cases = [c for c in report["cases"] if c["layout"] == layout]
        assert {c["transport"] for c in cases} == {"in-process", "http-loopback", "native-100/40"}
        for case in cases:
            assert len(case["samples"]) == 3
            for sample in case["samples"]:
                stats = sample["stats"]
                assert stats["chunks"] == 19 and stats["elements"] == 4864
                assert stats["party_key_bytes"] == keys
                assert sum(a["requests"] for a in sample["actions"].values()) == requests
                assert sum(a["upload_bytes"] + a["download_bytes"] for a in sample["actions"].values()) == bodies
                assert sample["application_body_bytes"] == bodies
                if case["transport"] != "in-process":
                    assert case["memory_admission"]["admitted"]
                    assert case["maximum_swap_growth_bytes"] == 0
                    assert sample["aggregate_process_cpu_seconds"] == pytest.approx(
                        sample["process_cpu_seconds"] + sum(sample["worker_cpu_seconds"]))
                if case["transport"] == "native-100/40":
                    assert sum(sample["directed_tcp_stream_bytes"].values()) >= bodies


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

    def fail(samples, *, record):
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
