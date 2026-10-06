"""Activation-body lower bounds cannot become complete-protocol measurements."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_material_scopes_numeric_changes_and_missing_costs_remain_explicit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.nonlinear_frontier import CONFIGURATION

    report = json.loads((ROOT / "docs/evidence/research-nonlinear-frontier-qwen25.json").read_text())
    assert report["configuration"] == CONFIGURATION
    assert report["complete_block_cost"] is None
    assert not report["executable_sdk"]
    assert any("multiplication" in item for item in report["missing_costs"])
    assert any("conversion" in item for item in report["missing_costs"])
    domain = report["domain_evidence"]
    assert hashlib.sha256((ROOT / domain["path"]).read_bytes()).hexdigest() == domain["sha256"]
    for source, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    indexed = {case["method"]: case for case in report["cases"]}
    assert set(indexed) == set(CONFIGURATION["methods"])
    digests = {bytes(case["profile_digest"]) for case in indexed.values()}
    assert len(digests) == 1 and len(next(iter(digests))) == 32
    assert len({case["material_scope"] for case in indexed.values()}) == 3
    for method, case in indexed.items():
        assert not case["input_conversion_included"] and not case["gated_product_included"]
        samples, summary = case["samples"], case["summary"]
        assert len(samples) == CONFIGURATION["samples"]
        assert [row["public_input_q7"] for row in samples] == list(range(-128, 129, 16))
        errors = [abs(row["actual_q7"] - row["expected_q7"]) for row in samples]
        assert summary["encoded_matches"] == errors.count(0)
        assert summary["worst_encoded_error"] == max(errors)
        assert (summary["peer_frame_bytes_per_element"] is None) == (not method.startswith("curl"))
        if method in {"compact", "logrow", "curl0"}:
            assert not any(errors)
        else:
            assert any(errors)
    for geometry in report["geometry_projections"]:
        for projection in geometry["cases"]:
            sample = indexed[projection["method"]]["summary"]
            assert projection["activation_material_body_bytes_lower_bound"] == geometry["gated_elements"] * sample["material_body_bytes_per_element"]
            if sample["peer_frame_bytes_per_element"] is None:
                assert projection["activation_peer_frame_bytes_lower_bound"] is None
