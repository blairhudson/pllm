"""Complete-block comparison conserves resources and preserves negative results."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_complete_fusion_cohort_and_geometry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.masked_wire_fusion import CONFIGURATION, projections, summarize

    report = json.loads((ROOT / "docs/evidence/research-masked-wire-fusion-sdk.json").read_text())
    assert report["configuration"] == CONFIGURATION
    assert len(report["cases"]) == 18
    assert report["checked_outputs"] == 5778
    assert report["summary"] == summarize(report["cases"])
    assert report["geometry_projections"] == projections(report["summary"])
    geometry = report["geometry_source"]
    assert hashlib.sha256((ROOT / geometry["path"]).read_bytes()).hexdigest() == geometry["sha256"]
    for case in report["cases"]:
        measured = case["report"]
        resources = measured["resources"]
        assert measured["quality"]["exact_integer_outputs"] == 3 * case["lanes"]
        assert resources["protocol_allocation_estimate_bytes"] < 64 * 2**20
        for sample in measured["samples"]:
            assert [sum(phase[direction] for phase in sample["peer_frame_bytes_by_phase"])
                    for direction in (0, 1)] == resources["peer_frame_bytes_by_direction"]
    for lanes in CONFIGURATION["lanes"]:
        a, b = [r for r in report["summary"] if r["lanes"] == lanes]
        assert a["numeric_contract_digest"] == b["numeric_contract_digest"]
        assert (a["resources"]["peer_rounds"], b["resources"]["peer_rounds"]) == (11, 8)
        assert sum(b["resources"]["peer_frame_bytes_by_direction"]) < sum(a["resources"]["peer_frame_bytes_by_direction"])
        assert sum(b["resources"]["raw_key_payload_bytes_per_party"]) > sum(a["resources"]["raw_key_payload_bytes_per_party"])
    assert all(layout["whole_decoder_execution"] is False and layout["full_wire_bytes"] is None
               for row in report["geometry_projections"] for layout in row["layouts"])

    bad = copy.deepcopy(report["cases"])
    bad[0]["report"]["numeric_contract_digest"] = "0" * 64
    with pytest.raises(ValueError, match="numeric contract"):
        summarize(bad)
    with pytest.raises(ValueError, match="incomplete"):
        summarize(report["cases"][:-1])
