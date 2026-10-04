import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/evidence/wan-tps-matched-qwen25-2026-10-04.json"
spec = importlib.util.spec_from_file_location("wan_tps_summary", ROOT / "scripts/summarize_wan_tps.py")
assert spec is not None and spec.loader is not None
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_retained_cohort_recomputes_exactly_and_keeps_body_scope():
    observed = summary.summarize([REPORT])
    retained = json.loads((ROOT / "docs/evidence/wan-tps-qwen25-2026-10-04.json").read_text())
    assert observed == retained
    runs = {row["name"]: row for row in observed["runs"]}
    for control, candidate in (("prepared-control", "prepared-combined"), ("offset-control", "seed-first")):
        assert runs[control]["online_body_bytes"] == runs[candidate]["online_body_bytes"]
        assert runs[candidate]["end_to_end_tokens_per_second"] > runs[control]["end_to_end_tokens_per_second"]
    assert all(row["peak_client_memory_bytes"] is None for row in observed["runs"])


def test_separate_salted_invocations_do_not_become_a_matched_cohort():
    with pytest.raises(ValueError, match="multi-Experiment"):
        summary.summarize([ROOT / "docs/evidence/wan-tps-prepared-control-2026-10-04.json"])


@pytest.mark.parametrize("field", ["prompt", "source", "output", "network"])
def test_changed_cohort_or_kernel_evidence_rejects(tmp_path, field):
    document = json.loads(REPORT.read_text())
    report = document["candidates"][1]["report"]
    if field == "prompt":
        report["configuration"]["prompt_digest"] = "1" * 64
    elif field == "source":
        report["configuration"]["source_lock_digest"] = "1" * 64
    elif field == "output":
        report["runs"][0]["generation"]["output_text_digest"] = "1" * 64
    else:
        report["configuration"]["link_conditions_digest"] = "1" * 64
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        summary.summarize([path])
