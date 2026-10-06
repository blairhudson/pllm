"""Paper subcohorts preserve observations and ordinary SDK comparison gates."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/evidence/research-paper-controls-qwen25.json"
VERIFIED = [f"benchmarks.research.{name}:experiment" for name in
            ("slalom_qwen", "slalom_qwen_composed")]


@pytest.fixture
def extract(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("benchmarks.research.paper_cohorts").extract


def test_verified_subcohort_keeps_original_observations(extract):
    original = json.loads(REPORT.read_text())
    assert not original["checks"]["passed"]  # Mixed verifier contracts cannot rank.
    result = extract(REPORT, VERIFIED)
    assert result["checks"]["passed"]
    names = result["analysis_of"]["candidates"]
    assert result["candidates"] == [c for c in original["candidates"] if c["name"] in names]
    assert len(names) == 2


def test_paper_subcohort_cannot_merge_verifier_contracts_or_duplicate_targets(extract):
    with pytest.raises(ValueError, match="comparison policy"):
        extract(REPORT, [VERIFIED[0], "benchmarks.research.paper_baseline:experiment"])
    with pytest.raises(ValueError, match="distinct"):
        extract(REPORT, [VERIFIED[0], VERIFIED[0]])


@pytest.mark.parametrize("change, message", [
    (lambda c: c.update(configuration_digest="f" * 64), "factory"),
    (lambda c: c["report"]["checks"].update(passed=False), "runtime checks"),
    (lambda c: c["report"]["configuration"].update(prompt_digest="other invocation"), "comparison policy"),
    (lambda c: c["report"]["runs"][0]["generation"].update(output_text_digest="different output"), "comparison policy"),
])
def test_paper_subcohort_rejects_unmatched_evidence(extract, tmp_path, change, message):
    original = json.loads(REPORT.read_text())
    change(next(c for c in original["candidates"] if c["name"] == "paper-qwen-slalom-composed"))
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(original))
    with pytest.raises(ValueError, match=message):
        extract(altered, VERIFIED)
