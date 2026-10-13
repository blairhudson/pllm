from __future__ import annotations

from copy import deepcopy

import pytest

from benchmarks.research import low_memory_combinations as low
from pllm import Experiment, Model
from pllm.metrics import BuildRef


def test_six_installed_recipes_keep_numeric_and_lifecycle_identity():
    study = low.study(model=Model.tiny(), tokenizer=None, build=BuildRef(),
                      tiny=True, workloads=("tiny",), processes=3)
    work = study.plan(max_trials=18)
    assert len(work.suite.cases) == 6
    assert work.coverage.to_dict()["selected_trials"] == 18
    specs = [case.target.to_spec() for case in work.suite.cases]
    assert [s["parameters"]["model"]["experiment"]["pipeline"]["components"]["scheduler"]["params"]["tile_rows"]
            for s in specs] == [64, 32, 16, 64, 32, 16]
    assert len({case.target.digest for case in work.suite.cases}) == 6
    for spec in specs:
        experiment = Experiment.from_spec(spec["parameters"]["model"]["experiment"])
        assert Experiment.from_spec(experiment.to_spec()).canonical_bytes() == experiment.canonical_bytes()
        parts = experiment.pipeline.components
        assert parts["preparation"].params["kernel"] == "digit8"
        assert parts["preparation"].params["storage"] == "paged"
        assert parts["preparation"].params["min_rows"] == 64
        for key in ("quantization", "linear", "delivery", "cache", "inventory"):
            assert spec["parameters"]["model"]["experiment"]["pipeline"]["components"][key] == (
                specs[0]["parameters"]["model"]["experiment"]["pipeline"]["components"][key])
        assert spec["parameters"]["scenario"] == specs[0]["parameters"]["scenario"]
        assert spec["parameters"]["wan"] == specs[0]["parameters"]["wan"]


def cohort(peaks):
    return {"complete": True,
        "suite": {"sampling": {"processes": len(peaks)}, "cases": [{"name": "150"}]},
        "results": [{"status": "completed", "execution": {"case": {"name": "150"},
            "replicate": replicate, "details": {"report": {"client_process_memory": {
                "after": {"lifetime_peak_rss_bytes": peak}}}}}} for replicate, peak in enumerate(peaks)]}


@pytest.mark.parametrize(("peaks", "eligible", "status"), [
    ([200 << 20, 210 << 20, low.LOW_BYTES], True, "qualified"),
    ([200 << 20, 210 << 20, low.LOW_BYTES + 1], False, "over_budget"),
    ([200 << 20, None, 210 << 20], False, "unknown"),
])
def test_low_eligibility_uses_every_peak(peaks, eligible, status):
    report = low.memory_eligibility(cohort(peaks))
    row = report["cases"][0]
    assert report["max_bytes"] == 256 << 20
    assert row["eligible"] is eligible and row["status"] == status
    assert row["max_peak_bytes"] == (None if None in peaks else max(peaks))


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "extra", "invalid_peak"])
def test_incomplete_or_invalid_memory_evidence_cannot_qualify(change):
    document = cohort([200 << 20] * 3)
    if change == "missing":
        document["results"].pop()
    elif change == "duplicate":
        document["results"][1] = deepcopy(document["results"][0])
    elif change == "failed":
        document["results"][0]["status"] = "failed"
    elif change == "extra":
        document["results"][0]["execution"]["case"]["name"] = "other"
    else:
        document["results"][0]["execution"]["details"]["report"]["client_process_memory"]["after"]["lifetime_peak_rss_bytes"] = True
    with pytest.raises(low.base.Rejected):
        low.memory_eligibility(document)
