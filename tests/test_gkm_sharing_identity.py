"""Full/seeded GKM keep numeric equality and distinct privacy assumptions."""

from pllm._search_benchmark import _boundary
from pllm.runtime.benchmark_cli import build_comparison_report

from test_delegated_benchmark import (
    experiment as experiment, native_contract as native_contract, report,
)


def test_seeded_boundary_changes_privacy_without_changing_numeric(experiment, native_contract):
    full = _boundary(experiment)
    seeded = _boundary(experiment.with_params(pipeline__linear__sharing="seeded"))
    assert full["numeric"] == seeded["numeric"]
    assert full["source"] == seeded["source"]
    assert "input_sharing" not in full["privacy"]
    assert seeded["privacy"] == full["privacy"] | {
        "input_sharing": "gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1",
    }


def test_ordinary_comparison_retains_costs_but_does_not_rank_changed_privacy(experiment, native_contract):
    seeded = experiment.with_params(name="gkm-seeded", pipeline__linear__sharing="seeded")
    result = build_comparison_report([(experiment, report(experiment)), (seeded, report(seeded))])
    assert result["checks"]["all_candidates_passed"]
    assert result["checks"]["matched_numeric_contract"]
    assert result["checks"]["matched_verification"]
    assert result["checks"]["matched_sharing_contract"] is False
    assert all(not ranking for ranking in result["rankings"].values())
    assert all(winner is None for winner in result["winners"].values())
    assert len(result["candidates"]) == 2
