"""Independent numeric/provenance checks; component evidence never ranks as Qwen."""
import hashlib
import itertools
import json
import math
from pathlib import Path

import pytest

from pllm.components import NotYetImplementedError
from pllm.correlation import SigmaDealerFssKeys
from pllm.nonlinear import FuseFssNonlinearOperator, SigmaSharedNonlinear
from pllm.passes import FuseFssPredicateFusion
from pllm.protocols import SigmaFssDecoder


ROOT = Path(__file__).resolve().parents[1]


def test_retained_helper_pair_has_exact_independent_outputs_and_scope(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.shared_rescale_reference import CONFIGURATION, oracle_digest

    report = json.loads((ROOT / "docs/evidence/research-shared-rescale-reference-qwen25.json").read_text())
    assert report["configuration"] == CONFIGURATION
    replay = "benchmarks/research/shared_rescale_reference.py"
    assert report["source_sha256"][replay] == hashlib.sha256((ROOT / replay).read_bytes()).hexdigest()
    # Native source digests are historical provenance, not a reason to rewrite
    # old measured evidence whenever implementation evolves.
    assert not report["executable_sdk"]
    assert not report["privacy_reviewed"]
    assert not report["whole_decoder_numeric_parity"]
    library = json.loads((ROOT / "docs/data/research/paper-library.json").read_text())["papers"]
    for paper, source in CONFIGURATION["papers"].items():
        entry = next(p for p in library if p["id"] == paper)
        assert (entry["source_url"], entry["sha256"]) == (source["url"], source["sha256"])
    expected = set(itertools.product((tuple(x) for x in CONFIGURATION["shapes"]), CONFIGURATION["rounding"], CONFIGURATION["layouts"]))
    assert {((case["bits"], case["shift"]), case["rounding"], case["layout"]) for case in report["cases"]} == expected
    assert sum(case["checked_outputs"] for case in report["cases"]) == 3840
    for case in report["cases"]:
        assert case["output_sha256"] == oracle_digest(case["bits"], case["shift"], case["rounding"], case["lanes"])
        assert case["checked_outputs"] == case["lanes"] * case["samples"]
        assert case["rounds"] == (1 if case["layout"] == "fused" else 2)
        assert case["peer_frame_bytes"] == 2 * case["rounds"] * (78 + 4 * case["lanes"])
        assert case["total_allocation_estimate_bytes"] <= 64 << 20
        assert case["process_cpu_seconds"] > 0 and case["process_peak_rss_bytes"] > 0
        assert all(math.isfinite(value) and value > 0 for value in case["median_seconds"].values())
        assert not case["executable_sdk"]
    for projection in report["geometry_projections"]:
        count = projection["silu_elements"]
        assert projection["hypothetical_helper_evaluations"] == count["prefill"] + (projection["output_tokens"] - 1) * count["decode"]
        assert projection["semantic_silu_operations_per_phase"] == 24
        for case in projection["cases"]:
            measured = next(c for c in report["cases"] if all(c[k] == case[k] for k in ("bits", "shift", "rounding", "layout")))
            assert case["both_party_key_payload_bytes"] == 2 * measured["party_key_payload_bytes"] * projection["hypothetical_helper_evaluations"] // measured["lanes"]
            assert not case["executable_decoder"]


@pytest.mark.parametrize("component", [SigmaFssDecoder, SigmaDealerFssKeys, SigmaSharedNonlinear, FuseFssNonlinearOperator, FuseFssPredicateFusion])
def test_helper_reference_does_not_activate_planned_components(component):
    with pytest.raises(NotYetImplementedError):
        component()
