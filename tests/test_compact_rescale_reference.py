"""Matched backend evidence, independent integer oracles and material accounting."""
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_compact_helpers_preserve_numeric_scope_and_complete_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.compact_rescale_reference import CONFIGURATION
    from benchmarks.research.shared_rescale_reference import oracle_digest

    report = json.loads((ROOT / "docs/evidence/research-compact-rescale-reference-qwen25.json").read_text())
    assert report["schema"] == "pllm.compact_rescale_reference.v1"
    assert report["configuration"] == CONFIGURATION
    for source in ("benchmarks/research/compact_rescale_reference.py", "benchmarks/research/shared_rescale_reference.py"):
        assert report["source_sha256"][source] == hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
    assert len(report["source_sha256"]) == 9
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    assert not any(report[key] for key in ("executable_sdk", "privacy_reviewed", "whole_decoder_numeric_parity"))
    papers = {p["id"]: p for p in json.loads((ROOT / "docs/data/research/paper-library.json").read_text())["papers"]}
    for paper, source in CONFIGURATION["papers"].items():
        assert (source["url"], source["sha256"]) == (papers[paper]["source_url"], papers[paper]["sha256"])

    keys = ("bits", "shift", "rounding", "layout", "backend")
    indexed = {tuple(c[k] for k in keys): c for c in report["cases"]}
    expected = {(bits, shift, rounding, layout, backend) for (bits, shift), rounding, layout, backend in itertools.product(
        CONFIGURATION["shapes"], CONFIGURATION["rounding"], CONFIGURATION["layouts"], CONFIGURATION["backends"],
    )}
    assert set(indexed) == expected
    assert len(indexed) == len(report["cases"]) == 24
    assert sum(c["checked_outputs"] for c in report["cases"]) == 7680
    for case in report["cases"]:
        bits, shift = case["bits"], case["shift"]
        assert case["output_sha256"] == oracle_digest(bits, shift, case["rounding"], case["lanes"])
        assert case["checked_outputs"] == case["lanes"] * case["samples"] == 320
        widths = [shift, *([bits] * 2 if case["layout"] == "fused" else [bits - shift] * 2)]
        if case["rounding"] == "ties_even":
            widths += [shift + 1] * 4
        assert sorted(case["comparison_widths"]) == sorted(widths)
        rounds = 1 if case["layout"] == "fused" else 2
        assert case["rounds"] == rounds
        assert case["peer_frame_bytes"] == 2 * rounds * (78 + 4 * case["lanes"])
        if case["backend"] == "compact_dcf":
            compare_bytes = sum(20 + 22 * width for width in widths)
            prefix = indexed[(bits, shift, case["rounding"], case["layout"], "prefix_dpf")]
            assert case["party_key_payload_bytes"] < prefix["party_key_payload_bytes"]
            assert case["output_sha256"] == prefix["output_sha256"]
            assert case["peer_frame_bytes"] == prefix["peer_frame_bytes"]
        else:
            compare_bytes = sum(20 * width + 9 * width * (width + 1) for width in widths)
        # Every phase owns a mask share and three constant shares per lane.
        assert case["party_key_payload_bytes"] == case["lanes"] * (compare_bytes + 16 * rounds)
        assert case["total_allocation_estimate_bytes"] <= 64 << 20
        assert case["process_cpu_seconds"] > 0 and case["process_peak_rss_bytes"] > 0
        assert all(math.isfinite(x) and x > 0 for x in case["median_seconds"].values())
        assert not case["executable_sdk"]

    for projection in report["geometry_projections"]:
        count = projection["silu_elements"]
        evaluations = projection["hypothetical_helper_evaluations"]
        assert evaluations == count["prefill"] + (projection["output_tokens"] - 1) * count["decode"]
        assert projection["semantic_silu_operations_per_phase"] == 24
        assert len(projection["cases"]) == len(indexed)
        for case in projection["cases"]:
            measured = indexed[tuple(case[k] for k in keys)]
            assert case["both_party_key_payload_bytes"] == 2 * measured["party_key_payload_bytes"] * evaluations // measured["lanes"]
            assert case["peer_frame_bytes_at_64_lane_batches"] == measured["rounds"] * (8 * evaluations + 2 * 78 * projection["hypothetical_batches"])
            assert not case["executable_decoder"]

    registry = json.loads((ROOT / "docs/data/research/implementations.json").read_text())["papers"]
    for paper in CONFIGURATION["papers"]:
        assert registry[paper]["status"] == "Component reference"
        assert registry[paper]["evidence"] == "docs/evidence/research-compact-rescale-reference-qwen25.json"
