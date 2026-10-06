"""Universal-key cost evidence retains numeric, workload and one-use boundaries."""
import hashlib
import itertools
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_interval_helper_evidence_and_complete_key_accounting(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.interval_rescale_reference import CONFIGURATION
    from benchmarks.research.shared_rescale_reference import oracle_digest

    report = json.loads((ROOT / "docs/evidence/research-interval-rescale-reference-qwen25.json").read_text())
    assert report["schema"] == "pllm.interval_rescale_reference.v1"
    assert report["configuration"] == CONFIGURATION
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).is_file() and len(digest) == 64
        if source.endswith(".py"):
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    assert "crates/pllm-garble/src/interval_fss.rs" in report["source_sha256"]
    assert not any(report[key] for key in ("executable_sdk", "privacy_reviewed", "whole_decoder_numeric_parity"))
    papers = {p["id"]: p for p in json.loads((ROOT / "docs/data/research/paper-library.json").read_text())["papers"]}
    for paper, source in CONFIGURATION["papers"].items():
        assert (source["url"], source["sha256"]) == (papers[paper]["source_url"], papers[paper]["sha256"])

    axes = ("bits", "shift", "rounding", "layout", "backend")
    indexed = {tuple(c[key] for key in axes): c for c in report["cases"]}
    expected = {(bits, shift, rounding, layout, backend) for (bits, shift), rounding, layout, backend in itertools.product(
        CONFIGURATION["shapes"], CONFIGURATION["rounding"], CONFIGURATION["layouts"], CONFIGURATION["backends"],
    )}
    assert set(indexed) == expected and len(report["cases"]) == len(indexed) == 24
    assert sum(case["checked_outputs"] for case in report["cases"]) == 7680
    for case in report["cases"]:
        bits, shift = case["bits"], case["shift"]
        assert case["output_sha256"] == oracle_digest(bits, shift, case["rounding"], case["lanes"])
        assert case["checked_outputs"] == case["lanes"] * case["samples"] == 320
        if case["layout"] == "fused":
            phases = [[bits, shift, bits]]
        else:
            phases = [[shift], [bits - shift, bits - shift]]
        if case["rounding"] == "ties_even":
            phases[0] += [shift + 1] * 4
        assert case["comparison_evaluations_per_lane"] == sum(map(len, phases))
        if case["backend"] == "interval_dcf":
            # Deduplicate only within a phase. A later opening owns fresh material.
            widths = [width for phase in phases for width in set(phase)]
            control = indexed[(bits, shift, case["rounding"], case["layout"], "compact_dcf")]
            assert case["party_key_payload_bytes"] < control["party_key_payload_bytes"]
            assert case["peer_frame_bytes"] == control["peer_frame_bytes"]
            assert case["output_sha256"] == control["output_sha256"]
        else:
            widths = [width for phase in phases for width in phase]
        assert sorted(case["comparison_widths"]) == sorted(widths)
        # Root, every correction, one mask and three constant shares per phase.
        assert case["party_key_payload_bytes"] == case["lanes"] * (sum(20 + 22 * w for w in widths) + 16 * len(phases))
        assert case["rounds"] == len(phases)
        assert case["peer_frame_bytes"] == 2 * len(phases) * (78 + 4 * case["lanes"])
        assert case["total_allocation_estimate_bytes"] <= 64 << 20
        assert case["process_cpu_seconds"] > 0 and case["process_peak_rss_bytes"] > 0
        assert all(math.isfinite(value) and value > 0 for value in case["median_seconds"].values())
        assert not case["executable_sdk"]

    for projection in report["geometry_projections"]:
        count = projection["silu_elements"]
        evaluations = projection["hypothetical_helper_evaluations"]
        assert evaluations == count["prefill"] + (projection["output_tokens"] - 1) * count["decode"]
        assert projection["semantic_silu_operations_per_phase"] == 24
        assert len(projection["cases"]) == len(indexed)
        for case in projection["cases"]:
            measured = indexed[tuple(case[key] for key in axes)]
            assert case["both_party_key_payload_bytes"] == 2 * measured["party_key_payload_bytes"] * evaluations // measured["lanes"]
            assert case["peer_frame_bytes_at_64_lane_batches"] == measured["rounds"] * (8 * evaluations + 2 * 78 * projection["hypothetical_batches"])
            assert not case["executable_decoder"]

    registry = json.loads((ROOT / "docs/data/research/implementations.json").read_text())["papers"]
    for paper in CONFIGURATION["papers"]:
        assert registry[paper]["status"] == "Component reference"
        assert registry[paper]["configuration"] == "benchmarks/research/interval_rescale_reference.py"
        assert registry[paper]["evidence"] == "docs/evidence/research-interval-rescale-reference-qwen25.json"
