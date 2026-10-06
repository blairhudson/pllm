"""Complete costs and numeric-domain rejection cannot become a decoder score."""
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_complete_block_matches_independent_oracle_and_prices_all_operations(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.nonlinear_block_reference import CONFIGURATION, oracle_digest

    report = json.loads((ROOT / "docs/evidence/research-nonlinear-block-reference-qwen25.json").read_text())
    assert report["schema"] == "pllm.nonlinear_block_reference.v1"
    assert report["configuration"] == CONFIGURATION
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).is_file()
        assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    indexed = {(c["backend"], c["layout"], c["lanes"]): c for c in report["cases"]}
    assert set(indexed) == set(itertools.product(
        CONFIGURATION["backends"], CONFIGURATION["layouts"], CONFIGURATION["lanes"],
    ))
    assert sum(c["checked_outputs"] for c in indexed.values()) == 11540
    for (_, layout, lanes), case in indexed.items():
        assert case["output_sha256"] == oracle_digest(lanes)
        assert case["rescale_lanes_per_output"] == 4
        assert case["multiplications_per_output"] == 2
        assert case["rounds"] == (5 if layout == "fused" else 8)
        # Four helper input vectors, two pairs of Beaver differences; headers
        # coalesce the independent initial gate/up rescalings into one round.
        expected = 64 * lanes + 5 * 156
        if layout == "unfused":
            expected += 32 * lanes + 3 * 156
        assert case["peer_frame_bytes"] == expected
        assert case["input_sharing_payload_bytes"] == 16 * lanes
        assert case["output_sharing_payload_bytes"] == 8 * lanes
        assert 0 < case["total_allocation_estimate_bytes"] <= 64 << 20
        assert case["party_key_payload_bytes"] > 24 * lanes  # both triples plus FSS
        assert not case["executable_sdk"]
    for projection in report["geometry_projections"]:
        assert not projection["executable_decoder"]
        for case in projection["cases"]:
            sample = indexed[case["backend"], case["layout"], case["batch_lanes"]]
            count = projection["gated_elements"]
            assert case["both_party_key_payload_bytes"] == 2 * count * sample["party_key_payload_bytes"] // sample["lanes"]
            assert case["client_input_output_share_payload_bytes_if_not_resident"] == 24 * count


def test_checkpoint_domain_gate_rejects_instead_of_clipping():
    report = json.loads((ROOT / "docs/evidence/research-nonlinear-block-reference-qwen25.json").read_text())
    domain = report["numeric_domain_diagnostic"]
    assert domain["reference"] == "pinned compiled W8A8 clear-kernel trajectory; not upstream FP32 or broad task quality"
    assert len(domain["public_prompt_sha256"]) == 64
    assert len(domain["traces"]) == 8
    assert len(domain["layers"]) == 24
    counts = domain["aggregate"]
    assert counts["either_outside_q7"] == 14778202
    assert counts["elements"] == 40974336
    assert counts["gate_outside_scaled16"] > 0
    assert counts["in_domain_bit_exact_outputs"] > 0
    for key in ("q7_domain_admitted", "scaled16_gate_domain_admitted", "candidate_decoder_executed", "protected_checkpoint_execution"):
        assert not domain[key]
    for key in ("executable_sdk", "privacy_reviewed", "whole_generation_quality"):
        assert not report[key]
    assert report["full_wire_bytes"] is None
