"""Retained payload accounting and quality evidence keep distinct scopes."""
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_fresh_cohort_conserves_state_and_preserves_full_retention_control(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.retained_decoder import CONFIGURATION, MODES, PUBLIC_PROMPTS, canonical, summarize

    report = json.loads((ROOT / "docs/evidence/research-retained-decoder-qwen25.json").read_bytes())
    assert report["schema"] == "pllm.retained_decoder_reference.v1"
    assert report["configuration"] == CONFIGURATION
    assert report["public_prompt_sha256"] == hashlib.sha256(canonical(PUBLIC_PROMPTS)).hexdigest()
    for path, digest in report["source_sha256"].items():
        assert (ROOT / path).is_file() and len(digest) == 64
        if path.startswith(("benchmarks/", "scripts/")):
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    assert report["sdk_client_peak_rss_bytes"] is None
    assert report["provider_transport_bytes"] is None
    assert report["selection_executor"] == "trusted_client"
    assert not report["protected_selection"]
    assert not report["executable_experiment"]
    assert not report["whole_generation_quality"]
    candidates = {row["mode"]: row for row in report["candidates"]}
    assert tuple(candidates) == MODES
    dense = candidates["dense"]
    cohorts = [row["token_cohort_sha256"] for row in dense["observations"]]
    for mode, row in candidates.items():
        assert report["summary"][mode] == summarize(row)
        assert row["source"] == report["source"]
        assert [r["token_cohort_sha256"] for r in row["observations"]] == cohorts
        assert row["maximum_swap_growth_bytes"] == 0
        assert row["diagnostic_process_peak_rss_bytes"] > 0
        for observed, full in zip(row["observations"], dense["observations"], strict=True):
            assert observed["full_logits_exact"][0]
            assert observed["worst_abs_logit_errors"][0] == 0
            if mode in {"dense", "retained_full"}:
                assert all(observed["full_logits_exact"])
                assert observed["full_final_kv_exact"]
                assert not any(observed["worst_abs_logit_errors"])
            if mode != "dense":
                contract = row["contract"]
                assert contract["policy"] == row["policy"]
                assert contract["selection_executor"] == "trusted_client"
                assert contract["owned_array_bound_bytes"] <= 2 << 30
                assert observed["prefill"]["active_kv_bytes"] <= full["prefill"]["active_kv_bytes"]
                n = observed["input_tokens"]
                keep = max((n * a + b - 1) // b for a, b in (
                    CONFIGURATION["static_keep"][mode], CONFIGURATION["observation_window"]))
                expected = sum((keep + contract["max_decode_steps"]) *
                    (8 * layer["kv_heads"] * layer["head_dim"] + 4) for layer in contract["layers"])
                assert observed["prefill"]["owned_payload_bytes"] == expected
                assert observed["final"]["owned_payload_bytes"] == expected
                assert expected <= contract["retained_payload_bound_bytes"]
                if mode != "retained_full":
                    assert not observed["full_final_kv_exact"]


def test_cohort_rejects_implementation_change(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research import retained_decoder as diagnostic

    monkeypatch.setattr(diagnostic, "implementation", lambda: {"native_sha256": "changed"})
    with pytest.raises(ValueError, match="implementation changed"):
        diagnostic.check_implementation({"native_sha256": "original"})
