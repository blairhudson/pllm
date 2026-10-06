"""Selected attention views cannot masquerade as private or compacted KV state."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mpcache_report_preserves_full_state_control_and_separate_projections(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research.mpcache_reference import CONFIGURATION, PUBLIC_PROMPTS

    report = json.loads((ROOT / "docs/evidence/research-mpcache-reference-qwen25.json").read_text())
    assert report["schema"] == "pllm.mpcache_reference.v1"
    assert report["configuration"] == CONFIGURATION
    for source, digest in report["source_sha256"].items():
        assert (ROOT / source).is_file()
        if source.endswith(".py"):
            assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
    assert len(report["observations"]) == len(PUBLIC_PROMPTS) * len(CONFIGURATION["candidates"])
    assert report["independent_native_oracle_checks"] > 0
    assert not report["executable_sdk"]
    assert not report["protected_selection"]
    assert not report["whole_generation_quality"]
    assert report["process_peak_rss_bytes"] is None
    assert report["whole_response_network_bytes"] is None
    assert "full runtime KV is retained" in " ".join(report["limitations"])
    cohorts = {}
    for row in report["observations"]:
        cohorts.setdefault(row["candidate"], set()).add(row["token_cohort_sha256"])
        assert row["same_token_top1_matches"][0]
        assert row["worst_abs_logit_error"][0] == 0
        assert len(row["same_token_top1_matches"]) == CONFIGURATION["selected_positions"]
        assert 0 < row["projected_prefill_active_kv_bytes"] <= row["full_prefill_active_kv_bytes"]
        for selection in row["selection"]:
            assert 0 < selection["selected"] <= selection["available"]
            assert len(selection["original_index_sha256"]) == 64
        if row["candidate"] == "full_control":
            assert all(row["same_token_top1_matches"])
            assert not any(row["worst_abs_logit_error"])
            assert row["projected_prefill_active_kv_bytes"] == row["full_prefill_active_kv_bytes"]
    assert all(cohort == cohorts["full_control"] for cohort in cohorts.values())
    for candidate, summary in report["summary"].items():
        rows = [r for r in report["observations"] if r["candidate"] == candidate]
        assert summary["prefill_count"] == len(PUBLIC_PROMPTS)
        assert summary["decode_count"] == len(PUBLIC_PROMPTS) * (CONFIGURATION["selected_positions"] - 1)
        assert summary["decode_matches"] == sum(sum(r["same_token_top1_matches"][1:]) for r in rows)
        assert summary["selected_attention_rows"] == sum(s["selected"] for r in rows for s in r["selection"])
        assert summary["available_attention_rows"] == sum(s["available"] for r in rows for s in r["selection"])
