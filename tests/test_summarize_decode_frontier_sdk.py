"""Bounded, provider-free coverage of the single-cohort summarizer."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("frontier_summary", ROOT / "scripts/summarize_decode_frontier_sdk.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


@pytest.fixture
def cohort(monkeypatch):
    """Four synthetic cells: no model resolution or role subprocesses."""
    from pllm import Experiment
    from pllm.deployment import WanConditions
    sha = "a" * 64
    ledger = {"scope": "synthetic covered bodies", "unmeasured": ["physical wire"],
              "traffic_attribution": {"total_all_link_body_bytes": 12,
                  "known_base_all_link_body_bytes": 3,
                  "known_output_dependent_all_link_body_bytes": 4,
                  "unattributed_all_link_body_bytes": 5}}
    monkeypatch.setattr(summary.growing, "validate_audit", lambda audit: deepcopy(ledger))
    experiments, cells, admissions, geometries, attempts = {}, [], {}, {}, []
    for index, (policy, params) in enumerate(summary.growing.VARIANTS.items()):
        key = "full/" + policy
        config = {"schema": "pllm.experiment.v2", "name": "synthetic",
            "pipeline": {"model": {"source": "qwen2", "kind": "tiny"}, "components": {
                "inventory": {"component": "pllm/prepared-inventory-policy/v1", "params": {
                        "policy": "request-sized", "rows": 1, "allocation": "demand", "refill": "on-demand",
                        **{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}}}}},
            "deployment": {"kind": "local", "root": ".pllm/synthetic"},
            "budget": {"requests": 1, "max_input_tokens": 128, "max_new_tokens": 8}}
        experiments[key] = Experiment.from_spec(config).to_spec()
        admissions[key] = {"admitted": True, "reasons": [], "required_bytes": 20}
        geometries[key] = {"plan_digest": sha, "schedule_digest": sha,
            "stages": [{"remote": True, "terminal_prefill": False, "macs_per_row": 2}]}
        audit = dict.fromkeys(summary.COUNTERS, 0)
        audit["generation_executed_decode_steps"] = 7
        row = {"index": 0, "status": "incomplete", "input_tokens": 3, "output_tokens": 8,
               "state": {"position": 10, "pending_count": 1,
                         **dict.fromkeys(("bits", "tokens", "pending", "output", "history", "rendered_history"), sha)},
               "request_seconds": 2 + index, "first_text_seconds": None, "decode_tps": None}
        client = {"responses": [row], "audit": audit, "ledger": deepcopy(ledger), "failure": None,
            "cleanup": {"known_private_containers_empty": True}, "client_lifetime_cpu_seconds": index + 1,
            "client_peak_rss_bytes": 100 + index, "client_window_seconds": 5}
        wan = {"schema": "pllm.native_wan.v1", "backend": "test", "enforced": True,
               "conditions": WanConditions().to_spec(), "conditions_digest": WanConditions().digest, "role_parties": {},
               "burst_bytes": 1, "read_chunk_bytes": 1, "max_connections": 1, "scope": "TCP payload",
               "excluded": ["IP/TCP headers"], "total_stream_bytes": 15, "directed_stream_bytes": {"a->b": 15}}
        cells.append({"workload": "full", "variant": policy, "replicate": 0, "client": client,
            "provider_birth_cpu_seconds": {"preparation": 2, "inference": 1},
            "provider_sampled_peak_rss_bytes": {"preparation": 200, "inference": 200},
            "aggregate_cleanup_cpu_seconds": 4, "complete_case_seconds": 6, "admitted_host_bytes": 20,
            "wan": wan, "semantic_work": summary.growing.work_counts(geometries[key]["stages"], 3, 7)})
        attempts.append({"workload": "full", "variant": policy, "replicate": 0, "status": "completed"})
    checks = summary.growing.verify_results(cells, tiny=True)
    checks["remote_90_percent_verified_all_cells"] = True
    return {"schema": "pllm.growing_epochs_sdk_benchmark.v1", "complete": True, "tiny": True,
        "memory_guard": {"failed": False}, "source_lock_digest": sha, "cohort_commitment": sha,
        "build": {"python": "synthetic/python", "expected_digest": sha}, "frozen_sources": {"test": sha},
        "windows": deepcopy(summary.WINDOWS[False]), "limits": ["no physical wire"],
        "execution_isolation": {"sdk_native_file_lock_checked_per_child": True}, "experiments": experiments,
        "config_hashes": {k: hashlib.sha256(Experiment.from_spec(v).canonical_bytes()).hexdigest() for k, v in experiments.items()},
        "semantic_geometry": geometries, "memory_admission": admissions, "planned_cells": 4,
        "results": cells, "attempts": attempts, "checks": checks}


def test_synthetic_summary_preserves_unknown_and_residual(cohort):
    result = summary.summarize(cohort, "f" * 64)
    group = result["groups"]["full/control"]
    assert result["raw_evidence_sha256"] == "f" * 64
    assert result["evidence_class"].startswith("functionality only")
    assert len(result["priorities"]) == 7
    assert group["median_client_lifetime_cpu_seconds"] == 1
    assert group["low_rss_gate"] is True
    assert group["low_tier_checkpoint_qualification"] is False
    assert group["requests"][0]["median_ttft_seconds"] is None
    assert group["requests"][0]["median_charged_n_minus_1_tps"] is None
    assert group["requests"][0]["per_token_ranking_eligible"] is False
    assert group["covered_bodies"]["complete_output_body_bytes"] is None
    assert group["covered_bodies"]["cohort_sums"]["unattributed_all_link_body_bytes"] == 5
    assert result["unknown"]["overall_peak_memory_bytes"] is None
    assert result["largest_client_peak_by_policy"]["control"] == 100
    assert result["observed_low_rss_gate_by_policy"]["control"] is True


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(complete=False), "incomplete"),
    (lambda d: d.update(failure={"stage": "failed"}), "failed"),
    (lambda d: d["memory_guard"].update(failed=True), "guard"),
    (lambda d: d["results"].pop(), "partial cohort"),
    (lambda d: d["results"].append(deepcopy(d["results"][0])), "duplicate"),
    (lambda d: d["attempts"][0].update(status="started"), "attempt"),
    (lambda d: d["results"][0]["client"]["responses"][0].update(status="failed"), "status"),
    (lambda d: d["results"][0]["client"].pop("client_peak_rss_bytes"), "missing"),
    (lambda d: d["results"][0]["provider_birth_cpu_seconds"].update(preparation=None), "missing"),
    (lambda d: d["results"][0]["wan"]["conditions"].update(download_mbps=1), "WAN"),
    (lambda d: d["windows"].update(provider_rss="complete lifetime peak"), "scopes"),
    (lambda d: d["results"][0]["client"]["ledger"]["traffic_attribution"].update(known_base_all_link_body_bytes=99), "scoped"),
    (lambda d: d["results"][0]["client"]["responses"][0]["state"].update(output="b" * 64), "parity"),
    (lambda d: d["results"][0]["client"]["responses"][0]["state"].update(position=11), "position"),
    (lambda d: d["config_hashes"].update({"full/control": "b" * 64}), "configuration digest"),
])
def test_rejects_invalid_cohort(cohort, mutation, match):
    mutation(cohort)
    with pytest.raises((ValueError, RuntimeError), match=match):
        summary.summarize(cohort, "f" * 64)


def test_mixed_output_counts_never_rank(cohort):
    assert summary.median([1, None]) is None
    assert summary.numeric_tree_summary([{"read_bytes": 1}, {}])["read_bytes"] is None
    ledger = cohort["results"][0]["client"]["ledger"]
    mixed = summary.traffic([ledger, ledger], [8, 7])
    assert mixed["known_output_bytes_per_generated_token"] is None
    assert mixed["per_token_ranking_eligible"] is False


def test_low_gate_covers_every_declared_workload(cohort):
    for field in ("experiments", "config_hashes", "memory_admission", "semantic_geometry"):
        cohort[field].update({key.replace("full/", "early/"): deepcopy(value)
                              for key, value in list(cohort[field].items())})
    for field in ("results", "attempts"):
        additional = deepcopy(cohort[field])
        for row in additional:
            row["workload"] = "early"
        cohort[field].extend(additional)
    cohort["planned_cells"] *= 2
    cohort["results"][4]["client"]["client_peak_rss_bytes"] = summary.LOW_BYTES + 1
    cohort["checks"] = summary.growing.verify_results(cohort["results"], tiny=True)
    cohort["checks"]["remote_90_percent_verified_all_cells"] = True
    result = summary.summarize(cohort, "f" * 64)
    assert result["groups"]["full/control"]["low_rss_gate"] is True
    assert result["observed_low_rss_gate_by_policy"]["control"] is False
    assert result["largest_client_peak_by_policy"]["control"] == summary.LOW_BYTES + 1


@pytest.mark.parametrize("filename", ["growing-epochs-tiny-r4.json", "decode-cache-tiny-r5.json"])
def test_completed_tiny_functionality_only(filename):
    path = ROOT / "target" / filename
    if not path.exists():
        pytest.skip("optional completed functionality fixture unavailable")
    raw = path.read_bytes()
    result = summary.summarize(json.loads(raw), hashlib.sha256(raw).hexdigest())
    assert result["evidence_class"].startswith("functionality only")
    assert all(not g["low_tier_checkpoint_qualification"] for g in result["groups"].values())
    if filename.startswith("decode"):
        requests = result["groups"]["same_seed/decode"]["requests"]
        assert [r["window"] for r in requests] == ["cold", "later", "later"]
        assert requests[1]["actual_work_medians"]["zero_executed_work"] is True


def test_cli_refuses_existing_output(cohort, tmp_path):
    source, destination = tmp_path / "raw.json", tmp_path / "summary.json"
    source.write_text(json.dumps(cohort))
    destination.write_text("keep")
    with pytest.raises(SystemExit):
        summary.main(["--input", str(source), "--output", str(destination)])
    assert destination.read_text() == "keep"


def test_cli_new_output_and_stdout(cohort, tmp_path, capsys):
    source, destination = tmp_path / "raw.json", tmp_path / "summary.json"
    source.write_text(json.dumps(cohort))
    summary.main(["--input", str(source), "--output", str(destination)])
    result = json.loads(destination.read_text())
    assert result["raw_evidence_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    summary.main(["--input", str(source)])
    assert json.loads(capsys.readouterr().out) == result


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["results"][0]["client"]["responses"][1]["request_ledger"].update(all_link_serialized_body_bytes=1), "scoped"),
    (lambda d: d["results"][0]["client"]["responses"][1]["audit_delta"].update(prepared_stage_rows_claimed=0), "scoped"),
    (lambda d: d["results"][0]["client"]["storage_lifetime"].update(read_bytes=None), "missing"),
    (lambda d: d["results"][0]["client"]["storage_setup_cleanup_or_failed_request_residual"].update(read_bytes=1), "disk window"),
    (lambda d: d["results"][0]["client"]["responses"][1]["state"].update(output="b" * 64), "continuity|parity"),
])
def test_decode_scoped_rejections(mutation, match):
    path = ROOT / "docs/evidence/decode-cache-sdk-qwen25-2026-10-06.json"
    report = json.loads(path.read_bytes())
    mutation(report)
    with pytest.raises((ValueError, RuntimeError), match=match):
        summary.summarize(report, "f" * 64)


@pytest.mark.parametrize("name", ["growing-epochs", "decode-cache"])
def test_published_checkpoint_summary_replays_exactly(name):
    path = ROOT / "docs/evidence" / f"{name}-sdk-qwen25-2026-10-06.json"
    raw = path.read_bytes()
    result = summary.summarize(json.loads(raw), hashlib.sha256(raw).hexdigest())
    expected = json.loads(path.with_stem(path.stem + "-summary").read_bytes())
    assert result == expected
    assert result["evidence_class"] == "checkpoint cohort"
    assert all(result["observed_low_rss_gate_by_policy"].values())
    if name == "decode-cache":
        for row in result["groups"]["same_seed/decode"]["requests"][1:]:
            assert row["actual_work_medians"]["zero_executed_work"] is True
            assert row["counter_medians"]["prepared_stage_rows_issued"] == 0
            assert row["disk_io_medians"]["written_bytes"] == 0
            assert row["disk_io_medians"]["publications"] == 0
