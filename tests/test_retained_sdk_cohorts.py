"""Fresh-client SDK evidence preserves policy identities and complete cost scopes."""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
COHORTS = (("research-retained-sdk-qwen25", "short", 52),
           ("research-retained-sdk-long-qwen25", "long", 379))


def read(path):
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("directory,context,tokens", COHORTS)
def test_completed_sdk_candidates_keep_numerics_separate_and_conserve_costs(
        monkeypatch, directory, context, tokens):
    monkeypatch.syspath_prepend(str(ROOT))
    from benchmarks.research import retained_sdk
    folder = EVIDENCE / directory
    report = read(folder / f"{context}.json")
    preflight = read(folder / f"{context}-preflight.json")
    assert report["schema_version"] == "pllm.loopback_benchmark_comparison.v1"
    assert preflight["admitted"] is True
    assert len(report["candidates"]) == len(preflight["candidates"]) == 4
    assert report["checks"] == {
        "all_candidates_passed": True, "matched_kernel_backend": True,
        "matched_numeric_contract": False, "matched_verification": True,
        "matched_workload": True, "passed": False, "unique_configurations": True}
    assert all(ranking == [] for ranking in report["rankings"].values())
    configurations, fingerprints, covered = [], set(), set()
    for name, candidate in zip(retained_sdk.POLICIES, report["candidates"], strict=True):
        expected = getattr(retained_sdk, name)()
        assert candidate["name"] == expected.name
        assert candidate["configuration_digest"] == expected.configuration_digest()
        assert candidate["pipeline"] == expected.pipeline.to_spec()
        result = candidate["report"]
        config = result["configuration"]
        configurations.append(config)
        assert config["client_process_isolated"] is True
        assert config["provider_backend"] == "native" and config["wan_emulation"] is True
        assert (config["repetitions"], config["warmups"], config["max_output_tokens"]) == (1, 0, 8)
        assert result["checks"]["passed"] is True
        assert result["warmup_runs"] == [] and len(result["runs"]) == 1
        run = result["runs"][0]
        fingerprints.add(run["model_fingerprint"])
        assert run["tokens"] == {"authoritative": True, "input_tokens": tokens,
                                 "output_tokens": 8, "total_tokens": tokens + 8}
        assert run["privacy"]["plaintext_prompt_bytes_sent"] == 0
        assert run["privacy"]["plaintext_token_ids_sent"] == 0
        assert run["privacy"]["preparation_requests_during_online"] == 0
        assert run["privacy"]["bundle_cache_hits"] == 0
        assert run["privacy"]["bundle_cache_misses"] == 1
        assert len(run["generation"]["output_text_digest"]) == 64
        placement = result["client_body_placement"]
        assert placement["local_stage_count"] == 0 and placement["remote_stage_count"] == 96
        assert placement["declared_body_linear_macs_per_row_client"] == 0
        assert result["memory_guard"]["tripped"] is False
        assert result["memory_guard"]["maximum_host_swap_growth_bytes"] == 0
        memory = result["client_process_memory"]
        assert memory["after"]["lifetime_peak_rss_bytes"] >= memory["after"]["rss_bytes"] > 0
        assert result["native_process_memory"]["complete"] is True
        cpu = result["process_cpu_accounting"]
        assert cpu["aggregate_cold_first_response_cpu_seconds"] == pytest.approx(
            sum(cpu["cold_first_response_cpu_seconds_by_role"].values()))
        assert set(cpu["cold_first_response_cpu_seconds_by_role"]) == {"client", "inference", "preparation"}
        assert cpu["full_response_compute_cap_checked"] is False
        samples = result["native_network_accounting"]["samples"]
        assert samples["warmups"] == [] and len(samples["runs"]) == 1
        before, after = [samples["runs"][0][key]["wan_emulation"] for key in ("before", "after")]
        assert before["total_stream_bytes"] == samples["startup"]["wan_emulation"]["total_stream_bytes"]
        for snapshot in (before, after):
            assert snapshot["enforced"] is True
            assert snapshot["backend"] == "native-shared-tcp-pacer"
            assert (snapshot["conditions"]["download_mbps"], snapshot["conditions"]["upload_mbps"]) == (100, 40)
            assert sum(snapshot["directed_stream_bytes"].values()) == snapshot["total_stream_bytes"]
        summary = result["summary"]
        covered.add(summary["accounted_setup_through_first_response_body_bytes"])
        assert after["total_stream_bytes"] >= summary["accounted_setup_through_first_response_body_bytes"]
        assert summary["generated_output_tokens"] == 8 and summary["completed_runs"] == 1
    assert len(fingerprints) == len(covered) == 1
    for key in ("prompt_digest", "source_lock_digest", "wan_emulation_digest"):
        assert len({c[key] for c in configurations}) == 1
    # Full retention's complete generated text agrees with the dense control.
    # Sparse policies retain separate identities even when their text agrees.
    outputs = [c["report"]["runs"][0]["generation"]["output_text_digest"] for c in report["candidates"]]
    assert outputs[0] == outputs[1]
    assert [output == outputs[0] for output in outputs] == (
        [True, True, True, True] if context == "short" else [True, True, False, True])


def test_sdk_reproduction_binds_same_installed_runtime_and_public_workloads():
    original = read(EVIDENCE / COHORTS[0][0] / "reproduction.json")
    retry = read(EVIDENCE / COHORTS[1][0] / "reproduction.json")
    assert original["installed_package_sha256"] == retry["installed_package_sha256"]
    assert len(original["installed_package_sha256"]) > 100
    assert original["targets"] == retry["targets"]
    assert len(set(original["targets"])) == 4
    for metadata in (original, retry):
        assert len(metadata["factory_sha256"]) == 64
        for workload in metadata["workloads"].values():
            assert hashlib.sha256(workload["public_prompt"].encode()).hexdigest() == workload["prompt_sha256"]
            assert workload["output_cap"] == 8
        assert metadata["execution"]["fresh_clients"] is True
        assert metadata["execution"]["wan_mbps"] == [100, 40]
    assert retry["journal_rows"] == 22
    assert retry["workloads"]["long"]["prompt_sha256"] != original["workloads"]["long"]["prompt_sha256"]


def test_overlarge_residue_frame_failure_is_preserved_after_memory_admission():
    folder = EVIDENCE / "research-retained-sdk-qwen25"
    assert read(folder / "long-preflight.json")["admitted"] is True
    failed = read(folder / "long.json")
    assert failed["schema_version"] == "pllm.benchmark_progress.v1"
    assert failed["complete"] is False and failed["status"] == "failed"
    assert [c["status"] for c in failed["candidates"]] == ["failed", "pending", "pending", "pending"]
    assert all(c["report"] is None for c in failed["candidates"])
    assert "rankings" not in failed
    diagnosis = read(folder / "failure-diagnosis.json")
    assert diagnosis["frame_elements"] == (
        diagnosis["correction_rows"] * diagnosis["grouped_gate_up_outputs"])
    assert diagnosis["frame_elements"] > diagnosis["maximum_frame_elements"] == 4_000_000
    assert diagnosis["performance_result"] is None and diagnosis["completed_response"] is False
