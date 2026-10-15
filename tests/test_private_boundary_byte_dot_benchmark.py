"""Synthetic evidence tampering and orchestration checks; never model timings."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import pytest

from pllm import Experiment
from pllm.metrics import BenchmarkRun, BuildRef

from benchmarks.research import private_boundary_byte_dot as study


@lru_cache
def _archived():
    return json.loads(Path("docs/evidence/private-boundary-decode-confirmation-150x8.json").read_bytes())


@pytest.fixture
def cohort():
    """Wrap copied observations in explicitly synthetic, valid v2 test records."""
    from pllm.metrics.benchmark_runner import _result

    original = _archived()
    templates = {row["execution"]["replicate"]: row for row in original["results"]
        if row["execution"]["case"]["name"] == study.confirmation.WINNER}
    build = deepcopy(templates[0]["execution"]["build"])
    suite = study.matched_suite(build=BuildRef(expected_digest=build["digest"]))
    document = {"schema": "pllm.benchmark_run.v1", "complete": True, "cohort": "e" * 64,
        "suite": suite.to_spec(), "results": []}
    for case in suite.cases:
        exp = Experiment.from_spec(case.target.to_spec()["parameters"]["model"]["experiment"])
        for replicate in range(3):
            row = deepcopy(templates[replicate])
            execution = row["execution"]
            detail = execution["details"]
            detail["report"]["experiment"]["pipeline_digest"] = exp.pipeline.digest()
            payload = {"status": row["status"], "failure_code": execution["failure_code"],
                "comparison": execution["comparison"], "details": detail,
                "environment": row["environment"]["attributes"], "metrics": row["metrics"], "samples": row["samples"],
                "fixture": execution["fixture"], "oracle": execution["oracle"], "admission": execution["admission"],
                "limitations": ["Synthetic test evidence: copied observations, no byte-dot model execution."]}
            document["results"].append(_result(case, suite, replicate, document["cohort"], build, payload).to_dict())
    BenchmarkRun(document)
    return document


def test_complete_read_only_same_topology_kernel_comparison(cohort):
    before = deepcopy(cohort)
    result = study.audit(cohort)
    assert cohort == before
    assert result["clients"] == 6 and result["replicates"] == 3
    assert result["all_logits_active_kv_positions_selections_rng_and_outputs_match"]
    assert result["maximum_observed_swap_growth_bytes"] == 0
    assert tuple(result["groups"]) == study.NAMES
    comparison = result["comparisons"][study.CANDIDATE][study.CONTROL]
    assert set(comparison) == set(study.METRICS)
    assert all(value["delta"] == 0 and value["candidate_wins"] is False for value in comparison.values())
    assert all(row["head_reply_delta_bytes"] == 0 for row in result["body_reconciliation"][study.CANDIDATE])
    for name in study.NAMES:
        group = result["groups"][name]
        assert group["maximum_client_peak_bytes"] == max(sample["process.peak_rss"] for sample in group["samples"])
        assert all(set(row) == study.ROLES for row in group["cold_first_cpu_seconds_by_role"])


@pytest.mark.parametrize("tiny", (False, True))
def test_only_head_kernel_changes_in_matched_recipes(tiny):
    values = study.recipes(tiny=tiny)
    left, right = (values[name] for name in study.NAMES)
    a, b = left.to_spec(), right.to_spec()
    assert a != b
    assert left.pipeline.components["boundary"].to_spec()["params"] == {
        "head_dispatch": "parallel", "head_output_encoding": "uniform_residues"}
    assert right.pipeline.components["boundary"].get_params()["head_kernel"] == "byte_dot"
    b["pipeline"]["components"]["boundary"] = a["pipeline"]["components"]["boundary"]
    assert a == b
    suite = study.matched_suite(build=BuildRef(expected_digest="a" * 64), tiny=tiny)
    assert suite.sampling.processes == 3 and suite.sampling.order == "balanced"
    assert suite.sampling.threads == suite.sampling.repetitions == 1 and suite.sampling.warmups == 0
    assert all(case.target.to_spec()["parameters"]["state_audit"] for case in suite.cases)


@pytest.mark.parametrize("mutation", (
    "incomplete", "missing", "duplicate", "cohort", "build_attributes", "case_digest", "target_digest",
    "tps_aliases", "metric_windows", "sample_duration", "state", "rng", "output", "pipeline",
    "roles", "cpu_sum", "cpu_owner", "swap", "guard", "body_total", "tcp", "material",
    "head_count", "one_byte_head_change", "missing_owner", "client_peak", "final_peak", "kernel_case",
))
def test_rejects_forged_or_incomplete_evidence(cohort, mutation):
    row = cohort["results"][-1]
    execution = row["execution"]
    report = execution["details"]["report"]
    request = report["runs"][0]
    if mutation == "incomplete":
        cohort["complete"] = False
    elif mutation == "missing":
        cohort["results"].pop()
    elif mutation == "duplicate":
        cohort["results"][-1] = deepcopy(cohort["results"][-2])
    elif mutation == "cohort":
        execution["cohort"] = "d" * 64
    elif mutation == "build_attributes":
        execution["build"]["attributes"] = {"forged": True}
    elif mutation == "case_digest":
        execution["case_digest"] = "0" * 64
    elif mutation == "target_digest":
        row["configuration_digest"] = "0" * 64
    elif mutation == "tps_aliases":
        for metric in row["metrics"]:
            if metric["id"] in ("response.decode_tps", "request.0.decode_tps"):
                metric["value"] *= 2
    elif mutation == "metric_windows":
        for sample in cohort["results"]:
            for metric in sample["metrics"]:
                if metric["id"] in ("response.ttft", "request.0.ttft"):
                    metric["window"] = "after_first_output"
        BenchmarkRun(cohort)
    elif mutation == "sample_duration":
        row["samples"][0]["duration_seconds"] += 1
    elif mutation in ("state", "rng"):
        record = execution["details"]["response_state_audit"]["bindings"][0]["selections"][0]
        record["logits_and_active_kv" if mutation == "state" else "rng_after"] = "f" * 64
    elif mutation == "output":
        request["generation"]["output_text_digest"] = "f" * 64
    elif mutation == "pipeline":
        report["experiment"]["pipeline_digest"] = "f" * 64
    elif mutation == "roles":
        report["configuration"]["roles"].remove("boundary_b")
    elif mutation == "cpu_sum":
        report["process_cpu_accounting"]["aggregate_cold_first_response_cpu_seconds"] += 1
    elif mutation == "cpu_owner":
        del report["process_cpu_accounting"]["cold_first_response_cpu_seconds_by_role"]["boundary_b"]
    elif mutation == "swap":
        report["memory_guard"]["maximum_host_swap_growth_bytes"] = 4096
    elif mutation == "guard":
        report["memory_guard"]["tripped"] = True
    elif mutation == "body_total":
        report["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"] -= 1
    elif mutation == "tcp":
        report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]["total_stream_bytes"] -= 1
    elif mutation == "material":
        report["prepared_material_accounting"]["conserved"] = False
    elif mutation == "head_count":
        request["privacy"]["private_boundary_head_calls"] -= 1
    elif mutation == "one_byte_head_change":
        request["privacy"]["private_boundary_head_download_bytes"] += 1
    elif mutation == "missing_owner":
        execution["details"]["response_state_audit"]["client_memory_owners"].remove("native_wan_relay")
    elif mutation == "client_peak":
        report["client_process_memory"]["after"]["lifetime_peak_rss_bytes"] -= 1
    elif mutation == "final_peak":
        next(metric for metric in row["metrics"] if metric["id"] == "process.peak_rss")["value"] = 1
    elif mutation == "kernel_case":
        execution["case"]["target"]["parameters"]["model"]["experiment"]["pipeline"]["components"]["boundary"]["params"]["head_kernel"] = "existing"
    with pytest.raises(study.Rejected):
        study.audit(cohort)


def test_null_decode_rate_never_becomes_a_win():
    row = {metric: None if metric == "response.decode_tps" else 1 for metric in study.METRICS}
    group = {"medians": row, "samples": [row] * 3}
    rate = study._compare(group, group)["response.decode_tps"]
    assert rate == {"control": None, "candidate": None, "delta": None,
        "improvement_fraction": None, "candidate_wins": None, "paired_deltas": [None] * 3}


@pytest.fixture
def cli(monkeypatch, tmp_path):
    from scripts import benchmark_private_boundary_byte_dot as cli

    def forbidden(*args, **kwargs):
        raise AssertionError("no model or timing execution allowed")

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(cli, "read_fixture", lambda path: ("public prompt", "public hash"))
    monkeypatch.setattr(cli, "public_input_counts", lambda *args: ({"150": 150}, "source"))
    monkeypatch.setattr(cli.BuildRef, "capture", lambda: BuildRef(expected_digest="a" * 64))
    monkeypatch.setattr(cli, "preflight", lambda suite: [
        {"case": case.name, "admission": {"admitted": True}} for case in suite.cases])
    monkeypatch.setattr(cli, "hashes", lambda paths: {"fixture": "frozen"})
    monkeypatch.setattr(cli, "benchmark_suite", forbidden)
    return cli


def test_preflight_only_and_exclusive_prefix(cli, tmp_path):
    output = tmp_path / "preflight.json"
    assert cli.main(["--output", str(output), "--preflight-only"]) == 0
    assert not output.exists()
    plan = json.loads(output.with_suffix(".plan.json").read_bytes())
    assert len(plan["suite"]["cases"]) == 2
    progress = output.with_suffix(".progress.json")
    before = progress.read_bytes()
    assert json.loads(before)["status"] == "admitted"
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output), "--preflight-only"])
    assert progress.read_bytes() == before


def test_shared_lock_rejects_before_preflight(cli, tmp_path):
    from filelock import FileLock

    lock_path = tmp_path / ".cache/pllm/frontier-probe.lock"
    lock_path.parent.mkdir(parents=True)
    output = tmp_path / "busy.json"
    with FileLock(str(lock_path), timeout=0):
        assert cli.main(["--output", str(output), "--preflight-only"]) == 1
    progress = json.loads(output.with_suffix(".progress.json").read_bytes())
    assert progress["failure_code"] == "frontier_cohort_lock_busy"
    assert not output.with_suffix(".plan.json").exists()


def test_failed_cohort_retains_original_evidence(cli, monkeypatch, tmp_path, cohort):
    from types import SimpleNamespace

    cohort["complete"] = False
    output = tmp_path / "failed.json"

    def run(suite, *, inputs, output):
        assert tuple(inputs) == study.NAMES
        output.write_text(json.dumps(cohort))
        return SimpleNamespace(to_dict=lambda: cohort)

    monkeypatch.setattr(cli, "benchmark_suite", run)
    assert cli.main(["--output", str(output)]) == 1
    assert json.loads(output.read_bytes()) == cohort
    assert not output.with_suffix(".summary.json").exists()
    progress = json.loads(output.with_suffix(".progress.json").read_bytes())
    assert progress["status"] == "rejected" and not progress["measurement_complete"]
