"""Presentation contracts only: synthetic ordinary reports, no native/model work."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
from importlib.abc import MetaPathFinder
import io
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "paper/artifacts/gkm-sdk/qwen-response-v1/benchmark.json"


@pytest.fixture
def renderer(monkeypatch):
    class NoExecutionImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if any(fullname == name or fullname.startswith(name + ".") for name in (
                "pllm._native", "pllm.model_loader", "pllm.runtime.loaders",
                "torch", "transformers", "huggingface_hub", "numpy",
            )):
                raise AssertionError(f"renderer attempted execution import: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [NoExecutionImports(), *sys.meta_path])
    spec = importlib.util.spec_from_file_location("method_results", ROOT / "docs/scripts/render-method-results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sources():
    return json.loads((ROOT / "paper/methods/source-lock.v1.json").read_text())


def cell(path, *, name="gkm", method="gupta-katz-miers", case=None, scope="trained_checkpoint_response"):
    document = json.loads(path.read_text())
    return {"name": name, "method": method, "era": "historical", "scope": scope,
            "report": str(path), "case": case or document["suite"]["cases"][0]["name"],
            "cohort": document["cohort"]}


class SyntheticReport:
    """Uses ordinary SDK report construction with literal, independently checked values.

    Fixture 'checkpoint' labels are synthetic publisher assertions, NOT real
    checkpoint execution/admission or native cryptographic acceptance evidence.
    """

    def __init__(self, directory):
        from pllm._benchmark_contract import digest
        from pllm._benchmark_evidence import measurement
        from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, BuildRef, Sampling, Scenario
        from pllm.metrics.benchmark_runner import BenchmarkRun, _result
        from examples.papers.gkm import experiment

        build = {"attributes": {"test_only": "synthetic"}}
        build["digest"] = digest(build["attributes"])
        cases = tuple(BenchmarkCase(name, BenchmarkTarget.response(
            experiment().with_params(name=name), backend="native", scenario=Scenario.cold(max_output_tokens=2)),
            BuildRef(expected_digest=build["digest"])) for name in ("control", "candidate"))
        suite = BenchmarkSuite(cases, Sampling(processes=1))
        self.document = {"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(),
                         "cohort": "1" * 64, "complete": True, "results": []}
        for index, case in enumerate(cases):
            model = {"id": "synthetic-not-checkpoint", "checkpoint_digest": "a" * 64, "source_lock_digest": "b" * 64}
            runtime = {"source_lock_digest": "b" * 64, "checkpoint_digest": "a" * 64,
                "model_fingerprint": "f" * 64, "vocabulary": 7,
                "numeric": {"numeric_profile": "synthetic-q11"}, "sharing": "full", "runtime_kind": "gkm"}
            report = {"configuration": {"source_lock_digest": "b" * 64, "tiny": False,
                "provider_backend": "native", "roles": ["client", "worker_a", "worker_b"]},
                "runtime_contract": runtime, "checks": {"passed": True}, "runs": [{
                    "status": "completed", "tokens": {"authoritative": True, "input_tokens": 4, "output_tokens": 2},
                    "model_fingerprint": "f" * 64, "generation": {"output_text_digest": "e" * 64}}]}
            metrics = [measurement("request.0.full", 8 / (index + 1), window="request_setup_through_completion"),
                measurement("process.peak_rss", 1024, kind="memory", window="process_birth_through_target",
                            scope="process_lifetime_high_water"),
                measurement("client.owned_sampled_peak_rss_sum", 4096, kind="memory", origin="estimate",
                            scope="sampled_client_group_rss", owners=["client", "helper"]),
                measurement("client.owned_peak_rss_lower_bound", 6144, kind="memory", origin="lower_bound",
                            scope="client_group_os_high_water_lower_bound", owners=["client", "helper"]),
                measurement("request.0.decode_tps", 12, kind="throughput")]
            payload = {"status": "completed", "fixture": {"kind": "checkpoint", "digest": "b" * 64, "metadata": {}},
                "oracle": {"kind": "synthetic_test_only", "outcome": "passed", "checked_values": 2},
                "admission": {"admitted": True, "required_bytes": 1, "limit_bytes": 2,
                    "host_headroom_bytes": 2, "scope": "synthetic", "reason": None},
                "comparison": {key: "c" * 64 for key in ("source_digest", "numeric_digest", "privacy_digest",
                    "state_digest", "workload_digest", "output_digest", "measurement_digest")},
                "samples": [{"index": 0, "kind": "measurement", "status": "completed",
                    "duration_seconds": 8 / (index + 1), "artifact_digest": None}],
                "metrics": metrics, "details": {"model": model, "report": report},
                "environment": {"synthetic": True}, "limitations": ["synthetic publisher test fixture only"]}
            payload["comparison"]["source_digest"] = "b" * 64
            self.document["results"].append(_result(case, suite, 0, "1" * 64, build, payload).to_dict())
        self.path = directory / "synthetic-ordinary.json"
        self.run_type = BenchmarkRun
        self.save()

    def save(self):
        self.run = self.run_type(self.document)
        self.path.write_text(json.dumps(self.document))

    def selection(self, *, scope="trained_checkpoint_response"):
        return {"cells": [cell(self.path, name=name, case=name, scope=scope) for name in ("control", "candidate")],
                "comparisons": [{"control": "control", "candidate": "candidate", "metric": "request.0.full",
                                 "record": self.run.compare("control", "candidate", "request.0.full")}]}


def test_frozen_gkm_preserves_actual_source_counts_numeric_cost_windows_and_unknowns(renderer, sources):
    original = json.loads(FROZEN.read_text())
    rows = renderer.method_rows({"cells": [cell(FROZEN)]}, source_lock=sources)
    row = rows[0]
    assert row["status"] == "accepted_bounded_response" and row["era"] == "historical"
    assert row["qualification"] == "not established by this renderer"
    assert row["input_tokens"] == [4] and row["output_tokens"] == [2]
    assert row["full_seconds"] == 143.873030084 and row["covered_body_bytes"] == 87227888
    assert row["decode_tokens_per_second"] is None
    assert row["source"]["model"] == original["results"][0]["model"]
    details = original["results"][0]["execution"]["details"]
    assert row["numeric"]["runtime"] == details["runtime_contract"]["numeric"]
    assert row["workload"]["runtime_contract"]["vocabulary"] == 151936
    assert row["privacy"]["sharing"] == "full"
    assert row["provenance"]["report_sha256"] == hashlib.sha256(FROZEN.read_bytes()).hexdigest()
    assert row["provenance"]["cohort"] == original["cohort"]
    assert row["metric_contracts"]["process.peak_rss"]["window"] == "process_birth_through_target"
    assert row["metric_contracts"]["network.covered_body_bytes"]["measurement_scope"] == "covered_application_bodies"
    assert row["comparisons"] == []


def test_failed_fss_and_missing_eight_methods_never_become_zero_or_success(renderer, sources):
    selection = {"cells": [cell(FROZEN), {"name": "fss-last-attempt", "method": "fusefss",
        "scope": "trained_checkpoint_response", "era": "current_candidate", "status": "timeout",
        "reason": "no accepted response; failure cause unassigned"}]}
    rows = renderer.method_rows(selection, source_lock=sources)
    assert len(rows) == 10
    assert rows[1]["status"] == "timeout"
    assert rows[1]["provenance"] is None
    assert sum(r["status"] == "blocked" for r in rows) == 8
    assert all(r[k] is None for r in rows[1:] for k in renderer.METRICS)


def test_existing_comparison_record_only_and_independent_literal_ratio(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    comparison = rows[0]["comparisons"][0]["record"]
    assert comparison["status"] == "comparable"
    assert comparison["control_median"] == 8 and comparison["candidate_median"] == 4
    assert comparison["paired_control_over_candidate"] == [2]
    assert not any("rank" in r for r in rows)
    selection = fixture.selection()
    selection["comparisons"][0]["record"]["candidate_median"] = 0
    with pytest.raises(ValueError, match="comparison record differs"):
        renderer.method_rows(selection, source_lock=sources)


@pytest.mark.parametrize("drift", ["privacy_digest", "numeric_digest", "measurement_window"])
def test_ordinary_mismatched_contracts_remain_unranked(renderer, sources, tmp_path, drift):
    fixture = SyntheticReport(tmp_path)
    result = fixture.document["results"][1]
    if drift == "measurement_window":
        next(m for m in result["metrics"] if m["id"] == "request.0.full")["window"] = "different_cost_window"
    else:
        result["execution"]["comparison"][drift] = "d" * 64
    fixture.save()
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    assert rows[0]["comparisons"][0]["record"]["status"] == "unranked"


def test_separate_salts_cannot_pool_even_same_build_workload_and_output(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    selection = fixture.selection()
    other = copy.deepcopy(fixture.document)
    other["cohort"] = "2" * 64
    for result in other["results"]:
        result["execution"]["cohort"] = other["cohort"]
    other_path = tmp_path / "different-salt.json"
    other_path.write_text(json.dumps(other))
    selection["cells"][1] = cell(other_path, name="candidate", case="candidate")
    rows = renderer.method_rows(selection, source_lock=sources)
    assert rows[0]["comparisons"][0]["record"]["status"] == "unranked"
    assert rows[0]["provenance"]["cohort"] != rows[1]["provenance"]["cohort"]


def test_legacy_controls_keep_only_their_own_scope_and_comparison(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    for result in fixture.document["results"]:
        result["execution"]["details"]["report"].pop("runtime_contract")
    fixture.save()
    rows = renderer.method_rows(fixture.selection(scope="historical_control"), source_lock=sources)
    assert rows[0]["status"] == "completed_scoped_only"
    assert rows[0]["comparisons"][0]["record"]["status"] == "comparable"
    assert rows[0]["qualification"] == "not established by this renderer"


def test_retained_legacy_cohort_matches_ordinary_helper_without_becoming_ten_method_evidence(renderer, sources):
    from pllm.metrics import BenchmarkRun

    path = ROOT / "docs/evidence/private-boundary-decode-confirmation-150x8.json"
    run = BenchmarkRun.load(path)
    selected = []
    for recipe in ("maintained", "coupled-serial-u32"):
        item = cell(path, name=recipe, method="retained-boundary-control", case=recipe, scope="historical_control")
        item["source_reference"] = "paper/HISTORICAL-RESULTS.md; paper/artifacts/historical-results-v1.json"
        selected.append(item)
    record = run.compare("maintained", "coupled-serial-u32", "network.all_link_stream_bytes")
    rows = renderer.method_rows({"cells": selected, "comparisons": [{
        "control": "maintained", "candidate": "coupled-serial-u32",
        "metric": "network.all_link_stream_bytes", "record": record}]}, source_lock=sources)
    retained = [r for r in rows if r["method"] == "retained-boundary-control"]
    assert len(retained) == 6  # Each original process replicate remains separate.
    assert retained[0]["comparisons"][0]["record"] == record
    assert all(r["era"] == "historical" and r["scope"] == "historical_control" for r in retained)
    assert sum(r["status"] == "blocked" for r in rows) == 10
    assert len({r["provenance"]["cohort"] for r in retained}) == 1


def test_gkm_runtime_cannot_be_relabelled_as_another_method(renderer, sources):
    selected = cell(FROZEN, method="puma")
    with pytest.raises(ValueError, match="another method"):
        renderer.method_rows({"cells": [selected]}, source_lock=sources)


def test_frozen_cohort_is_explicit_not_auto_selected(renderer, sources):
    selected = cell(FROZEN)
    selected["cohort"] = "0" * 64
    with pytest.raises(ValueError, match="frozen cohort"):
        renderer.method_rows({"cells": [selected]}, source_lock=sources)


@pytest.mark.parametrize("claim", ["whole_method_complete", "publication_qualified"])
def test_passing_checks_do_not_authorize_whole_method_completion(renderer, sources, claim):
    selected = cell(FROZEN)
    selected[claim] = True
    with pytest.raises(ValueError, match="whole-method/publication"):
        renderer.method_rows({"cells": [selected]}, source_lock=sources)


def test_generated_probe_cannot_enter_trained_response_scope(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    fixture.document["results"][0]["execution"]["fixture"]["kind"] = "generated_checkpoint"
    fixture.save()
    with pytest.raises(ValueError, match="component/generated"):
        renderer.method_rows(fixture.selection(), source_lock=sources)


@pytest.mark.parametrize("missing", ["vocabulary", "body", "counts", "source_lock", "source_kind"])
def test_unknown_native_identity_cannot_supply_accepted_trained_row(renderer, sources, tmp_path, missing):
    fixture = SyntheticReport(tmp_path)
    report = fixture.document["results"][0]["execution"]["details"]["report"]
    if missing == "vocabulary":
        report["runtime_contract"].pop("vocabulary")
    elif missing == "body":
        report["runs"][0].pop("model_fingerprint")
    elif missing == "counts":
        report["runs"][0]["tokens"]["input_tokens"] = None
    elif missing == "source_lock":
        report["configuration"]["source_lock_digest"] = None
    else:
        report["configuration"].pop("tiny")
    fixture.save()
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    assert rows[0]["status"] == "completed_scoped_only"
    assert rows[0]["decode_tokens_per_second"] is None
    assert rows[0]["comparisons"][0]["record"]["status"] == "unranked"


def test_failed_attempt_retained_and_no_throughput_from_success_subset(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    fixture.document["results"][1]["status"] = "failed"
    fixture.document["results"][1]["execution"]["failure_code"] = "synthetic_failure"
    fixture.save()
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    assert rows[1]["status"] == "failed" and rows[1]["reason"] == "synthetic_failure"
    assert rows[1]["decode_tokens_per_second"] is None
    assert rows[1]["metric_contracts"]["request.0.decode_tps"]["value"] is None
    assert rows[0]["comparisons"][0]["record"]["status"] == "unranked"


def test_failed_before_source_admission_remains_failed_not_a_probe_error(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    result = fixture.document["results"][1]
    result["status"] = "timeout"
    result["execution"]["failure_code"] = "timeout"
    result["execution"]["fixture"] = {"kind": "none", "digest": None, "metadata": {}}
    result["execution"]["details"].pop("report")
    fixture.save()
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    assert rows[1]["status"] == "timeout" and rows[1]["output_tokens"] is None
    assert rows[1]["decode_tokens_per_second"] is None


def test_incomplete_cohort_missing_attempt_not_dropped_or_filled(renderer, sources, tmp_path):
    fixture = SyntheticReport(tmp_path)
    fixture.document["complete"] = False
    fixture.document["results"].pop()
    fixture.save()
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    assert rows[1]["cell"] == "candidate" and rows[1]["status"] == "unmeasured"
    assert rows[1]["full_seconds"] is None
    assert rows[0]["comparisons"][0]["record"]["status"] == "unranked"


def test_csv_markdown_preserve_units_scope_and_actual_report_cli(renderer, sources, tmp_path, capsys):
    fixture = SyntheticReport(tmp_path)
    rows = renderer.method_rows(fixture.selection(), source_lock=sources)
    csv_rows = list(csv.DictReader(io.StringIO(renderer.render(rows, "csv"))))
    first = csv_rows[0]
    assert float(first["process_peak_rss_bytes"]) == 1024
    assert float(first["client_group_sampled_rss_sum_bytes"]) == 4096
    assert float(first["client_group_os_hwm_lower_bound_bytes"]) == 6144
    contracts = json.loads(first["metric_contracts"])
    assert contracts["process.peak_rss"]["measurement_scope"] == "process_lifetime_high_water"
    assert contracts["client.owned_sampled_peak_rss_sum"]["origin"] == "estimate"
    assert contracts["client.owned_peak_rss_lower_bound"]["origin"] == "lower_bound"
    assert "not authenticity" in first["presentation_limits"]
    assert csv_rows[-1]["full_seconds"] == "unknown"
    selection = tmp_path / "cells.json"
    selection.write_text(json.dumps(fixture.selection()))
    output = tmp_path / "methods.md"
    renderer.main(["--cells", str(selection), "--output", str(output)])
    text = output.read_text()
    assert str(fixture.path) in text and "request_setup_through_completion" in text
    assert "sampled helper-group sums" in text and "not authenticity" in text
    assert "whole-method acceptance" in text and "synthetic-q11" in text
    assert capsys.readouterr().out == ""
