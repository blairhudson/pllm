"""Synthetic orchestration/audit tests; no model, native build or timings."""
from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

import pytest

from pllm.metrics import BuildRef
from benchmarks.research.private_boundary_decode_confirmation import (
    CONTROL, METRICS, NAMES, WINNER, Rejected, audit, matched_suite, preflight, recipes,
)
from test_private_boundary_decode_benchmark import (
    cohort as screen_cohort, change_prepared_reply, ledger,
)


@lru_cache
def archived():
    return json.loads(Path("docs/evidence/private-boundary-decode-qwen25-sdk.json").read_bytes())


def seal_contracts(document):
    """Create genuinely valid v2 records around synthetic observations, not mocks."""
    from pllm import Experiment
    from pllm._benchmark_contract import digest
    from pllm.metrics import BenchmarkRun, BenchmarkSuite
    from pllm.metrics.benchmark_runner import _result
    from pllm.schedulers import _numeric_component
    from pllm.runtime.prepared_accounting import material_accounting

    build = deepcopy(archived()["results"][0]["execution"]["build"])
    for case in document["suite"]["cases"]:
        case["build"]["expected_digest"] = build["digest"]
    suite = BenchmarkSuite.from_spec(document["suite"])
    cases = {case.name: case for case in suite.cases}

    def commitment(value):
        return value if isinstance(value, str) and len(value) == 64 else digest({"synthetic_commitment": value})

    for row in document["results"]:
        execution = row["execution"]
        case = cases[execution["case"]["name"]]
        execution.update(case=case.to_spec(), suite_digest=suite.digest, build=deepcopy(build))
        template = next(r for r in archived()["results"] if r["execution"]["case"]["name"] == execution["case"]["name"])
        for key in ("schema_version", "scope", "warmups", "repetitions", "samples", "environment"):
            row[key] = deepcopy(template[key])
        for key in ("admission", "failure_code"):
            execution[key] = deepcopy(template["execution"][key])
        report = execution["details"]["report"]
        request = report["runs"][0]
        cap = execution["case"]["target"]["parameters"]["scenario"]["max_output_tokens"]
        sampling = {"effective_temperature": 0.0, "mode": "greedy", "requested_temperature": 0.0, "top_p": None}
        report.update(schema_version="pllm.loopback_benchmark.v1", warmup_runs=[], memory_preflight={"admitted": True})
        report["checks"] = deepcopy(template["execution"]["details"]["report"]["checks"])
        report["configuration"].update(max_output_tokens=cap, sampling=sampling, repetitions=1, warmups=0)
        for key in ("source_lock_digest", "prompt_digest"):
            report["configuration"][key] = commitment(report["configuration"][key])
        execution["fixture"].update(kind="checkpoint", digest=report["configuration"]["source_lock_digest"])
        outputs = request["tokens"]["output_tokens"]
        request.update(max_output_tokens=cap, sampling=sampling, status="completed", failure=None,
            warm=False, model_fingerprint=commitment("same-body"))
        request["generation"]["output_text_digest"] = commitment(request["generation"]["output_text_digest"])
        request["generation"]["response_status"] = "completed" if outputs < cap else "incomplete"
        execution["details"]["topology_admission"] = {"admitted": True}
        record = execution["details"]["response_state_audit"]
        record["schema"] = "pllm.response_state_audit.v1"
        for binding in record["bindings"]:
            binding["numeric_body_fingerprint"] = commitment(binding["numeric_body_fingerprint"])
            for selection in binding["selections"]:
                for key in ("logits_and_active_kv", "selected_token", "rng_before", "rng_after"):
                    selection[key] = commitment(selection[key])
        execution["oracle"] = {"kind": "salted_full_logit_kv_sampler_commitments", "outcome": "passed",
            "checked_values": sum(s["checked_values"] for s in record["bindings"][0]["selections"])}
        startup = report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"]
        report["prepared_material_accounting"] = material_accounting({key: startup[key] + value
            for key, value in request["privacy"].items() if key.startswith("prepared_stage_rows_")})
        model = execution["case"]["target"]["parameters"]["model"]
        exp = Experiment.from_spec(model["experiment"]).with_params(pipeline__profile=model["profile"])
        numeric = {name: value.to_spec() for name, value in exp.pipeline.components.items()
            if name in {"quantization", "nonlinear", "scheduler"}
            and (name != "scheduler" or _numeric_component(value.to_spec()) is not None)}
        scenario = execution["case"]["target"]["parameters"]["scenario"]
        execution["comparison"] = {"source_digest": report["configuration"]["source_lock_digest"],
            "numeric_digest": digest({"bodies": [request["model_fingerprint"]], "contract": numeric}),
            "privacy_digest": template["execution"]["comparison"]["privacy_digest"], "state_digest": digest({"lifecycle": scenario}),
            "workload_digest": digest({"tokens": [request["tokens"]], "scenario": scenario,
                "backend": "native", "wan": report["configuration"]["wan_emulation_digest"], "sampling": sampling}),
            "output_digest": digest({"output": digest([request["generation"]["output_text_digest"]]), "state": record["bindings"]}),
            "measurement_digest": template["execution"]["comparison"]["measurement_digest"]}
        row["workload_digest"] = execution["comparison"]["workload_digest"]
        templates = {m["id"]: m for m in template["metrics"]}
        current = {m["id"]: m["value"] for m in row["metrics"]}
        for alias, source in (("request.0.ttft", "response.ttft"), ("request.0.decode_tps", "response.decode_tps")):
            if alias not in current:
                row["metrics"].append({"id": alias, "value": current[source]})
        for index, metric in enumerate(row["metrics"]):
            value = metric["value"]
            metadata = deepcopy(templates[metric["id"]])
            metadata["value"] = value
            if metric["id"] in ("response.decode_tps", "request.0.decode_tps"):
                metadata["denominator"]["value"] = max(outputs - 1, 0)
                if outputs <= 1:
                    metadata.update(value=None, sample_count=0, origin="not_available", unavailable_reason="measurement unavailable")
            row["metrics"][index] = metadata
        # Preserve the real v2 sample/execution mirror and raw measurement axes.
        current = {m["id"]: m["value"] for m in row["metrics"]}
        decode = current["response.decode_tps"]
        request["durations"] = {"full_seconds": current["request.0.full"], "ttft_seconds": current["response.ttft"],
            "generation_seconds": (outputs - 1) / decode if outputs > 1 else 0.0, "tokens_per_second": decode}
        report["process_cpu_accounting"] = {"cold_first_response_cpu_seconds_by_role": {
            role: current[f"{role}.cold_cpu"] for role in ("client", "preparation")}}
        request["processes"] = {"preparation": {"rss_peak_bytes": current["preparation.peak_rss"]}}
        row["samples"][0]["duration_seconds"] = current["target.latency"]
        detail = execution["details"]
        detail["timings"] = {"wall_seconds": [current["target.latency"]], "cpu_seconds": [1.0]}
        detail["model"] = {"id": exp.pipeline.model.source, "checkpoint_digest": None,
            "source_lock_digest": report["configuration"]["source_lock_digest"]}
        payload = {"status": row["status"], "failure_code": execution["failure_code"], "comparison": execution["comparison"],
            "details": detail, "environment": row["environment"]["attributes"], "metrics": row["metrics"], "samples": row["samples"],
            "fixture": execution["fixture"], "oracle": execution["oracle"], "admission": execution["admission"],
            "limitations": ["Synthetic validation fixture; not model execution or measured evidence."]}
        valid = _result(case, suite, execution["replicate"], document["cohort"], build, payload).to_dict()
        row.clear()
        row.update(valid)
    BenchmarkRun(document)  # Fixture contract must validate before any tamper.
    return document


def adapt(document, cap=8, outputs=None):
    """Build synthetic workload evidence from existing raw-counter test fixture."""
    document = deepcopy(document)
    suite = matched_suite(build=BuildRef(expected_digest="a" * 64), max_output_tokens=cap)
    document["suite"] = suite.to_spec()
    document["cohort"] = "c" * 64
    cases = {case.name: case.to_spec() for case in suite.cases}
    document["results"] = [row for row in document["results"] if row["execution"]["case"]["name"] in NAMES]
    outputs = cap if outputs is None else outputs
    selections = outputs + (outputs < cap)
    for row in document["results"]:
        execution = row["execution"]
        execution.update(case=cases[execution["case"]["name"]], suite_digest=suite.digest,
            sampling=suite.sampling.to_spec(), cohort=document["cohort"])
        detail = execution["details"]
        binding = detail["response_state_audit"]["bindings"][0]
        template = binding["selections"][0]
        binding["selections"] = [template | {"position": 150 + index, "cache_lengths": [150 + index]}
            for index in range(selections)]
        request = detail["report"]["runs"][0]
        request["tokens"]["output_tokens"] = outputs
        request["privacy"].update(generation_requested_output_tokens=cap,
            generation_executed_decode_steps=selections - 1, generation_output_tokens=outputs,
            private_boundary_head_calls=selections, private_boundary_lookup_rows=150 + selections - 1)
    return seal_contracts(document)


@pytest.fixture
def cohort(screen_cohort):
    return adapt(screen_cohort)


@pytest.mark.parametrize("cap", (8, 32, 64))
def test_independent_single_workload_contract_and_audit(screen_cohort, cap):
    document = adapt(screen_cohort, cap)
    before = deepcopy(document)
    summary = audit(document, max_output_tokens=cap)
    assert document == before
    assert summary["clients"] == 9 and summary["replicates"] == 3
    assert summary["output_cap"] == cap and summary["long_output_cap_reached"]
    assert tuple(summary["groups"]) == NAMES
    assert set(summary["comparisons"][WINNER]) == {CONTROL, "maintained"}
    assert summary["groups"][WINNER]["maximum_client_peak_bytes"] == 300 << 20
    assert not summary["groups"][WINNER]["observed_tiers"]["low"]
    assert summary["body_reconciliation"][WINNER][0]["head_reply_delta_bytes"] == -1000
    work = summary["groups"][WINNER]["executed_work"][0]
    assert work["output_tokens"] == cap and work["executed_decode_steps"] == cap - 1
    assert work["preparation_rows"] == 100
    with pytest.raises(Rejected):
        audit(document, max_output_tokens=32 if cap == 8 else 8)


@pytest.mark.parametrize("outputs", (1, 7, 15, 31))
def test_natural_eos_records_final_selection_without_forcing_cap(screen_cohort, outputs):
    summary = audit(adapt(screen_cohort, 32, outputs), max_output_tokens=32)
    assert not summary["long_output_cap_reached"]
    work = summary["groups"][WINNER]["executed_work"][0]
    assert work["output_tokens"] == outputs
    assert work["sampled_selections"] == outputs + 1
    assert work["executed_decode_steps"] == outputs
    assert work["terminated_before_cap"]
    if outputs == 1:
        comparison = summary["comparisons"][WINNER][CONTROL]["response.decode_tps"]
        assert comparison["candidate_wins"] is None and comparison["delta"] is None
        assert comparison["paired_deltas"] == [None] * 3


def test_zero_output_telemetry_explicitly_unsupported(screen_cohort):
    with pytest.raises(Rejected, match="zero_output_telemetry_unsupported"):
        audit(adapt(screen_cohort, 32, 0), max_output_tokens=32)


def test_fixture_is_valid_v2_before_tampering(cohort):
    from pllm.metrics import BenchmarkRun
    assert BenchmarkRun(cohort).to_dict() == cohort


@pytest.mark.parametrize("mutation", ("tps_aliases", "all_ttft_windows", "bogus_build", "one_output_measured_zero",
    "one_output_alias_zero", "sample_duration", "case_digest", "target_digest"))
def test_pending_fix_rejects_actual_record_tampering(screen_cohort, mutation):
    """Run after parent applies the staged source patch; no xfail or bypass."""
    from pllm.metrics import BenchmarkRun
    document = adapt(screen_cohort, 32, 1 if mutation.startswith("one_output") else 8)
    row = document["results"][-1]
    if mutation == "bogus_build":
        row["execution"]["build"]["attributes"] = {"bogus": True}
    elif mutation == "sample_duration":
        row["samples"][0]["duration_seconds"] += 1
    elif mutation == "case_digest":
        row["execution"]["case_digest"] = "0" * 64
    elif mutation == "target_digest":
        row["configuration_digest"] = "0" * 64
    elif mutation == "all_ttft_windows":
        for sample in document["results"]:
            for metric in sample["metrics"]:
                if metric["id"] in ("request.0.ttft", "response.ttft"):
                    metric["window"] = "after_first_output"
        BenchmarkRun(document)  # Syntactically valid v2, semantically wrong axis.
    else:
        for metric in row["metrics"]:
            aliases = ("request.0.decode_tps",) if mutation == "one_output_alias_zero" else (
                "request.0.decode_tps", "response.decode_tps")
            if metric["id"] in aliases:
                metric.update(value=123456789 if mutation == "tps_aliases" else 0,
                    origin="measured", sample_count=1, unavailable_reason=None)
    with pytest.raises(Rejected):
        audit(document, max_output_tokens=32)


def eos_inventory(screen_cohort, *, discarded=0):
    document = adapt(screen_cohort, 32, 7)
    for row in document["results"]:
        report = row["execution"]["details"]["report"]
        privacy = report["runs"][0]["privacy"]
        privacy.update(prepared_stage_rows_claimed=80 - discarded,
            prepared_stage_rows_burned=20, prepared_stage_rows_reserved=100 - discarded,
            prepared_stage_rows_discarded=discarded)
        report["topology_accounting"]["runs"][0] = ledger(
            row["execution"]["case"]["name"] != "maintained", privacy)
    return seal_contracts(document)


@pytest.mark.parametrize("discarded", (0, 5))
def test_real_raw_body_paths_conserve_unused_eos_material(screen_cohort, discarded):
    from benchmarks.research import private_boundary_decode as screen
    from benchmarks.research.private_boundary_decode_confirmation import _eos_body

    document = eos_inventory(screen_cohort, discarded=discarded)
    before = deepcopy(document)
    for row in document["results"]:
        report = row["execution"]["details"]["report"]
        metrics = {m["id"]: m["value"] for m in row["metrics"]}
        boundary = row["execution"]["case"]["name"] != "maintained"
        # Exercise the REAL frozen reconciliation, not a mock. It rejects
        # conserved EOS inventory because it explicitly assumes no unused rows.
        with pytest.raises(Rejected, match="decode_prepared_row_or_material_work_invalid"):
            screen._body(report, metrics, boundary)
        actual = _eos_body(report, metrics, boundary, chunk_rows=16)
        assert actual["fixed_work"]["prepared_stage_rows_burned"] == 20
        assert actual["fixed_work"]["prepared_stage_rows_claimed"] == 80 - discarded
        assert actual["total"] == metrics["network.covered_body_bytes"]
    summary = audit(document, max_output_tokens=32)
    work = summary["groups"][WINNER]["executed_work"][0]
    assert work["prepared_stage_rows_burned"] == 20 and work["prepared_stage_rows_discarded"] == discarded
    assert work["preparation_rows"] == 100 and work["prepared_stage_rows_claimed"] == 80 - discarded
    assert summary["body_reconciliation"][WINNER][0]["head_reply_delta_bytes"] == -1000
    assert document == before  # No sanitized inventory/ledger view is passed anywhere.


@pytest.mark.parametrize("mutation", ("missing_material", "false_conserved", "forged_counts", "row_loss",
    "reserved_loss", "stage_bytes", "body_total", "raw_startup", "burned_arm", "tcp"))
def test_eos_cannot_hide_missing_rows_material_or_wire(screen_cohort, mutation):
    document = eos_inventory(screen_cohort)
    row = document["results"][-1]
    report = row["execution"]["details"]["report"]
    privacy = report["runs"][0]["privacy"]
    if mutation == "missing_material":
        del report["prepared_material_accounting"]
    elif mutation == "false_conserved":
        report["prepared_material_accounting"]["conserved"] = False
    elif mutation == "forged_counts":
        report["prepared_material_accounting"]["counts"]["burned"] -= 1
    elif mutation in ("row_loss", "reserved_loss", "burned_arm"):
        privacy["prepared_stage_rows_" + ("reserved" if mutation == "reserved_loss" else "burned")] -= 1
        if mutation == "burned_arm":
            privacy["prepared_stage_rows_claimed"] += 1
            seal_contracts(document)  # Locally conserved, but unequal boundary-arm work.
    elif mutation == "stage_bytes":
        report["topology_accounting"]["stages"]["runs"][0]["body_bytes_by_stage_and_edge"]["layer.0"]["inference->client"] += 1
    elif mutation == "body_total":
        report["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"] -= 1
    elif mutation == "raw_startup":
        del report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"]["correction_ack_bytes"]
    else:
        report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]["directed_stream_bytes"]["inference->client"] = 0
    with pytest.raises(Rejected):
        audit(document, max_output_tokens=32)


def test_unused_rows_are_not_allowed_when_output_cap_reached(cohort):
    for row in cohort["results"]:
        report = row["execution"]["details"]["report"]
        report["runs"][0]["privacy"].update(prepared_stage_rows_claimed=80, prepared_stage_rows_burned=20)
    seal_contracts(cohort)
    with pytest.raises(Rejected, match="decode_prepared_row_or_material_work_invalid"):
        audit(cohort)


def test_eos_server_ns_remains_charged_and_requires_stage_proof(screen_cohort):
    document = eos_inventory(screen_cohort)
    change_prepared_reply(document["results"][-1], 2)
    paired = audit(document, max_output_tokens=32)["body_reconciliation"][WINNER][-1]
    assert paired["prepared_stage_reply_delta_bytes"] == 2
    assert paired["covered_body_delta_bytes"] == -968
    broken = eos_inventory(screen_cohort)
    change_prepared_reply(broken["results"][-1], 2, update_stage_evidence=False)
    with pytest.raises(Rejected, match="eos_stage_body_evidence_unreconciled"):
        audit(broken, max_output_tokens=32)


def test_archive_raw_reports_and_strict_eos_paths_agree_without_unused_rows():
    """Offline replay only: new wrapper metadata is synthetic, not new evidence."""
    from pllm.metrics import BenchmarkSuite
    from benchmarks.research import private_boundary_decode as screen
    from benchmarks.research.private_boundary_decode_confirmation import _eos_body

    document = deepcopy(archived())
    old_suite = BenchmarkSuite.from_spec(document["suite"])
    suite = matched_suite(build=old_suite.cases[0].build)
    cases = {case.name: case.to_spec() for case in suite.cases}
    document["suite"] = suite.to_spec()
    document["results"] = [row for row in document["results"] if row["execution"]["case"]["name"] in NAMES]
    for row in document["results"]:
        execution = row["execution"]
        name = execution["case"]["name"]
        execution.update(case=cases[name], suite_digest=suite.digest, sampling=suite.sampling.to_spec())
        report = execution["details"]["report"]
        metrics = {m["id"]: m["value"] for m in row["metrics"]}
        before = deepcopy(report)
        assert _eos_body(report, metrics, name != "maintained", chunk_rows=16) == screen._body(
            report, metrics, name != "maintained", chunk_rows=16)
        assert before == report
    summary = audit(document)
    assert summary["clients"] == 9 and summary["all_logits_active_kv_positions_selections_rng_and_outputs_match"]


@pytest.mark.parametrize("mutation", (
    "incomplete", "missing", "duplicate", "old_salt", "mixed_build", "state", "history", "rng",
    "output", "source", "body", "startup_raw", "wire", "owners", "swap", "guard", "metric",
    "duplicate_metric", "selection_count", "steps", "head_calls", "lookup_rows", "selection_position",
    "changed_budget", "changed_serializer_recipe", "state_audit_off", "sampling", "failed",
))
def test_fail_closed(cohort, mutation):
    row = cohort["results"][-1]
    execution = row["execution"]
    detail = execution["details"]
    report = detail["report"]
    binding = detail["response_state_audit"]["bindings"][0]
    privacy = report["runs"][0]["privacy"]
    if mutation == "incomplete":
        cohort["complete"] = False
    elif mutation == "missing":
        cohort["results"].pop()
    elif mutation == "duplicate":
        execution["replicate"] = 0
    elif mutation == "old_salt":
        execution["cohort"] = "b" * 64
    elif mutation == "mixed_build":
        execution["build"]["digest"] = "d" * 64
    elif mutation in ("state", "rng", "history"):
        key = {"state": "logits_and_active_kv", "rng": "rng_after", "history": "selected_token"}[mutation]
        binding["selections"][-1][key] = "different"
    elif mutation == "output":
        report["runs"][0]["generation"]["output_text_digest"] = "different"
    elif mutation == "source":
        execution["fixture"]["digest"] = "different"
    elif mutation == "body":
        privacy["inference_upload_bytes"] += 1
    elif mutation == "startup_raw":
        del report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"][
            "role_link.boundary_a.setup_upload_bytes"]
    elif mutation == "wire":
        report["native_network_accounting"]["samples"]["runs"][0]["after"]["wan_emulation"]["backend"] = "fake"
    elif mutation == "owners":
        detail["response_state_audit"]["client_memory_owners"].remove("audit")
    elif mutation == "swap":
        report["memory_guard"]["maximum_host_swap_growth_bytes"] = 1
    elif mutation == "guard":
        report["memory_guard"]["tripped"] = True
    elif mutation == "metric":
        next(m for m in row["metrics"] if m["id"] == "target.latency")["value"] = float("nan")
    elif mutation == "duplicate_metric":
        row["metrics"].append(deepcopy(row["metrics"][0]))
    elif mutation == "selection_count":
        binding["selections"].pop()
    elif mutation == "steps":
        privacy["generation_executed_decode_steps"] -= 1
    elif mutation == "head_calls":
        privacy["private_boundary_head_calls"] -= 1
    elif mutation == "lookup_rows":
        privacy["private_boundary_lookup_rows"] -= 1
    elif mutation == "selection_position":
        binding["selections"][-1]["position"] += 1
    elif mutation == "changed_budget":
        cohort["suite"]["cases"][-1]["target"]["parameters"]["model"]["experiment"]["budget"]["max_new_tokens"] = 32
    elif mutation == "changed_serializer_recipe":
        cohort["suite"]["cases"][-1]["target"]["parameters"]["model"]["experiment"]["pipeline"]["components"][
            "boundary"]["params"]["head_output_encoding"] = "u32"
    elif mutation == "state_audit_off":
        cohort["suite"]["cases"][-1]["target"]["parameters"]["state_audit"] = False
    elif mutation == "sampling":
        cohort["suite"]["sampling"]["processes"] = 1
    else:
        row["status"] = "failed"
    with pytest.raises(Rejected):
        audit(cohort)


@pytest.mark.parametrize("mutation", ("result_schema", "report_schema", "failure_code", "admission", "topology_admission",
    "memory_admission", "oracle", "warmups", "report_warmups", "warmup_rows", "request_warm", "request_failed",
    "config_cap", "request_cap", "sampler", "request_sampler", "check_false", "check_missing", "eos_status",
    "environment", "numeric", "state", "workload", "output", "measurement", "privacy", "unit", "scope", "denominator"))
def test_execution_comparison_environment_and_metric_contracts(cohort, mutation):
    from pllm.evidence import environment_digest

    row = cohort["results"][-1]
    execution = row["execution"]
    report = execution["details"]["report"]
    request = report["runs"][0]
    if mutation == "result_schema":
        row["schema_version"] = "old"
    elif mutation == "report_schema":
        report["schema_version"] = "old"
    elif mutation == "failure_code":
        execution["failure_code"] = "failure"
    elif mutation == "admission":
        execution["admission"]["admitted"] = False
    elif mutation == "topology_admission":
        execution["details"]["topology_admission"]["admitted"] = False
    elif mutation == "memory_admission":
        report["memory_preflight"]["admitted"] = False
    elif mutation == "oracle":
        execution["oracle"]["checked_values"] -= 1
    elif mutation == "warmups":
        row["warmups"] = 1
    elif mutation == "report_warmups":
        report["configuration"]["warmups"] = 1
    elif mutation == "warmup_rows":
        report["warmup_runs"] = [deepcopy(request)]
    elif mutation == "request_warm":
        request["warm"] = True
    elif mutation == "request_failed":
        request["failure"] = "failure"
    elif mutation in ("config_cap", "request_cap"):
        (report["configuration"] if mutation == "config_cap" else request)["max_output_tokens"] = 32
    elif mutation in ("sampler", "request_sampler"):
        (report["configuration"] if mutation == "sampler" else request)["sampling"] = {"mode": "greedy", "effective_temperature": 1}
    elif mutation == "check_false":
        report["checks"]["no_plaintext_token_ids_sent"] = False
    elif mutation == "check_missing":
        del report["checks"]["no_plaintext_token_ids_sent"]
    elif mutation == "eos_status":
        request["generation"]["response_status"] = "completed"
    elif mutation == "environment":
        row["environment"]["attributes"]["hardware"]["logical_cpus"] += 1
        row["environment"]["digest"] = environment_digest(row["environment"]["attributes"])
    elif mutation in ("numeric", "state", "workload", "output", "measurement", "privacy"):
        execution["comparison"][mutation + "_digest"] = "changed"
    else:
        metric = next(m for m in row["metrics"] if m["id"] == "response.decode_tps")
        if mutation == "unit":
            metric["unit"] = "bytes"
        elif mutation == "scope":
            metric["measurement_scope"] = "child_only"
        else:
            metric["denominator"]["value"] = 8
    with pytest.raises(Rejected):
        audit(cohort)


def test_server_ns_variation_remains_charged(cohort):
    change_prepared_reply(cohort["results"][-1], 2)
    paired = audit(cohort)["body_reconciliation"][WINNER][-1]
    assert paired["prepared_stage_reply_delta_bytes"] == 2
    assert paired["covered_body_delta_bytes"] == -968
    assert paired["bootstrap_and_metadata_residual_bytes"] == 30


@pytest.mark.parametrize("delta,independent", ((2, False), (10000, True)))
def test_cannot_excuse_unreconciled_or_unbounded_metadata(cohort, delta, independent):
    change_prepared_reply(cohort["results"][-1], delta, update_stage_evidence=independent)
    with pytest.raises(Rejected):
        audit(cohort)


def test_exact_three_and_tiny_bounds():
    with pytest.raises(ValueError, match="three"):
        matched_suite(build=BuildRef(), processes=1)
    with pytest.raises(ValueError):
        recipes(max_output_tokens=16)
    with pytest.raises(ValueError):
        recipes(tiny=True, max_output_tokens=32)
    suite = matched_suite(build=BuildRef(), tiny=True)
    assert len(suite.cases) == 3
    for case in suite.cases:
        parameters = case.target.to_spec()["parameters"]
        assert parameters["state_audit"] is True
        assert parameters["model"]["experiment"]["budget"] == {
            "max_input_tokens": 128, "max_new_tokens": 2, "requests": 1}


def test_tiny_full_audit_has_independent_smoke_scope(cohort):
    suite = matched_suite(build=BuildRef(expected_digest="a" * 64), tiny=True)
    cohort["suite"] = suite.to_spec()
    cases = {case.name: case.to_spec() for case in suite.cases}
    for row in cohort["results"]:
        execution = row["execution"]
        execution.update(case=cases[execution["case"]["name"]], suite_digest=suite.digest)
        detail = execution["details"]
        request = detail["report"]["runs"][0]
        request["tokens"].update(input_tokens=100, output_tokens=2)
        request["privacy"].update(generation_requested_output_tokens=2, generation_output_tokens=2,
            generation_executed_decode_steps=1, private_boundary_head_calls=2, private_boundary_lookup_rows=101)
        binding = detail["response_state_audit"]["bindings"][0]
        binding["selections"] = [selection | {"position": 100 + index, "cache_lengths": [100 + index]}
            for index, selection in enumerate(binding["selections"][:2])]
    seal_contracts(cohort)
    summary = audit(cohort, tiny=True)
    assert summary["tiny"] and summary["output_cap"] == 2
    assert "128+2" in summary["scope"]["request"]
    with pytest.raises(Rejected):
        audit(cohort)


def test_preflight_keeps_conservative_admission_and_audit_charge(monkeypatch):
    import pllm.metrics
    import pllm.runtime.benchmark_response_audit as state_audit
    seen, audits = [], []

    def memory(experiment, **kwargs):
        seen.append((experiment.budget.max_new_tokens, kwargs))
        return {"admitted": False}

    def scratch(report, memory_bytes, *, enforced_wan):
        audits.append((memory_bytes, enforced_wan))
        return report

    monkeypatch.setattr(pllm.metrics, "benchmark_memory", memory)
    monkeypatch.setattr(state_audit, "state_audit_admission", scratch)
    result = preflight(matched_suite(build=BuildRef(), max_output_tokens=64))
    assert len(result) == 3 and all(not row["admission"]["admitted"] for row in result)
    assert all(cap == 64 and kwargs == {"backend": "native", "enforced_wan": True,
        "memory_budget_bytes": 8 << 30} for cap, kwargs in seen)
    assert audits == [(8 << 30, True)] * 3


def test_cli_keeps_existing_attempts_and_rejects_mixed_tiny(tmp_path):
    from scripts.benchmark_private_boundary_decode_confirmation import main
    output = tmp_path / "attempt.json"
    output.write_text("original")
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
    assert output.read_text() == "original"
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "new.json"), "--tiny", "--max-output-tokens", "32"])


@pytest.fixture
def cli(monkeypatch):
    from scripts import benchmark_private_boundary_decode_confirmation as cli

    class Lock:
        def __init__(self, path, timeout):
            assert path.endswith("frontier-probe.lock") and timeout == 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def forbidden(*args, **kwargs):
        raise AssertionError("no model/timing execution allowed")

    monkeypatch.setattr(cli, "FileLock", Lock)
    monkeypatch.setattr(cli, "hashes", lambda paths: {"fixture": "frozen"})
    monkeypatch.setattr(cli, "read_fixture", lambda path: ("public prompt", "public hash"))
    monkeypatch.setattr(cli, "public_input_counts", lambda *args: ({"150": 150}, "source"))
    build_digest = archived()["results"][0]["execution"]["build"]["digest"]
    monkeypatch.setattr(cli.BuildRef, "capture", lambda: BuildRef(expected_digest=build_digest))
    monkeypatch.setattr(cli, "preflight", lambda suite: [
        {"case": case.name, "admission": {"admitted": True}} for case in suite.cases])
    monkeypatch.setattr(cli, "benchmark_suite", forbidden)
    return cli


def test_cli_preflight_never_runs_benchmark_and_keeps_cap(cli, tmp_path):
    import json
    output = tmp_path / "preflight.json"
    assert cli.main(["--output", str(output), "--max-output-tokens", "64", "--preflight-only"]) == 0
    assert not output.exists()
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    assert plan["output_cap"] == 64 and len(plan["suite"]["cases"]) == 3
    progress = json.loads(output.with_suffix(".progress.json").read_text())
    assert progress["status"] == "admitted" and not progress["measurement_complete"]
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output), "--preflight-only"])


def test_cli_rejected_attempt_retains_original_report_without_summary(cli, monkeypatch, tmp_path, cohort):
    import json
    from types import SimpleNamespace

    output = tmp_path / "attempt.json"
    cohort["complete"] = False

    def run(suite, *, inputs, output):
        assert tuple(inputs) == NAMES
        output.write_text(json.dumps(cohort))
        return SimpleNamespace(to_dict=lambda: cohort)

    monkeypatch.setattr(cli, "benchmark_suite", run)
    assert cli.main(["--output", str(output)]) == 1
    assert json.loads(output.read_text()) == cohort
    assert not output.with_suffix(".summary.json").exists()
    progress = json.loads(output.with_suffix(".progress.json").read_text())
    assert progress["status"] == "rejected" and progress["failure_code"] == "confirmation_incomplete_cohort"
    assert not progress["measurement_complete"]


def test_cli_busy_lock_fails_before_preflight(cli, monkeypatch, tmp_path):
    import json
    from filelock import Timeout

    def busy(*args, **kwargs):
        raise Timeout("frontier-probe.lock")

    monkeypatch.setattr(cli, "FileLock", busy)
    output = tmp_path / "busy.json"
    assert cli.main(["--output", str(output), "--preflight-only"]) == 1
    progress = json.loads(output.with_suffix(".progress.json").read_text())
    assert progress["failure_code"] == "frontier_cohort_lock_busy"
    assert not output.exists() and not output.with_suffix(".plan.json").exists()


def test_cli_exclusive_reservation_does_not_overwrite_racing_attempt(cli, monkeypatch, tmp_path):
    output = tmp_path / "race.json"
    paths = (output.with_suffix(".plan.json"), output.with_suffix(".progress.json"),
        output.with_suffix(".summary.json"))
    paths[1].write_text("other worker")
    monkeypatch.setattr(cli, "fresh_output_paths", lambda output: paths)
    with pytest.raises(FileExistsError):
        cli.main(["--output", str(output), "--preflight-only"])
    assert paths[1].read_text() == "other worker"


def test_cli_completed_audit_must_match_planned_source(cli, monkeypatch, tmp_path, cohort):
    import json
    from types import SimpleNamespace

    def run(suite, *, inputs, output):
        output.write_text(json.dumps(cohort))
        return SimpleNamespace(to_dict=lambda: cohort)

    monkeypatch.setattr(cli, "benchmark_suite", run)
    rejected = tmp_path / "source-mismatch.json"
    assert cli.main(["--output", str(rejected)]) == 1
    progress = json.loads(rejected.with_suffix(".progress.json").read_text())
    assert progress["failure_code"] == "planned_source_lock_mismatch"
    assert rejected.exists() and not rejected.with_suffix(".summary.json").exists()
    source = cohort["results"][0]["execution"]["details"]["report"]["configuration"]["source_lock_digest"]
    monkeypatch.setattr(cli, "public_input_counts", lambda *args: ({"150": 150}, source))
    accepted = tmp_path / "accepted.json"
    assert cli.main(["--output", str(accepted)]) == 0
    assert json.loads(accepted.with_suffix(".progress.json").read_text())["measurement_complete"]
