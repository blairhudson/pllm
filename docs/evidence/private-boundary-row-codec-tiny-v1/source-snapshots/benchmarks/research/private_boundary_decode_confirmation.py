"""Independent three-arm confirmation, one output workload per fresh cohort.

Reuse the frozen screen's generic raw-body/TCP reconciliation, not its fixed
eight-selection or five-cell admission. Configurations never import observations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

from pllm import Experiment
from pllm.deployment import WanConditions
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, Scenario

from . import private_boundary_decode as screen
from . import private_boundary_tiers as tiers
from .sota_combinations import Rejected

CONTROL = screen.CONTROL
WINNER = "coupled-parallel-uniform_residues"
NAMES = ("maintained", CONTROL, WINNER)
METRICS = screen.METRICS
preflight = tiers.preflight
_require = screen._require


def _eos_body(report, metrics, boundary, *, chunk_rows):
    """Bounded fork of screen._body for EOS inventory, never rewritten evidence.

    The frozen helper combines raw reconstruction with a no-unused-rows rule.
    Keep that helper unchanged for at-cap runs; this path changes only inventory
    conservation. Every byte, owner, stage and TCP check still uses original raw
    evidence, including prepared material that was issued but never claimed.
    """
    from pllm.runtime.topology_accounting import (
        ACK_COVERAGE, BOUNDARY_COVERAGE, body_coverage_identity,
        prepared_body_accounting, private_boundary_body_accounting,
    )
    count, counters = screen._count, screen._counter_values
    accounting = report["topology_accounting"]
    _require(accounting["warmups"] == [] and len(accounting["runs"]) == 1, "eos_body_windows_changed")
    windows = [accounting["startup"], *accounting["runs"]]
    account = private_boundary_body_accounting if boundary else prepared_body_accounting
    rebuilt = []

    def edges(ledger):
        rows = ledger["body_bytes_by_edge"]
        _require(isinstance(rows, list) and len(rows) <= 32 and all(
            set(row) == {"source", "destination", "phase", "serialized_body_bytes"}
            and count(row["serialized_body_bytes"]) for row in rows), "eos_body_edges_invalid")
        result = {(row["source"], row["destination"], row["phase"]): row["serialized_body_bytes"] for row in rows}
        _require(len(result) == len(rows), "eos_body_edges_duplicate")
        return result

    for index, window in enumerate(windows):
        coverage = BOUNDARY_COVERAGE if boundary else ACK_COVERAGE
        _require(body_coverage_identity(window) == coverage
            and count(window.get("all_link_serialized_body_bytes"))
            and count(window.get("lifecycle_control_serialized_body_bytes"))
            and window["lifecycle_control_serialized_body_bytes"] <= window["all_link_serialized_body_bytes"]
            and window.get("total_wire_bytes") is None, "eos_body_scope_incomplete")
        privacy = window["traffic_attribution"]["audit_counters"] if index == 0 else report["runs"][0]["privacy"]
        _require(isinstance(privacy, dict), "eos_raw_body_evidence_missing")
        prepared = prepared_body_accounting({"privacy": privacy}, initial_preparation=index == 0)
        fresh = account({"privacy": privacy}, initial_preparation=index == 0)
        _require(body_coverage_identity(prepared) == ACK_COVERAGE
            and prepared["tracked_body_counter_set_present"] is True
            and body_coverage_identity(fresh) == coverage
            and fresh["tracked_body_counter_set_present"] is True, "eos_raw_body_coverage_incomplete")
        _require(edges(window) == edges(fresh), "eos_body_edges_differ_from_raw")
        for key in ("all_link_serialized_body_bytes", "client_serialized_body_bytes",
                    "online_all_link_serialized_body_bytes", "online_client_serialized_body_bytes",
                    "lifecycle_control_serialized_body_bytes"):
            _require(count(window.get(key)) and window[key] == fresh[key], "eos_body_totals_differ_from_raw")
        rebuilt.append(fresh)
    samples = report["native_network_accounting"]["samples"]
    before = samples["runs"][0]["before"]["wan_emulation"]["directed_stream_bytes"]
    after = samples["runs"][0]["after"]["wan_emulation"]["directed_stream_bytes"]
    bounds = [samples["startup"]["wan_emulation"]["directed_stream_bytes"],
        {edge: value - before.get(edge, 0) for edge, value in after.items()}]
    for ledger, bound in zip(rebuilt, bounds):
        directed = {}
        for edge in ledger["body_bytes_by_edge"]:
            link = f'{edge["source"]}->{edge["destination"]}'
            directed[link] = directed.get(link, 0) + edge["serialized_body_bytes"]
        _require(all(value <= bound.get(link, 0) for link, value in directed.items()),
                 "eos_body_exceeds_matching_tcp_window")
    total = sum(window["all_link_serialized_body_bytes"] for window in windows)
    _require(total == metrics["network.covered_body_bytes"], "eos_body_metric_differs")
    privacy = report["runs"][0]["privacy"]
    result = {"total": total, "controls": sum(w["lifecycle_control_serialized_body_bytes"] for w in windows),
        "prepared": counters(privacy, screen.PREPARED_COUNTERS),
        "fixed_work": counters(privacy, screen.FIXED_WORK_COUNTERS),
        "prepared_reply": counters(privacy, ("inference_download_bytes",))["inference_download_bytes"]}
    work = result["fixed_work"]
    rows, calls = result["prepared"]["prepared_stage_rows_issued"], work["inference_stage_calls"]
    claimed, burned, discarded, reserved = (work[f"prepared_stage_rows_{key}"]
        for key in ("claimed", "burned", "discarded", "reserved"))
    _require(type(chunk_rows) is int and chunk_rows > 0 and 0 < calls <= claimed <= rows
        and rows == work["preparation_rows"] == claimed + burned + discarded
        and reserved == claimed + burned and reserved <= rows
        and rows == sum(work[key] for key in ("preparation_prefill_rows_issued",
            "preparation_decode_rows_issued", "preparation_padding_rows_issued"))
        and work["preparation_prefill_material_body_bytes"] + work["preparation_decode_material_body_bytes"]
            <= work["preparation_attributed_body_bytes"]
        == sum(privacy[key] for key in ("preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes")),
        "eos_prepared_row_or_material_work_invalid")
    stages = accounting["stages"]["runs"]
    _require(len(stages) == 1, "eos_stage_body_evidence_missing")
    stage, = stages
    measured = stage["body_bytes_by_stage_and_edge"]
    fields = {"client->inference": "inference_upload_bytes", "inference->client": "inference_download_bytes",
        "client->preparation": "preparation_upload_bytes", "preparation->client": "preparation_download_bytes",
        "preparation->inference": "correction_push_bytes"}
    _require(bool(measured) and len(measured) <= 4096 and all(set(edges) == set(fields)
        and all(count(value) for value in edges.values()) for edges in measured.values()), "eos_stage_body_evidence_missing")
    sums = {edge: sum(edges[edge] for edges in measured.values()) for edge in fields}
    _require(sums == {edge: privacy[field] for edge, field in fields.items()}
        == stage["measured_body_bytes_by_edge"] == stage["expected_stage_body_bytes_by_edge"]
        and stage["reconciled_with_protocol_bodies"] is True, "eos_stage_body_evidence_unreconciled")
    result["stage_bodies"] = measured
    # Only claimed rows can generate inference replies. Unused material stays
    # in body totals, never used to inflate timing-metadata compatibility.
    result["reply_metadata_delta_bound_bytes"] = 8 * ((claimed + chunk_rows - 1) // chunk_rows + calls - 1)
    if boundary:
        _require(type(privacy.get("private_boundary_body_coverage_failures")) is int
            and privacy["private_boundary_body_coverage_failures"] == 0, "eos_boundary_body_coverage_failed")
        result["boundary_fixed"] = counters(privacy, screen.BOUNDARY_COUNTERS)
        result["head_reply"] = counters(privacy, ("private_boundary_head_download_bytes",))["private_boundary_head_download_bytes"]
        online = counters(privacy, tuple(f"role_link.{role}.online_{direction}_bytes"
            for role in ("boundary_a", "boundary_b") for direction in ("upload", "download")))
        _require(result["head_reply"] > 0 and sum(online.values()) == result["head_reply"]
            + sum(result["boundary_fixed"][key] for key in screen.BOUNDARY_COUNTERS if key.endswith("_bytes")),
            "eos_boundary_online_body_does_not_reconcile")
    return result


def _material(report):
    """Recompute sealed post-close inventory from startup plus request counters."""
    from pllm.runtime.prepared_accounting import material_accounting
    windows = [report["topology_accounting"]["startup"]["traffic_attribution"]["audit_counters"],
        report["runs"][0]["privacy"]]
    keys = tuple(f"prepared_stage_rows_{key}" for key in ("issued", "reserved", "claimed", "burned", "discarded"))
    _require(all(screen._count(window.get(key)) for window in windows for key in keys),
             "confirmation_material_raw_missing")
    fresh = material_accounting({key: sum(window[key] for window in windows) for key in keys})
    _require(fresh["conserved"] is True and report["prepared_material_accounting"] == fresh,
             "confirmation_material_not_conserved")


def _compare(control, candidate):
    result = {}
    for metric, direction in METRICS.items():
        left, right = control["medians"][metric], candidate["medians"][metric]
        known = left is not None and right is not None
        result[metric] = {"control": left, "candidate": right,
            "delta": right - left if known else None,
            "improvement_fraction": ((1 - right / left) if direction == "min" else right / left - 1)
                if known and left else None,
            "candidate_wins": (right < left if direction == "min" else right > left) if known else None,
            "paired_deltas": [b[metric] - a[metric] if a[metric] is not None and b[metric] is not None else None
                for a, b in zip(control["samples"], candidate["samples"])]}
    return result


def output_cap(max_output_tokens=8, *, tiny=False):
    if type(max_output_tokens) is not int or max_output_tokens not in (8, 32, 64):
        raise ValueError("output workload must be 8, 32 or 64")
    if tiny and max_output_tokens != 8:
        raise ValueError("tiny smoke uses 128+2 only; no production workload selection")
    return 2 if tiny else max_output_tokens


def recipes(*, max_output_tokens=8, tiny=False,
            artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    cap = output_cap(max_output_tokens, tiny=tiny)
    values = screen.recipes(tiny=tiny, artifacts=artifacts, generated_source=generated_source)
    return {name: values[name].with_params(budget__max_new_tokens=cap) for name in NAMES}


def matched_suite(*, build, max_output_tokens=8, processes=3, tiny=False,
                  artifacts=Path("target/prepared-frontier-artifacts"), generated_source=None):
    if processes != 3:
        raise ValueError("exactly three balanced fresh replicates required")
    cap = output_cap(max_output_tokens, tiny=tiny)
    values = recipes(max_output_tokens=max_output_tokens, tiny=tiny,
        artifacts=artifacts, generated_source=generated_source)
    cases = [BenchmarkCase(name, BenchmarkTarget.response(exp, backend="native",
        scenario=Scenario.cold(max_output_tokens=cap), wan=WanConditions(), state_audit=True), build)
        for name, exp in values.items()]
    return BenchmarkSuite(cases, Sampling(processes=3, order="balanced", threads=1,
        repetitions=1, warmups=0, timeout_seconds=3600, memory_bytes=8 << 30),
        "private-boundary-decode-confirmation-v1")


def _validate_suite(document, tiny, max_output_tokens):
    cap = output_cap(max_output_tokens, tiny=tiny)
    suite = BenchmarkSuite.from_spec(document["suite"])
    _require(tuple(case.name for case in suite.cases) == NAMES,
             "confirmation_cells_changed")
    target = suite.cases[0].target.to_spec()["parameters"]
    exp = Experiment.from_spec(target["model"]["experiment"])
    tokenizer = exp.pipeline.components.get("tokenizer")
    artifacts = Path(tokenizer.get_params()["path"]).parent if tokenizer else Path("target/prepared-frontier-artifacts")
    expected = matched_suite(build=suite.cases[0].build, max_output_tokens=max_output_tokens,
        tiny=tiny, artifacts=artifacts, generated_source=exp.pipeline.model if tiny else None)
    _require(suite.to_spec() == expected.to_spec() and suite.cases[0].build.expected_digest,
             "confirmation_suite_contract_changed")
    return suite, cap


def _state_and_memory(row, cap, tiny):
    """Workload-aware checks: EOS is sampled but is not an emitted output token."""
    execution = row["execution"]
    detail = execution["details"]
    report = detail["report"]
    _require(report["checks"]["passed"] is True and len(report["runs"]) == 1,
             "confirmation_response_checks_failed")
    request, = report["runs"]
    record = detail["response_state_audit"]
    _require(set(record["client_memory_owners"]) == {
        "sdk", "dashboard", "native_wan_relay", "audit", "role_supervision"},
        "confirmation_client_owners_missing")
    binding, = record["bindings"]
    selections = binding["selections"]
    tokens, privacy = request["tokens"], request["privacy"]
    outputs = tokens["output_tokens"]
    # SDK does not attribute a completed generation before the first emitted
    # token. Retain such raw attempts, but do not fabricate their missing work.
    _require(outputs != 0, "confirmation_zero_output_telemetry_unsupported")
    _require(type(outputs) is int and 0 <= outputs <= cap
        and tokens["authoritative"] is True and request["cold"] is True
        and type(tokens["input_tokens"]) is int
        and (0 < tokens["input_tokens"] <= 128 if tiny else tokens["input_tokens"] == 150)
        and binding["failed"] is False and bool(binding["numeric_body_fingerprint"])
        and binding["sampler_seed"] == 42 and binding["temperature"] == 0
        and binding["cryptographic_randomness_frozen"] is False
        and len(selections) == outputs + (outputs < cap)
        and 1 <= len(selections) <= cap
        and privacy["generation_requested_output_tokens"] == cap
        and privacy["generation_output_tokens"] == outputs
        and privacy["generation_executed_decode_steps"] == len(selections) - 1,
        "confirmation_full_state_or_executed_work_incomplete")
    for index, selection in enumerate(selections):
        _require(all(selection.get(key) is not None for key in (
            "position", "cache_lengths", "logits_and_active_kv", "selected_token", "rng_before", "rng_after"))
            and type(selection["checked_values"]) is int and selection["checked_values"] > 0
            and type(selection["position"]) is int
            and selection["position"] == tokens["input_tokens"] + index
            and bool(selection["cache_lengths"])
            and all(type(length) is int and length == selection["position"]
                for length in selection["cache_lengths"]), "confirmation_full_state_missing")
    fixture = execution["fixture"]
    _require(fixture["metadata"]["cached_checkpoint"] is True
        and fixture["metadata"]["owned_client_cache"] == "fresh"
        and fixture["digest"] == report["configuration"]["source_lock_digest"],
        "confirmation_source_present_scope_missing")
    metrics = {metric["id"]: metric["value"] for metric in row["metrics"]}
    _require(len(metrics) == len(row["metrics"]) and all(
        (key == "response.decode_tps" and outputs == 1 and metrics.get(key) is None)
        or (type(metrics.get(key)) in (int, float) and math.isfinite(metrics[key]) and metrics[key] >= 0)
        for key in METRICS),
        "confirmation_priority_metric_invalid")
    memory, guard = report["client_process_memory"], report["memory_guard"]
    peak = memory["after"]["lifetime_peak_rss_bytes"]
    _require(type(peak) is int and peak > 0 and peak == metrics["client.peak_rss"]
        and metrics["process.peak_rss"] >= peak
        and metrics["process.peak_rss"] == int(metrics["process.peak_rss"])
        and "client and in-process dashboard" in memory["scope"]
        and guard["tripped"] is False
        and type(guard["maximum_host_swap_growth_bytes"]) is int
        and guard["maximum_host_swap_growth_bytes"] == 0
        and report["wan_readiness"]["emulation"]["native_stream_rate_snapshots_checked"] is True,
        "confirmation_whole_owner_memory_or_wan_missing")
    identity = {key: report["configuration"][key] for key in (
        "source_lock_digest", "prompt_digest", "wan_emulation_digest", "sampling")}
    identity.update(body=binding["numeric_body_fingerprint"], tokens=tokens,
        output=request["generation"]["output_text_digest"], build=execution["build"])
    work = {"output_tokens": outputs, "sampled_selections": len(selections),
        "executed_decode_steps": privacy["generation_executed_decode_steps"],
        "terminated_before_cap": outputs < cap}
    return report, metrics, record["bindings"], identity, work


def _checked_measurements(row):
    """Rebuild frozen SDK contracts and values from original raw observations."""
    from pllm._benchmark_evidence import measurement

    records = {metric["id"]: metric for metric in row["metrics"]}
    report = row["execution"]["details"]["report"]
    run, = report["runs"]
    durations, output = run["durations"], run["tokens"]["output_tokens"]
    cpu = report["process_cpu_accounting"]["cold_first_response_cpu_seconds_by_role"]
    topology = report["topology_accounting"]
    scope = "covered_application_bodies:" + topology["runs"][0]["body_counter_coverage"]
    wall, = row["execution"]["details"]["timings"]["wall_seconds"]

    def same_value(actual, expected):
        if expected is None:
            return actual is None
        return (type(actual) in (int, float) and type(expected) in (int, float)
            and math.isfinite(actual) and math.isfinite(expected) and actual >= 0 and expected >= 0
            and math.isclose(actual, expected, rel_tol=4 * 2**-52, abs_tol=4 * 2**-52))

    _require(same_value(row["samples"][0]["duration_seconds"], wall),
             "confirmation_sample_execution_duration_mismatch")
    decode = None
    if output > 1:
        seconds = durations["generation_seconds"]
        _require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0,
                 "confirmation_generation_duration_invalid")
        decode = (output - 1) / seconds
        _require(same_value(durations["tokens_per_second"], decode), "confirmation_raw_decode_rate_mismatch")
    expected = {}

    def add(name, value, **options):
        expected[name] = measurement(name, value, **options)

    add("client.peak_rss", report["client_process_memory"]["after"]["lifetime_peak_rss_bytes"],
        kind="memory", window="process_birth_through_scenario", scope="process_lifetime_high_water")
    for role in ("client", "preparation"):
        add(f"{role}.cold_cpu", cpu[role], kind="cpu", role=role, scope="process_cpu",
            window="benchmark_start_through_first_response" if role == "client" else "role_birth_through_first_response")
    add("preparation.peak_rss", run["processes"]["preparation"]["rss_peak_bytes"],
        kind="memory", role="preparation", window="request_windows", scope="sampled_process_rss")
    add("request.0.full", durations["full_seconds"],
        window="request_setup_through_completion", scope="ordinary_response_lifecycle")
    for name in ("request.0.ttft", "response.ttft"):
        add(name, durations["ttft_seconds"], window="online_to_first_token", phase="online")
    for name in ("request.0.decode_tps", "response.decode_tps"):
        add(name, decode, kind="throughput", window="after_first_output", phase="online",
            denominator={"value": max(output - 1, 0), "unit": "generated_tokens"})
    add("network.all_link_stream_bytes", report["native_network_accounting"]["samples"]["runs"][0]["after"][
        "wan_emulation"]["total_stream_bytes"], kind="bytes", role="all-links", statistic="sum",
        window="relay_start_through_scenario", scope="tcp_stream_payload")
    add("network.covered_body_bytes", topology["startup"]["all_link_serialized_body_bytes"]
        + topology["runs"][0]["all_link_serialized_body_bytes"], kind="bytes", role="all-links", statistic="sum",
        window="setup_through_scenario", scope=scope)
    add("target.latency", wall)
    # This metric is the finalized worker high-water observation itself; there
    # is no second raw finalized peak. Its ownership and >= SDK peak checks stay
    # in _state_and_memory, and the largest replicate still gates observed tiers.
    add("process.peak_rss", records["process.peak_rss"]["value"], kind="memory",
        window="process_birth_through_target", scope="process_lifetime_high_water")
    for name, fresh in expected.items():
        actual = records[name]
        _require({key: value for key, value in actual.items() if key != "value"}
            == {key: value for key, value in fresh.items() if key != "value"},
            "confirmation_explicit_measurement_contract_changed")
        _require(same_value(actual["value"], fresh["value"]), "confirmation_metric_differs_from_raw_observation")


def _contracts(row, case, exp, cap, work):
    """Bind result, request, sampler, oracle, admission and comparison contracts."""
    from pllm._benchmark_contract import digest
    from pllm.evidence import environment_digest
    from pllm.schedulers import _numeric_component

    execution, report = row["execution"], row["execution"]["details"]["report"]
    request, = report["runs"]
    config = report["configuration"]
    sampling = {"effective_temperature": 0.0, "mode": "greedy", "requested_temperature": 0.0, "top_p": None}
    checks = {"all_runs_completed", "authoritative_token_usage", "memory_guard_passed",
        "model_fingerprint_present", "no_online_preparation_requests", "no_plaintext_prompt_bytes_sent",
        "no_plaintext_token_ids_sent", "sampling_matches_request", "passed"}
    _require(row["schema_version"] == "pllm.benchmark_result.v2" and row["scope"] == "deployment"
        and row["warmups"] == 0 and row["repetitions"] == 1
        and report["schema_version"] == "pllm.loopback_benchmark.v1" and report["warmup_runs"] == []
        and checks <= set(report["checks"]) and all(value is True for value in report["checks"].values())
        and config["warmups"] == 0 and config["repetitions"] == 1
        and config["max_output_tokens"] == request["max_output_tokens"] == cap
        and config["sampling"] == request["sampling"] == sampling
        and request["status"] == "completed" and request["failure"] is None and request["warm"] is False
        and request["generation"]["response_status"] == ("completed" if work["terminated_before_cap"] else "incomplete")
        and execution["failure_code"] is None and execution["admission"]["admitted"] is True
        and execution["details"]["topology_admission"]["admitted"] is True
        and report["memory_preflight"]["admitted"] is True
        and execution["details"]["response_state_audit"]["schema"] == "pllm.response_state_audit.v1",
        "confirmation_result_request_or_admission_contract_changed")
    sample, = row["samples"]
    _require(sample["kind"] == "measurement" and sample["status"] == "completed" and sample["index"] == 0,
             "confirmation_sample_contract_changed")
    bindings = execution["details"]["response_state_audit"]["bindings"]
    _require(execution["oracle"] == {"kind": "salted_full_logit_kv_sampler_commitments", "outcome": "passed",
        "checked_values": sum(s["checked_values"] for s in bindings[0]["selections"])}, "confirmation_oracle_changed")
    numeric = {name: value.to_spec() for name, value in exp.pipeline.components.items()
        if name in {"quantization", "nonlinear", "scheduler"}
        and (name != "scheduler" or _numeric_component(value.to_spec()) is not None)}
    scenario = case.target.to_spec()["parameters"]["scenario"]
    comparison = execution["comparison"]
    expected = {"source_digest": config["source_lock_digest"],
        "numeric_digest": digest({"bodies": [request["model_fingerprint"]], "contract": numeric}),
        "state_digest": digest({"lifecycle": scenario}),
        "output_digest": digest({"output": digest([request["generation"]["output_text_digest"]]), "state": bindings})}
    _require(all(comparison[key] == value for key, value in expected.items())
        and row["workload_digest"] == comparison["workload_digest"]
        and all(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for key, value in comparison.items() if key != "source_digest"), "confirmation_comparison_binding_changed")
    environment = row["environment"]
    _require(environment["digest"] == environment_digest(environment["attributes"])
        and environment["attributes"]["threads"] == 1, "confirmation_environment_binding_changed")
    contracts = {}
    for metric in row["metrics"]:
        # Metadata, including unavailability and N-1 denominator, is evidence.
        required = {"id", "component", "unit", "origin", "measurement_scope", "owners", "role",
            "parameters", "phase", "sample_count", "statistic", "unavailable_reason", "window", "denominator"}
        _require(required <= set(metric) and metric["origin"] == ("not_available" if metric["value"] is None else "measured")
            and metric["sample_count"] == (0 if metric["value"] is None else 1)
            and (bool(metric["unavailable_reason"]) if metric["value"] is None else metric["unavailable_reason"] is None),
            "confirmation_metric_contract_invalid")
        contracts[metric["id"]] = {key: value for key, value in metric.items() if key != "value"}
    for key in METRICS:
        _require(contracts[key]["unit"] == ("bytes" if key.endswith("rss") or key.startswith("network.")
            else "per_second" if key == "response.decode_tps" else "seconds"), "confirmation_metric_unit_changed")
    _require(contracts["response.decode_tps"]["denominator"] == {
        "unit": "generated_tokens", "value": max(work["output_tokens"] - 1, 0)}, "confirmation_decode_denominator_changed")
    return {"environment": environment, "comparison": {key: comparison[key] for key in (
        "source_digest", "numeric_digest", "state_digest", "workload_digest", "output_digest")}}, {
        "comparison": comparison, "metrics": contracts}


def _reconcile(bodies):
    paired = []
    for rep in range(3):
        control, candidate = bodies[CONTROL, rep], bodies[WINNER, rep]
        _require(all(candidate[key] == control[key] for key in (
            "prepared", "fixed_work", "boundary_fixed")), "confirmation_fixed_work_changed")
        left, right = control["stage_bodies"], candidate["stage_bodies"]
        _require(set(left) == set(right) and all(
            {edge: value for edge, value in left[stage].items() if edge != "inference->client"}
            == {edge: value for edge, value in right[stage].items() if edge != "inference->client"}
            for stage in left), "confirmation_fixed_stage_work_changed")
        bound = control["reply_metadata_delta_bound_bytes"]
        _require(candidate["reply_metadata_delta_bound_bytes"] == bound
            and sum(abs(right[stage]["inference->client"] - left[stage]["inference->client"])
                for stage in left) <= bound, "confirmation_reply_metadata_bound_exceeded")
        delta = candidate["total"] - control["total"]
        reply = candidate["head_reply"] - control["head_reply"]
        controls = candidate["controls"] - control["controls"]
        metadata = candidate["prepared_reply"] - control["prepared_reply"]
        _require(reply < 0, "confirmation_head_reply_not_smaller")
        paired.append({"replicate": rep, "covered_body_delta_bytes": delta,
            "head_reply_delta_bytes": reply, "lifecycle_control_delta_bytes": controls,
            "prepared_stage_reply_delta_bytes": metadata,
            "prepared_stage_reply_metadata_compatibility_bound_bytes": bound,
            "bootstrap_and_metadata_residual_bytes": delta - reply - controls - metadata})
    return paired


def audit(document, *, max_output_tokens=8, tiny=False):
    """Fail closed; never pool workloads, normalize EOS away or repair evidence."""
    try:
        from pllm.metrics import BenchmarkRun
        try:
            # Validate original v2 records, observed build attributes/digest,
            # samples, target/case/suite bindings and exact cohort membership.
            # Do not normalize or replace the supplied evidence with a repair.
            BenchmarkRun(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise Rejected("confirmation_invalid_v2_cohort") from exc
        suite, cap = _validate_suite(document, tiny, max_output_tokens)
        _require(document["schema"] == "pllm.benchmark_run.v1"
            and document["complete"] is True and len(document["results"]) == 9,
                 "confirmation_incomplete_cohort")
        cohort = document["cohort"]
        _require(type(cohort) is str and len(cohort) == 64
            and all(c in "0123456789abcdef" for c in cohort), "confirmation_private_cohort_missing")
        cases = {case.name: case.to_spec() for case in suite.cases}
        samples, bodies, work_rows, seen = {}, {}, {}, set()
        states = identity = common_contract = common_metrics = None
        arm_contracts = {}
        checked = 0
        for row in document["results"]:
            execution = row["execution"]
            name, rep = execution["case"]["name"], execution["replicate"]
            _require(row["status"] == "completed" and name in NAMES
                and type(rep) is int and 0 <= rep < 3 and (name, rep) not in seen
                and execution["case"] == cases[name]
                and execution["cohort"] == cohort and execution["suite_digest"] == suite.digest
                and execution["sampling"] == suite.sampling.to_spec()
                and execution["build"]["digest"] == suite.cases[0].build.expected_digest
                and execution["isolation"]["fresh_process"] is True
                and execution["isolation"]["controller_included"] is False,
                "confirmation_changed_duplicate_or_mixed_cell")
            seen.add((name, rep))
            report, metrics, bindings, current, work = _state_and_memory(row, cap, tiny)
            _checked_measurements(row)
            _require((identity is None or current == identity) and (states is None or bindings == states),
                     "confirmation_source_history_logits_kv_selections_rng_or_output_mismatch")
            identity, states = current, bindings
            checked += sum(selection["checked_values"] for selection in bindings[0]["selections"])
            screen._wire(report, metrics, name != "maintained")
            model = execution["case"]["target"]["parameters"]["model"]
            exp = Experiment.from_spec(model["experiment"]).with_params(pipeline__profile=model["profile"])
            case = next(case for case in suite.cases if case.name == name)
            common, arm = _contracts(row, case, exp, cap, work)
            _require(common_contract is None or common_contract == common, "confirmation_cross_replicate_contract_changed")
            _require(name not in arm_contracts or arm_contracts[name] == arm, "confirmation_arm_contract_changed")
            normalized = {key: dict(value) for key, value in arm["metrics"].items()}
            for key in ("network.covered_body_bytes", "network.startup_body_bytes"):
                if key in normalized:
                    from pllm.runtime.topology_accounting import ACK_COVERAGE, BOUNDARY_COVERAGE
                    _require(normalized[key]["measurement_scope"] == "covered_application_bodies:" + (
                        ACK_COVERAGE if name == "maintained" else BOUNDARY_COVERAGE), "confirmation_body_metric_scope_changed")
                    normalized[key]["measurement_scope"] = "explicit_cross_topology_body_coverage"
            _require(common_metrics is None or common_metrics == normalized, "confirmation_cross_arm_metric_contract_changed")
            common_contract, common_metrics, arm_contracts[name] = common, normalized, arm
            _material(report)
            body_audit = _eos_body if work["terminated_before_cap"] else screen._body
            body = body_audit(report, metrics, name != "maintained",
                chunk_rows=exp.pipeline.components["linear"].get_params()["prefill_chunk_rows"])
            if name != "maintained":
                _require(body["boundary_fixed"]["private_boundary_head_calls"] == work["sampled_selections"]
                    and body["boundary_fixed"]["private_boundary_lookup_rows"]
                    == current["tokens"]["input_tokens"] + work["executed_decode_steps"],
                    "confirmation_boundary_executed_work_changed")
            bodies[name, rep], samples[name, rep] = body, metrics
            work_rows[name, rep] = work | {key: body["fixed_work"][key] for key in (
                "preparation_rows", "preparation_prefill_rows_issued", "preparation_decode_rows_issued",
                "preparation_padding_rows_issued", "inference_stage_calls", "prepared_stage_rows_reserved",
                "prepared_stage_rows_claimed", "prepared_stage_rows_burned", "prepared_stage_rows_discarded")}
        groups = {}
        for name in NAMES:
            rows = [samples[name, rep] for rep in range(3)]
            peak = max(row["process.peak_rss"] for row in rows)
            groups[name] = {"samples": rows, "maximum_client_peak_bytes": peak,
                "observed_tiers": {tier: peak <= ceiling for tier, ceiling in tiers.TIERS.items()},
                "medians": {metric: statistics.median(row[metric] for row in rows)
                    if all(row[metric] is not None for row in rows) else None for metric in METRICS},
                "executed_work": [work_rows[name, rep] for rep in range(3)]}
        return {"schema": "pllm.private_boundary_decode_confirmation.v1", "tiny": tiny,
            "complete": True, "cohort": cohort, "clients": len(seen), "replicates": 3,
            "output_cap": cap, "identity": identity, "checked_values": checked,
            "all_logits_active_kv_positions_selections_rng_and_outputs_match": True,
            "state_commitment": hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest(),
            "maximum_observed_swap_growth_bytes": 0, "groups": groups,
            "long_output_cap_reached": all(not work["terminated_before_cap"] for work in work_rows.values()),
            "comparisons": {WINNER: {control: _compare(groups[control], groups[WINNER])
                for control in (CONTROL, "maintained")}},
            "body_reconciliation": {WINNER: _reconcile(bodies)},
            "scope": {"request": f"single cold source-present client, {'128' if tiny else '150'}+{cap} cap; natural EOS",
                "client": "whole fresh worker: SDK, supervisor, dashboard, native WAN relay and full-state audit; idle launcher excluded",
                "memory": "largest finalized lifetime whole-worker RSS across all three replicates; observed tiers do not relax conservative admission",
                "wire": "native shared 100/40 Mbps caps; directed TCP stream payload, not physical wire",
                "serializer": "original numeric MessagePack bodies retained, including variable-width server_ns; metadata-compatible delta bound is not exact timing attribution",
                "evidence": "one independent cohort and one output cap only; no old observations or cross-workload pooling",
                "limits": "source/checkpoint/tokenizer already present; no distribution, Internet RTT, generation quality or global-optimum claim; two additional non-colluding boundary workers change role contract"}}
    except Rejected:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as exc:
        raise Rejected("confirmation_missing_or_invalid_evidence") from exc
