"""Adapters over the existing role and operator benchmark executors."""
from __future__ import annotations

from contextlib import nullcontext
import statistics

from pllm._benchmark_contract import Scenario, TargetUnavailable, digest


SERVING_METRICS = {
    "client.peak_rss": "min", "client.cold_cpu": "min",
    "preparation.cold_cpu": "min", "preparation.peak_rss": "min",
    "response.ttft": "min", "response.decode_tps": "max", "network.all_link_stream_bytes": "min",
}


AUDIT_COUNTERS = (
    "decode_epochs_started", "decode_epoch_pauses", "decode_epoch_pause_ns", "preparation_attempts",
    "preparation_rows", "preparation_failures", "prepared_stage_rows_issued",
    "prepared_stage_rows_reserved", "prepared_stage_rows_claimed",
    "prepared_stage_rows_burned", "prepared_stage_rows_discarded",
    "prefill_cache_hits", "prefill_cache_misses", "prefill_prefix_tokens_reused",
    "correction_push_bytes", "preparation_upload_bytes", "preparation_download_bytes",
    "session_authorization_upload_bytes", "session_authorization_download_bytes",
    "inference_control_upload_bytes", "inference_control_download_bytes",
    "preparation_control_upload_bytes", "preparation_control_download_bytes", "bundle_network_bytes",
    "generation_completed_responses", "generation_output_tokens",
    "inflight_producer_registrations", "inflight_publications", "inflight_joins",
    "inflight_failures", "inflight_cancellations", "inflight_cancelled_producers",
    "inflight_retired", "inflight_copy_bytes", "inflight_bound_rejections", "inflight_cache_bypasses",
)

# Owner gauges/high-water marks are not additive counters. Dashboard deltas
# cannot recover their absolute values; retain unknown rather than sum peaks.
INFLIGHT_NONADDITIVE = {
    "inflight_reserved_bytes", "inflight_peak_reserved_bytes", "inflight_requests",
    "inflight_nodes", "inflight_pending_nodes", "inflight_active_producers",
}


def audit_metrics(windows, m, *, window="request_windows"):
    """Only disjoint runtime audit deltas; never substitute lifetime counters."""
    names = set(AUDIT_COUNTERS)
    names.update(key for row in windows for key in row if key.startswith("inflight_"))
    totals = {}
    for name in sorted(names):
        values = [row.get(name) for row in windows]
        total = sum(values) if (name not in INFLIGHT_NONADDITIVE and values
            and all(type(v) is int and v >= 0 for v in values)) else None
        totals[name] = total
    m.details["runtime_audit"] = {"window": window, "scope": "disjoint_runtime_audit_deltas",
                                  "totals": totals, "windows": windows}
    reserved, claimed, burned = (totals[key] for key in (
        "prepared_stage_rows_reserved", "prepared_stage_rows_claimed", "prepared_stage_rows_burned"))
    m.details["runtime_audit"]["reserved_equals_claimed_plus_burned"] = (
        reserved == claimed + burned if all(v is not None for v in (reserved, claimed, burned)) else None)
    pauses = [row.get("decode_epoch_pause_ns") for row in windows]
    m.metric("response.epoch_pause", sum(pauses) / 1e9 if pauses and all(
        type(v) is int and v >= 0 for v in pauses) else None,
        statistic="sum", window=window, scope="runtime_idle_epoch_pause")


def response_metrics(report, m):
    from .topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    coverage = accounting_coverage_identity(report.get("topology_accounting"))
    body_scope = "covered_application_bodies"
    if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"}:
        body_scope += f":{coverage or 'mixed_or_unknown'}"
    runs = report["runs"]
    audit_metrics([run.get("privacy", {}) for run in runs], m)
    cpu = report.get("process_cpu_accounting", {}).get("cold_first_response_cpu_seconds_by_role", {}) or {}
    m.metric("client.peak_rss", report.get("client_process_memory", {}).get("after", {}).get("lifetime_peak_rss_bytes"),
             kind="memory", window="process_birth_through_scenario", scope="process_lifetime_high_water")
    for role in ("client", "preparation"):
        m.metric(f"{role}.cold_cpu", cpu.get(role), kind="cpu", role=role,
                 window="benchmark_start_through_first_response" if role == "client" else "role_birth_through_first_response",
                 scope="process_cpu")
    peaks = [run.get("processes", {}).get("preparation", {}).get("rss_peak_bytes") for run in runs]
    m.metric("preparation.peak_rss", max(peaks) if peaks and all(v is not None for v in peaks) else None,
             kind="memory", role="preparation", window="request_windows", scope="sampled_process_rss")
    for index, run in enumerate(runs):
        duration, tokens = run["durations"], run["tokens"]
        m.metric(f"request.{index}.full", duration.get("full_seconds"),
                 window="request_setup_through_completion", scope="ordinary_response_lifecycle")
        m.metric(f"request.{index}.preparation", duration.get("preparation_seconds"), phase="offline",
                 window="request_initial_preparation", scope="ordinary_response_initial_setup")
        pause = run.get("privacy", {}).get("decode_epoch_pause_ns")
        m.metric(f"request.{index}.epoch_pause", pause / 1e9 if type(pause) is int and pause >= 0 else None,
                 window="request_epoch_pauses", scope="runtime_idle_epoch_pause", statistic="sum")
        denominator = max(0, tokens["output_tokens"] - 1)
        m.metric(f"request.{index}.ttft", duration.get("ttft_seconds"), window="online_to_first_token", phase="online")
        m.metric(f"request.{index}.decode_tps", duration.get("tokens_per_second") if denominator else None,
                 kind="throughput", window="after_first_output", phase="online",
                 denominator={"value": denominator, "unit": "generated_tokens"})
    m.metric("response.ttft", statistics.median(r["durations"]["ttft_seconds"] for r in runs),
             window="online_to_first_token", phase="online", count=len(runs))
    decode = [r["durations"].get("tokens_per_second") if r["tokens"]["output_tokens"] > 1 else None for r in runs]
    m.metric("response.decode_tps", statistics.median(decode) if all(v is not None for v in decode) else None,
             kind="throughput", window="after_first_output", phase="online", count=len(runs),
             denominator={"value": sum(max(0, r["tokens"]["output_tokens"] - 1) for r in runs), "unit": "generated_tokens"})
    snapshots = report.get("native_network_accounting", {}).get("samples", {}).get("runs", [])
    emulation = report.get("wan_readiness", {}).get("emulation", {})
    total = (snapshots[-1].get("after", {}).get("wan_emulation", {}).get("total_stream_bytes")
             if snapshots and emulation.get("native_stream_rate_snapshots_checked") is True else None)
    m.metric("network.all_link_stream_bytes", total, kind="bytes", role="all-links", statistic="sum",
             window="relay_start_through_scenario", scope="tcp_stream_payload")
    body = report.get("communication_per_token", {}).get("summary", {})
    rate, count = body.get("setup_inclusive_mb_per_output_token"), body.get("generated_output_tokens")
    summary = report.get("summary", {})
    exact = summary.get("total_accounted_benchmark_body_bytes")
    covered = (exact if type(exact) is int else None) if "total_accounted_benchmark_body_bytes" in summary else (
        None if rate is None or count is None else rate * count * 1e6)
    if coverage is None:
        covered = None
    m.metric("network.covered_body_bytes", covered,
              kind="bytes", role="all-links", statistic="sum", window="setup_through_scenario", scope=body_scope)
    startup = (report.get("topology_accounting") or {}).get("startup") or {}
    m.metric("network.startup_body_bytes", startup.get("all_link_serialized_body_bytes"),
             kind="bytes", role="all-links", statistic="sum", window="ordinary_startup",
             scope=body_scope)


def execute_response(p, m):
    from pllm.configuration import Experiment
    from pllm.deployment import WanConditions
    from pllm.metrics import benchmark_memory
    from pllm.model_loader import expected_model_id
    from .benchmark_cli import run_loopback_benchmark
    from .benchmark_context import validate_context
    scenario = Scenario(**p["scenario"])
    if m.sampling.repetitions != 1 or m.sampling.warmups:
        raise TargetUnavailable("response_requires_one_scenario_per_process")
    field = "prompts" if scenario.kind == "sequence" else "prompt"
    if set(m.inputs) != {field}:
        raise TargetUnavailable("response_input_contract")
    prompt = m.inputs[field]
    try:
        if scenario.kind == "sequence":
            if not isinstance(prompt, list) or len(prompt) != scenario.count:
                raise ValueError("sequence length differs from scenario")
            prompt = [validate_context(context) for context in prompt]
        else:
            prompt = validate_context(prompt)
    except ValueError:
        raise TargetUnavailable("response_input_contract") from None
    if p.get("concurrent_requests", 1) > 1:
        from .benchmark_response_group import execute_group
        return execute_group(p, m, prompt)
    declared = p["model"]
    experiment = Experiment.from_spec(declared["experiment"]).with_params(pipeline__profile=declared["profile"])
    if experiment.budget.requests < scenario.count or experiment.budget.max_new_tokens < scenario.max_output_tokens:
        raise TargetUnavailable("scenario_exceeds_experiment_budget", status="blocked")
    # Only owned caches are reset; the shared checkpoint/compiled cache is explicit.
    experiment = experiment.with_params(deployment__root=str(m.directory / "roles"))
    wan = None if p["wan"] is None else WanConditions.from_spec(p["wan"])
    admission = benchmark_memory(experiment, backend=p["backend"], enforced_wan=wan is not None,
                                 memory_budget_bytes=m.sampling.memory_bytes)
    m.details["topology_admission"] = admission
    if not admission["admitted"]:
        raise TargetUnavailable("topology_memory_limit", status="blocked")
    m.admit(128 << 20)
    def execute(_):
        return run_loopback_benchmark(model=experiment.pipeline.model.source,
            model_id=expected_model_id(experiment.pipeline.model), tiny=False,
            experiment=experiment, prompt=prompt[0] if scenario.kind == "sequence" else prompt,
            prompt_sequence=prompt if scenario.kind == "sequence" else None,
            repetitions=1 if scenario.kind == "sequence" else scenario.count, warmups=0,
            max_output_tokens=scenario.max_output_tokens, timeout_seconds=m.sampling.timeout_seconds,
            temperature=0, capture_output_digest=True, _cohort_salt=m.salt,
            backend=p["backend"], wan=wan, emulate_wan=wan is not None,
            memory_budget_bytes=m.sampling.memory_bytes)
    selected = p.get("client_stage")
    stage_scope = nullcontext([])
    if selected is not None:
        from .benchmark_client_stage import client_stage_execution
        stage_scope = client_stage_execution(selected)
    mlp = p.get("client_mlp")
    if mlp is not None:
        from .benchmark_client_mlp import client_mlp_execution
        stage_scope = client_mlp_execution(mlp)
    metadata = p.get("client_metadata", False)
    if metadata:
        from .benchmark_client_metadata import client_metadata_attribution
        stage_scope = client_metadata_attribution()
    with stage_scope as stage_records:
        report, = m.run(execute)
    if selected is not None:
        if not stage_records or sum(record["calls"] for record in stage_records) == 0:
            raise TargetUnavailable("research_client_stage_not_executed", status="failed")
        m.details["client_stage_research"] = {"selection": selected, "bindings": stage_records,
            "scope": "one_selected_stage_in_complete_sdk_responses", "pipeline_component": False}
    if mlp is not None:
        if not stage_records or not any(row["prefill_calls"] for row in stage_records):
            raise TargetUnavailable("research_mlp_not_executed", status="failed")
        m.details["client_mlp_research"] = {"selection": mlp, "bindings": stage_records,
            "scope": "compiler_bound_mlp_tiling_in_complete_sdk", "pipeline_component": False}
    if metadata:
        if not stage_records or not any(row["phase"] == "runtime_binding" for row in stage_records):
            raise TargetUnavailable("client_metadata_attribution_missing", status="failed")
        m.details["client_metadata"] = {"phases": stage_records,
            "scope": "python_traced_construction_and_unique_public_metadata", "pipeline_component": False}
    if (report.get("checks", {}).get("passed") is not True or len(report["runs"]) != scenario.count
            or any(r["status"] != "completed" or r["tokens"].get("authoritative") is not True for r in report["runs"])):
        raise TargetUnavailable("response_checks_failed", status="failed")
    response_metrics(report, m)
    from pllm._search_benchmark import _boundary
    boundary = _boundary(experiment)
    config, runs = report["configuration"], report["runs"]
    m.comparison.update(source_digest=config["source_lock_digest"],
        numeric_digest=digest({"bodies": [r["model_fingerprint"] for r in runs], "contract": boundary["numeric"]}),
        privacy_digest=digest(boundary["privacy"]), state_digest=digest({"lifecycle": scenario.to_spec()}),
        workload_digest=digest({"tokens": [r["tokens"] for r in runs], "scenario": scenario.to_spec(),
            "backend": config["provider_backend"], "wan": config.get("wan_emulation_digest"), "sampling": config["sampling"]}),
        output_digest=digest([r["generation"]["output_text_digest"] for r in runs]),
        measurement_digest=digest({"schema": report["schema_version"], "scenario": scenario.to_spec(), "cached_checkpoint": True}))
    from .topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    coverage = accounting_coverage_identity(report.get("topology_accounting"))
    if coverage not in {ACCOUNTING_SCHEMA, "historical_coverage_not_recorded"}:
        m.comparison["measurement_digest"] = digest({"response": m.comparison["measurement_digest"],
                                                    "body_counter_coverage": coverage})
    if selected is not None:
        m.comparison["measurement_digest"] = digest({"response": m.comparison["measurement_digest"],
            "research_client_stage": {key: selected[key] for key in ("role", "layer")}})
        m.limitations += ["Explicit research stage-ownership adapter; not a selectable Pipeline component or native decoder.",
                          "Python, tokenizer, transport, model binding and response state remain in the measured client."]
    if mlp is not None:
        m.comparison["measurement_digest"] = digest({"response": m.comparison["measurement_digest"],
            "research": "client_mlp_working_set_v1"})
        m.limitations += ["Internal compiler-bound MLP tiling experiment; no selectable Pipeline component.",
                         "Ordinary conservative memory and residue-codec admission remain in force."]
    if metadata:
        m.comparison["measurement_digest"] = digest({"response": m.comparison["measurement_digest"],
            "research": "client_metadata_attribution_v1"})
        m.limitations += ["Allocation tracing changes CPU, latency and RSS; compare only identically traced responses.",
                         "Traced construction bytes include function temporaries; they are not removable metadata or native peaks."]
    m.fixture = {"kind": "generated_checkpoint" if experiment.pipeline.model.kind == "tiny" else "checkpoint",
        "digest": config["source_lock_digest"], "metadata": {"cached_checkpoint": True, "owned_client_cache": "fresh"}}
    m.oracle = {"kind": "ordinary_role_runtime_checks", "outcome": "passed", "checked_values": sum(r["tokens"]["output_tokens"] for r in runs)}
    m.details.update(model={"id": expected_model_id(experiment.pipeline.model), "checkpoint_digest": None,
                           "source_lock_digest": config["source_lock_digest"]},
        component_ids=sorted(c.component for c in experiment.pipeline.components.values()), report=report)
    m.limitations += ["Runtime checks and matched output commitments are not full-logit/KV or generation-quality oracles.",
        "Cached checkpoint; source resolution, public artifact publication and physical wire retain the ordinary report's exclusions."]


def execute_operator(p, m):
    from pllm.components import create_component
    from pllm.metrics import benchmark_operator
    component = create_component(p["component"]["component"], p["component"]["params"])
    if "encrypted-softmax" in component.component:
        required = (2 << 30) + (128 << 20)
    else:
        required = 512 << 20
    m.admit(required)
    reports = m.run(lambda _: benchmark_operator(component, lanes=p["lanes"], samples=1,
                            inputs=m.inputs or None, geometry=p["geometry"]))
    if any(r.quality.exact_integer_outputs != r.quality.checked_outputs
           and r.quality.outputs_within_tolerance != r.quality.checked_outputs for r in reports):
        raise TargetUnavailable("operator_oracle_failed", status="failed")
    report = reports[-1]
    m.fixture = {"kind": "caller_owned" if m.inputs else "synthetic", "digest": None,
                 "metadata": {"input_source": report.input_source}}
    m.oracle = {"kind": "independent_operator_oracle", "outcome": "passed",
                "checked_values": sum(r.quality.checked_outputs for r in reports)}
    m.comparison.update(source_digest=digest({"operator": report.method}), numeric_digest=report.numeric_contract_digest,
        privacy_digest=digest({"scope": "local_operator_reference"}), state_digest=digest({"state": "one_use"}),
        workload_digest=digest({"lanes": p["lanes"], "geometry": p["geometry"]}),
        output_digest=m.commitment({"numeric": report.numeric_contract_digest, "oracle": "passed"}),
        measurement_digest=digest({"scope": "bounded_operator"}))
    m.details.update(component_ids=[component.component], operator_reports=[r.to_dict() for r in reports])
    m.metric("operator.online", statistics.median(r.median_online_seconds for r in reports), phase="online")
    m.metric("operator.issuance", statistics.median(r.median_issuance_seconds for r in reports), phase="offline")
    m.metric("operator.material_bytes", report.resources.combined_raw_key_payload_bytes,
             kind="bytes", origin="deterministic", scope="both_party_raw_material")
    m.limitations.extend(report.limitations)
