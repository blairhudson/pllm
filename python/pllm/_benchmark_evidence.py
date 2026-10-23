"""Validation and canonical measurements for benchmark_result.v2."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from pllm._benchmark_contract import BenchmarkCase, Sampling, canonical, digest, identity


def _fields(value, fields, name):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{name} fields differ from benchmark_result.v2")
    return dict(value)


def _hash(value, name, nullable=False):
    from pllm.evidence import _digest
    return _digest(value, name, nullable=nullable)


def validate_execution(raw, result):
    e = _fields(raw, {"case", "case_digest", "suite_digest", "cohort", "replicate",
        "build", "sampling", "isolation", "fixture", "oracle", "admission",
        "comparison", "details", "failure_code", "budgets"}, "execution")
    case = BenchmarkCase.from_spec(e["case"])
    if e["case_digest"] != digest(case.to_spec()):
        raise ValueError("execution case digest differs")
    if result["configuration_digest"] != case.target.digest:
        raise ValueError("execution target digest differs")
    for field in ("suite_digest", "cohort"):
        _hash(e[field], field)
    sampling = Sampling(**e["sampling"])
    if type(e["replicate"]) is not int or not 0 <= e["replicate"] < sampling.processes:
        raise ValueError("execution replicate is outside its sampling contract")
    build = _fields(e["build"], {"digest", "attributes"}, "build")
    if not isinstance(build["attributes"], Mapping) or build["digest"] != digest(build["attributes"]):
        raise ValueError("observed build digest differs")
    if case.build.expected_digest is not None and case.build.expected_digest != build["digest"]:
        if e["failure_code"] != "build_mismatch" or result["status"] == "completed":
            raise ValueError("observed build differs from required implementation")
    iso = _fields(e["isolation"], {"fresh_process", "sample_owners", "controller_included"}, "isolation")
    if type(iso["fresh_process"]) is not bool or type(iso["controller_included"]) is not bool:
        raise ValueError("isolation flags must be Boolean")
    if not isinstance(iso["sample_owners"], list) or not iso["sample_owners"]:
        raise ValueError("sample owners are required")
    for owner in iso["sample_owners"]:
        identity(owner, "sample owner")
    fixture = _fields(e["fixture"], {"kind", "digest", "metadata"}, "fixture")
    if fixture["kind"] not in {"none", "synthetic", "checkpoint", "caller_owned", "generated_checkpoint", "projection"}:
        raise ValueError("unsupported fixture provenance")
    _hash(fixture["digest"], "fixture digest", nullable=True)
    if not isinstance(fixture["metadata"], Mapping):
        raise ValueError("fixture metadata must be an object")
    oracle = _fields(e["oracle"], {"kind", "outcome", "checked_values"}, "oracle")
    identity(oracle["kind"], "oracle kind")
    if oracle["outcome"] not in {"passed", "failed", "not_run"}:
        raise ValueError("unsupported oracle outcome")
    if type(oracle["checked_values"]) is not int or oracle["checked_values"] < 0:
        raise ValueError("invalid oracle output count")
    if result["status"] == "completed" and oracle["outcome"] != "passed":
        raise ValueError("completed evidence requires a passed oracle")
    admission = _fields(e["admission"], {"admitted", "required_bytes", "limit_bytes",
        "host_headroom_bytes", "scope", "reason"}, "admission")
    if admission["admitted"] is not None and type(admission["admitted"]) is not bool:
        raise ValueError("invalid admission outcome")
    for key in ("required_bytes", "limit_bytes", "host_headroom_bytes"):
        if admission[key] is not None and (type(admission[key]) is not int or admission[key] < 0):
            raise ValueError("invalid admission resource count")
    identity(admission["scope"], "admission scope")
    if admission["reason"] is not None:
        identity(admission["reason"], "admission reason")
    if result["status"] == "completed" and admission["admitted"] is not True:
        raise ValueError("completed targets require resource admission")
    if admission["admitted"] is True and (any(admission[k] is None for k in (
            "required_bytes", "limit_bytes", "host_headroom_bytes"))
            or admission["required_bytes"] > min(admission["limit_bytes"], admission["host_headroom_bytes"])):
        raise ValueError("admitted target exceeds resource limits")
    comparison = _fields(e["comparison"], {"source_digest", "numeric_digest", "privacy_digest",
        "state_digest", "workload_digest", "output_digest", "measurement_digest"}, "comparison")
    for key, value in comparison.items():
        _hash(value, key, nullable=True)
    if result["status"] == "completed" and any(value is None for value in comparison.values()):
        raise ValueError("completed evidence requires comparison commitments")
    if comparison["workload_digest"] != result["workload_digest"]:
        raise ValueError("workload commitments disagree")
    if e["failure_code"] is not None:
        identity(e["failure_code"], "failure code")
    if result["status"] == "completed" and e["failure_code"] is not None:
        raise ValueError("completed evidence cannot contain a failure")
    if not isinstance(e["details"], Mapping) or len(canonical(e["details"])) > 8 << 20:
        raise ValueError("benchmark details exceed bounded metadata")
    expected = assess_budgets(case, result["metrics"])
    if e["budgets"] != expected:
        raise ValueError("benchmark budget assessments differ from observations")
    return e


def measurement(name, value, *, kind="seconds", role="client", window="execution",
                scope="target", origin="measured", statistic="median", count=1,
                phase="full", detail=None, denominator=None, reason=None, owners=None):
    components = {
        "seconds": ("pllm/latency", {"statistic": statistic, "phase": phase}, "seconds"),
        "cpu": ("pllm/cpu-time/v1", {}, "seconds"),
        "memory": ("pllm/memory", {"kind": "peak_rss"}, "bytes"),
        "allocated": ("pllm/memory", {"kind": "allocated"}, "bytes"),
        "bytes": ("pllm/communication", {"direction": "total", "phase": phase}, "bytes"),
        "throughput": ("pllm/throughput", {"basis": "tokens"}, "per_second"),
    }
    component, params, unit = components[kind]
    if value is None:
        origin, count, reason = "not_available", 0, reason or "measurement unavailable"
    return {"id": name, "component": component, "parameters": params,
        "value": value, "unit": unit, "unit_detail": "tokens_per_second" if kind == "throughput" else detail,
        "statistic": statistic, "phase": phase, "role": role, "origin": origin,
        "unavailable_reason": reason, "sample_count": count, "window": window,
        "measurement_scope": scope, "denominator": denominator,
        "owners": list(owners or [role])}


def assess_budgets(case, metrics):
    values = {m["id"]: m for m in metrics}
    results = []
    for budget in case.budgets:
        metric = values.get(budget.metric)
        value = None if metric is None else metric["value"]
        outcome = "unknown"
        if value is not None:
            passed = value <= budget.maximum if budget.maximum is not None else value >= budget.minimum
            if metric["origin"] == "lower_bound":
                outcome = "target_blocked" if budget.maximum is not None and not passed else "unknown"
            elif metric["origin"] == "estimate":
                outcome = "estimated_pass" if passed else "estimated_miss"
            else:
                outcome = "met_in_scope" if passed else "missed_in_scope"
        results.append({**budget.to_spec(), "outcome": outcome, "observed": value})
    return results


def comparable_key(result, *, allow_build_changes=False):
    """Exact v2 cohort boundary. Missing provenance is never inferred from v1."""
    d = result.to_dict()
    if d["schema_version"] != "pllm.benchmark_result.v2" or d["status"] != "completed":
        return None
    e = d["execution"]
    if e["oracle"]["outcome"] != "passed" or not e["isolation"]["fresh_process"]:
        return None
    if any(v is None for v in e["comparison"].values()):
        return None
    return (d["scope"], e["cohort"], e["fixture"]["kind"],
            canonical(e["comparison"]), d["environment"]["digest"],
            None if allow_build_changes else e["build"]["digest"])


def metric_contract(metric):
    return canonical({key: value for key, value in metric.items()
        if key not in {"value", "sample_count", "unavailable_reason"}})


_CLIENT_MEMORY_SPECS = {
    "client.owned_sampled_peak_rss_sum": "estimate",
    "client.owned_peak_rss_lower_bound": "lower_bound",
    "client.owned_lifetime_peak_rss_sum_interval": "interval",
}
_CLIENT_LIFETIME_WINDOW = "isolated_client_group_birth_through_complete_retirement"
_ROOT_ENDPOINT_ID = "client.peak_rss"
_ROOT_ENDPOINT_QUALIFICATION = "root_only_report_endpoint"
_ROOT_ENDPOINT_WINDOW = "client_process_birth_through_report_endpoint"


@dataclass(frozen=True, slots=True)
class MetricSpec:
    """Opt-in client-memory selection, not a new metric executor or component.

    Existing string/component selectors retain their old meaning. The interval
    spec discovers metadata, never synthesizes a scalar measurement. Lifetime
    coverage is available only after the qualified client owner's final reap.
    Explicit qualification="root_only_report_endpoint" selects the existing
    client.peak_rss through a separate guard; None stays canonically omitted.
    """
    id: str
    qualification: str | None = None

    def __post_init__(self):
        endpoint = (self.id == _ROOT_ENDPOINT_ID
            and self.qualification == _ROOT_ENDPOINT_QUALIFICATION)
        if (type(self.id) is not str or not endpoint
                and (self.id not in _CLIENT_MEMORY_SPECS or self.qualification is not None)):
            raise ValueError("unsupported opt-in client memory metric")

    def to_spec(self):
        from pllm.runtime.client_memory_scope import WINDOW
        if self.qualification is not None:
            return {"id": self.id, "qualification": self.qualification, "unit": "bytes",
                "kind": "measured", "window": _ROOT_ENDPOINT_WINDOW,
                "source_setup_costs": "only_inside_isolated_birth_through_report_endpoint",
                "physical_peak": False}
        return {"id": self.id, "unit": "bytes", "kind": _CLIENT_MEMORY_SPECS[self.id],
            "window": _CLIENT_LIFETIME_WINDOW if self.id.endswith("_interval") else WINDOW,
            "source_setup_costs": ("only_inside_declared_isolated_lifetime" if self.id.endswith("_interval")
                else "pre_sampler_resolution_and_import_excluded"),
            "physical_peak": False}


def client_memory_metric_specs():
    """Discover three bounded optional selections; no workload/monitor is run."""
    return tuple(MetricSpec(name) for name in _CLIENT_MEMORY_SPECS)


def client_memory_metric_id(metric):
    if type(metric) is MetricSpec and metric.qualification == _ROOT_ENDPOINT_QUALIFICATION:
        return metric.id
    name = metric.id if type(metric) is MetricSpec else metric
    return name if type(name) is str and name in _CLIENT_MEMORY_SPECS else None


def has_client_memory_metric(document, name):
    if name.endswith("_interval"):
        report = document.get("execution", {}).get("details", {}).get("report")
        scope = report.get("client_owned_memory") if type(report) is dict else None
        return type(scope) is dict and type(scope.get("lifetime_peak_rss_sum_interval")) is dict
    return any(row["id"] == name for row in document["metrics"])


def _client_memory_observation(document, name):
    """Validate trusted normal report data, never demand invented PID receipts."""
    from pllm.runtime.client_memory_scope import HELPERS, SCHEMA, WINDOW
    execution = document["execution"]
    report = execution["details"].get("report", {})
    scope = report.get("client_owned_memory", {})
    if (execution["case"]["target"]["kind"] != "response"
            or report.get("schema_version") != "pllm.loopback_benchmark.v1"
            or report.get("checks", {}).get("passed") is not True
            or scope.get("schema") != SCHEMA or scope.get("window") != WINDOW):
        return None
    source = report.get("configuration", {}).get("source_lock_digest")
    if source is None or source != execution["comparison"]["source_digest"] or source != execution["fixture"]["digest"]:
        return None
    runs = report.get("runs")
    if type(runs) is not list or not 1 <= len(runs) <= 64:
        return None
    bodies = []
    for run in runs:
        tokens = run.get("tokens", {})
        body, output = run.get("model_fingerprint"), run.get("generation", {}).get("output_text_digest")
        if (run.get("status") != "completed" or tokens.get("authoritative") is not True
                or any(type(tokens.get(k)) is not int or not 0 <= tokens[k] <= (1 << 31)
                       for k in ("input_tokens", "output_tokens"))
                or any(type(v) is not str or len(v) != 64 for v in (body, output))):
            return None
        maximum = run.get("max_output_tokens")
        if type(maximum) is not int or not 0 <= maximum <= (1 << 31) or type(run.get("warm")) is not bool:
            return None
        bodies.append((body, output, tokens["input_tokens"], tokens["output_tokens"], maximum, run["warm"]))
    if name == _ROOT_ENDPOINT_ID:
        return _root_endpoint_observation(document, report, scope, bodies)
    if (name.endswith("_interval") and scope.get("lifetime_peak_rss_sum_interval", {}).get("source")
            in {"complete_owned_member_wait4_family_maxima",
                "owner_bound_individual_prefix_lower_and_wait4_family_upper"}):
        from pllm.runtime.client_process_usage import owned_family_interval
        interval = owned_family_interval(scope)
        return ({**interval, "kind": "interval", "window": _CLIENT_LIFETIME_WINDOW,
                 "bodies": canonical(bodies)} if interval is not None else None)
    members = scope.get("latest", {}).get("members")
    if (type(members) is not dict or not 1 <= len(members) <= 9 or "client" not in members
            or any(key not in ("client", *HELPERS) for key in members)):
        return None
    span = scope.get("maximum_sample_span_seconds")
    if (type(scope.get("samples")) is not int or scope["samples"] <= 0
            or type(span) not in (int, float) or not math.isfinite(span) or span < 0):
        return None

    def count(value):
        return type(value) is int and 0 <= value <= (1 << 63) - 1

    complete = (scope.get("membership_declared") is True
        and scope.get("membership_declared_before_sampling") is True
        and scope.get("sampling_coverage") == "all_declared_members_observed"
        and type(scope.get("samples")) is int and scope["samples"] > 0
        and type(scope.get("complete_member_samples")) is int
        and scope.get("complete_member_samples") == scope["samples"]
        and type(scope.get("missing_member_samples")) is int
        and scope.get("missing_member_samples") == 0
        and scope.get("latest", {}).get("missing_members") == []
        and all(type(row) is dict and row.get("reason") is None and count(row.get("rss_bytes"))
                for row in members.values()))
    if name.endswith("_interval"):
        bounds = scope.get("lifetime_peak_rss_sum_interval", {})
        hwms = bounds.get("member_hwm_bytes")
        if (not complete or bounds.get("window") != _CLIENT_LIFETIME_WINDOW
                or bounds.get("metric") != "client_owned_rss_sum_peak"
                or bounds.get("source") != "final_individual_os_hwm"
                or bounds.get("hwm_coverage") != "individual_process_only"
                or bounds.get("finalized_after_full_member_retirement") is not True
                or bounds.get("unit") != "bytes" or bounds.get("isolated_lifetime") is not True
                or bounds.get("membership_complete") is not True or bounds.get("reason") is not None
                or type(hwms) is not dict or set(hwms) != set(members)
                or not all(count(value) for value in hwms.values())
                or bounds.get("lower_bound_bytes") != max(hwms.values())
                or bounds.get("upper_bound_bytes") != sum(hwms.values())):
            return None
        return {"kind": "interval", "lower": bounds["lower_bound_bytes"],
            "upper": bounds["upper_bound_bytes"], "window": _CLIENT_LIFETIME_WINDOW,
            "bodies": canonical(bodies)}
    row = next((row for row in document["metrics"] if row["id"] == name), {})
    kind = _CLIENT_MEMORY_SPECS[name]
    semantics = ("non_atomic_same_pass_owner_declared_RSS_sum" if kind == "estimate"
                 else "maximum_individual_RSS_observation_not_group_peak")
    field = "sampled_peak_rss_sum_bytes" if kind == "estimate" else "sampled_peak_rss_lower_bound_bytes"
    if (not count(scope.get(field)) or type(row.get("value")) not in (int, float)
            or row.get("value") != scope.get(field)
            or row.get("origin") != kind or row.get("unit") != "bytes"
            or row.get("window") != WINDOW or row.get("measurement_scope") != semantics
            or row.get("statistic") != "measurement" or row.get("role") != "client"
            or row.get("component") != "pllm/memory" or row.get("parameters") != {"kind": "peak_rss"}
            or row.get("owners") != ["client"] or row.get("phase") != "full"
            or row.get("sample_count") != 1):
        return None
    if kind == "estimate":
        if (not complete or scope.get("sampled_peak_rss_sum_kind") !=
                "non_atomic_same_pass_estimate_not_simultaneous_peak_bound"
                or scope.get("latest", {}).get("same_pass_rss_sum_bytes") != sum(
                    member["rss_bytes"] for member in members.values())
                or row["value"] < scope["latest"]["same_pass_rss_sum_bytes"]):
            return None
    elif (scope.get("lower_bound_source") != "maximum_individual_RSS_observation"
            or not any(type(member) is dict and count(member.get("rss_bytes")) for member in members.values())
            or row["value"] < max(member["rss_bytes"] for member in members.values()
                if type(member) is dict and count(member.get("rss_bytes")))):
        return None
    return {"kind": kind, "value": row["value"], "window": WINDOW,
            "contract": metric_contract(row), "bodies": canonical(bodies)}


def _root_endpoint_observation(document, report, scope, bodies):
    """Qualify the EXISTING root measurement, never synthesize a group/lifetime HWM.

    Creator/retirement and parent birth binding are trusted normal owner records,
    not a new receipt. The original RUSAGE_SELF endpoint excludes later worker
    serialization/exit and outside-role/controller/source work. No wait4 maximum
    or sampled sum supplies this value, even when it would produce a nicer ratio.
    """
    from pllm.runtime.client_process_usage import WINDOW as lifetime_window
    execution = document["execution"]
    parameters = execution["case"]["target"]["parameters"]
    manifest = scope.get("lifetime_owner", {})
    prefix = scope.get("individual_client_hwm", {})
    members = manifest.get("members")
    if (execution["isolation"]["fresh_process"] is not True
            or execution["isolation"]["controller_included"] is not False
            or parameters.get("state_audit") is not True
            or manifest.get("schema") != "pllm.client_lifetime_owner.v1"
            or manifest.get("membership_complete") is not True
            or manifest.get("root_birth_bound") is not True
            or manifest.get("window") != lifetime_window or manifest.get("reason") is not None
            or manifest.get("classification") != "builtin_in_process_client_owner"
            or "_root_identity" in manifest or type(members) is not dict or set(members) != {"client"}
            or scope.get("membership_declared") is not True
            or set(scope.get("latest", {}).get("members", {})) != {"client"}
            or scope["latest"]["members"]["client"].get("reason") is not None
            or scope["latest"].get("missing_members") != []
            or prefix.get("source") != "resource.getrusage(RUSAGE_SELF)"
            or prefix.get("window") != _ROOT_ENDPOINT_WINDOW
            or prefix.get("coverage") != "individual_process_only_not_client_group"
            or prefix.get("isolated_lifetime") is False):
        return None
    outside = manifest.get("outside_isolated_role")
    if type(outside) is not str or not all(word in outside for word in ("unknown", "controller", "parent_source", "providers")):
        return None
    value = prefix.get("bytes")
    if (type(value) is not int or not 0 <= value <= (1 << 63) - 1
            or report.get("client_process_memory", {}).get("after", {}).get("lifetime_peak_rss_bytes") != value):
        return None
    rows = [row for row in document["metrics"] if row["id"] == _ROOT_ENDPOINT_ID]
    if len(rows) != 1:
        return None
    row = rows[0]
    if (row["value"] != value or row["origin"] != "measured" or row["unit"] != "bytes"
            or row["window"] != "process_birth_through_scenario"
            or row["measurement_scope"] != "process_lifetime_high_water"
            or row["statistic"] != "median" or row["sample_count"] != 1
            or row["role"] != "client" or row["owners"] != ["client"] or row["phase"] != "full"
            or row["component"] != "pllm/memory" or row["parameters"] != {"kind": "peak_rss"}):
        return None
    for run in report["runs"]:
        status = run.get("generation", {}).get("response_status")
        if run["tokens"]["output_tokens"] > run["max_output_tokens"]:
            return None
        if status == "incomplete":
            # API token-limit termination is normal for a strictly bounded
            # benchmark, but counts/reason alone are NOT native acceptance.
            # The existing terminal observation follows completion/cancel
            # barriers; native finally counters corroborate that issued event.
            details = run["generation"].get("incomplete_details")
            checks, counters = report["checks"], run.get("privacy", {})
            maximum = parameters["scenario"]["max_output_tokens"]
            if (type(details) is not dict or details != {"reason": "max_output_tokens"}
                    or any(value is not True for value in checks.values())
                    or any(checks.get(key) is not True for key in (
                        "all_runs_completed", "authoritative_token_usage", "sampling_matches_request",
                        "model_fingerprint_present", "no_plaintext_prompt_bytes_sent",
                        "no_plaintext_token_ids_sent", "no_online_preparation_requests",
                        "expected_input_tokens", "expected_output_tokens"))
                    or any(type(parameters.get(key)) is not int or parameters[key] <= 0
                           for key in ("expected_input_tokens", "expected_output_tokens"))
                    or run["max_output_tokens"] != maximum
                    or parameters["expected_output_tokens"] != maximum
                    or run["tokens"]["output_tokens"] != maximum
                    or type(report["configuration"].get("max_output_tokens")) is not int
                    or report["configuration"].get("max_output_tokens") != maximum
                    or any(type(counters.get(key)) is not int or counters[key] != expected
                           for key, expected in (
                               ("generation_completed_responses", 1),
                               ("generation_output_tokens", maximum),
                               ("generation_requested_output_tokens", maximum)))):
                return None
        elif status != "completed":
            return None
        for key in ("expected_input_tokens", "expected_output_tokens"):
            if key in parameters and (report["checks"].get(key) is not True
                    or run["tokens"][key.removeprefix("expected_")] != parameters[key]):
                return None
    # Reuse the ordinary full-state audit contract, and compare the actual public
    # commitments as well as the existing cohort/output/state digest guards.
    audit = execution["details"].get("response_state_audit", {})
    bindings = audit.get("bindings")
    if (audit.get("schema") != "pllm.response_state_audit.v1"
            or type(bindings) is not list or len(bindings) != 1
            or len(report["runs"]) != 1
            or execution["oracle"]["kind"] != "salted_full_logit_kv_sampler_commitments"):
        return None
    binding, run = bindings[0], report["runs"][0]
    selections = binding.get("selections")
    if (binding.get("failed") is not False or binding.get("temperature") != 0
            or binding.get("numeric_body_fingerprint") != run["model_fingerprint"]
            or type(selections) is not list
            or len(selections) not in {run["tokens"]["output_tokens"], run["tokens"]["output_tokens"] + 1}
            or not 1 <= len(selections) <= run["max_output_tokens"]):
        return None
    for selection in selections:
        if (type(selection) is not dict or type(selection.get("position")) is not int
                or selection["position"] < 0 or type(selection.get("checked_values")) is not int
                or selection["checked_values"] <= 0 or type(selection.get("cache_lengths")) is not list
                or not selection["cache_lengths"]
                or any(type(length) is not int or length < 0 for length in selection["cache_lengths"])):
            return None
        for field in ("logits_and_active_kv", "selected_token", "rng_before", "rng_after"):
            _hash(selection.get(field), field)
    if (binding.get("cryptographic_randomness_frozen") is not False
            or execution["oracle"]["checked_values"] != sum(row["checked_values"] for row in selections)):
        return None
    return {"kind": "endpoint", "value": value, "window": _ROOT_ENDPOINT_WINDOW,
        "contract": metric_contract(row), "bodies": canonical(bodies), "trajectory": canonical(audit)}


def compare_client_memory_pairs(pairs, name, *, allow_build_changes=False, minimum_reduction=10):
    """Extend normal comparison gates; estimates/bounds never become point ranks.

    At most 32 existing replicate pairs and nine member HWMs. No history, monitor,
    receipts or new runner. Fixed summaries use existing report metadata allowance.
    """
    if type(minimum_reduction) is not int or not 1 <= minimum_reduction <= 1_000_000:
        raise ValueError("minimum reduction must be an integer in [1, 1000000]")
    if not 1 <= len(pairs) <= 32:
        return {"status": "unranked", "reason": "missing_case"}
    observations = []
    for left, right in pairs:
        key = comparable_key(left, allow_build_changes=allow_build_changes)
        if key is None or key != comparable_key(right, allow_build_changes=allow_build_changes):
            return {"status": "unranked", "reason": "cohort_contract_differs"}
        try:
            a, b = (_client_memory_observation(result.to_dict(), name) for result in (left, right))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return {"status": "unranked", "reason": "client_memory_scope_unavailable_or_unsafe"}
        if a is None or b is None:
            return {"status": "unranked", "reason": "client_memory_scope_unavailable_or_unsafe"}
        if (a["kind"] != b["kind"] or a["window"] != b["window"] or a["bodies"] != b["bodies"]
                or a.get("contract") != b.get("contract") or a.get("trajectory") != b.get("trajectory")):
            return {"status": "unranked", "reason": "client_memory_scope_contract_differs"}
        if a["kind"] == "interval":
            observations.append({"control": {"lower_bytes": a["lower"], "upper_bytes": a["upper"]},
                "candidate": {"lower_bytes": b["lower"], "upper_bytes": b["upper"]},
                "reduction_established": a["lower"] > 0 and b["upper"] * minimum_reduction <= a["lower"]})
        else:
            observations.append({"control_bytes": a["value"], "candidate_bytes": b["value"]})
    if name == _ROOT_ENDPOINT_ID:
        import statistics
        controls = [row["control_bytes"] for row in observations]
        candidates = [row["candidate_bytes"] for row in observations]
        return {"status": "comparable_in_scope", "metric": name,
            "qualification": _ROOT_ENDPOINT_QUALIFICATION, "window": _ROOT_ENDPOINT_WINDOW,
            "control_median": statistics.median(controls), "candidate_median": statistics.median(candidates),
            "paired_control_over_candidate": [None if b == 0 else a / b for a, b in zip(controls, candidates)],
            "process_pairs": len(observations), "physical_peak": False, "whole_lifetime": False,
            "scope": "individual_client_process_RSS_HWM_through_report_endpoint",
            "outside_isolated_role": "unknown_including_providers_controller_parent_source",
            "reduction_status": "not_established", "confirmation": "separate finalist cohort required"}
    kind = _CLIENT_MEMORY_SPECS[name]
    established = kind == "interval" and all(row["reduction_established"] for row in observations)
    return {"status": "descriptive_only" if kind == "estimate" else "bounded",
        "metric": name, "kind": kind, "pairs": observations, "process_pairs": len(observations),
        "window": a["window"],
        "minimum_reduction": minimum_reduction,
        "reduction_status": "established_in_scope" if established else "not_established",
        "physical_peak": False, "scope": "client_owned_RSS_sum_not_physical_used_RAM"}
