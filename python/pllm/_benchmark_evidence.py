"""Validation and canonical measurements for benchmark_result.v2."""
from __future__ import annotations

from collections.abc import Mapping
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
