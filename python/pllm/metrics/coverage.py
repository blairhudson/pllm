"""Resolve SDK contracts to benchmark cases without running workloads.

Coverage is scoped to an exact configuration, build and private cohort. Merely
mentioning a component, an unavailable metric, or a partial set of replicates
does not establish a completed measurement. This module never ranks timings.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import json
import re

from pllm._benchmark_contract import BenchmarkSuite, canonical, digest
from pllm._benchmark_evidence import metric_contract
from pllm.evidence import EvidenceRegistry, EvidenceReport


def _subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _subset(value, actual[key]) for key, value in expected.items())
    # JSON booleans must not match integer choices.
    return type(expected) is type(actual) and expected == actual


def _components(value):
    if isinstance(value, dict):
        if set(value) == {"component", "params"} and isinstance(value["params"], dict):
            yield value
        for child in value.values():
            yield from _components(child)
    elif isinstance(value, list):
        for child in value:
            yield from _components(child)


def _target_components(target):
    """Only executable target selections, never incidental model metadata."""
    params = target.get("parameters", {})
    if target.get("kind") == "operator":
        yield from _components(params.get("component"))
    pipeline = params.get("model", {}).get("experiment", {}).get("pipeline", {})
    for slot, value in pipeline.items():
        if slot not in {"model", "metadata"}:
            yield from _components(value)


@dataclass(frozen=True, slots=True, init=False)
class BenchmarkRequirement:
    """An explicit component/parameter or target predicate; never executable code.

    ``target`` is a recursive subset of a BenchmarkTarget specification. Arrays
    match exactly. ``present`` requires explicitly serialized component parameters;
    omitted defaults are not invented. A blocked contract requires a reason.
    """

    _bytes: bytes

    def __init__(self, name, *, component=None, parameters=None, present=(), target=None,
                 disposition="required", reason=None):
        if type(name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:+=-]{0,255}", name):
            raise ValueError("requirement name must be a bounded public identity")
        if component is not None and (type(component) is not str or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._/:+=-]{0,255}", component)):
            raise ValueError("requirement component must be a bounded component identity")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a partial public component configuration")
        if target is not None and not isinstance(target, Mapping):
            raise TypeError("target must be a partial BenchmarkTarget specification")
        if (not isinstance(present, (tuple, list)) or len(present) != len(set(present))
                or any(type(key) is not str or not key for key in present)):
            raise ValueError("present must contain unique parameter names")
        if component is None and (parameters or present):
            raise ValueError("parameter requirements need a component")
        if disposition not in {"required", "unsupported", "declaration"}:
            raise ValueError("invalid requirement disposition")
        if disposition != "required" and (type(reason) is not str or not reason.strip()):
            raise ValueError("unsupported/declaration requirements need an explicit reason")
        if reason is not None and (type(reason) is not str or len(reason) > 2048):
            raise ValueError("requirement reason must be bounded public text")
        if disposition == "required" and component is None and not target:
            raise ValueError("required capability needs a component or target predicate")
        record = dict(name=name, component=component, parameters=dict(parameters or {}),
            present=sorted(present), target=dict(target or {}), disposition=disposition, reason=reason)
        payload = canonical(record)
        if len(payload) > 64 << 10:
            raise ValueError("benchmark requirement exceeds 64 KiB")
        object.__setattr__(self, "_bytes", payload)

    @property
    def name(self):
        return self.to_spec()["name"]

    def to_spec(self):
        return json.loads(self._bytes)

    @classmethod
    def from_spec(cls, spec):
        if not isinstance(spec, Mapping) or set(spec) != {
            "name", "component", "parameters", "present", "target", "disposition", "reason"
        }:
            raise ValueError("invalid benchmark requirement record")
        return cls(**spec)

    def matches(self, target, *, metrics=()):
        spec = self.to_spec()
        if spec["disposition"] != "required" or not _subset(spec["target"], target):
            return False
        if spec["component"] is None:
            return True
        return any(c["component"] == spec["component"]
            and _subset(spec["parameters"], c["params"])
            and all(key in c["params"] for key in spec["present"])
            for c in (*_target_components(target), *metrics))


def benchmark_capabilities(*, providers=(), include_parameters=True, include_planned=True):
    """Discover every registered component, parameter, and finite enum choice.

    Numeric/string/array parameters get an explicit-presence requirement. This
    does not enumerate infinite domains or imply Cartesian interaction coverage.
    Named interactions and non-component APIs use additional requirements.
    """
    from pllm.components import list_components, model_capabilities, planned_components

    rows = []
    for descriptor in list_components(providers=providers):
        component = descriptor.component
        rows.append(BenchmarkRequirement(component, component=component))
        if not include_parameters:
            continue
        schema = descriptor.to_dict()["parameter_schema"]
        for name, field in sorted(schema.get("properties", {}).items()):
            root = f"{component}/parameter/{name}"
            rows.append(BenchmarkRequirement(root, component=component, present=(name,)))
            choices = field.get("enum", [False, True] if field.get("type") == "boolean" else [])
            for value in choices:
                rows.append(BenchmarkRequirement(f"{root}/choice/{digest(value)[:16]}",
                    component=component, parameters={name: value}))
    if include_planned:
        for item in (*planned_components(), *model_capabilities()):
            if item.status == "pending":
                rows.append(BenchmarkRequirement(item.identity, disposition="unsupported", reason=item.gate))
    return tuple(sorted(rows, key=lambda row: row.name))


def _cells(registry, build_digest):
    groups, legacy = defaultdict(list), []
    for result in registry.list():
        record = result.to_dict()
        if record["schema_version"] != "pllm.benchmark_result.v2":
            legacy.append(result.id)
            continue
        e = record["execution"]
        # Never fill missing replicates using another salt, source, environment,
        # sampling plan, suite or implementation.
        key = canonical([record["scope"], record["model"], record["plan_lock_digest"],
            record["privacy_cohort"], record["numeric_cohort"], e["fixture"], e["case_digest"],
            record["configuration_digest"], e["cohort"], e["suite_digest"],
            e["build"]["digest"], e["sampling"], e["comparison"], record["environment"]["digest"]])
        groups[key].append(record)
    cells = []
    for key, records in sorted(groups.items()):
        e = records[0]["execution"]
        expected = e["sampling"]["processes"]
        replicas = [r["execution"]["replicate"] for r in records]
        complete = len(replicas) == expected and set(replicas) == set(range(expected))
        passed = all(r["status"] == "completed" and r["execution"]["oracle"]["outcome"] == "passed"
                     for r in records)
        locked = all(r["execution"]["case"]["build"]["expected_digest"] == e["build"]["digest"]
                     for r in records)
        observed = e["build"]["digest"]
        states = sorted({r["status"] for r in records})
        status = ("duplicate_replicate" if len(replicas) != len(set(replicas)) else
                  "incomplete" if not complete else
                  "blocked" if "blocked" in states else
                  "unsupported" if "unsupported" in states else
                   "failed" if not passed else
                   "not_isolated" if not all(r["execution"]["isolation"]["fresh_process"] for r in records) else
                   "unlocked_build" if not locked else
                  "stale_build" if build_digest is not None and observed != build_digest else "measured")
        metric_maps = [{m["id"]: m for m in r["metrics"]} for r in records]
        available = sorted(set.intersection(*(set(m) for m in metric_maps)))
        measured = [name for name in available if all(m[name]["value"] is not None
                    and m[name]["origin"] in {"measured", "deterministic"}
                    and metric_contract(m[name]) == metric_contract(metric_maps[0][name]) for m in metric_maps)]
        if status == "measured" and not measured:
            status = "no_measured_metrics"
        metric_components = [{"component": metric_maps[0][name]["component"],
            "params": metric_maps[0][name]["parameters"]} for name in measured
            if all(m[name]["parameters"] == metric_maps[0][name]["parameters"] for m in metric_maps)]
        cells.append({"id": digest(json.loads(key)), "case": e["case"], "status": status,
            "complete": complete, "oracle_passed": passed, "build_locked": locked,
            "build_digest": observed, "cohort": e["cohort"], "suite_digest": e["suite_digest"],
             "sampling": e["sampling"], "comparison": e["comparison"],
             "model": records[0]["model"], "scope": records[0]["scope"],
             "plan_lock_digest": records[0]["plan_lock_digest"],
             "privacy_cohort": records[0]["privacy_cohort"],
             "numeric_cohort": records[0]["numeric_cohort"], "fixture": e["fixture"],
             "environment_digest": records[0]["environment"]["digest"],
             "expected_replicates": expected, "observed_replicates": sorted(replicas),
            "result_ids": sorted(r["id"] for r in records), "recorded_statuses": states,
            "failure_codes": sorted({r["execution"]["failure_code"] for r in records
                                     if r["execution"]["failure_code"] is not None}),
            "measured_metrics": measured,
            "metric_semantics": {name: json.loads(metric_contract(metric_maps[0][name])) for name in measured},
            "unmeasured_metrics": sorted(set.union(*(set(m) for m in metric_maps)) - set(measured)),
            "metric_components": metric_components})
    return cells, sorted(legacy)


def benchmark_coverage(*, suites=(), registry=None, requirements=None, build_digest=None):
    """Return immutable capability → case → evidence coverage, with explicit gaps.

    With no build digest, completed evidence retains its recorded build; it is not
    asserted current. Supplying a digest exposes stale measurements. Raw timings
    are never combined or ranked. Successful configuration coverage alone does
    not prove cache hits, every code branch, model quality, or independent roles.
    """
    registry = EvidenceRegistry() if registry is None else registry
    if not isinstance(registry, EvidenceRegistry):
        raise TypeError("coverage requires an EvidenceRegistry")
    if build_digest is not None and (type(build_digest) is not str
            or re.fullmatch("[a-f0-9]{64}", build_digest) is None):
        raise ValueError("build_digest must be a SHA-256 digest")
    suites = tuple(suites)
    if len(suites) > 256 or any(not isinstance(s, BenchmarkSuite) for s in suites):
        raise ValueError("coverage requires at most 256 BenchmarkSuites")
    requirements = benchmark_capabilities() if requirements is None else tuple(requirements)
    if (len(requirements) > 8192 or any(not isinstance(r, BenchmarkRequirement) for r in requirements)
            or len({r.name for r in requirements}) != len(requirements)):
        raise ValueError("coverage requirements must be unique and bounded")
    if len(registry.list()) > 16384:
        raise ValueError("coverage exceeds 16384 evidence records")
    cells, legacy = _cells(registry, build_digest)
    planned = {digest(case.to_spec()): case.to_spec() for suite in suites for case in suite.cases}
    planned.update({digest(cell["case"]): cell["case"] for cell in cells})
    case_coverage = []
    for suite in suites:
        for case in suite.cases:
            key = digest(case.to_spec())
            matches = [cell for cell in cells if digest(cell["case"]) == key
                       and cell["suite_digest"] == suite.digest]
            case_coverage.append({"case_id": key, "suite_digest": suite.digest,
                "sampling": suite.sampling.to_spec(), "evidence_cells": [c["id"] for c in matches],
                "measured": any(c["status"] == "measured" for c in matches)})
    rows = []
    for requirement in requirements:
        spec = requirement.to_spec()
        matches = [cell for cell in cells if requirement.matches(cell["case"]["target"],
                                                                  metrics=cell["metric_components"])]
        cases = sorted({key for key, case in planned.items() if requirement.matches(case["target"])}
                       | {digest(cell["case"]) for cell in matches})
        states = {cell["status"] for cell in matches}
        status = (spec["disposition"] if spec["disposition"] != "required" else
                  next((s for s in ("measured", "stale_build", "incomplete", "blocked", "unsupported",
                      "duplicate_replicate", "not_isolated", "unlocked_build", "failed", "no_measured_metrics") if s in states),
                       "unmeasured" if cases else "unmapped"))
        gaps = [c for c in case_coverage if c["case_id"] in cases and not c["measured"]]
        if status == "measured" and gaps:
            status = "partial"
        rows.append({"requirement": spec, "status": status, "cases": cases,
            "has_measurement": "measured" in states,
            "evidence_cells": [cell["id"] for cell in matches],
            "unmeasured_cases": gaps,
            "measured_kinds": sorted({cell["case"]["target"]["kind"] for cell in matches
                                      if cell["status"] == "measured"}),
            "recorded_statuses": sorted(states)})
    return EvidenceReport(canonical({"schema_version": "pllm.benchmark_coverage.v1",
        "build_digest": build_digest, "requirements": rows, "cases": planned, "cells": cells,
        "case_coverage": case_coverage,
        "counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "legacy_result_ids": legacy,
        "scope": "exact configured benchmark cases; operator, state and response scopes remain distinct",
        "limitations": ["No Cartesian interaction or infinite parameter-domain coverage is inferred.",
            "A configured option does not prove its conditional path was exercised.",
            "Historical v1 records cannot establish exact-build case coverage.",
            "A complete cohort is not independent confirmation or a SOTA ranking."]}))
