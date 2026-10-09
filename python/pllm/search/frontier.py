"""Bounded benchmark propagation without inheriting timings across builds."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import fnmatch
import json

from pllm._benchmark_contract import BenchmarkCase, BenchmarkSuite, canonical, digest, identity
from pllm.evidence import EvidenceRegistry, EvidenceReport


@dataclass(frozen=True, slots=True, init=False)
class Variant:
    name: str
    _patch: bytes
    targets: tuple[str, ...]

    def __init__(self, name, parameters, *, targets=()):
        identity(name, "variant name")
        if not isinstance(parameters, Mapping) or not parameters:
            raise ValueError("variant requires a nonempty with_params patch")
        if any(type(t) is not str for t in targets):
            raise ValueError("variant targets must be case names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_patch", canonical(dict(parameters)))
        object.__setattr__(self, "targets", tuple(targets))

    @property
    def parameters(self):
        return json.loads(self._patch)

    def to_spec(self):
        return {"name": self.name, "parameters": self.parameters, "targets": list(self.targets)}


# Prioritization only. Unknown changes affect every cell. These rules never
# authorize reusing an old timing on a different observed build.
_DEPENDENCIES = {
    "*bundle*": {"artifacts", "response"}, "*paged*": {"artifacts", "preparation", "response"},
    "*preparation*": {"preparation", "stage", "response"},
    "*stage_protocol*": {"stage", "preparation", "response"},
    "*semantic*": {"geometry", "stage", "state", "projection", "response"},
    "*conversation*": {"state", "response"}, "*benchmark*": {"all"},
    "crates/pllm-core/*": {"artifacts", "stage", "preparation", "state", "response"},
}


def _coverage_constraint(target):
    # The planner has neither private inputs nor a source-file reader. Matching
    # software alone cannot attest today's prompt, caller values or checkpoint.
    if target.kind in {"response", "operator"}:
        return "runtime_inputs_require_revalidation"
    parameters = target.to_spec()["parameters"]
    if "model" in parameters:
        from pllm.configuration import Experiment
        model = parameters["model"]
        source = Experiment.from_spec(model["experiment"]).pipeline.model
        if source.kind != "tiny" and (model["config"] is None or parameters.get("weights") == "checkpoint"):
            return "external_source_requires_revalidation"
    return None


@dataclass(frozen=True, slots=True)
class FrontierWork:
    suite: BenchmarkSuite | None
    coverage: EvidenceReport
    input_sources: tuple[tuple[str, str], ...]

    def inputs(self, values):
        """Map caller-owned base-case inputs to selected variants; never serialize."""
        if not isinstance(values, Mapping):
            raise TypeError("inputs must map base case names")
        return {name: values[base] for name, base in self.input_sources if base in values}


@dataclass(frozen=True, slots=True)
class FrontierStudy:
    suite: BenchmarkSuite
    variants: tuple[Variant, ...] = ()
    interactions: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self):
        if not isinstance(self.suite, BenchmarkSuite) or any(not isinstance(v, Variant) for v in self.variants):
            raise TypeError("study requires a suite and immutable variants")
        names = {v.name for v in self.variants}
        if len(names) != len(self.variants) or len(names) > 32 or len(self.interactions) > 32:
            raise ValueError("variant names must be unique and bounded")
        cases = {c.name for c in self.suite.cases}
        if any(set(v.targets) - cases for v in self.variants):
            raise ValueError("variant refers to an unknown base case")
        for interaction in self.interactions:
            if not 2 <= len(interaction) <= 4 or len(set(interaction)) != len(interaction) or set(interaction) - names:
                raise ValueError("interactions require 2..4 distinct declared variants")
        object.__setattr__(self, "variants", tuple(self.variants))
        object.__setattr__(self, "interactions", tuple(tuple(v) for v in self.interactions))
        if len(self.suite.cases) * (1 + len(names) + len(self.interactions) * 5) > 512:
            raise ValueError("study exceeds bounded candidate geometry")

    @property
    def digest(self):
        return digest(self.to_spec())

    def to_spec(self):
        return {"schema": "pllm.frontier_study.v1", "suite": self.suite.to_spec(),
                "variants": [v.to_spec() for v in self.variants],
                "interactions": [list(names) for names in self.interactions]}

    @classmethod
    def from_spec(cls, spec):
        if (not isinstance(spec, Mapping) or set(spec) != {"schema", "suite", "variants", "interactions"}
                or spec["schema"] != "pllm.frontier_study.v1"):
            raise ValueError("invalid frontier study")
        variants = []
        for value in spec["variants"]:
            if not isinstance(value, Mapping) or set(value) != {"name", "parameters", "targets"}:
                raise ValueError("invalid frontier variant")
            variants.append(Variant(value["name"], value["parameters"], targets=value["targets"]))
        return cls(BenchmarkSuite.from_spec(spec["suite"]), tuple(variants), tuple(spec["interactions"]))

    def _cases(self):
        variants = {v.name: v for v in self.variants}
        choices = [(name,) for name in variants] + list(self.interactions)
        # Leave-one-change-out ablations remain visible, including prerequisites.
        for interaction in self.interactions:
            choices.extend(tuple(v for v in interaction if v != omitted) for omitted in interaction)
        choices = tuple(dict.fromkeys(choices))
        output, rejected = [], []
        for base in self.suite.cases:
            output.append((base, base.name))
            for names in choices:
                if any(variants[n].targets and base.name not in variants[n].targets for n in names):
                    continue
                name = base.name + "/" + "+".join(names)
                patch = {}
                conflict = False
                for variant in (variants[n] for n in names):
                    for key, value in variant.parameters.items():
                        if key in patch and patch[key] != value:
                            conflict = True
                        patch[key] = value
                try:
                    if conflict:
                        raise ValueError("conflicting interaction")
                    target = base.target.with_params(**patch)
                    output.append((BenchmarkCase(name, target, base.build, base.budgets), base.name))
                except (ValueError, TypeError, KeyError):
                    rejected.append({"case": name, "base": base.name, "status": "incompatible",
                                     "reason": "variant_contract_rejected"})
        if len(output) > 256:
            raise ValueError("expanded study exceeds 256 cases")
        return output, rejected

    def plan(self, registry=None, *, max_trials=64, changed_paths=()):
        """Inspect exact coverage; every selected neighbor reruns its control.

        BuildRef.capture() pins the current installed build before constructing a
        study. Unpinned specs can execute, but never satisfy current-build coverage.
        """
        if type(max_trials) is not int or not 1 <= max_trials <= 1024:
            raise ValueError("max_trials must be in [1,1024]")
        registry = EvidenceRegistry() if registry is None else registry
        if not isinstance(registry, EvidenceRegistry):
            raise TypeError("coverage requires an EvidenceRegistry")
        cases, coverage = self._cases()
        affected = set()
        for path in changed_paths:
            matches = [v for pattern, v in _DEPENDENCIES.items() if fnmatch.fnmatch(path, pattern)]
            affected.update(set().union(*matches) if matches else {"all"})
        pending, rows = set(), {}
        for case, base in cases:
            matches = []
            old = False
            outcomes = set()
            for result in registry.list():
                d = result.to_dict()
                if d["schema_version"] != "pllm.benchmark_result.v2" or d["configuration_digest"] != case.target.digest:
                    continue
                e = d["execution"]
                old = True
                outcomes.add((d["status"], e["failure_code"]))
                if (case.build.expected_digest == e["build"]["digest"] and d["status"] == "completed"
                        and e["oracle"]["outcome"] == "passed" and e["sampling"] == self.suite.sampling.to_spec()):
                    matches.append(e)
            cohorts = {}
            for e in matches:
                cohorts.setdefault(e["cohort"], set()).add(e["replicate"])
            measured = any(reps == set(range(self.suite.sampling.processes)) for reps in cohorts.values())
            constraint = _coverage_constraint(case.target)
            covered = measured and constraint is None
            status = ("covered" if covered else "revalidation_required" if measured and constraint is not None
                      else "unlocked_build" if case.build.expected_digest is None
                      else "stale_or_incomplete" if old else "missing")
            rows[case.name] = {"case": case.name, "base": base, "status": status,
                "target_digest": case.target.digest, "build_digest": case.build.expected_digest,
                "coverage_constraint": constraint,
                "recorded_outcomes": [{"status": status, "reason": reason}
                                      for status, reason in sorted(outcomes, key=lambda item: (item[0], item[1] or ""))],
                "priority": "affected" if "all" in affected or case.target.kind in affected else "ordinary"}
            if not covered:
                pending.add(case.name)
        selected, used = [], 0
        for base in sorted(self.suite.cases, key=lambda case: rows[case.name]["priority"] != "affected"):
            group = [(case, source) for case, source in cases if source == base.name and case.name in pending]
            if not group:
                continue
            if base.name not in {c.name for c, _ in group}:
                group.insert(0, (base, base.name))
            capacity = (max_trials - used) // self.suite.sampling.processes
            if capacity < min(2, len(group)):
                continue
            # Control first even when its preceding cohort already covered it.
            group.sort(key=lambda pair: (pair[0].name != base.name, rows[pair[0].name]["priority"] != "affected"))
            selected.extend(group[:capacity])
            used += len(group[:capacity]) * self.suite.sampling.processes
        names = {c.name for c, _ in selected}
        for name, row in rows.items():
            row["action"] = "run" if name in names else "budget_deferred" if name in pending else "retain_historical"
        coverage += list(rows.values())
        report = EvidenceReport(canonical({"schema_version": "pllm.frontier_coverage.v1", "study_digest": self.digest,
            "max_trials": max_trials, "selected_trials": used, "cases": coverage,
            "timings_propagated": False, "changed_paths": list(changed_paths)}))
        suite = BenchmarkSuite(tuple(c for c, _ in selected), self.suite.sampling, self.suite.name) if selected else None
        return FrontierWork(suite, report, tuple((c.name, base) for c, base in selected))

    def confirm(self, run, *, cases=None, metrics=None, processes=3):
        """Build a fresh finalist cohort, including every finalist's control.

        By default nominate measured Pareto neighbors. Explicit nominations must
        have complete comparable measurements in the screen. This schedules
        confirmation; it does not confer a performance or quality claim.
        """
        if type(processes) is not int or not 3 <= processes <= 32:
            raise ValueError("confirmation requires 3..32 fresh process replicates")
        screen = self.snapshot(run, metrics=metrics).to_dict()
        measured = {r["case"] for g in screen["groups"] for r in g["observations"]}
        nominated = ({n for g in screen["groups"] for n in g["nondominated"] if n != g["base"]}
                     if cases is None else set(cases))
        if nominated - measured:
            raise ValueError("confirmation nominees require complete comparable screen measurements")
        available, _ = self._cases()
        selected = set(nominated)
        selected.update(base for case, base in available if case.name in nominated)
        pairs = [(case, base) for case, base in available if case.name in selected]
        suite = (BenchmarkSuite(tuple(c for c, _ in pairs), replace(self.suite.sampling, processes=processes),
                                self.suite.name) if pairs else None)
        report = EvidenceReport(canonical({"schema_version": "pllm.frontier_confirmation.v1",
            "study_digest": self.digest, "screen_digest": run.digest,
            "nominees": sorted(nominated), "selected_trials": len(pairs) * processes,
            "status": "ready" if pairs else "no_measured_candidates"}))
        return FrontierWork(suite, report, tuple((c.name, base) for c, base in pairs))

    def snapshot(self, run, *, metrics=None, confirmation=None):
        """Pareto screen only matched cells; unknown measurements remain gaps."""
        from pllm.metrics.benchmark_runner import BenchmarkRun
        from pllm.runtime.benchmark_response_target import SERVING_METRICS
        if not isinstance(run, BenchmarkRun):
            raise TypeError("snapshot requires one exact BenchmarkRun cohort")
        metrics = SERVING_METRICS if metrics is None else dict(metrics)
        if not metrics or any(v not in {"min", "max"} for v in metrics.values()):
            raise ValueError("metrics require explicit min/max directions")
        cases, rejected = self._cases()
        declared = {c.name: c.to_spec() for c, _ in cases}
        for raw in run.to_dict()["suite"]["cases"]:
            if raw != declared.get(raw["name"]):
                raise ValueError("run contains a case outside this study")
        groups = []
        for base in self.suite.cases:
            observations, excluded = [], []
            for case, source in cases:
                if source != base.name:
                    continue
                scores = {}
                for metric in metrics:
                    pair = run.compare(base.name, case.name, metric)
                    if pair["status"] != "comparable":
                        excluded.append({"case": case.name, "metric": metric, "reason": pair["reason"]})
                        break
                    scores[metric] = pair["candidate_median"]
                else:
                    observations.append({"case": case.name, "metrics": scores})
            def dominates(a, b):
                signed = [(a["metrics"][k] - b["metrics"][k]) * (1 if direction == "min" else -1)
                          for k, direction in metrics.items()]
                return all(v <= 0 for v in signed) and any(v < 0 for v in signed)
            frontier = [a["case"] for a in observations if not any(dominates(b, a) for b in observations)]
            groups.append({"base": base.name, "observations": observations, "excluded": excluded,
                           "nondominated": frontier, "status": "exploratory"})
        confirmed = None
        if confirmation is not None:
            if (not isinstance(confirmation, BenchmarkRun) or confirmation.to_dict()["cohort"] == run.to_dict()["cohort"]
                    or confirmation.to_dict()["suite"]["sampling"]["processes"] < 3):
                raise ValueError("confirmation requires a separate cohort with at least three process replicates")
            confirmed = self.snapshot(confirmation, metrics=metrics).to_dict()
        return EvidenceReport(canonical({"schema_version": "pllm.frontier_snapshot.v1", "study_digest": self.digest,
            "cohort": run.to_dict()["cohort"], "run_digest": run.digest, "metrics": metrics,
            "groups": groups, "incompatible": rejected, "confirmation": confirmed,
            "scope": "best measured in each exact cell; no cross-build or cross-lifecycle propagation"}))
