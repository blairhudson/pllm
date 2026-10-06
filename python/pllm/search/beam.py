"""Bounded, evidence-guided neighborhood search over ordinary SearchSpace axes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pllm.configuration import ConfigurationError
from pllm.evidence import BenchmarkResult
from pllm.search import (
    _IDENTITY, _MAX_CANDIDATES, _key, _plain, ParetoFrontier, SearchCandidate,
    SearchError, SearchEvaluation, SearchSpace, evaluate_search,
)


class CandidateRejected(Exception):
    """An evaluator's explicit, public reason for rejecting one candidate."""

    def __init__(self, reason: str):
        if type(reason) is not str or not reason.strip() or len(reason) > 1024:
            raise ValueError("candidate rejection needs a bounded public reason")
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class SearchRejection:
    parameters: Mapping[str, Any]
    reason: str
    candidate: SearchCandidate | None = None

    def __post_init__(self):
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Measured trials and explicit stopping scope, never a global optimum proof."""

    evaluations: tuple[SearchEvaluation, ...]
    rejections: tuple[SearchRejection, ...]
    best: SearchEvaluation | None
    objective: str
    direction: str
    stop_reason: str
    attempted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "pllm.search_outcome.v1",
            "objective": self.objective, "direction": self.direction,
            "stop_reason": self.stop_reason, "attempted": self.attempted,
            "best_configuration_digest": self.best.candidate.configuration_digest if self.best else None,
            "evaluations": [{
                "trial_id": value.candidate.trial_id,
                "parameters": _plain(value.candidate.parameters),
                "experiment": value.candidate.experiment.to_spec(),
                "result": value.result.to_dict(),
            } for value in self.evaluations],
            "rejections": [{
                "parameters": _plain(value.parameters), "reason": value.reason,
                "experiment": value.candidate.experiment.to_spec() if value.candidate else None,
            } for value in self.rejections],
            "global_optimum_established": False,
        }


def _rank(evaluations, objective, direction):
    # Reuse the same exact-cohort and metric-semantic gate as grid/random users.
    frontier = ParetoFrontier(evaluations, objectives={objective: direction})
    excluded = {value.candidate.trial_id for value in frontier.excluded()}
    ranked = []
    for evaluation in evaluations:
        if evaluation.candidate.trial_id in excluded:
            continue
        metric = next(value for value in evaluation.result.to_dict()["metrics"] if value["id"] == objective)
        if metric["origin"] not in {
            "native_executed", "reference_executed", "external_measured", "imported_archive",
        }:
            continue
        ranked.append((metric["value"], evaluation))
    # Stable ties retain the incumbent and never imply a measured improvement.
    ranked.sort(key=lambda pair: pair[0], reverse=direction == "max")
    return [evaluation for _, evaluation in ranked]


@dataclass(frozen=True, slots=True)
class BeamSearch:
    """Evaluate an incumbent, then one-axis neighbors of the measured leaders.

    Width retains multiple paths through interactions. The attempt budget counts
    invalid and rejected proposals too. Only visited neighborhoods are built;
    the full Cartesian grid is never materialized. Coupled choices can be one
    whole-Pipeline axis. The base must belong to the space and its constraints.
    """

    identity: str
    space: SearchSpace
    objective: str
    direction: str
    width: int = 2
    max_trials: int = 32

    def __post_init__(self):
        if not isinstance(self.space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        for name in ("identity", "objective"):
            value = getattr(self, name)
            if type(value) is not str or _IDENTITY.fullmatch(value) is None:
                raise SearchError(f"search {name} is invalid")
        if self.direction not in {"min", "max"}:
            raise SearchError("objective direction must be min or max")
        if type(self.max_trials) is not int or not 1 <= self.max_trials <= _MAX_CANDIDATES:
            raise SearchError("max_trials must be a bounded positive integer")
        if type(self.width) is not int or not 1 <= self.width <= self.max_trials:
            raise SearchError("width must be positive and no greater than max_trials")
        paths = tuple(self.space.parameters)
        if any(b.startswith(a + "__") for a in paths for b in paths if a != b):
            raise SearchError("beam axes cannot overlap; group coupled choices in one axis")
        params = self.space.base.get_params(deep=True)
        for path, values in self.space.parameters.items():
            if _key(params[path]) not in {_key(value) for value in values}:
                raise SearchError("beam space must include its incumbent on every axis")
        if not all(constraint.accepts(self.space.base) for constraint in self.space.constraints):
            raise SearchError("beam incumbent must satisfy the search constraints")

    def evaluate(
        self, evaluator: Callable[[SearchCandidate], BenchmarkResult], *,
        admit: Callable[[SearchCandidate], str | None] | None = None,
    ) -> SearchOutcome:
        """Run measured search; admit may return a public rejection reason.

        Evaluator errors propagate unless explicitly raised as CandidateRejected.
        A result with unknown objectives remains evidence but cannot guide search.
        """
        if not callable(evaluator) or (admit is not None and not callable(admit)):
            raise TypeError("evaluator and optional admit must be callable")
        paths = tuple(self.space.parameters)
        axes = tuple(self.space.parameters.values())
        base_params = self.space.base.get_params(deep=True)
        start = tuple(next(i for i, value in enumerate(values) if _key(value) == _key(base_params[path]))
                      for path, values in zip(paths, axes, strict=True))
        visited: set[tuple[int, ...]] = set()
        digests: set[str] = set()
        positions: dict[str, tuple[int, ...]] = {}
        evaluations: list[SearchEvaluation] = []
        rejections: list[SearchRejection] = []

        def attempt(position):
            index = len(visited)
            visited.add(position)
            params = {path: values[i] for path, values, i in zip(paths, axes, position, strict=True)}
            try:
                experiment = self.space.base.with_params(**params)
            except (ConfigurationError, TypeError, ValueError):
                rejections.append(SearchRejection(params, "combined configuration is invalid"))
                return
            digest = experiment.configuration_digest()
            candidate = SearchCandidate(f"{self.identity}:{index}:{digest[:16]}", index,
                                        experiment, params, digest)
            if digest in digests:
                rejections.append(SearchRejection(params, "duplicate configuration", candidate))
                return
            digests.add(digest)
            if not all(constraint.accepts(experiment) for constraint in self.space.constraints):
                rejections.append(SearchRejection(params, "search constraint rejected", candidate))
                return
            try:
                # Resolve immutable component contracts before external execution.
                experiment.resolve()
            except (ConfigurationError, ValueError):
                rejections.append(SearchRejection(params, "component composition is incompatible", candidate))
                return
            if admit is not None:
                reason = admit(candidate)
                if reason is not None:
                    CandidateRejected(reason)  # Validate the public reason's type and bound.
                    rejections.append(SearchRejection(params, reason, candidate))
                    return
            try:
                evaluation, = evaluate_search((candidate,), evaluator)
            except CandidateRejected as exc:
                rejections.append(SearchRejection(params, str(exc), candidate))
                return
            evaluations.append(evaluation)
            positions[candidate.trial_id] = position
            _rank(evaluations, self.objective, self.direction)

        attempt(start)
        stop = "budget"
        while len(visited) < self.max_trials:
            ranked = _rank(evaluations, self.objective, self.direction)
            if not ranked:
                stop = "no_measured_objective"
                break
            beam = [positions[value.candidate.trial_id] for value in ranked[:self.width]]
            before = len(visited)
            # Interleave axes so a long first axis cannot consume the entire budget.
            max_choices = max((len(values) for values in axes), default=0)
            for choice in range(max_choices):
                for axis, values in enumerate(axes):
                    if choice >= len(values):
                        continue
                    for parent in beam:
                        position = (*parent[:axis], choice, *parent[axis + 1:])
                        if position in visited:
                            continue
                        attempt(position)
                        if len(visited) >= self.max_trials:
                            break
                    if len(visited) >= self.max_trials:
                        break
                if len(visited) >= self.max_trials:
                    break
            if len(visited) == before:
                stop = "neighborhood_exhausted"
                break
        ranked = _rank(evaluations, self.objective, self.direction)
        return SearchOutcome(tuple(evaluations), tuple(rejections), ranked[0] if ranked else None,
                             self.objective, self.direction, stop, len(visited))


__all__ = ["BeamSearch", "CandidateRejected", "SearchOutcome", "SearchRejection"]
