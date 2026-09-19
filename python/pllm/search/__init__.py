"""Typed, reproducible search over immutable PLLM experiments."""

from __future__ import annotations

import itertools
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pllm.configuration import ConfigurationError, Experiment
from pllm.evidence import BenchmarkResult

_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]{0,255}$")
_MAX_CANDIDATES = 1_000_000


class SearchError(ValueError):
    pass


def _plain(value: Any) -> Any:
    if hasattr(value, "to_spec"):
        return value.to_spec()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _key(value: Any) -> str:
    try:
        return json.dumps(
            _plain(value),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise SearchError("search values must be finite public configuration values") from exc


@dataclass(frozen=True, slots=True)
class Constraint:
    path: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or self.path.startswith("_"):
            raise SearchError("constraint path is invalid")
        if self.operator not in {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"}:
            raise SearchError("constraint operator is unsupported")
        _key(self.value)

    def accepts(self, experiment: Experiment) -> bool:
        params = experiment.get_params(deep=True)
        if self.path not in params:
            raise SearchError(f"constraint path is unknown: {self.path}")
        actual = params[self.path]
        try:
            if self.operator == "eq":
                return actual == self.value
            if self.operator == "ne":
                return actual != self.value
            if self.operator == "lt":
                return actual < self.value
            if self.operator == "le":
                return actual <= self.value
            if self.operator == "gt":
                return actual > self.value
            if self.operator == "ge":
                return actual >= self.value
            if self.operator == "in":
                return actual in self.value
            return actual not in self.value
        except TypeError as exc:
            raise SearchError(f"constraint cannot compare path {self.path!r}") from exc


@dataclass(frozen=True, slots=True, init=False)
class SearchSpace:
    base: Experiment
    parameters: Mapping[str, tuple[Any, ...]]
    constraints: tuple[Constraint, ...]

    def __init__(
        self,
        base: Experiment,
        parameters: Mapping[str, Sequence[Any]],
        *,
        constraints: Iterable[Constraint] = (),
    ) -> None:
        if not isinstance(base, Experiment):
            raise TypeError("base must be an Experiment")
        if not isinstance(parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        base_params = base.get_params(deep=True)
        normalized: dict[str, tuple[Any, ...]] = {}
        for path, values in sorted(parameters.items()):
            if type(path) is not str or path not in base_params:
                raise SearchError(f"search parameter path is unknown: {path!r}")
            if type(values) in {str, bytes} or not isinstance(values, Sequence) or not values:
                raise SearchError(f"search parameter {path!r} must have nonempty discrete values")
            values = tuple(values)
            keys = tuple(_key(value) for value in values)
            if len(keys) != len(set(keys)):
                raise SearchError(f"search parameter {path!r} repeats a value")
            for value in values:
                try:
                    base.with_params(**{path: value})
                except (ConfigurationError, TypeError, ValueError) as exc:
                    raise SearchError(f"search parameter {path!r} contains an invalid value") from exc
            normalized[path] = values
        normalized_constraints = tuple(constraints)
        if any(not isinstance(constraint, Constraint) for constraint in normalized_constraints):
            raise TypeError("constraints must contain Constraint objects")
        for constraint in normalized_constraints:
            constraint.accepts(base)
        cardinality = math_product(len(values) for values in normalized.values())
        if cardinality > _MAX_CANDIDATES:
            raise SearchError(f"search space exceeds {_MAX_CANDIDATES} candidates")
        object.__setattr__(self, "base", base)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))
        object.__setattr__(self, "constraints", normalized_constraints)


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    trial_id: str
    index: int
    experiment: Experiment
    parameters: Mapping[str, Any]
    configuration_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class SearchEvaluation:
    candidate: SearchCandidate
    result: BenchmarkResult


def math_product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _grid(space: SearchSpace, identity: str) -> tuple[SearchCandidate, ...]:
    if type(identity) is not str or _IDENTITY.fullmatch(identity) is None:
        raise SearchError("search identity is invalid")
    paths = tuple(space.parameters)
    combinations = itertools.product(*(space.parameters[path] for path in paths))
    candidates = []
    digests: set[str] = set()
    for values in combinations:
        parameters = dict(zip(paths, values, strict=True))
        try:
            experiment = space.base.with_params(**parameters)
        except (ConfigurationError, TypeError, ValueError) as exc:
            raise SearchError("combined search parameters produce an invalid experiment") from exc
        if not all(constraint.accepts(experiment) for constraint in space.constraints):
            continue
        digest = experiment.configuration_digest()
        if digest in digests:
            raise SearchError("search parameters produce duplicate configurations")
        digests.add(digest)
        index = len(candidates)
        candidates.append(
            SearchCandidate(
                trial_id=f"{identity}:{index}:{digest[:16]}",
                index=index,
                experiment=experiment,
                parameters=parameters,
                configuration_digest=digest,
            )
        )
    return tuple(candidates)


@dataclass(frozen=True, slots=True)
class GridSearch:
    identity: str
    space: SearchSpace

    def __post_init__(self) -> None:
        if not isinstance(self.space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        if type(self.identity) is not str or _IDENTITY.fullmatch(self.identity) is None:
            raise SearchError("search identity is invalid")

    def candidates(self) -> tuple[SearchCandidate, ...]:
        return _grid(self.space, self.identity)

    def evaluate(
        self, evaluator: Callable[[SearchCandidate], BenchmarkResult]
    ) -> tuple[SearchEvaluation, ...]:
        return evaluate_search(self.candidates(), evaluator)


@dataclass(frozen=True, slots=True, init=False)
class RandomSearch:
    identity: str
    space: SearchSpace
    count: int
    seed: int

    def __init__(
        self,
        identity: str,
        space: SearchSpace,
        *,
        count: int,
        seed: int,
    ) -> None:
        if not isinstance(space, SearchSpace):
            raise TypeError("space must be a SearchSpace")
        if type(count) is not int or count < 1:
            raise SearchError("count must be a positive integer")
        if type(seed) is not int:
            raise SearchError("seed must be an integer")
        if type(identity) is not str or _IDENTITY.fullmatch(identity) is None:
            raise SearchError("search identity is invalid")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "seed", seed)

    def candidates(self) -> tuple[SearchCandidate, ...]:
        candidates = _grid(self.space, self.identity)
        if self.count > len(candidates):
            raise SearchError("count exceeds the number of compatible candidates")
        indexes = random.Random(self.seed).sample(range(len(candidates)), self.count)
        return tuple(candidates[index] for index in indexes)

    def evaluate(
        self, evaluator: Callable[[SearchCandidate], BenchmarkResult]
    ) -> tuple[SearchEvaluation, ...]:
        return evaluate_search(self.candidates(), evaluator)


def evaluate_search(
    candidates: Iterable[SearchCandidate],
    evaluator: Callable[[SearchCandidate], BenchmarkResult],
) -> tuple[SearchEvaluation, ...]:
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    evaluations = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, SearchCandidate):
            raise TypeError("candidates must contain SearchCandidate objects")
        if candidate.configuration_digest in seen:
            raise SearchError("candidate configuration was evaluated more than once")
        seen.add(candidate.configuration_digest)
        result = evaluator(candidate)
        if not isinstance(result, BenchmarkResult):
            raise TypeError("evaluator must return BenchmarkResult")
        document = result.to_dict()
        expected_model = (
            candidate.experiment.pipeline.model.model_id
            or candidate.experiment.pipeline.model.source
        )
        required_components = {
            component.component
            for component in candidate.experiment.pipeline.components.values()
        }
        if result.id != candidate.trial_id:
            raise SearchError("benchmark result id does not match candidate trial id")
        if document["configuration_digest"] != candidate.configuration_digest:
            raise SearchError("benchmark result configuration does not match candidate")
        if document["profile"] != candidate.experiment.pipeline.profile:
            raise SearchError("benchmark result profile does not match candidate")
        if document["model"]["id"] != expected_model:
            raise SearchError("benchmark result model does not match candidate")
        if not required_components.issubset(document["component_ids"]):
            raise SearchError("benchmark result omits candidate components")
        evaluations.append(SearchEvaluation(candidate, result))
    return tuple(evaluations)


@dataclass(frozen=True, slots=True, init=False)
class ParetoFrontier:
    objectives: Mapping[str, str]
    _frontier: tuple[SearchEvaluation, ...]
    _excluded: tuple[SearchEvaluation, ...]

    def __init__(
        self,
        evaluations: Iterable[SearchEvaluation],
        *,
        objectives: Mapping[str, str],
    ) -> None:
        if not isinstance(objectives, Mapping) or not objectives:
            raise SearchError("objectives must be a nonempty mapping")
        normalized = dict(sorted(objectives.items()))
        if any(direction not in {"min", "max"} for direction in normalized.values()):
            raise SearchError("objective directions must be min or max")
        values = tuple(evaluations)
        if any(not isinstance(value, SearchEvaluation) for value in values):
            raise TypeError("evaluations must contain SearchEvaluation objects")
        cohort = None
        objective_contract = None
        ranked: list[tuple[SearchEvaluation, tuple[float, ...]]] = []
        excluded = []
        for evaluation in values:
            document = evaluation.result.to_dict()
            current_cohort = (
                document["scope"],
                document["profile"],
                json.dumps(document["model"], sort_keys=True, separators=(",", ":")),
                document["workload_digest"],
                document["environment"]["digest"],
                document["privacy_cohort"],
                document["numeric_cohort"],
                document["warmups"],
                document["repetitions"],
            )
            if cohort is None:
                cohort = current_cohort
            elif current_cohort != cohort:
                raise SearchError("Pareto inputs do not share an exact comparison cohort")
            by_id = {metric["id"]: metric for metric in document["metrics"]}
            if all(identity in by_id for identity in normalized):
                current_contract = tuple(
                    _key({
                        key: by_id[identity][key]
                        for key in (
                            "component",
                            "parameters",
                            "unit",
                            "unit_detail",
                            "statistic",
                            "phase",
                            "role",
                        )
                    })
                    for identity in normalized
                )
                if objective_contract is None:
                    objective_contract = current_contract
                elif current_contract != objective_contract:
                    raise SearchError("Pareto objective semantics do not match")
            if document["status"] != "completed" or any(
                identity not in by_id
                or by_id[identity]["value"] is None
                or by_id[identity]["origin"] == "not_available"
                for identity in normalized
            ):
                excluded.append(evaluation)
                continue
            ranked.append(
                (
                    evaluation,
                    tuple(float(by_id[identity]["value"]) for identity in normalized),
                )
            )
        frontier = []
        directions = tuple(normalized.values())
        for candidate, candidate_values in ranked:
            dominated = False
            for other, other_values in ranked:
                if other is candidate:
                    continue
                no_worse = all(
                    right <= left if direction == "min" else right >= left
                    for left, right, direction in zip(
                        candidate_values, other_values, directions, strict=True
                    )
                )
                strictly_better = any(
                    right < left if direction == "min" else right > left
                    for left, right, direction in zip(
                        candidate_values, other_values, directions, strict=True
                    )
                )
                if no_worse and strictly_better:
                    dominated = True
                    break
            if not dominated:
                frontier.append(candidate)
        object.__setattr__(self, "objectives", MappingProxyType(normalized))
        object.__setattr__(self, "_frontier", tuple(frontier))
        object.__setattr__(self, "_excluded", tuple(excluded))

    def results(self) -> tuple[SearchEvaluation, ...]:
        return self._frontier

    def excluded(self) -> tuple[SearchEvaluation, ...]:
        return self._excluded


__all__ = [
    "Constraint",
    "GridSearch",
    "ParetoFrontier",
    "RandomSearch",
    "SearchCandidate",
    "SearchError",
    "SearchEvaluation",
    "SearchSpace",
    "evaluate_search",
]
