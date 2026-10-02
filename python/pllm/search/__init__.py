"""Typed, reproducible search over immutable PLLM experiments."""

from __future__ import annotations

import itertools
import json
import math
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


@dataclass(frozen=True, slots=True)
class QualityLockedNetworkSearch:
    """Compare network costs across numeric variants only after a locked quality gate.

    The quality report covers same-token prefill, not generation quality. This
    opt-in comparison deliberately permits different W4/W8 body fingerprints
    while requiring the same immutable checkpoint and transport workload.
    """

    minimum_top1_agreement: float
    minimum_top_k_recall: float
    maximum_abs_logit_error: float

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_top1_agreement", self.minimum_top1_agreement),
            ("minimum_top_k_recall", self.minimum_top_k_recall),
        ):
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 <= value <= 1:
                raise SearchError(f"{name} must be a finite fraction in [0, 1]")
        if (
            type(self.maximum_abs_logit_error) not in {int, float}
            or not math.isfinite(self.maximum_abs_logit_error)
            or self.maximum_abs_logit_error < 0
        ):
            raise SearchError("maximum_abs_logit_error must be finite and non-negative")

    def select(
        self,
        quality_report: Mapping[str, Any],
        candidates: Sequence[tuple[Experiment, Mapping[str, Any]]],
    ) -> dict[str, Any]:
        from pllm.model_loader import expected_model_id

        if quality_report.get("schema_version") != "pllm.reference_quality_benchmark.v1":
            raise SearchError("quality report must be a versioned executed reference cohort")
        if quality_report.get("scope") != "local-compiled-clear-kernel-prefill-reference":
            raise SearchError("quality report has unsupported execution scope")
        source = quality_report.get("model")
        cohort = quality_report.get("cohort")
        if not isinstance(source, Mapping) or not isinstance(cohort, Mapping):
            raise SearchError("quality report is missing source and cohort locks")
        source_lock = source.get("source_lock_digest")
        if type(source_lock) is not str or re.fullmatch(r"[0-9a-f]{64}", source_lock) is None:
            raise SearchError("quality checkpoint requires a source lock digest")
        if (
            type(cohort.get("dataset_digest")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", cohort["dataset_digest"]) is None
            or type(cohort.get("token_cohort_digest")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", cohort["token_cohort_digest"]) is None
            or type(cohort.get("prompt_count")) is not int or cohort["prompt_count"] < 1
        ):
            raise SearchError("quality dataset and token cohort must be locked and nonempty")
        metric = cohort.get("metric")
        checkpoint = source.get("checkpoint_digest")
        if (
            type(checkpoint) is not str or re.fullmatch(r"[0-9a-f]{64}", checkpoint) is None
            or not isinstance(metric, Mapping)
            or metric.get("component") != "pllm/reference-agreement/v1"
            or not isinstance(metric.get("params"), Mapping)
            or metric["params"].get("dataset_digest") != cohort["dataset_digest"]
            or metric["params"].get("reference_checkpoint_digest") != checkpoint
        ):
            raise SearchError("quality metric does not bind the same dataset and checkpoint")
        quality_rows = quality_report.get("candidates")
        if not isinstance(quality_rows, list) or not quality_rows:
            raise SearchError("quality report has no evaluated numeric candidates")
        by_digest: dict[str, Mapping[str, Any]] = {}
        for row in quality_rows:
            if not isinstance(row, Mapping):
                raise SearchError("quality candidate is malformed")
            identity = row.get("configuration_digest")
            if type(identity) is not str or re.fullmatch(r"[0-9a-f]{64}", identity) is None:
                raise SearchError("quality candidate configuration digest is invalid")
            if identity in by_digest:
                raise SearchError("quality candidate configuration is repeated")
            for name in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
                value = row.get(name)
                if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
                    raise SearchError(f"quality candidate {name} is not finite and non-negative")
            if (
                row["top1_agreement"] > 1 or row["top_k_recall"] > 1
                or type(row.get("sample_count")) is not int
                or row["sample_count"] != cohort["prompt_count"]
            ):
                raise SearchError("quality candidate has incomplete or invalid sample coverage")
            by_digest[identity] = row
        if not isinstance(candidates, Sequence) or not candidates or len(candidates) > 8:
            raise SearchError("network search requires one to eight measured Experiments")

        cohort_key = None
        identities: set[str] = set()
        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []
        for experiment, report in candidates:
            if not isinstance(experiment, Experiment) or not isinstance(report, Mapping):
                raise SearchError("network candidates must be Experiment/report pairs")
            digest = experiment.configuration_digest()
            if digest in identities:
                raise SearchError("network candidate configuration is repeated")
            identities.add(digest)
            pipeline_digest = experiment.pipeline.digest()
            match = by_digest.get(digest)
            if match is None or match.get("pipeline_digest") != pipeline_digest:
                raise SearchError("network candidate has no exact matching quality evaluation")
            if expected_model_id(experiment.pipeline.model) != source.get("id"):
                raise SearchError("quality report targets another model source")
            configuration = report.get("configuration")
            measured = report.get("experiment")
            runs = report.get("runs")
            if (
                not isinstance(configuration, Mapping)
                or report.get("schema_version") != "pllm.loopback_benchmark.v1"
                or configuration.get("source_lock_digest") != source_lock
                or type(configuration.get("prompt_digest")) is not str
                or type(configuration.get("warmup_prompt_digest")) is not str
                or not isinstance(measured, Mapping)
                or measured.get("configuration_digest") != digest
                or measured.get("pipeline_digest") != pipeline_digest
                or report.get("checks", {}).get("passed") is not True
                or not isinstance(runs, list) or not runs
            ):
                raise SearchError("network candidate lacks a matching completed checkpoint-bound run")
            run_keys = {
                (
                    run.get("tokens", {}).get("input_tokens"),
                    run.get("tokens", {}).get("output_tokens"),
                    run.get("max_output_tokens"), run.get("warm"))
                for run in runs
            }
            if len(run_keys) != 1 or None in next(iter(run_keys)):
                raise SearchError("network candidate run token cohort is inconsistent")
            current = (
                source_lock, tuple(configuration.get("roles", ())),
                experiment.pipeline.components["linear"].component,
                experiment.pipeline.components["kernels"].component,
                configuration.get("warmup_prompt_digest"), configuration.get("prompt_digest"),
                configuration.get("warmups"), configuration.get("repetitions"),
                configuration.get("inventory_policy"), configuration.get("bundle_compression"),
                configuration.get("prefill_cache_mib"), configuration.get("prefill_cache_mode"),
                configuration.get("prefill_cache_bound_tokens"), next(iter(run_keys)),
            )
            if cohort_key is None:
                cohort_key = current
            elif current != cohort_key:
                raise SearchError("network candidates are not a matched transport workload")
            if (
                match["top1_agreement"] < self.minimum_top1_agreement
                or match["top_k_recall"] < self.minimum_top_k_recall
                or match["max_abs_logit_error"] > self.maximum_abs_logit_error
            ):
                rejected.append({"configuration_digest": digest, "reason": "quality threshold"})
                continue
            body = report.get("summary", {}).get("median_online_all_link_serialized_body_bytes")
            if type(body) not in {int, float} or not math.isfinite(body) or body < 0:
                rejected.append({"configuration_digest": digest, "reason": "online bodies unmeasured"})
                continue
            selected.append({
                "name": experiment.name,
                "configuration_digest": digest,
                "online_all_link_serialized_body_bytes": body,
                "quality_top1_agreement": match["top1_agreement"],
            })
        selected.sort(key=lambda item: (
            item["online_all_link_serialized_body_bytes"], item["configuration_digest"],
        ))
        return {
            "schema": "pllm.quality_locked_network_search.v1",
            "scope": "same-checkpoint quality-qualified prefill cohort; covered online bodies only",
            "source_lock_digest": source_lock,
            "dataset_digest": cohort["dataset_digest"],
            "token_cohort_digest": cohort["token_cohort_digest"],
            "qualified": selected,
            "rejected": rejected,
            "winner_configuration_digest": selected[0]["configuration_digest"] if selected else None,
            "full_wire_measured": False,
            "generation_quality_established": False,
        }


from pllm.search.placement import ArtifactCostEvidence, CandidateCostEvidence, PlanningPolicy, PlanningRequest

__all__ = [
    "CandidateCostEvidence",
    "ArtifactCostEvidence",
    "PlanningPolicy",
    "PlanningRequest",
    "Constraint",
    "GridSearch",
    "ParetoFrontier",
    "QualityLockedNetworkSearch",
    "RandomSearch",
    "SearchCandidate",
    "SearchError",
    "SearchEvaluation",
    "SearchSpace",
    "evaluate_search",
]
