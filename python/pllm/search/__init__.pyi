from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from pllm.configuration import Experiment
from pllm.evidence import BenchmarkResult

class SearchError(ValueError): ...

class Constraint:
    path: str
    operator: str
    value: Any
    def __init__(self, path: str, operator: str, value: Any) -> None: ...
    def accepts(self, experiment: Experiment) -> bool: ...

class SearchSpace:
    base: Experiment
    parameters: Mapping[str, tuple[Any, ...]]
    constraints: tuple[Constraint, ...]
    def __init__(
        self,
        base: Experiment,
        parameters: Mapping[str, Sequence[Any]],
        *,
        constraints: Iterable[Constraint] = ...,
    ) -> None: ...

class SearchCandidate:
    trial_id: str
    index: int
    experiment: Experiment
    parameters: Mapping[str, Any]
    configuration_digest: str

class SearchEvaluation:
    candidate: SearchCandidate
    result: BenchmarkResult

class GridSearch:
    identity: str
    space: SearchSpace
    def __init__(self, identity: str, space: SearchSpace) -> None: ...
    def candidates(self) -> tuple[SearchCandidate, ...]: ...
    def evaluate(
        self, evaluator: Callable[[SearchCandidate], BenchmarkResult]
    ) -> tuple[SearchEvaluation, ...]: ...

class RandomSearch:
    identity: str
    space: SearchSpace
    count: int
    seed: int
    def __init__(
        self, identity: str, space: SearchSpace, *, count: int, seed: int
    ) -> None: ...
    def candidates(self) -> tuple[SearchCandidate, ...]: ...
    def evaluate(
        self, evaluator: Callable[[SearchCandidate], BenchmarkResult]
    ) -> tuple[SearchEvaluation, ...]: ...

class ParetoFrontier:
    objectives: Mapping[str, str]
    def __init__(
        self,
        evaluations: Iterable[SearchEvaluation],
        *,
        objectives: Mapping[str, str],
    ) -> None: ...
    def results(self) -> tuple[SearchEvaluation, ...]: ...
    def excluded(self) -> tuple[SearchEvaluation, ...]: ...

def evaluate_search(
    candidates: Iterable[SearchCandidate],
    evaluator: Callable[[SearchCandidate], BenchmarkResult],
) -> tuple[SearchEvaluation, ...]: ...
