from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pllm.plan import CompiledPlan, _unwrap


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class EvidenceReport:
    """Immutable, canonical benchmark or assurance evidence."""

    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        document = json.loads(self._canonical_bytes)
        if not isinstance(document, dict) or not isinstance(document.get("schema_version"), str):
            raise ValueError("evidence report must be a versioned JSON object")

    @property
    def schema_version(self) -> str:
        return str(json.loads(self._canonical_bytes)["schema_version"])

    @property
    def data(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes))

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


def benchmark(
    plan: CompiledPlan,
    *,
    weights: bytes,
    input: bytes,
    id: str,
    privacy_cohort: str,
    numeric_cohort: str,
    environment: Mapping[str, Any],
    warmups: int = 3,
    repetitions: int = 10,
    threads: int = 1,
    simd: bool = True,
) -> EvidenceReport:
    """Benchmark one native compiled region against its scalar oracle."""
    from pllm import _native

    if not isinstance(plan, CompiledPlan):
        raise TypeError("plan must be returned by pllm.compile")
    if type(weights) is not bytes or type(input) is not bytes:
        raise TypeError("weights and input must be immutable bytes")
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    return EvidenceReport(
        _unwrap(plan).benchmark_wrap32(
            weights,
            input,
            id,
            privacy_cohort,
            numeric_cohort,
            _canonical(environment),
            warmups,
            repetitions,
            threads,
            simd,
        )
    )


def deployment_benchmark(request: Mapping[str, Any]) -> EvidenceReport:
    """Validate deployment observations and return canonical benchmark evidence."""
    from pllm import _native

    if not isinstance(request, Mapping):
        raise TypeError("request must be a mapping")
    return EvidenceReport(_native.deployment_benchmark_report(_canonical(request)))


def assure() -> EvidenceReport:
    """Run deterministic native assurance fixtures, including negative controls."""
    from pllm import _native

    return EvidenceReport(_native.assurance_report())


__all__ = ["EvidenceReport", "assure", "benchmark", "deployment_benchmark"]
