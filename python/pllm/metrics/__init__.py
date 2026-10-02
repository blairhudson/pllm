"""Typed benchmark metric declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _choice(value: object, allowed: set[str], path: str) -> str:
    if type(value) is not str or value not in allowed:
        raise ConfigurationError(f"{path} must be one of {sorted(allowed)}")
    return value


def _text(value: object, path: str) -> str:
    if type(value) is not str or not value or len(value) > 256:
        raise ConfigurationError(f"{path} must be a nonempty string of at most 256 characters")
    return value


class Metric(ComponentRef, ABC):
    __slots__ = ()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class Latency(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/latency",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "statistic": {"enum": ["mean", "median", "p50", "p95", "p99"]},
                "phase": {"enum": ["offline", "online", "full"]},
            },
            "required": ["statistic", "phase"],
            "additionalProperties": False,
        },
        capabilities=("seconds", "lower-is-better"),
    )

    def __init__(self, *, statistic: str = "median", phase: str = "online") -> None:
        super().__init__(
            self.descriptor.component,
            {
                "statistic": _choice(
                    statistic, {"mean", "median", "p50", "p95", "p99"}, "statistic"
                ),
                "phase": _choice(phase, {"offline", "online", "full"}, "phase"),
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Throughput(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/throughput",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {"basis": {"enum": ["tokens", "requests", "regions"]}},
            "required": ["basis"],
            "additionalProperties": False,
        },
        capabilities=("per-second", "higher-is-better"),
    )

    def __init__(self, *, basis: str = "tokens") -> None:
        super().__init__(
            self.descriptor.component,
            {"basis": _choice(basis, {"tokens", "requests", "regions"}, "basis")},
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Communication(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/communication",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "direction": {"enum": ["upload", "download", "total"]},
                "phase": {"enum": ["offline", "online", "full"]},
            },
            "required": ["direction", "phase"],
            "additionalProperties": False,
        },
        capabilities=("bytes", "lower-is-better"),
    )

    def __init__(self, *, direction: str = "total", phase: str = "online") -> None:
        super().__init__(
            self.descriptor.component,
            {
                "direction": _choice(direction, {"upload", "download", "total"}, "direction"),
                "phase": _choice(phase, {"offline", "online", "full"}, "phase"),
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Memory(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/memory",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {"kind": {"enum": ["peak_rss", "allocated", "workspace"]}},
            "required": ["kind"],
            "additionalProperties": False,
        },
        capabilities=("bytes", "lower-is-better"),
    )

    def __init__(self, *, kind: str = "peak_rss") -> None:
        super().__init__(
            self.descriptor.component,
            {"kind": _choice(kind, {"peak_rss", "allocated", "workspace"}, "kind")},
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Energy(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/energy",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {"source": {"enum": ["external_meter", "rapl", "nvml", "not_available"]}},
            "required": ["source"],
            "additionalProperties": False,
        },
        capabilities=("joules", "lower-is-better"),
    )

    def __init__(self, *, source: str = "not_available") -> None:
        super().__init__(
            self.descriptor.component,
            {
                "source": _choice(
                    source,
                    {"external_meter", "rapl", "nvml", "not_available"},
                    "source",
                )
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Accuracy(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/accuracy",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "dataset": {"type": "string", "minLength": 1, "maxLength": 256},
                "measure": {"enum": ["exact_match", "pass_at_1", "task_score"]},
            },
            "required": ["dataset", "measure"],
            "additionalProperties": False,
        },
        capabilities=("ratio", "higher-is-better"),
    )

    def __init__(self, *, dataset: str, measure: str = "task_score") -> None:
        super().__init__(
            self.descriptor.component,
            {
                "dataset": _text(dataset, "dataset"),
                "measure": _choice(measure, {"exact_match", "pass_at_1", "task_score"}, "measure"),
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Perplexity(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/perplexity",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {"dataset": {"type": "string", "minLength": 1, "maxLength": 256}},
            "required": ["dataset"],
            "additionalProperties": False,
        },
        capabilities=("ratio", "lower-is-better"),
    )

    def __init__(self, *, dataset: str) -> None:
        super().__init__(self.descriptor.component, {"dataset": _text(dataset, "dataset")})

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class Cost(Metric):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/cost",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "currency": {"type": "string", "pattern": "^[A-Z]{3}$"},
                "basis": {"enum": ["request", "million_input_tokens", "million_output_tokens"]},
            },
            "required": ["currency", "basis"],
            "additionalProperties": False,
        },
        capabilities=("currency", "lower-is-better"),
    )

    def __init__(self, *, currency: str = "USD", basis: str = "request") -> None:
        if (
            type(currency) is not str
            or len(currency) != 3
            or not currency.isascii()
            or not currency.isupper()
        ):
            raise ConfigurationError("currency must be a three-letter uppercase ASCII code")
        super().__init__(
            self.descriptor.component,
            {
                "currency": currency,
                "basis": _choice(
                    basis,
                    {"request", "million_input_tokens", "million_output_tokens"},
                    "basis",
                ),
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class ReferenceAgreement(Metric):
    """Opt-in same-token FP32 reference top-k agreement, not task accuracy."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/reference-agreement/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "dataset_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "reference_checkpoint_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["dataset_digest", "reference_checkpoint_digest", "top_k"],
            "additionalProperties": False,
        },
        capabilities=("ratio", "higher-is-better", "reference-executed", "same-token"),
    )

    def __init__(
        self, *, dataset_digest: str, reference_checkpoint_digest: str, top_k: int = 5
    ) -> None:
        for name, value in (
            ("dataset_digest", dataset_digest),
            ("reference_checkpoint_digest", reference_checkpoint_digest),
        ):
            if type(value) is not str or _SHA256.fullmatch(value) is None:
                raise ConfigurationError(f"{name} must be a lowercase SHA-256 digest")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ConfigurationError("top_k must be an integer between 1 and 100")
        super().__init__(
            self.descriptor.component,
            {
                "dataset_digest": dataset_digest,
                "reference_checkpoint_digest": reference_checkpoint_digest,
                "top_k": top_k,
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


def measure_reference_agreement(
    candidate_logits: Sequence[float] | np.ndarray,
    reference_logits: Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Compare complete same-token vocabulary logits; retain no logit payload."""
    import numpy as np

    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer between 1 and 100")
    for values in (candidate_logits, reference_logits):
        if isinstance(values, np.ndarray) and (values.ndim != 1 or values.size > 1_000_000):
            raise ValueError("reference logits exceed the bounded vocabulary domain")
        if isinstance(values, Sequence) and len(values) > 1_000_000:
            raise ValueError("reference logits exceed the bounded vocabulary domain")
    try:
        candidate = np.asarray(candidate_logits, dtype=np.float64)
        reference = np.asarray(reference_logits, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("reference logits must be numeric") from exc
    if (
        candidate.ndim != 1
        or reference.ndim != 1
        or candidate.shape != reference.shape
        or candidate.size < top_k
        or candidate.size > 1_000_000
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("reference logits must be matching finite bounded vocabulary vectors")
    # Break ties by vocabulary ID, including highly quantized flat logits.
    ids = np.arange(candidate.size)
    candidate_rank = np.lexsort((ids, -candidate))[:top_k]
    reference_rank = np.lexsort((ids, -reference))[:top_k]
    with np.errstate(over="ignore"):
        max_error = float(np.max(np.abs(candidate - reference)))
    if not np.isfinite(max_error):
        raise ValueError("reference logits difference is outside the finite domain")
    return {
        "top1_agreement": float(candidate_rank[0] == reference_rank[0]),
        "top_k_recall": float(np.isin(reference_rank, candidate_rank).sum() / top_k),
        "max_abs_logit_error": max_error,
    }


from .network_probes import (
    EncryptedLinearCostProbe,
    EncryptedQuadraticShareCostProbe,
    LatentResponseCostProbe,
    ResidentFusedGateCostProbe,
    ResidentMlpCostProbe,
    ResidentQuadraticGateCostProbe,
)
from .projected_polynomial import ProjectedPolynomialCostProbe
from .private_pages import PrivatePageLookupProbe
from .token_budget import TokenNetworkBudgetProbe

__all__ = [
    "Accuracy",
    "Communication",
    "Cost",
    "Energy",
    "EncryptedLinearCostProbe",
    "EncryptedQuadraticShareCostProbe",
    "LatentResponseCostProbe",
    "Latency",
    "Memory",
    "Metric",
    "Perplexity",
    "ProjectedPolynomialCostProbe",
    "PrivatePageLookupProbe",
    "TokenNetworkBudgetProbe",
    "ReferenceAgreement",
    "ResidentMlpCostProbe",
    "ResidentFusedGateCostProbe",
    "ResidentQuadraticGateCostProbe",
    "Throughput",
    "measure_reference_agreement",
]
