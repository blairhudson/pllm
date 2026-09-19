"""Typed benchmark metric declarations."""

from abc import ABC, abstractmethod

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
        distribution="pllm",
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
        distribution="pllm",
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
        distribution="pllm",
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
                "direction": _choice(
                    direction, {"upload", "download", "total"}, "direction"
                ),
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
        distribution="pllm",
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
        distribution="pllm",
        version="1",
        category="pllm/benchmark-metric",
        category_version="1",
        lifecycle_phase="benchmark",
        parameter_schema={
            "type": "object",
            "properties": {
                "source": {"enum": ["external_meter", "rapl", "nvml", "not_available"]}
            },
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
        distribution="pllm",
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
                "measure": _choice(
                    measure, {"exact_match", "pass_at_1", "task_score"}, "measure"
                ),
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
        distribution="pllm",
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
        distribution="pllm",
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
        if type(currency) is not str or len(currency) != 3 or not currency.isascii() or not currency.isupper():
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


__all__ = [
    "Accuracy",
    "Communication",
    "Cost",
    "Energy",
    "Latency",
    "Memory",
    "Metric",
    "Perplexity",
    "Throughput",
]
