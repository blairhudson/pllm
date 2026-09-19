from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

from pllm.plan import CompiledPlan, _unwrap

_BENCHMARK_RESULT_SCHEMA = "pllm.benchmark_result.v1"
_BENCHMARK_RESULT_DOMAIN = b"pllm.benchmark_result.v1\0"
_ENVIRONMENT_DOMAIN = b"pllm.environment.v1\0"
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:+-]{0,255}$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("evidence object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _plain(value),
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
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


def environment_digest(attributes: Mapping[str, Any]) -> str:
    if not isinstance(attributes, Mapping):
        raise TypeError("environment attributes must be a mapping")
    return hashlib.sha256(_ENVIRONMENT_DOMAIN + _canonical(attributes)).hexdigest()


def _identity(value: object, path: str) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{path} must be a bounded identity")
    return value


def _digest(value: object, path: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _keys(value: object, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if set(value) != expected:
        raise ValueError(f"{path} fields do not match the schema")
    return value


def _metric(record: object, index: int) -> dict[str, Any]:
    from pllm.components import create_component
    from pllm.metrics import Metric

    path = f"metrics[{index}]"
    fields = {
        "id",
        "component",
        "parameters",
        "value",
        "unit",
        "unit_detail",
        "statistic",
        "phase",
        "role",
        "origin",
        "unavailable_reason",
        "sample_count",
    }
    value = dict(_keys(record, fields, path))
    _identity(value["id"], f"{path}.id")
    component_id = _identity(value["component"], f"{path}.component")
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping):
        raise ValueError(f"{path}.parameters must be an object")
    component = create_component(component_id, parameters)
    if not isinstance(component, Metric):
        raise ValueError(f"{path}.component is not a built-in metric")
    origin = value["origin"]
    if origin not in {
        "native_executed",
        "reference_executed",
        "imported_archive",
        "source_reported",
        "external_measured",
        "not_available",
    }:
        raise ValueError(f"{path}.origin is unsupported")
    sample_count = value["sample_count"]
    if type(sample_count) is not int or sample_count < 0:
        raise ValueError(f"{path}.sample_count must be a nonnegative integer")
    measurement = value["value"]
    unavailable_reason = value["unavailable_reason"]
    if origin == "not_available":
        if measurement is not None or sample_count != 0:
            raise ValueError(f"{path} unavailable metrics must have null value and zero samples")
        if type(unavailable_reason) is not str or not unavailable_reason:
            raise ValueError(f"{path}.unavailable_reason is required")
    else:
        if (
            not isinstance(measurement, (int, float))
            or isinstance(measurement, bool)
            or not math.isfinite(measurement)
            or measurement < 0
            or sample_count < 1
            or unavailable_reason is not None
        ):
            raise ValueError(f"{path} measured value is invalid")
        value["value"] = float(measurement)
    if value["phase"] not in {"offline", "online", "full"}:
        raise ValueError(f"{path}.phase is unsupported")
    if value["statistic"] not in {
        "measurement",
        "mean",
        "median",
        "p50",
        "p95",
        "p99",
        "sum",
    }:
        raise ValueError(f"{path}.statistic is unsupported")
    role = value["role"]
    if role is not None and (
        type(role) is not str or re.fullmatch(r"[A-Za-z0-9._:/-]+", role) is None
    ):
        raise ValueError(f"{path}.role is invalid")
    expected_units = {
        "pllm/latency": "seconds",
        "pllm/throughput": "per_second",
        "pllm/communication": "bytes",
        "pllm/memory": "bytes",
        "pllm/energy": "joules",
        "pllm/accuracy": "ratio",
        "pllm/perplexity": "ratio",
        "pllm/cost": "currency",
    }
    if value["unit"] != expected_units[component_id]:
        raise ValueError(f"{path}.unit does not match its metric")
    if component_id == "pllm/latency" and (
        value["statistic"] != component.params["statistic"]
        or value["phase"] != component.params["phase"]
    ):
        raise ValueError(f"{path} does not match latency parameters")
    if component_id == "pllm/communication" and value["phase"] != component.params["phase"]:
        raise ValueError(f"{path} does not match communication parameters")
    if component_id == "pllm/throughput":
        expected_detail = f"{component.params['basis']}_per_second"
        if value["unit_detail"] != expected_detail:
            raise ValueError(f"{path}.unit_detail does not match throughput basis")
    elif component_id == "pllm/cost":
        if value["unit_detail"] != component.params["currency"]:
            raise ValueError(f"{path}.unit_detail does not match cost currency")
    elif value["unit_detail"] is not None and (
        type(value["unit_detail"]) is not str or len(value["unit_detail"]) > 128
    ):
        raise ValueError(f"{path}.unit_detail is invalid")
    if component_id == "pllm/energy" and (
        (component.params["source"] == "not_available") != (origin == "not_available")
    ):
        raise ValueError(f"{path} energy source does not match measurement availability")
    if component_id == "pllm/accuracy" and measurement is not None and measurement > 1:
        raise ValueError(f"{path}.value must be a ratio in [0, 1]")
    if component_id == "pllm/perplexity" and measurement is not None and measurement <= 0:
        raise ValueError(f"{path}.value must be positive")
    value["parameters"] = component.get_params()
    return value


def _benchmark_document(document: object) -> dict[str, Any]:
    fields = {
        "schema_version",
        "id",
        "created_at",
        "status",
        "scope",
        "profile",
        "component_ids",
        "model",
        "plan_lock_digest",
        "configuration_digest",
        "privacy_cohort",
        "numeric_cohort",
        "environment",
        "warmups",
        "repetitions",
        "samples",
        "metrics",
        "limitations",
    }
    value = dict(_keys(document, fields, "benchmark result"))
    if value["schema_version"] != _BENCHMARK_RESULT_SCHEMA:
        raise ValueError("unsupported benchmark result schema")
    _identity(value["id"], "id")
    _identity(value["profile"], "profile")
    _identity(value["privacy_cohort"], "privacy_cohort")
    _identity(value["numeric_cohort"], "numeric_cohort")
    try:
        timestamp = datetime.fromisoformat(str(value["created_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be an ISO 8601 timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    if value["status"] not in {"completed", "failed", "not_available"}:
        raise ValueError("status is unsupported")
    if value["scope"] not in {"kernel", "operator", "region", "full_model", "deployment"}:
        raise ValueError("scope is unsupported")
    components = value["component_ids"]
    if not isinstance(components, list) or len(components) != len(set(components)):
        raise ValueError("component_ids must be a unique array")
    value["component_ids"] = sorted(_identity(item, "component_ids") for item in components)
    model = dict(
        _keys(
            value["model"],
            {"id", "checkpoint_digest", "source_lock_digest"},
            "model",
        )
    )
    if (
        type(model["id"]) is not str
        or not model["id"]
        or len(model["id"]) > 512
        or any(ord(character) < 32 for character in model["id"])
    ):
        raise ValueError("model.id is invalid")
    _digest(model["checkpoint_digest"], "model.checkpoint_digest", nullable=True)
    _digest(model["source_lock_digest"], "model.source_lock_digest", nullable=True)
    value["model"] = model
    _digest(value["plan_lock_digest"], "plan_lock_digest", nullable=True)
    _digest(value["configuration_digest"], "configuration_digest", nullable=True)
    environment = dict(
        _keys(value["environment"], {"digest", "attributes"}, "environment")
    )
    attributes = environment["attributes"]
    if not isinstance(attributes, Mapping):
        raise ValueError("environment.attributes must be an object")
    expected_environment = environment_digest(attributes)
    if environment["digest"] != expected_environment:
        raise ValueError("environment digest does not match attributes")
    environment["attributes"] = json.loads(_canonical(attributes))
    value["environment"] = environment
    for name in ("warmups", "repetitions"):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    samples = value["samples"]
    if not isinstance(samples, list):
        raise ValueError("samples must be an array")
    normalized_samples = []
    for index, sample in enumerate(samples):
        sample = dict(
            _keys(
                sample,
                {"index", "kind", "status", "duration_seconds", "artifact_digest"},
                f"samples[{index}]",
            )
        )
        if type(sample["index"]) is not int or sample["index"] < 0:
            raise ValueError("sample index is invalid")
        if sample["kind"] not in {"warmup", "measurement"}:
            raise ValueError("sample kind is invalid")
        if sample["status"] not in {"completed", "failed"}:
            raise ValueError("sample status is invalid")
        duration = sample["duration_seconds"]
        if duration is not None and (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("sample duration is invalid")
        if sample["status"] == "completed" and duration is None:
            raise ValueError("completed samples require a duration")
        _digest(sample["artifact_digest"], "sample artifact_digest", nullable=True)
        normalized_samples.append(sample)
    indexes = sorted(sample["index"] for sample in normalized_samples)
    if indexes != list(range(len(normalized_samples))):
        raise ValueError("sample indexes must be unique and contiguous from zero")
    if sum(sample["kind"] == "warmup" for sample in normalized_samples) != value["warmups"]:
        raise ValueError("warmup count does not match samples")
    if sum(sample["kind"] == "measurement" for sample in normalized_samples) != value["repetitions"]:
        raise ValueError("repetition count does not match samples")
    value["samples"] = sorted(normalized_samples, key=lambda sample: sample["index"])
    metrics = value["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("metrics must be a nonempty array")
    value["metrics"] = sorted(
        (_metric(metric, index) for index, metric in enumerate(metrics)),
        key=lambda metric: metric["id"],
    )
    if len({metric["id"] for metric in value["metrics"]}) != len(value["metrics"]):
        raise ValueError("metric ids must be unique")
    limitations = value["limitations"]
    if (
        not isinstance(limitations, list)
        or not limitations
        or any(type(item) is not str or not item or len(item) > 1024 for item in limitations)
        or len(limitations) != len(set(limitations))
    ):
        raise ValueError("limitations must be a nonempty unique string array")
    value["limitations"] = sorted(limitations)
    if value["status"] == "completed" and (
        value["repetitions"] < 1
        or any(sample["status"] != "completed" for sample in value["samples"])
    ):
        raise ValueError("completed results require completed measurement samples")
    if value["status"] == "not_available" and (
        value["repetitions"] != 0
        or any(metric["origin"] != "not_available" for metric in value["metrics"])
    ):
        raise ValueError("not_available results cannot contain measured evidence")
    return value


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        if type(self._canonical_bytes) is not bytes:
            raise TypeError("benchmark result bytes must be immutable bytes")
        try:
            document = json.loads(self._canonical_bytes)
            normalized = _benchmark_document(document)
            canonical = _canonical(normalized)
        except (json.JSONDecodeError, UnicodeError, TypeError) as exc:
            raise ValueError("benchmark result must be canonical JSON") from exc
        if canonical != self._canonical_bytes:
            raise ValueError("benchmark result bytes are not canonical")

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> BenchmarkResult:
        if not isinstance(document, Mapping):
            raise TypeError("benchmark result must be a mapping")
        return cls(_canonical(_benchmark_document(document)))

    @property
    def id(self) -> str:
        return str(json.loads(self._canonical_bytes)["id"])

    @property
    def digest(self) -> str:
        return hashlib.sha256(_BENCHMARK_RESULT_DOMAIN + self._canonical_bytes).hexdigest()

    @property
    def data(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes))

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


@dataclass(frozen=True, slots=True, init=False)
class EvidenceRegistry:
    _records: Mapping[str, BenchmarkResult]

    def __init__(self, results: Iterable[BenchmarkResult] = ()) -> None:
        records: dict[str, BenchmarkResult] = {}
        for result in results:
            if not isinstance(result, BenchmarkResult):
                raise TypeError("registry entries must be BenchmarkResult objects")
            previous = records.get(result.id)
            if previous is not None and previous.digest != result.digest:
                raise ValueError(f"conflicting benchmark result id {result.id!r}")
            records[result.id] = result
        object.__setattr__(self, "_records", MappingProxyType(dict(sorted(records.items()))))

    def register(self, result: BenchmarkResult) -> EvidenceRegistry:
        if not isinstance(result, BenchmarkResult):
            raise TypeError("result must be a BenchmarkResult")
        previous = self._records.get(result.id)
        if previous is not None:
            if previous.digest != result.digest:
                raise ValueError(f"conflicting benchmark result id {result.id!r}")
            return self
        return EvidenceRegistry((*self._records.values(), result))

    def get(self, identity: str) -> BenchmarkResult:
        try:
            return self._records[identity]
        except KeyError:
            raise KeyError("benchmark result not found") from None

    def list(self) -> tuple[BenchmarkResult, ...]:
        return tuple(self._records.values())

    def query(
        self,
        *,
        profile: str | None = None,
        component: str | None = None,
        model: str | None = None,
        environment_digest: str | None = None,
        privacy_cohort: str | None = None,
        numeric_cohort: str | None = None,
        metric: str | None = None,
        status: str | None = None,
    ) -> tuple[BenchmarkResult, ...]:
        matches = []
        for result in self._records.values():
            document = result.to_dict()
            if profile is not None and document["profile"] != profile:
                continue
            if component is not None and component not in document["component_ids"]:
                continue
            if model is not None and document["model"]["id"] != model:
                continue
            if environment_digest is not None and document["environment"]["digest"] != environment_digest:
                continue
            if privacy_cohort is not None and document["privacy_cohort"] != privacy_cohort:
                continue
            if numeric_cohort is not None and document["numeric_cohort"] != numeric_cohort:
                continue
            if metric is not None and not any(
                measurement["component"] == metric for measurement in document["metrics"]
            ):
                continue
            if status is not None and document["status"] != status:
                continue
            matches.append(result)
        return tuple(matches)


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


__all__ = [
    "BenchmarkResult",
    "EvidenceRegistry",
    "EvidenceReport",
    "assure",
    "benchmark",
    "deployment_benchmark",
    "environment_digest",
]
