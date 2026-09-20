from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pllm.configuration import ComponentRef


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """Immutable bounded semantic graph produced by a model-family adapter."""

    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        document = json.loads(self._canonical_bytes)
        if document.get("schema_version") != "pllm.decoder_plan.v1":
            raise ValueError("invalid decoder model plan")

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"pllm.decoder_plan.v1\0" + self._canonical_bytes).hexdigest()

    @property
    def prefill(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes)["prefill"])

    @property
    def decode(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes)["decode"])

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)

    def coverage(self, profile: str = "research.single_evaluator") -> "DecoderCoverageReport":
        from pllm import _native

        return DecoderCoverageReport(_native.decoder_coverage(self._canonical_bytes, profile))

    def runtime_schedule(
        self, profile: str = "baseline.masked_linear_cpu"
    ) -> "DecoderRuntimeSchedule":
        from pllm import _native

        payload, native_digest = _native.decoder_runtime_schedule(self._canonical_bytes, profile)
        schedule = DecoderRuntimeSchedule(payload)
        if schedule.digest != native_digest:
            raise ValueError("decoder runtime schedule digest mismatch")
        return schedule

    def apply(self, component: "ComponentRef") -> "ModelPlan":
        """Apply one model-graph component and return a new immutable plan."""
        from pllm import _native
        from pllm.configuration import ComponentRef

        if not isinstance(component, ComponentRef):
            raise TypeError("component must be a ComponentRef")
        document = json.dumps(
            component.to_spec(),
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return ModelPlan(_native.apply_model_component(self._canonical_bytes, document))


@dataclass(frozen=True, slots=True)
class DecoderCoverageReport:
    """Compiler coverage that distinguishes primitives from executable operators."""

    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        document = json.loads(self._canonical_bytes)
        if document.get("schema_version") != "pllm.decoder_coverage_report.v1":
            raise ValueError("invalid decoder coverage report")

    @property
    def complete(self) -> bool:
        return bool(json.loads(self._canonical_bytes)["complete"])

    @property
    def operators(self) -> tuple[Mapping[str, Any], ...]:
        return _freeze(json.loads(self._canonical_bytes)["operators"])

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


@dataclass(frozen=True, slots=True)
class DecoderRuntimeSchedule:
    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        document = json.loads(self._canonical_bytes)
        if document.get("schema_version") != "pllm.decoder_runtime_schedule.v1":
            raise ValueError("invalid decoder runtime schedule")
        if not document.get("complete") or document.get("protected_execution"):
            raise ValueError("invalid decoder runtime schedule capability")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            b"pllm.decoder_runtime_schedule.v1\0" + self._canonical_bytes
        ).hexdigest()

    @property
    def profile(self) -> str:
        return str(json.loads(self._canonical_bytes)["profile"])

    @property
    def complete(self) -> bool:
        return bool(json.loads(self._canonical_bytes)["complete"])

    @property
    def protected_execution(self) -> bool:
        return bool(json.loads(self._canonical_bytes)["protected_execution"])

    @property
    def prefill(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes)["prefill"])

    @property
    def decode(self) -> Mapping[str, Any]:
        return _freeze(json.loads(self._canonical_bytes)["decode"])

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)


def lower_model(
    config: Mapping[str, Any] | bytes | str,
    *,
    batch: int,
    max_input_tokens: int,
    max_new_tokens: int,
) -> ModelPlan:
    """Lower a supported Hugging Face model config without loading weights."""
    from pllm import _native

    if isinstance(config, Mapping):
        document = json.dumps(
            config,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    elif type(config) is bytes:
        document = config
    elif type(config) is str:
        document = config.encode()
    else:
        raise TypeError("config must be a mapping, bytes, or JSON string")
    return ModelPlan(
        _native.lower_model(
            document,
            batch,
            max_input_tokens,
            max_new_tokens,
        )
    )


__all__ = ["DecoderCoverageReport", "DecoderRuntimeSchedule", "ModelPlan", "lower_model"]
