from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pllm.configuration import ComponentRef, Pipeline


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

    def coverage(self, composition: "Pipeline | None" = None) -> "DecoderCoverageReport":
        from pllm import _native

        canonical = None if composition is None else composition.canonical_bytes()
        return DecoderCoverageReport(_native.decoder_coverage(self._canonical_bytes, canonical))

    def runtime_schedule(self, composition: "Pipeline") -> "DecoderRuntimeSchedule":
        from pllm import _native

        payload, native_digest = _native.decoder_runtime_schedule(
            self._canonical_bytes, composition.canonical_bytes()
        )
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

    def continuation_schedule(self, composition: "Pipeline") -> "DecoderContinuationSchedule":
        """Compile an independently digested full-KV batched suffix extension."""
        from pllm import _native

        pipeline = composition.canonical_bytes()
        return DecoderContinuationSchedule(
            _native.decoder_continuation(self._canonical_bytes, pipeline),
            self._canonical_bytes,
            pipeline,
        )


@dataclass(frozen=True, slots=True)
class DecoderContinuationSchedule:
    _canonical_bytes: bytes
    _source_plan: bytes
    _composition: bytes

    def __post_init__(self) -> None:
        from pllm import _native

        if self._canonical_bytes != _native.decoder_continuation(self._source_plan, self._composition):
            raise ValueError("continuation source/numeric/schedule contract mismatch")

    @property
    def digest(self) -> str:
        return hashlib.sha256(b"pllm.decoder_continuation.v1\0" + self._canonical_bytes).hexdigest()

    def canonical_bytes(self) -> bytes:
        return self._canonical_bytes

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical_bytes)

    def handshake_spec(self) -> dict[str, Any]:
        """Public extension slot; contains no private prefix length or tokens."""
        document = self.to_dict()
        return {
            "schema": "pllm.decoder_continuation_session.v1",
            "digest": self.digest,
            **{key: document[key] for key in (
                "source_plan_digest", "model_config_digest", "composition_digest",
                "source_schedule_digest", "numeric_digest", "token_bound",
            )},
        }

    def admit(self, prefix: int, query: int, *, memory_bytes: int = 2 << 30) -> int:
        from pllm import _native

        if any(type(n) is not int or n < 0 for n in (prefix, query, memory_bytes)):
            raise ValueError("continuation bounds must be nonnegative integers")
        return int(_native.admit_decoder_continuation(
            self._source_plan, self._composition, self._canonical_bytes, prefix, query, memory_bytes
        ))


@dataclass(frozen=True, slots=True)
class DecoderCoverageReport:
    """Compiler coverage that distinguishes primitives from executable operators."""

    _canonical_bytes: bytes

    def __post_init__(self) -> None:
        document = json.loads(self._canonical_bytes)
        if document.get("schema_version") != "pllm.decoder_coverage_report.v2":
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
        if document.get("schema_version") != "pllm.decoder_runtime_schedule.v2":
            raise ValueError("invalid decoder runtime schedule")
        if not document.get("complete") or document.get("protected_execution"):
            raise ValueError("invalid decoder runtime schedule capability")

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            b"pllm.decoder_runtime_schedule.v2\0" + self._canonical_bytes
        ).hexdigest()

    @property
    def composition_digest(self) -> str:
        return str(json.loads(self._canonical_bytes)["composition_digest"])

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


__all__ = ["DecoderContinuationSchedule", "DecoderCoverageReport", "DecoderRuntimeSchedule", "ModelPlan", "lower_model"]
