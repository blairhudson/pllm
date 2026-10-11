"""Public persistent-state protocol declarations."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending
from pllm.components._model_capabilities import PendingModelCapability, model_capability_stub
from .public_prefix import PublicPrefixCapsule

if TYPE_CHECKING:
    from .checkpoint import CheckpointHead, CheckpointSession
    from .conversation import ConversationCheckpoint
    from pllm.runtime.semantic_hybrid import (
        BoundedDepthwiseCausalConvolution, BoundedGatedDeltaRecurrence, OwnedGatedDeltaRecurrence,
        HybridStateError, bounded_gated_delta_decay,
    )


def __getattr__(name: str) -> object:
    if name == "ConversationCheckpoint":
        from .conversation import ConversationCheckpoint
        return ConversationCheckpoint
    if name in {"CheckpointHead", "CheckpointSession"}:
        from .checkpoint import CheckpointHead, CheckpointSession
        return {"CheckpointHead": CheckpointHead, "CheckpointSession": CheckpointSession}[name]
    if name in {
        "BoundedDepthwiseCausalConvolution", "BoundedGatedDeltaRecurrence", "OwnedGatedDeltaRecurrence",
        "HybridStateError", "bounded_gated_delta_decay",
    }:
        from pllm.runtime.semantic_hybrid import (
            BoundedDepthwiseCausalConvolution, BoundedGatedDeltaRecurrence, OwnedGatedDeltaRecurrence,
            HybridStateError, bounded_gated_delta_decay,
        )

        return {
            "BoundedDepthwiseCausalConvolution": BoundedDepthwiseCausalConvolution,
            "BoundedGatedDeltaRecurrence": BoundedGatedDeltaRecurrence,
            "OwnedGatedDeltaRecurrence": OwnedGatedDeltaRecurrence,
            "HybridStateError": HybridStateError,
            "bounded_gated_delta_decay": bounded_gated_delta_decay,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class StateProtocol(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class ClientLocalKv(StateProtocol):
    """Client-owned KV, optionally retaining a static importance-selected prefix."""
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-local-kv",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/state-protocol",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "properties": {
            "retention": {"type": "object"},
        }, "additionalProperties": False},
        capabilities=("fixed-capacity-client-kv", "static-retained-client-kv"),
        role_eligibility=("client",),
    )

    def __init__(self, *, retention=None) -> None:
        from pllm.passes import KvCacheEviction
        if isinstance(retention, Mapping):
            if set(retention) != {"component", "params"} or retention["component"] != "pllm/kv-cache-eviction":
                raise ConfigurationError("retention requires a KV eviction component")
            retention = KvCacheEviction.from_params(retention["params"])
        if retention is not None:
            if type(retention) is not KvCacheEviction:
                raise ConfigurationError("retention must be KvCacheEviction")
            params = retention.params
            dynamic = params["dynamic_keep"]
            if (params["implementation"] != "pllm/importance-kv-cache-eviction/v1"
                    or dynamic["numerator"] != dynamic["denominator"] or params["share_adjacent_layers"]):
                raise ConfigurationError("client retention requires static selection without layer sharing")
        super().__init__(self.descriptor.component, {} if retention is None else {"retention": retention.to_spec()})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {key: value for key, value in super().get_params(deep).items() if key != "component"}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ClientPrefixReuse(ComponentRef):
    """Byte-bounded trusted-client cache for cost-admitted causal prefill prefixes."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-prefix-reuse/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/cache-optimization",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={
            "type": "object",
            "required": ["max_bytes", "fixed_input_tokens"],
            "properties": {
                "max_bytes": {"type": "integer", "minimum": 1048576, "maximum": 268435456, "multipleOf": 1048576},
                "fixed_input_tokens": {"type": "integer", "minimum": 2, "maximum": 4096},
                "generated_prefixes": {"type": "boolean"},
                "inflight": {"type": "boolean", "const": True},
                "inflight_max_requests": {"type": "integer", "minimum": 2, "maximum": 32},
                "inflight_max_bytes": {"type": "integer", "minimum": 1048576, "maximum": 268435456, "multipleOf": 1048576},
            },
            "additionalProperties": False,
            "dependentRequired": {
                "inflight": ["inflight_max_requests", "inflight_max_bytes"],
                "inflight_max_requests": ["inflight"], "inflight_max_bytes": ["inflight"],
            },
        },
        capabilities=("causal-prefix-kv-reuse", "client-only-state", "repeat-pattern-visible"),
        role_eligibility=("client",),
    )

    def __init__(self, *, max_bytes: int, fixed_input_tokens: int, generated_prefixes: bool = False,
                 inflight: bool = False, inflight_max_requests: int = 8,
                 inflight_max_bytes: int = 64 << 20) -> None:
        if type(max_bytes) is not int or not 1 <= max_bytes >> 20 <= 256 or max_bytes % (1 << 20):
            raise ConfigurationError("client prefix cache max_bytes must be whole MiB in [1, 256]")
        if type(fixed_input_tokens) is not int or not 2 <= fixed_input_tokens <= 4096:
            raise ConfigurationError("client prefix fixed_input_tokens must be in [2, 4096]")
        if type(generated_prefixes) is not bool:
            raise ConfigurationError("generated_prefixes must be a boolean")
        if type(inflight) is not bool:
            raise ConfigurationError("inflight must be a boolean")
        if type(inflight_max_requests) is not int or not 2 <= inflight_max_requests <= 32:
            raise ConfigurationError("inflight_max_requests must be an integer in [2, 32]")
        if (type(inflight_max_bytes) is not int or not 1 <= inflight_max_bytes >> 20 <= 256
                or inflight_max_bytes % (1 << 20)):
            raise ConfigurationError("inflight_max_bytes must be whole MiB in [1, 256]")
        if not inflight and (inflight_max_requests != 8 or inflight_max_bytes != 64 << 20):
            raise ConfigurationError("non-default in-flight limits require inflight=True")
        super().__init__(
            self.descriptor.component,
            {"max_bytes": max_bytes, "fixed_input_tokens": fixed_input_tokens,
              **({"generated_prefixes": True} if generated_prefixes else {}),
              **({"inflight": True, "inflight_max_requests": inflight_max_requests,
                  "inflight_max_bytes": inflight_max_bytes} if inflight else {})},
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"generated_prefixes": False, "inflight": False, "inflight_max_requests": 8,
                "inflight_max_bytes": 64 << 20, **self.params}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor

@pending("cachemir")
class EncryptedKvState(PendingMethod):
    pass


@pending("mpcache")
class ProtectedKvSelection(PendingMethod):
    pass


@pending("private-gpt-no-autoregression")
class SpeculativePrivateDecode(PendingMethod):
    pass


@model_capability_stub("gated-delta")
class GatedDeltaRecurrence(PendingModelCapability):
    pass


@model_capability_stub("causal-convolution")
class CausalConvolutionState(PendingModelCapability):
    pass


@model_capability_stub("latent-attention")
class LatentAttentionState(PendingModelCapability):
    pass


@model_capability_stub("longrope-cache-rerotation")
class LongRopeCacheRerotation(PendingModelCapability):
    pass


__all__ = [
    "ConversationCheckpoint",
    "CheckpointHead", "CheckpointSession",
    "BoundedDepthwiseCausalConvolution", "BoundedGatedDeltaRecurrence", "OwnedGatedDeltaRecurrence", "HybridStateError",
    "bounded_gated_delta_decay",
    "ClientLocalKv", "ClientPrefixReuse", "PublicPrefixCapsule", "StateProtocol", "EncryptedKvState", "ProtectedKvSelection", "SpeculativePrivateDecode",
    "GatedDeltaRecurrence", "CausalConvolutionState", "LatentAttentionState", "LongRopeCacheRerotation",
]

install_planned_components(globals())
