"""Public persistent-state protocol declarations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending
from pllm.components._model_capabilities import PendingModelCapability, model_capability_stub

if TYPE_CHECKING:
    from pllm.runtime.semantic_hybrid import (
        BoundedDepthwiseCausalConvolution, BoundedGatedDeltaRecurrence,
        HybridStateError, bounded_gated_delta_decay,
    )


def __getattr__(name: str) -> object:
    if name in {
        "BoundedDepthwiseCausalConvolution", "BoundedGatedDeltaRecurrence",
        "HybridStateError", "bounded_gated_delta_decay",
    }:
        from pllm.runtime.semantic_hybrid import (
            BoundedDepthwiseCausalConvolution, BoundedGatedDeltaRecurrence,
            HybridStateError, bounded_gated_delta_decay,
        )

        return {
            "BoundedDepthwiseCausalConvolution": BoundedDepthwiseCausalConvolution,
            "BoundedGatedDeltaRecurrence": BoundedGatedDeltaRecurrence,
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
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/client-local-kv",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/state-protocol",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("fixed-capacity-client-kv",),
        role_eligibility=("client",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

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
            },
            "additionalProperties": False,
        },
        capabilities=("causal-prefix-kv-reuse", "client-only-state", "repeat-pattern-visible"),
        role_eligibility=("client",),
    )

    def __init__(self, *, max_bytes: int, fixed_input_tokens: int) -> None:
        if type(max_bytes) is not int or not 1 <= max_bytes >> 20 <= 256 or max_bytes % (1 << 20):
            raise ConfigurationError("client prefix cache max_bytes must be whole MiB in [1, 256]")
        if type(fixed_input_tokens) is not int or not 2 <= fixed_input_tokens <= 4096:
            raise ConfigurationError("client prefix fixed_input_tokens must be in [2, 4096]")
        super().__init__(
            self.descriptor.component,
            {"max_bytes": max_bytes, "fixed_input_tokens": fixed_input_tokens},
        )

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
    "BoundedDepthwiseCausalConvolution", "BoundedGatedDeltaRecurrence", "HybridStateError",
    "bounded_gated_delta_decay",
    "ClientLocalKv", "ClientPrefixReuse", "StateProtocol", "EncryptedKvState", "ProtectedKvSelection", "SpeculativePrivateDecode",
    "GatedDeltaRecurrence", "CausalConvolutionState", "LatentAttentionState", "LongRopeCacheRerotation",
]

install_planned_components(globals())
