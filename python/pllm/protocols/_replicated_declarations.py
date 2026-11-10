"""Model- and method-neutral replicated protocol configuration, not executors.

These declarations provide no runtime handlers, certificates, admission,
provisioning, or access to seeds or checkpoints. Compiler/runtime binding and
independent validation of the selected numerical contracts remain required.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ReplicatedPrivateTokenInput(ComponentRef):
    """Declare replicated private-token input configuration, not input execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-private-token-input/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedCompleteTableLookup(ComponentRef):
    """Declare complete-table replicated lookup configuration, not lookup execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-complete-table-lookup/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedPublicMatmulFullSum(ComponentRef):
    """Declare replicated public matmul with a complete Q36 sum, not execution.

    Fresh stochastic truncation to Q18 must be selected as a separate component
    after the full sum; this declaration performs neither truncation nor rescaling.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-public-matmul-full-sum/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedSecretMultiply(ComponentRef):
    """Declare replicated secret multiplication configuration, not execution.

    Q36 products and separately selected fresh stochastic Q18 truncation remain
    distinct contracts; this declaration does not implement either operation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-secret-multiply/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedRingAddBias(ComponentRef):
    """Declare replicated ring bias-addition configuration, not ring execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-ring-add-bias/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedArithmeticBooleanConversion(ComponentRef):
    """Declare replicated arithmetic/Boolean conversion configuration, not execution."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-arithmetic-boolean-conversion/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedCausalQkSoftmaxAv(ComponentRef):
    """Declare replicated causal QK/softmax/AV configuration, not attention execution.

    Full Q36 sums precede separately selected fresh stochastic Q18 truncation;
    this declaration is not an opaque decoder or a rescaling implementation.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-causal-qk-softmax-av/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedCompleteStableArgmax(ComponentRef):
    """Declare complete-domain replicated argmax with lowest-index tie breaking.

    This configuration declaration neither executes comparisons nor releases
    intermediate logits or token IDs.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-complete-stable-argmax/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/protocols", category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ReplicatedFixedCountFinalTokenRelease(ComponentRef):
    """Declare final-only token-ID release after a fixed generation count.

    No intermediate token IDs, logits, streaming, or early-stop disclosure is
    selected. This configuration declaration does not execute or authorize release.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-fixed-count-final-token-release/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class AuthenticatedDirectedThreePartyTransport(ComponentRef):
    """Declare authenticated directed three-party transport configuration.

    This declaration is not a transport executor: it creates no channels,
    credentials, authentication, confidentiality, or provisioning authority.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/authenticated-directed-three-party-transport/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/protocols",
        category_version="1", lifecycle_phase="compile",
        parameter_schema={
            "type": "object", "properties": {}, "required": [], "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component, {})

    @classmethod
    def from_params(cls, params: Mapping[str, object]) -> Self:
        if not isinstance(params, Mapping):
            raise ConfigurationError("component params must be a mapping")
        return cast(Self, super().from_params(params))

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "ReplicatedPrivateTokenInput",
    "ReplicatedCompleteTableLookup",
    "ReplicatedPublicMatmulFullSum",
    "ReplicatedSecretMultiply",
    "ReplicatedRingAddBias",
    "ReplicatedArithmeticBooleanConversion",
    "ReplicatedCausalQkSoftmaxAv",
    "ReplicatedCompleteStableArgmax",
    "ReplicatedFixedCountFinalTokenRelease",
    "AuthenticatedDirectedThreePartyTransport",
]
