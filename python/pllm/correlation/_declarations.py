"""Model- and method-neutral correlation configuration declarations, not executors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class RaaMask(ComponentRef):
    """Declare RAA-mask configuration without generating or consuming masks.

    This model- and method-neutral declaration requires compiler/runtime binding.
    """

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/raa-mask/v1", provider="pllm", distribution="pllm.run",
        version="1", category="pllm/correlation-source", category_version="1",
        lifecycle_phase="compile",
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
        if params:
            raise ConfigurationError("RaaMask accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["RaaMask"]


class ReplicatedPairwisePrss(ComponentRef):
    """Declare replicated pairwise PRSS requirements without generating seeds."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/replicated-pairwise-prss/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/correlation-source", category_version="1", lifecycle_phase="compile",
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
        if params:
            raise ConfigurationError("ReplicatedPairwisePrss accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ += ["ReplicatedPairwisePrss"]


class DistributedFerretOt(ComponentRef):
    """Declare distributed Ferret OT requirements without provisioning correlations."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/distributed-ferret-ot/v1",
        provider="pllm", distribution="pllm.run", version="1",
        category="pllm/correlation-source", category_version="1", lifecycle_phase="compile",
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
        if params:
            raise ConfigurationError("DistributedFerretOt accepts no parameters")
        return cls()

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return self.to_spec()["params"]

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ += ["DistributedFerretOt"]
