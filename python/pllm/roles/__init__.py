"""Public runtime role component declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.roles.topology import (
    Channel, Role, RoleGraph, client_only_reference_graph, two_online_reference_graph,
)


class InferenceRole(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class Inference(InferenceRole):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/inference",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/inference-role",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("remote-linear-stage-execution",),
        role_eligibility=("inference",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class RoleTopology(ComponentRef, ABC):
    """One compatible placement of logical roles, not a privacy proof."""

    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class PreparedProviderRoles(RoleTopology):
    """Client, offline trusted Preparation, and one online Inference provider."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/one-online-provider-offline-preparation/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/role-topology",
        category_version="1",
        lifecycle_phase="offline+online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("prepared-public-linear", "single-online-provider"),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "Channel", "Inference", "InferenceRole", "PreparedProviderRoles", "Role", "RoleGraph",
    "RoleTopology", "client_only_reference_graph", "two_online_reference_graph",
]
