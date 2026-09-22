"""Public persistent-state protocol declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef


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


__all__ = ["ClientLocalKv", "StateProtocol"]
