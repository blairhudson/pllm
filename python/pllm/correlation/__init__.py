"""Public preprocessing correlation-source declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef


class CorrelationSource(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class SeededExpansion(CorrelationSource):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/seeded-expansion",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/correlation-source",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("domain-separated-mask-expansion",),
        role_eligibility=("client", "preparation"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["CorrelationSource", "SeededExpansion"]
