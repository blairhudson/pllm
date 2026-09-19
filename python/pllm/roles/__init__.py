"""Public runtime role component declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef


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
        distribution="pllm",
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


__all__ = ["Inference", "InferenceRole"]
