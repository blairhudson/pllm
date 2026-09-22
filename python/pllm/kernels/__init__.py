"""Public kernel backend declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class KernelBackend(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class Cpu(KernelBackend):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/cpu",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/kernel-backend",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {"threads": {"type": "integer", "minimum": 1}},
            "required": ["threads"],
            "additionalProperties": False,
        },
        capabilities=("cpu",),
        role_eligibility=("client", "preparation", "inference"),
    )

    def __init__(self, *, threads: int = 1) -> None:
        if type(threads) is not int or threads < 1:
            raise ConfigurationError("cpu.threads must be an integer >= 1")
        super().__init__(self.descriptor.component, {"threads": threads})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"threads": self.params["threads"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = ["Cpu", "KernelBackend"]
