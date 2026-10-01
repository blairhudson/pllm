"""Public kernel backend declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending
from pllm.components._model_capabilities import PendingModelCapability, model_capability_stub


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


class AppleMetal(KernelBackend):
    """Public integer stages on MLX/Metal above min_rows; CPU below it."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/apple-metal-int8/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/kernel-backend",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {"min_rows": {"type": "integer", "minimum": 2, "maximum": 256}},
            "required": ["min_rows"],
            "additionalProperties": False,
        },
        capabilities=("apple-metal-int8", "cpu-decode"),
        role_eligibility=("client", "preparation", "inference", "worker"),
    )

    def __init__(self, *, min_rows: int = 8) -> None:
        if type(min_rows) is not int or not 2 <= min_rows <= 256:
            raise ConfigurationError("apple-metal.min_rows must be an integer from 2 to 256")
        super().__init__(self.descriptor.component, {"min_rows": min_rows})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"min_rows": self.params["min_rows"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


@pending("piranha")
class SecureGpuBackend(PendingMethod):
    pass


@model_capability_stub("partial-mrope")
class PartialMultimodalRotary(PendingModelCapability):
    pass


@model_capability_stub("yarn-rotary-scaling")
class YarnRotaryScaling(PendingModelCapability):
    pass


__all__ = ["AppleMetal", "Cpu", "KernelBackend", "SecureGpuBackend", "PartialMultimodalRotary", "YarnRotaryScaling"]

install_planned_components(globals())
