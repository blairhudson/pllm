"""Public protected tensor scheduler declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


class ProtectedScheduler(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class BoundedIndependentElementsProtectedTensorSchedule(ProtectedScheduler):
    """One-use scheduling for bounded unary protected tensors."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/bounded-independent-elements/v1",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protected-scheduler",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {
                "max_elements": {"type": "integer", "minimum": 1, "maximum": 128},
            },
            "required": ["max_elements"],
            "additionalProperties": False,
        },
        capabilities=("one-use-protected-tensor-scheduling",),
        required_host_features=("authenticated-one-use-material",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, *, max_elements: int = 128) -> None:
        if type(max_elements) is not int or not 1 <= max_elements <= 128:
            raise ConfigurationError("max_elements must be an integer from 1 through 128")
        super().__init__(self.descriptor.component, {"max_elements": max_elements})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"max_elements": self.params["max_elements"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ScalarProtectedTensorSchedule(ProtectedScheduler):
    """Scalar one-use scheduling for protected tensor elements."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/scalar/v1",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protected-scheduler",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("one-use-protected-tensor-scheduling",),
        required_host_features=("authenticated-one-use-material",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class IndependentLanesProtectedTensorSchedule(ProtectedScheduler):
    """Independent-lane one-use scheduling for protected tensor elements."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/independent-lanes/v1",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protected-scheduler",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {
                "max_elements": {"type": "integer", "minimum": 2, "maximum": 4},
            },
            "required": ["max_elements"],
            "additionalProperties": False,
        },
        capabilities=("one-use-protected-tensor-scheduling",),
        required_host_features=("authenticated-one-use-material",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, *, max_elements: int = 4) -> None:
        if type(max_elements) is not int or max_elements < 2:
            raise ConfigurationError("max_elements must be an integer >= 2")
        if max_elements > 4:
            raise ConfigurationError("max_elements must be an integer <= 4")
        super().__init__(self.descriptor.component, {"max_elements": max_elements})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"max_elements": self.params["max_elements"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class ChunkedIndependentLanesProtectedTensorSchedule(ProtectedScheduler):
    """Authenticated bounded streaming schedule for larger protected tensors."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/chunked-independent-lanes/v1",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protected-scheduler",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {
                "max_elements": {"type": "integer", "minimum": 5, "maximum": 4_000_000},
            },
            "required": ["max_elements"],
            "additionalProperties": False,
        },
        capabilities=("one-use-protected-tensor-scheduling",),
        required_host_features=(
            "authenticated-one-use-material",
            "bounded-streaming-spool",
        ),
        role_eligibility=("client", "inference"),
    )

    def __init__(self, *, max_elements: int) -> None:
        if type(max_elements) is not int or max_elements < 5:
            raise ConfigurationError("max_elements must be an integer >= 5")
        if max_elements > 4_000_000:
            raise ConfigurationError("max_elements must be an integer <= 4000000")
        super().__init__(self.descriptor.component, {"max_elements": max_elements})

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"max_elements": self.params["max_elements"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "BoundedIndependentElementsProtectedTensorSchedule",
    "ChunkedIndependentLanesProtectedTensorSchedule",
    "IndependentLanesProtectedTensorSchedule",
    "ProtectedScheduler",
    "ScalarProtectedTensorSchedule",
]
