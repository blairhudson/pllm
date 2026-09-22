"""Public nonlinear protocol component declarations."""

from abc import ABC, abstractmethod

from pllm.configuration import ComponentDescriptor, ComponentRef


class NonlinearProtocol(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class ArithmeticGarblingSiluQ7(NonlinearProtocol):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/arithmetic-garbling-silu-q7/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="runtime",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-silu-q7",),
        required_host_features=("native-core",),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class BinaryTableGatedMultiplyQ7(NonlinearProtocol):
    """Binary-table implementation of protected Q7 ``SiLU(gate) * up``."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/binary-table/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-gated-multiply-q7",),
        required_host_features=("decoder-plan-v1", "signed-q7"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class R03CrtGatedMultiplyQ7(NonlinearProtocol):
    """R03 CRT implementation of protected Q7 ``SiLU(gate) * up``."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/r03-crt/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/nonlinear-protocol",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("protected-gated-multiply-q7",),
        required_host_features=("decoder-plan-v1", "signed-q7"),
        role_eligibility=("client", "inference"),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "ArithmeticGarblingSiluQ7",
    "BinaryTableGatedMultiplyQ7",
    "NonlinearProtocol",
    "R03CrtGatedMultiplyQ7",
]
