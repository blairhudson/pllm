"""Public verification component declarations."""

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any

from pllm.configuration import ComponentDescriptor, ComponentRef

if TYPE_CHECKING:
    from pllm.runtime.linear_integrity import (
        LinearCheckKey as LinearCheckKey,
        LinearCheckResult as LinearCheckResult,
        LinearIntegrityError as LinearIntegrityError,
        check_linear_result as check_linear_result,
        create_linear_check_key as create_linear_check_key,
    )


class VerificationScheme(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class LinearIntegrity(VerificationScheme):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/linear-integrity",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/verification-scheme",
        category_version="1",
        lifecycle_phase="online",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("hidden-linear-integrity-check",),
        role_eligibility=("client",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class FreivaldsVerify(VerificationScheme):
    """Authenticated one-use verification for prepared public linear stages."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/freivalds-verify/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/verification-scheme",
        category_version="1",
        lifecycle_phase="offline-online",
        parameter_schema={
            "type": "object",
            "properties": {"target_failure_bits": {"type": "integer", "minimum": 1, "maximum": 80}},
            "required": ["target_failure_bits"],
            "additionalProperties": False,
        },
        capabilities=("authenticated-one-use-freivalds",),
        required_host_features=("prepared-public-linear-v1",),
        role_eligibility=("client", "preparation"),
    )

    def __init__(self, target_failure_bits: int = 40) -> None:
        if isinstance(target_failure_bits, bool) or not isinstance(target_failure_bits, int):
            raise TypeError("target_failure_bits must be an integer")
        if not 1 <= target_failure_bits <= 80:
            raise ValueError("target_failure_bits must be in [1, 80]")
        super().__init__(
            self.descriptor.component,
            {"target_failure_bits": target_failure_bits},
        )

    @property
    def target_failure_bits(self) -> int:
        return self.params["target_failure_bits"]

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"target_failure_bits": self.target_failure_bits}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


_RUNTIME_EXPORTS = {
    "LinearCheckKey",
    "LinearCheckResult",
    "FreivaldsVerify",
    "LinearIntegrityError",
    "check_linear_result",
    "create_linear_check_key",
}

__all__ = [
    "FreivaldsVerify",
    "LinearCheckKey",
    "LinearCheckResult",
    "LinearIntegrity",
    "LinearIntegrityError",
    "VerificationScheme",
    "check_linear_result",
    "create_linear_check_key",
]


def __getattr__(name: str) -> Any:
    if name not in _RUNTIME_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("pllm.runtime.linear_integrity"), name)
    globals()[name] = value
    return value
