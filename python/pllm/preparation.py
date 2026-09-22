from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from typing import TYPE_CHECKING, Any

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError

if TYPE_CHECKING:
    from pllm.runtime.preparation_server import create_preparation_app as create_preparation_app


class PreparationProvider(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class ModelAwareCorrections(PreparationProvider):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/model-aware-corrections",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/preparation-provider",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("one-use-masked-linear-corrections",),
        role_eligibility=("preparation",),
    )

    def __init__(self) -> None:
        super().__init__(self.descriptor.component)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class BFVCorrelations(PreparationProvider):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/bfv-correlations/v1",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/preparation-provider",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={
            "type": "object",
            "properties": {
                "poly_modulus_degree": {"type": "integer", "enum": [4096, 8192, 16384]},
            },
            "required": ["poly_modulus_degree"],
            "additionalProperties": False,
        },
        capabilities=("bfv-correlation-generation",),
        required_host_features=("tenseal-backend",),
        role_eligibility=("client", "preparation"),
    )

    def __init__(self, *, poly_modulus_degree: int = 4096) -> None:
        if poly_modulus_degree not in {4096, 8192, 16384} or type(poly_modulus_degree) is not int:
            raise ConfigurationError("poly_modulus_degree must be 4096, 8192, or 16384")
        super().__init__(
            self.descriptor.component,
            {"poly_modulus_degree": poly_modulus_degree},
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"poly_modulus_degree": self.params["poly_modulus_degree"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


class HEAuthenticatedPreprocessing(PreparationProvider):
    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/he-authenticated-preprocessing",
        provider="pllm",
        distribution="pllm.run",
        version="1",
        category="pllm/preparation-provider",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={
            "type": "object",
            "properties": {
                "poly_modulus_degree": {"type": "integer", "enum": [4096, 8192, 16384]},
                "threads": {"type": "integer", "minimum": 1},
                "enable_linear_correlations": {"type": "boolean"},
            },
            "required": ["poly_modulus_degree", "threads", "enable_linear_correlations"],
            "additionalProperties": False,
        },
        capabilities=("he-authenticated-preprocessing",),
        required_host_features=("tenseal-backend",),
        role_eligibility=("client", "preparation"),
    )

    def __init__(
        self,
        *,
        poly_modulus_degree: int = 4096,
        threads: int = 1,
        enable_linear_correlations: bool = True,
    ) -> None:
        if poly_modulus_degree not in {4096, 8192, 16384} or type(poly_modulus_degree) is not int:
            raise ConfigurationError("poly_modulus_degree must be 4096, 8192, or 16384")
        if type(threads) is not int or threads < 1:
            raise ConfigurationError("threads must be an integer >= 1")
        if type(enable_linear_correlations) is not bool:
            raise ConfigurationError("enable_linear_correlations must be a bool")
        super().__init__(
            self.descriptor.component,
            {
                "poly_modulus_degree": poly_modulus_degree,
                "threads": threads,
                "enable_linear_correlations": enable_linear_correlations,
            },
        )

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return dict(self.params)

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


__all__ = [
    "BFVCorrelations",
    "HEAuthenticatedPreprocessing",
    "ModelAwareCorrections",
    "PreparationProvider",
    "create_preparation_app",
]


def __getattr__(name: str) -> Any:
    if name != "create_preparation_app":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = import_module("pllm.runtime.preparation_server").create_preparation_app
    globals()[name] = value
    return value
