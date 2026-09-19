"""Public component configuration and built-in discovery contracts."""

from collections.abc import Mapping
from typing import Any

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.correlation import SeededExpansion as _SeededExpansion
from pllm.kernels import Cpu as _Cpu
from pllm.nonlinear import (
    BinaryTableGatedMultiplyQ7 as _BinaryTableGatedMultiplyQ7,
    R03CrtGatedMultiplyQ7 as _R03CrtGatedMultiplyQ7,
)
from pllm.passes import KvCacheEviction as _KvCacheEviction
from pllm.preparation import (
    BFVCorrelations as _BFVCorrelations,
    HEAuthenticatedPreprocessing as _HEAuthenticatedPreprocessing,
    ModelAwareCorrections as _ModelAwareCorrections,
)
from pllm.protocols import (
    BlindedLinear as _BlindedLinear,
    CleartextLinear as _CleartextLinear,
    DirectFHE as _DirectFHE,
    GuardedLinear as _GuardedLinear,
    MaskedLinear as _MaskedLinear,
    SecureLinear as _SecureLinear,
)
from pllm.roles import Inference as _Inference
from pllm.schedulers import (
    ChunkedIndependentLanesProtectedTensorSchedule as _ChunkedIndependentLanesProtectedTensorSchedule,
    IndependentLanesProtectedTensorSchedule as _IndependentLanesProtectedTensorSchedule,
    ScalarProtectedTensorSchedule as _ScalarProtectedTensorSchedule,
)
from pllm.state import ClientLocalKv as _ClientLocalKv
from pllm.verification import LinearIntegrity as _LinearIntegrity

_BUILTIN_CLASSES: tuple[type[ComponentRef], ...] = (
    _BFVCorrelations,
    _BinaryTableGatedMultiplyQ7,
    _BlindedLinear,
    _ChunkedIndependentLanesProtectedTensorSchedule,
    _CleartextLinear,
    _ClientLocalKv,
    _Cpu,
    _DirectFHE,
    _GuardedLinear,
    _HEAuthenticatedPreprocessing,
    _IndependentLanesProtectedTensorSchedule,
    _Inference,
    _KvCacheEviction,
    _LinearIntegrity,
    _MaskedLinear,
    _ModelAwareCorrections,
    _R03CrtGatedMultiplyQ7,
    _ScalarProtectedTensorSchedule,
    _SeededExpansion,
    _SecureLinear,
)
_BUILTINS = {component.describe().component: component for component in _BUILTIN_CLASSES}
if len(_BUILTINS) != len(_BUILTIN_CLASSES):
    raise RuntimeError("built-in component identities must be unique")

__all__ = [
    "ComponentDescriptor",
    "ComponentRef",
    "create_component",
    "get",
    "get_component",
    "list_component_classes",
    "list_components",
]


def list_component_classes() -> tuple[type[ComponentRef], ...]:
    return tuple(_BUILTINS[identity] for identity in sorted(_BUILTINS))


def list_components() -> tuple[ComponentDescriptor, ...]:
    """Return built-in descriptors in stable component-identity order."""
    return tuple(component.describe() for component in list_component_classes())


def get(identity: str) -> type[ComponentRef]:
    if type(identity) is not str:
        raise TypeError("component identity must be a string")
    try:
        return _BUILTINS[identity]
    except KeyError:
        raise KeyError("built-in component not found") from None


def get_component(identity: str) -> ComponentDescriptor:
    """Return one built-in descriptor by canonical component identity."""
    return get(identity).describe()


def create_component(identity: str, params: Mapping[str, Any]) -> ComponentRef:
    if not isinstance(params, Mapping):
        raise TypeError("component params must be a mapping")
    try:
        component = get(identity)
    except KeyError:
        return ComponentRef(identity, params)
    schema = component.describe().parameter_schema
    properties = set(schema.get("properties", {}))
    required = set(schema.get("required", ()))
    allowed = properties | required
    unknown = set(params) - allowed
    missing = required - set(params)
    if unknown:
        raise ConfigurationError(f"component params have unknown fields: {sorted(unknown)}")
    if missing:
        raise ConfigurationError(f"component params have missing fields: {sorted(missing)}")
    return component.from_params(params)
