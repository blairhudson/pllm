"""Public component configuration and built-in discovery contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.correlation import SeededExpansion as _SeededExpansion
from pllm.kernels import Cpu as _Cpu
from pllm.metrics import (
    Accuracy as _Accuracy,
    Communication as _Communication,
    Cost as _Cost,
    Energy as _Energy,
    Latency as _Latency,
    Memory as _Memory,
    Perplexity as _Perplexity,
    Throughput as _Throughput,
)
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

if TYPE_CHECKING:
    from pllm.providers import ProviderDescriptor

_BUILTIN_CLASSES: tuple[type[ComponentRef], ...] = (
    _Accuracy,
    _BFVCorrelations,
    _BinaryTableGatedMultiplyQ7,
    _BlindedLinear,
    _ChunkedIndependentLanesProtectedTensorSchedule,
    _CleartextLinear,
    _ClientLocalKv,
    _Communication,
    _Cost,
    _Cpu,
    _DirectFHE,
    _Energy,
    _GuardedLinear,
    _HEAuthenticatedPreprocessing,
    _IndependentLanesProtectedTensorSchedule,
    _Inference,
    _KvCacheEviction,
    _Latency,
    _LinearIntegrity,
    _MaskedLinear,
    _Memory,
    _ModelAwareCorrections,
    _Perplexity,
    _R03CrtGatedMultiplyQ7,
    _ScalarProtectedTensorSchedule,
    _SeededExpansion,
    _SecureLinear,
    _Throughput,
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


def list_component_classes(
    *, providers: Iterable[ProviderDescriptor] = ()
) -> tuple[type[ComponentRef], ...]:
    classes = list(_BUILTIN_CLASSES)
    providers = tuple(providers)
    if providers:
        from pllm.providers import provider_component_classes

        classes.extend(provider_component_classes(providers))
    by_identity = {component.describe().component: component for component in classes}
    if len(by_identity) != len(classes):
        raise ConfigurationError("component identities must be unique")
    return tuple(by_identity[identity] for identity in sorted(by_identity))


def list_components(
    *, providers: Iterable[ProviderDescriptor] = ()
) -> tuple[ComponentDescriptor, ...]:
    """Return descriptors in stable component-identity order."""
    return tuple(
        component.describe() for component in list_component_classes(providers=providers)
    )


def get(
    identity: str, *, providers: Iterable[ProviderDescriptor] = ()
) -> type[ComponentRef]:
    if type(identity) is not str:
        raise TypeError("component identity must be a string")
    providers = tuple(providers)
    by_identity = {
        component.describe().component: component
        for component in list_component_classes(providers=providers)
    }
    try:
        return by_identity[identity]
    except KeyError:
        message = "component not found" if providers else "built-in component not found"
        raise KeyError(message) from None


def get_component(
    identity: str, *, providers: Iterable[ProviderDescriptor] = ()
) -> ComponentDescriptor:
    """Return one descriptor by canonical component identity."""
    return get(identity, providers=providers).describe()


def create_component(
    identity: str,
    params: Mapping[str, Any],
    *,
    providers: Iterable[ProviderDescriptor] = (),
) -> ComponentRef:
    if not isinstance(params, Mapping):
        raise TypeError("component params must be a mapping")
    try:
        component = get(identity, providers=providers)
    except KeyError:
        return ComponentRef(identity, params)
    schema = component.describe().parameter_schema
    if isinstance(schema, Mapping):
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
