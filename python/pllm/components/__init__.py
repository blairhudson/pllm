"""Public component configuration and built-in discovery contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pllm.components._planned import (
    NotYetImplementedError,
    PlannedComponent,
    planned_component,
    planned_components,
    require_implemented_identity,
)
from pllm.components._model_capabilities import (
    ModelCapabilityStub,
    ModelCapabilityUnavailable,
    model_capabilities,
    model_capability,
)
from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError

if TYPE_CHECKING:
    from pllm.providers import ProviderDescriptor


@lru_cache(maxsize=1)
def _builtin_classes() -> tuple[type[ComponentRef], ...]:
    """Delay family imports so each family can expose planned component symbols."""
    from pllm.correlation import SeededExpansion
    from pllm.kernels import AppleMetal, Cpu
    from pllm.metrics import (
        Accuracy,
        Communication,
        Cost,
        Energy,
        Latency,
        Memory,
        Perplexity,
        ReferenceAgreement,
        Throughput,
    )
    from pllm.nonlinear import (
        ArithmeticGarblingSiluQ7,
        BinaryTableGatedMultiplyQ7,
        R03CrtGatedMultiplyQ7,
    )
    from pllm.passes import KvCacheEviction
    from pllm.preparation import (
        BFVCorrelations,
        HEAuthenticatedPreprocessing,
        ModelAwareCorrections,
        PreparedInventory,
    )
    from pllm.quantization import PublicPerChannelEqualized, SymmetricPerRow
    from pllm.protocols import (
        BlindedLinear,
        ClientBundleTransport,
        CleartextLinear,
        DirectFHE,
        GuardedLinear,
        MaskedLinear,
        SecureLinear,
        TwoOnlineOffsetLinear,
    )
    from pllm.roles import (
        ClientOnlyRoles,
        ClientLinearRoles,
        ClientPrefixLayers,
        Inference,
        OutputHeadAtInference,
        PreparedProviderRoles,
        TwoOnlineOffsetRoles,
    )
    from pllm.schedulers import (
        BoundedIndependentElementsProtectedTensorSchedule,
        ChunkedIndependentLanesProtectedTensorSchedule,
        IndependentLanesProtectedTensorSchedule,
        ScalarProtectedTensorSchedule,
    )
    from pllm.state import ClientLocalKv, ClientPrefixReuse, PublicPrefixCapsule
    from pllm.tokenization import IndexedTokenizer
    from pllm.verification import FreivaldsVerify, LinearIntegrity

    classes = (
        Accuracy,
        AppleMetal,
        ArithmeticGarblingSiluQ7,
        BFVCorrelations,
        BinaryTableGatedMultiplyQ7,
        BlindedLinear,
        BoundedIndependentElementsProtectedTensorSchedule,
        ChunkedIndependentLanesProtectedTensorSchedule,
        CleartextLinear,
        ClientLocalKv,
        ClientOnlyRoles,
        ClientPrefixReuse,
        PublicPrefixCapsule,
        IndexedTokenizer,
        Communication,
        Cost,
        Cpu,
        DirectFHE,
        Energy,
        FreivaldsVerify,
        GuardedLinear,
        HEAuthenticatedPreprocessing,
        IndependentLanesProtectedTensorSchedule,
        Inference,
        KvCacheEviction,
        Latency,
        LinearIntegrity,
        MaskedLinear,
        Memory,
        ModelAwareCorrections,
        OutputHeadAtInference,
        ClientPrefixLayers,
        ClientLinearRoles,
        PreparedInventory,
        ClientBundleTransport,
        Perplexity,
        PreparedProviderRoles,
        PublicPerChannelEqualized,
        R03CrtGatedMultiplyQ7,
        ReferenceAgreement,
        ScalarProtectedTensorSchedule,
        SeededExpansion,
        SecureLinear,
        SymmetricPerRow,
        Throughput,
        TwoOnlineOffsetLinear,
        TwoOnlineOffsetRoles,
    )
    if len({component.describe().component for component in classes}) != len(classes):
        raise RuntimeError("built-in component identities must be unique")
    return classes


__all__ = [
    "ComponentDescriptor",
    "ComponentRef",
    "NotYetImplementedError",
    "ModelCapabilityStub",
    "ModelCapabilityUnavailable",
    "PlannedComponent",
    "create_component",
    "get",
    "get_component",
    "list_component_classes",
    "list_components",
    "planned_component",
    "planned_components",
    "model_capability",
    "model_capabilities",
]


def list_component_classes(
    *, providers: Iterable[ProviderDescriptor] = ()
) -> tuple[type[ComponentRef], ...]:
    classes = list(_builtin_classes())
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
    return tuple(component.describe() for component in list_component_classes(providers=providers))


def get(identity: str, *, providers: Iterable[ProviderDescriptor] = ()) -> type[ComponentRef]:
    if type(identity) is not str:
        raise TypeError("component identity must be a string")
    require_implemented_identity(identity)
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
