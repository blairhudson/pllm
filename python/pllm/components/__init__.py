"""Public component configuration and built-in discovery contracts."""

from typing import Mapping, cast

import pllm.configuration as _configuration
from pllm.configuration import ComponentDescriptor, ComponentRef, KvCacheEviction

_BUILTIN_DESCRIPTORS = cast(
    Mapping[str, ComponentDescriptor], getattr(_configuration, "_BUILTIN_DESCRIPTORS")
)

__all__ = [
    "ComponentDescriptor",
    "ComponentRef",
    "KvCacheEviction",
    "get_component",
    "list_components",
]


def list_components() -> tuple[ComponentDescriptor, ...]:
    """Return built-in descriptors in stable component-identity order."""
    return tuple(_BUILTIN_DESCRIPTORS[identity] for identity in sorted(_BUILTIN_DESCRIPTORS))


def get_component(identity: str) -> ComponentDescriptor:
    """Return one built-in descriptor by canonical component identity."""
    if type(identity) is not str:
        raise TypeError("component identity must be a string")
    try:
        return _BUILTIN_DESCRIPTORS[identity]
    except KeyError:
        raise KeyError("built-in component not found") from None
