from collections.abc import Iterable, Mapping
from typing import Any

from pllm.configuration import ComponentDescriptor as ComponentDescriptor
from pllm.configuration import ComponentRef as ComponentRef
from pllm.providers import ProviderDescriptor

def list_component_classes(
    *, providers: Iterable[ProviderDescriptor] = ...,
) -> tuple[type[ComponentRef], ...]: ...
def list_components(
    *, providers: Iterable[ProviderDescriptor] = ...,
) -> tuple[ComponentDescriptor, ...]: ...
def get(
    identity: str, *, providers: Iterable[ProviderDescriptor] = ...,
) -> type[ComponentRef]: ...
def get_component(
    identity: str, *, providers: Iterable[ProviderDescriptor] = ...,
) -> ComponentDescriptor: ...
def create_component(
    identity: str,
    params: Mapping[str, Any],
    *,
    providers: Iterable[ProviderDescriptor] = ...,
) -> ComponentRef: ...
