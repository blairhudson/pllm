from collections.abc import Collection, Iterable, Mapping
from typing import Any

from pllm.configuration import ComponentDescriptor, ComponentRef

PROVIDER_ENTRY_POINT_GROUP: str
PROVIDER_MANIFEST_SCHEMA: str
HOST_COMPONENT_STANDARD_VERSION: str

class ProviderDiscoveryError(RuntimeError): ...

class ProviderResource:
    path: str
    sha256: str
    abi_version: str | None
    targets: tuple[str, ...]
    def __init__(
        self,
        path: str,
        sha256: str,
        abi_version: str | None = None,
        targets: tuple[str, ...] = (),
    ) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...

class ProviderDescriptor:
    provider: str
    distribution: str
    version: str
    host_versions: tuple[str, ...]
    category_versions: Mapping[str, tuple[str, ...]]
    components: tuple[ComponentDescriptor, ...]
    schemas: tuple[ProviderResource, ...]
    docs: tuple[ProviderResource, ...]
    native_artifacts: tuple[ProviderResource, ...]
    python_factories: Mapping[str, str]
    tested_builds: tuple[Mapping[str, Any], ...]
    entry_point: str
    manifest_digest: str
    editable: bool
    component_classes: tuple[type[ComponentRef], ...]
    def to_dict(self) -> dict[str, Any]: ...

def discover_providers(
    *,
    entry_points: Iterable[Any] | None = None,
    allow_editable: bool = False,
    host_version: str = HOST_COMPONENT_STANDARD_VERSION,
) -> tuple[ProviderDescriptor, ...]: ...

def provider_component_classes(
    providers: Iterable[ProviderDescriptor],
) -> tuple[type[ComponentRef], ...]: ...

def load_component_factory(
    provider: ProviderDescriptor,
    component: str,
    *,
    approved_providers: Collection[str],
) -> Any: ...
