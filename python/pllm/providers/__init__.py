"""Inert provider metadata discovery and explicit loading boundaries."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from pllm.configuration import (
    ComponentDescriptor,
    ComponentRef,
    ConfigurationError,
)

PROVIDER_ENTRY_POINT_GROUP = "pllm.providers.v1"
PROVIDER_MANIFEST_SCHEMA = "pllm.provider_manifest.v1"
HOST_COMPONENT_STANDARD_VERSION = "1"
HOST_NATIVE_PLUGIN_ABI_VERSION = "1"
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_RESOURCE_BYTES = 16 << 20
_DIGEST_DOMAIN = b"pllm.provider_manifest.v1\0"


class ProviderDiscoveryError(RuntimeError):
    pass


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _schema() -> Mapping[str, Any]:
    path = files(__package__).joinpath("provider-manifest.schema.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _duplicates_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderDiscoveryError(f"provider manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_limited(path: Path, limit: int, label: str) -> bytes:
    try:
        before = path.stat()
        if not path.is_file() or before.st_size > limit:
            raise ProviderDiscoveryError(f"{label} is missing, non-regular, or oversized")
        payload = path.read_bytes()
        after = path.stat()
    except ProviderDiscoveryError:
        raise
    except OSError as exc:
        raise ProviderDiscoveryError(f"cannot read {label}") from exc
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if len(payload) != before.st_size or identity(before) != identity(after):
        raise ProviderDiscoveryError(f"{label} changed while it was read")
    return payload


def _hash_file(path: Path, expected: str, label: str) -> None:
    try:
        before = path.stat()
        if not path.is_file() or before.st_size > _MAX_RESOURCE_BYTES:
            raise ProviderDiscoveryError(f"{label} is missing, non-regular, or oversized")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = path.stat()
    except ProviderDiscoveryError:
        raise
    except OSError as exc:
        raise ProviderDiscoveryError(f"cannot read {label}") from exc
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise ProviderDiscoveryError(f"{label} changed while it was hashed")
    if digest.hexdigest() != expected:
        raise ProviderDiscoveryError(f"{label} digest does not match its manifest")


def _resource_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ProviderDiscoveryError(f"provider resource path is not normalized: {value!r}")
    return path


@dataclass(frozen=True, slots=True)
class ProviderResource:
    path: str
    sha256: str
    abi_version: str | None = None
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.abi_version is not None:
            result["abi_version"] = self.abi_version
            result["targets"] = list(self.targets)
        return result


@dataclass(frozen=True, slots=True)
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
    component_classes: tuple[type[ComponentRef], ...] = field(compare=False, repr=False)
    _verified: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        for name in (
            "host_versions",
            "components",
            "schemas",
            "docs",
            "native_artifacts",
            "component_classes",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "category_versions",
            MappingProxyType({key: tuple(values) for key, values in sorted(self.category_versions.items())}),
        )
        object.__setattr__(
            self,
            "python_factories",
            MappingProxyType(dict(sorted(self.python_factories.items()))),
        )
        object.__setattr__(self, "tested_builds", tuple(_freeze(item) for item in self.tested_builds))

    def __hash__(self) -> int:
        return hash((self.provider, self.distribution, self.version, self.manifest_digest))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVIDER_MANIFEST_SCHEMA,
            "provider": self.provider,
            "distribution": self.distribution,
            "version": self.version,
            "host_versions": list(self.host_versions),
            "category_versions": {key: list(values) for key, values in self.category_versions.items()},
            "components": [component.to_dict() for component in self.components],
            "schemas": [resource.to_dict() for resource in self.schemas],
            "docs": [resource.to_dict() for resource in self.docs],
            "native_artifacts": [resource.to_dict() for resource in self.native_artifacts],
            "python_factories": dict(self.python_factories),
            "tested_builds": [_thaw(item) for item in self.tested_builds],
            "entry_point": self.entry_point,
            "manifest_digest": self.manifest_digest,
            "editable": self.editable,
        }


def _component_descriptor(value: Mapping[str, Any]) -> ComponentDescriptor:
    return ComponentDescriptor(
        component=value["component"],
        provider=value["provider"],
        distribution=value["distribution"],
        version=value["version"],
        category=value["category"],
        category_version=value["category_version"],
        lifecycle_phase=value["lifecycle_phase"],
        parameter_schema=value["parameter_schema"],
        capabilities=tuple(value["capabilities"]),
        required_host_features=tuple(value["required_host_features"]),
        role_eligibility=tuple(value["role_eligibility"]),
        artifacts=tuple(value["artifacts"]),
        evidence=tuple(value["evidence"]),
    )


def _category_base(category: str) -> type[ComponentRef]:
    from pllm.correlation import CorrelationSource
    from pllm.kernels import KernelBackend
    from pllm.nonlinear import NonlinearProtocol
    from pllm.passes import PlanPass
    from pllm.preparation import PreparationProvider
    from pllm.protocols import ProtocolMethod
    from pllm.roles import InferenceRole
    from pllm.schedulers import ProtectedScheduler
    from pllm.state import StateProtocol
    from pllm.verification import VerificationScheme

    return {
        "pllm/compiler-pass": PlanPass,
        "pllm/correlation-source": CorrelationSource,
        "pllm/inference-role": InferenceRole,
        "pllm/kernel-backend": KernelBackend,
        "pllm/nonlinear-protocol": NonlinearProtocol,
        "pllm/preparation-provider": PreparationProvider,
        "pllm/protected-scheduler": ProtectedScheduler,
        "pllm/protocol-method": ProtocolMethod,
        "pllm/state-protocol": StateProtocol,
        "pllm/verification-scheme": VerificationScheme,
    }.get(category, ComponentRef)


def _external_component_class(descriptor: ComponentDescriptor) -> type[ComponentRef]:
    digest = hashlib.sha256(
        json.dumps(descriptor.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    class_name = "".join(part.title() for part in re.split(r"[^A-Za-z0-9]+", descriptor.component) if part)
    class_name = f"External{class_name or 'Component'}_{digest}"

    def __init__(self: ComponentRef, **params: Any) -> None:
        errors = sorted(
            Draft202012Validator(descriptor.parameter_schema).iter_errors(params),
            key=lambda error: tuple(str(part) for part in error.absolute_path),
        )
        if errors:
            path = ".".join(str(part) for part in errors[0].absolute_path) or "params"
            raise ConfigurationError(f"invalid {descriptor.component} parameter at {path}: {errors[0].message}")
        ComponentRef.__init__(self, descriptor.component, params)

    def get_params(self: ComponentRef, deep: bool = True) -> dict[str, Any]:
        return _thaw(self.params)

    @classmethod
    def describe(cls: type[ComponentRef]) -> ComponentDescriptor:
        return descriptor

    namespace = {
        "__module__": __name__,
        "__slots__": (),
        "descriptor": descriptor,
        "__init__": __init__,
        "get_params": get_params,
        "describe": describe,
    }
    return type(class_name, (_category_base(descriptor.category),), namespace)


def _editable(distribution: Any) -> bool:
    try:
        text = distribution.read_text("direct_url.json")
    except (AttributeError, OSError):
        return False
    if not text:
        return False
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise ProviderDiscoveryError("provider distribution has invalid direct_url.json") from exc
    if not isinstance(value, Mapping) or not isinstance(value.get("dir_info", {}), Mapping):
        raise ProviderDiscoveryError("provider distribution has invalid direct_url.json")
    return value.get("dir_info", {}).get("editable") is True


def _listed_files(distribution: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in distribution.files or ():
        rendered = str(item).replace(os.sep, "/")
        if rendered in result:
            raise ProviderDiscoveryError(f"provider distribution repeats file {rendered!r}")
        result[rendered] = item
    return result


def _located(distribution: Any, listed: Mapping[str, Any], relative: str, root: Path) -> Path:
    if relative not in listed:
        raise ProviderDiscoveryError(f"provider distribution does not list {relative!r}")
    try:
        path = Path(distribution.locate_file(listed[relative])).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProviderDiscoveryError(f"provider distribution cannot locate {relative!r}") from exc
    if not path.is_relative_to(root):
        raise ProviderDiscoveryError(f"provider distribution file escapes its root: {relative!r}")
    return path


def _resource(
    value: Mapping[str, Any],
    *,
    package_name: str,
    package_root: Path,
    distribution_root: Path,
    listed: Mapping[str, Any],
    distribution: Any,
) -> ProviderResource:
    relative = _resource_path(value["path"])
    distribution_path = f"{package_name}/{relative.as_posix()}"
    path = _located(distribution, listed, distribution_path, distribution_root)
    if not path.is_relative_to(package_root):
        raise ProviderDiscoveryError(f"provider resource escapes its package: {relative.as_posix()!r}")
    _hash_file(path, value["sha256"], f"provider resource {relative.as_posix()!r}")
    return ProviderResource(
        path=relative.as_posix(),
        sha256=value["sha256"],
        abi_version=value.get("abi_version"),
        targets=tuple(sorted(value.get("targets", ()))),
    )


def _discover_one(entry_point: Any, *, allow_editable: bool, host_version: str) -> ProviderDescriptor:
    entry_name = getattr(entry_point, "name", None)
    entry_value = getattr(entry_point, "value", None)
    if type(entry_name) is not str or not entry_name or type(entry_value) is not str or not entry_value:
        raise ProviderDiscoveryError("provider entry point metadata is incomplete")
    distribution = getattr(entry_point, "dist", None)
    if distribution is None:
        raise ProviderDiscoveryError(f"provider entry point {entry_name!r} has no distribution")
    module = entry_value.split(":", 1)[0]
    package_name = module.split(".", 1)[0]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", package_name):
        raise ProviderDiscoveryError(f"provider entry point {entry_name!r} has an invalid package")
    try:
        distribution_root = Path(distribution.locate_file("")).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProviderDiscoveryError(f"provider entry point {entry_name!r} has no distribution root") from exc
    listed = _listed_files(distribution)
    manifest_relative = f"{package_name}/pllm-plugin.json"
    manifest_path = _located(distribution, listed, manifest_relative, distribution_root)
    try:
        package_root = (distribution_root / package_name).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProviderDiscoveryError(f"provider package {package_name!r} cannot be resolved") from exc
    if manifest_path.parent != package_root:
        raise ProviderDiscoveryError("provider manifest is outside its declared package")
    payload = _read_limited(manifest_path, _MAX_MANIFEST_BYTES, "provider manifest")
    try:
        value = json.loads(payload, object_pairs_hook=_duplicates_rejected)
    except ProviderDiscoveryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderDiscoveryError("provider manifest is not valid UTF-8 JSON") from exc
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        path = ".".join(str(part) for part in errors[0].absolute_path) or "manifest"
        raise ProviderDiscoveryError(f"provider manifest is invalid at {path}: {errors[0].message}")
    editable = _editable(distribution)
    if editable and not allow_editable:
        raise ProviderDiscoveryError(f"editable provider {value['provider']!r} requires explicit opt-in")
    try:
        metadata_name = str(distribution.metadata["Name"])
        distribution_version = str(distribution.version)
    except (AttributeError, KeyError, TypeError) as exc:
        raise ProviderDiscoveryError("provider distribution metadata is incomplete") from exc
    if _canonical_distribution(value["distribution"]) != _canonical_distribution(metadata_name):
        raise ProviderDiscoveryError(f"provider {value['provider']!r} distribution identity does not match")
    if value["version"] != distribution_version:
        raise ProviderDiscoveryError(f"provider {value['provider']!r} version does not match")
    if value["provider"] != entry_name:
        raise ProviderDiscoveryError("provider manifest identity does not match its entry point")
    if host_version not in value["host_versions"]:
        raise ProviderDiscoveryError(f"provider {value['provider']!r} does not support host version {host_version}")
    components = tuple(sorted((_component_descriptor(item) for item in value["components"]), key=lambda item: item.component))
    component_ids = [component.component for component in components]
    if len(component_ids) != len(set(component_ids)):
        raise ProviderDiscoveryError(f"provider {value['provider']!r} repeats a component identity")
    for component in components:
        try:
            Draft202012Validator.check_schema(component.to_dict()["parameter_schema"])
        except SchemaError as exc:
            raise ProviderDiscoveryError(
                f"component {component.component!r} parameter schema is invalid"
            ) from exc
        if (
            component.provider != value["provider"]
            or _canonical_distribution(component.distribution)
            != _canonical_distribution(value["distribution"])
        ):
            raise ProviderDiscoveryError(f"component {component.component!r} provenance does not match provider")
        versions = value["category_versions"].get(component.category, ())
        if component.category_version not in versions:
            raise ProviderDiscoveryError(f"component {component.component!r} category version is unsupported")
    factories = dict(sorted(value["python_factories"].items()))
    if not set(factories).issubset(component_ids):
        raise ProviderDiscoveryError(f"provider {value['provider']!r} maps a factory for an unknown component")
    for component, target in factories.items():
        target_module, _attribute = target.split(":", 1)
        if target_module.split(".", 1)[0] != package_name:
            raise ProviderDiscoveryError(f"component {component!r} factory is outside its provider package")
    resource_args = {
        "package_name": package_name,
        "package_root": package_root,
        "distribution_root": distribution_root,
        "listed": listed,
        "distribution": distribution,
    }
    schemas = tuple(sorted((_resource(item, **resource_args) for item in value["schemas"]), key=lambda item: item.path))
    docs = tuple(sorted((_resource(item, **resource_args) for item in value["docs"]), key=lambda item: item.path))
    native = tuple(
        sorted(
            (_resource(item, **resource_args) for item in value["native_artifacts"]),
            key=lambda item: item.path,
        )
    )
    if any(
        resource.abi_version != HOST_NATIVE_PLUGIN_ABI_VERSION for resource in native
    ):
        raise ProviderDiscoveryError(
            f"provider {value['provider']!r} declares an unsupported native plugin ABI"
        )
    resource_paths = [resource.path for resource in (*schemas, *docs, *native)]
    if len(resource_paths) != len(set(resource_paths)):
        raise ProviderDiscoveryError(f"provider {value['provider']!r} repeats a resource path")
    resource_digests = {
        resource.path: resource.sha256 for resource in (*schemas, *docs, *native)
    }
    for component in components:
        for artifact in component.to_dict()["artifacts"]:
            if resource_digests.get(artifact["path"]) != artifact["sha256"]:
                raise ProviderDiscoveryError(
                    f"component {component.component!r} references an unverified artifact"
                )
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    classes = tuple(_external_component_class(component) for component in components)
    tested_builds = tuple(
        sorted(
            (_freeze(item) for item in value["tested_builds"]),
            key=lambda item: json.dumps(_thaw(item), sort_keys=True, separators=(",", ":")),
        )
    )
    return ProviderDescriptor(
        provider=value["provider"],
        distribution=value["distribution"],
        version=value["version"],
        host_versions=tuple(sorted(value["host_versions"])),
        category_versions={key: tuple(sorted(versions)) for key, versions in value["category_versions"].items()},
        components=components,
        schemas=schemas,
        docs=docs,
        native_artifacts=native,
        python_factories=factories,
        tested_builds=tested_builds,
        entry_point=f"{entry_name}={entry_value}",
        manifest_digest=hashlib.sha256(_DIGEST_DOMAIN + canonical).hexdigest(),
        editable=editable,
        component_classes=classes,
        _verified=True,
    )


def discover_providers(
    *,
    entry_points: Iterable[Any] | None = None,
    allow_editable: bool = False,
    host_version: str = HOST_COMPONENT_STANDARD_VERSION,
) -> tuple[ProviderDescriptor, ...]:
    selected = tuple(
        importlib.metadata.entry_points().select(group=PROVIDER_ENTRY_POINT_GROUP)
        if entry_points is None
        else entry_points
    )
    selected = tuple(
        sorted(
            selected,
            key=lambda item: (str(getattr(item, "name", "")), str(getattr(item, "value", ""))),
        )
    )
    providers = tuple(
        sorted(
            (_discover_one(entry_point, allow_editable=allow_editable, host_version=host_version) for entry_point in selected),
            key=lambda provider: (provider.provider, provider.distribution, provider.version),
        )
    )
    provider_ids = [provider.provider for provider in providers]
    distributions = [_canonical_distribution(provider.distribution) for provider in providers]
    if len(provider_ids) != len(set(provider_ids)):
        raise ProviderDiscoveryError("duplicate provider identity")
    if len(distributions) != len(set(distributions)):
        raise ProviderDiscoveryError("duplicate provider distribution entry point")
    identities: set[str] = set()
    for provider in providers:
        for component in provider.components:
            if component.component in identities:
                raise ProviderDiscoveryError(f"duplicate external component identity {component.component!r}")
            identities.add(component.component)
    return providers


def provider_component_classes(
    providers: Iterable[ProviderDescriptor],
) -> tuple[type[ComponentRef], ...]:
    from pllm.components import list_component_classes

    providers = tuple(providers)
    if any(not isinstance(provider, ProviderDescriptor) or not provider._verified for provider in providers):
        raise ProviderDiscoveryError("provider descriptor was not produced by verified discovery")
    builtins = {component.describe().component for component in list_component_classes()}
    classes = tuple(
        sorted(
            (component for provider in providers for component in provider.component_classes),
            key=lambda component: component.describe().component,
        )
    )
    identities: set[str] = set()
    for component in classes:
        identity = component.describe().component
        if identity in builtins or identity in identities:
            raise ProviderDiscoveryError(f"component identity collision {identity!r}")
        identities.add(identity)
    return classes


def load_component_factory(
    provider: ProviderDescriptor,
    component: str,
    *,
    approved_providers: Collection[str],
) -> Any:
    if not isinstance(provider, ProviderDescriptor) or not provider._verified:
        raise ProviderDiscoveryError("provider descriptor was not produced by verified discovery")
    if isinstance(approved_providers, (str, bytes)):
        raise ProviderDiscoveryError("approved_providers must be a collection of provider identities")
    if provider.provider not in approved_providers:
        raise ProviderDiscoveryError(f"provider {provider.provider!r} is not approved for code loading")
    target = provider.python_factories.get(component)
    if target is None:
        raise ProviderDiscoveryError(f"provider {provider.provider!r} has no Python factory for {component!r}")
    module_name, attribute = target.split(":", 1)
    try:
        value: Any = importlib.import_module(module_name)
        for name in attribute.split("."):
            value = getattr(value, name)
        return value
    except Exception as exc:
        raise ProviderDiscoveryError(
            f"approved provider {provider.provider!r} cannot load factory for {component!r}"
        ) from exc


__all__ = [
    "HOST_COMPONENT_STANDARD_VERSION",
    "HOST_NATIVE_PLUGIN_ABI_VERSION",
    "PROVIDER_ENTRY_POINT_GROUP",
    "PROVIDER_MANIFEST_SCHEMA",
    "ProviderDescriptor",
    "ProviderDiscoveryError",
    "ProviderResource",
    "discover_providers",
    "load_component_factory",
    "provider_component_classes",
]
