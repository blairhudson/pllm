"""Immutable public configuration values and strict serialization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import yaml

_DIGEST_DOMAIN = b"pllm.configuration.v1\0"
_EXPERIMENT_SCHEMA = "pllm.experiment.v1"
_MAX_DOCUMENT_BYTES = 1_048_576


class ConfigurationError(ValueError):
    """Configuration is malformed or outside the supported schema."""


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Static, immutable identity and compatibility metadata for a built-in component."""

    component: str
    provider: str
    distribution: str
    version: str
    category: str
    category_version: str
    lifecycle_phase: str
    parameter_schema: Mapping[str, Any]
    capabilities: tuple[str, ...] = ()
    required_host_features: tuple[str, ...] = ()
    role_eligibility: tuple[str, ...] = ()
    artifacts: tuple[Mapping[str, Any], ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "component",
            "provider",
            "distribution",
            "version",
            "category",
            "category_version",
            "lifecycle_phase",
        ):
            _string(getattr(self, name), f"descriptor.{name}")
        object.__setattr__(
            self,
            "parameter_schema",
            _freeze_json(self.parameter_schema, "descriptor.parameter_schema"),
        )
        for name in ("capabilities", "required_host_features", "role_eligibility"):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise ConfigurationError(f"descriptor.{name} must be a tuple")
            object.__setattr__(
                self,
                name,
                tuple(_string(value, f"descriptor.{name}") for value in values),
            )
        for name in ("artifacts", "evidence"):
            values = getattr(self, name)
            if type(values) is not tuple:
                raise ConfigurationError(f"descriptor.{name} must be a tuple")
            frozen = _freeze_json(values, f"descriptor.{name}")
            object.__setattr__(self, name, frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "provider": self.provider,
            "distribution": self.distribution,
            "version": self.version,
            "category": self.category,
            "category_version": self.category_version,
            "lifecycle_phase": self.lifecycle_phase,
            "parameter_schema": _thaw_json(self.parameter_schema),
            "capabilities": list(self.capabilities),
            "required_host_features": list(self.required_host_features),
            "role_eligibility": list(self.role_eligibility),
            "artifacts": _thaw_json(self.artifacts),
            "evidence": _thaw_json(self.evidence),
        }

    def __hash__(self) -> int:
        return hash(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value:
        raise ConfigurationError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ConfigurationError(f"{path} must be an integer >= {minimum}")
    return value


def _freeze_json(value: object, path: str = "value", active: set[int] | None = None) -> Any:
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ConfigurationError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ConfigurationError(f"{path} contains a cycle")
        active.add(identity)
        frozen_mapping: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ConfigurationError(f"{path} keys must be strings")
            if not key or "__" in key:
                raise ConfigurationError(f"{path} contains invalid key {key!r}")
            frozen_mapping[key] = _freeze_json(item, f"{path}.{key}", active)
        active.remove(identity)
        return MappingProxyType(frozen_mapping)
    if isinstance(value, (list, tuple)):
        active = set() if active is None else active
        identity = id(value)
        if identity in active:
            raise ConfigurationError(f"{path} contains a cycle")
        active.add(identity)
        frozen_sequence = tuple(
            _freeze_json(item, f"{path}[{index}]", active) for index, item in enumerate(value)
        )
        active.remove(identity)
        return frozen_sequence
    raise ConfigurationError(f"{path} has unsupported type {type(value).__name__}")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _hashable_json(value: object) -> object:
    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable_json(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_hashable_json(item) for item in value)
    return value


def _flatten_mapping(prefix: str, value: Mapping[str, Any], output: dict[str, Any]) -> None:
    for key, item in value.items():
        path = f"{prefix}__{key}"
        output[path] = item
        if isinstance(item, Mapping):
            _flatten_mapping(path, item, output)


class _Configuration:
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        params = {field.name: getattr(self, field.name) for field in fields(cast(Any, self))}
        if not deep:
            return params
        output = dict(params)
        for name, value in params.items():
            if isinstance(value, _Configuration):
                for nested_name, nested_value in value.get_params(deep=True).items():
                    output[f"{name}__{nested_name}"] = nested_value
            elif isinstance(value, Mapping):
                for key, item in value.items():
                    path = f"{name}__{key}"
                    output[path] = item
                    if isinstance(item, _Configuration):
                        for nested_name, nested_value in item.get_params(deep=True).items():
                            output[f"{path}__{nested_name}"] = nested_value
                    elif isinstance(item, Mapping):
                        _flatten_mapping(path, item, output)
        return output

    def with_params(self, **changes: object) -> Any:
        paths = [path.split("__") for path in changes]
        if any(not path or any(not part for part in path) for path in paths):
            raise ConfigurationError("parameter paths must contain non-empty '__'-separated names")
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                shorter = min(len(path), len(other))
                if path[:shorter] == other[:shorter]:
                    raise ConfigurationError("overlapping parameter paths are not allowed")
        result: Any = self
        for path, value in zip(paths, changes.values(), strict=True):
            result = _replace_path(result, path, value)
        return result


@dataclass(frozen=True, slots=True)
class Model(_Configuration):
    source: str

    def __post_init__(self) -> None:
        _string(self.source, "model.source")

    def to_spec(self) -> dict[str, Any]:
        return {"source": self.source}


@dataclass(frozen=True, slots=True)
class ComponentRef(_Configuration):
    component: str
    params: Mapping[str, Any]

    def __init__(self, component: str, params: Mapping[str, Any] | None = None) -> None:
        object.__setattr__(self, "component", _string(component, "component.component"))
        frozen = _freeze_json({} if params is None else params, "component.params")
        if "component" in frozen or "params" in frozen:
            raise ConfigurationError("component parameter names cannot be 'component' or 'params'")
        object.__setattr__(self, "params", frozen)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        output = {"component": self.component, **self.params}
        if deep:
            for key, value in self.params.items():
                if isinstance(value, Mapping):
                    _flatten_mapping(key, value, output)
        return output

    def to_spec(self) -> dict[str, Any]:
        return {"component": self.component, "params": _thaw_json(self.params)}

    def __hash__(self) -> int:
        return hash((self.component, _hashable_json(self.params)))


class Cpu(ComponentRef):
    __slots__ = ()

    def __init__(self, *, threads: int = 1) -> None:
        super().__init__("pllm/cpu", {"threads": _integer(threads, "cpu.threads")})

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"threads": self.params["threads"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return _BUILTIN_DESCRIPTORS["pllm/cpu"]


class ModelAwareCorrections(ComponentRef):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("pllm/model-aware-corrections")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return _BUILTIN_DESCRIPTORS["pllm/model-aware-corrections"]


class MaskedLinear(ComponentRef):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("pllm/masked-linear")

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return _BUILTIN_DESCRIPTORS["pllm/masked-linear"]


class KvCacheEviction(ComponentRef):
    """Bounded KV-cache eviction component backed by a named implementation."""

    __slots__ = ()

    def __init__(
        self,
        *,
        implementation: str = "pllm/mpcache/v1",
        observation_window: tuple[int, int] = (1, 5),
        static_keep: tuple[int, int] = (3, 10),
        dynamic_keep: tuple[int, int] = (1, 4),
        alpha: tuple[int, int] = (3, 5),
        cluster_sizes: tuple[int, ...] = (32, 16),
        share_adjacent_layers: bool = True,
    ) -> None:
        sizes = tuple(
            _integer(size, f"cluster_sizes[{index}]")
            for index, size in enumerate(cluster_sizes)
        )
        if not sizes:
            raise ConfigurationError("cluster_sizes must be non-empty")
        if any(left <= right or left % right for left, right in zip(sizes, sizes[1:])):
            raise ConfigurationError(
                "cluster_sizes must be a strictly descending divisible hierarchy"
            )
        if type(share_adjacent_layers) is not bool:
            raise ConfigurationError("share_adjacent_layers must be a bool")
        super().__init__(
            "pllm/kv-cache-eviction",
            {
                "implementation": _string(implementation, "implementation"),
                "observation_window": _ratio(observation_window, "observation_window"),
                "static_keep": _ratio(static_keep, "static_keep"),
                "dynamic_keep": _ratio(dynamic_keep, "dynamic_keep"),
                "alpha": _ratio(alpha, "alpha"),
                "cluster_sizes": sizes,
                "share_adjacent_layers": share_adjacent_layers,
            },
        )

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return _BUILTIN_DESCRIPTORS["pllm/kv-cache-eviction"]


def _ratio(value: object, path: str) -> dict[str, int]:
    if isinstance(value, Mapping):
        data = _fields(value, {"numerator", "denominator"}, path)
        numerator = _integer(data["numerator"], f"{path}.numerator")
        denominator = _integer(data["denominator"], f"{path}.denominator")
    elif type(value) is tuple and len(value) == 2:
        numerator = _integer(value[0], f"{path}.numerator")
        denominator = _integer(value[1], f"{path}.denominator")
    else:
        raise ConfigurationError(f"{path} must be a (numerator, denominator) tuple")
    if numerator > denominator:
        raise ConfigurationError(f"{path} must be in (0, 1]")
    return {"numerator": numerator, "denominator": denominator}


@dataclass(frozen=True, slots=True)
class Pipeline(_Configuration):
    profile: str
    model: Model
    components: Mapping[str, ComponentRef]

    def __post_init__(self) -> None:
        _string(self.profile, "pipeline.profile")
        if not isinstance(self.model, Model):
            raise ConfigurationError("pipeline.model must be a Model")
        if not isinstance(self.components, Mapping):
            raise ConfigurationError("pipeline.components must be a mapping")
        copied: dict[str, ComponentRef] = {}
        for name, component in self.components.items():
            _string(name, "pipeline component name")
            if "__" in name:
                raise ConfigurationError("pipeline component names cannot contain '__'")
            if not isinstance(component, ComponentRef):
                raise ConfigurationError(f"pipeline.components.{name} must be a ComponentRef")
            copied[name] = component
        object.__setattr__(self, "components", MappingProxyType(copied))

    @classmethod
    def from_profile(
        cls,
        profile: str,
        *,
        model: Model,
        components: Mapping[str, ComponentRef],
    ) -> Pipeline:
        return cls(profile=profile, model=model, components=components)

    def to_spec(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "model": self.model.to_spec(),
            "components": {
                name: component.to_spec() for name, component in self.components.items()
            },
        }

    def __hash__(self) -> int:
        return hash((self.profile, self.model, tuple(sorted(self.components.items()))))


@dataclass(frozen=True, slots=True)
class Deployment(_Configuration):
    kind: str
    root: str

    def __post_init__(self) -> None:
        if self.kind != "local":
            raise ConfigurationError("deployment.kind must be 'local'")
        _string(self.root, "deployment.root")

    @classmethod
    def local(cls, *, root: str) -> Deployment:
        return cls(kind="local", root=root)

    def to_spec(self) -> dict[str, Any]:
        return {"kind": self.kind, "root": self.root}


@dataclass(frozen=True, slots=True)
class ExecutionBudget(_Configuration):
    requests: int
    max_input_tokens: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        _integer(self.requests, "budget.requests")
        _integer(self.max_input_tokens, "budget.max_input_tokens")
        _integer(self.max_new_tokens, "budget.max_new_tokens")

    def to_spec(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True, slots=True)
class Experiment(_Configuration):
    name: str
    pipeline: Pipeline
    deployment: Deployment
    budget: ExecutionBudget

    def __post_init__(self) -> None:
        _string(self.name, "experiment.name")
        if not isinstance(self.pipeline, Pipeline):
            raise ConfigurationError("experiment.pipeline must be a Pipeline")
        if not isinstance(self.deployment, Deployment):
            raise ConfigurationError("experiment.deployment must be a Deployment")
        if not isinstance(self.budget, ExecutionBudget):
            raise ConfigurationError("experiment.budget must be an ExecutionBudget")

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> Experiment:
        return _experiment_from_spec(spec)

    @classmethod
    def from_file(cls, path: str | Path) -> Experiment:
        return load_configuration(path)

    def to_spec(self) -> dict[str, Any]:
        return {
            "schema": _EXPERIMENT_SCHEMA,
            "name": self.name,
            "pipeline": self.pipeline.to_spec(),
            "deployment": self.deployment.to_spec(),
            "budget": self.budget.to_spec(),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self)

    def configuration_digest(self) -> str:
        return configuration_digest(self)

    def resolve(self) -> ExperimentProfile:
        return ExperimentProfile(self)


@dataclass(frozen=True, slots=True, init=False)
class ExperimentProfile:
    """Immutable native resolution of one supported Experiment profile."""

    model: str
    canonical_profile: bytes
    configuration_digest: str

    def __init__(self, experiment: Experiment) -> None:
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be an Experiment")
        from pllm import _native

        try:
            resolved = _native.resolve_experiment(experiment.canonical_bytes())
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        object.__setattr__(self, "model", resolved.model)
        object.__setattr__(self, "canonical_profile", resolved.canonical_profile)
        object.__setattr__(self, "configuration_digest", resolved.configuration_digest)


def _replace_path(target: Any, path: list[str], value: object) -> Any:
    name, *rest = path
    if isinstance(target, ComponentRef):
        current = target.get_params(deep=False)
        if name not in current:
            raise ConfigurationError(f"unknown parameter path: {'__'.join(path)}")
        if rest:
            changed = _replace_path(current[name], rest, value)
        else:
            changed = value
        if type(target) is ComponentRef:
            component = changed if name == "component" else target.component
            params = dict(target.params)
            if name != "component":
                params[name] = changed
            return ComponentRef(cast(str, component), params)
        params = dict(target.params)
        params[name] = changed
        return _component_from_spec(
            {"component": target.component, "params": params},
            "component",
        )
    if isinstance(target, _Configuration):
        field_names = {field.name for field in fields(cast(Any, target))}
        if name not in field_names:
            raise ConfigurationError(f"unknown parameter path: {'__'.join(path)}")
        changed = _replace_path(getattr(target, name), rest, value) if rest else value
        return replace(cast(Any, target), **{name: changed})
    if isinstance(target, Mapping):
        if name not in target:
            raise ConfigurationError(f"unknown parameter path: {'__'.join(path)}")
        copied = dict(target)
        copied[name] = _replace_path(target[name], rest, value) if rest else value
        return copied
    raise ConfigurationError(f"parameter path continues through non-configuration value {name!r}")


def _fields(value: object, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a mapping")
    keys: set[str] = set()
    for key in value:
        if type(key) is not str:
            raise ConfigurationError(f"{path} keys must be strings")
        keys.add(key)
    unknown = keys - expected
    missing = expected - keys
    if unknown:
        raise ConfigurationError(
            f"{path} has unknown fields: {', '.join(sorted(map(str, unknown)))}"
        )
    if missing:
        raise ConfigurationError(f"{path} is missing fields: {', '.join(sorted(missing))}")
    return value


def _component_from_spec(value: object, path: str) -> ComponentRef:
    data = _fields(value, {"component", "params"}, path)
    component = _string(data["component"], f"{path}.component")
    params = data["params"]
    if not isinstance(params, Mapping):
        raise ConfigurationError(f"{path}.params must be a mapping")
    if component == "pllm/cpu":
        exact = _fields(params, {"threads"}, f"{path}.params")
        return Cpu(threads=exact["threads"])
    if component == "pllm/model-aware-corrections":
        _fields(params, set(), f"{path}.params")
        return ModelAwareCorrections()
    if component == "pllm/masked-linear":
        _fields(params, set(), f"{path}.params")
        return MaskedLinear()
    if component == "pllm/kv-cache-eviction":
        exact = _fields(
            params,
            {
                "implementation",
                "observation_window",
                "static_keep",
                "dynamic_keep",
                "alpha",
                "cluster_sizes",
                "share_adjacent_layers",
            },
            f"{path}.params",
        )
        return KvCacheEviction(
            implementation=exact["implementation"],
            observation_window=exact["observation_window"],
            static_keep=exact["static_keep"],
            dynamic_keep=exact["dynamic_keep"],
            alpha=exact["alpha"],
            cluster_sizes=exact["cluster_sizes"],
            share_adjacent_layers=exact["share_adjacent_layers"],
        )
    return ComponentRef(component, params)


def _experiment_from_spec(value: object) -> Experiment:
    data = _fields(value, {"schema", "name", "pipeline", "deployment", "budget"}, "experiment")
    if data["schema"] != _EXPERIMENT_SCHEMA:
        raise ConfigurationError(f"unsupported schema: {data['schema']!r}")
    pipeline_data = _fields(data["pipeline"], {"profile", "model", "components"}, "pipeline")
    model_data = _fields(pipeline_data["model"], {"source"}, "pipeline.model")
    raw_components = pipeline_data["components"]
    if not isinstance(raw_components, Mapping):
        raise ConfigurationError("pipeline.components must be a mapping")
    components = {
        _string(name, "pipeline component name"): _component_from_spec(
            component, f"pipeline.components.{name}"
        )
        for name, component in raw_components.items()
    }
    deployment_data = _fields(data["deployment"], {"kind", "root"}, "deployment")
    budget_data = _fields(
        data["budget"], {"requests", "max_input_tokens", "max_new_tokens"}, "budget"
    )
    return Experiment(
        name=data["name"],
        pipeline=Pipeline.from_profile(
            pipeline_data["profile"],
            model=Model(model_data["source"]),
            components=components,
        ),
        deployment=Deployment(kind=deployment_data["kind"], root=deployment_data["root"]),
        budget=ExecutionBudget(
            requests=budget_data["requests"],
            max_input_tokens=budget_data["max_input_tokens"],
            max_new_tokens=budget_data["max_new_tokens"],
        ),
    )


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise ConfigurationError("mapping keys must be scalar strings") from exc
        if duplicate:
            raise ConfigurationError(f"duplicate mapping key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"duplicate mapping key: {key!r}")
        result[key] = value
    return result


def loads_configuration(text: str, *, format: str | None = None) -> Experiment:
    """Parse one strict JSON or safe YAML experiment document."""
    if type(text) is not str:
        raise TypeError("configuration text must be a string")
    try:
        document_bytes = len(text.encode("utf-8"))
    except UnicodeError as exc:
        raise ConfigurationError("configuration must be valid UTF-8") from exc
    if document_bytes > _MAX_DOCUMENT_BYTES:
        raise ConfigurationError(f"configuration exceeds {_MAX_DOCUMENT_BYTES} byte limit")
    selected = ("json" if text.lstrip().startswith("{") else "yaml") if format is None else format
    try:
        if selected == "json":
            value = json.loads(
                text,
                object_pairs_hook=_json_pairs,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ConfigurationError(f"non-finite number: {constant}")
                ),
            )
        elif selected in ("yaml", "yml"):
            value = yaml.load(text, Loader=_UniqueKeyLoader)
        else:
            raise ConfigurationError("format must be 'json' or 'yaml'")
        return _experiment_from_spec(value)
    except ConfigurationError:
        raise
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON: {exc.msg}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML: {exc}") from exc
    except (RecursionError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {selected} document structure") from exc


def load_configuration(path: str | Path) -> Experiment:
    """Read and parse one experiment file; constructors themselves perform no I/O."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in (".json", ".yaml", ".yml"):
        raise ConfigurationError("configuration file must use .json, .yaml, or .yml")
    if source.stat().st_size > _MAX_DOCUMENT_BYTES:
        raise ConfigurationError(f"configuration exceeds {_MAX_DOCUMENT_BYTES} byte limit")
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise ConfigurationError("configuration must be valid UTF-8") from exc
    return loads_configuration(text, format=suffix[1:])


def canonical_bytes(configuration: Experiment | Mapping[str, Any]) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a supported public specification."""
    if isinstance(configuration, Experiment):
        spec = configuration.to_spec()
    elif isinstance(configuration, Mapping):
        spec = _experiment_from_spec(configuration).to_spec()
    else:
        raise TypeError("configuration must be an Experiment or experiment mapping")
    return json.dumps(
        spec, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def configuration_digest(configuration: Experiment | Mapping[str, Any]) -> str:
    """Return domain-separated SHA-256 identity for public configuration."""
    return hashlib.sha256(_DIGEST_DOMAIN + canonical_bytes(configuration)).hexdigest()


_BUILTIN_DESCRIPTORS = {
    "pllm/cpu": ComponentDescriptor(
        component="pllm/cpu",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/kernel-backend",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={
            "type": "object",
            "properties": {"threads": {"type": "integer", "minimum": 1}},
            "required": ["threads"],
            "additionalProperties": False,
        },
        capabilities=("cpu",),
        role_eligibility=("inference",),
    ),
    "pllm/model-aware-corrections": ComponentDescriptor(
        component="pllm/model-aware-corrections",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/preparation-provider",
        category_version="1",
        lifecycle_phase="preparation",
        parameter_schema={"type": "object", "additionalProperties": False},
        role_eligibility=("preparation",),
    ),
    "pllm/masked-linear": ComponentDescriptor(
        component="pllm/masked-linear",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/protocol-method",
        category_version="1",
        lifecycle_phase="compilation",
        parameter_schema={"type": "object", "additionalProperties": False},
        capabilities=("masked-linear",),
        role_eligibility=("client", "preparation", "inference"),
    ),
    "pllm/kv-cache-eviction": ComponentDescriptor(
        component="pllm/kv-cache-eviction",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/compiler-pass",
        category_version="1",
        lifecycle_phase="model-lowering",
        parameter_schema={
            "type": "object",
            "required": [
                "implementation",
                "observation_window",
                "static_keep",
                "dynamic_keep",
                "alpha",
                "cluster_sizes",
                "share_adjacent_layers",
            ],
            "additionalProperties": False,
        },
        capabilities=("bounded-kv-cache-eviction",),
        required_host_features=("decoder-plan-v1",),
    ),
}


__all__ = [
    "ComponentDescriptor",
    "ComponentRef",
    "ConfigurationError",
    "Cpu",
    "Deployment",
    "ExecutionBudget",
    "Experiment",
    "ExperimentProfile",
    "KvCacheEviction",
    "MaskedLinear",
    "Model",
    "ModelAwareCorrections",
    "Pipeline",
    "canonical_bytes",
    "configuration_digest",
    "load_configuration",
    "loads_configuration",
]
