"""Public compiler pass declarations."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError


def _integer(value: object, path: str) -> int:
    if type(value) is not int or value < 1:
        raise ConfigurationError(f"{path} must be an integer >= 1")
    return value


def _ratio(value: object, path: str) -> dict[str, int]:
    if isinstance(value, Mapping):
        if set(value) != {"numerator", "denominator"}:
            raise ConfigurationError(f"{path} must contain numerator and denominator")
        numerator = _integer(value["numerator"], f"{path}.numerator")
        denominator = _integer(value["denominator"], f"{path}.denominator")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        numerator = _integer(value[0], f"{path}.numerator")
        denominator = _integer(value[1], f"{path}.denominator")
    else:
        raise ConfigurationError(f"{path} must be a (numerator, denominator) pair")
    if numerator > denominator:
        raise ConfigurationError(f"{path} must be in (0, 1]")
    return {"numerator": numerator, "denominator": denominator}


class PlanPass(ComponentRef, ABC):
    __slots__ = ()

    @classmethod
    @abstractmethod
    def describe(cls) -> ComponentDescriptor: ...


class KvCacheEviction(PlanPass):
    """Bounded KV-cache eviction component backed by a named implementation."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/kv-cache-eviction",
        provider="pllm",
        distribution="pllm",
        version="1",
        category="pllm/compiler-pass",
        category_version="1",
        lifecycle_phase="model-lowering",
        parameter_schema={
            "type": "object",
            "properties": {
                "implementation": {"type": "string", "minLength": 1},
                "observation_window": {"$ref": "#/$defs/ratio"},
                "static_keep": {"$ref": "#/$defs/ratio"},
                "dynamic_keep": {"$ref": "#/$defs/ratio"},
                "alpha": {"$ref": "#/$defs/ratio"},
                "cluster_sizes": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "integer", "minimum": 1},
                },
                "share_adjacent_layers": {"type": "boolean"},
            },
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
            "$defs": {
                "ratio": {
                    "type": "object",
                    "properties": {
                        "numerator": {"type": "integer", "minimum": 1},
                        "denominator": {"type": "integer", "minimum": 1},
                    },
                    "required": ["numerator", "denominator"],
                    "additionalProperties": False,
                }
            },
        },
        capabilities=("bounded-kv-cache-eviction",),
        required_host_features=("decoder-plan-v1",),
    )

    def __init__(
        self,
        *,
        implementation: str = "pllm/importance-kv-cache-eviction/v1",
        observation_window: tuple[int, int] | Mapping[str, int] = (1, 5),
        static_keep: tuple[int, int] | Mapping[str, int] = (3, 10),
        dynamic_keep: tuple[int, int] | Mapping[str, int] = (1, 4),
        alpha: tuple[int, int] | Mapping[str, int] = (3, 5),
        cluster_sizes: Sequence[int] = (32, 16),
        share_adjacent_layers: bool = True,
    ) -> None:
        if type(implementation) is not str or not implementation:
            raise ConfigurationError("implementation must be a non-empty string")
        sizes = tuple(
            _integer(size, f"cluster_sizes[{index}]") for index, size in enumerate(cluster_sizes)
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
            self.descriptor.component,
            {
                "implementation": implementation,
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
        return cls.descriptor


__all__ = ["KvCacheEviction", "PlanPass"]
