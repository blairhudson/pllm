"""Public compiler pass declarations."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError
from pllm.components._planned import PendingComponent as PendingMethod, install_planned_components, planned as pending
from pllm.components._model_capabilities import PendingModelCapability, model_capability_stub


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
        distribution="pllm.run",
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


@pending("atlas")
class AtlasApproximationSearch(PendingMethod):
    pass


@pending("encformer")
class EncformerRepresentationAssignment(PendingMethod):
    pass


@pending("fusefss")
class FuseFssPredicateFusion(PendingMethod):
    pass


@pending("rosetta")
class RosettaCkksTfheAssignment(PendingMethod):
    pass


@pending("breaking-layer-barrier")
class BreakingLayerBarrierAssignment(PendingMethod):
    pass


class CipherPruneTokenPruning(PlanPass):
    """Bounded operator benchmark for private pruning and disclosed degree selection."""

    __slots__ = ()
    descriptor = ComponentDescriptor(
        component="pllm/cipherprune-token-pruning/v1", provider="pllm",
        distribution="pllm.run", version="1", category="pllm/compiler-pass",
        category_version="1", lifecycle_phase="online-execution",
        parameter_schema={
            "type": "object", "additionalProperties": False,
            "properties": {
                "threshold_q16": {"type": "integer", "minimum": 0, "maximum": 65535},
                "reduction_threshold_q16": {"type": "integer", "minimum": 1, "maximum": 65536},
                "backend": {"type": "string", "enum": ["half_gates", "mixed"]},
                "batch_size": {"type": "integer", "enum": [1, 4, 16]},
            },
            "required": ["threshold_q16", "reduction_threshold_q16"],
        },
        capabilities=("bounded-private-token-pruning", "operator-benchmark"),
        required_host_features=("native-token-pruning-reference-v1",),
    )

    def __init__(self, *, threshold_q16: int = 8192, reduction_threshold_q16: int = 24576,
                 backend: str = "half_gates", batch_size: int = 1) -> None:
        if (type(threshold_q16) is not int or type(reduction_threshold_q16) is not int
                or not 0 <= threshold_q16 < reduction_threshold_q16 <= 65536):
            raise ConfigurationError("pruning thresholds must satisfy 0 <= theta < beta <= 65536")
        if type(backend) is not str or backend not in ("half_gates", "mixed"):
            raise ConfigurationError("pruning backend must be half_gates or mixed")
        if type(batch_size) is not int or batch_size not in (1, 4, 16) or (backend != "mixed" and batch_size != 1):
            raise ConfigurationError("pruning batch_size requires mixed backend and 1, 4 or 16")
        super().__init__(self.descriptor.component, {
            "threshold_q16": threshold_q16, "reduction_threshold_q16": reduction_threshold_q16,
            **({"backend": backend} if backend != "half_gates" else {}),
            **({"batch_size": batch_size} if batch_size != 1 else {}),
        })

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"backend": "half_gates", "batch_size": 1, **self.to_spec()["params"]}

    @classmethod
    def describe(cls) -> ComponentDescriptor:
        return cls.descriptor


@pending("redash")
class RedashGarbledScaling(PendingMethod):
    pass


@pending("ditto")
class DittoQuantizationAwareTransform(PendingMethod):
    pass


@pending("mpc-minimized")
class MpcMinimizedClientServerPartition(PendingMethod):
    pass


@pending("mpcformer")
class MpcFormerDistillation(PendingMethod):
    pass


@pending("hycc")
class HyccHybridProtocolAssignment(PendingMethod):
    pass


@pending("fevbdd")
class FevBddDecisionDiagramFactoring(PendingMethod):
    pass


@model_capability_stub("expert-routing")
class SparseExpertRouting(PendingModelCapability):
    pass


__all__ += [
    "AtlasApproximationSearch", "EncformerRepresentationAssignment", "FuseFssPredicateFusion",
    "RosettaCkksTfheAssignment", "BreakingLayerBarrierAssignment", "CipherPruneTokenPruning",
    "RedashGarbledScaling", "DittoQuantizationAwareTransform", "MpcMinimizedClientServerPartition",
    "MpcFormerDistillation", "HyccHybridProtocolAssignment", "FevBddDecisionDiagramFactoring",
    "SparseExpertRouting",
]

install_planned_components(globals())
