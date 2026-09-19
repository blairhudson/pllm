"""Typed built-in pipeline profiles."""

from __future__ import annotations

from typing import Any

from pllm.configuration import ConfigurationError, Model, Pipeline
from pllm.kernels import Cpu, KernelBackend
from pllm.preparation import ModelAwareCorrections, PreparationProvider
from pllm.protocols import MaskedLinear, ProtocolMethod
from pllm.roles import Inference, InferenceRole
from pllm.sources import ModelSource

_DEFAULT_LINEAR = MaskedLinear()
_DEFAULT_PREPARATION = ModelAwareCorrections()
_DEFAULT_INFERENCE = Inference()
_DEFAULT_KERNELS = Cpu()


class MaskedLinearCpu(Pipeline):
    PROFILE = "baseline.masked_linear_cpu"
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_LINEAR,
        preparation: PreparationProvider = _DEFAULT_PREPARATION,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
    ) -> None:
        if not isinstance(model, ModelSource):
            raise ConfigurationError("model must satisfy the ModelSource contract")
        normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
        for name, value, category in (
            ("linear", linear, ProtocolMethod),
            ("preparation", preparation, PreparationProvider),
            ("inference", inference, InferenceRole),
            ("kernels", kernels, KernelBackend),
        ):
            if not isinstance(value, category):
                raise ConfigurationError(f"{name} must be a {category.__name__}")
        super().__init__(
            profile=self.PROFILE,
            model=normalized_model,
            components={
                "linear": linear,
                "preparation": preparation,
                "inference": inference,
                "kernels": kernels,
            },
        )

    @property
    def linear(self) -> ProtocolMethod:
        return self.components["linear"]

    @property
    def preparation(self) -> PreparationProvider:
        return self.components["preparation"]

    @property
    def inference(self) -> InferenceRole:
        return self.components["inference"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        values: dict[str, Any] = {
            "model": self.model,
            "linear": self.linear,
            "preparation": self.preparation,
            "inference": self.inference,
            "kernels": self.kernels,
        }
        if deep:
            for name, value in tuple(values.items()):
                for nested_name, nested_value in value.get_params(deep=True).items():
                    values[f"{name}__{nested_name}"] = nested_value
        return values

    def with_params(self, **changes: object) -> MaskedLinearCpu:
        paths = [name.split("__") for name in changes]
        if any(not path or any(not item for item in path) for path in paths):
            raise ConfigurationError("parameter paths must contain non-empty '__'-separated names")
        for index, path in enumerate(paths):
            for other in paths[index + 1 :]:
                shorter = min(len(path), len(other))
                if path[:shorter] == other[:shorter]:
                    raise ConfigurationError("overlapping parameter paths are not allowed")
        values = self.get_params(deep=False)
        for path, value in zip(paths, changes.values(), strict=True):
            name, *nested = path
            if name not in values:
                raise ConfigurationError(f"unknown parameter path: {'__'.join(path)}")
            if nested:
                current = values[name]
                if not hasattr(current, "with_params"):
                    raise ConfigurationError(
                        f"parameter path continues through non-configuration value {name!r}"
                    )
                values[name] = current.with_params(**{"__".join(nested): value})
            else:
                values[name] = value
        return type(self)(**values)


__all__ = ["MaskedLinearCpu"]
