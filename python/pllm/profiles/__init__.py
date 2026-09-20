"""Typed built-in pipeline profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pllm.configuration import ConfigurationError, Model, Pipeline
from pllm.kernels import Cpu, KernelBackend
from pllm.preparation import ModelAwareCorrections, PreparationProvider
from pllm.protocols import (
    BlindedLinear,
    DirectFHE as DirectFHEMethod,
    GuardedLinear,
    MaskedLinear,
    ProtocolMethod,
)
from pllm.roles import Inference, InferenceRole
from pllm.sources import ModelSource
from pllm.verification import FreivaldsVerify, VerificationScheme

_DEFAULT_MASKED = MaskedLinear()
_DEFAULT_GUARDED = GuardedLinear()
_DEFAULT_BLINDED = BlindedLinear()
_DEFAULT_DIRECT = DirectFHEMethod()
_DEFAULT_PREPARATION = ModelAwareCorrections()
_DEFAULT_INFERENCE = Inference()
_DEFAULT_KERNELS = Cpu()
_DEFAULT_FREIVALDS = FreivaldsVerify()


def _model(value: ModelSource) -> Model:
    if not isinstance(value, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    return value if isinstance(value, Model) else Model.from_spec(value.to_spec())


def _slot(name: str, value: Any, category: type, identity: str | None = None) -> Any:
    if not isinstance(value, category):
        raise ConfigurationError(f"{name} must be a {category.__name__}")
    if identity is not None and value.component != identity:
        raise ConfigurationError(f"{name} must be component {identity}")
    return value


class _TypedPipeline(Pipeline):
    __slots__ = ()
    SLOT_NAMES: tuple[str, ...] = ()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        values: dict[str, Any] = {"model": self.model}
        values.update({name: self.components[name] for name in self.SLOT_NAMES})
        if deep:
            for name, value in tuple(values.items()):
                for nested_name, nested_value in value.get_params(deep=True).items():
                    values[f"{name}__{nested_name}"] = nested_value
        return values

    def with_params(self, **changes: object) -> _TypedPipeline:
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


class MaskedLinearCpu(_TypedPipeline):
    PROFILE = "baseline.masked_linear_cpu"
    SLOT_NAMES = ("linear", "preparation", "inference", "kernels")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_MASKED,
        preparation: PreparationProvider = _DEFAULT_PREPARATION,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
    ) -> None:
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod),
                "preparation": _slot("preparation", preparation, PreparationProvider),
                "inference": _slot("inference", inference, InferenceRole),
                "kernels": _slot("kernels", kernels, KernelBackend),
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


class VerifiedMaskedLinearCpu(_TypedPipeline):
    PROFILE = "research.verified_masked_linear_cpu"
    SLOT_NAMES = ("linear", "preparation", "inference", "kernels", "verification")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_MASKED,
        preparation: PreparationProvider = _DEFAULT_PREPARATION,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
        verification: VerificationScheme = _DEFAULT_FREIVALDS,
    ) -> None:
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod, "pllm/masked-linear"),
                "preparation": _slot(
                    "preparation", preparation, PreparationProvider, "pllm/model-aware-corrections"
                ),
                "inference": _slot("inference", inference, InferenceRole, "pllm/inference"),
                "kernels": _slot("kernels", kernels, KernelBackend, "pllm/cpu"),
                "verification": _slot(
                    "verification", verification, VerificationScheme, "pllm/freivalds-verify/v1"
                ),
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

    @property
    def verification(self) -> FreivaldsVerify:
        return self.components["verification"]


class ProprietaryGuarded(_TypedPipeline):
    PROFILE = "runtime.proprietary_guarded"
    SLOT_NAMES = ("linear", "inference", "kernels")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: GuardedLinear = _DEFAULT_GUARDED,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
    ) -> None:
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod, "pllm/guarded-linear/v1"),
                "inference": _slot("inference", inference, InferenceRole, "pllm/inference"),
                "kernels": _slot("kernels", kernels, KernelBackend, "pllm/cpu"),
            },
        )

    @property
    def linear(self) -> GuardedLinear:
        return self.components["linear"]

    @property
    def inference(self) -> InferenceRole:
        return self.components["inference"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]


class ProprietaryBlinded(_TypedPipeline):
    PROFILE = "runtime.proprietary_blinded"
    SLOT_NAMES = ("linear", "inference", "kernels")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: BlindedLinear = _DEFAULT_BLINDED,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
    ) -> None:
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod, "pllm/blinded-linear/v1"),
                "inference": _slot("inference", inference, InferenceRole, "pllm/inference"),
                "kernels": _slot("kernels", kernels, KernelBackend, "pllm/cpu"),
            },
        )

    @property
    def linear(self) -> BlindedLinear:
        return self.components["linear"]

    @property
    def inference(self) -> InferenceRole:
        return self.components["inference"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]


class DirectFHEProfile(_TypedPipeline):
    PROFILE = "runtime.direct_fhe"
    SLOT_NAMES = ("linear", "inference", "kernels")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: DirectFHEMethod = _DEFAULT_DIRECT,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
    ) -> None:
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod, "pllm/direct-fhe"),
                "inference": _slot("inference", inference, InferenceRole, "pllm/inference"),
                "kernels": _slot("kernels", kernels, KernelBackend, "pllm/cpu"),
            },
        )

    @property
    def linear(self) -> DirectFHEMethod:
        return self.components["linear"]

    @property
    def inference(self) -> InferenceRole:
        return self.components["inference"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]


@dataclass(frozen=True, slots=True)
class _RuntimeProfileOptions:
    privacy_mode: str
    proprietary_protocol: str
    requires_preparation: bool
    correlation_mode: str
    client_runtime: str
    privacy_protocol: str | None
    guard_max_rows_per_request: int = 4096
    guard_max_rows_per_owner_stage: int = 16_384
    guard_max_requests_per_minute: int = 4096
    output_dither_bound: int = 0
    verification_component: str | None = None
    verification_target_failure_bits: int = 0


def _runtime_profile_options(pipeline: Pipeline) -> _RuntimeProfileOptions | None:
    identities = {name: component.component for name, component in pipeline.components.items()}
    kernels = pipeline.components.get("kernels")
    kernels_valid = kernels is not None and set(kernels.params) == {"threads"}
    if isinstance(pipeline, VerifiedMaskedLinearCpu):
        expected = {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
            "verification": "pllm/freivalds-verify/v1",
        }
        verification = pipeline.components["verification"]
        if (
            identities != expected
            or not kernels_valid
            or set(verification.params) != {"target_failure_bits"}
        ):
            return None
        return _RuntimeProfileOptions(
            privacy_mode="public",
            proprietary_protocol="guarded",
            requires_preparation=True,
            correlation_mode="bfv",
            client_runtime="masked_transformer_v1",
            privacy_protocol=None,
            verification_component="pllm/freivalds-verify/v1",
            verification_target_failure_bits=verification.params["target_failure_bits"],
        )
    if (
        isinstance(pipeline, MaskedLinearCpu)
        and identities
        == {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.linear.params
        and not pipeline.preparation.params
        and not pipeline.inference.params
    ):
        return _RuntimeProfileOptions(
            "public", "guarded", True, "bfv", "masked_transformer_v1", None
        )
    if (
        isinstance(pipeline, ProprietaryGuarded)
        and identities
        == {
            "linear": "pllm/guarded-linear/v1",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.inference.params
    ):
        params = pipeline.linear.params
        return _RuntimeProfileOptions(
            "proprietary",
            "guarded",
            False,
            "bfv",
            "guarded_blinded_transformer_v1",
            "guarded_blinded_w4a4",
            guard_max_rows_per_request=params["max_rows_per_request"],
            guard_max_rows_per_owner_stage=params["max_rows_per_owner_stage"],
            guard_max_requests_per_minute=params["max_requests_per_minute"],
            output_dither_bound=params["output_dither_bound"],
        )
    if (
        isinstance(pipeline, ProprietaryBlinded)
        and identities
        == {
            "linear": "pllm/blinded-linear/v1",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.linear.params
        and not pipeline.inference.params
    ):
        return _RuntimeProfileOptions(
            "proprietary",
            "blinded",
            False,
            "bfv",
            "blinded_ole_transformer_v1",
            "blinded_ole_w4a4",
        )
    if (
        isinstance(pipeline, DirectFHEProfile)
        and identities
        == {
            "linear": "pllm/direct-fhe",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.linear.params
        and not pipeline.inference.params
    ):
        return _RuntimeProfileOptions(
            "proprietary",
            "direct",
            False,
            "bfv",
            "direct_fhe_transformer_v1",
            "direct_bfv_w4a4",
        )
    return None


__all__ = [
    "DirectFHEProfile",
    "MaskedLinearCpu",
    "VerifiedMaskedLinearCpu",
    "ProprietaryBlinded",
    "ProprietaryGuarded",
]
