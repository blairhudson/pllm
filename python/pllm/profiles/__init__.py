"""Typed built-in pipeline profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pllm.configuration import ConfigurationError, Model, Pipeline
from pllm.kernels import AppleMetal, Cpu, KernelBackend
from pllm.preparation import ModelAwareCorrections, PreparationProvider
from pllm.quantization import PublicPerChannelEqualized, QuantizationScheme, SymmetricPerRow
from pllm.protocols import (
    BlindedLinear,
    CleartextLinear,
    DirectFHE as DirectFHEMethod,
    GuardedLinear,
    MaskedLinear,
    ProtocolMethod,
    TwoOnlineOffsetLinear,
)
from pllm.roles import (
    ClientOnlyRoles, Inference, InferenceRole, PreparedProviderRoles, RoleTopology,
    TwoOnlineOffsetRoles,
)
from pllm.sources import ModelSource
from pllm.verification import FreivaldsVerify, VerificationScheme

_DEFAULT_MASKED = MaskedLinear()
_DEFAULT_CLEARTEXT = CleartextLinear()
_DEFAULT_GUARDED = GuardedLinear()
_DEFAULT_BLINDED = BlindedLinear()
_DEFAULT_DIRECT = DirectFHEMethod()
_DEFAULT_PREPARATION = ModelAwareCorrections()
_DEFAULT_INFERENCE = Inference()
_DEFAULT_CLIENT_ONLY = ClientOnlyRoles()
_DEFAULT_OFFSET_LINEAR = TwoOnlineOffsetLinear()
_DEFAULT_OFFSET_ROLES = TwoOnlineOffsetRoles()
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
        values.update({name: self.components.get(name) for name in self.SLOT_NAMES})
        if deep:
            for name, value in tuple(values.items()):
                if value is None:
                    continue
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
    SLOT_NAMES = ("linear", "preparation", "inference", "kernels", "quantization")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_MASKED,
        preparation: PreparationProvider = _DEFAULT_PREPARATION,
        inference: InferenceRole = _DEFAULT_INFERENCE,
        kernels: KernelBackend = _DEFAULT_KERNELS,
        quantization: QuantizationScheme | None = None,
    ) -> None:
        if quantization is not None:
            _slot("quantization", quantization, QuantizationScheme)
            if quantization.component not in {
                SymmetricPerRow.descriptor.component,
                PublicPerChannelEqualized.descriptor.component,
            }:
                raise ConfigurationError("masked linear requires a supported quantization component")
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod),
                "preparation": _slot("preparation", preparation, PreparationProvider),
                "inference": _slot("inference", inference, InferenceRole),
                "kernels": _slot("kernels", kernels, KernelBackend),
                **({"quantization": quantization} if quantization is not None else {}),
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
    def quantization(self) -> QuantizationScheme | None:
        return self.components.get("quantization")


class ClientOnlyCpu(_TypedPipeline):
    """Client-owned decoder with local quantized body linear stages."""

    PROFILE = "baseline.client_only_cpu"
    SLOT_NAMES = ("linear", "kernels", "quantization", "topology")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_CLEARTEXT,
        kernels: KernelBackend = _DEFAULT_KERNELS,
        quantization: QuantizationScheme | None = None,
        topology: RoleTopology = _DEFAULT_CLIENT_ONLY,
    ) -> None:
        if quantization is not None:
            _slot("quantization", quantization, QuantizationScheme)
            if quantization.component not in {
                SymmetricPerRow.descriptor.component,
                PublicPerChannelEqualized.descriptor.component,
            }:
                raise ConfigurationError("client-only requires a supported quantization component")
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot(
                    "linear", linear, ProtocolMethod, CleartextLinear.descriptor.component,
                ),
                "kernels": _slot("kernels", kernels, KernelBackend),
                "topology": _slot(
                    "topology", topology, RoleTopology, ClientOnlyRoles.descriptor.component,
                ),
                **({"quantization": quantization} if quantization is not None else {}),
            },
        )

    @property
    def linear(self) -> CleartextLinear:
        return self.components["linear"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]

    @property
    def quantization(self) -> QuantizationScheme | None:
        return self.components.get("quantization")

    @property
    def topology(self) -> ClientOnlyRoles:
        return self.components["topology"]


class ClientOnlyMetal(ClientOnlyCpu):
    """Client-owned compiled decoder with Metal prefill and CPU decode."""

    PROFILE = "baseline.client_only_metal"
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_CLEARTEXT,
        kernels: KernelBackend = AppleMetal(),
        quantization: QuantizationScheme | None = None,
        topology: RoleTopology = _DEFAULT_CLIENT_ONLY,
    ) -> None:
        _slot("kernels", kernels, KernelBackend, AppleMetal.descriptor.component)
        super().__init__(
            model, linear=linear, kernels=kernels,
            quantization=quantization, topology=topology,
        )


class TwoOnlineOffsetCpu(_TypedPipeline):
    """Bounded two-public-worker additive offset comparator."""

    PROFILE = "baseline.two_online_offset_cpu"
    SLOT_NAMES = ("linear", "kernels", "quantization", "topology")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_OFFSET_LINEAR,
        kernels: KernelBackend = _DEFAULT_KERNELS,
        quantization: QuantizationScheme | None = None,
        topology: RoleTopology = _DEFAULT_OFFSET_ROLES,
    ) -> None:
        if quantization is not None:
            _slot(
                "quantization", quantization, QuantizationScheme,
                SymmetricPerRow.descriptor.component,
            )
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot(
                    "linear", linear, ProtocolMethod, TwoOnlineOffsetLinear.descriptor.component,
                ),
                "kernels": _slot("kernels", kernels, KernelBackend),
                "topology": _slot(
                    "topology", topology, RoleTopology, TwoOnlineOffsetRoles.descriptor.component,
                ),
                **({"quantization": quantization} if quantization is not None else {}),
            },
        )

    @property
    def linear(self) -> TwoOnlineOffsetLinear:
        return self.components["linear"]

    @property
    def kernels(self) -> KernelBackend:
        return self.components["kernels"]

    @property
    def quantization(self) -> SymmetricPerRow | None:
        return self.components.get("quantization")

    @property
    def topology(self) -> TwoOnlineOffsetRoles:
        return self.components["topology"]


class VerifiedMaskedLinearCpu(_TypedPipeline):
    PROFILE = "research.verified_masked_linear_cpu"
    SLOT_NAMES = (
        "linear", "preparation", "inference", "kernels", "verification", "quantization",
        "topology",
    )
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
        quantization: QuantizationScheme | None = None,
        topology: RoleTopology | None = None,
    ) -> None:
        if quantization is not None:
            _slot(
                "quantization",
                quantization,
                QuantizationScheme,
                SymmetricPerRow.descriptor.component,
            )
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod, "pllm/masked-linear"),
                "preparation": _slot(
                    "preparation", preparation, PreparationProvider, "pllm/model-aware-corrections"
                ),
                "inference": _slot("inference", inference, InferenceRole, "pllm/inference"),
                "kernels": _slot("kernels", kernels, KernelBackend),
                "verification": _slot(
                    "verification", verification, VerificationScheme, "pllm/freivalds-verify/v1"
                ),
                **({"quantization": quantization} if quantization is not None else {}),
                **({"topology": _slot(
                    "topology", topology, RoleTopology, PreparedProviderRoles.descriptor.component,
                )} if topology is not None else {}),
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

    @property
    def quantization(self) -> SymmetricPerRow | None:
        return self.components.get("quantization")

    @property
    def topology(self) -> PreparedProviderRoles | None:
        return self.components.get("topology")


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
class RuntimeComposition:
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
    weight_bits: int = 8
    activation_bits: int = 8
    public_equalization_digest: str | None = None


def resolve_runtime_composition(pipeline: Pipeline) -> RuntimeComposition | None:
    identities = {name: component.component for name, component in pipeline.components.items()}
    quantization = pipeline.components.get("quantization")
    equalization_digest = None
    if quantization is not None:
        if quantization.component == PublicPerChannelEqualized.descriptor.component:
            if (
                set(quantization.params) != {"profile_digest"}
                or type(quantization.params["profile_digest"]) is not str
                or len(quantization.params["profile_digest"]) != 64
                or any(character not in "0123456789abcdef" for character in quantization.params["profile_digest"])
                or identities.get("linear") not in {"pllm/masked-linear", "pllm/cleartext-linear"}
                or "verification" in identities
            ):
                return None
            equalization_digest = quantization.params["profile_digest"]
        elif (
            quantization.component != SymmetricPerRow.descriptor.component
            or set(quantization.params) != {"weight_bits", "activation_bits"}
            or any(type(value) is not int or value not in {4, 8} for value in quantization.params.values())
            or identities.get("linear") not in {
                "pllm/masked-linear", "pllm/cleartext-linear", "pllm/two-online-offset-linear/v1",
            }
        ):
            return None
        del identities["quantization"]
    bits = (
        {
            "weight_bits": quantization.params["weight_bits"],
            "activation_bits": quantization.params["activation_bits"],
        }
        if quantization is not None and equalization_digest is None
        else {}
    )
    kernels = pipeline.components.get("kernels")
    kernels_valid = kernels is not None and set(kernels.params) == {"threads"}
    metal_valid = (
        kernels is not None
        and kernels.component == AppleMetal.descriptor.component
        and set(kernels.params) == {"min_rows"}
        and type(kernels.params["min_rows"]) is int
        and 2 <= kernels.params["min_rows"] <= 256
    )
    kernel_id = identities.get("kernels")
    public_kernel_valid = (kernel_id == "pllm/cpu" and kernels_valid) or metal_valid
    topology = pipeline.components.get("topology")
    if identities == {
        "linear": "pllm/cleartext-linear",
        "kernels": identities.get("kernels"),
        "topology": "pllm/client-only/v1",
    } and public_kernel_valid and all(
        not pipeline.components[slot].params for slot in ("linear", "topology")
    ):
        return RuntimeComposition(
            "client_only", "none", False, "none", "compiled_client_local_v1",
            f"local_clear_w{bits.get('weight_bits', 8)}a{bits.get('activation_bits', 8)}",
            public_equalization_digest=equalization_digest,
            **bits,
        )
    if identities == {
        "linear": "pllm/two-online-offset-linear/v1",
        "kernels": kernel_id,
        "topology": "pllm/two-online-offset-workers/v1",
    } and public_kernel_valid and all(
        not pipeline.components[slot].params for slot in ("linear", "topology")
    ):
        return RuntimeComposition(
            "offset_public", "none", False, "none", "compiled_offset_v1",
            f"two_online_offset_w{bits.get('weight_bits', 8)}a{bits.get('activation_bits', 8)}",
            public_equalization_digest=equalization_digest,
            **bits,
        )
    if topology is not None:
        if (
            topology.component != "pllm/one-online-provider-offline-preparation/v1"
            or topology.params
            or identities.get("linear") != "pllm/masked-linear"
            or identities.get("preparation") != "pllm/model-aware-corrections"
        ):
            return None
        del identities["topology"]
    if identities == {
        "linear": "pllm/masked-linear",
        "preparation": "pllm/model-aware-corrections",
        "inference": "pllm/inference",
        "kernels": kernel_id,
        "verification": "pllm/freivalds-verify/v1",
    } and public_kernel_valid:
        expected = {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": kernel_id,
            "verification": "pllm/freivalds-verify/v1",
        }
        verification = pipeline.components["verification"]
        if (
            identities != expected
            or not public_kernel_valid
            or set(verification.params) != {"target_failure_bits"}
        ):
            return None
        return RuntimeComposition(
            privacy_mode="public",
            proprietary_protocol="guarded",
            requires_preparation=True,
            correlation_mode="bfv",
            client_runtime="masked_transformer_v1",
            privacy_protocol=None,
            verification_component="pllm/freivalds-verify/v1",
            verification_target_failure_bits=verification.params["target_failure_bits"],
            **bits,
        )
    if (
        identities
        == {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": kernel_id,
        }
        and public_kernel_valid
        and not pipeline.components["linear"].params
        and not pipeline.components["preparation"].params
        and not pipeline.components["inference"].params
    ):
        return RuntimeComposition(
            "public", "guarded", True, "bfv", "masked_transformer_v1", None,
            public_equalization_digest=equalization_digest, **bits,
        )
    if (
        identities
        == {
            "linear": "pllm/guarded-linear/v1",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.components["inference"].params
    ):
        params = pipeline.components["linear"].params
        return RuntimeComposition(
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
        identities
        == {
            "linear": "pllm/blinded-linear/v1",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.components["linear"].params
        and not pipeline.components["inference"].params
    ):
        return RuntimeComposition(
            "proprietary",
            "blinded",
            False,
            "bfv",
            "blinded_ole_transformer_v1",
            "blinded_ole_w4a4",
        )
    if (
        identities
        == {
            "linear": "pllm/direct-fhe",
            "inference": "pllm/inference",
            "kernels": "pllm/cpu",
        }
        and kernels_valid
        and not pipeline.components["linear"].params
        and not pipeline.components["inference"].params
    ):
        return RuntimeComposition(
            "proprietary",
            "direct",
            False,
            "bfv",
            "direct_fhe_transformer_v1",
            "direct_bfv_w4a4",
        )
    return None


__all__ = [
    "ClientOnlyCpu",
    "ClientOnlyMetal",
    "TwoOnlineOffsetCpu",
    "DirectFHEProfile",
    "MaskedLinearCpu",
    "VerifiedMaskedLinearCpu",
    "ProprietaryBlinded",
    "ProprietaryGuarded",
]
