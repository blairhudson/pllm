"""Typed built-in pipeline profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pllm.configuration import ConfigurationError, Model, Pipeline
from pllm.kernels import AppleMetal, Cpu, KernelBackend
from pllm.preparation import ModelAwareCorrections, PreparationProvider, PreparedInventory
from pllm.quantization import PublicPerChannelEqualized, QuantizationScheme, SymmetricPerRow
from pllm.protocols import (
    BlindedLinear,
    ClientBundleTransport,
    CleartextLinear,
    DirectFHE as DirectFHEMethod,
    GuardedLinear,
    MaskedLinear,
    ProtocolMethod,
    TwoOnlineOffsetLinear,
)
from pllm.roles import (
    ClientOnlyRoles,
    ClientPlacement,
    ClientLinearRoles,
    ClientPrefixLayers,
    Inference,
    InferenceRole,
    OutputHeadAtInference,
    PreparedProviderRoles,
    RoleTopology,
    TwoOnlineOffsetRoles,
)
from pllm.sources import ModelSource
from pllm.state import ClientPrefixReuse
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
    SLOT_NAMES = (
        "linear",
        "preparation",
        "inference",
        "kernels",
        "quantization",
        "cache",
        "boundary",
        "placement",
        "inventory",
        "delivery",
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
        quantization: QuantizationScheme | None = None,
        cache: ClientPrefixReuse | None = None,
        boundary: OutputHeadAtInference | None = None,
        placement: ClientPlacement | None = None,
        inventory: PreparedInventory | None = None,
        delivery: ClientBundleTransport | None = None,
    ) -> None:
        if quantization is not None:
            _slot("quantization", quantization, QuantizationScheme)
            if quantization.component not in {
                SymmetricPerRow.descriptor.component,
                PublicPerChannelEqualized.descriptor.component,
            }:
                raise ConfigurationError(
                    "masked linear requires a supported quantization component"
                )
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot("linear", linear, ProtocolMethod),
                "preparation": _slot("preparation", preparation, PreparationProvider),
                "inference": _slot("inference", inference, InferenceRole),
                "kernels": _slot("kernels", kernels, KernelBackend),
                **({"quantization": quantization} if quantization is not None else {}),
                **(
                    {"cache": _slot("cache", cache, ClientPrefixReuse)} if cache is not None else {}
                ),
                **(
                    {"boundary": _slot("boundary", boundary, OutputHeadAtInference)}
                    if boundary is not None
                    else {}
                ),
                **(
                    {"placement": _slot("placement", placement, ClientPlacement)}
                    if placement is not None
                    else {}
                ),
                **(
                    {"inventory": _slot("inventory", inventory, PreparedInventory)}
                    if inventory is not None
                    else {}
                ),
                **(
                    {"delivery": _slot("delivery", delivery, ClientBundleTransport)}
                    if delivery is not None
                    else {}
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
    def quantization(self) -> QuantizationScheme | None:
        return self.components.get("quantization")

    @property
    def cache(self) -> ClientPrefixReuse | None:
        return self.components.get("cache")

    @property
    def boundary(self) -> OutputHeadAtInference | None:
        return self.components.get("boundary")

    @property
    def placement(self) -> ClientPlacement | None:
        return self.components.get("placement")

    @property
    def inventory(self) -> PreparedInventory | None:
        return self.components.get("inventory")

    @property
    def delivery(self) -> ClientBundleTransport | None:
        return self.components.get("delivery")


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
                    "linear",
                    linear,
                    ProtocolMethod,
                    CleartextLinear.descriptor.component,
                ),
                "kernels": _slot("kernels", kernels, KernelBackend),
                "topology": _slot(
                    "topology",
                    topology,
                    RoleTopology,
                    ClientOnlyRoles.descriptor.component,
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
            model,
            linear=linear,
            kernels=kernels,
            quantization=quantization,
            topology=topology,
        )


class TwoOnlineOffsetCpu(_TypedPipeline):
    """Bounded two-public-worker additive offset comparator."""

    PROFILE = "baseline.two_online_offset_cpu"
    SLOT_NAMES = ("linear", "kernels", "quantization", "topology", "cache", "delivery")
    __slots__ = ()

    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = _DEFAULT_OFFSET_LINEAR,
        kernels: KernelBackend = _DEFAULT_KERNELS,
        quantization: QuantizationScheme | None = None,
        topology: RoleTopology = _DEFAULT_OFFSET_ROLES,
        cache: ClientPrefixReuse | None = None,
        delivery: ClientBundleTransport | None = None,
    ) -> None:
        if quantization is not None:
            _slot(
                "quantization",
                quantization,
                QuantizationScheme,
                SymmetricPerRow.descriptor.component,
            )
        if cache is not None:
            _slot("cache", cache, ClientPrefixReuse, ClientPrefixReuse.descriptor.component)
        if delivery is not None:
            _slot("delivery", delivery, ClientBundleTransport, ClientBundleTransport.descriptor.component)
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                "linear": _slot(
                    "linear",
                    linear,
                    ProtocolMethod,
                    TwoOnlineOffsetLinear.descriptor.component,
                ),
                "kernels": _slot("kernels", kernels, KernelBackend),
                "topology": _slot(
                    "topology",
                    topology,
                    RoleTopology,
                    TwoOnlineOffsetRoles.descriptor.component,
                ),
                **({"quantization": quantization} if quantization is not None else {}),
                **({"cache": cache} if cache is not None else {}),
                **({"delivery": delivery} if delivery is not None else {}),
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

    @property
    def cache(self) -> ClientPrefixReuse | None:
        return self.components.get("cache")

    @property
    def delivery(self) -> ClientBundleTransport | None:
        return self.components.get("delivery")


class VerifiedMaskedLinearCpu(_TypedPipeline):
    PROFILE = "research.verified_masked_linear_cpu"
    SLOT_NAMES = (
        "linear",
        "preparation",
        "inference",
        "kernels",
        "verification",
        "quantization",
        "topology",
        "cache", "placement", "inventory", "delivery", "boundary",
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
        cache: ClientPrefixReuse | None = None,
        placement: ClientPlacement | None = None,
        inventory: PreparedInventory | None = None,
        delivery: ClientBundleTransport | None = None,
        boundary: OutputHeadAtInference | None = None,
    ) -> None:
        if quantization is not None:
            _slot(
                "quantization",
                quantization,
                QuantizationScheme,
                SymmetricPerRow.descriptor.component,
            )
        baseline = MaskedLinearCpu(model, linear=linear, preparation=preparation, inference=inference,
            kernels=kernels, quantization=quantization, cache=cache,
            placement=placement, inventory=inventory, delivery=delivery, boundary=boundary)
        super().__init__(
            profile=self.PROFILE,
            model=_model(model),
            components={
                **baseline.components,
                **({"topology": _slot("topology", topology, RoleTopology,
                    PreparedProviderRoles.descriptor.component)} if topology is not None else {}),
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

    @property
    def quantization(self) -> SymmetricPerRow | None:
        return self.components.get("quantization")

    @property
    def topology(self) -> PreparedProviderRoles | None:
        return self.components.get("topology")

    @property
    def cache(self) -> ClientPrefixReuse | None:
        return self.components.get("cache")

    @property
    def placement(self) -> ClientPlacement | None:
        return self.components.get("placement")

    @property
    def inventory(self) -> PreparedInventory | None:
        return self.components.get("inventory")

    @property
    def delivery(self) -> ClientBundleTransport | None:
        return self.components.get("delivery")

    @property
    def boundary(self) -> OutputHeadAtInference | None:
        return self.components.get("boundary")


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
    prefix_cache_bytes: int = 0
    prefix_cache_bound_tokens: int | None = None
    remote_output_head: bool = False
    client_prefix_layers: int = 0
    client_linear_roles: tuple[str, ...] = ()
    inventory_policy: str = "prewarm"
    prepared_inventory_rows: int = 64
    background_inventory_refill: bool = True
    inventory_refill: str | None = None
    bundle_compression: str = "none"
    causal_reduction: str | None = None
    prepared_output_encoding: str = "raw"


def resolve_runtime_composition(pipeline: Pipeline) -> RuntimeComposition | None:
    identities = {name: component.component for name, component in pipeline.components.items()}
    transport_options: dict[str, Any] = {}
    linear = pipeline.components.get("linear")
    if linear is not None and linear.component == "pllm/masked-linear":
        if (set(linear.params) - {"output_encoding", "prefill_chunk_rows", "request_encoding", "prefill_pruning"}
            or type(linear.params.get("prefill_pruning", "none")) is not str
            or linear.params.get("prefill_pruning", "none") not in {"none", "terminal"}
            or type(linear.params.get("request_encoding", "raw")) is not str
            or linear.params.get("request_encoding", "raw") not in {"raw", "compact", "stage_packed"}
            or (linear.params.get("request_encoding") == "stage_packed"
                and linear.params.get("output_encoding") != "row_residues")
            or type(linear.params.get("output_encoding", "raw")) is not str
            or linear.params.get("output_encoding", "raw") not in {"raw", "row_residues"}
            or type(linear.params.get("prefill_chunk_rows", 0)) is not int
            or linear.params.get("prefill_chunk_rows", 0) not in {0, 4, 8, 16, 32}):
            return None
        transport_options["prepared_output_encoding"] = linear.params.get("output_encoding", "raw")
    inventory = pipeline.components.get("inventory")
    delivery = pipeline.components.get("delivery")
    if inventory is not None:
        if (
            inventory.component != PreparedInventory.descriptor.component
            or not {"policy", "rows"} <= set(inventory.params)
            or set(inventory.params) - {"policy", "rows", "refill", "stage_window"}
            or type(inventory.params.get("stage_window", 1)) is not int
            or not 1 <= inventory.params.get("stage_window", 1) <= 4
            or inventory.params.get("refill", "idle") not in {"idle", "on-demand"}
            or inventory.params["policy"] not in {"prewarm", "request-sized"}
            or type(inventory.params["rows"]) is not int
            or not 1 <= inventory.params["rows"] <= 4096
            or identities.get("linear") != "pllm/masked-linear"
        ):
            return None
        transport_options.update(
            inventory_policy=inventory.params["policy"],
            prepared_inventory_rows=inventory.params["rows"],
            background_inventory_refill=inventory.params.get("refill", "idle") == "idle",
            inventory_refill=inventory.params.get("refill"),
        )
        del identities["inventory"]
    if delivery is not None:
        if (
            delivery.component != ClientBundleTransport.descriptor.component
            or "encoding" not in delivery.params
            or set(delivery.params) - {"encoding", "compression", "batch_objects", "storage"}
            or ("storage" in delivery.params and (
                delivery.params["storage"] != "paged" or delivery.params["encoding"] != "artifacts"
                or "placement" in identities and identities.get("kernels") == "pllm/apple-metal-int8/v1"))
            or delivery.params["encoding"] not in {"none", "zlib", "artifacts"}
            or ("compression" in delivery.params and (
                delivery.params["encoding"] != "artifacts" or delivery.params["compression"] != "zlib"))
            or ("batch_objects" in delivery.params and (
                delivery.params["encoding"] != "artifacts"
                or type(delivery.params["batch_objects"]) is not int
                or not 1 <= delivery.params["batch_objects"] <= 64))
            or identities.get("linear") not in {"pllm/masked-linear", "pllm/two-online-offset-linear/v1"}
        ):
            return None
        transport_options["bundle_compression"] = (
            "artifacts-zlib" if "compression" in delivery.params else delivery.params["encoding"]
        )
        del identities["delivery"]
    cache = pipeline.components.get("cache")
    cache_options: dict[str, int | None] = {}
    if cache is not None:
        if (
            cache.component != ClientPrefixReuse.descriptor.component
            or set(cache.params) not in ({"max_bytes", "fixed_input_tokens"},
                {"max_bytes", "fixed_input_tokens", "generated_prefixes"})
            or ("generated_prefixes" in cache.params and cache.params["generated_prefixes"] is not True)
            or (cache.params.get("generated_prefixes") and (
                pipeline.components.get("quantization") is None or
                pipeline.components["quantization"].params.get("causal_reduction") != "prefix_f32"))
            or type(cache.params["max_bytes"]) is not int
            or not 1 <= cache.params["max_bytes"] >> 20 <= 256
            or cache.params["max_bytes"] % (1 << 20)
            or type(cache.params["fixed_input_tokens"]) is not int
            or not 2 <= cache.params["fixed_input_tokens"] <= 4096
            or identities.get("linear") not in {"pllm/masked-linear", "pllm/two-online-offset-linear/v1"}
        ):
            return None
        cache_options = {
            "prefix_cache_bytes": cache.params["max_bytes"],
            "prefix_cache_bound_tokens": cache.params["fixed_input_tokens"],
        }
        del identities["cache"]
    boundary = pipeline.components.get("boundary")
    if boundary is not None:
        if (
            boundary.component != OutputHeadAtInference.descriptor.component
            or boundary.params
            or identities.get("linear") != "pllm/masked-linear"
        ):
            return None
        del identities["boundary"]
    placement = pipeline.components.get("placement")
    placement_options: dict[str, Any] = {}
    if placement is not None:
        prefix_valid = (
            placement.component == ClientPrefixLayers.descriptor.component
            and set(placement.params) == {"layers"}
            and type(placement.params["layers"]) is int
            and 1 <= placement.params["layers"] <= 8
        )
        roles = placement.params.get("roles")
        roles_valid = (
            placement.component == ClientLinearRoles.descriptor.component
            and set(placement.params) in ({"roles"}, {"roles", "prefix_layers"})
            and ("prefix_layers" not in placement.params or (
                type(placement.params["prefix_layers"]) is int and 1 <= placement.params["prefix_layers"] <= 8))
            and isinstance(roles, (list, tuple))
            and 1 <= len(roles) <= 4
            and all(
                type(role) is str
                and role
                in ClientLinearRoles.descriptor.parameter_schema["properties"]["roles"]["items"][
                    "enum"
                ]
                for role in roles
            )
            and tuple(roles) == tuple(sorted(set(roles)))
        )
        if (
            not (prefix_valid or roles_valid)
            or identities.get("linear") != "pllm/masked-linear"
        ):
            return None
        placement_options = (
            {"client_prefix_layers": placement.params["layers"]}
            if prefix_valid
            else {"client_linear_roles": tuple(roles), "client_prefix_layers": placement.params.get("prefix_layers", 0)}
        )
        del identities["placement"]
    quantization = pipeline.components.get("quantization")
    equalization_digest = None
    if quantization is not None:
        if quantization.component == PublicPerChannelEqualized.descriptor.component:
            if (
                set(quantization.params) != {"profile_digest"}
                or type(quantization.params["profile_digest"]) is not str
                or len(quantization.params["profile_digest"]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in quantization.params["profile_digest"]
                )
                or identities.get("linear") not in {"pllm/masked-linear", "pllm/cleartext-linear"}
                or "verification" in identities
            ):
                return None
            equalization_digest = quantization.params["profile_digest"]
        elif (
            quantization.component != SymmetricPerRow.descriptor.component
            or set(quantization.params) not in ({"weight_bits", "activation_bits"},
                {"weight_bits", "activation_bits", "causal_reduction"})
            or ("causal_reduction" in quantization.params
                and quantization.params["causal_reduction"] != "prefix_f32")
            or any(
                type(value) is not int or value not in {4, 8}
                for value in (quantization.params["weight_bits"], quantization.params["activation_bits"])
            )
            or identities.get("linear")
            not in {
                "pllm/masked-linear",
                "pllm/cleartext-linear",
                "pllm/two-online-offset-linear/v1",
            }
        ):
            return None
        del identities["quantization"]
    bits = (
        {
            "weight_bits": quantization.params["weight_bits"],
            "activation_bits": quantization.params["activation_bits"],
            "causal_reduction": quantization.params.get("causal_reduction"),
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
    if (
        identities
        == {
            "linear": "pllm/cleartext-linear",
            "kernels": identities.get("kernels"),
            "topology": "pllm/client-only/v1",
        }
        and public_kernel_valid
        and all(not pipeline.components[slot].params for slot in ("linear", "topology"))
    ):
        return RuntimeComposition(
            "client_only",
            "none",
            False,
            "none",
            "compiled_client_local_v1",
            f"local_clear_w{bits.get('weight_bits', 8)}a{bits.get('activation_bits', 8)}",
            public_equalization_digest=equalization_digest,
            **bits,
        )
    if (
        identities
        == {
            "linear": "pllm/two-online-offset-linear/v1",
            "kernels": kernel_id,
            "topology": "pllm/two-online-offset-workers/v1",
        }
        and public_kernel_valid
        and not pipeline.components["topology"].params
        and set(pipeline.components["linear"].params) <= {"input_encoding", "output_encoding", "dispatch"}
        and type(pipeline.components["linear"].params.get("input_encoding", "raw")) is str
        and pipeline.components["linear"].params.get("input_encoding", "raw") in {"raw", "seeded"}
        and type(pipeline.components["linear"].params.get("output_encoding", "raw")) is str
        and pipeline.components["linear"].params.get("output_encoding", "raw") in {"raw", "row_residues"}
        and pipeline.components["linear"].params.get("dispatch", "sequential") in {"sequential", "seed_first"}
        and (pipeline.components["linear"].params.get("dispatch") != "seed_first"
             or pipeline.components["linear"].params.get("input_encoding") == "seeded")
    ):
        return RuntimeComposition(
            "offset_public",
            "none",
            False,
            "none",
            "compiled_offset_v1",
            f"two_online_offset_w{bits.get('weight_bits', 8)}a{bits.get('activation_bits', 8)}",
            public_equalization_digest=equalization_digest,
            **bits,
            **cache_options,
            **transport_options,
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
    if (
        identities
        == {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": kernel_id,
            "verification": "pllm/freivalds-verify/v1",
        }
        and public_kernel_valid
    ):
        expected = {
            "linear": "pllm/masked-linear",
            "preparation": "pllm/model-aware-corrections",
            "inference": "pllm/inference",
            "kernels": kernel_id,
            "verification": "pllm/freivalds-verify/v1",
        }
        verification = pipeline.components["verification"]
        target_bits = verification.params.get("target_failure_bits")
        extra_bits = FreivaldsVerify._CACHE_LINEAGE_BITS if cache is not None else 0
        if (
            identities != expected
            or not public_kernel_valid
            or set(verification.params) != {"target_failure_bits"}
            or type(target_bits) is not int or not 1 <= target_bits <= 80 - extra_bits
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
            verification_target_failure_bits=target_bits + extra_bits,
            **bits,
            **cache_options,
            remote_output_head=boundary is not None,
            **placement_options,
            **transport_options,
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
        and not pipeline.components["preparation"].params
        and not pipeline.components["inference"].params
    ):
        return RuntimeComposition(
            "public",
            "guarded",
            True,
            "bfv",
            "masked_transformer_v1",
            None,
            public_equalization_digest=equalization_digest,
            **bits,
            **cache_options,
            remote_output_head=boundary is not None,
            **placement_options,
            **transport_options,
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
