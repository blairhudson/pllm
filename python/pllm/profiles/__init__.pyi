from typing import Any, Literal

from pllm.configuration import Pipeline
from pllm.kernels import Cpu, KernelBackend
from pllm.preparation import PreparationProvider, PreparedInventory
from pllm.quantization import QuantizationScheme, SymmetricPerRow
from pllm.protocols import (
    BlindedLinear,
    CleartextLinear,
    ClientBundleTransport,
    DirectFHE,
    GuardedLinear,
    OpenWeightVerifiableOutsourcing,
    PumaThreePartyDecoder,
    FuseFssSourceDecoder,
    ProtocolMethod,
    TwoOnlineOffsetLinear,
)
from pllm.roles import BoundaryPlacement, ClientPlacement, ClientOnlyRoles, InferenceRole, OutputHeadAtInference, RoleTopology, PreparedProviderRoles, TwoOnlineOffsetRoles, TwoOnlineDelegationRoles
from pllm.sources import ModelSource
from pllm.state import ClientLocalKv, ClientPrefixReuse, PublicPrefixCapsule
from pllm.state import ReplicatedPrivatePrefix
from pllm.roles import ThreePartyReplicatedRoles
from pllm.roles import TrustedDealerTwoShareRoles
from pllm.state import FixedCountPrivatePrefix, StateProtocol
from pllm.schedulers import MlpPrefillTiling
from pllm.tokenization import IndexedTokenizer
from pllm.verification import FreivaldsVerify, PublishedLinearVerify, VerificationScheme

class RuntimeComposition:
    privacy_mode: str
    proprietary_protocol: str
    requires_preparation: bool
    correlation_mode: str
    client_runtime: str
    privacy_protocol: str | None
    guard_max_rows_per_request: int
    guard_max_rows_per_owner_stage: int
    guard_max_requests_per_minute: int
    output_dither_bound: int
    verification_component: str | None
    verification_target_failure_bits: int
    weight_bits: int
    activation_bits: int
    causal_reduction: str | None
    prepared_output_encoding: str
    preparation_storage: str
    preparation_kernel: str
    preparation_min_rows: int
    prefix_cache_bytes: int
    prefix_cache_bound_tokens: int | None
    remote_output_head: bool
    private_boundary: bool
    client_prefix_layers: int
    client_linear_roles: tuple[str, ...]
    client_linear_layer_roles: tuple[tuple[int, str], ...]
    inventory_policy: str
    prepared_inventory_rows: int
    background_inventory_refill: bool
    inventory_refill: str | None
    preparation_decode_epoch_tokens: int | None
    preparation_decode_epoch_schedule: tuple[int, ...] | None
    decode: bool
    decode_max_bytes: int
    inflight: bool
    inflight_max_requests: int
    inflight_max_bytes: int
    bundle_compression: str
    provider_artifact_cache_bytes: int
    runtime_kind: str | None
    numeric_profile: str | None
    local_numeric_profile: str | None
    sharing: str | None
    generation_policy: str | None
    verification_window: str | None
    setup_owner: str | None
    def __init__(
        self,
        privacy_mode: str,
        proprietary_protocol: str,
        requires_preparation: bool,
        correlation_mode: str,
        client_runtime: str,
        privacy_protocol: str | None,
        guard_max_rows_per_request: int = ...,
        guard_max_rows_per_owner_stage: int = ...,
        guard_max_requests_per_minute: int = ...,
        output_dither_bound: int = ...,
        verification_component: str | None = ...,
        verification_target_failure_bits: int = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        causal_reduction: str | None = ...,
        prepared_output_encoding: str = ...,
        private_boundary: bool = ...,
        runtime_kind: str | None = ...,
        numeric_profile: str | None = ...,
        local_numeric_profile: str | None = ...,
        sharing: str | None = ...,
        generation_policy: str | None = ...,
        verification_window: str | None = ...,
        setup_owner: str | None = ...,
    ) -> None: ...

def resolve_runtime_composition(pipeline: Pipeline) -> RuntimeComposition | None: ...

class MaskedLinearCpu(Pipeline):
    PROFILE: str
    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = ...,
        preparation: PreparationProvider = ...,
        inference: InferenceRole = ...,
        kernels: KernelBackend = ...,
        quantization: QuantizationScheme | None = ...,
        cache: ClientPrefixReuse | None = ...,
        boundary: BoundaryPlacement | None = ...,
        topology: RoleTopology | None = ...,
        placement: ClientPlacement | None = ...,
        inventory: PreparedInventory | None = ...,
        delivery: ClientBundleTransport | None = ...,
        tokenizer: IndexedTokenizer | None = ...,
        public_prefix: PublicPrefixCapsule | None = ...,
        state: ClientLocalKv | None = ...,
        scheduler: MlpPrefillTiling | None = ...,
    ) -> None: ...
    @property
    def linear(self) -> ProtocolMethod: ...
    @property
    def preparation(self) -> PreparationProvider: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def quantization(self) -> SymmetricPerRow | None: ...
    @property
    def cache(self) -> ClientPrefixReuse | None: ...
    @property
    def boundary(self) -> BoundaryPlacement | None: ...
    @property
    def topology(self) -> RoleTopology | None: ...
    @property
    def placement(self) -> ClientPlacement | None: ...
    @property
    def inventory(self) -> PreparedInventory | None: ...
    @property
    def delivery(self) -> ClientBundleTransport | None: ...
    @property
    def tokenizer(self) -> IndexedTokenizer | None: ...
    @property
    def public_prefix(self) -> PublicPrefixCapsule | None: ...
    @property
    def state(self) -> ClientLocalKv | None: ...
    @property
    def scheduler(self) -> MlpPrefillTiling | None: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> MaskedLinearCpu: ...

class ClientOnlyCpu(Pipeline):
    PROFILE: str
    def __init__(self, model: ModelSource, *, linear: ProtocolMethod = ...,
                 kernels: KernelBackend = ..., quantization: QuantizationScheme | None = ...,
                 topology: RoleTopology = ..., state: ClientLocalKv | None = ...,
                 scheduler: MlpPrefillTiling | None = ...) -> None: ...
    @property
    def linear(self) -> CleartextLinear: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def quantization(self) -> QuantizationScheme | None: ...
    @property
    def topology(self) -> ClientOnlyRoles: ...
    @property
    def state(self) -> ClientLocalKv | None: ...
    def with_params(self, **changes: object) -> ClientOnlyCpu: ...

class ClientOnlyMetal(ClientOnlyCpu):
    def with_params(self, **changes: object) -> ClientOnlyMetal: ...

class TwoOnlineOffsetCpu(Pipeline):
    PROFILE: str
    def __init__(self, model: ModelSource, *, linear: ProtocolMethod = ...,
                 kernels: KernelBackend = ..., quantization: QuantizationScheme | None = ...,
                 topology: RoleTopology = ..., cache: ClientPrefixReuse | None = ...,
                 delivery: ClientBundleTransport | None = ...) -> None: ...
    @property
    def linear(self) -> TwoOnlineOffsetLinear: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def quantization(self) -> SymmetricPerRow | None: ...
    @property
    def topology(self) -> TwoOnlineOffsetRoles: ...
    @property
    def cache(self) -> ClientPrefixReuse | None: ...
    @property
    def delivery(self) -> ClientBundleTransport | None: ...
    def with_params(self, **changes: object) -> TwoOnlineOffsetCpu: ...

class FuseFssSourceCpu(Pipeline):
    PROFILE: str
    SLOT_NAMES: tuple[str, ...]
    def __init__(self, model: ModelSource, *, linear: ProtocolMethod | None = ...,
                 numeric_profile: Literal["fusefss.gpt2-complete-ring50-q12.v1", "fusefss.dense-rope-ring50-q12.v1", "fusefss.dense-rope-checked-ranges-ring50-q12.v1"] | None = ...,
                 capacity: Literal["capsule", "decoder"] = "capsule",
                 material_delivery: Literal["single", "batch-v1"] = "single",
                 online_schedule: Literal["serial", "wavefront-v1"] = "serial",
                 topology: RoleTopology = ..., state: StateProtocol = ..., kernels: KernelBackend = ...) -> None: ...
    @property
    def linear(self) -> FuseFssSourceDecoder: ...
    @property
    def topology(self) -> TrustedDealerTwoShareRoles: ...
    @property
    def state(self) -> FixedCountPrivatePrefix: ...
    @property
    def kernels(self) -> Cpu: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> FuseFssSourceCpu: ...

class PumaThreePartyCpu(Pipeline):
    PROFILE: str
    def __init__(self, model: ModelSource, *, linear: ProtocolMethod | None = ...,
                  topology: RoleTopology = ..., state: ReplicatedPrivatePrefix = ...,
                  kernels: KernelBackend = ..., weights: Literal["private", "public"] = "private",
                  range_policy: Literal["static", "checked"] = "static", max_abs_activation: int = 256,
                  lookup: Literal["per_token", "tiled", "trie"] = "per_token",
                  selection: Literal["expanded", "tiled"] = "expanded",
                  peer_transport: Literal["http", "streams"] = "http") -> None: ...
    @property
    def linear(self) -> PumaThreePartyDecoder: ...
    @property
    def topology(self) -> ThreePartyReplicatedRoles: ...
    @property
    def state(self) -> ReplicatedPrivatePrefix: ...
    @property
    def kernels(self) -> KernelBackend: ...
    def with_params(self, **changes: object) -> PumaThreePartyCpu: ...

class OpenWeightVerifiableCpu(Pipeline):
    PROFILE: str
    def __init__(self, model: ModelSource, *,
                 linear: ProtocolMethod = ..., kernels: KernelBackend = ...,
                 verification: VerificationScheme = ..., topology: RoleTopology = ...,
                 state: ClientLocalKv = ...) -> None: ...
    @property
    def linear(self) -> OpenWeightVerifiableOutsourcing: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def verification(self) -> PublishedLinearVerify: ...
    @property
    def topology(self) -> TwoOnlineDelegationRoles: ...
    @property
    def state(self) -> ClientLocalKv: ...
    def with_params(self, **changes: object) -> OpenWeightVerifiableCpu: ...

class VerifiedMaskedLinearCpu(Pipeline):
    PROFILE: str
    def __init__(
        self,
        model: ModelSource,
        *,
        linear: ProtocolMethod = ...,
        preparation: PreparationProvider = ...,
        inference: InferenceRole = ...,
        kernels: KernelBackend = ...,
        verification: VerificationScheme = ...,
        quantization: QuantizationScheme | None = ...,
        topology: RoleTopology | None = ...,
        cache: ClientPrefixReuse | None = ...,
        placement: ClientPlacement | None = ...,
        inventory: PreparedInventory | None = ...,
        delivery: ClientBundleTransport | None = ...,
        boundary: OutputHeadAtInference | None = ...,
    ) -> None: ...
    @property
    def linear(self) -> ProtocolMethod: ...
    @property
    def preparation(self) -> PreparationProvider: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def verification(self) -> FreivaldsVerify: ...
    @property
    def quantization(self) -> SymmetricPerRow | None: ...
    @property
    def topology(self) -> PreparedProviderRoles | None: ...
    @property
    def cache(self) -> ClientPrefixReuse | None: ...
    @property
    def placement(self) -> ClientPlacement | None: ...
    @property
    def inventory(self) -> PreparedInventory | None: ...
    @property
    def delivery(self) -> ClientBundleTransport | None: ...
    @property
    def boundary(self) -> OutputHeadAtInference | None: ...
    def with_params(self, **changes: object) -> VerifiedMaskedLinearCpu: ...

class ProprietaryGuarded(Pipeline):
    PROFILE: str
    def __init__(
        self,
        model: ModelSource,
        *,
        linear: GuardedLinear = ...,
        inference: InferenceRole = ...,
        kernels: KernelBackend = ...,
    ) -> None: ...
    @property
    def linear(self) -> GuardedLinear: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> ProprietaryGuarded: ...

class ProprietaryBlinded(Pipeline):
    PROFILE: str
    def __init__(
        self,
        model: ModelSource,
        *,
        linear: BlindedLinear = ...,
        inference: InferenceRole = ...,
        kernels: KernelBackend = ...,
    ) -> None: ...
    @property
    def linear(self) -> BlindedLinear: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> ProprietaryBlinded: ...

class DirectFHEProfile(Pipeline):
    PROFILE: str
    def __init__(
        self,
        model: ModelSource,
        *,
        linear: DirectFHE = ...,
        inference: InferenceRole = ...,
        kernels: KernelBackend = ...,
    ) -> None: ...
    @property
    def linear(self) -> DirectFHE: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> DirectFHEProfile: ...
