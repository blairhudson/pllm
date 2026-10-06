from typing import Any

from pllm.configuration import Pipeline
from pllm.kernels import KernelBackend
from pllm.preparation import PreparationProvider, PreparedInventory
from pllm.quantization import QuantizationScheme, SymmetricPerRow
from pllm.protocols import (
    BlindedLinear,
    CleartextLinear,
    ClientBundleTransport,
    DirectFHE,
    GuardedLinear,
    ProtocolMethod,
    TwoOnlineOffsetLinear,
)
from pllm.roles import ClientPlacement, ClientOnlyRoles, InferenceRole, OutputHeadAtInference, RoleTopology, PreparedProviderRoles, TwoOnlineOffsetRoles
from pllm.sources import ModelSource
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule
from pllm.tokenization import IndexedTokenizer
from pllm.verification import FreivaldsVerify, VerificationScheme

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
    prefix_cache_bytes: int
    prefix_cache_bound_tokens: int | None
    remote_output_head: bool
    client_prefix_layers: int
    client_linear_roles: tuple[str, ...]
    inventory_policy: str
    prepared_inventory_rows: int
    background_inventory_refill: bool
    inventory_refill: str | None
    bundle_compression: str
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
        boundary: OutputHeadAtInference | None = ...,
        placement: ClientPlacement | None = ...,
        inventory: PreparedInventory | None = ...,
        delivery: ClientBundleTransport | None = ...,
        tokenizer: IndexedTokenizer | None = ...,
        public_prefix: PublicPrefixCapsule | None = ...,
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
    def boundary(self) -> OutputHeadAtInference | None: ...
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
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> MaskedLinearCpu: ...

class ClientOnlyCpu(Pipeline):
    PROFILE: str
    def __init__(self, model: ModelSource, *, linear: ProtocolMethod = ...,
                 kernels: KernelBackend = ..., quantization: QuantizationScheme | None = ...,
                 topology: RoleTopology = ...) -> None: ...
    @property
    def linear(self) -> CleartextLinear: ...
    @property
    def kernels(self) -> KernelBackend: ...
    @property
    def quantization(self) -> QuantizationScheme | None: ...
    @property
    def topology(self) -> ClientOnlyRoles: ...
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
