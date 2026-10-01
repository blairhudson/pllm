from typing import Any

from pllm.configuration import Pipeline
from pllm.kernels import KernelBackend
from pllm.preparation import PreparationProvider, PreparedInventory
from pllm.quantization import QuantizationScheme, SymmetricPerRow
from pllm.protocols import (
    BlindedLinear,
    ClientBundleTransport,
    DirectFHE,
    GuardedLinear,
    ProtocolMethod,
)
from pllm.roles import ClientPlacement, InferenceRole, OutputHeadAtInference
from pllm.sources import ModelSource
from pllm.state import ClientPrefixReuse
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
    prefix_cache_bytes: int
    prefix_cache_bound_tokens: int | None
    remote_output_head: bool
    client_prefix_layers: int
    client_linear_roles: tuple[str, ...]
    inventory_policy: str
    prepared_inventory_rows: int
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
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> MaskedLinearCpu: ...

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
