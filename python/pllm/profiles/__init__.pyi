from typing import Any

from pllm.configuration import Pipeline
from pllm.kernels import KernelBackend
from pllm.preparation import PreparationProvider
from pllm.protocols import (
    BlindedLinear,
    DirectFHE,
    GuardedLinear,
    ProtocolMethod,
)
from pllm.roles import InferenceRole
from pllm.sources import ModelSource
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
    ) -> None: ...
    @property
    def linear(self) -> ProtocolMethod: ...
    @property
    def preparation(self) -> PreparationProvider: ...
    @property
    def inference(self) -> InferenceRole: ...
    @property
    def kernels(self) -> KernelBackend: ...
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
