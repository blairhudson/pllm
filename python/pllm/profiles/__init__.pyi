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
