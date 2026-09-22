from pathlib import Path
from typing import Any, Iterable, Mapping, Self

from pllm.providers import ProviderDescriptor

class ConfigurationError(ValueError): ...

class ComponentDescriptor:
    component: str
    provider: str
    distribution: str
    version: str
    category: str
    category_version: str
    lifecycle_phase: str
    parameter_schema: Mapping[str, Any]
    capabilities: tuple[str, ...]
    required_host_features: tuple[str, ...]
    role_eligibility: tuple[str, ...]
    artifacts: tuple[Mapping[str, Any], ...]
    evidence: tuple[Mapping[str, Any], ...]
    def to_dict(self) -> dict[str, Any]: ...

class Model:
    source: str
    kind: str
    model_id: str | None
    revision: str | None
    local_files_only: bool
    endpoint: str | None
    def __init__(
        self,
        source: str,
        *,
        kind: str = "huggingface",
        model_id: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        endpoint: str | None = None,
    ) -> None: ...
    @classmethod
    def hf(
        cls,
        repo_id: str,
        *,
        model_id: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
    ) -> Model: ...
    @classmethod
    def path(
        cls,
        path: str,
        *,
        format: str = "huggingface",
        model_id: str | None = None,
    ) -> Model: ...
    @classmethod
    def tiny(cls, name: str = "qwen2", *, model_id: str | None = None) -> Model: ...
    @classmethod
    def ollama(
        cls,
        name: str,
        *,
        endpoint: str = "http://127.0.0.1:11434",
        model_id: str | None = None,
    ) -> Model: ...
    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> Model: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...
    def canonical_bytes(self) -> bytes: ...
    def digest(self) -> str: ...
    def to_runtime_spec(self) -> dict[str, Any]: ...

class ComponentRef:
    component: str
    params: Mapping[str, Any]
    def __init__(self, component: str, params: Mapping[str, Any] | None = None) -> None: ...
    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> ComponentRef: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...

from pllm.kernels import Cpu as Cpu
from pllm.nonlinear import ArithmeticGarblingSiluQ7 as ArithmeticGarblingSiluQ7
from pllm.schedulers import (
    BoundedIndependentElementsProtectedTensorSchedule as BoundedIndependentElementsProtectedTensorSchedule,
)
from pllm.nonlinear import BinaryTableGatedMultiplyQ7 as BinaryTableGatedMultiplyQ7
from pllm.nonlinear import R03CrtGatedMultiplyQ7 as R03CrtGatedMultiplyQ7
from pllm.passes import KvCacheEviction as KvCacheEviction
from pllm.preparation import ModelAwareCorrections as ModelAwareCorrections
from pllm.protocols import MaskedLinear as MaskedLinear
from pllm.schedulers import (
    ChunkedIndependentLanesProtectedTensorSchedule as ChunkedIndependentLanesProtectedTensorSchedule,
)
from pllm.schedulers import (
    IndependentLanesProtectedTensorSchedule as IndependentLanesProtectedTensorSchedule,
)
from pllm.schedulers import ScalarProtectedTensorSchedule as ScalarProtectedTensorSchedule

class Pipeline:
    model: Model
    components: Mapping[str, ComponentRef]
    profile: str | None
    def __init__(
        self,
        model: Model,
        components: Mapping[str, ComponentRef],
        profile: str | None = ...,
    ) -> None: ...
    @classmethod
    def from_profile(
        cls, profile: str, *, model: Model, components: Mapping[str, ComponentRef]
    ) -> Self: ...
    @classmethod
    def from_spec(
        cls, value: Mapping[str, Any], *, providers: Iterable[ProviderDescriptor] = ...
    ) -> Pipeline: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...
    def canonical_bytes(self) -> bytes: ...
    def digest(self) -> str: ...

class Deployment:
    kind: str
    root: str
    def __init__(self, kind: str, root: str) -> None: ...
    @classmethod
    def local(cls, *, root: str) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...

class ExecutionBudget:
    requests: int
    max_input_tokens: int
    max_new_tokens: int
    def __init__(self, requests: int, max_input_tokens: int, max_new_tokens: int) -> None: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...

class Experiment:
    name: str
    pipeline: Pipeline
    deployment: Deployment
    budget: ExecutionBudget
    def __init__(
        self, name: str, pipeline: Pipeline, deployment: Deployment, budget: ExecutionBudget
    ) -> None: ...
    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], *, providers: Iterable[ProviderDescriptor] = ...
    ) -> Self: ...
    @classmethod
    def from_file(cls, path: str | Path, *, providers: Iterable[ProviderDescriptor] = ...) -> Self: ...
    def get_params(self, deep: bool = True) -> dict[str, Any]: ...
    def with_params(self, **changes: object) -> Self: ...
    def to_spec(self) -> dict[str, Any]: ...
    def canonical_bytes(self) -> bytes: ...
    def configuration_digest(self) -> str: ...
    def resolve(self) -> ExperimentProfile: ...

class ExperimentProfile:
    model: str
    canonical_composition: bytes
    composition_digest: str
    configuration_digest: str
    profile: str | None
    privacy_mode: str
    proprietary_protocol: str
    requires_preparation: bool
    client_runtime: str
    privacy_protocol: str | None
    verification_component: str | None
    verification_target_failure_bits: int
    def __init__(self, experiment: Experiment) -> None: ...

def loads_configuration(
    text: str, *, format: str | None = None, providers: Iterable[ProviderDescriptor] = ...
) -> Experiment: ...
def load_configuration(
    path: str | Path, *, providers: Iterable[ProviderDescriptor] = ...
) -> Experiment: ...
def canonical_bytes(
    configuration: Experiment | Mapping[str, Any], *, providers: Iterable[ProviderDescriptor] = ...
) -> bytes: ...
def configuration_digest(
    configuration: Experiment | Mapping[str, Any], *, providers: Iterable[ProviderDescriptor] = ...
) -> str: ...
