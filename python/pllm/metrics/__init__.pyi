from collections.abc import Mapping, Sequence
from typing import Any
from dataclasses import dataclass
import numpy as np
from pllm.configuration import ComponentDescriptor, ComponentRef, Experiment, Model, Pipeline
from pllm.modeling import ModelPlan
from pllm.deployment.wan import WanConditions
from pllm._benchmark_contract import (
    BenchmarkTarget as BenchmarkTarget, BenchmarkCase as BenchmarkCase,
    BenchmarkSuite as BenchmarkSuite, BuildRef as BuildRef, Sampling as Sampling,
    Scenario as Scenario, TargetBudget as TargetBudget,
)
from pllm.metrics.benchmark_runner import (
    BenchmarkRun as BenchmarkRun, benchmark_target as benchmark_target,
    benchmark_suite as benchmark_suite,
)
from pllm.metrics.coverage import (
    BenchmarkRequirement as BenchmarkRequirement,
    benchmark_capabilities as benchmark_capabilities,
    benchmark_coverage as benchmark_coverage,
)

class Metric(ComponentRef): ...

class CpuTime(Metric):
    def __init__(self) -> None: ...

@dataclass(frozen=True)
class OperatorCitation:
    paper_id: str
    title: str
    url: str
    source_sha256: str
    scope: str

@dataclass(frozen=True)
class OperatorBenchmarkSample:
    issuance_seconds: float
    online_seconds: float
    clear_seconds: float
    native_call_process_cpu_seconds: float
    peer_frame_bytes_by_phase: tuple[tuple[int, int], ...]
    public_setup_seconds: float = ...
    correlation_expansion_seconds: float = ...
    input_sharing_seconds: float = ...
    evaluation_seconds: float | None = ...
    verification_seconds: float | None = ...
    homomorphic_operations: tuple[tuple[str, int], ...] | None = ...
    minimum_remaining_depth: int | None = ...

@dataclass(frozen=True)
class OperatorBenchmarkResources:
    raw_key_payload_bytes_per_party: tuple[int, int]
    raw_key_payload_bits_per_party: tuple[int, int]
    peer_frame_bytes_by_direction: tuple[int, int]
    peer_payload_bits: int
    peer_rounds: int
    public_table_bytes_per_party: int
    protocol_allocation_estimate_bytes: int
    client_mask_payload_estimate_bytes: int
    client_entry_exit_payload_estimate_bytes: int
    role_names: tuple[str, str] = ...
    public_metadata_bytes_estimate: int | None = ...
    public_profile_bytes_estimate: int | None = ...
    evaluator_ciphertext_bytes: int | None = ...
    evaluator_constant_label_bytes: int | None = ...
    key_payload_bytes_by_operation: tuple[tuple[str, int, int], ...] | None = ...
    expanded_correlation_bytes_per_party: tuple[int, int] | None = ...
    prf_field_evaluations_per_party: tuple[int, int] | None = ...
    public_table_bytes_by_party: tuple[int, int] | None = ...
    setup_frame_bytes_by_operation: tuple[tuple[str, int, int], ...] | None = ...
    homomorphic_polynomial_degree: int | None = ...
    homomorphic_coefficient_bits: tuple[int, ...] | None = ...
    homomorphic_scale_bits: int | None = ...
    @property
    def combined_raw_key_payload_bytes(self) -> int: ...
    @property
    def peer_frame_bytes(self) -> int: ...

@dataclass(frozen=True)
class OperatorBenchmarkQuality:
    checked_outputs: int
    exact_integer_outputs: int
    float_reference: str
    maximum_absolute_error: float
    mean_absolute_error: float
    base_profile_exact_outputs: int | None = ...
    maximum_encoded_profile_error: int | None = ...
    absolute_tolerance: float | None = ...
    outputs_within_tolerance: int | None = ...

@dataclass(frozen=True)
class OperatorBenchmarkReport:
    method: str
    configuration_json: str
    configuration_digest: str
    numeric_contract_digest: str
    native_implementation_digest: str
    sdk_implementation_digest: str
    input_source: str
    resources: OperatorBenchmarkResources
    quality: OperatorBenchmarkQuality
    samples: tuple[OperatorBenchmarkSample, ...]
    citations: tuple[OperatorCitation, ...]
    environment: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    @property
    def configuration(self) -> dict[str, Any]: ...
    @property
    def median_online_seconds(self) -> float: ...
    @property
    def median_issuance_seconds(self) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...
    def to_json(self) -> str: ...

def benchmark_operator(component: ComponentRef, *, lanes: int = 64, samples: int = 3,
                       inputs: Mapping[str, Sequence[int | float]] | None = None,
                       geometry: Mapping[str, int] | None = None) -> OperatorBenchmarkReport: ...
def replay_operator_benchmark(configuration: Mapping[str, Any], *,
                              inputs: Mapping[str, Sequence[int | float]] | None = None) -> OperatorBenchmarkReport: ...

def communication_per_token(report: Mapping[str, Any]) -> dict[str, Any]: ...
def wan_readiness(report: Mapping[str, Any], conditions: WanConditions | None = None) -> dict[str, Any]: ...
def benchmark_memory(model: Model | Pipeline | Experiment | str, *, max_input_tokens: int | None = None,
                     max_output_tokens: int | None = None, inventory_rows: int | None = None,
                     cache_bytes: int = 0, cache_bound_tokens: int | None = None,
                     backend: str = "native", enforced_wan: bool = False,
                     memory_budget_bytes: int | None = None) -> dict[str, Any]: ...

class ProjectedResharingProbe:
    def run(self, weights: Any = None) -> dict[str, Any]: ...
    def project(self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int) -> dict[str, Any]: ...

class PreparedResidueProbe:
    def __init__(self, rows: int = 8, repetitions: int = 3) -> None: ...
    def run(self, weights: Any = None, *, dense_wire_bits: int | None = None) -> dict[str, Any]: ...

class PreparedDuplexProbe:
    def __init__(self, rows: int = ..., chunk_rows: tuple[int, ...] = ..., repetitions: int = ...,
                 download_mbps: float = ..., upload_mbps: float = ..., latency_ms: float = ...) -> None: ...
    def run(self, weights: Any = None, *, dense_wire_bits: int | None = None) -> dict[str, Any]: ...

class ArtifactPlaneProbe:
    def __init__(self, repetitions: int = 3) -> None: ...
    def run(self, payload: bytes | None = None) -> dict[str, Any]: ...

class ArtifactEntropyProbe:
    def __init__(self, repetitions: int = 3) -> None: ...
    def run(self, payload: bytes | None = None) -> dict[str, Any]: ...

class BatchedPrivateLookupProbe:
    def __init__(self, records: int = 256, record_bytes: int = 64, batch_size: int = 8) -> None: ...
    def run(self, table: bytes | None = None) -> dict[str, Any]: ...

class OrthogonalActivationProbe:
    def __init__(self, activation_bits: int = 6, block: int = 128, decode_steps: int = 3) -> None: ...
    def run(self, compiled: Any, remote: Any, weights: Mapping[str, Any], token_cohort: Sequence[Sequence[int]]) -> dict[str, Any]: ...

class GeneratedStateReuseProbe:
    steps: int
    def __init__(self, steps: int = 4) -> None: ...
    def run(self, compiled: Any, remote: Any, token_ids: Sequence[int]) -> dict[str, Any]: ...

class StateCompatibilityProbe:
    def compare(self, left: Any, right: Any) -> dict[str, Any]: ...

class MaskedAggregationProbe:
    repetitions: int
    def __init__(self, repetitions: int = 3) -> None: ...
    def run(self, share_a: Any, share_b: Any, widths: Any) -> dict[str, Any]: ...

class TokenLocalProjectionProbe:
    cache_bytes: int
    decode_steps: int
    def __init__(self, cache_bytes: int = 1 << 20, decode_steps: int = 4) -> None: ...
    def run(self, compiled: Any, remote: Any, weights: Mapping[str, Any], token_cohort: Sequence[Sequence[int]]) -> dict[str, Any]: ...

class ProgressiveHeadProbe:
    prefix_bits: int
    measure_private_lookup: bool
    def __init__(self, prefix_bits: int = 5, measure_private_lookup: bool = True) -> None: ...
    def run(self, weights: Any, scales: Any, queries: Any, activation_scales: Any) -> dict[str, Any]: ...

class PrivatePageLookupProbe:
    records: int
    record_bytes: int
    page_records: int
    queries: int
    def __init__(self, records: int = 64, record_bytes: int = 32, page_records: int = 1, queries: int = 3) -> None: ...
    def run(self, table: bytes | None = None) -> dict[str, Any]: ...

class PrivateHeadRetrievalProbe:
    """Bounded public-index/private-row greedy quality and client-cost diagnostic."""
    def __init__(self, rank: int = 8, candidate_counts: tuple[int, ...] = (1, 8, 32), fit_rows: int = 2048, page_records: int = 8) -> None: ...
    def run(self, *, weights: Any = None, scales: Any = None, inputs: Any = None, input_scales: Any = None, public_calibration: Any = None) -> dict[str, Any]: ...

class Latency(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, statistic: str = "median", phase: str = "online") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Throughput(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, basis: str = "tokens") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Communication(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, direction: str = "total", phase: str = "online") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Memory(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, kind: str = "peak_rss") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Energy(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, source: str = "not_available") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Accuracy(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, dataset: str, measure: str = "task_score") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class Perplexity(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, dataset: str) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ReferenceAgreement(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, dataset_digest: str, reference_checkpoint_digest: str, top_k: int = 5) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

def measure_reference_agreement(
    candidate_logits: Sequence[float] | np.ndarray,
    reference_logits: Sequence[float] | np.ndarray,
    *,
    top_k: int = 5,
) -> dict[str, float]: ...

class Cost(Metric):
    descriptor: ComponentDescriptor
    def __init__(self, *, currency: str = "USD", basis: str = "request") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ResidentMlpCostProbe:
    fixed_scale_bits: int
    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int
    def __init__(
        self, fixed_scale_bits: int, maximum_material_bytes_per_party: int,
        maximum_online_all_link_body_bytes: int,
    ) -> None: ...
    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]: ...

class ResidentFusedGateCostProbe:
    domain_bits: int
    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int
    maximum_online_body_bytes_per_layer: int
    def __init__(
        self, domain_bits: int, maximum_material_bytes_per_party: int,
        maximum_online_all_link_body_bytes: int,
        maximum_online_body_bytes_per_layer: int,
    ) -> None: ...
    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]: ...

class ResidentQuadraticGateCostProbe:
    maximum_material_bytes_per_party: int
    maximum_online_all_link_body_bytes: int
    maximum_online_body_bytes_per_layer: int
    def __init__(
        self, maximum_material_bytes_per_party: int,
        maximum_online_all_link_body_bytes: int,
        maximum_online_body_bytes_per_layer: int,
    ) -> None: ...
    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]: ...
    def run_two_source_layer_bound(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]: ...

class LatentResponseCostProbe:
    maximum_online_all_link_body_bytes: int
    maximum_total_all_link_body_bytes: int
    maximum_material_bytes_per_party: int
    def __init__(
        self, maximum_online_all_link_body_bytes: int,
        maximum_total_all_link_body_bytes: int,
        maximum_material_bytes_per_party: int,
    ) -> None: ...
    def run(
        self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
    ) -> dict[str, Any]: ...

class EncryptedLinearCostProbe:
    input_width: int
    output_width: int
    rows: int
    modulus: int
    def __init__(
        self, input_width: int, output_width: int, rows: int = 1, modulus: int = 65_537,
    ) -> None: ...
    def run(self) -> dict[str, Any]: ...

class EncryptedQuadraticShareCostProbe:
    width: int
    rows: int
    islands: int
    def __init__(self, width: int = 32, rows: int = 1, islands: int = 1) -> None: ...
    def run(self) -> dict[str, Any]: ...

class ProjectedPolynomialCostProbe:
    mode: str
    ring_bits: int
    hidden: int
    channels: int
    outputs: int
    rows: int
    def __init__(self, mode: str = "seeded", ring_bits: int = 24, hidden: int = 8,
                 channels: int = 32, outputs: int = 8, rows: int = 4) -> None: ...
    def run(self) -> dict[str, Any]: ...
    def project(self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int) -> dict[str, Any]: ...

class TokenNetworkBudgetProbe:
    reduction_factor: int
    ring_bits: int
    def __init__(self, reduction_factor: int = 100, ring_bits: int = 24) -> None: ...
    def project(self, plan: ModelPlan, composition: Pipeline, *, response_new_tokens: int,
                baseline_online_body_bytes: int, baseline_total_body_bytes: int) -> dict[str, Any]: ...
