from collections.abc import Sequence
from typing import Any
import numpy as np
from pllm.configuration import ComponentDescriptor, ComponentRef, Pipeline
from pllm.modeling import ModelPlan

class Metric(ComponentRef): ...

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
