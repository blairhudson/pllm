"""Immutable reports for bounded, local research operators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import statistics
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatorCitation:
    """Primary source and the exact component adapted by an operator."""

    paper_id: str
    title: str
    url: str
    source_sha256: str
    scope: str


@dataclass(frozen=True, slots=True)
class OperatorBenchmarkSample:
    """One fresh issuance; wall timings cover local roles, not network latency."""

    issuance_seconds: float
    online_seconds: float
    clear_seconds: float
    native_call_process_cpu_seconds: float
    peer_frame_bytes_by_phase: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class OperatorBenchmarkResources:
    """Per-invocation material, measured peer frames and bounded allocation estimate."""

    raw_key_payload_bytes_per_party: tuple[int, int]
    raw_key_payload_bits_per_party: tuple[int, int]
    peer_frame_bytes_by_direction: tuple[int, int]
    peer_payload_bits: int
    peer_rounds: int
    public_table_bytes_per_party: int
    protocol_allocation_estimate_bytes: int
    client_mask_payload_estimate_bytes: int
    client_entry_exit_payload_estimate_bytes: int

    @property
    def combined_raw_key_payload_bytes(self) -> int:
        return sum(self.raw_key_payload_bytes_per_party)

    @property
    def peer_frame_bytes(self) -> int:
        return sum(self.peer_frame_bytes_by_direction)


@dataclass(frozen=True, slots=True)
class OperatorBenchmarkQuality:
    """Exact declared-integer parity and separate approximation error on supplied inputs."""

    checked_outputs: int
    exact_integer_outputs: int
    float_reference: str
    maximum_absolute_error: float
    mean_absolute_error: float


@dataclass(frozen=True, slots=True)
class OperatorBenchmarkReport:
    """Replayable local measurements with independent paper/SDK/decoder status.

    Reports intentionally omit input/output values, their hashes and secret
    material. Caller-supplied data must be supplied again when replaying.
    """

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
    def configuration(self) -> dict[str, Any]:
        """Return an independent configuration for the probe's ``from_dict``."""
        return json.loads(self.configuration_json)

    @property
    def median_online_seconds(self) -> float:
        return statistics.median(sample.online_seconds for sample in self.samples)

    @property
    def median_issuance_seconds(self) -> float:
        return statistics.median(sample.issuance_seconds for sample in self.samples)

    def to_dict(self) -> dict[str, Any]:
        """Export JSON-compatible, aggregate-only evidence without automatic ranking."""
        result = asdict(self)
        result.pop("configuration_json")
        result.update(
            schema="pllm.operator_benchmark_report.v1",
            configuration=self.configuration,
            environment=dict(self.environment),
            median_online_seconds=self.median_online_seconds,
            median_issuance_seconds=self.median_issuance_seconds,
            scope="bounded in-process operator; fresh one-use material per sample",
            paper_coverage="component-reference",
            sdk_availability="operator-benchmark",
            benchmark_status="completed",
            resources_scope="per-sample",
            whole_model_executable=False,
            independent_role_transport=False,
            privacy_reviewed=False,
            full_wire_bytes=None,
            peak_memory_bytes=None,
            checkpoint_quality=None,
        )
        # Normalize tuples as arrays, including nested frozen records.
        return json.loads(json.dumps(result, allow_nan=False))

    def to_json(self) -> str:
        """Serialize one report; no files or telemetry are written implicitly."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
