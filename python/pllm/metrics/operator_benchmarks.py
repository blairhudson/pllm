"""Immutable reports for bounded, local research operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import json
import statistics
from typing import Any

from pllm.configuration import ComponentRef, ConfigurationError


_CONFIGURATION_SCHEMA = "pllm.operator_benchmark.v1"


@dataclass(frozen=True, slots=True)
class _OperatorBenchmark:
    component: ComponentRef
    lanes: int = 64
    samples: int = 3

    def __post_init__(self) -> None:
        for name, maximum in (("lanes", 256), ("samples", 16)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ConfigurationError(f"{name} must be an integer in [1, {maximum}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _CONFIGURATION_SCHEMA,
            "component": self.component.to_spec(),
            "lanes": self.lanes,
            "samples": self.samples,
        }

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()


def _sdk_digest(runner: str) -> str:
    digest = hashlib.sha256()
    sources = (
        ("pllm.nonlinear", "__init__.py"),
        ("pllm.metrics", "operator_benchmarks.py"),
        ("pllm.metrics", runner),
    )
    if runner == "activations.py":
        sources += (("pllm.protocols", "__init__.py"), ("pllm.components", "_q7_activation.py"))
    if runner in ("logrow_product.py", "compact_product.py"):
        sources += (("pllm.metrics", "wavelet_gated.py"), ("pllm.metrics", "activations.py"))
    if runner == "secret_replication.py":
        sources += (("pllm.correlation", "__init__.py"),)
    for package, name in sources:
        contents = files(package).joinpath(name).read_bytes()
        digest.update(f"{package}/{name}".encode() + b"\0")
        digest.update(len(contents).to_bytes(8, "little"))
        digest.update(contents)
    return digest.hexdigest()


def benchmark_operator(
    component: ComponentRef,
    *,
    lanes: int = 64,
    samples: int = 3,
    inputs: Mapping[str, Sequence[int]] | None = None,
) -> OperatorBenchmarkReport:
    """Measure a supported operator component with fresh one-use material.

    Method parameters belong to the component; lanes and samples describe this
    benchmark. Omit inputs for public boundary fixtures, or supply ``gate_q24``
    and ``up_q24`` for masked-wire multiplication, ``values`` for rescaling,
    ``gate_q18`` and ``up_q18`` for complete Haar, LogRow or Compact gated blocks, or
    ``values_q7`` for the bounded Compact, Curl and LogRow activations.
    PCF scalar/vector multiplication accepts ``scalar`` (one value) and ``vector``.
    Input values and their fingerprints are never stored in the report.
    """
    from pllm.components import create_component
    from pllm.correlation import SecretReplicationPcf
    from pllm.nonlinear import (
        CompactPiecewiseActivation, CompactGatedMultiplyQ12, CurlWaveletEncodedTable, CurlWaveletGatedMultiplyQ9,
        LogRowGatedMultiplyQ9, MaskedWireGatedMultiplyQ12, SharedRescale,
    )
    from pllm.protocols import LogRowGarbledLookup
    from .activations import _ActivationBenchmark
    from .masked_wire import _MaskedWireBenchmark
    from .shared_rescale import _SharedRescaleBenchmark
    from .wavelet_gated import _WaveletGatedBenchmark
    from .logrow_product import _LogRowProductBenchmark
    from .compact_product import _CompactProductBenchmark
    from .secret_replication import _SecretReplicationBenchmark

    if not isinstance(component, ComponentRef):
        raise TypeError("component must be a ComponentRef")
    component = create_component(component.component, component.params)
    if isinstance(component, SecretReplicationPcf):
        benchmark = _SecretReplicationBenchmark(component, lanes, samples)
        fields = {"scalar", "vector"}
    elif isinstance(component, MaskedWireGatedMultiplyQ12):
        if component.get_params()["dispatch"] != "serial":
            raise ConfigurationError("overlap dispatch requires a compiled two-worker benchmark")
        benchmark = _MaskedWireBenchmark(component, lanes, samples)
        fields = {"gate_q24", "up_q24"}
    elif isinstance(component, CurlWaveletGatedMultiplyQ9):
        benchmark = _WaveletGatedBenchmark(component, lanes, samples)
        fields = {"gate_q18", "up_q18"}
    elif isinstance(component, LogRowGatedMultiplyQ9):
        benchmark = _LogRowProductBenchmark(component, lanes, samples)
        fields = {"gate_q18", "up_q18"}
    elif isinstance(component, CompactGatedMultiplyQ12):
        benchmark = _CompactProductBenchmark(component, lanes, samples)
        fields = {"gate_q18", "up_q18"}
    elif isinstance(component, SharedRescale):
        benchmark = _SharedRescaleBenchmark(component, lanes, samples)
        fields = {"values"}
    elif isinstance(component, (CompactPiecewiseActivation, CurlWaveletEncodedTable, LogRowGarbledLookup)):
        benchmark = _ActivationBenchmark(component, lanes, samples)
        fields = {"values_q7"}
    else:
        raise ConfigurationError(f"no operator benchmark executor for {component.component}")
    if inputs is not None and (
        not isinstance(inputs, Mapping)
        or set(inputs) != fields
        or any(value is None for value in inputs.values())
    ):
        raise ConfigurationError(f"inputs must contain exactly {sorted(fields)}")
    return benchmark.run(**({} if inputs is None else inputs))


def replay_operator_benchmark(
    configuration: Mapping[str, Any],
    *,
    inputs: Mapping[str, Sequence[int]] | None = None,
) -> OperatorBenchmarkReport:
    """Replay a report's configuration; caller-owned inputs must be supplied again.

    Historical probe-v1 records are migrated to the component configuration.
    Their new configuration/SDK digests identify that migration; retained
    measurements and their original identities are not rewritten.
    """
    from pllm.components import create_component
    from pllm.nonlinear import MaskedWireGatedMultiplyQ12, SharedRescale
    from .masked_wire import _PROFILE
    from .shared_rescale import _NUMERIC

    if not isinstance(configuration, Mapping):
        raise ConfigurationError("operator benchmark configuration must be a mapping")
    schema = configuration.get("schema")
    if schema == _CONFIGURATION_SCHEMA:
        if set(configuration) != {"schema", "component", "lanes", "samples"}:
            raise ConfigurationError("unsupported operator benchmark configuration fields")
        spec = configuration["component"]
        if not isinstance(spec, Mapping) or set(spec) != {"component", "params"}:
            raise ConfigurationError("invalid operator benchmark component specification")
        component = create_component(spec["component"], spec["params"])
    elif schema in ("pllm.masked_wire_gated_block_probe.v1", "pllm.shared_rescale_probe.v1"):
        rescale = schema == "pllm.shared_rescale_probe.v1"
        fields = {"bits", "shift", "rounding", "layout", "backend"} if rescale else set()
        if (
            set(configuration) != fields | {"schema", "numeric_profile", "lanes", "samples"}
            or configuration["numeric_profile"] != (_NUMERIC if rescale else _PROFILE)
        ):
            raise ConfigurationError("unsupported historical operator benchmark configuration")
        component = (
            SharedRescale(**{name: configuration[name] for name in fields})
            if rescale else MaskedWireGatedMultiplyQ12()
        )
    else:
        raise ConfigurationError("unsupported operator benchmark configuration schema")
    return benchmark_operator(
        component, lanes=configuration["lanes"], samples=configuration["samples"], inputs=inputs,
    )


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
    public_setup_seconds: float = 0.0
    correlation_expansion_seconds: float = 0.0


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
    role_names: tuple[str, str] = ("party_0", "party_1")
    public_metadata_bytes_estimate: int | None = None
    public_profile_bytes_estimate: int | None = None
    evaluator_ciphertext_bytes: int | None = None
    evaluator_constant_label_bytes: int | None = None
    key_payload_bytes_by_operation: tuple[tuple[str, int, int], ...] | None = None
    expanded_correlation_bytes_per_party: tuple[int, int] | None = None
    prf_field_evaluations_per_party: tuple[int, int] | None = None

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
    base_profile_exact_outputs: int | None = None
    maximum_encoded_profile_error: int | None = None


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
        """Return an independent configuration for ``replay_operator_benchmark``."""
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
