"""Compact, client-local forecasts compiled from matched network measurements."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import statistics

from pllm.deployment.network import (
    NetworkError, PublicRecord, canonical, digest_value, finite, integer,
)


def _digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


@lru_cache(maxsize=1)
def _runtime_build():
    """One-time public build/hardware inspection, outside repeated decisions."""
    import importlib.metadata
    from pllm import _native
    from pllm._benchmark_build import file_digest, hardware_identity

    root = Path(__file__).resolve().parents[1]
    return _digest({
        "sdk": {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(root.rglob("*.py"))},
        "native": file_digest(_native.__file__), "python": platform.python_version(),
        "platform": platform.platform(), "hardware": hardware_identity(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "httpx", "tokenizers", "psutil")},
    })


def runtime_identity():
    return _digest({"build": _runtime_build(), "threads": {
        name: os.getenv(name) for name in ("PLLM_NATIVE_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})


def assignment_context(snapshot, assignment, native_threads):
    """Exclude volatile capacity; retain installed hosts, epochs and link conditions."""
    offers = {row.party_id: row for row in snapshot.offers}
    parties = set(assignment.values())
    return _digest({"network": snapshot.network_id, "roles": assignment,
        "native_threads": native_threads,
        "parties": [{"id": party, "operator": offers[party].operator_id,
            "epoch": offers[party].instance_epoch,
            "descriptor": getattr(offers[party], "descriptor_digest", None),
            "devices": offers[party].devices} for party in sorted(parties)],
        "links": [{key: value for key, value in link.to_spec().items()
                   if key not in {"observed_at_ms", "expires_at_ms"}}
                  for link in snapshot.links
                  if {link.source_party_id, link.target_party_id} <= parties]})


@dataclass(frozen=True, slots=True)
class PerformanceSample(PublicRecord):
    SCHEMA = "pllm.performance_sample.v1"
    pipeline_digest: str
    assignment_context_digest: str
    ttft_ms: float
    decode_ms_per_token: float
    samples: int

    def __post_init__(self):
        for name in ("pipeline_digest", "assignment_context_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        for name in ("ttft_ms", "decode_ms_per_token"):
            finite(getattr(self, name), name, minimum=0.000001)
        integer(self.samples, "samples", minimum=1, maximum=100)

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class PlanningPrior(PublicRecord):
    """Forecasts for one measured workload within an exact compiled envelope.

    The caller trusts this local performance input. It cannot reduce allocation
    admission, establish quality, reserve capacity or authorize execution.
    """
    SCHEMA = "pllm.planning_prior.v1"
    model_plan_digest: str
    source_lock_digest: str
    workload_cohort_digest: str
    runtime_digest: str
    evidence_digest: str
    cohort_digest: str
    observed_at_ms: int
    expires_at_ms: int
    input_tokens: int
    output_tokens: int
    max_output_tokens: int
    warm: bool
    rows: tuple[PerformanceSample, ...]

    def __post_init__(self):
        for name in ("model_plan_digest", "source_lock_digest", "workload_cohort_digest",
                     "runtime_digest", "evidence_digest", "cohort_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        integer(self.observed_at_ms, "observed_at_ms")
        integer(self.expires_at_ms, "expires_at_ms", minimum=self.observed_at_ms + 1,
                maximum=self.observed_at_ms + 30 * 86400_000)
        integer(self.input_tokens, "input_tokens", minimum=1)
        integer(self.output_tokens, "output_tokens", minimum=2)
        integer(self.max_output_tokens, "max_output_tokens", minimum=self.output_tokens)
        if type(self.warm) is not bool:
            raise NetworkError("prior warm state must be Boolean")
        if (type(self.rows) is not tuple or not 1 <= len(self.rows) <= 32
                or any(not isinstance(row, PerformanceSample) for row in self.rows)):
            raise NetworkError("prior requires 1..32 performance samples")
        keys = [(r.pipeline_digest, r.assignment_context_digest) for r in self.rows]
        if len(set(keys)) != len(keys):
            raise NetworkError("duplicate prior assignment")
        object.__setattr__(self, "rows", tuple(sorted(self.rows,
            key=lambda r: (r.pipeline_digest, r.assignment_context_digest))))
        if len(self.canonical_bytes()) > 32 << 10:
            raise NetworkError("planning prior exceeds 32 KiB")

    @classmethod
    def from_spec(cls, value):
        data = cls._fields(value)
        if type(data["rows"]) is not list:
            raise NetworkError("prior rows must be an array")
        data["rows"] = tuple(PerformanceSample.from_spec(row) for row in data["rows"])
        return cls(**data)

    def matches(self, request):
        return (self.model_plan_digest == request.model_plan.digest
            and self.source_lock_digest == request.source_lock_digest
            and self.workload_cohort_digest == request.workload_cohort_digest
            and self.runtime_digest == runtime_identity()
            and self.observed_at_ms <= request.policy.evaluated_at_ms < self.expires_at_ms)

    @classmethod
    def from_network_benchmark(cls, report, *, lifetime_seconds=86400):
        """Compile a checked comparison; no benchmark or model loading occurs here."""
        from pllm.runtime.benchmark_cli import _comparison_key

        integer(lifetime_seconds, "lifetime_seconds", minimum=1, maximum=30 * 86400)
        if (report.get("scope") != "selected-network-feasible-controls-diagnostic"
                or report.get("checks", {}).get("passed") is not True):
            raise NetworkError("prior requires a passed matched network comparison")
        candidates = report.get("candidates", [])
        if not 1 <= len(candidates) <= 32:
            raise NetworkError("prior comparison requires 1..32 candidates")
        keys, outputs, inputs, rows, times = [], [], [], [], []
        for candidate in candidates:
            inner = candidate["report"]
            if inner["checks"].get("passed") is not True:
                raise NetworkError("prior candidate checks failed")
            key = _comparison_key(inner)
            if key is None:
                raise NetworkError("prior workload is not comparable")
            keys.append(key)
            samples, ttft, decode = [], [], []
            if not 1 <= len(inner["runs"]) <= 100:
                raise NetworkError("prior requires 1..100 measured responses per candidate")
            for run in inner["runs"]:
                if (run["status"] != "completed" or run["tokens"].get("authoritative") is not True
                        or run["tokens"]["output_tokens"] < 2
                        or run["network_audit"].get("lease_closed") is not True):
                    raise NetworkError("prior requires completed multi-token responses and retired leases")
                output = run.get("generation", {}).get("output_text_digest")
                digest_value(output, "output commitment", nonzero=True)
                outputs.append(output)
                binding = run["network_audit"].get("performance_binding")
                if not isinstance(binding, dict) or binding.get("measurement_schema") != "pllm.network_performance.v1":
                    raise NetworkError("historical report lacks performance context")
                samples.append(binding)
                first = run["network_audit"].get("request_ttft_seconds")
                finite(first, "admission-through-first-output seconds", minimum=0.000001)
                ttft.append(1000 * first)
                rate = run["durations"]["tokens_per_second"]
                finite(rate, "decode rate", minimum=0.000001)
                decode.append(1000 / rate)
                times.append(run["timestamps"]["finished_at_ns"] // 1_000_000)
            if any(row != samples[0] for row in samples):
                raise NetworkError("performance context changed within benchmark")
            binding = samples[0]
            inputs.append({key: binding[key] for key in ("model_plan_digest", "source_lock_digest",
                "workload_cohort_digest", "runtime_digest")})
            rows.append(PerformanceSample(binding["pipeline_digest"], binding["assignment_context_digest"],
                statistics.median(ttft), statistics.median(decode), len(samples)))
        if len(set(keys)) != 1 or len(set(outputs)) != 1 or any(row != inputs[0] for row in inputs):
            raise NetworkError("prior source, numeric, output, workload or runtime cohorts differ")
        key = keys[0]
        return cls(**inputs[0], evidence_digest=_digest(report), cohort_digest=_digest(key),
            observed_at_ms=max(times), expires_at_ms=max(times) + lifetime_seconds * 1000,
            input_tokens=key[1], output_tokens=key[2], max_output_tokens=key[3], warm=key[4], rows=tuple(rows))


def performance_binding(result):
    """Public-only execution identity recorded by the ordinary network benchmark."""
    assignment = {row["role_id"]: row["party_id"] for row in result.native_placement["roles"]}
    return {"model_plan_digest": result.request.model_plan.digest,
        "measurement_schema": "pllm.network_performance.v1",
        "source_lock_digest": result.request.source_lock_digest,
        "workload_cohort_digest": result.request.workload_cohort_digest,
        "pipeline_digest": result.experiment.pipeline.digest(),
        "runtime_digest": runtime_identity(),
        "assignment_context_digest": assignment_context(result.snapshot, assignment,
            result.request.policy.max_client_native_threads)}
