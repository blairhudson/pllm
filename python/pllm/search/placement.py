"""Bounded deterministic placement search; native code owns execution legality."""

from __future__ import annotations

import itertools
import math
import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

from pllm.configuration import Experiment
from pllm.deployment.network import (
    LivePartyOffer,
    NetworkError,
    NetworkSnapshot,
    PublicRecord,
    canonical,
    digest_value,
    finite,
    identity,
    integer,
    strings,
    strict_load,
)
from pllm.modeling import ModelPlan
from pllm.deployment.resources import ClientResources
from .performance import PlanningPrior, assignment_context

PRIOR_OBJECTIVES = ("forecast_ttft_ms", "forecast_decode_ms_per_token")

OBJECTIVES = frozenset(
    {
        "online_all_link_body_bytes",
        "preprocessing_all_link_body_bytes",
        "total_arithmetic_body_bytes",
        "link_transfer_ms",
        "client_weight_bytes",
        "client_payload_bytes",
        "client_linear_macs",
        "client_cpu_ns",
        "client_peak_memory_bytes",
        "full_wire_bytes",
        "setup_ms",
        "artifact_miss_bytes",
        "horizon_accounted_body_bytes",
        "prefill_dependent_exchanges",
        "decode_dependent_exchanges_per_token",
        "prefill_round_trip_ms",
        "decode_round_trip_ms_per_token",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateCostEvidence(PublicRecord):
    """Declared arithmetic estimate, bound to an exact source/config/schedule/cohort.

    This is not a measurement certificate. Supplied arithmetic quantities must
    replay against native stage geometry; unknown CPU/peak/wire terms stay unknown.
    """

    SCHEMA = "pllm.candidate_cost_evidence.v1"
    configuration_digest: str
    model_plan_digest: str
    schedule_digest: str
    source_lock_digest: str
    workload_cohort_digest: str
    source: str
    online_all_link_body_bytes: int
    preprocessing_all_link_body_bytes: int
    origin: str = "estimate"
    scope: str = "bounded-workload-arithmetic-body"

    def __post_init__(self):
        for name in (
            "configuration_digest",
            "model_plan_digest",
            "schedule_digest",
            "source_lock_digest",
            "workload_cohort_digest",
        ):
            digest_value(getattr(self, name), name)
        identity(self.source, "evidence source")
        integer(self.online_all_link_body_bytes, "online_all_link_body_bytes")
        integer(self.preprocessing_all_link_body_bytes, "preprocessing_all_link_body_bytes")
        if self.origin != "estimate" or self.scope != "bounded-workload-arithmetic-body":
            raise NetworkError("candidate declarations cannot claim measured or full-wire evidence")

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class PlanningPolicy(PublicRecord):
    SCHEMA = "pllm.planning_policy.v1"
    client_party_id: str = "client"
    evaluated_at_ms: int = 0
    max_candidates: int = 64
    max_assignments: int = 4096
    objectives: tuple[str, ...] = ("client_weight_bytes",)
    max_client_weight_bytes: int | None = None
    max_client_payload_bytes: int | None = None
    minimum_remote_mac_fraction: float = 0.0
    remote_mac_denominator: str = "body_linear"
    buffer_reserve_bytes: int = 1 << 20
    max_observation_age_ms: int = 60_000
    max_client_cpu_ns: int | None = None
    max_client_peak_memory_bytes: int | None = None
    max_full_wire_bytes: int | None = None
    require_verified_privacy: bool = False
    allow_incomplete_execution: bool = False
    reuse_horizon: int = 1
    incumbent_configuration_digest: str | None = None
    switch_body_bytes: int = 0
    minimum_switch_improvement_fraction: float = 0.0
    allocation_model: str = "payload"
    candidate_selection: str = "installed"
    max_client_native_threads: int | None = None
    prior_objectives: tuple[str, ...] = PRIOR_OBJECTIVES

    def __post_init__(self) -> None:
        identity(self.client_party_id, "client_party_id")
        integer(self.evaluated_at_ms, "evaluated_at_ms")
        integer(self.max_candidates, "max_candidates", minimum=1, maximum=1024)
        integer(self.max_assignments, "max_assignments", minimum=1, maximum=1_000_000)
        integer(self.buffer_reserve_bytes, "buffer_reserve_bytes")
        if self.allocation_model not in {"payload", "native_shared_v1"}:
            raise NetworkError("unsupported allocation owner model")
        if self.candidate_selection not in {"installed", "pinned"}:
            raise NetworkError("candidate selection must be installed or pinned")
        if self.max_client_native_threads is not None:
            integer(self.max_client_native_threads, "max_client_native_threads", minimum=1, maximum=32)
        integer(self.max_observation_age_ms, "max_observation_age_ms")
        integer(self.reuse_horizon, "reuse_horizon", minimum=1, maximum=1024)
        integer(self.switch_body_bytes, "switch_body_bytes", maximum=1 << 40)
        finite(self.minimum_switch_improvement_fraction, "minimum_switch_improvement_fraction")
        if self.minimum_switch_improvement_fraction > 1:
            raise NetworkError("switch improvement must be a fraction in [0,1]")
        if self.incumbent_configuration_digest is not None:
            digest_value(self.incumbent_configuration_digest, "incumbent configuration")
        elif self.switch_body_bytes or self.minimum_switch_improvement_fraction:
            raise NetworkError("switch costs and hysteresis require an incumbent")
        if self.switch_body_bytes and (not self.objectives or self.objectives[0] != "horizon_accounted_body_bytes"):
            raise NetworkError("body switching costs require a horizon-body primary objective")
        strings(self.objectives, "objectives", choices=OBJECTIVES)
        if self.prior_objectives:
            strings(self.prior_objectives, "prior_objectives", choices=frozenset(PRIOR_OBJECTIVES))
        elif type(self.prior_objectives) is not tuple:
            raise NetworkError("prior_objectives must be a tuple")
        finite(self.minimum_remote_mac_fraction, "minimum_remote_mac_fraction")
        if self.minimum_remote_mac_fraction > 1 or self.remote_mac_denominator not in {
            "body_linear",
            "all_linear",
        }:
            raise NetworkError("remote MAC policy needs an explicit supported denominator/fraction")
        for name in (
            "max_client_weight_bytes",
            "max_client_payload_bytes",
            "max_client_cpu_ns",
            "max_client_peak_memory_bytes",
            "max_full_wire_bytes",
        ):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)
        for name in ("require_verified_privacy", "allow_incomplete_execution"):
            if type(getattr(self, name)) is not bool:
                raise NetworkError(f"{name} must be boolean")

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> PlanningPolicy:
        data = cls._fields({"prior_objectives": list(PRIOR_OBJECTIVES), "max_client_native_threads": None, "candidate_selection": "installed", "allocation_model": "payload", "reuse_horizon": 1, "incumbent_configuration_digest": None,
                           "switch_body_bytes": 0, "minimum_switch_improvement_fraction": 0.0, **value})
        if type(data["objectives"]) is not list:
            raise NetworkError("objectives must be an array")
        data["objectives"] = tuple(data["objectives"])
        if type(data["prior_objectives"]) is not list:
            raise NetworkError("prior_objectives must be an array")
        data["prior_objectives"] = tuple(data["prior_objectives"])
        return cls(**data)

    def to_spec(self):
        value = super(PlanningPolicy, self).to_spec()
        if self.reuse_horizon == 1:
            value.pop("reuse_horizon")
        for name, default in (("incumbent_configuration_digest", None), ("switch_body_bytes", 0),
                               ("minimum_switch_improvement_fraction", 0.0), ("allocation_model", "payload"),
                               ("candidate_selection", "installed"), ("max_client_native_threads", None),
                               ("prior_objectives", PRIOR_OBJECTIVES)):
            if getattr(self, name) == default:
                value.pop(name)
        return value


@dataclass(frozen=True, slots=True)
class ArtifactCostEvidence(PublicRecord):
    """Public client-artifact estimate; never grants execution or claims residency proof.

    Object keys bind source, physical tensor metadata and numeric context through
    the existing authenticated bundle format. Execution rechecks actual cache bytes.
    """

    SCHEMA = "pllm.artifact_cost_evidence.v1"
    configuration_digest: str
    model_plan_digest: str
    source_lock_digest: str
    manifest_digest: str
    bundle_digest: str
    manifest_bytes: int
    objects: tuple[tuple[str, int], ...]
    resident_keys: tuple[str, ...] = ()
    origin: str = "estimate"
    scope: str = "public-client-artifacts"
    object_encoding: str = "identity"
    object_transfer_bytes: tuple[tuple[str, int], ...] = ()

    def __post_init__(self):
        for field in ("configuration_digest", "model_plan_digest", "source_lock_digest",
                      "manifest_digest", "bundle_digest"):
            digest_value(getattr(self, field), field)
        integer(self.manifest_bytes, "manifest_bytes", minimum=1, maximum=16 << 20)
        if type(self.objects) is not tuple or not 1 <= len(self.objects) <= 4096:
            raise NetworkError("artifact objects must be a bounded nonempty tuple")
        seen = set()
        for row in self.objects:
            if type(row) is not tuple or len(row) != 2:
                raise NetworkError("artifact object must contain key and byte length")
            key, size = row
            digest_value(key, "artifact key")
            integer(size, "artifact bytes", minimum=1, maximum=4 << 30)
            if key in seen:
                raise NetworkError("duplicate artifact object")
            seen.add(key)
        if (type(self.resident_keys) is not tuple or len(set(self.resident_keys)) != len(self.resident_keys)
                or not set(self.resident_keys) <= seen):
            raise NetworkError("resident keys must be a unique subset of required public objects")
        if self.origin != "estimate" or self.scope != "public-client-artifacts":
            raise NetworkError("artifact declarations cannot claim measured or protected residency")
        object.__setattr__(self, "objects", tuple(sorted(self.objects)))
        object.__setattr__(self, "resident_keys", tuple(sorted(self.resident_keys)))
        if self.object_encoding not in {"identity", "zlib"}:
            raise NetworkError("unsupported artifact cost encoding")
        if type(self.object_transfer_bytes) is not tuple:
            raise NetworkError("artifact transfer lengths must be immutable")
        transfers = {}
        for row in self.object_transfer_bytes:
            if type(row) is not tuple or len(row) != 2 or row[0] in transfers:
                raise NetworkError("duplicate or malformed artifact transfer length")
            digest_value(row[0], "artifact transfer key")
            integer(row[1], "artifact transfer bytes", minimum=1, maximum=(4 << 30) + (1 << 20))
            transfers[row[0]] = row[1]
        if (self.object_encoding == "identity" and transfers
                or self.object_encoding == "zlib" and set(transfers) != seen):
            raise NetworkError("encoded artifact costs require every raw object exactly once")
        object.__setattr__(self, "object_transfer_bytes", tuple(sorted(transfers.items())))

    @classmethod
    def from_manifest(cls, experiment, model_plan, source_lock_digest, payload, *,
                      bundle_digest, bundle_bytes, resident_keys=(), objects=None):
        """Price metadata and declared residency; compression checks supplied public objects."""
        from pllm.runtime.bundle_artifacts import parse_manifest, verify_object
        import msgpack

        profile = experiment.resolve()
        if profile.bundle_compression not in {"artifacts", "artifacts-zlib"}:
            raise NetworkError("artifact costs require executable artifact bundle delivery")
        delivery = experiment.pipeline.components["delivery"]
        if delivery.params.get("batch_objects", 1) > 1 and profile.bundle_compression == "artifacts-zlib":
            raise NetworkError("coalesced compression needs group-bound costs; per-object transfer lengths cannot price it")
        manifest = parse_manifest(payload, fingerprint=bundle_digest, size=bundle_bytes)
        skeleton = msgpack.unpackb(manifest["skeleton"], raw=False)
        if skeleton["manifest"]["metadata"]["source_lock_digest"] != source_lock_digest:
            raise NetworkError("artifact manifest source mismatch")
        encoding, transfers = "identity", ()
        if profile.bundle_compression == "artifacts-zlib":
            from pllm.runtime.bundle_compression import encode_bundle_frames
            if not isinstance(objects, Mapping) or set(objects) != {row["sha256"] for row in manifest["objects"]}:
                raise NetworkError("compressed artifact costing requires the exact committed public objects")
            sizes = []
            for row in manifest["objects"]:
                raw = objects[row["sha256"]]
                verify_object(row, raw)
                sizes.append((row["sha256"], sum(len(frame) for frame in encode_bundle_frames(raw))))
            encoding, transfers = "zlib", tuple(sizes)
        return cls(experiment.configuration_digest(), model_plan.digest, source_lock_digest,
                    hashlib.sha256(payload).hexdigest(), bundle_digest, len(payload),
                    tuple((row["sha256"], row["size"]) for row in manifest["objects"]), tuple(resident_keys),
                    object_encoding=encoding, object_transfer_bytes=transfers)

    @classmethod
    def from_spec(cls, value):
        encoded = value.get("schema") == "pllm.artifact_cost_evidence.v2"
        if not encoded and ("object_encoding" in value or "object_transfer_bytes" in value):
            raise NetworkError("encoded artifact costs require schema v2")
        data = cls._fields({"object_encoding": "identity", "object_transfer_bytes": [], **value,
                            "schema": cls.SCHEMA if encoded else value.get("schema")})
        if type(data["objects"]) is not list or any(type(row) is not list for row in data["objects"]):
            raise NetworkError("artifact objects must be arrays")
        if type(data["resident_keys"]) is not list:
            raise NetworkError("resident keys must be an array")
        data["objects"] = tuple(tuple(row) for row in data["objects"])
        data["resident_keys"] = tuple(data["resident_keys"])
        if type(data["object_transfer_bytes"]) is not list or any(type(row) is not list for row in data["object_transfer_bytes"]):
            raise NetworkError("artifact transfer lengths must be arrays")
        data["object_transfer_bytes"] = tuple(tuple(row) for row in data["object_transfer_bytes"])
        if encoded and data["object_encoding"] != "zlib":
            raise NetworkError("artifact cost schema v2 requires explicit zlib costs")
        return cls(**data)

    def to_spec(self):
        value = super(ArtifactCostEvidence, self).to_spec()
        value["objects"] = [list(row) for row in self.objects]
        if self.object_encoding == "identity":
            value.pop("object_encoding")
            value.pop("object_transfer_bytes")
        else:
            value["schema"] = "pllm.artifact_cost_evidence.v2"
            value["object_transfer_bytes"] = [list(row) for row in self.object_transfer_bytes]
        return value


@dataclass(frozen=True, slots=True)
class ClientStateCostEvidence(PublicRecord):
    """Client-local reuse assumption, never a cache hit or state-transfer authority.

    Keep this record in client-side planning, outside discovery offers. It contains
    neither tokens nor cache keys. Runtime revalidates actual private cache state.
    """

    SCHEMA = "pllm.client_state_cost_evidence.v1"
    configuration_digest: str
    model_plan_digest: str
    source_lock_digest: str
    client_party_id: str
    reusable_prefill_rows: int
    resident_bytes: int
    state_basis: str = "completed-prefill"
    origin: str = "estimate"
    scope: str = "client-local-prefill-reuse"

    def __post_init__(self):
        for name in ("configuration_digest", "model_plan_digest", "source_lock_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        identity(self.client_party_id, "client_party_id")
        integer(self.reusable_prefill_rows, "reusable_prefill_rows", minimum=1, maximum=4096)
        integer(self.resident_bytes, "resident_bytes", minimum=1, maximum=256 << 20)
        if (self.state_basis != "completed-prefill" or self.origin != "estimate"
                or self.scope != "client-local-prefill-reuse"):
            raise NetworkError("client state costs cannot authorize incremental state or claim measured execution")

    @classmethod
    def from_spec(cls, value):
        return cls(**cls._fields(value))


@dataclass(frozen=True, slots=True)
class PlanningRequest(PublicRecord):
    SCHEMA = "pllm.planning_request.v1"
    model_plan: ModelPlan
    candidates: tuple[Experiment, ...]
    policy: PlanningPolicy
    source_lock_digest: str | None = None
    cost_evidence: tuple[CandidateCostEvidence, ...] = ()
    artifact_evidence: tuple[ArtifactCostEvidence, ...] = ()
    state_evidence: tuple[ClientStateCostEvidence, ...] = ()
    client_resources: ClientResources | None = None
    performance_prior: PlanningPrior | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_plan, ModelPlan) or not isinstance(
            self.policy, PlanningPolicy
        ):
            raise TypeError("request requires ModelPlan and PlanningPolicy")
        if self.performance_prior is not None:
            if not isinstance(self.performance_prior, PlanningPrior):
                raise TypeError("performance_prior must be PlanningPrior")
        if self.client_resources is not None:
            if not isinstance(self.client_resources, ClientResources):
                raise TypeError("client_resources must be ClientResources")
            if self.policy.allocation_model != "native_shared_v1":
                raise NetworkError("client resource admission requires complete native allocation pricing")
            age = self.policy.evaluated_at_ms - self.client_resources.observed_at_ms
            if not 0 <= age <= self.policy.max_observation_age_ms:
                raise NetworkError("stale client resource observation")
        raw_plan = self.model_plan.canonical_bytes()
        if type(raw_plan) is not bytes or canonical(strict_load(raw_plan)) != raw_plan:
            raise NetworkError("ModelPlan must contain strict canonical immutable JSON bytes")
        if type(self.candidates) is not tuple or not 1 <= len(self.candidates) <= 1024:
            raise NetworkError("candidates must be an explicit tuple of 1..1024 Experiments")
        if any(not isinstance(item, Experiment) for item in self.candidates):
            raise TypeError("candidates must contain Experiments")
        if self.source_lock_digest is not None:
            digest_value(self.source_lock_digest, "source_lock")
        first = self.candidates[0]
        precision = first.pipeline.components.get("quantization")
        plan = self.model_plan.to_dict()
        for item in self.candidates:
            if item.pipeline.model != first.pipeline.model:
                raise NetworkError("candidates must use the exact same model source")
            if item.pipeline.components.get("quantization") != precision:
                raise NetworkError("numeric contract must match; silent quality changes forbidden")
            if item.budget != first.budget:
                raise NetworkError("candidate budgets must match exactly")
            if (
                plan["prefill"]["batch"] != 1
                or plan["prefill"]["query_sequence"] != item.budget.max_input_tokens
                or plan["decode"]["maximum_key_sequence"]
                != (item.budget.max_input_tokens + item.budget.max_new_tokens - 1)
            ):
                raise NetworkError("ModelPlan work bounds must match candidate budget exactly")
        unique = {item.configuration_digest(): item for item in self.candidates}
        if (self.policy.incumbent_configuration_digest is not None
                and self.policy.incumbent_configuration_digest not in unique):
            raise NetworkError("incumbent must be an explicit candidate")
        object.__setattr__(self, "candidates", tuple(unique[key] for key in sorted(unique)))
        if type(self.cost_evidence) is not tuple or len(self.cost_evidence) > len(unique):
            raise NetworkError(
                "cost_evidence must be a bounded tuple with at most one record per candidate"
            )
        seen = set()
        for evidence in self.cost_evidence:
            if not isinstance(evidence, CandidateCostEvidence):
                raise TypeError("cost_evidence must contain CandidateCostEvidence")
            if evidence.configuration_digest not in unique or evidence.configuration_digest in seen:
                raise NetworkError("cost evidence requires a unique matching candidate")
            if (
                evidence.model_plan_digest != self.model_plan.digest
                or evidence.source_lock_digest != self.source_lock_digest
                or evidence.workload_cohort_digest != self.workload_cohort_digest
            ):
                raise NetworkError("cost evidence source/plan/workload cohort mismatch")
            seen.add(evidence.configuration_digest)
        object.__setattr__(
            self,
            "cost_evidence",
            tuple(sorted(self.cost_evidence, key=lambda item: item.configuration_digest)),
        )
        if type(self.artifact_evidence) is not tuple or len(self.artifact_evidence) > len(unique):
            raise NetworkError("artifact evidence must be bounded to one per candidate")
        seen = set()
        for evidence in self.artifact_evidence:
            if not isinstance(evidence, ArtifactCostEvidence):
                raise TypeError("artifact_evidence requires ArtifactCostEvidence")
            candidate = unique.get(evidence.configuration_digest)
            if (candidate is None or evidence.configuration_digest in seen
                    or evidence.model_plan_digest != self.model_plan.digest
                    or evidence.source_lock_digest != self.source_lock_digest
                    or candidate.resolve().bundle_compression != (
                        "artifacts-zlib" if evidence.object_encoding == "zlib" else "artifacts")):
                raise NetworkError("artifact evidence source/plan/configuration mismatch")
            if (evidence.object_encoding == "zlib"
                    and candidate.pipeline.components["delivery"].params.get("batch_objects", 1) > 1):
                raise NetworkError("coalesced compression needs group-bound artifact costs")
            seen.add(evidence.configuration_digest)
        object.__setattr__(self, "artifact_evidence", tuple(sorted(
            self.artifact_evidence, key=lambda row: row.configuration_digest)))
        if type(self.state_evidence) is not tuple or len(self.state_evidence) > len(unique):
            raise NetworkError("state evidence must be bounded to one per candidate")
        seen = set()
        for evidence in self.state_evidence:
            if not isinstance(evidence, ClientStateCostEvidence):
                raise TypeError("state_evidence requires ClientStateCostEvidence")
            candidate = unique.get(evidence.configuration_digest)
            if (candidate is None or evidence.configuration_digest in seen
                    or evidence.model_plan_digest != self.model_plan.digest
                    or evidence.source_lock_digest != self.source_lock_digest
                    or evidence.client_party_id != self.policy.client_party_id):
                raise NetworkError("state evidence owner/source/plan/configuration mismatch")
            profile = candidate.resolve()
            if (not profile.prefix_cache_bytes
                    or evidence.resident_bytes > profile.prefix_cache_bytes
                    or evidence.reusable_prefill_rows > candidate.budget.max_input_tokens
                    or profile.client_runtime not in {"masked_transformer_v1", "compiled_offset_v1"}
                    or candidate.pipeline.components.get("quantization") is None
                    or candidate.pipeline.components["quantization"].params.get("causal_reduction") != "prefix_f32"):
                raise NetworkError("state evidence requires bounded canonical public prefill reuse")
            seen.add(evidence.configuration_digest)
        object.__setattr__(self, "state_evidence", tuple(sorted(
            self.state_evidence, key=lambda row: row.configuration_digest)))
        self.canonical_bytes()

    @property
    def workload_cohort_digest(self) -> str:
        """Bounds-only cohort: not a claim about prompts, quality or measured sampling."""
        return hashlib.sha256(
            b"pllm.planning_workload_bounds.v1\0"
            + canonical({"batch": 1, "budget": self.candidates[0].budget.to_spec()})
        ).hexdigest()

    def to_spec(self) -> dict[str, Any]:
        result = {
            "schema": self.SCHEMA,
            "model_plan": self.model_plan.to_dict(),
            "candidates": [item.to_spec() for item in self.candidates],
            "policy": self.policy.to_spec(),
            "source_lock_digest": self.source_lock_digest,
            "cost_evidence": [item.to_spec() for item in self.cost_evidence],
        }
        if self.artifact_evidence:
            result["artifact_evidence"] = [item.to_spec() for item in self.artifact_evidence]
        if self.state_evidence:
            result["state_evidence"] = [item.to_spec() for item in self.state_evidence]
        if self.client_resources is not None:
            result["client_resources"] = self.client_resources.to_spec()
        if self.performance_prior is not None:
            result["performance_prior"] = self.performance_prior.to_spec()
        return result

    @classmethod
    def from_spec(cls, value: Mapping[str, Any]) -> PlanningRequest:
        data = cls._fields({"performance_prior": None,
            "artifact_evidence": [], "state_evidence": [], "client_resources": None, **value})
        if data["performance_prior"] is not None:
            data["performance_prior"] = PlanningPrior.from_spec(data["performance_prior"])
        if data["client_resources"] is not None:
            data["client_resources"] = ClientResources.from_spec(data["client_resources"])
        data["model_plan"] = ModelPlan(canonical(data["model_plan"]))
        if type(data["candidates"]) is not list:
            raise NetworkError("candidates must be an array")
        data["candidates"] = tuple(Experiment.from_spec(item) for item in data["candidates"])
        data["policy"] = PlanningPolicy.from_spec(data["policy"])
        if type(data["cost_evidence"]) is not list:
            raise NetworkError("cost_evidence must be an array")
        data["cost_evidence"] = tuple(
            CandidateCostEvidence.from_spec(item) for item in data["cost_evidence"]
        )
        if type(data["artifact_evidence"]) is not list:
            raise NetworkError("artifact_evidence must be an array")
        data["artifact_evidence"] = tuple(ArtifactCostEvidence.from_spec(item) for item in data["artifact_evidence"])
        if type(data["state_evidence"]) is not list:
            raise NetworkError("state_evidence must be an array")
        data["state_evidence"] = tuple(ClientStateCostEvidence.from_spec(item) for item in data["state_evidence"])
        return cls(**data)


def _geometry(request: PlanningRequest, experiment: Experiment, requirements: dict) -> dict:
    from pllm.runtime.semantic_stages import _scheduled_stage_specs
    from pllm.profiles import resolve_runtime_composition

    plan = request.model_plan
    native_limit = request.policy.max_client_native_threads
    if (native_limit is not None
            and experiment.resolve().client_runtime == "compiled_client_local_v1"
            and experiment.pipeline.components["kernels"].params.get("threads", 1) > native_limit):
        raise NetworkError("client-owned model loading exceeds native worker limit")
    admitted_schedule, stages = _scheduled_stage_specs(plan, experiment.pipeline)
    schedule = admitted_schedule.to_dict()
    steps = [
        item
        for item in schedule["prefill"]["steps"]
        if item["executor"]
        in {
            "remote_stage",
            "verified_remote_stage",
            "client_linear",
        }
    ]
    if len(steps) != len(stages):
        raise NetworkError("schedule stage geometry mismatch")
    profile = experiment.resolve()
    numeric = resolve_runtime_composition(experiment.pipeline)
    if numeric is None:
        raise NetworkError("runtime numeric contract is unavailable")
    role_ids = {item["role_id"] for item in requirements["roles"]}
    client_only = role_ids == {"client"}
    workers = ["worker_a", "worker_b"] if "worker_a" in role_ids else ["inference"]
    rows = experiment.budget.max_input_tokens + experiment.budget.max_new_tokens - 1
    requests = experiment.budget.requests
    state = next((row for row in request.state_evidence
                  if row.configuration_digest == experiment.configuration_digest()), None)
    reused_rows = state.reusable_prefill_rows if state is not None else 0
    edges: Counter[tuple[str, str, str]] = Counter()
    macs: Counter[str] = Counter()
    body_total = all_total = body_remote = all_remote = 0
    workspace = 0
    exchanges: Counter[tuple[str, str]] = Counter()
    tiling = schedule.get("mlp_tiling")
    tiled_steps = ({index for region in tiling["regions"]
                    for index in range(region["first_step"], region["end_step"])} if tiling else set())
    indices = {id(step): index for index, step in enumerate(schedule["prefill"]["steps"])}
    for stage, step in zip(stages, steps, strict=True):
        count = experiment.budget.max_new_tokens if stage.role == "lm_head" else rows
        # Capacity must accommodate a miss and ordinary fallback; reuse is only
        # a cost assumption, never permission to shrink live memory admission.
        workspace = max(workspace, 4 * count * (stage.in_features + stage.out_features))
        if stage.role == "lm_head":
            if reused_rows == experiment.budget.max_input_tokens:
                count -= 1  # Exact completed-prefill hit includes its logits.
        else:
            count -= reused_rows
            if step.get("terminal_row_only") and not reused_rows:
                count -= experiment.budget.max_input_tokens - 1
        local = (
            client_only
            or step["executor"] == "client_linear"
            or stage.role == "token_lookup"
            or (stage.role == "lm_head" and not profile.remote_output_head)
        )
        if stage.role == "token_lookup":
            continue  # Lookup is not a matrix multiplication in the installed runtime.
        work = requests * count * stage.in_features * stage.out_features
        all_total += work
        if stage.role != "lm_head":
            body_total += work
        if local:
            macs["client"] += work
            continue
        parameters = experiment.pipeline.components["linear"].params
        bound = stage.in_features * ((1 << (numeric.weight_bits - 1)) - 1) * ((1 << (numeric.activation_bits - 1)) - 1)
        residue_width = min(32, (2 * bound).bit_length())
        ring_width = next(width for width in (16, 24, 32) if width >= residue_width)
        prepared = "preparation" in role_ids
        input_row_bytes = stage.in_features * (ring_width // 8 if prepared else 4)
        output_row_bytes = stage.out_features * (ring_width // 8 if prepared else 4)
        if parameters.get("output_encoding") == "row_residues":
            # Shape/quantizer maxima are public upper bounds. Checkpoint-bound
            # widths can be smaller; no sampled sparsity or compression is assumed.
            output_row_bytes = (stage.out_features * residue_width + 7) // 8
        if prepared and parameters.get("request_encoding") == "stage_packed":
            input_row_bytes = (stage.in_features * residue_width + 7) // 8
        all_remote += work
        if stage.role != "lm_head":
            body_remote += work
        for worker in workers:
            prefill_calls = int(reused_rows < experiment.budget.max_input_tokens)
            if prefill_calls and indices[id(step)] in tiled_steps:
                prefill_calls = math.ceil(experiment.budget.max_input_tokens / tiling["tile_rows"])
            exchanges[(worker, "prefill")] += prefill_calls
            exchanges[(worker, "decode")] += 1
            macs[worker] += work
            seeded = worker == "worker_b" and parameters.get("input_encoding") == "seeded"
            batches = experiment.budget.max_new_tokens - int(reused_rows == experiment.budget.max_input_tokens)
            edges[("client", worker, "online")] += requests * (
                batches * 32 if seeded else count * input_row_bytes)
            edges[(worker, "client", "online")] += requests * count * output_row_bytes
        if prepared:
            inventory = experiment.pipeline.components.get("inventory")
            demand = inventory is not None and inventory.params.get("allocation") == "demand"
            correction_rows = count if demand else max(profile.prepared_inventory_rows or 64, rows - reused_rows)
            macs["preparation"] += requests * correction_rows * stage.in_features * stage.out_features
            # Actual seeded inventory directions: one seed per stage; corrections
            # travel only Preparation -> Inference. Tickets, framing, projections
            # and authentication are outside this payload estimate.
            edges[("client", "preparation", "preprocessing")] += requests * 32
            edges[("preparation", "inference", "preprocessing")] += (
                requests * correction_rows * output_row_bytes
            )
    retained = schedule.get("retained_state")
    state_bytes = (retained["owned_array_bound_bytes"] if retained is not None else
                   sum(_product(item["shape"]) * 4 for item in schedule["decode"]["state_outputs"]))
    # A cache miss must remain feasible even before any resident-state evidence
    # exists. Charge its full declared capacity, not only today's hit snapshot.
    payload = state_bytes + workspace + profile.prefix_cache_bytes
    weights = {role: 6 * floor for role, floor in requirements["minimum_weight_bytes"].items()}
    memory = {
        role: max(requirements["minimum_memory_bytes"][role], value)
        + request.policy.buffer_reserve_bytes
        + (payload if role == "client" else workspace)
        for role, value in weights.items()
    }
    # Legacy payload/weight ceilings have their own conservative scope. Do not
    # subtract a six-copy weight bound from the shared/paged peak estimate.
    payloads = {role: value - weights[role] - request.policy.buffer_reserve_bytes
                for role, value in memory.items()}
    online = sum(value for (*_, phase), value in edges.items() if phase == "online")
    prep = sum(value for (*_, phase), value in edges.items() if phase == "preprocessing")
    allocation = {}
    if request.policy.allocation_model == "native_shared_v1":
        from pllm.runtime.benchmark_memory import BenchmarkMemoryError, estimate_plan_memory

        try:
            estimate = estimate_plan_memory(plan, experiment.pipeline,
                max_input_tokens=experiment.budget.max_input_tokens,
                max_output_tokens=experiment.budget.max_new_tokens,
                inventory_rows=profile.prepared_inventory_rows or 64)
        except (BenchmarkMemoryError, AttributeError) as exc:
            raise NetworkError("native allocation owners unavailable") from exc
        if estimate["limitations"]:
            raise NetworkError("native allocation owners unsupported: " + "; ".join(estimate["limitations"]))
        memory = {"client": estimate["client_peak_bytes"], **estimate["provider_peak_bytes"]}
        memory = {role: max(value, requirements["minimum_memory_bytes"][role])
                  + request.policy.buffer_reserve_bytes for role, value in memory.items()}
        allocation = {"allocation_model": "native_shared_v1", "role_peak_memory_estimates": memory,
            "client_peak_memory_bytes": memory["client"],
            "memory_scope": "shared native allocation owners plus 25% margin and explicit reserve; estimate, not measured RSS"}
    denominator = request.policy.remote_mac_denominator
    remote = body_remote if denominator == "body_linear" else all_remote
    total = body_total if denominator == "body_linear" else all_total
    return {
        "scope": "shape-bounded arithmetic and seed payload; excludes tickets/control, verifier material, public delivery, full wire and CPU",
        "origin": "estimate",
        "cost_model": "pllm.schedule_arithmetic_cost.v5",
        "online_all_link_body_bytes": online,
        "preprocessing_all_link_body_bytes": prep,
        "total_arithmetic_body_bytes": online + prep,
        "client_weight_bytes": weights["client"],
        "client_payload_bytes": payload,
        "client_linear_macs": macs["client"],
        "remote_mac_fraction": remote / total if total else 0,
        "remote_mac_denominator": denominator,
        "logical_remote_linear_macs": remote,
        "logical_total_linear_macs": total,
        "role_linear_macs": dict(sorted(macs.items())),
        "prefill_dependent_exchanges": sum(value for (_, phase), value in exchanges.items() if phase == "prefill"),
        "decode_dependent_exchanges_per_token": sum(value for (_, phase), value in exchanges.items() if phase == "decode"),
        "role_exchanges": [{"worker_role": worker, "phase": phase, "count": count}
                           for (worker, phase), count in sorted(exchanges.items())],
        "role_memory_estimates": memory,
        "role_weight_estimates": weights,
        "role_payload_estimates": payloads,
        "memory_scope": "6 bytes per owned element plus explicit buffers; not complete peak RAM",
        "client_cpu_ns": None,
        "client_peak_memory_bytes": None,
        "full_wire_bytes": None,
        "setup_ms": None,
        "artifact_miss_bytes": None,
        "horizon_accounted_body_bytes": None,
        **({"client_native_thread_limit": native_limit,
            "client_worker_scope": "per client-owned native executor; excludes Python, BLAS, tokenizer and GPU workers"}
           if native_limit is not None else {}),
        **({"client_state_cost_evidence_digest": state.digest,
            "client_state_scope": "assumed completed-prefill reuse; revalidated by client; no state migration",
            "reused_prefill_rows_per_request": reused_rows,
            "resident_client_state_bytes": state.resident_bytes} if state is not None else {}),
        "role_links": [
            {"source_role": a, "target_role": b, "phase": phase, "bytes": value}
            for (a, b, phase), value in sorted(edges.items())
        ],
        **allocation,
    }


def _product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def _costs(
    base: dict, assignment: dict[str, str], snapshot: NetworkSnapshot, policy: PlanningPolicy
) -> dict:
    costs = {**base}
    client_roles = [role for role, party in assignment.items() if party == policy.client_party_id]
    costs["client_weight_bytes"] = sum(base["role_weight_estimates"][role] for role in client_roles)
    costs["client_payload_bytes"] = sum(base["role_payload_estimates"][role] for role in client_roles)
    costs["client_linear_macs"] = sum(
        base["role_linear_macs"].get(role, 0) for role in client_roles
    )
    if "role_peak_memory_estimates" in base:
        costs["client_peak_memory_bytes"] = sum(base["role_peak_memory_estimates"][role] for role in client_roles)
    if "client_incremental_memory_bytes" in base:
        costs["client_incremental_memory_bytes"] = base["client_request_allocation_bytes"] + sum(
            base["role_peak_memory_estimates"][role] for role in client_roles if role != "client")
    # Logical work is offloaded only if no online provider runs at the client party.
    # Two-offset duplicates are not extra denominator credit.
    if any(role in client_roles for role in ("inference", "worker_a", "worker_b")):
        costs["logical_remote_linear_macs"] = 0
        costs["remote_mac_fraction"] = 0.0
    observations = {(item.source_party_id, item.target_party_id): item for item in snapshot.links}
    transfer_terms = []
    unknown_transfer = False
    distribution = Counter()
    for edge in base["role_links"]:
        left, right = assignment[edge["source_role"]], assignment[edge["target_role"]]
        distribution[(left, right, edge["phase"])] += edge["bytes"]
        if left == right:
            continue
        observation = observations.get((left, right))
        if (
            observation is None
            or observation.expires_at_ms <= policy.evaluated_at_ms
            or observation.observed_at_ms > policy.evaluated_at_ms
            or policy.evaluated_at_ms - observation.observed_at_ms > policy.max_observation_age_ms
        ):
            unknown_transfer = True
            continue
        transfer_terms.append(
            1000
            * edge["bytes"]
            / observation.bytes_per_second
            * (1 + observation.uncertainty_fraction)
        )
    costs["link_transfer_ms"] = None if unknown_transfer else math.fsum(transfer_terms)
    costs["link_transfer_scope"] = (
        "serialized arithmetic arrays / directed bandwidth; excludes RTT and overlap"
    )
    # Latency terms remain separate from bandwidth and compute. A duplex stage
    # contributes one dependent exchange; its pipelined frames are not serial RTTs.
    # Offset workers are conservatively serialized: this is not seed-first timing.
    for phase, metric in (("prefill", "prefill_round_trip_ms"), ("decode", "decode_round_trip_ms_per_token")):
        terms, unknown = [], False
        for edge in base["role_exchanges"]:
            if edge["phase"] != phase or not edge["count"]:
                continue
            left, right = assignment["client"], assignment[edge["worker_role"]]
            if left == right:
                continue
            for pair in ((left, right), (right, left)):
                observation = observations.get(pair)
                if (observation is None or observation.expires_at_ms <= policy.evaluated_at_ms
                        or not 0 <= policy.evaluated_at_ms - observation.observed_at_ms <= policy.max_observation_age_ms):
                    unknown = True
                else:
                    terms.append(edge["count"] * observation.latency_ms * (1 + observation.uncertainty_fraction))
        costs[metric] = None if unknown else math.fsum(terms)
    costs["round_trip_scope"] = "per-request dependent stage exchanges times declared RTT; excludes bandwidth, compute, preparation, control and overlap"
    costs["party_links"] = [
        {"source_party": left, "target_party": right, "phase": phase, "bytes": value}
        for (left, right, phase), value in sorted(distribution.items())
    ]
    return costs


def _policy_rejection(costs: dict, policy: PlanningPolicy) -> str | None:
    if policy.require_verified_privacy:
        return "UNRESOLVED_VERIFIED_PRIVACY"
    if ("client_incremental_memory_bytes" in costs
            and costs["client_incremental_memory_bytes"] > costs["client_allocation_headroom_bytes"]):
        return "INSUFFICIENT_CLIENT_HEADROOM"
    for bound, metric in (
        ("max_client_weight_bytes", "client_weight_bytes"),
        ("max_client_payload_bytes", "client_payload_bytes"),
        ("max_client_cpu_ns", "client_cpu_ns"),
        ("max_client_peak_memory_bytes", "client_peak_memory_bytes"),
        ("max_full_wire_bytes", "full_wire_bytes"),
    ):
        limit = getattr(policy, bound)
        if limit is not None:
            if costs[metric] is None:
                return f"UNKNOWN_REQUIRED_COST:{metric}"
            if costs[metric] > limit:
                return f"POLICY_BOUND:{metric}"
    if costs["remote_mac_fraction"] < policy.minimum_remote_mac_fraction:
        return "POLICY_BOUND:remote_mac_fraction"
    if any(costs[name] is None for name in policy.objectives):
        return "UNKNOWN_REQUIRED_COST:objective"
    for name in policy.objectives:
        finite(costs[name], name)
    return None


def search_placements(request: PlanningRequest, snapshot: NetworkSnapshot) -> Any:
    from ._planning_cache import _cache
    from pllm.plan import PlanningResult

    if not isinstance(request, PlanningRequest) or not isinstance(snapshot, NetworkSnapshot):
        raise TypeError("plan requires PlanningRequest and NetworkSnapshot")
    policy = request.policy
    reasons: Counter[tuple[str, str]] = Counter()
    alternatives = {}
    winner = None
    best_key: tuple[Any, ...] | None = None
    assignments = candidates = feasible = 0
    exhaustive = True
    incumbent = None
    incumbent_key: tuple[Any, ...] | None = None
    prior = request.performance_prior
    prior_index = {(row.pipeline_digest, row.assignment_context_digest): row for row in prior.rows
                   } if (prior is not None and prior.matches(request) and policy.prior_objectives
                         and not policy.switch_body_bytes
                         and all(0 <= policy.evaluated_at_ms - link.observed_at_ms <= policy.max_observation_age_ms
                                 and policy.evaluated_at_ms < link.expires_at_ms
                                 for link in snapshot.links)) else {}
    prior_complete = bool(prior_index)
    prior_winner = prior_incumbent = prior_key = prior_incumbent_key = None
    prior_alternatives = {}
    incumbent_components = next(
        (row.pipeline.components for row in request.candidates
         if row.configuration_digest() == policy.incumbent_configuration_digest), None
    )
    snapshot_stale = (
        snapshot.expires_at_ms <= policy.evaluated_at_ms
        or snapshot.observed_at_ms > policy.evaluated_at_ms
        or policy.evaluated_at_ms - snapshot.observed_at_ms > policy.max_observation_age_ms
    )
    for index, experiment in enumerate(request.candidates):
        if index >= policy.max_candidates:
            exhaustive = False
            break
        candidates += 1
        digest = experiment.configuration_digest()
        try:
            admitted, requirements, base = _cache.get(request, experiment)
            if request.client_resources is not None:
                resources = request.client_resources
                memory = base["role_memory_estimates"]
                incremental = memory["client"]
                if incremental > resources.headroom_bytes:
                    raise NetworkError("insufficient client host memory headroom")
                memory["client"] += resources.process_memory_bytes + resources.reserved_memory_bytes
                base["role_peak_memory_estimates"] = dict(memory)
                base.update(client_peak_memory_bytes=memory["client"],
                    client_incremental_memory_bytes=incremental,
                    client_request_allocation_bytes=incremental,
                    client_allocation_headroom_bytes=resources.headroom_bytes,
                    client_resource_observation_digest=resources.digest,
                    memory_scope="native owner estimate plus current process RSS, concurrent reservations and reserve; conservative, not an RSS guarantee")
            for evidence in request.artifact_evidence:
                if evidence.configuration_digest != digest:
                    continue
                raw_missing = sum(size for key, size in evidence.objects if key not in evidence.resident_keys)
                sizes = evidence.object_transfer_bytes or evidence.objects
                missing = sum(size for key, size in sizes if key not in evidence.resident_keys)
                horizon = policy.reuse_horizon
                base.update(artifact_miss_bytes=missing,
                    artifact_raw_miss_bytes=raw_missing,
                    artifact_object_encoding=evidence.object_encoding,
                    artifact_manifest_bytes=evidence.manifest_bytes,
                    artifact_evidence_digest=evidence.digest,
                    reuse_horizon=horizon,
                    horizon_accounted_body_bytes=missing + horizon * (
                        evidence.manifest_bytes + base["total_arithmetic_body_bytes"]),
                    horizon_scope="public object misses once; manifest and fresh online/preprocessing bodies each workload; excludes control/wire")
            for evidence in request.cost_evidence:
                if evidence.configuration_digest != digest:
                    continue
                if evidence.schedule_digest != requirements["schedule_digest"]:
                    raise NetworkError("cost evidence schedule mismatch")
                for name in ("online_all_link_body_bytes", "preprocessing_all_link_body_bytes"):
                    if getattr(evidence, name) != base[name]:
                        raise NetworkError(f"cost evidence arithmetic geometry mismatch: {name}")
                base["declared_cost_evidence_digest"] = evidence.digest
                base["declared_cost_evidence_source"] = evidence.source
        except (ValueError, TypeError) as exc:
            reasons[(digest, f"UNSUPPORTED_COMPOSITION:{exc}")] += 1
            continue
        if snapshot_stale:
            reasons[(digest, "STALE_SNAPSHOT")] += 1
            continue
        if policy.require_verified_privacy:
            reasons[(digest, "UNRESOLVED_VERIFIED_PRIVACY")] += 1
            continue
        roles = requirements["roles"]
        offers_by_id = {item.party_id: item for item in snapshot.offers}
        kernel = experiment.pipeline.components["kernels"].component
        device = "metal" if kernel == "pllm/apple-metal-int8/v1" else "cpu"
        eligible = []
        for role in roles:
            choices = []
            for offer in snapshot.offers:
                reason = None
                if role["role_id"] == "client" and offer.party_id != policy.client_party_id:
                    continue
                if offer.expires_at_ms <= policy.evaluated_at_ms:
                    reason = "STALE_OFFER"
                elif isinstance(offer, LivePartyOffer) and not offer.accepting:
                    reason = "PARTY_DRAINING"
                elif isinstance(offer, LivePartyOffer) and role["role_id"] not in offer.role_ids:
                    reason = "ROLE_INSTANCE_MISMATCH"
                elif isinstance(offer, LivePartyOffer) and offer.workload is not None and offer.workload != experiment.budget:
                    reason = "WORKLOAD_NOT_INSTALLED"
                elif isinstance(offer, LivePartyOffer) and (
                    offer.observed_at_ms > policy.evaluated_at_ms
                    or policy.evaluated_at_ms - offer.observed_at_ms > policy.max_observation_age_ms
                ):
                    reason = "STALE_OFFER"
                elif isinstance(offer, LivePartyOffer) and (
                    request.source_lock_digest is not None
                    and offer.source_lock_digest != request.source_lock_digest
                ):
                    reason = "SOURCE_LOCK_MISMATCH"
                elif role["capability"] not in offer.capabilities or device not in offer.devices:
                    reason = "CAPABILITY_MISMATCH"
                elif (
                    "*" not in offer.allowed_models
                    and request.model_plan.to_dict()["config_digest"] not in offer.allowed_models
                ):
                    reason = "MODEL_NOT_ALLOWED"
                elif (
                    "*" not in offer.allowed_compositions
                    and experiment.pipeline.digest() not in offer.allowed_compositions
                ):
                    reason = "COMPOSITION_NOT_ALLOWED"
                if reason:
                    reasons[(digest, reason)] += 1
                else:
                    choices.append(offer.party_id)
            eligible.append(choices)
        if any(not choices for choices in eligible):
            reasons[(digest, "NO_ELIGIBLE_PARTY")] += 1
            continue
        for parties in itertools.product(*eligible):
            if assignments >= policy.max_assignments:
                exhaustive = False
                break
            assignments += 1
            assignment = {
                role["role_id"]: party for role, party in zip(roles, parties, strict=True)
            }
            selected_offers = [offers_by_id[key].native_spec() for key in sorted(set(parties))]
            native_request = {
                "schema": "pllm.network_placement.v1",
                "snapshot_digest": snapshot.digest,
                "evaluated_at_ms": policy.evaluated_at_ms,
                "client_party_id": policy.client_party_id,
                "roles": [
                    {"role_id": role, "party_id": party}
                    for role, party in sorted(assignment.items())
                ],
                "parties": selected_offers,
                "required_memory_bytes": base["role_memory_estimates"],
                "required_weight_bytes": base["role_weight_estimates"],
            }
            try:
                validated = admitted.validate(canonical(native_request))
            except ValueError as exc:
                reasons[(digest, f"NATIVE_ADMISSION:{exc}")] += 1
                continue
            costs = _costs(base, assignment, snapshot, policy)
            if policy.incumbent_configuration_digest is not None:
                switch = policy.switch_body_bytes if digest != policy.incumbent_configuration_digest else 0
                costs["switch_body_bytes"] = switch
                if costs["horizon_accounted_body_bytes"] is not None:
                    costs["horizon_accounted_body_bytes"] += switch
            rejection = _policy_rejection(costs, policy)
            if rejection:
                reasons[(digest, rejection)] += 1
                continue
            feasible += 1
            sample = prior_index.get((experiment.pipeline.digest(), assignment_context(
                snapshot, assignment, policy.max_client_native_threads))) if prior_index else None
            if sample is None:
                prior_complete = False
            else:
                costs.update(forecast_ttft_ms=sample.ttft_ms,
                    forecast_decode_ms_per_token=sample.decode_ms_per_token,
                    performance_prior_digest=prior.digest,
                    performance_evidence_digest=prior.evidence_digest,
                    performance_prior_samples=sample.samples,
                    performance_forecast_scope="prior matched workload inside this compiled envelope; not a measurement of the current private request")
            changes = 0 if incumbent_components is None else sum(
                incumbent_components.get(slot) != experiment.pipeline.components.get(slot)
                for slot in set(incumbent_components) | set(experiment.pipeline.components)
            )
            key = (
                *[costs[name] for name in policy.objectives],
                changes,
                digest,
                tuple(sorted(assignment.items())),
            )
            previous = alternatives.get(digest)
            if previous is None:
                alternatives[digest] = {"key": key, "feasible_assignments": 1}
            else:
                previous["feasible_assignments"] += 1
                previous["key"] = min(previous["key"], key)
            if best_key is None or key < best_key:
                best_key = key
                winner = (experiment, validated, canonical(costs))
            if (digest == policy.incumbent_configuration_digest
                    and (incumbent_key is None or key < incumbent_key)):
                incumbent_key = key
                incumbent = (experiment, validated, canonical(costs))
            if sample is not None:
                measured_key = (*[costs[name] for name in policy.prior_objectives], *key)
                previous = prior_alternatives.get(digest)
                if previous is None:
                    prior_alternatives[digest] = {"key": measured_key, "feasible_assignments": 1}
                else:
                    previous["feasible_assignments"] += 1
                    previous["key"] = min(previous["key"], measured_key)
                if prior_key is None or measured_key < prior_key:
                    prior_key, prior_winner = measured_key, (experiment, validated, canonical(costs))
                if (digest == policy.incumbent_configuration_digest
                        and (prior_incumbent_key is None or measured_key < prior_incumbent_key)):
                    prior_incumbent_key, prior_incumbent = measured_key, (experiment, validated, canonical(costs))
        if not exhaustive:
            break
    objectives = policy.objectives
    if prior_complete and exhaustive and prior_winner is not None:
        winner, best_key = prior_winner, prior_key
        incumbent, incumbent_key = prior_incumbent, prior_incumbent_key
        alternatives = prior_alternatives
        objectives = (*policy.prior_objectives, *objectives)
    held_incumbent = False
    if (policy.minimum_switch_improvement_fraction > 0
            and incumbent is not None and winner is not None and incumbent_key is not None
            and best_key is not None and incumbent_key != best_key):
        gain = float(incumbent_key[0]) - float(best_key[0])
        if gain <= float(incumbent_key[0]) * policy.minimum_switch_improvement_fraction:
            winner, best_key = incumbent, incumbent_key
            held_incumbent = True
    if winner is not None and prior is not None:
        costs = strict_load(winner[2])
        costs["selection_basis"] = "matched_performance_prior" if prior_complete and exhaustive else "geometry_fallback"
        costs["effective_objectives"] = list(objectives)
        costs["prior_coverage_complete"] = bool(prior_complete and exhaustive)
        winner = (winner[0], winner[1], canonical(costs))
    status = "feasible" if winner is not None else ("infeasible" if exhaustive else "inconclusive")
    comparisons = tuple(
        {
            "configuration_digest": digest,
            "best_objective_values": list(item["key"][:len(objectives)]),
            "component_changes_from_incumbent": item["key"][-3],
            "best_roles": [{"role_id": role, "party_id": party} for role, party in item["key"][-1]],
            "feasible_assignments": item["feasible_assignments"],
            "selected": item["key"] == best_key,
            "reason": ("SWITCH_HYSTERESIS" if held_incumbent else "selected")
            if item["key"] == best_key
            else "LEXICOGRAPHIC_SCORE_OR_STABLE_IDENTITY",
        }
        for digest, item in sorted(alternatives.items())
    )
    return PlanningResult._create(
        request,
        snapshot,
        status,
        exhaustive,
        candidates,
        assignments,
        feasible,
        winner,
        tuple((digest, reason, count) for (digest, reason), count in sorted(reasons.items())),
        comparisons,
    )


__all__ = ["ArtifactCostEvidence", "CandidateCostEvidence", "PlanningPolicy", "PlanningRequest"]
