"""Public selected execution contracts; client search and resource policy stay local."""

from __future__ import annotations

from dataclasses import dataclass

from pllm.configuration import Deployment, ExecutionBudget, Experiment, Pipeline
from pllm.deployment.network import (
    NetworkError, NetworkSnapshot, PublicRecord, canonical, digest_value, identity,
    integer, strict_load,
)
from pllm.modeling import ModelPlan


@dataclass(frozen=True, slots=True)
class ExecutionContract(PublicRecord):
    """Inputs needed to independently admit one selected execution, not its search."""

    SCHEMA = "pllm.execution_contract.v1"
    pipeline: Pipeline
    budget: ExecutionBudget
    deployment: Deployment
    model_plan: ModelPlan
    source_lock_digest: str
    schedule_digest: str
    selection_snapshot_digest: str
    snapshot: NetworkSnapshot
    roles: tuple[tuple[str, str], ...]

    def __post_init__(self):
        for value, kind in ((self.pipeline, Pipeline), (self.budget, ExecutionBudget),
                            (self.deployment, Deployment), (self.model_plan, ModelPlan),
                            (self.snapshot, NetworkSnapshot)):
            if not isinstance(value, kind):
                raise TypeError(f"execution contract requires {kind.__name__}")
        for name in ("source_lock_digest", "schedule_digest", "selection_snapshot_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        if self.deployment.kind != "network" or self.budget.requests != 1:
            raise NetworkError("execution contract requires one network response")
        if (type(self.roles) is not tuple or not 1 <= len(self.roles) <= 8
                or any(type(row) is not tuple or len(row) != 2 for row in self.roles)):
            raise NetworkError("invalid execution role assignments")
        for role, party in self.roles:
            identity(role, "role")
            identity(party, "party")
        if len(dict(self.roles)) != len(self.roles) or "client" not in dict(self.roles):
            raise NetworkError("duplicate or missing execution role")
        if ({party for _, party in self.roles} != {row.party_id for row in self.snapshot.offers}
                or self.snapshot.links or self.snapshot.source != "selected-execution"):
            raise NetworkError("execution contract must contain only selected offers")
        graph = self.model_plan.to_dict()
        if (graph["prefill"]["batch"] != 1
                or graph["prefill"]["query_sequence"] != self.budget.max_input_tokens
                or graph["decode"]["maximum_key_sequence"] != (
                    self.budget.max_input_tokens + self.budget.max_new_tokens - 1)):
            raise NetworkError("execution contract workload differs from ModelPlan")
        object.__setattr__(self, "roles", tuple(sorted(self.roles)))
        self.canonical_bytes()

    def to_spec(self):
        return {**super(ExecutionContract, self).to_spec(), "model_plan": self.model_plan.to_dict()}

    @classmethod
    def from_spec(cls, value):
        data = cls._fields(value)
        selected = Experiment.from_spec({"schema": "pllm.experiment.v3", "name": "admitted",
            **{name: data[name] for name in ("pipeline", "budget", "deployment")}})
        for name in ("pipeline", "budget", "deployment"):
            data[name] = getattr(selected, name)
        data["snapshot"] = NetworkSnapshot.from_spec(data["snapshot"])
        data["model_plan"] = ModelPlan(canonical(data["model_plan"]))
        if type(data["roles"]) is not list or any(type(row) is not list for row in data["roles"]):
            raise NetworkError("execution roles must be an array")
        data["roles"] = tuple(tuple(row) for row in data["roles"])
        return cls(**data)

    @classmethod
    def from_result(cls, result, snapshot):
        selected = result.experiment
        if selected is None:
            raise NetworkError("NO_FEASIBLE_PLACEMENT")
        native = result.native_placement
        roles = tuple((row["role_id"], row["party_id"]) for row in native["roles"])
        parties = {party for _, party in roles}
        public = NetworkSnapshot(snapshot.network_id,
            tuple(row for row in snapshot.offers if row.party_id in parties), (),
            "selected-execution", snapshot.observed_at_ms, snapshot.expires_at_ms)
        return cls(selected.pipeline, selected.budget, selected.deployment,
            result.request.model_plan, result.request.source_lock_digest,
            native["schedule_digest"], result.snapshot.digest, public, roles)

    @property
    def experiment(self):
        return Experiment("admitted", self.pipeline, self.deployment, self.budget)

    def validate(self, *, evaluated_at_ms):
        from pllm import _native
        from pllm.runtime.benchmark_memory import BenchmarkMemoryError, estimate_plan_memory

        integer(evaluated_at_ms, "evaluated_at_ms")
        if not self.snapshot.observed_at_ms <= evaluated_at_ms < self.snapshot.expires_at_ms:
            raise NetworkError("STALE_SNAPSHOT")
        requirements = strict_load(_native.network_placement_requirements(
            self.model_plan.canonical_bytes(), self.pipeline.canonical_bytes()))
        if requirements["schedule_digest"] != self.schedule_digest:
            raise NetworkError("compiled execution schedule mismatch")
        # Each provider independently prices its public allocation owners. No client
        # budget, local RSS, cache assumption or caller-supplied cost is authoritative.
        try:
            estimate = estimate_plan_memory(self.model_plan, self.pipeline,
                max_input_tokens=self.budget.max_input_tokens,
                max_output_tokens=self.budget.max_new_tokens,
                inventory_rows=max(64, self.experiment.resolve().prepared_inventory_rows))
        except BenchmarkMemoryError as exc:
            raise NetworkError("provider allocation owners unavailable") from exc
        if estimate["limitations"]:
            raise NetworkError("provider allocation owners unsupported")
        memory = dict(requirements["minimum_memory_bytes"])
        for role, amount in estimate["provider_peak_bytes"].items():
            memory[role] = max(memory[role], amount)
        native = {
            "schema": "pllm.network_placement.v1", "snapshot_digest": self.snapshot.digest,
            "evaluated_at_ms": evaluated_at_ms, "client_party_id": dict(self.roles)["client"],
            "roles": [{"role_id": role, "party_id": party} for role, party in self.roles],
            "parties": [row.native_spec() for row in self.snapshot.offers],
            "required_memory_bytes": memory,
            "required_weight_bytes": requirements["minimum_weight_bytes"],
        }
        return strict_load(_native.validate_network_placement(self.model_plan.canonical_bytes(),
            self.pipeline.canonical_bytes(), canonical(native)))


@dataclass(frozen=True, slots=True)
class RoleAdmission(PublicRecord):
    SCHEMA = "pllm.role_admission.v1"
    role_id: str
    execution: ExecutionContract

    def __post_init__(self):
        if not isinstance(self.execution, ExecutionContract):
            raise TypeError("role admission requires ExecutionContract")
        identity(self.role_id, "role")
        if self.role_id == "client" or self.role_id not in dict(self.execution.roles):
            raise NetworkError("role admission requires a selected provider")

    @classmethod
    def from_spec(cls, value):
        data = cls._fields(value)
        data["execution"] = ExecutionContract.from_spec(data["execution"])
        return cls(**data)
