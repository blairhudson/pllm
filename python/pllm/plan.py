"""Public immutable compiled-plan facade."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pllm import _native


@dataclass(frozen=True, slots=True)
class CompiledPlan:
    """Opaque compiled plan exposing records but no execution methods."""

    _native: Any = field(repr=False)

    def __post_init__(self) -> None:
        from pllm import _native

        if not isinstance(self._native, _native.CompiledPlan):
            raise TypeError("compiled plan handle must come from pllm.compile")

    @property
    def logical_plan(self) -> bytes:
        return self._native.logical_plan

    @property
    def execution_plan(self) -> bytes:
        return self._native.execution_plan

    @property
    def plan_lock(self) -> bytes:
        return self._native.plan_lock

    @property
    def region_program(self) -> bytes:
        return self._native.region_program

    @property
    def configuration_digest(self) -> str:
        return self._native.configuration_digest

    @property
    def logical_plan_digest(self) -> str:
        return self._native.logical_plan_digest

    @property
    def execution_plan_digest(self) -> str:
        return self._native.execution_plan_digest

    @property
    def plan_lock_digest(self) -> str:
        return self._native.plan_lock_digest

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._native.input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._native.output_shape


def _wrap(native: _native.CompiledPlan) -> CompiledPlan:
    return CompiledPlan(native)


def _unwrap(plan: CompiledPlan) -> _native.CompiledPlan:
    return plan._native


@dataclass(frozen=True, slots=True, init=False)
class PlanningResult:
    """Replayable decision, not a reservation or serialized opaque native handle."""

    _document: bytes = field(repr=False)
    request: Any
    snapshot: Any
    experiment: Any

    @classmethod
    def _create(
        cls,
        request,
        snapshot,
        status,
        exhaustive,
        candidates,
        assignments,
        feasible,
        winner,
        rejections,
        alternatives,
    ):
        import hashlib

        from pllm.deployment.network import canonical, strict_load

        selection = None
        experiment = None
        if winner is not None:
            experiment, native, costs = winner
            selection = {
                "configuration_digest": experiment.configuration_digest(),
                "native_placement": strict_load(native),
                "costs": strict_load(costs),
            }
        document = {
            "schema": "pllm.planning_result.v1",
            "request": request.to_spec(),
            "snapshot": snapshot.to_spec(),
            "request_digest": request.digest,
            "snapshot_digest": snapshot.digest,
            "model_plan_digest": request.model_plan.digest,
            "policy_digest": request.policy.digest,
            "status": status,
            "coverage": {
                "exhaustive": exhaustive,
                "candidates_evaluated": candidates,
                "assignments_evaluated": assignments,
                "feasible_assignments": feasible,
            },
            "selection": selection,
            "alternatives": list(alternatives),
            "rejections": [
                {"configuration_digest": digest, "reason": reason, "count": count}
                for digest, reason, count in rejections
            ],
            "privacy_scope": "declared operator separation; unauthenticated local development, not physical proof",
        }
        document["result_digest"] = hashlib.sha256(
            b"pllm.planning_result.v1\0" + canonical(document)
        ).hexdigest()
        result = object.__new__(cls)
        object.__setattr__(result, "_document", canonical(document))
        object.__setattr__(result, "request", request)
        object.__setattr__(result, "snapshot", snapshot)
        object.__setattr__(result, "experiment", experiment)
        return result

    def canonical_bytes(self) -> bytes:
        return self._document

    def to_spec(self) -> dict:
        from pllm.deployment.network import strict_load

        return strict_load(self._document)

    @property
    def digest(self) -> str:
        return self.to_spec()["result_digest"]

    @property
    def status(self) -> str:
        return self.to_spec()["status"]

    @property
    def exhaustive(self) -> bool:
        return self.to_spec()["coverage"]["exhaustive"]

    @property
    def costs(self):
        from pllm.modeling import _freeze

        selection = self.to_spec()["selection"]
        return None if selection is None else _freeze(selection["costs"])

    @property
    def native_placement(self):
        from pllm.modeling import _freeze

        selection = self.to_spec()["selection"]
        return None if selection is None else _freeze(selection["native_placement"])

    @classmethod
    def from_spec(cls, value, *, request=None, snapshot=None):
        from pllm.compiler import plan
        from pllm.deployment.network import NetworkError, NetworkSnapshot, canonical
        from pllm.search.placement import PlanningRequest

        # Replay all inputs and the entire bounded search. Scores and claims are not assertions.
        if not isinstance(value, dict) or value.get("schema") != "pllm.planning_result.v1":
            raise NetworkError("invalid planning result schema")
        canonical(value)
        try:
            embedded_request = PlanningRequest.from_spec(value["request"])
            embedded_snapshot = NetworkSnapshot.from_spec(value["snapshot"])
        except KeyError as exc:
            raise NetworkError("missing planning result input") from exc
        if request is not None and request.canonical_bytes() != embedded_request.canonical_bytes():
            raise NetworkError("request input mismatch")
        if (
            snapshot is not None
            and snapshot.canonical_bytes() != embedded_snapshot.canonical_bytes()
        ):
            raise NetworkError("snapshot input mismatch")
        result = plan(embedded_request, snapshot=embedded_snapshot)
        if result.canonical_bytes() != canonical(value):
            raise NetworkError("planning result tamper or replay mismatch")
        return result

    @classmethod
    def from_file(cls, path, *, request=None, snapshot=None):
        from pathlib import Path

        from pllm.deployment.network import MAX_DOCUMENT_BYTES, strict_load

        with Path(path).expanduser().open("rb") as stream:
            value = strict_load(stream.read(MAX_DOCUMENT_BYTES + 1))
        return cls.from_spec(value, request=request, snapshot=snapshot)

    def validate(self, *, snapshot=None, evaluated_at_ms=None):
        """Revalidate selected binding at a caller-specified clock/epoch; no reservation."""
        from pllm import _native
        from pllm.deployment.network import NetworkError, canonical, integer, strict_load
        from pllm.search.placement import _costs, _policy_rejection
        from dataclasses import replace

        if self.status != "feasible":
            raise NetworkError("NO_FEASIBLE_PLACEMENT")
        snapshot = self.snapshot if snapshot is None else snapshot
        now = self.request.policy.evaluated_at_ms if evaluated_at_ms is None else evaluated_at_ms
        integer(now, "evaluated_at_ms")
        if snapshot.network_id != self.snapshot.network_id:
            raise NetworkError("network identity mismatch")
        if (
            snapshot.observed_at_ms > now
            or snapshot.expires_at_ms <= now
            or now - snapshot.observed_at_ms > self.request.policy.max_observation_age_ms
        ):
            raise NetworkError("STALE_SNAPSHOT")
        value = self.to_spec()["selection"]["native_placement"]
        offers = {item.party_id: item for item in snapshot.offers}
        original = {item["party_id"]: item for item in value["parties"]}
        for party, previous in original.items():
            current = offers.get(party)
            if current is None or current.instance_epoch != previous["instance_epoch"]:
                raise NetworkError("STALE_INSTANCE_EPOCH")
            if current.operator_id != previous["operator_id"]:
                raise NetworkError("operator identity changed")
            kernel = self.experiment.pipeline.components["kernels"].component
            device = "metal" if kernel == "pllm/apple-metal-int8/v1" else "cpu"
            if device not in current.devices or (
                "*" not in current.allowed_compositions
                and self.experiment.pipeline.digest() not in current.allowed_compositions
            ):
                raise NetworkError("CAPABILITY_MISMATCH")
        names = {
            "schema",
            "snapshot_digest",
            "evaluated_at_ms",
            "client_party_id",
            "roles",
            "parties",
            "required_memory_bytes",
            "required_weight_bytes",
        }
        native_request = {key: value[key] for key in names}
        native_request.update(
            snapshot_digest=snapshot.digest,
            evaluated_at_ms=now,
            parties=[offers[key].native_spec() for key in sorted(original)],
        )
        validated = strict_load(
            _native.validate_network_placement(
                self.request.model_plan.canonical_bytes(),
                self.experiment.pipeline.canonical_bytes(),
                canonical(native_request),
            )
        )
        assignment = {item["role_id"]: item["party_id"] for item in value["roles"]}
        policy = replace(self.request.policy, evaluated_at_ms=now)
        costs = _costs(self.to_spec()["selection"]["costs"], assignment, snapshot, policy)
        rejection = _policy_rejection(costs, policy)
        if rejection:
            raise NetworkError(rejection)
        return validated


__all__ = ["CompiledPlan", "PlanningResult"]
