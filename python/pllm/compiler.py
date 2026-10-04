from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pllm.plan import CompiledPlan, _wrap

if TYPE_CHECKING:
    from pathlib import Path
    from pllm.configuration import Experiment
    from pllm.deployment.network import NetworkSnapshot
    from pllm.plan import PlanningResult
    from pllm.search import SearchSpace
    from pllm.search.placement import (
        ArtifactCostEvidence,
        ClientStateCostEvidence,
        PlanningPolicy,
        PlanningRequest,
    )


class CompilationError(ValueError):
    """The compiler rejected an invalid or unsupported request."""


def compile(request: Mapping[str, Any] | bytes | str) -> CompiledPlan:
    """Compile one strict request into an immutable native plan."""
    from pllm import _native

    if isinstance(request, Mapping):
        try:
            document = json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise CompilationError(f"compile request is not canonical JSON: {exc}") from exc
    elif type(request) is bytes:
        document = request
    elif type(request) is str:
        document = request.encode("utf-8")
    else:
        raise TypeError("request must be a mapping, bytes, or JSON string")
    try:
        return _wrap(_native.compile_plan(document))
    except ValueError as exc:
        raise CompilationError(str(exc)) from exc


def plan(request: PlanningRequest, *, snapshot: NetworkSnapshot) -> PlanningResult:
    """Plan explicit candidates offline against one immutable static snapshot."""
    from pllm.search.placement import search_placements

    return search_placements(request, snapshot)


def plan_on_load(
    experiment: Experiment,
    *,
    snapshot: NetworkSnapshot,
    policy: PlanningPolicy,
    space: SearchSpace | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    artifact_evidence: tuple[ArtifactCostEvidence, ...] = (),
    state_evidence: tuple[ClientStateCostEvidence, ...] = (),
) -> PlanningResult:
    """Resolve a locked checkpoint once and select an ordinary executable plan.

    Explicitly performs the source-resolution step of ``load_model``. Does not
    load tensor values, benchmark candidates, reserve roles or issue material.
    Returned PlanningResult uses the existing SDK/gateway/benchmark path.
    Unknown required costs still reject a candidate.
    """
    import hashlib
    import math
    from dataclasses import replace

    from pllm.configuration import Experiment
    from pllm.model_loader import resolve_model
    from pllm.modeling import lower_model
    from pllm.protocols import TwoOnlineOffsetLinear
    from pllm.search import (
        GridSearch,
        PlanningPolicy,
        PlanningRequest,
        SearchSpace,
        optimization_space,
    )

    if not isinstance(experiment, Experiment) or not isinstance(policy, PlanningPolicy):
        raise TypeError("plan_on_load requires Experiment and PlanningPolicy")
    if space is None:
        space = optimization_space(experiment, max_candidates=policy.max_candidates)
    if (
        not isinstance(space, SearchSpace)
        or space.base.configuration_digest() != experiment.configuration_digest()
    ):
        raise ValueError("optimization space must belong to the supplied base Experiment")
    if math.prod(len(values) for values in space.parameters.values()) > policy.max_candidates:
        raise ValueError("optimization space exceeds the candidate admission bound")
    candidates = tuple(row.experiment for row in GridSearch("load", space).candidates())
    if not candidates:
        raise ValueError("optimization space has no candidates satisfying its constraints")
    base_components = experiment.pipeline.components
    adjustable = {"kernels", "inventory", "delivery", "placement", "cache"}
    for candidate in candidates:
        if (
            candidate.pipeline.model != experiment.pipeline.model
            or candidate.budget != experiment.budget
            or candidate.deployment != experiment.deployment
        ):
            raise ValueError("load-time optimization cannot change model, workload or deployment")
        keys = set(base_components) | set(candidate.pipeline.components)
        if any(
            base_components.get(key) != candidate.pipeline.components.get(key)
            for key in keys - adjustable - {"linear"}
        ):
            raise ValueError("load-time optimization cannot change numeric or privacy contracts")
        before, after = base_components.get("linear"), candidate.pipeline.components.get("linear")
        if before != after and (
            before is None
            or after is None
            or before.component != after.component
            or before.component != TwoOnlineOffsetLinear.descriptor.component
        ):
            raise ValueError("load-time optimization cannot change protocol or topology")
    if experiment.configuration_digest() not in {row.configuration_digest() for row in candidates}:
        raise ValueError(
            "load-time optimization space must include its incumbent; constraints are never bypassed"
        )
    if policy.incumbent_configuration_digest is None:
        policy = replace(policy, incumbent_configuration_digest=experiment.configuration_digest())

    resolved = resolve_model(experiment.pipeline.model, token=token, cache_dir=cache_dir)
    if resolved.path is None:
        raise CompilationError("load-time planning requires a local source-locked checkpoint")
    try:
        with (resolved.path / "config.json").open("rb") as stream:
            config_bytes = stream.read((2 << 20) + 1)
    except OSError as exc:
        raise CompilationError("load-time planning requires a source-locked config.json") from exc
    if len(config_bytes) > 2 << 20:
        raise CompilationError("source configuration exceeds the bounded planning input")
    lock = resolved.manifest.metadata.get("source_lock", {})
    expected = [
        row.get("sha256") for row in lock.get("files", ()) if row.get("path") == "config.json"
    ]
    if expected != [hashlib.sha256(config_bytes).hexdigest()]:
        raise CompilationError("source configuration changed after checkpoint resolution")
    model_plan = lower_model(
        json.loads(config_bytes),
        batch=1,
        max_input_tokens=experiment.budget.max_input_tokens,
        max_new_tokens=experiment.budget.max_new_tokens,
    )
    request = PlanningRequest(
        model_plan,
        candidates,
        policy,
        source_lock_digest=resolved.manifest.source_lock_digest,
        artifact_evidence=artifact_evidence,
        state_evidence=state_evidence,
    )
    return plan(request, snapshot=snapshot)


__all__ = ["CompilationError", "compile", "plan", "plan_on_load"]
