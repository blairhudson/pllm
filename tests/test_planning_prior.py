"""Prior ranking reuses immutable work but never reuses live admission."""
from dataclasses import replace
import itertools

import pytest

from pllm import _native
from pllm.compiler import plan
from pllm.deployment import NetworkError
from pllm.plan import PlanningResult
from pllm.search import PerformanceSample, PlanningPrior, PlanningRequest
from pllm.search._planning_cache import _cache, _PlanningCache
from pllm.search.performance import assignment_context, runtime_identity
from test_network_planning import assigned, request, snapshot

pytestmark = pytest.mark.rust


def prior_request():
    req = replace(request(), source_lock_digest="c" * 64)
    snap = snapshot()
    desired = {"client": "client", "inference": "b", "preparation": "c"}
    rows = []
    for inference, preparation in itertools.permutations(("a", "b", "c"), 2):
        assignment = {"client": "client", "inference": inference, "preparation": preparation}
        rows.append(PerformanceSample(req.candidates[0].pipeline.digest(),
            assignment_context(snap, assignment, None),
            10.0 if assignment == desired else 100.0, 5.0, 3))
    prior = PlanningPrior(req.model_plan.digest, req.source_lock_digest, req.workload_cohort_digest,
        runtime_identity(), "e" * 64, "f" * 64, 0, 900, 3, 2, 2, True, tuple(rows))
    return replace(req, performance_prior=prior), snap, desired


def test_matched_prior_changes_choice_roundtrips_and_stays_client_local():
    req, snap, desired = prior_request()
    control = plan(replace(req, performance_prior=None), snapshot=snap)
    result = plan(req, snapshot=snap)
    assert assigned(control) != desired
    assert assigned(result) == desired
    assert result.costs["selection_basis"] == "matched_performance_prior"
    assert result.costs["forecast_ttft_ms"] == 10
    assert result.costs["effective_objectives"][:2] == ("forecast_ttft_ms", "forecast_decode_ms_per_token")
    assert PlanningRequest.from_spec(req.to_spec()) == req
    assert PlanningResult.from_spec(result.to_spec()).canonical_bytes() == result.canonical_bytes()
    assert "performance_prior" not in result.native_placement


@pytest.mark.parametrize("mismatch", ("expired", "future", "source", "model", "runtime", "bounds",
                                       "epoch", "link", "link_expired", "link_future", "missing", "disabled", "truncated"))
def test_missing_or_mismatched_prior_uses_geometry(mismatch):
    req, snap, _ = prior_request()
    if mismatch in {"source", "model", "runtime", "bounds"}:
        name = {"source": "source_lock_digest", "model": "model_plan_digest",
                "runtime": "runtime_digest", "bounds": "workload_cohort_digest"}[mismatch]
        req = replace(req, performance_prior=replace(req.performance_prior, **{name: "9" * 64}))
    elif mismatch == "expired":
        req = replace(req, performance_prior=replace(req.performance_prior, expires_at_ms=100))
    elif mismatch == "future":
        req = replace(req, performance_prior=replace(req.performance_prior, observed_at_ms=101))
    elif mismatch == "epoch":
        snap = replace(snap, offers=tuple(replace(row, instance_epoch="8" * 64) for row in snap.offers))
    elif mismatch == "link":
        snap = replace(snap, links=tuple(replace(row, latency_ms=7) for row in snap.links))
    elif mismatch == "link_expired":
        snap = replace(snap, links=tuple(replace(row, expires_at_ms=100) for row in snap.links))
    elif mismatch == "link_future":
        snap = replace(snap, links=tuple(replace(row, observed_at_ms=101) for row in snap.links))
    elif mismatch == "missing":
        req = replace(req, performance_prior=replace(req.performance_prior, rows=req.performance_prior.rows[:-1]))
    elif mismatch == "disabled":
        req = replace(req, policy=replace(req.policy, prior_objectives=()))
    else:
        req = replace(req, policy=replace(req.policy, max_assignments=2))
    if mismatch in {"link_expired", "link_future"}:
        req = replace(req, policy=replace(req.policy, objectives=("total_arithmetic_body_bytes",)))
    result = plan(req, snapshot=snap)
    control = plan(replace(req, performance_prior=None), snapshot=snap)
    assert assigned(result) == assigned(control)
    assert result.costs["selection_basis"] == "geometry_fallback"


def test_replayed_request_checks_current_runtime(monkeypatch):
    req, snap, _ = prior_request()
    document = req.to_spec()
    monkeypatch.setenv("PLLM_NATIVE_THREADS", "31")
    restored = PlanningRequest.from_spec(document)
    assert plan(restored, snapshot=snap).costs["selection_basis"] == "geometry_fallback"


def test_faster_prior_cannot_override_live_capacity_or_memory():
    req, snap, desired = prior_request()
    result = plan(req, snapshot=snap)
    saturated = replace(snap, offers=tuple(replace(row, active_sessions=row.max_sessions)
        if row.party_id == "b" else row for row in snap.offers))
    assert assigned(plan(req, snapshot=saturated)) != desired
    with pytest.raises(ValueError):
        result.validate(snapshot=saturated)
    impossible = replace(req, policy=replace(req.policy, max_client_payload_bytes=1))
    assert plan(impossible, snapshot=snap).status == "infeasible"


def test_warm_decisions_reuse_native_facts_but_revalidate_every_assignment(monkeypatch):
    _cache.clear()
    constructor = _native.NetworkPlacementAdmission
    constructed, validated = [], []

    class Tracked:
        def __init__(self, *args):
            constructed.append(1)
            self.inner = constructor(*args)

        def requirements(self):
            return self.inner.requirements()

        def validate(self, document):
            validated.append(document)
            return self.inner.validate(document)

    monkeypatch.setattr(_native, "NetworkPlacementAdmission", Tracked)
    try:
        req, snap, _ = prior_request()
        first = plan(req, snapshot=snap)
        count = len(validated)
        second = plan(req, snapshot=snap)
        assert first.canonical_bytes() == second.canonical_bytes()
        assert len(constructed) == 1 and len(validated) == count * 2
        _, _, copy = _cache.get(req, req.candidates[0])
        copy["role_memory_estimates"]["client"] = 0
        assert _cache.get(req, req.candidates[0])[2]["role_memory_estimates"]["client"] > 0
        plan(replace(req, policy=replace(req.policy, buffer_reserve_bytes=2 << 20)), snapshot=snap)
        assert len(constructed) == 2
    finally:
        _cache.clear()


def test_cache_limits_and_prior_strict_bounds():
    req, _, _ = prior_request()
    cache = _PlanningCache(max_entries=1, max_bytes=64 << 10)
    cache.get(req, req.candidates[0])
    cache.get(replace(req, policy=replace(req.policy, buffer_reserve_bytes=2 << 20)), req.candidates[0])
    assert len(cache._entries) == 1 and cache._bytes <= 64 << 10
    too_small = _PlanningCache(max_bytes=1)
    too_small.get(req, req.candidates[0])
    assert not too_small._entries
    for changes in ({"rows": req.performance_prior.rows * 2}, {"output_tokens": 1},
                    {"expires_at_ms": 31 * 86400_000}):
        with pytest.raises(NetworkError):
            replace(req.performance_prior, **changes)
    with pytest.raises(NetworkError):
        replace(req.performance_prior.rows[0], ttft_ms=float("nan"))
