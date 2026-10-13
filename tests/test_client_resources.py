"""Client-local budgets change exact schedules; concurrent owners cannot double-spend."""

from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from pllm import ExecutionBudget
from pllm.compiler import plan
from pllm.deployment import ClientResources, NetworkError
from pllm.deployment.resources import _ClientCapacity
from pllm.modeling import lower_model
from pllm.plan import PlanningResult
from pllm.schedulers import MlpPrefillTiling
from pllm.search import PlanningPolicy, PlanningRequest
from test_network_planning import CONFIG, request, snapshot

pytestmark = pytest.mark.rust


@pytest.fixture
def choices():
    graph = lower_model({**CONFIG, "hidden_size": 64, "intermediate_size": 4096,
                         "num_hidden_layers": 2}, batch=1, max_input_tokens=64, max_new_tokens=2)
    full = replace(request().candidates[0], budget=ExecutionBudget(1, 64, 2))
    tiled = full.with_params(pipeline__scheduler=MlpPrefillTiling(tile_rows=16))
    snap = snapshot(offers=tuple(replace(row, max_memory_bytes=2 << 30,
        max_weight_bytes=2 << 30) for row in snapshot().offers))
    policy = PlanningPolicy(evaluated_at_ms=100, allocation_model="native_shared_v1",
        objectives=("prefill_dependent_exchanges",))
    resources = ClientResources(100, 4 << 30, 128 << 20, 1 << 30)
    return PlanningRequest(graph, (full, tiled), policy, client_resources=resources), snap, full, tiled


def test_local_memory_budget_selects_full_tiled_or_rejects_and_replays(choices):
    req, snap, full, tiled = choices
    roomy = plan(req, snapshot=snap)
    assert roomy.experiment == full
    small = plan(replace(req, candidates=(tiled,)), snapshot=snap)
    assert small.costs["client_peak_memory_bytes"] < roomy.costs["client_peak_memory_bytes"]
    budget = small.costs["client_peak_memory_bytes"]
    tight = plan(replace(req, policy=replace(req.policy, max_client_peak_memory_bytes=budget)), snapshot=snap)
    assert tight.experiment == tiled
    assert tight.costs["client_peak_memory_bytes"] == (
        tight.costs["client_incremental_memory_bytes"] + req.client_resources.process_memory_bytes)
    assert PlanningResult.from_spec(tight.to_spec()).digest == tight.digest
    assert plan(replace(req, policy=replace(req.policy,
        max_client_peak_memory_bytes=budget - 1)), snapshot=snap).status == "infeasible"
    headroom = small.costs["client_incremental_memory_bytes"]
    observed = replace(req.client_resources,
                       available_memory_bytes=req.client_resources.reserve_bytes + headroom)
    assert plan(replace(req, client_resources=observed), snapshot=snap).experiment == tiled
    assert plan(replace(req, client_resources=replace(observed, reserved_memory_bytes=1)),
                snapshot=snap).status == "infeasible"


def test_dependent_exchanges_count_both_directed_delays_and_missing_reverse(choices):
    req, snap, full, tiled = choices
    # Two dense layers: QKV, attention output, gate/up, down. Only the last two tile.
    snap = replace(snap, links=tuple(replace(link,
        latency_ms=3 if link.source_party_id == "client" else 7,
        uncertainty_fraction=0.5) for link in snap.links))
    baseline = plan(replace(req, candidates=(full,)), snapshot=snap)
    candidate = plan(replace(req, candidates=(tiled,)), snapshot=snap)
    assert baseline.costs["prefill_dependent_exchanges"] == 8
    assert candidate.costs["prefill_dependent_exchanges"] == 20
    assert baseline.costs["prefill_round_trip_ms"] == 8 * (3 + 7) * 1.5
    assert candidate.costs["prefill_round_trip_ms"] == 20 * (3 + 7) * 1.5
    for result in (baseline, candidate):
        assert result.costs["decode_dependent_exchanges_per_token"] == 8
        assert result.costs["decode_round_trip_ms_per_token"] == 8 * (3 + 7) * 1.5
        assert result.costs["client_cpu_ns"] is None
    assert baseline.costs["online_all_link_body_bytes"] == candidate.costs["online_all_link_body_bytes"]
    incomplete = replace(snap, links=tuple(link for link in snap.links if link.target_party_id != "client"))
    result = plan(req, snapshot=incomplete)
    assert result.costs["prefill_round_trip_ms"] is None
    assert plan(replace(req, policy=replace(req.policy, objectives=("prefill_round_trip_ms",))),
                snapshot=incomplete).status == "infeasible"


def test_named_maximum_selects_tiling_on_roomy_host_and_rechecks_headroom(choices):
    req, snap, full, tiled = choices
    snap = replace(snap, offers=tuple(replace(row, max_memory_bytes=64 << 30)
        if row.party_id == "client" else row for row in snap.offers))
    small = plan(replace(req, candidates=(tiled,)), snapshot=snap)
    limit = 2 << 30
    incremental = small.costs["client_incremental_memory_bytes"]
    resources = replace(req.client_resources, available_memory_bytes=64 << 30,
                        process_memory_bytes=limit - incremental)
    req = replace(req, client_resources=resources, policy=replace(req.policy, max_client_memory="medium"))
    selected = plan(req, snapshot=snap)
    assert selected.experiment == tiled
    assert selected.costs["client_peak_memory_bytes"] == limit
    assert PlanningResult.from_spec(selected.to_spec()).digest == selected.digest
    assert plan(replace(req, policy=replace(req.policy, max_client_memory="high")), snapshot=snap).experiment == full
    assert plan(replace(req, policy=replace(req.policy, max_client_memory="low")), snapshot=snap).status == "infeasible"
    assert plan(replace(req, client_resources=replace(resources,
        available_memory_bytes=resources.reserve_bytes + incremental - 1)), snapshot=snap).status == "infeasible"
    assert plan(replace(req, policy=replace(req.policy,
        max_client_peak_memory_bytes=limit - 1)), snapshot=snap).status == "infeasible"


def test_named_maximum_replay_validation_and_explicit_byte_intersection():
    for tier, limit in (("low", 256 << 20), ("medium", 2 << 30), ("high", 16 << 30)):
        policy = PlanningPolicy(max_client_memory=tier)
        assert policy.client_peak_memory_limit_bytes == limit
        assert PlanningPolicy.from_spec(policy.to_spec()) == policy
        assert replace(policy, max_client_peak_memory_bytes=limit + 1).client_peak_memory_limit_bytes == limit
        assert replace(policy, max_client_peak_memory_bytes=0).client_peak_memory_limit_bytes == 0
    for invalid in ("auto", "LOW", "", 256, True, [], {}):
        with pytest.raises(NetworkError, match="max_client_memory"):
            PlanningPolicy(max_client_memory=invalid)
    policy = PlanningPolicy()
    assert policy.client_peak_memory_limit_bytes is None
    assert "max_client_memory" not in policy.to_spec()
    assert PlanningPolicy.from_spec(policy.to_spec()) == policy


def test_named_maximum_rechecked_at_reservation_even_with_host_capacity(monkeypatch):
    capacity = _ClientCapacity()
    result = SimpleNamespace(costs={"client_incremental_memory_bytes": 100 << 20},
        request=SimpleNamespace(policy=PlanningPolicy(max_client_memory="low")))
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, 64 << 30, 100 << 20, 1 << 30, sum(capacity._owners.values())))
    with capacity.hold(lambda: result):
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with capacity.hold(lambda: result):
                pytest.fail("concurrent requests exceeded named maximum")
    assert not capacity._owners


def test_paged_peak_does_not_turn_legacy_payload_budget_negative(choices):
    from pllm.protocols import ClientBundleTransport

    req, snap, full, _ = choices
    graph = lower_model({**CONFIG, "vocab_size": 151936, "hidden_size": 896,
        "intermediate_size": 4864, "num_attention_heads": 14, "num_key_value_heads": 2,
        "num_hidden_layers": 2}, batch=1, max_input_tokens=64, max_new_tokens=2)
    paged = full.with_params(pipeline__delivery=ClientBundleTransport("artifacts", storage="paged"))
    req = replace(req, model_plan=graph, candidates=(paged,))
    result = plan(req, snapshot=snap)
    assert result.status == "feasible"
    assert result.costs["client_peak_memory_bytes"] < result.costs["client_weight_bytes"]
    assert result.costs["client_payload_bytes"] > 0
    assert plan(replace(req, policy=replace(req.policy, max_client_payload_bytes=0)),
                snapshot=snap).status == "infeasible"


def test_resource_observation_requires_fresh_complete_costs(choices):
    req, _, _, _ = choices
    assert ClientResources.from_spec(req.client_resources.to_spec()) == req.client_resources
    with pytest.raises(NetworkError, match="complete native"):
        replace(req, policy=replace(req.policy, allocation_model="payload"))
    with pytest.raises(NetworkError, match="stale"):
        replace(req, client_resources=replace(req.client_resources, observed_at_ms=101))
    with pytest.raises(NetworkError, match="stale"):
        replace(req, policy=replace(req.policy, evaluated_at_ms=100_000))


def test_native_worker_limit_is_bounded_replayable_and_rejects_oversized_local_loader(choices):
    from pllm.kernels import Cpu
    from pllm.profiles import ClientOnlyCpu

    req, snap, full, _ = choices
    for invalid in (True, 0, 33, 1.0):
        with pytest.raises(NetworkError):
            replace(req.policy, max_client_native_threads=invalid)
    assert "max_client_native_threads" not in req.policy.to_spec()
    policy = replace(req.policy, max_client_native_threads=1)
    assert PlanningPolicy.from_spec(policy.to_spec()) == policy
    result = plan(replace(req, policy=policy), snapshot=snap)
    assert result.status == "feasible" and result.costs["client_native_thread_limit"] == 1
    assert PlanningResult.from_spec(result.to_spec()).digest == result.digest
    local = replace(full, pipeline=ClientOnlyCpu(full.pipeline.model, kernels=Cpu(threads=2)))
    result = plan(replace(req, policy=policy, candidates=(local,)), snapshot=snap)
    assert result.status == "infeasible"
    assert "native worker limit" in result.canonical_bytes().decode()


def test_process_capacity_is_atomic_across_threads_and_released_after_failure(monkeypatch):
    capacity = _ClientCapacity()
    result = SimpleNamespace(costs={"client_incremental_memory_bytes": 100},
        request=SimpleNamespace(policy=PlanningPolicy(max_client_peak_memory_bytes=150)))
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, 175, 25, 25, sum(capacity._owners.values())))
    entered, leave = threading.Event(), threading.Event()
    failures = []

    def first():
        try:
            with capacity.hold(lambda: result):
                entered.set()
                assert leave.wait(5)
                raise RuntimeError("cancelled execution")
        except RuntimeError:
            pass
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert entered.wait(5)
        with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
            with capacity.hold(lambda: result):
                pytest.fail("concurrent request spent reserved memory twice")
    finally:
        leave.set()
        worker.join(5)
    assert not worker.is_alive() and not failures and not capacity._owners
    with capacity.hold(lambda: result):
        assert sum(capacity._owners.values()) == 100
    assert not capacity._owners


def test_unavailable_os_observation_rejects(monkeypatch):
    import psutil

    def unavailable():
        raise OSError("unavailable")

    monkeypatch.setattr(psutil, "virtual_memory", unavailable)
    with pytest.raises(NetworkError, match="OBSERVATION_UNAVAILABLE"):
        _ClientCapacity().observe()


@pytest.mark.parametrize("version", (1, 2))
def test_container_budget_includes_parent_pressure_and_no_swap_credit(tmp_path, version):
    from pllm.deployment.resources import _linux_memory_headroom

    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    mount = tmp_path / "memory-controller"
    child = mount / "app"
    child.mkdir(parents=True)
    kind = "cgroup2" if version == 2 else "cgroup"
    controllers = "" if version == 2 else "memory"
    (proc / "cgroup").write_text(f"0:{controllers}:/app\n")
    (proc / "mountinfo").write_text(f"29 23 0:26 / {mount} rw - {kind} cgroup rw,memory\n")
    limit, usage = (("memory.max", "memory.current") if version == 2
                    else ("memory.limit_in_bytes", "memory.usage_in_bytes"))
    (mount / limit).write_text("1000")
    (mount / usage).write_text("850")
    (child / limit).write_text("500")
    (child / usage).write_text("200")
    # The parent, including sibling allocations, is stricter than the child.
    assert _linux_memory_headroom(tmp_path / "proc") == 150
    resources = ClientResources(100, 4000, 100, 1000, 25, 150)
    assert resources.headroom_bytes == 125
    assert ClientResources.from_spec(resources.to_spec()) == resources
    (mount / usage).write_text("1001")
    assert _linux_memory_headroom(tmp_path / "proc") == 0
    (mount / usage).write_text("invalid")
    with pytest.raises(ValueError):
        _linux_memory_headroom(tmp_path / "proc")


def test_namespace_cgroup_root_and_inaccessible_controller(tmp_path):
    from pllm.deployment.resources import _linux_memory_headroom

    proc = tmp_path / "proc/self"
    proc.mkdir(parents=True)
    mount = tmp_path / "memory-controller"
    mount.mkdir()
    (proc / "cgroup").write_text("0::/\n")
    (proc / "mountinfo").write_text(f"29 23 0:26 /containers/demo {mount} rw - cgroup2 cgroup rw\n")
    (mount / "memory.max").write_text("500")
    (mount / "memory.current").write_text("125")
    assert _linux_memory_headroom(tmp_path / "proc") == 375
    (mount / "memory.max").write_text("max")
    assert _linux_memory_headroom(tmp_path / "proc") is None
    (mount / "memory.max").unlink()
    with pytest.raises(FileNotFoundError):
        _linux_memory_headroom(tmp_path / "proc")


def test_saved_payload_decision_rechecks_current_process_without_rewriting(choices, monkeypatch):
    req, snap, full, _ = choices
    old = plan(replace(req, candidates=(full,), client_resources=None,
                      policy=replace(req.policy, allocation_model="payload")), snapshot=snap)
    assert old.status == "feasible" and old.costs["client_peak_memory_bytes"] is None
    original = old.canonical_bytes()
    capacity = _ClientCapacity()
    available = [4 << 30]
    monkeypatch.setattr(capacity, "observe", lambda: ClientResources(
        100, available[0], 128 << 20, 1 << 30, sum(capacity._owners.values())))
    with capacity.hold(lambda: old) as selected:
        assert selected is old
        assert sum(capacity._owners.values()) > 256 << 20
    assert not capacity._owners
    available[0] = 1 << 30
    with pytest.raises(NetworkError, match="CAPACITY_CHANGED"):
        with capacity.hold(lambda: old):
            pytest.fail("saved payload-only decision bypassed current memory admission")
    assert not capacity._owners and old.canonical_bytes() == original
