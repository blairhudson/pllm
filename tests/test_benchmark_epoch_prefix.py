from __future__ import annotations

from types import SimpleNamespace
import copy
import threading

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm._benchmark_contract import BenchmarkTarget, Scenario, TargetUnavailable
from pllm._benchmark_worker import Measurement
from pllm.metrics import Sampling
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_response_group import (
    CONCURRENCY_CONTRACT, group_memory_contract, inspect_private_cleanup, prefix_owners,
    require_concurrency_contract, run_group,
)
from pllm.runtime.benchmark_response_target import audit_metrics, response_metrics


def experiment():
    return Experiment("groups", MaskedLinearCpu(Model.tiny()),
                      Deployment.local(root="local://groups"), ExecutionBudget(8, 128, 8))


def test_group_contract_serial_identity_and_geometry():
    serial = BenchmarkTarget.response(experiment())
    assert serial.to_spec() == BenchmarkTarget.response(experiment(), concurrent_requests=1).to_spec()
    assert "concurrent_requests" not in serial.to_spec()["parameters"]
    group = BenchmarkTarget.response(experiment(), concurrent_requests=2)
    assert BenchmarkTarget.from_spec(group.to_spec()).digest == group.digest
    assert group.digest != BenchmarkTarget.response(experiment(), concurrent_requests=3).digest


@pytest.mark.parametrize("count", [True, 0, 9, 2.5])
def test_group_contract_bounds(count):
    with pytest.raises(ValueError):
        BenchmarkTarget.response(experiment(), concurrent_requests=count)


def test_group_rejects_overlapping_dashboard_lifecycles():
    with pytest.raises(ValueError, match="cold native"):
        BenchmarkTarget.response(experiment(), scenario=Scenario.repeat(), concurrent_requests=2)
    with pytest.raises(ValueError, match="cold native"):
        BenchmarkTarget.response(experiment(), backend="docker", concurrent_requests=2)


def test_old_runtime_cannot_claim_concurrent_semantics():
    with pytest.raises(TargetUnavailable, match="group_runtime_concurrency_contract_unavailable"):
        require_concurrency_contract(SimpleNamespace())
    require_concurrency_contract(SimpleNamespace(_core=SimpleNamespace(
        prepared_concurrency_contract=CONCURRENCY_CONTRACT)))


def test_group_real_call_contract_natural_eos_counts_and_redaction():
    gate = threading.Barrier(2)
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        gate.wait(timeout=2)
        return SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=5, output_tokens=1),
                               output_text="private output", incomplete_details=None)
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    rows = run_group(client, "private input", requests=2, max_output_tokens=8,
                     timeout_seconds=2, commitment=lambda text: "salted-commitment")
    assert len(calls) == 2 and all(call["store"] is True for call in calls)
    assert [row["tokens"]["output_tokens"] for row in rows] == [1, 1]
    assert "private" not in str(rows)


def test_group_failure_drains_other_branch_not_completed():
    drained = threading.Event()
    gate = threading.Barrier(2)
    def create(**kwargs):
        assert kwargs["store"] is True
        index = gate.wait(timeout=2)
        if index == 0:
            return SimpleNamespace(status="cancelled", usage=None)
        drained.set()
        return SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                               output_text="secret", incomplete_details=None)
    with pytest.raises(TargetUnavailable, match="group_response_incomplete"):
        run_group(SimpleNamespace(responses=SimpleNamespace(create=create)), "secret", requests=2,
                  max_output_tokens=2, timeout_seconds=2, commitment=lambda text: "digest")
    assert drained.is_set()


def test_full_cap_is_terminal_but_aborted_work_is_not():
    response = SimpleNamespace(status="incomplete", incomplete_details={"reason": "max_output_tokens"},
                               usage=SimpleNamespace(input_tokens=3, output_tokens=2), output_text="secret")
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: response))
    rows = run_group(client, "secret", requests=2, max_output_tokens=2,
                     timeout_seconds=2, commitment=lambda text: "digest")
    assert all(row["response_status"] == "incomplete" for row in rows)
    response.incomplete_details = {"reason": "cancelled"}
    with pytest.raises(TargetUnavailable):
        run_group(client, "secret", requests=2, max_output_tokens=2,
                  timeout_seconds=2, commitment=lambda text: "digest")


def test_pause_totals_only_disjoint_deltas_missing_stays_unknown(tmp_path):
    from pllm.evidence import _metric

    m = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
                     "inputs": {}, "directory": str(tmp_path)})
    audit_metrics([{"decode_epoch_pause_ns": 20, "decode_epoch_pauses": 1, "inflight_joins": 1},
                   {"decode_epoch_pause_ns": 30, "decode_epoch_pauses": 2, "inflight_joins": 2}], m)
    assert m.details["runtime_audit"]["totals"]["decode_epoch_pauses"] == 3
    assert m.details["runtime_audit"]["totals"]["inflight_joins"] == 3
    assert m.metrics[-1]["value"] == 50 / 1e9
    assert _metric(m.metrics[-1], 0, version=2)["statistic"] == "sum"
    audit_metrics([{"decode_epoch_pause_ns": 99}, {}], m)
    assert m.metrics[-1]["value"] is None
    assert _metric(m.metrics[-1], 0, version=2)["origin"] == "not_available"
    audit_metrics([{"inflight_peak_reserved_bytes": 100}, {"inflight_peak_reserved_bytes": 20}], m)
    assert m.details["runtime_audit"]["totals"]["inflight_peak_reserved_bytes"] is None


@pytest.mark.parametrize("total", [None, 123])
def test_response_covered_bodies_prefer_exact_ledger_unknown_not_invented(tmp_path, total):
    from pllm.evidence import _metric

    m = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
                     "inputs": {}, "directory": str(tmp_path)})
    report = {"runs": [{"durations": {"ttft_seconds": 1, "tokens_per_second": None},
                        "tokens": {"output_tokens": 1}, "privacy": {"decode_epoch_pause_ns": 5}}],
              "summary": {"total_accounted_benchmark_body_bytes": total},
              "communication_per_token": {"summary": {"setup_inclusive_mb_per_output_token": 9,
                                                        "generated_output_tokens": 1}}}
    response_metrics(report, m)
    metrics = {row["id"]: row for row in m.metrics}
    assert metrics["network.covered_body_bytes"]["value"] == total
    assert metrics["response.epoch_pause"]["value"] == 5 / 1e9
    for index, metric in enumerate(m.metrics):
        _metric(metric, index, version=2)


@pytest.mark.parametrize("supported", [True, False])
def test_group_executor_one_ledger_setup_cleanup_and_failure_charge(tmp_path, monkeypatch, supported):
    import pllm.metrics
    import pllm.model_loader
    import pllm.runtime.servers
    from pllm.runtime.client import PrivacyAudit
    from pllm.runtime.benchmark_response_group import execute_group

    audit = PrivacyAudit()
    events = []
    retained_counts = []
    lock = threading.Lock()
    def create(**kwargs):
        assert kwargs["store"] is True
        response = SimpleNamespace(status="completed", usage=SimpleNamespace(input_tokens=3, output_tokens=1),
                                   output_text="never-archive-output", incomplete_details=None)
        with lock:
            audit.prepared_stage_rows_reserved += 1
            audit.prepared_stage_rows_claimed += 1
            audit.inference_upload_bytes += 10
            key = f"private-response-{len(client._core.cache)}"
            client._core.cache[key] = response
            client._core.histories[key] = kwargs["input"]
            client._core._transformer_conversations[key] = ["private-token"]
            retained_counts.append(len(client._core.cache))
        return response
    def close():
        events.append("client-close")
        audit.preparation_control_upload_bytes += 7
        for name in ("cache", "histories", "_transformer_conversations"):
            getattr(client._core, name).clear()
    client = SimpleNamespace(privacy_audit=audit, close=close,
        runtime=SimpleNamespace(capabilities=lambda: events.append("capabilities")),
        prepared_rows_for_response=lambda prompt, cap, **kw: events.append(("sizing", cap, kw["store"])),
        responses=SimpleNamespace(create=create), _core=SimpleNamespace(
            prepared_concurrency_contract=CONCURRENCY_CONTRACT if supported else None,
            cache={}, histories={}, message_histories={}, _pending_response_store={},
            _transformer_states={}, _transformer_conversations={}, _decode_epoch_responses={}))
    class Topology:
        def __enter__(self):
            events.append("roles-start")
            return self
        def client(self, **kwargs):
            audit.bundle_network_bytes += 11
            return client
        def __exit__(self, *args):
            events.append("roles-close")
    monkeypatch.setattr(pllm.runtime.servers, "serve_local", lambda *a, **kw: Topology())
    monkeypatch.setattr(pllm.model_loader, "resolve_model", lambda *a, **kw: SimpleNamespace(
        source_lock_digest="ab" * 32))
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {
        "admitted": True, "candidates": {"native": {"required_host_bytes": 1024}},
        "estimate": memory_estimate()})
    m = Measurement({"sampling": Sampling(processes=1).to_spec(), "salt": "ab" * 32,
                     "inputs": {}, "directory": str(tmp_path)})
    monkeypatch.setattr(m, "admit", lambda required: m.admission.update(admitted=True, required_bytes=required))
    p = BenchmarkTarget.response(experiment(), concurrent_requests=2).to_spec()["parameters"]
    if supported:
        execute_group(p, m, "never-archive-input")
        assert len(m.samples) == 1
        assert len(m.details["report"]["runs"]) == 2
        assert m.details["runtime_audit"]["totals"]["prepared_stage_rows_claimed"] == 2
        assert m.details["report"]["privacy"]["bundle_network_bytes"] == 11
        assert m.details["report"]["privacy"]["preparation_control_upload_bytes"] == 7
        assert m.comparison["workload_digest"] and m.comparison["output_digest"]
        assert m.details["report"]["geometry"]["store"] is True
        assert m.details["report"]["private_cleanup"]["known_private_containers_empty"] is True
        assert retained_counts == [1, 2]
    else:
        with pytest.raises(TargetUnavailable, match="group_runtime_concurrency_contract_unavailable"):
            execute_group(p, m, "never-archive-input")
        assert not m.samples
        assert m.details["group_lifecycle_audit"]["bundle_network_bytes"] == 11
        assert m.details["group_lifecycle"]["includes_cleanup"] is True
    assert events == (["roles-start", "capabilities", ("sizing", 8, True), "client-close", "roles-close"]
                      if supported else ["roles-start", "client-close", "roles-close"])
    assert "never-archive" not in str(m.details)


def memory_estimate(bound=1):
    return {"concurrent_response_bound": bound, "compiled_input_bound": 128, "max_output_tokens": 8,
            "allocation_slack_percent": 25,
            "tensor_storage": "full_kv_bounded_live_roots_v2; completed benchmark response histories are retired",
            "components": {"client_state_work_bytes": 60 * bound, "client_live_tensor_bytes": 100 * bound}}


def test_group_memory_no_alias_budget_or_duplicate_pipeline_concurrency():
    admission = {"candidates": {"native": {"required_host_bytes": 1 << 30}}, "estimate": memory_estimate(8)}
    original = copy.deepcopy(admission)
    priced = group_memory_contract(admission, requests=2, max_output_tokens=8)
    assert admission == original
    assert priced["shared_topology_bytes"] == 1 << 30
    assert priced["unpriced_caller_count"] == 0 and priced["queued_scratch_bytes"] == 0
    assert priced["retained_response_state_bytes"] == 60  # distinct stored history, not owner/cache alias
    assert priced["required_bytes"] < 2 << 30
    assert priced["provider_checkpoints_duplicated"] is False
    admission["estimate"] = memory_estimate(1)
    baseline = group_memory_contract(admission, requests=2, max_output_tokens=8)
    assert baseline["queued_scratch_bytes"] == 100
    assert baseline["retained_response_state_bytes"] == priced["retained_response_state_bytes"]


@pytest.mark.parametrize("change", [
    {"tensor_storage": "unreleased_outputs"}, {"concurrent_response_bound": True},
    {"max_output_tokens": 1}, {"components": {"client_state_work_bytes": 0, "client_live_tensor_bytes": 100}},
])
def test_group_memory_unknown_or_mismatched_geometry_rejected(change):
    admission = {"candidates": {"native": {"required_host_bytes": 1 << 30}},
                 "estimate": {**memory_estimate(), **change}}
    with pytest.raises(TargetUnavailable, match="group_memory_bound_unavailable"):
        group_memory_contract(admission, requests=2, max_output_tokens=8)


def test_private_cleanup_inspects_counts_not_keys_or_results():
    core = SimpleNamespace(cache={"private-token": object()}, histories={}, message_histories={},
                           _pending_response_store={}, _transformer_states={},
                           _transformer_conversations={}, _decode_epoch_responses={})
    with pytest.raises(TargetUnavailable, match="^group_private_cleanup_failed$"):
        inspect_private_cleanup(SimpleNamespace(_core=core))
    core.cache.clear()
    assert inspect_private_cleanup(SimpleNamespace(_core=core))["known_private_containers_empty"] is True


def test_private_owner_cleanup_keeps_only_public_counts():
    from pllm.runtime.inflight_prefix import InflightPrefixOwner
    core = SimpleNamespace(cache={}, histories={}, message_histories={}, _pending_response_store={},
                           _transformer_states={}, _transformer_conversations={}, _decode_epoch_responses={})
    owner = InflightPrefixOwner(core)
    core.owner = owner
    client = SimpleNamespace(_core=core)
    handles = prefix_owners(client)
    assert handles == (owner,)
    owner.close()
    core.owner = None
    result = inspect_private_cleanup(client, handles)
    assert result["owner_counters_after_close"][0]["nodes"] == 0
    assert result["owner_counters_after_close"][0]["reserved_bytes"] == 0
    owner._stats["reserved_bytes"] = 1
    with pytest.raises(TargetUnavailable, match="group_private_owner_cleanup_failed"):
        inspect_private_cleanup(client, handles)
