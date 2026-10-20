"""Authenticated counter snapshots survive dashboard deltas and history storage."""

import json
import time
from contextlib import closing
from types import SimpleNamespace as NS

import httpx
import pytest

from pllm.runtime.benchmark_delegated import FSS_PHASE_COUNTERS, fss_phase_timing
from pllm.runtime.benchmark_history import BenchmarkHistory
from pllm.runtime.dashboard import DashboardConfig, DashboardRuntime, OTelStore, _RunCapture
from pllm.runtime.servers import LocalTopology


ROLES = ("worker_a", "worker_b", "dealer")


def counters(**changes):
    return dict.fromkeys(FSS_PHASE_COUNTERS, 0) | changes


@pytest.mark.parametrize("sample", ["complete", "missing", "reset", "unknown_field"])
def test_worker_to_dashboard_to_history_phase_measurements(monkeypatch, sample):
    before = {
        "dealer": counters(preparation_calls=2, preparation_ns=1000,
                           delivery_callback_ns=130, seal_callback_ns=80),
        "worker_a": counters(online_calls=2, online_ns=300),
        "worker_b": counters(online_calls=2, online_ns=400),
    }
    after = {
        "dealer": counters(preparation_calls=3, preparation_ns=1300,
                           delivery_callback_ns=160, seal_callback_ns=130),
        "worker_a": counters(online_calls=3, online_ns=400),
        "worker_b": counters(online_calls=3, online_ns=530),
    }
    if sample == "missing":
        before["worker_a"] = after["worker_a"] = None
    elif sample == "reset":
        after["worker_a"] = counters(online_calls=1, online_ns=100)
    elif sample == "unknown_field":
        after["worker_a"]["private_prompt"] = "must never enter history"
    samples = before

    def get(url, *, headers, timeout, trust_env):
        request = httpx.Request("GET", url)
        role = request.url.host
        assert request.url.path == "/v1/fss/metrics"
        assert headers == {"authorization": f"Bearer credential-{role}"}
        assert timeout == 2.0 and trust_env is False
        return httpx.Response(200, request=request, json={
            "schema": "pllm.fss.worker.metrics.v1", "cpu_ns": 100,
            "peak_rss_bytes": 1000, "phase_timing": samples[role],
        })

    monkeypatch.setattr("pllm.runtime.servers.httpx.get", get)
    topology = NS(_privacy_mode="fss_public", requires_preparation=False,
                  statuses=tuple(NS(role=role, running=True) for role in ROLES),
                  worker_connections=lambda: {
                      role: (f"http://{role}", f"credential-{role}") for role in ROLES})
    topology.worker_process_metrics = lambda: LocalTopology.worker_process_metrics(topology)
    initial = topology.worker_process_metrics()
    samples = after
    with closing(BenchmarkHistory(":memory:")) as history:
        config = DashboardConfig(host="127.0.0.1", port=0, model_path=None,
                                 model_id="generated-phase", default_max_output_tokens=2)
        runtime = DashboardRuntime(config, OTelStore(), history)
        runtime._delegated = runtime._fss = True
        runtime._worker_roles = ROLES
        runtime._topology = topology
        runtime._state["output_delivery"] = "buffered_final_ids"
        capture = _RunCapture(
            run_id="timing", max_output_tokens=2, cold=False,
            started_at_ns=time.time_ns(), audit_before={},
            started_monotonic_ns=time.monotonic_ns(), protocol_start_cursor=0,
            input_tokens=1, output_tokens=2, token_usage_authoritative=True,
            worker_metrics_before=initial,
        )
        runtime._finish_run(capture, status="completed", error=None)
        assert runtime._state["error"] is None
        saved = history.get("timing")

    assert saved["status"] == "completed"
    assert saved["processes"]["dealer"]["phase_timing"] == counters(
        preparation_calls=1, preparation_ns=300, delivery_callback_ns=30, seal_callback_ns=50)
    assert saved["processes"]["worker_b"]["phase_timing"] == counters(online_calls=1, online_ns=130)
    timing = fss_phase_timing(saved)
    assert timing["preparation_seconds"] == 300 / 1e9
    assert timing["preparation_outside_callbacks_seconds"] == 220 / 1e9
    assert timing["online_seconds"] == (130 / 1e9 if sample == "complete" else None)
    assert saved["durations"]["online_seconds"] is None
    assert "must never enter history" not in json.dumps(saved)
    assert "credential-" not in json.dumps(saved)
    if sample != "complete":
        assert "phase_timing" not in saved["processes"]["worker_a"]
