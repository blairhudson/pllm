from __future__ import annotations

from dataclasses import replace
import json

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import AppleMetal
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.runtime import benchmark_memory as memory
from pllm.runtime.benchmark_memory import BenchmarkMemoryError, GiB, HostMemory, MiB
from pllm.sources import _tiny_model_config


def _estimate(pipeline=None, config=None):
    return memory.estimate_memory(config or _tiny_model_config(), pipeline or MaskedLinearCpu(Model.tiny()),
                                  max_input_tokens=16, max_output_tokens=2, inventory_rows=64)


def _host(**changes):
    return replace(HostMemory(32 * GiB, 24 * GiB, 0, 100 * GiB), **changes)


def test_estimate_prices_complete_graph_and_native_copies():
    value = _estimate()
    assert set(value["provider_peak_bytes"]) == {"inference", "preparation"}
    assert value["components"]["per_engine_i8_and_native_bytes"] > 0
    assert value["native_total_peak_bytes"] == value["client_peak_bytes"] + sum(value["provider_peak_bytes"].values())
    assert value["components"]["legacy_client_mask_bytes"] > value["components"]["provider_correction_bytes"] > 0
    assert 0 < value["components"]["client_mask_bytes"] < value["components"]["legacy_client_mask_bytes"]
    assert value["components"]["client_mask_cursor_bytes"] > 0
    assert value["components"]["client_tensor_work_bytes"] > 0
    residency = value["components"]["per_role_i8_and_native_bytes"]
    assert residency["preparation"] < residency["inference"] < value["components"]["per_engine_i8_and_native_bytes"]
    assert all(n > 0 for n in value["components"]["per_role_loading_temporary_bytes"].values())
    json.dumps(value)  # every report field is a scalar/record, not a plan handle
    offset = _estimate(TwoOnlineOffsetCpu(Model.tiny()))
    assert set(offset["provider_peak_bytes"]) == {"worker_a", "worker_b"}
    assert offset["components"]["client_mask_bytes"] == 0


def test_paged_client_prices_bounded_ram_and_distinct_disk_owners():
    from pllm.protocols import ClientBundleTransport
    resident = _estimate(MaskedLinearCpu(Model.tiny(), delivery=ClientBundleTransport("artifacts")))
    paged = _estimate(MaskedLinearCpu(Model.tiny(), delivery=ClientBundleTransport("artifacts", storage="paged")))
    assert paged["client_peak_bytes"] < resident["client_peak_bytes"]
    assert paged["provider_peak_bytes"] == resident["provider_peak_bytes"]
    assert paged["docker_provider_peak_bytes"] == resident["docker_provider_peak_bytes"]
    assert paged["components"]["raw_client_bundle_bytes"] == resident["components"]["raw_client_bundle_bytes"]
    assert paged["components"]["client_bundle_work_bytes"] < resident["components"]["client_bundle_work_bytes"]
    a = memory.admit_memory(resident, _host())["candidates"]["native"]
    b = memory.admit_memory(paged, _host())["candidates"]["native"]
    assert b["required_disk_bytes"] - a["required_disk_bytes"] == paged["components"]["client_additional_paged_disk_bytes"] > 0


@pytest.mark.parametrize("storage", [None, object(), bytearray(b"\x00"), b"\x00"])
def test_snapshot_estimate_rejects_reference_stale_or_mutable_backend(monkeypatch, storage):
    from types import SimpleNamespace
    from pllm.runtime import _native_support
    backend = None if storage is None else SimpleNamespace(Matrix=lambda *_: storage)
    monkeypatch.setattr(_native_support, "extension", lambda: backend)
    with pytest.raises(BenchmarkMemoryError, match="native snapshot storage"):
        _estimate()


def test_large_dense_model_rejected_on_32gib_before_any_values(monkeypatch):
    config = _tiny_model_config() | {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        "vocab_size": 151936, "attention_bias": False, "attention_dropout": 0.0,
        "max_position_embeddings": 40960, "use_sliding_window": False,
        "sliding_window": None, "max_window_layers": 36, "use_cache": True}
    value = _estimate(config=config)
    assert value["components"]["per_engine_i8_and_native_bytes"] > 4 * GiB
    assert value["components"]["legacy_per_engine_i8_and_native_bytes"] > 8 * GiB
    assert value["components"]["float_quantization_work_bytes"] <= 28 * MiB
    assert all(value["docker_provider_peak_bytes"][role] > value["provider_peak_bytes"][role]
               for role in value["provider_peak_bytes"])
    # A 32 GiB machine can admit a sufficiently small workload when otherwise
    # idle. This occupied-host fixture must still reject before tensor loading.
    report = memory.admit_memory(value, _host(available=18 * GiB), backend="auto", docker_capacity=(20 * GiB, 0))
    assert report["selected_backend"] is None
    assert not any(candidate["admitted"] for candidate in report["candidates"].values())
    with pytest.raises(BenchmarkMemoryError, match="exceeds safe headroom"):
        memory.require_admission(report)


def test_docker_vm_headroom_and_native_fallback_do_not_change_pipeline():
    estimate = _estimate()
    report = memory.admit_memory(estimate, _host(), backend="auto", docker_capacity=(GiB, 0))
    assert report["selected_backend"] == "native"
    assert report["estimate"]["pipeline_digest"] == estimate["pipeline_digest"]
    wan = memory.admit_memory(estimate, _host(), backend="auto", docker_capacity=(GiB, 0), enforced_wan=True)
    assert wan["selected_backend"] is None
    assert "enforced WAN" in wan["candidates"]["native"]["reasons"][0]
    occupied = memory.admit_memory(estimate, _host(), backend="docker", docker_capacity=(8 * GiB, 7 * GiB))
    assert not occupied["admitted"]


def test_swap_and_explicit_budget_cannot_increase_available_ram():
    estimate = _estimate()
    for budget in (None, 100 * GiB):
        report = memory.admit_memory(estimate, _host(available=8 * GiB, swap_used=64 * GiB),
                                     memory_budget_bytes=budget)
        assert not report["admitted"]
        assert report["host"]["admission_budget_bytes"] == 0
    assert not memory.admit_memory(estimate, _host(), memory_budget_bytes=MiB)["admitted"]
    with pytest.raises(ValueError, match="positive"):
        memory.admit_memory(estimate, _host(), memory_budget_bytes=True)


def test_docker_capacity_uses_bytes_instead_of_container_relative_percent(monkeypatch):
    from types import SimpleNamespace
    replies = iter((SimpleNamespace(stdout=str(8 * GiB)),
                    SimpleNamespace(stdout="200.0MiB / 400MiB\n1.50GB / 2GB\n")))
    monkeypatch.setattr(memory.subprocess, "run", lambda *_a, **_k: next(replies))
    total, used = memory._docker_capacity()
    assert total == 8 * GiB
    assert 200 * MiB + 1_500_000_000 < used < 210 * MiB + 1_520_000_000
    with pytest.raises(ValueError, match="unknown Docker memory unit"):
        memory._docker_usage_bytes("nanGiB")


def test_metal_prices_unified_gpu_copies_and_only_selects_native(monkeypatch):
    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(memory.platform, "machine", lambda: "arm64")
    cpu = _estimate()
    gpu = _estimate(MaskedLinearCpu(Model.tiny(), kernels=AppleMetal(min_rows=2)))
    assert gpu["components"]["per_provider_metal_bytes"] > 0
    assert gpu["native_total_peak_bytes"] > cpu["native_total_peak_bytes"]
    report = memory.admit_memory(gpu, _host(), backend="auto", docker_capacity=(8 * GiB, 0))
    assert report["selected_backend"] == "native"
    assert not report["candidates"]["docker"]["admitted"]


def test_header_only_probe_does_not_materialize_tiny_weights(monkeypatch):
    import pllm.sources
    monkeypatch.setattr(pllm.sources, "_materialize_tiny_model", lambda *a: pytest.fail("allocated tensors"))
    monkeypatch.setattr(memory, "host_memory", _host)
    experiment = Experiment("memory", MaskedLinearCpu(Model.tiny()), Deployment.local(root="local://memory"),
                            ExecutionBudget(requests=1, max_input_tokens=16, max_new_tokens=2))
    report = memory.benchmark_memory(experiment)
    assert report["admitted"] and report["estimate"]["max_input_tokens"] == 16


def test_native_probe_does_not_contact_docker(monkeypatch):
    monkeypatch.setattr(memory, "host_memory", _host)
    monkeypatch.setattr(memory, "_docker_capacity", lambda: pytest.fail("contacted Docker"))
    assert memory.benchmark_memory(Model.tiny(), backend="native")["admitted"]


def test_cache_compilation_bound_is_priced_separately_from_request_limit(monkeypatch):
    monkeypatch.setattr(memory, "host_memory", _host)
    narrow = memory.benchmark_memory(Model.tiny(), max_input_tokens=16)["estimate"]
    cached = memory.benchmark_memory(Model.tiny(), max_input_tokens=16, cache_bytes=MiB,
                                     cache_bound_tokens=256)["estimate"]
    assert cached["max_input_tokens"] == 16
    assert cached["compiled_input_bound"] == 256
    assert cached["client_peak_bytes"] > narrow["client_peak_bytes"] + MiB


@pytest.mark.parametrize("count", [0, False, -1])
def test_preflight_does_not_turn_invalid_explicit_counts_into_defaults(monkeypatch, count):
    monkeypatch.setattr(memory, "host_memory", _host)
    with pytest.raises(BenchmarkMemoryError, match="bounded positive"):
        memory.benchmark_memory(Model.tiny(), max_input_tokens=count)


def test_watchdog_stops_on_pressure_and_new_swap_after_recovery():
    report = memory.admit_memory(_estimate(), _host(swap_used=10 * GiB))
    samples = iter((_host(swap_used=2 * GiB), _host(swap_used=2 * GiB + 257 * MiB)))
    guard = memory.MemoryWatchdog(report, lambda reason: None, sample=lambda: next(samples))
    guard.check()
    with pytest.raises(BenchmarkMemoryError, match="swap grew"):
        guard.check()
    guard = memory.MemoryWatchdog(report, lambda reason: None, sample=lambda: _host(available=3 * GiB))
    with pytest.raises(BenchmarkMemoryError, match="runtime reserve"):
        guard.check()


@pytest.mark.asyncio
async def test_dashboard_rejection_precedes_role_creation(monkeypatch):
    from pllm.runtime import dashboard
    monkeypatch.setattr(memory, "host_memory", lambda: _host(available=9 * GiB))
    monkeypatch.setattr(dashboard, "build_roles", lambda *a, **k: pytest.fail("started provider"))
    config = dashboard.DashboardConfig(host="127.0.0.1", port=9876, model_path=None,
        model_id="tiny", default_max_output_tokens=2, history_path=":memory:", memory_budget_bytes=MiB)
    runtime = dashboard.DashboardRuntime(config, dashboard.OTelStore())
    await runtime.start()
    assert not runtime.memory_preflight["admitted"]
    assert runtime._topology is None
    assert "memory preflight rejected" in runtime.snapshot()["run"]["error"]
    await runtime.stop()


def test_cli_preflight_only_and_insufficient_budget(tmp_path):
    import subprocess
    import sys
    output = tmp_path / "admission.json"
    result = subprocess.run([sys.executable, "-m", "pllm", "benchmark", "run", "--tiny",
        "--backend", "native", "--memory-budget-mib", "1", "--preflight-only", "--output", str(output)],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["preflight_only"] and not report["admitted"]
    assert report["candidates"][0]["host"]["admission_budget_bytes"] == MiB


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["auto", "docker"])
async def test_selected_docker_backend_receives_legacy_safe_limits(monkeypatch, backend):
    from pllm.runtime import dashboard
    estimate = _estimate()
    admission = memory.admit_memory(estimate, _host(), backend=backend, docker_capacity=(8 * GiB, 0))
    assert admission["selected_backend"] == "docker"
    monkeypatch.setattr(memory, "benchmark_memory", lambda *a, **k: admission)
    captured = {}
    def build(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after observing admitted role configuration")
    monkeypatch.setattr(dashboard, "build_roles", build)
    runtime = dashboard.DashboardRuntime(dashboard.DashboardConfig(
        host="127.0.0.1", port=9876, model_path=None, model_id="tiny",
        default_max_output_tokens=2, history_path=":memory:", backend=backend), dashboard.OTelStore())
    try:
        await runtime.start()
        assert captured["docker"]
        assert captured["memory_limits"] == estimate["docker_provider_peak_bytes"]
        assert captured["memory_limits"] != estimate["provider_peak_bytes"]
    finally:
        await runtime.stop()


def test_watchdog_abort_callback_is_once_and_retires_thread():
    import threading
    report = memory.admit_memory(_estimate(), _host())
    called, reasons = threading.Event(), []
    samples = iter((_host(), _host(available=GiB)))
    def abort(reason):
        reasons.append(reason)
        called.set()
    guard = memory.MemoryWatchdog(report, abort, sample=lambda: next(samples))
    guard.start()
    assert called.wait(2)
    guard.close()
    assert len(reasons) == 1 and "runtime reserve" in reasons[0]
    assert not guard._thread.is_alive()


@pytest.mark.integration
@pytest.mark.skipif(memory.platform.system() != "Darwin" or memory.platform.machine() != "arm64",
                    reason="native Apple Metal")
def test_native_cpu_metal_benchmark_fallback_preserves_output(monkeypatch):
    pytest.importorskip("mlx.core")
    from pathlib import Path
    import runpy
    targets = runpy.run_path(str(Path(__file__).parents[1] / "examples/benchmarks/memory_safety.py"))
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    from pllm.runtime.dashboard import DashboardRuntime
    errors = []
    finish = DashboardRuntime._finish_run
    def collect_error(self, capture, status, error):
        if error is not None:
            errors.append(f"{type(error).__name__}: {error}")
        return finish(self, capture, status, error)
    monkeypatch.setattr(DashboardRuntime, "_finish_run", collect_error)
    monkeypatch.setattr(memory, "_docker_capacity", lambda: (GiB, 0))
    reports = []
    for experiment in (targets["cpu"], targets["metal"]):
        try:
            report = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
                prompt="Hi", max_output_tokens=2, warmups=0, repetitions=1, timeout_seconds=90,
                experiment=experiment, backend="auto", temperature=0, capture_output_digest=True,
                _cohort_salt=b"memory-safety-matched-test")
        except RuntimeError as exc:
            pytest.fail(str(errors) if errors else str(exc))
        assert report["configuration"]["provider_backend"] == "native"
        assert report["checks"]["passed"]
        assert report["memory_preflight"]["selected_backend"] == "native"
        assert not report["memory_guard"]["tripped"]
        reports.append(report)
    assert (reports[0]["runs"][0]["generation"]["output_text_digest"]
            == reports[1]["runs"][0]["generation"]["output_text_digest"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_runtime_pressure_abort_retires_real_native_roles(monkeypatch):
    import asyncio
    import threading
    import time
    import psutil
    from pllm.runtime.dashboard import DashboardConfig, DashboardRuntime, OTelStore
    trip = threading.Event()
    sample = memory.host_memory
    def pressure():
        current = sample()
        return HostMemory(current.total, 1, current.swap_used, current.disk_free) if trip.is_set() else current
    monkeypatch.setattr(memory, "host_memory", pressure)
    runtime = DashboardRuntime(DashboardConfig(host="127.0.0.1", port=9876, model_path=None,
        model_id="memory-abort", default_max_output_tokens=2, history_path=":memory:",
        backend="native", memory_input_tokens=32), OTelStore())
    try:
        await runtime.start()
        assert runtime.memory_preflight["admitted"]
        assert runtime._topology is not None and runtime._topology.started
        pids = [status.pid for status in runtime._topology.statuses]
        assert len(pids) == 2 and all(pid is not None for pid in pids)
        trip.set()
        deadline = time.monotonic() + 10
        while any(psutil.pid_exists(pid) for pid in pids) and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert not any(psutil.pid_exists(pid) for pid in pids)
        assert "runtime reserve" in runtime._memory_guard.error
    finally:
        await runtime.stop()
    assert not runtime._memory_guard._thread.is_alive()
