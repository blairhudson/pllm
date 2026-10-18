"""Pressure and reclaimable-memory admission without stressing the host."""

from dataclasses import replace
import json
import platform

import pytest

from pllm import Model
from pllm.deployment.resources import _ClientCapacity
from pllm.deployment import NetworkError
from pllm.runtime import benchmark_memory as memory


def _mac_host(*, pressure="normal", available=12 * memory.GiB):
    return memory.HostMemory(32 * memory.GiB, available, 0, 100 * memory.GiB,
        availability_source="macos_mach_dispatch_v2", pressure=pressure,
        free=available // 4, cached=available - available // 4, purgeable=0, reusable=0)


@pytest.mark.parametrize("pressure", ["warning", "critical"])
@pytest.mark.parametrize("margin", [0, 25])
def test_pressure_rejects_before_inspection_and_aborts_running_owners(monkeypatch, pressure, margin):
    from pllm.profiles import MaskedLinearCpu
    from pllm.sources import _tiny_model_config

    estimate = memory.estimate_memory(_tiny_model_config(), MaskedLinearCpu(Model.tiny()),
                                     max_input_tokens=16, max_output_tokens=2,
                                     memory_safety_margin_percent=margin)
    normal = _mac_host()
    report = memory.admit_memory(estimate, normal)
    assert report["admitted"]
    pressured = replace(normal, pressure=pressure)
    assert not memory.admit_memory(estimate, pressured)["admitted"]
    guard = memory.MemoryWatchdog(report, lambda _: None, sample=lambda: pressured)
    with pytest.raises(memory.BenchmarkMemoryError, match="memory pressure"):
        guard.check()
    monkeypatch.setattr(memory, "host_memory", lambda: pressured)
    with pytest.raises(memory.BenchmarkMemoryError, match="before model inspection"):
        memory.benchmark_memory(object(), memory_safety_margin_percent=margin)
    with pytest.raises(NetworkError, match="CLIENT_MEMORY_PRESSURE"):
        _ClientCapacity().observe()


def test_reclaimable_memory_uses_workload_margin_without_total_ram_reserve(monkeypatch):
    from pllm.metrics import benchmark_memory

    monkeypatch.setattr(memory, "host_memory", _mac_host)
    low = benchmark_memory(Model.tiny(), memory_safety_margin_percent=10)
    high = memory.benchmark_memory(Model.tiny(), memory_safety_margin_percent=50)
    assert low["host"]["reserve_bytes"] == 0
    assert low["host"]["admission_budget_bytes"] == low["host"]["available"]
    assert low["estimate"]["allocation_slack_percent"] == 10
    assert high["estimate"]["allocation_slack_percent"] == 50
    required = low["candidates"]["native"]["required_host_bytes"]
    exact = _mac_host(available=required)
    assert memory.admit_memory(low["estimate"], exact)["admitted"]
    assert not memory.admit_memory(low["estimate"], _mac_host(available=required - 1))["admitted"]
    assert not memory.admit_memory(high["estimate"], exact)["admitted"]
    assert not memory.admit_memory(low["estimate"], exact, memory_budget_bytes=required - 1)["admitted"]
    margin = low["candidates"]["native"]["safety_margin_bytes"]
    assert low["runtime_floor_bytes"] == margin > 0
    guard = memory.MemoryWatchdog(low, lambda _: None, sample=lambda: _mac_host(available=margin))
    guard.check()
    guard.sample = lambda: _mac_host(available=margin - 1)
    with pytest.raises(memory.BenchmarkMemoryError, match="runtime reserve"):
        guard.check()


@pytest.mark.parametrize("margin", [-1, 101, True, False, 1.5, "25"])
def test_invalid_margin_rejects_before_os_or_source_access(monkeypatch, margin):
    monkeypatch.setattr(memory, "host_memory", lambda: pytest.fail("sampled before validation"))
    with pytest.raises(ValueError, match="safety margin"):
        memory.benchmark_memory(object(), memory_safety_margin_percent=margin)


def test_macos_sampling_ignores_psutil_used_and_inactive(monkeypatch):
    import psutil
    from pllm import _native

    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(psutil, "virtual_memory", lambda: pytest.fail("used psutil Darwin availability"))
    sample = {"total": 32 * memory.GiB, "free": memory.GiB, "cached": 10 * memory.GiB,
              "purgeable": memory.GiB, "reusable": 0,
              "available": 12 * memory.GiB, "pressure": "normal"}
    monkeypatch.setattr(_native, "macos_host_memory", lambda: json.dumps(sample).encode())
    host = memory.host_memory()
    assert host.available == sum(sample[key] for key in ("free", "cached", "purgeable"))
    assert host.reserve == 0
    resources = _ClientCapacity().observe()
    assert resources.headroom_bytes == host.available
    monkeypatch.setattr(_native, "macos_host_memory", lambda: b'{"pressure":"unknown"}')
    with pytest.raises(memory.BenchmarkMemoryError, match="cannot sample"):
        memory.host_memory()


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mach/libdispatch")
def test_live_native_observation():
    host = memory.host_memory()
    assert host.availability_source == "macos_mach_dispatch_v2"
    assert host.pressure in {"normal", "warning", "critical"}
    assert host.available == host.free + host.cached + max(host.purgeable, host.reusable)
    assert 0 <= host.available <= host.total


def test_allocator_reusable_capacity_reaches_sdk_and_benchmark(monkeypatch):
    from pllm import _native

    monkeypatch.setattr(memory.platform, "system", lambda: "Darwin")
    raw = {"total": 32 * memory.GiB, "free": memory.GiB, "cached": 6 * memory.GiB,
           "purgeable": memory.GiB, "reusable": 5 * memory.GiB,
           "available": 12 * memory.GiB, "pressure": "normal"}
    monkeypatch.setattr(_native, "macos_host_memory", lambda: json.dumps(raw).encode())
    host = memory.host_memory()
    assert host.reusable == 5 * memory.GiB
    assert host.reserve == 0
    assert _ClientCapacity().observe().headroom_bytes == host.available == 12 * memory.GiB
    report = memory.benchmark_memory(Model.tiny(), memory_safety_margin_percent=0)
    assert report["admitted"]
    assert report["host"]["reusable"] == 5 * memory.GiB
    assert report["host"]["availability_source"] == "macos_mach_dispatch_v2"
    # Summing these potentially overlapping anonymous counters would fabricate
    # another GiB of capacity. Reject such observations, including on SDK entry.
    raw["available"] = 13 * memory.GiB
    with pytest.raises(memory.BenchmarkMemoryError, match="telemetry"):
        memory.host_memory()
    with pytest.raises(NetworkError, match="CLIENT_MEMORY_OBSERVATION_UNAVAILABLE"):
        _ClientCapacity().observe()


def test_historical_macos_observations_keep_their_original_semantics():
    host = replace(_mac_host(), availability_source="macos_mach_dispatch_v1", reusable=None)
    assert host.reserve == 0
    assert host.available == host.free + host.cached + host.purgeable
    with pytest.raises(memory.BenchmarkMemoryError, match="telemetry"):
        replace(host, reusable=memory.GiB)


@pytest.mark.parametrize("margin", [0, 40])
def test_cli_preflight_prices_selected_margin_without_launch(monkeypatch, capsys, margin):
    from pllm.cli import main

    monkeypatch.setattr(memory, "host_memory", _mac_host)
    main(["benchmark", "run", "--tiny", "--backend", "native", "--preflight-only",
          "--memory-safety-margin-percent", str(margin), "--format", "json"])
    report = json.loads(capsys.readouterr().out)["data"]["candidates"][0]
    assert report["estimate"]["allocation_slack_percent"] == margin
    assert report["host"]["reserve_bytes"] == 0
    assert report["runtime_floor_bytes"] == report["candidates"]["native"]["safety_margin_bytes"]
    if margin == 0:
        assert report["runtime_floor_bytes"] == 0
        native = report["candidates"]["native"]
        required = native["required_host_bytes"]
        assert required == native["expected_peak_bytes"]
        assert memory.admit_memory(report["estimate"], _mac_host(available=required))["admitted"]
        assert not memory.admit_memory(report["estimate"], _mac_host(available=required - 1))["admitted"]
