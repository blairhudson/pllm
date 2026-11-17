"""Launcher diagnostics using synthetic metadata and real memory admission only.

No native source factory, checkpoint, tensors, children or host probes are used.
"""
from types import SimpleNamespace as NS
import json

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import PumaThreePartyCpu
from pllm.runtime import benchmark_memory as memory, replicated_source, servers


GiB = 1 << 30
MESSAGE = "complete topology exceeds safe memory admission"


class BeforeWorkerLaunch(Exception):
    pass


@pytest.fixture
def launch(tmp_path, monkeypatch):
    from pllm import model_loader

    experiment = Experiment("memory-diagnostic-double", PumaThreePartyCpu(Model.tiny()),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(1, 2, 1))
    monkeypatch.setattr(Experiment, "resolve",
        lambda self: NS(model=model_loader.expected_model_id(self.pipeline.model)))
    estimate = {
        "runtime_kind": "replicated", "kernel": "cpu", "allocation_slack_percent": 25,
        "limitations": [], "native_total_peak_bytes": 5 * GiB,
        "client_peak_bytes": 2 * GiB,
        "provider_peak_bytes": {"worker_a": GiB, "worker_b": GiB, "worker_c": GiB},
        "components": {"per_engine_compiled_disk_bytes": 1024, "raw_client_bundle_bytes": 512},
    }
    state = NS(events=[], estimate=estimate, host=None, stop=BeforeWorkerLaunch("before worker launch"))

    def admit_source(exp):
        assert exp is experiment
        state.events.append("source.metadata")
        return NS(config={"metadata_only": True})

    def estimate_memory(config, pipeline, **kwargs):
        assert config == {"metadata_only": True} and pipeline is experiment.pipeline
        assert kwargs == {"max_input_tokens": 2, "max_output_tokens": 1}
        state.events.append("estimate")
        return estimate

    def host_memory():
        state.events.append("host")
        assert state.events.count("host") == 1
        return state.host

    def resolve_model(*args, **kwargs):
        state.events.append("before_worker_launch")
        assert state.topology.memory_admission["admitted"]
        raise state.stop

    def forbidden(*args, **kwargs):
        pytest.fail("child creation/tensor loading must not be reached")

    monkeypatch.setattr(replicated_source, "admit_source", admit_source)
    monkeypatch.setattr(memory, "estimate_memory", estimate_memory)
    monkeypatch.setattr(memory, "host_memory", host_memory)
    monkeypatch.setattr(model_loader, "resolve_model", resolve_model)
    monkeypatch.setattr(servers.LocalTopology, "_spawn", forbidden)
    monkeypatch.setattr(servers.subprocess, "Popen", forbidden)
    state.topology = servers.build_roles(experiment)
    assert isinstance(state.topology, servers.LocalTopology)
    assert state.topology.memory_admission is None
    return state


def host(*, available=12 * GiB, disk_free=32 * GiB, pressure="normal"):
    return memory.HostMemory(24 * GiB, available, 0, disk_free,
        availability_source="macos_mach_dispatch_v2", pressure=pressure,
        free=available, cached=0, purgeable=0, reusable=0)


def assert_report_copy(topology, expected):
    report = topology.memory_admission
    assert report == expected
    report["host"].clear()
    report["candidates"]["native"]["reasons"].append("caller mutation")
    report["estimate"]["provider_peak_bytes"].clear()
    assert topology.memory_admission == expected


@pytest.mark.parametrize("sample", [
    host(available=4 * GiB),
    host(pressure="warning"),
    host(pressure="critical"),
    host(disk_free=2 * GiB),
    memory.HostMemory(24 * GiB, 5 * GiB, 0, 32 * GiB),
], ids=["insufficient-headroom", "warning-pressure", "critical-pressure", "insufficient-disk",
        "reserved-headroom-unknown-pressure"])
def test_refusal_retains_actual_admission_and_bounded_scalar_note(launch, sample):
    state = launch
    state.host = sample
    expected = memory.admit_memory(state.estimate, sample, backend="native")
    assert not expected["admitted"]
    with pytest.raises(servers.TopologyError) as caught:
        state.topology.start()
    error = caught.value
    assert type(error) is servers.TopologyError
    assert str(error) == MESSAGE and error.args == (MESSAGE,)
    assert error.__cause__ is None
    assert len(error.__notes__) == 1
    note = error.__notes__[0]
    assert note.isascii() and len(note) <= 512
    native, observed = expected["candidates"]["native"], expected["host"]
    assert json.loads(note) == {
        "pressure": sample.pressure,
        "native.required_host_bytes": native["required_host_bytes"],
        "host.admission_budget_bytes": observed["admission_budget_bytes"],
        "native.required_disk_bytes": native["required_disk_bytes"],
        "host.disk_free": sample.disk_free,
        "host.available": sample.available,
        "host.reserve_bytes": sample.reserve,
    }
    assert state.events == ["source.metadata", "estimate", "host"]
    assert state.topology.closed and not state.topology._starting
    assert not state.topology._processes
    assert_report_copy(state.topology, expected)


def test_success_retains_admission_before_worker_launch(launch):
    state = launch
    state.host = host()
    expected = memory.admit_memory(state.estimate, state.host, backend="native")
    assert expected["admitted"]
    with pytest.raises(BeforeWorkerLaunch) as caught:
        state.topology.start()
    assert caught.value is state.stop
    assert not getattr(caught.value, "__notes__", [])
    assert state.events == ["source.metadata", "estimate", "host", "before_worker_launch"]
    assert state.topology.closed and not state.topology._starting
    assert not state.topology._processes
    assert_report_copy(state.topology, expected)
