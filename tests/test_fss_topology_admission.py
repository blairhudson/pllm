"""Source qualification and ordinary launcher ordering, with explicit native doubles.

No checkpoint tensors, material tapes, children, or network servers are created.
"""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import FuseFssSourceCpu, MaskedLinearCpu
from pllm.runtime import benchmark_memory as memory, fss_source, servers

GiB = 1 << 30
PROFILE = "fusefss.dense-rope-ring50-q12.v1"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from pllm import _native
    from pllm.deployment import resources
    from pllm.model_loader import expected_model_id

    experiment = Experiment("qualification-double", FuseFssSourceCpu(
        Model.path(str(tmp_path / "source")), numeric_profile=PROFILE, capacity="decoder"),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(2, 2, 3))
    monkeypatch.setattr(Experiment, "resolve", lambda self: NS(model=expected_model_id(self.pipeline.model)))
    contract = {"schema_version": "pllm.fss.sdk-contract.v1", "runtime_kind": "fss",
        "capacity": "decoder", "numeric_profile": PROFILE, "ring_bits": 50, "fractional_bits": 12,
        "input_rows": 2, "output_tokens": 3, "vocabulary": 7, "width": 8,
        "plan": {"config_digest": "a" * 64}, "source_float32_bytes": GiB,
        "source_q12_value_bytes": 2 * GiB, "source_disk_bytes": 2 * GiB,
        "source_resident_bound_bytes": 8 << 20,
        "admission": {"source_minimum_owner_bytes": 1024, "numeric": "not_performed", "program_resources": None}}
    pins = {name: str(index) * 64 for index, name in enumerate(("checkpoint", "source_lock", "tokenizer", "config"), 1)}
    source = NS(contract=contract, contract_bytes=json.dumps(contract).encode(),
        config_bytes=b'{"model_type":"qwen2"}', snapshot_pins=pins)
    state = NS(experiment=experiment, source=source, events=[], leases={}, loaded=None, topology=None,
        free=20 * GiB, fail_import=False, fail_price=False, fail_spawn=False, fail_stop=False,
        on_price=None, after_close=None, fail_graph_reservation=False)

    def runtime_contract(exp, rows, outputs):
        assert exp.canonical_bytes() == experiment.canonical_bytes()
        assert (rows, outputs) == (2, 3)
        return source.contract_bytes
    source.runtime_contract = runtime_contract

    def resolve(exp):
        assert exp is experiment
        state.events.append("source.metadata")
        return source
    monkeypatch.setattr(fss_source, "resolve_client_source", resolve)
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(24 * GiB, 20 * GiB, 0, state.free))
    monkeypatch.setattr("shutil.disk_usage", lambda path: NS(total=512 * GiB, free=state.free))

    @contextmanager
    def reserve(kind, amount):
        assert type(amount) is int and amount > 0
        phase = "graph" if state.loaded is not None and state.loaded._closed else "qualification"
        label = f"{phase}.{kind}"
        if state.fail_graph_reservation and label == "graph.disk":
            raise ValueError("disk capacity changed before graph reservation")
        state.events.append(label + ".enter")
        assert label not in state.leases
        state.leases[label] = amount
        try:
            yield
        finally:
            state.events.append(label + ".exit")
            del state.leases[label]

    monkeypatch.setattr(resources, "reserve_incremental_memory", lambda amount: reserve("memory", amount))
    def disk(amount, *, directory, minimum_free_bytes):
        assert directory.is_dir() and minimum_free_bytes == 2 * GiB
        return reserve("disk", amount)
    monkeypatch.setattr(resources, "reserve_incremental_disk", disk)

    manifest = {"schema": "pllm.fss.channel.program.v1", "contract": hashlib.sha256(source.contract_bytes).hexdigest(),
        "program": "4" * 64, "source": "5" * 64, "input_digest": "6" * 64}
    pricing = {"schema": "pllm.fss.program_transport_admission.v1", "capacity": "decoder",
        "scope": "program-pricing-not-request-admission", "numeric_profile": PROFILE,
        "contract_digest": manifest["contract"], "program_digest": manifest["program"],
        "source_digest": manifest["source"], "input_digest": manifest["input_digest"],
        "initial_tokens": 2, "selections": 3, "vocabulary": 7, "largest_packet_bytes": 512,
        "cumulative_packet_bytes_per_party_bound": 5000 * 512, "combined_reservation_bytes": 2048,
        "client_peak_bytes": 128, "peer_frame_bytes_bound": 1024, "kernels": 5000,
        "primitive_requests_per_party": 5000, "callback_metadata_bytes_bound": 1024,
        "input_frame_bytes": 128, "output_frame_bytes": 128, "guard_disk_bytes": 8192,
        "guard_reservation_bytes": 1024}
    program_metadata = {"schema": "pllm.fss.program_resources.v1", "program_digest": manifest["program"],
        "source_digest": manifest["source"], "original_input_digest": manifest["input_digest"],
        "numeric_profile": PROFILE, "initial_tokens": 2, "selections": 3, "vocabulary": 7}

    def load(exp, *, expected_pins, source, cancelled, lifecycle=None, on_failure=None):
        assert exp is experiment and source is state.source and expected_pins == pins
        if lifecycle is not None or on_failure is not None:
            assert callable(lifecycle) and callable(on_failure)
        assert not cancelled()
        assert state.leases == {"qualification.memory": 384 << 20, "qualification.disk": 2 * GiB}
        assert state.topology.memory_admission["estimate"]["native_total_peak_bytes"] > 384 << 20
        state.events.append("source.import")
        if state.fail_import:
            raise ValueError("native numeric/range rejection")
        def close():
            assert set(state.leases) == {"qualification.memory", "qualification.disk"}
            state.events.append("source.close")
            if state.after_close is not None:
                state.after_close()
        owner = NS(close=close, metadata=lambda: json.dumps(program_metadata).encode())
        state.loaded = fss_source.LoadedFssSource(source, source.contract_bytes, owner,
                                                json.dumps(manifest).encode(), b"{}")
        return state.loaded
    monkeypatch.setattr(fss_source, "load_role_source", load)

    def native_price(program, raw):
        assert program is state.loaded.native_program and not state.loaded._closed
        assert raw == source.contract_bytes
        state.events.append("source.price")
        if state.fail_price:
            raise ValueError("native transport rejection")
        if state.on_price is not None:
            state.on_price()
        return json.dumps(pricing).encode()
    monkeypatch.setattr(_native, "fss_transport_admission", native_price, raising=False)

    def spawn(self, role, command):
        assert state.loaded._closed and self._source_pins == pins
        assert set(state.leases) == {"graph.memory", "graph.disk"}
        assert self.memory_admission["admitted"] and self.memory_admission["full_topology_admitted"]
        state.events.append("spawn." + role)
        if state.fail_spawn:
            raise ValueError("child creation failed")
        proc = NS(dead=False, pid=100 + len(self._processes))
        proc.poll = lambda: 0 if proc.dead else None
        self._processes[role] = proc
        return proc
    monkeypatch.setattr(servers.LocalTopology, "_spawn", spawn)
    monkeypatch.setattr(servers.LocalTopology, "_wait", lambda *args: None)
    monkeypatch.setattr(servers.LocalTopology, "_free_port", staticmethod(lambda excluded: 18000 + len(excluded)))
    def stop(processes):
        state.events.append("children.stop")
        if state.fail_stop:
            raise ValueError("child failed to retire")
        for proc in processes:
            proc.dead = True
    monkeypatch.setattr(servers.LocalTopology, "_stop_all", staticmethod(stop))
    return state


def build(state, **kwargs):
    state.topology = servers.build_roles(state.experiment, **kwargs)
    return state.topology


def test_execution_phase_keeps_full_memory_and_watchdog_floor(setup):
    kwargs = {"memory_budget_bytes": 12 * GiB, "memory_safety_margin_percent": 40, "enforced_wan": True}
    ceiling = memory.benchmark_memory(setup.experiment, **kwargs)
    phase = memory.execution_memory_admission(setup.experiment, **kwargs)
    assert not ceiling["admitted"] and phase["admitted"]
    assert phase["admission_scope"] == "source_qualification_only" and not phase["full_topology_admitted"]
    assert phase["candidates"]["native"]["required_disk_bytes"] == 4 * GiB
    for field in ("required_host_bytes", "expected_peak_bytes", "safety_margin_bytes"):
        assert phase["candidates"]["native"][field] == ceiling["candidates"]["native"][field]
    assert phase["runtime_floor_bytes"] == ceiling["runtime_floor_bytes"]
    for field in ("client_peak_bytes", "provider_peak_bytes", "native_total_peak_bytes", "allocation_slack_percent"):
        assert phase["estimate"][field] == ceiling["estimate"][field]
    assert setup.events == ["source.metadata", "source.metadata"]


@pytest.mark.parametrize("kind", ["model", "tiny", "pipeline", "capsule"])
def test_execution_helper_preserves_generic_benchmark_mock_and_return(kind, monkeypatch, tmp_path):
    from pllm import metrics
    from pllm.sources import TinyModel
    model = {"model": Model.path(str(tmp_path)), "tiny": TinyModel(),
        "pipeline": MaskedLinearCpu(Model.path(str(tmp_path))),
        "capsule": Experiment("capsule", FuseFssSourceCpu(Model.path(str(tmp_path)), numeric_profile=PROFILE),
            Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(1, 2, 1))}[kind]
    calls, result = [], {"admitted": False}
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *a, **k: calls.append((a, k)) or result)
    assert memory.execution_memory_admission(model, memory_budget_bytes=123, memory_safety_margin_percent=42) is result
    assert calls[0][0] == (model,) and calls[0][1]["memory_budget_bytes"] == 123
    assert calls[0][1]["memory_safety_margin_percent"] == 42


def test_capsule_start_keeps_metadata_gate_without_qualification(setup, monkeypatch):
    decoder = setup.experiment
    capsule = Experiment(decoder.name, FuseFssSourceCpu(decoder.pipeline.model, numeric_profile=PROFILE),
                         decoder.deployment, decoder.budget)
    contract = {key: value for key, value in setup.source.contract.items()
                if key not in {"capacity", "source_disk_bytes", "source_resident_bound_bytes"}}
    contract.update(source_float32_bytes=224, source_q12_value_bytes=448)
    estimate = memory._fss_memory({}, capsule.pipeline, 2, 3, 0, None, 224, contract=contract)
    monkeypatch.setattr(fss_source, "admit_source", lambda exp: setup.source)
    monkeypatch.setattr(memory, "estimate_memory", lambda *args, **kwargs: estimate)
    def spawn(self, role, command):
        assert self.memory_admission is None and self.source_qualification is None
        assert self._source_pins == setup.source.snapshot_pins and not setup.leases
        setup.events.append("spawn." + role)
        proc = NS(dead=False, pid=100 + len(self._processes))
        proc.poll = lambda: 0 if proc.dead else None
        self._processes[role] = proc
        return proc
    monkeypatch.setattr(servers.LocalTopology, "_spawn", spawn)
    topology = servers.build_roles(capsule).start()
    assert setup.loaded is None and all(event.startswith("spawn.") for event in setup.events)
    topology.close()


@pytest.mark.parametrize("free", [20 * GiB, 300 * GiB])
def test_always_qualifies_then_holds_actual_graph_leases_until_children_retire(setup, free):
    setup.free = free
    topology = build(setup, memory_budget_bytes=12 * GiB, memory_safety_margin_percent=40)
    assert topology.memory_admission is None
    topology.start()
    report = topology.memory_admission
    assert report["admission_scope"] == "complete_topology_before_children"
    assert report["estimate"]["allocation_slack_percent"] == 40
    assert report["host"]["admission_budget_bytes"] == 12 * GiB
    # Independent disk arithmetic: 3 sources + 2 tapes/guards + ordinary reserve.
    # Decoder metadata remains authenticated external AAD, not stored bytes.
    transport = 2 * (5000 * (512 + 40) + 8192)
    assert report["candidates"]["native"]["required_disk_bytes"] == 8 * GiB + transport
    assert setup.leases["graph.disk"] == 6 * GiB + transport
    assert setup.leases["graph.memory"] == report["candidates"]["native"]["required_host_bytes"]
    assert setup.events.count("source.import") == setup.events.count("source.price") == 1
    assert setup.events.index("source.close") < setup.events.index("qualification.disk.exit")
    assert setup.events.index("qualification.memory.exit") < setup.events.index("graph.memory.enter")
    assert setup.events.index("graph.disk.enter") < setup.events.index("spawn.worker_a")
    stats = topology.source_qualification
    assert stats == report["source_qualification"] and stats["completed"] and stats["included_in_cold_cpu"]
    assert stats["owner_retired"] and stats["native_program_priced"]
    assert stats["wall_seconds"] >= 0 and stats["process_cpu_seconds"] >= 0
    report["estimate"]["fss_admission"]["source_pins"].clear()
    assert topology.memory_admission["estimate"]["fss_admission"]["source_pins"] == setup.source.snapshot_pins
    topology.close()
    topology.close()
    assert not setup.leases
    assert setup.events.index("children.stop") < setup.events.index("graph.disk.exit")


@pytest.mark.parametrize("failure", ["import", "price", "disk", "reservation", "cancel", "spawn"])
def test_failure_retires_qualification_and_never_starts_unadmitted_children(setup, failure):
    topology = build(setup)
    if failure == "import":
        setup.fail_import = True
    elif failure == "price":
        setup.fail_price = True
    elif failure == "disk":
        setup.after_close = lambda: setattr(setup, "free", 6 * GiB)
    elif failure == "reservation":
        setup.fail_graph_reservation = True
    elif failure == "cancel":
        setup.on_price = topology.close
    else:
        setup.fail_spawn = True
    with pytest.raises(servers.TopologyError):
        topology.start()
    assert not setup.leases and topology.closed
    if setup.loaded is not None:
        assert setup.loaded._closed and topology.source_qualification["owner_retired"]
    if failure != "spawn":
        assert not any(event.startswith("spawn.") for event in setup.events)
        assert not topology.memory_admission["full_topology_admitted"]
    if failure == "disk":
        assert not topology.memory_admission["admitted"]
        assert topology.memory_admission["host"]["disk_free"] == 6 * GiB


@pytest.mark.parametrize("limited", ["memory", "source_disk"])
def test_source_stage_refuses_before_tensor_import(setup, limited):
    if limited == "source_disk":
        setup.free = 3 * GiB
    topology = build(setup, **({"memory_budget_bytes": 1 << 20} if limited == "memory" else {}))
    with pytest.raises(servers.TopologyError):
        topology.start()
    assert setup.loaded is None and not setup.leases and "source.import" not in setup.events
    assert not topology.memory_admission["admitted"]


def test_nondefault_margin_is_enforced_before_import_not_only_reported(setup):
    default = memory.execution_memory_admission(setup.experiment)
    tighter = memory.execution_memory_admission(setup.experiment, memory_safety_margin_percent=40)
    budget = (default["candidates"]["native"]["required_host_bytes"]
              + tighter["candidates"]["native"]["required_host_bytes"]) // 2
    assert memory.execution_memory_admission(setup.experiment, memory_budget_bytes=budget)["admitted"]
    topology = build(setup, memory_budget_bytes=budget, memory_safety_margin_percent=40)
    with pytest.raises(servers.TopologyError):
        topology.start()
    assert "source.import" not in setup.events and not setup.leases
    assert not topology.memory_admission["admitted"]


def test_failed_child_retirement_retains_complete_graph_leases(setup):
    topology = build(setup).start()
    setup.fail_stop = True
    with pytest.raises(ValueError, match="failed to retire"):
        topology.close()
    assert set(setup.leases) == {"graph.memory", "graph.disk"}
    setup.fail_stop = False
    topology.close()
    assert not setup.leases


@pytest.mark.parametrize("kwargs", [{"memory_budget_bytes": True}, {"memory_budget_bytes": 0},
    {"memory_safety_margin_percent": -1}, {"memory_safety_margin_percent": 101}])
def test_invalid_policy_rejects_without_source_work(setup, kwargs):
    with pytest.raises(ValueError, match="memory"):
        build(setup, **kwargs)
    assert setup.events == []


def test_native_wan_forwards_policy_without_changing_supported_graphs(setup):
    from pllm.deployment import WanConditions
    from pllm.profiles import ProprietaryGuarded
    from pllm.runtime.native_wan import NativeWanTopology
    topology = servers.build_roles(Model.hf("org/model"), wan=WanConditions(),
        memory_budget_bytes=12 * GiB, memory_safety_margin_percent=40)
    assert isinstance(topology, NativeWanTopology)
    assert topology._memory_budget_bytes == 12 * GiB and topology._memory_safety_margin_percent == 40
    assert topology._memory_enforced_wan
    assert topology.memory_admission is None
    topology.close()
    # FuseFSS now has admitted dealer/holder WAN routes; construction stays lazy.
    events_before_fss = tuple(setup.events)
    topology = build(setup, wan=WanConditions(),
        memory_budget_bytes=12 * GiB, memory_safety_margin_percent=40)
    try:
        assert isinstance(topology, NativeWanTopology)
        assert topology._privacy_mode == "fss_public"
        assert set(topology._role_ids) == {"dealer", "worker_a", "worker_b"}
        assert topology._memory_budget_bytes == 12 * GiB and topology._memory_safety_margin_percent == 40
        assert topology._memory_enforced_wan
        assert topology.memory_admission is None and topology.source_qualification is None
        assert tuple(setup.events) == events_before_fss
    finally:
        topology.close()
    with pytest.raises(servers.TopologyError, match="admitted public role topology"):
        servers.build_roles(ProprietaryGuarded(Model.hf("org/model")), wan=WanConditions())


def test_non_fss_topology_property_and_docker_forwarding_are_unchanged():
    from pllm.runtime.docker_roles import DockerTopology
    topology = servers.build_roles(Model.hf("org/model"), docker=True,
        memory_budget_bytes=12 * GiB, memory_safety_margin_percent=40)
    assert isinstance(topology, DockerTopology)
    assert topology._memory_budget_bytes == 12 * GiB and topology._memory_safety_margin_percent == 40
    assert topology.memory_admission is None and topology.source_qualification is None
