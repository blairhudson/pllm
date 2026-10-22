"""Cheap phase and source-lifetime guards; doubles prove no numeric acceptance."""

from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

import pytest

from pllm.runtime import fss_source as source

from test_fss_source import admit, bridge as bridge, write_checkpoint
from test_fss_topology_admission import setup as qualification


def test_checkpoint_qualification_uses_ordinary_phase_and_preserves_rejection(monkeypatch):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    experiment = object()
    report = {
        "admitted": True,
        "admission_scope": "source_qualification_only",
        "full_topology_admitted": False,
    }
    calls = []

    def qualify(candidate, **kwargs):
        calls.append((candidate, kwargs))
        return report

    monkeypatch.setattr(memory, "execution_memory_admission", qualify)
    assert checkpoint._source_qualification_report(experiment) is report
    assert calls == [(experiment, {"backend": "native"})]
    # Existing pressure/memory/disk rejection must propagate, not become a skip
    # or a reduced-margin retry. This test creates neither sources nor tapes.
    failure = memory.BenchmarkMemoryError("qualification rejected")

    def rejected(candidate, **kwargs):
        raise failure

    monkeypatch.setattr(memory, "execution_memory_admission", rejected)
    with pytest.raises(memory.BenchmarkMemoryError) as observed:
        checkpoint._source_qualification_report(experiment)
    assert observed.value is failure


@pytest.mark.parametrize(
    "scope,full",
    [
        ("complete_topology_before_children", True),
        ("source_qualification_only", True),
        (None, False),
    ],
)
def test_checkpoint_qualification_rejects_wrong_phase(monkeypatch, scope, full):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    monkeypatch.setattr(
        memory,
        "execution_memory_admission",
        lambda *args, **kwargs: {
            "admitted": True,
            "admission_scope": scope,
            "full_topology_admitted": full,
        },
    )
    with pytest.raises(memory.BenchmarkMemoryError, match="source-only qualification"):
        checkpoint._source_qualification_report(object())


def test_checkpoint_qualification_matches_runtime_price_and_floor(qualification, monkeypatch):
    """Exercise the actual phase gate with metadata doubles, not tensor import."""
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
        24 << 30, 20 << 30, 0, qualification.free,
        availability_source="macos_mach_dispatch_v2", pressure="normal",
        free=20 << 30, cached=0, purgeable=0, reusable=0,
    ))
    ordinary = memory.benchmark_memory(qualification.experiment, backend="native")
    phase = checkpoint._source_qualification_report(qualification.experiment)
    # Conservative full-graph disk rejection must not price nonexistent tapes
    # into this one-source phase. Memory and watchdog guards remain unchanged.
    assert not ordinary["admitted"]
    assert phase["admission_scope"] == "source_qualification_only"
    assert phase["full_topology_admitted"] is False
    assert phase["estimate"] == memory._fss_source_qualification_estimate(ordinary["estimate"])
    assert phase["candidates"]["native"]["required_disk_bytes"] == (
        qualification.source.contract["source_disk_bytes"] + (2 << 30)
    )
    for field in ("required_host_bytes", "expected_peak_bytes", "safety_margin_bytes"):
        assert phase["candidates"]["native"][field] == ordinary["candidates"]["native"][field]
    assert phase["runtime_floor_bytes"] == ordinary["runtime_floor_bytes"]
    assert phase["runtime_floor_bytes"] == phase["candidates"]["native"]["safety_margin_bytes"]
    assert qualification.events == ["source.metadata", "source.metadata"]
    assert qualification.loaded is None and not qualification.leases


@pytest.mark.parametrize("limited", ["disk", "memory", "pressure"])
def test_checkpoint_qualification_preserves_runtime_capacity_rejection(
    qualification, monkeypatch, limited
):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    if limited == "disk":
        # One byte below single source + ordinary 2 GiB reserve; no discount.
        qualification.free = qualification.source.contract["source_disk_bytes"] + (2 << 30) - 1
        match = "disk headroom"
    else:
        available = (1 << 30) if limited == "memory" else (20 << 30)
        monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
            24 << 30, available, 0, qualification.free,
            availability_source="macos_mach_dispatch_v2",
            pressure="warning" if limited == "pressure" else "normal",
            free=available, cached=0, purgeable=0, reusable=0,
        ))
        match = "safe headroom" if limited == "memory" else "memory pressure"
    with pytest.raises(memory.BenchmarkMemoryError, match=match):
        checkpoint._source_qualification_report(qualification.experiment)
    assert qualification.loaded is None and not qualification.leases
    assert "source.import" not in qualification.events


@pytest.fixture
def priced_qualification(qualification, monkeypatch):
    """Existing explicit native doubles; exercise the real imported-pricing API."""
    import test_fss_checkpoint_admission as checkpoint
    from pllm.deployment import resources
    from pllm.runtime import fss_tape, servers

    def forbidden(*args, **kwargs):
        pytest.fail("admission must not start children or create material tapes")

    monkeypatch.setattr(servers.LocalTopology, "start", forbidden)
    monkeypatch.setattr(servers.LocalTopology, "_spawn", forbidden)
    monkeypatch.setattr(fss_tape.MaterialTape, "__init__", forbidden)
    qualification.phase = checkpoint._source_qualification_report(qualification.experiment)
    qualification.topology = SimpleNamespace(memory_admission=qualification.phase)

    @contextmanager
    def owned_source():
        with ExitStack() as leases:
            leases.enter_context(resources.reserve_incremental_memory(
                qualification.phase["estimate"]["incremental_owner_bounds"]["per_role"],
            ))
            leases.enter_context(resources.reserve_incremental_disk(
                qualification.source.contract["source_disk_bytes"],
                directory=Path(tempfile.gettempdir()), minimum_free_bytes=2 << 30,
            ))
            loaded = source.load_role_source(
                qualification.experiment, expected_pins=qualification.source.snapshot_pins,
                source=qualification.source, cancelled=lambda: False,
            )

            def check_memory():
                assert not loaded._closed
                assert set(qualification.leases) == {"qualification.memory", "qualification.disk"}
                qualification.events.append("qualification.watchdog")

            try:
                yield qualification.experiment, loaded, check_memory
            finally:
                loaded.close()

    return owned_source


def test_checkpoint_source_only_remains_independent_of_graph_pricing(
    qualification, priced_qualification, monkeypatch
):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    def forbidden(*args, **kwargs):
        pytest.fail("source-only admission must not price or reserve the graph")

    monkeypatch.setattr(memory, "benchmark_memory", forbidden)
    monkeypatch.setattr(checkpoint, "_complete_topology_admission", forbidden)
    monkeypatch.setattr(checkpoint, "_qualified_qwen_source", lambda *args: priced_qualification())
    checkpoint.test_pinned_qwen_checked_tensor_source_admission(1, monkeypatch)
    assert qualification.loaded._closed and not qualification.leases
    assert "source.price" not in qualification.events
    assert "graph.memory.enter" not in qualification.events


def test_checkpoint_complete_topology_prices_live_owner_then_retires_before_graph(
    qualification, priced_qualification, monkeypatch
):
    import test_fss_checkpoint_admission as checkpoint
    from pllm.runtime import benchmark_memory as memory

    calls = []
    ordinary = memory.benchmark_memory

    def price(experiment, **kwargs):
        calls.append((experiment, kwargs))
        return ordinary(experiment, **kwargs)

    monkeypatch.setattr(memory, "benchmark_memory", price)
    report = checkpoint._complete_topology_admission(priced_qualification())
    assert calls == [(qualification.experiment, {
        "backend": "native", "fss_imported": qualification.loaded,
    })]
    assert report["admitted"] and report["full_topology_admitted"]
    assert report["admission_scope"] == "complete_topology_before_children"
    estimate = report["estimate"]
    assert estimate["allocation_slack_percent"] == 25
    assert estimate["incremental_owner_bounds"] == qualification.phase["estimate"]["incremental_owner_bounds"]
    # Independent full topology arithmetic: three sources, two native-priced
    # tapes/replay guards, plus the unchanged 2 GiB free-space reserve.
    assert report["candidates"]["native"]["required_disk_bytes"] == (
        (8 << 30) + 2 * (5000 * (512 + 40) + 8192)
    )
    events = qualification.events
    assert events.index("source.price") < events.index("source.close")
    assert events.index("qualification.watchdog") < events.index("source.close")
    assert events.index("source.close") < events.index("qualification.disk.exit")
    assert events.index("qualification.memory.exit") < events.index("graph.memory.enter")
    assert events.index("graph.memory.enter") < events.index("graph.disk.enter")
    assert qualification.loaded._closed and not qualification.leases
    assert not any(event.startswith("spawn.") for event in events)


@pytest.mark.parametrize(
    "failure", [
        "owner", "closed", "envelope", "identity", "price", "disk", "tempfile_disk",
        "memory", "pressure", "reservation",
    ],
)
def test_checkpoint_complete_topology_rejects_and_retires_without_children(
    qualification, priced_qualification, monkeypatch, failure
):
    import test_fss_checkpoint_admission as checkpoint
    from pllm import _native
    from pllm.runtime import benchmark_memory as memory

    if failure == "price":
        qualification.fail_price = True
    elif failure == "disk":
        qualification.after_close = lambda: setattr(qualification, "free", 6 << 30)
    elif failure == "tempfile_disk":
        def disk_short(directory):
            assert directory == Path(tempfile.gettempdir())
            return SimpleNamespace(total=512 << 30, free=6 << 30)

        monkeypatch.setattr(checkpoint.shutil, "disk_usage", disk_short)
    elif failure in {"memory", "pressure"}:
        sample = memory.host_memory

        def memory_short_after_retirement():
            if qualification.loaded is not None and qualification.loaded._closed:
                if failure == "memory":
                    return memory.HostMemory(24 << 30, 1 << 30, 0, qualification.free)
                return memory.HostMemory(
                    24 << 30, 20 << 30, 0, qualification.free,
                    availability_source="macos_mach_dispatch_v2", pressure="warning",
                    free=20 << 30, cached=0, purgeable=0, reusable=0,
                )
            return sample()

        monkeypatch.setattr(memory, "host_memory", memory_short_after_retirement)
    elif failure == "reservation":
        qualification.fail_graph_reservation = True
    elif failure == "identity":
        native_price = _native.fss_transport_admission

        def wrong_program(*args):
            price = json.loads(native_price(*args))
            price["program_digest"] = "f" * 64
            return json.dumps(price).encode()

        monkeypatch.setattr(_native, "fss_transport_admission", wrong_program)

    @contextmanager
    def candidate():
        with priced_qualification() as (experiment, loaded, check_memory):
            if failure == "owner":
                # A lookalike record cannot stand in for an imported owner.
                loaded = SimpleNamespace(source=loaded.source, contract_bytes=loaded.contract_bytes)
            elif failure == "closed":
                loaded.close()
            elif failure == "envelope":
                # A price for a smaller prompt cannot cover the saved envelope.
                contract = json.loads(loaded.contract_bytes)
                contract["input_rows"] = 1
                loaded.contract_bytes = json.dumps(contract).encode()
            yield experiment, loaded, check_memory

    exception = ValueError if failure == "reservation" else memory.BenchmarkMemoryError
    match = (
        "disk capacity changed" if failure == "reservation" else
        "benchmark memory preflight rejected" if failure in {"disk", "tempfile_disk", "memory", "pressure"} else
        "imported source pricing rejected"
    )
    with pytest.raises(exception, match=match) as observed:
        checkpoint._complete_topology_admission(candidate())
    assert qualification.loaded._closed and not qualification.leases
    assert "source.close" in qualification.events
    assert not any(event.startswith("spawn.") for event in qualification.events)
    if failure in {"owner", "closed", "envelope"}:
        assert "source.price" not in qualification.events
    if failure in {"disk", "tempfile_disk", "memory", "pressure"}:
        assert observed.value.report["full_topology_admitted"] is False
    if failure != "reservation":
        assert "graph.memory.enter" not in qualification.events


@pytest.mark.parametrize("initial_rows", [1, 4])
def test_cancel_after_tensor_seal_retires_published_program(
    tmp_path, monkeypatch, bridge, initial_rows
):
    """Cancellation after allocation must close both program and pending builder."""
    write_checkpoint(tmp_path)
    admitted = admit(
        bridge,
        tmp_path,
        rows=initial_rows,
        outputs=1,
        material_delivery="batch-v1",
        online_schedule="tensor-v1",
    )
    cancelled = threading.Event()
    builder_type = source._native.FssSourceBuilder
    seal_tensor = builder_type.seal_tensor

    def seal_then_cancel(builder):
        program = seal_tensor(builder)
        assert not builder.closed and not program.closed
        cancelled.set()
        return program

    monkeypatch.setattr(builder_type, "seal_tensor", seal_then_cancel)
    with pytest.raises(source.FssSourceError, match="cancelled"):
        source.load_role_source(
            bridge.experiment,
            source=admitted,
            expected_pins=admitted.snapshot_pins,
            input_rows=initial_rows,
            outputs=1,
            cancelled=cancelled.is_set,
        )
    owner = bridge.owners[-1]
    assert owner.online_schedule == "tensor-v1"
    assert owner.frames and owner.closed
    assert owner.program is not None and owner.program.closed
