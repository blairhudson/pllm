"""CPU-only small functional checks; no GPU construction or timing cohorts."""
from dataclasses import replace

import numpy as np
import pytest

from scripts.probe_preparation_factory import (
    PreparationFactory, admission_groups, check_ticket, execute_cells, file_identity, request_for,
)
from pllm.runtime.native import MaskedGEMM


def fixture_factory(*, window_rows=5, backend="integer"):
    weights = np.array([[127, -128, 1], [-7, 13, 0]], dtype=np.int8)
    engine = MaskedGEMM(threads=1)
    assert engine.native
    return weights, PreparationFactory(engine.compile(weights), window_rows=window_rows, backend=backend)


@pytest.mark.parametrize("backend", ["integer", "digit8"])
def test_cross_inventory_stream_is_exact_and_once(backend):
    weights, factory = fixture_factory(backend=backend)
    tickets = [factory.add(request_for(weights, rows=n)) for n in (1, 7, 4)]
    assert len({t.binding for t in tickets}) == 3
    try:
        metrics = factory.execute(tickets)
        assert metrics["rows"] == 12 and metrics["windows"] == 3
        assert sum(check_ticket(t, weights) for t in tickets) == 24
        assert all(t.state == "SEALED" for t in tickets)
        with pytest.raises(ValueError, match="not fresh"):
            factory.execute(tickets)
        assert all(t.state == "BURNED" and t.file.closed for t in tickets)
    finally:
        factory.close()


def test_separate_and_batch_have_identical_private_outputs():
    weights, separate = fixture_factory()
    _, batch = fixture_factory()
    # Same requests used only for matched test-local oracle/control attribution.
    requests = [request_for(weights, rows=n) for n in (3, 7, 2)]
    left, right = [separate.add(r) for r in requests], [batch.add(r) for r in requests]
    try:
        for t in left:
            separate.execute([t])
        batch.execute(right)
        for a, b in zip(left, right, strict=True):
            a.file.seek(0)
            b.file.seek(0)
            assert a.file.read() == b.file.read()
    finally:
        separate.close()
        batch.close()


@pytest.mark.parametrize("width,bits", [(1, 16), (3, 24), (513, 32)])
@pytest.mark.parametrize("backend", ["integer", "digit8"])
def test_exact_all_supported_rings(width, bits, backend):
    weights = np.random.default_rng(41).integers(-128, 128, (2, width), dtype=np.int8)
    factory = PreparationFactory(MaskedGEMM(threads=1).compile(weights), backend=backend,
                                 window_rows=3)
    tickets = [factory.add(request_for(weights, rows=4)) for _ in range(2)]
    try:
        assert all(t.request.wire_bits == bits for t in tickets)
        factory.execute(tickets)
        assert sum(check_ticket(t, weights) for t in tickets) == 16
    finally:
        factory.close()


@pytest.mark.parametrize("fault", ["kernel", "receiver", "cancel"])
def test_partial_failure_burns_entire_owner(fault):
    weights, factory = fixture_factory(window_rows=2)
    tickets = [factory.add(request_for(weights, rows=3)) for _ in range(2)]
    if fault == "cancel":
        factory.close()
    else:
        def fail(*args):
            raise RuntimeError("injected failure")
        if fault == "kernel":
            factory.matrix.wrap32 = fail
        else:
            accept = factory._accept
            calls = []
            def fail_after_partial(*args):
                if calls:
                    fail()
                calls.append(True)
                accept(*args)
            factory._accept = fail_after_partial
        with pytest.raises(RuntimeError):
            factory.execute(tickets)
    assert factory.closed and not factory.key
    assert all(t.state == "BURNED" and t.file.closed for t in tickets)


@pytest.mark.parametrize("fault", ["replay", "stage", "geometry"])
def test_bad_admission_burns_existing_tickets(fault):
    weights, factory = fixture_factory()
    request = request_for(weights, rows=3)
    ticket = factory.add(request)
    bad = request if fault == "replay" else replace(request,
        **({"stage_id": "other"} if fault == "stage" else {"in_features": 4}))
    with pytest.raises(ValueError):
        factory.add(bad)
    assert ticket.state == "BURNED" and factory.closed


def test_receiver_rejects_reordering_and_wrong_authentication():
    weights, factory = fixture_factory()
    ticket = factory.add(request_for(weights, rows=3))
    try:
        with pytest.raises(ValueError, match="attribution"):
            factory._accept(ticket, 1, 1, b"bad", b"bad")
        assert ticket.received == 0
    finally:
        factory.close()


def test_first_ticket_cannot_substitute_weight_or_numeric_binding():
    for change in ({"weight_digest": "0" * 64}, {"weight_bits": 4}):
        weights, factory = fixture_factory()
        with pytest.raises(ValueError, match="binding"):
            factory.add(replace(request_for(weights, rows=1), **change))
        assert factory.closed


def test_bounded_capacity_and_workspace():
    weights, factory = fixture_factory()
    factory.max_tickets = 1
    ticket = factory.add(request_for(weights, rows=1))
    with pytest.raises(ValueError, match="capacity"):
        factory.add(request_for(weights, rows=1))
    assert ticket.state == "BURNED"
    with pytest.raises(ValueError, match="capacity"):
        PreparationFactory(factory.matrix, window_rows=129)
    factory = PreparationFactory(factory.matrix, workspace_bytes=1)
    with pytest.raises(ValueError, match="workspace"):
        factory.add(request_for(weights, rows=1))


def test_deadline_fill_wait_and_small_batch_fallback():
    assert admission_groups([0, .001, .002, .003], capacity=4) == [([0, 1, 2, 3], .003)]
    assert admission_groups([0, .010], capacity=4) == [([0], .005), ([1], .015)]
    assert admission_groups([0, .001], capacity=1, deadline_seconds=0) == [([0], 0), ([1], .001)]
    weights, factory = fixture_factory()
    class ForbiddenGPU:
        def wrap32(self, _):
            raise AssertionError("small batch must not touch GPU")
    factory.backend, factory.gpu = "adaptive", ForbiddenGPU()
    ticket = factory.add(request_for(weights, rows=1))
    try:
        factory.execute([ticket])
        assert factory.calls == {"integer": 1, "digit8": 0, "metal": 0}
        assert check_ticket(ticket, weights) == 2
    finally:
        factory.close()


def test_batched_threshold_dispatch_with_cpu_only_gpu_double():
    weights, factory = fixture_factory(window_rows=64)
    class CPUOnlyGPU:
        def wrap32(self, inputs):
            return factory.matrix.wrap32(inputs)
    factory.backend, factory.gpu = "adaptive", CPUOnlyGPU()
    tickets = [factory.add(request_for(weights, rows=16)) for _ in range(4)]
    try:
        factory.execute(tickets)
        assert factory.calls == {"integer": 0, "digit8": 0, "metal": 1}
        assert sum(check_ticket(t, weights) for t in tickets) == 128
    finally:
        factory.close()


def test_pressure_failure_exits_entire_cell_list_without_retries():
    class Guard:
        error = None
    guard, calls = Guard(), []
    specs = [{"cell": i} for i in range(8)]
    def run(payload):
        calls.append(payload["cell"])
        if payload["cell"] == 1:
            guard.error = "injected pressure"
            raise RuntimeError("private diagnostics not persisted")
        return {"cell": payload["cell"]}
    samples, failures, stopped = execute_cells(specs, run, guard, {}, capture_identity=lambda: {})
    assert calls == [0, 1]
    assert samples == [{"cell": 0}]
    assert failures == [{"spec": specs[1], "error_class": "RuntimeError"}]
    assert stopped == {"reason": "host_pressure", "next_cell_index": 2}


def test_pressure_before_next_cell_never_launches_it():
    class Guard:
        error = "lost headroom"
    def forbidden(_):
        raise AssertionError("must not launch")
    assert execute_cells([{}], forbidden, Guard(), {}, capture_identity=lambda: {}) == (
        [], [], {"reason": "host_pressure", "next_cell_index": 0})


def test_identity_change_stops_before_remaining_cells(tmp_path):
    path = tmp_path / "implementation.py"
    path.write_text("old")
    capture = lambda: {"file": file_identity(path)}
    frozen = capture()
    class Guard:
        error = None
    def run(payload):
        assert payload["expected_identity"] == frozen
        path.write_text("new")
        return {"completed": True}
    samples, failures, stopped = execute_cells([{}, {}, {}], run, Guard(), frozen,
                                               capture_identity=capture)
    assert samples == [{"completed": True}] and failures == []
    assert stopped == {"reason": "runtime_identity_changed", "next_cell_index": 1}
    assert capture() != frozen


def test_missing_identity_file_preserves_completed_cells():
    class Guard:
        error = None
    completed = []
    def capture():
        if completed:
            raise FileNotFoundError("implementation removed")
        return {}
    def run(_):
        completed.append(True)
        return {"complete": True}
    samples, failures, stopped = execute_cells([{}, {}], run, Guard(), {}, capture_identity=capture)
    assert samples == [{"complete": True}] and failures == []
    assert stopped == {"reason": "runtime_identity_unverifiable", "next_cell_index": 1,
                       "error_class": "FileNotFoundError"}
