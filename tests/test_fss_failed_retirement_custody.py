"""Source-authored FSS custody regressions; fake HTTP/native, real shared close."""
import asyncio
import gc
import json
import sys
import threading
from types import SimpleNamespace as NS
import weakref

import pytest

from pllm.deployment import resources
from pllm.runtime.client import ProtocolError
from pllm.runtime.fss_client import FssClient
from pllm.runtime import fss_protocol as p
from pllm.runtime import fss_transport as transport
from pllm.runtime import delegated_transport as shared


_REAL_PEERS = shared._Peers
_REAL_IO_OWNER = shared._IOOwner


@pytest.fixture
def funded_source(monkeypatch, tmp_path):
    """Controlled telemetry, real module-rooted memory/slot/disk reservations.

    Every ledger belongs to this fixture. Sticky fake-native failures deliberately
    remain charged until fixture disposal; teardown never repairs production state
    or calls a second empty close to manufacture an ACK.
    """
    from pllm.runtime import benchmark_memory as memory, servers

    capacity, disk_capacity = resources._ClientCapacity(), resources._DiskCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(resources, "_disk_capacity", disk_capacity)
    # Patch classes, not instances: monkeypatch's undo stack must not itself root
    # the fresh ledgers in the GC custody test.
    monkeypatch.setattr(resources._ClientCapacity, "observe", lambda self: NS(
        headroom_bytes=(32 << 30) - sum(self._owners.values())))
    monkeypatch.setattr(resources._DiskCapacity, "observe", lambda self, directory, minimum: resources.DiskResources(
        1, 64 << 30, sum(lease._bound for lease in self._owners.values()), minimum))
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
        64 << 30, 32 << 30, 0, 64 << 30))
    monkeypatch.setattr("shutil.disk_usage", lambda path: NS(free=64 << 30))
    monkeypatch.setattr(servers.tempfile, "gettempdir", lambda: str(tmp_path))
    # Exercise the ordinary watchdog check, without a worker thread or real probe.
    monkeypatch.setattr(memory.MemoryWatchdog, "start", lambda self: self.check())
    topology = object.__new__(servers.LocalTopology)
    topology._process_lock, topology._stopping = threading.Lock(), threading.Event()
    topology._startup_breadcrumbs = {"phase": "unknown"}
    topology._memory_admission, topology._memory_enforced_wan = None, False
    topology._failure_observer = None
    state = NS(topology=topology, capacity=capacity, disk_capacity=disk_capacity,
        events=[], refs={}, fail=None, retire_fail=None, stop=None, observer_fail=False,
        raw=1700000003, other=99000007, p=384 << 20, source_disk=123456,
        controls=[], entered=threading.Event(), resume=threading.Event(), refund_count=0,
        reprice_hook=None, primary=None, between_call_drain=False)
    frozen = NS(snapshot_pins={"checkpoint": "a" * 64, "tokenizer": "b" * 64})

    class Ledger(dict):
        def pop(self, token, *default):
            charged = token in self
            result = super().pop(token, *default)
            if charged:
                state.refund_count += 1
            return result

    capacity._owners = Ledger()
    # Metadata admission leaves ordinary dictionary backing behind. Warm through
    # the REAL reservation API so precomputed equality budgets use that shape.
    with resources.reserve_incremental_memory(384 << 20):
        pass
    state.refund_count = 0

    def observe_failure():
        state.events.append("failure")
        if state.observer_fail:
            raise KeyboardInterrupt("observer secondary")

    topology._failure_observer = observe_failure

    class Owner:
        def __init__(self):
            self.calls, self.consumed = 0, False
            state.refs["owner"] = weakref.ref(self)

        def close(self):
            self.calls += 1
            state.events.append("owner.close")
            assert sum(resources._capacity._owners.values()) == state.p + state.controls[-1]
            assert resources._disk_capacity._owners  # Original disk lease remains unexited.
            if state.retire_fail == "false":
                return False
            if state.retire_fail:
                self.consumed = True
                raise RuntimeError("consume-then-raise")

    def resolve(lifecycle):
        state.events.append("metadata")
        # Source metadata lease is disjoint: not present in qualification P+C.
        with resources.reserve_incremental_memory(384 << 20):
            assert sum(resources._capacity._owners.values()) == 384 << 20
        return frozen

    def price(source, pins, control):
        assert source is frozen and pins == frozen.snapshot_pins
        state.events.append("price:" + str(control))
        state.controls.append(control)
        if control and state.reprice_hook is not None:
            state.reprice_hook()
        # Generic adapter contract: whole memory quote and single-source disk.
        client = memory._slack(state.raw + control, state.margin)
        estimate = {"client_peak_bytes": client, "provider_peak_bytes": {"provider": state.other},
            "native_total_peak_bytes": client + state.other, "docker_provider_peak_bytes": {},
            "runtime_kind": "fss", "allocation_slack_percent": state.margin,
            "limitations": [], "kernel": "cpu", "components": {
                "per_engine_compiled_disk_bytes": 0, "raw_client_bundle_bytes": 0,
                "client_additional_paged_disk_bytes": state.source_disk}}
        return estimate, state.raw, state.other, state.p, state.source_disk

    def produce(source, pins, payload, lifecycle):
        assert source is frozen and pins == frozen.snapshot_pins
        state.events.append("produce")
        state.refs["payload"] = weakref.ref(payload.memory)  # Original manager, not a copied lease.
        state.refs["lifetime"] = payload
        slot = next(iter(resources._capacity._cleanup.values()))
        assert slot.owns(payload, retire=slot._retire)
        assert slot.control_bytes == state.controls[-1]
        assert sorted(resources._capacity._owners.values()) == sorted([state.p, slot.control_bytes])
        assert len(resources._disk_capacity._owners) == 1
        assert payload.disk is next(iter(resources._disk_capacity._owners.values()))
        assert payload.producing and not payload.active
        if state.between_call_drain:
            with pytest.raises(servers_error(), match="retirement unknown"):
                slot.drain()
            assert not slot.closed and slot.owns(payload)
            assert sum(resources._capacity._owners.values()) == state.p + slot.control_bytes
        if state.fail == "constructor":
            payload.begin()
            if state.primary is not None:
                raise state.primary
            raise ValueError("constructor primary")
        payload.begin()
        owner = Owner()
        payload.acquired(1, owner)
        if state.fail == "operation":
            if state.primary is not None:
                raise state.primary
            raise ValueError("operation primary")
        payload.begin()
        state.entered.set()
        if state.stop == "inflight":
            assert state.resume.wait(5)
        payload.returned()
        if state.stop in ("after", "inflight"):
            topology.request_stop()
        return {"extent": "partial", "first_material_refusal": {"limit": 7},
                "material_wire_bytes": None}

    def consume(value, source, pins):
        state.events.append("consume")
        return value

    def run(*, margin=25, budget=None):
        state.margin = margin
        return servers._funded_source_qualification(topology, resolve=resolve, price=price,
            produce=produce, consume=consume, memory_budget_bytes=budget,
            memory_safety_margin_percent=margin)

    state.run = run
    del capacity, disk_capacity  # No fixture-frame alternative to the module root.
    yield state
    # Only this fixture's fresh fake owners/roots exist. Unknown retirement is
    # intentionally not drained/retried. Drop fixture references explicitly;
    # monkeypatch restores original module registries after fixture teardown.
    state.refs.clear()
    state.topology = state.run = None


@pytest.mark.parametrize("margin", [0, 25])
def test_funded_source_exact_control_ceiling_and_independent_role_charge(funded_source, margin):
    state = funded_source
    c = state.capacity._cleanup_control_bytes()
    budget = ((state.raw + c) * (100 + margin) + 99) // 100 + state.other
    value, _, _ = state.run(margin=margin, budget=budget)
    assert value["material_wire_bytes"] is None and value["extent"] == "partial"
    assert state.controls == [0, c]
    assert not state.capacity._owners and not state.capacity._cleanup
    assert not state.disk_capacity._owners
    assert state.events.index("produce") > state.events.index("price:" + str(c))
    assert state.events.index("consume") < state.events.index("owner.close")
    assert state.refund_count == 3  # Disjoint metadata, original P, independent C.


@pytest.mark.parametrize("margin", [0, 25])
def test_funded_source_one_byte_short_refuses_before_productive_import(funded_source, margin):
    state = funded_source
    c = state.capacity._cleanup_control_bytes()
    budget = ((state.raw + c) * (100 + margin) + 99) // 100 + state.other - 1
    with pytest.raises(Exception):
        state.run(margin=margin, budget=budget)
    assert "produce" not in state.events
    assert not state.capacity._owners and not state.capacity._cleanup
    assert not state.disk_capacity._owners


def test_funded_source_fractional_slack_uses_exact_maximum_before_atomic_slot(
    funded_source, monkeypatch,
):
    state = funded_source
    actual = state.capacity._cleanup_control_bytes() + 1
    monkeypatch.setattr(state.capacity, "_cleanup_control_bytes", lambda: actual)
    acquire = resources.reserve_cleanup_slot
    maxima = []

    def bounded(*, max_control_bytes=None):
        maxima.append(max_control_bytes)
        return acquire(max_control_bytes=max_control_bytes)

    monkeypatch.setattr(resources, "reserve_cleanup_slot", bounded)
    budget = ((state.raw + actual) * 125 + 99) // 100 + state.other
    separately_rounded = ((state.raw * 125 + 99) // 100
                          + (actual * 125 + 99) // 100 + state.other)
    assert separately_rounded == budget + 1
    state.run(margin=25, budget=budget)
    assert maxima == [actual]
    assert state.controls == [0, actual]
    assert not state.capacity._owners


def test_funded_source_between_call_drain_cannot_refund_live_producer(funded_source):
    state = funded_source
    state.between_call_drain = True
    state.run()
    assert not state.capacity._owners and not state.disk_capacity._owners
    assert state.events.count("owner.close") == 1
    assert state.refund_count == 3


@pytest.mark.parametrize("cut", ["constructor", "operation"])
@pytest.mark.parametrize("publication", ["normal", "before", "after"])
@pytest.mark.parametrize("primary_type", [ValueError, KeyboardInterrupt])
def test_funded_source_primary_and_unexited_funding_survive_handoff_and_gc(
    funded_source, monkeypatch, cut, publication, primary_type,
):
    state = funded_source
    state.fail, state.retire_fail, state.observer_fail = cut, True, True
    state.primary = primary_type(cut + " primary")
    original = resources._MemoryReservation.retain_failed

    def retain(self, owners):
        if publication == "before":
            raise KeyboardInterrupt("before publication")
        original(self, owners)
        if publication == "after":
            raise KeyboardInterrupt("after publication")

    monkeypatch.setattr(resources._MemoryReservation, "retain_failed", retain)
    with pytest.raises(primary_type, match=cut + " primary") as caught:
        state.run()
    assert caught.value is state.primary
    primary = caught.value
    state.primary = None
    if cut == "operation":
        assert state.events.index("failure") < state.events.index("owner.close")
    assert sum(state.capacity._owners.values()) == state.p + state.controls[-1]
    assert len(state.disk_capacity._owners) == 1
    reservation = state.refs["payload"]
    _detach(primary)
    del primary, caught
    state.refs.pop("lifetime")
    state.topology = None
    state.run = None
    state.capacity = None
    state.disk_capacity = None
    gc.collect()
    # Lookup only through the existing module registry, after caller roots vanish.
    slot = next(iter(resources._capacity._cleanup.values()))
    payload = slot._payload
    assert payload.memory is reservation()
    assert payload.disk is next(iter(resources._disk_capacity._owners.values()))
    assert payload.memory_stack._exit_callbacks and payload.disk_stack._exit_callbacks
    assert sum(resources._capacity._owners.values()) == state.p + slot.control_bytes
    if publication != "before":
        assert payload.memory.pending_retirement
        with pytest.raises(servers_error(), match="retirement unknown"):
            payload.memory.drain_failed()
    with pytest.raises(servers_error(), match="retirement unknown"):
        slot.drain()
    assert not slot.closed
    assert state.refund_count == 1  # Metadata only; no qualification refunds.


def servers_error():
    from pllm.runtime.servers import TopologyError
    return TopologyError


def test_funded_source_stop_after_blocking_return_suppresses_scalars(funded_source):
    state = funded_source
    state.stop = "after"
    with pytest.raises(servers_error(), match="stopped"):
        state.run()
    assert "consume" not in state.events
    assert state.events.index("failure") < state.events.index("owner.close")
    assert not state.capacity._owners and not state.disk_capacity._owners


def test_source_false_retirement_never_refunds_and_is_not_retried(funded_source):
    state = funded_source
    state.retire_fail = "false"
    with pytest.raises(servers_error(), match="did not acknowledge"):
        state.run()
    slot = next(iter(state.capacity._cleanup.values()))
    assert sum(state.capacity._owners.values()) == state.p + slot.control_bytes
    assert state.disk_capacity._owners and not slot.closed
    with pytest.raises(servers_error(), match="retirement unknown"):
        slot.drain()
    assert state.refs["owner"]().calls == 1
    assert state.refund_count == 1


def test_source_owner_drain_is_separate_from_funding_refund_and_program_not_double_closed(funded_source, tmp_path):
    from pllm.runtime.fss_source import LoadedFssSource
    from pllm.runtime.servers import _FundedSourceLifetime
    state = funded_source
    payload = _FundedSourceLifetime(object(), {})
    slot = resources.reserve_cleanup_slot()
    payload.memory = resources.reserve_incremental_memory(state.p)
    payload.memory_stack.enter_context(payload.memory)
    payload.disk_manager = resources.reserve_incremental_disk(state.source_disk,
        directory=tmp_path, minimum_free_bytes=2 << 30)
    payload.disk = payload.disk_stack.enter_context(payload.disk_manager)
    calls = []

    def native_close():
        calls.append("native.close")
        assert state.capacity._owners and state.disk_capacity._owners

    program = NS(close=native_close)
    loaded = LoadedFssSource(None, b"{}", program, b"{}", b"{}")
    payload.begin()
    payload.acquired(2, program)
    payload.begin()
    payload.acquired(3, loaded, replaces=2)
    assert payload.owners[2] is None and payload.owners[3] is loaded

    def retire():
        payload.close()
        payload.memory_stack.close()

    slot.retain(payload, retire)
    payload.memory.retain_failed(payload.failed_owners)
    payload.memory.drain_failed()  # Owner close MUST NOT recursively drain itself.
    assert calls == ["native.close"] and loaded._closed and payload.retired
    assert not state.disk_capacity._owners
    assert list(state.capacity._owners.values()) == [slot.control_bytes]
    slot.drain()
    assert calls == ["native.close"] and not state.capacity._owners
    assert state.refund_count == 2  # Original P once, independent C once.


def test_funded_source_stop_before_entry_does_no_resolution(funded_source):
    state = funded_source
    state.topology.request_stop()
    with pytest.raises(servers_error(), match="stopped"):
        state.run()
    assert not state.events and not state.capacity._owners


def test_funded_source_fresh_pressure_refusal_drains_only_inert_slot(funded_source, monkeypatch):
    from pllm.runtime import benchmark_memory as memory
    state = funded_source

    def change_pressure():
        monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
            64 << 30, 32 << 30, 0, 64 << 30, availability_source="macos_mach_dispatch_v2",
            pressure="warning", free=32 << 30, cached=0, purgeable=0, reusable=0))

    state.reprice_hook = change_pressure
    with pytest.raises(memory.BenchmarkMemoryError, match="preflight rejected"):
        state.run()
    assert "produce" not in state.events
    assert not state.capacity._owners and not state.capacity._cleanup
    assert not state.disk_capacity._owners
    assert state.refund_count == 2  # Metadata + inert C; no P issued.


@pytest.mark.parametrize("cut", ["ack", "published_then_raise"])
def test_funded_source_binding_must_ack_before_funding_or_import(funded_source, monkeypatch, cut):
    state = funded_source
    if cut == "ack":
        monkeypatch.setattr(resources._CleanupSlot, "owns", lambda *a, **k: False)
    else:
        original = resources._CleanupSlot.retain

        def retain(self, payload, retire):
            original(self, payload, retire)
            raise ValueError("binding publication primary")

        monkeypatch.setattr(resources._CleanupSlot, "retain", retain)
    with pytest.raises((ValueError, servers_error())):
        state.run()
    assert "produce" not in state.events
    assert not state.capacity._owners and not state.capacity._cleanup
    assert not state.disk_capacity._owners
    assert state.refund_count == 2  # Metadata + C; no P or native owner issued.


def test_funded_source_negative_control_allowance_never_acquires_slot(funded_source, monkeypatch):
    from unittest.mock import Mock
    state = funded_source
    acquire = Mock(side_effect=AssertionError("negative Cmax must precede slot"))
    monkeypatch.setattr(resources, "reserve_cleanup_slot", acquire)
    with pytest.raises(Exception, match="cannot fund base owners"):
        state.run(budget=state.other)
    acquire.assert_not_called()
    assert "produce" not in state.events and not state.capacity._owners


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "123", 1 << 64])
def test_funded_source_invalid_budget_refuses_before_metadata(funded_source, budget):
    state = funded_source
    with pytest.raises(ValueError, match="bounded nonnegative"):
        state.run(budget=budget)
    assert not state.events and not state.capacity._owners


def test_cleanup_actual_control_quote_survives_ledger_growth(funded_source):
    state = funded_source
    managers = []
    slot = None
    try:
        for _ in range(96):
            manager = resources.reserve_incremental_memory(1)
            managers.append(manager)
            manager.__enter__()
        actual = state.capacity._cleanup_control_bytes()
        with pytest.raises(Exception, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
            resources.reserve_cleanup_slot(max_control_bytes=actual - 1)
        slot = resources.reserve_cleanup_slot(max_control_bytes=actual)
        assert slot.control_bytes == actual
        assert state.capacity._owners[slot._token] == actual
        for _ in range(192):
            manager = resources.reserve_incremental_memory(1)
            managers.append(manager)
            manager.__enter__()
        assert state.capacity._cleanup_control_bytes() > actual
        assert slot.control_bytes == actual  # Historical ACTUAL issued C.
        assert state.capacity._owners[slot._token] == actual
    finally:
        if slot is not None:
            slot.drain()  # Inert test slot only.
        for manager in reversed(managers):
            manager.close()
    assert not state.capacity._owners and not state.capacity._cleanup


def test_source_retirement_busy_callback_raises_not_false(funded_source):
    from pllm.runtime.servers import _FundedSourceLifetime
    state = funded_source
    payload = _FundedSourceLifetime(object(), {})
    slot = resources.reserve_cleanup_slot()
    slot.retain(payload, payload.close)
    entered, release = threading.Event(), threading.Event()

    def holder():
        with payload.lock:
            entered.set()
            assert release.wait(5)

    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert entered.wait(5)
        with pytest.raises(servers_error(), match="retirement busy"):
            slot.drain()
        assert not slot.closed and slot.owns(payload)
        assert state.capacity._owners
    finally:
        release.set()
        thread.join(5)
    slot.drain()  # No productive owners; explicit teardown of this fresh test slot.
    assert slot.closed and not state.capacity._owners


def test_funded_source_inflight_stop_keeps_both_charges_until_return(funded_source):
    state = funded_source
    state.stop = "inflight"
    outcomes = []

    def worker():
        try:
            state.run()
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert state.entered.wait(5)
        state.topology.request_stop()
        assert sum(state.capacity._owners.values()) == state.p + state.controls[-1]
        assert state.disk_capacity._owners
        slot = next(iter(state.capacity._cleanup.values()))
        with pytest.raises(servers_error(), match="retirement unknown"):
            slot.drain()
        assert not slot.closed and "owner.close" not in state.events
    finally:
        state.resume.set()
        thread.join(5)
    assert not thread.is_alive()
    assert len(outcomes) == 1 and isinstance(outcomes[0], servers_error())
    _detach(outcomes.pop())
    assert not state.capacity._owners and not state.disk_capacity._owners


def _receipt(role):
    storage = {} if role == "dealer" else {
        "state": "finished", "encrypted_file_bytes_written": 42,
        "encrypted_file_bytes_read": 42, "packet_bytes_written": 1,
        "packet_bytes_read": 1, "records_written": 1, "records_read": 1,
        "sealed_file_bytes": 42, "sealed_digest": "d" * 64}
    return p.pack({"frame": b"" if role == "dealer" else role.encode(),
                   "peer_bytes": p.peer_bytes(), "storage": storage})


@pytest.fixture
def custody(monkeypatch):
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    amount = transport.client_memory_bytes("capsule")
    monkeypatch.setattr(capacity, "observe", lambda: NS(
        headroom_bytes=amount - sum(capacity._owners.values())))
    client = object.__new__(FssClient)
    client._lock = threading.RLock()
    client._closing = False
    client._active = client._compact_cleanup = client._failed_request_cleanup = None
    client._tokenizer_scopes = set()
    client.records = {}
    client._source = client._result_hook = None
    client._admit = lambda body: body
    state = NS(client=client, attempt=client.events({}), capacity=capacity,
               amount=amount, refs={}, tokens=[], events=[], markers=[], frozen=[],
               failures=set(), operation=False, audit=False, record=False,
               handoff=False, completed=[], calls=[], packed_cut=None,
               inspect_payload=False, payload_checks=[], refunds=0)
    client._fss_failure_observer = state.frozen.append

    class ObservedLedger(dict):
        def pop(self, token, *default):
            charged = token in self
            value = super().pop(token, *default)
            if charged and token not in self:
                state.refunds += 1
            return value

    capacity._owners = ObservedLedger(capacity._owners)

    def payload_cleared(stage):
        if not state.inspect_payload:
            return
        frame = sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code is transport.execute_request.__code__:
                    assert frame.f_locals["payload"] is None
                    state.payload_checks.append(stage)
                    return
                frame = frame.f_back
            pytest.fail("executing transport frame missing before retirement")
        finally:
            del frame

    def boundary(role, action, status):
        if (role, action, status) == ("client", "retire", "entered"):
            payload_cleared("retire.entered")
        type(state.attempt).record_control_boundary(state.attempt, role, action, status)
        state.markers.append((role, action, status))

    state.attempt.record_control_boundary = boundary

    def memory(size):
        reservation = resources.reserve_incremental_memory(size)
        state.refs["reservation"] = weakref.ref(reservation)
        state.tokens.append(id(reservation._token))  # no caller root through token.retained
        return reservation

    def close(owner, name):
        assert not capacity.lock._is_owned()
        payload_cleared(name + ".close")
        state.events.append((name, "close"))
        if name in state.failures:
            # Fresh exceptions: fixture never stores tracebacks/owner references.
            raise RuntimeError(name + " retirement failed")
        owner.closed = True

    class Native:
        def __init__(self, *args):
            self.closed = False
            state.refs["native"] = weakref.ref(self)

        def request_bytes(self):
            return b"request-frame"

        def take_input(self, index):
            return b"input-share"

        def finish(self, *frames):
            assert frames == (b"worker_a", b"worker_b")
            if state.operation:
                raise ValueError("operation failed")
            return [4]

        def close(self):
            close(self, "native")

    class Peers:
        def start(self, connections, timeout, cancelled, **options):
            assert options == {"roles": p.ROLES, "namespace": "fss"}
            self.deadline = transport.time.monotonic() + timeout
            self.live, self.counts, self.closed = [False] * 3, {}, False
            state.refs["peers"] = weakref.ref(self)

        def check(self):
            pass

        def post(self, index, action, payload=b"", **options):
            role = p.ROLES[index]
            state.calls.append((role, action))
            if action == state.packed_cut:
                assert type(payload) is bytes and payload
                raise ValueError("packed " + action + " failed")
            if action == "manifest":
                return p.pack({"schema": "pllm.fss.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "b" * 64,
                    "tokenizer": "c" * 64, "transport": p.TRANSPORT,
                    "epoch": str(index + 1) * 64})
            if action == "admit":
                return p.pack({"manifest": b"native-manifest"})
            if action == "finish":
                return _receipt(role)
            return p.pack({"prepared" if action == "prepare" else
                           "started" if action == "start" else "accepted": True})

        def close(self):
            close(self, "peers")

        def audit(self):
            payload_cleared("peers.audit")
            state.events.append(("peers", "audit"))
            if state.audit:
                raise LookupError("audit failed")
            return dict(self.counts)

    def finish(peers, boundary, failure, **options):
        # Protocol/I/O fake only: real execute_request validates and reconstructs.
        receipts = []
        for index, role in enumerate(p.HOLDERS):
            boundary(role, "finish", "entered")
            receipts.append(transport._receipt(_receipt(role), role, **options))
            peers.live[index] = False
            boundary(role, "finish", "validated")
        return receipts

    def record(audit):
        if state.record:
            raise LookupError("record failed")
        type(state.attempt).record_audit(state.attempt, audit)

    state.attempt.record_audit = record

    def handoff(attempt, reservation):
        assert capacity._owners[reservation._token] == amount
        assert id(reservation._token) == state.tokens[0]
        assert reservation._token.retained is reservation
        if state.handoff:
            raise KeyboardInterrupt("handoff detail must not appear in notes")
        FssClient._retain_failed_request_reservation(state.client, attempt, reservation)

    client._retain_failed_request_reservation = handoff
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_native", NS(FssClientRequest=Native))
    monkeypatch.setattr(transport, "_Peers", Peers)
    monkeypatch.setattr(transport, "_finish_receipts", finish)
    experiment = NS(configuration_digest=lambda: "a" * 64,
                    pipeline=NS(components={"linear": NS(params={})}))
    contract = {"vocabulary": 8, "capacity": "capsule"}
    source = NS(contract=contract, source_lock_digest="b" * 64,
                tokenizer_digest="c" * 64,
                runtime_contract=lambda *args: json.dumps(contract).encode())

    def execute():
        return transport.execute_request(experiment, source, [1, 2, 3], 1,
            {role: (f"http://127.0.0.1:{7300 + i}", str(i) * 32)
             for i, role in enumerate(p.ROLES)}, 23, state.attempt,
            completed_program=state.completed.append)

    state.execute = execute
    del client  # GC tests must not retain a hidden fixture-frame client root.
    yield state
    state.failures.clear()
    # Explicit test teardown only; never production automatic retry.
    for token in tuple(capacity._owners):
        if token.retained is not None:
            if not getattr(state, "real_shared", False):
                token.retained.drain_failed()
            # Real sticky failures deliberately keep this isolated ledger charged.
            # Discarding the fixture's ledger is NOT a successful refund oracle.
    if state.attempt is not None:
        state.attempt.close()
    if state.client is not None:
        if not getattr(state, "real_shared", False):
            state.client.close()


def _detach(error):
    """Remove every exception root before testing admission-ledger custody."""
    while error is not None:
        context = error.__context__
        error.__traceback__ = error.__context__ = error.__cause__ = None
        error = context


def _assert_payloads_cleared(error):
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code is transport.execute_request.__code__:
            assert all(frame.f_locals[name] is None for name in (
                "frame", "frames", "raw", "value", "storage", "manifests", "native", "peers", "payload"))
            return
        traceback = traceback.tb_next
    pytest.fail("actual execute_request frame missing")


@pytest.mark.parametrize("cut", ["admit", "bind"])
def test_fss_packed_payload_cleared_before_successful_refund(custody, cut):
    state = custody
    state.packed_cut, state.inspect_payload = cut, True
    with pytest.raises(ValueError, match="packed " + cut + " failed") as caught:
        state.execute()
    # Inspect escaped actual frame BEFORE detachment or GC, after real context
    # exit has refunded the original token. No copied frame/local surrogate.
    _assert_payloads_cleared(caught.value)
    assert not state.capacity._owners and state.refunds == 1
    assert len(state.tokens) == 1
    assert state.payload_checks == ["retire.entered"] + (
        ["native.close"] if cut == "bind" else []) + ["peers.close", "peers.audit"]
    assert state.frozen[0][:3] == (p.ROLES[0], cut, "failed")
    assert ("client", "retire", "validated") in state.markers
    _detach(caught.value)


@pytest.mark.parametrize("cut,failed", [
    (None, {"native"}), (None, {"peers"}), (None, {"native", "peers"}),
    ("admit", {"peers"}), ("bind", {"native"}), ("bind", {"peers"}),
    ("bind", {"native", "peers"}),
])
def test_fss_packed_payload_cleared_on_cleanup_failure(custody, failed, cut):
    state = custody
    state.failures, state.packed_cut, state.inspect_payload = set(failed), cut, True
    with pytest.raises(RuntimeError) as caught:
        state.execute()
    _assert_payloads_cleared(caught.value)
    token = next(iter(state.capacity._owners))
    reservation = state.refs["reservation"]()
    assert id(token) == state.tokens[0] and token.retained is reservation
    assert reservation._token is token
    assert reservation._failed == tuple(state.refs[name]() for name in ("native", "peers")
                                        if name in failed)
    assert state.capacity._owners == {token: state.amount} and state.refunds == 0
    assert state.payload_checks == ["retire.entered"] + (
        [] if cut == "admit" else ["native.close"]) + ["peers.close"] + (
        [] if "peers" in failed else ["peers.audit"])
    assert state.events.count(("peers", "close")) == 1
    assert state.events.count(("native", "close")) == (0 if cut == "admit" else 1)
    assert ("client", "retire", "validated") not in state.markers
    _detach(caught.value)


@pytest.mark.parametrize("failed", [{"native"}, {"peers"}, {"native", "peers"}])
@pytest.mark.parametrize("operation", [False, True])
def test_original_token_retains_only_failed_owners_and_preserves_chain(custody, failed, operation):
    state = custody
    state.failures, state.operation = set(failed), operation
    with pytest.raises(RuntimeError) as caught:
        state.execute()
    error = caught.value
    assert str(error) == ("peers" if "peers" in failed else "native") + " retirement failed"
    chain = []
    while error is not None:
        chain.append(str(error))
        assert error.__cause__ is None and not error.__suppress_context__
        error = error.__context__
    assert chain == (["peers retirement failed"] if "peers" in failed else []) + (
        ["native retirement failed"] if "native" in failed else []) + (
        ["operation failed"] if operation else [])
    assert state.events[:2] == [("native", "close"), ("peers", "close")]
    assert (("peers", "audit") in state.events) == ("peers" not in failed)
    assert state.frozen[0][:3] == ("client", "reconstruct" if operation else "retire", "failed")
    assert ("client", "retire", "validated") not in state.markers
    assert state.completed == ([] if operation else [b"native-manifest"])
    _assert_payloads_cleared(caught.value)
    _detach(caught.value)
    del caught
    gc.collect()
    reservation = state.refs["reservation"]()
    assert reservation is state.client._failed_request_cleanup
    assert id(reservation._token) == state.tokens[0]
    assert state.capacity._owners == {reservation._token: state.amount}
    assert reservation._token.retained is reservation
    assert reservation._failed == tuple(state.refs[name]() for name in ("native", "peers") if name in failed)
    for name in {"native", "peers"} - failed:
        assert state.refs[name]() is None  # audit success/failure is not ownership
    reservation.__exit__(None, None, None)
    state.attempt.close()
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        state.client.events({})
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(state.amount):
            pytest.fail("original charge must block another reservation")
    assert state.events.count(("native", "close")) == state.events.count(("peers", "close")) == 1
    state.failures.clear()
    state.client.close()  # explicit lifecycle drain, not a transport retry
    assert not state.capacity._owners and not reservation.pending_retirement
    state.client.close()
    assert all(state.events.count((name, "close")) == (2 if name in failed else 1)
               for name in ("native", "peers"))


@pytest.mark.parametrize("step", ["audit", "record"])
@pytest.mark.parametrize("native_failure", [False, True])
def test_audit_failure_overrides_native_without_retaining_closed_peers(custody, step, native_failure):
    state = custody
    setattr(state, step, True)
    state.failures = {"native"} if native_failure else set()
    with pytest.raises(LookupError, match=step + " failed") as caught:
        state.execute()
    context = caught.value.__context__
    assert (str(context) if context else None) == ("native retirement failed" if native_failure else None)
    _detach(caught.value)
    del caught, context
    gc.collect()
    assert state.refs["peers"]() is None
    if native_failure:
        reservation = state.refs["reservation"]()
        assert reservation._failed == (state.refs["native"](),)
        assert state.capacity._owners == {reservation._token: state.amount}
    else:
        assert not state.capacity._owners
        assert state.refs["reservation"]() is state.refs["native"]() is None
    assert ("client", "retire", "validated") not in state.markers


@pytest.mark.parametrize("handoff_failure", [False, True])
def test_ledger_alone_keeps_failed_owners_after_all_caller_roots_drop(custody, handoff_failure):
    state = custody
    state.failures, state.operation, state.handoff = {"native", "peers"}, True, handoff_failure
    with pytest.raises(RuntimeError, match="peers retirement failed") as caught:
        state.execute()
    if handoff_failure:
        assert caught.value.__notes__ == ["FuseFSS cleanup handoff failed: KeyboardInterrupt"]
        assert state.client._failed_request_cleanup is None
    assert str(caught.value.__context__) == "native retirement failed"
    assert str(caught.value.__context__.__context__) == "operation failed"
    _detach(caught.value)
    del caught
    state.attempt.close()
    # Break fixture/client closures too. Only module ledger may own payloads.
    state.client._failed_request_cleanup = None
    client_ref, attempt_ref = weakref.ref(state.client), weakref.ref(state.attempt)
    state.client = state.attempt = None
    gc.collect()
    assert client_ref() is attempt_ref() is None
    assert all(reference() is not None for reference in state.refs.values())
    token = next(iter(state.capacity._owners))
    assert id(token) == state.tokens[0]
    assert token.retained is state.refs["reservation"]()
    assert token.retained._failed == (state.refs["native"](), state.refs["peers"]())
    assert state.events == [("native", "close"), ("peers", "close")]
    # A failed explicit drain also keeps original charge and both owners.
    # Fake-owner custody oracle only: this does not prove real _Peers/_IOOwner
    # retirement or reclose behavior after their closed flags have been set.
    with pytest.raises(RuntimeError, match="native retirement failed") as caught:
        token.retained.drain_failed()
    _detach(caught.value)
    del caught
    gc.collect()
    assert state.capacity._owners == {token: state.amount}
    assert next(iter(state.capacity._owners)) is token
    assert id(token) == state.tokens[0]
    assert state.refs["native"]() is not None
    assert state.refs["peers"]() is not None
    assert state.refs["reservation"]() is not None
    assert token.retained is state.refs["reservation"]()
    assert token.retained._token is token
    assert token.retained._failed == (state.refs["native"](), state.refs["peers"]())
    assert state.events == [("native", "close"), ("peers", "close")] * 2
    assert state.events.count(("native", "close")) == 2
    assert state.events.count(("peers", "close")) == 2
    state.failures.clear()
    token.retained.drain_failed()
    gc.collect()
    assert not state.capacity._owners and token.retained is None
    assert all(reference() is None for reference in state.refs.values())


def test_success_returns_final_only_and_refunds_original_reservation(custody):
    state = custody
    assert state.execute().selected == (4,)
    assert state.completed == [b"native-manifest"]
    assert not state.capacity._owners and not state.frozen
    assert state.markers[-1] == ("client", "retire", "validated")
    assert state.events == [("native", "close"), ("peers", "close"), ("peers", "audit")]
    assert state.calls[:6] == [(role, action) for action in ("manifest", "admit") for role in p.ROLES]
    gc.collect()
    assert all(reference() is None for reference in state.refs.values())


def test_explicit_drain_restores_admission_then_success_refunds(custody):
    """Generic recoverable-owner mechanism, not real shared-owner recovery."""
    state = custody
    state.failures = {"native"}
    with pytest.raises(RuntimeError, match="native retirement failed") as caught:
        state.execute()
    _detach(caught.value)
    del caught
    original = state.client._failed_request_cleanup
    token = original._token
    state.attempt.close()
    state.failures.clear()
    original.drain_failed()  # explicit caller action; no implicit request retry
    assert not state.capacity._owners and token.retained is None
    state.attempt = state.client.events({})
    assert state.execute().selected == (4,)
    assert len(state.tokens) == 2 and state.tokens[1] != id(token)
    assert not state.capacity._owners


@pytest.fixture
def shared_custody(custody, monkeypatch):
    """Keep real constructors/close methods; replace only HTTP/native boundaries.

    Future approved execution imports the normal native module, but never calls
    its endpoint/native factory. Real selector creation may allocate socketpair;
    HTTP streams below cannot open network sockets. Every held producer is finite.
    """
    state = custody
    state.real_shared = True
    state.pool_failure = False
    state.hold_close = False
    state.entered = threading.Event()
    state.closers = []
    state.closer_errors = []  # strings only: never retain exception/traceback roots
    state.close_waiter = None
    state.before_finish = lambda: None
    native_type = transport._native.FssClientRequest

    class Native(native_type):
        def finish(self, *frames):
            state.before_finish()
            return super().finish(*frames)

    class Response:
        headers = {"x-pllm-server-ns": "1"}
        status_code = 200

        def __init__(self, role, action, payload):
            self.request = NS(content=payload)
            if action == "manifest":
                self.body = p.pack({"schema": "pllm.fss.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "b" * 64,
                    "tokenizer": "c" * 64, "transport": p.TRANSPORT,
                    "epoch": str(p.ROLES.index(role) + 1) * 64})
            elif action == "admit":
                self.body = p.pack({"manifest": b"native-manifest"})
            elif action == "finish":
                self.body = _receipt(role)
            else:
                self.body = p.pack({"prepared" if action == "prepare" else
                    "started" if action == "start" else "accepted": True})

        async def aiter_raw(self):
            yield self.body

    class Stream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, *args):
            return False

    class Http:
        def __init__(self, **options):
            self.index = len(state.http_created)
            state.http_created.append(self.index)

        def stream(self, method, path, *, content, **options):
            action = path.rsplit("/", 1)[-1]
            role = p.ROLES[self.index]
            state.calls.append((role, action))
            return Stream(Response(role, action, content))

        async def aclose(self):
            assert not state.capacity.lock._is_owned()
            state.events.append((f"pool_{self.index}", "aclose"))
            if self.index == 0:
                if state.hold_close:
                    state.close_waiter = asyncio.Event()
                    state.entered.set()
                    await asyncio.wait_for(state.close_waiter.wait(), timeout=5)
                if state.pool_failure:
                    raise RuntimeError("controlled HTTP pool retirement failed")

    state.http_created = []
    # The generic fixture installed a fake _Peers. Restore captured real class
    # BEFORE installing the factory; neither real close method is overridden.
    monkeypatch.setattr(transport, "_Peers", _REAL_PEERS)
    monkeypatch.setattr(shared.httpx, "AsyncClient", Http)
    monkeypatch.setattr(transport, "_native", NS(FssClientRequest=Native))

    real_start = _REAL_PEERS.start

    def peers_start(peers, *args, **options):
        try:
            real_start(peers, *args, **options)
        finally:
            # Shell already published to caller. Capture physical handles after
            # acquisition; retain no strong owner across yield.
            if peers.io is not None:
                state.io_thread, state.io_loop = peers.io.thread, peers.io.loop
                state.refs["peers"] = weakref.ref(peers)
                state.refs["io"] = weakref.ref(peers.io)
        assert type(peers) is _REAL_PEERS and type(peers.io) is _REAL_IO_OWNER
        assert type(peers).close is _REAL_PEERS.close
        assert type(peers.io).close is _REAL_IO_OWNER.close
        join = peers.io.thread.join

        def counted_join(timeout=None):
            state.events.append(("io", "join"))
            return join(timeout)

        monkeypatch.setattr(peers.io.thread, "join", counted_join)

    monkeypatch.setattr(_REAL_PEERS, "start", peers_start)
    try:
        yield state
    finally:
        failures = []
        state.hold_close = False  # A teardown-only first close must not add a hold.
        try:
            if state.close_waiter is not None and not state.io_loop.is_closed():
                state.io_loop.call_soon_threadsafe(state.close_waiter.set)
        except BaseException as error:
            failures.append("release pool waiter: " + type(error).__name__ + ": " + str(error))
            _detach(error)

        # Only an owner with NO admitted close may start normal shutdown here.
        # Sticky failure and an in-flight original closer must never be retried.
        try:
            peers = state.refs.get("peers", lambda: None)()
            io = state.refs.get("io", lambda: None)()
            if (peers is not None and io is not None and io.thread.is_alive()
                    and not peers.closed and not io.closed and not io._closing
                    and not peers._close_gate.locked() and not io._close_gate.locked()):
                def close_unadmitted():
                    try:
                        owner = state.refs["peers"]()
                        if owner is not None:
                            owner.close()
                    except BaseException as error:
                        failures.append("teardown first close: " + type(error).__name__ + ": " + str(error))
                        _detach(error)

                closer = threading.Thread(target=close_unadmitted,
                    name="fss-custody-teardown-close", daemon=False)
                state.closers.append(closer)
                closer.start()
        except BaseException as error:
            failures.append("start first shutdown: " + type(error).__name__ + ": " + str(error))
            _detach(error)
        finally:
            peers = io = None

        # One failed/unstarted helper must not prevent any other join or I/O
        # cleanup. Report only after all bounded cleanup attempts have run.
        for closer in state.closers:
            try:
                if closer.ident is not None:
                    closer.join(7)
            except BaseException as error:
                failures.append("join " + closer.name + ": " + type(error).__name__ + ": " + str(error))
                _detach(error)
        if hasattr(state, "io_thread"):
            try:
                if state.io_thread.is_alive() and not state.io_loop.is_closed():
                    # Test-only emergency physical shutdown, NOT logical close,
                    # success evidence, pool retirement, ledger drain or refund.
                    failures.append("emergency physical I/O stop required")
                    state.io_loop.call_soon_threadsafe(state.io_loop.stop)
            except BaseException as error:
                failures.append("stop I/O: " + type(error).__name__ + ": " + str(error))
                _detach(error)
            try:
                if state.io_thread.ident is not None:
                    state.io_thread.join(2)
            except BaseException as error:
                failures.append("join I/O: " + type(error).__name__ + ": " + str(error))
                _detach(error)
            try:
                if state.io_thread.is_alive():
                    failures.append("I/O thread still alive")
                elif not state.io_loop.is_closed():
                    state.io_loop.close()  # physical disposal only; no witness writes
            except BaseException as error:
                failures.append("dispose I/O loop: " + type(error).__name__ + ": " + str(error))
                _detach(error)
        for closer in state.closers:
            try:
                if closer.is_alive():
                    closer.join(2)
                if closer.is_alive():
                    failures.append(closer.name + " still alive")
            except BaseException as error:
                failures.append("final join " + closer.name + ": " + type(error).__name__ + ": " + str(error))
                _detach(error)
        failures.extend("original closer: " + error for error in state.closer_errors)
        assert not failures, "fixture cleanup failures: " + "; ".join(failures)
        # No second admitted real close, witness writes or ledger mutation/refund.


def _shared_charge(state, names):
    reservation = state.refs["reservation"]()
    assert reservation is not None
    token = next(iter(state.capacity._owners))
    assert id(token) == state.tokens[0]
    assert reservation._token is token and token.retained is reservation
    assert state.capacity._owners == {token: state.amount}
    assert sum(state.capacity._owners.values()) == state.amount
    assert reservation.pending_retirement
    assert all(state.refs[name]() is not None for name in names)
    assert reservation._failed == tuple(state.refs[name]() for name in names)
    return reservation


@pytest.mark.parametrize("native_failure", [False, True])
def test_real_shared_failed_close_stays_charged_after_gc_and_explicit_drain(
        shared_custody, native_failure):
    state = shared_custody
    state.operation = state.pool_failure = True
    state.failures = {"native"} if native_failure else set()
    with pytest.raises(RuntimeError, match="controlled HTTP pool retirement failed") as caught:
        state.execute()
    assert str(caught.value.__context__) == (
        "native retirement failed" if native_failure else "operation failed")
    if native_failure:
        assert str(caught.value.__context__.__context__) == "operation failed"
    _assert_payloads_cleared(caught.value)
    _detach(caught.value)
    del caught
    state.attempt.close()
    state.client._failed_request_cleanup = None
    client_ref, attempt_ref = weakref.ref(state.client), weakref.ref(state.attempt)
    state.client = state.attempt = None
    gc.collect()
    # Failed peers legitimately retain the cancellation handle and its client.
    assert attempt_ref() is not None
    assert state.refs["peers"]().cancelled is attempt_ref()
    assert attempt_ref().owner is client_ref()
    names = ("native", "peers") if native_failure else ("peers",)
    reservation = _shared_charge(state, names)
    assert state.refs["io"]() is state.refs["peers"]().io
    if not native_failure:
        assert state.refs["native"]() is None
    assert not state.io_thread.is_alive() and state.io_loop.is_closed()
    assert not state.refs["peers"]()._close_succeeded
    assert not state.refs["io"]()._close_succeeded
    assert state.events == [("native", "close"), ("pool_0", "aclose"),
                            ("pool_1", "aclose"), ("pool_2", "aclose"), ("io", "join")]
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(state.amount):
            pytest.fail("sticky real owner must block follow-on admission")
    token = reservation._token
    del reservation
    # Physical cleanup completed; it does not turn a failed close into success.
    # Only the generic fake native may recover on the next explicit drain.
    state.failures.clear()
    for drain in range(2):
        with pytest.raises(ProtocolError, match="retirement previously failed") as caught:
            token.retained.drain_failed()
        _detach(caught.value)
        del caught
        gc.collect()
        assert _shared_charge(state, ("peers",))._token is token
        assert state.refs["native"]() is None
        assert state.refs["io"]() is state.refs["peers"]().io
        assert state.events.count(("native", "close")) == (2 if native_failure else 1)
        assert all(state.events.count((f"pool_{i}", "aclose")) == 1 for i in range(3))
        assert state.events.count(("io", "join")) == 1


def test_real_shared_inflight_close_rejection_retains_until_original_success(
        shared_custody):
    state = shared_custody
    state.operation = state.hold_close = True

    def start_original_close():
        def close():
            try:
                state.refs["peers"]().close()
            except BaseException as error:
                state.closer_errors.append(type(error).__name__ + ": " + str(error))
                _detach(error)

        closer = threading.Thread(target=close, name="fss-custody-original-close", daemon=False)
        state.closers.append(closer)
        closer.start()
        assert state.entered.wait(2), "real pool cleanup never entered"

    state.before_finish = start_original_close
    # FSS cleanup loses the close gate to the already-admitted original closer.
    # Its rejection is not an admitted close failure and must not poison success.
    with pytest.raises(ProtocolError, match="retirement is already active") as caught:
        state.execute()
    assert str(caught.value.__context__) == "operation failed"
    _detach(caught.value)
    del caught
    reservation = _shared_charge(state, ("peers",))
    assert reservation is state.client._failed_request_cleanup
    token = reservation._token
    state.attempt.close()
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        state.client.events({})
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        with resources.reserve_incremental_memory(state.amount):
            pytest.fail("inflight real close must remain charged")

    def drain():
        try:
            reservation.drain_failed()
        except BaseException as error:
            drain_errors.append((type(error), str(error)))
            _detach(error)

    drain_errors = []
    drainer = threading.Thread(target=drain, name="fss-custody-drain", daemon=False)
    state.closers.append(drainer)
    drainer.start()
    drainer.join(1)
    assert not drainer.is_alive(), "concurrent drain blocked behind original close"
    assert drain_errors == [(ProtocolError, "GKM worker channel retirement is already active")]
    assert _shared_charge(state, ("peers",)) is reservation
    assert not state.refs["peers"]()._close_succeeded
    assert not state.refs["io"]()._close_succeeded
    assert state.events.count(("pool_0", "aclose")) == 1
    state.io_loop.call_soon_threadsafe(state.close_waiter.set)
    state.closers[0].join(7)
    assert not state.closers[0].is_alive() and not state.closer_errors
    assert not state.io_thread.is_alive() and state.io_loop.is_closed()
    assert state.refs["peers"]()._close_succeeded
    assert state.refs["io"]()._close_succeeded
    assert _shared_charge(state, ("peers",)) is reservation
    cleanup_events = list(state.events)
    reservation.drain_failed()
    assert not state.capacity._owners and token.retained is None
    assert not reservation.pending_retirement and reservation._failed == ()
    assert state.events == cleanup_events  # success witness: no second physical cleanup/join
    state.attempt = state.client.events({})
    state.attempt.close()
