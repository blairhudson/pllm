"""Canonical compact storage selection; metadata/mocks only, no native execution."""
import asyncio
from contextlib import contextmanager
import gc
import json
import threading
from types import SimpleNamespace
import weakref

import httpx
import pytest
from pllm.deployment import resources

from pllm.configuration import ComponentRef, Experiment
from pllm.profiles import resolve_runtime_composition
from pllm.protocols import PumaThreePartyDecoder
from pllm.runtime import replicated_source
from pllm.runtime.replicated_protocol import (
    execution_binding, execution_policy, storage_policy, validate_execution_binding,
)
from test_replicated_execution_policy import experiment_for


@pytest.fixture(autouse=True)
def _isolated_cleanup_admission(monkeypatch):
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=1 << 40))
    yield
    assert not capacity._cleanup, "test leaked durable cleanup; never discard live payloads"


def _scope():
    return resources.reserve_cleanup_slot()


@pytest.mark.parametrize("choices", [{}, {"weights": "public"},
    {"weights": "public", "range_policy": "checked"}])
def test_expanded_omitted_and_explicit_keep_historical_identity(choices):
    implicit = experiment_for(**choices)
    explicit = experiment_for(**choices, storage="expanded")
    assert implicit.canonical_bytes() == explicit.canonical_bytes()
    assert implicit.configuration_digest() == explicit.configuration_digest()
    assert implicit.pipeline.digest() == explicit.pipeline.digest()
    assert "storage" not in implicit.pipeline.components["linear"].params
    assert execution_policy(implicit) == execution_policy(explicit)
    assert storage_policy(implicit) == "expanded"


def test_historical_component_spec_golden():
    assert PumaThreePartyDecoder().to_spec() == {
        "component": "pllm/puma-three-party-decoder/v1", "params": {}}
    assert PumaThreePartyDecoder(weights="public", range_policy="checked").to_spec() == {
        "component": "pllm/puma-three-party-decoder/v1",
        "params": {"weights": "public", "range_policy": "checked", "max_abs_activation": 256}}


def test_compact_roundtrip_binds_composition_not_arithmetic_policy():
    expanded = experiment_for(weights="public", range_policy="checked")
    compact = experiment_for(weights="public", range_policy="checked", storage="compact")
    restored = Experiment.from_spec(json.loads(compact.canonical_bytes()))
    assert restored.canonical_bytes() == compact.canonical_bytes()
    assert compact.configuration_digest() != expanded.configuration_digest()
    assert compact.pipeline.digest() != expanded.pipeline.digest()
    assert execution_policy(compact) == execution_policy(expanded)
    assert execution_binding(compact) == execution_binding(expanded)
    assert "storage" not in execution_binding(compact)
    assert resolve_runtime_composition(compact.pipeline).numeric_profile == resolve_runtime_composition(expanded.pipeline).numeric_profile
    assert storage_policy(restored) == "compact"
    component = compact.pipeline.components["linear"]
    with pytest.raises((AttributeError, TypeError)):
        component.storage = "expanded"
    assert PumaThreePartyDecoder(**component.get_params()) == component


@pytest.mark.parametrize("storage", [None, False, 1, [], "Compact", "tiled", ""])
def test_unknown_storage_refuses(storage):
    with pytest.raises(ValueError):
        PumaThreePartyDecoder(weights="public", range_policy="checked", storage=storage)


@pytest.mark.parametrize("changes", [{"weights": "private"}, {"range_policy": "static"},
    {"lookup": "tiled"}, {"lookup": "trie"}, {"selection": "tiled"}, {"peer_transport": "streams"}])
def test_unsupported_compact_combination_refuses_before_admission(changes):
    choices = {"weights": "public", "range_policy": "checked", "storage": "compact", **changes}
    with pytest.raises(ValueError):
        experiment_for(**choices)


def test_redundant_storage_component_cannot_bypass_canonical_gate():
    component = ComponentRef(PumaThreePartyDecoder.descriptor.component,
                             {"weights": "public", "storage": "expanded"})
    malformed = SimpleNamespace(pipeline=SimpleNamespace(components={"linear": component}))
    with pytest.raises(ValueError, match="noncanonical"):
        storage_policy(malformed)


def test_storage_cannot_be_supplied_as_arithmetic_wire_permission():
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    with pytest.raises(ValueError, match="execution binding"):
        validate_execution_binding(experiment, {
            "execution": {**execution_binding(experiment), "storage": "compact"}})


@pytest.mark.parametrize("native_boundary", ["missing", "refuses"])
def test_compact_never_uses_legacy_publisher_or_reads_source(monkeypatch, native_boundary):
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("source_or_legacy")
        raise AssertionError("legacy native publisher/source access forbidden")

    def refusal(*args, **kwargs):
        calls.append("native_refusal")
        raise ValueError("native compact factory refused")

    class Native:
        __new__ = staticmethod(forbidden)

    if native_boundary == "refuses":
        Native._from_compact_source_request = staticmethod(refusal)
    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=Native))
    monkeypatch.setattr(replicated_source.ReplicatedSource, "runtime_contract", forbidden)
    monkeypatch.setattr(replicated_source.ReplicatedSource, "_append_source", forbidden)
    @contextmanager
    def memory(*args):
        calls.append("caller_admitted")
        try:
            yield
        finally:
            calls.append("caller_refunded")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    source = SimpleNamespace(experiment_bytes=experiment.canonical_bytes(), config_bytes=b"admitted config",
        source_lock_digest="a" * 64, runtime_contract=forbidden, _append_source=forbidden)
    error_type = RuntimeError if native_boundary == "missing" else ValueError
    message = "native publisher capability unavailable" if native_boundary == "missing" else "native compact factory refused"
    with pytest.raises(error_type, match=message):
        scope = _scope()
        try:
            replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2, lambda: False,
                cleanup_scope=scope)
        finally:
            scope.drain()
    assert calls == ([] if native_boundary == "missing" else
                     ["caller_admitted", "native_refusal", "caller_refunded"])


@pytest.mark.parametrize("outputs", [1, 3, True, 2.0])
def test_compact_unsupported_output_count_refuses_before_factory(monkeypatch, outputs):
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported workload reached native/caller/source admission")

    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=forbidden))
    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", forbidden)
    with pytest.raises(ValueError, match="exactly two outputs"):
        replicated_source.ReplicatedSource.publisher(SimpleNamespace(), experiment, [2, 2, 3, 3], outputs, forbidden)


@pytest.mark.parametrize("failure", [None, "filesystem", "directory_refused", "disk", "append", "cancel"])
def test_compact_actual_factory_signature_and_resource_order_mock(monkeypatch, failure):
    """ABI/ownership mocks only; no source validation, compilation or IDs proof."""
    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    encoded = experiment.canonical_bytes()
    config = b"exact admitted client config"
    events = []

    @contextmanager
    def memory(*args):
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(amount, *, directory):
        assert amount == 40 and directory == "/actual-native-directory-mocked"
        events.append("disk_admission")
        if failure == "disk":
            raise ValueError("disk refused")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        collection_disk_bytes = 40
        shapes = ()

        @property
        def collection_disk_directory(self):
            if failure == "filesystem":
                raise AttributeError("collection_disk_directory unavailable")
            if failure == "directory_refused":
                raise ValueError("directory refused")
            return "/actual-native-directory-mocked"

        def append(self, *args):
            events.append("append")
            if failure == "append":
                raise ValueError("append refused")

        def seal(self, ids):
            assert ids == b"".join(token.to_bytes(4, "little") for token in [2, 2, 3, 3])
            events.append("seal")
            return json.dumps({"execution": execution_binding(experiment)}).encode()

        def close(self):
            events.append("payload_closed")

    class Native:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("legacy constructor fallback")

        @staticmethod
        def _from_compact_source_request(actual_experiment, actual_config, rows, outputs, source, composition):
            assert events == ["caller_admitted"]
            assert actual_experiment is encoded and actual_config is config
            assert (rows, outputs) == (4, 2)
            assert source == bytes.fromhex("a" * 64)
            assert composition == bytes.fromhex(experiment.pipeline.digest())
            events.append("factory")
            return NativeOwner()

    def runtime(*args):
        assert events.index("disk_admission") < len(events)
        events.append("runtime")

    def append(shapes, callback, cancelled):
        events.append("collection")
        callback()  # IO mock only, not an actual source frame

    source = SimpleNamespace(experiment_bytes=encoded, config_bytes=config,
        source_lock_digest="a" * 64, runtime_contract=runtime, _append_source=append)
    monkeypatch.setattr(replicated_source, "_native", SimpleNamespace(ReplicatedPublisher=Native))
    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    if failure:
        expected = AttributeError if failure == "filesystem" else ValueError
        message = {"filesystem": "collection_disk_directory unavailable", "directory_refused": "directory refused", "disk": "disk refused",
                   "append": "append refused", "cancel": "import cancelled"}[failure]
        with pytest.raises(expected, match=message):
            replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2,
                lambda: failure == "cancel", cleanup_scope=_scope())
        if failure in {"filesystem", "directory_refused", "disk"}:
            assert "collection" not in events and "runtime" not in events
        assert "seal" not in events
    else:
        owner, manifest = replicated_source.ReplicatedSource.publisher(source, experiment, [2, 2, 3, 3], 2, lambda: False,
            cleanup_scope=_scope())
        assert json.loads(manifest)["execution"] == execution_binding(experiment)
        assert "caller_refunded" not in events and "disk_refunded" not in events
        owner.close()
    assert events.index("payload_closed") < events.index("caller_refunded")
    if "disk_refunded" in events:
        assert events.index("payload_closed") < events.index("disk_refunded")


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["expanded_composition", "unsupported_execution"])
async def test_compact_server_rejects_binding_before_constructor_or_import(monkeypatch, tamper):
    from pllm.runtime import replicated_worker as worker
    from pllm.runtime.delegated_protocol import pack, unpack

    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    expanded = experiment_for(weights="public", range_policy="checked")
    events = []

    def forbidden(*args, **kwargs):
        raise AssertionError("rejected binding reached native constructor/import")

    class Native:
        __new__ = staticmethod(forbidden)
        _from_source_request = staticmethod(forbidden)

    class Reservation:
        def __enter__(self):
            events.append("reserved")
            return self

        def __exit__(self, *exc):
            events.append("refunded")

    retained_experiment = experiment.canonical_bytes()
    source = SimpleNamespace(source_lock_digest="a" * 64, config_bytes=b"retained server config",
        experiment_bytes=retained_experiment,
        contract={"weights": "public", "public_weight_snapshot_bytes_per_worker": 1},
        runtime_contract=lambda *args: None, populate_public_worker=forbidden,
        runtime_contract_digest="b" * 64, tokenizer_digest="d" * 64)

    def admit(selected):
        assert selected.canonical_bytes() == retained_experiment
        return source  # admission IO/native contract deliberately mocked

    monkeypatch.setattr(worker, "admit_source", admit)
    monkeypatch.setattr(worker._native, "ReplicatedWorker", Native)
    monkeypatch.setattr(worker, "reserve_incremental_memory", lambda *args: Reservation())
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda *args: SimpleNamespace(free=1 << 40))
    key = "compact-test-client-12345678"
    peers = {"worker_a": ("http://127.0.0.1:1", "outgoing-a-12345678", "incoming-a-12345678"),
             "worker_c": ("http://127.0.0.1:2", "outgoing-c-12345678", "incoming-c-12345678")}
    app = worker.create_replicated_worker_app(experiment, "worker_b", key, peers)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app), base_url="http://test",
            headers={"authorization": f"Bearer {key}"}) as client:
        epoch = unpack((await client.get("/v1/replicated/manifest")).content)["epoch"]
        value = {"session": list(bytes.fromhex("c" * 64)), "source": list(bytes.fromhex("a" * 64)),
            "composition": list(bytes.fromhex(experiment.pipeline.digest())),
            "config": list(source.config_bytes), "weights": "public", "rows": 2, "outputs": 2,
            "execution": execution_binding(experiment),
            "storage": "expanded"}  # untrusted selector cannot override SERVER compact policy
        if tamper == "expanded_composition":
            value["composition"] = list(bytes.fromhex(expanded.pipeline.digest()))
        else:
            value["execution"]["lookup"] = "tiled"
        response = await client.post("/v1/replicated/admit", content=pack({
            "request": "c" * 64, "epoch": epoch, "manifest": json.dumps(value).encode()}))
        assert response.status_code == 400
        metrics = await client.get("/v1/replicated/metrics")
        metrics.raise_for_status()
        assert metrics.json()["active_sessions"] == 0
        assert events == ["reserved", "refunded"]
        assert "selected" not in response.text and "token_ids" not in response.text


@pytest.mark.parametrize("failure", [None, "disk", "append"])
def test_compact_caller_and_disk_owners_retire_payload_before_refund(monkeypatch, failure):
    events = []
    directory = "/not-opened/native-snapshot-filesystem"

    @contextmanager
    def memory(amount):
        from pllm.runtime.replicated_protocol import REQUEST_MEMORY
        assert amount == REQUEST_MEMORY
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(amount, *, directory):
        assert amount == 40 and directory == "/not-opened/native-snapshot-filesystem"
        events.append("disk_admission")
        if failure == "disk":
            raise ValueError("disk refused")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        def append(self):
            events.append("append")
            if failure == "append":
                raise ValueError("append refused")

        def seal(self):
            events.append("seal")

        def close(self):
            events.append("payload_closed")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    owner = replicated_source._CompactPublisherOwner(_scope())
    assert events == ["caller_admitted"]  # must precede future real native factory
    owner._retain(NativeOwner())
    with pytest.raises(ValueError, match="disk not admitted"):
        owner.append()
    try:
        if failure:
            with pytest.raises(ValueError, match=f"{failure} refused"):
                owner._admit_disk(40, directory=directory)
                owner.append()
        else:
            owner._admit_disk(40, directory=directory)
            owner.append()
            owner.seal()
            assert "caller_refunded" not in events and "disk_refunded" not in events
    finally:
        owner.close()
    assert events.index("payload_closed") < events.index("caller_refunded")
    if failure != "disk":
        assert events.index("payload_closed") < events.index("disk_refunded")
    else:
        assert "append" not in events
    previous = events[:]
    owner.close()
    assert events == previous
    with pytest.raises(ValueError, match="retired"):
        owner.append()
    with pytest.raises(ValueError, match="already retained"):
        owner._retain(NativeOwner())


@pytest.mark.parametrize("error_type", [ValueError, asyncio.CancelledError])
def test_compact_cleanup_failure_preserves_error_and_holds_charges(monkeypatch, error_type):
    refunded = []

    @contextmanager
    def lease(*args, **kwargs):
        try:
            yield
        finally:
            refunded.append(True)

    class NativeOwner:
        refuses = True

        def close(self):
            if self.refuses:
                raise RuntimeError("native close failed")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", lease)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", lease)
    owner = replicated_source._CompactPublisherOwner(_scope())
    native = NativeOwner()
    owner._retain(native)
    owner._admit_disk(40, directory="/not-opened")
    original = error_type("collection cancelled or failed")
    owner._close_preserving(original)
    assert not refunded and owner._owner is native
    assert "RuntimeError" in original.__notes__[0]
    with pytest.raises(error_type) as captured:
        raise original
    assert captured.value is original
    native.refuses = False
    owner.close()
    assert len(refunded) == 2


def test_compact_failed_close_survives_caller_and_traceback_gc_until_explicit_drain(monkeypatch):
    """H716 custody regression; mocks actual resource owner, NOT native evidence.

    No external owner/exception/traceback root may make this pass. The actual
    admission ledger must retain the slot, not a per-client cycle alone.
    """
    events = []

    @contextmanager
    def memory(*args):
        events.append("caller_admitted")
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(*args, **kwargs):
        events.append("disk_admitted")
        try:
            yield
        finally:
            events.append("disk_refunded")

    class NativeOwner:
        refuses = True

        def close(self):
            if self.refuses:
                events.append("close_failed")
                raise RuntimeError("native close failed")
            events.append("payload_retired")

    class ObservedOwner(replicated_source._CompactPublisherOwner):
        # Permit weak observation, not strong custody, without changing the
        # production slots/destructor or substituting its resource methods.
        pass

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)

    def fail_and_drop_all_caller_roots():
        owner = ObservedOwner(_scope())
        native = NativeOwner()
        owner._retain(native)
        owner._admit_disk(40, directory="/not-opened")
        references = weakref.ref(owner), weakref.ref(native)
        original = ValueError("original work failed")
        try:
            raise original
        except ValueError as caught:
            owner._close_preserving(caught)
            preserved = caught is original and str(caught) == "original work failed"
            assert "RuntimeError" in caught.__notes__[0]
        # Exceptions/tracebacks must NOT act as the hidden failed-owner root.
        original.__traceback__ = original.__context__ = original.__cause__ = None
        del original, owner, native
        return references, preserved

    (owner_ref, native_ref), preserved = fail_and_drop_all_caller_roots()
    gc.collect()
    assert preserved
    retained = owner_ref()
    try:
        assert retained is not None and native_ref() is not None, (
            f"failed native payload/leases lost durable custody after GC: {events}")
        assert "caller_refunded" not in events and "disk_refunded" not in events
        assert retained._cleanup_scope in resources._capacity._cleanup.values()
        assert sum(resources._capacity._owners.values()) == resources._capacity._owners[retained._cleanup_scope._token]
        assert sum(resources._capacity._owners.values()) >= 64 << 10
        native_ref().refuses = False
        retained.close()  # explicit successful cleanup, never timeout-as-success
        retained.close()  # idempotent cleanup must not refund twice
        assert events.count("payload_retired") == 1
        assert events.count("caller_refunded") == events.count("disk_refunded") == 1
        assert events.index("payload_retired") < events.index("disk_refunded") < events.index("caller_refunded")
    finally:
        # Test cleanup if an assertion fails after a genuine retained owner is
        # observed; does not establish the missing production handoff itself.
        if retained is not None and retained._owner is not None:
            retained._owner.refuses = False
            retained.close()


def _lifecycle_client():
    from pllm.runtime.delegated_client import DelegatedClient
    # Actual lifecycle methods; admission/tokenizer/model construction absent.
    client = object.__new__(DelegatedClient)
    client._lock = threading.RLock()
    client._closing = False
    client._active = client._compact_cleanup = client._failed_request_cleanup = None
    client._tokenizer_scopes = set()
    client.records = {}
    client._source = object()
    client._result_hook = None
    return client


def test_compact_pending_client_close_drains_outside_locks_and_blocks_new_request(monkeypatch):
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    events = []
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    client._active = attempt
    slot = client._new_compact_cleanup_scope(attempt)
    original_source = client._source

    @contextmanager
    def memory(*args):
        try:
            yield
        finally:
            events.append("caller_refunded")

    @contextmanager
    def disk(*args, **kwargs):
        try:
            yield
        finally:
            events.append("disk_refunded")

    class Native:
        refuses = True

        def close(self):
            assert not client._lock._is_owned()
            assert not attempt.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert attempt.closed and client._active is None
            assert slot._lock.acquire(blocking=False)
            slot._lock.release()
            if self.refuses:
                raise RuntimeError("native drain refused")
            events.append("payload_retired")

    monkeypatch.setattr(replicated_source, "reserve_incremental_memory", memory)
    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    owner = replicated_source._CompactPublisherOwner(slot)
    native = Native()
    owner._retain(native)
    owner._admit_disk(40, directory="/not-opened")
    original = ValueError("original operation failed")
    owner._close_preserving(original)
    assert "RuntimeError" in original.__notes__[0]
    client._clear_owned_state()
    assert client._source is original_source and client._compact_cleanup is slot
    with pytest.raises(ProtocolError, match="pending compact cleanup"):
        client.events({})
    with pytest.raises(RuntimeError, match="native drain refused"):
        client.close()
    assert attempt.closed and client._active is None
    assert not events and not slot.closed and resources._capacity._cleanup
    native.refuses = False
    client.close()
    client.close()
    assert events == ["payload_retired", "disk_refunded", "caller_refunded"]
    assert client._compact_cleanup is client._source is None
    assert not resources._capacity._owners and not resources._capacity._cleanup


def test_cleanup_slot_control_is_admitted_before_allocation_and_one_short_refuses(monkeypatch):
    capacity = resources._capacity
    original = resources._CleanupSlot
    allocations = []
    quote = capacity._cleanup_control_bytes()

    def allocate(actual_capacity, token):
        assert actual_capacity is capacity
        assert capacity._owners[token] == quote
        allocations.append(token)
        return original(actual_capacity, token)

    monkeypatch.setattr(resources, "_CleanupSlot", allocate)
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=quote - 1))
    with pytest.raises(resources.NetworkError, match="CLIENT_MEMORY_CAPACITY_CHANGED"):
        resources.reserve_cleanup_slot()
    assert not allocations and not capacity._owners
    monkeypatch.setattr(capacity, "observe", lambda: SimpleNamespace(headroom_bytes=quote))
    slot = resources.reserve_cleanup_slot()
    assert len(allocations) == 1 and capacity._cleanup
    slot.drain()
    slot.drain()
    assert not capacity._owners


def test_replicated_transport_preserves_work_error_and_cleans_peers_after_close_failure(monkeypatch):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    experiment = experiment_for(weights="public", range_policy="checked")
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    attempt_ref = weakref.ref(attempt)
    client._active = attempt
    original_source = client._source
    original = ValueError("original work exception")
    events = []
    references = []

    def memory(amount):
        reservation = resources.reserve_incremental_memory(amount)
        references.append(weakref.ref(reservation))
        return reservation

    class Publisher:
        refuses = True
        _storage_kind = "compact"  # a flag/echo is NOT independently rooted custody

        def __init__(self):
            references.append(weakref.ref(self))

        def close(self):
            pending = attempt_ref()
            assert not client._lock._is_owned()
            assert pending is None or not pending.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert (pending is None or pending.closed) and client._active is None
            if self.refuses:
                events.append("publisher_close_failed")
                raise RuntimeError("publisher cleanup failed")
            events.append("publisher_retired")

    class Peers:
        refuses = True

        def __init__(self, *args, **kwargs):
            references.append(weakref.ref(self))

        def post(self, *args, **kwargs):
            raise original

        def close(self):
            pending = attempt_ref()
            assert not client._lock._is_owned()
            assert pending is None or not pending.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            if client._closing:
                assert (pending is None or pending.closed) and client._active is None
            if self.refuses:
                events.append("peers_close_failed")
                raise RuntimeError("peers cleanup failed")
            events.append("peers_retired")

        def audit(self):
            events.append("audit")
            return {}

    source = SimpleNamespace(publisher=lambda *args: (Publisher(), json.dumps({"session": [0] * 32}).encode()))
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_Peers", Peers)
    with pytest.raises(ValueError) as caught:
        transport.execute_request(experiment, source, [2, 2, 3, 3], 2, {}, 1, attempt)
    assert caught.value is original and len(original.__notes__) == 2
    original.__traceback__ = original.__context__ = original.__cause__ = None
    del caught
    original = None
    gc.collect()
    reservation, peers, publisher = (reference() for reference in references)
    assert reservation is not None and publisher is not None and peers is not None
    assert events == ["publisher_close_failed", "peers_close_failed", "audit"]
    assert resources._capacity._owners[reservation._token] == transport.REQUEST_MEMORY
    assert reservation._token.retained is reservation
    assert client._failed_request_cleanup is reservation
    assert client._compact_cleanup is None
    reservation.__exit__(None, None, None)  # failed close is never successful refund
    assert resources._capacity._owners
    client._clear_owned_state()
    assert client._source is original_source
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        client.events({})
    with pytest.raises(ProtocolError, match="pending request cleanup"):
        client._check_open()  # ordinary admission, not just iterator construction
    with pytest.raises(RuntimeError, match="publisher cleanup failed"):
        client.close()  # active request must drain before BOTH failed owners
    assert attempt.closed and client._active is None
    assert resources._capacity._owners and client._source is original_source
    publisher.refuses = peers.refuses = False
    # Caller/request/reservation/publisher references disappear. Production
    # client teardown, NOT a manual reservation recovery, must do the draining.
    del publisher, peers, reservation, attempt, source
    gc.collect()
    assert all(reference() is not None for reference in references)
    client.close()
    client.close()
    assert events[-2:] == ["publisher_retired", "peers_retired"]
    assert not resources._capacity._owners
    assert client._failed_request_cleanup is client._source is None


@pytest.mark.parametrize("peer_failure", [False, True])
def test_only_actual_rooted_compact_publisher_can_release_outer_after_close_failure(monkeypatch, peer_failure):
    from pllm.runtime import replicated_transport as transport
    from pllm.runtime.delegated_client import _Attempt

    experiment = experiment_for(weights="public", range_policy="checked", storage="compact")
    original = ValueError("work failed")
    client = _lifecycle_client()
    attempt = _Attempt(client, {})
    client._active = attempt
    events, reservations = [], []

    @contextmanager
    def disk(*args, **kwargs):
        try:
            yield
        finally:
            events.append("disk_refunded")

    class Native:
        refuses = True

        def close(self):
            if self.refuses:
                raise RuntimeError("native close failed")
            events.append("payload_retired")

    class Peers:
        refuses = peer_failure

        def __init__(self, *args, **kwargs):
            pass

        def post(self, *args, **kwargs):
            raise original

        def close(self):
            events.append("peers_close")
            if self.refuses:
                raise RuntimeError("peers close failed")

        def audit(self):
            return {}

    def memory(amount):
        reservation = resources.reserve_incremental_memory(amount)
        reservations.append(reservation)
        return reservation

    monkeypatch.setattr(replicated_source, "reserve_incremental_disk", disk)
    publisher = replicated_source._CompactPublisherOwner(client._new_compact_cleanup_scope(attempt))
    native = Native()
    publisher._retain(native)
    publisher._admit_disk(40, directory="/not-opened")
    before = sum(resources._capacity._owners.values())
    source = SimpleNamespace(publisher=lambda *args: (publisher, json.dumps({"session": [0] * 32}).encode()))
    monkeypatch.setattr(transport, "reserve_incremental_memory", memory)
    monkeypatch.setattr(transport, "_Peers", Peers)
    with pytest.raises(ValueError) as caught:
        transport.execute_request(experiment, source, [2, 2, 3, 3], 2, {}, 1, attempt)
    assert caught.value is original
    outer = reservations[0]
    assert replicated_source._independently_rooted_compact_publisher(publisher)
    assert sum(resources._capacity._owners.values()) == before + (transport.REQUEST_MEMORY if peer_failure else 0)
    assert "disk_refunded" not in events
    if peer_failure:
        assert outer._token.retained is outer and len(outer._failed) == 1
        assert client._failed_request_cleanup is outer
    with pytest.raises(RuntimeError, match="native close failed"):
        client.close()  # tries failed peer too, even when compact cleanup fails
    if peer_failure:
        assert events.count("peers_close") == 2
        assert outer.pending_retirement
        Peers.refuses = False
    native.refuses = False
    client.close()
    client.close()
    assert events.index("payload_retired") < events.index("disk_refunded")
    assert not resources._capacity._owners


def test_failed_outer_handoff_while_client_closing_uses_only_actual_active_seat():
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.delegated_client import _Attempt

    client = _lifecycle_client()
    active, foreign = _Attempt(client, {}), _Attempt(client, {})
    client._active = active
    retired = []

    class Payload:
        def close(self):
            assert active.closed and client._active is None
            assert not client._lock._is_owned() and not active.lock._is_owned()
            assert not resources._capacity.lock._is_owned()
            retired.append(True)

    reservation = resources.reserve_incremental_memory(replicated_source.REQUEST_MEMORY)
    reservation.__enter__()
    reservation.retain_failed((Payload(),))
    # Models close() having signalled cancellation while waiting for the
    # executing producer; its finally must still hand custody to this client.
    client._closing = True
    with pytest.raises(ProtocolError, match="active request seat"):
        client._retain_failed_request_reservation(foreign, reservation)
    assert reservation.pending_retirement and client._failed_request_cleanup is None
    client._retain_failed_request_reservation(active, reservation)
    foreign.close()
    client.close()  # ONLY production boundary invokes the explicit drain
    client.close()
    assert retired == [True] and not resources._capacity._owners
    assert client._failed_request_cleanup is None
