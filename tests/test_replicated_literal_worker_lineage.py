"""One literal toy shard through the SDK publisher and an ordinary PUMA worker."""

from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from types import SimpleNamespace

import httpx
import psutil
import pytest

from pllm.deployment import resources
from pllm.runtime import benchmark_memory
from pllm.runtime import delegated_source as shard_io, replicated_worker as worker
from pllm.runtime.delegated_protocol import pack, unpack
from pllm.runtime.replicated_protocol import ROLES
from pllm.runtime.replicated_source import ReplicatedSource, admit_source
from test_replicated_sdk import experiment_fixture


_SHARD = "model.safetensors"
_HEADER_LEN = 2048
_PAYLOAD = 8 + _HEADER_LEN
_EMBED = "model.embed_tokens.weight"
_NORM = "model.norm.weight"
_LAYER = "model.layers.0."
_ATTN = _LAYER + "self_attn."
_GATE, _UP = _LAYER + "mlp.gate_proj.weight", _LAYER + "mlp.up_proj.weight"


@dataclass(frozen=True)
class Record:
    name: str
    shape: tuple[int, ...]
    dtype: str
    position: int  # Relative to the fixed payload start, not parsed from a header.
    words: tuple[int, ...]
    widened: tuple[int, ...] = ()  # Literal F32 words for the two half-width records.

    @property
    def fmt(self):
        return "I" if self.dtype == "F32" else "H"

    @property
    def width(self):
        return 4 if self.dtype == "F32" else 2

    @property
    def span(self):
        return _PAYLOAD + self.position, self.width * len(self.words)


# Physical order deliberately differs from the importer's lexicographic order.
_RECORDS = (
    Record(_EMBED, (5, 2), "BF16", 0,
           (0x0000, 0x8000, 0x3E00, 0xBE80, 0x3D80, 0x3E80, 0xBE00, 0x3D00, 0x3F00, 0xBD80),
           (0x00000000, 0x80000000, 0x3E000000, 0xBE800000, 0x3D800000,
            0x3E800000, 0xBE000000, 0x3D000000, 0x3F000000, 0xBD800000)),
    Record(_LAYER + "input_layernorm.weight", (2,), "F32", 20, (0x3F800000, 0x3F000000)),
    Record(_GATE, (2, 2), "F32", 28, (0x3D000000, 0x3D800000, 0xBD000000, 0xBC800000)),
    Record(_UP, (2, 2), "F32", 44, (0x3D000000, 0x3D800000, 0x3D000000, 0xBC800000)),
    Record(_LAYER + "mlp.down_proj.weight", (2, 2), "F32", 60,
           (0x3D000000, 0x3D800000, 0x3D000000, 0x3C800000)),
    Record(_LAYER + "post_attention_layernorm.weight", (2,), "F32", 76, (0x3F400000, 0x3F800000)),
    Record(_ATTN + "k_proj.bias", (2,), "F32", 84, (0x3C000000, 0xBC800000)),
    Record(_ATTN + "k_proj.weight", (2, 2), "F32", 92, (0x3D000000, 0xBD800000, 0x3D000000, 0x3C800000)),
    Record(_ATTN + "o_proj.weight", (2, 2), "F32", 108, (0xBD000000, 0x3D800000, 0x3D000000, 0x3C800000)),
    Record(_ATTN + "q_proj.bias", (2,), "F32", 124, (0x3C000000, 0x3C800000)),
    Record(_ATTN + "q_proj.weight", (2, 2), "F32", 132, (0x3D000000, 0x3D800000, 0xBD000000, 0x3C800000)),
    Record(_ATTN + "v_proj.bias", (2,), "F32", 148, (0xBC000000, 0x3C800000)),
    Record(_ATTN + "v_proj.weight", (2, 2), "F32", 156, (0x3D000000, 0x3D800000, 0x3D000000, 0x00000000)),
    Record("lm_head.weight", (5, 2), "F32", 172,
           (0x3F000000, 0xBF000000, 0x3F800000, 0x00000000, 0x00000000,
            0x3F800000, 0xBF800000, 0x00000000, 0x00000000, 0xBF800000)),
    Record(_NORM, (2,), "F16", 212, (0x0000, 0x8000), (0x00000000, 0x80000000)),
)
_BY_NAME = {r.name: r for r in _RECORDS}
_ORDER = tuple(sorted(_BY_NAME))
_SHAPES = {r.name: r.shape for r in _RECORDS}


def _literal(*, swap=False):
    payload = bytearray()
    header = {"__metadata__": {"format": "pt"}}
    positions = {_GATE: _BY_NAME[_UP].position, _UP: _BY_NAME[_GATE].position} if swap else {}
    for r in _RECORDS:
        assert len(payload) == r.position and len(r.words) == math.prod(r.shape)
        payload.extend(struct.pack(f"<{len(r.words)}{r.fmt}", *r.words))
        start = positions.get(r.name, r.position)
        header[r.name] = {"dtype": r.dtype, "shape": r.shape,
                          "data_offsets": (start, start + r.span[1])}
    encoded = json.dumps(header, separators=(",", ":")).encode().ljust(_HEADER_LEN, b" ")
    assert len(encoded) == _HEADER_LEN
    return b"".join((struct.pack("<Q", _HEADER_LEN), encoded, payload))


def _bound_readers(selected, source_view, reread_view):
    assert type(selected) is bytes
    for view in (source_view, reread_view):
        assert view.obj is selected, "view does not export the selected shard"
        assert view.readonly and view.format == "B" and view.shape == (len(selected),)
    assert source_view is not reread_view

    def source_bytes(offset, size):
        assert 0 <= offset <= offset + size <= len(source_view)
        return source_view[offset:offset + size].tobytes()

    def reread(*, endpoints=False):
        for r in reversed(_RECORDS) if endpoints else _RECORDS:
            offset, _ = r.span
            if endpoints:
                for lane in (len(r.words) - 1, 0):
                    word, = struct.unpack_from("<" + r.fmt, reread_view, offset + lane * r.width)
                    assert word == r.words[lane]
            else:
                assert struct.unpack_from(f"<{len(r.words)}{r.fmt}", reread_view, offset) == r.words

    return source_bytes, reread


def _is_shard(name):
    return name.endswith((".safetensors", ".safetensors.index.json"))


class ReadLog:
    def __init__(self, shard_digest):
        self.shard_digest = shard_digest
        self.ranges = []
        self.locks = 0

    def mark(self):
        return len(self.ranges), self.locks

    def expect(self, mark, keys, *, lock_passes):
        ranges, locks = mark
        headers = [(0, 8), (8, _HEADER_LEN)]
        assert self.ranges[ranges:] == headers + [_BY_NAME[key].span for key in keys]
        assert self.locks - locks == lock_passes


@contextmanager
def _fixture(tmp_path, monkeypatch, *, swap=False, short_when=None):
    experiment = experiment_fixture(tmp_path, weights="public").with_params(
        pipeline__components__linear__range_policy="checked",
        pipeline__components__linear__max_abs_activation=256,
    )
    root = Path(experiment.pipeline.model.source).resolve()
    path = root / _SHARD
    selected = _literal(swap=swap)
    path.write_bytes(selected)
    identity = shard_io._identity(path)
    assert identity[9] == len(selected)
    source_bytes, reread = _bound_readers(selected, memoryview(selected), memoryview(selected))
    original_range, original_open = shard_io._read_range, shard_io._LockPath.open
    log = ReadLog(sha256(selected).hexdigest())

    def check_inventory():
        assert {p.name for p in root.iterdir() if _is_shard(p.name)} == {_SHARD}
        assert shard_io._identity(path) == identity

    def read_range(candidate, admitted, offset, size, *, limit=shard_io._MAX_SOURCE_BYTES):
        if candidate.parent != root or not _is_shard(candidate.name):
            return original_range(candidate, admitted, offset, size, limit=limit)
        check_inventory()
        assert candidate == path and admitted == identity
        assert type(offset) is type(size) is int
        assert 0 <= size <= min(limit, shard_io._DECODER_METADATA_BYTES)
        assert 0 <= offset <= offset + size <= admitted[9] == len(selected)
        log.ranges.append((offset, size))
        short = short_when is not None and (offset, size) == _BY_NAME[_NORM].span and short_when()
        raw = source_bytes(offset, size - 1 if short else size)
        if len(raw) != size:
            raise shard_io.DelegatedSourceError("literal final source range truncated")
        return raw

    @contextmanager
    def cursor():
        position = 0
        eof = False

        class Reader:
            def read(self, count):
                nonlocal position, eof
                assert type(count) is int and count > 0
                size = min(count, 64 << 10, len(selected) - position)
                raw = source_bytes(position, size)
                position += len(raw)
                eof = not raw
                return raw

        yield Reader()
        assert position == len(selected) and eof
        check_inventory()
        log.locks += 1

    def lock_open(lock_path, mode):
        if lock_path.root != root:
            return original_open(lock_path, mode)
        check_inventory()
        assert {name for name in lock_path.identities if _is_shard(name)} == {_SHARD}
        if lock_path.filename is None or not _is_shard(lock_path.filename):
            return original_open(lock_path, mode)
        assert lock_path.path == path and mode == "rb"
        assert lock_path.identities[_SHARD] == identity
        return cursor()

    check_inventory()
    with monkeypatch.context() as scoped:
        # Toy source-lineage fixture, not host-telemetry or real-resource evidence.
        # Keep observe(), byte pricing, the shared ledger and reservation lifecycle
        # real; replace only machine-dependent memory samples, never disk policy.
        memory = benchmark_memory.HostMemory(8 << 30, 6 << 30, 0, 0)
        scoped.setattr(benchmark_memory, "host_memory", lambda: memory)
        scoped.setattr(psutil.Process, "memory_info", lambda self: SimpleNamespace(rss=64 << 20))
        scoped.setattr(resources, "_linux_memory_headroom", lambda: 6 << 30)
        scoped.setattr(shard_io, "_read_range", read_range)
        scoped.setattr(shard_io._LockPath, "open", lock_open)
        yield experiment, reread, log


def _assert_inventory(source):
    assert {k: tuple(v) for k, v in source.contract["source_shapes"].items()} == _SHAPES
    store, specs = shard_io._store(source.path, source._identities, source.contract)
    try:
        assert specs == source._tensor_specs
        expected = {r.name: (_SHARD, r.shape, r.dtype, *r.span) for r in _RECORDS}
        physical = dict(specs)
        resolved = {key: physical[store.resolve(key)] for key in _ORDER}
        assert resolved == expected, "literal physical mapping differs"
    finally:
        store.clear_cache()


@contextmanager
def _publish(experiment, log):
    mark = log.mark()
    source = admit_source(experiment)
    log.expect(mark, (), lock_passes=2)
    _assert_inventory(source)
    assert {row["path"]: row["sha256"] for row in source.source_lock["files"]}[_SHARD] == log.shard_digest
    mark = log.mark()
    publisher, manifest = source.publisher(experiment, (3, 4), 2, lambda: False)
    try:
        log.expect(mark, _ORDER, lock_passes=2)
        request = json.loads(manifest)
        assert (request["weights"], request["rows"], request["outputs"]) == ("public", 2, 2)
        records = request["tensors"]
        assert {key: tuple(item["shape"]) for key, item in records.items()} == _SHAPES
        assert records[_NORM]["public_q18"] == [0, 0]
        assert records[_EMBED]["public_q18"] is None
        q18 = (0, 0, 32768, -65536, 16384, 65536, -32768, 8192, 131072, -16384)
        assert bytes(records[_EMBED]["digest"]) == sha256(struct.pack("<10q", *q18)).digest()
        yield source, manifest, bytes(request["session"]).hex()
    finally:
        publisher.close()


def _expect_words(key, frame):
    r = _BY_NAME[key]
    assert type(frame) is bytes and len(frame) == 4 * len(r.words)
    assert struct.unpack(f"<{len(r.words)}I", frame) == (r.widened or r.words), f"F32 words differ for {key}"


def _observe_worker(monkeypatch, source, manifest, roles, owners, *, fault=None):
    native = worker._native.ReplicatedWorker
    factory = native._from_source_request  # Required: no fallback to the old binding.
    capacity = resources._capacity
    with capacity.lock:
        observed_tokens = set(capacity._owners)

    class Observer:
        def __init__(self, owner, role, memory_token, memory_amount):
            self.owner, self.role = owner, role
            self.memory_token, self.memory_amount = memory_token, memory_amount
            self.append = owner.append_public_source
            self.keys = []
            self.setup_calls = []
            self.sealed = self.activated = self.closed = False
            self.checked_refusal = self.native_refusal = False

        def assert_reserved(self):
            with capacity.lock:
                assert capacity._owners.get(self.memory_token) == self.memory_amount

        def assert_released(self):
            with capacity.lock:
                assert self.memory_token not in capacity._owners

        def append_public_source(self, key, offset, frame):
            self.assert_reserved()
            assert not self.closed and key == _ORDER[len(self.keys)] and offset == 0
            if (fault, key) in {("inline_sign", _NORM), ("out_of_line_sign", _EMBED)}:
                _expect_words(key, frame)  # Check the genuine frame before negative-only injection.
                frame = frame[:7] + bytes([frame[7] ^ 0x80]) + frame[8:]
                assert struct.unpack_from("<f", frame, 4)[0] == 0.0
                try:
                    _expect_words(key, frame)
                except AssertionError:
                    self.checked_refusal = True
                    raise
            _expect_words(key, frame)
            if fault == "q18" and key == _NORM:
                changed = struct.pack("<I", 0x3F000000) + frame[4:]  # 0 -> 1/2
                try:
                    self.append(key, offset, changed)
                except ValueError:
                    self.assert_retired()  # Native refusal itself consumed the handle.
                    self.native_refusal = True
                    raise
                pytest.fail("native accepted a changed inline Q18 coefficient")
            self.append(key, offset, frame)  # Forward the identical bytes object exactly once.
            self.keys.append(key)

        def seal_public_source(self):
            self.assert_reserved()
            assert not self.closed and not self.sealed and tuple(self.keys) == _ORDER
            self.owner.seal_public_source()
            self.sealed = True

        def call(self, action, payload, sender):
            self.assert_reserved()
            assert self.sealed and not self.closed and not self.activated
            assert action in {"offer", "public", "seed", "receive_seed"}
            result = self.owner.call(action, payload, sender)
            self.setup_calls.append(action)
            return result

        def activate(self):
            self.assert_reserved()
            assert self.sealed and not self.closed and not self.activated
            self.owner.activate()
            self.activated = True

        def close(self):
            self.assert_reserved()
            self.owner.close()
            self.closed = True

        def assert_retired(self):
            with pytest.raises(ValueError, match="replicated worker retired"):
                self.owner.call("offer", b"", self.role)

    class Constructors:
        def __new__(cls, *_):
            pytest.fail("ordinary constructor cannot witness the source-request join")

        @staticmethod
        def _from_source_request(raw, role, *, server_policy):
            assert type(raw) is bytes and raw == manifest
            assert role in roles and role not in owners
            assert server_policy == (source.experiment_bytes, source.config_bytes)
            with capacity.lock:
                added = set(capacity._owners) - observed_tokens
                assert len(added) == 1, "native entry must hold one real memory reservation"
                token, = added
                amount = capacity._owners[token]
                assert type(amount) is int and amount > 0
                observed_tokens.add(token)
            owner = Observer(factory(raw, role, server_policy=server_policy), role, token, amount)
            owners[role] = owner
            return owner

    monkeypatch.setattr(worker._native, "ReplicatedWorker", Constructors)


@contextmanager
def _observe_errors(monkeypatch):
    errors = []
    original = worker.clear_failure_frames

    def clear(error):
        errors.append((type(error), str(error)))
        original(error)

    with monkeypatch.context() as scoped:
        scoped.setattr(worker, "clear_failure_frames", clear)
        yield errors


@contextmanager
def _cancel_import(monkeypatch, fault, cancellations):
    original = ReplicatedSource.populate_public_worker

    def populate(self, owner, cancelled):
        def predicate():
            if owner.keys:
                cancellations.append(tuple(owner.keys))
                return True
            return cancelled()

        return original(self, owner, predicate)

    with monkeypatch.context() as scoped:
        if fault == "cancel_import":
            scoped.setattr(ReplicatedSource, "populate_public_worker", populate)
        yield


async def _open_apps(stack, monkeypatch, experiment, log, role_indices, request_id):
    origins = tuple(f"http://127.0.0.1:{31001 + i}" for i in range(3))
    keys = tuple(f"literal-client-{i}-secret" for i in range(3))
    apps = {}
    async_client = httpx.AsyncClient

    def peer_client(*args, **kwargs):
        origin = kwargs.get("base_url")
        if origin in origins:
            recipient = origins.index(origin)

            async def route(request):
                assert f"{request.url.scheme}://{request.url.host}:{request.url.port}" == origin
                assert request.method == "POST" and request.url.path.startswith(
                    f"/v1/replicated/{request_id}/peer/"
                )
                assert recipient in apps, "peer RPC in a single-app admission"
                return await httpx.ASGITransport(apps[recipient]).handle_async_request(request)

            kwargs["transport"] = httpx.MockTransport(route)
        return async_client(*args, **kwargs)

    for i in role_indices:
        peers = {role: (origins[j], f"literal-directed-{i}-{j}", f"literal-directed-{j}-{i}")
                 for j, role in enumerate(ROLES) if j != i}
        mark = log.mark()
        apps[i] = worker.create_replicated_worker_app(experiment, ROLES[i], keys[i], peers)
        log.expect(mark, (), lock_passes=2)
    # Only substitute local HTTP peer connections, not the peer handler or setup.
    scoped = stack.enter_context(monkeypatch.context())
    scoped.setattr(worker.httpx, "AsyncClient", peer_client)
    clients = {}
    for i, app in apps.items():
        await stack.enter_async_context(app.router.lifespan_context(app))
        clients[i] = await stack.enter_async_context(async_client(
            transport=httpx.ASGITransport(app), base_url=origins[i],
            headers={"authorization": f"Bearer {keys[i]}"},
        ))
    return clients


async def _admit(client, role, source, manifest, request_id):
    advertised = (await client.get("/v1/replicated/manifest")).raise_for_status()
    value = unpack(advertised.content)
    assert value["source_lock"] == source.source_lock_digest and value["role"] == ROLES[role]
    return await client.post("/v1/replicated/admit", content=pack({
        "request": request_id, "epoch": value["epoch"], "manifest": manifest,
    }))


async def _post(client, request_id, action):
    response = await client.post(f"/v1/replicated/{request_id}/{action}")
    return unpack(response.raise_for_status().content)


async def _assert_retired(client, request_id, owner, *, peer_frames=0):
    assert owner.closed
    owner.assert_retired()
    owner.assert_released()
    response = await client.post(f"/v1/replicated/{request_id}/cancel")
    assert response.status_code == 409
    metrics = (await client.get("/v1/replicated/metrics")).raise_for_status().json()
    assert metrics["active_sessions"] == 0
    assert metrics["peer_counts"] == {"frames": peer_frames, "controls": 0}


@pytest.mark.parametrize("side", ["source", "reread"])
def test_bit_identical_exporter_is_not_selected(side):
    selected = _literal()
    other = bytes(bytearray(selected))
    assert other == selected and other is not selected
    with pytest.raises(AssertionError, match="does not export the selected shard"):
        _bound_readers(selected, memoryview(other if side == "source" else selected),
                       memoryview(other if side == "reread" else selected))


def test_equal_shape_swapped_records_are_not_literal_mapping(tmp_path, monkeypatch):
    assert _BY_NAME[_GATE].shape == _BY_NAME[_UP].shape
    assert _BY_NAME[_GATE].words != _BY_NAME[_UP].words
    with _fixture(tmp_path, monkeypatch, swap=True) as (experiment, reread, log):
        source = admit_source(experiment)  # Real shape and header validation still succeeds.
        log.expect((0, 0), (), lock_passes=2)
        reread()  # Physical bytes are unchanged; only the labeled offsets differ.
        with pytest.raises(AssertionError, match="literal physical mapping differs"):
            _assert_inventory(source)


@pytest.mark.asyncio
async def test_literal_admit_finishes_without_activation_and_cancels(tmp_path, monkeypatch):
    with (
        _fixture(tmp_path, monkeypatch) as (experiment, reread, log),
        _publish(experiment, log) as (source, manifest, request_id),
        monkeypatch.context() as scoped,
    ):
        owners = {}
        reread()
        _observe_worker(scoped, source, manifest, (0,), owners)
        async with AsyncExitStack() as stack:
            client = (await _open_apps(stack, monkeypatch, experiment, log, (0,), request_id))[0]
            mark = log.mark()
            response = await _admit(client, 0, source, manifest, request_id)
            assert unpack(response.raise_for_status().content) == {"accepted": True}
            owner = owners[0]
            log.expect(mark, _ORDER, lock_passes=2)
            assert owner.sealed and not owner.activated and not owner.closed and not owner.setup_calls
            reread()
            reread(endpoints=True)
            assert await _post(client, request_id, "cancel") == {"retired": True}
            await _assert_retired(client, request_id, owner)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["inline_sign", "out_of_line_sign", "q18", "short_final_range", "cancel_import"])
async def test_literal_worker_import_refusals(tmp_path, monkeypatch, fault):
    owners, cancellations = {}, []

    def short_when():
        return fault == "short_final_range" and 0 in owners and tuple(owners[0].keys) == _ORDER[:-1]

    with (
        _fixture(tmp_path, monkeypatch, short_when=short_when) as (experiment, reread, log),
        _publish(experiment, log) as (source, manifest, request_id),
        _observe_errors(monkeypatch) as errors,
        _cancel_import(monkeypatch, fault, cancellations),
        monkeypatch.context() as scoped,
    ):
        reread()
        _observe_worker(scoped, source, manifest, (0,), owners, fault=fault)
        async with AsyncExitStack() as stack:
            client = (await _open_apps(stack, monkeypatch, experiment, log, (0,), request_id))[0]
            mark = log.mark()
            assert (await _admit(client, 0, source, manifest, request_id)).status_code == 400
            owner = owners[0]
            assert not owner.sealed and not owner.activated and not owner.setup_calls
            if fault == "cancel_import":
                assert tuple(owner.keys) == _ORDER[:1] and cancellations == [_ORDER[:1]]
                assert errors == [(ValueError, "replicated source import cancelled")]
                log.expect(mark, _ORDER[:1], lock_passes=1)
            else:
                key = _EMBED if fault == "out_of_line_sign" else _NORM
                at = _ORDER.index(key)
                assert tuple(owner.keys) == _ORDER[:at]
                log.expect(mark, _ORDER[:at + 1], lock_passes=1)
                assert not cancellations
                if fault == "short_final_range":
                    assert errors == [(shard_io.DelegatedSourceError, "literal final source range truncated")]
                elif fault != "q18":
                    assert len(errors) == 1
                    kind, message = errors[0]
                    assert kind is AssertionError
                    assert message.partition("\n")[0] == f"F32 words differ for {key}"
            assert owner.native_refusal == (fault == "q18")
            assert owner.checked_refusal == (fault in {"inline_sign", "out_of_line_sign"})
            await _assert_retired(client, request_id, owner)
            reread(endpoints=True)


@pytest.mark.asyncio
async def test_literal_three_app_setup_activates_only_role_a(tmp_path, monkeypatch):
    with (
        _fixture(tmp_path, monkeypatch) as (experiment, reread, log),
        _publish(experiment, log) as (source, manifest, request_id),
        monkeypatch.context() as scoped,
    ):
        owners = {}
        reread()
        roles = tuple(range(3))
        _observe_worker(scoped, source, manifest, roles, owners)
        async with AsyncExitStack() as stack:
            clients = await _open_apps(stack, monkeypatch, experiment, log, roles, request_id)
            for i, client in clients.items():
                mark = log.mark()
                response = await _admit(client, i, source, manifest, request_id)
                assert unpack(response.raise_for_status().content) == {"accepted": True}
                log.expect(mark, _ORDER, lock_passes=2)
            assert all(owner.sealed and not owner.activated for owner in owners.values())
            reread()
            reread(endpoints=True)
            # Same stage order as replicated_transport.execute_request, stopping
            # before token import and activating only A after all peer deliveries.
            for action in ("setup_public", "setup_dispatch", "setup_seed", "setup_dispatch"):
                for client in clients.values():
                    assert await _post(client, request_id, action) == {"accepted": True}
            assert all(owner.setup_calls == ["offer", "public", "public", "seed", "receive_seed"]
                       for owner in owners.values())
            assert await _post(clients[0], request_id, "activate") == {"accepted": True}
            assert [i for i, owner in owners.items() if owner.activated] == [0]
            reread(endpoints=True)
            for i, client in clients.items():
                assert await _post(client, request_id, "cancel") == {"retired": True}
                await _assert_retired(client, request_id, owners[i], peer_frames=3)
