"""Mocked source-stage HTTP seam; no new wheel or native value loading."""

import asyncio
from contextlib import contextmanager
import copy
import json
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import weakref
from types import SimpleNamespace

import httpx
import pytest
from starlette.requests import Request

from pllm.runtime import delegated_protocol as protocol
from pllm.runtime import delegated_transport as transport
from pllm.runtime import delegated_worker as worker
from pllm.runtime import delegated_source as source_io
from pllm.runtime import benchmark_memory
from pllm.runtime.client import ProtocolError


def runtime():
    return {"decoder": {"execution_payload_estimate_bytes": 1024}, "batch_execution": {
        "schema_version": "pllm.gkm.source-stage-batch-contract.v1",
        "execution": protocol.BATCH_EXECUTION, "batch_plan": "a" * 64,
        "resources": {"max_aggregate_bytes": 4096, "retained_frame_bytes": 32,
                      "transport_scratch_bytes": 16, "extra_native_owner_bytes": 48,
                      "extra_wire_bytes": 24},
        "compiler_encoded_input_peak_bytes": 32,
        "client_payload_bytes": 1128, "worker_payload_bytes": 1096,
        "descriptors": [{"batch_index": 0, "phase_index": 0, "operation_index": 0, "first_claim": 0,
                         "rows": 2, "kind": "linear", "stage": 0,
                         "columns": 2, "output_width": 3,
                         "request_bytes": [7, 9], "reply_bytes": [11, 13]}]}}


def test_copy_phases_do_not_reuse_native_retention_or_scratch():
    value = runtime()
    prices = protocol.batch_copy_prices(value)
    assert prices == {
        "batch_python_client_peak_bytes": 7 + 9 + 11 + 2 * 13 + (64 << 10) + (512 << 10),
        "batch_python_worker_peak_bytes": {"worker_a": 22 + (1 << 20), "worker_b": 27 + (1 << 20)},
        "batch_runtime_backing_delta_bytes": 16,
        "batch_binding_token_peak_bytes": 0}
    value["batch_execution"]["resources"]["retained_frame_bytes"] *= 1000
    value["batch_execution"]["resources"]["transport_scratch_bytes"] *= 1000
    assert protocol.batch_copy_prices(value) == prices


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_request_and_source_use_same_separate_copy_prices(capacity):
    value = runtime()
    if capacity == "decoder":
        value["decoder"]["decoder_resources"] = {"client_payload_bytes": 1024, "worker_payload_bytes": 1024}
    source = SimpleNamespace(capacity=capacity, execution_payload_estimate_bytes=1024,
                             contract=value["decoder"], runtime_contract_bytes=json.dumps(value).encode())
    prices = protocol.batch_memory_bytes(value, capacity)
    assert transport.request_memory_bytes(source) == prices["client_payload_bytes"] + (64 << 20)
    for role in protocol.ROLES:
        assert transport.request_memory_bytes(source, worker=True, role=role) == prices["worker_payload_bytes"][role] + (64 << 20)
    probe = object.__new__(source_io.AdmittedSource)
    object.__setattr__(probe, "runtime_contract_bytes", source.runtime_contract_bytes)
    object.__setattr__(probe, "contract_bytes", json.dumps(value["decoder"]).encode())
    # Capsule needs no capacity-profile metadata in its source record.
    if capacity == "capsule":
        assert probe.batch_copy_prices == prices


@pytest.mark.parametrize("change", ["missing", "bool-length", "oversize", "claims", "order", "native-quote", "capacity"])
def test_incomplete_or_invalid_compiler_handshake_fails_closed(change):
    value = runtime()
    batch = value["batch_execution"]
    if change == "missing":
        del batch["descriptors"]
    elif change == "bool-length":
        batch["descriptors"][0]["request_bytes"][0] = True
    elif change == "oversize":
        batch["descriptors"][0]["reply_bytes"][1] = 4097
    elif change == "claims":
        batch["descriptors"][0]["first_claim"] = 1
    elif change == "order":
        batch["descriptors"].append(copy.deepcopy(batch["descriptors"][0]))
    elif change == "native-quote":
        batch["worker_payload_bytes"] += 1
    else:
        batch["descriptors"][0]["output_width"] = 64 << 20
    with pytest.raises(ValueError):
        protocol.batch_memory_bytes(value, "capsule")


def test_scalar_metadata_and_request_price_unchanged():
    assert protocol.batch_contract({}) is None
    assert protocol.batch_copy_prices({}) == protocol.batch_memory_bytes({}, "capsule") == {}
    source = SimpleNamespace(capacity="capsule", execution_payload_estimate_bytes=123)
    assert transport.request_memory_bytes(source) == 123 + (64 << 20)


@pytest.mark.parametrize("quoted", [None, True, -1, 0, 31, 33])
def test_compiler_input_quote_rejects_missing_invalid_or_wrong_owner(quoted):
    value = runtime()
    if quoted is None:
        del value["batch_execution"]["compiler_encoded_input_peak_bytes"]
    else:
        value["batch_execution"]["compiler_encoded_input_peak_bytes"] = quoted
    with pytest.raises(ValueError, match="compiler input quote"):
        protocol.batch_memory_bytes(value, "capsule")


@pytest.mark.parametrize("capacity", ["capsule", "decoder"])
def test_compiler_input_charge_applies_only_to_client_and_can_reject_capacity(capacity):
    value = runtime()
    value["batch_execution"]["resources"]["max_aggregate_bytes"] = 1 << 20
    descriptor = value["batch_execution"]["descriptors"][0]
    descriptor["request_bytes"] = descriptor["reply_bytes"] = [1 << 20, 1 << 20]
    if capacity == "decoder":
        value["decoder"]["decoder_resources"] = {"worker_payload_bytes": 1024}
    cap = 4 << 30 if capacity == "decoder" else 64 << 20
    prices = protocol.batch_memory_bytes(value, capacity)
    batch = value["batch_execution"]
    assert prices["compiler_encoded_input_peak_bytes"] == 32
    assert batch["client_payload_bytes"] - batch["worker_payload_bytes"] == 32
    # Scalar + core + Python owners fit; the separately owned compiler Vec does not.
    scalar = cap - (prices["client_payload_bytes"] - 1024) + 16
    value["decoder"]["execution_payload_estimate_bytes"] = scalar
    if capacity == "decoder":
        value["decoder"]["decoder_resources"] = {"worker_payload_bytes": 1024}
    batch["client_payload_bytes"] = scalar + 48 + 24 + 32
    batch["worker_payload_bytes"] = (scalar if capacity == "capsule" else 1024) + 48 + 24
    assert all(batch["worker_payload_bytes"] + size <= cap
               for size in prices["batch_python_worker_peak_bytes"].values())
    with pytest.raises(ValueError, match="capacity"):
        protocol.batch_memory_bytes(value, capacity)


@pytest.mark.parametrize("role", protocol.ROLES)
def test_batch_https_rejects_before_io_backend_or_tls_setup(monkeypatch, role):
    def forbidden(*args, **kwargs):
        pytest.fail("HTTPS batch reached I/O, backend or TLS setup")

    monkeypatch.setattr(transport, "_IOOwner", forbidden)
    monkeypatch.setattr(transport, "_batch_http_transport", forbidden)
    monkeypatch.setattr(transport.httpx, "AsyncClient", forbidden)
    monkeypatch.setattr(ssl, "create_default_context", forbidden)
    connections = {"worker_a": ("http://127.0.0.1:1", "a" * 32),
                   "worker_b": ("http://127.0.0.1:2", "b" * 32)}
    origin, key = connections[role]
    connections[role] = (origin.replace("http:", "https:"), key)
    with pytest.raises(ValueError, match="batch TLS is unsupported"):
        transport._Peers(connections, 1, lambda: False, batch=protocol.batch_contract(runtime()))


def test_batch_remote_plaintext_remains_forbidden(monkeypatch):
    monkeypatch.setattr(transport, "_IOOwner", lambda: pytest.fail("remote plaintext reached I/O"))
    with pytest.raises(ValueError, match="remote worker channels require HTTPS"):
        transport._Peers({"worker_a": ("http://192.0.2.1", "a" * 32),
                          "worker_b": ("http://127.0.0.1:2", "b" * 32)},
                         1, lambda: False, batch=protocol.batch_contract(runtime()))


def test_scalar_https_keeps_verified_tls_setup_and_skips_batch_gate(monkeypatch):
    contexts = []
    create = ssl.create_default_context

    def observed(*args, **kwargs):
        context = create(*args, **kwargs)
        contexts.append(context)
        return context

    def forbidden(*args, **kwargs):
        pytest.fail("scalar HTTPS entered batch-only gate/backend")

    monkeypatch.setattr(ssl, "create_default_context", observed)
    monkeypatch.setattr(transport, "_check_batch_http_connections", forbidden)
    monkeypatch.setattr(transport, "_batch_http_transport", forbidden)
    peers = transport._Peers({"worker_a": ("https://127.0.0.1:1", "a" * 32),
                              "worker_b": ("https://127.0.0.1:2", "b" * 32)}, 1, lambda: False)
    peers.close()
    assert contexts and all(context.verify_mode == ssl.CERT_REQUIRED and context.check_hostname
                            for context in contexts)
    assert not peers.io.thread.is_alive() and peers.io.loop.is_closed()


def test_batch_stream_start_tls_rejects_without_tls_allocator(monkeypatch):
    from httpcore._backends.anyio import AnyIOStream

    def forbidden(*args, **kwargs):
        pytest.fail("batch stream entered TLS allocator/handshake")

    monkeypatch.setattr(ssl, "create_default_context", forbidden)
    monkeypatch.setattr(ssl, "SSLContext", forbidden)
    backend_transport = transport._batch_http_transport()

    async def run():
        async def connect(*args, **kwargs):
            return SimpleNamespace(), SimpleNamespace()

        monkeypatch.setattr(asyncio.get_running_loop(), "create_connection", connect)
        monkeypatch.setattr(AnyIOStream, "start_tls", forbidden)
        stream = await backend_transport._pool._network_backend.connect_tcp("127.0.0.1", 1)
        with pytest.raises(ValueError, match="batch TLS is unsupported"):
            await stream.start_tls(object(), server_hostname="127.0.0.1")
        await backend_transport.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("option", ["ssl_certfile", "ssl_keyfile", "ssl_context_factory"])
def test_batch_worker_ssl_config_rejects_before_loading_or_proof_marker(monkeypatch, option):
    import uvicorn
    from uvicorn.protocols.http.h11_impl import H11Protocol

    def forbidden(*args, **kwargs):
        pytest.fail("batch worker allocated TLS or initialized its HTTP proof")

    monkeypatch.setattr(uvicorn.config, "create_ssl_context", forbidden)
    monkeypatch.setattr(H11Protocol, "__init__", forbidden)
    value = forbidden if option == "ssl_context_factory" else "unused.pem"
    config = worker._batch_worker_config(forbidden, **{option: value})
    with pytest.raises(ValueError, match="batch TLS is unsupported"):
        config.load()
    assert not config.loaded
    protocol_type = worker._batch_h11_protocol()
    candidate = protocol_type.__new__(protocol_type)
    with pytest.raises(ValueError, match="batch TLS is unsupported"):
        protocol_type.__init__(candidate, config, uvicorn.server.ServerState(), {})
    assert getattr(candidate, "scope", None) is None


def test_source_binding_requires_explicit_selector_and_complete_quote(monkeypatch):
    value = runtime()
    pipeline = SimpleNamespace(components={"linear": SimpleNamespace(params={
        "sharing": "full", "execution": protocol.BATCH_EXECUTION})}, digest=lambda: "c" * 64)
    experiment = SimpleNamespace(configuration_digest=lambda: "b" * 64, pipeline=pipeline)
    value.update(schema_version=source_io._RUNTIME_SCHEMA, configuration="b" * 64,
                 composition="c" * 64, source_lock="d" * 64, tokenizer="e" * 64)
    value["decoder"].update(input_rows=2, decode_steps=0)
    decoder = json.dumps(value["decoder"]).encode()
    encoded = json.dumps(value).encode()
    monkeypatch.setattr(source_io._native, "gkm_runtime_contract", lambda *args: encoded)
    assert source_io._bind_runtime(experiment, b"experiment", b"config", decoder, "d" * 64, "e" * 64) is encoded
    del value["batch_execution"]["descriptors"]
    encoded = json.dumps(value).encode()
    with pytest.raises(ValueError, match="trusted per-descriptor"):
        source_io._bind_runtime(experiment, b"experiment", b"config", decoder, "d" * 64, "e" * 64)


def test_watchdog_estimate_prices_exact_same_batch_deltas(monkeypatch):
    from pllm import configuration
    from pllm.runtime import delegated_decoder

    value = runtime()
    value["decoder"].update(source_float32_bytes=32, plan={"config_digest": "a" * 64},
                            capsule_payload_allowance_bytes=64, transport_resources={})
    raw_decoder = json.dumps(value["decoder"]).encode()
    monkeypatch.setattr(delegated_decoder, "decoder_contract", lambda *a, **k: raw_decoder)
    monkeypatch.setattr(source_io._native, "gkm_runtime_contract", lambda *a: json.dumps(value).encode())
    monkeypatch.setattr(configuration, "Experiment", lambda *a, **k: SimpleNamespace(canonical_bytes=lambda: b"experiment"))
    pipeline = SimpleNamespace(components={"linear": SimpleNamespace(params={"sharing": "full"})}, digest=lambda: "b" * 64)
    scalar = benchmark_memory._delegated_memory({}, pipeline, 2, 1, 0, None, 0, 0)
    pipeline.components["linear"].params["execution"] = protocol.BATCH_EXECUTION
    batch = benchmark_memory._delegated_memory({}, pipeline, 2, 1, 0, None, 0, 0)
    prices = protocol.batch_memory_bytes(value, "capsule")
    assert batch["client_peak_bytes"] - scalar["client_peak_bytes"] == prices["client_payload_bytes"] - 1024
    for role in protocol.ROLES:
        assert batch["provider_peak_bytes"][role] - scalar["provider_peak_bytes"][role] == prices["worker_payload_bytes"][role] - 1024
    assert batch["native_total_peak_bytes"] == batch["client_peak_bytes"] + sum(batch["provider_peak_bytes"].values())
    for key in ("batch_python_client_peak_bytes", "batch_python_worker_peak_bytes", "batch_runtime_backing_delta_bytes"):
        assert batch["verification_allocation"][key] == prices[key]


def peers_stub():
    peers = object.__new__(transport._Peers)
    peers.batch = protocol.batch_contract(runtime())
    peers.batch_index = peers.batch_exchanges = peers.scalar_claims = 0
    peers.check = lambda: None
    peers.roles, peers.namespace, peers.request_id = protocol.ROLES, "delegation", "a" * 64
    peers.phase, peers.phase_bytes = "prefill", {"prefill": 0, "decode": 0}
    peers.counts = {f"role_link.{role}.{phase}_{direction}_bytes": 0
                    for role in protocol.ROLES for phase in ("setup", "online", "teardown")
                    for direction in ("upload", "download")}
    peers.unknown = set()
    peers.calls = peers.server_ns = 0
    peers.calls_known = peers.server_known = True
    return peers


def test_exchange_uses_role_lengths_and_never_encodes_lists():
    peers = peers_stub()
    observed = []

    def post(role, action, payload, **kwargs):
        observed.append((role, action, payload, kwargs))
        peers.calls += 1
        return b"d" * kwargs["expected"]

    peers.post = post
    descriptor = peers.batch_descriptor("linear", 0)
    assert peers.exchange_batch((b"q" * 7, b"q" * 9), descriptor) == (b"d" * 11, b"d" * 13)
    assert [(row[0], len(row[2]), row[3]["limit"]) for row in observed] == [(0, 7, 11), (1, 9, 13)]
    assert all(row[1] == "step" and type(row[2]) is bytes for row in observed)
    audit = peers.audit()
    assert audit["inference_stage_calls"] == 2
    assert audit["batch_exchanges"] == 1 and audit["native_scalar_claims"] == 2
    assert audit["online_steps"] == 2  # Row claims, not one aggregate exchange.


def test_wrong_batch_request_rejects_before_http():
    peers = peers_stub()
    peers.post = lambda *a, **k: pytest.fail("wrong role length reached HTTP")
    with pytest.raises(ProtocolError):
        peers.exchange_batch((b"q" * 7, b"q" * 8), peers.batch_descriptor("linear", 0))
    with pytest.raises(ProtocolError):
        peers.batch_descriptor("lookup")
    assert peers.batch_index == peers.batch_exchanges == peers.scalar_claims == 0


@pytest.mark.parametrize("declared,blocks,accepted", [("13", [b"d"] * 13, True), ("12", [], False),
    (None, [], False), ("13", [b"d" * 12], False), ("13", [b"d" * 14], False)])
def test_exact_reply_declaration_before_collector_allocation(monkeypatch, declared, blocks, accepted):
    peers = peers_stub()
    allocations = []

    def allocate(size):
        allocations.append(size)
        return bytearray(size)

    monkeypatch.setattr(transport, "bytearray", allocate, raising=False)

    class Response:
        headers = httpx.Headers({"x-pllm-server-ns": "12", **({"content-length": declared} if declared is not None else {})})
        request = SimpleNamespace(content=b"q" * 9)
        status_code = 200

        async def aiter_raw(self):
            for block in blocks:
                yield block

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, *args):
            return False

    peers.clients = [None, SimpleNamespace(stream=lambda *a, **k: Stream())]

    async def run():
        call = peers._post(1, "step", b"q" * 9, phase="online", limit=13, expected=13, timeout=1)
        if accepted:
            assert await call == b"d" * 13
        else:
            with pytest.raises(ProtocolError):
                await call

    asyncio.run(run())
    assert allocations == ([13] if declared == "13" else [])
    assert peers.counts["role_link.worker_b.online_download_bytes"] == sum(map(len, blocks))


def test_native_exchange_drops_requests_before_acceptance():
    peers = peers_stub()

    class Frames(bytes):
        live = 0

        def __new__(cls, size):
            cls.live += 1
            return super().__new__(cls, b"q" * size)

        def __del__(self):
            type(self).live -= 1

    class Native:
        def linear_batch(self, stage, values):
            assert stage == 0 and len(values) == 16
            return Frames(7), Frames(9)

        def accept_batch(self, zero, one):
            assert Frames.live == 0
            assert (len(zero), len(one)) == (11, 13)
            return bytes(24)

    # Native frame subclasses stand in for lifetime probes only; production
    # exchange requires exact bytes and checks them in the separate tests above.
    peers.exchange_batch = lambda frames, descriptor: (bytes(11), bytes(13))
    native = transport._NativeExchange(Native(), peers, 0)
    assert native.linear_batch(0, bytes(16)) == bytes(24)
    assert peers.phase == "decode"


def test_lookup_batch_uses_trusted_rows_and_width():
    peers = peers_stub()
    item = peers.batch["descriptors"][0]
    item.update(kind="lookup", stage=None, columns=0)
    peers.batch["compiler_encoded_input_peak_bytes"] = 0
    protocol.batch_contract({"batch_execution": peers.batch})
    prices = protocol.batch_copy_prices({"batch_execution": peers.batch})
    assert prices["batch_binding_token_peak_bytes"] == 16
    assert prices["batch_runtime_backing_delta_bytes"] == 0
    seen = []

    class Native:
        def lookup_batch(self, tokens):
            seen.append(tokens)
            return bytes(7), bytes(9)

        def accept_batch(self, zero, one):
            assert (len(zero), len(one)) == (11, 13)
            return bytes(24)

    peers.exchange_batch = lambda frames, descriptor: (bytes(11), bytes(13))
    exchange = transport._NativeExchange(Native(), peers, 0)
    assert exchange.lookup_batch((1, 2)) == bytes(24)
    assert seen == [(1, 2)]
    with pytest.raises(ProtocolError):
        exchange.lookup_batch((1,))


def test_runtime_execution_comes_from_native_not_outer_metadata():
    peers = peers_stub()
    native = SimpleNamespace(execution=lambda: protocol.BATCH_EXECUTION)
    assert transport._NativeExchange(native, peers, 0).execution() == protocol.BATCH_EXECUTION
    native.execution = lambda: None
    with pytest.raises(ProtocolError):
        transport._NativeExchange(native, peers, 0).execution()


@pytest.fixture
def batch_worker(monkeypatch):
    experiment = SimpleNamespace(resolve=lambda: SimpleNamespace(client_runtime="compiled_gkm_v1"),
        pipeline=SimpleNamespace(components={"linear": SimpleNamespace(params={"execution": protocol.BATCH_EXECUTION})}))
    monkeypatch.setattr(worker, "admit_source", lambda _: SimpleNamespace())
    app = worker.create_delegated_worker_app(experiment, "worker_a", "k" * 32)
    owner = app.state.delegated_owner
    calls, refunds = [], []

    @contextmanager
    def reserve():
        try:
            yield
        finally:
            refunds.append(True)

    class Native:
        def execute(self, raw):
            calls.append(raw)
            return b"d" * 11

        def abort(self):
            pass

        def finish(self):
            return b"{}"

    owner.active, owner.request, owner.deadline = Native(), "a" * 64, worker.time.monotonic() + 60
    owner.batch = protocol.batch_contract(runtime())
    owner.reservation = worker._Reservation(reserve())
    endpoint = next(route.endpoint for route in app.routes if route.path.endswith("/{request}/{action}"))

    def request(length=7, authorized=True, rid="a" * 64, action="step"):
        reads = []

        async def receive():
            reads.append(True)
            return {"type": "http.request", "body": b"q" * length, "more_body": False}

        headers = [(b"content-length", str(length).encode())]
        if authorized:
            headers.append((b"authorization", b"Bearer " + b"k" * 32))
        value = Request({"type": "http", "method": "POST", "headers": headers,
                         "pllm.batch_http_backend": protocol.BATCH_HTTP_BACKEND,
                         "path_params": {"request": rid, "action": action}}, receive)

        async def connected():
            return False

        value.is_disconnected = connected
        return value, reads

    return SimpleNamespace(owner=owner, endpoint=endpoint, request=request, calls=calls, refunds=refunds)


def test_outstanding_batch_reply_rejects_body_before_allocation(batch_worker):
    h = batch_worker

    async def run():
        reply = await h.endpoint(h.request()[0])
        assert reply.status_code == 200 and reply.body == b"d" * 11
        assert h.owner.batch_operation_outstanding
        request, reads = h.request()
        assert (await h.endpoint(request)).status_code == 409 and not reads
        assert len(h.calls) == 1
        h.owner.retire()
        assert not h.refunds  # Unsent response owns reservation and slot.
        reply.close()
        assert not h.owner.batch_operation_outstanding and h.refunds == [True]

    asyncio.run(run())


@pytest.mark.parametrize("outcome", ["complete", "cancel", "error"])
def test_batch_slot_survives_send_unwind(batch_worker, outcome):
    h = batch_worker

    async def run():
        reply = await h.endpoint(h.request()[0])
        started, proceed = asyncio.Event(), asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                started.set()
                await proceed.wait()
                if outcome == "error":
                    raise ValueError("controlled send failure")

        task = asyncio.create_task(reply({"type": "http"}, None, send))
        await started.wait()
        h.owner.retire()
        reply.close()
        assert h.owner.batch_operation_outstanding and not h.refunds
        if outcome == "cancel":
            task.cancel()
        else:
            proceed.set()
        if outcome == "complete":
            await task
        else:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else ValueError):
                await task
        assert not h.owner.batch_operation_outstanding and h.refunds == [True]
        assert reply.body == b""

    asyncio.run(run())


def test_worker_wrong_declared_length_rejects_before_fixed_allocation(batch_worker, monkeypatch):
    h = batch_worker
    monkeypatch.setattr(worker, "bytearray", lambda *a: pytest.fail("allocated wrong batch length"), raising=False)
    request, reads = h.request(length=6)
    assert asyncio.run(h.endpoint(request)).status_code == 400
    assert not reads and not h.calls
    assert not h.owner.batch_operation_outstanding and h.refunds == [True]


def test_unproved_batch_host_rejects_without_touching_live_native(batch_worker):
    h = batch_worker
    request, reads = h.request()
    del request.scope["pllm.batch_http_backend"]
    active = h.owner.active
    assert asyncio.run(h.endpoint(request)).status_code == 503
    assert not reads and h.owner.active is active and not h.calls and not h.refunds
    h.owner.retire()


@pytest.mark.parametrize("action", ["finish", "cancel"])
def test_terminal_batch_reply_also_holds_operation_slot(batch_worker, action):
    h = batch_worker

    async def run():
        reply = await h.endpoint(h.request(length=0, action=action)[0])
        assert reply.status_code == 200 and h.owner.active is None
        assert h.owner.batch_operation_outstanding and not h.refunds
        reply.close()
        assert not h.owner.batch_operation_outstanding and h.refunds == [True]

    asyncio.run(run())


def test_batch_backend_streams_large_raw_frames_and_drains_io_owner():
    request = b"q" * (2 * protocol.BATCH_HTTP_CHUNK_BYTES + 3)
    reply = b"d" * (2 * protocol.BATCH_HTTP_CHUNK_BYTES + 7)
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            observed.append(self.rfile.read(int(self.headers["content-length"])))
            self.send_response(200)
            self.send_header("content-length", str(len(reply)))
            self.send_header("x-pllm-server-ns", "12")
            self.end_headers()
            self.wfile.write(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    peers = None
    try:
        peers = transport._Peers({"worker_a": (f"http://127.0.0.1:{server.server_port}", "a" * 32),
                                  "worker_b": ("http://127.0.0.1:1", "b" * 32)},
                                 2, lambda: False, batch=protocol.batch_contract(runtime()))
        peers.request_id = "a" * 64
        assert peers.post(0, "step", request, phase="online", limit=len(reply), expected=len(reply)) == reply
        assert observed == [request]
        assert peers.counts["role_link.worker_a.online_upload_bytes"] == len(request)
        assert peers.counts["role_link.worker_a.online_download_bytes"] == len(reply)
    finally:
        if peers is not None:
            peers.close()
            assert not peers.io.thread.is_alive() and peers.io.loop.is_closed()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def test_batch_cancellation_drains_bounded_backend_and_collector_with_retained_error(monkeypatch):
    cancelled, stop = threading.Event(), threading.Event()
    buffers = []

    class Buffer(bytearray):
        pass

    def allocate(size):
        value = Buffer(size)
        buffers.append(weakref.ref(value))
        return value

    monkeypatch.setattr(transport, "bytearray", allocate, raising=False)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["content-length"]))
            self.send_response(200)
            self.send_header("content-length", "100")
            self.end_headers()
            self.wfile.write(b"d")
            self.wfile.flush()
            cancelled.set()
            stop.wait(3)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    peers = None
    try:
        peers = transport._Peers({"worker_a": (f"http://127.0.0.1:{server.server_port}", "a" * 32),
                                  "worker_b": ("http://127.0.0.1:1", "b" * 32)},
                                 2, cancelled.is_set, batch=protocol.batch_contract(runtime()))
        peers.request_id = "a" * 64
        with pytest.raises(ProtocolError, match="cancelled") as caught:
            peers.post(0, "step", b"q" * 7, phase="online", limit=100, expected=100)
        assert caught.value.__traceback__ is not None
        assert all(ref() is None for ref in buffers)

        async def pending():
            return asyncio.all_tasks() - {asyncio.current_task()}

        assert peers.io.run(pending(), timeout=0.1) == set()
        assert "inference_stage_calls" not in peers.audit()
    finally:
        if peers is not None:
            peers.close()
            assert not peers.io.thread.is_alive() and peers.io.loop.is_closed()
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(2)
        assert not thread.is_alive()


def test_batch_server_protocol_pins_selector_read_size_and_scope_proof():
    import uvicorn

    async def run():
        observed = []
        block_sizes = []
        expected = 3 * protocol.BATCH_HTTP_CHUNK_BYTES + 19

        async def app(scope, receive, send):
            observed.append(scope.get("pllm.batch_http_backend"))
            while True:
                body = await receive()
                block_sizes.append(len(body["body"]))
                assert body["body"] == b"x" * len(body["body"])
                if not body["more_body"]:
                    break
            await send({"type": "http.response.start", "status": 200,
                        "headers": [(b"content-length", b"2")]})
            await send({"type": "http.response.body", "body": b"ok"})

        config = worker._batch_worker_config(app, lifespan="off", access_log=False)
        config.load()
        protocol_type = config.http_protocol_class
        state = uvicorn.server.ServerState()
        loop = asyncio.get_running_loop()
        server = await loop.create_server(lambda: protocol_type(config, state, {}), "127.0.0.1", 0)
        try:
            port = server.sockets[0].getsockname()[1]
            idle = [await asyncio.open_connection("127.0.0.1", port) for _ in range(4)]
            rejected_reader, rejected_writer = await asyncio.open_connection("127.0.0.1", port)
            assert await rejected_reader.read() == b""
            rejected_writer.close()
            await rejected_writer.wait_closed()
            assert len(state.connections) == 4
            for _, idle_writer in idle:
                idle_writer.close()
                await idle_writer.wait_closed()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(f"POST / HTTP/1.1\r\nHost: local\r\nContent-Length: {expected}\r\nConnection: close\r\n\r\n".encode() + b"x" * expected)
            await writer.drain()
            result = await reader.read()
            writer.close()
            await writer.wait_closed()
            assert result.endswith(b"ok") and observed == [protocol.BATCH_HTTP_BACKEND]
            assert sum(block_sizes) == expected and max(block_sizes) <= protocol.BATCH_HTTP_SERVER_BODY_BYTES
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(run())
