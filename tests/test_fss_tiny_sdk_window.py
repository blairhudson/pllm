"""Tiny ordinary SDK + authenticated sockets + installed native window provider.

Generated public weights only. Arithmetic, source import, material/tape/provider,
peer frames, receipts and final reconstruction are real. Fault injection truncates
one genuine B material packet; it does not replace native run or manufacture an
accepted output. This is not trained quality, process isolation or RSS evidence.
"""
import asyncio
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import struct
import threading
import time
from types import SimpleNamespace as NS

import httpx
import pytest
import uvicorn

from examples.papers.fusefss_smoke import dense_tensor_smoke
from pllm import OpenAI, _native
from pllm.runtime import fss_protocol as p, fss_source, fss_transport, fss_worker
from pllm.runtime.client import ProtocolError
from test_fss_worker import client_key, credentials, key


pytestmark = [pytest.mark.integration, pytest.mark.rust]
ROOT = Path(__file__).resolve().parents[1]


def q12_expected(root):
    """Feed original IEEE source words to the existing independent Python oracle."""
    path = ROOT / "paper/reproduction-contracts/fusefss/decoder/oracle.py"
    spec = importlib.util.spec_from_file_location("tiny_sdk_q12_oracle", path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    data = (root / "model.safetensors").read_bytes()
    size = int.from_bytes(data[:8], "little")
    header, payload = json.loads(data[8:8 + size]), data[8 + size:]
    names = {"model.embed_tokens.weight": "embedding", "model.norm.weight": "final.gamma",
             "model.layers.0.input_layernorm.weight": "layer.0.norm1.gamma",
             "model.layers.0.post_attention_layernorm.weight": "layer.0.norm2.gamma"}
    for native, role in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"), ("o_proj", "o")):
        for suffix in ("weight", "bias") if role != "o" else ("weight",):
            names[f"model.layers.0.self_attn.{native}.{suffix}"] = f"layer.0.{role}.{suffix}"
    for role in ("gate", "up", "down"):
        names[f"model.layers.0.mlp.{role}_proj.weight"] = f"layer.0.{role}.weight"
    assert set(names) == set(header)
    tensors = {}
    for name, target in names.items():
        item = header[name]
        assert item["dtype"] == "F32"
        begin, end = item["data_offsets"]
        tensors[target] = {"dtype": "f32", "shape": item["shape"], "words": [
            payload[offset:offset + 4][::-1].hex() for offset in range(begin, end, 4)]}
    # Frozen independent F32 sin/cos words, not the production rotary importer.
    cases = json.loads(path.with_name("inputs.json").read_bytes())["cases"]
    frozen, = [case for case in cases if case["id"] == "qwen2-tied-generated"]
    for name in ("rope.cos", "rope.sin"):
        words = frozen["tensors"][name]["words"]
        tensors[name] = {"dtype": "f32", "shape": [4, 2],
                         "words": [words[row * 4] for row in range(4) for _ in range(2)]}
    case = {"id": "tiny-sdk-window", "family": "qwen2", "profile": oracle.DENSE_PROFILE,
            "graph_status": oracle.GRAPH, "dimensions": dict(hidden=2, intermediate=2, layers=1,
                heads=1, kv_heads=1, head_dim=2, vocab=3, max_positions=4),
            "prefix": [1], "generation_count": 2, "eos_id": 0, "tied_head": True,
            "attention_scale": {"dtype": "f32", "shape": [1], "words": ["3f3504f3"]},
            "norm_contract": "d13-rms-sumsq-plus-one",
            "rope_contract": "d13-split-half-per-product-floor", "tensors": tensors}
    result = oracle.run(case)
    assert result["final"]["generated_ids"] == [2, 2]
    assert any(oracle.ieee_q12("f32", word) for word in tensors["embedding"]["words"])
    return tuple(result["final"]["generated_ids"])


@pytest.fixture
def tiny(tmp_path, monkeypatch, request):
    experiment = dense_tensor_smoke(tmp_path)
    if hasattr(request, "param"):
        experiment = experiment.with_params(budget__requests=request.param)
    root = Path(experiment.pipeline.model.source)
    # Keep the real nonzero fixture/tied head, but make the last vocabulary
    # row win. A silently truncated vocabulary would now fail the oracle.
    path = root / "model.safetensors"
    data = bytearray(path.read_bytes())
    size = int.from_bytes(data[:8], "little")
    begin, _ = json.loads(data[8:8 + size])["model.embed_tokens.weight"]["data_offsets"]
    row_one = struct.unpack_from("<2f", data, 8 + size + begin + 8)
    struct.pack_into("<2f", data, 8 + size + begin + 16, *(2 * value for value in row_one))
    path.write_bytes(data)  # Before any source is resolved/pinned, never during an attempt.
    state = NS(experiment=experiment, fault=False, injected=False, sessions=[], roles=[],
        native_errors=[], windows=[[], []], scalar=[0, 0], exchanges=[0, 0],
        finish_entered=[threading.Event(), threading.Event()], waiting=threading.Event(),
        wait_snapshot=None, frozen=None, parsed=set(), reconstructed=0, results=[],
        native_failure_at=None, client_owners=[], peers=[], threads=[], servers=[])
    state.expected = q12_expected(root)
    original_role, original_client = _native.FssRoleRequest, _native.FssClientRequest
    original_attempt, original_receipt = fss_worker._Attempt.__init__, fss_transport._receipt
    original_peers = fss_transport._Peers

    class Mailbox(asyncio.Event):
        def __init__(self, attempt):
            super().__init__()
            self.attempt = attempt

        async def wait(self):
            attempt = self.attempt
            if (state.fault and attempt.native is not None and attempt.native.role == 0
                    and attempt.peer_sent and attempt.mailbox is None):
                def blocked():
                    assert any(not future.done() for future in self._waiters)
                    state.wait_snapshot = (attempt.peer_sent, attempt.peer_received, attempt.callback_pending)
                    state.waiting.set()
                attempt.loop.call_soon(blocked)
            return await super().wait()

    def attempt_init(attempt, *args, **kwargs):
        original_attempt(attempt, *args, **kwargs)
        attempt.mail_available = Mailbox(attempt)
        state.sessions.append(attempt)

    class Role:
        """Observation/fault boundary around, never instead of, the real owner."""
        def __init__(self, source, contract, request, role, **options):
            assert options == {"material_window": True}
            self.real = original_role(source, contract, request, role, **options)
            self.role, self.closed = role, False
            state.roles.append(self)
            if role < 2:
                assert json.loads(self.real.metadata())["material_receive"]["records_bound"] == 8

        def __getattr__(self, name):
            return getattr(self.real, name)

        def run(self, frame, callbacks):
            role = self.role

            class Provider:
                def __getattr__(self, name):
                    return getattr(callbacks, name)

                def material(self, raw):
                    state.scalar[role] += 1
                    return callbacks.material(raw)

                def material_window(self, metadata):
                    packets = callbacks.material_window(metadata)  # Real encrypted tape/window.
                    state.windows[role].append(len(metadata))
                    if state.fault and role == 1 and not state.injected:
                        assert state.waiting.wait(8), "A never waited for B's peer frame"
                        assert all(event.wait(8) for event in state.finish_entered), "finish RPCs not concurrent"
                        assert state.exchanges[1] == 0 and state.reconstructed == 0
                        state.injected = True
                        return (packets[0][:-1], *packets[1:])
                    return packets

                def exchange(self, metadata, data):
                    state.exchanges[role] += 1
                    return callbacks.exchange(metadata, data)

            try:
                return self.real.run(frame, Provider())
            except BaseException as error:
                state.native_errors.append((role, type(error)))
                if role == 1:
                    state.native_failure_at = time.monotonic()
                raise

        def close(self):
            self.real.close()
            self.closed = True

    class Client:
        def __init__(self, *args):
            self.real, self.closed = original_client(*args), False
            state.client_owners.append(self)

        def __getattr__(self, name):
            return getattr(self.real, name)

        def take_input(self, role):
            assert any(owner.role == 2 and owner.closed for owner in state.roles)
            return self.real.take_input(role)

        def finish(self, *frames):
            assert state.parsed == set(p.ROLES)
            assert not state.fault, "failed request reached final reconstruction"
            state.reconstructed += 1
            return self.real.finish(*frames)

        def close(self):
            self.real.close()
            self.closed = True

    def parsed(raw, role, **options):
        result = original_receipt(raw, role, **options)
        state.parsed.add(role)
        return result

    def peers(*args, **kwargs):
        owner = original_peers(*args, **kwargs)
        state.peers.append(owner)
        return owner

    monkeypatch.setattr(fss_worker._Attempt, "__init__", attempt_init)
    monkeypatch.setattr(_native, "FssRoleRequest", Role)
    monkeypatch.setattr(_native, "FssClientRequest", Client)
    monkeypatch.setattr(fss_transport, "_receipt", parsed)
    monkeypatch.setattr(fss_transport, "_Peers", peers)

    class Server(uvicorn.Server):
        def __init__(self, app):
            super().__init__(uvicorn.Config(app, log_level="error", access_log=False, lifespan="on"))
            self.ready = threading.Event()

        async def startup(self, sockets=None):
            await super().startup(sockets=sockets)
            self.ready.set()

    with ExitStack() as stack:
        sockets = {}
        for role in p.ROLES:
            listener = stack.enter_context(socket.socket())
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.setblocking(False)
            sockets[role] = listener
        origins = {role: f"http://127.0.0.1:{listener.getsockname()[1]}" for role, listener in sockets.items()}
        state.connections = {role: (origin, client_key(role)) for role, origin in origins.items()}

        def shutdown():
            for server in state.servers:
                server.should_exit = True
            for thread in state.threads:
                thread.join(10)
                assert not thread.is_alive(), "tiny native worker did not drain"

        stack.callback(shutdown)  # Also owns partial startup failures.
        for role in p.ROLES:
            outgoing = p.HOLDERS if role == "dealer" else tuple(r for r in p.HOLDERS if r != role)
            app = fss_worker.create_fss_worker_app(experiment, role, client_key(role),
                peer_connections={r: (origins[r], key(role, r)) for r in outgoing},
                peer_credentials=credentials(role), ttl=60, tape_directory=tmp_path)

            async def observed(scope, receive, send, app=app, role=role):
                if scope["type"] == "http" and scope["path"].endswith("/finish") and role in p.HOLDERS:
                    state.finish_entered[p.HOLDERS.index(role)].set()
                return await app(scope, receive, send)

            server = Server(observed)
            thread = threading.Thread(target=server.run, kwargs={"sockets": [sockets[role]]},
                                      name=f"tiny-fss-{role}")
            state.servers.append(server)
            state.threads.append(thread)
            thread.start()
            assert server.ready.wait(8), f"{role} did not start"
        try:
            # Authentication remains real; no transport/HTTP response replacement.
            for origin in origins.values():
                assert httpx.get(origin + "/v1/fss/metrics", trust_env=False, timeout=5).status_code == 401
            yield state
        finally:
            stack.close()
            assert all(owner.closed for owner in state.roles + state.client_owners)
            assert all(session.reservation is None and not session.callback_tasks
                       and not session.callback_futures for session in state.sessions)
            assert all(owner.io.closed and not owner.io.thread.is_alive() for owner in state.peers)


def assert_retired(state, client):
    assert client._core._active is None and not client._core.records
    for role, (origin, credential) in state.connections.items():
        response = httpx.get(origin + "/v1/fss/metrics", headers={"authorization": f"Bearer {credential}"},
                             timeout=5, trust_env=False)
        response.raise_for_status()
        assert response.json()["active_sessions"] == 0, role
    assert all(session.reservation is None and session.native is None and session.loaded is None
               and not session.callback_tasks and not session.callback_futures for session in state.sessions)


def test_tiny_sdk_real_window_full_vocabulary_and_final_only_acceptance(tiny):
    source = fss_source.resolve_client_source(tiny.experiment)
    assert source.contract["vocabulary"] == 3
    with OpenAI(experiment=tiny.experiment, role_connections=tiny.connections, timeout=30) as client:
        client._core._result_hook = tiny.results.append
        with client.responses.create(input="hello", max_output_tokens=2, stream=True, store=False) as stream:
            events = []
            for event in stream:
                assert tiny.reconstructed == 1 and len(tiny.results) == 1
                events.append(event)
        assert events and tiny.results[0].selected == tiny.expected
        assert tiny.scalar == [0, 0] and all(any(count > 1 for count in sizes) for sizes in tiny.windows)
        assert all(tiny.exchanges) and not tiny.native_errors
        assert_retired(tiny, client)


def test_tiny_sdk_b_native_packet_failure_while_a_waits_drains_without_output(tiny):
    tiny.fault = True
    with OpenAI(experiment=tiny.experiment, role_connections=tiny.connections, timeout=30) as client:
        client._core._result_hook = tiny.results.append
        def frozen(marker):
            if tiny.frozen is None:
                tiny.frozen = marker
        client._core._fss_failure_observer = frozen
        with pytest.raises(ProtocolError):
            with client.responses.create(input="hello", max_output_tokens=2, stream=True, store=False) as stream:
                for _ in stream:
                    pytest.fail("native failure disclosed an application event")
        assert tiny.injected and tiny.waiting.is_set() and all(event.is_set() for event in tiny.finish_entered)
        assert tiny.wait_snapshot == (1, 0, True)
        assert any(role == 1 for role, _ in tiny.native_errors) and tiny.exchanges[1] == 0
        assert tiny.native_failure_at is not None and time.monotonic() - tiny.native_failure_at < 8
        assert tiny.frozen[:3] == ("worker_b", "finish", "failed")
        assert not tiny.results and tiny.reconstructed == 0
        assert_retired(tiny, client)


def observe_serial_material(state, monkeypatch):
    """Observe real provider packets without retaining plaintext packet/view owners."""
    fingerprints = {}
    original_run = _native.FssRoleRequest.run  # Existing fixture's delegating native wrapper.

    def run(owner, frame, callbacks):
        class ObservedCallbacks:
            def __getattr__(self, name):
                return getattr(callbacks, name)

            def material_window(self, metadata):
                packets = callbacks.material_window(metadata)
                identity = (callbacks.attempt.request_id, owner.role)
                if identity not in fingerprints:
                    fingerprints[identity] = hashlib.sha256(packets[0]).digest()
                return packets

        return original_run(owner, frame, ObservedCallbacks())

    monkeypatch.setattr(_native.FssRoleRequest, "run", run)
    return fingerprints


def assert_serial_retired(state, client, request_number):
    assert_retired(state, client)
    assert not client._core._tokenizer_scopes and client._core._tokenizer_owner.get() is None
    assert client._core._fss_control_snapshot() is None
    assert len(state.sessions) == 3 * request_number
    assert len(state.roles) == 3 * request_number and len(state.client_owners) == request_number
    assert all(owner.closed for owner in state.roles + state.client_owners)
    assert len(state.peers) == request_number
    for peers in state.peers:
        assert peers.io.closed and not peers.io.thread.is_alive()
        assert not peers.io._fss_finish_pending
        assert all(channel.is_closed for channel in peers.clients)
    current = state.peers[-1]
    assert {session.request_id for session in state.sessions[-3:]} == {current.request_id}
    return current


def consume_serial_success(state, client):
    """Only test observations reset; no production attempt/root/token state touched."""
    state.parsed.clear()
    before = state.reconstructed
    with client.responses.create(input="hello", max_output_tokens=2, stream=True, store=False) as stream:
        events = []
        for event in stream:
            assert state.reconstructed == before + 1 and len(state.results) == state.reconstructed
            events.append(event)
    assert events and state.results[-1].selected == state.expected


def assert_fresh_serial_owners(state, fingerprints, first, second):
    assert first.request_id != second.request_id and first.io is not second.io
    assert first.io._fss_finish_pending is not second.io._fss_finish_pending
    assert state.client_owners[0].real is not state.client_owners[1].real
    assert all(a.real is not b.real for a in state.roles[:3] for b in state.roles[3:])
    for role in (0, 1):
        assert fingerprints[first.request_id, role] != fingerprints[second.request_id, role]
    assert state.scalar == [0, 0] and all(any(size > 1 for size in sizes) for sizes in state.windows)
    # Real authenticated endpoints reject the retired identity after fresh success.
    for role in p.HOLDERS:
        origin, credential = state.connections[role]
        response = httpx.post(origin + f"/v1/fss/{first.request_id}/finish",
            headers={"authorization": f"Bearer {credential}"}, timeout=5, trust_env=False)
        assert response.status_code == 409


@pytest.mark.parametrize("tiny", [2], indirect=True)
def test_tiny_sdk_same_client_two_fresh_successes(tiny, monkeypatch):
    fingerprints = observe_serial_material(tiny, monkeypatch)
    with OpenAI(experiment=tiny.experiment, role_connections=tiny.connections, timeout=30) as client:
        client._core._result_hook = tiny.results.append
        consume_serial_success(tiny, client)
        first = assert_serial_retired(tiny, client, 1)
        source = client._core._source
        consume_serial_success(tiny, client)
        second = assert_serial_retired(tiny, client, 2)
        assert client._core._source is source  # Same admitted public owner, fresh request material.
        assert tiny.reconstructed == 2 and not tiny.native_errors
        assert_fresh_serial_owners(tiny, fingerprints, first, second)


@pytest.mark.parametrize("tiny", [2], indirect=True)
def test_tiny_sdk_same_client_recovers_after_native_b_failure(tiny, monkeypatch):
    fingerprints = observe_serial_material(tiny, monkeypatch)
    tiny.fault = True
    markers = []
    with OpenAI(experiment=tiny.experiment, role_connections=tiny.connections, timeout=30) as client:
        client._core._result_hook = tiny.results.append
        client._core._fss_failure_observer = markers.append
        with pytest.raises(ProtocolError):
            with client.responses.create(input="hello", max_output_tokens=2, stream=True, store=False) as stream:
                for _ in stream:
                    pytest.fail("failed first request disclosed an application event")
        assert tiny.injected and tiny.wait_snapshot == (1, 0, True)
        assert len(markers) == 1 and markers[0][:3] == ("worker_b", "finish", "failed")
        assert tiny.reconstructed == 0 and not tiny.results and tiny.exchanges[1] == 0
        first = assert_serial_retired(tiny, client, 1)
        source, errors = client._core._source, tuple(tiny.native_errors)
        tiny.fault = False  # Disable only this test's packet corruptor; do not repair SDK state.
        consume_serial_success(tiny, client)
        second = assert_serial_retired(tiny, client, 2)
        assert client._core._source is source and tuple(tiny.native_errors) == errors
        assert tiny.reconstructed == 1 and len(markers) == 1
        assert_fresh_serial_owners(tiny, fingerprints, first, second)
