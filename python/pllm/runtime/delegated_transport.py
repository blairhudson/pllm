"""Bounded GKM HTTP execution, with request-end native acceptance.

Each authenticated channel receives only its role's frames. Local HTTP is for
co-located development; other IP-literal origins require verified HTTPS. HTTP
localhost is canonicalized to 127.0.0.1; DNS names are not admitted. One joined
I/O thread owns each response's asynchronous sockets, including when its caller
already runs an event loop. There are no retries, reconnects, speculative output
callbacks or reusable request state.
"""

from __future__ import annotations

import hashlib
import asyncio
from collections.abc import Mapping
import ipaddress
import json
import math
import threading
import time
from urllib.parse import urlsplit

import httpx

from pllm import _native
from pllm.deployment.resources import reserve_incremental_memory

from .client import ProtocolError
from .delegated_decoder import _execute_verified_window, _validate_decode_tokens, result_payload_limit
from .delegated_protocol import clear_failure_frames, digest, limits, pack, read_json, unpack
from .delegated_protocol import (BATCH_EXECUTION, BATCH_HTTP_CHUNK_BYTES, batch_contract, batch_memory_bytes,
                                 batch_input_bytes, batch_output_bytes)


_ROLES = ("worker_a", "worker_b")
_RETIRE_TIMEOUT = 5.0
_IO_DRAIN_TIMEOUT = 1.0
_TRANSPORT_ALLOWANCE_BYTES = 64 << 20


def request_memory_bytes(source, *, worker=False, role=None):
    """Native envelope plus outer copies, results, two pools and one I/O owner.

    The existing 64 MiB transport allowance includes the single thread's stack
    and selector loop. No DNS executor, additional pool or detached thread is
    admitted; the thread retires before this request's reservation is released.
    """
    size = source.execution_payload_estimate_bytes
    expanded = getattr(source, "capacity", "capsule") == "decoder"
    if type(size) is not int or not 0 < size <= (4 << 30 if expanded else 64 << 20):
        raise ValueError("GKM execution payload has no admitted bound")
    if expanded:
        resources = source.contract["decoder_resources"]
        if size != resources["client_payload_bytes"]:
            raise ValueError("GKM native resource contract differs")
        if worker:
            size = resources["worker_payload_bytes"]
    raw_runtime = getattr(source, "runtime_contract_bytes", None)
    if raw_runtime is not None:
        prices = batch_memory_bytes(json.loads(raw_runtime), "decoder" if expanded else "capsule")
        if prices:
            size = (prices["worker_payload_bytes"][role] if worker and role is not None else
                    max(prices["worker_payload_bytes"].values()) if worker else prices["client_payload_bytes"])
    return size + _TRANSPORT_ALLOWANCE_BYTES


def _batch_http_transport():
    # Plaintext loopback only. TLS/OpenSSL/BIO ownership is not admitted by this
    # backend's copy proof; endpoint admission and stream upgrade both reject it.
    # HTTP/2 is deliberately disabled: its buffering/flow-control proof differs.
    # Audit the installed HTTP/1 read bound as well as checking each yielded
    # chunk. chunk_size= would add HTTPX's chunker buffers/copies, so do not use it.
    from httpcore._async.http11 import AsyncHTTP11Connection
    from httpcore._backends.anyio import AnyIOBackend, AnyIOStream
    from anyio._backends._asyncio import SocketStream, StreamProtocol
    from asyncio.selector_events import _SelectorSocketTransport
    import httpcore

    if not 0 < AsyncHTTP11Connection.READ_NUM_BYTES <= BATCH_HTTP_CHUNK_BYTES:
        raise ValueError("GKM batch HTTP backend read bound is unsupported")

    class PlaintextStream(AnyIOStream):
        async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
            raise ValueError("GKM batch TLS is unsupported; bounded loopback HTTP only")

    class BoundedProtocol(StreamProtocol):
        def connection_made(self, socket_transport):
            if not isinstance(socket_transport, _SelectorSocketTransport):
                socket_transport.close()
                raise ValueError("GKM batch requires bounded asyncio selector I/O")
            self.transport = socket_transport
            socket_transport.max_size = BATCH_HTTP_CHUNK_BYTES
            super().connection_made(socket_transport)
            socket_transport.set_write_buffer_limits(high=0, low=0)
            socket_transport.pause_reading()

        def data_received(self, data):
            self.transport.pause_reading()
            if len(data) > BATCH_HTTP_CHUNK_BYTES or self.read_queue:
                self.transport.close()
                return
            super().data_received(data)

    class BoundedBackend(AnyIOBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            # Literal admission has already run. This loop forbids DNS/executors.
            try:
                async with asyncio.timeout(timeout):
                    socket_transport, protocol = await asyncio.get_running_loop().create_connection(
                        BoundedProtocol, host, port,
                        local_addr=(local_address, 0) if local_address else None)
                    stream = SocketStream(socket_transport, protocol)
                    if socket_options:
                        for option in socket_options:
                            socket_transport.get_extra_info("socket").setsockopt(*option)
                    return PlaintextStream(stream)
            except TimeoutError as error:
                raise httpcore.ConnectTimeout(str(error)) from error
            except OSError as error:
                raise httpcore.ConnectError(str(error)) from error

    class PlaintextTransport(httpx.AsyncHTTPTransport):
        def __init__(self):
            # HTTPX's default constructor eagerly builds an unused SSLContext,
            # even for HTTP. This selected backend admits no TLS setup at all.
            self._pool = httpcore.AsyncConnectionPool(
                ssl_context=None, http1=True, http2=False, retries=0,
                max_connections=1, max_keepalive_connections=1, keepalive_expiry=5.0,
                network_backend=BoundedBackend())

    return PlaintextTransport()


def _check_batch_http_connections(connections):
    # _connections has already enforced IP literals and loopback-only plaintext.
    # Do not weaken that confidentiality policy to support remote batch HTTP.
    if any(urlsplit(origin).scheme != "http" for origin, _ in connections):
        raise ValueError("GKM batch TLS is unsupported; bounded loopback HTTP only")


def _connections(connections, roles=_ROLES):
    if not isinstance(connections, Mapping) or len(connections) != len(roles) or set(connections) != set(roles):
        raise ValueError("GKM requires exactly two authenticated worker channels")
    result = []
    for role in roles:
        item = connections[role]
        if (type(item) not in (tuple, list) or len(item) != 2
                or any(type(value) is not str for value in item)):
            raise ValueError("GKM worker connection is invalid")
        origin, key = item
        url = urlsplit(origin)
        if (len(origin) > 2048 or not 16 <= len(key) <= 512
                or any(ord(c) < 33 or ord(c) > 126 for c in key)
                or url.scheme not in {"http", "https"} or not url.hostname
                or url.username is not None or url.password is not None
                or url.query or url.fragment or url.path not in {"", "/", "/v1", "/v1/"}):
            raise ValueError("GKM requires a bounded HTTP(S) origin and credential")
        host = "127.0.0.1" if url.scheme == "http" and url.hostname == "localhost" else url.hostname
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            raise ValueError("GKM worker origins require IP literals; DNS is not admitted") from None
        if "%" in host:
            raise ValueError("GKM scoped IP addresses are not admitted")
        if url.scheme == "http" and not address.is_loopback:
            raise ValueError("GKM remote worker channels require HTTPS")
        port = url.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("GKM worker port is outside its limit")
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
        if port is not None and port != (443 if url.scheme == "https" else 80):
            host = f"{host}:{port}"
        result.append((f"{url.scheme}://{host}", key))
    if len({row[0] for row in result}) != len(roles) or len({row[1] for row in result}) != len(roles):
        raise ValueError("GKM requires distinct worker origins and credentials")
    return result


class _NetworkLoop(asyncio.SelectorEventLoop):
    def run_in_executor(self, executor, func, *args):
        # IP admission makes DNS unnecessary. Fail closed if a backend ever
        # attempts executor work; a cancelled getaddrinfo cannot be joined safely.
        raise RuntimeError("GKM I/O does not admit executor work")


class _IOOwner:
    """One response-local selector thread; no caller-loop or executor ownership."""

    def __init__(self):
        self.loop = _NetworkLoop()
        self.closed = False
        self.thread = threading.Thread(target=self.loop.run_forever, name="pllm-gkm-io", daemon=False)
        try:
            self.thread.start()
        except BaseException:
            self.loop.close()
            raise

    def run(self, coroutine, *, timeout):
        if self.closed:
            coroutine.close()
            raise ProtocolError("GKM I/O owner is retired")
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        except BaseException:
            coroutine.close()
            raise
        try:
            # The coroutine owns its absolute deadline. This outer bound also
            # covers dispatch and cooperative cancellation cleanup.
            return future.result(timeout=max(0.0, timeout) + _IO_DRAIN_TIMEOUT)
        except BaseException:
            future.cancel()
            raise

    def close(self, clients):
        if self.closed:
            return

        async def drain():
            pending = asyncio.all_tasks() - {asyncio.current_task()}
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            closing = [asyncio.create_task(client.aclose()) for client in clients]
            unfinished = set()
            try:
                if closing:
                    _, unfinished = await asyncio.wait(closing, timeout=_IO_DRAIN_TIMEOUT)
            finally:
                for task in closing:
                    task.cancel()
                results = await asyncio.gather(*closing, return_exceptions=True)
            async with asyncio.timeout(_IO_DRAIN_TIMEOUT):
                await self.loop.shutdown_asyncgens()
            if unfinished:
                raise ProtocolError("GKM I/O pool retirement timed out")
            for result in results:
                if isinstance(result, BaseException):
                    raise result

        try:
            self.run(drain(), timeout=3 * _IO_DRAIN_TIMEOUT)
        finally:
            self.closed = True
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(_IO_DRAIN_TIMEOUT)
            if self.thread.is_alive():
                raise ProtocolError("GKM I/O thread did not retire")
            self.loop.close()


class _Peers:
    def __init__(self, connections, timeout, cancelled, *, capacity="capsule", roles=_ROLES, namespace="delegation", batch=None):
        # Admit both origins before constructing clients or any I/O owner.
        connections = _connections(connections, roles)
        if batch is not None:
            _check_batch_http_connections(connections)
        self.batch = batch
        self.batch_index = self.batch_exchanges = self.scalar_claims = 0
        self.roles, self.namespace = roles, namespace
        self.frame_limit, self.admission_limit = limits(capacity)
        self.io = _IOOwner()
        self.closed = False
        self.clients = []
        self.keys = []
        self.request_id = None
        self.live = [False] * len(roles)
        self.cancelled = cancelled
        self.deadline = time.monotonic() + timeout
        self.phase = "prefill"
        self.counts = {f"role_link.{role}.{phase}_{direction}_bytes": 0
                       for role in roles for phase in ("setup", "online", "teardown")
                       for direction in ("upload", "download")}
        self.phase_bytes = {"prefill": 0, "decode": 0}
        self.unknown = set()
        self.calls = self.server_ns = 0
        self.server_known = self.calls_known = True
        try:
            for origin, key in connections:
                self.clients.append(httpx.AsyncClient(
                    base_url=origin, timeout=timeout, verify=True, trust_env=False, follow_redirects=False,
                    limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                    headers={"authorization": f"Bearer {key}", "accept-encoding": "identity"},
                    **({"transport": _batch_http_transport()} if batch is not None else {}),
                ))
                self.keys.append(key)
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.cancelled() or time.monotonic() >= self.deadline:
            raise ProtocolError("GKM request cancelled or timed out", 499)

    def post(self, role, action, payload=b"", *, phase, limit, retire=False, expected=None):
        if self.closed:
            raise ProtocolError("GKM worker channels are retired")
        if not retire:
            self.check()
        deadline = time.monotonic() + _RETIRE_TIMEOUT if retire else self.deadline

        async def owned():
            async def cancellation():
                while True:
                    self.check()
                    await asyncio.sleep(0.01)

            task = watcher = None
            try:
                timeout = max(0.0, deadline - time.monotonic())
                async with asyncio.timeout(timeout):
                    task = asyncio.create_task(self._post(role, action, payload, phase=phase, limit=limit,
                                                          timeout=timeout, **({"expected": expected} if expected is not None else {})))
                    if retire:
                        return await task
                    watcher = asyncio.create_task(cancellation())
                    await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
                    if watcher.done():
                        await watcher
                    value = await task
                    self.check()
                    return value
            except TimeoutError:
                raise ProtocolError("GKM request cancelled or timed out", 499) from None
            finally:
                pending = [item for item in (task, watcher) if item is not None]
                for item in pending:
                    item.cancel()
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)

        try:
            return self.io.run(owned(), timeout=deadline - time.monotonic())
        except BaseException:
            if phase == "online":
                # Failed exchanges can have performed work even when no complete
                # body/headers arrived. Partial totals cannot mean zero compute.
                self.calls_known = self.server_known = False
            raise

    async def _post(self, role, action, payload, *, phase, limit, timeout, expected=None):
        prefix = f"role_link.{self.roles[role]}.{phase}_"
        observed = 0
        response_seen = False
        buffer = view = prefix_view = chunk = None
        upload = None
        try:
            path = (f"/v1/{self.namespace}/{action}" if action in {"manifest", "admit"}
                    else f"/v1/{self.namespace}/{self.request_id}/{action}")
            headers = {"content-type": "application/octet-stream"}
            content = payload
            if expected is not None:
                async def upload_chunks():
                    whole = memoryview(payload)
                    part = None
                    try:
                        for begin in range(0, len(whole), BATCH_HTTP_CHUNK_BYTES):
                            part = whole[begin:begin + BATCH_HTTP_CHUNK_BYTES]
                            yield part
                            part.release()
                            part = None
                    finally:
                        if part is not None:
                            part.release()
                        whole.release()

                upload = upload_chunks()
                content = upload
                headers["content-length"] = str(len(payload))
            async with self.clients[role].stream(
                "GET" if action == "manifest" else "POST", path,
                content=content, headers=headers,
                timeout=timeout,
            ) as response:
                response_seen = True
                sent = len(payload) if expected is not None else len(response.request.content)
                self.counts[prefix + "upload_bytes"] += sent
                if phase == "online":
                    self.phase_bytes[self.phase] += sent
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ProtocolError("GKM worker changed response encoding")
                if expected is not None:
                    lengths = response.headers.get_list("content-length")
                    if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                            or len(lengths[0]) > 10 or int(lengths[0]) != expected or expected != limit):
                        raise ProtocolError("GKM batch reply length differs from trusted descriptor")
                buffer = bytearray(limit)
                async for chunk in response.aiter_raw():
                    observed += len(chunk)
                    self.counts[prefix + "download_bytes"] += len(chunk)
                    if phase == "online":
                        self.phase_bytes[self.phase] += len(chunk)
                    if observed > limit:
                        raise ProtocolError("GKM worker response exceeds its bound")
                    if expected is not None and len(chunk) > BATCH_HTTP_CHUNK_BYTES:
                        raise ProtocolError("GKM batch HTTP backend chunk exceeds its bound")
                    buffer[observed - len(chunk):observed] = chunk
                if expected is not None and observed != expected:
                    raise ProtocolError("GKM truncated batch reply")
                if response.status_code != 200:
                    raise ProtocolError("GKM worker rejected its one-use request", response.status_code)
                if phase == "online":
                    self.calls += 1
                    elapsed = response.headers.get("x-pllm-server-ns", "")
                    if not elapsed.isascii() or not elapsed.isdigit() or len(elapsed) > 18:
                        self.server_known = False
                    else:
                        self.server_ns += int(elapsed)
                view = memoryview(buffer)
                prefix_view = view[:observed]
                return bytes(prefix_view)
        except BaseException:
            # A failed connection can have sent bytes before receiving headers.
            # Never manufacture a zero total for that unknown direction/window.
            if not response_seen:
                self.unknown.update((prefix + "upload_bytes", prefix + "download_bytes"))
            raise
        finally:
            try:
                if upload is not None:
                    await upload.aclose()
            finally:
                if prefix_view is not None:
                    prefix_view.release()
                if view is not None:
                    view.release()
                buffer = view = prefix_view = chunk = payload = content = upload = None

    def admit(self, contract, publication, pin, starts, experiment, source, request_id, rows, outputs):
        self.request_id = digest(request_id.hex())
        for index, role in enumerate(_ROLES):
            self.check()
            manifest = unpack(self.post(index, "manifest", phase="setup", limit=4096))
            expected = {"schema": "pllm.gkm.worker.v1", "role": role,
                        "configuration": experiment.configuration_digest(),
                        "source_lock": source.source_lock_digest, "tokenizer": source.tokenizer_digest,
                         "frame_limit": self.frame_limit}
            if (set(manifest) != set(expected) | {"epoch"}
                    or any(manifest[key] != value for key, value in expected.items())):
                raise ProtocolError("GKM worker manifest differs from the admitted source")
            epoch = digest(manifest["epoch"])
            metadata = {
                "schema": "pllm.gkm.admit.v1", "request": self.request_id, "epoch": epoch,
                "configuration": experiment.configuration_digest(),
                "source_lock": source.source_lock_digest, "tokenizer": source.tokenizer_digest,
                "input_rows": rows, "output_tokens": outputs,
                "publication": publication, "pin": pin, "start": starts[index],
            }
            # Burn this ID even if acknowledgement is lost or malformed.
            self.live[index] = True
            acknowledgement = unpack(self.post(
                index, "admit", pack(metadata, limit=self.admission_limit),
                phase="setup", limit=16 << 10,
            ))
            expected_ack = {"schema": "pllm.gkm.ready.v1", "request": self.request_id, "role": role,
                            "contract_sha256": hashlib.sha256(contract).hexdigest(), "epoch": epoch}
            if acknowledgement != expected_ack:
                raise ProtocolError("GKM worker admission differs from the native request")

    def exchange(self, frames):
        self.check()
        if type(frames) is not tuple or len(frames) != 2 or any(
                type(frame) is not bytes or not 0 < len(frame) <= self.frame_limit for frame in frames):
            raise ProtocolError("GKM native role frames exceed the transport contract")
        result = tuple(self.post(index, "step", frame, phase="online", limit=self.frame_limit)
                       for index, frame in enumerate(frames))
        self.check()
        return result

    def batch_descriptor(self, kind, stage=None):
        if self.batch is None or self.batch_index >= len(self.batch["descriptors"]):
            raise ProtocolError("GKM batch descriptor is unavailable")
        descriptor = self.batch["descriptors"][self.batch_index]
        if descriptor["kind"] != kind or kind == "linear" and descriptor["stage"] != stage:
            raise ProtocolError("GKM batch operation differs from trusted descriptor")
        return descriptor

    def exchange_batch(self, frames, descriptor):
        self.check()
        if (self.batch is None or descriptor is not self.batch["descriptors"][self.batch_index]
                or type(frames) is not tuple or len(frames) != 2
                or any(type(frame) is not bytes or len(frame) != expected
                       for frame, expected in zip(frames, descriptor["request_bytes"], strict=True))):
            raise ProtocolError("GKM batch request length differs from trusted descriptor")
        replies = tuple(self.post(role, "step", frame, phase="online", limit=expected, expected=expected)
                        for role, (frame, expected) in enumerate(zip(frames, descriptor["reply_bytes"], strict=True)))
        self.batch_index += 1
        self.batch_exchanges += 1
        self.scalar_claims += descriptor["rows"]
        self.check()
        return replies

    def finish(self):
        receipts = []
        for index in range(2):
            receipt = read_json(self.post(index, "finish", phase="teardown", limit=16 << 10))
            self.live[index] = False
            receipts.append(receipt)
        self.check()
        return receipts

    def audit(self):
        snapshot = {name: value for name, value in self.counts.items() if name not in self.unknown}
        for direction in ("upload", "download"):
            for name, phases in ((f"inference_{direction}_bytes", ("online",)),
                                 (f"inference_control_{direction}_bytes", ("setup", "teardown"))):
                keys = [f"role_link.{role}.{phase}_{direction}_bytes" for role in self.roles for phase in phases]
                if all(key in snapshot for key in keys):
                    snapshot[name] = sum(snapshot[key] for key in keys)
        if not any(".online_" in key for key in self.unknown):
            snapshot.update({f"generation_{phase}_online_body_bytes": value
                             for phase, value in self.phase_bytes.items()})
        if self.calls_known:
            snapshot["inference_stage_calls"] = self.calls
            if getattr(self, "batch", None) is None:
                snapshot["online_steps"] = self.calls // 2
            else:
                snapshot["batch_exchanges"] = self.batch_exchanges
                snapshot["native_scalar_claims"] = self.scalar_claims
                # Public online_steps counts scalar-equivalent work, not HTTP
                # pairs. Never infer batch row claims from calls // 2.
                snapshot["online_steps"] = self.scalar_claims
        if self.server_known:
            snapshot["inference_server_ns"] = self.server_ns
        return snapshot

    def close(self):
        if self.closed:
            return
        self.closed = True
        deadline = time.monotonic() + _RETIRE_TIMEOUT

        async def retire(index):
            try:
                await self._post(index, "cancel", b"", phase="teardown", limit=1024,
                                 timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                pass  # Server expiry is the fallback for an unreachable peer.
            finally:
                self.live[index] = False

        async def retire_all():
            tasks = [asyncio.create_task(retire(index)) for index in range(len(self.clients))
                     if self.live[index]]
            try:
                async with asyncio.timeout(max(0.0, deadline - time.monotonic())):
                    if tasks:
                        await asyncio.gather(*tasks)
            except TimeoutError:
                pass
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        try:
            self.io.run(retire_all(), timeout=deadline - time.monotonic())
        finally:
            self.io.close(self.clients)


class _NativeExchange:
    def __init__(self, native, peers, head_stage):
        self.native, self.peers, self.head_stage = native, peers, head_stage

    def contract(self):
        return self.native.decoder()

    def execution(self):
        selected = getattr(self.native, "execution", lambda: None)()
        expected = BATCH_EXECUTION if getattr(self.peers, "batch", None) is not None else None
        if selected != expected:
            raise ProtocolError("GKM native execution differs from admitted HTTP contract")
        return selected

    def coefficients(self):
        return self.native.coefficients()

    def lookup(self, token):
        return self.native.accept(*self.peers.exchange(self.native.lookup(token)))

    def linear(self, stage, values):
        result = self.native.accept(*self.peers.exchange(self.native.linear(stage, values)))
        if stage == self.head_stage:
            self.peers.phase = "decode"
        return result

    def lookup_batch(self, tokens):
        descriptor = self.peers.batch_descriptor("lookup")
        if (type(tokens) is not tuple or len(tokens) != descriptor["rows"]
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ProtocolError("GKM batch lookup differs from trusted descriptor")
        frames = replies = result = None
        try:
            frames = self.native.lookup_batch(tokens)
            tokens = None
            replies = self.peers.exchange_batch(frames, descriptor)
            frames = None  # Native acceptance must not overlap Python requests.
            result = self.native.accept_batch(*replies)
            if type(result) is not bytes or len(result) != batch_output_bytes(descriptor):
                raise ProtocolError("GKM batch lookup result differs from trusted descriptor")
            return result
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            tokens = frames = replies = result = None

    def linear_batch(self, stage, values):
        descriptor = self.peers.batch_descriptor("linear", stage)
        if type(values) is not bytes or len(values) != batch_input_bytes(descriptor):
            raise ProtocolError("GKM batch linear input differs from trusted descriptor")
        frames = replies = result = None
        try:
            frames = self.native.linear_batch(stage, values)
            values = None
            replies = self.peers.exchange_batch(frames, descriptor)
            frames = None
            result = self.native.accept_batch(*replies)
            if type(result) is not bytes or len(result) != batch_output_bytes(descriptor):
                raise ProtocolError("GKM batch linear result differs from trusted descriptor")
            if stage == self.head_stage:
                self.peers.phase = "decode"
            return result
        except BaseException as error:
            clear_failure_frames(error)
            raise
        finally:
            values = frames = replies = result = None

    def finish(self):
        completions = self.peers.finish()
        acceptance = self.native.finish()
        accepted = read_json(acceptance)
        if getattr(self.peers, "batch", None) is not None:
            descriptors = self.peers.batch["descriptors"]
            exchanges = len(descriptors)
            claims = sum(descriptor["rows"] for descriptor in descriptors)
            if (type(accepted.get("steps")) is not int or accepted["steps"] != claims
                    or self.peers.batch_index != exchanges or self.peers.batch_exchanges != exchanges
                    or self.peers.scalar_claims != claims or not self.peers.calls_known
                    or self.peers.calls != 2 * exchanges):
                raise ProtocolError("GKM batch counters differ from accepted scalar steps or role calls")
        for index, completion in enumerate(completions):
            expected = {key: accepted[key] for key in (
                "contract", "decoder_contract", "numeric_profile", "source", "publication",
                "context", "request", "steps", "sharing",
            )}
            expected.update(schema_version="pllm.gkm.sdk-worker-completion.v1", worker=index)
            if completion != expected:
                raise ProtocolError("GKM worker completion differs from final native acceptance")
        self.peers.check()
        return acceptance

    def abort(self):
        self.native.abort()


def execute_request(experiment, source, token_ids, output_tokens, connections, timeout, cancelled,
                    *, decode_tokens=None):
    """Return complete accepted state; all error paths retire local and remote owners."""
    capacity = getattr(source, "capacity", "capsule")
    expanded = capacity == "decoder"
    if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 3600
            or type(token_ids) not in (list, tuple) or not 1 <= len(token_ids) <= (512 if expanded else 32)
            or any(type(token) is not int or not 0 <= token < source.contract["vocabulary"] for token in token_ids)
            or type(output_tokens) is not int or not 1 <= output_tokens <= (65 if expanded else 9) or not callable(cancelled)):
        raise ValueError("GKM request exceeds the bounded transport contract")
    _validate_decode_tokens(decode_tokens, output_tokens - 1, source.contract["vocabulary"])
    connections = dict(zip(_ROLES, _connections(connections), strict=True))
    # Full-budget source admission is already sealed. Actual native binding must
    # also precede value loading, even for a smaller request.
    contract_bytes = source.runtime_contract(experiment, len(token_ids), output_tokens)
    pipeline = getattr(experiment, "pipeline", None)
    linear = None if pipeline is None else pipeline.components.get("linear")
    selected = None if linear is None else linear.params.get("execution")
    batch = None
    if selected is not None:
        if selected != BATCH_EXECUTION:
            raise ProtocolError("GKM unsupported HTTP execution selector")
        batch = batch_contract(json.loads(contract_bytes))
        if batch is None:
            raise ProtocolError("GKM batch HTTP metadata differs from Experiment")
        _check_batch_http_connections(connections.values())
    result_payload_limit(source.contract, capacity=capacity)
    native = peers = decoder = record = None
    with reserve_incremental_memory(request_memory_bytes(source)):
        try:
            try:
                if cancelled():
                    raise ProtocolError("GKM request cancelled", 499)
                native = source.load_transport(experiment, len(token_ids), output_tokens, cancelled=cancelled)
                if native.contract() != contract_bytes:
                    raise ProtocolError("GKM client native admission differs from the source contract")
                decoder = json.loads(native.decoder())
                peers = _Peers(connections, timeout, cancelled, capacity=capacity,
                               **({"batch": batch} if batch is not None else {}))
                peers.admit(contract_bytes, native.publication(), native.publication_digest(),
                            native.starts(), experiment, source, native.request_id(), len(token_ids), output_tokens)
                return _execute_verified_window(
                    _NativeExchange(native, peers, len(decoder["matrices"]) - 1),
                    decoder, tuple(token_ids), cancelled=cancelled, decode_tokens=decode_tokens,
                )
            finally:
                try:
                    if native is not None:
                        native.abort()
                finally:
                    if peers is not None:
                        try:
                            peers.close()
                        finally:
                            record = getattr(cancelled, "record_audit", None)
                            if record is not None:
                                record(peers.audit())
        except BaseException as error:
            # Joined I/O and unwound decoder/import frames may otherwise keep
            # tentative payloads in a propagated exception after budget refund.
            clear_failure_frames(error)
            raise
        finally:
            native = peers = decoder = record = None
