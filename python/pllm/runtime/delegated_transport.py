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


_ROLES = ("worker_a", "worker_b")
_RETIRE_TIMEOUT = 5.0
_IO_DRAIN_TIMEOUT = 1.0
_TRANSPORT_ALLOWANCE_BYTES = 64 << 20


def request_memory_bytes(source, *, worker=False):
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
    return size + _TRANSPORT_ALLOWANCE_BYTES


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
    def __init__(self, connections, timeout, cancelled, *, capacity="capsule", roles=_ROLES, namespace="delegation"):
        # Admit both origins before constructing clients or any I/O owner.
        connections = _connections(connections, roles)
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
                ))
                self.keys.append(key)
        except BaseException:
            self.close()
            raise

    def check(self):
        if self.cancelled() or time.monotonic() >= self.deadline:
            raise ProtocolError("GKM request cancelled or timed out", 499)

    def post(self, role, action, payload=b"", *, phase, limit, retire=False):
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
                                                          timeout=timeout))
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

    async def _post(self, role, action, payload, *, phase, limit, timeout):
        prefix = f"role_link.{self.roles[role]}.{phase}_"
        observed = 0
        response_seen = False
        buffer = view = prefix_view = chunk = None
        try:
            path = (f"/v1/{self.namespace}/{action}" if action in {"manifest", "admit"}
                    else f"/v1/{self.namespace}/{self.request_id}/{action}")
            async with self.clients[role].stream(
                "GET" if action == "manifest" else "POST", path,
                content=payload, headers={"content-type": "application/octet-stream"},
                timeout=timeout,
            ) as response:
                response_seen = True
                self.counts[prefix + "upload_bytes"] += len(response.request.content)
                if phase == "online":
                    self.phase_bytes[self.phase] += len(response.request.content)
                if response.headers.get("content-encoding", "identity") != "identity":
                    raise ProtocolError("GKM worker changed response encoding")
                buffer = bytearray(limit)
                async for chunk in response.aiter_raw():
                    observed += len(chunk)
                    self.counts[prefix + "download_bytes"] += len(chunk)
                    if phase == "online":
                        self.phase_bytes[self.phase] += len(chunk)
                    if observed > limit:
                        raise ProtocolError("GKM worker response exceeds its bound")
                    buffer[observed - len(chunk):observed] = chunk
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
            if prefix_view is not None:
                prefix_view.release()
            if view is not None:
                view.release()
            buffer = view = prefix_view = chunk = payload = None

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
            snapshot["online_steps"] = self.calls // 2
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

    def coefficients(self):
        return self.native.coefficients()

    def lookup(self, token):
        return self.native.accept(*self.peers.exchange(self.native.lookup(token)))

    def linear(self, stage, values):
        result = self.native.accept(*self.peers.exchange(self.native.linear(stage, values)))
        if stage == self.head_stage:
            self.peers.phase = "decode"
        return result

    def finish(self):
        completions = self.peers.finish()
        acceptance = self.native.finish()
        accepted = read_json(acceptance)
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
                peers = _Peers(connections, timeout, cancelled, capacity=capacity)
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
