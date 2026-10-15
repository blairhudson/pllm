#!/usr/bin/env python3
"""Research-only paired HTTP stripe screen against the installed boundary path.

No checkpoint loading or native build. Snapshot arm explicitly duplicates weights;
sliced arm cannot start transmitting before the full native head finishes.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import time
import signal
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import anyio
import httpx
import numpy as np
import uvicorn
from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse

from pllm import _native
from pllm.deployment import WanConditions
from pllm.runtime.native import MaskedGEMM
from pllm.runtime.native_wan import READ_CHUNK_BYTES, SharedPacer
from pllm.runtime.private_boundary_head_codec import HEADER as CODEC_HEADER, HeadOutputCodec
from pllm.runtime.private_boundary_protocol import (
    BoundaryAdmission, BoundaryDescriptor, BoundaryError, ROLES, canonical,
    pack_frame, public_lookup_digest, unpack_frame,
)
from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
from pllm.runtime.private_boundary_worker import create_private_boundary_worker_app
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

LENGTH = struct.Struct("!I")
MODES = ("parallel", "sliced", "snapshots")
KEYS = ("research-boundary-a-credential", "research-boundary-b-credential")
LOCK_PATH = Path.home() / ".cache/pllm/frontier-probe.lock"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Fixture:
    def __init__(self, vocabulary=257, hidden=32, stripe_rows=64, seed=17,
                 snapshots=True, activation_bits=8):
        if (type(vocabulary) is not int or type(hidden) is not int
                or type(stripe_rows) is not int or not 1 <= vocabulary <= 262144
                or not 1 <= hidden <= 4096 or not 1 <= stripe_rows <= vocabulary
                or vocabulary * hidden > 160_000_000
                or (vocabulary + stripe_rows - 1) // stripe_rows > 1024):
            raise ValueError("invalid bounded research geometry")
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        self.weights = rng.integers(-128, 128, (vocabulary, hidden), dtype=np.int8)
        self.scales = rng.uniform(.001, .01, vocabulary).astype(np.float32)
        self.value = rng.normal(size=(1, hidden)).astype(np.float32)
        self.q = quantize_activation_per_row(self.value, bits=activation_bits)
        identity = sha(canonical({"generated": seed, "vocabulary": vocabulary, "hidden": hidden}))
        self.descriptor = BoundaryDescriptor(
            source_digest=identity, checkpoint_digest=identity, model_config_digest=identity,
            composition_digest=identity, schedule_digest=identity,
            head_weight_digest=sha(self.weights.tobytes()),
            head_scales_digest=sha(self.scales.tobytes()),
            lookup_digest=public_lookup_digest(
                ((memoryview(w), s.tobytes()) for w, s in zip(self.weights, self.scales)),
                vocabulary=vocabulary, hidden=hidden),
            vocabulary=vocabulary, hidden=hidden,
            signed_output_bound=hidden * 128 * ((1 << (activation_bits - 1)) - 1),
            activation_bits=activation_bits, head_dispatch="parallel",
            head_output_encoding="uniform_residues")
        self.admission = BoundaryAdmission(self.descriptor.digest, identity, 1, 1)
        self.geometry = {"schema": "pllm.research.head-stripes.v1",
                         "descriptor": self.descriptor.digest, "rows": stripe_rows,
                         "vocabulary": vocabulary, "order": "ascending-complete",
                         "codec": "installed-uniform-residues", "lookahead": 1}
        self.geometry_digest = sha(canonical(self.geometry))
        self.ranges = [(s, min(stripe_rows, vocabulary - s))
                       for s in range(0, vocabulary, stripe_rows)]
        self.codecs, self.admissions = [], []
        for index, (start, rows) in enumerate(self.ranges):
            # Retain original source/quantizer/ring; bind stripe coordinates in the
            # installed codec's descriptor commitment, not an unauthenticated header.
            binding = sha(canonical({"geometry": self.geometry_digest,
                                     "index": index, "start": start, "rows": rows}))
            descriptor = replace(self.descriptor, vocabulary=rows, composition_digest=binding)
            self.codecs.append(HeadOutputCodec(descriptor))
            self.admissions.append(replace(self.admission, descriptor_digest=descriptor.digest))
        engine = MaskedGEMM(threads=1)
        if not engine.native:
            raise RuntimeError("installed native extension required")
        self.full = engine.compile(self.weights)
        self.stripes = ([engine.compile(self.weights[s:s + n]) for s, n in self.ranges]
                        if snapshots else None)
        self.oracle = np.empty((1, vocabulary), np.int64)
        for start in range(0, vocabulary, 256):
            self.oracle[:, start:start + 256] = (
                self.q.values.astype(np.int64) @ self.weights[start:start + 256].astype(np.int64).T)
        self.logits = dequantize_matmul(self.oracle, self.q.scales, self.scales,
                                       output_shape=(1, vocabulary))
        self.setup = {"wall_seconds": time.perf_counter() - started,
                      "generated_weight_bytes": self.weights.nbytes,
                      "full_native_weight_bytes": self.full.weight_bytes,
                      "additional_snapshot_weight_bytes": self.weights.nbytes if snapshots else 0,
                      "codec_layout_bytes": sum(len(c.widths) for c in self.codecs),
                      "scope": "generated source, independent chunked i64 oracle; not model admission"}

    def codec_args(self, index, sid, sequence=0):
        return dict(admission=self.admissions[index], paired_session=sid, epoch=0, sequence=sequence)


@dataclass
class Owner:
    attempted: bool = False
    retired: bool = False
    native_calls: int = 0
    drained_native_calls: int = 0
    peak_output_slots: int = 0
    peak_output_bytes: int = 0
    peak_pending_native: int = 0
    replay_rejected: bool = False


class Reader:
    """One bounded HTTP read chunk plus one bounded application frame."""
    def __init__(self, response, ledger, chunk_bytes, timing, started):
        self.iterator = response.aiter_bytes(chunk_size=chunk_bytes).__aiter__()
        self.buffer = bytearray()
        self.ledger = ledger
        self.timing, self.started = timing, started

    async def exact(self, size):
        while len(self.buffer) < size:
            try:
                chunk = await anext(self.iterator)
            except StopAsyncIteration as exc:
                raise BoundaryError("partial stripe reply") from exc
            self.ledger["observed_reply_body_bytes"] += len(chunk)
            self.timing.setdefault("first_client_body_seconds", time.perf_counter() - self.started)
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    async def frame(self, maximum):
        size = LENGTH.unpack(await self.exact(LENGTH.size))[0]
        if not 1 <= size <= maximum:
            raise BoundaryError("stripe frame exceeds fixed geometry")
        return await self.exact(size)

    async def end(self):
        if self.buffer:
            raise BoundaryError("trailing stripe reply")
        try:
            chunk = await anext(self.iterator)
        except StopAsyncIteration:
            return
        self.ledger["observed_reply_body_bytes"] += len(chunk)
        raise BoundaryError("trailing stripe reply")


async def run_case(fixture, mode, *, download_mbps=100., upload_mbps=40.,
                   fault=None, cancel_after=None, native_hook=None,
                   read_chunk_bytes=READ_CHUNK_BYTES):
    """One admitted pair. Loopback HTTP/1.1, actual TCP-payload shared pacing.

    Candidate route is research-only: installed admission validates full source;
    separate one-shot owner reserves sequence zero. It is NOT an installed worker
    slot implementation. Both owners and native tasks drain before return.
    """
    if mode not in MODES or (mode == "snapshots" and fixture.stripes is None):
        raise ValueError("unavailable research arm")
    if type(read_chunk_bytes) is not int or not 1 <= read_chunk_bytes <= 65536:
        raise ValueError("invalid bounded HTTP read chunk")
    d = fixture.descriptor
    pacer = SharedPacer(WanConditions(download_mbps=download_mbps, upload_mbps=upload_mbps),
                        ("client", *ROLES))
    owners = [Owner(), Owner()]
    ledger = [{"attempted_request_body_bytes": 0, "generated_reply_body_bytes": 0,
               "observed_reply_body_bytes": 0} for _ in ROLES]
    servers, server_tasks, sockets, proxies = [], [], [], []
    proxy_tasks, writers, native_tasks = set(), set(), set()
    sid = None
    timestamps = [{"native_seconds": 0., "encode_seconds": 0.} for _ in ROLES]
    clients = []
    transport = None
    error = None
    result = None
    replay_status = []

    async def proxy(reader, writer, index, target):
        task = asyncio.current_task()
        proxy_tasks.add(task)
        peer = None
        copies = []
        writers.add(writer)
        try:
            remote, peer = await asyncio.open_connection("127.0.0.1", target,
                                                         limit=2 * READ_CHUNK_BYTES)
            writers.add(peer)
            for stream in (writer, peer):
                stream.transport.set_write_buffer_limits(high=2 * READ_CHUNK_BYTES,
                                                         low=READ_CHUNK_BYTES)

            async def copy(origin, output, source, destination):
                while chunk := await origin.read(READ_CHUNK_BYTES):
                    await pacer.reserve(source, destination, len(chunk))
                    output.write(chunk)
                    await output.drain()
            copies = [asyncio.create_task(copy(reader, peer, "client", ROLES[index])),
                      asyncio.create_task(copy(remote, writer, ROLES[index], "client"))]
            await asyncio.wait(copies, return_when=asyncio.FIRST_COMPLETED)
        except (ConnectionError, OSError):
            pass
        finally:
            for task_copy in copies:
                task_copy.cancel()
            await asyncio.gather(*copies, return_exceptions=True)
            for stream in (writer, peer):
                if stream is not None:
                    stream.close()
                    writers.discard(stream)
            proxy_tasks.discard(task)

    def attach(app, index):
        owner, role = owners[index], ROLES[index]

        @app.post("/research/head-stripes")
        async def stripes(request: Request):
            credential = request.headers.get("authorization", "")
            if not credential.isascii() or not secrets.compare_digest(credential, "Bearer " + KEYS[index]):
                raise HTTPException(401)
            if owner.attempted:
                owner.replay_rejected = True
                raise HTTPException(409)
            owner.attempted = True  # Malformed first attempt also burns this owner.
            try:
                if request.headers.get("x-stripe-geometry") != fixture.geometry_digest:
                    raise BoundaryError("unadmitted stripe geometry")
                raw = bytearray()
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > d.hidden * 4 + 1024:
                        raise BoundaryError("oversized stripe request")
                    raw.extend(chunk)
                row = unpack_frame(bytes(raw))
                if ((row["role"], row["session"], row["epoch"], row["sequence"], row["operation"])
                        != (role, sid, 0, 0, "head") or len(row["frames"]) != 1
                        or len(row["frames"][0]) != d.hidden * 4):
                    raise BoundaryError("stripe request context differs")
                share = np.frombuffer(row["frames"][0], "<u4").reshape(1, d.hidden)
            except (ValueError, TypeError):
                owner.retired = True
                raise HTTPException(409) from None

            async def stream():
                pending = None
                full = None

                def submit(matrix):
                    owner.native_calls += 1
                    owner.peak_pending_native = 1
                    owner.peak_output_slots = max(owner.peak_output_slots, 1)
                    owner.peak_output_bytes = max(owner.peak_output_bytes, matrix.shape[0] * 4)

                    def evaluate():
                        started = time.perf_counter()
                        if native_hook is not None:
                            native_hook(index)
                        value = matrix.wrap32(share)
                        timestamps[index]["native_seconds"] += time.perf_counter() - started
                        return value
                    task = asyncio.create_task(asyncio.to_thread(evaluate))
                    native_tasks.add(task)
                    return task

                try:
                    if mode == "sliced":
                        pending = submit(fixture.full)
                        full = await asyncio.shield(pending)
                        native_tasks.discard(pending)
                        pending = None
                    for number, (start, rows) in enumerate(fixture.ranges):
                        if full is not None:
                            output = full[:, start:start + rows]
                            slots, size = 1, full.nbytes
                        else:
                            if pending is None:
                                pending = submit(fixture.stripes[number])
                            output = await asyncio.shield(pending)
                            native_tasks.discard(pending)
                            pending = None
                            slots, size = 1, output.nbytes
                            if number + 1 < len(fixture.ranges):
                                pending = submit(fixture.stripes[number + 1])
                                slots += 1
                                size += fixture.ranges[number + 1][1] * 4
                        owner.peak_output_slots = max(owner.peak_output_slots, slots)
                        owner.peak_output_bytes = max(owner.peak_output_bytes, size)
                        args = fixture.codec_args(number, sid)
                        started = time.perf_counter()
                        encoded = fixture.codecs[number].encode(output.astype("<u4", copy=False).tobytes(),
                                                                role=role, **args)
                        body = pack_frame(role=role, paired_session=sid, epoch=0,
                                          sequence=0, operation="head", frames=[encoded], response=True)
                        timestamps[index]["encode_seconds"] += time.perf_counter() - started
                        if index == 1:
                            if fault == "context":
                                body = pack_frame(role=role, paired_session="f" * 32, epoch=0,
                                                  sequence=0, operation="head", frames=[encoded], response=True)
                            if fault == "reorder":
                                other = (number + 1) % len(fixture.ranges)
                                if fixture.ranges[other][1] == rows:
                                    encoded = fixture.codecs[other].encode(output.tobytes(), role=role,
                                                                         **fixture.codec_args(other, sid))
                                    body = pack_frame(role=role, paired_session=sid, epoch=0,
                                                      sequence=0, operation="head", frames=[encoded], response=True)
                            if fault == "malformed":
                                body = b"bad"
                        raw_reply = LENGTH.pack(len(body)) + body
                        if index == 1 and fault == "truncate":
                            raw_reply = raw_reply[:-1]
                        ledger[index]["generated_reply_body_bytes"] += len(raw_reply)
                        timestamps[index].setdefault("first_yield_seconds", time.perf_counter() - started_exchange)
                        yield raw_reply
                        del output, encoded, body, raw_reply
                        if index == 1 and fault in ("truncate", "partial"):
                            return
                    if index == 1 and fault == "trailing":
                        ledger[index]["generated_reply_body_bytes"] += 1
                        yield b"!"
                finally:
                    if pending is not None:
                        # Starlette cancels its streaming task on disconnect. Shield
                        # the AnyIO scope too: asyncio.shield alone is insufficient.
                        try:
                            with anyio.CancelScope(shield=True):
                                await asyncio.shield(pending)
                        except Exception:
                            pass
                        finally:
                            native_tasks.discard(pending)
                            owner.drained_native_calls += 1
                    owner.retired = True
            return StreamingResponse(stream(), media_type="application/octet-stream")

    setup_started = time.perf_counter()
    try:
        origins = []
        for index, role in enumerate(ROLES):
            def admit(actual, requested):
                if actual != d or requested != fixture.admission:
                    raise BoundaryError("generated trusted admission differs")
                return fixture.admission
            app = create_private_boundary_worker_app(
                descriptor=d, head=fixture.full, head_scales=fixture.scales, role_id=role,
                api_key=KEYS[index], peer_credential_digest=sha(KEYS[1 - index].encode()),
                admit=admit, max_sessions=1)
            attach(app, index)

            async def observed(scope, receive, send, inner=app, i=index):
                online = (scope["type"] == "http" and
                          (scope["path"].endswith("/execute") or scope["path"] == "/research/head-stripes"))

                status = None

                async def counted_send(message):
                    nonlocal status
                    if message["type"] == "http.response.start":
                        status = message["status"]
                    if online and status == 200 and message["type"] == "http.response.body" and message.get("body"):
                        timestamps[i].setdefault("first_asgi_body_seconds", time.perf_counter() - started_exchange)
                        timestamps[i]["asgi_response_body_bytes"] = (
                            timestamps[i].get("asgi_response_body_bytes", 0) + len(message["body"]))
                    await send(message)
                await inner(scope, receive, counted_send)

            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(16)
            sock.setblocking(False)
            sockets.append(sock)
            server = uvicorn.Server(uvicorn.Config(observed, log_level="critical", lifespan="on",
                                                   access_log=False, loop="asyncio"))
            servers.append(server)
            server_tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
            for _ in range(500):
                if server.started:
                    break
                if server_tasks[-1].done():
                    await server_tasks[-1]
                await asyncio.sleep(.002)
            else:
                raise RuntimeError("HTTP worker startup timed out")
            proxy_server = await asyncio.start_server(
                lambda r, w, i=index, p=sock.getsockname()[1]: proxy(r, w, i, p),
                "127.0.0.1", 0, limit=2 * READ_CHUNK_BYTES)
            proxies.append(proxy_server)
            origins.append("http://127.0.0.1:" + str(proxy_server.sockets[0].getsockname()[1]))
        transport = PrivateBoundaryTransport(worker_a=origins[0], worker_b=origins[1],
            api_key_a=KEYS[0], api_key_b=KEYS[1], descriptor=d, head_scales=fixture.scales)
        await asyncio.to_thread(transport.open, fixture.admission)
        sid = transport.session_id
        if mode != "parallel":
            clients.extend(httpx.AsyncClient(base_url=origin, timeout=5., trust_env=False) for origin in origins)
        setup_seconds = time.perf_counter() - setup_started
        wire_before = dict(pacer.links)
        started_exchange, cpu = time.perf_counter(), time.process_time()

        async def candidate():
            right = np.frombuffer(secrets.token_bytes(d.hidden * 4), "<u4").reshape(1, -1)
            quantized = quantize_activation_per_row(fixture.value, bits=d.activation_bits)
            clear = (quantized.values.astype(np.int64) % (1 << 32)).astype("<u4")
            left = (clear - right).astype("<u4")
            requests = [pack_frame(role=role, paired_session=sid, epoch=0, sequence=0,
                                   operation="head", frames=[share.tobytes()])
                        for role, share in zip(ROLES, (left, right))]
            replies = []
            try:
                async def send(index):
                    payload = requests[index]
                    geometry = fixture.geometry_digest
                    key = KEYS[index]
                    if index == 1:
                        if fault == "request_sequence":
                            payload = pack_frame(role=ROLES[index], paired_session=sid, epoch=0,
                                                 sequence=1, operation="head", frames=[right.tobytes()])
                        if fault == "request_malformed":
                            payload = b"malformed"
                        if fault == "request_oversized":
                            payload = bytes(d.hidden * 4 + 1025)
                        if fault == "request_geometry":
                            geometry = "0" * 64
                        if fault == "credential":
                            key = "wrong-research-key"
                    ledger[index]["attempted_request_body_bytes"] += len(payload)
                    return await clients[index].send(clients[index].build_request("POST", "/research/head-stripes",
                        content=payload, headers={"authorization": "Bearer " + key,
                        "x-stripe-geometry": geometry, "accept-encoding": "identity"}), stream=True)
                opened = await asyncio.gather(*(send(i) for i in range(2)), return_exceptions=True)
                replies.extend(r for r in opened if isinstance(r, httpx.Response))
                if any(isinstance(r, BaseException) for r in opened):
                    raise BoundaryError("paired HTTP open failed")
                if any(r.status_code != 200 or r.headers.get("content-encoding", "identity") != "identity"
                       for r in replies):
                    raise BoundaryError("stripe HTTP response differs")
                readers = [Reader(r, ledger[i], read_chunk_bytes, timestamps[i], started_exchange)
                           for i, r in enumerate(replies)]
                centered = np.empty((1, d.vocabulary), np.int64)
                for number, (start, rows) in enumerate(fixture.ranges):
                    sizes = [len(pack_frame(role=role, paired_session=sid, epoch=0, sequence=0,
                              operation="head", frames=[bytes(fixture.codecs[number].payload_bytes + CODEC_HEADER.size)],
                              response=True)) for role in ROLES]
                    raw_pair = await asyncio.gather(*(reader.frame(size) for reader, size in zip(readers, sizes)))
                    if any(len(raw) != size for raw, size in zip(raw_pair, sizes)):
                        raise BoundaryError("stripe frame length differs from fixed geometry")
                    frames = []
                    for role, raw in zip(ROLES, raw_pair):
                        row = unpack_frame(raw, response=True)
                        if ((row["role"], row["session"], row["epoch"], row["sequence"], row["operation"])
                                != (role, sid, 0, 0, "head") or len(row["frames"]) != 1):
                            raise BoundaryError("stripe reply context/order differs")
                        frames.append(row["frames"][0])
                    centered[:, start:start + rows] = fixture.codecs[number].decode(
                        *frames, **fixture.codec_args(number, sid))
                    if number == 0:
                        for timing in timestamps:
                            timing["first_paired_decoded_stripe_seconds"] = time.perf_counter() - started_exchange
                await asyncio.gather(*(reader.end() for reader in readers))
                np.testing.assert_array_equal(centered, fixture.oracle)
                return dequantize_matmul(centered, quantized.scales, fixture.scales,
                                          output_shape=(1, d.vocabulary))
            finally:
                await asyncio.gather(*(r.aclose() for r in replies))
                left.fill(0)

        operation = asyncio.create_task(asyncio.to_thread(transport.linear, d.head_stage_id, fixture.value)
                                        if mode == "parallel" else candidate())
        if cancel_after is not None and mode != "parallel":
            await asyncio.sleep(cancel_after)
            operation.cancel()
        try:
            result = await operation
            np.testing.assert_array_equal(result, fixture.logits)
        except (Exception, asyncio.CancelledError) as exc:
            error = type(exc).__name__ + ": " + str(exc)
        exchange_seconds = time.perf_counter() - started_exchange
        exchange_cpu = time.process_time() - cpu
        wire_online = {link: value - wire_before.get(link, 0) for link, value in pacer.links.items()}
        if mode != "parallel":
            for i, client in enumerate(clients):
                replay = await client.post("/research/head-stripes", content=b"replay",
                                           headers={"authorization": "Bearer " + KEYS[i]})
                replay_status.append(replay.status_code)
        counters = transport.operation_counters
        if mode == "parallel":
            ledger = [{"installed_online_upload_bytes": transport.role_link_bodies[r]["online_upload_bytes"],
                       "installed_online_download_bytes": transport.role_link_bodies[r]["online_download_bytes"]}
                      for r in ROLES]
    finally:
        if transport is not None:
            await asyncio.to_thread(transport.close)
        await asyncio.gather(*(client.aclose() for client in clients))
        for proxy_server in proxies:
            proxy_server.close()
            await proxy_server.wait_closed()
        for writer in list(writers):
            writer.close()
        for task in list(proxy_tasks):
            task.cancel()
        await asyncio.gather(*list(proxy_tasks), return_exceptions=True)
        for server in servers:
            server.should_exit = True
        await asyncio.gather(*server_tasks)
        await asyncio.gather(*list(native_tasks), return_exceptions=True)
        for sock in sockets:
            sock.close()
    return {"mode": mode, "success": result is not None and error is None, "error": error,
            "exchange_seconds": exchange_seconds, "exchange_cpu_seconds": exchange_cpu,
            "setup_seconds": setup_seconds, "geometry": fixture.geometry,
            "candidate_read_chunk_bytes": read_chunk_bytes,
            "owners": [asdict(o) for o in owners] if mode != "parallel" else None,
            "baseline_ownership": "installed worker/session and paired executor" if mode == "parallel" else None,
            "remaining_native_tasks": sum(not t.done() for t in native_tasks),
            "online_coverage_complete": error is None,
            "replay_status": replay_status, "per_role_bodies": dict(zip(ROLES, ledger)),
            "installed_operation_counters": counters, "timing_breakdown": timestamps,
            "candidate_operation_counters": ({role: {"head_calls_reserved": int(o.attempted),
                "sequence_reserved": 0 if o.attempted else None, "complete_stripes_expected": len(fixture.ranges)}
                for role, o in zip(ROLES, owners)} if mode != "parallel" else None),
            "online_tcp_payload_bytes": wire_online, "all_phases_network": pacer.snapshot(),
            "logits_sha256": sha(result.tobytes()) if result is not None else None,
            "accounting": "TCP payload includes actual HTTP headers/chunk framing; body counters separate; no TLS/IP/ACK",
            "ownership": ("full native output before first stripe" if mode == "sliced" else
                          "extra immutable stripe snapshots; current plus one next native output" if mode == "snapshots"
                          else "installed parallel full uniform-residue replies")}


def arm_orders(modes, repeats):
    orders = list(itertools.permutations(modes))
    return [list(orders[r % len(orders)]) for r in range(repeats)]


async def main(args, emit=None):
    if len(args.modes) != len(set(args.modes)):
        raise ValueError("delivery arms must be distinct")
    fixture = Fixture(args.vocabulary, args.hidden, args.stripe_rows, args.seed,
                      snapshots="snapshots" in args.modes, activation_bits=args.activation_bits)
    reports = []
    orders = arm_orders(args.modes, args.repeats)
    if emit:
        emit({"kind": "fixture", "setup": fixture.setup, "descriptor_sha256": fixture.descriptor.digest,
              "geometry_sha256": fixture.geometry_digest, "source_sha256": fixture.descriptor.source_digest,
              "weight_sha256": fixture.descriptor.head_weight_digest,
              "scales_sha256": fixture.descriptor.head_scales_digest})
    for repeat, order in enumerate(orders):
        for position, mode in enumerate(order):
            report = await run_case(fixture, mode, download_mbps=args.download_mbps,
                                   upload_mbps=args.upload_mbps, read_chunk_bytes=args.read_chunk_bytes)
            report.update(repeat=repeat, position=position)
            reports.append(report)
            if emit:
                emit({"kind": "run", "report": report})
            if not report["success"]:
                raise RunRejected("correctness_or_exchange_failure")
    binary = Path(_native.__file__)
    return {"schema": "pllm.research.head-residue-stripes.v1", "setup": fixture.setup,
            "source_sha256": fixture.descriptor.source_digest,
            "descriptor_sha256": fixture.descriptor.digest,
            "codec_geometry_sha256": fixture.geometry_digest,
            "weight_sha256": fixture.descriptor.head_weight_digest,
            "scales_sha256": fixture.descriptor.head_scales_digest,
            "native_filename": binary.name, "native_sha256": sha(binary.read_bytes()),
            "installed_matrix_methods": [m for m in dir(_native.Matrix) if not m.startswith("_")],
            "installed_executor_methods": [m for m in dir(_native.Executor) if not m.startswith("_")],
            "repeat_orders": orders, "runs": reports, "production_ready": False}


class RunRejected(RuntimeError):
    pass


def sample_host():
    from pllm.runtime.benchmark_memory import host_memory
    return host_memory()


def allocation_allowance(args):
    # Source, snapshots, native compile serialization, outputs, full oracles,
    # codec/HTTP owners and interpreter slack. Conservative allowance, not RSS.
    return max(1 << 30, 6 * args.vocabulary * args.hidden + 512 * (1 << 20))


class HostGuard:
    def __init__(self, allowance):
        self.initial = sample_host()
        self.reserve = max(4 << 30, self.initial.reserve)
        self.allowance = allowance
        self.low_swap = self.initial.swap_used
        self.minimum_available = self.initial.available
        self.maximum_swap_growth = 0
        if self.initial.available < self.reserve + allowance:
            raise RunRejected("host_headroom_rejected")
        if self.initial.disk_free < allowance:
            raise RunRejected("host_disk_rejected")

    def check(self, *, admission=False):
        host = sample_host()
        self.minimum_available = min(self.minimum_available, host.available)
        self.low_swap = min(self.low_swap, host.swap_used)
        self.maximum_swap_growth = max(self.maximum_swap_growth, host.swap_used - self.low_swap)
        if self.maximum_swap_growth:
            raise RunRejected("new_swap_growth")
        if host.available < self.reserve:
            raise RunRejected("runtime_reserve_rejected")
        if admission and host.available < self.reserve + self.allowance:
            raise RunRejected("host_headroom_rejected")

    def report(self):
        return {"host": asdict(self.initial), "reserve_bytes": self.reserve,
                "allocation_allowance_bytes": self.allowance,
                "minimum_available_bytes": self.minimum_available,
                "maximum_swap_growth_bytes": self.maximum_swap_growth,
                "sample_interval_seconds": .1, "policy": "zero observed new swap, full reserve retained"}


def provenance():
    from pllm.metrics import BuildRef
    from pllm.runtime import (native, native_wan, private_boundary_head_codec,
                              private_boundary_protocol, private_boundary_transport,
                              private_boundary_worker, quantization)
    modules = (native, native_wan, private_boundary_head_codec, private_boundary_protocol,
               private_boundary_transport, private_boundary_worker, quantization)
    sources = {m.__name__: sha(Path(m.__file__).read_bytes()) for m in modules}
    root = Path(__file__).resolve().parents[1]
    for relative in ("scripts/probe_boundary_head_residue_stripes.py",
                     "tests/test_boundary_head_residue_stripes.py",
                     "docs/research/boundary-head-residue-stripes.md"):
        sources[relative] = sha((root / relative).read_bytes())
    captured = BuildRef.capture().to_spec()
    build = {"expected_digest": captured["expected_digest"],
             "interpreter_sha256": sha(Path(sys.executable).resolve().read_bytes())}
    return {"build": build, "sources": sources, "sources_sha256": sha(canonical(sources)),
            "native_sha256": sha(Path(_native.__file__).read_bytes()),
            "python_version": sys.version, "python_executable_name": Path(sys.executable).name}


def write_owned(stream, value):
    stream.seek(0)
    json.dump(value, stream, indent=2)
    stream.write("\n")
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--stripe-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--repeats", type=int, choices=range(1, 25), default=1)
    parser.add_argument("--download-mbps", type=float, default=100.)
    parser.add_argument("--upload-mbps", type=float, default=40.)
    parser.add_argument("--read-chunk-bytes", type=int, default=READ_CHUNK_BYTES)
    parser.add_argument("--sampled-qwen", action="store_true",
                        help="freeze generated 151936x896, rows8192, six orders, all three arms")
    parser.add_argument("--sampled-worker", type=int, choices=(1, 2, 3))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def cli(argv=None):
    from filelock import FileLock, Timeout
    args = parser().parse_args(argv)
    if args.sampled_qwen:
        if args.sampled_worker is None or args.read_chunk_bytes not in (8192, 1024):
            parser().error("sampled Qwen requires worker 1..3 and read chunk 8192 or 1024")
        args.vocabulary, args.hidden, args.stripe_rows = 151936, 896, 8192
        args.repeats, args.modes, args.seed, args.activation_bits = 6, list(MODES), 17, 8
        args.download_mbps, args.upload_mbps = 100., 40.
    if len(args.modes) != len(set(args.modes)):
        parser().error("delivery arms must be distinct")
    output = args.output
    paths = [output, Path(str(output) + ".progress.json"), Path(str(output) + ".runs.jsonl")]
    # Never unlink or replace a pre-existing path, including dangling symlinks.
    if any(os.path.lexists(path) for path in paths):
        print(json.dumps({"status": "rejected", "failure_code": "output_exists"}))
        return 1
    reserved = []
    process = None
    guard = None
    plan = None

    @contextmanager
    def worker_lifetime():
        try:
            yield
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def terminate(_signum, _frame):
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    progress = {"schema": "pllm.research.head-residue-stripes-progress.v1",
                "status": "reserved", "measurement_complete": False}
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        for path in paths:
            reserved.append(path.open("x+"))
        write_owned(reserved[0], progress)
        write_owned(reserved[1], progress)
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(LOCK_PATH), timeout=0), worker_lifetime():
            # Admission precedes provenance subprocesses and all Fixture allocation.
            guard = HostGuard(allocation_allowance(args))
            frozen = provenance()
            geometry = {k: getattr(args, k) for k in ("vocabulary", "hidden", "stripe_rows", "seed",
                "activation_bits", "modes", "repeats", "read_chunk_bytes", "download_mbps", "upload_mbps")}
            plan = {"geometry": geometry, "geometry_sha256": sha(canonical(geometry)),
                    "repeat_orders": arm_orders(args.modes, args.repeats),
                    "sampled_worker": args.sampled_worker, "sampled_qwen": args.sampled_qwen,
                    "provenance": frozen, "host_admission": guard.report(),
                    "scope": "fresh generated-source worker; no model; shared matrices/executor; allocation extras charged"}
            progress.update(status="running", planned_runs=args.repeats * len(args.modes),
                            geometry_sha256=plan["geometry_sha256"], sources_sha256=frozen["sources_sha256"])
            write_owned(reserved[0], {"plan": plan, "progress": progress})
            write_owned(reserved[1], progress)
            guard.check(admission=True)
            env = os.environ.copy()
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
                env[name] = "4"
            token = secrets.token_hex(16)
            env.update(PLLM_RESIDUE_STRIPES_WORKER=token, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false")
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker"],
                stdin=subprocess.PIPE, stdout=reserved[2], stderr=subprocess.DEVNULL, text=True, env=env)
            process.stdin.write(json.dumps({"token": token, "args": geometry}))
            process.stdin.close()
            started = time.monotonic()
            while process.poll() is None:
                guard.check()
                if time.monotonic() - started > 1800:
                    raise RunRejected("worker_timeout")
                elapsed = int(time.monotonic() - started)
                if elapsed != progress.get("elapsed_seconds"):
                    progress["elapsed_seconds"] = elapsed
                    write_owned(reserved[1], progress)
                time.sleep(.1)
            guard.check()
            if process.returncode:
                raise RunRejected("worker_failed")
            if provenance() != frozen:
                raise RunRejected("source_or_build_changed")
            guard.check()
            reserved[2].seek(0)
            rows = [json.loads(line) for line in reserved[2] if line.strip()]
            if not rows or rows[-1].get("kind") != "complete":
                raise RunRejected("incomplete_worker_report")
            report = rows[-1]["report"]
            expected = [(repeat, position, mode) for repeat, order in enumerate(plan["repeat_orders"])
                        for position, mode in enumerate(order)]
            actual = [(r["repeat"], r["position"], r["mode"]) for r in report["runs"]]
            emitted = [row["report"] for row in rows if row.get("kind") == "run"]
            if (actual != expected or report["runs"] != emitted
                    or report["repeat_orders"] != plan["repeat_orders"]
                    or not all(r["success"] for r in report["runs"])
                    or len({r["logits_sha256"] for r in report["runs"]}) != 1
                    or report["native_sha256"] != frozen["native_sha256"]):
                raise RunRejected("worker_report_audit_failed")
            progress.update(status="completed", measurement_complete=True, completed_runs=len(report["runs"]))
            report.update(plan=plan, host_guard=guard.report(), progress=progress)
            write_owned(reserved[0], report)
            write_owned(reserved[1], progress)
            print(json.dumps(progress))
            return 0
    except BaseException as exc:
        code = ("frontier_cohort_lock_busy" if isinstance(exc, Timeout) else
                str(exc) if isinstance(exc, RunRejected) else
                "output_reservation_race" if isinstance(exc, FileExistsError) else
                "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "preflight_or_execution_error")
        progress.update(status="rejected", failure_code=code)
        try:
            if len(reserved) >= 3:
                reserved[2].seek(0)
                progress["retained_completed_runs"] = sum('"kind": "run"' in line for line in reserved[2])
            if reserved:
                # Own open descriptor only: never overwrite another process's path.
                write_owned(reserved[0], {"plan": plan, "progress": progress,
                                          "host_guard": guard.report() if guard else None})
            if len(reserved) >= 2:
                write_owned(reserved[1], progress)
        except OSError:
            progress["retention_error"] = "output_io_error"
        print(json.dumps(progress))  # No exception strings, credentials or paths.
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        for stream in reserved:
            try:
                stream.close()
            except OSError:
                pass


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        payload = json.load(sys.stdin)
        if not secrets.compare_digest(payload.get("token", ""), os.environ.get("PLLM_RESIDUE_STRIPES_WORKER", "!")):
            raise SystemExit(1)
        def emit(row):
            print(json.dumps(row), flush=True)
        report = asyncio.run(main(argparse.Namespace(**payload["args"]), emit=emit))
        emit({"kind": "complete", "report": report})
    else:
        raise SystemExit(cli())
