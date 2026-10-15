#!/usr/bin/env python3
"""Research-only native full-head delivery fixture; not a serving protocol."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import secrets
import struct
import time
from dataclasses import dataclass

import numpy as np

from pllm.deployment import WanConditions
from pllm.runtime.native import MaskedGEMM
from pllm.runtime.native_wan import READ_CHUNK_BYTES, SharedPacer
from pllm.runtime.private_boundary_protocol import (
    BoundaryError, ROLES, pack_frame, unpack_frame,
)
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row

# Public research envelope: magic, stripe index, start, rows, total rows, body size.
HEADER = struct.Struct("!4sIIIII")
LENGTH = struct.Struct("!I")
MODES = ("serial", "parallel", "striped", "pipelined")


def arm_orders(repeats, offset=0):
    """Four-arm Williams design: each position and directed predecessor balanced."""
    design = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
    return [[MODES[i] for i in design[(offset + r) % 4]] for r in range(repeats)]


@dataclass
class Owner:
    attempted: bool = False
    busy: bool = False
    retired: bool = False
    native_calls: int = 0
    drained_native_calls: int = 0
    max_pending_native_tasks: int = 0
    peak_reserved_output_slots: int = 0
    peak_reserved_output_bytes: int = 0


class Fixture:
    """Deterministic generated source and native arithmetic, no model/role admission."""

    def __init__(self, vocabulary=257, hidden=32, stripe_rows=64, seed=17):
        if not (1 <= vocabulary <= 262144 and 1 <= hidden <= 4096
                and 1 <= stripe_rows <= vocabulary):
            raise ValueError("invalid bounded geometry")
        # Prevent accidental heavy checkpoint-sized native setup while agents coexist.
        if vocabulary * hidden > 160_000_000:
            raise ValueError("fixture matrix exceeds research bound")
        wall, cpu = time.perf_counter(), time.process_time()
        self.vocabulary, self.hidden, self.stripe_rows = vocabulary, hidden, stripe_rows
        rng = np.random.default_rng(seed)
        self.weights = rng.integers(-128, 128, (vocabulary, hidden), dtype=np.int8)
        self.scales = rng.uniform(.001, .01, vocabulary).astype(np.float32)
        self.value = rng.normal(size=(1, hidden)).astype(np.float32)
        self.quantized = quantize_activation_per_row(self.value, bits=8)
        self.clear = (self.quantized.values.astype(np.int64) % (1 << 32)).astype("<u4")
        engine = MaskedGEMM(threads=1)
        self.full = engine.compile(self.weights)
        self.stripes = [(start, engine.compile(self.weights[start:start + stripe_rows]))
                        for start in range(0, vocabulary, stripe_rows)]
        # Independent dense integer oracle, plus full native oracle.
        self.oracle = np.empty((1, vocabulary), np.int64)
        for start in range(0, vocabulary, 256):
            self.oracle[:, start:start + 256] = (
                self.quantized.values.astype(np.int64)
                @ self.weights[start:start + 256].astype(np.int64).T)
        native = self.full.wrap32(self.clear)
        np.testing.assert_array_equal(native, (self.oracle % (1 << 32)).astype(np.uint32))
        self.logits = dequantize_matmul(self.oracle, self.quantized.scales, self.scales,
                                       output_shape=(1, vocabulary))
        self.setup = {"wall_seconds": time.perf_counter() - wall,
                      "cpu_seconds": time.process_time() - cpu,
                      "delivered_network_bytes": 0,
                      "generated_weight_bytes": self.weights.nbytes,
                      "note": "local generation, full + stripe compilation and oracles; not cold distribution"}


async def paced_write(writer, raw, pacer, source, destination):
    """Bound each reservation and write; no per-stripe task queue."""
    for start in range(0, len(raw), READ_CHUNK_BYTES):
        chunk = memoryview(raw)[start:start + READ_CHUNK_BYTES]
        await pacer.reserve(source, destination, len(chunk))
        writer.write(chunk)
        await writer.drain()


async def read_reply(reader, *, role, session, stripe, start, rows, vocabulary):
    header = await reader.readexactly(HEADER.size)
    magic, actual_stripe, actual_start, actual_rows, total, size = HEADER.unpack(header)
    if (magic, actual_stripe, actual_start, actual_rows, total) != (
            b"BHD1", stripe, start, rows, vocabulary):
        raise BoundaryError("research reply stripe ownership/order differs")
    if not rows * 4 <= size <= rows * 4 + 1024:
        raise BoundaryError("research reply body size differs")
    raw = await reader.readexactly(size)
    row = unpack_frame(raw, response=True)
    if (row["role"], row["session"], row["epoch"], row["sequence"], row["operation"]) != (
            role, session, 0, 0, "head") or len(row["frames"][0]) != rows * 4:
        raise BoundaryError("research reply context/width differs")
    return row["frames"][0]


async def run_case(fixture, mode, *, download_mbps=100., upload_mbps=40.,
                   fault=None, cancel_after=None, timeout_seconds=30.):
    """One head, exactly two worker handlers. Failure never returns partial logits.

    Test-only faults affect boundary_b: fail, truncate, malformed, reorder, context.
    Cancellation drains any active native call before owner retirement.
    """
    if mode not in MODES:
        raise ValueError("unknown delivery mode")
    if not 0 < timeout_seconds <= 120:
        raise ValueError("timeout must be in (0, 120]")
    setup_started, setup_cpu = time.perf_counter(), time.process_time()
    pacer = SharedPacer(WanConditions(download_mbps=download_mbps,
                                     upload_mbps=upload_mbps), ("client", *ROLES))
    owners = [Owner(), Owner()]
    handlers, client_writers, worker_writers = set(), set(), set()
    session = secrets.token_hex(16)
    right = np.frombuffer(secrets.token_bytes(fixture.hidden * 4), "<u4").reshape(1, -1)
    left = (fixture.clear - right).astype("<u4")
    striped = mode in ("striped", "pipelined")
    matrices = fixture.stripes if striped else [(0, fixture.full)]
    ledger = {role: {"request_body_bytes": 0, "reply_body_bytes": 0,
                     "request_envelope_bytes": 0, "reply_envelope_bytes": 0}
              for role in ROLES}
    started, cpu = setup_started, setup_cpu
    transport_setup = None

    async def worker(reader, writer, index):
        task = asyncio.current_task()
        handlers.add(task)
        worker_writers.add(writer)
        owner, role = owners[index], ROLES[index]
        owner.attempted = True
        pending_native = None

        def submit(matrix, rows):
            if owner.busy:
                raise RuntimeError("research owner already has pending native work")
            owner.busy = True
            owner.native_calls += 1
            owner.max_pending_native_tasks = 1
            owner.peak_reserved_output_slots = max(owner.peak_reserved_output_slots, 1)
            owner.peak_reserved_output_bytes = max(owner.peak_reserved_output_bytes, rows * 4)
            return asyncio.create_task(asyncio.to_thread(matrix.wrap32, share))

        writer.transport.set_write_buffer_limits(high=2 * READ_CHUNK_BYTES,
                                                 low=READ_CHUNK_BYTES)
        try:
            size = LENGTH.unpack(await reader.readexactly(LENGTH.size))[0]
            if not fixture.hidden * 4 <= size <= fixture.hidden * 4 + 1024:
                raise BoundaryError("research request body size differs")
            row = unpack_frame(await reader.readexactly(size))
            if (row["role"], row["session"], row["epoch"], row["sequence"], row["operation"]) != (
                    role, session, 0, 0, "head") or len(row["frames"][0]) != fixture.hidden * 4:
                raise BoundaryError("research request ownership differs")
            share = np.frombuffer(row["frames"][0], "<u4").reshape(1, -1)
            if index == 1 and fault == "fail":
                return
            if index == 1 and fault == "stall":
                await asyncio.Event().wait()
            for stripe, (start, matrix) in enumerate(matrices):
                if pending_native is None:
                    current_rows = min(fixture.stripe_rows, fixture.vocabulary - start) if striped else fixture.vocabulary
                    pending_native = submit(matrix, current_rows)
                output = await asyncio.shield(pending_native)
                pending_native = None
                owner.busy = False
                rows = output.shape[1]
                reserved_bytes = output.nbytes
                slots = 1
                if mode == "pipelined" and stripe + 1 < len(matrices):
                    # Single lookahead, never a task per remaining stripe.
                    next_start, next_matrix = matrices[stripe + 1]
                    next_rows = min(fixture.stripe_rows, fixture.vocabulary - next_start)
                    pending_native = submit(next_matrix, next_rows)
                    reserved_bytes += next_rows * 4
                    slots = 2
                owner.peak_reserved_output_slots = max(owner.peak_reserved_output_slots, slots)
                owner.peak_reserved_output_bytes = max(owner.peak_reserved_output_bytes, reserved_bytes)
                body = pack_frame(role=role, paired_session=("f" * 32 if index == 1 and fault == "context" else session),
                                  epoch=0, sequence=0, operation="head",
                                  frames=[output.astype("<u4", copy=False).tobytes()], response=True)
                number = stripe + 1 if index == 1 and fault == "reorder" else stripe
                header = HEADER.pack(b"BAD!" if index == 1 and fault == "malformed" else b"BHD1",
                                     number, start, rows, fixture.vocabulary, len(body))
                raw = header + body
                if index == 1 and fault == "truncate":
                    raw = raw[:-1]
                # Attempted body/envelope ledger is distinct from admitted stream ledger.
                ledger[role]["reply_body_bytes"] += len(body)
                ledger[role]["reply_envelope_bytes"] += len(header)
                await paced_write(writer, raw, pacer, role, "client")
                del output, body, raw, header  # Do not retain previous stripe during next await.
                if index == 1 and fault == "truncate":
                    return
        except Exception:
            # Client sees EOF; no partial output can become a successful exchange.
            pass
        finally:
            if pending_native is not None:
                try:
                    # Await Rust native call even when handler failed/cancelled during upload.
                    await asyncio.shield(pending_native)
                except Exception:
                    pass
                finally:
                    owner.drained_native_calls += 1
                    owner.busy = False
            owner.retired = True
            writer.close()
            await writer.wait_closed()
            worker_writers.discard(writer)
            handlers.discard(task)

    servers = []
    clients = []

    async def client(index):
        role = ROLES[index]
        port = servers[index].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port,
                                                       limit=2 * READ_CHUNK_BYTES)
        client_writers.add(writer)
        writer.transport.set_write_buffer_limits(high=2 * READ_CHUNK_BYTES,
                                                 low=READ_CHUNK_BYTES)
        try:
            raw = pack_frame(role=role, paired_session=session, epoch=0, sequence=0,
                             operation="head", frames=[(left if index == 0 else right).tobytes()])
            ledger[role]["request_body_bytes"] += len(raw)
            ledger[role]["request_envelope_bytes"] += LENGTH.size
            await paced_write(writer, LENGTH.pack(len(raw)) + raw, pacer, "client", role)
            output = np.empty((1, fixture.vocabulary), dtype="<u4")
            for stripe, (start, _) in enumerate(matrices):
                rows = min(fixture.stripe_rows, fixture.vocabulary - start) if striped else fixture.vocabulary
                body = await read_reply(reader, role=role, session=session, stripe=stripe,
                                        start=start, rows=rows, vocabulary=fixture.vocabulary)
                output[0, start:start + rows] = np.frombuffer(body, "<u4")
            if await reader.read(1):
                raise BoundaryError("research reply has trailing bytes")
            return output
        finally:
            writer.close()
            await writer.wait_closed()
            client_writers.discard(writer)

    async def exchange():
        if mode == "serial":
            return [await client(0), await client(1)]
        clients.extend(asyncio.create_task(client(i)) for i in range(2))
        return await asyncio.gather(*clients)

    operation = None
    alarm = None
    error, logits = None, None
    try:
        for index in range(2):
            servers.append(await asyncio.start_server(
                lambda r, w, i=index: worker(r, w, i), "127.0.0.1", 0,
                limit=2 * READ_CHUNK_BYTES, backlog=1))
        transport_setup = {"wall_seconds": time.perf_counter() - setup_started,
                           "cpu_seconds": time.process_time() - setup_cpu,
                           "network_bytes": 0}
        started, cpu = time.perf_counter(), time.process_time()
        operation = asyncio.create_task(exchange())
        if cancel_after is not None:
            alarm = asyncio.get_running_loop().call_later(cancel_after, operation.cancel)
        a, b = await asyncio.wait_for(operation, timeout_seconds)
        combined = a + b
        signed = combined.astype(np.int64)
        signed[combined >= (1 << 31)] -= 1 << 32
        np.testing.assert_array_equal(signed, fixture.oracle)
        logits = dequantize_matmul(signed, fixture.quantized.scales, fixture.scales,
                                  output_shape=(1, fixture.vocabulary))
        np.testing.assert_array_equal(logits, fixture.logits)
    except (Exception, asyncio.CancelledError) as exc:
        error = type(exc).__name__
    finally:
        if alarm:
            alarm.cancel()
        for task in clients:
            task.cancel()
        if clients:
            await asyncio.gather(*clients, return_exceptions=True)
        for server in servers:
            server.close()
        for writer in list(client_writers | worker_writers):
            writer.close()
        pending = list(handlers)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for server in servers:
            await server.wait_closed()
        for owner in owners:
            owner.retired = True  # Includes unattempted reserved owners.
        left.fill(0)
        # right owns immutable random bytes; this fixture does not claim secure erasure.
    measured = {"wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu}
    snapshot = pacer.snapshot()
    # The primitive's generic scope describes TCP relay use; this fixture paces bodies directly.
    snapshot["scope"] = "actual socket application stream: research + production frames; no HTTP/TLS"
    return {"mode": mode, "ok": error is None, "error": error,
            "transport_setup": transport_setup, "online": measured,
            "online_scope": "TCP connect, framed requests, native work, paced replies, full reconstruction/checks, drain/close; no imports or fixture compilation",
            "full_logits_sha256": None if logits is None else hashlib.sha256(logits.tobytes()).hexdigest(),
            "ledger": ledger, "wan": snapshot,
            "owners": [vars(owner) for owner in owners],
            "resources": {"process_lifetime_peak_rss_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          "rss_unit": "bytes on macOS, KiB on Linux", "max_client_tasks": 2,
                          "max_worker_handlers": 2, "max_active_native_calls": 2,
                          "streamreader_limit_bytes_each": 2 * READ_CHUNK_BYTES,
                          "writer_high_water_bytes_each": 2 * READ_CHUNK_BYTES,
                          "complete_reply_arrays_bytes": 2 * fixture.vocabulary * 4,
                          "worker_reply_rows_bound": fixture.stripe_rows if striped else fixture.vocabulary,
                          "worker_native_output_slots_bound": 2 if mode == "pipelined" and len(matrices) > 1 else 1,
                          "worker_native_output_bytes_bound": min(fixture.stripe_rows * 2, fixture.vocabulary) * 4 if mode == "pipelined" else (fixture.stripe_rows if striped else fixture.vocabulary) * 4,
                          "worker_pending_native_tasks_bound": 1,
                          "worker_current_serialization_bytes_bound": (fixture.stripe_rows if striped else fixture.vocabulary) * 12 + 2 * 1024 + HEADER.size,
                          "additional_buffers": "current native output + at most one next native output; current bytes + framed body + envelope copy per worker; client decoded frame; reconstruction/oracle arrays; OS socket buffers unmeasured",
                          "note": "one process includes generated source, native full/stripe matrices, workers, client and oracle; not whole-client qualification"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--stripe-rows", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--download-mbps", type=float, default=100.)
    parser.add_argument("--upload-mbps", type=float, default=40.)
    parser.add_argument("--timeout-seconds", type=float, default=30.)
    parser.add_argument("--order-offset", type=int, default=0,
                        help="Williams-design row offset; vary 0..3 across fresh-process runs")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("repeats must be 1..20")
    fixture = Fixture(args.vocabulary, args.hidden, args.stripe_rows)
    orders = arm_orders(args.repeats, args.order_offset)
    async def run():
        results = []
        for repetition, order in enumerate(orders):
            for position, mode in enumerate(order):
                row = await run_case(fixture, mode, download_mbps=args.download_mbps,
                                     upload_mbps=args.upload_mbps, timeout_seconds=args.timeout_seconds)
                row.update(repetition=repetition, order_position=position)
                results.append(row)
        return results
    results = asyncio.run(run())
    import pllm
    from pllm import _native
    import pllm.runtime.native_wan as native_wan
    import pllm.runtime.private_boundary_protocol as protocol
    provenance = {"installed_pllm": str(pllm.__file__),
                  "native_extension_path": str(_native.__file__),
                  "native_extension_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
                  "numpy_version": np.__version__,
                  "msgpack_version": importlib.metadata.version("msgpack"),
                  "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in (Path(__file__), Path(native_wan.__file__), Path(protocol.__file__))}}
    print(json.dumps({"schema": "pllm.research.boundary_head_duplex.v1",
                      "scope": "generated-source native arithmetic + paced loopback TCP transport fixture",
                      "geometry": vars(args), "provenance": provenance,
                      "setup": fixture.setup, "arm_orders": orders,
                      "order_design": "four-arm Williams; complete balance per four repetitions",
                      "runs": results,
                      "integration_gaps": ["trusted source/compiled role admission and authentication",
                                           "coupled lookup/body/KV and selection/RNG parity",
                                           "HTTP/TLS, physical wire, RTT, deployment processes",
                                           "cold traffic and complete-client <=256 MiB qualification"],
                      "claims": {"production_integration": False, "privacy_qualification": False,
                                 "sdk_performance_comparison": False}}, indent=2))
    return 0 if all(row["ok"] for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
