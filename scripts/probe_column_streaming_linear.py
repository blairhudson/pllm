"""Bounded research-only authenticated column stream; no SDK protocol changes.

Run with installed target/frontier-sdk-env/bin/python, never PYTHONPATH=python.
Only --transport socketpair measures local transport; WAN schedules are models.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import resource
import secrets
import socket
import struct
import sys
import threading
import time

import msgpack
import numpy as np
from pllm import _native
from pllm.native import MaskedGEMM
from pllm.runtime.residue_codec import PREPARED_ROW_MAGIC, unpack_row_response

MAGIC = b"PLLMCS09"
MAX_FRAME = 8 * 1024 * 1024
MAX_TICKETS = 128


class StreamError(ValueError):
    pass


def oracle(w, x):
    """Independent arbitrary-precision integer oracle: no NumPy dot/native call."""
    return np.array([[sum(int(a) * int(b) for a, b in zip(row, weight, strict=True))
                      for weight in w] for row in x], dtype=np.int64)


@dataclass(frozen=True)
class Contract:
    session: str
    stage: str
    tickets: tuple[str, ...]
    rows: int
    columns: int
    outputs: int
    block: int
    bits: int
    weight_digest: str
    widths: bytes

    def __post_init__(self):
        if (any(type(n) is not int for n in (self.rows, self.columns, self.outputs, self.block, self.bits))
                or not 1 <= self.rows <= 32 or not 1 <= self.columns <= 1024
                or not 1 <= self.outputs <= 1024 or not 1 <= self.block <= self.columns
                or self.bits not in (16, 24, 32)
                or type(self.widths) is not bytes or len(self.widths) != self.outputs
                or not self.widths or min(self.widths) < 1 or max(self.widths) > self.bits
                or type(self.tickets) is not tuple or len(self.tickets) != self.rows
                or len(set(self.tickets)) != self.rows
                or any(type(t) is not str or len(t) != 32 or any(c not in '0123456789abcdef' for c in t)
                       for t in self.tickets)
                or any(type(s) is not str or not 1 <= len(s) <= 64 for s in (self.session, self.stage))
                or type(self.weight_digest) is not str or len(self.weight_digest) != 64
                or any(c not in '0123456789abcdef' for c in self.weight_digest)):
            raise StreamError("public contract exceeds bounded namespace or ring")

    @property
    def digest(self):
        return hashlib.sha256(msgpack.packb([
            "pllm.column-streaming.research.v1", self.session, self.stage, self.tickets,
            self.rows, self.columns, self.outputs, self.block, self.bits,
            self.weight_digest, self.widths], use_bin_type=True)).digest()

    @property
    def result_ticket(self):
        return self.digest.hex()


class TicketBook:
    """Bounded volatile authority: no restart/rollback durability claim."""
    def __init__(self, contract, correction):
        self.contract = contract
        self.correction = correction
        self.burned = set()
        self.claims = 0
        if len(contract.tickets) > MAX_TICKETS:
            raise StreamError("ticket admission")
        expected = (contract.rows * sum(contract.widths) + 7) // 8
        if type(correction) is not bytes or len(correction) != expected:
            raise StreamError("correction size")

    def claim(self, contract):
        if contract != self.contract or self.burned or self.correction is None:
            raise StreamError("ticket replay or namespace mismatch")
        # Burn entire projection before computation, same as consume_preloaded_batch.
        self.burned.update((contract.session, contract.stage, t) for t in contract.tickets)
        value, self.correction = self.correction, None
        self.claims += 1
        return value

    def burn(self):
        self.burned.update((self.contract.session, self.contract.stage, t) for t in self.contract.tickets)
        self.correction = None


def seal(key, contract, kind, sequence, start, end, payload=b""):
    if type(key) is not bytes or len(key) != 32:
        raise StreamError("authentication key admission")
    record = msgpack.packb([contract.digest, kind, sequence, start, end, payload], use_bin_type=True)
    body = MAGIC + record
    body += hmac.digest(key, body, "sha256")
    if len(body) > MAX_FRAME:
        raise StreamError("frame admission")
    return struct.pack("!I", len(body)) + body


def open_frame(key, contract, frame):
    if (type(key) is not bytes or len(key) != 32
            or type(frame) is not bytes or not 44 <= len(frame) <= MAX_FRAME + 4
            or struct.unpack("!I", frame[:4])[0] != len(frame) - 4
            or frame[4:12] != MAGIC
            or not hmac.compare_digest(frame[-32:], hmac.digest(key, frame[4:-32], "sha256"))):
        raise StreamError("frame authentication or length")
    try:
        record = msgpack.unpackb(frame[12:-32], raw=False, max_array_len=6,
                                max_map_len=0, max_str_len=16, max_bin_len=MAX_FRAME, max_ext_len=0)
    except (ValueError, TypeError, msgpack.UnpackException) as exc:
        raise StreamError("frame encoding") from exc
    if (type(record) is not list or len(record) != 6 or record[0] != contract.digest
            or record[1] not in ("start", "column", "finish", "result", "cancel")
            or any(type(n) is not int or not 0 <= n <= 1026 for n in record[2:5])
            or type(record[5]) is not bytes):
        raise StreamError("frame namespace or fields")
    return record[1:]


class Provider:
    def __init__(self, w, contract, book, key):
        setup_cpu, setup_wall = time.process_time(), time.perf_counter()
        if (w.dtype != np.int8 or w.shape != (contract.outputs, contract.columns)
                or hashlib.sha256(w.tobytes()).hexdigest() != contract.weight_digest
                or _native.offset_row_bits(w.tobytes(), contract.columns, 127) != contract.widths):
            raise StreamError("weight/layout commitment")
        self.contract, self.book, self.key = contract, book, key
        owner = MaskedGEMM(threads=1)
        if owner.backend != "rust":
            raise RuntimeError("installed native integer kernel required; no build/fallback")
        # Column slices are copied once, before online execution; total packed weight
        # bytes equal full W, but one native matrix/launch per column block.
        self.kernels = [owner.compile(np.ascontiguousarray(w[:, s:s + contract.block]))
                        for s in range(0, contract.columns, contract.block)]
        self.accumulator = np.zeros((contract.rows, contract.outputs), dtype="<u4")
        self.sequence = 0
        self.offset = 0
        self.state = "reserved"
        self.correction = None
        self.correction_applications = 0
        self.launches = 0
        self.compute_seconds = []
        self.cpu_seconds = 0.0
        self.first_partial_started_at = None
        self.setup_cpu_seconds = time.process_time() - setup_cpu
        self.setup_wall_seconds = time.perf_counter() - setup_wall

    def cancel(self):
        self.book.burn()
        self.accumulator.fill(0)
        self.correction = None
        self.state = "terminal"

    def receive(self, frame):
        began_cpu = time.thread_time()
        try:
            if self.state == "terminal":
                raise StreamError("terminal projection")
            kind, seq, start, end, payload = open_frame(self.key, self.contract, frame)
            if kind == "cancel":
                if (seq, start, end, payload) != (self.sequence, self.offset, self.offset, b""):
                    raise StreamError("cancel sequence")
                self.cancel()
                return None
            if seq != self.sequence:
                raise StreamError("reordered or duplicate chunk")
            if self.state == "reserved":
                if (kind, start, end, payload) != ("start", 0, 0, b""):
                    raise StreamError("missing start")
                self.correction = self.book.claim(self.contract)
                self.state = "streaming"
            elif kind == "column":
                c = self.contract
                stop = min(c.columns, self.offset + c.block)
                if start != self.offset or end != stop or start >= c.columns:
                    raise StreamError("column geometry/order")
                widths = bytes([c.bits]) * (end - start)
                if len(payload) != c.rows * ((sum(widths) + 7) // 8):
                    raise StreamError("truncated column")
                raw = _native.unpack_residue_rows(payload, widths, c.rows)
                block = np.frombuffer(raw, "<u4").reshape(c.rows, end - start)
                began = time.perf_counter()
                if self.first_partial_started_at is None:
                    self.first_partial_started_at = began
                partial = self.kernels[self.launches].wrap32(block)
                # wrap32 plus u32 accumulation is exact modulo 2^32. Reducing to
                # u16/u24 once at output preserves lower ring homomorphism.
                np.add(self.accumulator, partial, out=self.accumulator, casting="unsafe")
                self.compute_seconds.append(time.perf_counter() - began)
                self.launches += 1
                self.offset = end
            elif kind == "finish":
                c = self.contract
                if (start, end, payload) != (c.columns, c.columns, b"") or self.offset != c.columns:
                    raise StreamError("incomplete projection")
                packed = _native.prepared_pack_output(self.accumulator.tobytes(), self.correction, c.widths, c.rows)
                self.correction_applications += 1
                response = PREPARED_ROW_MAGIC + msgpack.packb([
                    c.result_ticket, c.stage, c.rows, c.bits, hashlib.sha256(c.widths).digest(), 0, packed],
                    use_bin_type=True)
                result = seal(self.key, c, "result", seq, c.columns, c.columns, response)
                self.cancel()
                return result
            else:
                raise StreamError("unexpected frame kind")
            self.sequence += 1
            return None  # Never disclose prefix sums, including masked prefix output.
        except BaseException:
            self.cancel()
            raise
        finally:
            self.cpu_seconds += time.thread_time() - began_cpu

    def eof(self):
        self.cancel()
        raise StreamError("incomplete transport burns projection")


class Client:
    def __init__(self, contract, x, r, s, key):
        self.contract, self.x, self.r, self.s, self.key = contract, x, r, s, key
        if (x.dtype != np.int8 or np.any(x == -128) or x.shape != (contract.rows, contract.columns)
                or r.dtype != np.dtype("<u4") or r.shape != x.shape
                or s.dtype != np.dtype("<u4") or s.shape != (contract.rows, contract.outputs)
                or np.any(r.astype(np.uint64) >= 1 << contract.bits)
                or np.any(s.astype(np.uint64) >= 1 << contract.bits)):
            raise StreamError("client ring/activation admission")
        self.closed = False
        self.frames_issued = False

    def cancel(self):
        self.closed = True
        self.r.fill(0)
        self.s.fill(0)

    def frames(self):
        if self.closed or self.frames_issued:
            raise StreamError("client stream replay or cancellation")
        self.frames_issued = True
        c = self.contract
        yield seal(self.key, c, "start", 0, 0, 0)
        seq = 1
        for start in range(0, c.columns, c.block):
            end = min(c.columns, start + c.block)
            # Materialize only one block; no full masked input or frame list.
            masked = _native.prepared_mask_input(np.ascontiguousarray(self.x[:, start:end]).tobytes(),
                np.ascontiguousarray(self.r[:, start:end]).tobytes(), c.bits)
            payload = _native.offset_pack_rows(masked, bytes([c.bits]) * (end - start), c.rows)
            yield seal(self.key, c, "column", seq, start, end, payload)
            seq += 1
        yield seal(self.key, c, "finish", seq, c.columns, c.columns)

    def finish(self, frame):
        if self.closed:
            raise StreamError("client result replay")
        self.closed = True
        c = self.contract
        kind, seq, start, end, payload = open_frame(self.key, c, frame)
        if (kind, seq, start, end) != ("result", math.ceil(c.columns / c.block) + 1, c.columns, c.columns):
            raise StreamError("final result order")
        packed, _ = unpack_row_response(payload, ticket=c.result_ticket, stage=c.stage,
            widths=c.widths, rows=c.rows, bits=c.bits, namespace="prepared")
        return np.frombuffer(_native.prepared_unpack_output(packed, self.s.tobytes(), c.widths, c.rows),
                             "<i8").reshape(c.rows, c.outputs)


def fixture(rows=3, columns=17, outputs=7, block=4, bits=24, w=None, x=None):
    if (any(type(n) is not int for n in (rows, columns, outputs, block, bits))
            or not 1 <= rows <= 32 or not 1 <= columns <= 1024 or not 1 <= outputs <= 1024
            or not 1 <= block <= columns or bits not in (16, 24, 32)):
        raise StreamError("fixture admission before allocation")
    rng = np.random.default_rng(9009)
    w = rng.integers(-63, 64, (outputs, columns), dtype=np.int8) if w is None else w
    x = rng.integers(-127, 128, (rows, columns), dtype=np.int8) if x is None else x
    widths = _native.offset_row_bits(w.tobytes(), columns, 127)
    c = Contract(secrets.token_hex(16), "projection9", tuple(secrets.token_hex(16) for _ in range(rows)),
                 rows, columns, outputs, block, bits, hashlib.sha256(w.tobytes()).hexdigest(), widths)
    key = secrets.token_bytes(32)  # Auth-only key; independent of masks. Not an attestation mechanism.
    # Preparation knows r,s,W; provider receives ONLY correction, never mask seed/r/s.
    r = np.frombuffer(secrets.token_bytes(rows * columns * 4), "<u4").copy().reshape(rows, columns)
    s = np.frombuffer(secrets.token_bytes(rows * outputs * 4), "<u4").copy().reshape(rows, outputs)
    r &= (1 << bits) - 1
    s &= (1 << bits) - 1
    began = time.process_time()
    kernel = MaskedGEMM(threads=1).compile(w)
    if kernel.owner.backend != "rust":
        raise RuntimeError("native required")
    correction = _native.prepared_pack_correction(kernel.wrap32(r).tobytes(), s.tobytes(), widths, rows)
    preparation_cpu = time.process_time() - began
    book = TicketBook(c, correction)
    return Client(c, x, r, s, key), Provider(w, c, book, key), oracle(w, x), preparation_cpu


def recv_frame(sock):
    def exact(n):
        data = bytearray()
        while len(data) < n:
            part = sock.recv(n - len(data))
            if not part:
                raise StreamError("truncated socket stream")
            data.extend(part)
        return bytes(data)
    prefix = exact(4)
    size = struct.unpack("!I", prefix)[0]
    if not 40 <= size <= MAX_FRAME:
        raise StreamError("socket frame admission")
    return prefix + exact(size)


def _execute(client, provider, transport="direct"):
    uploads = []
    input_upload_completed_at = None
    last_input_frame_number = math.ceil(client.contract.columns / client.contract.block) + 1
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    if transport == "direct":
        response = None
        for frame in client.frames():
            uploads.append(len(frame))
            if len(uploads) == last_input_frame_number:
                input_upload_completed_at = time.perf_counter()
            reply = provider.receive(frame)
            if reply is not None:
                response = reply
    elif transport == "socketpair":
        errors = []
        left, right = socket.socketpair()
        left.settimeout(10)
        right.settimeout(10)
        def serve():
            try:
                while provider.state != "terminal":
                    reply = provider.receive(recv_frame(right))
                    if reply is not None:
                        right.sendall(reply)
            except BaseException as exc:
                errors.append(str(exc))
                provider.cancel()
            finally:
                right.close()
        worker = threading.Thread(target=serve, name="column-stream-provider")
        worker.start()
        try:
            for frame in client.frames():
                uploads.append(len(frame))
                left.sendall(frame)
                if len(uploads) == last_input_frame_number:
                    input_upload_completed_at = time.perf_counter()
            response = recv_frame(left)
        finally:
            left.close()
            worker.join(11)
            if worker.is_alive():
                provider.cancel()
                raise StreamError("socket worker deadline")
        if errors:
            raise StreamError(errors[0])
    else:
        raise StreamError("transport choice")
    result = client.finish(response)
    return result, {"upload_frame_bytes": uploads, "download_frame_bytes": len(response),
                    "complete_projection_wall_seconds": time.perf_counter() - wall_start,
                    "combined_client_provider_cpu_seconds": time.process_time() - cpu_start,
                    "provider_online_thread_cpu_seconds": provider.cpu_seconds,
                    "client_plus_transport_cpu_seconds": max(0, time.process_time() - cpu_start - provider.cpu_seconds),
                    "partial_native_and_accumulate_seconds": provider.compute_seconds,
                    "provider_cold_matrix_setup_cpu_seconds": provider.setup_cpu_seconds,
                    "provider_cold_matrix_setup_wall_seconds": provider.setup_wall_seconds,
                    "first_partial_started_before_local_input_upload_completed":
                        provider.first_partial_started_at < input_upload_completed_at,
                    "upload_completion_scope": "last column sendall return for socketpair; before last receive call for direct",
                    "native_launches": provider.launches, "correction_applications": provider.correction_applications}


def execute(client, provider, transport="direct"):
    try:
        return _execute(client, provider, transport)
    except BaseException:
        provider.cancel()
        client.cancel()
        raise


def modeled_schedule(metrics, upload_mbps, download_mbps, delay_ms, contention):
    """Serial access link, single compute lane, no prefix downloads/ACK frames."""
    up, down = upload_mbps * 1e6 / 8, download_mbps * 1e6 / 8
    uploaded = computed = 0.0
    cpu = iter(metrics["partial_native_and_accumulate_seconds"])
    frames = metrics["upload_frame_bytes"]
    for index, size in enumerate(frames):
        uploaded += size / up
        if 0 < index < len(frames) - 1:
            computed = max(computed, uploaded + delay_ms / 1000) + next(cpu) * contention
    return max(computed, uploaded + delay_ms / 1000) + metrics["download_frame_bytes"] / down + delay_ms / 1000


def admission(rows, columns, outputs, block, transport):
    # Data lifetimes, packed frames/copies, native matrices, thread-stack reserve.
    mask = 4 * rows * columns
    output = 4 * rows * outputs
    frame = rows * block * 4 + 1024
    values = {"client_input_and_full_r_bytes": rows * columns + mask,
            "client_s_and_result_bytes": output + 8 * rows * outputs,
            "client_block_and_codec_copy_reserve_bytes": 8 * frame,
            "provider_private_accumulator_bytes": output,
            "provider_partial_result_bytes": output,
            "provider_correction_upper_bytes": output,
            "provider_source_plus_packed_weights_bytes": 2 * columns * outputs,
            "preparation_r_s_wr_and_source_upper_bytes": mask + 2 * output + 2 * columns * outputs,
            "thread_stack_reserve_bytes": (2 if transport == "socketpair" else 1) * 8 * 1024 * 1024,
            "native_matrix_metadata_reserve_bytes": math.ceil(columns / block) * 65536,
            "runtime_import_reserve_bytes": 128 * 1024 * 1024,
            "host_headroom_recommendation_bytes": 512 * 1024 * 1024,
            "whole_process_admission_limit_bytes": 256 * 1024 * 1024,
            "qualification": "estimate only; whole-client fresh-process replicated RSS gate missing"}
    values["conservative_estimated_co_resident_bytes"] = sum(value for key, value in values.items()
        if key.endswith("_bytes") and key not in ("host_headroom_recommendation_bytes", "whole_process_admission_limit_bytes"))
    return values


def run(args):
    budget = admission(args.rows, args.columns, args.outputs, args.block, args.transport)
    if (budget["conservative_estimated_co_resident_bytes"] > budget["whole_process_admission_limit_bytes"]
            or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > budget["whole_process_admission_limit_bytes"]):
        raise StreamError("co-resident memory admission exceeds 256 MiB")
    records = []
    rng = np.random.default_rng(9009)
    weights = rng.integers(-63, 64, (args.outputs, args.columns), dtype=np.int8)
    activations = rng.integers(-127, 128, (args.rows, args.columns), dtype=np.int8)
    # Row baseline executes distinct one-use prepared row groups. It uses same
    # research authentication envelope (not production wire-byte parity).
    for mode, block, group in [("full-stage", args.columns, args.rows),
                               ("column-stream", args.block, args.rows),
                               ("row-duplex-baseline", args.columns, min(16, args.rows))]:
        for replicate in range(args.repetitions):
            groups = []
            prep = 0.0
            for start in range(0, args.rows, group):
                client, provider, expected, preparation = fixture(min(group, args.rows - start),
                    args.columns, args.outputs, block, args.bits, weights, activations[start:start + group])
                correction_bytes = len(provider.book.correction)
                result, metrics = execute(client, provider, args.transport)
                if not np.array_equal(result, expected):
                    raise AssertionError("integer oracle mismatch")
                assert provider.book.claims == 1 and provider.correction_applications == 1
                prep += preparation
                metrics["reference_offline_payload_ledger"] = {
                    "preparation_to_provider_preloaded_correction_bytes": correction_bytes,
                    "preparation_to_client_raw_r_s_bytes": client.r.nbytes + client.s.nbytes,
                    "scope": "payloads only, not transported; offline authenticated frames/session costs remain unmeasured"}
                metrics["modeled_overlap_seconds"] = modeled_schedule(metrics, args.upload_mbps,
                    args.download_mbps, args.delay_ms, args.contention)
                groups.append(metrics)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > budget["whole_process_admission_limit_bytes"]:
                    raise StreamError("observed co-resident RSS exceeded admission; stop")
            records.append({"mode": mode, "replicate": replicate, "groups": groups,
                            "preparation_correction_cpu_seconds": prep, "exact_independent_integer_oracle": True})
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "tests/test_column_streaming_linear_probe.py",
             root / "docs/research/column-streaming-linear-reference.md",
             root / "python/pllm/runtime/correction_rendezvous.py",
             root / "python/pllm/runtime/server.py", root / "python/pllm/runtime/residue_codec.py",
             root / "python/pllm/metrics/prepared_duplex.py", root / "python/pllm/runtime/native.py",
             root / "crates/pllm-core/src/offset_transport.rs"]
    native = Path(_native.__file__)
    return {"schema": "pllm.column_streaming_linear.research.v1", "shape": [args.rows, args.columns, args.outputs],
            "bits": args.bits, "block": args.block, "transport": args.transport,
            "timing_scope": "measured local complete projection, native kernels/codecs; modeled WAN overlap only",
            "records": records, "admission": admission(args.rows, args.columns, args.outputs, args.block, args.transport),
            "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "application_ack_frames": 0, "transport_frames_include": "4-byte length, 8-byte magic, MessagePack, 32-byte HMAC",
            "all_link_exclusions": ["TCP ACK/IP/TLS (socketpair is AF_UNIX)", "session authorization/reservation",
                "cold weights", "Preparation correction delivery and mask-seed delivery"],
            "wan_model": {"upload_mbps": args.upload_mbps, "download_mbps": args.download_mbps,
                          "one_way_delay_ms": args.delay_ms, "compute_contention_multiplier": args.contention,
                          "excluded": "client masking/auth CPU, final codec/correction, queue jitter, packet ACKs/TLS"},
            "role_measurement_limits": {"client_cpu": "combined minus measured provider receive thread CPU; includes transport overhead",
                "client_rss": "roles co-resident; process peak is not client-only tier qualification",
                "preparation_cpu": "native correction and matrix compile only; randomness, mask delivery excluded",
                "preparation_rss": "estimated live bytes only, not fresh process measured",
                "TTFT": "complete synthetic projection only, not whole decoder TTFT",
                "N_minus_1_decode_throughput": "not measured; optimization targets prefill"},
            "row_baseline_scope": "serial authenticated row-group reference; not production duplex WebSocket timing",
            "replay_command": ["target/frontier-sdk-env/bin/python", "scripts/probe_column_streaming_linear.py", *sys.argv[1:]],
            "environment": {"python": sys.version, "native_threads": 1, "pid": os.getpid()},
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()},
            "installed_native": {"path": str(native), "sha256": hashlib.sha256(native.read_bytes()).hexdigest()},
            "gates_missing": ["production authenticated session adapter", "restart-safe ticket authority",
                "serial parent transport cohort", "matched whole-client tier RSS", "complete checkpoint TTFT/N-1 decode",
                "all-link accounting and >=90% remote body MAC placement"], "qualified": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=3)
    p.add_argument("--columns", type=int, default=17)
    p.add_argument("--outputs", type=int, default=7)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--bits", type=int, choices=(16, 24, 32), default=24)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--transport", choices=("direct", "socketpair"), default="direct")
    p.add_argument("--upload-mbps", type=float, default=40)
    p.add_argument("--download-mbps", type=float, default=100)
    p.add_argument("--delay-ms", type=float, default=20)
    p.add_argument("--contention", type=float, default=1)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (not 1 <= args.rows <= 32 or not 1 <= args.columns <= 1024 or not 1 <= args.outputs <= 1024
            or not 1 <= args.block <= args.columns or not 1 <= args.repetitions <= 8
            or any(not math.isfinite(v) or not 0.001 <= v <= 1000 for v in (args.upload_mbps, args.download_mbps))
            or not math.isfinite(args.delay_ms) or not 0 <= args.delay_ms <= 1000
            or not math.isfinite(args.contention) or not 1 <= args.contention <= 16):
        p.error("bounded resource/rate admission")
    evidence = run(args)
    text = json.dumps(evidence, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
