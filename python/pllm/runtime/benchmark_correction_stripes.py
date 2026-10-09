"""Test-local ordered correction push/ack with bounded immutable stripe files.

This is a single-stage benchmark fixture, not a provider inventory protocol. Its
private inherited pipe and fresh HMAC key identify a pair of local processes.
It never exposes incomplete corrections as READY or imports reusable tickets.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import select
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

from .stage_protocol import pack_residues, unpack_residues

_NAMESPACE = b"pllm.correction_stripe_fixture.v1\0"
_MAX_FRAME = (16 << 20) + 4096


def _admit(key, context, rows, columns, stripe_columns, wire_bits):
    if (type(key) is not bytes or len(key) != 32 or type(context) is not str or len(context) != 64
            or any(type(n) is not int for n in (rows, columns, stripe_columns, wire_bits))
            or not 1 <= rows <= 4096 or not 1 <= stripe_columns <= columns
            or rows * columns > 4_000_000 or wire_bits not in (16, 24, 32)):
        raise ValueError("invalid correction stripe admission")


def _authenticated(key, direction, metadata, body=b""):
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    if len(header) > 2048 or len(body) > _MAX_FRAME - 4096:
        raise ValueError("correction stripe frame exceeds bound")
    raw = struct.pack("!I", len(header)) + header + body
    return raw + hmac.digest(key, _NAMESPACE + direction + raw, "sha256")


def _open(key, direction, frame):
    if type(frame) is not bytes or not 36 <= len(frame) <= _MAX_FRAME:
        raise ValueError("invalid correction stripe frame length")
    raw, tag = frame[:-32], frame[-32:]
    if not hmac.compare_digest(tag, hmac.digest(key, _NAMESPACE + direction + raw, "sha256")):
        raise ValueError("correction stripe authentication failed")
    size, = struct.unpack("!I", raw[:4])
    if not 1 <= size <= 2048 or size + 4 > len(raw):
        raise ValueError("invalid correction stripe header")
    return json.loads(raw[4:size + 4]), raw[size + 4:]


class CorrectionStripeReceiver:
    def __init__(self, directory, *, key, context, rows, columns, stripe_columns, wire_bits):
        _admit(key, context, rows, columns, stripe_columns, wire_bits)
        self.key, self.context = key, context
        self.rows, self.columns, self.stripe_columns, self.wire_bits = rows, columns, stripe_columns, wire_bits
        self.root = Path(directory)
        self.root.mkdir(mode=0o700)
        self.next_column = 0
        self.sequence = 0
        self.ready = self.closed = False
        self.frames = self.received = self.sent = 0

    def accept(self, frame):
        try:
            if self.closed or self.ready:
                raise ValueError("correction stripe owner is terminal")
            header, body = _open(self.key, b"push", frame)
            if (not isinstance(header, dict)
                    or set(header) != {"context", "sequence", "operation", "column", "width"}
                    or header["context"] != self.context or type(header["sequence"]) is not int
                    or header["sequence"] != self.sequence or type(header["column"]) is not int
                    or header["column"] != self.next_column or type(header["width"]) is not int):
                raise ValueError("correction stripe ordering or binding differs")
            if header["operation"] == "seal":
                if body or header["width"] or self.next_column != self.columns:
                    raise ValueError("incomplete correction inventory cannot become READY")
                self.ready = True
            elif header["operation"] == "stripe":
                width = min(self.stripe_columns, self.columns - self.next_column)
                if (width < 1 or header["width"] != width
                        or len(body) != self.rows * width * (self.wire_bits // 8)):
                    raise ValueError("correction stripe geometry differs")
                # Decode under the admitted shape before acknowledging acceptance.
                decoded = unpack_residues(body, (self.rows, width), self.wire_bits)
                if decoded.shape != (self.rows, width):
                    raise ValueError("correction stripe codec differs")
                with (self.root / str(self.next_column)).open("xb") as sink:
                    sink.write(body)
                self.next_column += width
            else:
                raise ValueError("unknown correction stripe operation")
            ack = _authenticated(self.key, b"ack", {"context": self.context,
                "sequence": self.sequence, "next_column": self.next_column, "ready": self.ready,
                "accepted_digest": hashlib.sha256(frame).hexdigest()})
            self.sequence += 1
            self.frames += 1
            self.received += len(frame) + 4
            self.sent += len(ack) + 4
            return ack
        except BaseException:
            self.cancel()
            raise

    def read(self):
        if not self.ready or self.closed:
            raise ValueError("corrections are not READY")
        for column in range(0, self.columns, self.stripe_columns):
            width = min(self.stripe_columns, self.columns - column)
            with (self.root / str(column)).open("rb") as source:
                raw = source.read(self.rows * width * (self.wire_bits // 8) + 1)
            yield column, unpack_residues(raw, (self.rows, width), self.wire_bits)

    def cancel(self):
        self.closed, self.ready = True, False
        self.key = b""
        shutil.rmtree(self.root, ignore_errors=True)


class CorrectionStripeSender:
    def __init__(self, *, key, context, rows, columns, stripe_columns, wire_bits):
        _admit(key, context, rows, columns, stripe_columns, wire_bits)
        self.key, self.context = key, context
        self.rows, self.columns, self.stripe_columns, self.wire_bits = rows, columns, stripe_columns, wire_bits
        self.column = self.sequence = 0
        self.pending = None
        self.closed = False

    def frame(self, values=None):
        try:
            if self.closed or self.pending is not None:
                raise ValueError("correction stripe sender is not ready")
            seal = values is None
            width = 0 if seal else min(self.stripe_columns, self.columns - self.column)
            if (seal and self.column != self.columns or not seal and (width < 1
                    or type(values) is not np.ndarray or values.dtype != np.uint32
                    or values.shape != (self.rows, width))):
                raise ValueError("invalid correction stripe output")
            frame = _authenticated(self.key, b"push", {"context": self.context, "sequence": self.sequence,
                "operation": "seal" if seal else "stripe", "column": self.column, "width": width},
                b"" if seal else pack_residues(values, self.wire_bits))
            self.pending = (hashlib.sha256(frame).hexdigest(), self.column + width, seal)
            return frame
        except BaseException:
            self.cancel()
            raise

    def cancel(self):
        self.closed, self.pending, self.key = True, None, b""

    def acknowledge(self, frame):
        try:
            if self.closed or self.pending is None:
                raise ValueError("no pending correction stripe")
            header, body = _open(self.key, b"ack", frame)
            expected = {"context": self.context, "sequence": self.sequence,
                "next_column": self.pending[1], "ready": self.pending[2], "accepted_digest": self.pending[0]}
            if (body or header != expected or type(header.get("ready")) is not bool
                    or type(header.get("sequence")) is not int or type(header.get("next_column")) is not int):
                raise ValueError("correction stripe acknowledgement differs")
            self.column, self.closed = self.pending[1], self.pending[2]
            self.sequence += 1
            self.pending = None
        except BaseException:
            self.cancel()
            raise


def _read_exact(stream, size, deadline):
    result = bytearray()
    while len(result) < size:
        left = deadline - time.monotonic()
        if left <= 0 or not select.select([stream], [], [], left)[0]:
            raise TimeoutError("correction stripe peer deadline")
        chunk = os.read(stream.fileno(), size - len(result))
        if not chunk:
            raise ValueError("correction stripe peer closed")
        result.extend(chunk)
    return bytes(result)


def receive(stream, *, maximum=_MAX_FRAME):
    deadline = time.monotonic() + 60
    size, = struct.unpack("!I", _read_exact(stream, 4, deadline))
    if not 1 <= size <= maximum:
        raise ValueError("correction stripe stream length exceeds bound")
    return _read_exact(stream, size, deadline)


def send(stream, frame):
    if not 1 <= len(frame) <= _MAX_FRAME:
        raise ValueError("correction stripe stream length exceeds bound")
    stream.write(struct.pack("!I", len(frame)))
    stream.write(frame)
    stream.flush()


def worker():
    from pllm._benchmark_worker import peak_rss
    request = json.loads(receive(sys.stdin.buffer, maximum=16384))
    request["key"] = bytes.fromhex(request["key"])
    receiver = CorrectionStripeReceiver(**request)
    try:
        while not receiver.ready:
            ack = receiver.accept(receive(sys.stdin.buffer))
            send(sys.stdout.buffer, ack)
        send(sys.stdout.buffer, json.dumps({"cpu_seconds": time.process_time(),
            "peak_rss_bytes": peak_rss(), "frames": receiver.frames,
            "received": receiver.received, "sent": receiver.sent}).encode())
        # The parent owns this fixture directory and reads completed corrections
        # after the measured window. Failure never leaves an incomplete owner.
    except BaseException:
        receiver.cancel()
        raise


def execute_stripes(p, m):
    from pllm._benchmark_contract import TargetUnavailable, canonical, digest
    from pllm._benchmark_worker import peak_rss
    from pllm.native import MaskedGEMM, capabilities
    from .benchmark_targets import _oracle, _select, _weight_file, model_context
    from .preparation_protocol import PreparationRequest, expand_preparation_mask, prepared_mask_rows, seeded_ring_profile

    context = model_context(p, m)
    stage = _select(context[3], p)
    rows, width, out = p["rows"], stage.in_features, stage.out_features
    columns = min(p["output_columns"] or out, out)
    count = m.sampling.repetitions + m.sampling.warmups
    if context[4].verification_component is not None:
        raise TargetUnavailable("verified_striped_corrections_not_implemented")
    if (rows > context[0].budget.max_input_tokens or rows * max(width, out) > 4_000_000
            or rows * width * out * count > 2_000_000_000):
        raise TargetUnavailable("striped_correction_geometry_limit", status="blocked")
    if p["kernel"] == "digit8_native" and not capabilities().get("digit8_cpu"):
        raise TargetUnavailable("digit8_cpu_unavailable")
    m.admit((288 << 20) + 8 * width * out + 32 * rows * (width + out)
            + ((224 << 20) if p["kernel"] == "digit8_native" else 0))
    path, weight_hash, bound = _weight_file(stage, p, context, m)
    weights = np.memmap(path, dtype=np.int8, mode="r", shape=(out, width))
    executor = MaskedGEMM(threads=m.sampling.threads)
    if not executor.native:
        raise TargetUnavailable("native_extension_required")
    cpu, wall = time.process_time(), time.perf_counter()
    matrices = [executor.compile(weights[start:start + columns]) for start in range(0, out, columns)]
    m.details["kernel_binding"] = {"cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.perf_counter() - wall, "immutable_weight_bytes": width * out}
    profile = seeded_ring_profile(bound)
    samples = []

    def execute(index):
        req = PreparationRequest(secrets.token_hex(16), secrets.token_hex(16), "benchmark-stage",
            weight_hash, stage.id, weight_hash, rows, width, out, stage.weight_bits, stage.activation_bits,
            bound, profile.ring, profile.modulus, profile.wire_bits, secrets.token_bytes(32))
        root = m.directory / f"receiver-{index}"
        key = secrets.token_bytes(32)
        binding = digest({"plan": context[2].digest, "request": req.metadata()})
        contract = dict(key=key, context=binding, rows=rows, columns=out,
                        stripe_columns=columns, wire_bits=profile.wire_bits)
        sender = CorrectionStripeSender(**contract)
        cursor = prepared_mask_rows(req)
        process = None
        complete = False
        try:
            if cursor is None:
                raise TargetUnavailable("native_mask_stream_required")
            before = time.process_time()
            raw_r, raw_s = cursor.take(0, rows)
            r = np.frombuffer(raw_r, "<u4").reshape(rows, width)
            s = np.frombuffer(raw_s, "<u4").reshape(rows, out)
            mask_cpu = time.process_time() - before
            process = subprocess.Popen([sys.executable, "-I", "-m", __name__],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            bootstrap = canonical({**contract, "key": key.hex(), "directory": str(root)})
            send(process.stdin, bootstrap)
            pushed = acknowledged = 0
            kernel_cpu = exchange_wall = 0.0
            for number, matrix in enumerate(matrices):
                start = number * columns
                before = time.process_time()
                correction = (matrix.digit8(r, profile.wire_bits) if p["kernel"] == "digit8_native"
                    else matrix.wrap32(r) if profile.modulus == 1 << 32 else matrix.modular(r, profile.modulus))
                correction = np.subtract(correction, s[:, start:start + correction.shape[1]], dtype=np.uint32)
                correction &= np.uint32(profile.modulus - 1)
                kernel_cpu += time.process_time() - before
                before_wall = time.perf_counter()
                frame = sender.frame(correction)
                del correction
                send(process.stdin, frame)
                ack = receive(process.stdout)
                sender.acknowledge(ack)
                exchange_wall += time.perf_counter() - before_wall
                pushed += len(frame) + 4
                acknowledged += len(ack) + 4
                del frame, ack
            frame = sender.frame()
            send(process.stdin, frame)
            ack = receive(process.stdout)
            sender.acknowledge(ack)
            pushed += len(frame) + 4
            acknowledged += len(ack) + 4
            raw_stats = receive(process.stdout, maximum=4096)
            stats = json.loads(raw_stats)
            process.wait(timeout=5)
            if (process.returncode or stats["received"] != pushed or stats["sent"] != acknowledged
                    or stats["frames"] != len(matrices) + 1):
                raise ValueError("correction receiver accounting differs")
            stats.update(push_bytes=pushed, acknowledgement_bytes=acknowledged,
                control_bytes=len(bootstrap) + len(raw_stats) + 8, mask_cpu_seconds=mask_cpu,
                kernel_cpu_seconds=kernel_cpu, exchange_wall_seconds=exchange_wall)
            samples.append(stats)
            if index < m.sampling.warmups:
                shutil.rmtree(root)
            complete = True
            return root, req
        finally:
            sender.cancel()
            if cursor is not None:
                cursor.cancel()
            if process is not None:
                process.stdin.close()
                process.stdout.close()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            if not complete:
                shutil.rmtree(root, ignore_errors=True)

    values = m.run(execute)
    m.details["target_peak_rss_bytes"] = peak_rss()
    m.details["target_cpu_from_birth_seconds"] = time.process_time()
    samples = samples[m.sampling.warmups:]
    m.details["timings"]["cpu_seconds"] = [own + row["cpu_seconds"] for own, row in
        zip(m.details["timings"]["cpu_seconds"], samples, strict=True)]
    for root, req in values:
        expected = (_oracle(weights, expand_preparation_mask(req), profile.modulus)
                    - expand_preparation_mask(req, "output-s")) & np.uint32(profile.modulus - 1)
        for start in range(0, out, columns):
            shape = (rows, min(columns, out - start))
            with (root / str(start)).open("rb") as source:
                actual = unpack_residues(source.read(rows * shape[1] * (profile.wire_bits // 8) + 1), shape, profile.wire_bits)
            if not np.array_equal(actual, expected[:, start:start + shape[1]]):
                raise TargetUnavailable("striped_correction_integer_parity_failed", status="failed")
        shutil.rmtree(root)
    m.oracle = {"kind": "independent_signed_i64", "outcome": "passed", "checked_values": rows * out * len(values)}
    m.comparison.update(workload_digest=digest({"shape": [rows, width, out], "seed": p["seed"]}),
        output_digest=m.commitment({"algebra": "W-r-minus-s", "weight": weight_hash, "ring": profile.to_dict(), "shape": [rows, out]}),
        measurement_digest=digest({"scope": "correction_stripe_pipe_ack_v1", "oracle_outside_timing": True}))
    m.details.update(output_columns=columns, input_mask_bytes=rows * width * 4,
        retained_output_mask_bytes=rows * out * 4, maximum_correction_array_bytes=rows * columns * 4,
        receiver_samples=samples, target_cpu_owners=["preparation", "receiver_child"], integer_macs=rows * width * out,
        private_correction_file_bytes=rows * out * (profile.wire_bits // 8) * len(values),
        per_sample_correction_file_bytes=rows * out * (profile.wire_bits // 8))
    m.metric("receiver.peak_rss", max(row["peak_rss_bytes"] for row in samples), kind="memory", role="receiver",
        window="process_birth_through_acknowledged_seal", scope="process_lifetime_high_water")
    for name in ("push_bytes", "acknowledgement_bytes", "control_bytes"):
        m.metric("stripe." + name, sum(row[name] for row in samples), kind="bytes", scope="framed_local_private_pipe")
    m.limitations += ["Single compiler-selected stage with separate local pipe peers; not an HTTP provider inventory or independently operated service.",
        "Full input and output masks remain resident; only correction arrays and delivery frames are output-striped.",
        "Both-party sample CPU includes receiver startup; initial weight binding, private disk and endpoint peaks stay separate.",
        "No WAN, physical wire, paged-weight stripe kernel or whole-response measurement follows from this fixture."]


if __name__ == "__main__":
    worker()
