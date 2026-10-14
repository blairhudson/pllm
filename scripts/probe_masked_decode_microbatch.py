"""Bounded online masked-row reference; not an SDK or transport integration.

Owner methods serialize state with a lock. dispatch() is a single compute owner;
call next_wake()/dispatch() from an external event loop/timer. No hidden threads.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import resource
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PublicKey:
    model: str
    stage: str
    weight_digest: str
    ring: str
    modulus: int
    input_width: int
    output_width: int
    codec: str = "raw-u32-le-v1"
    wire_bits: int = 32
    weight_bits: int = 8
    activation_bits: int = 8
    body_fingerprint: str = "synthetic-body-v1"

    def encoded(self):
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()


def oracle(weights, rows, modulus):
    """Independent Python integer accumulation, no NumPy dot or native kernels."""
    return np.array([[sum(int(a) * int(b) for a, b in zip(row, w)) % modulus
                      for w in weights] for row in rows], dtype=np.uint32)


def signature(secret, session, ticket, key, payload):
    header = json.dumps([session, ticket], separators=(",", ":")).encode()
    return hmac.digest(secret, header + b"\0" + key.encoded() + b"\0" + payload, "sha256")


@dataclass
class Reply:
    output: np.ndarray | None = None
    mac: bytes | None = None
    error: str | None = None
    done: bool = False
    queue_seconds: float = 0.0


@dataclass
class Ticket:
    key: PublicKey
    correction: bytes | None
    state: str = "prepared"


@dataclass
class Session:
    secret: bytes
    last_active: float
    tickets: dict[str, Ticket] = field(default_factory=dict)
    pending: bool = False


@dataclass
class Row:
    session: str
    generation: Session
    ticket: str
    key: PublicKey
    payload: bytes
    entered: float
    expires: float
    reply: Reply


class BatchOwner:
    """Fixed matrix, bounded sessions/tickets/queue, at most one pending/session.

    Public compatibility only. Invalid credentials never burn another owner's
    ticket. Authenticated malformed/replayed rows fail locally. Consume before
    native execution; compute failure/cancellation never restores fresh material.
    """
    def __init__(self, weights, *, batch_size=8, wait_seconds=.00025,
                 max_sessions=32, tickets_per_session=32, queue_capacity=32,
                 memory_bytes=8 << 20, idle_seconds=30., adaptive=False,
                 clock=time.monotonic, kernel=None):
        integers = ((batch_size, 32), (max_sessions, 32), (tickets_per_session, 32),
                    (queue_capacity, 32), (memory_bytes, 128 << 20))
        if (any(type(value) is not int or not 1 <= value <= cap for value, cap in integers)
                or type(wait_seconds) not in (int, float)
                or not 0 <= wait_seconds <= 1. or not math.isfinite(wait_seconds)
                or type(idle_seconds) not in (int, float)
                or not 0 < idle_seconds <= 3600. or not math.isfinite(idle_seconds)
                or type(adaptive) is not bool or not callable(clock)
                or (kernel is not None and not callable(kernel))):
            raise ValueError("invalid owner limits")
        # Inspect public geometry without np.asarray(), serialization, or native
        # allocation. Array-like conversion can itself allocate unbounded memory.
        if (not isinstance(weights, np.ndarray) or weights.ndim != 2
                or not 1 <= min(weights.shape) <= max(weights.shape) <= 1024
                or weights.dtype != np.int8):
            raise ValueError("weights require bounded int8 ndarray [out,in], dimensions <=1024")
        raw = weights
        out, inp = raw.shape
        initial_charge = 2 * raw.size + (1 << 20) + batch_size * (inp * 12 + out * 20)
        if initial_charge > memory_bytes:
            raise ValueError("memory admission exhausted")
        if kernel is None:
            from pllm.runtime._native_support import extension
            from pllm.runtime.native import MaskedGEMM
            if extension() is None:
                raise ValueError("installed native backend required; no Python fallback")
        self.weights = np.frombuffer(raw.tobytes(), dtype=np.int8).reshape(raw.shape)
        self.digest = hashlib.sha256(self.weights.tobytes()).hexdigest()
        self.clock, self.batch_size, self.wait = clock, batch_size, wait_seconds
        self.max_sessions, self.ticket_limit = max_sessions, tickets_per_session
        self.queue_capacity, self.memory_limit = queue_capacity, memory_bytes
        self.idle, self.adaptive = idle_seconds, adaptive
        self.sessions, self.queue, self.inflight = {}, [], []
        self.lock, self.compute_lock = threading.RLock(), threading.Lock()
        self.peak_charge = 0
        self.metrics = {"batches": 0, "rows": 0, "burned": 0, "size_dispatch": 0,
                        "deadline_dispatch": 0, "idle_dispatch": 0}
        if kernel is None:
            self.executor = MaskedGEMM(threads=1, build=False)
            if not self.executor.native or self.executor.backend != "rust":
                raise ValueError("installed native backend required; no Python fallback")
            self.matrix = self.executor.compile(self.weights)
            self.backend = self.executor.backend
            self.kernel = self.matrix.modular
        else:
            self.backend, self.kernel = "test-injected", kernel
        self._admit()

    def charge(self):
        # Conservative explicit reservation, not a Python allocator/RSS proof.
        out, inp = self.weights.shape
        live = {id(s): s for s in self.sessions.values()}
        live.update({id(r.generation): r.generation for r in self.inflight})
        return (2 * self.weights.nbytes + (1 << 20)
                + len(live) * 4096
                + sum(1024 + len(t.key.encoded()) + (len(t.correction) if t.correction else 0)
                      for s in live.values() for t in s.tickets.values())
                + (len(self.queue) + len(self.inflight)) * (2048 + inp * 4 + out * 4)
                + sum(len(r.key.encoded()) for r in self.queue + self.inflight)
                + self.batch_size * (inp * 12 + out * 20))

    def _admit(self):
        value = self.charge()
        if value > self.memory_limit:
            raise ValueError("memory admission exhausted")
        self.peak_charge = max(self.peak_charge, value)

    def public_key(self, *, stage="decode.linear", modulus=65521):
        return PublicKey("synthetic", stage, self.digest, "prime", modulus,
                         self.weights.shape[1], self.weights.shape[0])

    def _validate_key(self, key):
        if (key.model != "synthetic" or key.weight_digest != self.digest or key.ring != "prime"
                or type(key.modulus) is not int or not 2 <= key.modulus < 2**31
                or key.input_width != self.weights.shape[1]
                or key.output_width != self.weights.shape[0]
                or key.codec != "raw-u32-le-v1" or key.wire_bits != 32
                or key.weight_bits != 8 or key.activation_bits != 8
                or not key.stage or len(key.encoded()) > 2048):
            raise ValueError("unsupported public compatibility key")

    def open_session(self, session, secret):
        with self.lock:
            self.cleanup()
            if (not isinstance(session, str) or not 1 <= len(session.encode()) <= 128
                    or type(secret) is not bytes or len(secret) != 32):
                raise ValueError("invalid session credentials")
            if session in self.sessions or len(self.sessions) >= self.max_sessions:
                raise ValueError("session admission exhausted")
            self.sessions[session] = Session(secret, self.clock())
            try:
                self._admit()
            except ValueError:
                del self.sessions[session]
                raise

    def preload(self, session, ticket, key, correction):
        """Trusted Preparation fixture handoff, distinct from online auth."""
        with self.lock:
            self.cleanup()
            self._validate_key(key)
            s = self.sessions[session]
            if (not isinstance(ticket, str) or not 1 <= len(ticket.encode()) <= 128
                    or ticket in s.tickets or len(s.tickets) >= self.ticket_limit):
                raise ValueError("ticket admission exhausted")
            c = np.asarray(correction)
            if c.dtype != np.uint32 or c.shape != (key.output_width,) or np.any(c >= key.modulus):
                raise ValueError("malformed correction")
            s.tickets[ticket] = Ticket(key, c.tobytes())
            try:
                self._admit()
            except ValueError:
                del s.tickets[ticket]
                raise
            s.last_active = self.clock()

    def _burn(self, ticket):
        if ticket.state not in ("consumed", "burned"):
            self.metrics["burned"] += 1
            ticket.state, ticket.correction = "burned", None

    def submit(self, session, ticket, key, payload, mac, *, expires):
        with self.lock:
            self.cleanup()
            s = self.sessions.get(session)
            if (s is None or type(payload) is not bytes or type(mac) is not bytes
                    or len(payload) > self.weights.shape[1] * 4
                    or len(key.encoded()) > 2048
                    or not hmac.compare_digest(signature(s.secret, session, ticket, key, payload), mac)):
                raise ValueError("authentication rejected")
            t = s.tickets.get(ticket)
            if t is None or t.state != "prepared":
                raise ValueError("ticket replay or unknown")
            try:
                self._validate_key(key)
                if (t.key != key or len(payload) != key.input_width * 4
                        or np.any(np.frombuffer(payload, "<u4") >= key.modulus)
                        or not math.isfinite(expires) or expires <= self.clock()):
                    raise ValueError("malformed or expired row")
            except ValueError:
                self._burn(t)
                raise
            if s.pending or len(self.queue) + len(self.inflight) >= self.queue_capacity:
                raise ValueError("row admission exhausted")
            reply = Reply()
            row = Row(session, s, ticket, key, payload, self.clock(), expires, reply)
            self.queue.append(row)
            try:
                self._admit()
            except ValueError:
                self.queue.remove(row)
                raise
            s.pending, s.last_active, t.state = True, self.clock(), "queued"
            return reply

    def _fail(self, row, reason):
        row.reply.error, row.reply.done = reason, True
        s = self.sessions.get(row.session)
        if s is row.generation:
            self._burn(s.tickets[row.ticket])
            s.pending = False

    def cancel(self, session):
        with self.lock:
            for row in self.queue + self.inflight:
                if row.session == session:
                    self._fail(row, "session cancelled")
            self.queue = [r for r in self.queue if r.session != session]
            s = self.sessions.pop(session, None)
            if s:
                for t in s.tickets.values():
                    self._burn(t)

    def cleanup(self):
        with self.lock:
            now = self.clock()
            for sid, s in list(self.sessions.items()):
                if now - s.last_active >= self.idle:
                    self.cancel(sid)
            kept = []
            for row in self.queue:
                if row.expires <= now:
                    self._fail(row, "row expired")
                else:
                    kept.append(row)
            self.queue = kept

    def next_wake(self):
        with self.lock:
            self.cleanup()
            return min([r.entered + self.wait for r in self.queue]
                       + [r.expires for r in self.queue]
                       + [s.last_active + self.idle for s in self.sessions.values()], default=None)

    def dispatch(self):
        """One eligible group; oldest eligible group first, FIFO within group."""
        with self.compute_lock:
            with self.lock:
                self.cleanup()
                now = self.clock()
                groups = {}
                for row in self.queue:
                    groups.setdefault(row.key, []).append(row)
                eligible = [rows for rows in groups.values()
                            if len(rows) >= self.batch_size or now >= rows[0].entered + self.wait
                            or (self.adaptive and len(self.queue) == 1)]
                if not eligible:
                    return []
                rows = min(eligible, key=lambda items: items[0].entered)[:self.batch_size]
                reason = ("size_dispatch" if len(rows) == self.batch_size else
                          "idle_dispatch" if self.adaptive and len(self.queue) == 1 else "deadline_dispatch")
                self.metrics[reason] += 1
                corrections = []
                for r in rows:
                    t = self.sessions[r.session].tickets[r.ticket]
                    # All row checks/consume are atomic under owner lock; never
                    # roll back valid peers because another session is malformed.
                    corrections.append(np.frombuffer(t.correction, "<u4"))
                    t.state, t.correction = "consumed", None
                    r.reply.queue_seconds = now - r.entered
                selected = {id(r) for r in rows}
                self.queue = [r for r in self.queue if id(r) not in selected]
                self.inflight = rows
                self._admit()
            failure = None
            try:
                packed = np.stack([np.frombuffer(r.payload, "<u4") for r in rows])
                output = self.kernel(packed, rows[0].key.modulus)
                if output.shape != (len(rows), rows[0].key.output_width) or output.dtype != np.uint32:
                    raise ValueError("kernel result mismatch")
                combined = (output.astype(np.uint64) + np.stack(corrections)) % rows[0].key.modulus
            except Exception as exc:
                failure = "kernel failure: " + type(exc).__name__
            with self.lock:
                self.cleanup()
                for i, r in enumerate(rows):
                    s = self.sessions.get(r.session)
                    if not r.reply.done:
                        if failure or s is not r.generation or r.expires <= self.clock():
                            self._fail(r, failure or "row expired or session terminal")
                        else:
                            r.reply.output = combined[i].astype(np.uint32)
                            r.reply.mac = signature(s.secret, r.session, r.ticket, r.key,
                                                    b"reply\0" + r.reply.output.tobytes())
                            r.reply.done = True
                    if s is r.generation:
                        s.pending, s.last_active = False, self.clock()
                self.inflight = []
                self.metrics["batches"] += 1
                self.metrics["rows"] += len(rows)
                return [r.reply for r in rows]

    def close(self):
        """Retire sessions/material only; matrix/executor retire on owner release.

        Inflight work is not joined. Its cancelled generation stays charged until
        dispatch unwinds. Existing owner remains allocated and can admit sessions.
        """
        with self.lock:
            for sid in list(self.sessions):
                self.cancel(sid)


def fixture(owner, sid, ticket, rng, *, key=None, reference=True):
    """Independent fresh masks per request. Owner sees neither x nor r nor s."""
    key = key or owner.public_key()
    p = key.modulus
    x = rng.integers(0, 128, key.input_width, dtype=np.uint32)
    r = rng.integers(0, p, key.input_width, dtype=np.uint32)
    mask = rng.integers(0, p, key.output_width, dtype=np.uint32)
    wr = oracle(owner.weights, [r], p)[0] if reference else owner.matrix.modular(r[None], p)[0]
    correction = ((mask.astype(np.int64) - wr) % p).astype(np.uint32)
    owner.preload(sid, ticket, key, correction)
    payload = ((x.astype(np.uint64) + r) % p).astype("<u4").tobytes()
    mac = signature(owner.sessions[sid].secret, sid, ticket, key, payload)
    return key, payload, mac, x, mask


def benchmark(args):
    """Serial wall-clock cohorts, synthetic online work, no live SDK integration."""
    rng = np.random.default_rng(args.seed)
    weights = rng.integers(-8, 8, (args.outputs, args.inputs), dtype=np.int8)
    report = {"workload": vars(args), "cohorts": [], "scope": "synthetic online row owner only"}
    for size, load, adaptive in [(1, args.sessions, False), (args.batch, args.sessions, False),
                                 (1, 1, False), (args.batch, 1, False), (args.batch, 1, True)]:
        prep_wall, prep_cpu = time.perf_counter(), time.process_time()
        owner = BatchOwner(weights, batch_size=size, wait_seconds=args.wait_ms / 1000,
                           max_sessions=load, queue_capacity=load,
                           tickets_per_session=args.rounds, adaptive=adaptive)
        for i in range(load):
            owner.open_session(str(i), hashlib.sha256(f"synthetic-{i}".encode()).digest())
        # Preparation fixture outside online timing; no prompt/cache reuse.
        prepared = [[fixture(owner, str(i), str(n), rng, reference=False)
                     for i in range(load)] for n in range(args.rounds)]
        prep_wall, prep_cpu = time.perf_counter() - prep_wall, time.process_time() - prep_cpu
        latencies, waits, cpu = [], [], time.process_time()
        started = time.perf_counter()
        for n, requests in enumerate(prepared):
            submitted, replies = [], []
            for i, (key, payload, mac, _, _) in enumerate(requests):
                submitted.append(time.perf_counter())
                replies.append(owner.submit(str(i), str(n), key, payload, mac,
                                            expires=time.monotonic() + 5))
            while not all(r.done for r in replies):
                owner.dispatch()
                for i, r in enumerate(replies):
                    if r.done and submitted[i] is not None:
                        latencies.append(time.perf_counter() - submitted[i])
                        waits.append(r.queue_seconds)
                        submitted[i] = None
                if not all(r.done for r in replies):
                    wake = owner.next_wake()
                    time.sleep(max(0., wake - time.monotonic()))
            if any(r.error for r in replies):
                raise RuntimeError("synthetic reply failed")
        elapsed, cpu = time.perf_counter() - started, time.process_time() - cpu
        # Independent oracle outside timing, checking every complete masked reply
        # would retain outputs; retain/check last wave here, full coverage in tests.
        for r, (_, _, _, x, mask) in zip(replies, prepared[-1]):
            clear = ((r.output.astype(np.int64) - mask) % 65521).astype(np.uint32)
            np.testing.assert_array_equal(clear, oracle(weights, [x], 65521)[0])
        report["cohorts"].append({"batch": size, "sessions": load, "adaptive": adaptive,
            "backend": owner.backend, "rows": load * args.rounds, "wall_seconds": elapsed,
            "preparation_fixture_wall_seconds": prep_wall,
            "preparation_fixture_cpu_seconds": prep_cpu,
            "cpu_seconds": cpu, "rows_per_second": load * args.rounds / elapsed,
            "latency_p50_seconds": float(np.quantile(latencies, .5)),
            "latency_p95_seconds": float(np.quantile(latencies, .95)),
            "queue_p95_seconds": float(np.quantile(waits, .95)),
            "peak_owner_charge_bytes": owner.peak_charge, "metrics": owner.metrics.copy()})
        owner.close()
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["fixture_retention_upper_bytes"] = args.sessions * args.rounds * (8 * args.inputs + 4 * args.outputs + 2048)
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    report["source_hashes"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [root / "tests/test_masked_decode_microbatch_probe.py",
                     root / "python/pllm/runtime/server.py",
                     root / "python/pllm/runtime/masked_runtime.py",
                     root / "python/pllm/runtime/transformer_engine.py",
                     root / "python/pllm/runtime/correction_rendezvous.py",
                     root / "python/pllm/runtime/native.py"]}
    if owner.executor.library_path:
        report["native_extension_sha256"] = hashlib.sha256(owner.executor.library_path.read_bytes()).hexdigest()
        report["native_extension_path"] = str(owner.executor.library_path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=int, default=128)
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--sessions", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--wait-ms", type=float, default=.25)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if (not 1 <= min(args.inputs, args.outputs, args.sessions, args.batch, args.rounds)
            or max(args.inputs, args.outputs) > 1024 or args.sessions > 32
            or args.batch > 32 or args.rounds > 32):
        parser.error("bounded geometry required: dimensions <=1024, sessions/batch/rounds <=32")
    result = benchmark(args)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
