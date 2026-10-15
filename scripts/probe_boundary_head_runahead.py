"""Bounded research owner; not a production protocol or an SDK benchmark.

Compile synthetic worker matrices before online execution. Every issued mask uses
OS randomness, including measured runs. No Preparation, key expansion or reusable
mask pool is used online. Logical workers are co-located only for this oracle.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import secrets
import threading
import time

import numpy as np


@dataclass(frozen=True)
class Binding:
    context: str
    source: str
    stage: str
    step: int
    role: str = "B"

    def validate(self):
        if (not all(type(x) is str and x for x in (self.context, self.source, self.stage))
                or type(self.step) is not int or self.step < 0 or self.role not in ("A", "B")):
            raise ValueError("invalid binding")


@dataclass
class Reply:
    binding: Binding
    values: np.ndarray

    def destroy(self):
        self.values.fill(0)


def frame_bytes(binding, payload_bytes):
    """Research envelope: u32 header length + canonical JSON + u32 payload length.

    Not current HTTP frame format; includes all new ownership metadata.
    """
    return 8 + len(json.dumps(asdict(binding), sort_keys=True, separators=(",", ":")).encode()) + payload_bytes


class NativeRole:
    """One logical role with its own compiled matrix; no cross-role root/material.

    Research context/source are trusted caller commitments, not authenticated wire
    authority. A deployed worker must derive these from admitted descriptor/session.
    """
    def __init__(self, compiled, authority, role):
        self.compiled = compiled
        self.authority = replace(authority, role=role)
        self.next_step = 0
        self.lock = threading.Lock()

    def __call__(self, binding, share):
        binding.validate()
        with self.lock:
            if binding != replace(self.authority, step=self.next_step):
                raise ValueError("worker binding/replay differs")
            self.next_step += 1  # Burn before native work; never retry.
            # Installed native API returns immutable bytes-backed arrays. Own a
            # mutable copy for deterministic wiping; original temporary is released,
            # not securely erased (an explicit production integration gate).
            return Reply(binding, self.compiled.wrap32(share).copy())


class OneAhead:
    """Single caller, at most one B future and one mask; terminal on any error.

    close() may run concurrently to signal cancellation, then drains native work.
    Callbacks MUST terminate (production needs HTTP/native deadline). Python cannot
    forcibly cancel running native work. No callback may recursively call owner.
    """
    def __init__(self, authority, hidden, vocabulary, worker_a, worker_b):
        authority.validate()
        if authority.role != "B" or authority.step != 0 or hidden <= 0 or vocabulary <= 0:
            raise ValueError("invalid owner geometry/authority")
        self.authority, self.hidden, self.vocabulary = authority, hidden, vocabulary
        self.a, self.b = worker_a, worker_b
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.RLock()
        self.api = threading.RLock()
        self.closed = False
        self.step = 0
        self.slot = None
        self.stats = dict(issued_b=0, consumed=0, unused_b=0, upload_bytes=0,
                          download_bytes=0, worker_cpu_seconds=0.0, mask_cpu_seconds=0.0,
                          drainage_seconds=0.0, peak_futures=0)

    def _call(self, worker, binding, share):
        start = time.thread_time()
        try:
            reply = worker(binding, share)
            if isinstance(reply, Reply):
                with self.lock:
                    self.stats["download_bytes"] += frame_bytes(reply.binding, reply.values.nbytes)
            return reply
        finally:
            with self.lock:
                self.stats["worker_cpu_seconds"] += time.thread_time() - start
                self.stats["upload_bytes"] += frame_bytes(binding, share.nbytes)
                # Native oracle has no partial network responses; failed wire
                # coverage is unknown, not promoted to complete accounting.

    def _validate(self, reply, binding):
        if (not isinstance(reply, Reply) or reply.binding != binding
                or reply.values.dtype != np.dtype("uint32")
                or reply.values.shape != (1, self.vocabulary)):
            if isinstance(reply, Reply):
                reply.destroy()
            raise ValueError("reply substitution/geometry differs")
        return reply

    def issue(self, binding):
        """Admit CURRENT head after its embedding lookup, before its body.

        Do not issue the next head until its lookup and decision to execute body.
        Normal EOS/cap termination therefore has no unconsumed reservation.
        """
        with self.api:
            with self.lock:
                if self.closed or self.slot is not None or binding != replace(self.authority, step=self.step):
                    raise ValueError("owner closed, occupied or binding differs")
                start = time.thread_time()
                mask = np.frombuffer(bytearray(secrets.token_bytes(4 * self.hidden)), "<u4").reshape(1, -1)
                self.stats["mask_cpu_seconds"] += time.thread_time() - start
                future = self.pool.submit(self._call, self.b, binding, mask)
                self.slot = (binding, mask, future)
                self.stats["issued_b"] += 1
                self.stats["peak_futures"] = 1

    def consume(self, binding, quantized):
        with self.api:
            left = ar = br = None
            try:
                with self.lock:
                    if self.closed or self.slot is None or binding != self.slot[0]:
                        raise ValueError("head ownership differs")
                    _, mask, future = self.slot
                q = np.asarray(quantized)
                if q.dtype != np.int8 or q.shape != (1, self.hidden):
                    raise ValueError("head requires one quantized i8 row")
                left = np.subtract(q.astype(np.uint32), mask, dtype=np.uint32)
                # B is already running; A progresses independently. No second future.
                ab = replace(binding, role="A")
                ar = self._validate(self._call(self.a, ab, left), ab)
                br = self._validate(future.result(), binding)
                result = np.add(ar.values, br.values, dtype=np.uint32).view(np.int32).copy()
                with self.lock:
                    if self.closed:
                        result.fill(0)
                        raise ValueError("cancelled before delivery")
                    self.stats["consumed"] += 1
                    self.step += 1
                    self.slot = None
                mask.fill(0)
                return result
            except BaseException:
                self.close()
                raise
            finally:
                if left is not None:
                    left.fill(0)
                for reply in (ar, br):
                    if reply is not None:
                        reply.destroy()

    def close(self):
        with self.lock:
            self.closed = True  # Signal before waiting for caller/native work.
        with self.api:
            start = time.perf_counter()
            if self.slot is not None:
                binding, mask, future = self.slot
                # Do not cancel queued work: all issued work executes and is charged.
                try:
                    reply = future.result()
                    self._validate(reply, binding).destroy()
                except Exception:
                    pass
                finally:
                    mask.fill(0)
                    self.slot = None
                    self.stats["unused_b"] += 1
            self.pool.shutdown(wait=True)
            self.stats["drainage_seconds"] += time.perf_counter() - start


class ReadyParallel(OneAhead):
    """Native owner control: fresh B dispatch only once q is already ready.

    Same one-future ownership, arithmetic, drainage and counters as runahead;
    caller executes A while B future runs. Actual concurrency depends on native
    GIL/resource behavior, not assumed to produce a wall-time speedup.
    """
    def execute(self, binding, quantized):
        try:
            self.issue(binding)
            return self.consume(binding, quantized)
        except BaseException:
            self.close()
            raise


def schedule(body_ms, a_ms, b_ms, steps, unused=0):
    """Admitted-step pure model. Optional unused B is a cancellation ablation."""
    if (not all(math.isfinite(x) and x >= 0 for x in (body_ms, a_ms, b_ms))
            or steps < 1 or unused not in (0, 1)):
        raise ValueError("invalid modeled window")
    serial = steps * (body_ms + a_ms + b_ms)
    parallel = steps * (body_ms + max(a_ms, b_ms))
    ahead = steps * max(body_ms + a_ms, b_ms) + unused * b_ms
    return dict(label="modeled_not_measured_sdk", body_window_ms=body_ms,
                a_service_ms=a_ms, b_service_ms=b_ms,
                ready_serial_ms=serial, ready_parallel_ms=parallel,
                one_ahead_including_unused_tail_ms=ahead,
                gain_vs_ready_parallel_ms=parallel - ahead,
                case="admitted_current_step" if not unused else "unused_tail_ablation",
                issued_b=steps + unused, unused_b=unused,
                charged_worker_ms=steps * a_ms + (steps + unused) * b_ms)


def resources(hidden, vocabulary):
    return dict(mask_bytes=4 * hidden, retained_b_result_bytes=4 * vocabulary,
                share_payload_bytes=4 * hidden, reply_payload_bytes=4 * vocabulary,
                two_workers_matrix_bytes=2 * hidden * vocabulary,
                owner_array_peak_upper_bytes=12 * hidden + 16 * vocabulary,
                note="array bound excludes quantization, framing copies, native scratch, scales, allocator, RSS; matrix not client-owned in deployment")


def provenance():
    from pllm import _native
    import pllm.runtime.native as native_wrapper
    import pllm.runtime.quantization as quantization

    def artifact(path):
        path = Path(path).resolve()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)

    return dict(probe_source=artifact(__file__), native_build=artifact(_native.__file__),
                installed_native_wrapper=artifact(native_wrapper.__file__),
                installed_quantization=artifact(quantization.__file__), numpy_version=np.__version__)


def run(vocabulary=257, hidden=32, steps=3, repeats=1, windows=(0, 10, 100),
        service_windows=((10, 10), (5, 15), (15, 5)), unused_tail=False):
    from pllm.runtime.native import MaskedGEMM
    from pllm.runtime.quantization import quantize_activation_per_row, dequantize_matmul
    if not (1 <= vocabulary <= 4096 and 1 <= hidden <= 1024 and 1 <= steps <= 16 and 1 <= repeats <= 10):
        raise ValueError("bounded synthetic probe only")
    rng = np.random.default_rng(61202)  # Public fixture only; NEVER mask entropy.
    weights = rng.integers(-127, 128, (vocabulary, hidden), dtype=np.int8)
    authority = Binding(secrets.token_hex(16), hashlib.sha256(weights.tobytes()).hexdigest(), "lm_head", 0)
    # Preparation/compilation outside online timing, independent native role roots.
    kernel = MaskedGEMM(threads=1)
    roots = [kernel.compile(weights.copy()) for _ in range(2)]
    scales = rng.uniform(.001, .1, vocabulary).astype(np.float32)
    rows = rng.standard_normal((steps, hidden), dtype=np.float32)
    modes = {}
    native_ms = []
    ablations = {}
    requested_modes = ("ready_serial", "ready_parallel", "one_ahead")
    if unused_tail:
        requested_modes += ("one_ahead_unused_tail",)
    for mode in requested_modes:
        samples = []
        for _ in range(repeats):
            a, b = (NativeRole(root, authority, role) for root, role in zip(roots, ("A", "B")))
            cpu_start = time.process_time()
            if mode == "ready_serial":
                ledger = dict(issued_a=steps, issued_b=steps, unused_b=0,
                              upload_bytes=0, download_bytes=0)
                start = time.perf_counter()
                for step, row in enumerate(rows):
                    q = quantize_activation_per_row(row.reshape(1, -1), bits=8)
                    mask = np.frombuffer(bytearray(secrets.token_bytes(hidden * 4)), "<u4").reshape(1, -1)
                    left = np.subtract(q.values.astype(np.uint32), mask, dtype=np.uint32)
                    bind = replace(authority, step=step)
                    parts = []
                    for worker, role, share in ((a, "A", left), (b, "B", mask)):
                        tick = time.perf_counter()
                        parts.append(worker(replace(bind, role=role), share))
                        native_ms.append((time.perf_counter() - tick) * 1000)
                        ledger["upload_bytes"] += frame_bytes(replace(bind, role=role), hidden * 4)
                        ledger["download_bytes"] += frame_bytes(replace(bind, role=role), vocabulary * 4)
                    output = np.add(parts[0].values, parts[1].values, dtype=np.uint32).view(np.int32)
                    _oracle(output, q, weights, scales, dequantize_matmul)
                    for part in parts:
                        part.destroy()
                    mask.fill(0)
                    left.fill(0)
                samples.append({"wall_seconds": time.perf_counter() - start,
                                "process_cpu_seconds_including_oracle": time.process_time() - cpu_start,
                                "ledger": ledger})
            else:
                owner_class = ReadyParallel if mode == "ready_parallel" else OneAhead
                owner = owner_class(authority, hidden, vocabulary, a, b)
                start = time.perf_counter()
                try:
                    for step, row in enumerate(rows):
                        binding = replace(authority, step=step)
                        # Synthetic row stands in for current admitted body output.
                        # No embedding lookup/body timing is measured here. B never
                        # runs for a future step before its lookup/admission.
                        if mode != "ready_parallel":
                            owner.issue(binding)
                        q = quantize_activation_per_row(row.reshape(1, -1), bits=8)
                        output = (owner.execute(binding, q.values) if mode == "ready_parallel"
                                  else owner.consume(binding, q.values))
                        _oracle(output, q, weights, scales, dequantize_matmul)
                    if mode == "one_ahead_unused_tail":
                        # Explicit cancellation after admitting another body step,
                        # not normal EOS/cap behavior and not candidate overhead.
                        owner.issue(replace(authority, step=steps))
                finally:
                    owner.close()
                samples.append({"wall_seconds": time.perf_counter() - start,
                                "process_cpu_seconds_including_oracle": time.process_time() - cpu_start,
                                "ledger": owner.stats})
        (ablations if mode == "one_ahead_unused_tail" else modes)[mode] = samples
    mean_native = sum(native_ms) / len(native_ms)
    return dict(schema="boundary-head-runahead-research/v1", scope="synthetic native arithmetic; no body, HTTP or SDK timing",
                shape=[vocabulary, hidden], exact_full_logits=True, masks="fresh OS entropy per issue, including timing",
                provenance=provenance(), fixture_weight_sha256=authority.source,
                measured_synthetic=modes, measured_ablations=ablations,
                modeled=[schedule(w, a, b, steps) for a, b in service_windows for w in windows],
                modeled_assumptions="explicit hypothetical A/B service windows, not measured native or SDK service",
                modeled_native_equal_service=[schedule(w, mean_native, mean_native, steps) for w in windows],
                modeled_ablations=([schedule(w, a, b, steps, unused=1)
                                    for a, b in service_windows for w in windows] if unused_tail else []),
                key_negative_result="Equal independent A/B service: max(T+a,b) = T+a = T+max(a,b); zero modeled gain over ready-parallel for all T >= 0.",
                native_ready_call_wall_ms=native_ms,
                resources=resources(hidden, vocabulary), qwen_geometry_resources=resources(896, 151936),
                timing_caveat="includes oracle cost and thread scheduling; co-located logical workers, not serving speedup",
                state_caveat="KV/token/RNG not executed or modified; integration equivalence remains gate")


def _oracle(output, q, weights, scales, dequantize):
    expected = q.values.astype(np.int64) @ weights.astype(np.int64).T
    if not np.array_equal(output.astype(np.int64), expected):
        raise AssertionError("native modular integer oracle differs")
    # Same float32 order as runtime; no head decomposition after dequantization.
    actual = dequantize(output, q.scales, scales)
    reference = dequantize(expected, q.scales, scales)
    if actual.tobytes() != reference.tobytes():
        raise AssertionError("full float32 logits differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocabulary", type=int, default=257)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--body-window-ms", type=float, nargs="+", default=[0, 10, 100])
    parser.add_argument("--service-window-ms", type=float, nargs=2, action="append", metavar=("A", "B"),
                        help="repeat for explicit modeled asymmetric A/B service times")
    parser.add_argument("--unused-tail", action="store_true", help="separate charged cancellation ablation")
    args = parser.parse_args()
    print(json.dumps(run(args.vocabulary, args.hidden, args.steps, args.repeats, args.body_window_ms,
                         args.service_window_ms or ((10, 10), (5, 15), (15, 5)), args.unused_tail), sort_keys=True))


if __name__ == "__main__":
    main()
