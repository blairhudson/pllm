"""Research-only one-slot native mask runahead; no provider or SDK registration.

Public fixture seeds are NOT production randomness. The caller transfers exclusive
ownership of an already-reserved lease and reports transport dispatch/completion.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import resource
import threading
import time

import numpy as np
from pllm.runtime.preparation_protocol import PreparationRequest, seeded_ring_profile
from pllm.runtime.transformer_client import PreparedInventory, PreparedInventoryLease, PreparedStageRows


class MaskRunahead:
    """One future stage/chunk, exclusive lease, drain-before-close semantics.

    This is not compatible with NativePreparedLinear's direct cursor consumption.
    Hooks exist only for deterministic tests, never provider calls. close() must
    run on the owner/control thread, not the executor thread.
    """

    def __init__(self, lease, order, *, chunk_rows, cap_bytes, base_peak_bytes,
                 stack_bytes=8 << 20, executor_bytes=64 << 10,
                 before_expand=lambda: None, after_expand=lambda: None,
                 on_closing=lambda: None):
        if (type(lease) is not PreparedInventoryLease
                or type(chunk_rows) is not int or not 1 <= chunk_rows <= 32
                or len(order) != len(set(order)) or set(order) != set(lease.stages)
                or lease._closed or lease._offsets
                or any(s._mask_rows is None or s.verification is not None
                       for s in lease.stages.values())):
            raise ValueError("requires untouched exclusive native unverified lease and public stage order")
        if any(type(v) is not int or v < 0 for v in
               (cap_bytes, base_peak_bytes, stack_bytes, executor_bytes)):
            raise ValueError("memory charges must be nonnegative integer bytes")
        if stack_bytes < 8 << 20 or executor_bytes < 64 << 10:
            raise ValueError("cannot undercharge executor or thread stack reserve")
        self.buffer_bound = max(4 * min(chunk_rows, lease.rows, s.request.rows)
                                * (s.request.in_features + s.request.out_features)
                                for s in lease.stages.values())
        # Native Vec -> PyBytes transient copy: <=2B; boundary copies/overlap
        # reserve another B. Base must already include current-stage ownership.
        self.charges = dict(fresh_buffer_bytes=self.buffer_bound,
                            copy_overlap_bytes=2 * self.buffer_bound,
                            executor_bytes=executor_bytes, thread_stack_bytes=stack_bytes,
                            base_peak_bytes=base_peak_bytes)
        self.required_bytes = sum(self.charges.values())
        if self.required_bytes > cap_bytes:
            raise ValueError("whole-client conservative cap exceeded")
        self.lease, self.order, self.chunk_rows = lease, tuple(order), chunk_rows
        self._before, self._after = before_expand, after_expand
        self._on_closing = on_closing
        self._lock, self._close_lock = threading.Lock(), threading.Lock()
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="mask_runahead")
        self._state, self._remote, self._armed = "open", None, False
        self._slot, self._serial = None, 0
        self.waste_bytes = self.expanded_bytes = self.delivered_bytes = 0

    def remote_started(self, stage):
        with self._lock:
            if self._state != "open" or self._remote is not None or stage not in self.order:
                raise RuntimeError("invalid remote window")
            self._remote, self._armed = stage, False

    def remote_finished(self):
        with self._lock:
            if self._state != "open" or self._remote is None:
                raise RuntimeError("no remote window")
            self._remote = None

    def prefetch(self, stage, count):
        with self._lock:
            index = self.order.index(self._remote) if self._remote is not None else -1
            if (self._state != "open" or index < 0 or index + 1 >= len(self.order)
                    or stage != self.order[index + 1] or self._armed or self._slot is not None
                    or type(count) is not int or not 1 <= count <= self.chunk_rows
                    or count > self.lease.rows):
                raise RuntimeError("prefetch requires one next-stage chunk in remote window")
            self._serial += 1
            token = self._serial
            size = 4 * count * (self.lease.stages[stage].request.in_features
                                + self.lease.stages[stage].request.out_features)
            future = self._pool.submit(self._expand, stage, count, size)
            self._slot = (token, stage, count, size, future)
            self._armed = True
            return token

    def _expand(self, stage, count, size):
        self._before()
        result = self.lease.take(stage, count)  # native burns before allocation
        with self._lock:
            self.expanded_bytes += size
        self._after()
        return result

    def consume(self, token):
        try:
            with self._lock:
                if (self._state != "open" or self._remote is not None
                        or self._slot is None or token != self._slot[0]):
                    raise RuntimeError("prefetched handle unavailable or replayed")
                slot = self._slot
            result = slot[4].result()
            with self._lock:
                if self._state != "open" or self._slot is not slot:
                    raise RuntimeError("prefetched material burned by cancellation or competing consume")
                self._slot = None
                self.delivered_bytes += slot[3]
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        with self._close_lock:
            with self._lock:
                if self._state == "closed":
                    return
                self._state = "closing"
            # Never cancel queued work and later reuse its reservation. Drain it,
            # then burn all unused rows through the original lease lifecycle.
            try:
                self._on_closing()
            finally:
                self._pool.shutdown(wait=True, cancel_futures=False)
                try:
                    self.lease.close()
                finally:
                    with self._lock:
                        if self._slot is not None:
                            self.waste_bytes += self._slot[3]  # conservative even on failure
                        self._slot = None
                        self._remote, self._state = None, "closed"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    @property
    def state(self):
        with self._lock:
            return self._state


def fixture(*, bits=24, rows=3, width=17, output=11):
    profile = seeded_ring_profile({16: 123, 24: 40000, 32: 1 << 24}[bits])
    stages = {}
    for index, name in enumerate(("current", "next", "last")):
        request = PreparationRequest(f"{index + 1:032x}", "mask-runahead-public", "fixture",
            "public-body", name, "public-weight", rows, width, output, 8, 8,
            profile.signed_output_bound, profile.ring, profile.modulus, profile.wire_bits,
            hashlib.sha256(f"public mask_runahead fixture {index}".encode()).digest())
        stages[name] = PreparedStageRows(request)
    return PreparedInventory("mask-runahead-public", rows, stages)


def digest(result):
    r, s, tickets = result
    return hashlib.sha256(r.tobytes() + s.tobytes() + "".join(tickets).encode()).hexdigest()


def bounded_timing(width=2048, output=4096, rows=4, repeats=3):
    """Measures native expansion, NOT network overlap or complete responses."""
    if not (1 <= width <= 19456 and 1 <= output <= 19456
            and 1 <= rows <= 32 and 1 <= repeats <= 5):
        raise ValueError("bounded timing shape/repeat limit")
    samples = []
    for mode in ("synchronous", "executor"):
        for _ in range(repeats):
            inv = fixture(rows=rows, width=width, output=output)
            lease = inv.reserve(rows)
            wrapper = None
            try:
                if mode == "executor":
                    wrapper = MaskRunahead(lease, inv.stages, chunk_rows=rows,
                        cap_bytes=268435456, base_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        * (1 if platform.system() == "Darwin" else 1024))
                wall, cpu = time.perf_counter_ns(), time.process_time_ns()
                if wrapper:
                    wrapper.remote_started("current")
                    token = wrapper.prefetch("next", rows)
                    wrapper.remote_finished()
                    result = wrapper.consume(token)
                else:
                    result = lease.take("next", rows)
                wall, cpu = time.perf_counter_ns() - wall, time.process_time_ns() - cpu
                samples.append(dict(mode=mode, wall_ns=wall, client_cpu_ns=cpu,
                                    mask_ticket_sha256=digest(result)))
                del result
            finally:
                wrapper.close() if wrapper else lease.close()
                inv.cancel()
    assert len({s["mask_ticket_sha256"] for s in samples}) == 1
    return dict(scope="native expansion only; no response gain or tier qualification",
                shape=[rows, width, output], samples=samples,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if platform.system() == "Darwin" else 1024))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true", required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--output-width", type=int, default=4096)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    report = bounded_timing(args.width, args.output_width, args.rows, args.repeats)
    report["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
