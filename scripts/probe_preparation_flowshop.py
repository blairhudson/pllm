#!/usr/bin/env python3
"""Bounded scheduling model and test-local one-use lifecycle; no SDK activation."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
import hashlib
import hmac
import importlib.util
import itertools
import json
import math
from pathlib import Path
import secrets
import statistics
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WINDOW_SOURCE = ROOT / "python/pllm/runtime/preparation_window.py"
spec = importlib.util.spec_from_file_location("flowshop_window_source", WINDOW_SOURCE)
window = importlib.util.module_from_spec(spec)
spec.loader.exec_module(window)


@dataclass(frozen=True)
class Job:
    stage: str
    compute: float
    upload: float  # Includes full ACK latency, not merely socket transmission.
    work: int
    deadline: float | None = None


def validate(jobs, width, budget, order):
    if not jobs or len({j.stage for j in jobs}) != len(jobs):
        raise ValueError("empty or duplicate stages")
    window.validate_preparation_window(width=width, max_bytes=budget)
    # Unlike baseline width=1 exemption, this reference rejects oversize work.
    for j in jobs:
        if (type(j.work) is not int or not 0 < j.work <= budget
                or not all(math.isfinite(t) and t > 0 for t in (j.compute, j.upload))
                or (j.deadline is not None and not (math.isfinite(j.deadline) and j.deadline > 0))):
            raise ValueError("invalid time/work/deadline or oversize job")
    if sorted(order) != list(range(len(jobs))):
        raise ValueError("order must be exact permutation")


def priority(jobs):
    """Johnson-inspired heuristic only: parallel compute/retained slots invalidate theorem."""
    early = sorted((i for i, j in enumerate(jobs) if j.compute <= j.upload),
                   key=lambda i: (jobs[i].compute, i))
    late = sorted((i for i, j in enumerate(jobs) if j.compute > j.upload),
                  key=lambda i: (-jobs[i].upload, i))
    return early + late


def simulate(jobs, order, *, width=4, budget=16 << 20, max_wait=None):
    """Discrete events: strict issuance list, parallel ideal compute, one FIFO ACK lane.

    No hidden bypass: head not fitting stalls admission, exactly like source window.
    Slots/work remain charged while computed output waits or push awaits ACK.
    All jobs released at time zero. Completion of all ACKs is earliest seal time.
    """
    validate(jobs, width, budget, order)
    if max_wait is not None and (not math.isfinite(max_wait) or max_wait < 0):
        raise ValueError("invalid max wait")
    t, next_job, retained, peak_bytes, peak_slots = 0.0, 0, 0, 0, 0
    active, ready, lane = {}, [], None
    starts, ends, accepted, events = {}, {}, [], []
    while len(ends) < len(jobs):
        if lane is not None and lane[1] <= t:
            i, _ = lane
            ends[i] = t
            accepted.append(i)
            retained -= jobs[i].work
            del active[i]
            events.append([t, "ack", jobs[i].stage])
            lane = None
        for i, finish in sorted(active.items(), key=lambda pair: (pair[1], pair[0])):
            if finish <= t and i not in ready and i not in ends and (lane is None or lane[0] != i):
                ready.append(i)
        while next_job < len(order) and len(active) < width:
            i = order[next_job]
            if retained + jobs[i].work > budget:
                break
            starts[i] = t
            active[i] = t + jobs[i].compute
            retained += jobs[i].work
            next_job += 1
            events.append([t, "issue", jobs[i].stage])
            peak_bytes = max(peak_bytes, retained)
            peak_slots = max(peak_slots, len(active))
        if lane is None and ready:
            i = ready.pop(0)
            lane = (i, t + jobs[i].upload)
            events.append([t, "push", jobs[i].stage])
        future = [finish for i, finish in active.items()
                  if finish > t and (lane is None or lane[0] != i)]
        if lane is not None:
            future.append(lane[1])
        if len(ends) == len(jobs):
            break
        if not future:
            raise RuntimeError("scheduler stalled")
        t = min(future)
    misses = [j.stage for i, j in enumerate(jobs)
              if j.deadline is not None and ends[i] > j.deadline]
    waits = [j.stage for i, j in enumerate(jobs) if max_wait is not None and starts[i] > max_wait]
    busy = sum(j.upload for j in jobs)
    return {"modeled_seconds": t, "upload_busy_seconds": busy,
            "upload_idle_seconds": t - busy, "peak_work_bytes": peak_bytes,
            "peak_stage_slots": peak_slots, "issue_order": [jobs[i].stage for i in order],
            "accepted_order": [jobs[i].stage for i in accepted],
            "start_seconds": {jobs[i].stage: starts[i] for i in starts},
            "ack_seconds": {jobs[i].stage: ends[i] for i in ends},
            "deadline_misses": misses, "max_wait_violations": waits,
            "feasible": not misses and not waits, "events": events}


def exhaustive(jobs, *, width=4, budget=16 << 20, max_wait=None):
    """Exact optimum over <=8 strict eager list schedules, NOT all possible schedules."""
    if len(jobs) > 8:
        raise ValueError("exhaustive oracle bounded to eight jobs")
    best, best_order, count = None, None, 0
    for order in itertools.permutations(range(len(jobs))):
        result = simulate(jobs, order, width=width, budget=budget, max_wait=max_wait)
        count += 1
        if result["feasible"] and (best is None or result["modeled_seconds"] < best["modeled_seconds"]):
            best, best_order = result, order
    return {"scope": "all strict eager issuance permutations; FIFO completion upload",
            "enumerated": count, "order": best_order, "best": best}


def tick_cost(jobs, order, *, width, budget):
    """Independent integer-time oracle: decrement clocks, scan ready timestamps.

    Deliberately no event-jump machinery or calls to simulate.
    """
    if any(type(j.compute) is not int or type(j.upload) is not int for j in jobs):
        raise ValueError("tick oracle needs integer durations")
    todo, running, ready, ack_lane = list(order), {}, {}, None
    used, done, tick = 0, 0, 0
    while done < len(jobs):
        if tick:
            for i in list(running):
                running[i] -= 1
                if running[i] == 0:
                    ready[i] = tick
                    del running[i]
            if ack_lane is not None:
                i, remaining = ack_lane
                ack_lane = (i, remaining - 1)
                if ack_lane[1] == 0:
                    used -= jobs[i].work
                    ack_lane = None
                    done += 1
        while todo and len(running) + len(ready) + (ack_lane is not None) < width:
            i = todo[0]
            if used + jobs[i].work > budget:
                break
            todo.pop(0)
            running[i] = jobs[i].compute
            used += jobs[i].work
        if ack_lane is None and ready:
            i = min(ready, key=lambda k: (ready[k], k))
            del ready[i]
            ack_lane = (i, jobs[i].upload)
        if done < len(jobs):
            tick += 1
        if tick > 100000:
            raise RuntimeError("tick oracle stalled")
    return tick


@dataclass(frozen=True)
class Binding:
    session: str
    attempt: str
    stage: str
    plan: str
    weight_digest: str
    rows: int
    in_features: int
    out_features: int
    ring: str
    seed_commitment: str

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class Material:
    binding: Binding
    payload: bytearray
    burned: bool = False

    def burn(self):
        self.payload[:] = b"\x00" * len(self.payload)
        self.payload.clear()
        self.burned = True


class LocalLedger:
    """Test-local HMAC/one-use receiver; not independent-party protocol authentication."""
    def __init__(self, bindings):
        if len({b.stage for b in bindings}) != len(bindings) or len({b.attempt for b in bindings}) != len(bindings):
            raise ValueError("duplicate stage/attempt")
        if len({b.session for b in bindings}) != 1:
            raise ValueError("mixed sessions")
        self.expected = {b.stage: b for b in bindings}
        self.key = secrets.token_bytes(32)
        self.reserved = set()
        self.accepted, self.acked, self.materials = [], [], []
        self.state = "OPEN"

    def reserve(self, binding):
        if (self.state != "OPEN" or binding != self.expected.get(binding.stage)
                or binding.stage in self.reserved):
            raise ValueError("one-use reservation violation")
        self.reserved.add(binding.stage)

    def commitment(self, material, seq):
        body = f"push:{seq}:{material.binding.digest()}:{len(material.payload)}:".encode()
        return hmac.new(self.key, body + material.payload, "sha256").digest()

    def push(self, material, seq):
        b = material.binding
        if (self.state != "OPEN" or material.burned or seq != len(self.accepted)
                or b != self.expected.get(b.stage) or b.stage not in self.reserved
                or b.stage in self.accepted):
            raise ValueError("one-use binding/accepted-order violation")
        self.accepted.append(b.stage)
        self.materials.append(material)
        return hmac.new(self.key, b"ack:" + self.commitment(material, seq), "sha256").digest()

    def acknowledge(self, material, seq, ack):
        expected = hmac.new(self.key, b"ack:" + self.commitment(material, seq), "sha256").digest()
        if (self.state != "OPEN" or seq != len(self.acked)
                or self.accepted[seq:seq + 1] != [material.binding.stage]
                or not hmac.compare_digest(expected, ack)):
            raise ValueError("wrong/replayed/reordered ACK")
        self.acked.append(material.binding.stage)

    def seal(self, accepted_order):
        if (self.state != "OPEN" or accepted_order != self.accepted
                or self.acked != self.accepted or set(self.accepted) != set(self.expected)):
            raise ValueError("incomplete ACKs or wrong accepted-order seal")
        self.state = "READY"

    def abort(self):
        self.state = "BURNED"
        self.key = b""
        for material in self.materials:
            material.burn()


def run_local(jobs, bindings, order, *, compute, ledger, width=4,
              budget=16 << 20, cancelled=lambda: False, timeout=5, push=None,
              max_wait=None):
    """Real bounded fake-transport execution. Abort first, drain, burn ALL issued results.

    Callback compute(binding, stop) must cooperate with cancellation/deadline and
    burn private partial material before raising (caller cannot recover locals).
    Python cannot kill a hung thread. Parent must use process isolation for real peers.
    Deadline is checked before issuance, before push, and after blocking push/ACK.
    """
    validate(jobs, width, budget, order)
    if ([b.stage for b in bindings] != [j.stage for j in jobs]
            or any(ledger.expected.get(b.stage) != b for b in bindings)):
        raise ValueError("job/ledger binding mismatch")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid timeout")
    if max_wait is not None and (not math.isfinite(max_wait) or max_wait < 0):
        raise ValueError("invalid max wait")
    began = time.monotonic()
    stop, deadline = threading.Event(), began + timeout
    pool = ThreadPoolExecutor(max_workers=width, thread_name_prefix="flowshop-local")
    pending, issued, used, next_index = {}, [], 0, 0
    accepted, peak_bytes, peak_slots = [], 0, 0
    finish_order, finish_lock = [], threading.Lock()

    def completed(future):
        with finish_lock:
            finish_order.append(future)

    def check():
        if cancelled() or time.monotonic() >= deadline:
            raise RuntimeError("cancelled or deadline expired")

    try:
        while pending or next_index < len(order):
            check()
            while len(pending) < width and next_index < len(order):
                i = order[next_index]
                if used + jobs[i].work > budget:
                    break
                check()
                if max_wait is not None and time.monotonic() - began > max_wait:
                    raise RuntimeError("maximum issuance wait expired")
                ledger.reserve(bindings[i])
                future = pool.submit(compute, bindings[i], stop)
                pending[future] = i
                issued.append(future)
                future.add_done_callback(completed)
                used += jobs[i].work
                next_index += 1
                peak_bytes = max(peak_bytes, used)
                peak_slots = max(peak_slots, len(pending))
            wait(pending, timeout=0.01, return_when=FIRST_COMPLETED)
            with finish_lock:
                ready = [f for f in finish_order if f in pending]
            if not ready:
                continue
            future = ready[0]
            material = future.result()
            i = pending[future]
            if not isinstance(material, Material):
                raise ValueError("worker returned invalid material")
            if material.binding != bindings[i] or material.burned:
                raise ValueError("worker substituted material")
            if len(material.payload) > jobs[i].work:
                raise ValueError("returned payload exceeds declared work")
            check()
            if jobs[i].deadline is not None and time.monotonic() - began > jobs[i].deadline:
                raise RuntimeError("stage deadline expired")
            seq = len(accepted)
            ack = (push or ledger.push)(material, seq)
            check()
            if jobs[i].deadline is not None and time.monotonic() - began > jobs[i].deadline:
                raise RuntimeError("stage deadline expired")
            ledger.acknowledge(material, seq, ack)
            accepted.append(material.binding.stage)
            # Test-local receiver keeps only commitments/one-use metadata. Real
            # correction inventory must have separate charged receiver ownership.
            material.burn()
            used -= jobs[i].work
            del pending[future]
        check()
        ledger.seal(accepted)
        return {"accepted_order": accepted, "peak_work_bytes": peak_bytes,
                "peak_stage_slots": peak_slots, "state": ledger.state}
    except BaseException:
        stop.set()
        for future in pending:
            future.cancel()
        try:
            ledger.abort()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for future in issued:
                if not future.cancelled() and future.exception() is None:
                    value = future.result()
                    if isinstance(value, Material):
                        value.burn()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def source_jobs(rows, bandwidth, ack_latency):
    """Replay prior measured *local* phases, not measurements of flowshop transport."""
    path = ROOT / "docs/evidence/research-preparation-factory.json"
    data = json.loads(path.read_text())
    jobs, provenance = [], []
    for role in ("qkv_projection", "mlp_gate_up", "mlp_down"):
        samples = [s for s in data["samples"] if s["spec"]["role"] == role
                   and s["spec"]["rows"] == rows and s["spec"]["backend"] == "digit8"
                   and s["spec"]["layout"] == "separate"]
        if not samples:
            raise ValueError(f"missing prior measured stage {role}/{rows}")
        shapes = {tuple(s["shape"]) for s in samples}
        if len(shapes) != 1:
            raise ValueError("source geometry changed")
        out_features, in_features = shapes.pop()
        services = [v for s in samples for v in s["services"]]
        compute = statistics.median(v["mask_seconds"] + v["kernel_seconds"] for v in services)
        local = statistics.median(v["encoding_seconds"] for v in services)
        encoded = statistics.median(v["encoded_bytes"] for v in services)
        work = window.preparation_job_work_bytes(rows=rows, in_features=in_features,
                                                  out_features=out_features)
        for layer in range(2):
            jobs.append(Job(f"layer{layer}:{role}", compute,
                            local + encoded / bandwidth + ack_latency, work))
        provenance.append({"role": role, "shape_out_in": [out_features, in_features],
                           "rows": rows, "measured_samples": len(samples),
                           "measured_local_phase_records": len(services),
                           "prior_mask_plus_kernel_median_seconds": compute,
                           "prior_encoding_decode_file_io_median_seconds": local,
                           "prior_encoded_bytes_median": encoded,
                           "source": samples[0]["source"], "declared_work_bytes": work})
    # Source role order per layer rather than role-major order.
    jobs.sort(key=lambda j: (int(j.stage[5]), ["qkv_projection", "mlp_gate_up", "mlp_down"].index(j.stage.split(":")[1])))
    return jobs, provenance


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def report():
    replay = []
    for rows, budget in ((16, 16 << 20), (150, 64 << 20)):
        for bandwidth, latency in ((10_000_000, 0.001), (100_000_000, 0.01)):
            jobs, provenance = source_jobs(rows, bandwidth, latency)
            source = simulate(jobs, list(range(len(jobs))), budget=budget)
            candidate = simulate(jobs, priority(jobs), budget=budget)
            oracle = exhaustive(jobs, budget=budget)
            replay.append({"rows": rows, "width": 4, "budget": budget,
                           "simulated_bandwidth_bytes_per_second": bandwidth,
                           "simulated_full_ack_latency_seconds": latency,
                           "phase_provenance": provenance, "jobs": [asdict(j) for j in jobs],
                           "source_order": source, "heuristic": candidate, "oracle": oracle})
    tiny = []
    for name, times in (("upload_busy_gain", [(1, 1), (2, 1), (2, 2)]),
                        ("reorder_can_lose", [(1, 1), (2, 1), (1, 1)]),
                        ("long_compute_hidden_by_source", [(8, 1), (1, 8), (2, 4), (7, 1)])):
        jobs = [Job(str(i), c, u, 1 << 20) for i, (c, u) in enumerate(times)]
        oracle = exhaustive(jobs, width=2, budget=2 << 20)
        for order in itertools.permutations(range(len(jobs))):
            assert simulate(jobs, order, width=2, budget=2 << 20)["modeled_seconds"] == tick_cost(
                jobs, order, width=2, budget=2 << 20)
        tiny.append({"name": name, "jobs": [asdict(j) for j in jobs],
                     "independently_checked_orders": math.factorial(len(jobs)),
                     "source": simulate(jobs, list(range(len(jobs))), width=2, budget=2 << 20),
                     "heuristic": simulate(jobs, priority(jobs), width=2, budget=2 << 20),
                     "oracle": oracle})
    paths = [Path(__file__), ROOT / "tests/test_preparation_flowshop_probe.py", WINDOW_SOURCE,
             ROOT / "python/pllm/runtime/client.py", ROOT / "python/pllm/runtime/preparation_server.py",
             ROOT / "docs/evidence/research-preparation-factory.json",
             ROOT / "docs/evidence/research-correction-stripes-cohort.json",
             ROOT / "docs/research/preparation-flowshop-reference.md"]
    return {"schema": "pllm.preparation_flowshop.v1", "scope": "simulated scheduling; prior local measured phase replay",
            "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in paths},
            "replay": replay, "tiny": tiny,
            "unknown": {k: None for k in ("real_transport_contention", "concurrent_compute_cost",
                       "preparation_cpu_rss", "whole_client_cpu_rss", "TTFT", "decode_throughput",
                       "all_link_traffic", "physical_wire_bytes", "provider_completion")},
            "production_activation": False, "tier_qualification": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = report()
    text = json.dumps(result, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
