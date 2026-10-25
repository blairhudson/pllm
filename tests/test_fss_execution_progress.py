"""Cached progress on real mocked lifecycle/exec FD; no native/model execution."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from pllm.runtime import benchmark_failure as failure, fss_child_status as status
from pllm.runtime import fss_worker as worker
from pllm.runtime.servers import LocalTopology, _spawn_role_process
from test_fss_worker import cluster, double, Routing  # noqa: F401
from test_fss_benchmark import contract as price_contract, experiment  # noqa: F401


def legacy(generation=1, sequence=0, kind=status._UPDATE):
    return status._RECORD.pack(status._MAGIC, generation, sequence, kind, 0)


def packet(*, generation=1, sequence=0, kind=status._UPDATE, stage=4, words=None):
    return status._PROGRESS_RECORD.pack(status._PROGRESS_MAGIC, generation, sequence,
        kind, stage, *(words or (2, 3, 4, 8, 5, 5, 1_000_000_000, 900_000_000, 100)))


def test_cached_public_gauge_coalesces_and_counters_are_not_import_claims():
    class Writer:
        generation = 1
        calls = []
        def progress(self, *args): self.calls.append(args)
    writer = Writer()
    progress = status.ExecutionProgress(writer)
    attempt = SimpleNamespace(peer_sent=2, peer_received=3)
    meta = {"kernel": {"ordinal": 4}, "pc": 9}
    for _ in range(1000):
        progress.enter("tape", meta)
        progress.enter("native")
    assert writer.calls == []
    progress.returned = 8
    progress.sample(attempt)
    assert len(writer.calls) == 1
    assert progress.cached["materials_returned"] == 8
    assert progress.cached["kernel"] == 4 and progress.cached["pc"] == 9
    assert progress.cached["frame"] is None
    assert progress.cached["peer_acked"] == 2 and progress.cached["peer_received"] == 3
    assert "import" not in repr(progress.cached)
    progress.sample(attempt, status._FAIL)
    frozen = progress.cached.copy()
    progress.enter("complete")
    progress.sample(attempt)
    assert progress.cached == frozen and progress.writer is None
    assert len(writer.calls) == 2


def test_reader_latest_generation_success_clear_and_nonblocking_snapshot():
    reader = status.StatusReader("worker_a", None, None)
    try:
        assert reader.accept(legacy())
        assert reader.accept(legacy(sequence=1, kind=status._END))
        assert reader.accept(packet())
        assert reader.progress_snapshot()["stage"] == "mailbox"
        assert reader.accept(packet(sequence=1, kind=status._END))
        assert reader.progress_snapshot() is None
        assert reader.accept(legacy(generation=2))
        assert reader.accept(packet(generation=2))
        assert reader.accept(legacy(generation=3))
        assert reader.progress_snapshot() is None
        assert reader.accept(packet(generation=2, sequence=1))
        assert reader.progress_snapshot() is None
        assert reader.accept(packet(generation=3))
        with reader.lock:
            started = time.monotonic()
            assert reader.progress_snapshot() is None
            assert time.monotonic() - started < .1
        assert reader.progress_snapshot()["pc"] == 3
    finally:
        reader.close()
    assert reader.progress_snapshot() is None


@pytest.mark.parametrize("change", ["kind", "stage", "count", "clock", "sequence", "length"])
def test_malformed_progress_rejected_without_stop_or_failure_callback(change):
    events = []
    reader = status.StatusReader("worker_a", lambda: events.append("stop"), lambda: events.append("fail"))
    try:
        assert reader.accept(legacy())
        options = {}
        if change == "kind": options["kind"] = 42
        if change == "stage": options["stage"] = 255
        if change == "sequence": options["sequence"] = 1
        if change in ("count", "clock"):
            words = [2, 3, 4, 8, 5, 5, 100, 90, 10]
            words[3 if change == "count" else 7] = (1 << 63) if change == "count" else 101
            options["words"] = words
        raw = packet(**options)
        assert not reader.accept(raw[:-1] if change == "length" else raw)
        assert reader.progress_snapshot() is None and events == []
    finally:
        reader.close()


@pytest.mark.skipif(os.name != "posix", reason="inherited Unix FD")
@pytest.mark.parametrize("fragmented", [False, True])
def test_real_exec_fd_preserves_import_records_fragmented_progress_and_eof_cache(fragmented):
    reader = status.StatusReader("worker_b", None, None)
    seen = threading.Event()
    accept = reader.accept
    def observe(raw):
        valid = accept(raw)
        if reader.progress_snapshot() is not None:
            seen.set()
        return valid
    reader.accept = observe
    runtime = str(Path(status.__file__).parent)
    script = f"""
import os, sys
import pllm.runtime
pllm.runtime.__path__.insert(0, {runtime!r})
from pllm.runtime import fss_child_status as s
w = s.StatusWriter.from_environment()
assert w is not None and not w.channel.get_inheritable()
with s.source_status_scope(w): pass
raw = {packet(kind=status._FAIL)!r}
if {fragmented!r}:
    for offset in range(0, len(raw), 3):
        w.channel.sendall(raw[offset:offset+3])
else:
    w.progress(1, 4, (2,3,4,8,5,5,1000000000,900000000,100), s._FAIL)
w.close()
"""
    process = None
    try:
        reader.start()
        descriptor = reader.child.fileno()
        process = _spawn_role_process([sys.executable, "-I", "-c", script],
            environment={**os.environ, status.ENV: str(descriptor)}, pass_fds=(descriptor,),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        reader.spawned()
        assert seen.wait(3)
        assert reader.progress_snapshot()["materials_returned"] == 8
        output, error = process.communicate(timeout=3)
        assert process.returncode == 0, error
        assert output == error == b""
        reader.thread.join(1)
        assert not reader.thread.is_alive()
        # Failed in-flight evidence survives EOF, not explicit owner close.
        assert reader.snapshot() is None
        assert reader.progress_snapshot()["stage"] == "mailbox"
    finally:
        reader.close()
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=3)
    assert reader.progress_snapshot() is None


def diagnostics(progress):
    maximum = (1 << 63) - 1
    return {"schema_version": failure.SCHEMA, "phase": "request", "role": "worker_a",
        "elapsed_seconds": 1.2345678901234567e-100,
        "qualification": dict.fromkeys(("priced", "completed", "retired"), True),
        "spawned_roles": list(failure.ROLES), "ready_roles": list(failure.ROLES),
        "host": {"pressure": "critical", "available_bytes": maximum, "swap_used_bytes": maximum},
        "process_rss": {"latest_total_bytes": maximum, "peak_total_bytes": maximum,
            "sample_age_seconds": 1.2345678901234567e-100, "sampling_incomplete": False,
            "roles": {r: dict(latest_bytes=maximum, peak_bytes=maximum) for r in failure.ROLES}},
        "fss_control_boundary": dict(role="worker_b", action="finish", status="validated",
            elapsed_seconds=1.2345678901234567e-100), "fss_execution_progress": progress}


@pytest.mark.parametrize("seconds", [1.2345678901234567e-100, 2.2250738585072014e-308, 31535999.999999996])
def test_full_schema_bound_filtering_idempotence_and_frozen_precleanup(seconds):
    maximum = (1 << 63) - 1
    record = dict.fromkeys(("kernel", "pc", "frame", "materials_returned", "peer_acked", "peer_received", "cpu_ns"), maximum)
    record.update(stage="outbound_ack", elapsed_seconds=seconds,
                  changed_seconds=seconds, sample_age_seconds=seconds)
    raw = diagnostics({r: record.copy() for r in ("worker_a", "worker_b")})
    raw.update(phase=max(failure.PHASES, key=len), role="preparation", elapsed_seconds=seconds)
    raw["process_rss"]["sample_age_seconds"] = seconds
    raw["fss_control_boundary"].update(role="client", action="client_request", elapsed_seconds=seconds)
    clean = failure.sanitize_failure_diagnostics(raw)
    assert len(json.dumps(clean).encode()) < 4096
    assert len(json.dumps(clean, indent=2).encode()) < 4096
    assert failure.sanitize_failure_diagnostics(clean) == clean
    runtime = SimpleNamespace()
    collector = failure.FailureDiagnostics(runtime)
    frozen = collector.capture(observation=raw)
    raw["fss_execution_progress"]["worker_a"]["stage"] = "complete"
    raw["fss_execution_progress"].clear()  # Subsequent cleanup cannot change first failure.
    assert collector.capture(observation=raw) == frozen
    class Hostile:
        __hash__ = object.__hash__
        def __str__(self): raise AssertionError("stringified private tag")
        def __eq__(self, other): raise AssertionError("compared private tag")
    record["stage"] = Hostile()
    assert failure.sanitize_execution_progress({"worker_a": record}) is None
    assert failure.sanitize_execution_progress({Hostile(): record}) is None


@pytest.fixture
def progress_attempts(monkeypatch):
    attempts = []
    original = status.ExecutionProgress.sample
    def sample(progress, attempt, kind=status._UPDATE):
        if not any(a is attempt for a in attempts):
            attempts.append(attempt)
            attempt._test_samples = 0
        attempt._test_samples += 1
        attempt._test_progress = progress
        return original(progress, attempt, kind)
    monkeypatch.setattr(status.ExecutionProgress, "sample", sample)
    return attempts


async def wait_for_stage(attempts, stage):
    async with asyncio.timeout(3):
        while True:
            for attempt in attempts:
                if attempt.progress is not None and status.PROGRESS_STAGES[attempt.progress.point[0]] == stage:
                    attempt.progress_sample()
                    return attempt, dict(attempt.progress.cached)
            await asyncio.sleep(.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["tape", "outbound_ack", "mailbox"])
async def test_actual_mocked_lifecycle_distinguishes_tape_ack_mailbox_then_advances(cluster, monkeypatch, progress_attempts, gate):
    entered, release = threading.Event(), threading.Event()
    async_release = asyncio.Event()
    take, route = worker.MaterialTape.take, Routing.handle_async_request
    if gate == "tape":
        def blocked(tape, *args, **kwargs):
            if tape._test_role == "worker_a":
                entered.set()
                assert release.wait(3)
            return take(tape, *args, **kwargs)
        monkeypatch.setattr(worker.MaterialTape, "take", blocked)
    if gate == "outbound_ack":
        async def blocked_route(self, request):
            if request.url.path.endswith("/peer") and request.url.port == 7301:
                await async_release.wait()
            return await route(self, request)
        monkeypatch.setattr(Routing, "handle_async_request", blocked_route)
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    try:
        assert (await cluster.post("worker_a", "start")).status_code == 200
        attempt, before = await wait_for_stage(progress_attempts, gate)
        gauge = attempt.progress
        assert before["materials_returned"] == (0 if gate == "tape" else 1)
        assert before["peer_acked"] == (1 if gate == "mailbox" else 0)
        assert before["peer_received"] == 0
        assert before["kernel"] == 0
        # No metrics HTTP polling: read the actual cached owner directly in test.
        release.set()
        async_release.set()
        assert (await cluster.post("worker_b", "start")).status_code == 200
        replies = await asyncio.wait_for(asyncio.gather(*(cluster.post(r, "finish") for r in ("worker_a", "worker_b"))), 3)
        assert all(r.status_code == 200 for r in replies)
        assert gauge.cached["materials_returned"] == 2
        assert gauge.cached["peer_acked"] == gauge.cached["peer_received"] == 2
        assert gauge.cached["cpu_ns"] > before["cpu_ns"]
        assert gauge.cached["stage"] == "retiring" and gauge.ended
        assert attempt.progress is None  # Retirement releases association.
    finally:
        release.set()
        async_release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["enter", "sample"])
async def test_recording_faults_do_not_change_results_or_retirement(cluster, monkeypatch, method):
    def broken(*args, **kwargs): raise RuntimeError("diagnostic-only")
    monkeypatch.setattr(status.ExecutionProgress, method, broken)
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    for role in ("worker_a", "worker_b"):
        assert (await cluster.post(role, "start")).status_code == 200
    assert all(r.status_code == 200 for r in await asyncio.wait_for(asyncio.gather(
        *(cluster.post(r, "finish") for r in ("worker_a", "worker_b"))), 3))


def test_fixed_owner_inventory_with_full_copy_overlap():
    # Enumerated Python logical owners only, not allocator/stack/RSS proof.
    point = (4, *(int("9223372036854775807") for _ in range(4)))
    sample = status._progress_value(4, (2, 3, 4, *((1 << 63) - 1 for _ in range(6))))
    encoded = packet()
    def size(value):
        total = sys.getsizeof(value)
        if isinstance(value, dict): total += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, tuple): total += sum(size(v) for v in value)
        return total
    timer = worker._sample_progress(None)
    try:
        timer_bytes = sys.getsizeof(timer) + sys.getsizeof(timer.cr_frame)
    finally:
        timer.close()
    assert (3 * size(point) + 4 * size(sample) + 4 * size(encoded) + 2048
            + timer_bytes < status.PROGRESS_WORKER_BYTES)
    sample["sample_age_seconds"] = 1.2345678901234567e-100
    assert 4 * size(sample) + 4 * size(encoded) + 4096 < status.PROGRESS_SUPERVISOR_BYTES
    assert len(encoded) == 88
    # Worst-case legacy phases + start/timer/terminal fit ORIGINAL per-scope and
    # aggregate channel caps, even for all 64 admitted requests plus startup.
    per_scope = len(status.SOURCE_IMPORT_PHASES) + 2 + status.PROGRESS_SAMPLES + 3
    assert per_scope <= status._STEPS
    assert (status._SCOPES - 1) * per_scope + len(status.SOURCE_IMPORT_PHASES) + 2 <= status._MAX_RECORDS


def test_supervisor_collector_reads_cached_gauge_before_cleanup_not_during_startup():
    reader = status.StatusReader("worker_a", None, None)
    topology = object.__new__(LocalTopology)
    topology._process_lock = threading.Lock()
    topology._startup_breadcrumbs = {"phase": "ready"}
    topology._child_source_status = {"worker_a": reader}
    marker = ("worker_a", "finish", "entered", 2.0)
    core = SimpleNamespace(runtime_id="compiled_fss_v1", _fss_control_snapshot=lambda: marker)
    runtime = SimpleNamespace(_topology=topology, _lock=threading.Lock(), _state={"phase": "ready"},
        _active_run=None, _client=SimpleNamespace(_core=core), _memory_guard=None, _has_started_run=True)
    try:
        assert reader.accept(legacy())
        assert reader.accept(legacy(sequence=1, kind=status._END))
        assert reader.accept(packet())
        collector = failure.FailureDiagnostics(runtime)
        assert "fss_execution_progress" not in collector.observe()
        runtime._active_run = object()
        marker = ("worker_a", "manifest", "entered", 0.0)
        assert "fss_execution_progress" not in collector.observe()  # Old failed request cache.
        marker = ("worker_a", "finish", "entered", 2.0)
        with reader.lock:
            assert "fss_execution_progress" not in collector.observe()
        frozen = collector.capture()
        assert frozen["fss_execution_progress"]["worker_a"]["stage"] == "mailbox"
        reader.close()
        runtime._active_run = None
        assert collector.capture() == frozen
    finally:
        reader.close()


def test_explicit_reader_close_hides_cache_even_with_contended_lock():
    reader = status.StatusReader("worker_a", None, None)
    assert reader.accept(legacy()) and reader.accept(packet())
    with reader.lock:
        reader.close()
        assert reader.progress_snapshot() is None
    assert reader.progress_snapshot() is None


@pytest.mark.asyncio
async def test_partial_tape_failure_counts_only_returned_materials_and_preserves_error(cluster, monkeypatch, progress_attempts):
    boom = ValueError("private source error")
    original_take = worker.MaterialTape.take
    errors = []
    def fail_after_read(tape, *args, **kwargs):
        result = original_take(tape, *args, **kwargs)
        if tape._test_role == "worker_a":
            errors.append(boom)
            raise boom
        return result
    monkeypatch.setattr(worker.MaterialTape, "take", fail_after_read)
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    assert (await cluster.post("worker_a", "start")).status_code == 200
    response = await cluster.post("worker_a", "finish")
    assert response.status_code == 400 and b"private source error" not in response.content
    assert errors == [boom]
    # Owner may already have drained; capture the pre-cleanup sample via hook.
    assert all(a.progress is None for a in progress_attempts)
    assert len(progress_attempts) == 1
    assert progress_attempts[0]._test_progress.cached["stage"] == "tape"
    assert progress_attempts[0]._test_progress.cached["materials_returned"] == 0


def test_progress_quote_adds_real_owners_and_rejects_over_cap(price_contract, experiment, monkeypatch):
    from pllm.runtime import benchmark_memory as memory
    after = memory._fss_memory({}, experiment.pipeline, 2, 3, 0, None, 0, contract=price_contract)
    with monkeypatch.context() as patch:
        patch.setattr(status, "PROGRESS_WORKER_BYTES", 0)
        patch.setattr(status, "PROGRESS_SUPERVISOR_BYTES", 0)
        before = memory._fss_memory({}, experiment.pipeline, 2, 3, 0, None, 0, contract=price_contract)
    assert after["client_peak_bytes"] - before["client_peak_bytes"] == memory._slack(2 * (16 << 10), 25)
    for role in ("worker_a", "worker_b"):
        assert after["provider_peak_bytes"][role] - before["provider_peak_bytes"][role] == memory._slack(16 << 10, 25)
    assert after["provider_peak_bytes"]["dealer"] == before["provider_peak_bytes"]["dealer"]
    assert after["incremental_owner_bounds"] == before["incremental_owner_bounds"]
    assert after["allocation_owners"]["each_role"]["native_kernel_owner"] == 64 << 20
    host = memory.HostMemory(total=32 << 30, available=24 << 30, swap_used=0, disk_free=1 << 40)
    budget = before["native_total_peak_bytes"]
    assert memory.admit_memory(before, host, backend="native", memory_budget_bytes=budget)["admitted"]
    assert not memory.admit_memory(after, host, backend="native", memory_budget_bytes=budget)["admitted"]
    monkeypatch.setattr(status, "PROGRESS_WORKER_BYTES", worker.p.ROLE_MEMORY + 1)
    with pytest.raises(memory.BenchmarkMemoryError, match="memory caps"):
        memory._fss_memory({}, experiment.pipeline, 2, 3, 0, None, 0, contract=price_contract)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [.5], indirect=True)
async def test_existing_expiry_task_samples_blocked_job_without_http_polling(cluster, progress_attempts):
    await cluster.ready()
    await cluster.prepare()
    assert (await cluster.post("dealer", "finish")).status_code == 200
    await cluster.inputs()
    assert (await cluster.post("worker_a", "start")).status_code == 200
    attempt, _ = await wait_for_stage(progress_attempts, "mailbox")
    gauge = attempt.progress
    before = attempt._test_samples
    async with asyncio.timeout(.25):
        while attempt._test_samples == before:
            await asyncio.sleep(.001)  # Test barrier, not production HTTP polling.
    assert gauge.cached["stage"] == "mailbox"
    assert gauge.cached["materials_returned"] == gauge.cached["peer_acked"] == 1
    assert gauge.cached["peer_received"] == 0
    assert (await cluster.post("worker_b", "start")).status_code == 200
    assert all(r.status_code == 200 for r in await asyncio.wait_for(asyncio.gather(
        *(cluster.post(r, "finish") for r in ("worker_a", "worker_b"))), .25))
    # Includes the one explicit stage-barrier sample above. No per-record send.
    assert attempt._test_samples <= status.PROGRESS_SAMPLES + 4


class ProgressClock:
    """Only the sampler/gauge see this clock; pytest's loop remains untouched."""

    def __init__(self, origin):
        self.now = origin
        self.sleeps = []
        self.after_sleep = None

    def monotonic(self):
        return self.now

    async def sleep(self, delay):
        assert math.isfinite(delay) and delay > 0
        self.sleeps.append(delay)
        self.now += delay
        if self.after_sleep is not None:
            self.after_sleep()


@pytest.fixture
def progress_timer(monkeypatch):
    def make(ttl=1800, *, origin=1000, writer=None):
        clock = ProgressClock(origin)
        monkeypatch.setattr(worker, "time", SimpleNamespace(monotonic=clock.monotonic))
        monkeypatch.setattr(worker, "asyncio", SimpleNamespace(
            sleep=clock.sleep, CancelledError=asyncio.CancelledError))
        monkeypatch.setattr(status, "time", SimpleNamespace(
            monotonic_ns=lambda: int(clock.now * 1_000_000_000),
            process_time_ns=lambda: int(clock.now * 1_000_000)))
        attempt = SimpleNamespace(expires=origin + ttl, retiring=False,
            stop=threading.Event(), peer_sent=0, peer_received=0,
            progress=status.ExecutionProgress(writer))
        samples = []
        def sample(kind=status._UPDATE):
            worker._Attempt.progress_sample(attempt, kind)
            samples.append((clock.now, kind))
        attempt.progress_sample = sample
        return clock, attempt, samples
    return make


@pytest.mark.asyncio
async def test_progress_timer_spans_remaining_deadline_and_observes_late_phases(progress_timer):
    clock, attempt, samples = progress_timer()
    def change_phase():
        elapsed = clock.now - 1000
        attempt.progress.enter("tape" if elapsed < 600 else
                               "outbound_ack" if elapsed < 1200 else "mailbox")
    clock.after_sleep = change_phase
    await worker._sample_progress(attempt)
    assert samples == [(1000 + 179 * index, status._UPDATE) for index in range(1, 11)]
    assert attempt.progress.cached["stage"] == "mailbox"
    assert attempt.progress.cached["elapsed_seconds"] == 1790
    assert attempt.expires == 2800 and not attempt.stop.is_set()
    assert len(clock.sleeps) == status.PROGRESS_SAMPLES


@pytest.mark.asyncio
async def test_progress_timer_uses_remaining_time_without_renewing_after_admission(progress_timer):
    clock, attempt, samples = progress_timer(origin=500)
    clock.now = 1000  # Time already spent admitting/importing this same attempt.
    await worker._sample_progress(attempt)
    assert attempt.expires == 2300
    assert samples == [(1000 + 129 * index, status._UPDATE) for index in range(1, 11)]
    assert clock.now == 2290


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [.001, .5])
async def test_progress_timer_short_ttl_uses_same_fraction_without_minimum_delay(progress_timer, ttl):
    clock, attempt, samples = progress_timer(ttl, origin=0)
    await worker._sample_progress(attempt)
    assert len(samples) == status.PROGRESS_SAMPLES
    assert all(0 < at < ttl for at, _ in samples)
    assert samples[-1][0] == pytest.approx(ttl * 15 / 16)
    assert sum(clock.sleeps) == pytest.approx(samples[-1][0])


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0, -1])
async def test_progress_timer_already_expired_has_no_samples(progress_timer, ttl):
    clock, attempt, samples = progress_timer(ttl)
    await worker._sample_progress(attempt)
    assert samples == clock.sleeps == []


@pytest.mark.asyncio
async def test_progress_timer_insufficient_representable_ttl_does_not_burst(progress_timer):
    origin = 1_000_000_000.0
    ttl = math.nextafter(origin, math.inf) - origin
    clock, attempt, samples = progress_timer(ttl, origin=origin)
    await worker._sample_progress(attempt)
    assert samples == clock.sleeps == []
    assert attempt.expires == origin + ttl


@pytest.mark.asyncio
async def test_progress_timer_skips_missed_slots_without_zero_delay_catchup(progress_timer):
    clock, attempt, samples = progress_timer()
    def stalled_loop():
        if len(clock.sleeps) == 1:
            clock.now = 1900
    clock.after_sleep = stalled_loop
    await worker._sample_progress(attempt)
    assert [at for at, _ in samples] == [1900, 2074, 2253, 2432, 2611, 2790]
    assert len(clock.sleeps) == 6 < status.PROGRESS_SAMPLES


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["expired", "stopped", "retiring"])
async def test_progress_timer_rechecks_expiry_and_owner_after_waking(progress_timer, reason):
    clock, attempt, samples = progress_timer()
    def end_attempt():
        if reason == "expired": clock.now = attempt.expires
        elif reason == "stopped": attempt.stop.set()
        else: attempt.retiring = True
    clock.after_sleep = end_attempt
    await worker._sample_progress(attempt)
    assert samples == [] and len(clock.sleeps) == 1


@pytest.mark.asyncio
async def test_progress_timer_task_cancel_propagates_without_sampling_or_retirement(progress_timer, monkeypatch):
    clock, attempt, samples = progress_timer()
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocked_sleep(delay):
        assert delay > 0
        entered.set()
        await release.wait()
    monkeypatch.setattr(worker.asyncio, "sleep", blocked_sleep)
    task = asyncio.create_task(worker._sample_progress(attempt))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert samples == clock.sleeps == []
    assert not attempt.retiring and not attempt.stop.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_at", [1, 2, 3])
@pytest.mark.parametrize("fault", ["raise", "nan", "inf", "backwards"])
async def test_progress_timer_clock_fault_disables_sampling_without_guessing_deadline(
        progress_timer, monkeypatch, failure_at, fault):
    clock, attempt, samples = progress_timer()
    reads = 0
    def broken_clock():
        nonlocal reads
        reads += 1
        if reads == failure_at:
            if fault == "raise": raise OSError("diagnostic clock unavailable")
            return {"nan": math.nan, "inf": math.inf, "backwards": 999}[fault]
        return clock.now
    monkeypatch.setattr(worker.time, "monotonic", broken_clock)
    await worker._sample_progress(attempt)
    # A bad first reading has no earlier origin against which to prove a
    # regression. Later failures stop at the affected read, without retry.
    if fault == "backwards" and failure_at == 1:
        assert len(samples) <= status.PROGRESS_SAMPLES
    else:
        assert samples == []
    assert attempt.expires == 2800 and not attempt.retiring and not attempt.stop.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["monotonic_ns", "process_time_ns"])
async def test_progress_timer_gauge_clock_fault_keeps_last_snapshot_and_sample_cap(
        progress_timer, monkeypatch, field):
    clock, attempt, samples = progress_timer()
    attempt.progress_sample()
    cached = attempt.progress.cached.copy()
    def broken_clock(): raise OSError("optional gauge clock unavailable")
    monkeypatch.setattr(status.time, field, broken_clock)
    await worker._sample_progress(attempt)
    assert attempt.progress.cached == cached
    assert len(samples) == status.PROGRESS_SAMPLES + 1
    assert clock.now == 2790 and not attempt.stop.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["blocked", "partial", "lock"])
async def test_progress_timer_backpressure_never_retries_and_cache_stays_latest(progress_timer, fault):
    class Channel:
        sends = 0
        def send(self, raw):
            assert len(raw) == status._PROGRESS_RECORD.size
            self.sends += 1
            if fault == "blocked": raise BlockingIOError()
            return len(raw) - 1
        def close(self): pass
    writer = object.__new__(status.StatusWriter)
    writer.channel, writer.lock = Channel(), threading.Lock()
    writer.closed, writer.current, writer.generation = False, None, 1
    writer.progress_generation = writer.progress_sequence = 0
    clock, attempt, samples = progress_timer(writer=writer)
    if fault == "lock": writer.lock.acquire()
    try:
        await worker._sample_progress(attempt)
    finally:
        if fault == "lock": writer.lock.release()
    assert len(samples) == status.PROGRESS_SAMPLES
    assert writer.channel.sends == (0 if fault == "lock" else 1)
    assert writer.closed == (fault != "lock")
    assert attempt.progress.cached["elapsed_seconds"] == 1790
    assert clock.now < attempt.expires and not attempt.stop.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [status._END, status._FAIL])
async def test_progress_timer_preserves_separate_initial_and_terminal_samples(progress_timer, kind):
    _, attempt, samples = progress_timer()
    attempt.progress_sample()
    assert samples == [(1000, status._UPDATE)]
    await worker._sample_progress(attempt)
    attempt.progress.enter("retiring")
    attempt.progress_sample(kind)
    assert len(samples) == status.PROGRESS_SAMPLES + 2
    assert samples[-1] == (2790, kind)
    assert attempt.progress.cached["stage"] == "retiring" and attempt.progress.ended
    cached = attempt.progress.cached.copy()
    attempt.progress.enter("complete")
    worker._Attempt.progress_sample(attempt)
    assert attempt.progress.cached == cached


@pytest.mark.asyncio
async def test_progress_timer_missing_deadline_has_no_authoritative_fallback(progress_timer):
    clock, attempt, samples = progress_timer()
    del attempt.expires
    await worker._sample_progress(attempt)
    assert samples == clock.sleeps == []
