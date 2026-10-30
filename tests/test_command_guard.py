"""Definitions only. Fake I/O drives REAL supervisor loop; no processes/kills."""
from dataclasses import replace
from pathlib import Path
import signal

import pytest

from scripts import command_guard as g

SLOT = Path("/main/target/native-command.slot")


def host(**changes):
    raw = changes.get("free_bytes", 24 * g.GIB)
    available = changes.get("available_estimate_bytes", raw)
    page = 16384
    return dict(schema="pllm.host_swap_observation.v1",
                counter_source="darwin.HOST_VM_INFO64.rev1.swapouts", page_bytes=16384,
                swapout_bytes=100, swap_used_bytes=100, free_bytes=raw,
                free_count_including_speculative=raw // page, speculative_count=0,
                inactive_count=(available - raw) // page,
                available_estimate_bytes=available, physical_total_bytes=64 * g.GIB,
                headroom_metric="available_estimate_bytes",
                free_source=g.host_swap.FREE_SOURCE, page_size_source=g.host_swap.PAGE_SOURCE,
                available_estimate_source=g.host_swap.AVAILABLE_SOURCE) | changes


class Fake:
    def __init__(self):
        self.time = 100.0
        self.interrupted = False
        self.events = []
        self.pending = False
        self.final = False
        self.launched = False
        self.hang = self.crash = self.malformed = False
        self.hang_after_launch = False
        self.survivor = self.escape = self.reuse = False
        self.running = False
        self.reap_hang = False
        self.exit_code = 0
        self.worker_exit_code = -15
        self.host_value = host()
        self.growth = 0
        self.stops = {}
        self.killed = set()
        self.reaped = set()
        self.acquired = False
        self.slot_exists = False
        self.unknown_birth = False
        self.alternate = False
        self.observer_survivor = False
        self.slot_pending = False
        self.acquiring = False
        self.release_error = False
        self.request_error = False
        self.external_signal = False
        self.free_after_acquire = None

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds

    def pid(self, name):
        return {"worker": 2, "payload": 10}[name]

    def startup_step(self, allow_go=True):
        return True

    def close(self, name):
        self.events.append(("close", name, self.time))

    def launch(self, argv, cwd, env):
        self.events.append(("launch", argv, cwd, env))
        self.launched = True
        return 10

    def acquire(self, slot):
        self.events.append(("acquire", slot, self.time))
        self.slot_pending = True

    def accept(self, data, acquire=False):
        if acquire:
            self.acquired, self.slot_pending = True, False

    def release(self, slot, commit=True):
        assert self.reaped == ({"worker", "payload"} if self.launched else {"worker"})
        self.events.append(("release", slot, self.time))
        if self.release_error:
            raise g.Blocked("authority/slot replaced before release")
        self.acquired = False

    def signal(self, name, sig):
        assert name not in self.reaped
        self.events.append(("signal", name, sig, self.time))
        if sig == signal.SIGTERM:
            self.stops[name] = self.time
        else:
            assert self.time - self.stops[name] >= 1
            self.killed.add(name)

    def reap(self, name):
        assert name in self.killed
        self.events.append(("reap", name, self.time))
        if self.reap_hang:
            return None
        self.reaped.add(name)
        return self.exit_code if name == "payload" else self.worker_exit_code

    def request(self, final, acquire=False):
        if self.request_error:
            raise g.Blocked("request-side IPC failure")
        assert not self.pending
        self.pending, self.final = True, final
        self.acquiring = acquire
        self.events.append(("request", final, self.time))

    def read(self):
        if self.external_signal and self.launched:
            self.interrupted = True
        if self.acquiring and self.slot_exists:
            raise FileExistsError("fixed slot competing or stale")
        if self.acquiring and self.free_after_acquire is not None:
            self.host_value = host(available_estimate_bytes=self.free_after_acquire,
                                   free_bytes=66 * (1 << 20))
        if self.crash:
            raise g.Blocked("worker crash/EOF")
        if self.hang or (self.hang_after_launch and self.launched):
            return None  # never returns a sample; clock/signals still progress
        self.pending = False
        if self.malformed:
            return {"sealed": self.final}
        rows = [[1, 1.0, 0, 7, 1000, False], [2, 2.0, 1, 2, 2000, False],
                [3, 3.0, 2, 2, 3000, False]]  # sampled, already-reaped ps helper
        if self.observer_survivor:
            rows.append([4, 4.0, 999, 2, 1000, False])
        if self.launched:
            rows.append([10, None if self.unknown_birth else 10.0, 1, 10, 4000,
                         not self.running or "payload" in self.killed])
            if self.survivor or "payload" not in self.killed:
                rows.append([11, 11.0, 999, 10, 8000, False])
            if self.escape:
                rows.append([12, 12.0, 10, 12, 8000, False])
            if self.reuse:
                rows.append([99, 99.0, 999, 99, 8000, False])
        return dict(slot=str(Path("/alternate/slot") if self.alternate else SLOT),
                    sealed=self.final, host=self.host_value.copy(),
                    disk_growth=self.growth, processes=rows, reaped_helpers=[[3, 3.0]])


def run(fake, **changes):
    limits = replace(g.Limits(campaign_deadline=1000), **changes)
    return g.run(["cargo", "test", "--no-run"], Path("/work"), {"PIN": "caller"},
                 SLOT, limits, fake, 1)


@pytest.mark.parametrize("path", ["host", "process", "disk", "birth"])
def test_each_hung_probe_stops_without_probe_return(path):
    fake = Fake()
    fake.hang_after_launch = path  # named probe remains stuck inside worker
    fake.running = True
    outcome = run(fake, seconds=0.5)
    assert outcome.code == 78 and outcome.held
    term = next(e for e in fake.events if e[:3] == ("signal", "payload", signal.SIGTERM))
    assert term[3] <= 100.55
    assert fake.killed == {"worker", "payload"}
    assert fake.time <= 104


def test_responsive_probe_cannot_reset_command_deadline():
    fake = Fake()
    fake.running = True
    result = run(fake, seconds=0.5)
    assert result.code == 1 and not result.held
    term = next(e for e in fake.events if e[:3] == ("signal", "payload", signal.SIGTERM))
    assert term[3] == 100.5
    assert sum(e[0] == "launch" for e in fake.events) == 1


@pytest.mark.parametrize("kind", ["hang", "crash", "malformed", "reap_hang"])
def test_worker_failure_and_terminal_wait_are_bounded_and_hold_slot(kind):
    fake = Fake()
    setattr(fake, kind, True)
    outcome = run(fake)
    assert outcome.code == 78 and outcome.held
    assert fake.time <= 105
    assert "worker" in fake.killed


def test_preflight_available_estimate_exact_and_one_page_short():
    exact = Fake()
    assert run(exact).code == 0
    short = Fake()
    short.host_value = host(free_bytes=66 * (1 << 20), available_estimate_bytes=24 * g.GIB - 16384)
    short.host_value["free_speculative_bytes"] = 100 * g.GIB
    short.host_value.update(inactive_bytes=100 * g.GIB, purgeable_bytes=100 * g.GIB,
                            filebacked_bytes=100 * g.GIB, available_bytes=100 * g.GIB)
    result = run(short)
    assert result.code == 78 and not result.held and not short.launched
    assert f"measured_estimate_bytes={24 * g.GIB - 16384}" in result.reason
    assert f"required_bytes={24 * g.GIB}" in result.reason
    assert "metric=available_estimate_bytes" in result.reason
    # Untrusted aggregate diagnostics cannot override the owned estimate.


@pytest.mark.parametrize("meter", ["rss", "disk", "reserve"])
def test_exact_and_one_over_meters(meter):
    limits = g.Limits(1000)
    args = dict(host=host(free_bytes=limits.reserve), baseline=host(), previous=host(),
                rss=limits.rss, growth=limits.disk_growth, limits=limits)
    assert g.resource_violation(**args) is None
    if meter == "reserve":
        args["host"] = host(free_bytes=limits.reserve - 16384)
    elif meter == "rss":
        args["rss"] += 1
    else:
        args["growth"] += 1
    assert g.resource_violation(**args)


@pytest.mark.parametrize("field", ["swapout_bytes", "swap_used_bytes"])
def test_swap_rise_then_partial_fall_above_baseline_rejected(field):
    with pytest.raises((RuntimeError, g.Blocked)):
        g.resource_violation(host(**{field: 110}), host(), host(**{field: 120}), 0, 0, g.Limits(1000))


@pytest.mark.parametrize("failure", [False, True])
def test_success_or_failure_descendants_term_kill_then_reap(failure):
    fake = Fake()
    fake.exit_code = 3 if failure else 0
    outcome = run(fake)
    assert outcome.code == (1 if failure else 0) and not outcome.held
    assert fake.events[0][0] == "request"
    assert ("launch", ["cargo", "test", "--no-run"], Path("/work"), {"PIN": "caller"}) in fake.events
    for name in ("worker", "payload"):
        term = next(e for e in fake.events if e[:3] == ("signal", name, signal.SIGTERM))
        kill = next(e for e in fake.events if e[:3] == ("signal", name, signal.SIGKILL))
        reap = next(e for e in fake.events if e[:2] == ("reap", name))
        assert kill[3] - term[3] >= 1 and reap[2] >= kill[3]


@pytest.mark.parametrize("kind", ["survivor", "escape"])
def test_term_resistance_or_reparent_escape_holds_even_after_leader_reap(kind):
    fake = Fake()
    setattr(fake, kind, True)
    result = run(fake)
    assert result.code == 78 and result.held
    assert fake.killed == {"worker", "payload"}
    assert all(e[1] in ("worker", "payload") for e in fake.events if e[0] == "signal")


def test_pid_reuse_not_adopted_and_reparented_known_birth_stays_tracked():
    family = g.Family((1, 1.0), (2, 2.0), (10, 10.0))
    common = [[1, 1.0, 0, 7, 1, False], [2, 2.0, 1, 2, 2, False]]
    family.observe(common + [[10, 10.0, 1, 10, 4, False], [11, 11.0, 10, 10, 8, False]])
    family.observe(common + [[10, 99.0, 999, 99, 4, False], [11, 11.0, 999, 11, 8, False]])
    assert (10, 99.0) not in family.known
    assert (11, 11.0) in family.payload_known and family.escaped


def test_cumulative_grow_truncate_regrow_delete_recreate():
    meter = g.ObservedDiskGrowth()
    key = (1, 2, 3)
    assert meter.observe({key: (100, 1)}) == 0
    assert meter.observe({key: (200, 2)}) == 100
    assert meter.observe({key: (50, 3)}) == 100
    assert meter.observe({key: (100, 4)}) == 150
    assert meter.observe({}) == 150
    assert meter.observe({key: (75, 5)}) == 225
    assert meter.observe({(1, 2, 9): (80, 6)}) == 305


def test_uncertain_identity_conservatively_charges_full_each_sample():
    meter = g.ObservedDiskGrowth()
    key = (1, 2, None)
    assert meter.observe({key: (100, 1)}) == 0
    assert meter.observe({key: (110, 2)}) == 110
    # Without birth, even unchanged metadata cannot prove no inode replacement.
    assert meter.observe({key: (110, 2)}) == 220


def test_root_inode_replacement_blocks_without_real_files(monkeypatch):
    from types import SimpleNamespace
    root = Path("/target")
    monkeypatch.setattr(g, "disk_roots", lambda paths: (root,))
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_dev=1, st_ino=2, st_birthtime=3))
    disk = g.Disk([root])
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_dev=1, st_ino=99, st_birthtime=3))
    with pytest.raises(g.Blocked, match="root replacement"):
        disk.snapshot()


def test_worker_and_sampled_helper_rss_are_all_counted():
    fake = Fake()
    result = run(fake, rss=5999)
    assert result.code == 78 and not fake.launched and not result.held


def test_cli_preserves_remainder_and_fixed_slot_argument():
    args = g.parser().parse_args(["--cwd", "/work", "--coordinator-root", "/main", "--slot", str(SLOT), "--target-dir", "/target",
        "--log-dir", "/target/log", "--temp-dir", "/target/tmp", "--campaign-deadline", "1000",
        "--cooperative-build", "--", "uv", "build", "--wheel"])
    assert args.argv == ["--", "uv", "build", "--wheel"]
    assert args.slot == SLOT


def test_competing_or_stale_slot_never_launches_payload_or_removes_slot():
    fake = Fake()
    fake.slot_exists = True
    result = run(fake)
    assert result.code == 78 and result.held and not fake.launched
    assert not fake.acquired
    assert sum(e[0] == "acquire" for e in fake.events) == 1


@pytest.mark.parametrize("kind", ["alternate", "unknown_birth"])
def test_alternate_slot_or_uncertain_payload_birth_refused(kind):
    fake = Fake()
    setattr(fake, kind, True)
    result = run(fake)
    assert result.code == 78 and result.held
    if kind == "alternate":
        assert not fake.acquired and not fake.launched
    else:
        assert fake.acquired and fake.killed == {"worker", "payload"}


def test_observer_descendant_cleanup_unknown_retains_acquired_slot():
    fake = Fake()
    fake.observer_survivor = True
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    assert fake.killed == {"worker", "payload"}


def test_observer_unexpected_exit_after_final_census_retains_slot():
    fake = Fake()
    fake.worker_exit_code = 78
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired


@pytest.mark.parametrize("changes", [dict(seconds=600.1), dict(rss=8 * g.GIB + 1),
    dict(reserve=16 * g.GIB - 1), dict(disk_growth=20 * g.GIB + 1),
    dict(campaign_deadline=7300.1), dict(interval=0.11), dict(probe_timeout=2.1),
    dict(reap_timeout=1.1), dict(grace=0.5)])
def test_ceiling_or_full_grace_cannot_be_relaxed(changes):
    with pytest.raises(g.Blocked):
        replace(g.Limits(1000), **changes).validate(100)


def test_invalid_bounds_cannot_weaken_existing_worker_cleanup_grace():
    fake = Fake()
    result = run(fake, grace=0.5)
    assert result.code == 78 and result.held and not fake.launched
    assert fake.killed == {"worker"} and fake.time >= 101


def test_worker_reparent_escape_is_sticky():
    family = g.Family((1, 1.0), (2, 2.0))
    common = [[1, 1.0, 0, 7, 1, False], [2, 2.0, 1, 2, 2, False]]
    family.observe(common + [[3, 3.0, 2, 2, 3, False]])
    family.observe(common + [[3, 3.0, 999, 3, 3, False]])
    assert family.worker_escaped and (3, 3.0) in family.worker_known


def mock_darwin_swap(monkeypatch):
    from types import SimpleNamespace
    import psutil
    monkeypatch.setattr(g.host_swap.sys, "platform", "darwin")
    monkeypatch.setattr(g.host_swap, "_darwin_kernel_page_size", lambda: 16384)
    monkeypatch.setattr(g.host_swap, "_darwin_physical_total_bytes", lambda: 64 * g.GIB)
    monkeypatch.setattr(psutil, "swap_memory", lambda: SimpleNamespace(used=100, sout=100))


@pytest.mark.parametrize("raw,speculative,expected", [(24, 10, 14), (34, 10, 24), (26, 10, 16)])
def test_darwin_actual_headroom_subtracts_speculative(monkeypatch, raw, speculative, expected):
    from types import SimpleNamespace
    mock_darwin_swap(monkeypatch)
    monkeypatch.setattr(g.host_swap, "_darwin_vm_statistics",
                        lambda: SimpleNamespace(free_count=raw * g.GIB // 16384,
                                                 speculative_count=speculative * g.GIB // 16384,
                                                inactive_count=0, swapouts=0, pageouts=0))
    probes = g.DarwinProbes.__new__(g.DarwinProbes)
    result = probes.host()
    assert result["free_bytes"] == expected * g.GIB
    violation = g.resource_violation(result, result, result, 0, 0, g.Limits(1000))
    assert result["available_estimate_bytes"] == raw * g.GIB
    assert violation is None  # Speculative is already included in Mach free.


@pytest.mark.parametrize("raw,speculative", [(-1, 0), (5, -1), (5, 6)])
def test_darwin_inconsistent_free_speculative_rejected(monkeypatch, raw, speculative):
    from types import SimpleNamespace
    mock_darwin_swap(monkeypatch)
    monkeypatch.setattr(g.host_swap, "_darwin_vm_statistics",
                        lambda: SimpleNamespace(free_count=raw, speculative_count=speculative,
                                                inactive_count=0, swapouts=0, pageouts=0))
    with pytest.raises(g.Blocked, match="inconsistent"):
        g.DarwinProbes.__new__(g.DarwinProbes).host()


@pytest.mark.parametrize("short", [False, True])
def test_computed_darwin_headroom_launch_exact_and_one_page_short(monkeypatch, short):
    from types import SimpleNamespace
    page = 16384
    mock_darwin_swap(monkeypatch)
    monkeypatch.setattr(g.host_swap, "_darwin_vm_statistics", lambda: SimpleNamespace(
        free_count=14 * g.GIB // page, speculative_count=10 * g.GIB // page,
        inactive_count=10 * g.GIB // page - int(short),
        swapouts=0, pageouts=0))
    fake = Fake()
    fake.host_value = g.DarwinProbes.__new__(g.DarwinProbes).host()
    result = run(fake)
    assert fake.host_value["free_bytes"] == 4 * g.GIB
    assert fake.host_value["available_estimate_bytes"] == 24 * g.GIB - int(short) * page
    assert result.code == (78 if short else 0)
    assert fake.launched is not short


@pytest.mark.parametrize("value", [None, True, False, -1, 24.0 * g.GIB, "25769803776"])
def test_preflight_invalid_free_bytes_before_threshold(value):
    fake = Fake()
    fake.host_value["free_bytes"] = value
    result = run(fake)
    assert result.code == 78 and not fake.launched and not fake.acquired
    assert "invalid Darwin headroom" in result.reason


def test_preflight_missing_free_bytes_before_threshold():
    fake = Fake()
    del fake.host_value["free_bytes"]
    result = run(fake)
    assert result.code == 78 and not fake.launched and not fake.acquired
    assert "invalid Darwin headroom" in result.reason


def test_launch_headroom_rechecked_after_acquisition_before_payload():
    fake = Fake()
    fake.free_after_acquire = 23 * g.GIB
    result = run(fake)
    assert result.code == 78 and not result.held and not fake.launched


def test_available_estimate_admits_low_raw_free_but_preserves_other_limits():
    row = host(free_bytes=66 * (1 << 20), available_estimate_bytes=25 * g.GIB)
    assert g.validated_headroom(row) == 25 * g.GIB
    fake = Fake()
    fake.host_value = row
    assert run(fake).code == 0  # Fake payload only; no real process launch.
    limits = g.Limits(1000)
    assert g.resource_violation(row, row, row, limits.rss + 1, 0, limits).startswith("aggregate RSS")
    assert g.resource_violation(row, row, row, 0, limits.disk_growth + 1, limits)
    for field, reason in (("swap_used_bytes", "swap-used growth"),
                          ("swapout_bytes", "actual Swapouts growth")):
        assert g.resource_violation(row | {field: 101}, row, row, 0, 0, limits) == reason
    with pytest.raises(RuntimeError):
        g.resource_violation(row | {"swapout_bytes": 99}, row, row, 0, 0, limits)
    with pytest.raises(g.Blocked, match="swap-used counter fell"):
        g.resource_violation(row | {"swap_used_bytes": 99}, row, row, 0, 0, limits)
    for field in ("swapout_bytes", "swap_used_bytes"):
        with pytest.raises(RuntimeError, match="invalid host swap counters"):
            g.resource_violation(row | {field: True}, row, row, 0, 0, limits)


def test_pressure_percentage_cannot_admit_twelve_point_three_gib():
    fake = Fake()
    available = (123 * g.GIB // 10 // 16384) * 16384
    fake.host_value = host(free_bytes=66 * (1 << 20), available_estimate_bytes=available,
                           memory_pressure_free_percentage=77, purgeable_bytes=100 * g.GIB,
                           filebacked_bytes=100 * g.GIB, available_bytes=100 * g.GIB)
    result = run(fake)
    assert result.code == 78 and not fake.launched
    assert f"measured_estimate_bytes={available}" in result.reason


@pytest.mark.parametrize("threshold", [16, 24])
def test_headroom_exact_and_one_byte_short_with_page_consistent_rows(threshold):
    # Mach bytes have page granularity. Tighten required threshold by ONE BYTE
    # instead of fabricating a non-page-aligned observed estimate.
    row = host(free_bytes=66 * (1 << 20), available_estimate_bytes=threshold * g.GIB)
    assert g.validated_headroom(row) == threshold * g.GIB
    if threshold == 16:
        exact = g.Limits(1000)
        assert g.resource_violation(row, row, row, 0, 0, exact) is None
        assert g.resource_violation(row, row, row, 0, 0, replace(exact, reserve=16 * g.GIB + 1))
    else:
        exact, short = Fake(), Fake()
        exact.host_value = row.copy()
        short.host_value = row.copy()
        assert run(exact).code == 0
        result = run(short, reserve=16 * g.GIB + 1)
        assert result.code == 78 and not short.launched
        assert f"required_bytes={24 * g.GIB + 1}" in result.reason


@pytest.mark.parametrize("field", [
    "available_estimate_bytes", "free_bytes", "physical_total_bytes", "inactive_count",
    "free_count_including_speculative", "speculative_count", "page_bytes",
    "available_estimate_source", "free_source", "page_size_source", "headroom_metric",
    "counter_source", "schema",
])
@pytest.mark.parametrize("value", [None, True, False, -1, 1.5, "unknown", "missing"])
def test_headroom_malformed_missing_identity_or_counter_fails_closed(field, value):
    fake = Fake()
    if value == "missing":
        del fake.host_value[field]
    else:
        fake.host_value[field] = value
    result = run(fake)
    assert result.code == 78 and not fake.launched and not fake.acquired


@pytest.mark.parametrize("change", [
    {"available_estimate_bytes": 25 * g.GIB}, {"free_bytes": 23 * g.GIB},
    {"inactive_count": 1 << 32}, {"free_count_including_speculative": 1 << 32},
    {"speculative_count": 1 << 32}, {"physical_total_bytes": 1 << 64},
    {"physical_total_bytes": 23 * g.GIB}, {"physical_total_bytes": 0},
    {"speculative_count": 24 * g.GIB // 16384 + 1},
])
def test_headroom_formula_and_physical_bounds_fail_closed(change):
    with pytest.raises(g.Blocked):
        g.validated_headroom(host() | change)


def test_headroom_identity_consistent_with_baseline_and_previous():
    row = host()
    other = host(physical_total_bytes=32 * g.GIB)
    for baseline, previous in ((other, row), (row, other)):
        with pytest.raises(g.Blocked, match="identity changed"):
            g.validated_headroom(row, baseline, previous)
    changed_raw = host(free_bytes=66 * (1 << 20), available_estimate_bytes=24 * g.GIB)
    assert g.validated_headroom(changed_raw, row, row) == 24 * g.GIB


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_external_interrupt_progresses_to_full_cleanup(signum):
    fake = Fake()
    fake.external_signal = signum  # I/O adapter delivers either handler's flag
    fake.running = True
    result = run(fake)
    assert result.code == 1 and not result.held and "interrupted" in result.reason
    assert fake.killed == {"payload", "worker"}


def test_campaign_limited_deadline_does_not_create_new_budget():
    fake = Fake()
    fake.running = True
    result = run(fake, campaign_deadline=115)
    term = next(e for e in fake.events if e[:3] == ("signal", "payload", signal.SIGTERM))
    assert term[3] == 106  # 115 - existing 9-second qualified-cleanup reserve
    assert result.code == 1 and not result.held and fake.time < 115


def test_request_side_ipc_failure_never_attempts_final_filesystem_release():
    fake = Fake()
    fake.request_error = True
    result = run(fake)
    assert result.code == 78 and result.held and fake.killed == {"worker"}
    assert not any(e[0] == "release" for e in fake.events)


def test_partial_real_worker_initialization_uses_same_bounded_cleanup(monkeypatch):
    clock = Fake()
    adapter = g.Real()
    adapter.now, adapter.sleep = clock.now, clock.sleep
    events = []
    monkeypatch.setattr(g, "__name__", "__main__")
    monkeypatch.setattr(g.signal, "getsignal", lambda sig: signal.SIG_DFL)
    monkeypatch.setattr(g.signal, "pthread_sigmask", lambda *args: set())
    pairs = iter([(20, 21), (22, 23), (24, 25), (26, 27)])
    monkeypatch.setattr(g.os, "pipe", lambda: next(pairs))
    monkeypatch.setattr(g.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(g.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(g.os, "fork", lambda: 2)
    closed = []
    def close(fd):
        closed.append(fd)
        if fd == 21:
            raise OSError("partial parent FD retirement")
    monkeypatch.setattr(g.os, "close", close)
    monkeypatch.setattr(g.os, "kill", lambda pid, sig: events.append(("signal", pid, sig, clock.now())))
    monkeypatch.setattr(g.os, "killpg", lambda *args: pytest.fail("no confirmed group"))
    def reap(pid, options):
        assert options == g.os.WNOHANG
        events.append(("reap", pid, clock.now()))
        return pid, signal.SIGTERM
    monkeypatch.setattr(g.os, "waitpid", reap)
    with pytest.raises(OSError):
        adapter.worker([])
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), adapter, 1)
    assert result.code == 78 and result.held
    assert events[0][:3] == ("signal", 2, signal.SIGTERM)
    assert events[1][:3] == ("signal", 2, signal.SIGKILL)
    assert events[1][3] - events[0][3] >= 1
    assert events[2][0] == "reap"
    assert len(closed) == len(set(closed))  # no retry of possibly reused FD


def test_final_authority_failure_runs_only_after_qualified_shutdown_and_holds():
    fake = Fake()
    fake.release_error = True
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    release = next(i for i, e in enumerate(fake.events) if e[0] == "release")
    assert all(i < release for i, e in enumerate(fake.events) if e[0] == "reap")


@pytest.mark.parametrize("operation", ["init", "retire"])
@pytest.mark.parametrize("uncertain", [False, True])
def test_management_never_launches_payload_or_mutates_before_observer_qualification(operation, uncertain):
    fake = Fake()
    if uncertain:
        fake.worker_exit_code = 0
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), fake, 1, operation)
    assert not fake.launched and fake.killed == {"worker"}
    assert result.code == (78 if uncertain else 0) and result.held == uncertain
    assert any(e[0] == "release" for e in fake.events) == (not uncertain)


def wire(monkeypatch, data):
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.read_fd, endpoint.write_fd = 123, 124
    endpoint.buffer = bytearray()
    monkeypatch.setattr(g.os, "read", lambda fd, size: data)
    return endpoint


@pytest.mark.parametrize("data", [b"", b"[]\n", b"invalid\n", b"{}\n{}\n", b'{"x":NaN}\n'])
def test_wire_crash_malformed_and_nonfinite_rejected(monkeypatch, data):
    with pytest.raises((g.Blocked, ValueError)):
        wire(monkeypatch, data).read()


def test_wire_response_cap_and_each_read_are_bounded(monkeypatch):
    endpoint = wire(monkeypatch, b"x" * 65536)
    reads = []
    def read(fd, size):
        reads.append(size)
        return b"x" * size
    monkeypatch.setattr(g.os, "read", read)
    with pytest.raises(g.Blocked, match="oversize"):
        for _ in range(g.MAX_RESPONSE // 65536 + 1):
            endpoint.read()
    assert reads == [65536] * (g.MAX_RESPONSE // 65536 + 1)


def test_wire_partial_or_unavailable_response_never_blocks(monkeypatch):
    endpoint = wire(monkeypatch, b'{"sealed":')
    assert endpoint.read() is None
    def unavailable(*args):
        raise BlockingIOError()
    monkeypatch.setattr(g.os, "read", unavailable)
    assert endpoint.read() is None
    assert endpoint.buffer == b'{"sealed":'


def test_probe_helper_failure_wait_has_timeout_and_stays_in_worker_group(monkeypatch):
    from types import SimpleNamespace
    calls = []
    class Helper:
        pid = 77
        returncode = None
        def communicate(self, timeout):
            calls.append(("communicate", timeout))
            raise g.subprocess.TimeoutExpired("ps", timeout)
        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = -9
            return -9
    def popen(argv, **kwargs):
        assert "start_new_session" not in kwargs and "preexec_fn" not in kwargs
        calls.append(("popen", argv))
        return Helper()
    monkeypatch.setattr(g.subprocess, "Popen", popen)
    monkeypatch.setattr(g.os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))
    probes = g.DarwinProbes.__new__(g.DarwinProbes)
    probes.psutil = SimpleNamespace(Process=lambda pid: SimpleNamespace(create_time=lambda: 1.0))
    with pytest.raises(g.subprocess.TimeoutExpired):
        probes.processes()
    assert ("communicate", 1) in calls and ("wait", 1) in calls
    assert ("kill", 77, signal.SIGKILL) in calls


def make_repo(path):
    """Real Git layout, not git subprocesses; canonical_slot stays UNMOCKED."""
    (path / ".git/objects").mkdir(parents=True)
    (path / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (path / "scripts").mkdir()
    (path / "scripts/command_guard.py").write_text("# trusted test script anchor\n")
    for child in ("build", "log", "tmp", "cargo", "rustup", "uv"):
        (path / "target" / child).mkdir(parents=True)
    (path / "target").chmod(0o700)
    return path.resolve()


def settings(root, **changes):
    from types import SimpleNamespace
    values = dict(cwd=root, coordinator_root=root, slot=root / "target" / g.SLOT_NAME,
                  target_dir=root / "target/build", log_dir=root / "target/log",
                   temp_dir=root / "target/tmp", disk_root=[], campaign_deadline=1000,
                   campaign_init=True, campaign_retire=False, disk_growth_bytes=20 * g.GIB)
    values.update(changes)
    return SimpleNamespace(**values)


def environment(args):
    return {"TMPDIR": str(args.temp_dir), "CARGO_TARGET_DIR": str(args.target_dir),
            "CARGO_HOME": str(args.coordinator_root / "target/cargo"),
            "RUSTUP_HOME": str(args.coordinator_root / "target/rustup"),
            "UV_CACHE_DIR": str(args.coordinator_root / "target/uv")}


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = make_repo(tmp_path / "main")
    monkeypatch.setattr(g, "__file__", str(root / "scripts/command_guard.py"))
    # Exercise real paths/lstat/inodes on non-Darwin pure-test hosts too. Only
    # unavailable birth metadata is synthetic; canonical_slot is never mocked.
    original = g.identity
    def birth(s):
        dev, ino, born = original(s)
        return dev, ino, 1 if born is None else born
    monkeypatch.setattr(g, "identity", birth)
    return root


BOOT = [1700000000, 42]


def authority(args, now=100):
    return g.Authority(args, environment(args), BOOT, now)


def complete_metadata(a, now=105):
    """Only metadata primitive tested here, NOT a substitute for lifecycle loop."""
    try:
        a.acquire(now - 1)
        a.debit({}, {})  # first paid commit, including slot and future campaign
        a.debit({}, {})  # sealed final paid commit
        g.finish_authority(a.custody(), a.slot, now)
    finally:
        a.close()


def linked(root, path):
    path.mkdir()
    admin = root / ".git/worktrees/w1"
    admin.mkdir(parents=True)
    (path / ".git").write_text(f"gitdir: {admin}\n")
    (admin / "commondir").write_text("../..\n")
    (admin / "gitdir").write_text(f"{path / '.git'}\n")
    return path


def test_real_foreign_repo_pair_cannot_choose_an_independent_slot(repository, tmp_path):
    foreign = make_repo(tmp_path / "foreign")
    args = settings(foreign)
    assert g.canonical_slot(foreign) != g.canonical_slot(repository)
    with pytest.raises(g.Blocked, match="trusted script"):
        authority(args)
    assert not args.slot.exists()


def test_main_linked_and_stable_alias_share_trusted_slot_and_campaign(repository, tmp_path):
    work = linked(repository, tmp_path / "work")
    alias = tmp_path / "stable-alias"
    alias.symlink_to(work, target_is_directory=True)
    assert g.canonical_slot(alias) == g.canonical_slot(repository)
    complete_metadata(authority(settings(repository)))
    args = settings(repository, cwd=alias, campaign_init=False)
    a = authority(args, 110)
    try:
        assert a.slot == repository / "target" / g.SLOT_NAME
        assert a.campaign["origin"] == 100 and a.campaign["deadline"] == 1000
        assert str(alias) in a.pins.entries
        a.check(111)
    finally:
        a.close()


def test_real_git_linked_administration_symlink_rejected(repository, tmp_path):
    work = linked(repository, tmp_path / "work")
    admin = repository / ".git/worktrees/w1"
    redirect = repository / ".git/worktrees/redirect"
    redirect.symlink_to(admin, target_is_directory=True)
    (work / ".git").write_text(f"gitdir: {redirect}\n")
    with pytest.raises(g.Blocked, match="Git administration symlink"):
        g.canonical_slot(work)


@pytest.mark.parametrize("where", ["coordinator", "slot", "primary_root"])
def test_real_authority_and_primary_roots_refused_before_slot_write(repository, tmp_path, where):
    args = settings(repository)
    if where == "coordinator":
        args.coordinator_root = make_repo(tmp_path / "foreign")
    elif where == "slot":
        args.slot = tmp_path / "alternate.slot"
    else:
        args.log_dir = tmp_path
    with pytest.raises(g.Blocked):
        authority(args)
    assert not args.slot.exists()


@pytest.mark.parametrize("boundary", ["sample", "acquire", "release"])
def test_persistent_nested_log_replacement_not_erased_by_walk_dedup(repository, boundary):
    args = settings(repository)
    a = authority(args)
    try:
        assert args.log_dir not in g.disk_roots(a.paths)
        if boundary == "release":
            a.acquire(100.5)
        args.log_dir.rename(repository / "target/old-log")
        args.log_dir.mkdir()
        with pytest.raises(g.Blocked, match="pinned ancestry/root"):
            if boundary == "sample":
                a.check(101)
            elif boundary == "acquire":
                a.acquire(101)
            else:
                g.finish_authority(a.custody(), args.slot, 105)
        assert args.slot.exists() == (boundary == "release")
    finally:
        a.close()


def test_slot_replacement_before_release_never_deletes_foreign_file(repository):
    args = settings(repository)
    a = authority(args)
    try:
        a.acquire(101)
        args.slot.rename(repository / "target/original-held.slot")
        args.slot.write_text("FOREIGN SLOT")
        with pytest.raises(g.Blocked, match="foreign|replaced"):
            g.finish_authority(a.custody(), args.slot, 105)
        assert args.slot.read_text() == "FOREIGN SLOT"
        assert not (repository / "target" / g.CAMPAIGN).exists()
    finally:
        a.close()


def test_explicit_campaign_init_run_retire_and_fixed_second_deadline(repository):
    init = settings(repository)
    complete_metadata(authority(init))
    state = repository / "target" / g.CAMPAIGN
    original = state.read_bytes()
    with pytest.raises(g.Blocked, match="substituted"):
        authority(settings(repository, campaign_init=False, campaign_deadline=1100), 200)
    assert state.read_bytes() == original
    complete_metadata(authority(settings(repository, campaign_init=False), 200), 205)
    assert state.read_bytes() == original
    retire = settings(repository, campaign_init=False, campaign_retire=True)
    complete_metadata(authority(retire, 210), 215)
    assert not state.exists() and not init.slot.exists()


def test_expired_campaign_no_automatic_renewal_but_explicit_retirement(repository):
    complete_metadata(authority(settings(repository)))
    state = repository / "target" / g.CAMPAIGN
    original = state.read_bytes()
    with pytest.raises(g.Blocked, match="expired"):
        authority(settings(repository, campaign_init=False), 1001)
    with pytest.raises(g.Blocked, match="existing"):
        authority(settings(repository), 1001)
    assert state.read_bytes() == original
    complete_metadata(authority(settings(repository, campaign_init=False, campaign_retire=True), 1001), 1005)
    assert not state.exists()


@pytest.mark.parametrize("operation", ["init", "retire"])
def test_init_retire_never_overwrites_existing_global_slot(repository, operation):
    if operation == "retire":
        complete_metadata(authority(settings(repository)))
    args = settings(repository, campaign_init=operation == "init", campaign_retire=operation == "retire")
    args.slot.write_text("live or stale global slot")
    a = authority(args, 200)
    try:
        with pytest.raises(FileExistsError):
            a.acquire(201)
        assert args.slot.read_text() == "live or stale global slot"
    finally:
        a.close()


def test_campaign_uncertain_ownership_and_changed_state_hold(repository):
    complete_metadata(authority(settings(repository)))
    state = repository / "target" / g.CAMPAIGN
    state.chmod(0o644)
    with pytest.raises(g.Blocked, match="ownership"):
        authority(settings(repository, campaign_init=False, campaign_retire=True), 200)
    state.chmod(0o600)
    a = authority(settings(repository, campaign_init=False, campaign_retire=True), 200)
    try:
        a.acquire(201)
        state.rename(repository / "target/original.campaign")
        state.write_text("FOREIGN STATE")
        state.chmod(0o600)
        with pytest.raises((g.Blocked, ValueError)):
            g.finish_authority(a.custody(), a.slot, 205)
        assert state.read_text() == "FOREIGN STATE" and a.slot.exists()
    finally:
        a.close()


def test_relative_linked_git_references_preserve_owned_common_authority(repository, tmp_path):
    work = linked(repository, tmp_path / "work")
    admin = repository / ".git/worktrees/w1"
    (work / ".git").write_text("gitdir: ../main/.git/worktrees/w1\n")
    (admin / "gitdir").write_text("../../../../work/.git\n")
    assert g.canonical_slot(work) == g.canonical_slot(repository)
    a = authority(settings(repository, cwd=work))
    a.close()


def test_environment_alias_provenance_checked_before_acquisition(repository, tmp_path):
    args = settings(repository)
    alias = tmp_path / "temp-alias"
    alias.symlink_to(args.temp_dir, target_is_directory=True)
    env = environment(args) | {"TMPDIR": str(alias)}
    a = g.Authority(args, env, BOOT, 100)
    try:
        alias.unlink()
        alias.symlink_to(repository / "target/cargo", target_is_directory=True)
        with pytest.raises(g.Blocked, match="pinned ancestry/root"):
            a.acquire(101)
        assert not args.slot.exists()
    finally:
        a.close()


def test_git_administration_identity_is_pinned_during_samples(repository):
    a = authority(settings(repository))
    try:
        objects = repository / ".git/objects"
        objects.rename(repository / ".git/old-objects")
        objects.mkdir()
        with pytest.raises(g.Blocked, match="pinned ancestry/root"):
            a.check(101)
    finally:
        a.close()


@pytest.mark.parametrize("changed", [dict(boot=[1700000001, 42]), dict(origin=101),
                                   dict(deadline=7300.1), dict(boot=[1700000000, True])])
def test_campaign_boot_origin_and_two_hour_ceiling_fail_closed(changed):
    roots = dict(declarations=[["/root", "/root", [1, 2, 3]]],
                 pins=[["/root", [1, 2, 3, g.statmod.S_IFDIR, 1, None, None, 0o700]]])
    record = dict(boot=BOOT, origin=100, deadline=1000, disk_cap_bytes=20 * g.GIB,
                  budget_identity=[1, 4, 5], roots=roots) | changed
    with pytest.raises(g.Blocked):
        g.validate_campaign(record, BOOT, 100)


def test_disk_scanner_hardlinks_and_rename_count_one_inode(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    root = Path("/target")
    monkeypatch.setattr(g, "disk_roots", lambda paths: (root,))
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_dev=1, st_ino=2, st_birthtime=3))
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    size = [100]
    def entry(name):
        return SimpleNamespace(path=f"/target/{name}", is_symlink=lambda: False,
            is_dir=lambda **kwargs: False, is_file=lambda **kwargs: True,
            stat=lambda **kwargs: SimpleNamespace(st_dev=1, st_ino=20, st_birthtime=4,
                st_size=size[0], st_blocks=0, st_ctime_ns=size[0]))
    entries = [entry("a"), entry("hardlink")]
    monkeypatch.setattr(g.os, "scandir", lambda path: nullcontext(iter(entries)))
    disk = g.Disk([root])
    assert disk.snapshot() == 0
    size[0] = 200
    assert disk.snapshot() == 100
    entries[:] = [entry("renamed")]
    assert disk.snapshot() == 100


def test_undeclared_outward_symlink_and_unknown_root_birth_block(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    root = Path("/target")
    monkeypatch.setattr(g, "disk_roots", lambda paths: (root,))
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_dev=1, st_ino=2, st_birthtime=3))
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: Path("/outside"))
    item = SimpleNamespace(path="/target/link", is_symlink=lambda: True)
    monkeypatch.setattr(g.os, "scandir", lambda path: nullcontext(iter([item])))
    with pytest.raises(g.Blocked, match="outward symlink"):
        g.Disk([root]).snapshot()
    monkeypatch.setattr(Path, "stat", lambda self: SimpleNamespace(st_dev=1, st_ino=2))
    with pytest.raises(g.Blocked, match="birth identity unavailable"):
        g.Disk([root])


def test_disk_root_dedup_and_filesystem_root_refusal(monkeypatch):
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    assert g.disk_roots(["/target/nested", "/target", "/cache", "/target"]) == (
        Path("/cache"), Path("/target"))
    with pytest.raises(g.Blocked, match="filesystem root"):
        g.disk_roots(["/"])


@pytest.mark.parametrize("key", ["GIT_DIR", "GIT_COMMON_DIR", "GIT_WORK_TREE"])
@pytest.mark.parametrize("value", ["/foreign", ""])
def test_git_environment_redirect_cannot_change_authority(repository, key, value):
    args = settings(repository)
    with pytest.raises(g.Blocked, match="redirect"):
        g.Authority(args, environment(args) | {key: value}, BOOT, 100)
    assert not args.slot.exists()


@pytest.mark.parametrize("orphan", [g.CAMPAIGN, g.BUDGET])
def test_init_requires_both_state_files_absent(repository, orphan):
    state = repository / "target" / orphan
    state.write_text("orphan, never adopt")
    state.chmod(0o600)
    with pytest.raises(g.Blocked, match="existing"):
        authority(settings(repository))
    assert state.read_text() == "orphan, never adopt"


def debit_metadata(a, files, baseline=None):
    return a.debit(files, files if baseline is None else baseline)


def test_campaign_a12_b12_between_commands_refuses_before_second_payload(repository):
    a = authority(settings(repository))
    key = (7, 8, 9)
    try:
        a.acquire(101)
        debit_metadata(a, {})
        debit_metadata(a, {key: (12 * g.GIB, 1)}, {})
        g.finish_authority(a.custody(), a.slot, 105)
    finally:
        a.close()
    second = authority(settings(repository, campaign_init=False), 200)
    try:
        second.acquire(201)
        with pytest.raises(g.Blocked, match="cannot fund"):
            debit_metadata(second, {key: (24 * g.GIB, 2)})
        assert second.slot.exists()  # exhaustion retains custody, no second admission
    finally:
        second.close()


def test_persistent_census_truncate_regrow_delete_recreate_and_control_charges(repository):
    a = authority(settings(repository))
    key = (7, 8, 9)
    try:
        a.acquire(101)
        first = debit_metadata(a, {key: (100, 1)})
        assert first == g.SLOT_CHARGE + g.CAMPAIGN_CHARGE + g.WRITE_CHARGE
        total = debit_metadata(a, {key: (200, 2)})
        assert total == first + 100 + g.WRITE_CHARGE
        g.finish_authority(a.custody(), a.slot, 105)
    finally:
        a.close()
    b = authority(settings(repository, campaign_init=False), 200)
    try:
        assert debit_metadata(b, {key: (50, 3)}) == total  # no shrink refund
        b.acquire(201)
        total2 = debit_metadata(b, {key: (100, 4)})
        assert total2 == total + 50 + g.SLOT_CHARGE + g.WRITE_CHARGE
        total3 = debit_metadata(b, {})
        assert total3 == total2 + g.WRITE_CHARGE  # same-size rewrites still pay
        total4 = debit_metadata(b, {key: (75, 5)})
        assert total4 == total3 + 75 + g.WRITE_CHARGE
        total5 = debit_metadata(b, {(7, 8, 10): (80, 6)})
        assert total5 == total4 + 80 + g.WRITE_CHARGE
        g.finish_authority(b.custody(), b.slot, 205)
    finally:
        b.close()


def test_unchanged_preexisting_cache_zero_then_growth_across_commands(repository):
    key = (7, 8, 9)
    a = authority(settings(repository))
    try:
        a.acquire(101)
        total = debit_metadata(a, {key: (12 * g.GIB, 1)})
        assert total == g.SLOT_CHARGE + g.CAMPAIGN_CHARGE + g.WRITE_CHARGE
        final = debit_metadata(a, {key: (12 * g.GIB, 1)})
        assert final == total + g.WRITE_CHARGE
        g.finish_authority(a.custody(), a.slot, 105)
    finally:
        a.close()
    b = authority(settings(repository, campaign_init=False), 200)
    try:
        assert b.budget.record["charged_bytes"] == final
        assert debit_metadata(b, {key: (12 * g.GIB + 1, 2)}) == final + 1
    finally:
        b.close()


@pytest.mark.parametrize("over", [0, 1])
def test_control_charge_exact_cap_and_one_over_no_unfunded_write(repository, monkeypatch, over):
    cap = g.SLOT_CHARGE + g.CAMPAIGN_CHARGE + g.WRITE_CHARGE
    a = authority(settings(repository, disk_growth_bytes=cap))
    writes = []
    original = g.os.write
    try:
        a.acquire(101)
        monkeypatch.setattr(g.os, "write", lambda fd, raw: (writes.append(len(raw)), original(fd, raw))[1])
        files = {(7, 8, 9): (over, 1)}
        if over:
            with pytest.raises(g.Blocked, match="unfunded"):
                a.debit(files, {})
            assert not writes and not (repository / "target" / g.BUDGET).exists()
        else:
            assert a.debit(files, {}) == cap
            assert len(writes) == 1
            with pytest.raises(g.Blocked, match="cannot fund"):
                a.debit(files, {})
            assert len(writes) == 1  # refusal never writes another refusal
        assert a.slot.exists()
    finally:
        a.close()


@pytest.mark.parametrize("failure", ["partial", "fsync", "allocation"])
def test_budget_mutation_uncertainty_retains_original_slot(repository, monkeypatch, failure):
    a = authority(settings(repository))
    try:
        a.acquire(101)
        if failure == "partial":
            monkeypatch.setattr(g.os, "write", lambda fd, raw: len(raw) - 1)
        elif failure == "fsync":
            monkeypatch.setattr(g.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("fsync")))
        else:
            original = g.footprint
            def excess(s):
                if a.budget is not None and list(g.identity(s)) == a.budget.stable and s.st_size:
                    return g.WRITE_CHARGE + 1  # AFTER the funded overwrite
                return original(s)
            monkeypatch.setattr(g, "footprint", excess)
        with pytest.raises((g.Blocked, OSError)):
            a.debit({}, {})
        assert a.slot.exists() and not (repository / "target" / g.CAMPAIGN).exists()
    finally:
        a.close()


@pytest.mark.parametrize("mutation", ["missing", "replaced", "foreign", "duplicate", "duplicate_inode",
                                      "unknown", "cap", "roots", "oversize"])
def test_budget_load_rejects_uncertain_or_altered_state(repository, mutation):
    complete_metadata(authority(settings(repository)))
    path = repository / "target" / g.BUDGET
    record = g.decode_state(path.read_bytes())
    if mutation == "missing":
        path.unlink()
    elif mutation == "replaced":
        raw = path.read_bytes()
        path.rename(repository / "target/old-budget")
        path.write_bytes(raw)
        path.chmod(0o600)
    elif mutation == "foreign":
        path.chmod(0o644)
    elif mutation == "duplicate":
        path.write_bytes(b'{"version":1,"version":1}')
    elif mutation == "oversize":
        path.write_bytes(b" " * (g.STATE_LIMIT + 1))
    else:
        if mutation == "cap":
            record["campaign"]["disk_cap_bytes"] += 1
        elif mutation == "unknown":
            record["extra"] = 1
        elif mutation == "duplicate_inode":
            record["previous"] = [[7, 8, 9, 10, 11], [7, 8, 9, 10, 11]]
        else:
            record["roots"]["declarations"].pop()
        path.write_bytes(g.encode_state(record))
    with pytest.raises((g.Blocked, OSError, ValueError)):
        authority(settings(repository, campaign_init=False), 200)


@pytest.mark.parametrize("target", [g.SLOT_NAME, g.CAMPAIGN])
def test_partial_control_write_never_releases_slot(repository, monkeypatch, target):
    a = authority(settings(repository))
    try:
        if target == g.SLOT_NAME:
            monkeypatch.setattr(g.os, "write", lambda fd, raw: len(raw) - 1)
            with pytest.raises(g.Blocked, match="partial"):
                a.acquire(101)
        else:
            a.acquire(101)
            a.debit({}, {})
            monkeypatch.setattr(g.os, "write", lambda fd, raw: len(raw) - 1)
            with pytest.raises(g.Blocked, match="partial"):
                g.finish_authority(a.custody(), a.slot, 105)
        assert a.slot.exists()
    finally:
        a.close()


def test_ready_partial_go_eagain_and_cloexec_transition_are_not_payload_success(monkeypatch):
    child = g.OwnedChild(42, 20, 23, fds={20, 23})
    chunks = iter([b"R:", b"42\n", BlockingIOError(), BlockingIOError(), b""])
    writes = iter([1, BlockingIOError(), 2])
    sent, closed = [], []
    def read(fd, size):
        value = next(chunks)
        if isinstance(value, BaseException):
            raise value
        return value
    def write(fd, raw):
        value = next(writes)
        sent.append(raw)
        if isinstance(value, BaseException):
            raise value
        return value
    monkeypatch.setattr(g.os, "read", read)
    monkeypatch.setattr(g.os, "write", write)
    monkeypatch.setattr(g.os, "close", closed.append)
    assert not child.advance(True) and not child.confirmed_group
    assert not child.advance(True) and child.go_sent == 1
    assert not child.advance(True) and child.go_sent == 1
    assert not child.advance(True) and child.go_sent == 3
    assert child.advance(True) and child.phase == "executed"
    assert child.returncode is None and set(closed) == {20, 23}
    assert sent == [b"GO\n", b"O\n", b"O\n"]


@pytest.mark.parametrize("chunks,allow", [([b""], True), ([b"R:42", b""], True),
    ([b"R:42\nR:42\n"], True), ([b"R:99\n"], True), ([b"R:42\n"], False),
    ([b"R:42\nR", b":42\n"], True), ([b"E:2\n"], True), ([b"R:42\n", b"E:2\n"], True)])
def test_startup_error_partial_duplicate_foreign_late_or_early_eof_holds(monkeypatch, chunks, allow):
    child = g.OwnedChild(42, 20, 23)
    values = iter(chunks)
    monkeypatch.setattr(g.os, "read", lambda *args: next(values))
    monkeypatch.setattr(g.os, "write", lambda fd, raw: len(raw))
    with pytest.raises(g.Blocked):
        for _ in chunks:
            child.advance(allow)
    assert child.returncode is None  # no implicit reap or birth inference


@pytest.mark.parametrize("ready", [False, True])
def test_setup_or_exec_hang_uses_pid_then_confirmed_group_and_deadline(monkeypatch, ready):
    clock = Fake()
    adapter = g.Real()
    adapter.now, adapter.sleep = clock.now, clock.sleep
    adapter.children["worker"] = g.OwnedChild(2, 20, 23, fds={20, 23})
    first = [ready]
    events = []
    def read(*args):
        if first[0]:
            first[0] = False
            return b"R:2\n"
        raise BlockingIOError()
    monkeypatch.setattr(g.os, "read", read)
    monkeypatch.setattr(g.os, "write", lambda fd, raw: len(raw))
    monkeypatch.setattr(g.os, "kill", lambda pid, sig: events.append(("pid", pid, sig, clock.now())))
    monkeypatch.setattr(g.os, "killpg", lambda pid, sig: events.append(("group", pid, sig, clock.now())))
    monkeypatch.setattr(g.os, "waitpid", lambda pid, options: (pid, signal.SIGTERM))
    monkeypatch.setattr(g.os, "close", lambda fd: events.append(("close", fd)))
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000, seconds=0.5), adapter, 1)
    assert result.code == 78 and result.held
    signals = [e for e in events if e[0] in ("pid", "group")]
    assert [e[0] for e in signals] == ["group" if ready else "pid"] * 2
    assert signals[0][3] <= 100.55 and signals[1][3] - signals[0][3] >= 1
    assert adapter.children["worker"].returncode == -signal.SIGTERM


def test_payload_startup_wait_is_inside_same_cancellation_deadline():
    fake = Fake()
    fake.startup_step = lambda allow_go=True: not fake.launched
    result = run(fake, seconds=0.5)
    assert result.code == 78 and result.held and fake.killed == {"worker", "payload"}
    term = next(e for e in fake.events if e[:3] == ("signal", "payload", signal.SIGTERM))
    assert term[3] <= 100.55 and fake.acquired


def test_fork_registration_precedes_unmask_cancel_and_no_go_or_reused_signal(monkeypatch):
    adapter, clock = g.Real(), Fake()
    adapter.now, adapter.sleep = clock.now, clock.sleep
    monkeypatch.setattr(g, "__name__", "__main__")
    monkeypatch.setattr(g.signal, "getsignal", lambda sig: signal.SIG_DFL)
    def mask(how, values):
        if how == signal.SIG_SETMASK:
            assert adapter.children["worker"].pid == 2
            adapter.interrupted = True  # pending INT/TERM delivered after registration
        return set()
    monkeypatch.setattr(g.signal, "pthread_sigmask", mask)
    fds = iter([(20, 21), (22, 23), (24, 25), (26, 27)])
    monkeypatch.setattr(g.os, "pipe", lambda: next(fds))
    flags, closed, signals = [], [], []
    monkeypatch.setattr(g.os, "set_inheritable", lambda fd, value: flags.append((fd, value)))
    monkeypatch.setattr(g.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(g.os, "fork", lambda: 2)
    monkeypatch.setattr(g.os, "close", closed.append)
    monkeypatch.setattr(g.os, "write", lambda *args: pytest.fail("no GO before custody/cancellation check"))
    monkeypatch.setattr(g.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(g.os, "killpg", lambda *args: pytest.fail("group never confirmed"))
    monkeypatch.setattr(g.os, "waitpid", lambda pid, options: (pid, signal.SIGTERM))
    assert adapter.worker([]) == 2
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), adapter, 1)
    assert result.code == 78 and result.held
    assert signals == [(2, signal.SIGTERM), (2, signal.SIGKILL)]
    assert len(closed) == len(set(closed)) == 8
    assert all(value is False for fd, value in flags)
    with pytest.raises(g.Blocked, match="reaped"):
        adapter.signal("worker", signal.SIGKILL)
    assert len(signals) == 2


@pytest.mark.parametrize("fault", ["signal", "reap", "close"])
def test_owned_terminal_exceptions_never_release_custody(fault):
    fake = Fake()
    def fail(*args):
        raise OSError(fault)
    setattr(fake, fault, fail)
    result = run(fake)
    assert result.code == 78 and result.held
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("fault", ["short_request", "sealed", "late"])
def test_real_wire_failure_or_late_sealed_mismatch_after_payload_never_releases(monkeypatch, fault):
    fake = Fake()
    old_request, old_read = fake.request, fake.read
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.write_fd, endpoint.buffer = 123, bytearray()
    monkeypatch.setattr(g.os, "write", lambda fd, raw: 1)
    def request(final, acquire=False):
        if fake.launched and fault == "short_request":
            endpoint.request(final, acquire)  # actual production framing check
        old_request(final, acquire)
    def read():
        data = old_read()
        if fake.final:
            if fault == "sealed":
                data["sealed"] = False
            elif fault == "late":
                fake.time += 2
        return data
    fake.request, fake.read = request, read
    result = run(fake)
    assert result.code == 78 and result.held and fake.launched and fake.acquired
    assert not any(e[0] == "release" for e in fake.events)


def test_final_budget_replacement_after_qualified_sample_never_releases_slot(repository):
    complete_metadata(authority(settings(repository)))
    a = authority(settings(repository, campaign_init=False), 200)
    try:
        a.acquire(201)
        a.debit({}, {})
        custody = a.custody()
        path = repository / "target" / g.BUDGET
        raw = path.read_bytes()
        path.rename(repository / "target/original-budget")
        path.write_bytes(raw)
        path.chmod(0o600)
        with pytest.raises(g.Blocked, match="identity|inode|birth"):
            g.finish_authority(custody, a.slot, 205)
        assert a.slot.exists() and path.read_bytes() == raw
    finally:
        a.close()


def test_parent_fork_failure_restores_mask_and_closes_prepared_fds(monkeypatch):
    adapter = g.Real()
    monkeypatch.setattr(g, "__name__", "__main__")
    monkeypatch.setattr(g.signal, "getsignal", lambda sig: signal.SIG_DFL)
    masks, closed = [], []
    monkeypatch.setattr(g.signal, "pthread_sigmask", lambda how, values: (masks.append(how), set())[1])
    pairs = iter([(20, 21), (22, 23)])
    monkeypatch.setattr(g.os, "pipe", lambda: next(pairs))
    monkeypatch.setattr(g.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(g.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(g.os, "close", closed.append)
    monkeypatch.setattr(g.os, "fork", lambda: (_ for _ in ()).throw(OSError("fork failed")))
    with pytest.raises(OSError):
        adapter.launch(["cargo"], Path("/work"), {})
    assert not adapter.children and sorted(closed) == [20, 21, 22, 23]
    assert masks[-1] == signal.SIG_SETMASK


def test_imported_embedding_refuses_before_any_fork(monkeypatch):
    monkeypatch.setattr(g.os, "fork", lambda: pytest.fail("unsupported embedding forked"))
    with pytest.raises(g.Blocked, match="standalone"):
        g.Real().worker([])


def test_command_start_is_not_reset_after_parent_startup_returns():
    fake = Fake()
    fake.time = 701
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), fake, 1, start=100)
    assert result.code == 78 and result.held and not fake.launched
    assert not any(e[0] == "request" for e in fake.events)


def test_stable_hardlink_or_rename_identity_across_commands_has_no_new_data_debit(repository):
    key = (7, 8, 9)
    a = authority(settings(repository))
    try:
        a.acquire(101)
        total = a.debit({key: (100, 1)}, {key: (100, 1)})
        g.finish_authority(a.custody(), a.slot, 105)
    finally:
        a.close()
    b = authority(settings(repository, campaign_init=False), 200)
    try:
        b.acquire(201)
        assert b.debit({key: (100, 2)}, {}) == total + g.SLOT_CHARGE + g.WRITE_CHARGE
        # Paths/hardlink aliases are intentionally not duplicated in unique census.
        assert len(b.budget.record["previous"]) == 1
    finally:
        b.close()


def test_logical_and_allocated_footprint_use_same_conservative_meter():
    from types import SimpleNamespace
    assert g.footprint(SimpleNamespace(st_size=1, st_blocks=2)) == 1024
    assert g.footprint(SimpleNamespace(st_size=2048, st_blocks=1)) == 2048
    for size, blocks in ((-1, 0), (0, -1), (True, 1), (1, 1.0)):
        with pytest.raises(g.Blocked):
            g.footprint(SimpleNamespace(st_size=size, st_blocks=blocks))


@pytest.mark.parametrize("phase,label", [("S", "status"), ("A", "acquisition"), ("F", "final-seal")])
@pytest.mark.parametrize("fault", ["stall", "malformed", "request"])
def test_diagnostic_phase_starts_before_request_and_only_validated_response_completes(monkeypatch, phase, label, fault):
    from io import StringIO
    # Mock semantic emission sink; this does not exercise stderr backpressure.
    sink = StringIO()
    monkeypatch.setattr(g, "diagnostic", lambda message: sink.write(message + "\n"))
    fake = Fake()
    original_request, original_read = fake.request, fake.read
    active = [None]
    observations = []
    def request(final, acquire=False):
        active[0] = "A" if acquire else "F" if final else "S"
        if active[0] == phase:
            # Record at actual entry; positive oracles belong outside run's catch.
            observations.append((active[0], sink.getvalue()))
            if fault == "request":
                raise g.Blocked("request-side IPC failure")
        original_request(final, acquire)
    def read():
        if active[0] == phase:
            if fault == "stall":
                return None
            return {"sealed": fake.final}
        return original_read()
    fake.request, fake.read = request, read
    result = run(fake)
    stderr = sink.getvalue()
    assert len(observations) == 1
    observed_phase, at_request = observations[0]
    assert observed_phase == phase
    assert f"telemetry {phase} {label} start" in at_request
    assert f"telemetry {phase} {label} completed" not in at_request
    assert result.code == 78 and result.held
    assert f"telemetry {phase} {label} completed" not in stderr
    assert not any(e[0] == "release" for e in fake.events)
    if phase == "S":
        assert "identities monitor=unknown worker=unknown boot=unknown" in stderr
        assert "monitor_pid=" not in stderr
    else:
        assert "identities monitor_pid=1 monitor_birth=1.0 worker_pid=2 worker_birth=2.0" in at_request
        assert "boot=unknown command_deadline=700.0 control_deadline=1000" in at_request
        assert "identities monitor=unknown" not in stderr
    if phase == "A":
        assert fake.slot_pending and not fake.acquired
        assert "slot custody=pending/unknown caller_HOLD=True" in stderr
        assert "cleaned" not in stderr and "slot reaped" not in stderr


@pytest.mark.parametrize("phase,fault", [
    (phase, fault) for phase in ("S", "A", "F") for fault in ("request", "stall", "invalid")
] + [("F", "helper"), (None, "launch")])
def test_diagnostic_terminal_reason_preserves_failed_phase(monkeypatch, phase, fault):
    # Emission deliberately unavailable: terminal reason must retain the phase.
    monkeypatch.setattr(g, "diagnostic", lambda message: False)
    fake = Fake()
    original_request, original_read = fake.request, fake.read
    active = [None]
    requests = []
    injected = []
    def request(final, acquire=False):
        active[0] = "A" if acquire else "F" if final else "S"
        requests.append(active[0])
        if active[0] == phase and fault == "request":
            injected.append(fault)
            raise g.Blocked("mock request failure")
        original_request(final, acquire)
    def read():
        if active[0] == phase:
            if fault == "stall":
                injected.append(fault)
                return None
            if fault == "invalid":
                injected.append(fault)
                return {"sealed": not fake.final}
        data = original_read()
        if active[0] == phase and fault == "helper":
            injected.append(fault)
            data["reaped_helpers"] = [[999, 999.0]]
        return data
    def launch(argv, cwd, env):
        fake.events.append(("launch", argv, cwd, env))
        injected.append("launch")
        raise g.Blocked("mock launch failure")
    fake.request, fake.read = request, read
    if fault == "launch":
        fake.launch = launch
    result = run(fake)
    labels = {"S": "S status", "A": "A acquisition", "F": "F final-seal"}
    errors = {"request": "mock request failure", "stall": "probe timeout/stall",
              "invalid": "late/malformed telemetry response",
              "helper": "unqualified reaped-helper identity", "launch": "mock launch failure"}
    assert injected and set(injected) == {fault}
    assert result.code == 78 and result.held
    if phase is None:
        assert requests == ["S", "A"]
        assert "telemetry/control failure: mock launch failure" in result.reason
        assert all(f"[{label}]" not in result.reason for label in labels.values())
    else:
        assert requests[-1] == phase
        assert f"telemetry/control failure [{labels[phase]}]: {errors[fault]}" in result.reason
    launches = [event for event in fake.events if event[0] == "launch"]
    assert len(launches) == (0 if phase in ("S", "A") else 1)
    owned = {"worker", "payload"} if phase == "F" else {"worker"}
    reaped = [event[1] for event in fake.events if event[0] == "reap"]
    assert sorted(reaped) == sorted(owned)
    assert fake.reaped == owned
    assert not any(event[0] == "release" for event in fake.events)
    if phase == "A":
        assert fake.slot_pending and not fake.acquired
        assert "slot retained/ownership unresolved" in result.reason
    elif phase == "S":
        assert "no validated slot acquired; caller HOLD required" in result.reason
    else:
        assert fake.acquired and "slot retained/ownership unresolved" in result.reason


def test_diagnostic_completed_phases_follow_request_and_validation_in_order(monkeypatch):
    # Mock semantic emission sink only; no producer threads or real stderr I/O.
    fake = Fake()
    events = []
    labels = {"S": "status", "A": "acquisition", "F": "final-seal"}
    original_request, original_accept = fake.request, fake.accept
    active = [None]
    def emit(message):
        if message.startswith("telemetry "):
            events.append(("emit", message))
    def request(final, acquire=False):
        active[0] = "A" if acquire else "F" if final else "S"
        events.append(("request", active[0]))
        original_request(final, acquire)
    def accept(data, acquire=False):
        original_accept(data, acquire)
        events.append(("validated", active[0]))
    monkeypatch.setattr(g, "diagnostic", emit)
    fake.request, fake.accept = request, accept
    result = run(fake)
    assert result.code == 0 and not result.held
    phases = [value for kind, value in events if kind == "request"]
    assert phases[:2] == ["S", "A"] and phases[-1] == "F"
    assert phases.count("A") == phases.count("F") == 1
    assert set(phases) == set(labels)
    expected = []
    for phase in phases:
        expected.extend([
            ("emit", f"telemetry {phase} {labels[phase]} start"),
            ("request", phase),
            ("validated", phase),
            ("emit", f"telemetry {phase} {labels[phase]} completed"),
        ])
    # Exact comparison rejects missing, delayed, duplicated or wrong-stage completion.
    assert events == expected


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered", "lost"])
@pytest.mark.parametrize("site", ["context", "argv", "control", "success", "hold-A", "hold-F"])
def test_main_reporting_never_uses_blocking_textio(monkeypatch, site, delivered):
    from types import SimpleNamespace

    class StrictTextIO:
        def write(self, text):
            raise AssertionError("guard reporting must not call TextIO.write")

        def flush(self):
            raise AssertionError("guard reporting must not call TextIO.flush")

    class UnformattableError(Exception):
        def __str__(self):
            raise AssertionError("reporting must not format arbitrary exceptions")

    fake = Fake()
    events = []
    messages = []
    held = site in ("hold-A", "hold-F")
    reason = (f"BLOCKED slot retained/ownership unresolved: telemetry/control failure "
              f"[{'A acquisition' if site == 'hold-A' else 'F final-seal'}]: mock failure"
              if held else "command exit 0; qualified cleanup")
    outcome = g.Outcome(78 if held else 0, held, reason)
    expected = {
        "context": "BLOCKED: execution context refused; exception details omitted",
        "argv": "BLOCKED: reviewed cooperative argv after -- required",
        "control": "BLOCKED slot retained if acquired: control failure; exception details omitted",
    }.get(site, reason)

    class MockReal:
        @staticmethod
        def context():
            events.append("context")
            if site == "context":
                raise UnformattableError()

        def __new__(cls):
            events.append("adapter")
            return fake

    def worker(options):
        events.append(("worker", options))
        if site == "control":
            # Simulate unresolved custody; reporting cannot release or rewrite it.
            fake.acquired, fake.slot_pending = True, True
            raise UnformattableError()

    def supervise(argv, cwd, env, slot, limits, adapter, monitor_pid, operation, started):
        assert adapter is fake and slot == SLOT and monitor_pid == 1
        assert argv == ["mock-command"] and cwd == Path("/work") and operation == "run"
        assert started == 100 and limits == g.Limits(campaign_deadline=1000)
        events.append("run")
        fake.acquired, fake.slot_pending = held, held
        if not held:
            fake.reaped = {"worker", "payload"}
            fake.launched = True
            fake.release(slot)
        events.append(("outcome", outcome))
        return outcome

    def emit(message):
        messages.append(message)
        events.append("diagnostic")
        return delivered

    args = SimpleNamespace(
        _telemetry=False, campaign_init=False, campaign_retire=False,
        cooperative_build=site != "argv", argv=["--", "mock-command"],
        campaign_deadline=1000, timeout=600, rss_bytes=8 * g.GIB,
        reserve_bytes=16 * g.GIB, disk_growth_bytes=20 * g.GIB,
        cwd=Path("/work"), slot=SLOT,
    )
    fake.worker = worker
    fake.children = {}
    monkeypatch.setattr(g, "Real", MockReal)
    monkeypatch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g, "run", supervise)
    monkeypatch.setattr(g, "diagnostic", emit)
    monkeypatch.setattr(g.sys, "stderr", StrictTextIO())
    monkeypatch.setattr(g.sys, "argv", ["command_guard.py", "--", "mock-command"])
    monkeypatch.setattr(g.os, "getpid", lambda: 1)
    monkeypatch.setattr(g.signal, "signal", lambda *args: None)

    assert g.main() == (0 if site == "success" else 78)
    assert messages == [expected]
    assert events[-1] == "diagnostic"
    if site in ("success", "hold-A", "hold-F"):
        assert events[-2] == ("outcome", outcome)
        assert outcome == g.Outcome(78 if held else 0, held, reason)
        assert (fake.acquired, fake.slot_pending) == (held, held)
        assert any(event[0] == "release" for event in fake.events) == (not held)
        assert fake.reaped == (set() if held else {"worker", "payload"})
    elif site == "control":
        assert (fake.acquired, fake.slot_pending) == (True, True)
        assert not fake.events and "run" not in events
    else:
        assert not fake.acquired and not fake.slot_pending and not fake.events
        assert "adapter" not in events and "run" not in events


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered", "lost"])
def test_worker_failure_reporting_never_uses_blocking_textio(monkeypatch, delivered):
    class StrictTextIO:
        def write(self, text):
            raise AssertionError("worker failure must not call TextIO.write")

        def flush(self):
            raise AssertionError("worker failure must not call TextIO.flush")

    class UnformattableError(Exception):
        def __str__(self):
            raise AssertionError("reporting must not format arbitrary exceptions")

    messages = []
    calls = []

    def probes(args):
        calls.append(args)
        raise UnformattableError()

    def emit(message):
        messages.append(message)
        return delivered

    args = object()
    monkeypatch.setattr(g, "DarwinProbes", probes)
    monkeypatch.setattr(g, "diagnostic", emit)
    monkeypatch.setattr(g.sys, "stderr", StrictTextIO())
    monkeypatch.setattr(g.sys, "stdout", StrictTextIO())
    monkeypatch.setattr(g.sys, "stdin", object())  # Failure precedes protocol access.
    assert g.telemetry_worker(args) == 78
    assert calls == [args]
    assert messages == ["telemetry worker failed: exception details omitted"]


@pytest.mark.parametrize("phase", ["A", "F"])
def test_diagnostic_failure_preserves_prior_validated_identity_and_deadline(diagnostic_messages, phase):
    fake = Fake()
    original = fake.read
    def read():
        if (phase == "A" and fake.acquiring) or (phase == "F" and fake.final):
            return None
        return original()
    fake.read = read
    result = run(fake)
    stderr = "\n".join(diagnostic_messages)
    assert result.code == 78 and result.held
    assert stderr.count("identities monitor_pid=1 monitor_birth=1.0 worker_pid=2 worker_birth=2.0") == 1
    assert "boot=unknown command_deadline=700.0 control_deadline=1000" in stderr
    assert stderr.index("monitor_pid=1") < stderr.index(f"telemetry {phase} ")
    assert "caller_HOLD=True" in stderr
    assert not any(e[0] == "release" for e in fake.events)


@pytest.fixture
def diagnostic_messages(monkeypatch):
    """Semantic emission capture only, not evidence of real stderr delivery."""
    messages = []
    def emit(message):
        messages.append(message)
        return True
    monkeypatch.setattr(g, "diagnostic", emit)
    return messages


@pytest.mark.parametrize("kind,known,returned,error,timeout,expected", [
    ("success", True, "0", False, False, "expected"),
    ("exit", True, "3", False, False, "unexpected"),
    ("timeout", False, "unknown", False, True, "unknown"),
    ("error", False, "unknown", True, True, "unknown"),
])
def test_diagnostic_reap_reports_actual_returns_or_unknown_custody(diagnostic_messages, kind, known, returned, error, timeout, expected):
    fake = Fake()
    if kind == "exit":
        fake.exit_code = 3
    elif kind == "timeout":
        fake.reap_hang = True
    elif kind == "error":
        def reap(name):
            raise OSError("reap failed")
        fake.reap = reap
    result = run(fake)
    stderr = "\n".join(diagnostic_messages)
    assert (f"reap payload terminal_known={known} returncode={returned} "
            f"reap_error_observed={error} wait_timeout={timeout} expected_exit={expected}") in stderr
    if kind in ("timeout", "error"):
        assert "expected_exit=unknown custody=pending/unknown" in stderr
        assert result.code == 78 and result.held
        assert not any(e[0] == "release" for e in fake.events)
    else:
        assert ("reap worker terminal_known=True returncode=-15 reap_error_observed=False "
                "wait_timeout=False expected_exit=expected") in stderr
        assert result.code == (1 if kind == "exit" else 0) and not result.held


def test_diagnostic_reap_error_stays_visible_after_later_terminal_return(diagnostic_messages):
    fake = Fake()
    original = fake.reap
    failed = []
    def reap(name):
        if name == "worker" and not failed:
            failed.append(name)
            raise OSError("transient wait error")
        return original(name)
    fake.reap = reap
    result = run(fake)
    assert failed == ["worker"]
    assert ("reap worker terminal_known=True returncode=-15 reap_error_observed=True "
            "wait_timeout=False expected_exit=expected custody=terminal_known") in diagnostic_messages
    assert result.code == 78 and result.held
    assert not any(e[0] == "release" for e in fake.events)


def test_diagnostic_failed_acquisition_ack_is_not_cleanup_authority(diagnostic_messages):
    fake = Fake()
    original = fake.accept
    def accept(data, acquire=False):
        if acquire:
            raise g.Blocked("acquisition not qualified")
        original(data, acquire)
    fake.accept = accept
    result = run(fake)
    stderr = "\n".join(diagnostic_messages)
    assert result.code == 78 and result.held and fake.slot_pending
    assert "telemetry A acquisition start" in stderr
    assert "telemetry A acquisition completed" not in stderr
    assert ("reap worker terminal_known=True returncode=-15 reap_error_observed=False "
            "wait_timeout=False expected_exit=expected") in stderr
    assert "slot custody=pending/unknown caller_HOLD=True" in stderr
    assert "cleaned" not in stderr and "slot reaped" not in stderr
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("failure", [False, True])
def test_diagnostic_blocking_sink_skipped_without_changing_control_or_cleanup(diagnostic_fd, failure):
    fake = Fake()
    fake.hang = failure
    state, events = diagnostic_fd
    state["flags"] = 0
    result = run(fake)
    assert not any(e[0] in ("write", "text_write", "flush") for e in events)
    assert all(e == ("fcntl", 2, g.fcntl.F_GETFL) for e in events if e[0] == "fcntl")
    assert result.code == (78 if failure else 0) and result.held == failure
    assert any(e[0] == "release" for e in fake.events) == (not failure)


def test_diagnostic_output_excludes_payload_environment_and_unvalidated_custody(diagnostic_messages):
    argv = ["MOCK_PRIVATE_EXEC_72533", "MOCK_PRIVATE_ARG_72533"]
    cwd = Path("/MOCK_PRIVATE_CWD_72533")
    env = {"MOCK_PRIVATE_ENV_KEY_72533": "MOCK_PRIVATE_ENV_VALUE_72533"}
    custody = {"nonce": "MOCK_PRIVATE_CUSTODY_NONCE_72533", "boot": [9876543210, 987653]}
    fake = Fake()
    original = fake.read
    supplied_custody = []
    def read():
        response = original() | {"custody": custody.copy()}
        supplied_custody.append(response["custody"])
        return response
    fake.read = read
    result = g.run(argv, cwd, env, SLOT, g.Limits(campaign_deadline=1000), fake, 1)
    assert result.code == 0 and not result.held
    assert [event for event in fake.events if event[0] == "launch"] == [("launch", argv, cwd, env)]
    assert supplied_custody and all(value == custody for value in supplied_custody)
    stderr = "\n".join(diagnostic_messages)
    for secret in (*argv, cwd.name, *env.keys(), *env.values(), custody["nonce"],
                   *(str(value) for value in custody["boot"]), repr(custody["boot"])):
        assert secret not in stderr
    assert "slot custody=released caller_HOLD=False" in stderr
    assert "boot=unknown" in stderr


def test_diagnostic_worker_boot_comes_from_validated_authority_only(monkeypatch, diagnostic_messages):
    from types import SimpleNamespace
    monkeypatch.setattr(g.sys, "platform", "darwin")
    monkeypatch.setitem(g.sys.modules, "psutil", SimpleNamespace())
    monkeypatch.setattr(g, "boot_identity", lambda: BOOT)
    observed_boots = []
    def validated_authority(args, env, boot, now):
        observed_boots.append(boot)
        return SimpleNamespace(boot=BOOT, campaign={"deadline": 1000}, slot=SLOT, paths=[])
    monkeypatch.setattr(g, "Authority", validated_authority)
    monkeypatch.setattr(g, "Disk", lambda paths: None)
    g.DarwinProbes(None)
    assert observed_boots == [BOOT]
    assert "worker authority boot_seconds=1700000000 boot_microseconds=42 campaign_deadline=1000" in diagnostic_messages


@pytest.fixture
def diagnostic_fd(monkeypatch):
    """Fully synthetic fd metadata/write boundary; no pipes, probes or TextIO."""
    from types import SimpleNamespace
    state = {"mode": g.statmod.S_IFIFO, "flags": g.os.O_NONBLOCK,
             "metadata_error": None, "write_result": "full"}
    events = []
    def fstat(fd):
        events.append(("fstat", fd))
        if state["metadata_error"] == "fstat":
            raise OSError("metadata error")
        return SimpleNamespace(st_mode=state["mode"])
    def fcntl(fd, command, *args):
        events.append(("fcntl", fd, command, *args))
        if state["metadata_error"] == "fcntl":
            raise OSError("flags error")
        return state["flags"]
    def write(fd, record):
        events.append(("write", fd, record))
        result = state["write_result"]
        if result == "again":
            raise BlockingIOError(11, "would block")
        if result == "error":
            raise OSError("write error")
        return len(record) if result == "full" else len(record) - 1 if result == "short" else 0
    class TextSink:
        def write(self, text):
            events.append(("text_write", text))
            return len(text)
        def flush(self):
            events.append(("flush",))
    monkeypatch.setattr(g.os, "fstat", fstat)
    monkeypatch.setattr(g.fcntl, "fcntl", fcntl)
    monkeypatch.setattr(g.os, "write", write)
    monkeypatch.setattr(g.sys, "stderr", TextSink())
    return state, events


@pytest.mark.parametrize("mode,nonblocking", [
    (g.statmod.S_IFIFO, False), (g.statmod.S_IFSOCK, False),
    (g.statmod.S_IFREG, False), (g.statmod.S_IFCHR, False),
    (g.statmod.S_IFREG, True), (g.statmod.S_IFCHR, True),
])
def test_diagnostic_emitter_skips_blocking_or_unsupported_sink(diagnostic_fd, mode, nonblocking):
    state, events = diagnostic_fd
    state.update(mode=mode, flags=g.os.O_NONBLOCK if nonblocking else 0)
    result = g.diagnostic("not delivered")
    assert result is False
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL)]


@pytest.mark.parametrize("mode", [g.statmod.S_IFIFO, g.statmod.S_IFSOCK])
@pytest.mark.parametrize("message,body", [("hello", b"hello"), ("\u00e9\nline\rend", b"? line end"),
                                         ("x" * 1000, b"x" * 496)])
def test_diagnostic_emitter_one_bounded_ascii_record(diagnostic_fd, mode, message, body):
    state, events = diagnostic_fd
    state["mode"] = mode
    result = g.diagnostic(message)
    record = b"command guard: " + body + b"\n"
    assert result is True
    assert len(record) <= 512 and record.isascii() and record.count(b"\n") == 1
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL), ("write", 2, record)]


@pytest.mark.parametrize("write_result", ["again", "error", "short", "zero"])
def test_diagnostic_emitter_write_failure_never_retries(diagnostic_fd, write_result):
    state, events = diagnostic_fd
    state["write_result"] = write_result
    result = g.diagnostic("one attempt")
    assert result is False
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL),
                      ("write", 2, b"command guard: one attempt\n")]


@pytest.mark.parametrize("stage", ["fstat", "fcntl"])
def test_diagnostic_emitter_metadata_failure_never_writes(diagnostic_fd, stage):
    state, events = diagnostic_fd
    state["metadata_error"] = stage
    result = g.diagnostic("metadata unavailable")
    assert result is False
    expected = [("fstat", 2)]
    if stage == "fcntl":
        expected.append(("fcntl", 2, g.fcntl.F_GETFL))
    assert events == expected
