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
        self.profile = "default"
        self.authority_sequence = self.observer_sequence = 0
        self.observer_pending = self.observer_final = False
        self.observer_sent = 0

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds

    def pid(self, name):
        return {"worker": 2, "observer": 6, "payload": 10}[name]

    def observer(self, options):
        pass  # main-boundary mocks only; real lifecycle tests use registered I/O

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
        assert self.reaped == ({"worker", "observer", "payload"} if self.launched else {"worker", "observer"})
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
        self.authority_sequence += 1
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
        rows = self.rows()
        complete = self.acquiring or self.acquired
        return dict(slot=str(Path("/alternate/slot") if self.alternate else SLOT),
                    sealed=self.final, host=self.host_value.copy(),
                    disk_growth=self.growth if complete else None, census_complete=complete,
                    processes=rows, reaped_helpers=[[3, 3.0]],
                    role="authority", sequence=self.authority_sequence,
                    custody={"profile": self.profile, "campaign": {"profile": self.profile},
                             "budget": {} if complete else None})

    def rows(self, observer=False):
        rows = [[1, 1.0, 0, 7, 1000, False], [2, 2.0, 1, 2, 2000, False],
                [6, 6.0, 1, 6, 2000, False],
                [5, 5.0, 6, 6, 3000, False] if observer else
                [3, 3.0, 2, 2, 3000, False]]  # sampled, already-reaped helper
        if self.observer_survivor and self.launched:
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
        return rows

    def observe_request(self, final=False):
        assert not self.observer_pending
        self.observer_pending, self.observer_final = True, final
        self.observer_sequence += 1
        self.observer_sent = self.time

    def observe_read(self):
        assert self.observer_pending
        self.observer_pending = False
        return dict(role="observer", sequence=self.observer_sequence,
                    observed_at=self.observer_sent, sealed=self.observer_final,
                    host=self.host_value.copy(), processes=self.rows(True),
                    reaped_helpers=[[5, 5.0]])


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
    assert fake.killed == {"worker", "observer", "payload"}
    assert fake.time <= 104


def test_responsive_probe_cannot_reset_command_deadline():
    fake = Fake()
    fake.running = True
    result = run(fake, seconds=0.5)
    assert result.code == 78 and result.held
    term = next(e for e in fake.events if e[:3] == ("signal", "payload", signal.SIGTERM))
    assert term[3] == 100.5
    assert sum(e[0] == "launch" for e in fake.events) == 1
    assert all(e[2] < 100.5 for e in fake.events if e[0] == "request")
    assert not any(e[0] == "release" for e in fake.events)


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
    for name in ("worker", "observer", "payload"):
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
    assert fake.killed == {"worker", "observer", "payload"}
    assert all(e[1] in ("worker", "observer", "payload") for e in fake.events if e[0] == "signal")


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
        assert fake.acquired and fake.killed == {"worker", "observer", "payload"}


def test_observer_descendant_cleanup_unknown_retains_acquired_slot():
    fake = Fake()
    fake.observer_survivor = True
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    assert fake.killed == {"worker", "observer", "payload"}


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
    assert fake.killed == {"worker", "observer"} and fake.time >= 101


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
    assert fake.killed == {"payload", "worker", "observer"}


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
    assert result.code == 78 and result.held and fake.killed == {"worker", "observer"}
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
    assert not fake.launched and fake.killed == {"worker", "observer"}
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
                    campaign_init=True, campaign_retire=False, disk_growth_bytes=20 * g.GIB,
                    profile="default")
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


BOOT = "abcdef01-2345-6789-abcd-ef0123456789"


@pytest.fixture
def boot_sysctl(monkeypatch):
    """Synthetic ctypes library only; never load libSystem or read real sysctls."""
    import ctypes
    from types import SimpleNamespace
    state = dict(raw=BOOT.encode("ascii") + b"\0", size=37, result=0,
                 error=None, wallclock=[1700000000, 42])
    calls = []
    class Sysctl:
        def __call__(self, name, value, size, new, new_size):
            calls.append((name, size._obj.value, new, new_size))
            assert name == b"kern.bootsessionuuid"
            assert len(value._obj.raw) == 37
            assert new is None and new_size == 0
            if state["error"] is not None:
                raise state["error"]
            value._obj.raw = state["raw"]
            size._obj.value = state["size"]
            return state["result"]
    def library(path, *, use_errno):
        assert path == "/usr/lib/libSystem.B.dylib" and use_errno is True
        return SimpleNamespace(sysctlbyname=Sysctl())
    monkeypatch.setattr(ctypes, "CDLL", library)
    return state, calls


@pytest.mark.parametrize("uppercase", [False, True])
def test_boot_identity_canonicalizes_exact_native_uuid(boot_sysctl, uppercase):
    state, calls = boot_sysctl
    state["raw"] = (BOOT.upper() if uppercase else BOOT).encode("ascii") + b"\0"
    assert g.boot_identity() == BOOT
    assert calls == [(b"kern.bootsessionuuid", 37, None, 0)]


@pytest.mark.parametrize("changes", [
    {"result": -1}, {"size": 0}, {"size": 36}, {"size": 38}, {"size": 4096},
    {"raw": BOOT.encode("ascii") + b"x"},
    {"raw": b"\0" + BOOT.encode("ascii")[1:] + b"\0"},
    {"raw": b"\xff" + BOOT.encode("ascii")[1:] + b"\0"},
    {"raw": b"g" + BOOT.encode("ascii")[1:] + b"\0"},
    {"raw": BOOT.replace("-", "_").encode("ascii") + b"\0"},
    {"raw": b"{" + BOOT.encode("ascii")[:35] + b"\0"},
    {"raw": b"00000000-0000-0000-0000-000000000000\0"},
    {"raw": b"\0" * 37},
])
def test_boot_identity_rejects_invalid_native_response_without_fallback(boot_sysctl, changes):
    state, calls = boot_sysctl
    state.update(changes)
    with pytest.raises(g.Blocked, match="OS boot identity"):
        g.boot_identity()
    assert calls == [(b"kern.bootsessionuuid", 37, None, 0)]


@pytest.mark.parametrize("error", [AttributeError("missing sysctl"), OSError("read failed")])
def test_boot_identity_sysctl_exception_fails_without_fallback(boot_sysctl, error):
    state, calls = boot_sysctl
    state["error"] = error
    with pytest.raises(type(error)):
        g.boot_identity()
    assert calls == [(b"kern.bootsessionuuid", 37, None, 0)]


def test_boot_identity_missing_library_symbol_fails_closed(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(AttributeError):
        g.boot_identity()


def test_boot_identity_ignores_changes_to_both_wallclock_fields(boot_sysctl):
    state, calls = boot_sysctl
    assert g.boot_identity() == BOOT
    state["wallclock"] = [1700000100, 632905]
    assert g.boot_identity() == BOOT
    assert calls == [(b"kern.bootsessionuuid", 37, None, 0)] * 2


@pytest.fixture
def boot_records():
    roots = dict(declarations=[["/root", "/root", [1, 2, 3]]],
                 pins=[["/root", [1, 2, 3, g.statmod.S_IFDIR, 1, None, None, 0o700]]])
    campaign = dict(boot=BOOT, origin=100, deadline=1000, disk_cap_bytes=20 * g.GIB,
                    budget_identity=[1, 4, 5], roots=roots, profile="default")
    budget = dict(version=1, campaign=g.campaign_binding(campaign), roots=roots,
                  sequence=1, charged_bytes=g.WRITE_CHARGE + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE,
                  previous=[])
    return campaign, budget, roots


def test_campaign_and_budget_accept_same_canonical_boot_uuid(boot_records):
    campaign, budget, roots = boot_records
    g.validate_campaign(campaign, BOOT, 100)
    assert g.validate_budget(budget, campaign, roots) == {}


@pytest.mark.parametrize("boot", [
    BOOT.upper(), BOOT.replace("-", ""),
    "{" + BOOT + "}", " " + BOOT, BOOT + "\n", BOOT[:-1] + "g",
    "00000000-0000-0000-0000-000000000000", [1700000000, 42],
    [1700000000, True], None, 1, True, {},
])
@pytest.mark.parametrize("boundary", ["campaign-record", "campaign-expected", "budget-record", "budget-both"])
def test_campaign_and_budget_reject_legacy_or_noncanonical_boot(boot_records, boot, boundary):
    campaign, budget, roots = boot_records
    with pytest.raises(g.Blocked):
        if boundary == "campaign-record":
            g.validate_campaign(campaign | {"boot": boot}, BOOT, 100)
        elif boundary == "campaign-expected":
            g.validate_campaign(campaign, boot, 100)
        else:
            budget["campaign"]["boot"] = boot
            expected = campaign | {"boot": boot} if boundary == "budget-both" else campaign
            g.validate_budget(budget, expected, roots)


@pytest.mark.parametrize("boundary", ["campaign-record", "campaign-expected", "budget-record"])
def test_campaign_and_budget_reject_changed_canonical_boot_uuid(boot_records, boundary):
    campaign, budget, roots = boot_records
    changed = "bbcdef01-2345-6789-abcd-ef0123456789"
    with pytest.raises(g.Blocked):
        if boundary == "campaign-record":
            g.validate_campaign(campaign | {"boot": changed}, BOOT, 100)
        elif boundary == "campaign-expected":
            g.validate_campaign(campaign, changed, 100)
        else:
            budget["campaign"]["boot"] = changed
            g.validate_budget(budget, campaign, roots)


def test_boot_identity_predicate_has_no_version_or_variant_restriction(boot_records):
    campaign, budget, roots = boot_records
    boot = "abcdef01-2345-ffff-0000-ef0123456789"
    campaign["boot"] = budget["campaign"]["boot"] = boot
    g.validate_campaign(campaign, boot, 100)
    assert g.validate_budget(budget, campaign, roots) == {}


def authority(args, now=100):
    return g.Authority(args, environment(args), BOOT, now)


def complete_metadata(a, now=105):
    """Only metadata primitive tested here, NOT a substitute for lifecycle loop."""
    try:
        a.acquire(now - 1)
        a.debit({}, {})  # first paid commit, including slot and future campaign
        a.debit({}, {})  # sealed final paid commit
        g.finish_authority(a.custody(), a.slot, now, profile=a.profile)
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


@pytest.mark.parametrize("changed", [dict(boot="bbcdef01-2345-6789-abcd-ef0123456789"), dict(origin=101),
                                   dict(deadline=7300.1), dict(boot=[1700000000, True])])
def test_campaign_boot_origin_and_two_hour_ceiling_fail_closed(changed):
    roots = dict(declarations=[["/root", "/root", [1, 2, 3]]],
                 pins=[["/root", [1, 2, 3, g.statmod.S_IFDIR, 1, None, None, 0o700]]])
    record = dict(boot=BOOT, origin=100, deadline=1000, disk_cap_bytes=20 * g.GIB,
                  budget_identity=[1, 4, 5], roots=roots, profile="default") | changed
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
    assert result.code == 78 and result.held and fake.killed == {"worker", "observer", "payload"}
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
    owned = {"worker", "observer", "payload"} if phase == "F" else {"worker", "observer"}
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
        "control": "BLOCKED slot retained if acquired: control failure: UnformattableError: <no safe string>",
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
            fake.reaped = {"worker", "observer", "payload"}
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
        profile="default",
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
    monkeypatch.setattr(g.time, "monotonic", fake.now)
    monkeypatch.setattr(g.signal, "signal", lambda *args: None)

    assert g.main() == (0 if site == "success" else 78)
    assert messages == [expected]
    assert events[-1] == "diagnostic"
    if site in ("success", "hold-A", "hold-F"):
        assert events[-2] == ("outcome", outcome)
        assert outcome == g.Outcome(78 if held else 0, held, reason)
        assert (fake.acquired, fake.slot_pending) == (held, held)
        assert any(event[0] == "release" for event in fake.events) == (not held)
        assert fake.reaped == (set() if held else {"worker", "observer", "payload"})
    elif site == "control":
        assert (fake.acquired, fake.slot_pending) == (True, True)
        assert not fake.events and "run" not in events
    else:
        assert not fake.acquired and not fake.slot_pending and not fake.events
        assert "adapter" not in events and "run" not in events


@pytest.fixture
def parent_control_failure(monkeypatch):
    """Synthetic main boundaries only; assert forbidden effects after patches exit."""
    def fail(exc, site="validate", delivered=True, registered=False, wire=False):
        events, messages, forbidden = [], [], []
        custody = {"held": True, "pending": True}
        args = type("Request", (), dict(
            _telemetry=False, campaign_init=False, campaign_retire=False,
            cooperative_build=True, argv=["--", "mock-command"],
            cwd=Path("/work"), slot=SLOT,
        ))()
        limits = g.Limits(campaign_deadline=1000)

        def deny(name):
            def forbidden_call(*args, **kwargs):
                forbidden.append(name)
                raise AssertionError("forbidden effect: " + name)
            return forbidden_call

        class Adapter:
            profile = "default"
            operation = "run"
            def __init__(self):
                self.children = {}
                self.custody = custody
            def now(self):
                events.append("now")
                return 100
            def worker(self, options):
                events.append("worker")
                if registered:
                    self.children["worker"] = child
                if site == "worker":
                    raise exc
            def observer(self, options):
                events.append("observer")

        child = object()
        adapter = Adapter()
        for name in ("release", "close", "reap", "signal", "acquire", "launch"):
            setattr(adapter, name, deny(name))

        class MockReal:
            @staticmethod
            def context():
                events.append("context")
            def __new__(cls):
                events.append("adapter")
                if site == "adapter":
                    raise exc
                return adapter

        def validate(options, env, now):
            events.append("validate")
            if site == "validate":
                raise exc
            return limits

        outcome = g.Outcome(78, True, "mock owned worker retained by control loop")
        def supervise(argv, cwd, env, slot, actual_limits, actual_adapter,
                      monitor_pid, operation, started):
            events.append("run")
            # Record mismatch outside production catches; no swallowed assertions.
            if not (argv == ["mock-command"] and cwd == Path("/work")
                    and env == {"MOCK": "only"} and slot == SLOT and actual_limits is limits
                    and actual_adapter is adapter and monitor_pid == 1
                    and operation == "run" and started == 100):
                forbidden.append("run arguments")
            if registered:
                return outcome
            raise exc

        emit = g.diagnostic
        def diagnostic(message):
            messages.append(message)
            events.append("diagnostic")
            return emit(message) if wire else delivered

        with monkeypatch.context() as patch:
            patch.setattr(g, "Real", MockReal)
            patch.setattr(g, "validate_request", validate)
            patch.setattr(g, "parser", lambda: type(
                "Parser", (), {"parse_args": staticmethod(lambda: args)})())
            patch.setattr(g, "run", supervise)
            patch.setattr(g, "diagnostic", diagnostic)
            patch.setattr(g.sys, "argv", ["command_guard.py", "--", "mock-command"])
            patch.setattr(g.os, "environ", {"MOCK": "only"})
            patch.setattr(g.os, "getpid", lambda: 1)
            patch.setattr(g.time, "monotonic", lambda: 100)
            patch.setattr(g.signal, "signal", lambda *args: events.append("handler"))
            for name in ("DarwinProbes", "Authority", "finish_authority",
                         "boot_identity", "child_bootstrap", "telemetry_worker"):
                patch.setattr(g, name, deny(name))
            for name in ("open", "fork", "kill", "waitpid", "stat", "lstat", "getcwd"):
                patch.setattr(g.os, name, deny(name))
            code = g.main()
        assert forbidden == []
        assert code == 78
        assert custody == {"held": True, "pending": True}
        assert adapter.custody is custody
        assert adapter.children == ({"worker": child} if registered else {})
        expected_events = ["context", "validate"]
        if site != "validate":
            expected_events.append("adapter")
        if site not in ("validate", "adapter"):
            expected_events.extend(["handler", "handler", "now", "worker"])
        if site == "run":
            expected_events.append("observer")
        if site == "run" or registered:
            expected_events.append("run")
        assert events == expected_events + ["diagnostic"]
        return messages
    return fail


@pytest.mark.parametrize("delivered", [True, False], ids=["delivered", "lost"])
@pytest.mark.parametrize("site", ["validate", "adapter", "worker", "run"])
def test_parent_control_failure_reports_each_boundary_without_effects(parent_control_failure, site, delivered):
    messages = parent_control_failure(g.Blocked("mock control rejection"), site, delivered)
    assert messages == [
        "BLOCKED slot retained if acquired: control failure: Blocked: mock control rejection",
    ]


@pytest.mark.parametrize("kind", ["formatting", "filename", "metaclass", "class",
                                  "object", "str-subclass", "large-int", "empty",
                                  "filename-object", "filename-str-subclass", "sanitized", "bounds"])
def test_parent_control_failure_safe_exact_detail(parent_control_failure, kind):
    hooks = []
    class HostileObject:
        def __str__(self):
            hooks.append("str")
            raise AssertionError("must not format")
        def __repr__(self):
            hooks.append("repr")
            raise AssertionError("must not represent")
    class HostileString(str):
        def __getitem__(self, key):
            hooks.append("slice")
            raise AssertionError("must not slice subclass")
        __str__ = HostileObject.__str__
        __repr__ = HostileObject.__repr__
    class HostileError(Exception):
        __str__ = HostileObject.__str__
        __repr__ = HostileObject.__repr__
        @property
        def args(self):
            hooks.append("args")
            raise AssertionError("must use base descriptor")
    class HostileFileError(OSError):
        @property
        def filename(self):
            hooks.append("filename")
            raise AssertionError("must use base descriptor")
    class HostileMeta(type):
        def __getattribute__(cls, name):
            hooks.append("metaclass")
            raise AssertionError("must use type name descriptor")
    class MetaError(Exception, metaclass=HostileMeta):
        pass
    class ClassError(Exception):
        @property
        def __class__(self):
            hooks.append("class")
            raise AssertionError("must check actual type")

    if kind == "formatting":
        exc, detail = HostileError("raw safe detail"), "HostileError: raw safe detail"
    elif kind == "filename":
        exc = HostileFileError(2, "missing campaign", "/fake/campaign.json")
        detail = "HostileFileError: missing campaign | /fake/campaign.json"
    elif kind == "metaclass":
        exc, detail = MetaError("safe"), "MetaError: safe"
    elif kind == "class":
        exc, detail = ClassError("safe"), "ClassError: safe"
    elif kind in ("filename-object", "filename-str-subclass"):
        exc = OSError(2, "missing campaign")
        OSError.filename.__set__(exc, HostileObject() if kind == "filename-object"
                                 else HostileString("unsafe"))
        detail = "FileNotFoundError: missing campaign"
    elif kind == "sanitized":
        exc = RuntimeError("before\r\n\t\0\x1b\x7f\u00e9after")
        detail = "RuntimeError: before???????after"
    elif kind == "bounds":
        exc = type("N" * 100, (OSError,), {})()
        BaseException.args.__set__(exc, ("a" * 140, "b" * 140, "THIRD_MUST_NOT_APPEAR"))
        OSError.filename.__set__(exc, "f" * 140)
        detail = "N" * 80 + ": " + "a" * 120 + " | " + "b" * 120 + " | " + "f" * 120
    else:
        values = {"object": (HostileObject(), HostileObject()),
                  "str-subclass": (HostileString("unsafe"), HostileString("unsafe")),
                  "large-int": (1 << 20000, 1 << 20000), "empty": ()}[kind]
        exc = Exception(*values, "THIRD_MUST_NOT_APPEAR") if values else Exception()
        detail = "Exception: <no safe string>"
    messages = parent_control_failure(exc)
    assert hooks == []
    assert messages == ["BLOCKED slot retained if acquired: control failure: " + detail]


def test_parent_control_failure_registered_worker_uses_existing_loop(parent_control_failure):
    messages = parent_control_failure(RuntimeError("partial wire setup"), "worker", registered=True)
    assert messages == ["mock owned worker retained by control loop"]


def test_parent_control_failure_bounds_existing_wire_record(parent_control_failure, diagnostic_fd):
    _, events = diagnostic_fd
    exc = type("N" * 100, (OSError,), {})()
    BaseException.args.__set__(exc, ("a" * 140, "b" * 140, "THIRD_MUST_NOT_APPEAR"))
    OSError.filename.__set__(exc, "f" * 140)
    messages = parent_control_failure(exc, wire=True)
    body = ("BLOCKED slot retained if acquired: control failure: " + "N" * 80 + ": "
            + "a" * 120 + " | " + "b" * 120 + " | " + "f" * 120)
    assert messages == [body]
    record = b"command guard: " + body[:496].encode("ascii") + b"\n"
    assert len(record) == 512 and record.isascii() and record.count(b"\n") == 1
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL), ("write", 2, record)]


@pytest.mark.parametrize("kind", ["metaclass", "class"])
def test_exception_detail_bypasses_type_hooks_in_worker(worker_failure, diagnostic_messages, kind):
    hooks = []
    class HostileMeta(type):
        def __getattribute__(cls, name):
            hooks.append("metaclass")
            raise AssertionError("must use type name descriptor")
    class MetaError(Exception, metaclass=HostileMeta):
        pass
    class ClassError(Exception):
        @property
        def __class__(self):
            hooks.append("class")
            raise AssertionError("must check actual type")
    exc = MetaError("safe") if kind == "metaclass" else ClassError("safe")
    worker_failure(exc)
    assert hooks == []
    expected = {"metaclass": "telemetry worker failed: MetaError: safe",
                "class": "telemetry worker failed: ClassError: safe"}
    assert diagnostic_messages == [expected[kind]]


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
    assert messages == ["telemetry worker failed: UnformattableError: <no safe string>"]


@pytest.fixture
def worker_failure(monkeypatch):
    """Raise from fake probe construction, before any stream or native access."""
    def fail(exc):
        calls = []
        args = object()
        def probes(options):
            calls.append(options)
            raise exc
        monkeypatch.setattr(g, "DarwinProbes", probes)
        monkeypatch.setattr(g.sys, "stdin", object())
        monkeypatch.setattr(g.sys, "stdout", object())
        assert g.telemetry_worker(args) == 78
        assert calls == [args]
    return fail


def test_worker_failure_preserves_ordinary_type_and_message(worker_failure, diagnostic_messages):
    worker_failure(RuntimeError("probe failed"))
    assert diagnostic_messages == ["telemetry worker failed: RuntimeError: probe failed"]


def test_worker_failure_includes_file_not_found_strerror_and_filename(worker_failure, diagnostic_messages):
    worker_failure(FileNotFoundError(2, "missing campaign", "/fake/campaign.json"))
    assert diagnostic_messages == [
        "telemetry worker failed: FileNotFoundError: missing campaign | /fake/campaign.json",
    ]


def test_worker_failure_bypasses_exception_formatting_and_args_override(worker_failure, diagnostic_messages):
    calls = []
    class UnformattableError(Exception):
        @property
        def args(self):
            calls.append("args")
            raise AssertionError("must read BaseException descriptor directly")

        def __str__(self):
            calls.append("str")
            raise AssertionError("must not format exception")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("must not represent exception")

    worker_failure(UnformattableError("raw safe detail"))
    assert calls == []
    assert diagnostic_messages == ["telemetry worker failed: UnformattableError: raw safe detail"]


def test_worker_failure_bypasses_oserror_filename_override(worker_failure, diagnostic_messages):
    calls = []
    class HostileFileError(FileNotFoundError):
        @property
        def filename(self):
            calls.append("filename")
            raise AssertionError("must read OSError descriptor directly")

    worker_failure(HostileFileError(2, "missing campaign", "/fake/campaign.json"))
    assert calls == []
    assert diagnostic_messages == [
        "telemetry worker failed: HostileFileError: missing campaign | /fake/campaign.json",
    ]


@pytest.mark.parametrize("kind", ["object", "str-subclass", "large-int", "empty"])
def test_worker_failure_ignores_nonexact_string_arguments(worker_failure, diagnostic_messages, kind):
    calls = []
    class HostileObject:
        def __str__(self):
            calls.append("str")
            raise AssertionError("must not format argument")

        def __repr__(self):
            calls.append("repr")
            raise AssertionError("must not represent argument")

    class HostileString(str):
        def __getitem__(self, key):
            calls.append("slice")
            raise AssertionError("must not slice string subclass")

        __str__ = HostileObject.__str__
        __repr__ = HostileObject.__repr__

    args = {"object": (HostileObject(), HostileObject()),
            "str-subclass": (HostileString("unsafe"), HostileString("unsafe")),
            "large-int": (1 << 20000, 1 << 20000), "empty": ()}[kind]
    exc = Exception(*args, "third argument must not be inspected") if args else Exception()
    worker_failure(exc)
    assert calls == []
    assert diagnostic_messages == ["telemetry worker failed: Exception: <no safe string>"]


def test_worker_failure_bounds_name_fields_and_existing_wire_record(monkeypatch, worker_failure, diagnostic_fd):
    _, events = diagnostic_fd
    error_type = type("N" * 100, (OSError,), {})
    exc = error_type()
    BaseException.args.__set__(exc, ("a" * 140, "b" * 140, "THIRD_MUST_NOT_APPEAR"))
    OSError.filename.__set__(exc, "f" * 140)
    messages = []
    emit = g.diagnostic
    def diagnostic(message):
        messages.append(message)
        return emit(message)
    monkeypatch.setattr(g, "diagnostic", diagnostic)
    worker_failure(exc)
    body = ("telemetry worker failed: " + "N" * 80 + ": "
            + "a" * 120 + " | " + "b" * 120 + " | " + "f" * 120)
    assert messages == [body]
    assert len(body) == 473 and len(body) <= 496
    record = b"command guard: " + body.encode("ascii") + b"\n"
    assert len(record) <= 512 and record.isascii() and record.count(b"\n") == 1
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL), ("write", 2, record)]


def test_worker_failure_sanitizes_controls_and_nonascii(monkeypatch, worker_failure, diagnostic_fd):
    _, events = diagnostic_fd
    messages = []
    emit = g.diagnostic
    def diagnostic(message):
        messages.append(message)
        return emit(message)
    monkeypatch.setattr(g, "diagnostic", diagnostic)
    worker_failure(RuntimeError("before\r\n\t\0\x1b\x7f\u00e9after"))
    body = "telemetry worker failed: RuntimeError: before???????after"
    assert messages == [body]
    assert all(" " <= char <= "~" for char in body)
    record = b"command guard: " + body.encode("ascii") + b"\n"
    assert len(record) <= 512 and record.isascii() and record.count(b"\n") == 1
    assert events == [("fstat", 2), ("fcntl", 2, g.fcntl.F_GETFL), ("write", 2, record)]


def test_diagnostic_acquisition_eof_worker_78_keeps_pending_custody(monkeypatch, diagnostic_messages):
    fake = Fake()
    fake.worker_exit_code = 78
    original = fake.read
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.read_fd, endpoint.buffer = 123, bytearray()
    reads = []
    def eof(fd, size):
        reads.append((fd, size))
        return b""
    def read():
        if fake.acquiring:
            return endpoint.read()  # Production EOF boundary; fake fd only.
        return original()
    monkeypatch.setattr(g.os, "read", eof)
    fake.read = read
    result = run(fake)
    stderr = "\n".join(diagnostic_messages)
    assert reads == [(123, 65536)]
    assert result.code == 78 and result.held
    assert fake.slot_pending and not fake.acquired and not fake.launched
    assert "telemetry S status completed" in stderr
    assert "telemetry A acquisition start" in stderr
    assert "telemetry A acquisition completed" not in stderr
    assert "telemetry/control failure [A acquisition]: telemetry worker crash/EOF" in result.reason
    assert "slot retained/ownership unresolved" in result.reason
    assert ("reap worker terminal_known=True returncode=78 reap_error_observed=False "
            "wait_timeout=False expected_exit=unexpected custody=terminal_known") in diagnostic_messages
    assert "slot custody=pending/unknown caller_HOLD=True" in diagnostic_messages
    assert fake.reaped == {"worker", "observer"}
    assert [event[0] for event in fake.events if event[0] == "request"] == ["request", "request"]
    assert not any(event[0] in ("release", "launch") for event in fake.events)


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
    custody = {"nonce": "MOCK_PRIVATE_CUSTODY_NONCE_72533",
               "boot": "98765432-10ab-cdef-9876-543210abcdef",
                "profile": "default", "campaign": {"profile": "default"}, "budget": {}}
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
                   custody["boot"], *custody["boot"].split("-"), repr(custody["boot"])):
        assert secret not in stderr
    assert "slot custody=released caller_HOLD=False" in stderr
    assert "boot=unknown" in stderr


def test_diagnostic_worker_boot_comes_from_validated_authority_only(monkeypatch, diagnostic_messages):
    from types import SimpleNamespace
    monkeypatch.setattr(g.sys, "platform", "darwin")
    monkeypatch.setitem(g.sys.modules, "psutil", SimpleNamespace())
    reader_boot = "bbcdef01-2345-6789-abcd-ef0123456789"
    monkeypatch.setattr(g, "boot_identity", lambda: reader_boot)
    observed_boots = []
    def validated_authority(args, env, boot, now):
        observed_boots.append(boot)
        return SimpleNamespace(boot=BOOT, profile="default",
                               campaign={"deadline": 1000, "profile": "default"}, slot=SLOT, paths=[])
    monkeypatch.setattr(g, "Authority", validated_authority)
    monkeypatch.setattr(g, "Disk", lambda paths: None)
    g.DarwinProbes(None)
    assert observed_boots == [reader_boot]
    assert diagnostic_messages == [f"worker authority boot_uuid={BOOT} campaign_deadline=1000"]
    assert reader_boot not in "\n".join(diagnostic_messages)


def test_diagnostic_worker_rejects_noncanonical_authority_without_formatting(monkeypatch, diagnostic_messages):
    from types import SimpleNamespace
    calls = []
    class HostileString(str):
        def __str__(self):
            calls.append("str")
            raise AssertionError("must not format unvalidated boot")

        def __format__(self, spec):
            calls.append("format")
            raise AssertionError("must not format unvalidated boot")
    monkeypatch.setattr(g.sys, "platform", "darwin")
    monkeypatch.setitem(g.sys.modules, "psutil", SimpleNamespace())
    monkeypatch.setattr(g, "boot_identity", lambda: BOOT)
    monkeypatch.setattr(g, "Authority", lambda *args: SimpleNamespace(boot=HostileString(BOOT)))
    with pytest.raises(g.Blocked, match="authority boot identity"):
        g.DarwinProbes(None)
    assert calls == [] and diagnostic_messages == []


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


# Native-profile definitions only; independent frozen literals, never production
# allowlist-derived. Six Cargo counts (1,1,1,7,4,1) and wheel build are UNRUN.
NATIVE_TRIAL = "native-build-tests-12-8-4"
NATIVE_TEST_ROOT = "/Users/blairhudson/workspace/pllm"
NATIVE_TEST_OUTPUT = NATIVE_TEST_ROOT + "/target/native-build-tests-12-8-4"
NATIVE_TEST_COMMANDS = (
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--test", "maverick_batch_reference",
     "bounded_draw_accepts_exactly_draw_32_but_refuses_draw_33", "--", "--exact", "--test-threads=1"),
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--test", "maverick_batch_reference",
     "bounded_draw_preserves_ceiling_boundary_and_handles_zero_one_and_exhaustion",
     "--", "--exact", "--test-threads=1"),
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--test", "puma_norm_allocation",
     "norm_admission_allocates_nothing_and_begin_failure_sweep_erases", "--", "--exact", "--test-threads=1"),
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--lib",
     "coefficient_rns_contraction::olt_trace_tests::olt_tests::", "--", "--test-threads=1"),
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--lib",
     "maverick_batch::arithmetic::tests::", "--", "--test-threads=1"),
    ("/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin/cargo", "test",
     "-p", "pllm-core", "--locked", "--offline", "-j1", "--lib",
     "puma::tensor::resident_builder::tests::pending_affine_refuses_both_finishes_before_entry_without_changing_owners",
     "--", "--exact", "--test-threads=1"),
    ("/opt/homebrew/bin/uv", "run", "--offline", "--no-sync", "--no-cache", "--no-env-file",
     "--no-python-downloads", "--python", "/Users/blairhudson/workspace/pllm/.venv/bin/python",
     "maturin", "build", "--release", "--locked", "--offline", "-j", "1", "--interpreter",
     "/Users/blairhudson/workspace/pllm/.venv/bin/python", "--out", "/Users/blairhudson/workspace/pllm/dist"),
)
NATIVE_TEST_ENV = {
    "HOME": "/Users/blairhudson",
    "PATH": "/Users/blairhudson/.rustup/toolchains/1.85.1-aarch64-apple-darwin/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
    "CARGO_HOME": "/Users/blairhudson/.cargo", "RUSTUP_HOME": "/Users/blairhudson/.rustup",
    "RUSTUP_TOOLCHAIN": "1.85.1-aarch64-apple-darwin", "CARGO_NET_OFFLINE": "true",
    "CARGO_TARGET_DIR": "/Users/blairhudson/workspace/pllm/target/native-build-tests-12-8-4/cargo",
    "TMPDIR": "/Users/blairhudson/workspace/pllm/target/native-build-tests-12-8-4/tmp",
    "UV_PROJECT_ENVIRONMENT": "/Users/blairhudson/workspace/pllm/.venv",
    "__CF_USER_TEXT_ENCODING": "0x1F5:0:0",
}
NATIVE_TEST_ROOTS = [Path("/Users/blairhudson/.cargo"), Path("/Users/blairhudson/.rustup"),
                     Path("/Users/blairhudson/.local/share/uv/python"),
                     Path("/Users/blairhudson/workspace/pllm/.venv"),
                     Path("/Users/blairhudson/workspace/pllm/dist")]


def native_options():
    return ["--profile", NATIVE_TRIAL, "--rss-bytes", "4294967296", "--reserve-bytes", "8589934592",
            "--cwd", NATIVE_TEST_ROOT, "--coordinator-root", NATIVE_TEST_ROOT,
            "--slot", NATIVE_TEST_ROOT + "/target/native-command.slot",
            "--target-dir", NATIVE_TEST_OUTPUT + "/cargo", "--log-dir", NATIVE_TEST_OUTPUT + "/logs",
            "--temp-dir", NATIVE_TEST_OUTPUT + "/tmp", "--campaign-deadline", "1000",
            "--disk-root", "/Users/blairhudson/.cargo", "--disk-root", "/Users/blairhudson/.rustup",
            "--disk-root", "/Users/blairhudson/.local/share/uv/python",
            "--disk-root", "/Users/blairhudson/workspace/pllm/.venv",
            "--disk-root", "/Users/blairhudson/workspace/pllm/dist", "--cooperative-build"]


@pytest.fixture
def native_request(monkeypatch):
    args = g.parser().parse_args(native_options() + ["--", *NATIVE_TEST_COMMANDS[0]])
    monkeypatch.setattr(g.os, "getcwd", lambda: NATIVE_TEST_ROOT)
    return args, NATIVE_TEST_ENV.copy()


@pytest.mark.parametrize("command", NATIVE_TEST_COMMANDS)
def test_native_profile_exact_independent_recipes(native_request, command):
    args, env = native_request
    args.argv = ["--", *command]
    assert len(env) == 12 and env == g.NATIVE_ENV
    assert len(NATIVE_TEST_COMMANDS) == 7
    assert tuple(NATIVE_TEST_COMMANDS) == g.NATIVE_COMMANDS
    if command in NATIVE_TEST_COMMANDS[:6]:
        assert str(Path(command[0]).parent) + ":/usr/bin:/bin:/usr/sbin:/sbin" == env["PATH"]
        assert command[1] == "test"
        assert not any(token.startswith("+") for token in command)
    assert args.disk_root == NATIVE_TEST_ROOTS
    limits = g.validate_request(args, env, 100)
    assert limits == g.Limits(1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=NATIVE_TRIAL)
    assert (limits.seconds, limits.disk_growth) == (600, 20 * g.GIB)
    assert g.budget_bounds(NATIVE_TRIAL) == (33554432, 67108864)


@pytest.mark.parametrize("mutation", ["cargo", "toolchain", "package", "offline", "locked", "jobs",
    "wrong-olt", "missing-lib", "extra", "prefix", "shell", "wheel", "sync", "install", "reorder",
    "proxy", "old-proxy-tuple", "direct-plus-selector"])
def test_native_profile_argv_mutations_block_before_effects(monkeypatch, native_request, mutation):
    args, env = native_request
    command = list(NATIVE_TEST_COMMANDS[3])
    if mutation == "cargo":
        command[0] = "cargo"
    elif mutation == "toolchain":
        command[0] = "/Users/blairhudson/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo"
    elif mutation == "package":
        command[3] = "pllm-compiler"
    elif mutation in ("offline", "locked", "missing-lib"):
        command.remove({"offline": "--offline", "locked": "--locked", "missing-lib": "--lib"}[mutation])
    elif mutation == "jobs":
        command[6] = "-j2"
    elif mutation == "wrong-olt":
        command[8] = "coefficient_rns_contraction::olt_trace_tests::"
    elif mutation == "prefix":
        command = command[:7]
    elif mutation == "extra":
        command.append("--ignored")
    elif mutation == "shell":
        command = ["/bin/sh", "-c", " ".join(command)]
    elif mutation in ("wheel", "sync", "install"):
        command = ["/Users/blairhudson/.local/bin/uv", "build", "--wheel"] if mutation == "wheel" else ["uv", mutation]
    elif mutation in ("proxy", "old-proxy-tuple", "direct-plus-selector"):
        if mutation != "direct-plus-selector":
            command[0] = "/Users/blairhudson/.cargo/bin/cargo"
        if mutation != "proxy":
            command.insert(1, "+1.85.1-aarch64-apple-darwin")
    else:
        command[4], command[5] = command[5], command[4]
    args.argv = ["--", *command]
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("index", range(20))
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_native_wheel_each_literal_token_required_before_effects(monkeypatch, native_request, index, mutation):
    args, env = native_request
    command = list(NATIVE_TEST_COMMANDS[6])  # independent literal, not production allowlist
    assert len(command) == 20
    if mutation == "missing":
        command.pop(index)
    else:
        command[index] = "wrong"
    args.argv = ["--", *command]
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("mutation", ["extra", "prefix", "shell", "no-project"])
def test_native_wheel_no_extensions_shell_prefix_or_no_project(monkeypatch, native_request, mutation):
    args, env = native_request
    command = list(NATIVE_TEST_COMMANDS[6])
    if mutation == "extra":
        command.append("--no-config")
    elif mutation == "prefix":
        command = command[:9]
    elif mutation == "shell":
        command = ["/bin/sh", "-c", " ".join(command)]
    else:
        command[3] = "--no-project"
    args.argv = ["--", *command]
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("role", ["run", "init", "retire", "worker", "worker-init", "worker-retire",
                                  "observer", "observer-init", "observer-retire"])
@pytest.mark.parametrize("command", NATIVE_TEST_COMMANDS)
def test_native_profile_one_env_and_root_contract_for_all_commands_and_roles(native_request, command, role):
    args, env = native_request
    args.campaign_init = role in ("init", "worker-init", "observer-init")
    args.campaign_retire = role in ("retire", "worker-retire", "observer-retire")
    args._telemetry = role.startswith("worker")
    args._observer = role.startswith("observer")
    args.argv = ["--", *command] if role == "run" else []
    assert g.validate_request(args, env, 100).profile == NATIVE_TRIAL
    assert env == NATIVE_TEST_ENV and args.disk_root == NATIVE_TEST_ROOTS


@pytest.mark.parametrize("key", list(NATIVE_TEST_ENV))
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_native_profile_env_mutations_before_effects(monkeypatch, native_request, key, mutation):
    args, env = native_request
    if mutation == "missing":
        del env[key]
    else:
        env[key] = "wrong"
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("key,value", [("UV_CACHE_DIR", "/cache"), ("UV_BUILD_DIR", "/tmp"),
    ("PATH", "/Users/blairhudson/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin"),
    ("UV_CONFIG_FILE", "/config"), ("UV_NO_CONFIG", "1"), ("UV_PROJECT", "/other"),
    ("RUSTFLAGS", ""), ("LC_CTYPE", "C"), ("LC_ALL", "UTF-8"), ("GIT_DIR", ""), ("SHELL", "/bin/sh")])
def test_native_profile_extra_env_before_effects(monkeypatch, native_request, key, value):
    args, env = native_request
    env[key] = value
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("locale", [None, "UTF-8", "C.UTF-8", "C.utf8"])
def test_native_profile_only_recognized_locale(native_request, locale):
    args, env = native_request
    if locale is not None:
        env["LC_CTYPE"] = locale
    assert g.validate_request(args, env, 100).profile == NATIVE_TRIAL


@pytest.mark.parametrize("mutation", ["cwd", "coordinator_root", "slot", "target_dir", "log_dir", "temp_dir",
    "actual-cwd", "roots-empty", "roots-missing", "roots-extra", "roots-order", "roots-duplicate",
    "roots-substitute", "roots-old-cargo-manifest"])
def test_native_profile_root_mutations_before_effects(monkeypatch, native_request, mutation):
    args, env = native_request
    if mutation == "actual-cwd":
        monkeypatch.setattr(g.os, "getcwd", lambda: "/foreign")
    elif mutation.startswith("roots-"):
        roots = args.disk_root.copy()
        if mutation == "roots-empty":
            roots = []
        elif mutation == "roots-missing":
            roots.pop()
        elif mutation == "roots-extra":
            roots.append(Path("/extra"))
        elif mutation == "roots-duplicate":
            roots.append(roots[0])
        elif mutation == "roots-substitute":
            roots[2] = Path("/extra")
        elif mutation == "roots-old-cargo-manifest":
            roots = roots[:3]
        else:
            roots.reverse()
        args.disk_root = roots
    else:
        setattr(args, mutation, Path("/wrong"))
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("rss,reserve", [(8 * g.GIB, 16 * g.GIB), (4 * g.GIB - 1, 8 * g.GIB),
    (4 * g.GIB + 1, 8 * g.GIB), (4 * g.GIB, 8 * g.GIB - 1), (4 * g.GIB, 8 * g.GIB + 1)])
def test_native_profile_exact_memory_options(monkeypatch, native_request, rss, reserve):
    args, env = native_request
    args.rss_bytes, args.reserve_bytes = rss, reserve
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("role", ["init", "retire", "worker", "observer"])
@pytest.mark.parametrize("fault", ["payload", "env", "bounds"])
def test_native_profile_roles_cannot_bypass_validation(monkeypatch, native_request, role, fault):
    args, env = native_request
    args.campaign_init, args.campaign_retire = role == "init", role == "retire"
    args._telemetry, args._observer = role == "worker", role == "observer"
    args.argv = []
    if fault == "payload":
        args.argv = ["--", "arbitrary"]
    elif fault == "env":
        env["UV_CACHE_DIR"] = "/cache"
    else:
        args.timeout = 601
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("profile", ["default", "compiler-tests-12-8-4", NATIVE_TRIAL])
def test_native_write_roots_omit_uv_default_only_for_new_profile(monkeypatch, native_request, profile):
    args, env = native_request
    args.profile = profile
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(Path, "home", lambda: Path("/Users/blairhudson"))
    paths = g.write_roots(args, env)
    uv = Path("/Users/blairhudson/Library/Caches/uv")
    assert (uv in paths) is (profile != NATIVE_TRIAL)
    if profile == NATIVE_TRIAL:
        assert set(paths) == {Path(NATIVE_TEST_OUTPUT + "/cargo"), Path(NATIVE_TEST_OUTPUT + "/logs"),
                              Path(NATIVE_TEST_OUTPUT + "/tmp"), *NATIVE_TEST_ROOTS}


@pytest.fixture
def init_main(monkeypatch):
    """Real parser/main/run, entirely synthetic effects; definitions only, UNRUN."""
    original_run = g.run
    original_diagnostic = g.diagnostic

    def invoke(profile=NATIVE_TRIAL, opted=True, fault=None, cleanup_fault=None, operation="init",
               worker=False, payload=None, spelling="--init-phase-exit-codes", sink="empty"):
        from types import SimpleNamespace
        fake, results, messages, options_seen = Fake(), [], [], []
        fake.profile = profile
        options = native_options() if profile != "compiler-tests-12-8-4" else compiler_options()
        options[options.index("--profile") + 1] = profile
        if profile == "default":
            options[options.index("--rss-bytes") + 1] = str(8 * g.GIB)
            options[options.index("--reserve-bytes") + 1] = str(16 * g.GIB)
        request_slot = g.parser().parse_args(options).slot
        original_response = fake.read
        def response():
            data = original_response()
            if data is not None:
                data["slot"] = str(request_slot)
            return data
        fake.read = response  # fixture-specific; do not alter shared Fake or slot checks
        if operation != "run":
            options.append("--campaign-" + operation)
        if worker:
            options.append("--_telemetry" if worker == "authority" else "--_observer")
        if opted:
            options.append(spelling)
        env = (NATIVE_TEST_ENV.copy() if profile != "compiler-tests-12-8-4" else {
            "HOME": "/Users/blairhudson",
            "PATH": "/Users/blairhudson/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
            "CARGO_HOME": "/Users/blairhudson/.cargo", "RUSTUP_HOME": "/Users/blairhudson/.rustup",
            "RUSTUP_TOOLCHAIN": "1.85.1-aarch64-apple-darwin", "CARGO_NET_OFFLINE": "true",
            "CARGO_TARGET_DIR": TRIAL_OUTPUT + "/cargo", "TMPDIR": TRIAL_OUTPUT + "/tmp",
            "UV_CACHE_DIR": TRIAL_OUTPUT + "/tmp", "__CF_USER_TEXT_ENCODING": "0x1F5:0:0"})
        fake.children = {}
        def bootstrap(original):
            options_seen.append(("worker", original.copy()))
            if fault == "bootstrap":
                raise g.Blocked("bootstrap fault")
            if fault == "partial-bootstrap":
                fake.children = {"worker": SimpleNamespace(pid=2), "observer": SimpleNamespace(pid=6)}
                raise g.Blocked("partial registration")
            fake.children["worker"] = SimpleNamespace(pid=2)
        fake.worker = bootstrap
        def observer_bootstrap(original):
            options_seen.append(("observer", original.copy()))
            fake.children["observer"] = SimpleNamespace(pid=6)
        fake.observer = observer_bootstrap
        def fail(*args, **kwargs):
            raise g.Blocked("injected classified fault")
        if fault == "validation":
            options += ["--timeout", "601"]
        elif fault == "parse":
            options += ["--timeout"]
        elif fault == "management-conflict":
            options += ["--campaign-retire"]
        elif fault == "legacy-abbreviation":
            options[options.index("--campaign-deadline")] = "--campaign-dead"
        elif fault == "startup":
            fake.startup_step = fail
        elif fault == "status":
            fake.observe_request = fail  # observer before authority request dispatch
        elif fault == "acquire-marker":
            fake.acquire = fail
        elif fault in ("acquisition", "final", "partial-bootstrap"):
            original_read = fake.read
            def read():
                if (fake.acquiring if fault != "final" else fake.final):
                    fail()
                return original_read()
            fake.read = read
        elif fault == "release":
            fake.release_error = True
        elif fault == "denial":
            fake.host_value = host(free_bytes=8 * g.GIB)
        elif fault == "cleanup-only":
            fake.close = fail
        if fault == "escape-finally" or cleanup_fault == "escape-finally":
            original_sleep = fake.sleep
            def sleep(seconds):
                if fake.stops:
                    fail()
                original_sleep(seconds)
            fake.sleep = sleep
        if cleanup_fault == "signal":
            original_signal = fake.signal
            def signaling(*args):
                original_signal(*args)
                fail()
            fake.signal = signaling
        elif cleanup_fault == "reap":
            original_reap, failed = fake.reap, []
            def reaping(name):
                if not failed:
                    failed.append(name)
                    fail()
                return original_reap(name)
            fake.reap = reaping
        elif cleanup_fault == "close":
            original_close = fake.close
            def closing(name):
                original_close(name)
                fail()
            fake.close = closing
        class MockReal:
            @staticmethod
            def context():
                if fault == "context":
                    fail()
            def __new__(cls):
                return fake
        def supervise(*args, **kwargs):
            if fault == "outer":
                fail()
            result = original_run(*args, **kwargs)
            results.append(result)
            return result
        argv = ["command_guard.py", *options]
        if payload is not None:
            argv += ["--", *payload]
        def emit(message):
            messages.append(message)
            if sink == "fd":
                return original_diagnostic(message)
            if sink == "blocked":
                return False
            return False  # no FD or TextIO operation, including empty diagnostics
        with monkeypatch.context() as patch:
            patch.setattr(g, "Real", MockReal)
            patch.setattr(g, "run", supervise)
            patch.setattr(g, "diagnostic", emit)
            patch.setattr(g.os, "environ", env)
            patch.setattr(g.os, "getcwd", lambda: NATIVE_TEST_ROOT)
            patch.setattr(g.os, "getpid", lambda: 1)
            patch.setattr(g.time, "monotonic", fake.now)
            patch.setattr(g.signal, "signal", lambda *args: None)
            patch.setattr(g.sys, "argv", argv)
            if fault == "final-qualify":
                patch.setattr(g.Family, "live_payload", fail)
            if fault in ("parse", "management-conflict"):
                patch.setattr(g.argparse.ArgumentParser, "_print_message", lambda *args: None)
            code = g.main()
        return SimpleNamespace(code=code, fake=fake, results=results, messages=messages,
                               options=options, options_seen=options_seen)
    return invoke


@pytest.mark.parametrize("profile", ["default", "compiler-tests-12-8-4", NATIVE_TRIAL])
@pytest.mark.parametrize("fault,phase,code", [
    ("validation", None, 64), ("bootstrap", None, 65), ("startup", "worker-setup", 65),
    ("status", "initial-status", 66), ("acquire-marker", "acquisition", 67),
    ("acquisition", "acquisition", 67), ("final", "final-seal", 68),
    ("final-qualify", "final-seal", 68), ("release", "final-authority", 69),
])
def test_init_phase_codes_keep_actual_outcome_and_cleanup_identical(init_main, profile, fault, phase, code):
    ordinary = init_main(profile, opted=False, fault=fault)
    opted = init_main(profile, fault=fault)
    assert ordinary.code == 78 and opted.code == code
    assert ordinary.fake.events == opted.fake.events
    assert ordinary.fake.reaped == opted.fake.reaped
    assert ordinary.fake.acquired == opted.fake.acquired
    assert ordinary.fake.slot_pending == opted.fake.slot_pending
    assert len(ordinary.results) == len(opted.results)
    if ordinary.results:
        before, after = ordinary.results[0], opted.results[0]
        assert replace(after, diagnostic_phase=None) == before
        assert after.code == 78 and after.diagnostic_phase == phase
        assert after.held  # known phase is not proof of released/known custody
    expected = [token for token in opted.options if token != "--init-phase-exit-codes"]
    assert all(seen == expected for _, seen in opted.options_seen)


@pytest.mark.parametrize("cleanup_fault", ["signal", "reap", "close"])
@pytest.mark.parametrize("fault,phase,code", [("status", "initial-status", 66),
                                           ("acquisition", "acquisition", 67), ("final", "final-seal", 68)])
def test_init_phase_first_sample_fault_survives_handled_cleanup_faults(init_main, cleanup_fault, fault, phase, code):
    ordinary = init_main(opted=False, fault=fault, cleanup_fault=cleanup_fault)
    opted = init_main(fault=fault, cleanup_fault=cleanup_fault)
    assert ordinary.code == 78 and opted.code == code
    assert ordinary.fake.events == opted.fake.events
    assert replace(opted.results[0], diagnostic_phase=None) == ordinary.results[0]
    assert opted.results[0].diagnostic_phase == phase and opted.results[0].held


def test_init_phase_partial_bootstrap_latches_before_later_acquisition_fault(init_main):
    ordinary = init_main(opted=False, fault="partial-bootstrap")
    opted = init_main(fault="partial-bootstrap")
    assert ordinary.code == 78 and opted.code == 65
    assert opted.results[0].diagnostic_phase == "worker-setup"
    assert "A acquisition" in opted.results[0].reason
    assert replace(opted.results[0], diagnostic_phase=None) == ordinary.results[0]
    assert ordinary.fake.events == opted.fake.events


@pytest.mark.parametrize("fault", ["denial", "cleanup-only", "escape-finally", "outer", "context"])
def test_init_phase_unclassified_denial_cleanup_and_escaping_finalizer_stay_78(init_main, fault):
    ordinary = init_main(opted=False, fault=fault)
    opted = init_main(fault=fault)
    assert ordinary.code == opted.code == 78
    assert ordinary.fake.events == opted.fake.events
    assert ordinary.results == opted.results
    assert all(result.diagnostic_phase is None for result in opted.results)


@pytest.mark.parametrize("fault,phase,code", [("status", "initial-status", 66),
                                           ("acquisition", "acquisition", 67), ("final", "final-seal", 68)])
def test_init_phase_classified_sampling_then_escaping_finalizer_returns_unclassified_78(init_main, fault, phase, code):
    classified = init_main(fault=fault)
    assert classified.code == code and classified.results[0].diagnostic_phase == phase
    ordinary = init_main(opted=False, fault=fault, cleanup_fault="escape-finally")
    opted = init_main(fault=fault, cleanup_fault="escape-finally")
    assert ordinary.code == opted.code == 78
    assert ordinary.results == opted.results == []  # finalizer escaped run: no Outcome to map
    assert ordinary.fake.events == opted.fake.events
    assert any(event[0] == "signal" for event in opted.fake.events)


@pytest.mark.parametrize("profile", ["default", "compiler-tests-12-8-4", NATIVE_TRIAL])
@pytest.mark.parametrize("sink", ["empty", "blocked"])
def test_init_phase_success_and_missing_diagnostic_delivery_do_not_change_status(init_main, sink, profile):
    success = init_main(profile=profile, sink=sink)
    assert success.code == 0 and success.results == [g.Outcome(0, False, "explicit campaign init")]
    assert init_main(profile=profile, fault="acquisition", sink=sink).code == 67


@pytest.mark.parametrize("sink", ["blocking", "empty", "blocked"])
def test_init_phase_real_diagnostic_boundary_cannot_change_exit_status(init_main, diagnostic_fd, sink):
    state, events = diagnostic_fd
    if sink == "blocking":
        state["flags"] = 0
    else:
        state["write_result"] = "zero" if sink == "empty" else "again"
    assert init_main(fault="acquisition", sink="fd").code == 67
    assert init_main(sink="fd").code == 0
    assert events and not any(event[0] in ("text_write", "flush") for event in events)
    if sink == "blocking":
        assert not any(event[0] == "write" for event in events)


@pytest.mark.parametrize("operation,worker,payload", [
    ("retire", False, None), ("run", False, ["cargo", "test"]),
    ("init", "authority", None), ("init", "observer", None), ("init", False, ["cargo", "test"]),
])
def test_init_phase_bad_combinations_refused_before_children(init_main, operation, worker, payload):
    result = init_main(operation=operation, worker=worker, payload=payload)
    assert result.code == 78 and result.options_seen == [] and result.fake.events == []
    assert len(result.messages) == 1


@pytest.mark.parametrize("spelling", ["--init-phase", "--init-phase-exit", "--init-phase-exit-code"])
def test_init_phase_abbreviations_refused_before_children(init_main, spelling):
    result = init_main(spelling=spelling)
    assert result.code == 78 and result.options_seen == [] and result.fake.events == []
    assert "exact spelling" in result.messages[0]


def test_init_phase_ordinary_parser_syntax_errors_still_exit_two(init_main):
    with pytest.raises(SystemExit) as caught:
        init_main(fault="parse")
    assert caught.value.code == 2


@pytest.mark.parametrize("profile", ["default", "compiler-tests-12-8-4", NATIVE_TRIAL])
def test_init_phase_simultaneous_management_flags_refused_before_children(init_main, profile):
    result = init_main(profile=profile, fault="management-conflict")
    assert result.code == 78 and result.results == []
    assert result.options_seen == [] and result.fake.events == []
    assert len(result.messages) == 1 and "payload-free parent campaign init" in result.messages[0]


@pytest.mark.parametrize("payload", [None, ["--init-phase-exit-codes"]])
def test_init_phase_without_optin_management_conflict_still_parser_two(init_main, payload):
    with pytest.raises(SystemExit) as caught:
        init_main(opted=False, fault="management-conflict", payload=payload)
    assert caught.value.code == 2


def test_init_phase_legacy_option_abbreviation_still_accepted(init_main):
    result = init_main(profile="default", opted=False, fault="legacy-abbreviation")
    assert result.code == 0 and result.results[0].diagnostic_phase is None
    assert result.options_seen and "--campaign-dead" in result.options
    assert all(options == result.options for _, options in result.options_seen)


@pytest.mark.parametrize("exit_code", [0, 9, -signal.SIGTERM])
def test_init_phase_default_payload_status_and_payload_flag_tokens_unchanged(monkeypatch, init_main, exit_code):
    original = Fake.__init__
    def initialize(fake):
        original(fake)
        fake.exit_code = exit_code
    monkeypatch.setattr(Fake, "__init__", initialize)
    payload = ["mock-only", "--init-phase-exit-codes", "--init-phase", "--campaign-init", "--campaign-retire"]
    result = init_main(profile="default", opted=False, operation="run", payload=payload)
    assert result.code == (1 if exit_code else 0)
    assert result.results[0].diagnostic_phase is None
    launch = next(event for event in result.fake.events if event[0] == "launch")
    assert launch[1] == payload
    assert all("--init-phase-exit-codes" not in options for _, options in result.options_seen)


def test_init_phase_run_bounds_failure_keeps_outcome_78_and_legacy_positional_api(diagnostic_messages):
    fake = Fake()
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000, seconds=601), fake, 1, "init",
                   init_phase_exit_codes=True)
    assert result.code == 78 and result.held and result.diagnostic_phase == "request-validation"
    assert g.Outcome(78, True, "legacy").diagnostic_phase is None


@pytest.mark.parametrize("role", ["run", "retire", "authority", "observer", "init-payload"])
def test_init_phase_bad_roles_use_existing_forbidden_effects_harness(monkeypatch, native_request, role):
    args, env = native_request
    args.init_phase_exit_codes = True
    args.campaign_init, args.campaign_retire = role not in ("run", "retire"), role == "retire"
    args._telemetry, args._observer = role == "authority", role == "observer"
    if role != "run":
        args.argv = ["--", "forbidden"] if role == "init-payload" else []
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("selected,persisted", [
    ("default", NATIVE_TRIAL), ("compiler-tests-12-8-4", NATIVE_TRIAL),
    (NATIVE_TRIAL, "default"), (NATIVE_TRIAL, "compiler-tests-12-8-4"),
])
@pytest.mark.parametrize("boundary", ["campaign", "budget", "commit", "debit", "accept", "release"])
def test_native_cross_profile_refusal_precedes_effects(monkeypatch, boot_records, selected, persisted, boundary):
    from types import SimpleNamespace
    campaign, record, roots = boot_records
    campaign["profile"] = record["campaign"]["profile"] = persisted
    calls = []
    def forbidden(*args, **kwargs):
        calls.append("effect")
        raise g.Blocked("forbidden effect")
    with monkeypatch.context() as patch:
        patch.setattr(g.os, "open", forbidden)
        patch.setattr(g.os, "write", forbidden)
        patch.setattr(g.os, "unlink", forbidden)
        patch.setattr(g.os, "fstat", forbidden)
        if boundary == "campaign":
            operation = lambda: g.validate_campaign(campaign, BOOT, 100, profile=selected)
        elif boundary == "budget":
            operation = lambda: g.validate_budget(record, campaign, roots, profile=selected)
        elif boundary in ("commit", "debit"):
            a = SimpleNamespace(profile=selected, campaign=campaign)
            if boundary == "debit":
                operation = lambda: g.Authority.debit(a, {}, {})
            else:
                budget = g.Budget.__new__(g.Budget)
                budget.authority = a
                operation = lambda: budget.commit({})
        elif boundary == "accept":
            adapter = g.Real()
            adapter.profile, adapter.operation = selected, "init"
            adapter.now = lambda: 100
            operation = lambda: adapter.accept({"custody": dict(profile=persisted, campaign=campaign)}, True)
        else:
            operation = lambda: g.finish_authority(dict(profile=persisted, campaign=campaign), SLOT, 105,
                                                   profile=selected)
        with pytest.raises(g.Blocked, match="profile"):
            operation()
    assert calls == []


@pytest.mark.parametrize("profile,operation,persisted", [
    (profile, operation, profile)
    for profile in ("default", "compiler-tests-12-8-4", NATIVE_TRIAL)
    for operation in ("init", "run", "retire")
] + [(selected, operation, persisted)
     for selected, persisted in (("default", NATIVE_TRIAL), ("compiler-tests-12-8-4", NATIVE_TRIAL),
                                 (NATIVE_TRIAL, "default"), (NATIVE_TRIAL, "compiler-tests-12-8-4"))
     for operation in ("run", "retire")])
def test_native_authority_expected_profile_chain_with_mocked_paths(monkeypatch, boot_records, profile,
                                                                  operation, persisted):
    from types import SimpleNamespace
    campaign, _, _ = boot_records
    campaign["profile"] = persisted
    args = settings(Path("/main"), profile=profile, campaign_init=operation == "init",
                    campaign_retire=operation == "retire")
    metadata = SimpleNamespace(st_dev=1, st_ino=2, st_birthtime=3, st_uid=7,
                               st_mode=g.statmod.S_IFDIR | 0o700)
    token = [1, 2, 3, g.statmod.S_IFDIR, 7, None, None, 0o700]
    loaded, closed, reads = [], [], []
    class MockPins:
        def __init__(self):
            self.entries = {}
        def add(self, path):
            return Path(path)
        def check(self):
            pass
        def wire(self):
            return [("/main/target", token)]
    def load(a):
        loaded.append(a.profile)
        return SimpleNamespace(fd=11, verify=lambda: None)
    def read(fd, *args):
        reads.append(fd)
        return campaign, []
    monkeypatch.setattr(g, "Pins", MockPins)
    monkeypatch.setattr(g, "canonical_slot", lambda *args: SLOT)
    monkeypatch.setattr(g, "write_roots", lambda *args: [])
    monkeypatch.setattr(Path, "stat", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(Path, "is_dir", lambda *args: True)
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)
    monkeypatch.setattr(g.os, "open", lambda *args: 10)
    monkeypatch.setattr(g.os, "fstat", lambda *args: metadata)
    monkeypatch.setattr(g.os, "getuid", lambda: 7)
    monkeypatch.setattr(g.os, "close", closed.append)
    monkeypatch.setattr(g, "absent", lambda *args: None)
    monkeypatch.setattr(g, "read_campaign", read)
    monkeypatch.setattr(g.Budget, "load", load)
    campaign["roots"] = dict(declarations=[["/main/target", "/main/target", [1, 2, 3]]],
                              pins=[["/main/target", token]])
    if persisted != profile:
        with pytest.raises(g.Blocked, match="profile"):
            g.Authority(args, {}, BOOT, 100)
        assert loaded == [] and closed == [10] and reads == [10]
        return
    a = g.Authority(args, {}, BOOT, 100)
    assert a.profile == a.campaign["profile"] == profile
    assert a.operation == operation
    assert loaded == ([] if operation == "init" else [profile])
    a.close()
    assert closed == ([10] if operation == "init" else [11, 10])


@pytest.mark.parametrize("operation", ["init", "retire"])
def test_native_mocked_budget_finish_lifecycle(budget_memory_io, monkeypatch, operation):
    from types import SimpleNamespace
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = NATIVE_TRIAL
    budget = g.Budget(a, 11)
    budget.commit({}, g.SLOT_CHARGE + g.CAMPAIGN_CHARGE)
    proof = budget.proof()
    custody = dict(profile=NATIVE_TRIAL, campaign=a.campaign, slot=str(SLOT), pins=[],
                   directory=list(g.identity(state.stat)), owned_slot=g.owned_file(state.stat),
                   operation=operation, campaign_token=[], budget=proof)
    monkeypatch.setattr(g, "Pins", lambda entries: SimpleNamespace(check=lambda: None))
    monkeypatch.setattr(g, "absent", lambda *args: None)
    monkeypatch.setattr(g, "read_campaign", lambda *args: (a.campaign, g.owned_file(state.stat)))
    if operation == "retire":
        custody["campaign_token"] = g.owned_file(state.stat)
    # Synthetic publication shares metadata provider, but not stored budget bytes.
    writes = []
    monkeypatch.setattr(g.os, "write", lambda fd, raw: (writes.append(raw), len(raw))[1])
    unlinked = []
    monkeypatch.setattr(g.os, "unlink", lambda name, **kwargs: unlinked.append(name))
    g.finish_authority(custody, SLOT, 105, profile=NATIVE_TRIAL)
    assert unlinked == ([g.SLOT_NAME] if operation == "init" else [g.CAMPAIGN, g.BUDGET, g.SLOT_NAME])
    assert len(writes) == (1 if operation == "init" else 0)
    if writes:
        assert g.decode_state(writes[0])["profile"] == NATIVE_TRIAL


# Compiler-profile definitions only. Commands/env are independent recipe literals,
# never copied from the production allowlist. No compiler execution in these tests.
TRIAL = "compiler-tests-12-8-4"
TRIAL_ROOT = "/Users/blairhudson/workspace/pllm"
TRIAL_OUTPUT = TRIAL_ROOT + "/target/qwen-qualification-20261105"
TRIAL_COMMANDS = [
    ["/Users/blairhudson/.cargo/bin/cargo", "+1.85.1-aarch64-apple-darwin", "test",
     "-p", "pllm-compiler", "--locked", "-j1", "--lib",
     "replicated_decoder::program::request_origin::compact_storage::context_quote_tests::",
     "--", "--test-threads=1"],
    ["/Users/blairhudson/.cargo/bin/cargo", "+1.85.1-aarch64-apple-darwin", "test",
     "-p", "pllm-compiler", "--locked", "-j1", "--lib",
     "replicated_decoder::program::request_origin::source_inspection_extent::tests::",
     "--", "--test-threads=1"],
]


def compiler_options():
    return ["--profile", TRIAL, "--rss-bytes", "4294967296", "--reserve-bytes", "8589934592",
            "--cwd", TRIAL_ROOT, "--coordinator-root", TRIAL_ROOT,
            "--slot", TRIAL_ROOT + "/target/native-command.slot",
            "--target-dir", TRIAL_OUTPUT + "/cargo", "--log-dir", TRIAL_OUTPUT + "/logs",
            "--temp-dir", TRIAL_OUTPUT + "/tmp", "--campaign-deadline", "1000",
            "--disk-root", "/Users/blairhudson/.cargo",
            "--disk-root", "/Users/blairhudson/.rustup",
            "--disk-root", "/Users/blairhudson/.local/share/uv/python",
            "--cooperative-build"]


@pytest.fixture
def compiler_request(monkeypatch):
    args = g.parser().parse_args(compiler_options() + ["--", *TRIAL_COMMANDS[0]])
    env = {
        "HOME": "/Users/blairhudson",
        "PATH": "/Users/blairhudson/.cargo/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "CARGO_HOME": "/Users/blairhudson/.cargo", "RUSTUP_HOME": "/Users/blairhudson/.rustup",
        "RUSTUP_TOOLCHAIN": "1.85.1-aarch64-apple-darwin", "CARGO_NET_OFFLINE": "true",
        "CARGO_TARGET_DIR": TRIAL_OUTPUT + "/cargo", "TMPDIR": TRIAL_OUTPUT + "/tmp",
        "UV_CACHE_DIR": TRIAL_OUTPUT + "/tmp",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0:0",
    }
    monkeypatch.setattr(g.os, "getcwd", lambda: TRIAL_ROOT)
    return args, env


def rejected_request_main(monkeypatch, args, env):
    """Log forbidden calls; assert outside production catches and patched OS scope."""
    import ctypes
    from types import SimpleNamespace
    calls, messages, contexts = [], [], []
    def forbidden(name):
        def call(*values, **kwargs):
            calls.append(name)
            raise g.Blocked("forbidden test effect")
        return call
    class NoReal:
        @staticmethod
        def context():
            contexts.append("context")
        def __new__(cls):
            return forbidden("Real")()
    with monkeypatch.context() as patch:
        patch.setattr(g, "Real", NoReal)
        patch.setattr(g.sys, "argv", ["command_guard.py"])
        patch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
        patch.setattr(g, "diagnostic", messages.append)
        patch.setattr(g, "telemetry_worker", forbidden("dispatch"))
        patch.setattr(g, "Authority", forbidden("authority"))
        patch.setattr(g, "Disk", forbidden("disk"))
        patch.setattr(g, "DarwinProbes", forbidden("probes"))
        patch.setattr(g, "boot_identity", forbidden("boot"))
        patch.setattr(g.host_swap, "host_snapshot", forbidden("host"))
        patch.setattr(ctypes, "CDLL", forbidden("native-library"))
        patch.setattr(g.subprocess, "Popen", forbidden("subprocess"))
        patch.setattr(g.os, "fork", forbidden("fork"))
        patch.setattr(g.os, "open", forbidden("open/write-authority"))
        patch.setattr(Path, "resolve", forbidden("resolve"))
        patch.setattr(Path, "stat", forbidden("stat"))
        patch.setattr(Path, "lstat", forbidden("lstat"))
        patch.setattr(g.os, "environ", env)
        patch.setattr(g.time, "monotonic", lambda: 100)
        code = g.main()
    assert code == 78
    assert contexts == ["context"] and calls == []
    assert len(messages) == 1
    return messages


def test_profile_legacy_defaults_and_full_positional_numeric_api():
    legacy = g.Limits(1000, 600, 8 * g.GIB, 16 * g.GIB, 20 * g.GIB, 0.1, 2, 1, 1)
    assert legacy == g.Limits(1000) and legacy.profile == "default"
    legacy.validate(100)
    replace(legacy, rss=1, reserve=24 * g.GIB, seconds=1).validate(100)


@pytest.mark.parametrize("field", ["campaign_deadline", "seconds", "rss", "reserve", "disk_growth",
                                        "interval", "probe_timeout", "grace", "reap_timeout"])
@pytest.mark.parametrize("value", [True, None, "1", 0, -1, float("nan")])
def test_profile_does_not_exempt_numeric_bounds(field, value):
    with pytest.raises(g.Blocked, match="invalid bounds"):
        replace(g.Limits(1000), **{field: value}).validate(100)


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "", "DEFAULT", "compiler-tests"])
def test_profile_type_and_closed_set_refused_before_effects(monkeypatch, compiler_request, value):
    args, env = compiler_request
    args.profile = value
    with pytest.raises(g.Blocked, match="profile"):
        replace(g.Limits(1000), profile=value).validate(100)
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("command", TRIAL_COMMANDS)
def test_profile_accepts_each_exact_independent_compiler_recipe(compiler_request, command):
    args, env = compiler_request
    assert len(env) == 12 and env == g.COMPILER_ENV
    assert args.disk_root == [Path("/Users/blairhudson/.cargo"),
                              Path("/Users/blairhudson/.rustup"),
                              Path("/Users/blairhudson/.local/share/uv/python")]
    args.argv = ["--", *command]
    limits = g.validate_request(args, env, 100)
    assert limits == g.Limits(1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=TRIAL)


@pytest.mark.parametrize("rss,reserve", [
    (8 * g.GIB, 16 * g.GIB), (4 * g.GIB - 1, 8 * g.GIB),
    (4 * g.GIB + 1, 8 * g.GIB), (4 * g.GIB, 8 * g.GIB - 1),
    (4 * g.GIB, 8 * g.GIB + 1),
])
def test_profile_trial_requires_exact_four_eight_not_new_ranges(monkeypatch, compiler_request, rss, reserve):
    args, env = compiler_request
    args.rss_bytes, args.reserve_bytes = rss, reserve
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("omit", ["--rss-bytes", "--reserve-bytes", "both"])
def test_profile_trial_cli_requires_both_explicit_memory_options(monkeypatch, compiler_request, omit):
    _, env = compiler_request
    options = compiler_options()
    for name in ("--rss-bytes", "--reserve-bytes"):
        if omit in (name, "both"):
            index = options.index(name)
            del options[index:index + 2]
    args = g.parser().parse_args(options + ["--", *TRIAL_COMMANDS[0]])
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("change", [dict(seconds=600.1), dict(disk_growth=20 * g.GIB + 1),
    dict(campaign_deadline=7300.1), dict(interval=0.11), dict(probe_timeout=2.1),
    dict(grace=0.5), dict(reap_timeout=1.1)])
def test_profile_trial_keeps_shared_time_disk_poll_cleanup_bounds(change, profile):
    limits = g.Limits(1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=profile)
    with pytest.raises(g.Blocked):
        replace(limits, **change).validate(100)


@pytest.mark.parametrize("mutation", ["relative-cargo", "toolchain", "build", "package", "unlocked",
    "jobs", "no-run", "ignored", "prefix", "filter-suffix", "exact", "shell", "reordered"])
def test_profile_wrong_compiler_argv_refused_before_effects(monkeypatch, compiler_request, mutation):
    args, env = compiler_request
    command = TRIAL_COMMANDS[0].copy()
    if mutation == "relative-cargo":
        command[0] = "cargo"
    elif mutation == "toolchain":
        command[1] = "+stable"
    elif mutation == "build":
        command[2] = "build"
    elif mutation == "package":
        command[4] = "pllm"
    elif mutation == "unlocked":
        command.remove("--locked")
    elif mutation == "jobs":
        command[6] = "-j2"
    elif mutation == "no-run":
        command.insert(8, "--no-run")
    elif mutation == "ignored":
        command.append("--ignored")
    elif mutation == "prefix":
        command[8] = "replicated_decoder::"
    elif mutation == "filter-suffix":
        command[8] = command[8].rstrip(":")
    elif mutation == "exact":
        command.append("--exact")
    elif mutation == "shell":
        command = ["/bin/sh", "-c", " ".join(command)]
    else:
        command[5], command[6] = command[6], command[5]
    args.argv = ["--", *command]
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("field", ["cwd", "coordinator_root", "slot", "target_dir", "log_dir", "temp_dir",
                                  "disk_root", "actual-cwd"])
def test_profile_wrong_paths_refused_before_effects(monkeypatch, compiler_request, field):
    args, env = compiler_request
    if field == "actual-cwd":
        monkeypatch.setattr(g.os, "getcwd", lambda: "/foreign")
    else:
        setattr(args, field, [Path("/extra")] if field == "disk_root" else Path("/wrong"))
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("mutation", ["empty", "missing", "substituted", "extra", "duplicate", "order"])
def test_profile_disk_roots_require_exact_approved_sequence_before_effects(monkeypatch, compiler_request, mutation):
    args, env = compiler_request
    roots = [Path("/Users/blairhudson/.cargo"), Path("/Users/blairhudson/.rustup"),
             Path("/Users/blairhudson/.local/share/uv/python")]
    if mutation == "empty":
        roots = []
    elif mutation == "missing":
        roots.pop()
    elif mutation == "substituted":
        roots[2] = Path("/extra")  # same length, unlike the existing singleton /extra case
    elif mutation == "extra":
        roots.append(Path("/extra"))
    elif mutation == "duplicate":
        roots.append(roots[0])
    else:
        roots[0], roots[1] = roots[1], roots[0]
    args.disk_root = roots
    messages = rejected_request_main(monkeypatch, args, env)
    assert messages == [
        "BLOCKED slot retained if acquired: control failure: Blocked: "
        "compiler-test profile requires fixed paths and approved disk roots",
    ]


@pytest.mark.parametrize("key", ["HOME", "PATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
    "CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN", "CARGO_NET_OFFLINE", "CARGO_TARGET_DIR", "TMPDIR",
    "UV_CACHE_DIR", "__CF_USER_TEXT_ENCODING"])
@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_profile_clean_environment_values_are_required(monkeypatch, compiler_request, key, mutation):
    args, env = compiler_request
    if mutation == "missing":
        del env[key]
    else:
        env[key] = "wrong"
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("value", [
    "", "501:0:0", "0x1f5:0:0", "0x1F6:0:0", "0x1F5:1:0", "0x1F5:0:1",
])
def test_profile_noncanonical_cf_encoding_refused_before_effects(monkeypatch, compiler_request, value):
    args, env = compiler_request
    env["__CF_USER_TEXT_ENCODING"] = value
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("key,value", [
    ("RUSTFLAGS", ""), ("CARGO_ENCODED_RUSTFLAGS", ""), ("CARGO_BUILD_JOBS", "1"),
    ("DYLD_INSERT_LIBRARIES", "/extra"), ("LD_PRELOAD", "/extra"), ("PYTHONPATH", "/extra"),
    ("GIT_DIR", ""), ("LC_CTYPE", "C"), ("LC_CTYPE", "en_US.UTF-8"),
    ("LC_ALL", "UTF-8"), ("SHELL", "/bin/zsh"),
])
def test_profile_arbitrary_environment_and_locale_extras_refused(monkeypatch, compiler_request, key, value):
    args, env = compiler_request
    env[key] = value
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("locale", [None, "UTF-8", "C.UTF-8", "C.utf8"])
def test_profile_only_optional_cpython_coercion_locales_accepted(compiler_request, locale):
    args, env = compiler_request
    if locale is not None:
        env["LC_CTYPE"] = locale
    assert g.validate_request(args, env, 100).profile == TRIAL


@pytest.mark.parametrize("role", ["run", "init", "retire", "worker", "worker-init", "worker-retire",
                                  "observer", "observer-init", "observer-retire"])
@pytest.mark.parametrize("fault", ["payload", "bounds", "env", "cf-missing", "cf-wrong", "cf-zero"])
def test_profile_management_and_private_role_cannot_bypass_validation(monkeypatch, compiler_request, role, fault):
    args, env = compiler_request
    args.campaign_init = role in ("init", "worker-init", "observer-init")
    args.campaign_retire = role in ("retire", "worker-retire", "observer-retire")
    args._telemetry = role in ("worker", "worker-init", "worker-retire")
    args._observer = role in ("observer", "observer-init", "observer-retire")
    if role != "run":
        args.argv = []
    if fault == "payload":
        args.argv = ["--", "arbitrary-payload"]
    elif fault == "bounds":
        args.rss_bytes = 8 * g.GIB
    elif fault == "cf-missing":
        del env["__CF_USER_TEXT_ENCODING"]
    elif fault in ("cf-wrong", "cf-zero"):
        env["__CF_USER_TEXT_ENCODING"] = "wrong" if fault == "cf-wrong" else "0x1F5:0x0:0x0"
    else:
        env["RUSTFLAGS"] = ""
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("operation", ["run", "init", "retire"])
def test_profile_payload_free_worker_dispatch_validates_before_probes(monkeypatch, compiler_request,
                                                                    native_request, operation, profile):
    from types import SimpleNamespace
    args, env = native_request if profile == NATIVE_TRIAL else compiler_request
    args.argv, args._telemetry = [], True
    args.campaign_init, args.campaign_retire = operation == "init", operation == "retire"
    calls = []
    class NoReal:
        @staticmethod
        def context():
            calls.append("context")
        def __new__(cls):
            calls.append("forbidden-adapter")
            raise g.Blocked("no parent adapter for worker")
    def worker(options):
        calls.append(("worker", options.profile, options.argv.copy()))
        return 0
    monkeypatch.setattr(g, "Real", NoReal)
    monkeypatch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g, "telemetry_worker", worker)
    monkeypatch.setattr(g, "diagnostic", lambda message: calls.append(("diagnostic", message)))
    monkeypatch.setattr(g.os, "environ", env)
    monkeypatch.setattr(g.time, "monotonic", lambda: 100)
    assert g.main() == 0
    assert calls == ["context", ("worker", profile, [])]


@pytest.mark.parametrize("profile", ["default", TRIAL])
def test_profile_parent_forwards_original_option_list_not_dataclass(monkeypatch, compiler_request, profile):
    from types import SimpleNamespace
    args, env = compiler_request
    args.profile = profile
    if profile == "default":
        args.rss_bytes, args.reserve_bytes = 8 * g.GIB, 16 * g.GIB
    options = compiler_options()
    options[1] = profile
    if profile == "default":
        options[options.index("--rss-bytes") + 1] = str(8 * g.GIB)
        options[options.index("--reserve-bytes") + 1] = str(16 * g.GIB)
    fake, calls = Fake(), []
    fake.children = {}
    fake.worker = lambda original: calls.append(("worker", original))
    fake.observer = lambda original: calls.append(("observer", original))
    class MockReal:
        @staticmethod
        def context():
            calls.append("context")
        def __new__(cls):
            calls.append("adapter")
            return fake
    def supervise(argv, cwd, child_env, slot, limits, adapter, pid, operation, started):
        calls.append(("run", argv, limits.profile, adapter.profile, operation, started))
        return g.Outcome(0, False, "mock success")
    monkeypatch.setattr(g, "Real", MockReal)
    monkeypatch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g, "run", supervise)
    monkeypatch.setattr(g, "diagnostic", lambda message: None)
    monkeypatch.setattr(g.os, "environ", env)
    monkeypatch.setattr(g.os, "getpid", lambda: 1)
    monkeypatch.setattr(g.time, "monotonic", fake.now)
    monkeypatch.setattr(g.signal, "signal", lambda *values: None)
    monkeypatch.setattr(g.sys, "argv", ["command_guard.py", *options, "--", *TRIAL_COMMANDS[0]])
    assert g.main() == 0
    assert calls == ["context", "adapter", ("worker", options), ("observer", options),
                     ("run", TRIAL_COMMANDS[0], profile, profile, "run", 100)]


def test_budget_profile_real_bounds_and_unrelated_limits_unchanged():
    assert g.budget_bounds("default") == (983040, 2097152)
    assert g.budget_bounds(TRIAL) == (33554432, 67108864)
    assert g.COMPILER_BUDGET_STATE_LIMIT == 32 * 1024 * 1024
    assert g.COMPILER_BUDGET_WRITE_CHARGE == 64 * 1024 * 1024
    assert g.MAX_RESPONSE == 1048576
    assert g.MAX_RESPONSE // 4 == 262144
    assert g.SLOT_CHARGE == 65536
    assert g.CAMPAIGN_CHARGE == g.WRITE_CHARGE == 2097152
    assert g.Limits(1000).disk_growth == 20 * g.GIB
    assert (20 * g.GIB) // g.COMPILER_BUDGET_WRITE_CHARGE == 320


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("over", [0, 1])
def test_budget_codec_exact_boundary_before_json(monkeypatch, profile, over):
    # Small codec boundary only; real adopted constants asserted separately.
    monkeypatch.setattr(g, "STATE_LIMIT", 128)
    monkeypatch.setattr(g, "COMPILER_BUDGET_STATE_LIMIT", 256)
    limit, _ = g.budget_bounds(profile)
    raw = b'"' + b"x" * (limit - 2 + over) + b'"'
    calls = []
    original = g.json.loads
    def decode(*args, **kwargs):
        calls.append("decode")
        return original(*args, **kwargs)
    monkeypatch.setattr(g.json, "loads", decode)
    if over:
        with pytest.raises(g.Blocked, match="budget decode bound"):
            g.decode_budget(raw, profile)
        with pytest.raises(g.Blocked, match="budget encode bound") as caught:
            g.encode_budget({"previous": [], "padding": "x" * limit}, profile)
        detail = g.exception_detail(caught.value)
        assert "encoded_bytes=" in detail and f"limit={limit}" in detail
        assert "identity_count=0" in detail and "padding" not in detail
        assert calls == []
    else:
        assert g.decode_budget(raw, profile) == "x" * (limit - 2)
        record = {"previous": [], "padding": ""}
        base = len(g.encode_budget(record, profile))
        record["padding"] = "x" * (limit - base)
        assert len(g.encode_budget(record, profile)) == limit
        record["padding"] += "x"
        with pytest.raises(g.Blocked, match="budget encode bound"):
            g.encode_budget(record, profile)
        assert calls == ["decode"]


def test_budget_compiler_codec_larger_json_not_default_or_campaign(monkeypatch):
    # Logical JSON capacity, NOT physical memory/allocation evidence.
    monkeypatch.setattr(g, "STATE_LIMIT", 128)
    monkeypatch.setattr(g, "COMPILER_BUDGET_STATE_LIMIT", 512)
    record = {"previous": [], "padding": "x" * 200}
    raw = g.encode_budget(record, TRIAL)
    assert g.decode_budget(raw, TRIAL) == record
    with pytest.raises(g.Blocked):
        g.encode_budget(record, "default")
    with pytest.raises(g.Blocked):
        g.decode_budget(raw, "default")
    with pytest.raises(g.Blocked):
        g.encode_state(record)
    with pytest.raises(g.Blocked):
        g.decode_state(raw)


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("sequence", [1, 3])
@pytest.mark.parametrize("short", [0, 1])
def test_budget_profile_sequence_charge_floor(boot_records, profile, sequence, short):
    campaign, record, roots = boot_records
    campaign["profile"] = record["campaign"]["profile"] = profile
    record["sequence"] = sequence
    record["charged_bytes"] = (sequence * g.budget_bounds(profile)[1]
                               + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE - short)
    if short:
        with pytest.raises(g.Blocked):
            g.validate_budget(record, campaign, roots, profile=profile)
    else:
        assert g.validate_budget(record, campaign, roots, profile=profile) == {}


@pytest.mark.parametrize("claimed", [TRIAL, NATIVE_TRIAL, "unknown", None, True])
def test_budget_claimed_profile_cannot_escalate_expected_default(boot_records, claimed):
    campaign, record, roots = boot_records
    campaign["profile"] = record["campaign"]["profile"] = claimed
    record["charged_bytes"] = g.COMPILER_BUDGET_WRITE_CHARGE + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE
    with pytest.raises(g.Blocked, match="profile"):
        g.validate_budget(record, campaign, roots, profile="default")


@pytest.fixture
def budget_memory_io(monkeypatch, boot_records):
    """Budget-only fake metadata/I/O. No provider, real Authority, fork or files."""
    from types import SimpleNamespace
    campaign, record, roots = boot_records
    s = SimpleNamespace(st_dev=1, st_ino=4, st_birthtime_ns=5, st_uid=7,
                        st_mode=g.statmod.S_IFREG | 0o600, st_nlink=1,
                        st_size=0, st_blocks=0, st_ctime_ns=1)
    state = SimpleNamespace(raw=b"", events=[], reads=[], opens=[], before=0, after=0,
                            partial=False, mismatch=False, changed=False, stat=s,
                            read_prefix=None)
    def metadata(*args, **kwargs):
        return s
    def opening(*args, **kwargs):
        state.events.append("open")
        state.opens.append((args[0], args[1]))
        return 11
    def write(fd, raw):
        state.events.append("write")
        written = len(raw) - int(state.partial)
        state.raw = raw[:written]
        s.st_size, s.st_ctime_ns = len(state.raw), s.st_ctime_ns + 1
        if state.changed:
            s.st_ino += 1
        return written
    def read(fd, count):
        state.reads.append(count)
        raw = state.raw if state.read_prefix is None else state.read_prefix
        return (b"mismatch" if state.mismatch else raw)[:count]
    monkeypatch.setattr(g.os, "getuid", lambda: 7)
    monkeypatch.setattr(g.os, "fstat", metadata)
    monkeypatch.setattr(g.os, "stat", metadata)
    monkeypatch.setattr(g.os, "open", opening)
    monkeypatch.setattr(g.os, "close", lambda fd: None)
    monkeypatch.setattr(g.os, "lseek", lambda *args: 0)
    monkeypatch.setattr(g.os, "write", write)
    monkeypatch.setattr(g.os, "read", read)
    monkeypatch.setattr(g.os, "fsync", lambda fd: state.events.append("fsync"))
    monkeypatch.setattr(g.os, "ftruncate", lambda fd, size: setattr(s, "st_size", size))
    monkeypatch.setattr(g, "footprint", lambda stat: state.after if "write" in state.events else state.before)
    a = SimpleNamespace(campaign=campaign, manifest=roots, profile="default", fd=10,
                        cap=campaign["disk_cap_bytes"], owned_slot=[1, 2, 3],
                        pending_slot=g.SLOT_CHARGE, budget=None,
                        check_directory=lambda: None, check_slot=lambda: None,
                        check=lambda now: None)
    return state, a


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
def test_budget_oversize_read_rejected_before_json(budget_memory_io, monkeypatch, profile):
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    monkeypatch.setattr(g, "STATE_LIMIT", 128)
    monkeypatch.setattr(g, "COMPILER_BUDGET_STATE_LIMIT", 256)
    limit, _ = g.budget_bounds(profile)
    state.raw = b" " * (limit + 1)
    state.stat.st_size = len(state.raw)
    calls = []
    monkeypatch.setattr(g.json, "loads", lambda *args, **kwargs: calls.append("decode"))
    with pytest.raises(g.Blocked, match="budget logical size"):
        g.read_budget(a.fd, a.campaign, a.manifest, profile=profile)
    assert state.reads == [] and calls == []


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("case", ["complete", "valid-prefix", "oversize-valid-prefix"])
def test_budget_read_requires_complete_captured_size(budget_memory_io, boot_records, monkeypatch, profile, case):
    state, a = budget_memory_io
    _, record, _ = boot_records
    a.profile = a.campaign["profile"] = record["campaign"]["profile"] = profile
    record["charged_bytes"] = g.budget_bounds(profile)[1] + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE
    monkeypatch.setattr(g, "STATE_LIMIT", 2048)
    monkeypatch.setattr(g, "COMPILER_BUDGET_STATE_LIMIT", 4096)
    limit, _ = g.budget_bounds(profile)
    prefix = g.encode_budget(record, profile)
    assert len(prefix) < limit
    # Prefix itself is valid budget JSON; unread suffix, not JSON parse failure.
    assert g.decode_budget(prefix, profile) == record
    assert g.validate_budget(record, a.campaign, a.manifest, profile=profile) == {}
    if case == "complete":
        state.raw = prefix
    elif case == "valid-prefix":
        state.raw, state.read_prefix = prefix + b" unread trailing data", prefix
    else:
        state.raw, state.read_prefix = prefix + b" " * (limit + 1 - len(prefix)), prefix
    state.stat.st_size = len(state.raw)
    calls = []
    original = g.json.loads
    def decode(*args, **kwargs):
        calls.append("decode")
        return original(*args, **kwargs)
    monkeypatch.setattr(g.json, "loads", decode)
    if case == "complete":
        persisted, token = g.read_budget(a.fd, a.campaign, a.manifest, profile=profile)
        assert persisted == record and token == g.owned_file(state.stat)
        assert calls == ["decode"]
    else:
        message = "budget logical size" if case == "oversize-valid-prefix" else "incomplete captured-content"
        with pytest.raises(g.Blocked, match=message):
            g.read_budget(a.fd, a.campaign, a.manifest, profile=profile)
        assert calls == []
    assert state.reads == ([] if case == "oversize-valid-prefix" else [limit + 1])


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("site", ["acquire", "existing-acquire", "preflight", "precreation", "commit"])
def test_budget_insufficient_funding_never_creates_or_writes(budget_memory_io, monkeypatch, profile, site):
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    charge = g.budget_bounds(profile)[1]
    if site == "acquire":
        a.cap = g.SLOT_CHARGE + charge + g.CAMPAIGN_CHARGE - 1
        operation = lambda: g.Authority.acquire(a, 100)
    elif site in ("existing-acquire", "preflight"):
        a.budget = g.Budget(a, 11)
        a.cap = g.SLOT_CHARGE + charge - 1
        a.owned_slot = None
        operation = (lambda: g.Authority.acquire(a, 100)) if site == "existing-acquire" else (
            lambda: g.Authority.debit(a, {}, {}))
    elif site == "precreation":
        monkeypatch.setattr(g, "absent", lambda *args: None)
        a.cap = g.SLOT_CHARGE + charge + g.CAMPAIGN_CHARGE - 1
        operation = lambda: g.Authority.debit(a, {}, {})
    else:
        b = g.Budget(a, 11)
        a.cap = charge - 1
        operation = lambda: b.commit({})
    with pytest.raises(g.Blocked, match="fund"):
        operation()
    assert state.events == []


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
def test_budget_fake_precreation_exact_funding_and_nonrefundable_growth(budget_memory_io, monkeypatch, profile):
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    charge = g.budget_bounds(profile)[1]
    a.cap = g.SLOT_CHARGE + g.CAMPAIGN_CHARGE + charge
    monkeypatch.setattr(g, "absent", lambda *args: None)
    key = (7, 8, 9)
    assert g.Authority.debit(a, {key: (100, 1)}, {key: (100, 1)}) == a.cap
    assert a.pending_slot == 0 and a.budget.stable == [1, 4, 5]
    assert state.events.count("open") == 2 and state.events.count("write") == 1
    assert state.opens == [
        (g.BUDGET, g.os.O_RDWR | g.os.O_CREAT | g.os.O_EXCL | g.os.O_NOFOLLOW),
        (g.BUDGET, g.os.O_RDONLY | g.os.O_NOFOLLOW),
    ]
    a.cap = a.campaign["disk_cap_bytes"]
    first = a.budget.meter.total
    assert g.Authority.debit(a, {key: (50, 2)}, {}) == first + charge
    assert g.Authority.debit(a, {key: (75, 3)}, {}) == first + 2 * charge + 25
    assert state.events.count("write") == 3


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("site,over", [("before", 0), ("before", 1), ("after", 0), ("after", 1)])
def test_budget_allocation_exact_bound_and_one_over(budget_memory_io, profile, site, over):
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    charge = g.budget_bounds(profile)[1]
    setattr(state, site, charge + over)
    b = g.Budget(a, 11)
    extra = g.SLOT_CHARGE + g.CAMPAIGN_CHARGE
    if over:
        with pytest.raises(g.Blocked, match="allocation"):
            b.commit({}, extra)
        assert state.events.count("write") == (site == "after")
    else:
        assert b.commit({}, extra) == charge + extra
        assert state.events.count("write") == 1
        assert state.reads == [g.budget_bounds(profile)[0] + 1] * 2


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("failure", ["partial", "mismatch", "changed"])
def test_budget_fake_mutation_uncertainty_holds(budget_memory_io, profile, failure):
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    setattr(state, failure, True)
    b = g.Budget(a, 11)
    message = {"partial": "partial budget write", "mismatch": "complete-content readback",
               "changed": "owned budget FD identity changed"}[failure]
    with pytest.raises(g.Blocked, match=message):
        b.commit({}, g.SLOT_CHARGE + g.CAMPAIGN_CHARGE)
    assert state.events.count("write") == 1
    assert a.owned_slot == [1, 2, 3]


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("changed", [False, True])
def test_budget_fake_load_stable_identity_and_terminal_verification(budget_memory_io, monkeypatch, profile, changed):
    from types import SimpleNamespace
    state, a = budget_memory_io
    a.profile = a.campaign["profile"] = profile
    b = g.Budget(a, 11)
    b.commit({}, g.SLOT_CHARGE + g.CAMPAIGN_CHARGE)
    loaded = g.Budget.load(a)
    assert loaded.stable == b.stable and loaded.verify() == b.verify()
    proof = loaded.proof()
    custody = dict(profile=profile, campaign=a.campaign, slot=str(SLOT), pins=[],
                   directory=list(g.identity(state.stat)), owned_slot=g.owned_file(state.stat),
                   operation="run", campaign_token=[], budget=proof)
    monkeypatch.setattr(g, "Pins", lambda entries: SimpleNamespace(check=lambda: None))
    monkeypatch.setattr(g, "read_campaign", lambda *args: (a.campaign, []))
    unlinked = []
    monkeypatch.setattr(g.os, "unlink", lambda name, **kwargs: unlinked.append(name))
    if changed:
        # Same inode, tampered debit after sealed proof: terminal check must reject.
        record = dict(b.record, charged_bytes=b.record["charged_bytes"] + 1)
        state.raw = g.encode_budget(record, profile)
        state.stat.st_size = len(state.raw)
        with pytest.raises(g.Blocked, match="final debit"):
            g.finish_authority(custody, SLOT, 105, profile=profile)
        assert unlinked == []
    else:
        g.finish_authority(custody, SLOT, 105, profile=profile)
        assert unlinked == [g.SLOT_NAME]
    assert all(count == g.budget_bounds(profile)[0] + 1 for count in state.reads)


@pytest.mark.parametrize("profile", ["unknown", None, True, []])
def test_budget_bounds_unknown_expected_profile_rejected(profile):
    with pytest.raises(g.Blocked, match="profile"):
        g.budget_bounds(profile)


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("mutation", ["duplicate-field", "duplicate-identity", "bool", "short-row", "unknown-field"])
def test_budget_codec_and_census_strict_contracts(boot_records, profile, mutation):
    campaign, record, roots = boot_records
    campaign["profile"] = record["campaign"]["profile"] = profile
    record["charged_bytes"] = g.budget_bounds(profile)[1] + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE
    if mutation == "duplicate-field":
        with pytest.raises(g.Blocked, match="duplicate"):
            g.decode_budget(b'{"previous":[],"previous":[]}', profile)
        return
    if mutation == "duplicate-identity":
        record["previous"] = [[7, 8, 9, 10, 11], [7, 8, 9, 10, 11]]
    elif mutation == "bool":
        record["previous"] = [[True, 8, 9, 10, 11]]
    elif mutation == "short-row":
        record["previous"] = [[7, 8, 9, 10]]
    else:
        record["extra"] = 1
    decoded = g.decode_budget(g.encode_budget(record, profile), profile)
    with pytest.raises(g.Blocked):
        g.validate_budget(decoded, campaign, roots, profile=profile)


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
def test_profile_campaign_and_budget_bind_one_shared_projection(boot_records, profile):
    campaign, budget, roots = boot_records
    campaign["profile"] = budget["campaign"]["profile"] = profile
    budget["charged_bytes"] = g.budget_bounds(profile)[1] + g.CAMPAIGN_CHARGE + g.SLOT_CHARGE
    g.validate_campaign(campaign, BOOT, 100, profile=profile)
    assert g.validate_budget(budget, campaign, roots, profile=profile) == {}
    assert set(budget["campaign"]) == {"boot", "origin", "deadline", "disk_cap_bytes", "profile"}


@pytest.mark.parametrize("value", ["missing", "unknown", None, True, []])
@pytest.mark.parametrize("boundary", ["campaign", "budget", "budget-both"])
def test_profile_persisted_missing_unknown_or_wrong_type_has_no_migration(boot_records, value, boundary):
    campaign, budget, roots = boot_records
    targets = ([campaign] if boundary == "campaign" else [budget["campaign"]]
               if boundary == "budget" else [campaign, budget["campaign"]])
    for record in targets:
        if value == "missing":
            del record["profile"]
        else:
            record["profile"] = value
    with pytest.raises(g.Blocked):
        if boundary == "campaign":
            g.validate_campaign(campaign, BOOT, 100)
        else:
            g.validate_budget(budget, campaign, roots)


@pytest.mark.parametrize("selected,persisted", [("default", TRIAL), (TRIAL, "default")])
def test_profile_selected_campaign_and_budget_cannot_substitute(boot_records, selected, persisted):
    campaign, budget, roots = boot_records
    campaign["profile"] = persisted
    budget["campaign"]["profile"] = selected
    with pytest.raises(g.Blocked, match="profile"):
        g.validate_campaign(campaign, BOOT, 100, profile=selected)
    with pytest.raises(g.Blocked, match="campaign"):
        g.validate_budget(budget, campaign, roots)


@pytest.mark.parametrize("selected,persisted", [("default", TRIAL), (TRIAL, "default")])
@pytest.mark.parametrize("operation", ["run", "retire"])
def test_profile_authority_mismatch_precedes_budget_load_and_all_writes(repository, monkeypatch,
                                                                          selected, persisted, operation):
    # Metadata layer only. CLI's fixed paths/env are tested separately above.
    complete_metadata(authority(settings(repository, profile=persisted)))
    campaign = repository / "target" / g.CAMPAIGN
    budget = repository / "target" / g.BUDGET
    original_campaign, original_budget = campaign.read_bytes(), budget.read_bytes()
    calls = []
    def forbidden(*values, **kwargs):
        calls.append("budget-load-or-write")
        raise g.Blocked("forbidden effect")
    monkeypatch.setattr(g.Budget, "load", forbidden)
    monkeypatch.setattr(g.os, "write", forbidden)
    args = settings(repository, profile=selected, campaign_init=False,
                    campaign_retire=operation == "retire")
    with pytest.raises(g.Blocked, match="profile"):
        authority(args, 200)
    assert calls == [] and not args.slot.exists()
    assert campaign.read_bytes() == original_campaign and budget.read_bytes() == original_budget


@pytest.mark.parametrize("operation", ["init", "run", "retire"])
@pytest.mark.parametrize("boundary", ["check", "acquire", "debit", "commit", "release"])
def test_profile_authority_rechecks_at_each_mutation_and_release(repository, monkeypatch, operation, boundary):
    if operation != "init":
        complete_metadata(authority(settings(repository)))
    args = settings(repository, campaign_init=operation == "init", campaign_retire=operation == "retire")
    a = authority(args, 200)
    try:
        if boundary in ("debit", "commit", "release"):
            a.acquire(201)
            a.debit({}, {})
        a.campaign["profile"] = TRIAL
        writes = []
        def forbidden(*values, **kwargs):
            writes.append("write-or-unlink")
            raise g.Blocked("forbidden mutation")
        monkeypatch.setattr(g.os, "write", forbidden)
        monkeypatch.setattr(g.os, "unlink", forbidden)
        with pytest.raises(g.Blocked, match="profile|contents changed"):
            if boundary == "check":
                a.check(202)
            elif boundary == "acquire":
                a.acquire(202)
            elif boundary == "debit":
                a.debit({}, {})
            elif boundary == "commit":
                a.budget.commit({})
            else:
                g.finish_authority(a.custody(), a.slot, 205)
        assert writes == []
        assert a.slot.exists() == (boundary in ("debit", "commit", "release"))
    finally:
        a.close()


@pytest.mark.parametrize("telemetry", [False, True])
def test_profile_private_role_still_obeys_context_first(monkeypatch, telemetry):
    from types import SimpleNamespace
    calls = []
    class NoContext:
        @staticmethod
        def context():
            calls.append("context")
            raise g.Blocked("inherited/embedded context refused")
        def __new__(cls):
            calls.append("forbidden-adapter")
            raise g.Blocked("forbidden effect")
    def parse():
        calls.append("forbidden-parse")
        return SimpleNamespace(_telemetry=telemetry)
    messages = []
    monkeypatch.setattr(g, "Real", NoContext)
    monkeypatch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=parse))
    monkeypatch.setattr(g, "diagnostic", messages.append)
    assert g.main() == 78
    assert calls == ["context"]
    assert messages == ["BLOCKED: execution context refused; exception details omitted"]


@pytest.mark.parametrize("profile", ["default", TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("operation", ["init", "run", "retire"])
def test_profile_real_parent_accepts_matching_worker_role(boot_records, profile, operation):
    campaign, _, _ = boot_records
    campaign["profile"] = profile
    if operation == "init":
        campaign["budget_identity"] = None
    adapter = g.Real()
    adapter.profile, adapter.operation = profile, operation
    adapter.now = lambda: 100
    custody = dict(profile=profile, campaign=campaign, operation=operation,
                   owned_slot=[1, 2, 3], budget=None if operation == "init" else {})
    adapter.slot_pending = True
    adapter.accept({"custody": custody}, acquire=True)
    assert adapter.custody is custody and adapter.acquired and not adapter.slot_pending


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("operation", ["init", "run", "retire"])
@pytest.mark.parametrize("stage", ["status", "acquisition", "final"])
@pytest.mark.parametrize("tamper", ["missing-custody", "missing-campaign", "unknown", "other", "operation"])
def test_profile_worker_frame_tamper_holds_prior_custody_without_ack_or_release(
        boot_records, diagnostic_messages, operation, stage, tamper, profile):
    campaign, _, _ = boot_records
    fake = Fake()
    fake.profile, fake.operation = profile, operation
    campaign["profile"] = profile
    original_read = fake.read
    injected, prior = [], []
    fake.accept = g.Real.accept.__get__(fake, g.Real)
    def read():
        data = original_read()
        custody = dict(profile=profile, campaign=campaign.copy(), operation=operation,
                       owned_slot=[1, 2, 3] if fake.acquiring or fake.acquired else None, budget={})
        data["custody"] = custody
        active = "final" if fake.final else "acquisition" if fake.acquiring else "status"
        if active == stage and not injected:
            injected.append(active)
            prior.append(getattr(fake, "custody", None))
            if tamper == "missing-custody":
                del custody["profile"]
            elif tamper == "missing-campaign":
                del custody["campaign"]["profile"]
            elif tamper in ("unknown", "other"):
                custody["profile"] = custody["campaign"]["profile"] = (
                    "unknown" if tamper == "unknown" else "default")
            else:
                custody["operation"] = "retire" if operation != "retire" else "run"
        return data
    fake.read = read
    limits = g.Limits(1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=profile)
    result = g.run(["mock-only"], Path("/work"), {}, SLOT, limits, fake, 1, operation)
    assert injected == [stage] and result.code == 78 and result.held
    assert getattr(fake, "custody", None) is prior[0]
    assert not any(event[0] == "release" for event in fake.events)
    assert fake.launched == (operation == "run" and stage == "final")
    if stage == "acquisition":
        assert fake.slot_pending and not fake.acquired
    elif stage == "final":
        assert fake.acquired


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("short", [False, True])
def test_profile_twelve_gib_launch_floor_exact_and_one_page_short(diagnostic_messages, short, profile):
    fake = Fake()
    fake.profile = profile
    available = 12 * g.GIB - int(short) * 16384
    fake.host_value = host(free_bytes=66 * (1 << 20), available_estimate_bytes=available)
    result = run(fake, profile=profile, rss=4 * g.GIB, reserve=8 * g.GIB)
    assert result.code == (78 if short else 0) and not result.held
    assert fake.launched is not short
    if short:
        assert "preflight needs >=12 GiB" in result.reason
        assert f"required_bytes={12 * g.GIB}" in result.reason
        assert ">=24 GiB" not in result.reason


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("short", [False, True])
def test_profile_twelve_gib_rechecked_at_acquisition_before_payload(diagnostic_messages, short, profile):
    fake = Fake()
    fake.profile = profile
    fake.free_after_acquire = 12 * g.GIB - int(short) * 16384
    result = run(fake, profile=profile, rss=4 * g.GIB, reserve=8 * g.GIB)
    assert result.code == (78 if short else 0) and not result.held
    assert fake.launched is not short


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("meter", ["reserve", "rss", "disk"])
@pytest.mark.parametrize("over", [False, True])
def test_profile_eight_gib_ongoing_four_gib_rss_and_twenty_gib_disk_boundaries(meter, over, profile):
    limits = g.Limits(1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=profile)
    row = host(free_bytes=66 * (1 << 20),
               available_estimate_bytes=8 * g.GIB - (16384 if over and meter == "reserve" else 0))
    result = g.resource_violation(row, row, row, limits.rss + int(over and meter == "rss"),
                                  limits.disk_growth + int(over and meter == "disk"), limits)
    assert bool(result) is over


@pytest.mark.parametrize("profile", [TRIAL, NATIVE_TRIAL])
@pytest.mark.parametrize("over", [0, 1])
def test_profile_four_gib_aggregate_counts_monitor_worker_helpers_and_payload(diagnostic_messages, over, profile):
    fake = Fake()
    fake.profile = profile
    original_read = fake.read
    totals = []
    def read():
        data = original_read()
        if fake.launched:
            rows = data["processes"]
            rows[0][4] = 4 * g.GIB - sum(row[4] for row in rows[1:]) + over
            totals.append(sum(row[4] for row in rows))
        return data
    fake.read = read
    result = run(fake, profile=profile, rss=4 * g.GIB, reserve=8 * g.GIB)
    assert totals and set(totals) == {4 * g.GIB + over}
    assert fake.launched and result.code == (78 if over else 0) and result.held is bool(over)
    if over:
        assert "aggregate RSS including supervisor/worker/helpers" in result.reason
        assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("profile", ["default", TRIAL])
@pytest.mark.parametrize("field,reason", [("swapout_bytes", "actual Swapouts growth"),
                                         ("swap_used_bytes", "swap-used growth")])
def test_profile_no_swap_growth_allowance_is_added(profile, field, reason):
    limits = g.Limits(1000) if profile == "default" else g.Limits(
        1000, rss=4 * g.GIB, reserve=8 * g.GIB, profile=TRIAL)
    baseline = host()
    assert g.resource_violation(baseline, baseline, baseline, 0, 0, limits) is None
    assert g.resource_violation(host(**{field: 101}), baseline, baseline, 0, 0, limits) == reason


@pytest.mark.parametrize("cf_encoding", [None, "wrong", "0x1F5:0:0"])
def test_profile_omitted_cli_and_request_api_remain_legacy_default(compiler_request, cf_encoding):
    _, env = compiler_request
    if cf_encoding is None:
        del env["__CF_USER_TEXT_ENCODING"]
    else:
        env["__CF_USER_TEXT_ENCODING"] = cf_encoding
    options = compiler_options()[6:]  # omit profile and both explicit memory options
    args = g.parser().parse_args(options + ["--", "arbitrary-reviewed-default-command"])
    assert args.profile == "default" and args.rss_bytes == 8 * g.GIB and args.reserve_bytes == 16 * g.GIB
    del args.profile
    assert g.validate_request(args, env, 100) == g.Limits(1000)


@pytest.mark.parametrize("worker", [False, True])
def test_profile_parent_and_worker_run_require_cooperative_flag(monkeypatch, compiler_request, worker):
    args, env = compiler_request
    args.cooperative_build, args._telemetry = False, worker
    if worker:
        args.argv = []
    rejected_request_main(monkeypatch, args, env)


@pytest.mark.parametrize("profile", ["default", TRIAL])
def test_profile_metadata_lifecycle_keeps_binding_across_init_run_retire(repository, profile):
    complete_metadata(authority(settings(repository, profile=profile)))
    path = repository / "target" / g.CAMPAIGN
    original = path.read_bytes()
    campaign = g.decode_state(original)
    assert campaign["profile"] == profile
    complete_metadata(authority(settings(repository, profile=profile, campaign_init=False), 200), 205)
    assert path.read_bytes() == original
    budget = g.decode_state((repository / "target" / g.BUDGET).read_bytes())
    assert budget["campaign"]["profile"] == profile
    complete_metadata(authority(settings(repository, profile=profile, campaign_init=False,
                                         campaign_retire=True), 210), 215)
    assert not path.exists() and not (repository / "target" / g.BUDGET).exists()
    assert not (repository / "target" / g.SLOT_NAME).exists()


@pytest.mark.parametrize("tamper", ["missing-custody", "unknown", "other", "missing-campaign", "campaign-other"])
@pytest.mark.parametrize("commit", [False, True])
def test_profile_release_tampered_custody_refused_before_metadata_or_unlink(repository, monkeypatch,
                                                                         tamper, commit):
    a = authority(settings(repository))
    try:
        a.acquire(101)
        a.debit({}, {})
        custody = a.custody()
        custody["campaign"] = custody["campaign"].copy()
        if tamper == "missing-custody":
            del custody["profile"]
        elif tamper in ("unknown", "other"):
            custody["profile"] = "unknown" if tamper == "unknown" else TRIAL
        elif tamper == "missing-campaign":
            del custody["campaign"]["profile"]
        else:
            custody["campaign"]["profile"] = TRIAL
        calls = []
        def forbidden(*values, **kwargs):
            calls.append("metadata-or-unlink")
            raise g.Blocked("forbidden effect")
        monkeypatch.setattr(g.os, "open", forbidden)
        monkeypatch.setattr(g.os, "unlink", forbidden)
        with pytest.raises(g.Blocked, match="profile"):
            g.finish_authority(custody, a.slot, 105, commit=commit)
        assert calls == [] and a.slot.exists()
    finally:
        a.close()


class DelayedAcquisition(Fake):
    """Independent fake channels; authority may stall while observer stays live."""
    def __init__(self, delay=3, delayed_phase="A"):
        super().__init__()
        self.delay = delay
        self.delayed_phase = delayed_phase
        self.delayed_sent = None
        self.delayed_requests = 0
        self.delayed_completed = None
        self.final_completed = None
        self.authority_requests = []
        self.acquisition_sent = None
        self.acquisition_requests = 0
        self.observations = []

    def request(self, final, acquire=False):
        super().request(final, acquire)
        self.authority_requests.append((self.time, self.authority_sequence, final, acquire))
        if acquire:
            self.acquisition_requests += 1
            self.acquisition_sent = self.time
        selected = (acquire if self.delayed_phase == "A" else final
                    if self.delayed_phase == "F" else self.acquired and not final)
        if selected and self.delayed_sent is None:
            self.delayed_sent = self.time
            self.delayed_sequence = self.authority_sequence
            self.delayed_requests += 1

    def read(self):
        if (self.delayed_sent is not None
                and self.authority_sequence == self.delayed_sequence
                and self.time < self.delayed_sent + self.delay):
            return None
        data = super().read()
        if data is not None and self.final:
            self.final_completed = self.time
        if (self.delayed_sent is not None
                and self.authority_sequence == self.delayed_sequence):
            self.delayed_completed = self.time
        return data

    def observe_read(self):
        data = super().observe_read()
        self.observations.append((self.time, data["sequence"], data["sealed"]))
        return data


def test_delayed_acquisition_keeps_fresh_independent_observations_and_single_issuance():
    fake = DelayedAcquisition()
    result = run(fake, seconds=10)
    assert result.code == 0 and not result.held and fake.launched
    assert fake.acquisition_requests == 1
    during = [at for at, _, _ in fake.observations
              if fake.acquisition_sent <= at <= fake.acquisition_sent + fake.delay]
    assert during and max(during) - min(during) > 2
    assert all(b - a <= 0.1 + 1e-9 for a, b in zip(during, during[1:]))
    assert fake.observations[-1][2] is True
    assert fake.observer_sequence == len(fake.observations)


@pytest.mark.parametrize("bound", ["command", "campaign", "cancel"])
def test_pending_acquisition_obeys_original_deadlines_and_cancellation_without_payload(bound):
    fake = DelayedAcquisition(delay=1000)
    changes = {"seconds": 0.5} if bound == "command" else {"campaign_deadline": 115}
    original = fake.observe_read
    def observe():
        data = original()
        if bound == "cancel" and fake.acquisition_sent is not None and fake.time >= 100.25:
            fake.interrupted = True
        return data
    fake.observe_read = observe
    result = run(fake, **changes)
    cutoff = 100.5 if bound == "command" else 106 if bound == "campaign" else 100.35
    terms = [event[3] for event in fake.events if event[0] == "signal" and event[2] == signal.SIGTERM]
    assert result.code == 78 and result.held and not fake.launched
    assert fake.acquisition_requests == 1 and fake.slot_pending and not fake.acquired
    assert terms and max(terms) <= cutoff + 1e-9
    assert fake.reaped == {"worker", "observer"}
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("fault", ["rss-worker", "rss-observer", "rss-helper", "headroom",
    "swap-used", "swapouts", "swap-regression", "used-regression", "host-identity",
    "monitor-birth", "worker-birth", "observer-birth", "worker-zombie", "observer-zombie",
    "worker-parent", "observer-parent", "escape", "helper-ownership"])
def test_mid_acquisition_whole_family_and_host_failures_hold_without_payload(fault):
    fake = DelayedAcquisition()
    original = fake.observe_read
    injected = []
    def observe():
        data = original()
        if fake.acquisition_sent is None or fake.time < fake.acquisition_sent + 0.15:
            return data
        injected.append(fault)
        if fault.startswith("rss-"):
            pid = {"rss-worker": 2, "rss-observer": 6, "rss-helper": 5}[fault]
            next(row for row in data["processes"] if row[0] == pid)[4] = 8 * g.GIB
        elif fault == "headroom":
            data["host"] = host(free_bytes=15 * g.GIB)
        elif fault in ("swap-used", "swapouts", "swap-regression", "used-regression"):
            field = "swap_used_bytes" if fault in ("swap-used", "used-regression") else "swapout_bytes"
            data["host"][field] = 99 if "regression" in fault else 101
        elif fault == "host-identity":
            data["host"] = host(physical_total_bytes=32 * g.GIB)
        elif fault.endswith("-birth"):
            pid = {"monitor-birth": 1, "worker-birth": 2, "observer-birth": 6}[fault]
            next(row for row in data["processes"] if row[0] == pid)[1] = None
        elif fault.endswith("-zombie") or fault.endswith("-parent"):
            pid = 2 if fault.startswith("worker-") else 6
            next(row for row in data["processes"] if row[0] == pid)[5 if fault.endswith("-zombie") else 2] = (
                True if fault.endswith("-zombie") else 999)
        elif fault == "escape":
            next(row for row in data["processes"] if row[0] == 2)[3] = 99
        else:
            data["processes"].append([8, 8.0, 2, 6, 1000, False])
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and not fake.acquired and fake.acquisition_requests == 1
    assert fake.time < 103
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("fault", ["hang", "eof", "partial", "stale", "future", "sequence",
                                  "bool-sequence", "role", "custody", "wrong-seal", "missing-host"])
def test_pending_acquisition_observer_wire_and_freshness_fail_closed(monkeypatch, fault):
    fake = DelayedAcquisition(delay=20)
    original = fake.observe_read
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.read_fd, endpoint.buffer = 123, bytearray()
    monkeypatch.setattr(g.os, "read", lambda fd, size: b"" if fault == "eof" else b'{"host":')
    injected = []
    def observe():
        if fake.acquisition_sent is None:
            return original()
        injected.append(fault)
        if fault == "hang":
            return None
        if fault in ("eof", "partial"):
            return endpoint.read()
        data = original()
        if fault == "stale":
            data["observed_at"] -= 3
        elif fault == "future":
            data["observed_at"] += 1
        elif fault == "sequence":
            data["sequence"] += 1
        elif fault == "bool-sequence":
            data["sequence"] = True
        elif fault == "role":
            data["role"] = "authority"
        elif fault == "wrong-seal":
            data["sealed"] = True
        elif fault == "missing-host":
            del data["host"]
        else:
            data["custody"] = {}
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and fake.acquisition_requests == 1 and fake.time < 105
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("fault", ["census", "disk", "custody", "persistence"])
def test_incomplete_acquisition_never_becomes_launch_or_release_authority(fault):
    fake = Fake()
    original = fake.read
    def read():
        data = original()
        if fake.acquiring:
            if fault == "census":
                data["census_complete"] = False
            elif fault == "disk":
                data["disk_growth"] = None
            elif fault == "custody":
                del data["custody"]
            else:
                data["custody"]["budget"] = None
        return data
    fake.read = read
    result = run(fake)
    assert result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and not fake.acquired
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("role", ["authority", "observer"])
def test_either_worker_final_seal_failure_retains_slot(role):
    fake = Fake()
    original = fake.read if role == "authority" else fake.observe_read
    def read():
        data = original()
        final = fake.final if role == "authority" else fake.observer_final
        if final:
            data["sealed"] = False
        return data
    if role == "authority":
        fake.read = read
    else:
        fake.observe_read = read
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    assert fake.reaped == {"worker", "observer", "payload"}
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("expired_helper", [False, True])
def test_both_worker_reaps_share_one_budget_and_broadcast_signals_before_waits(expired_helper):
    fake = TransientCrossChannelHelper(proof="missing", delay=1000) if expired_helper else Fake()
    if expired_helper:
        original_read = fake.read
        def read():
            data = original_read()
            if data is not None and not fake.acquiring and not fake.acquired and not fake.launched:
                data["processes"].append([8, 8.0, 2, 2, 1000, False])
            return data
        fake.read = read
    original_sleep = fake.sleep
    def sleep(seconds):
        assert seconds > 0  # an expired helper must not freeze terminal grace/reap
        original_sleep(seconds)
    fake.sleep = sleep
    fake.reap_hang = True
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), fake, 1, "init")
    assert result.code == 78 and result.held
    terms = [event for event in fake.events if event[0] == "signal" and event[2] == signal.SIGTERM]
    kills = [i for i, event in enumerate(fake.events) if event[0] == "signal" and event[2] == signal.SIGKILL]
    reaps = [i for i, event in enumerate(fake.events) if event[0] == "reap"]
    assert {event[1] for event in terms} == {"worker", "observer"}
    assert max(event[3] for event in terms) == min(event[3] for event in terms)
    assert max(kills) < min(reaps)
    assert fake.time <= terms[0][3] + 2 + 1e-9
    if expired_helper:
        assert "worker helper reap proof deadline" in result.reason
        assert fake.time >= terms[0][3] + 2 - 1e-9
        assert fake.acquisition_requests == 1 and fake.slot_pending and not fake.acquired
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("role", ["worker", "observer"])
def test_either_worker_unexpected_reap_holds_after_both_seals(role):
    fake = Fake()
    original = fake.reap
    def reap(name):
        code = original(name)
        return 78 if name == role else code
    fake.reap = reap
    result = run(fake)
    assert fake.observer_final and fake.final
    assert result.code == 78 and result.held and fake.acquired
    assert fake.reaped == {"worker", "observer", "payload"}
    assert not any(event[0] == "release" for event in fake.events)


def test_real_worker_roles_register_distinct_children_and_distinct_wires_before_go(monkeypatch):
    adapter = g.Real()
    adapter.worker_env = {"FROZEN": "parent"}
    monkeypatch.setattr(g.Real, "context", staticmethod(lambda: None))
    pairs = iter((20 + 2 * i, 21 + 2 * i) for i in range(8))
    pids = iter([2, 6])
    monkeypatch.setattr(g.os, "pipe", lambda: next(pairs))
    monkeypatch.setattr(g.os, "fork", lambda: next(pids))
    monkeypatch.setattr(g.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(g.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(g.signal, "pthread_sigmask", lambda *args: set())
    monkeypatch.setattr(g.os, "close", lambda fd: None)
    assert adapter.worker([]) == 2 and adapter.observer([]) == 6
    assert set(adapter.children) == set(adapter.wires) == {"worker", "observer"}
    assert adapter.wires["worker"] is not adapter.wires["observer"]
    assert adapter.children["worker"] is not adapter.children["observer"]
    assert all(child.go_sent == 0 and child.phase == "setup" for child in adapter.children.values())
    with pytest.raises(g.Blocked, match="duplicate"):
        adapter.observer([])


@pytest.mark.parametrize("command", [b"A\n", b"S\n"])
def test_observer_endpoint_cannot_acquire_or_construct_authority(monkeypatch, command):
    from io import BytesIO
    from types import SimpleNamespace
    forbidden, samples = [], []
    def deny(*args, **kwargs):
        forbidden.append("authority")
        raise AssertionError("observer authority access")
    monkeypatch.setattr(g.sys, "platform", "darwin")
    monkeypatch.setitem(g.sys.modules, "psutil", SimpleNamespace())
    for name in ("Authority", "Disk", "boot_identity", "finish_authority"):
        monkeypatch.setattr(g, name, deny)
    monkeypatch.setattr(g.Real, "spawn", deny)
    monkeypatch.setattr(g.ProcessObservations, "host", lambda self: host())
    def processes(self):
        samples.append("processes")
        return [], []
    monkeypatch.setattr(g.ProcessObservations, "processes", processes)
    monkeypatch.setattr(g.time, "monotonic", lambda: 100)
    output = BytesIO()
    monkeypatch.setattr(g.sys, "stdin", SimpleNamespace(buffer=BytesIO(command)))
    monkeypatch.setattr(g.sys, "stdout", SimpleNamespace(buffer=output))
    assert g.telemetry_worker(SimpleNamespace(_observer=True)) == 78  # next EOF is refused
    assert forbidden == []
    assert samples == ([] if command == b"A\n" else ["processes"])
    if command == b"S\n":
        data = g.json.loads(output.getvalue())
        assert data["role"] == "observer" and data["sequence"] == 1
        assert not {"custody", "slot", "disk_growth"} & set(data)
    else:
        assert output.getvalue() == b""


def test_observer_sample_rejects_acquire_before_any_provider(monkeypatch):
    calls = []
    def provider(self):
        calls.append("provider")
        raise AssertionError("must reject before observation")
    monkeypatch.setattr(g.ProcessObservations, "host", provider)
    monkeypatch.setattr(g.ProcessObservations, "processes", provider)
    with pytest.raises(g.Blocked, match="cannot acquire"):
        g.ProcessObservations.__new__(g.ProcessObservations).sample(False, True)
    assert calls == []


@pytest.mark.parametrize("operation", ["run", "init", "retire"])
def test_profile_observer_dispatch_cannot_construct_parent_or_other_observers(monkeypatch, compiler_request, operation):
    from types import SimpleNamespace
    args, env = compiler_request
    args._observer, args._telemetry, args.argv = True, False, []
    args.campaign_init, args.campaign_retire = operation == "init", operation == "retire"
    calls, forbidden = [], []
    class NoParent:
        @staticmethod
        def context():
            calls.append("context")
        def __new__(cls):
            forbidden.append("parent")
            raise AssertionError("observer cannot create parent")
    def dispatch(options):
        calls.append(("observer", options._observer, options.argv))
        return 0
    monkeypatch.setattr(g, "Real", NoParent)
    monkeypatch.setattr(g, "telemetry_worker", dispatch)
    monkeypatch.setattr(g, "parser", lambda: SimpleNamespace(parse_args=lambda: args))
    monkeypatch.setattr(g.os, "environ", env)
    monkeypatch.setattr(g.time, "monotonic", lambda: 100)
    assert g.main() == 0
    assert forbidden == [] and calls == ["context", ("observer", True, [])]


def test_partial_observer_parent_setup_keeps_both_registered_children_in_common_cleanup(monkeypatch):
    adapter, clock = g.Real(), Fake()
    adapter.now, adapter.sleep = clock.now, clock.sleep
    monkeypatch.setattr(g.Real, "context", staticmethod(lambda: None))
    pairs = iter((20 + 2 * i, 21 + 2 * i) for i in range(8))
    pids = iter([2, 6])
    monkeypatch.setattr(g.os, "pipe", lambda: next(pairs))
    monkeypatch.setattr(g.os, "fork", lambda: next(pids))
    monkeypatch.setattr(g.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(g.os, "set_inheritable", lambda *args: None)
    monkeypatch.setattr(g.signal, "pthread_sigmask", lambda *args: set())
    closed, events = [], []
    def close(fd):
        closed.append(fd)
        if fd == 29:
            raise OSError("partial observer descriptor retirement")
    monkeypatch.setattr(g.os, "close", close)
    monkeypatch.setattr(g.os, "kill", lambda pid, sig: events.append(("signal", pid, sig, clock.now())))
    monkeypatch.setattr(g.os, "waitpid", lambda pid, flags: (events.append(("reap", pid, clock.now())),
                                                           (pid, signal.SIGTERM))[1])
    assert adapter.worker([]) == 2
    with pytest.raises(OSError):
        adapter.observer([])
    def startup(allow_go=True):
        raise g.Blocked("partial observer startup")
    adapter.startup_step = startup
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000), adapter, 1)
    assert result.code == 78 and result.held
    assert set(adapter.children) == {"worker", "observer"}
    assert adapter.children["observer"].phase == "failed"
    signals = [event for event in events if event[0] == "signal"]
    assert {(event[1], event[2]) for event in signals} == {
        (2, signal.SIGTERM), (6, signal.SIGTERM), (2, signal.SIGKILL), (6, signal.SIGKILL)}
    assert all(event[0] == "signal" for event in events[:4])
    assert len(closed) == len(set(closed))


def test_unqualified_helper_identity_loss_during_acquisition_holds():
    fake = DelayedAcquisition()
    original = fake.observe_read
    seen = []
    def observe():
        data = original()
        if fake.acquisition_sent is not None and fake.time < fake.acquisition_sent + 0.15:
            data["processes"].append([8, 8.0, 2, 2, 1000, False])
            seen.append(8)
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert seen and result.code == 78 and result.held and not fake.launched
    assert "unresolved worker helper reap proof" in result.reason
    assert fake.slot_pending and fake.acquisition_requests == 1
    assert not any(event[0] == "release" for event in fake.events)


def test_authority_partial_frames_cannot_renew_absolute_acquisition_deadline(monkeypatch):
    fake = DelayedAcquisition(delay=1000)
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.read_fd, endpoint.buffer = 123, bytearray()
    monkeypatch.setattr(g.os, "read", lambda fd, size: b'{"custody":')
    original = fake.read
    def read():
        return endpoint.read() if fake.acquiring else original()
    fake.read = read
    result = run(fake, seconds=0.5)
    assert result.code == 78 and result.held and not fake.launched
    assert fake.acquisition_requests == 1 and fake.slot_pending and not fake.acquired
    assert fake.observations and fake.time <= 102.5 + 1e-9
    assert endpoint.buffer and not any(event[0] == "release" for event in fake.events)


def test_authority_endpoint_refuses_second_acquisition_without_retry(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    calls = []
    class Probes:
        def sample(self, sealed, acquire=False):
            calls.append((sealed, acquire))
            return {"sealed": sealed}
    monkeypatch.setattr(g, "DarwinProbes", lambda args: Probes())
    monkeypatch.setattr(g.sys, "stdin", SimpleNamespace(buffer=BytesIO(b"A\nA\n")))
    output = BytesIO()
    monkeypatch.setattr(g.sys, "stdout", SimpleNamespace(buffer=output))
    assert g.telemetry_worker(object()) == 78
    assert calls == [(False, True)]
    assert g.json.loads(output.getvalue()) == {"sealed": False, "sequence": 1, "role": "authority"}


def test_parent_acquisition_marker_refuses_second_issuance():
    adapter = g.Real()
    adapter.acquire(SLOT)
    with pytest.raises(g.Blocked, match="only once"):
        adapter.acquire(SLOT)
    assert adapter.slot_pending and not adapter.acquired


@pytest.mark.parametrize("role", ["authority", "observer"])
def test_final_seal_stalls_never_extend_original_campaign_cleanup_reserve(role):
    fake = Fake()
    fake.running = True
    original = fake.read if role == "authority" else fake.observe_read
    def read():
        final = fake.final if role == "authority" else fake.observer_final
        if final:
            return None
        return original()
    if role == "authority":
        fake.read = read
    else:
        fake.observe_read = read
    result = run(fake, campaign_deadline=115)
    assert result.code == 78 and result.held and fake.acquired
    assert fake.time <= 115
    assert fake.killed == {"worker", "observer", "payload"}
    assert fake.reaped == {"worker", "observer", "payload"}
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("role", ["authority", "observer"])
def test_sealed_worker_cannot_claim_other_workers_reaped_helper(role):
    fake = Fake()
    original = fake.read if role == "authority" else fake.observe_read
    def read():
        data = original()
        final = fake.final if role == "authority" else fake.observer_final
        if final:
            helper, leader = (5, 6) if role == "authority" else (3, 2)
            data["processes"].append([helper, float(helper), leader, leader, 3000, False])
            data["reaped_helpers"] = [[helper, float(helper)]]
        return data
    if role == "authority":
        fake.read = read
    else:
        fake.observe_read = read
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    assert "unqualified reaped-helper identity" in result.reason
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("zombie", [False, True])
def test_observer_unreaped_descendant_prevents_terminal_release(zombie):
    fake = Fake()
    original = fake.observe_read
    def observe():
        data = original()
        if fake.observer_final:
            data["processes"].append([8, 8.0, 6, 6, 1000, zombie])
        return data
    fake.observe_read = observe
    # Fault begins at final seal; neither live nor zombie is qualified reaping.
    authority_read = fake.read
    def read():
        data = authority_read()
        if fake.final:
            data["processes"].append([8, 8.0, 6, 6, 1000, zombie])
        return data
    fake.read = read
    result = run(fake)
    assert result.code == 78 and result.held and fake.acquired
    assert "unresolved worker helper reap proof" in result.reason
    assert fake.reaped == {"worker", "observer", "payload"}
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("observer", [(2, 2.0), (2, 99.0), (1, 1.0)])
def test_family_roles_cannot_alias_registered_leader_pid(observer):
    with pytest.raises(g.Blocked, match="distinct birth-qualified PIDs"):
        g.Family((1, 1.0), (2, 2.0), observer=observer)


def test_both_worker_roles_receive_frozen_environment_and_original_options(monkeypatch):
    adapter = g.Real()
    frozen = dict(g.COMPILER_ENV)
    adapter.worker_env = frozen
    monkeypatch.setattr(g.os, "environ", {"UNREVIEWED_LATER_VALUE": "not forwarded"})
    calls = []
    def spawn(name, argv, cwd, env, ipc=False):
        calls.append((name, argv, cwd, env, ipc))
        return 2 if name == "worker" else 6
    monkeypatch.setattr(adapter, "spawn", spawn)
    options = compiler_options()
    assert adapter.worker(options) == 2 and adapter.observer(options) == 6
    for name, argv, cwd, env, ipc in calls:
        assert argv[:2] == [g.sys.executable, "-B"]
        assert argv[3] == ("--_telemetry" if name == "worker" else "--_observer")
        assert argv[4:] == options and cwd is None and ipc is True
        assert env == frozen and env is not frozen
        assert len(env) == 12 and env["__CF_USER_TEXT_ENCODING"] == "0x1F5:0:0"
    assert calls[0][1][2] == calls[1][1][2]
    assert calls[0][3] is not calls[1][3]


class TransientCrossChannelHelper(DelayedAcquisition):
    """Observer sees a real-shaped transient; its owner's proof arrives later."""
    def __init__(self, proof="late", delay=3):
        super().__init__(delay)
        self.proof = proof
        self.helper_birth = 8.1 if proof == "replay" else 8.0
        self.helper_frames = []
        self.proof_frames = []

    def observe_read(self):
        data = super().observe_read()
        if self.acquisition_sent is not None:
            elapsed = self.time - self.acquisition_sent
            present = 0.05 <= elapsed < 0.25
            if present:
                data["processes"].append([8, self.helper_birth, 2, 2, 1000, False])
            self.helper_frames.append((self.time, present))
            if self.proof == "wrong-owner" and elapsed >= 0.35:
                data["reaped_helpers"].append([8, self.helper_birth])
                self.proof_frames.append(("observer", self.time, self.helper_birth))
        return data

    def read(self):
        data = super().read()
        if data is None:
            return None
        if self.proof == "replay" and not self.acquiring and not self.acquired and not self.launched:
            data["processes"].append([8, 8.0, 2, 2, 1000, False])
            data["reaped_helpers"].append([8, 8.0])
        if self.acquiring and self.proof in ("late", "replay"):
            birth = 8.0 if self.proof == "replay" else self.helper_birth
            # No helper row: its already-qualified owner reports the precise
            # birth explicitly. Absence alone supplies no proof.
            data["reaped_helpers"].append([8, birth])
            self.proof_frames.append(("authority", self.time, birth))
        return data


def test_cross_channel_helper_absence_waits_for_late_owner_proof_without_false_hold():
    fake = TransientCrossChannelHelper()
    result = run(fake, seconds=10)
    assert result.code == 0 and not result.held and fake.launched
    assert fake.acquisition_requests == 1
    seen = [at for at, present in fake.helper_frames if present]
    absent = [at for at, present in fake.helper_frames if not present
              and at < fake.acquisition_sent + fake.delay]
    assert seen and absent and min(absent) > min(seen)
    assert max(absent) - min(seen) > 2
    assert len(fake.proof_frames) == 1
    role, at, birth = fake.proof_frames[0]
    assert role == "authority" and birth == 8.0
    assert fake.acquisition_sent + fake.delay <= at <= fake.acquisition_sent + fake.delay + 0.05 + 1e-9
    assert any(event[0] == "release" for event in fake.events)


def test_same_observer_frame_applies_owner_proof_before_absence_reconciliation():
    fake = Fake()
    authority_read, observer_read = fake.read, fake.observe_read
    delivered = []
    def read():
        data = authority_read()
        if fake.acquiring:
            data["processes"].append([8, 8.0, 6, 6, 1000, False])
        return data
    def observe():
        data = observer_read()
        if fake.acquiring:
            assert all(row[0] != 8 for row in data["processes"])
            data["reaped_helpers"].append([8, 8.0])
            delivered.append((data["sequence"], data["reaped_helpers"].copy()))
        return data
    fake.read, fake.observe_read = read, observe
    result = run(fake)
    assert delivered and result.code == 0 and not result.held and fake.launched
    assert fake.reaped == {"worker", "observer", "payload"}


@pytest.mark.parametrize("bound", ["response", "absolute-command"])
def test_pending_helper_missing_proof_at_existing_bound_never_launches_or_releases(bound):
    fake = TransientCrossChannelHelper(proof="missing", delay=3 if bound == "response" else 1000)
    result = run(fake, seconds=10 if bound == "response" else 0.5)
    assert result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and not fake.acquired and fake.acquisition_requests == 1
    assert any(present for _, present in fake.helper_frames)
    assert any(not present for _, present in fake.helper_frames)
    assert fake.proof_frames == []
    if bound == "response":
        assert "unresolved worker helper reap proof" in result.reason
        assert fake.time >= fake.acquisition_sent + fake.delay
    else:
        terms = [event[3] for event in fake.events
                 if event[0] == "signal" and event[2] == signal.SIGTERM]
        assert terms and max(terms) <= 100.5 + 1e-9
    assert not any(event[0] in ("launch", "release") for event in fake.events)


@pytest.mark.parametrize("proof", ["wrong-owner", "replay"])
def test_wrong_owner_or_old_birth_replay_cannot_discharge_pending_helper(proof):
    fake = TransientCrossChannelHelper(proof=proof)
    result = run(fake, seconds=10)
    assert result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and not fake.acquired and fake.acquisition_requests == 1
    assert fake.proof_frames
    if proof == "wrong-owner":
        assert "unqualified reaped-helper identity" in result.reason
        assert fake.proof_frames[0][0] == "observer"
    else:
        assert fake.helper_birth == 8.1 and fake.proof_frames[0][2] == 8.0
        assert "unresolved worker helper reap proof" in result.reason
    assert not any(event[0] == "release" for event in fake.events)


def test_absent_pending_helper_keeps_observed_rss_until_explicit_owner_reap():
    fake = TransientCrossChannelHelper()
    original = fake.observe_read
    injected = []
    def observe():
        data = original()
        if (fake.acquisition_sent is not None
                and fake.time >= fake.acquisition_sent + 0.25):
            assert all(row[0] != 8 for row in data["processes"])
            next(row for row in data["processes"] if row[0] == 2)[4] += 1001
            injected.append(sum(row[4] for row in data["processes"]))
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10, rss=10000)
    assert injected and set(injected) == {9001}  # alone below cap; pending 1000 pushes over
    assert result.code == 78 and result.held and not fake.launched
    assert "aggregate RSS" in result.reason
    assert fake.slot_pending and not fake.acquired and fake.time < 103
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("fault", ["live", "zombie", "disappearing"])
def test_unknown_direct_supervisor_child_custody_never_permits_release(fault):
    fake = DelayedAcquisition()
    original = fake.observe_read
    injected = []
    def observe():
        data = original()
        transient = (fake.acquisition_sent is not None
                     and 0.15 <= fake.time - fake.acquisition_sent < 0.25)
        # The disappearing schedule plans absence on later frames. First sight
        # must already cause terminal HOLD, not wait to infer reap from absence.
        if (fault == "disappearing" and transient
                or fault != "disappearing" and fake.observer_final):
            data["processes"].append([9, 9.0, 1, 9, 1000, fault == "zombie"])
            injected.append((fake.time, fault))
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held
    assert "unknown direct supervisor-child custody" in result.reason
    assert fake.launched is (fault != "disappearing")
    assert fake.acquired is (fault != "disappearing")
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("fault", ["rss", "reserve", "swap-used", "swapouts",
                                  "used-regression", "swapout-regression"])
def test_sticky_launch_floor_denial_never_masks_new_hard_acquisition_violation(fault):
    fake = DelayedAcquisition(delay=1000)
    fake.host_value = host(free_bytes=23 * g.GIB)  # below launch floor, above hard reserve
    original = fake.observe_read
    injected = []
    def observe():
        data = original()
        if (fake.acquisition_sent is None
                or fake.time < fake.acquisition_sent + 0.15):
            return data
        injected.append(fault)
        if fault == "rss":
            next(row for row in data["processes"] if row[0] == 2)[4] = 8 * g.GIB
        elif fault == "reserve":
            data["host"] = host(free_bytes=15 * g.GIB)
        else:
            field = "swap_used_bytes" if fault in ("swap-used", "used-regression") else "swapout_bytes"
            data["host"][field] = 99 if "regression" in fault else 101
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held and not fake.launched
    assert fake.slot_pending and not fake.acquired and fake.acquisition_requests == 1
    terms = [event[3] for event in fake.events
             if event[0] == "signal" and event[2] == signal.SIGTERM]
    assert terms and max(terms) <= fake.acquisition_sent + 0.3 + 1e-9
    assert not any(event[0] == "release" for event in fake.events)


def test_launch_floor_denial_alone_preserves_qualified_refusal_after_monitored_acquisition():
    fake = DelayedAcquisition()
    fake.host_value = host(free_bytes=23 * g.GIB)
    result = run(fake, seconds=10)
    assert result.code == 78 and not result.held and not fake.launched
    assert "preflight needs >=24 GiB" in result.reason
    assert fake.acquisition_requests == 1 and not fake.acquired and not fake.slot_pending
    assert fake.time >= fake.acquisition_sent + fake.delay
    assert fake.observer_final and fake.final
    assert any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("schedule", ["original", "just-before", "equal", "non-grid", "observer-pending"])
def test_pending_helper_response_bound_is_not_renewed_by_next_acquisition(schedule):
    observer_pending = schedule == "observer-pending"
    fake = TransientCrossChannelHelper(proof="missing" if observer_pending else "late",
                                       delay=1.9 if observer_pending else 3)
    if schedule == "non-grid":
        fake.time = 99.983
    expiry = fake.time + 2
    original = fake.read
    stepped, replies, observer_waits = [], [], []
    def read():
        data = original()
        if data is not None and not fake.acquiring and not fake.acquired and not fake.launched:
            data["processes"].append([8, 8.0, 2, 2, 1000, False])
        if fake.acquiring:
            if data is not None:
                replies.append(fake.time)
            elif schedule in ("just-before", "equal", "non-grid") and not stepped:
                fake.time = expiry if schedule == "equal" else expiry - 0.001
                stepped.append(fake.time)
        return data
    fake.read = read
    original_observe = fake.observe_read
    def observe():
        if observer_pending and fake.acquisition_sent is not None and fake.time >= 101.75:
            observer_waits.append((fake.time, bool(replies)))
            return None
        return original_observe()
    fake.observe_read = observe
    original_sleep = fake.sleep
    def sleep(seconds):
        assert seconds > 0  # equality must fail closed, not enter a zero-nap loop
        original_sleep(seconds)
    fake.sleep = sleep
    result = run(fake, seconds=10)
    assert result.code == 78 and result.held and not fake.launched
    assert "worker helper reap proof deadline" in result.reason
    assert fake.acquisition_requests == 1 and fake.slot_pending and not fake.acquired
    terms = [event[3] for event in fake.events
             if event[0] == "signal" and event[2] == signal.SIGTERM]
    assert terms and max(terms) <= 102 + 1e-9  # original S response bound, not A's 110
    assert min(terms) >= expiry and max(terms) <= expiry + 1e-9
    if schedule in ("just-before", "equal", "non-grid"):
        assert stepped == [expiry if schedule == "equal" else expiry - 0.001]
    if observer_pending:
        assert replies and replies[0] < expiry
        assert any(after_reply for _, after_reply in observer_waits)
    assert fake.proof_frames == []
    assert not any(event[0] == "release" for event in fake.events)


@pytest.mark.parametrize("phase,operation", [("F", "run"), ("S", "run"), ("F", "init")])
def test_delayed_full_census_keeps_observer_live_and_seals_before_common_reaps(phase, operation):
    fake = DelayedAcquisition(delayed_phase=phase)
    result = g.run([], Path("/work"), {}, SLOT, g.Limits(1000, seconds=10), fake, 1, operation)
    assert result.code == 0 and not result.held
    assert fake.launched is (operation == "run")
    assert fake.delayed_requests == fake.acquisition_requests == 1
    assert sum(sequence == fake.delayed_sequence
               for _, sequence, _, _ in fake.authority_requests) == 1
    assert sum(final for _, _, final, _ in fake.authority_requests) == 1
    during = [at for at, _, sealed in fake.observations
              if fake.delayed_sent <= at < fake.delayed_completed and not sealed]
    assert during and max(during) - min(during) > 2
    assert all(b - a <= 0.1 + 1e-9 for a, b in zip(during, during[1:]))
    seals = [at for at, _, sealed in fake.observations if sealed]
    assert len(seals) == 1 and seals[0] >= fake.final_completed >= fake.delayed_completed
    assert fake.final and fake.observer_final
    release = next(i for i, e in enumerate(fake.events) if e[0] == "release")
    assert all(i < release for i, e in enumerate(fake.events) if e[0] == "reap")
    reaps = [i for i, e in enumerate(fake.events) if e[0] == "reap"]
    kills = [i for i, e in enumerate(fake.events)
             if e[0] == "signal" and e[2] == signal.SIGKILL]
    assert max(kills) < min(reaps)
    assert all(e[2] >= seals[0] for e in fake.events if e[0] == "reap")
    assert fake.reaped == ({"worker", "observer", "payload"} if fake.launched
                           else {"worker", "observer"})


@pytest.mark.parametrize("phase", ["F", "S"])
@pytest.mark.parametrize("boundary", ["dispatch", "response", "accept"])
@pytest.mark.parametrize("late", [0, 0.001])
def test_full_census_absolute_cutoff_rejects_dispatch_response_and_acceptance(monkeypatch, phase,
                                                                           boundary, late):
    fake = DelayedAcquisition(delay=0, delayed_phase=phase)
    injected = []
    original_read, original_observe = fake.read, fake.observe_read
    accepted = []
    original_accept = fake.accept
    def accept(data, acquire=False):
        accepted.append((fake.time, fake.delayed_completed))
        original_accept(data, acquire)
    def diagnostic(message):
        if (boundary == "dispatch" and fake.acquired and not injected
                and message == f"telemetry {'F final-seal' if phase == 'F' else 'S status'} start"):
            fake.time = 110 + late
            injected.append(fake.time)
    def read():
        if boundary == "accept" and fake.delayed_sent is not None and not injected:
            fake.time = 110 - 0.001  # authority reply remains inside its cutoff
        data = original_read()
        if boundary == "response" and fake.delayed_sent is not None and not injected:
            fake.time = 110 + late
            injected.append(fake.time)
        return data
    def observe():
        data = original_observe()
        if (boundary == "accept" and fake.delayed_completed is not None and not injected):
            # Qualifying observation is fresh within its own original request
            # bound, but finishes at/after the authority's absolute cutoff.
            fake.time = 110 + late
            data["observed_at"] = fake.time
            injected.append(fake.time)
        return data
    monkeypatch.setattr(g, "diagnostic", diagnostic)
    fake.read, fake.observe_read = read, observe
    fake.accept = accept
    result = run(fake, seconds=10)
    assert injected == [110 + late] and result.code == 78 and result.held
    assert all(e[2] < 110 for e in fake.events if e[0] == "request")
    assert fake.delayed_requests == (0 if boundary == "dispatch" else 1)
    assert all(at < 110 and completion is None for at, completion in accepted)
    assert fake.time <= 119 and not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
def test_full_census_partial_authority_frames_never_extend_original_cutoff(monkeypatch, phase):
    fake = DelayedAcquisition(delayed_phase=phase)
    fake.reap_hang = True
    endpoint = g.Wire.__new__(g.Wire)
    endpoint.read_fd, endpoint.buffer = 123, bytearray()
    monkeypatch.setattr(g.os, "read", lambda fd, size: b'{"custody":')
    original = fake.read
    def read():
        return endpoint.read() if fake.delayed_sent is not None else original()
    fake.read = read
    result = run(fake, seconds=10)
    assert result.code == 78 and result.held and endpoint.buffer
    assert fake.delayed_requests == 1 and fake.time <= 112 + 1e-9
    terms = [e[3] for e in fake.events if e[0] == "signal" and e[2] == signal.SIGTERM
             and e[1] != "payload"]
    assert terms and max(terms) <= 110 + 1e-9
    assert fake.killed == {"worker", "observer", "payload"}
    assert {e[1] for e in fake.events if e[0] == "reap"} == fake.killed
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
@pytest.mark.parametrize("fault", ["missing", "stale", "expired"])
def test_full_census_pending_observer_failure_retains_custody(phase, fault):
    fake = DelayedAcquisition(delayed_phase=phase)
    original = fake.observe_read
    injected = []
    def observe():
        if fake.delayed_sent is None:
            return original()
        injected.append(fault)
        if fault == "missing":
            return None
        data = original()
        if fault == "stale":
            data["observed_at"] -= 3
        else:
            fake.time = fake.observer_sent + 2
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held and fake.acquired
    assert fake.time < fake.delayed_sent + 5 + 1e-9
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
@pytest.mark.parametrize("fault", ["rss", "reserve", "cancel"])
def test_full_census_hard_stop_preserves_admission_semantics(phase, fault):
    fake = DelayedAcquisition(delay=1000, delayed_phase=phase)
    fake.host_value = host(free_bytes=23 * g.GIB)
    # A sticky prelaunch floor denial reaches F without launching. Post-A S
    # requires launch, so its sub-floor reading must retain ongoing semantics.
    if phase == "S":
        fake.host_value = host()
    original = fake.observe_read
    injected = []
    def observe():
        data = original()
        if fake.delayed_sent is not None:
            data["host"] = host(free_bytes=23 * g.GIB)
            if fake.time >= fake.delayed_sent + 0.15:
                injected.append(fault)
                if fault == "rss":
                    next(row for row in data["processes"] if row[0] == 2)[4] = 8 * g.GIB
                elif fault == "reserve":
                    data["host"] = host(free_bytes=15 * g.GIB)
                else:
                    fake.interrupted = True
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10)
    assert injected and result.code == 78 and result.held and fake.acquired
    assert fake.launched is (phase == "S")
    terms = [e[3] for e in fake.events if e[0] == "signal" and e[2] == signal.SIGTERM
             and e[1] != "payload"]
    assert terms and max(terms) <= fake.delayed_sent + 0.3 + 1e-9
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
@pytest.mark.parametrize("proof", ["qualified", "expired"])
def test_full_census_helper_requires_owner_proof_before_original_bound(phase, proof):
    fake = DelayedAcquisition(delay=3 if proof == "qualified" else 1000, delayed_phase=phase)
    original_read, original_observe = fake.read, fake.observe_read
    seen, proofs = [], []
    def observe():
        data = original_observe()
        if (fake.delayed_sent is not None
                and fake.time < fake.delayed_sent + 0.15):
            data["processes"].append([8, 8.0, 2, 2, 1000, False])
            seen.append(fake.time)
        return data
    def read():
        data = original_read()
        if (data is not None and fake.delayed_sent is not None
                and fake.authority_sequence == fake.delayed_sequence):
            data["reaped_helpers"].append([8, 8.0])
            proofs.append(fake.time)
        return data
    fake.read, fake.observe_read = read, observe
    result = run(fake, seconds=10)
    assert seen and fake.delayed_requests == 1
    assert fake.time <= 112 + 1e-9
    if proof == "qualified":
        assert proofs and result.code == 0 and not result.held
        assert any(e[0] == "release" for e in fake.events)
    else:
        assert proofs == [] and result.code == 78 and result.held
        assert "worker helper reap proof deadline" in result.reason
        assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
def test_full_census_compiler_ongoing_reserve_does_not_reapply_launch_floor(phase):
    fake = DelayedAcquisition(delayed_phase=phase)
    fake.profile = TRIAL
    original = fake.observe_read
    def observe():
        data = original()
        if fake.delayed_sent is not None:
            data["host"] = host(free_bytes=10 * g.GIB)
        return data
    fake.observe_read = observe
    result = run(fake, seconds=10, profile=TRIAL, rss=4 * g.GIB, reserve=8 * g.GIB)
    assert result.code == 0 and not result.held and fake.launched
    assert fake.delayed_completed >= fake.delayed_sent + 3
    assert fake.final and fake.observer_final
    assert any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["F", "S"])
@pytest.mark.parametrize("fault", ["census", "budget", "disk", "regressed-debit",
                                  "missing-seal", "wrong-seal"])
def test_full_census_incomplete_persistence_or_seal_retains_hold(phase, fault):
    fake = DelayedAcquisition(delay=0, delayed_phase=phase)
    fake.growth = 1
    original = fake.read
    def read():
        data = original()
        if fake.delayed_sent is not None:
            if fault == "census":
                data["census_complete"] = False
            elif fault == "budget":
                data["custody"]["budget"] = None
            elif fault == "disk":
                data["disk_growth"] = None
            elif fault == "regressed-debit":
                data["disk_growth"] = 0
            elif fault == "missing-seal":
                del data["sealed"]
            else:
                data["sealed"] = not fake.final
        return data
    fake.read = read
    result = run(fake, seconds=10)
    assert result.code == 78 and result.held and fake.acquired
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase,fault,sticky", [
    ("F", "rss", True), ("F", "reserve", True),
    ("F", "swap-used", True), ("F", "swapouts", True),
    ("S", "rss", False), ("S", "reserve", False),
    ("S", "swap-used", False), ("S", "swapouts", False),
    ("F", "disk", True), ("S", "disk", False),
    ("A", "disk", True), ("A", "rss", True),
])
def test_full_census_post_reply_hard_violation_never_qualifies(phase, fault, sticky):
    fake = DelayedAcquisition(delay=0, delayed_phase=phase)
    if sticky:
        fake.host_value = host(free_bytes=23 * g.GIB)
    limits = g.Limits(1000, seconds=10)
    original_read, original_observe, original_accept = fake.read, fake.observe_read, fake.accept
    injected, accepted = [], []
    def read():
        data = original_read()
        if fault == "disk" and fake.delayed_completed is not None:
            data["disk_growth"] = limits.disk_growth + 1
            injected.append(fake.time)
        return data
    def observe():
        data = original_observe()
        if fault != "disk" and fake.delayed_completed is not None:
            injected.append(fake.time)
            if fault == "rss":
                next(row for row in data["processes"] if row[0] == 2)[4] = limits.rss
            elif fault == "reserve":
                data["host"] = host(free_bytes=limits.reserve - 16384)
            else:
                field = "swap_used_bytes" if fault == "swap-used" else "swapout_bytes"
                data["host"][field] += 1
        return data
    def accept(data, acquire=False):
        if fake.delayed_completed is not None:
            accepted.append(fake.time)
        original_accept(data, acquire)
    fake.read, fake.observe_read, fake.accept = read, observe, accept
    result = g.run([], Path("/work"), {}, SLOT, limits, fake, 1)
    assert injected and result.code == 78 and result.held
    assert "hard resource refusal" in result.reason
    assert accepted == [] and fake.delayed_requests == 1
    assert fake.acquired is (phase != "A")
    assert fake.slot_pending is (phase == "A")
    assert fake.launched is (phase == "S")
    terms = [e[3] for e in fake.events if e[0] == "signal" and e[2] == signal.SIGTERM
             and e[1] != "payload"]
    assert terms and max(terms) <= injected[0] + 0.05 + 1e-9
    assert fake.time <= injected[0] + 2 + 1e-9
    if fault == "disk":
        assert not fake.observer_final  # known authority violation cannot request a seal
        assert not any(sealed for _, _, sealed in fake.observations)
    assert not any(e[0] == "release" for e in fake.events)


@pytest.mark.parametrize("phase", ["A", "F", "S"])
def test_full_census_cancellation_during_post_reply_observer_wait_is_immediate(phase):
    fake = DelayedAcquisition(delay=0, delayed_phase=phase)
    if phase != "S":
        fake.host_value = host(free_bytes=23 * g.GIB)
    original_observe, original_accept = fake.observe_read, fake.accept
    waits, cancelled, accepted = [], [], []
    def observe():
        if fake.delayed_completed is None:
            return original_observe()
        waits.append(fake.time)
        if fake.time >= fake.delayed_completed + 0.15:
            fake.interrupted = True
            cancelled.append(fake.time)
        return None  # pending frame supplies neither freshness nor qualification
    def accept(data, acquire=False):
        if fake.delayed_completed is not None:
            accepted.append(fake.time)
        original_accept(data, acquire)
    fake.observe_read, fake.accept = observe, accept
    result = run(fake, seconds=10)
    assert waits and cancelled and result.code == 78 and result.held
    assert accepted == [] and fake.delayed_requests == 1
    assert fake.acquired is (phase != "A") and fake.slot_pending is (phase == "A")
    assert fake.launched is (phase == "S")
    terms = [e[3] for e in fake.events if e[0] == "signal" and e[2] == signal.SIGTERM
             and e[1] != "payload"]
    assert terms and max(terms) <= cancelled[0] + 1e-9
    assert cancelled[0] <= fake.delayed_completed + 0.2 + 1e-9
    assert fake.time <= cancelled[0] + 2 + 1e-9
    assert not any(sealed for _, _, sealed in fake.observations)
    assert not any(e[0] == "release" for e in fake.events)
