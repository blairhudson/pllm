"""Model-free audit tests. All timing values below are synthetic, never evidence."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import probe_boundary_scan_fusion_full as probe
from scripts import summarize_boundary_scan_fusion_full as reader


def fixture():
    source_map = {name: "a" * 64 for name in ("one.rs", "two.rs", "three.py")}
    build_log = "\n".join(json.dumps(r) for r in (
        {"reason": "compiler-artifact", "target": {"name": "pllm_garble"},
         "profile": {"test": True}, "executable": "/synthetic/pllm_garble"},
        {"reason": "build-finished", "success": True}))
    pins = {"schema": "boundary-scan-fusion-full-pins-v1", "rows": probe.ROWS,
            "columns": probe.COLS, "workers": 3, "repetitions": 2, "sources": source_map,
            "binary": "/synthetic/pllm_garble", "binary_sha256": "b" * 64,
            "build_log": build_log, "build_log_sha256": probe.sha(build_log.encode()),
            "prebuild_sources_sha256": probe.sha(probe.canonical(source_map) + b"\n"),
            "rustc": "synthetic rustc", "cargo": "synthetic cargo", "head": "c" * 40,
            "allocation_bound_bytes": probe.ALLOCATION, "reserve_bytes": probe.RESERVE,
            "test": probe.TEST}
    pins_raw = probe.canonical(pins)
    guard = {"available_bytes": probe.RESERVE + probe.ALLOCATION, "swap_used_bytes": 0}
    samples = []
    for iteration in range(2):
        for party in range(2):
            shift = iteration // 2 + party
            order = [probe.ARMS[(i + shift) % 5] for i in range(5)]
            if iteration % 2:
                order.reverse()
            a, b = order.index("original_pair"), order.index("fused_pair")
            if (a > b) == ((iteration + party) % 2 == 0):
                order[a], order[b] = order[b], order[a]
            samples.append({"iteration": iteration, "party": party, "order": order,
                            "timings": {arm: {"wall_ns": 100 + n, "process_cpu_ns": 70 + n}
                                        for n, arm in enumerate(probe.ARMS)},
                            "parity": True, "replay_rejected": True, "rows_visited": probe.ROWS,
                            "weight_values_visited": probe.ROWS * probe.COLS,
                            "scale_bytes_visited": 4 * probe.ROWS})
    native = {"schema": "boundary-scan-fusion-full-v1", "rows": probe.ROWS, "columns": probe.COLS,
              "repetitions": 2, "samples": samples, "setup_wall_ns": 1,
              "allocation_bound_bytes": probe.ALLOCATION,
              "work_bound_weight_values": probe.ROWS * probe.COLS * (2 * 2 * 12 + 8),
              "native_initial_guard": guard, "native_guards": [guard, guard],
              "native_final_guard": guard, "reserve_bytes": probe.RESERVE,
              "native_simd": True, "matrix_owners": 1, "parties": 2, "correct": True,
              "fault_controls": True, "physical_memory_scans_measured": False, "sdk_selectable": False}
    reports = []
    for worker in range(3):
        reports.append({"schema": "boundary-scan-fusion-full-worker-v1", "worker": worker,
                        "pins_sha256": probe.sha(pins_raw), "observed_binary_sha256": "b" * 64,
                        "observed_sources": source_map, "guards": [guard, guard],
                        "status": "complete", "mixed_fixture_sampled_peak_rss_bytes": 123456,
                        "pid": 100 + worker, "returncode": 0, "pins_unchanged_after": True,
                        "failure": None, "stderr": "", "stdout": stdout(native)})
    return pins, reports, native


def stdout(native):
    progress = "\n".join("FUSION_FULL_PROGRESS " + json.dumps(s) for s in native["samples"])
    return ("running 1 test\ntest fixture ... " + progress + "\n" + json.dumps(native)
            + "\nok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 9 filtered out\n")


def audit(tmp_path, pins, reports):
    paths = []
    for i, report in enumerate(reports):
        path = tmp_path / f"worker{i}.json"
        path.write_bytes(probe.canonical(report))
        paths.append(path)
    before = {p: p.read_bytes() for p in paths}
    result = reader.summarize(paths, probe.canonical(pins))
    assert before == {p: p.read_bytes() for p in paths}
    return result


def test_replay_read_only_and_explicit_regression(tmp_path):
    pins, reports, _ = fixture()
    result = audit(tmp_path, pins, reports)
    assert result["scalar_fusion_regresses"] is True
    assert result["median_worker_paired_cpu_ratio"] == 73 / 72
    assert result["logical_two_worker_source_bytes"]["candidate"] == 3 * probe.ROWS * probe.COLS + 8 * probe.ROWS
    assert "mixed-fixture" in result["memory_scope"]
    assert "blocked" in result["promotion_gate"]
    assert len(result["workers"][0]["analytical_schedules"]) == 4
    assert result["workers"][0]["two_worker_token_cpu"][0] == {
        "baseline_two_worker_cpu_ns": 216, "candidate_two_worker_cpu_ns": 217}


@pytest.mark.parametrize("field,value", [
    ("rows", 4096), ("columns", 897), ("workers", 2), ("repetitions", 3),
    ("reserve_bytes", 0), ("allocation_bound_bytes", 0), ("test", "other"),
    ("build_log_sha256", "0" * 64), ("head", "bad"), ("prebuild_sources_sha256", "0" * 64),
])
def test_bad_pins(tmp_path, field, value):
    pins, reports, _ = fixture()
    pins[field] = value
    with pytest.raises(ValueError):
        audit(tmp_path, pins, reports)


@pytest.mark.parametrize("change", [
    "native_geometry", "native_repetitions", "parity", "sample_parity", "faults", "duplicate_sample",
    "missing_order", "wrong_order", "missing_sample", "missing_arm", "negative_cpu", "bool_cpu",
    "work", "owners", "physical", "native_headroom", "native_swap", "native_guard_gap", "native_cache",
])
def test_bad_native_rejected(tmp_path, change):
    pins, reports, native = fixture()
    if change == "native_geometry": native["rows"] = 4096
    elif change == "native_repetitions": native["repetitions"] = 4
    elif change == "parity": native["correct"] = False
    elif change == "sample_parity": native["samples"][0]["parity"] = False
    elif change == "faults": native["fault_controls"] = False
    elif change == "duplicate_sample": native["samples"][1] = native["samples"][0]
    elif change == "missing_order": native["samples"][0]["order"].pop()
    elif change == "wrong_order": native["samples"][0]["order"].reverse()
    elif change == "missing_sample": native["samples"].pop()
    elif change == "missing_arm": native["samples"][0]["timings"].pop("head")
    elif change == "negative_cpu": native["samples"][0]["timings"]["head"]["process_cpu_ns"] = -1
    elif change == "bool_cpu": native["samples"][0]["timings"]["head"]["process_cpu_ns"] = True
    elif change == "work": native["samples"][0]["weight_values_visited"] = 0
    elif change == "owners": native["matrix_owners"] = 2
    elif change == "physical": native["physical_memory_scans_measured"] = True
    elif change == "native_headroom": native["native_final_guard"] = {"available_bytes": 0, "swap_used_bytes": 0}
    elif change == "native_swap": native["native_final_guard"] = {"available_bytes": 10**12, "swap_used_bytes": 1}
    elif change == "native_guard_gap": native["native_guards"].pop()
    elif change == "native_cache": native["summary"] = {"ratio": .1}
    reports[0]["stdout"] = stdout(native)
    with pytest.raises(ValueError):
        audit(tmp_path, pins, reports)


@pytest.mark.parametrize("change", [
    "binary", "sources", "after", "status", "returncode", "headroom", "swap", "guard_gap",
    "cached_summary", "cached_native", "pid", "pid_duplicate", "worker_duplicate", "cohort_gap", "libtest_gap",
])
def test_bad_reports_rejected(tmp_path, change):
    pins, reports, _ = fixture()
    report = reports[0]
    if change == "binary": report["observed_binary_sha256"] = "0" * 64
    elif change == "sources": report["observed_sources"] = {}
    elif change == "after": report["pins_unchanged_after"] = False
    elif change == "status": report["status"] = "failed"
    elif change == "returncode": report["returncode"] = 1
    elif change == "headroom": report["guards"] = [{"available_bytes": 0, "swap_used_bytes": 0}] * 2
    elif change == "swap": report["guards"] = [{"available_bytes": 10**12, "swap_used_bytes": n} for n in (0, 1)]
    elif change == "guard_gap": report["guards"] = []
    elif change == "cached_summary": report["summary"] = {"ratio": .1}
    elif change == "cached_native": report["native"] = {"correct": True}
    elif change == "pid": report["pid"] = None
    elif change == "pid_duplicate": reports[1]["pid"] = report["pid"]
    elif change == "worker_duplicate": reports[1]["worker"] = 0
    elif change == "cohort_gap": reports.pop()
    elif change == "libtest_gap": report["stdout"] = report["stdout"].replace("test result: ok.", "incomplete")
    with pytest.raises(ValueError):
        audit(tmp_path, pins, reports)


def test_schedule_explicit_delay_and_ready_parallel():
    result = reader.schedule(10, 12, 20, 25, 20, 100)
    assert result == {"installed_ready_parallel_body_start_ns": 12,
                      "installed_ready_parallel_finish_ns": 132, "fused_body_start_ns": 25,
                      "fused_finish_ns": 145, "fused_body_start_delay_ns": 13,
                      "separate_runahead_finish_ns": 132}


def test_allocation_is_aggregate_and_bounded():
    assert 2 * probe.ROWS * probe.COLS + 3 * 4 * probe.ROWS < probe.ALLOCATION < 384 * 1024**2
    with pytest.raises(RuntimeError):
        probe.check_guard({"available_bytes": probe.ALLOCATION, "swap_used_bytes": 0}, {"swap_used_bytes": 0})
    with pytest.raises(RuntimeError):
        probe.check_guard({"available_bytes": 10**12, "swap_used_bytes": 1}, {"swap_used_bytes": 0})


def test_exclusive_write_preserves_partial(tmp_path):
    path = tmp_path / "partial.json"
    probe.write_new(path, {"incomplete": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_new(path, {"incomplete": False})
    assert path.read_bytes() == original


def test_fail_fast_shared_lock(tmp_path):
    import fcntl
    assert probe.LOCK == Path.home() / ".cache/pllm/frontier-probe.lock"
    path = tmp_path / "lock"
    with path.open("a+") as first, path.open("a+") as second:
        fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(second, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_native_host_parse_conservative(monkeypatch):
    class Darwin:
        sysname = "Darwin"
    monkeypatch.setattr(probe.os, "uname", lambda: Darwin())
    monkeypatch.setattr(probe, "command", lambda *args: (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free: 100000.\nPages inactive: 200000.\nPages speculative: 999999.\n"
        if args[0].endswith("vm_stat") else "total = 0.00M  used = 0.00M  free = 0.00M"))
    assert probe.host() == {"available_bytes": 300000 * 16384, "swap_used_bytes": 0}


def test_progress_must_match_native_samples(tmp_path):
    pins, reports, _ = fixture()
    reports[0]["stdout"] = reports[0]["stdout"].replace("FUSION_FULL_PROGRESS ", "FORGED ", 1)
    with pytest.raises(ValueError):
        audit(tmp_path, pins, reports)


@pytest.mark.parametrize("script", ["probe_boundary_scan_fusion_full.py", "summarize_boundary_scan_fusion_full.py"])
def test_help_model_free(script):
    result = subprocess.run([sys.executable, str(Path("scripts") / script), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "usage:" in result.stdout
