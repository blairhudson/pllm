#!/usr/bin/env python3
"""Fail-closed Poulpy admission and guarded build receipts; stdlib only.

No encrypted workload exists for the frozen eager-key profile: its payload
lower bound exceeds 8 GiB. `qualify` records that refusal without launching HE.
Only the coordinating parent should invoke that phase. Build phases never run
native tests or the native admission binary. No root Cargo files are touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid
from fractions import Fraction

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "benchmarks/native/poulpy_refresh"
GIB = 1 << 30
GUARD_BYTES = 64 << 20
NIGHTLY = "nightly-2026-05-14"
MAX_TRACE = 64 << 20
EXPECTED_REV = "ca3532570ff62b96599a74f7adb85de7726b0446"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    # Exclusive creation preserves failed attempts and all earlier freezes.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def hashes():
    paths = [PROBE / "Cargo.toml", PROBE / "Cargo.lock", PROBE / "rust-toolchain.toml",
             PROBE / ".cargo/config.toml", PROBE / "contract.json", PROBE / "build_only.py", Path(__file__),
             *sorted((PROBE / "src").glob("*.rs")), *sorted((PROBE / "tests").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            if p.is_file() else None for p in paths}


def contract():
    c = json.loads((PROBE / "contract.json").read_text())
    if c["dependency"]["commit"] != EXPECTED_REV or c["dependency"]["nightly"] != NIGHTLY:
        raise ValueError("source/toolchain contract drift")
    if c["resources"]["owner_envelope_ceiling_bytes"] != 8 * GIB:
        raise ValueError("owner ceiling drift")
    if c["resources"]["cpu_threads"] != 1 or c["operations"]["maximum_refreshes"] != 2:
        raise ValueError("execution cap drift")
    return c


def key_lower_bound(c):
    """Independent integer geometry, never a backend call or key allocation."""
    p = c["parameters"]
    if ((p["ring_degree"], p["base2k"], p["rank"], p["input_k"], p["bootstrap_k"], p["output_k"])
            != (65536, 52, 1, 40, 1427, 600)):
        raise ValueError("unsupported profile")
    slots = p["slots"]
    rotations = set()
    for encode, schedule in ((True, p["c2s_schedule"]), (False, p["s2c_schedule"])):
        layer = 15
        for merged, giant in schedule:
            diagonals = {0}
            for _ in range(merged):
                shift = 1 << (layer - 1 if encode else 15 - layer)
                diagonals = {(d + s) % slots for d in diagonals for s in (0, shift, -shift)}
                layer -= 1
            rotations.update(d % giant for d in diagonals)
            rotations.update(d - d % giant for d in diagonals)
        if layer != 0:
            raise ValueError("incomplete DFT geometry")
    rotations.discard(0)
    if sorted(rotations) != c["keys"]["rotation_shifts"]:
        raise ValueError("rotation inventory drift")
    count = len(rotations) + 1  # conjugation is separate
    h = p["high_key"]
    # Rectangular MatZnx/VmpPMat storage; Q120bScalar is four u64 lanes.
    limbs = (h["dnum"] * h["dsize"] * p["base2k"] + h["k_aux"] + p["base2k"] - 1) // p["base2k"]
    elements = p["ring_degree"] * h["dnum"] * p["rank"] * (p["rank"] + 1) * limbs
    raw_one, prepared_one = elements * 8, elements * 32
    result = {"raw_word_bytes": 8, "prepared_word_bytes": 32,
              "one_raw_high_key_payload_bytes": raw_one,
              "one_prepared_high_key_payload_bytes": prepared_one,
              "raw_automorphism_payload_bytes": count * raw_one,
              "prepared_automorphism_payload_bytes": count * prepared_one,
              "automorphism_preparation_overlap_payload_bytes": count * (raw_one + prepared_one)}
    if result != c["resources"]["known_lower_bounds"] or count != c["keys"]["automorphism_count"]:
        raise ValueError("frozen payload lower bound drift")
    return result


def admission_blockers(c):
    b = key_lower_bound(c)
    blockers = []
    if b["prepared_automorphism_payload_bytes"] > c["resources"]["owner_envelope_ceiling_bytes"]:
        blockers.append("prepared_automorphism_payload_exceeds_8_gib")
    if c["resources"]["full_owner_bound_bytes"] is None:
        blockers.append("full_key_setup_context_copies_scratch_owner_bound_missing")
    pq = c["parameters"]["parameter_qualification"]
    if pq["lattice_estimator_receipt"] is None or pq["whole_circuit_failure_probability_receipt"] is None:
        blockers.append("independent_parameter_qualification_missing")
    # No CLI/env bypass exists. A genuinely different admitted implementation
    # needs a separately reviewed contract and code, not edited evidence fields.
    blockers.append("encrypted_pipeline_not_implemented_for_rejected_profile")
    return blockers


def fixture_slot(j):
    if not 0 <= j < 32768:
        raise ValueError("slot index")
    return ((Fraction((17 * j % 129) - 64, 1024), Fraction((29 * j % 129) - 64, 2048))
            if j < 127 else (Fraction(0), Fraction(0)))


def oracle_slot(checkpoint, j):
    """Closed-form oracle independent of encrypted implementation/encoding."""
    if checkpoint == "fresh":
        return fixture_slot(j)
    if checkpoint in {"affine_1", "refresh_1"}:
        re_, im = fixture_slot((j + 1) % 32768)
        return 2 * re_ + Fraction(1, 64), -2 * im - Fraction(1, 128)
    if checkpoint in {"affine_2", "entry_2", "refresh_2"}:
        re_, im = fixture_slot((j + 9) % 32768)
        return 4 * re_, 4 * im + Fraction(1, 32)
    raise ValueError("unknown checkpoint")


def check_slots(checkpoint, real, imag, tolerance):
    if len(real) != 32768 or len(imag) != 32768:
        raise ValueError("full valid and padding vector required")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("error threshold")
    worst = 0.0
    for j, (re_, im) in enumerate(zip(real, imag)):
        expected = oracle_slot(checkpoint, j)
        for got, want in zip((re_, im), expected):
            if isinstance(got, bool) or not isinstance(got, (int, float)) or not math.isfinite(got):
                raise ValueError(f"non-finite/non-numeric slot {j}")
            error = abs(got - float(want))
            if error > tolerance:
                raise ValueError(f"slot {j} exceeds frozen error threshold")
            worst = max(worst, error)
    return {"components_checked": 65536, "max_absolute_error": worst}


def command_text(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=5).stdout


def parse_host(vm, swap, physical):
    page = re.search(r"page size of (\d+) bytes", vm)
    if not page:
        raise ValueError("unknown vm_stat header")
    rows = {k.strip(): int(v) for k, v in re.findall(r"^([^:\n]+):\s*(\d+)\.?$", vm, re.M)}
    used = re.search(r"used\s*=\s*([0-9.]+)([MG])", swap)
    if not used or not {"Pages free", "Pages speculative", "Swapouts"} <= rows.keys():
        raise ValueError("missing memory/swap counters")
    return {"physical_bytes": int(physical), "page_bytes": int(page[1]),
            "free_speculative_bytes": (rows["Pages free"] + rows["Pages speculative"]) * int(page[1]),
            "swap_used_bytes": int(float(used[1]) * (GIB if used[2] == "G" else 1 << 20)),
            "swapout_pages": rows["Swapouts"]}


def host_snapshot():
    if sys.platform != "darwin" or os.uname().machine != "arm64":
        raise ValueError("profile requires ARM64 macOS")
    return parse_host(command_text(["vm_stat"]), command_text(["sysctl", "vm.swapusage"]),
                      command_text(["sysctl", "-n", "hw.memsize"]))


def processes():
    text = command_text(["ps", "-axo", "pid=,ppid=,pgid=,rss=,stat="])
    result = []
    for line in text.splitlines():
        pid, ppid, pgid, rss, state = line.split(maxsplit=4)
        result.append({"pid": int(pid), "ppid": int(ppid), "pgid": int(pgid),
                       "rss_bytes": int(rss) * 1024, "state": state})
    return result


def group_members(pgid):
    return [p for p in processes() if p["pgid"] == pgid and not p["state"].startswith("Z")]


def cleanup_group(child):
    """Also kill surviving descendants when the leader already exited."""
    # Reap an exited leader before signalling. An empty zombie-only group can
    # reject killpg on macOS even though there is no live process to retire.
    child.poll()
    if child.returncode is not None and not group_members(child.pid):
        return {"complete": True, "remaining_pids": []}
    for sig, grace in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            child.poll()
            if child.returncode is not None and not group_members(child.pid):
                return {"complete": True, "remaining_pids": []}
            raise
        until = time.monotonic() + grace
        while time.monotonic() < until:
            child.poll()  # reap leader
            if not group_members(child.pid):
                child.wait(timeout=1)
                return {"complete": True, "remaining_pids": []}
            time.sleep(0.05)
    return {"complete": False, "remaining_pids": [p["pid"] for p in group_members(child.pid)]}


def resource_violation(sample, baseline, owner_cap, reserve, elapsed, seconds):
    if elapsed > seconds:
        return "wall_clock_limit"
    if sample["owner_rss_bytes"] > owner_cap:
        return "aggregate_owner_rss_limit"
    if sample["host"]["free_speculative_bytes"] < reserve:
        return "host_reserve_breach"
    if sample["host"]["swap_used_bytes"] > baseline["swap_used_bytes"]:
        return "additional_swap"
    if sample["host"]["swapout_pages"] > baseline["swapout_pages"]:
        return "additional_swapout"
    return None


def guarded(command, directory, *, seconds, owner_cap, reserve, disk_required,
            sample_host=host_snapshot, interval=0.1):
    """Operational build guard, not proof of cryptographic owner admission.

    Includes a fixed guard/helper reserve plus aggregate group RSS. Sampling
    cannot prove a missed peak or prevent another host application's swap, so
    runtime receipts never substitute for a complete preallocation bound.
    """
    directory = Path(directory)
    result = {"command": command, "status": "blocked_prelaunch", "exit_code": None,
              "seconds_ceiling": seconds, "owner_ceiling_bytes": owner_cap,
              "host_reserve_bytes": reserve, "peak_observed_owner_rss_bytes": None,
              "child_launched": False,
              "trace_kind": "sampled_build_guard_not_complete_HE_owner_proof",
              "trace_complete": False, "cleanup": {"complete": True, "remaining_pids": []}}
    child = None
    started = time.monotonic()
    try:
        baseline = sample_host()
        result["baseline_host"] = baseline
        if baseline["free_speculative_bytes"] < owner_cap + reserve:
            result["reason"] = "insufficient_conservative_physical_headroom"
            return result
        initial_disk_free = shutil.disk_usage(PROBE).free
        result["baseline_disk_free_bytes"] = initial_disk_free
        if initial_disk_free < disk_required:
            result["reason"] = "insufficient_disk_reserve"
            return result
        env = dict(os.environ, CARGO_BUILD_JOBS="2", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", BLIS_NUM_THREADS="1",
                   CARGO_TARGET_DIR=str(PROBE / "target"), RUSTUP_TOOLCHAIN=NIGHTLY)
        # Explicit arguments select toolchain; do not inherit RUSTFLAGS or
        # RUSTC_WRAPPER hooks which could change semantics or spawn workloads.
        for key in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"):
            env.pop(key, None)

        def limits():
            os.setsid()
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            resource.setrlimit(resource.RLIMIT_FSIZE, (512 << 20, 512 << 20))
            cpu_seconds = 2 * math.ceil(seconds) + 10
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        with (directory / "stdout.log").open("xb") as out, (directory / "stderr.log").open("xb") as err, \
                (directory / "resources.jsonl").open("xb") as trace:
            child = subprocess.Popen(command, cwd=PROBE, env=env, stdout=out, stderr=err, preexec_fn=limits)
            result.update(status="running", pid=child.pid, pgid=child.pid, child_launched=True)
            previous = time.monotonic()
            known = {child.pid}
            while True:
                now = time.monotonic()
                table = processes()
                # Track descendants too; an escaped process group is a failure.
                changed = True
                while changed:
                    found = {p["pid"] for p in table if p["ppid"] in known}
                    changed = not found <= known
                    known |= found
                members = [p for p in table if p["pgid"] == child.pid or p["pid"] in known]
                sample = {"elapsed_seconds": now - started, "host": sample_host(), "processes": members,
                          "owner_rss_bytes": GUARD_BYTES + sum(p["rss_bytes"] for p in members),
                          "guard_self_rss_bytes": next(p["rss_bytes"] for p in table if p["pid"] == os.getpid()),
                          "disk_free_bytes": shutil.disk_usage(PROBE).free}
                # Observers can block while a short-lived child exits. Admit
                # neither the deadline nor the sample gap using an earlier
                # timestamp, including on the final successful child poll.
                observed = time.monotonic()
                sample["elapsed_seconds"] = observed - started
                sample["read_seconds"] = observed - now
                trace.write(canonical(sample) + b"\n")
                trace.flush()
                result["peak_observed_owner_rss_bytes"] = max(result["peak_observed_owner_rss_bytes"] or 0, sample["owner_rss_bytes"])
                violation = resource_violation(sample, baseline, owner_cap, reserve, observed - started, seconds)
                if sample["guard_self_rss_bytes"] > GUARD_BYTES:
                    violation = violation or "guard_reserve_exceeded"
                if disk_required and (sample["disk_free_bytes"] < 8 * GIB
                                      or initial_disk_free - sample["disk_free_bytes"] > 4 * GIB):
                    violation = violation or "disk_budget_breach"
                if observed - previous > 2.0:
                    violation = violation or "resource_sample_gap"
                if trace.tell() > MAX_TRACE or out.tell() + err.tell() > MAX_TRACE:
                    violation = violation or "bounded_trace_or_log_limit"
                escaped = [p for p in members if p["pgid"] != child.pid and not p["state"].startswith("Z")]
                if escaped:
                    violation = "escaped_process_group"
                    for p in escaped:
                        try:
                            os.kill(p["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if violation:
                    result.update(status="failed_guard", reason=violation)
                    break
                previous = observed
                code = child.poll()
                if code is not None:
                    result.update(status="passed" if code == 0 else "failed_command", exit_code=code)
                    break
                time.sleep(interval)
            result["trace_complete"] = result["status"] == "passed"
    except BaseException as exc:
        result.update(status="failed_observer", reason=f"{type(exc).__name__}: {exc}")
    finally:
        if child is not None:
            try:
                result["cleanup"] = cleanup_group(child)
                result["exit_code"] = child.poll()
                result["final_host"] = sample_host()
                if (result["final_host"]["swap_used_bytes"] > baseline["swap_used_bytes"]
                        or result["final_host"]["swapout_pages"] > baseline["swapout_pages"]):
                    result.update(status="failed_guard", reason="additional_swap_during_cleanup")
                if not result["cleanup"]["complete"]:
                    result.update(status="failed_cleanup", reason="live_child_group_after_kill")
            except BaseException as exc:
                last_signal_error = None
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except OSError as kill_error:
                    last_signal_error = type(kill_error).__name__
                try:
                    child.wait(timeout=1)
                except (subprocess.TimeoutExpired, OSError):
                    pass
                result.update(status="failed_cleanup", reason=f"cleanup observer: {exc}",
                              cleanup={"complete": False, "remaining_pids": None,
                                       "last_signal_error": last_signal_error})
        result["elapsed_seconds"] = time.monotonic() - started
        if result["status"] != "passed":
            result["trace_complete"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("toolchain", "resolve", "build", "qualify"))
    args = parser.parse_args()
    def interrupted(signum, _frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    c = contract()
    lower = key_lower_bound(c)
    PROBE.joinpath("receipts").mkdir(exist_ok=True)
    directory = PROBE / "receipts" / (time.strftime("%Y%m%dT%H%M%S") + "-" + args.phase + "-" + uuid.uuid4().hex[:8])
    directory.mkdir(mode=0o700)
    lock = PROBE / ".guard.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        write_json(directory / "result.json", {"status": "blocked_prelaunch", "reason": "probe_guard_lock_exists"})
        return 78
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(os.getpid()))
        # Freeze all inputs and limits before any child, including failures.
        freeze = {"schema": "pllm.poulpy-build-freeze.v1", "phase": args.phase, "contract": c,
                  "input_sha256": hashes(), "key_lower_bound": lower,
                  "execution_owner": "coordinating_parent" if args.phase == "qualify" else "build_only"}
        write_json(directory / "freeze.json", freeze)
        if args.phase == "qualify":
            result = {"status": "blocked_before_keygen", "blockers": admission_blockers(c),
                      "encrypted_refreshes": 0, "child_launched": False, "key_lower_bound": lower,
                      "full_owner_bound_bytes": None, "complete_qualification_trace": False}
        else:
            if args.phase == "toolchain":
                command = ["rustup", "toolchain", "install", NIGHTLY, "--profile", "minimal",
                           "--target", "aarch64-apple-darwin", "--no-self-update"]
                seconds, cap = c["build"]["toolchain_setup_seconds"], GIB
            elif args.phase == "resolve":
                # Installed stable Cargo can resolve this edition-2024 lock;
                # it never compiles or executes the nightly-only dependency.
                command = ["rustup", "run", "1.85.1", "cargo", "generate-lockfile", "--manifest-path", str(PROBE / "Cargo.toml")]
                seconds, cap = c["build"]["resolution_seconds"], GIB
            else:
                command = [sys.executable, str(PROBE / "build_only.py")]
                seconds, cap = c["build"]["compile_seconds"], 4 * GIB
            freeze["command"] = command
            # Separate command file is also immutable and predates launch.
            write_json(directory / "command.json", {"argv": command, "seconds": seconds, "owner_cap": cap})
            if args.phase == "build" and not (PROBE / "Cargo.lock").is_file():
                result = {"status": "blocked_prelaunch", "reason": "standalone_dependency_lock_missing"}
            else:
                result = guarded(command, directory, seconds=seconds, owner_cap=cap,
                                 reserve=4 * GIB, disk_required=12 * GIB)
        result["input_sha256_after"] = hashes()
        changes = [name for name, value in freeze["input_sha256"].items()
                   if result["input_sha256_after"].get(name) != value
                   and not (args.phase == "resolve" and name.endswith("/Cargo.lock"))]
        if changes:
            result.update(status="failed_input_drift", reason="frozen_inputs_changed", changed_inputs=changes)
        result["receipt_directory"] = str(directory.relative_to(ROOT))
        result["performance_claim"] = False
        write_json(directory / "result.json", result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result["status"] == "passed" else 78
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
