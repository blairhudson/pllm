#!/usr/bin/env python3
"""Parent-only, pinned, serial full-geometry fusion fixture. Never builds Rust."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
ROWS, COLS = 151936, 896
RESERVE = 2 * 1024**3
ALLOCATION = (2 * ROWS * COLS + 3 * 4 * ROWS + 16 * 4 * ROWS
              + 4 * ROWS + 32 * 4 * COLS + 32 * 1024**2)
ARMS = ("lookup", "head", "original_pair", "fused_pair", "complementary_head")
TEST = "private_pages::boundary_scan_fusion_probe::full_probe::full_fixture"
LOCK = Path.home() / ".cache/pllm/frontier-probe.lock"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sources():
    paths = set(ROOT.glob("crates/**/*.rs")) | set(ROOT.glob("crates/**/Cargo.toml"))
    paths |= {ROOT / "Cargo.toml", ROOT / "Cargo.lock"}
    paths |= {ROOT / p for p in (
        "scripts/probe_boundary_scan_fusion_full.py",
        "scripts/summarize_boundary_scan_fusion_full.py",
        "tests/test_boundary_scan_fusion_full.py")}
    return {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sorted(paths)}


def freeze(binary, build_log, source_manifest, workers, repetitions):
    if not 3 <= workers <= 16 or repetitions not in range(2, 17, 2):
        raise ValueError("need 3..16 workers and 2..16 even repetitions")
    binary = Path(binary).resolve()
    source_raw = Path(source_manifest).read_bytes()
    frozen_sources = json.loads(source_raw)
    if frozen_sources != sources():
        raise ValueError("sources changed since pre-build snapshot")
    log = Path(build_log).read_bytes()
    records = [json.loads(line) for line in log.splitlines() if line.startswith(b"{")]
    if not any(r.get("reason") == "compiler-artifact" and r.get("profile", {}).get("test")
               and r.get("target", {}).get("name") == "pllm_garble"
               and r.get("executable") and Path(r["executable"]).resolve() == binary
               for r in records):
        raise ValueError("binary absent from parent cargo test build log")
    if not any(r.get("reason") == "build-finished" and r.get("success") is True for r in records):
        raise ValueError("successful build log required")
    return {"schema": "boundary-scan-fusion-full-pins-v1", "rows": ROWS, "columns": COLS,
            "workers": workers, "repetitions": repetitions, "sources": frozen_sources,
            "prebuild_sources_sha256": sha(source_raw),
            "binary": str(binary), "binary_sha256": sha(binary.read_bytes()),
            "build_log_sha256": sha(log), "build_log": log.decode(),
            "rustc": command("rustc", "-Vv"), "cargo": command("cargo", "-V"),
            "head": command("git", "rev-parse", "HEAD"), "allocation_bound_bytes": ALLOCATION,
            "reserve_bytes": RESERVE, "test": TEST}


def host():
    if os.uname().sysname != "Darwin":
        raise RuntimeError("fail closed: native macOS headroom/swap guard required")
    vm = command("/usr/bin/vm_stat")
    page = int(re.search(r"page size of (\d+) bytes", vm)[1])
    pages = sum(int(re.search(rf"^{name}:\s+(\d+)", vm, re.M)[1])
                for name in ("Pages free", "Pages inactive"))
    swap = command("/usr/sbin/sysctl", "-n", "vm.swapusage")
    used = re.search(r"used =\s*([\d.]+)M", swap)
    if used is None:
        raise RuntimeError("unknown native swap units")
    return {"available_bytes": page * pages, "swap_used_bytes": int(float(used[1]) * 1024**2)}


def check_guard(observation, initial):
    if observation["available_bytes"] < RESERVE + ALLOCATION:
        raise RuntimeError("headroom rejected: full envelope plus 2GiB reserve required")
    if observation["swap_used_bytes"] != initial["swap_used_bytes"]:
        raise RuntimeError("zero-swap guard rejected")


def run(pins_path, prefix):
    pins_raw = Path(pins_path).read_bytes()
    pins = json.loads(pins_raw)
    # Reuse pure reader validation before any native dispatch.
    try:
        from scripts.summarize_boundary_scan_fusion_full import audit_worker, validate_pins
    except ModuleNotFoundError:
        from summarize_boundary_scan_fusion_full import audit_worker, validate_pins
    validate_pins(pins)
    binary = Path(pins["binary"])
    if sources() != pins["sources"] or sha(binary.read_bytes()) != pins["binary_sha256"]:
        raise ValueError("source/binary changed after freeze")
    if command("rustc", "-Vv") != pins["rustc"] or command("cargo", "-V") != pins["cargo"]:
        raise ValueError("toolchain changed after freeze")
    prefix = Path(prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if list(prefix.parent.glob(prefix.name + "*")):
        raise FileExistsError("exclusive report prefix already used")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Exclusive claim survives crashes; never overwrite/restart a partial cohort.
        write_new(str(prefix) + ".claim.json", {"pins_sha256": sha(pins_raw), "pins": pins})
        with Path(str(prefix) + ".progress.jsonl").open("x") as progress:
            for worker in range(pins["workers"]):
                if sources() != pins["sources"] or sha(binary.read_bytes()) != pins["binary_sha256"]:
                    raise ValueError("pins changed between workers")
                initial = host()
                check_guard(initial, initial)
                observations = [initial]
                stdout_path = Path(f"{prefix}.worker{worker}.stdout")
                stderr_path = Path(f"{prefix}.worker{worker}.stderr")
                report = {"schema": "boundary-scan-fusion-full-worker-v1", "worker": worker,
                          "pins_sha256": sha(pins_raw), "observed_binary_sha256": sha(binary.read_bytes()),
                          "observed_sources": sources(), "guards": observations,
                          "status": "failed", "mixed_fixture_sampled_peak_rss_bytes": 0}
                env = os.environ | {"PLLM_FUSION_FULL_ADMITTED": "1",
                                    "PLLM_FUSION_FULL_REPETITIONS": str(pins["repetitions"])}
                failure = None
                with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
                    child = subprocess.Popen([str(binary), TEST, "--exact", "--ignored", "--nocapture",
                                              "--test-threads=1"], env=env, stdout=out, stderr=err)
                    report["pid"] = child.pid
                    started = time.monotonic()
                    try:
                        while child.poll() is None:
                            if time.monotonic() - started > 600:
                                raise RuntimeError("bounded worker 600-second timeout")
                            observation = host()
                            observations.append(observation)
                            check_guard(observation, initial)
                            rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", str(child.pid)],
                                                 capture_output=True, text=True, check=False)
                            if rss.stdout.strip():
                                report["mixed_fixture_sampled_peak_rss_bytes"] = max(
                                    report["mixed_fixture_sampled_peak_rss_bytes"], int(rss.stdout) * 1024)
                            time.sleep(.1)
                        observation = host()
                        observations.append(observation)
                        check_guard(observation, initial)
                    except BaseException as exc:
                        failure = repr(exc)
                        child.kill()
                    finally:
                        report["returncode"] = child.wait()
                report["stdout"] = stdout_path.read_text()
                report["stderr"] = stderr_path.read_text()
                report["pins_unchanged_after"] = (sources() == pins["sources"]
                    and sha(binary.read_bytes()) == pins["binary_sha256"])
                if failure is None and report["returncode"] == 0 and report["pins_unchanged_after"]:
                    report["status"] = "complete"
                report["failure"] = failure
                if report["status"] == "complete":
                    try:
                        audit_worker(report, pins, sha(pins_raw))
                    except (ValueError, KeyError, TypeError) as exc:
                        report["status"] = "failed"
                        report["failure"] = repr(exc)
                path = Path(f"{prefix}.worker{worker}.json")
                write_new(path, report)
                progress.write(json.dumps({"worker": worker, "path": str(path),
                                           "sha256": sha(path.read_bytes()), "status": report["status"]}) + "\n")
                progress.flush()
                os.fsync(progress.fileno())
                if report["status"] != "complete":
                    raise RuntimeError(f"worker rejected; partial evidence retained: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    snapshot = sub.add_parser("sources", help="snapshot sources before parent build")
    snapshot.add_argument("--output", required=True)
    pin = sub.add_parser("pin", help="freeze existing parent-built libtest binary; never build")
    pin.add_argument("--binary", required=True)
    pin.add_argument("--build-log", required=True)
    pin.add_argument("--source-manifest", required=True)
    pin.add_argument("--output", required=True)
    pin.add_argument("--workers", type=int, default=3)
    pin.add_argument("--repetitions", type=int, default=10)
    launch = sub.add_parser("run", help="serial fresh native workers under shared fail-fast lock")
    launch.add_argument("--pins", required=True)
    launch.add_argument("--prefix", required=True)
    args = parser.parse_args()
    if args.action == "sources":
        write_new(args.output, sources())
    elif args.action == "pin":
        write_new(args.output, freeze(args.binary, args.build_log, args.source_manifest,
                                     args.workers, args.repetitions))
    else:
        run(args.pins, args.prefix)


if __name__ == "__main__":
    main()
