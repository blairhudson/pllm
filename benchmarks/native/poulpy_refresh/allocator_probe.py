"""Guarded, dependency-free rustc allocator tests. Never build or execute HE."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GUARD_PATH = ROOT / "scripts/qualify_rust_ckks_refresh.py"


def commands(directory):
    prefix = ["rustup", "run", "1.85.1", "rustc", "--edition=2021", "--target=aarch64-apple-darwin", "-C", "codegen-units=1", "-C", "debuginfo=0"]
    unit, smoke = directory / "allocator-unit", directory / "allocator-global-smoke"
    return {
        "compile": [
            [*prefix, "--crate-name", "allocator_unit", "--test", str(HERE / "src/bounded_allocator.rs"), "-o", str(unit)],
            [*prefix, "--crate-name", "allocator_global_smoke", str(HERE / "tests/allocator/smoke.rs"), "-o", str(smoke)],
        ],
        "run": [[str(unit), "--test-threads=1"], [str(smoke)]],
    }


def input_hashes():
    paths = [Path(__file__), GUARD_PATH, HERE / "bounded-allocator-contract.json", HERE / "allocator_plan.py",
             HERE / "staged_admission.py", HERE / "staged_admission.sources.json", HERE / "tests/test_allocator_plan.py",
             HERE / "src/bounded_allocator.rs", HERE / "tests/allocator/unit.rs", HERE / "tests/allocator/smoke.rs"]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    def interrupted(signum, _frame):
        raise SystemExit(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", choices=("compile", "run"), help=argparse.SUPPRESS)
    parser.add_argument("--binary-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.binary_dir is None:
            parser.error("missing binary directory")
        if args.child == "compile":
            identity = subprocess.check_output(["rustup", "run", "1.85.1", "rustc", "--version", "--verbose"], text=True)
            required = json.loads((HERE / "bounded-allocator-contract.json").read_text())["isolated_tests"]
            if ("commit-hash: " + required["rustc_commit"]) not in identity or "host: aarch64-apple-darwin" not in identity:
                raise SystemExit("installed isolated-test compiler identity mismatch")
            print(json.dumps({"rustc_identity": identity}), flush=True)
        # Invoked only beneath the outer process-group guard. All children are
        # sequential; explicit installed rustup toolchain never installs one.
        for command in commands(args.binary_dir)[args.child]:
            print(json.dumps({"command": command}), flush=True)
            subprocess.run(command, cwd=HERE, check=True)
        return 0
    guard = runpy.run_path(str(GUARD_PATH))
    contract = json.loads((HERE / "bounded-allocator-contract.json").read_text())
    limits = contract["isolated_tests"]
    suffix = time.strftime("%Y%m%dT%H%M%S") + "-allocator-" + uuid.uuid4().hex[:8]
    receipt = HERE / "receipts" / suffix
    receipt.mkdir(mode=0o700, parents=True)
    binary_dir = HERE / "target" / suffix
    frozen = input_hashes()
    calculated = runpy.run_path(str(HERE / "allocator_plan.py"))["plan"]()
    guard["write_json"](receipt / "freeze.json", {
        "contract": contract, "input_sha256": frozen, "commands": commands(binary_dir),
        "source_calculated_plan_not_measurement": calculated,
    })
    lock = HERE / ".guard.lock"
    result = {"admitted_for_he": False, "performance_claim": False, "receipt_directory": str(receipt.relative_to(ROOT))}
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        result.update(status="blocked_prelaunch", reason="probe_guard_lock_exists", child_launched=False)
        guard["write_json"](receipt / "result.json", result)
        print(json.dumps(result, indent=2))
        return 78
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(str(os.getpid()))
        binary_dir.mkdir(mode=0o700, parents=True)
        for phase in ("compile", "run"):
            phase_dir = receipt / phase
            phase_dir.mkdir(mode=0o700)
            command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--child", phase, "--binary-dir", str(binary_dir)]
            phase_result = guard["guarded"](
                command, phase_dir, seconds=limits[phase + "_seconds"],
                owner_cap=limits[phase + "_owner_ceiling_bytes"], reserve=limits["host_reserve_bytes"],
                disk_required=limits["disk_reserve_bytes"] + limits["disk_allocation_bytes"],
            )
            guard["write_json"](phase_dir / "result.json", phase_result)
            result[phase] = phase_result
            result["status"] = phase_result["status"]
            if phase_result["status"] != "passed":
                break
        result["input_sha256_after"] = input_hashes()
        if result["input_sha256_after"] != frozen:
            result.update(status="failed_input_drift", reason="frozen_inputs_changed")
        if result["status"] == "passed":
            result["binary_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in binary_dir.iterdir() if p.is_file()}
        guard["write_json"](receipt / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 78
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
