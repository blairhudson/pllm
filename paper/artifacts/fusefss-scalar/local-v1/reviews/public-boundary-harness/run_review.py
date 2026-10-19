"""One-job bounded review runner; all artifacts remain in this directory."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = [
    "crates/pllm-garble/src/fusefss_backend/mod.rs",
    "crates/pllm-garble/src/fusefss_backend/compiled.rs",
    "crates/pllm-garble/src/fusefss_backend/dcf.rs",
    "crates/pllm-garble/src/shared_affine_lookup_reference/ring.rs",
    "crates/pllm-garble/src/interval_fss.rs",
    "crates/pllm-garble/src/compact_dcf.rs",
    "crates/pllm-garble/src/shared_wide_product/ring.rs",
    "crates/pllm-garble/src/fusefss/mod.rs",
    "crates/pllm-garble/src/fusefss/compile.rs",
    "crates/pllm-garble/src/fusefss/fixture.rs",
    "paper/reproduction-contracts/fusefss/gelu-table.json",
    "paper/reproduction-contracts/fusefss/source-lock.json",
]


def hashes():
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCES}


def main():
    before = hashes()
    env = {**os.environ, "CARGO_BUILD_JOBS": "1", "CARGO_TARGET_DIR": str(HERE / "build")}
    command = ["cargo", "run", "--locked", "--offline", "--manifest-path", str(HERE / "Cargo.toml"),
               "--bin", "public-boundary-review"]
    started = time.monotonic()
    try:
        native = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=60)
    except subprocess.TimeoutExpired as error:
        (HERE / "results.txt").write_bytes(error.stdout or b"")
        (HERE / "build.log").write_bytes(error.stderr or b"")
        raise SystemExit("bounded native review timed out; no retry or cap increase")
    elapsed = time.monotonic() - started
    (HERE / "results.txt").write_bytes(native.stdout)
    (HERE / "build.log").write_bytes(native.stderr)
    algebra = subprocess.run([sys.executable, "-B", str(HERE / "algebra.py")], cwd=ROOT,
                             capture_output=True, timeout=10)
    (HERE / "algebra-results.json").write_bytes(algebra.stdout)
    after = hashes()
    receipt = {
        "schema": "pllm.fusefss.public-boundary-independent-review.v1",
        "native_command": command, "cargo_jobs": 1, "native_timeout_seconds": 60,
        "native_command_elapsed_seconds": round(elapsed, 6), "native_exit_code": native.returncode,
        "algebra_exit_code": algebra.returncode,
        "sources_sha256_before": before, "sources_sha256_after": after,
        "sources_stable_during_run": before == after,
        "native_results_sha256": hashlib.sha256(native.stdout).hexdigest(),
        "algebra_results_sha256": hashlib.sha256(algebra.stdout).hexdigest(),
        "private_selected_coefficients_or_mask_exposed_to_evaluators": False,
        "security_certification": False, "whole_model_qualification": False,
    }
    (HERE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("native_exit_code", "algebra_exit_code",
          "native_command_elapsed_seconds", "sources_stable_during_run")}, sort_keys=True))
    return int(native.returncode != 0 or algebra.returncode != 0 or before != after)


if __name__ == "__main__":
    raise SystemExit(main())
