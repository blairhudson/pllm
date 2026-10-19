"""Bounded local scalar qualification; no checkpoints, wheels or serving roles."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path.cwd()
OUT = ROOT / "target/fusefss-final-v1"
OUT.mkdir(exist_ok=True)
BASE_ENV = dict(os.environ, CARGO_BUILD_JOBS="1", PYTHONDONTWRITEBYTECODE="1")
for name in ("PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE", "PLLM_FUSEFSS_EXPORT_PROTECTED_GELU"):
    BASE_ENV.pop(name, None)
FILES = [ROOT / "Cargo.toml", ROOT / "Cargo.lock", ROOT / "crates/pllm-garble/Cargo.toml"]
FILES += sorted((ROOT / "crates/pllm-garble/src").rglob("*.rs"))
FILES += sorted(p for p in (ROOT / "paper/reproduction-contracts/fusefss").rglob("*") if p.is_file() and p.suffix in (".py", ".json"))

def inventory():
    return {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}

before = inventory()
receipt = {"schema": "pllm.fusefss.scalar-validation.v1", "status": "running", "steps": [], "source_unchanged": False}
(OUT / "source-inventory.json").write_text(json.dumps({"base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "rustc": subprocess.check_output(["rustc", "-Vv"], text=True), "files": before}, indent=2, sort_keys=True) + "\n")
with tarfile.open(OUT / "inputs.tar.gz", "w:gz") as archive:
    for path in FILES:
        archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)

def save():
    (OUT / "run-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

def run(name, command, *, expected=0, env=None, timeout=1800):
    started = time.monotonic()
    with (OUT / f"{name}.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=dict(BASE_ENV, **(env or {})), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    entry = {"name": name, "command": command, "environment_overrides": env or {}, "returncode": code, "expected_returncode": expected, "elapsed_seconds": time.monotonic() - started}
    receipt["steps"].append(entry)
    save()
    if code != expected:
        raise RuntimeError(f"{name}: exit {code}, expected {expected}; see log")
    print(json.dumps(entry), flush=True)
    return (OUT / f"{name}.log").read_text()

try:
    export = ["cargo", "test", "--release", "--locked", "--offline", "-p", "pllm-garble", "--lib", "fusefss_backend::conformance::export_public_boundary_frozen_gelu", "--", "--ignored", "--exact", "--test-threads=1", "--nocapture"]
    stale = ROOT / "target/fusefss-native-protected-v1"
    assert all((stale / name).is_file() for name in ("protected-receipt.json", "runtime-receipt.json")), "stale-success control needs the archived prior receipts"
    failure = run("stale-receipt-negative", export, expected=101)
    assert "test result: FAILED. 0 passed; 1 failed" in failure
    assert not any((stale / name).exists() for name in ("protected-receipt.json", "runtime-receipt.json"))
    receipt["stale_receipt_guard"] = "both prior success receipts removed before rejected opt-in"
    run("protected-native", export, env={"PLLM_FUSEFSS_EXPORT_PROTECTED_GELU": "1"})
    compiler = export.copy()
    compiler[compiler.index("fusefss_backend::conformance::export_public_boundary_frozen_gelu")] = "fusefss::conformance::export_compiler_conformance_all"
    run("compiler-native", compiler, env={"PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE": "1"})
    checker = "paper/reproduction-contracts/fusefss/native-conformance/check_native.py"
    for suite, path in (("all", "target/fusefss-native-conformance-v1/compiler.jsonl"), ("gelu", "target/fusefss-native-protected-v1/protected.jsonl")):
        text = run(f"checker-{suite}", [sys.executable, "-B", checker, "--suite", suite, path], timeout=60)
        assert json.loads(text)["status"] == "pass"
    run("native-library", ["cargo", "test", "--release", "--locked", "--offline", "-p", "pllm-garble", "--lib", "--", "--test-threads=1"], timeout=2700)
    for name, directory in (("frozen-reference", "paper/reproduction-contracts/fusefss"), ("independent-fixture", "paper/reproduction-contracts/fusefss/native-conformance")):
        run(name, [sys.executable, "-B", "-m", "unittest", "discover", "-s", directory, "-p", "test_*.py", "-v"], timeout=120)
    run("format", ["cargo", "fmt", "-p", "pllm-garble", "--", "--check"], timeout=120)
    receipt["source_unchanged"] = inventory() == before
    assert receipt["source_unchanged"]
    receipt["status"] = "pass"
except BaseException as error:
    receipt["status"] = "failed"
    receipt["failure_type"] = type(error).__name__
    raise
finally:
    save()
