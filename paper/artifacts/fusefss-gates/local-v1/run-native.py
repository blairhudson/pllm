"""Fresh bounded gate qualification; no checkpoints, wheels or serving roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[4]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT / "target")
    # Every attempt gets a fresh directory. An earlier success cannot stand in
    # for a rejected, interrupted or incomplete invocation.
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, CARGO_BUILD_JOBS="1", PYTHONDONTWRITEBYTECODE="1")
    for name in tuple(env):
        if name.startswith("PLLM_FUSEFSS_"):
            del env[name]

    def input_files() -> list[Path]:
        files = {ROOT / p for p in ("Cargo.toml", "Cargo.lock", "crates/pllm-garble/Cargo.toml")}
        files.update((ROOT / "crates/pllm-garble/src").rglob("*.rs"))
        files.update(
            p
            for p in (ROOT / "paper/reproduction-contracts/fusefss").rglob("*")
            if p.is_file() and p.suffix in (".py", ".json")
        )
        # The gate reference imports the shared mathematical oracle outside its
        # own directory. Archive the exact dependencies already pinned by its
        # source lock, and recheck them before/after the complete invocation.
        lock = json.loads(
            (ROOT / "paper/reproduction-contracts/fusefss/gate-set/source-lock.json").read_bytes()
        )
        for name, expected in lock["local_sha256"].items():
            path = (ROOT / name).resolve()
            path.relative_to(ROOT)
            if sha(path.read_bytes()) != expected:
                raise RuntimeError(f"changed source-locked oracle dependency: {name}")
            files.add(path)
        files.add(Path(__file__).resolve())
        return sorted(files)

    def inventory() -> dict:
        return {
            str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
            for p in input_files()
        }

    before = inventory()
    source = {
        "base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "rustc": subprocess.check_output(["rustc", "-Vv"], cwd=ROOT, text=True),
        "files": before,
    }
    (out / "source-inventory.json").write_text(json.dumps(source, indent=2, sort_keys=True) + "\n")
    with tarfile.open(out / "inputs.tar.gz", "w:gz") as archive:
        for path in input_files():
            archive.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    receipt = {
        "schema": "pllm.fusefss.gates-validation.v1",
        "status": "running",
        "source_unchanged": False,
        "steps": [],
    }

    def save() -> None:
        pending = out / "run-receipt.json.pending"
        pending.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        pending.replace(out / "run-receipt.json")

    def run(
        name: str, command: list[str], *, overrides: dict | None = None, timeout: int = 1800
    ) -> None:
        started = time.monotonic()
        path = out / f"{name}.log"
        with path.open("xb") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=dict(env, **(overrides or {})),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise
        entry = {
            "name": name,
            "command": command,
            "returncode": code,
            "expected_returncode": 0,
            "environment_overrides": overrides or {},
            "elapsed_seconds": time.monotonic() - started,
            "log_sha256": sha(path.read_bytes()),
        }
        receipt["steps"].append(entry)
        save()
        print(json.dumps(entry), flush=True)
        if code:
            raise RuntimeError(f"{name}: exit {code}; see {path}")

    save()
    try:
        run(
            "clippy",
            [
                "cargo",
                "clippy",
                "--locked",
                "--offline",
                "-p",
                "pllm-garble",
                "-p",
                "pllm-python",
                "--all-targets",
                "--",
                "-D",
                "warnings",
                "-A",
                "clippy::needless_range_loop",
                "-A",
                "clippy::nonminimal_bool",
                "-A",
                "clippy::assertions_on_constants",
            ],
        )
        native = [
            "cargo",
            "test",
            "--release",
            "--locked",
            "--offline",
            "-p",
            "pllm-garble",
            "--lib",
        ]
        overrides = {"PLLM_FUSEFSS_EXPORT_GATE_SET": "1", "PLLM_FUSEFSS_F2_EXPORT_DIR": str(out)}
        for name, test in (
            ("selected-native", "export_selected_gates"),
            ("tables-native", "export_tables"),
        ):
            run(
                name,
                [
                    *native,
                    f"fusefss_backend::gate_conformance::{test}",
                    "--",
                    "--ignored",
                    "--exact",
                    "--test-threads=1",
                ],
                overrides=overrides,
                timeout=3600,
            )
        base = "paper/reproduction-contracts/fusefss/gate-set/"
        run(
            "checker-selected",
            [
                sys.executable,
                "-B",
                base + "check_outputs.py",
                "--gates",
                "shift,b2a,compare,phi-select,external",
                str(out / "selected-gates.jsonl"),
            ],
            timeout=120,
        )
        run(
            "checker-tables",
            [sys.executable, "-B", base + "check_tables.py", str(out / "tables.jsonl")],
            timeout=120,
        )
        run("native-library", [*native, "--", "--test-threads=1"], timeout=2700)
        for name, directory in (
            ("frozen-reference", "paper/reproduction-contracts/fusefss"),
            ("independent-fixture", "paper/reproduction-contracts/fusefss/native-conformance"),
            ("gate-set-fixture", "paper/reproduction-contracts/fusefss/gate-set"),
        ):
            run(
                name,
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    directory,
                    "-p",
                    "test_*.py",
                    "-v",
                ],
                timeout=180,
            )
        run("format", ["cargo", "fmt", "-p", "pllm-garble", "--", "--check"], timeout=120)
        receipt["source_unchanged"] = inventory() == before
        if not receipt["source_unchanged"]:
            raise RuntimeError("qualification inputs changed during execution")
        receipt["status"] = "pass"
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure_type"] = type(error).__name__
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
