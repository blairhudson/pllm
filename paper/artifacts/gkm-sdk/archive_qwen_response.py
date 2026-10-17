"""Retain an original Qwen response and post-run source/installed-byte checks."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pllm
import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ORIGINALS = (
    "benchmark.json", "execution.log", "experiment.json", "inputs.json",
    "installed-precheck.json", "native-contract.json", "preflight.json",
    "runner.py", "runtime-contract.json", "source-lock.json",
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "qwen-response-v1")
    args = parser.parse_args()
    payloads = {name: (args.source / name).read_bytes() for name in ORIGINALS}
    conformance = HERE / "decoder-capacity-v1"
    index = json.loads((conformance / "index.json").read_bytes())
    wheel = ROOT / index["retained_wheel"]
    site = Path(pllm.__file__).resolve().parent.parent
    records = {}
    with zipfile.ZipFile(wheel) as packed:
        for name in sorted(packed.namelist()):
            if not name.startswith("pllm/") or name.endswith("/"):
                continue
            actual = (site / name).read_bytes()
            if actual != packed.read(name):
                raise ValueError("installed bytes differ: " + name)
            records[name] = {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()}
    actual_names = {
        "pllm/" + path.relative_to(site / "pllm").as_posix()
        for path in (site / "pllm").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual_names != set(records):
        raise ValueError("installed package member set differs")
    with wheel.open("rb") as stream:
        wheel_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    installed = {"files": records, "wheel_sha256": wheel_digest}
    if installed != json.loads(payloads["installed-precheck.json"]):
        raise ValueError("installed package changed during response")
    payloads["installed-postcheck.json"] = encoded(installed)

    experiment = json.loads(payloads["experiment.json"])
    source = Path(experiment["pipeline"]["model"]["source"])
    lock = json.loads(payloads["source-lock.json"])
    for row in lock["files"]:
        path = source / row["path"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != row["size"] or digest != row["sha256"]:
            raise ValueError("checkpoint changed: " + row["path"])
    payloads["source-lock-postcheck.json"] = encoded(lock)

    # This checks present processes, not historical PID ownership or attestation.
    prefix = [sys.executable, "-I", "-m", "pllm.runtime.delegated_worker"]
    live = []
    inaccessible = 0
    for process in psutil.process_iter():
        try:
            if process.cmdline()[:4] == prefix:
                live.append(process.pid)
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            inaccessible += 1
    if live:
        raise ValueError("delegated worker from the measured environment remains live")
    script = Path(__file__).read_bytes()
    payloads["archive.py"] = script
    payloads["capture.json"] = encoded({
        "schema": "pllm.gkm_qwen_response_capture.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(script).hexdigest(),
        "python": sys.executable,
        "site_packages": str(site),
        "live_worker_pids": live,
        "inaccessible_processes": inaccessible,
        "process_check_scope": "post-run process table, exact interpreter/module; not historical PID attestation",
        "original_directory": str(args.source.resolve()),
    })
    manifest = {
        "schema": "pllm.gkm_qwen_response_archive.v1",
        "conformance_archive": "decoder-capacity-v1",
        "conformance_index_sha256": hashlib.sha256((conformance / "index.json").read_bytes()).hexdigest(),
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(payloads.items())
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    (args.output / "index.json").write_bytes(encoded(manifest))
    print(json.dumps({"archive": str(args.output), "files": len(payloads),
                      "installed_files": len(records), "live_workers": live}))


if __name__ == "__main__":
    main()
