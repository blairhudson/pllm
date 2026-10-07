"""Fresh-process, interleaved comparison of two frozen native issuance binaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

from .masked_wire_issuance import validate

SOURCES = (
    "crates/pllm-garble/src/compact_dpf.rs",
    "crates/pllm-garble/src/masked_wire_reference.rs",
    "crates/pllm-garble/src/masked_wire_reference/transport.rs",
    "crates/pllm-garble/src/masked_wire_reference/issuance_profile.rs",
    "Cargo.lock",
)
TEST = "masked_wire_reference::issuance_profile::profile_issuance_and_codecs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, destination):
    result = subprocess.run(
        ["cargo", "test", "--release", "--locked", "-p", "pllm-garble", "--lib",
         "--no-run", "--message-format=json"], cwd=root,
        env=dict(os.environ, CARGO_BUILD_JOBS="2"), capture_output=True, text=True, check=True,
    )
    executable = next(json.loads(line)["executable"] for line in result.stdout.splitlines()
                      if line.startswith("{") and json.loads(line).get("executable"))
    destination.mkdir()
    shutil.copy2(executable, destination / "pllm-garble-tests")
    metadata = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "executable_sha256": digest(destination / "pllm-garble-tests"),
        "source_sha256": {path: digest(root / path) for path in SOURCES},
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def measure(directory):
    result = subprocess.run([str(directory / "pllm-garble-tests"), TEST, "--exact", "--ignored",
                             "--nocapture", "--test-threads=1"],
                            capture_output=True, text=True, check=True, timeout=180)
    marker = "PLLM_ISSUANCE_PROFILE="
    report = json.loads(next(line.split(marker, 1)[1] for line in result.stdout.splitlines()
                             if marker in line))
    validate(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-control", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.save_control:
        if args.control or args.output:
            parser.error("save-control is separate from a comparison")
        freeze(root, args.save_control.resolve())
        return
    if args.control is None or args.output is None:
        parser.error("comparison requires control and output")
    if args.output.exists():
        raise FileExistsError(args.output)
    control = args.control.resolve()
    metadata = json.loads((control / "metadata.json").read_text())
    if digest(control / "pllm-garble-tests") != metadata["executable_sha256"]:
        raise ValueError("frozen control executable changed")
    with tempfile.TemporaryDirectory(prefix="pllm-issuance-") as temporary:
        candidate = Path(temporary) / "candidate"
        freeze(root, candidate)
        reports = []
        for pair in range(3):
            for label in (("control", "candidate") if pair % 2 == 0 else ("candidate", "control")):
                reports.append({"pair": pair, "implementation": label,
                                "report": measure(control if label == "control" else candidate)})
        report = {
            "schema": "pllm.masked_wire_issuance_comparison.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {"platform": platform.platform(), "machine": platform.machine(),
                "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip()},
            "implementations": {"control": metadata,
                "candidate": json.loads((candidate / "metadata.json").read_text())},
            "reports": reports,
            "checked_integer_outputs": sum(case["complete"]["checked_outputs"]
                for entry in reports for case in entry["report"]["cases"]),
            "scope": "six alternating fresh native processes; same public fixtures and release settings; local wall time; no HTTP, RSS or whole-response measurement",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, indent=2, sort_keys=True, allow_nan=False)
        destination.write("\n")
    print(f"Checked {report['checked_integer_outputs']} outputs in six fresh processes.")


if __name__ == "__main__":
    main()
