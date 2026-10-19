"""Read-only scalar evidence replay: no Rust build, keys, models or network."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LIMIT = 32 << 20


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def relative(name: str) -> Path:
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, "unsafe artifact path")
    return Path(*path.parts)


def read(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(LIMIT + 1)
    require(len(data) <= LIMIT, "artifact byte limit")
    return data


def load(name: str) -> dict:
    return json.loads(read(HERE / relative(name)))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_inputs(prefix: str) -> dict:
    inventory = load(prefix + "source-inventory.json")["files"]
    seen: set[str] = set()
    total = 0
    with tarfile.open(HERE / (prefix + "inputs.tar.gz"), "r:gz") as archive:
        for member in archive:
            relative(member.name)
            require(member.isfile() and member.name not in seen, "unexpected source entry")
            require(0 <= member.size <= LIMIT, "source entry byte limit")
            total += member.size
            require(total <= LIMIT, "source archive byte limit")
            stream = archive.extractfile(member)
            require(stream is not None, "source entry missing")
            with stream:
                data = stream.read(LIMIT + 1)
            require(
                inventory.get(member.name) == {"bytes": len(data), "sha256": sha(data)},
                f"source pin mismatch: {member.name}",
            )
            seen.add(member.name)
    require(seen == set(inventory), "source inventory coverage")
    return inventory


def main() -> dict:
    manifest = load("manifest.json")
    require(manifest["schema"] == "pllm.fusefss.scalar-evidence-files.v1", "manifest schema")
    for name, pin in manifest["files"].items():
        data = read(HERE / relative(name))
        require(pin == {"bytes": len(data), "sha256": sha(data)}, f"artifact mismatch: {name}")
    sources = check_inputs("")
    reviewed = check_inputs("initial/")
    review = load("reviews/public-boundary-receipt.json")
    require(review["sources_sha256_before"] == review["sources_sha256_after"], "review source changed")
    for name, digest in review["sources_sha256_before"].items():
        require(reviewed[name]["sha256"] == digest, f"review snapshot mismatch: {name}")
    require(review["native_exit_code"] == review["algebra_exit_code"] == 0, "review failure")
    for name, key in (
        ("reviews/public-boundary-results.txt", "native_results_sha256"),
        ("reviews/public-boundary-algebra-results.json", "algebra_results_sha256"),
    ):
        require(sha(read(HERE / name)) == review[key], "review result linkage")

    validation = load("run-receipt.json")
    require(validation["status"] == "pass" and validation["source_unchanged"] is True, "native validation incomplete")
    expected_steps = {
        "stale-receipt-negative", "protected-native", "compiler-native", "checker-all",
        "checker-gelu", "native-library", "frozen-reference", "independent-fixture", "format",
    }
    require({step["name"] for step in validation["steps"]} == expected_steps, "validation step coverage")
    require(len(validation["steps"]) == len(expected_steps), "duplicate validation step")
    for step in validation["steps"]:
        expected = 101 if step["name"] == "stale-receipt-negative" else 0
        require(step["returncode"] == expected, f"validation exit: {step['name']}")
    runtime = load("runtime-receipt.json")
    protected = load("protected-receipt.json")
    compiler = load("compiler-receipt.json")
    for source in compiler["source_identity"]["sources"]:
        require(
            sources[source["path"]]
            == {"bytes": source["bytes"], "sha256": source["sha256"]},
            f"compiler source mismatch: {source['path']}",
        )
    clippy = load("clippy.json")
    require(clippy["status"] == "pass", "lint check incomplete")
    require(runtime["adapter_output"] == protected["output"], "protected receipt linkage")
    require(runtime["backend_sources"] == protected["backend"], "protected backend linkage")
    require(runtime["final_budget_bytes"] == 0, "native budget not retired")
    for name, digest in runtime["backend_sources"]["source_sha256"].items():
        require(sources[name]["sha256"] == digest, f"backend source mismatch: {name}")
    batches = runtime["batch_counts_by_lanes"]
    require(batches == [0, 0, 8, 10880], "batch geometry")
    values = sum((i + 1) * count for i, count in enumerate(batches))
    require(values == runtime["gate_evaluations"] == 43544, "protected output coverage")
    require(runtime["logical_private_material_bytes"] == values * 2 * 1372, "private material conservation")
    require(runtime["public_local_table_copy_bytes"] == values * 2 * 12296, "public-copy conservation")
    require(runtime["actual_peer_frame_bytes"] == sum(count * (320 + 48 * (i + 1)) for i, count in enumerate(batches)), "frame conservation")
    require(max(runtime["admitted_combined_bytes_by_lanes"]) <= runtime["live_native_limit_bytes"] == 64 << 20, "live admission")

    checker = "paper/reproduction-contracts/fusefss/native-conformance/check_native.py"
    for name, pin in sources.items():
        if name.startswith("paper/reproduction-contracts/fusefss/") and name.endswith((".py", ".json")):
            require(sha(read(ROOT / relative(name))) == pin["sha256"], f"live checker/fixture changed: {name}")
    checks = {}
    for suite, filename, receipt, expected in (
        ("all", "compiler.jsonl.gz", compiler, 58400),
        ("gelu", "protected.jsonl.gz", protected, 43544),
    ):
        with gzip.open(HERE / filename, "rb") as stream:
            data = stream.read(LIMIT + 1)
        require(len(data) <= LIMIT, "decoded output byte limit")
        require(sha(data) == receipt["output"]["sha256"] and len(data) == receipt["output"]["bytes"], "original output pin")
        result = subprocess.run([sys.executable, "-B", str(ROOT / checker), "--suite", suite, "-"], input=data, capture_output=True, timeout=60, check=True)
        summary = json.loads(result.stdout)
        require(summary["status"] == "pass" and summary["checked"]["evaluations"] == expected, "independent checker failure")
        require(summary["checked"]["external_encoding_cases"] == 38, "external encoding coverage")
        require(summary["security_qualified"] is False and summary["whole_model_qualified"] is False, "qualification scope changed")
        checks[suite] = summary["checked"]
    return {"status": "pass", "checked": checks, "actual_peer_frame_bytes": runtime["actual_peer_frame_bytes"], "final_budget_bytes": 0, "scope": "native scalar evidence only"}


if __name__ == "__main__":
    try:
        print(json.dumps(main(), sort_keys=True, separators=(",", ":")))
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "reject", "reason": str(error)}), file=sys.stderr)
        raise SystemExit(1) from None
