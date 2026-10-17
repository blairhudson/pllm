"""Read-only integrity/binding audit; does not rerun builds, scans or inference."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent / "2026-10-17-installed-v6"


def verify():
    index = json.loads((ARCHIVE / "index.json").read_text())
    if index["schema"] != "pllm.gkm_installed_source_archive.v1":
        raise ValueError("unknown archive")
    payloads = {}
    for row in index["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in payloads:
            raise ValueError("invalid or duplicate archive path")
        path = ARCHIVE / relative
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError("linked archive path")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"archive bytes changed: {relative}")
        payloads[row["path"]] = raw
    actual = {path.relative_to(ARCHIVE).as_posix() for path in ARCHIVE.rglob("*") if path.is_file()}
    if actual != set(payloads) | {"index.json", "README.md"}:
        raise ValueError("unindexed archive file")
    build = json.loads(payloads["build/receipt.json"])
    final = json.loads(payloads["scan/finalization.json"])
    report = json.loads(payloads["scan/report.json"])
    raw_report = payloads["scan/report.json"]
    if final["report_artifact"] != {"bytes": len(raw_report), "sha256": hashlib.sha256(raw_report).hexdigest()}:
        raise ValueError("final receipt/report mismatch")
    if (build["status"] != "passed" or build["source_changed"] or build["observers_changed"]
            or build["cleanup_errors"] or build["max_swap_used_growth_bytes"]
            or build["max_swapout_growth_bytes"]):
        raise ValueError("build not qualified")
    if (final["status"] != "complete" or final["source_numeric_admission"] != "pass"
            or not final["guarded_report_window_passed"] or not report["immutable_inputs_verified"]
            or final["complete_gkm_admission"] != "blocked" or report["model_inference_executed"]):
        raise ValueError("source-only scan not qualified")
    guard = final["resource_guard_after_report_serialization"]
    for key in ("swap_used_bytes", "swapout_bytes", "counter_source"):
        if guard["baseline"][key] != guard["last"][key]:
            raise ValueError("swap observation changed")
    for original, digest in report["probe_and_contract_hashes"].items():
        suffix = original.split("/workspace/pllm/", 1)[1]
        if hashlib.sha256(payloads["sources/" + suffix]).hexdigest() != digest:
            raise ValueError("source/contract binding changed")
    suites = ET.fromstring(payloads["build/inventory-tests-v3.xml"]).findall("testsuite")
    if sum(int(suite.get("tests")) for suite in suites) != 67 or any(
            int(suite.get(key)) for suite in suites for key in ("errors", "failures", "skipped")):
        raise ValueError("installed suite incomplete")
    wheel = ROOT / index["retained_wheel"]
    if wheel.exists():
        if hashlib.sha256(wheel.read_bytes()).hexdigest() != build["wheel_sha256"][wheel.name]:
            raise ValueError("retained wheel changed")
        with zipfile.ZipFile(wheel) as packed:
            for name, module in report["executed_metadata"]["modules"].items():
                if name != "pllm" and not name.startswith("pllm."):
                    continue
                member = module["module_file"].split("/site-packages/", 1)[1]
                digest = hashlib.sha256(packed.read(member)).hexdigest()
                if digest != module["artifacts"][0]["sha256_before"]:
                    raise ValueError("executed module differs from wheel")
    return {"verified_files": len(payloads), "verified_bytes": sum(map(len, payloads.values())),
            "retained_wheel_checked": wheel.exists(), "installed_tests": 67,
            "source_numeric_admission": "pass", "complete_gkm_admission": "blocked",
            "scope": "recorded artifact consistency; no source/model execution"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
