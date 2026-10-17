"""Read-only consistency checks for bounded SDK conformance evidence."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent / "local-v1"
EXPECTED_NATIVE = {
    "compiler-tests": [14, 22, 5, 3, 4], "core-tests": [42],
    "wire-tests": [4, 2, 10, 1, 2], "ownership-tests": [8],
}
STREAM_NATIVE = {
    **EXPECTED_NATIVE, "compiler-tests": [14, 22, 5, 3, 6], "source-import-tests": [1],
}
CASES = {
    "pllm.gkm_sdk_archive.v1": (547, EXPECTED_NATIVE),
    "pllm.gkm_sdk_archive.v2": (571, STREAM_NATIVE),
}
spec = importlib.util.spec_from_file_location(
    "gkm_decoder_archive", ARCHIVE.parent.parent / "gkm-decoder/verify.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def verify(archive=ARCHIVE):
    index = json.loads(common.read_regular(archive / "index.json"))
    if index["schema"] not in CASES:
        raise ValueError("unknown SDK archive")
    expected_python, expected_native = CASES[index["schema"]]
    rows = index["files"]
    if not 1 <= len(rows) <= 256 or sum(row["bytes"] for row in rows) > 32 << 20:
        raise ValueError("archive exceeds its bound")
    payloads = {}
    for row in rows:
        name = row["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in payloads:
            raise ValueError("invalid archive member")
        data = common.read_regular(archive / relative, row["bytes"])
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("archive member changed")
        payloads[name] = data
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if actual != set(payloads) | {"index.json"}:
        raise ValueError("archive manifest incomplete")

    def record(name):
        return json.loads(payloads[name])

    build = record("build/receipt.json")
    if (
        build["status"] != "passed" or not build["cached_build"]
        or build["exit_code"] or build["stop_reason"] or build["cleanup_errors"]
        or build["source_changed"] or build["observers_changed"]
        or build["max_swap_used_growth_bytes"] or build["max_swapout_growth_bytes"]
        or build["swap_counter_source"] != "darwin.HOST_VM_INFO64.rev1.swapouts"
    ):
        raise ValueError("build did not qualify")
    source = record("build/sources-before.json")
    if source != record("build/sources-after.json"):
        raise ValueError("build inputs changed")
    for name, digest in record("build/observer-hashes.json").items():
        if hashlib.sha256(payloads["build/" + name]).hexdigest() != digest:
            raise ValueError("build observer changed")

    test_sources = record("tests/sources-before.json")
    if (test_sources != record("tests/sources-after.json")
            or record("tests/receipt.json") != {"exit_code": 0, "test_sources_changed": False}):
        raise ValueError("installed test inputs changed or failed")
    result = common.cases(payloads["tests/results.xml"])
    passed, failed, skipped = common.counts(result)
    if failed or skipped or passed != expected_python or index["installed_tests_passed"] != expected_python:
        raise ValueError("installed conformance checks incomplete")
    required = {
        "tests.test_delegated_source", "tests.test_gkm_sdk_declarations",
        "tests.test_delegated_client", "tests.test_delegated_benchmark",
        "tests.test_gkm_benchmark_admission", "tests.test_delegated_decoder",
        "tests.test_delegated_lifecycle_regressions", "tests.test_delegated_transport_binding",
        "tests.test_delegated_sdk", "tests.test_delegated_interfaces",
        "tests.test_sidecar_context_lifetime", "tests.test_delegated_transport_ownership",
    }
    if index["schema"] == "pllm.gkm_sdk_archive.v2":
        required.add("tests.test_delegated_source_streaming")
    if not required <= {name.split(".Test")[0] for name, _ in result}:
        raise ValueError("required conformance suite absent")

    native = record("native/receipt.json")
    jobs = dict.fromkeys((*expected_native, "clippy"), 0)
    if native != {"exit_codes": jobs, "source_changed": False}:
        raise ValueError("native checks incomplete")
    native_sources = record("native/sources-before.json")
    if native_sources != record("native/sources-after.json"):
        raise ValueError("native source changed")
    if any(source["files"].get(name) != digest for name, digest in native_sources.items()):
        raise ValueError("native suite differs from wheel build")
    native_counts = {
        name: [int(n) for n in re.findall(rb"test result: ok\. (\d+) passed; 0 failed;",
                                        payloads[f"native/{name}.log"])]
        for name in jobs if name != "clippy"
    }
    if native_counts != expected_native or index["native_test_groups"] != expected_native:
        raise ValueError("native test groups changed")
    for name, data in payloads.items():
        if name.startswith("sources/"):
            original = name.removeprefix("sources/")
            expected = test_sources.get(original, source["files"].get(original))
            if expected != hashlib.sha256(data).hexdigest():
                raise ValueError("frozen source differs from executed input")
    if any("sources/" + name not in payloads for name in test_sources):
        raise ValueError("executed test source not retained")

    installed = record("build/installed-precheck.json")
    if installed != record("build/installed-postcheck.json"):
        raise ValueError("installed package changed during checks")
    wheel_path = Path(index["retained_wheel"])
    if wheel_path.is_absolute() or ".." in wheel_path.parts:
        raise ValueError("invalid wheel reference")
    expected = build["wheel_sha256"][wheel_path.name]
    if installed["wheel_sha256"] != expected or len(installed["files"]) != 362:
        raise ValueError("installed package identity differs")
    for name, row in installed["files"].items():
        frozen = payloads.get("sources/python/" + name)
        if frozen is not None and hashlib.sha256(frozen).hexdigest() != row["sha256"]:
            raise ValueError("frozen Python differs from installed package")
    wheel = ROOT / wheel_path
    if wheel.exists():
        with wheel.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError("retained wheel changed")
        with zipfile.ZipFile(wheel) as packed:
            names = {n for n in packed.namelist() if n.startswith("pllm/") and not n.endswith("/")}
            if names != set(installed["files"]):
                raise ValueError("wheel member set differs")
            for name, row in installed["files"].items():
                data = packed.read(name)
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("installed bytes differ from wheel")
    return {
        "verified_files": len(payloads), "installed_tests_passed": passed,
        "native_tests_passed": sum(sum(values) for values in native_counts.values()),
        "clippy_passed": True, "installed_files": len(installed["files"]),
        "retained_wheel_checked": wheel.exists(),
        "scope": "local recorded conformance; cached build; no checkpoint performance qualification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=("local-v1", "source-stream-v1"), default="local-v1")
    args = parser.parse_args()
    print(json.dumps(verify(ARCHIVE.parent / args.archive), indent=2, sort_keys=True))
