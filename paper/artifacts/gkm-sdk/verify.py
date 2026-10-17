"""Read-only consistency checks for bounded SDK conformance evidence."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
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
CAPACITY_NATIVE = {
    **STREAM_NATIVE, "compiler-tests": [14, 22, 5, 5, 5, 6], "core-tests": [46],
    "wire-tests": [4, 3, 2, 10, 1, 2],
}
CASES = {
    "pllm.gkm_sdk_archive.v1": (547, EXPECTED_NATIVE),
    "pllm.gkm_sdk_archive.v2": (571, STREAM_NATIVE),
    "pllm.gkm_sdk_archive.v3": (640, CAPACITY_NATIVE),
}
spec = importlib.util.spec_from_file_location(
    "gkm_decoder_archive", ARCHIVE.parent.parent / "gkm-decoder/verify.py"
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def verify_capacity(payloads):
    """Reconcile recorded metadata rejection and generated full-result checks."""
    def record(name):
        return json.loads(payloads[name])

    def digest(domain, value):
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, separators=(",", ":")).encode()
        return hashlib.sha256(domain.encode() + b"\0" + raw).hexdigest()

    prefix = "qwen-smoke-v2/"
    inputs = record(prefix + "inputs.json")
    if hashlib.sha256(payloads[prefix + "runner.py"]).hexdigest() != inputs["script_sha256"]:
        raise ValueError("checkpoint preflight runner changed")
    decoder = record(prefix + "native-contract.json")
    runtime = record(prefix + "runtime-contract.json")
    lock = record(prefix + "source-lock.json")
    preflight = record(prefix + "preflight.json")
    estimate = preflight["estimate"]
    admission = estimate["gkm_admission"]
    if (runtime["decoder"] != decoder or len(decoder["matrices"]) != 97
            or len(decoder["source_shapes"]) != 290 or len(decoder["local_tensors"]) != 121
            or decoder["vocabulary"] != 151936
            or decoder["decoder_resources"]["profile"] != "pllm.gkm.decoder-capacity.v1"
            or decoder["decoder_resources"]["source_import_bytes"] != 1537852928
            or decoder["decoder_resources"]["source_chunk_bytes"] != 65536):
        raise ValueError("checkpoint capacity contract differs")
    if (lock["checkpoint_digest"] != digest("pllm.checkpoint.v1", {"files": lock["files"]})
            or runtime["source_lock"] != digest("pllm.model_source_lock.v1", lock)
            or admission["checkpoint_digest"] != lock["checkpoint_digest"]
            or admission["source_lock_digest"] != runtime["source_lock"]
            or admission["tokenizer_digest"] != runtime["tokenizer"]
            or estimate["pipeline_digest"] != runtime["composition"]):
        raise ValueError("checkpoint source or composition differs")
    files = {row["path"]: row for row in lock["files"]}
    if (files["model.safetensors"]["sha256"] !=
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
            or files["tokenizer.json"]["sha256"] != runtime["tokenizer"]):
        raise ValueError("checkpoint source pin differs")
    if (inputs["max_input_tokens"] != decoder["input_rows"]
            or inputs["max_output_tokens"] != decoder["decode_steps"] + 1
            or admission["max_input_tokens"] != decoder["input_rows"]
            or admission["max_output_tokens"] != decoder["decode_steps"] + 1):
        raise ValueError("checkpoint workload differs")
    required = estimate["client_peak_bytes"] + sum(estimate["provider_peak_bytes"].values())
    host = preflight["host"]
    candidate = preflight["candidates"]["native"]
    if (preflight["admitted"] is not False or candidate["admitted"] is not False
            or preflight["selected_backend"] is not None
            or required != estimate["native_total_peak_bytes"]
            or required != candidate["required_host_bytes"]
            or host["admission_budget_bytes"] != max(0, host["available"] - host["reserve_bytes"])
            or required <= host["admission_budget_bytes"]
            or not any("exceeds safe headroom" in reason for reason in candidate["reasons"])
            or any(name.startswith(prefix) and "benchmark" in name for name in payloads)):
        raise ValueError("checkpoint rejection or memory accounting differs")
    checks = {}
    for case in ET.fromstring(payloads["tests/results.xml"]).iter("testcase"):
        name = case.get("name", "")
        if (case.get("classname") == "tests.test_delegated_capacity"
                and name.startswith("test_decoder_capacity_real_workers_preserve_full_logits_and_kv[")):
            props = case.findall("properties/property")
            if len(props) != 1 or props[0].get("name") != "checked_float32_values":
                raise ValueError("full-result oracle count missing")
            checks[name] = int(props[0].get("value"))
    expected = {
        f"test_decoder_capacity_real_workers_preserve_full_logits_and_kv[{tied}-{family}]": 3396
        for tied in ("False", "True") for family in ("qwen2", "qwen3", "llama")
    }
    if checks != expected:
        raise ValueError("generated capacity parity checks incomplete")
    return {"checkpoint_execution": "host-admission-blocked",
            "checkpoint_required_host_bytes": required,
            "checkpoint_safe_headroom_bytes": host["admission_budget_bytes"],
            "generated_capacity_checked_float32_values": sum(checks.values())}


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
    if index["schema"] in {"pllm.gkm_sdk_archive.v2", "pllm.gkm_sdk_archive.v3"}:
        required.add("tests.test_delegated_source_streaming")
    if index["schema"] == "pllm.gkm_sdk_archive.v3":
        required.update({"tests.test_delegated_capacity", "tests.test_delegated_result_lifetimes",
                         "tests.test_delegated_worker_lifetimes"})
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
        **(verify_capacity(payloads) if index["schema"] == "pllm.gkm_sdk_archive.v3" else {}),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=("local-v1", "source-stream-v1", "decoder-capacity-v1"),
                        default="local-v1")
    args = parser.parse_args()
    print(json.dumps(verify(ARCHIVE.parent / args.archive), indent=2, sort_keys=True))
