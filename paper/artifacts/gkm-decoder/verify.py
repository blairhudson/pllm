"""Read-only conformance archive checks; does not execute models or benchmarks."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(__file__).resolve().parent / "local-v1"
FORMATTED_ARCHIVE = ARCHIVE.with_name("local-v2")
FINAL_ARCHIVE = ARCHIVE.with_name("local-v3")


def read_regular(path, expected=None):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("linked archive path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 << 20:
            raise ValueError("archive file type or size")
        if expected is not None and info.st_size != expected:
            raise ValueError("archive size mismatch")
        data = handle.read(info.st_size + 1)
        if len(data) != info.st_size:
            raise ValueError("archive file changed")
        return data


def cases(raw):
    result = {}
    for case in ET.fromstring(raw).iter("testcase"):
        key = (case.get("classname"), case.get("name"))
        if not all(key) or key in result:
            raise ValueError("invalid test identity")
        result[key] = (
            "failed"
            if case.find("failure") is not None or case.find("error") is not None
            else "skipped"
            if case.find("skipped") is not None
            else "passed"
        )
    return result


def counts(result):
    return tuple(list(result.values()).count(kind) for kind in ("passed", "failed", "skipped"))


def verify(archive=ARCHIVE):
    index = json.loads(read_regular(archive / "index.json"))
    if index["schema"] not in {
        "pllm.gkm_decoder_archive.v1",
        "pllm.gkm_decoder_archive.v2",
        "pllm.gkm_decoder_archive.v3",
    }:
        raise ValueError("unknown archive")
    if (
        not 1 <= len(index["files"]) <= 128
        or sum(row["bytes"] for row in index["files"]) > 16 << 20
    ):
        raise ValueError("archive budget")
    payloads = {}
    for row in index["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or row["path"] in payloads:
            raise ValueError("invalid archive path")
        data = read_regular(archive / relative, row["bytes"])
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"archive hash mismatch: {relative}")
        payloads[row["path"]] = data
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if actual != set(payloads) | {"index.json", "README.md"}:
        raise ValueError("archive manifest incomplete")

    build = json.loads(payloads["build/receipt.json"])
    if (
        build["status"] != "passed"
        or build["source_changed"]
        or build["observers_changed"]
        or build["exit_code"]
        or build["cleanup_errors"]
        or build["stop_reason"]
        or build["max_swap_used_growth_bytes"]
        or build["max_swapout_growth_bytes"]
        or build["swap_counter_source"] != "darwin.HOST_VM_INFO64.rev1.swapouts"
    ):
        raise ValueError("build receipt rejected")
    before = json.loads(payloads["build/sources-before.json"])
    after = json.loads(payloads["build/sources-after.json"])
    if before != after:
        raise ValueError("build source identity changed")
    for name, data in payloads.items():
        if name.startswith("sources/"):
            original = name.removeprefix("sources/")
            if before["files"][original] != hashlib.sha256(data).hexdigest():
                raise ValueError("frozen source differs from built source")
    for name, digest in json.loads(payloads["build/observer-hashes.json"]).items():
        if hashlib.sha256(payloads["build/" + name]).hexdigest() != digest:
            raise ValueError("observer binding changed")

    repaired = index["schema"] == "pllm.gkm_decoder_archive.v3"
    if index["schema"] == "pllm.gkm_decoder_archive.v1":
        decoder = cases(payloads["build/decoder-tests.xml"])
        initial = cases(payloads["build/shared-executor-tests.xml"])
        repair = cases(payloads["build/shared-executor-repair-tests.xml"])
        if (
            counts(decoder) != (19, 0, 0)
            or counts(initial) != (151, 4, 2)
            or counts(repair) != (7, 0, 2)
        ):
            raise ValueError("recorded test results changed")
        if set(repair) - set(initial) or counts(initial | repair) != (155, 0, 2):
            raise ValueError("shared-executor failures not resolved")
    else:
        result = cases(
            payloads["build/installed-tests.xml" if repaired else "build/installed-tests-retry.xml"]
        )
        decoder = {
            key: value for key, value in result.items() if key[0] == "tests.test_delegated_decoder"
        }
        shared = {key: value for key, value in result.items() if key not in decoder}
        if counts(decoder) != (51 if repaired else 19, 0, 0) or counts(shared) != (155, 0, 2):
            raise ValueError("final installed-wheel checks incomplete")
    native = re.findall(
        rb"test result: ok\. (\d+) passed; 0 failed;",
        payloads["native/native-tests.log" if repaired else "build/native-tests.log"],
    )
    if native != [b"14", b"22", b"5" if repaired else b"4"]:
        raise ValueError("native checks incomplete")

    if repaired:
        native_receipt = json.loads(payloads["native/receipt.json"])
        if native_receipt != {
            "status": "passed",
            "exit_codes": {"model-admission-tests": 0, "native-tests": 0, "clippy": 0},
            "source_changed": False,
        }:
            raise ValueError("native admission checks incomplete")
        for name in ("native/sources-before.json", "native/sources-after.json"):
            if json.loads(payloads[name]) != before:
                raise ValueError("native checks differ from built source")
        model_tests = re.findall(
            rb"test result: ok\. (\d+) passed; 0 failed;",
            payloads["native/model-admission-tests.log"],
        )
        if model_tests != [b"3"]:
            raise ValueError("source expansion checks incomplete")
        test_receipt = json.loads(payloads["build/test-receipt.json"])
        test_before = json.loads(payloads["build/test-sources-before.json"])
        test_after = json.loads(payloads["build/test-sources-after.json"])
        if (
            test_receipt["exit_code"]
            or test_receipt["test_sources_changed"]
            or test_before != test_after
        ):
            raise ValueError("installed tests changed or failed")
        for name, data in payloads.items():
            if name.startswith("tests/"):
                original = name.removeprefix("tests/")
                observed = before["files"] if original.startswith("crates/") else test_before
                if observed[original] != hashlib.sha256(data).hexdigest():
                    raise ValueError("test snapshot differs from executed input")

    installed = json.loads(payloads["build/installed-postcheck.json"])
    if repaired and json.loads(payloads["build/installed-precheck.json"]) != installed:
        raise ValueError("installed package changed during tests")
    wheel = ROOT / index["retained_wheel"]
    expected = build["wheel_sha256"][wheel.name]
    if installed["wheel_sha256"] != expected or len(installed["files"]) != 356:
        raise ValueError("installed package identity changed")
    for original, item in installed["files"].items():
        source = "sources/python/" + original
        if source in payloads and hashlib.sha256(payloads[source]).hexdigest() != item["sha256"]:
            raise ValueError("installed source differs from frozen source")
    if wheel.exists():
        with wheel.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                raise ValueError("retained wheel changed")
        with zipfile.ZipFile(wheel) as packed:
            members = {
                n for n in packed.namelist() if n.startswith("pllm/") and not n.endswith("/")
            }
            if members != set(installed["files"]):
                raise ValueError("installed wheel file set changed")
            for name, item in installed["files"].items():
                data = packed.read(name)
                if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise ValueError("installed/wheel mismatch")
    return {
        "verified_files": len(payloads),
        "native_tests": 44 if repaired else 40,
        "decoder_tests": 51 if repaired else 19,
        "shared_executor_tests_passed": 155,
        "optional_real_checkpoint_skips": 2,
        "retained_wheel_checked": wheel.exists(),
        "scope": "recorded local conformance consistency; no workload or resource measurement",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {path.name: verify(path) for path in (ARCHIVE, FORMATTED_ARCHIVE, FINAL_ARCHIVE)},
            indent=2,
        )
    )
