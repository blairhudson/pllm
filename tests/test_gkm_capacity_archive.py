"""Negative controls for the recorded capacity evidence's qualification boundary."""

import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "paper/artifacts/gkm-sdk/decoder-capacity-v1"
SPEC = importlib.util.spec_from_file_location("gkm_sdk_archive", ARCHIVE.parent / "verify.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def payloads():
    index = json.loads((ARCHIVE / "index.json").read_bytes())
    return {row["path"]: (ARCHIVE / row["path"]).read_bytes() for row in index["files"]}


def replace_record(payloads, name, change):
    record = json.loads(payloads[name])
    change(record)
    payloads[name] = json.dumps(record).encode()


def test_capacity_archive_reconciles_without_checkpoint_execution():
    result = AUDIT.verify(ARCHIVE)
    assert result["installed_tests_passed"] == 640
    assert result["native_tests_passed"] == 134
    assert result["generated_capacity_checked_float32_values"] == 20376
    assert result["checkpoint_execution"] == "host-admission-blocked"


def test_memory_rejection_cannot_be_promoted_to_execution(payloads):
    replace_record(payloads, "qwen-smoke-v2/preflight.json", lambda value: value.update(admitted=True))
    with pytest.raises(ValueError, match="checkpoint rejection"):
        AUDIT.verify_capacity(payloads)


def test_omitted_worker_peak_cannot_supply_admission(payloads):
    def omit(value):
        value["estimate"]["provider_peak_bytes"].pop("worker_b")
    replace_record(payloads, "qwen-smoke-v2/preflight.json", omit)
    with pytest.raises(ValueError, match="memory accounting"):
        AUDIT.verify_capacity(payloads)


def test_source_lock_substitution_rejects(payloads):
    def substitute(value):
        value["files"][0]["sha256"] = "0" * 64
    replace_record(payloads, "qwen-smoke-v2/source-lock.json", substitute)
    with pytest.raises(ValueError, match="checkpoint source"):
        AUDIT.verify_capacity(payloads)


def test_generated_parity_cannot_omit_one_full_result(payloads):
    tree = ET.fromstring(payloads["tests/results.xml"])
    for case in tree.iter("testcase"):
        if "capacity_real_workers" in case.get("name", ""):
            case.find("properties").remove(case.find("properties/property"))
            break
    payloads["tests/results.xml"] = ET.tostring(tree)
    with pytest.raises(ValueError, match="oracle count missing"):
        AUDIT.verify_capacity(payloads)


def test_preflight_script_substitution_rejects(payloads):
    payloads["qwen-smoke-v2/runner.py"] += b"\n# changed\n"
    with pytest.raises(ValueError, match="runner changed"):
        AUDIT.verify_capacity(payloads)
