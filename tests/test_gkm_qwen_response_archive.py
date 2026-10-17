"""Keep the real-checkpoint smoke within its measured functionality contract."""

import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1] / "paper/artifacts/gkm-sdk"
SPEC = importlib.util.spec_from_file_location("gkm_qwen_archive", HERE / "verify_qwen_response.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.fixture
def payloads():
    return AUDIT.load_payloads(AUDIT.ARCHIVE)[1]


@pytest.fixture(scope="module")
def conformance():
    return AUDIT.load_payloads(AUDIT.CONFORMANCE)[1]


def replace_record(payloads, name, change):
    value = json.loads(payloads[name])
    change(value)
    payloads[name] = json.dumps(value).encode()


def test_original_qwen_response_reconciles_with_conformance():
    result = AUDIT.verify()
    assert result["status"] == "completed"
    assert result["installed_files"] == 362
    assert (result["input_tokens"], result["output_tokens"]) == (4, 2)
    assert result["covered_all_link_body_bytes"] == 87227888
    assert result["maximum_observed_host_swap_growth_bytes"] == 0
    assert result["decode_tps"] is None
    assert result["full_logits_kv_oracle"] is False
    assert result["quality_or_comparative_result"] is False


@pytest.mark.parametrize(("name", "change", "message"), [
    ("installed-postcheck.json", lambda value: value.update(wheel_sha256="0" * 64), "installed bytes"),
    ("source-lock-postcheck.json", lambda value: value["files"][0].update(sha256="0" * 64), "source changed"),
    ("preflight.json", lambda value: value.update(admitted=False), "host admission"),
    ("preflight.json", lambda value: value["estimate"]["provider_peak_bytes"].pop("worker_b"), "host admission"),
    ("runtime-contract.json", lambda value: value["decoder"].update(vocabulary=256), "native source"),
    ("capture.json", lambda value: value.update(live_worker_pids=[123]), "post-run capture"),
    ("execution.log", lambda value: value.update(status="failed"), "execution log"),
])
def test_source_and_admission_substitutions_reject(payloads, conformance, name, change, message):
    replace_record(payloads, name, change)
    with pytest.raises(ValueError, match=message):
        AUDIT.verify_response(payloads, conformance)


@pytest.mark.parametrize(("path", "key", "value", "message"), [
    (("execution", "build", "attributes"), "native_sha256", "0" * 64, "executed build"),
    (("execution", "details", "report", "checks"), "fixed_count_verified_delivery", False, "verified response"),
    (("execution", "details", "report", "memory_guard"), "maximum_host_swap_growth_bytes", 4096, "host pressure"),
    (("execution", "details", "report", "runs", 0, "tokens"), "output_tokens", 1, "verified response"),
    (("execution", "details", "report", "native_process_memory"), "complete", False, "simultaneous memory"),
    (("execution", "details", "report", "privacy_admission"), "independent_operators_verified", True, "beyond its oracle"),
    (("execution", "oracle"), "kind", "complete_logits_and_kv", "beyond its oracle"),
    (("execution", "details", "report", "topology_accounting", "runs", 0), "all_link_serialized_body_bytes", 0, "directed bodies"),
    (("execution", "details", "runtime_contract"), "verification_window", "none", "runtime contract"),
])
def test_runtime_evidence_cannot_be_promoted(payloads, conformance, path, key, value, message):
    def change(document):
        target = document["results"][0]
        for part in path:
            target = target[part]
        target[key] = value
    replace_record(payloads, "benchmark.json", change)
    with pytest.raises(ValueError, match=message):
        AUDIT.verify_response(payloads, conformance)


@pytest.mark.parametrize(("name", "key", "value", "message"), [
    ("response.decode_tps", "value", 1000000, "buffered emission"),
    ("client.peak_rss", "value", 128 << 20, "metric differs"),
    ("client.peak_rss", "window", "request_only", "metric differs"),
    ("client.cold_cpu", "value", float("nan"), "metric differs"),
])
def test_metric_substitutions_reject(payloads, conformance, name, key, value, message):
    def change(document):
        row, = [row for row in document["results"][0]["metrics"] if row["id"] == name]
        row[key] = value
    replace_record(payloads, "benchmark.json", change)
    with pytest.raises(ValueError, match=message):
        AUDIT.verify_response(payloads, conformance)


def test_runner_substitution_rejects(payloads, conformance):
    payloads["runner.py"] += b"\n# changed\n"
    with pytest.raises(ValueError, match="runner changed"):
        AUDIT.verify_response(payloads, conformance)
