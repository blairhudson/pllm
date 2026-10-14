"""Evidence must reject wrong lifecycle, state, material accounting and text leaks."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("epoch_prefix_benchmark", scripts / "benchmark_epoch_prefix_sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result(name="shared", variant="control"):
    return {"workload": name, "variant": variant, "client": {
        "audit": {"inflight_joins": 1 if name == "shared" and variant in {"inflight", "combined"} else 0},
        "responses": [{"index": index, "status": "incomplete", "state": {"bits": "salted"},
                       "input_tokens": 40, "output_tokens": 8} for index in range(2)]}}


def test_all_frozen_experiments_use_public_component_contracts(driver):
    base = driver.runpy.run_path(str(driver.ROOT / "examples/prepared_sdk.py"))["build_experiment"]()
    for name, work in driver.WORKLOADS.items():
        for variant in (driver.VARIANTS if len(work["prompts"]) > 1 else ("control", "epochs")):
            exp = driver.experiment(base, work, variant)
            assert exp.budget.requests == len(work["prompts"])
            exp.resolve()


def test_complete_group_and_launcher_are_admitted_before_process_launch(driver, monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_response_group

    topology = {"admitted": True, "host": {"admission_budget_bytes": 600 << 20},
                "candidates": {"native": {"reasons": []}}}
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *args, **kwargs: topology)
    monkeypatch.setattr(benchmark_response_group, "group_memory_contract",
                        lambda *args, **kwargs: {"required_bytes": 512 << 20})
    exp = SimpleNamespace(budget=SimpleNamespace(requests=2, max_new_tokens=8))
    report = driver.admission(exp, require=False)
    assert report["required_bytes"] == 768 << 20
    assert not report["admitted"] and topology["admitted"]
    assert topology["candidates"]["native"]["reasons"] == []
    with pytest.raises(RuntimeError, match="group memory admission rejected"):
        driver.admission(exp)
    topology["host"]["admission_budget_bytes"] = 1 << 30
    assert driver.admission(exp)["admitted"]


def test_rejected_cohort_preserves_redacted_preflight_without_workers(driver, monkeypatch, tmp_path):
    from pllm import metrics, model_loader

    monkeypatch.setattr(model_loader, "resolve_model", lambda *_: SimpleNamespace(source_lock_digest="a" * 64))
    monkeypatch.setattr(metrics.BuildRef, "capture", lambda: SimpleNamespace(
        to_spec=lambda: {"python": "installed", "expected_digest": "b" * 64}))
    monkeypatch.setattr(driver, "admission", lambda *args, **kwargs: {
        "admitted": False, "required_bytes": 768 << 20, "reasons": ["insufficient headroom"]})
    monkeypatch.setattr(driver, "fresh_workers", lambda *args, **kwargs: pytest.fail("launched rejected cohort"))
    path = tmp_path / "blocked.json"
    with pytest.raises(RuntimeError, match="bounded details saved"):
        driver.main(["--tiny", "--workload", "shared", "--output", str(path)])
    evidence = json.loads(path.read_text())
    assert not evidence["complete"] and evidence["results"] == []
    assert evidence["failure"] == {"stage": "memory_admission", "blocked_cells": [
        "shared/" + variant for variant in driver.VARIANTS]}
    assert evidence["build"]["expected_digest"] == "b" * 64
    assert "Explain sky." not in path.read_text()


@pytest.mark.parametrize("failure", ["admission", "worker", "interrupt"])
def test_mid_cohort_failure_preserves_completed_cells_and_redacted_cause(driver, monkeypatch, tmp_path, failure):
    from pllm import model_loader

    monkeypatch.setattr(model_loader, "resolve_model", lambda *_: SimpleNamespace(source_lock_digest="a" * 64))
    admitted = {"admitted": True, "required_bytes": 768 << 20, "reasons": [],
                "topology": {"host": {"admission_budget_bytes": 1 << 30}}}
    monkeypatch.setattr(driver, "admission", lambda *args, **kwargs: admitted)
    guard = SimpleNamespace(error=None, minimum_available_bytes=2 << 30, maximum_swap_growth_bytes=0)
    run = SimpleNamespace(build={"python": "installed"})
    monkeypatch.setattr(driver, "fresh_workers", lambda *args, **kwargs: nullcontext((run, guard, None)))
    calls = []
    rejected = {**admitted, "admitted": False, "reasons": ["insufficient headroom"],
                "topology": {"host": {"admission_budget_bytes": 512 << 20}}}
    error = (driver.MemoryAdmissionRejected(rejected) if failure == "admission" else
             KeyboardInterrupt() if failure == "interrupt" else RuntimeError("private prompt and credential"))

    def cell(exp, workload, name, variant, replicate, salt, run):
        calls.append(variant)
        if len(calls) > 1:
            raise error
        return result(name, variant)

    monkeypatch.setattr(driver, "case", cell)
    path = tmp_path / "partial.json"
    with pytest.raises(type(error)):
        driver.main(["--tiny", "--workload", "shared", "--output", str(path)])
    evidence = json.loads(path.read_text())
    assert not evidence["complete"] and len(evidence["results"]) == 1
    assert calls == ["control", "epochs"]
    assert evidence["failure"]["cell"] == {"workload": "shared", "variant": "epochs", "replicate": 0}
    assert evidence["failure"]["error_type"] == type(error).__name__
    assert evidence["failure"]["stage"] == ("memory_admission" if failure == "admission" else "cell_execution")
    if failure == "admission":
        assert evidence["failure"]["admission"] == rejected
    assert evidence["memory_guard"]["maximum_swap_growth_bytes"] == 0
    assert "checks" not in evidence
    assert "Explain sky." not in path.read_text() and "private prompt" not in path.read_text()


def test_state_lifecycle_and_actual_pending_join_gates(driver):
    control = result()
    candidate = result(variant="combined")
    assert driver.verify_results([control, candidate])["checked_response_states"] == 4
    candidate["client"]["responses"][1]["state"]["bits"] = "different"
    with pytest.raises(RuntimeError, match="parity"):
        driver.verify_results([control, candidate])
    candidate = result(variant="inflight")
    candidate["client"]["audit"]["inflight_joins"] = 0
    with pytest.raises(RuntimeError, match="pending-prefix"):
        driver.verify_results([control, candidate])


def test_early_eos_cannot_be_relabelled_from_cap_or_truncation(driver):
    row = result("early")
    with pytest.raises(RuntimeError, match="natural-EOS"):
        driver.verify_results([row])
    for response in row["client"]["responses"]:
        response["status"] = "completed"
    assert driver.verify_results([row])["workload_lifecycles_checked"] is True
    with pytest.raises(RuntimeError, match="full-cap"):
        driver.verify_results([{**row, "workload": "full"}])


def test_lifetime_audit_requires_complete_control_and_one_use_conservation(driver):
    from pllm.runtime.client import PrivacyAudit
    audit = PrivacyAudit().to_dict()
    # This historical driver validates v2 evidence, before correction ACKs
    # became a separately accounted v3 edge.
    audit.pop("correction_ack_bytes")
    audit.update(prepared_stage_rows_reserved=3, prepared_stage_rows_claimed=2,
                 prepared_stage_rows_burned=1, bundle_network_bytes=100,
                 inference_control_upload_bytes=7)
    ledger = driver.validate_audit(audit)
    assert ledger["all_link_serialized_body_bytes"] == 107
    assert ledger["window_includes_bundle_and_preparation"]
    for key, value in (("preparation_requests_during_online", 1), ("prepared_stage_rows_burned", 0)):
        with pytest.raises(RuntimeError, match="accounting"):
            driver.validate_audit({**audit, key: value})
    broken = copy.deepcopy(audit)
    del broken["preparation_control_upload_bytes"]
    with pytest.raises(RuntimeError, match="accounting"):
        driver.validate_audit(broken)


def test_public_stream_reading_redacts_text_and_checks_emission_counts(driver, monkeypatch):
    response = {"id": "private-id", "model": "tiny", "status": "completed", "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 1, "total_tokens": 11}}
    events = [SimpleNamespace(type="response.created"), SimpleNamespace(type="response.output_text.delta"),
              SimpleNamespace(type="response.completed", response=response)]
    kwargs = []
    client = SimpleNamespace(tokenizer_scope=lambda: nullcontext(), responses=SimpleNamespace(
        create=lambda **value: (kwargs.append(value), nullcontext(iter(events)))[1]))
    monkeypatch.setattr(driver, "state_commitment", lambda *args: {"bits": "salted"})
    row = driver.collect_response(client, "private input", 8, b"x" * 32, 0)
    assert "private" not in str(row) and row["decode_tps"] is None
    assert kwargs[0]["store"] and kwargs[0]["stream"] and kwargs[0]["temperature"] == 0
    events.pop(1)
    with pytest.raises(RuntimeError, match="event accounting"):
        driver.collect_response(client, "private input", 8, b"x" * 32, 0)


def test_actual_state_bits_history_and_pending_are_independently_committed(driver):
    import numpy as np
    state = SimpleNamespace(next_logits=np.array([1.0, -0.0], dtype=np.float32),
        snapshot=SimpleNamespace(position=1, caches=[SimpleNamespace(length=1,
            key=np.ones((2, 1, 2), dtype=np.float32), value=np.ones((2, 1, 2), dtype=np.float32))]),
        token_ids=[1, 2], pending_token_ids=[3])
    client = SimpleNamespace(_core=SimpleNamespace(_transformer_conversations={"id": state}))
    response = SimpleNamespace(id="id", output_text="private output")
    first = driver.state_commitment(client, response, b"x" * 32)
    state.snapshot.caches[0].key[1] = 9  # inactive capacity is not decoder state
    assert driver.state_commitment(client, response, b"x" * 32) == first
    state.next_logits[1] = 0.0
    assert driver.state_commitment(client, response, b"x" * 32)["bits"] != first["bits"]
    state.pending_token_ids = [4]
    assert driver.state_commitment(client, response, b"x" * 32)["pending"] != first["pending"]
    assert driver.state_commitment(client, response, b"y" * 32)["output"] != first["output"]
    assert "private output" not in str(first)
