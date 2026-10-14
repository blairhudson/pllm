"""Frozen cells, accounting conservation, privacy and bounded failure records."""
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
    spec = importlib.util.spec_from_file_location("growing_epoch_driver", scripts / "benchmark_growing_epochs_sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit():
    from pllm.runtime.client import PrivacyAudit
    value = PrivacyAudit().to_dict()
    value.update(prepared_stage_rows_issued=13, prepared_stage_rows_discarded=3,
        prepared_stage_rows_reserved=10, prepared_stage_rows_claimed=4,
        prepared_stage_rows_burned=6, bundle_network_bytes=100,
        correction_push_bytes=20, correction_ack_accounted_push_bytes=20,
        correction_ack_bytes=3, inference_control_upload_bytes=7)
    return value


def result(name="early", variant="control", replicate=0):
    value = audit()
    value["generation_executed_decode_steps"] = 3 if name == "early" else 63
    return {"workload": name, "variant": variant, "replicate": replicate, "client": {
        "failure": None, "audit": value, "client_peak_rss_bytes": 128 << 20,
        "responses": [{"status": "completed" if name == "early" else "incomplete",
            "state": {"bits": "salted", "history": "salted", "rendered_history": "salted",
                "pending": "salted", "output": "salted", "position": 57 if name == "early" else 117,
                "pending_count": 0 if name == "early" else 1},
            "input_tokens": 54, "output_tokens": 3 if name == "early" else 64}]}}


def test_frozen_cap64_controls_and_full_history_admission(driver):
    factory = driver.runpy.run_path(str(driver.ROOT / "examples/prepared_sdk.py"))["build_experiment"]
    for work in driver.WORKLOADS.values():
        assert work["cap"] == 64 and len(work["prompts"]) == 1
        for variant in driver.VARIANTS:
            exp = driver.experiment(factory(), work, variant)
            assert exp.budget.max_new_tokens == 64 and exp.budget.max_input_tokens == 128
            assert exp.budget.requests == 1
            assert exp.pipeline.components["cache"].params["fixed_input_tokens"] == 128
            params = exp.pipeline.components["inventory"].params
            if variant == "growing":
                assert tuple(params["decode_epoch_schedule"]) == (4, 60)
                assert params.get("decode_epoch_tokens") is None
            exp.resolve()


def test_admission_keeps_launcher_and_shared_topology(driver, monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_response_group
    topology = {"admitted": True, "host": {"admission_budget_bytes": 600 << 20},
                "candidates": {"native": {"reasons": []}}}
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *a, **k: topology)
    monkeypatch.setattr(benchmark_response_group, "group_memory_contract", lambda *a, **k: {"required_bytes": 512 << 20})
    exp = SimpleNamespace(budget=SimpleNamespace(requests=1, max_new_tokens=64))
    report = driver.admission(exp, require=False)
    assert report["required_bytes"] == 768 << 20 and not report["admitted"]
    assert topology["candidates"]["native"]["reasons"] == []
    with pytest.raises(driver.MemoryAdmissionRejected):
        driver.admission(exp)


def test_v3_charges_cold_bundle_controls_ack_and_unused_rows(driver):
    value = audit()
    ledger = driver.validate_audit(value)
    assert ledger["all_link_serialized_body_bytes"] == 130
    part = ledger["traffic_attribution"]
    assert part["known_base_all_link_body_bytes"] + part["known_output_dependent_all_link_body_bytes"] + part["unattributed_all_link_body_bytes"] == 130
    assert part["unused_output_material_body_bytes"] is None
    for key in ("correction_ack_bytes", "preparation_control_upload_bytes"):
        broken = copy.deepcopy(value)
        del broken[key]
        with pytest.raises(RuntimeError, match="accounting"):
            driver.validate_audit(broken)
    value["prepared_stage_rows_burned"] = 0
    with pytest.raises(RuntimeError, match="accounting"):
        driver.validate_audit(value)


def test_epoch_partition_unknowns_not_inferred_from_row_ratios(driver):
    value = audit()
    value["decode_epochs_started"] = 2
    part = driver.validate_audit(value)["traffic_attribution"]
    assert part["base_all_link_body_bytes"] is None
    assert part["unused_output_material_body_bytes"] is None
    assert part["unattributed_all_link_body_bytes"] >= 30


def test_exact_state_lifecycle_and_largest_peak_across_reps(driver):
    rows = [result(), result(variant="growing", replicate=2)]
    rows[1]["client"]["client_peak_rss_bytes"] = driver.LOW_BYTES + 1
    checks = driver.verify_results(rows)
    assert not checks["low_rss_gate"] and checks["largest_client_lifetime_peak_rss_bytes"] == driver.LOW_BYTES + 1
    rows[1]["client"]["responses"][0]["state"]["pending"] = "different"
    with pytest.raises(RuntimeError, match="parity"):
        driver.verify_results(rows)
    full = result("full")
    assert driver.verify_results([full])["workload_lifecycles_checked"]
    full["client"]["responses"][0]["output_tokens"] = 8
    with pytest.raises(RuntimeError, match="full cap"):
        driver.verify_results([full])


def test_remote_work_counts_terminal_pruning_and_local_ownership(driver):
    geometry = [{"macs_per_row": 90, "terminal_prefill": False, "remote": True},
                {"macs_per_row": 10, "terminal_prefill": True, "remote": False}]
    value = driver.work_counts(geometry, 36, 63)
    assert value["body_macs"] == 99 * 90 + 64 * 10
    assert value["remote_90_percent_verified"] and value["whole_body_compute_fraction"] is None
    geometry[0]["remote"] = False
    assert not driver.work_counts(geometry, 36, 63)["remote_90_percent_verified"]


def test_failed_worker_retires_and_keeps_actual_audit_without_exception_text(driver, monkeypatch):
    import pllm
    from pllm.runtime import benchmark_response_group
    closed = []
    client = SimpleNamespace(tokenizer_scope=lambda: nullcontext(),
        runtime=SimpleNamespace(capabilities=lambda: None),
        prepared_rows_for_response=lambda *a, **k: None,
        close=lambda: closed.append(True), privacy_audit=SimpleNamespace(to_dict=audit),
        _core=SimpleNamespace(_prepared_queue=SimpleNamespace(requests=[]),
            _online_active=False, _preparation_active=False))
    monkeypatch.setattr(pllm, "OpenAI", lambda **k: client)
    monkeypatch.setattr(pllm.Experiment, "from_spec", lambda *a: None)
    monkeypatch.setattr(benchmark_response_group, "prefix_owners", lambda *a: ())
    monkeypatch.setattr(benchmark_response_group, "require_concurrency_contract", lambda *a: None)
    monkeypatch.setattr(benchmark_response_group, "inspect_private_cleanup", lambda *a: {"known_private_containers_empty": True})
    def fail(*a):
        raise RuntimeError("private text/token/key")
    monkeypatch.setattr(driver, "collect_response", fail)
    row = driver.client_worker({"experiment": {}, "connection": {}, "cache": "unused",
        "prompts": ["private prompt"], "cap": 64, "salt": "aa" * 32})
    assert closed == [True] and row["failure"]["error_type"] == "RuntimeError"
    assert row["ledger"]["all_link_serialized_body_bytes"] == 130
    assert row["audit"]["prepared_stage_rows_burned"] == 6
    assert "private prompt" not in json.dumps(row) and "private text/token/key" not in json.dumps(row)


@pytest.mark.parametrize("error", [RuntimeError("secret prompt/key"), KeyboardInterrupt()])
def test_interrupted_attempt_bounded_progress_no_private_errors(driver, monkeypatch, tmp_path, error):
    monkeypatch.setattr(driver, "freeze_installed_implementation", lambda build: None)
    from pllm import metrics, model_loader
    build = {"digest": "installed"}
    monkeypatch.setattr(metrics.BuildRef, "capture", lambda: SimpleNamespace(to_spec=lambda: build))
    monkeypatch.setattr(model_loader, "resolve_model", lambda *a: SimpleNamespace(source_lock_digest="a" * 64))
    monkeypatch.setattr(driver, "semantic_geometry", lambda *a: {"stages": []})
    monkeypatch.setattr(driver, "admission", lambda *a, **k: {"admitted": True, "required_bytes": 768 << 20})
    guard = SimpleNamespace(error=None, minimum_available_bytes=2 << 30, maximum_swap_growth_bytes=0)
    monkeypatch.setattr(driver, "fresh_workers", lambda *a, **k: nullcontext((SimpleNamespace(build=build), guard, None)))
    def fail(*a):
        raise error
    monkeypatch.setattr(driver, "case", fail)
    output = tmp_path / "interrupted.json"
    with pytest.raises(RuntimeError, match="redacted progress"):
        driver.main(["--tiny", "--replicates", "1", "--output", str(output)])
    report = json.loads(output.read_text())
    assert not report["complete"] and report["results"] == []
    assert len(report["attempts"]) == 1 and report["attempts"][0]["status"] != "completed"
    assert report["failure"]["error_type"] == type(error).__name__
    assert "secret" not in output.read_text() and "Explain sky." not in output.read_text()


def test_preflight_never_launches_workers(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "freeze_installed_implementation", lambda build: None)
    from pllm import metrics, model_loader
    monkeypatch.setattr(metrics.BuildRef, "capture", lambda: SimpleNamespace(to_spec=lambda: {"digest": "installed"}))
    monkeypatch.setattr(model_loader, "resolve_model", lambda *a: SimpleNamespace(source_lock_digest="a" * 64))
    monkeypatch.setattr(driver, "semantic_geometry", lambda *a: {"stages": []})
    monkeypatch.setattr(driver, "admission", lambda *a, **k: {"admitted": False, "required_bytes": 768 << 20})
    monkeypatch.setattr(driver, "fresh_workers", lambda *a, **k: pytest.fail("launched preflight"))
    output = tmp_path / "preflight.json"
    driver.main(["--tiny", "--preflight-only", "--output", str(output)])
    report = json.loads(output.read_text())
    assert not report["complete"] and report["planned_cells"] == 24
    assert len(report["failure"]["blocked_cells"]) == 8


@pytest.mark.parametrize("key,value", [("issued", 0), ("discarded", 0), ("burned", -1),
    ("claimed", True), ("reserved", None)])
def test_complete_retired_material_ledger_rejects_missing_or_unpriced_rows(driver, key, value):
    value_audit = audit()
    value_audit["prepared_stage_rows_" + key] = value
    with pytest.raises(RuntimeError, match="retired lifetime"):
        driver.validate_audit(value_audit)


@pytest.mark.parametrize("name,steps,position,pending", [
    ("full", 64, 118, 0), ("early", 2, 56, 1), ("early", 3, 58, 0), ("full", 63, 117, 0)])
def test_final_pending_and_eos_are_status_specific_not_n_or_n_minus_one(driver, name, steps, position, pending):
    cell = result(name)
    cell["client"]["audit"]["generation_executed_decode_steps"] = steps
    cell["client"]["responses"][0]["state"].update(position=position, pending_count=pending)
    with pytest.raises(RuntimeError, match="position/pending"):
        driver.verify_results([cell])


def test_actual_histories_committed_without_publishing_private_values(driver, monkeypatch):
    from pllm.runtime import types
    import benchmark_conversation_restore
    monkeypatch.setattr(driver, "state_commitment", lambda *args: {"bits": "unchanged"})
    saved = SimpleNamespace(rendered_context="private rendered context", pending_token_ids=[1])
    core = SimpleNamespace(_transformer_conversations={"id": saved},
        message_histories={"id": [{"role": "assistant", "content": "private history"}]},
        histories={"id": "private rendered plus raw output"})
    client, response = SimpleNamespace(_core=core), types.Response(id="id", model="fixture", output=[])
    first = driver.complete_state_commitment(client, response, b"s" * 32)
    core.message_histories["id"][0]["content"] += "changed"
    second = driver.complete_state_commitment(client, response, b"s" * 32)
    assert first["bits"] == second["bits"] and first["history"] != second["history"]
    saved.rendered_context += "changed"
    assert second["rendered_history"] != driver.complete_state_commitment(client, response, b"s" * 32)["rendered_history"]
    assert "private" not in json.dumps(first)
    assert driver.bounded_commitment(b"s" * 32, "bounded") == benchmark_conversation_restore.commitment(b"s" * 32, "bounded")
    with pytest.raises(RuntimeError, match="bound"):
        driver.bounded_commitment(b"s" * 32, [None] * 32769)


@pytest.mark.parametrize("module", [False, True])
def test_isolated_client_and_role_commands_disable_startup_injection(driver, tmp_path, monkeypatch, module):
    import subprocess
    import sys
    # No role/checkpoint launched: small Python process inspects bootstrap only.
    poison = tmp_path / "poison"
    poison.mkdir()
    marker = tmp_path / "startup-ran"
    (poison / "sitecustomize.py").write_text(f"open({str(marker)!r},'w').write('injected')")
    (poison / "pllm.py").write_text("raise RuntimeError('shadow SDK')")
    monkeypatch.setenv("PYTHONPATH", str(poison))
    probe = tmp_path / "probe.py"
    probe.write_text("import sys,pllm,json; print(json.dumps({'isolated':sys.flags.isolated,'no_site':sys.flags.no_site,'sdk':pllm.__file__,'args':sys.argv[1:]}))")
    if module:
        # Role command adapter uses same -m bootstrap; fixture module is installed stdlib.
        command = driver.isolated_command([sys.executable, "-m", "json.tool", "--help"])
        assert command[1:4] == ["-I", "-S", "-c"]
        output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
        assert "usage:" in output
    else:
        command = driver.isolated_command([sys.executable, str(probe), "kept"])
        output = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        assert output["isolated"] == output["no_site"] == 1 and output["args"] == ["kept"]
        assert "site-packages" in output["sdk"]
    assert not marker.exists()


def test_frozen_actual_implementation_digest_checked_before_child_target(driver, tmp_path, monkeypatch):
    import subprocess
    import sys
    monkeypatch.setattr(driver, "_IMPLEMENTATION_LOCK", ("0" * 64, "0" * 64))
    marker = tmp_path / "target-ran"
    script = tmp_path / "target.py"
    script.write_text(f"open({str(marker)!r},'w').write('must not run')")
    result = subprocess.run(driver.isolated_command([sys.executable, str(script)]), capture_output=True, text=True)
    assert result.returncode != 0 and "SDK implementation changed" in result.stderr
    assert not marker.exists()


def test_matching_installed_sdk_native_lock_allows_clean_child(driver, tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import subprocess
    import sys
    import pllm
    root = Path(pllm.__file__).parent
    sources = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".py", ".json"}}
    sdk = hashlib.sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    native = hashlib.sha256(Path(importlib.util.find_spec("pllm._native").origin).read_bytes()).hexdigest()
    monkeypatch.setattr(driver, "_IMPLEMENTATION_LOCK", (sdk, native))
    script = tmp_path / "probe.py"
    script.write_text("print('locked target')")
    result = subprocess.run(driver.isolated_command([sys.executable, str(script)]), capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == "locked target"
    monkeypatch.setattr(driver, "_IMPLEMENTATION_LOCK", (sdk, "0" * 64))
    result = subprocess.run(driver.isolated_command([sys.executable, str(script)]), capture_output=True, text=True)
    assert result.returncode != 0 and "native implementation changed" in result.stderr


def test_spawn_adapter_covers_metadata_and_role_module_commands_preserving_popen_type(driver, monkeypatch):
    import subprocess
    import sys
    # Simulate already-clean launcher flags without changing interpreter globals.
    monkeypatch.setattr(driver, "sys", SimpleNamespace(**(vars(sys) | {
        "flags": SimpleNamespace(isolated=True, no_site=True)})))
    original = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", original)  # Restore process-wide adapter after unit test.
    driver.installed_bootstrap(driver.__file__)
    assert issubclass(subprocess.Popen, original)
    process = subprocess.Popen(args=[sys.executable, "-m", "json.tool", "--help"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, error = process.communicate(timeout=30)
    assert process.returncode == 0 and "usage:" in out and not error
    assert process.args[1:4] == ["-I", "-S", "-c"]
