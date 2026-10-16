"""Read-only-audit regressions: no public weights, engine loads, or provider calls."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from benchmarks.research import checkpoint_compression as c
from scripts import benchmark_checkpoint_compression as d
from tests.test_checkpoint_compression import admission_fixture


def header(path, parameters=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"synthetic": {"dtype": "F32", "shape": [parameters], "data_offsets": [0, 0]}}
    raw = json.dumps(value).encode()
    path.write_bytes(len(raw).to_bytes(8, "little") + raw)


def snapshot(tmp_path):
    from pllm import Model
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    header(tmp_path / "model.safetensors", 3)
    spec = {"model": "synthetic/metadata-only", "revision": "a" * 40,
            "files": {n: c.file_hash(tmp_path / n) for n in ("config.json", "model.safetensors")}}
    choice = Model.hf(spec["model"], revision=spec["revision"], local_files_only=True)
    return spec, choice


def test_complete_snapshot_includes_undeclared_and_nested_index_shards(monkeypatch, tmp_path):
    spec, _ = snapshot(tmp_path)
    header(tmp_path / "other.safetensors", 5)
    header(tmp_path / "nested" / "part.safetensors", 7)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "one": "other.safetensors", "two": "nested/part.safetensors"}}))
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(tmp_path / "config.json"))
    result = d.metadata(spec)
    assert result["teacher_parameters"] == 3 + 5 + 7
    assert set(result["files"]) == {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert "nested/part.safetensors" in result["files"]
    assert result["resolved_source_lock"]["files"]
    assert result["entire_helper_parameters"] == 19_776


@pytest.mark.parametrize("shard", ["missing.safetensors", "../escape.safetensors", "/absolute.safetensors"])
def test_missing_or_escaping_index_shards_fail_before_values(tmp_path, shard):
    _, choice = snapshot(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": shard}}))
    with pytest.raises(d.ProvenanceError, match="shard"):
        d.snapshot_inventory(tmp_path, choice)


def test_teacher_over_1b_is_separate_from_entire_helper_cap(monkeypatch, tmp_path):
    spec, _ = snapshot(tmp_path)
    # Shape metadata only. Deliberately no tensor payload or engine values.
    header(tmp_path / "model.safetensors", 1_200_000_000)
    spec["files"]["model.safetensors"] = c.file_hash(tmp_path / "model.safetensors")
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: str(tmp_path / "config.json"))
    result = d.metadata(spec)
    assert result["teacher_parameters"] == 1_200_000_000
    assert result["entire_helper_parameters"] == 19_776
    assert result["research_teacher_plus_helper_parameters"] > 1_000_000_000
    assert sum(c.Encoder("source-A").inventory.values()) == 19_776
    with pytest.raises(ValueError, match="strictly"):
        c.runtime_parameters({"helper_backbone": 1_000_000_000, "head": 1})


@pytest.mark.parametrize("mutation", ["extra_shard", "changed_declared", "changed_nested_shard", "wrong_sdk_digest", "wrong_manifest_lock"])
def test_actual_resolved_inventory_checked_before_engine_constructor(monkeypatch, tmp_path, mutation):
    from pllm import Model, model_loader
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from pllm.runtime import transformer_engine
    root = create_tiny_llama_checkpoint(tmp_path / "generated", hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)
    local = model_loader.resolve_model(Model.path(str(root)))
    source = {"model": local.manifest.id, "revision": "generated-only"}
    choice = Model.hf(source["model"], revision=source["revision"], local_files_only=True)
    header(root / "nested" / "part.safetensors", 2)
    # Empty nonselected index filename is avoided; store only reads the first index.
    (root / "additional.safetensors.index.json").write_text(json.dumps({"weight_map": {"nested": "nested/part.safetensors"}}))
    expected = {"pilot": source, **d.snapshot_inventory(root, choice)}
    local = replace(local, checkpoint_digest=expected["checkpoint_digest"], source_lock_digest=expected["source_lock_digest"])
    local.manifest.metadata["source_lock"] = expected["resolved_source_lock"]
    monkeypatch.setattr(model_loader, "resolve_model", lambda _: local)
    monkeypatch.setattr(transformer_engine, "MaskedTransformerEngine", lambda **_: pytest.fail("engine constructor must not run"))
    if mutation == "extra_shard":
        header(root / "rogue.safetensors", 6)
    elif mutation == "changed_declared":
        (root / "config.json").write_text("{}")
    elif mutation == "changed_nested_shard":
        header(root / "nested" / "part.safetensors", 9)
    elif mutation == "wrong_sdk_digest":
        local = replace(local, source_lock_digest="0" * 64)
    else:
        local.manifest.metadata["source_lock"] = {"changed": True}
    with pytest.raises(d.ProvenanceError, match="before engine load"):
        d.make_teacher(expected)


def test_late_parent_provenance_failure_invalidates_completed_items(monkeypatch, tmp_path):
    from scripts import frontier_probe_support
    from pllm.runtime.benchmark_memory import HostMemory
    rows, answers = admission_fixture()
    manifest = {"digest": "synthetic", "data": {"rows": rows}}
    calls = 0
    def validate(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise d.ProvenanceError("code/native/environment changed after child")
        return manifest
    monkeypatch.setattr(d, "validate", validate)
    host = HostMemory(16 << 30, 8 << 30, 0, 4 << 30)
    monkeypatch.setattr("pllm.runtime.benchmark_memory.host_memory", lambda: host)
    @contextmanager
    def workers(*args, **kwargs):
        def run(payload):
            path = Path(payload["attempt"])
            doc = json.loads(path.read_text())
            doc.update(status="complete", outcome="complete")
            doc["items"] = {key: {"status": "complete", "full": answers[key], "passes": []} for key in doc["ids"]}
            d.atomic(path, doc)
            return {"status": "complete"}
        run.build = {}
        yield run, SimpleNamespace(sample=lambda: host, minimum_available_bytes=host.available, error=None, check=lambda: None), host
    monkeypatch.setattr(frontier_probe_support, "fresh_workers", workers)
    with pytest.raises(d.ProvenanceError):
        d.run_stage("synthetic", tmp_path, "baseline")
    ledger = d.Ledger(tmp_path, manifest)
    doc = json.loads(next(tmp_path.glob("baseline-*.json")).read_text())
    assert doc["provenance_invalid"] and doc["items"]
    assert ledger.completed("baseline") == {}
    assert not d.admission(manifest, ledger)["admitted"]


def test_valid_timeout_items_distinct_from_provenance_invalid_items(tmp_path):
    manifest = {"digest": "A", "data": {"rows": admission_fixture()[0]}}
    ledger = d.Ledger(tmp_path, manifest)
    path = ledger.start("baseline", [str(i) for i in range(17)])
    doc = json.loads(path.read_text())
    doc.update(outcome="parent_aborted", parent_error={"type": "TimeoutExpired"})
    doc["items"] = {str(i): {"status": "complete", "full": {"text": "wrong", "eos": True}, "passes": []} for i in range(17)}
    d.atomic(path, doc)
    assert d.admission(manifest, ledger)["status"] == "baseline_rejected"
    doc["provenance_invalid"] = True
    d.atomic(path, doc)
    assert d.admission(manifest, ledger)["status"] != "baseline_rejected"
    assert not d.admission(manifest, ledger)["admitted"]


def test_worker_post_validation_invalidates_previously_completed_items(monkeypatch, tmp_path):
    rows, results = admission_fixture()
    manifest = {"digest": "A", "source": {}, "data": {"rows": rows}}
    calls = 0
    def validate(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise d.ProvenanceError("source changed after work")
        return manifest
    monkeypatch.setattr(d, "validate", validate)
    teacher = SimpleNamespace(lock={"synthetic": True}, execute=lambda *a, **k: results["0"])
    monkeypatch.setattr(d, "make_teacher", lambda _: teacher)
    ledger = d.Ledger(tmp_path, manifest)
    path = ledger.start("baseline", ["0", "1"])
    with pytest.raises(d.ProvenanceError):
        d.worker({"manifest": "synthetic", "attempt": str(path), "artifact": None})
    doc = json.loads(path.read_text())
    assert len(doc["items"]) == 2 and all(i["status"] == "complete" for i in doc["items"].values())
    assert doc["provenance_invalid"] and ledger.completed("baseline") == {}
    assert doc["measured_worker_cpu_seconds"] >= 0
    assert "partial" in doc["worker_measurement_scope"]


def test_optimistic_qa_threshold_16_possible_17_rejected():
    rows, _ = admission_fixture()
    errors = {str(i): {"text": "wrong", "eos": True} for i in range(16)}
    assert not c.optimistic_baseline(rows, errors)["baseline_rejected"]
    errors["16"] = {"text": "wrong", "eos": True}
    assert c.optimistic_baseline(rows, errors)["baseline_rejected"]


@pytest.mark.parametrize("kind", c.POLICY["control_kinds"])
def test_optimistic_control_threshold_one_possible_two_rejected(kind):
    rows, _ = admission_fixture()
    selected = [r for r in rows if r["kind"] == kind]
    errors = {selected[0]["id"]: {"text": "wrong", "eos": True}}
    assert not c.optimistic_baseline(rows, errors)["baseline_rejected"]
    errors[selected[1]["id"]] = {"text": "wrong", "eos": False}
    assert c.optimistic_baseline(rows, errors)["impossible_kinds"] == [kind]


def test_rejected_baseline_does_not_launch_more_target_work(monkeypatch, tmp_path):
    manifest = {"digest": "A", "data": {"rows": admission_fixture()[0]}}
    monkeypatch.setattr(d, "validate", lambda _: manifest)
    ledger = d.Ledger(tmp_path, manifest)
    path = ledger.start("baseline", [str(i) for i in range(17)])
    doc = json.loads(path.read_text())
    doc["items"] = {str(i): {"status": "complete", "full": {"text": "wrong", "eos": True}} for i in range(17)}
    d.atomic(path, doc)
    monkeypatch.setattr("scripts.frontier_probe_support.fresh_workers", lambda *a, **k: pytest.fail("must not launch"))
    result = d.run_stage("synthetic", tmp_path, "baseline")
    assert result["status"] == "baseline_rejected"
    assert (tmp_path / "baseline_rejected.json").exists()
    with pytest.raises(RuntimeError, match="blocked"):
        d.run_stage("synthetic", tmp_path, "labels")


@pytest.mark.parametrize("kind", ["public_qa", "copy"])
def test_worker_stops_mid_chunk_immediately_after_decisive_quality_error(monkeypatch, tmp_path, kind):
    rows, _ = admission_fixture()
    manifest = {"digest": "A", "source": {}, "data": {"rows": rows}}
    monkeypatch.setattr(d, "validate", lambda _: manifest)
    selected = [r for r in rows if r["kind"] == kind]
    previous_count = 16 if kind == "public_qa" else 1
    ledger = d.Ledger(tmp_path, manifest)
    prior = ledger.start("baseline", [r["id"] for r in selected[:previous_count]])
    doc = json.loads(prior.read_text())
    wrong = {"text": "wrong", "eos": True}
    doc["items"] = {r["id"]: {"status": "complete", "full": wrong, "passes": []} for r in selected[:previous_count]}
    d.atomic(prior, doc)
    ids = [r["id"] for r in selected[previous_count:previous_count + 2]]
    path = ledger.start("baseline", ids)
    calls = []
    teacher = SimpleNamespace(lock={"synthetic": True}, execute=lambda *a, **k: calls.append(a) or wrong)
    monkeypatch.setattr(d, "make_teacher", lambda _: teacher)
    d.worker({"manifest": "synthetic", "attempt": str(path), "artifact": None})
    final = json.loads(path.read_text())
    assert len(calls) == 1 and set(final["items"]) == {ids[0]}
    assert final["outcome"] == "baseline_rejected"
    assert d.admission(manifest, ledger)["status"] == "baseline_rejected"


def test_pressure_invalid_errors_and_inflight_failures_do_not_reject_quality(tmp_path):
    rows, _ = admission_fixture()
    manifest = {"digest": "A", "data": {"rows": rows}}
    ledger = d.Ledger(tmp_path, manifest)
    path = ledger.start("baseline", [str(i) for i in range(20)])
    doc = json.loads(path.read_text())
    doc["items"] = {str(i): {"status": "complete", "full": {"text": "wrong", "eos": True}, "passes": []} for i in range(20)}
    doc["guard"] = {"error": "swap growth", "maximum_swap_growth_bytes": 1}
    d.atomic(path, doc)
    assert d.admission(manifest, ledger)["status"] != "baseline_rejected"
    doc["guard"] = {}
    for item in doc["items"].values():
        item["status"] = "incomplete"
    d.atomic(path, doc)
    assert d.admission(manifest, ledger)["optimistic"]["valid_completed_quality_errors"]["public_qa"] == 0


def test_atomic_publication_and_encoder_save_fsync_directories(monkeypatch, tmp_path):
    syncs = []
    original = c.fsync_directory
    def sync(path):
        syncs.append(Path(path))
        original(path)
    monkeypatch.setattr(c, "fsync_directory", sync)
    path = tmp_path / "artifact.json"
    c.Encoder("A").save(path)
    assert syncs == [tmp_path, tmp_path]
    assert c.Encoder.load(path, "A").binding == "A"
    prior = path.read_bytes()
    with pytest.raises(FileExistsError):
        c.Encoder("B").save(path)
    assert path.read_bytes() == prior
    monkeypatch.setattr(c.os, "link", lambda *a: (_ for _ in ()).throw(OSError("publication failure")))
    with pytest.raises(OSError):
        c.Encoder("B").save(tmp_path / "failed.json")
    assert list(tmp_path.iterdir()) == [path]


def test_per_arm_costs_and_killed_child_unknowns_not_zero(tmp_path):
    manifest = {"digest": "A", "data": {"rows": admission_fixture()[0]}}
    ledger = d.Ledger(tmp_path, manifest)
    path = ledger.start("evaluate", ["unfinished"])
    doc = json.loads(path.read_text())
    result = {"input_tokens": 10, "selected_tokens": 2, "executed_sequence_rows": 11,
              "stage_rows": 22, "stage_calls": 3, "cpu_seconds": 1.0, "wall_seconds": 2.0,
              "projected_arithmetic_bytes": {"client_to_inference": 10, "inference_to_client": 20, "preparation_to_inference": 30}}
    doc["items"] = {"unfinished": {"status": "incomplete", "passes": [
        {"mode": "learned", "status": "complete", "result": result},
        {"mode": "learned_fresh_full_fallback", "status": "complete", "result": result},
        {"mode": "lexical", "status": "incomplete"}]}}
    d.atomic(path, doc)
    cost = d.report(manifest, tmp_path)["costs"]
    assert cost["child_full_lifetime_cpu_seconds"] is None
    assert cost["worker_partial_window_cpu_seconds"]["total"] is None
    assert cost["worker_partial_window_cpu_seconds"]["missing_records"] == 1
    learned = cost["per_arm_all_attempts"]["learned"]
    assert learned["raw"]["counts"]["input_tokens"]["total"] == 10
    assert learned["fallback_inclusive"]["counts"]["input_tokens"]["total"] == 20
    assert learned["fallback_inclusive"]["projected_arithmetic_bytes"]["inference_to_client"]["total"] == 40
    lexical = cost["per_arm_all_attempts"]["lexical"]["raw"]
    assert lexical["counts"]["input_tokens"]["total"] is None
    assert lexical["incomplete_passes_unknown_not_zero"] == 1
