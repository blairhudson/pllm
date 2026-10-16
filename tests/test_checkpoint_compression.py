"""Independent neural/data/lifecycle gates; no public checkpoint tensor values."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmarks.research import checkpoint_compression as c
from scripts import benchmark_checkpoint_compression as driver


def row(split="train", key="one"):
    return {"id": key, "group": key, "split": split, "kind": "public_qa",
            "context": "Amber owl rests. Blue fox runs. Cedar bird sings. Dark cat sleeps. ",
            "question": "Which fox runs?", "answers": ["Blue fox"]}


def independent_forward(model, example):
    """Scalar-loop convolution oracle, no production encoder/forward calls."""
    w = model.weights
    def encode(text):
        ids = [0] + [b + 1 for b in text.encode()] + [0]
        outputs = []
        for i in range(1, len(ids) - 1):
            vector = np.concatenate([w["embedding"][ids[j]] for j in (i - 1, i, i + 1)])
            outputs.append(np.tanh(vector @ w["conv"] + w["conv_bias"]))
        return sum(outputs) / len(outputs)
    query = encode(example["question"])
    sent = [encode(s) for s in c.sentences(example["context"])]
    scores = []
    for i, vector in enumerate(sent):
        prev = sent[i - 1] if i else np.zeros_like(vector)
        nxt = sent[i + 1] if i + 1 < len(sent) else np.zeros_like(vector)
        x = np.concatenate([vector, query, vector * query, (prev + nxt)/2, [i/len(sent), (i+1)/len(sent)]])
        scores.append(np.tanh(x @ w["context_head"] + w["head_bias"]) @ w["output"])
    return np.array(scores)


def test_forward_independent_scalar_cnn_oracle_and_total_inventory():
    model = c.Encoder("A")
    scores, _ = model.forward(row())
    np.testing.assert_allclose(scores, independent_forward(model, row()), rtol=1e-13, atol=1e-14)
    assert sum(model.inventory.values()) == 19_776
    assert model.inventory["embedding"] == 257 * 32
    assert model.inventory["frozen_backbone"] == 0
    with pytest.raises(ValueError):
        c.runtime_parameters({**model.inventory, "hidden_backbone": 1_000_000_000})


@pytest.mark.parametrize("key", ["embedding", "conv", "conv_bias", "context_head", "head_bias", "output"])
def test_reverse_math_finite_difference_every_owner(key):
    model, example = c.Encoder("A"), row()
    losses = [0.1, 2.5, 1.7][:len(c.masks(example))]
    _, gradients = c.objective(model, example, losses)
    # Include padding embedding, lexical bytes, product/query/neighbor and positional paths.
    indices = [(0, 0), (66, 3), (111, 4)] if key == "embedding" else [tuple(0 for _ in model.weights[key].shape), tuple(s - 1 for s in model.weights[key].shape)]
    for index in indices:
        original = model.weights[key][index]
        epsilon = 1e-5
        model.weights[key][index] = original + epsilon
        plus = c.objective(model, example, losses)[0]
        model.weights[key][index] = original - epsilon
        minus = c.objective(model, example, losses)[0]
        model.weights[key][index] = original
        assert gradients[key][index] == pytest.approx((plus-minus)/(2*epsilon), abs=2e-9, rel=2e-4)


def test_order_context_and_no_gold_or_split_inputs():
    model = c.Encoder("A")
    example = row()
    original = model.forward(example)[0]
    changed = dict(example, answers=["SECRET"], split="heldout", group="different", id="different")
    np.testing.assert_array_equal(model.forward(changed)[0], original)
    changed = dict(example, context=example["context"].replace("Blue fox", "fox Blue"))
    assert not np.allclose(original, model.forward(changed)[0], atol=1e-9)
    changed = dict(example, context="Another creature walks. " + example["context"])
    assert not np.allclose(original, model.forward(changed)[0][1:])


def test_training_reduces_independent_cross_entropy_and_all_owners_change(tmp_path):
    example = row()
    losses = [0.1, 3.0, 5.0][:len(c.masks(example))]
    initial = c.Encoder("A")
    fitted = c.fit([example], [losses], "A")
    def independent_loss(model):
        scores = independent_forward(model, example)
        logits = np.array([scores[list(mask)].mean() for mask in c.masks(example)])
        target = np.exp(-np.array(losses)); target /= target.sum()
        return float(target @ (np.logaddexp.reduce(logits) - logits))
    assert independent_loss(fitted) < independent_loss(initial) - 0.05
    assert all(not np.array_equal(v, fitted.weights[k]) for k, v in initial.weights.items())
    artifact = tmp_path / "encoder.json"
    fitted.save(artifact)
    restored = c.Encoder.load(artifact, "A")
    assert restored.choose(example) == fitted.choose(example)
    with pytest.raises(FileExistsError):
        fitted.save(artifact)
    with pytest.raises(ValueError, match="binding"):
        c.Encoder.load(artifact, "B")
    doc = json.loads(artifact.read_text()); doc["weights"]["output"][0] += 1
    artifact.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="binding"):
        c.Encoder.load(artifact, "A")


@pytest.mark.parametrize("split", ["admission", "heldout"])
def test_fit_never_accepts_evaluation(split):
    with pytest.raises(ValueError, match="train split"):
        c.fit([row(split)], [[1, 2, 3]], "A")


def test_bad_shapes_nonfinite_and_byte_bounds():
    model = c.Encoder("A")
    with pytest.raises(ValueError, match="finite"):
        c.objective(model, row(), [float("nan")] * len(c.masks(row())))
    with pytest.raises(ValueError, match="byte bound"):
        model.forward(dict(row(), question="x" * 513))
    bad = dict(model.weights, output=np.zeros(1))
    with pytest.raises(ValueError, match="inventory"):
        c.Encoder("A", weights=bad)


def test_masks_preserve_source_order_verbatim():
    example = row()
    for mask in [*c.masks(example), c.lexical_mask(example), c.Encoder("A").choose(example)]:
        assert list(mask) == sorted(set(mask))
        assert c.extract(example, mask) == "".join(c.sentences(example["context"])[i] for i in mask)
    with pytest.raises(ValueError):
        c.extract(example, [2, 1])


def test_split_integrity_article_context_id_and_pilot_exclusions():
    left, right = row(), row("heldout", "two")
    with pytest.raises(ValueError, match="overlap"):
        c.assert_split_integrity([left, right])
    right["context"] = "Different context. Next sentence. "
    right["group"] = left["group"]
    with pytest.raises(ValueError, match="overlap"):
        c.assert_split_integrity([left, right])
    with pytest.raises(ValueError, match="prior-pilot"):
        c.assert_split_integrity([left], [left["group"]])
    with pytest.raises(ValueError, match="duplicate"):
        c.assert_split_integrity([left, left])


def test_public_corpus_preparation_fresh_counts_if_prepositioned():
    path = Path("target/checkpoint-tiny-model-squad-dev-v1.1.json")
    if not path.exists():
        pytest.skip("optional prepositioned public text; no downloads")
    data = c.prepare_data(path)
    assert data == c.prepare_data(path)
    assert len(data["rows"]) == 544
    c.assert_split_integrity(data["rows"], data["excluded_articles"])
    assert len({r["context"] for r in data["rows"]}) == 544
    for split, count in c.POLICY["counts"].items():
        assert sum(r["kind"] == "public_qa" and r["split"] == split for r in data["rows"]) == count


def admission_fixture():
    rows = [dict(row("admission", str(i)), context=f"Context {i}. Next sentence. ") for i in range(64)]
    rows += c.controls("admission", 8)
    results = {r["id"]: {"text": r["answers"][0], "eos": True} for r in rows}
    return rows, results


def test_baseline_admission_absolute_quality_denominators_eos_and_controls():
    rows, results = admission_fixture()
    assert c.baseline_gate(rows, results)["admitted"]
    assert not c.baseline_gate(rows, {})["admitted"]
    assert not c.baseline_gate(rows[:-1], results)["admitted"]
    for r in rows[:17]:
        results[r["id"]]["text"] = "wrong"
    assert not c.baseline_gate(rows, results)["admitted"]  # 47/64 < .75
    rows, results = admission_fixture()
    for r in [r for r in rows if r["kind"] == "order"][:2]:
        results[r["id"]]["eos"] = False
    assert not c.baseline_gate(rows, results)["admitted"]


def test_non_oracle_fallback_and_frozen_budgets():
    assert not c.needs_fallback("wrong", True, "wrong answer here")
    assert c.needs_fallback("invented", True, "wrong answer here")
    assert c.POLICY["max_fallbacks_per_chunk"] == 2
    assert c.POLICY["max_fallback_fraction"] == .1
    assert c.wilson_lower(48, 64) == pytest.approx(0.631833, abs=1e-5)


def test_ledger_resumes_only_completed_items_retains_partial_and_refuses_cohort(tmp_path):
    manifest = {"digest": "A"}
    ledger = driver.Ledger(tmp_path, manifest)
    path = ledger.start("baseline", ["one", "two"])
    doc = json.loads(path.read_text())
    doc["items"] = {"one": {"status": "complete", "full": {"text": "A"}},
                    "two": {"status": "incomplete", "passes": [{"status": "incomplete"}]}}
    driver.atomic(path, doc)
    assert set(ledger.completed("baseline")) == {"one"}
    assert json.loads(path.read_text())["items"]["two"]["passes"]
    with pytest.raises(ValueError, match="incompatible"):
        driver.Ledger(tmp_path, {"digest": "B"})
    with pytest.raises(FileExistsError):
        driver.atomic(path, {}, exclusive=True)


def mock_worker(monkeypatch, tmp_path, stage="baseline"):
    rows, _ = admission_fixture()
    manifest = {"digest": "synthetic", "source": {"pilot": {}, "files": {}}, "data": {"rows": rows}}
    monkeypatch.setattr(driver, "validate", lambda _: manifest)
    ledger = driver.Ledger(tmp_path, manifest)
    path = ledger.start(stage, [r["id"] for r in rows[:2]])
    return manifest, ledger, path, {"manifest": "synthetic", "attempt": str(path), "artifact": str(tmp_path / "model.json")}


def test_worker_failure_retains_completed_and_inflight_pass_no_session_retry(monkeypatch, tmp_path):
    from scripts import train_checkpoint_tiny_model as shared
    _, ledger, path, payload = mock_worker(monkeypatch, tmp_path)
    class BrokenTeacher:
        lock = {"source": "synthetic"}
        calls = 0
        def __init__(self, _):
            pass
        def execute(self, text, answer=None):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("injected second-item failure")
            return {"text": "Blue fox", "eos": True}
    monkeypatch.setattr(shared, "Teacher", BrokenTeacher)
    with pytest.raises(RuntimeError, match="second-item"):
        driver.worker(payload)
    doc = json.loads(path.read_text())
    assert doc["status"] == "incomplete" and doc["outcome"] == "failed"
    assert doc["items"]["1"]["passes"][0]["status"] == "incomplete"
    assert set(ledger.completed("baseline")) == {"0"}
    assert doc["process_memory"]["lifetime_peak_rss_bytes"] > 0


def test_worker_training_blocked_before_teacher_creation(monkeypatch, tmp_path):
    from scripts import train_checkpoint_tiny_model as shared
    _, _, path, payload = mock_worker(monkeypatch, tmp_path, "labels")
    monkeypatch.setattr(shared, "Teacher", lambda _: pytest.fail("teacher must not load"))
    with pytest.raises(RuntimeError, match="blocked"):
        driver.worker(payload)
    assert json.loads(path.read_text())["items"] == {}


def test_environment_json_roundtrip_locks_shared_sdk_and_native():
    environment = driver.environment()
    assert json.loads(json.dumps(environment)) == environment
    names = environment["code_native"]
    assert any("frontier_probe_support.py" in p for p in names)
    assert any("train_checkpoint_tiny_model.py" in p for p in names)
    assert any("_native" in p and p.endswith((".so", ".dylib", ".pyd")) for p in names)


def test_new_driver_with_real_generated_teacher(monkeypatch, tmp_path):
    """Actual source engine/native/compiler, only source resolution and freeze substituted."""
    from pllm import Model, model_loader
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    root = create_tiny_llama_checkpoint(tmp_path / "generated", hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)
    local = model_loader.resolve_model(Model.path(str(root)))
    source = {"model": local.manifest.id, "revision": "generated-never-public"}
    monkeypatch.setattr(model_loader, "resolve_model", lambda _: local)
    manifest, _, path, payload = mock_worker(monkeypatch, tmp_path / "ledger")
    manifest["source"] = {"pilot": source, "files": {n: {"sha256": c.file_hash(root / n)} for n in ("config.json", "model.safetensors", "tokenizer_config.json")}}
    manifest["source"].update(driver.snapshot_inventory(root, Model.hf(source["model"], revision=source["revision"], local_files_only=True)))
    local = replace(local, checkpoint_digest=manifest["source"]["checkpoint_digest"],
                    source_lock_digest=manifest["source"]["source_lock_digest"])
    local.manifest.metadata["source_lock"] = manifest["source"]["resolved_source_lock"]
    # Generated tokenizer is byte-based; short fixture stays inside the locked 512-token cap.
    sessions = []
    original = CompiledRuntimeModel.runtime
    def observe(self, *args, **kwargs):
        runtime = original(self, *args, **kwargs)
        sessions.append(runtime)
        return runtime
    monkeypatch.setattr(CompiledRuntimeModel, "runtime", observe)
    result = driver.worker(payload)
    assert result["status"] == "complete"
    doc = json.loads(path.read_text())
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(s.position == 0 and all(k.length == 0 for k in s.caches) for s in sessions)
    assert all(i["full"]["stage_calls"] > 0 for i in doc["items"].values())
    assert all(i["full"]["selected_tokens"] <= 24 for i in doc["items"].values())


def test_atomic_failed_snapshot_keeps_original(monkeypatch, tmp_path):
    path = tmp_path / "status.json"
    driver.atomic(path, {"status": "incomplete"}, exclusive=True)
    original = path.read_bytes()
    monkeypatch.setattr(driver.os, "replace", lambda *_: (_ for _ in ()).throw(OSError("injected fs failure")))
    with pytest.raises(OSError):
        driver.atomic(path, {"status": "complete"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_strict_watchdog_rejects_any_new_swap_and_full_reserve():
    from pllm.runtime.benchmark_memory import HostMemory
    host = HostMemory(total=16 << 30, available=8 << 30, swap_used=100, disk_free=2 << 30)
    sampler = driver.StrictHostSampler(lambda: host, host)
    assert sampler() == host
    sampler.sample = lambda: HostMemory(host.total, host.available, 101, host.disk_free)
    with pytest.raises(RuntimeError, match="no-new-swap"):
        sampler()
    assert sampler.maximum_swap_growth_bytes == 1
    sampler = driver.StrictHostSampler(lambda: HostMemory(host.total, (2 << 30) - 1, 100, host.disk_free), host)
    with pytest.raises(RuntimeError, match="full host reserve"):
        sampler()


def test_parent_timeout_preserves_attempt_and_work_intent(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import subprocess
    from pllm.runtime.benchmark_memory import HostMemory
    from scripts import frontier_probe_support
    manifest, ledger, _, _ = mock_worker(monkeypatch, tmp_path)
    host = HostMemory(16 << 30, 8 << 30, 0, 4 << 30)
    monkeypatch.setattr("pllm.runtime.benchmark_memory.host_memory", lambda: host)
    @contextmanager
    def guard(*args, **kwargs):
        assert kwargs["timeout"] == 900 and kwargs["memory_bytes"] == 4 << 30
        def run(payload):
            path = Path(payload["attempt"])
            doc = json.loads(path.read_text())
            doc["items"][doc["ids"][0]] = {"status": "incomplete", "passes": [{"mode": "full", "status": "incomplete"}]}
            driver.atomic(path, doc)
            raise subprocess.TimeoutExpired("synthetic-child", 900)
        run.build = {"synthetic": True}
        yield run, SimpleNamespace(sample=lambda: host, minimum_available_bytes=host.available, error=None), host
    monkeypatch.setattr(frontier_probe_support, "fresh_workers", guard)
    with pytest.raises(subprocess.TimeoutExpired):
        driver.run_stage("synthetic", tmp_path, "baseline")
    docs = [json.loads(p.read_text()) for p in tmp_path.glob("baseline-*.json")]
    aborted = next(d for d in docs if d["outcome"] == "parent_aborted")
    assert aborted["status"] == "incomplete" and aborted["parent_cpu_seconds"] >= 0
    assert len(aborted["items"]) == 1
    assert ledger.completed("baseline") == {}
    assert driver.admission(manifest, ledger)["admitted"] is False


def test_evaluate_executes_fresh_fallbacks_and_enforces_chunk_budget(monkeypatch, tmp_path):
    from scripts import train_checkpoint_tiny_model as shared
    manifest, _, old_path, _ = mock_worker(monkeypatch, tmp_path)
    examples = [row("heldout", str(i)) for i in range(3)]
    training = row("train", "training")
    manifest["data"]["rows"] = [training, *examples]
    monkeypatch.setattr(driver, "admission", lambda *_: {"admitted": True})
    binding = c.digest(manifest["source"])
    model = c.Encoder(binding, c.digest([training]))
    artifact = tmp_path / "model.json"
    model.save(artifact)
    ledger = driver.Ledger(tmp_path, manifest)
    fit_path = ledger.start("fit", [])
    fit_doc = json.loads(fit_path.read_text())
    fit_doc.update(status="complete", artifact={"sha256": c.file_hash(artifact)})
    driver.atomic(fit_path, fit_doc)
    path = ledger.start("evaluate", [r["id"] for r in examples])
    class Teacher:
        lock = {"source": "synthetic"}
        calls = 0
        def __init__(self, _):
            pass
        def execute(self, text, answer=None):
            Teacher.calls += 1
            return {"text": "invented", "eos": True}
    monkeypatch.setattr(shared, "Teacher", Teacher)
    driver.worker({"manifest": "synthetic", "attempt": str(path), "artifact": str(artifact)})
    doc = json.loads(path.read_text())
    assert Teacher.calls == 3 * 4 + 2
    assert sum("fresh_full_fallback" in p["mode"] for i in doc["items"].values() for p in i["passes"]) == 2
    assert all("learned" in i["fallback_budget_exhausted"] for i in doc["items"].values())
    assert old_path.exists()  # Original incomplete attempts untouched.


def test_report_distinguishes_useful_quality_from_learned_value(tmp_path):
    rows = [row("heldout", str(i)) for i in range(128)] + c.controls("heldout", 16)
    manifest = {"digest": "synthetic", "data": {"rows": admission_fixture()[0] + rows}}
    ledger = driver.Ledger(tmp_path, manifest)
    path = ledger.start("evaluate", [r["id"] for r in rows])
    doc = json.loads(path.read_text())
    def item(r, lexical=True):
        result = {"text": r["answers"][0], "eos": True}
        return {"status": "complete", "full": result, "lexical": result if lexical else {"text": "wrong", "eos": True},
                "untrained": result, "learned": result, "passes": []}
    doc["items"] = {r["id"]: item(r) for r in rows}
    doc["status"] = "complete"
    driver.atomic(path, doc)
    result = driver.report(manifest, tmp_path)
    assert result["useful_quality_pass"] and not result["learned_value_pass"]
    assert not result["serving_qualified"]
    doc["items"] = {r["id"]: item(r, False) for r in rows}
    driver.atomic(path, doc)
    assert driver.report(manifest, tmp_path)["learned_value_pass"]
    for r in rows:
        doc["items"][r["id"]]["learned_raw"] = {"text": "wrong", "eos": True}
    driver.atomic(path, doc)
    assert not driver.report(manifest, tmp_path)["learned_value_pass"]  # Retry alone is not learned value.
    for r in rows[:100]:
        doc["items"][r["id"]]["full"] = {"text": "wrong", "eos": True}
    driver.atomic(path, doc)
    result = driver.report(manifest, tmp_path)
    assert not result["useful_quality_pass"] and not result["learned_value_pass"]


def test_generated_teacher_labels_and_independent_nll_oracle(monkeypatch, tmp_path):
    from pllm import Model, model_loader
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
    from scripts import train_checkpoint_tiny_model as shared
    root = create_tiny_llama_checkpoint(tmp_path / "generated-labels", hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)
    local = model_loader.resolve_model(Model.path(str(root)))
    monkeypatch.setattr(model_loader, "resolve_model", lambda _: local)
    manifest, _, path, payload = mock_worker(monkeypatch, tmp_path / "ledger", "labels")
    example = dict(row(), answers=["ok"])
    manifest["data"]["rows"] = [example]
    manifest["source"] = {"pilot": {"model": local.manifest.id, "revision": "generated-only"},
        "files": {n: {"sha256": c.file_hash(root / n)} for n in ("config.json", "model.safetensors", "tokenizer_config.json")}}
    source = manifest["source"]["pilot"]
    manifest["source"].update(driver.snapshot_inventory(root, Model.hf(source["model"], revision=source["revision"], local_files_only=True)))
    local = replace(local, checkpoint_digest=manifest["source"]["checkpoint_digest"],
                    source_lock_digest=manifest["source"]["source_lock_digest"])
    local.manifest.metadata["source_lock"] = manifest["source"]["resolved_source_lock"]
    doc = json.loads(path.read_text()); doc["ids"] = [example["id"]]; driver.atomic(path, doc)
    monkeypatch.setattr(driver, "admission", lambda *_: {"admitted": True})
    sessions, teachers = [], []
    real_runtime, real_teacher = CompiledRuntimeModel.runtime, shared.Teacher
    class ObservedTeacher(real_teacher):
        def __init__(self, expected):
            super().__init__(expected)
            teachers.append(self)
    def observe(self, *args, **kwargs):
        runtime = real_runtime(self, *args, **kwargs)
        sessions.append(runtime)
        return runtime
    monkeypatch.setattr(shared, "Teacher", ObservedTeacher)
    monkeypatch.setattr(CompiledRuntimeModel, "runtime", observe)
    driver.worker(payload)
    doc = json.loads(path.read_text())
    assert len(sessions) == len(c.masks(example))
    assert all(s.position == 0 and all(k.length == 0 for k in s.caches) for s in sessions)
    teacher, session = teachers[0], sessions[0]
    text = c.prompt(example, c.extract(example, c.masks(example)[0]))
    rendered = teacher.bundle.render_prompt([{"role": "user", "content": text}])
    _, logits, _ = session.prepare_ids(session.encode_prompt(rendered))
    tokens = session.tokenizer.encode("ok", add_bos=False)
    losses = []
    for index, token in enumerate(tokens):
        values = logits.astype(np.float64)
        losses.append(float(np.logaddexp.reduce(values) - values[token]))
        if index + 1 < len(tokens):
            logits = session.forward_ids([token])[-1]
    session.reset()
    assert doc["items"][example["id"]]["losses"][0] == pytest.approx(np.mean(losses), abs=1e-12)


@pytest.mark.parametrize("stage", ["prepare", "preflight", "baseline", "labels", "fit", "evaluate", "report"])
def test_cli_stage_dispatch_with_synthetic_admission(monkeypatch, tmp_path, stage, capsys):
    """Exact parent CLI shapes, mocked admission only; never authorizes real values."""
    from pllm.runtime.benchmark_memory import HostMemory
    manifest_path = tmp_path / "manifest.json"
    output = tmp_path / "output.json"
    artifact = tmp_path / "helper.json"
    manifest = {"digest": "synthetic", "status": "prepared_unadmitted"}
    calls = []
    monkeypatch.setattr(driver, "prepare", lambda *args: calls.append(("prepare", args)) or manifest)
    monkeypatch.setattr(driver, "validate", lambda _: manifest)
    monkeypatch.setattr(driver, "report", lambda *args: {"status": "synthetic_report"})
    monkeypatch.setattr("pllm.runtime.benchmark_memory.host_memory", lambda: HostMemory(16 << 30, 8 << 30, 0, 2 << 30))
    def run(*args):
        calls.append(("run", args))
        return {"stage": args[2], "status": "synthetic_stage"}
    monkeypatch.setattr(driver, "run_stage", run)
    monkeypatch.setattr(driver.sys, "argv", ["scripts/benchmark_checkpoint_compression.py", stage,
        "--manifest", str(manifest_path), "--data", "target/checkpoint-tiny-model-squad-dev-v1.1.json",
        "--run-dir", str(tmp_path / "runs"), "--artifact", str(artifact), "--output", str(output), "--limit", "8"])
    driver.main()
    result = json.loads(capsys.readouterr().out)
    if stage in {"baseline", "labels", "fit", "evaluate"}:
        assert result["stage"] == stage and calls[0][1][-1] == 8
    elif stage == "prepare":
        assert result["manifest_digest"] == "synthetic"
    else:
        assert output.is_file()


def test_pressure_invalidated_items_preserved_but_never_admitted(tmp_path):
    ledger = driver.Ledger(tmp_path, {"digest": "A"})
    path = ledger.start("baseline", ["one"])
    doc = json.loads(path.read_text())
    doc["items"]["one"] = {"status": "complete", "full": {"text": "ok", "eos": True}}
    doc["guard"] = {"maximum_swap_growth_bytes": 1, "error": "no-new-swap"}
    driver.atomic(path, doc)
    assert not ledger.completed("baseline")
    assert json.loads(path.read_text())["items"]


def test_manifest_canonical_rederivation_source_environment_and_policy(monkeypatch, tmp_path):
    data = {"sha256": "public", "rows": [row()], "split_digest": "canonical"}
    source = {"pilot": {"model": "synthetic", "revision": "a" * 40},
              "files": {"fixture": {"bytes": 1}}}
    environment = {"code_native": {"synthetic": "locked"}}
    monkeypatch.setattr(c, "prepare_data", lambda _: data)
    monkeypatch.setattr(driver, "metadata", lambda _: source)
    monkeypatch.setattr(driver, "environment", lambda: environment)
    public = tmp_path / "public.json"; public.write_text("public fixture")
    manifest = tmp_path / "manifest.json"
    prepared = driver.prepare(public, manifest)
    assert driver.validate(manifest) == prepared
    with pytest.raises(FileExistsError):
        driver.prepare(public, manifest)
    # Even a recomputed manifest hash cannot inject different/private data.
    doc = json.loads(manifest.read_text())
    doc["data"]["rows"][0]["answers"] = ["injected"]
    doc.pop("digest"); doc["digest"] = c.digest(doc)
    driver.atomic(manifest, doc)
    with pytest.raises(ValueError, match="canonical public data"):
        driver.validate(manifest)
    driver.atomic(manifest, prepared)
    monkeypatch.setattr(driver, "environment", lambda: {"changed": True})
    with pytest.raises(ValueError, match="code/native/environment"):
        driver.validate(manifest)


def test_attempt_budgets_reject_third_item_attempt_preserve_originals(tmp_path):
    ledger = driver.Ledger(tmp_path, {"digest": "A"})
    first = ledger.start("baseline", ["one"])
    second = ledger.start("baseline", ["one"])
    with pytest.raises(RuntimeError, match="per-item attempt budget"):
        ledger.start("baseline", ["one"])
    assert first.is_file() and second.is_file()
