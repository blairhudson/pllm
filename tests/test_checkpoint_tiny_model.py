"""Independent behavior gates. Synthetic tests do not establish useful training."""
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.research import checkpoint_tiny_model as tiny


def example(split="train"):
    return {"id": split, "group": split, "split": split, "kind": "public_qa",
            "context": "The oak grows slowly. The elm needs water. The birch has pale bark. The yew is evergreen. ",
            "question": "Which tree has pale bark?", "answers": ["birch"]}


def test_runtime_parameter_limit_counts_frozen_and_vocab():
    assert tiny.runtime_parameters({"frozen": 999_999_997, "embedding": 1, "head": 1}) == 999_999_999
    with pytest.raises(ValueError, match="strictly"):
        tiny.runtime_parameters({"adapter": 1, "frozen": 999_999_999})
    with pytest.raises(ValueError, match="strictly"):
        tiny.runtime_parameters({"frozen": 900_000_000, "embedding": 60_000_000, "head": 60_000_000})


@pytest.mark.parametrize("inventory", [{}, {"missing": None}, {"negative": -1}, {"bool": True}, {"zero": 0}])
def test_unknown_or_invalid_parameter_inventory_fails(inventory):
    with pytest.raises(ValueError):
        tiny.runtime_parameters(inventory)


def test_source_spans_verbatim_and_ordered():
    row = example()
    assert "".join(tiny.sentences(row["context"])) == row["context"]
    source = tiny.sentences(row["context"])
    for candidate in tiny.candidates(row):
        assert candidate["indices"] == sorted(candidate["indices"])
        assert candidate["text"] == "".join(source[i] for i in candidate["indices"])
    assert "birch" in tiny.candidates(row)[2]["text"]


def test_features_do_not_read_answer_or_split():
    row = example()
    candidate = tiny.candidates(row)[1]
    before = tiny.features(row, candidate)
    changed = dict(row, answers=["SECRET-NOT-IN-PROMPT"], split="heldout")
    assert np.array_equal(before, tiny.features(changed, candidate))
    assert before.shape == (264,)


def test_bigram_features_distinguish_token_order():
    row = dict(example(), question="red blue", context="red blue. blue red. ")
    left, right, _ = tiny.candidates(row)
    assert not np.array_equal(tiny.features(row, left)[8:], tiny.features(row, right)[8:])


def test_fit_moves_toward_teacher_without_eval_leakage(tmp_path):
    row = example()
    model = tiny.fit([row], [[8.0, 0.1, 0.2]], "checkpoint-A")
    choices = tiny.candidates(row)
    assert tiny.features(row, choices[1]) @ model.weights > tiny.features(row, choices[0]) @ model.weights
    path = tmp_path / "helper.json"
    model.save(path)
    restored = tiny.Ranker.load(path, "checkpoint-A")
    assert restored.choose(row) == model.choose(row)
    assert sum(restored.parameter_inventory.values()) == 264
    with pytest.raises(ValueError, match="binding"):
        tiny.Ranker.load(path, "checkpoint-B")
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="train split"):
        tiny.fit([example("heldout")], [[1, 2, 3]], "checkpoint-A")


def test_artifact_tampering_rejected(tmp_path):
    path = tmp_path / "helper.json"
    tiny.fit([example()], [[1, 2, 3]], "A").save(path)
    doc = json.loads(path.read_text())
    doc["weights"][0] = 100
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="binding"):
        tiny.Ranker.load(path, "A")


@pytest.mark.parametrize("losses", [[[1, 2]], [[1, float("nan"), 2]], []])
def test_bad_teacher_losses_fail_closed(losses):
    with pytest.raises(ValueError):
        tiny.fit([example()], losses, "A")


def test_tied_teacher_does_not_invent_training_signal():
    model = tiny.fit([example()], [[1, 1, 1]], "A")
    assert np.array_equal(model.weights, np.zeros(264))


def test_group_split_stable_no_paragraph_crossing():
    groups = ["article-" + str(i) for i in range(100)]
    mapping = {group: tiny.split_for(group) for group in groups}
    assert set(mapping.values()) == {"train", "dev", "heldout"}
    assert mapping == {group: tiny.split_for(group) for group in reversed(groups)}


def test_stress_strings_and_rare_answers_disjoint():
    cohorts = [tiny.stress_rows(split) for split in ("train", "dev", "heldout")]
    assert len({tiny.digest(row) for cohort in cohorts for row in cohort}) == 6
    assert len({row["context"] for cohort in cohorts for row in cohort}) == 6
    assert len({row["answers"][0] for cohort in cohorts for row in cohort}) == 6


def test_public_data_source_lock_not_arbitrary_private_prompts(tmp_path):
    path = tmp_path / "private.json"
    path.write_text('{"private": "never train"}')
    with pytest.raises(ValueError, match="public SQuAD source lock"):
        tiny.prepare_data(path)


def test_fallback_is_structural_not_answer_oracle():
    assert not tiny.needs_fallback("wrong", True, "The wrong answer is here.")
    assert tiny.needs_fallback("invented", True, "The answer is here.")
    assert tiny.needs_fallback("here", False, "here")
    assert tiny.needs_fallback("", True, "here")


def test_grading_keeps_rare_and_order_failures():
    assert not tiny.exact_match("QJ-123", ["ZX-123"])
    assert not tiny.exact_match("red blue", ["blue red"])
    assert tiny.exact_match("The Birch.", ["birch"])


def test_complete_projected_directions_and_reservation():
    stages = [{"inputs": 3, "outputs": 5, "bits": 16}, {"inputs": 2, "outputs": 4, "bits": 24}]
    costs = tiny.projected_cost(stages, 7, 11)
    assert costs == {"client_to_inference": 7 * (6 + 6),
                     "inference_to_client": 7 * (10 + 12),
                     "preparation_to_inference": 11 * (10 + 12)}
    with pytest.raises(ValueError):
        tiny.projected_cost(stages, 12, 11)


def test_support_never_promotes_semantic_only():
    table = tiny.support_table(Path(__file__).resolve().parents[1])
    real = [row for row in table["rows"] if row["real_checkpoint_runtime"]]
    assert len(real) == 7
    trained = [row for row in real if row["tiny_artifact_status"] != "untrained"]
    assert len(trained) == 1
    assert trained[0]["tiny_artifact_status"] == "trained_but_failed_heldout_screen"
    assert not trained[0]["completed_pilot"]["serving_qualified"]
    pilot = trained[0]["completed_pilot"]
    assert (pilot["heldout_qa_correct"], pilot["heldout_qa_count"]) == (1, 4)
    assert not pilot["screen_pass"]
    assert pilot["heldout_fallback_ratio"] == pytest.approx(
        pilot["heldout_fallback_inclusive_projected_bytes"] / pilot["heldout_full_projected_bytes"])
    assert pilot["heldout_fallback_ratio"] > 1
    assert all(row["tiny_artifact_status"].startswith("blocked") for row in table["rows"] if not row["real_checkpoint_runtime"])


def test_support_includes_evidence_bound_qwen3_4b_not_adapter_family_inference():
    root = Path(__file__).resolve().parents[1]
    table = tiny.support_table(root)
    row, = [r for r in table["rows"] if r["checkpoint"] == "Qwen/Qwen3-4B"]
    assert row["real_checkpoint_runtime"] and row["tiny_artifact_status"] == "untrained"
    assert row["training_admission"].startswith("blocked")
    assert row["source_lock"]["revision"] == "1cfa9a7208912126459214e8b04321603b3df60c"
    assert row["config"]["hidden_size"] == 2560 and row["config"]["num_hidden_layers"] == 36
    assert row["config_sha256"] == "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
    assert row["source_lock"]["revision"] in row["config_reference"]
    assert row["quality_status"] == "memory-rejected"
    assert row["evidence_sha256"]
    assert all(tiny.file_hash(root / name) == sha for name, sha in row["evidence_sha256"].items())
    assert "not an exhaustive" in table["inventory_scope"]


def test_driver_worker_complete_mechanics_without_real_tensors(monkeypatch, tmp_path):
    import scripts.train_checkpoint_tiny_model as driver

    class FakeTeacher:
        def __init__(self, _expected):
            self.lock = {"checkpoint": "synthetic-no-real-training"}

        def execute(self, text, *, answer=None):
            # Forces retry; fake is deliberately not a language-quality oracle.
            return {"text": "not-in-context", "eos": False, "answer_nll": 1.0 if answer else None,
                    "cpu_seconds": 0.1, "wall_seconds": 0.2,
                    "projected_arithmetic_bytes": {"client_to_inference": 3,
                                                   "inference_to_client": 5,
                                                   "preparation_to_inference": 7}}

    monkeypatch.setattr(driver, "Teacher", FakeTeacher)
    rows = [example(split) for split in ("train", "dev", "heldout")]
    rows += [row for split in ("train", "dev", "heldout") for row in tiny.stress_rows(split)]
    monkeypatch.setattr(tiny, "prepare_data", lambda _: {"rows": rows})
    result = driver.worker({"preflight": {}, "data": {"rows": rows}, "data_path": "synthetic"}, tmp_path / "model.json")
    assert len(result["teacher_training"]) == 3
    assert len(result["evaluation"]) == 6
    for row in result["evaluation"]:
        fallback = row["results"]["learned_then_full"]
        assert fallback["fallback"]
        assert sum(fallback["projected_arithmetic_bytes"].values()) == 30
        assert fallback["cpu_seconds"] >= 0.2
    assert not result["summary"]["heldout"]["gate"]["screen_pass"]


def test_mechanics_report_is_deterministic_and_honest():
    root = Path(__file__).resolve().parents[1]
    first = tiny.mechanics_report(root)
    assert first == tiny.mechanics_report(root)
    assert first["real_tensor_loads"] == 0
    assert first["strict_parameter_check"] == 264
    assert first["status"] == "synthetic_mechanics_only_no_new_checkpoint_training"


def test_preflight_fails_unknown_memory_before_source_access(monkeypatch):
    import scripts.train_checkpoint_tiny_model as driver
    from pllm.runtime import benchmark_memory

    def unknown():
        raise benchmark_memory.BenchmarkMemoryError("telemetry unavailable")

    monkeypatch.setattr(benchmark_memory, "host_memory", unknown)
    with pytest.raises(benchmark_memory.BenchmarkMemoryError, match="unavailable"):
        driver.preflight({"sha256": "public-data"})


def test_worker_rejects_manifest_injected_private_rows_before_teacher(monkeypatch, tmp_path):
    import scripts.train_checkpoint_tiny_model as driver
    monkeypatch.setattr(tiny, "prepare_data", lambda _: {"rows": [example()]})
    with pytest.raises(ValueError, match="locked public corpus"):
        driver.worker({"data_path": "public", "data": {"rows": [dict(example(), context="private")]}},
                      tmp_path / "never-written.json")
    assert not (tmp_path / "never-written.json").exists()


def test_preflight_counts_checkpoint_metadata_without_loading_values(monkeypatch, tmp_path):
    import huggingface_hub
    import scripts.train_checkpoint_tiny_model as driver
    from pllm.runtime import benchmark_memory

    header = json.dumps({"embedding.weight": {"shape": [17, 4], "dtype": "F32", "data_offsets": [0, 272]},
                         "head.weight": {"shape": [17, 4], "dtype": "F32", "data_offsets": [272, 544]},
                         "frozen.weight": {"shape": [4, 4], "dtype": "F32", "data_offsets": [544, 608]}}).encode()
    # No tensor values appended. Metadata counting must not request any.
    (tmp_path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (tmp_path / name).write_text("{}")
    pilot = {"model": "synthetic", "revision": "source-lock",
             "files": {name: tiny.file_hash(tmp_path / name) for name in ("config.json", "model.safetensors")}}
    monkeypatch.setattr(tiny, "PILOT", pilot)
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda _model, name, **_: str(tmp_path / name))
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: benchmark_memory.HostMemory(
        total=16 << 30, available=12 << 30, swap_used=0, disk_free=2 << 30))
    result = driver.preflight({"sha256": "public-data"})
    inventory = result["parameter_inventory"]
    assert inventory["target_all_checkpoint_tensors_including_embedding_head_norm_bias"] == 152
    assert inventory["helper_total"] == 264
    assert inventory["pilot_local_target_plus_helper"] == 416
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: benchmark_memory.HostMemory(
        total=16 << 30, available=5 << 30, swap_used=0, disk_free=2 << 30))
    with pytest.raises(RuntimeError, match="post-reserve"):
        driver.preflight({"sha256": "public-data"})


def test_real_teacher_api_generated_checkpoint_nll_greedy_and_reset(monkeypatch, tmp_path):
    """Real engine/compiler/runtime; only public-source resolution is substituted."""
    import scripts.train_checkpoint_tiny_model as driver
    from pllm import Model
    from pllm import model_loader
    from pllm.runtime.model_binding import CompiledRuntimeModel
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    root = create_tiny_llama_checkpoint(tmp_path / "generated-teacher", hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8)
    local = model_loader.resolve_model(Model.path(str(root)))
    source = {"model": local.manifest.id, "revision": "generated-never-public"}

    def generated_source(choice):
        assert choice.source == source["model"]
        assert choice.revision == source["revision"]
        assert choice.local_files_only
        return local

    monkeypatch.setattr(model_loader, "resolve_model", generated_source)
    expected = {"pilot": source, "files": {name: {"sha256": tiny.file_hash(root / name)}
                 for name in ("config.json", "model.safetensors", "tokenizer_config.json")}}
    # Observe real constructed sessions; do not replace tensor/model execution.
    runtimes = []
    original_runtime = CompiledRuntimeModel.runtime

    def observe(self, *args, **kwargs):
        runtime = original_runtime(self, *args, **kwargs)
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(CompiledRuntimeModel, "runtime", observe)
    teacher = driver.Teacher(expected)
    forced = teacher.execute("hi", answer="ok")
    assert np.isfinite(forced["answer_nll"]) and forced["answer_nll"] > 0
    assert forced["selected_tokens"] == 2
    assert forced["stage_calls"] > 0 and forced["stage_rows"] > 0
    assert forced["executed_sequence_rows"] == forced["input_tokens"] + 1
    first = teacher.execute("hello")
    second = teacher.execute("hello")
    assert first["text"] == second["text"]
    assert first["selected_tokens"] == second["selected_tokens"]
    assert first["input_digest"] == second["input_digest"]
    assert first["stage_calls"] > 0 and first["stage_rows"] > 0
    assert first["executed_sequence_rows"] == first["input_tokens"] + first["selected_tokens"] - 1
    assert len(runtimes) == 3 and len({id(runtime) for runtime in runtimes}) == 3
    for runtime in runtimes:
        assert runtime._causal_reduction == "prefix_f32"
        assert runtime.position == 0
        assert all(cache.length == 0 for cache in runtime.caches)
    # Independent answer-NLL calculation on real logits, not Teacher's reduction.
    runtime = runtimes[0]
    rendered = teacher.bundle.render_prompt([{"role": "user", "content": "hi"}])
    _, scores, _ = runtime.prepare_ids(runtime.encode_prompt(rendered))
    nll = []
    answer_ids = runtime.tokenizer.encode("ok", add_bos=False)
    for index, token in enumerate(answer_ids):
        scores64 = scores.astype(np.float64)
        nll.append(float(np.logaddexp.reduce(scores64) - scores64[token]))
        if index + 1 < len(answer_ids):
            scores = runtime.forward_ids([token])[-1]
    runtime.reset()
    assert forced["answer_nll"] == pytest.approx(float(np.mean(nll)), abs=1e-12)


def test_future_manifest_covers_actual_teacher_helper_and_shared_guard():
    from scripts.train_checkpoint_tiny_model import code_hashes, validate_code_hashes
    locks = code_hashes()
    assert "scripts/decoder_probe_support.py" in locks
    assert "scripts/frontier_probe_support.py" in locks
    validate_code_hashes(locks)
    del locks["scripts/decoder_probe_support.py"]
    with pytest.raises(ValueError, match="dependencies"):
        validate_code_hashes(locks)
    locks = code_hashes()
    locks["scripts/frontier_probe_support.py"] = "substituted"
    with pytest.raises(ValueError, match="dependencies"):
        validate_code_hashes(locks)


def test_progress_exclusive_atomic_bounded_and_success_schema(tmp_path):
    from scripts.train_checkpoint_tiny_model import RunProgress
    path, artifact = tmp_path / "output.json", tmp_path / "helper.json"
    state = RunProgress.reserve(path, artifact)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        RunProgress.reserve(path, artifact)
    assert path.read_bytes() == original
    state.update("evaluation", evaluation=[{"id": "completed-public-example"}])
    prior = path.read_bytes()
    with pytest.raises(ValueError, match="1MiB"):
        state.update("evaluation", unbounded="x" * (1 << 20))
    assert path.read_bytes() == prior
    assert not list(tmp_path.glob("output.json.*"))
    state.complete({"schema": tiny.SCHEMA, "status": "public_checkpoint_pilot_not_serving_qualification",
                    "summary": {"heldout": {"gate": {"screen_pass": False}}}})
    final = json.loads(path.read_text())
    assert final["schema"] == tiny.SCHEMA
    assert final["summary"]["heldout"]["gate"]["screen_pass"] is False
    assert final["run_lifecycle"]["outcome"] == "complete"
    assert not final["run_lifecycle"]["serving_qualified"]
    with pytest.raises(ValueError, match="completed"):
        state.update("overwrite")


@pytest.mark.parametrize("fail_on,phase,helper_exists", [(2, "train_teacher", False),
                                                         (5, "evaluation", True),
                                                         (11, "evaluation", True)])
def test_injected_worker_failure_retains_train_eval_and_unqualified_helper(
    monkeypatch, tmp_path, fail_on, phase, helper_exists
):
    import scripts.train_checkpoint_tiny_model as driver
    rows = [example("train"), example("dev"), example("heldout")]
    monkeypatch.setattr(tiny, "prepare_data", lambda _: {"rows": rows})

    class FailingTeacher:
        def __init__(self, _expected):
            self.lock = {"checkpoint": "synthetic-only"}
            self.calls = 0

        def execute(self, _text, *, answer=None):
            self.calls += 1
            if self.calls == fail_on:
                raise RuntimeError("injected teacher failure")
            return {"text": "not-a-span", "eos": False, "answer_nll": 1.0 if answer else None,
                    "cpu_seconds": 0.1, "wall_seconds": 0.1,
                    "projected_arithmetic_bytes": {"client_to_inference": 1,
                         "inference_to_client": 1, "preparation_to_inference": 1}}

    monkeypatch.setattr(driver, "Teacher", FailingTeacher)
    output, artifact = tmp_path / "output.json", tmp_path / "helper.json"
    progress = driver.RunProgress.reserve(output, artifact)
    with pytest.raises(RuntimeError, match="injected") as error:
        driver.worker({"data_path": "synthetic", "data": {"rows": rows}, "preflight": {}}, artifact, progress)
    progress.fail(error.value)  # Same terminal transition performed by CLI parent.
    doc = json.loads(output.read_text())
    assert doc["status"] == "incomplete" and doc["outcome"] == "failed" and doc["phase"] == phase
    assert doc["artifact"]["exists"] == helper_exists == artifact.exists()
    assert not doc["artifact"]["serving_qualified"]
    if fail_on == 2:
        assert len(doc["current_train"]["completed"]) == 1
    else:
        assert len(doc["teacher_training"]) == 1
        assert "full" in doc["current_evaluation"]["completed"]
    if fail_on == 11:
        assert len(doc["evaluation"]) == 1  # Prior completed row survives next-row failure.


def test_parent_timeout_retains_child_progress_and_marks_timeout(monkeypatch, tmp_path):
    from contextlib import contextmanager
    import subprocess
    import scripts.train_checkpoint_tiny_model as driver
    import scripts.frontier_probe_support as guards
    data = {"rows": []}
    manifest = {"data": data, "data_path": "public", "preflight": {"files": {}}, "code": driver.code_hashes()}
    manifest["manifest_digest"] = tiny.digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(tiny, "prepare_data", lambda _: data)
    monkeypatch.setattr(driver, "preflight", lambda _: {"files": {}})

    @contextmanager
    def timeout_runner(*_args, **_kwargs):
        def run(payload):
            progress = driver.RunProgress(**payload["progress"])
            progress.update("train_teacher", teacher_training=[{"id": "completed-before-timeout"}])
            raise subprocess.TimeoutExpired("synthetic-child", 1800)
        yield run, None, None

    monkeypatch.setattr(guards, "fresh_workers", timeout_runner)
    output = tmp_path / "output.json"
    with pytest.raises(subprocess.TimeoutExpired):
        driver.run_pilot(path, tmp_path / "helper.json", output)
    doc = json.loads(output.read_text())
    assert doc["outcome"] == "timeout" and doc["status"] == "incomplete"
    assert doc["teacher_training"] == [{"id": "completed-before-timeout"}]
    assert doc["phase"] == "train_teacher" and not doc["artifact"]["serving_qualified"]


def test_parent_validation_failure_already_has_structured_reservation(tmp_path):
    from scripts.train_checkpoint_tiny_model import run_pilot
    output = tmp_path / "output.json"
    with pytest.raises(FileNotFoundError):
        run_pilot(tmp_path / "missing-manifest.json", tmp_path / "helper.json", output)
    doc = json.loads(output.read_text())
    assert doc["phase"] == "manifest_validation" and doc["outcome"] == "failed"
    assert doc["evaluation"] == [] and not doc["artifact"]["exists"]


def test_parent_success_keeps_legacy_result_and_unqualified_lifecycle(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from types import SimpleNamespace
    import scripts.train_checkpoint_tiny_model as driver
    import scripts.frontier_probe_support as guards
    from pllm.runtime.benchmark_memory import HostMemory
    data = {"rows": []}
    manifest = {"data": data, "data_path": "public", "preflight": {"files": {}}, "code": driver.code_hashes()}
    manifest["manifest_digest"] = tiny.digest(manifest)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(tiny, "prepare_data", lambda _: data)
    monkeypatch.setattr(driver, "preflight", lambda _: {"files": {}})

    @contextmanager
    def success_runner(*_args, **_kwargs):
        def run(payload):
            Path(payload["artifact"]).write_text("{}")
            driver.RunProgress(**payload["progress"]).update("summarize", evaluation=[{"id": "synthetic"}])
            return {"status": "public_checkpoint_pilot_not_serving_qualification",
                    "summary": {"heldout": {"gate": {"screen_pass": False}}}}
        run.build = {"synthetic": True}
        yield run, SimpleNamespace(maximum_swap_growth_bytes=0), HostMemory(8 << 30, 6 << 30, 0, 2 << 30)

    monkeypatch.setattr(guards, "fresh_workers", success_runner)
    output = tmp_path / "output.json"
    result = driver.run_pilot(path, tmp_path / "helper.json", output)
    doc = json.loads(output.read_text())
    assert doc["schema"] == tiny.SCHEMA and doc["status"] == result["status"]
    assert doc["summary"] == result["summary"]
    assert doc["run_lifecycle"]["outcome"] == "complete"
    assert doc["run_lifecycle"]["artifact"]["exists"]
    assert not doc["run_lifecycle"]["artifact"]["serving_qualified"]


def test_atomic_update_failure_keeps_prior_snapshot_and_cleans_temp(monkeypatch, tmp_path):
    import scripts.train_checkpoint_tiny_model as driver
    output = tmp_path / "output.json"
    state = driver.RunProgress.reserve(output, tmp_path / "helper.json")
    original = output.read_bytes()

    def fail_replace(*_args):
        raise OSError("injected atomic replacement failure")

    monkeypatch.setattr(driver.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        state.update("evaluation", evaluation=[{"id": "new"}])
    assert output.read_bytes() == original
    assert not list(tmp_path.glob("output.json.*"))


def test_interrupted_state_keeps_worker_error_when_parent_wraps(tmp_path):
    from scripts.train_checkpoint_tiny_model import RunProgress
    state = RunProgress.reserve(tmp_path / "output.json", tmp_path / "helper.json")
    state.update("fallback")
    state.fail(ValueError("worker cause"))
    state.fail(KeyboardInterrupt("parent cancelled"))
    doc = state.read()
    assert doc["outcome"] == "interrupted" and doc["phase"] == "fallback"
    assert doc["failure"]["message"] == "worker cause"
    assert doc["parent_failure"]["type"] == "KeyboardInterrupt"
