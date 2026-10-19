"""Shared record-mode control tests. Fake executors are not native qualification."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from pllm import Model
from pllm.configuration import Deployment, ExecutionBudget, Experiment
from pllm.profiles import OpenWeightVerifiableCpu, resolve_runtime_composition
from pllm.runtime import reference_benchmark as quality, quality_outcomes as outcomes


KEY = b"k" * 32


@pytest.fixture
def quality_controls(monkeypatch):
    from pllm.runtime import benchmark_memory, servers, replicated_source, delegated_source
    from pllm.runtime import replicated_transport, delegated_transport
    from pllm.runtime.client import ProtocolError

    class NumericDomainRejected(ValueError):
        pass
    monkeypatch.setattr(outcomes._native, "NumericDomainRejected", NumericDomainRejected, raising=False)
    monkeypatch.setattr(Experiment, "resolve", lambda self: resolve_runtime_composition(self.pipeline))
    manifest = SimpleNamespace(id="frozen-source", vocab_size=5, metadata={"source_lock": {
        "files": [{"path": "model.safetensors", "size": 40}]}})
    resolved = SimpleNamespace(path=Path("/unit-only-source"), manifest=manifest,
        source_lock_digest="a" * 64, checkpoint_digest="b" * 64)
    contract = {"vocabulary": 5, "numeric_profile": "unit-numeric", "local_numeric_profile": "unit-local"}
    source = SimpleNamespace(source=resolved, contract=contract, tokenizer_descriptor={"add_bos_token": False},
        tokenizer=lambda: SimpleNamespace(encode=lambda text, **kwargs: (3 if text == "pass" else 4 if text == "reject" else 2,)))
    monkeypatch.setattr(replicated_source, "admit_source", lambda experiment: source)
    monkeypatch.setattr(delegated_source, "admit_source", lambda experiment: source)
    monkeypatch.setattr(quality, "resolve_model", lambda model: resolved)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 1, __version__="unit"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="unit"))
    calls, lifecycle = [], []

    class Guard:
        def __init__(self, *args):
            pass
        def start(self):
            lifecycle.append("guard_start")
        def check(self):
            pass
        def close(self):
            lifecycle.append("guard_close")

    @contextmanager
    def roles(*args, **kwargs):
        lifecycle.append("roles_start")
        try:
            yield SimpleNamespace(worker_connections=lambda: {})
        finally:
            lifecycle.append("roles_close")

    monkeypatch.setattr(benchmark_memory, "benchmark_memory", lambda *args, **kwargs: {"admitted": True})
    monkeypatch.setattr(benchmark_memory, "MemoryWatchdog", Guard)
    monkeypatch.setattr(servers, "build_roles", roles)

    def execute(experiment, source, ids, outputs, *args, **kwargs):
        calls.append(tuple(ids))
        if ids[0] == 4:
            raise NumericDomainRejected("INTERNAL_SECRET_SITE")
        if ids[0] == 2:
            raise ProtocolError("SECRET_PROMPT transport failed", 401)
        vector = np.array([0, 0, 0, 0, 1], dtype="<f4").tobytes()
        return SimpleNamespace(selected=(4,), logits=(vector,) * outputs,
            executed=tuple(ids) + tuple(kwargs.get("decode_tokens") or (4,) * (outputs - 1)))

    monkeypatch.setattr(replicated_transport, "execute_request", execute)
    monkeypatch.setattr(delegated_transport, "execute_request", execute)

    def reference(source, configurations, ids, continuations, *args):
        return [[np.array([0, 0, 0, 0, 1], dtype=np.float32) for _ in range(len(teacher) + 1)]
                for teacher in continuations], None
    monkeypatch.setattr(quality, "_reference_trajectories", reference)
    return SimpleNamespace(calls=calls, lifecycle=lifecycle, source=source, rejection=NumericDomainRejected)


def experiment(kind="puma", name="quality-candidate"):
    if kind == "puma":
        from test_replicated_execution_policy import experiment_for
        return experiment_for(weights="public", range_policy="checked", lookup="tiled",
            selection="tiled", peer_transport="streams").with_params(name=name,
                budget__requests=8, budget__max_new_tokens=3)
    return Experiment(name, OpenWeightVerifiableCpu(Model.path("/unit-only-source")),
        Deployment.local(root="/unit-only-roles"), ExecutionBudget(requests=8, max_input_tokens=8, max_new_tokens=3))


def run(candidate, prompts, **kwargs):
    return quality.run_reference_benchmark([candidate], prompts, outcomes="record",
        cohort_key=KEY, cohort_key_id="test-campaign", **kwargs)


@pytest.mark.parametrize("kind", ["puma", "gkm"])
def test_mixed_positions_through_shared_runner_and_adapters(quality_controls, kind):
    report = run(experiment(kind), ["pass", "reject", "pass"])
    row, = report["candidates"]
    assert row["coverage"] == {"total": 3, "accepted": 2, "rejected": 1, "failed": 0}
    assert row["top1_agreement"] is None
    assert row["accepted_agreement"]["top1_agreement"] == 1
    assert [item["status"] for item in row["outcomes"]] == ["accepted", "rejected", "accepted"]
    assert quality_controls.calls == [(3,), (4,), (3,)]
    assert quality_controls.lifecycle == ["guard_start", "roles_start", "roles_close", "guard_close"]
    outcomes.read_report(json.dumps(report))
    with pytest.raises(ValueError, match="partial"):
        outcomes.require_complete_for_ranking(report)
    persisted = json.dumps(report)
    assert "INTERNAL_SECRET_SITE" not in persisted and "unit-only-source" not in persisted
    assert '"pass"' not in persisted and '"reject"' not in persisted


def test_transport_failure_fences_remaining_positions_not_numeric(quality_controls):
    report = run(experiment(), ["pass", "boom", "pass"])
    row, = report["candidates"]
    assert row["coverage"] == {"total": 3, "accepted": 1, "rejected": 0, "failed": 2}
    assert row["outcomes"][1]["code"] == "transport_failed"
    assert row["outcomes"][2]["attempted"] is False
    assert report["terminal_state"] == "execution_failed"
    assert quality_controls.calls == [(3,), (2,)]
    assert "SECRET" not in json.dumps(report)
    outcomes.validate_report(report)


def test_default_v1_still_aborts(quality_controls):
    with pytest.raises(quality_controls.rejection):
        quality.run_reference_benchmark([experiment()], ["pass", "reject", "pass"])
    assert quality_controls.calls == [(3,), (4,)]


def test_record_decode_uses_frozen_teacher_even_after_rejection(quality_controls):
    report = run(experiment(), ["reject", "pass"], decode_steps=1, continuations=[(3,), (3,)])
    row, = report["candidates"]
    assert report["cohort"]["same_token_decode"]["teacher_forcing"] == "caller-frozen"
    assert quality_controls.calls == [(4,), (4, 3), (3,), (3, 3)]
    assert row["all_positions"]["coverage"] == {"total": 4, "accepted": 2, "rejected": 2, "failed": 0}
    outcomes.validate_report(report)


def test_record_decode_without_teacher_fails_before_execution(quality_controls):
    with pytest.raises(quality.ReferenceBenchmarkError, match="frozen"):
        run(experiment(), ["pass"], decode_steps=1)
    assert not quality_controls.calls


def test_json_and_atomic_output_are_strict(quality_controls, tmp_path):
    report = run(experiment(), ["pass"])
    path = tmp_path / "quality.json"
    outcomes.write_report(report, path)
    assert outcomes.read_report(path.read_bytes()) == report
    with pytest.raises(FileExistsError):
        outcomes.write_report(report, path)
    outcomes.write_report(report, path, force=True)
    bad = json.loads(path.read_text())
    bad["candidates"][0]["coverage"]["accepted"] = True
    with pytest.raises(ValueError):
        outcomes.validate_report(bad)
    with pytest.raises(ValueError):
        outcomes.read_report('{"schema_version":1,"schema_version":2}')


def test_native_error_text_never_classifies_as_numeric(quality_controls):
    assert not outcomes.numeric_rejection(ValueError("replicated request failed protected range checks"))
    assert outcomes.numeric_rejection(quality_controls.rejection("anything"))


@pytest.mark.parametrize("prompts,success", [(["pass", "reject", "pass"], True), (["pass", "boom", "pass"], False)])
def test_exact_cli_record_invocation_persists_outcomes(quality_controls, tmp_path, capsys, prompts, success):
    from pllm.cli import main

    candidate = experiment()
    configuration, cohort, key, output = [tmp_path / name for name in ("experiment.json", "prompts.json", "cohort.key", "quality.json")]
    configuration.write_bytes(candidate.canonical_bytes())
    cohort.write_text(json.dumps(prompts))
    key.write_bytes(KEY)
    try:
        main(["benchmark", "quality", "--experiment", str(configuration), "--prompts-file", str(cohort),
            "--outcomes", "record", "--cohort-key-file", str(key), "--cohort-key-id", "test-campaign",
            "--output", str(output), "--format", "json"])
        code = 0
    except SystemExit as error:
        code = error.code
    assert (code == 0) is success
    report = outcomes.read_report(output.read_bytes())
    assert report["schema_version"] == outcomes.SCHEMA
    counts = report["candidates"][0]["all_positions"]["coverage"]
    assert counts["total"] == 3 and counts["accepted"] == (2 if success else 1)
    if success:
        assert json.loads(capsys.readouterr().out)["data"]["report"] == report
    else:
        assert report["terminal_state"] == "execution_failed"
        assert "SECRET_PROMPT" not in capsys.readouterr().out


def test_record_reference_failure_keeps_coverage_and_null_scores(quality_controls, monkeypatch):
    def fail(*args):
        raise ValueError("SECRET_REFERENCE_TENSOR")
    monkeypatch.setattr(quality, "_reference_trajectories", fail)
    report = run(experiment(), ["pass", "reject", "pass"])
    assert report["reference_status"] == "unavailable"
    row, = report["candidates"]
    assert row["coverage"]["accepted"] == 2 and row["top1_agreement"] is None
    assert row["sample_count"] == 0
    assert "SECRET" not in json.dumps(report)
    outcomes.validate_report(report)


def test_record_cancellation_seals_then_raises_and_stops_owners(quality_controls, monkeypatch):
    from pllm.runtime import replicated_transport
    original = replicated_transport.execute_request
    def cancel(experiment, source, ids, *args, **kwargs):
        if ids == (4,):
            raise KeyboardInterrupt()
        return original(experiment, source, ids, *args, **kwargs)
    monkeypatch.setattr(replicated_transport, "execute_request", cancel)
    with pytest.raises(outcomes.QualityRunCancelled) as cancelled:
        run(experiment(), ["pass", "reject", "pass"])
    report = cancelled.value.report
    assert report["terminal_state"] == "cancelled"
    counts = report["candidates"][0]["coverage"]
    assert counts == {"total": 3, "accepted": 1, "rejected": 0, "failed": 2}
    assert quality_controls.lifecycle[-2:] == ["roles_close", "guard_close"]
    outcomes.validate_report(report)
    traceback = cancelled.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "run_reference_benchmark":
            private = traceback.tb_frame.f_locals
            assert all(private[name] is None for name in ("prompts", "cohort_key", "token_cohort", "continuations", "value"))
            assert not private["expected"] and not private["pending"]
        traceback = traceback.tb_next


def test_clear_kernel_record_uses_shared_ledger_without_numeric_error_string_matching(quality_controls, monkeypatch, tmp_path):
    from pllm.profiles import MaskedLinearCpu
    import pllm.modeling

    quality_controls.source.source.path = tmp_path
    (tmp_path / "config.json").write_text("{}")
    vector = np.array([0, 0, 0, 0, 1], dtype=np.float32)
    closed = []
    class Engine:
        def __init__(self, **kwargs):
            self._public_equalization_profile = None
        async def load(self, manifest):
            pass
        def client_bundle(self, model):
            return b"unit-only"
    class Session:
        def prefill_ids(self, ids):
            if ids == (4,):
                raise ValueError("failed protected range checks SECRET_CLEAR_VALUE")
            return vector.copy()
        def close(self):
            closed.append(True)
    compiled = SimpleNamespace(validate=lambda: None, session=lambda remote: Session(),
        runtime=lambda remote: SimpleNamespace(encode_prompt=lambda text: (3 if text == "pass" else 4,)))
    monkeypatch.setattr(quality, "MaskedTransformerEngine", Engine)
    monkeypatch.setattr(quality.ClientBundle, "unpack", lambda value: object())
    monkeypatch.setattr(pllm.modeling, "lower_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(quality, "compile_runtime_model", lambda *args, **kwargs: compiled)
    monkeypatch.setattr(quality, "_candidate_remote", lambda *args: object())
    candidate = Experiment("clear-shared", MaskedLinearCpu(Model.path(str(tmp_path))),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(requests=3, max_input_tokens=8, max_new_tokens=1))
    report = run(candidate, ["pass", "reject", "pass"])
    counts = report["candidates"][0]["coverage"]
    assert counts == {"total": 3, "accepted": 1, "rejected": 0, "failed": 2}
    assert len(closed) == 2
    assert "SECRET_CLEAR_VALUE" not in json.dumps(report)
    outcomes.validate_report(report)


def test_shared_public_facade_passes_record_contract(quality_controls):
    from pllm.evidence import benchmark_reference_quality
    report = benchmark_reference_quality(iter([experiment()]), iter(["pass", "reject"]),
        outcomes="record", cohort_key=KEY, cohort_key_id="test-campaign")
    assert report["candidates"][0]["coverage"]["rejected"] == 1


def test_ranking_path_rejects_any_partial_candidate_before_subset_selection(quality_controls):
    from pllm.search import QualityLockedNetworkSearch, SearchError
    report = run(experiment(), ["pass", "reject"])
    search = QualityLockedNetworkSearch(0, 0, 1)
    with pytest.raises(SearchError, match="complete accepted"):
        search.select(report, [])


def test_complete_v2_enters_existing_quality_locked_ranking(quality_controls):
    from pllm.profiles import MaskedLinearCpu
    from pllm.search import QualityLockedNetworkSearch
    report = run(experiment("gkm"), ["pass"])
    # Replay-only ranking fixture: full-vector metrics + ordinary clear scope.
    # This does not qualify an actual model/network measurement.
    candidate = Experiment("clear-ranking", MaskedLinearCpu(Model.path("/unit-only-source", model_id="frozen-source")),
        Deployment.local(root="/unit-only-roles"), ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1))
    digest = candidate.configuration_digest()
    report["scope"] = "local-compiled-clear-kernel-prefill-reference"
    report["candidates"][0].update(configuration_digest=digest, pipeline_digest=candidate.pipeline.digest())
    network = {"schema_version": "pllm.loopback_benchmark.v1", "configuration": {
        "source_lock_digest": report["model"]["source_lock_digest"], "prompt_digest": "c" * 64,
        "warmup_prompt_digest": "d" * 64, "roles": ["worker"], "warmups": 0, "repetitions": 1},
        "experiment": {"configuration_digest": digest, "pipeline_digest": candidate.pipeline.digest()},
        "checks": {"passed": True}, "runs": [{"tokens": {"input_tokens": 1, "output_tokens": 1},
            "max_output_tokens": 1, "warm": False}],
        "summary": {"median_online_all_link_serialized_body_bytes": 123}}
    result = QualityLockedNetworkSearch(0, 0, 1).select(report, [(candidate, network)])
    assert result["winner_configuration_digest"] == digest


@pytest.mark.parametrize("mutation", ["extra", "position", "nan", "mask"])
def test_replay_rejects_unknown_payloads_and_forged_coverage(quality_controls, mutation):
    report = run(experiment(), ["pass", "reject"])
    row = report["candidates"][0]
    if mutation == "extra":
        row["tokens"] = [4]
    elif mutation == "position":
        row["all_positions"]["outcomes"][0]["position"] = True
    elif mutation == "nan":
        row["accepted_agreement"]["top1_agreement"] = float("nan")
    else:
        row["all_positions"]["outcomes"].reverse()
    with pytest.raises(ValueError):
        outcomes.validate_report(report)


def test_maximum_logical_outcomes_fit_bounded_serialization(quality_controls, tmp_path):
    # Ledger/schema bound only, not a model/native-work qualification.
    import copy
    report = run(experiment(), ["pass"])
    report["cohort"].update(prompt_count=32, position_count=288, input_token_counts=[1] * 32,
        same_token_decode={"steps_per_prompt": 8, "teacher_forcing": "caller-frozen", "token_cohort_digest": "e" * 64})
    report["scope"] = "local-compiled-decoder-prefill-and-same-token-decode-reference"
    batch = outcomes.CandidateOutcomes(288)
    for index in range(288):
        batch.record(index, outcomes.Outcome("accepted"), 4)
    expected = {index: None for index in range(288)}
    metric = lambda *args: {"top1_agreement": 1., "top_k_recall": None, "max_abs_logit_error": None}
    prefill = outcomes.summary(batch, list(range(0, 288, 9)), expected, metric)
    decode = outcomes.summary(batch, [index for index in range(288) if index % 9], expected, metric)
    all_positions = outcomes.summary(batch, list(range(288)), expected, metric)
    report["candidates"] = [{"name": f"bound-{index}", "configuration_digest": f"{index:064x}",
        "pipeline_digest": "f" * 64, **copy.deepcopy(prefill), "same_token_decode": copy.deepcopy(decode),
        "all_positions": copy.deepcopy(all_positions)} for index in range(8)]
    output = tmp_path / "bounded.json"
    outcomes.write_report(report, output)
    assert output.stat().st_size <= 1 << 20
    assert len(outcomes.read_report(output.read_bytes())["candidates"]) == 8
    with pytest.raises(ValueError):
        outcomes.CandidateOutcomes(289)


def test_runtime_memory_admission_failure_is_failed_unattempted_not_rejected(quality_controls, monkeypatch):
    from pllm.runtime import benchmark_memory
    def reject(*args, **kwargs):
        raise benchmark_memory.BenchmarkMemoryError("SECRET_ALLOCATION", {})
    monkeypatch.setattr(benchmark_memory, "benchmark_memory", reject)
    report = run(experiment(), ["pass", "pass"])
    row = report["candidates"][0]
    assert row["coverage"] == {"total": 2, "accepted": 0, "rejected": 0, "failed": 2}
    assert row["outcomes"][0] == {"position": 0, "status": "failed", "code": "admission_failed", "attempted": False}
    assert not quality_controls.calls and not quality_controls.lifecycle
    assert "SECRET" not in json.dumps(report)
    outcomes.validate_report(report)


@pytest.mark.parametrize("global_mismatch", [False, True])
def test_tokenizer_substitution_never_executes_or_becomes_numeric(quality_controls, global_mismatch):
    instances = []
    def tokenizer():
        instances.append(True)
        token = 3 if len(instances) == 1 else 4
        return SimpleNamespace(encode=lambda *args, **kwargs: (token,))
    quality_controls.source.tokenizer = tokenizer
    if global_mismatch:
        with pytest.raises(quality.ReferenceBenchmarkError, match="preflight binding"):
            quality.run_reference_benchmark([experiment(name="left"), experiment("gkm", name="right")], ["pass"], outcomes="record",
                cohort_key=KEY, cohort_key_id="test-campaign")
    else:
        report = run(experiment(), ["pass"])
        row, = report["candidates"]
        assert row["coverage"] == {"total": 1, "accepted": 0, "rejected": 0, "failed": 1}
        assert row["outcomes"][0]["code"] == "source_integrity_failed"
        assert row["outcomes"][0]["attempted"] is False
        outcomes.validate_report(report)
    assert not quality_controls.calls and not quality_controls.lifecycle


def test_cli_frozen_decode_and_human_partial_scores_are_explicit(quality_controls, tmp_path, capsys):
    from pllm.cli import main
    configuration, cohort, teacher, key, output = [tmp_path / name for name in (
        "experiment.json", "prompts.json", "teacher.json", "cohort.key", "quality.json")]
    configuration.write_bytes(experiment().canonical_bytes())
    cohort.write_text(json.dumps(["pass", "reject"]))
    teacher.write_text(json.dumps([[3], [3]]))
    key.write_bytes(KEY)
    main(["benchmark", "quality", "--experiment", str(configuration), "--prompts-file", str(cohort),
        "--outcomes", "record", "--cohort-key-file", str(key), "--cohort-key-id", "test-campaign",
        "--decode-steps", "1", "--continuations-file", str(teacher), "--output", str(output)])
    report = outcomes.read_report(output.read_bytes())
    assert report["candidates"][0]["all_positions"]["coverage"]["rejected"] == 2
    text = capsys.readouterr().out
    assert "Accepted-only diagnostic" in text and "not a complete-cohort score" in text
    assert "top-1 unknown" in text


def test_cli_cancellation_writes_sanitized_report_before_exit130(quality_controls, tmp_path, monkeypatch):
    from pllm.cli import main
    from pllm.runtime import replicated_transport
    configuration, cohort, key, output = [tmp_path / name for name in (
        "experiment.json", "prompts.json", "cohort.key", "quality.json")]
    configuration.write_bytes(experiment().canonical_bytes())
    cohort.write_text(json.dumps(["pass"]))
    key.write_bytes(KEY)
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(replicated_transport, "execute_request", cancel)
    with pytest.raises(SystemExit) as stopped:
        main(["benchmark", "quality", "--experiment", str(configuration), "--prompts-file", str(cohort),
            "--outcomes", "record", "--cohort-key-file", str(key), "--cohort-key-id", "test-campaign",
            "--output", str(output), "--format", "json"])
    assert stopped.value.code == 130
    report = outcomes.read_report(output.read_bytes())
    assert report["terminal_state"] == "cancelled"
    assert report["candidates"][0]["outcomes"][0]["code"] == "cancelled"
    assert quality_controls.lifecycle[-2:] == ["roles_close", "guard_close"]


@pytest.mark.parametrize("facade", [False, True])
def test_reference_cancellation_drops_successful_result_and_wrapper_aliases(quality_controls, monkeypatch, facade):
    from pllm.evidence import benchmark_reference_quality
    reference_frames = []
    def cancel_reference(*args):
        private_reference_value = "PRIVATE_REFERENCE_PAYLOAD"
        reference_frames.append(sys._getframe())
        raise KeyboardInterrupt()
    monkeypatch.setattr(quality, "_reference_trajectories", cancel_reference)
    prompts, teacher, key = ["pass"], [[3]], b"r" * 32
    runner = benchmark_reference_quality if facade else quality.run_reference_benchmark
    with pytest.raises(outcomes.QualityRunCancelled) as cancelled:
        runner([experiment()], prompts, outcomes="record", cohort_key=key,
            cohort_key_id="reference-cancel", decode_steps=1, continuations=teacher)
    report = cancelled.value.report
    assert report["terminal_state"] == "cancelled"
    assert report["candidates"][0]["all_positions"]["coverage"]["accepted"] == 2
    assert quality_controls.calls == [(3,), (3, 3)]
    assert reference_frames and all(not frame.f_locals for frame in reference_frames)
    found = set()
    traceback = cancelled.value.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == "run_reference_benchmark":
            found.add("runtime")
            if facade:
                assert not frame.f_locals  # Inactive inner frame cleared by facade.
            else:
                local = frame.f_locals
                for name in ("result_rows", "kwargs", "cohort_key", "prompts", "continuations", "token_cohort", "selected", "teacher"):
                    assert local[name] is None, name
                assert not local["pending"] and not local["expected"]
        if frame.f_code.co_name == "benchmark_reference_quality":
            found.add("facade")
            local = frame.f_locals
            assert not local["extra"]
            assert all(local[name] is None for name in ("prompts", "cohort", "cohort_key", "continuations", "choices"))
        traceback = traceback.tb_next
    assert found == ({"runtime", "facade"} if facade else {"runtime"})
    assert prompts == ["pass"] and teacher == [[3]] and key == b"r" * 32


@pytest.mark.parametrize("failure", ["transport", "cancellation"])
@pytest.mark.parametrize("following", ["puma", "gkm"])
def test_already_stopped_compiled_candidate_is_unattempted_candidate_stopped(quality_controls, monkeypatch, failure, following):
    from pllm.runtime import replicated_transport
    from pllm.runtime.client import ProtocolError
    original = replicated_transport.execute_request
    def fail(candidate, *args, **kwargs):
        if candidate.name == "first":
            if failure == "cancellation":
                raise KeyboardInterrupt()
            raise ProtocolError("PRIVATE_TRANSPORT_DETAIL", 503)
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(replicated_transport, "execute_request", fail)
    candidates = [experiment(name="first"), experiment(following, name="following")]
    kwargs = dict(outcomes="record", cohort_key=KEY, cohort_key_id="stopped")
    if failure == "cancellation":
        with pytest.raises(outcomes.QualityRunCancelled) as cancelled:
            quality.run_reference_benchmark(candidates, ["pass"], **kwargs)
        report = cancelled.value.report
    else:
        report = quality.run_reference_benchmark(candidates, ["pass"], **kwargs)
    assert report["candidates"][1]["outcomes"] == [
        {"position": 0, "status": "failed", "code": "candidate_stopped", "attempted": False}]
    assert quality_controls.lifecycle == ["guard_start", "roles_start", "roles_close", "guard_close"]
    assert not quality_controls.calls
    outcomes.validate_report(report)


@pytest.mark.parametrize("as_bytes", [False, True])
def test_multibyte_report_limit_counts_utf8_bytes_before_parsing(monkeypatch, as_bytes):
    text = '"' + "é" * (1 << 19) + '"'
    assert len(text) < 1 << 20 < len(text.encode("utf-8"))
    def unexpected_parse(*args, **kwargs):
        pytest.fail("oversized UTF-8 report reached parser")
    monkeypatch.setattr(outcomes.json, "loads", unexpected_parse)
    with pytest.raises(ValueError, match="size bound"):
        outcomes.read_report(text.encode("utf-8") if as_bytes else text)


def test_cancellation_after_successful_candidate_releases_retained_result(quality_controls, monkeypatch):
    from pllm.runtime import replicated_transport
    original = replicated_transport.execute_request
    def cancel_second(candidate, *args, **kwargs):
        if candidate.name == "second":
            raise KeyboardInterrupt()
        return original(candidate, *args, **kwargs)
    monkeypatch.setattr(replicated_transport, "execute_request", cancel_second)
    prompts, teacher = ["pass"], [[3]]
    with pytest.raises(outcomes.QualityRunCancelled) as cancelled:
        quality.run_reference_benchmark([experiment(name="first"), experiment(name="second"),
            experiment("gkm", name="third")], prompts, outcomes="record", cohort_key=KEY,
            cohort_key_id="after-success", decode_steps=1, continuations=teacher)
    report = cancelled.value.report
    assert report["candidates"][0]["all_positions"]["coverage"]["accepted"] == 2
    assert all(row["code"] == "candidate_stopped" and not row["attempted"]
               for row in report["candidates"][2]["all_positions"]["outcomes"])
    assert quality_controls.calls == [(3,), (3, 3)]
    traceback = cancelled.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == "run_reference_benchmark":
            local = traceback.tb_frame.f_locals
            assert local["result_rows"] is None and local["kwargs"] is None
            assert local["selected"] is None and not local["pending"]
        traceback = traceback.tb_next
    assert prompts == ["pass"] and teacher == [[3]]
