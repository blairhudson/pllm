"""Ordinary CLI/scorer conformance with synthetic execution, never trained evidence."""

import importlib
import hashlib
import json
from pathlib import Path
import shlex
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

# Re-export the safe-source fixture for pytest discovery in this module.
from test_paper_gkm_quality_experiment import quality as quality


ROOT = Path(__file__).resolve().parents[1]


def documented_argv(output):
    section = (ROOT / "examples/papers/README.md").read_text().split(
        "## Broader GKM source-model quality", 1
    )[1]
    command = shlex.split(section.split("```sh\n", 1)[1].split("```", 1)[0].replace("\\\n", " "))
    assert command[:5] == ["python", "-m", "pllm", "benchmark", "quality"]
    # The documented command is executed, changing only its artifact destination.
    command[command.index("--output") + 1] = str(output)
    return command[3:]


@pytest.fixture
def lane(quality, monkeypatch, tmp_path):
    """Real factories, native config resolution, adapter, scorer and CLI; fake I/O."""
    from pllm.protocols import OpenWeightVerifiableOutsourcing as GKM

    def module(name, **attrs):
        result = ModuleType(name)
        result.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, result)
        return result

    forbidden = Mock(side_effect=AssertionError("W8A8/plaintext fallback or model load attempted"))
    module("pllm.runtime.model_binding", compile_runtime_model=forbidden)
    module("pllm.runtime.transformer_client", ClientBundle=forbidden)
    module("pllm.runtime.transformer_engine", MaskedTransformerEngine=forbidden)
    module("pllm.runtime.client", ProtocolError=type("ProtocolError", (RuntimeError,), {}))
    module("torch", __version__="synthetic-test-only", get_num_threads=lambda: 1)
    module("transformers", __version__="synthetic-test-only", AutoModelForCausalLM=forbidden)

    prompts = json.loads((ROOT / quality.PROMPTS_FILE).read_text())
    tokens = {prompt: (index % 5, 1) for index, prompt in enumerate(prompts)}
    tokenizer = SimpleNamespace(encode=Mock(side_effect=lambda prompt, **_: tokens[prompt]))
    vector = np.array([0, 1, 4, 2, 3], dtype="<f4")
    source_path = tmp_path / "synthetic-source-not-a-checkpoint"
    source_path.mkdir()
    source_file = source_path / "model.safetensors"
    source_file.write_bytes(vector.tobytes())  # Numeric fixture, NOT a Safetensors model.
    model = quality.experiment().pipeline.model
    pin_events = []

    def original_source_lock(path, selected_model):
        """Finite source-I/O double with the ordinary reader's exact triple/schema.

        Re-read fixture bytes on EVERY invocation, including the reference's real
        pre/post guards. Never return the cached original pins as current truth.
        This path is not an HF cache; publisher/Hub authentication is not mocked.
        """
        assert path == source_path and selected_model == model
        rows = []
        for file in sorted(path.iterdir()):
            before = file.stat()
            data = file.read_bytes()
            after = file.stat()
            assert (before.st_ino, before.st_size, before.st_mtime_ns) == (
                after.st_ino, after.st_size, after.st_mtime_ns)
            rows.append({"path": file.name, "size": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()})

        def digest(domain, value):
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                                 sort_keys=True, separators=(",", ":")).encode()
            return hashlib.sha256(domain + encoded).hexdigest()

        checkpoint = digest(b"pllm.checkpoint.v1\0", {"files": rows})
        lock = {"schema": "pllm.model_source_lock.v1", "kind": model.kind,
                "repo_id": None, "revision": model.revision, "commit": None,
                "files": rows, "checkpoint_digest": checkpoint}
        return checkpoint, digest(b"pllm.model_source_lock.v1\0", lock), lock

    checkpoint, source_lock_digest, source_lock = original_source_lock(source_path, model)

    def current_source_lock(path, selected_model):
        pin_events.append("reference_pin_check")
        return original_source_lock(path, selected_model)

    source_reader = Mock(side_effect=current_source_lock)
    # Extend the fixture's existing fake module, not the production pin validator.
    monkeypatch.setattr(sys.modules["pllm.model_loader"], "_source_lock", source_reader, raising=False)
    resolved = SimpleNamespace(
        path=source_path, model=model,
        source_lock_digest=source_lock_digest, checkpoint_digest=checkpoint,
        manifest=SimpleNamespace(id="synthetic-test-only", vocab_size=5,
            metadata={"source_lock": source_lock}),
    )
    source = SimpleNamespace(source=resolved, tokenizer=lambda: tokenizer,
        tokenizer_descriptor={"add_bos_token": False}, contract={
            "vocabulary": 5, "numeric_profile": GKM.NUMERIC_PROFILE,
            "local_numeric_profile": GKM.LOCAL_NUMERIC_PROFILE,
        })
    admission = Mock(return_value=source)
    module("pllm.runtime.delegated_source", admit_source=admission)

    def candidate_source_check(experiment, phase):
        sharing = experiment.pipeline.components["linear"].params["sharing"]
        pin_events.append(("candidate", sharing, phase))
        # Model the existing source import's own pre/post source-lock checks;
        # quality's reference guard still runs independently through its reader.
        if original_source_lock(source_path, model) != (checkpoint, source_lock_digest, source_lock):
            raise ValueError("synthetic candidate source changed")

    def accepted(experiment, admitted, ids, selections, connections, timeout, cancelled,
                  *, decode_tokens=None):
        assert admitted is source
        assert experiment.pipeline.model == model
        candidate_source_check(experiment, "before")
        assert selections == quality.DECODE_STEPS + 1
        continuation = (2,) * quality.DECODE_STEPS
        if decode_tokens is not None:
            assert tuple(decode_tokens) == continuation
        # This is a synthetic transport acceptance boundary, not a native proof.
        result = SimpleNamespace(logits=[vector.tobytes()] * selections,
                                 executed=tuple(ids) + continuation)
        candidate_source_check(experiment, "after")
        return result

    execute = Mock(side_effect=accepted)
    module("pllm.runtime.delegated_transport", execute_request=execute)
    topology = Mock()
    topology.__enter__ = Mock(return_value=topology)
    topology.__exit__ = Mock(return_value=False)
    roles = Mock(return_value=topology)
    module("pllm.runtime.servers", build_roles=roles)

    reference = SimpleNamespace(
        trajectory=Mock(side_effect=lambda ids, continuation: [vector.copy()] * (len(continuation) + 1)),
        weight_bound=20, allocation_estimate_bytes=20, kv_bound_bytes=0, buffer_bytes=0,
        _guard=SimpleNamespace(maximum_swap_growth_bytes=0), stats={},
    )
    reference_context = Mock()
    reference_context.__enter__ = Mock(return_value=reference)
    reference_context.__exit__ = Mock(return_value=False)
    streamed = Mock(return_value=reference_context)
    module("pllm.runtime.streamed_reference", StreamedFloat32Reference=streamed)

    benchmark_name = "pllm.runtime.reference_benchmark"
    already_loaded = benchmark_name in sys.modules
    benchmark = importlib.import_module(benchmark_name)
    # Also protect suites where this module was imported before this fixture.
    monkeypatch.setattr(benchmark, "resolve_model", forbidden)
    monkeypatch.setattr(benchmark, "compile_runtime_model", forbidden)
    monkeypatch.setattr(benchmark, "MaskedTransformerEngine", forbidden)
    memory = importlib.import_module("pllm.runtime.benchmark_memory")
    monkeypatch.setattr(memory, "benchmark_memory", Mock(return_value={"synthetic_test_only": True}))
    monkeypatch.setattr(memory, "require_admission", Mock())
    watchdog = Mock()
    monkeypatch.setattr(memory, "MemoryWatchdog", Mock(return_value=watchdog))
    monkeypatch.chdir(ROOT)
    try:
        yield SimpleNamespace(quality=quality, benchmark=benchmark, source=source, vector=vector,
            admission=admission, execute=execute, accepted=accepted, reference=reference,
            streamed=streamed, forbidden=forbidden, roles=roles, watchdog=watchdog,
            prompts=prompts, tokens=tokens, output=tmp_path / "synthetic-quality.json",
            source_file=source_file, source_reader=source_reader, pin_events=pin_events,
            topology=topology)
    finally:
        if not already_loaded:
            # Do not leave imports of synthetic execution dependencies cached for
            # subsequent real integration tests in the same pytest process.
            sys.modules.pop(benchmark_name, None)
            vars(sys.modules["pllm.runtime"]).pop("reference_benchmark", None)


def test_quality_cli_help_uses_real_parser_without_models(quality, capsys):
    from pllm.cli import main

    with pytest.raises(SystemExit) as stopped:
        main(["benchmark", "quality", "--help"])
    assert stopped.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--experiment", "--reference-storage", "--decode-steps", "--outcomes"):
        assert flag in help_text


def test_documented_cli_scores_both_real_factories_on_shared_source_and_tokens(lane, capsys):
    from pllm._cli.app import build_parser
    from pllm.cli import main

    argv = documented_argv(lane.output)
    args = build_parser().parse_args(argv)
    assert args.experiment == ["examples.papers.gkm_quality:experiment", "examples.papers.gkm_quality:seeded"]
    assert args.factory and args.trust_python
    assert args.reference_storage == "streamed" and args.outcomes == "abort"
    assert args.decode_steps == lane.quality.DECODE_STEPS
    assert args.memory_safety_margin_percent == 25 and not args.force
    main(argv)
    envelope = json.loads(capsys.readouterr().out)
    report = json.loads(lane.output.read_text())
    assert envelope["data"]["report"] == report
    assert report["schema_version"] == "pllm.reference_quality_benchmark.v1"
    assert report["model"]["source_lock_digest"] == lane.source.source.source_lock_digest
    assert report["model"]["checkpoint_digest"] == lane.source.source.checkpoint_digest
    assert report["environment"]["reference_dtype"] == "float32"
    assert report["environment"]["reference_attention"] == "eager"
    assert report["environment"]["reference_weight_storage"] == "module-streamed"
    assert report["cohort"]["prompt_count"] == len(lane.prompts)
    assert report["cohort"]["input_token_counts"] == [2] * len(lane.prompts)
    factories = [lane.quality.experiment(), lane.quality.seeded()]
    assert [row["configuration_digest"] for row in report["candidates"]] == [
        experiment.configuration_digest() for experiment in factories
    ]
    assert report["cohort"]["same_token_decode"]["teacher_configuration_digest"] == factories[0].configuration_digest()
    for row in report["candidates"]:
        assert row["sample_count"] == len(lane.prompts)
        assert row["same_token_decode"]["sample_count"] == len(lane.prompts) * lane.quality.DECODE_STEPS
        assert row["execution"] == "two-worker-request-end-verified"
        assert row["weight_bits"] == 16 and row["activation_bits"] is None
        assert row["numeric_profile"] == lane.source.contract["numeric_profile"]
        assert row["local_numeric_profile"] == lane.source.contract["local_numeric_profile"]
        assert row["top1_agreement"] == 1 and row["max_abs_logit_error"] == 0
    assert lane.execute.call_count == len(factories) * len(lane.prompts)
    for call, factory in zip(lane.admission.call_args_list, factories, strict=True):
        assert call.args[0].pipeline.model == factory.pipeline.model
    lane.streamed.assert_called_once_with(lane.source.source.path, max_input_tokens=64,
        max_decode_steps=3, memory_safety_margin_percent=25)
    assert [call.args for call in lane.reference.trajectory.call_args_list] == [
        (lane.tokens[prompt], (2, 2, 2)) for prompt in lane.prompts
    ]
    lane.forbidden.assert_not_called()
    assert lane.watchdog.close.call_count == 2
    assert lane.source_reader.call_count == 2
    assert lane.pin_events == [
        ("candidate", sharing, phase)
        for sharing in ("full", "seeded") for _ in lane.prompts for phase in ("before", "after")
    ] + ["reference_pin_check"] * 2


@pytest.mark.parametrize("bad_output", ["missing", "width", "nonfinite", "ids-only"])
def test_final_accepted_logits_must_cover_every_selection_and_vocabulary(lane, bad_output, capsys):
    from pllm.cli import main

    def malformed(*args, **kwargs):
        result = lane.accepted(*args, **kwargs)
        if bad_output == "missing":
            result.logits.pop()
        elif bad_output == "width":
            result.logits[0] = np.zeros(4, dtype="<f4").tobytes()
        elif bad_output == "nonfinite":
            result.logits[0] = np.full(5, np.nan, dtype="<f4").tobytes()
        else:
            result.logits = []  # Final IDs do not authorize invented GKM logits.
        return result

    lane.execute.side_effect = malformed
    with pytest.raises(SystemExit) as stopped:
        main(documented_argv(lane.output))
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    assert captured.out == "" and not lane.output.exists()
    assert json.loads(captured.err)["error"]["code"] == "QUALITY_FAILED"
    lane.streamed.assert_not_called()
    assert lane.execute.call_count == 1
    lane.watchdog.close.assert_called_once()


@pytest.mark.parametrize("failed_call", [1, 13])
def test_abort_preserves_method_failure_instead_of_success_subset(lane, failed_call, capsys):
    from pllm.cli import main

    count = 0

    def fail(*args, **kwargs):
        nonlocal count
        count += 1
        if count == failed_call:
            raise ValueError("synthetic final acceptance failure")
        return lane.accepted(*args, **kwargs)

    lane.execute.side_effect = fail
    with pytest.raises(SystemExit) as stopped:
        main(documented_argv(lane.output))
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    diagnostic = json.loads(captured.err)["error"]
    assert diagnostic["code"] == "QUALITY_FAILED"
    assert "synthetic final acceptance failure" in diagnostic["message"]
    assert captured.out == "" and not lane.output.exists()
    assert lane.execute.call_count == failed_call
    lane.streamed.assert_not_called()
    lane.forbidden.assert_not_called()


def test_record_mode_retains_declared_positions_and_failed_method(lane, capsys, tmp_path):
    from pllm.cli import main
    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking

    key = tmp_path / "synthetic-cohort.key"
    key.write_bytes(b"k" * 32)
    continuations = tmp_path / "synthetic-continuations.json"
    continuations.write_text(json.dumps([[2, 2, 2]] * len(lane.prompts)))
    argv = documented_argv(lane.output) + [
        "--outcomes", "record", "--cohort-key-file", str(key),
        "--cohort-key-id", "synthetic-test-only", "--continuations-file", str(continuations),
    ]

    def fail_seeded(experiment, *args, **kwargs):
        if experiment.pipeline.components["linear"].params["sharing"] == "seeded":
            raise ValueError("synthetic final acceptance failure")
        return lane.accepted(experiment, *args, **kwargs)

    lane.execute.side_effect = fail_seeded
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"]["code"] == "QUALITY_EXECUTION_FAILED"
    report = read_report(lane.output.read_bytes())  # Real schema/coverage validation.
    assert json.loads(captured.out)["data"]["report"] == report
    assert report["terminal_state"] == "execution_failed"
    assert report["reference_status"] == "unavailable"
    total = len(lane.prompts) * (lane.quality.DECODE_STEPS + 1)
    assert report["cohort"]["position_count"] == total
    assert [row["configuration_digest"] for row in report["candidates"]] == [
        lane.quality.experiment().configuration_digest(), lane.quality.seeded().configuration_digest(),
    ]
    first, failed = [row["all_positions"] for row in report["candidates"]]
    assert first["coverage"] == {"total": total, "accepted": total, "rejected": 0, "failed": 0}
    assert failed["coverage"] == {"total": total, "accepted": 0, "rejected": 0, "failed": total}
    assert failed["outcomes"][0] == {
        "position": 0, "status": "failed", "code": "execution_failed", "attempted": True,
    }
    assert all(row["code"] == "candidate_stopped" and not row["attempted"]
               for row in failed["outcomes"][1:])
    for section in (first, failed):
        assert section["planned_sample_count"] == total
        assert section["sample_count"] == 0 and section["top1_agreement"] is None
    with pytest.raises(ValueError, match="unavailable"):
        require_complete_for_ranking(report)
    lane.streamed.assert_not_called()
    assert lane.execute.call_count == len(lane.prompts) + 1


def test_reference_vocabulary_shape_mismatch_is_not_scored(lane, capsys):
    from pllm.cli import main

    lane.reference.trajectory.side_effect = lambda ids, continuation: [np.zeros(4, dtype="<f4")] * 4
    with pytest.raises(SystemExit) as stopped:
        main(documented_argv(lane.output))
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"]["code"] == "QUALITY_FAILED"
    assert captured.out == "" and not lane.output.exists()


def test_different_source_lock_aborts_before_workers_or_reference(lane, capsys):
    from pllm.cli import main

    other = SimpleNamespace(**vars(lane.source.source))
    other.source_lock_digest = "c" * 64
    lane.admission.side_effect = [lane.source, SimpleNamespace(**{**vars(lane.source), "source": other})]
    with pytest.raises(SystemExit) as stopped:
        main(documented_argv(lane.output))
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    assert json.loads(captured.err)["error"]["code"] == "QUALITY_FAILED"
    assert "same source lock" in json.loads(captured.err)["error"]["message"]
    assert captured.out == "" and not lane.output.exists()
    lane.execute.assert_not_called()
    lane.streamed.assert_not_called()


@pytest.mark.parametrize("outcomes", ["abort", "record"])
@pytest.mark.parametrize("when", ["before_reference", "during_reference"])
def test_cli_reference_pin_drift_retains_declared_methods_without_scores(
        lane, tmp_path, capsys, outcomes, when):
    from pllm.cli import main

    original_pins = (lane.source.source.checkpoint_digest, lane.source.source.source_lock_digest)
    original_model = lane.quality.experiment().pipeline.model.to_spec()

    def drift():
        # Same byte length and complete vocabulary: acceptance/shape alone cannot
        # catch this different source. Never change cached original report pins.
        lane.source_file.write_bytes(np.array([0, 1, 4, 2, 7], dtype="<f4").tobytes())

    def accepted(*args, **kwargs):
        result = lane.accepted(*args, **kwargs)
        assert len(result.logits) == lane.quality.DECODE_STEPS + 1
        assert all(np.frombuffer(value, dtype="<f4").shape == (5,) for value in result.logits)
        return result

    lane.execute.side_effect = accepted
    if when == "before_reference":
        def retire(*args):
            if lane.roles.call_count == 2:
                drift()  # Both full/seeded candidates already finally accepted.
            return False

        lane.topology.__exit__.side_effect = retire
    else:
        def trajectory(ids, continuation):
            assert tuple(continuation) == (2,) * lane.quality.DECODE_STEPS
            if lane.reference.trajectory.call_count == len(lane.prompts):
                drift()
            return [lane.vector.copy()] * (len(continuation) + 1)

        lane.reference.trajectory.side_effect = trajectory

    argv = documented_argv(lane.output)
    if outcomes == "record":
        key = tmp_path / "pin-drift-cohort.key"
        key.write_bytes(b"k" * 32)
        continuations = tmp_path / "pin-drift-continuations.json"
        continuations.write_text(json.dumps([[2, 2, 2]] * len(lane.prompts)))
        argv += ["--outcomes", "record", "--cohort-key-file", str(key),
                 "--cohort-key-id", "synthetic-pin-drift",
                 "--continuations-file", str(continuations)]

    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    diagnostic = json.loads(captured.err)["error"]
    assert lane.execute.call_count == 2 * len(lane.prompts)
    assert lane.source_reader.call_count == (1 if when == "before_reference" else 2)
    assert lane.streamed.call_count == (0 if when == "before_reference" else 1)
    assert lane.reference.trajectory.call_count == (0 if when == "before_reference" else len(lane.prompts))
    assert (lane.source.source.checkpoint_digest, lane.source.source.source_lock_digest) == original_pins
    assert lane.quality.experiment().pipeline.model.to_spec() == original_model
    lane.forbidden.assert_not_called()
    if outcomes == "abort":
        assert diagnostic["code"] == "QUALITY_FAILED"
        assert "source binding" in diagnostic["message"]
        assert captured.out == "" and not lane.output.exists()
        return

    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking
    assert diagnostic["code"] == "QUALITY_EXECUTION_FAILED"
    report = read_report(lane.output.read_bytes())
    assert json.loads(captured.out)["data"]["report"] == report
    assert report["terminal_state"] == "execution_failed"
    assert report["reference_status"] == "unavailable"
    assert (report["model"]["checkpoint_digest"], report["model"]["source_lock_digest"]) == original_pins
    assert report["cohort"]["same_token_decode"]["teacher_forcing"] == "caller-frozen"
    assert [row["configuration_digest"] for row in report["candidates"]] == [
        lane.quality.experiment().configuration_digest(), lane.quality.seeded().configuration_digest()]
    for row in report["candidates"]:
        assert row["execution"] == "two-worker-request-end-verified"
        for section in (row, row["all_positions"], row["same_token_decode"]):
            assert section["coverage"]["accepted"] == section["planned_sample_count"]
            assert section["score_status"] == "reference_unavailable"
            assert section["sample_count"] == section["accepted_agreement"]["sample_count"] == 0
            for metric in ("top1_agreement", "top_k_recall", "max_abs_logit_error"):
                assert section[metric] is section["accepted_agreement"][metric] is None
    with pytest.raises(ValueError, match="unavailable"):
        require_complete_for_ranking(report)
