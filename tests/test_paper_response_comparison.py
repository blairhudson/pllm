"""Campaign contracts and ordinary-runner conformance; all execution is synthetic."""

import importlib
from importlib.abc import MetaPathFinder
import hashlib
import json
from pathlib import Path
import shlex
import signal
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def campaign(monkeypatch, request):
    class NoModelImports(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            # Real CLI/dashboard imports NumPy metadata; inert config tests still
            # prohibit it. Neither path permits source loaders or tensor models.
            if any(fullname == name or fullname.startswith(name + ".") for name in (
                "pllm.model_loader", "pllm.runtime.delegated_source",
                "pllm.runtime.fss_source", "pllm.runtime.replicated_source",
                "torch", "transformers", "huggingface_hub",
            )) or fullname == "numpy" and "cli_runtime" not in request.fixturenames:
                raise AssertionError(f"inert campaign attempted model/source import: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [NoModelImports(), *sys.meta_path])
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("examples.papers.accepted_response_comparison")


def captured_build():
    """Synthetic build metadata, not a real installed implementation attestation."""
    from pllm._benchmark_contract import digest
    from pllm.metrics import BuildRef

    attributes = {"synthetic_test_only": True}
    observed = {"attributes": attributes, "digest": digest(attributes)}
    return BuildRef(expected_digest=observed["digest"]), observed


def make_suite(campaign, **changes):
    build, _ = captured_build()
    return campaign.suite(**{"build": build, "memory_bytes": 8 << 30, **changes})


def test_only_complete_decoder_factories_enter_the_same_pinned_3x1_workload(campaign):
    from pllm import ExecutionBudget, Experiment, Pipeline
    from pllm.profiles import resolve_runtime_composition
    from examples.papers import fusefss, gkm

    recipes = campaign.recipes()
    budget = ExecutionBudget(1, 3, 1)
    assert recipes == {
        "gkm-full": gkm.experiment(budget=budget),
        "gkm-seeded": gkm.seeded(budget=budget),
        "fusefss-checked-tensor": fusefss.qwen_checked_tensor_minimal_smoke(),
    }
    assert tuple(recipes) == campaign.CASE_NAMES
    for experiment in recipes.values():
        assert isinstance(experiment, Experiment) and isinstance(experiment.pipeline, Pipeline)
        assert experiment.pipeline.model.to_spec() == {
            "source": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "local_files_only": True,
        }
        assert experiment.budget == budget
        assert experiment.pipeline.components["linear"].params["capacity"] == "decoder"
        options = resolve_runtime_composition(experiment.pipeline)
        assert options.client_runtime in {"compiled_gkm_v1", "compiled_fss_v1"}
        restored = Experiment.from_spec(json.loads(experiment.canonical_bytes()))
        assert restored.canonical_bytes() == experiment.canonical_bytes()
    fss = recipes["fusefss-checked-tensor"].pipeline.components["linear"].params
    assert fss["numeric_profile"] == "fusefss.dense-rope-checked-ranges-ring50-q12.v1"
    assert fss["material_delivery"] == "batch-v1" and fss["online_schedule"] == "tensor-v1"


def test_public_suite_round_trip_keeps_final_response_count_and_metric_contracts(campaign, monkeypatch):
    from pllm.metrics import BenchmarkSuite, BenchmarkTarget, BuildRef

    monkeypatch.setattr(BuildRef, "capture", Mock(side_effect=AssertionError("implicit build inspection")))
    selected = make_suite(campaign)
    assert isinstance(selected, BenchmarkSuite)
    restored = BenchmarkSuite.from_spec(json.loads(json.dumps(selected.to_spec(), allow_nan=False)))
    assert restored.to_spec() == selected.to_spec() and restored.digest == selected.digest
    assert selected.sampling.processes == 3
    assert selected.sampling.repetitions == 1 and selected.sampling.warmups == 0
    assert selected.sampling.order == "balanced" and selected.sampling.threads == 1
    assert selected.sampling.timeout_seconds == 1800
    for case in selected.cases:
        assert isinstance(case.target, BenchmarkTarget) and case.target.kind == "response"
        parameters = case.target.to_spec()["parameters"]
        assert parameters["backend"] == "native" and parameters["wan"] is None
        assert parameters["scenario"] == {
            "name": "request", "kind": "cold", "count": 1, "max_output_tokens": 1,
        }
        assert parameters["expected_input_tokens"] == 3
        assert parameters["expected_output_tokens"] == 1
        assert "state_audit" not in parameters  # No fabricated full-logit/KV FSS capability.
        assert parameters["model"]["config"] is None  # No generated shape or tiny substitute.
        metrics = {budget.metric for budget in case.budgets}
        assert {"request.0.full", "process.peak_rss", "client.peak_rss",
                "client.cold_cpu", "worker_a.cold_cpu", "worker_b.peak_rss",
                "network.covered_body_bytes"} <= metrics
        assert "response.decode_tps" not in metrics and "network.all_link_stream_bytes" not in metrics
        assert ("dealer.cold_cpu" in metrics) == (case.name == "fusefss-checked-tensor")
        assert all(budget.minimum == 0 and budget.maximum is None for budget in case.budgets)
    BuildRef.capture.assert_not_called()


@pytest.mark.parametrize("changes", [
    {"build": None}, {"memory_bytes": True}, {"memory_bytes": 16 << 30 | 1},
    {"processes": 0}, {"processes": True}, {"timeout_seconds": 0},
])
def test_campaign_rejects_unbound_build_and_invalid_existing_resource_bounds(campaign, changes):
    with pytest.raises((TypeError, ValueError)):
        make_suite(campaign, **changes)


def test_uncaptured_build_is_rejected_without_inspection(campaign):
    from pllm.metrics import BuildRef

    with pytest.raises(ValueError, match="caller-captured"):
        make_suite(campaign, build=BuildRef())


def test_cells_exclude_cross_numeric_privacy_rankings_and_unavailable_methods(campaign):
    declaration = json.loads(json.dumps(campaign.compatibility_cells(), allow_nan=False))
    cells = {cell["name"]: cell for cell in declaration["cells"]}
    assert tuple(cells) == campaign.CASE_NAMES
    assert {cell["method"] for cell in cells.values()} == {"Gupta-Katz-Miers", "FuseFSS"}
    assert all(cell["status"] == "planned_unmeasured" and not cell["publication_qualified"]
               for cell in cells.values())
    full, seeded, fss = [cells[name] for name in campaign.CASE_NAMES]
    assert full["numeric_profile"] == seeded["numeric_profile"] != fss["numeric_profile"]
    assert full["sharing"] != seeded["sharing"]
    assert fss["functionality"] == "no_accepted_trained_response"
    assert fss["acceptance"] == "checked_numeric_domain_then_final_ids"
    assert fss["state"] == "private_complete_prefix_recomputation"
    assert {role["id"] for role in full["role_graph"]["roles"]} == {"client", "worker_a", "worker_b"}
    assert {role["id"] for role in fss["role_graph"]["roles"]} == {"client", "dealer", "worker_a", "worker_b"}
    assert declaration["rankable_pairs"] == [] and len(declaration["pair_exclusions"]) == 3
    assert {tuple(pair["cases"]) for pair in declaration["pair_exclusions"]} == {
        ("gkm-full", "gkm-seeded"), ("gkm-full", "fusefss-checked-tensor"),
        ("gkm-seeded", "fusefss-checked-tensor"),
    }
    excluded = {cell["method"]: cell for cell in declaration["excluded_methods"]}
    assert excluded["PUMA"]["status"] == "blocked"
    assert excluded["SIGMA"]["status"] == "unsupported"
    assert len(excluded) == 8 and not set(excluded) & {cell["method"] for cell in cells.values()}
    assert declaration["cli"]["status"] == "supported"
    assert declaration["cli"]["report_schema"] == "pllm.benchmark_run.v1"


def test_inputs_are_fresh_identical_public_contexts_not_padding_or_tokenizer_claims(campaign):
    first, second = campaign.inputs(), campaign.inputs()
    assert first == {name: {"prompt": "hi"} for name in campaign.CASE_NAMES}
    first["gkm-full"]["prompt"] = "changed"
    assert second == campaign.inputs()


@pytest.fixture
def published_factory(campaign, monkeypatch, tmp_path):
    """Execute the documented zero-arg wrapper, not a replacement suite factory."""
    from pllm.metrics import BuildRef

    build, _ = captured_build()
    capture = Mock(return_value=build)
    monkeypatch.setattr(BuildRef, "capture", capture)
    path = tmp_path / "accepted_response_suite.py"
    path.write_text(campaign.__doc__.split("```python\n", 1)[1].split("```", 1)[0])
    return SimpleNamespace(path=path, target=f"{path}:campaign", capture=capture,
                           output=tmp_path / "synthetic-suite.json")


def published_arguments(campaign, published_factory):
    command = shlex.split(campaign.__doc__.split("```sh\n", 1)[1].split("```", 1)[0])
    assert command[:5] == ["python", "-m", "pllm", "benchmark", "run"]
    # Only temporary file destinations change; options/resource guards stay exact.
    command[command.index("--experiment") + 1] = published_factory.target
    command[command.index("--output") + 1] = str(published_factory.output)
    return command[3:]


def test_suite_resolver_is_explicit_trusted_opt_in_not_global_target_widening(campaign, published_factory):
    from pllm._cli.errors import ResolutionError
    from pllm._cli.targets import resolve_target
    from pllm.metrics import BenchmarkSuite

    options = dict(factory=True, no_input=True, trust_python=True, output_format="json")
    with pytest.raises(ResolutionError) as rejected:
        resolve_target(published_factory.target, **options)
    assert rejected.value.code == "TARGET_TYPE"
    selected = resolve_target(published_factory.target, allow_benchmark_suite=True, **options).configuration
    assert type(selected) is BenchmarkSuite and selected.to_spec() == make_suite(campaign).to_spec()
    with pytest.raises(ResolutionError) as untrusted:
        resolve_target(published_factory.target, allow_benchmark_suite=True,
                       **{**options, "trust_python": False})
    assert untrusted.value.code == "PYTHON_TRUST_REQUIRED"


def test_strict_suite_serialization_rejects_unknown_claim_fields(campaign):
    from pllm.metrics import BenchmarkSuite

    document = make_suite(campaign).to_spec()
    document["publication_qualified"] = True
    with pytest.raises(ValueError, match="invalid benchmark suite"):
        BenchmarkSuite.from_spec(document)


@pytest.fixture
def ordinary_runner(campaign, monkeypatch):
    """Mock only build observation and isolated execution; use the real runner."""
    from pllm.metrics import benchmark_runner
    from pllm._benchmark_contract import digest
    from pllm._benchmark_evidence import measurement

    _, observed = captured_build()
    monkeypatch.setattr(benchmark_runner, "inspect_build", Mock(return_value=observed))
    declarations = {cell["name"]: cell for cell in campaign.compatibility_cells()["cells"]}

    def synthetic(case, sampling, inputs, salt, directory):
        cell = declarations[case.name]
        # Explicitly synthetic fixture/output: no trained response, metric or
        # source acceptance is performed by this mock isolated execution.
        return {
            "status": "completed", "fixture": {"kind": "synthetic", "digest": digest("test-only"),
                                               "metadata": {"synthetic_test_only": True}},
            "oracle": {"kind": "synthetic_test_only", "outcome": "passed", "checked_values": 1},
            "admission": {"admitted": True, "required_bytes": 1, "limit_bytes": sampling.memory_bytes,
                          "host_headroom_bytes": sampling.memory_bytes, "scope": "test_only", "reason": None},
            "comparison": {
                "source_digest": digest(cell["source"]), "numeric_digest": digest(cell["numeric_profile"]),
                "privacy_digest": digest([cell["privacy_protocol"], cell["sharing"], cell["role_graph"]]),
                "state_digest": digest(cell["state"]), "workload_digest": digest(cell["workload"]),
                "output_digest": digest("synthetic-output-not-a-trained-result"),
                "measurement_digest": digest(cell["acceptance"]),
            },
            "samples": [{"index": 0, "kind": "measurement", "status": "completed",
                         "duration_seconds": 1, "artifact_digest": None}],
            "metrics": [measurement("request.0.full", 1)],
            "limitations": ["Synthetic unit-test execution only; never trained campaign evidence."],
        }

    execute = Mock(side_effect=synthetic)
    monkeypatch.setattr(benchmark_runner, "_execute", execute)
    return benchmark_runner, execute, synthetic


def test_one_ordinary_invocation_keeps_one_salt_and_rejects_incompatible_comparisons(campaign, ordinary_runner, tmp_path):
    from pllm.metrics import BenchmarkRun, benchmark_suite

    runner, execute, _ = ordinary_runner
    selected = make_suite(campaign, processes=2)
    output = tmp_path / "synthetic-only.json"
    run = benchmark_suite(selected, inputs=campaign.inputs(), output=output)
    assert BenchmarkRun.load(output).digest == run.digest
    rows = [row.to_dict() for row in run.results]
    assert len(rows) == 6 and run.to_dict()["complete"]
    assert len({row["execution"]["cohort"] for row in rows}) == 1
    assert len({call.args[3] for call in execute.call_args_list}) == 1
    assert [call.args[0].name for call in execute.call_args_list] == [
        "gkm-full", "gkm-seeded", "fusefss-checked-tensor",
        "fusefss-checked-tensor", "gkm-seeded", "gkm-full",
    ]
    assert all(call.args[2] == {"prompt": "hi"} for call in execute.call_args_list)
    assert runner.inspect_build.call_count == 2  # Initial and unchanged-after-cohort observations.
    assert all(row["execution"]["fixture"]["kind"] == "synthetic" for row in rows)
    assert any(budget["outcome"] == "unknown" for row in rows for budget in row["execution"]["budgets"])
    for pair in campaign.compatibility_cells()["pair_exclusions"]:
        assert run.compare(*pair["cases"], "request.0.full") == {
            "status": "unranked", "reason": "cohort_contract_differs",
        }


@pytest.mark.parametrize("status", ["blocked", "unsupported", "failed", "timeout"])
def test_ordinary_cohort_retains_every_unaccepted_method_attempt(campaign, ordinary_runner, status):
    from pllm.metrics import BenchmarkRun, benchmark_suite

    _, execute, synthetic = ordinary_runner

    def one_unaccepted(case, *args):
        if case.name == "fusefss-checked-tensor":
            return {"status": status, "failure_code": "synthetic_test_only"}
        return synthetic(case, *args)

    execute.side_effect = one_unaccepted
    run = benchmark_suite(make_suite(campaign, processes=2), inputs=campaign.inputs())
    rows = [row.to_dict() for row in run.results]
    failed = [row for row in rows if row["execution"]["case"]["name"] == "fusefss-checked-tensor"]
    assert run.to_dict()["complete"] and len(rows) == 6 and len(failed) == 2
    assert all(row["status"] == status and row["execution"]["failure_code"] == "synthetic_test_only"
               for row in failed)
    assert run.compare("gkm-full", "fusefss-checked-tensor", "request.0.full")["status"] == "unranked"
    document = run.to_dict()
    document["results"].pop()
    with pytest.raises(ValueError, match="missing trials"):
        BenchmarkRun(document)


@pytest.mark.parametrize("change", [
    lambda report: report["runs"][0]["tokens"].update(input_tokens=2),
    lambda report: report["runs"][0]["tokens"].update(output_tokens=0),
    lambda report: report["runs"][0]["tokens"].update(authoritative=False),
])
def test_configured_ordinary_count_gate_never_accepts_estimates_or_different_work(campaign, change):
    from pllm.runtime.benchmark_response_target import _expected_token_count_checks

    parameters = make_suite(campaign).cases[0].target.to_spec()["parameters"]
    report = {"checks": {"passed": True}, "runs": [{"status": "completed", "tokens": {
        "authoritative": True, "input_tokens": 3, "output_tokens": 1,
    }}]}
    change(report)
    before = json.loads(json.dumps(report["runs"]))
    assert _expected_token_count_checks(parameters, report) is False
    assert report["checks"]["passed"] is False and report["runs"] == before


@pytest.fixture
def cli_runtime(campaign, monkeypatch):
    """Real dispatch, response/count/metric adapters and suite engine; fake I/O only.

    Source, tensor execution, host admission and installed-build inspection are
    doubled. No native resolution, role startup, checkpoint or measured cost.
    """
    from pllm import Experiment
    from pllm._benchmark_contract import TargetUnavailable, digest
    from pllm._benchmark_worker import Measurement
    from pllm.metrics import benchmark_runner as engine
    from pllm.runtime import benchmark_cli, benchmark_memory, benchmark_response_target

    forbidden = Mock(side_effect=AssertionError("unexpected model/native admission or execution"))
    monkeypatch.setattr(Experiment, "resolve", forbidden)
    loader = ModuleType("pllm.model_loader")
    loader.expected_model_id = lambda model: model.source
    loader.resolve_model = forbidden
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    _, observed = captured_build()
    monkeypatch.setattr(engine, "inspect_build", Mock(return_value=observed))
    import pllm.metrics
    memory = Mock(return_value={"admitted": True, "synthetic_test_only": True})
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", memory)
    monkeypatch.setattr(benchmark_memory, "execution_memory_admission", memory)
    cells = campaign.compatibility_cells()["cells"]
    by_profile = {campaign.recipes()[cell["name"]].pipeline.digest(): cell for cell in cells}

    def declaration(experiment):
        return by_profile[experiment.pipeline.digest()]

    boundary = importlib.import_module("pllm._search_benchmark")
    monkeypatch.setattr(boundary, "_boundary", lambda experiment: {
        "numeric": {"synthetic_test_only": True},
        "privacy": {key: declaration(experiment)[key] for key in ("privacy_protocol", "sharing", "role_graph")},
    })
    calls, workers = [], []
    control = {"input_tokens": 3, "output_tokens": 1, "authoritative": True,
               "checks": True, "fail_fss": False}

    def response(**options):
        cell = declaration(options["experiment"])
        calls.append(options)
        fss = cell["name"] == "fusefss-checked-tensor"
        if fss and control["fail_fss"]:
            raise TargetUnavailable("synthetic_no_accepted_response", status="failed")
        from pllm.runtime.benchmark_delegated import output_contract
        numeric = {"numeric_profile": cell["numeric_profile"], "local_numeric_profile": cell["local_numeric_profile"]}
        return {"schema_version": "pllm.loopback_benchmark.v1",
            "checks": {"passed": control["checks"]},
            "configuration": {"source_lock_digest": digest("synthetic-source-not-checkpoint"),
                "provider_backend": "native", "sampling": {"temperature": 0},
                "roles": [role["id"] for role in cell["role_graph"]["roles"]],
                "prompt_digest": hashlib.sha256(b"pllm.benchmark.prompt.v1\0"
                    + options["_cohort_salt"] + options["prompt"].encode()).hexdigest()},
            "runtime_contract": {"runtime_kind": "fss" if fss else "gkm", "numeric": numeric,
                "checkpoint_digest": digest("synthetic-checkpoint-not-weights"),
                "output_delivery": output_contract(options["experiment"])[0]},
            "topology_accounting": {},
            "runs": [{"status": "completed", "model_fingerprint": digest("synthetic-body"),
                "tokens": {key: control[key] for key in ("input_tokens", "output_tokens", "authoritative")},
                "durations": {"full_seconds": 1}, "privacy": {}, "processes": {},
                "generation": {"output_text_digest": digest("synthetic-output-not-trained")}}]}

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", response)
    supervise = engine._execute

    def execute(case, sampling, inputs, salt, directory):
        workers.append((case, sampling, dict(inputs), salt, directory))
        m = Measurement({"sampling": sampling.to_spec(), "salt": salt.hex(), "inputs": inputs,
                         "directory": str(directory)})
        m.admit = lambda required: m.admission.update(admitted=True, required_bytes=required,
                                                     host_headroom_bytes=sampling.memory_bytes)
        status, code = "completed", None
        try:
            benchmark_response_target.execute_response(case.target.to_spec()["parameters"], m)
        except TargetUnavailable as error:
            status, code = error.status, error.code
        m.finalize_workload(case.target, completed=status == "completed")
        return {"status": status, "failure_code": code, "metrics": m.metrics, "samples": m.samples,
            "fixture": m.fixture, "oracle": m.oracle, "admission": m.admission, "comparison": m.comparison,
            "details": m.details, "limitations": [*m.limitations,
                "Synthetic unit-test boundaries; no trained response or measured cost evidence."],
            "environment": {"scope": "synthetic_test_only"}}

    monkeypatch.setattr(engine, "_execute", execute)
    return SimpleNamespace(engine=engine, supervise=supervise, calls=calls, workers=workers,
                           control=control, forbidden=forbidden)


def test_published_command_real_parser_factory_dispatch_counts_salt_and_scope(
        campaign, published_factory, cli_runtime, capsys):
    from pllm._cli.app import build_parser
    from pllm.cli import main
    from pllm.metrics import BenchmarkRun

    argv = published_arguments(campaign, published_factory)
    parsed = build_parser().parse_args(argv)
    assert parsed.factory and parsed.trust_python and parsed.prompt == "hi"
    assert parsed.max_output_tokens == 1 and parsed.timeout == 1800 and parsed.backend == "native"
    main(argv)
    report = json.loads(capsys.readouterr().out)["data"]["report"]
    run = BenchmarkRun.load(published_factory.output)
    assert report == run.to_dict() and report["schema"] == "pllm.benchmark_run.v1"
    assert report["complete"] and len(report["results"]) == 9
    assert published_factory.capture.call_count == 1
    assert len(cli_runtime.calls) == len(cli_runtime.workers) == 9
    assert len({worker[3] for worker in cli_runtime.workers}) == 1
    assert report["cohort"] == hashlib.sha256(cli_runtime.workers[0][3]).hexdigest()
    assert all(worker[2] == {"prompt": "hi"} and not worker[4].exists() for worker in cli_runtime.workers)
    for row in report["results"]:
        assert row["status"] == "completed"
        e = row["execution"]
        parameters = e["case"]["target"]["parameters"]
        assert (parameters["expected_input_tokens"], parameters["expected_output_tokens"]) == (3, 1)
        assert e["details"]["report"]["checks"] == {
            "passed": True, "expected_input_tokens": True, "expected_output_tokens": True}
        assert any(budget["outcome"] == "unknown" for budget in e["budgets"])
        metrics = {metric["id"]: metric for metric in row["metrics"]}
        assert metrics["request.0.full"]["unit"] == "seconds"
        assert metrics["request.0.full"]["measurement_scope"] == "ordinary_response_lifecycle"
        assert metrics["request.0.full"]["window"] == "request_setup_through_completion"
        assert metrics["client.peak_rss"]["value"] is None
        assert metrics["client.peak_rss"]["measurement_scope"] == "process_lifetime_high_water"
        assert metrics["request.0.decode_tps"]["value"] is None
        assert any("Synthetic unit-test" in text for text in row["limitations"])
    for pair in campaign.compatibility_cells()["pair_exclusions"]:
        assert run.compare(*pair["cases"], "request.0.full") == {
            "status": "unranked", "reason": "cohort_contract_differs"}
    cli_runtime.forbidden.assert_not_called()


@pytest.mark.parametrize("extra", [
    ["--memory-budget-mib", "64"], ["--max-output-tokens", "2"], ["--timeout", "1799"],
])
def test_cli_cannot_override_the_published_wrapper_resource_or_count_contract(
        campaign, published_factory, cli_runtime, capsys, extra):
    from pllm.cli import main

    with pytest.raises(SystemExit) as stopped:
        main([*published_arguments(campaign, published_factory), *extra])
    assert stopped.value.code == 3
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "BENCHMARK_SUITE_OVERRIDE"
    assert not cli_runtime.calls and not cli_runtime.workers and not published_factory.output.exists()


@pytest.mark.parametrize("changes,code", [
    ({"input_tokens": 2}, "response_expected_token_counts_failed"),
    ({"output_tokens": 0}, "response_expected_token_counts_failed"),
    ({"authoritative": False}, "response_checks_failed"),
    ({"checks": False}, "response_checks_failed"),
    ({"fail_fss": True}, "synthetic_no_accepted_response"),
])
def test_published_cli_keeps_failed_counts_or_unaccepted_fss_unranked(
        campaign, published_factory, cli_runtime, capsys, changes, code):
    from pllm.cli import main
    from pllm.metrics import BenchmarkRun

    cli_runtime.control.update(changes)
    main(published_arguments(campaign, published_factory))
    report = json.loads(capsys.readouterr().out)["data"]["report"]
    run = BenchmarkRun.load(published_factory.output)
    assert run.to_dict() == report and report["complete"] and len(report["results"]) == 9
    failed = [row for row in report["results"] if row["status"] == "failed"]
    assert len(failed) == (3 if changes.get("fail_fss") else 9)
    assert all(row["execution"]["failure_code"] == code for row in failed)
    if code == "response_expected_token_counts_failed":
        for row in failed:
            actual = row["execution"]["details"]["report"]["runs"][0]["tokens"]
            assert actual["input_tokens"] == cli_runtime.control["input_tokens"]
            assert actual["output_tokens"] == cli_runtime.control["output_tokens"]
            assert row["execution"]["oracle"]["outcome"] == "failed"
    assert run.compare("gkm-full", "fusefss-checked-tensor", "request.0.full")["status"] == "unranked"


def test_published_cli_cancellation_drains_real_worker_supervisor_and_checkpoints(
        campaign, published_factory, cli_runtime, monkeypatch, capsys):
    from pllm._benchmark_worker import PROTOCOL
    from pllm.cli import main
    from pllm.metrics import BenchmarkRun
    from pllm.runtime import benchmark_memory

    engine, events, trackers = cli_runtime.engine, [], []
    monkeypatch.setattr(engine, "_execute", cli_runtime.supervise)

    class Guard:
        def __init__(self, *args):
            pass

        def start(self):
            events.append("guard-start")

        def close(self):
            events.append("guard-close")

    class Process:
        pid = 2147483647

        def communicate(self, body, timeout):
            request = json.loads(body)
            assert request["protocol"] == PROTOCOL and request["inputs"] == {"prompt": "hi"}
            assert request["target"]["parameters"]["expected_input_tokens"] == 3
            assert timeout == 1800
            events.append("communicate")
            raise KeyboardInterrupt

        def poll(self):
            return None

        def wait(self, timeout=None):
            events.append("wait")

    def popen(command, **options):
        assert command[1:] == ["-I", "-m", "pllm._benchmark_worker"]
        assert options["start_new_session"] is True
        assert "hi" not in command
        return Process()

    real_thread = engine.threading.Thread

    def track_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        trackers.append(thread)
        return thread

    monkeypatch.setattr(engine.threading, "Thread", track_thread)
    monkeypatch.setattr(benchmark_memory, "MemoryWatchdog", Guard)
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: SimpleNamespace(
        reserve=0, swap_used=0, available=16 << 30))
    # Optional usage-owner composition is outside this patch. If installed in a
    # shared working tree, double its OS boundary too; do not require that WIP.
    try:
        from pllm.runtime import client_process_usage
    except ImportError:
        client_process_usage = None
    if client_process_usage is not None:
        monkeypatch.setattr(client_process_usage, "client_popen_factory", lambda factory: factory)
        monkeypatch.setattr(client_process_usage, "finalize_client_payload", Mock())
    monkeypatch.setattr(engine.subprocess, "Popen", popen)
    monkeypatch.setattr(engine.os, "killpg", lambda pid, sig: events.append(sig))
    with pytest.raises(SystemExit) as stopped:
        main(published_arguments(campaign, published_factory))
    assert stopped.value.code == 130
    report = BenchmarkRun.load(published_factory.output).to_dict()
    assert report["complete"] is False and len(report["results"]) == 1
    assert report["results"][0]["status"] == "cancelled"
    assert events == ["guard-start", "communicate", "guard-close", signal.SIGTERM, "wait", signal.SIGKILL, "wait"]
    assert len(trackers) == 1 and not trackers[0].is_alive()
    assert capsys.readouterr().out == ""
    assert not cli_runtime.calls  # No source/model/response executor was reached.
