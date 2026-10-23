"""Metadata-only experiment checks; no checkpoint, native import or serving.

Run without repository serving fixtures/plugins:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest -q \
    tests/test_paper_low_memory_experiment.py
"""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from pllm import Experiment
from pllm._benchmark_evidence import assess_budgets, measurement
from pllm.metrics import BenchmarkSuite, BenchmarkTarget, BuildRef
from pllm.preparation import PreparedInventory
from pllm.profiles import resolve_runtime_composition
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse

from examples.papers import low_memory_qwen3 as experiment


BUILD = BuildRef(expected_digest="a" * 64)  # Synthetic serialization fixture, not a captured build.


def suite(**options):
    return experiment.matched_suite(build=BUILD, memory_bytes=12 << 30, **options)


def test_metadata_import_and_factories_do_not_load_models_or_native():
    root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys
class RejectHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'numpy' or fullname.startswith(('pllm._native', 'pllm.runtime',
                'pllm.model_loader', 'torch', 'transformers', 'tokenizers'))):
            raise AssertionError('heavy import: ' + fullname)
sys.meta_path.insert(0, RejectHeavy())
from examples.papers.low_memory_qwen3 import recipes, matched_suite
from pllm.metrics import BuildRef
recipes()
matched_suite(build=BuildRef(expected_digest='a' * 64), memory_bytes=12 << 30)
"""
    import os

    env = dict(os.environ, PYTHONPATH=str(root / "python") + os.pathsep + str(root))
    subprocess.run([sys.executable, "-c", code], cwd=root, env=env, check=True,
                   capture_output=True, text=True, timeout=30)


def test_suite_and_experiments_round_trip_and_lock_known_offline_source():
    value = suite()
    assert BenchmarkSuite.from_spec(json.loads(json.dumps(value.to_spec()))).to_spec() == value.to_spec()
    assert value.sampling.processes == 3
    assert value.sampling.order == "balanced"
    assert value.sampling.repetitions == 1 and value.sampling.warmups == 0
    assert value.sampling.memory_bytes == 12 << 30
    assert [case.name for case in value.cases] == list(experiment.NAMES)
    for case in value.cases:
        params = case.target.to_spec()["parameters"]
        spec = params["model"]["experiment"]
        restored = Experiment.from_spec(spec)
        assert restored.to_spec() == spec
        assert restored.pipeline.model.to_spec() == {
            "source": "Qwen/Qwen3-4B", "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
            "local_files_only": True,
        }
        assert restored.budget == experiment.BUDGET
        assert params["state_audit"] is True and params["backend"] == "native"
        assert params["expected_input_tokens"] == 64 and params["expected_output_tokens"] == 8
        assert params["scenario"] == {"name": "request", "kind": "cold", "count": 1,
                                       "max_output_tokens": 8}
        assert case.build == BUILD


def test_candidate_changes_only_existing_delivery_not_reference_body_or_inventory():
    control, candidate = experiment.recipes().values()
    left, right = control.pipeline.components, candidate.pipeline.components
    assert set(right) - set(left) == {"delivery"}
    assert all(component == right[name] for name, component in left.items())
    assert right["quantization"].params == {
        "weight_bits": 8, "activation_bits": 8, "causal_reduction": "prefix_f32"}
    inventory = right["inventory"].get_params()
    assert inventory["policy"] == "prewarm" and inventory["allocation"] == "uniform"
    assert inventory["rows"] == 64 + 8 - 1 == experiment.PREPARED_ROWS
    assert inventory["refill"] == "idle"
    assert inventory["decode_epoch_tokens"] is None and inventory["decode_epoch_schedule"] is None
    assert right["delivery"].get_params()["storage"] == "paged"
    assert right["delivery"].get_params()["encoding"] == "artifacts"
    assert not {"placement", "cache", "state", "boundary", "topology"} & set(right)
    assert control.pipeline.model == candidate.pipeline.model


def test_private_prompt_is_shared_but_not_serialized_as_public_suite_metadata():
    prompt = "private fixture; exact token count has not been established"
    inputs = experiment.private_inputs(prompt)
    assert inputs == {name: {"prompt": prompt} for name in experiment.NAMES}
    assert inputs["maintained"] is not inputs["paged-prepared"]
    assert prompt not in json.dumps(suite().to_spec())


@pytest.mark.parametrize("prompt", ["", "  ", None, [1] * 64, "é" * 8193])
def test_invalid_private_context_is_rejected_not_padded_or_tokenized(prompt):
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        experiment.private_inputs(prompt)


def test_build_and_native_sampling_limits_are_not_weakened():
    with pytest.raises(ValueError, match="caller-captured"):
        experiment.matched_suite(build=BuildRef(), memory_bytes=12 << 30)
    with pytest.raises(TypeError):
        experiment.matched_suite(build=BUILD)  # No guessed host/client memory default.
    with pytest.raises(ValueError, match="memory_bytes"):
        experiment.matched_suite(build=BUILD, memory_bytes=(16 << 30) + 1)
    with pytest.raises(ValueError, match="processes"):
        suite(processes=33)


def test_rss_windows_and_missing_metric_assessments_are_not_estimates_or_tier_passes():
    case = suite().cases[0]
    assert experiment.MEASUREMENTS["process.peak_rss"] == (
        "process_birth_through_target", "process_lifetime_high_water")
    assert experiment.MEASUREMENTS["client.peak_rss"][0] == "process_birth_through_scenario"
    assert experiment.MEASUREMENTS["preparation.peak_rss"][0] == "request_windows"
    assert experiment.MEASUREMENTS["network.all_link_stream_bytes"][1] == "tcp_stream_payload"
    assert all(b.maximum is None and b.minimum == 0 for b in case.budgets)
    assert all(row["outcome"] == "unknown" for row in assess_budgets(case, []))
    unavailable = measurement("process.peak_rss", None, kind="memory",
                              window="process_birth_through_target", scope="process_lifetime_high_water")
    assert unavailable["origin"] == "not_available"
    assert assess_budgets(case, [unavailable])[0]["outcome"] == "unknown"


def test_boundary_alternative_is_explicitly_blocked_not_a_working_recipe():
    blocked = next(cell for cell in experiment.blocked_cells() if cell["name"] == "coupled-boundary")
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "source_boundary_contract_not_qualified"
    assert "tied bias-free W8A8" in blocked["required_contract"]
    assert "four-provider" in blocked["required_contract"]
    assert blocked["name"] not in [case.name for case in suite().cases]


def test_exact_count_contract_changes_public_identity_not_private_payload():
    original = suite()
    legacy = replace(original, cases=tuple(replace(case, target=case.target.with_params(
        expected_input_tokens=None, expected_output_tokens=None)) for case in original.cases))
    assert original.digest != legacy.digest
    assert suite().digest == original.digest
    restored = BenchmarkSuite.from_spec(json.loads(json.dumps(original.to_spec())))
    assert restored.digest == original.digest
    for case, old in zip(restored.cases, legacy.cases, strict=True):
        assert case.target.digest != old.target.digest
        assert case.target.with_params(expected_input_tokens=64, expected_output_tokens=8).digest == case.target.digest
        assert BenchmarkTarget.from_spec(case.target.to_spec()).digest == case.target.digest
        assert case.build == old.build and case.budgets == old.budgets


def test_no_forced_length_quality_or_observed_memory_claim_in_factory():
    for case in suite().cases:
        parameters = case.target.to_spec()["parameters"]
        assert parameters["scenario"]["max_output_tokens"] == parameters["expected_output_tokens"] == 8
        assert parameters["expected_input_tokens"] == 64
        assert parameters["state_audit"] is True
        assert not {"force_length", "min_new_tokens", "suppress_eos", "quality_oracle"} & parameters.keys()
    assert "0.83" not in json.dumps(suite().to_spec())


def test_existing_runtime_composition_recognizes_prepared_paged_contract_without_native():
    control, candidate = experiment.recipes().values()
    experiment.validate_pair(control, candidate)
    for value in (control, candidate):
        runtime = resolve_runtime_composition(value.pipeline)
        assert runtime is not None
        assert runtime.requires_preparation and runtime.privacy_mode == "public"
        assert runtime.inventory_policy == "prewarm" and runtime.prepared_inventory_rows == 71
        assert runtime.background_inventory_refill and runtime.inventory_refill == "idle"
        assert runtime.preparation_decode_epoch_tokens is None and runtime.preparation_decode_epoch_schedule is None
        assert runtime.weight_bits == runtime.activation_bits == 8
        assert not runtime.private_boundary and not runtime.decode and not runtime.inflight


@pytest.mark.parametrize("inventory", [
    PreparedInventory(rows=1, allocation="demand", refill="on-demand"),
    PreparedInventory("request-sized", rows=71, refill="on-demand"),
    PreparedInventory("prewarm", rows=64, refill="idle"),
    PreparedInventory("prewarm", rows=71, refill="on-demand"),
    PreparedInventory(rows=1, allocation="demand", refill="on-demand", decode_epoch_tokens=1),
])
def test_candidate_rejects_foreground_or_insufficient_inventory(inventory, monkeypatch):
    control, candidate = experiment.recipes().values()
    candidate = candidate.with_params(pipeline__inventory=inventory)
    with pytest.raises(ValueError, match="committed prewarm inventory"):
        experiment.validate_pair(control, candidate)
    monkeypatch.setattr(experiment, "recipes", lambda: dict(zip(experiment.NAMES, (control, candidate), strict=True)))
    with pytest.raises(ValueError, match="committed prewarm inventory"):
        suite()


@pytest.mark.parametrize("changes,reason", [
    ({"pipeline__model__revision": "b" * 40}, "pinned offline source"),
    ({"pipeline__model__local_files_only": False}, "pinned offline source"),
    ({"pipeline__quantization": SymmetricPerRow(weight_bits=4, activation_bits=8)}, "numeric/body/state"),
    ({"pipeline__quantization": SymmetricPerRow(weight_bits=8, activation_bits=4)}, "numeric/body/state"),
    ({"pipeline__cache": ClientPrefixReuse(max_bytes=1 << 20, fixed_input_tokens=64)}, "numeric/body/state"),
    ({"budget__max_input_tokens": 65}, "64/8 execution budget"),
])
def test_candidate_rejects_different_source_numerics_state_and_geometry(changes, reason):
    control, candidate = experiment.recipes().values()
    with pytest.raises(ValueError, match=reason):
        experiment.validate_pair(control, candidate.with_params(**changes))


def test_foreground_demand_is_a_blocked_cell_not_campaign_arm():
    blocked = next(cell for cell in experiment.blocked_cells() if cell["name"] == "paged-demand")
    assert blocked["status"] == "blocked"
    assert blocked["reason"] == "foreground_missing_inventory_preparation_not_eligible"
    assert "fail-fast exhaustion" in blocked["required_contract"]
    assert blocked["name"] not in [case.name for case in suite().cases]


@pytest.fixture
def mock_executor(monkeypatch, tmp_path):
    """Real ordinary executor, synthetic transport/state records; no model oracle."""
    path = Path(__file__).resolve().parents[1] / "python/pllm/runtime/benchmark_response_target.py"
    spec = importlib.util.spec_from_file_location("pllm.runtime.benchmark_response_target", path)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)

    def install(name, **values):
        module = ModuleType(name)
        module.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, module)

    if "pllm.runtime" not in sys.modules:
        install("pllm.runtime", __path__=[str(path.parent)])
    calls = []
    observed = {
        "schema_version": "pllm.loopback_benchmark.v1", "checks": {"passed": True},
        "configuration": {"source_lock_digest": "s" * 64, "provider_backend": "native",
                          "sampling": {"temperature": 0}}, "topology_accounting": {},
        "runs": [{"status": "completed", "model_fingerprint": "b" * 64,
                  "tokens": {"authoritative": True, "input_tokens": 64, "output_tokens": 8},
                  "generation": {"output_text_digest": "d" * 64}}],
    }

    def transport(**options):
        calls.append(options)
        return observed

    install("pllm.model_loader", expected_model_id=lambda _: "model")
    install("pllm.runtime.benchmark_cli", run_loopback_benchmark=transport)
    install("pllm.runtime.benchmark_context", validate_context=lambda text: text)
    install("pllm.runtime.benchmark_delegated", is_buffered=lambda _: False,
            is_replicated=lambda _: False, is_fss=lambda _: False, admit_options=lambda *a, **kw: None,
            TPS_UNAVAILABLE="mock buffered runtime")
    install("pllm.runtime.benchmark_response_audit", GKM_AUDIT_SCHEMA="unused",
            state_audit_admission=lambda value, *a, **kw: value,
            response_state_audit=lambda *a, **kw: nullcontext([
                {"failed": False, "selections": [{"checked_values": 2}] * 8}]))
    install("pllm._search_benchmark", _boundary=lambda _: {"numeric": {"bits": 8}, "privacy": {}})
    install("pllm.runtime.topology_accounting", ACCOUNTING_SCHEMA="accounting",
            accounting_coverage_identity=lambda _: "accounting")
    import pllm.metrics
    monkeypatch.setattr(pllm.metrics, "benchmark_memory", lambda *a, **kw: {"admitted": True})
    executor.actual_response_metrics = executor.response_metrics
    monkeypatch.setattr(executor, "response_metrics", lambda *a: None)
    m = SimpleNamespace(
        sampling=SimpleNamespace(repetitions=1, warmups=0, memory_bytes=12 << 30, timeout_seconds=60),
        inputs={"prompt": "private fixture"}, directory=tmp_path, salt=b"x" * 32,
        details={}, comparison={}, limitations=[], admit=lambda _: None, run=lambda action: [action(0)])
    return executor, m, observed, calls


@pytest.mark.parametrize("arm", experiment.NAMES)
@pytest.mark.parametrize("inputs,outputs,accepted", [(64, 8, True), (63, 8, False), (64, 7, False)])
def test_factory_targets_execute_ordinary_exact_count_gate_with_mock_transport(mock_executor, arm, inputs, outputs, accepted):
    from pllm._benchmark_contract import TargetUnavailable

    executor, m, observed, calls = mock_executor
    observed["runs"][0]["tokens"].update(input_tokens=inputs, output_tokens=outputs)
    before = deepcopy(observed["runs"])
    target = next(case.target for case in suite().cases if case.name == arm)
    if accepted:
        executor.execute_response(target.to_spec()["parameters"], m)
        assert m.oracle["outcome"] == "passed"
    else:
        with pytest.raises(TargetUnavailable) as error:
            executor.execute_response(target.to_spec()["parameters"], m)
        assert error.value.code == "response_expected_token_counts_failed"
        assert error.value.status == "failed" and m.oracle["outcome"] == "failed"
    assert observed["checks"]["passed"] is accepted
    assert observed["runs"] == before
    assert m.details["response_state_audit"]["bindings"]
    assert m.comparison["output_digest"]
    assert calls[0]["max_output_tokens"] == 8 and calls[0]["temperature"] == 0
    assert "expected_input_tokens" not in calls[0] and "expected_output_tokens" not in calls[0]


@pytest.mark.parametrize("change", [
    lambda report: report["runs"][0]["tokens"].update(authoritative=False),
    lambda report: report["runs"][0].update(status="failed"),
    lambda report: report["checks"].update(passed=False, no_online_preparation_requests=False),
])
def test_factory_count_gate_does_not_override_original_unknown_failed_or_privacy_rejection(mock_executor, change):
    from pllm._benchmark_contract import TargetUnavailable

    executor, m, observed, _ = mock_executor
    change(observed)
    target = suite().cases[1].target
    with pytest.raises(TargetUnavailable) as error:
        executor.execute_response(target.to_spec()["parameters"], m)
    assert error.value.code == "response_checks_failed"
    assert "expected_output_tokens" not in observed["checks"]


def test_actual_ordinary_response_metrics_keep_windows_and_missing_values_unknown(mock_executor, monkeypatch):
    executor, m, observed, _ = mock_executor
    communication = ModuleType("pllm.runtime.benchmark_communication")
    communication.benchmark_traffic_decomposition = lambda _: {
        "schema": "mock-decomposition",
        "summary": {"base_all_link_mb_per_request": None, "incremental_all_link_mb_per_generated_output": None,
                    "measured_requests": 1, "generated_output_tokens": 8},
        "covered_lifecycle": {"unattributed_all_link_body_bytes": None},
    }
    monkeypatch.setitem(sys.modules, communication.__name__, communication)
    observed["runs"][0]["durations"] = {"full_seconds": 2, "ttft_seconds": 1, "tokens_per_second": 7}
    observed["client_process_memory"] = {"after": {"lifetime_peak_rss_bytes": 1234}}
    observed["client_owned_memory"] = {
        "schema": "pllm.client_owned_memory.v1", "sampled_peak_rss_sum_bytes": None,
        "sampled_peak_rss_lower_bound_bytes": None,
    }
    captured = {}
    m.metric = lambda name, value, **options: captured.update({name: (value, options)})
    executor.actual_response_metrics(observed, m)
    for name in ("client.peak_rss", "client.owned_sampled_peak_rss_sum", "client.owned_peak_rss_lower_bound",
                 "preparation.peak_rss", "network.all_link_stream_bytes", "request.0.full"):
        _, options = captured[name]
        assert (options["window"], options["scope"]) == experiment.MEASUREMENTS[name]
    assert captured["client.peak_rss"][0] == 1234  # Synthetic metric wiring, not a measured peak.
    for name in ("client.owned_sampled_peak_rss_sum", "client.owned_peak_rss_lower_bound",
                 "preparation.peak_rss", "network.all_link_stream_bytes"):
        value, options = captured[name]
        assert value is None
        record = measurement(name, value, **options)
        assert record["origin"] == "not_available"
        assert next(row for row in assess_budgets(suite().cases[0], [record])
                     if row["metric"] == name)["outcome"] == "unknown"


CLI_SETTINGS = {
    "PLLM_LOW_MEMORY_BUILD_DIGEST": "a" * 64,
    "PLLM_LOW_MEMORY_BYTES": str(12 << 30),
    "PLLM_LOW_MEMORY_PROCESSES": "2",
    "PLLM_LOW_MEMORY_TIMEOUT_SECONDS": "120",
}


def test_cli_wrapper_round_trips_all_explicit_choices_without_capture_or_execution(monkeypatch):
    for key, value in CLI_SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(BuildRef, "capture", lambda *a, **kw: pytest.fail("implicit build capture"))
    configured = experiment.cli_suite()
    expected = experiment.matched_suite(build=BUILD, memory_bytes=12 << 30, processes=2, timeout_seconds=120)
    assert configured.to_spec() == expected.to_spec()
    restored = BenchmarkSuite.from_spec(json.loads(json.dumps(configured.to_spec())))
    assert restored.digest == configured.digest
    for case in restored.cases:
        assert case.target.to_spec()["parameters"]["expected_input_tokens"] == 64
        assert case.target.to_spec()["parameters"]["expected_output_tokens"] == 8
        assert case.target.to_spec()["parameters"]["state_audit"] is True
    monkeypatch.setenv("PLLM_LOW_MEMORY_BYTES", str(11 << 30))
    assert experiment.cli_suite().digest != configured.digest


@pytest.mark.parametrize("missing", CLI_SETTINGS)
def test_cli_wrapper_requires_every_build_resource_argument(monkeypatch, missing):
    for key, value in CLI_SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv(missing)
    with pytest.raises(ValueError, match=missing):
        experiment.cli_suite()


@pytest.mark.parametrize("key,value", [
    ("PLLM_LOW_MEMORY_BUILD_DIGEST", "not-a-build"),
    ("PLLM_LOW_MEMORY_BYTES", str((16 << 30) + 1)),
    ("PLLM_LOW_MEMORY_BYTES", "0"),
    ("PLLM_LOW_MEMORY_PROCESSES", "33"),
    ("PLLM_LOW_MEMORY_PROCESSES", "0"),
    ("PLLM_LOW_MEMORY_TIMEOUT_SECONDS", "3601"),
    ("PLLM_LOW_MEMORY_TIMEOUT_SECONDS", "nan"),
])
def test_cli_wrapper_preserves_ordinary_public_caps(monkeypatch, key, value):
    for name, setting in CLI_SETTINGS.items():
        monkeypatch.setenv(name, setting)
    monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        experiment.cli_suite()


def test_whole_client_opt_in_selections_never_alias_individual_process_metrics():
    from pllm.metrics import MetricSpec

    specs = experiment.CLIENT_MEMORY_METRICS
    assert all(type(spec) is MetricSpec for spec in specs)
    assert [spec.id for spec in specs] == ["client.owned_sampled_peak_rss_sum",
        "client.owned_peak_rss_lower_bound", "client.owned_lifetime_peak_rss_sum_interval"]
    assert [spec.to_spec()["kind"] for spec in specs] == ["estimate", "lower_bound", "interval"]
    assert all(spec.to_spec()["physical_peak"] is False for spec in specs)
    assert specs[2].to_spec()["window"] == "isolated_client_group_birth_through_complete_retirement"
    case = suite().cases[0]
    assert {spec.id for spec in specs[:2]} <= {budget.metric for budget in case.budgets}
    assert all(row["outcome"] == "unknown" for row in assess_budgets(case, []))


def test_optional_root_endpoint_selection_is_separate_and_leaves_target_identities_unchanged():
    selected = experiment.ROOT_ENDPOINT_MEMORY_METRIC
    assert selected.id == "client.peak_rss" and selected.qualification == "root_only_report_endpoint"
    assert selected not in experiment.CLIENT_MEMORY_METRICS
    assert selected.to_spec()["window"] == "client_process_birth_through_report_endpoint"
    assert selected.to_spec()["physical_peak"] is False
    # Frozen ff0bc8e1 metadata fixture: selection changes no suite/target/budget.
    configured = suite()
    assert configured.digest == "5e623ff074b3bc97b4daaed108e1724aa2cffd4802215cd773f604d37b74391d"
    assert [case.target.digest for case in configured.cases] == [
        "63320f1e022185b3af79f6c139df03af4f467e68b02ee797a47c93a5f8c28d27",
        "2e39a4f9de6086fdb549cdfa4f4b6ba9e164562b556f7b8c8eb5c5c28c9bb981"]
    assert "qualification" not in json.dumps(configured.to_spec())
    assert [budget.metric for budget in configured.cases[0].budgets] == list(experiment.MEASUREMENTS)


@pytest.mark.parametrize("birth_bound", [True, False])
def test_campaign_endpoint_uses_ordinary_guard_without_promoting_lifetime_or_legacy(monkeypatch, birth_bound):
    from test_client_endpoint_memory_comparison import run, scope
    from pllm.runtime import client_process_usage

    # Reuse the independently tested synthetic ordinary Report fixture, not a
    # checkpoint run or a new metric producer. Darwin usage records are doubles.
    monkeypatch.setattr(client_process_usage, "supported", lambda: True)
    observed = run(change=lambda row: scope(row)["lifetime_owner"].update(root_birth_bound=birth_bound))
    before = observed.to_dict()
    endpoint = observed.compare("control", "candidate", experiment.ROOT_ENDPOINT_MEMORY_METRIC)
    if birth_bound:
        assert endpoint["status"] == "comparable_in_scope"
        assert endpoint["paired_control_over_candidate"] == [100]
        assert endpoint["window"] == "client_process_birth_through_report_endpoint"
        assert not endpoint["physical_peak"] and not endpoint["whole_lifetime"]
        interval = observed.compare("control", "candidate", experiment.CLIENT_MEMORY_METRICS[2])
        assert interval["status"] == "bounded" and interval["reduction_status"] == "not_established"
        assert interval["pairs"][0]["candidate"]["upper_bytes"] == 1 << 30
        assert endpoint["reduction_status"] == "not_established"
    else:
        assert endpoint["status"] == "unranked"
        assert observed.compare("control", "candidate", experiment.CLIENT_MEMORY_METRICS[2])["status"] == "unranked"
    # The legacy selector remains unchanged, but is not an opt-in fallback.
    assert observed.compare("control", "candidate", "client.peak_rss")["status"] == "comparable"
    assert all(row["status"] == "completed" for row in before["results"])
    assert observed.to_dict() == before


def test_tracked_public_asset_bytes_and_shared_inputs_without_tokenizer_claim():
    import hashlib

    raw = (Path(experiment.__file__).parent / "low_memory_qwen3_prompt.txt").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "34281fc41a22c45475b85498b511a7cd4ca8852a13e7fa39912bb4f7bb6626a7"
    assert raw.endswith(b"\n")  # Do not silently strip whitespace in either arm.
    assert experiment.private_inputs(raw.decode()) == {
        name: {"prompt": raw.decode()} for name in experiment.NAMES}


def test_public_cli_wrapper_dry_run_preserves_entire_suite(monkeypatch, capsys):
    from pllm._cli.app import main
    import pllm.metrics

    for key, value in CLI_SETTINGS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(pllm.metrics, "benchmark_suite", lambda *a, **kw: pytest.fail("dry run executed"))
    main(["benchmark", "run", "--experiment", "examples.papers.low_memory_qwen3:cli_suite",
          "--factory", "--trust-python", "--no-input", "--dry-run", "--format", "json",
          "--prompt-file", str(Path(experiment.__file__).parent / "low_memory_qwen3_prompt.txt")])
    data = json.loads(capsys.readouterr().out)["data"]
    assert data["suite"] == experiment.cli_suite().to_spec()
    assert data["report_schema"] == "pllm.benchmark_run.v1"
    assert data["dry_run"] is True
