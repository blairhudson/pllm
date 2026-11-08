"""Source-level integration regressions; installed v3 SDK bindings are doubled.

These tests use genuine bounded Safetensors/tokenizer fixtures and real decoder
geometry admission. They do not build native code, launch roles, or time models.
"""

from dataclasses import replace
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm import model_loader, modeling
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.runtime import benchmark_memory as memory
from pllm.runtime import delegated_source

from test_delegated_source import native_sdk_double, write_source_fixture


@pytest.fixture
def sdk_native(monkeypatch, native_sdk_double):
    """Only the new SDK native interface is doubled; source/geometry stay real."""
    native_resolve = _native.resolve_experiment

    def resolve(payload):
        native_resolve(payload)
        experiment = Experiment.from_spec(json.loads(payload))
        return SimpleNamespace(
            canonical_composition=experiment.pipeline.canonical_bytes(),
            composition_digest=experiment.pipeline.digest(),
            configuration_digest=experiment.configuration_digest(),
        )

    monkeypatch.setattr(_native, "resolve_experiment", resolve)
    return native_sdk_double


@pytest.fixture
def experiment(tmp_path, sdk_native):
    if not callable(getattr(_native, "gkm_decoder_contract", None)):
        pytest.skip("source integration requires the installed GKM decoder geometry binding")
    root = tmp_path / "source"
    write_source_fixture(root)
    return Experiment("gkm-benchmark-admission", OpenWeightVerifiableCpu(Model.path(str(root))),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=2, max_input_tokens=3, max_new_tokens=3))


def forbidden(*args, **kwargs):
    raise AssertionError("ordinary lowering or tensor materialization was attempted")


def forbid_ordinary(monkeypatch):
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(modeling, "lower_model", forbidden)
    monkeypatch.setattr(delegated_source.AdmittedSource, "load_values", forbidden)


def test_memory_admits_saved_envelope_before_smaller_measurement(experiment, sdk_native, monkeypatch):
    forbid_ordinary(monkeypatch)
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
        8 << 30, 7 << 30, 0, 20 << 30))
    result = memory.benchmark_memory(experiment, max_input_tokens=1, max_output_tokens=1)
    assert result["admitted"] is True
    assert result["selected_backend"] == "native"
    assert result["estimate"]["runtime_kind"] == "gkm"
    assert result["estimate"]["model_plan_digest"] is None
    assert set(result["estimate"]["provider_peak_bytes"]) == {"worker_a", "worker_b"}
    # A small timed request must not relax the saved source/runtime envelope.
    assert sdk_native["runtime"][0][2:4] == (3, 3)
    envelope = result["estimate"]["gkm_admission"]
    assert (envelope["max_input_tokens"], envelope["max_output_tokens"]) == (3, 3)
    assert envelope["source_lock_digest"] == sdk_native["runtime"][0][4]
    assert envelope["tokenizer_digest"] == sdk_native["runtime"][0][5]
    assert envelope["numeric_profile"].startswith("gkm.")
    assert result["estimate"]["max_input_tokens"] == 3
    assert result["estimate"]["max_output_tokens"] == 3
    assert result["estimate"]["verification_allocation"]["contract_execution_payload_estimate_bytes"] == envelope["execution_payload_estimate_bytes"]
    assert result["candidates"]["docker"]["admitted"] is False


def test_oversize_saved_budget_cannot_hide_behind_small_measurement(experiment, sdk_native, monkeypatch):
    forbid_ordinary(monkeypatch)
    monkeypatch.setattr(delegated_source, "_files", forbidden)
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(
        8 << 30, 7 << 30, 0, 20 << 30))
    oversized = experiment.with_params(budget__max_input_tokens=33)
    with pytest.raises(ValueError, match="limit"):
        memory.benchmark_memory(oversized, max_input_tokens=1, max_output_tokens=1)
    assert sdk_native["runtime"] == []


def test_benchmark_identity_reuses_admitted_native_bytes(experiment, sdk_native, monkeypatch):
    from pllm.runtime.benchmark_delegated import inspect_runtime

    forbid_ordinary(monkeypatch)
    metadata, native_contract = inspect_runtime(experiment)
    admitted = delegated_source.admit_source(experiment)
    assert metadata["source_lock_digest"] == admitted.source_lock_digest
    assert metadata["checkpoint_digest"] == admitted.checkpoint_digest
    assert metadata["tokenizer_digest"] == admitted.tokenizer_digest
    assert native_contract == json.loads(admitted.runtime_contract_bytes)
    assert metadata["native_runtime_contract_digest"] == admitted.runtime_contract_digest
    assert metadata["numeric"]["numeric_profile"] == admitted.contract["numeric_profile"]
    assert metadata["output_delivery"] == "buffered_until_verified"
    assert metadata["decode_tps_supported"] is False
    # One native SDK binding per source admission, with no second unvalidated bind.
    assert len(sdk_native["runtime"]) == 2


def test_cli_config_roundtrip_needs_no_model_lowering(experiment, tmp_path, capsys, monkeypatch):
    from pllm._cli.app import main
    from pllm._cli.targets import resolve_target

    forbid_ordinary(monkeypatch)
    path = tmp_path / "experiment.json"
    path.write_bytes(experiment.canonical_bytes())
    main(["config", "show", str(path), "--format", "json"])
    output = json.loads(capsys.readouterr().out)["data"]
    assert output["configuration"] == experiment.to_spec()
    exported = tmp_path / "experiment.yaml"
    main(["config", "export", str(path), "--output", str(exported), "--format", "json"])
    capsys.readouterr()
    restored = resolve_target(str(exported), factory=False, no_input=True,
                              trust_python=False, output_format="json").configuration
    assert restored.canonical_bytes() == experiment.canonical_bytes()
    assert restored.resolve().client_runtime == "compiled_gkm_v1"


def test_cli_gkm_dry_run_records_effective_greedy_policy(experiment, tmp_path, capsys, monkeypatch):
    from pllm._cli.app import main

    forbid_ordinary(monkeypatch)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    path = tmp_path / "experiment.json"
    path.write_bytes(experiment.canonical_bytes())
    main(["benchmark", "run", "--experiment", str(path), "--prompt", "hello",
          "--max-output-tokens", "3", "--dry-run", "--format", "json"])
    config = json.loads(capsys.readouterr().out)["data"]["configuration"]
    assert config["sampling"]["effective_temperature"] == 0
    assert config["sampling"]["mode"] == "greedy"
    assert config["inventory_policy"] == "none"


def test_buffered_history_cannot_report_emission_as_decode_speed():
    from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun

    record = BenchmarkRun(run_id="buffered", status="completed", model_id="fixture",
        max_output_tokens=3, cold=True, warm=False, started_at_ns=1,
        online_started_at_ns=2, first_token_at_ns=4_000_000_002,
        last_token_at_ns=4_000_001_002, finished_at_ns=4_000_002_002,
        input_tokens=3, output_tokens=3, token_usage_authoritative=True,
        output_delivery="buffered_until_verified")
    history = BenchmarkHistory(":memory:")
    try:
        history.add(record)
        saved = history.get("buffered")
        assert saved["durations"]["ttft_seconds"] == 4
        assert saved["durations"]["generation_seconds"] is None
        assert saved["durations"]["tokens_per_second"] is None
        assert saved["measurement"]["decode_tps_supported"] is False
        assert replace(record, output_delivery="incremental").to_dict()["durations"]["tokens_per_second"] == 2_000_000
    finally:
        history.close()


def buffered_report(*, missing_counter=False, ttft=None):
    from pllm.runtime.benchmark_cli import build_loopback_report
    from pllm.runtime.benchmark_history import BenchmarkRun
    from test_delegated_client import transport_counters

    counters = transport_counters() | {
        "bundle_network_bytes": 0, "plaintext_prompt_bytes_sent": 0,
        "plaintext_token_ids_sent": 0, "preparation_requests_during_online": 0,
        "generation_completed_responses": 1, "generation_output_tokens": 3,
        "generation_decode_output_tokens": 2,
    }
    startup = dict.fromkeys(counters, 0)
    if missing_counter:
        del counters["role_link.worker_b.teardown_download_bytes"]
    run = BenchmarkRun(run_id="buffered", status="completed", model_id="fixture",
        model_fingerprint="a" * 64, max_output_tokens=3, cold=True, warm=False,
        started_at_ns=1, finished_at_ns=4_000_000_001, full_seconds=4,
        ttft_seconds=ttft, input_tokens=3, output_tokens=3, token_usage_authoritative=True,
        privacy_delta=counters, output_delivery="buffered_until_verified").to_dict()
    return build_loopback_report(model_id="fixture", tiny=False, max_output_tokens=3,
        warmup_runs=[], runs=[run], roles=("client", "worker_a", "worker_b"),
        initial_preparation_audit=startup, temperature=0, runtime_contract={
            "runtime_kind": "gkm", "output_delivery": "buffered_until_verified"})


def test_missing_worker_counter_is_unknown_not_zero():
    report = buffered_report(missing_counter=True)
    assert report["checks"]["two_worker_bodies_tracked"] is False
    assert report["checks"]["passed"] is False
    assert report["summary"]["total_accounted_benchmark_body_bytes"] is None
    assert report["summary"]["median_tokens_per_second"] is None


def test_response_metrics_preserve_empty_text_and_unmeasured_counters():
    from pllm._benchmark_evidence import measurement
    from pllm.runtime.benchmark_response_target import response_metrics

    metrics = {}
    measured = SimpleNamespace(details={}, metric=lambda name, value, **kwargs:
                              metrics.update({name: measurement(name, value, **kwargs)}))
    report = buffered_report()
    assert report["checks"]["passed"] is True
    response_metrics(report, measured)
    assert metrics["response.ttft"]["value"] is None
    assert metrics["response.ttft"]["window"] == "online_to_first_accepted_text"
    assert metrics["response.decode_tps"]["value"] is None
    assert metrics["response.decode_tps"]["unavailable_reason"] == "request_end_verification_buffers_all_output"
    assert metrics["request.0.decode_tps"]["value"] is None
    assert metrics["network.covered_body_bytes"]["value"] == 94
    assert metrics["network.all_link_stream_bytes"]["value"] is None
    assert metrics["worker_a.peak_rss"]["value"] is None
    assert metrics["worker_b.cold_cpu"]["value"] is None


def test_missing_ttft_is_not_dropped_from_aggregate():
    from pllm.runtime.benchmark_cli import build_loopback_report

    missing = buffered_report()
    available = buffered_report(ttft=1)
    report = build_loopback_report(model_id="fixture", tiny=False, max_output_tokens=3,
        warmup_runs=[], runs=[*missing["runs"], *available["runs"]],
        roles=("client", "worker_a", "worker_b"), temperature=0,
        runtime_contract=missing["runtime_contract"])
    assert report["runs"][0]["durations"]["ttft_seconds"] is None
    assert report["runs"][1]["durations"]["ttft_seconds"] == 1
    assert report["summary"]["median_ttft_seconds"] is None


def test_geometry_target_never_claims_reencoded_plan_as_native_identity(experiment, monkeypatch):
    from pllm._benchmark_contract import BenchmarkTarget
    from pllm.runtime.benchmark_targets import execute_target

    forbid_ordinary(monkeypatch)
    import pllm
    monkeypatch.setattr(pllm, "lower_model", forbidden)
    measured = SimpleNamespace(details={}, comparison={}, inputs={}, admit=lambda _: None,
                              run=lambda execute: [execute(0)])
    execute_target(BenchmarkTarget.geometry(experiment), measured)
    assert measured.details["runtime_contract"]["runtime_kind"] == "gkm"
    # Only the native SDK contract has a native canonical identity here. A JSON
    # re-encoding of its nested plan need not retain the native float spelling.
    assert measured.details["plan_digest"] is None
    assert measured.details["runtime_contract"]["native_runtime_contract_digest"] == measured.comparison["output_digest"]
    assert measured.fixture["metadata"]["values_loaded"] is False


def test_response_target_preserves_actual_source_pins(experiment, tmp_path, monkeypatch):
    """Adapter propagation only: the loopback runner is an explicit test double."""
    from pllm import metrics
    from pllm._benchmark_contract import BenchmarkTarget, Sampling, Scenario
    from pllm.runtime import benchmark_cli
    from pllm.runtime.benchmark_delegated import inspect_runtime
    from pllm.runtime.benchmark_response_target import execute_response

    forbid_ordinary(monkeypatch)
    metadata, _ = inspect_runtime(experiment)
    report = buffered_report(ttft=1)
    report["runtime_contract"] = metadata
    report["configuration"].update(source_lock_digest=metadata["source_lock_digest"],
                                   provider_backend="native")
    report["runs"][0]["model_fingerprint"] = metadata["model_fingerprint"]
    report["runs"][0]["generation"] = {"output_text_digest": "b" * 64}
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *args, **kwargs: {"admitted": True})
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        assert kwargs["temperature"] == 0
        assert kwargs["experiment"].pipeline == experiment.pipeline
        return report

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    measured = SimpleNamespace(inputs={"prompt": "hello"}, directory=tmp_path,
        sampling=Sampling(processes=1), salt=b"x" * 32, details={}, comparison={}, limitations=[],
        admit=lambda _: None, run=lambda execute: [execute(0)], metric=lambda *args, **kwargs: None)
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3))
    execute_response(target.to_spec()["parameters"], measured)
    assert len(calls) == 1
    assert measured.details["model"]["checkpoint_digest"] == metadata["checkpoint_digest"]
    assert measured.details["model"]["source_lock_digest"] == metadata["source_lock_digest"]
    assert measured.details["runtime_contract"]["numeric"] == metadata["numeric"]


def test_dashboard_identity_and_unknown_live_metrics(experiment, monkeypatch):
    from pllm.runtime.dashboard import DashboardConfig, DashboardRuntime, OTelStore
    from pllm.runtime.benchmark_delegated import inspect_runtime
    from pllm.runtime.delegated_client import DelegatedClient
    from test_delegated_client import CONNECTIONS

    forbid_ordinary(monkeypatch)
    metadata, _ = inspect_runtime(experiment)
    core = DelegatedClient(experiment, CONNECTIONS)
    runtime = DashboardRuntime(DashboardConfig(host="127.0.0.1", port=8791,
        model_path=experiment.pipeline.model.source, model_id=experiment.resolve().model,
        default_max_output_tokens=3, experiment=experiment), OTelStore())
    runtime._runtime_contract = metadata
    runtime._client = SimpleNamespace(models=SimpleNamespace(list=core.list_models),
                                     privacy_audit=core.audit)
    try:
        assert runtime._inventory_rows == 0
        assert runtime._discover_model_identity() == (
            metadata["model_fingerprint"], metadata["source_lock_digest"])
        runtime._state.update(tokens=3, first_token_at=1, last_token_at=1.000001)
        snapshot = runtime.snapshot()["run"]
        assert snapshot["tps"] is None
        assert all(value is None for value in snapshot["online_traffic"].values())
        assert snapshot["decode_tps_supported"] is False
    finally:
        core.close()


def test_local_gateway_passes_exact_experiment_without_ordinary_client_options(experiment, monkeypatch):
    from pllm.runtime import cli

    calls = {}
    app = object()

    @contextmanager
    def roles(model, **kwargs):
        assert model is experiment
        assert kwargs["weight_bits"] is None and kwargs["activation_bits"] is None
        calls["roles"] = kwargs

        def gateway_app(**options):
            calls["gateway"] = options
            return app

        yield SimpleNamespace(gateway_app=gateway_app)

    monkeypatch.setattr(cli, "build_roles", roles)
    monkeypatch.setattr(cli.uvicorn, "run", lambda value, **kwargs:
                        calls.update(served=value))
    cli.run_local_gateway(SimpleNamespace(resolved_experiment=experiment, port=8080,
                                         api_key="test-local-key"))
    assert calls["gateway"] == {"local_api_key": "test-local-key", "timeout": 300}
    assert calls["served"] is app


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("requests,input_tokens,output_tokens,admitted", [
    (32, 64, 8, True),
    (64, 512, 65, True),
    (1, 513, 1, False),
    (1, 1, 66, False),
    (65, 1, 1, False),
])
def test_decoder_benchmark_options_keep_saved_envelope_before_source(
    monkeypatch, sharing, requests, input_tokens, output_tokens, admitted,
):
    """Option admission only; accepted metadata is not native qualification."""
    from examples.papers import gkm
    from pllm.runtime.benchmark_delegated import admit_options

    forbid_ordinary(monkeypatch)
    monkeypatch.setattr(Experiment, "resolve", forbidden)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    experiment = gkm.experiment(budget=ExecutionBudget(
        requests=requests, max_input_tokens=input_tokens, max_new_tokens=output_tokens,
    )).with_params(pipeline__linear__sharing=sharing)
    # A one-token measurement must neither shrink the saved decoder envelope
    # into capsule bounds nor hide an over-limit saved budget.
    if admitted:
        admit_options(experiment, backend="native", output_tokens=1, requests=1)
    else:
        with pytest.raises(ValueError, match="saved source/workload envelope"):
            admit_options(experiment, backend="native", output_tokens=1, requests=1)
