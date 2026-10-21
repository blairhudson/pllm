"""Ordinary benchmark/gateway integration with bounded source and role doubles.

No model execution or native acceptance claim: the parent runs post-wheel tests
against real authenticated worker processes and source checkpoints.
"""

import asyncio
from contextlib import closing, contextmanager, nullcontext
from dataclasses import asdict
import json
import threading
from types import SimpleNamespace as NS

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model, _native
from pllm._benchmark_contract import TargetUnavailable
from pllm._benchmark_worker import Measurement
from pllm.metrics import BenchmarkTarget, Sampling, Scenario
from pllm.profiles import OpenWeightVerifiableCpu, resolve_runtime_composition
from pllm.runtime import benchmark_cli, benchmark_delegated as delegated, dashboard
from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun
from pllm.runtime.benchmark_response_target import execute_response, response_metrics
from pllm.runtime.benchmark_targets import execute_target, model_context


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    value = Experiment("gkm-ordinary-unit",
        OpenWeightVerifiableCpu(Model.path(str(tmp_path / "source"), model_id="gkm-unit")),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(4, 8, 3))

    def resolve(item):
        from pllm.roles.topology import graph_for_runtime

        options = resolve_runtime_composition(item.pipeline)
        return NS(**asdict(options), model="gkm-unit", profile=item.pipeline.profile,
                  role_graph=graph_for_runtime(options))

    # The installed extension need not yet contain the parent's new bindings.
    monkeypatch.setattr(Experiment, "resolve", resolve)
    return value


@pytest.fixture
def native_contract(experiment, monkeypatch):
    from pllm.runtime import delegated_source
    from pllm.protocols import OpenWeightVerifiableOutsourcing as Gkm

    source = NS(config_bytes=b'{"vocab_size":32}', checkpoint_digest="1" * 64,
                source_lock_digest="2" * 64, tokenizer_digest="3" * 64)
    document = {"decoder": {"plan": {"schema_version": "pllm.decoder_plan.v1",
        "config_digest": "4" * 64}, "numeric_profile": Gkm.NUMERIC_PROFILE,
        "local_numeric_profile": Gkm.LOCAL_NUMERIC_PROFILE, "input_rows": 8,
        "decode_steps": 2, "vocabulary": 32, "width": 4, "matrices": [{"rows": 4, "columns": 4}],
        "source_float32_bytes": 128, "execution_payload_estimate_bytes": 2048}}
    calls = []

    source.runtime_contract_bytes = json.dumps(document, sort_keys=True).encode()
    source.runtime_contract_digest = "6" * 64

    def admit(exp):
        assert (exp.budget.max_input_tokens, exp.budget.max_new_tokens) == (8, 3)
        calls.append(exp.to_spec())
        return source

    monkeypatch.setattr(delegated_source, "admit_source", admit)
    monkeypatch.setattr(_native, "gkm_runtime_contract",
                        lambda *_a: pytest.fail("admitted contract rebound"), raising=False)
    monkeypatch.setattr("pllm.lower_model", lambda *_a, **_kw: pytest.fail("ordinary W8A8 lowering"))
    return NS(source=source, document=document, calls=calls)


def counters():
    result = dict(plaintext_prompt_bytes_sent=0, plaintext_token_ids_sent=0,
        bundle_network_bytes=0, generation_completed_responses=1, generation_output_tokens=3,
        generation_decode_output_tokens=2,
        preparation_requests_during_online=0, inference_upload_bytes=20, inference_download_bytes=28,
        inference_control_upload_bytes=36, inference_control_download_bytes=10,
        generation_prefill_online_body_bytes=30, generation_decode_online_body_bytes=18)
    for role in ("worker_a", "worker_b"):
        for key, value in dict(setup_upload_bytes=11, setup_download_bytes=3,
            online_upload_bytes=10, online_download_bytes=14,
            teardown_upload_bytes=7, teardown_download_bytes=2).items():
            result[f"role_link.{role}.{key}"] = value
    return result


def record(**overrides):
    return BenchmarkRun(run_id="gkm_run", status="completed", model_id="gkm-unit",
        max_output_tokens=3, cold=True, warm=False, started_at_ns=1_000_000_000,
        online_started_at_ns=1_100_000_000, first_token_at_ns=2_900_000_000,
        last_token_at_ns=2_900_001_000, finished_at_ns=3_000_000_000,
        input_tokens=3, output_tokens=3, token_usage_authoritative=True,
        privacy_delta=counters(), decode_online_body_bytes=18, decode_output_tokens=2,
        **overrides)


def report(experiment):
    metadata, _ = delegated.inspect_runtime(experiment)
    run = record(model_fingerprint=metadata["model_fingerprint"],
                 output_delivery=delegated.DELIVERY).to_dict()
    run["generation"] = {"output_text_digest": "a" * 64, "response_status": "incomplete"}
    startup = {key: 0 for key in counters()}
    return benchmark_cli.build_loopback_report(model_id="gkm-unit", tiny=False,
        max_output_tokens=3, runs=[run], warmup_runs=[], roles=("client", "worker_a", "worker_b"),
        initial_preparation_audit=startup, runtime_contract=metadata, inventory_policy="none",
        source_lock_digest=metadata["source_lock_digest"], temperature=0, capture_output_digest=True)


def meter(tmp_path):
    return Measurement(dict(sampling=Sampling(processes=1).to_spec(), salt="ab" * 32,
                            inputs={}, directory=str(tmp_path)))


def test_native_metadata_pins_full_envelope_and_numeric_identity(experiment, native_contract):
    metadata, contract = delegated.inspect_runtime(experiment)
    assert native_contract.calls == [experiment.to_spec()]
    assert contract == native_contract.document
    assert metadata["runtime_kind"] == "gkm"
    assert metadata["numeric"]["lookup"] == "original_float32"
    assert metadata["output_tokens_bound"] == 3
    assert metadata["native_runtime_contract_digest"] == native_contract.source.runtime_contract_digest
    assert metadata["preparation_inventory"] is False
    assert metadata["decode_tps_supported"] is False
    assert metadata["model_fingerprint"] != native_contract.source.source_lock_digest
    native_contract.source.tokenizer_digest = "5" * 64
    other, _ = delegated.inspect_runtime(experiment)
    assert other["model_fingerprint"] != metadata["model_fingerprint"]
    assert not any(key in json.dumps(metadata) for key in ("weight_bits", "activation_bits"))


def test_seeded_benchmark_keeps_numeric_identity_and_reports_actual_sharing(experiment, native_contract):
    full, _ = delegated.inspect_runtime(experiment)
    seeded = experiment.with_params(pipeline__linear__sharing="seeded")
    native_contract.source.runtime_contract_digest = "7" * 64
    changed, _ = delegated.inspect_runtime(seeded)
    assert changed["sharing"] == "seeded" and full["sharing"] == "full"
    assert changed["numeric"] == full["numeric"]
    assert changed["model_fingerprint"] == full["model_fingerprint"]
    assert changed["native_runtime_contract_digest"] != full["native_runtime_contract_digest"]
    assert changed["verification_window"] == full["verification_window"] == "request_end"
    assert changed["decode_tps_supported"] is False


@pytest.mark.parametrize("options", [
    {"backend": "docker"}, {"backend": "auto"}, {"docker": True},
    {"docker_image": "image"}, {"docker_network": object()}, {"cache_bytes": 1},
    {"cache_mode": "prefix"}, {"cache_bound": 8}, {"inventory_policy": "prewarm"},
    {"inventory_policy": "request-sized"}, {"bundle_compression": "zlib"},
    {"temperature": 0.7}, {"temperature": True}, {"output_tokens": 4},
    {"output_tokens": 0}, {"requests": 5}, {"requests": 0},
])
def test_unsupported_options_fail_without_source(experiment, monkeypatch, options):
    from pllm.runtime import delegated_source
    monkeypatch.setattr(delegated_source, "admit_source", lambda _: pytest.fail("source touched"))
    with pytest.raises(ValueError):
        delegated.admit_options(experiment, **{"output_tokens": 3, **options})


@pytest.mark.parametrize("changes", [
    {"budget__max_input_tokens": 33}, {"budget__max_new_tokens": 10},
    {"budget__requests": 65}, {"pipeline__model": Model.tiny()},
])
def test_saved_envelope_is_admitted_before_smaller_request(experiment, changes):
    with pytest.raises(ValueError, match="saved source/workload envelope"):
        delegated.admit_options(experiment.with_params(**changes), output_tokens=1)


def test_history_persists_buffered_timing_without_fabricated_tps():
    ordinary = record().to_dict()
    buffered = record(output_delivery=delegated.DELIVERY, generation_seconds=0.000001).to_dict()
    assert ordinary["durations"]["tokens_per_second"] == 2_000_000
    assert "measurement" not in ordinary
    assert buffered["durations"]["tokens_per_second"] is None
    assert buffered["durations"]["generation_seconds"] is None
    assert buffered["durations"]["ttft_seconds"] == 1.8
    assert buffered["network"]["decode_scope"] == "semantic_decode_before_request_end_verification"
    assert buffered["measurement"]["decode_tps_unavailable_reason"] == delegated.TPS_UNAVAILABLE
    with closing(BenchmarkHistory(":memory:")) as history:
        history.add(record(output_delivery=delegated.DELIVERY))
        assert history.get("gkm_run")["measurement"] == buffered["measurement"]


def test_report_and_metrics_keep_worker_coverage_and_missing_resources(experiment, native_contract, tmp_path):
    result = report(experiment)
    assert result["checks"]["passed"], result["checks"]
    assert result["checks"]["fixed_count_verified_delivery"]
    assert result["configuration"]["inventory_policy"] == "none"
    assert result["configuration"]["runtime_kind"] == "gkm"
    assert result["summary"]["median_tokens_per_second"] is None
    m = meter(tmp_path)
    response_metrics(result, m)
    by_id = {item["id"]: item for item in m.metrics}
    assert by_id["response.decode_tps"]["value"] is None
    assert by_id["response.decode_tps"]["unavailable_reason"] == delegated.TPS_UNAVAILABLE
    for role in ("worker_a", "worker_b"):
        assert by_id[f"{role}.cold_cpu"]["value"] is None
        assert by_id[f"{role}.peak_rss"]["value"] is None
    assert by_id["network.all_link_stream_bytes"]["value"] is None
    totals = m.details["runtime_audit"]["totals"]
    assert totals["generation_prefill_online_body_bytes"] == 30
    assert totals["generation_decode_online_body_bytes"] == 18
    assert totals["role_link.worker_b.teardown_download_bytes"] == 2
    assert "inference_server_ns" not in totals
    # Empty decoded text (e.g. special tokens only) has no first-text observation.
    result["runs"][0]["durations"]["ttft_seconds"] = None
    result["runs"][0]["durations"]["tokens_per_second"] = 1e99
    m = meter(tmp_path)
    response_metrics(result, m)
    assert next(item for item in m.metrics if item["id"] == "response.ttft")["value"] is None
    assert next(item for item in m.metrics if item["id"] == "response.decode_tps")["value"] is None


def test_native_geometry_target_never_enters_ordinary_lowering(experiment, native_contract, tmp_path):
    m = meter(tmp_path)
    execute_target(BenchmarkTarget.geometry(experiment), m)
    assert m.oracle["kind"] == "native_gkm_semantic_contract"
    assert m.details["runtime_contract"]["numeric"]["kv"] == "full"
    assert m.details["model"]["checkpoint_digest"] == "1" * 64
    assert m.fixture["metadata"]["values_loaded"] is False
    with pytest.raises(TargetUnavailable, match="gkm_requires_native_geometry"):
        model_context(BenchmarkTarget.geometry(experiment).to_spec()["parameters"], m)


@pytest.mark.parametrize("extra", [{"concurrent_requests": 2},
                                  {"client_metadata": True}, {"client_mlp": {"enabled": True}}])
def test_response_target_rejects_non_native_extensions_before_memory(experiment, monkeypatch, tmp_path, extra):
    monkeypatch.setattr("pllm.metrics.benchmark_memory", lambda *_a, **_k: pytest.fail("memory/source touched"))
    parameters = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3)).to_spec()["parameters"]
    parameters.update(extra)
    m = meter(tmp_path)
    m.inputs = {"prompt": "hello"}
    with pytest.raises(TargetUnavailable, match="gkm_response_options_unsupported"):
        execute_response(parameters, m)


@pytest.fixture
def roles(experiment, native_contract, monkeypatch):
    from pllm.runtime import benchmark_memory

    emitted, entered, release = threading.Event(), threading.Event(), threading.Event()
    evictions, options = [], []
    audit = {key: 0 for key in ("plaintext_prompt_bytes_sent", "plaintext_token_ids_sent",
                              "preparation_requests_during_online")}

    class Client:
        def __init__(self):
            self._core = self
            self.closed = False
            self.failure = None
            self.responses = NS(create=self.create)
            self.models = NS(list=lambda: {"data": [{"id": "gkm-unit", "runtime": {
                key: getattr(native_contract.source, key)
                for key in ("checkpoint_digest", "source_lock_digest", "tokenizer_digest")}}]})
            self.privacy_audit = NS(to_dict=lambda: dict(audit))

        def tokenizer_scope(self):
            return nullcontext()

        def _response_input_tokens(self, model, prompt):
            return 3

        def create(self, **kwargs):
            assert kwargs["temperature"] == 0 and kwargs["stream"] is True
            entered.set()
            assert release.wait(5), "finish gate was not released"
            if self.closed:
                raise RuntimeError("cancelled before acceptance")
            if self.failure is not None:
                raise self.failure
            for key, value in counters().items():
                audit[key] = audit.get(key, 0) + value
            emitted.set()
            yield {"type": "response.created", "response": {"id": "resp_unit"}}
            yield {"type": "response.output_text.delta", "delta": "accepted words"}
            yield {"type": "response.incomplete", "response": {"id": "resp_unit",
                "status": "incomplete", "usage": {"input_tokens": 3, "output_tokens": 3},
                "incomplete_details": {"reason": "max_output_tokens"}, "output": []}}

        def evict_response(self, value):
            evictions.append(value)

        def close(self):
            self.closed = True
            release.set()

    client = Client()

    class Topology:
        started = False
        closed = False
        requires_preparation = False
        statuses = tuple(NS(role=role, url=f"http://{role}", pid=None, running=True)
                         for role in ("worker_a", "worker_b"))
        _process_lock = threading.Lock()
        _processes = {role: NS(pid=None) for role in ("worker_a", "worker_b")}

        def start(self):
            self.started = True
            return self

        def close(self):
            self.closed = True

        def client(self, **kwargs):
            options.append(kwargs)
            return client

        def worker_process_metrics(self):
            return {role: {"cpu_ns": None, "peak_rss_bytes": None} for role in ("worker_a", "worker_b")}

    topology = Topology()
    monkeypatch.setattr(dashboard, "build_roles", lambda *_a, **_kw: topology)
    monkeypatch.setattr(benchmark_memory, "benchmark_memory", lambda *_a, **_kw: {
        "admitted": True, "selected_backend": "native", "estimate": {"max_output_tokens": 3,
            "max_input_tokens": 8,
            "provider_peak_bytes": {"worker_a": 1 << 20, "worker_b": 1 << 20}}})
    monkeypatch.setattr(benchmark_memory, "MemoryWatchdog", lambda *_a, **_kw:
                        NS(start=lambda: None, close=lambda: None, error=None,
                           minimum_available_bytes=None, maximum_swap_growth_bytes=None))
    return NS(topology=topology, client=client, options=options, entered=entered,
              release=release, emitted=emitted, evictions=evictions)


def runtime(experiment, history=None, **kwargs):
    return dashboard.DashboardRuntime(dashboard.DashboardConfig(host="127.0.0.1", port=0,
        model_path=experiment.pipeline.model.source, model_id="gkm-unit", default_max_output_tokens=3,
        experiment=experiment, **kwargs), dashboard.OTelStore(), history)


def test_dashboard_finish_gate_and_thread_cleanup(experiment, roles):
    history = BenchmarkHistory(":memory:")
    value = runtime(experiment, history)

    async def exercise():
        await value.start()
        try:
            assert value.snapshot()["run"]["phase"] == "ready", value.snapshot()["run"]
            assert roles.options == [{"timeout": 300}]
            assert value._inventory_rows == 0
            identifier = value.begin("hello", 3)
            assert await asyncio.to_thread(roles.entered.wait, 2)
            assert not roles.emitted.is_set()
            snapshot = value.snapshot()["run"]
            assert snapshot["first_token_at"] is None and snapshot["text"] == ""
            assert snapshot["tps"] is None and history.get(identifier) is None
            worker = value._worker
            roles.release.set()
            await asyncio.to_thread(worker.join, 3)
            assert not worker.is_alive()
            persisted = history.get(identifier)
            assert persisted["status"] == "completed"
            assert persisted["tokens"]["output_tokens"] == 3
            assert persisted["durations"]["tokens_per_second"] is None
            assert persisted["network"]["decode_online_body_bytes"] == 18
            assert value.snapshot()["run"]["tps"] is None
            assert roles.evictions == ["resp_unit"]
        finally:
            await value.stop()

    asyncio.run(exercise())
    assert roles.client.closed and roles.topology.closed
    assert not value.has_live_worker() and not value._background_threads
    history.close()


def test_dashboard_stop_cancels_and_drains_unaccepted_request(experiment, roles):
    history = BenchmarkHistory(":memory:")
    value = runtime(experiment, history)

    async def exercise():
        await value.start()
        identifier = value.begin("hello", 3)
        assert await asyncio.to_thread(roles.entered.wait, 2)
        worker = value._worker
        await value.stop()
        assert not worker.is_alive() and not roles.emitted.is_set()
        assert history.get(identifier)["status"] == "failed"
        assert value.snapshot()["run"]["text"] == ""

    asyncio.run(exercise())
    assert not value._background_threads and roles.topology.closed
    history.close()


@pytest.mark.parametrize("through_target", [False, True])
def test_ordinary_loopback_benchmark_uses_native_no_inventory_contract(experiment, roles, tmp_path, through_target):
    roles.release.set()
    if through_target:
        m = meter(tmp_path)
        m.inputs = {"prompt": "hello"}
        target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3))
        execute_target(target, m)
        result = m.details["report"]
        assert m.details["model"]["checkpoint_digest"] == "1" * 64
        assert m.details["runtime_contract"] == result["runtime_contract"]
        assert all(value is not None for value in m.comparison.values())
    else:
        result = benchmark_cli.run_loopback_benchmark(experiment=experiment, model="ignored",
            model_id=None, tiny=False,
            prompt="hello", max_output_tokens=3, warmups=0, repetitions=1, timeout_seconds=10)
    assert result["checks"]["passed"], result["checks"]
    assert result["configuration"]["inventory_policy"] == "none"
    assert result["configuration"]["sampling"]["effective_temperature"] == 0
    assert result["runtime_contract"]["output_delivery"] == delegated.DELIVERY
    assert result["runs"][0]["measurement"]["decode_tps_supported"] is False
    assert roles.options == [{"timeout": 300 if through_target else 10}]
    assert roles.topology.closed and roles.client.closed


def test_tokenizer_scope_admission_failure_finishes_dashboard_request(experiment, roles, monkeypatch):
    value = runtime(experiment, BenchmarkHistory(":memory:"))

    def reject():
        raise ValueError("source changed during tokenizer admission")

    monkeypatch.setattr(roles.client, "tokenizer_scope", reject)

    async def exercise():
        await value.start()
        try:
            identifier = value.begin("hello", 3)
            worker = value._worker
            await asyncio.to_thread(worker.join, 3)
            assert not worker.is_alive() and not roles.entered.is_set()
            assert value.history.get(identifier)["status"] == "failed"
            assert value._active_run is None
        finally:
            await value.stop()

    asyncio.run(exercise())
    value.history.close()


def test_verification_failure_records_no_text_or_success(experiment, roles):
    value = runtime(experiment, BenchmarkHistory(":memory:"))
    roles.client.failure = RuntimeError("GKM request-end verification failed")
    roles.release.set()

    async def exercise():
        await value.start()
        try:
            identifier = value.begin("hello", 3)
            worker = value._worker
            await asyncio.to_thread(worker.join, 3)
            assert not worker.is_alive() and roles.entered.is_set()
            assert not roles.emitted.is_set() and not roles.evictions
            row = value.history.get(identifier)
            assert row["status"] == "failed" and not row["tokens"]["authoritative"]
            assert row["durations"]["ttft_seconds"] is None
            assert value.snapshot()["run"]["text"] == ""
        finally:
            await value.stop()

    asyncio.run(exercise())
    value.history.close()


@pytest.mark.parametrize("options", [{"startup_inventory_rows": 1}, {"prefill_cache_mib": 1},
                                  {"backend": "auto"}, {"temperature": 1.0}])
def test_dashboard_rejects_incompatible_configuration(experiment, options):
    with pytest.raises(ValueError):
        runtime(experiment, **options)


def test_ordinary_local_gateway_omits_inventory_defaults(experiment, monkeypatch):
    from pllm._cli.app import build_parser
    from pllm.runtime import cli

    monkeypatch.delenv("PLLM_PYDEPS", raising=False)
    args = build_parser().parse_args(["gateway", "--local"])
    args.resolved_experiment = experiment
    calls = []

    @contextmanager
    def build(model, **kwargs):
        assert model is experiment
        calls.append(("roles", kwargs))
        try:
            yield NS(gateway_app=lambda **kwargs: calls.append(("gateway", kwargs)))
        finally:
            calls.append(("closed", {}))

    monkeypatch.setattr(cli, "build_roles", build)
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_a, **_kw: None)
    cli.run_local_gateway(args)
    assert calls[1] == ("gateway", {"local_api_key": "local", "timeout": 300.0})
    assert calls[-1][0] == "closed"
    for field, value in (("transport", "http"), ("prepared_inventory_rows", 1), ("weight_bits", 8)):
        changed = NS(**vars(args))
        setattr(changed, field, value)
        with pytest.raises(cli.RuntimeCLIError, match="exact Experiment"):
            cli.run_local_gateway(changed)
    assert len(calls) == 3
