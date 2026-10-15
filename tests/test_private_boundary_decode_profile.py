"""Model-free timing ownership and bootstrap tests; not performance evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from benchmarks.research import private_boundary_decode_profile as profile
from benchmarks.research.sota_combinations import Rejected
from pllm.metrics import BuildRef
from pllm._benchmark_contract import digest
from test_private_boundary_decode_benchmark import cohort  # noqa: F401


def span(label, start, end):
    return {"label": label, "start_ns": start, "end_ns": end, "calling_thread_cpu_ns": 0}


def test_parallel_workers_never_sum_into_elapsed():
    rows = [span("forward", 0, 100), span("head", 10, 90), span("paired_exchange", 20, 80),
            span("http_exchange", 25, 75), span("http_exchange", 30, 80)]
    result = profile.summarize(rows, (0, 100))
    assert result["elapsed_ns"] == 100
    assert result["exclusive_wall_ns"]["paired_exchange"] == 60
    assert result["exclusive_wall_ns"]["head_unclassified"] == 20
    assert result["exclusive_wall_ns"]["body_execution_excluding_lookup_and_head"] == 20
    assert result["diagnostic_nonadditive"]["http_exchange"]["union_wall_ns"] == 55
    assert result["partition_reconciled"]
    assert profile.union_ns([(0, 10), (2, 3), (5, 20)]) == 20


def test_decode_window_clips_first_token_work_and_request_tails():
    rows = [span("forward", 0, 100), span("token_lookup", 0, 10),
            span("lookup_exchange", 2, 9), span("audit_and_sampler", 100, 120)]
    result = profile.summarize(rows, (5, 110))
    assert result["elapsed_ns"] == 105
    assert result["exclusive_wall_ns"]["token_lookup"] == 5
    assert result["exclusive_wall_ns"]["paired_exchange"] == 0
    assert result["exclusive_wall_ns"]["audit_and_sampler"] == 10
    assert profile.summarize(rows, (200, 220))["exclusive_wall_ns"]["unclassified"] == 20


def test_wrappers_preserve_return_exception_and_keep_no_payload():
    record = profile.Recorder()
    secret = object()
    assert record.wrap("forward", lambda value: value)(secret) is secret
    error = ValueError("PRIVATE CONTENT")
    def fail():
        raise error
    with pytest.raises(ValueError) as caught:
        record.wrap("head", fail)()
    assert caught.value is error
    assert len(record.spans) == 2
    assert "PRIVATE" not in json.dumps(record.spans)
    assert set(record.spans[0]) == {"label", "start_ns", "end_ns", "calling_thread_cpu_ns"}


def test_span_bound_fails_closed(monkeypatch):
    monkeypatch.setattr(profile, "MAX_SPANS", 1)
    record = profile.Recorder()
    record.wrap("forward", lambda: None)()
    with pytest.raises(RuntimeError, match="span bound"):
        record.wrap("forward", lambda: None)()
    assert record.failed
    assert len(record.spans) == 1


def test_exact_monotonic_capture_and_no_epoch_disclosure():
    record = profile.Recorder()
    record.spans = [span("forward", 1000, 1100)]
    record.capture(SimpleNamespace(started_monotonic_ns=1000, finished_monotonic_ns=1200,
        first_token_monotonic_ns=1050, last_token_monotonic_ns=1150))
    report = record.report()
    assert report["windows_relative_ns"]["after_first_output"] == [50, 150]
    assert report["request"]["elapsed_ns"] == 200
    assert report["after_first_output"]["elapsed_ns"] == 100
    assert report["spans"][0]["start_ns"] == 0
    assert report["role_cpu_attribution"] is None
    assert report["share_generation_attribution"] is None
    with pytest.raises(RuntimeError):
        record.capture(SimpleNamespace(started_monotonic_ns=0, finished_monotonic_ns=1,
            first_token_monotonic_ns=0, last_token_monotonic_ns=1))


def test_instrument_restores_on_failure_and_rejects_concurrency():
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime import benchmark_response_audit
    original = PrivateBoundaryTransport.linear
    forward = SemanticDecoderRuntime._forward
    audit = benchmark_response_audit.response_state_audit
    with pytest.raises(ValueError):
        with profile.instrument():
            assert PrivateBoundaryTransport.linear is not original
            with pytest.raises(RuntimeError, match="concurrent"):
                with profile.instrument():
                    pass
            raise ValueError()
    assert PrivateBoundaryTransport.linear is original
    assert SemanticDecoderRuntime._forward is forward
    assert benchmark_response_audit.response_state_audit is audit


def test_lookup_exchange_does_not_become_head_time(monkeypatch):
    from pllm.runtime.private_boundary_transport import PrivateBoundaryTransport
    monkeypatch.setattr(PrivateBoundaryTransport, "_exchange", lambda *a, **k: "unchanged")
    with profile.instrument() as record:
        assert PrivateBoundaryTransport._exchange(None, "lookup", [], []) == "unchanged"
        assert PrivateBoundaryTransport._exchange(None, "head", [], []) == "unchanged"
    assert [row["label"] for row in record.spans] == ["lookup_exchange", "paired_exchange"]


def test_matched_suite_keeps_native_audit_and_balanced_serial_contract():
    suite = profile.matched_suite(build=BuildRef(expected_digest="a" * 64))
    assert len(suite.cases) == 6
    assert suite.name == profile.IDENTITY
    assert suite.sampling.processes == 3
    assert suite.sampling.threads == 1
    for control, probe in zip(suite.cases[::2], suite.cases[1::2]):
        assert control.name.startswith("control--") and probe.name.startswith("probe--")
        assert control.target.to_spec() == probe.target.to_spec()
        assert control.target.to_spec()["parameters"]["backend"] == "native"
        assert control.target.to_spec()["parameters"]["state_audit"] is True


def test_worker_redirect_only_exact_worker_command(monkeypatch):
    from pllm.metrics import benchmark_runner as runner
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return "child"
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    def execute(*args):
        assert runner.subprocess.Popen(["python", "-I", "-m", "pllm._benchmark_worker"]) == "child"
        runner.subprocess.Popen(["python", "-m", "provider"])
        return "result"
    monkeypatch.setattr(runner, "_execute", execute)
    with profile.research_workers("/research/driver.py", "source-id"):
        assert runner._execute(SimpleNamespace(name="probe--maintained")) == "result"
    assert commands == [["python", "-I", "/research/driver.py", "--worker", "probe", "source-id"],
                        ["python", "-m", "provider"]]
    assert runner._execute is execute
    assert runner.subprocess.Popen is popen


@pytest.fixture
def profile_cohort(cohort):
    from pllm.metrics.benchmark_runner import _result
    from pllm._benchmark_evidence import measurement
    from pllm.runtime.topology_accounting import ACCOUNTING_SCHEMA, accounting_coverage_identity
    build = {"attributes": {"synthetic": True}, "digest": digest({"synthetic": True})}
    suite = profile.matched_suite(build=BuildRef(expected_digest=build["digest"]))
    result = {**cohort, "suite": suite.to_spec(), "results": []}
    for case in suite.cases:
        mode, name = case.name.split("--", 1)
        for old in cohort["results"]:
            if old["execution"]["case"]["name"] != name:
                continue
            execution = deepcopy(old["execution"])
            details = execution["details"]
            report = details["report"]
            report.update(schema_version="pllm.loopback_benchmark.v1", warmup_runs=[])
            report["configuration"].update(warmups=0, repetitions=1, source_lock_digest=digest("source"))
            report["runs"][0].update(status="completed", warm=False, model_fingerprint=digest("body"))
            report["runs"][0]["generation"]["output_text_digest"] = digest("output")
            report["runs"][0]["durations"] = {"full_seconds": 1, "generation_seconds": .5,
                "online_seconds": .8, "ttft_seconds": .1}
            details["topology_admission"] = {"admitted": True, "estimate": {"components": {
                "private_boundary_profile_bytes": profile.EXTRA_MEMORY, "response_state_audit_bytes": 32 << 20}}}
            if mode == "probe":
                recorder = profile.Recorder()
                for i in range(8):
                    start = 200_000_000 if i == 0 else 320_000_000 + (i - 1) * 60_000_000
                    def at(label, a, b):
                        return span(label, start + a * 1_000_000, start + b * 1_000_000)
                    recorder.spans += [at("forward", 0, 50), at("token_lookup", 1, 5),
                        at("audit_and_sampler", 52, 55), at("sampler", 53, 54)]
                    if name != "maintained":
                        recorder.spans += [at("head", 10, 40), at("paired_exchange", 15, 25),
                            at("head_quantization", 11, 12), at("head_dequantization", 30, 31)]
                        recorder.spans += [at(label, 19, 24) for label in ("http_exchange_a", "http_exchange_b")]
                        recorder.spans += [at(label, 16, 17) for label in (
                            "frame_pack", "frame_pack", "frame_unpack", "frame_unpack")]
                        if "uniform_residues" in name:
                            recorder.spans.append(at("head_codec_decode", 27, 28))
                recorder.windows = [[0, 1_000_000_000, 300_000_000, 800_000_000]]
                details["private_boundary_decode_profile"] = recorder.report()
            comparison = profile.ordinary_comparison(case.target, details)
            details["research_profile_identity"] = {"schema": profile.IDENTITY, "mode": mode, "sources": "c" * 64,
                "executor_client_process_cpu_ns": 1000, "response_workload_digest": comparison["workload_digest"]}
            comparison["workload_digest"] = digest({"synthetic_private_hmac": comparison["workload_digest"]})
            comparison["measurement_digest"] = digest({"ordinary": comparison["measurement_digest"],
                "research": profile.IDENTITY, "mode": mode, "sources": "c" * 64})
            values = {m["id"]: m["value"] for m in old["metrics"]}
            metrics = profile.required_metric_contracts(8)
            for key, metric in metrics.items():
                metric["value"] = values.get(key, 1)
            metrics["request.0.full"]["value"] = 1
            for key in ("response.ttft", "request.0.ttft"):
                metrics[key]["value"] = .1
            for key in ("response.decode_tps", "request.0.decode_tps"):
                metrics[key]["value"] = 14
            coverage = accounting_coverage_identity(report["topology_accounting"])
            scope = "covered_application_bodies" + (f":{coverage}" if coverage != ACCOUNTING_SCHEMA else "")
            metrics["network.covered_body_bytes"] = measurement("network.covered_body_bytes",
                values["network.covered_body_bytes"], kind="bytes", role="all-links", statistic="sum",
                window="setup_through_scenario", scope=scope)
            payload = {"status": "completed", "failure_code": None, "details": details,
                "comparison": comparison, "metrics": list(metrics.values()),
                "samples": [{"index": 0, "kind": "measurement", "status": "completed", "duration_seconds": 1,
                    "artifact_digest": None}], "fixture": {"kind": "checkpoint", "digest": digest("source"),
                    "metadata": {"cached_checkpoint": True, "owned_client_cache": "fresh"}},
                "admission": {"admitted": True, "required_bytes": 128 << 20, "limit_bytes": 8 << 30,
                    "host_headroom_bytes": 8 << 30, "scope": "target_working_set", "reason": None},
                "oracle": {"kind": "salted_full_logit_kv_sampler_commitments", "outcome": "passed", "checked_values": 800},
                "environment": {"fixture": "synthetic"}}
            result["results"].append(_result(case, suite, execution["replicate"], result["cohort"], build, payload).to_dict())
    return result


def test_synthetic_cohort_requires_all_state_controls_and_reports_overhead(profile_cohort):
    result = profile.audit(profile_cohort, source_identity="c" * 64)
    assert result["clients"] == 18
    assert len(result["profiles"]) == 9
    assert result["all_logits_active_kv_positions_selections_rng_and_outputs_match"]
    assert result["overhead_pairs"]["maintained"][0]["metric_deltas_probe_minus_control"]["response.decode_tps"] == 0


@pytest.mark.parametrize("mutation", ["state", "source", "partition", "span", "missing", "window", "control", "owners", "sampler"])
def test_synthetic_rejects_incomplete_or_mixed_profiles(profile_cohort, mutation):
    row = profile_cohort["results"][-1]
    details = row["execution"]["details"]
    if mutation == "state":
        details["response_state_audit"]["bindings"][0]["selections"][0]["rng_after"] = "different"
    elif mutation == "source":
        details["research_profile_identity"]["sources"] = "different"
    elif mutation == "partition":
        details["private_boundary_decode_profile"]["request"]["exclusive_wall_ns"]["paired_exchange"] += 1
    elif mutation == "span":
        details["private_boundary_decode_profile"]["spans"] = []
    elif mutation == "missing":
        profile_cohort["results"].pop()
    elif mutation == "window":
        details["report"]["runs"][0]["durations"]["generation_seconds"] = 1
    elif mutation == "control":
        profile_cohort["results"][0]["execution"]["details"]["private_boundary_decode_profile"] = {}
    elif mutation == "owners":
        details["response_state_audit"]["client_memory_owners"].pop()
    else:
        details["response_state_audit"]["bindings"][0]["cryptographic_randomness_frozen"] = True
    with pytest.raises(Rejected):
        profile.audit(profile_cohort, source_identity="c" * 64)


def test_worker_bootstrap_uses_original_executor_and_separate_identity(monkeypatch):
    from pllm import metrics
    from pllm.runtime import benchmark_targets
    target = profile.matched_suite(build=BuildRef(expected_digest="a" * 64)).cases[0].target
    calls = []
    def execute(target, measurement):
        calls.append(target)
        measurement.comparison["measurement_digest"] = "ordinary-digest"
        measurement.comparison["workload_digest"] = "workload-digest"
    monkeypatch.setattr(benchmark_targets, "execute_target", execute)
    original_memory = metrics.benchmark_memory
    identities = []
    for mode in ("control", "probe"):
        measurement = SimpleNamespace(details={"report": {"runs": [{"durations": {
            "full_seconds": 1, "generation_seconds": .5}}]}}, comparison={}, limitations=[])
        recorder = profile.Recorder()
        recorder.windows = [[0, 1_000_000_000, 100_000_000, 600_000_000]]
        from contextlib import contextmanager
        @contextmanager
        def fake_instrument():
            yield recorder
        monkeypatch.setattr(profile, "instrument", fake_instrument)
        with profile.worker_bootstrap(mode, "c" * 64):
            benchmark_targets.execute_target(target, measurement)
        assert metrics.benchmark_memory is original_memory
        assert benchmark_targets.execute_target is execute
        assert ("private_boundary_decode_profile" in measurement.details) == (mode == "probe")
        assert measurement.details["research_profile_identity"]["mode"] == mode
        identities.append(measurement.comparison["measurement_digest"])
    assert calls == [target, target]
    assert identities[0] != identities[1]


def test_preflight_prices_both_arms_without_model_or_execution(monkeypatch):
    from benchmarks.research import private_boundary_decode as decode
    suite = profile.matched_suite(build=BuildRef(expected_digest="a" * 64))
    monkeypatch.setattr(decode, "preflight", lambda suite: [
        {"case": case.name, "admission": {"ordinary": True}} for case in suite.cases])
    calls = []
    def price(report, memory_bytes):
        calls.append((report, memory_bytes))
        return {"admitted": True, "profile_extra_bytes": profile.EXTRA_MEMORY}
    monkeypatch.setattr(profile, "priced_admission", price)
    rows = profile.preflight(suite)
    assert len(rows) == len(calls) == 6
    assert all(row["admission"]["profile_extra_bytes"] == 8 << 20 for row in rows)


@pytest.mark.parametrize("mutation", ["schema", "failure", "admission", "oracle", "warmup", "topology",
    "environment", "numeric", "state", "workload", "output", "measurement", "metric_window", "metric_count",
    "cpu_negative", "cpu_bool", "cpu_float", "request_shift", "decode_shift", "decode_negative", "sampler_owner",
    "codec", "pre_hmac"])
def test_hardened_provenance_and_window_rejections(profile_cohort, mutation):
    from pllm.evidence import environment_digest
    row = profile_cohort["results"][-1]
    execution = row["execution"]
    details = execution["details"]
    timing = details["private_boundary_decode_profile"]
    if mutation == "schema":
        row["schema_version"] = "pllm.benchmark_result.v1"
    elif mutation == "failure":
        execution["failure_code"] = "ignored_failure"
    elif mutation == "admission":
        execution["admission"]["admitted"] = False
    elif mutation == "oracle":
        execution["oracle"]["kind"] = "ordinary_role_runtime_checks"
    elif mutation == "warmup":
        details["report"]["configuration"]["warmups"] = 1
    elif mutation == "topology":
        details["topology_admission"]["admitted"] = False
    elif mutation == "environment":
        attributes = {"fixture": "other_machine"}
        row["environment"] = {"attributes": attributes, "digest": environment_digest(attributes)}
    elif mutation in {"numeric", "state", "output", "measurement"}:
        execution["comparison"][mutation + "_digest"] = "d" * 64
    elif mutation == "workload":
        execution["comparison"]["workload_digest"] = row["workload_digest"] = "d" * 64
    elif mutation == "pre_hmac":
        details["research_profile_identity"]["response_workload_digest"] = "d" * 64
    elif mutation in {"metric_window", "metric_count"}:
        for other in profile_cohort["results"]:
            if other["execution"]["case"]["name"].endswith("coupled-parallel-uniform_residues"):
                metric = next(m for m in other["metrics"] if m["id"] == "response.decode_tps")
                metric["window" if mutation == "metric_window" else "sample_count"] = "request" if mutation == "metric_window" else 2
    elif mutation.startswith("cpu_"):
        details["research_profile_identity"]["executor_client_process_cpu_ns"] = {
            "cpu_negative": -1, "cpu_bool": True, "cpu_float": 1.25}[mutation]
    elif mutation == "request_shift":
        timing["windows_relative_ns"]["request"] = [1, 1_000_000_001]
    elif mutation == "decode_shift":
        timing["windows_relative_ns"]["after_first_output"] = [310_000_000, 810_000_000]
    elif mutation == "decode_negative":
        timing["windows_relative_ns"]["after_first_output"] = [-1, 499_999_999]
    elif mutation == "sampler_owner":
        sample = next(s for s in timing["spans"] if s["label"] == "sampler")
        sample.update(start_ns=1, end_ns=2)
    else:
        timing["spans"].pop(next(i for i, s in enumerate(timing["spans"]) if s["label"] == "head_codec_decode"))
    # Recompute summaries so shifted-window/nesting tests cannot pass merely by
    # relying on the cached-partition consistency gate.
    for window, bounds in timing["windows_relative_ns"].items():
        timing[window] = profile.summarize(timing["spans"], bounds)
    with pytest.raises(Rejected):
        profile.audit(profile_cohort, source_identity="c" * 64)


@pytest.mark.parametrize("identity", ["source-id", "a" * 63, "A" * 64, "g" * 64, None])
def test_source_identity_requires_sha256(profile_cohort, identity):
    with pytest.raises(Rejected):
        profile.audit(profile_cohort, source_identity=identity)


def test_output_prefix_exclusively_reserved_and_never_overwritten(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from scripts.profile_private_boundary_decode import reserve_output_paths
    output = tmp_path / "cohort.json"
    def reserve():
        try:
            reserve_output_paths(output)
            return True
        except FileExistsError:
            return False
    with ThreadPoolExecutor(max_workers=2) as owner:
        assert sorted(owner.map(lambda _: reserve(), range(2))) == [False, True]
    output.write_text("original evidence")
    with pytest.raises(FileExistsError):
        reserve_output_paths(output)
    assert output.read_text() == "original evidence"


def test_tiny_preflight_does_not_create_checkpoint_or_run_model(monkeypatch, tmp_path):
    from contextlib import nullcontext
    import filelock
    from scripts import profile_private_boundary_decode as driver
    from benchmarks.research import sota_combinations
    from pllm.runtime import tiny_llama
    from pllm.metrics import BuildRef
    def forbidden(*args, **kwargs):
        raise AssertionError("tensor creation/model execution/build inspection forbidden")
    monkeypatch.setattr(tiny_llama, "create_tiny_llama_checkpoint", forbidden)
    # Synthetic CLI test must not contend with parent's real model-job lock.
    monkeypatch.setattr(filelock, "FileLock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(BuildRef, "capture", classmethod(lambda cls: BuildRef(expected_digest="a" * 64)))
    monkeypatch.setattr(sota_combinations, "public_input_counts", lambda *a: ({"tiny": 100}, None))
    monkeypatch.setattr(profile, "preflight", lambda suite: [{"case": c.name, "admission": {"admitted": True}}
        for c in suite.cases])
    output = tmp_path / "metadata.json"
    assert driver.main(["--output", str(output), "--tiny", "--preflight-only"]) == 0
    plan = json.loads(output.with_suffix(".plan.json").read_text())
    assert len(plan["suite"]["cases"]) == 6
    from pllm import Model
    assert all(Model.from_spec(c["target"]["parameters"]["model"]["experiment"]["pipeline"]["model"]).kind == "tiny"
        for c in plan["suite"]["cases"])
    assert not output.with_suffix(".model").exists()
    assert not output.exists()


def head_order_spans(codec):
    rows = [span("forward", 0, 100), span("token_lookup", 2, 5), span("head", 10, 70),
        span("head_quantization", 15, 20), span("paired_exchange", 25, 45),
        span("head_dequantization", 60, 65), span("audit_and_sampler", 110, 120), span("sampler", 112, 118),
        span("http_exchange_a", 28, 44), span("http_exchange_b", 30, 43)]
    if codec:
        rows.append(span("head_codec_decode", 50, 55))
    return rows


@pytest.mark.parametrize("codec", [False, True])
def test_head_synchronous_order_preserves_parallel_http_and_residual(codec):
    rows = head_order_spans(codec)
    profile._validate_owner_sequence(rows, 1, True, 130, 140, 150)
    result = profile.summarize(rows, (0, 150))
    assert result["exclusive_wall_ns"]["paired_exchange"] == 20
    assert result["diagnostic_nonadditive"]["http_exchange"]["union_wall_ns"] == 16
    assert result["exclusive_wall_ns"]["head_unclassified"] == (25 if codec else 30)
    assert result["partition_reconciled"]


@pytest.mark.parametrize("codec", [False, True])
@pytest.mark.parametrize("mutation", ["swapped_quantizer_dequantizer", "overlapping_quantizer_exchange",
    "overlapping_exchange_next_child", "reordered_exchange_dequantizer", "overlapping_dequantizer_previous_child"])
def test_head_synchronous_order_rejects_reordered_or_overlapping_children(codec, mutation):
    rows = head_order_spans(codec)
    children = {row["label"]: row for row in rows}
    quantizer, exchange, dequantizer = [children[label] for label in (
        "head_quantization", "paired_exchange", "head_dequantization")]
    if mutation == "swapped_quantizer_dequantizer":
        quantizer.update(start_ns=60, end_ns=65)
        dequantizer.update(start_ns=15, end_ns=20)
    elif mutation == "overlapping_quantizer_exchange":
        quantizer["end_ns"] = 26
    elif mutation == "overlapping_exchange_next_child":
        exchange["end_ns"] = 51 if codec else 61
    elif mutation == "reordered_exchange_dequantizer":
        exchange.update(start_ns=60, end_ns=65)
        dequantizer.update(start_ns=25, end_ns=45)
    else:
        dequantizer["start_ns"] = 54 if codec else 44
    # Recomputed partitions still reconcile; consistency alone cannot establish
    # original synchronous owner order.
    assert profile.summarize(rows, (0, 150))["partition_reconciled"]
    with pytest.raises(Rejected, match="profile_head_child_order_invalid"):
        profile._validate_owner_sequence(rows, 1, True, 130, 140, 150)
