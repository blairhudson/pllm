"""Ordinary benchmark integration with explicit metadata/source/native doubles.

No real wheel acceptance, role launch, model execution or performance evidence.
"""
import copy
from contextlib import closing
import json
from types import SimpleNamespace as NS

import pytest

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import FuseFssSourceCpu, resolve_runtime_composition
from pllm.runtime import benchmark_delegated as delegated
from pllm.runtime import benchmark_memory as memory
from pllm.runtime import fss_source
from pllm.runtime.fss_protocol import ROLES, PEER_COUNTERS, CLIENT_MEMORY, ROLE_MEMORY

PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    value = Experiment("fss-benchmark-double", FuseFssSourceCpu(
        Model.path(str(tmp_path / "source")), numeric_profile=PROFILE),
        Deployment.local(root=str(tmp_path / "roles")), ExecutionBudget(2, 2, 3))
    monkeypatch.setattr(Experiment, "resolve", lambda self: resolve_runtime_composition(self.pipeline))
    return value


@pytest.fixture
def contract():
    return {"schema_version": "pllm.fss.sdk-contract.v1", "runtime_kind": "fss",
        "client_runtime": "compiled_fss_v1", "numeric_profile": PROFILE,
        "ring_bits": 50, "fractional_bits": 12, "input_rows": 2, "output_tokens": 3,
        "maximum_sequence": 4, "plan": {"config_digest": "a" * 64},
        "source_shapes": {"embedding": [7, 8]}, "source_layouts": {"embedding": "rows_features"},
        "source_float32_bytes": 224, "source_q12_value_bytes": 448, "rotary_float32_bytes": 0,
        "vocabulary": 7, "width": 8, "admission": {"geometry": "bounded", "numeric": "not_performed",
            "source_minimum_owner_bytes": 1024, "program_resources": None, "serving_qualified": False}}


@pytest.fixture
def source(contract, monkeypatch):
    value = NS(config_bytes=b'{"model_type":"gpt2"}', contract=contract,
        contract_bytes=json.dumps(contract).encode(), checkpoint_digest="1" * 64,
        source_lock_digest="2" * 64, tokenizer_digest="3" * 64,
        snapshot_pins={"checkpoint": "1" * 64, "source_lock": "2" * 64, "tokenizer": "3" * 64, "config": "4" * 64})
    monkeypatch.setattr(fss_source, "resolve_client_source", lambda exp: value)
    return value


def counters():
    result = {f"role_link.{role}.{phase}_{direction}_bytes": i + 1
              for i, (role, phase, direction) in enumerate((r, p, d) for r in ROLES
                  for p in ("setup", "online", "teardown") for d in ("upload", "download"))}
    result.update({key: index + 10 for index, key in enumerate(sorted(PEER_COUNTERS))})
    result.update({f"fss_storage.{role}.{op}_bytes": 999 for role in ROLES[:2] for op in ("read", "write")})
    return result


def test_runtime_recognition_and_nonstream_contract(experiment):
    options = resolve_runtime_composition(experiment.pipeline)
    assert options.client_runtime == "compiled_fss_v1" and options.privacy_mode == "fss_public"
    assert delegated.is_fss(experiment) and delegated.is_buffered(experiment)
    assert not delegated.is_delegated(experiment) and not delegated.is_replicated(experiment)
    assert delegated.output_contract(experiment) == ("buffered_final_ids", "final_only_private_token_disclosure")
    delegated.admit_options(experiment, output_tokens=3)


@pytest.mark.parametrize("kwargs", [{"backend": "docker"}, {"cache_bytes": 1}, {"temperature": 1},
                                   {"inventory_policy": "request-sized"}, {"output_tokens": 4}])
def test_unsupported_options_reject_before_source(experiment, monkeypatch, kwargs):
    monkeypatch.setattr(fss_source, "resolve_client_source", lambda *a: pytest.fail("source inspected"))
    with pytest.raises(ValueError):
        delegated.admit_options(experiment, **({"output_tokens": 3} | kwargs))


def test_metadata_not_numeric_or_complete_program_admission(experiment, source):
    metadata, contract = delegated.inspect_runtime(experiment)
    assert contract is source.contract
    assert metadata["numeric"]["numeric_profile"] == PROFILE
    assert metadata["preparation_roles"] == ["dealer"]
    assert metadata["execution_payload_estimate_bytes"] is None
    assert metadata["complete_program_admitted"] is False
    assert metadata["native_runtime_contract_digest"] is None and metadata["native_program_digest"] is None
    assert len(metadata["source_contract_bytes_sha256"]) == 64
    assert metadata["decode_body_scope"] == "unmeasured" and not metadata["decode_tps_supported"]
    assert metadata["ttft_scope"] == "sdk_call_to_final_disclosure_including_preprocessing"
    assert metadata["admission_scope"].startswith("metadata_preflight_only;")
    assert not metadata["cross_numeric_rankings_supported"]
    assert not metadata["source_checkpoint_delivery_measured"]


def test_all_body_links_conserve_and_disk_is_separate():
    counts = counters()
    ledger = delegated.fss_body_accounting({"privacy": counts,
        "processes": {r: {"cpu_seconds": 1} for r in ("client", *ROLES)}})
    expected = sum(v for key, v in counts.items() if key.startswith(("role_link.", "peer_link.")))
    assert ledger["all_link_serialized_body_bytes"] == expected
    assert sum(e["serialized_body_bytes"] for e in ledger["body_bytes_by_edge"]) == expected
    assert ledger["preparation_roles"] == ["dealer"] and not ledger["preparation_inventory"]
    assert set(ledger["run_window_cpu_seconds_by_role"]) == {"client", *ROLES}
    assert sum(ledger["storage_bytes"].values()) == 4 * 999
    assert ledger["decode_online_body_bytes"] is None
    assert ledger["roles"] == ["client", *ROLES]


def test_ordinary_report_and_history_preserve_dealer_and_final_disclosure(experiment, source):
    from pllm.runtime.benchmark_cli import build_loopback_report
    from pllm.runtime.benchmark_history import BenchmarkHistory, BenchmarkRun
    from pllm.runtime.topology_accounting import accounting_coverage_identity, FSS_COVERAGE

    privacy = counters() | {"plaintext_prompt_bytes_sent": 0, "plaintext_token_ids_sent": 0,
                            "preparation_requests_during_online": 0}
    run = BenchmarkRun(run_id="fss_report", status="completed", model_id="fss-unit",
        model_fingerprint="a" * 64, max_output_tokens=3, cold=True, warm=False,
        started_at_ns=1_000_000_000, finished_at_ns=4_000_000_000,
        online_started_at_ns=1_000_000_000, first_token_at_ns=3_000_000_000,
        last_token_at_ns=3_000_001_000, input_tokens=2, output_tokens=3,
        token_usage_authoritative=True, privacy_delta=privacy,
        process_metrics={role: {"cpu_seconds": 1, "rss_peak_bytes": 128} for role in ("client", *ROLES)},
        output_delivery="buffered_final_ids")
    with closing(BenchmarkHistory(":memory:")) as history:
        history.add(run)
        saved = history.get("fss_report")
    assert saved["privacy"] == privacy
    assert saved["processes"]["dealer"]["cpu_seconds"] == 1
    metadata, _ = delegated.inspect_runtime(experiment)
    report = build_loopback_report(model_id="fss-unit", tiny=False, max_output_tokens=3,
        runs=[saved], warmup_runs=[], roles=("client", *ROLES), runtime_contract=metadata,
        initial_preparation_audit=dict.fromkeys(privacy, 0), inventory_policy="none", temperature=0)
    assert report["checks"]["passed"], report["checks"]
    assert report["checks"]["fixed_count_final_ids_delivery"]
    assert report["configuration"]["preparation_roles"] == ["dealer"]
    assert not report["privacy_admission"]["independent_operators_verified"]
    assert accounting_coverage_identity(report["topology_accounting"]) == FSS_COVERAGE
    ledger = report["topology_accounting"]["runs"][0]
    expected = sum(value for key, value in privacy.items() if key.startswith(("role_link.", "peer_link.")))
    assert ledger["all_link_serialized_body_bytes"] == expected
    assert ledger["aggregate_run_window_cpu_seconds"] == 4
    assert sum(ledger["storage_bytes"].values()) == 4 * 999
    assert report["summary"]["total_accounted_benchmark_body_bytes"] == expected
    assert report["summary"]["median_ttft_seconds"] is None
    assert report["summary"]["median_final_disclosure_seconds"] == 2
    assert report["summary"]["median_online_seconds"] is None
    durations = report["runs"][0]["durations"]
    assert durations["final_disclosure_seconds"] == 2
    assert durations["ttft_seconds"] is durations["online_seconds"] is None
    assert saved["durations"]["ttft_seconds"] == 2  # Report conversion leaves history alone.
    assert report["summary"]["median_tokens_per_second"] is None
    ledger["tracked_body_counter_set_present"] = False
    assert accounting_coverage_identity(report["topology_accounting"]) is None


@pytest.mark.parametrize("bad", [None, True, -1, 1.0])
def test_missing_or_invalid_peer_link_never_becomes_free(bad):
    values = counters()
    values[next(iter(PEER_COUNTERS))] = bad
    result = delegated.fss_body_accounting({"privacy": values})
    assert not result["tracked_body_counter_set_present"]
    assert result["all_link_serialized_body_bytes"] is None


def test_memory_uses_actual_source_api_and_full_saved_envelope(experiment, source, monkeypatch):
    from pllm import model_loader, modeling
    monkeypatch.setattr(model_loader, "resolve_model", lambda *a: pytest.fail("generic source loader"))
    monkeypatch.setattr(modeling, "lower_model", lambda *a, **k: pytest.fail("generic lowering"))
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(8 << 30, 7 << 30, 0, 20 << 30))
    report = memory.benchmark_memory(experiment, max_input_tokens=1, max_output_tokens=1)
    assert report["admitted"] and report["selected_backend"] == "native"
    estimate = report["estimate"]
    assert estimate["max_input_tokens"] == 2 and estimate["max_output_tokens"] == 3
    assert set(estimate["provider_peak_bytes"]) == set(ROLES)
    assert estimate["native_total_peak_bytes"] == estimate["client_peak_bytes"] + sum(estimate["provider_peak_bytes"].values())
    assert estimate["allocation_slack_percent"] == 25
    assert sum(estimate["allocation_owners"]["client"].values()) <= CLIENT_MEMORY
    assert sum(estimate["allocation_owners"]["each_role"].values()) <= ROLE_MEMORY
    assert estimate["fss_admission"]["source_pins_verified"] is True
    assert estimate["fss_admission"]["complete_program_resources"] is None
    assert not report["candidates"]["docker"]["admitted"]
    assert report["candidates"]["native"]["required_disk_bytes"] >= 4 << 30


def test_source_failure_blocks_before_generic_model_loading(experiment, monkeypatch):
    from pllm import model_loader, modeling
    monkeypatch.setattr(memory, "host_memory", lambda: memory.HostMemory(8 << 30, 7 << 30, 0, 20 << 30))
    def reject(*args):
        raise ValueError("full checkpoint exceeds native 8MiB geometry")
    monkeypatch.setattr(fss_source, "resolve_client_source", reject)
    monkeypatch.setattr(model_loader, "resolve_model", lambda *a: pytest.fail("generic source loader"))
    monkeypatch.setattr(modeling, "lower_model", lambda *a, **k: pytest.fail("generic lowering"))
    with pytest.raises(memory.BenchmarkMemoryError, match="before model loading"):
        memory.benchmark_memory(experiment)


def test_bare_metadata_estimate_uses_geometry_first_unverified_pins(experiment, contract, monkeypatch):
    calls = []
    monkeypatch.setattr(fss_source, "_geometry", lambda *args: calls.append("geometry"))
    def native_double(exp, encoded, config, rows, outputs, sourcepin, tokenizerpin):
        assert calls == ["geometry"]
        assert sourcepin == "1" * 64 and tokenizerpin == "2" * 64
        return json.dumps(contract).encode(), contract
    monkeypatch.setattr(fss_source, "_contract", native_double)
    result = memory.estimate_memory({"model_type": "gpt2"}, experiment.pipeline,
        max_input_tokens=2, max_output_tokens=3)
    assert result["fss_admission"]["synthetic_nonzero_pins_used"]
    assert not result["fss_admission"]["source_pins_verified"]
    assert not result["fss_admission"]["serving_qualified"]


def test_conservative_owners_reject_without_raising_native_caps(experiment, contract):
    too_large = copy.deepcopy(contract)
    too_large["source_float32_bytes"] = 8 << 20
    with pytest.raises(memory.BenchmarkMemoryError, match="incremental owners"):
        memory._fss_memory({}, experiment.pipeline, 2, 3, 0, None, 0, contract=too_large)
    assert CLIENT_MEMORY == 64 << 20 and ROLE_MEMORY == 256 << 20


def test_native_binding_presence_does_not_qualify_execution():
    from pllm import _native
    if not all(hasattr(_native, name) for name in ("FssSourceBuilder", "fss_source_contract")):
        pytest.skip("actual FSS native wheel unavailable; integration above uses explicit doubles")
    # Presence alone does not qualify roles, complete source admission or execution.
    assert callable(_native.fss_source_contract)


def test_ordinary_response_target_reaches_shared_memory_gate(experiment, monkeypatch, tmp_path):
    from pllm.metrics import BenchmarkTarget, Scenario
    from pllm._benchmark_contract import TargetUnavailable
    from pllm.runtime.benchmark_targets import execute_target
    calls = []
    def admission(exp, **kwargs):
        calls.append(exp)
        assert delegated.is_fss(exp)
        return {"admitted": False}
    monkeypatch.setattr("pllm.metrics.benchmark_memory", admission)
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3))
    meter = NS(inputs={"prompt": "hello"}, details={}, directory=tmp_path,
               sampling=NS(repetitions=1, warmups=0, memory_bytes=8 << 30))
    with pytest.raises(TargetUnavailable, match="topology_memory_limit"):
        execute_target(target, meter)
    assert len(calls) == 1


def test_ordinary_geometry_target_checks_metadata_not_numeric_admission(experiment, source):
    from pllm.metrics import BenchmarkTarget
    from pllm.runtime.benchmark_targets import execute_target
    meter = NS(inputs={}, details={}, comparison={}, admit=lambda value: None,
               run=lambda function: [function(0)])
    execute_target(BenchmarkTarget.geometry(experiment), meter)
    assert meter.oracle == {"kind": "native_fss_metadata_geometry_contract", "outcome": "passed", "checked_values": 1}
    assert meter.fixture["metadata"]["values_loaded"] is False
    assert meter.details["runtime_contract"]["complete_program_admitted"] is False
    assert meter.details["plan_digest"] is None


def test_geometry_source_metadata_rejection_is_blocked_not_generic_fallback(experiment, monkeypatch):
    from pllm.metrics import BenchmarkTarget
    from pllm._benchmark_contract import TargetUnavailable
    from pllm.runtime.benchmark_targets import execute_target
    def reject(exp):
        raise ValueError("checkpoint geometry exceeds existing source cap")
    monkeypatch.setattr(fss_source, "resolve_client_source", reject)
    meter = NS(inputs={}, details={}, comparison={}, admit=lambda value: None,
               run=lambda function: [function(0)])
    with pytest.raises(TargetUnavailable, match="fss_source_metadata_admission"):
        execute_target(BenchmarkTarget.geometry(experiment), meter)


def test_imported_program_pricing_only_when_native_owner_supplied(contract, monkeypatch):
    from pllm import _native
    owner = object()
    loaded = NS(native_program=owner, contract_bytes=json.dumps(contract).encode())
    pricing = {"schema": "pllm.fss.program_transport_admission.v1",
        "scope": "program-pricing-not-request-admission", "numeric_profile": PROFILE,
        "initial_tokens": 2, "selections": 3, "vocabulary": 7, "largest_packet_bytes": 1024,
        "cumulative_packet_bytes_per_party_bound": 4096, "combined_reservation_bytes": 2048,
        "client_peak_bytes": 128, "peer_frame_bytes_bound": 1024, "kernels": 4,
        "primitive_requests_per_party": 4, "callback_metadata_bytes_bound": 1024,
        "input_frame_bytes": 128, "output_frame_bytes": 128}
    def native_double(program, raw):
        assert program is owner and raw == loaded.contract_bytes
        return json.dumps(pricing).encode()
    monkeypatch.setattr(_native, "fss_transport_admission", native_double, raising=False)
    assert delegated.inspect_fss_imported(loaded) == pricing
    pricing["largest_packet_bytes"] = (16 << 20) + 1
    with pytest.raises(ValueError, match="transport bounds"):
        delegated.inspect_fss_imported(loaded)
    pricing["largest_packet_bytes"] = 1024
    pricing["primitive_requests_per_party"] = 4097
    with pytest.raises(ValueError, match="limits"):
        delegated.inspect_fss_imported(loaded)


@pytest.mark.parametrize("kind,roles", [("fss", ROLES), ("gkm", ROLES[:2]),
                                      ("replicated", ("worker_a", "worker_b", "worker_c"))])
def test_response_metrics_final_disclosure_and_old_buffered_behavior(kind, roles, monkeypatch):
    from pllm.runtime.benchmark_response_target import response_metrics
    from pllm.runtime import benchmark_communication
    # Attribution double isolates metric semantics, not any measured network result.
    attribution = {"schema": "unit-attribution-double", "summary": {
        "base_all_link_mb_per_request": None, "incremental_all_link_mb_per_generated_output": None,
        "measured_requests": 1, "generated_output_tokens": 3},
        "covered_lifecycle": {"unattributed_all_link_body_bytes": None}}
    monkeypatch.setattr(benchmark_communication, "benchmark_traffic_decomposition", lambda report: attribution)
    report = {"runs": [{"durations": {"full_seconds": 3, "ttft_seconds": 2, "tokens_per_second": 999},
                        "tokens": {"output_tokens": 3}, "privacy": counters(),
                        "processes": {r: {"cpu_seconds": 1, "rss_peak_bytes": 123} for r in ("client", *roles)}}],
              "runtime_contract": {"runtime_kind": kind}, "configuration": {"roles": ("client", *roles)}}
    measured = {}
    meter = NS(details={}, metric=lambda name, value, **kwargs: measured.__setitem__(name, (value, kwargs)))
    response_metrics(report, meter)
    assert measured["response.decode_tps"][0] is None
    assert measured["response.ttft"][0] == (None if kind == "fss" else 2)
    if kind == "fss":
        assert measured["response.final_disclosure"][0] == 2
        assert measured["response.final_disclosure"][1]["window"] == "sdk_call_to_final_disclosure_including_preprocessing"
        assert measured["network.covered_body_bytes"][0] is None  # No complete startup ledger supplied.
        assert measured["network.fss_request_bodies"][0] == delegated.fss_body_accounting({"privacy": counters()})["all_link_serialized_body_bytes"]
        assert measured["fss_storage.worker_a.read_bytes"][0] == 999
        assert meter.details["fss_measurement_scope"]["decode_body_partition"] is None
        assert all(f"{role}.cold_cpu" in measured and f"{role}.peak_rss" in measured for role in ROLES)
        assert "preparation.peak_rss" not in measured
