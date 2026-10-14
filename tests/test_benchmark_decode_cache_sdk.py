"""Reject inferred savings, incomplete cohorts, private leaks and mismatched states."""
from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest


@pytest.fixture
def driver(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("decode_cache_driver", scripts / "benchmark_decode_cache_sdk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEOMETRY = [{"stage": "body", "macs_per_row": 100, "remote": True, "terminal_prefill": False},
            {"stage": "terminal", "macs_per_row": 10, "remote": True, "terminal_prefill": True}]


def request(driver, *, index=0, warm=False, scenario="same_seed"):
    from pllm.runtime.topology_accounting import ACK_COVERAGE
    counts = {"requested_prefill_tokens": 4, "requested_decode_steps": 7,
              "executed_prefill_tokens": 0 if warm else 4,
              "executed_terminal_prefill_rows": 0 if warm else 1,
              "executed_decode_steps": 0 if warm else 7}
    delta = {driver.WORK_COUNTERS[k]: v for k, v in counts.items()}
    delta.update({name: 0 for name in driver.CACHE_COUNTERS.values()})
    if warm:
        delta.update({driver.CACHE_COUNTERS[k]: v for k, v in {
            "root_hits": 1, "branch_hits": 7, "skipped_remote_rows": 15,
            "skipped_prefill_tokens": 4, "skipped_decode_steps": 7, "warm_no_preparation": 1}.items()})
    delta.update({key: 0 if warm else 19 for key in ("prepared_stage_rows_issued",
        "prepared_stage_rows_reserved", "preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes")})
    delta["prepared_stage_rows_claimed"] = 0 if warm else 19
    return {"index": index, "harness_numpy_seed": driver.SCENARIOS[scenario][index], "status": "incomplete",
        "input_tokens": 4, "output_tokens": 8, "first_text_seconds": .1, "decode_tps": 7.,
        "request_seconds": 1.1, "request_cpu_seconds": .2,
        "state": {"bits": "salted", "tokens": "salted", "pending": "salted", "output": "salted",
            "history": "salted", "rendered_history": "salted", "position": 11, "pending_count": 1,
            "rng": {"initial": "a" * 64, "final": "b" * 64, "trajectory": "c" * 64,
                    "samples": 8, "runtimes": 1}},
        "audit_delta": delta,
        "request_ledger": {"tracked_body_counter_set_present": True, "body_counter_coverage": ACK_COVERAGE},
        "disk_io": {key: 1 for key in driver.STORAGE_COUNTERS}}


def result(driver, policy="ordinary", scenario="same_seed", replicate=0):
    return {"workload": scenario, "variant": policy, "replicate": replicate, "client": {
        "failure": None, "audit": {}, "client_peak_rss_bytes": 128 << 20,
        "responses": [request(driver, index=i, warm=policy == "decode" and i > 0, scenario=scenario) for i in range(3)],
        "storage_lifetime": {key: 10 for key in driver.STORAGE_FIELDS}}}


def geometries(driver):
    return {s + "/" + p: {"stages": GEOMETRY} for s in driver.SCENARIOS for p in driver.POLICIES}


def test_exact_frozen_alternating_cell_order(driver):
    keys = [s + "/" + p for s in driver.SCENARIOS for p in driver.POLICIES]
    ordered = driver.cohort_order(keys, 3)
    assert len(ordered) == 18 and len(set(ordered)) == 18
    assert [key for r, key in ordered if r == 0] == keys
    assert [key for r, key in ordered if r == 1] == keys[::-1]
    assert [key for r, key in ordered if r == 2] == keys
    assert driver.SCENARIOS == {"same_seed": (7, 7, 7), "distinct_seeds": (7, 11, 19)}


def test_requested_equivalent_and_actual_work_have_distinct_denominators(driver):
    row = {"input_tokens": 4, "output_tokens": 8, "status": "incomplete",
        "state": {"position": 11, "pending_count": 1}, "audit_delta": {
        driver.WORK_COUNTERS[k]: v for k, v in {
            "requested_prefill_tokens": 4, "requested_decode_steps": 7,
            "executed_prefill_tokens": 0, "executed_terminal_prefill_rows": 0,
            "executed_decode_steps": 3}.items()}}
    work = driver.measured_work(GEOMETRY, row)
    assert work["requested_equivalent_body_macs"] == 1180
    assert work["actually_executed_body_macs"] == 330
    assert work["eliminated_body_macs"] == 850
    assert work["remote_fraction_of_executed_body_macs"] == 1
    assert work["remote_fraction_of_requested_equivalent_body_macs"] < .9
    assert work["remote_90_percent_verified"] and work["whole_body_compute_fraction"] is None
    row["audit_delta"][driver.WORK_COUNTERS["executed_decode_steps"]] = 0
    zero = driver.measured_work(GEOMETRY, row)
    assert zero["zero_executed_work"] and zero["remote_90_percent_verified"] is None


def test_missing_actual_work_never_becomes_zero_or_hit_estimate(driver):
    row = {"input_tokens": 4, "output_tokens": 8,
           "audit_delta": {driver.CACHE_COUNTERS["branch_hits"]: 7}}
    work = driver.measured_work(GEOMETRY, row)
    assert not work["available"] and work["actually_executed_body_macs"] is None
    assert work["remote_90_percent_verified"] is None


@pytest.mark.parametrize("suffix", [1, 2, 3])
def test_resident_prefix_continuation_charges_all_terminal_suffix_rows(driver, suffix):
    row = request(driver)
    row["audit_delta"].update({
        driver.WORK_COUNTERS["executed_prefill_tokens"]: suffix,
        driver.WORK_COUNTERS["executed_terminal_prefill_rows"]: suffix,
        "prepared_stage_rows_claimed": 2 * (suffix + 7),
    })
    work = driver.validate_request(row, "ordinary", GEOMETRY)
    assert work["requested_equivalent_body_macs"] == 1180
    assert work["actually_executed_body_macs"] == 110 * (suffix + 7)
    assert work["actual_remote_stage_rows"] == 2 * (suffix + 7)
    row["audit_delta"][driver.WORK_COUNTERS["executed_terminal_prefill_rows"]] = 0
    with pytest.raises(RuntimeError, match="semantic work conservation"):
        driver.measured_work(GEOMETRY, row)


def test_fully_warm_hits_require_zero_fresh_inventory_and_semantic_work(driver):
    row = request(driver, warm=True)
    assert driver.validate_request(row, "decode", GEOMETRY)["zero_executed_work"]
    row["audit_delta"]["prepared_stage_rows_issued"] = 1
    with pytest.raises(RuntimeError, match="minted fresh preparation"):
        driver.validate_request(row, "decode", GEOMETRY)


def test_exact_states_and_largest_peak_per_policy_not_median(driver, monkeypatch):
    monkeypatch.setattr(driver, "validate_audit", lambda a: {})
    rows = [result(driver), result(driver, "fixed1"), result(driver, "decode")]
    rows[-1]["client"]["client_peak_rss_bytes"] = driver.LOW_BYTES + 1
    checks = driver.verify_results(rows, geometries(driver))
    assert checks["checked_requests"] == 9
    assert not checks["low_gate_by_policy"]["decode"] and checks["low_gate_by_policy"]["fixed1"]
    rows[-1]["client"]["responses"][2]["state"]["pending"] = "different"
    with pytest.raises(RuntimeError, match="parity"):
        driver.verify_results(rows, geometries(driver))


def test_cold_publication_and_complete_disk_io_cannot_be_omitted(driver, monkeypatch):
    monkeypatch.setattr(driver, "validate_audit", lambda a: {})
    cell = result(driver, "decode")
    cell["client"]["storage_lifetime"]["written_bytes"] = 0
    with pytest.raises(RuntimeError, match="cold encrypted publication"):
        driver.verify_results([cell], geometries(driver))
    cell = result(driver, "decode")
    cell["client"]["storage_lifetime"]["read_bytes"] = 2
    with pytest.raises(RuntimeError, match="disk counters"):
        driver.verify_results([cell], geometries(driver))


def test_same_seed_must_warm_but_distinct_seeds_do_not_force_divergence(driver, monkeypatch):
    monkeypatch.setattr(driver, "validate_audit", lambda a: {})
    cell = result(driver, "decode")
    cell["client"]["responses"][1] = request(driver, index=1)
    with pytest.raises(RuntimeError, match="warm root/branch path"):
        driver.verify_results([cell], geometries(driver))
    distinct = result(driver, "ordinary", "distinct_seeds")
    checks = driver.verify_results([distinct], geometries(driver))
    assert checks["sampler_variation"]["distinct_seeds"][0]["distinct_output_commitments"] == 1


def test_duplicate_and_missing_cells_reject_confirmation(driver, monkeypatch):
    monkeypatch.setattr(driver, "validate_audit", lambda a: {})
    cell = result(driver)
    with pytest.raises(RuntimeError, match="duplicate"):
        driver.verify_results([cell, copy.deepcopy(cell)], geometries(driver))
    with pytest.raises(RuntimeError, match="coverage incomplete"):
        driver.verify_results([cell], geometries(driver), require_complete=True)


def test_harness_rng_fixture_and_sampling_metrics_no_seed_sdk_kwarg(driver, monkeypatch):
    import numpy as np
    class FixtureRuntime:
        def __init__(self):
            self.rng = np.random.default_rng(42)
        def sample(self):
            return int(self.rng.integers(100))
    shim = driver.harness_sampler_seed
    monkeypatch.setattr(driver, "harness_sampler_seed", lambda seed, **kwargs: shim(seed, runtime_class=FixtureRuntime, **kwargs))
    response = {"id": "private-id", "model": "tiny", "status": "completed", "output": [],
                "usage": {"input_tokens": 4, "output_tokens": 1, "total_tokens": 5}}
    events = [SimpleNamespace(type="response.output_text.delta"),
              SimpleNamespace(type="response.completed", response=response)]
    calls, draws = [], []
    def create(**kwargs):
        calls.append(kwargs)
        runtime = FixtureRuntime.__new__(FixtureRuntime)  # Constructor hook cannot seed this actual sampler.
        draws.append([runtime.sample() for _ in range(response["usage"]["output_tokens"] + 1)])
        return nullcontext(iter(events))
    client = SimpleNamespace(tokenizer_scope=lambda: nullcontext(), responses=SimpleNamespace(create=create))
    monkeypatch.setattr(driver, "complete_state_commitment", lambda *a: {"output": "salted"})
    first = driver.sampling_response(client, "private prompt", 8, 7, b"x" * 32, 0)
    driver.sampling_response(client, "private prompt", 8, 7, b"x" * 32, 1)
    assert draws[0] == draws[1]
    assert calls[0]["temperature"] == .7 and calls[0]["top_p"] == .9 and "seed" not in calls[0]
    assert first["decode_tps"] is None and first["request_cpu_seconds"] >= 0
    assert first["first_text_seconds"] <= first["request_seconds"]
    assert "private" not in json.dumps(first)
    events.pop(0)
    response["usage"]["output_tokens"] = 0
    zero = driver.sampling_response(client, "private prompt", 8, 7, b"x" * 32, 2)
    assert zero["first_text_seconds"] is None and zero["decode_tps"] is None


def test_owner_already_priced_without_double_add_and_missing_bound_rejects(driver, monkeypatch):
    base = {"admitted": True, "required_bytes": 512 << 20, "reasons": [],
        "topology": {"host": {"admission_budget_bytes": 600 << 20, "disk_free": 3 << 30},
            "candidates": {"native": {"required_disk_bytes": 1 << 30}},
            "estimate": {"components": {"client_decode_owner_bytes": 32 << 20}}}}
    exp = SimpleNamespace(pipeline=SimpleNamespace(components={"cache": SimpleNamespace(params={"decode_max_bytes": 32 << 20})}))
    monkeypatch.setattr(driver, "base_admission", lambda *a, **k: base)
    ordinary = driver.admission(exp, "ordinary", require=False)
    candidate = driver.admission(exp, "decode", require=False)
    assert ordinary["admitted"] and candidate["admitted"]
    assert candidate["required_bytes"] == 512 << 20 and candidate["decode_owner_extra_bytes"] == 0
    assert candidate["decode_owner"]["already_charged_bytes"] == 32 << 20 and base["reasons"] == []
    del base["topology"]["estimate"]["components"]["client_decode_owner_bytes"]
    with pytest.raises(driver.MemoryAdmissionRejected):
        driver.admission(exp, "decode")


def test_pending_api_preflight_no_workers_and_no_private_material(driver, monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "freeze_installed_implementation", lambda build: None)
    from pllm import metrics, model_loader
    exp = SimpleNamespace(to_spec=lambda: {}, canonical_bytes=lambda: b"{}")
    monkeypatch.setattr(driver, "experiment", lambda *a: exp)
    monkeypatch.setattr(driver.runpy, "run_path", lambda *a: {"build_experiment": lambda: SimpleNamespace(pipeline=SimpleNamespace(model=None))})
    monkeypatch.setattr(metrics.BuildRef, "capture", lambda: SimpleNamespace(to_spec=lambda: {"digest": "installed"}))
    monkeypatch.setattr(model_loader, "resolve_model", lambda *a: SimpleNamespace(source_lock_digest="a" * 64))
    monkeypatch.setattr(driver, "semantic_geometry", lambda *a: {"stages": []})
    monkeypatch.setattr(driver, "admission", lambda *a, **k: {"admitted": True, "required_bytes": 768 << 20})
    monkeypatch.setattr(driver, "api_boundary", lambda: {"structurally_ready": False, "missing": ["pending API"]})
    monkeypatch.setattr(driver, "fresh_workers", lambda *a, **k: pytest.fail("launched pending API"))
    path = tmp_path / "preflight.json"
    driver.main(["--tiny", "--preflight-only", "--output", str(path)])
    report = json.loads(path.read_text())
    assert not report["complete"] and report["planned_clients"] == 18 and report["planned_requests"] == 54
    assert report["failure"]["missing_api"] == ["pending API"]
    assert "Explain sky." not in path.read_text() and "decode_key" not in path.read_text()


def test_disk_admission_adds_complete_cache_cap_to_existing_reserved_scope(driver, monkeypatch):
    base = {"admitted": True, "required_bytes": 512 << 20, "reasons": [],
        "topology": {"host": {"admission_budget_bytes": 2 << 30, "disk_free": 1100 << 20},
            "candidates": {"native": {"required_disk_bytes": 1 << 30}},
            "estimate": {"components": {"client_decode_owner_bytes": 32 << 20}}}}
    exp = SimpleNamespace(pipeline=SimpleNamespace(components={"cache": SimpleNamespace(params={"decode_max_bytes": 32 << 20})}))
    monkeypatch.setattr(driver, "base_admission", lambda *a, **k: base)
    report = driver.admission(exp, "decode", require=False)
    assert report["required_disk_bytes"] == 1280 << 20 and not report["admitted"]
    assert "disk headroom" in report["reasons"][-1]


def test_failed_worker_keeps_disk_and_body_charges_and_closes_caller_owner(driver, monkeypatch):
    import pllm
    from pllm.runtime import benchmark_response_group
    from pllm.runtime.client import PrivacyAudit
    closed = []
    raw_audit = PrivacyAudit().to_dict()
    raw_audit.update(correction_push_bytes=20, correction_ack_accounted_push_bytes=20,
                     correction_ack_bytes=3)
    owner = SimpleNamespace(storage_metrics={key: 10 for key in driver.STORAGE_FIELDS},
                            close=lambda: closed.append("owner"))
    client = SimpleNamespace(tokenizer_scope=lambda: nullcontext(),
        runtime=SimpleNamespace(capabilities=lambda: None),
        prepared_rows_for_response=lambda *a, **k: None,
        close=lambda: closed.append("client"), privacy_audit=SimpleNamespace(to_dict=lambda: raw_audit),
        _core=SimpleNamespace(_prepared_queue=SimpleNamespace(requests=[]),
            _online_active=False, _preparation_active=False))
    monkeypatch.setattr(driver, "owner_factory", lambda *a: owner)
    monkeypatch.setattr(pllm, "OpenAI", lambda **k: client)
    monkeypatch.setattr(pllm.Experiment, "from_spec", lambda *a: None)
    monkeypatch.setattr(benchmark_response_group, "prefix_owners", lambda *a: ())
    monkeypatch.setattr(benchmark_response_group, "require_concurrency_contract", lambda *a: None)
    monkeypatch.setattr(benchmark_response_group, "inspect_private_cleanup", lambda *a: {})
    def fail(*a):
        raise RuntimeError("private prompt/token/key")
    monkeypatch.setattr(driver, "sampling_response", fail)
    row = driver.client_worker({"policy": "decode", "cache": "unused", "decode_key": "aa" * 32,
        "experiment": {}, "connection": {}, "prompts": ["private prompt"] * 3,
        "cap": 8, "seeds": (7, 7, 7), "salt": "bb" * 32})
    assert closed == ["client", "owner"]
    assert row["failure"]["error_type"] == "RuntimeError" and row["responses"] == []
    assert row["ledger"]["all_link_serialized_body_bytes"] == 23
    assert row["storage_lifetime"]["written_bytes"] == 10
    assert row["storage_setup_cleanup_or_failed_request_residual"]["written_bytes"] == 10
    assert "private prompt/token/key" not in json.dumps(row) and "aaaa" not in json.dumps(row)


def test_sample_hook_seeds_actual_constructor_bypassed_runtime_and_restores(driver):
    import numpy as np
    class FixtureRuntime:
        def __init__(self, value):
            self.value = value
            self.rng = np.random.default_rng(42)
        def sample(self):
            return int(self.rng.integers(100))
    original = FixtureRuntime.sample
    states, draws = [], []
    for seed in (7, 7, 11, 19):
        with driver.harness_sampler_seed(seed, runtime_class=FixtureRuntime, salt=b"s" * 32) as witness:
            runtime = FixtureRuntime.__new__(FixtureRuntime)
            first = runtime.sample()
            states.append(copy.deepcopy(runtime.rng.bit_generator.state))
            draws.append([first, *(runtime.sample() for _ in range(15))])
            assert witness["runtimes"] == 1 and witness["samples"] == 16
        assert FixtureRuntime.sample is original
    assert states[0] == states[1] and draws[0] == draws[1]
    assert len({json.dumps(state, sort_keys=True) for state in states}) == 3
    assert draws[0] != draws[2] and draws[2] != draws[3]
    with pytest.raises(KeyboardInterrupt):
        with driver.harness_sampler_seed(7, runtime_class=FixtureRuntime, salt=b"s" * 32):
            raise KeyboardInterrupt()
    assert FixtureRuntime.sample is original
    assert FixtureRuntime("restored").rng.bit_generator.state == np.random.default_rng(42).bit_generator.state


def test_pending_last_selection_not_counted_as_evaluated_decode_path(driver):
    row = {"input_tokens": 4, "output_tokens": 8, "status": "incomplete", "audit_delta": {
        driver.WORK_COUNTERS[k]: v for k, v in {
            "requested_prefill_tokens": 4, "requested_decode_steps": 8,
            "executed_prefill_tokens": 4, "executed_terminal_prefill_rows": 1,
            "executed_decode_steps": 7}.items()}}
    with pytest.raises(RuntimeError, match="semantic work conservation"):
        driver.measured_work(GEOMETRY, row)


def test_owner_transient_must_fit_composition_envelope(driver, monkeypatch):
    report = {"admitted": True, "required_bytes": 512 << 20, "reasons": [],
        "topology": {"host": {"admission_budget_bytes": 2 << 30, "disk_free": 3 << 30},
            "candidates": {"native": {"required_disk_bytes": 1 << 30}},
            "estimate": {"components": {"client_decode_owner_bytes": 32 << 20}}}}
    exp = SimpleNamespace(pipeline=SimpleNamespace(components={"cache": SimpleNamespace(params={"decode_max_bytes": 32 << 20})}))
    monkeypatch.setattr(driver, "base_admission", lambda *a, **k: report)
    monkeypatch.setitem(driver.OWNER_LIMITS, "max_transient_bytes", 64 << 20)
    assert not driver.admission(exp, "decode", require=False)["admitted"]


def test_zero_preparation_delta_retains_v3_namespace_source_only(driver, monkeypatch):
    # Read-only Python accounting modules, isolated from installed PLLM/native SDK.
    package = ModuleType("decode_bench_accounting_fixture")
    package.__path__ = [str(driver.ROOT / "python/pllm/runtime")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    name = package.__name__ + ".topology_accounting"
    spec = importlib.util.spec_from_file_location(name, driver.ROOT / "python/pllm/runtime/topology_accounting.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    fields = ("session_authorization_upload_bytes", "session_authorization_download_bytes",
        "preparation_upload_bytes", "preparation_download_bytes", "correction_push_bytes",
        "bundle_network_bytes", "inference_upload_bytes", "inference_download_bytes",
        "inference_control_upload_bytes", "inference_control_download_bytes",
        "preparation_control_upload_bytes", "preparation_control_download_bytes",
        "correction_ack_bytes", "correction_ack_accounted_push_bytes")
    before = dict.fromkeys(fields, 0)
    before.update(correction_push_bytes=80, correction_ack_accounted_push_bytes=80, correction_ack_bytes=25)
    delta = driver.counter_delta(before, dict(before))
    assert set(delta) == set(fields) and set(delta.values()) == {0}
    ledger = module.prepared_body_accounting({"privacy": delta})
    assert ledger["body_counter_coverage"] == module.ACK_COVERAGE
    assert ledger["correction_ack_counter_set_present"] and ledger["tracked_body_counter_set_present"]
    assert ledger["all_link_serialized_body_bytes"] == 0
    # Validate an all-hit request using only this isolated accounting namespace.
    fake_pllm, fake_runtime = ModuleType("pllm"), ModuleType("pllm.runtime")
    fake_pllm.__path__, fake_runtime.__path__ = [], []
    monkeypatch.setitem(sys.modules, "pllm", fake_pllm)
    monkeypatch.setitem(sys.modules, "pllm.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "pllm.runtime.topology_accounting", module)
    row = request(driver, warm=True)
    row["request_ledger"] = ledger
    assert driver.validate_request(row, "decode", GEOMETRY)["zero_executed_work"]
    row["audit_delta"]["prepared_stage_rows_claimed"] = 1
    with pytest.raises(RuntimeError, match="claimed prepared rows"):
        driver.validate_request(row, "decode", GEOMETRY)


def test_real_sample_method_end_to_end_seed_witness_and_unseen_extra_draw(driver):
    import numpy as np
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    original = SemanticDecoderRuntime.sample
    reports, outputs = [], []
    for seed in (7, 7, 11):
        runtime = SemanticDecoderRuntime.__new__(SemanticDecoderRuntime)
        with driver.harness_sampler_seed(seed, salt=b"s" * 32) as witness:
            outputs.append([runtime.sample(np.array([0., .2, .3], dtype=np.float32), temperature=.7, top_p=.9)
                            for _ in range(8)])
            reports.append(dict(witness))
            assert witness["samples"] == 8 and witness["runtimes"] == 1
    assert outputs[0] == outputs[1] and reports[0] == reports[1]
    assert reports[0]["initial"] != reports[2]["initial"] and reports[0]["final"] != reports[2]["final"]
    assert SemanticDecoderRuntime.sample is original
    # Peaky logits keep outputs unchanged despite extra RNG consumption.
    def extra(runtime, *args, **kwargs):
        token = original(runtime, *args, **kwargs)
        runtime.rng.random()
        return token
    reports, outputs = [], []
    for sampler in (original, extra):
        SemanticDecoderRuntime.sample = sampler
        try:
            runtime = SemanticDecoderRuntime.__new__(SemanticDecoderRuntime)
            with driver.harness_sampler_seed(7, salt=b"s" * 32) as witness:
                outputs.append(runtime.sample(np.array([0., -1000.], dtype=np.float32), temperature=.7))
                reports.append(dict(witness))
        finally:
            SemanticDecoderRuntime.sample = original
    assert outputs == [0, 0] and reports[0]["final"] != reports[1]["final"]


def test_streaming_measurement_adapter_seeds_real_sample_runtime_not_constructor(driver, monkeypatch):
    import numpy as np
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.types import Response
    draws = []
    def create(**kwargs):
        runtime = SemanticDecoderRuntime.__new__(SemanticDecoderRuntime)
        # The real sampler would fail without sample-hook initialization here.
        token = runtime.sample(np.array([0., .1, .3], dtype=np.float32), temperature=.7, top_p=.9)
        draws.append(token)
        response = Response(id="fixture", model="fixture", output=[], status="incomplete",
            incomplete_details={"reason": "max_output_tokens"})
        from pllm.runtime.types import ResponseUsage
        response.usage = ResponseUsage(4, 1, 5)
        return nullcontext(iter([SimpleNamespace(type="response.output_text.delta"),
            SimpleNamespace(type="response.incomplete", response=response.to_dict())]))
    client = SimpleNamespace(tokenizer_scope=lambda: nullcontext(), responses=SimpleNamespace(create=create))
    monkeypatch.setattr(driver, "complete_state_commitment", lambda *a: {"output": "salted"})
    one = driver.sampling_response(client, "private input", 1, 7, b"s" * 32, 0)
    two = driver.sampling_response(client, "private input", 1, 7, b"s" * 32, 1)
    different = driver.sampling_response(client, "private input", 1, 11, b"s" * 32, 2)
    assert draws[0] == draws[1] and one["state"]["rng"] == two["state"]["rng"]
    assert one["state"]["rng"]["initial"] != different["state"]["rng"]["initial"]
    assert "private input" not in json.dumps(one)


@pytest.mark.parametrize("field", ["rng", "history", "rendered_history"])
def test_missing_real_state_witness_cannot_claim_parity(driver, field):
    row = request(driver)
    del row["state"][field]
    with pytest.raises(RuntimeError, match="witnesses unavailable"):
        driver.validate_request(row, "ordinary", GEOMETRY)
