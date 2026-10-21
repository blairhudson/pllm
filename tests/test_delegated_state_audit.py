"""Generated-only per-selection state commitments and verified publication."""

import gc
import hashlib
import hmac
import json
import threading
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest

from pllm._benchmark_contract import TargetUnavailable
from pllm.metrics import BenchmarkTarget, Sampling, Scenario
from pllm.runtime import benchmark_response_audit as audit
from pllm.runtime import delegated_decoder as decoder
from pllm.runtime.transformer_client import MaskedTransformerClientRuntime

from test_delegated_decoder import clear_oracle, config, fixture
from test_delegated_benchmark import experiment, native_contract  # noqa: F401


SALT = b"x" * 32


def commit(value, salt=SALT):
    return hmac.new(salt, json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
                    hashlib.sha256).hexdigest()


def state_hash(logits, kv, salt=SALT):
    """Independent byte construction; no production hash helper/kernel calls."""
    payload = b"pllm.response-gkm-full-state.v1\0"
    payload += json.dumps([(len(logits) // 4,), "<f4"], separators=(",", ":")).encode()
    payload += logits
    for shape, key, value in kv:
        for raw in (key, value):
            payload += json.dumps([shape, "<f4"], separators=(",", ":")).encode() + raw
    return hmac.new(salt, payload, hashlib.sha256).hexdigest()


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_generated_full_seeded_per_selection_matches_independent_oracle(family):
    cfg = config(family)
    _, arrays, source = fixture(cfg)
    tokens, observed = (0, 4, 9), []
    sample = MaskedTransformerClientRuntime.sample
    for seeded in (False, True):
        with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
            result = decoder.execute_gkm_reference(cfg, source, tokens, decode_steps=2, seeded=seeded)
            assert records == []  # Even verified commits wait for enclosing benchmark success.
        rows = records[0]["selections"]
        assert len(rows) == 3
        for step, row in enumerate(rows):
            logits, kv, selected, executed, position = clear_oracle(cfg, arrays, tokens, step)
            assert row["logits_and_active_kv"] == state_hash(logits[-1], kv)
            assert row["position"] == row["executed_token_count"] == position
            assert row["cache_lengths"] == [position] * cfg["num_hidden_layers"]
            assert row["executed_history"] == commit(executed)
            assert row["selected_token"] == row["pending_selected_token"] == commit(selected[-1])
            assert row["checked_values"] == (len(logits[-1]) + sum(len(k) + len(v) for _, k, v in kv)) // 4
            assert "rng_before" not in row and "rng_after" not in row
        assert records[0]["sampler_kind"] == "deterministic_argmax_no_rng"
        assert "sampler_seed" not in records[0]
        assert result.position == 5
        observed.append(records)
    assert observed[0] == observed[1]
    assert MaskedTransformerClientRuntime.sample is sample


@pytest.fixture
def scripted(monkeypatch):
    """Trusted decoder lifecycle with known bytes and injectable intermediate KV."""
    contract = {"decode_steps": 2, "input_rows": 2, "vocabulary": 3,
                "numeric_profile": "q11", "local_numeric_profile": "f32",
                "plan": {"config_digest": "1" * 64}}
    refs, state = [], NS(changed=False, retire_failure=False, phase_failure=False)

    class Request:
        def __init__(self, native, contract):
            self.position = 0
            self.caches = []
            refs.append(weakref.ref(self))

        def phase(self, ids):
            if state.phase_failure:
                raise ValueError("local phase rejected")
            self.position += len(ids)
            key = np.arange(self.position * 2, dtype="<f4").reshape(self.position, 1, 2)
            if state.changed and self.position == 2:
                key.flat[0] = 17
            self.caches = [NS(key=key, value=-key, length=self.position)]
            return np.array([1, 2, 3], dtype="<f4").tobytes(), 2

        def retire(self):
            self.caches.clear()
            if state.retire_failure:
                raise RuntimeError("cleanup failed")

    class Native:
        def __init__(self, *, failure=False, cancel=None):
            self.failure, self.cancel, self.aborted = failure, cancel, False

        def contract(self):
            return json.dumps(contract)

        def finish(self):
            if self.failure:
                raise RuntimeError("verification failed")
            if self.cancel is not None:
                self.cancel.set()
            return b'{"accepted":true}'

        def abort(self):
            self.aborted = True

    monkeypatch.setattr(decoder, "_Request", Request)
    return NS(contract=contract, native=Native, state=state, refs=refs)


def execute(scripted, **kwargs):
    return decoder._execute_verified_window(
        scripted.native(), scripted.contract, (0, 1), **kwargs)


def test_intermediate_kv_change_detected_even_when_final_state_same(scripted):
    results, records = [], []
    for changed in (False, True):
        scripted.state.changed = changed
        with audit.response_state_audit(SALT, runtime_kind="gkm") as rows:
            results.append(execute(scripted))
        records.append(rows[0]["selections"])
    assert results[0] == results[1]
    assert records[0][0]["logits_and_active_kv"] != records[1][0]["logits_and_active_kv"]
    assert records[0][1:] == records[1][1:]
    assert records[0][0]["logits_and_active_kv"] == state_hash(
        np.array([1, 2, 3], dtype="<f4").tobytes(),
        [((2, 1, 2), np.arange(4, dtype="<f4").tobytes(), (-np.arange(4, dtype="<f4")).tobytes())])
    # Frozen independently using struct.pack('<3f'/'<4f'), including negative zero.
    assert records[0][0]["logits_and_active_kv"] == "59addaaf3006cff6eb3b18d27326c9c6163f98fd01ef54e25210404fb865a907"
    gc.collect()
    assert all(ref() is None for ref in scripted.refs)


@pytest.mark.parametrize("failure", ["verifier", "cancel", "cleanup", "phase", "request", "abandoned"])
def test_failure_never_publishes_and_releases_scope(scripted, failure):
    cancelled = threading.Event()
    native = scripted.native(failure=failure == "verifier", cancel=cancelled if failure == "cancel" else None)
    scripted.state.retire_failure = failure == "cleanup"
    scripted.state.phase_failure = failure == "phase"
    with pytest.raises((RuntimeError, ValueError, TargetUnavailable)):
        with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
            decoder._execute_verified_window(native, scripted.contract,
                (True, 1) if failure == "request" else (0, 1), cancelled=cancelled.is_set)
            assert records == []
            if failure == "abandoned":
                raise RuntimeError("benchmark abandoned after native acceptance")
    assert records == [] and native.aborted
    assert audit._GKM_OWNER is None
    scripted.state.retire_failure = scripted.state.phase_failure = False
    with audit.response_state_audit(SALT, runtime_kind="gkm") as fresh:
        execute(scripted)
    assert len(fresh) == 1
    gc.collect()
    assert all(ref() is None for ref in scripted.refs)


def test_closed_scope_cannot_accept_late_work(scripted):
    with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
        transaction = audit._begin_gkm_request_audit(scripted.contract, (0, 1), None)
    assert records == []
    with pytest.raises(TargetUnavailable, match="retired"):
        transaction.accept(NS())
    transaction.close()
    assert records == []


def test_cross_thread_capture_and_no_rng_instrumentation(scripted, monkeypatch):
    sample = MaskedTransformerClientRuntime.sample
    monkeypatch.setattr(np.random, "default_rng", lambda *_a: pytest.fail("sampler RNG modified"))
    errors = []

    def run():
        try:
            execute(scripted)
        except BaseException as error:
            errors.append(error)

    with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
        thread = threading.Thread(target=run)
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and not errors
        assert records == []
    assert len(records) == 1 and MaskedTransformerClientRuntime.sample is sample


def test_bounds_overlap_and_disabled_path(scripted):
    assert audit._begin_gkm_request_audit({}, (), None) is None
    with audit.response_state_audit(SALT, runtime_kind="gkm"):
        with pytest.raises(TargetUnavailable, match="concurrent"):
            with audit.response_state_audit(SALT):
                pass
    for contract, continuation in (({**scripted.contract, "decode_steps": 64}, None),
                                    (scripted.contract, (2, 2))):
        with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
            with pytest.raises(TargetUnavailable):
                audit._begin_gkm_request_audit(contract, (0, 1), continuation)
        assert records == []
    with audit.response_state_audit(SALT, runtime_kind="gkm") as records:
        execute(scripted)
        with pytest.raises(TargetUnavailable, match="one_response"):
            execute(scripted)
    assert records == []


def test_admission_uses_same_32_mib_for_both_sharing_profiles(monkeypatch):
    from pllm.runtime import benchmark_memory
    monkeypatch.setattr(benchmark_memory, "host_memory", lambda: "host")
    monkeypatch.setattr(benchmark_memory, "admit_memory", lambda estimate, host, **kw: estimate)
    for sharing in ("full", "seeded"):
        report = {"estimate": {"client_peak_bytes": 100, "native_total_peak_bytes": 300,
                                "components": {}, "sharing": sharing}}
        priced = audit.state_audit_admission(report, 1 << 30, enforced_wan=True)
        assert priced["client_peak_bytes"] == 100 + (32 << 20)
        assert priced["native_total_peak_bytes"] == 300 + (32 << 20)
        assert report["estimate"]["components"] == {}


@pytest.mark.parametrize("success", [True, False])
def test_ordinary_target_records_gkm_coverage(experiment, native_contract, scripted, monkeypatch, tmp_path, success):
    from test_delegated_benchmark import report
    from pllm.runtime import benchmark_cli, benchmark_response_target
    from pllm import metrics
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *_a, **_kw: {"admitted": True})
    admissions = []
    monkeypatch.setattr(audit, "state_audit_admission", lambda value, *a, **kw:
                        admissions.append(kw) or value)

    def run(**kwargs):
        execute(scripted)
        result = report(experiment)
        result["configuration"]["provider_backend"] = "native"
        result["checks"]["passed"] = success
        return result

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    measured = NS(inputs={"prompt": "hello"}, directory=tmp_path, salt=SALT,
        sampling=Sampling(processes=1, memory_bytes=1 << 30, timeout_seconds=30), details={}, comparison={}, limitations=[],
        admit=lambda _: None, run=lambda callback: [callback(0)], metric=lambda *a, **kw: None)
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3), state_audit=True)
    if not success:
        with pytest.raises(TargetUnavailable, match="response_checks_failed"):
            benchmark_response_target.execute_response(target.to_spec()["parameters"], measured)
        assert "response_state_audit" not in measured.details
        assert "output_digest" not in measured.comparison
        return
    benchmark_response_target.execute_response(target.to_spec()["parameters"], measured)
    assert admissions == [{"enforced_wan": False}]
    record = measured.details["response_state_audit"]
    assert record["schema"] == audit.GKM_AUDIT_SCHEMA
    assert len(record["bindings"][0]["selections"]) == 3
    assert record["sampler_kind"] == "deterministic_argmax_no_rng"
    assert not record["terminal_eos_selection_may_exceed_emitted_count"]
    assert measured.oracle["kind"] == "salted_full_logit_kv_history_commitments"


@pytest.mark.parametrize("kind", ["replicated", "fss"])
def test_other_buffered_targets_still_reject_state_audit(experiment, monkeypatch, tmp_path, kind):
    from pllm.runtime import benchmark_delegated, benchmark_response_target
    from pllm import metrics
    monkeypatch.setattr(benchmark_delegated, "is_buffered", lambda _: True)
    monkeypatch.setattr(benchmark_delegated, "is_fss", lambda _: kind == "fss")
    monkeypatch.setattr(benchmark_delegated, "is_replicated", lambda _: kind == "replicated")
    monkeypatch.setattr(metrics, "benchmark_memory", lambda *_a, **_kw: pytest.fail("memory touched"))
    target = BenchmarkTarget.response(experiment, scenario=Scenario.cold(max_output_tokens=3), state_audit=True)
    measured = NS(inputs={"prompt": "hello"}, directory=tmp_path, sampling=Sampling(processes=1))
    with pytest.raises(TargetUnavailable, match=f"{kind}_response_options_unsupported"):
        benchmark_response_target.execute_response(target.to_spec()["parameters"], measured)
