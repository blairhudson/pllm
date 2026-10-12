import threading
import time

import pytest

from pllm.runtime.preparation_window import (
    WINDOW_BYTES, preparation_job_work_bytes, run_preparation_window,
    validate_preparation_window,
)


def run(items, **kwargs):
    return run_preparation_window(items, width=kwargs.pop("width", 4),
        cost=kwargs.pop("cost", lambda _: 1),
        prepare=kwargs.pop("prepare", lambda item: item),
        accept=kwargs.pop("accept", lambda _: None),
        abort=kwargs.pop("abort", lambda: None), discard=kwargs.pop("discard", lambda _: None),
        cancelled=kwargs.pop("cancelled", lambda: False), **kwargs)


def test_window_rejects_aggregate_work_before_issuing_any_stage():
    called = []
    with pytest.raises(ValueError, match="declared work window"):
        run([1, 2], cost=lambda x: WINDOW_BYTES * x, prepare=called.append)
    assert not called


def test_long_prefill_prices_complete_job_and_serial_has_separate_contract():
    # Public Qwen2.5 gate/up shape: 288 prefill + seven prepaid decode rows.
    size = preparation_job_work_bytes(rows=295, in_features=896, out_features=9728)
    assert size == 73_107_456
    with pytest.raises(ValueError, match="exceeds the declared work window"):
        validate_preparation_window(width=4, max_bytes=64 << 20, job_bytes=[size])
    validate_preparation_window(width=1, job_bytes=[size])
    seen = []
    run([size], width=1, cost=lambda _: pytest.fail("serial work uses topology admission"),
        prepare=seen.append)
    assert seen == [size]


def test_verifier_window_work_prices_authorized_not_pruned_rows():
    ordinary = preparation_job_work_bytes(rows=8, in_features=896, out_features=4864)
    verified = preparation_job_work_bytes(rows=8, in_features=896, out_features=4864,
        verification_checks=3, verification_rows=295)
    assert verified - ordinary == 295 * 3 * 896 * 16


@pytest.mark.parametrize("budget", [WINDOW_BYTES, 64 << 20])
def test_window_overlaps_with_fixed_declared_memory_bound(budget):
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active, peak, results = 0, 0, []
    def prepare(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait(timeout=2)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value
    run(range(8), max_bytes=budget, cost=lambda _: budget // 2, prepare=prepare, accept=results.append)
    assert sorted(results) == list(range(8)) and peak == 2


def test_failure_burns_before_join_and_discards_other_issued_results():
    started, cancelled = threading.Event(), threading.Event()
    discarded, accepted = [], []
    def prepare(value):
        if value == 0:
            assert started.wait(2)
            raise ValueError("bad ack")
        started.set()
        assert cancelled.wait(2), "cancellation must precede waiting for issued work"
        return value
    with pytest.raises(ValueError, match="bad ack"):
        run(range(9), width=2, prepare=prepare, accept=accepted.append,
            abort=cancelled.set, discard=discarded.append)
    assert cancelled.is_set() and discarded == [1] and accepted == []


def test_cancelled_client_cannot_start_or_admit_material():
    calls = []
    with pytest.raises(RuntimeError, match="cancelled"):
        run([1], cancelled=lambda: True, prepare=calls.append)
    assert calls == []


def test_explicit_window_is_bound_by_component_and_native_schedule():
    from pllm import Model, lower_model
    from pllm.configuration import ComponentRef, ConfigurationError
    from pllm.preparation import PreparedInventory
    from pllm.profiles import MaskedLinearCpu
    from pllm.sources import _tiny_model_config

    base = PreparedInventory(refill="on-demand", allocation="demand")
    assert "stage_window_bytes" not in base.params
    candidate = base.with_params(stage_window=4, stage_window_bytes=64 << 20)
    plan = lower_model(_tiny_model_config(), batch=1, max_input_tokens=16, max_new_tokens=2)
    plan.runtime_schedule(MaskedLinearCpu(Model.tiny(), inventory=candidate))
    for value in (True, 0, (64 << 20) + 1):
        with pytest.raises(ConfigurationError):
            base.with_params(stage_window=4, stage_window_bytes=value)
        forged = ComponentRef(base.component, {**candidate.params, "stage_window_bytes": value})
        with pytest.raises(ValueError):
            plan.runtime_schedule(MaskedLinearCpu(Model.tiny(), inventory=forged))
    with pytest.raises(ConfigurationError):
        base.with_params(stage_window_bytes=64 << 20)
