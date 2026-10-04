import threading
import time

import pytest

from pllm.runtime.preparation_window import WINDOW_BYTES, run_preparation_window


def run(items, **kwargs):
    return run_preparation_window(items, width=kwargs.pop("width", 4),
        cost=kwargs.pop("cost", lambda _: 1),
        prepare=kwargs.pop("prepare", lambda item: item),
        accept=kwargs.pop("accept", lambda _: None),
        abort=kwargs.pop("abort", lambda: None), discard=kwargs.pop("discard", lambda _: None),
        cancelled=kwargs.pop("cancelled", lambda: False), **kwargs)


def test_window_rejects_aggregate_work_before_issuing_any_stage():
    called = []
    with pytest.raises(ValueError, match="16-MiB"):
        run([1, 2], cost=lambda x: WINDOW_BYTES * x, prepare=called.append)
    assert not called


def test_window_overlaps_with_fixed_declared_memory_bound():
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
    run(range(8), cost=lambda _: WINDOW_BYTES // 2, prepare=prepare, accept=results.append)
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
