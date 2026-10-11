from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import threading

import numpy as np
import pytest

from pllm.runtime.prepared_queue import (
    OwnedPreparedAttempt, PreparedCancellation, PreparedExecutionQueue, PreparedQueueClosed,
    prepared_attempt, prepared_cancellation,
)
from pllm.runtime.inflight_prefix import InflightBinding, InflightPrefixOwner, InflightResources
from pllm.runtime.transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis


def test_pending_prefix_waiter_does_not_block_its_later_producer():
    queue = PreparedExecutionQueue(max_requests=2)
    with queue.admit() as waiter, queue.admit() as producer:
        producer.enter()
        assert queue.requests == 2
        with pytest.raises(RuntimeError, match="capacity"):
            queue.admit()
        producer.close()
        waiter.enter()
    assert queue.requests == 0


def test_close_wakes_queued_scope_but_keeps_active_allocation_charged():
    queue = PreparedExecutionQueue(max_requests=2)
    active = queue.admit()
    active.enter()
    ready = threading.Event()

    def waiting():
        with queue.admit() as scope:
            ready.set()
            with pytest.raises(PreparedQueueClosed):
                scope.enter()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(waiting)
        assert ready.wait(2)
        queue.close()
        future.result(timeout=2)
    assert queue.requests == 1
    with pytest.raises(PreparedQueueClosed):
        queue.admit()
    active.close()
    assert queue.requests == 0


def test_cancelling_one_queued_scope_preserves_active_and_later_work():
    queue = PreparedExecutionQueue(max_requests=3)
    active = queue.admit()
    active.enter()
    cancelled = threading.Event()
    started = threading.Event()

    def waiting():
        with queue.admit() as scope:
            started.set()
            with pytest.raises(PreparedQueueClosed):
                scope.enter(cancelled=cancelled)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(waiting)
        assert started.wait(2)
        cancelled.set()
        future.result(timeout=2)
    assert queue.requests == 1
    active.close()
    with queue.admit() as later:
        later.enter()
        with pytest.raises(RuntimeError, match="consumed"):
            later.enter()
    assert queue.requests == 0


def test_failure_releases_execution_seat():
    queue = PreparedExecutionQueue(max_requests=1)
    with pytest.raises(ValueError):
        with queue.admit() as scope:
            scope.enter()
            raise ValueError("failed request")
    with queue.admit() as replacement:
        replacement.enter()
    assert queue.requests == 0


TOKENIZER = ContextVar("test_prepared_tokenizer", default=None)
BINDING = InflightBinding(*[str(n) * 64 for n in range(1, 6)])
RESOURCES = InflightResources(1, 1, 1, 4, 8)


class Client:
    """Core-name-compatible harness; no model/session/network/native execution."""

    def __init__(self, events, maximum=8):
        self._prepared_queue = PreparedExecutionQueue(maximum)
        self._prepared_attempt_lock = threading.RLock()
        self._prepared_attempts = set()
        self._closing = False
        self._events = events
        self._inflight_prefix_owner = InflightPrefixOwner(self)
        self.order = []
        self.bundles_closed = threading.Event()

    @contextmanager
    def tokenizer_scope(self):
        token = TOKENIZER.set(object())
        try:
            yield
        finally:
            self.order.append("tokenizer-exit")
            TOKENIZER.reset(token)

    def close(self):
        with self._prepared_attempt_lock:
            self._closing = True
            attempts = tuple(self._prepared_attempts)
        self._prepared_queue.close()
        # Signal every scope before blocking on any running generator.
        for attempt in attempts:
            attempt.cancel()
        for attempt in attempts:
            attempt.close()
        assert self._inflight_prefix_owner.close(timeout=0)
        self.order.append("bundles-close")
        self.bundles_closed.set()


def prefix(client):
    return client._inflight_prefix_owner.lookup_or_register(
        client, BINDING, [1, 2], RESOURCES, admit=lambda p, s: None)


def publish(attempt):
    execution = hashlib.sha256(b"pllm.decoder.execution-basis.v1\0" + BINDING.native_prefill_digest.encode()
                               + np.arange(2, dtype="<i8").tobytes()
                               + np.asarray([1, 2], dtype="<i8").tobytes()).hexdigest()
    basis = _snapshot_state_basis("completed_prefill", 2, BINDING.state_binding_digest,
                                  BINDING.native_prefill_digest, execution, None, 2)
    key = np.array([1, 2], dtype=np.float32).reshape(2, 1, 1).copy()
    saved = RuntimeSnapshot(2, [LayerCache(key, key.copy(), 2)], {}, BINDING.state_binding_digest, basis)
    logits = np.arange(4, dtype=np.float32)
    attempt.prefix.publish(saved, logits, validate_snapshot=lambda s: None)
    attempt.published = True
    key.fill(0)
    saved.caches[0].value.fill(0)
    logits.fill(0)


def assert_retired(client, signal=None):
    assert not client._prepared_attempts
    assert client._prepared_queue.requests == client._prepared_queue.active_requests == 0
    counters = client._inflight_prefix_owner.counters()
    assert counters["nodes"] == counters["requests"] == counters["reserved_bytes"] == 0
    if signal is not None:
        assert not signal._attempts
    assert prepared_attempt.get() is None and TOKENIZER.get() is None


def test_natural_exhaustion_is_not_cancellation_and_cleanup_runs_once():
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        attempt.enter()
        try:
            publish(attempt)
            yield "response"
        finally:
            client.order.append("runtime-reset")
    client = Client(events)
    attempt = OwnedPreparedAttempt(client, {})
    assert list(attempt) == ["response"]
    before = client._inflight_prefix_owner.counters()
    assert before["failures"] == before["cancellations"] == before["cancelled_producers"] == 0
    assert not attempt.cancelled.is_set()
    attempt.close()
    attempt.cancel()
    attempt.finish_prefix()
    assert client._inflight_prefix_owner.counters() == before
    assert client.order == ["runtime-reset", "tokenizer-exit"]
    assert_retired(client)


def test_parent_early_finish_hook_is_idempotent_at_wrapper_exhaustion():
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        attempt.enter()
        publish(attempt)
        client.order.append("runtime-reset")
        attempt.finish_prefix()
        attempt.finish_prefix()
        yield "complete"
    client = Client(events)
    attempt = OwnedPreparedAttempt(client, {})
    assert list(attempt) == ["complete"]
    assert client._inflight_prefix_owner.counters()["failures"] == 0
    assert_retired(client)


def test_shared_producer_cancelled_while_queued_still_prefills_then_stops_decode():
    bound, entered = threading.Event(), threading.Event()
    continuation = threading.Event()
    decode = []
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        bound.set()
        assert continuation.wait(3)
        attempt.enter()
        entered.set()
        try:
            publish(attempt)
            attempt.check_cancelled()  # Cancellation resumes after shared prefill.
            decode.append("must-not-run")
            yield "response"
        finally:
            client.order.append("runtime-reset")
    client = Client(events)
    active = client._prepared_queue.admit()
    active.enter()
    attempt = OwnedPreparedAttempt(client, {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, attempt)
        assert bound.wait(3)
        survivor = prefix(client)
        assert not survivor.is_producer
        attempt.cancel()
        assert not attempt.prefix.producer_cancel.is_set()
        continuation.set()
        assert not entered.is_set()
        active.close()
        with pytest.raises(PreparedQueueClosed, match="request cancelled"):
            future.result(timeout=3)
    assert entered.is_set() and not decode
    lease = survivor.wait(timeout=0)
    assert lease.logits[1] == 1
    lease.close()
    assert client._inflight_prefix_owner.counters()["publications"] == 1
    assert_retired(client)


def test_last_waiter_cancel_revokes_queued_producer_deferral():
    bound, continue_enter = threading.Event(), threading.Event()
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        bound.set()
        assert continue_enter.wait(3)
        attempt.enter()
        yield "unexpected"
    client = Client(events)
    active = client._prepared_queue.admit()
    active.enter()
    attempt = OwnedPreparedAttempt(client, {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, attempt)
        assert bound.wait(3)
        survivor = prefix(client)
        attempt.cancel()
        survivor.cancel()
        assert attempt.prefix.producer_cancel.is_set()
        continue_enter.set()
        with pytest.raises(PreparedQueueClosed):
            future.result(timeout=3)
    assert client._prepared_queue.requests == 1
    active.close()
    assert_retired(client)


def test_context_registry_cancels_prefix_waiter_before_response_id():
    waiting = threading.Event()
    signal = PreparedCancellation()
    def events(body):
        attempt = prepared_attempt.get()
        assert prepared_cancellation.get() is signal and TOKENIZER.get() is not None
        request = prefix(client)
        assert not request.is_producer
        attempt.bind_prefix(request)
        waiting.set()
        try:
            request.wait()
            attempt.enter()
            yield "unexpected"
        finally:
            client.order.append("runtime-reset")
    client = Client(events)
    producer = prefix(client)
    token = prepared_cancellation.set(signal)
    try:
        attempt = OwnedPreparedAttempt(client, {})
    finally:
        prepared_cancellation.reset(token)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, attempt)
        assert waiting.wait(3)
        signal.cancel()
        from pllm.runtime.inflight_prefix import InflightCancelled
        with pytest.raises(InflightCancelled):
            future.result(timeout=3)
    assert not producer.producer_cancel.is_set()
    assert client._prepared_queue.requests == 0 and not signal._attempts
    producer.finish_producer()
    producer.release()
    assert_retired(client, signal)


def test_precancelled_context_never_initializes_body_or_leaks_scope():
    signal = PreparedCancellation()
    signal.cancel()
    called = []
    def events(body):
        called.append(True)
        yield "unexpected"
    client = Client(events)
    token = prepared_cancellation.set(signal)
    try:
        attempt = OwnedPreparedAttempt(client, {})
    finally:
        prepared_cancellation.reset(token)
    with pytest.raises(PreparedQueueClosed):
        next(attempt)
    assert not called
    assert_retired(client, signal)


def test_caller_close_blocks_shared_producer_until_safe_reset_and_publication():
    working, finish, close_started = threading.Event(), threading.Event(), threading.Event()
    decoded = []
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        yield "initialization"  # Stream may suspend before online prefill.
        attempt.enter()
        working.set()
        try:
            assert finish.wait(3)
            publish(attempt)
            attempt.check_cancelled()
            decoded.append(True)
            yield "unexpected"
        finally:
            client.order.append("runtime-reset")
    client = Client(events)
    attempt = OwnedPreparedAttempt(client, {})
    assert next(attempt) == "initialization"
    survivor = prefix(client)
    def close():
        close_started.set()
        attempt.close()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(close)
        assert close_started.wait(3) and working.wait(3)
        assert not future.done()
        assert client._prepared_queue.active_requests == 1
        assert client._inflight_prefix_owner.counters()["active_producers"] == 1
        finish.set()
        future.result(timeout=3)
    assert not decoded and client.order[0] == "runtime-reset"
    survivor.wait(timeout=0).close()
    attempt.close()
    assert_retired(client)


def test_client_close_drains_active_inner_runtime_before_bundles():
    working, reset_allowed, close_started = threading.Event(), threading.Event(), threading.Event()
    def events(body):
        attempt = prepared_attempt.get()
        attempt.enter()
        working.set()
        try:
            assert reset_allowed.wait(3)
            attempt.check_cancelled()
            yield "unexpected"
        finally:
            assert client._prepared_queue.active_requests == 1
            assert not client.bundles_closed.is_set()
            client.order.append("runtime-reset")
    client = Client(events)
    attempt = OwnedPreparedAttempt(client, {})
    def close():
        close_started.set()
        client.close()
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker = pool.submit(next, attempt)
        assert working.wait(3)
        closer = pool.submit(close)
        assert close_started.wait(3)
        assert attempt.cancelled.wait(3)
        assert not closer.done() and not client.bundles_closed.is_set()
        reset_allowed.set()
        with pytest.raises(PreparedQueueClosed):
            worker.result(timeout=3)
        closer.result(timeout=3)
    assert client.order == ["runtime-reset", "tokenizer-exit", "bundles-close"]
    assert_retired(client)


def test_one_active_attempt_holds_execution_scope_across_all_epochs():
    second_ready, second_entered = threading.Event(), threading.Event()
    def events(body):
        attempt = prepared_attempt.get()
        if body == "second":
            second_ready.set()
        attempt.enter()
        if body == "second":
            second_entered.set()
        try:
            for epoch in range(3):
                yield epoch
        finally:
            client.order.append("runtime-reset")
    client = Client(events)
    first, second = OwnedPreparedAttempt(client, "first"), OwnedPreparedAttempt(client, "second")
    assert next(first) == 0
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, second)
        assert second_ready.wait(3)
        assert not second_entered.is_set()
        assert next(first) == 1 and next(first) == 2
        assert not second_entered.is_set()
        with pytest.raises(StopIteration):
            next(first)
        assert future.result(timeout=3) == 0
    second.close()
    assert_retired(client)


def test_prefix_failure_still_finishes_releases_and_resets_context():
    class FailedClaim:
        is_producer = True
        producer_cancel = threading.Event()
        def fail(self):
            calls.append("fail")
            raise ValueError("failed cleanup hook")
        def finish_producer(self):
            calls.append("finish")
        def release(self):
            calls.append("release")
        def cancel(self):
            calls.append("cancel")
    calls = []
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(FailedClaim())
        attempt.enter()
        try:
            yield "one"
        finally:
            calls.append("runtime-reset")
    client = Client(events)
    attempt = OwnedPreparedAttempt(client, {})
    assert next(attempt) == "one"
    with pytest.raises(ValueError, match="cleanup hook"):
        next(attempt)
    assert calls == ["runtime-reset", "fail", "finish", "release"]
    attempt.close()
    assert_retired(client)


def test_cancelled_failed_queue_entry_retains_initialization_seat_through_reset(monkeypatch):
    initialized, enter_allowed, resetting, reset_allowed = (threading.Event() for _ in range(4))
    def events(body):
        attempt = prepared_attempt.get()
        try:
            attempt.enter()
            yield "unexpected"
        finally:
            resetting.set()
            assert reset_allowed.wait(3)
            client.order.append("runtime-reset")
    client = Client(events, maximum=2)
    active = client._prepared_queue.admit()
    active.enter()
    original_enter = client._prepared_queue._enter
    def gated_enter(scope, cancelled):
        initialized.set()  # check_cancelled passed; initialization owns state.
        assert enter_allowed.wait(3)
        return original_enter(scope, cancelled)
    monkeypatch.setattr(client._prepared_queue, "_enter", gated_enter)
    attempt = OwnedPreparedAttempt(client, {})
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, attempt)
        assert initialized.wait(3)
        attempt.cancel()
        enter_allowed.set()
        assert resetting.wait(3)
        assert client._prepared_queue.requests == 2
        with pytest.raises(RuntimeError, match="bound"):
            client._prepared_queue.admit()
        reset_allowed.set()
        with pytest.raises(PreparedQueueClosed):
            future.result(timeout=3)
    assert client._prepared_queue.requests == 1
    active.close()
    assert_retired(client)


def test_active_producer_signal_cancel_finishes_prefix_without_private_decode():
    working, finish = threading.Event(), threading.Event()
    signal = PreparedCancellation()
    decoded = []
    def events(body):
        attempt = prepared_attempt.get()
        attempt.bind_prefix(prefix(client))
        attempt.enter()
        working.set()
        try:
            assert finish.wait(3)
            attempt.check_cancelled()  # Defers while survivor still needs prefix.
            publish(attempt)
            attempt.check_cancelled()
            decoded.append(True)
            yield "unexpected"
        finally:
            assert client._prepared_queue.active_requests == 1
            assert client._inflight_prefix_owner.counters()["active_producers"] == 1
            client.order.append("runtime-reset")
    client = Client(events)
    token = prepared_cancellation.set(signal)
    try:
        attempt = OwnedPreparedAttempt(client, {})
    finally:
        prepared_cancellation.reset(token)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, attempt)
        assert working.wait(3)
        survivor = prefix(client)
        signal.cancel()
        counters = client._inflight_prefix_owner.counters()
        assert counters["requests"] == 1 and counters["active_producers"] == 1
        assert counters["reserved_bytes"] == RESOURCES.producer_bytes(2) + RESOURCES.branch_bytes(2)
        assert not attempt.prefix.producer_cancel.is_set()
        finish.set()
        with pytest.raises(PreparedQueueClosed):
            future.result(timeout=3)
    assert not decoded
    survivor.wait(timeout=0).close()
    assert_retired(client, signal)


def test_attempt_capacity_bounds_initialization_before_ready_queue():
    initialized = []
    def events(body):
        initialized.append(body)
        yield "initialized"
    client = Client(events, maximum=2)
    attempts = [OwnedPreparedAttempt(client, i) for i in range(3)]
    assert next(attempts[0]) == next(attempts[1]) == "initialized"
    assert client._prepared_queue.requests == 2 and client._prepared_queue.active_requests == 0
    with pytest.raises(RuntimeError, match="capacity"):
        next(attempts[2])
    assert initialized == [0, 1]
    for attempt in attempts:
        attempt.close()
    assert_retired(client)
