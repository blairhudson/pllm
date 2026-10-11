"""Lifecycle checks for queued caller admission around one-use prepared work."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from pllm.runtime.prepared_queue import PreparedExecutionQueue, PreparedQueueClosed


def test_waiting_prefix_caller_does_not_block_its_later_producer():
    queue = PreparedExecutionQueue(2)
    with queue.admit() as waiter, queue.admit() as producer:
        producer.enter()
        assert queue.active_requests == 1 and queue.reserved_requests == 2
        producer.close()
        waiter.enter()
        assert queue.active_requests == 1
    assert queue.reserved_requests == queue.active_requests == 0


def test_active_execution_keeps_its_slot_through_shutdown():
    queue = PreparedExecutionQueue(2)
    with queue.admit() as active:
        active.enter()
        ready = threading.Event()

        def waiting():
            with queue.admit() as ticket:
                ready.set()
                ticket.enter()

        with ThreadPoolExecutor(max_workers=1) as workers:
            future = workers.submit(waiting)
            assert ready.wait(2)
            queue.close()
            with pytest.raises(PreparedQueueClosed):
                future.result(timeout=2)
        assert queue.active_requests == queue.reserved_requests == 1
        with pytest.raises(PreparedQueueClosed):
            queue.admit()
    assert queue.active_requests == queue.reserved_requests == 0


def test_cancelled_waiter_retires_without_releasing_another_execution():
    queue = PreparedExecutionQueue(3)
    cancelled = threading.Event()
    with queue.admit() as active, queue.admit() as waiter:
        active.enter()
        cancelled.set()
        with pytest.raises(PreparedQueueClosed):
            waiter.enter(cancelled)
        assert queue.active_requests == queue.reserved_requests == 1
        with pytest.raises(RuntimeError, match="retired"):
            waiter.enter()
    assert queue.reserved_requests == 0


def test_admission_bounds_pending_callers_and_releases_failed_setup():
    queue = PreparedExecutionQueue(1)
    with pytest.raises(ValueError):
        with queue.admit():
            with pytest.raises(RuntimeError, match="bound"):
                queue.admit()
            raise ValueError("setup failed before execution")
    with queue.admit() as ticket:
        ticket.enter()
        with pytest.raises(RuntimeError, match="already entered"):
            ticket.enter()
    assert queue.peak_requests == 1 and queue.reserved_requests == 0


def test_ready_callers_exclude_one_another():
    queue = PreparedExecutionQueue(2)
    waiting = threading.Event()
    entered = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as workers:
        with queue.admit() as first:
            first.enter()

            def second():
                with queue.admit() as ticket:
                    waiting.set()
                    ticket.enter()
                    entered.set()

            future = workers.submit(second)
            assert waiting.wait(2)
            assert not entered.is_set()
        future.result(timeout=2)
    assert entered.is_set() and queue.reserved_requests == 0
