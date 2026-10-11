"""Bounded same-client scopes; online execution and Preparation never overlap.

Scopes reserve a seat before private prefix lookup. Only ready scopes enter the
FIFO: a prefix waiter cannot block the producer it is waiting for. Shutdown wakes
queued callers without releasing the still-running caller's seat.
"""
from __future__ import annotations

from collections import deque
from contextvars import ContextVar, copy_context
import threading


class PreparedQueueClosed(RuntimeError):
    pass


class PreparedCancellation:
    """Cross-thread cancellation reaches a waiter before a response ID exists."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self._lock = threading.Lock()
        self._attempts: set[OwnedPreparedAttempt] = set()

    def register(self, attempt) -> None:
        with self._lock:
            self._attempts.add(attempt)
            cancelled = self.event.is_set()
        if cancelled:
            attempt.cancel()

    def unregister(self, attempt) -> None:
        with self._lock:
            self._attempts.discard(attempt)

    def cancel(self) -> None:
        self.event.set()
        with self._lock:
            attempts = tuple(self._attempts)
        for attempt in attempts:
            attempt.cancel()


prepared_cancellation: ContextVar[PreparedCancellation | None] = ContextVar(
    "pllm_prepared_cancellation", default=None)
prepared_attempt: ContextVar[OwnedPreparedAttempt | None] = ContextVar(
    "pllm_prepared_attempt", default=None)


class OwnedPreparedAttempt:
    """One generator, tokenizer context, queue seat and temporary prefix claim."""

    def __init__(self, client, body) -> None:
        self.client = client
        self.context = copy_context()
        self.lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._done = threading.Event()
        self.cancelled = threading.Event()
        self.cancellation = prepared_cancellation.get() or PreparedCancellation()
        self.scope: PreparedScope | None = None
        self.prefix = None
        self.published = False
        self._prefix_finished = False
        self.closed = False
        self.source = self._events(body)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            if self.closed:
                raise StopIteration
            try:
                return self.context.run(next, self.source)
            except StopIteration:
                # Exhaustion already ran inner reset and our retirement finally.
                # Successful completion is not cancellation and must not replay
                # fail/finish/release on a retired prefix claim.
                self.closed = True
                raise
            except BaseException:
                self.close()
                raise

    def _events(self, body):
        owner = self.client
        with owner._prepared_attempt_lock:
            if owner._closing:
                raise PreparedQueueClosed("prepared client closed")
            self.scope = owner._prepared_queue.admit(retain_until_close=True)
            owner._prepared_attempts.add(self)
        token = prepared_attempt.set(self)
        try:
            self.cancellation.register(self)
            self.check_cancelled()
            with owner.tokenizer_scope():
                yield from owner._events(body)
        finally:
            # The inner executor retires its runtime before returning here.
            # Neither cancellation nor shutdown may erase a borrowed live slab.
            try:
                try:
                    self.finish_prefix()
                finally:
                    if self.prefix is not None:
                        self.prefix.release()
            finally:
                try:
                    self.scope.close()
                finally:
                    try:
                        with owner._prepared_attempt_lock:
                            owner._prepared_attempts.discard(self)
                    finally:
                        try:
                            self.cancellation.unregister(self)
                        finally:
                            prepared_attempt.reset(token)
                            self._done.set()

    def bind_prefix(self, request) -> None:
        with self._state_lock:
            if self._done.is_set() or self.closed or self.prefix is not None:
                raise RuntimeError("prepared prefix claim already bound or retired")
            self.prefix = request
            if self.cancelled.is_set() and request is not None:
                request.cancel()

    def finish_prefix(self) -> None:
        """Idempotent producer hook AFTER original runtime reset/copy erasure.

        Set ``published = True`` immediately after successful publication. Parent
        may invoke this hook early; wrapper invokes it again at final retirement.
        Branch claim stays reserved until inner executor exits and release runs.
        """
        with self._state_lock:
            if self.prefix is None or not self.prefix.is_producer or self._prefix_finished:
                return
            self._prefix_finished = True
            try:
                if not self.published:
                    self.prefix.fail()
            finally:
                self.prefix.finish_producer()

    def _defer_cancel(self) -> bool:
        # Read-only predicate: queue invokes this under its condition lock.
        # Do not acquire attempt lock or prefix owner lock here (lock inversion).
        return (self.prefix is not None and self.prefix.is_producer
                and not self.published and not self._prefix_finished
                and not self.prefix.producer_cancel.is_set())

    def is_set(self) -> bool:
        """Queue cancellation view; preserve pending producer with live waiters."""
        return self.client._closing or self.cancelled.is_set() and not self._defer_cancel()

    def check_cancelled(self) -> None:
        if self.client._closing:
            raise PreparedQueueClosed("prepared client closed")
        if self.cancelled.is_set():
            # Finish an already shared producer prefill for its surviving
            # callers; stop the departing caller before any private decode.
            if self._defer_cancel():
                return
            raise PreparedQueueClosed("prepared request cancelled")

    def enter(self) -> None:
        self.check_cancelled()
        # Dynamic view matters: last surviving waiter can depart while producer
        # is queued. Passing raw cancelled Event would kill shared work early.
        self.scope.enter(cancelled=self)
        self.check_cancelled()

    def cancel(self) -> None:
        try:
            with self._state_lock:
                if self._done.is_set() or self.closed or self.cancelled.is_set():
                    return
                self.cancelled.set()
                if self.prefix is not None:
                    self.prefix.cancel()
        finally:
            self.client._prepared_queue.wake()

    def close(self) -> None:
        self.cancel()
        with self.lock:
            if self.closed:
                return
            try:
                # A suspended streaming producer may not have executed prefill
                # yet. Drain only its shared prefix, never its private decode.
                while not self._done.is_set() and self._defer_cancel():
                    try:
                        self.context.run(next, self.source)
                    except (StopIteration, PreparedQueueClosed):
                        break
            finally:
                self.closed = True
                try:
                    self.context.run(self.source.close)
                finally:
                    self._done.set()


class PreparedExecutionQueue:
    def __init__(self, max_requests: int = 8) -> None:
        if type(max_requests) is not int or not 1 <= max_requests <= 64:
            raise ValueError("invalid prepared request capacity")
        self._maximum = max_requests
        self._condition = threading.Condition()
        self._scopes: set[PreparedScope] = set()
        self._ready: deque[PreparedScope] = deque()
        self._active: PreparedScope | None = None
        self._closed = False
        self._peak = 0

    def admit(self, *, retain_until_close: bool = False) -> PreparedScope:
        """Reserve initialization capacity, before prefix lookup or ready enqueue.

        Wrapper-owned scopes retain even failed-entry seats until inner cleanup
        calls close. Legacy standalone queued scopes still retire on entry error.
        """
        if type(retain_until_close) is not bool:
            raise ValueError("invalid prepared scope retirement policy")
        with self._condition:
            if self._closed:
                raise PreparedQueueClosed("prepared client closed")
            if len(self._scopes) >= self._maximum:
                raise RuntimeError("prepared concurrent request bound/capacity exceeded")
            scope = PreparedScope(self, retain_until_close=retain_until_close)
            self._scopes.add(scope)
            self._peak = max(self._peak, len(self._scopes))
            return scope

    def _enter(self, scope: PreparedScope,
               cancelled: threading.Event | OwnedPreparedAttempt | None) -> None:
        with self._condition:
            if scope not in self._scopes:
                raise RuntimeError("prepared request scope retired")
            if scope._entered:
                raise RuntimeError("prepared request scope already entered/consumed")
            scope._entered = True
            self._ready.append(scope)
            try:
                while True:
                    if self._closed or cancelled is not None and cancelled.is_set():
                        raise PreparedQueueClosed("prepared queued request cancelled")
                    if scope not in self._scopes:
                        raise PreparedQueueClosed("prepared queued request retired")
                    if self._active is None and self._ready[0] is scope:
                        self._ready.popleft()
                        self._active = scope
                        return
                    self._condition.wait(timeout=0.05)
            except BaseException:
                if scope._retain_until_close:
                    # Initialization may already own a runtime or material. The
                    # wrapper retires this seat only AFTER inner reset finishes.
                    if scope in self._ready:
                        self._ready.remove(scope)
                    self._condition.notify_all()
                else:
                    self._release(scope)
                raise

    def _release(self, scope: PreparedScope) -> None:
        with self._condition:
            if scope not in self._scopes:
                return
            self._scopes.remove(scope)
            if scope is self._active:
                self._active = None
            if scope in self._ready:
                self._ready.remove(scope)
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def wake(self) -> None:
        """Wake cancellation checks without releasing any active scope."""
        with self._condition:
            self._condition.notify_all()

    @property
    def requests(self) -> int:
        with self._condition:
            return len(self._scopes)

    @property
    def reserved_requests(self) -> int:
        return self.requests

    @property
    def active_requests(self) -> int:
        with self._condition:
            return int(self._active is not None)

    @property
    def peak_requests(self) -> int:
        with self._condition:
            return self._peak


class PreparedScope:
    def __init__(self, owner: PreparedExecutionQueue, *, retain_until_close: bool = False) -> None:
        self._owner = owner
        self._entered = False
        self._retain_until_close = retain_until_close

    def enter(self, cancelled: threading.Event | OwnedPreparedAttempt | None = None) -> None:
        self._owner._enter(self, cancelled)

    def close(self) -> None:
        self._owner._release(self)

    def __enter__(self) -> PreparedScope:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def __copy__(self):
        raise TypeError("prepared scopes cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("prepared scopes cannot be copied")
