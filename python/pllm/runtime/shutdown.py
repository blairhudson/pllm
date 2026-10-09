"""Defer repeated termination signals until owned resources have retired."""
from __future__ import annotations

import signal
import threading
from contextlib import contextmanager


@contextmanager
def defer_termination_signals():
    # Only the main thread can install Python signal handlers. Nested cleanup
    # scopes forward their pending signal to the outer scope on exit.
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    pending = []

    def defer(signum, _frame):
        if not pending:
            pending.append(signum)

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in previous:
            signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            signal.raise_signal(pending[0])
