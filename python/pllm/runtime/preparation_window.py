"""Bounded independent offline work; cancellation precedes joining issued jobs."""
from __future__ import annotations

import contextvars
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

WINDOW_BYTES = 16 << 20


def run_preparation_window(items, *, width, cost, prepare, accept, abort, discard, cancelled,
                           max_bytes=WINDOW_BYTES):
    if type(width) is not int or not 1 <= width <= 4:
        raise ValueError("preparation window must be in [1, 4]")
    if (type(max_bytes) is not int or not 1 << 20 <= max_bytes <= 64 << 20
            or (width == 1 and max_bytes != WINDOW_BYTES)):
        raise ValueError("preparation work window requires 1–64 MiB and concurrent stages")
    if width == 1:
        value = None
        try:
            for item in items:
                if cancelled():
                    raise RuntimeError("preparation cancelled before issuance")
                value = prepare(item)
                if cancelled():
                    raise RuntimeError("preparation cancelled before acknowledgement admission")
                accept(value)
                value = None
        except BaseException:
            try:
                abort()
            finally:
                if value is not None:
                    discard(value)
            raise
        return
    weighted = [(item, cost(item)) for item in items]
    if any(type(size) is not int or not 0 < size <= max_bytes for _, size in weighted):
        raise ValueError("preparation job exceeds the declared work window")
    pool = ThreadPoolExecutor(max_workers=width, thread_name_prefix="pllm-preparation")
    pending, next_index, retained = {}, 0, 0
    try:
        while pending or next_index < len(weighted):
            if cancelled():
                raise RuntimeError("preparation cancelled during issuance")
            while len(pending) < width and next_index < len(weighted):
                item, size = weighted[next_index]
                if retained + size > max_bytes:
                    break
                pending[pool.submit(contextvars.copy_context().run, prepare, item)] = size
                retained += size
                next_index += 1
            ready, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            for future in ready:
                value = future.result()
                if cancelled():
                    raise RuntimeError("preparation cancelled before acknowledgement admission")
                accept(value)
                retained -= pending.pop(future)
    except BaseException:
        for future in pending:
            future.cancel()
        try:
            abort()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for future in pending:
                if not future.cancelled() and future.exception() is None:
                    discard(future.result())
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
