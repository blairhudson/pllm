"""Explicit lifetime for synchronous HTTP responses owned by the runtime."""
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import httpx


@contextmanager
def owned_response(response: httpx.Response) -> Iterator[httpx.Response]:
    """Retire a consumed response's transport without waiting for cyclic GC.

    HTTPX binds its stream back to the response. Even a closed stream can retain
    both request and reply bodies through that cycle. Close the real transport
    first, then replace the public stream with an empty, already-retired body.
    Buffered content, headers, status and elapsed time remain readable. Returned
    body bytes still belong to their caller; this does not erase those bytes.
    """
    try:
        yield response
    finally:
        try:
            response.close()
        finally:
            if response.is_closed:
                response.stream = httpx.ByteStream(b"")
