from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx as _standard_httpx

try:
    import httpx2 as _httpx  # OpenAI Python 3.x transport package
except ImportError:  # pragma: no cover
    _httpx = _standard_httpx

from .client import HEAPIError, HEClientCore
from .responses import sse_done, sse_event


def _request_httpx(request: Any) -> Any:
    if isinstance(request, _standard_httpx.Request):
        return _standard_httpx
    return _httpx


def _sync_stream(http: Any, iterator: Iterator[bytes]) -> Any:
    class Stream(http.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            return iterator

        def close(self) -> None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    return Stream()


def _async_stream(http: Any, iterator: AsyncIterator[bytes]) -> Any:
    class Stream(http.AsyncByteStream):
        def __aiter__(self) -> AsyncIterator[bytes]:
            return iterator

        async def aclose(self) -> None:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
                return
            sync_close = getattr(iterator, "close", None)
            if sync_close is not None:
                sync_close()

    return Stream()


class HETransport(_httpx.BaseTransport):
    """Custom transport for the official OpenAI Python client.

    The public call remains `client.responses.create(...)`. The transport
    intercepts the request before its plaintext body reaches the network and
    runs the private client protocol instead. Non-Responses routes are forwarded
    to the gateway.
    """

    def __init__(
        self,
        *,
        gateway_url: str,
        api_key: str = "he-local",
        preparation_url: str | None = None,
        preparation_api_key: str | None = None,
        he_transport: str = "http",
        correlation_mode: str = "bfv",
        correlation_prefetch: int = 4,
        prepared_inventory_rows: int = 64,
        token_cache_size: int = 512,
        bundle_cache_mode: str = "read-write",
        bundle_cache_dir: str | None = None,
        tenseal_path: str | None = None,
    ) -> None:
        self.core = HEClientCore(
            base_url=gateway_url,
            api_key=api_key,
            preparation_base_url=preparation_url,
            preparation_api_key=preparation_api_key,
            he_transport=he_transport,
            correlation_mode=correlation_mode,
            correlation_prefetch=correlation_prefetch,
            prepared_inventory_rows=prepared_inventory_rows,
            token_cache_size=token_cache_size,
            bundle_cache_mode=bundle_cache_mode,
            bundle_cache_dir=bundle_cache_dir,
            tenseal_path=tenseal_path,
        )
        self.forward = _httpx.Client(base_url=gateway_url, timeout=300.0)

    def preprocess(self, model: str, *, count: int | None = None) -> dict[str, Any]:
        """Build a READY public inventory before sending Responses requests."""
        target = count if count is not None else self.core.prepared_inventory_rows
        return self.core.preprocess(model, count=target)

    def handle_request(self, request: Any) -> Any:
        response_httpx = _request_httpx(request)
        path = request.url.path
        try:
            if request.method == "POST" and path.rstrip("/") == "/v1/responses":
                body = json.loads(request.read() or b"{}")
                if body.get("stream"):
                    def chunks() -> Iterator[bytes]:
                        for event in self.core.events(body):
                            yield sse_event(event)
                        yield sse_done()
                    return response_httpx.Response(
                        200,
                        headers={"Content-Type": "text/event-stream", "X-HE-Transport": "1"},
                        stream=_sync_stream(response_httpx, chunks()),
                        request=request,
                    )
                result = self.core.create(body)
                return response_httpx.Response(
                    200, json=result.to_dict(), headers={"X-HE-Transport": "1"}, request=request,
                )
            if request.method == "GET" and path.startswith("/v1/responses/"):
                response_id = path.rsplit("/", 1)[-1]
                return response_httpx.Response(200, json=self.core.retrieve(response_id).to_dict(), request=request)
            if request.method == "POST" and path.startswith("/v1/responses/") and path.endswith("/cancel"):
                response_id = path.split("/")[-2]
                return response_httpx.Response(200, json=self.core.cancel(response_id).to_dict(), request=request)
            return self._forward(request)
        except HEAPIError as exc:
            return response_httpx.Response(
                exc.status_code,
                json={"error": {"message": str(exc), "type": "invalid_request_error", "code": "he_error"}},
                request=request,
            )

    def _forward(self, request: Any) -> Any:
        response_httpx = _request_httpx(request)
        headers = {key: value for key, value in request.headers.items() if key.lower() != "authorization"}
        # The public SDK credential is a local placeholder. Control-plane calls
        # must use the remote gateway credential held by the HE transport.
        headers["Authorization"] = f"Bearer {self.core.api_key}"
        response = self.forward.request(
            request.method,
            request.url.path,
            params=request.url.query.decode() if isinstance(request.url.query, bytes) else request.url.query,
            headers=headers,
            content=request.read(),
        )
        return response_httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

    def close(self) -> None:
        self.core.close()
        self.forward.close()


class HEAsyncTransport(_httpx.AsyncBaseTransport):
    def __init__(self, **kwargs: Any) -> None:
        self.sync = HETransport(**kwargs)

    async def handle_async_request(self, request: Any) -> Any:
        response_httpx = _request_httpx(request)
        content = await request.aread()
        path = request.url.path
        if request.method == "POST" and path.rstrip("/") == "/v1/responses":
            body = json.loads(content or b"{}")
            if body.get("stream"):
                async def chunks() -> AsyncIterator[bytes]:
                    iterator = self.sync.core.events(body)
                    sentinel = object()
                    while True:
                        event = await asyncio.to_thread(_next_or, iterator, sentinel)
                        if event is sentinel:
                            break
                        yield sse_event(event)
                    yield sse_done()
                return response_httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream", "X-HE-Transport": "1"},
                    stream=_async_stream(response_httpx, chunks()),
                    request=request,
                )
            result = await asyncio.to_thread(self.sync.core.create, body)
            return response_httpx.Response(200, json=result.to_dict(), request=request)
        sync_request = _httpx.Request(request.method, request.url, headers=request.headers, content=content)
        response = await asyncio.to_thread(self.sync.handle_request, sync_request)
        return response_httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.read(),
            request=request,
        )

    async def aclose(self) -> None:
        await asyncio.to_thread(self.sync.close)

    async def preprocess(
        self,
        model: str,
        *,
        count: int | None = None,
        stages: list[str] | None = None,
    ) -> dict[str, Any]:
        target = count if count is not None else self.sync.core.prepared_inventory_rows
        return await asyncio.to_thread(
            self.sync.core.preprocess,
            model,
            count=target,
            stages=stages,
        )


def _next_or(iterator: Iterator[Any], sentinel: Any) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel
