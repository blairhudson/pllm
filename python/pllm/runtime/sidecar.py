from __future__ import annotations

import json
from contextlib import asynccontextmanager
from collections.abc import Iterator
from typing import Any, cast

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from .client import OpenAI, ProtocolError, RuntimeClient
from .http_gateway import (
    SSE_DONE,
    GatewayError,
    chat_completion,
    chat_to_response_body,
    compact_response,
    error_envelope,
    exception_envelope,
    integer_timestamps,
    iter_chat_sse,
    resource_dict,
    response_error_sse,
    response_sse_event,
    validate_response_body,
)
from .responses import ResponsesError


def create_sidecar_app(
    *,
    remote_base_url: str,
    remote_api_key: str,
    local_api_key: str = "local",
    client: OpenAI | RuntimeClient | None = None,
    **client_kwargs: Any,
) -> FastAPI:
    owned = client is None
    owner = client or OpenAI(base_url=remote_base_url, api_key=remote_api_key, **client_kwargs)
    runtime_client = cast(RuntimeClient, getattr(owner, "_core", owner))

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            if owned:
                owner.close()

    app = FastAPI(title="PLLM Local Gateway", version="0.14.0", lifespan=lifespan)

    def auth(value: str | None) -> None:
        if value != f"Bearer {local_api_key}":
            raise GatewayError(
                "Invalid API key",
                status_code=401,
                error_type="authentication_error",
                code="invalid_api_key",
            )

    @app.exception_handler(GatewayError)
    async def gateway_error(_: Request, exc: GatewayError):
        return JSONResponse(exc.to_dict(), status_code=exc.status_code)

    @app.exception_handler(ProtocolError)
    async def protocol_error(_: Request, exc: ProtocolError):
        return JSONResponse(exception_envelope(exc), status_code=exc.status_code)

    @app.exception_handler(ResponsesError)
    async def responses_error(_: Request, exc: ResponsesError):
        return JSONResponse(
            error_envelope(str(exc), exc.code, exc.code, param=exc.param),
            status_code=400,
        )

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse(
            error_envelope(str(exc), "invalid_request_error", "invalid_request_error"),
            status_code=400,
        )

    @app.exception_handler(Exception)
    async def internal_error(_: Request, exc: Exception):
        return JSONResponse(exception_envelope(exc), status_code=500)

    async def json_body(request: Request) -> dict[str, Any]:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/json":
            raise GatewayError(
                "Content-Type must be application/json",
                status_code=415,
                code="unsupported_media_type",
            )
        try:
            value = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise GatewayError("Malformed JSON body", code="invalid_json") from None
        if not isinstance(value, dict):
            raise GatewayError("Request body must be a JSON object")
        return value

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)):
        auth(authorization)
        value = await run_in_threadpool(runtime_client.list_models)
        return JSONResponse(integer_timestamps(value))

    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        body = validate_response_body(await json_body(request))
        _validate_continuation(runtime_client, body)
        if body.get("stream"):
            result = runtime_client.create(body)
            return StreamingResponse(
                _response_stream(result),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        value = resource_dict(await run_in_threadpool(runtime_client.create, body))
        return JSONResponse(value)

    @app.websocket("/v1/responses")
    async def response_websocket(websocket: WebSocket) -> None:
        if websocket.headers.get("authorization") != f"Bearer {local_api_key}":
            await websocket.close(code=4401, reason="Invalid API key")
            return
        await websocket.accept()
        connection_responses: set[str] = set()
        try:
            while True:
                previous_id: str | None = None
                try:
                    request_value = await websocket.receive_json()
                    if (
                        not isinstance(request_value, dict)
                        or request_value.get("type") != "response.create"
                    ):
                        raise GatewayError(
                            "WebSocket messages must have type response.create",
                            param="type",
                        )
                    body = dict(request_value)
                    body.pop("type", None)
                    body = validate_response_body(body)
                    body["stream"] = True
                    previous_value = body.get("previous_response_id")
                    previous_id = str(previous_value) if previous_value is not None else None
                    if previous_id is not None:
                        lookup = getattr(
                            runtime_client, "continuation_response", runtime_client.retrieve
                        )
                        try:
                            previous = resource_dict(lookup(previous_id))
                        except Exception as exc:
                            raise GatewayError(
                                "Previous response not found",
                                status_code=404,
                                param="previous_response_id",
                                code="previous_response_not_found",
                            ) from exc
                        if (
                            not bool(previous.get("store", True))
                            and previous_id not in connection_responses
                        ):
                            raise GatewayError(
                                "Previous response not found",
                                status_code=404,
                                param="previous_response_id",
                                code="previous_response_not_found",
                            )
                    _validate_continuation(runtime_client, body)
                    created_id: str | None = None
                    completed = False
                    source = await run_in_threadpool(runtime_client.create, body)
                    iterator = iter(source)
                    while True:
                        event = await run_in_threadpool(_next_event, iterator)
                        if event is None:
                            break
                        value = resource_dict(event)
                        response = value.get("response")
                        if isinstance(response, dict) and isinstance(response.get("id"), str):
                            created_id = response["id"]
                        await websocket.send_json(value)
                        if value.get("type") in {"response.completed", "response.incomplete"}:
                            completed = True
                    if completed and created_id and not bool(body.get("store", True)):
                        connection_responses.add(created_id)
                except WebSocketDisconnect:
                    raise
                except Exception as exc:
                    if websocket.client_state is WebSocketState.DISCONNECTED:
                        raise WebSocketDisconnect
                    if previous_id is not None and previous_id in connection_responses:
                        runtime_client.evict_response(str(previous_id))
                        connection_responses.discard(str(previous_id))
                    payload = exception_envelope(exc)["error"]
                    await websocket.send_json(
                        {
                            "type": "error",
                            "status": int(getattr(exc, "status_code", 500)),
                            "error": payload,
                        }
                    )
        except WebSocketDisconnect:
            pass
        finally:
            for response_id in connection_responses:
                runtime_client.evict_response(response_id)

    @app.post("/v1/responses/compact")
    async def compact(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        return JSONResponse(compact_response(await json_body(request)))

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        request_body = await json_body(request)
        body = chat_to_response_body(request_body)
        if body.get("stream"):
            result = runtime_client.create(body)
            return StreamingResponse(
                _chat_stream(
                    result,
                    body["model"],
                    include_usage=bool(
                        (request_body.get("stream_options") or {}).get("include_usage")
                    ),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        result = await run_in_threadpool(runtime_client.create, body)
        return JSONResponse(chat_completion(result))

    @app.post("/v1/preprocess")
    async def preprocess(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        body = await json_body(request)
        if set(body) - {"model", "count"}:
            raise GatewayError("Unsupported preprocess field")
        model = body.get("model") or getattr(runtime_client, "default_model", None)
        if not isinstance(model, str) or not model:
            raise GatewayError("model is required", param="model")
        count = int(body.get("count") or getattr(runtime_client, "prepared_inventory_rows", 64))
        value = await run_in_threadpool(runtime_client.preprocess, model, count=count)
        return JSONResponse(value)

    @app.get("/v1/responses/{response_id}")
    async def retrieve(response_id: str, authorization: str | None = Header(default=None)):
        auth(authorization)
        try:
            value = resource_dict(await run_in_threadpool(runtime_client.retrieve, response_id))
        except KeyError as exc:
            raise GatewayError("Response not found", status_code=404, code="not_found") from exc
        if value.get("store") is False:
            raise GatewayError("Response not found", status_code=404, code="not_found")
        return JSONResponse(value)

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel(response_id: str, authorization: str | None = Header(default=None)):
        auth(authorization)
        value = await run_in_threadpool(runtime_client.cancel, response_id)
        return JSONResponse(resource_dict(value))

    from .telemetry import instrument_fastapi

    instrument_fastapi(app)
    return app


def _validate_continuation(runtime_client: Any, body: dict[str, Any]) -> None:
    previous_id = body.get("previous_response_id")
    if previous_id is None:
        return
    try:
        lookup = getattr(runtime_client, "continuation_response", runtime_client.retrieve)
        previous = resource_dict(lookup(str(previous_id)))
    except Exception as exc:
        raise GatewayError(
            "Previous response not found",
            status_code=404,
            param="previous_response_id",
            code="previous_response_not_found",
        ) from exc
    call_ids = {
        str(item.get("call_id"))
        for item in previous.get("output", [])
        if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id")
    }
    value = body.get("input")
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if not isinstance(item, dict) or item.get("type") != "function_call_output":
            continue
        if str(item.get("call_id")) not in call_ids:
            raise GatewayError(
                "function_call_output does not match the previous response",
                param=f"input.{index}.call_id",
                code="invalid_function_call_output",
            )


def _next_event(iterator: Iterator[Any]) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _response_stream(stream: Any, *, on_complete: Any = None) -> Iterator[bytes]:
    sequence = 0
    response_id: str | None = None
    terminal = False
    try:
        for event in stream:
            value = resource_dict(event)
            response = value.get("response")
            if isinstance(response, dict) and isinstance(response.get("id"), str):
                response_id = response["id"]
            if value.get("type") in {
                "response.completed",
                "response.incomplete",
                "response.failed",
                "error",
            }:
                terminal = True
            sequence = max(sequence, int(value.get("sequence_number", sequence))) + 1
            yield response_sse_event(value)
    except Exception as exc:
        terminal = True
        yield response_error_sse(exc, sequence)
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
        if on_complete is not None and response_id is not None:
            on_complete(response_id)
    if not terminal:
        yield response_error_sse(
            RuntimeError("Response stream ended without a terminal event"), sequence
        )
    yield SSE_DONE


def _chat_stream(stream: Any, model: str, *, include_usage: bool) -> Iterator[bytes]:
    try:
        yield from iter_chat_sse(stream, model, include_usage=include_usage)
    except Exception as exc:
        payload = exception_envelope(exc)
        yield f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    yield SSE_DONE
