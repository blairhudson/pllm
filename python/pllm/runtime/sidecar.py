from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import OpenAI, ResponseStream
from .responses import sse_done, sse_event


def create_sidecar_app(
    *,
    remote_base_url: str,
    remote_api_key: str,
    local_api_key: str = "local",
    client: OpenAI | None = None,
    **client_kwargs: Any,
) -> FastAPI:
    owned = client is None
    runtime_client = client or OpenAI(base_url=remote_base_url, api_key=remote_api_key, **client_kwargs)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            core = getattr(runtime_client, "_core", None)
            model = getattr(core, "default_model", None)
            if core is not None and model and getattr(core, "preparation_http", None):
                runtime_client.preprocess(model, count=max(core.prepared_inventory_rows, 256))
            yield
        finally:
            if owned:
                runtime_client.close()

    app = FastAPI(title="PLLM Client Sidecar", version="0.14.0", lifespan=lifespan)

    def auth(value: str | None) -> None:
        if value != f"Bearer {local_api_key}":
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "invalid API key", "type": "authentication_error"}},
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)):
        auth(authorization)
        return runtime_client.models.list()

    @app.post("/v1/responses")
    async def responses(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        body = await request.json()
        result = runtime_client.responses.create(**body)
        if isinstance(result, ResponseStream):

            def generate():
                try:
                    for event in result:
                        yield sse_event(event)
                    yield sse_done()
                finally:
                    result.close()

            return StreamingResponse(generate(), media_type="text/event-stream")
        return JSONResponse(result.to_dict())

    @app.post("/v1/preprocess")
    async def preprocess(request: Request, authorization: str | None = Header(default=None)):
        auth(authorization)
        body = await request.json()
        core = getattr(runtime_client, "_core", None)
        model = body.get("model") or getattr(core, "default_model", None)
        if not isinstance(model, str) or not model:
            raise HTTPException(status_code=400, detail={"error": {"message": "model is required"}})
        count = int(body.get("count") or getattr(core, "prepared_inventory_rows", 64))
        return JSONResponse(runtime_client.preprocess(model, count=count))

    @app.get("/v1/responses/{response_id}")
    async def retrieve(response_id: str, authorization: str | None = Header(default=None)):
        auth(authorization)
        return JSONResponse(runtime_client.responses.retrieve(response_id).to_dict())

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel(response_id: str, authorization: str | None = Header(default=None)):
        auth(authorization)
        return JSONResponse(runtime_client.responses.cancel(response_id).to_dict())

    from .telemetry import instrument_fastapi

    instrument_fastapi(app)
    return app
