from __future__ import annotations

import asyncio
import base64
import json
import secrets
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import msgpack
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response as FastAPIResponse, StreamingResponse

from .backends import BackendRegistry
from .config import GatewayConfig
from .engine import HEEngine
from .engine import EngineStageExecutor
from .hf_hub import resolve_huggingface_source
from .he_runtime import (
    BFVCorrelationServer,
    BigramStageExecutor,
    CorrelationPool,
    LocalCorrelationFactory,
    MaskCorrelation,
    MaskedBigramModel,
    StageBatchScheduler,
)
from .loaders import load_gguf, load_hf_directory, load_mlx_directory, load_ollama_model
from .models import ModelManifest as ImportedModelManifest
from .protocol import (
    BINARY_MEDIA_TYPE, HEEnvelope, ProtocolError, ReplayWindow, encode_length_prefixed,
    iter_length_prefixed, pack_envelope, sse_event, unpack_envelope,
)
from .stage_protocol import blinded_correlation_to_wire, correlation_to_wire
from .responses import ResponsesError
from .security import bearer_token, derive_session_key
from .types import new_id


@dataclass(slots=True)
class HESession:
    id: str
    response_id: str
    model_id: str
    api_key: str
    created_at: float = field(default_factory=time.time)
    context_id: str | None = None
    context_ids: set[str] = field(default_factory=set)
    replay: ReplayWindow = field(default_factory=ReplayWindow)
    canceled: bool = False
    completed: bool = False
    online_steps: int = 0
    correlation_steps: int = 0

    @property
    def key(self) -> bytes:
        return derive_session_key(self.api_key, self.id)


class RetainedResponses:
    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self.values: dict[str, tuple[float, dict[str, Any]]] = {}

    def put(self, value: dict[str, Any]) -> None:
        self.values[str(value["id"])] = (time.time(), value)
        self._expire()

    def get(self, response_id: str) -> dict[str, Any]:
        self._expire()
        if response_id not in self.values:
            raise KeyError(response_id)
        return self.values[response_id][1]

    def cancel(self, response_id: str) -> dict[str, Any]:
        value = dict(self.get(response_id))
        value["status"] = "cancelled"
        self.put(value)
        return value

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for key, (created, _) in list(self.values.items()):
            if created < cutoff:
                del self.values[key]


def create_app(
    config: GatewayConfig | None = None,
    *,
    backend_registry: BackendRegistry | None = None,
    strict_models: dict[str, MaskedBigramModel] | None = None,
    engines: dict[str, HEEngine] | None = None,
    audit_hook: Callable[[str, bytes], None] | None = None,
) -> FastAPI:
    config = config or GatewayConfig()
    backend_registry = backend_registry or BackendRegistry.from_config(list(config.backends))
    engines = engines or {}
    if strict_models is None:
        alphabet = config.reference_alphabet.replace("\\n", "\n")
        strict_models = {
            config.reference_model_id: MaskedBigramModel(
                model_id=config.reference_model_id,
                alphabet=alphabet,
                phrase=config.reference_phrase,
            )
        }
    schedulers = {
        model_id: StageBatchScheduler(
            BigramStageExecutor(model),
            max_batch_size=config.max_batch_size,
            max_wait_ms=config.max_batch_wait_ms,
            adaptive_wait=config.adaptive_batching,
        )
        for model_id, model in strict_models.items()
    }
    engine_schedulers: dict[tuple[str, str], StageBatchScheduler] = {}
    bfv_servers: dict[str, BFVCorrelationServer] = {}
    context_owners: dict[tuple[str, str], str] = {}
    proprietary_owner_by_principal: dict[tuple[str, str], str] = {}
    for model_id, model in strict_models.items():
        try:
            bfv_servers[model_id] = BFVCorrelationServer(model.weight, pydeps_path=config.tenseal_path)
        except ImportError:
            # TenSEAL is an optional runtime dependency. Clear passthrough and
            # local-test protocol validation still work without it.
            pass
    local_factories = {
        model_id: LocalCorrelationFactory(model.weight, model.modulus, seed=17)
        for model_id, model in strict_models.items()
    }
    sessions: dict[str, HESession] = {}
    responses = RetainedResponses(config.response_retention_seconds)
    imported: dict[str, ImportedModelManifest] = {}
    model_engine_routes: dict[str, str] = {}
    backend_models: list[dict[str, Any]] = []

    def bind_proprietary_owner(session: HESession, owner_id: str) -> None:
        key = (session.model_id, session.api_key)
        existing = proprietary_owner_by_principal.get(key)
        if existing is None:
            proprietary_owner_by_principal[key] = owner_id
        elif existing != owner_id:
            raise ValueError(
                "proprietary owner identity is already bound to this API principal and model"
            )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        nonlocal backend_models
        for source in config.engine_models:
            body = dict(source)
            engine_name = str(body.pop("engine", "masked-transformer-w4a4"))
            manifest = await inspect_source(body)
            await register_engine_model(engine_name, manifest)
        if backend_registry.adapters:
            try:
                backend_models = [row.to_model_object() for row in await backend_registry.refresh()]
            except Exception as exc:
                backend_models = [{
                    "id": "backend-discovery-error",
                    "object": "model",
                    "owned_by": "pllm",
                    "he": {"error": str(exc), "privacy_mode": "trusted_backend"},
                }]
        try:
            yield
        finally:
            for scheduler in schedulers.values():
                await scheduler.close()
            for scheduler in engine_schedulers.values():
                await scheduler.close()
            for adapter in backend_registry.adapters.values():
                close = getattr(adapter, "close", None)
                if close:
                    await close()
            for engine in engines.values():
                close = getattr(engine, "close", None)
                if close:
                    value = close()
                    if asyncio.iscoroutine(value):
                        await value

    app = FastAPI(title="Private LLM Inference Gateway", version="0.14.0", lifespan=lifespan)
    app.state.config = config
    app.state.strict_models = strict_models
    app.state.schedulers = schedulers
    app.state.sessions = sessions
    app.state.imported_manifests = imported
    app.state.backend_registry = backend_registry
    app.state.engines = engines
    app.state.engine_schedulers = engine_schedulers

    def audit(kind: str, payload: bytes) -> None:
        if audit_hook is not None:
            audit_hook(kind, payload)

    def auth_token(value: str | None) -> str:
        token = bearer_token(value)
        if token is None or (config.api_keys and token not in config.api_keys):
            raise HTTPException(
                status_code=401,
                detail={"error": {"message": "Invalid API key", "type": "authentication_error", "code": "invalid_api_key"}},
            )
        return token

    @app.get("/healthz")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "privacy_mode": config.privacy_mode,
            "strict_he_models": len(strict_models),
            "trusted_backends": len(backend_registry.adapters),
            "he_engines": len(engines),
            "loaded_engine_models": len(model_engine_routes),
            "sessions": len(sessions),
        }

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        strict = [model.manifest.to_model_object() for model in strict_models.values()]
        manifests = []
        for manifest in imported.values():
            engine_name = model_engine_routes.get(manifest.id)
            manifests.append({
                "id": manifest.id,
                "object": "model",
                "created": int(manifest.created_at),
                "owned_by": "pllm",
                "he": {
                    "privacy_mode": (
                        manifest.metadata.get("privacy_mode", "strict_he_engine")
                        if engine_name else "manifest_only"
                    ),
                    "privacy_protocol": manifest.metadata.get("privacy_protocol"),
                    "engine": engine_name,
                    "source_format": manifest.source_format,
                    "architecture": manifest.architecture,
                    "stage_count": len(manifest.stages),
                    "fingerprint": manifest.fingerprint,
                    "body_fingerprint": manifest.metadata.get("body_fingerprint"),
                },
            })
        return {"object": "list", "data": strict + manifests + backend_models}

    @app.get("/v1/he/capabilities")
    async def capabilities(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        return {
            "object": "he.capabilities",
            "protocol": "he-responses/1",
            "responses_api": True,
            "client_transports": ["http-binary", "websocket-binary", "local-sidecar"],
            "server_privacy_mode": config.privacy_mode,
            "correlation_modes": (
                ["bfv-exact"] + (["local-test"] if config.allow_insecure_local_correlations else [])
                if config.privacy_mode == "public"
                else []
            ),
            "strict_models": [model.manifest.to_model_object() for model in strict_models.values()],
            "backend_adapters": {
                name: adapter.capabilities.to_dict()
                for name, adapter in backend_registry.adapters.items()
            },
            "engine_plugin_contract": {
                "manifest": "ModelManifest",
                "stage_frame": "length-prefixed opaque binary payloads",
                "supports": ["load", "unload", "execute_stage"],
            },
            "engines": {name: engine.capabilities.to_dict() for name, engine in engines.items()},
        }

    async def inspect_source(body: dict[str, Any]) -> ImportedModelManifest:
        kind = str(body.get("kind", "huggingface"))
        model_id = body.get("model_id")
        if kind in {"huggingface", "safetensors", "vllm"}:
            source = str(body.get("repo_id") or body.get("path") or "")
            if not source:
                raise HTTPException(status_code=400, detail={"error": {"message": "path or repo_id is required"}})
            local_path = Path(source).expanduser()
            if local_path.exists():
                # Keep generic engine-plugin support for local, config-only fixtures.
                # Strict Transformer engines validate their required tensor files at load time.
                return load_hf_directory(str(local_path.resolve()), model_id=model_id)
            resolved = resolve_huggingface_source(
                source,
                model_id=model_id,
                revision=body.get("revision"),
                token=body.get("token"),
                cache_dir=body.get("cache_dir"),
                local_files_only=bool(body.get("local_files_only", False)),
            )
            return load_hf_directory(str(resolved.path), model_id=resolved.model_id)
        if kind in {"mlx", "mlx-lm"}:
            return load_mlx_directory(str(body["path"]), model_id=model_id)
        if kind in {"gguf", "llama.cpp"}:
            return load_gguf(str(body["path"]), model_id=model_id)
        if kind == "ollama":
            return await load_ollama_model(
                str(body.get("base_url", "http://127.0.0.1:11434")),
                str(body["name"]),
                api_key=body.get("api_key"),
            )
        raise HTTPException(status_code=400, detail={"error": {"message": f"Unsupported model source {kind}"}})

    async def register_engine_model(engine_name: str, manifest: ImportedModelManifest) -> ImportedModelManifest:
        engine = engines.get(engine_name)
        if engine is None:
            raise ValueError(f"unknown HE engine {engine_name!r}")
        if manifest.id in model_engine_routes:
            raise ValueError(f"model {manifest.id!r} is already loaded")
        await engine.load(manifest)
        manifest_getter = getattr(engine, "model_manifest", None)
        if manifest_getter is not None:
            manifest = manifest_getter(manifest.id)
        imported[manifest.id] = manifest
        model_engine_routes[manifest.id] = engine_name
        for stage in manifest.stages:
            engine_schedulers[(manifest.id, stage.id)] = StageBatchScheduler(
                EngineStageExecutor(engine, manifest.id, stage),
                max_batch_size=config.max_batch_size,
                max_wait_ms=config.max_batch_wait_ms,
                adaptive_wait=config.adaptive_batching,
            )
        return manifest

    @app.get("/v1/he/engines")
    async def list_engines(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "he.engine", "capabilities": engine.capabilities.to_dict()}
                for name, engine in engines.items()
            ],
        }

    @app.post("/v1/he/models/load")
    async def load_model_plugin(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        body = await request.json()
        engine_name = str(body.get("engine", ""))
        try:
            manifest = await inspect_source(body)
            manifest = await register_engine_model(engine_name, manifest)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc)}}) from exc
        return {**manifest.to_dict(), "engine": engine_name, "status": "ready"}

    @app.delete("/v1/he/models/{model_id:path}")
    async def unload_model_plugin(model_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        engine_name = model_engine_routes.pop(model_id, None)
        if engine_name is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Model is not loaded by an HE engine"}})
        for key, scheduler in list(engine_schedulers.items()):
            if key[0] == model_id:
                await scheduler.close()
                del engine_schedulers[key]
        await engines[engine_name].unload(model_id)
        imported.pop(model_id, None)
        return {"id": model_id, "object": "he.model", "status": "unloaded", "engine": engine_name}

    @app.post("/v1/he/engines/{engine_name}/models/{model_id:path}/stages/{stage_id}")
    async def execute_engine_stage(
        engine_name: str,
        model_id: str,
        stage_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        auth_token(authorization)
        engine = engines.get(engine_name)
        manifest = imported.get(model_id)
        if engine is None or manifest is None or model_engine_routes.get(model_id) != engine_name:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown engine/model route"}})
        stage = next((value for value in manifest.stages if value.id == stage_id), None)
        if stage is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown model stage"}})
        raw = await request.body()
        try:
            payloads = list(iter_length_prefixed(raw))
            if not payloads:
                raise ProtocolError("empty stage batch")
            results = await engine.execute_stage(model_id, stage, payloads)
            if len(results) != len(payloads):
                raise ProtocolError("HE engine returned the wrong result count")
        except (ProtocolError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "invalid_stage_batch"}})
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/he/models/inspect")
    async def inspect_model(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        body = await request.json()
        manifest = await inspect_source(body)
        if body.get("register", True):
            imported[manifest.id] = manifest
        return manifest.to_dict()

    @app.get("/v1/he/models/{model_id:path}/client-bundle")
    async def model_client_bundle(
        model_id: str,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        auth_token(authorization)
        engine_name = model_engine_routes.get(model_id)
        if engine_name is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Model has no client bundle"}})
        engine = engines[engine_name]
        getter = getattr(engine, "client_bundle", None)
        if getter is None:
            raise HTTPException(status_code=501, detail={"error": {"message": "HE engine does not expose a client bundle"}})
        return FastAPIResponse(getter(model_id), media_type="application/msgpack")

    @app.get("/v1/he/models/{model_id:path}")
    async def model_manifest(model_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        if model_id in strict_models:
            return strict_models[model_id].manifest.to_model_object()
        if model_id in imported:
            return imported[model_id].to_dict()
        raise HTTPException(status_code=404, detail={"error": {"message": "Unknown model"}})

    @app.post("/v1/he/sessions")
    async def create_session(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        api_key = auth_token(authorization)
        body = await request.json()
        audit("session", json.dumps(body, separators=(",", ":")).encode())
        model_id = str(body.get("model", ""))
        model = strict_models.get(model_id)
        imported_manifest = imported.get(model_id)
        engine_name = model_engine_routes.get(model_id)
        if model is None and (imported_manifest is None or engine_name is None):
            raise HTTPException(status_code=400, detail={"error": {"message": "Model has no strict HE runtime", "code": "he_runtime_required"}})
        session_id = new_id("hes")
        session = HESession(session_id, new_id("resp"), model_id, api_key)
        requested_contexts = list(body.get("context_ids") or [])
        if body.get("context_id") is not None:
            requested_contexts.append(body.get("context_id"))
        for requested_context in requested_contexts:
            requested_context = str(requested_context)
            if context_owners.get((model_id, requested_context)) != api_key:
                raise HTTPException(status_code=409, detail={"error": {"message": "Unknown or unauthorized HE context", "code": "he_context_unavailable"}})
            session.context_ids.add(requested_context)
            session.context_id = requested_context
        sessions[session_id] = session
        if model is not None:
            session_manifest = model.manifest.to_model_object()
        else:
            assert imported_manifest is not None and engine_name is not None
            session_manifest = {
                "id": model_id,
                "object": "model",
                "created": int(imported_manifest.created_at),
                "owned_by": "pllm",
                "he": {
                    "privacy_mode": imported_manifest.metadata.get("privacy_mode", config.privacy_mode),
                    "privacy_protocol": imported_manifest.metadata.get("privacy_protocol"),
                    "online_fhe": bool(imported_manifest.metadata.get("online_fhe", False)),
                    "he_preprocessed": bool(imported_manifest.metadata.get("he_preprocessed", False)),
                    "model_privacy_threat_model": imported_manifest.metadata.get("model_privacy_threat_model"),
                    "malicious_client_model_privacy": imported_manifest.metadata.get("malicious_client_model_privacy"),
                    "runtime": imported_manifest.metadata.get("client_runtime"),
                    "engine": engine_name,
                    "fingerprint": imported_manifest.fingerprint,
                    "max_output_tokens": imported_manifest.context_length,
                    "stage_count": len(imported_manifest.stages),
                },
            }
        return {
            "id": session.id,
            "object": "he.session",
            "response_id": session.response_id,
            "model": model_id,
            "manifest": session_manifest,
            "websocket_path": f"/v1/he/ws/{session.id}",
        }

    def get_session(session_id: str, api_key: str) -> HESession:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown HE session"}})
        if session.api_key != api_key:
            raise HTTPException(status_code=403, detail={"error": {"message": "HE session belongs to another credential"}})
        return session

    @app.put("/v1/he/sessions/{session_id}/contexts/{context_id}")
    async def register_context(
        session_id: str,
        context_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        payload = await request.body()
        audit("bfv_context", payload)
        if not payload:
            raise HTTPException(status_code=400, detail={"error": {"message": "Empty public context"}})
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        if session.model_id not in bfv_servers and engine is None:
            raise HTTPException(status_code=503, detail={"error": {"message": "BFV runtime is not installed", "code": "he_runtime_unavailable"}})
        try:
            if engine is not None:
                register = getattr(engine, "register_bfv_context", None)
                if register is None:
                    raise ValueError("HE engine does not support BFV correlations")
                await asyncio.to_thread(register, session.model_id, context_id, payload)
            else:
                bfv_servers[session.model_id].register_context(context_id, payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "invalid_he_context"}})
        session.context_id = context_id
        session.context_ids.add(context_id)
        context_owners[(session.model_id, context_id)] = api_key
        return {"id": context_id, "object": "he.context", "status": "ready", "bytes": len(payload), "reusable": True}

    @app.post("/v1/he/sessions/{session_id}/correlations/bfv")
    async def bfv_correlation(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        context_id = request.headers.get("x-he-context-id") or session.context_id
        if context_id is None or context_id not in session.context_ids:
            raise HTTPException(status_code=409, detail={"error": {"message": "Register the requested HE context first"}})
        stage_id = request.headers.get("x-he-stage-id")
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        if session.model_id not in bfv_servers and engine is None:
            raise HTTPException(status_code=503, detail={"error": {"message": "BFV runtime is not installed", "code": "he_runtime_unavailable"}})
        encrypted_mask = await request.body()
        audit("bfv_mask", encrypted_mask)
        try:
            if engine is not None:
                if not stage_id:
                    raise ValueError("x-he-stage-id is required for transformer BFV correlations")
                evaluate = getattr(engine, "evaluate_bfv_correlation", None)
                if evaluate is None:
                    raise ValueError("HE engine does not support BFV correlations")
                result = await asyncio.to_thread(
                    evaluate,
                    session.model_id,
                    stage_id,
                    context_id,
                    encrypted_mask,
                )
            else:
                result = await asyncio.to_thread(
                    bfv_servers[session.model_id].evaluate,
                    context_id,
                    encrypted_mask,
                )
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "he_correlation_failed"}})
        session.correlation_steps += 1
        return FastAPIResponse(result, media_type="application/octet-stream")

    @app.post("/v1/he/sessions/{session_id}/correlations/bfv/batch")
    async def bfv_correlation_batch(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_he_stage_id: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        context_id = request.headers.get("x-he-context-id") or session.context_id
        if context_id is None or context_id not in session.context_ids:
            raise HTTPException(status_code=409, detail={"error": {"message": "Register the requested HE context first"}})
        if not x_he_stage_id:
            raise HTTPException(status_code=400, detail={"error": {"message": "x-he-stage-id is required"}})
        raw = await request.body()
        audit("bfv_correlation_batch", raw)
        try:
            encrypted_masks = list(iter_length_prefixed(raw))
            if not encrypted_masks:
                raise ValueError("empty BFV correlation batch")
            if len(encrypted_masks) > config.max_batch_size:
                raise ValueError("BFV correlation batch is too large")
            engine_name = model_engine_routes.get(session.model_id)
            engine = engines.get(engine_name) if engine_name else None
            if engine is None:
                raise ValueError("transformer BFV batch requires an HE engine")
            evaluate_many = getattr(engine, "evaluate_bfv_correlations", None)
            if evaluate_many is None:
                evaluate_one = getattr(engine, "evaluate_bfv_correlation", None)
                if evaluate_one is None:
                    raise ValueError("HE engine does not support BFV correlations")
                results = await asyncio.to_thread(
                    lambda: [
                        evaluate_one(session.model_id, x_he_stage_id, context_id, item)
                        for item in encrypted_masks
                    ]
                )
            else:
                cancel_event = threading.Event()
                worker = asyncio.create_task(asyncio.to_thread(
                    evaluate_many,
                    session.model_id,
                    x_he_stage_id,
                    context_id,
                    encrypted_masks,
                    cancel_event=cancel_event,
                ))
                while not worker.done():
                    await asyncio.sleep(0.1)
                    if await request.is_disconnected():
                        cancel_event.set()
                        try:
                            await worker
                        except Exception:
                            pass
                        raise HTTPException(
                            status_code=499,
                            detail={"error": {"message": "Client disconnected"}},
                        )
                results = await worker
        except (ProtocolError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "he_correlation_failed"}}) from exc
        session.correlation_steps += len(results)
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/he/sessions/{session_id}/correlations/proprietary/bfv/batch")
    async def proprietary_bfv_correlation_batch(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_he_stage_id: str | None = Header(default=None),
        x_he_owner_id: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        context_id = request.headers.get("x-he-context-id") or session.context_id
        if context_id is None or context_id not in session.context_ids:
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": "Register the requested HE context first"}},
            )
        if not x_he_stage_id or not x_he_owner_id:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "x-he-stage-id and x-he-owner-id are required"}},
            )
        try:
            bind_proprietary_owner(session, x_he_owner_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"error": {"message": str(exc)}}) from exc
        raw = await request.body()
        audit("bfv_blinded_correlation_batch", raw)
        try:
            encrypted_masks = list(iter_length_prefixed(raw))
            if not encrypted_masks:
                raise ValueError("empty proprietary BFV correlation batch")
            if len(encrypted_masks) > config.max_batch_size:
                raise ValueError("proprietary BFV correlation batch is too large")
            engine_name = model_engine_routes.get(session.model_id)
            engine = engines.get(engine_name) if engine_name else None
            if engine is None:
                raise ValueError("proprietary BFV batch requires an HE engine")
            evaluate_many = getattr(engine, "evaluate_bfv_blinded_correlations", None)
            if evaluate_many is None:
                raise ValueError("HE engine does not support output-blinded correlations")
            results = await asyncio.to_thread(
                evaluate_many,
                session.model_id,
                x_he_stage_id,
                x_he_owner_id,
                context_id,
                encrypted_masks,
            )
        except (ProtocolError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "he_correlation_failed"}},
            ) from exc
        session.correlation_steps += len(results)
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/he/sessions/{session_id}/correlations/proprietary/local-test")
    async def proprietary_local_correlations(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        if not config.allow_insecure_local_correlations:
            raise HTTPException(status_code=404, detail={"error": {"message": "Not enabled"}})
        body = await request.json()
        audit("local_blinded_correlation", json.dumps(body, separators=(",", ":")).encode())
        count = max(1, min(int(body.get("count", 1)), 4096))
        stage_id = body.get("stage_id")
        owner_id = body.get("owner_id")
        if not stage_id or not owner_id:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "stage_id and owner_id are required"}},
            )
        try:
            bind_proprietary_owner(session, str(owner_id))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={"error": {"message": str(exc)}}) from exc
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        create = getattr(engine, "create_local_blinded_correlations", None) if engine else None
        if create is None:
            raise HTTPException(
                status_code=501,
                detail={"error": {"message": "HE engine cannot create blinded test correlations"}},
            )
        rows = await asyncio.to_thread(
            create,
            session.model_id,
            str(stage_id),
            str(owner_id),
            count,
        )
        session.correlation_steps += count
        payload = msgpack.packb(
            {
                "warning": "insecure test correlations; server knows both masks",
                "items": [blinded_correlation_to_wire(row) for row in rows],
            },
            use_bin_type=True,
        )
        return FastAPIResponse(payload, media_type="application/msgpack")

    @app.post("/v1/he/sessions/{session_id}/correlations/local-test")
    async def local_correlations(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        if not config.allow_insecure_local_correlations:
            raise HTTPException(status_code=404, detail={"error": {"message": "Not enabled"}})
        local_body = await request.json()
        audit("local_correlation", json.dumps(local_body, separators=(",", ":")).encode())
        count = max(1, min(int(local_body.get("count", 1)), 4096))
        stage_id = local_body.get("stage_id")
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        if engine is not None:
            if not stage_id:
                raise HTTPException(status_code=400, detail={"error": {"message": "stage_id is required"}})
            create = getattr(engine, "create_local_correlations", None)
            if create is None:
                raise HTTPException(status_code=501, detail={"error": {"message": "HE engine cannot create test correlations"}})
            rows = await asyncio.to_thread(
                create,
                session.model_id,
                str(stage_id),
                count,
                ring=local_body.get("ring"),
            )
        else:
            rows = local_factories[session.model_id].create(count)
        session.correlation_steps += count
        payload = msgpack.packb(
            {
                "warning": "insecure test correlations; server knows masks",
                "items": [
                    correlation_to_wire(row)
                    if engine is not None
                    else {
                        "id": row.id,
                        "mask": row.mask.astype("<i8").tobytes(),
                        "transformed": row.transformed_mask.astype("<i8").tobytes(),
                        "shape": list(row.mask.shape),
                    }
                    for row in rows
                ],
            },
            use_bin_type=True,
        )
        return FastAPIResponse(payload, media_type="application/msgpack")

    async def execute_envelope(session: HESession, envelope: HEEnvelope) -> HEEnvelope:
        envelope.verify(session.key)
        session.replay.accept(envelope)
        if session.canceled:
            raise ProtocolError("session cancelled")
        if envelope.session_id != session.id or envelope.model != session.model_id:
            raise ProtocolError("session/model mismatch")
        if envelope.kind == "masked.linear":
            result = await schedulers[session.model_id].submit(envelope.payload)
            result_kind = "masked.linear.result"
        elif envelope.kind == "masked.transformer.stage":
            stage_id = str(envelope.metadata.get("stage_id", ""))
            scheduler = engine_schedulers.get((session.model_id, stage_id))
            if scheduler is None:
                raise ProtocolError("unknown transformer stage")
            result = await scheduler.submit(envelope.payload)
            result_kind = "masked.transformer.stage.result"
        else:
            raise ProtocolError("unsupported HE frame kind")
        session.online_steps += 1
        return HEEnvelope.create(
            request_id=envelope.request_id,
            session_id=session.id,
            model=session.model_id,
            kind=result_kind,
            sequence=envelope.sequence,
            payload=result,
            metadata={},
            key=session.key,
        )

    @app.post("/v1/he/sessions/{session_id}/execute")
    async def execute_http(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        try:
            execute_payload = await request.body()
            audit("execute_http", execute_payload)
            envelope = unpack_envelope(execute_payload)
            result = await execute_envelope(session, envelope)
        except ProtocolError as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "invalid_he_frame"}})
        return FastAPIResponse(pack_envelope(result), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/he/sessions/{session_id}/stages/{stage_id}")
    async def execute_session_stage_batch(
        session_id: str,
        stage_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        """Execute a prefill-sized batch at one model stage.

        Decode requests continue to use the authenticated envelope path so
        independent sessions can be continuously coalesced. Prefill already
        has many same-stage rows inside one session and is more efficient as a
        single length-prefixed batch.
        """
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        manifest = imported.get(session.model_id)
        if engine is None or manifest is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Session model has no transformer engine"}})
        stage = next((value for value in manifest.stages if value.id == stage_id), None)
        if stage is None:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown transformer stage"}})
        raw = await request.body()
        audit("execute_stage_batch", raw)
        try:
            payloads = list(iter_length_prefixed(raw))
            if not payloads:
                raise ProtocolError("empty stage batch")
            if len(payloads) > config.max_batch_size * 16:
                raise ProtocolError("stage prefill batch is too large")
            results = await engine.execute_stage(session.model_id, stage, payloads)
            if len(results) != len(payloads):
                raise ProtocolError("HE engine returned the wrong result count")
        except (ProtocolError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc), "code": "invalid_stage_batch"}})
        session.online_steps += len(payloads)
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.websocket("/v1/he/ws/{session_id}")
    async def execute_websocket(websocket: WebSocket, session_id: str) -> None:
        token = bearer_token(websocket.headers.get("authorization"))
        if token is None or (config.api_keys and token not in config.api_keys):
            await websocket.close(code=4401, reason="invalid API key")
            return
        try:
            session = get_session(session_id, token)
        except HTTPException:
            await websocket.close(code=4404, reason="unknown session")
            return
        await websocket.accept(subprotocol="he-responses-v1")
        try:
            while True:
                ws_payload = await websocket.receive_bytes()
                audit("execute_websocket", ws_payload)
                envelope = unpack_envelope(ws_payload)
                result = await execute_envelope(session, envelope)
                await websocket.send_bytes(pack_envelope(result))
        except WebSocketDisconnect:
            return
        except ProtocolError as exc:
            await websocket.close(code=4400, reason=str(exc))

    @app.post("/v1/he/sessions/{session_id}/complete")
    async def complete_session(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        body = await request.json()
        audit("complete", json.dumps(body, separators=(",", ":")).encode())
        session.completed = True
        # Operational record only: plaintext output remains in the client cache.
        value = {
            "id": session.response_id,
            "object": "response",
            "created_at": session.created_at,
            "status": "completed",
            "model": session.model_id,
            "output": [],
            "usage": body.get("usage"),
            "he_redacted": True,
        }
        responses.put(value)
        return value

    @app.post("/v1/responses")
    async def create_response(request: Request, authorization: str | None = Header(default=None)):
        auth_token(authorization)
        body = await request.json()
        model_id = str(body.get("model", ""))
        if model_id in strict_models or model_id in model_engine_routes:
            raise HTTPException(
                status_code=426,
                headers={"Upgrade": "he-responses/1"},
                detail={
                    "error": {
                        "message": "Strict-HE model requires pllm.runtime.OpenAI or HETransport; plaintext input was rejected",
                        "type": "invalid_request_error",
                        "code": "he_client_required",
                    }
                },
            )
        try:
            adapter, upstream_model_id = backend_registry.resolve(model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"error": {"message": f"Unknown model {model_id}"}})
        upstream_body = dict(body)
        upstream_body["model"] = upstream_model_id

        def publicize(value: Any) -> Any:
            """Keep public model IDs stable when a backend route is prefixed."""
            if isinstance(value, dict):
                return {
                    key: (model_id if key == "model" and item == upstream_model_id else publicize(item))
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [publicize(item) for item in value]
            return value

        if body.get("stream"):
            async def generate():
                async for raw_event in adapter.stream_response(upstream_body):
                    event = publicize(raw_event)
                    if event.get("type") == "response.completed" and isinstance(event.get("response"), dict):
                        responses.put(event["response"])
                    yield sse_event(event)
                yield b"data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")
        try:
            value = publicize(await adapter.create_response(upstream_body))
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": {"message": str(exc), "type": "backend_error"}})
        responses.put(value)
        return JSONResponse(value)

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(response_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        try:
            return responses.get(response_id)
        except KeyError:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown response"}})

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(response_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        for session in sessions.values():
            if session.response_id == response_id:
                session.canceled = True
        try:
            return responses.cancel(response_id)
        except KeyError:
            value = {"id": response_id, "object": "response", "status": "cancelled", "output": []}
            responses.put(value)
            return value

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return {
            "sessions": {
                "active": sum(not row.completed and not row.canceled for row in sessions.values()),
                "total": len(sessions),
                "online_steps": sum(row.online_steps for row in sessions.values()),
                "correlation_steps": sum(row.correlation_steps for row in sessions.values()),
            },
            "stage_schedulers": {
                **{model: scheduler.stats() for model, scheduler in schedulers.items()},
                **{f"{model}:{stage}": scheduler.stats() for (model, stage), scheduler in engine_schedulers.items()},
            },
            "engines": {
                name: (getattr(engine, "metrics")() if hasattr(engine, "metrics") else {})
                for name, engine in engines.items()
            },
        }

    return app
