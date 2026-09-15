from __future__ import annotations

import asyncio
import base64
import hashlib
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
from .correction_channel import (
    CORRECTION_CHANNEL_SUBPROTOCOL,
    CorrectionChannelError,
)
from .correction_rendezvous import CorrectionRendezvous, RendezvousError
from .engine import InferenceEngine
from .engine import EngineStageExecutor
from .hf_hub import resolve_huggingface_source
from .bfv_correlations import BFVCorrelationServer
from .masked_runtime import (
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
    BINARY_MEDIA_TYPE,
    ProtocolEnvelope,
    ProtocolError,
    ReplayWindow,
    encode_length_prefixed,
    iter_length_prefixed,
    pack_envelope,
    sse_event,
    unpack_envelope,
)
from .privacy import PrivacyMode
from .preparation_protocol import (
    PREPARATION_MAX_IDENTIFIER_BYTES,
    CorrectionPush,
    PreparationAck,
    PreparationRequest,
    SeededRingProfile,
    SessionAuthorization,
    SessionAuthorizationAck,
    derive_online_attempt_id,
    validate_attempt_id,
)
from .stage_protocol import (
    MaskedStageRequest,
    MaskedStageResponse,
    PreparedStageBatchRequest,
    PreparedStageBatchResponse,
    blinded_correlation_to_wire,
    correlation_to_wire,
    prepared_stage_batch_rows,
)
from .responses import ResponsesError
from .security import bearer_token, derive_session_key
from .types import new_id


@dataclass(slots=True)
class RuntimeSession:
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
    execution: str = "runtime"
    inventory_id: str | None = None
    inventory_rows: int = 0
    inventory_stages: frozenset[str] = frozenset()
    prepared_stages: set[str] = field(default_factory=set)
    inventory_roots: dict[str, PreparationRequest] = field(default_factory=dict)
    inventory_next_row: int = 0
    inventory_ready: bool = False
    reserved_attempts: frozenset[tuple[str, str]] = frozenset()
    reservation_start: int = 0
    reservation_rows: int = 0
    inventory_reserved_entries: int = 0
    inventory_reserved_bytes: int = 0
    last_active: float = field(default_factory=time.monotonic)

    @property
    def key(self) -> bytes:
        return derive_session_key(self.api_key, self.id)


class RetainedResponses:
    def __init__(self, ttl: float) -> None:
        self.ttl = ttl
        self.values: dict[tuple[str, str], tuple[float, dict[str, Any]]] = {}

    def put(self, value: dict[str, Any], principal: str) -> None:
        self.values[(principal, str(value["id"]))] = (time.time(), value)
        self._expire()

    def get(self, response_id: str, principal: str) -> dict[str, Any]:
        self._expire()
        key = (principal, response_id)
        if key not in self.values:
            raise KeyError(response_id)
        return self.values[key][1]

    def cancel(self, response_id: str, principal: str) -> dict[str, Any]:
        value = dict(self.get(response_id, principal))
        value["status"] = "cancelled"
        self.put(value, principal)
        return value

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for key, (created, _) in list(self.values.items()):
            if created < cutoff:
                del self.values[key]


async def _read_limited_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) < 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
        if int(declared) > limit:
            raise HTTPException(status_code=413, detail="Request body is too large")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise HTTPException(status_code=413, detail="Request body is too large")
    return bytes(body)


def create_app(
    config: GatewayConfig | None = None,
    *,
    backend_registry: BackendRegistry | None = None,
    private_models: dict[str, MaskedBigramModel] | None = None,
    engines: dict[str, InferenceEngine] | None = None,
    audit_hook: Callable[[str, bytes], None] | None = None,
) -> FastAPI:
    config = config or GatewayConfig()
    if config.provider_push_api_key and config.provider_push_api_key in config.api_keys:
        raise ValueError("client and correction-push credentials must be distinct")
    backend_registry = backend_registry or BackendRegistry.from_config(list(config.backends))
    engines = engines or {}
    if private_models is None:
        alphabet = config.reference_alphabet.replace("\\n", "\n")
        private_models = {
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
        for model_id, model in private_models.items()
    }
    engine_schedulers: dict[tuple[str, str], StageBatchScheduler] = {}
    bfv_servers: dict[str, BFVCorrelationServer] = {}
    context_owners: dict[tuple[str, str], str] = {}
    proprietary_owner_by_principal: dict[tuple[str, str], str] = {}
    for model_id, model in private_models.items():
        try:
            bfv_servers[model_id] = BFVCorrelationServer(
                model.weight, pydeps_path=config.tenseal_path
            )
        except ImportError:
            # TenSEAL is an optional runtime dependency. Clear passthrough and
            # local-test protocol validation still work without it.
            pass
    local_factories = {
        model_id: LocalCorrelationFactory(model.weight, model.modulus, seed=17)
        for model_id, model in private_models.items()
    }
    sessions: dict[str, RuntimeSession] = {}
    inventory_lock = threading.Lock()
    responses = RetainedResponses(config.response_retention_seconds)
    imported: dict[str, ImportedModelManifest] = {}
    model_engine_routes: dict[str, str] = {}
    client_bundles: dict[tuple[str, str], tuple[bytes, dict[str, Any]]] = {}
    backend_models: list[dict[str, Any]] = []
    rendezvous = CorrectionRendezvous(
        timeout=config.rendezvous_timeout_seconds,
        capacity=config.rendezvous_capacity,
        max_bytes=config.rendezvous_max_bytes,
        session_capacity=config.prepared_session_capacity,
        session_idle_timeout=config.prepared_session_idle_seconds,
    )
    correction_channel_metrics = {
        "connections": 0,
        "frames": 0,
        "correction_bytes": 0,
        "acknowledgement_bytes": 0,
        "processing_ns": 0,
        "failures": 0,
    }

    def bind_proprietary_owner(session: RuntimeSession, owner_id: str) -> None:
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
                backend_models = [
                    {
                        "id": "backend-discovery-error",
                        "object": "model",
                        "owned_by": "pllm",
                        "runtime": {"error": str(exc), "privacy_mode": "trusted_backend"},
                    }
                ]
        try:
            yield
        finally:
            rendezvous.close()
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
    app.state.private_models = private_models
    app.state.schedulers = schedulers
    app.state.sessions = sessions
    app.state.imported_manifests = imported
    app.state.backend_registry = backend_registry
    app.state.engines = engines
    app.state.engine_schedulers = engine_schedulers
    app.state.correction_rendezvous = rendezvous

    def audit(kind: str, payload: bytes) -> None:
        if audit_hook is not None:
            audit_hook(kind, payload)

    def burn_prepared_reservation(session: RuntimeSession) -> None:
        if session.inventory_id and session.reserved_attempts:
            rendezvous.burn_reserved(
                session.inventory_id,
                frozenset(attempt_id for _, attempt_id in session.reserved_attempts),
            )

    def retire_exhausted_inventory(session: RuntimeSession) -> None:
        if session.execution != "seeded-preparation" or not session.inventory_id:
            return
        with inventory_lock:
            inventory = sessions.get(session.inventory_id)
            if (
                inventory is None
                or inventory.execution != "seeded-inventory"
                or inventory.canceled
                or inventory.completed
                or inventory.inventory_next_row < inventory.inventory_rows
                or any(
                    candidate.execution == "seeded-preparation"
                    and candidate.inventory_id == inventory.id
                    and not candidate.canceled
                    and not candidate.completed
                    for candidate in sessions.values()
                )
            ):
                return
            inventory.completed = True
            inventory.last_active = time.monotonic()
            rendezvous.terminal(inventory.id)

    def cleanup_prepared_sessions(*, reclaim_terminal: bool = False) -> None:
        cutoff = time.monotonic() - config.prepared_session_idle_seconds
        for session_id, session in list(sessions.items()):
            if session.execution not in {"seeded-preparation", "seeded-inventory"} or not (
                session.last_active <= cutoff
                or reclaim_terminal
                and (session.canceled or session.completed)
            ):
                continue
            if not session.completed:
                session.canceled = True
            if sessions.get(session_id) is session:
                sessions.pop(session_id)
            if session.execution == "seeded-inventory":
                rendezvous.terminal(session_id)
            else:
                burn_prepared_reservation(session)

    def auth_token(value: str | None) -> str:
        token = bearer_token(value)
        if token is None or (config.api_keys and token not in config.api_keys):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid API key",
                        "type": "authentication_error",
                        "code": "invalid_api_key",
                    }
                },
            )
        return token

    def auth_push_token(value: str | None) -> None:
        token = bearer_token(value)
        expected = config.provider_push_api_key
        if expected is None or token is None or not secrets.compare_digest(token, expected):
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "Invalid correction push credential",
                        "code": "invalid_push_key",
                    }
                },
            )

    def client_bundle_record(engine_name: str, model_id: str) -> tuple[bytes, dict[str, Any]]:
        key = (engine_name, model_id)
        cached = client_bundles.get(key)
        if cached is not None:
            return cached
        getter = getattr(engines[engine_name], "client_bundle", None)
        if getter is None:
            raise HTTPException(
                status_code=501,
                detail={"error": {"message": "Runtime engine does not expose a client bundle"}},
            )
        payload = getter(model_id)
        try:
            schema = int(msgpack.unpackb(payload, raw=False, strict_map_key=False)["v"])
        except (KeyError, TypeError, ValueError, msgpack.UnpackException) as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": {"message": "Runtime engine returned an invalid client bundle"}},
            ) from exc
        fingerprint = hashlib.sha256(payload).hexdigest()
        descriptor = {
            "schema": schema,
            "sha256": fingerprint,
            "size": len(payload),
            "etag": f'"{fingerprint}"',
        }
        cached = (payload, descriptor)
        client_bundles[key] = cached
        return cached

    @app.get("/healthz")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        cleanup_prepared_sessions(reclaim_terminal=True)
        return {
            "status": "ok",
            "privacy_mode": config.privacy_mode,
            "private_models": len(private_models),
            "trusted_backends": len(backend_registry.adapters),
            "runtime_engines": len(engines),
            "loaded_engine_models": len(model_engine_routes),
            "sessions": len(sessions),
        }

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        private = [model.manifest.to_model_object() for model in private_models.values()]
        manifests = []
        for manifest in imported.values():
            engine_name = model_engine_routes.get(manifest.id)
            bundle_descriptor = (
                client_bundle_record(engine_name, manifest.id)[1]
                if engine_name is not None
                and getattr(engines[engine_name], "client_bundle", None) is not None
                else None
            )
            manifests.append(
                {
                    "id": manifest.id,
                    "object": "model",
                    "created": int(manifest.created_at),
                    "owned_by": "pllm",
                    "runtime": {
                        "privacy_mode": (
                            manifest.metadata.get("privacy_mode", "private_engine")
                            if engine_name
                            else "manifest_only"
                        ),
                        "privacy_protocol": manifest.metadata.get("privacy_protocol"),
                        "engine": engine_name,
                        "source_format": manifest.source_format,
                        "architecture": manifest.architecture,
                        "stage_count": len(manifest.stages),
                        "fingerprint": manifest.fingerprint,
                        "body_fingerprint": manifest.metadata.get("body_fingerprint"),
                        "stage_commitment": manifest.metadata.get("seeded_stage_commitment"),
                        "client_bundle": bundle_descriptor,
                    },
                    "client_bundle": bundle_descriptor,
                }
            )
        return {"object": "list", "data": private + manifests + backend_models}

    @app.get("/v1/runtime/capabilities")
    async def capabilities(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        return {
            "object": "runtime.capabilities",
            "protocol": "pllm-runtime/1",
            "responses_api": True,
            "client_transports": ["http-binary", "websocket-binary", "local-sidecar"],
            "server_privacy_mode": config.privacy_mode,
            "correlation_modes": (
                ["bfv-exact"] + (["local-test"] if config.allow_insecure_local_correlations else [])
                if config.privacy_mode == "public"
                else []
            ),
            "private_models": [model.manifest.to_model_object() for model in private_models.values()],
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
                raise HTTPException(
                    status_code=400, detail={"error": {"message": "path or repo_id is required"}}
                )
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
        raise HTTPException(
            status_code=400, detail={"error": {"message": f"Unsupported model source {kind}"}}
        )

    async def register_engine_model(
        engine_name: str, manifest: ImportedModelManifest
    ) -> ImportedModelManifest:
        engine = engines.get(engine_name)
        if engine is None:
            raise ValueError(f"unknown runtime engine {engine_name!r}")
        if manifest.id in model_engine_routes:
            raise ValueError(f"model {manifest.id!r} is already loaded")
        await engine.load(manifest)
        client_bundles.pop((engine_name, manifest.id), None)
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

    @app.get("/v1/runtime/engines")
    async def list_engines(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        auth_token(authorization)
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "runtime.engine", "capabilities": engine.capabilities.to_dict()}
                for name, engine in engines.items()
            ],
        }

    @app.post("/v1/runtime/models/load")
    async def load_model_plugin(
        request: Request, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        auth_token(authorization)
        body = await request.json()
        engine_name = str(body.get("engine", ""))
        try:
            manifest = await inspect_source(body)
            manifest = await register_engine_model(engine_name, manifest)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": str(exc)}}) from exc
        return {**manifest.to_dict(), "engine": engine_name, "status": "ready"}

    @app.delete("/v1/runtime/models/{model_id:path}")
    async def unload_model_plugin(
        model_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        auth_token(authorization)
        engine_name = model_engine_routes.pop(model_id, None)
        if engine_name is None:
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": "Model is not loaded by a runtime engine"}},
            )
        for key, scheduler in list(engine_schedulers.items()):
            if key[0] == model_id:
                await scheduler.close()
                del engine_schedulers[key]
        await engines[engine_name].unload(model_id)
        client_bundles.pop((engine_name, model_id), None)
        imported.pop(model_id, None)
        return {"id": model_id, "object": "runtime.model", "status": "unloaded", "engine": engine_name}

    @app.post("/v1/runtime/engines/{engine_name}/models/{model_id:path}/stages/{stage_id}")
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
            raise HTTPException(
                status_code=404, detail={"error": {"message": "Unknown engine/model route"}}
            )
        stage = next((value for value in manifest.stages if value.id == stage_id), None)
        if stage is None:
            raise HTTPException(
                status_code=404, detail={"error": {"message": "Unknown model stage"}}
            )
        raw = await request.body()
        try:
            payloads = list(iter_length_prefixed(raw))
            if not payloads:
                raise ProtocolError("empty stage batch")
            results = await engine.execute_stage(model_id, stage, payloads)
            if len(results) != len(payloads):
                raise ProtocolError("Runtime engine returned the wrong result count")
        except (ProtocolError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "invalid_stage_batch"}},
            )
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/runtime/models/inspect")
    async def inspect_model(
        request: Request, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        auth_token(authorization)
        body = await request.json()
        manifest = await inspect_source(body)
        if body.get("register", True):
            imported[manifest.id] = manifest
        return manifest.to_dict()

    @app.get("/v1/runtime/models/{model_id:path}/client-bundle")
    async def model_client_bundle(
        model_id: str,
        authorization: str | None = Header(default=None),
        if_none_match: str | None = Header(default=None, alias="If-None-Match"),
    ) -> FastAPIResponse:
        auth_token(authorization)
        engine_name = model_engine_routes.get(model_id)
        if engine_name is None:
            raise HTTPException(
                status_code=404, detail={"error": {"message": "Model has no client bundle"}}
            )
        payload, descriptor = client_bundle_record(engine_name, model_id)
        headers = {
            "ETag": str(descriptor["etag"]),
            "X-PLLM-Bundle-SHA256": str(descriptor["sha256"]),
            "X-PLLM-Bundle-Schema": str(descriptor["schema"]),
            "X-PLLM-Model-ID": model_id,
        }
        if if_none_match == descriptor["etag"]:
            return FastAPIResponse(status_code=304, headers=headers)
        return FastAPIResponse(payload, media_type="application/msgpack", headers=headers)

    @app.get("/v1/runtime/models/{model_id:path}")
    async def model_manifest(
        model_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        auth_token(authorization)
        if model_id in private_models:
            return private_models[model_id].manifest.to_model_object()
        if model_id in imported:
            engine_name = model_engine_routes[model_id]
            result = imported[model_id].to_dict()
            if getattr(engines[engine_name], "client_bundle", None) is not None:
                result["client_bundle"] = client_bundle_record(engine_name, model_id)[1]
            return result
        raise HTTPException(status_code=404, detail={"error": {"message": "Unknown model"}})

    @app.post("/v1/runtime/sessions")
    async def create_session(
        request: Request, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        body = await request.json()
        cleanup_prepared_sessions(reclaim_terminal=True)
        audit("session", json.dumps(body, separators=(",", ":")).encode())
        model_id = str(body.get("model", ""))
        model = private_models.get(model_id)
        imported_manifest = imported.get(model_id)
        engine_name = model_engine_routes.get(model_id)
        if model is None and (imported_manifest is None or engine_name is None):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "Model has no private runtime",
                        "code": "runtime_required",
                    }
                },
            )
        if (
            imported_manifest is not None
            and PrivacyMode.parse(
                imported_manifest.metadata.get("privacy_mode", config.privacy_mode)
            )
            is PrivacyMode.PUBLIC
            and (engine_name or "").startswith("masked-transformer")
            and body.get("execution") != "seeded-preparation"
        ):
            raise HTTPException(
                status_code=410,
                detail={
                    "error": {
                        "message": "Public transformer runtime sessions were replaced by seeded preparation",
                        "code": "seeded_preparation_required",
                    }
                },
            )
        if body.get("execution") == "seeded-preparation" and not config.provider_push_api_key:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "Correction push credential is not configured",
                        "code": "prepared_runtime_unavailable",
                    }
                },
            )
        session_id = new_id("rts")
        session = RuntimeSession(session_id, new_id("resp"), model_id, api_key)
        session.execution = str(body.get("execution") or "runtime")
        requested_contexts = list(body.get("context_ids") or [])
        if body.get("context_id") is not None:
            requested_contexts.append(body.get("context_id"))
        for requested_context in requested_contexts:
            requested_context = str(requested_context)
            if context_owners.get((model_id, requested_context)) != api_key:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": {
                            "message": "Unknown or unauthorized HE context",
                            "code": "runtime_context_unavailable",
                        }
                    },
                )
            session.context_ids.add(requested_context)
            session.context_id = requested_context
        if session.execution == "seeded-preparation":
            if (
                sum(item.execution == "seeded-preparation" for item in sessions.values())
                >= config.prepared_session_capacity
            ):
                raise HTTPException(
                    status_code=503,
                    detail={"error": {"message": "Prepared session capacity exceeded"}},
                )
            inventory_id = str(body.get("inventory_id") or "")
            with inventory_lock:
                inventory = sessions.get(inventory_id)
                if (
                    inventory is None
                    or inventory.execution != "seeded-inventory"
                    or inventory.api_key != api_key
                    or inventory.model_id != model_id
                    or not inventory.inventory_ready
                    or inventory.canceled
                    or inventory.completed
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={"error": {"message": "Prepared inventory is not ready"}},
                    )
                try:
                    reservation_start = int(body.get("inventory_start", -1))
                    reservation_rows = int(body.get("inventory_rows", 0))
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": {"message": "Invalid inventory reservation"}},
                    ) from exc
                if (
                    reservation_start != inventory.inventory_next_row
                    or reservation_rows <= 0
                    or reservation_start + reservation_rows > inventory.inventory_rows
                    or set(inventory.inventory_roots) != set(inventory.inventory_stages)
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={"error": {"message": "Inventory reservation is unavailable"}},
                    )
                reserved_attempts = frozenset(
                    (stage_id, derive_online_attempt_id(root, row))
                    for stage_id, root in inventory.inventory_roots.items()
                    for row in range(reservation_start, reservation_start + reservation_rows)
                )
                inventory.inventory_next_row += reservation_rows
                inventory.last_active = time.monotonic()
            session.inventory_id = inventory_id
            session.reservation_start = reservation_start
            session.reservation_rows = reservation_rows
            session.reserved_attempts = reserved_attempts
            expected_authorization = None
        else:
            expected_authorization = None
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
                "runtime": {
                    "privacy_mode": imported_manifest.metadata.get(
                        "privacy_mode", config.privacy_mode
                    ),
                    "privacy_protocol": imported_manifest.metadata.get("privacy_protocol"),
                    "online_fhe": bool(imported_manifest.metadata.get("online_fhe", False)),
                    "preprocessed": bool(
                        imported_manifest.metadata.get("preprocessed", False)
                    ),
                    "model_privacy_threat_model": imported_manifest.metadata.get(
                        "model_privacy_threat_model"
                    ),
                    "malicious_client_model_privacy": imported_manifest.metadata.get(
                        "malicious_client_model_privacy"
                    ),
                    "runtime": imported_manifest.metadata.get("client_runtime"),
                    "engine": engine_name,
                    "fingerprint": imported_manifest.fingerprint,
                    "max_output_tokens": imported_manifest.context_length,
                    "stage_count": len(imported_manifest.stages),
                },
            }
        result = {
            "id": session.id,
            "object": "runtime.session",
            "response_id": session.response_id,
            "model": model_id,
            "manifest": session_manifest,
            "websocket_path": f"/v1/runtime/ws/{session.id}",
        }
        if expected_authorization is not None:
            result["preparation_authorization"] = {
                "body_fingerprint": expected_authorization.body_fingerprint,
                "stage_commitment": expected_authorization.stage_commitment,
                "weight_bits": expected_authorization.weight_bits,
                "activation_bits": expected_authorization.activation_bits,
                "max_attempts": expected_authorization.max_attempts,
            }
        return result

    @app.post("/v1/runtime/inventories")
    async def create_prepared_inventory(
        request: Request, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        body = await request.json()
        cleanup_prepared_sessions(reclaim_terminal=True)
        audit("inventory", json.dumps(body, separators=(",", ":")).encode())
        model_id = str(body.get("model", ""))
        manifest = imported.get(model_id)
        engine_name = model_engine_routes.get(model_id)
        if manifest is None or engine_name is None or not config.provider_push_api_key:
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "Prepared inventory runtime is unavailable"}},
            )
        try:
            rows = int(body.get("rows", 0))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"error": {"message": "Invalid rows"}}) from exc
        if rows <= 0 or rows > manifest.context_length:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": "Inventory rows exceed model context"}},
            )
        engine = engines[engine_name]
        stage_ids = engine.seeded_stage_ids(model_id)
        if not stage_ids:
            raise HTTPException(status_code=400, detail={"error": {"message": "No prepared stages"}})
        max_attempts = rows * len(stage_ids)
        stage_map = {stage.id: stage for stage in manifest.stages}
        reserved_bytes = sum(
            rows
            * stage_map[stage_id].out_features
            * (engine.seeded_profile(model_id, stage_id).wire_bits // 8)
                + 8 * PREPARATION_MAX_IDENTIFIER_BYTES
                + 4_096
            for stage_id in stage_ids
        )
        with inventory_lock:
            active_inventories = [
                item
                for item in sessions.values()
                if item.execution == "seeded-inventory"
                and not item.canceled
                and not item.completed
            ]
            if (
                max_attempts > config.rendezvous_max_attempts_per_session
                or sum(item.inventory_reserved_entries for item in active_inventories)
                + max_attempts
                > config.rendezvous_capacity
                or sum(item.inventory_reserved_bytes for item in active_inventories)
                + reserved_bytes
                > config.rendezvous_max_bytes
                or len(active_inventories) >= config.prepared_session_capacity
            ):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": {
                            "message": "Prepared inventory capacity exhausted"
                        }
                    },
                )
        session_id = new_id("rti")
        session = RuntimeSession(session_id, "", model_id, api_key)
        session.execution = "seeded-inventory"
        session.inventory_rows = rows
        session.inventory_stages = frozenset(stage_ids)
        session.inventory_reserved_entries = max_attempts
        session.inventory_reserved_bytes = reserved_bytes
        session_authorization = getattr(
            engines[engine_name], "seeded_session_authorization", None
        )
        if session_authorization is None:
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "Transformer engine cannot authorize inventory"}},
            )
        expected = session_authorization(model_id, session_id, max_attempts)
        try:
            rendezvous.register_session(expected)
        except RendezvousError as exc:
            raise HTTPException(
                status_code=503, detail={"error": {"message": str(exc)}}
            ) from exc
        sessions[session_id] = session
        return {
            "id": session_id,
            "object": "runtime.inventory",
            "model": model_id,
            "status": "preparing",
            "rows": rows,
            "stage_count": len(stage_ids),
            "preparation_authorization": {
                "body_fingerprint": expected.body_fingerprint,
                "stage_commitment": expected.stage_commitment,
                "weight_bits": expected.weight_bits,
                "activation_bits": expected.activation_bits,
                "max_attempts": expected.max_attempts,
                "rows": expected.rows,
                "stage_ids": list(expected.stage_ids),
            },
        }

    def get_session(session_id: str, api_key: str) -> RuntimeSession:
        cleanup_prepared_sessions()
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(
                status_code=404, detail={"error": {"message": "Unknown runtime session"}}
            )
        if session.api_key != api_key:
            raise HTTPException(
                status_code=403,
                detail={"error": {"message": "Runtime session belongs to another credential"}},
            )
        session.last_active = time.monotonic()
        return session

    def prepared_profile(engine: Any, model_id: str, stage_id: str) -> SeededRingProfile:
        getter = getattr(engine, "seeded_profile", None)
        if getter is None:
            raise ProtocolError("transformer engine lacks seeded ring profiles")
        return getter(model_id, stage_id)

    def validate_prepared_request(
        session: RuntimeSession,
        stage_id: str,
        request: MaskedStageRequest,
        engine: Any,
    ) -> None:
        if session.canceled or session.completed:
            raise ProtocolError("session is already terminal")
        if session.execution != "seeded-preparation":
            raise ProtocolError("session does not accept prepared corrections")
        inventory = sessions.get(session.inventory_id or "")
        if (
            inventory is None
            or inventory.execution != "seeded-inventory"
            or not inventory.inventory_ready
            or inventory.canceled
            or inventory.completed
            or inventory.api_key != session.api_key
            or inventory.model_id != session.model_id
        ):
            raise ProtocolError("prepared inventory is not active")
        inventory.last_active = time.monotonic()
        validate_attempt_id(request.correlation_id)
        if (
            request.session_id != inventory.id
            or request.model != session.model_id
            or request.stage_id != stage_id
            or stage_id not in inventory.inventory_stages
            or (request.stage_id, request.correlation_id) not in session.reserved_attempts
        ):
            raise ProtocolError("prepared activation session/model/stage mismatch")
        validate_activation = getattr(engine, "validate_seeded_activation", None)
        if validate_activation is None:
            raise ProtocolError("transformer engine cannot validate prepared activations")
        validate_activation(request)
        profile = SeededRingProfile(
            request.signed_output_bound,
            request.ring or "prime",
            request.modulus,
            request.wire_bits,
        )
        if profile != prepared_profile(engine, session.model_id, stage_id):
            raise ProtocolError("prepared activation ring profile mismatch")

    async def run_prepared_requests(
        session: RuntimeSession,
        stage_id: str,
        engine: Any,
        requests: list[MaskedStageRequest],
        engine_payloads: list[bytes],
        runner: Callable[[list[bytes]], Any],
    ) -> tuple[list[CorrectionPush], list[bytes]]:
        for prepared_request in requests:
            validate_prepared_request(session, stage_id, prepared_request, engine)
        activations = []
        try:
            for prepared_request in requests:
                activations.append(rendezvous.activate(prepared_request))
        except BaseException:
            for prepared_request, entry in zip(requests, activations):
                rendezvous.abort(prepared_request, entry)
            raise
        if session.canceled or session.completed:
            for prepared_request, entry in zip(requests, activations, strict=True):
                rendezvous.abort(prepared_request, entry)
            raise ProtocolError("session is already terminal")
        correction_tasks = [
            asyncio.create_task(rendezvous.correction(prepared_request, entry))
            for prepared_request, entry in zip(requests, activations, strict=True)
        ]
        results_task = asyncio.create_task(runner(engine_payloads))
        try:
            corrections, results = await asyncio.gather(
                asyncio.gather(*correction_tasks), results_task
            )
        except BaseException:
            for task in correction_tasks:
                task.cancel()
            results_task.cancel()
            await asyncio.gather(*correction_tasks, results_task, return_exceptions=True)
            for prepared_request, entry in zip(requests, activations, strict=True):
                rendezvous.abort(prepared_request, entry)
            raise
        if session.canceled or session.completed:
            raise ProtocolError("session is already terminal")
        return corrections, results

    async def execute_prepared_payloads(
        session: RuntimeSession,
        stage_id: str,
        engine: Any,
        payloads: list[bytes],
        runner: Callable[[list[bytes]], Any],
    ) -> list[bytes]:
        manifest = imported.get(session.model_id)
        if manifest is None:
            raise ProtocolError("prepared session model is unavailable")
        batch_rows = (
            prepared_stage_batch_rows(payloads[0]) if len(payloads) == 1 else None
        )
        if batch_rows is not None:
            batch = PreparedStageBatchRequest.unpack(
                payloads[0],
                max_rows=min(manifest.context_length, config.prepared_stage_batch_rows),
                max_tensor_elements=config.prepared_tensor_max_elements,
            )
            inventory = sessions.get(session.inventory_id or "")
            root = inventory.inventory_roots.get(stage_id) if inventory is not None else None
            if root is None:
                raise ProtocolError("prepared inventory stage is unavailable")
            if batch.wire_bits != root.wire_bits:
                raise ProtocolError("prepared stage batch wire width mismatch")

            def batch_request(correlation_id: str, masked_input: np.ndarray):
                return MaskedStageRequest(
                    model=root.model,
                    stage_id=stage_id,
                    correlation_id=correlation_id,
                    masked_input=masked_input,
                    activation_scales=np.ones(1, dtype=np.float32),
                    modulus=root.modulus,
                    wire_bits=root.wire_bits,
                    ring=root.ring,
                    body_fingerprint=root.body_fingerprint,
                    weight_digest=root.weight_digest,
                    weight_bits=root.weight_bits,
                    activation_bits=root.activation_bits,
                    session_id=root.session_id,
                    out_features=root.out_features,
                    signed_output_bound=root.signed_output_bound,
                )

            engine_request = batch_request(batch.batch_id, batch.masked_input)
            validation_request = batch_request(
                batch.correlation_ids[0], batch.masked_input[0:1]
            )
            validate_prepared_request(session, stage_id, validation_request, engine)
            if any(
                (stage_id, correlation_id) not in session.reserved_attempts
                for correlation_id in batch.correlation_ids
            ):
                raise ProtocolError("stage correlation was not reserved for this response")
            corrections = rendezvous.consume_preloaded_batch(
                engine_request, batch.correlation_ids
            )
            results = await runner([engine_request.pack()])
            if len(results) != 1:
                raise ProtocolError("Runtime engine returned the wrong batch result count")
            result = MaskedStageResponse.unpack(results[0])
            correction = np.concatenate(
                [item.correction for item in corrections], axis=0
            )
            if (
                result.correlation_id != batch.batch_id
                or result.stage_id != stage_id
                or result.ring != root.ring
                or result.modulus != root.modulus
                or result.wire_bits != root.wire_bits
                or result.masked_output.shape != correction.shape
            ):
                raise ProtocolError("prepared inference batch result mismatch")
            masked = (
                result.masked_output.astype(np.int64) + correction.astype(np.int64)
            ) % result.modulus
            return [
                PreparedStageBatchResponse(
                    batch_id=batch.batch_id,
                    masked_output=masked.astype(np.uint32),
                    wire_bits=result.wire_bits,
                    server_ns=result.server_ns,
                ).pack()
            ]

        requests = [
            MaskedStageRequest.unpack(
                payload,
                max_rows=manifest.context_length,
                max_tensor_elements=config.prepared_tensor_max_elements,
            )
            for payload in payloads
        ]
        if len(requests) == 1:
            request = requests[0]
            validate_prepared_request(session, stage_id, request, engine)
            corrections = rendezvous.consume_preloaded_batch(
                request,
                (request.correlation_id,),
            )
            results = await runner(payloads)
        else:
            corrections, results = await run_prepared_requests(
                session, stage_id, engine, requests, payloads, runner
            )
        if session.canceled or session.completed:
            raise ProtocolError("session is already terminal")
        if len(results) != len(payloads):
            raise ProtocolError("Runtime engine returned the wrong result count")
        combined: list[bytes] = []
        for prepared_request, correction, payload in zip(
            requests, corrections, results, strict=True
        ):
            result = MaskedStageResponse.unpack(payload)
            if (
                result.correlation_id != prepared_request.correlation_id
                or result.stage_id != stage_id
                or result.ring != prepared_request.ring
                or result.modulus != prepared_request.modulus
                or result.wire_bits != prepared_request.wire_bits
                or result.masked_output.shape != correction.correction.shape
            ):
                raise ProtocolError("prepared inference result mismatch")
            masked = (
                result.masked_output.astype(np.int64) + correction.correction.astype(np.int64)
            ) % result.modulus
            combined.append(
                MaskedStageResponse(
                    correlation_id=result.correlation_id,
                    masked_output=masked.astype(np.uint32),
                    modulus=result.modulus,
                    wire_bits=result.wire_bits,
                    server_ns=result.server_ns,
                    stage_id=result.stage_id,
                    ring=result.ring,
                ).pack()
            )
        return combined

    def prepared_endpoint_context(session_id: str) -> tuple[RuntimeSession, Any, int]:
        cleanup_prepared_sessions()
        session = sessions.get(session_id)
        if session is None:
            raise ProtocolError("unknown correction session")
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        manifest = imported.get(session.model_id)
        if engine is None or manifest is None:
            raise ProtocolError("correction session has no transformer engine")
        if session.canceled or session.completed or session.execution != "seeded-inventory":
            raise ProtocolError("correction inventory is not active")
        session.last_active = time.monotonic()
        return session, engine, manifest.context_length

    def accept_prepared_correction(
        session_id: str,
        raw: bytes | memoryview,
        *,
        attempt_id: str | None = None,
        correction: CorrectionPush | None = None,
    ) -> bytes:
        session, engine, max_rows = prepared_endpoint_context(session_id)
        if correction is None:
            correction = CorrectionPush.unpack(
                raw,
                max_rows=max_rows,
                max_tensor_elements=config.prepared_tensor_max_elements,
            )
        elif correction.rows > max_rows:
            raise ProtocolError("correction rows exceed the model context length")
        if correction.session_id != session_id:
            raise ProtocolError("correction session mismatch")
        if attempt_id is not None and correction.attempt_id != attempt_id:
            raise ProtocolError("correction route mismatch")
        if correction.model != session.model_id:
            raise ProtocolError("correction model mismatch")
        if correction.rows != session.inventory_rows:
            raise ProtocolError("correction inventory row count mismatch")
        if correction.stage_id not in session.inventory_stages:
            raise ProtocolError("correction inventory stage mismatch")
        validate_correction = getattr(engine, "validate_seeded_correction", None)
        if validate_correction is None:
            raise ProtocolError("transformer engine cannot validate corrections")
        validate_correction(correction)
        with inventory_lock:
            if correction.stage_id in session.prepared_stages:
                raise ProtocolError("correction inventory stage was already prepared")
            rendezvous.preload_bulk(correction, len(raw))
            session.prepared_stages.add(correction.stage_id)
            session.inventory_roots[correction.stage_id] = PreparationRequest(
                attempt_id=correction.attempt_id,
                session_id=correction.session_id,
                model=correction.model,
                body_fingerprint=correction.body_fingerprint,
                stage_id=correction.stage_id,
                weight_digest=correction.weight_digest,
                rows=correction.rows,
                in_features=correction.in_features,
                out_features=correction.out_features,
                weight_bits=correction.weight_bits,
                activation_bits=correction.activation_bits,
                signed_output_bound=correction.signed_output_bound,
                ring=correction.ring,
                modulus=correction.modulus,
                wire_bits=correction.wire_bits,
                seed=b"\0" * 32,
            )
        return PreparationAck(
            correction.attempt_id,
            correction.stage_id,
            len(raw),
            correction.server_ns,
        ).pack()

    @app.post("/v1/runtime/inventories/{session_id}/authorize")
    async def authorize_prepared_session(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        auth_push_token(authorization)
        try:
            session, engine, _ = prepared_endpoint_context(session_id)
            raw = await _read_limited_body(request, config.preparation_request_max_bytes)
            value = SessionAuthorization.unpack(raw)
            if value.session_id != session_id or value.model != session.model_id:
                raise ProtocolError("session authorization route/model mismatch")
            validate_authorization = getattr(engine, "validate_seeded_session_authorization", None)
            if validate_authorization is None:
                raise ProtocolError("transformer engine cannot validate session authorization")
            validate_authorization(value)
            rendezvous.authorize_session(value, len(raw))
        except (ProtocolError, RendezvousError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": str(exc), "code": "session_authorization_rejected"}},
            ) from exc
        return FastAPIResponse(
            SessionAuthorizationAck(value.session_id).pack(),
            media_type=BINARY_MEDIA_TYPE,
        )

    @app.post("/v1/runtime/inventories/{session_id}/corrections/{attempt_id}")
    async def push_prepared_correction(
        session_id: str,
        attempt_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        auth_push_token(authorization)
        try:
            raw = await _read_limited_body(request, config.prepared_payload_max_bytes)
            ack = accept_prepared_correction(session_id, raw, attempt_id=attempt_id)
        except (ProtocolError, RendezvousError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": str(exc), "code": "correction_rejected"}},
            ) from exc
        return FastAPIResponse(ack, media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/runtime/inventories/{session_id}/ready")
    async def seal_prepared_inventory(
        session_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        with inventory_lock:
            if session.execution != "seeded-inventory" or session.canceled or session.completed:
                raise HTTPException(
                    status_code=409, detail={"error": {"message": "Inventory is not active"}}
                )
            if session.prepared_stages != set(session.inventory_stages):
                raise HTTPException(
                    status_code=409,
                    detail={"error": {"message": "Inventory preparation is incomplete"}},
                )
            session.inventory_ready = True
            session.last_active = time.monotonic()
        return {
            "id": session.id,
            "object": "runtime.inventory",
            "status": "ready",
            "rows": session.inventory_rows,
            "stage_count": len(session.inventory_stages),
        }

    @app.get("/v1/runtime/inventories/{session_id}")
    async def get_prepared_inventory(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        cleanup_prepared_sessions(reclaim_terminal=True)
        session = sessions.get(session_id)
        if (
            session is None
            or session.execution != "seeded-inventory"
            or session.api_key != api_key
            or session.canceled
            or session.completed
        ):
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": "Unknown prepared inventory"}},
            )
        session.last_active = time.monotonic()
        return {
            "id": session_id,
            "status": "ready" if session.inventory_ready else "preparing",
            "rows": session.inventory_rows,
            "next_row": session.inventory_next_row,
            "stage_count": len(session.prepared_stages),
        }

    @app.post("/v1/runtime/inventories/{session_id}/cancel")
    async def cancel_prepared_inventory(
        session_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        if session.execution != "seeded-inventory":
            raise HTTPException(
                status_code=409, detail={"error": {"message": "Session is not an inventory"}}
            )
        session.canceled = True
        rendezvous.terminal(session.id)
        return {"id": session.id, "object": "runtime.inventory", "status": "cancelled"}

    @app.websocket("/v1/runtime/corrections/ws")
    async def push_prepared_correction_websocket(websocket: WebSocket) -> None:
        token = bearer_token(websocket.headers.get("authorization"))
        expected = config.provider_push_api_key
        if expected is None or token is None or not secrets.compare_digest(token, expected):
            await websocket.close(code=4401, reason="invalid correction push credential")
            return
        offered = {
            item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
        }
        if CORRECTION_CHANNEL_SUBPROTOCOL not in offered:
            await websocket.close(code=4406, reason="correction subprotocol required")
            return
        await websocket.accept(subprotocol=CORRECTION_CHANNEL_SUBPROTOCOL)
        correction_channel_metrics["connections"] += 1
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                frame = message.get("bytes")
                if not isinstance(frame, bytes):
                    correction_channel_metrics["failures"] += 1
                    await websocket.close(code=4400, reason="binary correction frame required")
                    return
                started = time.perf_counter_ns()
                try:
                    if not frame or len(frame) > config.prepared_payload_max_bytes:
                        raise CorrectionChannelError("correction frame is too large")
                    correction = CorrectionPush.unpack(
                        frame,
                        max_tensor_elements=config.prepared_tensor_max_elements,
                    )
                    accept_prepared_correction(
                        correction.session_id,
                        frame,
                        correction=correction,
                    )
                    await websocket.send_bytes(
                        PreparationAck(
                            correction.attempt_id,
                            correction.stage_id,
                            len(frame),
                            0,
                        ).pack()
                    )
                except (
                    CorrectionChannelError,
                    ProtocolError,
                    RendezvousError,
                    ValueError,
                    RuntimeError,
                ):
                    correction_channel_metrics["failures"] += 1
                    await websocket.close(code=4409, reason="correction rejected")
                    return
                correction_channel_metrics["frames"] += 1
                correction_channel_metrics["correction_bytes"] += len(frame)
                correction_channel_metrics["processing_ns"] += time.perf_counter_ns() - started
        except WebSocketDisconnect:
            return

    @app.put("/v1/runtime/sessions/{session_id}/contexts/{context_id}")
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
            raise HTTPException(
                status_code=400, detail={"error": {"message": "Empty public context"}}
            )
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        if session.model_id not in bfv_servers and engine is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "BFV runtime is not installed",
                        "code": "runtime_unavailable",
                    }
                },
            )
        try:
            if engine is not None:
                register = getattr(engine, "register_bfv_context", None)
                if register is None:
                    raise ValueError("Runtime engine does not support BFV correlations")
                await asyncio.to_thread(register, session.model_id, context_id, payload)
            else:
                bfv_servers[session.model_id].register_context(context_id, payload)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "invalid_he_context"}},
            )
        session.context_id = context_id
        session.context_ids.add(context_id)
        context_owners[(session.model_id, context_id)] = api_key
        return {
            "id": context_id,
            "object": "runtime.context",
            "status": "ready",
            "bytes": len(payload),
            "reusable": True,
        }

    @app.post("/v1/runtime/sessions/{session_id}/correlations/bfv")
    async def bfv_correlation(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        context_id = request.headers.get("x-he-context-id") or session.context_id
        if context_id is None or context_id not in session.context_ids:
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": "Register the requested HE context first"}},
            )
        stage_id = request.headers.get("x-he-stage-id")
        engine_name = model_engine_routes.get(session.model_id)
        engine = engines.get(engine_name) if engine_name else None
        if session.model_id not in bfv_servers and engine is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": {
                        "message": "BFV runtime is not installed",
                        "code": "runtime_unavailable",
                    }
                },
            )
        encrypted_mask = await request.body()
        audit("bfv_mask", encrypted_mask)
        try:
            if engine is not None:
                if not stage_id:
                    raise ValueError("x-he-stage-id is required for transformer BFV correlations")
                evaluate = getattr(engine, "evaluate_bfv_correlation", None)
                if evaluate is None:
                    raise ValueError("Runtime engine does not support BFV correlations")
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
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "runtime_correlation_failed"}},
            )
        session.correlation_steps += 1
        return FastAPIResponse(result, media_type="application/octet-stream")

    @app.post("/v1/runtime/sessions/{session_id}/correlations/bfv/batch")
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
            raise HTTPException(
                status_code=409,
                detail={"error": {"message": "Register the requested HE context first"}},
            )
        if not x_he_stage_id:
            raise HTTPException(
                status_code=400, detail={"error": {"message": "x-he-stage-id is required"}}
            )
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
                raise ValueError("transformer BFV batch requires a runtime engine")
            evaluate_many = getattr(engine, "evaluate_bfv_correlations", None)
            if evaluate_many is None:
                evaluate_one = getattr(engine, "evaluate_bfv_correlation", None)
                if evaluate_one is None:
                    raise ValueError("Runtime engine does not support BFV correlations")
                results = await asyncio.to_thread(
                    lambda: [
                        evaluate_one(session.model_id, x_he_stage_id, context_id, item)
                        for item in encrypted_masks
                    ]
                )
            else:
                cancel_event = threading.Event()
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        evaluate_many,
                        session.model_id,
                        x_he_stage_id,
                        context_id,
                        encrypted_masks,
                        cancel_event=cancel_event,
                    )
                )
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
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "runtime_correlation_failed"}},
            ) from exc
        session.correlation_steps += len(results)
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/runtime/sessions/{session_id}/correlations/proprietary/bfv/batch")
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
                raise ValueError("proprietary BFV batch requires a runtime engine")
            evaluate_many = getattr(engine, "evaluate_bfv_blinded_correlations", None)
            if evaluate_many is None:
                raise ValueError("Runtime engine does not support output-blinded correlations")
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
                detail={"error": {"message": str(exc), "code": "runtime_correlation_failed"}},
            ) from exc
        session.correlation_steps += len(results)
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/runtime/sessions/{session_id}/correlations/proprietary/local-test")
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
                detail={"error": {"message": "Runtime engine cannot create blinded test correlations"}},
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

    @app.post("/v1/runtime/sessions/{session_id}/correlations/local-test")
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
                raise HTTPException(
                    status_code=400, detail={"error": {"message": "stage_id is required"}}
                )
            create = getattr(engine, "create_local_correlations", None)
            if create is None:
                raise HTTPException(
                    status_code=501,
                    detail={"error": {"message": "Runtime engine cannot create test correlations"}},
                )
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

    async def execute_envelope(
        session: RuntimeSession, envelope: ProtocolEnvelope
    ) -> ProtocolEnvelope:
        envelope.verify(session.key)
        session.replay.accept(envelope)
        logical_rows = 1
        if session.canceled or session.completed:
            raise ProtocolError("session is already terminal")
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
            if session.execution == "seeded-preparation":
                logical_rows = prepared_stage_batch_rows(envelope.payload) or 1
                engine_name = model_engine_routes.get(session.model_id)
                engine = engines.get(engine_name) if engine_name else None
                if engine is None:
                    raise ProtocolError("session model has no transformer engine")

                async def run_prepared(values: list[bytes]) -> list[bytes]:
                    return [await scheduler.submit(values[0])]

                result = (
                    await execute_prepared_payloads(
                        session, stage_id, engine, [envelope.payload], run_prepared
                    )
                )[0]
            else:
                result = await scheduler.submit(envelope.payload)
            result_kind = "masked.transformer.stage.result"
        else:
            raise ProtocolError("unsupported runtime frame kind")
        session.online_steps += logical_rows
        return ProtocolEnvelope.create(
            request_id=envelope.request_id,
            session_id=session.id,
            model=session.model_id,
            kind=result_kind,
            sequence=envelope.sequence,
            payload=result,
            metadata={},
            key=session.key,
        )

    @app.post("/v1/runtime/sessions/{session_id}/execute")
    async def execute_http(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        try:
            execute_payload = await _read_limited_body(request, config.prepared_payload_max_bytes)
            audit("execute_http", execute_payload)
            envelope = unpack_envelope(execute_payload)
            result = await execute_envelope(session, envelope)
        except ProtocolError as exc:
            raise HTTPException(
                status_code=400, detail={"error": {"message": str(exc), "code": "invalid_runtime_frame"}}
            )
        return FastAPIResponse(pack_envelope(result), media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/runtime/sessions/{session_id}/stages/{stage_id}")
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
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": "Session model has no transformer engine"}},
            )
        stage = next((value for value in manifest.stages if value.id == stage_id), None)
        if stage is None:
            raise HTTPException(
                status_code=404, detail={"error": {"message": "Unknown transformer stage"}}
            )
        raw = await _read_limited_body(request, config.prepared_payload_max_bytes)
        audit("execute_stage_batch", raw)
        try:
            payloads = list(iter_length_prefixed(raw))
            if not payloads:
                raise ProtocolError("empty stage batch")
            compact_rows = (
                prepared_stage_batch_rows(payloads[0]) if len(payloads) == 1 else None
            )
            logical_rows = compact_rows or len(payloads)
            if logical_rows > config.prepared_stage_batch_rows:
                raise ProtocolError("stage prefill batch is too large")
            if session.execution == "seeded-preparation":

                async def run_prepared(values: list[bytes]) -> list[bytes]:
                    return await engine.execute_stage(session.model_id, stage, values)

                results = await execute_prepared_payloads(
                    session, stage_id, engine, payloads, run_prepared
                )
            else:
                if session.canceled or session.completed:
                    raise ProtocolError("session is already terminal")
                results = await engine.execute_stage(session.model_id, stage, payloads)
            if len(results) != len(payloads):
                raise ProtocolError("Runtime engine returned the wrong result count")
        except (ProtocolError, ValueError, RuntimeError) as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": str(exc), "code": "invalid_stage_batch"}},
            )
        session.online_steps += logical_rows
        return FastAPIResponse(encode_length_prefixed(results), media_type=BINARY_MEDIA_TYPE)

    @app.websocket("/v1/runtime/ws/{session_id}")
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
        await websocket.accept(subprotocol="pllm-runtime-v1")
        try:
            while True:
                ws_payload = await websocket.receive_bytes()
                if len(ws_payload) > config.prepared_payload_max_bytes:
                    raise ProtocolError("WebSocket frame is too large")
                audit("execute_websocket", ws_payload)
                envelope = unpack_envelope(ws_payload)
                result = await execute_envelope(session, envelope)
                await websocket.send_bytes(pack_envelope(result))
        except WebSocketDisconnect:
            return
        except ProtocolError as exc:
            await websocket.close(code=4400, reason=str(exc))

    @app.post("/v1/runtime/sessions/{session_id}/cancel")
    async def cancel_runtime_session(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        session = get_session(session_id, api_key)
        session.canceled = True
        session.last_active = time.monotonic()
        burn_prepared_reservation(session)
        retire_exhausted_inventory(session)
        return {"id": session_id, "status": "cancelled"}

    @app.post("/v1/runtime/sessions/{session_id}/complete")
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
        burn_prepared_reservation(session)
        retire_exhausted_inventory(session)
        # Operational record only: plaintext output remains in the client cache.
        value = {
            "id": session.response_id,
            "object": "response",
            "created_at": session.created_at,
            "status": "completed",
            "model": session.model_id,
            "output": [],
            "usage": body.get("usage"),
            "runtime_redacted": True,
        }
        responses.put(value, session.api_key)
        return value

    @app.post("/v1/responses")
    async def create_response(request: Request, authorization: str | None = Header(default=None)):
        api_key = auth_token(authorization)
        body = await request.json()
        model_id = str(body.get("model", ""))
        if model_id in private_models or model_id in model_engine_routes:
            raise HTTPException(
                status_code=426,
                headers={"Upgrade": "pllm-runtime/1"},
                detail={
                    "error": {
                        "message": "Private-runtime model requires pllm.runtime.OpenAI or PLLMTransport; plaintext input was rejected",
                        "type": "invalid_request_error",
                        "code": "runtime_client_required",
                    }
                },
            )
        try:
            adapter, upstream_model_id = backend_registry.resolve(model_id)
        except KeyError:
            raise HTTPException(
                status_code=404, detail={"error": {"message": f"Unknown model {model_id}"}}
            )
        upstream_body = dict(body)
        upstream_body["model"] = upstream_model_id

        def publicize(value: Any) -> Any:
            """Keep public model IDs stable when a backend route is prefixed."""
            if isinstance(value, dict):
                return {
                    key: (
                        model_id
                        if key == "model" and item == upstream_model_id
                        else publicize(item)
                    )
                    for key, item in value.items()
                }
            if isinstance(value, list):
                return [publicize(item) for item in value]
            return value

        if body.get("stream"):

            async def generate():
                async for raw_event in adapter.stream_response(upstream_body):
                    event = publicize(raw_event)
                    if event.get("type") == "response.completed" and isinstance(
                        event.get("response"), dict
                    ):
                        responses.put(event["response"], api_key)
                    yield sse_event(event)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
        try:
            value = publicize(await adapter.create_response(upstream_body))
        except Exception as exc:
            raise HTTPException(
                status_code=502, detail={"error": {"message": str(exc), "type": "backend_error"}}
            )
        responses.put(value, api_key)
        return JSONResponse(value)

    @app.get("/v1/responses/{response_id}")
    async def retrieve_response(
        response_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        try:
            return responses.get(response_id, api_key)
        except KeyError:
            raise HTTPException(status_code=404, detail={"error": {"message": "Unknown response"}})

    @app.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(
        response_id: str, authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        api_key = auth_token(authorization)
        for session in sessions.values():
            if session.response_id == response_id and session.api_key == api_key:
                session.canceled = True
                burn_prepared_reservation(session)
        try:
            return responses.cancel(response_id, api_key)
        except KeyError:
            value = {"id": response_id, "object": "response", "status": "cancelled", "output": []}
            responses.put(value, api_key)
            return value

    @app.get("/metrics")
    async def metrics(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        auth_token(authorization)
        cleanup_prepared_sessions(reclaim_terminal=True)
        return {
            "sessions": {
                "active": sum(not row.completed and not row.canceled for row in sessions.values()),
                "total": len(sessions),
                "online_steps": sum(row.online_steps for row in sessions.values()),
                "correlation_steps": sum(row.correlation_steps for row in sessions.values()),
            },
            "stage_schedulers": {
                **{model: scheduler.stats() for model, scheduler in schedulers.items()},
                **{
                    f"{model}:{stage}": scheduler.stats()
                    for (model, stage), scheduler in engine_schedulers.items()
                },
            },
            "engines": {
                name: (getattr(engine, "metrics")() if hasattr(engine, "metrics") else {})
                for name, engine in engines.items()
            },
            "correction_rendezvous": rendezvous.stats(),
            "correction_channel": dict(correction_channel_metrics),
        }

    from .telemetry import instrument_fastapi

    instrument_fastapi(app)
    return app
