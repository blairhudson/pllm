from __future__ import annotations

import asyncio
import ipaddress
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse, urlunparse

import httpx
import msgpack
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response as FastAPIResponse

from pllm.model_loader import model_from_runtime_spec, resolve_model
from pllm import _native

from .config import GatewayConfig
from .correction_channel import (
    CORRECTION_CHANNEL_PATH,
    CorrectionChannelError,
    CorrectionWebSocketClient,
)
from .preparation_protocol import (
    PREPARATION_PROTOCOL_VERSION,
    PreparationAck,
    PreparationRequest,
    PreparationRequestContext,
    SessionAuthorization,
    SessionAuthorizationAck,
    freivalds_session_id,
    validate_attempt_id,
)
from .privacy import PrivacyMode
from .protocol import BINARY_MEDIA_TYPE, ProtocolError
from .security import bearer_token
from .telemetry import record_protocol_bytes
from .transformer_engine import MaskedTransformerEngine, TransformerEngineError


def _freivalds_policy(value, max_elements):
    work = min(max_elements * max(value.max_attempts, 1) * 8, (1 << 64) - 1)
    return _native.FreivaldsPolicy(
        value.verification_target_failure_bits,
        value.max_attempts,
        max_elements,
        max_elements,
        max_elements,
        max_elements,
        work,
        value.rows,
        freivalds_session_id(value),
        False,
    )


@dataclass(slots=True)
class PreparationMetrics:
    authorization_calls: int = 0
    authorization_upload_bytes: int = 0
    authorization_download_bytes: int = 0
    authorization_push_bytes: int = 0
    calls: int = 0
    rows: int = 0
    upload_bytes: int = 0
    download_bytes: int = 0
    correction_push_attempts: int = 0
    correction_push_bytes: int = 0
    correction_push_ack_bytes: int = 0
    correction_channel_upload_bytes: int = 0
    correction_channel_download_bytes: int = 0
    correction_compute_ns: int = 0
    correction_push_ns: int = 0
    failures: int = 0


@dataclass(slots=True)
class _AuthorizedSession:
    authorization: SessionAuthorization
    owner: str
    attempts: set[bytes]
    stages: set[str]
    last_active: float
    active: bool = False


class PreparationSessionRegistry:
    """Bounded authorization state used to hydrate compact preparation requests."""

    def __init__(self, capacity: int, idle_timeout: float) -> None:
        if capacity <= 0 or idle_timeout <= 0:
            raise ValueError("preparation session registry limits must be positive")
        self.capacity = int(capacity)
        self.idle_timeout = float(idle_timeout)
        self._lock = threading.Lock()
        self._sessions: dict[str, _AuthorizedSession] = {}

    def _cleanup_locked(self, now: float) -> None:
        cutoff = now - self.idle_timeout
        for session_id, session in list(self._sessions.items()):
            if session.last_active <= cutoff:
                self._sessions.pop(session_id)

    def reserve(self, authorization: SessionAuthorization, owner: str) -> None:
        authorization._validate()
        with self._lock:
            now = time.monotonic()
            self._cleanup_locked(now)
            if authorization.session_id in self._sessions:
                raise ProtocolError("preparation session is already registered")
            if len(self._sessions) >= self.capacity:
                raise ProtocolError("preparation session capacity exceeded")
            self._sessions[authorization.session_id] = _AuthorizedSession(
                authorization, owner, set(), set(), now
            )

    def activate(self, session_id: str, owner: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.owner != owner:
                raise ProtocolError("preparation session is not registered")
            session.active = True
            session.last_active = time.monotonic()

    def release(self, session_id: str, owner: str, *, force: bool = False) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and session.owner == owner and (force or not session.active):
                self._sessions.pop(session_id)

    def authorization(self, session_id: str, owner: str) -> SessionAuthorization:
        with self._lock:
            self._cleanup_locked(time.monotonic())
            session = self._sessions.get(session_id)
            if session is None or session.owner != owner or not session.active:
                raise ProtocolError("preparation session is not authorized")
            return session.authorization

    def consume(
        self,
        session_id: str,
        stage_id: str,
        rows: int,
        attempt_id: str,
        owner: str,
    ) -> None:
        validate_attempt_id(attempt_id)
        attempt = bytes.fromhex(attempt_id)
        with self._lock:
            now = time.monotonic()
            self._cleanup_locked(now)
            session = self._sessions.get(session_id)
            if session is None or session.owner != owner or not session.active:
                raise ProtocolError("preparation session is not authorized")
            if (
                stage_id not in session.authorization.stage_ids
                or rows != session.authorization.rows
            ):
                raise ProtocolError("preparation request is outside the authorized inventory")
            if attempt in session.attempts:
                raise ProtocolError("preparation attempt was already consumed")
            if stage_id in session.stages:
                raise ProtocolError("preparation stage was already consumed")
            if len(session.attempts) >= session.authorization.max_attempts:
                raise ProtocolError("preparation session attempt capacity exceeded")
            session.attempts.add(attempt)
            session.stages.add(stage_id)
            session.last_active = now

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._cleanup_locked(time.monotonic())
            return {
                "sessions": len(self._sessions),
                "authorized_sessions": sum(s.active for s in self._sessions.values()),
                "attempts": sum(len(s.attempts) for s in self._sessions.values()),
            }

    def close(self) -> None:
        with self._lock:
            self._sessions.clear()


def _validate_inference_url(value: str | None) -> str:
    if not value:
        raise ValueError("preparation inference URL is required")
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path.rstrip("/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("preparation inference URL must be an HTTP(S) origin")
    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("preparation inference URL has an invalid port") from exc
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if parsed.scheme != "https" and not loopback:
        raise ValueError("preparation inference URL must use HTTPS outside loopback")
    return urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _correction_websocket_url(inference_url: str) -> str:
    parsed = urlparse(inference_url)
    return urlunparse(
        (
            "wss" if parsed.scheme == "https" else "ws",
            parsed.netloc,
            CORRECTION_CHANNEL_PATH,
            "",
            "",
            "",
        )
    )


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


def create_preparation_app(
    config: GatewayConfig,
    engine: MaskedTransformerEngine,
    *,
    audit_hook: Callable[[str, bytes], None] | None = None,
    push_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    if PrivacyMode.parse(config.privacy_mode) is not PrivacyMode.PUBLIC:
        raise ValueError("preparation service supports public-weight models only")
    inference_url = _validate_inference_url(config.preparation_inference_url)
    if not config.api_keys or not config.preparation_push_api_key:
        raise ValueError("preparation client and push credentials are required")
    if config.preparation_push_api_key in config.api_keys:
        raise ValueError("preparation client and push credentials must be distinct")
    metrics = PreparationMetrics()
    session_registry = PreparationSessionRegistry(
        config.prepared_session_capacity,
        config.prepared_session_idle_seconds,
    )
    freivalds_policies: dict[str, Any] = {}
    owns_push_client = push_client is None
    push_http = push_client or httpx.AsyncClient(
        base_url=inference_url,
        timeout=config.preparation_push_timeout_seconds,
    )
    correction_channel = (
        None
        if push_client is not None
        else CorrectionWebSocketClient(
            _correction_websocket_url(inference_url),
            config.preparation_push_api_key,
            timeout=config.preparation_push_timeout_seconds,
            max_payload_bytes=config.prepared_payload_max_bytes,
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        for source in config.engine_models:
            body = dict(source)
            model = model_from_runtime_spec(body)
            if model.kind not in {"huggingface", "safetensors", "vllm", "mlx", "mlx-lm"}:
                raise RuntimeError(
                    f"preparation service does not support model kind {model.kind!r}"
                )
            resolved = await asyncio.to_thread(resolve_model, model)
            await engine.load(resolved.manifest)
        try:
            yield
        finally:
            if correction_channel is not None:
                await correction_channel.close()
            session_registry.close()
            if owns_push_client:
                await push_http.aclose()
            close = getattr(engine, "close", None)
            if close:
                value = close()
                if asyncio.iscoroutine(value):
                    await value

    app = FastAPI(title="PLLM Trusted Preparation Service", version="1", lifespan=lifespan)
    app.state.config = config
    app.state.engine = engine
    app.state.metrics = metrics
    app.state.session_registry = session_registry

    def authenticate(value: str | None) -> str:
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

    @app.get("/healthz")
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "role": "trusted-preparation",
            "models": len(engine.models),
        }

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authenticate(authorization)
        rows = []
        for model in engine.models.values():
            manifest = model.manifest
            rows.append(
                {
                    "id": manifest.id,
                    "object": "model",
                    "owned_by": "pllm-preparation",
                    "preparation": {
                        "protocol": "seeded-inventory/v1",
                        "architecture": manifest.architecture,
                        "stage_count": len(manifest.stages),
                        "body_fingerprint": manifest.metadata.get("body_fingerprint"),
                        "stage_commitment": manifest.metadata.get("seeded_stage_commitment"),
                        "weight_bits": engine.weight_bits,
                        "activation_bits": engine.activation_bits,
                    },
                }
            )
        return {"object": "list", "data": rows}

    @app.post("/v1/preparation/inventories/{session_id}/authorize")
    async def authorize_inventory(
        session_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        owner = authenticate(authorization)
        raw = await _read_limited_body(request, config.preparation_request_max_bytes)
        reserved = False
        policy_reserved = False
        try:
            value = SessionAuthorization.unpack(raw)
            if value.session_id != session_id:
                raise ProtocolError("session authorization route mismatch")
            validate_authorization = getattr(engine, "validate_seeded_session_authorization", None)
            if validate_authorization is None:
                raise ProtocolError("preparation engine cannot validate session authorization")
            validate_authorization(value)
            session_registry.reserve(value, owner)
            reserved = True
            if value.verification_component != "none":
                freivalds_policies[session_id] = _freivalds_policy(
                    value, config.prepared_tensor_max_elements
                )
                policy_reserved = True
            pushed = await push_http.post(
                f"/v1/runtime/inventories/{session_id}/authorize",
                headers={
                    "Authorization": f"Bearer {config.preparation_push_api_key}",
                    "Content-Type": BINARY_MEDIA_TYPE,
                },
                content=raw,
            )
            pushed.raise_for_status()
            ack = SessionAuthorizationAck.unpack(pushed.content)
            if ack.session_id != session_id:
                raise ProtocolError("inference session authorization acknowledgement mismatch")
            session_registry.activate(session_id, owner)
            result = ack.pack()
        except (ProtocolError, TransformerEngineError, ValueError, httpx.HTTPError) as exc:
            if policy_reserved:
                freivalds_policies.pop(session_id, None)
            if reserved:
                session_registry.release(session_id, owner)
            metrics.failures += 1
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": str(exc),
                        "code": "session_authorization_failed",
                    }
                },
            ) from exc
        metrics.authorization_calls += 1
        metrics.authorization_upload_bytes += len(raw)
        metrics.authorization_download_bytes += len(result)
        metrics.authorization_push_bytes += len(raw)
        if audit_hook is not None:
            audit_hook(
                "authorize_session",
                msgpack.packb(
                    {
                        "model": value.model,
                        "request_bytes": len(raw),
                        "response_bytes": len(result),
                    },
                    use_bin_type=True,
                ),
            )
        return FastAPIResponse(result, media_type=BINARY_MEDIA_TYPE)

    @app.post("/v1/preparation/inventories/{session_id}/cancel")
    async def cancel_inventory(
        session_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        owner = authenticate(authorization)
        session_registry.release(session_id, owner, force=True)
        freivalds_policies.pop(session_id, None)
        return {"id": session_id, "status": "canceled"}

    @app.post("/v1/preparation/inventories/{session_id}/stages/{stage_id}")
    async def prepare_inventory_stage(
        session_id: str,
        stage_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> FastAPIResponse:
        owner = authenticate(authorization)
        raw = await _read_limited_body(request, config.preparation_request_max_bytes)
        try:
            try:
                envelope = msgpack.unpackb(raw, raw=False, strict_map_key=False)
            except Exception as exc:
                raise ProtocolError("invalid preparation request") from exc
            if not isinstance(envelope, dict) or set(envelope) != {"v", "a", "h", "r", "z"}:
                raise ProtocolError("invalid preparation request schema")
            if int(envelope["v"]) != PREPARATION_PROTOCOL_VERSION:
                raise ProtocolError("unsupported preparation protocol")
            if str(envelope.get("h", "")) != session_id:
                raise ProtocolError("inventory route mismatch")
            session = session_registry.authorization(session_id, owner)
            request_stage_id = stage_id
            model = engine.models[session.model]
            stage = await engine.stage_metadata(session.model, request_stage_id)
            profile = engine.seeded_profile(session.model, request_stage_id)
            value = PreparationRequest.unpack(
                raw,
                context=PreparationRequestContext(
                    stage_id=request_stage_id,
                    model=session.model,
                    body_fingerprint=session.body_fingerprint,
                    weight_digest=stage.weight_digest,
                    in_features=stage.in_features,
                    out_features=stage.out_features,
                    weight_bits=session.weight_bits,
                    activation_bits=session.activation_bits,
                    signed_output_bound=profile.signed_output_bound,
                    ring=profile.ring,
                    modulus=profile.modulus,
                    wire_bits=profile.wire_bits,
                ),
            )
            if value.stage_id != stage_id:
                raise ProtocolError("preparation stage route mismatch")
            value._validate(
                max_rows=model.manifest.context_length,
                max_tensor_elements=config.prepared_tensor_max_elements,
            )
            validate_preparation = getattr(engine, "validate_seeded_preparation", None)
            if validate_preparation is None:
                raise ProtocolError("preparation engine cannot validate requests")
            validate_preparation(value)
            session_registry.consume(
                value.session_id,
                value.stage_id,
                value.rows,
                value.attempt_id,
                owner,
            )
            correction = await engine.prepare_seeded_stage(value)
            correction_payload = correction.pack()
            metrics.correction_compute_ns += correction.server_ns
            metrics.correction_push_attempts += 1
            metrics.correction_push_bytes += len(correction_payload)
            push_started = time.perf_counter_ns()
            try:
                if correction_channel is None:
                    metrics.correction_channel_upload_bytes += len(correction_payload)
                    pushed = await push_http.post(
                        f"/v1/runtime/inventories/{value.session_id}/corrections/{value.attempt_id}",
                        headers={
                            "Authorization": f"Bearer {config.preparation_push_api_key}",
                            "Content-Type": BINARY_MEDIA_TYPE,
                        },
                        content=correction_payload,
                    )
                    pushed.raise_for_status()
                    pushed_content = pushed.content
                else:
                    metrics.correction_channel_upload_bytes += len(correction_payload)
                    pushed_content, wire_bytes = await correction_channel.push(
                        value.session_id,
                        value.attempt_id,
                        correction_payload,
                        wait_for_ack=True,
                    )
                    if wire_bytes != len(correction_payload):
                        raise ProtocolError("correction channel byte accounting mismatch")
            finally:
                correction_push_ns = time.perf_counter_ns() - push_started
                metrics.correction_push_ns += correction_push_ns
            metrics.correction_push_ack_bytes += len(pushed_content)
            metrics.correction_channel_download_bytes += len(pushed_content)
            if pushed_content:
                push_ack = PreparationAck.unpack(pushed_content)
                if (
                    push_ack.attempt_id != value.attempt_id
                    or push_ack.stage_id != value.stage_id
                    or push_ack.correction_bytes != len(correction_payload)
                ):
                    raise ProtocolError("inference correction acknowledgement mismatch")
            record_protocol_bytes(
                "preparation", "inference", len(correction_payload), value.stage_id
            )
            verification = {}
            if session.verification_component != "none":
                policy = freivalds_policies.get(session_id)
                if policy is None:
                    raise ProtocolError("Freivalds policy is unavailable for the session")
                material = await asyncio.to_thread(
                    engine.prepare_freivalds,
                    session,
                    value,
                    policy,
                )
                verification = {
                    "verification_component": session.verification_component,
                    "verification_checks": material.checks,
                    "verification_max_row_l1": material.max_row_l1,
                    "verification_material_id": material.material_id,
                    "verification_tag": material.authentication_tag(value.seed),
                    "verification_payload": material.payload(),
                }
            result = PreparationAck(
                value.attempt_id,
                value.stage_id,
                len(correction_payload),
                correction.server_ns,
                correction_push_ns,
                **verification,
            ).pack()
        except (
            CorrectionChannelError,
            ProtocolError,
            TransformerEngineError,
            ValueError,
            httpx.HTTPError,
        ) as exc:
            metrics.failures += 1
            session_registry.release(session_id, owner)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": str(exc),
                        "code": "preparation_failed",
                    }
                },
            ) from exc
        except BaseException:
            session_registry.release(session_id, owner)
            raise
        metrics.calls += 1
        metrics.rows += value.rows
        metrics.upload_bytes += len(raw)
        metrics.download_bytes += len(result)
        if audit_hook is not None:
            audit_hook(
                "prepare_stage",
                msgpack.packb(
                    {
                        "model": value.model,
                        "stage_id": value.stage_id,
                        "rows": value.rows,
                        "request_bytes": len(raw),
                        "response_bytes": len(result),
                        "correction_bytes": len(correction_payload),
                    },
                    use_bin_type=True,
                ),
            )
        return FastAPIResponse(result, media_type=BINARY_MEDIA_TYPE)

    @app.get("/metrics")
    async def service_metrics(
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authenticate(authorization)
        registry = session_registry.stats()
        return {
            "role": "trusted-preparation",
            "preparation": {
                "authorization_calls": metrics.authorization_calls,
                "authorization_upload_bytes": metrics.authorization_upload_bytes,
                "authorization_download_bytes": metrics.authorization_download_bytes,
                "authorization_push_bytes": metrics.authorization_push_bytes,
                "active_sessions": registry["authorized_sessions"],
                "session_attempts": registry["attempts"],
                "calls": metrics.calls,
                "rows": metrics.rows,
                "upload_bytes": metrics.upload_bytes,
                "download_bytes": metrics.download_bytes,
                "correction_push_attempts": metrics.correction_push_attempts,
                "correction_push_bytes": metrics.correction_push_bytes,
                "correction_push_ack_bytes": metrics.correction_push_ack_bytes,
                "correction_channel_upload_bytes": metrics.correction_channel_upload_bytes,
                "correction_channel_download_bytes": metrics.correction_channel_download_bytes,
                "correction_compute_ns": metrics.correction_compute_ns,
                "correction_push_ns": metrics.correction_push_ns,
                "failures": metrics.failures,
            },
        }

    from .telemetry import instrument_fastapi

    instrument_fastapi(app)
    return app
