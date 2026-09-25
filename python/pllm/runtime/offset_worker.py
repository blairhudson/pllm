"""Isolated research worker for a bounded, two-online-offset comparison.

This is not a public Experiment or provider role. Each worker verifies the
compiled semantic decoder and its local weight commitments before accepting
one-use additive input shares over an authenticated HTTP session.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import Response

from pllm.modeling import lower_model
from pllm.roles import two_online_reference_graph
from pllm.runtime.semantic_source import semantic_source_config
from pllm.runtime.stage_protocol import MaskedStageRequest, ProtocolError
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

_SESSION_SCHEMA = "pllm.offset_worker_session.v1"
_MAX_BODY_BYTES = 16 * 1024 * 1024 + 8_192
_MAX_SESSIONS = 8
_SESSION_IDLE_SECONDS = 90.0
_MAX_TOTAL_CALLS = 8_192
_MAX_TENSOR_ELEMENTS = 4_000_000


@dataclass(slots=True)
class _WorkerSession:
    id: str
    plan_digest: str
    max_input_tokens: int
    max_new_tokens: int
    last_active: float
    used_correlations: set[str] = field(default_factory=set)
    calls_by_stage: dict[str, int] = field(default_factory=dict)
    terminal: bool = False


def _reject(message: str, *, status: int = 409) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": {"message": message}})


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate session field")
        value[key] = item
    return value


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


async def _limited_body(request: Request, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise _reject("offset worker request exceeds the body limit", status=413)
        chunks.append(chunk)
    return b"".join(chunks)


def create_offset_worker_app(
    engine: MaskedTransformerEngine, *, model_id: str, role_id: str, api_key: str,
) -> FastAPI:
    """Bind one loaded public-weight worker to a fixed role and key."""
    if (
        type(engine) is not MaskedTransformerEngine
        or role_id not in {"worker_a", "worker_b"}
        or type(api_key) is not str or len(api_key) < 16
    ):
        raise ValueError("invalid bounded offset worker configuration")
    model = engine._model(model_id)
    metadata = model.manifest.metadata
    if (
        metadata.get("privacy_mode") != "public"
        or metadata.get("decoder_execution") != "semantic_schedule_v1"
        or any(type(metadata.get(key)) is not str for key in (
            "body_fingerprint", "seeded_stage_commitment", "runtime_config_digest",
        ))
    ):
        raise ValueError("offset workers require a compiled public decoder body")

    graph_digest = two_online_reference_graph().digest()
    sessions: dict[str, _WorkerSession] = {}
    lock = threading.Lock()
    app = FastAPI(title="PLLM bounded offset worker", docs_url=None, redoc_url=None,
                  openapi_url=None)

    def authenticate(authorization: str | None) -> None:
        if (
            not isinstance(authorization, str)
            or not authorization.startswith("Bearer ")
            or not secrets.compare_digest(authorization[7:], api_key)
        ):
            raise _reject("offset worker credential is invalid", status=401)

    def lookup(session_id: str) -> _WorkerSession:
        with lock:
            session = sessions.get(session_id)
            if session is None or session.terminal:
                raise _reject("offset worker session is not active")
            if time.monotonic() - session.last_active > _SESSION_IDLE_SECONDS:
                session.terminal = True
                raise _reject("offset worker session has expired")
            session.last_active = time.monotonic()
            return session

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "role": role_id}

    @app.get("/v1/offset-reference/metrics")
    async def process_metrics(
        authorization: str | None = Header(default=None),
    ) -> dict[str, str | int | None]:
        authenticate(authorization)
        return {
            "schema": "pllm.offset_worker_process_metrics.v1",
            "cpu_ns": time.process_time_ns(),
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    @app.post("/v1/offset-reference/sessions")
    async def open_session(
        request: Request, authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        authenticate(authorization)
        try:
            body = json.loads((await _limited_body(request, 4_096)).decode("utf-8"),
                              object_pairs_hook=_strict_object)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise _reject("offset worker session document is invalid", status=400) from exc
        expected = {
            "schema", "model", "role", "topology_digest", "decoder_plan",
            "max_input_tokens", "max_new_tokens", "body_fingerprint",
            "stage_commitment", "runtime_config_digest",
        }
        if type(body) is not dict or set(body) != expected:
            raise _reject("offset worker session schema differs")
        bound = body["max_input_tokens"]
        output_bound = body["max_new_tokens"]
        if (
            body["schema"] != _SESSION_SCHEMA
            or body["model"] != model_id
            or body["role"] != role_id
            or body["topology_digest"] != graph_digest
            or type(bound) is not int or not 1 <= bound <= 64
            or type(output_bound) is not int or not 1 <= output_bound <= 8
            or type(body["decoder_plan"]) is not str
            or len(body["decoder_plan"]) != 64
            or any(char not in "0123456789abcdef" for char in body["decoder_plan"])
            or body["body_fingerprint"] != metadata["body_fingerprint"]
            or body["stage_commitment"] != metadata["seeded_stage_commitment"]
            or body["runtime_config_digest"] != metadata["runtime_config_digest"]
            or len(model.stages) * (1 + output_bound) > _MAX_TOTAL_CALLS
        ):
            raise _reject("offset worker session commitments differ")
        try:
            plan = lower_model(
                semantic_source_config(model.config), batch=1,
                max_input_tokens=bound, max_new_tokens=output_bound,
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _reject("offset worker has no matching compiled plan") from exc
        if plan.digest != body["decoder_plan"]:
            raise _reject("offset worker decoder plan differs")

        with lock:
            now = time.monotonic()
            for key, old in list(sessions.items()):
                if old.terminal or now - old.last_active > _SESSION_IDLE_SECONDS:
                    del sessions[key]
            if len(sessions) >= _MAX_SESSIONS:
                raise _reject("offset worker session capacity exceeded", status=503)
            session_id = secrets.token_hex(16)
            sessions[session_id] = _WorkerSession(
                session_id, plan.digest, bound, output_bound, now,
            )
        return {
            "schema": _SESSION_SCHEMA,
            "id": session_id,
            "role": role_id,
            "topology_digest": graph_digest,
            "decoder_plan": plan.digest,
            "body_fingerprint": metadata["body_fingerprint"],
            "stage_commitment": metadata["seeded_stage_commitment"],
        }

    @app.post("/v1/offset-reference/sessions/{session_id}/stages/{stage_id}")
    async def execute_stage(
        session_id: str, stage_id: str, request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        authenticate(authorization)
        session = lookup(session_id)
        try:
            payload = await _limited_body(request, _MAX_BODY_BYTES)
            entry = model.stages.get(stage_id)
            if entry is None or entry.seeded_profile is None:
                raise ValueError("offset stage is not in the admitted decoder")
            parsed = MaskedStageRequest.unpack(
                payload, max_rows=session.max_input_tokens,
                max_tensor_elements=_MAX_TENSOR_ELEMENTS,
            )
            if (
                parsed.session_id != session_id
                or parsed.model != model_id
                or parsed.stage_id != stage_id
                or parsed.body_fingerprint != metadata["body_fingerprint"]
                or parsed.weight_digest != entry.weight_digest
                or parsed.weight_bits != entry.spec.weight_bits
                or parsed.activation_bits != entry.spec.activation_bits
                or parsed.out_features != entry.spec.out_features
                or parsed.masked_input.ndim != 2
                or parsed.masked_input.shape[1] != entry.spec.in_features
                or parsed.masked_input.shape[0] * max(
                    entry.spec.in_features, entry.spec.out_features,
                ) > _MAX_TENSOR_ELEMENTS
                or parsed.signed_output_bound != entry.seeded_profile.signed_output_bound
                or parsed.ring != entry.seeded_profile.ring
                or parsed.modulus != entry.seeded_profile.modulus
                or parsed.wire_bits != entry.seeded_profile.wire_bits
                or len(parsed.correlation_id) != 32
                or any(char not in "0123456789abcdef" for char in parsed.correlation_id)
            ):
                raise ValueError("offset stage request differs from its session contract")
            with lock:
                if session.terminal:
                    raise ValueError("offset worker session is terminal")
                if (
                    parsed.correlation_id in session.used_correlations
                    or session.calls_by_stage.get(stage_id, 0) >= 1 + session.max_new_tokens
                ):
                    raise ValueError("offset stage ticket is replayed or exhausted")
                session.used_correlations.add(parsed.correlation_id)
                session.calls_by_stage[stage_id] = session.calls_by_stage.get(stage_id, 0) + 1
            results = await engine.execute_stage(model_id, entry.spec, [payload])
            if len(results) != 1:
                raise ValueError("offset worker returned the wrong stage result count")
        except BaseException as exc:
            with lock:
                session.terminal = True
            if isinstance(exc, HTTPException):
                raise
            if isinstance(exc, (ProtocolError, TransformerEngineError, TypeError, ValueError)):
                raise _reject("offset worker stage was rejected", status=400) from exc
            raise
        return Response(results[0], media_type="application/msgpack")

    @app.post("/v1/offset-reference/sessions/{session_id}/complete")
    async def complete_session(
        session_id: str, authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        authenticate(authorization)
        session = lookup(session_id)
        with lock:
            session.terminal = True
        return {"id": session_id, "status": "completed"}

    @app.post("/v1/offset-reference/sessions/{session_id}/cancel")
    async def cancel_session(
        session_id: str, authorization: str | None = Header(default=None),
    ) -> dict[str, str]:
        authenticate(authorization)
        session = lookup(session_id)
        with lock:
            session.terminal = True
        return {"id": session_id, "status": "cancelled"}

    return app


__all__ = ["create_offset_worker_app"]


def main() -> None:
    """Run one loopback-only research worker in a separate process."""
    import argparse
    import asyncio

    import uvicorn

    from pllm.runtime.loaders import load_hf_directory

    parser = argparse.ArgumentParser(description="Bounded internal offset reference worker")
    parser.add_argument("checkpoint")
    parser.add_argument("--model-id", default="offset-model")
    parser.add_argument("--role", required=True, choices=("worker_a", "worker_b"))
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--weight-bits", required=True, type=int, choices=(4, 8))
    parser.add_argument("--activation-bits", required=True, type=int, choices=(4, 8))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("worker port is outside the valid range")
    api_key = os.environ.get("PLLM_OFFSET_WORKER_API_KEY", "")
    if len(api_key) < 16:
        parser.error("PLLM_OFFSET_WORKER_API_KEY is required")
    worker = MaskedTransformerEngine(
        threads=1, weight_bits=args.weight_bits, activation_bits=args.activation_bits,
    )
    asyncio.run(worker.load(load_hf_directory(args.checkpoint, model_id=args.model_id)))
    app = create_offset_worker_app(worker, model_id=args.model_id,
                                   role_id=args.role, api_key=api_key)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
