"""Authenticated bounded capacity controller around installed role adapters.

Operator labels declare ownership, not physical separation or attestation.
Only capacity credentials cross admission; no provider static key is exported.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import secrets
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from pllm.deployment.network import (
    LivePartyOffer, NetworkError, NetworkSpec, PartySpec, canonical, credential,
    digest_value, exact, integer, strict_load,
)
from pllm.plan import PlanningResult
from pllm.runtime.servers import load_role_adapter


def now_ms():
    return time.time_ns() // 1_000_000


async def bounded_json(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 1 << 20:
            raise NetworkError("document exceeds 1 MiB")
        chunks.append(chunk)
    return strict_load(b"".join(chunks))


def authenticate(request, key):
    headers = request.headers.getlist("authorization")
    if len(headers) != 1 or not secrets.compare_digest(headers[0], f"Bearer {key}"):
        raise HTTPException(401, detail="credential rejected")


@dataclass(repr=False)
class _Reservation:
    attempt: str
    role: str
    binding_digest: str
    expires_at_ms: int
    credential: str
    memory: int
    weights: int
    budget: object
    armed: bool = False
    revoked: bool = False
    opened: bool = False
    sessions: set[str] = field(default_factory=set)
    calls: set = field(default_factory=set)
    adapter: object | None = None
    lifespan: object | None = None
    push_credential: str | None = None
    inventory_opened: bool = False


class PartyController:
    """One restart epoch; authoritative clock and aggregate role capacity."""

    def __init__(self, spec: PartySpec, network: NetworkSpec, *, credentials=None, clock=now_ms):
        if network.backend != "http":
            raise NetworkError("party requires authenticated http network")
        roots = {item.party_id: item for item in network.parties}
        if spec.party_id not in roots or not set(spec.peer_party_ids) <= set(roots):
            raise NetworkError("party or peer absent from configured trust roots")
        self.spec, self.network = spec, network
        self.trust = roots[spec.party_id]
        self._key = credential(self.trust.credential_env, credentials)
        self.clock = clock
        self.epoch = secrets.token_hex(32)
        self.accepting = True
        self.left = False
        self._reservations = {}
        self._used_attempts = set()
        self._adapters = {}
        self._static_keys = {}
        self._engines = {}
        self._descriptor_digest = None
        self._model_plan = None
        self._lock = asyncio.Lock()

    async def start(self):
        from pathlib import Path
        from pllm.model_loader import resolve_model
        from pllm.modeling import lower_model

        if self._descriptor_digest is not None:
            raise NetworkError("party is already running")
        self.epoch = secrets.token_hex(32)
        self._used_attempts.clear()
        self.accepting = True
        self.left = False
        model = self.spec.experiment.pipeline.model
        if model.kind not in {"huggingface", "safetensors"} or not Path(model.source).expanduser().is_dir():
            raise NetworkError("party artifact policy requires an already installed checkpoint directory")
        resolved = resolve_model(self.spec.experiment.pipeline.model)
        if resolved.source_lock_digest != self.spec.source_lock_digest or resolved.path is None:
            raise NetworkError("actual checkpoint source lock mismatch")
        with (Path(resolved.path) / "config.json").open("rb") as stream:
            config = strict_load(stream.read((1 << 20) + 1))
        budget = self.spec.experiment.budget
        self._model_plan = lower_model(config, batch=1, max_input_tokens=budget.max_input_tokens,
                                       max_new_tokens=budget.max_new_tokens)
        self._model_plan.runtime_schedule(self.spec.experiment.pipeline)
        descriptor = {}
        try:
            for role in self.spec.role_ids:
                key = secrets.token_urlsafe(32)
                app, engine = await load_role_adapter(self.spec.experiment, role, key)
                self._static_keys[role] = key
                self._adapters[role] = app
                self._engines[role] = engine
                metadata = engine._model(self.spec.experiment.resolve().model).manifest.metadata
                if len(engine._model(self.spec.experiment.resolve().model).stages) * (
                    1 + budget.max_new_tokens
                ) > 8192:
                    raise NetworkError("UNSUPPORTED_NETWORK_GRAPH: workload exceeds installed stage-call bound")
                descriptor[role] = {name: metadata[name] for name in (
                    "body_fingerprint", "seeded_stage_commitment", "runtime_config_digest")}
            self._descriptor_digest = hashlib.sha256(
                b"pllm.installed_party_descriptor.v1\0" + canonical({
                    "pipeline": self.spec.experiment.pipeline.to_spec(),
                    "source_lock_digest": resolved.source_lock_digest,
                    "model_plan_digest": self._model_plan.digest,
                    "role_descriptors": descriptor,
                })
            ).hexdigest()
        except BaseException:
            self._adapters.clear()
            self._static_keys.clear()
            raise

    def offer(self):
        if self._descriptor_digest is None or self.left:
            raise NetworkError("party unavailable")
        stamp = self.clock()
        return LivePartyOffer(
            self.spec.party_id, self.trust.operator_id, self.epoch,
            stamp + self.spec.offer_seconds * 1000, tuple(sorted({
                "trusted_preparation" if role == "preparation" else
                "masked_linear_provider" if role == "inference" else "public_linear_provider"
                for role in self.spec.role_ids
            })),
            self.spec.max_memory_bytes, self.spec.max_weight_bytes, self.spec.max_sessions,
            len(self._reservations), (self._model_plan.to_dict()["config_digest"],),
            (self.spec.experiment.pipeline.digest(),), authenticated=True,
            origin=self.trust.origin, descriptor_digest=self._descriptor_digest,
            source_lock_digest=self.spec.source_lock_digest, role_ids=self.spec.role_ids,
            observed_at_ms=stamp, accepting=self.accepting,
        )

    async def _burn(self, row):
        # Revoke before awaiting calls. No new call may enter during cancellation.
        row.revoked = True
        current = asyncio.current_task()
        tasks = [task for task in row.calls if task is not current]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        adapter = row.adapter or self._adapters[row.role]
        key = row.credential if row.adapter is not None else self._static_keys[row.role]
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=adapter),
                                      base_url="http://adapter") as client:
            if row.role == "inference":
                sessions = adapter.state.sessions
                for session, state in sorted(sessions.items(), key=lambda item: item[1].execution == "seeded-inventory"):
                    kind = "inventories" if state.execution == "seeded-inventory" else "sessions"
                    await client.post(f"/v1/runtime/{kind}/{session}/cancel",
                                      headers={"authorization": f"Bearer {key}"})
            elif row.role in {"worker_a", "worker_b"}:
                for session in row.sessions:
                    await client.post(f"/v1/offset-reference/sessions/{session}/cancel",
                                      headers={"authorization": f"Bearer {key}"})
        if row.lifespan is not None:
            await row.lifespan.__aexit__(None, None, None)
            row.lifespan = None
        row.adapter = None
        row.push_credential = None
        row.sessions.clear()
        row.credential = ""

    async def purge(self):
        async with self._lock:
            for key, row in list(self._reservations.items()):
                if row.expires_at_ms <= self.clock():
                    await self._burn(row)
                    del self._reservations[key]

    async def close(self):
        async with self._lock:
            self.accepting = False
            for row in list(self._reservations.values()):
                await self._burn(row)
            self._reservations.clear()
            for role, adapter in self._adapters.items():
                if role in {"inference", "preparation"}:
                    async with adapter.router.lifespan_context(adapter):
                        pass
            for engine in self._engines.values():
                for model_id in list(engine.models):
                    await engine.unload(model_id)
            self._engines.clear()
            self._adapters.clear()
            self._static_keys.clear()
            self._descriptor_digest = None

    def _validate_binding(self, body):
        fields = {"schema", "attempt_id", "ttl_seconds", "role_id", "instance_epoch",
                  "descriptor_digest", "binding_digest", "result"}
        if body.get("role_id") in {"inference", "preparation"}:
            fields.add("prepared_peer")
        exact(body, fields, "pllm.party_reserve.v1")
        integer(body["ttl_seconds"], "ttl_seconds", minimum=5, maximum=self.spec.max_lease_seconds)
        digest_value(body["attempt_id"], "attempt", nonzero=True)
        digest_value(body["binding_digest"], "binding", nonzero=True)
        role = body["role_id"]
        if role not in self.spec.role_ids:
            raise NetworkError("UNSUPPORTED_NETWORK_GRAPH")
        if body["instance_epoch"] != self.epoch:
            raise NetworkError("STALE_INSTANCE_EPOCH")
        if body["descriptor_digest"] != self._descriptor_digest:
            raise NetworkError("CAPABILITY_MISMATCH: installed descriptor differs")
        result = PlanningResult.from_spec(body["result"])
        if not result.exhaustive and not result.request.policy.allow_incomplete_execution:
            raise NetworkError("SEARCH_LIMIT: incomplete execution needs explicit policy permission")
        selected = result.experiment
        if selected is None or selected.deployment.kind != "network":
            raise NetworkError("explicit v3 network intent required")
        intent = selected.deployment
        if (intent.network_id != self.network.network_id or intent.network_spec_digest != self.network.digest
            or (intent.snapshot_digest is not None and intent.snapshot_digest != result.snapshot.digest)):
            raise NetworkError("network intent mismatch")
        if selected.pipeline != self.spec.experiment.pipeline or selected.budget != self.spec.experiment.budget:
            raise NetworkError("CAPABILITY_MISMATCH: exact installed pipeline/workload required")
        if result.request.source_lock_digest != self.spec.source_lock_digest:
            raise NetworkError("source lock mismatch")
        if result.request.model_plan.digest != self._model_plan.digest:
            raise NetworkError("actual checkpoint ModelPlan digest mismatch")
        native = result.validate(evaluated_at_ms=self.clock())
        assignment = {item["role_id"]: item["party_id"] for item in native["roles"]}
        if role in {"inference", "preparation"}:
            peer = body["prepared_peer"]
            exact(peer, {"inference_party_id", "preparation_party_id", "push_credential"})
            if set(assignment) != {"client", "inference", "preparation"} or any(
                peer[name + "_party_id"] != assignment[name] for name in ("inference", "preparation")
            ):
                raise NetworkError("prepared peer assignment mismatch")
            token = peer["push_credential"]
            if (type(token) is not str or not 32 <= len(token) <= 128
                or not all(char.isascii() and (char.isalnum() or char in "-_") for char in token)
                or secrets.compare_digest(token, self._key)):
                raise NetworkError("invalid private push credential")
        if assignment.get(role) != self.spec.party_id:
            raise NetworkError("forbidden role reassignment")
        remote = {party for name, party in assignment.items() if name != "client"}
        if not remote <= {self.spec.party_id, *self.spec.peer_party_ids}:
            raise NetworkError("peer party not allowed")
        roots = {item.party_id: item for item in self.network.parties}
        offered = {item.party_id: item for item in result.snapshot.offers}
        for name, party in assignment.items():
            if name == "client":
                seed = next((item for item in self.network.snapshot.offers
                             if item.party_id == party and party not in roots), None)
                if seed is None or offered[party] != seed:
                    raise NetworkError("trusted client authority differs from configured root")
                continue
            trust, public = roots[party], offered[party]
            if (not isinstance(public, LivePartyOffer) or public.operator_id != trust.operator_id
                or public.origin != trust.origin or name not in public.role_ids
                or public.source_lock_digest != self.spec.source_lock_digest):
                raise NetworkError("peer installed offer differs from approved trust root")
        offer = next(item for item in result.snapshot.offers if item.party_id == self.spec.party_id)
        if not isinstance(offer, LivePartyOffer) or (
            offer.instance_epoch != self.epoch or offer.descriptor_digest != self._descriptor_digest
            or offer.origin != self.trust.origin or offer.operator_id != self.trust.operator_id
        ):
            raise NetworkError("CAPABILITY_MISMATCH: authenticated installed offer required")
        expected = hashlib.sha256(b"pllm.execution_attempt.v1\0" + canonical({
            "attempt_id": body["attempt_id"], "result_digest": result.digest,
            "network_digest": self.network.digest,
        })).hexdigest()
        if body["binding_digest"] != expected:
            raise NetworkError("execution binding mismatch")
        return result, native

    async def reserve(self, body):
        result, native = self._validate_binding(body)
        async with self._lock:
            if not self.accepting or self.left:
                raise NetworkError("PARTY_DRAINING")
            attempt, role = body["attempt_id"], body["role_id"]
            key = (attempt, role)
            if key in self._used_attempts:
                raise NetworkError("attempt/role already consumed; retry requires fresh attempt")
            if len(self._used_attempts) >= 8192:
                raise NetworkError("epoch attempt history full; restart required")
            memory, weights = native["required_memory_bytes"][role], native["required_weight_bytes"][role]
            rows = tuple(self._reservations.values())
            if (len(rows) >= self.spec.max_sessions
                or sum(row.memory for row in rows) + memory > self.spec.max_memory_bytes
                or sum(row.weights for row in rows) + weights > self.spec.max_weight_bytes):
                raise NetworkError("RESERVATION_CONFLICT: capacity exceeded")
            self._used_attempts.add(key)
            row = _Reservation(attempt, role, body["binding_digest"],
                               self.clock() + body["ttl_seconds"] * 1000,
                                secrets.token_urlsafe(32), memory, weights, result.experiment.budget)
            if role in {"inference", "preparation"}:
                roots = {item.party_id: item for item in self.network.parties}
                peer = body["prepared_peer"]
                row.push_credential = peer["push_credential"]
                row.adapter, _ = await load_role_adapter(
                    self.spec.experiment, role, row.credential,
                    preloaded_engine=self._engines[role], push_key=row.push_credential,
                    inference_url=roots[peer["inference_party_id"]].origin + "/roles/inference",
                )
                row.lifespan = row.adapter.router.lifespan_context(row.adapter)
                await row.lifespan.__aenter__()
            self._reservations[key] = row
            return {"schema": "pllm.party_capacity.v1", "attempt_id": attempt, "role_id": role,
                    "expires_at_ms": row.expires_at_ms, "capacity_token": row.credential,
                    "binding_digest": row.binding_digest, "instance_epoch": self.epoch}

    async def arm_or_release(self, body, *, release):
        exact(body, {"attempt_id", "role_id", "binding_digest", "instance_epoch"})
        for name in ("attempt_id", "binding_digest", "instance_epoch"):
            digest_value(body[name], name, nonzero=True)
        if type(body["role_id"]) is not str or body["role_id"] not in self.spec.role_ids:
            raise NetworkError("role not admitted")
        async with self._lock:
            if body["instance_epoch"] != self.epoch:
                raise NetworkError("STALE_INSTANCE_EPOCH")
            key = (body["attempt_id"], body["role_id"])
            row = self._reservations.get(key)
            if row is None:
                if release:
                    return {"status": "released"}
                raise NetworkError("lease absent or expired")
            if row.binding_digest != body["binding_digest"]:
                raise NetworkError("execution binding mismatch")
            if release:
                await self._burn(row)
                del self._reservations[key]
                return {"status": "released"}
            if row.revoked or row.expires_at_ms <= self.clock() or self.left:
                raise NetworkError("lease expired or revoked")
            row.armed = True
            return {"status": "armed", "binding_digest": row.binding_digest,
                    "expires_at_ms": row.expires_at_ms}

    async def role_call(self, role, scope, receive, send):
        headers = [value for name, value in scope["headers"] if name.lower() == b"authorization"]
        from pllm.runtime.correction_channel import CORRECTION_CHANNEL_PATH

        path = scope["path"]
        push = role == "inference" and (
            path == CORRECTION_CHANNEL_PATH or (
                path.startswith("/v1/runtime/inventories/") and
                (path.endswith("/authorize") or "/corrections/" in path)
            )
        )
        row = next((row for row in self._reservations.values()
                     if row.role == role and len(headers) == 1 and
                     secrets.compare_digest(headers[0], f"Bearer {row.push_credential if push else row.credential}".encode())), None)
        if row is None or not row.armed or row.revoked or row.expires_at_ms <= self.clock():
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4401})
            else:
                await JSONResponse({"error": "armed lease credential required"}, status_code=401)(scope, receive, send)
            return
        if scope["type"] == "http" and scope["method"] == "GET" and path == "/v1/party/metrics" and not push:
            from .offset_worker import _peak_rss_bytes
            await JSONResponse({"cpu_ns": time.process_time_ns(), "peak_rss_bytes": _peak_rss_bytes()})(scope, receive, send)
            return
        if role in {"inference", "preparation"}:
            await self._prepared_call(row, scope, receive, send, push=push)
            return
        task = asyncio.current_task()
        row.calls.add(task)
        path = scope["path"]
        prefix = "/v1/offset-reference/sessions"
        opening = path == prefix and scope["method"] == "POST"
        sent = False
        try:
            if path.startswith(prefix + "/"):
                session = path[len(prefix) + 1:].split("/")[0]
                if session not in row.sessions:
                    await JSONResponse({"error": "session belongs to another attempt"}, status_code=403)(scope, receive, send)
                    return
            if opening:
                if row.opened:
                    raise NetworkError("one response session per lease")
                row.opened = True  # Failed/ambiguous opening burns the attempt too.
                chunks, size = [], 0
                while True:
                    message = await receive()
                    if message["type"] != "http.request":
                        raise NetworkError("session opening interrupted")
                    size += len(message.get("body", b""))
                    if size > 4096:
                        raise NetworkError("session document exceeds bound")
                    chunks.append(message.get("body", b""))
                    if not message.get("more_body"):
                        break
                body = strict_load(b"".join(chunks))
                if (type(body.get("max_input_tokens")) is not int or
                    not 1 <= body["max_input_tokens"] <= row.budget.max_input_tokens or
                    type(body.get("max_new_tokens")) is not int or
                    not 1 <= body["max_new_tokens"] <= row.budget.max_new_tokens):
                    raise NetworkError("session workload exceeds admitted lease")
                async def replay():
                    return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
                receive = replay
            translated = dict(scope)
            translated["headers"] = [(name, value) for name, value in scope["headers"]
                                     if name.lower() != b"authorization"] + [
                (b"authorization", f"Bearer {self._static_keys[role]}".encode())]
            captured, status = [], [0]
            async def forward(message):
                nonlocal sent
                if message["type"] == "http.response.start":
                    status[0] = message["status"]
                if opening and message["type"] == "http.response.body":
                    captured.append(message.get("body", b""))
                    if not message.get("more_body") and status[0] == 200:
                        value = strict_load(b"".join(captured))
                        row.sessions.add(value["id"])
                if row.revoked or row.expires_at_ms <= self.clock():
                    raise NetworkError("lease expired during provider call")
                await send(message)
                sent = True
            await self._adapters[role](translated, receive, forward)
        except NetworkError:
            if sent:
                raise
            await JSONResponse({"error": "lease protocol rejected"}, status_code=409)(scope, receive, send)
        finally:
            row.calls.discard(task)

    async def _prepared_call(self, row, scope, receive, send, *, push):
        """One attempt owns its adapter, rendezvous, credentials and inventory.

        Shared immutable weights do not share mutable provider sessions. API model
        mutation and application-facing endpoints are absent from the role allowlist.
        """
        path, method = scope["path"], scope.get("method", "WEBSOCKET")
        public = method == "GET" and (path == "/v1/models" or path == "/metrics"
                                     or path.startswith("/v1/runtime/models/"))
        runtime = path.startswith("/v1/runtime/sessions") or path.startswith("/v1/runtime/inventories")
        websocket = scope["type"] == "websocket" and path.startswith("/v1/runtime/ws/")
        preparation = row.role == "preparation" and path.startswith("/v1/preparation/inventories/")
        if not (public or push or preparation or (row.role == "inference" and (runtime or websocket))):
            await JSONResponse({"error": "role endpoint not admitted"}, status_code=403)(scope, receive, send)
            return
        task = asyncio.current_task()
        row.calls.add(task)
        try:
            opening = method == "POST" and path in {"/v1/runtime/sessions", "/v1/runtime/inventories"}
            if opening:
                if (path.endswith("sessions") and row.opened) or (path.endswith("inventories") and row.inventory_opened):
                    raise NetworkError("one inventory and response session per lease")
                raw = bytearray()
                while True:
                    message = await receive()
                    if message["type"] != "http.request":
                        raise NetworkError("opening interrupted")
                    raw.extend(message.get("body", b""))
                    if len(raw) > 65536:
                        raise NetworkError("opening document exceeds bound")
                    if not message.get("more_body"):
                        break
                body = strict_load(bytes(raw))
                if body.get("model") != self.spec.experiment.resolve().model:
                    raise NetworkError("opening model differs from installed plan")
                if path.endswith("inventories"):
                    cap = max(64, row.budget.max_input_tokens + row.budget.max_new_tokens - 1,
                              self.spec.experiment.resolve().prepared_inventory_rows or 64)
                    if type(body.get("rows")) is not int or not 1 <= body["rows"] <= cap:
                        raise NetworkError("inventory rows exceed admitted workload")
                    row.inventory_opened = True
                else:
                    contract = body.get("decoder_plan")
                    if (type(contract) is not dict or type(contract.get("max_input_tokens")) is not int
                        or not 1 <= contract["max_input_tokens"] <= row.budget.max_input_tokens
                        or type(body.get("max_output_tokens")) is not int
                        or not 1 <= body["max_output_tokens"] <= row.budget.max_new_tokens):
                        raise NetworkError("session workload exceeds admitted lease")
                    row.opened = True
                async def replay():
                    return {"type": "http.request", "body": bytes(raw), "more_body": False}
                receive = replay
            async def forward(message):
                if row.revoked or row.expires_at_ms <= self.clock():
                    raise NetworkError("lease expired during provider call")
                await send(message)
            await row.adapter(scope, receive, forward)
        except NetworkError:
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4409})
            else:
                await JSONResponse({"error": "lease protocol rejected"}, status_code=409)(scope, receive, send)
        finally:
            row.calls.discard(task)


def create_party_app(spec: PartySpec, network: NetworkSpec, *, credentials=None, clock=now_ms):
    controller = PartyController(spec, network, credentials=credentials, clock=clock)

    @contextlib.asynccontextmanager
    async def lifespan(app):
        await controller.start()
        async def janitor():
            while True:
                await asyncio.sleep(0.25)
                await controller.purge()
        task = asyncio.create_task(janitor())
        async def membership():
            while True:
                if network.directory_origin is not None:
                    verb = "withdraw" if not controller.accepting else "renew"
                    try:
                        from pllm.deployment.network import control_request

                        await asyncio.to_thread(control_request, network.directory_origin,
                                                controller.trust.credential_env,
                                                f"/v1/directory/{verb}", credentials=credentials,
                                                body={"party_id": spec.party_id})
                    except NetworkError:
                        pass  # Membership expires; directory never promises capacity.
                await asyncio.sleep(spec.offer_seconds / 3)
        renewal = asyncio.create_task(membership())
        try:
            yield
        finally:
            task.cancel()
            renewal.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            with contextlib.suppress(asyncio.CancelledError):
                await renewal
            await controller.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.controller = controller

    @app.exception_handler(NetworkError)
    async def rejected(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=409)

    @app.get("/healthz")
    async def health():
        return {"role": "party", "ready": controller._descriptor_digest is not None}

    @app.get("/v1/party/offer")
    async def offer(request: Request):
        authenticate(request, controller._key)
        await controller.purge()
        return controller.offer().to_spec()

    @app.post("/v1/party/{verb}")
    async def command(verb: str, request: Request):
        authenticate(request, controller._key)
        body = await bounded_json(request)
        await controller.purge()
        if verb == "reserve":
            return await controller.reserve(body)
        if verb in {"arm", "release"}:
            return await controller.arm_or_release(body, release=verb == "release")
        if verb in {"drain", "leave"}:
            exact(body, set())
            controller.accepting = False
            if verb == "leave":
                # Leave completes only after admitted work drains; no material moves.
                if controller._reservations:
                    return {"status": "draining"}
                controller.left = True
            return {"status": "left" if controller.left else "draining"}
        raise HTTPException(404)

    for role in spec.role_ids:
        async def adapter(scope, receive, send, role_id=role):
            # Starlette mount retains full path in modern versions. Strip mount once.
            inner = dict(scope)
            inner["path"] = scope["path"].removeprefix(f"/roles/{role_id}")
            inner["raw_path"] = inner["path"].encode()
            inner["root_path"] = ""
            await controller.role_call(role_id, inner, receive, send)
        app.mount(f"/roles/{role}", adapter)
    return app


def main():
    """Internal local worker entry point uses the same installed role factory."""
    import argparse
    import os
    import uvicorn
    from pllm.configuration import Experiment

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--role", choices=("worker_a", "worker_b"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--weight-bits", type=int, required=True)
    parser.add_argument("--activation-bits", type=int, required=True)
    args = parser.parse_args()
    experiment = Experiment.from_spec(strict_load(os.environ["PLLM_OFFSET_EXPERIMENT_JSON"]))
    if experiment.resolve().model != args.model_id:
        parser.error("worker model identity differs from Experiment")
    app, _engine = asyncio.run(load_role_adapter(experiment, args.role,
                                               os.environ["PLLM_OFFSET_WORKER_API_KEY"]))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
