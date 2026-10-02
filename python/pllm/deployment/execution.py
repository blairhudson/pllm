"""Selected local/network execution through shared role adapters and session lifecycle."""

from __future__ import annotations

import threading
import time
import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pllm.deployment.network import (
    LivePartyOffer, NetworkError, NetworkSpec, canonical, control_request,
    digest_value, discover, exact, integer, strict_load,
)
from pllm.plan import PlanningResult

_LEASE_KEY = object()


@dataclass(frozen=True, slots=True)
class ExecutionBinding:
    """Public commitment to one local admission; contains no credentials."""

    result_digest: str
    network_digest: str
    placement_bytes: bytes

    def __post_init__(self):
        import hashlib

        digest_value(self.result_digest, "result")
        digest_value(self.network_digest, "network")
        if type(self.placement_bytes) is not bytes:
            raise TypeError("placement_bytes must be immutable native JSON bytes")
        placement = strict_load(self.placement_bytes)
        if canonical(placement) != self.placement_bytes:
            raise NetworkError("binding placement must be canonical JSON")
        commitment = placement.pop("placement_digest", None)
        expected = hashlib.sha256(
            b"pllm.network_placement.validated.v1\0" + canonical(placement)
        ).hexdigest()
        if commitment != expected:
            raise NetworkError("binding placement commitment mismatch")

    def to_spec(self) -> dict:
        return {
            "schema": "pllm.execution_binding.local.v1",
            "result_digest": self.result_digest,
            "network_digest": self.network_digest,
            "placement": strict_load(self.placement_bytes),
        }


@dataclass(frozen=True, slots=True)
class LiveExecutionBinding(ExecutionBinding):
    attempt_id: str
    attempt_digest: str
    source_lock_digest: str
    expires_at_ms: int

    def __post_init__(self):
        ExecutionBinding.__post_init__(self)
        for name in ("attempt_id", "attempt_digest", "source_lock_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        integer(self.expires_at_ms, "expires_at_ms", minimum=1)
        expected = hashlib.sha256(b"pllm.execution_attempt.v1\0" + canonical({
            "attempt_id": self.attempt_id, "result_digest": self.result_digest,
            "network_digest": self.network_digest,
        })).hexdigest()
        if expected != self.attempt_digest:
            raise NetworkError("execution attempt binding mismatch")

    def to_spec(self):
        return {**ExecutionBinding.to_spec(self), "schema": "pllm.execution_binding.http.v1",
                "attempt_id": self.attempt_id, "attempt_digest": self.attempt_digest,
                "source_lock_digest": self.source_lock_digest, "expires_at_ms": self.expires_at_ms}


class _LiveTopology:
    """Reserved endpoint adapter implementing the same client/close launcher interface."""

    def __init__(self, result, network, credentials, clock, ttl):
        self.result, self.network, self.clock = result, network, clock
        self._credentials = credentials
        self._rows = []
        self._connections = {}
        self.attempt = secrets.token_hex(32)
        self.attempt_digest = hashlib.sha256(b"pllm.execution_attempt.v1\0" + canonical({
            "attempt_id": self.attempt, "result_digest": result.digest,
            "network_digest": network.digest,
        })).hexdigest()
        self.expires_at_ms = clock() + ttl * 1000

    def start(self, ttl):
        roots = {item.party_id: item for item in self.network.parties}
        snapshot = discover(self.network, credentials=self._credentials, clock=self.clock)
        native = self.result.validate(snapshot=snapshot, evaluated_at_ms=self.clock())
        selected = self.result.experiment
        intent = selected.deployment
        if (intent.kind != "network" or intent.network_id != self.network.network_id
            or intent.network_spec_digest != self.network.digest
            or (intent.snapshot_digest is not None and intent.snapshot_digest != self.result.snapshot.digest)):
            raise NetworkError("explicit v3 network intent mismatch")
        if selected.budget.requests != 1:
            raise NetworkError("network lease supports one bounded response attempt")
        roles = native["roles"]
        role_ids = {item["role_id"] for item in roles}
        if role_ids not in ({"client", "worker_a", "worker_b"}, {"client", "inference", "preparation"}):
            raise NetworkError("UNSUPPORTED_NETWORK_GRAPH: live role adapter unavailable")
        peer = None
        if "preparation" in role_ids:
            assignments = {item["role_id"]: item["party_id"] for item in roles}
            peer = {"inference_party_id": assignments["inference"],
                    "preparation_party_id": assignments["preparation"],
                    "push_credential": secrets.token_urlsafe(32)}
        offers = {item.party_id: item for item in snapshot.offers}
        original = {item.party_id: item for item in self.result.snapshot.offers}
        pending = []
        expiries = {}
        for item in roles:
            role, party = item["role_id"], item["party_id"]
            if role == "client":
                if party in roots:
                    raise NetworkError("trusted client cannot be a remote hosted role")
                continue
            offer, previous = offers[party], original[party]
            if (not isinstance(offer, LivePartyOffer) or not isinstance(previous, LivePartyOffer)
                or offer.descriptor_digest != previous.descriptor_digest
                or offer.source_lock_digest != self.result.request.source_lock_digest
                or not offer.accepting or role not in offer.role_ids):
                raise NetworkError("CAPABILITY_MISMATCH: live installed role/source required")
            pending.append((party, role, roots[party], previous))
        try:
            for _party, role, trust, offer in sorted(pending):
                control = {"attempt_id": self.attempt, "role_id": role,
                           "binding_digest": self.attempt_digest, "instance_epoch": offer.instance_epoch}
                # Record before request: a lost response may still have reserved capacity.
                self._rows.append((trust, control))
                value = control_request(trust.origin, trust.credential_env, "/v1/party/reserve",
                                        credentials=self._credentials, body={
                                            **control, "schema": "pllm.party_reserve.v1",
                                            "ttl_seconds": ttl, "descriptor_digest": offer.descriptor_digest,
                                             "result": self.result.to_spec(),
                                             **({"prepared_peer": peer} if peer is not None else {}),
                                        })
                exact(value, {"schema", "attempt_id", "role_id", "expires_at_ms", "capacity_token",
                              "binding_digest", "instance_epoch"}, "pllm.party_capacity.v1")
                if any(value[name] != control[name] for name in control):
                    raise NetworkError("capacity acknowledgement binding mismatch")
                integer(value["expires_at_ms"], "expires_at_ms", minimum=self.clock() + 1)
                if value["expires_at_ms"] > self.clock() + ttl * 1000 + 1000:
                    raise NetworkError("unbounded capacity expiry")
                token = value["capacity_token"]
                if type(token) is not str or not 32 <= len(token) <= 128:
                    raise NetworkError("invalid private capacity credential")
                self.expires_at_ms = min(self.expires_at_ms, value["expires_at_ms"])
                expiries[(trust.party_id, role)] = value["expires_at_ms"]
                self._connections[role] = (f"{trust.origin}/roles/{role}", token)
            for trust, control in self._rows:
                armed = control_request(trust.origin, trust.credential_env, "/v1/party/arm",
                                        credentials=self._credentials, body=control)
                exact(armed, {"status", "binding_digest", "expires_at_ms"})
                if armed["status"] != "armed" or armed["binding_digest"] != self.attempt_digest:
                    raise NetworkError("ambiguous capacity commit; abort attempt")
                if (armed["expires_at_ms"] != expiries[(trust.party_id, control["role_id"])]
                    or armed["expires_at_ms"] <= self.clock()):
                    raise NetworkError("capacity commit expiry differs; abort attempt")
            return native
        except BaseException:
            self.close()
            raise

    def client(self):
        from pllm.runtime.servers import role_client

        if self.clock() >= self.expires_at_ms or not self._connections:
            raise NetworkError("execution capacity lease expired")
        return role_client(self.result.experiment, self._connections)

    def close(self):
        failed = []
        for trust, control in reversed(self._rows):
            try:
                control_request(trust.origin, trust.credential_env, "/v1/party/release",
                                credentials=self._credentials, body=control)
            except NetworkError:
                failed.append((trust, control))
        self._rows = list(reversed(failed))
        self._connections.clear()
        # Unreachable controllers revoke at their authoritative TTL. Preserve
        # control references for a later cleanup retry, never permit execution.
        if not self._rows:
            self._credentials = None

    @property
    def cleanup_complete(self):
        return not self._rows


class _TokenizerView:
    """Reuse installed client renderer/tokenizer without loading provider bundles."""

    def __init__(self, descriptor: dict, config: dict, manifest: Any):
        self.tokenizer_descriptor = descriptor
        self.cfg = config
        self.manifest = {"chat_template": manifest.chat_template}

    def render_prompt(self, *args, **kwargs):
        from pllm.runtime.transformer_client import ClientBundle

        return ClientBundle.render_prompt(self, *args, **kwargs)

    def tokenizer(self):
        from pllm.runtime.transformer_client import ClientBundle

        return ClientBundle.tokenizer(self)


class _LeaseResponses:
    def __init__(self, lease: ExecutionLease, resource: Any):
        self._lease = lease
        self._resource = resource

    def create(self, **kwargs):
        # This check precedes manifest requests, preparation, session opening and streaming.
        self._lease._admit_request(kwargs)
        return self._resource.create(**kwargs)

    def retrieve(self, response_id):
        return self._resource.retrieve(response_id)

    def cancel(self, response_id):
        return self._resource.cancel(response_id)


class ExecutionLease:
    """Opaque single-owner lease. Exit closes sessions and releases role capacity."""

    __slots__ = (
        "_result",
        "_network",
        "_clock",
        "_topology",
        "_client",
        "_closed",
        "_lock",
        "_requests",
        "_tokenizer",
        "_binding",
        "_cleanup_complete",
    )

    def __init__(self, result, network, clock, topology, tokenizer, binding, *, _key=None):
        if _key is not _LEASE_KEY:
            raise TypeError("ExecutionLease must be obtained from open_execution")
        self._result = result
        self._network = network
        self._clock = clock
        self._topology = topology
        self._tokenizer = tokenizer
        self._binding = binding
        self._client = None
        self._closed = False
        self._cleanup_complete = False
        self._lock = threading.Lock()
        self._requests = 0

    def __repr__(self):
        return f"<ExecutionLease {self._network.backend} closed={self._closed}>"

    def __copy__(self):
        raise TypeError("ExecutionLease cannot be cloned")

    def __deepcopy__(self, memo):
        raise TypeError("ExecutionLease cannot be cloned")

    def __reduce__(self):
        raise TypeError("ExecutionLease cannot be serialized")

    @property
    def binding(self) -> ExecutionBinding:
        return self._binding

    @property
    def closed(self) -> bool:
        return self._closed

    def _check(self):
        if self._closed:
            raise NetworkError("execution lease is closed")
        if isinstance(self._binding, LiveExecutionBinding):
            if self._clock() >= self._binding.expires_at_ms:
                raise NetworkError("execution capacity lease expired")
            # Admission committed the immutable result once. After arm, capacity
            # TTL/epoch and provider session checks govern the bounded attempt;
            # old offer expiry cannot renew or revoke a capacity reservation.
            return
        self._result.validate(snapshot=self._result.snapshot if self._network.backend == "http"
                              else self._network.snapshot, evaluated_at_ms=self._clock())

    def _construct_client(self):
        with self._lock:
            self._check()
            if self._client is not None:
                raise NetworkError("execution lease already has a client")
            try:
                client = self._topology.client()
                client.responses = _LeaseResponses(self, client.responses)
                self._client = client
                return client
            except BaseException:
                self._closed = True
                self._topology.close()
                self._cleanup_complete = True
                raise

    def _admit_request(self, body: dict):
        from pllm.runtime.http_gateway import validate_response_body
        from pllm.runtime.responses import normalize_input
        from pllm.runtime.tools import tool_policy

        with self._lock:
            self._check()
            budget = self._result.experiment.budget
            expected_model = self._result.experiment.resolve().model
            body.setdefault("model", expected_model)
            if body["model"] != expected_model:
                raise NetworkError("request model conflicts with selected plan")
            body.setdefault("max_output_tokens", budget.max_new_tokens)
            validated = validate_response_body(body)
            cap = validated.get("max_output_tokens")
            if type(cap) is not int or not 1 <= cap <= budget.max_new_tokens:
                raise NetworkError("request output cap exceeds selected budget")
            messages = [
                row.to_prompt_dict()
                for row in normalize_input(
                    body.get("input", ""), instructions=body.get("instructions")
                )
            ]
            previous = body.get("previous_response_id")
            if previous:
                history = self._client._core.message_histories.get(str(previous))
                if history is None:
                    raise NetworkError("unknown previous_response_id")
                messages = [*history, *messages]
            policy = tool_policy(body)
            instruction = policy.prompt_instruction()
            if instruction:
                messages.insert(0, {"role": "system", "content": instruction})
            rendered = self._tokenizer.render_prompt(
                messages,
                add_generation_prompt=True,
                tools=policy.prompt_tools() if policy.enabled else None,
            )
            descriptor = self._tokenizer.tokenizer_descriptor
            tokens = self._tokenizer.tokenizer().encode(
                rendered, add_bos=bool(descriptor.get("add_bos_token", True))
            )
            if max(1, len(tokens)) > budget.max_input_tokens:
                raise NetworkError("request input exceeds selected budget")
            if self._requests >= budget.requests:
                raise NetworkError("execution request budget exhausted")
            self._requests += 1  # Failed/cancelled attempts also consume capacity.

    def close(self):
        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
        client_closed = False
        try:
            if self._client is not None:
                self._client.close()
            client_closed = True
        finally:
            self._topology.close()
            self._cleanup_complete = client_closed and getattr(self._topology, "cleanup_complete", True)

    def __enter__(self):
        try:
            self._check()
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, *_):
        import asyncio

        closing = asyncio.create_task(asyncio.to_thread(self.close))
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            await closing
            raise


def open_execution(
    result: PlanningResult, *, network: NetworkSpec, credentials=None, clock=None,
    ttl_seconds: int = 60,
) -> ExecutionLease:
    """Fresh native admission, config/source lock, then shared role admission/launch.

    Static local backend generates isolated role credentials. HTTP admission uses
    configured environment references (or their named local mapping), and private
    capacity tokens scoped to an attempt. Neither enters public bindings/archives.
    """
    from pllm.model_loader import resolve_model
    from pllm.modeling import lower_model
    from pllm.runtime.servers import build_roles
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    if not isinstance(result, PlanningResult) or not isinstance(network, NetworkSpec):
        raise TypeError("open_execution requires PlanningResult and NetworkSpec")
    if credentials is not None and network.backend == "local":
        raise NetworkError("local backend generates credentials; credential overrides unsupported")
    clock = (lambda: time.time_ns() // 1_000_000) if clock is None else clock
    if not callable(clock):
        raise TypeError("clock must be callable")
    # Replay before accepting even a manually constructed result object.
    result = PlanningResult.from_spec(result.to_spec())
    if not result.exhaustive and not result.request.policy.allow_incomplete_execution:
        raise NetworkError("SEARCH_LIMIT: incomplete execution needs explicit policy permission")
    native = result.validate(snapshot=result.snapshot if network.backend == "http"
                             else network.snapshot, evaluated_at_ms=clock())
    experiment = result.experiment
    if network.backend == "local" and experiment.deployment.kind != "local":
        raise NetworkError("network intent cannot execute through local backend")
    model = experiment.pipeline.model
    source = Path(model.source).expanduser()
    if model.kind not in {"huggingface", "safetensors"} or not source.is_dir():
        raise NetworkError("local selected execution requires an explicit existing checkpoint path")
    with (source / "config.json").open("rb") as stream:
        config = strict_load(stream.read((1 << 20) + 1))
    budget = experiment.budget
    lowered = lower_model(
        config,
        batch=1,
        max_input_tokens=budget.max_input_tokens,
        max_new_tokens=budget.max_new_tokens,
    )
    # Apply explicit native transformations only; no substitution of supplied semantics.
    for transform in result.request.model_plan.to_dict().get("transformations", []):
        raise NetworkError(
            f"selected local launcher cannot replay model transformation: {transform}"
        )
    if lowered.digest != result.request.model_plan.digest:
        raise NetworkError("actual checkpoint ModelPlan digest mismatch")
    resolved = resolve_model(model)
    if (
        result.request.source_lock_digest is not None
        and resolved.source_lock_digest != result.request.source_lock_digest
    ):
        raise NetworkError("actual checkpoint source lock mismatch")
    text_config = config.get("text_config", config)
    descriptor = MaskedTransformerEngine._load_tokenizer_descriptor(
        source, resolved.manifest, text_config
    )
    tokenizer = _TokenizerView(descriptor, text_config, resolved.manifest)
    if network.backend == "http":
        integer(ttl_seconds, "ttl_seconds", minimum=5, maximum=300)
        if result.request.source_lock_digest is None:
            raise NetworkError("network execution requires an exact source lock")
        topology = _LiveTopology(result, network, credentials, clock, ttl_seconds)
        try:
            native = topology.start(ttl_seconds)
            binding = LiveExecutionBinding(result.digest, network.digest, canonical(native),
                                           topology.attempt, topology.attempt_digest,
                                           result.request.source_lock_digest, topology.expires_at_ms)
            return ExecutionLease(result, network, clock, topology, tokenizer, binding, _key=_LEASE_KEY)
        except BaseException:
            topology.close()
            raise
    topology = build_roles(experiment)
    try:
        topology.start()
        # Startup time consumes offer validity too.
        native = result.validate(snapshot=network.snapshot, evaluated_at_ms=clock())
        binding = ExecutionBinding(result.digest, network.digest, canonical(native))
        return ExecutionLease(result, network, clock, topology, tokenizer, binding, _key=_LEASE_KEY)
    except BaseException:
        topology.close()
        raise


async def async_open_execution(result, *, network, credentials=None, ttl_seconds=60):
    """Cancellation-safe asynchronous admission; cleanup waits for partial reserves."""
    import asyncio

    task = asyncio.create_task(asyncio.to_thread(open_execution, result, network=network,
                                                credentials=credentials, ttl_seconds=ttl_seconds))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            lease = await task
        except Exception:
            pass
        else:
            await asyncio.to_thread(lease.close)
        raise


__all__ = ["ExecutionBinding", "LiveExecutionBinding", "ExecutionLease", "open_execution", "async_open_execution"]
