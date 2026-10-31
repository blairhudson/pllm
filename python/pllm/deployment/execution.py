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
from pllm.deployment.admission import ExecutionContract, RoleAdmission

_LEASE_KEY = object()


def _provisional_model_id(experiment):
    """Original routing convention from Python composition metadata, not admission."""
    from pllm.profiles import resolve_runtime_composition
    from pllm.model_loader import expected_model_id

    composition = resolve_runtime_composition(experiment.pipeline)
    if composition is None:
        raise NetworkError("pending composition has no recognized routing metadata")
    model = experiment.pipeline.model
    # Same distinction as ExperimentProfile, without its native resolution.
    if composition.privacy_mode == "proprietary":
        return model.model_id or model.source
    return expected_model_id(model)


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
    execution_digest: str

    def __post_init__(self):
        ExecutionBinding.__post_init__(self)
        for name in ("attempt_id", "attempt_digest", "source_lock_digest", "execution_digest"):
            digest_value(getattr(self, name), name, nonzero=True)
        integer(self.expires_at_ms, "expires_at_ms", minimum=1)
        expected = hashlib.sha256(b"pllm.execution_attempt.v2\0" + canonical({
            "attempt_id": self.attempt_id, "execution_digest": self.execution_digest,
            "network_digest": self.network_digest,
        })).hexdigest()
        if expected != self.attempt_digest:
            raise NetworkError("execution attempt binding mismatch")

    def to_spec(self):
        return {**ExecutionBinding.to_spec(self), "schema": "pllm.execution_binding.http.v2",
                 "attempt_id": self.attempt_id, "attempt_digest": self.attempt_digest,
                 "execution_digest": self.execution_digest,
                "source_lock_digest": self.source_lock_digest, "expires_at_ms": self.expires_at_ms}


class _LiveTopology:
    """Reserved endpoint adapter implementing the same client/close launcher interface."""

    def __init__(self, result, network, credentials, clock, ttl):
        self.result, self.network, self.clock = result, network, clock
        self._credentials = credentials
        self._rows = []
        self._connections = {}
        self.attempt = secrets.token_hex(32)
        self.contract = None
        self.attempt_digest = None
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
        self.contract = ExecutionContract.from_result(self.result, snapshot)
        native = self.contract.validate(evaluated_at_ms=self.clock())
        self.attempt_digest = hashlib.sha256(b"pllm.execution_attempt.v2\0" + canonical({
            "attempt_id": self.attempt, "execution_digest": self.contract.digest,
            "network_digest": self.network.digest,
        })).hexdigest()
        try:
            for _party, role, trust, offer in sorted(pending):
                admission = RoleAdmission(role, self.contract)
                control = {"attempt_id": self.attempt, "role_id": role,
                           "binding_digest": self.attempt_digest, "instance_epoch": offer.instance_epoch}
                # Record before request: a lost response may still have reserved capacity.
                self._rows.append((trust, control))
                value = control_request(trust.origin, trust.credential_env, "/v1/party/reserve",
                                        credentials=self._credentials, body={
                                             **control, "schema": "pllm.party_reserve.v2",
                                            "ttl_seconds": ttl, "descriptor_digest": offer.descriptor_digest,
                                              "admission": admission.to_spec(),
                                             **({"prepared_peer": peer} if peer is not None else {}),
                                        })
                exact(value, {"schema", "attempt_id", "role_id", "expires_at_ms", "capacity_token",
                              "binding_digest", "instance_epoch", "admission_digest"}, "pllm.party_capacity.v2")
                if value["admission_digest"] != admission.digest:
                    raise NetworkError("role admission acknowledgement mismatch")
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
                exact(armed, {"status", "binding_digest", "expires_at_ms", "admission_digest"})
                if armed["admission_digest"] != RoleAdmission(control["role_id"], self.contract).digest:
                    raise NetworkError("armed role admission mismatch")
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

    def __init__(self, descriptor: dict, config: dict, manifest: Any, *, original_config=None,
                 source_lock_digest=None):
        self.tokenizer_descriptor = descriptor
        self.cfg = config
        self.manifest = {"chat_template": manifest.chat_template}
        self.original_config = config if original_config is None else original_config
        self.source_lock_digest = source_lock_digest

    def _check_owned_prompt_metadata(self, source, model):
        """Check the lease snapshot before borrowing a client's source descriptor.

        BOS/EOS and template fallback belong to that existing source factory,
        not this generic importer's legacy defaults. The complete actual
        descriptor is still bound by _owned_prompt_source/consume afterward.
        """
        from pllm.runtime.responses import _prompt_snapshot

        if (
            _prompt_snapshot(source.model.to_spec()) != _prompt_snapshot(model.to_spec())
            or _prompt_snapshot(strict_load(source.config_source_bytes))
            != _prompt_snapshot(self.original_config)
            or source.source_lock_digest != self.source_lock_digest
        ):
            raise NetworkError("request source differs from selected source metadata")
        descriptor = source.tokenizer_descriptor
        # Full original tokenizer bytes/type/vocabulary must match. Do not
        # normalize zero to a default, or infer a template from observed content.
        for name in ("type", "model", "vocab_size"):
            if _prompt_snapshot(descriptor.get(name)) != _prompt_snapshot(
                self.tokenizer_descriptor.get(name)
            ):
                raise NetworkError("request tokenizer differs from selected source metadata")
    @classmethod
    def _from_prompt_snapshot(cls, snapshot):
        # Provisional request-scoped view, NOT an authenticated source owner.
        view = cls.__new__(cls)
        view._prompt_snapshot = snapshot
        view._prompt_tokenizer = None
        view.tokenizer_descriptor = snapshot.tokenizer_descriptor
        config = snapshot.config
        view.cfg = config.get("text_config", config)
        view.original_config = config
        view.manifest = {"chat_template": view.tokenizer_descriptor.get("chat_template")}
        view.source_lock_digest = snapshot.expected_source_lock
        return view

    def _check_prompt_snapshot(self, snapshot, model):
        from pllm.runtime.responses import _prompt_snapshot
        if (_prompt_snapshot(model.to_spec()) != snapshot.model_spec
                or _prompt_snapshot(self.original_config) != _prompt_snapshot(snapshot.config)):
            raise NetworkError("request metadata differs from selected source metadata")
        for name in ("type", "model", "vocab_size"):
            if _prompt_snapshot(self.tokenizer_descriptor.get(name)) != _prompt_snapshot(
                    snapshot.tokenizer_descriptor.get(name)):
                raise NetworkError("request tokenizer differs from selected source metadata")
    def _prepare_pending_prompt(self, body, *, messages, tools, _preparation_owner):
        from pllm.runtime.responses import _prepare_prompt

        snapshot = self._prompt_snapshot
        snapshot.check_unchanged()
        # Admission already retains this SAME owner. Let this helper and every
        # constructor/render/encode frame unwind before admission retires them.
        owner = _preparation_owner
        owner._reserve_metadata(self._prompt_memory_bytes)
        request_view = self._from_prompt_snapshot(snapshot)
        tokenizer = request_view.tokenizer()
        owner._adopt_tokenizer(tokenizer)
        return _prepare_prompt(body, descriptor=request_view.tokenizer_descriptor,
            tokenizer=tokenizer, source=snapshot.association(), messages=messages,
            tools=tools, _pending_source=snapshot, _preparation_owner=owner)
    def render_prompt(self, *args, **kwargs):
        from pllm.runtime.transformer_client import ClientBundle

        return ClientBundle.render_prompt(self, *args, **kwargs)

    def tokenizer(self):
        if hasattr(self, "_prompt_snapshot"):
            if self._prompt_tokenizer is None:
                self._prompt_tokenizer = self._prompt_snapshot.tokenizer()
            return self._prompt_tokenizer
        from pllm.runtime.transformer_client import ClientBundle

        return ClientBundle.tokenizer(self)


@dataclass(frozen=True, slots=True)
class _PendingPlanningInputs:
    """Immutable replay inputs, NOT validation, capacity or source authority."""

    document: bytes
    request: bytes
    snapshot: bytes
    experiment: bytes
    network: bytes
    source_lock: str | None

    @classmethod
    def capture(cls, result, network):
        document = result.canonical_bytes()
        if type(document) is not bytes or canonical(strict_load(document)) != document:
            raise NetworkError("pending result document is not immutable canonical JSON")
        value = strict_load(document)
        selection = value.get("selection")
        if type(selection) is not dict:
            raise NetworkError("pending result has no selected experiment")
        request = result.request.canonical_bytes()
        snapshot = result.snapshot.canonical_bytes()
        experiment = result.experiment.canonical_bytes()
        if (canonical(value.get("request")) != request
                or canonical(value.get("snapshot")) != snapshot
                or selection.get("configuration_digest")
                != result.experiment.configuration_digest()):
            raise NetworkError("pending result inputs differ from original document")
        return cls(document, request, snapshot, experiment, canonical(network.to_spec()),
                   result.request.source_lock_digest)

    def check(self, result, network, now, *, capacity_live=False):
        # Only local immutable identities, lifecycle and public metadata expiry.
        # No replay, planning validate, lowering, resolver or native contract.
        if (result.canonical_bytes() != self.document
                or result.request.canonical_bytes() != self.request
                or result.snapshot.canonical_bytes() != self.snapshot
                or result.experiment.canonical_bytes() != self.experiment
                or result.request.source_lock_digest != self.source_lock
                or canonical(network.to_spec()) != self.network):
            raise NetworkError("pending planning inputs changed")
        integer(now, "evaluated_at_ms")
        if capacity_live:
            return  # Original HTTP capacity TTL supersedes old offer expiry.
        snapshot = result.snapshot if network.backend == "http" else network.snapshot
        if (snapshot.observed_at_ms > now or snapshot.expires_at_ms <= now
                or now - snapshot.observed_at_ms > result.request.policy.max_observation_age_ms):
            raise NetworkError("STALE_SNAPSHOT")

    def replay(self, original, network, clock):
        self.check(original, network, clock())
        replayed = PlanningResult.from_spec(strict_load(self.document))
        self.check(replayed, network, clock())
        return replayed


class _LeaseResponses:
    def __init__(self, lease: ExecutionLease, resource: Any):
        self._lease = lease
        self._resource = resource

    @property
    def _delivery_owner(self):
        """Borrow actual delivery authority; never activate a pending lease."""
        if self._resource is not None:
            return self._resource.core
        with self._lease._lock:
            client = self._lease._client
            if self._lease._state != "active" or client is None:
                raise NetworkError("protected delivery owner unavailable before activation")
            from pllm.runtime.delegated_client import DelegatedClient

            owner = client._core
            if not isinstance(owner, DelegatedClient):
                raise NetworkError("activated delivery owner is not the protected client")
            return owner

    def _close_failed_delivery(self):
        # Original activation/attempt owners drain even if no response ID was
        # published. This is cleanup, not a replacement acceptance authority.
        self._lease.close()

    def create(self, **kwargs):
        if self._resource is None:
            return self._create_pending(kwargs)
        # This check precedes manifest requests, preparation, session opening and streaming.
        from pllm.runtime.responses import _prepared_prompt_request
        import inspect

        if inspect.iscoroutinefunction(self._resource.create):
            async def complete():
                prepared = self._lease._admit_request(kwargs)
                token = _prepared_prompt_request.set(prepared)
                try:
                    return await self._resource.create(**kwargs)
                finally:
                    _prepared_prompt_request.reset(token)
                    self._lease._finish_request_owner(prepared)
            return complete()
        prepared = self._lease._admit_request(kwargs)
        token = _prepared_prompt_request.set(prepared)
        try:
            return self._resource.create(**kwargs)
        finally:
            _prepared_prompt_request.reset(token)
            self._lease._finish_request_owner(prepared)

    def _create_pending(self, kwargs):
        from pllm.runtime.client import ResponseStream
        from pllm.runtime.responses import _prepared_prompt_request

        prepared = token = value = None
        transferred = False
        reserved = False
        with self._lease._lock:
            failure_serial = self._lease._failure_serial
        try:
            self._lease._reserve_request()
            reserved = True
            prepared = self._lease._admit_request(kwargs)
            self._lease._revalidate_counted_request()
            self._lease._activate_pending(prepared, kwargs)
            token = _prepared_prompt_request.set(prepared)
            # Actual post-arm resource. Pending binding never mints a receipt.
            self._lease._check()
            with self._lease._lock:
                if self._lease._closed:
                    raise NetworkError("execution lease is closed")
                resource = self._lease._client.responses
            value = resource.create(**kwargs)
            if isinstance(value, ResponseStream):
                retirement = threading.Lock()
                retired = False
                retirement_thread = retirement_event = None
                def retire():
                    nonlocal retired, retirement_thread, retirement_event
                    from pllm.runtime.delegated_client import _has_delegated_extent
                    client = self._lease._client
                    core = None if client is None else getattr(client, "_core", None)
                    if core is not None and _has_delegated_extent(core):
                        self._lease.close()
                        return
                    ident = threading.get_ident()
                    while True:
                        with retirement:
                            if retired or retirement_thread == ident:
                                return
                            if retirement_thread is None:
                                retirement_thread = ident
                                previous = retirement_event
                                retirement_event = event = threading.Event()
                                event._error = None
                                break
                            event = retirement_event
                        event.wait()
                        if event._error is not None:
                            raise event._error
                        return
                    previous = None  # Old generation/error drops outside gate.
                    import sys
                    primary = sys.exception()
                    try:
                        value.close()
                        self._lease._finish_request_owner(prepared)
                        with retirement:
                            retired = True
                    except BaseException as cleanup:
                        with retirement:
                            event._error = primary if primary is not None else cleanup
                        if primary is None or primary is cleanup:
                            raise
                        primary.add_note(f"stream cleanup retained owner: {type(cleanup).__name__}")
                    finally:
                        with retirement:
                            retirement_thread = None
                            event.set()
                transferred = True
                def terminal(response_id):
                    try:
                        if value._terminal is not None:
                            value._terminal(response_id)
                    finally:
                        retire()
                # Same iterator/cancellation and terminal owner; no new producer.
                return ResponseStream(value.iterator, close=retire, terminal=terminal)
            return value
        except BaseException as error:
            with self._lease._lock:
                failed_owner = reserved or not self._lease._request_busy
                # Even capacity that expires immediately after publication is
                # owned here. Do not leak active reservations on the late check.
                failed_prepared = prepared if prepared is not None else self._lease._request_owner
                should_close = (failed_owner and failure_serial == self._lease._failure_serial
                    and (failed_prepared is None or not failed_prepared._close_failed)
                    and self._lease._state in {"pending", "activating", "active", "closing"})
            if should_close:
                try:
                    self._lease.close()
                except BaseException as cleanup:
                    error.add_note(f"request cleanup retained owners: {type(cleanup).__name__}")
            raise
        finally:
            if token is not None:
                _prepared_prompt_request.reset(token)
            if reserved and not transferred:
                import sys
                primary = sys.exception()
                try:
                    if (prepared is not None and not prepared._close_failed
                            and failure_serial == self._lease._failure_serial):
                        self._lease._finish_request_owner(prepared)
                except BaseException as cleanup:
                    if primary is not None and primary is not cleanup:
                        primary.add_note(f"prepared cleanup retained owner: {type(cleanup).__name__}")
                    else:
                        raise
                else:
                    if prepared is None:
                        with self._lease._lock:
                            # Admission may retain a failed prepared owner even
                            # though it never returned one to this frame. Do not
                            # release its request or an in-flight closing state.
                            if (self._lease._request_owner is None
                                    and self._lease._state != "closing"):
                                self._lease._request_busy = False
    def retrieve(self, response_id):
        if self._resource is None:
            with self._lease._lock:
                self._lease._check()
                if self._lease._state != "active":
                    raise NetworkError("response resource unavailable before activation")
                return self._lease._client.responses.retrieve(response_id)
        return self._resource.retrieve(response_id)

    def cancel(self, response_id):
        if self._resource is None:
            with self._lease._lock:
                self._lease._check()
                if self._lease._state != "active":
                    raise NetworkError("response resource unavailable before activation")
                return self._lease._client.responses.cancel(response_id)
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
        "_pending_factory", "_pending_metadata_owner", "_source_owner", "_deferred",
        "_state", "_activation_event", "_activation_thread", "_activation_cancel",
        "_facade", "_request_busy", "_request_owner",
        "_pending_inputs",
        "_expected_protected_type", "_operation", "_operation_thread", "_operation_event",
        "_drain_attempt",
        "_close_ticket", "_failure_serial", "_operation_serial",
    )

    def __init__(self, result, network, clock, topology, tokenizer, binding, *, _key=None,
                  _pending_factory=None, _metadata_owner=None, _pending_inputs=None,
                  _expected_protected_type=None):
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
        self._pending_factory = _pending_factory
        self._pending_metadata_owner = _metadata_owner
        self._source_owner = None
        self._deferred = _pending_factory is not None
        self._state = "pending" if self._deferred else "active"
        self._activation_event = threading.Event()
        self._activation_event.set()
        self._activation_thread = self._activation_cancel = self._facade = None
        self._request_busy = False
        self._request_owner = None
        self._pending_inputs = _pending_inputs
        self._expected_protected_type = _expected_protected_type
        self._operation = "idle"
        self._operation_thread = None
        self._operation_event = threading.Event()
        self._operation_event.set()
        self._drain_attempt = None
        self._close_ticket = None
        self._failure_serial = 0
        self._operation_serial = 0

    def _uses_gkm_custody(self):
        from pllm.runtime.delegated_client import DelegatedClient
        return self._expected_protected_type is DelegatedClient

    def _begin_gkm_operation(self, operation):
        return self._begin_operation(operation)

    def _begin_operation(self, operation):
        """Only status under _lock; callbacks and generation waits stay outside.

        A producer may reenter while another thread is waiting on its attempt.
        Consult its independent status lock, never the attempt execution lock.
        """
        current = threading.get_ident()
        client = original = attempt = None
        while True:
            with self._lock:
                if self._operation == "idle":
                    generation = threading.Event()
                    generation._error = None
                    generation._failed = False
                    generation._operation = operation
                    self._operation_serial += 1
                    self._operation = operation
                    self._operation_thread = current
                    previous = self._operation_event
                    self._operation_event = generation
                    return True
                if self._operation_thread == current:
                    return False
                generation = self._operation_event
                client, original = self._client, self._drain_attempt
            if (self._uses_gkm_custody() and client is not None
                    and type(client._core) is self._expected_protected_type):
                attempt, producer, _, _ = client._core._lease_drain_snapshot(original)
                if producer == current:
                    attempt.cancel()
                    return False
            generation.wait()
            # Observe the captured generation, never a later mutable last error
            # or a recovered owner's new state. Waiting retirement is not retry.
            if generation._failed:
                raise generation._error
            if generation._operation == "retiring":
                return False
            generation = client = original = attempt = None
            # Final decrefs from an observed preparation/finishing generation
            # must finish before the next short state-lock acquisition.

    def _end_gkm_operation(self):
        self._end_operation()

    def _end_operation(self, *, failed=False, error=None):
        import sys
        selected = sys.exception() if error is None else error
        ticket = None
        with self._lock:
            self._operation_event._error = selected if failed else None
            self._operation_event._failed = failed
            if failed:
                self._failure_serial += 1
            if self._operation == "retiring" and not failed:
                ticket = self._close_ticket
                self._close_ticket = None  # This retirement consumed the request.
            self._operation = "idle"
            self._operation_thread = None
            self._operation_event.set()
        ticket = None  # Displaced custody's final decrefs run outside the gate.

    def _defer_from_delegated(self, core, attempt, generation):
        with self._lock:
            ticket = self._close_ticket
            if ticket is None:
                ticket = self._close_ticket = [object(), self._failure_serial, core, generation,
                    self._operation_serial, self._request_owner, attempt, self._client]
            elif ticket[2] is None:
                ticket[2:4] = core, generation
            elif ticket[2] is not core or ticket[3] != generation:
                self._failure_serial += 1  # Stale boundary cannot retire a new seat.
            if ticket[6] is None:
                ticket[6] = attempt
            if self._uses_gkm_custody() and self._drain_attempt is None:
                self._drain_attempt = attempt
            return ticket

    def _fail_deferred_close(self, ticket):
        with self._lock:
            if self._close_ticket is ticket:
                self._failure_serial += 1

    def _resume_deferred_close(self, ticket, core, generation):
        # Never wait behind a newer closer; never retry a failed cleanup here.
        with self._lock:
            client = self._client
        actual_core = None if client is None else getattr(client, "_core", None)
        with self._lock:
            if (self._close_ticket is not ticket or ticket[1] != self._failure_serial
                    or self._operation != "idle" or self._activation_thread is not None
                    or self._client is not client or actual_core is not core
                    or ticket[2] is not core or ticket[3] != generation
                    or ticket[4] != self._operation_serial
                    or ticket[5] is not self._request_owner or ticket[7] is not self._client
                    or (ticket[6] is not None and (ticket[6].owner is not core
                        or ticket[6]._generation != generation))):
                return
            self._operation_serial += 1
            self._operation = "retiring"
            self._operation_thread = threading.get_ident()
            previous = self._operation_event
            self._operation_event = threading.Event()
            self._operation_event._error = None
            self._operation_event._failed = False
            self._operation_event._operation = "retiring"
        failed = False
        try:
            # Admission seal plus ABA check; no lease lock or callbacks here.
            with core._lock:
                if getattr(core, "_attempt_generation", 0) != generation:
                    return
                core._closing = True
            if self._uses_gkm_custody():
                self._retire_gkm_payloads()
            elif self._deferred:
                self._retire_legacy_payloads()
            else:
                self._retire_eager_payloads()
        except BaseException:
            failed = True
            raise
        finally:
            self._end_operation(failed=failed)

    def _service_close_request(self, primary=None):
        from pllm.runtime.delegated_client import _defer_delegated_close

        with self._lock:
            ticket = self._close_ticket
            if (ticket is None or not self._closed or ticket[1] != self._failure_serial
                    or ticket[4] != self._operation_serial):
                return
            client = self._client
        core = None if client is None else getattr(client, "_core", None)
        if core is not None and _defer_delegated_close(core, self):
            return
        with self._lock:
            if (self._close_ticket is not ticket or ticket[1] != self._failure_serial
                    or ticket[4] != self._operation_serial or self._operation != "idle"
                    or self._activation_thread is not None or self._cleanup_complete):
                return
            # Atomic automatic claim: never wait behind, then retry, a newly
            # failed competing retirement after a check-then-close race.
            self._operation_serial += 1
            self._operation = "retiring"
            self._operation_thread = threading.get_ident()
            previous = self._operation_event
            self._operation_event = threading.Event()
            self._operation_event._error = None
            self._operation_event._failed = False
            self._operation_event._operation = "retiring"
        failed = False
        selected = None
        try:
            if self._uses_gkm_custody():
                self._retire_gkm_payloads()
            elif self._deferred:
                self._retire_legacy_payloads()
            else:
                self._retire_eager_payloads()
        except BaseException as cleanup:
            failed = True
            selected = primary if primary is not None else cleanup
            if primary is None or primary is cleanup:
                raise
            primary.add_note(f"deferred lease cleanup retained owners: {type(cleanup).__name__}")
        finally:
            self._end_operation(failed=failed, error=selected)

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
        if self._deferred:
            with self._lock:
                if self._state != "active" or self._binding is None:
                    raise NetworkError("execution binding is unavailable before successful activation")
        return self._binding

    @property
    def closed(self) -> bool:
        return self._closed

    def _diagnostic_client(self):
        """Read-only borrowed client, never an activation or delivery authority."""
        with self._lock:
            if self._closed or self._state != "active":
                return None
            return self._client

    def _check(self):
        if self._closed:
            raise NetworkError("execution lease is closed")
        if self._deferred and self._pending_inputs is not None:
            self._pending_inputs.check(self._result, self._network, self._clock(),
                                      capacity_live=isinstance(self._binding, LiveExecutionBinding))
        if isinstance(self._binding, LiveExecutionBinding):
            if self._clock() >= self._binding.expires_at_ms:
                raise NetworkError("execution capacity lease expired")
            # Admission committed the immutable result once. After arm, capacity
            # TTL/epoch and provider session checks govern the bounded attempt;
            # old offer expiry cannot renew or revoke a capacity reservation.
            return
        if self._deferred:
            # Pending facade/open/exclusion/count must not indirectly replay or
            # validate. Original planning validation follows successful count.
            return
        self._result.validate(snapshot=self._result.snapshot if self._network.backend == "http"
                              else self._network.snapshot, evaluated_at_ms=self._clock())

    def __getattr__(self, name):
        # Existing OpenAI facade may inspect models/runtime only after activation.
        if name.startswith("_"):
            raise AttributeError(name)
        with self._lock:
            if self._state != "active" or self._client is None:
                raise NetworkError("execution is pending; runtime metadata is unavailable")
            core = self._client._core
        return getattr(core, name)

    def _reserve_request(self):
        if not self._begin_operation("preparing"):
            raise NetworkError("execution request preparation is busy")
        try:
            self._check()
            with self._lock:
                if self._closed:
                    raise NetworkError("execution lease is closed")
                if self._request_busy or self._state == "activating":
                    raise NetworkError("execution already owns a request/activation")
                if self._network.backend == "http" and self._requests:
                    raise NetworkError("HTTP execution has one admitted attempt")
                self._request_busy = True
        finally:
            import sys
            primary = sys.exception()
            self._end_operation()
            self._service_close_request(primary)

    def _release_request(self):
        with self._lock:
            owner = self._request_owner
            if owner is not None and not (owner._closed or owner._claimed):
                raise NetworkError("execution request cleanup is pending")
            self._request_owner = None
            self._request_busy = False

    def _finish_request_owner(self, prepared):
        # Existing producer/attempt drain remains authoritative. A confirmed
        # claim transfers custody; an unclaimed failed close stays lease-rooted.
        import sys
        primary = sys.exception()
        if not self._begin_operation("finishing"):
            return
        failed = False
        selected = None
        try:
            try:
                prepared.close_unclaimed()
            except BaseException as cleanup:
                failed = True
                selected = primary if primary is not None else cleanup
                with self._lock:
                    self._closed = True
                    self._state = "closing"
                if primary is not None and primary is not cleanup:
                    primary.add_note(f"prepared cleanup retained owner: {type(cleanup).__name__}")
                    return
                raise
            with self._lock:
                if (not self._closed and self._request_owner is prepared
                        and (prepared._claimed or prepared._closed)):
                    self._request_owner = None
                    self._request_busy = False
        finally:
            self._end_operation(failed=failed, error=selected)
            if not failed:
                self._service_close_request(primary)

    def _revalidate_counted_request(self):
        if not self._begin_operation("preparing"):
            raise NetworkError("execution request validation is busy")
        try:
            self._check()
            if self._state == "active" and not isinstance(self._binding, LiveExecutionBinding):
                self._result.validate(snapshot=self._network.snapshot, evaluated_at_ms=self._clock())
            with self._lock:
                if self._closed:
                    raise NetworkError("execution lease is closed")
        finally:
            import sys
            primary = sys.exception()
            self._end_operation()
            self._service_close_request(primary)

    def _check_activation_cancelled(self):
        cancellation = self._activation_cancel
        if self._closed or (cancellation is not None and cancellation.event.is_set()):
            raise NetworkError("execution activation cancelled")

    def _activate_pending(self, prepared, body):
        from pllm.runtime.prepared_queue import PreparedCancellation, prepared_cancellation

        self._check()
        with self._lock:
            self._check_activation_cancelled()
            if self._state == "active":
                return
            if self._state != "pending" or self._pending_factory is None:
                raise NetworkError("execution cannot activate")
            self._state = "activating"
            self._activation_thread = threading.get_ident()
            self._activation_cancel = prepared_cancellation.get() or PreparedCancellation()
            self._activation_event.clear()
            factory = self._pending_factory
        try:
            self._check_activation_cancelled()
            binding = factory(self, prepared, body)
            self._check_activation_cancelled()
            if isinstance(binding, LiveExecutionBinding):
                if self._clock() >= binding.expires_at_ms:
                    raise NetworkError("execution capacity lease expired before publication")
            else:
                self._result.validate(snapshot=self._network.snapshot,
                    evaluated_at_ms=self._clock())
            with self._lock:
                self._check_activation_cancelled()
                # Source verification / native thread binding may consume the
                # remainder of real capacity or planning validity. Validate the
                # RETURNED binding here, inside the failure-owning activation
                # frame, immediately before either binding or active is visible.
                self._binding = binding  # Only genuine start/arm has returned it.
                self._state = "active"
                self._pending_factory = None
        except BaseException as error:
            with self._lock:
                self._closed = True
                self._state = "closing"
            try:
                self._retire_owners()
            except BaseException as cleanup:
                error.add_note(f"activation cleanup retained owners: {type(cleanup).__name__}")
            raise
        finally:
            with self._lock:
                self._activation_thread = None
                self._activation_cancel = None
                self._activation_event.set()

    def _retire_owners(self):
        if self._uses_gkm_custody():
            return self._retire_gkm_owners()
        if not self._begin_operation("retiring"):
            return
        failed = False
        try:
            if not self._cleanup_complete:
                if self._deferred:
                    self._retire_legacy_payloads()
                else:
                    self._retire_eager_payloads()
        except BaseException:
            failed = True
            raise
        finally:
            self._end_operation(failed=failed)

    def _retire_legacy_payloads(self):
        # HEAD deferred policy: topology is attempted even after client failure.
        errors = []
        client, owner, topology = self._client, self._request_owner, self._topology
        if client is not None:
            try:
                client.close()
                with self._lock:
                    self._client = None
            except BaseException as error:
                errors.append(error)
        if not errors and owner is not None:
            try:
                owner.close()
                with self._lock:
                    self._request_owner = None
            except BaseException as error:
                errors.append(error)
        if topology is not None:
            try:
                topology.close()
                if getattr(topology, "cleanup_complete", True):
                    with self._lock:
                        self._topology = None
                else:
                    errors.append(NetworkError("execution topology cleanup is pending"))
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"additional cleanup failure: {type(error).__name__}")
            raise errors[0]
        metadata = self._pending_metadata_owner
        if metadata is not None:
            metadata.close()
            self._pending_metadata_owner = None
        # Displaced references are locally retained through state publication.
        source, tokenizer, factory = self._source_owner, self._tokenizer, self._pending_factory
        with self._lock:
            self._source_owner = self._tokenizer = self._pending_factory = None
            self._request_busy = False
            self._cleanup_complete = True
            self._state = "closed"

    def _retire_eager_payloads(self):
        # HEAD eager FINALLY precedence, including a busy RuntimeClient close.
        client, owner, topology = self._client, self._request_owner, self._topology
        complete = False
        try:
            if client is not None:
                client.close()
            if owner is not None:
                owner.close()
            complete = True
        finally:
            if topology is not None:
                topology.close()
            self._cleanup_complete = complete and (topology is None or getattr(topology, "cleanup_complete", True))
            if self._cleanup_complete:
                self._client = self._request_owner = self._topology = None
                self._request_busy = False
                self._state = "closed"

    def _retire_gkm_owners(self):
        if not self._begin_gkm_operation("retiring"):
            return
        failed = False
        try:
            self._retire_gkm_payloads()
        except BaseException:
            failed = True
            with self._lock:
                self._closed = True
                self._state = "closing"
            raise
        finally:
            self._end_operation(failed=failed)

    def _retire_gkm_payloads(self):
        try:
            with self._lock:
                if self._cleanup_complete:
                    return
                client, original = self._client, self._drain_attempt
            if client is not None:
                core = client._core
                if type(core) is not self._expected_protected_type:
                    # Published partial custody is retained, not handed to a
                    # different family's unqualified retirement implementation.
                    raise NetworkError("activated client differs from protected capability; cleanup unresolved")
                attempt, _, _, _ = core._lease_drain_snapshot(original, seal=True)
                with self._lock:
                    self._drain_attempt = attempt
                client.close()
                if attempt is not None:
                    captured, producer, completion, retired = core._lease_drain_snapshot(attempt)
                    if captured is not attempt or producer is not None or not retired or not completion.is_set():
                        return  # Cancel-only is NOT drain; retain all dependencies.
                with self._lock:
                    self._client = None
                    self._drain_attempt = None
            owner = self._request_owner
            if owner is not None:
                owner.close()
                with self._lock:
                    self._request_owner = None
            topology = self._topology
            if topology is not None:
                topology.close()
                if not getattr(topology, "cleanup_complete", True):
                    raise NetworkError("execution topology cleanup is pending")
                with self._lock:
                    self._topology = None
            metadata = self._pending_metadata_owner
            if metadata is not None:
                metadata.close()
                with self._lock:
                    self._pending_metadata_owner = None
            with self._lock:
                source, tokenizer, factory = self._source_owner, self._tokenizer, self._pending_factory
                self._source_owner = self._tokenizer = self._pending_factory = None
                self._request_busy = False
                self._cleanup_complete = True
                self._state = "closed"
        except BaseException:
            with self._lock:
                self._closed = True
                self._state = "closing"
            raise

    @property
    def default_model(self):
        # Gateway routing metadata only; never a source/admission receipt.
        if self._deferred:
            return _provisional_model_id(self._result.experiment)
        return self._result.experiment.resolve().model

    def _pending_facade(self):
        from pllm.runtime.client import OpenAI, ModelsResource, RuntimeResource

        facade = OpenAI.__new__(OpenAI)  # Existing facade, no provider/client started.
        facade._core = self
        facade._owned_topology = None
        facade.responses = _LeaseResponses(self, None)
        facade.models = ModelsResource(self)
        facade.runtime = RuntimeResource(self)
        return facade
    def _close_pending(self):
        return self.close()

    def _construct_client(self):
        if not self._begin_operation("preparing"):
            raise NetworkError("execution client construction is busy")
        failed = False
        try:
            self._check()
            with self._lock:
                if self._closed:
                    raise NetworkError("execution lease is closed")
                if self._facade is not None or self._client is not None:
                    raise NetworkError("execution lease already has a client")
            if self._deferred:
                facade = self._pending_facade()
                with self._lock:
                    if self._closed:
                        raise NetworkError("execution lease is closed")
                    self._facade = facade
                return facade
            client = None
            try:
                client = self._topology.client()
                with self._lock:
                    self._client = client  # Root partial construction BEFORE callbacks.
                    if self._closed:
                        raise NetworkError("execution lease is closed")
                limit = self._result.request.policy.max_client_native_threads
                if (limit is not None
                        and self._result.experiment.resolve().client_runtime == "compiled_client_local_v1"):
                    client._core._bind_native_worker_limit(limit)
                client.responses = _LeaseResponses(self, client.responses)
                with self._lock:
                    if self._closed:
                        raise NetworkError("execution lease is closed")
                return client
            except BaseException:
                with self._lock:
                    self._closed = True
                    self._state = "closing"
                try:
                    self._retire_eager_payloads()
                finally:
                    failed = not self._cleanup_complete
                raise
        finally:
            import sys
            primary = sys.exception()
            self._end_operation(failed=failed)
            if not failed:
                self._service_close_request(primary)

    def _admit_request(self, body: dict):
        if not self._begin_operation("preparing"):
            raise NetworkError("execution request preparation is busy")
        with self._lock:
            previous_owner = self._request_owner
        try:
            return self._prepare_admitted_request(body)
        finally:
            import sys
            primary = sys.exception()
            owner = self._request_owner
            failed = (primary is not None and owner is not None
                      and owner is not previous_owner and not owner._closed)
            self._end_operation(failed=failed)
            if not failed:
                self._service_close_request(primary)

    def _prepare_gkm_request(self, body):
        return self._prepare_admitted_request(body)

    def _prepare_admitted_request(self, body):
        from pllm.runtime.http_gateway import validate_response_body
        from pllm.runtime.responses import normalize_input, _PreparedPromptUse
        from pllm.runtime.tools import tool_policy

        with self._lock:
            if self._closed:
                raise NetworkError("execution lease is closed")
            if self._request_owner is not None:
                raise NetworkError("execution already owns request preparation/cleanup")
            prepared = _PreparedPromptUse._empty()
            self._request_owner = prepared
            self._request_busy = True
            client = self._client
            pending = self._deferred and self._state == "pending"
        try:
            # The empty root and exclusive operation precede ALL callbacks.
            self._check()
            budget = self._result.experiment.budget
            expected_model = self.default_model
            body.setdefault("model", expected_model)
            if body["model"] != expected_model:
                raise NetworkError("request model conflicts with selected plan")
            body.setdefault("max_output_tokens", budget.max_new_tokens)
            validated = validate_response_body(body)
            cap = validated.get("max_output_tokens")
            if type(cap) is not int or not 1 <= cap <= budget.max_new_tokens:
                raise NetworkError("request output cap exceeds selected budget")
            messages = [row.to_prompt_dict() for row in normalize_input(
                body.get("input", ""), instructions=body.get("instructions"))]
            core = None if client is None else client._core
            if (self._uses_gkm_custody() and core is not None
                    and type(core) is not self._expected_protected_type):
                raise NetworkError("activated client differs from protected capability")
            previous = body.get("previous_response_id")
            if previous:
                histories = None if core is None else getattr(core, "message_histories", None)
                history = None if histories is None else histories.get(str(previous))
                if history is None:
                    raise NetworkError("unknown previous_response_id")
                messages = [*history, *messages]
            policy = tool_policy(body)
            instruction = policy.prompt_instruction()
            if instruction:
                messages.insert(0, {"role": "system", "content": instruction})
            if self._requests >= budget.requests:
                raise NetworkError("execution request budget exhausted")
            self._check()  # A validation/tool callback may have requested close.
            self._populate_admitted_prompt(prepared, body, core, pending, messages, policy)
            if len(prepared._payload.ids) > budget.max_input_tokens:
                raise NetworkError("request input exceeds selected budget")
            self._check()
            with self._lock:
                if self._closed:
                    raise NetworkError("execution lease is closed")
                self._requests += 1
            return prepared
        except BaseException as error:
            prepared._failure = error
            with self._lock:
                self._closed = True
                self._state = "closing"
            try:
                prepared.close()
            except BaseException as cleanup:
                error.add_note(f"admission cleanup retained owner: {type(cleanup).__name__}")
            else:
                with self._lock:
                    self._request_owner = None
                    self._request_busy = False
            raise

    def _populate_admitted_prompt(self, prepared, body, core, pending, messages, policy):
        from pllm.runtime.delegated_client import DelegatedClient
        from pllm.runtime.responses import _prepare_prompt, _prompt_source

        tools = policy.prompt_tools() if policy.enabled else None
        if pending:
            self._tokenizer._prepare_pending_prompt(body, messages=messages,
                tools=tools, _preparation_owner=prepared)
        elif isinstance(core, DelegatedClient):
            from pllm.runtime.http_gateway import GatewayError
            try:
                core._prepare_request_prompt(body, messages=messages, tools=tools,
                    metadata_view=self._tokenizer, _preparation_owner=prepared)
            except GatewayError as error:
                raise NetworkError(str(error)) from error
        else:
            descriptor = self._tokenizer.tokenizer_descriptor
            tokenizer = self._tokenizer.tokenizer()
            prepared._adopt_tokenizer(tokenizer)
            template = self._tokenizer.manifest.get("chat_template")
            source = _prompt_source(self._result.experiment.pipeline.model,
                self._tokenizer.original_config, descriptor, template,
                self._tokenizer.source_lock_digest)
            _prepare_prompt(body, descriptor=descriptor, tokenizer=tokenizer,
                source=source, messages=messages, template=template, tools=tools,
                _preparation_owner=prepared)

    def close(self):
        from pllm.runtime.delegated_client import _defer_delegated_close

        with self._lock:
            if self._cleanup_complete:
                return
            self._closed = True
            self._state = "closing"
            if self._close_ticket is None:
                self._close_ticket = [object(), self._failure_serial, None, None,
                    self._operation_serial, self._request_owner, None, self._client]
            thread, cancellation = self._activation_thread, self._activation_cancel
            operation_thread, client = self._operation_thread, self._client
        core = None if client is None else getattr(client, "_core", None)
        # BEFORE even an idle retirement claim: cancel-only is never drain proof.
        if core is not None and _defer_delegated_close(core, self):
            return
        if cancellation is not None:
            cancellation.cancel()
        if operation_thread == threading.get_ident() or thread == threading.get_ident():
            return
        if thread is not None:
            self._activation_event.wait()
        self._retire_owners()

    def _close_gkm(self):
        return self.close()

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


def _open_protected_pending(result, network, credentials, clock, ttl_seconds, config, client_type,
                            *, _planning_inputs=None):
    """Existing lease, deferred original admission. No active topology or binding."""
    from contextlib import ExitStack
    from pllm.deployment.resources import reserve_incremental_memory
    from pllm.runtime import tokenizer_admission

    experiment = result.experiment
    model = experiment.pipeline.model
    if network.backend == "http":
        integer(ttl_seconds, "ttl_seconds", minimum=5, maximum=300)
        if result.request.source_lock_digest is None:
            raise NetworkError("network execution requires an exact source lock")
    capacity = experiment.pipeline.components["linear"].params.get("capacity", "capsule")
    metadata_owner = ExitStack()
    try:
        metadata_owner.enter_context(reserve_incremental_memory(
            tokenizer_admission.DECODER_METADATA_MEMORY_BYTES if capacity == "decoder" else 1 << 20))
        snapshot = tokenizer_admission._capture_prompt_metadata(model,
            policy=client_type._prompt_descriptor_policy, capacity=capacity,
            input_rows=experiment.budget.max_input_tokens, outputs=experiment.budget.max_new_tokens,
            expected_source_lock=result.request.source_lock_digest)
        from pllm.runtime.responses import _prompt_snapshot
        if _prompt_snapshot(config) != _prompt_snapshot(snapshot.config):
            raise NetworkError("source config changed during pending plan validation")
        view = _TokenizerView._from_prompt_snapshot(snapshot)
        view._prompt_memory_bytes = (tokenizer_admission.DECODER_METADATA_MEMORY_BYTES
                                     if capacity == "decoder" else 1 << 20)
        if _planning_inputs is None:
            _planning_inputs = _PendingPlanningInputs.capture(result, network)

        def activate(lease, prepared, body):
            from pllm.runtime.servers import build_roles

            lease._check_activation_cancelled()
            from pllm.modeling import lower_model

            admitted = _planning_inputs.replay(result, network, clock)
            lease._check_activation_cancelled()
            # Provisional Python recognition NEVER replaces original complete
            # runtime resolution (including native resolution where required).
            from pllm.runtime.client import _protected_client_type
            resolved_profile = admitted.experiment.resolve()
            if _protected_client_type(resolved_profile) is not client_type:
                raise NetworkError("replayed runtime differs from pending composition")
            if resolved_profile.model != _provisional_model_id(admitted.experiment):
                raise NetworkError("replayed runtime model differs from pending metadata")
            admitted.validate(snapshot=admitted.snapshot if network.backend == "http"
                              else network.snapshot, evaluated_at_ms=clock())
            # Same original snapshot config and exact budget. No source
            # resolver has run, and no topology/reservation exists yet.
            lowered = lower_model(snapshot.config, batch=1,
                max_input_tokens=admitted.experiment.budget.max_input_tokens,
                max_new_tokens=admitted.experiment.budget.max_new_tokens)
            for transform in admitted.request.model_plan.to_dict().get("transformations", []):
                raise NetworkError(
                    f"selected local launcher cannot replay model transformation: {transform}")
            if lowered.digest != admitted.request.model_plan.digest:
                raise NetworkError("actual checkpoint ModelPlan digest mismatch")
            _planning_inputs.check(admitted, network, clock())
            with lease._lock:
                lease._check_activation_cancelled()
                lease._result = admitted
            # ORIGINAL source resolver/lock/contract path, now after prompt count.
            source = client_type._resolve_prompt_source(experiment)
            lease._source_owner = source
            lease._check_activation_cancelled()
            prepared._bind_owned_source(body, source)
            lease._tokenizer.source_lock_digest = source.source_lock_digest
            admitted.validate(snapshot=admitted.snapshot if network.backend == "http" else network.snapshot,
                            evaluated_at_ms=clock())
            lease._check_activation_cancelled()
            if network.backend == "http":
                topology = _LiveTopology(admitted, network, credentials, clock, ttl_seconds)
                lease._topology = topology  # Retained BEFORE reserve/arm can fail.
                native = topology.start(ttl_seconds)
                binding = LiveExecutionBinding(admitted.digest, network.digest, canonical(native),
                    topology.attempt, topology.attempt_digest, admitted.request.source_lock_digest,
                    topology.expires_at_ms, topology.contract.digest)
            else:
                topology = build_roles(experiment)
                lease._topology = topology
                topology.start()
                native = admitted.validate(snapshot=network.snapshot, evaluated_at_ms=clock())
                binding = ExecutionBinding(admitted.digest, network.digest, canonical(native))
            lease._check_activation_cancelled()
            client = topology.client()
            lease._client = client  # Keep partial owner if later validation/cleanup fails.
            core = client._core
            if (type(core) is not client_type if lease._uses_gkm_custody()
                    else not isinstance(core, client_type)):
                raise NetworkError("activated client differs from protected capability")
            with core._lock:
                core._check_open()
                if core._source is not None:
                    raise NetworkError("activation cannot adopt a separately retained source")
                core._source = source  # SAME admitted owner, not a digest/descriptor adopter.
                core._source_checked = False
            core._ensure_source()  # Original per-client contract checks remain intact.
            limit = admitted.request.policy.max_client_native_threads
            if limit is not None and resolved_profile.client_runtime == "compiled_client_local_v1":
                core._bind_native_worker_limit(limit)
            lease._check_activation_cancelled()
            return binding

        return ExecutionLease(result, network, clock, None, view, None,
            _key=_LEASE_KEY, _pending_factory=activate, _metadata_owner=metadata_owner,
            _pending_inputs=_planning_inputs, _expected_protected_type=client_type)
    except BaseException as error:
        snapshot = view = None
        try:
            metadata_owner.close()
        except BaseException as cleanup:
            error.add_note(f"pending metadata cleanup retained owner: {type(cleanup).__name__}")
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
    from pllm.runtime.client import _protected_client_type
    from pllm.profiles import resolve_runtime_composition

    # Decide the already-declared protected path without replaying search,
    # validating a native plan or lowering any model. Inputs remain provisional.
    provisional = (None if result.experiment is None
                   else resolve_runtime_composition(result.experiment.pipeline))
    protected_client = _protected_client_type(provisional)
    if protected_client is not None:
        from pllm.runtime.tokenizer_admission import _read_named
        from pllm.deployment.resources import reserve_incremental_memory

        pending = _PendingPlanningInputs.capture(result, network)
        pending.check(result, network, clock())
        if not result.exhaustive and not result.request.policy.allow_incomplete_execution:
            raise NetworkError("SEARCH_LIMIT: incomplete execution needs explicit policy permission")
        if network.backend == "local" and result.experiment.deployment.kind != "local":
            raise NetworkError("network intent cannot execute through local backend")
        model = result.experiment.pipeline.model
        source = Path(model.source).expanduser()
        if model.kind not in {"huggingface", "safetensors"} or not source.is_dir():
            raise NetworkError("local selected execution requires an explicit existing checkpoint path")
        with reserve_incremental_memory((1 << 20) + 1):
            config_bytes, _ = _read_named(source, "config.json", 1 << 20)
            config = strict_load(config_bytes)
        return _open_protected_pending(result, network, credentials, clock, ttl_seconds,
                                       config, protected_client, _planning_inputs=pending)
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
    from pllm.runtime.client import _protected_client_type
    protected_client = _protected_client_type(experiment.resolve())
    if protected_client is not None:
        from pllm.runtime.tokenizer_admission import _read_named
        from pllm.deployment.resources import reserve_incremental_memory
        # Original startup bound, including its oversize-detection byte.
        with reserve_incremental_memory((1 << 20) + 1):
            config_bytes, _ = _read_named(source, "config.json", 1 << 20)
            config = strict_load(config_bytes)
    else:
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
    if protected_client is not None:
        return _open_protected_pending(result, network, credentials, clock, ttl_seconds, config, protected_client)
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
    tokenizer = _TokenizerView(descriptor, text_config, resolved.manifest,
                               original_config=config, source_lock_digest=resolved.source_lock_digest)
    if network.backend == "http":
        integer(ttl_seconds, "ttl_seconds", minimum=5, maximum=300)
        if result.request.source_lock_digest is None:
            raise NetworkError("network execution requires an exact source lock")
        topology = _LiveTopology(result, network, credentials, clock, ttl_seconds)
        try:
            native = topology.start(ttl_seconds)
            binding = LiveExecutionBinding(result.digest, network.digest, canonical(native),
                                           topology.attempt, topology.attempt_digest,
                                           result.request.source_lock_digest, topology.expires_at_ms,
                                           topology.contract.digest)
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
