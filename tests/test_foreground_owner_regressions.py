"""Independent SDK-entry controls for foreground/preparation ownership.

Model execution and HTTP services are doubled; authority, inventory reservation,
SDK/context entry and retirement are real. No native/numeric or RSS claim.
Known-defect cases assert the required contract, rather than accepting a private
flag value as evidence that borrowed source/material owners remain live.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import json
import signal
import threading
from types import SimpleNamespace

import httpx
import msgpack
import pytest

from pllm import OpenAI
from pllm.runtime.client import ModelError, ProtocolError, RuntimeClient
from pllm.runtime.types import Response, ResponseEvent
from test_paged_prewarmed_foreground_gate import (
    MODEL,
    PRIVATE,
    STAGES,
    DeferredExecutor,
    harness,  # noqa: F401 -- reuse authority/service fixture, not its execution oracle
)


@pytest.fixture
def endpoint(harness, monkeypatch):
    """Construct real public client; install already committed synthetic rows."""
    h = harness
    sdk = OpenAI(base_url=str(h.core.http.base_url), api_key="inference-key",
                 default_model=MODEL, preparation_api_key="preparation-key",
                 http_client=h.core.http, preparation_http_client=h.core.preparation_http,
                 bundle_cache_dir=h.core.bundle_cache_dir / "independent")
    core = sdk._core
    bundle = replace(h.bundle, arrays={})
    monkeypatch.setattr(core, "_load_client_bundle_record", lambda *a, **kw: (bundle, "f" * 64))
    bound = core._transformer_state(MODEL)
    inventory = h.inventory(name="independent-committed")
    inventory._audit = core.audit.record_inventory
    bound.prepared_inventory = inventory
    spare = h.inventory(name="independent-committed-spare")
    spare._audit = core.audit.record_inventory
    bound.prepared_inventory_spare = spare
    core._provider_executor.shutdown(wait=True)
    core._provider_executor = DeferredExecutor()
    observation = SimpleNamespace(sdk=sdk, core=core, bound=bound, inventory=inventory,
                                  leases=[], retirement=[], h=h, reset_error=None,
                                  execute_hook=None)

    def numeric_double(body, messages, descriptor, **kwargs):
        # This is an independent numeric boundary, not L's probe implementation.
        # Real dispatch has already normalized/validated the public request.
        current = core._transformer_state(MODEL)
        core._ensure_prepared_inventory(MODEL, current, 4)
        session, current, lease = core._open_transformer_session(
            MODEL, max_output_tokens=2, required_rows=4)
        observation.leases.append(lease)
        complete = False
        try:
            if observation.execute_hook is not None:
                observation.execute_hook()
            yield ResponseEvent.from_dict({"type": "response.created", "sequence_number": 0,
                "response": Response(id=session["response_id"], model=MODEL,
                                     output=[], status="in_progress").to_dict()})
            for stage in STAGES:
                lease.take(stage, 1)
            complete = True  # Mock early EOS; remaining admitted rows must burn.
            yield ResponseEvent.from_dict({"type": "response.completed", "sequence_number": 1,
                "response": Response(id=session["response_id"], model=MODEL,
                                     output=[], status="completed").to_dict()})
        finally:
            try:
                if not complete:
                    core._cancel_transformer_session(session["id"])
            finally:
                core._finish_prepared_response(MODEL, current, lease)
                observation.retirement.append(core._foreground_active)
                if observation.reset_error is not None:
                    raise observation.reset_error

    monkeypatch.setattr(core, "_transformer_events", numeric_double)
    h.http.clear()
    try:
        yield observation
    finally:
        sdk.close()


def test_real_sdk_context_first_committed_response_and_explicit_idle_preparation(endpoint):
    e = endpoint
    with e.sdk as client:
        # Context entry plus committed inventory is sufficient for FIRST create;
        # do not repair its state with an intervening preparation call.
        response = client.responses.create(input=PRIVATE, max_output_tokens=2)
        assert response.status == "completed"
        assert e.leases[0]._closed and e.retirement == [1]
        status = e.inventory.status()
        assert (status["reserved"], status["consumed"], status["burned"]) == (0, 1, 3)
        assert e.core._foreground_active == e.core._online_active == 0
        assert client.preprocess(count=4)["generated"] == 0
        assert not any("/stages/" in r.url.path or r.url.path == "/v1/runtime/inventories"
                       for r in e.h.http)
        assert all(PRIVATE.encode() not in r.content for r in e.h.http)


def test_caller_throw_keeps_original_error_and_owner_through_failing_cleanup(endpoint, monkeypatch):
    e = endpoint
    violations = observe_retirement(e, monkeypatch)
    original = ValueError("caller failure")
    e.reset_error = RuntimeError("mock numeric retirement failure")
    with pytest.raises(ValueError) as caught:
        with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
            assert next(stream).type == "response.created"
            raise original
    assert caught.value is original
    assert e.leases[0]._closed and e.retirement == [1]
    assert e.core._foreground_active == e.core._online_active == 0
    assert not violations


def test_independent_client_can_prepare_while_other_client_holds_foreground(endpoint):
    e = endpoint
    e.h.bound.prepared_inventory_spare = e.h.inventory(name="other-client-spare")
    with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
        assert next(stream).type == "response.created"
        assert e.h.core.preprocess(MODEL, 1)["generated"] == 0
        assert e.core._foreground_active == 1 and e.h.core._foreground_active == 0
    assert e.leases[0]._closed and e.retirement == [1]


def test_wrong_source_authority_fails_without_session_or_material(endpoint):
    e = endpoint
    e.bound.preparation_verified = False
    e.h.authority_override["body_fingerprint"] = "0" * 64
    with pytest.raises(ProtocolError, match="commitments do not match"):
        e.sdk.responses.create(input=PRIVATE, max_output_tokens=2)
    assert not e.leases and not any(r.method == "POST" for r in e.h.http)
    assert e.core._foreground_active == e.core._online_active == 0


def test_corrupt_raw_source_uses_real_loader_and_retires_foreground(endpoint, monkeypatch):
    e = endpoint
    transport = e.core.http._transport
    original = transport.handle_request

    def corrupt(request):
        if request.url.path.endswith("/client-bundle"):
            e.h.http.append(request)
            # Descriptor promises f*64; genuine byte verification must reject
            # before unpack/native factories. No fake integrity checker.
            return httpx.Response(200, content=b"\xc0",
                                  headers={"X-PLLM-Bundle-SHA256": "f" * 64})
        return original(request)

    monkeypatch.setattr(transport, "handle_request", corrupt)
    monkeypatch.setattr(e.core, "_load_client_bundle_record",
                        RuntimeClient._load_client_bundle_record.__get__(e.core))
    e.bound.bundle_fingerprint = "0" * 64
    with pytest.raises(ProtocolError, match="fingerprint mismatch"):
        e.sdk.responses.create(input=PRIVATE, max_output_tokens=2)
    assert any(r.url.path.endswith("/client-bundle") for r in e.h.http)
    assert not e.leases and not any(r.method == "POST" for r in e.h.http)
    assert e.core._foreground_active == e.core._online_active == 0


def test_client_close_retires_or_rejects_borrowed_plain_foreground_owner(endpoint, monkeypatch):
    e = endpoint
    violations = observe_retirement(e, monkeypatch)
    stream = e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True)
    try:
        assert next(stream).type == "response.created"
        lease = e.leases[0]
        try:
            e.sdk.close()
        except Exception:
            pass  # Busy/error is judged by ownership below, not a new error type.
        if lease._closed:
            assert e.retirement == [1]
        else:
            # Busy/deferred close is safe only while original resources/counters
            # stay resident. This does NOT accept the old cancel-live-owner.
            assert not e.inventory._cancelled
            assert e.inventory.status()["reserved"] == 4
            assert e.core._foreground_active > 0 and e.core._online_active > 0
        assert not violations
    finally:
        stream.close()
    assert lease._closed and e.core._foreground_active == e.core._online_active == 0
    e.sdk.close()


class MaterialAttempted(RuntimeError):
    pass


@pytest.fixture
def legacy():
    observation = SimpleNamespace(requests=[], foreground_at_generation=[], sdk=None)

    def handler(request):
        observation.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == "/v1/runtime/models/legacy":
            return httpx.Response(200, json={"id": "legacy", "metadata": {
                "client_runtime": "masked_bigram_v1", "privacy_mode": "proprietary"}})
        if request.method == "POST" and path == "/v1/runtime/sessions":
            ordinal = sum(r.method == "POST" and r.url.path == path for r in observation.requests)
            return httpx.Response(200, json={"id": f"legacy-session-{ordinal}", "response_id": f"legacy-response-{ordinal}",
                "manifest": {"runtime": {"alphabet": "ab ", "modulus": 257, "max_output_tokens": 4}}})
        if request.method == "POST" and "/correlations/" in path:
            core = observation.sdk._core
            if core._foreground_active:
                observation.foreground_at_generation.append(core._foreground_active)
                raise MaterialAttempted("foreground reached legacy fresh-material endpoint")
            dimension = core._crypto_states["legacy"].model.tokenizer.vocab_size
            count = json.loads(request.content)["count"]
            assert 1 <= count <= 4
            return httpx.Response(200, content=msgpack.packb({"items": [
                {"id": f"idle-{index}", "shape": [dimension], "mask": bytes(8 * dimension),
                 "transformed": bytes(8 * dimension)} for index in range(count)]}, use_bin_type=True))
        if request.method == "POST" and path.endswith(("/cancel", "/complete")):
            return httpx.Response(200, json={"status": "cancelled"})
        pytest.fail(f"unexpected legacy control: {request.method} {path}")

    with httpx.Client(base_url="https://legacy.test", transport=httpx.MockTransport(handler)) as http:
        sdk = OpenAI(base_url=str(http.base_url), api_key="test-key", default_model="legacy",
                     correlation_mode="local-test", http_client=http)
        observation.sdk = sdk
        try:
            yield observation
        finally:
            sdk.close()


def test_closed_sdk_rejects_before_legacy_authority_or_session_controls(legacy):
    legacy.sdk.close()
    before = len(legacy.requests)
    with legacy.sdk.responses.create(input="ab", max_output_tokens=1, stream=True) as stream:
        with pytest.raises(ModelError, match="closed"):
            next(stream)
    assert len(legacy.requests) == before


def test_legacy_underflow_rejects_instead_of_implicitly_generating_online(legacy):
    # Legacy local-test transport makes the otherwise BFV generation boundary
    # observable without running crypto/model code. No protocol bypass is added.
    with legacy.sdk.responses.create(input="ab", max_output_tokens=1, stream=True) as stream:
        with pytest.raises(ModelError):
            list(stream)
    assert not legacy.foreground_at_generation


class WatchdogExpired(BaseException):
    pass


@contextmanager
def no_self_wait():
    """Interrupt a bad same-thread close; never abandon a deadlocked thread.

    All ordering comes from Events below. This main-thread watchdog only bounds
    a broken implementation waiting for its own running generator to retire.
    """
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        pytest.skip("same-thread deadlock watchdog requires main-thread POSIX signals")
    expired = []

    def interrupt(*_):
        expired.append(True)
        raise WatchdogExpired("close waited for its own foreground/preparation callback")

    previous = signal.signal(signal.SIGALRM, interrupt)
    timer = signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *timer)
        assert not expired, "same-thread close/admission did not return within watchdog"


def observe_retirement(e, monkeypatch, *, preparation=False):
    """Observe real cancel/deallocation entry, not just activity-counter values."""
    violations = []
    inventory_type = type(e.inventory)
    bundle_type = type(e.bound.bundle)
    cancel = inventory_type.cancel
    close_bundle = bundle_type.close
    owned = (e.inventory, e.bound.prepared_inventory_spare)

    def check():
        if (any(not lease._closed for lease in e.leases)
                or any(value is not None and value.status()["reserved"] for value in owned)
                or preparation and e.core._preparation_active):
            violations.append("resource retirement entered before original borrower retired")

    def watched_cancel(value):
        if any(value is owner for owner in owned) and not value._cancelled:
            check()
        return cancel(value)

    def watched_bundle(value):
        if value is e.bound.bundle:
            check()
        return close_bundle(value)

    monkeypatch.setattr(inventory_type, "cancel", watched_cancel)
    monkeypatch.setattr(bundle_type, "close", watched_bundle)
    return violations


def finish_thread(thread, release):
    release.set()
    thread.join(timeout=4)
    assert not thread.is_alive(), "cooperative test caller did not drain"


def test_same_thread_execution_callback_close_never_self_waits_or_retires_borrower(endpoint, monkeypatch):
    e = endpoint
    violations = observe_retirement(e, monkeypatch)
    called = []

    def callback():
        called.append(True)
        with no_self_wait():
            try:
                e.sdk.close()
            except Exception:
                pass  # Explicit busy is permitted; exact new error text is not fixed.
        assert not violations
        assert not e.leases[0]._closed and e.inventory.status()["reserved"] == 4
        assert e.core._foreground_active > 0

    e.execute_hook = callback
    stream = e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True)
    try:
        with no_self_wait():
            try:
                list(stream)
            except Exception:
                pass  # Accepted deferred shutdown may stop this response safely.
    finally:
        stream.close()
    assert called == [True] and not violations
    assert e.leases[0]._closed and e.core._foreground_active == e.core._online_active == 0
    e.sdk.close()  # Complete an explicitly busy/deferred close after retirement.


def test_same_thread_preparation_authority_callback_close_preserves_original_error(endpoint, monkeypatch):
    e = endpoint
    violations = observe_retirement(e, monkeypatch, preparation=True)
    original = ValueError("original authority callback failure")
    called = []

    def callback():
        called.append(True)
        assert e.core._preparation_active > 0
        with no_self_wait():
            try:
                e.sdk.close()
            except Exception:
                pass
        assert not violations
        raise original

    e.h.authority_callback = callback
    try:
        with pytest.raises(ValueError) as caught:
            with no_self_wait():
                e.sdk.preprocess(count=4)
        assert caught.value is original
    finally:
        e.h.authority_callback = None
    assert called == [True] and not violations
    assert e.core._foreground_active == e.core._preparation_active == 0
    e.sdk.close()


def test_concurrent_close_and_preparation_do_not_release_blocked_foreground_lease(endpoint, monkeypatch):
    e = endpoint
    violations = observe_retirement(e, monkeypatch)
    entered, release, closing = threading.Event(), threading.Event(), threading.Event()
    foreground_errors, preparation_errors, preparation_requests, coordination_errors = [], [], [], []
    transport = e.core.http._transport
    handle = transport.handle_request
    preparer = None
    set_attribute = RuntimeClient.__setattr__

    def announce_close(value, name, content):
        # Observe the REAL close's admission write, never set a private flag as
        # a substitute for entering close. Busy rejection also announces below.
        set_attribute(value, name, content)
        if value is e.core and name == "_closing" and content:
            closing.set()

    monkeypatch.setattr(RuntimeClient, "__setattr__", announce_close)

    def audit(request):
        if threading.current_thread() is preparer:
            preparation_requests.append(request)
        return handle(request)

    monkeypatch.setattr(transport, "handle_request", audit)

    def callback():
        entered.set()
        assert release.wait(4), "test did not release foreground callback"

    e.execute_hook = callback

    def foreground():
        try:
            e.sdk.responses.create(input=PRIVATE, max_output_tokens=2)
        except BaseException as error:
            foreground_errors.append(error)

    def preparation():
        try:
            assert closing.wait(4), "close did not stop admission or return busy"
            try:
                e.sdk.preprocess(count=4)
            except Exception as error:
                preparation_errors.append(error)
            assert not violations
            assert not e.leases[0]._closed and e.inventory.status()["reserved"] == 4
            assert e.core._foreground_active > 0
        except BaseException as error:
            coordination_errors.append(error)
        finally:
            release.set()

    worker = threading.Thread(target=foreground, name="foreground-regression")
    preparer = threading.Thread(target=preparation, name="preparation-regression")
    worker.start()
    try:
        assert entered.wait(3)
        preparer.start()
        try:
            # Keep potentially blocking close on the interruptible main thread.
            # Real closing-admission write releases the preparation competitor;
            # a busy/deferred return releases it in finally instead.
            with no_self_wait():
                try:
                    e.sdk.close()
                except Exception:
                    pass
        finally:
            closing.set()
        preparer.join(timeout=4)
    finally:
        release.set()
        closing.set()
        if preparer.ident is not None:
            finish_thread(preparer, release)
        finish_thread(worker, release)
    assert not coordination_errors and not violations and not preparation_requests
    assert len(preparation_errors) == 1
    assert not any(isinstance(error, (AssertionError, WatchdogExpired, MaterialAttempted))
                   for error in foreground_errors)
    assert e.leases[0]._closed and e.core._foreground_active == e.core._online_active == 0
    e.sdk.close()


def test_late_admission_during_real_resource_close_rejects_without_network(endpoint, monkeypatch):
    e = endpoint
    closing, release = threading.Event(), threading.Event()
    failures = []
    bundle = e.bound.bundle
    original = type(bundle).close

    def blocked_close(value):
        if value is bundle:
            closing.set()
            assert release.wait(4), "test did not release resource close"
        return original(value)

    monkeypatch.setattr(type(bundle), "close", blocked_close)

    def closer():
        try:
            e.sdk.close()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=closer, name="closing-regression")
    thread.start()
    try:
        assert closing.wait(3)
        before = len(e.h.http)
        with no_self_wait():
            with e.sdk.responses.create(input=PRIVATE, max_output_tokens=2, stream=True) as stream:
                with pytest.raises(Exception):
                    next(stream)
        assert len(e.h.http) == before
    finally:
        finish_thread(thread, release)
    assert not failures and e.core._foreground_active == 0


def test_legacy_idle_prepare_valid_and_foreground_prepare_rejected_before_controls(legacy):
    assert legacy.sdk.preprocess(count=1)["generated"] == 1
    with legacy.sdk.responses.create(input="ab", max_output_tokens=1, stream=True) as stream:
        assert next(stream).type == "response.created"
        before = len(legacy.requests)
        with pytest.raises(ModelError):
            legacy.sdk.preprocess(count=1)
        assert len(legacy.requests) == before
    assert not legacy.foreground_at_generation


@pytest.fixture
def proprietary(harness, monkeypatch):
    """Real proprietary provider/session; tiny local-test wire replies only."""
    h = harness
    from pllm.runtime import _native_support
    # Select the EXISTING Python compact codec. Keep real wire/schema validation
    # and all native factories trapped; no native codec call or provider bypass.
    monkeypatch.setattr(_native_support, "extension", lambda: None)
    privacy = {**h.bundle.privacy, "mode": "proprietary", "protocol": "blinded_ole_w4a4"}
    bundle = replace(h.bundle, privacy=privacy, arrays={})
    observation = SimpleNamespace(requests=[], material=[], sdk=None, claimed=[])
    metadata = {"client_runtime": "blinded_ole_transformer_v1", "privacy_mode": "proprietary",
                "privacy_protocol": "blinded_ole_w4a4"}

    def handler(request):
        observation.requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == f"/v1/runtime/models/{MODEL}":
            return httpx.Response(200, json={"id": MODEL, "metadata": metadata,
                "client_bundle": {"schema": 2, "sha256": "e" * 64, "size": 1,
                                  "etag": '"' + "e" * 64 + '"'}})
        if request.method == "POST" and path == "/v1/runtime/sessions":
            ordinal = sum(r.method == "POST" and r.url.path == path for r in observation.requests)
            return httpx.Response(200, json={"id": f"proprietary-session-{ordinal}",
                                          "response_id": f"proprietary-response-{ordinal}"})
        if request.method == "POST" and path.endswith("/correlations/proprietary/local-test"):
            core = observation.sdk._core
            observation.material.append(core._foreground_active)
            if core._foreground_active:
                raise MaterialAttempted("foreground reached proprietary fresh-material endpoint")
            value = json.loads(request.content)
            assert value["count"] == 1
            stage = bundle.stages[value["stage_id"]]
            # Ordinary blinded_correlation_from_wire schema, no fake provider or
            # crypto invocation. Zero data is only a lifecycle fixture.
            item = {"id": "idle-" + stage.id, "stage_id": stage.id, "owner_id": value["owner_id"],
                    "ring": stage.ring, "modulus": int(stage.modulus), "wire_bits": stage.wire_bits,
                    "input_shape": [stage.in_features], "output_shape": [stage.out_features],
                    "mask": bytes(stage.in_features * stage.wire_bits // 8),
                    "blinded_transformed": bytes(stage.out_features * stage.wire_bits // 8)}
            return httpx.Response(200, content=msgpack.packb({"items": [item]}, use_bin_type=True))
        if request.method == "POST" and path.endswith(("/cancel", "/complete")):
            return httpx.Response(200, json={"status": "cancelled"})
        pytest.fail(f"unexpected proprietary control: {request.method} {path}")

    with httpx.Client(base_url="https://proprietary.test", transport=httpx.MockTransport(handler)) as http:
        sdk = OpenAI(base_url=str(http.base_url), api_key="test-key", default_model=MODEL,
                     correlation_mode="local-test", http_client=http)
        core = sdk._core
        observation.sdk = sdk
        monkeypatch.setattr(core, "_load_client_bundle_record", lambda *a, **kw: (bundle, "e" * 64))

        def numeric(body, messages, descriptor, **kwargs):
            session, bound, provider = core._open_transformer_session(MODEL, max_output_tokens=1)
            try:
                assert core._foreground_active > 0
                for stage in STAGES:
                    values = provider.take_many(bound.bundle.stages[stage], 1)
                    observation.claimed.extend(values)
                yield ResponseEvent.from_dict({"type": "response.completed", "sequence_number": 0,
                    "response": Response(id=session["response_id"], model=MODEL,
                                         output=[], status="completed").to_dict()})
            finally:
                core._cancel_transformer_session(session["id"])

        monkeypatch.setattr(core, "_transformer_events", numeric)
        try:
            yield observation
        finally:
            sdk.close()


def test_proprietary_underflow_cannot_reach_fresh_endpoint_in_foreground(proprietary):
    p = proprietary
    with pytest.raises(ModelError):
        p.sdk.responses.create(input=PRIVATE, max_output_tokens=1)
    assert not p.material and not p.claimed
    assert p.sdk._core._foreground_active == 0


def test_proprietary_idle_preparation_then_real_provider_claim_needs_no_fresh_online_rows(proprietary):
    p = proprietary
    result = p.sdk.preprocess(count=1)
    assert result["generated"] == len(STAGES)
    assert p.material == [0] * len(STAGES)
    assert p.sdk.responses.create(input=PRIVATE, max_output_tokens=1).status == "completed"
    assert len(p.claimed) == len(STAGES) and all(value.consumed for value in p.claimed)
    assert p.material == [0] * len(STAGES)
    assert p.sdk._core._foreground_active == p.sdk._core._preparation_active == 0
