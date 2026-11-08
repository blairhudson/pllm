"""Fake native, real original module ledger/token; not native recovery proof.

Integration consumes independently repaired shared rooted-exit custody; no
private substitute or shared-artifact import. NOT_CLOSED stays failing without it.
Collection imports pllm._native and ordinary repository fixtures may create
temporary files. These tests avoid source/model reads and native BACKEND calls;
they do not claim import-free or filesystem-free test collection/execution.
Opaque unreturned constructor resources, arbitrary shared-manager corruption,
physical-byte qualification and cross-thread shared reply counters are excluded.
"""

import asyncio
import gc
from types import SimpleNamespace as NS
import weakref

import pytest
from starlette.requests import Request

from pllm.deployment import resources
from pllm.runtime import delegated_worker as worker
from pllm.runtime import delegated_transport as transport


@pytest.fixture
def custody(monkeypatch):
    capacity = resources._ClientCapacity()
    monkeypatch.setattr(resources, "_capacity", capacity)
    monkeypatch.setattr(capacity, "observe", lambda: NS(headroom_bytes=1 << 40))
    state = NS(capacity=capacity, amount=64 << 20, aborts=0, loads=0,
               failure=None, constructor=None, execute_error=None, refunds=[], refs={})

    class ObservedLedger(dict):
        def pop(self, token, *default):
            charged = token in self
            amount = self.get(token)
            result = super().pop(token, *default)
            if charged and token not in self:
                state.refunds.append((id(token), amount))
            return result

    # Delegate ACTUAL ledger pop, ordinary or shared direct commit; never refund
    # in a fixture or equate backend return with original-token removal.
    capacity._owners = ObservedLedger(capacity._owners)
    monkeypatch.setattr(worker, "reserve_incremental_memory", resources.reserve_incremental_memory)
    monkeypatch.setattr(worker, "request_memory_bytes", lambda *a, **k: state.amount)

    class Native:
        def request_id(self):
            return bytes.fromhex("a" * 64)

        def abort(self):
            assert not capacity.lock._is_owned()
            state.aborts += 1
            if state.failure is not None:
                raise state.failure

        def execute(self, raw):
            if state.execute_error is not None:
                raise state.execute_error
            return b"reply"

        def finish(self):
            return b"receipt"

    def load(*args, **kwargs):
        state.loads += 1
        if state.constructor is not None:
            raise state.constructor
        native = Native()
        state.refs["native"] = weakref.ref(native)
        return native

    source = NS(capacity="capsule", source_lock_digest="c" * 64,
                tokenizer_digest="d" * 64, runtime_contract=lambda *a: b"contract",
                load_transport=load)
    monkeypatch.setattr(worker, "admit_source", lambda _: source)
    experiment = NS(resolve=lambda: NS(client_runtime="compiled_gkm_v1"),
                    pipeline=NS(components={"linear": NS(params={})}),
                    budget=NS(requests=64), configuration_digest=lambda: "b" * 64)

    def app():
        result = worker.create_delegated_worker_app(experiment, "worker_a", "k" * 32)
        return result

    def endpoint(app, suffix):
        return next(r.endpoint for r in app.routes if r.path.endswith(suffix))

    def request(owner, *, action="admit", authorized=True, body=None):
        if body is None:
            body = worker.pack({"schema": "pllm.gkm.admit.v1", "request": "a" * 64,
                "epoch": owner.epoch, "configuration": "b" * 64,
                "source_lock": "c" * 64, "tokenizer": "d" * 64,
                "input_rows": 1, "output_tokens": 1, "publication": b"publication",
                "pin": bytes(32), "start": bytes.fromhex("a" * 64)})
        headers = [(b"content-length", str(len(body)).encode())]
        if authorized:
            headers.append((b"authorization", b"Bearer " + b"k" * 32))

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def connected():
            return False

        result = Request({"type": "http", "method": "POST", "path": "/",
            "headers": headers, "path_params": {"request": "a" * 64, "action": action}}, receive)
        result.is_disconnected = connected
        return result

    async def start():
        result = app()
        owner = result.state.delegated_owner
        ready = await endpoint(result, "/admit")(request(owner))
        assert ready.status_code == 200
        ready.close()
        return result, owner

    state.app, state.endpoint, state.request, state.start = app, endpoint, request, start
    state.source = source
    state.experiment = experiment
    return state


def _original(retirement):
    """Capture identities BEFORE failure; weak refs cannot fake GC custody."""
    manager = retirement.manager
    return NS(manager=weakref.ref(manager), token=manager._token,
              root=weakref.ref(retirement.root) if retirement.root is not None else None,
              backend=weakref.ref(manager._manager),
              capacity=manager._capacity, amount=manager._amount,
              retirement_id=id(retirement))


def _same(retirement, original):
    assert id(retirement) == original.retirement_id
    assert retirement.manager is original.manager()
    assert retirement.manager._token is original.token
    assert retirement.manager._capacity is original.capacity
    assert retirement.manager._manager is original.backend()
    assert retirement.manager._amount == original.amount
    assert retirement.root is (original.root() if original.root is not None else None)
    assert original.root is None or original.root() is not None


def _charged(h, retirement, original):
    _same(retirement, original)
    manager = original.manager()
    assert type(manager) is resources._MemoryReservation
    assert original.capacity is resources._capacity is h.capacity
    assert original.capacity._owners == {original.token: original.amount}
    assert original.token.retained is manager
    assert len(manager._failed) == 1 and manager._failed[0] is retirement
    assert retirement.custody == (retirement,)


def _refunded(h, retirement, original):
    _same(retirement, original)
    assert not original.capacity._owners
    assert h.refunds == [(id(original.token), original.amount)]


async def _blocked_drain(h, owner, original):
    attempts = h.aborts
    refunds = list(h.refunds)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
            await owner._drain_retirement()
        _charged(h, owner.retirement, original)
    assert h.aborts == attempts and h.refunds == refunds


async def _custody_http(app, path, raw, *, final_body=None, forbid_receive=False):
    delivered = False
    statuses, bodies = [], []

    async def receive():
        nonlocal delivered
        if forbid_receive:
            pytest.fail("failed retirement ingress received a body")
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": raw, "more_body": False}
        await asyncio.Future()

    async def send(message):
        if message["type"] == "http.response.start":
            statuses.append(message["status"])
        elif message["type"] == "http.response.body":
            bodies.append(message.get("body", b""))
            assert not message.get("more_body", False)
            if final_body is not None:
                final_body()

    scope = {"type": "http", "http_version": "1.1", "method": "POST",
        "scheme": "http", "path": path, "root_path": "", "query_string": b"",
        "server": ("testserver", 80), "client": ("testclient", 1),
        "headers": [(b"content-length", str(len(raw)).encode()),
                    (b"authorization", b"Bearer " + b"k" * 32)]}
    # No lifespan: expiry/shutdown must not supply the retirement witness.
    await app(scope, receive, send)
    assert len(statuses) == len(bodies) == 1
    return NS(status=statuses[0], body=bodies[0], delivered=delivered)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_http_backend_failure_after_cancel_send_keeps_original_custody(custody, monkeypatch, failure_type):
    h = custody

    async def run():
        app = h.app()
        owner = app.state.delegated_owner
        raw = await h.request(owner).body()
        fresh = worker.unpack(raw)
        fresh["request"] = "b" * 64
        fresh["start"] = bytes.fromhex("b" * 64)
        fresh_raw = worker.pack(fresh)
        ready = await _custody_http(app, "/v1/delegation/admit", raw)
        assert ready.status == 200 and ready.delivered and ready.body
        retirement = owner.retirement
        original = _original(retirement)
        native = retirement.native
        _charged(h, retirement, original)
        backend_type = type(original.backend())
        real_exit = backend_type.__exit__
        failure = failure_type("HTTP backend exit")
        exits, during_send = [], []

        def fail_exit(backend, *exc):
            if backend is original.backend():
                assert owner.lock.locked()
                assert not h.capacity.lock._is_owned()
                _charged(h, retirement, original)
                assert retirement.exit_seen and retirement.root._references == 0
                exits.append(id(original.token))
                raise failure
            return real_exit(backend, *exc)

        monkeypatch.setattr(backend_type, "__exit__", fail_exit)

        def final_body():
            _charged(h, retirement, original)
            assert retirement.terminal and retirement.abort_ok and retirement.native is None
            assert retirement.root_closed and retirement.root._closed
            assert retirement.root._references == 1 and not retirement.exit_seen
            assert retirement.root._manager is retirement
            assert not retirement.done and exits == [] and h.aborts == 1
            during_send.append(True)

        cancelled = await _custody_http(app, f"/v1/delegation/{'a' * 64}/cancel", b"",
                                       final_body=final_body)
        assert cancelled.status == 200 and cancelled.delivered
        assert worker.unpack(cancelled.body)["action"] == "cancel"
        assert during_send == [True]
        _charged(h, retirement, original)
        assert retirement.exit_seen and retirement.root._references == 0
        assert retirement.root._manager is None
        assert retirement.abort_ok and retirement.native is None and not retirement.done
        assert owner.active is owner.request is owner.reservation is None
        assert exits == [] and h.aborts == 1 and not h.refunds

        def trap(*args, **kwargs):
            pytest.fail("failed retirement ingress reached body/source/load/native")

        monkeypatch.setattr(worker, "_body", trap)
        monkeypatch.setattr(h.source, "runtime_contract", trap)
        monkeypatch.setattr(h.source, "load_transport", trap)
        monkeypatch.setattr(native, "execute", trap)
        for path, body in [("/v1/delegation/admit", fresh_raw),
                           ("/v1/delegation/admit", fresh_raw),
                           (f"/v1/delegation/{'a' * 64}/step", b"x")]:
            blocked = await _custody_http(app, path, body, forbid_receive=True)
            assert blocked.status == 409 and blocked.body == b"" and not blocked.delivered
            _charged(h, owner.retirement, original)
            assert owner.retirement is retirement and owner.pending_retirement
            assert retirement.busy and not retirement.done
            assert exits == [id(original.token)]
            assert h.loads == h.aborts == 1 and owner.seen == {"a" * 64}
            assert not h.refunds

    asyncio.run(run())


def test_http_failed_native_abort_fences_healthy_repeated_ingress(custody, monkeypatch):
    h = custody

    async def run():
        app = h.app()
        owner = app.state.delegated_owner
        raw = await h.request(owner).body()
        fresh = worker.unpack(raw)
        fresh["request"] = "b" * 64
        fresh["start"] = bytes.fromhex("b" * 64)
        fresh_raw = worker.pack(fresh)
        ready = await _custody_http(app, "/v1/delegation/admit", raw)
        assert ready.status == 200 and ready.delivered and ready.body
        retirement = owner.retirement
        original = _original(retirement)
        native = retirement.native
        _charged(h, retirement, original)
        backend_type = type(original.backend())
        real_exit = backend_type.__exit__
        exits, errors = [], []
        failure = RuntimeError("HTTP native abort")
        clear = worker._clear_failure_frames

        def observe_exit(backend, *exc):
            if backend is original.backend():
                exits.append(id(original.token))
                pytest.fail("failed native abort reached backend exit")
            return real_exit(backend, *exc)

        def observe_error(error):
            errors.append(error)
            return clear(error)

        monkeypatch.setattr(backend_type, "__exit__", observe_exit)
        monkeypatch.setattr(worker, "_clear_failure_frames", observe_error)
        h.failure = failure
        cancelled = await _custody_http(app, f"/v1/delegation/{'a' * 64}/cancel", b"")
        assert cancelled.status == 400 and cancelled.delivered and cancelled.body == b""
        assert errors == [failure] and errors[0] is h.failure
        _charged(h, retirement, original)
        assert retirement.terminal and retirement.started and not retirement.abort_ok
        assert retirement.native is native is h.refs["native"]()
        assert not retirement.root_closed and not retirement.root._closed
        assert not retirement.exit_seen and retirement.root._references == 1
        assert retirement.root._manager is retirement and owner.reservation is retirement.root
        assert owner.active is owner.request is None
        assert not retirement.done and exits == [] and h.aborts == 1
        h.failure = None  # Native is healthy now; ingress still must not retry.

        def trap(*args, **kwargs):
            pytest.fail("failed retirement ingress reached body/source/load/native")

        monkeypatch.setattr(worker, "_body", trap)
        monkeypatch.setattr(h.source, "runtime_contract", trap)
        monkeypatch.setattr(h.source, "load_transport", trap)
        monkeypatch.setattr(native, "execute", trap)
        for path, body in [("/v1/delegation/admit", fresh_raw),
                           ("/v1/delegation/admit", fresh_raw),
                           (f"/v1/delegation/{'a' * 64}/step", b"x")]:
            blocked = await _custody_http(app, path, body, forbid_receive=True)
            assert blocked.status == 409 and blocked.body == b"" and not blocked.delivered
            _charged(h, owner.retirement, original)
            assert owner.retirement is retirement and owner.pending_retirement
            assert retirement.native is native and not retirement.abort_ok
            assert not retirement.root_closed and not retirement.exit_seen and not retirement.done
            assert retirement.root._references == 1 and errors == [failure]
            assert exits == [] and h.loads == h.aborts == 1
            assert owner.seen == {"a" * 64} and not h.refunds

    asyncio.run(run())


def test_failed_abort_survives_owner_app_gc_without_handoff(custody):
    h = custody

    async def run():
        app, owner = await h.start()
        retirement = owner.retirement
        original = _original(retirement)
        manager_ref = weakref.ref(retirement.manager)
        native_ref = h.refs["native"]
        owner_ref, app_ref = weakref.ref(owner), weakref.ref(app)
        h.failure = ValueError("abort")
        with pytest.raises(ValueError):
            owner.retire()
        h.failure.__traceback__ = None
        del retirement, owner, app
        gc.collect()
        assert owner_ref() is app_ref() is None
        assert native_ref() is not None and manager_ref() is not None
        manager = original.token.retained
        assert manager is original.manager()
        retirement = manager._failed[0]
        _charged(h, retirement, original)
        assert retirement.terminal and not retirement.abort_ok and not retirement.root_closed
        with pytest.raises(RuntimeError, match="not refundable"):
            manager.drain_failed()
        assert h.aborts == 1
        h.failure = None
        # Owner is gone: even its original loop/lock cannot authorize a retry.
        async with retirement.lock:
            with pytest.raises(TypeError):
                retirement.retire(retry=True)
            retirement.retire()
        _charged(h, retirement, original)
        assert native_ref() is not None and h.aborts == 1 and not h.refunds

    asyncio.run(run())


@pytest.mark.parametrize("route", ["/admit", "/{request}/{action}"])
def test_pending_gate_precedes_expiry_body_source_import_and_native(custody, monkeypatch, route):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        h.failure = ValueError("abort")
        with pytest.raises(ValueError):
            owner.retire()

        def trap(*a, **k):
            pytest.fail("pending request reached body/source/import/native/expiry")

        monkeypatch.setattr(owner, "expire", trap)
        monkeypatch.setattr(worker, "_body", trap)
        monkeypatch.setattr(h.source, "runtime_contract", trap)
        monkeypatch.setattr(h.source, "load_transport", trap)
        endpoint = h.endpoint(app, route)
        for authorized, status in [(False, 401), (True, 409)]:
            result = await endpoint(h.request(owner, action="step", authorized=authorized, body=b""))
            assert result.status_code == status
        assert owner.seen == {"a" * 64} and h.aborts == 1
        _charged(h, owner.retirement, original)
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())

@pytest.mark.parametrize("published", [False, True])
def test_prearm_rejection_never_imports_and_refunds_empty_original(custody, monkeypatch, published):
    h = custody
    original = resources._MemoryReservation.retain_failed
    originals = []

    def reject(manager, owners):
        originals.append(_original(owners[0]))
        if published:
            original(manager, owners)
        raise ValueError("prearm injection")

    monkeypatch.setattr(resources._MemoryReservation, "retain_failed", reject)

    async def run():
        app = h.app()
        owner = app.state.delegated_owner
        result = await h.endpoint(app, "/admit")(h.request(owner))
        assert result.status_code == 400 and h.loads == h.aborts == 0
        assert owner.retirement.done and not h.capacity._owners
        _refunded(h, owner.retirement, originals[0])
        assert owner.seen == {"a" * 64}

    asyncio.run(run())

@pytest.mark.parametrize("stalled,cancel", [(False, False), (True, False), (True, True)])
def test_successful_retirement_waits_for_reply_and_never_repeats_abort(custody, stalled, cancel):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        reply = await h.endpoint(app, "/{request}/{action}")(
            h.request(owner, action="step", body=b"x"))
        entered, release = asyncio.Event(), asyncio.Event()

        async def send(message):
            if message["type"] == "http.response.body":
                entered.set()
                await release.wait()

        task = None
        if stalled:
            task = asyncio.create_task(reply({"type": "http"}, None, send))
            await entered.wait()
        owner.retire()
        retirement = owner.retirement
        _charged(h, retirement, original)
        with pytest.raises(RuntimeError, match="incomplete"):
            await owner._drain_retirement()
        assert retirement.abort_ok and retirement.root_closed and not retirement.exit_seen
        assert h.aborts == 1
        _charged(h, retirement, original)
        owner.retire()
        with pytest.raises(RuntimeError, match="incomplete"):
            await owner._drain_retirement()
        assert h.aborts == 1
        reply.close()
        if stalled:
            assert reply.body == b"reply" and h.capacity._owners
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                await task
        _charged(h, retirement, original)
        assert retirement.exit_seen and not retirement.done
        await owner._drain_retirement()
        assert retirement.done and not h.capacity._owners
        await owner._drain_retirement()
        assert h.aborts == 1
        _refunded(h, retirement, original)

    asyncio.run(run())

@pytest.mark.parametrize("primary", [ValueError("execute"), asyncio.CancelledError("execute")])
def test_primary_execution_or_cancellation_identity_survives_abort_failure(custody, primary):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        h.failure = RuntimeError("cleanup")
        h.execute_error = primary
        endpoint = h.endpoint(app, "/{request}/{action}")
        if isinstance(primary, Exception):
            observed = []
            original_clear = worker._clear_failure_frames
            # Observe identity without replacing cleanup behavior.
            def clear(error):
                observed.append(error)
                original_clear(error)
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(worker, "_clear_failure_frames", clear)
                assert (await endpoint(h.request(owner, action="step", body=b"x"))).status_code == 400
            assert observed[0] is primary
        else:
            with pytest.raises(asyncio.CancelledError) as caught:
                await endpoint(h.request(owner, action="step", body=b"x"))
            assert caught.value is primary
        _charged(h, owner.retirement, original)
        assert owner.active is owner.request is None
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())

def test_foreign_loop_refuses_drain_and_foreign_final_release_stays_charged(custody):
    h = custody
    loop = asyncio.new_event_loop()

    async def prepare():
        app, owner = await h.start()
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        owner.retire()
        return owner, reply

    try:
        owner, reply = loop.run_until_complete(prepare())
        original = _original(owner.retirement)

        async def foreign():
            with pytest.raises(RuntimeError, match="owning loop"):
                await owner._drain_retirement()
            reply.close()
            assert owner.retirement.exit_seen and not owner.retirement.done
            _charged(h, owner.retirement, original)
            with pytest.raises(RuntimeError, match="not refundable"):
                owner.retirement.manager.drain_failed()

        asyncio.run(foreign())
        loop.run_until_complete(owner._drain_retirement())
        assert h.aborts == 1
        _refunded(h, owner.retirement, original)
    finally:
        loop.close()


@pytest.mark.parametrize("route", ["/admit", "/{request}/{action}"])
@pytest.mark.parametrize("failure_kind", ["abort", "backend"])
def test_ingress_never_retries_failed_or_unknown_retirement(custody, monkeypatch, route, failure_kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        exits = []
        if failure_kind == "abort":
            h.failure = ValueError("abort")
        else:
            backend_type = type(owner.retirement.manager._manager)

            def fail_exit(backend, *exc):
                exits.append(backend)
                raise ValueError("backend ambiguity")

            monkeypatch.setattr(backend_type, "__exit__", fail_exit)
        with pytest.raises(ValueError):
            owner.retire()
        async with owner.lock:
            ingress = asyncio.create_task(h.endpoint(app, route)(h.request(owner, action="step", body=b"x")))
            drain = asyncio.create_task(owner._drain_retirement())
            await asyncio.sleep(0)
            assert not ingress.done() and not drain.done() and h.aborts == 1
            h.failure = None
        assert (await ingress).status_code == 409
        with pytest.raises(RuntimeError, match="incomplete"):
            await drain
        assert h.aborts == 1 and not h.refunds
        assert len(exits) == (failure_kind == "backend")
        for _ in range(2):
            assert (await h.endpoint(app, route)(h.request(owner, action="step", body=b"x"))).status_code == 409
        assert len(exits) == (failure_kind == "backend") and h.aborts == 1
        _charged(h, owner.retirement, original)

    asyncio.run(run())

@pytest.mark.parametrize("endpoint", ["admission", "operation"])
@pytest.mark.parametrize("kind", ["error", "cancel"])
@pytest.mark.parametrize("diagnostic", ["ordinary", "interrupt", "exit", "metadata"])
def test_diagnostic_failure_preserves_primary_and_fences_owner(custody, monkeypatch, endpoint, kind, diagnostic):
    h = custody

    async def run():
        base = ValueError if kind == "error" else asyncio.CancelledError

        class Primary(base):
            def __getattribute__(self, name):
                if diagnostic == "metadata" and name == "__traceback__":
                    raise RuntimeError("throwing diagnostic metadata")
                return BaseException.__getattribute__(self, name)

        primary = Primary("known primary")
        if endpoint == "operation":
            app, owner = await h.start()
            original = _original(owner.retirement)
            h.execute_error = primary
            handler = h.endpoint(app, "/{request}/{action}")
            request = h.request(owner, action="step", body=b"x")
        else:
            app = h.app()
            owner = app.state.delegated_owner
            originals = []
            load = h.source.load_transport

            def fail_request_id(*args, **kwargs):
                native = load(*args, **kwargs)
                originals.append(_original(owner.retirement))

                def request_id():
                    raise primary

                native.request_id = request_id
                return native

            monkeypatch.setattr(h.source, "load_transport", fail_request_id)
            handler = h.endpoint(app, "/admit")
            request = h.request(owner)

        original_clear = worker._clear_failure_frames
        reentry = []
        fences = []
        diagnostic_errors = {"ordinary": RuntimeError("clearing"),
                             "interrupt": KeyboardInterrupt("clearing"),
                             "exit": SystemExit("clearing")}

        def clear(error):
            fences.append((error is primary, owner.active is None, owner.request is None,
                           owner.retirement.terminal, owner.retirement.uncertain))
            reentry.append(asyncio.create_task(h.endpoint(app, "/admit")(h.request(owner))))
            if diagnostic == "metadata":
                original_clear(error)
            else:
                raise diagnostic_errors[diagnostic]

        monkeypatch.setattr(worker, "_clear_failure_frames", clear)
        selected = None
        try:
            response = await handler(request)
        except BaseException as error:
            selected = error
        if kind == "cancel":
            assert selected is primary
        else:
            assert selected is None and response.status_code == 400
        if endpoint == "admission":
            original = originals[0]
        assert fences == [(True, True, True, True, True)]
        assert reentry and (await reentry[0]).status_code == 409
        assert owner.retirement.uncertain and owner.retirement.native is not None
        assert not owner.retirement.started and h.aborts == 0
        assert not hasattr(primary, "__notes__")
        _charged(h, owner.retirement, original)
        with pytest.raises(RuntimeError, match="not refundable"):
            owner.retirement.close()
        with pytest.raises(RuntimeError, match="not refundable"):
            owner.retirement._finalize()
        with pytest.raises(RuntimeError):
            original.manager().drain_failed()
        await _blocked_drain(h, owner, original)
        with pytest.raises(RuntimeError, match="incomplete"):
            async with app.router.lifespan_context(app):
                pass
        assert h.aborts == 0 and not h.refunds
        _charged(h, owner.retirement, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
@pytest.mark.parametrize("cleanup_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("path", ["endpoint", "lifespan"])
def test_first_established_primary_wins_cleanup_baseexception(custody, kind, cleanup_type, path):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        primary = (ValueError("primary") if kind == "error" else asyncio.CancelledError("primary"))
        cleanup = cleanup_type("cleanup")
        h.failure = cleanup
        if path == "endpoint":
            h.execute_error = primary
            selected = None
            try:
                response = await h.endpoint(app, "/{request}/{action}")(
                    h.request(owner, action="step", body=b"x"))
            except BaseException as error:
                selected = error
            if kind == "error":
                assert selected is None and response.status_code == 400
            else:
                assert selected is primary
        else:
            with pytest.raises(type(primary)) as caught:
                async with app.router.lifespan_context(app):
                    raise primary
            assert caught.value is primary
        assert not hasattr(primary, "__notes__")
        assert not hasattr(cleanup, "__notes__")
        assert owner.active is owner.request is None and owner.retirement.terminal
        assert h.aborts == 1
        _charged(h, owner.retirement, original)
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
def test_send_diagnostic_uncertainty_blocks_successful_abort_refund(custody, monkeypatch, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        reply = await h.endpoint(app, "/{request}/{action}")(
            h.request(owner, action="cancel", body=b""))
        assert owner.retirement.abort_ok and owner.retirement.root_closed
        primary = ValueError("send") if kind == "error" else asyncio.CancelledError("send")
        clearing = SystemExit("diagnostic")

        def fail_clear(error):
            raise clearing

        async def send(message):
            if message["type"] == "http.response.body":
                raise primary

        monkeypatch.setattr(worker, "_clear_failure_frames", fail_clear)
        with pytest.raises(type(primary)) as caught:
            await reply({"type": "http"}, None, send)
        assert caught.value is primary and not hasattr(primary, "__notes__")
        retirement = owner.retirement
        assert retirement.exit_seen and retirement.root._references == 0
        assert retirement.uncertain and retirement.native is None
        assert reply.body == b"" and h.aborts == 1
        _charged(h, retirement, original)
        with pytest.raises(RuntimeError, match="not refundable"):
            retirement.close()
        with pytest.raises(RuntimeError, match="not refundable"):
            retirement._finalize()
        assert (await h.endpoint(app, "/admit")(h.request(owner))).status_code == 409
        await _blocked_drain(h, owner, original)
        assert h.aborts == 1 and not h.refunds

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
def test_constructor_error_without_return_has_only_empty_refund_proof(custody, monkeypatch, kind):
    h = custody

    async def run():
        h.constructor = ValueError("constructor") if kind == "error" else asyncio.CancelledError("constructor")
        app = h.app()
        owner = app.state.delegated_owner
        originals = []
        load = h.source.load_transport

        def failing_load(*args, **kwargs):
            originals.append(_original(owner.retirement))
            return load(*args, **kwargs)

        monkeypatch.setattr(h.source, "load_transport", failing_load)
        if kind == "cancel":
            with pytest.raises(asyncio.CancelledError) as caught:
                await h.endpoint(app, "/admit")(h.request(owner))
            assert caught.value is h.constructor
        else:
            assert (await h.endpoint(app, "/admit")(h.request(owner))).status_code == 400
        assert h.loads == 1 and h.aborts == 0
        assert owner.retirement.done and not h.capacity._owners and len(h.refunds) == 1
        _refunded(h, owner.retirement, originals[0])

    asyncio.run(run())


def test_expiry_task_failure_still_runs_locked_shutdown_without_abort_retry(custody, monkeypatch):
    h = custody

    async def run():
        app, owner = await h.start()
        original_charge = _original(owner.retirement)
        owner.ttl = 0.001
        owner.deadline = 0
        h.failure = ValueError("expiry abort")
        original = owner.retire
        locks = []

        def retire():
            locks.append(owner.lock.locked())
            return original()

        monkeypatch.setattr(owner, "retire", retire)
        with pytest.raises(ValueError, match="expiry abort") as caught:
            async with app.router.lifespan_context(app):
                # Wait for the terminal fence, not a guessed long sleep.
                async with asyncio.timeout(1.0):
                    while not owner.pending_retirement:
                        await asyncio.sleep(0.001)
        assert owner.closed and locks == [True, True] and h.aborts == 1
        assert caught.value is h.failure
        _charged(h, owner.retirement, original_charge)
        h.failure = None
        await _blocked_drain(h, owner, original_charge)

    asyncio.run(run())


def test_shutdown_abort_failure_stays_charged_without_physical_retry(custody):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        h.failure = ValueError("shutdown abort")
        with pytest.raises(ValueError, match="shutdown abort") as caught:
            async with app.router.lifespan_context(app):
                pass
        assert owner.closed and h.aborts == 1
        assert caught.value is h.failure
        _charged(h, owner.retirement, original)
        h.failure = None
        await _blocked_drain(h, owner, original)
        assert h.aborts == 1

    asyncio.run(run())


def test_worker_adapter_does_not_change_sticky_shared_transport_close(monkeypatch):
    import threading

    peers = transport._Peers.__new__(transport._Peers)
    peers.io = None
    peers._close_gate = threading.Lock()
    peers._close_succeeded = peers.closed = False
    attempts = []

    def fail():
        attempts.append(True)
        raise ValueError("shared close")

    monkeypatch.setattr(peers, "_close_owned", fail)
    with pytest.raises(ValueError, match="shared close"):
        peers.close()
    with pytest.raises(transport.ProtocolError, match="previously failed"):
        peers.close()
    assert attempts == [True] and not peers._close_succeeded


def test_prearm_foreign_custody_is_never_overwritten_or_removed(custody, monkeypatch):
    h = custody
    original = resources._MemoryReservation.retain_failed
    closes = []
    foreign = NS(close=lambda: closes.append(True))
    published = []

    def inject(manager, owners):
        other = (foreign,)
        original(manager, other)
        published.append(other)
        raise ValueError("foreign publication")

    monkeypatch.setattr(resources._MemoryReservation, "retain_failed", inject)

    async def run():
        app = h.app()
        owner = app.state.delegated_owner
        assert (await h.endpoint(app, "/admit")(h.request(owner))).status_code == 400
        manager = owner.retirement.manager
        assert h.loads == h.aborts == 0 and not closes
        assert h.capacity._owners == {manager._token: h.amount}
        assert manager._token.retained is manager and manager._failed is published[0]
        assert owner.pending_retirement
        # Only that foreign owner's own explicit drain can authorize its refund.
        manager.drain_failed()
        assert closes == [True] and not h.capacity._owners

    asyncio.run(run())


@pytest.mark.parametrize("physical_close", [False, True])
def test_NOT_CLOSED_ambiguous_exit_original_custody_survives_owner_app_gc(custody, monkeypatch, physical_close):
    """Integration FAILS without shared rooted exit; no xfail/private fake root."""
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        app_ref, owner_ref = weakref.ref(app), weakref.ref(owner)
        assert not owner.pending_retirement
        real_exit = type(original.backend()).__exit__
        attempts = []

        def fail_exit(backend, *exc):
            if backend is original.backend():
                manager = original.manager()
                retirement = manager._failed[0]
                _charged(h, retirement, original)
                assert retirement.terminal and not retirement.done
                assert owner_ref().pending_retirement
                assert not manager._capacity.lock._is_owned()
                attempts.append(True)
                if physical_close:
                    real_exit(backend, *exc)
                    assert backend.gen.gi_frame is None
                    _charged(h, retirement, original)
                    assert not h.refunds
                raise ValueError("backend exit ambiguous")
            return real_exit(backend, *exc)

        monkeypatch.setattr(type(original.backend()), "__exit__", fail_exit)
        with pytest.raises(ValueError, match="backend exit ambiguous"):
            owner.retire()
        assert owner.retirement.abort_ok and owner.retirement.native is None
        assert owner.retirement.root_closed and owner.retirement.exit_seen
        assert owner.pending_retirement and not owner.retirement.done
        _charged(h, owner.retirement, original)
        with pytest.raises(RuntimeError, match="ambiguous"):
            await owner._drain_retirement()
        with pytest.raises(RuntimeError, match="ambiguous"):
            async with app.router.lifespan_context(app):
                pass
        assert attempts == [True] and h.aborts == 1 and not h.refunds
        del app, owner
        gc.collect()
        assert app_ref() is owner_ref() is None
        # Only original token/capacity and WEAK manager/root references survive.
        # Original real backend may have physically closed, yet failure is NOT
        # a refund witness. Consume shared sticky custody, never rebuild it.
        assert original.capacity._owners == {original.token: original.amount}
        assert original.manager() is not None and original.root() is not None
        assert original.token.retained is original.manager()
        retirement = original.manager()._failed[0]
        assert retirement.terminal and not retirement.done
        _charged(h, retirement, original)
        for _ in range(2):
            with pytest.raises(resources.NetworkError, match="CLIENT_FAILED_RETIREMENT_NOT_DRAINABLE"):
                original.manager().drain_failed()
        assert attempts == [True] and not h.refunds and h.aborts == 1
        _charged(h, retirement, original)

    asyncio.run(run())


def test_prearmed_original_root_is_published_before_native_acquisition(custody, monkeypatch):
    h = custody

    async def run():
        app = h.app()
        owner = app.state.delegated_owner
        original = h.source.load_transport
        snapshots = []

        def load(*args, **kwargs):
            retirement = owner.retirement
            original_charge = _original(retirement)
            _charged(h, retirement, original_charge)
            assert not hasattr(retirement, "__dict__")
            assert retirement.lock is owner.lock
            assert retirement.loop is asyncio.get_running_loop()
            assert retirement.root is owner.reservation
            assert retirement.root._manager is retirement
            assert retirement.root._references == 1
            assert retirement.native is None
            assert retirement.manager._failed is retirement.custody
            snapshots.append(original_charge)
            native = original(*args, **kwargs)
            native_ref = weakref.ref(native)

            def request_id():
                assert retirement.native is native_ref()
                return bytes.fromhex("a" * 64)

            native.request_id = request_id
            return native

        monkeypatch.setattr(h.source, "load_transport", load)
        response = await h.endpoint(app, "/admit")(h.request(owner))
        assert response.status_code == 200
        response.close()
        owner.retire()
        _refunded(h, owner.retirement, snapshots[0])
        assert h.aborts == 1 and owner.retirement.done

    asyncio.run(run())


def test_direct_refund_only_drain_never_aborts_and_owner_observes_completion(custody):
    h = custody
    loop = asyncio.new_event_loop()

    async def prepare():
        app, owner = await h.start()
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        owner.retire()
        return owner, reply

    try:
        owner, reply = loop.run_until_complete(prepare())
        original = _original(owner.retirement)
        reply.close()  # No running loop: completion recorded, refund deferred.
        _charged(h, owner.retirement, original)

        async def finish():
            owner.retirement.manager.drain_failed()
            assert not h.capacity._owners and h.aborts == 1
            await owner._drain_retirement()
            assert owner.retirement.done and not owner.pending_retirement
            assert h.aborts == 1
            _refunded(h, owner.retirement, original)

        loop.run_until_complete(finish())
    finally:
        loop.close()


def test_batch_reply_slot_and_full_charge_survive_failed_abort(custody, monkeypatch):
    h = custody

    async def run():
        import json

        batch = {"execution": worker.BATCH_EXECUTION,
            "schema_version": "pllm.gkm.source-stage-batch-contract.v1",
            "batch_plan": "e" * 64,
            "resources": {"max_aggregate_bytes": 5, "retained_frame_bytes": 5,
                "transport_scratch_bytes": 5, "extra_native_owner_bytes": 0,
                "extra_wire_bytes": 0},
            "compiler_encoded_input_peak_bytes": 8,
            "client_payload_bytes": 24, "worker_payload_bytes": 16,
            "descriptors": [{"batch_index": 0, "phase_index": 0, "operation_index": 0,
                "first_claim": 0, "rows": 1, "columns": 1, "output_width": 1,
                "kind": "linear", "stage": 0, "request_bytes": [1, 1], "reply_bytes": [5, 5]}]}
        h.experiment.pipeline.components["linear"].params["execution"] = worker.BATCH_EXECUTION
        runtime = {"batch_execution": batch,
                   "decoder": {"execution_payload_estimate_bytes": 16}}
        h.source.execution_payload_estimate_bytes = 16
        h.source.runtime_contract_bytes = json.dumps(runtime).encode()
        h.source.runtime_contract = lambda *a: h.source.runtime_contract_bytes
        pricing = []
        expected_amount = (64 << 20) + 16 + 10 + 16 * (64 << 10)

        def price(source, *, worker=False, role=None):
            amount = transport.request_memory_bytes(source, worker=worker, role=role)
            pricing.append((source is h.source, worker, role, amount))
            return amount

        monkeypatch.setattr(worker, "request_memory_bytes", price)
        app = h.app()
        owner = app.state.delegated_owner
        admission = h.request(owner)
        admission.scope["pllm.batch_http_backend"] = worker.BATCH_HTTP_BACKEND
        ready = await h.endpoint(app, "/admit")(admission)
        assert ready.status_code == 200 and owner.batch_selected
        assert owner.batch == worker.batch_contract({"batch_execution": batch})
        assert pricing == [(True, True, "worker_a", expected_amount)]
        ready.close()
        original = _original(owner.retirement)
        assert original.amount == expected_amount
        request = h.request(owner, action="step", body=b"x")
        request.scope["pllm.batch_http_backend"] = worker.BATCH_HTTP_BACKEND
        reply = await h.endpoint(app, "/{request}/{action}")(request)
        assert reply.status_code == 200 and owner.batch_operation_outstanding
        h.failure = ValueError("abort")
        with pytest.raises(ValueError):
            owner.retire()
        _charged(h, owner.retirement, original)
        assert owner.batch_operation_outstanding
        h.failure = None
        await _blocked_drain(h, owner, original)
        _charged(h, owner.retirement, original)
        assert owner.batch_operation_outstanding and h.aborts == 1
        reply.close()
        assert not owner.batch_operation_outstanding
        _charged(h, owner.retirement, original)
        assert not owner.retirement.exit_seen and not owner.retirement.done
        await _blocked_drain(h, owner, original)
        assert not h.refunds

    asyncio.run(run())


def test_serial_foreign_thread_final_release_records_completion_without_refund(custody):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        owner.retire()
        # Serial release only: no new concurrent cross-thread counter guarantee.
        await asyncio.to_thread(reply.close)
        assert owner.retirement.exit_seen and not owner.retirement.done
        _charged(h, owner.retirement, original)
        await owner._drain_retirement()
        assert owner.retirement.done and not h.capacity._owners
        assert h.aborts == 1
        _refunded(h, owner.retirement, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel", "unsent_gc"])
def test_last_reply_release_never_calls_backend_or_masks_exact_send_primary(custody, monkeypatch, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        owner.retire()  # Successful abort; outstanding reply still owns charge.
        assert h.aborts == 1 and owner.retirement.abort_ok
        _charged(h, owner.retirement, original)
        backend_error = RuntimeError("backend exit")
        primary = ValueError("send") if kind == "error" else asyncio.CancelledError("send")
        calls = []
        real_exit = type(original.backend()).__exit__

        def fail_exit(backend, *exc):
            if backend is original.backend():
                _charged(h, original.manager()._failed[0], original)
                calls.append(id(original.token))
                raise backend_error
            return real_exit(backend, *exc)

        monkeypatch.setattr(type(original.backend()), "__exit__", fail_exit)
        if kind == "unsent_gc":
            ref = weakref.ref(reply)
            del reply
            gc.collect()
            assert ref() is None
        else:
            async def send(message):
                if message["type"] == "http.response.body":
                    raise primary

            with pytest.raises(type(primary)) as caught:
                await reply({"type": "http"}, None, send)
            assert caught.value is primary
            assert reply.body == b""
        assert calls == []  # No fallible retirement from finally or __del__.
        assert owner.retirement.exit_seen and owner.pending_retirement
        _charged(h, owner.retirement, original)
        with pytest.raises(RuntimeError) as caught:
            await owner._drain_retirement()
        assert caught.value is backend_error
        _charged(h, owner.retirement, original)
        assert calls == [id(original.token)] and h.aborts == 1
        assert original.capacity._owners == {original.token: original.amount}
        for _ in range(2):
            with pytest.raises(RuntimeError, match="ambiguous"):
                await owner._drain_retirement()
        assert calls == [id(original.token)] and not h.refunds
        _charged(h, owner.retirement, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
def test_lifespan_body_primary_identity_wins_over_shutdown_abort_failure(custody, monkeypatch, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        primary = ValueError("body") if kind == "error" else asyncio.CancelledError("body")
        h.failure = RuntimeError("abort")
        locked = []
        retire = owner.retire

        def checked_retire():
            locked.append(owner.lock.locked())
            return retire()

        monkeypatch.setattr(owner, "retire", checked_retire)
        with pytest.raises(type(primary)) as caught:
            async with app.router.lifespan_context(app):
                raise primary
        assert caught.value is primary
        assert not hasattr(primary, "__notes__")
        assert primary.__context__ is not h.failure
        assert primary.__cause__ is not h.failure
        assert owner.closed and locked == [True] and h.aborts == 1
        _charged(h, owner.retirement, original)
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel", "normal"])
def test_lifespan_body_then_expiry_failure_precedence_with_locked_shutdown(custody, monkeypatch, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        owner.ttl = 0.001
        owner.deadline = 0
        expiry = RuntimeError("expiry abort")
        h.failure = expiry
        primary = (ValueError("body") if kind == "error" else
                   asyncio.CancelledError("body") if kind == "cancel" else expiry)
        locked = []
        retire = owner.retire

        def checked_retire():
            locked.append(owner.lock.locked())
            return retire()

        monkeypatch.setattr(owner, "retire", checked_retire)
        with pytest.raises(type(primary)) as caught:
            async with app.router.lifespan_context(app):
                async with asyncio.timeout(1.0):
                    while not owner.pending_retirement:
                        await asyncio.sleep(0.001)
                if kind != "normal":
                    raise primary
        assert caught.value is primary
        if kind != "normal":
            assert not hasattr(primary, "__notes__")
            assert primary.__context__ is not expiry
            assert primary.__cause__ is not expiry
        assert owner.closed and locked == [True, True] and h.aborts == 1
        _charged(h, owner.retirement, original)
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
@pytest.mark.parametrize("fault", ["notes", "type", "frames"])
def test_lifespan_diagnostic_fault_never_masks_exact_primary(custody, monkeypatch, kind, fault):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        primary = ValueError("body") if kind == "error" else asyncio.CancelledError("body")
        if fault == "notes":
            primary.__notes__ = 17  # Builtin BaseException.add_note must fail.
        if fault == "type":
            class InaccessibleName(type):
                def __getattribute__(cls, name):
                    if name == "__name__":
                        raise KeyboardInterrupt("diagnostic type access")
                    return super().__getattribute__(name)

            class CleanupFailure(BaseException, metaclass=InaccessibleName):
                pass

            h.failure = CleanupFailure()
        else:
            h.failure = RuntimeError("abort")
        if fault == "frames":
            def fail_clear(error):
                raise KeyboardInterrupt("diagnostic frame clear")

            monkeypatch.setattr(worker, "_clear_failure_frames", fail_clear)
        with pytest.raises(type(primary)) as caught:
            async with app.router.lifespan_context(app):
                raise primary
        assert caught.value is primary
        assert owner.closed and h.aborts == 1
        _charged(h, owner.retirement, original)
        if fault == "notes":
            assert primary.__notes__ == 17
        else:
            assert not hasattr(primary, "__notes__")
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())


@pytest.mark.parametrize("state", ["released", "outstanding", "failed_abort"])
def test_lifespan_terminal_shortcut_drains_or_reports_incomplete_without_abort_retry(custody, monkeypatch, state):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        if state == "failed_abort":
            h.failure = RuntimeError("first abort")
            with pytest.raises(RuntimeError):
                owner.retire()
            h.failure = None  # Shutdown must NOT retry even though retry could work.
        else:
            owner.retire()
        if state == "released":
            reply.close()
        drain = owner._drain_retirement
        drain_locks = []

        async def checked_drain():
            drain_locks.append(owner.lock.locked())
            return await drain()

        monkeypatch.setattr(owner, "_drain_retirement", checked_drain)
        _charged(h, owner.retirement, original)
        if state == "released":
            async with app.router.lifespan_context(app):
                pass
            assert drain_locks == [False]
            assert owner.retirement.done and not owner.pending_retirement
            _refunded(h, owner.retirement, original)
        else:
            with pytest.raises(RuntimeError, match="GKM worker retirement incomplete"):
                async with app.router.lifespan_context(app):
                    pass
            assert drain_locks == []
            _charged(h, owner.retirement, original)
            assert owner.pending_retirement
            reply.close()  # No backend on last release, even after shutdown.
            _charged(h, owner.retirement, original)
            if state == "failed_abort":
                await _blocked_drain(h, owner, original)
            else:
                await owner._drain_retirement()
                _refunded(h, owner.retirement, original)
        assert owner.closed
        assert h.aborts == 1

    asyncio.run(run())


@pytest.mark.parametrize("body_cancel", [False, True])
@pytest.mark.parametrize("expiry_error", [False, True])
def test_real_parent_cancel_during_same_expiry_wait_preserves_exact_primary_and_joins(custody, monkeypatch, body_cancel, expiry_error):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        h.failure = RuntimeError("shutdown abort")
        expiry_failure = RuntimeError("expiry cancellation cleanup")
        started, release = asyncio.Event(), asyncio.Event()
        body_ready, body_release = asyncio.Event(), asyncio.Event()
        joining, interrupted = asyncio.Event(), asyncio.Event()
        expiry_tasks, body_errors, wait_errors, propagated = [], [], [], []
        parent = None
        real_sleep, real_wait, real_create = asyncio.sleep, asyncio.wait, asyncio.create_task

        def capture_task(coroutine):
            task = real_create(coroutine)
            if coroutine.cr_code.co_name == "expire":
                expiry_tasks.append(task)
            return task

        async def controlled_sleep(delay):
            if expiry_tasks and asyncio.current_task() is expiry_tasks[0]:
                started.set()
                try:
                    await real_sleep(3600)
                finally:
                    # Delay cancellation cleanup at an awaitable boundary, not a
                    # substitute lifespan/expiry coroutine or direct fake cancel.
                    await release.wait()
                    if expiry_error:
                        raise expiry_failure
            else:
                await real_sleep(delay)

        async def observed_wait(tasks, **kwargs):
            if asyncio.current_task() is parent:
                joining.set()
                try:
                    return await real_wait(tasks, **kwargs)
                except asyncio.CancelledError as error:
                    wait_errors.append(error)  # Actual delivered parent object.
                    interrupted.set()
                    raise
            return await real_wait(tasks, **kwargs)

        monkeypatch.setattr(worker.asyncio, "create_task", capture_task)
        monkeypatch.setattr(worker.asyncio, "sleep", controlled_sleep)
        monkeypatch.setattr(worker.asyncio, "wait", observed_wait)

        async def serve():
            try:
                async with app.router.lifespan_context(app):
                    await started.wait()
                    body_ready.set()
                    if body_cancel:
                        try:
                            await body_release.wait()
                        except asyncio.CancelledError as error:
                            body_errors.append(error)
                            raise
            except BaseException as error:
                propagated.append(error)
                raise

        parent = asyncio.create_task(serve())
        try:
            async with asyncio.timeout(1.0):
                await body_ready.wait()
                if body_cancel:
                    parent.cancel("body marker")
                await joining.wait()
                assert len(expiry_tasks) == 1 and not expiry_tasks[0].done()
                parent.cancel("teardown marker")
                await interrupted.wait()
                assert not expiry_tasks[0].done()  # Parent cancel not forwarded.
                _charged(h, owner.retirement, original)
                release.set()
                done, pending = await real_wait({parent, *expiry_tasks}, timeout=0.5)
                assert not pending
            expected = body_errors[0] if body_cancel else wait_errors[0]
            with pytest.raises(asyncio.CancelledError) as caught:
                parent.result()
            assert caught.value is expected is propagated[0]
            assert wait_errors[0] is not body_errors[0] if body_cancel else not body_errors
            assert not hasattr(expected, "__notes__")
            assert owner.closed and h.aborts == 1
            _charged(h, owner.retirement, original)
        finally:
            # All existing tasks joined even on assertion/timeout. Test bounds
            # only; production intentionally has no timeout or detached task.
            release.set()
            body_release.set()
            tasks = {parent, *expiry_tasks}
            done, pending = await real_wait(tasks, timeout=1.0)
            for task in pending:
                task.cancel()
            if pending:
                done, pending = await real_wait(pending, timeout=1.0)
            assert not pending
            for task in tasks:
                if not task.cancelled():
                    task.exception()
        h.failure = None
        await _blocked_drain(h, owner, original)

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["raise_primary", "return_mutated"])
@pytest.mark.parametrize("kind", ["error", "cancel"])
def test_lifespan_does_not_inspect_diagnostic_metadata_or_change_original_tail(custody, monkeypatch, mode, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        fields = tuple(BaseException.__dict__[name] for name in
                       ("__context__", "__cause__", "__traceback__", "__suppress_context__"))
        primary = ValueError("body") if kind == "error" else asyncio.CancelledError("body")
        primary.__notes__ = ["original note"]
        context, cause = LookupError("original context"), RuntimeError("original cause")

        class Payload:
            pass

        def user_tail():
            user_payload = Payload()
            try:
                raise primary
            except BaseException:
                return fields[2].__get__(primary, BaseException), user_payload

        original_tail, user_payload = user_tail()
        for field, value in zip(fields, (context, cause, original_tail, False), strict=True):
            field.__set__(primary, value)
        hook_snapshots, clear_calls = [], []
        failure = None

        class DiagnosticName(type):
            def __getattribute__(cls, name):
                if name == "__name__":
                    hook_snapshots.append(tuple(field.__get__(primary, BaseException)
                                                for field in fields))
                    # Each field changes. Raising the EXACT primary while the
                    # secondary is handled additionally changes its traceback
                    # and implicit context under ordinary Python semantics.
                    for field, value in zip(fields, (failure, failure, None, True), strict=True):
                        field.__set__(primary, value)
                    if mode == "raise_primary":
                        raise primary
                    return "CleanupFailure"
                return super().__getattribute__(name)

        class CleanupFailure(RuntimeError, metaclass=DiagnosticName):
            pass

        failure = CleanupFailure("secondary")
        failure.payload = Payload()
        secondary_ref = weakref.ref(failure.payload)
        h.failure = failure

        def forbidden_clear(error):
            # Were this called on the secondary, this frame would retain its
            # payload while raising primary. Frame clearing is irreversible;
            # restoring only exception metadata cannot repair user frame loss.
            clear_calls.append(True)
            raise primary

        with monkeypatch.context() as patch:
            patch.setattr(worker, "_clear_failure_frames", forbidden_clear)
            with pytest.raises(type(primary)) as caught:
                async with app.router.lifespan_context(app):
                    raise primary
            assert caught.value is primary
            assert clear_calls == []
        assert hook_snapshots == []
        before = (context, cause, original_tail, False)
        assert before[0] is context and before[1] is cause and before[3] is False
        assert fields[0].__get__(primary, BaseException) is context
        assert fields[1].__get__(primary, BaseException) is cause
        assert fields[3].__get__(primary, BaseException) is False
        trace = fields[2].__get__(primary, BaseException)
        nodes = []
        while trace is not None:
            nodes.append(trace)
            trace = trace.tb_next
        assert any(node is before[2] for node in nodes)
        assert any(node is original_tail for node in nodes)
        assert all(node.tb_frame.f_code.co_name not in
                   ("_retirement_error", "__getattribute__", "forbidden_clear") for node in nodes)
        assert original_tail.tb_frame.f_locals["user_payload"] is user_payload
        assert primary.__notes__ == ["original note"]
        assert owner.closed and owner.pending_retirement and h.aborts == 1
        _charged(h, owner.retirement, original)
        # Drop TEST-owned records/class/closures, not merely production locals.
        # Keep primary and its original user frames alive throughout collection.
        h.failure = failure = CleanupFailure = DiagnosticName = forbidden_clear = caught = None
        gc.collect()
        assert secondary_ref() is None
        _charged(h, owner.retirement, original)
        assert original.backend().gen.gi_frame is not None
        with pytest.raises(RuntimeError):
            original.manager().drain_failed()  # Proof missing; NEVER aborts native.
        assert owner.pending_retirement and h.aborts == 1 and not h.refunds
        _charged(h, owner.retirement, original)
        await _blocked_drain(h, owner, original)
        assert h.aborts == 1

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
def test_diagnostic_selection_requires_no_fallible_metadata_setup(custody, kind):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        primary = ValueError("body") if kind == "error" else asyncio.CancelledError("body")
        primary.__notes__ = ["original note"]
        fields = tuple(BaseException.__dict__[name] for name in
                       ("__context__", "__cause__", "__traceback__", "__suppress_context__"))

        class Payload:
            pass

        def user_tail():
            user_payload = Payload()
            try:
                raise primary
            except BaseException:
                return fields[2].__get__(primary, BaseException), user_payload

        tail, user_payload = user_tail()
        context, cause = LookupError("original context"), RuntimeError("original cause")
        for field, value in zip(fields, (context, cause, tail, False), strict=True):
            field.__set__(primary, value)
        failure = RuntimeError("abort")
        failure.payload = Payload()
        secondary_ref = weakref.ref(failure.payload)
        h.failure = failure
        try:
            owner.retire()
        except BaseException as secondary:
            assert secondary is failure
            selected = worker._retirement_error(primary, secondary, "shutdown")
        assert selected is primary
        assert primary.__notes__ == ["original note"]
        assert fields[0].__get__(primary, BaseException) is context
        assert fields[1].__get__(primary, BaseException) is cause
        assert fields[2].__get__(primary, BaseException) is tail
        assert fields[3].__get__(primary, BaseException) is False
        assert tail.tb_frame.f_locals["user_payload"] is user_payload
        assert owner.pending_retirement and h.aborts == 1
        _charged(h, owner.retirement, original)
        h.failure = failure = None
        gc.collect()
        assert secondary_ref() is None
        _charged(h, owner.retirement, original)
        assert original.backend().gen.gi_frame is not None and not h.refunds
        await _blocked_drain(h, owner, original)
        assert h.aborts == 1

    asyncio.run(run())


@pytest.mark.parametrize("failure_type", [RuntimeError, asyncio.CancelledError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("path", ["drain", "lifespan", "lifespan_body_error", "lifespan_body_cancel",
                                  "admission", "operation"])
def test_first_completion_backend_failure_identity_and_ingress_classification(
        custody, monkeypatch, failure_type, path):
    h = custody

    async def run():
        app, owner = await h.start()
        retirement = owner.retirement
        original = _original(retirement)
        reply = worker._PayloadResponse(b"reply", owner.reservation, limit=32)
        async with owner.lock:
            owner.retire()
        _charged(h, retirement, original)
        backend_error = failure_type("first backend finalization")
        calls = []
        real_exit = type(original.backend()).__exit__

        def fail_exit(backend, *exc):
            if backend is original.backend():
                calls.append((backend, owner.lock.locked(), h.capacity.lock._is_owned()))
                raise backend_error
            return real_exit(backend, *exc)

        monkeypatch.setattr(type(original.backend()), "__exit__", fail_exit)
        reply.close()  # Recording only, even with a failing backend installed.
        assert calls == [] and retirement.exit_seen and retirement.root._references == 0
        assert h.aborts == 1 and retirement.abort_ok and retirement.native is None
        _charged(h, retirement, original)

        body_primary = None
        if path == "lifespan_body_error":
            body_primary = ValueError("established body primary")
        elif path == "lifespan_body_cancel":
            body_primary = asyncio.CancelledError("established body primary")
        selected = body_primary if body_primary is not None else backend_error
        if path == "drain":
            with pytest.raises(failure_type) as caught:
                await owner._drain_retirement()
            assert caught.value is backend_error
        elif path.startswith("lifespan"):
            with pytest.raises(type(selected)) as caught:
                async with app.router.lifespan_context(app):
                    if body_primary is not None:
                        raise body_primary
            assert caught.value is selected
        else:
            suffix = "/admit" if path == "admission" else "/{request}/{action}"
            response = await h.endpoint(app, suffix)(h.request(owner, action="step", body=b"x"))
            assert response.status_code == 409

        assert calls == [(original.backend(), True, False)]
        assert retirement.busy and not retirement.done and not h.refunds
        assert not hasattr(backend_error, "__notes__")
        _charged(h, retirement, original)
        for suffix in ("/admit", "/{request}/{action}"):
            assert (await h.endpoint(app, suffix)(
                h.request(owner, action="step", body=b"x"))).status_code == 409
        await _blocked_drain(h, owner, original)
        with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
            async with app.router.lifespan_context(app):
                pass
        assert calls == [(original.backend(), True, False)] and h.aborts == 1
        assert h.loads == 1 and not h.refunds
        _charged(h, retirement, original)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["error", "cancel"])
@pytest.mark.parametrize("annotation_failure", ["primary", "ordinary"])
def test_diagnostic_selection_never_calls_annotation_or_primary_attribute_hooks(custody, kind, annotation_failure):
    h = custody

    async def run():
        app, owner = await h.start()
        original = _original(owner.retirement)
        fields = tuple(BaseException.__dict__[name] for name in
                       ("__context__", "__cause__", "__traceback__", "__suppress_context__"))
        metadata_names = ("__context__", "__cause__", "__traceback__", "__suppress_context__")
        armed = False
        metadata_access, annotation_calls = [], []
        failure = annotation_error = None
        base = ValueError if kind == "error" else asyncio.CancelledError

        class Primary(base):
            def __getattribute__(self, name):
                if armed and name in metadata_names:
                    metadata_access.append(("get", name))
                    raise AssertionError("metadata subclass getter must be bypassed")
                if armed and name == "__notes__":
                    annotation_calls.append(True)
                    for field, value in zip(fields, (failure, failure, None, True), strict=True):
                        field.__set__(self, value)
                    if annotation_failure == "primary":
                        raise self
                    raise annotation_error
                return BaseException.__getattribute__(self, name)

            def __setattr__(self, name, value):
                if armed and name in metadata_names:
                    metadata_access.append(("set", name))
                    raise AssertionError("metadata subclass setter must be bypassed")
                BaseException.__setattr__(self, name, value)

        class Payload:
            pass

        primary = Primary("body")
        notes = ["original note"]
        BaseException.__setattr__(primary, "__notes__", notes)

        def user_tail():
            user_payload = Payload()
            try:
                raise primary
            except BaseException:
                return fields[2].__get__(primary, BaseException), user_payload

        tail, user_payload = user_tail()
        context, cause = LookupError("original context"), RuntimeError("original cause")
        for field, value in zip(fields, (context, cause, tail, False), strict=True):
            field.__set__(primary, value)
        failure = RuntimeError("abort")
        failure.payload = Payload()
        secondary_ref = weakref.ref(failure.payload)
        annotation_error = RuntimeError("annotation")
        annotation_error.payload = Payload()
        annotation_ref = weakref.ref(annotation_error.payload)
        h.failure = failure
        try:
            owner.retire()
        except BaseException as secondary:
            assert secondary is failure
            armed = True
            try:
                selected = worker._retirement_error(primary, secondary, "shutdown")
            finally:
                armed = False
        assert selected is primary and annotation_calls == []
        assert metadata_access == []
        assert BaseException.__getattribute__(primary, "__notes__") is notes
        assert notes == ["original note"]
        assert fields[0].__get__(primary, BaseException) is context
        assert fields[1].__get__(primary, BaseException) is cause
        assert fields[2].__get__(primary, BaseException) is tail
        assert fields[3].__get__(primary, BaseException) is False
        assert tail.tb_frame.f_locals["user_payload"] is user_payload
        assert owner.pending_retirement and h.aborts == 1
        _charged(h, owner.retirement, original)
        # Primary still owns its class: explicitly empty secondary closure cells.
        h.failure = failure = annotation_error = Primary = None
        gc.collect()
        assert secondary_ref() is annotation_ref() is None
        _charged(h, owner.retirement, original)
        assert original.backend().gen.gi_frame is not None and not h.refunds
        await _blocked_drain(h, owner, original)
        assert h.aborts == 1

    asyncio.run(run())
