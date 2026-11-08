"""Source-only current-envelope regressions, not long-prefix/native qualification.

Synthetic boundaries and a generated public tokenizer/template fixture only.
No cached Qwen cohort, tensor payload, tape price or serving fit is established.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pllm import _native
from pllm.runtime import fss_source as source
from pllm.runtime import fss_tape, fss_transport

_RESOLVE_CLIENT_SOURCE = source.resolve_client_source


def _experiment(rows=32, outputs=2):
    # Boundary tests stop before Experiment/native/source admission is needed.
    return SimpleNamespace(budget=SimpleNamespace(
        max_input_tokens=rows, max_new_tokens=outputs))


@pytest.fixture
def forbidden_acquisition(monkeypatch):
    calls = []

    def forbidden(name):
        def fail(*args, **kwargs):
            calls.append(name)
            pytest.fail(f"early refusal reached {name}")
        return fail

    for name in ("resolve_client_source", "_experiment", "_geometry", "_files",
                 "_locked", "_store", "_unchanged"):
        monkeypatch.setattr(source, name, forbidden(name))
    for name in ("resolve_experiment", "fss_source_contract", "FssSourceBuilder",
                 "FssClientRequest", "FssRoleRequest"):
        monkeypatch.setattr(_native, name, forbidden(name), raising=False)
    monkeypatch.setattr(source.io, "_read_range", forbidden("source body read"))
    monkeypatch.setattr(fss_transport.secrets, "token_hex", forbidden("transport entropy"))
    monkeypatch.setattr(fss_transport.secrets, "token_bytes", forbidden("request entropy"))
    # FssRoleRequest also owns dealer construction. No transport entry or tape
    # pricing/allocation may follow the source-boundary refusal.
    monkeypatch.setattr(fss_tape.MaterialTape, "price", forbidden("tape price"))
    monkeypatch.setattr(fss_tape.MaterialTape, "__init__", forbidden("tape factory"))
    monkeypatch.setattr(fss_transport, "execute_request", forbidden("transport/dealer"))
    return calls


@pytest.mark.parametrize("rows,outputs,prefixes,allowed", [
    (30, 2, (30, 31), False),
    (32, 2, (32, 33), False),
    (16, 2, (16, 17), False),
    (16, 1, (16,), True),
    (15, 2, (15, 16), True),
    (13, 4, (13, 14, 15, 16), True),
    (14, 4, (14, 15, 16, 17), False),
    (1, 4, (1, 2, 3, 4), True),
])
@pytest.mark.parametrize("configured", [False, True])
def test_exact_scalar_boundaries_before_acquisition(
        monkeypatch, forbidden_acquisition, rows, outputs, prefixes, allowed, configured):
    # Independent literal phase oracle: each selection sees all preceding
    # feedback, not just the original rows or a truncated/padded prompt.
    assert prefixes[0] == rows and len(prefixes) == outputs
    assert allowed is all(1 <= prefix <= 16 for prefix in prefixes)
    experiment = _experiment(rows, outputs) if configured else _experiment()
    kwargs = {} if configured else {"input_rows": rows, "outputs": outputs}
    phases, failures, reached = [], [], []
    pins = {"unchanged": "opaque expected source pins"}

    class NextGate(Exception):
        pass

    if allowed:
        def resolve(actual, **options):
            reached.append(actual)
            assert actual is experiment and options["expected_pins"] is pins
            raise NextGate("existing source admission still required")
        monkeypatch.setattr(source, "resolve_client_source", resolve)

    error = NextGate if allowed else source.FssSourceError
    message = "existing source admission" if allowed else "^FuseFSS invalid request geometry$"
    with pytest.raises(error, match=message):
        source.load_role_source(experiment, expected_pins=pins,
            lifecycle=phases.append, on_failure=lambda: failures.append("failed"), **kwargs)
    assert len(reached) == int(allowed)
    assert phases == ["qualification_metadata"] and failures == ["failed"]
    assert forbidden_acquisition == []
    assert (experiment.budget.max_input_tokens, experiment.budget.max_new_tokens) == (
        (rows, outputs) if configured else (32, 2))


class _IntSubclass(int):
    pass


@pytest.mark.parametrize("rows,outputs", [
    (0, 2), (-1, 2), (True, 2), (False, 2), (1.0, 2), ("1", 2),
    (_IntSubclass(1), 2), (None, 2),
    (1, 0), (1, -1), (1, True), (1, False), (1, 2.0), (1, "2"),
    (1, _IntSubclass(2)), (1, None), (1, 5),
])
def test_noninteger_and_out_of_range_configured_geometry_refuses(
        forbidden_acquisition, rows, outputs):
    # None is a default sentinel in load_role_source, not a legal budget value.
    with pytest.raises(source.FssSourceError, match="^FuseFSS invalid request geometry$"):
        source.load_role_source(_experiment(rows, outputs), expected_pins={})
    assert forbidden_acquisition == []


@pytest.mark.parametrize("rows,outputs", [
    (0, 2), (-1, 2), (True, 2), (False, 2), (1.0, 2), ("1", 2),
    (_IntSubclass(1), 2), (1, 0), (1, -1), (1, True), (1, False),
    (1, 2.0), (1, "2"), (1, _IntSubclass(2)), (1, 5),
])
def test_invalid_explicit_geometry_is_not_coerced(forbidden_acquisition, rows, outputs):
    with pytest.raises(source.FssSourceError, match="^FuseFSS invalid request geometry$"):
        source.load_role_source(_experiment(), expected_pins={}, input_rows=rows, outputs=outputs)
    assert forbidden_acquisition == []


def _frozen_source():
    return source.FrozenFssSource(
        model=object(), path=None, experiment_bytes=b"fixture", config_bytes=b"{}",
        config_source_bytes=b"{}", contract_bytes=b"{}", checkpoint_digest="1" * 64,
        source_lock_digest="2" * 64, tokenizer_digest="3" * 64,
        tokenizer_bytes=b"", _lock=b"{}", _tokenizer_options=b"{}",
        _identities=(), _specs=())


@pytest.mark.parametrize("cancelled", [False, True])
def test_original_supplied_owner_errors_precede_geometry(forbidden_acquisition, cancelled):
    owned = _frozen_source()
    phases, failures = [], []
    expected = "source import cancelled" if cancelled else "original snapshot pins differ"
    with pytest.raises(source.FssSourceError, match=expected):
        source.load_role_source(_experiment(), expected_pins={}, source=owned,
            input_rows=30, outputs=2, cancelled=lambda: cancelled,
            lifecycle=phases.append, on_failure=lambda: failures.append("failed"))
    assert phases == ["qualification_metadata"] and failures == ["failed"]
    assert forbidden_acquisition == []


def test_supplied_wrong_type_keeps_original_error(forbidden_acquisition):
    with pytest.raises(TypeError, match="requires FrozenFssSource metadata"):
        source.load_role_source(_experiment(), expected_pins={}, source=object(), input_rows=30)
    assert forbidden_acquisition == []


def test_supplied_owner_neither_reacquired_nor_admitted_by_geometry(monkeypatch, forbidden_acquisition):
    owned = _frozen_source()
    experiment = _experiment()
    calls = []

    def experiment_contract(actual):
        calls.append(actual)
        return b"different Experiment", None, None, None

    monkeypatch.setattr(source, "_experiment", experiment_contract)
    with pytest.raises(source.FssSourceError, match="invalid request geometry"):
        source.load_role_source(experiment, source=owned, expected_pins=owned.snapshot_pins,
                                input_rows=16, outputs=2)
    assert calls == []
    with pytest.raises(source.FssSourceError, match="requires exact admitted Experiment"):
        source.load_role_source(experiment, source=owned, expected_pins=owned.snapshot_pins,
                                input_rows=15, outputs=2)
    assert calls == [experiment] and forbidden_acquisition == []


@pytest.mark.parametrize("is_cancelled", [False, True])
def test_cancellation_and_observer_failure_keep_original_error(forbidden_acquisition, is_cancelled):
    events = []

    def cancelled():
        events.append("cancel")
        return is_cancelled

    def failure():
        events.append("failure")
        raise RuntimeError("observer must not replace refusal")

    expected = "source import cancelled" if is_cancelled else "invalid request geometry"
    with pytest.raises(source.FssSourceError, match=f"^FuseFSS {expected}$"):
        source.load_role_source(_experiment(), expected_pins={}, cancelled=cancelled,
                                lifecycle=events.append, on_failure=failure)
    assert events == ["qualification_metadata", "cancel", "failure"]
    assert forbidden_acquisition == []


@pytest.mark.parametrize("callback_error", [None, RuntimeError, KeyboardInterrupt])
def test_real_inner_resolution_failure_notifies_once_before_metadata_retirement(
        tmp_path, monkeypatch, forbidden_acquisition, callback_error):
    from contextlib import contextmanager
    from test_fss_source import experiment as fss_experiment

    # Keep BOTH real resolver handlers and the actual _resolve_client_source
    # frame. Fail at its first config identity access, inside a retained lease;
    # replacing resolve_client_source with a throwing stub would miss the bug.
    experiment = fss_experiment(tmp_path, rows=15, outputs=2, capacity="decoder")
    monkeypatch.setattr(source, "resolve_client_source", _RESOLVE_CLIENT_SOURCE)
    monkeypatch.setattr(source, "_experiment", lambda actual: (
        b"metadata-only", actual.pipeline.model, "metadata-only", "decoder"))
    events, shared_callbacks, notifications = [], [], []
    lease = SimpleNamespace(live=False, retirements=0)
    original = source.FssSourceError("injected config identity failure")
    raised_by_callback = None if callback_error is None else callback_error("observer failed")

    @contextmanager
    def reserve_metadata():
        assert not lease.live and lease.retirements == 0
        lease.live = True
        events.append("metadata_enter")
        try:
            yield lease
        finally:
            assert lease.live
            events.append("metadata_retire")
            lease.live = False
            lease.retirements += 1

    def identity(path):
        assert lease.live and path.name == "config.json"
        events.append("config_identity_failure")
        raise original

    def failure():
        notifications.append(lease.live)
        assert lease.live and lease.retirements == 0
        events.append("notify")
        # Reenter the exact wrapper forwarded to the REAL inner handler. It
        # must already be marked, including when this callback raises below.
        assert len(shared_callbacks) == 1
        shared_callbacks[0]()
        if raised_by_callback is not None:
            raise raised_by_callback

    notify = source._notify_failure

    def observe_notification(observer):
        if observer is not failure:
            shared_callbacks.append(observer)
        return notify(observer)  # Real Exception suppression/BaseException policy.

    monkeypatch.setattr(source, "_notify_failure", observe_notification)
    monkeypatch.setattr(source.tokenizer_admission, "reserve_decoder_metadata", reserve_metadata)
    monkeypatch.setattr(source.io, "_identity", identity)
    def read(*args, **kwargs):
        pytest.fail("config identity refusal reached metadata/body bytes")
    monkeypatch.setattr(source.io, "_read_bounded", read)
    expected = raised_by_callback if callback_error is KeyboardInterrupt else original
    with pytest.raises(type(expected)) as caught:
        source.load_role_source(experiment, expected_pins={}, lifecycle=events.append, on_failure=failure)
    assert caught.value is expected
    assert notifications == [True] and len(shared_callbacks) == 1
    assert events == ["qualification_metadata", "source_resolution", "metadata_enter",
                      "source_config", "config_identity_failure", "notify", "metadata_retire"]
    assert not lease.live and lease.retirements == 1
    # An obsolete outer notification must not call the observer or revive the
    # retired lease, even after callback BaseException replaced the source error.
    shared_callbacks[0]()
    assert notifications == [True] and not lease.live and lease.retirements == 1
    assert forbidden_acquisition == []


@pytest.fixture
def source_abort_owners(monkeypatch, forbidden_acquisition):
    # Exercise the real loader/handlers with explicit source/native doubles.
    # Empty tensor inventory avoids payload work; this is NOT source admission,
    # native execution, reservation funding or failed-close custody evidence.
    owned = _frozen_source()
    contract = source.io._canonical({
        "source_float32_bytes": 1, "source_shapes": {},
        "numeric_profile": "metadata-only", "vocabulary": 1})
    state = SimpleNamespace(
        events=[], original=source.FssSourceError("injected manifest failure"),
        manifest_error=True, cancel_at=None, cancelled=False,
        program_closes=0, builder_closes=0, store_clears=0,
        program_live=True, builder_live=True, store_live=True,
        program_error=None, builder_error=None, notifications=0,
        observer_error=None, shared_notify=None)

    class Program:
        def manifest(self, actual_contract):
            assert actual_contract is contract
            if state.manifest_error:
                raise state.original
            return source.io._canonical({
                "schema": "pllm.fss.channel.program.v1",
                "contract": source.hashlib.sha256(contract).hexdigest(),
                "input_digest": source.hashlib.sha256(
                    b"pllm.fss.original-ieee-stream.v1\0").hexdigest(),
                "profile": "metadata-only", "initial": 1, "selections": 1,
                "vocabulary": 1, "source": "1" * 64, "program": "2" * 64})

        def close(self):
            state.events.append("program")
            state.program_closes += 1
            assert state.program_closes == 1
            if state.program_error is not None:
                raise state.program_error
            state.program_live = False

    program = Program()

    class Builder:
        def seal(self):
            return program

        def close(self):
            state.events.append("builder")
            state.builder_closes += 1
            assert state.builder_closes == 1
            if state.builder_error is not None:
                raise state.builder_error
            state.builder_live = False

    def clear_cache():
        state.events.append("store")
        state.store_clears += 1
        assert state.store_clears == 1
        state.store_live = False

    def resolve(experiment, **options):
        state.shared_notify = options["on_failure"]
        return owned

    def runtime_contract(actual, experiment, rows, outputs):
        assert actual is owned and (rows, outputs) == (1, 1)
        return contract

    def phase(name):
        if name == state.cancel_at:
            state.cancelled = True

    def failure():
        state.events.append("notify")
        state.notifications += 1
        assert state.notifications == 1
        assert state.program_live and state.builder_live and state.store_live
        # Actual invocation-local notifier, already marked before this callback.
        if state.shared_notify is not None:
            state.shared_notify()
        if state.observer_error is not None:
            raise state.observer_error

    monkeypatch.setattr(source, "resolve_client_source", resolve)
    monkeypatch.setattr(source.FrozenFssSource, "runtime_contract", runtime_contract)
    monkeypatch.setattr(source, "_locked", lambda *args: (
        owned.checkpoint_digest, owned.source_lock_digest, {}))
    monkeypatch.setattr(source, "_store", lambda *args: (
        SimpleNamespace(clear_cache=clear_cache), ()))
    monkeypatch.setattr(_native, "FssSourceBuilder", lambda *args, **kwargs: Builder())
    monkeypatch.setattr(source, "_rotary_specs", lambda record: ())

    def load():
        return source.load_role_source(
            _experiment(1, 1), expected_pins=owned.snapshot_pins,
            cancelled=lambda: state.cancelled, lifecycle=phase, on_failure=failure)

    state.load, state.program, state.forbidden = load, program, forbidden_acquisition
    return state


def test_real_source_abort_closes_program_before_builder_and_store(source_abort_owners):
    state = source_abort_owners
    with pytest.raises(source.FssSourceError) as caught:
        state.load()
    assert caught.value is state.original
    assert state.events == ["notify", "program", "builder", "store"]
    assert not state.program_live and not state.builder_live and not state.store_live
    assert state.forbidden == []


@pytest.mark.parametrize("observer_error", [RuntimeError, KeyboardInterrupt])
def test_real_source_abort_observer_error_keeps_cleanup_and_chain(source_abort_owners, observer_error):
    state = source_abort_owners
    state.observer_error = observer_error("injected observer failure")
    expected = state.observer_error if observer_error is KeyboardInterrupt else state.original
    with pytest.raises(type(expected)) as caught:
        state.load()
    assert caught.value is expected
    if observer_error is KeyboardInterrupt:
        assert caught.value.__context__ is state.original
    assert state.events == ["notify", "program", "builder", "store"]
    assert state.notifications == 1 and state.program_closes == 1
    assert state.forbidden == []


@pytest.mark.parametrize("observer_error", [None, KeyboardInterrupt])
def test_real_source_abort_failed_program_close_propagates_without_retry(
        source_abort_owners, observer_error):
    state = source_abort_owners
    state.program_error = RuntimeError("injected program close failure")
    if observer_error is not None:
        state.observer_error = observer_error("injected observer failure")
    with pytest.raises(RuntimeError) as caught:
        state.load()
    assert caught.value is state.program_error
    assert caught.value.__context__ is (state.observer_error or state.original)
    if state.observer_error is not None:
        assert state.observer_error.__context__ is state.original
    assert state.events == ["notify", "program", "builder", "store"]
    assert state.program_closes == 1 and state.program_live
    assert not state.builder_live and not state.store_live
    assert state.forbidden == []
    # No durable owner/lease retention claim: that fifth regression stays gated.


@pytest.mark.parametrize("cancel_at", ["qualification_metadata", "source_recheck", "source_manifest"])
def test_real_source_cancel_and_builder_failure_still_clear_store(source_abort_owners, cancel_at):
    state = source_abort_owners
    state.cancel_at = cancel_at
    state.manifest_error = False
    if cancel_at == "source_recheck":
        state.builder_error = RuntimeError("injected builder close failure")
    expected = RuntimeError if state.builder_error is not None else source.FssSourceError
    with pytest.raises(expected) as caught:
        state.load()
    if state.builder_error is not None:
        assert caught.value is state.builder_error
        assert isinstance(caught.value.__context__, source.FssSourceError)
        assert str(caught.value.__context__) == "FuseFSS source import cancelled"
    else:
        assert str(caught.value) == "FuseFSS source import cancelled"
    cleanup = {"qualification_metadata": [], "source_recheck": ["builder", "store"],
               "source_manifest": ["program", "builder", "store"]}[cancel_at]
    assert state.events == ["notify", *cleanup]
    assert state.program_closes == int(cancel_at == "source_manifest")
    assert state.store_clears == int(cancel_at != "qualification_metadata")
    assert state.forbidden == []


def test_real_source_success_keeps_returned_program_open(source_abort_owners):
    state = source_abort_owners
    state.manifest_error = False
    loaded = state.load()
    assert loaded.native_program is state.program
    assert state.events == ["builder", "store"]
    assert state.program_closes == 0 and state.program_live
    assert state.notifications == 0 and state.forbidden == []
    loaded.close()
    assert state.program_closes == 1 and not state.program_live


@pytest.mark.parametrize("ending", ["geometry", "wrong_source", "wrong_request", "cancel"])
def test_source_owned_template_ids_and_original_caller_retirement(
        tmp_path, monkeypatch, forbidden_acquisition, ending):
    from test_prepared_prompt import _real_prompt_metadata, _observe_source_tokenizer
    from pllm.configuration import Experiment
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.fss_client import FssClient
    from pllm.runtime.responses import _PreparedPromptUse, _prepared_prompt_request

    # Reuse existing source/tokenizer admission fixture and GKM-owned preparation
    # API. This generated template is NOT the actual Qwen tokenizer/cohort.
    load = source.load_role_source
    monkeypatch.setattr(Experiment, "resolve", lambda experiment: SimpleNamespace(
        client_runtime="compiled_fss_v1", model="metadata-only"))
    template = "{{ bos_token }} SYS {{ messages[0].content }} {% if add_generation_prompt %}ASSISTANT{% endif %}"
    exp, _, _, old_core, owned = _real_prompt_metadata(
        tmp_path, monkeypatch, "fss", template=template)
    old_core.close()
    exp = replace(exp, budget=replace(exp.budget, max_input_tokens=32, max_new_tokens=2))
    owned = replace(owned, experiment_bytes=exp.canonical_bytes())
    core = FssClient(exp, {role: (f"http://{role}.invalid/v1", "metadata-only")
                          for role in FssClient.role_names}, source=owned)
    core._source_checked = True  # Metadata fixture only, never source qualification.
    observed = _observe_source_tokenizer(monkeypatch, owned)
    prepared = _PreparedPromptUse._empty()
    request = {"model": core.default_model, "input": " ".join(["hello"] * 26),
               "max_output_tokens": 2}
    original_request = dict(request)
    core._prepare_request_prompt(request, _preparation_owner=prepared)
    payload = prepared._payload
    expected_ids = (0, 0, 3, *([4] * 26), 5)
    assert payload.ids == expected_ids and len(expected_ids) == 30
    assert observed[0].calls == [("<bos> SYS " + request["input"] + " ASSISTANT", True)]
    reached, failures = [], []

    def execute(actual, ids, count, attempt):
        reached.append((ids, count))
        assert actual is owned and ids is payload.ids and count == 2
        assert observed[0].closed == 0 and attempt._prepared_prompt is prepared

        def failure():
            assert observed[0].closed == 0
            assert prepared._payload is payload and not prepared._closed
            assert attempt._prepared_prompt is prepared and core._active is attempt
            assert payload.ids is ids and request == original_request
            failures.append("failed")

        return load(exp, expected_pins=owned.snapshot_pins, input_rows=len(ids), outputs=count,
                    cancelled=attempt, on_failure=failure)

    monkeypatch.setattr(core, "_execute", execute)
    # _real_prompt_metadata already forbids native/tensor/transport work. Block
    # source resolution too: the scalar refusal must not fetch even metadata.
    def resolve(*args, **kwargs):
        pytest.fail("prepared impossible geometry reacquired complete source")
    monkeypatch.setattr(source, "resolve_client_source", resolve)
    if ending == "wrong_source":
        core._source = replace(owned, source_lock_digest="4" * 64)
    body = dict(request)
    if ending == "wrong_request":
        body["input"] = "changed"
    token = _prepared_prompt_request.set(prepared)
    try:
        attempt = core.events(body)
    finally:
        _prepared_prompt_request.reset(token)
    try:
        if ending == "cancel":
            attempt.cancel()
        error = source.FssSourceError if ending == "geometry" else (
            ProtocolError if ending == "cancel" else ValueError)
        match = "invalid request geometry" if ending == "geometry" else (
            "cancelled" if ending == "cancel" else "association changed")
        with pytest.raises(error, match=match):
            next(attempt)
        assert reached == ([(expected_ids, 2)] if ending == "geometry" else [])
        assert failures == (["failed"] if ending == "geometry" else [])
        assert payload.ids == expected_ids and request == original_request
    finally:
        attempt.close()
        core.close()
    assert prepared._payload is None and core._active is None
    assert len(observed) == 1 and len(observed[0].calls) == 1
    assert observed[0].closed == 1
    assert forbidden_acquisition == []
