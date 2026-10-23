"""Cached guard notes; mocked admission/native owners, real Python I/O drain.

No sockets, checkpoints, Torch models, or native execution. The CLI regression
reuses existing synthetic source/quality fixtures, never a replacement scorer.
"""
import asyncio
from contextlib import contextmanager
import hashlib
import inspect
import json
import sys
import threading
from types import SimpleNamespace as NS
import traceback

import pytest

from pllm.runtime import benchmark_memory as memory
from pllm.runtime import delegated_transport as transport
from pllm.runtime import reference_benchmark as benchmark
from pllm.runtime import servers
from pllm.runtime.client import ProtocolError
from test_paper_quality_cli import documented_argv, lane as cli_lane, quality as quality


WARNING = "benchmark stopped: macOS memory pressure is warning"
OBSERVATION = {"pressure": "warning", "available_bytes": 512, "swap_used_bytes": 1024}


@pytest.fixture
def admission(monkeypatch):
    events, instances = [], []
    state = NS(mode="guard", cleanup=set(), mutate=False, primary=None)
    cleanup_errors = {name: RuntimeError(name + " cleanup failed")
                      for name in ("native", "peers", "roles", "guard")}
    ordinary_error = ProtocolError("ordinary admission failure", 503)
    good = NS(pressure="normal", available=4096, swap_used=0)
    bad = NS(**{key.replace("_bytes", ""): value for key, value in OBSERVATION.items()})
    report = {"host": {"available": 4096, "swap_used": 0, "reserve_bytes": 0},
              "runtime_floor_bytes": 256, "admitted": True}
    raw_contract = b"mock-native-contract"
    entered = threading.Event()
    real_watchdog, real_peers = memory.MemoryWatchdog, transport._Peers

    def sample():
        return bad if entered.is_set() and state.mode == "guard" else good

    def watchdog(report, abort):
        guard = real_watchdog(report, abort, sample=sample)
        state.guard = guard
        close = guard.close

        def closing():
            close()
            events.append("guard_closed")
            if "guard" in state.cleanup:
                raise cleanup_errors["guard"]
        guard.close = closing
        return guard

    class Native:
        def contract(self):
            return raw_contract

        def decoder(self):
            return b'{"matrices":[{}]}'

        def publication(self):
            return b"mock-publication"

        def publication_digest(self):
            return b"p" * 32

        def starts(self):
            return (b"zero", b"one")

        def request_id(self):
            return b"r" * 32

        def abort(self):
            events.append("native_aborted")
            if "native" in state.cleanup:
                raise cleanup_errors["native"]

    native = Native()
    source = NS(capacity="capsule", execution_payload_estimate_bytes=1024,
        contract={"vocabulary": 5, "numeric_profile": "mock-only", "local_numeric_profile": "mock-only"},
        source_lock_digest="1" * 64, tokenizer_digest="2" * 64,
        tokenizer=lambda: NS(encode=lambda *args, **kwargs: (0,)),
        tokenizer_descriptor={"add_bos_token": False},
        runtime_contract=lambda *args: raw_contract,
        load_transport=lambda *args, **kwargs: native)
    experiment = NS(name="mock-only", budget=NS(max_input_tokens=1),
        configuration_digest=lambda: "a" * 64,
        pipeline=NS(components={"kernels": NS(params={"threads": 1}, component="pllm/cpu")},
                    digest=lambda: "b" * 64))

    class Peers(real_peers):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def check(self):
            try:
                super().check()
            except ProtocolError as error:
                if state.primary is None:
                    state.primary = error
                    if getattr(state, "break_add_note", False):
                        def broken_note(note):
                            raise RuntimeError("note callback failed")
                        error.add_note = broken_note
                raise

        async def _post(self, index, action, payload=b"", **kwargs):
            role = ("worker_a", "worker_b")[index]
            if action == "manifest":
                return transport.pack({"schema": "pllm.gkm.worker.v1", "role": role,
                    "configuration": "a" * 64, "source_lock": "1" * 64, "tokenizer": "2" * 64,
                    "frame_limit": self.frame_limit, "epoch": "e" * 64})
            if action == "cancel":
                events.append("peer_cancelled")
                return b"ok"
            assert action == "admit"
            if state.mode == "caller_error":
                raise ordinary_error
            if state.mode == "success":
                return transport.pack({"schema": "pllm.gkm.ready.v1", "request": self.request_id,
                    "role": role, "contract_sha256": hashlib.sha256(raw_contract).hexdigest(),
                    "epoch": "e" * 64})
            entered.set()
            if state.mode == "manual_cancel":
                self.cancelled.__self__.set()
            try:
                # A finite safety bound, not a changed production deadline.
                await asyncio.wait_for(asyncio.Future(), timeout=3)
                raise AssertionError("gated admission unexpectedly completed")
            finally:
                events.append("admission_drained")

        def close(self):
            super().close()
            events.append("peers_closed")
            if "peers" in state.cleanup:
                raise cleanup_errors["peers"]

    @contextmanager
    def reservation(*args, **kwargs):
        events.append("reserved")
        try:
            yield
        finally:
            assert all(not peer.io.thread.is_alive() for peer in instances)
            events.append("refunded")

    @contextmanager
    def roles(*args, **kwargs):
        events.append("roles_started")
        try:
            yield NS(worker_connections=lambda: {
                "worker_a": ("http://127.0.0.1:1", "a" * 32),
                "worker_b": ("http://127.0.0.1:2", "b" * 32)})
        finally:
            state.before_roles_error = sys.exception()
            state.before_roles_notes = tuple(getattr(state.before_roles_error, "__notes__", ()))
            events.append("roles_closed")
            if state.mutate:
                state.guard.error = "benchmark stopped: host memory fell below runtime reserve"
                state.guard.failure_observation = {"pressure": "normal", "available_bytes": 0,
                                                   "swap_used_bytes": 99}
            if "roles" in state.cleanup:
                raise cleanup_errors["roles"]

    monkeypatch.setattr(memory, "benchmark_memory", lambda *args, **kwargs: report)
    monkeypatch.setattr(memory, "MemoryWatchdog", watchdog)
    monkeypatch.setattr(servers, "build_roles", roles)
    monkeypatch.setattr(transport, "_Peers", Peers)
    monkeypatch.setattr(transport, "reserve_incremental_memory", reservation)
    monkeypatch.setattr(transport, "_execute_verified_window", lambda *args, **kwargs:
        NS(logits=[b"\0" * 20], executed=(0,), selected=(0,)))
    state.call = lambda: benchmark._delegated_candidate(experiment, ("not-persisted",), source, 0, None)
    state.events, state.instances = events, instances
    state.cleanup_errors, state.ordinary_error = cleanup_errors, ordinary_error
    return state


def assert_drained(state):
    assert state.events[-3:] == ["refunded", "roles_closed", "guard_closed"]
    assert state.events.index("native_aborted") < state.events.index("peers_closed")
    assert state.events.index("peers_closed") < state.events.index("refunded")
    assert not state.guard._thread.is_alive()
    remember = inspect.getclosurevars(state.guard.abort).nonlocals["remember_guard"]
    assert inspect.getclosurevars(remember).nonlocals["guard"] is None
    for peer in state.instances:
        assert not peer.io.thread.is_alive()
        assert peer.io.loop.is_closed() and not asyncio.all_tasks(peer.io.loop)
        assert all(client.is_closed for client in peer.clients)
        assert not any(peer.live)


def test_gated_admission_preserves_error_and_first_guard_snapshot_before_cleanup(admission):
    admission.mutate = True
    with pytest.raises(ProtocolError) as caught:
        admission.call()
    error = caught.value
    assert error is admission.primary
    assert str(error) == "GKM request cancelled or timed out"
    assert error.status_code == 499
    assert len(error.__notes__) == 1
    note = error.__notes__[0]
    assert WARNING in note and "pressure=warning" in note
    assert "available_bytes=512" in note and "swap_used_bytes=1024" in note
    assert "runtime reserve" not in note and len(note.encode()) <= 256
    assert WARNING in "".join(traceback.format_exception(error))
    assert admission.before_roles_error is error
    assert admission.before_roles_notes == (note,)
    assert {frame.name for frame in traceback.extract_tb(error.__traceback__)} >= {"owned", "cancellation", "check"}
    assert admission.events.index("admission_drained") < admission.events.index("native_aborted")
    assert admission.events.count("peer_cancelled") == 1
    assert_drained(admission)


@pytest.mark.parametrize("mode", ["manual_cancel", "caller_error"])
def test_no_guard_trigger_invents_no_note(admission, mode):
    admission.mode = mode
    with pytest.raises(ProtocolError) as caught:
        admission.call()
    assert caught.value is (admission.primary if mode == "manual_cancel" else admission.ordinary_error)
    assert not getattr(caught.value, "__notes__", None)
    assert admission.guard.error is admission.guard.failure_observation is None
    assert_drained(admission)


def test_success_result_and_cleanup_unchanged(admission):
    admission.mode = "success"
    metadata, actual, decoded, ids, selected = admission.call()
    assert metadata["execution"] == "two-worker-request-end-verified"
    assert len(actual) == 1 and actual[0].tolist() == [0] * 5
    assert decoded == [] and ids == ((0,),) and selected == [()]
    assert admission.guard.error is admission.guard.failure_observation is None
    assert_drained(admission)


@pytest.mark.parametrize("sites", [{"native"}, {"peers"}, {"roles"}, {"guard"},
                                  {"native", "peers", "roles", "guard"}])
def test_cleanup_precedence_chain_and_first_trigger_unchanged(admission, sites):
    admission.cleanup = sites
    admission.mutate = True
    with pytest.raises(RuntimeError) as caught:
        admission.call()
    order = [site for site in ("native", "peers", "roles", "guard") if site in sites]
    assert caught.value is admission.cleanup_errors[order[-1]]
    assert str(caught.value) == order[-1] + " cleanup failed"
    chain, error = [], caught.value
    while error is not None:
        chain.append(error)
        error = error.__context__
    assert chain[:len(order)] == [admission.cleanup_errors[site] for site in reversed(order)]
    assert admission.primary in chain
    assert WARNING in caught.value.__notes__[0] and "available_bytes=512" in caught.value.__notes__[0]
    assert_drained(admission)


def test_annotation_failure_cannot_replace_primary_or_prevent_drain(admission, monkeypatch):
    def broken(*args):
        raise RuntimeError("diagnostic failure")
    monkeypatch.setattr(benchmark, "_memory_watchdog_note", broken)
    with pytest.raises(ProtocolError) as caught:
        admission.call()
    assert caught.value is admission.primary and not getattr(caught.value, "__notes__", None)
    assert_drained(admission)


def test_add_note_failure_preserves_primary_and_drain(admission):
    admission.break_add_note = True
    with pytest.raises(ProtocolError) as caught:
        admission.call()
    assert caught.value is admission.primary and not getattr(caught.value, "__notes__", None)
    assert caught.value.__context__ is None
    assert_drained(admission)


@pytest.mark.parametrize("site", ["native", "peers", "roles", "guard"])
def test_cleanup_without_trigger_gets_no_invented_guard_reason(admission, site):
    admission.mode, admission.cleanup = "success", {site}
    with pytest.raises(RuntimeError) as caught:
        admission.call()
    assert caught.value is admission.cleanup_errors[site]
    assert not getattr(caught.value, "__notes__", None)
    assert_drained(admission)


@pytest.mark.parametrize("reason", [WARNING,
    "benchmark stopped: macOS memory pressure is critical",
    "benchmark stopped: host memory fell below runtime reserve",
    "benchmark stopped: host swap grew by more than 256 MiB",
    "/private/secret/prompt-output\n" * 1000, ""])
def test_note_bound_sanitizes_only_cached_builtin_public_data(reason):
    guard = NS(error=reason, failure_observation={"pressure": "critical",
        "available_bytes": (1 << 63) - 1, "swap_used_bytes": (1 << 63) - 1})
    note = benchmark._memory_watchdog_note(guard)
    assert len(note.encode("ascii")) <= 256
    assert "/private" not in note and "prompt-output" not in note and "\n" not in note
    assert "available_bytes=9223372036854775807" in note


def test_hostile_observation_never_runs_user_formatting_or_invents_reason():
    class Hostile:
        def __str__(self):
            raise AssertionError("stringification")

        def __eq__(self, other):
            raise AssertionError("comparison")

    guard = NS(error=Hostile(), failure_observation={
        "pressure": Hostile(), "available_bytes": True, "swap_used_bytes": -1})
    assert benchmark._memory_watchdog_note(guard) is None


@pytest.mark.parametrize("triggered", [True, False])
def test_record_cli_retains_sanitized_note_in_existing_limitations_without_scoring(
        cli_lane, monkeypatch, tmp_path, capsys, triggered):
    from pllm.cli import main
    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking
    guard = NS(error=WARNING if triggered else None,
               failure_observation=dict(OBSERVATION) if triggered else None,
               start=lambda: None, close=lambda: None, check=lambda: None)
    monkeypatch.setattr(memory, "MemoryWatchdog", lambda *args: guard)
    error = sys.modules["pllm.runtime.client"].ProtocolError("synthetic admission failed")
    cli_lane.execute.side_effect = error
    key, teacher = tmp_path / "synthetic.key", tmp_path / "synthetic-teacher.json"
    key.write_bytes(b"k" * 32)
    teacher.write_text(json.dumps([[2, 2, 2]] * 12))
    argv = documented_argv(cli_lane.output) + ["--outcomes", "record", "--cohort-key-file", str(key),
        "--cohort-key-id", "guard-note-test", "--continuations-file", str(teacher)]
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 6
    report = read_report(cli_lane.output.read_bytes())  # Existing strict schema.
    if triggered:
        assert error.__notes__[0] in report["limitations"]
    else:
        assert not getattr(error, "__notes__", None)
        assert not any("memory watchdog observed:" in text for text in report["limitations"])
    assert report["terminal_state"] == "execution_failed" and report["reference_status"] == "unavailable"
    for candidate in report["candidates"]:
        section = candidate["all_positions"]
        assert section["coverage"] == {"total": 48, "accepted": 0, "rejected": 0, "failed": 48}
        assert section["sample_count"] == 0
        assert all(section[name] is None for name in ("top1_agreement", "top_k_recall", "max_abs_logit_error"))
    with pytest.raises(ValueError):
        require_complete_for_ranking(report)
    cli_lane.streamed.assert_not_called()
    assert cli_lane.execute.call_count == 1
    cli_lane.topology.__exit__.assert_called_once()
    capsys.readouterr()
