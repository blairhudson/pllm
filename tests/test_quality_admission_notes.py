"""Synthetic cached admission notes; no native/framework import or model run."""
from copy import deepcopy
from importlib.abc import MetaPathFinder
import json
import runpy
import sys
import threading
import traceback
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from test_paper_quality_cli import lane as cli_lane, quality as quality


@pytest.fixture(scope="module", autouse=True)
def no_execution_imports():
    monkeypatch = pytest.MonkeyPatch()
    class NoExecution(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "transformers"} or fullname == "pllm._native":
                raise AssertionError("real execution import forbidden: " + fullname)

    before = set(sys.modules)
    native = ModuleType("pllm._native")
    native.NumericDomainRejected = type("NumericDomainRejected", (ValueError,), {})
    monkeypatch.setitem(sys.modules, "pllm._native", native)
    monkeypatch.setattr(sys, "meta_path", [NoExecution(), *sys.meta_path])
    from pllm.configuration import Experiment
    from pllm.profiles import resolve_runtime_composition
    from pllm.runtime.benchmark_memory import MemoryWatchdog
    monkeypatch.setattr(Experiment, "resolve", lambda self: resolve_runtime_composition(self.pipeline))
    yield NS(watchdog=MemoryWatchdog)
    monkeypatch.undo()
    # Do not leave fake execution-module bindings cached for other test files.
    for name in sorted(set(sys.modules) - before, reverse=True):
        if name in {"pllm", "examples"} or name.startswith(("pllm.", "examples.")):
            module = sys.modules.pop(name, None)
            parent_name, _, child = name.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and vars(parent).get(child) is module:
                vars(parent).pop(child, None)


def quote():
    # Exact synthetic integers deliberately unrelated to historical r1 values.
    return {"admitted": False, "requested_backend": "native",
        "host": {"admission_budget_bytes": 941302987, "available": 1941302987},
        "estimate": {"allocation_slack_percent": 25, "private_owner": "PRIVATE_PATH_OR_PROMPT"},
        "candidates": {"native": {"admitted": False, "required_host_bytes": 963526091,
            "reasons": ["synthetic rounded memory refusal"]}}}


@pytest.fixture
def admission(cli_lane, monkeypatch):
    from pllm.runtime import benchmark_memory as memory
    cached = quote()
    error = memory.BenchmarkMemoryError("synthetic rounded preflight refusal", cached)
    error.code = "QUALITY_MEMORY"
    memory.benchmark_memory.return_value = cached
    memory.require_admission.side_effect = error
    monkeypatch.setattr(memory, "host_memory", Mock(side_effect=AssertionError("diagnostic resampling")))
    return NS(lane=cli_lane, memory=memory, report=cached, error=error)


def invoke(state, *, outcomes=None):
    lane = state.lane
    return lane.benchmark._delegated_candidate(lane.quality.experiment(), lane.prompts,
        lane.source, 3, None, outcomes=outcomes)


def assert_no_started_owners(state):
    state.lane.roles.assert_not_called()
    state.lane.execute.assert_not_called()
    state.memory.MemoryWatchdog.assert_not_called()
    state.memory.host_memory.assert_not_called()


def test_exact_cached_bytes_preserve_error_identity_string_code_and_traceback(admission):
    original = deepcopy(admission.report)
    with pytest.raises(type(admission.error)) as caught:
        invoke(admission)
    assert caught.value is admission.error and caught.value.code == "QUALITY_MEMORY"
    assert str(caught.value) == "synthetic rounded preflight refusal"
    note, = caught.value.__notes__
    assert note == ("quality memory admission: backend=native; estimated_peak_bytes=963526091; "
        "safe_headroom_bytes=941302987; available_bytes=1941302987; margin_percent=25")
    rendered = "".join(traceback.format_exception(caught.value))
    assert note in rendered and "_delegated_candidate" in rendered
    assert "PRIVATE" not in note and len(note.encode("ascii")) <= 256
    assert admission.report == original
    assert_no_started_owners(admission)


def test_record_cli_keeps_note_in_existing_limitations_without_scoring(admission, tmp_path, capsys):
    from pllm.cli import main
    from pllm.runtime.quality_outcomes import read_report, require_complete_for_ranking
    from test_paper_quality_cli import documented_argv
    key, teacher = tmp_path / "key", tmp_path / "teacher.json"
    key.write_bytes(b"k" * 32)
    teacher.write_text(json.dumps([[2, 2, 2]] * 12))
    argv = documented_argv(admission.lane.output) + ["--outcomes", "record", "--cohort-key-file", str(key),
        "--cohort-key-id", "admission-note-test", "--continuations-file", str(teacher)]
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 6
    report = read_report(admission.lane.output.read_bytes())
    assert admission.error.__notes__[0] in report["limitations"]
    assert report["terminal_state"] == "execution_failed" and report["reference_status"] == "unavailable"
    for candidate in report["candidates"]:
        section = candidate["all_positions"]
        assert section["coverage"] == {"total": 48, "accepted": 0, "rejected": 0, "failed": 48}
        assert section["sample_count"] == 0 and section["top1_agreement"] is None
    with pytest.raises(ValueError):
        require_complete_for_ranking(report)
    assert "PRIVATE" not in json.dumps(report)
    assert_no_started_owners(admission)
    capsys.readouterr()


@pytest.mark.parametrize("field", ["required", "budget"])
@pytest.mark.parametrize("value", [True, -1, 1 << 64, 1.0, "PRIVATE_VALUE", None])
def test_malformed_required_values_add_no_invented_note(admission, field, value):
    owner, key = ((admission.report["candidates"]["native"], "required_host_bytes") if field == "required"
                  else (admission.report["host"], "admission_budget_bytes"))
    owner[key] = value
    with pytest.raises(type(admission.error)) as caught:
        invoke(admission)
    assert caught.value is admission.error and not getattr(caught.value, "__notes__", None)
    assert_no_started_owners(admission)


def test_fixed_integer_maxima_bound_without_owner_repr(admission):
    class Hostile:
        def __str__(self):
            raise AssertionError("private formatting")
        __repr__ = __str__

    cached = admission.report
    cached["candidates"]["native"]["required_host_bytes"] = (1 << 63) - 1
    cached["host"].update(admission_budget_bytes=(1 << 63) - 1, available=(1 << 63) - 1)
    cached["estimate"].update(allocation_slack_percent=100, private_owner=Hostile())
    with pytest.raises(type(admission.error)):
        invoke(admission)
    note, = admission.error.__notes__
    assert len(note.encode("ascii")) <= 256 and note.count(str((1 << 63) - 1)) == 3
    cached["host"]["available"] = cached["estimate"]["allocation_slack_percent"] = Hostile()
    note = admission.lane.benchmark._memory_admission_note(cached)
    assert "available_bytes" not in note and "margin_percent" not in note


def test_subclasses_never_run_custom_mapping_or_integer_formatting(admission):
    class Mapping(dict):
        def get(self, *args):
            raise AssertionError("custom mapping access")

    class Integer(int):
        def __format__(self, spec):
            raise AssertionError("custom integer formatting")

    formatter = admission.lane.benchmark._memory_admission_note
    assert formatter(Mapping(admission.report)) is None
    admission.report["host"] = Mapping(admission.report["host"])
    assert formatter(admission.report) is None
    admission.report["host"] = dict(admission.report["host"])
    admission.report["candidates"]["native"]["required_host_bytes"] = Integer(3)
    assert formatter(admission.report) is None


def test_quote_note_does_not_claim_which_preflight_guard_rejected(admission):
    admission.report["host"]["admission_budget_bytes"] = 2000000000
    admission.report["candidates"]["native"]["reasons"] = ["synthetic disk rejection"]
    with pytest.raises(type(admission.error)):
        invoke(admission)
    note, = admission.error.__notes__
    assert "safe_headroom_bytes=2000000000" in note
    assert all(word not in note for word in ("exceeds", "pressure", "disk", "swap"))


@pytest.mark.parametrize("change", ["missing_host", "wrong_backend", "admitted", "wrong_native", "foreign_report"])
def test_unavailable_or_unrelated_report_never_invents_native_quote(admission, change):
    cached = admission.report
    if change == "missing_host":
        cached["host"] = None
    elif change == "wrong_backend":
        cached["requested_backend"] = "auto"
    elif change == "admitted":
        cached["admitted"] = True
    elif change == "wrong_native":
        cached["candidates"]["native"] = []
    else:
        admission.error.report = quote()
    with pytest.raises(type(admission.error)):
        invoke(admission)
    assert not getattr(admission.error, "__notes__", None)


def test_success_keeps_original_admission_calls_and_adds_no_failure_note(admission):
    admission.report["admitted"] = True
    admission.report["candidates"]["native"]["admitted"] = True
    admission.memory.require_admission.side_effect = None
    # Direct success, not a scored quality claim.
    result = invoke(admission)
    assert result[0]["memory_admission"] is admission.report
    admission.memory.require_admission.assert_called_once_with(admission.report)
    admission.memory.MemoryWatchdog.assert_called_once()
    admission.lane.topology.__exit__.assert_called_once()
    assert not getattr(admission.error, "__notes__", None)


def test_non_memory_error_identity_unannotated(admission):
    error = RuntimeError("caller refusal")
    admission.memory.require_admission.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        invoke(admission)
    assert caught.value is error and not getattr(error, "__notes__", None)
    assert_no_started_owners(admission)


def test_first_record_note_and_duplicate_exception_note_preserved(admission):
    owner = NS(_memory_guard_note="memory watchdog observed: pressure=warning")
    note = admission.lane.benchmark._memory_admission_note(admission.report)
    admission.error.add_note(note)
    with pytest.raises(type(admission.error)):
        invoke(admission, outcomes=owner)
    assert owner._memory_guard_note == "memory watchdog observed: pressure=warning"
    assert admission.error.__notes__ == [note]


@pytest.mark.parametrize("site", ["formatter", "add_note", "record_owner"])
def test_annotation_failures_never_replace_original_or_start_cleanup_owners(admission, monkeypatch, site):
    if site == "formatter":
        monkeypatch.setattr(admission.lane.benchmark, "_memory_admission_note", Mock(side_effect=KeyboardInterrupt()))
        owner = NS()
    elif site == "add_note":
        admission.error.add_note = Mock(side_effect=RuntimeError("diagnostics failed"))
        owner = NS()
    else:
        class Owner:
            def __getattribute__(self, name):
                raise RuntimeError("diagnostics owner failed")
        owner = Owner()
    with pytest.raises(type(admission.error)) as caught:
        invoke(admission, outcomes=owner)
    assert caught.value is admission.error and str(caught.value) == "synthetic rounded preflight refusal"
    assert_no_started_owners(admission)


def invoke_cli(lane, monkeypatch, route="console", output_format="json"):
    from test_paper_quality_cli import documented_argv
    argv = documented_argv(lane.output)
    argv[argv.index("--format") + 1] = output_format
    if route == "module":
        monkeypatch.setattr(sys, "argv", ["pllm", *argv])
        runpy.run_module("pllm", run_name="__main__")
    else:
        from pllm.cli import main
        main(argv)


@pytest.mark.parametrize("route", ["console", "module"])
@pytest.mark.parametrize("output_format", ["human", "json", "jsonl"])
def test_normal_cli_byte_exact_refusal_preserves_objects_and_abort_no_report(
        admission, monkeypatch, capsys, route, output_format):
    from pllm._cli import app
    from pllm._cli.output import emit_error
    seen = []

    def render(error, *args, **kwargs):
        seen.append(error)
        return emit_error(error, *args, **kwargs)

    monkeypatch.setattr(app, "emit_error", render)
    with pytest.raises(SystemExit) as stopped:
        invoke_cli(admission.lane, monkeypatch, route, output_format)
    assert stopped.value.code == 6 and not admission.lane.output.exists()
    wrapper, = seen
    assert wrapper.__cause__ is admission.error and wrapper.code == "QUALITY_MEMORY"
    assert str(wrapper) == wrapper.message == str(admission.error) == "synthetic rounded preflight refusal"
    note, = admission.error.__notes__
    captured = capsys.readouterr()
    assert not captured.out
    if output_format == "human":
        assert note in captured.err
    else:
        payload = json.loads(captured.err)
        assert payload["schema_version"] == "pllm.cli.error.v1"
        assert payload["error"] == {"code": "QUALITY_MEMORY", "exit_code": 6, "stage": "runtime",
            "details": {}, "message": str(admission.error) + "\n" + note}
    assert_no_started_owners(admission)


@pytest.mark.parametrize("route", ["console", "module"])
@pytest.mark.parametrize("output_format", ["human", "json"])
def test_normal_cli_gated_admission_watchdog_snapshot_and_cleanup(
        cli_lane, no_execution_imports, monkeypatch, capsys, route, output_format):
    from pllm.runtime import benchmark_memory as memory
    gate = threading.Event()
    normal = NS(pressure="normal", available=4096, swap_used=0)
    failed = NS(pressure="warning", available=512, swap_used=1024)
    report = {"admitted": True, "host": {"available": 4096, "swap_used": 0, "reserve_bytes": 0},
              "runtime_floor_bytes": 256}
    state, io = NS(), []
    memory.benchmark_memory.return_value = report

    def watchdog(report, abort):
        guard = no_execution_imports.watchdog(report, abort,
            sample=lambda: failed if gate.is_set() else normal)
        state.guard = guard
        # Exercise real check/cache semantics without starting a periodic thread.
        guard.start = guard.check
        return guard

    def admission(*args, **kwargs):
        try:
            io.append("admit_entered")
            gate.set()
            state.guard.check()
        except memory.BenchmarkMemoryError as error:
            state.error = error
            raise
        finally:
            io.append("admit_drained")

    memory.MemoryWatchdog.side_effect = watchdog
    cli_lane.execute.side_effect = admission
    with pytest.raises(SystemExit) as stopped:
        invoke_cli(cli_lane, monkeypatch, route, output_format)
    assert stopped.value.code == 6 and not cli_lane.output.exists()
    assert io == ["admit_entered", "admit_drained"]
    cli_lane.topology.__exit__.assert_called_once()
    assert state.guard._stop.is_set() and not state.guard._thread.is_alive()
    assert state.guard.failure_observation == {"pressure": "warning", "available_bytes": 512, "swap_used_bytes": 1024}
    assert str(state.error) == "benchmark stopped: macOS memory pressure is warning"
    note, = state.error.__notes__
    captured = capsys.readouterr()
    message = json.loads(captured.err)["error"]["message"] if output_format == "json" else captured.err
    assert note in message and "available_bytes=512" in message and "swap_used_bytes=1024" in message


@pytest.mark.parametrize("malicious", [
    "PRIVATE_PROMPT", "memory watchdog observed: PRIVATE_PROMPT",
    "memory watchdog observed: pressure=warning\nPRIVATE_PROMPT",
    "memory watchdog observed: pressure=warning; available_bytes=9999999999999999999",
    "quality memory admission: backend=native; estimated_peak_bytes=1; safe_headroom_bytes=1; PRIVATE_PROMPT=1",
    "quality memory admission: backend=docker; estimated_peak_bytes=1; safe_headroom_bytes=1",
    "memory watchdog observed: " + "x" * 257,
    "memory watchdog observed: pressure=warning\x1b[31m",
])
@pytest.mark.parametrize("output_format", ["human", "json"])
def test_normal_cli_does_not_print_unrelated_spoofed_or_oversized_notes(
        admission, monkeypatch, capsys, malicious, output_format):
    admission.error.add_note(malicious)
    with pytest.raises(SystemExit) as stopped:
        invoke_cli(admission.lane, monkeypatch, output_format=output_format)
    assert stopped.value.code == 6
    captured = capsys.readouterr()
    message = json.loads(captured.err)["error"]["message"] if output_format == "json" else captured.err
    assert malicious not in message and "PRIVATE_PROMPT" not in message
    assert "estimated_peak_bytes=963526091" in message
    assert admission.error.__notes__[0] == malicious  # Original notes/identity are not mutated.


def test_renderer_two_prefix_cap_tail_bound_and_unchanged_schema(admission, capsys):
    from pllm._cli.errors import RuntimeFailure
    from pllm._cli.output import emit_error
    note = admission.lane.benchmark._memory_admission_note(admission.report)
    guard = "memory watchdog observed: pressure=warning; available_bytes=512; swap_used_bytes=1024"
    source = admission.error
    source.__notes__ = ["PRIVATE_PROMPT"] * 128 + [note, guard]
    wrapper = RuntimeFailure("QUALITY_MEMORY", "original")
    wrapper.__cause__ = source
    wrapper.add_note(guard)
    emit_error(wrapper, "json", quality_notes=True)
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["message"] == "original\n" + note + "\n" + guard
    assert str(wrapper) == wrapper.message == "original" and wrapper.__cause__ is source
    emit_error(wrapper, "json")
    assert json.loads(capsys.readouterr().err)["error"]["message"] == "original"
    wrapper.code = "OTHER_ERROR"
    emit_error(wrapper, "json", quality_notes=True)
    assert json.loads(capsys.readouterr().err)["error"]["message"] == "original"


@pytest.mark.parametrize("site", ["note_collection", "extended_renderer", "all_rendering"])
def test_cli_render_failure_preserves_exit_original_cause_and_no_report(
        admission, monkeypatch, capsys, site):
    from pllm._cli import app, output
    seen = []
    real = app.emit_error
    if site == "note_collection":
        monkeypatch.setattr(output, "_quality_memory_notes", Mock(side_effect=RuntimeError("render failed")))

    def emit(error, *args, **kwargs):
        seen.append(error)
        if site == "all_rendering" or site == "extended_renderer" and kwargs:
            raise RuntimeError("render failed")
        return real(error, *args, **kwargs)

    monkeypatch.setattr(app, "emit_error", emit)
    with pytest.raises(SystemExit) as stopped:
        invoke_cli(admission.lane, monkeypatch)
    assert stopped.value.code == 6 and not admission.lane.output.exists()
    assert all(error.__cause__ is admission.error for error in seen)
    assert str(admission.error) == "synthetic rounded preflight refusal"
    if site != "all_rendering":
        assert json.loads(capsys.readouterr().err)["error"]["message"] == str(admission.error)


@pytest.mark.parametrize("route", ["console", "module"])
def test_quality_failed_route_keeps_original_cause_and_safe_cached_note(cli_lane, monkeypatch, capsys, route):
    from pllm.runtime import benchmark_memory as memory
    from pllm._cli import app
    from pllm._cli.output import emit_error
    error, seen = ValueError("original execution rejection"), []
    cli_lane.execute.side_effect = error
    memory.MemoryWatchdog.return_value = NS(
        error="benchmark stopped: macOS memory pressure is warning",
        failure_observation={"pressure": "warning", "available_bytes": 512, "swap_used_bytes": 1024},
        start=lambda: None, check=lambda: None, close=lambda: None)

    def render(wrapper, *args, **kwargs):
        seen.append(wrapper)
        return emit_error(wrapper, *args, **kwargs)

    monkeypatch.setattr(app, "emit_error", render)
    with pytest.raises(SystemExit) as stopped:
        invoke_cli(cli_lane, monkeypatch, route)
    assert stopped.value.code == 6 and not cli_lane.output.exists()
    wrapper, = seen
    assert wrapper.__cause__ is error and wrapper.code == "QUALITY_FAILED"
    assert str(wrapper) == str(error) == "original execution rejection"
    assert error.__notes__[0] in json.loads(capsys.readouterr().err)["error"]["message"]


def test_all_existing_watchdog_formatter_shapes_are_accepted(cli_lane):
    from pllm._cli.output import _quality_memory_note
    reasons = (None, "benchmark stopped: macOS memory pressure is warning",
        "benchmark stopped: macOS memory pressure is critical",
        "benchmark stopped: host memory fell below runtime reserve",
        "benchmark stopped: host swap grew by more than 256 MiB", "PRIVATE_PROMPT")
    for reason in reasons:
        for pressure in (None, "normal", "warning", "critical"):
            for available in (None, (1 << 63) - 1):
                for swap in (None, (1 << 63) - 1):
                    note = cli_lane.benchmark._memory_watchdog_note(NS(error=reason,
                        failure_observation={"pressure": pressure, "available_bytes": available,
                                             "swap_used_bytes": swap}))
                    if note is not None:
                        assert _quality_memory_note(note)


def test_note_subclasses_and_hostile_access_never_format_private_objects(cli_lane):
    from pllm._cli.errors import RuntimeFailure
    from pllm._cli.output import _quality_memory_note, _quality_memory_notes

    class Text(str):
        def __str__(self):
            raise AssertionError("private string formatting")

    class Notes(list):
        def __getitem__(self, index):
            raise AssertionError("private sequence traversal")

    class Error(RuntimeError):
        @property
        def __notes__(self):
            raise RuntimeError("private notes access")

    assert not _quality_memory_note(Text("memory watchdog observed: pressure=warning"))
    assert not _quality_memory_note("memory watchdog observed: available_bytes=١")
    wrapper = RuntimeFailure("QUALITY_MEMORY", "original")
    wrapper.__cause__ = Error("original")
    assert _quality_memory_notes(wrapper) == ()
    wrapper.__cause__ = RuntimeError("original")
    wrapper.__cause__.__notes__ = Notes(["memory watchdog observed: pressure=warning"])
    assert _quality_memory_notes(wrapper) == ()


@pytest.mark.parametrize("output_format", ["human", "json"])
def test_parse_failure_before_namespace_keeps_original_exit(cli_lane, capsys, output_format):
    from pllm.cli import main
    with pytest.raises(SystemExit) as stopped:
        main(["--format", output_format, "--definitely-not-an-option"])
    assert stopped.value.code == 2
    stderr = capsys.readouterr().err
    assert "CLI_USAGE" in stderr and "UnboundLocalError" not in stderr


def test_nonquality_dispatch_does_not_opt_in_to_notes(cli_lane, monkeypatch, capsys):
    from pllm.cli import main
    from pllm._cli import app
    from pllm._cli.errors import RuntimeFailure
    cause = RuntimeError("original")
    cause.add_note("memory watchdog observed: pressure=warning")
    error = RuntimeFailure("QUALITY_MEMORY", "original")
    error.__cause__ = cause
    monkeypatch.setattr(app, "_components", Mock(side_effect=error))
    with pytest.raises(SystemExit) as stopped:
        main(["components", "list", "--format", "json"])
    assert stopped.value.code == 6
    assert json.loads(capsys.readouterr().err)["error"]["message"] == "original"
