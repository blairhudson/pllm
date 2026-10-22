"""In-memory source/native doubles only; no checkpoints, roles or native execution."""
from __future__ import annotations

from contextlib import contextmanager
import asyncio
import hashlib
import json
import struct
import threading
from types import SimpleNamespace as NS

import pytest

from pllm.runtime import benchmark_failure as diagnostics
from pllm.runtime import fss_source as source
from pllm.runtime import servers


METADATA_PHASES = (
    "source_resolution", "source_config", "source_geometry", "source_inventory",
    "source_lock", "source_contract", "source_headers", "source_tokenizer", "source_recheck",
)
IMPORT_PHASES = (
    "qualification_metadata", "source_contract", "source_lock", "source_headers",
    "qualification_import", "source_weights", "source_rotary", "source_recheck",
    "qualification_seal", "source_manifest",
)
SECRET = "private-prompt-token-id-credential-/checkpoint/path"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def importer(tmp_path, monkeypatch):
    """Keep real source control flow, hashes and manifest checks; replace all I/O."""
    state = NS(events=[], fail=None, counts={}, builder=None, program=None)

    def event(name):
        state.events.append(name)
        state.counts[name] = state.counts.get(name, 0) + 1
        if state.fail == (name, state.counts[name]):
            raise ValueError(SECRET)

    config = canonical({"model_type": "qwen3"})
    tokenizer = b"original public tokenizer bytes"
    model = NS(kind="safetensors", source=str(tmp_path))
    experiment = NS(budget=NS(max_input_tokens=2, max_new_tokens=1))
    values = 3 * source._FRAME // 4  # Three frames, still one phase breadcrumb.
    record = {"source_shapes": {"public.weight": [1, values]},
              "source_float32_bytes": values * 4, "numeric_profile": "fixture-profile",
              "vocabulary": 32}
    contract = canonical(record)
    identities = (("config.json", (1,)), ("tokenizer.json", (2,)),
                  ("model.safetensors", (3,)))
    specs = (("public.weight", ("model.safetensors", (1, values), "F32", 0, values * 4)),)
    lock = {"files": [{"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
                      for name, raw in (("config.json", config), ("tokenizer.json", tokenizer))]}

    def bounded(path, _limit):
        event("read_config" if path.name == "config.json" else "read_tokenizer")
        return config if path.name == "config.json" else tokenizer

    def locked(*_args):
        event("lock")
        return "a" * 64, "b" * 64, lock

    def store(*_args):
        event("headers")
        return NS(clear_cache=lambda: event("clear_cache")), specs

    def source_contract(*_args):
        event("contract")
        return contract, record

    def read_range(_path, _identity, _offset, length, **_kwargs):
        event("read_frame")
        return b"\0" * length

    class Program:
        closed = False

        def manifest(self, raw):
            event("manifest")
            return canonical({"schema": "pllm.fss.channel.program.v1",
                              "contract": hashlib.sha256(raw).hexdigest(),
                              "input_digest": state.builder.digest.hexdigest(),
                              "profile": record["numeric_profile"], "source": "c" * 64,
                              "program": "d" * 64, "initial": 2, "selections": 1,
                              "vocabulary": 32})

        def close(self):
            self.closed = True
            event("program_close")

    class Builder:
        closed = False

        def __init__(self, *_args, **_kwargs):
            event("builder")
            state.builder = self
            self.digest = hashlib.sha256(b"pllm.fss.original-ieee-stream.v1\0")

        def append(self, name, _dtype, begin, raw):
            event("append")
            if begin == 0:
                self.digest.update(struct.pack("<Q", len(name)) + name.encode() + b"\0")
                self.digest.update(struct.pack("<QQQ", 2, 1, values))
            self.digest.update(raw)

        def append_rotary(self, operation, _begin, cosine, sine):
            event("rotary")
            self.digest.update(struct.pack("<Q", len(operation)) + operation.encode())
            self.digest.update(struct.pack("<QQQ", 10000, 1, 2))
            for i in range(0, len(cosine), 4):
                self.digest.update(cosine[i:i+4] + sine[i:i+4])

        def seal(self):
            event("seal")
            state.program = Program()
            return state.program

        def close(self):
            self.closed = True
            event("builder_close")

    monkeypatch.setattr(source, "_experiment", lambda _exp:
                        (b"experiment", model, "fixture-profile", "capsule"))
    monkeypatch.setattr(source, "_geometry", lambda *_args: event("geometry"))
    monkeypatch.setattr(source, "_files", lambda *_args: event("inventory") or identities)
    monkeypatch.setattr(source, "_unchanged", lambda *_args: event("unchanged"))
    monkeypatch.setattr(source, "_locked", locked)
    monkeypatch.setattr(source, "_store", store)
    monkeypatch.setattr(source, "_contract", source_contract)
    monkeypatch.setattr(source, "_tokenizer", lambda *_args:
                        event("tokenizer") or (tokenizer, canonical({"vocab_size": 32})))
    monkeypatch.setattr(source, "_rotary_specs", lambda _record: [("public.rope", [1, 2], 10000)])
    monkeypatch.setattr(source.io, "_identity", lambda _path: (1,))
    monkeypatch.setattr(source.io, "_read_bounded", bounded)
    monkeypatch.setattr(source.io, "_read_range", read_range)
    monkeypatch.setattr(source._native, "FssSourceBuilder", Builder, raising=False)
    state.experiment = experiment
    state.metadata = source.resolve_client_source(experiment)
    state.events.clear()
    state.counts.clear()
    return state


def test_metadata_breadcrumbs_precede_work_and_contain_only_fixed_names(importer):
    phases = []

    def lifecycle(phase):
        phases.append(phase)
        importer.events.append(phase)

    resolved = source.resolve_client_source(importer.experiment, lifecycle=lifecycle)
    assert resolved == importer.metadata
    assert tuple(phases) == METADATA_PHASES
    for phase, operation in (("source_config", "read_config"),
                             ("source_geometry", "geometry"),
                             ("source_inventory", "inventory"),
                             ("source_lock", "lock"), ("source_contract", "contract"),
                             ("source_headers", "headers"), ("source_tokenizer", "tokenizer")):
        assert importer.events.index(phase) < importer.events.index(operation)
    assert importer.events.index("source_recheck") < len(importer.events) - 1
    assert SECRET not in json.dumps(phases)


@pytest.mark.parametrize("operation,ordinal,phase", [
    ("read_config", 1, "source_config"), ("geometry", 1, "source_geometry"),
    ("tokenizer", 1, "source_tokenizer"), ("lock", 2, "source_recheck"),
])
def test_metadata_failure_keeps_last_prework_phase(importer, operation, ordinal, phase):
    phases = []
    importer.fail = operation, ordinal
    with pytest.raises(ValueError, match=SECRET):
        source.resolve_client_source(importer.experiment, lifecycle=phases.append)
    assert phases[-1] == phase
    assert importer.builder is importer.program is None


def test_resolution_phase_precedes_decoder_metadata_reservation(importer, monkeypatch):
    phases = []
    importer.experiment.pipeline = NS(components={"linear": NS(params={"capacity": "decoder"})})
    monkeypatch.setattr(source, "Experiment", type(importer.experiment))

    @contextmanager
    def reservation():
        importer.events.append("reservation")
        raise ValueError("mock metadata reservation failure")
        yield  # pragma: no cover

    monkeypatch.setattr(source.tokenizer_admission, "reserve_decoder_metadata", reservation)
    with pytest.raises(ValueError, match="mock metadata reservation failure"):
        source.resolve_client_source(importer.experiment, lifecycle=phases.append)
    assert phases == ["source_resolution"]
    assert importer.events == ["reservation"]


@pytest.mark.parametrize("provided", [False, True])
def test_import_breadcrumbs_are_per_phase_not_per_tensor_frame(importer, provided):
    phases = []
    metadata = importer.metadata
    with source.load_role_source(
        importer.experiment, expected_pins=metadata.snapshot_pins,
        source=metadata if provided else None, lifecycle=phases.append,
    ) as loaded:
        assert loaded.manifest["input_digest"] == importer.builder.digest.hexdigest()
        assert loaded.source == metadata
        assert not importer.program.closed
    expected = IMPORT_PHASES if provided else (
        IMPORT_PHASES[:1] + METADATA_PHASES + IMPORT_PHASES[1:])
    assert tuple(phases) == expected
    assert importer.events.count("append") == 3
    assert phases.count("source_weights") == phases.count("source_rotary") == 1
    assert importer.builder.closed and importer.program.closed


@pytest.mark.parametrize("operation,ordinal,phase", [
    ("contract", 1, "source_contract"), ("lock", 1, "source_lock"),
    ("headers", 1, "source_headers"), ("builder", 1, "qualification_import"),
    ("append", 2, "source_weights"), ("rotary", 1, "source_rotary"),
    ("lock", 2, "source_recheck"), ("seal", 1, "qualification_seal"),
    ("manifest", 1, "source_manifest"),
])
def test_failure_notifies_before_owned_cleanup(importer, operation, ordinal, phase):
    phases, failures = [], []
    importer.fail = operation, ordinal

    def failure():
        failures.append((phases[-1], tuple(importer.events)))

    with pytest.raises(ValueError, match=SECRET):
        source.load_role_source(importer.experiment, expected_pins=importer.metadata.snapshot_pins,
                                source=importer.metadata, lifecycle=phases.append, on_failure=failure)
    assert len(failures) == 1 and failures[0][0] == phase
    assert not {"builder_close", "program_close", "clear_cache"}.intersection(failures[0][1])
    if importer.builder is not None:
        assert importer.builder.closed
    if importer.program is not None:
        assert importer.program.closed


@pytest.mark.parametrize("early", [False, True])
def test_cancellation_notifies_before_cleanup_without_advancing_phase(importer, early):
    phases, failures = [], []

    def failure():
        failures.append((phases[-1], tuple(importer.events)))

    with pytest.raises(source.FssSourceError, match="cancelled"):
        source.load_role_source(
            importer.experiment, expected_pins=importer.metadata.snapshot_pins,
            source=importer.metadata, lifecycle=phases.append, on_failure=failure,
            cancelled=lambda: early or "append" in importer.events,
        )
    assert len(failures) == 1
    assert failures[0][0] == ("qualification_metadata" if early else "source_weights")
    assert "builder_close" not in failures[0][1]
    if early:
        assert importer.builder is None
    else:
        assert importer.builder.closed


def test_failing_diagnostic_callback_cannot_replace_source_error_or_cleanup(importer):
    importer.fail = "manifest", 1

    def failure():
        raise RuntimeError("diagnostic observation failed")

    with pytest.raises(ValueError, match=SECRET):
        source.load_role_source(importer.experiment, expected_pins=importer.metadata.snapshot_pins,
                                source=importer.metadata, on_failure=failure)
    assert importer.builder.closed and importer.program.closed


def topology_runtime():
    topology = servers.LocalTopology.__new__(servers.LocalTopology)
    topology._process_lock = threading.Lock()
    topology._stopping = threading.Event()
    topology._failure_observer = None
    topology._startup_breadcrumbs = {"phase": "ready", "role": None}
    runtime = NS(_topology=topology, _lock=threading.Lock(), _state={"phase": "online"},
                 _active_run=object(), _has_started_run=True,
                 _memory_guard=NS(failure_observation={"pressure": "warning"}))
    return topology, runtime


def test_integrated_import_failure_preserves_precleanup_snapshot_and_size_bound(importer):
    topology, runtime = topology_runtime()
    sampled = []

    def cached_rss():
        sampled.append(topology._stopping.is_set())
        return None  # No live native calls, IPC, process scans or RSS probes.

    collector = diagnostics.FailureDiagnostics(runtime, NS(failure_snapshot=cached_rss))
    observed = []

    def observe():
        observed.append((topology._stopping.is_set(), importer.program.closed))
        collector.capture()

    topology._failure_observer = observe
    importer.fail = "manifest", 1
    with pytest.raises(ValueError, match=SECRET):
        with topology._source_import_scope("worker_b") as lifecycle:
            source.load_role_source(
                importer.experiment, expected_pins=importer.metadata.snapshot_pins,
                source=importer.metadata, lifecycle=lifecycle, on_failure=topology._observe_failure,
            )
    topology._startup_step("qualification_retirement", "client")
    snapshot = collector.capture()
    assert observed[0] == (True, False)
    assert sampled == [True]
    assert snapshot["phase"] == "source_manifest"
    assert snapshot["role"] == "worker_b"
    assert snapshot["host"]["pressure"] == "warning"
    assert importer.builder.closed and importer.program.closed
    encoded = json.dumps(snapshot).encode()
    assert len(encoded) < 4096
    assert SECRET.encode() not in encoded


def test_import_precedes_active_request_only_inside_live_scope():
    topology, runtime = topology_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    with topology._source_import_scope("worker_a") as lifecycle:
        lifecycle("source_weights")
        value = collector.observe()
        assert (value["phase"], value["role"]) == ("source_weights", "worker_a")
    assert topology.startup_breadcrumbs["source_import"]["active"] is False
    # Completion retains no authority, even if the ordinary last phase is stale
    # or a caller invokes a previously retained callback after scope completion.
    lifecycle("source_manifest")
    value = collector.observe()
    assert (value["phase"], value["role"]) == ("request", None)
    assert topology.startup_breadcrumbs["phase"] == "source_weights"
    runtime._active_run = None
    runtime._state["phase"] = "ready"
    assert collector.observe()["phase"] == "ready"


def test_metadata_failure_is_frozen_before_metadata_lease_retirement(importer, monkeypatch):
    topology, runtime = topology_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    topology._failure_observer = collector.capture
    importer.experiment.pipeline = NS(components={"linear": NS(params={"capacity": "decoder"})})
    monkeypatch.setattr(source, "Experiment", type(importer.experiment))
    retired = []

    @contextmanager
    def reservation():
        try:
            yield
        finally:
            retired.append((topology._stopping.is_set(), collector._snapshot))

    monkeypatch.setattr(source.tokenizer_admission, "reserve_decoder_metadata", reservation)
    importer.fail = "geometry", 1
    with pytest.raises(ValueError, match=SECRET):
        with topology._source_import_scope("client") as lifecycle:
            source.resolve_client_source(importer.experiment, lifecycle=lifecycle,
                                         on_failure=topology._observe_failure)
    assert len(retired) == 1 and retired[0][0] is True
    assert retired[0][1]["phase"] == "source_geometry"
    assert retired[0][1]["role"] == "client"
    assert collector.capture() == retired[0][1]


def test_replaced_import_scope_callback_cannot_override_current_setup():
    topology, runtime = topology_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    with topology._source_import_scope("worker_a") as first:
        with topology._source_import_scope("worker_b") as second:
            second("source_rotary")
            first("source_weights")
            assert collector.observe()["role"] == "worker_b"
        first("source_manifest")
        assert collector.observe()["phase"] == "request"


@pytest.mark.parametrize("phase,role,active", [
    ("source_weights", "worker_a", False), ("source_weights", "worker_a", 1),
    (SECRET, "worker_a", True), ("request", "worker_a", True),
    ("source_weights", SECRET, True), ("source_weights", "client", "true"),
])
def test_unscoped_or_invalid_import_record_does_not_override_request(phase, role, active):
    topology, runtime = topology_runtime()
    topology._startup_breadcrumbs["source_import"] = {
        "phase": phase, "role": role, "active": active, "payload": SECRET}
    value = diagnostics.FailureDiagnostics(runtime).observe()
    assert (value["phase"], value["role"]) == ("request", None)
    assert SECRET not in json.dumps(value)


def test_cancelled_scope_signals_before_capture_and_clears_active_marker():
    topology, runtime = topology_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    stopped_at_observation = []

    def observe():
        stopped_at_observation.append(topology._stopping.is_set())
        collector.capture()

    topology._failure_observer = observe
    with pytest.raises(asyncio.CancelledError):
        with topology._source_import_scope("worker_a") as lifecycle:
            lifecycle("source_headers")
            raise asyncio.CancelledError
    assert stopped_at_observation == [True]
    assert topology.startup_breadcrumbs["source_import"]["active"] is False
    assert collector.capture()["phase"] == "source_headers"
    assert collector.capture()["role"] == "worker_a"


def test_import_capture_never_waits_for_diagnostic_or_topology_locks():
    topology, runtime = topology_runtime()
    collector = diagnostics.FailureDiagnostics(runtime)
    topology._failure_observer = collector.capture
    with topology._source_import_scope("worker_a") as lifecycle:
        lifecycle("source_weights")
        with topology._process_lock:
            assert collector.observe()["phase"] == "request"
        with runtime._lock:
            assert collector.observe()["phase"] == "unknown"
            assert collector.observe()["role"] is None
        with topology._process_lock, runtime._lock, collector._lock:
            topology._observe_failure()
            assert topology._stopping.is_set()
            assert collector._snapshot is None
        assert collector.capture()["phase"] == "source_weights"
