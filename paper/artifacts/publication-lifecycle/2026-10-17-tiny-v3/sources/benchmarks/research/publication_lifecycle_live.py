"""Bounded research observers around the installed, ordinary response path.

The client streams observations to private disk. The controller replays them only
after worker exit. No numerical, transport, or shutdown algorithm is replaced.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import stat
import sys
import threading
import time
from unittest.mock import patch

import psutil

from .publication_lifecycle import (
    CLEANUP_FLAGS, InvalidObservation, MAX_PHASE_EVENTS, SCHEMA, TRACE_ADMISSION_BYTES, recorder_capacity,
    network_counters,
)

REQUEST_FILE = "publication-request.json"
STATE_FILE = "publication-state.json"
SAMPLES_FILE = "publication-samples.jsonl"
FINAL_FILE = "publication-final.json"
PROVENANCE_ADMISSION_BYTES = 16 << 20


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    state = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(65536):
            state.update(chunk)
    return state.hexdigest()


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_nlink, info.st_uid, info.st_gid)


@contextmanager
def sidecar_directory(path):
    """Pin every directory component; never follow symlinks or lexical escapes."""
    path = Path(path)
    if ".." in path.parts:
        raise InvalidObservation("sidecar_path")
    path = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    with ExitStack() as stack:
        fd = os.open(path.anchor, flags)
        stack.callback(os.close, fd)
        chain = []
        for name in path.parts[1:]:
            child = os.open(name, flags, dir_fd=fd)
            stack.callback(os.close, child)
            info = os.fstat(child)
            chain.append((fd, name, info.st_dev, info.st_ino))
            fd = child
        yield fd
        for parent, name, device, inode in chain:
            info = os.stat(name, dir_fd=parent, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != (device, inode):
                raise InvalidObservation("sidecar_directory_changed")


def check_commitment(value, maximum):
    if (not isinstance(value, dict) or set(value) != {"bytes", "sha256"}
            or type(value["bytes"]) is not int or not 0 <= value["bytes"] <= maximum
            or type(value["sha256"]) is not str or len(value["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in value["sha256"])):
        raise InvalidObservation("sidecar_commitment_bound")


class SidecarReader:
    """One size-bounded FD supplies both parsed bytes and their commitment."""
    def __init__(self, file, info, directory, name, expected):
        self.file, self.info, self.directory, self.name = file, info, directory, name
        self.expected, self.remaining = expected, info.st_size
        self.hasher = hashlib.sha256()

    def _read(self, size, *, line=False):
        if type(size) is not int or size < 0:
            raise InvalidObservation("sidecar_read_bound")
        size = min(size, self.remaining)
        raw = self.file.readline(size) if line else self.file.read(size)
        if size and not raw:
            raise InvalidObservation("sidecar_changed")
        self.remaining -= len(raw)
        self.hasher.update(raw)
        return raw

    def read(self, size):
        return self._read(size)

    def readline(self, size):
        return self._read(size, line=True)

    @property
    def commitment(self):
        return {"bytes": self.info.st_size, "sha256": self.hasher.hexdigest()}

    def verify(self):
        current = os.fstat(self.file.fileno())
        named = os.stat(self.name, dir_fd=self.directory, follow_symlinks=False)
        if (self.remaining or _file_identity(current) != _file_identity(self.info)
                or _file_identity(named) != _file_identity(self.info)):
            raise InvalidObservation("sidecar_changed")
        if self.expected is not None and self.commitment != self.expected:
            raise InvalidObservation("trace_file_binding")


@contextmanager
def sidecar_file(path, maximum, *, expected=None, directory=None):
    """Check type/size before any read; hold the same regular FD through verification."""
    if type(maximum) is not int or maximum < 0:
        raise InvalidObservation("sidecar_size")
    if expected is not None:
        check_commitment(expected, maximum)
    path = Path(path)
    with ExitStack() as stack:
        if directory is None:
            directory = stack.enter_context(sidecar_directory(path.parent))
        elif path.name != str(path) or path.name in {"", ".", ".."}:
            raise InvalidObservation("sidecar_path")
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=directory)
        stack.callback(os.close, fd)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise InvalidObservation("sidecar_not_regular")
        if not 0 <= info.st_size <= maximum:
            raise InvalidObservation("sidecar_size")
        if expected is not None and info.st_size != expected["bytes"]:
            raise InvalidObservation("trace_file_binding")
        file = stack.enter_context(os.fdopen(fd, "rb", closefd=False))
        reader = SidecarReader(file, info, directory, path.name, expected)
        yield reader
        reader.verify()


def sidecar_json(reader):
    raw = reader.read(reader.remaining)
    reader.verify()
    return json.loads(raw)


def read_private_json(path, maximum=1 << 20):
    with sidecar_file(path, maximum) as reader:
        return sidecar_json(reader)


def write_private(path, value, *, replace=False):
    """Exclusive receipts; atomic replacement only for the private progress journal."""
    path = Path(path)
    target = path.with_name(path.name + ".tmp") if replace else path
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(canonical(value))
        file.flush()
    if replace:
        os.replace(target, path)


def process_identity(pid):
    process = psutil.Process(pid)
    return {"pid": pid, "create_time_ns": round(process.create_time() * 1e9)}


def observation_clock():
    return {"name": "monotonic", "resolution_ns": max(1, math.ceil(time.get_clock_info("monotonic").resolution * 1e9))}


def runtime_provenance(build):
    """Compare actual imports/files with the independently inspected installed build."""
    import pllm
    root = Path(pllm.__file__).resolve().parent
    sources = {p.relative_to(root).as_posix(): file_digest(p)
        for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in {".py", ".json"}}
    native = importlib.util.find_spec("pllm._native")
    native_path = None if native is None else native.origin
    native_hash = file_digest(native_path) if native_path else None
    python_hash = file_digest(sys.executable)
    expected = build["attributes"]
    if (sources != expected["sdk_sources"] or native_hash != expected["native_sha256"]
            or python_hash != expected["python_binary"]):
        raise InvalidObservation("installed_runtime_provenance")
    return {"build_digest": build["digest"], "sdk_sources_digest": digest(sources),
        "sdk_root": str(root), "native_origin": native_path, "native_sha256": native_hash,
        "python_executable": sys.executable, "python_binary": python_hash}


def failure(error, where):
    return {"where": where, "exception": type(error).__name__,
        "code": str(error) if isinstance(error, InvalidObservation) else "execution_failed"}


def priced_admission(report, memory_budget_bytes):
    """Charge client recorder scratch and each provider's receipt bootstrap."""
    from pllm.runtime.benchmark_memory import admit_memory, host_memory
    estimate = copy.deepcopy(report["estimate"])
    providers = estimate["provider_peak_bytes"]
    extra = len(providers) * PROVENANCE_ADMISSION_BYTES
    for role in providers:
        providers[role] += PROVENANCE_ADMISSION_BYTES
    estimate["client_peak_bytes"] += TRACE_ADMISSION_BYTES
    estimate["native_total_peak_bytes"] += TRACE_ADMISSION_BYTES + extra
    estimate["components"].update(publication_lifecycle_trace_bytes=TRACE_ADMISSION_BYTES,
        publication_lifecycle_provider_receipts_bytes=extra)
    return admit_memory(estimate, host_memory(), backend="native", enforced_wan=True,
                        memory_budget_bytes=memory_budget_bytes)


class Capture:
    def __init__(self, runtime, configuration, directory):
        self.runtime, self.configuration = runtime, configuration
        self.directory = Path(directory)
        self.topology = self.network = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.phase = "public_setup"
        self.count = self.disk_bytes = 0
        self.hasher = hashlib.sha256()
        self.last = None
        self.ended = False
        self.client_closed = self.dashboard_joined = False
        self.target_done = False
        self.ordinary_sampler = None
        self.request_workers = []  # Strong thread references survive DashboardRuntime._worker clearing.
        self.capacity = recorder_capacity(configuration["timeout_seconds"])
        disk = os.statvfs(self.directory)
        if disk.f_bavail * disk.f_frsize < self.capacity["maximum_trace_disk_bytes"] + (2 << 20):
            raise InvalidObservation("trace_disk_admission")
        owners = {role: {"role": role, "host": "local_host", "client": role == "client",
            "pid": None, "create_time_ns": None, "first_sample": None}
            for role in configuration["expected_roles"]}
        owners["client"].update(**process_identity(os.getpid()), first_sample=0)
        self.document = {"schema": SCHEMA, "complete": False, "dropped_samples": 0,
            "nominal_sample_interval_ns": self.capacity["interval_ns"], "capacity": self.capacity,
            "clock": configuration["clock"], "request_workers": [],
            "expected_roles": configuration["expected_roles"], "owners": owners, "events": [],
            "binding": configuration["binding"], "runtime_provenance": configuration["runtime"],
            "errors": [], "target_succeeded": False, "terminal_received": False,
            "semantic_work": {"prefill_forwards": 0, "decode_forwards": 0, "selections": 0,
                "terminal_eos": False, "text_events": 0, "preparation_calls": 0},
            "helper_census": "serving window: undeclared descendants reject; after target: metadata helpers excluded",
            "network_conditions": configuration["wan"]}
        fd = os.open(self.directory / SAMPLES_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.file = os.fdopen(fd, "wb")
        self.thread = threading.Thread(target=self._loop, name="publication-lifecycle", daemon=True)

    def error(self, error, where):
        with self.lock:
            if len(self.document["errors"]) < 32:
                self.document["errors"].append(failure(error, where))
            try:
                self.checkpoint()
            except Exception:
                # Persistence failure must never replace the original exception.
                # Missing/truncated final receipts fail closed at the parent.
                pass

    def checkpoint(self):
        with self.lock:
            self.document["sample_count"] = self.count
            write_private(self.directory / STATE_FILE, self.document, replace=True)

    def bind_topology(self, topology):
        if topology is None:
            return
        if self.topology is not None and self.topology is not topology:
            raise InvalidObservation("topology_replaced")
        if set(topology._role_ids) != set(self.document["owners"]) - {"client"}:
            raise InvalidObservation("expected_role_graph")
        self.topology, self.network = topology, topology._native_network
        if self.network is not None:
            origin = self.network.pacer.started
            if self.document.setdefault("relay_started_monotonic_seconds", origin) != origin:
                raise InvalidObservation("relay_clock_replaced")

    def register(self, role, process, command, started_ns):
        owner = self.document["owners"].get(role)
        if owner is None or owner["pid"] is not None or process.pid is None:
            raise InvalidObservation("unexpected_process_incarnation")
        owner.update(**process_identity(process.pid), command_digest=digest(command),
            first_sample=self.count, spawn_started_ns=started_ns, spawn_returned_ns=time.monotonic_ns())

    def _observe(self):
        start = time.monotonic_ns()
        self.bind_topology(self.runtime._topology)
        processes = {}
        if self.topology is not None:
            with self.topology._process_lock:
                processes = dict(self.topology._processes)
        owners = self.document["owners"]
        if set(processes) - set(owners):
            raise InvalidObservation("undeclared_provider")
        rss, alive = {}, []
        for role, owner in owners.items():
            rss[role] = 0
            pid = owner["pid"]
            if pid is None:
                if role in processes:
                    raise InvalidObservation("unbound_child")
                continue
            if role != "client":
                child = processes.get(role)
                if child is None or child.pid != pid:
                    raise InvalidObservation("child_ownership_changed")
                code = child.poll()
                if code is not None:
                    owner.setdefault("reaped_ns", time.monotonic_ns())
                    owner["returncode"] = code
                    if self.phase != "cleanup":
                        raise InvalidObservation("provider_exited_before_cleanup")
                    continue
            try:
                process = psutil.Process(pid)
                if round(process.create_time() * 1e9) != owner["create_time_ns"]:
                    raise InvalidObservation("process_replaced")
                rss[role] = process.memory_info().rss
                alive.append(role)
            except psutil.NoSuchProcess:
                if role == "client" or self.phase != "cleanup":
                    raise
        if not self.target_done:
            # Spawning is serialized with this lock, so a declared child cannot
            # appear between Popen and its incarnation registration.
            known = {owner["pid"] for owner in owners.values()}
            if any(child.pid not in known for child in psutil.Process().children(recursive=True)):
                raise InvalidObservation("undeclared_serving_helper")
        counters = {"directed_stream_bytes": {}, "total_stream_bytes": 0,
            "parties": {role: {"upload": 0, "download": 0} for role in owners}}
        elapsed = None
        if self.network is not None:
            if self.network.error:
                raise InvalidObservation("relay_error")
            if self.network.closed:
                if self.network.thread is not None and self.network.thread.is_alive():
                    raise InvalidObservation("relay_not_joined")
                snapshot = self.network.pacer.snapshot()
            else:
                snapshot = self.network.snapshot()
            if (snapshot["conditions"] != self.configuration["wan"]
                    or snapshot["role_parties"] != {role: role for role in owners}):
                raise InvalidObservation("network_identity")
            counters = {"directed_stream_bytes": snapshot["directed_stream_bytes"],
                "total_stream_bytes": snapshot["total_stream_bytes"],
                "parties": {role: {direction: snapshot["parties"][role][direction]["admitted_bytes"]
                    for direction in ("upload", "download")} for role in owners}}
            elapsed = snapshot["elapsed_seconds"]
        network_counters(counters, list(owners))
        swap = psutil.swap_memory().used
        return {"start_ns": start, "end_ns": time.monotonic_ns(), "rss_bytes": rss,
            "alive": alive, "directed_stream_bytes": counters["directed_stream_bytes"], "network": counters,
            "host_swap_used_bytes": swap, "network_elapsed_seconds": elapsed}

    def sample(self):
        with self.lock:
            if self.count >= self.capacity["maximum_samples"]:
                raise InvalidObservation("trace_sample_bound")
            row = self._observe()
            start = self.document.setdefault("observation_started_ns", row["start_ns"])
            if row["end_ns"] - start > (self.capacity["timeout_seconds"] + self.capacity["shutdown_grace_seconds"]) * 1e9:
                raise InvalidObservation("trace_duration_exceeds_capacity")
            encoded = canonical(row) + b"\n"
            if len(encoded) > self.capacity["maximum_sample_bytes"]:
                raise InvalidObservation("trace_row_bound")
            self.file.write(encoded)
            self.file.flush()
            self.hasher.update(encoded)
            self.disk_bytes += len(encoded)
            self.count += 1
            self.last = row
            return self.count - 1

    def mark(self, phase):
        with self.lock:
            if self.document["events"] and phase == self.phase:
                return
            if len(self.document["events"]) >= MAX_PHASE_EVENTS:
                raise InvalidObservation("phase_event_bound")
            at = self.sample()
            self.document["events"].append({"phase": phase, "sample": at})
            self.phase = phase
            self.checkpoint()

    def selection(self, eos):
        work = self.document["semantic_work"]
        work["selections"] += 1
        if work["selections"] > 64 or work["terminal_eos"]:
            raise InvalidObservation("selection_sequence")
        work["terminal_eos"] = bool(eos)
        self.mark("cleanup" if eos else "unclassified")

    def text(self):
        work = self.document["semantic_work"]
        work["text_events"] += 1
        if work["text_events"] > self.configuration["output_cap"]:
            raise InvalidObservation("emission_bound")
        if work["text_events"] == self.configuration["output_cap"]:
            self.mark("cleanup")

    def request_finished(self, capture):
        self.document["request"] = {"input_tokens": capture.input_tokens, "output_tokens": capture.output_tokens,
            "admitted_ns": capture.started_monotonic_ns, "completed_ns": capture.finished_monotonic_ns,
            "first_output_ns": capture.first_token_monotonic_ns, "last_output_ns": capture.last_token_monotonic_ns,
            "preparation_started_ns": capture.preparation_started_monotonic_ns,
            "preparation_finished_ns": capture.preparation_finished_monotonic_ns,
            "online_started_ns": capture.online_started_monotonic_ns}
        self.checkpoint()

    def run_request_worker(self, function, *args, **kwargs):
        thread = threading.current_thread()
        with self.lock:
            if self.request_workers:
                raise InvalidObservation("multiple_request_workers")
            row = {"ident": thread.ident, "native_id": thread.native_id, "entered_ns": time.monotonic_ns(),
                "returned_ns": None, "joined_ns": None, "outcome": "running"}
            self.request_workers.append((thread, row))
            self.document["request_workers"].append(row)
            self.checkpoint()
        try:
            value = function(*args, **kwargs)
        except BaseException as error:
            row["outcome"] = "failed"
            self.error(error, "request_worker_teardown")
            raise
        else:
            row["outcome"] = "returned"
            return value
        finally:
            row["returned_ns"] = time.monotonic_ns()
            try:
                self.checkpoint()
            except Exception as error:
                self.error(error, "request_worker_receipt")

    def join_request_workers(self):
        with self.lock:
            workers = list(self.request_workers)
        for thread, row in workers:
            if thread is threading.current_thread():
                self.error(InvalidObservation("request_worker_self_join"), "request_worker_join")
                continue
            thread.join(timeout=10)
            if thread.is_alive():
                self.error(InvalidObservation("request_worker_not_joined"), "request_worker_join")
            else:
                row["joined_ns"] = row["joined_ns"] or time.monotonic_ns()

    def _loop(self):
        while not self.stop.wait(self.capacity["interval_ns"] / 1e9):
            try:
                self.sample()
            except BaseException as error:
                self.error(error, "sampler")
                return

    def start(self):
        self.mark("public_setup")
        self.thread.start()

    def health(self):
        runtime, network = self.runtime, self.network
        threads = [runtime._worker, *runtime._background_threads, *(t for t, _ in self.request_workers)]
        guard = runtime._memory_guard
        return {"dashboard_joined": self.dashboard_joined, "client_close_returned": self.client_closed,
            "client_workers_joined": len(self.request_workers) == 1
                and all(row["joined_ns"] is not None and row["outcome"] == "returned" for _, row in self.request_workers)
                and not any(t is not None and t.is_alive() for t in threads),
            "memory_guard_joined": guard is not None and not guard._thread.is_alive(),
            "ordinary_sampler_joined": self.ordinary_sampler is not None and not self.ordinary_sampler.thread.is_alive(),
            "observer_joined": not self.thread.is_alive(),
            "topology_closed": self.topology is not None and self.topology.closed,
            "relay_closed": network is not None and network.closed,
            "relay_joined": network is not None and (network.thread is None or not network.thread.is_alive()),
            "relay_empty": network is not None and not network.connections and not network.writers,
            "relay_loop_closed": network is not None and network.loop is not None and network.loop.is_closed(),
            "relay_listeners_closed": network is not None and bool(network.servers)
                and all(not server.is_serving() for server in network.servers),
            "relay_error": network.error if network is not None else "missing_relay",
            "no_live_children": not psutil.Process().children(recursive=True)}

    def finish(self, *, after_serialization):
        """Called once from atexit, after the ordinary stdout envelope was closed."""
        if self.ended:
            return
        self.ended = True
        self.join_request_workers()
        self.stop.set()
        self.thread.join(timeout=10)
        try:
            if self.thread.is_alive():
                raise InvalidObservation("sampler_not_joined")
            self.document["cleanup_health"] = self.health()
            if (any(self.document["cleanup_health"].get(key) is not True for key in CLEANUP_FLAGS)
                    or self.document["cleanup_health"].get("relay_error") is not None):
                raise InvalidObservation("cleanup_health")
            for role, owner in self.document["owners"].items():
                if role != "client":
                    path = self.directory / f"publication-child-{role}.json"
                    owner["child_receipt"] = read_private_json(path)
            self.mark("end")
            self.document["final_network"] = self.last["network"]
            self.document["complete"] = bool(after_serialization and self.document["target_succeeded"]
                                              and not self.document["errors"])
        except BaseException as error:
            self.error(error, "finalization")
        finally:
            self.file.close()
        self.checkpoint()
        receipt = {"schema": "pllm.publication_finalization.v2", "document": self.document,
            "after_worker_serialization": bool(after_serialization),
            "samples_sha256": self.hasher.hexdigest(), "samples_bytes": self.disk_bytes,
            "process_self_high_water_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                * (1 if sys.platform == "darwin" else 1024),
            "observed_ns": time.monotonic_ns(),
            "high_water_scope": "RUSAGE_SELF birth through post-serialization atexit observation; excludes later receipt serialization/interpreter teardown"}
        write_private(self.directory / FINAL_FILE, receipt)


@contextmanager
def instrument(configuration, directory, script):
    """Wrap bounded ownership/timing boundaries, delegating every original call."""
    from pllm import metrics
    from pllm.runtime import benchmark_cli, benchmark_resources, native_wan, benchmark_response_audit
    from pllm.runtime.client import ResponseStream, RuntimeClient
    from pllm.runtime.dashboard import DashboardRuntime
    from pllm.runtime.semantic_executor import SemanticDecoderRuntime
    from pllm.runtime.transformer_client import MaskedTransformerClientRuntime
    from pllm.runtime.servers import LocalTopology

    session = {"capture": None}
    def current():
        if session["capture"] is None:
            raise InvalidObservation("missing_capture_owner")
        return session["capture"]

    with ExitStack() as stack:
        def install(owner, name, factory):
            stack.enter_context(patch.object(owner, name, factory(getattr(owner, name))))

        def start(original):
            def wrapped(owner):
                if session["capture"] is not None:
                    raise InvalidObservation("multiple_benchmark_owners")
                session["capture"] = Capture(owner.runtime, configuration, directory)
                current().ordinary_sampler = owner
                current().start()
                return original(owner)
            return wrapped

        def spawn(original):
            def wrapped(owner, role, command):
                capture = current()
                with capture.lock:
                    capture.bind_topology(owner)
                    capture.sample()
                    started = time.monotonic_ns()
                    # The child verifies the installed code before delegating the
                    # unchanged -m entry point. No protocol/model implementation is copied.
                    launch = [command[0], "-I", str(script), "--provider", role, str(directory), *command[1:]]
                    process = original(owner, role, launch)
                    capture.register(role, process, command[1:], started)
                    capture.sample()
                    capture.checkpoint()
                    return process
            return wrapped

        def close(original, label):
            def wrapped(owner, *args, **kwargs):
                capture = current()
                try:
                    value = original(owner, *args, **kwargs)
                except BaseException as error:
                    capture.error(error, label)
                    raise
                if label == "client_close":
                    capture.client_closed = True
                elif label == "dashboard_close":
                    capture.join_request_workers()
                    capture.dashboard_joined = not owner.thread.is_alive()
                    if owner.error is not None:
                        capture.error(owner.error, label)
                return value
            return wrapped

        def network_close(original):
            wrapped = close(original, "relay_close")
            def locked(owner):
                with current().lock:
                    return wrapped(owner)
            return locked

        def begin(original):
            def wrapped(owner, *args, **kwargs):
                current().mark("request_setup")
                return original(owner, *args, **kwargs)
            return wrapped

        def execute_run(original):
            def wrapped(owner, *args, **kwargs):
                return current().run_request_worker(original, owner, *args, **kwargs)
            return wrapped

        def stop_runtime(original):
            async def wrapped(owner):
                try:
                    current().mark("cleanup")
                except BaseException as error:
                    current().error(error, "cleanup_boundary")
                return await original(owner)
            return wrapped

        def prepare(original):
            def wrapped(owner, *args, **kwargs):
                capture = current()
                previous = capture.phase
                if previous not in {"public_setup", "request_setup"}:
                    raise InvalidObservation("unsupported_overlapping_preparation")
                capture.document["semantic_work"]["preparation_calls"] += 1
                capture.mark("private_preparation")
                try:
                    value = original(owner, *args, **kwargs)
                except BaseException:
                    # Never replace an execution error with a failed boundary read.
                    raise
                capture.mark(previous)
                return value
            return wrapped

        def forward(original):
            def wrapped(owner, *args, **kwargs):
                capture = current()
                phase = "prefill" if owner.position == 0 else "decode"
                capture.document["semantic_work"][phase + "_forwards"] += 1
                capture.mark(phase)
                return original(owner, *args, **kwargs)
            return wrapped

        def audit(original):
            @contextmanager
            def wrapped(salt):
                with original(salt) as records:
                    audited_sample = MaskedTransformerClientRuntime.sample
                    def sample(owner, *args, **kwargs):
                        token = audited_sample(owner, *args, **kwargs)
                        current().selection(token == int(owner.cfg["eos_token_id"]))
                        return token
                    with patch.object(MaskedTransformerClientRuntime, "sample", sample):
                        yield records
            return wrapped

        def next_event(original):
            def wrapped(owner):
                event = original(owner)
                kind = event.get("type") if isinstance(event, dict) else event.type
                if kind == "response.output_text.delta":
                    current().text()
                elif kind in {"response.completed", "response.incomplete"}:
                    current().document["terminal_received"] = True
                    current().mark("cleanup")
                return event
            return wrapped

        def finish_run(original):
            def wrapped(owner, capture, *args, **kwargs):
                current().request_finished(capture)
                return original(owner, capture, *args, **kwargs)
            return wrapped

        def memory(original):
            def wrapped(*args, **kwargs):
                return priced_admission(original(*args, **kwargs), kwargs.get("memory_budget_bytes"))
            return wrapped

        install(benchmark_resources.ProcessMemorySampler, "start", start)
        install(LocalTopology, "_spawn", spawn)
        install(LocalTopology, "close", lambda original: close(original, "topology_close"))
        install(RuntimeClient, "close", lambda original: close(original, "client_close"))
        install(ResponseStream, "close", lambda original: close(original, "stream_close"))
        install(benchmark_cli._DashboardHandle, "close", lambda original: close(original, "dashboard_close"))
        install(native_wan.NativePartyNetwork, "close", network_close)
        install(DashboardRuntime, "begin", begin)
        install(DashboardRuntime, "_execute_run", execute_run)
        install(DashboardRuntime, "stop", stop_runtime)
        install(DashboardRuntime, "_finish_run", finish_run)
        install(RuntimeClient, "_prepare_inventory", prepare)
        install(SemanticDecoderRuntime, "_forward", forward)
        install(benchmark_response_audit, "response_state_audit", audit)
        install(ResponseStream, "__next__", next_event)
        install(metrics, "benchmark_memory", memory)
        try:
            yield session
        except BaseException as error:
            if session["capture"] is not None:
                current().error(error, "target")
            raise
