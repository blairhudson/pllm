"""Bounded same-window process RSS sampling for native benchmark roles."""
from __future__ import annotations

import os
import threading
import time

import psutil


class ProcessMemorySampler:
    def __init__(self, runtime, *, interval=0.05):
        from .client_memory_scope import ClientMemoryScope
        self.runtime = runtime
        declared = getattr(runtime, "_benchmark_client_memory_owner", None)
        if (type(declared) is ClientMemoryScope and declared.root_pid == os.getpid()
                and declared.claim_sampling()):
            self.client_memory_scope = declared
        else:
            self.client_memory_scope = ClientMemoryScope()
            self.client_memory_scope.claim_sampling()
        runtime._benchmark_client_memory_owner = self.client_memory_scope
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.max_sample_seconds = 0.0
        self.error = False
        self.identities = {}
        self._snapshot_lock = threading.Lock()
        self._latest = None
        self._latest_at = None
        self._role_peaks = {}
        self.thread = threading.Thread(target=self._run, name="pllm-process-rss", daemon=True)

    def sample(self):
        start = time.monotonic()
        self.client_memory_scope.begin_sample()
        topology = self.runtime._topology
        pids = {"client": os.getpid()}
        readings = {}
        if topology is not None:
            with topology._process_lock:
                processes = dict(topology._processes)
            for role, process in processes.items():
                pids[role] = process.pid
        native_roles = set(pids)
        pids.update(self.client_memory_scope.targets(
            {pid for role, pid in pids.items() if role != "client"}))
        try:
            total = 0
            observed = {}
            for role, pid in pids.items():
                try:
                    if pid is None:
                        raise ValueError("nonlocal resource domain")
                    process = psutil.Process(pid)
                    identity = (pid, process.create_time())
                    rss = process.memory_info().rss
                    if role not in native_roles and psutil.Process(pid).create_time() != identity[1]:
                        # Recheck registered helper birth with a fresh handle:
                        # Process.create_time() itself caches its first result.
                        raise ValueError("client helper identity changed during RSS read")
                    readings[role] = (*identity, rss)
                    if role in native_roles:
                        if self.identities.setdefault(role, identity) != identity:
                            raise ValueError("role process replaced during measurement")
                        total += rss
                        observed[role] = rss
                except (psutil.Error, OSError, ValueError):
                    if role in native_roles:
                        self.error = True
            if set(observed) != native_roles:
                return
            self.peak = max(self.peak, total)
            self.samples += 1
            from .benchmark_failure import ROLES
            with self._snapshot_lock:
                self._latest = {"latest_total_bytes": total, "peak_total_bytes": self.peak,
                    "roles": {role: {"latest_bytes": rss,
                        "peak_bytes": max(rss, self._role_peaks.get(role, 0))}
                        for role, rss in observed.items() if role in ROLES}}
                for role, value in self._latest["roles"].items():
                    self._role_peaks[role] = value["peak_bytes"]
                self._latest_at = time.monotonic()
        except (psutil.Error, OSError, ValueError):
            self.error = True
        finally:
            span = time.monotonic() - start
            self.client_memory_scope.observe(readings, span_seconds=span)
            self.max_sample_seconds = max(self.max_sample_seconds, span)

    def failure_snapshot(self):
        """Last complete sample only; no psutil calls on the abort path."""
        from copy import deepcopy
        if not self._snapshot_lock.acquire(blocking=False):
            return None
        try:
            if self._latest is None:
                return None
            value = deepcopy(self._latest)
            value["sample_age_seconds"] = max(0, time.monotonic() - self._latest_at)
            value["sampling_incomplete"] = self.error
            return value
        finally:
            self._snapshot_lock.release()

    def _run(self):
        while not self.stop_event.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=5)
        self.sample()
        return {
            "schema": "pllm.native_process_memory.v1",
            "sampled_total_peak_rss_bytes": None if self.error or not self.samples else self.peak,
            "samples": self.samples, "sample_interval_seconds": self.interval,
            "maximum_sample_span_seconds": self.max_sample_seconds,
            "complete": not self.error,
            "scope": "same-window sum of client/dashboard and live provider RSS, startup through responses; includes native relay and sampler",
            "limitations": "sampling can miss short peaks; shared pages may count in more than one process; excludes OS file cache and GPU allocations not charged to process RSS",
        }
