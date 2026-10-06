"""Bounded same-window process RSS sampling for native benchmark roles."""
from __future__ import annotations

import os
import threading
import time

import psutil


class ProcessMemorySampler:
    def __init__(self, runtime, *, interval=0.05):
        self.runtime = runtime
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.max_sample_seconds = 0.0
        self.error = False
        self.identities = {}
        self.thread = threading.Thread(target=self._run, name="pllm-process-rss", daemon=True)

    def sample(self):
        start = time.monotonic()
        topology = self.runtime._topology
        pids = {"client": os.getpid()}
        if topology is not None:
            with topology._process_lock:
                processes = dict(topology._processes)
            for role, process in processes.items():
                if process.pid is None:
                    self.error = True  # Docker is a different resource domain.
                    return
                pids[role] = process.pid
        try:
            total = 0
            for role, pid in pids.items():
                process = psutil.Process(pid)
                identity = (pid, process.create_time())
                if self.identities.setdefault(role, identity) != identity:
                    raise ValueError("role process replaced during measurement")
                total += process.memory_info().rss
            self.peak = max(self.peak, total)
            self.samples += 1
        except (psutil.Error, OSError, ValueError):
            self.error = True
        self.max_sample_seconds = max(self.max_sample_seconds, time.monotonic() - start)

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
