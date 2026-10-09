"""Private bounded subprocess protocol for installed benchmark targets.

Inputs and the cohort salt arrive over stdin, never command arguments or reports.
Only an explicit allowlist of built-in executors can run. No callbacks or pickle.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

from pllm._benchmark_contract import BenchmarkTarget, Sampling, TargetUnavailable, canonical, digest

PROTOCOL = "pllm.benchmark_worker.v1"


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class Measurement:
    def __init__(self, request):
        self.sampling = Sampling(**request["sampling"])
        self.salt = bytes.fromhex(request["salt"])
        if len(self.salt) != 32:
            raise ValueError("invalid private cohort salt")
        self.inputs = request["inputs"]
        self.directory = Path(request["directory"])
        self.metrics, self.samples, self.details = [], [], {}
        self.fixture = {"kind": "none", "digest": None, "metadata": {}}
        self.oracle = {"kind": "none", "outcome": "not_run", "checked_values": 0}
        self.comparison = {key: None for key in ("source_digest", "numeric_digest", "privacy_digest",
            "state_digest", "workload_digest", "output_digest", "measurement_digest")}
        self.admission = {"admitted": None, "required_bytes": None,
            "limit_bytes": self.sampling.memory_bytes, "host_headroom_bytes": None,
            "scope": "target_working_set", "reason": None}
        self.limitations = ["Target-local evidence; full-response benefits require matched confirmation."]

    def commitment(self, value):
        return hmac.new(self.salt, canonical(value), hashlib.sha256).hexdigest()

    def admit(self, required):
        # Reuse the same reserve and no-swap admission as ordinary benchmarks.
        from pllm.runtime.benchmark_memory import host_memory
        host = host_memory()
        headroom = max(0, host.available - host.reserve)
        admitted = required <= self.sampling.memory_bytes and required <= headroom
        self.admission.update(admitted=admitted, required_bytes=required,
                              host_headroom_bytes=headroom, reason=None if admitted else "memory_limit")
        if not admitted:
            raise TargetUnavailable("memory_limit", status="blocked")

    def metric(self, name, value, **kwargs):
        # Metadata construction follows the target's captured high-water mark.
        from pllm._benchmark_evidence import measurement
        self.metrics.append(measurement(name, value, **kwargs))

    def run(self, action):
        values, cpus, walls = [], [], []
        for index in range(self.sampling.warmups + self.sampling.repetitions):
            wall, cpu = time.perf_counter(), time.process_time()
            value = action(index)
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
            warmup = index < self.sampling.warmups
            self.samples.append({"index": index, "kind": "warmup" if warmup else "measurement",
                "status": "completed", "duration_seconds": wall, "artifact_digest": None})
            if not warmup:
                values.append(value)
                walls.append(wall)
                cpus.append(cpu)
        self.details["timings"] = {"wall_seconds": walls, "cpu_seconds": cpus}
        return values


def _runtime(target, m):
    import subprocess
    role = target.to_spec()["parameters"]["role"]
    m.admit(256 << 20)
    imports = {"sdk": "import pllm", "native": "from pllm import _native",
               "client": "from pllm import OpenAI",
               "preparation": "from pllm.runtime.preparation_server import create_preparation_app"}
    def execute(_):
        code = ("import resource,sys,time,json;" + imports[role] + ";"
                "r=resource.getrusage(resource.RUSAGE_SELF);"
                "print(json.dumps({'rss':int(r.ru_maxrss*(1 if sys.platform=='darwin' else 1024)),"
                "'cpu':time.process_time()}))")
        return json.loads(subprocess.check_output([sys.executable, "-I", "-c", code], stderr=subprocess.DEVNULL))
    observed = m.run(execute)
    m.details["timings"]["cpu_seconds"] = [v["cpu"] for v in observed]
    m.details["target_peak_rss_bytes"] = max(v["rss"] for v in observed)
    m.details["target_cpu_from_birth_seconds"] = sum(v["cpu"] for v in observed) / len(observed)
    m.details["sample_owners"] = ["runtime_import_child"]
    m.oracle = {"kind": "import_completion", "outcome": "passed", "checked_values": 1}
    m.comparison.update({key: digest({"runtime": role}) for key in m.comparison})
    m.limitations += ["Fresh import child excludes the benchmark harness, model, allocator workspace and active connections.",
                       "Observed import residency is not a mathematical lower bound on a redesigned client."]


def main():
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    # Preserve a single result channel; library and role logs must not pollute it.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        raw = sys.stdin.buffer.read((16 << 20) + 1)
        if len(raw) > 16 << 20:
            raise ValueError("benchmark input exceeds 16 MiB")
        request = json.loads(raw)
        if request["protocol"] != PROTOCOL:
            raise ValueError("unsupported worker protocol")
        target = BenchmarkTarget.from_spec(request["target"])
        m = Measurement(request)
        status, failure = "completed", None
        try:
            if target.kind == "runtime":
                _runtime(target, m)
            else:
                from pllm.runtime.benchmark_targets import execute_target
                execute_target(target, m)
        except TargetUnavailable as exc:
            status, failure = exc.status, exc.code
            m.details.update(exc.details)
        except KeyboardInterrupt:
            status, failure = "cancelled", "cancelled"
        except BaseException:
            # Exception strings may contain prompts, keys or caller-owned values.
            status, failure = "failed", "target_execution_failed"
        observed_peak = m.details.pop("target_peak_rss_bytes", peak_rss())
        cpu_birth = m.details.pop("target_cpu_from_birth_seconds", time.process_time())
        import statistics
        timings = m.details.get("timings", {})
        role = "preparation" if target.kind == "preparation" or target.kind == "runtime" and target.to_spec()["parameters"]["role"] == "preparation" else "client"
        m.metric("target.latency", statistics.median(timings["wall_seconds"]) if timings.get("wall_seconds") else None,
                 role=role, count=len(timings.get("wall_seconds", [])))
        m.metric("target.cpu", statistics.median(timings["cpu_seconds"]) if timings.get("cpu_seconds") else None,
                 kind="cpu", role=role, count=len(timings.get("cpu_seconds", [])),
                 owners=m.details.get("target_cpu_owners"))
        m.metric("process.peak_rss", observed_peak, kind="memory", role=role, window="process_birth_through_target",
                  scope="process_lifetime_high_water")
        m.metric("process.cpu", cpu_birth, kind="cpu", role=role, window="process_birth_through_target")
        m.comparison["workload_digest"] = m.commitment({"target_workload": m.comparison["workload_digest"],
                                                       "inputs": m.inputs})
        from pllm._benchmark_build import hardware_identity
        environment = {"platform": sys.platform, "machine": os.uname().machine,
                       "release": os.uname().release, "threads": m.sampling.threads,
                       "hardware": hardware_identity()}
        result = {"protocol": PROTOCOL, "status": status, "failure_code": failure,
                  "metrics": m.metrics, "samples": m.samples, "fixture": m.fixture,
                  "oracle": m.oracle, "admission": m.admission, "comparison": m.comparison,
                   "details": m.details, "limitations": m.limitations, "environment": environment,
                   "sample_owners": m.details.get("sample_owners", ["benchmark_worker"])}
    output.write(json.dumps(result, sort_keys=True, allow_nan=False))
    output.close()


if __name__ == "__main__":
    main()
