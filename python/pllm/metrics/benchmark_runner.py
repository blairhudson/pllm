"""One isolated, bounded lifecycle for installed benchmark targets."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import threading
from collections.abc import Mapping

from pllm._benchmark_contract import (BenchmarkCase, BenchmarkSuite, BenchmarkTarget,
    BuildRef, Sampling, canonical, digest)
from pllm._benchmark_evidence import assess_budgets, comparable_key, measurement, metric_contract
from pllm.evidence import BenchmarkResult, environment_digest


def _environment(threads=1):
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("PYTHON"):
            env.pop(key)
    env.update(OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads),
        NUMEXPR_NUM_THREADS=str(threads))
    return env


def inspect_build(build=BuildRef()):
    process = subprocess.run([build.python, "-I", "-m", "pllm._benchmark_build"],
        capture_output=True, timeout=120, env=_environment(), check=True)
    if len(process.stdout) > 4 << 20:
        raise ValueError("build record exceeds 4 MiB")
    observed = json.loads(process.stdout)
    if set(observed) != {"digest", "attributes"} or digest(observed["attributes"]) != observed["digest"]:
        raise ValueError("invalid installed build record")
    return observed


def _stop(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    # A leader can exit before its children. Retire the owned process group too.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _execute(case, sampling, inputs, salt, directory):
    from pllm._benchmark_worker import PROTOCOL
    from pllm.runtime.benchmark_memory import MemoryWatchdog, host_memory
    host = host_memory()
    pressure = []
    request = {"protocol": PROTOCOL, "target": case.target.to_spec(),
        "sampling": sampling.to_spec(), "inputs": inputs, "salt": salt.hex(),
        "directory": str(directory)}
    body = canonical(request)
    if len(body) > 16 << 20:
        raise ValueError("private target inputs exceed 16 MiB")
    process = subprocess.Popen([case.build.python, "-I", "-m", "pllm._benchmark_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
         cwd=directory, env=_environment(sampling.threads), start_new_session=True)
    # Providers can own separate process groups. Remember their process identities
    # before an interrupted worker becomes an orphan; never kill a reused PID.
    import psutil
    descendants, stop_tracking = {}, threading.Event()
    def track():
        while not stop_tracking.is_set():
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    descendants[child.pid] = child.create_time()
            except psutil.Error:
                pass
            stop_tracking.wait(0.02)
    tracker = threading.Thread(target=track, daemon=True)
    tracker.start()
    def abort(_):
        pressure.append(True)
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    monitor = MemoryWatchdog({"host": {"reserve_bytes": host.reserve, "swap_used": host.swap_used,
                                       "available": host.available}}, abort)
    try:
        monitor.start()
        try:
            stdout, _ = process.communicate(body, timeout=sampling.timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "failure_code": "timeout"}
        if pressure:
            return {"status": "blocked", "failure_code": "host_pressure"}
        if process.returncode or len(stdout) > 16 << 20:
            return {"status": "failed", "failure_code": "worker_failed"}
        try:
            payload = json.loads(stdout)
            if payload["protocol"] != PROTOCOL:
                raise ValueError("worker protocol differs")
            return payload
        except (ValueError, KeyError, TypeError):
            return {"status": "failed", "failure_code": "invalid_worker_result"}
    finally:
        monitor.close()
        _stop(process)
        stop_tracking.set()
        tracker.join()
        for pid, born in descendants.items():
            try:
                child = psutil.Process(pid)
                if child.create_time() == born:
                    child.kill()
                    child.wait(timeout=5)
            except psutil.Error:
                pass


def _result(case, suite, replicate, cohort, build, payload):
    status, failure = payload["status"], payload.get("failure_code")
    comparison = payload.get("comparison", {key: None for key in ("source_digest", "numeric_digest",
        "privacy_digest", "state_digest", "workload_digest", "output_digest", "measurement_digest")})
    comparison["workload_digest"] = comparison["workload_digest"] or digest({"unmeasured": cohort, "target": case.target.digest})
    details = payload.get("details", {})
    environment = payload.get("environment", {"scope": "not_observed"})
    metrics = payload.get("metrics") or [measurement("target.latency", None, reason=failure or "unmeasured")]
    samples = payload.get("samples", [])
    model = details.get("model", {"id": "benchmark-fixture", "checkpoint_digest": None, "source_lock_digest": None})
    execution = {"case": case.to_spec(), "case_digest": digest(case.to_spec()), "suite_digest": suite.digest,
        "cohort": cohort, "replicate": replicate, "build": build, "sampling": suite.sampling.to_spec(),
        "isolation": {"fresh_process": True, "sample_owners": payload.get("sample_owners", ["benchmark_worker"]), "controller_included": False},
        "fixture": payload.get("fixture", {"kind": "none", "digest": None, "metadata": {}}),
        "oracle": payload.get("oracle", {"kind": "none", "outcome": "not_run", "checked_values": 0}),
        "admission": payload.get("admission", {"admitted": None, "required_bytes": None,
            "limit_bytes": suite.sampling.memory_bytes, "host_headroom_bytes": None,
            "scope": "target_working_set", "reason": failure}), "comparison": comparison,
        "details": details, "failure_code": failure, "budgets": assess_budgets(case, metrics)}
    return BenchmarkResult.from_dict({"schema_version": "pllm.benchmark_result.v2",
        "id": f"{case.name}/{cohort[:16]}/{replicate}", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "scope": {"artifacts": "artifact", "response": "deployment"}.get(case.target.kind, case.target.kind),
        "profile": "pllm/benchmark-target/v1", "component_ids": details.get("component_ids", []),
        "model": model, "plan_lock_digest": details.get("plan_digest"), "configuration_digest": case.target.digest,
        "workload_digest": comparison["workload_digest"], "privacy_cohort": "target-local",
        "numeric_cohort": "target-bound", "environment": {"attributes": environment, "digest": environment_digest(environment)},
        "warmups": sum(s["kind"] == "warmup" for s in samples), "repetitions": sum(s["kind"] == "measurement" for s in samples),
        "samples": samples, "metrics": metrics, "limitations": payload.get("limitations", ["No completed target measurement."]),
        "execution": execution})


@dataclass(frozen=True, slots=True, init=False)
class BenchmarkRun:
    """Immutable exact cohort, including blocked and failed cases."""

    _bytes: bytes

    def __init__(self, document):
        if (not isinstance(document, Mapping) or set(document) != {"schema", "suite", "cohort", "complete", "results"}
                or document["schema"] != "pllm.benchmark_run.v1" or type(document["complete"]) is not bool):
            raise ValueError("invalid benchmark run")
        suite = BenchmarkSuite.from_spec(document["suite"])
        results = [BenchmarkResult.from_dict(r) for r in document["results"]]
        expected = {(case.name, rep) for case in suite.cases for rep in range(suite.sampling.processes)}
        actual = set()
        cases = {case.name: case for case in suite.cases}
        for result in results:
            e = result.to_dict()["execution"]
            name = e["case"]["name"]
            key = (name, e["replicate"])
            if (key not in expected or key in actual or e["case"] != cases[name].to_spec()
                    or e["suite_digest"] != suite.digest or e["cohort"] != document["cohort"]
                    or e["sampling"] != suite.sampling.to_spec()):
                raise ValueError("benchmark result differs from its cohort")
            actual.add(key)
        if document["complete"] and actual != expected:
            raise ValueError("complete benchmark run is missing trials")
        object.__setattr__(self, "_bytes", canonical(document))

    def to_dict(self):
        return json.loads(self._bytes)

    @property
    def results(self):
        return tuple(BenchmarkResult.from_dict(r) for r in self.to_dict()["results"])

    @property
    def digest(self):
        return hashlib.sha256(self._bytes).hexdigest()

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as source:
            payload = source.read((256 << 20) + 1)
        if len(payload) > 256 << 20:
            raise ValueError("benchmark run exceeds 256 MiB")
        return cls(json.loads(payload))

    def compare(self, control, candidate, metric, *, allow_build_changes=False):
        """Paired ratios only inside the exact cohort and measurement contract."""
        import statistics
        if not self.to_dict()["complete"]:
            return {"status": "unranked", "reason": "incomplete_cohort"}
        groups = {name: {} for name in (control, candidate)}
        for result in self.results:
            e = result.to_dict()["execution"]
            if e["case"]["name"] in groups:
                groups[e["case"]["name"]][e["replicate"]] = result
        expected = set(range(self.to_dict()["suite"]["sampling"]["processes"]))
        if any(set(group) != expected for group in groups.values()):
            return {"status": "unranked", "reason": "missing_case"}
        a, b, ratios = [], [], []
        for rep in sorted(expected):
            left, right = groups[control][rep], groups[candidate][rep]
            key = comparable_key(left, allow_build_changes=allow_build_changes)
            if key is None or key != comparable_key(right, allow_build_changes=allow_build_changes):
                return {"status": "unranked", "reason": "cohort_contract_differs"}
            x = next((m for m in left.to_dict()["metrics"] if m["id"] == metric), None)
            y = next((m for m in right.to_dict()["metrics"] if m["id"] == metric), None)
            if (x is None or y is None or x["value"] is None or y["value"] is None
                    or metric_contract(x) != metric_contract(y) or x["origin"] not in {"measured", "deterministic"}):
                return {"status": "unranked", "reason": "measurement_contract_differs"}
            a.append(x["value"])
            b.append(y["value"])
            ratios.append(None if y["value"] == 0 else x["value"] / y["value"])
        return {"status": "comparable", "control_median": statistics.median(a), "candidate_median": statistics.median(b),
            "paired_control_over_candidate": ratios, "process_pairs": len(a),
            "scope": "target-local", "confirmation": "separate finalist cohort required"}


def _checkpoint(path, document):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(canonical(document))
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def benchmark_suite(suite, *, inputs=None, output=None):
    """Run balanced fresh workers; persist only public specs and commitments.

    ``inputs`` maps case names to private input dictionaries. Completed trials
    checkpoint atomically; interruption never resumes one-use runtime state.
    """
    if not isinstance(suite, BenchmarkSuite):
        raise TypeError("benchmark_suite requires a BenchmarkSuite")
    if inputs is not None and (not isinstance(inputs, Mapping) or set(inputs) - {c.name for c in suite.cases}):
        raise ValueError("inputs must map declared case names")
    inputs = dict(inputs or {})
    if any(not isinstance(v, Mapping) for v in inputs.values()) or len(canonical(inputs)) > 16 << 20:
        raise ValueError("private suite inputs must be bounded mappings")
    if output is not None and Path(output).exists():
        raise FileExistsError("benchmark output already exists")
    salt = secrets.token_bytes(32)
    cohort = hashlib.sha256(salt).hexdigest()
    builds = {python: inspect_build(BuildRef(python)) for python in dict.fromkeys(c.build.python for c in suite.cases)}
    document = {"schema": "pllm.benchmark_run.v1", "suite": suite.to_spec(), "cohort": cohort, "complete": False, "results": []}
    _checkpoint(output, document)
    try:
        for rep in range(suite.sampling.processes):
            cases = list(suite.cases)
            if suite.sampling.order == "balanced":
                offset = (rep // 2) % len(cases)
                cases = cases[offset:] + cases[:offset]
                if rep % 2:
                    cases.reverse()
            for case in cases:
                build = builds[case.build.python]
                if case.build.expected_digest is not None and build["digest"] != case.build.expected_digest:
                    payload = {"status": "blocked", "failure_code": "build_mismatch"}
                else:
                    with tempfile.TemporaryDirectory(prefix="pllm-benchmark-") as folder:
                        try:
                            payload = _execute(case, suite.sampling, dict(inputs.get(case.name, {})), salt, Path(folder))
                        except KeyboardInterrupt:
                            payload = {"status": "cancelled", "failure_code": "cancelled"}
                            document["results"].append(_result(case, suite, rep, cohort, build, payload).to_dict())
                            raise
                document["results"].append(_result(case, suite, rep, cohort, build, payload).to_dict())
                _checkpoint(output, document)
        # Editable installs can mutate during a run. Such trials remain evidence,
        # but never become completed measurements attributed to a stale lock.
        for python, before in builds.items():
            after = inspect_build(BuildRef(python))
            if after["digest"] != before["digest"]:
                for row in document["results"]:
                    if row["execution"]["case"]["build"]["python"] == python:
                        row["status"] = "failed"
                        row["execution"]["failure_code"] = "build_changed_during_run"
        document["complete"] = True
        result = BenchmarkRun(document)
        _checkpoint(output, document)
        return result
    finally:
        if not document["complete"]:
            _checkpoint(output, document)


def benchmark_target(target, *, sampling=Sampling(), build=BuildRef(), budgets=(), inputs=None, output=None):
    if not isinstance(target, BenchmarkTarget):
        raise TypeError("benchmark_target requires a BenchmarkTarget")
    return benchmark_suite(BenchmarkSuite((BenchmarkCase("target", target, build, budgets),), sampling),
                           inputs={"target": dict(inputs or {})}, output=output)
