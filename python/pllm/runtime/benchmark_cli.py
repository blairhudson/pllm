"""Non-interactive loopback benchmark orchestration for the CLI."""

from __future__ import annotations

import math
import secrets
import signal
import socket
import statistics
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import uvicorn

if TYPE_CHECKING:
    from pllm.configuration import Experiment


REPORT_SCHEMA = "pllm.loopback_benchmark.v1"
COMPARISON_REPORT_SCHEMA = "pllm.loopback_benchmark_comparison.v1"
ProgressCallback = Callable[[str], None]


class LoopbackBenchmarkError(RuntimeError):
    """Raised when the local benchmark roles cannot produce a complete run."""


def _run_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    run = snapshot.get("run")
    if not isinstance(run, dict):
        raise LoopbackBenchmarkError("benchmark dashboard returned an invalid snapshot")
    return run


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


_DASHBOARD_PORT: int | None = None
_DASHBOARD_TOKEN: str | None = None
_DASHBOARD_LOCK = threading.Lock()


class _DashboardHandle:
    def __init__(self, app: Any, port: int) -> None:
        self.error: BaseException | None = None
        self.server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        self.thread = threading.Thread(
            target=self._run,
            name="pllm-benchmark-dashboard",
            daemon=True,
        )

    def _run(self) -> None:
        try:
            self.server.run()
        except BaseException as exc:
            self.error = exc

    def start(self) -> None:
        self.thread.start()

    def poll(self) -> int | None:
        if self.thread.is_alive():
            return None
        return 1 if self.error is not None else 0

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=20)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise LoopbackBenchmarkError("benchmark dashboard did not stop")

    def diagnostic(self) -> str:
        return "" if self.error is None else f"{type(self.error).__name__}: {self.error}"


def _wait_for_dashboard_listener(handle: _DashboardHandle, origin: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if handle.poll() is not None:
            raise LoopbackBenchmarkError("benchmark dashboard exited during startup")
        try:
            if httpx.get(f"{origin}/api/snapshot", timeout=0.5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    raise LoopbackBenchmarkError("benchmark dashboard did not start before timeout")


def _median(records: list[dict[str, Any]], section: str, key: str) -> float | None:
    values = [record.get(section, {}).get(key) for record in records]
    numbers = [float(value) for value in values if value is not None]
    return statistics.median(numbers) if numbers else None


def build_loopback_report(
    *,
    model_id: str,
    tiny: bool,
    max_output_tokens: int,
    warmup_runs: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    roles: tuple[str, ...] = ("client", "preparation", "inference"),
    initial_preparation_audit: dict[str, int] | None = None,
    client_model_ownership: dict[str, int | None] | None = None,
    cold_process_cpu: dict[str, dict[str, float | None] | None] | None = None,
) -> dict[str, Any]:
    """Build a text-free report from dashboard benchmark records."""
    from .topology_accounting import (
        client_owned_body_accounting, cold_process_cpu_accounting,
        prepared_body_accounting, two_worker_body_accounting,
    )

    all_runs = [*warmup_runs, *runs]
    checks = {
        "all_runs_completed": all(record.get("status") == "completed" for record in all_runs),
        "authoritative_token_usage": all(
            record.get("tokens", {}).get("authoritative") is True for record in all_runs
        ),
        "model_fingerprint_present": all(
            bool(record.get("model_fingerprint")) for record in all_runs
        ),
        "no_plaintext_prompt_bytes_sent": all(
            record.get("privacy", {}).get("plaintext_prompt_bytes_sent") == 0 for record in all_runs
        ),
        "no_plaintext_token_ids_sent": all(
            record.get("privacy", {}).get("plaintext_token_ids_sent") == 0 for record in all_runs
        ),
        "no_online_preparation_requests": all(
            record.get("privacy", {}).get("preparation_requests_during_online") == 0
            for record in all_runs
        ),
    }
    if roles == ("client",):
        checks["no_provider_stage_traffic"] = all(
            client_owned_body_accounting(record)["tracked_body_counter_set_present"]
            for record in all_runs
        )
    if set(roles) == {"client", "worker_a", "worker_b"}:
        checks["two_worker_bodies_tracked"] = all(
            two_worker_body_accounting(record)["tracked_body_counter_set_present"]
            for record in all_runs
        )
    output_tokens = [
        int(record["tokens"]["output_tokens"])
        for record in runs
        if record.get("tokens", {}).get("output_tokens") is not None
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "scope": "single-host-loopback-diagnostic",
        "configuration": {
            "model_id": model_id,
            "tiny": tiny,
            "max_output_tokens": max_output_tokens,
            "warmups": len(warmup_runs),
            "repetitions": len(runs),
            "roles": list(roles),
        },
        "checks": {"passed": all(checks.values()), **checks},
        "privacy_admission": (
            {"independent_operators_verified": False,
             "reason": "loopback offset workers share one operator and host"}
            if set(roles) == {"client", "worker_a", "worker_b"} else None
        ),
        "summary": {
            "completed_runs": sum(record.get("status") == "completed" for record in runs),
            "median_full_seconds": _median(runs, "durations", "full_seconds"),
            "median_online_seconds": _median(runs, "durations", "online_seconds"),
            "median_ttft_seconds": _median(runs, "durations", "ttft_seconds"),
            "median_tokens_per_second": _median(runs, "durations", "tokens_per_second"),
            "total_output_tokens": sum(output_tokens),
        },
        "warmup_runs": warmup_runs,
        "runs": runs,
        "process_cpu_accounting": cold_process_cpu_accounting(
            cold_process_cpu,
            roles=roles,
            first_measurement_is_cold=(
                not warmup_runs and bool(runs) and runs[0].get("cold") is True
            ),
        ),
        "topology_accounting": (
            {"startup": (
                {"schema": "pllm.topology_model_ownership.v1", **client_model_ownership}
                if client_model_ownership is not None else None
             ),
             "warmups": [client_owned_body_accounting(run) for run in warmup_runs],
             "runs": [client_owned_body_accounting(run) for run in runs]}
            if roles == ("client",) else {
                "startup": (
                    prepared_body_accounting(
                        {"privacy": initial_preparation_audit}, initial_preparation=True,
                    )
                    if initial_preparation_audit is not None else None
                ),
                "warmups": [prepared_body_accounting(run) for run in warmup_runs],
                "runs": [prepared_body_accounting(run) for run in runs],
            }
            if "preparation" in roles
            else {
                "startup": None,
                "warmups": [two_worker_body_accounting(run) for run in warmup_runs],
                "runs": [two_worker_body_accounting(run) for run in runs],
            }
            if set(roles) == {"client", "worker_a", "worker_b"}
            else None
        ),
        "limitations": [
            "single host and loopback network",
            "diagnostic record, not a canonical EvidenceReport",
            "does not establish model quality, energy, price, adversarial security, or non-collusion",
        ],
    }


def _comparison_key(report: dict[str, Any]) -> tuple[object, ...] | None:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    keys = {
        (
            run.get("model_fingerprint"),
            run.get("tokens", {}).get("input_tokens"),
            run.get("tokens", {}).get("output_tokens"),
            run.get("max_output_tokens"),
            run.get("warm"),
        )
        for run in runs
    }
    return next(iter(keys)) if len(keys) == 1 else None


def build_comparison_report(
    candidates: list[tuple[Experiment, dict[str, Any]]],
) -> dict[str, Any]:
    """Build one matched report over independently executed Experiment pipelines."""
    if len(candidates) < 2:
        raise ValueError("comparison requires at least two Experiment pipelines")
    records = [
        {
            "name": experiment.name,
            "configuration_digest": experiment.configuration_digest(),
            "pipeline": experiment.pipeline.to_spec(),
            "report": report,
        }
        for experiment, report in candidates
    ]
    digests = [record["configuration_digest"] for record in records]
    names = [record["name"] for record in records]
    if len(set(digests)) != len(digests):
        raise ValueError("comparison Experiment pipelines must have unique configurations")
    if len(set(names)) != len(names):
        raise ValueError("comparison Experiment names must be unique")

    keys = [_comparison_key(report) for _, report in candidates]
    comparable = all(key is not None for key in keys) and len(set(keys)) == 1
    kernel_backends = {
        experiment.pipeline.components["kernels"].component for experiment, _ in candidates
    }
    matched_backend = len(kernel_backends) == 1
    comparison_key = cast(tuple[object, ...], keys[0]) if comparable else None
    metrics = {
        "full_seconds": ("median_full_seconds", False),
        "online_seconds": ("median_online_seconds", False),
        "ttft_seconds": ("median_ttft_seconds", False),
        "tokens_per_second": ("median_tokens_per_second", True),
    }
    rankings: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    winners: dict[str, str | None] = {metric: None for metric in metrics}
    if comparable and matched_backend and all(
        report.get("checks", {}).get("passed") is True for _, report in candidates
    ):
        for metric, (summary_key, reverse) in metrics.items():
            values = [
                (
                    float(record["report"]["summary"][summary_key]),
                    str(record["configuration_digest"]),
                    str(record["name"]),
                )
                for record in records
                if record["report"]["summary"].get(summary_key) is not None
            ]
            values.sort(key=lambda item: ((-item[0]) if reverse else item[0], item[1]))
            rankings[metric] = [
                {
                    "rank": rank,
                    "name": name,
                    "configuration_digest": digest,
                    "value": value,
                }
                for rank, (value, digest, name) in enumerate(values, start=1)
            ]
            if values:
                winners[metric] = values[0][1]

    checks = {
        "all_candidates_passed": all(
            report.get("checks", {}).get("passed") is True for _, report in candidates
        ),
        "unique_configurations": len(set(digests)) == len(digests),
        "matched_workload": comparable,
        "matched_kernel_backend": matched_backend,
    }
    comparison = None
    if comparison_key is not None:
        comparison = {
            "model_fingerprint": comparison_key[0],
            "input_tokens": comparison_key[1],
            "output_tokens": comparison_key[2],
            "max_output_tokens": comparison_key[3],
            "warm": comparison_key[4],
        }
    offset_references = [
        record for (experiment, _), record in zip(candidates, records, strict=True)
        if (
            experiment.pipeline.components.get("topology") is not None
            and experiment.pipeline.components["topology"].component
            == "pllm/two-online-offset-workers/v1"
        )
    ]
    cpu_comparison: dict[str, Any] | None = None
    if comparable and matched_backend and checks["all_candidates_passed"] and len(offset_references) == 1:
        baseline = offset_references[0]
        reference_cpu = baseline["report"].get("process_cpu_accounting", {}).get(
            "aggregate_cold_first_response_cpu_seconds"
        )
        if type(reference_cpu) in (int, float) and math.isfinite(reference_cpu) and reference_cpu > 0:
            observations: list[dict[str, Any]] = []
            for record in records:
                value = record["report"].get("process_cpu_accounting", {}).get(
                    "aggregate_cold_first_response_cpu_seconds"
                )
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    observations = []
                    break
                observations.append({
                    "configuration_digest": record["configuration_digest"],
                    "cpu_seconds": value,
                    "ratio_to_offset": value / reference_cpu,
                    "measured_cpu_not_above_offset": value <= reference_cpu,
                })
            if observations:
                cpu_comparison = {
                    "schema": "pllm.offset_cold_cpu_diagnostic.v1",
                    "reference_configuration_digest": baseline["configuration_digest"],
                    "reference_cpu_seconds": reference_cpu,
                    "observations": observations,
                    "full_response_compute_cap_admitted": False,
                    "scope": "single-host cold process CPU; one measured response per composition",
                    "limitation": (
                        "Diagnostic CPU ratios do not establish independent-operator privacy, "
                        "accelerator compute, full wire cost, or a representative cohort"
                    ),
                }
    return {
        "schema_version": COMPARISON_REPORT_SCHEMA,
        "scope": "single-host-loopback-diagnostic-comparison",
        "checks": {"passed": all(checks.values()), **checks},
        "comparison_key": comparison,
        "compute_cap_diagnostic": cpu_comparison,
        "candidates": records,
        "rankings": rankings,
        "winners": winners,
        "limitations": [
            "single host and loopback network",
            "diagnostic comparison, not a canonical EvidenceReport",
            "rankings require exact matched measured workloads and kernel backends",
            "does not establish model quality, energy, price, adversarial security, or non-collusion",
        ],
    }


def _wait_for_ready(
    client: httpx.Client,
    process: Any,
    deadline: float,
    progress: ProgressCallback | None,
) -> None:
    started = time.monotonic()
    last_status: tuple[str, str] | None = None
    next_update = started
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LoopbackBenchmarkError("benchmark dashboard exited during startup")
        try:
            snapshot = client.get("/api/snapshot").raise_for_status().json()
        except (httpx.HTTPError, ValueError):
            time.sleep(0.2)
            continue
        run = _run_state(snapshot)
        phase = str(run.get("phase", "starting"))
        startup_step = str(run.get("startup_step", "dashboard"))
        if phase == "ready":
            return
        status = (phase, startup_step)
        now = time.monotonic()
        if progress is not None and (status != last_status or now >= next_update):
            labels = {
                "dashboard": "Starting benchmark dashboard",
                "inference": "Loading inference role",
                "preparation": "Loading preparation role",
                "inventory": "Preparing offline inventory",
            }
            label = labels.get(startup_step, f"Starting benchmark ({startup_step})")
            progress(f"{label} ({int(now - started)}s)")
            last_status = status
            next_update = now + 10
        if phase == "error":
            raise LoopbackBenchmarkError(str(run.get("error") or "benchmark startup failed"))
        time.sleep(0.2)
    step = last_status[1] if last_status is not None else "dashboard"
    raise LoopbackBenchmarkError(f"benchmark startup timed out during {step}")


def _run_once(
    client: httpx.Client,
    process: Any,
    *,
    prompt: str,
    max_output_tokens: int,
    timeout_seconds: float,
    progress: ProgressCallback | None = None,
    progress_label: str = "Benchmark run",
) -> dict[str, Any]:
    run_id = f"bench-{secrets.token_hex(16)}"
    response = client.post(
        "/api/run",
        json={
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
            "request_id": run_id,
        },
    )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LoopbackBenchmarkError("benchmark run was rejected") from exc

    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    next_update = started + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LoopbackBenchmarkError("benchmark dashboard exited during a run")
        try:
            snapshot = client.get("/api/snapshot").raise_for_status().json()
        except (httpx.HTTPError, ValueError):
            time.sleep(0.2)
            continue
        run = _run_state(snapshot)
        now = time.monotonic()
        if progress is not None and now >= next_update:
            progress(f"{progress_label} ({int(now - started)}s)")
            next_update = now + 10
        if run.get("run_id") != run_id or run.get("active_run") is not None:
            time.sleep(0.2)
            continue
        if run.get("phase") not in {"ready", "error"}:
            time.sleep(0.2)
            continue
        try:
            record = client.get(f"/api/runs/{run_id}").raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            if run.get("phase") == "error" or run.get("error"):
                raise LoopbackBenchmarkError(
                    str(run.get("error") or "benchmark execution failed")
                ) from exc
            raise LoopbackBenchmarkError("benchmark record could not be read") from exc
        if record.get("status") != "completed":
            failure = record.get("failure", {}).get("type") or "unknown failure"
            raise LoopbackBenchmarkError(f"benchmark run failed: {failure}")
        return record
    raise LoopbackBenchmarkError("benchmark run did not complete before timeout")


def _run_loopback_benchmark(
    *,
    model: str,
    model_id: str | None,
    tiny: bool,
    prompt: str,
    max_output_tokens: int,
    warmups: int,
    repetitions: int,
    timeout_seconds: float,
    show_dashboard: bool = False,
    progress: ProgressCallback | None = None,
    experiment: Experiment | None = None,
) -> dict[str, Any]:
    global _DASHBOARD_PORT, _DASHBOARD_TOKEN
    if _DASHBOARD_PORT is None:
        _DASHBOARD_PORT = _free_port()
    if _DASHBOARD_TOKEN is None:
        _DASHBOARD_TOKEN = secrets.token_urlsafe(32)
    port = _DASHBOARD_PORT
    roles = ("client", "preparation", "inference")
    effective_tiny = tiny
    if experiment is not None:
        if tiny:
            raise ValueError("Experiment pipelines cannot use the generated tiny model")
        experiment.resolve()
        from pllm.profiles import resolve_runtime_composition
        from pllm.roles.topology import graph_for_runtime

        runtime_options = resolve_runtime_composition(experiment.pipeline)
        if runtime_options is None:
            raise ValueError("Experiment profile is not supported by local benchmarking")
        roles = tuple(role.id for role in graph_for_runtime(runtime_options).roles)
        effective_tiny = experiment.pipeline.model.kind == "tiny"
        model = experiment.pipeline.model.source
        resolved_model_id = experiment.pipeline.model.model_id or model
    else:
        resolved_model_id = model_id or ("pllm-benchmark-tiny" if tiny else model)
    startup_inventory_rows = (
        min(64, len(prompt.encode("utf-8")) + max_output_tokens + 20)
        if effective_tiny
        else 64
    )
    from pllm.runtime.dashboard import DashboardConfig, create_dashboard_app

    candidate = Path(model).expanduser()
    model_path = (
        None if effective_tiny else candidate.resolve() if candidate.exists() else model
    )
    config = DashboardConfig(
        host="127.0.0.1",
        port=port,
        model_path=model_path,
        model_id=resolved_model_id,
        default_max_output_tokens=max_output_tokens,
        history_path=":memory:",
        startup_inventory_rows=startup_inventory_rows,
        experiment=experiment,
        otel_token=_DASHBOARD_TOKEN,
    )
    origin = f"http://127.0.0.1:{port}"
    dashboard_app = create_dashboard_app(config)
    handle = _DashboardHandle(dashboard_app, port)
    handle.start()
    try:
        _wait_for_dashboard_listener(handle, origin, timeout_seconds)
    except Exception:
        handle.close()
        _DASHBOARD_PORT = None
        raise
    previous_sigterm: Any = None

    def terminate(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate)
    try:
        if show_dashboard:
            webbrowser.open(origin)
        with httpx.Client(base_url=origin, timeout=5.0) as client:
            _wait_for_ready(
                client,
                handle,
                time.monotonic() + timeout_seconds,
                progress,
            )
            initial_preparation_audit = (
                dashboard_app.state.dashboard_runtime.initial_preparation_audit()
                if "preparation" in roles else None
            )
            client_model_ownership = (
                dashboard_app.state.dashboard_runtime.client_model_ownership()
                if roles == ("client",) else None
            )
            warmup_runs = []
            for index in range(warmups):
                if progress is not None:
                    progress(f"Running warmup {index + 1}/{warmups}")
                warmup_runs.append(
                    _run_once(
                        client,
                        handle,
                        prompt=prompt,
                        max_output_tokens=max_output_tokens,
                        timeout_seconds=timeout_seconds,
                        progress=progress,
                        progress_label=f"Warmup {index + 1}/{warmups} running",
                    )
                )
            runs = []
            for index in range(repetitions):
                if progress is not None:
                    progress(f"Running measurement {index + 1}/{repetitions}")
                runs.append(
                    _run_once(
                        client,
                        handle,
                        prompt=prompt,
                        max_output_tokens=max_output_tokens,
                        timeout_seconds=timeout_seconds,
                        progress=progress,
                        progress_label=f"Measurement {index + 1}/{repetitions} running",
                    )
                )
            cold_process_cpu = dashboard_app.state.dashboard_runtime.cold_process_cpu()
    except KeyboardInterrupt as exc:
        raise LoopbackBenchmarkError("benchmark interrupted") from exc
    except LoopbackBenchmarkError as exc:
        diagnostic = handle.diagnostic()
        suffix = f"\n{diagnostic}" if diagnostic else ""
        raise LoopbackBenchmarkError(f"{exc}{suffix}") from exc
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if progress is not None:
            progress("Stopping benchmark roles")
        handle.close()

    report = build_loopback_report(
        model_id=resolved_model_id,
        tiny=effective_tiny,
        max_output_tokens=max_output_tokens,
        warmup_runs=warmup_runs,
        runs=runs,
        roles=roles,
        initial_preparation_audit=initial_preparation_audit,
        client_model_ownership=client_model_ownership,
        cold_process_cpu=cold_process_cpu,
    )
    if not report["checks"]["passed"]:
        raise LoopbackBenchmarkError("benchmark runtime or privacy checks failed")
    return report


def run_loopback_benchmark(
    *,
    model: str,
    model_id: str | None,
    tiny: bool,
    prompt: str,
    max_output_tokens: int,
    warmups: int,
    repetitions: int,
    timeout_seconds: float,
    show_dashboard: bool = False,
    progress: ProgressCallback | None = None,
    experiment: Experiment | None = None,
) -> dict[str, Any]:
    """Run the profile's real client and service roles on loopback."""
    with _DASHBOARD_LOCK:
        return _run_loopback_benchmark(
            model=model,
            model_id=model_id,
            tiny=tiny,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            warmups=warmups,
            repetitions=repetitions,
            timeout_seconds=timeout_seconds,
            show_dashboard=show_dashboard,
            progress=progress,
            experiment=experiment,
        )
