"""Non-interactive loopback benchmark orchestration for the CLI."""

from __future__ import annotations

import hashlib
import json
import math
import secrets
import signal
import socket
import statistics
import threading
import time
import webbrowser
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
import uvicorn

if TYPE_CHECKING:
    from pllm.configuration import Experiment


REPORT_SCHEMA = "pllm.loopback_benchmark.v1"


def accounted_benchmark_body_totals(accounting: Mapping[str, Any]) -> dict[str, int | None]:
    """Charge covered startup once plus ordered windows; missing phases stay unknown."""
    startup = accounting.get("startup")
    windows = [*accounting.get("warmups", []), *accounting.get("runs", [])]
    rows = [startup, *windows]
    complete = bool(windows) and all(
        type(row) is dict
        and row.get("tracked_body_counter_set_present") is True
        and type(row.get("all_link_serialized_body_bytes")) is int
        and row["all_link_serialized_body_bytes"] >= 0
        for row in rows
    )
    return {
        "total_accounted_benchmark_body_bytes": sum(
            row["all_link_serialized_body_bytes"] for row in rows
        )
        if complete
        else None,
        "accounted_setup_through_first_response_body_bytes": sum(
            row["all_link_serialized_body_bytes"] for row in rows[:2]
        )
        if complete
        else None,
    }


COMPARISON_REPORT_SCHEMA = "pllm.loopback_benchmark_comparison.v1"
ProgressCallback = Callable[[str], None]


class LoopbackBenchmarkError(RuntimeError):
    """Raised when the local benchmark roles cannot produce a complete run."""


def _generation_metadata(value: dict[str, Any], *, capture_output_digest: bool) -> dict[str, Any]:
    """Terminal status by default; fingerprints only for opted-in public diagnostics."""
    result = {"response_status": value.get("response_status")}
    if capture_output_digest and "output_text_digest" in value:
        result["output_text_digest"] = value["output_text_digest"]
    return result


def _report_record(record: dict[str, Any], *, capture_output_digest: bool) -> dict[str, Any]:
    result = dict(record)
    if isinstance(result.get("generation"), dict):
        result["generation"] = _generation_metadata(
            result["generation"], capture_output_digest=capture_output_digest
        )
    return result


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
    inventory_policy: str = "prewarm",
    bundle_compression: str = "none",
    prefill_cache_mib: int = 0,
    prefill_cache_mode: str = "exact",
    prefill_cache_bound_tokens: int | None = None,
    warmup_prompt_digest: str | None = None,
    prompt_digest: str | None = None,
    prompt_sequence_digest: str | None = None,
    source_lock_digest: str | None = None,
    client_model_ownership: dict[str, int | None] | None = None,
    cold_process_cpu: dict[str, dict[str, float | None] | None] | None = None,
    stage_snapshots: dict[str, Any] | None = None,
    temperature: float | None = None,
    capture_output_digest: bool = False,
) -> dict[str, Any]:
    """Build a text-free report from dashboard benchmark records."""
    from .topology_accounting import (
        client_owned_body_accounting,
        cold_process_cpu_accounting,
        prepared_body_accounting,
        prepared_stage_body_attribution,
        two_worker_body_accounting,
    )

    from .dashboard import _sampling_choice, _validate_output_digest_capture

    capture_output_digest = _validate_output_digest_capture(capture_output_digest)
    sampling = _sampling_choice(temperature)
    warmup_runs = [
        _report_record(run, capture_output_digest=capture_output_digest) for run in warmup_runs
    ]
    runs = [_report_record(run, capture_output_digest=capture_output_digest) for run in runs]
    all_runs = [*warmup_runs, *runs]
    checks = {
        "all_runs_completed": all(record.get("status") == "completed" for record in all_runs),
        "sampling_matches_request": all(
            record.get("sampling", sampling) == sampling for record in all_runs
        ),
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
    account = (
        client_owned_body_accounting
        if roles == ("client",)
        else two_worker_body_accounting
        if set(roles) == {"client", "worker_a", "worker_b"}
        else prepared_body_accounting
        if set(roles) == {"client", "preparation", "inference"}
        else None
    )
    body_records = [account(record) for record in runs] if account is not None else []
    online_bodies = [row["online_all_link_serialized_body_bytes"] for row in body_records]
    covered_bodies = [row["all_link_serialized_body_bytes"] for row in body_records]
    median_online_bodies = (
        statistics.median(online_bodies)
        if online_bodies and all(type(value) is int and value >= 0 for value in online_bodies)
        else None
    )
    median_covered_bodies = (
        statistics.median(covered_bodies)
        if covered_bodies
        and all(
            row.get("tracked_body_counter_set_present") is True
            and type(value) is int
            and value >= 0
            for row, value in zip(body_records, covered_bodies, strict=True)
        )
        else None
    )
    stage_attribution = None
    if "preparation" in roles and stage_snapshots is not None:
        before_ready = stage_snapshots.get("before_ready", {})
        after_ready = stage_snapshots.get("after_ready", {})
        warm_pairs = stage_snapshots.get("warmups", [])
        run_pairs = stage_snapshots.get("runs", [])
        if len(warm_pairs) != len(warmup_runs) or len(run_pairs) != len(runs):
            raise ValueError("stage attribution does not cover the benchmark windows")
        stage_attribution = {
            "startup": (
                prepared_stage_body_attribution(
                    before_ready, after_ready, initial_preparation_audit
                )
                if initial_preparation_audit is not None
                else None
            ),
            "warmups": [
                prepared_stage_body_attribution(before, after, run["privacy"])
                for (before, after), run in zip(warm_pairs, warmup_runs, strict=True)
            ],
            "runs": [
                prepared_stage_body_attribution(before, after, run["privacy"])
                for (before, after), run in zip(run_pairs, runs, strict=True)
            ],
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "single-host-loopback-diagnostic",
        "configuration": {
            "model_id": model_id,
            "tiny": tiny,
            "max_output_tokens": max_output_tokens,
            "warmups": len(warmup_runs),
            "repetitions": len(runs),
            "roles": list(roles),
            "inventory_policy": inventory_policy,
            "bundle_compression": bundle_compression,
            "prefill_cache_mib": prefill_cache_mib,
            "prefill_cache_mode": prefill_cache_mode,
            "prefill_cache_bound_tokens": prefill_cache_bound_tokens,
            "warmup_prompt_digest": warmup_prompt_digest,
            "prompt_digest": prompt_digest,
            "prompt_sequence_digest": prompt_sequence_digest,
            "source_lock_digest": source_lock_digest,
            "sampling": sampling,
            "capture_output_digest": capture_output_digest,
        },
        "checks": {"passed": all(checks.values()), **checks},
        "privacy_admission": (
            {
                "independent_operators_verified": False,
                "reason": "loopback offset workers share one operator and host",
            }
            if set(roles) == {"client", "worker_a", "worker_b"}
            else None
        ),
        "summary": {
            "completed_runs": sum(record.get("status") == "completed" for record in runs),
            "median_full_seconds": _median(runs, "durations", "full_seconds"),
            "median_online_seconds": _median(runs, "durations", "online_seconds"),
            "median_ttft_seconds": _median(runs, "durations", "ttft_seconds"),
            "median_tokens_per_second": _median(runs, "durations", "tokens_per_second"),
            "median_online_all_link_serialized_body_bytes": median_online_bodies,
            "median_covered_all_link_serialized_body_bytes": median_covered_bodies,
            "total_run_online_all_link_serialized_body_bytes": (
                sum(online_bodies) if median_online_bodies is not None else None
            ),
            "total_run_covered_all_link_serialized_body_bytes": (
                sum(covered_bodies) if median_covered_bodies is not None else None
            ),
            "total_run_full_seconds": (
                sum(record["durations"]["full_seconds"] for record in runs)
                if all(
                    record.get("durations", {}).get("full_seconds") is not None for record in runs
                )
                else None
            ),
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
            {
                "startup": (
                    {"schema": "pllm.topology_model_ownership.v1", **client_model_ownership}
                    if client_model_ownership is not None
                    else None
                ),
                "warmups": [client_owned_body_accounting(run) for run in warmup_runs],
                "runs": [client_owned_body_accounting(run) for run in runs],
            }
            if roles == ("client",)
            else {
                "startup": (
                    prepared_body_accounting(
                        {"privacy": initial_preparation_audit},
                        initial_preparation=True,
                    )
                    if initial_preparation_audit is not None
                    else None
                ),
                "warmups": [prepared_body_accounting(run) for run in warmup_runs],
                "runs": [prepared_body_accounting(run) for run in runs],
                "stages": stage_attribution,
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
    if report["topology_accounting"] is not None:
        report["summary"].update(accounted_benchmark_body_totals(report["topology_accounting"]))
    return report


def _comparison_key(report: dict[str, Any]) -> tuple[object, ...] | None:
    runs = report.get("runs")
    if not isinstance(runs, list) or not runs:
        return None
    sampling = report.get("configuration", {}).get("sampling")
    if not isinstance(sampling, dict):
        return None  # Historical unreported sampling cannot be silently ranked.
    effective: Any = sampling.get("effective_temperature")
    if (
        type(effective) not in (int, float)
        or not 0 <= effective <= 2
        or not math.isfinite(effective)
        or sampling.get("mode") != ("greedy" if effective == 0 else "temperature")
        or sampling.get("top_p") is not None
        or any(run.get("sampling", sampling) != sampling for run in runs)
    ):
        return None
    sampling_key = (float(effective), sampling["mode"], None)
    if report.get("configuration", {}).get("prompt_sequence_digest") is not None:
        if len({run.get("model_fingerprint") for run in runs}) != 1:
            return None
        return (
            runs[0].get("model_fingerprint"),
            tuple(run.get("tokens", {}).get("input_tokens") for run in runs),
            tuple(run.get("tokens", {}).get("output_tokens") for run in runs),
            tuple(run.get("max_output_tokens") for run in runs),
            tuple(run.get("warm") for run in runs),
            report["configuration"].get("warmup_prompt_digest"),
            report["configuration"]["prompt_sequence_digest"],
            sampling_key,
        )
    keys = {
        (
            run.get("model_fingerprint"),
            run.get("tokens", {}).get("input_tokens"),
            run.get("tokens", {}).get("output_tokens"),
            run.get("max_output_tokens"),
            run.get("warm"),
            report.get("configuration", {}).get("warmup_prompt_digest"),
            report.get("configuration", {}).get("prompt_digest"),
            sampling_key,
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
        "online_all_link_serialized_body_bytes": (
            "median_online_all_link_serialized_body_bytes",
            False,
        ),
        "covered_all_link_serialized_body_bytes": (
            "median_covered_all_link_serialized_body_bytes",
            False,
        ),
        "accounted_setup_through_first_response_body_bytes": (
            "accounted_setup_through_first_response_body_bytes",
            False,
        ),
        "total_accounted_benchmark_body_bytes": ("total_accounted_benchmark_body_bytes", False),
    }
    if candidates[0][1].get("configuration", {}).get("prompt_sequence_digest") is not None:
        metrics.update(
            {
                "sequence_full_seconds": ("total_run_full_seconds", False),
                "sequence_online_all_link_serialized_body_bytes": (
                    "total_run_online_all_link_serialized_body_bytes",
                    False,
                ),
                "sequence_covered_all_link_serialized_body_bytes": (
                    "total_run_covered_all_link_serialized_body_bytes",
                    False,
                ),
            }
        )
    rankings: dict[str, list[dict[str, Any]]] = {metric: [] for metric in metrics}
    winners: dict[str, str | None] = {metric: None for metric in metrics}
    if (
        comparable
        and matched_backend
        and all(report.get("checks", {}).get("passed") is True for _, report in candidates)
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
        effective_sampling = cast(tuple[float, str, None], comparison_key[7])
        comparison = {
            "model_fingerprint": comparison_key[0],
            "input_tokens": comparison_key[1],
            "output_tokens": comparison_key[2],
            "max_output_tokens": comparison_key[3],
            "warm": comparison_key[4],
            "warmup_prompt_digest": comparison_key[5],
            "prompt_digest": comparison_key[6],
            "effective_sampling": {
                "temperature": effective_sampling[0],
                "mode": effective_sampling[1],
                "top_p": effective_sampling[2],
            },
        }
    offset_references = [
        record
        for (experiment, _), record in zip(candidates, records, strict=True)
        if (
            experiment.pipeline.components.get("topology") is not None
            and experiment.pipeline.components["topology"].component
            == "pllm/two-online-offset-workers/v1"
        )
    ]
    cpu_comparison: dict[str, Any] | None = None
    if (
        comparable
        and matched_backend
        and checks["all_candidates_passed"]
        and len(offset_references) == 1
    ):
        baseline = offset_references[0]
        reference_cpu = (
            baseline["report"]
            .get("process_cpu_accounting", {})
            .get("aggregate_cold_first_response_cpu_seconds")
        )
        if (
            type(reference_cpu) in (int, float)
            and math.isfinite(reference_cpu)
            and reference_cpu > 0
        ):
            observations: list[dict[str, Any]] = []
            for record in records:
                value = (
                    record["report"]
                    .get("process_cpu_accounting", {})
                    .get("aggregate_cold_first_response_cpu_seconds")
                )
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    observations = []
                    break
                observations.append(
                    {
                        "configuration_digest": record["configuration_digest"],
                        "cpu_seconds": value,
                        "ratio_to_offset": value / reference_cpu,
                        "measured_cpu_not_above_offset": value <= reference_cpu,
                    }
                )
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
    temperature: float | None = None,
    capture_output_digest: bool = False,
) -> dict[str, Any]:
    from .dashboard import _validate_output_digest_capture, _validate_request_temperature

    capture_output_digest = _validate_output_digest_capture(capture_output_digest)
    temperature = _validate_request_temperature(temperature)
    run_id = f"bench-{secrets.token_hex(16)}"
    payload: dict[str, Any] = {
        "prompt": prompt,
        "max_output_tokens": max_output_tokens,
        "request_id": run_id,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    response = client.post(
        "/api/run",
        json=payload,
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
        # Ephemeral metadata stays outside the immutable history schema. Text-free.
        if isinstance(run.get("sampling"), dict):
            record["sampling"] = run["sampling"]
        if isinstance(run.get("generation"), dict):
            record["generation"] = run["generation"]
        return _report_record(record, capture_output_digest=capture_output_digest)
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
    inventory_policy: str | None = None,
    bundle_compression: str | None = None,
    prefill_cache_mib: int = 0,
    prefill_cache_mode: str = "exact",
    prefill_cache_bound_tokens: int | None = None,
    warmup_prompt: str | None = None,
    prompt_sequence: tuple[str, ...] | list[str] | None = None,
    _cohort_salt: bytes | None = None,
    temperature: float | None = None,
    capture_output_digest: bool = False,
) -> dict[str, Any]:
    from .dashboard import _validate_output_digest_capture, _validate_request_temperature

    capture_output_digest = _validate_output_digest_capture(capture_output_digest)
    temperature = _validate_request_temperature(temperature)
    if prompt_sequence is not None:
        if (
            not isinstance(prompt_sequence, (list, tuple))
            or not 1 <= len(prompt_sequence) <= 32
            or any(
                not isinstance(value, str) or not value.strip() or len(value.encode()) > 16_384
                for value in prompt_sequence
            )
        ):
            raise ValueError(
                "prompt sequence requires 1-32 nonempty contexts of at most 16384 bytes"
            )
        if (
            experiment is not None
            and warmups + repetitions * len(prompt_sequence) > experiment.budget.requests
        ):
            raise ValueError("prompt sequence exceeds the Experiment request budget")
        prompt_sequence = tuple(prompt_sequence)
        prompt = prompt_sequence[0]
    selected_inventory_rows = None
    if experiment is not None:
        resolved = experiment.resolve()
        if "inventory" in experiment.pipeline.components:
            if inventory_policy is not None and inventory_policy != resolved.inventory_policy:
                raise ValueError("benchmark inventory flag conflicts with immutable Experiment")
            inventory_policy = resolved.inventory_policy
            selected_inventory_rows = resolved.prepared_inventory_rows
        if "delivery" in experiment.pipeline.components:
            if bundle_compression is not None and bundle_compression != resolved.bundle_compression:
                raise ValueError("benchmark bundle encoding conflicts with immutable Experiment")
            bundle_compression = resolved.bundle_compression
    inventory_policy = inventory_policy or "prewarm"
    bundle_compression = bundle_compression or "none"
    if inventory_policy not in {"prewarm", "request-sized"}:
        raise ValueError("inventory policy must be prewarm or request-sized")
    if bundle_compression not in {"none", "zlib"}:
        raise ValueError("bundle compression must be none or zlib")
    if type(prefill_cache_mib) is not int or not 0 <= prefill_cache_mib <= 256:
        raise ValueError("prefill cache must be in [0, 256] MiB")
    if experiment is not None:
        resolved = experiment.resolve()
        if resolved.prefix_cache_bytes:
            if prefill_cache_mib not in {0, resolved.prefix_cache_bytes >> 20} or (
                prefill_cache_bound_tokens is not None
                and prefill_cache_bound_tokens != resolved.prefix_cache_bound_tokens
            ):
                raise ValueError("benchmark cache flags conflict with immutable Experiment")
            prefill_cache_mib = max(1, resolved.prefix_cache_bytes >> 20)
            prefill_cache_mode = "prefix"
            prefill_cache_bound_tokens = resolved.prefix_cache_bound_tokens
    if prefill_cache_mode == "prefix":
        if (
            prefill_cache_mib < 1
            or type(prefill_cache_bound_tokens) is not int
            or not 2 <= prefill_cache_bound_tokens <= 4096
        ):
            raise ValueError("prefix cache requires memory and a fixed 2-4096 token input bound")
    elif prefill_cache_mode != "exact" or prefill_cache_bound_tokens is not None:
        raise ValueError("prefix cache bound requires prefix mode")
    if warmup_prompt is not None and (not warmups or not warmup_prompt.strip()):
        raise ValueError("distinct warmup prompt requires at least one warmup")
    if _cohort_salt is None:
        _cohort_salt = secrets.token_bytes(32)
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
        min(64, len(prompt.encode("utf-8")) + max_output_tokens + 20) if effective_tiny else 64
    )
    if selected_inventory_rows is not None:
        startup_inventory_rows = selected_inventory_rows
    from pllm.runtime.dashboard import DashboardConfig, create_dashboard_app
    from pllm.runtime.telemetry import local_stage_body_snapshot

    candidate = Path(model).expanduser()
    model_path = None if effective_tiny else candidate.resolve() if candidate.exists() else model
    config = DashboardConfig(
        host="127.0.0.1",
        port=port,
        model_path=model_path,
        model_id=resolved_model_id,
        default_max_output_tokens=max_output_tokens,
        history_path=":memory:",
        startup_inventory_rows=startup_inventory_rows,
        defer_inventory_until_request=inventory_policy == "request-sized",
        bundle_compression=bundle_compression,
        prefill_cache_mib=prefill_cache_mib,
        prefill_cache_mode=prefill_cache_mode,
        prefill_cache_bound_tokens=prefill_cache_bound_tokens,
        experiment=experiment,
        temperature=temperature,
        capture_output_digest=capture_output_digest,
        otel_token=_DASHBOARD_TOKEN,
    )
    origin = f"http://127.0.0.1:{port}"
    dashboard_app = create_dashboard_app(config)

    def stage_snapshot() -> dict[tuple[str, str, str], int]:
        return dashboard_app.state.dashboard_runtime.store.stage_body_snapshot(
            client_local=local_stage_body_snapshot()
        )

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
            stage_snapshots: dict[str, Any] | None = (
                {"before_ready": stage_snapshot(), "warmups": [], "runs": []}
                if "preparation" in roles
                else None
            )
            _wait_for_ready(
                client,
                handle,
                time.monotonic() + timeout_seconds,
                progress,
            )
            if stage_snapshots is not None:
                stage_snapshots["after_ready"] = stage_snapshot()
            initial_preparation_audit = (
                dashboard_app.state.dashboard_runtime.initial_preparation_audit()
                if "preparation" in roles
                else None
            )
            client_model_ownership = (
                dashboard_app.state.dashboard_runtime.client_model_ownership()
                if roles == ("client",)
                else None
            )
            warmup_runs = []
            for index in range(warmups):
                if progress is not None:
                    progress(f"Running warmup {index + 1}/{warmups}")
                before = stage_snapshot() if stage_snapshots is not None else None
                run = _run_once(
                    client,
                    handle,
                    prompt=warmup_prompt if warmup_prompt is not None else prompt,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout_seconds,
                    progress=progress,
                    progress_label=f"Warmup {index + 1}/{warmups} running",
                    temperature=temperature,
                    capture_output_digest=capture_output_digest,
                )
                warmup_runs.append(run)
                if stage_snapshots is not None and before is not None:
                    after = stage_snapshot()
                    stage_snapshots["warmups"].append((before, after))
            runs = []
            measured_prompts = (prompt,) if prompt_sequence is None else prompt_sequence
            for index in range(repetitions * len(measured_prompts)):
                current_prompt = measured_prompts[index % len(measured_prompts)]
                if progress is not None:
                    progress(
                        f"Running measurement {index + 1}/{repetitions * len(measured_prompts)}"
                    )
                before = stage_snapshot() if stage_snapshots is not None else None
                run = _run_once(
                    client,
                    handle,
                    prompt=current_prompt,
                    max_output_tokens=max_output_tokens,
                    timeout_seconds=timeout_seconds,
                    progress=progress,
                    progress_label=f"Measurement {index + 1}/{repetitions * len(measured_prompts)} running",
                    temperature=temperature,
                    capture_output_digest=capture_output_digest,
                )
                if prompt_sequence is not None:
                    run["context_index"] = index % len(measured_prompts)
                    run["sequence_repetition"] = index // len(measured_prompts)
                runs.append(run)
                if stage_snapshots is not None and before is not None:
                    after = stage_snapshot()
                    stage_snapshots["runs"].append((before, after))
            cold_process_cpu = dashboard_app.state.dashboard_runtime.cold_process_cpu()
            client_body_placement = dashboard_app.state.dashboard_runtime.client_body_placement()
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
        inventory_policy=inventory_policy,
        bundle_compression=bundle_compression,
        prefill_cache_mib=prefill_cache_mib,
        prefill_cache_mode=prefill_cache_mode,
        prefill_cache_bound_tokens=prefill_cache_bound_tokens,
        warmup_prompt_digest=hashlib.sha256(
            b"pllm.benchmark.prompt.v1\0" + _cohort_salt + (warmup_prompt or prompt).encode()
        ).hexdigest(),
        prompt_digest=hashlib.sha256(
            b"pllm.benchmark.prompt.v1\0" + _cohort_salt + prompt.encode()
        ).hexdigest(),
        prompt_sequence_digest=(
            hashlib.sha256(
                b"pllm.benchmark.context-sequence.v1\0"
                + _cohort_salt
                + json.dumps(prompt_sequence, ensure_ascii=False, separators=(",", ":")).encode()
            ).hexdigest()
            if prompt_sequence is not None
            else None
        ),
        source_lock_digest=dashboard_app.state.dashboard_runtime.source_lock_digest,
        client_model_ownership=client_model_ownership,
        cold_process_cpu=cold_process_cpu,
        stage_snapshots=stage_snapshots,
        temperature=temperature,
        capture_output_digest=capture_output_digest,
    )
    if prompt_sequence is not None:
        report["configuration"].update(
            sequence_length=len(prompt_sequence), sequence_repetitions=repetitions
        )
    report["client_body_placement"] = client_body_placement
    if experiment is not None:
        report["experiment"] = {
            "name": experiment.name,
            "configuration_digest": experiment.configuration_digest(),
            "pipeline_digest": experiment.pipeline.digest(),
        }
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
    inventory_policy: str | None = None,
    bundle_compression: str | None = None,
    prefill_cache_mib: int = 0,
    prefill_cache_mode: str = "exact",
    prefill_cache_bound_tokens: int | None = None,
    warmup_prompt: str | None = None,
    prompt_sequence: tuple[str, ...] | list[str] | None = None,
    _cohort_salt: bytes | None = None,
    temperature: float | None = None,
    capture_output_digest: bool = False,
) -> dict[str, Any]:
    """Run ordinary loopback roles; None preserves SDK sampling, 0 requests greedy."""
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
            inventory_policy=inventory_policy,
            bundle_compression=bundle_compression,
            prefill_cache_mib=prefill_cache_mib,
            prefill_cache_mode=prefill_cache_mode,
            prefill_cache_bound_tokens=prefill_cache_bound_tokens,
            warmup_prompt=warmup_prompt,
            prompt_sequence=prompt_sequence,
            _cohort_salt=_cohort_salt,
            temperature=temperature,
            capture_output_digest=capture_output_digest,
        )
