"""Non-interactive loopback benchmark orchestration for the CLI."""

from __future__ import annotations

import os
import re
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any, BinaryIO

import httpx


REPORT_SCHEMA = "pllm.loopback_benchmark.v1"
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
) -> dict[str, Any]:
    """Build a text-free report from dashboard benchmark records."""
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
            "roles": ["client", "preparation", "inference"],
        },
        "checks": {"passed": all(checks.values()), **checks},
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
        "limitations": [
            "single host and loopback network",
            "diagnostic record, not a canonical EvidenceReport",
            "does not establish model quality, energy, price, adversarial security, or non-collusion",
        ],
    }


def _wait_for_ready(
    client: httpx.Client,
    process: subprocess.Popen[bytes],
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
    process: subprocess.Popen[bytes],
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
            raise LoopbackBenchmarkError("benchmark record could not be read") from exc
        if record.get("status") != "completed":
            failure = record.get("failure", {}).get("type") or "unknown failure"
            raise LoopbackBenchmarkError(f"benchmark run failed: {failure}")
        return record
    raise LoopbackBenchmarkError("benchmark run did not complete before timeout")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGINT)
    else:
        process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)


def _diagnostic_log(log: BinaryIO) -> str:
    log.flush()
    log.seek(0, os.SEEK_END)
    size = log.tell()
    log.seek(max(0, size - 4_000))
    text = log.read().decode("utf-8", errors="replace").strip()
    return re.sub(r"dash_[A-Za-z0-9_-]+", "<redacted>", text)


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
) -> dict[str, Any]:
    """Run the real client, preparation, and inference roles on loopback."""
    port = _free_port()
    resolved_model_id = model_id or ("pllm-benchmark-tiny" if tiny else model)
    startup_inventory_rows = (
        min(64, len(prompt.encode("utf-8")) + max_output_tokens + 20) if tiny else 64
    )
    command = [
        sys.executable,
        "-m",
        "pllm",
        "dev",
        "dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-output-tokens",
        str(max_output_tokens),
        "--history-db",
        ":memory:",
        "--startup-inventory-rows",
        str(startup_inventory_rows),
    ]
    if not show_dashboard:
        command.append("--no-open")
    if tiny:
        command.append("--tiny")
    else:
        command.extend(("--model", model))
        if model_id is not None:
            command.extend(("--model-id", model_id))

    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        previous_sigterm: Any = None

        def terminate(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        if threading.current_thread() is threading.main_thread():
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, terminate)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
                _wait_for_ready(
                    client,
                    process,
                    time.monotonic() + timeout_seconds,
                    progress,
                )
                warmup_runs = []
                for index in range(warmups):
                    if progress is not None:
                        progress(f"Running warmup {index + 1}/{warmups}")
                    warmup_runs.append(
                        _run_once(
                            client,
                            process,
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
                            process,
                            prompt=prompt,
                            max_output_tokens=max_output_tokens,
                            timeout_seconds=timeout_seconds,
                            progress=progress,
                            progress_label=f"Measurement {index + 1}/{repetitions} running",
                        )
                    )
        except KeyboardInterrupt as exc:
            raise LoopbackBenchmarkError("benchmark interrupted") from exc
        except LoopbackBenchmarkError as exc:
            diagnostic = _diagnostic_log(log)
            suffix = f"\n{diagnostic}" if diagnostic else ""
            raise LoopbackBenchmarkError(f"{exc}{suffix}") from exc
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
            if progress is not None:
                progress("Stopping benchmark roles")
            _stop_process(process)

    report = build_loopback_report(
        model_id=resolved_model_id,
        tiny=tiny,
        max_output_tokens=max_output_tokens,
        warmup_runs=warmup_runs,
        runs=runs,
    )
    if not report["checks"]["passed"]:
        raise LoopbackBenchmarkError("benchmark runtime or privacy checks failed")
    return report
