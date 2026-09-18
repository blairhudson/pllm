from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from pllm._cli.app import build_parser
from pllm.configuration import Experiment
from pllm.runtime.benchmark_cli import (
    _run_once,
    _wait_for_ready,
    build_comparison_report,
    build_loopback_report,
    run_loopback_benchmark,
)


def _record(
    run_id: str, *, ttft: float, throughput: float, full: float = 4.0
) -> dict[str, object]:
    return {
        "schema_version": "pllm.benchmark_run.v3",
        "run_id": run_id,
        "status": "completed",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_fingerprint": "sha256:model",
        "durations": {
            "full_seconds": full,
            "online_seconds": 3.0,
            "ttft_seconds": ttft,
            "tokens_per_second": throughput,
        },
        "tokens": {
            "input_tokens": 39,
            "output_tokens": 24,
            "total_tokens": 63,
            "authoritative": True,
        },
        "privacy": {
            "plaintext_prompt_bytes_sent": 0,
            "plaintext_token_ids_sent": 0,
            "preparation_requests_during_online": 0,
        },
    }


def _report(*, full: float = 4.0) -> dict[str, object]:
    return build_loopback_report(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tiny=False,
        max_output_tokens=24,
        warmup_runs=[],
        runs=[_record("run-1", ttft=0.98, throughput=8.46, full=full)],
    )


def _experiment(name: str, threads: int) -> Experiment:
    return Experiment.from_spec(
        {
            "schema": "pllm.experiment.v1",
            "name": name,
            "pipeline": {
                "profile": "baseline.masked_linear_cpu",
                "model": {"source": "Qwen/Qwen2.5-0.5B-Instruct"},
                "components": {
                    "inference": {"component": "pllm/inference", "params": {}},
                    "kernels": {"component": "pllm/cpu", "params": {"threads": threads}},
                    "linear": {"component": "pllm/masked-linear", "params": {}},
                    "preparation": {
                        "component": "pllm/model-aware-corrections",
                        "params": {},
                    },
                },
            },
            "deployment": {"kind": "local", "root": "local://benchmark"},
            "budget": {"requests": 4, "max_input_tokens": 256, "max_new_tokens": 24},
        }
    )


def test_benchmark_parser_defaults_to_real_qwen() -> None:
    args = build_parser().parse_args(["benchmark", "run"])
    assert args.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert args.tiny is False
    assert args.warmups == 0
    assert args.repetitions == 1
    assert args.show_dashboard is False


def test_benchmark_dashboard_display_is_explicit() -> None:
    args = build_parser().parse_args(["benchmark", "run", "--show-dashboard"])
    assert args.show_dashboard is True


@pytest.mark.integration
def test_tiny_benchmark_runs_in_process_over_shared_role_topology() -> None:
    report = run_loopback_benchmark(
        model="unused",
        model_id=None,
        tiny=True,
        prompt="private",
        max_output_tokens=1,
        warmups=0,
        repetitions=1,
        timeout_seconds=120,
    )
    assert report["checks"]["passed"] is True
    assert report["summary"]["completed_runs"] == 1
    processes = report["runs"][0]["processes"]
    assert set(processes) == {"client", "inference", "preparation"}
    assert processes["client"]["cpu_seconds"] is not None


class _RunningProcess:
    def poll(self) -> None:
        return None


def test_benchmark_polling_reads_nested_dashboard_run_state(monkeypatch) -> None:
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/api/snapshot":
            return httpx.Response(
                200,
                json={
                    "run": {
                        "phase": "ready",
                        "startup_step": "ready",
                        "run_id": "bench-fixed",
                        "active_run": None,
                    },
                    "otel": {},
                },
            )
        if request.url.path == "/api/run":
            return httpx.Response(200, json={"status": "started"})
        if request.url.path == "/api/runs/bench-fixed":
            return httpx.Response(200, json={"status": "completed"})
        raise AssertionError(request.url.path)

    monkeypatch.setattr("pllm.runtime.benchmark_cli.secrets.token_hex", lambda _: "fixed")
    with httpx.Client(
        transport=httpx.MockTransport(respond), base_url="http://dashboard"
    ) as client:
        process: Any = _RunningProcess()
        _wait_for_ready(client, process, float("inf"), None)
        record = _run_once(
            client,
            process,
            prompt="private",
            max_output_tokens=2,
            timeout_seconds=1,
        )

    assert record == {"status": "completed"}
    assert paths == [
        "/api/snapshot",
        "/api/run",
        "/api/snapshot",
        "/api/runs/bench-fixed",
    ]


def test_loopback_report_summarizes_records_and_has_no_text_payloads() -> None:
    report = build_loopback_report(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        tiny=False,
        max_output_tokens=24,
        warmup_runs=[_record("warmup-1", ttft=1.5, throughput=7.0)],
        runs=[
            _record("run-1", ttft=1.0, throughput=8.0),
            _record("run-2", ttft=2.0, throughput=10.0),
        ],
    )

    assert report["schema_version"] == "pllm.loopback_benchmark.v1"
    assert report["checks"]["passed"] is True
    assert report["summary"]["median_ttft_seconds"] == 1.5
    assert report["summary"]["median_tokens_per_second"] == 9.0
    assert report["summary"]["total_output_tokens"] == 48
    serialized = json.dumps(report)
    assert '"prompt"' not in serialized
    assert '"generated_text"' not in serialized
    assert '"token_ids"' not in serialized


def test_comparison_report_ranks_only_matched_pipeline_runs() -> None:
    slow, fast = _experiment("slow", 1), _experiment("fast", 4)
    report = build_comparison_report([(slow, _report(full=5.0)), (fast, _report(full=3.0))])

    assert report["schema_version"] == "pllm.loopback_benchmark_comparison.v1"
    assert report["checks"]["matched_workload"] is True
    assert report["rankings"]["full_seconds"][0]["name"] == "fast"
    assert report["winners"]["full_seconds"] == fast.configuration_digest()

    mismatched: Any = _report(full=2.0)
    mismatched["runs"][0]["tokens"]["output_tokens"] = 12
    report = build_comparison_report([(slow, _report()), (fast, mismatched)])
    assert report["checks"]["matched_workload"] is False
    assert report["rankings"]["full_seconds"] == []
    assert report["winners"]["full_seconds"] is None


def test_benchmark_command_writes_sanitized_report(monkeypatch, capsys, tmp_path: Path) -> None:
    from pllm.cli import main
    from pllm.runtime import benchmark_cli

    captured: dict[str, object] = {}

    def run(**kwargs):
        captured.update(kwargs)
        return _report()

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    output = tmp_path / "benchmark.json"
    main(
        [
            "benchmark",
            "run",
            "--prompt",
            "private prompt",
            "--output",
            str(output),
            "--format",
            "json",
        ]
    )

    result = json.loads(capsys.readouterr().out)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert captured["prompt"] == "private prompt"
    assert captured["show_dashboard"] is False
    assert callable(captured["progress"])
    assert result["command"] == "benchmark.run"
    assert report["checks"]["passed"] is True
    assert "private prompt" not in output.read_text(encoding="utf-8")


def test_benchmark_command_compares_experiments_and_saves_lowest_latency(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    from pllm.cli import main
    from pllm.runtime import benchmark_cli

    slow, fast = _experiment("slow", 1), _experiment("fast", 4)
    slow_path, fast_path = tmp_path / "slow.json", tmp_path / "fast.json"
    slow_path.write_bytes(slow.canonical_bytes())
    fast_path.write_bytes(fast.canonical_bytes())

    def run(**kwargs):
        threads = kwargs["experiment"].pipeline.components["kernels"].params["threads"]
        return _report(full=5.0 if threads == 1 else 3.0)

    monkeypatch.setattr(benchmark_cli, "run_loopback_benchmark", run)
    output, best = tmp_path / "comparison.json", tmp_path / "best.json"
    main(
        [
            "benchmark",
            "run",
            "--experiment",
            str(slow_path),
            "--experiment",
            str(fast_path),
            "--output",
            str(output),
            "--save-best",
            str(best),
            "--format",
            "json",
        ]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    saved = Experiment.from_spec(json.loads(best.read_text(encoding="utf-8")))
    assert report["rankings"]["full_seconds"][0]["name"] == "fast"
    assert saved.configuration_digest() == fast.configuration_digest()
    assert "slow" not in capsys.readouterr().err
