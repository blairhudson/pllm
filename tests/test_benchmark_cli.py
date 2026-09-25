from __future__ import annotations

import json
import subprocess
import sys
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
from pllm.runtime.topology_accounting import prepared_body_accounting


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
            "schema": "pllm.experiment.v2",
            "name": name,
            "pipeline": {
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
    startup = report["topology_accounting"]["startup"]
    assert startup is not None and startup["tracked_body_counter_set_present"]
    assert startup["all_link_serialized_body_bytes"] > 0
    assert any(
        edge["source"] == "preparation" and edge["destination"] == "inference"
        and edge["serialized_body_bytes"] > 0
        for edge in startup["body_bytes_by_edge"]
    )
    topology = report["topology_accounting"]["runs"][0]
    assert topology["tracked_body_counter_set_present"] is True
    assert topology["total_wire_bytes"] is None
    assert topology["full_response_compute_cap_checked"] is False
    assert sum(edge["serialized_body_bytes"] for edge in topology["body_bytes_by_edge"]) == (
        topology["all_link_serialized_body_bytes"]
    )


@pytest.mark.integration
def test_two_worker_experiment_runs_through_standard_benchmark(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    experiment = Experiment(
        name="offset-benchmark",
        pipeline=TwoOnlineOffsetCpu(Model.path(str(checkpoint), model_id="offset-benchmark")),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )
    report = run_loopback_benchmark(
        model=str(checkpoint), model_id="offset-benchmark", tiny=False,
        prompt="A", max_output_tokens=2, warmups=0,
        repetitions=1, timeout_seconds=120, experiment=experiment,
    )
    assert report["checks"]["passed"]
    assert report["summary"]["completed_runs"] == 1
    assert set(report["runs"][0]["processes"]) == {"client", "worker_a", "worker_b"}
    assert report["runs"][0]["privacy"]["role_link.worker_a.online_upload_bytes"] > 0
    assert report["runs"][0]["privacy"]["role_link.worker_b.online_download_bytes"] > 0
    assert report["privacy_admission"]["independent_operators_verified"] is False
    accounting = report["topology_accounting"]["runs"][0]
    assert accounting["tracked_body_counter_set_present"]
    assert accounting["all_link_serialized_body_bytes"] > 0
    assert accounting["aggregate_run_window_cpu_seconds"] is not None
    assert accounting["full_response_compute_cap_checked"] is False


def test_two_worker_experiment_uses_existing_benchmark_cli(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    experiment = Experiment(
        name="offset-cli",
        pipeline=TwoOnlineOffsetCpu(Model.path(str(checkpoint), model_id="offset-cli")),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )
    target = tmp_path / "offset-experiment.json"
    target.write_bytes(experiment.canonical_bytes())
    result = subprocess.run(
        [sys.executable, "-m", "pllm", "benchmark", "run", "--experiment", str(target),
         "--prompt", "A", "--max-output-tokens", "2", "--warmups", "0",
         "--repetitions", "1", "--format", "json"],
        text=True, capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["command"] == "benchmark.run"
    report = output["data"]["report"]
    assert report["checks"]["passed"]
    assert report["configuration"]["roles"] == ["client", "worker_a", "worker_b"]
    assert report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"] > 0
    assert "user: A" not in result.stdout
    assert report["runs"][0]["privacy"]["plaintext_token_ids_sent"] == 0
    assert all("token_ids" not in run for run in report["runs"])


@pytest.mark.integration
def test_client_offset_prepared_topologies_share_one_w8a8_benchmark_cohort(
    tmp_path: Path,
) -> None:
    from pllm import Deployment, ExecutionBudget, Model
    from pllm.profiles import ClientOnlyCpu, MaskedLinearCpu, TwoOnlineOffsetCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint

    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    model_id = "matched-three-topologies"
    source = Model.path(str(checkpoint), model_id=model_id)
    experiments = [
        Experiment(
            name=name,
            pipeline=kind(source, quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)),
            deployment=Deployment.local(root=str(tmp_path)),
            budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
        )
        for name, kind in (
            ("client-owned", ClientOnlyCpu), ("two-online-offset", TwoOnlineOffsetCpu),
            ("prepared", MaskedLinearCpu),
        )
    ]
    runs = [
        (
            experiment,
            run_loopback_benchmark(
                model=str(checkpoint), model_id=model_id, tiny=False,
                prompt="A", max_output_tokens=2, warmups=0,
                repetitions=1, timeout_seconds=120, experiment=experiment,
            ),
        )
        for experiment in experiments
    ]
    assert all(report["checks"]["passed"] for _, report in runs)
    assert len({report["runs"][0]["model_fingerprint"] for _, report in runs}) == 1
    assert len({(
        report["runs"][0]["tokens"]["input_tokens"],
        report["runs"][0]["tokens"]["output_tokens"],
    ) for _, report in runs}) == 1
    comparison = build_comparison_report(runs)
    assert comparison["checks"]["matched_workload"]
    assert comparison["comparison_key"]["model_fingerprint"]
    assert runs[0][1]["topology_accounting"]["runs"][0]["online_client_serialized_body_bytes"] == 0
    ownership = runs[0][1]["topology_accounting"]["startup"]
    assert ownership["schema"] == "pllm.topology_model_ownership.v1"
    assert ownership["checkpoint_artifact_bytes"] > 0
    assert ownership["loaded_weight_and_local_tensor_bytes"] > 0
    assert ownership["cold_checkpoint_transfer_bytes"] is None
    assert ownership["client_peak_memory_bytes"] is None
    assert runs[1][1]["privacy_admission"]["independent_operators_verified"] is False
    assert runs[1][1]["topology_accounting"]["runs"][0]["online_client_serialized_body_bytes"] > 0
    assert runs[2][1]["topology_accounting"]["startup"]["all_link_serialized_body_bytes"] > 0


def test_prepared_link_ledger_charges_serialized_body_once_and_never_invents_wire() -> None:
    record = _record("accounted", ttft=1.0, throughput=1.0)
    privacy = record["privacy"]
    assert isinstance(privacy, dict)
    record["privacy"] = {
        **privacy,
        "session_authorization_upload_bytes": 7,
        "preparation_upload_bytes": 11,
        "session_authorization_download_bytes": 5,
        "preparation_download_bytes": 13,
        "correction_push_bytes": 17,
        "bundle_network_bytes": 19,
        "inference_upload_bytes": 23,
        "inference_download_bytes": 29,
        "masked_online_upload_bytes": 1_000,  # duplicate view, never charged twice
    }
    record["processes"] = {
        "client": {"cpu_seconds": 1.0},
        "preparation": {"cpu_seconds": 2.0},
        "inference": {"cpu_seconds": 3.0},
    }
    accounted = prepared_body_accounting(record)
    assert accounted["all_link_serialized_body_bytes"] == 124
    assert accounted["client_serialized_body_bytes"] == 107
    assert accounted["online_client_serialized_body_bytes"] == 52
    assert accounted["online_all_link_serialized_body_bytes"] == 52
    assert accounted["aggregate_run_window_cpu_seconds"] == 6.0
    assert accounted["body_bytes_by_edge"] == [
        {"source": "client", "destination": "preparation", "phase": "offline", "serialized_body_bytes": 18},
        {"source": "preparation", "destination": "client", "phase": "offline", "serialized_body_bytes": 18},
        {"source": "preparation", "destination": "inference", "phase": "offline", "serialized_body_bytes": 17},
        {"source": "inference", "destination": "client", "phase": "cold", "serialized_body_bytes": 19},
        {"source": "client", "destination": "inference", "phase": "online", "serialized_body_bytes": 23},
        {"source": "inference", "destination": "client", "phase": "online", "serialized_body_bytes": 29},
    ]
    assert accounted["total_wire_bytes"] is None
    assert accounted["full_response_compute_cap_checked"] is False


def test_link_ledger_marks_partial_counters_and_missing_cpu_unavailable() -> None:
    partial = prepared_body_accounting(_record("partial", ttft=1.0, throughput=1.0))
    assert partial["tracked_body_counter_set_present"] is False
    assert partial["all_link_serialized_body_bytes"] is None
    assert partial["body_bytes_by_edge"] is None
    assert partial["aggregate_run_window_cpu_seconds"] is None
    invalid = _record("invalid", ttft=1.0, throughput=1.0)
    counters = {key: 0 for key in (
        "session_authorization_upload_bytes", "preparation_upload_bytes",
        "session_authorization_download_bytes", "preparation_download_bytes",
        "correction_push_bytes", "bundle_network_bytes", "inference_upload_bytes",
        "inference_download_bytes",
    )}
    counters["correction_push_bytes"] = True
    invalid["privacy"] = counters
    assert prepared_body_accounting(invalid)["tracked_body_counter_set_present"] is False


class _RunningProcess:
    def poll(self) -> None:
        return None


@pytest.mark.he
@pytest.mark.integration
def test_tiny_guarded_benchmark_runs_over_one_role_profile() -> None:
    import pllm
    from pllm.profiles import ProprietaryGuarded
    from pllm.sources import TinyModel

    experiment = pllm.Experiment(
        name="guarded-benchmark",
        pipeline=ProprietaryGuarded(TinyModel(model_id="guarded-benchmark")),
        deployment=pllm.Deployment.local(root="local://guarded-benchmark"),
        budget=pllm.ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=1),
    )
    report = run_loopback_benchmark(
        model="unused",
        model_id=None,
        tiny=False,
        prompt="private",
        max_output_tokens=1,
        warmups=0,
        repetitions=1,
        timeout_seconds=120,
        experiment=experiment,
    )
    assert report["checks"]["passed"] is True
    assert report["configuration"]["tiny"] is True
    assert report["configuration"]["roles"] == ["client", "inference"]
    assert set(report["runs"][0]["processes"]) == {"client", "inference"}


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
