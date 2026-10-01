"""Ordinary benchmark sampling controls: SDK omission, validation and cohort isolation."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.benchmark_cli import _run_once, build_comparison_report, build_loopback_report
from pllm.runtime.client import _sampling_temperature
from pllm.runtime.dashboard import (
    DashboardConfig,
    DashboardRuntime,
    OTelStore,
    _sampling_choice,
    create_dashboard_app,
)


INVALID = [True, False, -0.1, 2.1, float("nan"), float("inf"), "0", 10**1000]


@pytest.mark.parametrize("temperature", INVALID)
def test_temperature_validation_before_benchmark_startup(temperature):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    with pytest.raises(ValueError, match="temperature must be a finite number"):
        DashboardConfig("127.0.0.1", 8791, None, "model", 8, temperature=temperature)
    with pytest.raises(ValueError, match="temperature must be a finite number"):
        run_loopback_benchmark(
            model="unused",
            model_id=None,
            tiny=True,
            prompt="p",
            max_output_tokens=1,
            warmups=0,
            repetitions=1,
            timeout_seconds=1,
            temperature=temperature,
        )


@pytest.mark.parametrize("temperature", [None, 0.0, 0.8, 2.0])
@pytest.mark.parametrize("terminal", ["completed", "incomplete"])
@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_dashboard_forwards_only_explicit_temperature_and_reports_terminal_digest(
    temperature,
    terminal,
    capture_output_digest,
):
    captured = {}
    text = "public fixture"

    def create(**kwargs):
        captured.update(kwargs)
        return [
            {
                "type": f"response.{terminal}",
                "response": {
                    "status": terminal,
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": text}]}
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
            }
        ]

    runtime = DashboardRuntime(
        DashboardConfig(
            "127.0.0.1",
            8791,
            None,
            "model",
            8,
            temperature=temperature,
            capture_output_digest=capture_output_digest,
        ),
        OTelStore(),
    )
    runtime._client = SimpleNamespace(
        responses=SimpleNamespace(create=create),
        privacy_audit=None,
        prepared_inventory_status=lambda _model: {},
    )
    runtime._run_chat("p", 8)
    assert captured["stream"] is True
    if temperature is None:
        assert "temperature" not in captured
    else:
        assert captured["temperature"] == temperature
    assert runtime._state["sampling"]["effective_temperature"] == _sampling_temperature(captured)
    expected = {"response_status": terminal}
    if capture_output_digest:
        expected["output_text_digest"] = hashlib.sha256(
            json.dumps(text, sort_keys=True).encode()
        ).hexdigest()
    assert runtime._state["generation"] == expected
    assert runtime._state["capture_output_digest"] is capture_output_digest
    assert "capture_output_digest" not in captured
    assert text not in json.dumps(runtime._state["generation"])


def test_dashboard_run_request_accepts_sampling_and_rejects_invalid_values(monkeypatch):
    async def noop(_runtime):
        pass

    monkeypatch.setattr(DashboardRuntime, "start", noop)
    monkeypatch.setattr(DashboardRuntime, "stop", noop)
    app = create_dashboard_app(DashboardConfig("127.0.0.1", 8791, None, "model", 8, ":memory:"))
    captured = []

    def begin(*args, **kwargs):
        captured.append(kwargs)
        return "sampling-test"

    monkeypatch.setattr(app.state.dashboard_runtime, "begin", begin)
    with TestClient(app, base_url="http://127.0.0.1:8791") as client:
        for temperature in INVALID:
            response = client.post(
                "/api/run",
                content=json.dumps({"prompt": "p", "temperature": temperature}),
                headers={"content-type": "application/json"},
            )
            assert response.status_code == 400
        for temperature in (None, 0, 2):
            assert (
                client.post(
                    "/api/run", json={"prompt": "p", "temperature": temperature}
                ).status_code
                == 202
            )
    assert captured == [{}, {"temperature": 0.0}, {"temperature": 2.0}]


def report(temperature, *, sequence=False):
    return build_loopback_report(
        model_id="model",
        tiny=True,
        max_output_tokens=2,
        warmup_runs=[],
        runs=[
            {
                "status": "completed",
                "model_fingerprint": "a" * 64,
                "max_output_tokens": 2,
                "warm": False,
                "tokens": {"input_tokens": 5, "output_tokens": 2, "authoritative": True},
                "durations": {"full_seconds": 1},
                "privacy": {
                    "plaintext_prompt_bytes_sent": 0,
                    "plaintext_token_ids_sent": 0,
                    "preparation_requests_during_online": 0,
                },
                "sampling": _sampling_choice(temperature),
            }
        ],
        temperature=temperature,
        prompt_digest="b" * 64,
        prompt_sequence_digest="c" * 64 if sequence else None,
    )


def experiments():
    return [
        Experiment(
            name=name,
            pipeline=MaskedLinearCpu(Model.hf("Qwen/Qwen2.5-0.5B-Instruct")),
            deployment=Deployment.local(root=f"local://{name}"),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=2),
        )
        for name in ("first", "second")
    ]


@pytest.mark.parametrize("sequence", [False, True])
def test_comparison_cannot_rank_default_08_with_greedy_0(sequence):
    first, second = experiments()
    default, greedy = report(None, sequence=sequence), report(0.0, sequence=sequence)
    assert default["configuration"]["sampling"]["effective_temperature"] == 0.8
    compared = build_comparison_report([(first, default), (second, greedy)])
    assert compared["checks"]["matched_workload"] is False
    assert compared["comparison_key"] is None
    assert all(not ranks for ranks in compared["rankings"].values())
    same = build_comparison_report([(first, default), (second, report(0.8, sequence=sequence))])
    assert same["checks"]["matched_workload"] is True
    assert same["comparison_key"]["effective_sampling"]["temperature"] == 0.8
    assert len(same["rankings"]["full_seconds"]) == 2
    del greedy["configuration"]["sampling"]
    assert (
        build_comparison_report([(first, greedy), (second, greedy)])["checks"]["matched_workload"]
        is False
    )


@pytest.mark.parametrize("temperature", [None, 0.0])
@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_benchmark_post_omits_default_and_keeps_only_text_free_sampling_metadata(
    temperature,
    monkeypatch,
    capture_output_digest,
):
    monkeypatch.setattr("pllm.runtime.benchmark_cli.secrets.token_hex", lambda _n: "fixed")
    captured = {}
    sampling = _sampling_choice(temperature)
    generation = {"response_status": "completed", "output_text_digest": "a" * 64}

    def handler(request):
        if request.url.path == "/api/run":
            captured.update(json.loads(request.content))
            return httpx.Response(202, json={})
        if request.url.path == "/api/snapshot":
            return httpx.Response(
                200,
                json={
                    "run": {
                        "run_id": "bench-fixed",
                        "phase": "ready",
                        "active_run": None,
                        "text": "private text must not be copied",
                        "sampling": sampling,
                        "generation": generation,
                    }
                },
            )
        return httpx.Response(200, json={"status": "completed"})

    with httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://benchmark"
    ) as client:
        result = _run_once(
            client,
            SimpleNamespace(poll=lambda: None),
            prompt="p",
            max_output_tokens=2,
            timeout_seconds=1,
            temperature=temperature,
            capture_output_digest=capture_output_digest,
        )
    assert ("temperature" in captured) == (temperature is not None)
    if temperature is not None:
        assert captured["temperature"] == 0.0
    expected_generation = generation if capture_output_digest else {"response_status": "completed"}
    assert result == {
        "status": "completed",
        "sampling": sampling,
        "generation": expected_generation,
    }


@pytest.mark.parametrize("effective", INVALID)
def test_invalid_report_sampling_cannot_enter_ranked_cohort(effective):
    first, second = experiments()
    invalid = report(0.0)
    invalid["configuration"]["sampling"]["effective_temperature"] = effective
    compared = build_comparison_report([(first, invalid), (second, report(0.0))])
    assert compared["checks"]["matched_workload"] is False
    assert all(not ranks for ranks in compared["rankings"].values())


def test_run_sampling_disagreement_disqualifies_report_and_comparison():
    first, second = experiments()
    altered = report(0.0)
    altered["runs"][0]["sampling"] = _sampling_choice(None)
    compared = build_comparison_report([(first, altered), (second, report(0.0))])
    assert compared["checks"]["matched_workload"] is False
    rebuilt = build_loopback_report(
        model_id="model",
        tiny=True,
        max_output_tokens=2,
        warmup_runs=[],
        runs=altered["runs"],
        temperature=0.0,
    )
    assert rebuilt["checks"]["sampling_matches_request"] is False
    assert rebuilt["checks"]["passed"] is False


@pytest.mark.parametrize("invalid", [None, 0, 1, 0.0, "true", [], {}])
def test_digest_capture_requires_exact_bool_before_startup(invalid):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    with pytest.raises(ValueError, match="capture_output_digest must be Boolean"):
        DashboardConfig("127.0.0.1", 8791, None, "model", 8, capture_output_digest=invalid)
    with pytest.raises(ValueError, match="capture_output_digest must be Boolean"):
        run_loopback_benchmark(
            model="unused",
            model_id=None,
            tiny=True,
            prompt="p",
            max_output_tokens=1,
            warmups=0,
            repetitions=1,
            timeout_seconds=1,
            capture_output_digest=invalid,
        )
    with pytest.raises(ValueError, match="capture_output_digest must be Boolean"):
        build_loopback_report(
            model_id="model",
            tiny=True,
            max_output_tokens=1,
            warmup_runs=[],
            runs=[],
            capture_output_digest=invalid,
        )


@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_benchmark_api_propagates_capture_policy_to_dashboard_config(
    capture_output_digest,
    monkeypatch,
):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    captured = {}

    class StopBeforeStartup(Exception):
        pass

    def config(**kwargs):
        captured.update(kwargs)
        raise StopBeforeStartup

    monkeypatch.setattr("pllm.runtime.dashboard.DashboardConfig", config)
    with pytest.raises(StopBeforeStartup):
        run_loopback_benchmark(
            model="unused",
            model_id=None,
            tiny=True,
            prompt="p",
            max_output_tokens=1,
            warmups=0,
            repetitions=1,
            timeout_seconds=1,
            capture_output_digest=capture_output_digest,
        )
    assert captured["capture_output_digest"] is capture_output_digest
    assert captured["temperature"] is None


def test_default_dashboard_does_not_read_output_for_fingerprinting():
    class PrivateResponse:
        status = "completed"
        usage = SimpleNamespace(input_tokens=3, output_tokens=2)

        @property
        def output_text(self):
            raise AssertionError("ordinary run must not read output for fingerprinting")

    runtime = DashboardRuntime(DashboardConfig("127.0.0.1", 8791, None, "model", 8), OTelStore())
    runtime._client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_kwargs: [
                SimpleNamespace(type="response.completed", response=PrivateResponse())
            ]
        ),
        privacy_audit=None,
        prepared_inventory_status=lambda _model: {},
    )
    runtime._run_chat("p", 8)
    assert runtime._state["generation"] == {"response_status": "completed"}


@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_report_capture_policy_sanitizes_generation_without_mutating_archive_record(
    capture_output_digest,
):
    archived = report(None)["runs"][0]
    archived["generation"] = {
        "response_status": "completed",
        "output_text_digest": "a" * 64,
        "text": "private text",
        "token_ids": [1, 2],
    }
    original = json.dumps(archived, sort_keys=True)
    result = build_loopback_report(
        model_id="model",
        tiny=True,
        max_output_tokens=2,
        warmup_runs=[archived],
        runs=[archived],
        capture_output_digest=capture_output_digest,
    )
    assert result["configuration"]["capture_output_digest"] is capture_output_digest
    for row in result["runs"] + result["warmup_runs"]:
        assert row["generation"]["response_status"] == "completed"
        assert ("output_text_digest" in row["generation"]) is capture_output_digest
        assert "text" not in row["generation"]
        assert "token_ids" not in row["generation"]
    assert json.dumps(archived, sort_keys=True) == original


@pytest.mark.parametrize("capture_output_digest", [False, True])
def test_completed_run_archive_contains_neither_text_nor_output_fingerprint(
    capture_output_digest,
    monkeypatch,
):
    from pllm.runtime.benchmark_history import BenchmarkHistory

    text = "private fixture output"
    fingerprint = hashlib.sha256(json.dumps(text, sort_keys=True).encode()).hexdigest()
    history = BenchmarkHistory(":memory:")
    runtime = DashboardRuntime(
        DashboardConfig(
            "127.0.0.1", 8791, None, "model", 8, capture_output_digest=capture_output_digest
        ),
        OTelStore(),
        history,
    )
    runtime._client = SimpleNamespace(
        privacy_audit=None,
        responses=SimpleNamespace(
            create=lambda **_kwargs: [
                {
                    "type": "response.completed",
                    "response": {
                        "status": "completed",
                        "usage": {"input_tokens": 3, "output_tokens": 2},
                        "output": [
                            {"type": "message", "content": [{"type": "output_text", "text": text}]}
                        ],
                    },
                }
            ]
        ),
    )
    runtime._topology = SimpleNamespace(
        started=True,
        closed=False,
        requires_preparation=False,
        statuses=(SimpleNamespace(role="inference", running=True),),
    )
    runtime._state["phase"] = "ready"
    runtime._model_fingerprint = "b" * 64
    monkeypatch.setattr("pllm.runtime.dashboard.time.sleep", lambda _seconds: None)
    run_id = runtime.begin("private prompt", 8)
    worker = runtime._worker
    if worker is not None:
        worker.join(3)
        assert not worker.is_alive()
    stored = history.get(run_id)
    assert stored is not None and stored["status"] == "completed"
    serialized = json.dumps(stored)
    assert text not in serialized
    assert fingerprint not in serialized
    assert "output_text_digest" not in serialized
    assert "generation" not in stored
    history.close()
