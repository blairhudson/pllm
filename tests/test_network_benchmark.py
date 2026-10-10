"""Slice C exercises ordinary report/cohort helpers over live admitted HTTP roles."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from pllm._cli.app import build_parser
from pllm._cli.errors import UsageError
from pllm.runtime.network_benchmark import run_network_benchmark

pytestmark = pytest.mark.rust

_spec = importlib.util.spec_from_file_location("network_scenarios_fixture",
    Path(__file__).resolve().parents[1] / "examples/benchmarks/network_scenarios.py")
_fixture_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture_module)
scenario_fixture = _fixture_module.scenario_fixture
free_port = _fixture_module.free_port
running_http = _fixture_module.running_http


@pytest.fixture
def network_fixture(tmp_path):
    with scenario_fixture(tmp_path, hidden_size=8) as fixture:
        yield fixture


def test_live_benchmark_matched_controls_new_attempts_and_text_free_cost_scopes(network_fixture, monkeypatch, tmp_path):
    fixture = network_fixture
    report = run_network_benchmark(
        network=fixture["network"], request=fixture["request"], credentials=fixture["credentials"],
        prompt="Hi", max_output_tokens=2, warmups=1, repetitions=3, temperature=0,
        compare_feasible=True, capture_output_digest=True,
    )
    assert report["checks"]["matched_workload"]
    assert len(report["candidates"]) == 4
    assert report["summary"]["selected_plan_latency_regret_seconds"] is not None
    digests = set()
    bodies = set()
    for candidate in report["candidates"]:
        inner = candidate["report"]
        assert inner["checks"]["passed"]
        assert inner["configuration"]["sampling"]["effective_temperature"] == 0
        for record in inner["runs"]:
            digests.add(record["generation"]["output_text_digest"])
            bodies.add(record["model_fingerprint"])
            assert record["tokens"] == {"authoritative": True, "input_tokens": 22, "output_tokens": 2, "total_tokens": 24}
            assert record["network_audit"]["lease_closed"]
            assert record["network_audit"]["full_wire_bytes"] is None
            assert record["network_audit"]["full_response_cpu_seconds"] is None
            assert record["network_audit"]["controller_request_count"] is None
            assert record["warm"] == (record["privacy"]["bundle_cache_hits"] > 0)
        assert inner["summary"]["median_covered_all_link_serialized_body_bytes"] > 0
    assert len(digests) == len(bodies) == 1
    # Four controls, four attempts each, two controllers per attempt, fresh native session.
    for verb in ("reserve", "arm", "release"):
        assert sum(proxy.snapshot()[verb]["requests"] for proxy in fixture["proxies"]) == 32
    assert sum(len(child.state.controller._used_attempts) for child in fixture["children"]) == 32
    assert all(not child.state.controller._reservations for child in fixture["children"])
    serialized = json.dumps(report)
    for credential in fixture["credentials"].values():
        assert credential not in serialized
    assert '"output_text"' not in serialized and '"prompt"' not in serialized
    assert '"token_ids"' not in serialized and '"seed"' not in serialized
    from dataclasses import replace
    import time
    from pllm.compiler import plan
    from pllm.deployment import discover
    from pllm.search import PlanningPrior

    prior = PlanningPrior.from_network_benchmark(report)
    assert len(prior.rows) == 4 and all(row.samples == 3 for row in prior.rows)
    assert prior.input_tokens == 22 and prior.output_tokens == 2
    assert prior.warm == report["comparison_key"]["warm"]
    assert PlanningPrior.from_spec(prior.to_spec()) == prior
    fresh = discover(fixture["network"], credentials=fixture["credentials"])
    req = replace(fixture["request"], performance_prior=prior,
        policy=replace(fixture["request"].policy, evaluated_at_ms=time.time_ns() // 1_000_000))
    decision = plan(req, snapshot=fresh)
    assert decision.costs["selection_basis"] == "matched_performance_prior"
    best = min(prior.rows, key=lambda row: (row.ttft_ms, row.decode_ms_per_token))
    assert decision.costs["forecast_ttft_ms"] == best.ttft_ms
    # Compiling and replaying evidence never reserves providers.
    assert sum(len(child.state.controller._used_attempts) for child in fixture["children"]) == 32
    import copy
    from pllm import OpenAI
    from pllm.deployment import NetworkError
    from pllm.deployment.admission import ExecutionContract
    from pllm.settings import ClientSettings

    bad = copy.deepcopy(report)
    bad["candidates"][0]["report"]["runs"][0]["generation"]["output_text_digest"] = "a" * 64
    with pytest.raises(NetworkError, match="cohorts differ"):
        PlanningPrior.from_network_benchmark(bad)
    del bad["candidates"][0]["report"]["runs"][0]["network_audit"]["performance_binding"]
    with pytest.raises(NetworkError, match="lacks performance context"):
        PlanningPrior.from_network_benchmark(bad)
    prior_path = tmp_path / "performance.json"
    prior_path.write_bytes(prior.canonical_bytes())
    config_path = tmp_path / "client.toml"
    experiment_path = tmp_path / "experiment.json"
    experiment_path.write_bytes(req.candidates[0].canonical_bytes())
    ClientSettings(network=str(fixture["paths"]["network"]), experiment=str(experiment_path),
                   performance_prior=str(prior_path)).save(config_path)
    monkeypatch.setenv("PLLM_CONFIG", str(config_path))
    for key, value in fixture["credentials"].items():
        monkeypatch.setenv(key, value)
    with OpenAI() as client:
        response = client.responses.create(input="Hi", max_output_tokens=2, temperature=0)
        selection = client.planning(response.id)
        assert selection.costs["selection_basis"] == "matched_performance_prior"
        assert selection.costs["forecast_ttft_ms"] == best.ttft_ms
        contract = ExecutionContract.from_result(selection, selection.snapshot).canonical_bytes()
        assert b"performance_prior" not in contract and b"forecast_ttft" not in contract
    assert all(not child.state.controller._reservations for child in fixture["children"])


def test_fixed_plan_default_sampling_and_opt_in_digest(network_fixture):
    fixture = network_fixture
    report = run_network_benchmark(network=fixture["network"], result=fixture["result"],
        credentials=fixture["credentials"], prompt="Hi", max_output_tokens=2)
    assert report["configuration"]["sampling"]["effective_temperature"] == 0.8
    assert "output_text_digest" not in report["runs"][0]["generation"]


def test_dry_run_cli_files_and_conflicts_never_admit(network_fixture):
    fixture = network_fixture
    paths = fixture["paths"]
    for family in (("gateway",), ("benchmark", "run")):
        for mode in ("plan", "request"):
            extra = ("--max-output-tokens", "2", "--temperature", "0") if family[0] == "benchmark" else ()
            command = [sys.executable, "-m", "pllm", "--format", "json", "--dry-run", *family,
                       "--" + mode, str(paths[mode]), "--network", str(paths["network"]), *extra]
            result = subprocess.run(command, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
            assert json.loads(result.stdout)["data"]["dry_run"]
    assert all(not child.state.controller._used_attempts for child in fixture["children"])
    for override in (("--local",), ("--weight-bits", "4"), ("--inference-url", "http://127.0.0.1:1")):
        result = subprocess.run([sys.executable, "-m", "pllm", "--format", "json", "--dry-run", "gateway",
            "--plan", str(paths["plan"]), "--network", str(paths["network"]), *override], capture_output=True, text=True)
        assert result.returncode != 0
        assert "NETWORK_OVERRIDE_CONFLICT" in result.stderr


def test_parser_selection_exclusion_and_temperature():
    parser = build_parser()
    assert parser.parse_args(["benchmark", "run", "--temperature", "0"]).temperature == 0
    assert parser.parse_args(["benchmark", "run"]).temperature is None
    for command in (("gateway",), ("benchmark", "run")):
        with pytest.raises(UsageError):
            parser.parse_args([*command, "--plan", "p", "--request", "r"])
        with pytest.raises(UsageError):
            parser.parse_args([*command, "--plan", "p", "--experiment", "e"])


@pytest.mark.parametrize("profile_name", ("ClientOnlyCpu", "MaskedLinearCpu", "TwoOnlineOffsetCpu"))
def test_local_selected_backend_uses_same_driver_and_gateway(tmp_path, profile_name):
    from pllm import Deployment, ExecutionBudget, Experiment, Model
    from pllm.compiler import plan
    from pllm.deployment import NetworkSnapshot, NetworkSpec, PartyOffer
    from pllm.model_loader import resolve_model
    from pllm.modeling import lower_model
    from pllm import profiles
    from pllm.runtime.network_benchmark import SelectedClientFactory
    from pllm.runtime.sidecar import create_sidecar_app
    from pllm.search import PlanningPolicy, PlanningRequest
    from fastapi.testclient import TestClient
    import time

    checkpoint = _fixture_module.create_tiny_llama_checkpoint(tmp_path / "checkpoint", hidden_size=8,
        intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=4)
    model = Model.path(str(checkpoint), model_id="local-selected-tiny")
    model_plan = lower_model(json.loads((checkpoint / "config.json").read_bytes()), batch=1,
                              max_input_tokens=32, max_new_tokens=2)
    experiment = Experiment(profile_name, getattr(profiles, profile_name)(model),
        Deployment.local(root="local://selected-test"), ExecutionBudget(1, 32, 2))
    stamp = time.time_ns() // 1_000_000
    offers = tuple(PartyOffer(name, name, str(index + 1) * 64, stamp + 300_000,
        ("trusted_client",) if name == "client" else ("trusted_preparation", "masked_linear_provider", "public_linear_provider"),
        256 << 20, 64 << 20, 4, 0, ("*",), ("*",)) for index, name in enumerate(("client", "a", "b")))
    snapshot = NetworkSnapshot("local", offers, (), "local-fixture", stamp, stamp + 300_000)
    network = NetworkSpec("local", snapshot)
    request = PlanningRequest(model_plan, (experiment,), PlanningPolicy("client", stamp, 4, 1000,
        ("client_weight_bytes",), max_observation_age_ms=300_000), source_lock_digest=resolve_model(model).source_lock_digest)
    result = plan(request, snapshot=snapshot)
    assert result.status == "feasible"
    report = run_network_benchmark(network=network, result=result, prompt="Hi", max_output_tokens=2, temperature=0)
    assert report["checks"]["passed"] and report["summary"]["total_output_tokens"] == 2
    with TestClient(create_sidecar_app(client_factory=SelectedClientFactory(network, result=result))) as gateway:
        response = gateway.post("/v1/responses", headers={"authorization": "Bearer local"},
            json={"model": "local-selected-tiny", "input": "Hi", "max_output_tokens": 2, "temperature": 0})
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["output_tokens"] == 2
