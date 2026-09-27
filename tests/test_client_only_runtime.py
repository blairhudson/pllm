"""Client-owned execution through the ordinary compiled Responses path."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import ClientOnlyCpu, ClientOnlyMetal, MaskedLinearCpu
from pllm.runtime.benchmark_cli import build_comparison_report, run_loopback_benchmark
from pllm.runtime.client import OpenAI, ProtocolError
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.masked_runtime import ModelError
from pllm.runtime.servers import TopologyError, build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _require_metal() -> None:
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or importlib.util.find_spec("mlx") is None
    ):
        pytest.skip("Apple Silicon and pllm.run[metal] are required")


def _setup(tmp_path: Path) -> tuple[Experiment, MaskedTransformerEngine]:
    checkpoint = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    model_id = "client-owned-qwen"
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(load_hf_directory(checkpoint, model_id=model_id)))
    experiment = Experiment(
        name="client-only-check",
        pipeline=ClientOnlyCpu(Model.path(str(checkpoint), model_id=model_id)),
        deployment=Deployment.local(root="local://client-owned-check"),
        budget=ExecutionBudget(requests=3, max_input_tokens=64, max_new_tokens=2),
    )
    return experiment, engine


def test_client_only_responses_have_no_provider_channel_or_inventory(tmp_path: Path) -> None:
    experiment, engine = _setup(tmp_path)
    with OpenAI(experiment=experiment, local_engine=engine) as client:
        assert client.models.list()["data"][0]["owned_by"] == "client"
        response = client.responses.create(
            model=experiment.resolve().model, input="A", max_output_tokens=2,
        )
        assert response.model == experiment.resolve().model
        assert response.usage.input_tokens > 0
        assert response.status in {"completed", "incomplete"}
        assert client.privacy_audit.inference_stage_calls == 0
        assert client.privacy_audit.inference_upload_bytes == 0
        assert client.privacy_audit.preparation_upload_bytes == 0
        with pytest.raises(ProtocolError, match="provider endpoint"):
            client.runtime.capabilities()
        with pytest.raises(ModelError, match="preparation inventory"):
            client.preprocess()


def test_client_metal_executes_prefill_on_gpu_and_decode_on_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_metal()
    from pllm.kernels import AppleMetal
    from pllm.runtime.metal import MetalCompiledMatrix

    cpu, engine = _setup(tmp_path)
    metal = Experiment(
        name="client-metal-check",
        pipeline=ClientOnlyMetal(cpu.pipeline.model, kernels=AppleMetal(min_rows=2)),
        deployment=cpu.deployment,
        budget=cpu.budget,
    )
    assert metal.resolve().composition_digest != cpu.resolve().composition_digest
    calls: list[int] = []
    original_clear = MetalCompiledMatrix.clear

    def counted_clear(self, inputs):
        calls.append(inputs.shape[0])
        return original_clear(self, inputs)

    monkeypatch.setattr(MetalCompiledMatrix, "clear", counted_clear)
    with OpenAI(experiment=cpu, local_engine=engine) as baseline:
        expected = baseline.responses.create(input="A", max_output_tokens=2, temperature=0)
    with OpenAI(experiment=metal, local_engine=engine) as client:
        actual = client.responses.create(input="A", max_output_tokens=2, temperature=0)
        assert client.privacy_audit.inference_stage_calls == 0
    assert actual.output_text == expected.output_text
    assert actual.usage.output_tokens == expected.usage.output_tokens
    assert calls and all(rows >= 2 for rows in calls)
    with build_roles(metal) as topology, TestClient(
        topology.gateway_app(local_api_key="metal-test")
    ) as gateway:
        response = gateway.post(
            "/v1/responses",
            headers={"Authorization": "Bearer metal-test"},
            json={"model": metal.resolve().model, "input": "A", "max_output_tokens": 2},
        )
        assert response.status_code == 200, response.text
        assert topology.statuses == ()
    report = run_loopback_benchmark(
        model=metal.pipeline.model.source,
        model_id=metal.resolve().model,
        tiny=False,
        prompt="A",
        max_output_tokens=2,
        warmups=0,
        repetitions=1,
        timeout_seconds=120.0,
        experiment=metal,
    )
    assert report["checks"]["passed"], report["checks"]
    assert report["configuration"]["roles"] == ["client"]
    mixed = build_comparison_report([(cpu, report), (metal, report)])
    assert mixed["checks"]["matched_workload"]
    assert not mixed["checks"]["matched_kernel_backend"]
    assert all(winner is None for winner in mixed["winners"].values())
    assert mixed["compute_cap_diagnostic"] is None


def test_metal_kernel_requires_explicit_component_selection() -> None:
    from pllm.kernels import AppleMetal, Cpu
    from pllm.sources import TinyModel

    source = TinyModel(model_id="metal-provider-rejection")
    with pytest.raises(ValueError, match="pllm/apple-metal-int8/v1"):
        ClientOnlyMetal(source, kernels=Cpu(threads=2))
    experiment = Experiment(
        name="metal-provider-rejection",
        pipeline=MaskedLinearCpu(source).with_params(kernels=AppleMetal(min_rows=8)),
        deployment=Deployment.local(root="local://metal-provider-rejection"),
        budget=ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=1),
    )
    assert experiment.resolve().composition_digest == experiment.pipeline.digest()
    with pytest.raises(ValueError, match="min_rows"):
        AppleMetal(min_rows=1)


def test_client_only_rejects_endpoint_overrides_and_wrong_weight_ownership(tmp_path: Path) -> None:
    experiment, engine = _setup(tmp_path)
    with pytest.raises(ValueError, match="only client-owned"):
        OpenAI(experiment=experiment, local_engine=engine, base_url="https://provider.invalid")
    with pytest.raises(ValueError, match="only client-owned"):
        OpenAI(experiment=experiment, local_engine=engine, http_client=object())
    with pytest.raises(ValueError, match="only client-owned"):
        OpenAI(experiment=experiment, local_engine=engine, preparation_api_key="outside")
    with pytest.raises(ValueError, match="only client-owned"):
        OpenAI(experiment=experiment, local_engine=engine, session_transport="websocket")
    with OpenAI(experiment=experiment) as client:
        assert client.responses.create(input="A", max_output_tokens=2).usage.input_tokens > 0
        assert client._owned_topology.statuses == ()


def test_client_only_uses_existing_role_builder_without_provider_children(tmp_path: Path) -> None:
    experiment, _ = _setup(tmp_path)
    with build_roles(experiment) as topology:
        assert topology.is_healthy()
        assert topology.statuses == ()
        with pytest.raises(TopologyError, match="no inference provider"):
            _ = topology.inference_url
        with topology.client() as client:
            response = client.responses.create(input="A", max_output_tokens=2)
            assert response.usage.input_tokens > 0
            assert client.privacy_audit.inference_stage_calls == 0
    assert not topology.is_healthy()


def test_closed_client_owned_role_cannot_reuse_compiled_weights(tmp_path: Path) -> None:
    experiment, _ = _setup(tmp_path)
    topology = build_roles(experiment).start()
    client = topology.client()
    topology.close()
    try:
        with pytest.raises(ModelError, match="no longer loaded"):
            client.responses.create(input="A", max_output_tokens=2)
    finally:
        client.close()


def test_client_only_gateway_uses_the_same_compiled_response_path(tmp_path: Path) -> None:
    experiment, _ = _setup(tmp_path)
    with build_roles(experiment) as topology, TestClient(
        topology.gateway_app(local_api_key="gateway-test")
    ) as gateway:
        response = gateway.post(
            "/v1/responses",
            headers={"Authorization": "Bearer gateway-test"},
            json={"model": experiment.resolve().model, "input": "A", "max_output_tokens": 2},
        )
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["input_tokens"] > 0
        assert topology.statuses == ()


def test_client_only_uses_existing_benchmark_run_driver(tmp_path: Path) -> None:
    experiment, _ = _setup(tmp_path)
    report = run_loopback_benchmark(
        model=experiment.pipeline.model.source,
        model_id=experiment.resolve().model,
        tiny=False,
        prompt="A",
        max_output_tokens=2,
        warmups=0,
        repetitions=1,
        timeout_seconds=120.0,
        experiment=experiment,
    )
    assert report["checks"]["passed"], report["checks"]
    assert report["configuration"]["roles"] == ["client"]
    assert report["runs"][0]["tokens"]["authoritative"]
    assert report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"] == 0
    assert report["checks"]["no_provider_stage_traffic"]


def test_client_only_runs_through_existing_benchmark_cli(tmp_path: Path) -> None:
    experiment, _ = _setup(tmp_path)
    target = tmp_path / "experiment.json"
    target.write_bytes(experiment.canonical_bytes())
    completed = subprocess.run(
        [sys.executable, "-m", "pllm", "benchmark", "run", "--experiment", str(target),
         "--format", "json", "--prompt", "A", "--max-output-tokens", "2",
         "--repetitions", "1"],
        text=True, capture_output=True, timeout=120, check=True,
    )
    report = json.loads(completed.stdout)["data"]["report"]
    assert report["checks"]["passed"]
    assert report["configuration"]["roles"] == ["client"]
    assert report["topology_accounting"]["runs"][0]["all_link_serialized_body_bytes"] == 0
    assert "client-owned-qwen" in completed.stdout
    assert '"input": "A"' not in completed.stdout

    gateway = subprocess.run(
        [sys.executable, "-m", "pllm", "gateway", "--local", "--experiment",
         str(target), "--dry-run", "--format", "json"],
        text=True, capture_output=True, timeout=60, check=True,
    )
    assert json.loads(gateway.stdout)["data"]["dry_run"] is True


def test_prepared_verified_and_client_owned_match_in_existing_experiment_comparison(
    tmp_path: Path,
) -> None:
    local, _ = _setup(tmp_path)
    prepared = Experiment(
        name="prepared-same-checkpoint",
        pipeline=MaskedLinearCpu(local.pipeline.model),
        deployment=Deployment.local(root=str(tmp_path / "prepared")),
        budget=local.budget,
    )
    from pllm.profiles import VerifiedMaskedLinearCpu
    from pllm.roles import PreparedProviderRoles

    verified = Experiment(
        name="verified-same-checkpoint",
        pipeline=VerifiedMaskedLinearCpu(
            local.pipeline.model, topology=PreparedProviderRoles(),
        ),
        deployment=Deployment.local(root=str(tmp_path / "verified")),
        budget=local.budget,
    )
    candidates = []
    for experiment in (local, prepared, verified):
        report = run_loopback_benchmark(
            model=experiment.pipeline.model.source,
            model_id=experiment.resolve().model,
            tiny=False,
            prompt="A",
            max_output_tokens=2,
            warmups=0,
            repetitions=1,
            timeout_seconds=120.0,
            experiment=experiment,
        )
        assert report["checks"]["passed"]
        candidates.append((experiment, report))
    comparison = build_comparison_report(candidates)
    assert comparison["checks"]["passed"], comparison["checks"]
    assert comparison["checks"]["matched_workload"]
    assert comparison["comparison_key"]["output_tokens"] > 0
    assert {
        frozenset(report["configuration"]["roles"]) for _, report in candidates
    } == {frozenset({"client"}), frozenset({"client", "preparation", "inference"})}
    verified_report = candidates[2][1]
    assert verified_report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"] > 0
    assert comparison["candidates"][2]["pipeline"]["components"]["verification"]["component"] == (
        "pllm/freivalds-verify/v1"
    )


@pytest.mark.parametrize("topology", ["prepared", "verified", "offset"])
def test_metal_kernel_executes_public_provider_experiment(
    tmp_path: Path, topology: str,
) -> None:
    _require_metal()
    from pllm.kernels import AppleMetal
    from pllm.profiles import TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
    from pllm.roles import PreparedProviderRoles

    local, _ = _setup(tmp_path)
    model = local.pipeline.model
    if topology == "prepared":
        pipeline = MaskedLinearCpu(model, kernels=AppleMetal(min_rows=2))
    elif topology == "verified":
        pipeline = VerifiedMaskedLinearCpu(
            model, topology=PreparedProviderRoles(), kernels=AppleMetal(min_rows=2),
        )
    else:
        pipeline = TwoOnlineOffsetCpu(model, kernels=AppleMetal(min_rows=2))
    metal = Experiment(
        name=f"metal-{topology}",
        pipeline=pipeline,
        deployment=Deployment.local(root=str(tmp_path / topology)),
        budget=local.budget,
    )
    assert metal.resolve().composition_digest == pipeline.digest()
    report = run_loopback_benchmark(
        model=model.source,
        model_id=metal.resolve().model,
        tiny=False,
        prompt="A",
        max_output_tokens=2,
        warmups=0,
        repetitions=1,
        timeout_seconds=120.0,
        experiment=metal,
    )
    assert report["checks"]["passed"], report["checks"]
    assert report["configuration"]["roles"] == (
        ["client", "inference", "preparation"]
        if topology != "offset" else ["client", "worker_a", "worker_b"]
    )
    assert report["topology_accounting"]["runs"][0]["online_all_link_serialized_body_bytes"] > 0
    reference_pipeline = (
        MaskedLinearCpu(model) if topology == "prepared"
        else VerifiedMaskedLinearCpu(model, topology=PreparedProviderRoles())
        if topology == "verified" else TwoOnlineOffsetCpu(model)
    )
    reference = Experiment(
        name=f"cpu-{topology}", pipeline=reference_pipeline,
        deployment=Deployment.local(root=str(tmp_path / f"cpu-{topology}")),
        budget=local.budget,
    )
    outputs = []
    for candidate in (reference, metal):
        with build_roles(candidate) as roles, TestClient(
            roles.gateway_app(local_api_key="metal-parity")
        ) as gateway:
            result = gateway.post(
                "/v1/responses",
                headers={"Authorization": "Bearer metal-parity"},
                json={"model": candidate.resolve().model, "input": "A", "max_output_tokens": 2,
                      "temperature": 0},
            )
            assert result.status_code == 200, result.text
            outputs.append(result.json()["output"][0]["content"][0]["text"])
    assert outputs[0] == outputs[1]
