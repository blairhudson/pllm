"""Two separately supervised workers run one compiled decoder session."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

import pllm
from pllm.protocols import TwoOnlineOffsetLinear
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_reference import OffsetReferenceError, TwoOnlineOffsetTransport
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, RemoteLinear
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def _offset_experiment(
    root: Path, model_id: str, deployment_root: Path, *, max_input_tokens: int = 4,
    input_encoding: str = "raw", output_encoding: str = "raw",
):
    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.quantization import SymmetricPerRow

    return Experiment(
        name="offset-worker-test",
        pipeline=TwoOnlineOffsetCpu(
            pllm.Model.path(str(root), model_id=model_id),
            quantization=SymmetricPerRow(weight_bits=4, activation_bits=4),
            linear=TwoOnlineOffsetLinear(input_encoding=input_encoding, output_encoding=output_encoding),
        ),
        deployment=Deployment.local(root=str(deployment_root)),
        budget=ExecutionBudget(max_input_tokens=max_input_tokens, max_new_tokens=2, requests=1),
    )


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("input_encoding", ["raw", "seeded"])
@pytest.mark.parametrize("output_encoding", ["raw", "row_residues"])
def test_two_worker_experiment_uses_shared_roles_and_responses(
    tmp_path: Path, model_type: str, input_encoding: str, output_encoding: str,
) -> None:
    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.profiles import TwoOnlineOffsetCpu
    from pllm.runtime.servers import build_roles

    model_id = f"offset-{model_type}"
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1,
        model_type=model_type, with_qkv_bias=model_type == "qwen2",
        qk_norm=model_type == "qwen3",
    )
    experiment = Experiment(
        name=f"offset-{model_type}",
        pipeline=TwoOnlineOffsetCpu(pllm.Model.path(str(root), model_id=model_id),
            linear=TwoOnlineOffsetLinear(input_encoding=input_encoding, output_encoding=output_encoding)),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )
    assert tuple(role.id for role in experiment.resolve().role_graph.roles) == (
        "client", "worker_a", "worker_b",
    )
    with build_roles(experiment, engine_threads=1) as topology:
        statuses = topology.statuses
        assert {status.role for status in statuses} == {"worker_a", "worker_b"}
        assert all(status.running for status in statuses)
        assert statuses[0].pid != statuses[1].pid
        connections = topology.worker_connections()
        with topology.client() as client:
            response = client.responses.create(model=model_id, input="A", max_output_tokens=2)
            assert response.model == model_id
            assert response.usage.input_tokens > 0
            assert response.usage.output_tokens > 0
            audit = client.privacy_audit.to_dict()
            assert audit["inference_stage_calls"] > 0
            for role in ("worker_a", "worker_b"):
                assert audit[f"role_link.{role}.online_upload_bytes"] > 0
                assert audit[f"role_link.{role}.online_download_bytes"] > 0
        for role in connections:
            assert all(connections[role][1] not in str(status) for status in statuses)
        with TestClient(topology.gateway_app(local_api_key="local")) as gateway:
            answer = gateway.post(
                "/v1/responses", headers={"Authorization": "Bearer local"},
                json={"model": model_id, "input": "A", "max_output_tokens": 2},
            )
            assert answer.status_code == 200, answer.text
            assert answer.json()["model"] == model_id


def test_two_worker_sdk_owns_and_closes_its_selected_roles(tmp_path: Path) -> None:
    from pllm import Deployment, ExecutionBudget, Experiment
    from pllm.profiles import TwoOnlineOffsetCpu

    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2",
        with_qkv_bias=True,
    )
    model_id = "sdk-offset"
    experiment = Experiment(
        name=model_id,
        pipeline=TwoOnlineOffsetCpu(pllm.Model.path(str(root), model_id=model_id)),
        deployment=Deployment.local(root=str(tmp_path)),
        budget=ExecutionBudget(max_input_tokens=64, max_new_tokens=2, requests=1),
    )
    with pllm.OpenAI(experiment=experiment) as client:
        response = client.responses.create(model=model_id, input="A", max_output_tokens=2)
        assert response.usage.input_tokens > 0
        assert client.privacy_audit.to_dict()["inference_stage_calls"] > 0
    with pytest.raises(RuntimeError, match="closed"):
        client.responses.create(model=model_id, input="A", max_output_tokens=2)


def test_closing_client_burns_paused_two_worker_stream(tmp_path: Path) -> None:
    from pllm.runtime.servers import build_roles

    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2",
        with_qkv_bias=True,
    )
    model_id = "paused-offset"
    experiment = _offset_experiment(root, model_id, tmp_path, max_input_tokens=64)
    with build_roles(experiment, engine_threads=1) as topology:
        client = topology.client()
        worker, key = client._core._offset_workers["worker_a"]
        sessions: list[str] = []

        def observe(response: httpx.Response) -> None:
            if response.request.url.path == "/v1/offset-reference/sessions" and response.status_code == 200:
                response.read()
                sessions.append(response.json()["id"])

        worker.event_hooks["response"].append(observe)
        stream = client.responses.create(
            model=model_id, input="A", max_output_tokens=2, stream=True,
        )
        next(stream)
        assert len(sessions) == 1
        client.close()
        with httpx.Client(base_url=topology.worker_connections()["worker_a"][0]) as verifier:
            replay = verifier.post(
                f"/v1/offset-reference/sessions/{sessions[0]}/complete",
                headers={"authorization": f"Bearer {key}"},
            )
            assert replay.status_code == 409
        stream.close()


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
@pytest.mark.parametrize("encoding", ["raw", "combined"])
def test_separate_offset_worker_processes_match_compiled_prefill_and_decode(
    tmp_path: Path, model_type: str, encoding: str,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type=model_type,
        with_qkv_bias=model_type == "qwen2", qk_norm=model_type == "qwen3",
    )
    model_id = "offset-model"
    manifest = load_hf_directory(root, model_id=model_id)
    local_worker = MaskedTransformerEngine(threads=1)
    asyncio.run(local_worker.load(manifest))
    output_encoding = "row_residues" if encoding == "combined" else "raw"
    bundle = ClientBundle.unpack(local_worker.client_bundle(model_id, placement="offset", output_encoding=output_encoding))
    baseline_bundle = ClientBundle.unpack(local_worker.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2)
    experiment = _offset_experiment(root, model_id, tmp_path,
        input_encoding="seeded" if encoding == "combined" else "raw", output_encoding=output_encoding)
    compiled = compile_runtime_model(
        plan, bundle, composition=experiment.pipeline,
    )
    prepared_compiled = compile_runtime_model(plan, baseline_bundle)
    from pllm.runtime.servers import build_roles

    with build_roles(experiment, engine_threads=1) as topology:
        connections = topology.worker_connections()
        assert topology.statuses[0].pid != topology.statuses[1].pid
        assert all(
            key not in str(status) for _role, (_url, key) in connections.items()
            for status in topology.statuses
        )
        clients = [httpx.Client(base_url=connections[role][0]) for role in ("worker_a", "worker_b")]
        keys = [connections[role][1] for role in ("worker_a", "worker_b")]
        with clients[0], clients[1]:
            _check_two_worker_parity(
                compiled, prepared_compiled, bundle, baseline_bundle, local_worker,
                model_id, clients, keys, topology,
            )


def _check_two_worker_parity(
    compiled, prepared_compiled, bundle, baseline_bundle, local_worker, model_id,
    clients, keys, topology,
) -> None:
        with TwoOnlineOffsetTransport(
            compiled, model_id=model_id,
            worker_a=clients[0], worker_b=clients[1],
            api_key_a=keys[0], api_key_b=keys[1],
        ) as transport:

            class LocalCorrelations:
                model_id = "offset-model"

                def take_many(self, stage, count):
                    return local_worker.create_local_correlations(model_id, stage.id, count)

            def exchange(stage_id: str, payloads: list[bytes]) -> list[bytes]:
                stage = local_worker.models[model_id].stages[stage_id].spec
                return asyncio.run(local_worker.execute_stage(model_id, stage, payloads))

            baseline = prepared_compiled.session(
                RemoteLinear(baseline_bundle.stages, LocalCorrelations(), exchange)
            )
            candidate = compiled.session(transport)

            def assert_numeric_parity():
                np.testing.assert_array_equal(candidate.logits, baseline.logits)
                for actual, expected in zip(candidate._runtime.caches, baseline._runtime.caches, strict=True):
                    assert actual.length == expected.length
                    np.testing.assert_array_equal(actual.key[:actual.length], expected.key[:expected.length])
                    np.testing.assert_array_equal(actual.value[:actual.length], expected.value[:expected.length])

            for session in (baseline, candidate):
                session.prefill_ids([0, 2])
            assert_numeric_parity()
            assert candidate.select_next() == baseline.select_next()
            for session in (baseline, candidate):
                session.decode_selected()
            assert_numeric_parity()
            assert candidate.select_next() == baseline.select_next()
            candidate.finish()
            baseline.finish()
            transport.complete()
            assert transport.costs.stages == 8
            assert transport.costs.total_integer_macs == 55_296
            assert transport.costs.total_stage_body_bytes > 0
            bodies = transport.http_body_costs
            assert sum(
                item["online_upload_bytes"] + item["online_download_bytes"]
                for item in bodies.values()
            ) == transport.costs.total_stage_body_bytes
            assert all(item["setup_upload_bytes"] > 0 for item in bodies.values())
            assert all(item["teardown_download_bytes"] > 0 for item in bodies.values())
            metrics = topology.worker_process_metrics()
            assert all(metrics[role]["cpu_ns"] > 0 for role in ("worker_a", "worker_b"))


def test_mismatched_second_worker_cancels_first_admitted_session(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "first", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    other = create_tiny_llama_checkpoint(
        tmp_path / "second", num_hidden_layers=1, model_type="qwen2",
        with_qkv_bias=True, gate_weight_scale=0.02,
    )
    model_id = "offset-model"
    experiment = _offset_experiment(root, model_id, tmp_path)
    first = MaskedTransformerEngine(threads=1)
    asyncio.run(first.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(first.client_bundle(model_id, placement="offset"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
        composition=experiment.pipeline,
    )
    from pllm.runtime.servers import build_roles

    other_experiment = _offset_experiment(other, model_id, tmp_path / "other-deployment")
    with (
        build_roles(experiment, engine_threads=1) as topology,
        build_roles(other_experiment, engine_threads=1) as mismatched,
    ):
        worker_url, worker_key = topology.worker_connections()["worker_a"]
        other_url, other_key = mismatched.worker_connections()["worker_b"]
        with httpx.Client(base_url=worker_url) as worker_a, httpx.Client(base_url=other_url) as second:
            _check_mismatched_worker_cancellation(
                compiled, bundle, model_id, worker_a, second, worker_key, other_key,
            )


def _check_mismatched_worker_cancellation(
    compiled, bundle, model_id, worker_a, worker_b, key_a, key_b,
) -> None:
        sessions: list[str] = []

        def observe(response: httpx.Response) -> None:
            if response.request.url.path == "/v1/offset-reference/sessions" and response.status_code == 200:
                response.read()
                sessions.append(response.json()["id"])

        worker_a.event_hooks["response"].append(observe)
        with pytest.raises(OffsetReferenceError, match="declined the bound decoder"):
            TwoOnlineOffsetTransport(
                compiled, model_id=model_id,
                worker_a=worker_a, worker_b=worker_b,
                api_key_a=key_a, api_key_b=key_b,
            )
        assert len(sessions) == 1
        stage = next(binding for binding in compiled._stages if binding.client_weight_layout is None)
        stage_metadata = bundle.stages[stage.stage_id]
        profile = stage_metadata.seeded_profile
        assert profile is not None
        request = MaskedStageRequest(
            model=model_id, stage_id=stage.stage_id, correlation_id="ab" * 16,
            masked_input=np.zeros((1, stage.in_features), dtype=np.uint32),
            activation_scales=np.ones(1, np.float32), ring=profile.ring,
            modulus=profile.modulus, wire_bits=profile.wire_bits,
            body_fingerprint=bundle.manifest["metadata"]["body_fingerprint"],
            weight_digest=stage_metadata.weight_digest,
            weight_bits=stage_metadata.weight_bits,
            activation_bits=stage_metadata.activation_bits,
            session_id=sessions[0], out_features=stage.out_features,
            signed_output_bound=profile.signed_output_bound,
        ).pack()
        response = worker_a.post(
            f"/v1/offset-reference/sessions/{sessions[0]}/stages/{stage.stage_id}",
            content=request, headers={"authorization": f"Bearer {key_a}"},
        )
        assert response.status_code == 409


def test_forged_worker_result_burns_both_admitted_sessions(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    model_id = "offset-model"
    experiment = _offset_experiment(root, model_id, tmp_path)
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="offset"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
        composition=experiment.pipeline,
    )
    from pllm.runtime.servers import build_roles

    with build_roles(experiment, engine_threads=1) as topology:
        connections = topology.worker_connections()
        clients = [httpx.Client(base_url=connections[role][0]) for role in ("worker_a", "worker_b")]
        keys = [connections[role][1] for role in ("worker_a", "worker_b")]
        with clients[0], clients[1]:
            _check_forged_worker_burn(compiled, model_id, clients, keys)


def _check_forged_worker_burn(compiled, model_id, clients, keys) -> None:
        sessions: list[list[str]] = [[], []]

        def observer(index: int):
            def observe(response: httpx.Response) -> None:
                if response.request.url.path == "/v1/offset-reference/sessions" and response.status_code == 200:
                    response.read()
                    sessions[index].append(response.json()["id"])
                if index == 1 and "/stages/" in response.request.url.path and response.status_code == 200:
                    response.read()
                    value = MaskedStageResponse.unpack(response.content)
                    response._content = replace(value, correlation_id="0" * 32).pack()
            return observe

        for index, client in enumerate(clients):
            client.event_hooks["response"].append(observer(index))
        with TwoOnlineOffsetTransport(
            compiled, model_id=model_id,
            worker_a=clients[0], worker_b=clients[1],
            api_key_a=keys[0], api_key_b=keys[1],
        ) as transport:
            stage = next(binding for binding in compiled._stages if binding.client_weight_layout is None)
            with pytest.raises(OffsetReferenceError, match="result differs"):
                transport(stage.stage_id, np.zeros((1, stage.in_features), dtype=np.float32))
            assert transport.costs.stages == 0
        for index, client in enumerate(clients):
            assert len(sessions[index]) == 1
            response = client.post(
                f"/v1/offset-reference/sessions/{sessions[index][0]}/complete",
                headers={"authorization": f"Bearer {keys[index]}"},
            )
            assert response.status_code == 409
