"""Two independently hosted research workers run one compiled decoder session."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest

import pllm
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_cluster import LocalOffsetCluster
from pllm.runtime.offset_reference import OffsetReferenceError, TwoOnlineOffsetTransport
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, RemoteLinear
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.mark.parametrize("model_type", ["qwen2", "qwen3"])
def test_separate_offset_worker_processes_match_compiled_prefill_and_decode(
    tmp_path: Path, model_type: str,
) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type=model_type,
        with_qkv_bias=model_type == "qwen2", qk_norm=model_type == "qwen3",
    )
    model_id = "offset-model"
    manifest = load_hf_directory(root, model_id=model_id)
    local_worker = MaskedTransformerEngine(threads=1)
    asyncio.run(local_worker.load(manifest))
    bundle = ClientBundle.unpack(local_worker.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
    )
    with LocalOffsetCluster(root, model_id=model_id) as cluster:
        assert cluster.processes[0].pid != cluster.processes[1].pid
        for worker in cluster.processes:
            assert all(key not in " ".join(map(str, worker.args)) for key in cluster.keys)
        with TwoOnlineOffsetTransport(
            compiled, model_id=model_id,
            worker_a=cluster.clients[0], worker_b=cluster.clients[1],
            api_key_a=cluster.keys[0], api_key_b=cluster.keys[1],
        ) as transport:

            class LocalCorrelations:
                model_id = "offset-model"

                def take_many(self, stage, count):
                    return local_worker.create_local_correlations(model_id, stage.id, count)

            def exchange(stage_id: str, payloads: list[bytes]) -> list[bytes]:
                stage = local_worker.models[model_id].stages[stage_id].spec
                return asyncio.run(local_worker.execute_stage(model_id, stage, payloads))

            baseline = compiled.session(RemoteLinear(bundle.stages, LocalCorrelations(), exchange))
            candidate = compiled.session(transport)
            for session in (baseline, candidate):
                session.prefill_ids([0, 2])
            np.testing.assert_allclose(candidate.logits, baseline.logits, rtol=0, atol=1e-4)
            assert candidate.select_next() == baseline.select_next()
            for session in (baseline, candidate):
                session.decode_selected()
            np.testing.assert_allclose(candidate.logits, baseline.logits, rtol=0, atol=1e-4)
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


def test_mismatched_second_worker_cancels_first_admitted_session(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "first", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    other = create_tiny_llama_checkpoint(
        tmp_path / "second", num_hidden_layers=1, model_type="qwen2",
        with_qkv_bias=True, gate_weight_scale=0.02,
    )
    model_id = "offset-model"
    first = MaskedTransformerEngine(threads=1)
    asyncio.run(first.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(first.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
    )
    with LocalOffsetCluster(root, model_id=model_id, second_checkpoint=other) as cluster:
        sessions: list[str] = []

        def observe(response: httpx.Response) -> None:
            if response.request.url.path == "/v1/offset-reference/sessions" and response.status_code == 200:
                response.read()
                sessions.append(response.json()["id"])

        cluster.clients[0].event_hooks["response"].append(observe)
        with pytest.raises(OffsetReferenceError, match="declined the bound decoder"):
            TwoOnlineOffsetTransport(
                compiled, model_id=model_id,
                worker_a=cluster.clients[0], worker_b=cluster.clients[1],
                api_key_a=cluster.keys[0], api_key_b=cluster.keys[1],
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
        response = cluster.clients[0].post(
            f"/v1/offset-reference/sessions/{sessions[0]}/stages/{stage.stage_id}",
            content=request, headers={"authorization": f"Bearer {cluster.keys[0]}"},
        )
        assert response.status_code == 409


def test_forged_worker_result_burns_both_admitted_sessions(tmp_path: Path) -> None:
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    model_id = "offset-model"
    engine = MaskedTransformerEngine(threads=1)
    asyncio.run(engine.load(load_hf_directory(root, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
    )
    with LocalOffsetCluster(root, model_id=model_id) as cluster:
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

        for index, client in enumerate(cluster.clients):
            client.event_hooks["response"].append(observer(index))
        with TwoOnlineOffsetTransport(
            compiled, model_id=model_id,
            worker_a=cluster.clients[0], worker_b=cluster.clients[1],
            api_key_a=cluster.keys[0], api_key_b=cluster.keys[1],
        ) as transport:
            stage = next(binding for binding in compiled._stages if binding.client_weight_layout is None)
            with pytest.raises(OffsetReferenceError, match="result differs"):
                transport(stage.stage_id, np.zeros((1, stage.in_features), dtype=np.float32))
            assert transport.costs.stages == 0
        for index, client in enumerate(cluster.clients):
            assert len(sessions[index]) == 1
            response = client.post(
                f"/v1/offset-reference/sessions/{sessions[index][0]}/complete",
                headers={"authorization": f"Bearer {cluster.keys[index]}"},
            )
            assert response.status_code == 409
