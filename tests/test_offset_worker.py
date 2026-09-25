"""Research workers admit compiled sessions before processing private shares."""

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np

import pllm
from pllm.configuration import Pipeline
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import two_online_reference_graph
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.offset_worker import _MAX_BODY_BYTES, create_offset_worker_app
from pllm.runtime.stage_protocol import MaskedStageRequest, MaskedStageResponse
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_worker_rejects_unbound_launch_before_checkpoint_load(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PLLM_OFFSET_WORKER_API_KEY"] = "test-key-" + "a" * 32
    environment.pop("PLLM_OFFSET_EXPERIMENT_JSON", None)
    result = subprocess.run(
        [
            sys.executable, "-m", "pllm.runtime.offset_worker",
            str(tmp_path / "missing-checkpoint"), "--model-id", "missing-model",
            "--role", "worker_a", "--port", "45678", "--weight-bits", "8",
            "--activation-bits", "8",
        ],
        env=environment, text=True, capture_output=True, timeout=15, check=False,
    )
    assert result.returncode == 2
    assert "requires an immutable Experiment" in result.stderr
    assert "missing-checkpoint" not in result.stderr


def _fixture(tmp_path: Path):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", num_hidden_layers=1, model_type="qwen2", with_qkv_bias=True,
    )
    manifest = load_hf_directory(root, model_id="offset-model")
    workers = (MaskedTransformerEngine(threads=1), MaskedTransformerEngine(threads=1))
    for worker in workers:
        asyncio.run(worker.load(manifest))
    bundle = ClientBundle.unpack(workers[0].client_bundle("offset-model"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    compiled = compile_runtime_model(
        pllm.lower_model(config, batch=1, max_input_tokens=4, max_new_tokens=2), bundle,
    )
    return compiled, workers


def _session_body(compiled, role: str) -> dict[str, str | int]:
    metadata = compiled._bundle.manifest["metadata"]
    return {
        "schema": "pllm.offset_worker_session.v1",
        "model": "offset-model", "role": role,
        "topology_digest": two_online_reference_graph().digest(),
        "decoder_plan": compiled._plan.digest,
        "max_input_tokens": 4, "max_new_tokens": 2,
        "body_fingerprint": metadata["body_fingerprint"],
        "stage_commitment": metadata["seeded_stage_commitment"],
        "runtime_config_digest": metadata["runtime_config_digest"],
        "composition_digest": Pipeline.from_spec(
            json.loads(compiled._canonical_composition)
        ).digest(),
    }


def _request(compiled, session_id: str, *, ticket: str = "ab" * 16) -> tuple[str, bytes]:
    binding = next(stage for stage in compiled._stages if stage.client_weight_layout is None)
    stage = compiled._bundle.stages[binding.stage_id]
    profile = stage.seeded_profile
    assert profile is not None
    return binding.stage_id, MaskedStageRequest(
        model="offset-model", stage_id=binding.stage_id, correlation_id=ticket,
        masked_input=np.zeros((1, stage.in_features), dtype=np.uint32),
        activation_scales=np.ones(1, dtype=np.float32),
        modulus=profile.modulus, wire_bits=profile.wire_bits, ring=profile.ring,
        body_fingerprint=compiled._bundle.manifest["metadata"]["body_fingerprint"],
        weight_digest=stage.weight_digest,
        weight_bits=stage.weight_bits, activation_bits=stage.activation_bits,
        session_id=session_id, out_features=stage.out_features,
        signed_output_bound=profile.signed_output_bound,
    ).pack()


def test_offset_workers_admit_same_compiled_plan_and_burn_replayed_stage(
    tmp_path: Path,
) -> None:
    compiled, (first, second) = _fixture(tmp_path)
    token = "a" * 32

    async def scenario() -> None:
        apps = (
            create_offset_worker_app(first, model_id="offset-model", role_id="worker_a", api_key=token),
            create_offset_worker_app(second, model_id="offset-model", role_id="worker_b", api_key=token),
        )
        clients = [httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://worker.test") for app in apps]
        try:
            sessions = []
            for role, client in zip(("worker_a", "worker_b"), clients, strict=True):
                body = _session_body(compiled, role)
                assert (await client.post("/v1/offset-reference/sessions", json=body)).status_code == 401
                response = await client.post(
                    "/v1/offset-reference/sessions", json=body,
                    headers={"authorization": f"Bearer {token}"},
                )
                assert response.status_code == 200, response.text
                admitted = response.json()
                assert admitted["decoder_plan"] == compiled._plan.digest
                assert admitted["role"] == role
                sessions.append(admitted["id"])
            assert sessions[0] != sessions[1]

            for client, session in zip(clients, sessions, strict=True):
                stage, payload = _request(compiled, session)
                route = f"/v1/offset-reference/sessions/{session}/stages/{stage}"
                response = await client.post(
                    route, content=payload, headers={"authorization": f"Bearer {token}"},
                )
                assert response.status_code == 200, response.text
                decoded = MaskedStageResponse.unpack(response.content)
                assert decoded.correlation_id == "ab" * 16
                assert np.all(decoded.masked_output == 0)
                assert (await client.post(route, content=payload,
                                          headers={"authorization": f"Bearer {token}"})).status_code == 400
                assert (await client.post(route, content=_request(compiled, session, ticket="cd" * 16)[1],
                                          headers={"authorization": f"Bearer {token}"})).status_code == 409
        finally:
            for client in clients:
                await client.aclose()

    asyncio.run(scenario())


def test_offset_worker_rejects_forged_plan_role_and_shape_before_execution(
    tmp_path: Path,
) -> None:
    compiled, (worker, _) = _fixture(tmp_path)
    token = "a" * 32

    async def scenario() -> None:
        app = create_offset_worker_app(worker, model_id="offset-model", role_id="worker_a", api_key=token)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://worker.test") as client:
            headers = {"authorization": f"Bearer {token}"}
            assert (await client.get("/v1/offset-reference/metrics")).status_code == 401
            metrics = await client.get("/v1/offset-reference/metrics", headers=headers)
            assert metrics.status_code == 200
            assert metrics.json()["cpu_ns"] > 0
            descriptor = await client.get("/v1/runtime/models/offset-model", headers=headers)
            assert descriptor.status_code == 200
            assert descriptor.json()["runtime"]["client_runtime"] == "compiled_offset_v1"
            assert (await client.get(
                "/v1/runtime/models/offset-model/client-bundle",
            )).status_code == 401
            packed = await client.get(
                "/v1/runtime/models/offset-model/client-bundle", headers=headers,
            )
            assert packed.status_code == 200
            assert packed.headers["X-PLLM-Bundle-SHA256"] == descriptor.json()["client_bundle"]["sha256"]
            offset_bundle = ClientBundle.unpack(packed.content)
            assert offset_bundle.privacy["preprocessed"] is False
            offset_plan = compiled._plan
            assert compile_runtime_model(
                offset_plan, offset_bundle,
                composition=TwoOnlineOffsetCpu(
                    pllm.Model("offset-model"),
                    quantization=SymmetricPerRow(weight_bits=4, activation_bits=4),
                ),
            ).complete
            body = _session_body(compiled, "worker_a")
            for field, forged in (("role", "worker_b"), ("decoder_plan", "0" * 64),
                                  ("stage_commitment", "0" * 64),
                                  ("topology_digest", "0" * 64)):
                changed = dict(body)
                changed[field] = forged
                assert (await client.post("/v1/offset-reference/sessions", json=changed,
                                          headers=headers)).status_code == 409
            response = await client.post("/v1/offset-reference/sessions", json=body,
                                         headers=headers)
            assert response.status_code == 200, response.text
            session = response.json()["id"]
            stage, payload = _request(compiled, session)
            forged = replace(MaskedStageRequest.unpack(payload), session_id="0" * 32)
            route = f"/v1/offset-reference/sessions/{session}/stages/{stage}"
            assert (await client.post(route, content=forged.pack(), headers=headers)).status_code == 400
            assert (await client.post(route, content=payload, headers=headers)).status_code == 409
            assert worker.models["offset-model"].stages[stage].calls == 0

    asyncio.run(scenario())


def test_experiment_bound_worker_rejects_other_composition(tmp_path: Path) -> None:
    compiled, (worker, _) = _fixture(tmp_path)
    composition = TwoOnlineOffsetCpu(
        pllm.Model("offset-model"),
        quantization=SymmetricPerRow(weight_bits=4, activation_bits=4),
    )
    token = "a" * 32

    async def scenario() -> None:
        app = create_offset_worker_app(
            worker, model_id="offset-model", role_id="worker_a", api_key=token,
            composition=composition,
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://worker.test") as client:
            headers = {"authorization": f"Bearer {token}"}
            body = _session_body(compiled, "worker_a")
            assert (await client.post(
                "/v1/offset-reference/sessions", json=body, headers=headers,
            )).status_code == 409
            body["composition_digest"] = composition.digest()
            assert (await client.post(
                "/v1/offset-reference/sessions", json=body, headers=headers,
            )).status_code == 200

    asyncio.run(scenario())


def test_oversize_share_burns_admitted_session_before_kernel(tmp_path: Path) -> None:
    compiled, (worker, _) = _fixture(tmp_path)
    token = "a" * 32

    async def scenario() -> None:
        app = create_offset_worker_app(worker, model_id="offset-model", role_id="worker_a", api_key=token)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://worker.test") as client:
            headers = {"authorization": f"Bearer {token}"}
            admitted = await client.post(
                "/v1/offset-reference/sessions", json=_session_body(compiled, "worker_a"),
                headers=headers,
            )
            assert admitted.status_code == 200
            session_id = admitted.json()["id"]
            stage, valid = _request(compiled, session_id)
            endpoint = f"/v1/offset-reference/sessions/{session_id}/stages/{stage}"
            assert (await client.post(endpoint, content=b"x" * (_MAX_BODY_BYTES + 1),
                                      headers=headers)).status_code == 413
            assert (await client.post(endpoint, content=valid, headers=headers)).status_code == 409
            assert worker.models["offset-model"].stages[stage].calls == 0

    asyncio.run(scenario())
