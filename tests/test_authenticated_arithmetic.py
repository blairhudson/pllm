from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest

from pllm.runtime.authenticated_mpc import (
    AuthenticatedMPC,
    AuthenticationError,
    TrustedPreprocessor,
)
from pllm.runtime.formal_security import SECURE_PREVIEW, PROPRIETARY_GUARDED
from pllm.runtime.guarded_engine import GuardPolicy, GuardedBlindedTransformerEngine
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.privacy import ProprietaryProtocol
from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.stage_protocol import BlindedStageRequest, BlindedStageResponse
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import TransformerEngineError


def run(value):
    return asyncio.run(value)


def tiny_model(root: Path) -> Path:
    path = create_tiny_gemma4_checkpoint(
        root,
        vocab_size=8,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    config = json.loads((path / "config.json").read_text())
    config.update({"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 3})
    (path / "config.json").write_text(json.dumps(config))
    return path


def _request(engine, manifest, stage, activation, owner="owner"):
    runtime = engine.models[manifest.id].stages[stage.id]
    qa = quantize_activation_per_row(np.asarray(activation, dtype=np.float32), bits=4)
    rows = engine.create_local_blinded_correlations(
        manifest.id, stage.id, owner, qa.rows, seed=None
    )
    masks = np.stack([row.mask for row in rows])
    request = BlindedStageRequest(
        model=manifest.id,
        stage_id=stage.id,
        owner_id=owner,
        correlation_ids=tuple(row.id for row in rows),
        masked_input=((qa.values.astype(np.int64) - masks.astype(np.int64)) % runtime.modulus).astype(np.uint32),
        activation_scales=qa.scales,
        modulus=runtime.modulus,
        wire_bits=runtime.wire_bits,
        ring="prime",
    )
    blinded = np.stack([row.blinded_transformed_mask for row in rows])
    return qa, rows, request, blinded, runtime


def test_protocol_names_and_branding():
    import pllm

    assert pllm.__version__ == __import__("pllm._version", fromlist=["__version__"]).__version__
    assert ProprietaryProtocol.parse("fast") is ProprietaryProtocol.GUARDED
    assert ProprietaryProtocol.parse("mpc") is ProprietaryProtocol.SECURE
    assert PROPRIETARY_GUARDED.malicious_client_security is False
    assert SECURE_PREVIEW.malicious_client_security is True


def test_guarded_stage_exact_and_budgeted(tmp_path: Path):
    root = tiny_model(tmp_path / "guarded")
    manifest = load_hf_directory(root, model_id="tiny-guarded")
    engine = GuardedBlindedTransformerEngine(
        threads=1,
        guard_policy=GuardPolicy(
            max_rows_per_request=2,
            max_rows_per_owner_stage=2,
            max_requests_per_owner_minute=4,
        ),
    )
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    activation = np.random.default_rng(1).normal(size=(2, stage.in_features)).astype(np.float32)
    qa, _, request, blinded, runtime = _request(engine, manifest, stage, activation)
    raw = run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
    response = BlindedStageResponse.unpack(raw)
    actual = (
        response.masked_output.astype(np.uint64) + blinded.astype(np.uint64)
    ) % runtime.modulus
    actual = actual.astype(np.int64)
    actual = np.where(actual > runtime.modulus // 2, actual - runtime.modulus, actual)
    expected = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
    assert np.array_equal(actual, expected)
    assert engine.guard_stats()["accepted_requests"] == 1

    _, _, second, _, _ = _request(
        engine,
        manifest,
        stage,
        np.zeros((1, stage.in_features), dtype=np.float32),
    )
    with pytest.raises(TransformerEngineError, match="budget"):
        run(engine.execute_stage(manifest.id, stage, [second.pack()]))
    assert engine.guard_stats()["rejected_requests"] == 1

    bundle = msgpack.unpackb(engine.client_bundle(manifest.id), raw=False)
    assert bundle["privacy"]["protocol"] == "guarded_blinded_w4a4"
    assert bundle["privacy"]["malicious_client_model_privacy"] is False


def test_guarded_optional_dither_is_bounded(tmp_path: Path):
    root = tiny_model(tmp_path / "dither")
    manifest = load_hf_directory(root, model_id="tiny-dither")
    engine = GuardedBlindedTransformerEngine(
        threads=1,
        guard_policy=GuardPolicy(output_dither_bound=2),
    )
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    qa, _, request, blinded, runtime = _request(
        engine,
        manifest,
        stage,
        np.random.default_rng(2).normal(size=(1, stage.in_features)),
    )
    response = BlindedStageResponse.unpack(
        run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
    )
    actual = (
        response.masked_output.astype(np.uint64) + blinded.astype(np.uint64)
    ) % runtime.modulus
    actual = actual.astype(np.int64)
    actual = np.where(actual > runtime.modulus // 2, actual - runtime.modulus, actual)
    expected = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
    error = actual - expected
    assert np.max(np.abs(error)) <= 2


def test_authenticated_mpc_linear_and_tamper_detection():
    dealer = TrustedPreprocessor(seed=3)
    runtime = AuthenticatedMPC(dealer)
    x = np.array([[3, -2, 7, 1]], dtype=np.int64)
    weight = np.array([[2, 1, -1, 3], [-2, 4, 0, 1]], dtype=np.int64)
    shared_x = runtime.input(x, dealer.input_mask(x.shape))
    assert not np.array_equal(shared_x.client.value, x % dealer.modulus)
    correlation = dealer.linear_correlation(weight, x.shape)
    shared_y = runtime.linear(shared_x, weight, correlation)
    opened = runtime.centered(runtime.open(shared_y))
    assert np.array_equal(opened, x @ weight.T)

    tampered = shared_y.tamper_client(value_delta=1)
    with pytest.raises(AuthenticationError):
        runtime.open(tampered)


def test_authenticated_mpc_polynomial_graph():
    dealer = TrustedPreprocessor(seed=4)
    runtime = AuthenticatedMPC(dealer)
    x = np.array([[-2, -1, 0, 1, 2]], dtype=np.int64)
    shared = runtime.input(x, dealer.input_mask(x.shape))
    # f(x) = 3 + 2x + x^2
    triples = [dealer.multiplication_triple(x.shape) for _ in range(2)]
    output = runtime.polynomial(shared, [3, 2, 1], triples)
    clear = runtime.centered(runtime.open(output))
    assert np.array_equal(clear, 3 + 2 * x + x * x)
    assert runtime.stats.multiplication_rounds == 2


def test_cli_defaults_to_guarded_proprietary(tmp_path: Path, monkeypatch):
    from pllm.runtime import cli

    root = tiny_model(tmp_path / "cli")
    captured = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update(app=app))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pllm serve",
            "--mode",
            "proprietary",
            "--api-key",
            "test",
            "--model",
            str(root),
            "--model-id",
            "tiny",
            "--local-files-only",
        ],
    )
    cli.server_main()
    assert "guarded-transformer-proprietary" in captured["app"].state.engines


def test_gateway_binds_guarded_owner_to_api_principal(tmp_path: Path):
    import httpx
    from conftest import start_gateway

    root = tiny_model(tmp_path / "owner-binding")
    engine = GuardedBlindedTransformerEngine(threads=1)
    gateway = start_gateway(
        engines={engine.capabilities.name: engine},
        privacy_mode="proprietary",
    )
    headers = {"Authorization": f"Bearer {gateway.api_key}"}
    try:
        with httpx.Client(base_url=gateway.base_url, timeout=30) as http:
            loaded = http.post(
                "/v1/he/models/load",
                headers=headers,
                json={
                    "engine": engine.capabilities.name,
                    "kind": "huggingface",
                    "path": str(root),
                    "model_id": "tiny-owner-binding",
                },
            )
            assert loaded.status_code == 200, loaded.text
            sessions = []
            for _ in range(2):
                response = http.post(
                    "/v1/he/sessions",
                    headers=headers,
                    json={"model": "tiny-owner-binding", "max_output_tokens": 1},
                )
                assert response.status_code == 200, response.text
                sessions.append(response.json()["id"])
            body = {
                "count": 1,
                "stage_id": "layers.0.self_attn.qkv_proj",
                "owner_id": "owner-a",
            }
            first = http.post(
                f"/v1/he/sessions/{sessions[0]}/correlations/proprietary/local-test",
                headers=headers,
                json=body,
            )
            assert first.status_code == 200, first.text
            body["owner_id"] = "owner-b"
            second = http.post(
                f"/v1/he/sessions/{sessions[1]}/correlations/proprietary/local-test",
                headers=headers,
                json=body,
            )
            assert second.status_code == 409
    finally:
        gateway.close()
