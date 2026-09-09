from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from conftest import start_gateway, start_preparation
from pllm.runtime import OpenAI, PrivacyMode
from pllm.runtime import hf_hub
from pllm.runtime.client import _BFVStageClient
from pllm.runtime.config import GatewayConfig
from pllm.runtime.hf_hub import _configure_hf_transfer, resolve_huggingface_source
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.proprietary_engine import DirectFHETransformerEngine
from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.stage_protocol import DirectFHEStageRequest, DirectFHEStageResponse
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

PYDEPS = os.environ.get("HE_OPENAI_PYDEPS", "")


def run(value):
    return asyncio.run(value)


def test_privacy_mode_is_a_server_startup_configuration():
    assert PrivacyMode.parse("public") is PrivacyMode.PUBLIC
    assert PrivacyMode.parse("proprietary") is PrivacyMode.PROPRIETARY
    assert GatewayConfig.from_dict({"privacy_mode": "proprietary"}).privacy_mode == "proprietary"


def test_huggingface_local_source_resolution(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "hf-model")
    resolved = resolve_huggingface_source(root, model_id="tiny-hf")
    assert resolved.path == root.resolve()
    assert resolved.model_id == "tiny-hf"
    assert resolved.repo_id is None


def test_huggingface_repo_resolution_uses_parallel_downloader(tmp_path: Path, monkeypatch):
    root = create_tiny_gemma4_checkpoint(tmp_path / "snapshot")
    captured = {}

    def fake_download(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return root.resolve()

    monkeypatch.setattr(hf_hub, "download_huggingface_model", fake_download)
    resolved = resolve_huggingface_source(
        "org/model",
        revision="abc123",
        token="secret",
        cache_dir=tmp_path / "cache",
    )
    assert resolved.path == root.resolve()
    assert resolved.model_id == "org/model"
    assert captured["repo_id"] == "org/model"
    assert captured["revision"] == "abc123"
    assert captured["token"] == "secret"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert "*.safetensors" in captured["allow_patterns"]
    assert "*.bin" in captured["ignore_patterns"]


def test_huggingface_download_defaults_to_resumable_http(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    _configure_hf_transfer()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    _configure_hf_transfer()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


def test_huggingface_environment_auth_and_cache_are_delegated(tmp_path: Path, monkeypatch):
    root = create_tiny_gemma4_checkpoint(tmp_path / "snapshot")
    captured = {}

    def fake_download(repo_id, **kwargs):
        captured.update(repo_id=repo_id, **kwargs)
        return root.resolve()

    monkeypatch.setattr(hf_hub, "download_huggingface_model", fake_download)
    monkeypatch.setenv("HF_TOKEN", "environment-token")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    resolve_huggingface_source("org/model")
    assert captured["token"] is None
    assert captured["cache_dir"] is None


def test_public_and_proprietary_bundles_advertise_distinct_protocols(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "tiny", num_hidden_layers=1)
    manifest = load_hf_directory(root, model_id="tiny")
    public = MaskedTransformerEngine(threads=1, tenseal_path=PYDEPS)
    proprietary = DirectFHETransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(public.load(manifest))
    manifest2 = load_hf_directory(root, model_id="tiny-proprietary")
    run(proprietary.load(manifest2))
    public_bundle = ClientBundle.unpack(public.client_bundle("tiny"))
    proprietary_bundle = ClientBundle.unpack(proprietary.client_bundle("tiny-proprietary"))
    assert public_bundle.privacy["mode"] == "public"
    assert public_bundle.privacy["protocol"] == "masked_w4a4"
    assert public_bundle.privacy["model_weight_correlations_disclosed"] is True
    assert public_bundle.stages["token_lookup"].client_weight is not None
    assert public_bundle.stages["lm_head"].client_weight is not None
    assert proprietary_bundle.privacy["mode"] == "proprietary"
    assert proprietary_bundle.privacy["protocol"] == "direct_bfv_w4a4"
    assert proprietary_bundle.privacy["model_weight_correlations_disclosed"] is False
    assert proprietary_bundle.privacy["malicious_client_model_privacy"] is False
    assert proprietary_bundle.stages["token_lookup"].client_weight is None
    assert proprietary_bundle.stages["lm_head"].client_weight is None


@pytest.mark.he
def test_proprietary_direct_bfv_stage_matches_clear_w4a4(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-direct",
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    manifest = load_hf_directory(root, model_id="tiny-direct")
    engine = DirectFHETransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    runtime = engine.models[manifest.id].stages[stage.id]
    client = _BFVStageClient(plain_modulus=runtime.modulus, pydeps_path=PYDEPS)
    engine.register_bfv_context(manifest.id, "ctx", client.public_context)
    activation = np.random.default_rng(3).normal(size=(2, stage.in_features)).astype(np.float32)
    qa = quantize_activation_per_row(activation, bits=4)
    request = DirectFHEStageRequest(
        model=manifest.id,
        stage_id=stage.id,
        context_id="ctx",
        input_shape=qa.original_shape,
        activation_scales=qa.scales,
        encrypted_rows=tuple(client.encrypt(row) for row in qa.values),
    )
    response = DirectFHEStageResponse.unpack(
        run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
    )
    actual = np.stack([client.decrypt(row, stage_id=stage.id) for row in response.output_rows]).astype(np.int64)
    actual = np.where(actual > runtime.modulus // 2, actual - runtime.modulus, actual)
    expected = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
    assert np.array_equal(actual, expected)


@pytest.mark.he
def test_both_modes_complete_openai_responses_without_remote_plaintext(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(
        tmp_path / "tiny-api",
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config.update({"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 3})
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "he_tokenizer.json").write_text(json.dumps({
        "type": "alphabet",
        "alphabet": " a",
        "vocab_size": 4,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }))
    canary = "PRIVATE-MODE-CANARY-5f21"
    results = {}
    for mode, engine in (
        ("public", MaskedTransformerEngine(threads=1, tenseal_path=PYDEPS)),
        ("proprietary", DirectFHETransformerEngine(threads=1, tenseal_path=PYDEPS)),
    ):
        gateway = start_gateway(
            bfv=True,
            engines={engine.capabilities.name: engine},
            privacy_mode=mode,
        )
        preparation = None
        try:
            model_id = f"tiny-{mode}"
            if mode == "public":
                preparation_engine = MaskedTransformerEngine(threads=1)
                run(preparation_engine.load(load_hf_directory(root, model_id=model_id)))
                preparation = start_preparation(
                    preparation_engine, gateway.base_url, gateway.push_api_key
                )
            with httpx.Client(base_url=gateway.base_url, timeout=120) as admin:
                loaded = admin.post(
                    "/v1/he/models/load",
                    headers={"Authorization": f"Bearer {gateway.api_key}"},
                    json={
                        "engine": engine.capabilities.name,
                        "kind": "huggingface",
                        "path": str(root),
                        "model_id": model_id,
                    },
                )
                assert loaded.status_code == 200, loaded.text
                listed = admin.get(
                    "/v1/models", headers={"Authorization": f"Bearer {gateway.api_key}"}
                ).json()["data"]
                descriptor = next(row for row in listed if row["id"] == model_id)
                assert descriptor["he"]["privacy_mode"] == mode
            with OpenAI(
                api_key=gateway.api_key,
                base_url=gateway.base_url,
                preparation_base_url=preparation.base_url if preparation else None,
                preparation_api_key=preparation.api_key if preparation else None,
                correlation_mode="bfv",
                correlation_prefetch=1,
                tenseal_path=PYDEPS,
                timeout=120,
            ) as client:
                response = client.responses.create(
                    model=model_id,
                    input=canary,
                    max_output_tokens=1,
                    temperature=0,
                )
                assert response.status == "completed"
                audit = client.privacy_audit.to_dict()
                assert audit["plaintext_prompt_bytes_sent"] == 0
                assert audit["plaintext_token_ids_sent"] == 0
                if mode == "public":
                    assert audit["preparation_upload_bytes"] > 0
                    assert audit["inference_upload_bytes"] > 0
                    assert audit["online_steps"] > 0
                    assert audit["direct_fhe_steps"] == 0
                else:
                    assert audit["correlation_count"] == 0
                    assert audit["online_steps"] > 0
                    assert audit["direct_fhe_steps"] == audit["online_steps"]
                    assert audit["direct_fhe_upload_bytes"] > 0
                assert len(response.output_text) == 1
                assert response.usage is not None and response.usage.output_tokens == 1
                results[mode] = response.output_text
            remote_audit = b"\n".join(payload for _, payload in gateway.audit)
            assert canary.encode() not in remote_audit
        finally:
            gateway.close()
            if preparation is not None:
                preparation.close()
    assert set(results) == {"public", "proprietary"}
    assert results["public"] == results["proprietary"]
    assert len(results["public"]) == 1

@pytest.mark.parametrize(
    ("mode", "expected_engine"),
    [
        ("public", "masked-transformer-w4a4"),
        ("proprietary", "guarded-transformer-proprietary"),
    ],
)
def test_server_cli_mode_selects_engine_and_hf_model(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    expected_engine: str,
):
    import sys
    from pllm.runtime import cli

    root = create_tiny_gemma4_checkpoint(tmp_path / mode, num_hidden_layers=1)
    captured = {}

    def fake_run(app, **kwargs):
        captured["app"] = app
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "pllm serve",
        "--mode", mode,
        "--api-key", "test",
        "--provider-push-api-key", "push-test",
        "--model", str(root),
        "--model-id", f"tiny-{mode}",
        "--local-files-only",
    ])
    cli.server_main()
    app = captured["app"]
    assert app.state.config.privacy_mode == mode
    assert expected_engine in app.state.engines
    assert app.state.config.engine_models[0]["engine"] == expected_engine
    assert app.state.config.engine_models[0]["model_id"] == f"tiny-{mode}"
