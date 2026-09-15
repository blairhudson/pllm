from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest

from conftest import start_gateway, start_preparation
from pllm.runtime import OpenAI, ProprietaryProtocol
from pllm.runtime.blinded_engine import BlindedTransformerEngine
from pllm.runtime.client import _BFVStageClient
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.privacy import PrivacyMode
from pllm.runtime.quantization import quantize_activation_per_row
from pllm.runtime.stage_protocol import BlindedStageRequest, BlindedStageResponse
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.transformer_engine import TransformerEngineError

PYDEPS = os.environ.get("PLLM_TENSEAL_PATH", "")


def run(value):
    return asyncio.run(value)


def tiny_model(root: Path) -> Path:
    path = create_tiny_gemma4_checkpoint(
        root,
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config.update({"bos_token_id": 2, "eos_token_id": 3, "pad_token_id": 3})
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (path / "pllm_tokenizer.json").write_text(json.dumps({
        "type": "alphabet",
        "alphabet": " a",
        "vocab_size": 4,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
    }))
    return path


def test_mode_and_protocol_names():
    assert PrivacyMode.parse("proprietary") is PrivacyMode.PROPRIETARY
    assert ProprietaryProtocol.parse("fast") is ProprietaryProtocol.GUARDED
    assert ProprietaryProtocol.parse("direct") is ProprietaryProtocol.DIRECT


def test_local_blinded_stage_is_exact_and_single_use(tmp_path: Path):
    root = tiny_model(tmp_path / "local")
    manifest = load_hf_directory(root, model_id="tiny-blinded")
    engine = BlindedTransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    runtime = engine.models[manifest.id].stages[stage.id]
    activation = np.random.default_rng(11).normal(size=(2, stage.in_features)).astype(np.float32)
    qa = quantize_activation_per_row(activation, bits=4)
    rows = engine.create_local_blinded_correlations(
        manifest.id, stage.id, "owner-a", qa.rows, seed=7
    )
    masks = np.stack([row.mask for row in rows])
    blinded = np.stack([row.blinded_transformed_mask for row in rows])
    wr = engine.kernel(runtime.weight.values, masks, runtime.modulus)
    server_masks = np.stack([
        engine._derive_output_mask(
            manifest.id, "owner-a", stage.id, row.id,
            runtime.modulus, runtime.spec.out_features,
        )
        for row in rows
    ])
    assert not hasattr(engine._blinds[(manifest.id, "owner-a", stage.id, rows[0].id)], "output_mask")
    assert np.array_equal(blinded, (wr.astype(np.uint64) + server_masks) % runtime.modulus)
    assert not np.array_equal(blinded, wr)

    request = BlindedStageRequest(
        model=manifest.id,
        stage_id=stage.id,
        owner_id="owner-a",
        correlation_ids=tuple(row.id for row in rows),
        masked_input=((qa.values.astype(np.int64) - masks.astype(np.int64)) % runtime.modulus).astype(np.uint32),
        activation_scales=qa.scales,
        modulus=runtime.modulus,
        wire_bits=runtime.wire_bits,
        ring="prime",
    )
    response = BlindedStageResponse.unpack(
        run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
    )
    actual = (response.masked_output.astype(np.uint64) + blinded.astype(np.uint64)) % runtime.modulus
    actual = actual.astype(np.int64)
    actual = np.where(actual > runtime.modulus // 2, actual - runtime.modulus, actual)
    expected = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
    assert np.array_equal(actual, expected)
    assert engine.proprietary_inventory_stats() == {
        "available": 0,
        "generated": qa.rows,
        "consumed": qa.rows,
        "materialized_output_mask_bytes": 0,
        "derived_blind_records": 0,
    }
    with pytest.raises(TransformerEngineError, match="consumed"):
        run(engine.execute_stage(manifest.id, stage, [request.pack()]))


@pytest.mark.he
def test_real_bfv_blinded_preprocessing_and_online_stage(tmp_path: Path):
    root = tiny_model(tmp_path / "bfv")
    manifest = load_hf_directory(root, model_id="tiny-blinded-bfv")
    engine = BlindedTransformerEngine(threads=1, tenseal_path=PYDEPS)
    run(engine.load(manifest))
    stage = next(row for row in manifest.stages if row.id == "layers.0.self_attn.qkv_proj")
    runtime = engine.models[manifest.id].stages[stage.id]
    client = _BFVStageClient(plain_modulus=runtime.modulus, pydeps_path=PYDEPS)
    engine.register_bfv_context(manifest.id, "ctx", client.public_context)

    activation = np.random.default_rng(12).normal(size=(1, stage.in_features)).astype(np.float32)
    qa = quantize_activation_per_row(activation, bits=4)
    mask = np.random.default_rng(13).integers(
        0, runtime.modulus, size=stage.in_features, dtype=np.uint32
    )
    raw = engine.evaluate_bfv_blinded_correlation(
        manifest.id, stage.id, "owner-b", "ctx", client.encrypt(mask)
    )
    import msgpack
    envelope = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    correlation_id = str(envelope["correlation_id"])
    blinded = client.decrypt(raw, stage_id=stage.id)
    wr = engine.kernel(runtime.weight.values, mask[None, :], runtime.modulus)[0]
    assert not np.array_equal(blinded, wr)

    request = BlindedStageRequest(
        model=manifest.id,
        stage_id=stage.id,
        owner_id="owner-b",
        correlation_ids=(correlation_id,),
        masked_input=((qa.values.astype(np.int64) - mask[None, :].astype(np.int64)) % runtime.modulus).astype(np.uint32),
        activation_scales=qa.scales,
        modulus=runtime.modulus,
        wire_bits=runtime.wire_bits,
        ring="prime",
    )
    response = BlindedStageResponse.unpack(
        run(engine.execute_stage(manifest.id, stage, [request.pack()]))[0]
    )
    actual = (response.masked_output[0].astype(np.uint64) + blinded.astype(np.uint64)) % runtime.modulus
    actual = actual.astype(np.int64)
    actual = np.where(actual > runtime.modulus // 2, actual - runtime.modulus, actual)
    expected = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
    assert np.array_equal(actual, expected[0])


@pytest.mark.he
def test_public_and_fast_proprietary_responses_match(tmp_path: Path):
    from pllm.runtime.transformer_engine import MaskedTransformerEngine

    root = tiny_model(tmp_path / "api")
    canary = "PRIVATE-PLLM-CANARY-29f5"
    results: dict[str, str] = {}
    modes = (
        ("public", MaskedTransformerEngine(threads=1, tenseal_path=PYDEPS), "local-test"),
        ("proprietary", BlindedTransformerEngine(threads=1, tenseal_path=PYDEPS), "bfv"),
    )
    for mode, engine, correlation_mode in modes:
        gateway = start_gateway(
            bfv=correlation_mode == "bfv",
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
            with httpx.Client(base_url=gateway.base_url, timeout=180) as admin:
                loaded = admin.post(
                    "/v1/runtime/models/load",
                    headers={"Authorization": f"Bearer {gateway.api_key}"},
                    json={
                        "engine": engine.capabilities.name,
                        "kind": "huggingface",
                        "path": str(root),
                        "model_id": model_id,
                    },
                )
                assert loaded.status_code == 200, loaded.text
            with OpenAI(
                api_key=gateway.api_key,
                base_url=gateway.base_url,
                preparation_base_url=preparation.base_url if preparation else None,
                preparation_api_key=preparation.api_key if preparation else None,
                correlation_mode=correlation_mode,
                correlation_prefetch=1,
                tenseal_path=PYDEPS,
                timeout=180,
            ) as client:
                if mode == "public":
                    rows = client.prepared_rows_for_response(
                        canary, 1, model=model_id
                    )
                    client.preprocess(model_id, count=rows)
                response = client.responses.create(
                    model=model_id,
                    input=canary,
                    max_output_tokens=1,
                    temperature=0,
                )
                assert response.status == "completed"
                assert client.privacy_audit.plaintext_prompt_bytes_sent == 0
                assert client.privacy_audit.plaintext_token_ids_sent == 0
                if mode == "public":
                    assert client.privacy_audit.preparation_upload_bytes > 0
                else:
                    assert client.privacy_audit.correlation_count > 0
                results[mode] = response.output_text
            remote = b"\n".join(payload for _, payload in gateway.audit)
            assert canary.encode() not in remote
        finally:
            gateway.close()
            if preparation is not None:
                preparation.close()
    assert results["public"] == results["proprietary"]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], "guarded-transformer-proprietary"),
        (["--protocol", "direct"], "direct-bfv-transformer-proprietary"),
    ],
)
def test_proprietary_cli_default_and_direct_reference(tmp_path: Path, monkeypatch, extra, expected):
    from pllm.runtime import cli

    root = tiny_model(tmp_path / expected)
    captured = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: captured.update(app=app))
    monkeypatch.setattr(sys, "argv", [
        "pllm serve", "--mode", "proprietary", "--api-key", "test",
        "--model", str(root), "--model-id", "tiny", "--local-files-only", *extra,
    ])
    cli.server_main()
    assert expected in captured["app"].state.engines


def test_server_blinds_are_prf_derived_and_bounded():
    engine = BlindedTransformerEngine(threads=1)
    first = engine._derive_output_mask("m", "owner", "stage", "corr-1", 65521, 4096)
    repeat = engine._derive_output_mask("m", "owner", "stage", "corr-1", 65521, 4096)
    other = engine._derive_output_mask("m", "owner", "stage", "corr-2", 65521, 4096)
    assert np.array_equal(first, repeat)
    assert not np.array_equal(first, other)
    assert first.dtype == np.uint32
    assert int(first.max()) < 65521
