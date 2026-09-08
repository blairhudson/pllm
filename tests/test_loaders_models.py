from __future__ import annotations

import json
import struct
from pathlib import Path

import httpx
import pytest

from pllm.runtime.loaders import ModelLoadError, load_gguf, load_hf_directory, load_mlx_directory, load_ollama_model
from pllm.runtime.models import ModelManifest, transformer_stage_plan


def test_hf_and_mlx_manifest_loading(tmp_path: Path):
    config = {
        "name_or_path": "gemma-test",
        "architectures": ["GemmaForCausalLM"],
        "hidden_size": 256,
        "intermediate_size": 768,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "vocab_size": 1024,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
        "quantization_config": {"quant_method": "awq", "bits": 4},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"}))
    (tmp_path / "model.safetensors").write_bytes(b"")

    hf = load_hf_directory(tmp_path)
    mlx = load_mlx_directory(tmp_path, model_id="mlx-model")
    assert hf.id == "gemma-test"
    assert hf.architecture == "GemmaForCausalLM"
    assert hf.tied_embeddings is True
    assert hf.quantization["bits"] == 4
    assert len(hf.stages) == 4 * 4 + 1
    assert hf.stages[0].fused_from == ("q_proj", "k_proj", "v_proj")
    assert mlx.source_format == "mlx-lm"
    assert mlx.metadata["mlx_weight_files"] == ["model.safetensors"]


def _s(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf_value(value):
    if isinstance(value, str):
        return 8, _s(value)
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, int):
        return 10, struct.pack("<Q", value)
    if isinstance(value, list):
        body = struct.pack("<IQ", 8, len(value)) + b"".join(_s(str(item)) for item in value)
        return 9, body
    raise TypeError(value)


def _write_gguf(path: Path) -> None:
    metadata = {
        "general.architecture": "llama",
        "general.name": "tiny-gguf",
        "general.file_type": 2,
        "general.tie_word_embeddings": True,
        "llama.embedding_length": 128,
        "llama.feed_forward_length": 384,
        "llama.block_count": 3,
        "llama.attention.head_count": 4,
        "llama.attention.head_count_kv": 2,
        "llama.context_length": 4096,
        "tokenizer.ggml.tokens": ["a", "b", "c", "d"],
    }
    payload = bytearray(b"GGUF")
    payload += struct.pack("<IQQ", 3, 0, len(metadata))
    for key, value in metadata.items():
        kind, encoded = _gguf_value(value)
        payload += _s(key) + struct.pack("<I", kind) + encoded
    path.write_bytes(payload)


def test_minimal_gguf_manifest_loading(tmp_path: Path):
    path = tmp_path / "tiny.Q4_K_M.gguf"
    _write_gguf(path)
    manifest = load_gguf(path)
    assert manifest.architecture == "llama"
    assert manifest.hidden_size == 128
    assert manifest.vocab_size == 4
    assert manifest.context_length == 4096
    assert manifest.metadata["gguf_version"] == 3
    assert manifest.source_format == "gguf"


def test_invalid_gguf_rejected(tmp_path: Path):
    path = tmp_path / "bad.gguf"
    path.write_bytes(b"nope")
    with pytest.raises(ModelLoadError):
        load_gguf(path)


@pytest.mark.asyncio
async def test_ollama_model_inspection():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/show"
        return httpx.Response(200, json={
            "details": {"family": "llama", "quantization_level": "Q4_K_M"},
            "model_info": {
                "general.architecture": "llama",
                "llama.embedding_length": 256,
                "llama.feed_forward_length": 768,
                "llama.block_count": 6,
                "llama.attention.head_count": 8,
                "llama.attention.head_count_kv": 2,
                "llama.context_length": 8192,
                "tokenizer.ggml.tokens": ["a"] * 32,
            },
            "template": "{{ .Prompt }}",
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        manifest = await load_ollama_model("http://ollama", "tiny", client=client)
        assert manifest.id == "tiny"
        assert manifest.source_format == "ollama"
        assert manifest.hidden_size == 256
        assert manifest.vocab_size == 32
        assert manifest.quantization["level"] == "Q4_K_M"
    finally:
        await client.aclose()


def test_manifest_roundtrip_and_fingerprint(tmp_path: Path):
    manifest = ModelManifest(
        id="m", architecture="llama", source_format="hf", source="/x",
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, context_length=2048,
        stages=transformer_stage_plan(
            hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=100,
        ),
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    restored = ModelManifest.load(path)
    assert restored.fingerprint == manifest.fingerprint
    assert restored.stages[-1].op == "lm_head"
    assert len(restored.stages) == 9


def test_stage_plan_unfused_counts():
    stages = transformer_stage_plan(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=100,
        fuse_qkv=False, fuse_gate_up=False, include_lm_head=False,
    )
    assert len(stages) == 2 * 7
    assert not any(stage.fused_from for stage in stages)


def test_manifest_fingerprint_excludes_creation_time():
    first = ModelManifest(
        id="stable", architecture="llama", source_format="hf", source="/x",
        vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, context_length=2048,
        created_at=1.0,
    )
    second = ModelManifest.from_dict({**first.to_dict(include_fingerprint=False), "created_at": 2.0})
    assert first.fingerprint == second.fingerprint
