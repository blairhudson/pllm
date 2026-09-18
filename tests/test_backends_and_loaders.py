from __future__ import annotations

import json
import struct
from pathlib import Path

import httpx
import pytest

from pllm.runtime.backends.adapters import LlamaCppAdapter, MLXLMAdapter, OllamaAdapter, VLLMAdapter
from pllm.runtime.backends.registry import BackendRegistry
from pllm.runtime.loaders import GGUFReader, load_gguf, load_hf_directory, load_mlx_directory, load_ollama_model
from pllm.runtime.models import ModelManifest, transformer_stage_plan


def _sse(*events: dict) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events
    ) + b"data: [DONE]\n\n"


def test_vllm_native_responses_and_models():
    seen = []
    async def handler(request: httpx.Request):
        seen.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "v-model", "owned_by": "vllm"}]})
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, content=_sse({"type": "response.created", "response": {"id": "r"}}, {"type": "response.completed", "response": {"id": "r", "object": "response", "model": "v-model", "status": "completed", "output": []}}), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "r", "object": "response", "model": "v-model", "status": "completed", "output": []})
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = VLLMAdapter(base_url="http://upstream", client=client)
    async def run():
        models = await adapter.list_models()
        assert models[0].id == "v-model"
        response = await adapter.create_response({"model": "v-model", "input": "x"})
        assert response["id"] == "r"
        events = [event async for event in adapter.stream_response({"model": "v-model", "input": "x"})]
        assert events[-1]["type"] == "response.completed"
        await client.aclose()
    import asyncio; asyncio.run(run())


def test_ollama_strips_unsupported_server_state_fields():
    bodies = []
    async def handler(request: httpx.Request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen"}]})
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "r", "object": "response", "model": "qwen", "status": "completed", "output": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = OllamaAdapter(base_url="http://ollama", client=client)
    async def run():
        await adapter.create_response({"model": "qwen", "input": "x", "previous_response_id": "secret-local", "conversation": "also-local"})
        assert "previous_response_id" not in bodies[0]
        assert "conversation" not in bodies[0]
        await client.aclose()
    import asyncio; asyncio.run(run())


def test_mlx_chat_translation_nonstreaming_and_streaming():
    requests = []
    async def handler(request: httpx.Request):
        requests.append((request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "mlx"}]})
        body = requests[-1][1]
        if body.get("stream"):
            payload = (
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
                b'data: [DONE]\n\n'
            )
            return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"id": "c", "created": 1, "model": "mlx", "choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = MLXLMAdapter(base_url="http://mlx", client=client)
    async def run():
        value = await adapter.create_response({"model": "mlx", "input": "hi"})
        assert value["output"][0]["content"][0]["text"] == "hello"
        events = [e async for e in adapter.stream_response({"model": "mlx", "input": "hi"})]
        assert "".join(e.get("delta", "") for e in events) == "hello"
        assert events[-1]["type"] == "response.completed"
        assert requests[0][0] == "/v1/chat/completions"
        await client.aclose()
    import asyncio; asyncio.run(run())


def test_llama_cpp_falls_back_to_chat_when_responses_missing():
    async def handler(request: httpx.Request):
        if request.url.path == "/v1/responses":
            return httpx.Response(404)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"id": "c", "created": 1, "choices": [{"message": {"content": "ok"}}]})
        return httpx.Response(200, json={"data": [{"id": "gguf"}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = LlamaCppAdapter(base_url="http://llama", client=client)
    async def run():
        value = await adapter.create_response({"model": "gguf", "input": "x"})
        assert value["output"][0]["content"][0]["text"] == "ok"
        await client.aclose()
    import asyncio; asyncio.run(run())


def test_backend_registry_routes_colliding_model_ids():
    class Fake:
        capabilities = type("C", (), {"to_dict": lambda self: {}})()
        async def list_models(self):
            from pllm.runtime.backends.base import BackendModel
            return [BackendModel("same", "x", "fake", "trusted_backend")]
    registry = BackendRegistry(); registry.add("a", Fake()); registry.add("b", Fake())
    import asyncio
    rows = asyncio.run(registry.refresh())
    assert [r.id for r in rows] == ["same", "b/same"]
    assert registry.adapter_for("same") is registry.adapters["a"]
    assert registry.adapter_for("b/same") is registry.adapters["b"]


def test_hf_and_mlx_directory_manifest(tmp_path: Path):
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "vocab_size": 32000,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"")
    hf = load_hf_directory(tmp_path, model_id="hf")
    mlx = load_mlx_directory(tmp_path, model_id="mlx")
    assert hf.hidden_size == 4096 and hf.num_key_value_heads == 8
    assert len(hf.stages) == 32 * 4 + 1
    assert mlx.source_format == "mlx-lm"
    assert mlx.metadata["mlx_weight_files"] == ["model-00001-of-00002.safetensors"]
    saved = tmp_path / "manifest.json"; hf.save(saved)
    assert ModelManifest.load(saved).fingerprint == hf.fingerprint


def _write_string(stream, value: str):
    data = value.encode(); stream.write(struct.pack("<Q", len(data))); stream.write(data)


def _write_gguf(path: Path):
    metadata = [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "tiny"),
        ("llama.embedding_length", 5, 64),
        ("llama.feed_forward_length", 5, 128),
        ("llama.block_count", 5, 2),
        ("llama.attention.head_count", 5, 4),
        ("llama.attention.head_count_kv", 5, 2),
        ("llama.context_length", 5, 1024),
        ("llama.vocab_size", 5, 256),
    ]
    with path.open("wb") as f:
        f.write(b"GGUF"); f.write(struct.pack("<IQQ", 3, 0, len(metadata)))
        for key, kind, value in metadata:
            _write_string(f, key); f.write(struct.pack("<I", kind))
            if kind == 8: _write_string(f, value)
            elif kind == 5: f.write(struct.pack("<i", value))


def test_gguf_manifest_reader(tmp_path: Path):
    path = tmp_path / "tiny.Q4_K_M.gguf"; _write_gguf(path)
    manifest = load_gguf(path)
    assert manifest.source_format == "gguf"
    assert manifest.hidden_size == 64
    assert manifest.num_hidden_layers == 2
    assert manifest.vocab_size == 256


def test_ollama_manifest_loader():
    async def handler(request: httpx.Request):
        return httpx.Response(200, json={
            "details": {"family": "llama", "quantization_level": "Q4_K_M"},
            "model_info": {
                "general.architecture": "llama",
                "llama.embedding_length": 128,
                "llama.feed_forward_length": 256,
                "llama.block_count": 4,
                "llama.attention.head_count": 4,
                "llama.attention.head_count_kv": 2,
                "llama.context_length": 2048,
                "llama.vocab_size": 1024,
            },
        })
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    import asyncio
    manifest = asyncio.run(load_ollama_model("http://ollama", "tiny", client=client))
    asyncio.run(client.aclose())
    assert manifest.id == "tiny" and manifest.hidden_size == 128
    assert manifest.quantization["level"] == "Q4_K_M"


def test_transformer_stage_plan_fuses_qkv_and_gate_up():
    stages = transformer_stage_plan(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=256)
    assert [stage.id for stage in stages[:4]] == [
        "layers.0.self_attn.qkv_proj", "layers.0.self_attn.o_proj", "layers.0.mlp.gate_up_proj", "layers.0.mlp.down_proj"
    ]
    assert [stage.role for stage in stages[:4]] == [
        "qkv_projection", "attention_output", "mlp_gate_up", "mlp_down"
    ]
    assert all(stage.layer_index == 0 for stage in stages[:4])
    assert all(stage.layer_index == 1 for stage in stages[4:8])
    assert stages[-1].id == "lm_head"
    assert stages[-1].role == "lm_head"
    assert stages[-1].layer_index is None
