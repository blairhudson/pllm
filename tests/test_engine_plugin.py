from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from pllm.runtime import GatewayConfig, create_app
from pllm.runtime.engine import EngineCapabilities
from pllm.runtime.protocol import encode_length_prefixed, iter_length_prefixed


class FakeEngine:
    capabilities = EngineCapabilities(
        name="fake",
        model_sources=("huggingface", "gguf", "mlx-lm"),
        protocols=("opaque-test",),
        online_fhe=True,
        preprocessed=True,
        continuous_batching=True,
    )

    def __init__(self):
        self.loaded = {}

    async def load(self, manifest):
        self.loaded[manifest.id] = manifest

    async def unload(self, model_id):
        self.loaded.pop(model_id, None)

    async def execute_stage(self, model_id, stage, payloads):
        assert model_id in self.loaded
        return [payload[::-1] for payload in payloads]


def test_engine_load_execute_unload_contract(tmp_path: Path):
    config = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "max_position_embeddings": 1024,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    engine = FakeEngine()
    app = create_app(GatewayConfig(api_keys=("x",), allow_insecure_local_correlations=True), engines={"fake": engine})
    headers = {"Authorization": "Bearer x"}
    with TestClient(app) as client:
        engines = client.get("/v1/runtime/engines", headers=headers).json()
        assert engines["data"][0]["id"] == "fake"
        loaded = client.post("/v1/runtime/models/load", headers=headers, json={
            "engine": "fake", "kind": "huggingface", "path": str(tmp_path), "model_id": "org/tiny"
        })
        assert loaded.status_code == 200
        assert loaded.json()["status"] == "ready"
        listed = client.get("/v1/models", headers=headers).json()["data"]
        loaded_model = next(row for row in listed if row["id"] == "org/tiny")
        assert loaded_model["runtime"]["privacy_mode"] == "private_engine"
        assert loaded_model["runtime"]["engine"] == "fake"
        assert client.get("/v1/runtime/models/org/tiny", headers=headers).json()["id"] == "org/tiny"
        stage = loaded.json()["stages"][0]["id"]
        raw = encode_length_prefixed([b"abc", b"def"])
        result = client.post(
            f"/v1/runtime/engines/fake/models/org/tiny/stages/{stage}",
            headers={**headers, "Content-Type": "application/octet-stream"},
            content=raw,
        )
        assert result.status_code == 200
        assert list(iter_length_prefixed(result.content)) == [b"cba", b"fed"]
        assert client.delete("/v1/runtime/models/org/tiny", headers=headers).json()["status"] == "unloaded"
        assert "org/tiny" not in engine.loaded


def test_sdk_runtime_extension_loads_and_executes_engine_stage(tmp_path: Path):
    from conftest import start_gateway
    from pllm import Model
    from pllm.runtime import OpenAI

    config = {
        "model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "max_position_embeddings": 1024,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    engine = FakeEngine()
    gateway = start_gateway(engines={"fake": engine})
    try:
        with OpenAI(
            base_url=gateway.base_url,
            api_key=gateway.api_key,
            correlation_mode="local-test",
        ) as client:
            assert client.runtime.engines()["data"][0]["id"] == "fake"
            loaded = client.runtime.load_model(
                Model.path(str(tmp_path), model_id="tiny-sdk"),
                engine="fake",
            )
            stage = loaded["stages"][0]["id"]
            outputs = client.runtime.execute_stage(
                engine="fake", model="tiny-sdk", stage=stage, payloads=[b"alpha", b"beta"]
            )
            assert outputs == [b"ahpla", b"ateb"]
            assert client.runtime.unload_model("tiny-sdk")["status"] == "unloaded"
    finally:
        gateway.close()
