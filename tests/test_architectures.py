import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle, MaskedTransformerClientRuntime, RemoteLinear
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


def run(value):
    return asyncio.run(value)


def test_qwen2_biases_and_llama_semantics_are_preserved(tmp_path: Path):
    root = create_tiny_llama_checkpoint(tmp_path / "qwen")
    manifest = load_hf_directory(root, model_id="tiny-qwen")
    engine = MaskedTransformerEngine(threads=1)
    run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle("tiny-qwen"))

    qkv = bundle.stages["layers.0.self_attn.qkv_proj"]
    assert qkv.bias is not None
    assert qkv.bias.shape == (64,)
    assert bundle.cfg["model_family"] == "llama-compatible"
    assert bundle.cfg["block_style"] == "llama"
    assert bundle.cfg["embedding_multiplier"] == 1.0
    assert bundle.cfg["attention_scaling"] is None
    assert bundle.cfg["hidden_activation"] == "silu"

    class Provider:
        model_id = "tiny-qwen"

        def take_many(self, stage, count):
            return engine.create_local_correlations("tiny-qwen", stage.id, count)

    def exchange(stage_id, payloads):
        stage = next(row for row in manifest.stages if row.id == stage_id)
        return run(engine.execute_stage("tiny-qwen", stage, payloads))

    class ClearRemote:
        def __call__(self, stage_id, activation):
            stage = bundle.stages[stage_id]
            runtime = engine.models["tiny-qwen"].stages[stage_id]
            qa = quantize_activation_per_row(activation, bits=stage.activation_bits)
            integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
            output = dequantize_matmul(
                integer,
                qa.scales,
                runtime.weight.scales,
                output_shape=qa.original_shape[:-1] + (stage.out_features,),
            )
            return output if runtime.bias is None else output + runtime.bias

    masked = MaskedTransformerClientRuntime(
        bundle,
        RemoteLinear(bundle.stages, Provider(), exchange),
    )
    clear = MaskedTransformerClientRuntime(bundle, ClearRemote())
    _, masked_logits, _ = masked.prepare("bias parity")
    _, clear_logits, _ = clear.prepare("bias parity")
    assert np.array_equal(masked_logits, clear_logits)


def test_muse_glimmer_is_rejected_until_gated_attention_adapter_exists(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "muse")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["architectures"] = ["MuseGlimmerForCausalLM"]
    config["model_type"] = "muse_glimmer_text"
    config_path.write_text(json.dumps(config))
    manifest = load_hf_directory(root, model_id="muse-not-yet")
    engine = MaskedTransformerEngine(threads=1)
    with pytest.raises(TransformerEngineError, match="gated-attention"):
        run(engine.load(manifest))


def test_sparse_moe_is_rejected_without_route_private_adapter(tmp_path: Path):
    root = create_tiny_gemma4_checkpoint(tmp_path / "moe")
    config_path = root / "config.json"
    config = json.loads(config_path.read_text())
    config["enable_moe_block"] = True
    config["num_experts"] = 8
    config["top_k_experts"] = 2
    config_path.write_text(json.dumps(config))
    manifest = load_hf_directory(root, model_id="moe-not-yet")
    engine = MaskedTransformerEngine(threads=1)
    with pytest.raises(TransformerEngineError, match="route-private"):
        run(engine.load(manifest))
