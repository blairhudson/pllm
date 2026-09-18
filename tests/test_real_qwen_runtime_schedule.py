from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

MODEL_PATH = os.environ.get("PLLM_REAL_QWEN_PATH")
pytestmark = [
    pytest.mark.rust,
    pytest.mark.slow,
    pytest.mark.skipif(not MODEL_PATH, reason="PLLM_REAL_QWEN_PATH is not configured"),
]


def test_real_qwen_checkpoint_binds_and_executes_complete_schedule() -> None:
    root = Path(MODEL_PATH).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    model_id = "Qwen/Qwen2.5-0.5B-Instruct@runtime-schedule-test"
    manifest = load_hf_directory(root, model_id=model_id)
    engine = MaskedTransformerEngine(threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=16, max_new_tokens=2)
    schedule = plan.runtime_schedule()
    compiled = compile_runtime_model(plan, bundle)
    model = engine.models[model_id]

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = model.stages[stage_id]
        quantized = quantize_activation_per_row(
            activation, bits=stage.spec.activation_bits
        )
        integer = stage.compiled_weight.clear(quantized.values)
        output = dequantize_matmul(
            integer,
            quantized.scales,
            stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            output = output + stage.bias
        return np.ascontiguousarray(output, dtype=np.float32)

    runtime = compiled.runtime(remote)
    prompt_ids = runtime.encode_prompt("Hello")[-16:]
    session = compiled.session(remote)
    logits = session.prefill_ids(prompt_ids)
    first = session.select_next()
    next_logits = session.decode_selected()
    second = session.select_next()

    assert first == int(np.argmax(logits))
    assert second == int(np.argmax(next_logits))
    assert session.complete is True
    assert session.completeness_scope == "whole_decoder_runtime"
    assert session.status == "exhausted"
    assert session.position == len(prompt_ids) + 1
    assert plan.coverage("baseline.masked_linear_cpu").complete is True
    assert plan.coverage("research.single_evaluator").complete is False
    assert schedule.complete is True
    assert schedule.protected_execution is False
    assert schedule.digest == compiled.runtime_schedule_digest
    assert len(compiled.stage_bindings) == 98
    assert next_logits.shape == (config["vocab_size"],)
    assert np.all(np.isfinite(next_logits))
    assert 0 <= first < config["vocab_size"]
    assert 0 <= second < config["vocab_size"]
    assert runtime.bundle is bundle
    assert (
        bundle.stages["token_lookup"].client_weight
        is bundle.stages["lm_head"].client_weight
    )
    assert (
        bundle.stages["token_lookup"].client_weight_scales
        is bundle.stages["lm_head"].client_weight_scales
    )
