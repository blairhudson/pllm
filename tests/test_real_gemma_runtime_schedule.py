"""Opt-in official Gemma checkpoint execution and same-token quality probe."""

from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.metrics import measure_reference_agreement
from pllm.profiles import MaskedLinearCpu
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


@pytest.mark.skipif(
    not os.environ.get("PLLM_GEMMA4_E2B_PATH"), reason="set PLLM_GEMMA4_E2B_PATH"
)
def test_pinned_e2b_compiled_prefill_decode_and_reference_quality(request: pytest.FixtureRequest) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    root = Path(os.environ["PLLM_GEMMA4_E2B_PATH"])
    original_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    request.addfinalizer(lambda: torch.set_num_threads(original_threads))
    # Token cohorts are public, fixed and kept in process; do not archive logits.
    cohorts = ((9259,), (818, 5279, 529, 7001, 563))
    reference = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).eval()
    with torch.inference_mode():
        reference_logits = []
        for ids in cohorts:
            token_tensor = torch.tensor([ids], dtype=torch.long)
            reference_logits.append(
                reference(
                    input_ids=token_tensor,
                    attention_mask=torch.ones_like(token_tensor),
                    use_cache=False,
                ).logits[0, -1].float().numpy().copy()
            )
    del reference
    gc.collect()

    manifest = load_hf_directory(root, model_id="google/gemma-4-E2B-it@3e22461f")
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id))
    plan = lower_model((root / "config.json").read_bytes(), batch=1, max_input_tokens=5, max_new_tokens=2)
    compiled = compile_runtime_model(plan, bundle, composition=MaskedLinearCpu(Model(manifest.id)))
    assert len(compiled.stage_bindings) == 213
    stages = engine.models[manifest.id].stages

    def remote(stage_id: str, activation: np.ndarray) -> np.ndarray:
        stage = stages[stage_id]
        quantized = quantize_activation_per_row(activation, bits=stage.spec.activation_bits)
        integer = stage.compiled_weight.clear(quantized.values)
        result = dequantize_matmul(
            integer,
            quantized.scales,
            stage.weight.scales,
            output_shape=quantized.original_shape[:-1] + (stage.spec.out_features,),
        )
        if stage.bias is not None:
            result += stage.bias
        return np.ascontiguousarray(result, dtype=np.float32)

    measurements = []
    for ids, expected in zip(cohorts, reference_logits, strict=True):
        session = compiled.session(remote)
        logits = session.prefill_ids(ids)
        assert logits.shape == expected.shape == (262144,)
        measurements.append(measure_reference_agreement(logits, expected, top_k=5))
        session.select_next()
        decoded = session.decode_selected()
        assert decoded.shape == (262144,) and np.all(np.isfinite(decoded))
        session.finish()
        assert session.status == "exhausted"

    summary = {
        "samples": len(measurements),
        "top1_agreement": sum(row["top1_agreement"] for row in measurements) / len(measurements),
        "top5_recall": sum(row["top_k_recall"] for row in measurements) / len(measurements),
        "worst_abs_logit_error": max(row["max_abs_logit_error"] for row in measurements),
    }
    assert summary["top1_agreement"] == 1.0
    assert summary["top5_recall"] >= 0.6
    assert np.isfinite(summary["worst_abs_logit_error"])
    assert summary["worst_abs_logit_error"] < 4.0
    print(f"Gemma E2B compiled W8A8 local quality: {summary}")
