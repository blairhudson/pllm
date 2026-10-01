from __future__ import annotations

import asyncio
import itertools
import json
from unittest.mock import patch

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.decision_refinement_reference import refine_linear_choice
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine


def test_shared_tail_error_cancels_without_resolving_features():
    w = np.array([[5, 100, 100], [1, 100, 100]], dtype=np.int8)
    for tail in itertools.product((-127, -1, 0, 1, 127), repeat=2):
        x = np.array([10, *tail], dtype=np.int8)
        result = refine_linear_choice(w, x, np.ones(2, np.float32), 1.0, block=1)
        assert result["certified_before_full_head"]
        assert result["resolved_features"] == 1
        expected = np.argmax((w.astype(np.int64) @ x.astype(np.int64)).astype(np.float32))
        assert result["selected"] == expected


def test_ambiguous_prefix_refines_and_ties_use_actual_argmax():
    w = np.array([[2, -127, 5], [1, 127, 5]], dtype=np.int8)
    for x in itertools.product((-2, -1, 0, 1, 2), repeat=3):
        values = np.array(x, dtype=np.int8)
        scales = np.array([0.125, 0.25], np.float32)
        result = refine_linear_choice(w, values, scales, 0.5, block=2)
        expected = (
            (w.astype(np.int64) @ values.astype(np.int64)).astype(np.float32)
            * np.float32(0.5)
            * scales
        )
        assert result["selected"] == int(np.argmax(expected))
    tied = refine_linear_choice(
        np.ones((2, 3), np.int8), np.ones(3, np.int8), np.ones(2, np.float32), 1.0, block=2
    )
    assert tied["selected"] == 0
    assert tied["resolved_features"] == 3


@pytest.mark.parametrize("scale", [0, -1, np.nan, np.inf, 1e-300, 65537])
def test_certificate_rejects_invalid_scale(scale):
    with pytest.raises(ValueError):
        refine_linear_choice(
            np.ones((2, 3), np.int8), np.ones(3, np.int8), np.ones(2, np.float32), scale
        )


def test_float32_extremes_and_subnormals_match_independent_head():
    rng = np.random.default_rng(902)
    for scale in [np.nextafter(np.float32(0), np.float32(1)), 0.01, 2.0, 65536.0]:
        for _ in range(20):
            w = rng.integers(-128, 128, (7, 9), dtype=np.int8)
            x = rng.integers(-127, 128, 9, dtype=np.int8)
            s = rng.uniform(0.001, 1, 7).astype(np.float32)
            result = refine_linear_choice(w, x, s, float(scale), block=4)
            expected = (
                (w.astype(np.int64) @ x.astype(np.int64)).astype(np.float32) * np.float32(scale) * s
            )
            assert result["selected"] == int(np.argmax(expected))


def test_complete_tiny_compiled_prefill_decode_heads_keep_exact_state(tmp_path):
    root = create_tiny_llama_checkpoint(tmp_path / "model", model_type="qwen2", with_qkv_bias=True)
    manifest = load_hf_directory(root, model_id="certificate-tiny")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(manifest.id))
    plan = lower_model(
        json.loads((root / "config.json").read_text()),
        batch=1,
        max_input_tokens=4,
        max_new_tokens=2,
    )
    composition = MaskedLinearCpu(
        Model(manifest.id), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8)
    )
    compiled = compile_runtime_model(plan, bundle, composition=composition)

    def remote(key, value):
        stage = engine.models[manifest.id].stages[key]
        q = quantize_activation_per_row(value, bits=8)
        result = dequantize_matmul(
            stage.compiled_weight.clear(q.values),
            q.scales,
            stage.weight.scales,
            output_shape=q.original_shape[:-1] + (stage.spec.out_features,),
        )
        return result if stage.bias is None else result + stage.bias

    checked = []
    original = ClientBundle.local_linear

    def certified_head(self, key, value):
        expected = original(self, key, value)
        if key == "lm_head":
            q = quantize_activation_per_row(value, bits=8)
            head = self.stages[key]
            result = refine_linear_choice(
                head.client_weight,
                q.values[-1],
                head.client_weight_scales,
                float(q.scales[-1]),
                block=3,
            )
            checked.append(result["selected"])
            assert result["selected"] == int(np.argmax(expected[-1]))
        return expected

    with patch.object(ClientBundle, "local_linear", certified_head):
        session = compiled.session(remote)
        session.prefill_ids([1, 2])
        first = session.select_next()
        session.decode_selected()
        second = session.select_next()
    assert checked == [first, second]
    assert session.complete and session.position == 3
