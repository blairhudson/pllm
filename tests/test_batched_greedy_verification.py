"""Compiled all-logit/KV parity and charged rejected work, not a serving test."""

import json

import numpy as np
import pytest

from pllm import Model, lower_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.greedy_verification_reference import verify_batched_greedy
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import TransformerClientError
from test_decoder_continuation import bound_checkpoint, runtime_for


@pytest.fixture(params=["qwen2", "qwen3", "llama"])
def fixture(tmp_path, request):
    root = create_tiny_llama_checkpoint(
        tmp_path / "model", model_type=request.param, seed=941,
        hidden_size=48, intermediate_size=96, head_dim=12, num_hidden_layers=2,
        with_qkv_bias=request.param == "qwen2", qk_norm=request.param == "qwen3",
        tie_word_embeddings=False,
    )
    engine, original, _, _ = bound_checkpoint(root)
    plan = lower_model(json.loads((root / "config.json").read_bytes()),
                       batch=1, max_input_tokens=64, max_new_tokens=16)
    pipeline = MaskedLinearCpu(Model(original._bundle.model_id),
                              quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    compiled = compile_runtime_model(plan, original._bundle, composition=pipeline)
    return engine, compiled, plan.continuation_schedule(pipeline)


@pytest.mark.parametrize("width", [2, 4, 8, 16])
@pytest.mark.parametrize("wrong", [None, 0, 1])
def test_every_target_logit_and_kv_bit_match_serial_after_rejections(fixture, width, wrong):
    engine, compiled, contract = fixture
    prompt = [11, 102, 54, 9, 200]
    serial, serial_remote = runtime_for(compiled, engine)
    _, logits, _ = serial.prepare_ids(prompt)
    expected, golden = [], []
    for index in range(16):
        golden.append(logits.copy())
        token = int(np.argmax(logits))
        expected.append(token)
        if index < 15:
            logits = serial.decode_step(token, serial.caches)[0]

    def propose(history, cap):
        offset = len(history) - len(prompt)
        offered = expected[offset:offset + cap].copy()
        if wrong is not None and wrong < len(offered):
            offered[wrong] = (offered[wrong] + 1) % compiled._bundle.cfg["vocab_size"]
        return offered

    candidate, remote = runtime_for(compiled, engine)
    checked = []
    result = verify_batched_greedy(candidate, contract, prompt, propose,
                                  outputs=16, width=width, on_output=checked.append)
    assert result["tokens"] == expected
    np.testing.assert_array_equal(np.asarray(checked).view(np.uint32), np.asarray(golden).view(np.uint32))
    assert candidate.position == serial.position
    for actual, reference in zip(candidate.caches, serial.caches, strict=True):
        assert actual.length == reference.length
        np.testing.assert_array_equal(actual.key[:actual.length].view(np.uint32),
                                      reference.key[:reference.length].view(np.uint32))
        np.testing.assert_array_equal(actual.value[:actual.length].view(np.uint32),
                                      reference.value[:reference.length].view(np.uint32))
        assert not np.any(actual.key[actual.length:result["maximum_queried_position"]])
        assert not np.any(actual.value[actual.length:result["maximum_queried_position"]])
    # Tiny fixture has two layers and four grouped remote projections per layer.
    stage_count = 8
    assert len(remote.calls) == stage_count * (1 + result["target_batches"])
    attempted_rows = sum(shape[0] for _, shape in remote.calls) // stage_count - len(prompt)
    assert attempted_rows == result["target_decode_rows"] == 15 + result["rejected_target_rows"]
    if wrong is None:
        assert result["draft_accepted"] == result["draft_offered"]
        assert len(remote.calls) < len(serial_remote.calls)
    else:
        assert result["rejected_target_rows"] > 0


def test_failure_retires_reference_state_without_rewinding_calls(fixture):
    engine, compiled, contract = fixture
    runtime, remote = runtime_for(compiled, engine)

    def fail(history, cap):
        remote.inject = "nonfinite"
        return [1] * cap

    with pytest.raises(TransformerClientError, match="nonfinite"):
        verify_batched_greedy(runtime, contract, [1, 2], fail, outputs=8, width=4)
    assert len(remote.calls) > 8
    assert runtime._continuation_failed
    assert runtime.position == 0
    assert all(cache.key is None and cache.value is None for cache in runtime.caches)
    assert "_local" not in runtime.__dict__
