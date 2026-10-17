"""Complete generated decoder oracles for the bounded GKM composition.

The clear oracle implements the model equations directly. It calls neither the
semantic executor nor native GKM arithmetic. Float32 reduction layouts are part
of the declared local profile; each delegated product uses Python integer sums.
"""

import asyncio
import copy
import json
import math

import numpy as np
import pytest

from pllm import _native
from pllm.runtime import delegated_decoder as decoder


def config(family="qwen2", tied=True):
    result = {
        "model_type": family, "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "vocab_size": 32, "max_position_embeddings": 32, "hidden_act": "silu",
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": tied,
    }
    if family == "qwen3":
        result.update(head_dim=4, attention_bias=False, attention_dropout=0.0,
                      rope_scaling=None, sliding_window=None, use_sliding_window=False,
                      use_cache=True, max_window_layers=2,
                      layer_types=["full_attention", "full_attention"])
    return result


def fixture(cfg, rows=3, decode=2):
    contract = json.loads(decoder.decoder_contract(cfg, input_rows=rows, decode_steps=decode))
    rng = np.random.default_rng(5431)
    arrays = {}
    for name, shape in contract["source_shapes"].items():
        value = rng.normal(0, 0.15, shape).astype("<f4")
        if "norm" in name:
            value += np.float32(1)
        arrays[name] = value
    arrays[contract["table"]][0, 0] = np.float32(-0.0)
    return contract, arrays, {key: value.tobytes() for key, value in arrays.items()}


def clear_oracle(cfg, tensors, tokens, decode_steps):
    def linear(value, weight, bias=None):
        encoded = [[round(float(w) * 2048) for w in row] for row in tensors[weight]]
        output = np.empty((len(value), len(encoded)), dtype=np.float32)
        for i, row in enumerate(value):
            inputs = [round(float(x) * 2048) for x in row]
            for j, weights in enumerate(encoded):
                integer = sum(x*w for x, w in zip(inputs, weights, strict=True))
                assert abs(integer) < 2**53  # exact int->f64 before this fixture's f32 cast
                output[i, j] = np.float32(integer) / np.float32(4194304)
        return output if bias is None else output + tensors[bias]

    def norm(value, weight):
        return value / np.sqrt(np.mean(value*value, axis=-1, keepdims=True) + cfg["rms_norm_eps"]) * tensors[weight]

    def rotary(value, positions):
        width = value.shape[-1]
        frequencies = 1 / (cfg["rope_theta"] ** (np.arange(0, width, 2, dtype=np.float32) / width))
        angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, None]
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, None]
        half = width // 2
        return value*cos + np.concatenate([-value[..., half:], value[..., :half]], axis=-1)*sin

    caches = [None] * cfg["num_hidden_layers"]
    logits, selections, executed = [], [], list(tokens)
    position = 0
    ids = tokens
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    width, group = cfg["hidden_size"] // heads, heads // kv_heads
    for step in range(decode_steps + 1):
        positions = np.arange(position, position+len(ids), dtype=np.int64)
        hidden = tensors["model.embed_tokens.weight"][list(ids)].copy()
        for layer in range(cfg["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            residual = hidden
            normalized = norm(hidden, prefix+"input_layernorm.weight")
            qkv = []
            for name, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
                base = prefix+f"self_attn.{name}_proj"
                value = linear(normalized, base+".weight", base+".bias" if cfg["model_type"] == "qwen2" else None)
                value = value.reshape(1, len(ids), count, width).transpose(0, 2, 1, 3)
                if cfg["model_type"] == "qwen3" and name != "v":
                    value = norm(value, prefix+f"self_attn.{name}_norm.weight")
                qkv.append(rotary(value, positions) if name != "v" else value)
            query, key, value = qkv
            current = (key[0].transpose(1, 0, 2), value[0].transpose(1, 0, 2))
            old = caches[layer]
            key, value = current if old is None else tuple(np.concatenate((a, b)) for a, b in zip(old, current, strict=True))
            caches[layer] = (key, value)
            q = query.reshape(1, kv_heads, group, len(ids), width)
            scores = np.empty((1, kv_heads, group, len(ids), len(key)), dtype=np.float32)
            for row in range(len(ids)):
                scores[0, :, :, row] = np.einsum("kgd,tkd->kgt", q[0, :, :, row], key)
            scores = scores.reshape(1, heads, len(ids), len(key)) * float(1/math.sqrt(width))
            scores = np.where(np.arange(len(key))[None, :] <= positions[:, None], scores, np.float32(-np.inf))
            probability = np.exp(scores-scores.max(axis=-1, keepdims=True)).astype(np.float32)
            probability = probability / probability.sum(axis=-1, keepdims=True)
            probability = probability.reshape(1, kv_heads, group, len(ids), len(key))
            attended = np.empty((1, kv_heads, group, len(ids), width), dtype=np.float32)
            for row in range(len(ids)):
                attended[0, :, :, row] = np.einsum("kgt,tkd->kgd", probability[0, :, :, row], value)
            attended = attended.reshape(1, heads, len(ids), width).transpose(0, 2, 1, 3).reshape(len(ids), -1)
            hidden = residual + linear(attended, prefix+"self_attn.o_proj.weight")
            residual = hidden
            normalized = norm(hidden, prefix+"post_attention_layernorm.weight")
            gate = linear(normalized, prefix+"mlp.gate_proj.weight")
            up = linear(normalized, prefix+"mlp.up_proj.weight")
            activation = np.empty_like(gate)
            positive = gate >= 0
            activation[positive] = gate[positive] / (1 + np.exp(-gate[positive]))
            exp = np.exp(gate[~positive])
            activation[~positive] = gate[~positive] * exp / (1 + exp)
            hidden = residual + linear(activation*up, prefix+"mlp.down_proj.weight")
        head = "model.embed_tokens.weight" if cfg["tie_word_embeddings"] else "lm_head.weight"
        final = linear(norm(hidden, "model.norm.weight")[-1:], head).reshape(-1)
        logits.append(final.astype("<f4").tobytes())
        token = int(np.argmax(final))
        selections.append(token)
        position += len(ids)
        if step < decode_steps:
            ids = (token,)
            executed.append(token)
    kv = tuple((tuple(k.shape), k.astype("<f4").tobytes(), v.astype("<f4").tobytes()) for k, v in caches)
    return tuple(logits), kv, tuple(selections), tuple(executed), position


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_complete_prefill_decode_matches_independent_integer_float32_oracle(family, tied):
    cfg = config(family, tied)
    contract, arrays, source = fixture(cfg)
    expected = clear_oracle(cfg, arrays, (0, 4, 9), 2)
    receipts = []
    for seeded in (False, True):
        result = decoder.execute_gkm_reference(cfg, source, (0, 4, 9), decode_steps=2, seeded=seeded)
        assert (result.logits, result.kv, result.selected, result.executed, result.position) == expected
        receipt = json.loads(result.acceptance)
        assert receipt["steps"] == contract["window_steps"] == 48
        assert receipt["seeded"] is seeded
        receipts.append(receipt)
    assert receipts[0]["source"] == receipts[1]["source"]
    assert receipts[0]["contract"] == receipts[1]["contract"]
    assert receipts[0]["request"] != receipts[1]["request"]


def test_original_lookup_signed_zero_and_source_binding():
    cfg = config()
    contract, _, tensors = fixture(cfg, 1, 0)
    cfg_bytes = json.dumps(cfg).encode()
    session = _native._GkmDecoderSession(cfg_bytes, 1, 0, tensors)
    assert session.lookup(0) == tensors[contract["table"]][:contract["width"]*4]
    session.abort()
    first = decoder.execute_gkm_reference(cfg, tensors, (0,), decode_steps=0)
    changed = dict(tensors)
    name = next(name for name in tensors if "input_layernorm" in name)
    value = np.frombuffer(changed[name], dtype="<f4").copy()
    value[0] += np.float32(0.001)
    changed[name] = value.tobytes()
    second = decoder.execute_gkm_reference(cfg, changed, (0,), decode_steps=0)
    assert json.loads(first.acceptance)["source"] != json.loads(second.acceptance)["source"]


@pytest.mark.parametrize("call", [
    lambda session: session.lookup(True), lambda session: session.lookup("1"),
    lambda session: session.lookup(-1), lambda session: session.lookup(32),
    lambda session: session.linear(0, bytes(32)), lambda session: session.finish(),
    lambda session: session.linear(False, bytes(32)), lambda session: session.linear(0, bytearray(32)),
])
def test_invalid_native_call_burns_before_python_conversion(call):
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)
    session = _native._GkmDecoderSession(json.dumps(cfg).encode(), 1, 0, tensors, seeded=True)
    with pytest.raises((ValueError, TypeError, OverflowError)):
        call(session)
    with pytest.raises(ValueError, match="retired"):
        session.lookup(0)
    with pytest.raises(ValueError, match="retired"):
        session.finish()


@pytest.mark.parametrize("error", [ValueError, KeyboardInterrupt, asyncio.CancelledError])
def test_local_failure_retires_full_kv_and_native_window(monkeypatch, error):
    cfg = config()
    _, _, tensors = fixture(cfg)
    requests = []
    original_init, original_local = decoder._Request.__init__, decoder._Request._local

    def init(self, *args):
        original_init(self, *args)
        requests.append(self)

    def local(self, operation, *args):
        if operation["operator"] == "silu":
            assert any(cache.length for cache in self.caches)
            raise error("injected cancellation/failure")
        return original_local(self, operation, *args)

    monkeypatch.setattr(decoder._Request, "__init__", init)
    monkeypatch.setattr(decoder._Request, "_local", local)
    with pytest.raises(error):
        decoder.execute_gkm_reference(cfg, tensors, (0, 4, 9), decode_steps=2, seeded=True)
    assert len(requests) == 1
    request = requests[0]
    assert all(cache.length == 0 and cache.key is None and cache.value is None for cache in request.caches)
    assert request._coefficients == {}
    with pytest.raises(ValueError, match="retired"):
        request._native.finish()


def test_geometry_and_source_validation_precede_execution():
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)
    for key, value in [("vocab_size", 151936), ("num_hidden_layers", 24)]:
        large = {**cfg, key: value}
        with pytest.raises(ValueError):
            decoder.decoder_contract(large, input_rows=3, decode_steps=2)
    for tokens, steps in [((0,), True), ((True,), 0), ((32,), 0), ((), 0), ([1], 0)]:
        with pytest.raises(ValueError):
            decoder.execute_gkm_reference(cfg, tensors, tokens, decode_steps=steps)
    for bad in (b"short", bytes.fromhex("0000c07f")*8*32, np.full((32, 8), 16, dtype="<f4").tobytes()):
        changed = copy.copy(tensors)
        changed["model.embed_tokens.weight"] = bad
        with pytest.raises(ValueError):
            decoder.execute_gkm_reference(cfg, changed, (0,), decode_steps=0)
