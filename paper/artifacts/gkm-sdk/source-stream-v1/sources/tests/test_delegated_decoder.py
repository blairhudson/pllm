"""Complete generated decoder oracles for the bounded GKM composition.

The clear oracle implements the model equations directly. It calls neither the
semantic executor nor native GKM arithmetic. Float32 reduction layouts are part
of the declared local profile; each delegated product uses Python integer sums.
"""

import asyncio
from collections.abc import Mapping
import copy
import json
import math

import numpy as np
import pytest

from pllm import _native
from pllm.runtime import delegated_decoder as decoder


def config(family="qwen2", tied=True):
    result = {
        "model_type": family,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "vocab_size": 32,
        "max_position_embeddings": 32,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied,
    }
    if family == "qwen3":
        result.update(
            head_dim=4,
            attention_bias=False,
            attention_dropout=0.0,
            rope_scaling=None,
            sliding_window=None,
            use_sliding_window=False,
            use_cache=True,
            max_window_layers=2,
            layer_types=["full_attention", "full_attention"],
        )
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
                integer = sum(x * w for x, w in zip(inputs, weights, strict=True))
                assert abs(integer) < 2**53  # exact int->f64 before this fixture's f32 cast
                output[i, j] = np.float32(integer) / np.float32(4194304)
        return output if bias is None else output + tensors[bias]

    def norm(value, weight):
        return (
            value
            / np.sqrt(np.mean(value * value, axis=-1, keepdims=True) + cfg["rms_norm_eps"])
            * tensors[weight]
        )

    def rotary(value, positions):
        width = value.shape[-1]
        frequencies = 1 / (cfg["rope_theta"] ** (np.arange(0, width, 2, dtype=np.float32) / width))
        angles = positions.astype(np.float32)[:, None] * frequencies[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1)[None, None]
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1)[None, None]
        half = width // 2
        return value * cos + np.concatenate([-value[..., half:], value[..., :half]], axis=-1) * sin

    caches = [None] * cfg["num_hidden_layers"]
    logits, selections, executed = [], [], list(tokens)
    position = 0
    ids = tokens
    heads, kv_heads = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    width, group = cfg["hidden_size"] // heads, heads // kv_heads
    for step in range(decode_steps + 1):
        positions = np.arange(position, position + len(ids), dtype=np.int64)
        hidden = tensors["model.embed_tokens.weight"][list(ids)].copy()
        for layer in range(cfg["num_hidden_layers"]):
            prefix = f"model.layers.{layer}."
            residual = hidden
            normalized = norm(hidden, prefix + "input_layernorm.weight")
            qkv = []
            for name, count in (("q", heads), ("k", kv_heads), ("v", kv_heads)):
                base = prefix + f"self_attn.{name}_proj"
                value = linear(
                    normalized,
                    base + ".weight",
                    base + ".bias" if cfg["model_type"] == "qwen2" else None,
                )
                value = value.reshape(1, len(ids), count, width).transpose(0, 2, 1, 3)
                if cfg["model_type"] == "qwen3" and name != "v":
                    value = norm(value, prefix + f"self_attn.{name}_norm.weight")
                qkv.append(rotary(value, positions) if name != "v" else value)
            query, key, value = qkv
            current = (key[0].transpose(1, 0, 2), value[0].transpose(1, 0, 2))
            old = caches[layer]
            key, value = (
                current
                if old is None
                else tuple(np.concatenate((a, b)) for a, b in zip(old, current, strict=True))
            )
            caches[layer] = (key, value)
            q = query.reshape(1, kv_heads, group, len(ids), width)
            scores = np.empty((1, kv_heads, group, len(ids), len(key)), dtype=np.float32)
            for row in range(len(ids)):
                scores[0, :, :, row] = np.einsum("kgd,tkd->kgt", q[0, :, :, row], key)
            scores = scores.reshape(1, heads, len(ids), len(key)) * float(1 / math.sqrt(width))
            scores = np.where(
                np.arange(len(key))[None, :] <= positions[:, None], scores, np.float32(-np.inf)
            )
            probability = np.exp(scores - scores.max(axis=-1, keepdims=True)).astype(np.float32)
            probability = probability / probability.sum(axis=-1, keepdims=True)
            probability = probability.reshape(1, kv_heads, group, len(ids), len(key))
            attended = np.empty((1, kv_heads, group, len(ids), width), dtype=np.float32)
            for row in range(len(ids)):
                attended[0, :, :, row] = np.einsum("kgt,tkd->kgd", probability[0, :, :, row], value)
            attended = (
                attended.reshape(1, heads, len(ids), width)
                .transpose(0, 2, 1, 3)
                .reshape(len(ids), -1)
            )
            hidden = residual + linear(attended, prefix + "self_attn.o_proj.weight")
            residual = hidden
            normalized = norm(hidden, prefix + "post_attention_layernorm.weight")
            gate = linear(normalized, prefix + "mlp.gate_proj.weight")
            up = linear(normalized, prefix + "mlp.up_proj.weight")
            activation = np.empty_like(gate)
            positive = gate >= 0
            activation[positive] = gate[positive] / (1 + np.exp(-gate[positive]))
            exp = np.exp(gate[~positive])
            activation[~positive] = gate[~positive] * exp / (1 + exp)
            hidden = residual + linear(activation * up, prefix + "mlp.down_proj.weight")
        head = "model.embed_tokens.weight" if cfg["tie_word_embeddings"] else "lm_head.weight"
        final = linear(norm(hidden, "model.norm.weight")[-1:], head).reshape(-1)
        logits.append(final.astype("<f4").tobytes())
        token = int(np.argmax(final))
        selections.append(token)
        position += len(ids)
        if step < decode_steps:
            ids = (token,)
            executed.append(token)
    kv = tuple(
        (tuple(k.shape), k.astype("<f4").tobytes(), v.astype("<f4").tobytes()) for k, v in caches
    )
    return tuple(logits), kv, tuple(selections), tuple(executed), position


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_complete_prefill_decode_matches_independent_integer_float32_oracle(family, tied):
    cfg = config(family, tied)
    contract, arrays, source = fixture(cfg)
    expected = clear_oracle(cfg, arrays, (0, 4, 9), 2)
    receipts = []
    for seeded in (False, True):
        result = decoder.execute_gkm_reference(
            cfg, source, (0, 4, 9), decode_steps=2, seeded=seeded
        )
        assert (
            result.logits,
            result.kv,
            result.selected,
            result.executed,
            result.position,
        ) == expected
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
    assert session.lookup(0) == tensors[contract["table"]][: contract["width"] * 4]
    session.abort()
    first = decoder.execute_gkm_reference(cfg, tensors, (0,), decode_steps=0)
    changed = dict(tensors)
    name = next(name for name in tensors if "input_layernorm" in name)
    value = np.frombuffer(changed[name], dtype="<f4").copy()
    value[0] += np.float32(0.001)
    changed[name] = value.tobytes()
    second = decoder.execute_gkm_reference(cfg, changed, (0,), decode_steps=0)
    assert json.loads(first.acceptance)["source"] != json.loads(second.acceptance)["source"]


@pytest.mark.parametrize(
    "call",
    [
        lambda session: session.lookup(True),
        lambda session: session.lookup("1"),
        lambda session: session.lookup(-1),
        lambda session: session.lookup(32),
        lambda session: session.linear(0, bytes(32)),
        lambda session: session.finish(),
        lambda session: session.linear(False, bytes(32)),
        lambda session: session.linear(0, bytearray(32)),
    ],
)
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
    assert all(
        cache.length == 0 and cache.key is None and cache.value is None for cache in request.caches
    )
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
    for bad in (
        b"short",
        bytes.fromhex("0000c07f") * 8 * 32,
        np.full((32, 8), 16, dtype="<f4").tobytes(),
    ):
        changed = copy.copy(tensors)
        changed["model.embed_tokens.weight"] = bad
        with pytest.raises(ValueError):
            decoder.execute_gkm_reference(cfg, changed, (0,), decode_steps=0)


class _SessionAdmissionReached(Exception):
    pass


def _capture_session_admission(monkeypatch):
    calls = []

    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        raise _SessionAdmissionReached

    monkeypatch.setattr(decoder._native, "_GkmDecoderSession", constructor)
    return calls


class _LookupOnlyTensors(Mapping):
    def __init__(self, values, count=None):
        self.values = values
        self.count = len(values) if count is None else count
        self.reads = []

    def __len__(self):
        return self.count

    def __iter__(self):
        raise AssertionError("source keys must never be enumerated")

    def __getitem__(self, key):
        self.reads.append(key)
        # Stable lazy providers may return equal, newly loaded immutable bytes.
        return memoryview(self.values[key]).tobytes()


@pytest.mark.parametrize("extra", [-1, 1, 10**9])
def test_source_cardinality_rejects_before_any_entry_access(monkeypatch, extra):
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)
    source = _LookupOnlyTensors(tensors, len(tensors) + extra)
    calls = _capture_session_admission(monkeypatch)
    with pytest.raises(ValueError, match="source tensor set mismatch"):
        decoder.execute_gkm_reference(cfg, source, (0,), decode_steps=0)
    assert source.reads == []
    assert calls == []


@pytest.mark.parametrize("hidden_extras", [False, True])
def test_source_mapping_reads_only_declared_names_without_enumeration(monkeypatch, hidden_extras):
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)
    values = {**tensors, "undeclared": b""} if hidden_extras else tensors
    # A lying __len__ cannot make an undeclared key enter the native snapshot.
    source = _LookupOnlyTensors(values, len(tensors))
    calls = _capture_session_admission(monkeypatch)
    with pytest.raises(_SessionAdmissionReached):
        decoder.execute_gkm_reference(cfg, source, (0,), decode_steps=0)
    assert len(calls) == 1
    assert calls[0][0][3] == tensors
    assert source.reads == list(tensors) * 2


@pytest.mark.parametrize("change", ["missing", "cardinality", "replacement", "removal"])
def test_source_mapping_changes_reject_before_native_ownership(monkeypatch, change):
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)
    first = next(iter(tensors))

    class Changing(_LookupOnlyTensors):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            if change == "missing" and key == first:
                raise KeyError(key)
            if change == "cardinality":
                self.count += 1
            if len(self.reads) > len(tensors):
                if change == "replacement":
                    return bytes([value[0] ^ 1]) + value[1:]
                if change == "removal":
                    raise KeyError(key)
            return value

    source = Changing(tensors)
    calls = _capture_session_admission(monkeypatch)
    with pytest.raises(ValueError, match="source mapping"):
        decoder.execute_gkm_reference(cfg, source, (0,), decode_steps=0)
    assert len(source.reads) <= 2 * len(tensors)
    assert set(source.reads) <= tensors.keys()
    assert calls == []


class _CountedConfig(Mapping):
    def __init__(self, values):
        self.values = values
        self.reads = []
        self.visits = 0

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for key in self.values:
            self.visits += 1
            yield key

    def __getitem__(self, key):
        self.reads.append(key)
        return self.values[key]


def test_configuration_snapshot_is_encoded_once_and_shared_with_native_session(monkeypatch):
    cfg = config()
    _, _, tensors = fixture(cfg, 1, 0)

    class Once(_CountedConfig):
        def __getitem__(self, key):
            assert key not in self.reads, "configuration was read again after admission"
            return super().__getitem__(key)

    source = Once(cfg)
    encoded_calls = []
    original = decoder._native.gkm_decoder_contract

    def contract(encoded, *args):
        encoded_calls.append(encoded)
        return original(encoded, *args)

    monkeypatch.setattr(decoder._native, "gkm_decoder_contract", contract)
    calls = _capture_session_admission(monkeypatch)
    with pytest.raises(_SessionAdmissionReached):
        decoder.execute_gkm_reference(source, tensors, (0,), decode_steps=0)
    expected = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert encoded_calls == [expected]
    assert calls[0][0][0] == expected
    assert source.reads == list(cfg)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_bounded_configuration_encoding_preserves_bytes_and_contract(family):
    cfg = config(family)
    cfg["extra_metadata"] = {
        'escaped"\\\n': ["\x00\b\f\t\r\x7f", "é", "😀", -0.0, 1e-300, True, None, (), {}]
    }
    expected = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert decoder._encode_config(_CountedConfig(cfg)) == expected
    assert decoder.decoder_contract(cfg, input_rows=1, decode_steps=0) == (
        _native.gkm_decoder_contract(expected, 1, 0)
    )


def test_configuration_exact_byte_limit_includes_json_escaping():
    cfg = {"escaped": "😀\n\x7f", "padding": ""}
    empty = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    cfg["padding"] = "x" * ((16 << 10) - len(empty))
    expected = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    assert len(expected) == 16 << 10
    assert decoder._encode_config(cfg) == expected
    cfg["padding"] += "x"
    with pytest.raises(ValueError, match="limit"):
        decoder._encode_config(cfg)


@pytest.mark.parametrize("entrypoint", ["contract", "execute"])
@pytest.mark.parametrize(
    "oversize", ["mapping", "string", "escaped", "key", "array", "integer", "depth", "cycle"]
)
def test_configuration_limits_precede_bulk_serialization_and_native_calls(
    monkeypatch, entrypoint, oversize
):
    class HugeMapping(_CountedConfig):
        def __len__(self):
            return 10**9

        def __iter__(self):
            raise AssertionError("oversized mapping enumerated")

        def __getitem__(self, key):
            raise AssertionError("oversized mapping read")

    cfg = config()
    if oversize == "mapping":
        cfg = HugeMapping(cfg)
    elif oversize == "string":
        cfg["extra"] = "x" * ((16 << 10) + 1)
    elif oversize == "escaped":
        cfg["extra"] = "😀" * ((16 << 10) // 12)
    elif oversize == "key":
        cfg["x" * ((16 << 10) + 1)] = None
    elif oversize == "array":
        cfg["extra"] = [None] * (16 << 10)
    elif oversize == "integer":
        cfg["extra"] = 1 << (4 * (16 << 10))
    elif oversize == "depth":
        nested = 0
        for _ in range(70):
            nested = [nested]
        cfg["extra"] = nested
    else:
        cycle = []
        cycle.append(cycle)
        cfg["extra"] = cycle

    original_dumps = json.dumps

    def scalar_only(value, *args, **kwargs):
        assert value is None or type(value) in (bool, int, float), (
            "bulk serialization before admission"
        )
        assert type(value) is not int or value.bit_length() < 4 * (16 << 10)
        return original_dumps(value, *args, **kwargs)

    def native_forbidden(*args, **kwargs):
        raise AssertionError("native called before Python configuration admission")

    monkeypatch.setattr(decoder.json, "dumps", scalar_only)
    monkeypatch.setattr(decoder._native, "gkm_decoder_contract", native_forbidden)
    monkeypatch.setattr(decoder._native, "_GkmDecoderSession", native_forbidden)
    with pytest.raises(ValueError, match="limit"):
        if entrypoint == "contract":
            decoder.decoder_contract(cfg, input_rows=1, decode_steps=0)
        else:
            decoder.execute_gkm_reference(cfg, {}, (0,), decode_steps=0)


@pytest.mark.parametrize("nested", [False, True])
def test_lying_configuration_length_bounds_iteration(monkeypatch, nested):
    class Lying(_CountedConfig):
        def __len__(self):
            return 1

        def __iter__(self):
            for key in ("first", "second", "third"):
                self.visits += 1
                assert self.visits <= 2, "iteration continued beyond declared count plus sentinel"
                yield key

    source = Lying({"first": 0})
    cfg = {**config(), "extra": source} if nested else source

    def native_forbidden(*args):
        raise AssertionError("lying configuration reached native admission")

    monkeypatch.setattr(decoder._native, "gkm_decoder_contract", native_forbidden)
    with pytest.raises(ValueError, match="mapping changed"):
        decoder.decoder_contract(cfg, input_rows=1, decode_steps=0)
    assert source.visits == 2
    assert source.reads == ["first"]
