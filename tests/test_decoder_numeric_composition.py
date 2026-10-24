"""Actual public config adapter joins; synthetic data, no source authentication.

Run only with an approved matching installed native preloaded before source overlay.
No Torch, checkpoints, issuance, decode or trained measurements.
"""
from copy import deepcopy
import builtins
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.research import decoder_numeric_reference as r


@pytest.fixture(autouse=True)
def no_model_or_tensor_framework_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "numpy", "safetensors"}:
            raise AssertionError("model/tensor framework import forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def fixture(hidden=4, heads=2, kv=1):
    from pllm.modeling import lower_model
    config = json.dumps({
        "model_type": "qwen3", "hidden_size": hidden, "intermediate_size": hidden,
        "num_hidden_layers": 1, "num_attention_heads": heads,
        "num_key_value_heads": kv, "head_dim": 2, "vocab_size": 8,
        "max_position_embeddings": 9, "rms_norm_eps": 0.000001,
        "rope_theta": 1000000, "hidden_act": "silu", "attention_bias": False,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0, "rope_scaling": None, "sliding_window": None,
        "use_sliding_window": False, "use_cache": True, "max_window_layers": 1,
    }, separators=(",", ":")).encode()
    original = lower_model(config, batch=1, max_input_tokens=8, max_new_tokens=1)
    plan = r.parse_plan(original.canonical_bytes())
    ops = plan["prefill"]["operations"]
    lookup = ops[0]["attributes"]["weight"]
    table = tuple(r.encoding.round_bf16_bits(Fraction(((t+1)*(j+2)) % 7+1, 8))
                  for t in range(8) for j in range(hidden))
    tensors = {lookup: r.SourceTensor((8, hidden), "BF16", table),
               "separately-serialized-head": r.SourceTensor((8, hidden), "BF16", tuple(list(table)))}
    shapes = {op["id"]: op["output_shape"] for op in ops}
    for op in ops:
        if op["operator"] == "rms_norm":
            width = shapes[op["inputs"][0]][-1]
            tensors[op["attributes"]["weight"]] = r.SourceTensor((width,), "BF16", (0x3f80,)*width)
        if op["operator"] == "linear":
            rows = op["output_shape"][-1]
            columns = shapes[op["inputs"][0]][-1]
            matrix = tuple((128 if i % columns == j else 16*(1+i//2))
                           for i in range(rows) for j in range(columns))
            tensors[op["attributes"]["weight"]] = r.SourceTensor((rows, columns), "synthetic_Q8", matrix)
    return config, original, tensors, (lookup, "separately-serialized-head")


def execute(data, tokens=tuple(range(8))):
    config, _, tensors, pair = data
    return r.compose_fixed8(config, tensors, tokens, scale_refs={}, head_pair=pair)


@pytest.mark.parametrize("geometry", [(4, 2, 1), (8, 4, 2)])
def test_actual_adapter_original_schedule_full_synthetic_vocabulary(geometry):
    data = fixture(*geometry)
    config, original, tensors, _ = data
    before = deepcopy(tensors)
    original_bytes = original.canonical_bytes()
    outcome = execute(data)
    assert outcome.status == "complete_synthetic", outcome.refusal
    assert outcome.config_sha256 == hashlib.sha256(config).hexdigest()
    assert outcome.plan_digest == original.digest
    assert outcome.synthetic_fixture and outcome.source_status == "UNPROVED"
    assert outcome.result.source_and_no_wrap_status == "UNPROVED"
    assert len(outcome.result.logits_f32_bits) == 8
    assert len(outcome.result.linear_calls) == 57
    assert outcome.result.linear_calls[-1][2:] == (7, 8)
    assert any(outcome.result.logits_f32_bits)
    assert original.canonical_bytes() == original_bytes and tensors == before
    assert not hasattr(outcome.result, "kv")
    eps = [o["attributes"]["epsilon"] for o in r.parse_plan(original_bytes)["prefill"]["operations"]
           if o["operator"] == "rms_norm"]
    assert all(Fraction(e) == Fraction(1, 1000000) for e in eps)


def test_explicit_tie_and_original_id_row_permutation():
    data = fixture()
    baseline = execute(data)
    assert baseline.result is not None
    config, original, tensors, pair = data
    swapped = dict(tensors)
    table = tensors[pair[0]]
    permutation = (1, 0, 2, 3, 4, 5, 6, 7)
    words = tuple(w for t in permutation for w in table.words[t*4:(t+1)*4])
    for ref in pair:
        swapped[ref] = r.SourceTensor(table.shape, table.dtype, words)
    result = execute((config, original, swapped, pair), tokens=permutation)
    assert result.result is not None
    assert result.result.logits_f32_bits == tuple(baseline.result.logits_f32_bits[t] for t in permutation)
    swapped[pair[1]] = table
    assert execute((config, original, swapped, pair)).status == "refused"


def test_distinct_kv_group_permutation_with_matching_query_and_output_blocks():
    data = fixture(8, 4, 2)
    config, original, tensors, pair = data
    baseline = execute(data)
    assert baseline.result is not None
    linears = [o for o in r.parse_plan(original.canonical_bytes())["prefill"]["operations"]
               if o["operator"] == "linear"]
    changed = dict(tensors)
    for node, permutation in zip(linears[:3], ((4, 5, 6, 7, 0, 1, 2, 3), (2, 3, 0, 1), (2, 3, 0, 1))):
        ref = node["attributes"]["weight"]
        t = tensors[ref]
        changed[ref] = r.SourceTensor(t.shape, t.dtype,
                                     tuple(w for row in permutation for w in t.words[row*8:(row+1)*8]))
    ref = linears[3]["attributes"]["weight"]
    t = tensors[ref]
    changed[ref] = r.SourceTensor(t.shape, t.dtype,
                                 tuple(t.words[i*8+j] for i in range(8) for j in (4, 5, 6, 7, 0, 1, 2, 3)))
    result = execute((config, original, changed, pair))
    assert result.result is not None
    assert result.result.logits_f32_bits == baseline.result.logits_f32_bits


@pytest.mark.parametrize("linear_index", [2, 4])
def test_attention_and_gated_mlp_are_not_dead_zero_paths(linear_index):
    data = fixture()
    config, original, tensors, pair = data
    baseline = execute(data)
    linears = [o for o in r.parse_plan(original.canonical_bytes())["prefill"]["operations"]
               if o["operator"] == "linear"]
    ref = linears[linear_index]["attributes"]["weight"]
    t = tensors[ref]
    changed = dict(tensors)
    changed[ref] = r.SourceTensor(t.shape, t.dtype, (0,)*len(t.words))
    result = execute((config, original, changed, pair))
    assert baseline.result is not None and result.result is not None
    assert result.result.logits_f32_bits != baseline.result.logits_f32_bits


@pytest.mark.parametrize("mutation", ["missing", "scale", "tie", "tokens", "source_cap"])
def test_bad_bindings_refuse_before_coefficient_conversion(monkeypatch, mutation):
    config, original, tensors, pair = fixture()
    if mutation == "missing":
        tensors.pop(next(ref for ref in tensors if ref not in pair))
    elif mutation == "tie":
        pair = (pair[0], pair[0])
    elif mutation == "source_cap":
        tensors["foreign"] = r.SourceTensor((4097,), "synthetic_Q8", (0,)*4097)
    monkeypatch.setattr(r.encoding, "q8_i64", lambda *_: pytest.fail("conversion before preflight"))
    outcome = r.compose_fixed8(config, tensors, (0,)*7 if mutation == "tokens" else tuple(range(8)),
                              scale_refs={"unknown": "missing"} if mutation == "scale" else {}, head_pair=pair)
    assert outcome.status == "refused" and outcome.result is None


@pytest.mark.parametrize("mutation", ["attribute", "transform", "group", "shape", "suffix",
                                      "plan_cap", "ops_cap", "live_cap", "loop_cap"])
def test_foreign_original_records_refuse_before_operators(monkeypatch, mutation):
    from pllm import modeling
    config, original, tensors, pair = fixture(8, 4, 2)
    plan = json.loads(original.canonical_bytes())
    ops = plan["prefill"]["operations"]
    if mutation == "attribute":
        next(o for o in ops if o["operator"] == "silu")["attributes"]["approximation"] = "host"
    elif mutation == "transform":
        plan["transformations"] = ["foreign"]
    elif mutation == "group":
        next(o for o in ops if o["operator"] == "attention_values")["attributes"]["group_size"] = 1
    elif mutation == "shape":
        next(o for o in ops if o["operator"] == "reshape")["output_shape"][-1] = 4
    elif mutation == "plan_cap":
        plan["foreign"] = "x"*262145
    elif mutation == "ops_cap":
        ops.extend(deepcopy(ops))
    elif mutation in ("live_cap", "loop_cap"):
        next(o for o in ops if o["operator"] == "output_head")["output_shape"][-1] = (
            1000000 if mutation == "live_cap" else 10000)
    else:
        ops[-1]["attributes"]["foreign"] = True
    tampered = modeling.ModelPlan(json.dumps(plan).encode())
    monkeypatch.setattr(modeling, "lower_model", lambda *a, **k: tampered)
    monkeypatch.setattr(r.encoding, "q8_i64", lambda *_: pytest.fail("conversion before complete graph check"))
    assert r.compose_fixed8(config, tensors, tuple(range(8)), scale_refs={}, head_pair=pair).status == "refused"


def test_scalar_budget_refusal_preserves_typed_unresolved(monkeypatch):
    data = fixture()
    math = r.scalar_owner()
    real = math.sincos_f32
    monkeypatch.setattr(math, "sincos_f32", lambda x: real(x, limits=math.Limits(terms=1, dyadic_bits=16)))
    outcome = execute(data)
    assert outcome.status == "unresolved" and outcome.result is None
    assert isinstance(outcome.refusal, math.Unresolved)


def test_bad_config_and_word_caps_refuse_without_factory_import():
    assert r.compose_fixed8(b" "*16385, {}, (), scale_refs={}, head_pair=("a", "b")).status == "refused"
    assert r.compose_fixed8(b"[]", {}, (), scale_refs={}, head_pair=("a", "b")).status == "refused"


@pytest.mark.parametrize("layout", ["text_config", "unknown_nested", "deep_json"])
def test_nested_or_parser_exhaustion_refuses_before_factory_and_execution(monkeypatch, layout):
    from pllm import modeling
    # Build only the small valid fixture. Never lower the malicious config.
    config, _, tensors, pair = fixture()
    document = json.loads(config)
    if layout == "deep_json":
        malformed = b"["*1200+b"0"+b"]"*1200
    else:
        document["model_type"] = "qwen3_5" if layout == "text_config" else "qwen3"
        document[layout] = {"model_type": "qwen3_5_text", "num_hidden_layers": 32,
                            "hidden_size": 4096, "intermediate_size": 14336,
                            "num_attention_heads": 32, "num_key_value_heads": 8,
                            "head_dim": 128, "vocab_size": 248320}
        malformed = json.dumps(document).encode()
    before = bytes(malformed)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("native factory/operator reached before flat JSON preflight")

    monkeypatch.setattr(modeling, "lower_model", forbidden)
    monkeypatch.setattr(r, "run_fixed8", forbidden)
    monkeypatch.setattr(r.encoding, "q8_i64", forbidden)
    result = r.compose_fixed8(malformed, tensors, tuple(range(8)), scale_refs={}, head_pair=pair)
    assert result.status == "refused" and isinstance(result.refusal, r.PlanError)
    assert result.result is None and result.plan_digest is None and calls == []
    assert malformed == before and len(malformed) < 16384


def test_independent_cast_placement_fixture():
    # Exact BF16 pre-gamma rounding: sqrt(2) -> 181/128, times3 -> 543/128 -> 4.25.
    assert r.norm((0x3f80, 0), (0x4040, 0x3f80)) == (0x4088, 0)
