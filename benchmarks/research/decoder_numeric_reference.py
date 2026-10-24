"""Clear fixed-eight-ID operator oracle over exported DecoderPlan v1 records.

Research only: supplied plan/tensors are NOT authenticated OriginalSourcePlan or
verified checkpoint owners. No serving/admission/SDK/protocol authority is made.
The explicit D31/D32 binding selects the already approved arithmetic instead of
the original adapter's Q30/libm rotary annotation; the input plan is not mutated.
No generation or persistent KV is executed. Scalar work exhaustion propagates
typed Unresolved, with no host-math fallback or partially accepted output.

All matrices are public supplied source values. Integrity, on-disk dtype/extents,
complete source/domain finiteness and honest no-wrap remain independent gates.
Fixture Q8 matrices are permitted ONLY in explicitly synthetic operator fixtures.
IR owner: crates/pllm-models/src/lib.rs DecoderPlan/ModelOperation/lower_graph.
Only its dense grouped-query, gated-SiLU schedule is supported here; no other
operator is silently approximated. This does not mint a factory-origin owner.
This module reads no checkpoint, imports no model/framework, and takes no math
or source-loader callback. The fixed scalar-owner file is ordinary trusted code,
not publisher authentication or proof that an exported plan came from a factory.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
import importlib.util
import json
import hashlib
from math import prod
from pathlib import Path
import sys

from benchmarks.research import maverick_numeric_encoding_oracle as encoding


NUMERICS = "M-N/Qwen3-4B-8prompt-candidate-1"


@dataclass(frozen=True)
class Composition:
    """Synthetic development result; neither JSON nor tensors authenticate source."""

    status: str
    result: Result | None = None
    config_sha256: str | None = None
    plan_digest: str | None = None
    refusal: Exception | None = None
    source_status: str = "UNPROVED"
    synthetic_fixture: bool = True
    execution_scope: str = "DEVELOPMENT_SYNTHETIC"


def _bound_attributes(ops, tensors, d, scale_refs, head_pair):
    """Complete structural/source preflight: no tensor conversion or operators."""
    shapes = {op["id"]: op["output_shape"] for op in ops}
    used = set(head_pair)
    projections = set()
    for op in ops:
        kind, a = op["operator"], op["attributes"]
        if kind in ("linear", "rms_norm"):
            ref = a["weight"]
            used.add(ref)
            source = tensors[ref]
            width = shapes[op["inputs"][0]][-1]
            if kind == "linear":
                _attrs(a, {"weight": ref, "bias": None})
                _require(source.shape == (op["output_shape"][-1], width), "original matrix geometry")
                if source.dtype == "F8_E4M3":
                    scale = tensors[scale_refs[ref]]
                    used.add(scale_refs[ref])
                    projections.add(ref)
                    r, c = source.shape
                    _require(r % 128 == c % 128 == 0 and scale.dtype == "BF16"
                             and scale.shape == (r//128, c//128), "explicit block geometry")
                    # Existing immutable SourceTensor already validates BF16.
                    # Check sign/zero here without materializing converted values.
                    _require(all(0 < w < 0x7f80 for w in scale.words), "positive finite scales")
                else:
                    _require(source.dtype == "synthetic_Q8" and ref not in scale_refs, "explicit synthetic matrix")
            else:
                _require(set(a) == {"weight", "epsilon", "weight_offset"}
                         and type(a["epsilon"]) in (F, str) and F(a["epsilon"]) == F(1, 1000000)
                         and type(a["weight_offset"]) is int and a["weight_offset"] == 0
                         and source.dtype == "BF16" and source.shape == (width,), "norm source/attributes")
        elif kind == "token_lookup":
            _attrs(a, {"weight": head_pair[0]})
            _require(op["inputs"] == ["input.tokens"], "original IDs")
        elif kind == "output_head":
            _attrs(a, {"weight": head_pair[0], "tied": True, "input_layout": "batch_hidden",
                       "output_layout": "batch_vocabulary", "compute_dtype": "model_native",
                       "output_dtype": "model_native"})
        elif kind == "reshape":
            _attrs(a, {"layout": "batch_heads_sequence_feature" if len(op["output_shape"]) == 4
                       else "batch_sequence_hidden"})
        elif kind == "rotary_embedding":
            _attrs(a, {"theta": d.theta, "rotary_dimensions": d.width, "pairing": "split_half",
                       "position_policy": "sequential_absolute", "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
                       "input_layout": "batch_heads_sequence_feature", "output_layout": "batch_heads_sequence_feature",
                       "tail_policy": "unchanged"})
        elif kind == "kv_cache_append":
            _require(type(a.get("state")) is str and bool(a["state"]), "state reference")
            _attrs(a, {"state": a["state"], "mode": "initialize", "state_capacity": 8,
                       "state_layout": "batch_kv_heads_sequence_feature", "absolute_write_positions_input": "input.positions",
                       "padding_mask_input": "input.attention_mask", "valid_lengths_input": "input.sequence_lengths",
                       "attention_domain": {"layout": "batch_kv_heads_sequence_feature", "maximum_sequence": 8,
                                            "includes_current": True}})
        elif kind == "cache_suffix":
            _attrs(a, {"axis": 2, "maximum_sequence": 8, "semantics": "visible_valid_prefix",
                       "absolute_write_positions_input": "input.positions", "padding_mask_input": "input.attention_mask",
                       "valid_lengths_input": "input.sequence_lengths"})
        elif kind in ("attention_scores", "attention_values"):
            _attrs(a, {"group_size": d.heads//d.kv_heads})
        elif kind == "attention_scale":
            _attrs(a, {"head_dim": d.width})
        elif kind == "softmax":
            _attrs(a, {"axis": -1})
        elif kind == "last_token":
            _attrs(a, {"axis": 1, "selection": "last_valid", "valid_lengths_input": "input.sequence_lengths"})
        elif kind in ("residual_add", "multiply", "silu", "causal_mask"):
            _attrs(a, {})
        else:
            raise PlanError("unsupported original operator")
    _require(set(tensors) == used and set(scale_refs) == projections, "complete exact source bindings")


def compose_fixed8(config_bytes, tensors, tokens, *, scale_refs, head_pair, numeric_binding=NUMERICS):
    """Real public adapter -> original IR -> existing oracle, bounded synthetic only.

    No decode/selection/feedback execution. Failures return no partial result.
    Scalar Unresolved remains typed in refusal. Native factory needs an explicitly
    selected matching development installation, never checkpoint/model imports.
    """
    identity = digest = None
    try:
        _require(type(config_bytes) is bytes and len(config_bytes) <= 16384, "bounded config bytes")
        identity = hashlib.sha256(config_bytes).hexdigest()
        config = parse_plan(config_bytes)
        _require(type(config) is dict, "config object")
        # This bounded join supports ONLY this flat adapter grammar. A nested
        # text_config can drive a different native graph despite small top-level
        # geometry; never flatten it or let unchecked aliases reach the factory.
        fields = {"model_type", "hidden_size", "intermediate_size", "num_hidden_layers",
                  "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                  "max_position_embeddings", "rms_norm_eps", "rope_theta", "hidden_act",
                  "attention_bias", "tie_word_embeddings", "attention_dropout", "rope_scaling",
                  "sliding_window", "use_sliding_window", "use_cache", "max_window_layers"}
        _require(set(config) == fields and config["model_type"] == "qwen3"
                 and all(type(v) in (str, int, bool, F, type(None)) for v in config.values()),
                 "supported flat adapter config required")
        # Bound native factory allocation BEFORE lowering; geometry is still derived
        # independently from its output below, not supplied to the operator runner.
        _require(type(config.get("num_hidden_layers")) is int and config["num_hidden_layers"] == 1,
                 "bounded synthetic layer count")
        for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size"):
            _require(type(config.get(key)) is int and 0 < config[key] <= 64, "bounded explicit factory geometry")
        _require(type(tensors) is dict and all(type(t) is SourceTensor for t in tensors.values()),
                 "immutable source records")
        source_words = sum(len(t.words) for t in tensors.values())
        _require(source_words <= 4096, "bounded immutable source words")
        _require(type(scale_refs) is dict and all(type(k) is str and type(v) is str for k, v in scale_refs.items())
                 and type(head_pair) is tuple and len(head_pair) == 2
                 and all(type(k) is str for k in head_pair), "explicit bindings")
        from pllm.modeling import lower_model
        original = lower_model(config_bytes, batch=1, max_input_tokens=8, max_new_tokens=1)
        raw = original.canonical_bytes()
        _require(type(raw) is bytes and len(raw) <= 262144, "bounded canonical plan")
        digest = original.digest
        plan = parse_plan(raw)
        ops = plan["prefill"]["operations"]
        _require(type(ops) is list and len(ops) <= 64, "bounded original operations")
        _require(len({op["id"] for op in ops}) == len(ops)
                 and not {"input.tokens", "input.positions", "input.attention_mask", "input.sequence_lengths"}
                 .intersection(op["id"] for op in ops), "unique non-shadowing original IDs")
        heads = [op for op in ops if op["operator"] == "output_head"]
        _require(len(heads) == 1, "unique full head")
        layers = sum(op["operator"] == "linear" for op in ops)//7
        d = Dimensions(layers, ops[0]["output_shape"][-1], ops[25]["output_shape"][-1],
                       ops[5]["output_shape"][1], ops[6]["output_shape"][1], ops[5]["output_shape"][-1],
                       heads[0]["output_shape"][-1], ops[10]["attributes"]["theta"], "synthetic_operator_fixture")
        words = sum(prod(op["output_shape"]) for op in ops)
        transient = max(prod(op["output_shape"]) for op in ops)*4
        _require(words+4*source_words+transient <= 65536
                 and words*4*(16+max(d.hidden, d.intermediate, d.heads*d.width))+4*source_words <= 1000000,
                 "bounded live tensor words/structural loop visits")
        _bound_attributes(ops[:1+30*d.layers+3], tensors, d, scale_refs, head_pair)
        result = run_fixed8(plan, tensors, tokens, dimensions=d, scale_refs=scale_refs,
                            head_pair=head_pair, numeric_binding=numeric_binding)
        return Composition("complete_synthetic", result, identity, digest)
    except (PlanError, KeyError, TypeError, IndexError, ValueError, ArithmeticError, ImportError) as error:
        unresolved = isinstance(error, scalar_owner().Unresolved)
        # Do not leak tensor values or parser payloads in public diagnostics.
        refusal = error if unresolved else PlanError("composition preflight or numeric refusal")
        return Composition("unresolved" if unresolved else "refused", None, identity, digest, refusal)


class PlanError(ValueError):
    pass


@lru_cache(maxsize=1)
def scalar_owner():
    # Avoid pllm.__init__ and native imports. Never resolve code from plan data.
    path = Path(__file__).resolve().parents[2]/"python/pllm/runtime/exact_float_reference.py"
    name = "_decoder_exact_scalar_"+str(id(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Dimensions:
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    width: int
    vocabulary: int
    theta: int
    scope: str

    def __post_init__(self):
        values = (self.layers, self.hidden, self.intermediate, self.heads,
                  self.kv_heads, self.width, self.vocabulary, self.theta)
        if any(type(v) is not int or v <= 0 for v in values):
            raise PlanError("positive explicit integer dimensions required")
        if self.heads % self.kv_heads or self.width % 2:
            raise PlanError("grouped heads and split-half rotary dimensions required")
        if self.scope == "selected_fixed8":
            if values != (36, 2560, 9728, 32, 8, 128, 151936, 1000000):
                raise PlanError("selected source geometry cannot be narrowed")
        elif self.scope != "synthetic_operator_fixture":
            raise PlanError("explicit selected or synthetic scope required")


def selected_dimensions():
    return Dimensions(36, 2560, 9728, 32, 8, 128, 151936, 1000000, "selected_fixed8")


@dataclass(frozen=True)
class SourceTensor:
    shape: tuple[int, ...]
    dtype: str
    words: tuple[int, ...]

    def __post_init__(self):
        if (type(self.shape) is not tuple or type(self.words) is not tuple
                or any(type(n) is not int or n <= 0 for n in self.shape)
                or not self.shape or prod(self.shape) != len(self.words)):
            raise PlanError("explicit flat tensor geometry required")
        if self.dtype == "BF16":
            for word in self.words:
                encoding.decode_bf16(word)
        elif self.dtype == "F8_E4M3":
            for word in self.words:
                encoding.decode_e4m3fn(word)
        elif self.dtype == "synthetic_Q8":
            if any(type(w) is not int or not -(1 << 63) <= w < (1 << 63) for w in self.words):
                raise PlanError("fixture coefficients must fit signed i64")
        else:
            raise PlanError("no source dtype conversion or fallback")


@dataclass(frozen=True)
class Tensor:
    shape: tuple[int, ...]
    words: tuple  # logical BF16; score tensors alone use None for excluded future

    def __post_init__(self):
        if prod(self.shape) != len(self.words):
            raise PlanError("operator output geometry mismatch")


@dataclass(frozen=True)
class Result:
    logits_f32_bits: tuple[int, ...]
    linear_calls: tuple  # (operation ID, original weight ref, token row, output rows)
    scope: str

    @property
    def numeric_binding(self):
        return NUMERICS

    @property
    def source_and_no_wrap_status(self):
        return "UNPROVED"

    def little_endian_logits(self):
        return b"".join(w.to_bytes(4, "little") for w in self.logits_f32_bits)


def parse_plan(text):
    # Preserve decimal epsilon exactly; do not first round through host binary64.
    def reject_nonfinite(_):
        raise PlanError("nonfinite JSON literal")
    try:
        return json.loads(text, parse_float=F, parse_constant=reject_nonfinite)
    except RecursionError:
        raise PlanError("bounded JSON parser recursion exhausted") from None


def _r16(word):
    decoded = encoding.decode_f32(word)
    return encoding.round_bf16_bits(decoded.value, negative_zero=decoded.negative_zero)


def _promote(word):
    return encoding.promote_bf16_to_f32_bits(word)


def _op32(a, b, operation):
    math = scalar_owner()
    x, y = math.decode_f32(a), math.decode_f32(b)
    if operation == "add":
        z = x+y
        negative = not z and not x and not y and bool(a & b & 0x80000000)
    elif operation == "mul":
        z, negative = x*y, bool((a ^ b) & 0x80000000)
    elif operation == "div":
        if not y:
            raise math.NonFinite("non-mask division by zero")
        z, negative = x/y, bool((a ^ b) & 0x80000000)
    else:
        raise PlanError("unknown scalar operation")
    return math.round_f32(z, negative_zero=bool(negative and not z))


def _op16(a, b, operation):
    return _r16(_op32(_promote(a), _promote(b), operation))


def dot32(a, b):
    if len(a) != len(b):
        raise PlanError("dot geometry mismatch")
    z = 0
    for x, y in zip(a, b):
        z = _op32(z, _op32(_promote(x), _promote(y), "mul"), "add")
    return z


def norm(row, gamma):
    if not row or len(row) != len(gamma):
        raise PlanError("norm geometry mismatch")
    math = scalar_owner()
    z = dot32(row, row)
    mean = _op32(z, math.round_f32(len(row)), "div")
    a = _op32(mean, math.round_f32(F(1, 1000000)), "add")
    reciprocal = math.rsqrt_f32(a)
    # Cast BEFORE learned gamma; then R32 product and second BF16 cast.
    return tuple(_op16(g, _r16(_op32(_promote(x), reciprocal, "mul")), "mul")
                 for x, g in zip(row, gamma))


def rotary_coefficients(dims):
    math = scalar_owner()
    cosine, sine = [[] for _ in range(8)], [[] for _ in range(8)]
    for i in range(dims.width//2):
        exponent = math.decode_f32(math.round_f32(F(2*i, dims.width)))
        power = math.pow_ratio_f32(dims.theta, exponent.numerator, exponent.denominator)
        frequency = _op32(0x3f800000, power, "div")
        for position in range(8):
            angle = _op32(math.round_f32(position), frequency, "mul")
            s, c = math.sincos_f32(angle)
            cosine[position].append(_r16(c))
            sine[position].append(_r16(s))
    return tuple((tuple(c+c), tuple(s+s)) for c, s in zip(cosine, sine))


def rope(row, coefficients):
    cosine, sine = coefficients
    _require(bool(row) and len(row) % 2 == 0 and len(row) == len(cosine) == len(sine), "split-half rotary geometry mismatch")
    half = len(row)//2
    rotated = tuple(x ^ 0x8000 for x in row[half:])+tuple(row[:half])
    return tuple(_op16(_op16(x, c, "mul"), _op16(r, s, "mul"), "add")
                 for x, r, c, s in zip(row, rotated, cosine, sine))


def softmax(row):
    if not row or any(x is None for x in row):
        raise PlanError("softmax receives only the causal valid prefix")
    math = scalar_owner()
    maximum = max(row, key=lambda w: (encoding.decode_bf16(w).value, not bool(w & 0x8000)))
    exponentials = []
    z = 0
    for word in row:
        difference = _op32(_promote(word), _promote(maximum) ^ 0x80000000, "add")
        e = math.exp_f32(difference)
        exponentials.append(e)
        z = _op32(z, e, "add")
    return tuple(_r16(_op32(e, z, "div")) for e in exponentials)


_LAYER = ("rms_norm", "linear", "linear", "linear", "reshape", "reshape", "reshape",
          "rms_norm", "rms_norm", "rotary_embedding", "rotary_embedding",
          "kv_cache_append", "kv_cache_append", "cache_suffix", "cache_suffix",
          "attention_scores", "attention_scale", "causal_mask", "softmax", "attention_values",
          "reshape", "linear", "residual_add", "rms_norm", "linear", "linear",
          "silu", "multiply", "linear", "residual_add")


def _require(condition, message):
    if not condition:
        raise PlanError(message)


def _attrs(attrs, expected):
    def same(a, b):
        if type(a) is not type(b):
            return False
        if type(a) is dict:
            return a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
        if type(a) is list:
            return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    _require(same(attrs, expected), "unsupported operator attributes; no invented defaults")


def _connections(operations, d):
    """Validate positional IR dependencies, never parse text of operation IDs."""
    h, q, kv = (1, 8, d.hidden), (1, d.heads, 8, d.width), (1, d.kv_heads, 8, d.width)
    ql, kl, ff, scores = (1, 8, d.heads*d.width), (1, 8, d.kv_heads*d.width), (1, 8, d.intermediate), (1, d.heads, 8, 8)
    _require(tuple(operations[0]["output_shape"]) == h, "embedding output shape mismatch")
    previous = operations[0]["id"]
    extras = ["input.positions", "input.attention_mask", "input.sequence_lengths"]
    links = [None, [0], [0], [0], [1], [2], [3], [4], [5], [7, "input.positions"],
             [8, "input.positions"], [10]+extras, [6]+extras, [11]+extras, [12]+extras,
             [9, 13], [15], [16, "input.positions"], [17], [18, 14], [19], [20],
             ["previous", 21], [22], [23], [23], [24], [26, 25], [27], [22, 28]]
    shapes = [h, ql, kl, kl, q, kv, kv, q, kv, q, kv, kv, kv, kv, kv,
              scores, scores, scores, scores, q, ql, h, h, h, ff, ff, ff, ff, h, h]
    for layer in range(d.layers):
        nodes = operations[1+len(_LAYER)*layer:1+len(_LAYER)*(layer+1)]
        for i, (node, refs, shape) in enumerate(zip(nodes, links, shapes)):
            expected = [previous] if refs is None else [previous if r == "previous" else
                       nodes[r]["id"] if type(r) is int else r for r in refs]
            _require(node["inputs"] == expected and node["layer"] == layer
                     and tuple(node["output_shape"]) == shape, "original layer links/order/geometry mismatch")
            state = "key" if i in (11, 13) else "value" if i in (12, 14) else None
            _require(node["state_kind"] == state, "unexpected state annotation")
        previous = nodes[-1]["id"]
    _require(operations[-3]["inputs"] == [previous] and tuple(operations[-3]["output_shape"]) == h
             and tuple(operations[-2]["output_shape"]) == (1, d.hidden)
             and tuple(operations[-1]["output_shape"]) == (1, d.vocabulary), "final norm/full-head geometry mismatch")


def _projection(source, scale, fixture):
    rows, columns = source.shape
    if source.dtype == "synthetic_Q8":
        _require(fixture and scale is None, "fixture coefficients cannot substitute real FP8 source")
        return source.words
    _require(source.dtype == "F8_E4M3" and scale is not None, "projection requires FP8 and BF16 scales")
    _require(rows % 128 == columns % 128 == 0, "no edge-block padding rule is approved")
    _require(scale.dtype == "BF16" and scale.shape == (rows//128, columns//128), "block scale axes mismatch")
    for word in scale.words:
        _require(encoding.decode_bf16(word).value > 0, "scale must be positive finite")
    values = []
    for offset, byte in enumerate(source.words):
        i, j = divmod(offset, columns)
        scale_word = scale.words[(i//128)*(columns//128)+j//128]
        s = encoding.decode_f32(encoding.promote_bf16_to_f32_bits(scale_word)).value
        v = encoding.decode_e4m3fn(byte)
        # General positive-finite import, not the separate S<=2^45 proof premise.
        effective = encoding.round_f32_bits(v.value*s, negative_zero=v.negative_zero)
        values.append(encoding.q8_i64(encoding.decode_f32(effective).value))
    return tuple(values)


def _linear(matrix, rows, inputs):
    dimension = len(inputs)
    _require(rows > 0 and dimension > 0 and len(matrix) == rows*dimension, "public matrix geometry mismatch")
    x = tuple(encoding.q8_i64(encoding.decode_bf16(w).value) % encoding.BABYBEAR for w in inputs)
    result = []
    for row in range(rows):
        residue = 0
        for j, word in enumerate(x):
            residue = (residue+(matrix[row*dimension+j] % encoding.BABYBEAR)*word) % encoding.BABYBEAR
        result.append(encoding.centered_reply_to_bf16_bits(residue))
    return tuple(result)


def run_fixed8(plan, tensors, tokens, *, dimensions, scale_refs, head_pair, numeric_binding):
    """Execute original prefill operators through the unique full output head.

    All source references come from attributes plus explicit source-to-scale and
    separately serialized tie bindings. ID text is opaque. No loader/authentication
    callback, delegated provider, token selection or generation can be installed.
    """
    d = dimensions
    _require(type(plan) is dict and type(tensors) is dict and type(scale_refs) is dict
             and all(type(t) is SourceTensor for t in tensors.values()), "plain supplied data only; no loader callbacks")
    _require(type(d) is Dimensions and numeric_binding == NUMERICS, "explicit selected numeric binding required")
    _require(type(tokens) is tuple and len(tokens) == 8 and
             all(type(t) is int and 0 <= t < d.vocabulary for t in tokens), "exactly eight original token IDs required")
    _require(plan["schema_version"] == "pllm.decoder_plan.v1" and not plan["transformations"], "original untransformed plan required")
    graph = plan["prefill"]
    _require((graph["mode"], graph["batch"], graph["query_sequence"], graph["maximum_key_sequence"]) ==
             ("prefill", 1, 8, 8) and not graph["state_inputs"], "one request-local eight-token prefill required")
    operations = graph["operations"]
    expected = ("token_lookup",)+_LAYER*d.layers+("rms_norm", "last_token", "output_head")
    _require(tuple(o["operator"] for o in operations[:len(expected)]) == expected, "unrecognized original operator schedule")
    suffix = operations[len(expected):]
    _require(not suffix or tuple(o["operator"] for o in suffix) ==
             ("greedy_token_selection", "token_feedback"), "unrecognized non-logit suffix")
    operations = operations[:len(expected)]
    for op in operations+suffix:
        _require(type(op) is dict and set(op) == {"id", "operator", "layer", "state_kind", "inputs", "output_shape", "attributes"}
                 and type(op["id"]) is str and bool(op["id"])
                 and type(op["inputs"]) is list and all(type(s) is str for s in op["inputs"])
                 and type(op["output_shape"]) is list and all(type(n) is int and n > 0 for n in op["output_shape"])
                 and (op["layer"] is None or type(op["layer"]) is int)
                 and type(op["attributes"]) is dict, "canonical ModelOperation fields required")
    _require(len({o["id"] for o in operations}) == len(operations), "duplicate operation IDs")
    _connections(operations, d)
    if suffix:
        _require(suffix[0]["inputs"] == [operations[-1]["id"]] and
                 suffix[1]["inputs"] == [suffix[0]["id"]] and graph["output"] == suffix[1]["id"], "invalid generation suffix links")
        for node, shape in zip(suffix, ([1], [1, 1])):
            _attrs(node["attributes"], {"policy": "pllm.greedy.v1", "source": "execution_policy"})
            _require(node["output_shape"] == shape, "invalid generation suffix shape")
    else:
        _require(graph["output"] == operations[-1]["id"], "full output head required")
    states = []
    for layer in range(d.layers):
        for offset, kind in ((11, "key"), (12, "value")):
            node = operations[1+len(_LAYER)*layer+offset]
            states.append({"id": node["id"], "layer": layer, "kind": kind,
                           "shape": [1, d.kv_heads, 8, d.width], "maximum_sequence": 8})
    _require(graph["state_outputs"] == states, "original request-local KV output metadata mismatch")
    lookup_ref = operations[0]["attributes"]["weight"]
    _require(type(head_pair) is tuple and len(head_pair) == 2 and head_pair[0] == lookup_ref
             and head_pair[0] != head_pair[1], "separately serialized embedding/head tie required")
    embedding, head = (tensors[ref] for ref in head_pair)
    _require(embedding.dtype == head.dtype == "BF16" and
             embedding.shape == head.shape == (d.vocabulary, d.hidden) and
              embedding.words == head.words, "finite BF16 copies must agree bit-for-bit")
    _bound_attributes(operations, tensors, d, scale_refs, head_pair)
    head_matrix = tuple(encoding.q8_i64(encoding.decode_f32(encoding.promote_bf16_to_f32_bits(w)).value) for w in head.words)
    head_op = operations[-1]
    _attrs(head_op["attributes"], {"weight": lookup_ref, "tied": True,
                                 "input_layout": "batch_hidden", "output_layout": "batch_vocabulary",
                                 "compute_dtype": "model_native", "output_dtype": "model_native"})
    final_norm_id = operations[-3]["id"]
    _require(operations[-2]["inputs"] == [final_norm_id, "input.sequence_lengths"] and
             head_op["inputs"] == [operations[-2]["id"]], "head must follow final norm and position7 selection")
    values, matrices, calls = {}, {}, []
    used = set(head_pair)
    for op in operations:
        if op["operator"] in ("linear", "rms_norm"):
            used.add(op["attributes"]["weight"])
        if op["operator"] == "linear":
            ref = op["attributes"]["weight"]
            source = tensors[ref]
            _require(len(source.shape) == 2, "linear source must be output-by-input")
            scale_ref = scale_refs.get(ref)
            if scale_ref is not None:
                used.add(scale_ref)
            matrices[ref] = _projection(source, tensors[scale_ref] if scale_ref else None,
                                        d.scope == "synthetic_operator_fixture")
    _require(set(tensors) == used and set(scale_refs) == {ref for ref in matrices if tensors[ref].dtype == "F8_E4M3"},
             "extra/missing source or scale references")
    if d.scope == "selected_fixed8":
        _require(len(used) == 651, "selected complete tensor set requires651 distinct source references")
    coefficients = rotary_coefficients(d)
    math = scalar_owner()
    alpha = math.rsqrt_f32(math.round_f32(d.width))
    external = {"input.tokens", "input.positions", "input.attention_mask", "input.sequence_lengths"}
    _require(not external.intersection(o["id"] for o in operations), "operation cannot shadow original inputs")
    for op in operations:
        kind, attrs, shape = op["operator"], op["attributes"], tuple(op["output_shape"])
        args = [values[name] for name in op["inputs"] if name not in external]
        if kind == "token_lookup":
            _attrs(attrs, {"weight": lookup_ref})
            _require(op["inputs"] == ["input.tokens"], "lookup must use original IDs")
            result = Tensor((1, 8, d.hidden), tuple(w for t in tokens for w in embedding.words[t*d.hidden:(t+1)*d.hidden]))
        elif kind == "rms_norm":
            _require(set(attrs) == {"epsilon", "weight", "weight_offset"} and
                     type(attrs["epsilon"]) in (F, str) and F(attrs["epsilon"]) == F(1, 1000000)
                     and type(attrs["weight_offset"]) is int and attrs["weight_offset"] == 0, "selected epsilon/cast-before-gamma norm required")
            source = tensors[attrs["weight"]]
            _require(source.dtype == "BF16" and source.shape == (args[0].shape[-1],), "norm gamma geometry mismatch")
            width = source.shape[0]
            data = args[0].words[-width:] if op["id"] == final_norm_id else args[0].words
            if len(args[0].shape) == 4:
                words = [0]*len(data)
                for t in range(8):
                    for h in range(args[0].shape[1]):
                        k = (h*8+t)*width
                        words[k:k+width] = norm(data[k:k+width], source.words)
            else:
                words = [y for k in range(0, len(data), width) for y in norm(data[k:k+width], source.words)]
            result = Tensor((1, d.hidden) if op["id"] == final_norm_id else args[0].shape, tuple(words))
        elif kind in ("linear", "output_head"):
            ref = attrs["weight"]
            if kind == "linear":
                _attrs(attrs, {"weight": ref, "bias": None})
                rows, width = tensors[ref].shape
                matrix = matrices[ref]
            else:
                rows, width = d.vocabulary, d.hidden
                matrix = head_matrix
            _require(args[0].shape[-1] == width, "linear input width mismatch")
            words = []
            for t in range(len(args[0].words)//width):
                words.extend(_linear(matrix, rows, args[0].words[t*width:(t+1)*width]))
                calls.append((op["id"], ref, 7 if kind == "output_head" else t, rows))
            result = Tensor(args[0].shape[:-1]+(rows,), tuple(words))
        elif kind == "reshape":
            source = args[0]
            if len(shape) == 4:
                _attrs(attrs, {"layout": "batch_heads_sequence_feature"})
                _, heads, length, width = shape
                _require(length == 8 and width == d.width and source.shape == (1, 8, heads*width), "head reshape mismatch")
                words = tuple(source.words[(t*heads+h)*width+k] for h in range(heads) for t in range(8) for k in range(width))
            else:
                _attrs(attrs, {"layout": "batch_sequence_hidden"})
                _, heads, length, width = source.shape
                _require(shape == (1, 8, heads*width), "context concatenation mismatch")
                words = tuple(source.words[(h*8+t)*width+k] for t in range(8) for h in range(heads) for k in range(width))
            result = Tensor(shape, words)
        elif kind == "rotary_embedding":
            _attrs(attrs, {"theta": d.theta, "rotary_dimensions": d.width, "pairing": "split_half",
                           "position_policy": "sequential_absolute", "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
                           "input_layout": "batch_heads_sequence_feature", "output_layout": "batch_heads_sequence_feature",
                           "tail_policy": "unchanged"})
            _require(op["inputs"][1:] == ["input.positions"], "fixed original positions required")
            source = args[0]
            words = [0]*len(source.words)
            for t in range(8):
                for h in range(source.shape[1]):
                    k = (h*8+t)*d.width
                    words[k:k+d.width] = rope(source.words[k:k+d.width], coefficients[t])
            result = Tensor(source.shape, tuple(words))
        elif kind in ("kv_cache_append", "cache_suffix"):
            _require(op["inputs"][1:] == ["input.positions", "input.attention_mask", "input.sequence_lengths"], "only request-local KV prefix is supported")
            if kind == "kv_cache_append":
                _attrs(attrs, {"state": attrs["state"], "mode": "initialize", "state_capacity": 8,
                               "state_layout": "batch_kv_heads_sequence_feature", "absolute_write_positions_input": "input.positions",
                               "padding_mask_input": "input.attention_mask", "valid_lengths_input": "input.sequence_lengths",
                               "attention_domain": {"layout": "batch_kv_heads_sequence_feature", "maximum_sequence": 8, "includes_current": True}})
            else:
                _attrs(attrs, {"axis": 2, "maximum_sequence": 8, "semantics": "visible_valid_prefix",
                               "absolute_write_positions_input": "input.positions", "padding_mask_input": "input.attention_mask",
                               "valid_lengths_input": "input.sequence_lengths"})
            result = args[0]  # local alias, never exported as accepted persistent state
        elif kind == "attention_scores":
            _attrs(attrs, {"group_size": d.heads//d.kv_heads})
            q, k = args
            words = [None]*(d.heads*64)
            for t in range(8):
                for h in range(d.heads):
                    kv = h//(d.heads//d.kv_heads)
                    for j in range(t+1):
                        words[(h*8+t)*8+j] = _r16(dot32(q.words[(h*8+t)*d.width:(h*8+t+1)*d.width],
                                                      k.words[(kv*8+j)*d.width:(kv*8+j+1)*d.width]))
            result = Tensor((1, d.heads, 8, 8), tuple(words))
        elif kind == "attention_scale":
            _attrs(attrs, {"head_dim": d.width})
            result = Tensor(args[0].shape, tuple(None if w is None else _r16(_op32(_promote(w), alpha, "mul")) for w in args[0].words))
        elif kind == "causal_mask":
            _attrs(attrs, {})
            _require(op["inputs"][1:] == ["input.positions"], "original causal positions required")
            result = args[0]  # None denotes EXCLUDED future, never a field/float -inf
        elif kind == "softmax":
            _attrs(attrs, {"axis": -1})
            words = [0]*(d.heads*64)
            for t in range(8):
                for h in range(d.heads):
                    row = args[0].words[(h*8+t)*8:(h*8+t+1)*8]
                    _require(all(w is None for w in row[t+1:]), "future positions must be excluded")
                    words[(h*8+t)*8:(h*8+t+1)*8] = softmax(row[:t+1])+(0,)*(7-t)
            result = Tensor((1, d.heads, 8, 8), tuple(words))
        elif kind == "attention_values":
            _attrs(attrs, {"group_size": d.heads//d.kv_heads})
            probabilities, v = args
            words = [0]*(d.heads*8*d.width)
            for t in range(8):
                for h in range(d.heads):
                    kv = h//(d.heads//d.kv_heads)
                    p = probabilities.words[(h*8+t)*8:(h*8+t)*8+t+1]
                    for k in range(d.width):
                        words[(h*8+t)*d.width+k] = _r16(dot32(p, tuple(v.words[(kv*8+j)*d.width+k] for j in range(t+1))))
            result = Tensor((1, d.heads, 8, d.width), tuple(words))
        elif kind in ("residual_add", "multiply"):
            _attrs(attrs, {})
            _require(args[0].shape == args[1].shape, "elementwise geometry mismatch")
            result = Tensor(args[0].shape, tuple(_op16(a, b, "add" if kind == "residual_add" else "mul")
                                               for a, b in zip(args[0].words, args[1].words)))
        elif kind == "silu":
            _attrs(attrs, {})
            result = Tensor(args[0].shape, tuple(_r16(math.silu_f32(w)) for w in args[0].words))
        elif kind == "last_token":
            _attrs(attrs, {"axis": 1, "selection": "last_valid", "valid_lengths_input": "input.sequence_lengths"})
            result = args[0]  # final norm already computed ONLY on position7
        else:
            raise math.Unresolved("unsupported original semantic operator")
        _require(result.shape == shape or (op["id"] == final_norm_id and shape == (1, 8, d.hidden)), "declared operator output shape mismatch")
        values[op["id"]] = result
    _require(len(calls) == d.layers*7*8+1 and values[head_op["id"]].shape == (1, d.vocabulary), "incomplete semantic linear/head coverage")
    return Result(tuple(_promote(w) for w in values[head_op["id"]].words), tuple(calls), d.scope)
