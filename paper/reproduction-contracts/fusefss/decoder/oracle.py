"""Independent bounded integer decoder. Clear diagnostic traces, never MPC.

Only scalar TABLE BYTES are reused. All composition/arithmetic is local Python.
D13 graph decisions are enumerated in NUMERIC_DECISIONS.md.
"""

from fractions import Fraction
from hashlib import sha256
from math import isqrt
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ONE = 4096
HALF = 1 << 49
SCHEMA = "pllm.fusefss.decoder-exchange.v1"
BASE_PROFILE = "fusefss.source-derived-ring50-q12.v1"
GPT2_PROFILE = "fusefss.gpt2-complete-ring50-q12.v1"
DENSE_PROFILE = "fusefss.dense-rope-ring50-q12.v1"
GRAPH = "d13-approved-arithmetic"
MAX_BYTES = 8 << 20
# Bounded profile coefficient pins: F32 1/sqrt(head_dim), then exact Q12 encode.
ATTENTION_SCALE_F32 = {2: "3f3504f3", 4: "3f000000", 6: "3ed105ec", 8: "3eb504f3"}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def integer(x, lo, hi):
    if type(x) is not int or not lo <= x <= hi:
        raise ValueError("exact bounded integer required")
    return x


def checked(x):
    return integer(x, -HALF, HALF - 1)


def graph_profile(family):
    if family not in ("gpt2", "qwen2", "qwen3", "llama"):
        raise ValueError("unsupported family")
    return GPT2_PROFILE if family == "gpt2" else DENSE_PROFILE


def fields(x, names):
    if type(x) is not dict or set(x) != set(names.split()):
        raise ValueError("exact object fields required")


def ieee_q12(dtype, word):
    """Decode finite IEEE binary value as rational, then exact ties-even Q12."""
    if dtype not in ("f32", "f16", "bf16"):
        raise ValueError("unsupported external dtype")
    exponent_bits, fraction_bits = {"f32": (8, 23), "f16": (5, 10), "bf16": (8, 7)}[dtype]
    width = 1 + exponent_bits + fraction_bits
    if type(word) is not str or len(word) != width // 4 or any(c not in "0123456789abcdef" for c in word):
        raise ValueError("canonical lowercase IEEE hex required")
    bits = int(word, 16)
    sign = -1 if bits >> (width - 1) else 1
    exponent = (bits >> fraction_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:
        raise ValueError("nonfinite external value")
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent:
        mantissa += 1 << fraction_bits
    power = (exponent if exponent else 1) - bias - fraction_bits + 12
    rational = Fraction(sign * mantissa) * (Fraction(2) ** power)
    lower, remainder = divmod(rational.numerator, rational.denominator)
    twice = 2 * remainder
    rounded = lower + int(twice > rational.denominator or (twice == rational.denominator and lower % 2))
    return checked(rounded)


def external_tensor(value, shape):
    fields(value, "dtype shape words")
    if type(value["shape"]) is not list or len(value["shape"]) != len(shape):
        raise ValueError("external tensor rank")
    count = 1
    for actual, wanted in zip(value["shape"], shape):
        integer(actual, wanted, wanted)
        count *= actual
    if type(value["words"]) is not list or len(value["words"]) != count:
        raise ValueError("external tensor payload shape")
    return [ieee_q12(value["dtype"], word) for word in value["words"]]


class Trace:
    def __init__(self):
        self.values = []
        self.names = set()

    def add(self, name, values, shape, axes, fractional=12):
        if name in self.names or len(shape) != len(axes):
            raise ValueError("trace identity/axes")
        count = 1
        for dim in shape:
            count *= integer(dim, 1, 4096)
        if len(values) != count:
            raise ValueError("trace tensor shape")
        for value in values:
            checked(value)
        self.names.add(name)
        self.values.append({"name": name, "shape": shape, "axes": axes,
                            "fractional": fractional, "bits": 50, "signed": True,
                            "values": list(values)})
        return values


def rows(flat, width):
    return [flat[i:i + width] for i in range(0, len(flat), width)]


def flat(rows_):
    return [x for row in rows_ for x in row]


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("dot shape")
    accumulator = 0
    for x, y in zip(a, b):
        accumulator = checked(accumulator + checked(x * y))
    return accumulator


def linear(x, weights, bias, tr, name):
    raw = [[dot(row, w) for w in weights] for row in x]
    scaled = [[v // ONE for v in row] for row in raw]
    out = [[checked(v + b) for v, b in zip(row, bias)] for row in scaled]
    shape = [len(x), len(weights)]
    for label, value, scale in (("accumulator", raw, 24), ("rescaled", scaled, 12), ("output", out, 12)):
        tr.add(name + "." + label, flat(value), shape, ["sequence", "out_feature"], scale)
    return out


def norm(x, gamma, beta, kind, tr, name):
    k = len(gamma)
    means, centered, sums, exponents, mantissas, indices, inverses, products, q, affine, out = ([] for _ in range(11))
    totals, mean_products, squares = [], [], []
    for row in x:
        total = 0
        for v in row:
            total = checked(total + v)
        mean_product = checked(total * (ONE // k)) if kind == "layer" else 0
        mean = mean_product // ONE
        z = [checked(v - mean) for v in row]
        s = 0
        squared = [checked(v * v) for v in z]
        for v in squared:
            s = checked(s + v)
        s = checked(s + 1)
        e = s.bit_length() - 1
        m = (s * 128 // (1 << e)) - 128
        inverse = isqrt(((1 << 48) * k * 128) // ((128 + m) * (1 << e)))
        checked(inverse)
        p = [checked(v * inverse) for v in z]
        normalized = [v // ONE for v in p]
        a = [checked(v * g) for v, g in zip(normalized, gamma)]
        output = [checked(v // ONE + b) for v, b in zip(a, beta)]
        means.append(mean); centered.append(z); sums.append(s)
        exponents.append(e); mantissas.append(m); indices.append(64 * m + e)
        inverses.append(inverse); products.append(p); q.append(normalized)
        affine.append(a); out.append(output)
        totals.append(total); mean_products.append(mean_product); squares.append(squared)
    for label, value, fractional in (("row_sum", totals, 12), ("mean_product", mean_products, 24),
                                     ("mean", means, 12), ("sum_squares_plus_one", sums, 24),
                                     ("exponent", exponents, 0), ("mantissa", mantissas, 0),
                                     ("index", indices, 0), ("inverse", inverses, 12)):
        tr.add(name + "." + label, value, [len(x)], ["row"], fractional)
    for label, value, fractional in (("centered", centered, 12), ("squared", squares, 24), ("inverse_product", products, 24),
                                     ("normalized", q, 12), ("affine_product", affine, 24), ("output", out, 12)):
        tr.add(name + "." + label, flat(value), [len(x), k], ["row", "feature"], fractional)
    return out


def activation(x, table, tr, name):
    y = [[v // 64 for v in row] for row in x]
    indices = [[min(abs(v), len(table) - 1) for v in row] for row in y]
    payloads = [[table[i] for i in row] for row in indices]
    out = [[checked((v if reduced >= 0 else 0) - t) for v, reduced, t in zip(row, yr, ts)]
           for row, yr, ts in zip(x, y, payloads)]
    for label, value, scale in (("floor", y, 6), ("index", indices, 0), ("payload", payloads, 12), ("output", out, 12)):
        tr.add(name + "." + label, flat(value), [len(x), len(x[0])], ["sequence", "feature"], scale)
    return out


def softmax(row, query_position, tables, tr, name):
    maximum = max(row[:query_position + 1])
    difference, clipped, exps = [], [], []
    for t, value in enumerate(row):
        d = checked(maximum - value) if t <= query_position else 0
        c = min(d, 65535)
        e = tables["exp0"][c % 256] * tables["exp1"][c // 256] // ONE if t <= query_position else 0
        difference.append(d); clipped.append(c); exps.append(e)
    denominator = checked(sum(exps)) // 64
    integer(denominator, 64, 64 * (query_position + 1))
    reciprocal = (1 << 18) // denominator
    products = [checked(v * reciprocal) for v in exps]
    out = [v // ONE for v in products]
    for label, value, scale in (("mask", [int(t <= query_position) for t in range(len(row))], 0),
                                ("difference", difference, 12), ("clipped", clipped, 12),
                                ("exp", exps, 12), ("reciprocal_product", products, 24), ("output", out, 12)):
        tr.add(name + "." + label, value, [len(row)], ["key_sequence"], scale)
    for label, value, scale in (("maximum", maximum, 12), ("denominator", denominator, 6), ("reciprocal", reciprocal, 12)):
        tr.add(name + "." + label, [value], [1], ["scalar"], scale)
    return out


def heads(x, count, d):
    return [[row[h * d:(h + 1) * d] for row in x] for h in range(count)]


def head_trace(tr, name, x, fractional=12):
    # Physical native exchange axes always [batch,heads,sequence,feature].
    tr.add(name, [v for head in x for row in head for v in row],
           [1, len(x), len(x[0]), len(x[0][0])], ["batch", "heads", "sequence", "feature"], fractional)


def rotary(x, cos, sin, tr, name):
    out, left_products, right_products = [], [], []
    d = len(x[0][0])
    for head in x:
        result, lp, rp = [], [], []
        for p, row in enumerate(head):
            rotated = [-v for v in row[d // 2:]] + row[:d // 2]
            a = [checked(v * c) for v, c in zip(row, cos[p])]
            b = [checked(v * s) for v, s in zip(rotated, sin[p])]
            result.append([checked(v // ONE + w // ONE) for v, w in zip(a, b)])
            lp.append(a); rp.append(b)
        out.append(result); left_products.append(lp); right_products.append(rp)
    head_trace(tr, name + ".cos_product", left_products, 24)
    head_trace(tr, name + ".sin_product", right_products, 24)
    head_trace(tr, name + ".cos_rescaled", [[[v // ONE for v in row] for row in head] for head in left_products])
    head_trace(tr, name + ".sin_rescaled", [[[v // ONE for v in row] for row in head] for head in right_products])
    head_trace(tr, name + ".output", out)
    return out


def load_tables():
    # Frozen scalar bytes only, no reuse of scalar arithmetic or native code.
    gelu = json.loads((HERE.parent / "gelu-table.json").read_text())["values"]
    table_file = json.loads((HERE.parent / "gate-set" / "fixtures.json").read_text())
    tables = table_file["tables"]
    result = {"gelu": gelu, **{key: tables[key] for key in ("silu", "exp0", "exp1")}}
    expected = {"gelu": (256, "b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b"),
                "silu": (1024, "ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f"),
                "exp0": (256, "e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b"),
                "exp1": (256, "21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9")}
    for name, values in result.items():
        count, digest = expected[name]
        if type(values) is not list or len(values) != count:
            raise ValueError("scalar table shape")
        for value in values:
            integer(value, 0, ONE)
        actual = sha256(b"".join(v.to_bytes(8, "little") for v in values)).hexdigest()
        if actual != digest:
            raise ValueError("scalar table hash")
    return result


def validate_contract(c):
    fields(c, "id family profile graph_status dimensions prefix generation_count eos_id tied_head attention_scale norm_contract rope_contract tensors")
    if c["family"] not in ("gpt2", "qwen2", "qwen3", "llama") or c["graph_status"] != GRAPH:
        raise ValueError("unsupported graph arithmetic")
    if c["profile"] != graph_profile(c["family"]):
        raise ValueError("family/profile mismatch")
    if type(c["id"]) is not str or not c["id"].isascii() or not 1 <= len(c["id"]) <= 64:
        raise ValueError("bounded fixture ID")
    d = c["dimensions"]
    fields(d, "hidden intermediate layers heads kv_heads head_dim vocab max_positions")
    for key, maximum in (("hidden", 32), ("intermediate", 64), ("layers", 3), ("heads", 4),
                         ("kv_heads", 4), ("head_dim", 8), ("vocab", 32), ("max_positions", 16)):
        integer(d[key], 1, maximum)
    if d["hidden"] != d["heads"] * d["head_dim"] or d["heads"] % d["kv_heads"] or d["head_dim"] % 2:
        raise ValueError("head/GQA dimensions")
    if c["family"] == "gpt2" and d["kv_heads"] != d["heads"]:
        raise ValueError("GPT-2 requires MHA")
    if type(c["prefix"]) is not list or not 1 <= len(c["prefix"]) <= 8:
        raise ValueError("bounded nonempty prefix")
    for token in c["prefix"]:
        integer(token, 0, d["vocab"] - 1)
    integer(c["generation_count"], 1, 4)
    if len(c["prefix"]) + c["generation_count"] > d["max_positions"]:
        raise ValueError("feedback exceeds position capacity")
    integer(c["eos_id"], 0, d["vocab"] - 1)
    if type(c["tied_head"]) is not bool:
        raise ValueError("explicit head tying")
    expected_norm = "d8-layer-sumsq-plus-one" if c["family"] == "gpt2" else "d13-rms-sumsq-plus-one"
    expected_rope = "none" if c["family"] == "gpt2" else "d13-split-half-per-product-floor"
    if c["norm_contract"] != expected_norm or c["rope_contract"] != expected_rope:
        raise ValueError("norm/RoPE numeric contract")
    external_tensor(c["attention_scale"], [1])
    if c["attention_scale"] != {"dtype": "f32", "shape": [1], "words": [ATTENTION_SCALE_F32[d["head_dim"]]]}:
        raise ValueError("attention scale differs from bounded public F32 pin")
    h, m, v, p = (d[k] for k in ("hidden", "intermediate", "vocab", "max_positions"))
    specs = {"embedding": [v, h], "final.gamma": [h]}
    if not c["tied_head"]:
        specs["head"] = [v, h]
    if c["family"] == "gpt2":
        specs.update({"positions": [p, h], "final.beta": [h]})
    else:
        specs.update({"rope.cos": [p, d["head_dim"]], "rope.sin": [p, d["head_dim"]]})
    for layer in range(d["layers"]):
        base = f"layer.{layer}."
        for n in ("norm1", "norm2"):
            specs[base + n + ".gamma"] = [h]
            if c["family"] == "gpt2":
                specs[base + n + ".beta"] = [h]
        for n, output, input_ in (("q", h, h), ("k", d["kv_heads"] * d["head_dim"], h),
                                  ("v", d["kv_heads"] * d["head_dim"], h), ("o", h, h),
                                  ("up", m, h), ("down", h, m)):
            specs[base + n + ".weight"] = [output, input_]
            if c["family"] == "gpt2" or (c["family"] == "qwen2" and n in ("q", "k", "v")):
                specs[base + n + ".bias"] = [output]
        if c["family"] != "gpt2":
            specs[base + "gate.weight"] = [m, h]
        if c["family"] == "qwen3":
            for n in ("q_norm", "k_norm"):
                specs[base + n + ".gamma"] = [d["head_dim"]]
    if type(c["tensors"]) is not dict or set(c["tensors"]) != set(specs):
        raise ValueError("missing/unknown public weights")
    if c["family"] != "gpt2":
        for name in ("rope.cos", "rope.sin"):
            if type(c["tensors"][name]) is not dict or c["tensors"][name].get("dtype") != "f32":
                raise ValueError("D13 RoPE requires pinned F32 coefficient bytes")
    return {name: external_tensor(c["tensors"][name], shape) for name, shape in specs.items()}


def run(c, tables=None):
    weights = validate_contract(c)
    tables = load_tables() if tables is None else tables
    tr = Trace()
    for name in sorted(c["tensors"]):
        tensor = c["tensors"][name]
        tr.add("public." + name, weights[name], tensor["shape"], [f"axis{i}" for i in range(len(tensor["shape"]))])
    d = c["dimensions"]
    h, nh, nk, hd, vocab = (d[k] for k in ("hidden", "heads", "kv_heads", "head_dim", "vocab"))
    kind = "layer" if c["family"] == "gpt2" else "rms"
    embedding = rows(weights["embedding"], h)
    prefix = list(c["prefix"])
    scale = external_tensor(c["attention_scale"], [1])[0]
    tr.add("public.attention_scale", [scale], [1], ["scalar"])
    generated, states = [], []
    for step in range(c["generation_count"]):
        name = f"step.{step}"
        length = len(prefix)
        tr.add(name + ".input_ids", prefix, [length], ["sequence"], 0)
        tr.add(name + ".positions", list(range(length)), [length], ["sequence"], 0)
        one_hot = [[int(token == i) for i in range(vocab)] for token in prefix]
        tr.add(name + ".embedding_equality", flat(one_hot), [length, vocab], ["sequence", "vocab"], 0)
        x = [[checked(sum(bit * embedding[i][j] for i, bit in enumerate(row))) for j in range(h)] for row in one_hot]
        tr.add(name + ".token_embedding", flat(x), [length, h], ["sequence", "feature"])
        if kind == "layer":
            pos = rows(weights["positions"], h)
            x = [[checked(a + b) for a, b in zip(row, pos[p])] for p, row in enumerate(x)]
        tr.add(name + ".embedding_output", flat(x), [length, h], ["sequence", "feature"])
        for layer in range(d["layers"]):
            key = f"layer.{layer}."
            label = name + "." + key[:-1]

            def normalized(value, n):
                return norm(value, weights[key + n + ".gamma"], weights.get(key + n + ".beta", [0] * h), kind, tr, label + "." + n)

            def projected(value, n, output):
                return linear(value, rows(weights[key + n + ".weight"], len(value[0])),
                              weights.get(key + n + ".bias", [0] * output), tr, label + "." + n)

            a = normalized(x, "norm1")
            q = heads(projected(a, "q", h), nh, hd)
            k = heads(projected(a, "k", nk * hd), nk, hd)
            v = heads(projected(a, "v", nk * hd), nk, hd)
            for tag, value in (("q", q), ("k", k), ("v", v)):
                head_trace(tr, label + "." + tag + ".physical", value)
            if c["family"] == "qwen3":
                q = [norm(head, weights[key + "q_norm.gamma"], [0] * hd, "rms", tr, label + f".q_norm.head.{i}") for i, head in enumerate(q)]
                k = [norm(head, weights[key + "k_norm.gamma"], [0] * hd, "rms", tr, label + f".k_norm.head.{i}") for i, head in enumerate(k)]
            if kind == "rms":
                cos, sin = rows(weights["rope.cos"], hd), rows(weights["rope.sin"], hd)
                q = rotary(q, cos, sin, tr, label + ".q_rope")
                k = rotary(k, cos, sin, tr, label + ".k_rope")
            group = nh // nk
            tr.add(label + ".query_to_kv_head", [head // group for head in range(nh)], [nh], ["query_head"], 0)
            attended = []
            for head in range(nh):
                kv = head // group
                raw = [[dot(query, keyrow) for keyrow in k[kv]] for query in q[head]]
                unscaled = [[v_ // ONE for v_ in row] for row in raw]
                products = [[checked(v_ * scale) for v_ in row] for row in unscaled]
                scores = [[v_ // ONE for v_ in row] for row in products]
                for tag, value, fractional in (("qk_accumulator", raw, 24), ("qk_rescaled", unscaled, 12),
                                                ("scale_product", products, 24), ("scores", scores, 12)):
                    tr.add(label + f".head.{head}." + tag, flat(value), [length, length], ["query_sequence", "key_sequence"], fractional)
                probs = [softmax(row, p, tables, tr, label + f".head.{head}.softmax.{p}") for p, row in enumerate(scores)]
                acc = [[dot(row, [v[kv][t][j] for t in range(length)]) for j in range(hd)] for row in probs]
                out = [[v_ // ONE for v_ in row] for row in acc]
                tr.add(label + f".head.{head}.pv_accumulator", flat(acc), [length, hd], ["sequence", "feature"], 24)
                attended.append(out)
            head_trace(tr, label + ".attention.physical", attended)
            merged = [[v_ for head in attended for v_ in head[p]] for p in range(length)]
            tr.add(label + ".attention.merged", flat(merged), [length, h], ["sequence", "feature"])
            attn = projected(merged, "o", h)
            x = [[checked(a_ + b_) for a_, b_ in zip(row, ar)] for row, ar in zip(x, attn)]
            tr.add(label + ".attention_residual", flat(x), [length, h], ["sequence", "feature"])
            a = normalized(x, "norm2")
            up = projected(a, "up", d["intermediate"])
            if kind == "layer":
                activated = activation(up, tables["gelu"], tr, label + ".gelu")
            else:
                gate = projected(a, "gate", d["intermediate"])
                silu = activation(gate, tables["silu"], tr, label + ".silu")
                product = [[checked(a_ * b_) for a_, b_ in zip(row, ur)] for row, ur in zip(silu, up)]
                tr.add(label + ".gated_product", flat(product), [length, d["intermediate"]], ["sequence", "feature"], 24)
                activated = [[v_ // ONE for v_ in row] for row in product]
                tr.add(label + ".gated_output", flat(activated), [length, d["intermediate"]], ["sequence", "feature"])
            mlp = projected(activated, "down", h)
            x = [[checked(a_ + b_) for a_, b_ in zip(row, mr)] for row, mr in zip(x, mlp)]
            tr.add(label + ".mlp_residual", flat(x), [length, h], ["sequence", "feature"])
        x = norm(x, weights["final.gamma"], weights.get("final.beta", [0] * h), kind, tr, name + ".final_norm")
        head = embedding if c["tied_head"] else rows(weights["head"], h)
        logits = linear(x, head, [0] * vocab, tr, name + ".head")
        last = logits[-1]
        maximum = max(last)
        equal = [int(v_ == maximum) for v_ in last]
        token = equal.index(1)
        tr.add(name + ".argmax.maximum", [maximum], [1], ["scalar"])
        tr.add(name + ".argmax.equal_maximum", equal, [vocab], ["vocab"], 0)
        tr.add(name + ".argmax.one_hot", [int(i == token) for i in range(vocab)], [vocab], ["vocab"], 0)
        generated.append(token)
        prefix.append(token)
        tr.add(name + ".selected_id", [token], [1], ["scalar"], 0)
        tr.add(name + ".appended_ids", prefix, [len(prefix)], ["sequence"], 0)
        states.append({"step": step, "prefix_length_in": length, "prefix_length_out": len(prefix),
                       "positions_recomputed": list(range(length)), "persistent_kv": False,
                       "selected_id_private": token, "prefix_ids_private": list(prefix), "released": False})
    return {"schema": SCHEMA, "profile": c["profile"], "graph_status": GRAPH, "fixture_id": c["id"],
            "input_sha256": sha256(canonical(c)).hexdigest(), "tensors": tr.values, "states": states,
            "final": {"recipient": "client-only", "token_ids": prefix, "generated_ids": generated,
                      "selection_count": c["generation_count"], "persistent_kv": False}}


def strict_equal(expected, actual, path="exchange"):
    if type(expected) is not type(actual):
        raise ValueError(path + ": type mismatch")
    if type(expected) is dict:
        if set(expected) != set(actual):
            raise ValueError(path + ": fields mismatch")
        for key in expected:
            strict_equal(expected[key], actual[key], path + "." + key)
    elif type(expected) is list:
        if len(expected) != len(actual):
            raise ValueError(path + ": length mismatch")
        for i, (a, b) in enumerate(zip(expected, actual)):
            strict_equal(a, b, path + f"[{i}]")
    elif expected != actual:
        raise ValueError(path + ": value mismatch")


def read_json(path):
    data = Path(path).read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("exchange byte limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError("nonfinite JSON constant")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)
