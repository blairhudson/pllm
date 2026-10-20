"""Generated public tensors only. No model/checkpoint download or vendor imports."""

from hashlib import sha256
import struct

from oracle import GRAPH, graph_profile

# Frozen example coefficient bytes: binary32-rounded cos(p*f), sin(p*f),
# p=0..5, f=(1, .01), repeated across split halves; theta=10000, d=4.
# Captured from Python host libm once. They define ONLY this generated fixture;
# no claim of exact-real rounding, source framework byte parity, or checkpoint pin.
COS = [
    ["3f800000", "3f800000", "3f800000", "3f800000"],
    ["3f0a5140", "3f7ffcb9", "3f0a5140", "3f7ffcb9"],
    ["bed51133", "3f7ff2e5", "bed51133", "3f7ff2e5"],
    ["bf7d7026", "3f7fe283", "bf7d7026", "3f7fe283"],
    ["bf275530", "3f7fcb94", "bf275530", "3f7fcb94"],
    ["3e913c2c", "3f7fae19", "3e913c2c", "3f7fae19"],
]
SIN = [
    ["00000000", "00000000", "00000000", "00000000"],
    ["3f576aa4", "3c23d657", "3f576aa4", "3c23d657"],
    ["3f68c7b7", "3ca3d43e", "3f68c7b7", "3ca3d43e"],
    ["3e1081c3", "3cf5b920", "3e1081c3", "3cf5b920"],
    ["bf41bdcf", "3d23cbdb", "bf41bdcf", "3d23cbdb"],
    ["bf757c10", "3d4cb6f5", "bf757c10", "3d4cb6f5"],
]


def tensor(shape, codes, dtype="f32"):
    words = []
    for code in codes:
        # All generated codes multiples of 32: exactly representable in all
        # three formats. Encoding edge/tie tests live in test_decoder.py.
        value = code / 4096
        if dtype == "f16":
            word = struct.pack(">e", value).hex()
        else:
            full = struct.pack(">f", value).hex()
            word = full if dtype == "f32" else full[:4]
            if dtype == "bf16" and full[4:] != "0000":
                raise ValueError("generated BF16 value not exact")
        words.append(word)
    return {"shape": shape, "dtype": dtype, "words": words}


def fixture(family, *, tie_control=False):
    gpt = family == "gpt2"
    dims = dict(hidden=8, intermediate=12, layers=2, heads=2,
                kv_heads=2 if gpt else 1, head_dim=4, vocab=7, max_positions=6)
    if family == "qwen2":
        # Nontrivial grouping distinguishes contiguous [0,0,1,1] from modulo
        # [0,1,0,1], unlike a single-KV-head GQA example.
        dims.update(hidden=16, heads=4, kv_heads=2)
    h = dims["hidden"]
    tensors = {}

    def add(name, shape, role="weight"):
        size = 1
        for dim in shape:
            size *= dim
        dtype = ("f32", "f16", "bf16")[len(tensors) % 3]
        values = []
        for i in range(size):
            byte = sha256(f"fusefss-decoder-v1:{family}:{name}:{i}".encode("ascii")).digest()[0]
            if role == "gamma":
                code = 4096 + (byte % 5 - 2) * 32
            elif role == "embedding":
                code = (byte % 97 - 48) * 64
            elif role == "positions":
                code = (byte % 17 - 8) * 32
            elif role == "bias":
                code = (byte % 7 - 3) * 32
            else:
                code = (byte % 17 - 8) * 32
            values.append(code)
        tensors[name] = tensor(shape, values, dtype)

    add("embedding", [7, h], "embedding")
    add("final.gamma", [h], "gamma")
    if tie_control:
        # Every vocabulary score equal. EOS=0 must still generate exactly T.
        tensors["head"] = tensor([7, h], [0] * (7 * h))
    if gpt:
        add("positions", [6, h], "positions")
        add("final.beta", [h], "bias")
    else:
        for name, values in (("rope.cos", COS), ("rope.sin", SIN)):
            tensors[name] = {"shape": [6, 4], "dtype": "f32", "words": [w for row in values for w in row]}
    for layer in range(2):
        base = f"layer.{layer}."
        for n in ("norm1", "norm2"):
            add(base + n + ".gamma", [h], "gamma")
            if gpt:
                add(base + n + ".beta", [h], "bias")
        for n, output, input_ in (("q", h, h), ("k", dims["kv_heads"] * 4, h),
                                  ("v", dims["kv_heads"] * 4, h), ("o", h, h),
                                  ("up", 12, h), ("down", h, 12)):
            add(base + n + ".weight", [output, input_])
            if gpt or (family == "qwen2" and n in ("q", "k", "v")):
                add(base + n + ".bias", [output], "bias")
        if not gpt:
            add(base + "gate.weight", [12, h])
        if family == "qwen3":
            for n in ("q_norm", "k_norm"):
                add(base + n + ".gamma", [4], "gamma")
    return {"id": family + ("-untied-tie-eos-control" if tie_control else "-tied-generated"),
            "family": family, "profile": graph_profile(family), "graph_status": GRAPH, "dimensions": dims,
            "prefix": [2, 4], "generation_count": 3, "eos_id": 0,
            "tied_head": not tie_control, "attention_scale": tensor([1], [2048]),
            "norm_contract": "d8-layer-sumsq-plus-one" if gpt else "d13-rms-sumsq-plus-one",
            "rope_contract": "none" if gpt else "d13-split-half-per-product-floor", "tensors": tensors}


def fixtures():
    return [fixture("gpt2"), fixture("gpt2", tie_control=True),
            fixture("qwen2"), fixture("qwen3"), fixture("llama")]
