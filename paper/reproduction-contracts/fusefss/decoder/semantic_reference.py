"""Independent fixed-graph semantic oracle, including actually computed padding.

No native code, Rust process, source model or producer output is called/imported.
Original 5125-tensor fixture/arithmetic remains unchanged. This schema requires
every fixed semantic node, never a dynamically chosen subset of passing tensors.
"""
from hashlib import sha256
from math import prod

from oracle import (HERE, ONE, Trace, activation, canonical, checked, dot, fields,
                    flat, linear, load_tables, norm, read_json, rotary, rows,
                    strict_equal, validate_contract)

SCHEMA = "pllm.fusefss.semantic-execution-batch.v1"
RESULT = "pllm.fusefss.semantic-execution.v1"


def digest(value):
    return sha256(canonical(value)).hexdigest()


def blueprint(c):
    """Independently prescribed ID/operator order; not read from native output."""
    gpt = c["family"] == "gpt2"
    result = [("token_lookup", "TokenLookup")]
    if gpt:
        result += [("position_lookup", "TokenLookup"), ("embedding_sum", "ResidualAdd")]
    for layer in range(c["dimensions"]["layers"]):
        ops = [("input_norm", "LayerNorm" if gpt else "RmsNorm")]
        if gpt:
            ops += [("qkv_linear", "Linear")]
            for role in ("q", "k", "v"):
                ops += [(role + "_slice", "Slice"), (role + "_heads", "Reshape")]
            ops += [("key_append", "KvCacheAppend"), ("key_view", "CacheSuffix"),
                    ("value_append", "KvCacheAppend"), ("value_view", "CacheSuffix")]
        else:
            ops += [(role + "_linear", "Linear") for role in ("q", "k", "v")]
            ops += [(role + "_heads", "Reshape") for role in ("q", "k", "v")]
            if c["family"] == "qwen3":
                ops += [("q_norm", "RmsNorm"), ("k_norm", "RmsNorm")]
            ops += [("rope_q", "RotaryEmbedding"), ("rope_k", "RotaryEmbedding"),
                    ("key_append", "KvCacheAppend"), ("value_append", "KvCacheAppend"),
                    ("key_view", "CacheSuffix"), ("value_view", "CacheSuffix")]
        ops += [("attention_scores", "AttentionScores"), ("attention_scale", "AttentionScale"),
                ("causal_mask", "CausalMask"), ("softmax", "Softmax"),
                ("attention_values", "AttentionValues"), ("attention_hidden", "Reshape"),
                ("o_proj", "Linear"), ("attention_residual", "ResidualAdd"),
                ("post_norm", "LayerNorm" if gpt else "RmsNorm")]
        ops += ([("up_proj", "Linear"), ("gelu", "GeluTanh")] if gpt else
                [("gate_proj", "Linear"), ("up_proj", "Linear"),
                 ("silu", "Silu"), ("gated_multiply", "Multiply")])
        ops += [("down_proj", "Linear"), ("mlp_residual", "ResidualAdd")]
        result += [(f"layer.{layer}.{name}", kind) for name, kind in ops]
    return result + [("final_norm", "LayerNorm" if gpt else "RmsNorm"), ("last_hidden", "LastToken"),
                     ("output_head", "OutputHead"), ("token_selection", "GreedyTokenSelection"),
                     ("token_feedback", "TokenFeedback")]


def validate_graph(c, graph):
    fields(graph, "fixture_id operators")
    if graph["fixture_id"] != c["id"]:
        raise ValueError("graph fixture binding")
    strict_equal(blueprint(c), [(op["id"], op["operator"]) for op in graph["operators"]], "fixed-node-order")
    seen = {"input.tokens", "input.positions", "input.attention_mask", "input.sequence_lengths"}
    for index, op in enumerate(graph["operators"]):
        fields(op, "semantic_index id operator inputs declared_shape attributes")
        if type(op["semantic_index"]) is not int or op["semantic_index"] != index:
            raise ValueError("semantic index mismatch")
        if any(name not in seen for name in op["inputs"]):
            raise ValueError("non-topological or external state input")
        seen.add(op["id"])


def native_shape(op, sequence):
    shape = list(op["declared_shape"])
    if len(shape) == 3:
        shape[1] = sequence
    elif len(shape) == 4:
        shape[2] = sequence
        if op["operator"] in ("AttentionScores", "AttentionScale", "CausalMask", "Softmax"):
            shape[3] = sequence
    return shape


def source_role(op, bias=False):
    id_ = op["id"]
    if id_ == "token_lookup":
        return "embedding"
    if id_ == "position_lookup":
        return "positions"
    if id_ == "final_norm":
        return "final.beta" if bias else "final.gamma"
    if id_ == "output_head":
        return "head"
    base, suffix = id_.rsplit(".", 1)
    role = {"input_norm": "norm1", "post_norm": "norm2", "q_norm": "q_norm", "k_norm": "k_norm",
            "q_linear": "q", "k_linear": "k", "v_linear": "v", "o_proj": "o", "up_proj": "up",
            "down_proj": "down", "gate_proj": "gate"}[suffix]
    ending = ("beta" if bias else "gamma") if suffix.endswith("norm") else ("bias" if bias else "weight")
    return f"{base}.{role}.{ending}"


def execute_node(c, op, shape, operands, weights, token_buffer, prefix, tables):
    """Independent integer evaluation of one prescribed semantic operation."""
    kind, attr = op["operator"], op["attributes"]
    x = operands[0][1] if operands else None
    in_shape = operands[0][0] if operands else None
    diagnostic = Trace()  # Discard internal traces: not a native-export claim.
    if kind == "TokenLookup":
        w = rows(weights[source_role(op)], shape[-1])
        ids = list(range(len(token_buffer))) if op["id"] == "position_lookup" else token_buffer
        return flat([w[i] for i in ids])
    if kind in ("Linear", "OutputHead"):
        width = in_shape[-1]
        if op["id"].endswith(".qkv_linear"):
            base = op["id"].rsplit(".", 1)[0]
            w = [row for role in ("q", "k", "v") for row in rows(weights[f"{base}.{role}.weight"], width)]
            bias = [v for role in ("q", "k", "v") for v in weights[f"{base}.{role}.bias"]]
        else:
            role = "embedding" if kind == "OutputHead" and c["tied_head"] else source_role(op)
            w = rows(weights[role], width)
            bias = weights.get(source_role(op, True), [0] * len(w)) if kind != "OutputHead" else [0] * len(w)
        return flat(linear(rows(x, width), w, bias, diagnostic, "linear"))
    if kind in ("LayerNorm", "RmsNorm"):
        gamma = weights[source_role(op)]
        beta = weights.get(source_role(op, True), [0] * len(gamma))
        return flat(norm(rows(x, len(gamma)), gamma, beta, "layer" if kind == "LayerNorm" else "rms", diagnostic, "norm"))
    if kind in ("ResidualAdd", "Multiply"):
        y = operands[1][1]
        return [checked(a + b) if kind == "ResidualAdd" else checked(a * b) // ONE for a, b in zip(x, y)]
    if kind in ("GeluTanh", "Silu"):
        return flat(activation(rows(x, shape[-1]), tables["gelu" if kind == "GeluTanh" else "silu"], diagnostic, "activation"))
    if kind == "Slice":
        return flat([row[attr["start"]:attr["end"]] for row in rows(x, in_shape[-1])])
    if kind == "Reshape":
        if len(shape) == 4:
            _, heads, sequence, width = shape
            return [x[(p * heads + h) * width + j] for h in range(heads) for p in range(sequence) for j in range(width)]
        _, heads, sequence, width = in_shape
        return [x[(h * sequence + p) * width + j] for p in range(sequence) for h in range(heads) for j in range(width)]
    if kind == "RotaryEmbedding":
        _, heads, sequence, width = shape
        heads_ = [rows(x[h * sequence * width:(h + 1) * sequence * width], width) for h in range(heads)]
        return [v for head in rotary(heads_, rows(weights["rope.cos"], width), rows(weights["rope.sin"], width), diagnostic, "rotary") for row in head for v in row]
    if kind in ("KvCacheAppend", "CacheSuffix", "CausalMask", "TokenFeedback"):
        return list(x)  # Fresh full-prefix identity views; mask applied in softmax.
    if kind == "AttentionScores":
        _, heads, sequence, _ = shape
        width = in_shape[-1]
        y = operands[1][1]
        group = attr["group_size"]
        return [dot(x[(h * sequence + p) * width:(h * sequence + p + 1) * width],
                    y[((h // group) * sequence + k) * width:((h // group) * sequence + k + 1) * width]) // ONE
                for h in range(heads) for p in range(sequence) for k in range(sequence)]
    if kind == "AttentionScale":
        from oracle import external_tensor
        if attr["head_dim"] != c["dimensions"]["head_dim"]:
            raise ValueError("scale source binding")
        coefficient = external_tensor(c["attention_scale"], [1])[0]
        return [checked(v * coefficient) // ONE for v in x]
    if kind == "Softmax":
        from oracle import softmax
        _, heads, sequence, _ = shape
        return flat([softmax(x[(h * sequence + p) * sequence:(h * sequence + p + 1) * sequence],
                             min(p, prefix - 1), tables, diagnostic, f"softmax.{h}.{p}")
                     for h in range(heads) for p in range(sequence)])
    if kind == "AttentionValues":
        _, heads, sequence, width = shape
        y = operands[1][1]
        group = attr["group_size"]
        return [dot(x[(h * sequence + p) * sequence:(h * sequence + p + 1) * sequence],
                    [y[((h // group) * sequence + k) * width + j] for k in range(sequence)]) // ONE
                for h in range(heads) for p in range(sequence) for j in range(width)]
    if kind == "LastToken":
        width = shape[-1]
        return x[(prefix - 1) * width:prefix * width]
    if kind == "GreedyTokenSelection":
        return [x.index(max(x))]
    raise ValueError("operator not in independent semantic contract")


def source_encodings(c, weights):
    from oracle import external_tensor
    encoded = []
    for name, tensor in sorted({**c["tensors"], "attention_scale": c["attention_scale"]}.items()):
        values = external_tensor(tensor, [1]) if name == "attention_scale" else weights[name]
        encoded.append({"name": name, "dtype": tensor["dtype"], "shape": tensor["shape"],
                        "ieee_words": tensor["words"], "q12_values": values})
    return encoded


def run_semantic(c, graph):
    weights = validate_contract(c)
    validate_graph(c, graph)
    initial, selections = len(c["prefix"]), c["generation_count"]
    sequence = initial + selections - 1
    prefix_ids = list(c["prefix"])
    nodes, states, generated = [], [], []
    tables = load_tables()
    for selection in range(selections):
        prefix = initial + selection
        token_buffer = prefix_ids + [0] * (sequence - prefix)
        registers = {}
        for op in graph["operators"]:
            shape = native_shape(op, sequence)
            operands = [registers[id_] for id_ in op["inputs"] if id_ in registers]
            values = execute_node(c, op, shape, operands, weights, token_buffer, prefix, tables)
            if len(values) != prod(shape):
                raise ValueError("complete semantic tensor shape mismatch")
            for value in values:
                checked(value)
            registers[op["id"]] = (shape, values)
            nodes.append({"selection": selection, "semantic_index": op["semantic_index"], "id": op["id"],
                          "shape": shape, "fractional": 0 if op["operator"] in ("GreedyTokenSelection", "TokenFeedback") else 12,
                          "values": values})
        token = registers["token_feedback"][1][0]
        generated.append(token)
        after = prefix_ids + [token]
        states.append({"selection": selection, "active_prefix_ids_private": list(prefix_ids),
                       "input_token_buffer_private": token_buffer, "active_positions": list(range(prefix)),
                       "physical_positions_computed": list(range(sequence)), "selected_id_private": token,
                       "logical_prefix_after_selection_private": after, "persistent_kv": False, "released": False})
        prefix_ids = after
    return {"schema": RESULT, "fixture_id": c["id"], "profile": c["profile"],
            "input_sha256": digest(c), "semantic_graph_sha256": digest(graph),
            "source_encodings": source_encodings(c, weights), "nodes": nodes, "states": states,
            "final": {"recipient": "client-only", "token_ids": prefix_ids, "generated_ids": generated,
                      "selection_count": selections, "persistent_kv": False}}


def expected():
    inputs = read_json(HERE / "inputs.json")
    graphs = read_json(HERE / "semantic-graphs.json")
    if graphs["schema"] != "pllm.fusefss.semantic-graphs.v1" or len(graphs["cases"]) != len(inputs["cases"]):
        raise ValueError("fixed semantic graph bundle")
    return {"schema": SCHEMA, "results": [run_semantic(c, graph) for c, graph in zip(inputs["cases"], graphs["cases"])]}
