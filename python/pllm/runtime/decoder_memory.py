"""Conservative active-array estimates for an already admitted native schedule.

This inspects metadata, not tensor values. It does not change allocation or claim
RSS savings. Unknown operators/state layouts keep the unreleased-output estimate.
"""
from __future__ import annotations

from collections import Counter
import math

from .rotary_coefficients import rotary_coefficient_bound


_VIEWS = {"reshape", "permute", "slice", "last_token", "token_feedback",
          "cache_suffix", "kv_cache_append", "causal_mask"}
_OPERATORS = _VIEWS | {"token_lookup", "linear", "output_head", "rms_norm",
    "rotary_embedding", "attention_scores", "attention_scale", "softmax",
    "attention_values", "residual_add", "silu", "multiply", "greedy_token_selection"}


def _bytes(shape):
    if not isinstance(shape, (list, tuple)) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("invalid live tensor geometry")
    result = 8 * math.prod(shape)
    if result > (1 << 63) - 1:
        raise ValueError("live tensor geometry overflow")
    return result


def _phase_working_bytes(graph, schedule):
    """Retain whole allocation roots for views and grouped projection slices."""
    operations = {op["id"]: op for op in graph["operations"]}
    remaining = Counter(name for step in schedule["steps"] for name in step["input_ids"])
    roots, values, references, sizes = {}, {}, Counter(), {}
    sequence = graph["query_sequence"]
    external = {"input.tokens": [sequence], "input.positions": [sequence],
                "input.sequence_lengths": [1],
                "input.attention_mask": [graph.get("key_value_sequence", sequence)]}
    external.update({row["id"]: row["shape"] for row in graph["state_inputs"]})

    def retain(name, owners, size):
        if name in values:
            raise ValueError("live tensor is produced twice")
        values[name], sizes[name] = owners, size
        references.update(owners)

    def release(name):
        for owner in values.pop(name):
            references[owner] -= 1
            if references[owner] == 0:
                del references[owner]
                del roots[owner]

    for name, shape in external.items():
        size = _bytes(shape)
        roots[name] = size
        retain(name, frozenset((name,)), size)
    logits = {op["inputs"][0] for op in operations.values()
              if op["operator"] == "greedy_token_selection"}
    if len(logits) != 1:
        raise ValueError("live schedule has no unique logits source")
    peak, largest_step = sum(roots.values()), 0
    for index, step in enumerate(schedule["steps"]):
        inputs = step["input_ids"]
        if any(name not in values for name in inputs):
            raise ValueError("live schedule input is unavailable")
        outputs = step["outputs"]
        output_bytes = sum(_bytes(row["output_shape"]) for row in outputs)
        # Grouped remote outputs slice one allocation. A local operation can
        # allocate even when its common implementation is a view (reshape/BF16).
        owner = ("step", index)
        roots[owner] = output_bytes
        for row in outputs:
            name = row["operation_id"]
            owners = {owner}
            if operations[name]["operator"] in _VIEWS:
                for source in inputs:
                    owners.update(values[source])
            retain(name, frozenset(owners), _bytes(row["output_shape"]))
        step_bytes = output_bytes + sum(sizes[name] for name in inputs)
        # Eight four-byte arrays per declared input/output element. Includes
        # f64 reductions, quantization and Python/native conversion scratch.
        peak = max(peak, sum(roots.values()) + 4 * step_bytes)
        largest_step = max(largest_step, step_bytes)
        for name in inputs:
            remaining[name] -= 1
            if remaining[name] == 0 and name not in logits:
                release(name)
    # Loop-local input/output/combined references can outlive dictionary entries.
    return peak + 2 * largest_step


def decoder_memory(plan, schedule):
    graphs = (plan.prefill, plan.decode)
    rotary = max(rotary_coefficient_bound(graph) for graph in graphs)
    legacy = sum(_bytes(op["output_shape"]) for graph in graphs for op in graph["operations"])
    qualified = all(op["operator"] in _OPERATORS for graph in graphs for op in graph["operations"])
    states = {}
    for graph in graphs:
        for row in (*graph["state_inputs"], *graph["state_outputs"]):
            shape = row["shape"]
            if (row["kind"] not in {"key", "value"} or len(shape) != 4 or shape[0] != 1
                    or shape[2] != row["maximum_sequence"] or type(row.get("layer")) is not int):
                qualified = False
                continue
            identity = (row["layer"], row["kind"])
            states[identity] = max(states.get(identity, 0),
                4 * shape[1] * shape[3] * max(64, 1 << (shape[2] - 1).bit_length()))
    qualified &= bool(states) and all((layer, kind) in states for layer, _ in states for kind in ("key", "value"))
    if not qualified:
        return {"mode": "unreleased_outputs", "working_bytes": legacy + rotary,
                "state_bytes": 0, "legacy_working_bytes": legacy,
                "rotary_coefficient_bytes": rotary}
    phases = schedule.to_dict()
    working = max(_phase_working_bytes(graph, phases[phase])
                  for phase, graph in zip(("prefill", "decode"), graphs, strict=True))
    # Geometric KV capacity, growth overlap, restored state, completed-response
    # snapshot and prefix-cache qualification temporaries. Persistent prefix-cache
    # payload capacity is charged separately by the caller. Benchmark response
    # snapshots must be retired after each run; arbitrary SDK history is unbounded.
    state = 6 * sum(states.values())
    return {"mode": "full_kv_live_roots_v1", "working_bytes": working + rotary,
            "state_bytes": state, "legacy_working_bytes": legacy,
            "rotary_coefficient_bytes": rotary}
