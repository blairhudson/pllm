"""Bounded header-only semantic demand screen; no checkpoint or live traffic."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "crates/pllm-models/src/lib.rs",
    "crates/pllm-compiler/src/decoder_runtime_schedule.rs",
    "crates/pllm-compiler/src/decoder_continuation.rs",
    "python/pllm/runtime/semantic_stages.py",
    "python/pllm/runtime/semantic_executor.py",
    "python/pllm/runtime/quantization.py",
    "python/pllm/modeling.py",
    "docs/research/aggregate-network.md",
    "scripts/probe_subtensor_demand_pruning.py",
    "tests/test_subtensor_demand_pruning_probe.py",
    "docs/research/subtensor-demand-pruning-reference.md",
)
ROWWISE = {"linear", "output_head", "rms_norm", "residual_add", "silu",
           "multiply", "gelu_tanh", "softcap"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def config(family="qwen2", *, small=False):
    # Public representative shape inputs, not checkpoint identity or tensor data.
    shapes = {"qwen2": (896, 4864, 24, 14, 2, 64),
              "qwen3": (1024, 3072, 28, 16, 8, 128),
              "llama": (2048, 8192, 16, 32, 8, 64)}
    h, m, layers, heads, kv, dim = (16, 32, 2, 4, 2, 4) if small else shapes[family]
    return dict(model_type=family, hidden_act="silu", tie_word_embeddings=False,
                attention_bias=False, attention_dropout=0.0, use_cache=True,
                use_sliding_window=False, max_window_layers=layers,
                hidden_size=h, intermediate_size=m, num_hidden_layers=layers,
                num_attention_heads=heads, num_key_value_heads=kv, head_dim=dim,
                vocab_size=64 if small else 151936, max_position_embeddings=4096,
                rms_norm_eps=1e-6, rope_theta=10000)


def lower(family, rows, *, small=False):
    from pllm import Model, lower_model
    from pllm.profiles import MaskedLinearCpu
    from pllm.protocols import MaskedLinear

    cfg = config(family, small=small)
    plan = lower_model(cfg, batch=1, max_input_tokens=rows, max_new_tokens=8)
    pipeline = MaskedLinearCpu(Model("semantic-demand-probe"),
                               linear=MaskedLinear(prefill_pruning="terminal"))
    return cfg, plan.to_dict(), plan.runtime_schedule(pipeline).to_dict()


def liveness(graph, *, extended=True, extra_roots=(), all_rows=False):
    """0=dead, 1=terminal query row, 2=all rows. Unknown consumers widen demand.

    Query-axis transfer is separate from key-axis reduction. No ID/path parsing.
    Caller must certify dense batch-one last-valid semantics; continuation uses
    all_rows until a separate consumer contract is admitted.
    """
    if graph.get("batch") != 1:
        raise ValueError("only certified batch-one query layout is supported")
    ops = {o["id"]: o for o in graph["operations"]}
    if len(ops) > 10000 or len(ops) != len(graph["operations"]):
        raise ValueError("unbounded or duplicate semantic graph")
    queue = deque([(graph["output"], 2)] + [(s["id"], 2) for s in graph["state_outputs"]]
                  + [(r, 2) for r in extra_roots])
    if all_rows:
        queue.extend((o["id"], 2) for o in graph["operations"] if o["operator"] == "linear")
    demand = {}
    while queue:
        key, wanted = queue.popleft()
        if key not in ops or demand.get(key, 0) >= wanted:
            continue
        demand[key] = wanted
        op = ops[key]
        kind, attrs, inputs = op["operator"], op["attributes"], op["inputs"]
        upstream = [2] * len(inputs)
        if (kind == "last_token" and not all_rows and attrs.get("axis") == 1
                and attrs.get("selection") == "last_valid"
                and len(inputs) == 2 and attrs.get("valid_lengths_input") == inputs[1]):
            upstream = [1] + [2] * (len(inputs) - 1)
        elif kind in ROWWISE:
            upstream = [wanted] * len(inputs)
        elif extended:
            if kind == "reshape" and attrs.get("layout") in {
                "batch_heads_sequence_feature", "batch_sequence_hidden"
            } and inputs[0] in ops and _row_preserving_reshape(ops[inputs[0]]["output_shape"],
                                                            op["output_shape"], attrs["layout"]):
                upstream = [wanted] * len(inputs)
            elif kind == "rotary_embedding" and attrs.get("position_policy") == "sequential_absolute":
                upstream = [wanted] + [2] * (len(inputs) - 1)
            elif kind in {"attention_scores", "attention_values"} and len(inputs) == 2:
                upstream = [wanted, 2]  # Never slice key/value reduction domain.
            elif kind in {"attention_scale", "causal_mask"}:
                upstream = [wanted] + [2] * (len(inputs) - 1)
            elif kind == "softmax" and attrs.get("axis") == -1:
                upstream = [wanted] * len(inputs)
        queue.extend(zip(inputs, upstream, strict=True))
    return demand


def _row_preserving_reshape(source, target, layout):
    if layout == "batch_heads_sequence_feature":
        return (len(source) == 3 and len(target) == 4 and source[0] == target[0] == 1
                and source[1] == target[2] and source[2] == target[1]*target[3])
    return (len(source) == 4 and len(target) == 3 and source[0] == target[0] == 1
            and source[2] == target[1] and source[1]*source[3] == target[2])


def numeric_admission(*, original_input_width, proposed_input_width,
                      same_quantized_input=True, same_scales_bias=True, same_f32_order=True):
    return (original_input_width == proposed_input_width and same_quantized_input
            and same_scales_bias and same_f32_order)


def directed_bodies(input_lanes, output_lanes):
    """One prepared execution: one body per directed link, four bytes per u32."""
    return {"Client->Inference": 4*input_lanes,
            "Inference->Client": 4*output_lanes,
            "Preparation->Inference": 4*output_lanes}


def cost(graph, schedule, demand, *, rows, decode_steps=7):
    ops = {o["id"]: o for o in graph["operations"]}
    total = {k: 0 for k in ("input_bytes", "reply_bytes", "correction_bytes", "integer_macs")}
    candidates = []
    comparator_macs = 0
    for step in schedule["steps"]:
        if step["executor"] != "remote_stage" or set(step["operators"]) != {"linear"}:
            continue
        source = ops[step["input_ids"][0]]
        width = source["output_shape"][-1]
        base_rows = 1 if step.get("terminal_row_only") else rows
        widths = [r["stage_width"] for r in step["outputs"]]
        wanted = [0 if demand.get(r["operation_id"], 0) == 0 else
                  1 if demand[r["operation_id"]] == 1 else rows for r in step["outputs"]]
        if any(n > base_rows for n in wanted):
            raise ValueError("candidate consumer contract exceeds terminal comparator")
        base_lanes = base_rows * sum(widths)
        lanes = sum(n * w for n, w in zip(wanted, widths, strict=True))
        # One prepared response; one input, one reply, one correction body.
        # No input savings unless every output slice loses that row.
        total["input_bytes"] += 4 * base_rows * width
        total["reply_bytes"] += 4 * base_lanes
        total["correction_bytes"] += 4 * base_lanes
        total["integer_macs"] += base_lanes * width
        comparator_macs += (base_rows + decode_steps) * sum(widths) * width
        if lanes < base_lanes:
            candidates.append(dict(order=step["order"], layer=step["layer"],
                slices=[dict(operation_id=r["operation_id"], offset=r["stage_offset"],
                             width=r["stage_width"], demanded_rows=n)
                        for r, n in zip(step["outputs"], wanted, strict=True)],
                saved_output_lanes=base_lanes-lanes,
                saved_reply_bytes=4*(base_lanes-lanes),
                saved_correction_bytes=4*(base_lanes-lanes),
                saved_input_bytes=4*(base_rows-max(wanted))*width,
                saved_integer_macs=(base_lanes-lanes)*width))
    # Add unchanged decode linear bodies; exclude lookup/head/control/artifacts.
    for step in schedule["steps"]:
        if step["executor"] == "remote_stage" and set(step["operators"]) == {"linear"}:
            width = ops[step["input_ids"][0]]["output_shape"][-1]
            total["input_bytes"] += 4*decode_steps*width
            output_lanes = decode_steps*sum(r["stage_width"] for r in step["outputs"])
            total["reply_bytes"] += 4*output_lanes
            total["correction_bytes"] += 4*output_lanes
    saved = sum(c["saved_reply_bytes"] + c["saved_correction_bytes"] + c["saved_input_bytes"]
                for c in candidates)
    directed = directed_bodies(total["input_bytes"]//4, total["reply_bytes"]//4)
    body = sum(directed.values())
    return dict(candidates=candidates, terminal_comparator_body_bytes=body,
                prepared_response_executions=1, terminal_comparator_directed_body_bytes=directed,
                directed_link_multiplicity={link: 1 for link in directed},
                incremental_arithmetic_upper_bound_bytes=saved,
                incremental_fraction_of_included_bodies=saved/body,
                saved_integer_macs=sum(c["saved_integer_macs"] for c in candidates),
                remote_body_mac_retention=1-sum(c["saved_integer_macs"] for c in candidates)/comparator_macs,
                new_client_weights_bytes=0, client_rss_gate="unmeasured; <=256 MiB not established",
                accounting="one prepared response; one u32 body per directed input/reply/correction link; excludes unchanged head/lookup and all framing/setup")


def quantized_linear(x, weight, scales, bias, *, terminal=False):
    """Independent array arithmetic: full input-width scale and integer reduction."""
    work = x[:, -1:] if terminal else x
    a = np.max(np.abs(work), axis=-1, keepdims=True)
    a = np.where(a > 0, a/np.float32(127), np.float32(1)).astype(np.float32)
    q = np.clip(np.rint(work/a), -127, 127).astype(np.int64)
    out = (q @ weight.astype(np.int64).T).astype(np.float32)
    out = out * a * scales + bias  # Keep f32 multiply order, then bias.
    if terminal:
        expanded = np.zeros(x.shape[:-1] + (len(weight),), np.float32)
        expanded[:, -1:] = out
        return expanded
    return out


def array_oracle(graph, demand, *, seed=7):
    """Independent generated graph interpreter, synthetic f32 operator contract.

    Not runtime numeric parity: synthetic RoPE coefficients and NumPy kernels,
    no compiled checkpoint execution. Compares all head logits and active KV.
    """
    rng = np.random.default_rng(seed)
    values = {}
    states = {}
    head = None
    for op in graph["operations"]:
        key, kind, attrs = op["id"], op["operator"], op["attributes"]
        inputs = [values[i] for i in op["inputs"] if i in values]
        shape = tuple(op["output_shape"])
        if kind == "token_lookup":
            y = rng.normal(size=shape).astype(np.float32)
        elif kind in {"linear", "output_head"}:
            x = inputs[0]
            w = rng.integers(-8, 9, (shape[-1], x.shape[-1]), dtype=np.int8)
            s = rng.uniform(.01, .08, shape[-1]).astype(np.float32)
            b = rng.uniform(-.1, .1, shape[-1]).astype(np.float32) if attrs.get("bias") else np.zeros(shape[-1], np.float32)
            y = quantized_linear(x, w, s, b, terminal=kind == "linear" and demand.get(key) == 1)
            if kind == "output_head":
                head = y
        elif kind == "rms_norm":
            x = inputs[0]
            y = x / np.sqrt(np.mean(x*x, axis=-1, keepdims=True, dtype=np.float32) + np.float32(attrs["epsilon"]))
        elif kind == "reshape":
            x = inputs[0]
            if attrs["layout"] == "batch_heads_sequence_feature":
                y = x.reshape(shape[0], shape[2], shape[1], shape[3]).transpose(0, 2, 1, 3)
            else:
                y = x.transpose(0, 2, 1, 3).reshape(shape)
        elif kind == "rotary_embedding":
            x = inputs[0]
            half = x.shape[-1]//2
            angle = np.arange(x.shape[-2], dtype=np.float32)[None, None, :, None]*np.float32(.1)
            c, s = np.cos(angle), np.sin(angle)
            y = np.concatenate((x[..., :half]*c-x[..., half:]*s,
                                x[..., half:]*c+x[..., :half]*s), axis=-1)
        elif kind == "kv_cache_append":
            y = inputs[0].copy()
            states[key] = y
        elif kind == "cache_suffix":
            y = inputs[0]
        elif kind == "attention_scores":
            q, k = inputs
            k = np.repeat(k, attrs["group_size"], axis=1)
            y = q @ k.swapaxes(-1, -2)
        elif kind == "attention_scale":
            y = inputs[0] / np.float32(np.sqrt(attrs["head_dim"]))
        elif kind == "causal_mask":
            x = inputs[0]
            y = np.where(np.triu(np.ones(x.shape[-2:], bool), 1), -np.inf, x)
        elif kind == "softmax":
            x = inputs[0]
            e = np.exp(x-np.max(x, axis=-1, keepdims=True))
            y = e / np.sum(e, axis=-1, keepdims=True, dtype=np.float32)
        elif kind == "attention_values":
            y = inputs[0] @ np.repeat(inputs[1], attrs["group_size"], axis=1)
        elif kind == "residual_add":
            y = inputs[0] + inputs[1]
        elif kind == "silu":
            y = inputs[0] / (np.float32(1)+np.exp(-inputs[0]))
        elif kind == "multiply":
            y = inputs[0]*inputs[1]
        elif kind == "last_token":
            y = inputs[0][:, -1:]
        elif kind == "greedy_token_selection":
            y = np.argmax(inputs[0], axis=-1)
        elif kind == "token_feedback":
            y = inputs[0]
        else:
            raise ValueError(f"unsupported independent oracle operator {kind}")
        values[key] = y
    return head, states


def run(rows=39, outputs=8):
    if not 2 <= rows <= 512 or not 1 <= outputs <= 8:
        raise ValueError("bounded rows 2..512, outputs 1..8")
    results = []
    for family in ("qwen2", "qwen3", "llama"):
        cfg, plan, schedule = lower(family, rows)
        graph = plan["prefill"]
        d = liveness(graph)
        entry = dict(family=family, representative_config=cfg, graph_digest=digest(graph),
                     schedule_digest=digest(schedule), **cost(graph, schedule["prefill"], d,
                                                           rows=rows, decode_steps=outputs-1))
        small_cfg, small_plan, small_schedule = lower(family, 7, small=True)
        small = small_plan["prefill"]
        base = liveness(small, extended=False)
        proposed = liveness(small)
        samples = []
        for seed in (7, 19, 41):
            full, kv = array_oracle(small, {}, seed=seed)
            old, old_kv = array_oracle(small, base, seed=seed)
            new, new_kv = array_oracle(small, proposed, seed=seed)
            samples.append(dict(seed=seed, logits_equal=bool(np.array_equal(full, old) and np.array_equal(full, new)),
                all_active_kv_equal=all(np.array_equal(kv[k], old_kv[k]) and np.array_equal(kv[k], new_kv[k]) for k in kv),
                logits_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                kv_sha256=hashlib.sha256(b"".join(kv[k].tobytes() for k in sorted(kv))).hexdigest()))
        entry["independent_array_oracle"] = dict(config=small_cfg, samples=samples,
                                                  numeric_contract="synthetic NumPy f32; not production exactness")
        entry["compiler_comparator_matches_independent_liveness"] = all(
            bool(s.get("terminal_row_only")) == all(base.get(k) == 1 for k in s["operation_ids"])
            for s in small_schedule["prefill"]["steps"] if set(s["operators"]) == {"linear"})
        entry["decode_saved_lanes"] = 0
        entry["continuation_admitted_saved_lanes"] = 0
        entry["continuation_reason"] = "all-row logits/reuse consumers; separate certification required"
        results.append(entry)
    loaded_native = {name: dict(filename=Path(module.__file__).name,
                               sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
                     for name, module in sorted(sys.modules.items())
                     if name.startswith("pllm.") and getattr(module, "__file__", "")
                     and Path(module.__file__).suffix in {".so", ".dylib"}}
    return dict(schema="pllm.research.subtensor_demand_pruning.v1", rows=rows, outputs=outputs,
                loaded_native_sha256=loaded_native,
                source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                results=results, decision="reject production promotion: small incremental payload upper bound",
                full_compiled_execution_performed=False, checkpoint_tensors_loaded=False,
                whole_response_traffic_measured=False, client_rss_measured=False,
                admission_required=["public dense last-valid row contract and all consumer/state roots",
                    "query-axis-only transfer; full K/V and key-axis reductions",
                    "same input quantization domain, weight scales, biases and f32 order",
                    "new authenticated output-slice frame namespace and phase bindings",
                    "per-slice one-use material admission, cancellation/burn ledger",
                    "new public offset/width/row-demand/numeric metadata digest"],
                forbidden_consumers=["all-row logits/hidden export", "retained Q/attention outputs",
                    "unknown layout/state/custom operator", "query-axis softmax/norm/reduction",
                    "uncertified continuation/reuse/verification consumers"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=39)
    parser.add_argument("--outputs", type=int, default=8)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    result = run(args.rows, args.outputs)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
