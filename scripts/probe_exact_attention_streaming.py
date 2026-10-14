"""Bounded research-only attention lifetime probe; no decoder authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.metrics import BuildRef

LAYOUT = "batch_kv_heads_sequence_feature"
PROFILES = ("legacy", "dense", "prefix_f32", "bf16")


def identity():
    import pllm
    from pllm.runtime import semantic_attention, semantic_numeric, causal_reduction
    modules = (semantic_attention, semantic_numeric, causal_reduction,
               sys.modules[SemanticDecoderRuntime.__module__])
    return {"build": BuildRef.capture().to_spec(), "python": sys.executable,
            "pllm": pllm.__file__, "numpy": np.__version__,
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "operators": {m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(
                Path(m.__file__).read_bytes()).hexdigest()} for m in modules}}


def operator_runtime(profile, position):
    # Invoke the installed executor's actual local operators without loading a model.
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._causal_reduction = "prefix_f32" if profile == "prefix_f32" else None
    runtime.position = int(position)
    runtime._retained = None
    return runtime


def stages(query, keys, cache, positions, group, profile):
    runtime = operator_runtime(profile, positions[0])
    def local(kind, inputs, attrs):
        op = {"operator": kind, "inputs": list(inputs), "attributes": attrs}
        return runtime._local(op, inputs, {}, {})
    attrs = {"group_size": group}
    if profile in {"dense", "bf16"}:
        attrs["key_layout"] = LAYOUT
    scores = local("attention_scores", {"q": query, "k": keys}, attrs)
    scaled = local("attention_scale", {"s": scores}, {"head_dim": query.shape[-1]})
    del scores
    masked = local("causal_mask", {"s": scaled, "p": positions}, {})
    del scaled
    softattrs = {"axis": -1, "output_dtype": "bfloat16", "compute_dtype": "float32"} if profile == "bf16" else {}
    probabilities = local("softmax", {"s": masked}, softattrs)
    del masked
    attrs = {"group_size": group}
    if profile in {"dense", "bf16"}:
        attrs["value_layout"] = LAYOUT
    output = local("attention_values", {"p": probabilities, "v": cache}, attrs)
    return output


def stream(query, keys, cache, positions, group, profile, *, query_rows=1, kv_heads=1):
    if profile not in PROFILES or query_rows < 1 or kv_heads < 1:
        raise ValueError("invalid profile or tile geometry")
    if query.shape[1] != keys.shape[1] * group or keys.shape != cache.shape:
        raise ValueError("invalid GQA fixture")
    output = np.empty(query.shape, np.float32)
    for head in range(0, keys.shape[1], kv_heads):
        end = min(head + kv_heads, keys.shape[1])
        for row in range(0, query.shape[2], query_rows):
            stop = min(row + query_rows, query.shape[2])
            output[:, head * group:end * group, row:stop] = stages(
                query[:, head * group:end * group, row:stop], keys[:, head:end],
                cache[:, head:end], positions[row:stop], group, profile)
    return output


def fixture(seed=7, *, rows=9, capacity=23, width=33, heads=3, group=2,
            layout="c", start=0):
    if not (1 <= rows <= 384 and 0 <= start and rows + start <= capacity <= 4096
            and 1 <= width <= 256 and 1 <= heads <= 16 and 1 <= group <= 8
            and heads * group * rows * capacity <= 16_777_216):
        raise ValueError("fixture exceeds bounded research domain")
    rng = np.random.default_rng(seed)
    query = rng.normal(size=(1, heads * group, rows, width)).astype(np.float32)
    keys = rng.normal(size=(1, heads, capacity, width)).astype(np.float32)
    cache = rng.normal(size=keys.shape).astype(np.float32)
    # Model fixed-capacity zeros; preserve all slots through every default reduction.
    keys[:, :, start + rows:] = 0
    cache[:, :, start + rows:] = 0
    if layout == "sequence_major":
        query, keys, cache = [np.ascontiguousarray(a.transpose(0, 2, 1, 3)).transpose(
            0, 2, 1, 3) for a in (query, keys, cache)]
    elif layout == "reverse":
        query, keys, cache = [a[..., ::-1] for a in (query, keys, cache)]
    elif layout == "fortran":
        query, keys, cache = [np.asfortranarray(a) for a in (query, keys, cache)]
    elif layout != "c":
        raise ValueError("unknown fixture layout")
    return query, keys, cache, np.arange(start, start + rows, dtype=np.int64)


def bits_equal(left, right):
    return left.shape == right.shape and np.array_equal(left.view(np.uint32), right.view(np.uint32))


def owner_bytes(*arrays):
    owners = {}
    for array in arrays:
        while isinstance(array.base, np.ndarray):
            array = array.base
        owners[id(array)] = array.nbytes
    return sum(owners.values())


def accounting(query, keys, cache, positions, group, profile, query_rows, kv_heads):
    h = min(kv_heads, keys.shape[1]) * group
    r = min(query_rows, query.shape[2])
    t, d = keys.shape[2:]
    slab = 4 * h * r * t
    # Conservative numeric allocation envelope, not measured RSS. Includes softmax
    # centered/exp/division/BF16 conversions, masks and validation temporaries.
    scratch = 8 * slab + 8 * r * t + 8 * t + 16 * h * r + 4 * h * r * d
    scratch += max(query.nbytes // 4, keys.nbytes // 4, cache.nbytes // 4)
    if profile == "prefix_f32":
        scratch += 8 * t * d  # old/new products can overlap during loop assignment
    return {"input_unique_owner_bytes": owner_bytes(query, keys, cache, positions),
            "complete_output_bytes": query.nbytes,
            "conservative_numeric_scratch_bytes": scratch,
            "score_slab_bytes": slab,
            "full_score_bytes": query.shape[1] * query.shape[2] * t * 4,
            "attention_only_envelope_bytes": owner_bytes(query, keys, cache, positions) + query.nbytes + scratch,
            "exclusions": ["imports/interpreter", "other decoder layers and KV", "weights",
                           "remote exchange owners", "native allocator overhead", "filesystem cache"],
            "is_whole_client_admission": False}


def correctness():
    records = []
    for profile in PROFILES:
        for layout in ("c", "sequence_major", "reverse", "fortran"):
            for start, rows in ((0, 9), (8, 1), (5, 4)):
                args = fixture(layout=layout, start=start, rows=rows)
                q, k, v, p = args
                oracle = stages(*args, 2, profile)
                before = [a.copy() for a in args]
                for qr, kh in ((1, 1), (3, 2), (7, 3)):
                    candidate = stream(*args, 2, profile, query_rows=qr, kv_heads=kh)
                    records.append({"profile": profile, "layout": layout, "start": start,
                                    "rows": rows, "query_rows": qr, "kv_heads": kh,
                                    "bitwise": bits_equal(oracle, candidate),
                                    "changed_words": int(np.count_nonzero(oracle.view(np.uint32) != candidate.view(np.uint32)))})
                assert all(bits_equal(a, b) for a, b in zip(args, before))
    return {"identity": identity(), "cases": records,
            "all_bitwise": all(r["bitwise"] for r in records),
            "counterexamples": counterexamples(),
            "owner_example": accounting(*fixture(), 2, "dense", 1, 1),
            "scope": "synthetic local attention only; no whole decoder, remote MAC or timing claim"}


def counterexamples():
    records = []
    for kind in ("contiguous_copy", "truncate_zero_capacity"):
        q, k, v, p = fixture(0, layout="reverse" if kind == "contiguous_copy" else "c")
        original = stages(q, k, v, p, 2, "dense")
        if kind == "contiguous_copy":
            changed = stages(*(np.ascontiguousarray(a) for a in (q, k, v)), p, 2, "dense")
        else:
            changed = stages(q, k[:, :, :9], v[:, :, :9], p, 2, "dense")
        records.append({"transformation": kind, "seed": 0,
                        "changed_words": int(np.count_nonzero(original.view(np.uint32) != changed.view(np.uint32))),
                        "original_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
                        "changed_sha256": hashlib.sha256(changed.tobytes()).hexdigest()})
    q, k, v, p = fixture(0)
    default = stages(q, k, v, p, 2, "dense")
    prefix = stages(q, k, v, p, 2, "prefix_f32")
    records.append({"transformation": "replace_dense_with_prefix_f32", "seed": 0,
                    "changed_words": int(np.count_nonzero(default.view(np.uint32) != prefix.view(np.uint32)))})
    return records


def worker(payload):
    args = fixture(**payload["fixture"])
    group = payload["fixture"].get("group", 2)
    qr, kh = payload["query_rows"], payload["kv_heads"]
    profile = payload["profile"]
    begin_cpu, begin = time.process_time(), time.perf_counter()
    result = stages(*args, group, profile) if payload["arm"] == "control" else stream(
        *args, group, profile, query_rows=qr, kv_heads=kh)
    cpu, elapsed = time.process_time() - begin_cpu, time.perf_counter() - begin
    return {"identity": identity(), "payload": payload, "cpu_seconds": cpu,
            "wall_seconds": elapsed, "process_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "output_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
             "owners": accounting(*args, group, profile, qr if payload["arm"] != "control" else args[0].shape[2],
                                 kh if payload["arm"] != "control" else args[1].shape[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--measure", action="store_true", help="parent-only serial fresh process screen")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=33)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--query-rows", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--heads", type=int, default=3, help="Complete fixture KV-head count")
    parser.add_argument("--group", type=int, default=2, help="Query heads per KV head")
    parser.add_argument("--layout", choices=("c", "sequence_major", "reverse", "fortran"), default="c")
    parser.add_argument("--replicates", type=int, default=1, help="One to five alternating fresh pairs")
    parser.add_argument("--profile", choices=PROFILES, default="dense")
    options = parser.parse_args()
    if options.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if options.measure:
        from frontier_probe_support import fresh_workers
        if options.query_rows < 1 or options.kv_heads < 1 or not 1 <= options.replicates <= 5:
            parser.error("tile geometry must be positive and replicates must be 1..5")
        geometry = {"rows": options.rows, "capacity": options.capacity,
                    "width": options.width, "heads": options.heads,
                    "group": options.group, "layout": options.layout}
        args = fixture(**geometry)
        envelopes = [accounting(*args, options.group, options.profile, qr, kh)["attention_only_envelope_bytes"]
                     for qr, kh in ((options.rows, args[1].shape[1]), (options.query_rows, options.kv_heads))]
        del args
        with fresh_workers(Path(__file__).resolve(), memory_bytes=(256 << 20) + max(envelopes)) as (run, guard, host):
            payload = {"fixture": geometry, "query_rows": options.query_rows,
                        "kv_heads": options.kv_heads, "profile": options.profile}
            arms = []
            for replicate in range(options.replicates):
                order = ("control", "candidate") if replicate % 2 == 0 else ("candidate", "control")
                arms.extend(run(payload | {"arm": arm, "replicate": replicate}) for arm in order)
            if any(arm["identity"] != arms[0]["identity"] for arm in arms):
                raise RuntimeError("candidate/control operator or build identity changed")
            if any(arm["output_sha256"] != arms[0]["output_sha256"] for arm in arms):
                raise RuntimeError("candidate/control attention output bits differ")
            result = {"schema": "pllm.attention_lifetime_measurement.v1", "build": run.build, "arms": arms,
                       "replicates": options.replicates, "matching_outputs": True,
                       "observed_swap_growth_bytes": guard.maximum_swap_growth_bytes,
                       "qualification": "isolated synthetic attention, not whole-client tier evidence"}
    else:
        result = correctness()
    text = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
