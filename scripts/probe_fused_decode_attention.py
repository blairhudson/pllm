"""Single-query exclusive scratch reuse; synthetic research, not SDK authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
from pllm.runtime import causal_reduction
from pllm.runtime.semantic_attention import (
    mask_causal_scores, semantic_attention_scores, semantic_attention_values,
)
from pllm.runtime.semantic_executor import SemanticDecoderRuntime

LAYOUT = "batch_kv_heads_sequence_feature"
PROFILES = ("dense", "prefix_f32")
MAX_ITERATIONS = 1024
MAX_CHILD_PRODUCTS = 2**29
MAX_COHORT_PRODUCTS = 2**32
MAX_ADMISSION_BYTES = 512 * 2**20


def validate(q, k, v, p, group, profile):
    if (profile not in PROFILES or type(group) is not int or not 1 <= group <= 8
            or any(type(a) is not np.ndarray or a.dtype != np.float32 or a.ndim != 4
                   for a in (q, k, v))
            or q.shape[:1] != (1,) or q.shape[2] != 1
            or k.shape[0] != 1 or k.shape != v.shape
            or not 1 <= k.shape[1] <= 16 or q.shape[1] != k.shape[1] * group
            or not 1 <= k.shape[2] <= 4096 or not 1 <= k.shape[3] <= 256
            or q.shape[3] != k.shape[3] or p.dtype != np.int64 or p.shape != (1,)
            or not 0 <= p[0] < k.shape[2]
            or any(not np.all(np.isfinite(a)) for a in (q, k, v))):
        raise ValueError("outside bounded single-query full-capacity contract")


def baseline(q, k, v, p, group, profile):
    """Actual installed runtime stage dispatch, without model loading."""
    validate(q, k, v, p, group, profile)
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime.position = int(p[0])
    runtime._causal_reduction = "prefix_f32" if profile == "prefix_f32" else None
    runtime._retained = None

    def local(kind, operands, attrs):
        return runtime._local({"operator": kind, "inputs": list(operands),
                               "attributes": attrs}, operands, {}, {})

    s = local("attention_scores", {"q": q, "k": k},
              {"group_size": group, "key_layout": LAYOUT})
    s = local("attention_scale", {"s": s}, {"head_dim": q.shape[-1]})
    s = local("causal_mask", {"s": s, "p": p}, {})
    s = local("softmax", {"s": s}, {})
    return local("attention_values", {"s": s, "v": v},
                 {"group_size": group, "value_layout": LAYOUT})


def oracle(q, k, v, p, group, profile):
    """Independent original primitive orchestration: no candidate or _local."""
    if profile == "prefix_f32":
        s = causal_reduction.scores(q, k, p, group)
    else:
        s = semantic_attention_scores(q, k, group=group, layout=LAYOUT)
    s = s * float(1.0 / math.sqrt(q.shape[-1]))
    s = mask_causal_scores(s, p, {})
    if profile == "prefix_f32":
        return causal_reduction.values(causal_reduction.softmax(s), v, p, group)
    centered = s - s.max(axis=-1, keepdims=True)
    exponential = np.exp(centered).astype(np.float32)
    probabilities = exponential / exponential.sum(axis=-1, keepdims=True)
    return semantic_attention_values(probabilities, v, group=group, layout=LAYOUT)


def fused(q, k, v, p, group, profile, *, observe=None):
    """No head/key tiling. Reuse only private fresh operator output storage."""
    validate(q, k, v, p, group, profile)
    s = (causal_reduction.scores(q, k, p, group) if profile == "prefix_f32" else
         semantic_attention_scores(q, k, group=group, layout=LAYOUT))
    if any(np.shares_memory(s, a) for a in (q, k, v)) or not s.flags.writeable:
        raise ValueError("score storage is not exclusive")
    if observe:
        observe("scores", s)
    np.multiply(s, float(1.0 / math.sqrt(q.shape[-1])), out=s)
    if not np.all(np.isfinite(s)):
        raise ValueError("scaled scores exceed numeric domain")
    # Single-query ordinary full_causal: no padding/window/retained policies.
    # Future capacity remains present for dense sum and value reductions.
    s[..., int(p[0]) + 1:] = np.float32(-np.inf)
    if observe:
        observe("masked", s)
    if profile == "prefix_f32":
        probabilities = causal_reduction.softmax(s)
        del s
        if observe:
            observe("probabilities", probabilities)
        return causal_reduction.values(probabilities, v, p, group)
    maximum = s.max(axis=-1, keepdims=True)
    np.subtract(s, maximum, out=s)
    del maximum
    np.exp(s, out=s)
    denominator = s.sum(axis=-1, keepdims=True)
    np.divide(s, denominator, out=s)
    del denominator
    if observe:
        observe("probabilities", s)
    return semantic_attention_values(s, v, group=group, layout=LAYOUT)


def fixture(*, seed=0, capacity=23, width=33, heads=3, group=2,
            position=8, layout="c", extreme=False):
    if not (1 <= capacity <= 4096 and 1 <= width <= 256 and 1 <= heads <= 16
            and 1 <= group <= 8 and 0 <= position < capacity):
        raise ValueError("fixture exceeds bounded domain")
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(1, heads * group, 1, width)).astype(np.float32)
    k = rng.normal(size=(1, heads, capacity, width)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    if extreme:
        q *= np.float32(32)
        k *= np.float32(32)
        v *= np.float32(1e15)
        q[..., ::7] = np.float32(-0.0)
        v[..., ::5] = np.float32(-0.0)
    k[:, :, position + 1:] = 0
    v[:, :, position + 1:] = 0
    if layout == "sequence_major":
        q, k, v = [np.ascontiguousarray(a.transpose(0, 2, 1, 3)).transpose(0, 2, 1, 3)
                   for a in (q, k, v)]
    elif layout == "reverse":
        q, k, v = [a[..., ::-1] for a in (q, k, v)]
    elif layout == "fortran":
        q, k, v = [np.asfortranarray(a) for a in (q, k, v)]
    elif layout == "capacity_view":
        # Logical capacity is itself a view into larger retained owners.
        arrays = []
        for a in (q, k, v):
            owner = np.zeros((*a.shape[:-2], a.shape[-2] + 11, width), np.float32)
            owner[..., :a.shape[-2], :] = a
            arrays.append(owner[..., :a.shape[-2], :])
        q, k, v = arrays
    elif layout != "c":
        raise ValueError("unknown layout")
    return q, k, v, np.array([position], np.int64)


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        a.view(np.uint32), b.view(np.uint32))


def owner(a):
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def accounting(q, k, v, p, group, profile, arm):
    owners = {id(owner(a)): owner(a).nbytes for a in (q, k, v, p)}
    slab = 4 * q.shape[1] * k.shape[2]
    # Conservative simultaneous workspace; NOT exact allocation tracing/RSS.
    scratch = (8 if arm == "baseline" else 2 if profile == "prefix_f32" else 1) * slab
    scratch += 16 * q.shape[1] + 16 * k.shape[2]
    scratch += max(slab // 4, *(a.size for a in (q, k, v)))  # finite-validation booleans
    if profile == "prefix_f32":
        scratch += 8 * k.shape[2] * k.shape[3]  # overlapping product owners
    return {"input_unique_owner_bytes": sum(owners.values()),
            "logical_operand_bytes": sum(a.nbytes for a in (q, k, v, p)),
            "operand_views": [a.base is not None for a in (q, k, v, p)],
            "complete_output_bytes": q.nbytes, "full_score_bytes": slab,
            "conservative_scratch_bytes": scratch,
            "attention_only_envelope_bytes": sum(owners.values()) + q.nbytes + scratch,
            "is_whole_client_admission": False,
            "exclusions": ["interpreter/imports", "allocator/thread stacks", "weights",
                           "other layers/KV", "remote/transport", "preparation", "filesystem cache"]}


def identity():
    import pllm
    from pllm.metrics import BuildRef
    from pllm.runtime import semantic_attention
    modules = (semantic_attention, causal_reduction, sys.modules[SemanticDecoderRuntime.__module__])
    root = Path(__file__).resolve().parents[1]
    related = ("tests/test_fused_decode_attention_probe.py",
               "docs/research/fused-decode-attention-reference.md",
               "python/pllm/runtime/attention_tiling.py",
               "scripts/probe_exact_attention_streaming.py")
    return {"python": sys.executable, "pllm": pllm.__file__, "numpy": np.__version__,
            "build": BuildRef.capture().to_spec(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "related_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in related if (root / name).is_file()},
            "operators": {m.__name__: {"path": m.__file__, "sha256": hashlib.sha256(
                Path(m.__file__).read_bytes()).hexdigest()} for m in modules}}


def correctness():
    records = []
    for profile in PROFILES:
        for layout in ("c", "sequence_major", "reverse", "fortran", "capacity_view"):
            for group in (1, 2, 4):
                for position, extreme in ((0, False), (8, False), (22, False), (8, True)):
                    args = fixture(group=group, position=position, layout=layout, extreme=extreme)
                    before = [a.copy() for a in args]
                    expected = oracle(*args, group, profile)
                    actual = fused(*args, group, profile)
                    control = baseline(*args, group, profile)
                    records.append({"profile": profile, "layout": layout, "group": group,
                                    "position": position, "extreme": extreme,
                                    "bitwise": bits_equal(actual, expected),
                                    "baseline_bitwise": bits_equal(control, expected),
                                    "inputs_unchanged": all(bits_equal(a, b) for a, b in zip(args, before)),
                                    "changed_words": int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))),
                                    "output_sha256": hashlib.sha256(actual.tobytes()).hexdigest()})
    return {"identity": identity(), "cases": records,
            "all_bitwise": all(r["bitwise"] and r["baseline_bitwise"] and r["inputs_unchanged"] for r in records),
            "owners": {arm: accounting(*fixture(layout="capacity_view"), 2, "prefix_f32", arm)
                       for arm in ("baseline", "fused")},
            "default_measurement_plans": {profile: measurement_plan(
                {"capacity": 1024, "position": 287, "width": 64, "heads": 2,
                 "group": 7, "layout": "sequence_major"}, profile, 256, 3)
                for profile in PROFILES},
            "gain": None, "scope": "synthetic only; timing, complete client and traffic gates missing"}


def measurement_plan(geometry, profile, iterations, replicates):
    """Allocation-free validation and conservative fresh-process admission."""
    names = ("capacity", "position", "width", "heads", "group")
    if (profile not in PROFILES or any(type(geometry.get(key)) is not int for key in names)
            or geometry.get("layout") != "sequence_major"
            or not 1 <= geometry["capacity"] <= 4096
            or not 0 <= geometry["position"] < geometry["capacity"]
            or not 1 <= geometry["width"] <= 256 or not 1 <= geometry["heads"] <= 16
            or not 1 <= geometry["group"] <= 8
            or type(iterations) is not int or not 1 <= iterations <= MAX_ITERATIONS
            or type(replicates) is not int or not 1 <= replicates <= 5):
        raise ValueError("invalid bounded measurement geometry/iterations/replicates")
    t, d, k, g = (geometry[key] for key in ("capacity", "width", "heads", "group"))
    h = k * g
    domain = geometry["position"] + 1 if profile == "prefix_f32" else t
    products = 2 * h * domain * d  # score and weighted-value float products
    per_child = products * (iterations + 1)
    cohort = per_child * 2 * replicates
    if per_child > MAX_CHILD_PRODUCTS or cohort > MAX_COHORT_PRODUCTS:
        raise ValueError("measurement product-work bound exceeded")
    inputs, output, slab = 4 * h * d + 8 * k * t * d + 8, 4 * h * d, 4 * h * t
    scratch = 8 * slab + 16 * h + 16 * t + max(h * d, k * t * d, h * t)
    if profile == "prefix_f32":
        scratch += 8 * t * d
    # Four input-owner equivalents cover RNG float64/transposition construction;
    # two outputs cover old/new assignment overlap. Parent allocates no fixture.
    memory = 256 * 2**20 + 4 * inputs + scratch + 2 * output
    if memory > MAX_ADMISSION_BYTES:
        raise ValueError("measurement memory bound exceeded")
    return {"memory_admission_bytes": memory, "input_owner_bytes_per_child": inputs,
            "complete_output_bytes": output, "maximum_scratch_bytes": scratch,
            "products_per_op": products, "products_per_child": per_child,
            "products_cohort": cohort, "operations_per_arm": replicates * (iterations + 1),
            "cold_operations_per_child": 1, "repeated_operations_per_child": iterations,
            "bounds": {"iterations": MAX_ITERATIONS, "products_per_child": MAX_CHILD_PRODUCTS,
                       "products_cohort": MAX_COHORT_PRODUCTS, "memory_admission_bytes": MAX_ADMISSION_BYTES}}


def require_identity(actual, expected):
    if actual != expected:
        raise RuntimeError("frozen installed/source/probe identity mismatch; no gain admitted")


def worker(payload):
    plan = measurement_plan(payload["fixture"], payload["profile"], payload["iterations"], 1)
    if payload["arm"] not in ("baseline", "fused"):
        raise ValueError("invalid measurement arm")
    frozen = identity()
    require_identity(frozen, payload["expected_identity"])
    setup_cpu, setup_wall = time.process_time(), time.perf_counter()
    args = fixture(**payload["fixture"])
    setup_cpu, setup_wall = time.process_time() - setup_cpu, time.perf_counter() - setup_wall
    group = payload["fixture"]["group"]
    function = baseline if payload["arm"] == "baseline" else fused
    start_cpu, start = time.process_time(), time.perf_counter()
    result = function(*args, group, payload["profile"])
    cpu, wall = time.process_time() - start_cpu, time.perf_counter() - start
    cold_hash = hashlib.sha256(result.tobytes()).hexdigest()
    del result
    start_cpu, start = time.process_time(), time.perf_counter()
    for _ in range(payload["iterations"]):
        result = function(*args, group, payload["profile"])
    repeated_cpu, repeated_wall = time.process_time() - start_cpu, time.perf_counter() - start
    repeated_hash = hashlib.sha256(result.tobytes()).hexdigest()
    if cold_hash != repeated_hash:
        raise RuntimeError("cold/repeated output hash mismatch; no gain admitted")
    require_identity(identity(), frozen)
    return {"identity": frozen, "payload": payload,
            "fixture_setup": {"cpu_seconds": setup_cpu, "wall_seconds": setup_wall,
                              "scope": "fixture construction; imports/identity checks excluded"},
            "cold": {"operations": 1, "cpu_seconds": cpu, "wall_seconds": wall,
                     "output_sha256": cold_hash},
            "repeated_same_fixture": {"operations": payload["iterations"],
                                      "cpu_seconds_total": repeated_cpu,
                                      "wall_seconds_total": repeated_wall,
                                      "cpu_seconds_per_op": repeated_cpu / payload["iterations"],
                                      "wall_seconds_per_op": repeated_wall / payload["iterations"],
                                      "final_output_sha256": repeated_hash},
            "charged_operations": 1 + payload["iterations"],
            "charged_float_products": plan["products_per_child"],
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if sys.platform == "darwin" else 1024),
            "output_sha256": repeated_hash,
            "owners": accounting(*args, group, payload["profile"], payload["arm"]),
            "gain": None, "scope": "cold/repeated same-fixture operator only; not serving throughput"}


def measure(geometry, profile, iterations, replicates, *, runner=None):
    plan = measurement_plan(geometry, profile, iterations, replicates)
    frozen = json.loads(json.dumps(identity()))
    if runner is None:
        from frontier_probe_support import fresh_workers
        runner = fresh_workers
    records = []
    with runner(Path(__file__).resolve(), memory_bytes=plan["memory_admission_bytes"],
                timeout=60) as (run, guard, host):
        baseline_swap = getattr(host, "swap_used", None)
        for replicate in range(replicates):
            pair = []
            for arm in (("baseline", "fused") if replicate % 2 == 0 else ("fused", "baseline")):
                record = run({"fixture": geometry, "profile": profile, "arm": arm,
                              "iterations": iterations, "expected_identity": frozen})
                require_identity(record["identity"], frozen)
                if (record["charged_operations"] != iterations + 1
                        or record["charged_float_products"] != plan["products_per_child"]):
                    raise RuntimeError("child charged work differs from admitted plan")
                pair.append(record)
            if pair[0]["output_sha256"] != pair[1]["output_sha256"]:
                raise RuntimeError("bitwise hash mismatch; no gain admitted")
            records.extend(pair)
        guard.check()
        final_host = guard.sample() if callable(getattr(guard, "sample", None)) else None
        watchdog = {"baseline_swap_used_bytes": baseline_swap,
                    "final_swap_used_bytes": getattr(final_host, "swap_used", None),
                    "maximum_swap_growth_bytes": getattr(guard, "maximum_swap_growth_bytes", None),
                    "minimum_available_bytes": getattr(guard, "minimum_available_bytes", None),
                    "error": getattr(guard, "error", None),
                    "missing_values_mean": "unsupported; never evidence of zero swap"}
        watchdog["final_minus_baseline_swap_bytes"] = (
            watchdog["final_swap_used_bytes"] - baseline_swap
            if baseline_swap is not None and watchdog["final_swap_used_bytes"] is not None else None)
        if watchdog["error"]:
            raise RuntimeError(watchdog["error"])
        require_identity(identity(), frozen)
    return {"expected_identity": frozen, "records": records, "admission": plan,
            "watchdog": watchdog, "all_pair_hashes_match": True,
            "charged_operations_per_arm": {arm: sum(r["charged_operations"] for r in records
                                                     if r["payload"]["arm"] == arm)
                                           for arm in ("baseline", "fused")},
            "gain": None, "scope": "isolated cold/repeated same-fixture attention; not serving throughput or whole-client qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--measure", action="store_true", help="parent-only serial fresh processes")
    parser.add_argument("--profile", choices=PROFILES, default="prefix_f32")
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--position", type=int, default=287)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--group", type=int, default=7)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if options.measure:
        geometry = {key: getattr(options, key) for key in ("capacity", "position", "width", "heads", "group")}
        geometry["layout"] = "sequence_major"
        result = measure(geometry, options.profile, options.iterations, options.replicates)
    else:
        result = correctness()
    encoded = json.dumps(result, indent=2) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(encoded)
    else:
        print(encoded, end="")
    if result.get("all_bitwise") is False:
        raise SystemExit("exactness failed; no gain admitted")


if __name__ == "__main__":
    main()
