"""Fresh-process, pinned W8A8 static-retention quality/allocation diagnostic.

All body kernels are local clear callbacks. Process peaks include all checkpoint
weights; they are not SDK-client or provider-topology memory measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ("dense", "retained_full", "retained_75", "retained_50")
PUBLIC_PROMPTS = tuple(
    "The following fictional rainfall log is public. "
    + " ".join(f"Station {j} measured {(j * 11 + i * 5) % 29 + 1} units on day {j + 2}." for j in range(1, 13))
    + question
    for i, question in enumerate((
        " Why would a scientist preserve the original log?",
        " Give a concise title for these observations.",
        " Does this log identify the wettest month? Explain briefly.",
        " State one limitation of comparing the station readings.",
        " Why should the measurement unit be defined?",
        " Suggest a check that could detect a copying mistake.",
        " Which station appears last in the log?",
        " Are the observations all from the same day?",
        " What additional information would help compare rainfall fairly?",
        " Explain the difference between a measurement and a forecast.",
        " Is the instrument's precision specified?",
        " Why might repeated measurements be useful?",
    ))
)
CONFIGURATION = {"max_input_tokens": 256, "output_positions": 4, "public_prompts": len(PUBLIC_PROMPTS),
    "observation_window": [1, 4], "static_keep": {"retained_full": [1, 1], "retained_75": [3, 4], "retained_50": [1, 2]},
    "dynamic_keep": [1, 1], "share_adjacent_layers": False, "fresh_process_per_candidate": True,
    "selection": "mean final observation queries and all heads, stable top-k plus mandatory recent suffix",
    "policy_frozen_before_cohort": True}
SOURCE_FILES = (
    "benchmarks/research/retained_decoder.py", "scripts/decoder_probe_support.py",
    "crates/pllm-compiler/src/retained_decoder.rs", "crates/pllm-compiler/src/decoder_continuation.rs",
    "crates/pllm-compiler/src/decoder_runtime_schedule.rs", "crates/pllm-models/src/cache.rs",
    "crates/pllm-core/src/retained_kv.rs", "crates/pllm-core/src/cache_selection_reference.rs",
    "crates/pllm-python/src/retained_decoder.rs", "crates/pllm-python/src/retained_kv.rs",
    "python/pllm/runtime/retained_execution.py", "python/pllm/runtime/retained_state.py",
    "python/pllm/runtime/semantic_executor.py", "python/pllm/runtime/semantic_attention.py",
)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def implementation():
    from pllm import _native
    return {"source_sha256": {path: sha((ROOT / path).read_bytes()) for path in SOURCE_FILES},
        "native_sha256": sha(Path(_native.__file__).read_bytes())}


def check_implementation(expected):
    if implementation() != expected:
        raise ValueError("diagnostic implementation changed within matched cohort")


def state(runtime):
    from pllm.runtime.retained_state import RetainedLayerCache
    digest, allocated, active = hashlib.sha256(), 0, 0
    for layer, cache in enumerate(runtime.caches):
        if type(cache) is RetainedLayerCache:
            key, value, positions = cache.gather()
            allocated += cache.allocated_bytes
        else:
            key, value = cache.key[:cache.length], cache.value[:cache.length]
            positions = np.arange(cache.length, dtype=np.uint32)
            allocated += cache.key.nbytes + cache.value.nbytes
        active += key.nbytes + value.nbytes
        digest.update(canonical([layer, runtime.position, key.shape]))
        for array, dtype in [(positions, "<u4"), (key, "<f4"), (value, "<f4")]:
            digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes())
    return {"kv_sha256": digest.hexdigest(), "owned_payload_bytes": allocated, "active_kv_bytes": active}


def worker(mode, scratch, output):
    expected = json.loads((scratch / "implementation.json").read_bytes())
    check_implementation(expected)
    from pllm.passes import KvCacheEviction
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.runtime.retained_execution import retained_runtime
    from scripts.decoder_probe_support import DecoderFixture

    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("retained decoder diagnostic needs 4 GiB beyond host reserve")
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, lambda _: os.kill(os.getpid(), signal.SIGTERM))
    guard.start()
    reference = None
    try:
        started, cpu = time.perf_counter(), time.process_time()
        fixture = DecoderFixture(inputs=CONFIGURATION["max_input_tokens"], outputs=CONFIGURATION["output_positions"])
        policy = None if mode == "dense" else KvCacheEviction(
            static_keep=CONFIGURATION["static_keep"][mode], observation_window=CONFIGURATION["observation_window"],
            dynamic_keep=CONFIGURATION["dynamic_keep"], share_adjacent_layers=False)
        reference_path = scratch / "reference.npz"
        if mode != "dense":
            reference = np.load(reference_path, allow_pickle=False)
            if json.loads((scratch / "source.json").read_bytes()) != fixture.lock():
                raise ValueError("candidate source differs from matched reference")
        arrays, observations = {}, []
        contract = None
        for index, prompt in enumerate(PUBLIC_PROMPTS):
            guard.check()
            ids = fixture.tokens(prompt)
            cohort = sha(np.asarray(ids, dtype="<i8").tobytes())
            if reference is not None and bytes(reference[f"cohort_{index}"]).decode() != cohort:
                raise ValueError("candidate token cohort differs from reference")
            runtime = fixture.compiled.runtime(fixture.remote) if policy is None else retained_runtime(fixture.compiled, fixture.remote, policy=policy)
            if policy is not None:
                contract = runtime._retained.contract.to_dict()
            before, before_cpu = time.perf_counter(), time.process_time()
            _, logits, _ = runtime.prepare_ids(ids)
            prefill = state(runtime)
            observed, forcing = [logits.copy()], [int(np.argmax(logits))]
            control = None if reference is None else reference[f"logits_{index}"]
            for step in range(CONFIGURATION["output_positions"] - 1):
                token = forcing[-1] if control is None else int(np.argmax(control[step]))
                logits = runtime.forward_ids([token])[-1]
                observed.append(logits.copy())
                forcing.append(int(np.argmax(logits)))
            final = state(runtime)
            elapsed, elapsed_cpu = time.perf_counter() - before, time.process_time() - before_cpu
            observed = np.stack(observed)
            if control is None:
                control = observed.copy()
            # Every process owns the same shaped observed/reference arrays and
            # writes its diagnostic payload, including the dense control.
            arrays.update({f"logits_{index}": observed, f"cohort_{index}": np.frombuffer(cohort.encode(), dtype=np.uint8),
                f"state_{index}": np.frombuffer(final["kv_sha256"].encode(), dtype=np.uint8)})
            errors = np.max(np.abs(observed.astype(np.float64) - control.astype(np.float64)), axis=1).tolist()
            exact = [np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(observed, control, strict=True)]
            kv_exact = mode == "dense" or final["kv_sha256"] == bytes(reference[f"state_{index}"]).decode()
            if mode == "retained_full" and (not all(exact) or not kv_exact):
                raise ValueError("full retention changed logits or KV")
            if not exact[0]:
                raise ValueError("post-attention retention changed prefill logits")
            observations.append({"input_tokens": len(ids), "token_cohort_sha256": cohort,
                "same_token_top1_matches": (np.argmax(observed, axis=1) == np.argmax(control, axis=1)).tolist(),
                "full_logits_exact": exact, "worst_abs_logit_errors": errors, "full_final_kv_exact": kv_exact,
                "prefill": prefill, "final": final, "diagnostic_seconds": elapsed, "diagnostic_cpu_seconds": elapsed_cpu})
            runtime.reset()
            del runtime, observed, control
            gc.collect()
        if mode == "dense":
            np.savez(reference_path, **arrays)
            (scratch / "source.json").write_bytes(canonical(fixture.lock()))
        else:
            np.savez(scratch / f"{mode}.npz", **arrays)
        guard.check()
        check_implementation(expected)
        report = {"mode": mode, "source": fixture.lock(), "policy": None if policy is None else policy.to_spec(),
            "contract": contract, "observations": observations, "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "diagnostic_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "process_elapsed_seconds": time.perf_counter() - started, "process_cpu_seconds": time.process_time() - cpu,
            "process_peak_scope": "fresh local-clear diagnostic including all checkpoint/body weights, loading, tokenization, oracle comparison and twelve requests"}
        output.write_bytes(canonical(report))
    finally:
        if reference is not None:
            reference.close()
        guard.close()


def summarize(row):
    data = row["observations"]
    return {"prefill_matches": sum(r["same_token_top1_matches"][0] for r in data),
        "decode_matches": sum(sum(r["same_token_top1_matches"][1:]) for r in data),
        "prefill_count": len(data), "decode_count": len(data) * (CONFIGURATION["output_positions"] - 1),
        "exact_logit_positions": sum(sum(r["full_logits_exact"]) for r in data),
        "full_final_kv_exact_prompts": sum(r["full_final_kv_exact"] for r in data),
        "worst_abs_logit_error": max(max(r["worst_abs_logit_errors"]) for r in data),
        "prefill_owned_payload_bytes_median": float(np.median([r["prefill"]["owned_payload_bytes"] for r in data])),
        "prefill_active_kv_bytes_median": float(np.median([r["prefill"]["active_kv_bytes"] for r in data])),
        "diagnostic_process_peak_rss_bytes": row["diagnostic_process_peak_rss_bytes"],
        "maximum_swap_growth_bytes": row["maximum_swap_growth_bytes"]}


def run(output):
    if output.exists():
        raise ValueError("refusing to replace retained-decoder evidence")
    output.parent.mkdir(parents=True, exist_ok=True)
    frozen_implementation = implementation()
    with tempfile.TemporaryDirectory(prefix="retained-decoder-", dir=output.parent) as name:
        scratch = Path(name)
        (scratch / "implementation.json").write_bytes(canonical(frozen_implementation))
        candidates = []
        for mode in MODES:
            destination = scratch / f"{mode}.json"
            subprocess.run([sys.executable, "-m", "benchmarks.research.retained_decoder", "--worker", mode,
                "--scratch", str(scratch), "--output", str(destination)], cwd=ROOT, check=True, timeout=900,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false"))
            candidates.append(json.loads(destination.read_bytes()))
    check_implementation(frozen_implementation)
    report = {"schema": "pllm.retained_decoder_reference.v1", "configuration": CONFIGURATION,
        "public_prompt_sha256": sha(canonical(PUBLIC_PROMPTS)), **frozen_implementation, "source": candidates[0]["source"],
        "environment": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__},
        "source_papers": ["https://arxiv.org/abs/2501.06807"], "paper_sha256": sha((ROOT / "papers/r23-mpcache.pdf").read_bytes()),
        "summary": {row["mode"]: summarize(row) for row in candidates}, "candidates": candidates,
        "reference": "pinned W8A8 full cache; no upstream float32 or LongBench quality comparison",
        "selection_executor": "trusted_client", "protected_selection": False, "executable_experiment": False,
        "whole_generation_quality": False, "sdk_client_peak_rss_bytes": None, "provider_transport_bytes": None,
        "limitations": ["post-prefill static retention only; dynamic clusters and inter-layer sharing absent",
            "private query-dependent addresses remain client-local; no provider-held private-state contract",
            "fresh short-context public cohort with four same-token output positions; no trajectory quality or tuning",
            "owned-array admission excludes checkpoint weights, Python allocator and process overhead",
            "one fresh process per policy; process peak includes full local body weights and twelve sequential requests",
            "full-response SDK, protected selection, independent providers, wire costs and long-context quality unmeasured"]}
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--scratch", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.scratch is None:
            parser.error("--worker requires --scratch")
        worker(args.worker, args.scratch, args.output)
    else:
        run(args.output)
