"""Pinned full decoder layer paging and fresh-interpreter restart, local clear kernels.

All checkpoint weights remain in each process. This is not SDK/provider serving;
key/head custody belongs to the parent. Authentication, disk I/O and publication
are charged; source loading, execution and restart have separate timings.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import sys
import tempfile
import time


def worker(spec):
    import numpy as np
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.tiered_execution import tiered_runtime
    from decoder_probe_support import DecoderFixture

    fixture = DecoderFixture(inputs=spec["tokens"], outputs=8)
    fixture.composition = fixture.composition.with_params(
        quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    fixture.compiled = compile_runtime_model(fixture.plan, fixture.bundle, composition=fixture.composition)
    mode = spec["mode"]
    start, cpu = time.perf_counter(), time.process_time()
    if mode == "resident":
        runtime = fixture.compiled.runtime(fixture.remote)
    else:
        runtime = tiered_runtime(fixture.compiled, fixture.remote, spec["root"], bytes.fromhex(spec["key"]),
            head=spec.get("head"), page_rows=64)
    if mode == "restore":
        logits = runtime._tiered.logits.copy()
    else:
        # Fixed public numeric fixture, not a prompt-quality cohort.
        ids = [(i * 31 + 11) % 10000 for i in range(spec["tokens"])]
        _, logits, _ = runtime.prepare_ids(ids)
    hashes = [hashlib.sha256(logits.tobytes()).hexdigest()]
    if mode != "issue":
        for _ in range(3):
            logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
            hashes.append(hashlib.sha256(logits.tobytes()).hexdigest())
    result = {"mode": mode, "tokens": spec["tokens"], "source": fixture.lock(),
        "window_seconds": time.perf_counter() - start, "window_cpu_seconds": time.process_time() - cpu,
        "cold_process_cpu_seconds": time.process_time(), "memory": process_memory(),
        "logit_hashes": hashes, "position": runtime.position,
        "final_dense_kv_bytes": runtime.position * runtime.layers * runtime.default_kv_heads * runtime.default_head_dim * 8}
    # Parity inspection follows the measurement window and loads only one layer.
    kv = hashlib.sha256()
    if mode == "resident":
        for cache in runtime.caches:
            kv.update(cache.key[:cache.length].tobytes())
            kv.update(cache.value[:cache.length].tobytes())
    else:
        state = runtime._tiered
        result.update(peak_live_kv_bytes=state.peak_live_kv_bytes, read_bytes=state.read_bytes,
            written_bytes=state.written_bytes, disk_bytes=sum(p.stat().st_size for p in state.root.iterdir()))
        if mode == "issue":
            result["head"] = state.head
        state.begin(runtime, np.array([0], np.int64), False)
        for layer in range(runtime.layers):
            state.load(runtime, layer)
            cache = runtime.caches[layer]
            kv.update(cache.key[:cache.length].tobytes())
            kv.update(cache.value[:cache.length].tobytes())
            state._retire(runtime)
        state.close(runtime)
    result["kv_hash"] = kv.hexdigest()
    runtime.reset()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, default=3, choices=range(1, 4))
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 512], choices=[128, 512])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    from frontier_probe_support import fresh_workers
    samples = []
    with fresh_workers(__file__, memory_bytes=4 << 30) as (run, guard, host):
        for count in args.tokens:
            for repeat in range(args.processes):
                with tempfile.TemporaryDirectory(prefix="pllm-tiered-") as directory:
                    base = {"tokens": count, "key": secrets.token_bytes(32).hex()}
                    resident = run(base | {"mode": "resident"})
                    paged = run(base | {"mode": "paged", "root": directory + "/paged"})
                    issuer = run(base | {"mode": "issue", "root": directory + "/restart"})
                    restored = run(base | {"mode": "restore", "root": directory + "/restart", "head": issuer.pop("head")})
                    for candidate in (paged, restored):
                        for key in ("source", "logit_hashes", "position", "kv_hash"):
                            if candidate[key] != resident[key]:
                                raise AssertionError(f"tiered/restarted decoder differs: {key}")
                    if issuer["logit_hashes"] != resident["logit_hashes"][:1]:
                        raise AssertionError("issuer prefill differs")
                    samples.append({"repeat": repeat, "resident": resident, "paged": paged,
                        "issuer": issuer, "restored": restored})
    report = {"schema": "pllm.tiered_decoder_probe.v1", "scope": __doc__, "samples": samples,
        "host": asdict(host), "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "exact_parity": True,
        "build": run.build}
    with args.output.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(json.dumps({"pairs": len(samples), "parity": True}))


if __name__ == "__main__":
    main()
