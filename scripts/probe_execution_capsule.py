"""Fresh-process native instruction ownership on a pinned, complete Qwen decoder.

All clear checkpoint kernels remain resident. Publication is a separate process;
import, Python scheduling, token/head and all local decoder arithmetic are charged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time


def owned_size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(owned_size(k, seen) + owned_size(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        size += sum(owned_size(v, seen) for v in value)
    return size


def worker(spec):
    import numpy as np
    from pllm.runtime.benchmark_memory import process_memory
    from pllm.runtime.execution_capsule import publish_execution_capsule, capsule_runtime
    from decoder_probe_support import DecoderFixture

    fixture = DecoderFixture(inputs=150, outputs=8)
    start, cpu = time.perf_counter(), time.process_time()
    if spec["mode"] == "publish":
        payload, digest = publish_execution_capsule(fixture.compiled)
        Path(spec["path"]).write_bytes(payload)
        return {"digest": digest, "bytes": len(payload), "source": fixture.lock(),
                "wall_seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}
    if spec["mode"] == "native":
        payload = Path(spec["path"]).read_bytes()
        runtime = capsule_runtime(fixture.compiled, fixture.remote, payload, spec["digest"])
        del payload
    else:
        runtime = fixture.compiled.runtime(fixture.remote)
    construct_cpu, construct_wall = time.process_time() - cpu, time.perf_counter() - start
    footprint = owned_size((runtime._graphs, runtime._schedule,
                           None if runtime._capsule is None else runtime._capsule.facts))
    start, cpu = time.perf_counter(), time.process_time()
    _, logits, _ = runtime.prepare_ids([(i * 17 + 11) % 10000 for i in range(150)])
    scores = hashlib.sha256(logits.tobytes())
    for _ in range(7):
        logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
        scores.update(logits.tobytes())
    report = {"mode": spec["mode"], "source": fixture.lock(), "logits": scores.hexdigest(),
        "construction_cpu_seconds": construct_cpu, "construction_seconds": construct_wall,
        "execution_cpu_seconds": time.process_time() - cpu, "execution_seconds": time.perf_counter() - start,
        "cold_process_cpu_seconds": time.process_time(), "memory": process_memory(),
        "python_graph_schedule_bytes": footprint}
    kv = hashlib.sha256()
    for cache in runtime.caches:
        kv.update(cache.key[:cache.length].tobytes())
        kv.update(cache.value[:cache.length].tobytes())
    report["kv"] = kv.hexdigest()
    runtime.reset()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--processes", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.load(sys.stdin))))
        return
    if args.output is None or args.output.exists():
        parser.error("a new --output path is required")
    from frontier_probe_support import fresh_workers
    samples = []
    with tempfile.TemporaryDirectory(prefix="pllm-capsule-") as directory:
        with fresh_workers(__file__, memory_bytes=4 << 30) as (run, guard, host):
            path = directory + "/capsule.json"
            published = run({"mode": "publish", "path": path})
            for repeat in range(args.processes):
                pair = {}
                for mode in (("python", "native") if repeat % 2 == 0 else ("native", "python")):
                    pair[mode] = run({"mode": mode, "path": path, "digest": published["digest"]})
                for key in ("source", "logits", "kv"):
                    if pair["python"][key] != pair["native"][key]:
                        raise AssertionError(f"execution capsule changes {key}")
                samples.append(pair)
    report = {"schema": "pllm.execution_capsule_probe.v1", "scope": __doc__, "publication": published,
        "samples": samples, "host": asdict(host), "build": run.build,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes, "exact_parity": True}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pairs": len(samples), "parity": True}))


if __name__ == "__main__":
    main()
