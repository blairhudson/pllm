"""Observe client allocation phases through one ordinary native SDK benchmark.

Instrumentation adds overhead. Nested process samples/timings are non-additive,
and optional Python allocation tracing excludes unregistered native allocations.
Only code locations and aggregate sizes are retained; no arguments or results.
Use a fresh process/output for each diagnostic. Uninstrumented matched cohorts
remain the authority for performance and memory comparisons.
"""
from __future__ import annotations

import argparse
from functools import wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import signal
import threading
import time
import tracemalloc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="trusted Python module:factory")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allocations", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 1 <= args.max_output_tokens <= 32:
        parser.error("output bound must be 1..32")

    from pllm.runtime.benchmark_memory import process_memory
    phases, events, originals = {}, [], []
    lock = threading.Lock()
    started = time.perf_counter()

    def sample():
        result = process_memory()
        result["elapsed_seconds"] = time.perf_counter() - started
        if args.allocations:
            result["traced_current_bytes"], result["traced_peak_bytes"] = tracemalloc.get_traced_memory()
            result["tracer_bytes"] = tracemalloc.get_tracemalloc_memory()
        return result

    initial = sample()
    if args.allocations:
        tracemalloc.start(1)
    from pllm.runtime import (
        bundle_artifacts, client, http_response, model_binding, semantic_executor, transformer_client,
    )
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    from pllm.deployment import WanConditions

    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *(Path(module.__file__).resolve()
               for module in (bundle_artifacts, client, http_response, model_binding,
                              semantic_executor, transformer_client))]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}

    def watch(owner, name):
        descriptor = inspect.getattr_static(owner, name)
        special = isinstance(descriptor, (classmethod, staticmethod))
        original = descriptor.__func__ if special else descriptor
        label = f"{owner.__name__}.{name}"

        @wraps(original)
        def measured(*values, **kwargs):
            before = sample()
            wall, cpu = time.perf_counter(), time.process_time()
            try:
                return original(*values, **kwargs)
            finally:
                elapsed, compute = time.perf_counter() - wall, time.process_time() - cpu
                after = sample()
                with lock:
                    row = phases.setdefault(label, {"calls": 0, "wall_seconds": 0., "cpu_seconds": 0.,
                        "first_before": before, "rss_after_max": 0, "peak_after_max": 0,
                        "new_highwater_max": 0})
                    row["calls"] += 1
                    row["wall_seconds"] += elapsed
                    row["cpu_seconds"] += compute
                    row["last_after"] = after
                    row["rss_after_max"] = max(row["rss_after_max"], after["rss_bytes"] or 0)
                    row["peak_after_max"] = max(row["peak_after_max"], after["lifetime_peak_rss_bytes"] or 0)
                    row["new_highwater_max"] = max(row["new_highwater_max"],
                        (after["lifetime_peak_rss_bytes"] or 0) - (before["lifetime_peak_rss_bytes"] or 0))
                    if row["calls"] <= 2 and len(events) < 64:
                        events.append({"phase": label, "before": before, "after": after})

        originals.append((owner, name, descriptor))
        setattr(owner, name, type(descriptor)(measured) if special else measured)

    for name in ("_load_artifact_bundle", "_transformer_state", "_prepare_inventory", "_make_tokenizer"):
        watch(client.RuntimeClient, name)
    for name in ("parse_manifest", "reconstruct_document"):
        watch(client, name)
    watch(model_binding, "compile_runtime_model")
    watch(model_binding.CompiledRuntimeModel, "runtime")
    for name in ("_from_document", "local_linear", "local_token_lookup"):
        watch(transformer_client.ClientBundle, name)
    watch(transformer_client.PreparedRemoteLinear, "__call__")
    for name in ("__init__", "prepare_ids", "forward_ids", "snapshot"):
        watch(semantic_executor.SemanticDecoderRuntime, name)

    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    module, factory = args.experiment.split(":", 1)
    experiment = getattr(importlib.import_module(module), factory)()
    report = None
    try:
        report = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            experiment=experiment, prompt=args.prompt_file.read_text().strip(),
            max_output_tokens=args.max_output_tokens, warmups=0, repetitions=1,
            timeout_seconds=1800, temperature=0, capture_output_digest=True,
            backend="native", wan=WanConditions(), emulate_wan=True)
    finally:
        for owner, name, descriptor in reversed(originals):
            setattr(owner, name, descriptor)
        final = sample()
        allocations = None
        if args.allocations:
            allocations = [{"file": row.traceback[0].filename, "line": row.traceback[0].lineno,
                "bytes": row.size, "count": row.count}
                for row in tracemalloc.take_snapshot().statistics("lineno")[:40]]
            tracemalloc.stop()
        document = {"schema": "pllm.client_frontier_profile.v1", "scope": __doc__,
            "initial": initial, "final": final, "phases": phases, "events": events,
            "python_allocations": allocations, "report": report,
            "source_sha256": source_hashes,
            "sources_unchanged": all(hashlib.sha256(path.read_bytes()).hexdigest()
                == source_hashes[str(path.relative_to(root))] for path in sources)}
        with args.output.open("x") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"output": str(args.output), "memory": final,
                          "checks": None if report is None else report["checks"]}))


if __name__ == "__main__":
    main()
