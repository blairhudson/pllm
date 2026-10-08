"""Observe client allocation phases through one ordinary native SDK benchmark.

Instrumentation adds overhead. Nested process samples/timings are non-additive,
and optional Python allocation tracing excludes unregistered native allocations.
Only code locations and aggregate sizes are retained; no arguments or results.
Use a fresh process/output for each diagnostic. Uninstrumented matched cohorts
remain the authority for performance and memory comparisons.
"""
from __future__ import annotations

import argparse
from contextvars import ContextVar
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
    phases, events, originals, timings = {}, [], [], {}
    execution_phase = ContextVar("profile_execution_phase", default="setup")
    allocation_points = {}
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
        bundle_artifacts, client, http_response, model_binding, paged, semantic_executor,
        transformer_client,
    )
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    from pllm.deployment import WanConditions

    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *(Path(module.__file__).resolve()
               for module in (bundle_artifacts, client, http_response, model_binding,
                               paged, semantic_executor, transformer_client))]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}

    def allocation_sample():
        return [{"file": row.traceback[0].filename, "line": row.traceback[0].lineno,
                 "bytes": row.size, "count": row.count}
                for row in tracemalloc.take_snapshot().statistics("lineno")[:40]]

    def timed(label, function, *values, exchange=False, server=False, **kwargs):
        key = execution_phase.get() + "." + label
        wall, cpu = time.perf_counter(), time.process_time()
        received, server_seconds, succeeded = 0, 0., False
        sent = sum(map(len, values[1])) if exchange else 0
        try:
            result = function(*values, **kwargs)
            if exchange:
                received = sum(map(len, result))
            if server:
                server_seconds = result[1] / 1e9
            succeeded = True
            return result
        finally:
            elapsed, compute = time.perf_counter() - wall, time.process_time() - cpu
            with lock:
                row = timings.setdefault(key, {"calls": 0, "failures": 0,
                    "wall_seconds": 0., "cpu_seconds": 0., "upload_payload_bytes": 0,
                    "download_payload_bytes": 0, "reported_server_seconds": 0.})
                row["calls"] += 1
                row["failures"] += int(not succeeded)
                row["wall_seconds"] += elapsed
                row["cpu_seconds"] += compute
                row["upload_payload_bytes"] += sent
                row["download_payload_bytes"] += received
                row["reported_server_seconds"] += server_seconds

    def time_watch(owner, name, label, *, server=False):
        descriptor = inspect.getattr_static(owner, name)
        special = isinstance(descriptor, (classmethod, staticmethod))
        original = descriptor.__func__ if special else descriptor
        @wraps(original)
        def measured(*values, **kwargs):
            return timed(label, original, *values, server=server, **kwargs)
        originals.append((owner, name, descriptor))
        setattr(owner, name, type(descriptor)(measured) if special else measured)

    def watch(owner, name, *, trace_phase=False):
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
                    if args.allocations and trace_phase and label not in allocation_points:
                        allocation_points[label] = allocation_sample()

        originals.append((owner, name, descriptor))
        setattr(owner, name, type(descriptor)(measured) if special else measured)

    for name in ("_load_artifact_bundle", "_transformer_state", "_prepare_inventory", "_make_tokenizer"):
        watch(client.RuntimeClient, name, trace_phase=True)
    for name in ("parse_manifest", "reconstruct_document"):
        watch(client, name, trace_phase=True)
    watch(model_binding, "compile_runtime_model", trace_phase=True)
    watch(model_binding.CompiledRuntimeModel, "runtime")
    for name in ("_from_document", "local_linear", "local_token_lookup"):
        watch(transformer_client.ClientBundle, name)
    watch(transformer_client.PreparedRemoteLinear, "__call__")
    for name in ("__init__", "prepare_ids", "forward_ids", "snapshot"):
        watch(semantic_executor.SemanticDecoderRuntime, name, trace_phase=True)

    # Fine timers omit RSS sampling. These nested, process-wide CPU intervals
    # include relay/telemetry threads and are diagnostic, not additive costs.
    original_phase = semantic_executor.SemanticDecoderRuntime._forward_phase
    @wraps(original_phase)
    def phase(runtime, *values, **kwargs):
        token = execution_phase.set("prefill" if runtime.position == 0
                                    or kwargs.get("continuation") else "decode")
        try:
            return timed("decoder", original_phase, runtime, *values, **kwargs)
        finally:
            execution_phase.reset(token)
    originals.append((semantic_executor.SemanticDecoderRuntime, "_forward_phase", original_phase))
    semantic_executor.SemanticDecoderRuntime._forward_phase = phase

    original_remote = transformer_client.PreparedRemoteLinear.__init__
    @wraps(original_remote)
    def remote(runtime, *values, **kwargs):
        original_remote(runtime, *values, **kwargs)
        exchange = runtime.inference
        def measured_exchange(*args, **options):
            return timed("remote_exchange", exchange, *args, exchange=True, **options)
        runtime.inference = measured_exchange
    originals.append((transformer_client.PreparedRemoteLinear, "__init__", original_remote))
    transformer_client.PreparedRemoteLinear.__init__ = remote
    time_watch(transformer_client.PreparedRemoteLinear, "__call__", "remote_linear")
    time_watch(transformer_client.PreparedRemoteLinear, "_decode_reply", "reply_decode", server=True)
    for name in ("quantize_activation_per_row", "mask_prepared_input", "dequantize_matmul"):
        time_watch(transformer_client, name, name)
    for name in ("_from_raw_executor", "clear", "wrap32", "gather_rows", "gather_columns"):
        time_watch(paged.PagedGEMM, name, "paged_" + name.strip("_"))

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
            allocations = allocation_sample()
            tracemalloc.stop()
        document = {"schema": "pllm.client_frontier_profile.v2", "scope": __doc__,
            "initial": initial, "final": final, "phases": phases, "events": events,
            "phase_timings": timings,
            "timing_scope": "nested wall and process-wide client CPU; server-reported stage elapsed; "
                "exchange payloads exclude outer HTTP/WebSocket envelopes; pacing and network wait "
                "are not isolated from provider/serialization time",
            "python_allocations": allocations, "phase_python_allocations": allocation_points,
            "report": report,
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
