"""Scoped allocation attribution of the real client compiler and runtime binding.

Tracing changes process costs. These diagnostics have a separate measurement
identity from serving comparisons; they do not estimate a load-ready artifact's
benefit by subtracting traced bytes from RSS.
"""
from contextlib import contextmanager
import threading
import time
import tracemalloc

from pllm._benchmark_contract import TargetUnavailable

_LOCK = threading.Lock()


@contextmanager
def client_metadata_attribution():
    from . import model_binding
    from .benchmark_client_mlp import metadata_size

    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_client_metadata_attribution")
    original_compile = model_binding._compile_runtime_model
    original_runtime = model_binding.CompiledRuntimeModel.runtime
    original_demand = model_binding.CompiledRuntimeModel.prepared_stage_rows
    records = []
    sample_lock = threading.RLock()

    def measure(phase, function, args, kwargs, roots):
        with sample_lock:
            # Runtime binding includes revalidation. Count that nested compiler
            # once, inside its outer phase, rather than resetting a live trace.
            if tracemalloc.is_tracing():
                return function(*args, **kwargs)
            if len(records) >= 4096:
                raise TargetUnavailable("metadata_attribution_record_limit")
            tracemalloc.start(1)
            started, cpu = time.perf_counter(), time.process_time()
            try:
                result = function(*args, **kwargs)
                retained, peak = tracemalloc.get_traced_memory()
                timing = {"wall_seconds": time.perf_counter() - started,
                          "cpu_seconds": time.process_time() - cpu}
            finally:
                tracemalloc.stop()
            records.append({"phase": phase, **timing,
                "new_traced_retained_bytes": retained, "new_traced_peak_bytes": peak,
                "public_metadata_python_bytes": metadata_size(*roots(result))})
            return result

    def compiled_roots(compiled):
        return (compiled._plan.canonical_bytes(), compiled.canonical_bytes(),
                compiled._canonical_composition, compiled.stage_bindings, compiled.local_operations)

    def compile_model(*args, **kwargs):
        return measure("compile_binding", original_compile, args, kwargs,
                       lambda result: (*compiled_roots(result[0]), result[1]))

    def runtime(compiled, *args, **kwargs):
        return measure("runtime_binding", original_runtime, (compiled, *args), kwargs,
                        lambda result: (*compiled_roots(compiled), result._graphs, result._schedule,
                                        result._stages))

    def demand(compiled, *args, **kwargs):
        return measure("preparation_demand", original_demand, (compiled, *args), kwargs,
                       lambda result: (result,))

    try:
        if tracemalloc.is_tracing():
            raise TargetUnavailable("metadata_attribution_requires_own_tracer")
        model_binding._compile_runtime_model = compile_model
        model_binding.CompiledRuntimeModel.runtime = runtime
        model_binding.CompiledRuntimeModel.prepared_stage_rows = demand
        yield records
    finally:
        model_binding._compile_runtime_model = original_compile
        model_binding.CompiledRuntimeModel.runtime = original_runtime
        model_binding.CompiledRuntimeModel.prepared_stage_rows = original_demand
        _LOCK.release()
