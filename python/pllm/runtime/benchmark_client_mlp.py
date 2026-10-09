"""Fixed whole-response working-set selector and public metadata attribution."""
from contextlib import contextmanager
import dataclasses
import sys
import threading
import weakref

from pllm._benchmark_contract import TargetUnavailable

_LOCK = threading.Lock()


def metadata_size(*roots):
    """Unique Python object bytes, not RSS; exclude tensors and execution owners."""
    seen = set()
    def size(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        result = sys.getsizeof(value)
        if isinstance(value, dict):
            result += sum(size(k) + size(v) for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            result += sum(map(size, value))
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            result += sum(size(getattr(value, field.name)) for field in dataclasses.fields(value))
        return result
    return sum(map(size, roots))


@contextmanager
def client_mlp_execution(selection):
    from .model_binding import CompiledRuntimeModel
    from .mlp_tiling import MlpTiling
    from .semantic_executor import SemanticDecoderRuntime
    from .transformer_client import PreparedRemoteLinear

    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_client_mlp_research")
    original = CompiledRuntimeModel.runtime
    original_forward = SemanticDecoderRuntime._forward
    records, owners = [], weakref.WeakKeyDictionary()

    def bind(compiled, remote, **kwargs):
        if type(remote) is not PreparedRemoteLinear or remote.verification_component != "none":
            raise TargetUnavailable("mlp_tiling_requires_unverified_prepared")
        runtime = original(compiled, remote, **kwargs)
        # Both arms compile eligibility and expose the same instrumentation.
        tiling = MlpTiling(compiled, runtime, selection["tile_rows"] or 16)
        record = {"binding_digest": compiled.digest, "eligible_regions": len(tiling.regions),
            "metadata_python_bytes": metadata_size(runtime._graphs, runtime._schedule,
                compiled._plan.canonical_bytes(), compiled.canonical_bytes(), compiled._canonical_composition,
                compiled.stage_bindings, compiled.local_operations),
            "prefill_calls": 0, "tiles": 0, "peak_tile_rows": 0}
        records.append(record)
        if selection["tile_rows"]:
            runtime._mlp_tiling = tiling
        owners[runtime] = (tiling, record)
        return runtime

    def execute(runtime, ids, **arguments):
        entry = owners.get(runtime)
        if entry is None:
            return original_forward(runtime, ids, **arguments)
        tiling, record = entry
        if runtime.position == 0:
            record["prefill_calls"] += 1
        try:
            return original_forward(runtime, ids, **arguments)
        finally:
            record.update(tiles=tiling.tiles, peak_tile_rows=tiling.peak_tile_rows)

    try:
        CompiledRuntimeModel.runtime = bind
        SemanticDecoderRuntime._forward = execute
        yield records
    finally:
        CompiledRuntimeModel.runtime = original
        SemanticDecoderRuntime._forward = original_forward
        owners.clear()
        _LOCK.release()
