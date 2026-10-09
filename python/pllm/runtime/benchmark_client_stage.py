"""Fixed, worker-local research adapter over the real compiled SDK response path.

Only the selected stage's ownership changes. This is not a serving component or
an arbitrary benchmark callback. The enclosing fresh worker owns the temporary
runtime-construction adapter, and restores it on every exit.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading
import time

from pllm._benchmark_contract import TargetUnavailable

_LOCK = threading.Lock()


@contextmanager
def client_stage_execution(selection):
    from .model_binding import CompiledRuntimeModel
    from .prepared_stage import NativePreparedLinear
    from .transformer_client import PreparedInventoryLease, PreparedRemoteLinear

    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_client_stage_research")
    original = CompiledRuntimeModel.runtime
    records = []

    def bind(compiled, remote, **kwargs):
        if (type(remote) is not PreparedRemoteLinear or remote.verification_component != "none"
                or remote.request_encoding not in {"compact", "stage_packed"}
                or type(remote.inventory) is not PreparedInventoryLease):
            raise TargetUnavailable("client_stage_requires_unverified_packed_prepared")
        stages = [s for s in compiled.stage_bindings if s.role == selection["role"]
                  and s.layer_index == selection["layer"] and s.client_weight_layout is None]
        if len(stages) != 1 or remote.stages[stages[0].stage_id].output_residue_bits is None:
            raise TargetUnavailable("client_stage_requires_unique_remote_residue_stage")
        stage = stages[0]
        if selection["owner"] == "native":
            delegate = NativePreparedLinear(compiled, remote, [stage.stage_id])
        else:
            compiled.validate()
            delegate = remote
        record = {"binding_digest": compiled.digest, "stage_id": stage.stage_id,
                  "role": stage.role, "layer": stage.layer_index, "calls": 0, "rows": 0,
                  "client_cpu_seconds": 0., "wall_seconds": 0.}
        records.append(record)

        def execute(stage_id, activation):
            if stage_id != stage.stage_id:
                return delegate(stage_id, activation)
            cpu, wall = time.process_time(), time.perf_counter()
            result = delegate(stage_id, activation)
            record["client_cpu_seconds"] += time.process_time() - cpu
            record["wall_seconds"] += time.perf_counter() - wall
            record["calls"] += 1
            record["rows"] += activation.size // stage.in_features
            return result

        return original(compiled, execute, **kwargs)

    try:
        CompiledRuntimeModel.runtime = bind
        yield records
    finally:
        CompiledRuntimeModel.runtime = original
        _LOCK.release()
