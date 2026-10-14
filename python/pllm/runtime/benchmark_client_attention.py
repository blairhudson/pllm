"""Fixed response selector, charging full state audits in both matched arms."""
from contextlib import contextmanager
import hashlib
import hmac
import threading
import weakref

import numpy as np

from pllm._benchmark_contract import TargetUnavailable

_LOCK = threading.Lock()


def attention_memory_admission(report, memory_bytes, *, enforced_wan):
    """Retain full-stage pricing and add the bounded region/audit overlap."""
    import copy
    from .benchmark_memory import admit_memory, host_memory

    estimate = copy.deepcopy(report["estimate"])
    # Native region output is at most 64 MiB. Charge a second complete output,
    # bounded contract metadata and streaming hash scratch with 25% margin.
    extra = 160 << 20
    estimate["client_peak_bytes"] += extra
    estimate["native_total_peak_bytes"] += extra
    estimate["components"]["client_attention_research_overlap_bytes"] = extra
    return admit_memory(estimate, host_memory(), backend="native",
                        enforced_wan=enforced_wan, memory_budget_bytes=memory_bytes)


def _array_commitment(digest, array):
    digest.update(str((array.shape, array.dtype.str)).encode())
    # Bound packing of strided arrays; never snapshot the decoder for measurement.
    for chunk in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                           op_flags=["readonly"], order="C", buffersize=16384):
        digest.update(chunk.tobytes())


@contextmanager
def client_attention_execution(selection, salt):
    from .attention_tiling import AttentionTiling
    from .model_binding import CompiledRuntimeModel
    from .semantic_executor import SemanticDecoderRuntime
    from .transformer_client import PreparedRemoteLinear

    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_client_attention_research")
    original, original_forward = CompiledRuntimeModel.runtime, SemanticDecoderRuntime._forward
    records, owners = [], weakref.WeakKeyDictionary()

    def bind(compiled, remote, **kwargs):
        if type(remote) is not PreparedRemoteLinear or remote.verification_component != "none":
            raise TargetUnavailable("attention_tiling_requires_unverified_prepared")
        runtime = original(compiled, remote, **kwargs)
        tiling = AttentionTiling(compiled, runtime, selection["query_rows"], selection["kv_heads"])
        record = {"binding_digest": compiled.digest, "region_digest": tiling.digest,
            "eligible_regions": len(tiling.regions), "prefill_calls": 0,
            "tiles": 0, "peak_score_bytes": 0, "phases": [], "failed": False}
        records.append(record)
        if selection["enabled"]:
            runtime._attention_tiling = tiling
        owners[runtime] = (tiling, record)
        return runtime

    def forward(runtime, ids, **kwargs):
        entry = owners.get(runtime)
        if entry is None:
            return original_forward(runtime, ids, **kwargs)
        tiling, record = entry
        if runtime.position == 0:
            record["prefill_calls"] += 1
        try:
            logits = original_forward(runtime, ids, **kwargs)
            digest = hmac.new(salt, b"pllm.attention-state-audit.v1\0", hashlib.sha256)
            _array_commitment(digest, ids)
            _array_commitment(digest, logits)
            checked = logits.size
            for cache in runtime.caches:
                for array in (cache.key[:cache.length], cache.value[:cache.length]):
                    _array_commitment(digest, array)
                    checked += array.size
            record["phases"].append({"position": runtime.position,
                "checked_values": checked, "state_commitment": digest.hexdigest()})
            return logits
        except BaseException:
            record["failed"] = True
            raise
        finally:
            record.update(tiles=tiling.tiles, peak_score_bytes=tiling.peak_score_bytes)

    try:
        CompiledRuntimeModel.runtime = bind
        SemanticDecoderRuntime._forward = forward
        yield records
    finally:
        CompiledRuntimeModel.runtime = original
        SemanticDecoderRuntime._forward = original_forward
        owners.clear()
        _LOCK.release()
