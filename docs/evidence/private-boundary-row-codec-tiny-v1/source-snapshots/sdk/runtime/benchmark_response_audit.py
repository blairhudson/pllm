"""Bounded exact full-KV/logit/sampler audit for ordinary cold response targets.

Only salted commitments escape. The measured client owns hashing and the fixed
public greedy sampler fixture; cryptographic random generators are untouched.
"""
from contextlib import contextmanager
import hashlib
import hmac
import json
import threading
import weakref

import numpy as np

from pllm._benchmark_contract import TargetUnavailable

_LOCK = threading.Lock()
MAX_SELECTIONS = 64
SAMPLER_SEED = 42


def state_audit_admission(report, memory_bytes, *, enforced_wan):
    """Price bounded audit records and hash/conversion scratch before execution."""
    import copy
    from .benchmark_memory import admit_memory, host_memory

    estimate = copy.deepcopy(report["estimate"])
    extra = 32 << 20
    estimate["client_peak_bytes"] += extra
    estimate["native_total_peak_bytes"] += extra
    estimate["components"]["response_state_audit_bytes"] = extra
    return admit_memory(estimate, host_memory(), backend="native", enforced_wan=enforced_wan,
                        memory_budget_bytes=memory_bytes)


def _commit(salt, value):
    return hmac.new(salt, json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
                    hashlib.sha256).hexdigest()


def _array(digest, array):
    digest.update(json.dumps([array.shape, array.dtype.str], separators=(",", ":")).encode())
    for chunk in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                           op_flags=["readonly"], order="C", buffersize=16384):
        digest.update(chunk.tobytes())


@contextmanager
def response_state_audit(salt):
    from .semantic_executor import SemanticDecoderRuntime
    from .transformer_client import LayerCache, MaskedTransformerClientRuntime

    if type(salt) is not bytes or len(salt) != 32:
        raise ValueError("state audit requires a private cohort salt")
    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_response_state_audit")
    original = MaskedTransformerClientRuntime.sample
    owners = weakref.WeakKeyDictionary()
    records = []
    selections = 0

    def sample(runtime, logits, **kwargs):
        nonlocal selections
        if (not isinstance(runtime, SemanticDecoderRuntime)
                or not runtime.caches or any(type(cache) is not LayerCache for cache in runtime.caches)
                or kwargs.get("temperature", 0) != 0 or selections >= MAX_SELECTIONS):
            raise TargetUnavailable("state_audit_requires_bounded_greedy_full_kv")
        if runtime not in owners:
            if records:
                raise TargetUnavailable("state_audit_requires_one_response_runtime")
            record = {"sampler_seed": SAMPLER_SEED, "temperature": 0,
                      "numeric_body_fingerprint": runtime.bundle.privacy.get("body_fingerprint"),
                      "cryptographic_randomness_frozen": False, "selections": [], "failed": False}
            records.append(record)
            owners[runtime] = (runtime.rng, record)
            runtime.rng = np.random.default_rng(SAMPLER_SEED)
        _, record = owners[runtime]
        try:
            digest = hmac.new(salt, b"pllm.response-full-state.v1\0", hashlib.sha256)
            _array(digest, np.asarray(logits))
            checked = logits.size
            lengths = []
            for cache in runtime.caches:
                lengths.append(cache.length)
                for array in (cache.key[:cache.length], cache.value[:cache.length]):
                    _array(digest, array)
                    checked += array.size
            before = _commit(salt, runtime.rng.bit_generator.state)
            token = original(runtime, logits, **kwargs)
            record["selections"].append({"position": runtime.position, "cache_lengths": lengths,
                "checked_values": checked, "logits_and_active_kv": digest.hexdigest(),
                "selected_token": _commit(salt, int(token)), "rng_before": before,
                "rng_after": _commit(salt, runtime.rng.bit_generator.state)})
            if runtime._boundary is not None:
                # One bounded latest public snapshot, not per-token payloads or
                # an unbounded trace. Existing audit scratch prices this owner.
                record["private_boundary_head"] = runtime._boundary.head_output_accounting
            selections += 1
            return token
        except BaseException:
            record["failed"] = True
            raise

    try:
        MaskedTransformerClientRuntime.sample = sample
        yield records
    finally:
        MaskedTransformerClientRuntime.sample = original
        for runtime, (rng, _) in tuple(owners.items()):
            runtime.rng = rng
        owners.clear()
        _LOCK.release()
