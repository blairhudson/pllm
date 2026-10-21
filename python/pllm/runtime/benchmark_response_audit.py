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
GKM_AUDIT_SCHEMA = "pllm.response_state_audit.gkm.v1"
_GKM_OWNER = None


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


class _GkmRequestAudit:
    """Trusted, consuming request instrumentation; retains commitments only."""

    def __init__(self, owner, contract, tokens):
        self.owner = owner
        self.initial_count = len(tokens)
        self.initial_commit = _commit(owner.salt, tokens)
        self.count = contract["decode_steps"] + 1
        self.vocabulary = contract["vocabulary"]
        numeric = {key: contract[key] for key in ("numeric_profile", "local_numeric_profile")}
        numeric["config_digest"] = contract["plan"]["config_digest"]
        self.record = {
            "numeric_body_fingerprint": hashlib.sha256(json.dumps(
                numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "temperature": 0, "sampler_kind": "deterministic_argmax_no_rng",
            "generation_policy": "fixed_length_greedy", "eos_policy": "fixed_count",
            "cryptographic_randomness_frozen": False, "selections": [], "failed": False,
        }

    def selection(self, request, vector, token, executed):
        # This method is called only by the trusted decoder, never a user callback.
        with self.owner.lock:
            if self.owner.closed or self.record is None:
                raise TargetUnavailable("response_state_audit_retired")
            rows = self.record["selections"]
            position = self.initial_count + len(rows)
            if (len(rows) >= self.count or request.position != position
                    or len(executed) != position
                    or _commit(self.owner.salt, executed[:self.initial_count]) != self.initial_commit
                    or not request.caches or type(vector) is not bytes
                    or len(vector) != self.vocabulary * 4 or type(token) is not int):
                raise TargetUnavailable("response_state_audit_incomplete", status="failed")
            logits = np.frombuffer(vector, dtype="<f4")
            if not np.isfinite(logits).all() or token != int(np.argmax(logits)):
                raise TargetUnavailable("state_audit_requires_bounded_greedy_full_kv")
            digest = hmac.new(self.owner.salt, b"pllm.response-gkm-full-state.v1\0", hashlib.sha256)
            _array(digest, logits)
            checked, lengths = logits.size, []
            for cache in request.caches:
                if cache.length != position:
                    raise TargetUnavailable("response_state_audit_incomplete", status="failed")
                lengths.append(cache.length)
                for array in (cache.key[:cache.length], cache.value[:cache.length]):
                    if array.dtype != np.dtype("<f4") or array.shape[0] != position:
                        raise TargetUnavailable("state_audit_requires_bounded_greedy_full_kv")
                    _array(digest, array)
                    checked += array.size
            selected = _commit(self.owner.salt, token)
            rows.append({"position": position, "cache_lengths": lengths,
                "checked_values": checked, "logits_and_active_kv": digest.hexdigest(),
                "executed_history": _commit(self.owner.salt, executed),
                "executed_token_count": position, "selected_token": selected,
                "pending_selected_token": selected})

    def accept(self, result):
        with self.owner.lock:
            if self.owner.closed or self.record is None:
                raise TargetUnavailable("response_state_audit_retired")
            rows = self.record["selections"]
            if (len(rows) != self.count or type(result.acceptance) is not bytes
                    or not 0 < len(result.acceptance) <= 16 << 10
                    or len(result.selected) != self.count
                    or result.position != self.initial_count + self.count - 1
                    or len(result.executed) != result.position
                    or _commit(self.owner.salt, result.executed[:self.initial_count]) != self.initial_commit
                    or result.executed[self.initial_count:] != result.selected[:-1]
                    or any(row["selected_token"] != _commit(self.owner.salt, token)
                           for row, token in zip(rows, result.selected, strict=True))):
                raise TargetUnavailable("response_state_audit_incomplete", status="failed")
            self.owner.accepted = self.record
            self.record = None

    def close(self):
        with self.owner.lock:
            if self.record is not None:
                self.record["selections"].clear()
                self.record = None


class _GkmAuditOwner:
    def __init__(self, salt):
        self.salt, self.lock = salt, threading.Lock()
        self.closed = self.claimed = False
        self.accepted = None

    def begin(self, contract, tokens, decode_tokens):
        with self.lock:
            if self.closed or self.claimed:
                self.accepted = None
                raise TargetUnavailable("state_audit_requires_one_response_runtime")
            self.claimed = True
            if (decode_tokens is not None or not 1 <= contract["decode_steps"] + 1 <= MAX_SELECTIONS
                    or type(tokens) is not tuple or not 1 <= len(tokens) <= 512
                    or any(type(token) is not int or not 0 <= token < contract["vocabulary"] for token in tokens)):
                raise TargetUnavailable("state_audit_requires_bounded_greedy_full_kv")
            return _GkmRequestAudit(self, contract, tokens)


def _begin_gkm_request_audit(contract, tokens, decode_tokens):
    """Internal decoder seam. Disabled execution allocates no audit owner/state."""
    owner = _GKM_OWNER
    return None if owner is None else owner.begin(contract, tokens, decode_tokens)


@contextmanager
def _gkm_response_state_audit(salt):
    # Process-wide like the prepared instrumentation: dashboard executes on a
    # different thread. The shared lock excludes overlapping audit scopes.
    global _GKM_OWNER
    if type(salt) is not bytes or len(salt) != 32:
        raise ValueError("state audit requires a private cohort salt")
    if not _LOCK.acquire(blocking=False):
        raise TargetUnavailable("concurrent_response_state_audit")
    owner, records = _GkmAuditOwner(salt), []
    _GKM_OWNER = owner
    try:
        yield records
        with owner.lock:
            if owner.accepted is not None:
                records.append(owner.accepted)
    finally:
        with owner.lock:
            owner.closed = True
            owner.accepted = None
        _GKM_OWNER = None
        _LOCK.release()


@contextmanager
def response_state_audit(salt, *, runtime_kind=None):
    if runtime_kind == "gkm":
        with _gkm_response_state_audit(salt) as records:
            yield records
        return
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
