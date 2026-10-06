"""Authenticated, bounded import of explicitly public full-KV prefills."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import msgpack
import numpy as np

from pllm.configuration import Pipeline
from pllm.profiles import resolve_runtime_composition
from .model_binding import _canonical_json, _state_binding_digest, _state_numeric_contract
from .transformer_client import LayerCache, RuntimeSnapshot, _snapshot_state_basis

SCHEMA = "pllm.public_prefix_capsule.v1"


def _contract(compiled):
    composition = Pipeline.from_spec(json.loads(compiled._canonical_composition))
    options = resolve_runtime_composition(composition)
    if (options is None or options.causal_reduction != "prefix_f32" or not options.prefix_cache_bytes
        or options.verification_component or options.privacy_mode != "public"):
        raise ValueError("public prefix requires unverified prepared full-KV prefix_f32 cache")
    continuation = compiled._plan.continuation_schedule(composition)
    return composition, continuation, hashlib.sha256(_canonical_json(_state_numeric_contract(compiled))).hexdigest()


def _arrays(compiled, ids, states, logits):
    if (not isinstance(ids, (list, tuple)) or not 0 < len(ids) <= 4096
        or any(type(n) is not int or not 0 <= n < int(compiled._bundle.cfg["vocab_size"]) for n in ids)):
        raise ValueError("invalid public prefix token IDs")
    declarations = compiled._plan.to_dict()["decode"]["state_inputs"]
    if len(declarations) != 2 * len(states) or not states:
        raise ValueError("public prefix requires complete full KV")
    checked = set()
    for item in declarations:
        if item["kind"] not in {"key", "value"}:
            raise ValueError("public prefix requires full KV")
        index = (item["layer"], item["kind"])
        if index in checked or not 0 <= item["layer"] < len(states):
            raise ValueError("invalid public prefix state ownership")
        checked.add(index)
        value = states[item["layer"]][int(item["kind"] == "value")]
        if (value.dtype != np.float32 or value.shape != (len(ids), item["shape"][1], item["shape"][3])
            or not np.all(np.isfinite(value))):
            raise ValueError("invalid public prefix state tensor")
    if logits.dtype != np.float32 or logits.shape != (int(compiled._bundle.cfg["vocab_size"]),) or not np.all(np.isfinite(logits)):
        raise ValueError("invalid public prefix logits")


def publish(compiled, public_token_ids, snapshot, logits, path):
    from pllm.state import PublicPrefixCapsule
    compiled.validate()
    _, contract, numeric = _contract(compiled)
    if not isinstance(public_token_ids, (list, tuple)) or not 0 < len(public_token_ids) <= 4096:
        raise ValueError("invalid public prefix length")
    declarations = compiled._plan.to_dict()["decode"]["state_inputs"]
    estimated = 4 * (int(compiled._bundle.cfg["vocab_size"]) + sum(
        len(public_token_ids) * s["shape"][1] * s["shape"][3] for s in declarations))
    if estimated > (64 << 20) - (1 << 20):
        raise ValueError("public prefix exceeds artifact capacity")
    # Reuse the native full-KV and trusted-client provenance validation.
    qualified = compiled.transfer_snapshot(snapshot, compiled)
    if qualified.state_basis.phase != "completed_prefill" or qualified.position != len(public_token_ids):
        raise ValueError("only completed public prefills may be published")
    contract.admit(qualified.position, 1)
    states = [(c.key[:c.length], c.value[:c.length]) for c in qualified.caches]
    logits = np.asarray(logits)
    _arrays(compiled, public_token_ids, states, logits)
    if sum(a.nbytes for pair in states for a in pair) + logits.nbytes > (64 << 20) - (1 << 20):
        raise ValueError("public prefix exceeds artifact capacity")
    payload = msgpack.packb({"schema": SCHEMA, "numeric": numeric, "tokens": list(public_token_ids),
        "states": [[a.astype("<f4", copy=False).tobytes() for a in pair] for pair in states],
        "logits": logits.astype("<f4", copy=False).tobytes()}, use_bin_type=True)
    path = Path(path).resolve()
    with path.open("xb") as stream:
        stream.write(payload)
    return PublicPrefixCapsule(str(path), digest=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))


def import_capsule(compiled, selection, cache, bundle_fingerprint):
    composition, contract, numeric = _contract(compiled)
    bound = selection.params["size_bytes"]
    if bound > cache.max_bytes:
        raise ValueError("public capsule exceeds client cache capacity")
    with Path(selection.params["path"]).open("rb") as stream:
        payload = stream.read(bound + 1)
    if len(payload) != bound or hashlib.sha256(payload).hexdigest() != selection.params["digest"]:
        raise ValueError("public prefix artifact authentication failed")
    document = msgpack.unpackb(payload, raw=False, max_array_len=8192, max_map_len=8,
                              max_str_len=256, max_bin_len=bound)
    if (not isinstance(document, dict) or set(document) != {"schema", "numeric", "tokens", "states", "logits"}
        or document["schema"] != SCHEMA or document["numeric"] != numeric):
        raise ValueError("public prefix source or numeric contract mismatch")
    ids = document["tokens"]
    if not isinstance(ids, list) or not 0 < len(ids) <= 4096:
        raise ValueError("invalid public prefix length")
    contract.admit(len(ids), 1)
    declarations = compiled._plan.to_dict()["decode"]["state_inputs"]
    states = document["states"]
    if not isinstance(states, list) or len(states) * 2 != len(declarations) or any(not isinstance(pair, list) or len(pair) != 2 for pair in states):
        raise ValueError("invalid public prefix state count")
    arrays = [[None, None] for _ in states]
    for item in declarations:
        if item["kind"] not in {"key", "value"} or not 0 <= item["layer"] < len(states):
            raise ValueError("invalid public prefix state declaration")
        layer, slot = item["layer"], int(item["kind"] == "value")
        shape = (len(ids), item["shape"][1], item["shape"][3])
        raw = states[layer][slot]
        if type(raw) is not bytes or len(raw) != int(np.prod(shape)) * 4 or arrays[layer][slot] is not None:
            raise ValueError("invalid public prefix tensor length")
        arrays[layer][slot] = np.frombuffer(raw, "<f4").reshape(shape).copy()
    if type(document["logits"]) is not bytes:
        raise ValueError("invalid public prefix logits")
    logits = np.frombuffer(document["logits"], "<f4").copy()
    _arrays(compiled, ids, arrays, logits)
    binding = _state_binding_digest(compiled._plan, compiled._bundle, composition.digest())
    basis = _snapshot_state_basis("completed_prefill", len(ids), binding, contract.digest,
        selection.params["digest"], None, len(ids))
    snapshot = RuntimeSnapshot(len(ids), [LayerCache(k, v, len(ids)) for k, v in arrays], {}, binding, basis)
    from .prefill_cache import prefill_key
    key = prefill_key(compiled.digest, bundle_fingerprint, ids, causal_reduction="prefix_f32")
    cache.put(key, snapshot, logits)
    if cache.get(key, position=len(ids), layers=len(arrays)) is None:
        raise ValueError("public prefix state failed cache admission")
    return bound
