"""Compiler-bound client-local static retention research.

Uses ordinary checkpoint bindings and semantic execution. Pipeline state selects
this extension under its own execution identity. Selection addresses remain
client-local; snapshots and protected MPCache selection use different contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from pllm import _native
from pllm.passes import KvCacheEviction
from .retained_state import RetainedLayerCache, static_retention
from .transformer_client import LayerCache, TransformerClientError


@dataclass(frozen=True, slots=True)
class RetainedDecoderContract:
    _canonical: bytes
    _plan: bytes
    _composition: bytes
    _policy: bytes
    _memory_bytes: int

    def __post_init__(self):
        if any(type(value) is not bytes for value in (self._canonical, self._plan, self._composition, self._policy)):
            raise TypeError("retained contract documents must be immutable bytes")
        if type(self._memory_bytes) is not int or not 1 <= self._memory_bytes <= 2 << 30:
            raise ValueError("retained state/workspace budget must be in (0, 2 GiB]")
        expected = _native._retained_decoder_contract(self._plan, self._composition, self._policy, self._memory_bytes)
        if self._canonical != expected:
            raise ValueError("retained decoder source/policy/schedule mismatch")

    @property
    def digest(self):
        return hashlib.sha256(b"pllm.retained_decoder.v1\0" + self._canonical).hexdigest()

    def to_dict(self):
        return json.loads(self._canonical)


def compile_retained_contract(compiled, policy: KvCacheEviction, *, memory_bytes: int = 2 << 30) -> RetainedDecoderContract:
    """Compile the static subset of the existing component under client ownership."""
    if type(policy) is not KvCacheEviction:
        raise TypeError("retention requires a KvCacheEviction component")
    if type(memory_bytes) is not int or not 1 <= memory_bytes <= 2 << 30:
        raise ValueError("retained state/workspace budget must be in (0, 2 GiB]")
    compiled.validate()
    plan, composition = compiled._plan.canonical_bytes(), compiled._canonical_composition
    component = json.dumps(policy.to_spec(), sort_keys=True, separators=(",", ":")).encode()
    payload = _native._retained_decoder_contract(plan, composition, component, memory_bytes)
    return RetainedDecoderContract(payload, plan, composition, component, memory_bytes)


def retained_runtime(compiled, remote, *, policy: KvCacheEviction, memory_bytes: int = 2 << 30):
    contract = compile_retained_contract(compiled, policy, memory_bytes=memory_bytes)
    runtime = compiled.runtime(remote)
    runtime._install_retained(contract)
    return runtime


def _ceil(rows, ratio):
    return (rows * ratio["numerator"] + ratio["denominator"] - 1) // ratio["denominator"]


class RetainedExecution:
    """Response-owned controller; holds no second copy of native KV payloads."""

    def __init__(self, runtime, contract: RetainedDecoderContract):
        if type(contract) is not RetainedDecoderContract:
            raise TypeError("retained execution requires an immutable native contract")
        # Revalidate even after Python object construction: this is an admission
        # boundary, not trust in caller-authored source/schedule metadata.
        contract.__post_init__()
        document = contract.to_dict()
        source = dict(runtime._schedule)
        embedded = source.pop("retained_state", None)
        if embedded is not None and embedded != document:
            raise TransformerClientError("retained contract differs from compiled state")
        schedule = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(b"pllm.decoder_runtime_schedule.v2\0" + schedule).hexdigest()
        if (runtime.position or runtime._source_plan.canonical_bytes() != contract._plan
                or digest != document["source_schedule_digest"]
                or len(document["layers"]) != runtime.layers
                or runtime._continuation_contract is not None or runtime.nonlinear_evaluator is not None):
            raise TransformerClientError("retained contract differs from source runtime/state")
        self.contract = contract
        self.document = document
        self.layers = {row["layer"]: row for row in document["layers"]}
        self.policy = document["policy"]["params"]
        self.input_rows = 0
        self.owners = {}
        self.selections = {}
        self.positions = {}
        self.failed = False

    def reset(self):
        for owner in self.owners.values():
            owner.close()
        self.owners.clear()
        self.selections.clear()
        self.positions.clear()
        self.input_rows = 0
        self.failed = False

    def burn(self, runtime):
        self.reset()
        self.failed = True
        # A partial layer update cannot qualify for continuation or retry.
        runtime.caches = [LayerCache() for _ in range(runtime.layers)]
        runtime._snapshot_basis = None

    def begin(self, runtime, rows):
        if self.failed:
            raise TransformerClientError("retained execution is burned")
        self.positions.clear()
        self.selections.clear()
        if runtime.position == 0:
            if self.owners or not 1 <= rows <= self.document["max_input_tokens"]:
                raise TransformerClientError("retained prefill exceeds its contract")
            self.input_rows = rows
        elif (rows != 1 or not self.input_rows
              or runtime.position >= self.input_rows + self.document["max_decode_steps"]
              or set(self.owners) != set(self.layers)):
            raise TransformerClientError("retained decode exceeds its response bound")
        else:
            for layer, owner in self.owners.items():
                if (runtime.caches[layer] is not owner or owner.position != runtime.position
                        or owner.length != self.retained_rows + runtime.position - self.input_rows):
                    raise TransformerClientError("retained state owner/position mismatch")

    @property
    def retained_rows(self):
        return max(_ceil(self.input_rows, self.policy["static_keep"]), self.observation_rows)

    @property
    def observation_rows(self):
        return _ceil(self.input_rows, self.policy["observation_window"])

    @property
    def allocated_bytes(self):
        return sum(owner.allocated_bytes for owner in self.owners.values())

    def observe(self, operation, result, runtime):
        """Compiled semantic hook after probabilities / weighted attention values."""
        if runtime.position != 0 or operation.get("layer") not in self.layers:
            return
        layer = operation["layer"]
        binding = self.layers[layer]["prefill"]
        if operation["id"] == binding["probabilities"]:
            scores = np.asarray(result)
            if (scores.dtype != np.float32 or scores.ndim != 4 or scores.shape[0] != 1
                    or scores.shape[2:] != (self.input_rows, self.input_rows)
                    or not np.all(np.isfinite(scores))):
                raise TransformerClientError("retained importance lacks complete prefill probabilities")
            importance = scores[0, :, -self.observation_rows:, :].mean(axis=(0, 1), dtype=np.float32)
            self.selections[layer] = static_retention(importance, keep=self.retained_rows, recent=self.observation_rows)
        elif operation["id"] == binding["commit_after"]:
            cache = runtime.caches[layer]
            if (layer in self.owners or layer not in self.selections or type(cache) is not LayerCache
                    or cache.length != self.input_rows or cache.key is None or cache.value is None):
                raise TransformerClientError("retained prefill lacks paired state and selected positions")
            owner = RetainedLayerCache(cache.key[:cache.length], cache.value[:cache.length], self.selections.pop(layer),
                capacity=self.retained_rows + self.document["max_decode_steps"],
                maximum_position=self.input_rows + self.document["max_decode_steps"])
            self.owners[layer] = owner
            runtime.caches[layer] = owner

    def append(self, layer, position, key, value):
        owner = self.owners[layer]
        owner.append(position, key, value)
        keys, values, positions = owner.gather()
        self.positions[layer] = positions
        return keys.transpose(1, 0, 2)[None], values.transpose(1, 0, 2)[None]

    def finish(self, position):
        if (self.selections or set(self.owners) != set(self.layers)
                or any(owner.position != position for owner in self.owners.values())):
            raise TransformerClientError("retained decoder did not commit every state owner")
        self.positions.clear()
