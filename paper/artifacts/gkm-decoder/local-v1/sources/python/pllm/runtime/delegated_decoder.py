"""Internal fixed-length GKM decoder reference; all logical parties are local.

Original source float32 table bytes and nearest-even/2048 linears replace the
ordinary quantized boundary. This module admits no Pipeline, source downloader,
provider transport, reusable KV, stochastic sampler, or streaming acceptance.
The caller owns source buffers. Native publication/window payloads, local
metadata, decoder tensors and returned snapshots are distinct memory owners.
"""

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any, Mapping

import numpy as np

from pllm import _native
from .semantic_executor import SemanticLocalOperators
from .transformer_client import LayerCache


@dataclass(frozen=True)
class GkmDecoderResult:
    """Immutable accepted full logits/KV, with one final pending greedy token."""

    logits: tuple[bytes, ...]
    kv: tuple[tuple[tuple[int, ...], bytes, bytes], ...]
    selected: tuple[int, ...]
    executed: tuple[int, ...]
    position: int
    acceptance: bytes


def decoder_contract(config: Mapping[str, Any], *, input_rows: int, decode_steps: int) -> bytes:
    """Check public geometry before the caller reads or converts any tensors."""
    if type(input_rows) is not int or type(decode_steps) is not int:
        raise ValueError("GKM workload sizes must be integers")
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _native.gkm_decoder_contract(encoded, input_rows, decode_steps)


class _Request(SemanticLocalOperators):
    def __init__(self, native, contract):
        self._native = native
        self._contract = contract
        self._graphs = {phase: contract["plan"][phase] for phase in ("prefill", "decode")}
        self._coefficients = {key: np.frombuffer(value, dtype="<f4") for key, value in native.coefficients().items()}
        layers = {state["layer"] for state in self._graphs["decode"]["state_outputs"]}
        self.caches = [LayerCache() for _ in range(len(layers))]
        self.position = 0
        self._retained = self._tiered = self._causal_reduction = self.nonlinear_evaluator = None

    def _weight(self, name):
        return self._coefficients[name]

    def phase(self, ids):
        phase = "prefill" if self.position == 0 else "decode"
        graph, schedule = self._graphs[phase], self._contract[phase]
        operations = {op["id"]: op for op in graph["operations"]}
        steps = schedule["steps"]
        remaining = Counter(key for step in steps for key in step["input_ids"])
        logits_id = next(op["inputs"][0] for op in operations.values() if op["operator"] == "greedy_token_selection")
        values = {
            "input.tokens": np.asarray(ids, dtype=np.int64),
            "input.positions": np.arange(self.position, self.position + len(ids), dtype=np.int64),
            "input.sequence_lengths": np.asarray([self.position + len(ids)], dtype=np.int64),
            "input.attention_mask": np.ones((1, len(ids)), dtype=np.bool_),
        }
        for state in graph["state_inputs"]:
            cache = self.caches[state["layer"]]
            stored = cache.key if state["kind"] == "key" else cache.value
            values[state["id"]] = stored[:cache.length].transpose(1, 0, 2)[None]
        pending = {}
        kinds = self._declared_state_kinds(graph)
        stage = 0
        try:
            for step in steps:
                if step["executor"] == "client_local":
                    for key in step["operation_ids"]:
                        values[key] = self._local(operations[key], values, kinds, pending)
                elif step["operators"][0] == "token_lookup":
                    rows = [np.frombuffer(self._native.lookup(token), dtype="<f4") for token in ids]
                    values[step["operation_ids"][0]] = np.stack(rows)
                else:
                    spec = self._contract["matrices"][stage]
                    source = values[step["input_ids"][0]]
                    source = np.asarray(source, dtype="<f4").reshape(-1, spec["columns"])
                    rows = [np.frombuffer(self._native.linear(stage, row.tobytes()), dtype="<f4") for row in source]
                    combined = np.stack(rows)
                    for key, member in zip(step["operation_ids"], spec["members"], strict=True):
                        start, width = member["offset"], member["rows"]
                        output = combined[:, start:start + width]
                        if member["bias"] is not None:
                            output = output + self._weight(member["bias"])
                        values[key] = output
                    stage += 1
                for key in step["operation_ids"]:
                    value = np.asarray(values[key])
                    # Causal masks deliberately contain -inf; all other local
                    # outputs must remain finite before use or acceptance.
                    if operations[key]["operator"] != "causal_mask" and not np.all(np.isfinite(value)):
                        raise ValueError("GKM local arithmetic left its finite domain")
                for key in step["input_ids"]:
                    remaining[key] -= 1
                    if remaining[key] == 0 and key != logits_id:
                        values.pop(key, None)
            if pending or stage != len(self._contract["matrices"]):
                raise ValueError("GKM phase is incomplete")
            logits = np.asarray(values[logits_id], dtype="<f4").reshape(-1)
            if logits.size != self._contract["vocabulary"]:
                raise ValueError("GKM full-head shape mismatch")
            self.position += len(ids)
            return logits.tobytes(), int(np.argmax(logits))
        finally:
            values.clear()

    def retire(self):
        self._native.abort()
        for cache in self.caches:
            for array in (cache.key, cache.value):
                if array is not None:
                    array.fill(0)
            cache.key = cache.value = None
            cache.length = 0
        self._coefficients.clear()


def execute_gkm_reference(
    config: Mapping[str, Any], tensors: Mapping[str, bytes], token_ids: tuple[int, ...],
    *, decode_steps: int = 1, seeded: bool = False,
) -> GkmDecoderResult:
    """Execute one bounded greedy window; publish state only after verification.

    ``tensors`` contains exactly the contract's source names, as finite little-
    endian float32 bytes. No integer-quantized bundle is accepted. Prompt length
    and the fixed number of decode steps are public. EOS does not shorten this
    explicitly fixed-length conformance workload. No snapshot can resume it.
    """
    if type(token_ids) is not tuple or not 1 <= len(token_ids) <= 32 or any(type(t) is not int for t in token_ids):
        raise ValueError("GKM token_ids must be a bounded tuple of integers")
    if type(seeded) is not bool:
        raise ValueError("GKM seeded selector must be boolean")
    encoded = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    contract = json.loads(decoder_contract(config, input_rows=len(token_ids), decode_steps=decode_steps))
    if any(not 0 <= t < contract["vocabulary"] for t in token_ids):
        raise ValueError("GKM token index is outside its table")
    native = _native._GkmDecoderSession(encoded, len(token_ids), decode_steps, dict(tensors), seeded=seeded)
    request = None
    try:
        if json.loads(native.contract()) != contract:
            raise ValueError("GKM compiled contract mismatch")
        request = _Request(native, contract)
        logits, selected, executed = [], [], list(token_ids)
        current = token_ids
        for step in range(decode_steps + 1):
            vector, token = request.phase(current)
            logits.append(vector)
            selected.append(token)
            if step < decode_steps:
                current = (token,)
                executed.append(token)
        acceptance = native.finish()
        kv = tuple((tuple(cache.key[:cache.length].shape),
                    cache.key[:cache.length].astype("<f4", copy=False).tobytes(),
                    cache.value[:cache.length].astype("<f4", copy=False).tobytes()) for cache in request.caches)
        return GkmDecoderResult(tuple(logits), kv, tuple(selected), tuple(executed), request.position, acceptance)
    finally:
        if request is not None:
            request.retire()
        native.abort()
