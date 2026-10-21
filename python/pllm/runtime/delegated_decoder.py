"""Internal fixed-length GKM decoder reference; all logical parties are local.

Original source float32 table bytes and nearest-even/2048 linears replace the
ordinary quantized boundary. This module admits no Pipeline, source downloader,
provider transport, reusable KV, stochastic sampler, or streaming acceptance.
The caller owns source buffers. Native publication/window payloads, local
metadata, decoder tensors and returned snapshots are distinct memory owners.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from typing import Any

import numpy as np

from pllm import _native
from .delegated_protocol import clear_failure_frames
from .semantic_executor import SemanticLocalOperators
from .transformer_client import LayerCache


_MAX_CONFIG_BYTES = 16 << 10
_MAX_CONFIG_DEPTH = 64
_BATCH_EXECUTION = "source_stage_batch_v1"


def _batch_output(raw, rows, width):
    """Borrow one immutable native reconstruction; never stack/copy its rows."""
    try:
        if type(raw) is not bytes or len(raw) != 4 * rows * width:
            raise ValueError("GKM batch reconstruction shape mismatch")
        return np.frombuffer(raw, dtype="<f4").reshape(rows, width)
    finally:
        raw = None


def result_payload_limit(contract, *, capacity="capsule"):
    """Return the native-admitted immutable result allowance before execution."""
    if capacity == "capsule":
        return 16 << 20
    resources = contract.get("decoder_resources", {})
    limit = resources.get("accepted_result_bytes")
    total = contract.get("execution_payload_estimate_bytes")
    if (capacity != "decoder" or type(limit) is not int or type(total) is not int
            or not 0 < limit <= total <= 4 << 30):
        raise ValueError("GKM decoder result has no admitted bound")
    return limit


def _encode_config(config: Mapping[str, Any]) -> bytes:
    """Snapshot at most 16 KiB / 64 levels of JSON before bulk serialization."""
    remaining = _MAX_CONFIG_BYTES
    end = object()

    def charge(size):
        nonlocal remaining
        if size > remaining:
            raise ValueError("GKM configuration exceeds its 16 KiB limit")
        remaining -= size

    def string(value):
        if type(value) is not str:
            raise ValueError("GKM configuration requires string object keys")
        # Count ensure_ascii=True escaping before allocating an encoded string.
        if len(value) + 2 > remaining:
            raise ValueError("GKM configuration exceeds its 16 KiB limit")
        charge(2)
        for char in value:
            point = ord(char)
            if char in '"\\\b\f\n\r\t':
                charge(2)
            elif 0x20 <= point < 0x7F:
                charge(1)
            else:
                charge(6 if point <= 0xFFFF else 12)
        return value

    def snapshot(value, depth):
        if depth > _MAX_CONFIG_DEPTH:
            raise ValueError("GKM configuration nesting exceeds its limit")
        if type(value) is str:
            return string(value)
        if value is None or type(value) in (bool, int, float):
            # Decimal conversion must not first expand an arbitrary-size integer.
            if type(value) is int and value.bit_length() > remaining * 4:
                raise ValueError("GKM configuration integer exceeds its byte limit")
            charge(len(json.dumps(value, allow_nan=False)))
            return value
        if isinstance(value, Mapping):
            count = len(value)
            if (5 * count + 1 if count else 2) > remaining:
                raise ValueError("GKM configuration object exceeds its byte limit")
            charge(2 + count + max(0, count - 1))  # Braces, colons, commas.
            result = {}
            keys = iter(value)
            for _ in range(count):
                key = next(keys, end)
                if key is end:
                    raise ValueError("GKM configuration mapping changed")
                key = string(key)
                if key in result:
                    raise ValueError("GKM configuration has duplicate keys")
                result[key] = snapshot(value[key], depth + 1)
            if next(keys, end) is not end or len(value) != count:
                raise ValueError("GKM configuration mapping changed")
            return result
        if type(value) in (list, tuple):
            count = len(value)
            if (2 * count + 1 if count else 2) > remaining:
                raise ValueError("GKM configuration array exceeds its byte limit")
            charge(2 + max(0, count - 1))
            result = [snapshot(value[index], depth + 1) for index in range(count)]
            if len(value) != count:
                raise ValueError("GKM configuration array changed")
            return result
        raise ValueError("GKM configuration contains an unsupported JSON value")

    if not isinstance(config, Mapping):
        raise ValueError("GKM configuration must be a mapping")
    try:
        document = snapshot(config, 0)
    except (KeyError, IndexError, RuntimeError) as exc:
        raise ValueError("GKM configuration changed during admission") from exc
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    if len(encoded) != _MAX_CONFIG_BYTES - remaining:
        raise ValueError("GKM configuration byte accounting mismatch")
    return encoded


def _source_snapshot(tensors: Mapping[str, bytes], shapes) -> dict[str, bytes]:
    """Read only declared names; retain immutable bytes and detect observed changes."""
    count = len(shapes)
    if not isinstance(tensors, Mapping) or len(tensors) != count:
        raise ValueError("GKM source tensor set mismatch")
    result = {}
    try:
        for name, shape in shapes.items():
            value = tensors[name]
            if not isinstance(value, bytes) or bytes.__len__(value) != math.prod(shape) * 4:
                raise ValueError("GKM source must contain exact-shape float32 bytes")
            result[name] = value
        if len(tensors) != count:
            raise ValueError("GKM source mapping changed")
        # A second bounded lookup pass catches replaced values as well as removed
        # names. Equal freshly loaded bytes remain valid; caller keys are never enumerated.
        for name, prior in result.items():
            current = tensors[name]
            if (
                not isinstance(current, bytes)
                or bytes.__len__(current) != bytes.__len__(prior)
                or not bytes.__eq__(current, prior)
            ):
                raise ValueError("GKM source mapping changed")
        if len(tensors) != count:
            raise ValueError("GKM source mapping changed")
    except (KeyError, RuntimeError) as exc:
        raise ValueError("GKM source mapping is missing a declared tensor or changed") from exc
    return result


@dataclass(frozen=True)
class GkmDecoderResult:
    """Immutable accepted full logits/KV, with one final pending greedy token."""

    logits: tuple[bytes, ...]
    kv: tuple[tuple[tuple[int, ...], bytes, bytes], ...]
    selected: tuple[int, ...]
    executed: tuple[int, ...]
    position: int
    acceptance: bytes


def decoder_contract(config: Mapping[str, Any], *, input_rows: int, decode_steps: int,
                     capacity: str = "capsule", streamed: bool = False) -> bytes:
    """Check geometry before tensor access; streamed selects SDK ownership admission."""
    if type(input_rows) is not int or type(decode_steps) is not int:
        raise ValueError("GKM workload sizes must be integers")
    encoded = _encode_config(config)
    if type(capacity) is not str or capacity not in {"capsule", "decoder"}:
        raise ValueError("GKM capacity must be capsule or decoder")
    if type(streamed) is not bool:
        raise ValueError("GKM streamed must be a boolean")
    if streamed:
        return _native.gkm_decoder_contract(encoded, input_rows, decode_steps,
                                            capacity=capacity, streamed=True)
    if capacity != "capsule":
        return _native.gkm_decoder_contract(encoded, input_rows, decode_steps, capacity=capacity)
    return _native.gkm_decoder_contract(encoded, input_rows, decode_steps)


class _Request(SemanticLocalOperators):
    """One request's local graph, with optional native-selected stage batching.

    Python batch owner inventory (B rows, C input columns, R combined outputs):
    * Reconstruction O=4BR is the graph's combined root, not another copy beside
      its member views. Views/reshape/last-row aliases can retain all O bytes.
      The compiler decoder_tensor_bytes already prices these complete roots and
      alias lifetimes. Bias outputs and finite-check temporaries are unchanged.
    * Dtype coercion can own I=4BC alongside the graph input, as in scalar mode.
      C-order serialization additionally owns I through linear_batch's return.
      No flatten/contiguous array copy or per-row reconstruction list is made.
      An extra max(I) reserve conservatively prices batch serialization without
      charging coercion or O again; subtract a scalar-row reserve only if that
      reserve is explicitly present in the caller's existing accounting.
    * All raw/combined/input/serialization/output loop aliases are cleared before
      proceeding, including finally on failure/cancellation. Only graph-reachable
      views retain reconstructed bytes. Phase exit clears the complete graph.

    Native/HTTP frames, native accepted buffers, KV capacity, final snapshots and
    audit scratch are separate existing owners, not hidden in this inventory.
    """

    def __init__(self, native, contract):
        self._native = native
        self._contract = contract
        # Only the consuming native owner selects execution. Decoder/source
        # identity deliberately stays unchanged; caller metadata cannot opt in.
        execution = getattr(native, "execution", lambda: None)()
        if execution not in (None, _BATCH_EXECUTION):
            raise ValueError("GKM native execution profile is unsupported")
        self._batch = execution == _BATCH_EXECUTION
        self._phase_index = self._batch_claims = 0
        self._graphs = {phase: contract["plan"][phase] for phase in ("prefill", "decode")}
        self._coefficients = {
            key: np.frombuffer(value, dtype="<f4") for key, value in native.coefficients().items()
        }
        layers = {state["layer"] for state in self._graphs["decode"]["state_outputs"]}
        self.caches = [LayerCache() for _ in range(len(layers))]
        self.position = 0
        self._retained = self._tiered = self._causal_reduction = self.nonlinear_evaluator = None

    def _weight(self, name):
        return self._coefficients[name]

    def phase(self, ids):
        phase = "prefill" if self.position == 0 else "decode"
        graph, schedule = self._graphs[phase], self._contract[phase]
        if self._batch and (self._phase_index > self._contract["decode_steps"]
                            or len(ids) != schedule["query_sequence"]):
            raise ValueError("GKM batch phase differs from admitted row count")
        operations = {op["id"]: op for op in graph["operations"]}
        steps = schedule["steps"]
        remaining = Counter(key for step in steps for key in step["input_ids"])
        logits_id = next(
            op["inputs"][0]
            for op in operations.values()
            if op["operator"] == "greedy_token_selection"
        )
        values = {
            "input.tokens": np.asarray(ids, dtype=np.int64),
            "input.positions": np.arange(self.position, self.position + len(ids), dtype=np.int64),
            "input.sequence_lengths": np.asarray([self.position + len(ids)], dtype=np.int64),
            "input.attention_mask": np.ones((1, len(ids)), dtype=np.bool_),
        }
        for state in graph["state_inputs"]:
            cache = self.caches[state["layer"]]
            stored = cache.key if state["kind"] == "key" else cache.value
            values[state["id"]] = stored[: cache.length].transpose(1, 0, 2)[None]
        pending = {}
        kinds = self._declared_state_kinds(graph)
        stage = 0
        raw = serialized = None
        try:
            for step in steps:
                if step["executor"] == "client_local":
                    for key in step["operation_ids"]:
                        values[key] = self._local(operations[key], values, kinds, pending)
                elif step["operators"][0] == "token_lookup":
                    if self._batch:
                        batch_rows = self._batch_rows(step, schedule)
                        raw = self._native.lookup_batch(tuple(ids))
                        values[step["operation_ids"][0]] = _batch_output(
                            raw, batch_rows, self._contract["width"])
                        raw = None
                        self._batch_claims += batch_rows
                    else:
                        rows = [np.frombuffer(self._native.lookup(token), dtype="<f4") for token in ids]
                        values[step["operation_ids"][0]] = np.stack(rows)
                else:
                    spec = self._contract["matrices"][stage]
                    source = values[step["input_ids"][0]]
                    if self._batch:
                        batch_rows = self._batch_rows(step, schedule, spec)
                        source = np.asarray(source)
                        if (source.ndim < 2 or source.shape[-1] != spec["columns"]
                                or math.prod(source.shape[:-1]) != batch_rows):
                            raise ValueError("GKM batch input shape differs from admitted stage")
                        source = np.asarray(source, dtype="<f4")
                        # No contiguous coercion/flatten copy: tobytes handles
                        # strided views directly. At most dtype coercion I plus
                        # immutable serialization I coexist with graph input.
                        serialized = source.tobytes(order="C")
                        raw = self._native.linear_batch(stage, serialized)
                        combined = _batch_output(raw, batch_rows, spec["rows"])
                        serialized = raw = None
                        self._batch_claims += batch_rows
                    else:
                        source = np.asarray(source, dtype="<f4").reshape(-1, spec["columns"])
                        rows = [
                            np.frombuffer(self._native.linear(stage, row.tobytes()), dtype="<f4")
                            for row in source
                        ]
                        combined = np.stack(rows)
                    for key, member in zip(step["operation_ids"], spec["members"], strict=True):
                        start, width = member["offset"], member["rows"]
                        output = combined[:, start : start + width]
                        if member["bias"] is not None:
                            output = output + self._weight(member["bias"])
                        values[key] = output
                    stage += 1
                    if self._batch:
                        # Member views alone retain reconstruction O=4BR. Do not
                        # keep a dead full backing through these loop locals.
                        source = combined = output = None
                for key in step["operation_ids"]:
                    value = np.asarray(values[key])
                    # Causal masks deliberately contain -inf; all other local
                    # outputs must remain finite before use or acceptance.
                    if operations[key]["operator"] != "causal_mask" and not np.all(
                        np.isfinite(value)
                    ):
                        raise ValueError("GKM local arithmetic left its finite domain")
                    if self._batch:
                        value = None
                for key in step["input_ids"]:
                    remaining[key] -= 1
                    if remaining[key] == 0 and key != logits_id:
                        values.pop(key, None)
            if pending or stage != len(self._contract["matrices"]):
                raise ValueError("GKM phase is incomplete")
            if self._batch:
                self._phase_index += 1
                if (self._phase_index == self._contract["decode_steps"] + 1
                        and self._batch_claims != self._contract["window_steps"]):
                    raise ValueError("GKM batch scalar-claim cursor is incomplete")
            logits = np.asarray(values[logits_id], dtype="<f4").reshape(-1)
            if logits.size != self._contract["vocabulary"]:
                raise ValueError("GKM full-head shape mismatch")
            self.position += len(ids)
            return logits.tobytes(), int(np.argmax(logits))
        finally:
            values.clear()
            pending.clear()
            source = rows = combined = output = value = logits = stored = cache = None
            raw = serialized = None

    def _batch_rows(self, step, schedule, spec=None):
        """Compiler BatchPlan cursor: phase 0 prefill, subsequent phases decode.

        A remote schedule step is one source-stage descriptor, including grouped
        members. Output-head alone consumes the last row; all decode rows are 1.
        Native consumes/binds the actual phase/operation/claim cursor as well.
        """
        rows = 1 if step["operators"][0] == "output_head" else schedule["query_sequence"]
        outputs = step["outputs"]
        members = spec["members"] if spec is not None else [
            {"offset": 0, "rows": self._contract["width"]}]
        if len(outputs) != len(members) or len(outputs) != len(step["operation_ids"]):
            raise ValueError("GKM batch output descriptors differ from source stage")
        for key, output, member in zip(step["operation_ids"], outputs, members, strict=True):
            shape = output["output_shape"]
            if (not shape or math.prod(shape[:-1]) != rows or shape[-1] != member["rows"]
                    or output["operation_id"] != key or output["stage_offset"] != member["offset"]
                    or output["stage_width"] != member["rows"]):
                raise ValueError("GKM batch output descriptors differ from source stage")
        return rows

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
    config: Mapping[str, Any],
    tensors: Mapping[str, bytes],
    token_ids: tuple[int, ...],
    *,
    decode_steps: int = 1,
    seeded: bool = False,
) -> GkmDecoderResult:
    """Execute one bounded greedy window; publish state only after verification.

    ``tensors`` contains exactly the contract's source names, as finite little-
    endian float32 bytes. No integer-quantized bundle is accepted. Prompt length
    and the fixed number of decode steps are public. EOS does not shorten this
    explicitly fixed-length conformance workload. No snapshot can resume it.
    """
    if (
        type(token_ids) is not tuple
        or not 1 <= len(token_ids) <= 32
        or any(type(t) is not int for t in token_ids)
    ):
        raise ValueError("GKM token_ids must be a bounded tuple of integers")
    if type(seeded) is not bool:
        raise ValueError("GKM seeded selector must be boolean")
    if type(decode_steps) is not int:
        raise ValueError("GKM workload sizes must be integers")
    encoded = _encode_config(config)
    contract = json.loads(_native.gkm_decoder_contract(encoded, len(token_ids), decode_steps))
    if any(not 0 <= t < contract["vocabulary"] for t in token_ids):
        raise ValueError("GKM token index is outside its table")
    native = _native._GkmDecoderSession(
        encoded,
        len(token_ids),
        decode_steps,
        _source_snapshot(tensors, contract["source_shapes"]),
        seeded=seeded,
    )
    return _execute_verified_window(native, contract, token_ids)


def _validate_decode_tokens(tokens, steps, vocabulary):
    """Bound the shared quality driver's optional teacher-forced continuation."""
    if tokens is not None and (
        type(tokens) is not tuple or len(tokens) != steps
        or any(type(token) is not int or not 0 <= token < vocabulary for token in tokens)
    ):
        raise ValueError("GKM quality continuation differs from the admitted decode geometry")


def _execute_verified_window(native, contract, token_ids, *, cancelled=None,
                              decode_tokens=None) -> GkmDecoderResult:
    # No callback can expose tentative tensors. The internal benchmark owner
    # hashes live state in place and publishes only after this complete lifecycle.
    from .benchmark_response_audit import _begin_gkm_request_audit

    audit = result = None
    execution_entered = False
    try:
        audit = _begin_gkm_request_audit(contract, token_ids, decode_tokens)
        execution_entered = True
        result = _execute_verified_window_inner(
            native, contract, token_ids, cancelled=cancelled,
            decode_tokens=decode_tokens, _audit=audit,
        )
        if audit is not None and cancelled is not None and cancelled():
            raise RuntimeError("GKM request cancelled")
        if audit is not None:
            audit.accept(result)
        return result
    except BaseException as error:
        if not execution_entered:
            native.abort()
        clear_failure_frames(error)
        raise
    finally:
        if audit is not None:
            audit.close()
        result = None


def _execute_verified_window_inner(native, contract, token_ids, *, cancelled=None,
                                   decode_tokens=None, _audit=None) -> GkmDecoderResult:
    """Shared local semantics; native finish gates all returned state and tokens.

    The local reference and the role-separated runtime supply distinct native
    owners. Neither receives a callback that can publish tentative results.
    """
    request = vector = kv = acceptance = None
    logits = selected = executed = current = None
    try:
        if json.loads(native.contract()) != contract:
            raise ValueError("GKM compiled contract mismatch")
        _validate_decode_tokens(decode_tokens, contract["decode_steps"], contract["vocabulary"])
        if (
            type(token_ids) is not tuple
            or len(token_ids) != contract["input_rows"]
            or any(type(token) is not int or not 0 <= token < contract["vocabulary"]
                   for token in token_ids)
        ):
            raise ValueError("GKM input differs from admitted request")
        request = _Request(native, contract)
        logits, selected, executed = [], [], list(token_ids)
        current = token_ids
        decode_steps = contract["decode_steps"]
        for step in range(decode_steps + 1):
            if cancelled is not None and cancelled():
                raise RuntimeError("GKM request cancelled")
            vector, token = request.phase(current)
            if _audit is not None:
                _audit.selection(request, vector, token, executed)
            logits.append(vector)
            selected.append(token)
            if step < decode_steps:
                current = (token if decode_tokens is None else decode_tokens[step],)
                executed.append(current[0])
        if cancelled is not None and cancelled():
            raise RuntimeError("GKM request cancelled")
        acceptance = native.finish()
        kv = tuple(
            (
                tuple(cache.key[: cache.length].shape),
                cache.key[: cache.length].astype("<f4", copy=False).tobytes(),
                cache.value[: cache.length].astype("<f4", copy=False).tobytes(),
            )
            for cache in request.caches
        )
        return GkmDecoderResult(
            tuple(logits), kv, tuple(selected), tuple(executed), request.position, acceptance
        )
    except BaseException as error:
        clear_failure_frames(error)
        raise
    finally:
        try:
            if request is not None:
                request.retire()
        finally:
            request = vector = kv = acceptance = None
            logits = selected = executed = current = None
            native.abort()
