"""Bounded independent ORIGINAL float32/Q11 decoder reference.

Only checkpoint metadata, conversion and checked reads are shared with runtime.
No native GKM arithmetic or semantic executor participates. Decoder equations
are the existing independent generated-fixture oracle; its default Python-int
linear remains a separate cross-check of this scalable implementation.
"""

from __future__ import annotations

import math
import hashlib
import hmac
import json

import numpy as np


ROW_BLOCK = 128
FRAME_BYTES = 64 << 10
CONTRACTION_COLUMNS = 128
INPUT_BOUND = 1 << 31


def corrected_prompt_workspace_estimate(config, *, prefill=30, executed=31, selections=2):
    """Extra-owner allocation estimate, NOT a physical/allocator memory bound.

    Price ALL prefill rows (not only the last head row), full active KV at every
    layer and complete expected/accepted results. Prompt tokenizer retires before
    decoder work; final text decoding can overlap retained complete state.
    The caller keeps the historical 512 MiB allowance cap and refuses a larger
    estimate. Audit, BLAS and tokenizer allowances are heuristic ESTIMATES: no
    allocator, thread-pool or retained-arena ceiling is derived or enforced here.
    Shared reservations/admission/watchdog remain necessary, not launch permission.
    """
    if (prefill, executed, selections) != (30, 31, 2):
        raise ValueError("corrected prompt requires 30 prefill/31 executed/two selections")
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    heads, kv_heads = config["num_attention_heads"], config["num_key_value_heads"]
    width = config.get("head_dim", hidden // heads)
    layers, vocabulary = config["num_hidden_layers"], config["vocab_size"]
    columns = max(hidden, intermediate, heads * width, kv_heads * width)
    # Read/conversion/rint/i16 overlap, including the last loop's live block.
    blocks = 32 * ROW_BLOCK * columns + 2 * FRAME_BYTES
    # Int64 inputs, full f32 linear output, checked accumulation's masks/indexed
    # copies and binary64 operand/chunk overlap. Head output has ONE input row.
    contractions = (8 * prefill * columns
                    + 4 * max(prefill * columns, vocabulary)
                    + 96 * prefill * ROW_BLOCK
                    + 16 * CONTRACTION_COLUMNS * (prefill + ROW_BLOCK))
    # clear_oracle retains loop locals across layers/steps. Include norm/rotary,
    # residuals, qkv, both MLP branches, sigmoid boolean indexing, exp, attention
    # scores/mask/probability and temporary einsum output for EVERY position.
    activations = 4 * (32 * prefill * hidden + 16 * prefill * intermediate
                       + 16 * heads * prefill * executed
                       + 16 * prefill * kv_heads * width)
    # Four complete-state owners cover caches + concat/old overlap, final byte
    # conversion, retained expected and accepted state. Audit consumes rows only.
    complete_state = 4 * (8 * layers * executed * kv_heads * width
                          + 4 * selections * vocabulary)
    components = {
        "weight_blocks_bytes": blocks,
        "contractions_bytes": contractions,
        "all_position_activations_bytes": activations,
        "complete_state_bytes": complete_state,
        "audit_scratch_estimate_bytes": 64 << 20,
        "blas_workspace_estimate_bytes": 64 << 20,
    }
    # Reuse ordinary metadata allowance SIZE, not a claimed tokenizer hard limit.
    # Final decoding overlaps retained results; arenas may outlive parsed objects.
    # This phase model does not prove allocator reclamation between phases.
    components["tokenizer_phase_estimate_bytes"] = (384 << 20) + complete_state
    decoder = sum(value for key, value in components.items()
                  if key != "tokenizer_phase_estimate_bytes")
    components["required_estimate_bytes"] = max(components["tokenizer_phase_estimate_bytes"], decoder)
    return components


def _round_q11(value):
    value = np.asarray(value)
    if value.dtype != np.dtype("float32") or not np.all(np.isfinite(value)):
        raise ValueError("oracle requires finite original float32")
    # Binary32 -> binary64 and power-of-two scaling are exact, including ties.
    return np.rint(np.ldexp(value.astype(np.float64), 11))


def encode_weight_q11(value):
    """Encode original binary32 weights as nearest-even Q11 signed i16."""
    rounded = _round_q11(value)
    if np.any(rounded < -32768) or np.any(rounded > 32767):
        raise ValueError("oracle Q11 exceeds signed i16")
    return rounded.astype(np.int16)


def encode_activation_q11(value):
    """Encode binary32 activations as Q11 int64, with inclusive abs <= 2**31."""
    rounded = _round_q11(value)
    if np.any(rounded < -INPUT_BOUND) or np.any(rounded > INPUT_BOUND):
        raise ValueError("oracle Q11 exceeds activation bound")
    return rounded.astype(np.int64)


def exact_contract(inputs, weights):
    """Exact bounded products over the full admitted input/weight code domain.

    abs(input) <= 2**31 and abs(i16 weight) <= 2**15 imply products <= 2**46.
    Chunks of <= 128 columns bound every partial sum by 2**53 regardless of BLAS
    summation order: all products, additions and FMAs are exact binary64 integers.
    Chunk results convert exactly to int64, then checked int64 addition accumulates
    the COMPLETE contraction. Only the caller converts that total to float32.

    Retain the reference's 2**23 column envelope, but reject an int64 partial-sum
    overflow rather than wrapping. Native decoder width is <= 16_384, giving
    absolute partial sums <= 2**60: its full activation/weight domain fits int64.
    """
    inputs, weights = np.asarray(inputs), np.asarray(weights)
    if (inputs.dtype != np.int64 or weights.dtype != np.int16
            or inputs.ndim != 2 or weights.ndim != 2
            or inputs.shape[1] != weights.shape[1]):
        raise ValueError("oracle contraction requires matching rank-two int64 inputs/i16 weights")
    if not 0 < inputs.shape[1] <= (1 << 23):
        raise ValueError("oracle exact binary64 contraction bound exceeded")
    if np.any(inputs < -INPUT_BOUND) or np.any(inputs > INPUT_BOUND):
        raise ValueError("oracle contraction activation bound exceeded")
    total = np.zeros((len(inputs), len(weights)), dtype=np.int64)
    limits = np.iinfo(np.int64)
    for start in range(0, inputs.shape[1], CONTRACTION_COLUMNS):
        stop = min(start + CONTRACTION_COLUMNS, inputs.shape[1])
        part = (inputs[:, start:stop].astype(np.float64)
                @ weights[:, start:stop].astype(np.float64).T).astype(np.int64)
        positive, negative = part > 0, part < 0
        if (np.any(total[positive] > limits.max - part[positive])
                or np.any(total[negative] < limits.min - part[negative])):
            raise OverflowError("oracle contraction int64 accumulation overflow")
        total += part
    return total


class ArrayTensors:
    def __init__(self, arrays):
        self.arrays = arrays
        self.shapes = {name: tuple(value.shape) for name, value in arrays.items()}

    def __getitem__(self, name):
        return self.arrays[name]

    def read_rows(self, name, begin, end):
        return self.arrays[name][begin:end]


class AdmittedTensors:
    """Lazy bounded reference reads; construct only after external admission.

    Reuse admitted source's frozen specs, source conversion and identity checks,
    never native import/publication. No complete matrix is materialized. Caller
    owns memory admission, watchdog, and before/after source-lock validation.
    """

    def __init__(self, source, *, check=lambda: None):
        self.source, self.check = source, check
        self.shapes = source.source_shapes
        self.specs = dict(source._tensor_specs)
        self.identities = dict(source._identities)

    def verify_lock(self):
        from pllm.runtime.delegated_source import _locked

        self.check()
        source = self.source
        actual = _locked(source.path, source.model, source.source_float32_bytes,
                         source._identities, source.capacity)
        if actual != (source.checkpoint_digest, source.source_lock_digest, source.source_lock):
            raise ValueError("oracle source lock changed")
        self.check()

    def _range(self, name, begin, end):
        from pllm.runtime.delegated_source import _DTYPES, _source_float_frame

        filename, shape, dtype, offset, _ = self.specs[name]
        if not 0 <= begin <= end <= math.prod(shape):
            raise ValueError("oracle source range exceeds admitted shape")
        result = np.empty(end - begin, dtype="<f4")
        for start in range(begin, end, FRAME_BYTES // 4):
            self.check()
            count = min(FRAME_BYTES // 4, end - start)
            raw = _source_float_frame(self.source.path / filename, self.identities[filename],
                                      offset + start * _DTYPES[dtype].itemsize, count, dtype)
            result[start - begin:start - begin + count] = np.frombuffer(raw, dtype="<f4")
        self.check()
        return result

    def __getitem__(self, name):
        shape = self.shapes[name]
        if len(shape) != 1:
            raise ValueError("oracle matrices require bounded row reads")
        return self._range(name, 0, shape[0])

    def read_rows(self, name, begin, end):
        rows, columns = self.shapes[name]
        if not 0 <= begin < end <= rows or end - begin > ROW_BLOCK:
            raise ValueError("oracle row block exceeds bound")
        return self._range(name, begin * columns, end * columns).reshape(end - begin, columns)


def clear_decoder(config, tensors, tokens, decode_steps, *, decode_tokens=None,
                  row_block=ROW_BLOCK, check=lambda: None, selection_observer=None):
    """Complete logits, KV, selected IDs, executed history and position, byte-exact.

    Last selected token is pending: selected[:-1] is executed after prompt; final
    selected token is not inserted into KV/history. Forced decode tokens use same
    convention, enabling fixed-workload comparisons without oracle/native drift.
    Optional selection_observer(logits_bytes, borrowed_caches, token, executed,
    position) runs once per selection; it must not retain or mutate cache arrays.
    SelectionCommitments implements bounded immediate consumption for audits.
    """
    from test_delegated_decoder import clear_oracle

    if not 1 <= row_block <= ROW_BLOCK:
        raise ValueError("oracle row block exceeds bound")
    if not hasattr(tensors, "read_rows"):
        tensors = ArrayTensors(tensors)

    def linear(value, weight, bias=None):
        check()
        inputs = encode_activation_q11(value)
        rows, columns = tensors.shapes[weight]
        if inputs.ndim != 2 or inputs.shape[1] != columns:
            raise ValueError("oracle linear shape mismatch")
        output = np.empty((len(inputs), rows), dtype=np.float32)
        for start in range(0, rows, row_block):
            check()
            stop = min(start + row_block, rows)
            encoded = encode_weight_q11(tensors.read_rows(weight, start, stop))
            integer = exact_contract(inputs, encoded)
            # Contract requires int -> f32 first, *then* f32 / 2048**2.
            output[:, start:stop] = integer.astype(np.float32) / np.float32(4194304)
        return output if bias is None else output + tensors[bias]

    return clear_oracle(config, tensors, tokens, decode_steps, decode_tokens=decode_tokens,
                        linear_reference=linear, selection_observer=selection_observer)


class SelectionCommitments:
    """Independent bounded observer: consume borrowed tensors, retain hashes only.

    Canonical bytes are independently specified here, not delegated to production
    audit helpers. Logits stream in 64 KiB slices, KV one last-dimension row at a
    time. Native-admitted head dimensions keep each temporary row <= 64 KiB.
    Never retain array references, raw token IDs, logit bytes or KV snapshots.
    """

    def __init__(self, salt, *, max_selections=64):
        if type(salt) is not bytes or len(salt) != 32:
            raise ValueError("oracle audit requires private 32-byte salt")
        if type(max_selections) is not int or not 1 <= max_selections <= 64:
            raise ValueError("oracle audit selection bound")
        self.salt, self.max_selections, self.selections = salt, max_selections, []

    def _commit(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self.salt, raw, hashlib.sha256).hexdigest()

    def __call__(self, logits, caches, token, executed, position):
        if len(self.selections) >= self.max_selections:
            raise ValueError("oracle audit selection bound")
        assert len(logits) % 4 == 0 and len(executed) == position
        digest = hmac.new(self.salt, b"pllm.response-gkm-full-state.v1\0", hashlib.sha256)
        digest.update(json.dumps([(len(logits) // 4,), "<f4"], separators=(",", ":")).encode())
        for start in range(0, len(logits), FRAME_BYTES):
            digest.update(logits[start:start + FRAME_BYTES])
        checked, lengths = len(logits) // 4, []
        for key, value in caches:
            assert key.shape == value.shape and key.shape[0] == position
            lengths.append(position)
            for array in (key, value):
                assert array.dtype == np.dtype("<f4") and array.shape[-1] * 4 <= FRAME_BYTES
                digest.update(json.dumps([array.shape, "<f4"], separators=(",", ":")).encode())
                for index in np.ndindex(array.shape[:-1]):
                    # tobytes copies only one bounded final-dimension row, then
                    # releases it immediately after update; no full flatten copy.
                    digest.update(array[index].tobytes(order="C"))
                checked += array.size
        selected = self._commit(token)
        self.selections.append({
            "position": position, "cache_lengths": lengths, "checked_values": checked,
            "logits_and_active_kv": digest.hexdigest(),
            "executed_history": self._commit(executed), "executed_token_count": position,
            "selected_token": selected, "pending_selected_token": selected,
        })

    def expected_record(self, contract):
        numeric = {key: contract[key] for key in ("numeric_profile", "local_numeric_profile")}
        numeric["config_digest"] = contract["plan"]["config_digest"]
        fingerprint = hashlib.sha256(json.dumps(
            numeric, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return {
            "numeric_body_fingerprint": fingerprint, "temperature": 0,
            "sampler_kind": "deterministic_argmax_no_rng",
            "generation_policy": "fixed_length_greedy", "eos_policy": "fixed_count",
            "cryptographic_randomness_frozen": False,
            "selections": self.selections, "failed": False,
        }


def assert_complete_result(result, expected):
    """Check all accepted public state, including final pending selection."""
    actual = (result.logits, result.kv, result.selected, result.executed, result.position)
    assert actual == expected
    logits, kv, selected, executed, position = expected
    assert len(logits) == len(selected)
    assert len(executed) == position
    assert all(shape[0] == position for shape, _, _ in kv)
