"""Source-stage Python batching, independent of pending native batch bindings.

Use generated public compiler contracts and a strict fake consuming native owner.
The fake implements arithmetic only to check local operator/bias ordering; it is
not evidence for the native transport, signatures, seeded expansion or admission.
"""

import asyncio
import copy
import gc
import json
import weakref

import numpy as np
import pytest

from pllm.runtime import benchmark_response_audit as audit
from pllm.runtime import delegated_decoder as decoder

from gkm_decoder_oracle import SelectionCommitments, assert_complete_result
from test_delegated_decoder import clear_oracle, config, fixture


class BatchNative:
    """One consuming call per source stage; scalar fallback always fails."""

    def __init__(self, contract, arrays, *, seeded=False, failure=None):
        self.document = copy.deepcopy(contract)
        self.arrays, self.seeded, self.failure = arrays, seeded, failure
        self.phase = self.stage = 0
        self.calls = []
        self.aborted = self.finished = False

    def execution(self):
        return "source_stage_batch_v1"

    def contract(self):
        return json.dumps(self.document).encode()

    def coefficients(self):
        return {key: self.arrays[key].tobytes() for key in self.document["local_tensors"]}

    def lookup(self, *_):
        raise AssertionError("scalar lookup forbidden in selected batch execution")

    def linear(self, *_):
        raise AssertionError("scalar linear forbidden in selected batch execution")

    def lookup_batch(self, tokens):
        assert type(tokens) is tuple and self.stage == 0
        assert len(tokens) == (self.document["input_rows"] if self.phase == 0 else 1)
        self.calls.append((self.phase, "lookup", len(tokens)))
        raw = self.arrays[self.document["table"]][list(tokens)].tobytes(order="C")
        if self.failure == "lookup_shape":
            return raw[:-4]
        if self.failure == "lookup_mutable":
            return bytearray(raw)
        return raw

    def linear_batch(self, stage, values):
        assert type(values) is bytes and stage == self.stage
        spec = self.document["matrices"][stage]
        final_stage = stage == len(self.document["matrices"]) - 1
        rows = 1 if final_stage or self.phase else self.document["input_rows"]
        assert len(values) == 4 * rows * spec["columns"]
        self.calls.append((self.phase, stage, rows))
        if self.failure == "cancel":
            raise asyncio.CancelledError("batch cancelled")
        inputs = np.frombuffer(values, dtype="<f4").reshape(rows, spec["columns"])
        weights = np.concatenate([self.arrays[member["weight"]] for member in spec["members"]])
        codes = np.rint(inputs.astype(np.float64) * 2048).astype(np.int64)
        weights = np.rint(weights.astype(np.float64) * 2048).astype(np.int16)
        result = np.empty((rows, spec["rows"]), dtype="<f4")
        for i, row in enumerate(codes):
            for j, weight in enumerate(weights):
                integer = sum(int(x) * int(w) for x, w in zip(row, weight, strict=True))
                result[i, j] = np.float32(integer) / np.float32(4194304)
        self.stage += 1
        if final_stage:
            self.stage = 0
            self.phase += 1
        if self.failure == "linear_shape":
            return result.tobytes() + b"\0\0\0\0"
        if self.failure == "nonfinite":
            result[0, 0] = np.inf
        return result.tobytes(order="C")

    def finish(self):
        assert self.phase == self.document["decode_steps"] + 1 and self.stage == 0
        if self.failure == "finish":
            raise ValueError("final verification rejected")
        self.finished = True
        return b"fake-seeded-acceptance" if self.seeded else b"fake-full-acceptance"

    def abort(self):
        self.aborted = True


@pytest.fixture
def track_owners(monkeypatch):
    """Observe borrowed array roots without extending backing lifetimes."""
    roots, requests, cache_arrays, peaks = [], [], [], []
    output = decoder._batch_output
    request_class = decoder._Request

    def tracked(raw, rows, width):
        array = output(raw, rows, width)
        root = array
        while isinstance(root.base, np.ndarray):
            root = root.base
        assert type(root.base) is bytes
        assert array.dtype == np.dtype("<f4") and array.flags.c_contiguous
        assert not array.flags.writeable
        roots.append(weakref.ref(root))
        peaks.append(sum(ref() is not None for ref in roots))
        return array

    class Request(request_class):
        def __init__(self, *args):
            super().__init__(*args)
            requests.append(weakref.ref(self))

        def phase(self, ids):
            result = super().phase(ids)
            assert all(ref() is None for ref in roots)
            return result

        def retire(self):
            cache_arrays.extend(array for cache in self.caches
                                for array in (cache.key, cache.value) if array is not None)
            super().retire()
            assert not self._coefficients
            assert all(cache.key is cache.value is None and cache.length == 0 for cache in self.caches)

    monkeypatch.setattr(decoder, "_batch_output", tracked)
    monkeypatch.setattr(decoder, "_Request", Request)

    def no_stack(*args, **kwargs):
        raise AssertionError("selected batch execution must not stack scalar rows")

    monkeypatch.setattr(decoder.np, "stack", no_stack)
    yield roots, requests, cache_arrays, peaks
    gc.collect()
    assert all(ref() is None for ref in roots + requests)
    assert all(not np.any(array) for array in cache_arrays)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("seeded", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_one_batch_per_source_stage_complete_state_and_audit(family, rows, seeded, tied, track_owners):
    cfg = config(family, tied)
    contract, arrays, _ = fixture(cfg, rows=rows, decode=2)
    tokens = (0, 4)[:rows]
    salt = b"b" * 32
    observer = SelectionCommitments(salt, max_selections=3)
    expected = clear_oracle(cfg, arrays, tokens, 2, selection_observer=observer)
    native = BatchNative(contract, arrays, seeded=seeded)
    with audit.response_state_audit(salt, runtime_kind="gkm") as records:
        result = decoder._execute_verified_window(native, contract, tokens)
        assert_complete_result(result, expected)
        assert records == []
    assert records == [observer.expected_record(contract)]
    assert result.acceptance == (b"fake-seeded-acceptance" if seeded else b"fake-full-acceptance")
    assert native.finished and native.aborted
    expected_calls = []
    for phase in range(3):
        batch = rows if phase == 0 else 1
        expected_calls.append((phase, "lookup", batch))
        expected_calls.extend((phase, stage, 1 if stage == len(contract["matrices"]) - 1 else batch)
                              for stage in range(len(contract["matrices"])))
    assert native.calls == expected_calls
    assert native.document == contract  # No source/numeric identity rewriting.
    assert max(track_owners[3]) <= 2  # Residual lookup + grouped reconstruction.


@pytest.mark.parametrize("failure,exception", [
    ("lookup_shape", ValueError), ("lookup_mutable", ValueError),
    ("linear_shape", ValueError), ("nonfinite", ValueError),
    ("finish", ValueError), ("cancel", asyncio.CancelledError),
])
def test_failure_never_publishes_and_retires_batch_owners(failure, exception, track_owners):
    contract, arrays, _ = fixture(config(), rows=2, decode=1)
    native = BatchNative(contract, arrays, failure=failure)
    with pytest.raises(exception):
        with audit.response_state_audit(b"c" * 32, runtime_kind="gkm") as records:
            decoder._execute_verified_window(native, contract, (0, 4))
    assert records == []
    assert native.aborted and not native.finished


def test_cancellation_between_phases_retires_without_finish(track_owners):
    contract, arrays, _ = fixture(config(), rows=2, decode=1)
    native = BatchNative(contract, arrays)
    with pytest.raises(RuntimeError, match="cancelled"):
        decoder._execute_verified_window(native, contract, (0, 4), cancelled=lambda: native.phase == 1)
    assert native.aborted and not native.finished and native.phase == 1


@pytest.mark.parametrize("mutation", ["lookup_rows", "member_offset", "member_width", "input_rows"])
def test_trusted_descriptor_shape_checks_precede_batch_boundary(mutation, monkeypatch):
    contract, arrays, _ = fixture(config(), rows=2, decode=0)
    if mutation == "lookup_rows":
        contract["prefill"]["steps"][0]["outputs"][0]["output_shape"][1] = 1
    elif mutation in ("member_offset", "member_width"):
        field = "stage_offset" if mutation == "member_offset" else "stage_width"
        contract["prefill"]["steps"][2]["outputs"][1][field] += 1
    else:
        original = decoder._Request._local

        def wrong_input(self, operation, *args):
            value = original(self, operation, *args)
            return value[:1] if operation["id"] == "layer.0.input_norm" else value

        monkeypatch.setattr(decoder._Request, "_local", wrong_input)
    native = BatchNative(contract, arrays)
    with pytest.raises(ValueError, match="GKM batch"):
        decoder._execute_verified_window(native, contract, (0, 4))
    assert not any(type(stage) is int for _, stage, _ in native.calls)
    assert native.aborted


def test_strided_input_and_dtype_coercion_serialize_once_in_c_order(monkeypatch, track_owners):
    cfg = config()
    contract, arrays, _ = fixture(cfg, rows=2, decode=1)
    expected = clear_oracle(cfg, arrays, (0, 4), 1)
    original = decoder._Request._local

    def strided(self, operation, *args):
        value = original(self, operation, *args)
        if operation["id"] == "layer.0.input_norm":
            value = np.repeat(value.astype(">f8"), 2, axis=-1)[..., ::2]
            assert not value.flags.c_contiguous
        return value

    monkeypatch.setattr(decoder._Request, "_local", strided)
    native = BatchNative(contract, arrays)
    assert_complete_result(decoder._execute_verified_window(native, contract, (0, 4)), expected)


@pytest.mark.parametrize("selector", [None, "wrong-mode"])
def test_only_native_selected_execution_may_use_batch(selector):
    contract, arrays, _ = fixture(config(), rows=1, decode=0)
    # Caller fields cannot replace the native owner's selected execution identity.
    contract["execution"] = "source_stage_batch_v1"
    native = BatchNative(contract, arrays)
    native.execution = lambda: selector
    error = AssertionError if selector is None else ValueError
    with pytest.raises(error):
        decoder._execute_verified_window(native, contract, (0,))
    assert native.calls == [] and native.aborted


def test_contract_mismatch_rejected_before_execution_selection():
    contract, arrays, _ = fixture(config(), rows=1, decode=0)
    native = BatchNative(contract, arrays)
    contract["numeric_profile"] = "forged"
    with pytest.raises(ValueError, match="compiled contract mismatch"):
        decoder._execute_verified_window(native, contract, (0,))
    assert not native.calls and native.aborted


def test_batch_claim_cursor_must_cover_exact_scalar_window(track_owners):
    contract, arrays, _ = fixture(config(), rows=2, decode=1)
    contract["window_steps"] -= 1
    native = BatchNative(contract, arrays)
    with pytest.raises(ValueError, match="scalar-claim cursor"):
        decoder._execute_verified_window(native, contract, (0, 4))
    assert not native.finished and native.aborted


def test_singleton_decode_and_phase_count_come_from_contract():
    contract, arrays, _ = fixture(config(), rows=2, decode=1)
    native = BatchNative(contract, arrays)
    request = decoder._Request(native, contract)
    try:
        request.phase((0, 4))
        before = list(native.calls)
        with pytest.raises(ValueError, match="admitted row count"):
            request.phase((0, 4))
        assert native.calls == before
        request.phase((0,))
        before = list(native.calls)
        with pytest.raises(ValueError, match="admitted row count"):
            request.phase((0,))
        assert native.calls == before
    finally:
        request.retire()
