"""Independent complete-state GKM conformance, generated and opt-in offline Qwen.

Legacy opt-in: PLLM_RUN_GKM_CHECKPOINT=1. Corrected Hello opt-in separately:
PLLM_RUN_GKM_CORRECTED_PROMPT=1. Use ordinary HF_HUB_CACHE to select existing
cache location. No downloads, wheel builds, or benchmark driver needed.
Use pytest --basetemp=target/gkm-conformance for temporary source/role files.
"""

import copy
import gc
import hashlib
import json
import os
import threading
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from gkm_decoder_oracle import (
    AdmittedTensors, ArrayTensors, CONTRACTION_COLUMNS, INPUT_BOUND, ROW_BLOCK, SelectionCommitments,
    assert_complete_result, clear_decoder, corrected_prompt_workspace_estimate,
    encode_activation_q11, encode_weight_q11, exact_contract,
)
from test_delegated_decoder import clear_oracle, config, fixture


PIN = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello world"
OUTPUTS = 2


def python_contract(inputs, weights):
    return np.array([
        [sum(int(x) * int(w) for x, w in zip(row, weight, strict=True))
         for weight in weights] for row in inputs
    ], dtype=np.int64)


def test_q11_nearest_even_original_float32_and_signed_i16_edges():
    scaled = np.array([
        -32768.5, -32768, -32767.5, -2.5, -1.5, -0.5, -0.0,
        0.0, 0.5, 1.5, 2.5, 32766.5, 32767,
    ], dtype=np.float32)
    values = scaled / np.float32(2048)
    expected = np.array([round(float(value) * 2048) for value in values], dtype=np.int16)
    np.testing.assert_array_equal(encode_weight_q11(values), expected)
    for bad in (32767.5 / 2048, -32769 / 2048, np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError):
            encode_weight_q11(np.array([bad], dtype=np.float32))
    with pytest.raises(ValueError, match="original float32"):
        encode_weight_q11(np.array([1.0], dtype=np.float64))


def test_activation_q11_full_domain_ties_and_original_float32_boundary():
    values = np.array([
        -2**20, 2**20, -16.5, 16.5, -2.5 / 2048, -1.5 / 2048,
        -0.5 / 2048, -0.0, 0.0, 0.5 / 2048, 1.5 / 2048, 2.5 / 2048,
    ], dtype=np.float32)
    expected = np.array([round(float(value) * 2048) for value in values], dtype=np.int64)
    actual = encode_activation_q11(values)
    assert actual.dtype == np.int64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:2], [-INPUT_BOUND, INPUT_BOUND])
    # Float32 source rounding matters: binary64 values just outside the bound
    # round back to the legal endpoint in binary32, but must not enter as f64.
    for sign in (-1, 1):
        outside64 = sign * (float(2**20) + 0.01)
        assert abs(round(outside64 * 2048)) > INPUT_BOUND
        source32 = np.float32(outside64)
        assert int(encode_activation_q11(np.array([source32]))[0]) == sign * INPUT_BOUND
        with pytest.raises(ValueError, match="original float32"):
            encode_activation_q11(np.array([outside64], dtype=np.float64))
        outside32 = np.nextafter(source32, np.float32(sign * np.inf))
        with pytest.raises(ValueError, match="activation bound"):
            encode_activation_q11(np.array([outside32]))
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite original float32"):
            encode_activation_q11(np.array([bad], dtype=np.float32))
    # Weight and activation contracts intentionally differ.
    assert int(encode_activation_q11(np.array([16.5], dtype=np.float32))[0]) == 33792
    with pytest.raises(ValueError, match="signed i16"):
        encode_weight_q11(np.array([16.5], dtype=np.float32))


@pytest.mark.parametrize("columns", [1, 7, 127, 128, 129, 255, 256, 257, 2048, 8960, 16384])
def test_certified_contractions_match_python_int_with_full_input_and_weight_extremes(columns):
    rng = np.random.default_rng(718)
    inputs = rng.integers(-INPUT_BOUND, INPUT_BOUND + 1, (3, columns), dtype=np.int64)
    weights = rng.integers(-32768, 32768, (5, columns), dtype=np.int16)
    inputs[0] = -INPUT_BOUND
    inputs[1] = INPUT_BOUND
    weights[0] = -32768
    weights[1] = np.resize(np.array([-32768, 32767], dtype=np.int16), columns)
    expected = python_contract(inputs, weights)
    np.testing.assert_array_equal(exact_contract(inputs, weights), expected)
    np.testing.assert_array_equal(exact_contract(inputs[:, ::-1], weights[:, ::-1]), expected)
    # Negative control: float32 contractions lose integer low bits.
    if columns > 1:
        assert np.any((inputs.astype(np.float32) @ weights.astype(np.float32).T) != expected)


def test_contraction_rejects_uncertified_geometry_and_non_integer_inputs():
    with pytest.raises(ValueError, match="rank-two int64 inputs/i16 weights"):
        exact_contract(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 2), dtype=np.int16))
    # Zero-stride views exercise geometry rejection without allocating big data.
    wide = np.broadcast_to(np.int16(0), (1, (1 << 23) + 1))
    with pytest.raises(ValueError, match="bound exceeded"):
        exact_contract(np.broadcast_to(np.int64(0), wide.shape), wide)
    for code in (-INPUT_BOUND - 1, INPUT_BOUND + 1, np.iinfo(np.int64).min):
        with pytest.raises(ValueError, match="activation bound"):
            exact_contract(np.array([[code]], dtype=np.int64), np.ones((1, 1), dtype=np.int16))


def test_complete_integer_sum_precedes_float32_conversion_across_chunks():
    inputs = np.zeros((1, 257), dtype=np.int64)
    inputs[0, [0, 128, 256]] = [2**24, 1, -(2**24)]
    weights = np.ones((1, 257), dtype=np.int16)
    expected = python_contract(inputs, weights)
    assert expected.item() == 1
    np.testing.assert_array_equal(exact_contract(inputs, weights), expected)
    # Converting exact chunks to f32 before completing the integer contraction
    # loses the middle +1. Complete integer accumulation must retain it.
    wrong = np.zeros((1, 1), dtype=np.float32)
    for start in range(0, 257, CONTRACTION_COLUMNS):
        wrong += python_contract(inputs[:, start:start + CONTRACTION_COLUMNS],
                                 weights[:, start:start + CONTRACTION_COLUMNS]).astype(np.float32)
    assert wrong.item() == 0
    actual = exact_contract(inputs, weights).astype(np.float32) / np.float32(4194304)
    assert actual.item() == 1 / 4194304


def test_chunking_preserves_integer_low_bit_beyond_binary64_exact_sum_boundary():
    inputs = np.full((1, 130), -INPUT_BOUND, dtype=np.int64)
    weights = np.full((1, 130), -32768, dtype=np.int16)
    inputs[0, -1], weights[0, -1] = 1, 1
    expected = python_contract(inputs, weights)
    assert expected.item() == 129 * (1 << 46) + 1
    naive_float64 = inputs.astype(np.float64) @ weights.astype(np.float64).T
    assert int(naive_float64.item()) != expected.item()
    np.testing.assert_array_equal(exact_contract(inputs, weights), expected)
    # Cancellation after a large prefix must also retain a low integer bit.
    inputs = np.concatenate((inputs, -inputs[:, :-1]), axis=1)
    weights = np.concatenate((weights, weights[:, :-1]), axis=1)
    assert python_contract(inputs, weights).item() == 1
    assert exact_contract(inputs, weights).item() == 1


def test_full_input_cancellation_and_checked_int64_overflow_endpoints():
    # Every 128-column f64 chunk reaches the proof's 2**53 absolute-sum bound.
    columns = 1 << 17
    inputs = np.full((1, columns), INPUT_BOUND, dtype=np.int64)
    weights = np.full((1, columns), -32768, dtype=np.int16)
    assert exact_contract(inputs, weights).item() == -(1 << 63)
    with pytest.raises(OverflowError, match="int64 accumulation overflow"):
        exact_contract(-inputs, weights)  # +2**63 cannot be represented.
    with pytest.raises(OverflowError, match="int64 accumulation overflow"):
        exact_contract(np.full((1, columns + 1), INPUT_BOUND, dtype=np.int64),
                       np.full((1, columns + 1), -32768, dtype=np.int16))
    inputs[:, 1::2] *= -1
    assert exact_contract(inputs, weights).item() == 0


def test_generated_decoder_accepts_activation_codes_beyond_i16_with_original_weights():
    from pllm.runtime import delegated_decoder as decoder

    cfg = config()
    _, arrays, _ = fixture(cfg)
    # Final norm is a local float32 coefficient, not a delegated i16 weight.
    # Force head input beyond i16 while keeping all source matrices unchanged.
    arrays["model.norm.weight"][:] = np.float32(64)
    source = {name: value.tobytes() for name, value in arrays.items()}
    expected = clear_oracle(cfg, arrays, (0, 4, 9), 2)
    assert clear_decoder(cfg, arrays, (0, 4, 9), 2, row_block=3) == expected
    for seeded in (False, True):
        result = decoder.execute_gkm_reference(
            cfg, source, (0, 4, 9), decode_steps=2, seeded=seeded)
        assert_complete_result(result, expected)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("forced", [False, True])
def test_generated_complete_tensors_match_python_integer_and_native_oracles(family, tied, forced):
    from pllm.runtime import delegated_decoder as decoder

    cfg = config(family, tied)
    _, arrays, source = fixture(cfg)
    continuation = (6, 2) if forced else None
    expected = clear_oracle(cfg, arrays, (0, 4, 9), 2, decode_tokens=continuation)
    assert clear_decoder(cfg, arrays, (0, 4, 9), 2, decode_tokens=continuation,
                         row_block=3) == expected
    for seeded in (False, True):
        if forced:
            # Public reference is greedy-only; the existing quality seam admits
            # fixed continuation without changing any production API.
            contract = json.loads(decoder.decoder_contract(cfg, input_rows=3, decode_steps=2))
            native = decoder._native._GkmDecoderSession(
                json.dumps(cfg).encode(), 3, 2, source, seeded=seeded)
            result = decoder._execute_verified_window(
                native, contract, (0, 4, 9), decode_tokens=continuation)
        else:
            result = decoder.execute_gkm_reference(
                cfg, source, (0, 4, 9), decode_steps=2, seeded=seeded)
        assert_complete_result(result, expected)
        assert result.executed == (0, 4, 9) + (continuation or result.selected[:-1])
        assert len(result.selected) == 3 and result.position == 5


def test_reference_uses_original_lookup_and_bounded_complete_head_rows():
    cfg = config()
    _, arrays, _ = fixture(cfg)
    reads = []

    class Bounded(ArrayTensors):
        def __getitem__(self, name):
            assert len(self.shapes[name]) == 1, "complete matrix materialized"
            return super().__getitem__(name)

        def read_rows(self, name, begin, end):
            assert end - begin <= 3
            reads.append((name, begin, end))
            return super().read_rows(name, begin, end)

    tensors = Bounded(arrays)
    lookup = tensors.read_rows("model.embed_tokens.weight", 0, 1)
    assert lookup.tobytes() == arrays["model.embed_tokens.weight"][:1].tobytes()
    assert np.signbit(lookup[0, 0])
    assert np.any(lookup != encode_weight_q11(lookup).astype(np.float32) / np.float32(2048))
    assert clear_decoder(cfg, tensors, (0, 4, 9), 2, row_block=3) == clear_oracle(
        cfg, arrays, (0, 4, 9), 2)
    table_rows = {row for name, begin, end in reads if name == "model.embed_tokens.weight"
                  for row in range(begin, end)}
    assert table_rows == set(range(cfg["vocab_size"]))


def test_admitted_reader_exact_values_bounded_frames_and_changed_source(tmp_path, monkeypatch):
    from pllm import Model
    from pllm.runtime import delegated_source
    from test_delegated_source import write_source_fixture

    cfg, _, arrays, _ = write_source_fixture(tmp_path / "source")
    source = delegated_source.admit_source(Model.path(str(tmp_path / "source")),
                                           input_rows=3, decode_steps=2)
    reads, checks = [], []
    original = delegated_source._source_float_frame

    def frame(path, identity, offset, count, dtype):
        assert count * 4 <= 64 << 10
        reads.append((offset, count))
        return original(path, identity, offset, count, dtype)

    monkeypatch.setattr(delegated_source, "_source_float_frame", frame)
    tensors = AdmittedTensors(source, check=lambda: checks.append(True))
    assert not reads  # Construction and lock verification do not decode values.
    tensors.verify_lock()
    assert not reads
    assert clear_decoder(cfg, tensors, (0, 4, 9), 2, row_block=3) == clear_oracle(
        cfg, arrays, (0, 4, 9), 2)
    assert reads and checks
    tensors.verify_lock()
    with pytest.raises(ValueError, match="bounded row reads"):
        tensors["model.embed_tokens.weight"]
    with pytest.raises(ValueError, match="row block"):
        tensors.read_rows("model.embed_tokens.weight", 0, ROW_BLOCK + 1)
    path = source.path / "model.safetensors"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="changed"):
        tensors.verify_lock()
    with pytest.raises(ValueError, match="changed"):
        tensors.read_rows("model.embed_tokens.weight", 0, 1)


@pytest.mark.parametrize("field", ["logits", "kv", "selected", "executed", "position"])
def test_complete_result_comparison_rejects_every_corrupted_state_owner(field):
    cfg = config()
    _, arrays, _ = fixture(cfg)
    expected = clear_decoder(cfg, arrays, (0, 4, 9), 2)
    result = SimpleNamespace(**dict(zip(
        ("logits", "kv", "selected", "executed", "position"), expected, strict=True)))
    assert_complete_result(result, expected)
    if field == "logits":
        first = bytearray(result.logits[0])
        first[0] ^= 1  # A low bit, not merely greedy text/ID disagreement.
        result.logits = (bytes(first),) + result.logits[1:]
    elif field == "kv":
        shape, key, value = result.kv[0]
        result.kv = ((shape, key, bytes([value[0] ^ 1]) + value[1:]),) + result.kv[1:]
    elif field == "selected":
        result.selected = result.selected[:-1] + ((result.selected[-1] + 1) % cfg["vocab_size"],)
    elif field == "executed":
        result.executed = result.executed[:-1]
    else:
        result.position += 1
    with pytest.raises(AssertionError):
        assert_complete_result(result, expected)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_single_reference_observer_matches_every_selection_and_releases_tensors(family, tied):
    from test_delegated_state_audit import commit, state_hash

    cfg = config(family, tied)
    contract, arrays, _ = fixture(cfg)
    salt = os.urandom(32)
    observer = SelectionCommitments(salt, max_selections=3)
    references = []

    def observe(logits, caches, token, executed, position):
        references.extend(weakref.ref(array) for cache in caches for array in cache)
        observer(logits, caches, token, executed, position)

    final = clear_decoder(cfg, arrays, (0, 4, 9), 2, selection_observer=observe)
    integer_observer = SelectionCommitments(salt, max_selections=3)
    assert clear_oracle(cfg, arrays, (0, 4, 9), 2,
                        selection_observer=integer_observer) == final
    assert observer.expected_record(contract) == integer_observer.expected_record(contract)
    for step, row in enumerate(observer.selections):
        logits, kv, selected, executed, position = clear_oracle(cfg, arrays, (0, 4, 9), step)
        assert row == {
            "logits_and_active_kv": state_hash(logits[-1], kv, salt),
            "position": position, "executed_token_count": position,
            "cache_lengths": [position] * cfg["num_hidden_layers"],
            "executed_history": commit(executed, salt),
            "selected_token": commit(selected[-1], salt),
            "pending_selected_token": commit(selected[-1], salt),
            "checked_values": (len(logits[-1]) + sum(len(k) + len(v) for _, k, v in kv)) // 4,
        }
    gc.collect()
    assert all(reference() is None for reference in references)
    # Salt protects private values; a fresh cohort cannot reuse commitments.
    other = SelectionCommitments(os.urandom(32), max_selections=3)
    clear_decoder(cfg, arrays, (0, 4, 9), 2, selection_observer=other)
    assert other.selections[0]["executed_history"] != observer.selections[0]["executed_history"]
    with pytest.raises(ValueError, match="selection bound"):
        clear_decoder(cfg, arrays, (0,), 0, selection_observer=observer)


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
def test_inprocess_audit_matches_single_reference_observer(family):
    from pllm.runtime import benchmark_response_audit as audit, delegated_decoder as decoder

    cfg = config(family)
    contract, arrays, source = fixture(cfg)
    salt = os.urandom(32)
    observer = SelectionCommitments(salt, max_selections=3)
    expected = clear_decoder(cfg, arrays, (0, 4, 9), 2, selection_observer=observer)
    for seeded in (False, True):
        with audit.response_state_audit(salt, runtime_kind="gkm") as records:
            result = decoder.execute_gkm_reference(cfg, source, (0, 4, 9), decode_steps=2, seeded=seeded)
            assert_complete_result(result, expected)
            assert records == []
        assert records == [observer.expected_record(contract)]


@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
def test_generated_sdk_accepted_result_matches_bounded_checkpoint_reader(tmp_path, sharing, execution,
                                                                       record_property):
    from pllm import OpenAI
    from pllm.runtime import benchmark_response_audit as audit
    from pllm.runtime.responses import normalize_input, prompt_text
    from test_delegated_sdk import active_sessions, experiment_fixture

    experiment, cfg, _ = experiment_fixture(tmp_path, family="qwen3", tied=False, sharing=sharing,
                                           execution=execution)
    salt = os.urandom(32)
    observer = SelectionCommitments(salt, max_selections=OUTPUTS)
    captured = []
    with audit.response_state_audit(salt, runtime_kind="gkm") as records:
        with OpenAI(experiment=experiment) as client:
            topology = client._owned_topology
            source = client._core._ensure_source()
            tensors = AdmittedTensors(source)
            tensors.verify_lock()
            ids = tuple(source.tokenizer().encode(prompt_text(normalize_input(PROMPT)), add_bos=False))
            expected = clear_decoder(cfg, tensors, ids, OUTPUTS - 1, row_block=3,
                                     selection_observer=observer)
            client._core._result_hook = captured.append
            response = client.responses.create(input=PROMPT, max_output_tokens=OUTPUTS)
            result, = captured
            assert_complete_result(result, expected)
            acceptance = json.loads(result.acceptance)
            runtime = source.runtime_contract(experiment, len(ids), OUTPUTS)
            assert acceptance["contract"] == hashlib.sha256(
                b"pllm.gkm.sdk-contract.v1\0" + runtime).hexdigest()
            assert acceptance["sharing"] == sharing
            assert response.usage.input_tokens == len(ids)
            assert response.usage.output_tokens == OUTPUTS
            assert active_sessions(client._core) == [0, 0]
            tensors.verify_lock()
            assert records == []
        assert all(not status.running for status in topology.statuses)
        assert records == []
    assert records == [observer.expected_record(source.contract)]
    record_property("sharing", sharing)
    record_property("execution", execution or "scalar")
    record_property("input_tokens", len(ids))
    record_property("output_tokens", OUTPUTS)
    record_property("checked_selections", len(observer.selections))
    record_property("checked_selection_float32_values", sum(row["checked_values"] for row in observer.selections))


@pytest.mark.parametrize("sharing", ["full", "seeded"])
def test_generated_template_bos_batch_request_lengths(tmp_path, monkeypatch, sharing):
    from pllm import OpenAI
    from pllm.runtime import delegated_client, delegated_transport as transport, responses
    from pllm.runtime.transformer_client import TokenizersJSONTokenizer
    from test_delegated_sdk import active_sessions, experiment_fixture

    experiment, cfg, arrays = experiment_fixture(
        tmp_path, capacity="decoder", sharing=sharing, execution="source_stage_batch_v1",
        rows=8, requests=2)
    tokenizer_config = tmp_path / "source" / "tokenizer_config.json"
    options = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    options.update(add_bos_token=True, chat_template="world {{ messages[0].content }} world")
    tokenizer_config.write_text(json.dumps(options), encoding="utf-8")
    assert (experiment.budget.max_input_tokens, experiment.budget.max_new_tokens) == (8, 3)
    assert 8 + 3 - 1 <= cfg["max_position_embeddings"] == 32

    # Independent WordLevel/Whitespace IDs: BOS, template prefix, user, suffix.
    # Neither candidate rendering nor tokenization supplies these oracle inputs.
    literal_ids = ((0, 4, 3, 4), (0, 4, 3, 4, 4))
    expected = tuple(clear_decoder(cfg, arrays, ids, decode_steps=1, row_block=3)
                     for ids in literal_ids)
    assert [state[4] for state in expected] == [5, 6]
    assert all(len(state[2]) == 2 for state in expected)
    encoded, consumed, closed, batches, checked = [], [], [], [], []
    original_encode = TokenizersJSONTokenizer.encode
    original_consume = responses._PreparedPromptUse.consume
    original_close = responses._PreparedPromptUse.close
    original_lookup = transport._NativeExchange.lookup_batch
    original_linear = transport._NativeExchange.linear_batch
    original_validate = delegated_client.DelegatedClient._validate_result

    def encode(self, text, *, add_bos=False):
        ids = original_encode(self, text, add_bos=add_bos)
        encoded.append((text, add_bos, tuple(ids)))
        return ids

    def consume(self, body, source):
        tokenizer, ids = original_consume(self, body, source)
        assert type(ids) is tuple and all(type(token) is int for token in ids)
        assert ids == literal_ids[len(checked)]
        consumed.append(ids)
        return tokenizer, ids

    def close(self):
        had_payload = self._payload is not None
        original_close(self)
        assert self._payload is None and self._tokenizer is None and self._closed
        if had_payload:
            closed.append(True)

    def descriptor_snapshot(descriptor):
        return tuple(descriptor[name] for name in (
            "batch_index", "phase_index", "operation_index", "first_claim", "kind", "stage",
            "rows", "columns", "output_width")) + (
                tuple(descriptor["request_bytes"]), tuple(descriptor["reply_bytes"]))

    def observe(self, descriptor, result, input_bytes):
        rows = len(literal_ids[len(checked)]) if descriptor["phase_index"] == 0 else 1
        if descriptor["kind"] == "linear" and descriptor["stage"] == self.head_stage:
            rows = 1  # Head selects only the last prefill position.
        assert descriptor["rows"] == rows
        assert input_bytes == 4 * rows * descriptor["columns"]
        assert type(result) is bytes and len(result) == 4 * rows * descriptor["output_width"]
        assert all(type(size) is int and size > 0
                   for name in ("request_bytes", "reply_bytes") for size in descriptor[name])
        batches.append(descriptor_snapshot(descriptor))

    def lookup_batch(self, tokens):
        descriptor = self.peers.batch_descriptor("lookup")
        assert len(tokens) == descriptor["rows"]
        if descriptor["phase_index"] == 0:
            assert tokens == literal_ids[len(checked)]
        result = original_lookup(self, tokens)
        observe(self, descriptor, result, 0)
        return result

    def linear_batch(self, stage, values):
        descriptor = self.peers.batch_descriptor("linear", stage)
        input_bytes = len(values)
        # Transfer the argument without keeping an extra payload reference while
        # the original exchange retires its input before native acceptance.
        pending = [values]
        values = None
        result = original_linear(self, stage, pending.pop())
        observe(self, descriptor, result, input_bytes)
        return result

    monkeypatch.setattr(TokenizersJSONTokenizer, "encode", encode)
    monkeypatch.setattr(responses._PreparedPromptUse, "consume", consume)
    monkeypatch.setattr(responses._PreparedPromptUse, "close", close)
    monkeypatch.setattr(transport._NativeExchange, "lookup_batch", lookup_batch)
    monkeypatch.setattr(transport._NativeExchange, "linear_batch", linear_batch)
    with OpenAI(experiment=experiment) as client:
        # Direct OpenAI is eager: use its real core and already-owned topology.
        core, topology = client._core, client._owned_topology

        def check_result(core, result, source, ids, count):
            original_validate(core, result, source, ids, count)
            index = len(checked)
            assert_complete_result(result, expected[index])
            assert result.executed == literal_ids[index] + result.selected[:-1]
            assert result.position == (5, 6)[index] and len(result.selected) == 2
            # Reuse the existing SDK acceptance-contract check; no copied binder.
            runtime = core._ensure_source().runtime_contract(experiment, len(literal_ids[index]), 2)
            acceptance = json.loads(result.acceptance)
            assert acceptance["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
            assert acceptance["sharing"] == sharing
            assert acceptance["contract"] == hashlib.sha256(
                b"pllm.gkm.sdk-contract.v1\0" + runtime).hexdigest()
            descriptors = json.loads(runtime)["batch_execution"]["descriptors"]
            assert batches == [descriptor_snapshot(item) for item in descriptors]
            assert len(descriptors) == 2 * (7 * cfg["num_hidden_layers"] + 2)
            assert sum(item["rows"] for item in descriptors) == (
                (7 * cfg["num_hidden_layers"] + 1) * (len(literal_ids[index]) + 1) + 2)
            checked.append(True)  # Never retain accepted state or native owners.

        monkeypatch.setattr(delegated_client.DelegatedClient, "_validate_result", check_result)
        for index, prompt in enumerate(("hello", "hello world")):
            response = client.responses.create(input=prompt, max_output_tokens=2)
            assert response.usage.input_tokens == (4, 5)[index]
            assert response.usage.output_tokens == 2
            assert checked == closed == [True] * (index + 1)
            assert consumed == list(literal_ids[:index + 1])
            assert encoded == [(text, True, ids) for text, ids in zip(
                ("world hello world", "world hello world world")[:index + 1],
                literal_ids[:index + 1], strict=True)]
            assert core._active is None and active_sessions(core) == [0, 0]
            batches.clear()
    assert all(not status.running for status in topology.statuses)


def _generated_template_bos_batch_failed_retirement(
        tmp_path, monkeypatch, failure, prompt, literal_ids, position):
    from pllm import OpenAI
    from pllm.deployment import resources
    from pllm.runtime import delegated_transport as transport, responses, tokenizer_admission
    from pllm.runtime.client import ProtocolError
    from pllm.runtime.transformer_client import TokenizersJSONTokenizer
    from test_delegated_sdk import active_sessions, experiment_fixture

    experiment, cfg, arrays = experiment_fixture(
        tmp_path, capacity="decoder", sharing="full", execution="source_stage_batch_v1",
        rows=8, requests=2)
    tokenizer_config = tmp_path / "source" / "tokenizer_config.json"
    options = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    options.update(add_bos_token=True, chat_template="world {{ messages[0].content }} world")
    tokenizer_config.write_text(json.dumps(options), encoding="utf-8")
    assert (experiment.budget.max_input_tokens, experiment.budget.max_new_tokens) == (8, 3)
    assert 8 + 3 - 1 <= cfg["max_position_embeddings"] == 32
    expected = clear_decoder(cfg, arrays, literal_ids, decode_steps=1, row_block=3)
    assert expected[4] == position and len(expected[2]) == 2
    owners, reservations, encoded, batches, checked, failures = [], [], [], [], [], []
    failing = True
    original_reserve = resources.reserve_incremental_memory
    original_exit = resources._MemoryReservation.__exit__
    original_tokenizer_close = tokenizer_admission._PromptTokenizer.close
    original_consume = responses._PreparedPromptUse.consume
    original_encode = TokenizersJSONTokenizer.encode
    original_lookup = transport._NativeExchange.lookup_batch
    original_linear = transport._NativeExchange.linear_batch

    def reserve(amount):
        reservation = original_reserve(amount)
        if amount == tokenizer_admission.DECODER_METADATA_MEMORY_BYTES and not reservations:
            reservations.append(reservation)
        return reservation

    def consume(self, body, source):
        tokenizer, ids = original_consume(self, body, source)
        assert type(ids) is tuple and ids == literal_ids
        assert type(tokenizer) is tokenizer_admission._PromptTokenizer
        assert self._metadata_owner is not None
        # Capture real request custody, not a constructed prepared owner.
        owners.append((self, tokenizer, self._payload, self._metadata_owner))
        return tokenizer, ids

    def encode(self, text, *, add_bos=False):
        ids = original_encode(self, text, add_bos=add_bos)
        encoded.append((text, add_bos, tuple(ids)))
        return ids

    def close_tokenizer(self):
        if failing and failure == "tokenizer" and owners and self is owners[0][1]:
            failures.append("tokenizer")
            raise RuntimeError("generated batch tokenizer retirement failure")
        return original_tokenizer_close(self)

    def exit_metadata(self, *exc):
        if failing and failure == "metadata" and reservations and self is reservations[0]:
            failures.append("metadata")
            raise RuntimeError("generated batch metadata retirement failure")
        return original_exit(self, *exc)

    def observe(self, descriptor):
        rows = len(literal_ids) if descriptor["phase_index"] == 0 else 1
        if descriptor["kind"] == "linear" and descriptor["stage"] == self.head_stage:
            rows = 1
        assert descriptor["rows"] == rows
        batches.append(dict(descriptor))

    def lookup_batch(self, tokens):
        descriptor = self.peers.batch_descriptor("lookup")
        if descriptor["phase_index"] == 0:
            assert tokens == literal_ids
        result = original_lookup(self, tokens)
        observe(self, descriptor)
        return result

    def linear_batch(self, stage, values):
        descriptor = self.peers.batch_descriptor("linear", stage)
        pending = [values]
        values = None
        result = original_linear(self, stage, pending.pop())
        observe(self, descriptor)
        return result

    with OpenAI(experiment=experiment) as client:
        topology, core = client._owned_topology, client._core
        # Startup has its own metadata reservations; inject only after it ends.
        monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
        monkeypatch.setattr(resources._MemoryReservation, "__exit__", exit_metadata)
        monkeypatch.setattr(tokenizer_admission._PromptTokenizer, "close", close_tokenizer)
        monkeypatch.setattr(responses._PreparedPromptUse, "consume", consume)
        monkeypatch.setattr(TokenizersJSONTokenizer, "encode", encode)
        monkeypatch.setattr(transport._NativeExchange, "lookup_batch", lookup_batch)
        monkeypatch.setattr(transport._NativeExchange, "linear_batch", linear_batch)

        def check_result(result):
            assert_complete_result(result, expected)
            assert result.executed == literal_ids + result.selected[:-1]
            assert result.position == position and len(result.selected) == 2
            runtime = core._ensure_source().runtime_contract(experiment, len(literal_ids), 2)
            acceptance = json.loads(result.acceptance)
            assert acceptance["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
            assert acceptance["sharing"] == "full"
            assert acceptance["contract"] == hashlib.sha256(
                b"pllm.gkm.sdk-contract.v1\0" + runtime).hexdigest()
            descriptors = json.loads(runtime)["batch_execution"]["descriptors"]
            assert batches == descriptors
            assert len(descriptors) == 2 * (7 * cfg["num_hidden_layers"] + 2)
            assert sum(item["rows"] for item in descriptors) == (
                (7 * cfg["num_hidden_layers"] + 1) * (len(literal_ids) + 1) + 2)
            checked.append(True)

        core._result_hook = check_result
        body = {"input": prompt, "max_output_tokens": 2, "stream": True}
        attempt = core.events(body)
        try:
            # _produce runs the complete native batch before its first event.
            # Explicit close isolates retirement from __next__'s automatic retry.
            event = next(attempt)
            assert event.type not in {"response.completed", "response.incomplete"}
            assert checked == [True] and failures == []
            assert encoded == [(f"world {prompt} world", True, literal_ids)]
            assert active_sessions(core) == [0, 0]
            prepared, tokenizer, payload, metadata = owners[0]
            reservation, = reservations
            with resources._capacity.lock:
                charged = resources._capacity._owners[reservation._token]
            assert charged == tokenizer_admission.DECODER_METADATA_MEMORY_BYTES
            assert attempt._prepared_prompt is prepared and core._active is attempt

            for retry in range(2):
                with pytest.raises(RuntimeError, match=f"generated batch {failure} retirement failure"):
                    attempt.close()
                assert failures == [failure] * (retry + 1)
                assert not attempt.closed and core._active is attempt
                assert attempt._prepared_prompt is prepared and not prepared._closed
                assert prepared._metadata_owner is metadata
                if failure == "tokenizer":
                    assert prepared._tokenizer is tokenizer and prepared._payload is payload
                    assert tokenizer._inner is not None
                else:
                    assert prepared._tokenizer is None and prepared._payload is None
                    assert tokenizer._inner is None
                with resources._capacity.lock:
                    assert resources._capacity._owners[reservation._token] == charged
                assert not reservation._exited  # Failed retirement is not a refund.
                with pytest.raises(ProtocolError, match="exclusive request"):
                    core.events(body)
                assert checked == [True] and active_sessions(core) == [0, 0]

            failing = False
            attempt.close()
            assert attempt.closed and core._active is None
            assert attempt._prepared_prompt is None and prepared._closed
            assert prepared._tokenizer is prepared._payload is prepared._metadata_owner is None
            assert reservation._exited
            with resources._capacity.lock:
                assert reservation._token not in resources._capacity._owners
            # A replacement seat is allowed only after actual cleanup/refund.
            replacement = core.events(body)
            assert core._active is replacement
            replacement.close()
            assert core._active is None and active_sessions(core) == [0, 0]
        finally:
            failing = False
            attempt.close()
            # Keep a failing regression from leaking real admission into other
            # tests. This is teardown, never evidence of production retirement.
            if reservations:
                original_exit(reservations[0], None, None, None)
    assert all(not status.running for status in topology.statuses)


@pytest.mark.parametrize("prompt,literal_ids,position", [
    ("hello", (0, 4, 3, 4), 5), ("hello world", (0, 4, 3, 4, 4), 6),
])
def test_generated_template_bos_batch_tokenizer_close_failure_retains_attempt(
        tmp_path, monkeypatch, prompt, literal_ids, position):
    _generated_template_bos_batch_failed_retirement(
        tmp_path, monkeypatch, "tokenizer", prompt, literal_ids, position)


@pytest.mark.parametrize("prompt,literal_ids,position", [
    ("hello", (0, 4, 3, 4), 5), ("hello world", (0, 4, 3, 4, 4), 6),
])
def test_generated_template_bos_batch_metadata_close_failure_retains_attempt(
        tmp_path, monkeypatch, prompt, literal_ids, position):
    # Required contract: failed metadata callback remains retryable and charged
    # under the SAME attempt until successful close. Current ExitStack pops the
    # callback on failure, so second close can falsely retire without refund.
    # Freeze that regression; production repair needs separate approval.
    _generated_template_bos_batch_failed_retirement(
        tmp_path, monkeypatch, "metadata", prompt, literal_ids, position)


@pytest.mark.slow
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
@pytest.mark.skipif(os.getenv("PLLM_RUN_GKM_CHECKPOINT") != "1",
                    reason="opt-in pinned offline GKM historical eight-row refusal")
def test_pinned_qwen_checkpoint_source_template_oversize_refusal(tmp_path, monkeypatch,
                                                              record_property, sharing, execution):
    from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
    from pllm.profiles import OpenWeightVerifiableCpu
    from pllm.protocols import OpenWeightVerifiableOutsourcing
    from pllm.deployment.network import NetworkError
    from pllm.deployment import resources
    from pllm.runtime import benchmark_memory as memory, benchmark_response_audit as audit
    from pllm.runtime import delegated_client, delegated_source, delegated_transport, responses
    from pllm.runtime.safetensors_store import SafeTensorStore

    # Select the existing standard shared cache before pytest's per-test XDG
    # isolation. The optional offline checkpoint must not use an empty scratch cache.
    monkeypatch.setenv("HF_HUB_CACHE", HF_HUB_CACHE)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    snapshot = Path(HF_HUB_CACHE) / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / PIN
    if not snapshot.is_dir():
        pytest.skip(f"missing pinned offline checkpoint cache: {snapshot}; no download/fallback")
    model = Model.hf(MODEL, revision=PIN, local_files_only=True)
    experiment = Experiment(
        "gkm-checkpoint-conformance", OpenWeightVerifiableCpu(model,
            linear=OpenWeightVerifiableOutsourcing(capacity="decoder", sharing=sharing,
                                                   execution=execution)),
        Deployment.local(root=str(tmp_path / "roles")),
        ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=OUTPUTS),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint values loaded before memory admission")

    # Metadata/hash reads allowed; no tensor value conversion/import until both
    # ordinary role estimate and extra oracle workspace pass shared admission.
    with monkeypatch.context() as early:
        early.setattr(delegated_source, "_source_float_frame", forbidden)
        early.setattr(delegated_source.AdmittedSource, "load_transport", forbidden)
        early.setattr(SafeTensorStore, "get", forbidden)
        early.setattr(delegated_source._AdmittedTensorStore, "get", forbidden)
        report = memory.benchmark_memory(experiment, backend="native")
        estimate = copy.deepcopy(report["estimate"])
        # Preserve the historical admission allowance unchanged. This refusal
        # case no longer constructs tensor-oracle or accepted-state arrays.
        extra = 512 << 20
        estimate["client_peak_bytes"] += extra
        estimate["native_total_peak_bytes"] += extra
        estimate["components"]["independent_oracle_allowance_bytes"] = extra
        report = memory.admit_memory(estimate, memory.host_memory(), backend="native")
        memory.require_admission(report)

    stopped = threading.Event()
    clients = []

    def abort(reason):
        stopped.set()
        for client in clients:
            lease = client.responses._lease
            active = None if lease._client is None else lease._client._core._active
            if active is not None:
                active.cancel()

    guard = memory.MemoryWatchdog(report, abort)

    def check():
        if stopped.is_set():
            raise memory.BenchmarkMemoryError(guard.error or "checkpoint test cancelled")
        guard.check()

    captured, owners, reached = [], [], []
    salt = os.urandom(32)
    guard.start()
    try:
        source = delegated_source.admit_source(experiment)
        assert source.model == model and source.source_lock["commit"] == PIN
        assert source.source_lock_digest == estimate["gkm_admission"]["source_lock_digest"]
        assert source.checkpoint_digest == estimate["gkm_admission"]["checkpoint_digest"]
        # Authored opt-in oracle only: actual Hello world count is not yet
        # observed. Never substitute Hello's corrected 30-token count here.
        ids = _independent_pinned_qwen_prompt(source, PROMPT)
        assert len(ids) > experiment.budget.max_input_tokens == 8
        original_complete = responses._PreparedPromptUse._complete_payload

        def complete(owner, payload):
            result = original_complete(owner, payload)
            # Observe the actual SDK request owner; never inject oracle IDs.
            assert payload.ids == ids
            owners.append((owner, payload.tokenizer, owner._metadata_owner))
            return result

        def trap(name):
            def fail(*args, **kwargs):
                reached.append(name)
                raise AssertionError(f"historical oversized prompt reached {name}")
            return fail

        monkeypatch.setattr(responses._PreparedPromptUse, "_complete_payload", complete)
        monkeypatch.setattr(delegated_source, "_source_float_frame", trap("_source_float_frame"))
        monkeypatch.setattr(delegated_source.AdmittedSource, "load_values", trap("load_values"))
        monkeypatch.setattr(delegated_source.AdmittedSource, "load_transport", trap("load_transport"))
        monkeypatch.setattr(delegated_source._native, "_GkmTransportImport", trap("_GkmTransportImport"))
        monkeypatch.setattr(SafeTensorStore, "get", trap("SafeTensorStore.get"))
        monkeypatch.setattr(delegated_source._AdmittedTensorStore, "get", trap("_AdmittedTensorStore.get"))
        monkeypatch.setattr(delegated_client, "_execute_request", trap("client._execute_request"))
        monkeypatch.setattr(delegated_transport, "execute_request", trap("transport.execute_request"))
        monkeypatch.setattr(delegated_transport._Peers, "start", trap("_Peers.start"))

        def reject_result(*args):
            captured.append(True)
            raise AssertionError("historical oversized prompt reached accepted result validation")

        monkeypatch.setattr(delegated_client.DelegatedClient, "_validate_result", reject_result)
        with (audit.response_state_audit(salt, runtime_kind="gkm") as records,
              OpenAI(experiment=experiment, timeout=3600) as client):
            clients.append(client)
            lease = client.responses._lease
            assert lease._state == "pending" and lease._client is None
            check()
            with pytest.raises(NetworkError, match="request input exceeds selected budget"):
                client.responses.create(input=PROMPT, max_output_tokens=OUTPUTS)
            (owner, tokenizer, metadata), = owners
            assert owner._closed and owner._payload is owner._tokenizer is owner._metadata_owner is None
            assert owner._failure is None and tokenizer._inner is None
            assert metadata is not None and metadata._exited
            with resources._capacity.lock:
                assert metadata._token not in resources._capacity._owners
            assert lease._request_owner is None and not lease._request_busy
            assert lease._state == "closed" and lease._requests == 0
            assert lease._client is None and client._owned_topology is None
            assert reached == captured == records == []
            check()
        assert reached == captured == records == []
        record_property("sharing", sharing)
        record_property("execution", execution or "scalar")
        record_property("source_lock_digest", source.source_lock_digest)
        record_property("checkpoint_digest", source.checkpoint_digest)
        record_property("tokenizer_digest", source.tokenizer_digest)
        record_property("model_revision", PIN)
        record_property("input_tokens", len(ids))
        record_property("requested_output_tokens", OUTPUTS)
        record_property("max_input_tokens", 8)
        record_property("outcome", "source_template_oversize_refusal")
        record_property("oracle_and_audit_allowance_bytes", extra)
        record_property("checked_selections", 0)
    finally:
        guard.close()
    check()
    record_property("maximum_swap_growth_bytes", guard.maximum_swap_growth_bytes)


def test_corrected_prompt_workspace_prices_complete_owners_without_arrays():
    cfg = {"hidden_size": 896, "intermediate_size": 4864,
           "num_attention_heads": 14, "num_key_value_heads": 2,
           "num_hidden_layers": 24, "vocab_size": 151936}
    priced = corrected_prompt_workspace_estimate(cfg)
    # Independent shape/lifetime ledger: literal fixture sizes, not a second
    # transcription of the estimator's algebra. No arrays/model reads needed.
    # Each 128x4864 block: original f32, cast/scale/rint f64, current+old i16;
    # additionally two 64 KiB source/conversion frames can overlap.
    streamed_blocks = {
        "source_f32": 2490368, "cast_f64": 4980736,
        "scaled_f64": 4980736, "rounded_f64": 4980736,
        "current_i16": 1245184, "previous_i16": 1245184,
        "source_frames": 131072,
    }
    assert priced["weight_blocks_bytes"] == sum(streamed_blocks.values()) == 20054016
    # Inputs [30,4864] i64; full head [1,151936] f32; twelve [30,128]
    # i64-equivalent mask/index/accumulation temporaries; two f64 operand sets
    # [30,128] and [128,128]. These must not collapse to last-position pricing.
    contraction_owners = (1167360, 607744, 368640, 323584)
    assert priced["contractions_bytes"] == sum(contraction_owners) == 2467328
    # Live-local envelopes: 32 hidden [30,896], 16 MLP [30,4864],
    # 16 attention [14,30,31], 16 KV projection [30,2,64], all f32.
    activation_owners = (3440640, 9338880, 833280, 245760)
    assert priced["all_position_activations_bytes"] == sum(activation_owners) == 13858560
    # Complete KV [24,2,31,2,64] f32 = 761856; two complete vocabulary
    # selections [2,151936] f32 = 1215488. Four owner-equivalent copies cover
    # expected/accepted bytes, native/returned overlap and cache/concat snapshots.
    # Lifetime envelope is an estimate, not exact simultaneous allocator custody.
    assert priced["complete_state_bytes"] == 4 * (761856 + 1215488) == 7909376
    assert priced["audit_scratch_estimate_bytes"] == 67108864
    assert priced["blas_workspace_estimate_bytes"] == 67108864
    # Tokenizer objects retire before arithmetic, but final decode overlaps
    # results; 384 MiB allowance is NOT proof that retained arenas are reclaimed.
    assert priced["tokenizer_phase_estimate_bytes"] == 402653184 + 7909376 == 410562560
    assert (sum(streamed_blocks.values()) + sum(contraction_owners)
            + sum(activation_owners) + 7909376 + 134217728) == 178507008
    assert priced["required_estimate_bytes"] == 410562560  # Estimated peak: tokenizer + results.
    # Same fixed workload, larger admitted geometry: refuse, never shrink rows
    # or omit state checks to fit the unchanged extra-owner cap.
    assert corrected_prompt_workspace_estimate({**cfg, "intermediate_size": 1 << 20})[
        "required_estimate_bytes"] > 512 << 20
    with pytest.raises(ValueError, match="30 prefill/31 executed"):
        corrected_prompt_workspace_estimate(cfg, prefill=1)


def _corrected_prompt_memory_admission(experiment, snapshot):
    """Config-only estimate first; provisional bytes never grant source custody."""
    from pllm.runtime import benchmark_memory as memory, delegated_source, tokenizer_admission

    assert (experiment.budget.requests, experiment.budget.max_input_tokens,
            experiment.budget.max_new_tokens) == (1, 30, OUTPUTS)
    # Existing approved bounded named-metadata reader ONLY. No HF resolution,
    # tokenizer/index/header/tensor reads or native admission in this precheck.
    with tokenizer_admission.reserve_decoder_metadata():
        raw_config, _ = tokenizer_admission._read_named(snapshot, "config.json", 16 << 10)
        config = tokenizer_admission._json_metadata(raw_config,
            error=memory.BenchmarkMemoryError, label="corrected prompt provisional config")
        geometry = config.get("text_config") if type(config.get("text_config")) is dict else config
        workspace = corrected_prompt_workspace_estimate(geometry, prefill=30, executed=31,
                                                        selections=OUTPUTS)
        extra = 512 << 20  # Unchanged allowance cap; NOT a physical/RSS hard limit.
        if workspace["required_estimate_bytes"] > extra:
            raise memory.BenchmarkMemoryError(
                "corrected prompt independent oracle workspace estimate exceeds unchanged cap: "
                f"estimated={workspace['required_estimate_bytes']} cap={extra}; "
                "30 prefill/31 executed, all logits/KV/selection checks retained")

    # These calls may hash checkpoint bodies: never move above the precheck.
    report = memory.benchmark_memory(experiment, backend="native")
    estimate = copy.deepcopy(report["estimate"])
    source = delegated_source.admit_source(experiment)
    assert source.path.resolve() == snapshot.resolve()
    assert source.config_source_bytes == raw_config
    assert corrected_prompt_workspace_estimate(source.config) == workspace
    # Equality after REAL source admission associates provisional geometry with
    # its owner; labels/digests/precheck results cannot mint an admitted owner.
    estimate["client_peak_bytes"] += extra
    estimate["native_total_peak_bytes"] += extra
    estimate["components"]["independent_oracle_allowance_bytes"] = extra
    report = memory.admit_memory(estimate, memory.host_memory(), backend="native")
    memory.require_admission(report)
    return source, estimate, report, workspace, extra


@pytest.mark.parametrize("budget", [(1, 29, 2), (1, 31, 2), (1, 30, 3), (2, 30, 2)])
def test_corrected_prompt_budget_refuses_before_metadata(monkeypatch, budget):
    from pllm.runtime import benchmark_memory as memory, delegated_source, tokenizer_admission

    events = []

    def forbidden(name):
        def record_and_raise(*args, **kwargs):
            events.append(name)
            raise AssertionError(f"{name} before corrected prompt budget refusal")
        return record_and_raise

    class PoisonSnapshot:
        resolve = forbidden("snapshot.resolve")
        __fspath__ = forbidden("snapshot.__fspath__")
        __str__ = forbidden("snapshot.__str__")

    monkeypatch.setattr(tokenizer_admission, "reserve_decoder_metadata",
                        forbidden("reserve_decoder_metadata"))
    monkeypatch.setattr(tokenizer_admission, "_read_named", forbidden("_read_named"))
    monkeypatch.setattr(memory, "benchmark_memory", forbidden("benchmark_memory"))
    monkeypatch.setattr(delegated_source, "admit_source", forbidden("admit_source"))
    experiment = SimpleNamespace(budget=SimpleNamespace(
        requests=budget[0], max_input_tokens=budget[1], max_new_tokens=budget[2]))

    with pytest.raises(AssertionError) as refusal:
        _corrected_prompt_memory_admission(experiment, PoisonSnapshot())
    assert type(refusal.value) is AssertionError
    assert str(refusal.value).splitlines()[0] == f"assert {budget} == (1, 30, 2)"
    assert events == []


def test_corrected_prompt_oversized_geometry_refuses_before_native_or_checkpoint_body(monkeypatch):
    from pllm.deployment import resources
    from pllm.runtime import benchmark_memory as memory, delegated_source, tokenizer_admission

    events = []
    cfg = {"hidden_size": 896, "intermediate_size": 1 << 20,
           "num_attention_heads": 14, "num_key_value_heads": 2,
           "num_hidden_layers": 24, "vocab_size": 151936}

    @contextmanager
    def metadata_reservation(amount):
        assert amount == 402653184
        events.append("metadata-enter")
        try:
            yield
        finally:
            events.append("metadata-exit")

    def config_only(root, name, limit):
        assert root == Path("unused-offline-snapshot")
        assert (name, limit) == ("config.json", 16384)
        events.append("config-only")
        return json.dumps(cfg).encode(), None

    def forbidden(*args, **kwargs):
        events.append("forbidden-native-or-body")
        raise AssertionError("native admission/checkpoint access before workspace precheck")

    monkeypatch.setattr(resources, "reserve_incremental_memory", metadata_reservation)
    monkeypatch.setattr(tokenizer_admission, "_read_named", config_only)
    monkeypatch.setattr(memory, "benchmark_memory", forbidden)
    monkeypatch.setattr(delegated_source, "admit_source", forbidden)
    monkeypatch.setattr(delegated_source, "decoder_contract", forbidden)
    for name in ("resolve_experiment", "gkm_decoder_contract", "gkm_runtime_contract",
                 "_GkmTransportImport"):
        monkeypatch.setattr(delegated_source._native, name, forbidden)
    for name in ("_FixedReader", "_locked", "_store", "_files", "_read_bounded"):
        monkeypatch.setattr(delegated_source, name, forbidden)
    experiment = SimpleNamespace(budget=SimpleNamespace(
        requests=1, max_input_tokens=30, max_new_tokens=2))
    with pytest.raises(memory.BenchmarkMemoryError,
                       match="workspace estimate exceeds unchanged cap: .*30 prefill/31 executed"):
        _corrected_prompt_memory_admission(experiment, Path("unused-offline-snapshot"))
    assert events == ["metadata-enter", "config-only", "metadata-exit"]


@contextmanager
def _corrected_prompt_extra_allowance(extra):
    """Ordinary logical reservation; caller retires payloads before scope exit."""
    from pllm.deployment import resources

    assert 0 < extra <= 512 << 20
    reservation = resources.reserve_incremental_memory(extra)
    try:
        with reservation:
            with resources._capacity.lock:
                assert resources._capacity._owners[reservation._token] == extra
            yield reservation
    finally:
        if reservation._acquired:
            assert reservation._exited
            with resources._capacity.lock:
                assert reservation._token not in resources._capacity._owners
    # Ledger retirement is not evidence of physical allocator reclamation.


@pytest.mark.parametrize("fail_after_acquisition", [False, True])
def test_corrected_prompt_extra_allowance_retires_after_protected_work(
        monkeypatch, fail_after_acquisition):
    from pllm.deployment import resources

    # Small real ledger control, not a substitute workload or cap adjustment.
    extra = 64 << 10
    reservations, references, events = [], [], []
    original_reserve = resources.reserve_incremental_memory
    original_exit = resources._MemoryReservation.__exit__

    def reserve(amount):
        assert amount == extra
        reservation = original_reserve(amount)
        reservations.append(reservation)
        return reservation

    def retire(reservation, *exc):
        try:
            assert reservation is reservations[0]
            assert references and all(reference() is None for reference in references)
            with resources._capacity.lock:
                assert resources._capacity._owners[reservation._token] == extra
            events.append("owners-retired-still-charged")
        finally:
            original_exit(reservation, *exc)
        with resources._capacity.lock:
            assert reservation._token not in resources._capacity._owners
        events.append("allowance-retired")

    monkeypatch.setattr(resources, "reserve_incremental_memory", reserve)
    monkeypatch.setattr(resources._MemoryReservation, "__exit__", retire)

    def protected_work():
        with _corrected_prompt_extra_allowance(extra) as reservation:
            assert reservation is reservations[0]
            with resources._capacity.lock:
                assert resources._capacity._owners[reservation._token] == extra
            events.append("charged-before-arrays")
            array = np.empty(8, dtype=np.float32)
            references.append(weakref.ref(array))
            try:
                events.append("protected-work")
                if fail_after_acquisition:
                    raise RuntimeError("corrected allowance protected work failure")
            finally:
                array = None  # Retire payload on both paths, before reservation exit.

    if fail_after_acquisition:
        with pytest.raises(RuntimeError, match="corrected allowance protected work failure"):
            protected_work()
    else:
        protected_work()
    reservation, = reservations
    assert reservation._exited
    assert events == ["charged-before-arrays", "protected-work",
                      "owners-retired-still-charged", "allowance-retired"]


def _independent_pinned_qwen_prompt(source, prompt):
    """Render pinned source template independently; never call prepared helpers."""
    from jinja2 import StrictUndefined
    from jinja2.sandbox import SandboxedEnvironment
    from tokenizers import Tokenizer

    descriptor = source.tokenizer_descriptor
    assert hashlib.sha256(descriptor["model"]).hexdigest() == source.tokenizer_digest
    assert descriptor["chat_template"], "pinned Qwen chat template missing; no fallback"
    environment = SandboxedEnvironment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)

    def refuse(message):
        raise AssertionError(message)

    environment.globals["raise_exception"] = refuse
    rendered = environment.from_string(descriptor["chat_template"]).render(
        messages=[{"role": "user", "content": prompt}], tools=None,
        add_generation_prompt=True, bos_token=descriptor["bos_token"],
        eos_token=descriptor["eos_token"])
    assert rendered == (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    tokenizer = Tokenizer.from_buffer(descriptor["model"])
    assert tokenizer.truncation is None and tokenizer.padding is None
    ids = tuple(tokenizer.encode(rendered).ids)
    assert type(descriptor["add_bos_token"]) is bool
    if descriptor["add_bos_token"]:
        ids = (descriptor["bos_token_id"],) + ids
    return ids


def _independent_pinned_qwen_hello(source):
    ids = _independent_pinned_qwen_prompt(source, "Hello")
    assert len(ids) == 30, "pinned source ChatML/BOS/generation rendering must yield 30 tokens"
    return ids


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("PLLM_RUN_GKM_CORRECTED_PROMPT") != "1",
                    reason="opt-in pinned offline GKM corrected Hello complete checkpoint")
def test_pinned_qwen_corrected_prompt(tmp_path, monkeypatch, record_property):
    """Separate current-request qualification; legacy checkpoint cohort stays historical.

    Future launch requires explicit permission, coherent wheel and physical
    headroom. Opt-in is not permission to bypass ordinary source/memory admission.
    """
    # Import candidate configuration ONLY after opt-in. Never construct alternate
    # token-vector requests, pipelines or execution overrides for this cohort.
    from examples.benchmarks.gkm_corrected_prompt import experiment as saved
    from pllm import Deployment, OpenAI
    from pllm.runtime import benchmark_memory as memory, benchmark_response_audit as audit, delegated_source
    from pllm.runtime.safetensors_store import SafeTensorStore
    from test_delegated_sdk import active_sessions

    cache = Path(HF_HUB_CACHE).expanduser().resolve()
    snapshot = cache / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / PIN
    if not snapshot.is_dir():
        pytest.skip(f"missing pinned offline checkpoint cache: {snapshot}; no download/fallback")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    experiment = saved.with_params(deployment=Deployment.local(root=str(tmp_path / "roles")))
    model = experiment.pipeline.model
    assert model == saved.pipeline.model
    assert model.source == MODEL
    assert model.local_files_only and model.revision == PIN
    linear = experiment.pipeline.components["linear"].params
    assert linear["capacity"] == "decoder" and linear["sharing"] == "full"
    assert linear.get("execution") is None
    assert (experiment.budget.requests, experiment.budget.max_input_tokens,
            experiment.budget.max_new_tokens) == (1, 30, OUTPUTS)
    assert experiment.pipeline.digest() == saved.pipeline.digest()

    def forbidden(*args, **kwargs):
        raise AssertionError("corrected checkpoint values loaded before memory admission")

    with monkeypatch.context() as early:
        early.setattr(delegated_source, "_source_float_frame", forbidden)
        early.setattr(delegated_source.AdmittedSource, "load_transport", forbidden)
        early.setattr(SafeTensorStore, "get", forbidden)
        early.setattr(delegated_source._AdmittedTensorStore, "get", forbidden)
        source, estimate, report, workspace, extra = _corrected_prompt_memory_admission(
            experiment, snapshot)

    pins = estimate["gkm_admission"]
    assert source.model == model and source.source_lock["commit"] == PIN
    for key, value in {
        "source_lock_digest": source.source_lock_digest,
        "checkpoint_digest": source.checkpoint_digest,
        "tokenizer_digest": source.tokenizer_digest,
        "decoder_contract_digest": source.contract_digest,
        "runtime_contract_digest": source.runtime_contract_digest,
        "numeric_profile": source.contract["numeric_profile"],
        "local_numeric_profile": source.contract["local_numeric_profile"],
        "max_input_tokens": 30, "max_output_tokens": OUTPUTS,
    }.items():
        assert pins[key] == value
    assert source.configuration_digest == experiment.configuration_digest()
    assert source.composition_digest == experiment.pipeline.digest()
    assert source.contract["input_rows"] == 30 and source.contract["decode_steps"] == 1
    assert source.contract["numeric_profile"] == (
        "gkm.nearest-even.2048.i16.ring64.signed.f32-before-div22.le-finite-f32.sha256.secp256k1.v1")
    assert source.contract["local_numeric_profile"] == (
        "semantic.float32.dense-full-kv.client-bias-after-div22.v1")
    runtime = source.runtime_contract(experiment, 30, OUTPUTS)
    assert runtime == source.runtime_contract_bytes
    assert json.loads(runtime)["decoder"] == source.contract

    stopped, clients = threading.Event(), []

    def abort(reason):
        stopped.set()
        for client in clients:
            active = client._core._active
            if active is not None:
                active.cancel()

    guard = memory.MemoryWatchdog(report, abort)

    def check():
        if stopped.is_set():
            raise memory.BenchmarkMemoryError(guard.error or "corrected checkpoint test cancelled")
        guard.check()

    captured, salt = [], os.urandom(32)
    observer = SelectionCommitments(salt, max_selections=OUTPUTS)
    owners = ExitStack()
    try:
        # Reuse process-wide incremental-owner ledger. It accounts the allowance
        # against live capacity, but cannot enforce allocator/BLAS RSS limits.
        allowance = owners.enter_context(_corrected_prompt_extra_allowance(extra))
        guard.start()
        check()
        ids = _independent_pinned_qwen_hello(source)
        gc.collect()  # Independent tokenizer/template phase retired before arrays.
        check()
        tensors = AdmittedTensors(source, check=check)
        tensors.verify_lock()
        expected = clear_decoder(source.config, tensors, ids, OUTPUTS - 1, check=check,
                                 selection_observer=observer)
        assert [row["position"] for row in observer.selections] == [30, 31]
        cfg = source.config
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        for row, position in zip(observer.selections, (30, 31), strict=True):
            assert row["checked_values"] == (cfg["vocab_size"] + 2 * cfg["num_hidden_layers"]
                                             * position * cfg["num_key_value_heads"] * head_dim)
        tensors.verify_lock()
        with audit.response_state_audit(salt, runtime_kind="gkm") as records:
            with OpenAI(experiment=experiment, timeout=600) as client:
                topology = client._owned_topology
                clients.append(client)
                client._core._result_hook = captured.append
                check()
                # Ordinary one-use preparation remains runtime authority. Oracle
                # IDs never enter request, prepared payload or transport inputs.
                response = client.responses.create(input="Hello", max_output_tokens=OUTPUTS)
                assert response.status == "incomplete"
                assert response.incomplete_details == {"reason": "max_output_tokens"}
                actual = client._core._ensure_source()
                for attribute in ("model", "config_bytes", "contract_bytes", "experiment_bytes",
                                  "runtime_contract_bytes", "source_lock_digest", "checkpoint_digest",
                                  "tokenizer_digest", "configuration_digest", "composition_digest"):
                    assert getattr(actual, attribute) == getattr(source, attribute)
                result, = captured
                assert_complete_result(result, expected)
                assert result.executed == ids + result.selected[:-1]
                assert result.position == 31 and len(result.selected) == OUTPUTS
                acceptance = json.loads(result.acceptance)
                assert acceptance["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
                assert acceptance["sharing"] == "full"
                assert acceptance["contract"] == hashlib.sha256(
                    b"pllm.gkm.sdk-contract.v1\0" + runtime).hexdigest()
                assert response.usage.input_tokens == 30
                assert response.usage.output_tokens == OUTPUTS
                assert response.output_text == source.tokenizer().decode(list(result.selected))
                assert active_sessions(client._core) == [0, 0]
                roles = topology.statuses
                assert {role.role for role in roles} == {"worker_a", "worker_b"}
                assert len({role.pid for role in roles}) == 2
                assert all(role.running and role.pid != os.getpid() for role in roles)
                assert records == []
                check()
            assert all(not status.running for status in topology.statuses)
        tensors.verify_lock()
        assert records == [observer.expected_record(source.contract)]
        check()
        for key, value in {
            "cohort": "pinned_qwen_corrected_prompt", "model_revision": PIN,
            "configuration_digest": source.configuration_digest,
            "composition_digest": source.composition_digest,
            "config_digest": source.config_digest, **pins,
            "input_tokens": 30, "executed_positions": 31, "output_tokens": OUTPUTS,
            "sharing": "full", "execution": "scalar", "oracle_row_block": ROW_BLOCK,
            "oracle_workspace_estimate": workspace, "oracle_and_audit_allowance_bytes": extra,
            "checked_selections": len(observer.selections),
            "checked_selection_float32_values": sum(row["checked_values"] for row in observer.selections),
        }.items():
            record_property(key, value)
    except BaseException as error:
        from pllm.runtime.delegated_protocol import clear_failure_frames

        # Failed decoder/tokenizer frames must not keep payloads beyond refund.
        clear_failure_frames(error)
        raise
    finally:
        # OpenAI context cancels/joins roles on exceptions; watchdog cancellation
        # also reaches any live attempt. Closing guard never grants new work.
        try:
            guard.close()
        finally:
            try:
                captured.clear()
                clients.clear()
                expected = result = response = client = actual = tensors = source = None
                gc.collect()  # Drop references; not proof of physical reclamation.
            finally:
                owners.close()
    assert allowance._exited
    check()
    record_property("maximum_swap_growth_bytes", guard.maximum_swap_growth_bytes)
