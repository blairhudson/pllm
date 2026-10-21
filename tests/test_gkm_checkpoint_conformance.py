"""Independent complete-state GKM conformance, generated and opt-in offline Qwen.

Run optional test with PLLM_RUN_GKM_CHECKPOINT=1; use ordinary HF_HUB_CACHE to
select cache location. No downloads, wheel builds, or benchmark driver needed.
Use pytest --basetemp=target/gkm-conformance for temporary source/role files.
"""

import copy
import gc
import hashlib
import json
import os
import threading
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from gkm_decoder_oracle import (
    AdmittedTensors, ArrayTensors, CONTRACTION_COLUMNS, INPUT_BOUND, ROW_BLOCK, SelectionCommitments,
    assert_complete_result, clear_decoder, encode_activation_q11, encode_weight_q11,
    exact_contract,
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


@pytest.mark.slow
@pytest.mark.parametrize("sharing", ["full", "seeded"])
@pytest.mark.parametrize("execution", [None, "source_stage_batch_v1"])
@pytest.mark.skipif(os.getenv("PLLM_RUN_GKM_CHECKPOINT") != "1",
                    reason="opt-in pinned offline GKM complete checkpoint")
def test_pinned_qwen_checkpoint_two_role_sdk_complete_accepted_state(tmp_path, monkeypatch,
                                                                  record_property, sharing, execution):
    from pllm import Deployment, ExecutionBudget, Experiment, Model, OpenAI
    from pllm.profiles import OpenWeightVerifiableCpu
    from pllm.protocols import OpenWeightVerifiableOutsourcing
    from pllm.runtime import benchmark_memory as memory, benchmark_response_audit as audit, delegated_source
    from pllm.runtime.responses import normalize_input, prompt_text
    from pllm.runtime.safetensors_store import SafeTensorStore
    from test_delegated_sdk import active_sessions

    # Select the existing standard shared cache before pytest's per-test XDG
    # isolation. The optional offline checkpoint must not use an empty scratch cache.
    monkeypatch.setenv("HF_HUB_CACHE", HF_HUB_CACHE)
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
        # Bounded Qwen rows, f64 blocks, local activations, retained expected /
        # accepted full state, BLAS workspace; reserve 64 MiB within this allowance
        # for independent / production audit scratch. Extra owner, not cache.
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
            active = client._core._active
            if active is not None:
                active.cancel()

    guard = memory.MemoryWatchdog(report, abort)

    def check():
        if stopped.is_set():
            raise memory.BenchmarkMemoryError(guard.error or "checkpoint test cancelled")
        guard.check()

    captured = []
    salt = os.urandom(32)
    observer = SelectionCommitments(salt, max_selections=OUTPUTS)
    guard.start()
    try:
        source = delegated_source.admit_source(experiment)
        assert source.model == model and source.source_lock["commit"] == PIN
        assert source.source_lock_digest == estimate["gkm_admission"]["source_lock_digest"]
        assert source.checkpoint_digest == estimate["gkm_admission"]["checkpoint_digest"]
        ids = tuple(source.tokenizer().encode(prompt_text(normalize_input(PROMPT)), add_bos=False))
        assert 1 <= len(ids) <= experiment.budget.max_input_tokens
        tensors = AdmittedTensors(source, check=check)
        tensors.verify_lock()
        expected = clear_decoder(source.config, tensors, ids, OUTPUTS - 1, check=check,
                                 selection_observer=observer)
        tensors.verify_lock()
        with (audit.response_state_audit(salt, runtime_kind="gkm") as records,
              OpenAI(experiment=experiment, timeout=3600) as client):
            topology = client._owned_topology
            clients.append(client)
            client._core._result_hook = captured.append
            check()
            response = client.responses.create(input=PROMPT, max_output_tokens=OUTPUTS)
            actual_source = client._core._ensure_source()
            assert actual_source.source_lock_digest == source.source_lock_digest
            assert actual_source.checkpoint_digest == source.checkpoint_digest
            assert actual_source.tokenizer_digest == source.tokenizer_digest
            result, = captured
            assert_complete_result(result, expected)
            assert result.executed == ids + result.selected[:-1]
            acceptance = json.loads(result.acceptance)
            runtime = source.runtime_contract(experiment, len(ids), OUTPUTS)
            assert acceptance["schema_version"] == "pllm.gkm.sdk-acceptance.v1"
            assert acceptance["sharing"] == sharing
            assert acceptance["contract"] == hashlib.sha256(
                b"pllm.gkm.sdk-contract.v1\0" + runtime).hexdigest()
            assert response.usage.input_tokens == len(ids)
            assert response.usage.output_tokens == OUTPUTS
            assert response.output_text == source.tokenizer().decode(list(result.selected))
            assert active_sessions(client._core) == [0, 0]
            roles = client._owned_topology.statuses
            assert {role.role for role in roles} == {"worker_a", "worker_b"}
            assert len({role.pid for role in roles}) == 2
            assert all(role.running and role.pid != os.getpid() for role in roles)
            assert records == []
            check()
        tensors.verify_lock()
        assert all(not status.running for status in topology.statuses)
        assert records == [observer.expected_record(source.contract)]
        record_property("sharing", sharing)
        record_property("execution", execution or "scalar")
        record_property("source_lock_digest", source.source_lock_digest)
        record_property("checkpoint_digest", source.checkpoint_digest)
        record_property("tokenizer_digest", source.tokenizer_digest)
        record_property("model_revision", PIN)
        record_property("input_tokens", len(ids))
        record_property("output_tokens", OUTPUTS)
        record_property("oracle_row_block", ROW_BLOCK)
        record_property("oracle_and_audit_allowance_bytes", extra)
        record_property("checked_float32_values", (
            sum(map(len, result.logits))
            + sum(len(key) + len(value) for _, key, value in result.kv)) // 4)
        record_property("checked_selections", len(observer.selections))
        record_property("checked_selection_float32_values",
                        sum(row["checked_values"] for row in observer.selections))
    finally:
        guard.close()
    check()
    record_property("maximum_swap_growth_bytes", guard.maximum_swap_growth_bytes)
