"""Generated source -> native plan -> normal importer -> semantic execution.

Positive numerical oracles use no mocks or alternate loader. Negative tests
only observe real dispatch or corrupt a real lookup result at its boundary.
Explicit ClientOnlyCpu W8A8 source/reference path;
neither prepared encrypted execution nor encrypted-paper completion is claimed.
These regressions require a matching installed native extension. Do not xfail
missing operator/import support: completing that shared support is the gate.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

import pllm
from pllm.profiles import ClientOnlyCpu, PumaThreePartyCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.model_execution import RuntimeExecutionError
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_tensors import SemanticTensorError, preflight_semantic_checkpoint
from pllm.runtime.transformer_client import ClientBundle, TransformerClientError
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError

from absolute_position_decoder_oracle import evaluate, quantized_rows, tiny_checkpoint, write_checkpoint


MODEL_ID = "tiny-absolute-position-source"
# Only ordinary FP32 reduction/transcendental roundoff is allowed here. W8A8
# integer weights and scales are separately checked exactly below; this is not
# a tolerance against an unquantized model, Q11, Q18, or a protected approximation.
F32_ATOL = 2e-6
F32_RTOL = 2e-5


def _checkpoint(root: Path, *, canonical_context: bool = False, change=None):
    config, weights = tiny_checkpoint()
    if canonical_context:
        # Explicit second case isolates runtime while the n_positions-only case
        # below independently exposes any production importer alias omission.
        config["max_position_embeddings"] = config["n_positions"]
    if change is not None:
        change(weights)
    write_checkpoint(root, config, weights)
    return config, weights


def _plan(config):
    return pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)


def _bind(root, config):
    plan = _plan(config)
    composition = ClientOnlyCpu(
        pllm.Model(MODEL_ID),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    )
    schedule = plan.runtime_schedule(composition)
    assert plan.coverage(composition).complete
    assert schedule.complete and not schedule.protected_execution
    manifest = load_hf_directory(root, model_id=MODEL_ID)
    assert manifest.stages, "normal importer must derive real semantic stages"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=2)
    asyncio.run(engine.load(manifest))
    bundle = ClientBundle.unpack(engine.client_bundle(MODEL_ID, placement="client"))
    descriptor = json.loads((root / "pllm_tokenizer.json").read_text(encoding="utf-8"))
    assert bundle.tokenizer_descriptor == descriptor
    tokenizer = bundle.tokenizer()
    assert tokenizer.vocab_size == bundle.cfg["vocab_size"] == config["vocab_size"] == 11
    for name in ("bos_token_id", "eos_token_id"):
        assert getattr(tokenizer, name) == descriptor[name] == bundle.cfg[name]
    source_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        assert source_config[name] == descriptor[name]
        assert 0 <= descriptor[name] < tokenizer.vocab_size
    encoded = tokenizer.encode(descriptor["alphabet"], add_bos=True)
    assert encoded == [tokenizer.bos_token_id, *range(len(descriptor["alphabet"]))]
    assert all(0 <= token < tokenizer.vocab_size for token in encoded)
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    return plan, engine, compiled


def _assert_imported_numeric_contract(engine, weights):
    for stage in engine.models[MODEL_ID].stages.values():
        assert stage.spec.weight_bits == stage.spec.activation_bits == 8
        sources = [weights[key].T if stage.spec.transpose_weight else weights[key]
                   for key in stage.spec.weight_keys]
        expected_codes, expected_scales = quantized_rows(np.concatenate(sources, axis=0))
        np.testing.assert_array_equal(stage.weight.values, expected_codes)
        np.testing.assert_array_equal(stage.weight.scales, expected_scales)
        if stage.spec.bias_keys:
            expected_bias = np.concatenate([weights[key] for key in stage.spec.bias_keys])
            np.testing.assert_array_equal(stage.bias, expected_bias)


def _assert_result(session, actual_logits, expected, length):
    assert actual_logits.dtype == np.float32 and actual_logits.shape == (11,)
    np.testing.assert_allclose(actual_logits, expected.logits, atol=F32_ATOL, rtol=F32_RTOL)
    assert session.position == length
    assert session._runtime.position == length
    assert len(session._runtime.caches) == 2
    for cache, key, value in zip(session._runtime.caches, expected.keys, expected.values, strict=True):
        assert cache.length == length
        np.testing.assert_allclose(cache.key[:length], key, atol=F32_ATOL, rtol=F32_RTOL)
        np.testing.assert_allclose(cache.value[:length], value, atol=F32_ATOL, rtol=F32_RTOL)


def _observe_stage_dispatch(session):
    """Negative-only observer: forward every real stage to its original consumer.

    Positive whole-graph tests above/below do not install this instrumentation.
    Includes token/head stages as well as native body stages; no result is
    synthesized, and the bound native executor itself is never replaced.
    """
    original = session._runtime._linear_step
    calls = []

    def observed(phase, step, *args, **kwargs):
        calls.append((phase, tuple(step["operation_ids"])))
        return original(phase, step, *args, **kwargs)

    session._runtime._linear_step = observed
    return calls


def _active_snapshot(session, executor, calls):
    """Bitwise snapshot of every active KV element, not merely cache lengths."""
    return (
        session.position, session._runtime.position, session._status,
        session._generated, session._pending_token, session.logits.tobytes(),
        executor.integer_macs, tuple(calls), id(session._runtime.caches),
        tuple((cache.length, id(cache.key), id(cache.value),
               cache.key[:cache.length].shape, cache.key[:cache.length].dtype.str,
               cache.key[:cache.length].tobytes(),
               cache.value[:cache.length].shape, cache.value[:cache.length].dtype.str,
               cache.value[:cache.length].tobytes())
              for cache in session._runtime.caches),
    )


@pytest.mark.parametrize("initial_tokens", [[], [2, 9, 4]], ids=["fresh", "populated-prefix"])
def test_absolute_position_early_token_width_failure_preserves_real_owner_positions(
    tmp_path, monkeypatch, initial_tokens,
):
    root = tmp_path / "checkpoint"
    config, weights = _checkpoint(root, canonical_context=True)
    plan, engine, compiled = _bind(root, config)
    executor = compiled.client_linear_executor(engine)
    session = compiled.session(executor)
    try:
        if initial_tokens:
            _assert_result(session, session.prefill_ids(initial_tokens),
                           evaluate(weights, initial_tokens), len(initial_tokens))
        runtime = session._runtime
        owner = runtime._compiled_binding
        assert owner is compiled and session._compiled is compiled
        positions = (session.position, runtime.position)
        macs = executor.integer_macs
        phase = "prefill" if runtime.position == 0 else "decode"
        operation = next(op for op in plan.to_dict()[phase]["operations"]
                         if op["operator"] == "token_lookup" and op["inputs"] == ["input.tokens"])
        stage = runtime.bundle.stages[runtime._stages[f"{phase}:{operation['id']}"]]
        original_lookup = runtime._token_lookup
        calls = []

        def invalid_width(ids):
            primary, auxiliary = original_lookup(ids)
            calls.append(ids.copy())
            original_width = primary.shape[-1] + (0 if auxiliary is None else auxiliary.shape[-1])
            assert original_width == stage.out_features
            return primary[..., :-1], auxiliary

        # This probes only an early token-boundary failure, before body/KV work;
        # it does not claim general rollback after partially executed KV updates.
        with monkeypatch.context() as guard:
            guard.setattr(runtime, "_token_lookup", invalid_width)
            with pytest.raises(TransformerClientError, match="^semantic token lookup has an invalid width$"):
                runtime.forward_ids(np.asarray([2], dtype=np.int64))
        assert len(calls) == 1
        np.testing.assert_array_equal(calls[0], np.asarray([2], dtype=np.int64))
        assert session._runtime is runtime
        assert runtime._compiled_binding is owner and session._compiled is owner
        assert (session.position, runtime.position) == positions
        assert executor.integer_macs == macs
    finally:
        session.close()


@pytest.mark.parametrize("canonical_context", [False, True], ids=["source-n-positions", "matching-canonical-context"])
def test_absolute_position_checkpoint_executes_prefill_and_selected_decode(tmp_path, canonical_context):
    root = tmp_path / "checkpoint"
    config, weights = _checkpoint(root, canonical_context=canonical_context)
    plan, engine, compiled = _bind(root, config)
    original = plan.canonical_bytes()
    _assert_imported_numeric_contract(engine, weights)
    executor = compiled.client_linear_executor(engine)
    session = compiled.session(executor)
    tokens = [2, 9, 4]  # token 9 is valid despite position table cardinality 8.
    executed_tokens = []
    try:
        expected = evaluate(weights, tokens)
        _assert_result(session, session.prefill_ids(tokens), expected, len(tokens))
        for _ in range(2):
            selected = session.select_next()
            assert selected == expected.selected
            executed_tokens.append(selected)
            tokens.append(expected.selected)  # Oracle never consumes actual selected IDs.
            expected = evaluate(weights, tokens)
            _assert_result(session, session.decode_selected(), expected, len(tokens))
        reference_tokens = [2, 9, 4]
        for actual in executed_tokens:
            independent = evaluate(weights, reference_tokens).selected
            assert actual == independent
            reference_tokens.append(independent)
        assert executor.integer_macs > 0, "must execute admitted native integer body kernels"
        assert plan.canonical_bytes() == original
        assert _plan(config).canonical_bytes() == original
    finally:
        session.close()


def test_source_n_positions_imports_real_context_cardinality(tmp_path):
    root = tmp_path / "checkpoint"
    config, _ = _checkpoint(root)
    assert "max_position_embeddings" not in config
    manifest = load_hf_directory(root, model_id=MODEL_ID)
    assert manifest.context_length == 8, "load_hf_directory must honor existing source n_positions"
    plan = _plan(config)
    for graph in (plan.prefill, plan.decode):
        position = next(op for op in graph["operations"] if op["id"] == "position_lookup")
        assert list(position["inputs"]) == ["input.positions"]
        assert list(position["attributes"]["weight_shape"]) == [8, 4]


LOCAL_ARTIFACTS = ["transformer.wpe.weight", "transformer.ln_f.weight", "transformer.ln_f.bias"] + [
    f"transformer.h.{layer}.{norm}.{coefficient}"
    for layer in range(2) for norm in ("ln_1", "ln_2") for coefficient in ("weight", "bias")
]


@pytest.mark.parametrize("artifact", LOCAL_ARTIFACTS)
@pytest.mark.parametrize("damage", ["missing", "shape", "dtype"])
def test_absolute_position_import_refuses_missing_or_wrong_local_artifact(tmp_path, artifact, damage):
    def change(weights):
        if damage == "missing":
            del weights[artifact]
        elif damage == "shape":
            weights[artifact] = weights[artifact][:-1].copy()
        else:
            weights[artifact] = weights[artifact].astype(np.int64)

    root = tmp_path / "checkpoint"
    config, _ = _checkpoint(root, canonical_context=True, change=change)
    plan = _plan(config)
    original = plan.canonical_bytes()
    with pytest.raises(SemanticTensorError, match="missing|shape|dtype|artifact"):
        preflight_semantic_checkpoint(plan, SafeTensorStore(root))
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=2)
    manifest = load_hf_directory(root, model_id=MODEL_ID)
    with pytest.raises(TransformerEngineError, match="missing|shape|dtype|artifact"):
        asyncio.run(engine.load(manifest))
    assert MODEL_ID not in engine.models
    assert plan.canonical_bytes() == original


@pytest.mark.parametrize("token", [-1, 11])
def test_absolute_position_session_refuses_out_of_vocabulary_ids(tmp_path, token):
    root = tmp_path / "checkpoint"
    config, _ = _checkpoint(root, canonical_context=True)
    _, engine, compiled = _bind(root, config)
    executor = compiled.client_linear_executor(engine)
    session = compiled.session(executor)
    calls = _observe_stage_dispatch(session)
    try:
        with pytest.raises((RuntimeExecutionError, TransformerClientError), match="token|vocab|range|domain"):
            session.prefill_ids([token])
        assert session.position == 0
        assert executor.integer_macs == 0 and calls == []
        session.prefill_ids([2, 9, 4])
        assert executor.integer_macs > 0 and calls
        before = _active_snapshot(session, executor, calls)
        # Public facade only accepts its own selected token, not arbitrary decode
        # IDs. Exercise the actual session-owned runtime's decode admission, with
        # populated caches and the same bound native executor; no forged state.
        with pytest.raises(TransformerClientError, match="^lookup IDs are unavailable or outside their declared domain$"):
            session._runtime.decode_step(token, session._runtime.caches, session.position)
        assert _active_snapshot(session, executor, calls) == before
    finally:
        session.close()


@pytest.mark.parametrize("position", [-1, 8])
def test_absolute_position_local_lookup_unit_refuses_out_of_range_position_ids(tmp_path, monkeypatch, position):
    root = tmp_path / "checkpoint"
    config, _ = _checkpoint(root, canonical_context=True)
    plan, engine, compiled = _bind(root, config)
    session = compiled.session(compiled.client_linear_executor(engine))
    original_plan = plan.canonical_bytes()
    runtime = session._runtime
    original_local = runtime._local
    interceptions = []

    def invalid_position(operation, values, *args, **kwargs):
        if operation["id"] == "position_lookup":
            kernel = runtime._local_numeric_by_phase["prefill"][operation["id"]]
            assert runtime._phase_local_numeric[operation["id"]] is kernel
            assert operation == kernel.operation
            assert runtime._phase_numeric_rows == values["input.positions"].size == 1
            interceptions.append(operation["id"])
            values = dict(values)
            values["input.positions"] = np.array([position], dtype=np.int64)
        return original_local(operation, values, *args, **kwargs)

    try:
        # Probe active-consumer numeric domain refusal only. Token lookup may
        # already have executed; this makes no zero-dispatch or rollback claim.
        with monkeypatch.context() as guard:
            guard.setattr(runtime, "_local", invalid_position)
            with pytest.raises(
                RuntimeExecutionError,
                match="^compiled runtime prefill failed$",
            ) as caught:
                session.prefill_ids([2])
        assert type(caught.value.__cause__) is TransformerClientError
        assert str(caught.value.__cause__) == "embedding IDs exceed their compiled domain or row geometry"
        assert interceptions == ["position_lookup"]
        assert plan.canonical_bytes() == original_plan
        assert runtime._phase_local_numeric == {}
        assert runtime._phase_numeric_rows == 0
    finally:
        session.close()


def test_absolute_position_session_at_capacity_refuses_ninth_runtime_step_without_dispatch(tmp_path):
    root = tmp_path / "checkpoint"
    config, weights = _checkpoint(root, canonical_context=True)
    _, engine, compiled = _bind(root, config)
    executor = compiled.client_linear_executor(engine)
    session = compiled.session(executor)
    calls = _observe_stage_dispatch(session)
    tokens = [2, 9, 4, 10, 3, 1]
    try:
        expected = evaluate(weights, tokens)
        _assert_result(session, session.prefill_ids(tokens), expected, 6)
        for length in (7, 8):
            assert session.select_next() == expected.selected
            tokens.append(expected.selected)
            expected = evaluate(weights, tokens)
            _assert_result(session, session.decode_selected(), expected, length)
        assert executor.integer_macs > 0 and calls
        before = _active_snapshot(session, executor, calls)
        # Facade select_next at its final generation budget intentionally wipes
        # state. Do not fake a selected ninth token: submit to its real bound
        # runtime's existing decode API to test position/workload admission.
        with pytest.raises(TransformerClientError, match="^semantic execution exceeds compiled workload bounds$"):
            session._runtime.decode_step(expected.selected, session._runtime.caches, session.position)
        assert _active_snapshot(session, executor, calls) == before
    finally:
        session.close()


def test_absolute_position_unsupported_protected_composition_is_not_silently_clear():
    config, _ = tiny_checkpoint()
    plan = _plan(config)
    original = plan.canonical_bytes()
    protected = PumaThreePartyCpu(pllm.Model(MODEL_ID))
    coverage = plan.coverage(protected)
    assert not coverage.complete
    operators = {row["operator"]: row for row in coverage.to_dict()["operators"]}
    for kind in ("layer_norm", "gelu_tanh"):
        assert operators[kind]["level"] != "executable_region"
    # Exact current native lower_decoder_runtime_schedule admission diagnostic;
    # arbitrary source/numeric ValueError must not satisfy this regression.
    with pytest.raises(ValueError, match="^decoder runtime schedule requires an admitted linear component composition$"):
        plan.runtime_schedule(protected)
    assert plan.canonical_bytes() == original
    assert _plan(config).canonical_bytes() == original


def test_absolute_position_oracle_rounds_signed_ties_to_even():
    # Maximum 127 fixes row scale at exactly one; negative ties matter too.
    codes, scales = quantized_rows(np.array([[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5]], np.float32))
    np.testing.assert_array_equal(scales, np.array([1], np.float32))
    np.testing.assert_array_equal(codes, [[127, 0, 2, 2, 0, -2, -2]])


def test_absolute_position_admitted_quantizers_match_signed_tie_contract():
    from pllm.runtime.quantization import quantize_activation_per_row, quantize_weight_per_row

    source = np.array([[127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5],
                       [0, 0, 0, 0, 0, 0, 0]], np.float32)
    expected_codes, expected_scales = quantized_rows(source)
    for actual in (quantize_weight_per_row(source, bits=8),
                   quantize_activation_per_row(source, bits=8)):
        np.testing.assert_array_equal(actual.values, expected_codes)
        np.testing.assert_array_equal(actual.scales, expected_scales)


@pytest.mark.parametrize("artifact", LOCAL_ARTIFACTS)
def test_absolute_position_local_coefficients_are_live_whole_graph_inputs(tmp_path, artifact):
    _, baseline_weights = tiny_checkpoint()
    tokens = [2, 9, 4]
    baseline = evaluate(baseline_weights, tokens)

    def change(weights):
        # Alter one feature, not an all-feature shift that centered LayerNorm
        # could legitimately remove. Both blocks and final affine remain live.
        weights[artifact] = weights[artifact].copy()
        if weights[artifact].ndim == 2:
            weights[artifact][1, 0] += np.float32(0.5)
        else:
            weights[artifact][0] += np.float32(0.5)

    root = tmp_path / "checkpoint"
    config, weights = _checkpoint(root, canonical_context=True, change=change)
    expected = evaluate(weights, tokens)
    assert np.max(np.abs(expected.logits - baseline.logits)) > F32_ATOL
    _, engine, compiled = _bind(root, config)
    session = compiled.session(compiled.client_linear_executor(engine))
    try:
        _assert_result(session, session.prefill_ids(tokens), expected, len(tokens))
    finally:
        session.close()
