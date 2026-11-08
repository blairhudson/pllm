"""Shared embedding domains, centered affine FP32 norms, and checkpoint layout."""

from __future__ import annotations

import copy
import asyncio
import json
from decimal import Decimal, localcontext
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file

from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.semantic_attention import SemanticAttentionError, mask_causal_scores
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.semantic_numeric import (
    SemanticNumericError,
    bfloat16_rms_norm,
    float32_embedding_lookup,
    float32_gelu_tanh,
    float32_layer_norm,
    round_bfloat16,
    layer_norm_contract,
    local_numeric_workspace,
    _BoundLocalNumeric,
)
from pllm.runtime.semantic_stages import scheduled_stage_specs
from pllm.runtime.semantic_tensors import (
    SemanticTensorError,
    lookup_contract,
    preflight_semantic_checkpoint,
    projection_weight_shape,
    required_checkpoint_tensors,
    required_client_tensors,
)
from pllm.runtime.transformer_client import TransformerClientError


@pytest.fixture
def numeric_binding(tmp_path):
    """Use the ordinary source importer and private compiled tensor owner."""
    import pllm
    from pllm.profiles import ClientOnlyCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.loaders import load_hf_directory
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.transformer_client import ClientBundle
    from pllm.runtime.transformer_engine import MaskedTransformerEngine
    from absolute_position_decoder_oracle import tiny_checkpoint, write_checkpoint

    config, weights = tiny_checkpoint()
    write_checkpoint(tmp_path, config, weights)
    identity = "semantic-numeric-contract"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    asyncio.run(engine.load(load_hf_directory(tmp_path, model_id=identity)))
    bundle = ClientBundle.unpack(engine.client_bundle(identity, placement="client"))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)
    composition = ClientOnlyCpu(pllm.Model(identity), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    owner = compile_runtime_model(plan, bundle, composition=composition)
    return plan, owner, engine


def _kernel(binding, kind="layer_norm", phase="prefill"):
    plan, owner = binding[:2]
    operation = next(op for op in plan.to_dict()[phase]["operations"]
                     if op["operator"] == kind and (kind != "token_lookup" or op["inputs"] == ["input.positions"]))
    return _BoundLocalNumeric(plan, phase, operation["id"], owner)


@pytest.fixture
def continuation_binding(numeric_binding):
    import pllm
    from pllm.profiles import MaskedLinearCpu
    from pllm.quantization import SymmetricPerRow
    from pllm.runtime.model_binding import compile_runtime_model
    from pllm.runtime.transformer_client import ClientBundle
    from test_decoder_continuation import QuantizedRemote

    plan, client_owner, engine = numeric_binding
    identity = client_owner._bundle.model_id
    bundle = ClientBundle.unpack(engine.client_bundle(identity))
    composition = MaskedLinearCpu(pllm.Model(identity),
                                 quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    owner = compile_runtime_model(plan, bundle, composition=composition)
    remote = QuantizedRemote(engine, identity)
    runtime = owner.runtime(remote)
    runtime.prepare_ids([1, 2])
    contract = plan.continuation_schedule(composition)
    runtime.install_continuation(contract)
    return runtime, remote, contract


class _UnwalkableSequence:
    def __len__(self):
        raise AssertionError("custom sequence length invoked")

    def __iter__(self):
        raise AssertionError("nested/custom sequence traversed")

    def __array__(self, *args, **kwargs):
        raise AssertionError("custom sequence conversion invoked")


class _UnwalkableList(list):
    def __len__(self):
        raise AssertionError("custom list length invoked")

    def __iter__(self):
        raise AssertionError("custom list iterator invoked")


class _NotLiteralInt(int):
    def __int__(self):
        raise AssertionError("custom integer conversion invoked")


def _rejected_token_payload(case, limit):
    if case == "oversized_nested":
        return [[1] * (1 << 16)]
    if case == "nested":
        return [[1], [2]]
    if case == "boolean_mix":
        return [1, True]
    if case == "numpy_boolean_mix":
        return [1, np.bool_(False)]
    if case == "noninteger":
        return [0.5]
    if case == "nested_custom":
        return [_UnwalkableSequence()]
    if case == "custom":
        return _UnwalkableSequence()
    if case == "custom_list":
        return _UnwalkableList([1])
    if case == "custom_integer":
        return [_NotLiteralInt(1)]
    if case == "negative":
        return [-1]
    if case == "large_integer":
        return [1 << 80]
    if case == "tuple":
        return (1, 2)
    if case == "range":
        return range(2)
    if case == "oversized_list":
        return [_UnwalkableSequence()] * (limit + 1)
    if case == "oversized_array":
        return np.ones(limit + 1, dtype=np.int64)
    if case == "array_rank":
        return np.ones((1, limit + 1), dtype=np.int64)
    if case == "array_scalar":
        return np.asarray(1, dtype=np.int64)
    if case == "array_dtype":
        return np.ones(1, dtype=np.float32)
    if case == "array_boolean":
        return np.ones(1, dtype=np.bool_)
    if case == "array_object":
        return np.asarray([1], dtype=object)
    if case == "array_negative":
        return np.asarray([-1], dtype=np.int64)
    if case == "array_uint64_overflow":
        return np.asarray([1 << 63], dtype=np.uint64)
    raise AssertionError(case)


_REJECTED_TOKEN_CASES = (
    "oversized_nested", "nested", "boolean_mix", "numpy_boolean_mix", "noninteger",
    "nested_custom", "custom", "custom_list", "custom_integer", "negative", "large_integer",
    "tuple", "range", "oversized_list", "oversized_array", "array_rank", "array_scalar",
    "array_dtype", "array_boolean", "array_object", "array_negative", "array_uint64_overflow",
)


def _saved_kv(runtime):
    return runtime.position, runtime._snapshot_basis, [
        (cache.length, None if cache.key is None else cache.key.copy(),
         None if cache.value is None else cache.value.copy()) for cache in runtime.caches
    ]


def _assert_saved_kv(runtime, saved):
    position, basis, rows = saved
    assert runtime.position == position and runtime._snapshot_basis is basis
    for cache, (length, key, value) in zip(runtime.caches, rows, strict=True):
        assert cache.length == length
        for actual, expected in ((cache.key, key), (cache.value, value)):
            if expected is None:
                assert actual is None
            else:
                assert actual.dtype == expected.dtype and actual.shape == expected.shape
                assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("method", ["prepare_ids", "forward_ids"])
@pytest.mark.parametrize("case", _REJECTED_TOKEN_CASES)
def test_ordinary_preflight_rejects_before_array_conversion_or_any_stage_call(numeric_binding, monkeypatch, method, case):
    plan, owner, engine = numeric_binding
    remote = owner.client_linear_executor(engine)
    runtime = owner.runtime(remote)
    runtime.prepare_ids([1, 2])
    document = plan.to_dict()
    limit = (document["prefill"]["query_sequence"] if method == "prepare_ids"
             else document["decode"]["maximum_key_sequence"] - runtime.position)
    payload = _rejected_token_payload(case, limit)
    saved = _saved_kv(runtime)
    stage_calls = []

    def no_conversion(*args, **kwargs):
        raise AssertionError("rejected request reached np.asarray/token storage conversion")

    def no_stage(self, *args, **kwargs):
        stage_calls.append(args)
        raise AssertionError("rejected request reached a linear stage call")

    with monkeypatch.context() as guard:
        guard.setattr(np, "asarray", no_conversion)
        guard.setattr(type(remote), "__call__", no_stage)
        if case in {"oversized_list", "oversized_array", "array_rank", "array_dtype", "array_boolean", "array_object"}:
            guard.setattr(np, "any", no_conversion)
        with pytest.raises(TransformerClientError):
            getattr(runtime, method)(payload)
    assert stage_calls == []
    assert not runtime._continuation_failed
    _assert_saved_kv(runtime, saved)


@pytest.mark.parametrize("case", _REJECTED_TOKEN_CASES)
def test_continuation_preflight_conversion_stage_and_burn_order(continuation_binding, monkeypatch, case):
    from pllm.modeling import DecoderContinuationSchedule
    runtime, remote, contract = continuation_binding
    limit = contract.to_dict()["token_bound"] - runtime.position
    payload = _rejected_token_payload(case, limit)
    saved = _saved_kv(runtime)
    original_admit = DecoderContinuationSchedule.admit
    admissions, stage_calls, closes = [], [], []
    remote.inventory = SimpleNamespace(close=lambda: closes.append("closed"))

    def admit(self, *args, **kwargs):
        admissions.append(args)
        return original_admit(self, *args, **kwargs)

    def no_conversion(*args, **kwargs):
        raise AssertionError("rejected continuation reached token storage conversion/comparison")

    def no_stage(self, *args, **kwargs):
        stage_calls.append(args)
        raise AssertionError("rejected continuation reached a stage call")

    with monkeypatch.context() as guard:
        guard.setattr(DecoderContinuationSchedule, "admit", admit)
        guard.setattr(np, "asarray", no_conversion)
        guard.setattr(type(remote), "__call__", no_stage)
        if case in {"oversized_list", "oversized_array", "array_rank", "array_dtype", "array_boolean", "array_object"}:
            guard.setattr(np, "any", no_conversion)
        with pytest.raises((TransformerClientError, ValueError)):
            runtime.continue_ids(payload)
        expected_admissions = 1 if case in {"oversized_list", "oversized_array"} else 0
        assert len(admissions) == expected_admissions
        # Burned refusal precedes even inspecting a custom container, and does
        # not admit again or close/cancel optional one-use ownership twice.
        with pytest.raises(TransformerClientError, match="burned"):
            runtime.continue_ids(_UnwalkableSequence())
        assert len(admissions) == expected_admissions
    assert runtime._continuation_failed and closes == ["closed"] and stage_calls == []
    _assert_saved_kv(runtime, saved)


@pytest.mark.parametrize("method", ["prepare_ids", "forward_ids"])
def test_cached_graph_extent_tamper_cannot_widen_native_request_geometry(numeric_binding, monkeypatch, method):
    plan, owner, engine = numeric_binding
    remote = owner.client_linear_executor(engine)
    runtime = owner.runtime(remote)
    runtime.prepare_ids([1, 2])
    source = plan.to_dict()
    limit = (source["prefill"]["query_sequence"] if method == "prepare_ids"
             else source["decode"]["maximum_key_sequence"] - runtime.position)
    for phase in ("prefill", "decode"):
        runtime._graphs[phase]["query_sequence"] = 1 << 30
        runtime._graphs[phase]["maximum_key_sequence"] = 1 << 30
    payload = [1] * (limit + 1)
    saved = _saved_kv(runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("cached declaration widened allocation or stage authority")

    with monkeypatch.context() as guard:
        guard.setattr(np, "asarray", forbidden)
        guard.setattr(np, "any", forbidden)
        guard.setattr(type(remote), "__call__", forbidden)
        with pytest.raises(TransformerClientError, match="workload bounds"):
            getattr(runtime, method)(payload)
    _assert_saved_kv(runtime, saved)


def test_cached_continuation_declarations_cannot_widen_native_admission(continuation_binding, monkeypatch):
    from pllm.modeling import DecoderContinuationSchedule
    runtime, remote, contract = continuation_binding
    actual = contract.to_dict()
    payload = [1] * (actual["token_bound"] - runtime.position + 1)
    expanded = copy.deepcopy(actual)
    expanded["token_bound"] = 1 << 30
    expanded["graph"]["query_sequence"] = 1 << 30
    expanded["graph"]["maximum_key_sequence"] = 1 << 30
    for phase in ("prefill", "decode"):
        runtime._graphs[phase]["query_sequence"] = 1 << 30
        runtime._graphs[phase]["maximum_key_sequence"] = 1 << 30
    saved = _saved_kv(runtime)
    original_document = DecoderContinuationSchedule.to_dict
    stage_calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("cached continuation declarations authorized conversion/comparison")

    def no_stage(self, *args, **kwargs):
        stage_calls.append(args)
        raise AssertionError("cached continuation declarations authorized stage execution")

    with monkeypatch.context() as guard:
        guard.setattr(DecoderContinuationSchedule, "to_dict",
                      lambda self: expanded if self is contract else original_document(self))
        guard.setattr(np, "asarray", forbidden)
        guard.setattr(np, "any", forbidden)
        guard.setattr(type(remote), "__call__", no_stage)
        with pytest.raises(ValueError, match="bound"):
            runtime.continue_ids(payload)
    assert runtime._continuation_failed and stage_calls == []
    _assert_saved_kv(runtime, saved)


@pytest.mark.parametrize("container", [list, np.asarray])
@pytest.mark.parametrize("cardinalities,token,accepted", [
    ((17,), 12, True),   # Above legacy vocabulary 11, inside declared input table.
    ((5,), 7, False),    # Below output-head width, outside the input table.
    ((17, 13), 12, True),
    ((17, 5), 7, False), # The second declared input table must also be checked.
    ((5, 17), 7, False), # Not merely the last token-domain declaration.
])
def test_declared_lookup_domain_helpers_are_independent_of_legacy_vocabulary(
    monkeypatch, container, cardinalities, token, accepted,
):
    # This helper matrix does not establish native unequal-vocabulary continuation
    # or production iteration over multiple token tables.
    operations = []
    for index, cardinality in enumerate(cardinalities):
        operation = _lookup("tokens", cardinality)
        operation["id"] = f"tokens.{index}"
        operation["attributes"]["weight"] = f"tokens.{index}.table"
        operations.append(operation)
    actual = [lookup_contract(op, legacy_vocabulary=11)[2][0] for op in operations]
    assert actual == list(cardinalities)
    assert all(token < cardinality for cardinality in actual) is accepted
    payload = container([token])

    def no_conversion(*args, **kwargs):
        raise AssertionError("input-domain scalar check converted token storage")

    with monkeypatch.context() as guard:
        guard.setattr(np, "asarray", no_conversion)
        for cardinality in actual:
            if token < cardinality:
                SemanticDecoderRuntime._check_integer_token_values(payload, upper=cardinality - 1)
            else:
                with pytest.raises(TransformerClientError, match="integer domain"):
                    SemanticDecoderRuntime._check_integer_token_values(payload, upper=cardinality - 1)


def test_real_owner_continuation_ignores_smaller_cached_output_vocabulary(continuation_binding):
    runtime, remote, contract = continuation_binding
    head = next(op for op in json.loads(contract.canonical_bytes())["graph"]["operations"]
                if op["operator"] == "output_head")
    runtime.cfg["vocab_size"] = 2
    before = len(remote.calls)
    result = runtime.continue_ids([5])
    assert result.shape[-1] == head["output_shape"][-1]
    assert len(remote.calls) > before and runtime.position == 3
    assert not runtime._continuation_failed


@pytest.mark.parametrize("container", [list, np.asarray])
def test_real_owner_input_domain_refuses_before_conversion_admission_or_stage(
    continuation_binding, monkeypatch, container,
):
    runtime, remote, contract = continuation_binding
    graph = json.loads(contract.canonical_bytes())["graph"]
    cardinality = lookup_contract(next(op for op in graph["operations"]
                                     if op["operator"] == "token_lookup"
                                     and op["inputs"] == ["input.tokens"]))[2][0]
    runtime.cfg["vocab_size"] = cardinality + 10
    payload = container([cardinality])
    saved = _saved_kv(runtime)
    admissions, stages, closes = [], [], []
    remote.inventory = SimpleNamespace(close=lambda: closes.append("closed"))

    def no_conversion(*args, **kwargs):
        raise AssertionError("invalid declared-domain token reached conversion")

    def no_admission(self, *args, **kwargs):
        admissions.append(args)
        raise AssertionError("invalid declared-domain token reached native admission")

    def no_stage(self, *args, **kwargs):
        stages.append(args)
        raise AssertionError("invalid declared-domain token reached a stage call")

    with monkeypatch.context() as guard:
        guard.setattr(np, "asarray", no_conversion)
        guard.setattr(type(contract), "admit", no_admission)
        guard.setattr(type(remote), "__call__", no_stage)
        with pytest.raises(TransformerClientError, match="integer domain"):
            runtime.continue_ids(payload)
        with pytest.raises(TransformerClientError, match="burned"):
            runtime.continue_ids(payload)
    assert runtime._continuation_failed and closes == ["closed"]
    assert admissions == [] and stages == []
    _assert_saved_kv(runtime, saved)


def _norm_attrs(*, bias=True, width=3):
    attrs = {
        "weight": "norm.gamma", "weight_shape": [width], "epsilon": "0.00001",
        "axis": -1, "variance": "centered_mean_square",
        "epsilon_placement": "inside_sqrt", "affine_order": "normalize_multiply_add",
        "compute_dtype": "float32",
    }
    if bias:
        attrs.update(bias="norm.beta", bias_shape=[width])
    return attrs


def _lookup(domain="absolute_positions", cardinality=11):
    source = {"tokens": "input.tokens", "absolute_positions": "input.positions", "segments": "input.segments"}[domain]
    attrs = {"weight": domain + ".table", "weight_shape": [cardinality, 3],
             "index_domain": domain, "compute_dtype": "float32"}
    if domain == "absolute_positions":
        attrs.update(position_policy="sequential_absolute", out_of_bounds="reject")
    return {"id": domain, "operator": "token_lookup", "inputs": [source],
            "output_shape": [1, 2, 3], "attributes": attrs}


def _plan():
    token, position = _lookup("tokens", 7), _lookup()
    norm = {"id": "norm", "operator": "layer_norm", "inputs": [token["id"]],
            "output_shape": [1, 2, 3], "attributes": _norm_attrs()}
    linear = {"id": "projection", "operator": "linear", "inputs": ["norm"],
              "output_shape": [1, 2, 5], "attributes": {
                  "weight": "projection.weight", "weight_shape": [3, 5],
                  "weight_layout": "input_output", "bias": "projection.bias", "bias_shape": [5]}}
    head = {"id": "head", "operator": "output_head", "inputs": ["projection"],
            "output_shape": [1, 7], "attributes": {"weight": "head.weight", "weight_shape": [7, 5]}}
    operations = [token, position, norm, linear, head]
    graph = {"operations": operations}
    steps = []
    for order, operation in enumerate(operations):
        local = operation is position or operation is norm
        steps.append({
            "order": order, "layer": None, "operation_ids": [operation["id"]],
            "operators": [operation["operator"]], "input_ids": operation["inputs"],
            "weight_ids": [] if local else [operation["attributes"]["weight"]],
            "executor": "client_local" if local else "remote_stage",
            "outputs": [{"operation_id": operation["id"], "stage_offset": 0,
                         "stage_width": operation["output_shape"][-1]}],
        })
    document = {"prefill": graph, "decode": copy.deepcopy(graph)}
    schedule = SimpleNamespace(complete=True, protected_execution=False,
                               to_dict=lambda: {"prefill": {"steps": steps}})
    return SimpleNamespace(to_dict=lambda: document, prefill=graph,
                           runtime_schedule=lambda composition: schedule)


def test_checkpoint_requirements_include_domain_tables_gamma_beta_and_io_layout():
    plan = _plan()
    assert required_client_tensors(plan) == {
        "absolute_positions.table": (11, 3), "norm.gamma": (3,), "norm.beta": (3,),
    }
    assert required_checkpoint_tensors(plan) == {
        "tokens.table": (7, 3), "absolute_positions.table": (11, 3),
        "norm.gamma": (3,), "norm.beta": (3,), "projection.weight": (3, 5),
        "projection.bias": (5,), "head.weight": (7, 5),
    }
    stages = scheduled_stage_specs(plan, object())
    assert [stage.role for stage in stages] == ["token_lookup", "semantic_linear", "lm_head"]
    assert stages[0].in_features == 7
    assert stages[1].transpose_weight is True
    assert stages[1].bias_keys == ("projection.bias",)
    assert stages[2].transpose_weight is False


def test_actual_absolute_plan_requires_sixteen_unique_weights_without_wire_changes():
    from pllm.modeling import lower_model

    plan = lower_model({
        "model_type": "gpt2", "n_embd": 4, "n_head": 2, "n_layer": 1,
        "n_positions": 7, "vocab_size": 11, "n_inner": 9,
    }, batch=1, max_input_tokens=2, max_new_tokens=2)
    original = plan.canonical_bytes()
    assert required_checkpoint_tensors(plan) == {
        "transformer.wte.weight": (11, 4), "transformer.wpe.weight": (7, 4),
        "transformer.h.0.ln_1.weight": (4,), "transformer.h.0.ln_1.bias": (4,),
        "transformer.h.0.ln_2.weight": (4,), "transformer.h.0.ln_2.bias": (4,),
        "transformer.ln_f.weight": (4,), "transformer.ln_f.bias": (4,),
        "transformer.h.0.attn.c_attn.weight": (4, 12), "transformer.h.0.attn.c_attn.bias": (12,),
        "transformer.h.0.attn.c_proj.weight": (4, 4), "transformer.h.0.attn.c_proj.bias": (4,),
        "transformer.h.0.mlp.c_fc.weight": (4, 9), "transformer.h.0.mlp.c_fc.bias": (9,),
        "transformer.h.0.mlp.c_proj.weight": (9, 4), "transformer.h.0.mlp.c_proj.bias": (4,),
    }
    assert len(required_client_tensors(plan)) == 7
    assert plan.canonical_bytes() == original


@pytest.mark.parametrize("damage", [None, "missing_beta", "transposed_position", "integer_gamma"])
def test_normal_safetensor_store_preflight_is_header_only(tmp_path, monkeypatch, damage):
    plan = _plan()
    arrays = {name: np.zeros(shape, dtype=np.float32)
              for name, shape in required_checkpoint_tensors(plan).items()}
    if damage == "missing_beta":
        arrays.pop("norm.beta")
    elif damage == "transposed_position":
        arrays["absolute_positions.table"] = arrays["absolute_positions.table"].T.copy()
    elif damage == "integer_gamma":
        arrays["norm.gamma"] = arrays["norm.gamma"].astype(np.int64)
    save_file(arrays, tmp_path / "model.safetensors")
    store = SafeTensorStore(tmp_path)

    def no_materialization(*args, **kwargs):
        raise AssertionError("preflight must not materialize checkpoint tensors")

    monkeypatch.setattr(store, "get", no_materialization)
    if damage is None:
        assert preflight_semantic_checkpoint(plan, store) == 7
    else:
        with pytest.raises(SemanticTensorError):
            preflight_semantic_checkpoint(plan, store)


@pytest.mark.parametrize("domain,cardinality", [("tokens", 7), ("absolute_positions", 11), ("segments", 2)])
def test_lookup_cardinality_comes_from_declared_domain_not_head(domain, cardinality):
    operation = _lookup(domain, cardinality)
    assert lookup_contract(operation, legacy_vocabulary=999)[2] == (cardinality, 3)


def test_legacy_inference_only_for_actual_token_source():
    operation = _lookup("tokens", 7)
    operation["attributes"].pop("weight_shape")
    operation["attributes"].pop("index_domain")
    assert lookup_contract(operation, legacy_vocabulary=7) == ("tokens", "input.tokens", (7, 3))
    operation["inputs"] = ["input.positions"]
    with pytest.raises(SemanticTensorError, match="ambiguous"):
        lookup_contract(operation, legacy_vocabulary=7)


@pytest.mark.parametrize("declaration", [
    {"index_domain": None}, {"weight_shape": None}, {"index_domain": "tokens"},
    {"index_domain": "tokens", "weight_shape": None},
])
def test_legacy_fallback_requires_both_domain_and_table_shape_absent(declaration):
    operation = _lookup("tokens", 7)
    operation["attributes"].pop("index_domain")
    operation["attributes"].pop("weight_shape")
    operation["attributes"].update(declaration)
    with pytest.raises(SemanticTensorError):
        lookup_contract(operation, legacy_vocabulary=7)


def test_declared_token_table_without_domain_never_uses_head_cardinality():
    operation = _lookup("tokens", 13)
    operation["attributes"].pop("index_domain")
    assert lookup_contract(operation, legacy_vocabulary=7)[2] == (13, 3)


def test_segment_graph_refuses_without_existing_forward_input_support():
    plan = _plan()
    plan.prefill["operations"].append(_lookup("segments", 2))
    with pytest.raises(SemanticTensorError, match="does not supply input.segments"):
        required_client_tensors(plan)
    with pytest.raises(SemanticTensorError, match="does not supply input.segments"):
        required_checkpoint_tensors(plan)


@pytest.mark.parametrize("field,value", [
    ("index_domain", "tokens"), ("index_domain", []), ("source", "input.tokens"),
    ("weight_shape", [11, 2]), ("weight_shape", [True, 3]), ("weight_shape", None),
    ("weight", ""), ("bias", "unexpected"), ("position_policy", "wrap"),
    ("out_of_bounds", "clamp"), ("compute_dtype", "bfloat16"),
    ("weight_layout", "input_output"), ("weight_layout", None),
])
def test_lookup_rejects_ambiguous_malformed_or_unsupported_tables(field, value):
    operation = _lookup()
    operation["attributes"][field] = value
    with pytest.raises(SemanticTensorError):
        lookup_contract(operation, legacy_vocabulary=7)


@pytest.mark.parametrize("ids", [
    np.asarray([-1], dtype=np.int64), np.asarray([2], dtype=np.int64),
    np.asarray([0.5], dtype=np.float32), np.asarray([True]),
    np.asarray([0], dtype=np.int32), np.asarray([[0]], dtype=np.int64),
    np.asarray([], dtype=np.int64), np.asarray([2**63], dtype=np.uint64),
])
def test_lookup_rejects_ids_before_indexing(numeric_binding, ids):
    kernel = _kernel(numeric_binding, "token_lookup")
    if ids.dtype == np.int64 and ids.shape == (1,) and ids[0] == 2:
        ids = np.asarray([kernel.table_shape[0]], dtype=np.int64)
    with pytest.raises(SemanticNumericError):
        float32_embedding_lookup(ids, kernel)


@pytest.mark.parametrize("bad", ["mutable", "readonly_view_of_mutable", "readonly_owning"])
def test_lookup_rejects_unknown_or_mutable_tensor_backing(bad):
    table = np.ones((2, 3), dtype=np.float32)
    if bad == "readonly_view_of_mutable":
        table = table.view()
        table.setflags(write=False)
    elif bad == "readonly_owning":
        table.setflags(write=False)
    with pytest.raises(SemanticNumericError):
        _BoundLocalNumeric._coefficient(table, (2, 3))


def test_centered_population_layer_norm_affine_order_and_constant_rows(numeric_binding):
    kernel = _kernel(numeric_binding)
    source = np.asarray([[1, 3, 8, -2], [17, 17, 17, 17]], dtype=np.float32)
    gamma, beta = kernel.weight, kernel.bias
    centered = source - source.mean(-1, keepdims=True, dtype=np.float32)
    expected = centered / np.sqrt((centered * centered).mean(-1, keepdims=True) + kernel.epsilon)
    expected = expected * gamma
    expected = expected + beta
    result = float32_layer_norm(source, kernel)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[1], beta)
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime._phase_local_numeric = {kernel.operation["id"]: kernel}
    runtime._phase_numeric_rows = 2
    np.testing.assert_array_equal(runtime._local(
        kernel.operation, {kernel.operation["inputs"][0]: source}, {}, {}), expected)


@pytest.mark.parametrize("field,value", [
    ("variance", "uncentered_mean_square"), ("variance", "sample"),
    ("axis", 0), ("axis", True), ("epsilon_placement", "outside_sqrt"),
    ("affine_order", "add_multiply"), ("compute_dtype", "bfloat16"),
    ("output_dtype", "bfloat16"), ("epsilon", "nan"), ("epsilon", "inf"),
    ("output_dtype", None), ("bias", None), ("bias", ""),
    ("epsilon", "0"), ("epsilon", "-1"), ("epsilon", "1e-100"),
    ("epsilon", "1e100"), ("epsilon", 1e-5), ("weight_shape", [True]),
    ("epsilon", " 0.001"), ("epsilon", "1_0"),
    ("bias_shape", [2]), ("weight", ""), ("protected_mode", True),
])
def test_layer_norm_rejects_unsupported_numeric_or_affine_contract(field, value):
    attrs = _norm_attrs()
    attrs[field] = value
    with pytest.raises(SemanticNumericError):
        layer_norm_contract(attrs, 3)
    plan = _plan()
    for phase in ("prefill", "decode"):
        plan.to_dict()[phase]["operations"][2]["attributes"] = attrs
    with pytest.raises(SemanticTensorError):
        required_client_tensors(plan)


@pytest.mark.parametrize("bad", ["source_dtype", "nan", "rank", "width", "rows", "overflow"])
def test_layer_norm_rejects_malformed_operands_and_f32_overflow(numeric_binding, bad):
    kernel = _kernel(numeric_binding)
    source = np.asarray([[1, 3, 8, -2]], dtype=np.float32)
    if bad == "source_dtype":
        source = source.astype(np.float64)
    elif bad == "nan":
        source[0, 0] = np.nan
    elif bad == "rank":
        source = source[None, None]
    elif bad == "width":
        source = source[:, :3]
    elif bad == "rows":
        source = np.ones((kernel.max_rows + 1, kernel.width), dtype=np.float32)
    else:
        source[0] = [np.finfo(np.float32).max, 0, -np.finfo(np.float32).max, 0]
    with pytest.raises(SemanticNumericError):
        float32_layer_norm(source, kernel)


def test_layer_norm_requires_beta_not_implicit_optional_affine():
    attrs = _norm_attrs(bias=False)
    with pytest.raises(SemanticNumericError, match="bias"):
        layer_norm_contract(attrs, 3)


@pytest.mark.parametrize("explicit", [False, True])
def test_layer_norm_output_dtype_allows_only_absent_or_explicit_string_float32(explicit):
    attrs = _norm_attrs()
    if explicit:
        attrs["output_dtype"] = "float32"
    assert layer_norm_contract(attrs, 3) == np.float32(1e-5)


def _norm_contract_representation(attributes, frozen):
    if not frozen:
        return attributes
    return MappingProxyType({key: tuple(value) if type(value) is list else value
                             for key, value in attributes.items()})


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("width", [1, 3])
def test_layer_norm_json_and_frozen_affine_dimensions_are_identical(frozen, width):
    attributes = _norm_attrs(width=width)
    assert layer_norm_contract(_norm_contract_representation(attributes, frozen), width) == np.float32(1e-5)
    for key, value in (("weight_shape", None), ("weight_shape", []),
                       ("weight_shape", [True]), ("weight_shape", [width + 1]),
                       ("weight_shape", _UnwalkableList([width])),
                       ("bias_shape", None), ("bias_shape", [True]),
                       ("bias_shape", [width + 1]), ("bias_shape", _UnwalkableList([width])),
                       ("bias", None), ("output_dtype", None), ("output_dtype", False)):
        malformed = {**attributes, key: value}
        with pytest.raises(SemanticNumericError):
            layer_norm_contract(_norm_contract_representation(malformed, frozen), width)
    for key in ("weight_shape", "bias_shape", "weight", "bias"):
        missing = {name: value for name, value in attributes.items() if name != key}
        with pytest.raises(SemanticNumericError):
            layer_norm_contract(_norm_contract_representation(missing, frozen), width)


def test_layer_norm_rejects_unknown_mapping_and_sequence_types_without_hooks():
    class HookMapping(dict):
        def __iter__(self):
            raise AssertionError("custom norm mapping traversed")

        def get(self, *args):
            raise AssertionError("custom norm mapping accessed")

    class HookTuple(tuple):
        def __iter__(self):
            raise AssertionError("custom norm shape traversed")

        def __len__(self):
            raise AssertionError("custom norm shape measured")

    for attributes in (HookMapping(), _UnwalkableSequence(), _UnwalkableList(), None):
        with pytest.raises(SemanticNumericError):
            layer_norm_contract(attributes, 3)
    for frozen in (False, True):
        for key in ("weight_shape", "bias_shape"):
            attributes = {**_norm_attrs(), key: HookTuple((3,))}
            with pytest.raises(SemanticNumericError):
                layer_norm_contract(_norm_contract_representation(attributes, frozen), 3)


def test_native_frozen_layer_norm_contract_matches_to_dict():
    from pllm import lower_model
    from absolute_position_decoder_oracle import tiny_checkpoint

    config, _ = tiny_checkpoint()
    plan = lower_model(config, batch=1, max_input_tokens=2, max_new_tokens=2)
    for phase in ("prefill", "decode"):
        plain = {op["id"]: op for op in plan.to_dict()[phase]["operations"]}
        for operation in getattr(plan, phase)["operations"]:
            if operation["operator"] != "layer_norm":
                continue
            attributes = operation["attributes"]
            assert type(attributes) is MappingProxyType and type(attributes["weight_shape"]) is tuple
            width = operation["output_shape"][-1]
            assert layer_norm_contract(attributes, width) == layer_norm_contract(plain[operation["id"]]["attributes"], width)


@pytest.mark.parametrize("text,bits", [
    ("1.000000059604644775390625", 0x3f800000),
    ("1.000000059604644775390625000000000000000000000000001", 0x3f800001),
    ("1.000000059604644775390624999999999999999999999999999", 0x3f800000),
    ("1.4012984643248170709237295832899161312802619418765e-45", 1),
    ("3.40282346638528859811704183484516925440e38", 0x7f7fffff),
])
def test_native_parser_exact_f32_adversarial_midpoints_and_boundaries(text, bits):
    from pllm import _native as core
    attrs = _norm_attrs()
    attrs["epsilon"] = text
    expected = np.asarray([bits], dtype=np.uint32).view(np.float32)[0]
    assert np.float32(core.parse_positive_f32(text)) == expected
    assert layer_norm_contract(attrs, 3) == expected


def test_native_parser_subnormal_and_overflow_halfway_cases():
    from pllm import _native as core
    with localcontext() as context:
        context.prec = 400
        half_subnormal = Decimal(2) ** -150
        perturbation = Decimal(2) ** -300
        overflow_midpoint = Decimal(2) ** 128 - Decimal(2) ** 103
        assert np.float32(core.parse_positive_f32(str(half_subnormal + perturbation))).view(np.uint32) == 1
        assert np.float32(core.parse_positive_f32(str(overflow_midpoint - 1))).view(np.uint32) == 0x7f7fffff
        for value in (half_subnormal, half_subnormal - perturbation, overflow_midpoint, overflow_midpoint + 1):
            with pytest.raises(ValueError):
                core.parse_positive_f32(str(value))


def test_native_parser_has_no_python_or_missing_native_fallback(monkeypatch):
    from pllm import _native as core
    monkeypatch.delattr(core, "parse_positive_f32")
    with pytest.raises(AttributeError):
        layer_norm_contract(_norm_attrs(), 3)


def test_layer_norm_workspace_matches_one_output_one_square_row_buffer(numeric_binding, monkeypatch):
    import pllm.runtime.semantic_numeric as numeric
    plan, _, _ = numeric_binding
    kernel = _kernel(numeric_binding)
    source = np.ones((2, kernel.width), dtype=np.float32)
    original = numeric.np.empty
    allocations = []

    def record(shape, *args, **kwargs):
        result = original(shape, *args, **kwargs)
        allocations.append((result.shape, result.nbytes))
        return result

    monkeypatch.setattr(numeric.np, "empty", record)
    float32_layer_norm(source, kernel)
    width = kernel.width
    assert allocations == [((2, width), 8 * width), ((width,), 4 * width)]
    workspace = local_numeric_workspace(plan, "prefill", kernel.operation["id"])
    assert workspace.max_rows == plan.to_dict()["prefill"]["query_sequence"]
    assert workspace.output_bytes == 4 * workspace.max_rows * width
    assert workspace.row_scratch_bytes == 4 * width
    assert workspace.finite_check_bytes == width
    assert workspace.scalar_bytes == 16
    # Shared admission metadata may conservatively price scalar/validation
    # temporaries. Explicit ndarray allocations above are exact logical counts.
    assert workspace.temporary_bytes >= workspace.output_bytes + 5 * width + 16
    from pllm.runtime.semantic_workspace import semantic_local_workspace_bytes
    assert workspace.output_bytes + semantic_local_workspace_bytes(
        kernel.operation, (1, workspace.max_rows, width)) == workspace.temporary_bytes


@pytest.mark.parametrize("damage", ["dtype", "rank", "width", "extent", "nonfinite"])
def test_layer_norm_malformed_geometry_refuses_before_output_or_row_allocation(numeric_binding, monkeypatch, damage):
    import pllm.runtime.semantic_numeric as numeric
    kernel = _kernel(numeric_binding)
    source = np.ones((2, kernel.width), dtype=np.float32)
    if damage == "dtype":
        source = source.astype(np.float64)
    elif damage == "rank":
        source = source[None, None]
    elif damage == "width":
        source = source[:, :-1]
    elif damage == "extent":
        source = np.ones((kernel.max_rows + 1, kernel.width), dtype=np.float32)
    else:
        source[0, 0] = np.nan

    def no_allocation(*args, **kwargs):
        raise AssertionError("invalid geometry allocated normalization output/scratch")

    monkeypatch.setattr(numeric.np, "empty", no_allocation)
    with pytest.raises(SemanticNumericError):
        float32_layer_norm(source, kernel)


def test_request_row_geometry_and_raw_local_calls_refuse_without_allocation(numeric_binding, monkeypatch):
    import pllm.runtime.semantic_numeric as numeric
    kernel = _kernel(numeric_binding)
    source = np.ones((2, kernel.width), dtype=np.float32)
    runtime = object.__new__(SemanticDecoderRuntime)

    def no_allocation(*args, **kwargs):
        raise AssertionError("unadmitted local kernel allocated output/scratch")

    monkeypatch.setattr(numeric.np, "empty", no_allocation)
    values = {kernel.operation["inputs"][0]: source}
    with pytest.raises(TransformerClientError, match="active compiled"):
        runtime._local(kernel.operation, values, {}, {})
    runtime._phase_local_numeric = {kernel.operation["id"]: kernel}
    runtime._phase_numeric_rows = 1
    with pytest.raises(TransformerClientError, match="request rows"):
        runtime._local(kernel.operation, values, {}, {})


def test_repeated_lookup_uses_admitted_snapshot_without_table_rescan(numeric_binding, monkeypatch):
    import pllm.runtime.semantic_numeric as numeric
    kernel = _kernel(numeric_binding, "token_lookup")
    ids = np.asarray([0, 1], dtype=np.int64)
    expected = kernel.weight[ids].copy()
    original = numeric.np.isfinite

    def no_table_scan(value, *args, **kwargs):
        if isinstance(value, np.ndarray) and np.shares_memory(value, kernel.weight):
            raise AssertionError("lookup rescanned its already admitted immutable table")
        return original(value, *args, **kwargs)

    monkeypatch.setattr(numeric.np, "isfinite", no_table_scan)
    np.testing.assert_array_equal(float32_embedding_lookup(ids, kernel), expected)
    np.testing.assert_array_equal(float32_embedding_lookup(ids, kernel), expected)


def test_local_weight_retains_exact_immutable_snapshot_despite_bundle_replacement(numeric_binding):
    _, owner, engine = numeric_binding
    runtime = owner.runtime(owner.client_linear_executor(engine))
    kernel = _kernel(numeric_binding, "token_lookup")
    name = kernel.operation["attributes"]["weight"]
    expected = owner.local_tensor(name).copy()
    first, second = owner.local_tensor(name), owner.local_tensor(name)
    assert first is not second and np.shares_memory(first, second)
    backing = first
    while isinstance(backing, np.ndarray):
        assert not backing.flags.writeable
        backing = backing.base
    assert type(backing) is bytes
    key = runtime._tensors[name]
    owner._bundle.arrays[key] = np.full(first.shape, np.nan, dtype=np.float32)
    np.testing.assert_array_equal(runtime._weight(name), expected)
    np.testing.assert_array_equal(float32_embedding_lookup(np.asarray([0, 1], dtype=np.int64), kernel), expected[:2])


def test_raw_local_tensors_cannot_construct_runtime_without_private_compiled_owner(numeric_binding):
    plan, owner, _ = numeric_binding
    with pytest.raises(TransformerClientError, match="private compiled runtime owner"):
        SemanticDecoderRuntime(owner._bundle, lambda *args: None, plan=plan,
                               schedule=owner._validated_schedule(), stages={},
                               tensors={"unadmitted.gamma": "array"})


def test_prefill_decode_cache_reuse_does_not_reparse_local_epsilon(numeric_binding, monkeypatch):
    from pllm import _native as core
    _, owner, engine = numeric_binding
    runtime = owner.runtime(owner.client_linear_executor(engine))
    restored = owner.runtime(owner.client_linear_executor(engine))

    def no_reparse(*args, **kwargs):
        raise AssertionError("epsilon parsed inside token/row execution")

    monkeypatch.setattr(core, "parse_positive_f32", no_reparse)
    runtime.prepare_ids([1, 2])
    snapshot = runtime.snapshot()
    expected = runtime.forward_ids([3])
    restored.restore(snapshot)
    np.testing.assert_array_equal(restored.forward_ids([3]), expected)


@pytest.mark.parametrize("method", ["prepare_ids", "forward_ids"])
def test_oversized_request_rejected_before_remote_effects_and_numeric_allocation(numeric_binding, monkeypatch, method):
    import pllm.runtime.semantic_numeric as numeric
    plan, owner, engine = numeric_binding
    remote = owner.client_linear_executor(engine)
    runtime = owner.runtime(remote)
    ids = np.ones(plan.to_dict()["prefill"]["query_sequence"] + 1, dtype=np.int64)

    def no_allocation(*args, **kwargs):
        raise AssertionError("oversized request allocated local numeric output/scratch")

    monkeypatch.setattr(numeric.np, "empty", no_allocation)
    with pytest.raises(TransformerClientError, match="workload bounds"):
        getattr(runtime, method)(ids.tolist() if method == "prepare_ids" else ids)
    assert remote.integer_macs == 0
    assert runtime.position == 0


def test_late_bad_token_rejects_whole_request_before_first_remote_effect(numeric_binding):
    plan, owner, engine = numeric_binding
    remote = owner.client_linear_executor(engine)
    runtime = owner.runtime(remote)
    token = next(op for op in plan.to_dict()["prefill"]["operations"]
                 if op["operator"] == "token_lookup" and op["inputs"] == ["input.tokens"])
    with pytest.raises(TransformerClientError, match="lookup domain"):
        runtime.forward_ids([1, token["attributes"]["weight_shape"][0]])
    assert remote.integer_macs == 0
    assert runtime.position == 0


@pytest.mark.parametrize("layout,shape,transpose", [
    ("input_output", [3, 5], True), ("output_input", [5, 3], False),
])
def test_projection_stage_respects_declared_io_or_oi(layout, shape, transpose):
    plan = _plan()
    operation = plan.prefill["operations"][3]
    operation["attributes"].update(weight_layout=layout, weight_shape=shape)
    assert projection_weight_shape(operation, 3) == tuple(shape)
    assert scheduled_stage_specs(plan, object())[1].transpose_weight is transpose
    operation["attributes"]["weight_shape"] = list(reversed(shape))
    with pytest.raises(SemanticTensorError, match="layout"):
        projection_weight_shape(operation, 3)


def test_protected_schedule_refuses_no_clear_fallback():
    plan = _plan()
    schedule = plan.runtime_schedule(None)
    schedule.protected_execution = True
    with pytest.raises(ValueError, match="public baseline"):
        scheduled_stage_specs(plan, object())


def test_float32_gelu_has_no_bfloat16_rounding_and_handles_finite_extremes(numeric_binding):
    kernel = _kernel(numeric_binding, "gelu_tanh")
    source = np.linspace(-4, 4, 2 * kernel.width, dtype=np.float32).reshape(2, kernel.width)
    expected = (np.float32(0.5) * source) * (np.float32(1) + np.tanh(
        np.float32(np.sqrt(2 / np.pi)) * (source + np.float32(0.044715) * source * source * source)))
    np.testing.assert_array_equal(float32_gelu_tanh(source, kernel), expected)
    extremes = np.zeros((1, kernel.width), dtype=np.float32)
    extremes[0, :2] = [-np.finfo(np.float32).max, np.finfo(np.float32).max]
    assert np.all(np.isfinite(float32_gelu_tanh(extremes, kernel)))
    extremes[0, 0] = np.nan
    with pytest.raises(SemanticNumericError):
        float32_gelu_tanh(extremes, kernel)


def test_visible_prefix_causal_policy_uses_absolute_positions_and_valid_lengths():
    attrs = {"padding_mask_input": "input.attention_mask", "valid_lengths_input": "input.sequence_lengths",
             "cache_validity": "visible_valid_prefix", "masked_value": "negative_infinity"}
    scores = np.zeros((1, 1, 2, 4), dtype=np.float32)
    positions = np.asarray([2, 3], dtype=np.int64)
    mask, lengths = np.ones((1, 2), dtype=np.bool_), np.asarray([4], dtype=np.int64)
    result = mask_causal_scores(scores, positions, attrs, padding_mask=mask, valid_lengths=lengths)
    assert np.isneginf(result[0, 0, 0, 3])
    np.testing.assert_array_equal(result[0, 0, 1], np.zeros(4, dtype=np.float32))
    with pytest.raises(SemanticAttentionError):
        mask_causal_scores(scores, positions, attrs, padding_mask=mask, valid_lengths=lengths - 1)


def test_existing_rms_rotary_and_legacy_token_semantics_unchanged():
    source = np.asarray([[1, -2, 3]], dtype=np.float32)
    gamma = np.asarray([1, 2, 0.5], dtype=np.float32)
    expected = round_bfloat16(round_bfloat16(source) / np.sqrt(
        np.mean(round_bfloat16(source) ** 2, axis=-1, keepdims=True) + np.float32(1e-5)) * gamma)
    np.testing.assert_array_equal(bfloat16_rms_norm(source, epsilon="0.00001", weight=gamma, offset=0), expected)
    rotary = np.asarray([[[[1, 2, 3, 4]]]], dtype=np.float32)
    np.testing.assert_array_equal(SemanticDecoderRuntime._rotary(
        rotary, np.asarray([0], dtype=np.int64), {"rotary_dimensions": 4, "theta": 10000}), rotary)
    token = _lookup("tokens", 7)
    token["attributes"].pop("index_domain")
    token["attributes"].pop("weight_shape")
    assert lookup_contract(token, legacy_vocabulary=7)[2] == (7, 3)


@pytest.mark.parametrize("ids", [[0.5], [True], [True, 1], [[1]], [-1]])
def test_forward_ids_does_not_silently_coerce_malformed_ids(numeric_binding, ids):
    _, owner, engine = numeric_binding
    runtime = owner.runtime(owner.client_linear_executor(engine))
    with pytest.raises(TransformerClientError):
        runtime.forward_ids(ids)


@pytest.mark.parametrize("ids", [[0.5], [True], [True, 1], [[1]], [-1]])
def test_prepare_ids_rejects_malformed_ids_before_reset_or_input_coercion(numeric_binding, ids):
    _, owner, engine = numeric_binding
    runtime = owner.runtime(owner.client_linear_executor(engine))
    with pytest.raises(TransformerClientError):
        runtime.prepare_ids(ids)


def _head_split_operation(plan, phase):
    return copy.deepcopy(next(
        op for op in plan.to_dict()[phase]["operations"]
        if op["operator"] == "reshape"
        and op["attributes"].get("layout") == "batch_heads_sequence_feature"
    ))


@pytest.mark.parametrize("phase,extent", [
    ("prefill", "short"), ("prefill", "full"), ("decode", "one"),
])
@pytest.mark.parametrize("form", ["explicit_rows", "explicit_logical", "legacy_qwen", "permutation"])
@pytest.mark.parametrize("strided", [False, True])
def test_shared_head_split_preserves_exact_token_head_feature_coordinates(
    numeric_binding, phase, extent, form, strided,
):
    plan, owner, engine = numeric_binding
    runtime = owner.runtime(owner.client_linear_executor(engine))
    operation = _head_split_operation(plan, phase)
    batch, heads, bound, width = operation["output_shape"]
    assert batch == 1
    rows = {"short": 2, "full": bound, "one": 1}[extent]
    assert 0 < rows <= bound
    # Unequal signed, non-BF16-rounded values expose wrong axis order or casts.
    columns = heads * width
    backing = (np.arange(rows * columns * 2, dtype=np.float64).reshape(rows, columns * 2)
               / 97.0 - 0.37)
    source_rows = backing[:, ::2] if strided else backing[:, :columns].copy()
    if strided:
        assert not source_rows.flags.c_contiguous
    expected = np.empty((1, heads, rows, width), dtype=source_rows.dtype)
    for token in range(rows):
        for head in range(heads):
            for feature in range(width):
                expected[0, head, token, feature] = source_rows[token, head * width + feature]
    if form == "permutation":
        operation["attributes"] = {
            "layout": "batch_heads_sequence_feature", "input_layout": "batch_sequence_heads_feature",
        }
        source = source_rows.reshape(1, rows, heads, width)
    elif form == "legacy_qwen":
        operation["attributes"] = {"layout": "batch_heads_sequence_feature"}
        source = source_rows
    else:
        assert operation["attributes"] == {
            "layout": "batch_heads_sequence_feature", "input_layout": "batch_sequence_hidden",
        }
        source = source_rows[None] if form == "explicit_logical" else source_rows
    before = source.copy()
    result = runtime._local(operation, {operation["inputs"][0]: source}, {}, {})
    assert result.shape == (1, heads, rows, width) and result.dtype == source.dtype
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(source, before)
    # Existing inverse flatten and sequence-head reshape remain separate paths.
    flatten = {
        "id": "head_split.inverse", "operator": "reshape", "inputs": [operation["id"]],
        "attributes": {"layout": "batch_sequence_hidden", "input_layout": "batch_heads_sequence_feature"},
        "output_shape": [1, bound, columns],
    }
    np.testing.assert_array_equal(runtime._local(flatten, {operation["id"]: result}, {}, {}), source_rows)
    sequence = {
        "id": "head_split.sequence", "operator": "reshape", "inputs": ["rows"],
        "attributes": {"layout": "batch_sequence_heads_feature"},
        "output_shape": [1, bound, heads, width],
    }
    split = runtime._local(sequence, {"rows": source_rows}, {}, {})
    np.testing.assert_array_equal(split, expected.transpose(0, 2, 1, 3))


@pytest.mark.parametrize("form", ["explicit", "legacy_qwen", "permutation"])
def test_malformed_direct_head_split_preserves_state_and_dispatch_count(numeric_binding, monkeypatch, form):
    plan, owner, engine = numeric_binding
    remote = owner.client_linear_executor(engine)
    runtime = owner.runtime(remote)
    original_call = type(remote).__call__
    dispatches = []

    def count_call(self, *args, **kwargs):
        dispatches.append(args)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(type(remote), "__call__", count_call)
    # Populate real caches first. No claim of whole-phase rollback: projections
    # already dispatch before a later head split can reject malformed geometry.
    runtime.prepare_ids([1, 2])
    saved = _saved_kv(runtime)
    dispatch_count = len(dispatches)
    assert dispatch_count > 0
    operation = _head_split_operation(plan, "prefill")
    _, heads, bound, width = operation["output_shape"]
    columns = heads * width
    source_rows = np.arange(2 * columns, dtype=np.float32).reshape(2, columns) / 97
    operation["attributes"] = {"layout": "batch_heads_sequence_feature"}
    if form != "legacy_qwen":
        operation["attributes"]["input_layout"] = (
            "batch_sequence_heads_feature" if form == "permutation" else "batch_sequence_hidden"
        )
    source = source_rows.reshape(1, 2, heads, width) if form == "permutation" else source_rows
    cases = []
    for shape in (None, [], [1, heads, bound], [1, heads, bound, width, 1], [2, heads, bound, width]):
        cases.append(({**operation, "output_shape": shape}, source))
    for axis in range(4):
        for dimension in (0, -1, True, False, 1.0, np.int64(1), None, "1"):
            shape = list(operation["output_shape"])
            shape[axis] = dimension
            cases.append(({**operation, "output_shape": shape}, source))
    for attributes in (
        {**operation["attributes"], "unknown": True},
        {"layout": "batch_heads_sequence_feature", "input_layout": None},
        {"layout": "batch_heads_sequence_feature", "input_layout": "unknown"},
        {"layout": "batch_heads_sequence_feature", "input_layout": []},
    ):
        cases.append(({**operation, "attributes": attributes}, source))
    for inputs in ([], [operation["inputs"][0], "extra"]):
        cases.append(({**operation, "inputs": inputs}, source))
    if form == "permutation":
        bad_shapes = [
            (), (2, columns), (1, 2, columns), (2, 1, heads, width),
            (1, 0, heads, width), (1, bound + 1, heads, width),
            (1, 2, heads + 1, width), (1, 2, heads, width + 1),
            (1, 1, heads * 2, width), (1, 2, heads * width, 1),
        ]
    else:
        bad_shapes = [
            (), (2 * columns,), (0, columns), (bound + 1, columns),
            (2, columns - 1), (1, 2 * columns), (4, columns // 2),
            (2, 1, columns), (1, 0, columns), (1, bound + 1, columns),
            (1, 1, 2 * columns), (1, 2, heads, width),
        ]
        if form == "legacy_qwen":
            bad_shapes.append((1, 2, columns))
    cases.extend((operation, np.zeros(shape, dtype=np.float32)) for shape in bad_shapes)
    pending_tensor = np.asarray([17.25], dtype=np.float32)
    pending = {0: ("pending.key", pending_tensor)}
    for malformed, operand in cases:
        values = {operation["inputs"][0]: operand}
        before = operand.copy()
        shape, strides, dtype = operand.shape, operand.strides, operand.dtype
        with pytest.raises(TransformerClientError):
            runtime._local(malformed, values, {}, pending)
        assert operand.shape == shape and operand.strides == strides and operand.dtype == dtype
        assert operand.tobytes() == before.tobytes()
        assert len(values) == 1 and values[operation["inputs"][0]] is operand
        assert len(pending) == 1 and pending[0][0] == "pending.key" and pending[0][1] is pending_tensor
        np.testing.assert_array_equal(pending_tensor, [17.25])
        assert len(dispatches) == dispatch_count
        _assert_saved_kv(runtime, saved)
