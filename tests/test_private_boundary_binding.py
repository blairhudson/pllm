"""Source-level admission and routing tests; no provider or wheel build needed."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pllm.modeling import lower_model
from pllm.runtime.model_binding import (
    RuntimeBindingError,
    CompiledRuntimeModel,
    _validate_private_boundary_contract,
    _validate_private_boundary_stage,
    _private_boundary_source_identity,
)
from pllm.runtime.semantic_executor import SemanticDecoderRuntime
from pllm.runtime.transformer_client import TransformerClientError
from pllm.runtime.transformer_client import validate_private_boundary_bundle


@pytest.fixture(params=["qwen2", "qwen3", "llama"])
def boundary_plan(request):
    plan = lower_model({"model_type": request.param, "hidden_size": 8,
        "intermediate_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 32, "max_position_embeddings": 32,
        "attention_bias": request.param == "qwen2", "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "attention_dropout": 0.0, "use_sliding_window": False, "use_cache": True, "max_window_layers": 2,
        "tie_word_embeddings": True}, batch=1, max_input_tokens=3, max_new_tokens=2)
    document = plan.to_dict()
    phases = {}
    for phase in ("prefill", "decode"):
        ops = document[phase]["operations"]
        lookup = next(op for op in ops if op["operator"] == "token_lookup")
        head = next(op for op in ops if op["operator"] == "output_head")
        phases[phase] = {"lookup_operation": lookup["id"], "head_operation": head["id"],
            "lookup_rows": lookup["output_shape"][1],
            "head_rows": 1 if len(head["output_shape"]) == 2 else head["output_shape"][1]}
    contract = {"schema_version": "pllm.private_boundary.v1", "model_plan_digest": plan.digest,
        "model_config_digest": document["config_digest"], "composition_digest": "c" * 64,
        "weight_id": lookup["attributes"]["weight"], "vocabulary": 32, "hidden_width": 8,
        "owners": ["boundary_a", "boundary_b"], "lookup_protocol": "packed128_xor_dpf_v2",
        "head_protocol": "two_party_additive_full_head_v1", "weight_bits": 8, "activation_bits": 8,
        "head_ring_bits": 24, "head_signed_output_bound": 8 * 127 * 127,
        "embedding_numeric": "tied_per_token_row_f32_v1",
        "head_numeric": "ordinary_symmetric_per_row_integer_then_f32_v1", "lookup_batch_bound": 64,
        "lookup_query_bound": 4, "matrix_bytes": 256, "scale_bytes": 128,
        "lookup_scratch_bytes": 2048, "client_weight_bytes": 0, **phases}
    return plan, document, contract


def validate(plan, document, contract):
    _validate_private_boundary_contract(contract, document, plan_digest=plan.digest,
        composition_digest="c" * 64, vocabulary=32, hidden=8, activation_bits=8)


def test_generated_tied_plan_contract(boundary_plan):
    validate(*boundary_plan)


@pytest.mark.parametrize("field", ["schema_version", "model_plan_digest", "model_config_digest",
    "composition_digest", "weight_id", "vocabulary", "hidden_width", "owners", "lookup_protocol",
    "head_protocol", "weight_bits", "activation_bits", "head_ring_bits", "head_signed_output_bound",
    "embedding_numeric", "head_numeric", "lookup_batch_bound", "lookup_query_bound", "matrix_bytes",
    "scale_bytes", "lookup_scratch_bytes", "client_weight_bytes", "prefill", "decode"])
def test_native_identity_drift_rejected(boundary_plan, field):
    plan, document, contract = boundary_plan
    malformed = copy.deepcopy(contract)
    malformed[field] = None
    with pytest.raises(RuntimeBindingError, match="contract identities"):
        validate(plan, document, malformed)


def test_extra_fields_and_boolean_geometry_rejected(boundary_plan):
    plan, document, contract = boundary_plan
    for malformed in ({**contract, "extra": 0}, {**contract, "client_weight_bytes": False}):
        with pytest.raises(RuntimeBindingError):
            validate(plan, document, malformed)


@pytest.mark.parametrize("field", ["client_weight", "client_weight_scales", "client_aux_weight", "client_aux_scales"])
def test_no_local_matrix_payload_allowed(field):
    stage = SimpleNamespace(client_weight=None, client_weight_scales=None,
                            client_aux_weight=None, client_aux_scales=None)
    _validate_private_boundary_stage(stage)
    setattr(stage, field, np.zeros((1, 1), dtype=np.int8))
    with pytest.raises(RuntimeBindingError, match="client matrices"):
        _validate_private_boundary_stage(stage)


@pytest.mark.parametrize("mutation", ["extra", "digest", "records", "record_bytes", "head_stage", "scales"])
def test_malformed_thin_descriptor_rejected(mutation):
    descriptor = {"schema": "pllm.private_boundary_bundle.v1",
        "lookup": {"digest": "a" * 64, "records": 32, "record_bytes": 12},
        "head_stage": "lm_head", "head_weight_digest": "b" * 64,
        "head_scales_digest": "c" * 64}
    validate_private_boundary_bundle(descriptor, vocabulary=32, hidden=8)
    if mutation == "extra":
        descriptor["client_weight"] = b"secret"
    elif mutation == "head_stage":
        descriptor["head_stage"] = "subset_head"
    elif mutation == "scales":
        descriptor["head_scales_digest"] = "invalid"
    else:
        descriptor["lookup"][mutation] = "not-hex" if mutation == "digest" else 0
    with pytest.raises(TransformerClientError):
        validate_private_boundary_bundle(descriptor, vocabulary=32, hidden=8)


@pytest.mark.parametrize("owner", [None, SimpleNamespace(lookup=lambda ids: ids, head=lambda rows: rows)])
def test_missing_or_callback_boundary_owner_rejected(boundary_plan, owner):
    plan, _, contract = boundary_plan
    with pytest.raises(TransformerClientError, match="compiled response owner"):
        SemanticDecoderRuntime(SimpleNamespace(privacy={"private_boundary": {}}), lambda *args: None,
            plan=plan, schedule={"private_boundary": contract}, stages={}, tensors={}, boundary=owner)


def test_unadmitted_boundary_rejected(boundary_plan):
    plan, _, _ = boundary_plan
    with pytest.raises(TransformerClientError, match="ordinary schedule"):
        SemanticDecoderRuntime(SimpleNamespace(privacy={}), lambda *args: None,
            plan=plan, schedule={}, stages={}, tensors={}, boundary=object())


def test_lookup_and_full_head_preserve_geometry_without_local_fallback(boundary_plan):
    _, document, _ = boundary_plan
    runtime = object.__new__(SemanticDecoderRuntime)
    calls = []
    def lookup(ids):
        calls.append(("lookup", ids.copy()))
        return np.ones((ids.size, 8), dtype=np.float32)
    def head(rows):
        calls.append(("head", rows.copy()))
        return np.repeat(rows.sum(axis=-1, keepdims=True), 32, axis=-1)
    runtime._boundary = SimpleNamespace(token_lookup=lookup, linear=lambda stage_id, rows: head(rows))
    runtime.cfg = {"hidden_size": 8}
    runtime.bundle = SimpleNamespace(stages={"lm_head": SimpleNamespace(in_features=8, out_features=32)})
    runtime.remote = lambda *args: pytest.fail("boundary routed to ordinary remote")
    runtime._numeric_output = lambda operation, result: result
    ids = np.array([2, 1, 2], dtype=np.int64)
    embeddings, auxiliary = runtime._token_lookup(ids)
    assert embeddings.shape == (3, 8) and auxiliary is None
    np.testing.assert_array_equal(calls[0][1], ids)
    for phase in ("prefill", "decode"):
        operation = next(op for op in document[phase]["operations"] if op["operator"] == "output_head")
        op_id = operation["id"]
        runtime._stages = {f"{phase}:{op_id}": "lm_head"}
        # Exercise full-prefill and decode layouts, not vocabulary slicing.
        shape = (1, 3 if phase == "prefill" else 1, 8)
        values = {"hidden": np.arange(np.prod(shape), dtype=np.float32).reshape(shape)}
        runtime._linear_step(phase, {"operation_ids": [op_id], "input_ids": ["hidden"]},
            {op_id: operation}, ids, values, continuation=False)
        assert values[op_id].shape == shape[:-1] + (32,)
        assert values[op_id].dtype == np.float32
        np.testing.assert_array_equal(values[op_id], np.repeat(values["hidden"].sum(-1, keepdims=True), 32, -1))
    assert all(value.shape == (1, 8) and value.dtype == np.float32 for name, value in calls if name == "head")


@pytest.mark.parametrize("result", [np.zeros((2, 8), dtype=np.float32),
    np.zeros((3, 8), dtype=np.float64), np.full((3, 8), np.nan, dtype=np.float32)])
def test_invalid_lookup_results_rejected(result):
    runtime = object.__new__(SemanticDecoderRuntime)
    runtime.cfg = {"hidden_size": 8}
    runtime._boundary = SimpleNamespace(token_lookup=lambda ids: result)
    with pytest.raises(TransformerClientError, match="lookup shape/type"):
        runtime._token_lookup(np.array([1, 2, 3], dtype=np.int64))


def test_semantic_failure_burns_boundary():
    runtime = object.__new__(SemanticDecoderRuntime)
    burned = []
    runtime._boundary = SimpleNamespace(abort=lambda: burned.append(True))
    runtime._tiered = runtime._retained = runtime._attention_tiling = None
    def fail(*args, **kwargs):
        raise TransformerClientError("bad boundary response")
    runtime._forward_phase = fail
    with pytest.raises(TransformerClientError, match="bad boundary response"):
        runtime._forward(np.array([1], dtype=np.int64))
    assert burned == [True]


def locked_manifest():
    from pllm.model_loader import _CHECKPOINT_DOMAIN, _SOURCE_LOCK_DOMAIN, _digest
    files = [{"path": "config.json", "size": 20, "sha256": "a" * 64},
             {"path": "model.safetensors", "size": 100, "sha256": "b" * 64}]
    checkpoint = _digest(_CHECKPOINT_DOMAIN, {"files": files})
    lock = {"schema": "pllm.model_source_lock.v1", "kind": "safetensors", "repo_id": None,
            "revision": None, "commit": None, "files": files, "checkpoint_digest": checkpoint}
    return {"metadata": {"source_lock": lock, "source_lock_digest": _digest(_SOURCE_LOCK_DOMAIN, lock),
                         "checkpoint_digest": checkpoint, "config_only": False}}


def test_private_boundary_uses_locked_source_not_config_digest():
    manifest = locked_manifest()
    source, checkpoint = _private_boundary_source_identity(manifest)
    assert source == manifest["metadata"]["source_lock_digest"]
    assert checkpoint == manifest["metadata"]["checkpoint_digest"]
    assert source != checkpoint


@pytest.mark.parametrize("field", ["source_lock", "source_lock_digest", "checkpoint_digest", "config_only"])
def test_private_boundary_rejects_missing_source_evidence(field):
    manifest = locked_manifest()
    del manifest["metadata"][field]
    with pytest.raises(RuntimeBindingError, match="locked original source"):
        _private_boundary_source_identity(manifest)


@pytest.mark.parametrize("mutation", ["lock", "checkpoint", "file", "config_only"])
def test_private_boundary_rejects_drifted_source_evidence(mutation):
    manifest = locked_manifest()
    metadata = manifest["metadata"]
    if mutation == "lock":
        metadata["source_lock"]["revision"] = "different"
    elif mutation == "checkpoint":
        metadata["checkpoint_digest"] = "c" * 64
    elif mutation == "file":
        metadata["source_lock"]["files"][1]["sha256"] = "c" * 64
    else:
        metadata["config_only"] = True
    with pytest.raises(RuntimeBindingError):
        _private_boundary_source_identity(manifest)


def source_binding(boundary_plan):
    plan, _, contract = boundary_plan
    binding = object.__new__(CompiledRuntimeModel)
    binding._plan = plan
    binding._runtime_schedule_digest = "d" * 64
    binding._bundle = SimpleNamespace(manifest=locked_manifest(), privacy={"private_boundary": {
        "lookup": {"digest": "e" * 64}}}, stages={"lm_head": SimpleNamespace(
            seeded_profile=SimpleNamespace(signed_output_bound=80000))})
    binding._stages = (SimpleNamespace(stage_id="lm_head", weight_digest="a" * 64,
                                      weight_scales_digest="b" * 64),)
    binding._canonical = json.dumps({"private_boundary_contract": contract}).encode()
    return binding, contract


def test_descriptor_construction_uses_worker_derivable_source_fields(boundary_plan, monkeypatch):
    from pllm.runtime import private_boundary_protocol
    binding, contract = source_binding(boundary_plan)
    monkeypatch.setattr(CompiledRuntimeModel, "_validated_schedule", lambda self: {"private_boundary": contract})
    # Protocol owner changes concurrently; inspect construction independent of installed schema.
    monkeypatch.setattr(private_boundary_protocol, "BoundaryDescriptor", lambda **kwargs: SimpleNamespace(**kwargs))
    descriptor = binding.private_boundary_descriptor()
    metadata = binding._bundle.manifest["metadata"]
    assert descriptor.source_digest == metadata["source_lock_digest"]
    assert descriptor.checkpoint_digest == metadata["checkpoint_digest"]
    assert descriptor.model_config_digest == contract["model_config_digest"]
    assert descriptor.composition_digest == contract["composition_digest"]
    assert descriptor.schedule_digest == binding.runtime_schedule_digest
    assert descriptor.head_weight_digest == "a" * 64 and descriptor.head_scales_digest == "b" * 64
    assert descriptor.signed_output_bound == 80000
    assert not hasattr(descriptor, "body_fingerprint") and not hasattr(descriptor, "runtime_config_digest")


def test_fixed_plan_admission_and_full_prefill_diagnostic_gate(boundary_plan, monkeypatch):
    binding, _ = source_binding(boundary_plan)
    monkeypatch.setattr(CompiledRuntimeModel, "private_boundary_descriptor", lambda self: SimpleNamespace(digest="f" * 64))
    admission = binding.private_boundary_admission(2, 1)
    assert admission.plan_digest == binding.model_plan_digest
    assert admission.lookup_rows == 3 and admission.head_calls == 2
    with pytest.raises(RuntimeBindingError, match="workload"):
        binding.private_boundary_admission(4, 1)
    with pytest.raises(RuntimeBindingError, match="workload"):
        binding.private_boundary_admission(2, 2)
    with pytest.raises(RuntimeBindingError, match="explicit native full-head"):
        binding.private_boundary_admission(2, 0, full_prefill_head=True)
