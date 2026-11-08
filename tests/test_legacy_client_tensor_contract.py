"""Legacy bundle/source consistency, not external checkpoint authentication.

All controls use real checkpoint loading and public bundle import. Missing source
norms retain offset-zero raw-ones semantics, including consumed Q/K norms.
Boundary spies use actual producer/admission lookup owners. Lifecycle probes do
not qualify unavailable Metal hardware or claim native-backend execution.
"""

import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import threading

import msgpack
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from pllm import Model
from pllm.model_loader import resolve_model
from pllm.runtime import bundle_artifacts, bundle_document, bundle_storage, client_tensor_contract
from pllm.runtime import loaders, native, transformer_engine
from pllm.runtime.bundle_artifacts import ArtifactError, export_bundle, parse_manifest, reconstruct_bundle
from pllm.runtime.bundle_document import BundleDocument
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.quantization import dequantize_matmul, quantize_activation_per_row
from pllm.runtime.safetensors_store import SafeTensorStore
from pllm.runtime.tiny_gemma import create_tiny_gemma4_checkpoint, create_tiny_llama_checkpoint
from pllm.runtime.transformer_client import (
    ClientBundle,
    MaskedTransformerClientRuntime,
    TransformerClientError,
)
from pllm.runtime.transformer_engine import MaskedTransformerEngine, TransformerEngineError


FINAL = "model.norm.weight"
Q0 = "model.layers.0.self_attn.q_norm.weight"
K0 = "model.layers.0.self_attn.k_norm.weight"
SHARED_K = "model.layers.2.self_attn.k_norm.weight"
INVENTORY_ERROR = "legacy local tensor inventory differs from source presence"
PERMITTED_ERROR = "legacy source presence must be sorted unique permitted names"
VALUES_ERROR = "legacy source norm has invalid shape or values"
BOUND_ERROR = "legacy bundle exceeds metadata/raw byte bound"


def _exact(message):
    return "^" + re.escape(message) + "$"


def _checkpoint(path: Path, *, shared=False):
    root = create_tiny_gemma4_checkpoint(
        path, seed=17, num_hidden_layers=4 if shared else 2, ple_dim=4,
    )
    if shared:
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["layer_types"] = [
            "sliding_attention", "full_attention", "sliding_attention", "full_attention",
        ]
        config["num_kv_shared_layers"] = 2
        config_path.write_text(json.dumps(config), encoding="utf-8")
    return root


def _rewrite(root, mutate):
    tensors = load_file(root / "model.safetensors")
    mutate(tensors)
    save_file(tensors, root / "model.safetensors", metadata={"format": "pt"})


@contextmanager
def _loaded(root, **engine_options):
    # Source mutation must precede both manifest creation and engine loading.
    resolved = resolve_model(Model.path(str(root), model_id="legacy-contract"))
    manifest = resolved.manifest
    assert manifest.metadata["source_lock_digest"] == resolved.source_lock_digest
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8, **engine_options)
    bundle = None
    try:
        asyncio.run(engine.load(manifest))
        document = engine.client_bundle_document(manifest.id)
        payload = document.to_bytes()
        bundle = ClientBundle.unpack(payload)
        raw = msgpack.unpackb(payload, raw=False)
        yield manifest, engine, bundle, raw, document, payload
    finally:
        try:
            if bundle is not None:
                bundle.close()
        finally:
            if manifest.id in engine.models:
                asyncio.run(engine.unload(manifest.id))


@pytest.fixture
def genuine(tmp_path):
    root = _checkpoint(tmp_path / "model")
    with _loaded(root) as loaded:
        yield root, loaded


def _clone(raw):
    return msgpack.unpackb(msgpack.packb(raw, use_bin_type=True), raw=False)


def _rehash(raw):
    encoded = json.dumps(
        raw["config"], sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    raw["manifest"]["metadata"]["runtime_config_digest"] = hashlib.sha256(encoded).hexdigest()


def _reject(raw, *, match):
    _reject_payload(msgpack.packb(raw, use_bin_type=True), match=match)


def _reject_payload(payload, *, match):
    imported = None
    try:
        with pytest.raises(TransformerClientError, match=match) as caught:
            imported = ClientBundle.unpack(payload)
    finally:
        # Cleanup cannot satisfy the expected import exception. Unexpected
        # successful imports still retire their owner after pytest fails.
        if imported is not None:
            imported.close()
    return caught.value


def _reject_source(root, *, match):
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    manifest = None
    try:
        with pytest.raises((ValueError, TransformerEngineError), match=match):
            manifest = load_hf_directory(root, model_id="legacy-contract")
            asyncio.run(engine.load(manifest))
    finally:
        # Only registered models are ours to unload. Production owns cleanup
        # of resources from a failed load that never registered a model.
        if manifest is not None and manifest.id in engine.models:
            asyncio.run(engine.unload(manifest.id))


def _logits(loaded):
    manifest, engine, bundle, _, _, _ = loaded

    # Same real quantized-stage callback as test_transformer_engine; not a new
    # numerical oracle. Existing independent formulas/tolerances stay untouched.
    def qlinear(stage_id, activation):
        stage = bundle.stages[stage_id]
        runtime = engine.models[manifest.id].stages[stage_id]
        qa = quantize_activation_per_row(activation, bits=stage.activation_bits)
        integer = qa.values.astype(np.int32) @ runtime.weight.values.astype(np.int32).T
        return dequantize_matmul(
            integer, qa.scales, runtime.weight.scales,
            output_shape=qa.original_shape[:-1] + (stage.out_features,),
        )

    runtime = MaskedTransformerClientRuntime(bundle, qlinear, token_cache_size=0)
    _, logits, _ = runtime.prepare_ids([2, 7, 11, 23])
    return logits.copy()


def _permitted_names(layers):
    names = {FINAL, "model.per_layer_projection_norm.weight"}
    for index in range(layers):
        prefix = f"model.layers.{index}"
        names.update(f"{prefix}.{suffix}.weight" for suffix in (
            "input_layernorm", "post_attention_layernorm",
            "pre_feedforward_layernorm", "post_feedforward_layernorm",
            "self_attn.q_norm", "self_attn.k_norm", "post_per_layer_input_norm",
        ))
    return names


def test_genuine_contract_captures_original_source_and_header_census(genuine):
    root, (manifest, _, bundle, raw, _, _) = genuine
    source = json.loads((root / "config.json").read_text(encoding="utf-8"))
    contract = raw["config"]["legacy_source_contract"]
    assert type(contract["v"]) is int and contract["v"] == 1
    assert contract["source"] == source
    assert source["model_type"] == "gemma4_text"
    assert source["architectures"] == ["Gemma4ForCausalLM"]
    expected = sorted(set(load_file(root / "model.safetensors")) & _permitted_names(2))
    assert contract["present_local_tensors"] == expected
    assert len(expected) == len(set(expected))
    assert set(raw["local_tensors"]) == set(expected)
    assert not any(name.endswith("q_proj.weight") for name in expected)
    assert raw["v"] == bundle.schema_version == 2
    assert raw["runtime"] == "masked_transformer"
    assert raw["model"] == manifest.id
    before = raw["manifest"]["metadata"]["runtime_config_digest"]
    _rehash(raw)
    assert raw["manifest"]["metadata"]["runtime_config_digest"] == before
    for name in expected:
        row = raw["local_tensors"][name]
        assert row["dtype"] == "f4"
        assert row["shape"] == list(bundle.local_tensors[name].shape)
        assert type(row["data"]) is bytes


@pytest.mark.parametrize("name", [FINAL, Q0, K0])
def test_source_absent_consumed_norm_matches_explicit_raw_ones(tmp_path, name):
    present = _checkpoint(tmp_path / "present")
    with _loaded(present) as loaded:
        assert loaded[3]["config"]["norm_offset"] == 0
        expected = _logits(loaded)
        assert name in loaded[3]["config"]["legacy_source_contract"]["present_local_tensors"]
    absent = _checkpoint(tmp_path / "absent")
    _rewrite(absent, lambda tensors: tensors.pop(name))
    with _loaded(absent) as loaded:
        assert loaded[3]["config"]["norm_offset"] == 0
        assert name not in loaded[2].local_tensors
        assert name not in loaded[3]["config"]["legacy_source_contract"]["present_local_tensors"]
        np.testing.assert_allclose(_logits(loaded), expected, rtol=1e-5, atol=1e-5)


def test_positive_nonuniform_norm_preserved_and_changes_logits(tmp_path):
    control = _checkpoint(tmp_path / "control")
    with _loaded(control) as loaded:
        expected_logits = _logits(loaded)
    changed = _checkpoint(tmp_path / "changed")
    values = torch.linspace(0.25, 2.0, 32)
    _rewrite(changed, lambda tensors: tensors.__setitem__(FINAL, values))
    with _loaded(changed) as loaded:
        np.testing.assert_array_equal(loaded[2].local_tensors[FINAL], values.numpy())
        selected = np.asarray([2, 7, 11, 23])
        assert not np.allclose(
            _logits(loaded)[selected], expected_logits[selected], rtol=1e-5, atol=1e-5,
        )


def test_bfloat16_source_norm_converts_exactly_to_float32(tmp_path):
    control = _checkpoint(tmp_path / "control")
    with _loaded(control):
        pass
    root = _checkpoint(tmp_path / "bfloat16")
    values = torch.linspace(0.3, 1.9, 32).to(torch.bfloat16)
    _rewrite(root, lambda tensors: tensors.__setitem__(FINAL, values))
    with _loaded(root) as loaded:
        actual = loaded[2].local_tensors[FINAL]
        assert actual.dtype == np.dtype("float32")
        np.testing.assert_array_equal(actual, values.float().numpy())


def test_present_norm_payload_cannot_be_deleted(genuine):
    _, loaded = genuine  # Fixture already imported untouched baseline.
    raw = _clone(loaded[3])
    assert FINAL in raw["config"]["legacy_source_contract"]["present_local_tensors"]
    del raw["local_tensors"][FINAL]
    _reject(raw, match=_exact(INVENTORY_ERROR))


def test_absent_source_norm_payload_cannot_be_inserted(tmp_path):
    root = _checkpoint(tmp_path / "absent")
    _rewrite(root, lambda tensors: tensors.pop(FINAL))
    with _loaded(root) as loaded:
        raw = _clone(loaded[3])
        assert FINAL not in raw["config"]["legacy_source_contract"]["present_local_tensors"]
        raw["local_tensors"][FINAL] = {
            "shape": [32], "dtype": "f4", "data": np.ones(32, dtype="<f4").tobytes(),
        }
        _reject(raw, match=_exact(INVENTORY_ERROR))


@pytest.mark.parametrize("mutation", ["duplicate", "unsorted"])
def test_census_requires_sorted_unique_names(genuine, mutation):
    raw = _clone(genuine[1][3])
    census = raw["config"]["legacy_source_contract"]["present_local_tensors"]
    if mutation == "duplicate":
        census.insert(0, census[0])
    else:
        census.reverse()
    _rehash(raw)
    message = "invalid legacy source presence census" if mutation == "duplicate" else PERMITTED_ERROR
    _reject(raw, match=_exact(message))


@pytest.mark.parametrize("version", [True, False, 0, 2, 1.0, "1"])
def test_nested_contract_version_requires_exact_integer_one(genuine, version):
    raw = _clone(genuine[1][3])
    raw["config"]["legacy_source_contract"]["v"] = version
    _rehash(raw)
    _reject(raw, match=_exact("invalid legacy source presence contract"))


def test_missing_contract_rejected_even_with_consistent_digest(genuine):
    raw = _clone(genuine[1][3])
    del raw["config"]["legacy_source_contract"]
    _rehash(raw)
    _reject(raw, match=_exact("legacy source presence contract required; rebuild from original checkpoint"))


def test_wrong_digest_with_unchanged_valid_geometry_rejected(genuine):
    raw = _clone(genuine[1][3])
    original = raw["manifest"]["metadata"]["runtime_config_digest"]
    raw["manifest"]["metadata"]["runtime_config_digest"] = (
        ("1" if original[0] == "0" else "0") + original[1:]
    )
    _reject(raw, match=_exact("legacy runtime config digest mismatch"))


@pytest.mark.parametrize("side", ["runtime", "source"])
def test_rehashed_source_runtime_geometry_disagreement_rejected(genuine, side):
    raw = _clone(genuine[1][3])
    config = raw["config"] if side == "runtime" else raw["config"]["legacy_source_contract"]["source"]
    config["hidden_size"] += 4
    _rehash(raw)
    message = (
        "legacy runtime field 'hidden_size' disagrees with original source" if side == "runtime"
        else "legacy runtime field 'embedding_multiplier' disagrees with source profile"
    )
    _reject(raw, match=_exact(message))


def test_stage_geometry_mismatch_rejected_with_valid_config_digest(genuine):
    raw = _clone(genuine[1][3])
    raw["stages"]["layers.0.self_attn.qkv_proj"]["in_features"] += 1
    _rehash(raw)
    _reject(raw, match=_exact("legacy stage geometry disagrees with source"))


@pytest.mark.parametrize("mutation", ["dtype", "rank", "width", "nbytes", "nan", "inf", "neg_inf"])
def test_wire_norm_descriptor_rejected_after_genuine_import(genuine, mutation):
    raw = _clone(genuine[1][3])
    row = raw["local_tensors"][FINAL]
    if mutation == "dtype":
        row["dtype"] = "f8"  # Wire f4 only; no new source-dtype restriction.
    elif mutation == "rank":
        row["shape"] = [1, 32]
    elif mutation == "width":
        row["shape"] = [31]
        row["data"] = np.ones(31, dtype="<f4").tobytes()
    elif mutation == "nbytes":
        row["data"] = row["data"][:-1]
    else:
        values = np.ones(32, dtype="<f4")
        values[0] = {"nan": np.nan, "inf": np.inf, "neg_inf": -np.inf}[mutation]
        row["data"] = values.tobytes()
    message = (
        f"invalid client tensor storage for {FINAL}"
        if mutation in {"nan", "inf", "neg_inf"}
        else f"legacy local tensor {FINAL!r} differs from source geometry/storage"
    )
    _reject(raw, match=_exact(message))


@pytest.mark.parametrize("mutation", ["prefix_only", "prefix_collision", "scale", "learned_v"])
def test_wire_noncanonical_norm_names_rejected_with_rehashed_census(genuine, mutation):
    raw = _clone(genuine[1][3])
    rows = raw["local_tensors"]
    if mutation == "prefix_only":
        rows["language_model." + FINAL] = rows.pop(FINAL)
    elif mutation == "prefix_collision":
        rows["language_model." + FINAL] = _clone(rows[FINAL])
    elif mutation == "scale":
        rows["model.norm.scale"] = _clone(rows[FINAL])
    else:
        rows["model.layers.0.self_attn.v_norm.weight"] = _clone(rows[Q0])
    if mutation != "prefix_only":
        # Keep census length valid, so rejection tests forbidden identity, not
        # merely an overlong census. Canonical FINAL stays for alias collision.
        del rows["model.per_layer_projection_norm.weight"]
    raw["config"]["legacy_source_contract"]["present_local_tensors"] = sorted(rows)
    _rehash(raw)
    _reject(raw, match=_exact(PERMITTED_ERROR))


@pytest.mark.parametrize("mutation", [
    "prefix_only", "prefix_collision", "scale", "learned_v", "rank", "width", "nan", "inf", "neg_inf",
])
def test_independent_source_norm_mutations_rejected(tmp_path, mutation):
    control = _checkpoint(tmp_path / "control")
    with _loaded(control):
        pass
    root = _checkpoint(tmp_path / "invalid")

    def mutate(tensors):
        if mutation == "prefix_only":
            tensors["language_model." + FINAL] = tensors.pop(FINAL)
        elif mutation == "prefix_collision":
            tensors["language_model." + FINAL] = tensors[FINAL].clone()
        elif mutation == "scale":
            tensors["model.norm.scale"] = tensors[FINAL].clone()
        elif mutation == "learned_v":
            tensors["model.layers.0.self_attn.v_norm.weight"] = torch.ones(8)
        elif mutation == "rank":
            tensors[FINAL] = tensors[FINAL].reshape(1, 32)
        elif mutation == "width":
            tensors[FINAL] = torch.ones(31)
        else:
            tensors[FINAL][0] = {
                "nan": float("nan"), "inf": float("inf"), "neg_inf": -float("inf"),
            }[mutation]

    _rewrite(root, mutate)
    if mutation in {"prefix_only", "prefix_collision"}:
        message = f"unknown or noncanonical legacy source norm {'language_model.' + FINAL!r}"
    elif mutation == "scale":
        message = "unknown or noncanonical legacy source norm 'model.norm.scale'"
    elif mutation == "learned_v":
        message = "unknown or noncanonical legacy source norm 'model.layers.0.self_attn.v_norm.weight'"
    elif mutation in {"rank", "width"}:
        message = f"legacy source norm {FINAL!r} differs from source geometry"
    else:
        message = VALUES_ERROR
    _reject_source(root, match=_exact(message))


@pytest.mark.parametrize("name", [
    *(f"model.layers.{index}.self_attn.q_norm.weight" for index in range(4)),
    "model.layers.0.self_attn.k_norm.weight", "model.layers.1.self_attn.k_norm.weight",
    SHARED_K, "model.layers.3.self_attn.k_norm.weight",
])
def test_shared_kv_source_norms_optional_but_presence_recorded(tmp_path, name):
    control = _checkpoint(tmp_path / "control", shared=True)
    with _loaded(control) as loaded:
        assert loaded[3]["config"]["norm_offset"] == 0
        assert name in loaded[3]["config"]["legacy_source_contract"]["present_local_tensors"]
        expected = _logits(loaded)
    root = _checkpoint(tmp_path / "absent", shared=True)
    _rewrite(root, lambda tensors: tensors.pop(name))
    with _loaded(root) as loaded:
        assert loaded[3]["config"]["norm_offset"] == 0
        assert name not in loaded[2].local_tensors
        assert name not in loaded[3]["config"]["legacy_source_contract"]["present_local_tensors"]
        np.testing.assert_allclose(_logits(loaded), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mutation", ["rank", "width", "nan", "inf", "neg_inf"])
def test_present_unused_shared_k_still_shape_and_finite_checked(tmp_path, mutation):
    control = _checkpoint(tmp_path / "control", shared=True)
    with _loaded(control):
        pass
    root = _checkpoint(tmp_path / "invalid", shared=True)

    def mutate(tensors):
        if mutation == "rank":
            tensors[SHARED_K] = torch.ones(1, 8)
        elif mutation == "width":
            tensors[SHARED_K] = torch.ones(7)
        else:
            tensors[SHARED_K][0] = {
                "nan": float("nan"), "inf": float("inf"), "neg_inf": -float("inf"),
            }[mutation]

    _rewrite(root, mutate)
    message = (
        f"legacy source norm {SHARED_K!r} differs from source geometry"
        if mutation in {"rank", "width"} else VALUES_ERROR
    )
    _reject_source(root, match=_exact(message))


@pytest.mark.parametrize("as_document", [False, True])
def test_genuine_legacy_artifact_roundtrip_imports(genuine, as_document):
    _, (source_manifest, _, _, raw, document, payload) = genuine
    lock = source_manifest.metadata["source_lock_digest"]
    assert source_manifest.metadata["stage_origin"] == "legacy_stage_plan_v1"
    assert source_manifest.metadata["decoder_execution"] == "legacy_runtime_graph_v1"
    assert "semantic_source_config" not in raw["config"]
    exported = export_bundle(document if as_document else payload)
    manifest = parse_manifest(
        exported.manifest, fingerprint=hashlib.sha256(payload).hexdigest(), size=len(payload),
    )
    skeleton = msgpack.unpackb(manifest["skeleton"], raw=False)
    assert skeleton["manifest"]["metadata"]["source_lock_digest"] == lock
    assert manifest["objects"]
    assert all(row["metadata"]["source"]["lock"] == lock for row in manifest["objects"])
    rebuilt = reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]])
    assert rebuilt == payload
    imported = ClientBundle.unpack(rebuilt)
    try:
        assert imported.schema_version == 2
    finally:
        imported.close()


def test_genuine_unlocked_direct_loader_bundle_cannot_export_artifacts(genuine, tmp_path):
    # A resolved, locked artifact control succeeds independently of this direct
    # loader route. No source-lock fields are fabricated or removed.
    export_bundle(genuine[1][4])
    root = _checkpoint(tmp_path / "unlocked-direct")
    manifest = load_hf_directory(root, model_id="unlocked-direct")
    assert "source_lock_digest" not in manifest.metadata
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    imported = None
    try:
        asyncio.run(engine.load(manifest))
        document = engine.client_bundle_document(manifest.id)
        payload = document.to_bytes()
        imported = ClientBundle.unpack(payload)
        assert imported.schema_version == 2
        assert manifest.metadata["stage_origin"] == "legacy_stage_plan_v1"
        assert manifest.metadata["decoder_execution"] == "legacy_runtime_graph_v1"
        assert "source_lock_digest" not in manifest.metadata
        with pytest.raises(ArtifactError, match=_exact("invalid artifact digest")):
            export_bundle(document)
    finally:
        try:
            if imported is not None:
                imported.close()
        finally:
            if manifest.id in engine.models:
                asyncio.run(engine.unload(manifest.id))


def test_immutable_document_change_preserves_presence_rejection(genuine):
    raw = _clone(genuine[1][3])
    del raw["local_tensors"][FINAL]
    changed = BundleDocument(raw)
    _reject_payload(changed.to_bytes(), match=_exact(INVENTORY_ERROR))


def test_raw_messagepack_duplicate_map_key_rejected(genuine):
    raw = genuine[1][3]
    packer = msgpack.Packer(use_bin_type=True)
    # Dict cannot represent duplicate keys. Preserve every genuine field, then
    # repeat one key in the ordinary MessagePack map encoding.
    parts = [packer.pack_map_header(len(raw) + 1)]
    for key, value in raw.items():
        parts.extend((packer.pack(key), packer.pack(value)))
    parts.extend((packer.pack("runtime"), packer.pack(raw["runtime"])))
    error = _reject_payload(b"".join(parts), match=_exact("invalid transformer client bundle"))
    assert type(error.__cause__) is ValueError
    assert str(error.__cause__) == "bundle maps require unique string keys"


def _forbid(message):
    def forbidden(*args, **kwargs):
        pytest.fail(message)
    return forbidden


def _import_and_close(payload):
    imported = ClientBundle.unpack(payload)
    try:
        assert imported.schema_version in {1, 2}
    finally:
        imported.close()


@pytest.mark.parametrize("shape", [(1, 32), (31,)])
def test_source_header_shape_rejected_before_any_coefficient_get(genuine, tmp_path, monkeypatch, shape):
    root = _checkpoint(tmp_path / "wrong-header")
    _rewrite(root, lambda tensors: tensors.__setitem__(FINAL, torch.ones(shape)))
    with monkeypatch.context() as patch:
        patch.setattr(SafeTensorStore, "get", _forbid("header failure read source coefficients"))
        patch.setattr(MaskedTransformerEngine, "_load_stage", _forbid("header failure built stage"))
        _reject_source(root, match=_exact(f"legacy source norm {FINAL!r} differs from source geometry"))


@pytest.mark.parametrize("name", sorted(_permitted_names(4)))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_every_present_norm_finite_after_read_before_stage_ownership(
    tmp_path, monkeypatch, name, value,
):
    control = _checkpoint(tmp_path / "control", shared=True)
    with _loaded(control) as loaded:
        assert name in loaded[2].local_tensors
    root = _checkpoint(tmp_path / "nonfinite", shared=True)
    _rewrite(root, lambda tensors: tensors[name].__setitem__(0, value))
    reads = []
    original = SafeTensorStore.get

    def get(store, key_or_suffix, *, dtype=np.float32):
        result = original(store, key_or_suffix, dtype=dtype)
        reads.append((key_or_suffix, bool(np.all(np.isfinite(result)))))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(SafeTensorStore, "get", get)
        patch.setattr(MaskedTransformerEngine, "_load_stage", _forbid("nonfinite norm built stage"))
        _reject_source(root, match=_exact(VALUES_ERROR))
    assert (name, False) in reads


@pytest.mark.parametrize("mutation", ["digest", "runtime_geometry", "stage_geometry"])
def test_contract_failures_precede_source_stage_builder_and_array_owners(genuine, monkeypatch, mutation):
    raw = _clone(genuine[1][3])
    if mutation == "digest":
        digest = raw["manifest"]["metadata"]["runtime_config_digest"]
        raw["manifest"]["metadata"]["runtime_config_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
        message = "legacy runtime config digest mismatch"
    elif mutation == "runtime_geometry":
        raw["config"]["hidden_size"] += 4
        _rehash(raw)
        message = "legacy runtime field 'hidden_size' disagrees with original source"
    else:
        raw["stages"]["layers.0.self_attn.qkv_proj"]["in_features"] += 1
        _rehash(raw)
        message = "legacy stage geometry disagrees with source"
    with monkeypatch.context() as patch:
        if mutation != "stage_geometry":
            patch.setattr(client_tensor_contract, "gemma4_stage_plan", _forbid("invalid contract reached source builder"))
        patch.setattr(native.MaskedGEMM, "compile", _forbid("invalid contract created native matrix"))
        patch.setattr(bundle_storage, "immutable_f32_array", _forbid("invalid contract created tensor owner"))
        patch.setattr(np, "frombuffer", _forbid("invalid contract materialized numeric array"))
        _reject(raw, match=_exact(message))


def test_layer_373_binary_aggregate_prebound_precedes_builders_and_get(genuine, tmp_path, monkeypatch):
    root = _checkpoint(tmp_path / "too-many-layers")
    path = root / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["num_hidden_layers"] = 373
    path.write_text(json.dumps(config), encoding="utf-8")
    assert 8 + 20 * 373 == 7468 < 16384
    assert 21 + 44 * 373 == 16433 > 16384
    with monkeypatch.context() as patch:
        patch.setattr(loaders, "gemma4_stage_plan", _forbid("aggregate failure expanded source stages"))
        patch.setattr(transformer_engine, "gemma4_stage_plan", _forbid("aggregate failure expanded engine stages"))
        patch.setattr(client_tensor_contract, "gemma4_stage_plan", _forbid("aggregate failure built contract stages"))
        patch.setattr(MaskedTransformerEngine, "_load_stage", _forbid("aggregate failure built owned stage"))
        patch.setattr(SafeTensorStore, "get", _forbid("aggregate failure read coefficients"))
        _reject_source(root, match=_exact("legacy source aggregate object pre-bound exceeded"))


def test_lowered_source_binary_prebound_exact_boundary(genuine, monkeypatch):
    root = genuine[0]
    bound = 21 + 44 * 2
    with monkeypatch.context() as patch:
        patch.setattr(client_tensor_contract, "MAX_OBJECTS", bound)
        assert load_hf_directory(root, model_id="count-control").num_hidden_layers == 2
        patch.setattr(client_tensor_contract, "MAX_OBJECTS", bound - 1)
        patch.setattr(loaders, "gemma4_stage_plan", _forbid("lowered prebound expanded stages"))
        patch.setattr(SafeTensorStore, "get", _forbid("lowered prebound read coefficients"))
        _reject_source(root, match=_exact("legacy source aggregate object pre-bound exceeded"))


def test_actual_record_aggregate_counts_exact_boundary(genuine, monkeypatch):
    raw = genuine[1][3]
    stages, weights, tensors = (len(raw[key]) for key in ("stages", "client_weights", "local_tensors"))
    logical = stages + weights + tensors
    binary = 4 * stages + 2 * weights + tensors + 1
    assert binary > logical
    with monkeypatch.context() as patch:
        patch.setattr(client_tensor_contract, "MAX_OBJECTS", binary)
        client_tensor_contract.legacy_record_counts(stages, weights, tensors)
        patch.setattr(client_tensor_contract, "MAX_OBJECTS", binary - 1)
        with pytest.raises(ValueError, match=_exact("legacy bundle aggregate object bound exceeded")):
            client_tensor_contract.legacy_record_counts(stages, weights, tensors)


@pytest.mark.parametrize("mutation", ["depth", "nodes", "utf8"])
def test_oversized_metadata_precedes_digest_serializer_and_numeric_owners(genuine, monkeypatch, mutation):
    raw = _clone(genuine[1][3])
    if mutation == "depth":
        value = None
        for _ in range(65):
            value = [value]
        message = "bundle document exceeds its node/depth bound"
    elif mutation == "nodes":
        value = [None] * 250001
        message = "bundle document exceeds its node/depth bound"
    else:
        cap = max(
            bundle_document.bounded_metadata(raw)[0],
            bundle_document.bounded_metadata(raw["config"], json_mode=True)[0],
        ) + 100
        value = "é" * (cap + 1)
        message = "bundle metadata exceeds its byte bound"
    raw["config"]["admission_probe"] = value
    payload = msgpack.packb(raw, use_bin_type=True)
    with monkeypatch.context() as patch:
        if mutation == "utf8":
            patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", cap)
            _import_and_close(genuine[1][5])
        # client_tensor_contract.json is the actual digest serializer lookup.
        # All test payload serialization finished before installing this spy.
        patch.setattr(client_tensor_contract.json, "dumps", _forbid("oversized metadata reached digest serializer"))
        patch.setattr(bundle_storage, "immutable_f32_array", _forbid("oversized metadata created tensor owner"))
        patch.setattr(native.MaskedGEMM, "compile", _forbid("oversized metadata created matrix"))
        _reject_payload(payload, match=_exact(message))


def test_metadata_depth_and_node_bound_controls():
    value = None
    for _ in range(64):
        value = [value]
    bundle_document.bounded_metadata(value, json_mode=True)
    with pytest.raises(ValueError, match=_exact("bundle document exceeds its node/depth bound")):
        bundle_document.bounded_metadata([value], json_mode=True)
    bundle_document.bounded_metadata([None] * 249999, json_mode=True)
    with pytest.raises(ValueError, match=_exact("bundle document exceeds its node/depth bound")):
        bundle_document.bounded_metadata([None] * 250000, json_mode=True)


def test_utf8_metadata_limit_exact_boundary(monkeypatch):
    text = "é" * 8
    assert len(text.encode("utf-8")) == 16
    with monkeypatch.context() as patch:
        patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", 18)
        assert bundle_document.bounded_metadata(text, json_mode=True)[0] == 18
        patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", 17)
        with pytest.raises(ValueError, match=_exact("bundle metadata exceeds its byte bound")):
            bundle_document.bounded_metadata(text, json_mode=True)


@pytest.mark.parametrize("payload", [b"", bytearray(), memoryview(b"\x80"), "not bytes", None])
def test_common_unpack_input_gate_precedes_messagepack(genuine, monkeypatch, payload):
    with monkeypatch.context() as patch:
        patch.setattr(bundle_document.msgpack, "unpackb", _forbid("invalid input reached MessagePack"))
        _reject_payload(payload, match=_exact("invalid transformer client bundle"))


def test_common_unpack_explicit_byte_cap_exact_boundary(genuine, monkeypatch):
    payload = genuine[1][5]
    assert bundle_document.unpack_document(payload, maximum=len(payload))["v"] == 2
    with monkeypatch.context() as patch:
        patch.setattr(bundle_document.msgpack, "unpackb", _forbid("oversized input reached MessagePack"))
        with pytest.raises(ValueError, match=_exact("bundle document exceeds its byte bound")):
            bundle_document.unpack_document(payload, maximum=len(payload) - 1)
    # Direct importer uses unpack_document's intentionally captured 8 GiB
    # default. Explicit helper caps do not prove importer-global patchability.


def test_legacy_raw_byte_limit_direct_artifact_parity(genuine, monkeypatch):
    raw, payload = genuine[1][3], genuine[1][5]
    _, counted = bundle_document.bounded_metadata(raw)
    assert counted == len(payload)
    with monkeypatch.context() as patch:
        patch.setattr(bundle_document, "MAX_RAW_BYTES", counted)
        _import_and_close(payload)
        export_bundle(payload)
        patch.setattr(bundle_document, "MAX_RAW_BYTES", counted - 1)
        _reject_payload(payload, match=_exact(BOUND_ERROR))
        with pytest.raises(ArtifactError, match=_exact(BOUND_ERROR)):
            export_bundle(payload)


def test_legacy_metadata_byte_limit_direct_artifact_parity(genuine, monkeypatch):
    raw, payload = genuine[1][3], genuine[1][5]
    metadata, _ = bundle_document.bounded_metadata(raw)
    with monkeypatch.context() as patch:
        # Larger than the runtime-config JSON budget; this is the whole raw
        # document's metadata gate, not the separate digest JSON accounting.
        assert metadata > bundle_document.bounded_metadata(raw["config"], json_mode=True)[0]
        patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", metadata)
        _import_and_close(payload)
        export_bundle(payload)
        patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", metadata - 1)
        _reject_payload(payload, match=_exact(BOUND_ERROR))
        with pytest.raises(ArtifactError, match=_exact(BOUND_ERROR)):
            export_bundle(payload)


def test_artifact_transport_remains_stricter_than_direct_import(genuine, monkeypatch):
    payload = genuine[1][5]
    export_bundle(payload)
    with monkeypatch.context() as patch:
        patch.setattr(bundle_artifacts, "MAX_RAW_BYTES", len(payload))
        export_bundle(payload)
        patch.setattr(bundle_artifacts, "MAX_RAW_BYTES", len(payload) - 1)
        _import_and_close(payload)
        with pytest.raises(ArtifactError, match=_exact("artifact document exceeds its byte bound")):
            export_bundle(payload)


def test_artifact_binary_object_limit_exact_boundary(genuine, monkeypatch):
    payload = genuine[1][5]
    exported = export_bundle(payload)
    count = len(msgpack.unpackb(exported.manifest, raw=False)["objects"])
    raw = genuine[1][3]
    assert count > sum(len(raw[key]) for key in ("stages", "client_weights", "local_tensors"))
    with monkeypatch.context() as patch:
        patch.setattr(bundle_artifacts, "MAX_OBJECTS", count)
        export_bundle(payload)
        patch.setattr(bundle_artifacts, "MAX_OBJECTS", count - 1)
        with pytest.raises(ArtifactError, match=_exact("artifact manifest exceeds bounds")):
            export_bundle(payload)


@pytest.mark.parametrize("private", [False, True])
def test_canonical_route_not_subject_to_legacy_raw_metadata_limits(tmp_path, monkeypatch, private):
    root = create_tiny_llama_checkpoint(tmp_path / "canonical", num_hidden_layers=1)
    with _loaded(root, private_boundary=private) as loaded:
        payload = loaded[5]
        assert not client_tensor_contract.legacy_source_candidate(loaded[3])
        with monkeypatch.context() as patch:
            patch.setattr(bundle_document, "MAX_RAW_BYTES", 1)
            patch.setattr(bundle_document, "MAX_MANIFEST_BYTES", 1)
            patch.setattr(client_tensor_contract, "MAX_OBJECTS", 1)
            _import_and_close(payload)


@pytest.mark.parametrize("mutation", ["census", "runtime_geometry", "stage_geometry"])
def test_malformed_legacy_direct_artifact_export_parity(genuine, mutation):
    raw = _clone(genuine[1][3])
    if mutation == "census":
        del raw["local_tensors"][FINAL]
        message = INVENTORY_ERROR
    elif mutation == "runtime_geometry":
        raw["config"]["hidden_size"] += 4
        _rehash(raw)
        message = "legacy runtime field 'hidden_size' disagrees with original source"
    else:
        raw["stages"]["layers.0.self_attn.qkv_proj"]["in_features"] += 1
        message = "legacy stage geometry disagrees with source"
    export_bundle(genuine[1][4])
    changed = BundleDocument(raw)
    _reject_payload(changed.to_bytes(), match=_exact(message))
    with pytest.raises(ArtifactError, match=_exact(message)):
        export_bundle(changed)


@pytest.mark.parametrize("mutation", ["census", "runtime_geometry"])
def test_artifact_manifest_malformed_census_geometry_rejected(genuine, mutation):
    payload = genuine[1][5]
    exported = export_bundle(payload)
    fingerprint = hashlib.sha256(payload).hexdigest()
    manifest = parse_manifest(exported.manifest, fingerprint=fingerprint, size=len(payload))
    assert reconstruct_bundle(manifest, lambda row: exported.objects[row["sha256"]]) == payload
    raw = msgpack.unpackb(exported.manifest, raw=False)
    skeleton = msgpack.unpackb(raw["skeleton"], raw=False)
    if mutation == "census":
        del skeleton["local_tensors"][FINAL]
        message = INVENTORY_ERROR
    else:
        skeleton["config"]["hidden_size"] += 4
        _rehash(skeleton)
        message = "legacy runtime field 'hidden_size' disagrees with original source"
    raw["skeleton"] = msgpack.packb(skeleton, use_bin_type=True)
    malformed = msgpack.packb(raw, use_bin_type=True)
    with pytest.raises(ArtifactError, match=_exact(message)):
        parse_manifest(malformed, fingerprint=fingerprint, size=len(payload))


CANONICAL_GEMMA_CONFIGS = [
    "gemma-4-E2B-it-3e22461f-config.json",
    "gemma-4-E4B-it-ee0ef602-config.json",
]


@pytest.mark.parametrize("filename", CANONICAL_GEMMA_CONFIGS)
def test_real_canonical_gemma_envelope_reaches_canonical_lowering(tmp_path, monkeypatch, filename):
    from pllm import modeling

    fixture = Path(__file__).resolve().parents[1] / "crates" / "pllm-models" / "tests" / "fixtures" / filename
    source = json.loads(fixture.read_text(encoding="utf-8"))
    assert source["model_type"] == "gemma4"
    assert source["text_config"]["model_type"] == "gemma4_text"
    assert client_tensor_contract.legacy_source_geometry(source) is None
    root = tmp_path / "canonical-config-only"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(source), encoding="utf-8")
    calls = []
    original = modeling.lower_model

    def lower(config, **kwargs):
        plan = original(config, **kwargs)
        calls.append((config, plan))
        return plan

    with monkeypatch.context() as patch:
        patch.setattr(modeling, "lower_model", lower)
        patch.setattr(loaders, "gemma4_stage_plan", _forbid("canonical envelope entered legacy stage builder"))
        manifest = load_hf_directory(root, model_id="canonical-source")
    assert calls and calls[0][0] == source
    # Real Rust lowering ran; no coefficients or synthetic checkpoint are needed
    # for this config-only route ownership regression.
    assert calls[0][1].digest
    flattened = {**source["text_config"], "semantic_source_config": source}
    assert not client_tensor_contract.legacy_source_candidate({
        "config": flattened, "manifest": manifest.to_dict(),
    })


def test_explicit_legacy_contract_cannot_hide_behind_semantic_envelope(genuine):
    fixture = Path(__file__).resolve().parents[1] / "crates" / "pllm-models" / "tests" / "fixtures" / CANONICAL_GEMMA_CONFIGS[0]
    raw = _clone(genuine[1][3])
    raw["config"]["semantic_source_config"] = json.loads(fixture.read_text(encoding="utf-8"))
    assert client_tensor_contract.legacy_source_candidate(raw)
    _rehash(raw)
    _reject(raw, match=_exact("legacy runtime has unknown source-independent options"))


@pytest.mark.parametrize("field,value", [
    ("model_type", "llama"), ("architectures", ["Gemma4ForConditionalGeneration"]),
])
def test_flat_legacy_contradictory_identity_stays_rejected(genuine, tmp_path, field, value):
    root = _checkpoint(tmp_path / "contradictory")
    path = root / "config.json"
    source = json.loads(path.read_text(encoding="utf-8"))
    source[field] = value
    path.write_text(json.dumps(source), encoding="utf-8")
    _reject_source(root, match=_exact("legacy source requires flat gemma4_text / Gemma4ForCausalLM identity"))


@pytest.mark.parametrize("entry", ["direct_public", "direct_document", "artifact_export", "artifact_metadata"])
def test_actual_oversized_inventory_precedes_digest_and_builder(genuine, monkeypatch, entry):
    raw = _clone(genuine[1][3])
    cap = 21 + 44 * 2  # Still admits genuine source geometry.
    stages, weights, tensors = raw["stages"], raw["client_weights"], raw["local_tensors"]
    exemplar = next(iter(stages.values()))
    while 4 * len(stages) + 2 * len(weights) + len(tensors) + 1 <= cap:
        stages[f"extra_stage.{len(stages)}"] = _clone(exemplar)
    assert max(len(stages), len(weights), len(tensors)) <= cap
    assert len(stages) + len(weights) + len(tensors) <= cap
    changed = BundleDocument(raw)
    payload = changed.to_bytes()
    imported = None
    try:
        with monkeypatch.context() as patch:
            patch.setattr(client_tensor_contract, "MAX_OBJECTS", cap)
            patch.setattr(client_tensor_contract.json, "dumps", _forbid("oversized actual inventory reached digest serializer"))
            patch.setattr(client_tensor_contract, "gemma4_stage_plan", _forbid("oversized actual inventory reached stage builder"))
            if entry == "direct_public":
                with pytest.raises(TransformerClientError, match=_exact("legacy bundle aggregate object bound exceeded")):
                    imported = ClientBundle.unpack(payload)
            elif entry == "direct_document":
                with pytest.raises(TransformerClientError, match=_exact("legacy bundle aggregate object bound exceeded")):
                    imported = ClientBundle._from_document(raw)
            elif entry == "artifact_export":
                with pytest.raises(ArtifactError, match=_exact("legacy bundle aggregate object bound exceeded")):
                    export_bundle(changed)
            else:
                with pytest.raises(ArtifactError, match=_exact("legacy bundle aggregate object bound exceeded")):
                    bundle_artifacts._binary_metadata(raw)
    finally:
        if imported is not None:
            imported.close()


@pytest.mark.parametrize("failure_point", ["partial_stage", "runtime_config"])
def test_failed_load_retires_real_unpublished_stage_and_store_ownership(
    genuine, tmp_path, monkeypatch, failure_point,
):
    root = _checkpoint(tmp_path / "failing")
    manifest = load_hf_directory(root, model_id="unpublished-failure")
    engine = MaskedTransformerEngine(
        threads=1, weight_bits=8, activation_bits=8,
        compiled_cache_dir=tmp_path / "compiled", streaming_threshold_elements=1,
    )
    failure = RuntimeError("injected failure after genuine legacy ownership")
    runtimes, retired, releases, mappings, cache_clears = [], [], [], [], []
    ledgers, close_actions = [], []
    original_owned = MaskedTransformerEngine._load_stage_owned
    original_config = MaskedTransformerEngine._bundle_runtime_config
    original_retire = MaskedTransformerEngine._retire_failed_load
    original_release = transformer_engine.StageRuntime.release_weight
    original_close = MaskedTransformerEngine._close_array_mappings
    original_clear = SafeTensorStore.clear_cache
    original_cleanup = MaskedTransformerEngine._cleanup_attempt

    def stage_owned(self, store, stage, manifest, **kwargs):
        runtime = original_owned(self, store, stage, manifest, **kwargs)
        assert runtime.compiled_weight is not None
        runtimes.append(runtime)
        if failure_point == "partial_stage" and len(runtimes) == 2:
            raise failure
        return runtime

    def runtime_config(model):
        result = original_config(model)
        if failure_point == "runtime_config":
            assert model.stages and model.tokenizer and model.local_tensors
            raise failure
        return result

    def retire(self, model, **kwargs):
        assert model.manifest.id not in self.models
        had_tokenizer = bool(model.tokenizer)
        original_retire(self, model, **kwargs)
        retired.append((model, had_tokenizer))

    def release(runtime):
        original_release(runtime)
        releases.append(runtime)

    def close_arrays(arrays, **kwargs):
        assert kwargs.get("mappings") is not None
        ledgers.append(kwargs["mappings"])
        arrays = tuple(arrays)
        for array in arrays:
            while isinstance(array, np.ndarray):
                if isinstance(array, np.memmap) and array._mmap is not None:
                    mappings.append(array._mmap)
                array = array.base
        original_close(arrays, **kwargs)

    def cleanup(primary, label, action):
        if label == "array mapping close":
            close_actions.append(action.__self__)
        original_cleanup(primary, label, action)

    def clear(store):
        before = bool(store._cache)
        original_clear(store)
        cache_clears.append((store, before))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(MaskedTransformerEngine, "_load_stage_owned", stage_owned)
            patch.setattr(MaskedTransformerEngine, "_bundle_runtime_config", staticmethod(runtime_config))
            patch.setattr(MaskedTransformerEngine, "_retire_failed_load", retire)
            patch.setattr(transformer_engine.StageRuntime, "release_weight", release)
            patch.setattr(MaskedTransformerEngine, "_close_array_mappings", staticmethod(close_arrays))
            patch.setattr(SafeTensorStore, "clear_cache", clear)
            patch.setattr(MaskedTransformerEngine, "_cleanup_attempt", staticmethod(cleanup))
            with pytest.raises(RuntimeError, match=_exact(str(failure))) as caught:
                asyncio.run(engine.load(manifest))
        assert caught.value is failure
        assert manifest.id not in engine.models
        assert manifest.id not in engine._active_cache_entries
        assert manifest.id not in engine._metal_stages
        assert len(runtimes) >= 2 and retired and releases
        assert all(any(runtime is released for released in releases) for runtime in runtimes)
        assert all(runtime.compiled_weight is None and runtime.quantized_weight is None for runtime in runtimes)
        assert mappings and all(mapping.closed for mapping in mappings)
        assert ledgers and all(ledger is ledgers[0] for ledger in ledgers)
        assert len(close_actions) == len({id(mapping) for mapping in close_actions})
        assert {id(mapping) for mapping in close_actions} == {id(mapping) for mapping in mappings}
        assert cache_clears and any(before for _, before in cache_clears)
        assert all(not store._cache for store, _ in cache_clears)
        assert all(not model.stages and not model.local_tensors and not model.tokenizer for model, _ in retired)
        if failure_point == "runtime_config":
            assert any(had_tokenizer for _, had_tokenizer in retired)
        # Producer tokenizer is real byte/JSON metadata, not FilePayload.
    finally:
        if manifest.id in engine.models:
            asyncio.run(engine.unload(manifest.id))


def test_public_import_forwards_genuine_document_to_shared_decoder(genuine, monkeypatch):
    payload = genuine[1][5]
    original = bundle_document.unpack_document
    calls = []

    def forward(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(bundle_document, "unpack_document", forward)
        _import_and_close(payload)
    assert len(calls) == 1
    assert calls[0][0] == (payload,)
    assert calls[0][1] == {}
    assert calls[0][2] == genuine[1][3]
    # This proves public wiring, not global patchability of the captured cap.


@pytest.mark.parametrize("partial", [False, True], ids=["late-publication", "partial-stage"])
def test_cleanup_failure_preserves_primary_and_attempts_remaining_real_owners(
    genuine, tmp_path, monkeypatch, partial,
):
    root = _checkpoint(tmp_path / "cleanup-failure")
    manifest = load_hf_directory(root, model_id="cleanup-primary")
    engine = MaskedTransformerEngine(
        threads=1, weight_bits=8, activation_bits=8,
        compiled_cache_dir=tmp_path / "compiled", streaming_threshold_elements=1,
    )
    primary = RuntimeError("genuine load primary failure")
    cleanup_error = RuntimeError("injected owned stage release failure")
    runtimes, models, arrays, releases, clears = [], [], [], [], []
    worker_finished = threading.Event()
    injected = threading.Event()
    original_owned = MaskedTransformerEngine._load_stage_owned
    original_stage = MaskedTransformerEngine._load_stage
    original_config = MaskedTransformerEngine._bundle_runtime_config
    original_retire = MaskedTransformerEngine._retire_failed_load
    original_release = transformer_engine.StageRuntime.release_weight
    original_clear = SafeTensorStore.clear_cache

    def owned(self, store, stage, manifest, **kwargs):
        runtime = original_owned(self, store, stage, manifest, **kwargs)
        runtimes.append(runtime)
        arrays.extend((runtime.weight_scales, runtime.weight.values, runtime.weight.scales))
        if partial and len(runtimes) == 2:
            injected.set()
            raise primary
        return runtime

    def stage(self, *args, **kwargs):
        try:
            return original_stage(self, *args, **kwargs)
        finally:
            if injected.is_set():
                worker_finished.set()

    def config(model):
        result = original_config(model)
        if not partial:
            assert model.stages and model.local_tensors and model.tokenizer
            raise primary
        return result

    def retire(self, model, **kwargs):
        models.append((model, kwargs.get("rollback", True), kwargs.get("primary")))
        return original_retire(self, model, **kwargs)

    def release(runtime):
        releases.append(runtime)
        if len(releases) == 1:
            raise cleanup_error
        return original_release(runtime)

    def clear(store):
        clears.append((store, worker_finished.is_set()))
        return original_clear(store)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(MaskedTransformerEngine, "_load_stage_owned", owned)
            patch.setattr(MaskedTransformerEngine, "_load_stage", stage)
            patch.setattr(MaskedTransformerEngine, "_bundle_runtime_config", staticmethod(config))
            patch.setattr(MaskedTransformerEngine, "_retire_failed_load", retire)
            patch.setattr(transformer_engine.StageRuntime, "release_weight", release)
            patch.setattr(SafeTensorStore, "clear_cache", clear)
            with pytest.raises(RuntimeError) as caught:
                asyncio.run(engine.load(manifest))
        assert caught.value is primary and caught.value is not cleanup_error
        assert caught.value.args == ("genuine load primary failure",)
        assert str(caught.value) == "genuine load primary failure"
        assert "legacy load cleanup failed: stage release" in caught.value.__notes__
        assert len(runtimes) >= 2
        assert all(any(runtime is released for released in releases) for runtime in runtimes)
        assert all(not model.stages and not model.local_tensors and not model.tokenizer for model, _, _ in models)
        assert all(error is primary for _, _, error in models)
        root_models = [model for model, rollback, _ in models if rollback]
        assert len(root_models) == 1
        assert len(clears) == 1 and clears[0][0] is root_models[0].store
        if partial:
            assert worker_finished.is_set() and clears[0][1]
            assert any(not rollback for _, rollback, _ in models)
        assert not root_models[0].store._cache
        assert manifest.id not in engine.models
        assert manifest.id not in engine._loading_model_ids
        # Failed release is an attempted cleanup, not proof that it freed an
        # owner. Unpatched cleanup below retires captured genuine resources.
    finally:
        try:
            for runtime in runtimes:
                original_release(runtime)
        finally:
            try:
                MaskedTransformerEngine._close_array_mappings(arrays)
            finally:
                if manifest.id in engine.models:
                    asyncio.run(engine.unload(manifest.id))
                arrays.clear()


async def _scheduled_turn():
    loop = asyncio.get_running_loop()
    reached = loop.create_future()
    loop.call_soon(reached.set_result, None)
    await reached


@pytest.mark.parametrize("cleanup_fault", [False, True])
def test_repeated_cancel_drains_real_worker_excludes_same_id_and_allows_retry(
    genuine, tmp_path, monkeypatch, cleanup_fault,
):
    root = _checkpoint(tmp_path / "cancelled")
    manifest = load_hf_directory(root, model_id="cancelled-owner")
    other_root = _checkpoint(tmp_path / "other")
    other_manifest = load_hf_directory(other_root, model_id="other-owner")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    release_worker = threading.Event()
    worker_finished = threading.Event()
    runtimes, retired, clears, cancellations, release_attempts = [], [], [], [], []
    original_stage = MaskedTransformerEngine._load_stage
    original_retire = MaskedTransformerEngine._retire_failed_load
    original_clear = SafeTensorStore.clear_cache
    original_outcome = MaskedTransformerEngine._raise_legacy_worker_outcome
    original_release = transformer_engine.StageRuntime.release_weight
    armed = True
    loop = entered = None

    def stage(self, store, spec, source_manifest, **kwargs):
        nonlocal armed
        runtime = original_stage(self, store, spec, source_manifest, **kwargs)
        if source_manifest.id == manifest.id and armed:
            armed = False
            runtimes.append(runtime)
            loop.call_soon_threadsafe(entered.set)
            if not release_worker.wait(30):
                raise AssertionError("test did not release genuine source worker")
            worker_finished.set()
        return runtime

    def retire(self, model, **kwargs):
        retired.append((list(model.stages.values()), worker_finished.is_set(), kwargs.get("primary")))
        return original_retire(self, model, **kwargs)

    def clear(store):
        clears.append((store, worker_finished.is_set()))
        return original_clear(store)

    def outcome(worker_error, first_cancel):
        if first_cancel is not None:
            cancellations.append(first_cancel)
        return original_outcome(worker_error, first_cancel)

    def release(runtime):
        release_attempts.append(runtime)
        if cleanup_fault and len(release_attempts) == 1:
            raise RuntimeError("injected cancellation cleanup failure")
        return original_release(runtime)

    async def scenario():
        nonlocal loop, entered
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        await engine.load(other_manifest)
        other = engine.models[other_manifest.id]
        other_cache = engine._active_cache_entries[other_manifest.id]
        task = asyncio.create_task(engine.load(manifest))
        try:
            await asyncio.wait_for(entered.wait(), 30)
            with pytest.raises(TransformerEngineError, match=_exact(f"model {manifest.id!r} load is already in progress")):
                await engine.load(manifest)
            task.cancel("first-source-cancel")
            await _scheduled_turn()
            task.cancel("second-source-cancel")
            await _scheduled_turn()
            assert task.cancelling() == 2 and not task.done()
            assert manifest.id in engine._loading_model_ids
            assert manifest.id not in engine.models
            assert not retired and not clears
            assert runtimes[0].compiled_weight is not None
            with pytest.raises(TransformerEngineError, match=_exact(f"model {manifest.id!r} load is already in progress")):
                await engine.load(manifest)
            release_worker.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert len(cancellations) == 1 and caught.value is cancellations[0]
            assert caught.value.args == ("first-source-cancel",)
            assert retired and all(finished for _, finished, _ in retired)
            assert any(any(runtime is runtimes[0] for runtime in rows) for rows, _, _ in retired)
            assert all(primary is caught.value for _, _, primary in retired)
            assert len(clears) == 1 and clears[0][1]
            assert any(runtime is runtimes[0] for runtime in release_attempts)
            if cleanup_fault:
                assert "legacy load cleanup failed: stage release" in caught.value.__notes__
            assert manifest.id not in engine.models and manifest.id not in engine._loading_model_ids
            assert engine.models[other_manifest.id] is other
            assert engine._active_cache_entries[other_manifest.id] is other_cache
            await engine.load(load_hf_directory(root, model_id=manifest.id))
            _import_and_close(engine.client_bundle_document(manifest.id).to_bytes())
            with pytest.raises(TransformerEngineError, match=_exact(f"model {manifest.id!r} is already loaded")):
                await engine.load(manifest)
        finally:
            release_worker.set()
            if not task.done():
                task.cancel("test-finally")
            await asyncio.gather(task, return_exceptions=True)
            for runtime in runtimes:
                original_release(runtime)
            for model_id in list(engine.models):
                await engine.unload(model_id)

    with monkeypatch.context() as patch:
        patch.setattr(MaskedTransformerEngine, "_load_stage", stage)
        patch.setattr(MaskedTransformerEngine, "_retire_failed_load", retire)
        patch.setattr(SafeTensorStore, "clear_cache", clear)
        patch.setattr(MaskedTransformerEngine, "_raise_legacy_worker_outcome", staticmethod(outcome))
        patch.setattr(transformer_engine.StageRuntime, "release_weight", release)
        asyncio.run(scenario())


def test_asyncio_shutdown_drains_real_worker_before_source_retirement(genuine, tmp_path, monkeypatch):
    root = _checkpoint(tmp_path / "shutdown")
    manifest = load_hf_directory(root, model_id="shutdown-owner")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    release_worker, shutdown_started, worker_finished = (threading.Event() for _ in range(3))
    runtimes, retired, clears, first_cancels, observed_cancels, thread_errors, tasks = ([] for _ in range(7))
    original_stage = MaskedTransformerEngine._load_stage
    original_retire = MaskedTransformerEngine._retire_failed_load
    original_clear = SafeTensorStore.clear_cache
    original_outcome = MaskedTransformerEngine._raise_legacy_worker_outcome
    original_release = transformer_engine.StageRuntime.release_weight
    loop = entered = None

    def stage(self, *args, **kwargs):
        runtime = original_stage(self, *args, **kwargs)
        runtimes.append(runtime)
        loop.call_soon_threadsafe(entered.set)
        if not release_worker.wait(30):
            raise AssertionError("shutdown test did not release source worker")
        worker_finished.set()
        return runtime

    def retire(self, model, **kwargs):
        retired.append((list(model.stages.values()), worker_finished.is_set()))
        return original_retire(self, model, **kwargs)

    def clear(store):
        clears.append(worker_finished.is_set())
        return original_clear(store)

    def outcome(worker_error, first_cancel):
        if first_cancel is not None:
            first_cancels.append(first_cancel)
        return original_outcome(worker_error, first_cancel)

    def release_during_shutdown():
        try:
            if not shutdown_started.wait(30):
                raise AssertionError("event-loop shutdown never cancelled sentinel task")
            if retired or clears:
                raise AssertionError("source retired before shutdown worker was released")
        except BaseException as error:
            thread_errors.append(error)
        finally:
            release_worker.set()

    async def source_load():
        try:
            await engine.load(manifest)
        except asyncio.CancelledError as error:
            observed_cancels.append(error)
            raise

    async def main():
        nonlocal loop, entered
        loop = asyncio.get_running_loop()
        entered = asyncio.Event()
        sentinel_entered = asyncio.Event()

        async def shutdown_sentinel():
            sentinel_entered.set()
            try:
                await loop.create_future()
            finally:
                shutdown_started.set()

        tasks.extend((asyncio.create_task(source_load()), asyncio.create_task(shutdown_sentinel())))
        await asyncio.wait_for(entered.wait(), 30)
        await asyncio.wait_for(sentinel_entered.wait(), 30)
        assert not tasks[0].done() and manifest.id in engine._loading_model_ids
        # Return with genuine source task/worker pending. asyncio.run's shutdown
        # cancellation wakes the sentinel; independent thread releases the worker.

    releaser = threading.Thread(target=release_during_shutdown, daemon=True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(MaskedTransformerEngine, "_load_stage", stage)
            patch.setattr(MaskedTransformerEngine, "_retire_failed_load", retire)
            patch.setattr(SafeTensorStore, "clear_cache", clear)
            patch.setattr(MaskedTransformerEngine, "_raise_legacy_worker_outcome", staticmethod(outcome))
            releaser.start()
            asyncio.run(main())
        releaser.join(30)
        assert not releaser.is_alive() and not thread_errors
        assert worker_finished.is_set() and tasks[0].cancelled()
        assert len(first_cancels) == len(observed_cancels) == 1
        assert observed_cancels[0] is first_cancels[0]
        assert retired and all(finished for _, finished in retired)
        assert any(any(runtime is runtimes[0] for runtime in rows) for rows, _ in retired)
        assert clears == [True]
        assert runtimes[0].compiled_weight is None and runtimes[0].quantized_weight is None
        assert manifest.id not in engine.models and manifest.id not in engine._loading_model_ids
    finally:
        shutdown_started.set()
        release_worker.set()
        if releaser.ident is not None:
            releaser.join(30)
        for runtime in runtimes:
            original_release(runtime)
        if manifest.id in engine.models:
            asyncio.run(engine.unload(manifest.id))


def test_failed_load_preserves_foreign_exception_frame_locals(genuine, tmp_path, monkeypatch):
    root = _checkpoint(tmp_path / "foreign-frame")
    manifest = load_hf_directory(root, model_id="foreign-frame-owner")
    engine = MaskedTransformerEngine(threads=1, weight_bits=8, activation_bits=8)
    primary = RuntimeError("external publication callback failure")
    sentinels, models = [], []
    original = MaskedTransformerEngine._bundle_runtime_config

    def external_callback(model):
        sentinel = object()
        sentinels.append(sentinel)
        models.append(model)
        original(model)
        raise primary

    try:
        with monkeypatch.context() as patch:
            patch.setattr(MaskedTransformerEngine, "_bundle_runtime_config", staticmethod(external_callback))
            with pytest.raises(RuntimeError, match=_exact(str(primary))) as caught:
                asyncio.run(engine.load(manifest))
        assert caught.value is primary
        frames = []
        tb = caught.value.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code is external_callback.__code__:
                frames.append(tb.tb_frame)
            tb = tb.tb_next
        assert len(frames) == 1
        assert frames[0].f_locals["sentinel"] is sentinels[0]
        assert all(not model.stages and not model.local_tensors and not model.tokenizer for model in models)
        assert manifest.id not in engine.models and manifest.id not in engine._loading_model_ids
    finally:
        if manifest.id in engine.models:
            asyncio.run(engine.unload(manifest.id))
