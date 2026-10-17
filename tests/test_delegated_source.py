"""Real bounded source import; no model downloads, native builds or benchmarks."""

import hashlib
import json
import os
from pathlib import Path
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pllm import Model, _native
from pllm import model_loader, modeling, sources
from pllm.configuration import ComponentRef, Deployment, ExecutionBudget, Experiment, Pipeline
from pllm.runtime import delegated_source as source, loaders
from pllm.runtime.safetensors_store import SafeTensorStore

from test_delegated_decoder import config as decoder_config, fixture as decoder_fixture


def write_source_fixture(path, *, family="qwen2", tied=True, capacity="capsule", vocabulary=32):
    """Independent GKM tensors plus a genuine complete 32-token WordLevel model."""
    path.mkdir(parents=True, exist_ok=True)
    config = {**decoder_config(family, tied), "bos_token_id": 0, "eos_token_id": 1,
              "vocab_size": vocabulary}
    contract, arrays, tensors = decoder_fixture(config, capacity=capacity)
    (path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2, "hello": 3, "world": 4}
    vocab.update({f"word{i}": i for i in range(5, config["vocab_size"])})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens(["<bos>", "<eos>", "<unk>"])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "bos_token": "<bos>", "eos_token": "<eos>", "add_bos_token": False,
        "model_max_length": config["max_position_embeddings"],
    }), encoding="utf-8")
    save_file(arrays, path / "model.safetensors", metadata={"format": "pt"})
    return config, contract, arrays, tensors


def admit(path, **kwargs):
    return source.admit_source(Model.path(str(path)), input_rows=3, decode_steps=2, **kwargs)


def forbidden(*args, **kwargs):
    raise AssertionError("forbidden early lowering, hashing, or tensor value access")


def forbid_values(monkeypatch):
    for name in ("get", "get_slice", "iter_slices", "get_linear"):
        monkeypatch.setattr(SafeTensorStore, name, forbidden)
    monkeypatch.setattr(source._AdmittedTensorStore, "get", forbidden)


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("revision", [None, "a" * 40])
def test_hub_source_uses_shared_resolver_and_retains_original_model(tmp_path, monkeypatch,
                                                                 offline, revision):
    from pllm.runtime.hf_hub import ResolvedModelSource

    write_source_fixture(tmp_path)
    model = Model.hf("fixture/source", revision=revision, local_files_only=offline)
    calls = []

    def resolve(name, **kwargs):
        calls.append((name, kwargs))
        return ResolvedModelSource(tmp_path, name, name, kwargs["revision"])

    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", resolve)
    forbid_values(monkeypatch)
    admitted = source.admit_source(model, input_rows=3, decode_steps=2)
    assert admitted.model == model and admitted.path == tmp_path
    assert calls == [("fixture/source", {
        "model_id": None, "revision": revision, "local_files_only": offline,
    })]
    assert admitted.source.manifest.metadata["model_spec"] == model.to_spec()
    assert (admitted.checkpoint_digest, admitted.source_lock_digest, dict(admitted.source_lock)) == (
        model_loader._source_lock(tmp_path, model)
    )


@pytest.mark.parametrize("changed", [None, "revision", "blob"])
def test_hub_snapshot_checks_original_revision_and_blob_identity(tmp_path, monkeypatch, changed):
    from pllm.runtime.hf_hub import ResolvedModelSource

    commit = "a" * 40
    root = tmp_path / "models--fixture--source"
    snapshot = root / "snapshots" / commit
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    write_source_fixture(snapshot)
    for path in snapshot.iterdir():
        content = path.read_bytes()
        blob = blobs / hashlib.sha256(content).hexdigest()
        path.rename(blob)
        path.symlink_to(blob)
    if changed == "blob":
        blob = (snapshot / "model.safetensors").resolve()
        content = bytearray(blob.read_bytes())
        content[-1] ^= 1
        blob.write_bytes(content)
    model = Model.hf("fixture/source", revision="b" * 40 if changed == "revision" else commit,
                     local_files_only=True)
    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", lambda *a, **kw:
                        ResolvedModelSource(snapshot, model.source, model.source, commit))
    forbid_values(monkeypatch)
    if changed is not None:
        with pytest.raises(ValueError, match="pinned revision" if changed == "revision" else "blob digest"):
            source.admit_source(model, input_rows=3, decode_steps=2)
        return
    admitted = source.admit_source(model, input_rows=3, decode_steps=2)
    assert admitted.source_lock["repo_id"] == model.source
    assert admitted.source_lock["commit"] == commit
    assert (admitted.checkpoint_digest, admitted.source_lock_digest, dict(admitted.source_lock)) == (
        model_loader._source_lock(snapshot, model)
    )


def forbid_ordinary_loading(monkeypatch):
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(model_loader, "load_hf_directory", forbidden)
    monkeypatch.setattr(loaders, "load_hf_directory", forbidden)
    monkeypatch.setattr(modeling, "lower_model", forbidden)
    monkeypatch.setattr(source, "SafeTensorStore", forbidden)
    monkeypatch.setattr(source, "_store", forbidden)
    monkeypatch.setattr(model_loader, "_source_lock", forbidden)
    forbid_values(monkeypatch)


def test_identify_executed_modules(record_property):
    record_property("delegated_source_module", str(Path(source.__file__).resolve()))
    record_property("native_module", str(Path(_native.__file__).resolve()))
    assert callable(_native.gkm_decoder_contract)
    assert Path(source.__file__).name == "delegated_source.py"


def sdk_experiment(path, *, rows=3, outputs=3):
    """Exact SDK schema fixture; installed v3 public registry predates these IDs.

    Bypass only that older Python component registry. The production loader still
    receives an actual Experiment and must call native validation on its bytes.
    SDK-specific native calls are explicit doubles below, except the optional
    real-extension check. Lower-level decoder/source tests use the real v3 core.
    """
    entries = {
        "kernels": ("pllm/cpu", {"threads": 1}),
        "linear": ("pllm/open-weight-verifiable-outsourcing/v1",
                   {"sharing": "full", "generation": "fixed_length_greedy"}),
        "topology": ("pllm/two-online-delegation-workers/v1", {}),
        "state": ("pllm/client-local-kv", {}),
        "verification": ("pllm/published-linear-verify/v1", {"acceptance": "request"}),
    }
    components = {}
    for slot, (identity, params) in entries.items():
        value = object.__new__(ComponentRef)
        object.__setattr__(value, "component", identity)
        object.__setattr__(value, "params", MappingProxyType(params))
        components[slot] = value
    pipeline = object.__new__(Pipeline)
    object.__setattr__(pipeline, "model", Model.path(str(path), format="safetensors"))
    object.__setattr__(pipeline, "components", MappingProxyType(components))
    object.__setattr__(pipeline, "profile", None)
    return Experiment(
        "source-admission", pipeline, Deployment.local(root=str(path.parent / "deployment")),
        ExecutionBudget(requests=1, max_input_tokens=rows, max_new_tokens=outputs),
    )


@pytest.fixture
def native_sdk_double(monkeypatch):
    """Exercise the new bridge without implying a rebuilt SDK-capable wheel."""
    calls = {"resolve": [], "runtime": [], "bytes": []}

    def resolve(encoded):
        calls["resolve"].append(encoded)
        assert type(encoded) is bytes
        assert source._canonical(json.loads(encoded)) == encoded

    def runtime(experiment_bytes, config_bytes, rows, outputs, source_lock, tokenizer):
        args = (experiment_bytes, config_bytes, rows, outputs, source_lock, tokenizer)
        calls["runtime"].append(args)
        document = json.loads(experiment_bytes)
        decoder = _native.gkm_decoder_contract(config_bytes, rows, outputs - 1)
        fields = {
            "schema_version": source._RUNTIME_SCHEMA,
            "configuration": hashlib.sha256(b"pllm.configuration.v1\0" + experiment_bytes).hexdigest(),
            "composition": hashlib.sha256(
                b"pllm.pipeline.v2\0" + source._canonical(document["pipeline"])
            ).hexdigest(),
            "source_lock": source_lock, "tokenizer": tokenizer, "decoder": None,
        }
        # Keep the real decoder's canonical float spelling; no Python roundtrip.
        result = b"{" + b",".join(
            source._canonical(key) + b":" + (decoder if key == "decoder" else source._canonical(fields[key]))
            for key in sorted(fields)
        ) + b"}"
        calls["bytes"].append(result)
        return result

    monkeypatch.setattr(_native, "resolve_experiment", resolve)
    monkeypatch.setattr(_native, "gkm_runtime_contract", runtime, raising=False)
    return calls


@pytest.mark.parametrize("family", ["qwen2", "qwen3", "llama"])
@pytest.mark.parametrize("tied", [False, True])
def test_source_identity_complete_tokenizer_and_native_roundtrip(tmp_path, family, tied):
    config, contract, arrays, tensors = write_source_fixture(tmp_path, family=family, tied=tied)
    model = Model.path(str(tmp_path))
    admitted = admit(tmp_path)
    loaded = admitted.load_values()
    checkpoint, digest, lock = model_loader._source_lock(tmp_path, model)
    assert (admitted.checkpoint_digest, admitted.source_lock_digest, admitted.source_lock) == (
        checkpoint, digest, lock,
    )
    assert admitted.source.manifest.source_lock_digest == digest
    assert admitted.source.manifest.checkpoint_digest == checkpoint
    assert admitted.source.manifest.vocab_size == 32
    assert admitted.config_source_bytes == (tmp_path / "config.json").read_bytes()
    assert admitted.config == config == json.loads(admitted.config_bytes)
    assert admitted.contract == contract
    assert admitted.source_shapes == {name: array.shape for name, array in arrays.items()}
    assert admitted.source_float32_bytes == sum(map(len, tensors.values()))
    assert loaded.tensors == tensors
    assert set(loaded.coefficients) == set(contract["local_tensors"])
    assert all(loaded.coefficients[name] is loaded.tensors[name] for name in loaded.coefficients)
    assert loaded.tensors[contract["table"]][:4] == bytes.fromhex("00000080")
    assert admitted.tokenizer_bytes == (tmp_path / "tokenizer.json").read_bytes()
    assert admitted.tokenizer_digest == hashlib.sha256(admitted.tokenizer_bytes).hexdigest()
    tokenizer = admitted.tokenizer()
    assert tokenizer.vocab_size == 32
    assert tokenizer.encode("hello world", add_bos=True) == [0, 3, 4]
    assert tokenizer.decode([3, 4]) == "hello world"
    assert admitted.tokenizer_descriptor["bos_token_id"] == 0
    native = _native._GkmDecoderSession(admitted.config_bytes, 3, 2, dict(loaded.tensors))
    try:
        assert json.loads(native.contract()) == admitted.contract
        assert native.lookup(0) == tensors[contract["table"]][: contract["width"] * 4]
        coefficients = native.coefficients()
        assert coefficients == loaded.coefficients
    finally:
        native.abort()


def test_metadata_admission_never_materializes_values_and_records_are_isolated(tmp_path, monkeypatch):
    write_source_fixture(tmp_path)
    forbid_values(monkeypatch)
    monkeypatch.setattr(model_loader, "resolve_model", forbidden)
    monkeypatch.setattr(model_loader, "load_hf_directory", forbidden)
    monkeypatch.setattr(loaders, "load_hf_directory", forbidden)
    monkeypatch.setattr(modeling, "lower_model", forbidden)
    admitted = admit(tmp_path)
    assert admitted.contract["execution_payload_estimate_bytes"] > admitted.source_float32_bytes
    admitted.config["num_hidden_layers"] = 1000000
    admitted.contract["source_shapes"].clear()
    admitted.source_lock["files"].clear()
    admitted.source.manifest.metadata.clear()
    admitted.tokenizer_descriptor["bos_token_id"] = 999
    assert admitted.config["num_hidden_layers"] == 2
    assert admitted.source_shapes
    assert admitted.source_lock["files"]
    assert admitted.source.manifest.metadata["source_lock_digest"] == admitted.source_lock_digest
    assert admitted.tokenizer_descriptor["bos_token_id"] == 0


def test_experiment_budget_admits_complete_window_before_values(tmp_path, monkeypatch, native_sdk_double):
    root = tmp_path / "source"
    write_source_fixture(root)
    experiment = sdk_experiment(root)
    forbid_values(monkeypatch)
    admitted = source.admit_source(experiment)
    assert admitted.contract["input_rows"] == 3
    assert admitted.contract["decode_steps"] == 2
    assert admitted.source_lock["kind"] == "safetensors"
    assert native_sdk_double["resolve"] == [experiment.canonical_bytes()]
    assert admitted.experiment_bytes == experiment.canonical_bytes()
    assert admitted.runtime_contract_bytes is native_sdk_double["bytes"][0]
    assert native_sdk_double["runtime"] == [(
        experiment.canonical_bytes(), admitted.config_bytes, 3, 3,
        admitted.source_lock_digest, admitted.tokenizer_digest,
    )]
    with pytest.raises(ValueError, match="budget"):
        source.admit_source(experiment, input_rows=1, decode_steps=0)


def test_native_experiment_rejection_precedes_any_source_access(tmp_path, monkeypatch):
    experiment = sdk_experiment(tmp_path / "absent")
    calls = []

    def reject(encoded):
        calls.append(encoded)
        raise ValueError("injected native Experiment rejection")

    monkeypatch.setattr(_native, "resolve_experiment", reject)
    for name in ("_identity", "_read_bounded", "_files", "_locked", "_store", "_tokenizer", "decoder_contract"):
        monkeypatch.setattr(source, name, forbidden)
    with pytest.raises(ValueError, match="native Experiment rejection"):
        source.admit_source(experiment)
    assert calls == [experiment.canonical_bytes()]


def test_missing_sdk_native_binding_fails_before_source_access(tmp_path, monkeypatch):
    monkeypatch.setattr(_native, "resolve_experiment", lambda encoded: None)
    monkeypatch.delattr(_native, "gkm_runtime_contract", raising=False)
    monkeypatch.setattr(source, "_identity", forbidden)
    with pytest.raises(ValueError, match="lacks GKM runtime admission"):
        source.admit_source(sdk_experiment(tmp_path / "absent"))


@pytest.mark.parametrize("rows, outputs", [(33, 1), (1, 10)])
def test_complete_experiment_budget_rejects_before_hashing_or_headers(
    tmp_path, monkeypatch, native_sdk_double, rows, outputs,
):
    write_source_fixture(tmp_path)
    experiment = sdk_experiment(tmp_path, rows=rows, outputs=outputs)
    for name in ("_files", "_locked", "_store", "_tokenizer"):
        monkeypatch.setattr(source, name, forbidden)
    with pytest.raises(ValueError, match="limit"):
        source.admit_source(experiment)
    assert native_sdk_double["resolve"] == [experiment.canonical_bytes()]
    assert native_sdk_double["runtime"] == []


def test_sdk_runtime_bytes_and_native_decoder_digest_preserve_float_spelling(
    tmp_path, monkeypatch, native_sdk_double,
):
    write_source_fixture(tmp_path)
    original = _native.gkm_decoder_contract

    def native_spelling(*args):
        raw = original(*args)
        # V3 stores epsilon as a string. Add a numeric metadata probe to this
        # native test double so exponent re-encoding cannot pass unnoticed.
        # Real v3 lowering still supplies every geometry/semantic field.
        assert b',"input_bound":' in raw
        return raw.replace(b',"input_bound":',
                           b',"float_spelling_probe":1e-6,"input_bound":', 1)

    monkeypatch.setattr(_native, "gkm_decoder_contract", native_spelling)
    experiment = sdk_experiment(tmp_path)
    admitted = source.admit_source(experiment)
    native_bytes = _native.gkm_decoder_contract(admitted.config_bytes, 3, 2)
    assert admitted.contract_bytes == native_bytes
    # This bridge must preserve native bytes even when JSON equality hides a change.
    assert source._canonical(json.loads(native_bytes)) != native_bytes
    assert admitted.contract_digest == hashlib.sha256(
        b"pllm.gkm.decoder-reference.v1\0" + native_bytes
    ).hexdigest()
    assert admitted.runtime_contract_bytes is native_sdk_double["bytes"][0]
    raw = admitted.runtime_contract(experiment, 1, 1)
    assert raw is native_sdk_double["bytes"][-1]
    assert json.loads(raw)["decoder"] == json.loads(
        _native.gkm_decoder_contract(admitted.config_bytes, 1, 0)
    )
    assert native_sdk_double["runtime"][-1] == (
        experiment.canonical_bytes(), admitted.config_bytes, 1, 1,
        admitted.source_lock_digest, admitted.tokenizer_digest,
    )
    assert admitted.contract_bytes == native_bytes  # Owner retains full-budget pricing.
    assert admitted.configuration_digest == experiment.configuration_digest()
    assert admitted.composition_digest == experiment.pipeline.digest()


@pytest.mark.parametrize("rows, outputs", [(0, 1), (4, 1), (1, 4), (True, 1), (1, False)])
def test_sdk_actual_request_must_fit_exact_saved_budget(
    tmp_path, native_sdk_double, rows, outputs,
):
    write_source_fixture(tmp_path)
    experiment = sdk_experiment(tmp_path)
    admitted = source.admit_source(experiment)
    with pytest.raises(ValueError, match="budget"):
        admitted.runtime_contract(experiment, rows, outputs)
    assert len(native_sdk_double["runtime"]) == 1


def test_sdk_rejects_different_experiment_and_model_only_probe(tmp_path, native_sdk_double):
    write_source_fixture(tmp_path)
    experiment = sdk_experiment(tmp_path)
    admitted = source.admit_source(experiment)
    changed = replace(experiment, name="different-saved-experiment")
    with pytest.raises(ValueError, match="exact admitted Experiment"):
        admitted.runtime_contract(changed, 1, 1)
    probe = admit(tmp_path)
    assert probe.experiment_bytes is None and probe.runtime_contract_bytes is None
    with pytest.raises(ValueError, match="exact admitted Experiment"):
        probe.runtime_contract(experiment, 1, 1)
    assert len(native_sdk_double["runtime"]) == 1


@pytest.mark.parametrize("field", ["configuration", "composition", "source_lock", "tokenizer", "decoder"])
def test_native_sdk_binding_mismatch_rejects_before_tokenizer_construction(
    tmp_path, monkeypatch, native_sdk_double, field,
):
    write_source_fixture(tmp_path)
    original = _native.gkm_runtime_contract

    def altered(*args):
        document = json.loads(original(*args))
        if field == "decoder":
            document[field]["input_rows"] = 1
        else:
            document[field] = "0" * 64
        return source._canonical(document)

    monkeypatch.setattr(_native, "gkm_runtime_contract", altered)
    monkeypatch.setattr(source, "_tokenizer", forbidden)
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="differs from its source or Experiment"):
        source.admit_source(sdk_experiment(tmp_path))


@pytest.mark.skipif(not hasattr(_native, "gkm_runtime_contract"), reason="v3 core predates SDK binding")
def test_optional_real_native_sdk_contract(tmp_path):
    write_source_fixture(tmp_path)
    experiment = sdk_experiment(tmp_path)
    admitted = source.admit_source(experiment)
    assert admitted.runtime_contract(experiment, 1, 1) == _native.gkm_runtime_contract(
        experiment.canonical_bytes(), admitted.config_bytes, 1, 1,
        admitted.source_lock_digest, admitted.tokenizer_digest,
    )


@pytest.mark.parametrize("change", ["file_bytes", "encoded_bytes", "layers", "vocabulary", "depth"])
def test_geometry_and_config_limits_precede_ordinary_lowering_and_store(tmp_path, monkeypatch, change):
    cfg = decoder_config()
    if change == "file_bytes":
        payload = b" " * ((16 << 10) + 1)
    else:
        if change == "encoded_bytes":
            cfg["metadata"] = "😀" * 2000  # UTF-8 fits; canonical escaping does not.
        elif change == "layers":
            cfg["num_hidden_layers"] = 10**9
        elif change == "vocabulary":
            cfg["vocab_size"] = 258
        else:
            nested = []
            for _ in range(70):
                nested = [nested]
            cfg["metadata"] = nested
        payload = json.dumps(cfg, ensure_ascii=False).encode()
    (tmp_path / "config.json").write_bytes(payload)
    forbid_ordinary_loading(monkeypatch)
    with pytest.raises(ValueError):
        admit(tmp_path)


def test_tiny_full_byte_vocabulary_rejects_without_materialization(monkeypatch):
    monkeypatch.setattr(sources, "_materialize_tiny_model", forbidden)
    forbid_ordinary_loading(monkeypatch)
    with pytest.raises(ValueError, match="vocab|budget|limit"):
        source.admit_source(Model.tiny(), input_rows=1, decode_steps=0)


@pytest.mark.parametrize("change", ["extra_file", "extra_directory", "huge_tensor", "huge_tokenizer", "huge_record", "file_count"])
def test_file_admission_precedes_hashing_and_store(tmp_path, monkeypatch, change):
    write_source_fixture(tmp_path)
    if change == "extra_file":
        (tmp_path / "unpriced.bin").write_bytes(b"untrusted extra")
    elif change == "extra_directory":
        (tmp_path / "nested").mkdir()
    elif change == "file_count":
        for i in range(65):
            (tmp_path / f"extra-{i}.safetensors").touch()
    else:
        path = {"huge_tensor": "model.safetensors", "huge_tokenizer": "tokenizer.json",
                "huge_record": ".pllm-model.json"}[change]
        with (tmp_path / path).open("wb") as stream:
            stream.truncate(16 << 20)  # Sparse file: no large model allocation.
    monkeypatch.setattr(model_loader, "_source_lock", forbidden)
    monkeypatch.setattr(source, "SafeTensorStore", forbidden)
    with pytest.raises(ValueError, match="file|limit"):
        admit(tmp_path)


@pytest.mark.parametrize("change", ["missing", "extra", "rename", "shape", "dtype"])
def test_tensor_metadata_failures_precede_values(tmp_path, monkeypatch, change):
    _, _, arrays, _ = write_source_fixture(tmp_path)
    name = "model.layers.0.mlp.up_proj.weight"
    if change == "missing":
        arrays.pop(name)
    elif change == "extra":
        arrays["undeclared.weight"] = np.zeros((1,), dtype=np.float32)
    elif change == "rename":
        arrays["wrong.weight"] = arrays.pop(name)
    elif change == "shape":
        arrays[name] = arrays[name].T.copy()
    else:
        arrays[name] = arrays[name].astype(np.int32)
    save_file(arrays, tmp_path / "model.safetensors")
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="cardinality|missing|shape|dtype"):
        admit(tmp_path)


def test_sharded_index_is_complete_and_cannot_conceal_extra_tensors(tmp_path, monkeypatch):
    _, _, arrays, tensors = write_source_fixture(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    names = list(arrays)
    index = {}
    for number, subset in enumerate((names[::2], names[1::2])):
        shard = f"model-{number}.safetensors"
        save_file({name: arrays[name] for name in subset}, tmp_path / shard)
        index.update({name: shard for name in subset})
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": index}))
    assert admit(tmp_path).load_values().tensors == tensors
    save_file({"hidden.weight": np.zeros((1,), dtype=np.float32)}, tmp_path / "hidden.safetensors")
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="cardinality"):
        admit(tmp_path)


@pytest.mark.parametrize("change", ["external_path", "missing_key", "wrong_shard"])
def test_index_must_match_every_actual_header(tmp_path, monkeypatch, change):
    _, _, arrays, _ = write_source_fixture(tmp_path)
    index = dict.fromkeys(arrays, "model.safetensors")
    name = next(iter(index))
    if change == "missing_key":
        index.pop(name)
    else:
        index[name] = "../escape.safetensors" if change == "external_path" else "absent.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="index"):
        admit(tmp_path)


def test_oversize_safetensors_header_rejects_before_native_header_reader(tmp_path, monkeypatch):
    import safetensors

    write_source_fixture(tmp_path)
    (tmp_path / "model.safetensors").write_bytes((1 << 60).to_bytes(8, "little"))
    monkeypatch.setattr(safetensors, "safe_open", forbidden)
    with pytest.raises(ValueError, match="header.*limit"):
        admit(tmp_path)


@pytest.mark.parametrize("change", ["small_vocab", "extra_token", "noncontiguous", "context", "tokenizer_context", "truncate", "special_id"])
def test_complete_tokenizer_and_context_are_required(tmp_path, monkeypatch, change):
    cfg, _, _, _ = write_source_fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_bytes())
    if change == "small_vocab":
        data["model"]["vocab"].pop("word31")
    elif change == "extra_token":
        data["model"]["vocab"]["extra"] = 32
    elif change == "noncontiguous":
        data["model"]["vocab"]["word31"] = 32
    elif change in {"context", "special_id"}:
        cfg["max_position_embeddings" if change == "context" else "bos_token_id"] = 4 if change == "context" else 32
        (tmp_path / "config.json").write_text(json.dumps(cfg))
    elif change == "tokenizer_context":
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 4}))
    else:
        inner = Tokenizer.from_file(str(path))
        inner.enable_truncation(2)
        data = json.loads(inner.to_str())
    path.write_text(json.dumps(data))
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="vocabulary|context|position|truncat"):
        admit(tmp_path)


@pytest.mark.parametrize("added", [False, True])
def test_sparse_tokenizer_ids_reject_before_native_tokenizer_construction(tmp_path, monkeypatch, added):
    import tokenizers

    write_source_fixture(tmp_path)
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_bytes())
    if added:
        data["added_tokens"][0]["id"] = 10**9
    else:
        data["model"]["vocab"]["word31"] = 10**9
    path.write_text(json.dumps(data))

    class ForbiddenTokenizer:
        from_buffer = staticmethod(forbidden)

    monkeypatch.setattr(tokenizers, "Tokenizer", ForbiddenTokenizer)
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="vocabulary"):
        admit(tmp_path)


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "model.safetensors"])
def test_source_mutation_after_admission_rejects_before_values(tmp_path, monkeypatch, name):
    write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    path = tmp_path / name
    before = path.stat()
    data = path.read_bytes()
    # Preserve file length and mtime. ctime/inode/content locks must still bind it.
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    forbid_values(monkeypatch)
    with pytest.raises(ValueError, match="changed"):
        admitted.load_values()


@pytest.mark.parametrize("phase", ["contract", "lock"])
def test_source_mutation_during_admission_rejects_before_values(tmp_path, monkeypatch, phase):
    write_source_fixture(tmp_path)
    forbid_values(monkeypatch)
    if phase == "contract":
        original = source.decoder_contract

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = tmp_path / "config.json"
            path.write_bytes(path.read_bytes() + b" ")
            return result

        monkeypatch.setattr(source, "decoder_contract", changed)
    else:
        original = model_loader._source_lock

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            path = tmp_path / "tokenizer.json"
            path.write_bytes(path.read_bytes() + b" ")
            return result

        monkeypatch.setattr(model_loader, "_source_lock", changed)
    with pytest.raises(ValueError, match="changed"):
        admit(tmp_path)


def test_mutation_during_value_loading_never_returns_owner(tmp_path, monkeypatch):
    write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    get = source._AdmittedTensorStore.get
    touched = False

    def changed(self, *args, **kwargs):
        nonlocal touched
        value = get(self, *args, **kwargs)
        if not touched:
            touched = True
            path = tmp_path / "tokenizer_config.json"
            path.write_bytes(path.read_bytes() + b" ")
        return value

    monkeypatch.setattr(source._AdmittedTensorStore, "get", changed)
    with pytest.raises(ValueError, match="changed"):
        admitted.load_values()


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_rejects_when_values_are_loaded(tmp_path, value):
    _, _, arrays, _ = write_source_fixture(tmp_path)
    arrays["model.norm.weight"][0] = value
    save_file(arrays, tmp_path / "model.safetensors")
    admitted = admit(tmp_path)
    with pytest.raises(ValueError, match="invalid values"):
        admitted.load_values()


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.dtype("bfloat16")])
def test_supported_float_types_convert_to_immutable_little_endian_bytes(tmp_path, dtype):
    _, contract, arrays, _ = write_source_fixture(tmp_path)
    arrays = {name: value.astype(dtype) for name, value in arrays.items()}
    save_file(arrays, tmp_path / "model.safetensors")
    loaded = admit(tmp_path).load_values()
    assert loaded.tensors == {name: value.astype("<f4").tobytes() for name, value in arrays.items()}
    assert loaded.tensors[contract["table"]][:4] == bytes.fromhex("00000080")
    with pytest.raises(TypeError):
        loaded.tensors[contract["table"]] = b"changed"
    with pytest.raises(TypeError):
        loaded.coefficients["model.norm.weight"] = b"changed"


def test_existing_recorded_source_identity_is_verified_not_reinvented(tmp_path):
    write_source_fixture(tmp_path)
    _, _, lock = model_loader._source_lock(tmp_path, Model.path(str(tmp_path)))
    record = {"schema": 1, "repo_id": "trusted/fixture", "revision": "test",
              "commit": "a" * 40, "files": lock["files"]}
    path = tmp_path / ".pllm-model.json"
    path.write_text(json.dumps(record))
    admitted = admit(tmp_path)
    assert admitted.source_lock["repo_id"] == "trusted/fixture"
    assert admitted.source_lock["commit"] == "a" * 40
    record["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="digest mismatch"):
        admit(tmp_path)


def test_missing_absolute_source_never_resolves_as_hub_identifier(tmp_path, monkeypatch):
    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", forbidden)
    with pytest.raises(ValueError, match="existing local"):
        source.admit_source(Model.path(str(tmp_path / "missing")),
                            input_rows=1, decode_steps=0)


def test_shared_suffix_resolution_survives_bounded_metadata_import(tmp_path):
    _, _, arrays, tensors = write_source_fixture(tmp_path)
    save_file({"wrapped." + name: value for name, value in arrays.items()},
              tmp_path / "model.safetensors")
    assert admit(tmp_path).load_values().tensors == tensors


@pytest.mark.parametrize("change", ["grow", "truncate"])
def test_shared_hasher_uses_finite_reads_and_detects_mutation(tmp_path, monkeypatch, change):
    write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    target = tmp_path / "model.safetensors"
    original_open = Path.open
    reads = []

    class ChangingStream:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def read(self, count=-1):
            reads.append(count)
            assert len(reads) == 1, "hasher chased a mutated source tail"
            assert 0 < count <= min(64 << 10, dict(admitted._identities)[target.name][9])
            value = self.inner.read(count)
            with original_open(target, "ab" if change == "grow" else "wb") as writer:
                writer.write(b"changed")
            return value

    def opened(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return ChangingStream(stream) if path == target and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError, match="changed"):
        source._locked(admitted.path, admitted.model, admitted.source_float32_bytes,
                       admitted._identities)
    assert len(reads) == 1


def test_mutated_header_cannot_trigger_value_conversion_after_preflight(tmp_path, monkeypatch):
    write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    get = source._AdmittedTensorStore.get

    def changed(store, *args, **kwargs):
        path = tmp_path / "model.safetensors"
        with path.open("r+b") as stream:
            stream.write((1 << 60).to_bytes(8, "little"))
        return get(store, *args, **kwargs)

    monkeypatch.setattr(source._AdmittedTensorStore, "get", changed)
    monkeypatch.setattr(source.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="changed"):
        admitted.load_values()


@pytest.mark.parametrize("change", ["huge_offset", "gap", "overlap", "trailing"])
def test_malformed_tensor_ranges_reject_before_allocating_values(tmp_path, monkeypatch, change):
    write_source_fixture(tmp_path)
    path = tmp_path / "model.safetensors"
    raw = path.read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header, values = json.loads(raw[8:8 + size]), raw[8 + size:]
    names = [key for key in header if key != "__metadata__"]
    if change == "huge_offset":
        header[names[0]]["data_offsets"] = [i + (1 << 60) for i in header[names[0]]["data_offsets"]]
    elif change == "gap":
        header[names[0]]["data_offsets"] = [i + 4 for i in header[names[0]]["data_offsets"]]
    elif change == "overlap":
        begin, end = header[names[1]]["data_offsets"]
        header[names[1]]["data_offsets"] = [0, end - begin]
    else:
        values += b"x"
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + values)
    monkeypatch.setattr(source.np, "frombuffer", forbidden)
    with pytest.raises(ValueError, match="range"):
        admit(tmp_path)


def test_loaded_source_close_retires_views_and_context_owner(tmp_path):
    _, _, _, tensors = write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    loaded = admitted.load_values()
    values, coefficients = loaded.tensors, loaded.coefficients
    retained = values["model.norm.weight"]
    loaded.close()
    loaded.retire()
    assert values == {} and coefficients == {}
    assert retained == tensors["model.norm.weight"]  # Caller copies have no erasure promise.
    with pytest.raises(ValueError, match="retired"):
        loaded.tensors
    with pytest.raises(ValueError, match="retired"):
        loaded.coefficients
    with admitted.load_values() as owner:
        assert owner.tensors == tensors
    with pytest.raises(ValueError, match="retired"):
        owner.tensors


def test_loaded_source_public_constructor_owns_its_mapping_references(tmp_path):
    write_source_fixture(tmp_path)
    admitted = admit(tmp_path)
    loaded = admitted.load_values()
    tensors, coefficients = dict(loaded.tensors), dict(loaded.coefficients)
    other = source.LoadedSource(admission=admitted, tensors=tensors, coefficients=coefficients)
    tensors.clear()
    assert other.tensors == loaded.tensors
    other.close()
    assert coefficients == loaded.coefficients
