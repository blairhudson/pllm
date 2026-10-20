"""Generated checkpoints and explicit bridge doubles; no SDK serving qualification."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import struct

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from pllm import Deployment, ExecutionBudget, Model, _native, model_loader
from pllm.configuration import Experiment
from pllm.profiles import FuseFssSourceCpu
from pllm.runtime import fss_source as source
from pllm.runtime.safetensors_store import SafeTensorStore

GPT = "fusefss.gpt2-complete-ring50-q12.v1"
ROPE = "fusefss.dense-rope-ring50-q12.v1"
REAL_BRIDGE = callable(getattr(_native, "fss_source_contract", None)) and callable(getattr(_native, "FssSourceBuilder", None))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def inventory(config, profile):
    """Independent source-storage fixture: complete one-block GPT-2/Qwen3 shapes."""
    h, v = config.get("n_embd", config.get("hidden_size")), config["vocab_size"]
    if profile == GPT:
        shapes = {"transformer.wte.weight": [v, h], "transformer.wpe.weight": [config["n_positions"], h]}
        for norm in ("transformer.h.0.ln_1", "transformer.h.0.ln_2", "transformer.ln_f"):
            shapes.update({norm + ".weight": [h], norm + ".bias": [h]})
        for name, shape in (("attn.c_attn", [h, 3*h]), ("attn.c_proj", [h, h]),
                            ("mlp.c_fc", [h, 2*h]), ("mlp.c_proj", [2*h, h])):
            shapes["transformer.h.0." + name + ".weight"] = shape
            shapes["transformer.h.0." + name + ".bias"] = [shape[-1]]
        layouts = {name: "features" if len(shape) == 1 else
                   "rows_features" if name in {"transformer.wte.weight", "transformer.wpe.weight"} else
                   "input_output" for name, shape in shapes.items()}
    else:
        shapes = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h]}
        for norm in ("input_layernorm", "post_attention_layernorm"):
            shapes["model.layers.0." + norm + ".weight"] = [h]
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            shapes["model.layers.0.self_attn." + proj + ".weight"] = [h, h]
        for norm in ("q_norm", "k_norm"):
            shapes["model.layers.0.self_attn." + norm + ".weight"] = [h]
        for name, shape in (("gate_proj", [2*h, h]), ("up_proj", [2*h, h]), ("down_proj", [h, 2*h])):
            shapes["model.layers.0.mlp." + name + ".weight"] = shape
        layouts = {name: "features" if len(shape) == 1 else
                   "rows_features" if name == "model.embed_tokens.weight" else "output_input"
                   for name, shape in shapes.items()}
    if not config.get("tie_word_embeddings", True):
        shapes["lm_head.weight"], layouts["lm_head.weight"] = [v, h], "output_input"
    return shapes, layouts


def write_checkpoint(root, *, profile=GPT, vocabulary=32, tied=True, all_f32=False, all_dtype=None):
    root.mkdir(parents=True, exist_ok=True)
    config = {"model_type": "gpt2", "n_layer": 1, "n_embd": 4, "n_head": 1, "n_inner": 8,
              "n_positions": 16, "vocab_size": vocabulary, "activation_function": "gelu_new",
              "layer_norm_epsilon": 1e-5, "bos_token_id": 0, "eos_token_id": 1,
              "tie_word_embeddings": tied}
    if profile == ROPE:
        config = {"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 4,
                  "intermediate_size": 8, "num_attention_heads": 1, "num_key_value_heads": 1,
                  "head_dim": 4, "max_position_embeddings": 16, "vocab_size": vocabulary,
                  "hidden_act": "silu", "rms_norm_eps": 1e-6, "rope_theta": 10000,
                  "tie_word_embeddings": tied, "bos_token_id": 0, "eos_token_id": 1}
    (root / "config.json").write_text(json.dumps(config, indent=2))
    vocab = {"<bos>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"t{i}": i for i in range(3, vocabulary)})
    Tokenizer(WordLevel(vocab, unk_token="<unk>")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_bytes(canonical({"model_max_length": 16, "add_bos_token": False}))
    shapes, _ = inventory(config, profile)
    header, payload, originals = {}, bytearray(), {}
    for index, (name, shape) in enumerate(sorted(shapes.items())):
        dtype = all_dtype or ("F32" if all_f32 else ("F32", "F16", "BF16")[index % 3])
        raw = bytearray()
        for i in range(math.prod(shape)):
            value = 1.0 if len(shape) == 1 and name.endswith("weight") else (0.0, -0.0, 0.125, -0.125)[i % 4]
            if dtype == "BF16":
                raw.extend(struct.pack("<H", struct.unpack("<I", struct.pack("<f", value))[0] >> 16))
            else:
                raw.extend(struct.pack("<f" if dtype == "F32" else "<e", value))
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(payload), len(payload) + len(raw)]}
        payload.extend(raw)
        originals[name] = (dtype, bytes(raw))
    write_tensor_file(root / "model.safetensors", header, payload)
    return config, originals


def write_tensor_file(path, header, payload):
    encoded = canonical(header)
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def experiment(root, profile=GPT, rows=3, outputs=3):
    return Experiment(name="generated-fss-source", pipeline=FuseFssSourceCpu(Model.path(str(root)), numeric_profile=profile),
                      deployment=Deployment.local(root="local://fss-source-test"),
                      budget=ExecutionBudget(requests=1, max_input_tokens=rows, max_new_tokens=outputs))


@pytest.fixture
def bridge(monkeypatch):
    """Explicit metadata/import double, never a native numerical/serving claim."""
    class Bridge:
        owners = []
        events = []
        fail = None
        mutation = None
        contract_mutation = None
        manifest_mutation = None
        experiment = None

    class Program:
        closed = False

        def __init__(self, builder):
            self.builder = builder

        def close(self):
            self.closed = True

        def manifest(self, contract):
            if Bridge.fail == "manifest":
                raise ValueError("injected manifest error")
            record = json.loads(contract)
            data = {"schema": "pllm.fss.channel.program.v1", "contract": hashlib.sha256(contract).hexdigest(),
                    "input_digest": self.builder.digest.hexdigest(), "profile": record["numeric_profile"],
                    "source": "a"*64, "program": "b"*64, "initial": record["input_rows"],
                    "selections": record["output_tokens"], "vocabulary": record["vocabulary"]}
            if Bridge.manifest_mutation:
                Bridge.manifest_mutation(data)
            return canonical(data)

    class Builder:
        def __init__(self, config, rows, outputs, profile):
            config = json.loads(config)
            Bridge.events.append("geometry")
            if (rows + outputs - 1 > 16 or outputs > 4 or rows < 1 or outputs < 1
                    or config.get("n_embd", config.get("hidden_size")) * config["vocab_size"] > 32768):
                raise ValueError("native geometry bound")
            self.config, self.profile = config, profile
            self.rows, self.outputs = rows, outputs
            self.shapes, _ = inventory(config, profile)
            self.closed, self.frames, self.rotary_frames, self.program = False, [], [], None
            self.digest = hashlib.sha256(b"pllm.fss.original-ieee-stream.v1\0")
            Bridge.owners.append(self)

        def close(self):
            self.closed = True

        def append(self, name, dtype, offset, payload):
            Bridge.events.append("append")
            if Bridge.fail == "append":
                raise ValueError("injected append error")
            assert type(payload) is bytes and 0 < len(payload) <= 65536
            assert dtype in {"F32", "F16", "BF16"}
            if offset == 0:
                shape = self.shapes[name]
                self.digest.update(struct.pack("<Q", len(name.encode())) + name.encode())
                self.digest.update(bytes([{"F32": 0, "F16": 1, "BF16": 2}[dtype]]))
                self.digest.update(struct.pack("<Q", len(shape)))
                self.digest.update(b"".join(struct.pack("<Q", d) for d in shape))
            self.digest.update(payload)
            self.frames.append((name, dtype, offset, payload))
            if Bridge.mutation:
                Bridge.mutation()

        def append_rotary(self, operation, offset, cosine, sine):
            assert len(cosine) + len(sine) <= 65536
            if Bridge.fail == "rotary":
                raise ValueError("injected rotary error")
            if offset == 0:
                shape = [self.rows + self.outputs - 1, 4]
                self.digest.update(struct.pack("<Q", len(operation.encode())) + operation.encode())
                self.digest.update(struct.pack("<QQQ", 10000, *shape))
            for i in range(0, len(cosine), 4):
                self.digest.update(cosine[i:i+4] + sine[i:i+4])
            self.rotary_frames.append((operation, offset, cosine, sine))

        def seal(self):
            if Bridge.fail == "seal":
                raise ValueError("injected seal error")
            self.program = Program(self)
            return self.program

    def contract(encoded, config, rows, outputs, sourcepin, tokenizerpin):
        Bridge.events.append("pinned_contract")
        exp, config = Bridge.experiment, json.loads(config)
        root = Path(exp.pipeline.model.source)
        _, expected_pin, _ = model_loader._source_lock(root, exp.pipeline.model)
        assert sourcepin == expected_pin
        assert tokenizerpin == hashlib.sha256((root / "tokenizer.json").read_bytes()).hexdigest()
        assert encoded == exp.canonical_bytes()
        shapes, layouts = inventory(config, exp.pipeline.components["linear"].params["numeric_profile"])
        rotary_shapes, operations = {}, []
        if exp.pipeline.components["linear"].params["numeric_profile"] == ROPE:
            for operation in ("layer.0.key.rope", "layer.0.query.rope"):
                rotary_shapes[operation] = [rows + outputs - 1, 4]
                operations.append({"id": operation, "operator": "rotary_embedding", "output_shape": [1, 1, rows, 4],
                    "attributes": {"theta": 10000, "rotary_dimensions": 4, "pairing": "split_half",
                        "position_policy": "sequential_absolute", "coefficient_profile": "pllm.numeric.rope.q30.libm.v1",
                        "tail_policy": "unchanged", "frequency_scaling": None}})
        data = {"schema_version": "pllm.fss.sdk-contract.v1", "weights": "public", "source_lock": sourcepin,
                "tokenizer": tokenizerpin, "configuration": exp.configuration_digest(), "composition": exp.pipeline.digest(),
                "numeric_profile": exp.pipeline.components["linear"].params["numeric_profile"],
                "source_shapes": shapes, "source_layouts": layouts,
                "source_float32_bytes": sum(math.prod(s)*4 for s in shapes.values()),
                "rotary_shapes": rotary_shapes, "plan": {"prefill": {"operations": operations}},
                "input_rows": rows, "output_tokens": outputs, "maximum_sequence": rows + outputs - 1,
                "vocabulary": config["vocab_size"]}
        if Bridge.contract_mutation:
            Bridge.contract_mutation(data)
        return canonical(data)

    monkeypatch.setattr(_native, "resolve_experiment", lambda payload: Bridge.events.append("resolve"))
    monkeypatch.setattr(_native, "FssSourceBuilder", Builder, raising=False)
    monkeypatch.setattr(_native, "fss_source_contract", contract, raising=False)
    return Bridge


def admit(bridge, root, **kwargs):
    bridge.experiment = experiment(root, **kwargs)
    return source.resolve_client_source(bridge.experiment)


def forbidden(*args, **kwargs):
    raise AssertionError("unbounded or premature value read")


def test_missing_native_gate_before_filesystem(tmp_path, monkeypatch):
    monkeypatch.delattr(_native, "FssSourceBuilder", raising=False)
    monkeypatch.setattr(source.io, "_read_bounded", forbidden)
    with pytest.raises(ValueError, match="rebuild required"):
        source.resolve_client_source(experiment(tmp_path))


@pytest.mark.parametrize("rows,outputs", [(3, 3), (16, 2)])
def test_geometry_precedes_lock_hash_headers_and_tokenizer(tmp_path, monkeypatch, bridge, rows, outputs):
    write_checkpoint(tmp_path, vocabulary=8193 if rows == 3 else 32)
    monkeypatch.setattr(source.io, "_locked", forbidden)
    monkeypatch.setattr(source, "_store", forbidden)
    monkeypatch.setattr(source, "_tokenizer", forbidden)
    with pytest.raises(ValueError, match="native geometry bound"):
        admit(bridge, tmp_path, rows=rows, outputs=outputs)
    assert bridge.events == ["resolve", "geometry"]


def test_client_metadata_pins_original_bytes_without_value_decode(tmp_path, monkeypatch, bridge):
    write_checkpoint(tmp_path)
    for name in ("get", "get_slice", "get_linear", "iter_slices"):
        monkeypatch.setattr(SafeTensorStore, name, forbidden)
    admitted = admit(bridge, tmp_path)
    checkpoint, digest, _ = model_loader._source_lock(tmp_path, admitted.model)
    assert admitted.snapshot_pins == {"checkpoint": checkpoint, "source_lock": digest,
        "config": hashlib.sha256((tmp_path / "config.json").read_bytes()).hexdigest(),
        "tokenizer": hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest()}
    assert admitted.config_source_bytes != admitted.config_bytes
    assert admitted.tokenizer_descriptor["bos_token_id"] == 0
    assert all(owner.closed and not owner.frames for owner in bridge.owners)
    view = admitted.contract
    view["source_shapes"].clear()
    assert admitted.contract["source_shapes"]
    assert "append" not in bridge.events


@pytest.mark.parametrize("profile,tied", [(GPT, True), (GPT, False), (ROPE, True), (ROPE, False)])
def test_original_dtype_signed_zero_storage_orientation_and_pins(tmp_path, bridge, profile, tied):
    _, originals = write_checkpoint(tmp_path, profile=profile, tied=tied)
    admitted = admit(bridge, tmp_path, profile=profile)
    with source.load_role_source(bridge.experiment, expected_pins=admitted.snapshot_pins, source=admitted) as loaded:
        owner = loaded.native_program.builder
        assert [row[0] for row in owner.frames] == sorted(originals)
        for name, dtype, offset, payload in owner.frames:
            assert (dtype, payload) == originals[name] and offset == 0
        assert {row[1] for row in owner.frames} == {"F32", "F16", "BF16"}
        assert loaded.manifest["contract"] == hashlib.sha256(loaded.contract_bytes).hexdigest()
        assert loaded.manifest["input_digest"] == owner.digest.hexdigest()
        assert bool(owner.rotary_frames) == (profile == ROPE)
        descriptor = loaded.rotary_source
        assert len(descriptor["sha256"]) == 64
        assert bool(descriptor["tables"]) == (profile == ROPE)
        if profile == GPT:
            assert admitted.contract["source_layouts"]["transformer.h.0.attn.c_attn.weight"] == "input_output"
    assert owner.closed and loaded.native_program.closed
    loaded.close()


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
def test_large_tensor_streams_bounded_frames_without_full_reader(tmp_path, bridge, monkeypatch, dtype):
    _, originals = write_checkpoint(tmp_path, vocabulary=8192, all_dtype=dtype)
    admitted = admit(bridge, tmp_path)
    monkeypatch.setattr(SafeTensorStore, "get", forbidden)
    monkeypatch.setattr(source.io, "_source_float_frame", forbidden)
    with source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins) as loaded:
        frames = [row for row in loaded.native_program.builder.frames if row[0] == "transformer.wte.weight"]
        assert [row[2] for row in frames] == ([0, 16384] if dtype == "F32" else [0])
        assert all(len(row[3]) == 65536 for row in frames)
        assert b"".join(row[3] for row in frames) == originals["transformer.wte.weight"][1]


@pytest.mark.parametrize("pin", ["checkpoint", "source_lock", "tokenizer", "config"])
def test_worker_rejects_coordinator_original_pin_mismatch(tmp_path, bridge, pin):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    expected = admitted.snapshot_pins
    expected[pin] = "a"*64
    with pytest.raises(ValueError, match="original snapshot pins differ"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=expected)
    assert all(not owner.frames for owner in bridge.owners)


@pytest.mark.parametrize("rows,outputs", [(True, 2), (0, 2), (4, 2), (2, 4), (2, False)])
def test_actual_request_cannot_relax_or_exceed_saved_envelope(tmp_path, bridge, rows, outputs):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    with pytest.raises(ValueError, match="saved envelope"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins,
                               input_rows=rows, outputs=outputs)


@pytest.mark.parametrize("filename", ["config.json", "tokenizer.json", "model.safetensors"])
def test_snapshot_mutation_before_payload_rejected(tmp_path, bridge, filename):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    path = tmp_path / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed|differ"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins)
    assert all(not owner.frames for owner in bridge.owners)


@pytest.mark.parametrize("stage", ["append", "rotary", "seal", "manifest"])
def test_native_errors_close_pending_and_published_owners(tmp_path, bridge, stage):
    write_checkpoint(tmp_path, profile=ROPE)
    admitted = admit(bridge, tmp_path, profile=ROPE)
    bridge.fail = stage
    with pytest.raises(ValueError, match="injected"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins)
    assert all(owner.closed for owner in bridge.owners)
    assert all(owner.program is None or owner.program.closed for owner in bridge.owners)


def test_cancel_mid_import_closes_owner(tmp_path, bridge):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    with pytest.raises(ValueError, match="cancelled"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins,
                               cancelled=lambda: "append" in bridge.events)
    assert bridge.owners[-1].frames and bridge.owners[-1].closed


def test_observed_mutation_mid_import_closes_owner(tmp_path, bridge):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    def mutate():
        bridge.mutation = None
        path = tmp_path / "model.safetensors"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    bridge.mutation = mutate
    with pytest.raises(ValueError, match="changed"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins)
    assert bridge.owners[-1].frames and bridge.owners[-1].closed


def test_config_replacement_cannot_keep_original_source_pin(tmp_path, bridge):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    changed = json.loads(admitted.config_bytes)
    changed["activation_function"] = "other"
    modified = replace(admitted, config_bytes=canonical(changed))
    with pytest.raises(ValueError, match="canonical config differs"):
        source.load_role_source(bridge.experiment, source=modified, expected_pins=admitted.snapshot_pins)


@pytest.mark.parametrize("profile", [GPT, ROPE])
def test_actual_request_contract_and_coefficient_geometry(tmp_path, bridge, profile):
    write_checkpoint(tmp_path, profile=profile)
    admitted = admit(bridge, tmp_path, profile=profile)
    with source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins,
                                input_rows=2, outputs=2) as loaded:
        actual = json.loads(loaded.contract_bytes)
        assert actual["input_rows"] == 2 and actual["output_tokens"] == 2
        assert actual["maximum_sequence"] == 3
        assert all(table["shape"] == [3, 4] for table in loaded.rotary_source["tables"])
        assert admitted.contract["maximum_sequence"] == 5


@pytest.mark.parametrize("invalid", [False, True])
def test_sharded_inventory_uses_exact_index_not_external_paths(tmp_path, bridge, invalid):
    write_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header, payload = json.loads(data[8:8+size]), data[8+size:]
    locations = {}
    for i in range(2):
        shard, body = {}, bytearray()
        filename = f"model-{i}.safetensors"
        for j, (name, item) in enumerate(sorted(header.items())):
            if j % 2 == i:
                begin, stop = item["data_offsets"]
                shard[name] = {**item, "data_offsets": [len(body), len(body) + stop - begin]}
                body.extend(payload[begin:stop])
                locations[name] = filename
        write_tensor_file(tmp_path / filename, shard, body)
    path.unlink()
    if invalid:
        locations[next(iter(locations))] = "../external.safetensors"
    (tmp_path / "model.safetensors.index.json").write_bytes(canonical({"weight_map": locations}))
    if invalid:
        with pytest.raises(ValueError, match="shard index differs"):
            admit(bridge, tmp_path)
    else:
        admitted = admit(bridge, tmp_path)
        with source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins) as loaded:
            assert len(loaded.native_program.builder.frames) == len(header)


def test_hub_resolver_retains_original_model_and_ignores_unread_docs(tmp_path, bridge, monkeypatch):
    from pllm.runtime.hf_hub import ResolvedModelSource
    write_checkpoint(tmp_path)
    (tmp_path / "README.md").write_bytes(b"inert documentation")
    model = Model.hf("generated/fss", revision="a"*40, local_files_only=True)
    exp = replace(experiment(tmp_path), pipeline=FuseFssSourceCpu(model, numeric_profile=GPT))
    bridge.experiment = exp
    calls = []
    def resolve(name, **kwargs):
        calls.append((name, kwargs))
        return ResolvedModelSource(tmp_path, name, name, kwargs["revision"])
    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", resolve)
    # The double authenticates the actual locally resolved checkpoint, independent
    # of the public model identifier used in the immutable Experiment.
    # Source locks preserve the original Model, so test this resolver at the
    # metadata boundary without the path-specific contract double's assertion.
    def hf_contract(encoded, config, rows, outputs, pin, tokenpin):
        shapes, layouts = inventory(json.loads(config), GPT)
        assert pin == model_loader._source_lock(tmp_path, model)[1]
        return canonical({"schema_version": "pllm.fss.sdk-contract.v1", "weights": "public",
            "configuration": exp.configuration_digest(), "composition": exp.pipeline.digest(),
            "source_lock": pin, "tokenizer": tokenpin, "numeric_profile": GPT,
            "source_shapes": shapes, "source_layouts": layouts,
            "source_float32_bytes": sum(math.prod(s)*4 for s in shapes.values()), "rotary_shapes": {},
            "plan": {"prefill": {"operations": []}}, "input_rows": rows, "output_tokens": outputs,
            "maximum_sequence": rows+outputs-1, "vocabulary": 32})
    monkeypatch.setattr(_native, "fss_source_contract", hf_contract)
    admitted = source.resolve_client_source(exp)
    assert admitted.model == model and admitted.path == tmp_path
    assert calls == [(model.source, {"model_id": None, "revision": "a"*40, "local_files_only": True})]


def test_quantized_config_and_oversized_config_reject_before_geometry(tmp_path, bridge, monkeypatch):
    config, _ = write_checkpoint(tmp_path)
    config["quantization_config"] = {"bits": 4}
    (tmp_path / "config.json").write_bytes(canonical(config))
    monkeypatch.setattr(source, "_geometry", forbidden)
    with pytest.raises(ValueError, match="unquantized"):
        admit(bridge, tmp_path)
    (tmp_path / "config.json").write_bytes(b" " * 16385)
    with pytest.raises(ValueError, match="byte limit"):
        admit(bridge, tmp_path)


@pytest.mark.parametrize("kind", ["padding", "truncation", "sparse_ids", "incomplete", "null_model"])
def test_tokenizer_malformed_or_incomplete_rejects_before_source_payload(tmp_path, bridge, kind):
    write_checkpoint(tmp_path)
    path = tmp_path / "tokenizer.json"
    doc = json.loads(path.read_bytes())
    if kind == "padding":
        doc["padding"] = {"strategy": {"Fixed": 1000000000}}
    elif kind == "truncation":
        doc["truncation"] = {"max_length": 1}
    elif kind == "sparse_ids":
        doc["model"]["vocab"]["t3"] = 1000000000
    elif kind == "incomplete":
        del doc["model"]["vocab"]["t31"]
    else:
        doc["model"] = None
    path.write_bytes(canonical(doc))
    with pytest.raises(ValueError, match="tokenizer|complete"):
        admit(bridge, tmp_path)
    assert all(owner.closed and not owner.frames for owner in bridge.owners)


@pytest.mark.parametrize("changed", ["revision", "blob"])
def test_hub_original_snapshot_revision_and_blob_pin_checked(tmp_path, bridge, monkeypatch, changed):
    from pllm.runtime.hf_hub import ResolvedModelSource
    commit = "a"*40
    repo = tmp_path / "models--generated--fss"
    snapshot, blobs = repo / "snapshots" / commit, repo / "blobs"
    blobs.mkdir(parents=True)
    write_checkpoint(snapshot)
    for path in snapshot.iterdir():
        raw = path.read_bytes()
        target = blobs / hashlib.sha256(raw).hexdigest()
        path.rename(target)
        path.symlink_to(target)
    if changed == "blob":
        path = (snapshot / "model.safetensors").resolve()
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    model = Model.hf("generated/fss", revision="b"*40 if changed == "revision" else commit, local_files_only=True)
    exp = replace(experiment(snapshot), pipeline=FuseFssSourceCpu(model, numeric_profile=GPT))
    bridge.experiment = exp
    monkeypatch.setattr("pllm.runtime.hf_hub.resolve_huggingface_source", lambda *a, **kw:
                        ResolvedModelSource(snapshot, model.source, model.source, commit))
    with pytest.raises(ValueError, match="pinned revision" if changed == "revision" else "blob digest"):
        source.resolve_client_source(exp)
    assert all(owner.closed and not owner.frames for owner in bridge.owners)


@pytest.mark.parametrize("field", ["contract", "input_digest", "profile", "source", "program", "initial"])
def test_manifest_cannot_substitute_contract_ieee_stream_or_geometry(tmp_path, bridge, field):
    write_checkpoint(tmp_path)
    admitted = admit(bridge, tmp_path)
    bridge.manifest_mutation = lambda doc: doc.update({field: 999 if field == "initial" else "0"*64})
    with pytest.raises(ValueError, match="manifest differs"):
        source.load_role_source(bridge.experiment, source=admitted, expected_pins=admitted.snapshot_pins)
    assert bridge.owners[-1].program.closed and bridge.owners[-1].closed


@pytest.mark.parametrize("mutation", ["transpose", "dtype", "extra", "gap", "bool_shape"])
def test_header_malformations_reject_without_decode(tmp_path, bridge, monkeypatch, mutation):
    write_checkpoint(tmp_path)
    path = tmp_path / "model.safetensors"
    data = path.read_bytes()
    size = int.from_bytes(data[:8], "little")
    header, payload = json.loads(data[8:8+size]), data[8+size:]
    row = header["transformer.h.0.attn.c_attn.weight"]
    if mutation == "transpose":
        row["shape"] = list(reversed(row["shape"]))
    elif mutation == "dtype":
        row["dtype"] = "I32"
    elif mutation == "extra":
        header["hidden.extra"] = row
    elif mutation == "bool_shape":
        row["shape"][0] = True
    else:
        row["data_offsets"][0] += 1
    write_tensor_file(path, header, payload)
    monkeypatch.setattr(SafeTensorStore, "get", forbidden)
    with pytest.raises(ValueError, match="FuseFSS source"):
        admit(bridge, tmp_path)
    assert all(not owner.frames for owner in bridge.owners)


@pytest.mark.parametrize("attribute,value", [("theta", 0), ("frequency_scaling", {"factor": 2}),
    ("rotary_dimensions", 2), ("pairing", "interleaved"), ("coefficient_profile", "caller-supplied")])
def test_rotary_unapproved_attributes_reject(tmp_path, bridge, attribute, value):
    write_checkpoint(tmp_path, profile=ROPE)
    bridge.contract_mutation = lambda doc: doc["plan"]["prefill"]["operations"][0]["attributes"].update({attribute: value})
    with pytest.raises(ValueError, match="approved unscaled full split-half"):
        admit(bridge, tmp_path, profile=ROPE)


def test_rotary_matches_all_frozen_example_f32_words_without_executing_oracle():
    # Inert frozen specification data only. No paper/author generator imports.
    cos_words = ["3f800000 3f800000", "3f0a5140 3f7ffcb9", "bed51133 3f7ff2e5",
                 "bf7d7026 3f7fe283", "bf275530 3f7fcb94", "3e913c2c 3f7fae19"]
    sin_words = ["00000000 00000000", "3f576aa4 3c23d657", "3f68c7b7 3ca3d43e",
                 "3e1081c3 3cf5b920", "bf41bdcf 3d23cbdb", "bf757c10 3d4cb6f5"]
    _, cosine, sine = next(source._rotary_frames([6, 4], 10000))
    def expected(rows):
        return b"".join(struct.pack("<I", int(word, 16)) for row in rows for word in row.split()*2)
    assert cosine == expected(cos_words)
    assert sine == expected(sin_words)


@pytest.mark.skipif(not REAL_BRIDGE, reason="requires rebuilt wheel with actual FssSourceBuilder/contract bridge")
def test_rebuilt_native_generated_gpt2_import(tmp_path):
    """Real native numeric admission, not network transport or SDK serving."""
    write_checkpoint(tmp_path, all_f32=True)
    exp = experiment(tmp_path)
    metadata = source.resolve_client_source(exp)
    with source.load_role_source(exp, source=metadata, expected_pins=metadata.snapshot_pins) as loaded:
        assert loaded.manifest["contract"] == hashlib.sha256(loaded.contract_bytes).hexdigest()
        for malformed in (b"x", b"{}", b"not-json"):
            with pytest.raises(ValueError):
                loaded.native_program.manifest(malformed)
        for key, wrong in (("numeric_profile", ROPE), ("input_rows", 15),
                           ("width", 999), ("setup_owner", "client"), ("persistent_kv", True)):
            changed = json.loads(loaded.contract_bytes)
            changed[key] = wrong
            with pytest.raises(ValueError):
                loaded.native_program.manifest(canonical(changed))


@pytest.mark.skipif(not REAL_BRIDGE, reason="requires rebuilt native audit-fixed source bridge")
@pytest.mark.parametrize("layers", [36, 1024])
def test_rebuilt_native_rejects_metadata_amplification_before_any_append(layers):
    """Native gate evidence only when rebuilt; no mocked/native serving claim."""
    config = {"model_type": "qwen2", "hidden_size": 2, "intermediate_size": 2,
              "num_hidden_layers": layers, "num_attention_heads": 1, "num_key_value_heads": 1,
              "vocab_size": 3, "max_position_embeddings": 16, "hidden_act": "silu",
              "rope_theta": 10000, "rms_norm_eps": "0.000001" + "0" * 15000,
              "tie_word_embeddings": True}
    encoded = canonical(config)
    assert len(encoded) <= 16 * 1024
    with pytest.raises(ValueError, match="string"):
        _native.FssSourceBuilder(encoded, 1, 1, ROPE)
