"""Normal source binding keeps local positions separate from token boundaries."""

import asyncio
import dataclasses
import hashlib
import json

import numpy as np
import pytest

import pllm
from pllm.modeling import ModelPlan
from pllm.profiles import ClientOnlyCpu
from pllm.quantization import SymmetricPerRow
from pllm.runtime.loaders import load_hf_directory
from pllm.runtime.bundle_storage import immutable_f32_array
from pllm.runtime.model_binding import LOCAL_OPERATORS, RuntimeBindingError, compile_runtime_model
from pllm.runtime.semantic_tensors import SemanticTensorError, lookup_contract, required_client_tensors
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine

from absolute_position_decoder_oracle import tiny_checkpoint, write_checkpoint


@pytest.fixture
def source_binding(tmp_path):
    config, weights = tiny_checkpoint()
    write_checkpoint(tmp_path, config, weights)
    model_id = "semantic-local-routing"
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=1)
    asyncio.run(engine.load(load_hf_directory(tmp_path, model_id=model_id)))
    bundle = ClientBundle.unpack(engine.client_bundle(model_id, placement="client"))
    plan = pllm.lower_model(config, batch=1, max_input_tokens=6, max_new_tokens=3)
    composition = ClientOnlyCpu(pllm.Model(model_id),
                                quantization=SymmetricPerRow(weight_bits=8, activation_bits=8))
    return plan, bundle, composition, weights


def test_positions_local_tokens_still_stage_owned(source_binding):
    plan, bundle, composition, weights = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    assert "layer_norm" in LOCAL_OPERATORS
    assert "token_lookup" not in LOCAL_OPERATORS
    schedule = plan.runtime_schedule(composition).to_dict()
    for phase in ("prefill", "decode"):
        position = next(step for step in schedule[phase]["steps"]
                        if "position_lookup" in step["operation_ids"])
        assert position["operation_ids"] == ["position_lookup"]
        assert position["executor"] == "client_local" and position["weight_ids"] == []
        assert f"{phase}:position_lookup" in compiled.local_operations
        assert f"{phase}:token_lookup" not in compiled.local_operations
        token = next(row for row in compiled.stage_bindings if row.role == "token_lookup")
        assert f"{phase}:token_lookup" in token.semantic_operations
        assert not any(f"{phase}:position_lookup" in row.semantic_operations
                       for row in compiled.stage_bindings)
    local = next(row for row in compiled.to_spec()["local_tensors"]
                 if row["weight_id"] == "transformer.wpe.weight")
    assert local["shape"] == [8, 4]  # Not vocabulary cardinality 11.
    assert local["sha256"] == hashlib.sha256(weights[local["weight_id"]].tobytes()).hexdigest()


@pytest.mark.parametrize("damage", ["missing", "identity", "shape", "dtype", "nonfinite"])
def test_position_artifact_rejected_before_binding(source_binding, damage):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    key = next(row["key"] for row in compiled.to_spec()["local_tensors"]
               if row["weight_id"] == "transformer.wpe.weight")
    arrays = dict(bundle.arrays)
    if damage == "missing":
        del arrays[key]
    elif damage == "identity":
        arrays["unrelated.position.weight"] = arrays.pop(key)
    elif damage == "shape":
        arrays[key] = np.zeros((11, 4), np.float32)
    elif damage == "dtype":
        arrays[key] = arrays[key].astype(np.float64)
    else:
        arrays[key] = np.full((8, 4), np.nan, np.float32)
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(plan, dataclasses.replace(bundle, arrays=arrays), composition=composition)


def test_position_digest_mutation_invalidates_binding(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    key = next(row["key"] for row in compiled.to_spec()["local_tensors"]
               if row["weight_id"] == "transformer.wpe.weight")
    changed = bundle.arrays[key].copy()
    changed[0, 0] += np.float32(0.25)
    bundle.arrays[key] = immutable_f32_array(changed.tobytes(), changed.shape)
    with pytest.raises(RuntimeBindingError, match="changed since compilation"):
        compiled.validate()


@pytest.mark.parametrize("domain,source", [("unknown", "input.positions"), ("segments", "input.segments"),
                                           ("absolute_positions", "input.tokens")])
def test_invalid_lookup_domains_reject_shared_tensor_admission(source_binding, domain, source):
    plan, bundle, composition, _ = source_binding
    document = plan.to_dict()
    for phase in ("prefill", "decode"):
        operation = next(op for op in document[phase]["operations"] if op["id"] == "position_lookup")
        operation["attributes"]["index_domain"] = domain
        operation["inputs"] = [source]
    forged = ModelPlan(json.dumps(document).encode())
    with pytest.raises(SemanticTensorError, match="domain|segments"):
        required_client_tensors(forged)
    with pytest.raises(RuntimeBindingError):
        compile_runtime_model(forged, bundle, composition=composition)


def test_lookup_contract_keeps_position_and_token_cardinality_distinct(source_binding):
    plan, _, _, _ = source_binding
    operations = plan.to_dict()["prefill"]["operations"]
    position = next(op for op in operations if op["id"] == "position_lookup")
    token = next(op for op in operations if op["id"] == "token_lookup")
    assert lookup_contract(position, legacy_vocabulary=11) == ("absolute_positions", "input.positions", (8, 4))
    assert lookup_contract(token, legacy_vocabulary=11) == ("tokens", "input.tokens", (11, 4))


def test_alphabet_descriptor_binds_original_identity_and_token_ids(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    tokenizer = bundle.tokenizer()
    assert tokenizer.vocab_size == 11
    assert tokenizer.bos_token_id == 9 and tokenizer.eos_token_id == 10
    assert tokenizer.encode(" ab") == [0, 1, 2]
    assert tokenizer.encode(" ab", add_bos=True) == [9, 0, 1, 2]
    assert tokenizer.encode("?") == [0]
    expected = hashlib.sha256(json.dumps(
        bundle.tokenizer_descriptor, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    assert compiled.tokenizer_digest == expected
    assert compile_runtime_model(plan, bundle, composition=composition).digest == compiled.digest


def test_alphabet_position_swap_invalidates_compiled_identity(source_binding):
    plan, bundle, composition, _ = source_binding
    compiled = compile_runtime_model(plan, bundle, composition=composition)
    descriptor = bundle.tokenizer_descriptor
    original = descriptor["alphabet"]
    try:
        descriptor["alphabet"] = original[1] + original[0] + original[2:]
        with pytest.raises(RuntimeBindingError, match="changed since compilation"):
            compiled.validate()
    finally:
        descriptor["alphabet"] = original
    compiled.validate()


def _alphabet_descriptor():
    return {"type": "alphabet", "alphabet": " abcdefgh"}


@pytest.mark.parametrize("damage", [
    "unsupported-kind", "non-string", "duplicate", "wrong-length", "too-small",
    "conflicting-discriminator", "nonliteral-discriminator", "bool-vocab", "wrong-vocab",
    "bool-bos", "wrong-bos", "bool-eos", "wrong-eos", "bool-pad", "negative-pad",
    "large-pad", "null-pad", "source-pad", "unknown-field", "bad-template", "non-string-key",
])
def test_invalid_alphabet_descriptor_rejects_before_hash(monkeypatch, damage):
    import pllm.runtime.model_binding as binding

    descriptor = _alphabet_descriptor()
    runtime = {"vocab_size": 11, "bos_token_id": 9, "eos_token_id": 10}
    source = {"pad_token_id": 10}
    if damage == "unsupported-kind":
        descriptor["type"] = "unsupported"
    elif damage == "non-string":
        descriptor["alphabet"] = list(descriptor["alphabet"])
    elif damage == "duplicate":
        descriptor["alphabet"] = " abcdefgg"
    elif damage == "wrong-length":
        descriptor["alphabet"] += "i"
    elif damage == "too-small":
        descriptor["alphabet"] = "a"
        runtime["vocab_size"] = 3
    elif damage == "conflicting-discriminator":
        descriptor["kind"] = "byte"
    elif damage == "nonliteral-discriminator":
        descriptor["kind"] = 123
    elif damage in {"bool-vocab", "wrong-vocab", "bool-bos", "wrong-bos", "bool-eos", "wrong-eos"}:
        field = {"vocab": "vocab_size", "bos": "bos_token_id", "eos": "eos_token_id"}[damage.split("-")[1]]
        descriptor[field] = True if damage.startswith("bool") else runtime[field] - 1
    elif damage in {"bool-pad", "negative-pad", "large-pad", "null-pad", "source-pad"}:
        descriptor["pad_token_id"] = {
            "bool-pad": True, "negative-pad": -1, "large-pad": 11,
            "null-pad": None, "source-pad": 0,
        }[damage]
    elif damage == "unknown-field":
        descriptor["unbound_setting"] = None
    elif damage == "bad-template":
        descriptor["chat_template"] = []
    else:
        descriptor[1] = "not a string key"

    def forbidden_hash_input(_):
        pytest.fail("invalid alphabet descriptor reached canonical hashing")

    monkeypatch.setattr(binding, "_canonicalize_descriptor", forbidden_hash_input)
    with pytest.raises(RuntimeBindingError):
        binding._tokenizer_digest(descriptor, runtime, source_config=source)


def test_alphabet_length_bound_precedes_constructor_and_uniqueness(monkeypatch):
    import pllm.runtime.tokenizer as tokenizers
    from pllm.runtime.model_binding import _tokenizer_digest

    def forbidden_constructor(_):
        pytest.fail("out-of-vocabulary alphabet reached uniqueness constructor")

    monkeypatch.setattr(tokenizers, "AlphabetTokenizer", forbidden_constructor)
    descriptor = {"type": "alphabet", "alphabet": "a" * 12}
    with pytest.raises(RuntimeBindingError, match="length"):
        _tokenizer_digest(descriptor, {"vocab_size": 11, "bos_token_id": 9, "eos_token_id": 10})


@pytest.mark.parametrize("discriminators", [{"kind": "alphabet"}, {"type": "alphabet"},
                                            {"kind": "alphabet", "type": "alphabet"}])
@pytest.mark.parametrize("template", [None, "{{ messages }}"])
def test_alphabet_optional_fields_remain_in_original_hash(discriminators, template):
    from pllm.runtime.model_binding import _tokenizer_digest

    descriptor = {**discriminators, "alphabet": " abcdefgh", "chat_template": template,
                  "pad_token_id": 0}
    runtime = {"vocab_size": 11, "bos_token_id": 9, "eos_token_id": 10}
    expected = hashlib.sha256(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # PAD need not equal EOS. Compare concrete source PAD only when declared.
    assert _tokenizer_digest(descriptor, runtime, source_config={"pad_token_id": 0}) == expected
    assert _tokenizer_digest(descriptor, runtime, source_config={}) == expected


@pytest.mark.parametrize("key,value", [("vocab_size", 12), ("bos_token_id", 0), ("eos_token_id", 1)])
def test_alphabet_derived_ids_must_match_runtime_even_when_descriptor_omits_ids(key, value):
    from pllm.runtime.model_binding import _tokenizer_digest

    runtime = {"vocab_size": 11, "bos_token_id": 9, "eos_token_id": 10, key: value}
    with pytest.raises(RuntimeBindingError):
        _tokenizer_digest(_alphabet_descriptor(), runtime)


def test_alphabet_rejects_nonliteral_values_without_coercion():
    from pllm.runtime.model_binding import _tokenizer_digest

    class NotLiteral(str):
        def __str__(self):
            pytest.fail("alphabet validation coerced a nonliteral string")

    class NotOrdinary(dict):
        def get(self, *args):
            pytest.fail("alphabet validation called a custom mapping getter")

    runtime = {"vocab_size": 11, "bos_token_id": 9, "eos_token_id": 10}
    for descriptor in (
        {"type": NotLiteral("alphabet"), "alphabet": " abcdefgh"},
        {"type": "alphabet", "alphabet": NotLiteral(" abcdefgh")},
        NotOrdinary(_alphabet_descriptor()),
    ):
        with pytest.raises(RuntimeBindingError):
            _tokenizer_digest(descriptor, runtime)
