"""Opaque source provenance, strict composition and variable-S quote regressions."""
import hashlib
import json
import pickle

import pytest

from pllm import _native
from pllm.modeling import ForwardPlan, lower_forward_model


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def source(classifier=False):
    config = dict(model_type="bert", architectures=["BertForSequenceClassification" if classifier else "BertModel"],
                  vocab_size=17, hidden_size=8, num_hidden_layers=1, num_attention_heads=2,
                  intermediate_size=12, max_position_embeddings=16, type_vocab_size=2,
                  hidden_act="gelu", layer_norm_eps="1e-12", position_embedding_type="absolute",
                  torch_dtype="float32")
    if classifier:
        config["num_labels"] = 3
    return canonical(config)


def composition():
    return {"model": {"source": "fixture"}, "components": {
        "numeric": {"component": "pllm/source-float32/v1", "params": {}},
        "linear": {"component": "pllm/cleartext-linear", "params": {}},
        "topology": {"component": "pllm/client-only/v1", "params": {}},
        "kernels": {"component": "pllm/cpu", "params": {"threads": 1}},
    }}


def compile_contract(config=None, pipeline=None, **kwargs):
    return _native.ForwardContract.compile(
        source() if config is None else config,
        canonical(composition()) if pipeline is None else pipeline,
        max_sequence=kwargs.pop("max_sequence", 8),
        memory_bytes=kwargs.pop("memory_bytes", 1 << 30), **kwargs)


def test_forward_contract_opaque_and_original_source_provenance():
    with pytest.raises(TypeError):
        _native.ForwardContract()
    with pytest.raises(TypeError):
        type("Subclass", (_native.ForwardContract,), {})
    contract = compile_contract()
    assert type(contract) is _native.ForwardContract
    with pytest.raises(TypeError):
        pickle.dumps(contract)
    with pytest.raises((AttributeError, TypeError)):
        contract.memory_bytes = lambda: 1 << 63
    assert contract.source_config_bytes() == source()
    assert contract.composition_bytes() == canonical(composition())
    plan = json.loads(contract.plan_bytes())
    assert plan["schema_version"] == "pllm.forward_plan.v1"
    assert plan["numeric"] == "pllm.numeric.source_float32.v1"
    assert plan["output_contract"] == "pllm.forward.encoder_outputs.v1"
    assert set(plan["outputs"]) == {"last_hidden_state", "pooler_output"}
    assert contract.plan_digest() == hashlib.sha256(b"pllm.forward_plan.v1\0" + contract.plan_bytes()).hexdigest()
    schedule = json.loads(contract.schedule_bytes())
    assert schedule["schema_version"] == "pllm.forward_runtime_schedule.v1"
    assert schedule["complete"] is True and schedule["protected_execution"] is False
    assert [s["operation_id"] for s in schedule["steps"]] == [o["id"] for o in plan["operations"]]
    assert {s["executor"] for s in schedule["steps"]} == {"client_local_source_f32_v1"}
    assert contract.schedule_digest() == hashlib.sha256(b"pllm.forward_runtime_schedule.v1\0" + contract.schedule_bytes()).hexdigest()
    assert schedule["composition_digest"] == contract.composition_digest()
    assert schedule["numeric_digest"] == contract.numeric_digest()
    requirements = json.loads(contract.requirements_bytes())
    assert requirements["schema_version"] == "pllm.forward_requirements.v1"
    count = 0
    for tensor in requirements["tensors"]:
        assert tensor["dtype"] == "float32"
        n = 4
        for d in tensor["shape"]:
            n *= d
        count += n
    assert count == requirements["weight_bytes"]
    assert contract.max_sequence() == 8 and contract.memory_bytes() == 1 << 30


def test_forward_inspection_does_not_mint_execution_authority():
    plan = lower_forward_model(source(True), max_sequence=8)
    assert isinstance(plan, ForwardPlan)
    assert set(plan.outputs) == {"last_hidden_state", "pooler_output", "logits"}
    assert not hasattr(plan, "runtime_schedule")
    data = plan.to_dict()
    data["operations"][0]["attributes"]["weight"] = "forged"
    forged = ForwardPlan(canonical(data))
    assert forged.digest != plan.digest
    assert not hasattr(_native.ForwardContract, "from_plan")
    assert not hasattr(_native.ForwardContract, "from_schedule")
    with pytest.raises((ValueError, TypeError)):
        compile_contract(forged.canonical_bytes())


def test_noncanonical_source_preserves_raw_bytes_with_canonical_plan_identity():
    compact = source()
    document = json.loads(compact)
    reordered = dict(reversed(list(document.items())))
    raw = (" \n" + json.dumps(reordered, indent=2) + "\n\t").encode()
    assert raw != compact
    original = compile_contract(config=compact)
    alternate = compile_contract(config=raw)
    assert original.source_config_bytes() == compact
    assert alternate.source_config_bytes() == raw
    assert alternate.plan_bytes() == original.plan_bytes()
    assert alternate.plan_digest() == original.plan_digest()
    config_digest = hashlib.sha256(
        b"pllm.bert.forward_source_config.v1\0" + compact
    ).hexdigest()
    assert json.loads(alternate.plan_bytes())["config_digest"] == config_digest
    assert json.loads(original.plan_bytes())["config_digest"] == config_digest
    assert alternate.schedule_bytes() == original.schedule_bytes()
    assert lower_forward_model(raw, max_sequence=8).canonical_bytes() == original.plan_bytes()


@pytest.mark.parametrize("duplicate", [b'"hidden_size":8', b'"hidden_size":16'])
def test_duplicate_source_keys_are_refused_even_when_values_agree(duplicate):
    raw = source().replace(b'"hidden_size":8', b'"hidden_size":8,' + duplicate, 1)
    assert raw != source()
    with pytest.raises(ValueError):
        compile_contract(config=raw)
    with pytest.raises(ValueError):
        lower_forward_model(raw, max_sequence=8)


def test_duplicate_nested_source_keys_are_refused():
    raw = source(True)[:-1] + b',"id2label":{"0":"A","0":"B","1":"C","2":"D"}}'
    with pytest.raises(ValueError):
        compile_contract(config=raw)
    with pytest.raises(ValueError):
        lower_forward_model(raw, max_sequence=8)


def test_forward_inspection_is_deeply_immutable_and_dicts_are_detached():
    plan = lower_forward_model(source(True), max_sequence=8)
    before = plan.canonical_bytes()
    with pytest.raises((AttributeError, TypeError)):
        plan._canonical_bytes = b"forged"
    with pytest.raises(TypeError):
        plan.inputs["input.tokens"] = {}
    with pytest.raises(TypeError):
        plan.inputs["input.tokens"]["dtype"] = "float32"
    with pytest.raises(TypeError):
        plan.inputs["input.tokens"]["shape"][1] = 7
    with pytest.raises(TypeError):
        plan.operations[0] = {}
    with pytest.raises(TypeError):
        plan.operations[0]["attributes"]["weight_shape"][0] = 99
    with pytest.raises(TypeError):
        plan.operations[0]["attributes"]["semantic_config"]["architectures"][0] = "forged"
    with pytest.raises(TypeError):
        plan.outputs["logits"]["tensor"]["shape"][1] = 99
    detached = plan.to_dict()
    detached["operations"][0]["attributes"]["semantic_config"]["architectures"].append("forged")
    detached["inputs"]["input.tokens"]["shape"][1] = 7
    detached["outputs"]["logits"]["tensor"]["shape"][1] = 99
    assert plan.canonical_bytes() == before
    assert plan.inputs["input.tokens"]["shape"] == (1, "sequence")
    assert plan.outputs["logits"]["tensor"]["shape"] == (1, 3)
    assert plan.operations[0]["attributes"]["semantic_config"]["architectures"] == ("BertForSequenceClassification",)


def test_exact_ordered_41_tensor_classifier_requirements():
    config = json.loads(source(True))
    config["num_hidden_layers"] = 2
    contract = compile_contract(config=canonical(config))
    expected = {
        "bert.embeddings.word_embeddings.weight": [17, 8],
        "bert.embeddings.position_embeddings.weight": [16, 8],
        "bert.embeddings.token_type_embeddings.weight": [2, 8],
        "bert.embeddings.LayerNorm.weight": [8],
        "bert.embeddings.LayerNorm.bias": [8],
        "bert.pooler.dense.weight": [8, 8],
        "bert.pooler.dense.bias": [8],
        "classifier.weight": [3, 8],
        "classifier.bias": [3],
    }
    # Independent checkpoint-name/shape oracle, not copied from plan attributes.
    for layer in range(2):
        prefix = f"bert.encoder.layer.{layer}."
        for role in ("attention.self.query", "attention.self.key", "attention.self.value",
                     "attention.output.dense"):
            expected[prefix + role + ".weight"] = [8, 8]
            expected[prefix + role + ".bias"] = [8]
        for role in ("attention.output.LayerNorm", "output.LayerNorm"):
            expected[prefix + role + ".weight"] = [8]
            expected[prefix + role + ".bias"] = [8]
        expected[prefix + "intermediate.dense.weight"] = [12, 8]
        expected[prefix + "intermediate.dense.bias"] = [12]
        expected[prefix + "output.dense.weight"] = [8, 12]
        expected[prefix + "output.dense.bias"] = [8]
    assert len(expected) == 41
    requirements = json.loads(contract.requirements_bytes())
    assert set(requirements) == {"schema_version", "tensors", "weight_bytes"}
    assert requirements["schema_version"] == "pllm.forward_requirements.v1"
    assert requirements["tensors"] == [
        {"name": name, "dtype": "float32", "shape": shape}
        for name, shape in sorted(expected.items())
    ]
    assert type(requirements["weight_bytes"]) is int
    assert requirements["weight_bytes"] == 5836
    assert canonical(requirements) == contract.requirements_bytes()


def test_forward_quote_variable_sequence_boundary_and_weight_body():
    contract = compile_contract()
    small = json.loads(contract.quote(1))
    large = json.loads(contract.quote(8))
    assert small["sequence_length"] == 1 and large["sequence_length"] == 8
    assert small["required_bytes"] < large["required_bytes"]
    assert large["input_bytes"] == 8 * (8 + 8 + 8 + 1)
    assert large["retained_tensor_bytes"] > 8 * small["retained_tensor_bytes"] - 1000
    weights = json.loads(contract.requirements_bytes())["weight_bytes"]
    assert large["required_bytes"] == weights + sum(large[k] for k in (
        "input_bytes", "retained_tensor_bytes", "output_bytes", "workspace_bytes", "metadata_bytes"))
    assert type(large["metadata_bytes"]) is int and large["metadata_bytes"] > 0
    assert small["metadata_bytes"] == large["metadata_bytes"]
    exact = compile_contract(memory_bytes=large["required_bytes"])
    assert exact.quote(8) == contract.quote(8)
    larger_cap = compile_contract(memory_bytes=large["required_bytes"] + 1)
    assert larger_cap.quote(8) == contract.quote(8)
    assert larger_cap.quote(1) == contract.quote(1)
    with pytest.raises(ValueError):
        compile_contract(memory_bytes=large["required_bytes"] - 1)
    for bound in (0, 9, -1, 1 << 64, True, 1.0):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            contract.quote(bound)


@pytest.mark.parametrize("memory", [1, (8 << 20) - 1])
@pytest.mark.parametrize("malformed", ["source", "composition"])
def test_forward_public_tiny_cap_precedes_malformed_json_lexical_scan(memory, malformed):
    # Public classmethod, not the internal factory: malformed lexical bytes must
    # not win over the allocation-free backend floor in either input position.
    config = b"{not JSON" if malformed == "source" else source()
    pipeline = b"{not JSON" if malformed == "composition" else canonical(composition())
    with pytest.raises(ValueError, match="universal backend minimum"):
        compile_contract(config=config, pipeline=pipeline, memory_bytes=memory)


def test_forward_startup_abi_fact_ignores_later_host_metadata_changes(monkeypatch):
    import sys

    # Normal extension initialization already captured the real process ABI.
    # Mutable host metadata afterwards neither changes that ABI nor authorizes a
    # new one. Do not reset native state, prime a fake owner or replace authority.
    def inspections(owner):
        return tuple(getattr(owner, name)() for name in (
            "source_config_bytes", "composition_bytes", "plan_bytes", "schedule_bytes",
            "requirements_bytes", "plan_digest", "schedule_digest", "composition_digest",
            "numeric_digest")) + (owner.quote(1), owner.quote(8))

    expected = inspections(compile_contract())
    with monkeypatch.context() as patch:
        patch.setattr(sys, "abiflags", "unsupported", raising=False)
        patch.setattr(sys, "implementation", object())
        observed = inspections(compile_contract())
    assert observed == expected


def test_forward_public_compile_never_calls_installed_python_import_hook(monkeypatch):
    import builtins

    # All setup/imports precede the hook. Do not prime compile or its first
    # publication probe; use the genuine native classmethod without authority patches.
    raw = source()
    pipeline = canonical(composition())
    native_type = _native.ForwardContract
    compile_native = native_type.compile
    options = {"max_sequence": 8, "memory_bytes": 1 << 30}
    calls = []

    def forbid_import(name, *args, **kwargs):
        calls.append(name)
        raise AssertionError(f"native compile invoked Python import hook: {name}")

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", forbid_import)
        owner = compile_native(raw, pipeline, **options)
    assert type(owner) is native_type
    assert calls == []
    assert owner.source_config_bytes() == raw


@pytest.mark.parametrize("slot", ["quantization", "state", "scheduler", "preparation", "verification", "cache", "placement", "delivery", "unknown"])
def test_forward_raw_composition_rejects_extra_slots(slot):
    pipeline = composition()
    pipeline["components"][slot] = {"component": "unknown", "params": {}}
    with pytest.raises(ValueError):
        compile_contract(pipeline=canonical(pipeline))


@pytest.mark.parametrize("path", [(), ("model",), ("components", "numeric"),
                                  ("components", "kernels"), ("components", "numeric", "params")])
def test_forward_raw_composition_rejects_unknown_top_level_and_nested_keys(path):
    pipeline = composition()
    target = pipeline
    for key in path:
        target = target[key]
    target["unknown"] = True
    with pytest.raises(ValueError):
        compile_contract(pipeline=canonical(pipeline))


def test_forward_raw_composition_rejects_numeric_and_transport_tampering():
    for slot, field, value in (("numeric", "component", "pllm/source-float32/v2"),
                               ("numeric", "params", {"bits": 32}),
                               ("topology", "component", "pllm/two-online-offset-workers/v1"),
                               ("linear", "params", {"protected": True}),
                               ("kernels", "params", {"threads": 0})):
        pipeline = composition()
        pipeline["components"][slot][field] = value
        with pytest.raises(ValueError):
            compile_contract(pipeline=canonical(pipeline))
    with pytest.raises(ValueError):
        compile_contract(pipeline=json.dumps(composition()).encode())
    for bound in (0, -1, 1 << 64, True):
        with pytest.raises((ValueError, TypeError, OverflowError)):
            compile_contract(max_sequence=bound)
    with pytest.raises(ValueError):
        compile_contract(config=b" " * ((1 << 20) + 1))


def test_forward_source_geometry_refuses_nondivisible_hidden_width():
    # Geometry refusal, not an arithmetic-overflow test or architecture ceiling.
    bad = json.loads(source())
    bad["hidden_size"] = 9
    with pytest.raises(ValueError):
        compile_contract(config=canonical(bad))
