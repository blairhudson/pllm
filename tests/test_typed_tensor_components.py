"""Installed typed declarations only; no binding or native execution claims."""

import inspect
import json
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from pllm.configuration import ComponentRef, ConfigurationError
from pllm.nonlinear import F32Activation
from pllm.operators import (
    F32RmsNorm, F32Rotary, F32CausalAttention, F32Multiply,
    F32AddBias, F32Add, FullVocabularyHead, GreedySelection,
)


# One row per unique class, independent of paper recipes and descriptors.
CASES = (
    (F32RmsNorm, "pllm/f32-rmsnorm/v1", {"epsilon": "source"}, {}),
    (F32Rotary, "pllm/f32-rotary/v1", {"configuration": "source"}, {}),
    (F32CausalAttention, "pllm/f32-causal-attention/v1", {"heads": "source", "kv": "state"}, {}),
    (F32Activation, "pllm/f32-activation/v1", {"function": "source"}, {}),
    (F32Multiply, "pllm/f32-multiply/v1", {}, {}),
    (F32AddBias, "pllm/f32-add-bias/v1", {"placement": "after-reconstruction"}, {}),
    (F32Add, "pllm/f32-add/v1", {}, {}),
    (FullVocabularyHead, "pllm/full-vocabulary-head/v1", {"linear": "linear"},
     {"rows": "last-prefill-and-each-decode"}),
    (GreedySelection, "pllm/greedy-selection/v1", {"eos": "continue"}, {"logits": "complete"}),
)


class StringSubclass(str):
    pass


def _wire_bytes(component):
    # ComponentRef has no canonical_bytes method. Compare canonical JSON of
    # component specs directly, without constructing/admitting a Pipeline.
    return json.dumps(component.to_spec(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_defaults_keyword_only_signature_and_closed_descriptor(cls, identity, defaults, optional):
    signature = inspect.signature(cls)
    expected_fields = {**defaults, **optional, "execution_owner": "client"}
    assert set(signature.parameters) == set(expected_fields)
    for name, parameter in signature.parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default == defaults.get(name)
    with pytest.raises(TypeError):
        cls("source")
    component = cls()
    assert type(component) is cls
    assert component.to_spec() == {"component": identity, "params": defaults}
    assert component.get_params() == component.get_params(deep=False) == defaults
    descriptor = cls.describe()
    assert component.describe() is descriptor is cls.descriptor
    assert descriptor.component == identity
    assert descriptor.provider == "pllm" and descriptor.distribution == "pllm.run"
    assert descriptor.version == descriptor.category_version == "1"
    assert descriptor.lifecycle_phase == "compile"
    assert descriptor.category == (
        "pllm/nonlinear-protocol" if cls is F32Activation else "pllm/operators"
    )
    assert descriptor.to_dict()["parameter_schema"] == {
        "type": "object",
        "properties": {
            name: {"type": "string", "enum": [value]}
            for name, value in expected_fields.items()
        },
        "required": list(defaults),
        "additionalProperties": False,
    }
    restored = cls.from_params({})
    assert type(restored) is cls and restored == component


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_direct_and_from_params_preserve_genuine_legacy_wire_bytes(cls, identity, defaults, optional):
    for fields in ({}, optional):
        for owner in ({}, {"execution_owner": "client"}):
            params = {**defaults, **fields, **owner}
            legacy = ComponentRef(identity, params)
            assert type(legacy) is ComponentRef
            direct = cls(**params)
            restored = cls.from_params(MappingProxyType(params))
            assert type(direct) is type(restored) is cls
            assert direct == restored == legacy
            assert direct.to_spec() == restored.to_spec() == legacy.to_spec()
            assert _wire_bytes(direct) == _wire_bytes(restored) == _wire_bytes(legacy)
            assert direct.get_params() == restored.get_params(deep=False) == params


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_invalid_literals_and_non_exact_strings_rejected(cls, identity, defaults, optional):
    fields = {**defaults, **optional, "execution_owner": "client"}
    for name, valid in fields.items():
        for invalid in ("invalid", StringSubclass(valid), True, 1, [], {}):
            params = {**defaults, name: invalid}
            with pytest.raises(ConfigurationError):
                cls(**params)
            with pytest.raises(ConfigurationError):
                cls.from_params(params)
    for name in defaults:
        params = {**defaults, name: None}
        with pytest.raises(ConfigurationError):
            cls(**params)
        with pytest.raises(ConfigurationError):
            cls.from_params(params)


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_unknown_keywords_rejected_without_arbitrary_kwargs(cls, identity, defaults, optional):
    for name in ("unknown", "component", "params"):
        with pytest.raises(TypeError):
            cls(**{name: "invalid"})
        with pytest.raises(ConfigurationError):
            cls.from_params({name: "invalid"})


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
@pytest.mark.parametrize("params", (None, "source", True, 1, [], [("execution_owner", "client")]))
def test_from_params_rejects_nonmapping(cls, identity, defaults, optional, params):
    with pytest.raises(ConfigurationError):
        cls.from_params(params)


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_immutable_fields_params_and_defensive_copies(cls, identity, defaults, optional):
    params = {**defaults, **optional, "execution_owner": "client"}
    expected = dict(params)
    component = cls.from_params(params)
    params.clear()
    assert component.get_params() == expected
    for deep in (False, True):
        exported = component.get_params(deep=deep)
        exported.clear()
        assert component.get_params(deep=deep) == expected
    spec = component.to_spec()
    spec["params"].clear()
    assert component.to_spec() == {"component": identity, "params": expected}
    for field, value in (("component", "changed"), ("params", {})):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(component, field, value)
    with pytest.raises(TypeError):
        component.params["execution_owner"] = "changed"


@pytest.mark.parametrize("cls,identity,defaults,optional", CASES)
def test_optional_constructor_none_omits_but_explicit_wire_null_rejects(cls, identity, defaults, optional):
    nulls = {name: None for name in (*optional, "execution_owner")}
    omitted = cls(**nulls)
    assert omitted.to_spec() == {"component": identity, "params": defaults}
    for name in nulls:
        with pytest.raises(ConfigurationError):
            cls.from_params({**defaults, name: None})
