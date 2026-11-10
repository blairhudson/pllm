"""Installed configuration unit contracts, not execution or import-isolation proof."""

import inspect
import json
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from pllm.configuration import ComponentRef, ConfigurationError
from pllm.correlation import RaaMask
from pllm.nonlinear import F32Activation
from pllm.numeric import ExactSourceFieldBoundaries
from pllm.operators import (
    F32RmsNorm, F32Rotary, F32CausalAttention, F32Multiply, F32AddBias,
    F32Add, FullVocabularyHead, GreedySelection, PublicF32Lookup,
)
from pllm.protocols import PublicFieldMatvec, AuthenticatedConfidentialRoleTransport
from pllm.roles import ClientMaskSingleServerRoles
from pllm.sources import AuthenticatedSourceInputProducer, PinnedSemanticSourceBindings
from pllm.state import TentativeStateCommit
from pllm.verification import (
    TwoLevelCodedBatchVerification, ValidatedMatrixCodePublication, OriginalDotWindow,
)


NUMERIC = {
    "parameter_authority": None, "client_arithmetic": "f32-extension",
    "source_precision": "source", "source_to_field_rounding": None,
    "field_to_f32_rounding": None,
}
ARCHITECTURE = "BertForNextSentencePrediction"
THOR = {
    "required_model_type": "bert", "required_architectures": (ARCHITECTURE,),
    "required_output_logits": 2,
}
CASES = (
    (PublicFieldMatvec, "pllm/public-field-matvec/v1", {}),
    (RaaMask, "pllm/raa-mask/v1", {}),
    (TwoLevelCodedBatchVerification, "pllm/two-level-coded-batch-verification/v1", {}),
    (ValidatedMatrixCodePublication, "pllm/validated-matrix-code-publication/v1", {}),
    (OriginalDotWindow, "pllm/original-dot-window/v1", {}),
    (TentativeStateCommit, "pllm/tentative-state-commit/v1", {}),
    (AuthenticatedSourceInputProducer, "pllm/authenticated-source-input-producer/v1", {}),
    (ClientMaskSingleServerRoles, "pllm/client-mask-single-server-roles/v1", {}),
    (ExactSourceFieldBoundaries, "pllm/exact-source-field-boundaries/v1", NUMERIC),
    (PublicF32Lookup, "pllm/public-f32-lookup/v1", {"execution_owner": "client"}),
    (AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {}),
    (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {}),
)
FIRST_NINE = (
    (F32RmsNorm, "pllm/f32-rmsnorm/v1", {"epsilon": "source"}),
    (F32Rotary, "pllm/f32-rotary/v1", {"configuration": "source"}),
    (F32CausalAttention, "pllm/f32-causal-attention/v1", {"heads": "source", "kv": "state"}),
    (F32Activation, "pllm/f32-activation/v1", {"function": "source"}),
    (F32Multiply, "pllm/f32-multiply/v1", {}),
    (F32AddBias, "pllm/f32-add-bias/v1", {"placement": "after-reconstruction"}),
    (F32Add, "pllm/f32-add/v1", {}),
    (FullVocabularyHead, "pllm/full-vocabulary-head/v1", {"linear": "linear"}),
    (GreedySelection, "pllm/greedy-selection/v1", {"eos": "continue"}),
)
VARIANTS = (
    (AuthenticatedConfidentialRoleTransport, "pllm/authenticated-confidential-role-transport/v1", {"roles": "roles"}),
    (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", THOR),
    (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {"required_model_type": "bert"}),
    (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {"required_architectures": (ARCHITECTURE,)}),
    (PinnedSemanticSourceBindings, "pllm/pinned-semantic-source-bindings/v1", {"required_output_logits": 2}),
)


class StringSubclass(str):
    pass


def _wire_bytes(component):
    # ComponentRef has no canonical_bytes method; canonicalize its JSON spec.
    return json.dumps(component.to_spec(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


@pytest.mark.parametrize("cls,identity,params", CASES)
def test_closed_constructor_signature_and_actual_descriptor(cls, identity, params):
    signature = inspect.signature(cls)
    fields = (NUMERIC if cls is ExactSourceFieldBoundaries else
              {"execution_owner": "client"} if cls is PublicF32Lookup else
              {"roles": "roles"} if cls is AuthenticatedConfidentialRoleTransport else
              THOR if cls is PinnedSemanticSourceBindings else {})
    assert set(signature.parameters) == set(fields)
    for parameter in signature.parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is (
            inspect.Parameter.empty if cls in (ExactSourceFieldBoundaries, PublicF32Lookup) else None
        )
    with pytest.raises(TypeError):
        cls("invalid")
    descriptor = cls.describe()
    assert descriptor is cls.descriptor is cls(**params).describe()
    assert descriptor.component == identity
    assert descriptor.provider == "pllm" and descriptor.distribution == "pllm.run"
    assert descriptor.version == descriptor.category_version == "1"
    categories = {
        RaaMask: "pllm/correlation-source", TentativeStateCommit: "pllm/state-protocol",
        ClientMaskSingleServerRoles: "pllm/role-topology", ExactSourceFieldBoundaries: "pllm/numeric",
        PublicF32Lookup: "pllm/operators", AuthenticatedSourceInputProducer: "pllm/sources",
        PinnedSemanticSourceBindings: "pllm/sources",
    }
    verification = (TwoLevelCodedBatchVerification, ValidatedMatrixCodePublication, OriginalDotWindow)
    assert descriptor.category == categories.get(
        cls, "pllm/verification-scheme" if cls in verification else "pllm/protocols"
    )
    assert descriptor.lifecycle_phase == (
        "offline" if cls is ValidatedMatrixCodePublication else
        "online" if cls in (TwoLevelCodedBatchVerification, OriginalDotWindow) else "compile"
    )
    properties = {
        name: ({"type": "null"} if value is None else {"type": "string", "enum": [value]})
        for name, value in fields.items()
    }
    if cls is PinnedSemanticSourceBindings:
        properties["required_architectures"] = {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": {"type": "string", "enum": [ARCHITECTURE]},
        }
        properties["required_output_logits"] = {"type": "integer", "enum": [2]}
    schema = {"type": "object", "properties": properties, "additionalProperties": False}
    if cls not in (AuthenticatedSourceInputProducer, PinnedSemanticSourceBindings):
        schema["required"] = list(fields) if cls in (ExactSourceFieldBoundaries, PublicF32Lookup) else []
    assert descriptor.to_dict()["parameter_schema"] == schema


@pytest.mark.parametrize("cls,identity,params", CASES + VARIANTS)
def test_direct_and_from_params_equal_genuine_legacy_spec_and_bytes(cls, identity, params):
    legacy = ComponentRef(identity, params)
    assert type(legacy) is ComponentRef
    direct = cls(**params)
    restored = cls.from_params(MappingProxyType(params))
    assert type(direct) is type(restored) is cls
    assert issubclass(cls, ComponentRef)
    assert direct == restored == legacy
    assert direct.to_spec() == restored.to_spec() == legacy.to_spec()
    assert _wire_bytes(direct) == _wire_bytes(restored) == _wire_bytes(legacy)
    assert direct.get_params() == direct.get_params(deep=False) == legacy.to_spec()["params"]


@pytest.mark.parametrize("cls,identity,params", CASES + VARIANTS)
def test_frozen_configuration_and_defensive_public_params(cls, identity, params):
    supplied = json.loads(json.dumps(params))
    component = cls.from_params(supplied)
    expected = component.to_spec()
    supplied.clear()
    for deep in (False, True):
        public = component.get_params(deep=deep)
        assert public == expected["params"]
        public.clear()
        public[f"sentinel_deep_{deep}"] = True
        assert component.get_params(deep=deep) == expected["params"]
        assert component.get_params(deep=not deep) == expected["params"]
        assert component.to_spec() == expected
    spec = component.to_spec()
    spec["params"].clear()
    spec["params"]["sentinel_spec"] = True
    assert component.get_params(deep=False) == expected["params"]
    assert component.get_params(deep=True) == expected["params"]
    assert component.to_spec() == expected
    for name, value in (("component", "changed"), ("params", {})):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(component, name, value)
    with pytest.raises(TypeError):
        component.params["unknown"] = True


@pytest.mark.parametrize("cls,identity,params", CASES)
def test_unknown_constructor_and_from_params_fields_rejected(cls, identity, params):
    for name in ("unknown", "component", "params"):
        with pytest.raises(TypeError):
            cls(**{**params, name: "invalid"})
        with pytest.raises(ConfigurationError):
            cls.from_params({**params, name: "invalid"})


@pytest.mark.parametrize("cls,identity,params", CASES)
@pytest.mark.parametrize("invalid", (None, True, 1, "source", [], [["roles", "roles"]]))
def test_from_params_requires_mapping(cls, identity, params, invalid):
    with pytest.raises(ConfigurationError):
        cls.from_params(invalid)


@pytest.mark.parametrize("cls,params,name", (
    *((ExactSourceFieldBoundaries, NUMERIC, name) for name in NUMERIC),
    (PublicF32Lookup, {"execution_owner": "client"}, "execution_owner"),
))
def test_required_fields_cannot_be_omitted(cls, params, name):
    missing = {key: value for key, value in params.items() if key != name}
    with pytest.raises(TypeError):
        cls(**missing)
    with pytest.raises(ConfigurationError):
        cls.from_params(missing)


@pytest.mark.parametrize("name", ("parameter_authority", "source_to_field_rounding", "field_to_f32_rounding"))
def test_numeric_authority_and_rounding_remain_explicit_null(name):
    for invalid in ("claimed-authority", True, False, 0, [], {}):
        params = {**NUMERIC, name: invalid}
        with pytest.raises(ConfigurationError):
            ExactSourceFieldBoundaries(**params)
        with pytest.raises(ConfigurationError):
            ExactSourceFieldBoundaries.from_params(params)


@pytest.mark.parametrize("cls,params,name,valid", (
    (ExactSourceFieldBoundaries, NUMERIC, "client_arithmetic", "f32-extension"),
    (ExactSourceFieldBoundaries, NUMERIC, "source_precision", "source"),
    (PublicF32Lookup, {}, "execution_owner", "client"),
    (AuthenticatedConfidentialRoleTransport, {}, "roles", "roles"),
    (PinnedSemanticSourceBindings, {}, "required_model_type", "bert"),
))
def test_literal_strings_are_exact_and_closed(cls, params, name, valid):
    for invalid in ("invalid", StringSubclass(valid), True, 1, [], {}):
        fields = {**params, name: invalid}
        with pytest.raises(ConfigurationError):
            cls(**fields)
        with pytest.raises(ConfigurationError):
            cls.from_params(fields)
    if cls in (ExactSourceFieldBoundaries, PublicF32Lookup):
        with pytest.raises(ConfigurationError):
            cls(**{**params, name: None})
        with pytest.raises(ConfigurationError):
            cls.from_params({**params, name: None})


@pytest.mark.parametrize("cls,name", (
    (AuthenticatedConfidentialRoleTransport, "roles"),
    *((PinnedSemanticSourceBindings, name) for name in THOR),
))
def test_optional_constructor_null_omits_but_wire_null_rejects(cls, name):
    assert cls(**{name: None}).get_params() == {}
    with pytest.raises(ConfigurationError):
        cls.from_params({name: None})


def test_source_architecture_json_array_is_normalized_and_defensively_copied():
    original = {"required_architectures": [ARCHITECTURE]}
    component = PinnedSemanticSourceBindings.from_params(original)
    direct = PinnedSemanticSourceBindings(required_architectures=(ARCHITECTURE,))
    assert type(component) is PinnedSemanticSourceBindings
    assert component == direct
    assert component.params["required_architectures"] == (ARCHITECTURE,)
    original["required_architectures"].append("changed")
    for deep in (False, True):
        first = component.get_params(deep=deep)
        second = component.get_params(deep=deep)
        assert type(first["required_architectures"]) is list
        assert first["required_architectures"] is not second["required_architectures"]
        first["required_architectures"].append("changed")
        assert second == component.get_params(deep=deep) == {"required_architectures": [ARCHITECTURE]}
    assert component.to_spec()["params"] == {"required_architectures": [ARCHITECTURE]}
    assert _wire_bytes(component) == _wire_bytes(ComponentRef(
        "pllm/pinned-semantic-source-bindings/v1", {"required_architectures": [ARCHITECTURE]},
    ))


@pytest.mark.parametrize("value", (True, False, 2.0, "2", 1, 3, [], {}))
def test_source_output_logits_requires_exact_integer_two(value):
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings(required_output_logits=value)
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings.from_params({"required_output_logits": value})


@pytest.mark.parametrize("value", (
    (), (ARCHITECTURE, ARCHITECTURE), ("invalid",), (StringSubclass(ARCHITECTURE),),
    (True,), ARCHITECTURE, True, {},
))
def test_source_architecture_rejects_wrong_arity_literals_and_shapes(value):
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings(required_architectures=value)
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings.from_params({"required_architectures": value})


@pytest.mark.parametrize("value", ([], [ARCHITECTURE, ARCHITECTURE], ["invalid"], [True]))
def test_source_json_architecture_array_validates_contents(value):
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings.from_params({"required_architectures": value})


def test_source_constructor_requires_tuple_not_json_list():
    with pytest.raises(ConfigurationError):
        PinnedSemanticSourceBindings(required_architectures=[ARCHITECTURE])


@pytest.mark.parametrize("cls,identity,params", FIRST_NINE + CASES)
def test_configuration_registry_restores_all_twenty_one_concrete_types(cls, identity, params):
    # Source-audited built-in registry loads descriptors and calls from_params;
    # no Pipeline admission, runtime helper, provider, or source loader involved.
    from pllm.components import create_component
    from pllm.configuration import _component_from_spec

    legacy = ComponentRef(identity, params)
    wire = json.loads(_wire_bytes(legacy))
    created = create_component(identity, wire["params"])
    restored = _component_from_spec(wire, "component")
    assert type(created) is type(restored) is cls
    assert created.to_spec() == restored.to_spec() == legacy.to_spec()
    assert _wire_bytes(created) == _wire_bytes(restored) == _wire_bytes(legacy)
