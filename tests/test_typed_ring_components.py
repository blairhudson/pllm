"""Installed ring configuration contracts; not execution or collection-isolation proof."""

import inspect
import json
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

import pllm.correlation as correlation
import pllm.nonlinear as nonlinear
import pllm.numeric as numeric
import pllm.preparation as preparation
import pllm.protocols as protocols
import pllm.schedulers as schedulers
import pllm.sources as sources
import pllm.verification as verification
from pllm.configuration import ComponentRef, ConfigurationError


# Literal wire expectations, independent of the implementation descriptors.
# PUMA: 21 unique classes.
CASES = (
    (protocols.ReplicatedPrivateTokenInput, "pllm/replicated-private-token-input/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedCompleteTableLookup, "pllm/replicated-complete-table-lookup/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedPublicMatmulFullSum, "pllm/replicated-public-matmul-full-sum/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedSecretMultiply, "pllm/replicated-secret-multiply/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedRingAddBias, "pllm/replicated-ring-add-bias/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedArithmeticBooleanConversion, "pllm/replicated-arithmetic-boolean-conversion/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedCausalQkSoftmaxAv, "pllm/replicated-causal-qk-softmax-av/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedCompleteStableArgmax, "pllm/replicated-complete-stable-argmax/v1", {}, "pllm/protocols"),
    (protocols.ReplicatedFixedCountFinalTokenRelease, "pllm/replicated-fixed-count-final-token-release/v1", {}, "pllm/protocols"),
    (protocols.AuthenticatedDirectedThreePartyTransport, "pllm/authenticated-directed-three-party-transport/v1", {}, "pllm/protocols"),
    (nonlinear.ReplicatedTruncAprQuarterGap, "pllm/replicated-truncapr-quarter-gap/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18PiecewiseSiluSourceOrder, "pllm/q18-piecewise-silu-source-order/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18SourceReciprocal, "pllm/q18-source-reciprocal/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18SourceRsqrt, "pllm/q18-source-rsqrt/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18RmsNormEpsilon1, "pllm/q18-rmsnorm-epsilon1/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18SourceRotary, "pllm/q18-source-rotary/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.Q18ShiftedClipExp5Reciprocal, "pllm/q18-shifted-clip-exp5-reciprocal/v1", {}, "pllm/nonlinear-protocol"),
    (numeric.ReplicatedRing64Q18Edges, "pllm/replicated-ring64-q18-edges/v1", {"revision": "source-order-v1"}, "pllm/numeric"),
    (sources.PublicCommittedQ18Source, "pllm/public-committed-q18-source/v1", {}, "pllm/sources"),
    (correlation.ReplicatedPairwisePrss, "pllm/replicated-pairwise-prss/v1", {}, "pllm/correlation-source"),
    (verification.PerEdgeRangeWorkAdmission, "pllm/per-edge-range-work-admission/v1", {}, "pllm/verification-scheme"),
    # FSS: 16 more, including the shared Ring50Q12Edges only once.
    (protocols.FSSPublicTableLookup, "pllm/fss-private-public-table-lookup/v1", {}, "pllm/protocols"),
    (protocols.FSSPublicDot, "pllm/fss-public-dot/v1", {}, "pllm/protocols"),
    (protocols.FSSPrivateDot, "pllm/fss-private-dot/v1", {}, "pllm/protocols"),
    (protocols.FSSSecretMultiply, "pllm/fss-secret-gated-multiply/v1", {}, "pllm/protocols"),
    (protocols.FSSAdd, "pllm/fss-add/v1", {}, "pllm/protocols"),
    (protocols.FSSArgmax, "pllm/fss-argmax/v1", {}, "pllm/protocols"),
    (protocols.FSSAuthenticatedRoleTransport, "pllm/fss-authenticated-role-transport/v1", {}, "pllm/protocols"),
    (nonlinear.FSSSignedFloorRescale, "pllm/fss-signed-trunc-floor/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.FSSSigned38InputSiLU, "pllm/fss-silu/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.FSSSumSquaresTableRMSNorm, "pllm/fss-rmsnorm/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.FSSProductFloorRotary, "pllm/fss-rotary/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.FSSSourceTableSoftmax, "pllm/fss-softmax/v1", {}, "pllm/nonlinear-protocol"),
    (numeric.Ring50Q12Edges, "pllm/ring50-q12-edges/v1", {
        "revision": "fusefss.dense-rope-ring50-q12.v1", "ranges": "source-derived",
    }, "pllm/numeric"),
    (sources.FSSSourceBinding, "pllm/fss-source-binding/v1", {}, "pllm/sources"),
    (preparation.TrustedDealerOneUseFSSMaterial, "pllm/fss-one-use-material/v1", {}, "pllm/preparation-provider"),
    (schedulers.FSSOnlineSchedule, "pllm/fss-online-schedule/v1", {}, "pllm/protected-scheduler"),
    # Sigma: 18 more, with distinct arithmetic/ownership contracts.
    (protocols.RingPublicCoefficientDot, "pllm/ring-public-coefficient-dot/v1", {}, "pllm/protocols"),
    (protocols.NegativeMaskMatrixProduct, "pllm/negative-mask-matrix-product/v1", {}, "pllm/protocols"),
    (protocols.NegativeMaskElementwiseProduct, "pllm/negative-mask-elementwise-product/v1", {}, "pllm/protocols"),
    (protocols.MaskedRingAddSubProduct, "pllm/masked-ring-add-sub-product/v1", {}, "pllm/protocols"),
    (protocols.BoundedDpfPublicTable, "pllm/bounded-dpf-public-table/v1", {}, "pllm/protocols"),
    (protocols.XorPredicateFiveWordSelect, "pllm/xor-predicate-five-word-select/v1", {}, "pllm/protocols"),
    (protocols.CausalMaxNexpReciprocalAttention, "pllm/causal-max-nexp-reciprocal-attention/v1", {}, "pllm/protocols"),
    (protocols.StablePrivateArgmaxEmbedding, "pllm/stable-private-argmax-embedding/v1", {}, "pllm/protocols"),
    (protocols.AuthenticatedNativeInput, "pllm/authenticated-native-input/v1", {}, "pllm/protocols"),
    (protocols.ConsumingShareToCommonMask, "pllm/consuming-share-to-common-mask/v1", {}, "pllm/protocols"),
    (nonlinear.ReducedRingTruncation, "pllm/reduced-ring-truncation/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.SameRingGapARS, "pllm/same-ring-gapars/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.XorMaskedDReLU, "pllm/xor-masked-drelu/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.EarlyFloorResidualActivation, "pllm/early-floor-residual-activation/v1", {}, "pllm/nonlinear-protocol"),
    (nonlinear.IntegerRsqrtTableNorm, "pllm/integer-rsqrt-table-norm/v1", {
        "kind": "source", "epsilon": "source", "semantic_revision": None,
    }, "pllm/nonlinear-protocol"),
    (nonlinear.SourceRing50Q12Rotary, "pllm/source-ring50-q12-rotary/v1", {
        "semantic_revision": None,
    }, "pllm/nonlinear-protocol"),
    (preparation.OneUseRoleLocalMaterial, "pllm/one-use-role-local-material/v1", {}, "pllm/preparation-provider"),
    (schedulers.BoundedConsumingProtectedDag, "pllm/bounded-consuming-protected-dag/v1", {}, "pllm/protected-scheduler"),
)
VARIANTS = (
    (numeric.Ring50Q12Edges, "pllm/ring50-q12-edges/v1", {
        "revision": "sigma-source-order-v1", "ranges": "source-derived",
    }, "pllm/numeric"),
    (protocols.FSSPublicDot, "pllm/fss-public-dot/v1", {"capacity": "full-vocabulary"}, "pllm/protocols"),
    (protocols.RingPublicCoefficientDot, "pllm/ring-public-coefficient-dot/v1", {"capacity": "full-vocabulary"}, "pllm/protocols"),
)
PROPERTIES = {
    numeric.ReplicatedRing64Q18Edges: {
        "revision": {"type": "string", "enum": ["source-order-v1"]},
    },
    numeric.Ring50Q12Edges: {
        "revision": {"type": "string", "enum": [
            "sigma-source-order-v1", "fusefss.dense-rope-ring50-q12.v1",
        ]},
        "ranges": {"type": "string", "enum": ["source-derived"]},
    },
    protocols.FSSPublicDot: {
        "capacity": {"type": "string", "enum": ["full-vocabulary"]},
    },
    protocols.RingPublicCoefficientDot: {
        "capacity": {"type": "string", "enum": ["full-vocabulary"]},
    },
    nonlinear.IntegerRsqrtTableNorm: {
        "kind": {"type": "string", "enum": ["source"]},
        "epsilon": {"type": "string", "enum": ["source"]},
        "semantic_revision": {"type": "null"},
    },
    nonlinear.SourceRing50Q12Rotary: {"semantic_revision": {"type": "null"}},
}
CAPACITY_CLASSES = (protocols.FSSPublicDot, protocols.RingPublicCoefficientDot)
REQUIRED = {
    numeric.ReplicatedRing64Q18Edges: ("revision",),
    numeric.Ring50Q12Edges: ("revision", "ranges"),
    nonlinear.IntegerRsqrtTableNorm: ("kind", "epsilon", "semantic_revision"),
    nonlinear.SourceRing50Q12Rotary: ("semantic_revision",),
}
PUBLIC_FAMILIES = {
    "pllm/protocols": protocols,
    "pllm/nonlinear-protocol": nonlinear,
    "pllm/numeric": numeric,
    "pllm/sources": sources,
    "pllm/correlation-source": correlation,
    "pllm/verification-scheme": verification,
    "pllm/preparation-provider": preparation,
    "pllm/protected-scheduler": schedulers,
}


class StringSubclass(str):
    pass


def _wire_bytes(component):
    # ComponentRef has no canonical_bytes method. Use canonical JSON of its spec.
    return json.dumps(component.to_spec(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_matrix_has_fifty_five_unique_classes_and_wire_identities():
    assert len(CASES) == len({row[0] for row in CASES}) == len({row[1] for row in CASES}) == 55


@pytest.mark.parametrize("cls,identity,params,category", CASES)
def test_keyword_only_signature_and_closed_literal_descriptor(cls, identity, params, category):
    assert getattr(PUBLIC_FAMILIES[category], cls.__name__) is cls
    properties = PROPERTIES.get(cls, {})
    signature = inspect.signature(cls)
    assert set(signature.parameters) == set(properties)
    for parameter in signature.parameters.values():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is (None if cls in CAPACITY_CLASSES else inspect.Parameter.empty)
    with pytest.raises(TypeError):
        cls("invalid")
    descriptor = cls.describe()
    assert descriptor is cls.descriptor is cls(**params).describe()
    assert descriptor.component == identity
    assert descriptor.category == category
    assert descriptor.provider == "pllm" and descriptor.distribution == "pllm.run"
    assert descriptor.version == descriptor.category_version == "1"
    assert descriptor.lifecycle_phase == "compile"
    assert descriptor.to_dict()["parameter_schema"] == {
        "type": "object", "properties": properties,
        "required": list(REQUIRED.get(cls, ())), "additionalProperties": False,
    }


@pytest.mark.parametrize("cls,identity,params,category", CASES + VARIANTS)
def test_direct_and_both_mapping_constructors_preserve_genuine_legacy_wire(cls, identity, params, category):
    legacy = ComponentRef(identity, params)
    assert type(legacy) is ComponentRef
    direct = cls(**params)
    assert type(direct) is cls and issubclass(cls, ComponentRef)
    expected = {"component": identity, "params": params}
    assert direct.to_spec() == legacy.to_spec() == expected
    assert direct.get_params() == direct.get_params(deep=False) == params
    for supplied in (dict(params), MappingProxyType(dict(params))):
        restored = cls.from_params(supplied)
        assert type(restored) is cls
        assert restored == direct == legacy
        assert restored.to_spec() == expected
        assert _wire_bytes(restored) == _wire_bytes(direct) == _wire_bytes(legacy)
        assert restored.get_params() == restored.get_params(deep=False) == params


@pytest.mark.parametrize("cls,identity,params,category", CASES + VARIANTS)
def test_frozen_fields_and_fresh_input_params_and_specs(cls, identity, params, category):
    supplied = dict(params)
    component = cls.from_params(supplied)
    expected = {"component": identity, "params": dict(params)}
    # Insertion matters for no-argument declarations: clearing {} proves nothing.
    supplied["sentinel"] = "changed"
    assert component.to_spec() == expected
    for deep in (False, True):
        first = component.get_params(deep=deep)
        second = component.get_params(deep=deep)
        assert first == second == params and first is not second
        first["sentinel"] = "changed"
        assert second == component.get_params(deep=deep) == params
    first_spec, second_spec = component.to_spec(), component.to_spec()
    assert first_spec is not second_spec
    assert first_spec["params"] is not second_spec["params"]
    first_spec["params"]["sentinel"] = "changed"
    first_spec["component"] = "changed"
    assert second_spec == component.to_spec() == expected
    for name, value in (("component", "changed"), ("params", {})):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(component, name, value)
    with pytest.raises(TypeError):
        component.params["sentinel"] = "changed"


@pytest.mark.parametrize("cls,identity,params,category", CASES)
def test_unknown_and_reserved_constructor_and_wire_fields_rejected(cls, identity, params, category):
    for name in ("unknown", "component", "params"):
        supplied = {**params, name: "invalid"}
        with pytest.raises(TypeError):
            cls(**supplied)
        with pytest.raises(ConfigurationError):
            cls.from_params(supplied)


@pytest.mark.parametrize("cls,identity,params,category", CASES)
def test_from_params_requires_mapping_not_json_scalar_or_array(cls, identity, params, category):
    for invalid in (None, True, 1, "source", [], [["revision", "source-order-v1"]]):
        with pytest.raises(ConfigurationError):
            cls.from_params(invalid)


@pytest.mark.parametrize("cls,identity,params,category", CASES)
def test_real_registry_get_create_and_configuration_restoration(cls, identity, params, category):
    # Registry path only loads metadata and invokes from_params. No providers,
    # Pipeline admission, resolve, compilation, or native execution is requested.
    from pllm.components import get, create_component
    from pllm.configuration import _component_from_spec

    assert get(identity) is cls
    variants = [params, *(row[2] for row in VARIANTS if row[0] is cls)]
    for fields in variants:
        legacy = ComponentRef(identity, fields)
        wire = json.loads(_wire_bytes(legacy))
        created = create_component(identity, wire["params"])
        restored = _component_from_spec(wire, "component")
        assert type(created) is type(restored) is cls
        assert created.to_spec() == restored.to_spec() == legacy.to_spec()
        assert _wire_bytes(created) == _wire_bytes(restored) == _wire_bytes(legacy)


@pytest.mark.parametrize("cls,params,name", (
    (numeric.ReplicatedRing64Q18Edges, {"revision": "source-order-v1"}, "revision"),
    (numeric.Ring50Q12Edges, {"revision": "sigma-source-order-v1", "ranges": "source-derived"}, "revision"),
    (numeric.Ring50Q12Edges, {"revision": "sigma-source-order-v1", "ranges": "source-derived"}, "ranges"),
    (nonlinear.IntegerRsqrtTableNorm, {"kind": "source", "epsilon": "source", "semantic_revision": None}, "kind"),
    (nonlinear.IntegerRsqrtTableNorm, {"kind": "source", "epsilon": "source", "semantic_revision": None}, "epsilon"),
    (nonlinear.IntegerRsqrtTableNorm, {"kind": "source", "epsilon": "source", "semantic_revision": None}, "semantic_revision"),
    (nonlinear.SourceRing50Q12Rotary, {"semantic_revision": None}, "semantic_revision"),
))
def test_required_fields_including_null_revisions_cannot_be_omitted(cls, params, name):
    missing = {key: value for key, value in params.items() if key != name}
    with pytest.raises(TypeError):
        cls(**missing)
    with pytest.raises(ConfigurationError):
        cls.from_params(missing)


@pytest.mark.parametrize("cls,params,name,valid", (
    (numeric.ReplicatedRing64Q18Edges, {}, "revision", "source-order-v1"),
    (numeric.Ring50Q12Edges, {"ranges": "source-derived"}, "revision", "sigma-source-order-v1"),
    (numeric.Ring50Q12Edges, {"revision": "sigma-source-order-v1"}, "ranges", "source-derived"),
    (nonlinear.IntegerRsqrtTableNorm, {"epsilon": "source", "semantic_revision": None}, "kind", "source"),
    (nonlinear.IntegerRsqrtTableNorm, {"kind": "source", "semantic_revision": None}, "epsilon", "source"),
    (protocols.FSSPublicDot, {}, "capacity", "full-vocabulary"),
    (protocols.RingPublicCoefficientDot, {}, "capacity", "full-vocabulary"),
))
def test_literal_fields_reject_wrong_values_and_nonexact_strings(cls, params, name, valid):
    invalids = ("invalid", StringSubclass(valid), True, False, 1, [], {})
    if cls not in CAPACITY_CLASSES:
        invalids += (None,)
    for invalid in invalids:
        fields = {**params, name: invalid}
        with pytest.raises(ConfigurationError):
            cls(**fields)
        with pytest.raises(ConfigurationError):
            cls.from_params(fields)


@pytest.mark.parametrize("cls,params", (
    (nonlinear.IntegerRsqrtTableNorm, {"kind": "source", "epsilon": "source"}),
    (nonlinear.SourceRing50Q12Rotary, {}),
))
def test_unresolved_semantic_revision_rejects_nonnull_authority(cls, params):
    for revision in ("source", "source-order-v1", "sigma-source-order-v1",
                     "fusefss.dense-rope-ring50-q12.v1", StringSubclass("source"), True, False, 0, [], {}):
        fields = {**params, "semantic_revision": revision}
        with pytest.raises(ConfigurationError):
            cls(**fields)
        with pytest.raises(ConfigurationError):
            cls.from_params(fields)


@pytest.mark.parametrize("cls", CAPACITY_CLASSES)
def test_capacity_constructor_none_omits_but_explicit_wire_null_rejects(cls):
    assert cls().to_spec() == cls(capacity=None).to_spec()
    assert cls(capacity=None).get_params() == {}
    with pytest.raises(ConfigurationError):
        cls.from_params({"capacity": None})
