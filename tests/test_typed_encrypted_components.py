"""Installed configuration contracts, not encrypted execution or collection proof."""

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
import pllm.roles as roles
import pllm.schedulers as schedulers
import pllm.sources as sources
import pllm.state as state
import pllm._preparation_declarations as preparation_leaf
import pllm.correlation._declarations as correlation_leaf
import pllm.nonlinear._conversion_declarations as nc
import pllm.nonlinear._he_declarations as nh
import pllm.nonlinear._heterogeneous_declarations as nx
import pllm.numeric._declarations as numeric_leaf
import pllm.protocols._conversion_declarations as pc
import pllm.protocols._he_declarations as ph
import pllm.protocols._heterogeneous_declarations as px
import pllm.roles._declarations as roles_leaf
import pllm.schedulers._declarations as schedulers_leaf
import pllm.sources._declarations as sources_leaf
import pllm.state._declarations as state_leaf
from pllm.configuration import ComponentRef, ConfigurationError


# Independent literal expectations: no descriptor supplies an expected ID/schema.
BRIDGE = {
    "to_coefficients": "slot_coefficient_transform", "ring_switch": "ring_switch",
    "extract": "coefficient_extraction", "compare": "comparison", "evaluate": "segmented_affine",
    "key_switch": "key_switch", "modulus_raise": "modulus_raise",
    "to_slots": "coefficient_slot_transform", "evalmod": "evalmod", "numeric": "numeric",
}
GROUPS = (
    (protocols, pc, "pllm/protocols", (  # 13 conversion protocols.
        (protocols.PrivateTokenShare, "pllm/private-token-share/v1", {}),
        (protocols.CoefficientPrivateLookup, "pllm/coefficient-private-lookup/v1", {}),
        (protocols.CoefficientObliviousLinearTransform, "pllm/coefficient-olt/v1", {}),
        (protocols.SimdBilinearOle, "pllm/simd-bole/v1", {}),
        (protocols.ArithmeticToHomomorphic, "pllm/arithmetic-to-homomorphic/v1", {}),
        (protocols.TraceInterleaveRepacking, "pllm/trace-interleave-repacking/v1", {}),
        (protocols.CoefficientHomomorphicToArithmetic, "pllm/coefficient-homomorphic-to-arithmetic/v1", {}),
        (protocols.FloodedSimdHomomorphicToArithmetic, "pllm/flooded-simd-homomorphic-to-arithmetic/v1", {}),
        (protocols.CrtRingConversion, "pllm/crt-ring-conversion/v1", {}),
        (protocols.SignedRingExtension, "pllm/signed-ring-extension/v1", {}),
        (protocols.SourcePositionOperators, "pllm/source-position-operators/v1", {}),
        (protocols.SharedStableArgmax, "pllm/shared-stable-argmax/v1", {}),
        (protocols.PrivateFullPrefixFeedback, "pllm/private-full-prefix-feedback/v1", {}),
    )),
    (nonlinear, nc, "pllm/nonlinear-protocol", (  # 6 conversion nonlinear declarations.
        (nonlinear.SharedRingTruncation, "pllm/shared-ring-truncation/v1", {}),
        (nonlinear.SharedComparisonMux, "pllm/shared-comparison-mux/v1", {}),
        (nonlinear.ShiftedClipRepeatedSquareSoftmax, "pllm/shifted-clip-repeated-square-softmax/v1", {}),
        (nonlinear.SharedReciprocalInverseRootNormalization, "pllm/shared-reciprocal-inverse-root-normalization/v1", {}),
        (nonlinear.PiecewisePolynomialActivation, "pllm/piecewise-polynomial-activation/v1", {}),
        (nonlinear.SharedRingResidualBias, "pllm/shared-ring-residual-bias/v1", {}),
    )),
    (protocols, ph, "pllm/protocols", (  # 24 HE protocols, shared paper IDs once.
        (protocols.CkksRectangularLinear, "pllm/ckks-rectangular-linear/v1", {}),
        (protocols.InterleavedQK, "pllm/interleaved-qk/v1", {}),
        (protocols.InterleavedWeightedV, "pllm/interleaved-weighted-v/v1", {}),
        (protocols.RoutedGQA, "pllm/routed-gqa/v1", {}),
        (protocols.HERotaryTransform, "pllm/he-rotary-transform/v1", {}),
        (protocols.HECoefficientCompression, "pllm/he-coefficient-compression/v1", {}),
        (protocols.HECommutableLinear, "pllm/he-commutable-linear/v1", {}),
        (protocols.HEColumnProduct, "pllm/he-column-product/v1", {}),
        (protocols.HELayoutTransform, "pllm/he-layout-transform/v1", {}),
        (protocols.HEAttentionValue, "pllm/he-attention-value/v1", {}),
        (protocols.HEAttentionProducts, "pllm/he-attention-products/v1", {}),
        (protocols.HEHeadScaleMask, "pllm/he-head-scale-mask/v1", {}),
        (protocols.HEResidualAdd, "pllm/he-residual-add/v1", {}),
        (protocols.HEPrivateEmbedding, "pllm/he-private-embedding/v1", {}),
        (protocols.HESourceEmbeddingPositionSegment, "pllm/he-source-embedding-position-segment/v1", {}),
        (protocols.HEClsPooler, "pllm/he-cls-pooler/v1", {}),
        (protocols.HENspTwoLogitHead, "pllm/he-nsp-two-logit-head/v1", {}),
        (protocols.HEFoldSelection, "pllm/he-fold-selection/v1", {}),
        (protocols.NspLogitsOnlyOutputPolicy, "pllm/nsp-logits-only-output-policy/v1", {}),
        (protocols.HEBootstrap, "pllm/he-bootstrap/v1", {}),
        (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {}),
        (protocols.PrivateTokenSelection, "pllm/private-token-selection/v1", {}),
        (protocols.PrivateTokenFeedback, "pllm/private-token-feedback/v1", {}),
        (protocols.ClassifierOutputPolicy, "pllm/classifier-output-policy/v1", {"source": "model-metadata"}),
    )),
    (nonlinear, nh, "pllm/nonlinear-protocol", (  # 13 HE nonlinear declarations.
        (nonlinear.CalibratedPolynomialExp, "pllm/calibrated-polynomial-exp/v1", {"coefficients": None, "domain": None}),
        (nonlinear.StandardGoldschmidtReciprocal, "pllm/standard-goldschmidt-reciprocal/v1", {"iterations": None, "domain": None}),
        (nonlinear.CompositePolynomialActivation, "pllm/composite-polynomial-activation/v1", {"coefficients": None, "domain": None}),
        (nonlinear.InverseRoot, "pllm/certified-inverse-root/v1", {"iterations": None, "domain": None}),
        (nonlinear.HEMaskedSoftmax, "pllm/he-masked-softmax/v1", {}),
        (nonlinear.SourceNormalization, "pllm/source-normalization/v1", {}),
        (nonlinear.HERepeatedSquaringSoftmax, "pllm/he-repeated-squaring-softmax/v1", {"source_squarings": 8, "parameter_authority": None}),
        (nonlinear.HEPiecewisePolynomialActivation, "pllm/he-piecewise-polynomial-activation/v1", {}),
        (nonlinear.HERowNormalization, "pllm/he-row-normalization/v1", {}),
        (nonlinear.HESignComposition, "pllm/he-sign-composition/v1", {}),
        (nonlinear.HELayerNorm, "pllm/he-layernorm/v1", {}),
        (nonlinear.HESquareNormalizeSoftmax, "pllm/he-square-normalize-softmax/v1", {}),
        (nonlinear.HEPolynomialActivation, "pllm/he-polynomial-activation/v1", {"function": "source-gelu"}),
    )),
    (protocols, px, "pllm/protocols", (  # 8 heterogeneous protocols.
        (protocols.AuthenticatedClientInputExport, "pllm/authenticated-client-input-export/v1", {"roles": "roles", "transport": "transport", "birth": "ciphertext_birth"}),
        (protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": "query-key-dot-product", "numeric": "numeric"}),
        (protocols.HESlotCoefficientTransform, "pllm/he-slot-coefficient-transform/v1", {"numeric": "numeric"}),
        (protocols.HERingSwitch, "pllm/he-ring-switch/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
        (protocols.HECoefficientExtraction, "pllm/he-coefficient-extraction/v1", {"numeric": "numeric"}),
        (protocols.HEKeySwitch, "pllm/he-key-switch/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
        (protocols.HEModulusRaise, "pllm/he-modulus-raise/v1", {"numeric": "numeric"}),
        (protocols.HECoefficientSlotTransform, "pllm/he-coefficient-slot-transform/v1", {"numeric": "numeric"}),
    )),
    (nonlinear, nx, "pllm/nonlinear-protocol", (  # 6 heterogeneous nonlinear declarations.
        (nonlinear.LWEHighPrecisionComparison, "pllm/lwe-high-precision-comparison/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
        (nonlinear.RLWEEncryptedAccumulatorSegmentedAffine, "pllm/rlwe-encrypted-accumulator-segmented-affine/v1", {"comparison": "comparison", "key_graph": "key_graph", "numeric": "numeric"}),
        (nonlinear.HEEvalMod, "pllm/he-evalmod/v1", {"numeric": "numeric"}),
        (nonlinear.HENormalization, "pllm/he-normalization/v1", {"square_reduce": "normalization_square_reduce", "inverse_sqrt": "segmented_affine", "multiply": "normalization_multiply", **BRIDGE}),
        (nonlinear.HESoftmaxSquareNormalize, "pllm/he-softmax-square-normalize/v1", {"square_reduce": "softmax_square_reduce", "reciprocal": "segmented_affine", "multiply": "softmax_multiply", **BRIDGE}),
        (nonlinear.HEActivationBridge, "pllm/he-activation-bridge/v1", BRIDGE),
    )),
    (numeric, numeric_leaf, "pllm/numeric", (  # Common 21: numeric 3.
        (numeric.SourceCalibration, "pllm/source-authenticated-calibration/v1", {}),
        (numeric.NumericDomain, "pllm/numeric-domain-certificate/v1", {"parameter_authority": None}),
        (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None}),
    )),
    (preparation, preparation_leaf, "pllm/preparation-provider", (  # Preparation 5.
        (preparation.HEKeyContext, "pllm/he-key-context/v1", {}),
        (preparation.HEDualKeyContext, "pllm/he-dual-key-context/v1", {}),
        (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {}),
        (preparation.HEKeyGraph, "pllm/he-key-graph/v1", {"secret_key_owner": "client", "server_secret_keys": False}),
        (preparation.OneUsePartyLocalConversion, "pllm/one-use-party-local-conversion/v1", {}),
    )),
    (schedulers, schedulers_leaf, "pllm/protected-scheduler", (  # Scheduling 3.
        (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {}),
        (schedulers.HELevelSchemePlacement, "pllm/he-level-scheme-placement/v1", {"schedule": "level_error_schedule", "roles": "roles"}),
        (schedulers.SetupOnlineRetirementSchedule, "pllm/setup-online-retirement-schedule/v1", {}),
    )),
    (state, state_leaf, "pllm/state-protocol", (  # State 3.
        (state.InterleavedKVTransaction, "pllm/interleaved-kv-transaction/v1", {"cache_retention": None}),
        (state.EncryptedKVTransaction, "pllm/encrypted-kv-transaction/v1", {"birth": "ciphertext_birth", "schedule": "level_error_schedule"}),
        (state.SharedFullPrefixState, "pllm/shared-full-prefix-state/v1", {}),
    )),
    (roles, roles_leaf, "pllm/role-topology", (  # Roles 4.
        (roles.ClientKeyServerEvaluationRoles, "pllm/client-key-server-evaluation-roles/v1", {}),
        (roles.ClientSecretServerWeightSecretRoles, "pllm/client-secret-server-weight-secret-roles/v1", {}),
        (roles.ClientKeyServerConfidentialWeightRoles, "pllm/client-key-server-confidential-weight-roles/v1", {"secret_key_owner": "client", "weight_owner": "server"}),
        (roles.TwoOwnerPrivateInputPrivateModelRoles, "pllm/two-owner-private-input-private-model-roles/v1", {}),
    )),
    (sources, sources_leaf, "pllm/sources", (  # Sources 2.
        (sources.AuthenticatedSourceBindings, "pllm/authenticated-source-bindings/v1", {"weight_owner": "server", "bind_checkpoint_and_numeric_domain": True}),
        (sources.AuthenticatedOriginalProducer, "pllm/authenticated-original-producer/v1", {}),
    )),
    (correlation, correlation_leaf, "pllm/correlation-source", (  # Correlation 1.
        (correlation.DistributedFerretOt, "pllm/distributed-ferret-ot/v1", {}),
    )),
)
CASES = tuple((public, leaf, cls, identity, params, category)
              for public, leaf, category, rows in GROUPS for cls, identity, params in rows)
SLOT = {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}}
REFERENCE = {"type": "string", "minLength": 1, "pattern": r"\S", "not": {"pattern": "__"}}
OPERATIONS = [
    "query-key-dot-product", "probability-value-reduction", "normalization-square-reduction",
    "normalization-elementwise-product", "softmax-square-and-row-sum",
    "softmax-row-normalization", "activation-gated-product",
]
BRIDGE_PROPERTIES = {name: REFERENCE for name in (
    "to_coefficients", "ring_switch", "extract", "compare", "evaluate",
    "key_switch", "modulus_raise", "to_slots", "evalmod", "numeric",
)}
PROPERTIES = {
    protocols.HEPackedLinear: {
        "weight_owner": {"type": "string", "enum": ["server"]}, "source_bindings": REFERENCE,
        "operation": {"type": "string", "enum": ["source-output-head"]},
        "vocabulary": {"type": "string", "enum": ["full"]}, "selection": REFERENCE,
    },
    protocols.PrivateTokenSelection: {"comparison": REFERENCE, "roles": REFERENCE},
    protocols.PrivateTokenFeedback: {"selection": REFERENCE, "input": REFERENCE, "state": REFERENCE},
    protocols.ClassifierOutputPolicy: {"source": {"type": "string", "enum": ["model-metadata"]}},
    nonlinear.CalibratedPolynomialExp: {"coefficients": {"type": "null"}, "domain": {"type": "null"}},
    nonlinear.StandardGoldschmidtReciprocal: {"iterations": {"type": "null"}, "domain": {"type": "null"}},
    nonlinear.CompositePolynomialActivation: {"coefficients": {"type": "null"}, "domain": {"type": "null"}},
    nonlinear.InverseRoot: {"iterations": {"type": "null"}, "domain": {"type": "null"}},
    nonlinear.HERepeatedSquaringSoftmax: {"source_squarings": {"type": "integer", "enum": [8]}, "parameter_authority": {"type": "null"}},
    nonlinear.HEPolynomialActivation: {"function": {"type": "string", "enum": ["source-gelu", "source-tanh"]}},
    protocols.AuthenticatedClientInputExport: {"roles": SLOT, "transport": SLOT, "birth": SLOT},
    protocols.HEEncryptedMultiplyReduce: {"operation": {"type": "string", "minLength": 1, "enum": OPERATIONS}, "numeric": SLOT},
    protocols.HESlotCoefficientTransform: {"numeric": SLOT},
    protocols.HERingSwitch: {"key_graph": SLOT, "numeric": SLOT},
    protocols.HECoefficientExtraction: {"numeric": SLOT},
    protocols.HEKeySwitch: {"key_graph": SLOT, "numeric": SLOT},
    protocols.HEModulusRaise: {"numeric": SLOT},
    protocols.HECoefficientSlotTransform: {"numeric": SLOT},
    nonlinear.LWEHighPrecisionComparison: {"key_graph": REFERENCE, "numeric": REFERENCE},
    nonlinear.RLWEEncryptedAccumulatorSegmentedAffine: {"comparison": REFERENCE, "key_graph": REFERENCE, "numeric": REFERENCE},
    nonlinear.HEEvalMod: {"numeric": REFERENCE},
    nonlinear.HENormalization: {"square_reduce": REFERENCE, "inverse_sqrt": REFERENCE, "multiply": REFERENCE, **BRIDGE_PROPERTIES},
    nonlinear.HESoftmaxSquareNormalize: {"square_reduce": REFERENCE, "reciprocal": REFERENCE, "multiply": REFERENCE, **BRIDGE_PROPERTIES},
    nonlinear.HEActivationBridge: BRIDGE_PROPERTIES,
    numeric.NumericDomain: {"parameter_authority": {"type": "null"}},
    numeric.SourceNumericDomainCertificates: {
        "parameter_authority": {"type": "null"},
        "profile": {"type": "string", "enum": ["paper-ring64-q18"]},
        "extension": {"type": "string", "enum": ["ring32-q12-where-bound"]},
        "source": {"type": "string", "minLength": 1, "pattern": r"\S"},
    },
    preparation.HEKeyContext: {"prime_chain": {"type": "null"}, "key_parameters": {"type": "null"}},
    preparation.HECiphertextBirth: {"key_graph": SLOT, "numeric": SLOT},
    preparation.HEKeyGraph: {"secret_key_owner": {"type": "string", "enum": ["client"]}, "server_secret_keys": {"type": "boolean", "enum": [False]}},
    schedulers.HELevelErrorSchedule: {"numeric": SLOT, "key_graph": SLOT},
    schedulers.HELevelSchemePlacement: {"schedule": SLOT, "roles": SLOT},
    state.InterleavedKVTransaction: {"cache_retention": {"type": "null"}},
    state.EncryptedKVTransaction: {"birth": SLOT, "schedule": SLOT},
    roles.ClientKeyServerConfidentialWeightRoles: {"secret_key_owner": {"type": "string", "enum": ["client"]}, "weight_owner": {"type": "string", "enum": ["server"]}},
    sources.AuthenticatedSourceBindings: {"weight_owner": {"type": "string", "enum": ["server"]}, "bind_checkpoint_and_numeric_domain": {"type": "boolean", "enum": [True]}},
}
OPTIONAL = {
    protocols.HEPackedLinear: ("weight_owner", "source_bindings", "operation", "vocabulary", "selection"),
    protocols.PrivateTokenSelection: ("comparison", "roles"),
    protocols.PrivateTokenFeedback: ("selection", "input", "state"),
    numeric.SourceNumericDomainCertificates: ("profile", "extension", "source"),
    preparation.HEKeyContext: ("prime_chain", "key_parameters"),
    preparation.HECiphertextBirth: ("key_graph", "numeric"),
    schedulers.HELevelErrorSchedule: ("numeric", "key_graph"),
}
VARIANTS = (
    (nonlinear.HEPolynomialActivation, "pllm/he-polynomial-activation/v1", {"function": "source-tanh"}),
    (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {"weight_owner": "server", "source_bindings": "source_bindings"}),
    (protocols.HEPackedLinear, "pllm/he-packed-linear/v1", {"weight_owner": "server", "source_bindings": "source_bindings", "operation": "source-output-head", "vocabulary": "full", "selection": "selection"}),
    (protocols.PrivateTokenSelection, "pllm/private-token-selection/v1", {"comparison": "comparison", "roles": "roles"}),
    (protocols.PrivateTokenFeedback, "pllm/private-token-feedback/v1", {"selection": "selection", "input": "input", "state": "state"}),
    (preparation.HECiphertextBirth, "pllm/he-ciphertext-birth/v1", {"key_graph": "key_graph", "numeric": "numeric"}),
    (schedulers.HELevelErrorSchedule, "pllm/he-level-error-schedule/v1", {"numeric": "numeric", "key_graph": "key_graph"}),
    (preparation.HEKeyContext, "pllm/he-key-context/v1", {"prime_chain": None, "key_parameters": None}),
    (preparation.HEKeyContext, "pllm/he-key-context/v1", {"prime_chain": None}),
    (preparation.HEKeyContext, "pllm/he-key-context/v1", {"key_parameters": None}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "profile": "paper-ring64-q18", "extension": "ring32-q12-where-bound"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "source": "rosetta-53-bit-port-unresolved"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "profile": "paper-ring64-q18"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "extension": "ring32-q12-where-bound"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "source": " arbitrary__source label "}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "profile": "paper-ring64-q18", "source": "source-label"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "extension": "ring32-q12-where-bound", "source": "source-label"}),
    (numeric.SourceNumericDomainCertificates, "pllm/source-numeric-domain-certificates/v1", {"parameter_authority": None, "profile": "paper-ring64-q18", "extension": "ring32-q12-where-bound", "source": "source-label"}),
    *((protocols.HEEncryptedMultiplyReduce, "pllm/he-encrypted-multiply-reduce/v1", {"operation": operation, "numeric": "numeric"}) for operation in OPERATIONS[1:]),
)


class StringSubclass(str):
    pass


class IntSubclass(int):
    pass


def _wire_bytes(component):
    return json.dumps(component.to_spec(), ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_matrix_has_ninety_one_unique_classes_and_wire_ids():
    assert len(CASES) == len({row[2] for row in CASES}) == len({row[3] for row in CASES}) == 91
    assert tuple(len(group[3]) for group in GROUPS) == (13, 6, 24, 13, 8, 6, 3, 5, 3, 3, 4, 2, 1)


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", CASES)
def test_family_leaf_exports_signature_and_closed_schema(public, leaf, cls, identity, params, category):
    assert getattr(public, cls.__name__) is getattr(leaf, cls.__name__) is cls
    assert issubclass(cls, ComponentRef)
    properties = PROPERTIES.get(cls, {})
    optional = OPTIONAL.get(cls, ())
    required = [name for name in properties if name not in optional]
    signature = inspect.signature(cls)
    assert set(signature.parameters) == set(properties)
    for name, parameter in signature.parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if name in optional:
            assert parameter.default is not inspect.Parameter.empty
            if cls is not preparation.HEKeyContext:
                assert parameter.default is None
            else:
                assert parameter.default is not None  # Private omitted marker, never wire data.
        else:
            assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        cls("invalid")
    descriptor = cls.describe()
    assert descriptor is cls.descriptor is cls(**params).describe()
    assert descriptor.component == identity and descriptor.category == category
    assert descriptor.provider == "pllm" and descriptor.distribution == "pllm.run"
    assert descriptor.version == descriptor.category_version == "1"
    assert descriptor.lifecycle_phase == "compile"
    assert descriptor.to_dict()["parameter_schema"] == {
        "type": "object", "properties": properties,
        "required": required, "additionalProperties": False,
    }


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", CASES)
def test_direct_and_mapping_construction_have_genuine_legacy_wire(public, leaf, cls, identity, params, category):
    legacy = ComponentRef(identity, params)
    expected = {"component": identity, "params": params}
    for component in (cls(**params), cls.from_params(dict(params)), cls.from_params(MappingProxyType(dict(params)))):
        assert type(component) is cls
        assert component.to_spec() == expected == legacy.to_spec()
        assert _wire_bytes(component) == _wire_bytes(legacy)
        assert component.get_params() == params
        assert component.describe() is cls.describe()


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", CASES)
def test_frozen_fields_and_fresh_defensive_public_mappings(public, leaf, cls, identity, params, category):
    original = dict(params)
    component = cls.from_params(original)
    original["caller_mutation"] = True
    expected = {"component": identity, "params": params}
    assert component.to_spec() == expected
    for name, value in (("component", "pllm/changed/v1"), ("params", {})):
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            setattr(component, name, value)
    with pytest.raises(TypeError):
        component.params["mutation"] = True
    for deep in (False, True):
        first = component.get_params(deep=deep)
        second = component.get_params(deep=deep)
        assert type(first) is type(second) is dict and first is not second
        assert first == second == params
        first["mutation"] = True  # Insertion also exercises empty parameter maps.
        assert second == params and component.to_spec() == expected
    first_spec, second_spec = component.to_spec(), component.to_spec()
    assert first_spec is not second_spec and first_spec["params"] is not second_spec["params"]
    first_spec["params"]["mutation"] = True
    first_spec["component"] = "pllm/changed/v1"
    assert second_spec == expected == component.to_spec()


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", CASES)
def test_unknown_reserved_and_nonmapping_params_rejected(public, leaf, cls, identity, params, category):
    for name in ("unknown", "component", "params"):
        with pytest.raises(TypeError):
            cls(**{**params, name: "invalid"})
        with pytest.raises(ConfigurationError):
            cls.from_params({**params, name: "invalid"})
    for invalid in (None, [], [list(item) for item in params.items()], "{}", 1, True):
        with pytest.raises(ConfigurationError):
            cls.from_params(invalid)


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", tuple(
    row for row in CASES
    if set(PROPERTIES.get(row[2], {})) - set(OPTIONAL.get(row[2], ()))
))
def test_each_required_argument_missing_is_rejected(public, leaf, cls, identity, params, category):
    for name in PROPERTIES.get(cls, {}):
        if name in OPTIONAL.get(cls, ()):
            continue
        missing = dict(params)
        del missing[name]
        with pytest.raises(TypeError):
            cls(**missing)
        with pytest.raises(ConfigurationError):
            cls.from_params(missing)


@pytest.mark.parametrize("public,leaf,cls,identity,params,category", CASES)
def test_real_registry_and_configuration_restore_exact_class(public, leaf, cls, identity, params, category):
    from pllm.components import create_component, get
    from pllm.configuration import _component_from_spec

    assert get(identity) is cls
    legacy = ComponentRef(identity, params)
    for restored in (create_component(identity, dict(params)), _component_from_spec(legacy.to_spec(), "component")):
        assert type(restored) is cls
        assert restored.to_spec() == {"component": identity, "params": params}
        assert _wire_bytes(restored) == _wire_bytes(legacy)


@pytest.mark.parametrize("cls,identity,params", VARIANTS)
def test_independent_positive_variants_and_registry_wire(cls, identity, params):
    from pllm.components import create_component
    from pllm.configuration import _component_from_spec

    legacy = ComponentRef(identity, params)
    for component in (cls(**params), cls.from_params(MappingProxyType(dict(params))),
                      create_component(identity, params), _component_from_spec(legacy.to_spec(), "component")):
        assert type(component) is cls
        assert component.to_spec() == {"component": identity, "params": params}
        assert _wire_bytes(component) == _wire_bytes(legacy)


@pytest.mark.parametrize("cls,names", tuple(OPTIONAL.items()))
def test_optional_constructor_none_omits_but_wire_null_rejects(cls, names):
    if cls is preparation.HEKeyContext:
        # Null is real key-context data, independently accepted per field.
        assert cls().to_spec() == {"component": "pllm/he-key-context/v1", "params": {}}
        for params in ({"prime_chain": None}, {"key_parameters": None},
                       {"prime_chain": None, "key_parameters": None}):
            assert cls(**params).get_params() == cls.from_params(params).get_params() == params
        return
    base = {"parameter_authority": None} if cls is numeric.SourceNumericDomainCertificates else {}
    assert cls(**base, **{name: None for name in names}).get_params() == base
    for name in names:
        assert cls(**base, **{name: None}).get_params() == base
        with pytest.raises(ConfigurationError):
            cls.from_params({**base, name: None})


@pytest.mark.parametrize("cls,properties", tuple(PROPERTIES.items()))
def test_nonnull_values_and_exact_literal_types_are_checked(cls, properties):
    base = next(row[4] for row in CASES if row[2] is cls)
    for name, schema in properties.items():
        kind = schema["type"]
        if kind == "null":
            invalid_values = (False, True, 0, 1, "source", "calibrated-v1", [], {}, object())
        elif kind == "integer":
            invalid_values = (None, True, False, 8.0, "8", IntSubclass(8), 7, 9, [], {})
        elif kind == "boolean":
            invalid_values = (None, 0, 1, "true", not schema["enum"][0], [], {})
        else:
            valid = schema.get("enum", ["unresolved-slot"])[0]
            invalid_values = (True, 1, StringSubclass(valid), "", "   ", [], {})
            if not (cls is numeric.SourceNumericDomainCertificates and name == "source"):
                invalid_values += ("bad__slot",)
            if name not in OPTIONAL.get(cls, ()):
                invalid_values += (None,)
        for invalid in invalid_values:
            with pytest.raises(ConfigurationError):
                cls(**{**base, name: invalid})
            with pytest.raises(ConfigurationError):
                cls.from_params({**base, name: invalid})
        if kind == "string" and "enum" not in schema:
            # Logical references are not looked up or stripped at construction.
            value = " unresolved slot / independent label "
            assert cls(**{**base, name: value}).get_params()[name] == value
            assert cls.from_params({**base, name: value}).get_params()[name] == value
            if cls is numeric.SourceNumericDomainCertificates and name == "source":
                assert cls(**{**base, name: "__"}).get_params()[name] == "__"
                assert cls.from_params({**base, name: "__"}).get_params()[name] == "__"
