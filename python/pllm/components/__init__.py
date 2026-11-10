"""Public component configuration and built-in discovery contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from pllm.components._planned import (
    NotYetImplementedError,
    PlannedComponent,
    planned_component,
    planned_components,
    require_implemented_identity,
)
from pllm.components._model_capabilities import (
    ModelCapabilityStub,
    ModelCapabilityUnavailable,
    model_capabilities,
    model_capability,
)
from pllm.configuration import ComponentDescriptor, ComponentRef, ConfigurationError

if TYPE_CHECKING:
    from pllm.providers import ProviderDescriptor


@lru_cache(maxsize=1)
def _builtin_classes() -> tuple[type[ComponentRef], ...]:
    """Delay family imports so each family can expose planned component symbols."""
    from pllm.correlation import SecretReplicationPcf, SeededExpansion, RaaMask
    from pllm.correlation import ReplicatedPairwisePrss
    from pllm.correlation import DistributedFerretOt
    from pllm.kernels import AppleMetal, Cpu
    from pllm.numeric import SourceFloat32, Q11I16Ring64Boundary, ExactSourceFieldBoundaries
    from pllm.numeric import ReplicatedRing64Q18Edges, Ring50Q12Edges
    from pllm.numeric import SourceCalibration, NumericDomain, SourceNumericDomainCertificates
    from pllm.metrics import (
        Accuracy,
        Communication,
        Cost,
        CpuTime,
        Energy,
        Latency,
        Memory,
        Perplexity,
        ReferenceAgreement,
        Throughput,
    )
    from pllm.nonlinear import (
        SharedRingTruncation, SharedComparisonMux, ShiftedClipRepeatedSquareSoftmax,
        SharedReciprocalInverseRootNormalization, PiecewisePolynomialActivation, SharedRingResidualBias,
        CalibratedPolynomialExp, StandardGoldschmidtReciprocal, CompositePolynomialActivation,
        InverseRoot, HEMaskedSoftmax, SourceNormalization, HERepeatedSquaringSoftmax,
        HEPiecewisePolynomialActivation, HERowNormalization, HESignComposition,
        HELayerNorm, HESquareNormalizeSoftmax, HEPolynomialActivation,
        LWEHighPrecisionComparison, RLWEEncryptedAccumulatorSegmentedAffine, HEEvalMod,
        HENormalization, HESoftmaxSquareNormalize, HEActivationBridge,
        ReducedRingTruncation, SameRingGapARS, XorMaskedDReLU, EarlyFloorResidualActivation,
        IntegerRsqrtTableNorm, SourceRing50Q12Rotary,
        ReplicatedTruncAprQuarterGap, Q18PiecewiseSiluSourceOrder, Q18SourceReciprocal,
        Q18SourceRsqrt, Q18RmsNormEpsilon1, Q18SourceRotary, Q18ShiftedClipExp5Reciprocal,
        FSSSignedFloorRescale, FSSSigned38InputSiLU, FSSSumSquaresTableRMSNorm,
        FSSProductFloorRotary, FSSSourceTableSoftmax,
        ArithmeticGarblingSiluQ7,
        BinaryTableGatedMultiplyQ7,
        CompactPiecewiseActivation,
        CurlWaveletEncodedTable,
        CurlWaveletGatedMultiplyQ9,
        F32Activation,
        LogRowGatedMultiplyQ9,
        CompactGatedMultiplyQ12,
        MaskedWireGatedMultiplyQ12,
        R03CrtGatedMultiplyQ7,
        SharedRescale,
        PackedEncryptedSoftmax,
    )
    from pllm.operators import (
        F32RmsNorm,
        F32Rotary,
        F32CausalAttention,
        F32Multiply,
        F32AddBias,
        F32Add,
        FullVocabularyHead,
        GreedySelection,
        PublicF32Lookup,
    )
    from pllm.passes import CipherPruneTokenPruning, KvCacheEviction
    from pllm.preparation import (
        HEKeyContext, HEDualKeyContext, HECiphertextBirth, HEKeyGraph, OneUsePartyLocalConversion,
        TrustedDealerOneUseFSSMaterial, OneUseRoleLocalMaterial,
        BFVCorrelations,
        HEAuthenticatedPreprocessing,
        ModelAwareCorrections,
        PreparedInventory,
    )
    from pllm.quantization import PublicPerChannelEqualized, SymmetricPerRow
    from pllm.protocols import (
        PrivateTokenShare, CoefficientPrivateLookup, CoefficientObliviousLinearTransform,
        SimdBilinearOle, ArithmeticToHomomorphic, TraceInterleaveRepacking,
        CoefficientHomomorphicToArithmetic, FloodedSimdHomomorphicToArithmetic,
        CrtRingConversion, SignedRingExtension, SourcePositionOperators,
        SharedStableArgmax, PrivateFullPrefixFeedback,
        CkksRectangularLinear, InterleavedQK, InterleavedWeightedV, RoutedGQA, HERotaryTransform,
        PrivateTokenSelection, PrivateTokenFeedback, HECoefficientCompression,
        HECommutableLinear, HEColumnProduct, HELayoutTransform, HEAttentionValue,
        HEFoldSelection, HEBootstrap, HEResidualAdd, HEPrivateEmbedding, ClassifierOutputPolicy,
        HESourceEmbeddingPositionSegment, HEAttentionProducts, HEHeadScaleMask, HEPackedLinear,
        HEClsPooler, HENspTwoLogitHead, NspLogitsOnlyOutputPolicy,
        AuthenticatedClientInputExport, HEEncryptedMultiplyReduce, HESlotCoefficientTransform,
        HERingSwitch, HECoefficientExtraction, HEKeySwitch, HEModulusRaise, HECoefficientSlotTransform,
        RingPublicCoefficientDot, NegativeMaskMatrixProduct, NegativeMaskElementwiseProduct,
        MaskedRingAddSubProduct, BoundedDpfPublicTable, XorPredicateFiveWordSelect,
        CausalMaxNexpReciprocalAttention, StablePrivateArgmaxEmbedding,
        AuthenticatedNativeInput, ConsumingShareToCommonMask,
        ReplicatedPrivateTokenInput, ReplicatedCompleteTableLookup,
        ReplicatedPublicMatmulFullSum, ReplicatedSecretMultiply, ReplicatedRingAddBias,
        ReplicatedArithmeticBooleanConversion, ReplicatedCausalQkSoftmaxAv,
        ReplicatedCompleteStableArgmax, ReplicatedFixedCountFinalTokenRelease,
        AuthenticatedDirectedThreePartyTransport,
        FSSPublicTableLookup, FSSPublicDot, FSSPrivateDot, FSSSecretMultiply,
        FSSAdd, FSSArgmax, FSSAuthenticatedRoleTransport,
        PublicFieldMatvec,
        AuthenticatedConfidentialRoleTransport,
        TwoShareInput,
        VerifiedPublicF32Lookup,
        ClientF32DecoderRegions,
        BlindedLinear,
        ClientBundleTransport,
        CleartextLinear,
        DirectFHE,
        GuardedLinear,
        LogRowGarbledLookup,
        PrivateEmbeddingLookup,
        MaverickDelegatedLinear,
        OpenWeightVerifiableOutsourcing,
        PumaThreePartyDecoder,
        FuseFssSourceDecoder,
        PackedEncryptedLinear,
        MaskedLinear,
        SecureLinear,
        TwoOnlineOffsetLinear,
    )
    from pllm.roles import (
        ClientKeyServerEvaluationRoles, ClientSecretServerWeightSecretRoles,
        ClientKeyServerConfidentialWeightRoles, TwoOwnerPrivateInputPrivateModelRoles,
        ClientMaskSingleServerRoles,
        ClientOnlyRoles,
        ClientLinearRoles,
        ClientPrefixLayers,
        Inference,
        OutputHeadAtInference,
        PrivateLookupAndFullHead,
        PreparedBoundaryRoles,
        PreparedProviderRoles,
        TwoOnlineOffsetRoles,
        TwoOnlineDelegationRoles,
        ThreePartyReplicatedRoles,
        TrustedDealerTwoShareRoles,
    )
    from pllm.schedulers import (
        HELevelErrorSchedule, HELevelSchemePlacement, SetupOnlineRetirementSchedule,
        FSSOnlineSchedule, BoundedConsumingProtectedDag,
        MlpPrefillTiling,
        BoundedIndependentElementsProtectedTensorSchedule,
        ChunkedIndependentLanesProtectedTensorSchedule,
        IndependentLanesProtectedTensorSchedule,
        ScalarProtectedTensorSchedule,
    )
    from pllm.state import ClientLocalKv, ClientPrefixReuse, PublicPrefixCapsule, ReplicatedPrivatePrefix, FixedCountPrivatePrefix
    from pllm.state import TentativeStateCommit
    from pllm.state import InterleavedKVTransaction, EncryptedKVTransaction, SharedFullPrefixState
    from pllm.sources import AuthenticatedSourceInputProducer, PinnedSemanticSourceBindings
    from pllm.sources import PublicCommittedQ18Source, FSSSourceBinding
    from pllm.sources import AuthenticatedSourceBindings, AuthenticatedOriginalProducer
    from pllm.tokenization import IndexedTokenizer
    from pllm.verification import FreivaldsVerify, LinearIntegrity, PublishedLinearVerify
    from pllm.verification import PerEdgeRangeWorkAdmission
    from pllm.verification import TwoLevelCodedBatchVerification, ValidatedMatrixCodePublication, OriginalDotWindow

    classes = (
        TwoShareInput,
        VerifiedPublicF32Lookup,
        ClientF32DecoderRegions,
        Q11I16Ring64Boundary,
        Accuracy,
        AppleMetal,
        ArithmeticGarblingSiluQ7,
        BFVCorrelations,
        BinaryTableGatedMultiplyQ7,
        BlindedLinear,
        BoundedIndependentElementsProtectedTensorSchedule,
        ChunkedIndependentLanesProtectedTensorSchedule,
        CleartextLinear,
        ClientLocalKv,
        ClientOnlyRoles,
        ClientPrefixReuse,
        PublicPrefixCapsule,
        IndexedTokenizer,
        Communication,
        CompactPiecewiseActivation,
        Cost,
        CpuTime,
        Cpu,
        CurlWaveletEncodedTable,
        CurlWaveletGatedMultiplyQ9,
        LogRowGatedMultiplyQ9,
        CompactGatedMultiplyQ12,
        DirectFHE,
        Energy,
        FreivaldsVerify,
        GuardedLinear,
        HEAuthenticatedPreprocessing,
        IndependentLanesProtectedTensorSchedule,
        Inference,
        KvCacheEviction,
        CipherPruneTokenPruning,
        Latency,
        LinearIntegrity,
        LogRowGarbledLookup,
        PrivateEmbeddingLookup,
        MaverickDelegatedLinear,
        OpenWeightVerifiableOutsourcing,
        PumaThreePartyDecoder,
        FuseFssSourceDecoder,
        FixedCountPrivatePrefix,
        TrustedDealerTwoShareRoles,
        ReplicatedPrivatePrefix,
        ThreePartyReplicatedRoles,
        PackedEncryptedLinear,
        MaskedLinear,
        Memory,
        PackedEncryptedSoftmax,
        MaskedWireGatedMultiplyQ12,
        ModelAwareCorrections,
        MlpPrefillTiling,
        OutputHeadAtInference,
        PrivateLookupAndFullHead,
        PreparedBoundaryRoles,
        ClientPrefixLayers,
        ClientLinearRoles,
        PreparedInventory,
        ClientBundleTransport,
        Perplexity,
        PreparedProviderRoles,
        PublishedLinearVerify,
        PublicPerChannelEqualized,
        R03CrtGatedMultiplyQ7,
        ReferenceAgreement,
        ScalarProtectedTensorSchedule,
        SeededExpansion,
        SecretReplicationPcf,
        SecureLinear,
        SharedRescale,
        SymmetricPerRow,
        SourceFloat32,
        Throughput,
        TwoOnlineOffsetLinear,
        TwoOnlineOffsetRoles,
        TwoOnlineDelegationRoles,
        F32Activation,
        F32RmsNorm,
        F32Rotary,
        F32CausalAttention,
        F32Multiply,
        F32AddBias,
        F32Add,
        FullVocabularyHead,
        GreedySelection,
        PublicFieldMatvec,
        RaaMask,
        TwoLevelCodedBatchVerification,
        ValidatedMatrixCodePublication,
        OriginalDotWindow,
        TentativeStateCommit,
        AuthenticatedSourceInputProducer,
        ExactSourceFieldBoundaries,
        ClientMaskSingleServerRoles,
        AuthenticatedConfidentialRoleTransport,
        PinnedSemanticSourceBindings,
        PublicF32Lookup,
        ReplicatedPrivateTokenInput,
        ReplicatedCompleteTableLookup,
        ReplicatedPublicMatmulFullSum,
        ReplicatedSecretMultiply,
        ReplicatedRingAddBias,
        ReplicatedArithmeticBooleanConversion,
        ReplicatedCausalQkSoftmaxAv,
        ReplicatedCompleteStableArgmax,
        ReplicatedFixedCountFinalTokenRelease,
        AuthenticatedDirectedThreePartyTransport,
        ReplicatedTruncAprQuarterGap,
        Q18PiecewiseSiluSourceOrder,
        Q18SourceReciprocal,
        Q18SourceRsqrt,
        Q18RmsNormEpsilon1,
        Q18SourceRotary,
        Q18ShiftedClipExp5Reciprocal,
        ReplicatedRing64Q18Edges,
        PublicCommittedQ18Source,
        ReplicatedPairwisePrss,
        PerEdgeRangeWorkAdmission,
        FSSPublicTableLookup,
        FSSPublicDot,
        FSSPrivateDot,
        FSSSecretMultiply,
        FSSAdd,
        FSSArgmax,
        FSSAuthenticatedRoleTransport,
        FSSSignedFloorRescale,
        FSSSigned38InputSiLU,
        FSSSumSquaresTableRMSNorm,
        FSSProductFloorRotary,
        FSSSourceTableSoftmax,
        Ring50Q12Edges,
        FSSSourceBinding,
        TrustedDealerOneUseFSSMaterial,
        FSSOnlineSchedule,
        OneUseRoleLocalMaterial,
        BoundedConsumingProtectedDag,
        RingPublicCoefficientDot,
        NegativeMaskMatrixProduct,
        NegativeMaskElementwiseProduct,
        MaskedRingAddSubProduct,
        BoundedDpfPublicTable,
        XorPredicateFiveWordSelect,
        CausalMaxNexpReciprocalAttention,
        StablePrivateArgmaxEmbedding,
        AuthenticatedNativeInput,
        ConsumingShareToCommonMask,
        ReducedRingTruncation,
        SameRingGapARS,
        XorMaskedDReLU,
        EarlyFloorResidualActivation,
        IntegerRsqrtTableNorm,
        SourceRing50Q12Rotary,
        PrivateTokenShare,
        CoefficientPrivateLookup,
        CoefficientObliviousLinearTransform,
        SimdBilinearOle,
        ArithmeticToHomomorphic,
        TraceInterleaveRepacking,
        CoefficientHomomorphicToArithmetic,
        FloodedSimdHomomorphicToArithmetic,
        CrtRingConversion,
        SignedRingExtension,
        SourcePositionOperators,
        SharedStableArgmax,
        PrivateFullPrefixFeedback,
        SharedRingTruncation,
        SharedComparisonMux,
        ShiftedClipRepeatedSquareSoftmax,
        SharedReciprocalInverseRootNormalization,
        PiecewisePolynomialActivation,
        SharedRingResidualBias,
        CkksRectangularLinear,
        InterleavedQK,
        InterleavedWeightedV,
        RoutedGQA,
        HERotaryTransform,
        PrivateTokenSelection,
        PrivateTokenFeedback,
        HECoefficientCompression,
        HECommutableLinear,
        HEColumnProduct,
        HELayoutTransform,
        HEAttentionValue,
        HEFoldSelection,
        HEBootstrap,
        HEResidualAdd,
        HEPrivateEmbedding,
        ClassifierOutputPolicy,
        HESourceEmbeddingPositionSegment,
        HEAttentionProducts,
        HEHeadScaleMask,
        HEPackedLinear,
        HEClsPooler,
        HENspTwoLogitHead,
        NspLogitsOnlyOutputPolicy,
        CalibratedPolynomialExp,
        StandardGoldschmidtReciprocal,
        CompositePolynomialActivation,
        InverseRoot,
        HEMaskedSoftmax,
        SourceNormalization,
        HERepeatedSquaringSoftmax,
        HEPiecewisePolynomialActivation,
        HERowNormalization,
        HESignComposition,
        HELayerNorm,
        HESquareNormalizeSoftmax,
        HEPolynomialActivation,
        AuthenticatedClientInputExport,
        HEEncryptedMultiplyReduce,
        HESlotCoefficientTransform,
        HERingSwitch,
        HECoefficientExtraction,
        HEKeySwitch,
        HEModulusRaise,
        HECoefficientSlotTransform,
        LWEHighPrecisionComparison,
        RLWEEncryptedAccumulatorSegmentedAffine,
        HEEvalMod,
        HENormalization,
        HESoftmaxSquareNormalize,
        HEActivationBridge,
        SourceCalibration,
        NumericDomain,
        SourceNumericDomainCertificates,
        HEKeyContext,
        HEDualKeyContext,
        HECiphertextBirth,
        HEKeyGraph,
        OneUsePartyLocalConversion,
        HELevelErrorSchedule,
        HELevelSchemePlacement,
        SetupOnlineRetirementSchedule,
        InterleavedKVTransaction,
        EncryptedKVTransaction,
        SharedFullPrefixState,
        ClientKeyServerEvaluationRoles,
        ClientSecretServerWeightSecretRoles,
        ClientKeyServerConfidentialWeightRoles,
        TwoOwnerPrivateInputPrivateModelRoles,
        AuthenticatedSourceBindings,
        AuthenticatedOriginalProducer,
        DistributedFerretOt,
    )
    if len({component.describe().component for component in classes}) != len(classes):
        raise RuntimeError("built-in component identities must be unique")
    return classes


__all__ = [
    "ComponentDescriptor",
    "ComponentRef",
    "NotYetImplementedError",
    "ModelCapabilityStub",
    "ModelCapabilityUnavailable",
    "PlannedComponent",
    "create_component",
    "get",
    "get_component",
    "list_component_classes",
    "list_components",
    "planned_component",
    "planned_components",
    "model_capability",
    "model_capabilities",
]


def list_component_classes(
    *, providers: Iterable[ProviderDescriptor] = ()
) -> tuple[type[ComponentRef], ...]:
    classes = list(_builtin_classes())
    providers = tuple(providers)
    if providers:
        from pllm.providers import provider_component_classes

        classes.extend(provider_component_classes(providers))
    by_identity = {component.describe().component: component for component in classes}
    if len(by_identity) != len(classes):
        raise ConfigurationError("component identities must be unique")
    return tuple(by_identity[identity] for identity in sorted(by_identity))


def list_components(
    *, providers: Iterable[ProviderDescriptor] = ()
) -> tuple[ComponentDescriptor, ...]:
    """Return descriptors in stable component-identity order."""
    return tuple(component.describe() for component in list_component_classes(providers=providers))


def get(identity: str, *, providers: Iterable[ProviderDescriptor] = ()) -> type[ComponentRef]:
    if type(identity) is not str:
        raise TypeError("component identity must be a string")
    require_implemented_identity(identity)
    providers = tuple(providers)
    by_identity = {
        component.describe().component: component
        for component in list_component_classes(providers=providers)
    }
    try:
        return by_identity[identity]
    except KeyError:
        message = "component not found" if providers else "built-in component not found"
        raise KeyError(message) from None


def get_component(
    identity: str, *, providers: Iterable[ProviderDescriptor] = ()
) -> ComponentDescriptor:
    """Return one descriptor by canonical component identity."""
    return get(identity, providers=providers).describe()


def create_component(
    identity: str,
    params: Mapping[str, Any],
    *,
    providers: Iterable[ProviderDescriptor] = (),
) -> ComponentRef:
    if not isinstance(params, Mapping):
        raise TypeError("component params must be a mapping")
    try:
        component = get(identity, providers=providers)
    except KeyError:
        return ComponentRef(identity, params)
    schema = component.describe().parameter_schema
    if isinstance(schema, Mapping):
        properties = set(schema.get("properties", {}))
        required = set(schema.get("required", ()))
        allowed = properties | required
        unknown = set(params) - allowed
        missing = required - set(params)
        if unknown:
            raise ConfigurationError(f"component params have unknown fields: {sorted(unknown)}")
        if missing:
            raise ConfigurationError(f"component params have missing fields: {sorted(missing)}")
    return component.from_params(params)
