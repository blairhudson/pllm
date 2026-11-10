from pllm.configuration import ComponentDescriptor, ComponentRef
from pllm.components._planned import PendingComponent as PendingMethod
from pllm.nonlinear._float_declarations import F32Activation as F32Activation
from pllm.nonlinear._conversion_declarations import (
    SharedRingTruncation as SharedRingTruncation,
    SharedComparisonMux as SharedComparisonMux,
    ShiftedClipRepeatedSquareSoftmax as ShiftedClipRepeatedSquareSoftmax,
    SharedReciprocalInverseRootNormalization as SharedReciprocalInverseRootNormalization,
    PiecewisePolynomialActivation as PiecewisePolynomialActivation,
    SharedRingResidualBias as SharedRingResidualBias,
)
from pllm.nonlinear._he_declarations import (
    CalibratedPolynomialExp as CalibratedPolynomialExp,
    StandardGoldschmidtReciprocal as StandardGoldschmidtReciprocal,
    CompositePolynomialActivation as CompositePolynomialActivation,
    InverseRoot as InverseRoot,
    HEMaskedSoftmax as HEMaskedSoftmax,
    SourceNormalization as SourceNormalization,
    HERepeatedSquaringSoftmax as HERepeatedSquaringSoftmax,
    HEPiecewisePolynomialActivation as HEPiecewisePolynomialActivation,
    HERowNormalization as HERowNormalization,
    HESignComposition as HESignComposition,
    HELayerNorm as HELayerNorm,
    HESquareNormalizeSoftmax as HESquareNormalizeSoftmax,
    HEPolynomialActivation as HEPolynomialActivation,
)
from pllm.nonlinear._heterogeneous_declarations import (
    LWEHighPrecisionComparison as LWEHighPrecisionComparison,
    RLWEEncryptedAccumulatorSegmentedAffine as RLWEEncryptedAccumulatorSegmentedAffine,
    HEEvalMod as HEEvalMod,
    HENormalization as HENormalization,
    HESoftmaxSquareNormalize as HESoftmaxSquareNormalize,
    HEActivationBridge as HEActivationBridge,
)
from pllm.nonlinear._masked_declarations import (
    ReducedRingTruncation as ReducedRingTruncation,
    SameRingGapARS as SameRingGapARS,
    XorMaskedDReLU as XorMaskedDReLU,
    EarlyFloorResidualActivation as EarlyFloorResidualActivation,
    IntegerRsqrtTableNorm as IntegerRsqrtTableNorm,
    SourceRing50Q12Rotary as SourceRing50Q12Rotary,
)
from pllm.nonlinear._replicated_declarations import (
    ReplicatedTruncAprQuarterGap as ReplicatedTruncAprQuarterGap,
    Q18PiecewiseSiluSourceOrder as Q18PiecewiseSiluSourceOrder,
    Q18SourceReciprocal as Q18SourceReciprocal,
    Q18SourceRsqrt as Q18SourceRsqrt,
    Q18RmsNormEpsilon1 as Q18RmsNormEpsilon1,
    Q18SourceRotary as Q18SourceRotary,
    Q18ShiftedClipExp5Reciprocal as Q18ShiftedClipExp5Reciprocal,
)
from pllm.nonlinear._fss_declarations import (
    FSSSignedFloorRescale as FSSSignedFloorRescale,
    FSSSigned38InputSiLU as FSSSigned38InputSiLU,
    FSSSumSquaresTableRMSNorm as FSSSumSquaresTableRMSNorm,
    FSSProductFloorRotary as FSSProductFloorRotary,
    FSSSourceTableSoftmax as FSSSourceTableSoftmax,
)
from pllm._native import CompactQ7Reference, ScaledSiluQ7Reference
from collections.abc import Sequence

class NonlinearProtocol(ComponentRef): ...

class ArithmeticGarblingSiluQ7(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class BinaryTableGatedMultiplyQ7(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class R03CrtGatedMultiplyQ7(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class MaskedWireGatedMultiplyQ12(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, layout: str = "unfused") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class SharedRescale(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, bits: int = 16, shift: int = 7, rounding: str = "floor",
                 layout: str = "fused", backend: str = "interval_dcf") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...

class ShaftFixedPointTransformerOps(PendingMethod): ...
class CompactPiecewiseActivation(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, max_pieces: int = 4,
                 public_calibration_counts: Sequence[int] | None = None) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
class CurlWaveletEncodedTable(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, levels: int = 0, max_pieces: int = 4,
                 public_calibration_counts: Sequence[int] | None = None) -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
class CurlWaveletGatedMultiplyQ9(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, levels: int = 0, backend: str = "interval") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
class LogRowGatedMultiplyQ9(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, backend: str = "half_gates") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
class CompactGatedMultiplyQ12(NonlinearProtocol):
    descriptor: ComponentDescriptor
    def __init__(self, *, max_pieces: int = 8, degree: int = 3,
                 public_calibration_counts: Sequence[int] | None = None, backend: str = "half_gates") -> None: ...
    @classmethod
    def describe(cls) -> ComponentDescriptor: ...
class LlamaSecureMath(PendingMethod): ...
class SirnnSecureMath(PendingMethod): ...
class WaveletHomomorphicLookup(PendingMethod): ...
class CompactPrivateIntervalSelection(PendingMethod): ...
class CurlProbabilisticTruncation(PendingMethod): ...
class LlamaBoundedMathOperators(PendingMethod): ...
class BumbleBeeSharedNonlinear(PendingMethod): ...
class FuseFssNonlinearOperator(PendingMethod): ...
class SigmaSharedNonlinear(PendingMethod): ...
class LogRowPrivateRowIndex(PendingMethod): ...
class ScaleQuantConversion(PendingMethod): ...
class MpcFormerActivationApproximation(PendingMethod): ...
class CryptFlowExactTruncation(PendingMethod): ...
class FssFixedPointTruncation(PendingMethod): ...

def fit_compact_silu_q7_reference(
    public_calibration_counts: bytes, max_pieces: int = 4
) -> CompactQ7Reference: ...
def create_logrow_scaled_silu_q7_reference(max_abs: int) -> ScaledSiluQ7Reference: ...
