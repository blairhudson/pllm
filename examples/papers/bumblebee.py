"""Desired BumbleBee benchmark composition, not an admitted implementation.

The reusable component IDs below are proposed contracts, not claims that their
providers, numeric certificates, or complete Qwen execution already exist.
Ordinary SDK admission must reject unsupported composition; this example does
not relax admission or substitute a bounded local-dealer lookup demonstration.
"""

from pllm import (
    ConfigurationError,
    Deployment,
    ExecutionBudget,
    Experiment,
    Model,
    Pipeline,
    benchmark,
)
from pllm.kernels import Cpu
from pllm.correlation import DistributedFerretOt
from pllm.nonlinear import (
    SharedRingTruncation, SharedComparisonMux, ShiftedClipRepeatedSquareSoftmax,
    SharedReciprocalInverseRootNormalization, PiecewisePolynomialActivation, SharedRingResidualBias,
)
from pllm.numeric import SourceNumericDomainCertificates
from pllm.preparation import OneUsePartyLocalConversion
from pllm.protocols import (
    PrivateTokenShare, CoefficientPrivateLookup, CoefficientObliviousLinearTransform,
    SimdBilinearOle, ArithmeticToHomomorphic, TraceInterleaveRepacking,
    CoefficientHomomorphicToArithmetic, FloodedSimdHomomorphicToArithmetic,
    CrtRingConversion, SignedRingExtension, SourcePositionOperators,
    SharedStableArgmax, PrivateFullPrefixFeedback, AuthenticatedConfidentialRoleTransport,
)
from pllm.roles import TwoOwnerPrivateInputPrivateModelRoles
from pllm.schedulers import SetupOnlineRetirementSchedule
from pllm.sources import AuthenticatedOriginalProducer, PinnedSemanticSourceBindings
from pllm.state import SharedFullPrefixState
from pllm.sources import ModelSource

if __package__:
    from .common import MODEL, benchmark_arguments
else:
    from common import MODEL, benchmark_arguments


def benchmark_experiment(*, model=None, budget=None):
    """Declare independently checkable primitives, roles, and lifecycle bounds.

    MODEL is the shared offline, revision-pinned Qwen target, not evidence of
    support. GQA, RMSNorm, RoPE, and SiLU remain unsupported until their source
    semantics and numeric domains are bound to the relevant reusable operators.
    None denotes unresolved authority, never permission to choose parameters.
    """
    model = MODEL if model is None else model
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    budget = (
        ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
        if budget is None
        else budget
    )
    return Experiment(
        name="bumblebee",
        pipeline=Pipeline(
            model=model,
            components={
                "input": PrivateTokenShare(),
                "embedding": CoefficientPrivateLookup(),
                "linear": CoefficientObliviousLinearTransform(),
                "bilinear": SimdBilinearOle(),
                "arithmetic_to_homomorphic": ArithmeticToHomomorphic(),
                "repacking": TraceInterleaveRepacking(),
                "coefficient_to_arithmetic": CoefficientHomomorphicToArithmetic(),
                "simd_to_arithmetic": FloodedSimdHomomorphicToArithmetic(),
                "ring_conversion": CrtRingConversion(),
                # Distributed OT is required; a local trusted dealer is not an
                # implementation of this contract or of full-model inference.
                "ot": DistributedFerretOt(),
                "ring_extension": SignedRingExtension(),
                "truncation": SharedRingTruncation(),
                "comparison": SharedComparisonMux(),
                # Paper: Q18, shift -14, six squarings. Artifact: Q16, four
                # squarings; B-N1 discrepancy unresolved. Neither is silently
                # adopted as the Qwen numeric authority.
                "softmax": ShiftedClipRepeatedSquareSoftmax(),
                "normalization": SharedReciprocalInverseRootNormalization(),
                "activation": PiecewisePolynomialActivation(),
                "residual_bias": SharedRingResidualBias(),
                "position": SourcePositionOperators(),
                "selection": SharedStableArgmax(),
                "feedback": PrivateFullPrefixFeedback(),
                # FixedCountPrivatePrefix currently names client/worker_a/
                # worker_b placement. Compatibility with two input/model owners
                # is not established; do not adopt that state placement here.
                "state": SharedFullPrefixState(),
                "conversion_lifecycle": OneUsePartyLocalConversion(),
                "producer": AuthenticatedOriginalProducer(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                "topology": TwoOwnerPrivateInputPrivateModelRoles(),
                "source_bindings": PinnedSemanticSourceBindings(),
                # Exact primes, noise/flooding bounds, polynomial coefficients,
                # range certificates, and security parameters remain unadopted.
                # Profile labels identify desired domains, not certificates.
                "numeric_certificates": SourceNumericDomainCertificates(
                    parameter_authority=None,
                    profile="paper-ring64-q18",
                    extension="ring32-q12-where-bound",
                ),
                "schedule": SetupOnlineRetirementSchedule(),
                "kernels": Cpu(threads=1),
            },
        ),
        deployment=Deployment.local(
            root="target/paper-benchmarks/bumblebee-deployment"
        ),
        budget=budget,
    )


def main(argv=None):
    args = benchmark_arguments(
        "bumblebee", argv, task="response", max_input_tokens=32, max_new_tokens=2
    )
    return benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


if __name__ == "__main__":
    main()
