"""PUMA source-Q18 composition and desired, not yet measured, benchmark target."""

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
from pllm.correlation import ReplicatedPairwisePrss
from pllm.numeric import ReplicatedRing64Q18Edges
from pllm.nonlinear import (
    ReplicatedTruncAprQuarterGap, Q18PiecewiseSiluSourceOrder, Q18SourceReciprocal,
    Q18SourceRsqrt, Q18RmsNormEpsilon1, Q18SourceRotary, Q18ShiftedClipExp5Reciprocal,
)
from pllm.protocols import (
    ReplicatedPrivateTokenInput, ReplicatedCompleteTableLookup,
    ReplicatedPublicMatmulFullSum, ReplicatedSecretMultiply, ReplicatedRingAddBias,
    ReplicatedArithmeticBooleanConversion, ReplicatedCausalQkSoftmaxAv,
    ReplicatedCompleteStableArgmax, ReplicatedFixedCountFinalTokenRelease,
    AuthenticatedDirectedThreePartyTransport,
)
from pllm.protocols import PumaThreePartyDecoder
from pllm.roles import ThreePartyReplicatedRoles
from pllm.sources import ModelSource
from pllm.sources import PublicCommittedQ18Source
from pllm.verification import PerEdgeRangeWorkAdmission
from pllm.state import ReplicatedPrivatePrefix

if __package__:
    from .common import BUDGET, MODEL
else:
    from common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    """Compose components; generic overrides are unqualified configurations.

    Use the component map and generic parameter paths, not typed properties.
    """
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    return Experiment(
        "puma",
        Pipeline(
            model=normalized_model,
            profile="research.puma_three_party_cpu",
            components={
                "linear": PumaThreePartyDecoder(),
                "topology": ThreePartyReplicatedRoles(),
                "state": ReplicatedPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root=".pllm/papers/puma"),
        budget,
    )


def benchmark_experiment(*, model=None, budget=None):
    """Declare the desired primitive composition, not a measured native candidate.

    Proposed component IDs remain unbound: existing admission must refuse them
    until implementations satisfy their numerical and security contracts.
    """
    model = MODEL if model is None else model
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    budget = ExecutionBudget(1, 32, 2) if budget is None else budget
    return Experiment(
        "puma",
        Pipeline(
            model=normalized_model,
            components={
                "numeric": ReplicatedRing64Q18Edges(revision="source-order-v1"),
                # Public committed weights; private prompt; at most one
                # honest-but-curious worker may be corrupted.
                "weights": PublicCommittedQ18Source(),
                "input": ReplicatedPrivateTokenInput(),
                "embedding": ReplicatedCompleteTableLookup(),
                # Preserve the full Q36 sum, then fresh stochastic Q18 truncation.
                "linear": ReplicatedPublicMatmulFullSum(),
                "multiply": ReplicatedSecretMultiply(),
                "truncate": ReplicatedTruncAprQuarterGap(),
                "add_bias": ReplicatedRingAddBias(),
                "conversion": ReplicatedArithmeticBooleanConversion(),
                "activation": Q18PiecewiseSiluSourceOrder(),
                "reciprocal": Q18SourceReciprocal(),
                "rsqrt": Q18SourceRsqrt(),
                # RMSNorm epsilon is integer 1, not a floating-point epsilon.
                "normalization": Q18RmsNormEpsilon1(),
                "rotary": Q18SourceRotary(),
                "attention": ReplicatedCausalQkSoftmaxAv(),
                # Shifted softmax uses strict -14 clipping and five squares.
                "softmax": Q18ShiftedClipExp5Reciprocal(),
                # Complete stable argmax: lowest index wins ties.
                "selection": ReplicatedCompleteStableArgmax(),
                # Fixed-count final token IDs only; no intermediate release or KV reuse.
                "output": ReplicatedFixedCountFinalTokenRelease(),
                "randomness": ReplicatedPairwisePrss(),
                "transport": AuthenticatedDirectedThreePartyTransport(),
                "admission": PerEdgeRangeWorkAdmission(),
                "topology": ThreePartyReplicatedRoles(),
                "state": ReplicatedPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/puma-deployment"),
        budget,
    )


def main(argv=None):
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "puma", argv, task="response", max_input_tokens=32, max_new_tokens=2
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
