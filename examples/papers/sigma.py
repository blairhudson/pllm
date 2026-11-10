"""Sigma source-order target, composed from reusable primitive contracts.

This is a target recipe, not evidence of an admitted Sigma deployment. Missing
component implementations, source bindings, or authenticated role provisioning
must fail through normal SDK admission; there is no reference-runtime fallback.
The EzPC aa0f... source pin is an oracle only, never production execution or
authority for dealer randomness, client inputs, or protected-value handoff.
"""

from __future__ import annotations


def benchmark_experiment(*, model=None, budget=None):
    """Describe the bounded protected DAG without hiding its numeric regions.

    The common offline, revision-pinned Qwen target is an architectural extension:
    compatible RMSNorm/RoPE bindings need their own checked semantic revision.
    Selecting a checkpoint does not establish those bindings or source fidelity.
    """
    from pllm import (
        ConfigurationError,
        Deployment,
        ExecutionBudget,
        Experiment,
        Model,
        Pipeline,
    )
    from pllm.kernels import Cpu
    from pllm.nonlinear import (
        ReducedRingTruncation, SameRingGapARS, XorMaskedDReLU, EarlyFloorResidualActivation,
        IntegerRsqrtTableNorm, SourceRing50Q12Rotary,
    )
    from pllm.numeric import Ring50Q12Edges
    from pllm.preparation import OneUseRoleLocalMaterial
    from pllm.protocols import (
        RingPublicCoefficientDot, NegativeMaskMatrixProduct, NegativeMaskElementwiseProduct,
        MaskedRingAddSubProduct, BoundedDpfPublicTable, XorPredicateFiveWordSelect,
        CausalMaxNexpReciprocalAttention, StablePrivateArgmaxEmbedding,
        AuthenticatedNativeInput, ConsumingShareToCommonMask,
        AuthenticatedConfidentialRoleTransport,
    )
    from pllm.schedulers import BoundedConsumingProtectedDag
    from pllm.roles import TrustedDealerTwoShareRoles
    from pllm.sources import ModelSource, PinnedSemanticSourceBindings
    from pllm.state import FixedCountPrivatePrefix

    if model is None:
        if __package__:
            from .common import MODEL
        else:
            from common import MODEL

        model = MODEL
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    if budget is None:
        budget = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)

    return Experiment(
        "sigma",
        Pipeline(
            model=normalized_model,
            components={
                "public_dot": RingPublicCoefficientDot(),
                "full_head": RingPublicCoefficientDot(capacity="full-vocabulary"),
                "matrix_product": NegativeMaskMatrixProduct(),
                "gated_secret_multiply": NegativeMaskElementwiseProduct(),
                "ring_arithmetic": MaskedRingAddSubProduct(),
                # TR reduces the output ring; GapARS shifts within the same ring.
                # Neither may substitute for the other, even at equal Q precision.
                "truncation": ReducedRingTruncation(),
                "gapars": SameRingGapARS(),
                "drelu": XorMaskedDReLU(),
                "public_table": BoundedDpfPublicTable(),
                "predicate_select": XorPredicateFiveWordSelect(),
                # Floor at the source-prescribed early edge, before residual and
                # activation operations; a late floor changes the integer result.
                "residual_activation": EarlyFloorResidualActivation(),
                # These are unresolved Qwen extension binding requirements, not
                # authority: Sigma's original revision does not cover RMSNorm,
                # RoPE or gating. None marks missing semantic revision only,
                # never input credentials, dealer randomness or handoff authority.
                "normalization": IntegerRsqrtTableNorm(
                    kind="source", epsilon="source", semantic_revision=None,
                ),
                "rotary": SourceRing50Q12Rotary(semantic_revision=None),
                # Causal max -> negative exp -> reciprocal -> multiply. Preserve
                # source integer floors and unrenormalized softmax row sums.
                "attention": CausalMaxNexpReciprocalAttention(),
                "private_feedback": StablePrivateArgmaxEmbedding(),
                # Ring50/Q12 is not one global safety bound. Every source edge
                # needs its own signed range and truncation/GapARS guard proof.
                "numeric_edges": Ring50Q12Edges(
                    revision="sigma-source-order-v1", ranges="source-derived",
                ),
                # These are requirements, not placeholder credentials or seeds.
                # Production randomness needs real dealer-owned provisioning;
                # inputs need real authenticated native authority, never None or
                # a zero input standing in for authentication.
                "material": OneUseRoleLocalMaterial(),
                "input_authentication": AuthenticatedNativeInput(),
                # Conversion must consume the actual protected predecessor and
                # bind its owners, shape, ring and lifetime to the common mask.
                "handoff": ConsumingShareToCommonMask(),
                "schedule": BoundedConsumingProtectedDag(),
                "source_bindings": PinnedSemanticSourceBindings(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                "topology": TrustedDealerTwoShareRoles(),
                "state": FixedCountPrivatePrefix(),
                "kernels": Cpu(threads=1),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/sigma-deployment"),
        budget,
    )


def main(argv=None):
    """Use the shared public benchmark path; --help only parses arguments."""
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "sigma", argv, task="response", max_input_tokens=32, max_new_tokens=2
    )

    from pllm import benchmark

    return benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


if __name__ == "__main__":
    main()
