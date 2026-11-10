"""Maverick Protocol 2 destination composition, not an admitted implementation.

The source paper ends at next-token logits. Fixed eight-token continuation is
a held extension target, not source-paper qualification. Proposed component
references expose reusable contracts for shared lowering to implement later;
they do not establish support, numerical authority, or a security claim.

FP32 client-local arithmetic is a TARGET EXTENSION, not the source paper's
numeric identity. Its relation to source precision (including BF16) and exact
field encoding remains unresolved. Only designated public-field matvecs are
delegated. The client owns normalization, rotary, attention, activation,
multiply, add, bias, KV, reconstructed logits, and selection; head matvec
delegation does not imply downloading the full head weights to the client.
"""

from __future__ import annotations


def benchmark_experiment(*, model=None, budget=None):
    """Describe the public benchmark destination without bypassing admission."""
    from pllm import (
        ConfigurationError,
        Deployment,
        ExecutionBudget,
        Experiment,
        Model,
        Pipeline,
    )
    from pllm.correlation import RaaMask
    from pllm.kernels import Cpu
    from pllm.nonlinear import F32Activation
    from pllm.numeric import ExactSourceFieldBoundaries
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
    from pllm.protocols import PublicFieldMatvec, AuthenticatedConfidentialRoleTransport
    from pllm.roles import ClientMaskSingleServerRoles
    from pllm.sources import ModelSource, AuthenticatedSourceInputProducer, PinnedSemanticSourceBindings
    from pllm.state import ClientLocalKv, TentativeStateCommit
    from pllm.verification import (
        TwoLevelCodedBatchVerification,
        ValidatedMatrixCodePublication,
        OriginalDotWindow,
    )

    if __package__:
        from .common import MODEL
    else:
        from common import MODEL

    if model is None:
        model = MODEL
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    if budget is None:
        # Explicit continuation extension target; not an admitted envelope.
        budget = ExecutionBudget(requests=1, max_input_tokens=8, max_new_tokens=8)

    return Experiment(
        "maverick",
        Pipeline(
            model=normalized_model,
            components={
                # Protocol 2 is not MaverickDelegatedLinear's seeded/i8/41
                # adaptation. Never substitute that component for this method.
                "linear": PublicFieldMatvec(),
                "mask": RaaMask(),
                # Two verification barriers must both pass: validated matrix
                # code publication, then two-level coded batch verification of
                # delegated results. Challenges stay secret from the server;
                # original-dot checks retain their own bounded window.
                "verification": TwoLevelCodedBatchVerification(),
                "publication": ValidatedMatrixCodePublication(),
                "original_dot": OriginalDotWindow(),
                # Failed checks must discard tentative outputs and KV updates.
                "commit": TentativeStateCommit(),
                "producer": AuthenticatedSourceInputProducer(),
                # None means UNRESOLVED, never numerical authority. Binding
                # must reject until source field parameters and no-wrap bounds
                # have authenticated authority; do not invent numeric defaults.
                # These shared parameter names declare requirements, not native
                # capability authority or an exact encoding/rounding policy.
                "numeric": ExactSourceFieldBoundaries(
                    parameter_authority=None,
                    client_arithmetic="f32-extension",
                    source_precision="source",
                    source_to_field_rounding=None,
                    field_to_f32_rounding=None,
                ),
                "rmsnorm": F32RmsNorm(
                    epsilon="source", execution_owner="client",
                ),
                "rotary": F32Rotary(
                    configuration="source", execution_owner="client",
                ),
                "attention": F32CausalAttention(
                    heads="source", kv="state", execution_owner="client",
                ),
                "activation": F32Activation(
                    function="source", execution_owner="client",
                ),
                "multiply": F32Multiply(execution_owner="client"),
                "bias": F32AddBias(
                    placement="after-reconstruction", execution_owner="client",
                ),
                "add": F32Add(execution_owner="client"),
                "embedding": PublicF32Lookup(execution_owner="client"),
                # execution_owner covers post-reconstruction logits only; the
                # full-vocabulary matrix operation still delegates via linear.
                "head": FullVocabularyHead(
                    linear="linear", execution_owner="client",
                ),
                # Continue through EOS for the held eight-token extension.
                "selection": GreedySelection(
                    eos="continue", execution_owner="client",
                ),
                "state": ClientLocalKv(),
                "kernels": Cpu(threads=1),
                "topology": ClientMaskSingleServerRoles(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                # Bind source semantics, producer provenance, and current
                # security requirements to pinned inputs. Missing pins, unknown
                # no-wrap bounds, or unresolved authority must fail admission.
                "bindings": PinnedSemanticSourceBindings(),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/maverick-deployment"),
        budget,
    )


def main(argv=None):
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "maverick", argv, task="response", max_input_tokens=8, max_new_tokens=8
    )

    # Parse --help before importing the SDK or constructing a production plan.
    from pllm import benchmark

    benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


if __name__ == "__main__":
    main()
