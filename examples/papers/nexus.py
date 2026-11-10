"""Desired NEXUS benchmark: standalone offline BERT sequence classification.

This is a source-level composition target, not an implemented NEXUS runner or
measured paper result. Proposed reusable component identities below still need
provider schemas, source/import owners, semantic bindings, scheduling, transport,
and admission. Normal ComponentRef/Pipeline guards remain in force; declaring
these identities does not register implementations or bypass those guards.

The shared API also needs a backward-compatible public ``pllm.benchmark``
BenchmarkTarget overload accepting ``inputs``, ``sampling`` and ``output`` without
a response Scenario. Existing ``BenchmarkTarget.forward`` is a foothold, not
evidence that this public overload or NEXUS execution is supported. No local
runner, clear oracle, or paper-specific raising shim substitutes for that work.

Desired invocation (after those shared gaps are implemented)::

    python -m examples.papers.nexus --model-path /offline/bert-classifier \
        --forward-inputs /offline/classification-inputs.json

The shared CLI must require both options: Model.path supplies an offline BERT
classification checkpoint, and --forward-inputs supplies a JSON mapping of
forward tensors. Checkpoint architecture, classifier head and output policy must
be admitted from model metadata, not inferred from a fabricated Hub pin. No
autoregressive generation, Qwen workload, or KV-cache contract is assumed.

Pinned semantic sources and numeric-domain certificates must authorize the HE
parameters, approximation intervals, errors and lifecycle schedule before any
execution. None is deliberately unresolved parameter authority, not permission
to choose CKKS primes, scales or author parameters. Source softmax uses eight
squarings; the existing bounded softmax with three reciprocal iterations and
the +/-1/8 bound is not whole NEXUS. Client-secret/server-weight-secret roles,
dual-key contexts, ciphertext birth, bootstrap and authenticated confidential
transport must compose under the shared trust and admission contracts.
"""

from __future__ import annotations

from pllm import (
    ConfigurationError,
    Deployment,
    Experiment,
    ForwardBudget,
    Model,
    Pipeline,
)
from pllm.sources import ModelSource
from pllm.nonlinear import (
    HERepeatedSquaringSoftmax, HEPiecewisePolynomialActivation, HERowNormalization, HESignComposition,
)
from pllm.numeric import SourceNumericDomainCertificates
from pllm.preparation import HEDualKeyContext, HECiphertextBirth
from pllm.protocols import (
    HECoefficientCompression, HECommutableLinear, HEColumnProduct, HELayoutTransform,
    HEAttentionValue, HEFoldSelection, HEBootstrap, HEResidualAdd, HEPrivateEmbedding,
    ClassifierOutputPolicy, AuthenticatedConfidentialRoleTransport,
)
from pllm.roles import ClientSecretServerWeightSecretRoles
from pllm.schedulers import HELevelErrorSchedule
from pllm.sources import PinnedSemanticSourceBindings


def benchmark_experiment(*, model=None, budget=None):
    """Declare proposed reusable primitives; retain all normal SDK admission."""
    if model is None:
        raise ConfigurationError(
            "NEXUS requires an explicit offline BERT classification checkpoint; "
            "supply Model.path(...) or --model-path"
        )
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    normalized_model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    if not normalized_model.local_files_only:
        raise ConfigurationError("NEXUS requires an offline BERT classification checkpoint")
    if budget is None:
        budget = ForwardBudget(requests=1, max_input_tokens=32)
    if not isinstance(budget, ForwardBudget):
        raise ConfigurationError("NEXUS classification requires ForwardBudget")

    return Experiment(
        name="nexus",
        pipeline=Pipeline(
            model=normalized_model,
            components={
                "coefficient_compression": HECoefficientCompression(),
                "linear": HECommutableLinear(),
                "column_product": HEColumnProduct(),
                "layout": HELayoutTransform(),
                "attention_value": HEAttentionValue(),
                "softmax": HERepeatedSquaringSoftmax(
                    source_squarings=8, parameter_authority=None,
                ),
                "activation": HEPiecewisePolynomialActivation(),
                "normalization": HERowNormalization(),
                "sign": HESignComposition(),
                "selection": HEFoldSelection(),
                "bootstrap": HEBootstrap(),
                "residual": HEResidualAdd(),
                "embedding": HEPrivateEmbedding(),
                "key_context": HEDualKeyContext(),
                "ciphertext_birth": HECiphertextBirth(),
                "schedule": HELevelErrorSchedule(),
                "transport": AuthenticatedConfidentialRoleTransport(),
                "source_bindings": PinnedSemanticSourceBindings(),
                "roles": ClientSecretServerWeightSecretRoles(),
                "numeric_certificates": SourceNumericDomainCertificates(
                    parameter_authority=None,
                ),
                "output": ClassifierOutputPolicy(source="model-metadata"),
            },
        ),
        deployment=Deployment.local(root="target/paper-benchmarks/nexus-deployment"),
        budget=budget,
    )


def main(argv=None):
    """Express the desired shared forward benchmark API, pending shared support."""
    if __package__:
        from .common import benchmark_arguments
    else:
        from common import benchmark_arguments

    args = benchmark_arguments(
        "nexus", argv, task="forward", max_input_tokens=32, max_new_tokens=0
    )

    import pllm
    from pllm.metrics import BenchmarkTarget

    target = BenchmarkTarget.forward(
        benchmark_experiment(model=args.model, budget=args.budget)
    )
    return pllm.benchmark(
        target, inputs=args.inputs, sampling=args.sampling, output=args.output
    )


if __name__ == "__main__":
    main()
