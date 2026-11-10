"""Gupta–Katz–Miers: legacy compositions and an unmeasured benchmark target.

The canonical script declares planned reusable bindings, not implemented or
qualified execution. SDK admission may honestly refuse the target composition.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline, benchmark
from pllm.configuration import ConfigurationError
from pllm.kernels import Cpu
from pllm.nonlinear import F32Activation
from pllm.numeric import Q11I16Ring64Boundary
from pllm.operators import (
    F32RmsNorm,
    F32Rotary,
    F32CausalAttention,
    F32Multiply,
    F32AddBias,
    F32Add,
    FullVocabularyHead,
    GreedySelection,
)
from pllm.protocols import TwoShareInput, VerifiedPublicF32Lookup, ClientF32DecoderRegions
from pllm.protocols import OpenWeightVerifiableOutsourcing
from pllm.roles import TwoOnlineDelegationRoles
from pllm.state import ClientLocalKv
from pllm.sources import ModelSource
from pllm.verification import PublishedLinearVerify

if __package__:
    from examples.papers.common import BUDGET, MODEL, benchmark_arguments
else:
    from common import BUDGET, MODEL, benchmark_arguments


def _factory_model(value: ModelSource) -> Model:
    """Preserve the former typed profile's structural source override contract."""
    if not isinstance(value, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    return value if isinstance(value, Model) else Model.from_spec(value.to_spec())


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "gkm",
        Pipeline(
            _factory_model(model),
            components={
                "linear": OpenWeightVerifiableOutsourcing(capacity="decoder"),
                "kernels": Cpu(threads=1),
                "verification": PublishedLinearVerify(),
                "topology": TwoOnlineDelegationRoles(),
                "state": ClientLocalKv(),
            },
            profile="research.open_weight_verifiable_cpu",
        ),
        Deployment.local(root=".pllm/papers/gkm"),
        budget,
    )


def seeded(model=MODEL, budget=BUDGET):
    """Same source arithmetic with fresh role-local AES share expansion."""
    return experiment(model, budget).with_params(
        name="gkm-seeded", pipeline__components__linear__sharing="seeded",
    )


def smoke(model=MODEL):
    """Pinned short-request functionality cell; separate from publication workload."""
    return experiment(model).with_params(
        name="gkm-smoke", budget__max_input_tokens=8, budget__max_new_tokens=2,
    )


def seeded_smoke(model=MODEL):
    return smoke(model).with_params(
        name="gkm-seeded-smoke", pipeline__components__linear__sharing="seeded",
    )


def bounded_response():
    """Single-request declaration; source admission remains a separate gate."""
    return experiment().with_params(
        name="gkm-bounded-response", budget__requests=1,
        budget__max_input_tokens=32, budget__max_new_tokens=2,
    )


def composed_bounded_response(model=MODEL, budget=None):
    """Opt-in explicit bindings; declaration only, not qualified run evidence."""
    base = bounded_response()
    return Experiment(
        "gkm-composed-bounded-response",
        Pipeline(_factory_model(model), components={
            **base.pipeline.components,
            "input": TwoShareInput(),
            "lookup": VerifiedPublicF32Lookup(),
            "regions": ClientF32DecoderRegions(),
            "numeric": Q11I16Ring64Boundary(),
        }),
        base.deployment,
        base.budget if budget is None else budget,
    )


def benchmark_experiment(*, model=None, budget=None):
    """Explicit target declaration; planned components confer no run authority."""
    return Experiment(
        "gkm-benchmark",
        Pipeline(
            _factory_model(MODEL if model is None else model),
            components={
                "linear": OpenWeightVerifiableOutsourcing(
                    capacity="decoder", sharing="full", generation="fixed_length_greedy",
                ),
                "input": TwoShareInput(),
                "lookup": VerifiedPublicF32Lookup(),
                # Nearest-even Q11/i16 encoding, checked ring64 products and
                # signed reconstruction, then f32 conversion BEFORE / 2048**2.
                "numeric": Q11I16Ring64Boundary(),
                "rmsnorm": F32RmsNorm(epsilon="source"),
                "rotary": F32Rotary(configuration="source"),
                "attention": F32CausalAttention(heads="source", kv="state"),
                "activation": F32Activation(function="source"),
                "multiply": F32Multiply(),
                # Client bias follows signed reconstruction and f32 rescaling.
                "bias": F32AddBias(placement="after-reconstruction"),
                "add": F32Add(),
                "head": FullVocabularyHead(
                    linear="linear", rows="last-prefill-and-each-decode",
                ),
                "selection": GreedySelection(
                    eos="continue", logits="complete",
                ),
                # Final request verification gates ALL output publication;
                # tentative tokens/logits stay private until acceptance.
                "verification": PublishedLinearVerify(acceptance="request"),
                "topology": TwoOnlineDelegationRoles(),
                "state": ClientLocalKv(),
                "kernels": Cpu(threads=1),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/gkm-deployment"),
        ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2) if budget is None else budget,
    )


def main(argv=None):
    args = benchmark_arguments(
        "gkm", argv, task="response", max_input_tokens=32, max_new_tokens=2,
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
