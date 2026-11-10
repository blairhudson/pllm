"""FuseFSS primitive benchmark target; not an admitted full-checkpoint deployment."""

from pllm import Deployment, ExecutionBudget, Experiment, benchmark
from pllm.configuration import ConfigurationError, Model, Pipeline
from pllm.kernels import Cpu
from pllm.nonlinear import (
    FSSSignedFloorRescale, FSSSigned38InputSiLU, FSSSumSquaresTableRMSNorm,
    FSSProductFloorRotary, FSSSourceTableSoftmax,
)
from pllm.numeric import Ring50Q12Edges
from pllm.preparation import TrustedDealerOneUseFSSMaterial
from pllm.protocols import (
    FSSPublicTableLookup, FSSPublicDot, FSSPrivateDot, FSSSecretMultiply,
    FSSAdd, FSSArgmax, FSSAuthenticatedRoleTransport,
)
from pllm.schedulers import FSSOnlineSchedule
from pllm.sources import FSSSourceBinding
from pllm.protocols import FuseFssSourceDecoder
from pllm.roles import TrustedDealerTwoShareRoles
from pllm.sources import ModelSource
from pllm.state import FixedCountPrivatePrefix

if __package__:
    from .common import BUDGET, MODEL, benchmark_arguments
else:
    from common import BUDGET, MODEL, benchmark_arguments


def benchmark_experiment(*, model=None, budget=None):
    """Declare primitive bindings; missing registrations/admission fail closed.

    The offline Qwen source is a target, not evidence of decoder qualification.
    Existing native gates remain authoritative; no diagnostic execution fallback.
    """
    model = MODEL if model is None else model
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    # Explicitly fit the existing 16-position envelope: 15 + 1 - 1 = 15.
    # This does not admit the checkpoint or bypass actual prompt-token fit checks.
    budget = (
        ExecutionBudget(requests=1, max_input_tokens=15, max_new_tokens=1)
        if budget is None else budget
    )
    return Experiment(
        "fusefss",
        Pipeline(
            model=model,
            components={
                "embedding": FSSPublicTableLookup(),
                "linear": FSSPublicDot(),
                # Dot products and secret products have per-edge Q24 products;
                # signed floor rescaling restores Q12, never nearest rounding.
                "rescale": FSSSignedFloorRescale(),
                # Only SiLU's input has the signed-38 domain, not every edge.
                "activation": FSSSigned38InputSiLU(),
                "normalization": FSSSumSquaresTableRMSNorm(),
                "rotary": FSSProductFloorRotary(),
                "private_dot": FSSPrivateDot(),
                "softmax": FSSSourceTableSoftmax(),
                # Qwen's SiLU(gate) * up is a secret-secret product, followed
                # by the same explicit signed Q12 floor-rescale contract.
                "gated_multiply": FSSSecretMultiply(),
                "residual": FSSAdd(),
                # Full public output-head dot, not a shortlist or decoder wrapper.
                "head": FSSPublicDot(capacity="full-vocabulary"),
                "selection": FSSArgmax(),
                # Trusted dealer + two shareholders; Driver consumes each
                # material item once, including failed/abandoned requests.
                "material": TrustedDealerOneUseFSSMaterial(),
                "source": FSSSourceBinding(),
                # Source-derived bounds belong to individual edges/products;
                # no global signed-38 range assumption is introduced.
                "numeric": Ring50Q12Edges(
                    revision="fusefss.dense-rope-ring50-q12.v1",
                    ranges="source-derived",
                ),
                "schedule": FSSOnlineSchedule(),
                "transport": FSSAuthenticatedRoleTransport(),
                "topology": TrustedDealerTwoShareRoles(),
                "state": FixedCountPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root="target/paper-benchmarks/fusefss-deployment"),
        budget,
    )


def main(argv=None):
    args = benchmark_arguments(
        "fusefss", argv, task="response", max_input_tokens=15, max_new_tokens=1
    )
    return benchmark(
        benchmark_experiment(model=args.model, budget=args.budget),
        scenario=args.scenario,
        inputs=args.inputs,
        sampling=args.sampling,
        output=args.output,
    )


def experiment(model=MODEL, budget=BUDGET):
    if not isinstance(model, ModelSource):
        raise ConfigurationError("model must satisfy the ModelSource contract")
    model = model if isinstance(model, Model) else Model.from_spec(model.to_spec())
    # Generic components__ overrides and resolve-time category validation are
    # intentional. The linear component still declares the whole decoder.
    return Experiment(
        "fusefss",
        Pipeline(
            profile="research.fusefss_source_cpu",
            model=model,
            components={
                "linear": FuseFssSourceDecoder(
                    numeric_profile="fusefss.dense-rope-ring50-q12.v1", capacity="decoder"),
                "topology": TrustedDealerTwoShareRoles(),
                "state": FixedCountPrivatePrefix(),
                "kernels": Cpu(),
            },
        ),
        Deployment.local(root=".pllm/papers/fusefss"),
        budget,
    )


def qwen_smoke():
    """Bounded ordinary prompt plus one selection; source/numeric admission applies."""
    return experiment(
        budget=ExecutionBudget(requests=1, max_input_tokens=4, max_new_tokens=1)
    )


def qwen_checked_smoke():
    """Opt-in whole-request range checks; accepted requests retain D13 arithmetic."""
    return qwen_smoke().with_params(
        name="fusefss-checked-ranges",
        pipeline__components__linear__numeric_profile="fusefss.dense-rope-checked-ranges-ring50-q12.v1",
        deployment__root=".pllm/papers/fusefss-checked-ranges",
    )


def qwen_checked_batch_smoke():
    """Matched checked-range workload with explicit bounded material batching."""
    return qwen_checked_smoke().with_params(
        name="fusefss-checked-ranges-batch", pipeline__components__linear__material_delivery="batch-v1")


def qwen_checked_wavefront_smoke():
    """Same checked source and request with bounded dependency-aware online rounds."""
    return qwen_checked_batch_smoke().with_params(
        name="fusefss-checked-ranges-wavefront", pipeline__components__linear__online_schedule="wavefront-v1")


def qwen_checked_tensor_smoke():
    """Same checked Q12 request with compact templates and bounded tensor waves."""
    return qwen_checked_batch_smoke().with_params(
        name="fusefss-checked-ranges-tensor", pipeline__components__linear__online_schedule="tensor-v1")


def qwen_checked_tensor_minimal_smoke():
    """Separate 3+1 workload fitting `user: hi`; does not qualify the 4+1 default."""
    return qwen_checked_tensor_smoke().with_params(budget__max_input_tokens=3)


def qwen_empty_system():
    """Empty-system ChatML candidate; token count and material admission unverified."""
    return qwen_checked_tensor_smoke().with_params(
        name="fusefss-qwen-empty-system", budget__requests=1,
        budget__max_input_tokens=15, budget__max_new_tokens=1,
    )


if __name__ == "__main__":
    main()
