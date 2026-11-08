"""FuseFSS dense-RoPE track; full-checkpoint admission remains a separate gate."""

from pllm import Deployment, ExecutionBudget, Experiment
from pllm.profiles import FuseFssSourceCpu

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "fusefss",
        FuseFssSourceCpu(
            model,
            numeric_profile="fusefss.dense-rope-ring50-q12.v1",
            capacity="decoder",
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
        pipeline__linear__numeric_profile="fusefss.dense-rope-checked-ranges-ring50-q12.v1",
        deployment__root=".pllm/papers/fusefss-checked-ranges",
    )


def qwen_checked_batch_smoke():
    """Matched checked-range workload with explicit bounded material batching."""
    return qwen_checked_smoke().with_params(
        name="fusefss-checked-ranges-batch", pipeline__linear__material_delivery="batch-v1")


def qwen_checked_wavefront_smoke():
    """Same checked source and request with bounded dependency-aware online rounds."""
    return qwen_checked_batch_smoke().with_params(
        name="fusefss-checked-ranges-wavefront", pipeline__linear__online_schedule="wavefront-v1")


def qwen_checked_tensor_smoke():
    """Same checked Q12 request with compact templates and bounded tensor waves."""
    return qwen_checked_batch_smoke().with_params(
        name="fusefss-checked-ranges-tensor", pipeline__linear__online_schedule="tensor-v1")


def qwen_checked_tensor_minimal_smoke():
    """Separate 3+1 workload fitting `user: hi`; does not qualify the 4+1 default."""
    return qwen_checked_tensor_smoke().with_params(budget__max_input_tokens=3)


def qwen_empty_system():
    """Empty-system ChatML candidate; token count and material admission unverified."""
    return qwen_checked_tensor_smoke().with_params(
        name="fusefss-qwen-empty-system", budget__requests=1,
        budget__max_input_tokens=15, budget__max_new_tokens=1,
    )
