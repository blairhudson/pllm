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
