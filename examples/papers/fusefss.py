"""FuseFSS dense-RoPE track; full-checkpoint admission remains a separate gate."""

from pllm import Deployment, Experiment
from pllm.profiles import FuseFssSourceCpu

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "fusefss",
        FuseFssSourceCpu(model, numeric_profile="fusefss.dense-rope-ring50-q12.v1"),
        Deployment.local(root=".pllm/papers/fusefss"),
        budget,
    )
