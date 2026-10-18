"""PUMA source-Q18 composition; native admission retains its bounded envelope."""

from pllm import Deployment, Experiment
from pllm.profiles import PumaThreePartyCpu

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "puma",
        PumaThreePartyCpu(model),
        Deployment.local(root=".pllm/papers/puma"),
        budget,
    )
