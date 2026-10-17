"""Gupta–Katz–Miers: installed fixed-length, full-share CPU composition."""

from pllm import Deployment, Experiment
from pllm.kernels import Cpu
from pllm.profiles import OpenWeightVerifiableCpu
from pllm.protocols import OpenWeightVerifiableOutsourcing

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "gkm",
        OpenWeightVerifiableCpu(
            model, kernels=Cpu(threads=1),
            linear=OpenWeightVerifiableOutsourcing(capacity="decoder"),
        ),
        Deployment.local(root=".pllm/papers/gkm"),
        budget,
    )
