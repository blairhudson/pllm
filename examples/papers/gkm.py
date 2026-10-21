"""Gupta–Katz–Miers: fixed-length full/seeded CPU compositions."""

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


def seeded(model=MODEL, budget=BUDGET):
    """Same source arithmetic with fresh role-local AES share expansion."""
    return experiment(model, budget).with_params(
        name="gkm-seeded", pipeline__linear__sharing="seeded",
    )


def smoke(model=MODEL):
    """Pinned short-request functionality cell; separate from publication workload."""
    return experiment(model).with_params(
        name="gkm-smoke", budget__max_input_tokens=8, budget__max_new_tokens=2,
    )


def seeded_smoke(model=MODEL):
    return smoke(model).with_params(name="gkm-seeded-smoke", pipeline__linear__sharing="seeded")
