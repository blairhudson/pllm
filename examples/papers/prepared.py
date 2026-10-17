"""PLLM prepared W8A8 control on the same source and workload bounds."""

from pllm import Deployment, Experiment
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow

from examples.papers.common import BUDGET, MODEL


def experiment(model=MODEL, budget=BUDGET):
    return Experiment(
        "prepared-w8a8",
        MaskedLinearCpu(
            model, kernels=Cpu(threads=1),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        ),
        Deployment.local(root=".pllm/papers/prepared"),
        budget,
    )
