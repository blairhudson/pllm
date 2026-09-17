"""Side-effect-free PLLM configuration example."""

from pllm import ComponentRef, Experiment, Model, Pipeline
from pllm.deployment import Deployment
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections
from pllm.protocols.masked_linear import MaskedLinear
from pllm.runtime import ExecutionBudget

experiment = Experiment(
    name="qwen-local",
    pipeline=Pipeline.from_profile(
        "baseline.masked_linear_cpu",
        model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
        components={
            "linear": MaskedLinear(),
            "preparation": ModelAwareCorrections(),
            "inference": ComponentRef("pllm/inference"),
            "kernels": Cpu(threads=4),
        },
    ),
    deployment=Deployment.local(root=".pllm/qwen-local"),
    budget=ExecutionBudget(
        requests=1,
        max_input_tokens=128,
        max_new_tokens=32,
    ),
)
