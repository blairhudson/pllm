"""Side-effect-free PLLM configuration example."""

from pllm import Experiment, MaskedLinearCpu, Model
from pllm.deployment import Deployment
from pllm.kernels import Cpu
from pllm.runtime import ExecutionBudget


def build_experiment() -> Experiment:
    """Build a local public-model experiment."""
    return Experiment(
        name="qwen-local",
        pipeline=MaskedLinearCpu(
            Model("Qwen/Qwen2.5-0.5B-Instruct"),
            kernels=Cpu(threads=4),
        ),
        deployment=Deployment.local(root=".pllm/qwen-local"),
        budget=ExecutionBudget(
            requests=1,
            max_input_tokens=128,
            max_new_tokens=32,
        ),
    )


experiment = build_experiment()
