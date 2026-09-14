"""Proposed PLLM API example. Requires the planned composition/runtime API.

Importing this file constructs configuration only. It must not resolve weights,
start parties, prepare material, or open a live session.
"""
from pllm import Experiment, Model, Pipeline, Runtime
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


def main() -> None:
    with Runtime(experiment) as runtime:
        runtime.prepare()
        with runtime.session() as session:
            for delta in session.generate(
                "Explain private inference in two sentences.", stream=True
            ):
                print(delta.text, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
