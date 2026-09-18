from __future__ import annotations

import json

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
    budget=ExecutionBudget(requests=1, max_input_tokens=128, max_new_tokens=32),
)
resolved = experiment.resolve()

if __name__ == "__main__":
    print(
        json.dumps(
            {
                "configuration_digest": resolved.configuration_digest,
                "experiment": experiment.to_spec(),
            },
            sort_keys=True,
        )
    )
