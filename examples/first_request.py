from __future__ import annotations

import json

from pllm import Experiment, MaskedLinearCpu, Model
from pllm.deployment import Deployment
from pllm.kernels import Cpu
from pllm.runtime import ExecutionBudget

experiment = Experiment(
    name="qwen-local",
    pipeline=MaskedLinearCpu(
        Model("Qwen/Qwen2.5-0.5B-Instruct"),
        kernels=Cpu(threads=4),
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
