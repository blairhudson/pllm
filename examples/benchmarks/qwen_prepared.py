"""SDK-defined Experiment pipelines for matched local benchmarking."""

from pllm import Deployment, ExecutionBudget, Experiment, Model, Pipeline
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections
from pllm.protocols import MaskedLinear
from pllm.roles import Inference


def prepared_cpu(model: str, *, threads: int) -> Pipeline:
    """Build the reusable pipeline for any supported public-weight model."""
    return Pipeline.from_profile(
        "baseline.masked_linear_cpu",
        model=Model(model),
        components={
            "preparation": ModelAwareCorrections(),
            "inference": Inference(),
            "linear": MaskedLinear(),
            "kernels": Cpu(threads=threads),
        },
    )


def _prepared_experiment(name: str, model: str, threads: int) -> Experiment:
    return Experiment(
        name=name,
        pipeline=prepared_cpu(model, threads=threads),
        deployment=Deployment.local(root=f".pllm/benchmarks/{name}"),
        budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
    )


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

cpu_1 = _prepared_experiment("qwen-prepared-cpu-1", MODEL, threads=1)
cpu_4 = _prepared_experiment("qwen-prepared-cpu-4", MODEL, threads=4)
