"""SDK-defined Experiment pipelines for matched local benchmarking."""

from pllm import Deployment, ExecutionBudget, Experiment, MaskedLinearCpu, Model, Pipeline
from pllm.kernels import Cpu


def prepared_cpu(model: str, *, threads: int) -> Pipeline:
    """Build the reusable pipeline for any supported public-weight model."""
    return MaskedLinearCpu(
        Model(model),
        kernels=Cpu(threads=threads),
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
