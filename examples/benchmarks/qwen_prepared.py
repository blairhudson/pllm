"""SDK-defined Experiment pipelines for matched local benchmarking."""

from pllm import Deployment, ExecutionBudget, Experiment, MaskedLinearCpu, Model, Pipeline
from pllm.kernels import AppleMetal, Cpu
from pllm.profiles import ClientOnlyCpu, ClientOnlyMetal, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import PreparedProviderRoles
from pllm.verification import FreivaldsVerify


def prepared_cpu(model: str | Model, *, threads: int) -> Pipeline:
    """Build the reusable pipeline for any supported public-weight model."""
    return MaskedLinearCpu(
        Model(model) if isinstance(model, str) else model,
        kernels=Cpu(threads=threads),
    )


def _prepared_experiment(name: str, model: str | Model, threads: int) -> Experiment:
    return Experiment(
        name=name,
        pipeline=prepared_cpu(model, threads=threads),
        deployment=Deployment.local(root=f".pllm/benchmarks/{name}"),
        budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
    )


MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CHECKPOINT = Model.hf(MODEL, revision="7ae557604adf67be50417f59c2c2f167def9a775")

cpu_1 = _prepared_experiment("qwen-prepared-cpu-1", CHECKPOINT, threads=1)
cpu_4 = _prepared_experiment("qwen-prepared-cpu-4", CHECKPOINT, threads=4)

prepared_metal_8 = Experiment(
    name="qwen-prepared-metal-8",
    pipeline=MaskedLinearCpu(CHECKPOINT, kernels=AppleMetal(min_rows=8)),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-prepared-metal-8"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

verified_cpu_1 = Experiment(
    name="qwen-verified-cpu-1",
    pipeline=VerifiedMaskedLinearCpu(
        CHECKPOINT,
        kernels=Cpu(threads=1),
        verification=FreivaldsVerify(target_failure_bits=40),
        topology=PreparedProviderRoles(),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-verified-cpu-1"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

verified_metal_8 = Experiment(
    name="qwen-verified-metal-8",
    pipeline=VerifiedMaskedLinearCpu(
        CHECKPOINT,
        kernels=AppleMetal(min_rows=8),
        verification=FreivaldsVerify(target_failure_bits=40),
        topology=PreparedProviderRoles(),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-verified-metal-8"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

offset_cpu_1 = Experiment(
    name="qwen-offset-cpu-1",
    pipeline=TwoOnlineOffsetCpu(
        CHECKPOINT, kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-offset-cpu-1"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

offset_metal_8 = Experiment(
    name="qwen-offset-metal-8",
    pipeline=TwoOnlineOffsetCpu(
        CHECKPOINT, kernels=AppleMetal(min_rows=8),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-offset-metal-8"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

client_cpu_1 = Experiment(
    name="qwen-client-cpu-1",
    pipeline=ClientOnlyCpu(
        CHECKPOINT, kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-client-cpu-1"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)

client_metal_8 = Experiment(
    name="qwen-client-metal-8",
    pipeline=ClientOnlyMetal(
        CHECKPOINT,
        kernels=AppleMetal(min_rows=8),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    ),
    deployment=Deployment.local(root=".pllm/benchmarks/qwen-client-metal-8"),
    budget=ExecutionBudget(requests=4, max_input_tokens=128, max_new_tokens=16),
)
