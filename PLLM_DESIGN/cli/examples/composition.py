"""Additional composition proposals; no claim of complete model support."""
from pllm import Model, Pipeline
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareGarbling
from pllm.protocols.garbling import ArithmeticGarbling, BooleanHalfGates

single_evaluator = Pipeline.from_profile(
    "research.single_evaluator",
    model=Model("Qwen/Qwen2.5-0.5B-Instruct"),
    components={
        "linear": ArithmeticGarbling(),
        "nonlinear": BooleanHalfGates(),
        "preparation": ModelAwareGarbling(),
        "kernels": Cpu(threads=4),
    },
)

# The profile defines composition-slot categories, complete operation coverage,
# required conversions, numerical semantics, and the minimal-client contract.
# A Boolean nonlinear selector is not an implementation of every stateful op.
# Compilation must fail for missing attention, scaling, embedding, or sampling.
