"""Comparable W8A8 prepared baseline and bounded client prefix-reuse Experiments.

Use one shared warmup prompt and a new prompt with a long common prefix.
The cache stores client-side KV, not one-use inventory material.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse

_SOURCE = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
)
_PRECISION = SymmetricPerRow(weight_bits=8, activation_bits=8)
_BUDGET = ExecutionBudget(requests=2, max_input_tokens=256, max_new_tokens=1)

plain = Experiment(
    name="prepared-w8a8",
    pipeline=MaskedLinearCpu(_SOURCE, quantization=_PRECISION),
    deployment=Deployment.local(root="local://prefix-comparison"),
    budget=_BUDGET,
)

prefix = Experiment(
    name="prepared-client-prefix-reuse",
    pipeline=MaskedLinearCpu(
        _SOURCE, quantization=_PRECISION,
        cache=ClientPrefixReuse(max_bytes=128 << 20, fixed_input_tokens=256),
    ),
    deployment=Deployment.local(root="local://prefix-comparison"),
    budget=_BUDGET,
)

assert plain.pipeline.digest() != prefix.pipeline.digest()
