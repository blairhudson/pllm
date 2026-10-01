"""Compare untied output-head placement with the same W8A8 decoder body.

The Phi checkpoint is pinned; this requires its official weights for execution.
The prepared Inference role computes the masked head only in the second option.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.roles import OutputHeadAtInference

_MODEL = Model.hf(
    "microsoft/Phi-4-mini-instruct",
    revision="cfbefacb99257ffa30c83adab238a50856ac3083",
)
_PRECISION = SymmetricPerRow(weight_bits=8, activation_bits=8)
_BUDGET = ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=32)

client_head = Experiment(
    name="phi-client-output-head",
    pipeline=MaskedLinearCpu(_MODEL, quantization=_PRECISION),
    deployment=Deployment.local(root="local://phi-head-comparison"),
    budget=_BUDGET,
)

remote_head = Experiment(
    name="phi-inference-output-head",
    pipeline=MaskedLinearCpu(
        _MODEL, quantization=_PRECISION, boundary=OutputHeadAtInference(),
    ),
    deployment=Deployment.local(root="local://phi-head-comparison"),
    budget=_BUDGET,
)

assert client_head.pipeline.digest() != remote_head.pipeline.digest()
