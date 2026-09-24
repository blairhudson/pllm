"""Explicit W4A4/W8A8 reference-quality candidates for pinned Qwen3-0.6B."""

import os

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
_MODEL_ID = f"Qwen/Qwen3-0.6B@{_REVISION}"
_LOCAL = os.environ.get("PLLM_REAL_QWEN3_PATH")
_MODEL = (
    Model.path(_LOCAL, model_id=_MODEL_ID)
    if _LOCAL
    else Model.hf("Qwen/Qwen3-0.6B", revision=_REVISION, model_id=_MODEL_ID)
)


def _candidate(name: str, bits: int) -> Experiment:
    return Experiment(
        name=name,
        pipeline=MaskedLinearCpu(
            _MODEL,
            quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits),
        ),
        deployment=Deployment.local(root="local://qwen3-reference-quality"),
        budget=ExecutionBudget(requests=2, max_input_tokens=16, max_new_tokens=2),
    )


w4a4 = _candidate("qwen3-w4a4", 4)
w8a8 = _candidate("qwen3-w8a8", 8)
