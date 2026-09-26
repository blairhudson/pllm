"""Pinned Phi-4-mini reference-quality Experiments; requires local official weights."""

import os

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
_LOCAL = os.environ.get("PLLM_REAL_PHI_PATH")
if not _LOCAL:
    raise ValueError("PLLM_REAL_PHI_PATH must point to the pinned Phi-4-mini checkpoint")
_MODEL = Model.path(_LOCAL, model_id=f"microsoft/Phi-4-mini-instruct@{_REVISION}")


def _candidate(bits: int) -> Experiment:
    return Experiment(
        name=f"phi4-mini-w{bits}a{bits}",
        pipeline=MaskedLinearCpu(
            _MODEL,
            quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits),
        ),
        deployment=Deployment.local(root="local://phi-real-quality"),
        budget=ExecutionBudget(requests=2, max_input_tokens=32, max_new_tokens=1),
    )


w4a4 = _candidate(4)
w8a8 = _candidate(8)
