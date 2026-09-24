"""Explicit W4A4/W8A8 reference-quality candidates for pinned Qwen2.5-0.5B."""

import os

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow


_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
_MODEL_ID = f"Qwen/Qwen2.5-0.5B-Instruct@{_REVISION}"
_LOCAL = os.environ.get("PLLM_REAL_QWEN_PATH")
_MODEL = (
    Model.path(_LOCAL, model_id=_MODEL_ID)
    if _LOCAL
    else Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision=_REVISION, model_id=_MODEL_ID)
)


def _candidate(name: str, bits: int) -> Experiment:
    return Experiment(
        name=name,
        pipeline=MaskedLinearCpu(
            _MODEL,
            quantization=SymmetricPerRow(weight_bits=bits, activation_bits=bits),
        ),
        deployment=Deployment.local(root="local://qwen2-reference-quality"),
        budget=ExecutionBudget(requests=2, max_input_tokens=16, max_new_tokens=2),
    )


w4a4 = _candidate("qwen2-w4a4", 4)
w8a8 = _candidate("qwen2-w8a8", 8)
