"""Opt-in checkpoint-bound public equalization quality diagnostic."""

from __future__ import annotations

import os

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import PublicPerChannelEqualized


_PATH = os.environ.get("PLLM_REAL_PHI_PATH")
if not _PATH:
    raise ValueError("PLLM_REAL_PHI_PATH must point to the pinned local Phi-4-mini checkpoint")

_SOURCE = Model.path(
    _PATH,
    model_id="microsoft/Phi-4-mini-instruct@cfbefacb99257ffa30c83adab238a50856ac3083",
)
_PROFILE = "6e2074d87d546ac98e2eb8518ff26495996a34b9f3202bfc2ee9ecc953eea683"

equalized = Experiment(
    name="Pinned Phi-4-mini public equalized W8A8 reference quality",
    pipeline=MaskedLinearCpu(_SOURCE, quantization=PublicPerChannelEqualized(_PROFILE)),
    deployment=Deployment.local(root="local://phi-real-quality"),
    budget=ExecutionBudget(requests=2, max_input_tokens=32, max_new_tokens=1),
)
