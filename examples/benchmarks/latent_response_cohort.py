"""Pinned W8A8 Qwen2.5 prepared control for 8-/32-token response-cost research.

Run through ``pllm benchmark run --experiment
examples/benchmarks/latent_response_cohort.py:prepared --trust-python``.
This is the existing prepared control, not a protected latent-loop placement.
"""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow

prepared = Experiment(
    name="qwen25-prepared-latent-loop-control",
    pipeline=MaskedLinearCpu(
        Model.hf(
            "Qwen/Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
        ),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
    ),
    deployment=Deployment.local(root="local://latent-loop-cohort"),
    budget=ExecutionBudget(requests=3, max_input_tokens=128, max_new_tokens=32),
)
