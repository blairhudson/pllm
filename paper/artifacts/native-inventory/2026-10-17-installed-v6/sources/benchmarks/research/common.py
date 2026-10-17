"""Shared cohort contract. Artifact creation is an explicit offline step."""
from pathlib import Path
import os

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def baseline(name):
    return Experiment(name, MaskedLinearCpu(
        Model.hf(MODEL, revision=REVISION), kernels=Cpu(threads=4),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8,
                                    causal_reduction="prefix_f32")),
        Deployment.local(root="local://research-scorecard"),
        ExecutionBudget(requests=1, max_input_tokens=256, max_new_tokens=8))


def artifact_directory():
    return Path(os.environ["PLLM_RESEARCH_ARTIFACTS"]).resolve()
