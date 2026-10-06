"""Slalom-derived trusted-client Freivalds adaptation, without TEE attestation."""
from benchmarks.research.slalom_baseline import experiment as baseline
from pllm.profiles import VerifiedMaskedLinearCpu
from pllm.verification import FreivaldsVerify


def experiment():
    control = baseline()
    return control.with_params(name="slalom-freivalds-adaptation", pipeline=VerifiedMaskedLinearCpu(
        control.pipeline.model, kernels=control.pipeline.kernels,
        quantization=control.pipeline.quantization, inventory=control.pipeline.inventory,
        verification=FreivaldsVerify()))
