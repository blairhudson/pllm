"""Slalom-derived 40-bit trusted-client verifier; no TEE reproduction."""
from benchmarks.research.paper_baseline import experiment as control
from pllm.profiles import VerifiedMaskedLinearCpu


def experiment():
    baseline = control()
    return baseline.with_params(
        name="paper-qwen-slalom",
        pipeline=VerifiedMaskedLinearCpu(
            baseline.pipeline.model,
            kernels=baseline.pipeline.components["kernels"],
            quantization=baseline.pipeline.components["quantization"],
            inventory=baseline.pipeline.components["inventory"],
        ),
    )
