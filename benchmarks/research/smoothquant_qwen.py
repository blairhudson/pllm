"""Qwen SmoothQuant-style adaptation with an immutable public calibration profile."""
from benchmarks.research.smoothquant_baseline import experiment as control
from pllm.quantization import PublicPerChannelEqualized


def experiment():
    return control().with_params(
        name="paper-qwen-smoothquant",
        pipeline__quantization=PublicPerChannelEqualized(
            profile_digest="a1024e2b0376a5d8baa57b6c972d18f641a2817a57b204cc1e1c4182dc0f430b",
        ),
    )
