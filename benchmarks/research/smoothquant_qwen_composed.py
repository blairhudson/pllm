"""Public channel equalization composed with exact codecs and paged delivery."""
from benchmarks.research.smoothquant_prepared import experiment as composed
from benchmarks.research.smoothquant_qwen import experiment as equalized


def experiment():
    return composed().with_params(
        name="paper-qwen-smoothquant-composed",
        pipeline__quantization=equalized().pipeline.components["quantization"],
    )
