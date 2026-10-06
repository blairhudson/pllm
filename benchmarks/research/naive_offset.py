"""Unoptimized two-online-worker comparator for the matched research cohort."""
from benchmarks.research.common import baseline
from pllm.profiles import TwoOnlineOffsetCpu
from pllm.protocols import TwoOnlineOffsetLinear
from pllm.state import ClientPrefixReuse


def experiment():
    control = baseline("naive-offset-baseline")
    return control.with_params(pipeline=TwoOnlineOffsetCpu(
        control.pipeline.model,
        kernels=control.pipeline.components["kernels"],
        quantization=control.pipeline.components["quantization"],
        linear=TwoOnlineOffsetLinear(),
        # Admit a >64-row prefill. A cold response starts with an empty cache;
        # no prefix, seeded shares, packed outputs or overlapping dispatch.
        cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256),
    ))
