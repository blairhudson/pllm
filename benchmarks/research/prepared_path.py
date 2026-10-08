"""Prepared implementation ablation with response-sized public workload bounds."""
from benchmarks.research.qwen3_memory_frontier import paged


def qwen3():
    return paged().with_params(name="qwen3-prepared-path", budget__requests=2,
                               budget__max_input_tokens=16)
