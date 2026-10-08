"""Matched Qwen3-4B client-weight storage; all other SDK choices are shared."""
from benchmarks.research.qwen3_artifacts import indexed


def paged():
    return indexed().with_params(name="qwen3-memory-paged")


def resident():
    return paged().with_params(name="qwen3-memory-resident", pipeline__delivery__storage="memory")
