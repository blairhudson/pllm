"""Replay the paged Qwen3-4B client-memory candidate."""
from benchmarks.research.qwen3_memory_frontier import paged


def experiment():
    return paged()
