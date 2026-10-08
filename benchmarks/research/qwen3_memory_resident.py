"""Replay the resident Qwen3-4B client-memory control."""
from benchmarks.research.qwen3_memory_frontier import resident


def experiment():
    return resident()
