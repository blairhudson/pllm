"""Replay the Qwen3-4B paged SDK control without an indexed tokenizer."""
from benchmarks.research.qwen3_artifacts import control


def experiment():
    return control()
