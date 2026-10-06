"""Bounded prepared control for the Qwen paper-method study."""
from benchmarks.research.common import baseline
from pllm import ExecutionBudget
from pllm.preparation import PreparedInventory


def experiment():
    control = baseline("paper-qwen-prepared")
    return control.with_params(
        budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=8),
        pipeline__inventory=PreparedInventory(refill="on-demand"),
    )
