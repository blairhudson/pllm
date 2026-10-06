"""Bounded control for the memory-admitted Slalom adaptation cohort."""
from benchmarks.research.common import baseline
from pllm import ExecutionBudget
from pllm.preparation import PreparedInventory


def experiment():
    control = baseline("slalom-control")
    return control.with_params(budget=ExecutionBudget(1, 32, 1),
        pipeline=control.pipeline.with_params(inventory=PreparedInventory(refill="on-demand")))
