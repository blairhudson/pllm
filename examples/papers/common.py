"""Shared source and workload bounds for the ten-paper SDK study."""

from pllm import ExecutionBudget, Model


MODEL = Model.hf(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_files_only=True,
)
BUDGET = ExecutionBudget(requests=32, max_input_tokens=64, max_new_tokens=8)
