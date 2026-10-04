"""Small native CPU/Metal control for benchmark memory admission and fallback.

uv run python -m pllm benchmark run \
  --experiment examples/benchmarks/memory_safety.py:cpu \
  --experiment examples/benchmarks/memory_safety.py:metal --trust-python \
  --backend auto --prompt Hi --max-output-tokens 2 --warmups 0 --repetitions 1 \
  --temperature 0 --capture-output-digest

Add --preflight-only to inspect without loading any model tensors. Auto keeps
the selected Pipeline and uses native processes for Metal. This generated tiny
checkpoint checks execution and cleanup, not representative model quality.

qwen_cpu/qwen_metal select the pinned 0.5B checkpoint for an admitted native
functionality cohort. Use --prompt Hi --max-output-tokens 2 and --backend native. Admission
still runs for each candidate; these are not WAN-emulated measurements.
"""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import AppleMetal
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu

_model = Model.tiny(model_id="pllm-memory-tiny")
_budget = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
_deployment = Deployment.local(root="local://memory-safety")
_pipeline = MaskedLinearCpu(_model, inventory=PreparedInventory("request-sized", rows=1, refill="on-demand"))
cpu = Experiment("tiny-memory-cpu", _pipeline, _deployment, _budget)
metal = Experiment("tiny-memory-metal", MaskedLinearCpu(
    _model, kernels=AppleMetal(min_rows=2),
    inventory=PreparedInventory("request-sized", rows=1, refill="on-demand")), _deployment, _budget)

_qwen = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
_qwen_budget = ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=2)
qwen_cpu = Experiment("qwen-memory-cpu", MaskedLinearCpu(
    _qwen, inventory=PreparedInventory("request-sized", rows=1, refill="on-demand")), _deployment, _qwen_budget)
qwen_metal = Experiment("qwen-memory-metal", MaskedLinearCpu(
    _qwen, kernels=AppleMetal(min_rows=16),
    inventory=PreparedInventory("request-sized", rows=1, refill="on-demand")), _deployment, _qwen_budget)
