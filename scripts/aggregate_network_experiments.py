"""Matched ordinary SDK/benchmark ablations for the new exact transport choices."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow


def _experiment(name, *, compact=False, pruning=False, packed=False):
    return Experiment(name, MaskedLinearCpu(
        Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775"),
        kernels=Cpu(threads=4),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        linear=MaskedLinear(output_encoding="row_residues",
            request_encoding="stage_packed" if packed else "compact" if compact else "raw",
            prefill_pruning="terminal" if pruning else "none"),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage="paged")),
        Deployment.local(root="local://aggregate-network"),
        ExecutionBudget(max_input_tokens=64, max_new_tokens=8, requests=1))


def control():
    return _experiment("aggregate-control")


def compact():
    return _experiment("aggregate-compact", compact=True)


def terminal():
    return _experiment("aggregate-terminal", pruning=True)


def combined():
    return _experiment("aggregate-combined", compact=True, pruning=True)


def stage_packed():
    return _experiment("algebra-stage-packed", packed=True, pruning=True)
