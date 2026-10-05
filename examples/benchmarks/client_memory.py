"""Resident/paged client storage ablations; identical numeric and role contracts."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse


def candidate(name, source, storage, *, max_input_tokens=64):
    return Experiment(name, MaskedLinearCpu(source, kernels=Cpu(threads=1),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        linear=MaskedLinear(output_encoding="row_residues"),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4),
        cache=ClientPrefixReuse(max_bytes=128 << 20, fixed_input_tokens=max_input_tokens, generated_prefixes=True),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage=storage)),
        Deployment.local(root="local://client-memory"),
        ExecutionBudget(requests=1, max_input_tokens=max_input_tokens, max_new_tokens=8))


source = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
resident = candidate("client-resident", source, "memory")
paged = candidate("client-paged", source, "paged")
source_4b = Model.hf("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c")
qwen3_resident = candidate("qwen3-4b-client-resident", source_4b, "memory")
qwen3_paged = candidate("qwen3-4b-client-paged", source_4b, "paged")
