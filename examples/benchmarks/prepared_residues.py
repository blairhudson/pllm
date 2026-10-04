"""Exact prepared-output coding against the lean prepared conversation stack."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse

SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")


def candidate(source=SOURCE, *, encoding="raw"):
    return Experiment(f"prepared-{encoding}", MaskedLinearCpu(source,
        kernels=Cpu(threads=4), linear=MaskedLinear(output_encoding=encoding),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        inventory=PreparedInventory("request-sized", rows=1, refill="on-demand"),
        cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=128, generated_prefixes=True),
        delivery=ClientBundleTransport("artifacts", compression="zlib")),
        Deployment.local(root="local://prepared-residues"),
        ExecutionBudget(requests=3, max_input_tokens=128, max_new_tokens=32))


control = candidate()
compact = candidate(encoding="row_residues")
