"""Canonical generated-prefix reuse against the existing lean prepared stack."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse

SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")


def candidate(source=SOURCE, *, generated=False, bound=256):
    return Experiment("generated-prefix" if generated else "completed-prefill-control",
        MaskedLinearCpu(source, kernels=Cpu(threads=4),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
            cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=bound, generated_prefixes=generated),
            inventory=PreparedInventory("request-sized", rows=1, refill="on-demand"),
            delivery=ClientBundleTransport("artifacts", compression="zlib")),
        Deployment.local(root="local://generated-prefix"),
        ExecutionBudget(requests=3, max_input_tokens=bound, max_new_tokens=32))


control = candidate()
generated = candidate(generated=True)
