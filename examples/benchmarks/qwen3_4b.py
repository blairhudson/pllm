"""Offline-only cached Qwen3-4B combinations for ordinary benchmark run."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.state import ClientPrefixReuse

SOURCE = Model.hf("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c",
                  local_files_only=True)
NUMERIC = SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32")
DEPLOYMENT = Deployment.local(root="local://qwen3-4b")
BUDGET = ExecutionBudget(requests=2, max_input_tokens=64, max_new_tokens=8)


def prepared(name, *, optimized=False, placement=False):
    return Experiment(name, MaskedLinearCpu(SOURCE, kernels=Cpu(threads=4), quantization=NUMERIC,
        linear=MaskedLinear(output_encoding="row_residues", prefill_chunk_rows=4) if optimized else MaskedLinear(),
        inventory=(PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=4) if optimized
                   else PreparedInventory("prewarm", rows=64, refill="idle")),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64) if optimized else None,
        cache=ClientPrefixReuse(max_bytes=128 << 20, fixed_input_tokens=64, generated_prefixes=True) if optimized else None,
        placement=ClientLinearRoles(["attention_output"], prefix_layers=1) if placement else None,
    ), DEPLOYMENT, BUDGET)


baseline = prepared("qwen3-4b-prepared-control")
lean = prepared("qwen3-4b-prepared-lean", optimized=True)
placed = prepared("qwen3-4b-prepared-placed", optimized=True, placement=True)
offset = Experiment("qwen3-4b-offset-optimized", TwoOnlineOffsetCpu(SOURCE,
    kernels=Cpu(threads=4), quantization=NUMERIC,
    linear=TwoOnlineOffsetLinear(input_encoding="seeded", output_encoding="row_residues", dispatch="seed_first"),
    delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64),
    cache=ClientPrefixReuse(max_bytes=128 << 20, fixed_input_tokens=64, generated_prefixes=True)),
    DEPLOYMENT, BUDGET)
