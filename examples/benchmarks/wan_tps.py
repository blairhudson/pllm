"""Independent WAN-TPS ablations through ordinary benchmark run Experiments."""
from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse

SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")


def candidate(name, source=SOURCE, *, batch=1, window=1, offset=False, dispatch="sequential", chunk=0):
    common = dict(kernels=Cpu(threads=4),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=128, generated_prefixes=True),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=batch))
    pipeline = (TwoOnlineOffsetCpu(source, **common, linear=TwoOnlineOffsetLinear(
        input_encoding="seeded", output_encoding="row_residues", dispatch=dispatch)) if offset else
        MaskedLinearCpu(source, **common, linear=MaskedLinear(output_encoding="row_residues", prefill_chunk_rows=chunk),
            inventory=PreparedInventory("request-sized", rows=1, refill="on-demand", stage_window=window)))
    return Experiment(name, pipeline, Deployment.local(root="local://wan-tps"),
        ExecutionBudget(requests=2, max_input_tokens=128, max_new_tokens=32))


prepared_control = candidate("prepared-control")
coalesced_artifacts = candidate("coalesced-artifacts", batch=64)
windowed_preparation = candidate("windowed-preparation", window=4)
prepared_combined = candidate("prepared-combined", batch=64, window=4)
offset_control = candidate("offset-control", offset=True)
seed_first = candidate("seed-first", offset=True, dispatch="seed_first")
duplex_prefill = candidate("duplex-prefill", chunk=4)
prepared_all = candidate("prepared-all", batch=64, window=4, chunk=4)
