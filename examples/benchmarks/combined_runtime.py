"""Compatible current runtime optimisations, exposed to ordinary benchmark run."""

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import AppleMetal, Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu, TwoOnlineOffsetCpu, VerifiedMaskedLinearCpu
from pllm.protocols import ClientBundleTransport, TwoOnlineOffsetLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import ClientLinearRoles
from pllm.state import ClientPrefixReuse

SOURCE = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
PROMPT = "Explain why neither server can see the prompt."
CONTEXTS = (PROMPT, PROMPT, PROMPT + " Then explain the preparation role.")
NUMERIC = SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32")


def prepared(name, *, optimised=False, attention=False, metal=False, encoding="artifacts",
             object_compression="none", prefix_layers=0, verified=False):
    pipeline = (VerifiedMaskedLinearCpu if verified else MaskedLinearCpu)(
        SOURCE,
        kernels=AppleMetal(min_rows=32) if metal else Cpu(threads=4),
        quantization=NUMERIC,
        inventory=(
            PreparedInventory("request-sized", rows=1, refill="on-demand")
            if optimised
            else PreparedInventory("prewarm", rows=64, refill="idle")
        ),
        delivery=ClientBundleTransport(encoding if optimised else "none", compression=object_compression),
        cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=96) if optimised else None,
        placement=ClientLinearRoles(["qkv_projection", "attention_output"], prefix_layers=prefix_layers) if attention else None,
    )
    return Experiment(
        name,
        pipeline,
        Deployment.local(root="local://combined-runtime"),
        ExecutionBudget(requests=3, max_input_tokens=96, max_new_tokens=32),
    )


baseline = prepared("prepared-default")
lean = prepared("prepared-reuse-on-demand-artifacts", optimised=True)
combined = prepared("prepared-attention-reuse-on-demand-artifacts", optimised=True, attention=True)
metal = prepared("prepared-reuse-on-demand-artifacts-metal", optimised=True, metal=True)
compressed = prepared(
    "prepared-attention-reuse-on-demand-zlib", optimised=True, attention=True, encoding="zlib"
)
lean_compressed = prepared("prepared-reuse-on-demand-zlib", optimised=True, encoding="zlib")
compressed_metal = prepared(
    "prepared-attention-reuse-on-demand-zlib-metal",
    optimised=True,
    attention=True,
    metal=True,
    encoding="zlib",
)
offset = Experiment(
    "offset-seeded-row-residues",
    TwoOnlineOffsetCpu(
        SOURCE,
        kernels=Cpu(threads=4),
        quantization=NUMERIC,
        linear=TwoOnlineOffsetLinear(input_encoding="seeded", output_encoding="row_residues"),
    ),
    Deployment.local(root="local://combined-runtime"),
    ExecutionBudget(requests=3, max_input_tokens=96, max_new_tokens=32),
)

artifact_stack = prepared("prepared-reuse-compressed-artifacts", optimised=True, object_compression="zlib")
union_stack = prepared("prepared-union-reuse-compressed-artifacts", optimised=True,
    attention=True, prefix_layers=1, object_compression="zlib")
verified_control = prepared("verified-control", verified=True)
verified_stack = prepared("verified-union-reuse-compressed-artifacts", verified=True,
    optimised=True, attention=True, prefix_layers=1, object_compression="zlib")
offset_stack = Experiment("offset-seeded-packed-reuse-compressed-artifacts",
    offset.pipeline.with_params(cache=ClientPrefixReuse(max_bytes=64 << 20, fixed_input_tokens=96),
                               delivery=ClientBundleTransport("artifacts", compression="zlib")),
    offset.deployment, offset.budget)

CANDIDATES = {
    name: globals()[name]
    for name in (
        "baseline",
        "lean",
        "combined",
        "metal",
        "offset",
        "compressed",
        "lean_compressed",
        "compressed_metal",
        "artifact_stack", "union_stack", "offset_stack", "verified_control", "verified_stack",
    )
}
