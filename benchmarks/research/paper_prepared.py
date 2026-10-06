"""Exact codec/delivery composition, with provider-owned body stages."""
from benchmarks.research.paper_baseline import experiment as control
from pllm.preparation import PreparedInventory
from pllm.protocols import ClientBundleTransport, MaskedLinear


def experiment():
    return control().with_params(
        name="paper-qwen-prepared-composed",
        pipeline__inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        pipeline__linear=MaskedLinear(
            output_encoding="row_residues", request_encoding="stage_packed",
            prefill_pruning="terminal",
        ),
        pipeline__delivery=ClientBundleTransport(
            "artifacts", compression="zlib", storage="paged", batch_objects=64,
        ),
    )
