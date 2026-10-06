"""Current exact prepared stack, with no assumed public or reused prompt prefix."""
from benchmarks.research.common import artifact_directory, baseline
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.tokenization import IndexedTokenizer


def experiment():
    import json
    root = artifact_directory()
    descriptor = json.loads((root / "indexed.json").read_text())
    from pllm import Experiment
    indexed = IndexedTokenizer(**Experiment.from_spec(descriptor).pipeline.components["tokenizer"].params)
    control = baseline("prepared-exact-stack")
    return control.with_params(pipeline=control.pipeline.with_params(
        preparation=ModelAwareCorrections(storage="paged"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed",
                            prefill_pruning="terminal"),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64,
                                       storage="paged"),
        tokenizer=indexed))
