"""Matched SDK composition search. Every choice preserves the pinned W8A8 body."""
import json

from benchmarks.research.common import artifact_directory, baseline
from pllm import Experiment
from pllm.kernels import AppleMetal, Cpu
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import ClientOnlyCpu, ClientOnlyMetal, TwoOnlineOffsetCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear, TwoOnlineOffsetLinear
from pllm.roles import ClientLinearRoles, ClientPrefixLayers
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule
from pllm.tokenization import IndexedTokenizer
from pllm.search import SearchSpace


def build(name, *, kernel="cpu", threads=4, topology="prepared", window=1,
          chunk=0, delivery="none", compression="none", paged=False,
          preparation_paged=False, indexed=False, capsule=False, roles=(),
          prefix_layers=0, encoding="stage_packed", pruning="terminal"):
    control = baseline(name)
    kernels = AppleMetal(min_rows=8) if kernel == "metal" else Cpu(threads=threads)
    quantization = control.pipeline.components["quantization"]
    model = control.pipeline.model
    transport = ClientBundleTransport(delivery, compression=compression,
        batch_objects=64 if delivery == "artifacts" else 1,
        storage="paged" if paged else "memory")
    if topology == "client":
        profile = ClientOnlyMetal if kernel == "metal" else ClientOnlyCpu
        pipeline = profile(model, kernels=kernels, quantization=quantization)
    elif topology == "offset":
        pipeline = TwoOnlineOffsetCpu(model, kernels=kernels, quantization=quantization,
            linear=TwoOnlineOffsetLinear(input_encoding="seeded",
                output_encoding="row_residues", dispatch="seed_first"), delivery=transport,
            # The 150-token cohort needs the admitted >64-row continuation contract.
            # Every cold trial starts with an empty client cache.
            cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256))
    elif topology == "prepared":
        changes = dict(kernels=kernels, delivery=transport,
            preparation=ModelAwareCorrections(storage="paged" if preparation_paged else "resident"),
            inventory=PreparedInventory(refill="on-demand", allocation="demand", stage_window=window),
            linear=MaskedLinear(request_encoding=encoding, output_encoding="row_residues",
                prefill_pruning=pruning, prefill_chunk_rows=chunk))
        if roles:
            changes["placement"] = ClientLinearRoles(roles, prefix_layers=prefix_layers)
        elif prefix_layers:
            changes["placement"] = ClientPrefixLayers(prefix_layers)
        if indexed:
            descriptor = Experiment.from_spec(json.loads((artifact_directory() / "indexed.json").read_text()))
            changes["tokenizer"] = IndexedTokenizer(**descriptor.pipeline.components["tokenizer"].params)
        if capsule:
            published = Experiment.from_spec(json.loads((artifact_directory() / "capsule.json").read_text()))
            changes["cache"] = ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256)
            changes["public_prefix"] = PublicPrefixCapsule(**published.pipeline.components["public_prefix"].params)
        pipeline = control.pipeline.with_params(**changes)
    else:
        raise ValueError("unknown research topology")
    return control.with_params(pipeline=pipeline)


def search_space(topology="prepared", *, public_prefix=False):
    """Public choices only; the SDK owns proposal, admission and measurement."""
    if public_prefix and topology != "prepared":
        raise ValueError("this public-prefix cohort uses the prepared graph")
    base = baseline(f"qwen25-{topology}-search")
    if topology != "prepared":
        base = build(base.name, topology=topology)
    if public_prefix:
        published = Experiment.from_spec(json.loads((artifact_directory() / "capsule.json").read_text()))
        base = base.with_params(pipeline=base.pipeline.with_params(
            cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=256),
            public_prefix=PublicPrefixCapsule(**published.pipeline.components["public_prefix"].params)))
    axes = {"pipeline__kernels": [base.pipeline.components["kernels"], Cpu(threads=1), Cpu(threads=8), AppleMetal(min_rows=8)]}
    if topology == "client":
        return SearchSpace(base, axes)
    axes["pipeline__delivery"] = [base.pipeline.components.get("delivery"),
        ClientBundleTransport("artifacts", batch_objects=64),
        ClientBundleTransport("artifacts", batch_objects=64, compression="zlib"),
        ClientBundleTransport("artifacts", batch_objects=64, compression="zlib", storage="paged")]
    if topology == "offset":
        axes["pipeline__linear"] = [base.pipeline.components["linear"],
            TwoOnlineOffsetLinear(input_encoding="seeded", output_encoding="row_residues"),
            TwoOnlineOffsetLinear()]
        return SearchSpace(base, axes)
    descriptor = Experiment.from_spec(json.loads((artifact_directory() / "indexed.json").read_text()))
    axes.update({
        "pipeline__linear": [MaskedLinear(), MaskedLinear(output_encoding="row_residues"),
            MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues", prefill_pruning="terminal"),
            MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues", prefill_pruning="terminal", prefill_chunk_rows=4)],
        "pipeline__inventory": [None,
            PreparedInventory("request-sized", rows=1, refill="on-demand"),
            PreparedInventory(allocation="demand", refill="on-demand"),
            PreparedInventory(allocation="demand", refill="on-demand", stage_window=4)],
        "pipeline__preparation": [ModelAwareCorrections(), ModelAwareCorrections(storage="paged")],
        "pipeline__placement": [None, ClientLinearRoles(("qkv_projection", "attention_output")),
            ClientLinearRoles(("mlp_gate_up", "mlp_down")), ClientPrefixLayers(1),
            ClientLinearRoles(("qkv_projection", "attention_output"), prefix_layers=1)],
        "pipeline__tokenizer": [None, IndexedTokenizer(**descriptor.pipeline.components["tokenizer"].params)],
    })
    return SearchSpace(base, axes)
