"""Prepared body with private lookup and the complete tied head at two workers.

Run without arguments for a generated checkpoint. Local role children share one
operator; they do not establish the two non-collusion assumptions of this graph.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pllm
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.roles import PreparedBoundaryRoles, PrivateLookupAndFullHead
from pllm.tokenization import IndexedTokenizer


def build_experiment(model=None, *, tokenizer=None, private_boundary=True,
                     root=".pllm/private-boundary", max_input_tokens=256):
    """Matched cold-client choices; only boundary placement/topology vary."""
    return pllm.Experiment(
        "private-boundary" if private_boundary else "local-boundary-control",
        MaskedLinearCpu(
            model or pllm.Model.tiny(),
            kernels=Cpu(threads=1),
            quantization=SymmetricPerRow(weight_bits=8, activation_bits=8,
                                        causal_reduction="prefix_f32"),
            linear=MaskedLinear(request_encoding="stage_packed", output_encoding="row_residues",
                                prefill_pruning="terminal", prefill_chunk_rows=16),
            preparation=ModelAwareCorrections(storage="paged"),
            inventory=PreparedInventory(refill="on-demand", allocation="demand",
                                        stage_window=4, stage_window_bytes=64 << 20),
            delivery=ClientBundleTransport("artifacts", compression="zlib", storage="paged",
                                           batch_objects=64),
            tokenizer=tokenizer,
            boundary=PrivateLookupAndFullHead() if private_boundary else None,
            topology=PreparedBoundaryRoles() if private_boundary else None,
        ),
        pllm.Deployment.local(root=str(root)),
        pllm.ExecutionBudget(requests=1, max_input_tokens=max_input_tokens, max_new_tokens=8),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path(".pllm/private-boundary"))
    parser.add_argument("--write-experiment", type=Path)
    parser.add_argument("--local-boundary-control", action="store_true")
    parser.add_argument("--prompt", default="Explain why the sky is blue in one sentence.")
    args = parser.parse_args(argv)
    model, tokenizer = pllm.Model.tiny(), None
    if args.model_dir is not None:
        source = args.model_dir.expanduser().resolve()
        model = pllm.Model.path(str(source))
        tokenizer = IndexedTokenizer.compile(source / "tokenizer.json", args.root / "tokenizer")
    experiment = build_experiment(model, tokenizer=tokenizer,
        private_boundary=not args.local_boundary_control, root=args.root)
    experiment.resolve()
    if args.write_experiment is not None:
        args.write_experiment.parent.mkdir(parents=True, exist_ok=True)
        args.write_experiment.write_bytes(experiment.canonical_bytes())
        return
    with pllm.serve_local(experiment) as roles, roles.client(
            bundle_cache_dir=args.root / "client-artifacts", background_inventory_refill=False) as client:
        with client.tokenizer_scope():
            response = client.responses.create(input=args.prompt, max_output_tokens=8, temperature=0)
        print(response.output_text)
        print(json.dumps(client.privacy_audit.to_dict(), indent=2))


if __name__ == "__main__":
    main()
