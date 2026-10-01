"""Run a bounded W4A4/W8A8 quality-qualified network comparison.

Run from the repository root:
uv run python examples/benchmarks/quality_locked_network.py --checkpoint /path/to/checkpoint
Lock the checkpoint before interpreting either report as numeric evidence.
"""

from __future__ import annotations

import argparse
import json

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.evidence import benchmark_network_candidates, benchmark_reference_quality
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow
from pllm.search import QualityLockedNetworkSearch


def candidates(checkpoint: str) -> tuple[Experiment, Experiment]:
    model = Model.path(checkpoint, model_id="quality-network-checkpoint")
    def make(bits: int) -> Experiment:
        return Experiment(
            name=f"network-w{bits}a{bits}",
            pipeline=MaskedLinearCpu(
                model, quantization=SymmetricPerRow(
                    weight_bits=bits, activation_bits=bits,
                ),
            ),
            deployment=Deployment.local(root=".pllm/quality-locked-network"),
            budget=ExecutionBudget(requests=1, max_input_tokens=64, max_new_tokens=1),
        )
    return make(4), make(8)


def run(checkpoint: str) -> dict[str, object]:
    experiments = candidates(checkpoint)
    prompt = "A short public test prompt."
    quality = benchmark_reference_quality(experiments, [prompt], top_k=5)
    measured = benchmark_network_candidates(
        experiments, prompt, max_output_tokens=1, repetitions=1,
        inventory_policy="request-sized",
    )
    return QualityLockedNetworkSearch(
        minimum_top1_agreement=0,
        minimum_top_k_recall=0,
        maximum_abs_logit_error=1_000_000,
    ).select(quality, measured)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local pinned Hugging Face checkpoint")
    result = run(parser.parse_args().checkpoint)
    assert result["schema"] == "pllm.quality_locked_network_search.v1"
    print(json.dumps(result, sort_keys=True, indent=2))
