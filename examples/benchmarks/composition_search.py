"""Measure component combinations with the same SDK used by grid/random search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pllm import Deployment, ExecutionBudget, Experiment, Model, benchmark_search
from pllm.kernels import Cpu
from pllm.preparation import PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.search import BeamSearch, SearchSpace


def search(*, tiny=False, trials=8):
    model = Model.tiny() if tiny else Model(
        "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
    base = Experiment(
        "composition-search", MaskedLinearCpu(model, kernels=Cpu(threads=1),
            quantization=SymmetricPerRow(causal_reduction="prefix_f32")),
        Deployment.local(root="local://composition-search"), ExecutionBudget(1, 32, 2),
    )
    space = SearchSpace(base, {
        "pipeline__kernels": [Cpu(threads=1), Cpu(threads=2)],
        "pipeline__linear": [MaskedLinear(), MaskedLinear(output_encoding="row_residues")],
        "pipeline__inventory": [None, PreparedInventory("request-sized", rows=1, refill="on-demand")],
    })
    return BeamSearch("composition", space, "request_tps", "max", width=min(2, trials), max_trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark_search(search(tiny=args.tiny, trials=args.trials), args.prompt,
        max_output_tokens=2, progress=lambda candidate, message: print(f"[{candidate.index}] {message}", flush=True)).to_dict()
    configs = args.output.with_suffix("")
    configs.mkdir(parents=True, exist_ok=True)
    for record in report["candidates"]:
        path = configs / f"trial_{record['configuration_digest'][:16]}.py"
        path.write_text(record.pop("python_source"))
        record["configuration_file"] = str(path)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print("Best measured configuration:", report["search"]["best_configuration_digest"])


if __name__ == "__main__":
    main()
