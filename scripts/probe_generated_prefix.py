"""Three public conversation turns through the ordinary benchmark and role driver."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pllm import Experiment, Model
from pllm.profiles import ClientOnlyCpu
from pllm.runtime.benchmark_cli import accounted_benchmark_body_totals, run_loopback_benchmark
from pllm.runtime.servers import build_roles
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


def run(source, output, *, tokens, tiny=False):
    from examples.benchmarks.generated_prefix import candidate

    bound = 224 if tiny else 256
    base = candidate(source, bound=bound)
    oracle = Experiment("public-conversation-cohort", ClientOnlyCpu(source,
        quantization=base.pipeline.quantization, kernels=base.pipeline.kernels), base.deployment, base.budget)
    history, sequence = [], []
    with build_roles(oracle, engine_threads=4) as roles, roles.client() as client:
        prompts = ("Privacy?", "Why?", "Then?") if tiny else (
            "Explain why neither server can see the prompt.", "Explain the preparation role.", "Summarize the privacy assumptions.")
        for prompt in prompts:
            history.append({"role": "user", "content": prompt})
            sequence.append([dict(row) for row in history])
            response = client.responses.create(model=oracle.resolve().model, input=history,
                temperature=0, max_output_tokens=tokens)
            history.append({"role": "assistant", "content": response.output_text})
    del client, roles
    gc.collect()
    result = {"schema": "pllm.generated_prefix_benchmark.v1", "complete": False,
        "scope": "same three public conversation contexts; isolated client caches; co-located role children",
        "context_construction_cost_excluded": True, "full_wire_bytes": None, "candidates": []}
    for enabled in (False, True):
        experiment = candidate(source, generated=enabled, bound=bound)
        print(f"Running {experiment.name}", flush=True)
        report = run_loopback_benchmark(model=source.source, model_id=experiment.resolve().model,
            tiny=False, prompt=sequence[0], prompt_sequence=sequence, max_output_tokens=tokens,
            warmups=0, repetitions=1, timeout_seconds=120, experiment=experiment,
            temperature=0, capture_output_digest=True, _cohort_salt=b"pllm.public.generated-prefix.v1")
        result["candidates"].append({"experiment": experiment.to_spec(), "report": report,
            "bodies": accounted_benchmark_body_totals(report["topology_accounting"])})
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    signatures = [[(row["generation"]["output_text_digest"], row["tokens"]) for row in candidate["report"]["runs"]]
                  for candidate in result["candidates"]]
    if signatures[0] != signatures[1]:
        raise ValueError("generated-prefix optimization changed outputs or usage")
    result.update(complete=True, output_and_usage_parity=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps([{"name": row["experiment"]["name"], "bodies": row["bodies"]}
                     for row in result["candidates"]], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--tokens", type=int, choices=(8, 32), default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.real:
        from examples.benchmarks.generated_prefix import SOURCE
        run(SOURCE, args.output, tokens=args.tokens)
    else:
        with TemporaryDirectory(prefix="pllm-generated-prefix-") as folder:
            root = create_tiny_llama_checkpoint(Path(folder) / "model", hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, head_dim=8, seed=318)
            run(Model.path(str(root), model_id="generated-prefix-tiny"), args.output, tokens=args.tokens, tiny=True)
