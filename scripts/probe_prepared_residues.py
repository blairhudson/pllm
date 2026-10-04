"""Bounded canonical SDK/benchmark/WAN comparison; save each completed cohort."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pllm import Model
from pllm.deployment import PartyAccess, WanConditions
from pllm.metrics import wan_readiness
from pllm.runtime.benchmark_cli import accounted_benchmark_body_totals, run_loopback_benchmark
from pllm.runtime.tiny_llama import create_tiny_llama_checkpoint


def run(source, output, tokens, requests, profile_client=False):
    from examples.benchmarks.prepared_residues import candidate
    prompt = "Explain why privacy-preserving language model inference is useful in one concise paragraph."
    sequence = [prompt, prompt, prompt + " Then give one concrete example."][:requests]
    result = {"schema": "pllm.prepared_residue_benchmark.v1", "complete": False,
        "scope": "canonical benchmark; isolated client caches; co-located role children; hypothetical WAN access capacities",
        "full_wire_bytes": None, "peak_client_memory_bytes": None, "client_body_weights_added": 0, "candidates": []}
    for encoding in ("raw", "row_residues"):
        experiment = candidate(source, encoding=encoding)
        print(f"Running {experiment.name}: {requests} requests, {tokens} outputs/request", flush=True)
        from contextlib import ExitStack
        from unittest.mock import patch
        timings = {}
        def timed(name, function):
            def wrapper(*args, **kwargs):
                start = time.process_time_ns()
                try:
                    return function(*args, **kwargs)
                finally:
                    entry = timings.setdefault(name, {"calls": 0, "cpu_seconds": 0.0})
                    entry["calls"] += 1
                    entry["cpu_seconds"] += (time.process_time_ns() - start) / 1e9
            return wrapper
        with ExitStack() as stack:
            if profile_client:
                from pllm import _native
                from pllm.runtime import residue_codec, model_binding
                for obj, name in ((_native, "prepared_unpack_output"), (residue_codec, "compiled_row_layout"),
                                  (residue_codec, "unpack_row_response"), (residue_codec, "decode_layout"),
                                  (model_binding.CompiledRuntimeModel, "validate")):
                    stack.enter_context(patch.object(obj, name, timed(name, getattr(obj, name))))
            report = run_loopback_benchmark(model=source.source, model_id=experiment.resolve().model,
                tiny=False, prompt=sequence[0], prompt_sequence=sequence, max_output_tokens=tokens,
                warmups=0, repetitions=1, timeout_seconds=120, experiment=experiment,
                temperature=0, capture_output_digest=True, _cohort_salt=b"pllm.public.prepared-residues.v1")
        result["candidates"].append({"experiment": experiment.to_spec(), "report": report,
            "diagnostic_client_functions": timings,
            "bodies": accounted_benchmark_body_totals(report["topology_accounting"]),
            "slow_preparation_wan": wan_readiness(report, WanConditions(parties=(PartyAccess("preparation", upload_mbps=10),)))})
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        gc.collect()
    signatures = [[(r["generation"]["output_text_digest"], r["tokens"]) for r in c["report"]["runs"]] for c in result["candidates"]]
    if signatures[0] != signatures[1]:
        raise ValueError("prepared residue coding changed generated outputs or usage")
    result.update(complete=True, output_and_usage_parity=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps([{ "name": c["experiment"]["name"], "bodies": c["bodies"],
                       "wan": c["report"]["wan_readiness"]["summary"]} for c in result["candidates"]], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--tokens", type=int, choices=(8, 32), default=8)
    parser.add_argument("--requests", type=int, choices=(1, 3), default=3)
    parser.add_argument("--profile-client", action="store_true", help="Instrument bounded client functions; overlapping diagnostic CPU windows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.real:
        from examples.benchmarks.prepared_residues import SOURCE
        run(SOURCE, args.output, args.tokens, args.requests, args.profile_client)
    else:
        with TemporaryDirectory(prefix="pllm-residue-tiny-") as path:
            checkpoint = create_tiny_llama_checkpoint(Path(path) / "model", hidden_size=32,
                intermediate_size=64, num_hidden_layers=2, head_dim=8, seed=709)
            run(Model.path(str(checkpoint), model_id="residue-tiny"), args.output, args.tokens, args.requests, args.profile_client)
