"""Build trusted PUBLIC artifacts and canonical SDK benchmark Experiments.

All tokens and state published here derive from the fixed public prompt below.
Compilation/distribution are offline costs, separate from benchmark responses.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

import pllm
from pllm.kernels import Cpu
from pllm.model_loader import resolve_model
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.state import ClientPrefixReuse, PublicPrefixCapsule
from pllm.tokenization import IndexedTokenizer
from pllm.runtime.benchmark_memory import benchmark_memory, require_admission, MemoryWatchdog
from pllm.runtime.model_binding import compile_runtime_model
from pllm.runtime.transformer_client import ClientBundle
from pllm.runtime.transformer_engine import MaskedTransformerEngine
from decoder_probe_support import stage_output

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = ("This public tutorial explains private inference. The client owns the prompt and state. "
          "Preparation and Inference have separate duties and must not collude. " * 4
          + "Explain why neither server can see the prompt.")
INPUT_BOUND, OUTPUT_BOUND, PREFIX_ROWS = 256, 8, 96


def build(directory: Path):
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = pllm.Model.hf(MODEL, revision=REVISION)
    baseline = MaskedLinearCpu(source, kernels=Cpu(threads=4),
        quantization=SymmetricPerRow(weight_bits=8, activation_bits=8, causal_reduction="prefix_f32"),
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed", prefill_pruning="terminal"),
        inventory=PreparedInventory(refill="on-demand"),
        cache=ClientPrefixReuse(max_bytes=32 << 20, fixed_input_tokens=INPUT_BOUND),
        delivery=ClientBundleTransport("artifacts", compression="zlib", batch_objects=64, storage="paged"))
    budget = pllm.ExecutionBudget(1, INPUT_BOUND, OUTPUT_BOUND)
    deployment = pllm.Deployment.local(root="local://incremental-sdk")
    def experiment(name, pipeline):
        return pllm.Experiment(name, pipeline, deployment, budget)
    # The resident full prepared topology is a conservative bound on this
    # single trusted publisher's clear engine; do not load before admission.
    producer = baseline.with_params(delivery=None)
    admission = benchmark_memory(experiment("publisher", producer), cache_bytes=32 << 20)
    require_admission(admission)
    guard = MemoryWatchdog(admission, lambda message: None)
    engine = MaskedTransformerEngine(weight_bits=8, activation_bits=8, threads=4, prepared_output_encoding="row_residues")
    started, cpu = time.perf_counter(), time.process_time()
    resolved = resolve_model(source)
    if resolved.path is None:
        raise ValueError("checkpoint must resolve locally")
    guard.start()
    try:
        index = IndexedTokenizer.compile(resolved.path / "tokenizer.json", directory / "tokenizer")
        guard.check()
        asyncio.run(engine.load(resolved.manifest))
        guard.check()
        bundle = ClientBundle.unpack(engine.client_bundle(resolved.manifest.id))
        plan = pllm.lower_model((resolved.path / "config.json").read_bytes(), batch=1,
                               max_input_tokens=INPUT_BOUND, max_new_tokens=OUTPUT_BOUND)
        compiled = compile_runtime_model(plan, bundle, composition=producer)
        tokens = bundle.tokenizer().encode(bundle.render_prompt([{"role": "user", "content": PROMPT}]),
                         add_bos=bool(bundle.tokenizer_descriptor.get("add_bos_token", True)))
        if not PREFIX_ROWS < len(tokens) <= INPUT_BOUND:
            raise ValueError("public task outside compiled bounds")
        def remote(stage, values):
            guard.check()
            return stage_output(engine.models[resolved.manifest.id].stages[stage], values, 8)
        runtime = compiled.runtime(remote)
        _, logits, _ = runtime.prepare_ids(tokens[:PREFIX_ROWS])
        capsule = PublicPrefixCapsule.publish(compiled, public_token_ids=tokens[:PREFIX_ROWS],
            snapshot=runtime.snapshot(), logits=logits, path=directory / "prefix.msgpack")
        guard.check()
        pipelines = {
            "control": baseline,
            "demand": baseline.with_params(inventory=PreparedInventory(refill="on-demand", allocation="demand")),
            "paged": baseline.with_params(preparation=ModelAwareCorrections(storage="paged")),
            "indexed": baseline.with_params(tokenizer=index),
            "capsule": baseline.with_params(public_prefix=capsule),
            "combined": baseline.with_params(tokenizer=index, public_prefix=capsule,
                preparation=ModelAwareCorrections(storage="paged"),
                inventory=PreparedInventory(refill="on-demand", allocation="demand")),
        }
        for name, pipeline in pipelines.items():
            value = experiment(name, pipeline)
            value.resolve()
            (directory / (name + ".json")).write_text(json.dumps(value.to_spec(), indent=2) + "\n")
        (directory / "prompt.txt").write_text(PROMPT)
        evidence = {"schema": "pllm.public_artifact_build.v1", "source_lock_digest": resolved.source_lock_digest,
            "body_fingerprint": bundle.privacy["body_fingerprint"], "public_prefix_tokens": PREFIX_ROWS,
            "input_tokens": len(tokens), "output_cap": OUTPUT_BOUND,
            "tokenizer": {"digest": index.params["digest"],
                "database_bytes": (directory / "tokenizer/bpe.sqlite").stat().st_size,
                "contract_bytes": (directory / "tokenizer/contract.json").stat().st_size,
                "bytes": sum((directory / "tokenizer" / name).stat().st_size
                             for name in ("bpe.sqlite", "contract.json"))},
            "capsule": {"digest": capsule.params["digest"], "bytes": capsule.params["size_bytes"]},
            "offline_publisher_wall_seconds": time.perf_counter() - started,
            "offline_publisher_cpu_seconds": time.process_time() - cpu,
            "observed_new_swap_bytes": guard.maximum_swap_growth_bytes,
            "scope": "trusted public compilation including checkpoint load; pre-positioned artifact distribution unmeasured"}
        (directory / "publisher.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
    finally:
        if resolved.manifest.id in engine.models:
            asyncio.run(engine.unload(resolved.manifest.id))
        guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
