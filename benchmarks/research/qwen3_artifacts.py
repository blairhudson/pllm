"""Ordinary SDK artifact ablations; compile public artifacts before importing factories."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

from pllm import Deployment, ExecutionBudget, Experiment, Model
from pllm.kernels import Cpu
from pllm.preparation import ModelAwareCorrections, PreparedInventory
from pllm.profiles import MaskedLinearCpu
from pllm.protocols import ClientBundleTransport, MaskedLinear
from pllm.quantization import SymmetricPerRow
from pllm.tokenization import IndexedTokenizer

from benchmarks.research.common import artifact_directory

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PROMPT = "Explain private inference in one sentence."
SOURCE = Model.hf(MODEL, revision=REVISION, local_files_only=True)
TOKENIZER_CASES = (PROMPT, "Hello, world!", " café\nnaïve 🐋", "你好，世界", "A" * 128,
                   "<|im_start|>user\nPublic test.<|im_end|>\n", "1234567890", " tabs\tand  spaces\n")


def control():
    return Experiment("qwen3-sdk-paged-control", MaskedLinearCpu(SOURCE,
        kernels=Cpu(threads=4), quantization=SymmetricPerRow(weight_bits=8, activation_bits=8),
        linear=MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed", prefill_pruning="terminal"),
        inventory=PreparedInventory(refill="on-demand", allocation="demand"),
        preparation=ModelAwareCorrections(storage="paged"),
        delivery=ClientBundleTransport("artifacts", storage="paged", compression="zlib", batch_objects=64)),
        Deployment.local(root="local://research-qwen3-artifacts"),
        ExecutionBudget(requests=1, max_input_tokens=32, max_new_tokens=8))


def indexed():
    root = artifact_directory()
    metadata = json.loads((root / "publisher.json").read_text())
    if metadata["model"] != MODEL or metadata["revision"] != REVISION:
        raise ValueError("public tokenizer artifact belongs to another source")
    selection = IndexedTokenizer(str(root / "tokenizer"), digest=metadata["tokenizer_contract_sha256"])
    return control().with_params(name="qwen3-sdk-paged-indexed", pipeline__tokenizer=selection)


def build(directory: Path):
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.runtime.indexed_tokenizer import IndexedClientTokenizer
    from tokenizers import Tokenizer

    host = host_memory()
    if host.available - host.reserve < 2 * GiB:
        raise RuntimeError("public tokenizer compiler needs 2 GiB beyond host reserve")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
                           lambda _: os.kill(os.getpid(), signal.SIGTERM))
    guard.start()
    try:
        source = resolve_model(SOURCE)
        if source.path is None:
            raise ValueError("pinned checkpoint must already be cached")
        start, cpu = time.perf_counter(), time.process_time()
        selection = IndexedTokenizer.compile(source.path / "tokenizer.json", directory / "tokenizer")
        build_wall, build_cpu = time.perf_counter() - start, time.process_time() - cpu
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
        raw = (source.path / "tokenizer.json").read_bytes()
        config = json.loads((source.path / "config.json").read_text())
        # Tokenizer-only microprobe: the ordinary full SDK comparison below uses
        # real admitted bundles. No model values are loaded by this publisher.
        owner = SimpleNamespace(tokenizer_descriptor={"kind": "tokenizer_json", "model": raw}, cfg=config)
        original = Tokenizer.from_str(raw.decode())
        compiled = IndexedClientTokenizer(owner, selection)
        timing = {}
        try:
            for text in TOKENIZER_CASES:
                expected = original.encode(text, add_special_tokens=False).ids
                if compiled.encode(text) != expected or compiled.decode(expected) != original.decode(expected):
                    raise ValueError("public tokenizer encode/decode mismatch")
            for name, encode, decode in (
                ("original", lambda x: original.encode(x, add_special_tokens=False).ids, original.decode),
                ("indexed", compiled.encode, compiled.decode),
            ):
                tokens = [encode(text) for text in TOKENIZER_CASES]
                start = time.process_time()
                for _ in range(50):
                    for text in TOKENIZER_CASES:
                        encode(text)
                encode_cpu = time.process_time() - start
                start = time.process_time()
                for _ in range(50):
                    for ids in tokens:
                        decode(ids)
                timing[name] = {"encode_cpu_seconds": encode_cpu, "decode_cpu_seconds": time.process_time() - start,
                                "calls_per_direction": 50 * len(TOKENIZER_CASES)}
        finally:
            compiled.close()
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
        metadata = {"schema": "pllm.public_tokenizer_build.v1", "model": MODEL, "revision": REVISION,
            "source_lock_digest": source.source_lock_digest,
            "source_sha256": hashlib.sha256(raw).hexdigest(), "tokenizer_contract_sha256": selection.params["digest"],
            "artifact_bytes": sum((directory / "tokenizer" / p).stat().st_size for p in ("bpe.sqlite", "contract.json")),
            "offline_compiler_wall_seconds": build_wall, "offline_compiler_cpu_seconds": build_cpu,
            "warm_tokenizer_microprobe": timing, "checked_public_cases": len(TOKENIZER_CASES),
            "public_cases_sha256": hashlib.sha256(json.dumps(TOKENIZER_CASES).encode()).hexdigest(),
            "configuration_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "distribution": "pre-positioned; artifact bytes and compiler CPU additional to response counters",
            "scope": "public compilation and local tokenizer microprobe; fresh-client response RSS measured separately"}
        (directory / "publisher.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        (directory / "prompt.txt").write_text(PROMPT + "\n")
        print(json.dumps(metadata))
    finally:
        guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    build(parser.parse_args().directory)
