"""Memory-admitted held-out FP32 quality diagnostic for the 4B memory stack."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import signal

from benchmarks.research.qwen3_memory_frontier import resident

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "benchmarks/research/data/paper-quality-prompts.json"


def experiment():
    return resident().with_params(name="qwen3-memory-quality-w8a8", budget__requests=12)


def main():
    from safetensors import safe_open
    from pllm.model_loader import resolve_model
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.runtime.reference_benchmark import run_reference_benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; use a fresh path")
    candidate = experiment()
    source = resolve_model(candidate.pipeline.model)
    if source.path is None:
        raise ValueError("the pinned local checkpoint must be available")
    elements = 0
    tensors = 0
    for row in source.manifest.metadata["source_lock"]["files"]:
        if not row["path"].endswith(".safetensors"):
            continue
        with safe_open(source.path / row["path"], framework="np") as handle:
            for key in handle.keys():
                tensor = handle.get_slice(key)
                if tensor.get_dtype() not in {"BF16", "F16", "F32"}:
                    raise ValueError("quality memory admission needs floating-point weights")
                elements += math.prod(tensor.get_shape())
                tensors += 1
    if not elements:
        raise ValueError("checkpoint has no floating-point tensor payload")
    # Parameter payload alone is a lower bound. Price loader/workspace overlap
    # and 25% slack separately; do not use swap or the client-only estimate.
    fp32_bytes = elements * 4
    required = (fp32_bytes * 5 + 3) // 4 + 2 * GiB
    host = host_memory()
    admitted = host.available - host.reserve >= required
    report = {
        "schema": "pllm.qwen3_memory_quality_probe.v1",
        "status": "admitted" if admitted else "memory-rejected",
        "model": asdict(candidate.pipeline.model),
        "source_lock_digest": source.source_lock_digest,
        "checkpoint_digest": source.checkpoint_digest,
        "configuration_digest": candidate.configuration_digest(),
        "pipeline_digest": candidate.pipeline.digest(),
        "dataset_sha256": hashlib.sha256(PROMPTS.read_bytes()).hexdigest(),
        "scope": "held-out local clear-kernel prefill agreement with locked float32 reference",
        "admission": {"host": asdict(host), "reserve_bytes": host.reserve,
                      "headroom_bytes": host.available - host.reserve,
                      "fp32_weight_payload_lower_bound_bytes": fp32_bytes,
                      "required_bytes": required, "tensor_count": tensors},
        "quality": None,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__).resolve(),
                                       ROOT / "python/pllm/runtime/reference_benchmark.py")},
    }
    if admitted:
        guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
                               lambda _: os.kill(os.getpid(), signal.SIGTERM))
        guard.start()
        try:
            report["quality"] = run_reference_benchmark(
                [candidate], json.loads(PROMPTS.read_text()), top_k=5)
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            report["maximum_swap_growth_bytes"] = guard.maximum_swap_growth_bytes
            report["status"] = "completed"
        finally:
            guard.close()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
