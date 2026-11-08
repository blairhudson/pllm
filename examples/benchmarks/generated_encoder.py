"""Public generated FP32 encoder baseline, using the ordinary suite engine.

    pllm benchmark run --experiment examples/benchmarks/generated_encoder.py:candidate \
      --factory --trust-python --input-file target/generated-encoder/inputs.json \
      --output target/generated-encoder/benchmark.json

Factory writes a tiny deterministic checkpoint and default inputs under ignored
target/. This is public client-local inference, not a protocol/privacy benchmark,
THOR/NEXUS implementation, trained quality result, or independent numeric oracle.
"""
from pathlib import Path
import json

from pllm import Deployment, Experiment, ForwardBudget, Model
from pllm.kernels import Cpu
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling
from pllm.numeric import SourceFloat32
from pllm.profiles import ClientOnlyCpu
from pllm.sources.generated_encoder import write_checkpoint


def candidate():
    root = Path("target/generated-encoder").resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = write_checkpoint(root / "checkpoint")
    inputs = root / "inputs.json"
    if not inputs.exists():
        inputs.write_text(json.dumps({"schema_version": "pllm.forward_inputs.v1",
            "tokens": [[1, 2, 3]], "positions": [[0, 1, 2]],
            "segments": [[0, 0, 1]], "attention_mask": [[True, True, False]]}) + "\n")
    experiment = Experiment("generated-encoder",
        ClientOnlyCpu(Model.path(str(checkpoint), model_id="generated-encoder"),
                      kernels=Cpu(threads=2), numeric=SourceFloat32()),
        Deployment.local(root=str(root / "runtime")),
        ForwardBudget(requests=1, max_input_tokens=6))
    return BenchmarkSuite((BenchmarkCase("candidate", BenchmarkTarget.forward(experiment)),),
        # 704 MiB = 320 MiB overhead + 3 * 128 MiB runtime; shared partition unchanged.
        Sampling(processes=1, warmups=0, repetitions=1, threads=2, memory_bytes=704 << 20),
        name="generated-encoder")
