"""Fresh pinned-W8A8 public range audit; executes only the unchanged clear decoder."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import signal

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = (
    "Explain how a pendulum exchanges kinetic and potential energy.",
    "Give two reasons libraries use catalog numbers.",
    "What makes a numerical estimate reproducible?",
    "Describe the role of evaporation in cooling a wet cloth.",
    "Compare a queue and a stack using everyday examples.",
    "Why might a bridge expand during warm weather?",
    "Write a short checklist for measuring rainfall accurately.",
    "Explain how crop rotation can help soil fertility.",
    "What information belongs in a useful bug report?",
    "Describe why dividing by zero is undefined.",
    "How does a compass help someone navigate a trail?",
    "Summarize why independent experimental controls matter.",
)
SOURCES = ("benchmarks/research/encrypted_attention_range.py", "scripts/decoder_probe_support.py",
           "python/pllm/runtime/ckks_softmax.py", "python/pllm/metrics/encrypted_softmax.py")


def run():
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from scripts.decoder_probe_support import DecoderFixture, digest
    host = host_memory()
    if host.available - host.reserve < 4 * GiB:
        raise RuntimeError("range audit needs 4 GiB beyond host reserve")
    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
                           lambda _: os.kill(os.getpid(), signal.SIGTERM))
    guard.start()
    fingerprints = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}
    try:
        fixture = DecoderFixture(inputs=128, outputs=4)
        observations = []
        for prompt in PROMPTS:
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            tokens = fixture.tokens(prompt)
            runtime = fixture.compiled.runtime(fixture.remote)
            original = runtime._local
            counts = {"operators": 0, "valid_scores": 0, "outside_range": 0,
                      "maximum_absolute_score": 0.0, "maximum_key_extent": 0}
            def local(op, values, *state):
                if op["operator"] == "softmax":
                    scores = values[op["inputs"][0]]
                    if scores.dtype != np.float32 or scores.ndim != 4 or np.isnan(scores).any() or np.isposinf(scores).any():
                        raise ValueError("unexpected clear attention score contract")
                    valid = scores[np.isfinite(scores)]
                    counts["operators"] += 1
                    counts["valid_scores"] += valid.size
                    counts["outside_range"] += int(np.count_nonzero(np.abs(valid) > 0.125))
                    counts["maximum_absolute_score"] = max(counts["maximum_absolute_score"], float(np.max(np.abs(valid))))
                    counts["maximum_key_extent"] = max(counts["maximum_key_extent"], scores.shape[-1])
                return original(op, values, *state)
            runtime._local = local
            phases = []
            try:
                _, logits, _ = runtime.prepare_ids(tokens)
                phases.append({"phase": "prefill", **counts})
                for step in range(3):
                    for key in counts:
                        counts[key] = 0.0 if key == "maximum_absolute_score" else 0
                    logits = runtime.forward_ids([int(np.argmax(logits))])[-1]
                    phases.append({"phase": "decode", "step": step, **counts})
            finally:
                del runtime._local
            observations.append({"input_tokens": len(tokens), "phases": phases})
            del local, original, runtime
        if fingerprints != {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCES}:
            raise ValueError("range audit sources changed")
        phases = [phase for observation in observations for phase in observation["phases"]]
        return {"schema": "pllm.encrypted_attention_range.v1", "source_sha256": fingerprints,
            "model": fixture.lock(), "public_prompt_digest": digest(PROMPTS),
            "public_score_bound": 0.125, "prompt_count": len(PROMPTS), "positions": len(phases),
            "positions_outside_range": sum(p["outside_range"] > 0 for p in phases),
            "valid_scores": sum(p["valid_scores"] for p in phases),
            "outside_range": sum(p["outside_range"] for p in phases),
            "maximum_absolute_score": max(p["maximum_absolute_score"] for p in phases),
            "observations": observations,
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "maximum_swap_growth_bytes": guard.maximum_swap_growth_bytes,
            "scope": {"execution": "unchanged pinned W8A8 local-clear control", "public_inputs": True,
                "candidate_executed": False, "candidate_quality": None, "encrypted_outputs": 0,
                "private_range_fitting": False, "clipping": False,
                "latency_comparison": False, "wire_measurement": None, "measured_peak_memory": None}}
    finally:
        guard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in ("positions", "positions_outside_range", "maximum_absolute_score")}))
