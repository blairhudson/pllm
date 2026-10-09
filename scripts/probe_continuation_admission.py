"""Compare stateless native revalidation with a once-validated geometry handle.

This metadata-only probe loads no tensor values, starts no providers and measures
no response latency or client peak RSS. Both paths retain identical admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

from pllm import Model, _native, lower_model
from pllm.model_loader import resolve_model
from pllm.profiles import MaskedLinearCpu
from pllm.quantization import SymmetricPerRow

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"


def run(samples, calls):
    model = Model.hf(MODEL, revision=REVISION)
    source = resolve_model(model)
    config = (source.path / "config.json").read_bytes()
    plan = lower_model(config, batch=1, max_input_tokens=512, max_new_tokens=32)
    pipeline = MaskedLinearCpu(model, quantization=SymmetricPerRow(causal_reduction="prefix_f32"))
    contract = plan.continuation_schedule(pipeline)
    arguments = (plan.canonical_bytes(), pipeline.canonical_bytes(), contract.canonical_bytes())
    rows = []
    for sample in range(samples):
        for choice in (("stateless", "handle") if sample % 2 == 0 else ("handle", "stateless")):
            start = time.process_time()
            handle = _native.DecoderContinuationAdmission(*arguments) if choice == "handle" else None
            construction_cpu = time.process_time() - start
            start = time.process_time()
            admitted = []
            for index in range(calls):
                prefix, query = 39 + index % 128, 1 + index % 16
                admitted.append(handle.admit(prefix, query, 2 << 30) if handle is not None else
                                _native.admit_decoder_continuation(*arguments, prefix, query, 2 << 30))
            cpu = time.process_time() - start
            if any(value != contract.admit(39 + index % 128, 1 + index % 16)
                   for index, value in enumerate(admitted)):
                raise AssertionError("native admission changed resource counts")
            rows.append({"sample": sample, "choice": choice, "calls": calls,
                "construction_cpu_seconds": construction_cpu, "admission_cpu_seconds": cpu,
                "total_cpu_seconds": construction_cpu + cpu,
                "resource_digest": hashlib.sha256(json.dumps(admitted).encode()).hexdigest()})
    return {"schema": "pllm.continuation_admission_probe.v1", "scope": __doc__,
        "model": MODEL, "revision": REVISION, "plan_digest": plan.digest,
        "contract_digest": contract.digest, "composition_digest": pipeline.digest(),
        "source_lock_digest": source.source_lock_digest,
        "serialized_public_metadata_bytes": sum(map(len, arguments)),
        "median_total_cpu_seconds": {choice: statistics.median(
            row["total_cpu_seconds"] for row in rows if row["choice"] == choice)
            for choice in ("stateless", "handle")},
        "environment": {"python": sys.version, "platform": platform.platform()},
        "implementation_sha256": {name: hashlib.sha256(
            (Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest() for name in (
                "scripts/probe_continuation_admission.py",
                "crates/pllm-compiler/src/decoder_continuation.rs",
                "crates/pllm-python/src/continuation.rs", "python/pllm/modeling.py")},
        "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6, choices=range(1, 13))
    parser.add_argument("--calls", type=int, default=32, choices=range(1, 257))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result = run(args.samples, args.calls)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["median_total_cpu_seconds"]))
