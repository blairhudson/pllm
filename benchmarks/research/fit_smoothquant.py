"""Build Qwen's source-locked profile from the committed public calibration set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import tempfile
import time

from benchmarks.research.paper_baseline import experiment
from pllm import load_model
from pllm.metrics import benchmark_memory
from pllm.quantization import fit_public_equalization_profile
from pllm.runtime.benchmark_memory import MemoryWatchdog
from pllm.runtime.public_equalization import profile_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new reproduction report path")
    root = Path(__file__).parent
    calibration_path = root / "data/smoothquant-calibration.json"
    public_prompts = json.loads(calibration_path.read_text())
    held_out = json.loads((root / "data/paper-quality-prompts.json").read_text())
    if set(public_prompts) & set(held_out):
        raise ValueError("public calibration overlaps the held-out quality cohort")
    control = experiment().with_params(budget__max_input_tokens=128)
    admission = benchmark_memory(control, backend="native")
    if not admission["admitted"]:
        raise RuntimeError("calibration rejected by conservative model memory preflight")
    watchdog = MemoryWatchdog(admission, lambda _: signal.raise_signal(signal.SIGINT))
    started, cpu = time.perf_counter(), time.process_time()
    watchdog.start()
    try:
        from transformers import AutoTokenizer

        manifest = load_model(control.pipeline.model)
        tokenizer = AutoTokenizer.from_pretrained(
            manifest.source, local_files_only=True, trust_remote_code=False,
        )
        tokens = tuple(tuple(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_dict=False,
        )) for prompt in public_prompts)
        profile = fit_public_equalization_profile(control.pipeline.model, tokens, threads=4)
        watchdog.check()
        payload = profile.pack()
        target = profile_path(manifest.source, profile.digest)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("existing profile differs from its content address")
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            try:
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)
    finally:
        watchdog.close()
    report = {
        "schema": "pllm.public_equalization_build.v1",
        "scope": "public offline calibration; separate from online response accounting",
        "model": control.pipeline.model.to_spec(),
        "source_lock_digest": profile.source_lock_digest,
        "profile_digest": profile.digest,
        "calibration_digest": profile.calibration_digest,
        "calibration_file_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "calibration_token_counts": list(map(len, tokens)),
        "stage_count": len(profile.stage_scales),
        "profile_bytes": len(payload),
        "alpha_numerator": 3, "alpha_denominator": 4,
        "elapsed_seconds": time.perf_counter() - started,
        "process_cpu_seconds": time.process_time() - cpu,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
        "memory_admission": admission,
        "maximum_swap_growth_bytes": watchdog.maximum_swap_growth_bytes,
        "environment": {"platform": platform.platform(), "python": platform.python_version(),
                        "numpy": __import__("numpy").__version__,
                        "transformers": __import__("transformers").__version__},
        "limitations": [
            "eight public calibration sequences; not representative task-quality evidence",
            "float32 body calibration with the existing W8 token boundary",
            "fixed alpha, normalized bounded scales, and explicit client scaling differ from original SmoothQuant",
            "the public profile is installed beside the cached checkpoint; distribution is accounted separately",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "profile_digest", "stage_count", "profile_bytes", "calibration_token_counts",
        "elapsed_seconds", "process_cpu_seconds", "process_peak_rss_bytes",
    )}, indent=2))


if __name__ == "__main__":
    main()
