"""Source-locked, public-calibrated prepared precision screen.

Fit and quality are separate commands. Frozen W8A8, W4A8, W8A4 and equalized
W8A8 choices use ordinary Experiments and the shared clear-kernel quality driver.
The held-out set is never a calibration input. This does not benchmark providers
or select a serving default. Public profile compilation and bytes stay explicit.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = Path(__file__).with_name("data") / "smoothquant-calibration.json"
HELD_OUT = Path(__file__).with_name("data") / "prepared-precision-quality.json"


def candidates(control, profile_digest):
    from pllm.quantization import PublicPerChannelEqualized, SymmetricPerRow

    return [control.with_params(name="precision-" + name, pipeline__quantization=choice)
            for name, choice in (
                ("w8a8", SymmetricPerRow(weight_bits=8, activation_bits=8)),
                ("w4a8", SymmetricPerRow(weight_bits=4, activation_bits=8)),
                ("w8a4", SymmetricPerRow(weight_bits=8, activation_bits=4)),
                ("equalized-w8a8", PublicPerChannelEqualized(profile_digest)))]


def main():
    from pllm import load_model
    from pllm.runtime.benchmark_memory import GiB, MemoryWatchdog, host_memory
    from pllm.runtime.public_equalization import profile_path, load_public_equalization_profile
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "quality"))
    parser.add_argument("--experiment", required=True, help="trusted module:factory")
    parser.add_argument("--profile-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (args.command == "quality" and args.profile_report is None):
        parser.error("use a new output path; quality also needs --profile-report")
    module, factory = args.experiment.split(":", 1)
    public = json.loads(CALIBRATION.read_text())
    held_out = json.loads(HELD_OUT.read_text())
    if set(public) & set(held_out):
        raise ValueError("held-out prompts overlap public calibration")
    control = getattr(importlib.import_module(module), factory)().with_params(
        budget__requests=len(held_out), budget__max_input_tokens=128, budget__max_new_tokens=4)
    # The ordinary local-clear diagnostic imports resident boundary matrices.
    # Declare that storage choice in every scored Experiment instead of claiming
    # it executes the factory's paged delivery contract.
    if control.pipeline.components.get("delivery") is not None:
        control = control.with_params(pipeline__delivery__storage="memory")
    manifest = load_model(control.pipeline.model)
    source = Path(manifest.source)
    elements, largest = 0, 0
    for row in manifest.metadata["source_lock"]["files"]:
        if not row["path"].endswith(".safetensors"):
            continue
        with safe_open(source / row["path"], framework="np") as store:
            for key in store.keys():
                tensor = store.get_slice(key)
                if tensor.get_dtype() not in {"BF16", "F16", "F32"}:
                    raise ValueError("precision admission requires floating source weights")
                size = math.prod(tensor.get_shape())
                elements += size
                largest = max(largest, size)
    if not elements:
        raise ValueError("source has no floating weights")
    # Quantized all-stage engine, native snapshots, boundary import copies,
    # float32 calibration chunks and retained logits precede admission.
    required = (elements * 5 + 3) // 4 + 6 * largest + 2 * GiB
    if args.command == "quality":
        from pllm.runtime.streamed_reference import StreamedFloat32Reference

        with StreamedFloat32Reference(source, max_input_tokens=128, max_decode_steps=3) as reference:
            required = max(required, reference.allocation_estimate_bytes + 512 * 1024**2)
    host = host_memory()
    report = {"schema": "pllm.prepared_precision.v1", "command": args.command, "scope": __doc__,
        "experiment_factory": args.experiment, "model": control.pipeline.model.to_spec(),
        "source_lock_digest": manifest.source_lock_digest,
        "admission": {"host": asdict(host), "required_bytes": required,
                      "headroom_bytes": host.available - host.reserve},
        "calibration_sha256": hashlib.sha256(CALIBRATION.read_bytes()).hexdigest(),
        "held_out_sha256": hashlib.sha256(HELD_OUT.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(),
                                    ROOT / "python/pllm/runtime/public_equalization.py",
                                    ROOT / "python/pllm/runtime/reference_benchmark.py")},
        "status": "memory-rejected", "quality": None}
    if required <= host.available - host.reserve:
        guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}},
                               lambda _: signal.raise_signal(signal.SIGINT))
        started, cpu = time.perf_counter(), time.process_time()
        guard.start()
        try:
            if args.command == "fit":
                from transformers import AutoTokenizer
                from pllm.quantization import fit_public_equalization_profile

                tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
                tokens = tuple(tuple(tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], tokenize=True,
                    add_generation_prompt=True, return_dict=False)) for text in public)
                del tokenizer
                profile = fit_public_equalization_profile(control.pipeline.model, tokens, threads=4)
                payload = profile.pack()
                target = profile_path(source, profile.digest)
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError("existing content-addressed profile is corrupt")
                if not target.exists():
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                        temporary.write(payload)
                        temporary_path = Path(temporary.name)
                    try:
                        os.replace(temporary_path, target)
                    finally:
                        temporary_path.unlink(missing_ok=True)
                report.update(profile_digest=profile.digest, profile_bytes=len(payload),
                              calibration_digest=profile.calibration_digest,
                              calibration_token_counts=list(map(len, tokens)),
                              stage_count=len(profile.stage_scales))
            else:
                from pllm.runtime.reference_benchmark import run_reference_benchmark

                build = json.loads(args.profile_report.read_bytes())
                if (build.get("status") != "completed" or build.get("command") != "fit"
                        or build.get("source_lock_digest") != manifest.source_lock_digest
                        or build.get("calibration_sha256") != report["calibration_sha256"]):
                    raise ValueError("profile report differs from the frozen calibration/source")
                profile = load_public_equalization_profile(source, build["profile_digest"], manifest.source_lock_digest)
                options = candidates(control, profile.digest)
                report["experiments"] = [option.to_spec() for option in options]
                report["profile_digest"] = profile.digest
                report["profile_bytes"] = len(profile.pack())
                report["profile_report_sha256"] = hashlib.sha256(args.profile_report.read_bytes()).hexdigest()
                report["quality"] = run_reference_benchmark(options, held_out, top_k=5,
                                                           reference_storage="streamed", decode_steps=3)
            guard.check()
            if guard.error:
                raise RuntimeError(guard.error)
            report.update(status="completed", elapsed_seconds=time.perf_counter() - started,
                process_cpu_seconds=time.process_time() - cpu,
                maximum_swap_growth_bytes=guard.maximum_swap_growth_bytes,
                diagnostic_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024))
        finally:
            guard.close()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "profile_digest": report.get("profile_digest")}))


if __name__ == "__main__":
    main()
