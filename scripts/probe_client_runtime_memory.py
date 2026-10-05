"""Fresh-process client RSS through the ordinary supervised benchmark path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile

MODES = ("eager-resident", "lazy-resident", "lazy-paged")


def experiment(mode, *, tiny=False, four_b=False):
    from pllm import Model
    if tiny:
        model = Model.tiny(model_id="isolated-client-memory")
    elif four_b:
        model = Model.hf("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c")
    else:
        model = Model.hf("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775")
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples/benchmarks/client_memory.py"))
    return example["candidate"](mode, model, "paged" if mode == "lazy-paged" else "memory")


def worker(args):
    from pllm.runtime.benchmark_cli import run_loopback_benchmark
    profiles = {}
    if args.profile_only:
        from pllm.runtime import client as client_module, model_binding, semantic_executor, transformer_client
        from pllm.runtime.benchmark_memory import process_memory

        def watch(owner, name):
            original = getattr(owner, name)
            label = f"{owner.__name__}.{name}"

            def measured(*values, **kwargs):
                before = process_memory()
                try:
                    return original(*values, **kwargs)
                finally:
                    after = process_memory()
                    row = profiles.setdefault(label, {"calls": 0, "peak_before_max": 0, "peak_after_max": 0,
                                                       "rss_after_max": 0, "new_highwater_max": 0})
                    row["calls"] += 1
                    row["peak_before_max"] = max(row["peak_before_max"], before["lifetime_peak_rss_bytes"] or 0)
                    row["peak_after_max"] = max(row["peak_after_max"], after["lifetime_peak_rss_bytes"] or 0)
                    row["rss_after_max"] = max(row["rss_after_max"], after["rss_bytes"])
                    if before["lifetime_peak_rss_bytes"] and after["lifetime_peak_rss_bytes"]:
                        row["new_highwater_max"] = max(row["new_highwater_max"],
                            after["lifetime_peak_rss_bytes"] - before["lifetime_peak_rss_bytes"])
            setattr(owner, name, measured)

        watch(client_module.RuntimeClient, "_load_artifact_bundle")
        watch(model_binding, "compile_runtime_model")
        watch(transformer_client.ClientBundle, "tokenizer")
        for name in ("prepare_ids", "forward_ids", "snapshot"):
            watch(semantic_executor.SemanticDecoderRuntime, name)
    if args.child == "eager-resident":
        # Test-local historical allocation control. The ordinary SDK remains lazy.
        from pllm.runtime import client
        from pllm.runtime.preparation_protocol import expand_preparation_mask, expand_output_mask
        original = client.PreparedStageRows

        def eager(*values, **kwargs):
            request = kwargs.get("request", values[0] if values else None)
            kwargs.update(input_mask=expand_preparation_mask(request),
                          output_mask=expand_output_mask(request))
            return original(*values, **kwargs)

        client.PreparedStageRows = eager
    result = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
        experiment=experiment(args.child, tiny=args.tiny, four_b=args.four_b),
        prompt="Explain private inference in one sentence.", max_output_tokens=8,
        warmups=0, repetitions=1, timeout_seconds=300, temperature=0,
        capture_output_digest=True, _cohort_salt=bytes.fromhex(os.environ["PLLM_MEMORY_COHORT_SALT"]))
    result["client_process_memory"]["isolated_candidate"] = True
    result["client_process_memory"]["allocation_control"] = args.child
    if profiles:
        result["memory_phase_diagnostic"] = {"scope": "nested process samples, non-additive", "phases": profiles}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--four-b", action="store_true")
    parser.add_argument("--profile-only", action="store_true", help="single paged diagnostic; no paired comparison")
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.tiny and args.four_b:
        parser.error("choose one model")
    if args.child:
        def interrupted(*_):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        worker(args)
        return
    from huggingface_hub.constants import HF_HUB_CACHE
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_memory import BenchmarkMemoryError, require_admission
    records = {}
    salt = os.urandom(32).hex()
    def checkpoint(document):
        archive.seek(0)
        json.dump(document, archive, indent=2, sort_keys=True)
        archive.truncate()
        archive.flush()
    with args.output.open("x") as archive, tempfile.TemporaryDirectory(prefix="pllm-client-runtime-memory-") as directory:
        checkpoint({"schema": "pllm.client_runtime_memory_probe.v1", "status": "running", "records": records})
        root = Path(directory)
        for mode in (("lazy-paged",) if args.profile_only else MODES):
            preflight = benchmark_memory(experiment(mode, tiny=args.tiny, four_b=args.four_b), backend="native")
            require_admission(preflight)
            # The parent remains live and eager control restores historical masks.
            extra = 256 << 20
            if mode == "eager-resident":
                estimates = preflight["estimate"]
                extra += estimates["components"]["legacy_client_mask_bytes"] * 5 // 4
            required = preflight["estimate"]["native_total_peak_bytes"] + extra
            if required > preflight["host"]["admission_budget_bytes"]:
                raise BenchmarkMemoryError("isolated client probe exceeds physical headroom including control allocations")
            cache = root / mode
            cache.mkdir()
            output = root / f"{mode}.json"
            env = dict(os.environ, XDG_CACHE_HOME=str(cache), HF_HUB_CACHE=HF_HUB_CACHE,
                       HUGGINGFACE_HUB_CACHE=HF_HUB_CACHE, HF_HUB_OFFLINE="1",
                       PLLM_MEMORY_COHORT_SALT=salt)
            command = [sys.executable, __file__, "--child", mode, "--output", str(output)]
            if args.tiny:
                command.append("--tiny")
            if args.four_b:
                command.append("--four-b")
            if args.profile_only:
                command.append("--profile-only")
            print(f"running {mode}", flush=True)
            child = subprocess.Popen(command, env=env)
            try:
                if child.wait(timeout=420) != 0:
                    raise RuntimeError(f"{mode} failed")
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
            records[mode] = json.loads(output.read_bytes())
            checkpoint({"schema": "pllm.client_runtime_memory_probe.v1", "status": "running", "records": records})
    # Same salt, public prompt, workload, numeric body and outputs in all children.
    controls = [record["runs"][0] for record in records.values()]
    result = {"schema": "pllm.client_runtime_memory_probe.v1", "status": "complete", "records": records,
              "comparison_available": len(controls) > 1,
              "checks_passed": all(record["checks"]["passed"] for record in records.values())}
    for key in (() if args.profile_only else ("model_fingerprint", "input_tokens", "output_tokens", "output_text_digest")):
        values = [(run["tokens"].get(key) if key.endswith("_tokens") else
                   run["generation"].get(key) if key == "output_text_digest" else run.get(key)) for run in controls]
        result[f"matched_{key}"] = len({json.dumps(value, sort_keys=True) for value in values}) == 1
        if values[0] is None:
            raise ValueError(f"benchmark did not report {key}")
    if not args.profile_only:
        result["matched_prompt_digest"] = len({row["configuration"]["prompt_digest"] for row in records.values()}) == 1
    if not all(value for key, value in result.items() if key.startswith("matched_") or key == "checks_passed"):
        raise ValueError("client memory cohort mismatch")
    with args.output.open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps({mode: row["client_process_memory"] for mode, row in records.items()}, indent=2))


if __name__ == "__main__":
    main()
