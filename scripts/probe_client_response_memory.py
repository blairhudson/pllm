"""Matched fresh-client allocation ablations under native WAN caps.

Both children run the same ordinary Experiment. The historical control restores
only the two artifact walkers from a pinned PLLM commit and leaves HTTPX's closed
response streams attached, as before the lifetime fix. The optional kv-capacity
control instead restores geometric full-KV growth and its original admission
estimate. No tracing or forced GC runs inside either response. One private
parent-owned salt binds both reports; parent and child use matching admission.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "48ad67d"
KV_BASELINE = "6ba400fe0ed17b2bb8675506d77ad5b0638646f3"
MODES = ("legacy", "scoped")


def restore_legacy_lifetimes():
    from pllm.runtime import bundle_artifacts, client

    path = "python/pllm/runtime/bundle_artifacts.py"
    source = subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=ROOT)
    names = {"batched_reader", "reconstruct_document"}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("pinned artifact lifetime control is incomplete")
    namespace = dict(vars(bundle_artifacts))
    exec(compile(ast.Module(body=functions, type_ignores=[]), f"{BASELINE}:{path}", "exec"), namespace)
    bundle_artifacts.batched_reader = namespace["batched_reader"]
    bundle_artifacts.reconstruct_document = namespace["reconstruct_document"]
    client.reconstruct_document = namespace["reconstruct_document"]
    client.owned_response = nullcontext


@contextmanager
def allocation_control(control, mode):
    from pllm.runtime import bundle_artifacts, client, decoder_memory, transformer_client

    originals = [(bundle_artifacts, "batched_reader", bundle_artifacts.batched_reader),
        (bundle_artifacts, "reconstruct_document", bundle_artifacts.reconstruct_document),
        (client, "reconstruct_document", client.reconstruct_document),
        (client, "owned_response", client.owned_response),
        (transformer_client.LayerCache, "append", transformer_client.LayerCache.append),
        (decoder_memory, "decoder_memory", decoder_memory.decoder_memory)]
    try:
        if mode == "legacy" and control == "response-lifetimes":
            restore_legacy_lifetimes()
        elif mode == "legacy" and control == "kv-capacity":
            functions = []
            for module, name, owner in ((transformer_client, "append", "LayerCache"),
                                       (decoder_memory, "decoder_memory", None)):
                path = f"python/pllm/runtime/{module.__name__.rsplit('.', 1)[1]}.py"
                source = subprocess.check_output(["git", "show", f"{KV_BASELINE}:{path}"], cwd=ROOT)
                nodes = ast.parse(source).body
                if owner is not None:
                    nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner)
                function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
                namespace = dict(vars(module))
                exec(compile(ast.Module(body=[function], type_ignores=[]), f"{KV_BASELINE}:{path}", "exec"), namespace)
                functions.append(namespace[name])
            legacy_append, decoder_memory.decoder_memory = functions

            def append(self, key, value, *, maximum_sequence=None):
                return legacy_append(self, key, value)

            transformer_client.LayerCache.append = append
        yield
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def experiment(target, mode, control="response-lifetimes"):
    module, factory = target.split(":", 1)
    value = getattr(importlib.import_module(module), factory)()
    suffix = "lifetimes" if control == "response-lifetimes" else control
    return replace(value, name=f"{value.name}-{mode}-{suffix}")


def worker(args):
    from pllm.deployment import WanConditions
    from pllm.runtime.benchmark_cli import run_loopback_benchmark

    with allocation_control(args.allocation_control, args.child):
        report = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            experiment=experiment(args.experiment, args.child, args.allocation_control),
            prompt=args.prompt_file.read_text().strip(), max_output_tokens=args.max_output_tokens,
            warmups=0, repetitions=1, timeout_seconds=1800, temperature=0,
            capture_output_digest=True, backend="native", wan=WanConditions(), emulate_wan=True,
            _cohort_salt=bytes.fromhex(os.environ["PLLM_MEMORY_COHORT_SALT"]))
    report["client_process_memory"]["isolated_candidate"] = True
    report["client_process_memory"]["allocation_control"] = args.child
    report["client_process_memory"]["allocation_ablation"] = args.allocation_control
    report["configuration"]["client_process_isolated"] = True
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--allocation-control", choices=("response-lifetimes", "kv-capacity"),
                        default="response-lifetimes")
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 1 <= args.max_output_tokens <= 32:
        parser.error("output bound must be 1..32")

    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    if args.child:
        worker(args)
        return

    from huggingface_hub.constants import HF_HUB_CACHE
    from pllm.metrics import benchmark_memory
    from pllm.runtime.benchmark_cli import build_comparison_report
    from pllm.runtime.benchmark_memory import BenchmarkMemoryError, require_admission

    salt = os.urandom(32).hex()
    baseline = BASELINE if args.allocation_control == "response-lifetimes" else KV_BASELINE
    baseline_files = (("bundle_artifacts.py",) if args.allocation_control == "response-lifetimes"
                      else ("transformer_client.py", "decoder_memory.py"))
    baseline_hashes = {name: hashlib.sha256(subprocess.check_output(
        ["git", "show", f"{baseline}:python/pllm/runtime/{name}"], cwd=ROOT)).hexdigest()
        for name in baseline_files}
    sources = [Path(__file__).resolve(), *(ROOT / "python/pllm/runtime" / name
                for name in ("client.py", "bundle_artifacts.py", "http_response.py", "transformer_client.py",
                             "semantic_executor.py", "decoder_memory.py"))]
    provenance = {
        "baseline_commit": subprocess.check_output(["git", "rev-parse", baseline], cwd=ROOT, text=True).strip(),
        "allocation_control": args.allocation_control,
        "baseline_source_sha256": baseline_hashes,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "scope": __doc__,
    }
    candidates = []
    with args.output.open("x") as archive, tempfile.TemporaryDirectory(
        prefix="client-response-memory-", dir=ROOT / "target"
    ) as directory:
        def checkpoint(status):
            archive.seek(0)
            json.dump({"schema": "pllm.client_response_memory_progress.v1", "status": status,
                "provenance": provenance, "records": [report for _, report in candidates]},
                archive, indent=2, sort_keys=True, allow_nan=False)
            archive.truncate()
            archive.flush()

        checkpoint("running")
        root = Path(directory)
        try:
            for mode in MODES:
                selected = experiment(args.experiment, mode, args.allocation_control)
                with allocation_control(args.allocation_control, mode):
                    preflight = benchmark_memory(selected, backend="native", enforced_wan=True,
                                                 max_output_tokens=args.max_output_tokens)
                require_admission(preflight)
                if (preflight["candidates"]["native"]["required_host_bytes"] + (256 << 20)
                        > preflight["host"]["admission_budget_bytes"]):
                    raise BenchmarkMemoryError("allocation pair exceeds headroom including its idle launcher")
                cache = root / mode
                cache.mkdir()
                output = root / f"{mode}.json"
                env = dict(os.environ, XDG_CACHE_HOME=str(cache), HF_HUB_CACHE=HF_HUB_CACHE,
                    HUGGINGFACE_HUB_CACHE=HF_HUB_CACHE, HF_HUB_OFFLINE="1", PLLM_MEMORY_COHORT_SALT=salt)
                command = [sys.executable, "-m", "scripts.probe_client_response_memory",
                    "--child", mode, "--experiment", args.experiment, "--prompt-file", str(args.prompt_file.resolve()),
                    "--output", str(output), "--max-output-tokens", str(args.max_output_tokens),
                    "--allocation-control", args.allocation_control]
                print(f"running {mode} {args.allocation_control}", flush=True)
                child = subprocess.Popen(command, env=env, cwd=ROOT)
                try:
                    if child.wait(timeout=2100) != 0:
                        raise RuntimeError(f"{mode} allocation child failed")
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            import psutil
                            try:
                                descendants = psutil.Process(child.pid).children(recursive=True)
                            except psutil.Error:
                                descendants = []
                            for process in descendants:
                                try:
                                    process.kill()
                                except psutil.Error:
                                    pass
                            child.kill()
                            child.wait()
                report = json.loads(output.read_bytes())
                candidates.append((selected, report))
                checkpoint("running")
            comparison = build_comparison_report(candidates, compare_kernels=True)
            if comparison["checks"]["passed"] is not True:
                raise ValueError("allocation cohort did not pass canonical comparison")
            comparison["response_lifetime_probe" if args.allocation_control == "response-lifetimes"
                       else "allocation_probe"] = provenance
            for path in sources:
                if hashlib.sha256(path.read_bytes()).hexdigest() != provenance["source_sha256"][str(path.relative_to(ROOT))]:
                    raise ValueError("measured implementation changed during the cohort")
            archive.seek(0)
            json.dump(comparison, archive, indent=2, sort_keys=True, allow_nan=False)
            archive.truncate()
            archive.flush()
        except BaseException:
            checkpoint("incomplete")
            raise
    print(json.dumps({"output": str(args.output), "checks": comparison["checks"],
        "client_memory": {mode: report["client_process_memory"] for mode, (_, report) in zip(MODES, candidates)}}, indent=2))


if __name__ == "__main__":
    main()
