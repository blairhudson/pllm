"""Compare two installed builds using fresh processes and one private SDK cohort salt.

The kernel screen uses public synthetic Qwen3-shaped weights and exact int64
oracles. The response pair runs two ordinary prepared requests per child under
shared native 100/40 caps, with isolated artifact caches and a cached checkpoint.
No allocation tracing, monkeypatching or forced collection runs in the response.
Cold/warm request samples do not establish representative quality or physical wire.
"""
from __future__ import annotations

import argparse
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
MODES = ("baseline", "candidate")


def identity():
    import importlib.metadata
    import platform
    import pllm
    from pllm import _native

    root = Path(pllm.__file__).parent
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "native_sha256": hashlib.sha256(Path(_native.__file__).read_bytes()).hexdigest(),
        "python_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(root.rglob("*.py"))},
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("numpy", "httpx", "msgpack", "tokenizers", "psutil", "fastapi", "websockets")},
    }


def kernel_screen():
    import statistics
    import time
    import tracemalloc
    import numpy as np
    from pllm.native import MaskedGEMM
    from pllm.runtime.bundle_compression import CHUNK_BYTES, encode_bundle_frames, iter_bundle_frames

    rng = np.random.default_rng(109)
    executor = MaskedGEMM(threads=4)
    cases = []
    for out, width in ((2560, 2560), (9728, 2560), (2560, 9728)):
        weights = rng.integers(-128, 128, (out, width), dtype=np.int8)
        matrix = executor.compile(weights)
        for batch in (1, 23):
            inputs = rng.integers(0, 1 << 24, (batch, width), dtype=np.uint32)
            # The largest sum is < 2^45. Row chunks bound oracle weight copies.
            expected = np.empty((batch, out), dtype=np.uint32)
            for start in range(0, out, 64):
                expected[:, start:start + 64] = (
                    inputs.astype(np.int64) @ weights[start:start + 64].astype(np.int64).T
                ) % (1 << 24)
            if not np.array_equal(matrix.modular(inputs, 1 << 24), expected):
                raise ValueError("modular kernel disagrees with independent signed oracle")
            if not np.array_equal(matrix.wrap32(inputs) & ((1 << 24) - 1), expected):
                raise ValueError("wrap32 kernel disagrees with independent signed oracle")
            row = {"shape": [batch, width, out], "checked_outputs": int(expected.size),
                   "output_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
            for name, operation in (("modular", lambda matrix=matrix: matrix.modular(inputs, 1 << 24)),
                                    ("wrap32", lambda matrix=matrix: matrix.wrap32(inputs))):
                samples = []
                for _ in range(9):
                    wall, cpu = time.perf_counter(), time.process_time()
                    operation()
                    samples.append({"wall_seconds": time.perf_counter() - wall,
                                    "cpu_seconds": time.process_time() - cpu})
                row[name] = {"samples": samples, **{
                    "median_" + key: statistics.median(sample[key] for sample in samples)
                    for key in ("wall_seconds", "cpu_seconds")}}
            cases.append(row)
        del matrix, weights

    # Isolate Python frame-allocation peaks from payload and compressed-source owners.
    payload = rng.integers(0, 256, 4 * CHUNK_BYTES, dtype=np.uint8).tobytes()
    framed = b"".join(encode_bundle_frames(payload))
    chunks = tuple(framed[start:start + 65537] for start in range(0, len(framed), 65537))
    traced = []
    for _ in range(3):
        digest, wire = hashlib.sha256(), []
        tracemalloc.start()
        for block in iter_bundle_frames(chunks, expected_size=len(payload), on_wire_bytes=wire.append):
            digest.update(block)
        traced.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
        if digest.hexdigest() != hashlib.sha256(payload).hexdigest() or sum(wire) != len(framed):
            raise ValueError("compressed frame bytes or counters changed")
    return {"cases": cases, "frame_decode": {"raw_bytes": len(payload), "wire_bytes": len(framed),
            "python_traced_peak_bytes": traced, "scope": "decoder allocations; not process RSS"}}


def experiment(target, mode):
    module, factory = target.split(":", 1)
    value = getattr(importlib.import_module(module), factory)()
    return value.with_params(name=f"{value.name}-{mode}-prepared-path", budget__requests=2)


def worker(args):
    if args.probe == "kernels":
        report = kernel_screen()
    else:
        from pllm.deployment import WanConditions
        from pllm.runtime.benchmark_cli import run_loopback_benchmark

        prompt = args.prompt_file.read_text().strip()
        report = run_loopback_benchmark(model="unused", model_id=None, tiny=False,
            experiment=experiment(args.experiment, args.child), prompt=prompt,
            prompt_sequence=(prompt, prompt), max_output_tokens=args.max_output_tokens,
            warmups=0, repetitions=1, timeout_seconds=1800, temperature=0,
            capture_output_digest=True, backend="native", wan=WanConditions(), emulate_wan=True,
            _cohort_salt=bytes.fromhex(os.environ["PLLM_PREPARED_PATH_SALT"]))
        report["client_process_memory"]["isolated_candidate"] = True
        report["configuration"]["client_process_isolated"] = True
    # Hash implementation after measurement; provenance parsing cannot inflate response RSS.
    args.output.write_text(json.dumps({"identity": identity(), "report": report}, indent=2,
                                     sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("kernels", "responses"), required=True)
    parser.add_argument("--change-scope", choices=("kernels", "client-fusion", "prepared-refinements", "native-pacing"), default="kernels")
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if not 1 <= args.max_output_tokens <= 32:
        parser.error("output bound must be 1..32")
    if args.probe == "responses" and (not args.experiment or not args.prompt_file):
        parser.error("responses require an experiment and prompt file")

    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    if args.child:
        worker(args)
        return
    if not args.baseline_python or not args.candidate_python:
        parser.error("both installed-build interpreters are required")

    from huggingface_hub.constants import HF_HUB_CACHE
    from pllm.runtime.benchmark_cli import build_comparison_report
    from pllm.runtime.benchmark_memory import BenchmarkMemoryError, require_admission
    from pllm.metrics import benchmark_memory

    interpreters = dict(zip(MODES, (args.baseline_python.absolute(), args.candidate_python.absolute())))
    provenance = {"scope": __doc__, "change_scope": args.change_scope,
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    records = []
    salt = os.urandom(32).hex()
    with args.output.open("x") as archive, tempfile.TemporaryDirectory(prefix="prepared-path-", dir=ROOT / "target") as directory:
        def checkpoint(status, **extra):
            archive.seek(0)
            json.dump({"schema": "pllm.prepared_path_probe.v1", "status": status,
                       "probe": args.probe, "provenance": provenance, "records": records, **extra},
                      archive, indent=2, sort_keys=True, allow_nan=False)
            archive.truncate()
            archive.flush()

        checkpoint("running")
        try:
            for repetition in range(3 if args.probe == "kernels" else 1):
                for mode in MODES if repetition % 2 == 0 else reversed(MODES):
                    if args.probe == "responses":
                        preflight = benchmark_memory(experiment(args.experiment, mode), backend="native",
                            enforced_wan=True, max_output_tokens=args.max_output_tokens)
                        provenance["memory_preflight"] = preflight
                        require_admission(preflight)
                        if (preflight["candidates"]["native"]["required_host_bytes"] + (256 << 20)
                                > preflight["host"]["admission_budget_bytes"]):
                            raise BenchmarkMemoryError("response pair exceeds headroom including idle launcher")
                    root = Path(directory) / f"{mode}-{repetition}"
                    root.mkdir()
                    output = root / "report.json"
                    env = dict(os.environ, XDG_CACHE_HOME=str(root), HF_HUB_CACHE=HF_HUB_CACHE,
                        HUGGINGFACE_HUB_CACHE=HF_HUB_CACHE, HF_HUB_OFFLINE="1", PLLM_PREPARED_PATH_SALT=salt)
                    env.pop("PYTHONPATH", None)
                    command = [str(interpreters[mode]), "-m", "scripts.probe_prepared_path", "--probe", args.probe,
                        "--child", mode, "--output", str(output), "--max-output-tokens", str(args.max_output_tokens)]
                    if args.probe == "responses":
                        command += ["--experiment", args.experiment, "--prompt-file", str(args.prompt_file.resolve())]
                    print(f"running {mode} {args.probe} {repetition + 1}", flush=True)
                    child = subprocess.Popen(command, cwd=ROOT, env=env)
                    try:
                        if child.wait(timeout=2400) != 0:
                            raise RuntimeError(f"{mode} child failed")
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
                    records.append({"mode": mode, "repetition": repetition, **json.loads(output.read_bytes())})
                    checkpoint("running")
            identities = [row["identity"] for row in records]
            for key in ("python", "platform", "dependencies"):
                if any(item[key] != identities[0][key] for item in identities):
                    raise ValueError(f"installed builds have different {key}")
            if args.change_scope in {"prepared-refinements", "native-pacing"} and any(
                item["native_sha256"] != identities[0]["native_sha256"] for item in identities
            ):
                raise ValueError(f"{args.change_scope} requires identical native libraries")
            for mode in MODES:
                if len({json.dumps(row["identity"], sort_keys=True) for row in records if row["mode"] == mode}) != 1:
                    raise ValueError("installed implementation changed during cohort")
            changed = sorted(name for name in set(identities[0]["python_sha256"]) | set(identities[1]["python_sha256"])
                if identities[0]["python_sha256"].get(name) != identities[1]["python_sha256"].get(name))
            expected = {
                "kernels": ["runtime/bundle_compression.py"],
                "client-fusion": ["runtime/stage_protocol.py", "runtime/transformer_client.py"],
                "prepared-refinements": ["runtime/client.py", "runtime/transformer_engine.py"],
                "native-pacing": ["runtime/native_wan.py"],
            }[args.change_scope]
            if changed != expected:
                raise ValueError(f"installed Python changes differ from {args.change_scope}: {changed}")
            comparison = None
            if args.probe == "responses":
                comparison = build_comparison_report([(experiment(args.experiment, row["mode"]), row["report"])
                                                      for row in records], compare_kernels=True)
                if not comparison["checks"]["passed"]:
                    raise ValueError("canonical response comparison failed")
            elif len({json.dumps([(case["shape"], case["output_sha256"]) for case in row["report"]["cases"]])
                      for row in records}) != 1:
                raise ValueError("kernel output identities differ across fresh processes")
            checkpoint("complete", comparison=comparison, changed_python_files=changed)
        except BaseException:
            checkpoint("incomplete")
            raise
    print(json.dumps({"output": str(args.output), "status": "complete", "processes": len(records)}))


if __name__ == "__main__":
    main()
