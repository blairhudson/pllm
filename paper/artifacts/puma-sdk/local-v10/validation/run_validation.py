"""Installed-wheel conformance receipt using the ordinary SDK source path."""
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pllm
from pllm.deployment.resources import reserve_incremental_memory
from pllm.metrics import benchmark_memory
from pllm.runtime.benchmark_memory import MemoryWatchdog, host_memory, require_admission
from pllm.runtime.replicated_protocol import REQUEST_MEMORY
from pllm.runtime.replicated_source import admit_source

root = Path.cwd()
out = Path(__file__).resolve().parent
build = root / "target/puma-attention-range-build-v1"
wheel, = (build / "wheels").glob("*.whl")
package = Path(pllm.__file__).resolve().parent
assert Path(sys.prefix).resolve() in package.parents
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
with zipfile.ZipFile(wheel) as archive:
    expected = {n: hashlib.sha256(archive.read(n)).hexdigest() for n in archive.namelist()
                if n.startswith("pllm/") and not n.endswith("/")}


def write(name, value):
    with (out / name).open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def installed(label):
    actual = {"pllm/" + p.relative_to(package).as_posix(): sha(p)
              for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert actual == expected
    write(f"installed-files-{label}.json", {
        "files": actual, "all_match_wheel": True, "wheel_sha256": sha(wheel),
        "package_origin": str(package),
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    })


inputs = json.loads((build / "sources-before.json").read_text())["files"]
assert all(sha(root / p) == h for p, h in inputs.items())
for p in json.loads((root / "paper/artifacts/puma-sdk/local-v8/validation-inputs.json").read_text()):
    if p not in inputs and (root / p).is_file():
        inputs[p] = sha(root / p)
write("validation-inputs.json", inputs)
installed("before")
host = dataclasses.asdict(host_memory())
write("host-before.json", host)
assert host["pressure"] == "normal" and host["available"] > 4 * 1024**3, host


def checkpoint_admission():
    saved = json.loads((root / "paper/artifacts/puma-sdk/local-v7/qwen-memory-preflight.json").read_text())
    experiment = pllm.Experiment.from_spec(saved["experiment"])
    report = benchmark_memory(experiment, backend="native")
    require_admission(report)
    cancelled = threading.Event()
    guard = MemoryWatchdog(report, lambda reason: cancelled.set())
    owner = None
    started, cpu = time.monotonic(), time.process_time()
    result = {
        "scope": "ordinary SDK source publisher; four fixed public zero IDs; no role workers or online execution; process peak includes validation launcher",
        "memory_preflight": report, "wheel_sha256": sha(wheel),
    }
    guard.start()
    try:
        with reserve_incremental_memory(REQUEST_MEMORY):
            source = admit_source(experiment)
            result.update(source_lock=source.source_lock_digest, checkpoint=source.checkpoint_digest)
            owner, manifest = source.publisher(
                experiment, (0, 0, 0, 0), 2,
                lambda: cancelled.is_set() or time.monotonic() - started > 600,
            )
            result.update(status="program-admitted", program=json.loads(manifest)["program"])
    except ValueError as error:
        result.update(status="rejected", error_type=type(error).__name__, reason=str(error))
    finally:
        if owner is not None:
            owner.close()
        guard.close()
        result.update(
            elapsed_seconds=time.monotonic() - started,
            process_cpu_seconds=time.process_time() - cpu,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            watchdog_error=guard.error,
            max_swap_used_growth_bytes=guard.maximum_swap_growth_bytes,
            host_after=dataclasses.asdict(host_memory()),
        )
        write("qwen-program-admission.json", result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"host_after", "memory_preflight"}}), flush=True)
    assert result["watchdog_error"] is None
    assert result["source_lock"] == "383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29"
    assert result["checkpoint"] == "f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd"
    assert result["status"] == "program-admitted" or "replicated decoder admission" in result["reason"]


tests = [*map(str, sorted(Path("tests").glob("test_replicated_*.py"))),
         "tests/test_delegated_source.py", "tests/test_delegated_source_streaming.py",
         "tests/test_delegated_sdk.py", "tests/test_delegated_capacity.py",
         "tests/test_reference_agreement.py", "tests/test_reference_benchmark.py",
         "tests/test_profiles.py", "tests/test_component_families.py",
         "tests/test_benchmark_memory.py", "tests/test_benchmark_coverage.py"]
for p in tests:
    assert inputs[p] == sha(root / p)
commands = [
    [sys.executable, "-I", "-m", "pytest", "-x", "-q", "-o", "pythonpath=.", "-o", "junit_family=xunit1",
     *tests, f"--junitxml={out / 'installed-tests.xml'}"],
    ["cargo", "test", "--locked", "--offline", "-p", "pllm-core", "--lib", "puma", "--", "--test-threads=1"],
    ["cargo", "test", "--locked", "--offline", "-p", "pllm-compiler", "--lib", "replicated_decoder", "--", "--test-threads=1"],
    ["cargo", "test", "--locked", "--offline", "-p", "pllm-core", "--test", "puma_allocation", "--test", "puma_multiply",
     "--test", "puma_prss", "--test", "puma_truncation", "--", "--test-threads=1"],
    ["cargo", "test", "--locked", "--offline", "-p", "pllm-core", "--doc", "puma", "--", "--test-threads=1"],
    ["cargo", "clippy", "--locked", "--offline", "-p", "pllm-core", "-p", "pllm-compiler", "-p", "pllm-python",
     "--lib", "--", "-D", "warnings", "-A", "clippy::needless_range_loop", "-A", "clippy::nonminimal_bool"],
    [sys.executable, "-I", "scripts/benchmark_coverage.py", "--check"],
]
results = []
try:
    checkpoint_admission()
    for i, argv in enumerate(commands):
        with (out / f"validation-{i}.log").open("x") as log:
            result = subprocess.run(argv, env={**os.environ, "CARGO_BUILD_JOBS": "1"}, stdout=log, stderr=subprocess.STDOUT)
        results.append({"argv": argv, "returncode": result.returncode, "log": f"validation-{i}.log"})
        print(json.dumps(results[-1]), flush=True)
        if result.returncode:
            break
finally:
    unchanged = all(sha(root / p) == h for p, h in inputs.items())
    installed("after")
    write("host-after.json", dataclasses.asdict(host_memory()))
    status = "passed" if unchanged and len(results) == len(commands) and all(r["returncode"] == 0 for r in results) else "failed"
    receipt = {"status": status, "commands": results, "inputs_match_before_and_after": unchanged,
               "installed_file_count": len(expected), "wheel_sha256": sha(wheel),
                "build_receipt": "../puma-attention-range-build-v1/receipt.json"}
    write("validation-commands.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)
raise SystemExit(0 if status == "passed" else 1)
