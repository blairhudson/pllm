"""Local cached-build guard; no release or clean-build attestation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
from scripts.build_paper_release import source_snapshot

out = ROOT / sys.argv[1]
out.mkdir(parents=True, exist_ok=False)
shutil.copyfile(__file__, out / "guard.py")


def write(name, value):
    with (out / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


reserve = 4 << 30
limit = 4 << 30
initial_swap = psutil.swap_memory()
available = psutil.virtual_memory().available
if available < reserve + limit:
    write("receipt.json", {"status": "blocked", "reason": "build_headroom",
          "available_bytes": available, "owner_cap_bytes": limit, "reserve_bytes": reserve})
    raise SystemExit("Insufficient headroom for bounded cached build")
before = source_snapshot(ROOT)
write("sources-before.json", before)
argv = [str(ROOT / ".venv/bin/python"), "-I", "-B", "-m", "maturin", "build",
        "--release", "--locked", "--offline", "--interpreter",
        str(ROOT / ".venv/bin/python"), "--out", str(out / "wheels")]
env = {key: value for key, value in os.environ.items() if key not in
       {"PYTHONPATH", "PYTHONHOME", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}}
env.update(CARGO_BUILD_JOBS="2", CARGO_NET_OFFLINE="true", RAYON_NUM_THREADS="1",
           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
           CARGO_TERM_COLOR="never")
write("command.json", {"argv": argv, "cached_target": True, "limits": {
    "jobs": 2, "owner_rss_stop_bytes": limit, "host_reserve_bytes": reserve,
    "timeout_seconds": 1200, "additional_swap_bytes": 0}})
start = time.monotonic()
reason = None
samples = []
with (out / "build.log").open("x") as log:
    process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while True:
            owned = 0
            try:
                owner = psutil.Process(process.pid)
                for child in [owner, *owner.children(recursive=True)]:
                    try:
                        owned += child.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
            current = psutil.swap_memory()
            available = psutil.virtual_memory().available
            samples.append({"elapsed_seconds": time.monotonic() - start,
                "available_bytes": available, "owned_rss_bytes": owned,
                "swap_used_growth_bytes": max(0, current.used - initial_swap.used),
                "swapout_growth_bytes": max(0, current.sout - initial_swap.sout)})
            if available < reserve:
                reason = "host_reserve"
            elif owned > limit:
                reason = "owned_build_rss"
            elif current.used > initial_swap.used or current.sout > initial_swap.sout:
                reason = "host_swap_growth"
            elif time.monotonic() - start > 1200:
                reason = "timeout"
            if reason or process.poll() is not None:
                break
            time.sleep(0.25)
    finally:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
write("monitor.json", samples)
after = source_snapshot(ROOT)
write("sources-after.json", after)
wheels = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
          for path in (out / "wheels").glob("*.whl")}
receipt = {"status": "passed" if process.returncode == 0 and reason is None
           and before == after and len(wheels) == 1 else "failed",
           "exit_code": process.returncode, "stop_reason": reason,
           "source_changed": before != after, "wheel_sha256": wheels,
           "cached_build": True, "elapsed_seconds": time.monotonic() - start,
           "maximum_sampled_owned_rss_bytes": max(row["owned_rss_bytes"] for row in samples),
           "max_swap_used_growth_bytes": max(row["swap_used_growth_bytes"] for row in samples),
           "max_swapout_growth_bytes": max(row["swapout_growth_bytes"] for row in samples)}
write("receipt.json", receipt)
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if receipt["status"] == "passed" else 1)
