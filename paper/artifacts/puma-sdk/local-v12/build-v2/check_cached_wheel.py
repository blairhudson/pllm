"""Guard one cached offline wheel build; not a clean/release attestation."""

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_paper_release import source_snapshot
from scripts.host_swap import swap_snapshot, validate_counter_identity


def build(out):
    out.mkdir(parents=True, exist_ok=False)
    observer_paths = (Path(__file__), ROOT / "scripts/host_swap.py", ROOT / "scripts/build_paper_release.py")
    observer_hashes = {}
    for path in observer_paths:
        shutil.copyfile(path, out / path.name)
        observer_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def write(name, value):
        with (out / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")

    reserve, limit = 4 << 30, 4 << 30
    initial = swap_snapshot()
    minimum_used = initial["swap_used_bytes"]
    available = psutil.virtual_memory().available
    write("host-before.json", {"available_bytes": available, "swap": initial})
    write("observer-hashes.json", observer_hashes)
    if available < reserve + limit:
        write("receipt.json", {"status": "blocked", "reason": "build_headroom",
              "available_bytes": available, "owner_cap_bytes": limit, "reserve_bytes": reserve})
        return 1
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
        "timeout_seconds": 1200, "additional_swap_bytes": 0},
        "swap_counter_source": initial["counter_source"]})
    start = time.monotonic()
    reason, observation_error = None, None
    samples, cleanup_errors = [], []
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
                current = swap_snapshot()
                validate_counter_identity(current, initial)
                minimum_used = min(minimum_used, current["swap_used_bytes"])
                available = psutil.virtual_memory().available
                samples.append({"elapsed_seconds": time.monotonic() - start,
                    "available_bytes": available, "owned_rss_bytes": owned,
                    "swap_used_growth_bytes": max(0, current["swap_used_bytes"] - minimum_used),
                    "swapout_growth_bytes": current["swapout_bytes"] - initial["swapout_bytes"],
                    "swap": current})
                if available < reserve:
                    reason = "host_reserve"
                elif owned > limit:
                    reason = "owned_build_rss"
                elif current["swap_used_bytes"] > minimum_used or current["swapout_bytes"] > initial["swapout_bytes"]:
                    reason = "host_swap_growth"
                elif time.monotonic() - start > 1200:
                    reason = "timeout"
                if reason or process.poll() is not None:
                    break
                time.sleep(0.25)
        except BaseException as exc:
            reason = "observation_or_monitor_failure"
            observation_error = type(exc).__name__
        finally:
            write("monitor.json", samples)
            write("stop.json", {"reason": reason, "exit_code_before_cleanup": process.poll(),
                                "pid": process.pid, "observation_error": observation_error})
            if process.poll() is None:
                try:
                    owner = psutil.Process(process.pid)
                    owned_processes = [owner, *owner.children(recursive=True)]
                except psutil.Error:
                    owned_processes = []
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                except OSError as error:
                    cleanup_errors.append({"phase": "term", "type": type(error).__name__, "errno": error.errno})
                finally:
                    _, survivors = psutil.wait_procs(owned_processes, timeout=1)
                    if survivors or process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except OSError as error:
                            cleanup_errors.append({"phase": "kill", "type": type(error).__name__, "errno": error.errno})
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        cleanup_errors.append({"phase": "join", "type": "TimeoutExpired"})
                    _, survivors = psutil.wait_procs(owned_processes, timeout=1)
                    if survivors:
                        cleanup_errors.append({"phase": "survivors", "pids": [p.pid for p in survivors]})
    after = source_snapshot(ROOT)
    write("sources-after.json", after)
    observers_changed = observer_hashes != {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in observer_paths}
    wheels = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in (out / "wheels").glob("*.whl")}
    receipt = {"status": "passed" if process.returncode == 0 and reason is None and not cleanup_errors
               and not observers_changed and before == after and len(wheels) == 1 else "failed",
               "exit_code": process.returncode, "stop_reason": reason,
               "cleanup_errors": cleanup_errors, "observers_changed": observers_changed,
               "source_changed": before != after, "wheel_sha256": wheels,
               "cached_build": True, "elapsed_seconds": time.monotonic() - start,
               "maximum_sampled_owned_rss_bytes": max((row["owned_rss_bytes"] for row in samples), default=None),
               "max_swap_used_growth_bytes": max((row["swap_used_growth_bytes"] for row in samples), default=None),
               "max_swapout_growth_bytes": max((row["swapout_growth_bytes"] for row in samples), default=None),
               "swap_counter_source": initial["counter_source"],
               "scope": "sampled build process tree; observer/final receipt tail excluded; cached local build only"}
    write("receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    raise SystemExit(build(ROOT / args.output))
