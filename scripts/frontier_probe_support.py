"""Serial fresh-process research runner with shared host-pressure admission."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


@contextmanager
def fresh_workers(script, *, memory_bytes, timeout=240):
    from filelock import FileLock, Timeout
    path = Path.home() / ".cache" / "pllm" / "frontier-probe.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path), timeout=0)
    try:
        lock.acquire()
    except Timeout as error:
        raise RuntimeError("another frontier cohort owns this user's measurement slot") from error
    try:
        with _fresh_workers(script, memory_bytes=memory_bytes, timeout=timeout) as workers:
            yield workers
    finally:
        lock.release()


@contextmanager
def _fresh_workers(script, *, memory_bytes, timeout):
    from pllm.metrics import BuildRef
    from pllm.runtime.benchmark_memory import MemoryWatchdog, host_memory
    from pllm.runtime.shutdown import defer_termination_signals
    build = BuildRef.capture()
    host = host_memory()
    if host.available - host.reserve < memory_bytes:
        raise RuntimeError("frontier probe exceeds post-reserve headroom")
    active = []

    def signal_group(process, sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    def abort(_reason):
        for process in tuple(active):
            if process.poll() is None:
                signal_group(process, signal.SIGTERM)

    guard = MemoryWatchdog({"host": asdict(host) | {"reserve_bytes": host.reserve}}, abort)

    def run(payload):
        guard.check()
        if guard.error:
            raise RuntimeError(guard.error)
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS"):
            env[name] = "4"
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        process = None
        try:
            with defer_termination_signals():
                process = subprocess.Popen([sys.executable, str(script), "--worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                    start_new_session=os.name == "posix")
                active.append(process)
            out, error = process.communicate(json.dumps(payload), timeout=timeout)
        finally:
            with defer_termination_signals():
                if process is not None:
                    if process.poll() is None:
                        signal_group(process, signal.SIGKILL)
                    process.wait()
                    active.remove(process)
        guard.check()
        if process.returncode or guard.error:
            raise RuntimeError(guard.error or error)
        return json.loads(out)

    run.build = build.to_spec()
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        guard.start()
        yield run, guard, host
        if BuildRef.capture() != build:
            raise RuntimeError("frontier build changed during the cohort")
    finally:
        try:
            with defer_termination_signals():
                abort("cleanup")
                guard.close()
        finally:
            signal.signal(signal.SIGTERM, previous)
