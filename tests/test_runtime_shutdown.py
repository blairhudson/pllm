"""Real stubborn children exercise deadlines, concurrent owners and signals."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups and signals")
@pytest.mark.parametrize("interrupt", [False, True])
def test_topology_shared_deadline_survives_concurrent_close_and_signals(interrupt):
    import pllm

    # Run the supervisor itself out of process: pytest must never receive these
    # signals, and both roles deliberately refuse graceful termination.
    script = r'''
import json, signal, subprocess, sys, threading
from pllm.runtime.servers import LocalTopology, build_roles

child = "import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); signal.pause()"
topology = build_roles("unused/model")
for role in ("inference", "preparation"):
    process = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, start_new_session=True)
    assert process.stdout.readline() == b"ready\n"
    topology._processes[role] = process
print(json.dumps({"pids": [p.pid for p in topology._processes.values()]}), flush=True)
sys.stdin.readline()
old = LocalTopology._stop_all
waiters, failures = [], []
def concurrent_close():
    try:
        topology.close()
    except BaseException as exc:
        failures.append(type(exc).__name__)
def stop_all(processes):
    for _ in range(2):
        waiter = threading.Thread(target=concurrent_close)
        waiters.append(waiter)
        waiter.start()
    print("closing", flush=True)
    old(processes)
LocalTopology._stop_all = staticmethod(stop_all)
def interrupt(signum, frame):
    raise KeyboardInterrupt
signal.signal(signal.SIGTERM, interrupt)
previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
interrupted = False
try:
    topology.close()
except KeyboardInterrupt:
    interrupted = True
for waiter in waiters:
    waiter.join(timeout=1)
assert all(not waiter.is_alive() for waiter in waiters)
assert not failures, failures
assert all(signal.getsignal(sig) == handler for sig, handler in previous.items())
assert all(process.poll() == -signal.SIGKILL for process in topology._processes.values())
topology.close()
print(json.dumps({"interrupted": interrupted, "closed": topology.closed}), flush=True)
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True,
        env={**os.environ, "PYTHONPATH": str(Path(pllm.__file__).parent.parent)},
    )
    pids = []
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            assert ready.select(timeout=15), "supervisor did not start"
            pids = json.loads(process.stdout.readline())["pids"]
            start = time.monotonic()
            process.stdin.write("close\n")
            process.stdin.flush()
            assert ready.select(timeout=5), "cleanup did not start"
            assert process.stdout.readline().strip() == "closing"
        if interrupt:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGINT):
                os.kill(process.pid, sig)
                time.sleep(0.02)
        output, error = process.communicate(timeout=9)
        assert process.returncode == 0, error
        assert json.loads(output) == {"interrupted": interrupt, "closed": True}
        assert time.monotonic() - start < 10
        assert all(not psutil.pid_exists(pid) for pid in pids)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        for pid in pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
