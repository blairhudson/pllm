"""Fresh-process research helpers must retire detached workers on cancellation."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil
import pytest


def test_probe_slot_rejects_concurrent_cohort_and_releases_on_error(tmp_path, monkeypatch):
    helpers = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(helpers))
    monkeypatch.setenv("HOME", str(tmp_path))
    from frontier_probe_support import fresh_workers

    code = f'''import sys
sys.path.insert(0, {str(helpers)!r})
from frontier_probe_support import fresh_workers
with fresh_workers(__file__, memory_bytes=0):
    print("incorrect concurrent admission")
'''.replace("__file__", repr(str(tmp_path / "unused.py")))
    with pytest.raises(ValueError, match="cohort failure"):
        with fresh_workers(tmp_path / "unused.py", memory_bytes=0):
            contender = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
            assert contender.returncode != 0
            assert "another frontier cohort" in contender.stderr
            assert not contender.stdout
            raise ValueError("cohort failure")
    with fresh_workers(tmp_path / "unused.py", memory_bytes=0):
        pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX detached-process cancellation")
@pytest.mark.parametrize("termination", [signal.SIGINT, signal.SIGTERM])
def test_probe_cancellation_retires_owned_worker(tmp_path, termination):
    script = tmp_path / "worker.py"
    marker = tmp_path / "ready"
    helpers = Path(__file__).resolve().parents[1] / "scripts"
    script.write_text(f'''import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {str(helpers)!r})
if "--worker" in sys.argv:
    spec = json.load(sys.stdin)
    Path(spec["marker"]).write_text(str(os.getpid()))
    time.sleep(120)
else:
    from frontier_probe_support import fresh_workers
    with fresh_workers(__file__, memory_bytes=0) as (run, guard, host):
        run({{"marker": {str(marker)!r}}})
''')
    child = None
    with (tmp_path / "output").open("w") as output:
        parent = subprocess.Popen([sys.executable, str(script)], stdout=output, stderr=output)
        try:
            deadline = time.monotonic() + 30
            while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(0.025)
            assert marker.exists(), (tmp_path / "output").read_text()
            child = int(marker.read_text())
            parent.send_signal(termination)
            parent.wait(timeout=10)
            assert parent.returncode != 0
            assert not psutil.pid_exists(child)
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait()
            if child is not None and psutil.pid_exists(child):
                os.kill(child, signal.SIGKILL)
