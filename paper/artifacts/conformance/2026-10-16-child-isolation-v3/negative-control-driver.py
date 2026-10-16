"""Deliberately failed identity scenarios, separate from positive receipts."""
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time

root, mode = Path(__file__).parent, sys.argv[1]
directory = root / ("negative-" + mode)
policy = runpy.run_path(str(root / "python-child-isolation.py"))
policy["activate"](directory, root / "provenance.json")
ready = directory / "ready"
code = "import os,time,subprocess; from pathlib import Path; "
if mode == "timeout":
    code += "g=subprocess.Popen.__init__.__globals__; old=g['identity']; g['identity']=lambda expected: (time.sleep(5),old(expected))[1]; "
if mode == "end-mismatch":
    code += f"import pllm._native; pllm._native.__file__={str(root / 'shadow/pllm/__init__.py')!r}"
else:
    code += f"Path({str(ready)!r}).write_text('ready'); time.sleep(60)"
argv = [sys.executable, "-I", "-c", code]
with (directory / "child.log").open("w") as stream:
    process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    started = time.monotonic()
    try:
        if mode != "end-mismatch":
            deadline = time.monotonic() + 8
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("negative control did not reach its ready point")
                time.sleep(0.01)
            started = time.monotonic()
            os.killpg(process.pid, signal.SIGKILL if mode == "sigkill" else signal.SIGTERM)
        returncode = process.wait(timeout=8)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
elapsed = time.monotonic() - started
try:
    os.killpg(process.pid, 0)
except ProcessLookupError:
    group_absent = True
else:
    group_absent = False
summary = policy["finish"](directory)
control = {"mode": mode, "returncode": returncode, "signal_to_exit_seconds": elapsed,
           "process_group_absent": group_absent, "identity_gate_status": summary["status"],
           "captured_python_processes": summary["captured_python_processes"],
           "argv": argv, "scope": "intentional negative; never a successful identity execution"}
(directory / "control.json").write_text(json.dumps(control, indent=2, sort_keys=True) + "\n")
assert summary["status"] == "failed" and group_absent
if mode in {"sigkill", "timeout"}:
    assert not list(directory.glob("*.end.json"))
    assert returncode == (-signal.SIGKILL if mode == "sigkill" else -signal.SIGTERM)
    assert any(".end receipt" in error for error in summary["failures"])
    if mode == "timeout":
        assert elapsed < 5 and len(list(directory.glob("*.end-error.json"))) == 1
else:
    assert returncode == 0
    ending = json.loads(next(directory.glob("*.end.json")).read_text())
    assert ending["status"] == "failed" and "loaded native binary differs" in ending["errors"]
print(json.dumps(control))
