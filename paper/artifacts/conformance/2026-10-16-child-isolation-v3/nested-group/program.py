import os,signal,subprocess,sys,time
from pathlib import Path
def leave(signum, frame):
    raise SystemExit(17)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
os.kill(os.getpid(), signal.SIGTERM)
signal.signal(signal.SIGTERM, leave)
child = subprocess.Popen([sys.executable, '-c', "import time; from pathlib import Path; Path('/Users/blairhudson/workspace/pllm/paper/artifacts/conformance/2026-10-16-child-isolation-v3/nested-group/inner-ready').write_text('ready'); time.sleep(60)"])
try:
    deadline = time.monotonic() + 5
    while not Path('/Users/blairhudson/workspace/pllm/paper/artifacts/conformance/2026-10-16-child-isolation-v3/nested-group/inner-ready').exists():
        assert time.monotonic() < deadline and child.poll() is None
        time.sleep(0.01)
    Path('/Users/blairhudson/workspace/pllm/paper/artifacts/conformance/2026-10-16-child-isolation-v3/nested-group/outer-ready').write_text(str(child.pid))
    time.sleep(60)
finally:
    child.wait(timeout=4)
