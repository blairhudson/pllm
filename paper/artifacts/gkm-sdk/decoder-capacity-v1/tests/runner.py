"""Verify installed wheel bytes and run isolated GKM capacity regressions."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / sys.argv[1]
OUT = BASE / sys.argv[2] if len(sys.argv) > 2 else BASE
if OUT != BASE:
    OUT.mkdir()
PYTHON = BASE / "venv/bin/python"
wheel, = (BASE / "wheels").glob("*.whl")
site, = (BASE / "venv/lib").glob("python*/site-packages")


def write(name, value):
    with (OUT / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def installed():
    records = {}
    with zipfile.ZipFile(wheel) as packed:
        for name in sorted(packed.namelist()):
            if not name.startswith("pllm/") or name.endswith("/"):
                continue
            expected = packed.read(name)
            actual = (site / name).read_bytes()
            if actual != expected:
                raise ValueError("installed bytes differ: " + name)
            records[name] = {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()}
    return {"wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(), "files": records}


paths = set(json.loads((ROOT / "target/research/gkm-source-stream-wheel-v1/test-sources-before.json").read_text()))
paths.add("tests/test_delegated_capacity.py")
paths.update({"tests/test_delegated_result_lifetimes.py", "tests/test_delegated_worker_lifetimes.py"})


def snapshot():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sorted(paths)}


argv = json.loads((ROOT / "target/research/gkm-source-stream-wheel-v1/test-command.json").read_text())
argv[0] = str(PYTHON)
argv.insert(argv.index("-q"), "tests/test_delegated_capacity.py")
argv[argv.index("-q"):argv.index("-q")] = [
    "tests/test_delegated_result_lifetimes.py", "tests/test_delegated_worker_lifetimes.py"]
argv[-1] = "--junitxml=" + str(OUT / "installed-tests.xml")
write("test-runner.json", {"argv": sys.argv, "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
(OUT / "test-runner.py").write_bytes(Path(__file__).read_bytes())
write("test-command.json", argv)
before = snapshot()
write("test-sources-before.json", before)
precheck = installed()
write("installed-precheck.json", precheck)
env = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
with (OUT / "installed-tests.log").open("x") as handle:
    code = subprocess.run(argv, cwd=ROOT, env=env, stdout=handle,
                          stderr=subprocess.STDOUT, check=False).returncode
after = snapshot()
write("test-sources-after.json", after)
postcheck = installed()
write("installed-postcheck.json", postcheck)
if precheck != postcheck:
    raise ValueError("installed package changed during testing")
receipt = {"exit_code": code, "test_sources_changed": before != after}
write("test-receipt.json", receipt)
print(json.dumps({**receipt, "installed_files": len(postcheck["files"])}, indent=2))
raise SystemExit(code or int(before != after))
