"""Additional installed-wheel compatibility checks; generated sources only."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import pllm
from pllm import _native


ROOT = Path.cwd()
OUT = Path(__file__).resolve().parent
WHEEL, = (ROOT / "target/puma-checked-streams-build-v1/wheels").glob("*.whl")
PACKAGE = Path(pllm.__file__).resolve().parent
assert Path(sys.prefix).resolve() in PACKAGE.parents
assert Path(_native.__file__).resolve().parent == PACKAGE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


with zipfile.ZipFile(WHEEL) as archive:
    expected = {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()
                if name.startswith("pllm/") and not name.endswith("/")}


def installed(label):
    actual = {"pllm/" + path.relative_to(PACKAGE).as_posix(): sha(path)
              for path in PACKAGE.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts}
    assert actual == expected
    write(f"installed-{label}.json", {
        "all_match_wheel": True, "files": actual,
        "wheel_sha256": sha(WHEEL), "package_origin": str(PACKAGE),
        "native_origin": str(_native.__file__),
        "dependencies": {item.metadata["Name"]: item.version
                         for item in importlib.metadata.distributions()},
    })


# These seven files have separately retained, non-overlapping receipts.
excluded = {
    "test_replicated_execution_policy.py", "test_replicated_peer.py",
    "test_replicated_peer_handshake.py", "test_replicated_transport.py",
    "test_replicated_policy_quality_failures.py", "test_replicated_policy_sdk.py",
    "test_replicated_decoder_oracle.py",
}
tests = [str(path) for path in sorted(Path("tests").glob("test_replicated_*.py"))
         if path.name not in excluded]
tests += ["tests/test_delegated_source.py", "tests/test_delegated_source_streaming.py",
          "tests/test_delegated_sdk.py", "tests/test_delegated_capacity.py",
          "tests/test_reference_agreement.py", "tests/test_reference_benchmark.py",
          "tests/test_profiles.py", "tests/test_component_families.py",
          "tests/test_benchmark_memory.py", "tests/test_benchmark_coverage.py"]
inputs = {path: sha(ROOT / path) for path in tests}
inputs["tests/conftest.py"] = sha(ROOT / "tests/conftest.py")
write("validation-inputs.json", inputs)
installed("before")

child = r"""
import pathlib, sys
import pllm
from pllm import _native
import pytest
package = pathlib.Path(pllm.__file__).resolve().parent
assert pathlib.Path(sys.prefix).resolve() in package.parents
class InstalledOrigins:
    def pytest_sessionstart(self, session):
        assert pathlib.Path(pllm.__file__).resolve().parent == package
        assert pathlib.Path(_native.__file__).resolve().parent == package
    def pytest_sessionfinish(self, session, exitstatus):
        for name, module in tuple(sys.modules.items()):
            origin = getattr(module, '__file__', None)
            if origin and (name == 'pllm' or name.startswith('pllm.')):
                assert pathlib.Path(origin).resolve().is_relative_to(package), (name, origin)
raise SystemExit(pytest.main(sys.argv[1:], plugins=[InstalledOrigins()]))
"""
argv = [sys.executable, "-I", "-c", child, "-x", "-q", "-o", "pythonpath=.",
        "-o", "junit_family=xunit1", *tests, f"--junitxml={OUT / 'installed-tests.xml'}"]
write("command.json", {"argv": argv, "excluded_separate_receipts": sorted(excluded)})
environment = dict(os.environ)
environment.pop("PYTHONPATH", None)
environment.pop("PYTHONHOME", None)
started = time.monotonic()
with (OUT / "installed-tests.log").open("x") as log:
    result = subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
unchanged = all(sha(ROOT / path) == value for path, value in inputs.items())
installed("after")
receipt = {"status": "passed" if result.returncode == 0 and unchanged else "failed",
           "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
           "inputs_match_before_and_after": unchanged, "wheel_sha256": sha(WHEEL),
           "installed_package_files": len(expected), "test_files": tests}
write("receipt.json", receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt["status"] == "passed" else 1)
