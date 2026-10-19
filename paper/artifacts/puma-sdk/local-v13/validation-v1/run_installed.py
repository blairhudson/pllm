"""Frozen installed-wheel compatibility and SDK gates; generated sources only."""
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
BUILD = ROOT / "target/puma-trie-norm-build-v1"
WHEEL, = (BUILD / "wheels").glob("*.whl")
PACKAGE = Path(pllm.__file__).resolve().parent
assert Path(sys.prefix).resolve() in PACKAGE.parents
assert Path(_native.__file__).resolve().parent == PACKAGE
sys.path.insert(0, str(ROOT))
from scripts.build_paper_release import source_snapshot


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT / name).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


assert json.loads((BUILD / "receipt.json").read_text())["status"] == "passed"
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
        "all_match_wheel": True, "files": actual, "wheel_sha256": sha(WHEEL),
        "package_origin": str(PACKAGE), "native_origin": str(_native.__file__),
        "dependencies": {item.metadata["Name"]: item.version
                         for item in importlib.metadata.distributions()},
    })


tests = [str(path) for path in sorted(Path("tests").glob("test_replicated_*.py"))]
tests += ["tests/test_quality_outcomes.py", "tests/test_quality_outcomes_sdk.py",
          "tests/test_delegated_source.py", "tests/test_delegated_source_streaming.py",
          "tests/test_delegated_sdk.py", "tests/test_delegated_capacity.py",
          "tests/test_reference_agreement.py", "tests/test_reference_benchmark.py",
          "tests/test_profiles.py", "tests/test_component_families.py",
          "tests/test_benchmark_memory.py", "tests/test_benchmark_coverage.py",
          "tests/test_evidence.py", "tests/test_evidence_registry.py",
          "tests/test_search.py", "tests/test_search_benchmark.py",
          "tests/test_quality_locked_network_search.py"]
inputs = {str(path): sha(path) for path in Path("tests").rglob("*.py")}
inputs[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
write("validation-inputs.json", inputs)
before = source_snapshot(ROOT)
assert before["files"] == json.loads((BUILD / "sources-after.json").read_text())["files"]
write("sources-before.json", before)
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
argv = [sys.executable, "-I", "-B", "-c", child, "-x", "-q", "-o", "pythonpath=.",
        "-o", "junit_family=xunit1", *tests, f"--junitxml={OUT / 'installed-tests.xml'}"]
write("command.json", {"argv": argv, "generated_sources_only": True})
environment = dict(os.environ)
environment.pop("PYTHONPATH", None)
environment.pop("PYTHONHOME", None)
environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
started = time.monotonic()
with (OUT / "installed-tests.log").open("x") as log:
    result = subprocess.run(argv, env=environment, stdout=log, stderr=subprocess.STDOUT)
unchanged = all(sha(ROOT / path) == value for path, value in inputs.items())
after = source_snapshot(ROOT)
write("sources-after.json", after)
installed("after")
receipt = {"status": "passed" if result.returncode == 0 and unchanged and before == after else "failed",
           "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - started,
           "inputs_match_before_and_after": unchanged, "source_match_before_and_after": before == after,
           "wheel_sha256": sha(WHEEL), "installed_package_files": len(expected), "test_files": tests}
write("receipt.json", receipt)
print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt["status"] == "passed" else 1)
