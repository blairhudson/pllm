"""Freeze the original capacity receipts and executed inputs without rewriting them."""

import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "target/research/gkm-capacity-wheel-v3"
NATIVE = ROOT / "target/research/gkm-capacity-validation-v1/native-v4"
OUT = ROOT / "paper/artifacts/gkm-sdk/decoder-capacity-v1"
OUT.mkdir(exist_ok=False)


def copy(source, name):
    destination = OUT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


build_names = (
    "receipt.json", "sources-before.json", "sources-after.json", "observer-hashes.json",
    "build_paper_release.py", "check_cached_wheel.py", "host_swap.py", "command.json",
    "host-before.json", "monitor.json", "stop.json", "build.log", "install.log",
    "requirements.txt", "install-commands.json", "installed-precheck.json", "installed-postcheck.json",
)
for name in build_names:
    copy(BUILD / name, "build/" + name)
for source, name in {
    "test-command.json": "command.json", "test-receipt.json": "receipt.json",
    "test-sources-before.json": "sources-before.json", "test-sources-after.json": "sources-after.json",
    "installed-tests.log": "results.log", "installed-tests.xml": "results.xml",
    "test-runner.json": "runner.json", "test-runner.py": "runner.py",
}.items():
    copy(BUILD / source, "tests/" + name)
for path in NATIVE.iterdir():
    if path.is_file():
        copy(path, "native/" + path.name)

sources = json.loads((BUILD / "sources-before.json").read_text())["files"]
tests = json.loads((BUILD / "test-sources-before.json").read_text())
selected = {
    name for name in sources if name.startswith((
        "crates/pllm-core/src/gkm/", "crates/pllm-compiler/src/delegated_decoder",
        "crates/pllm-compiler/tests/delegated_", "crates/pllm-python/src/delegated_",
        "crates/pllm-core/tests/gkm_", "python/pllm/runtime/delegated_",
        "python/pllm/runtime/benchmark_delegated", "python/pllm/runtime/benchmark_memory",
    ))
}
selected.update({"python/pllm/protocols/__init__.py", "python/pllm/protocols/__init__.pyi",
                 "python/pllm/profiles/__init__.py"})
for name in sorted(selected | set(tests)):
    expected = tests.get(name, sources.get(name))
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
        raise ValueError("executed source changed: " + name)
    copy(ROOT / name, "sources/" + name)

# Preserve the failed installed attempts and original metadata-only smoke attempt.
for version in ("v1", "v2"):
    previous = ROOT / ("target/research/gkm-capacity-wheel-" + version)
    for name in ("receipt.json", "test-receipt.json", "test-sources-before.json",
                 "test-sources-after.json", "test-command.json", "installed-tests.log",
                 "installed-tests.xml"):
        copy(previous / name, "rejected-wheel-" + version + "/" + name)
for version in ("v1", "v2"):
    previous = ROOT / ("target/research/gkm-qwen-capacity-smoke-" + version)
    for path in previous.iterdir():
        if path.is_file():
            copy(path, "qwen-smoke-" + version + "/" + path.name)
copy(Path(__file__), "archive.py")

groups = {}
commands = json.loads((NATIVE / "commands.json").read_text())
for name in commands:
    if name != "clippy":
        groups[name] = [int(n) for n in re.findall(
            r"test result: ok\. (\d+) passed; 0 failed;", (NATIVE / (name + ".log")).read_text())]
files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size,
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
         for p in sorted(OUT.rglob("*")) if p.is_file()]
assert len(files) <= 256 and sum(row["bytes"] for row in files) <= 32 << 20
wheel, = (BUILD / "wheels").glob("*.whl")
index = {"schema": "pllm.gkm_sdk_archive.v3", "files": files,
         "installed_tests_passed": 640, "native_test_groups": groups,
         "retained_wheel": wheel.relative_to(ROOT).as_posix()}
with (OUT / "index.json").open("x") as stream:
    json.dump(index, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"files": len(files), "bytes": sum(row["bytes"] for row in files),
                  "native_checks": sum(map(sum, groups.values()))}, indent=2))
