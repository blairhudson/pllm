"""Collect original completed receipts; no tests or model execution."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path.cwd()
OUT = ROOT / "paper/artifacts/puma-sdk/local-v12"
TARGET = ROOT / "target"
SHELL = Path.home() / ".local/share/opencode/shell/0a18c7b41740937a82f9ef5b5d3cf4fd8e0ca47b"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_tree(source, dest):
    for path in source.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".whl":
            target = dest / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


for version in (1, 2):
    copy_tree(TARGET / f"puma-checked-streams-build-v{version}", OUT / f"build-v{version}")
copy_tree(TARGET / "puma-checked-streams-compat-v1", OUT / "compat")
copy_tree(TARGET / "puma-checked-streams-admission-v1", OUT / "checkpoint")
copy_tree(TARGET / "puma-oracle-distinct-gqa-v1", OUT / "oracle")
copy_tree(TARGET / "puma-source-delivery-next", OUT / "source-lifecycle")
write(OUT / "source-lifecycle/results.json", json.loads(subprocess.check_output(
    [sys.executable, "-I", str(OUT / "source-lifecycle/calculations.py")], text=True)))
sdk = TARGET / "puma-checked-streams-sdk-v1/run-20261019T084418283007Z"
copy_tree(sdk / "installed-fixture-fixed-v2", OUT / "sdk")
for path in sdk.iterdir():
    if path.name == "installed-fixture-fixed-v2":
        continue
    if path.is_dir():
        copy_tree(path, OUT / "sdk-attempts" / path.name)
    else:
        (OUT / "sdk-attempts").mkdir(exist_ok=True)
        shutil.copy2(path, OUT / "sdk-attempts" / path.name)

commands = [
    ("core", "sh_1535fc9f9001njR5M0Tu2yl2XI", 0,
     "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --lib puma -- --test-threads=1"),
    ("integration", "sh_1536121140017YTkDbCFJJYpQG", 0,
     "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --test puma_allocation --test puma_multiply --test puma_prss --test puma_truncation -- --test-threads=1"),
    ("compiler", "sh_1538dd120001qUbnJaFwDTLsVK", 0,
     "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-compiler --lib replicated_decoder -- --test-threads=1 --skip native_generated_capture_has_complete_checkpoints_and_actual_truncation_events --skip native_multilayer_gqa_guard_tiles_and_tensor_tails --skip native_full_vocabulary_narrow_fixture_admission_and_audit"),
    ("doctests", "sh_15381c191001aa5y7H40WYy3dU", 0,
     "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --doc puma -- --test-threads=1"),
    ("clippy", "sh_15371341b001gRoasky67cVs7A", 0,
     "CARGO_BUILD_JOBS=1 cargo clippy --locked --offline -p pllm-core -p pllm-compiler -p pllm-python --lib -- -D warnings -A clippy::needless_range_loop -A clippy::nonminimal_bool"),
    ("clippy-strict-prior-failure", "sh_1536f7ade001D0Sd1HFQoTBwt4", 101,
     "CARGO_BUILD_JOBS=1 cargo clippy --locked --offline -p pllm-core --all-targets -- -D warnings"),
    ("coverage", "sh_15381c195001Cs8NvIxaKzEt7D", 0,
     "target/puma-sdk-integration-env-v1/bin/python -I scripts/benchmark_coverage.py --check"),
]
(OUT / "native").mkdir(exist_ok=True)
excerpts = json.loads((TARGET / "puma-native-session-excerpts.json").read_text())
write(OUT / "native/session-excerpts.json", excerpts)
record_kinds = {}
for name, shell_id, code, command in commands:
    original = SHELL / f"{shell_id}.out"
    if original.is_file():
        shutil.copy2(original, OUT / "native" / f"{name}.log")
        record_kinds[name] = "original_stdout"
    else:
        record = excerpts["records"][name]
        assert record["original_shell_id"] == shell_id and record["exit_code"] == code
        (OUT / "native" / f"{name}.log").write_text(
            "TRANSCRIPT EXCERPT — full original stdout not retained; see session-excerpts.json\n"
            + "\n".join(record["stdout_excerpts"]) + "\n")
        record_kinds[name] = "explicit_session_excerpt"
write(OUT / "native/commands.json", [
    {"name": name, "command": command, "exit_code": code,
     "original_shell_id": shell_id, "log": f"{name}.log", "record_kind": record_kinds[name]}
    for name, shell_id, code, command in commands
])

wheel_payloads = []
for version in (1, 2):
    wheel, = (TARGET / f"puma-checked-streams-build-v{version}/wheels").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        files = {name: digest(archive.read(name)) for name in archive.namelist()
                 if not name.endswith("/")}
    wheel_payloads.append({"wheel_sha256": digest(wheel.read_bytes()), "files": files})
changes = [name for name in sorted(set(wheel_payloads[0]["files"]) | set(wheel_payloads[1]["files"]))
           if wheel_payloads[0]["files"].get(name) != wheel_payloads[1]["files"].get(name)]
assert changes == ["pllm_run-0.1.0a2.dist-info/RECORD",
                   "pllm_run-0.1.0a2.dist-info/sboms/pllm-python.cyclonedx.json"]
write(OUT / "wheel-equivalence.json", {
    "package_files_identical": True, "changed_payloads": changes, "wheels": wheel_payloads})

sdk_inputs = json.loads((sdk / "test-input-sha256-fixture-fixed-v2.json").read_text())
with tarfile.open(OUT / "sdk-tested-inputs.tar.gz", "w:gz") as archive:
    for name, expected in sorted(sdk_inputs.items()):
        data = (ROOT / name).read_bytes()
        if name == "tests/test_replicated_policy_sdk.py":
            data = data.replace(b'"""Installed-wheel gates:', b'"""Deferred installed-wheel gates:', 1)
            data = data.replace(b"-I -m pytest -o pythonpath=. tests/test_replicated_policy_sdk.py",
                                b"-I -m pytest tests/test_replicated_policy_sdk.py", 1)
        assert digest(data) == expected, name
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        archive.addfile(info, io.BytesIO(data))
write(OUT / "sdk-tested-inputs.json", sdk_inputs)

base = json.loads((ROOT / "paper/artifacts/puma-sdk/local-v11/validation-inputs.json").read_text())
build = json.loads((OUT / "build-v2/sources-before.json").read_text())
paths = set(base) | set(build["files"]) | set(sdk_inputs)
paths |= set(json.loads((OUT / "compat/validation-inputs.json").read_text()))
paths |= {str(p.relative_to(ROOT)) for p in (ROOT / "tests").glob("*replicated*.py")}
paths |= {"paper/methods/puma-completion-plan.md", "paper/methods/puma-feasibility.md"}
inputs = {}
with tarfile.open(OUT / "inputs.tar.gz", "w:gz") as archive:
    for name in sorted(paths):
        assert not Path(name).is_absolute() and ".." not in Path(name).parts, name
        data = (ROOT / name).read_bytes()
        inputs[name] = digest(data)
        info = tarfile.TarInfo(name)
        info.size, info.mode, info.mtime = len(data), 0o644, 0
        archive.addfile(info, io.BytesIO(data))
write(OUT / "validation-inputs.json", inputs)
with tarfile.open(OUT / "inputs.tar.gz", "r:gz") as archive:
    recovered = {entry.name: digest(archive.extractfile(entry).read())
                 for entry in archive.getmembers()}
assert recovered == inputs


def junit(path):
    suites = ET.parse(path).getroot().iter("testsuite")
    total = {name: 0 for name in ("tests", "errors", "failures", "skipped")}
    for suite in suites:
        for name in total:
            total[name] += int(suite.attrib.get(name, 0))
    total["passed"] = total["tests"] - total["errors"] - total["failures"] - total["skipped"]
    assert total["errors"] == total["failures"] == 0
    return total


python = {"native_sdk": junit(OUT / "sdk/native-fixed.xml"),
          "focused": junit(OUT / "sdk/focused-fixed.xml"),
          "compatibility": junit(OUT / "compat/installed-tests.xml")}
oracle = json.loads((OUT / "oracle/receipt-20261019T091458Z.json").read_text())
assert oracle["status"] == "passed" and oracle["source_unchanged_during_run"]
assert oracle["source_sha256"]["crates/pllm-compiler/src/replicated_decoder/oracle_trace.rs"] == digest(
    (OUT / "oracle/oracle_trace-tested.rs").read_bytes())
native_counts = {}
for name in ("core", "integration", "compiler", "doctests"):
    text = (OUT / f"native/{name}.log").read_text()
    native_counts[name] = sum(map(int, re.findall(r"test result: ok\. (\d+) passed; 0 failed", text)))
native_counts["oracle"] = sum(
    int(n) for n in re.findall(r"test result: ok\. (\d+) passed; 0 failed",
                               (OUT / "oracle/audit-20261019T091458Z.log").read_text()))
assert native_counts == {"core": 98, "integration": 46, "compiler": 66, "doctests": 5, "oracle": 3}
oracle_log = (OUT / "oracle/audit-20261019T091458Z.log").read_text()
assert re.search(r"89 passed", oracle_log)
assert sum(x["passed"] for x in python.values()) == 462
assert sum(x["skipped"] for x in python.values()) == 7
summary = {
    "schema": "pllm.puma.checked-execution-evidence.v1", "status": "bounded_conformance_passed",
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "installed_python": python, "installed_python_passed": 462, "optional_skips": 7,
    "independent_python_oracle_passed": 89, "native": native_counts,
    "native_distinct_passed": sum(native_counts.values()),
    "generated_native_cases": oracle["native_program_cases"],
    "oracle_negative_controls": oracle["negative_control_total"],
    "installed_package_files": 368, "validation_inputs": len(inputs),
    "checkpoint": json.loads((OUT / "checkpoint/qwen-program-admission.json").read_text()),
    "wheel_package_equivalence": True, "full_checkpoint_execution": "blocked",
    "post_validation_test_only_changes": [
        "SDK gate command documentation now includes pytest pythonpath override; exact tested bytes in sdk-tested-inputs.tar.gz",
        "Native oracle rustfmt-only changes; exact tested bytes in oracle/oracle_trace-tested.rs"],
}
write(OUT / "summary.json", summary)
shutil.copy2(__file__, OUT / "collect.py")
files = [p for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
(OUT / "SHA256SUMS").write_text("".join(
    f"{digest(p.read_bytes())}  {p.relative_to(OUT).as_posix()}\n" for p in files))
print(json.dumps({"status": "collected", "files": len(files),
                  "bytes": sum(p.stat().st_size for p in files),
                  "inputs": len(inputs), "python": 462, "native": sum(native_counts.values())}))
