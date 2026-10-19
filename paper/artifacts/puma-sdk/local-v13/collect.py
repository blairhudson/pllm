"""Freeze original PUMA receipts and source overlays; never rerun measurements."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
OUT = ROOT / "paper/artifacts/puma-sdk/local-v13"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads((ROOT / path).read_text())


def write(path, value):
    path = OUT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def archive(path, files):
    with (OUT / path).open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name, data in sorted(files.items()):
                    item = tarfile.TarInfo(name)
                    item.size = len(data)
                    item.mode = 0o644
                    tar.addfile(item, io.BytesIO(data))


copies = {
    "build-v1": "target/puma-trie-norm-build-v1",
    "build-v2": "target/puma-trie-norm-build-v2",
    "validation-v1": "target/puma-trie-norm-validation-v1",
    "validation-v2": "target/puma-trie-norm-validation-v2-retry",
    "checkpoint-v1": "target/puma-trie-norm-qwen-v1",
    "checkpoint-v2": "target/puma-trie-norm-qwen-v2",
    "attempts/wrapper-before-pytest": "target/puma-trie-norm-validation-v2",
}
for destination, source in copies.items():
    for path in sorted((ROOT / source).iterdir()):
        if path.is_file() and path.suffix in {".py", ".json", ".log", ".xml"}:
            target = OUT / destination / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            assert not target.exists(), target
            shutil.copyfile(path, target)
for name in ("run_installed.py", "base-harness.json", "exclusion.json", "receipt.json"):
    path = ROOT / "target/puma-trie-norm-validation-v3" / name
    target = OUT / "attempts/excluded-duplicate" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists(), target
    shutil.copyfile(path, target)

native = {
    "core-v1": "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --lib puma -- --test-threads=1",
    "compiler-v2": "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-compiler --lib replicated_decoder -- --test-threads=1 --nocapture",
    "native-integration-v1": "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --test puma_allocation --test puma_multiply --test puma_norm_allocation --test puma_prss --test puma_truncation -- --test-threads=1",
    "attention-v3": "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-compiler --lib attention -- --test-threads=1 --nocapture",
    "clippy-v3": "CARGO_BUILD_JOBS=1 cargo clippy --locked --offline -p pllm-core -p pllm-compiler -p pllm-python --lib -- -D warnings -A clippy::needless_range_loop -A clippy::nonminimal_bool",
    "fmt-v1": "cargo fmt --all -- --check",
    "coverage-v3": "target/puma-sdk-integration-env-v1/bin/python -I scripts/benchmark_coverage.py --check",
}
for name in native:
    shutil.copyfile(ROOT / f"target/puma-trie-norm-{name}.log", OUT / "native" / f"{name}.log")
write("native/commands.json", {"retained_stdout": native,
    "recovered_session_excerpt": {"norm-followup-session.txt":
        "CARGO_BUILD_JOBS=1 cargo test --locked --offline -p pllm-core --lib multiply::norm -- --test-threads=1"},
    "scope": "Replay command forms; original installed commands are in their own receipts. Follow-up suites overlap earlier suites."})

before = load("target/puma-trie-norm-build-v1/sources-after.json")["files"]
after = load("target/puma-trie-norm-build-v2/sources-after.json")["files"]
changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
assert changed == ["crates/pllm-core/src/puma/multiply_norm.rs", "crates/pllm-core/src/puma/multiply_norm_tests.rs"]
files = {name: (ROOT / name).read_bytes() for name in after}
assert all(sha(files[name]) == expected for name, expected in after.items())
inputs = load("target/puma-trie-norm-validation-v1/validation-inputs.json")
for name, expected in inputs.items():
    data = (ROOT / name).read_bytes()
    assert sha(data) == expected, name
    if name.startswith("tests/"):
        files[name] = data
for path in (ROOT / "paper/reproduction-contracts/puma").rglob("*"):
    if path.is_file() and "__pycache__" not in path.parts:
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
for name in ("benchmarks/coverage-surface.json", "scripts/benchmark_coverage.py",
             "scripts/build_paper_release.py", "scripts/check_cached_wheel.py", "scripts/host_swap.py"):
    files[name] = (ROOT / name).read_bytes()
write("validation-inputs.json", {name: sha(data) for name, data in sorted(files.items())})
archive("inputs.tar.gz", files)

# Reverse the recovered two-file follow-up, checking the exact original hashes.
name = changed[0]
text = files[name].decode()
start = text.index("        let mut raw = Interval::new(-radius, radius)?;")
end = text.index("        Ok(Self {", start)
text = text[:start] + text[end:]
text = text.replace("            raw,\n", "            raw: Interval::new(-radius, radius)?,\n", 1)
text = text.replace("/// Cauchy-Schwarz envelope intersected with any admitted scalar-product hull.",
                    "/// Inclusive conservative raw dot envelope: ±ceil(sqrt(left * right)).")
text = text.replace("/// row-energy admission intersected with any admitted scalar-product hull.",
                    "/// row-energy admission instead of the independent scalar product hull.")
overlay = {name: text.encode()}
name = changed[1]
text = files[name].decode()
start = text.index("#[test]\nfn loose_energy_certificates_cannot_widen_admitted_scalar_bounds()")
end = text.index("#[test]\nfn integer_square_root_and_overflow_boundaries()", start)
overlay[name] = (text[:start] + text[end:]).encode()
assert all(sha(data) == before[name] for name, data in overlay.items())
archive("build-v1/inputs-overlay.tar.gz", overlay)
write("build-v1/overlay-inputs.json", {name: sha(data) for name, data in overlay.items()})

def junit(folder):
    suites = list(ET.parse(OUT / folder / "installed-tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.attrib.get(key, "0")) for s in suites)
              for key in ("tests", "skipped", "errors", "failures")}
    assert counts["failures"] == counts["errors"] == 0
    counts["passed"] = counts["tests"] - counts["skipped"]
    receipt = json.loads((OUT / folder / "receipt.json").read_text())
    assert receipt["status"] == "passed" and receipt["inputs_match_before_and_after"]
    assert receipt["source_match_before_and_after"]
    return {"counts": counts, "receipt": receipt}

native_counts = {}
for path in sorted((OUT / "native").iterdir()):
    if path.suffix in {".log", ".txt"}:
        counts = re.findall(r"test result: ok\. (\d+) passed; (\d+) failed", path.read_text())
        if counts:
            assert all(int(failed) == 0 for _, failed in counts)
            native_counts[path.name] = sum(int(passed) for passed, _ in counts)
oracle = load("target/puma-trie-norm-native-summary-v1.json")
raw_cases = []
for line in (OUT / "native/compiler-v2.log").read_text().splitlines():
    if "ORACLE_CASE " in line:
        raw_cases.append(json.loads(line.split("ORACLE_CASE ", 1)[1]))
assert len(raw_cases) == oracle["oracle_cases"]
assert sum(case["checkpoints"] for case in raw_cases) == oracle["tensor_checkpoints"]
assert sum(sum(case["negative_controls"].values()) for case in raw_cases) == oracle["negative_controls"]
checkpoint = {}
for version in (1, 2):
    record = json.loads((OUT / f"checkpoint-v{version}/qwen-program-admission.json").read_text())
    assert record["status"] == "rejected" and record["max_swap_used_growth_bytes"] == 0
    assert "Some(RmsNorm), layer: Some(6)" in record["reason"]
    checkpoint[f"v{version}"] = record
write("summary.json", {
    "schema": "pllm.puma.sdk-validation.local-v13.v1",
    "baseline_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "installed": {name: junit(name) for name in ("validation-v1", "validation-v2")},
    "native_tests_by_log_not_additive": native_counts,
    "oracle": oracle, "checkpoint": checkpoint,
    "build_source_changes": changed,
    "excluded_attempts": ["wrapper-before-pytest", "excluded-duplicate"],
    "scope": "Generated conformance plus separately rejected pinned source publication; no complete checkpoint response or matched serving measurement",
})
shutil.copyfile(Path(__file__), OUT / "collect.py")
with (OUT / "SHA256SUMS").open("x") as stream:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            stream.write(f"{sha(path.read_bytes())}  {path.relative_to(OUT).as_posix()}\n")
print(json.dumps({"status": "collected", "path": str(OUT.relative_to(ROOT)),
                  "files": len(list(OUT.rglob("*"))), "oracle": oracle}))
