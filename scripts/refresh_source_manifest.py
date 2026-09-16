"""Regenerate the deterministic source inventory and checksum list."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    ".github",
    "benchmarks",
    "config",
    "crates",
    "deploy",
    "design",
    "docs",
    "examples",
    "infra",
    "paper",
    "python",
    "research",
    "schemas",
    "scripts",
    "tests",
    "verification",
)
EXCLUDED_PARTS = {
    ".next",
    ".terraform",
    ".pytest_cache",
    ".ruff_cache",
    ".source",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "out",
    "target",
}
EXCLUDED_FILES = {
    "docs/tsconfig.tsbuildinfo",
    "paper/main.pdf",
    "research/lifecycle/study-tests.xml",
}


def git_blob_contents(paths: list[Path]) -> dict[Path, bytes]:
    names = [path.relative_to(ROOT).as_posix() for path in paths]
    hashed = subprocess.run(
        ["git", "hash-object", "-w", "--filters", "--stdin-paths"],
        cwd=ROOT,
        check=True,
        input=("\n".join(names) + "\n").encode(),
        stdout=subprocess.PIPE,
    ).stdout.decode().splitlines()
    if len(hashed) != len(paths):
        raise RuntimeError("git did not hash every source file")

    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    process.stdin.write(("\n".join(hashed) + "\n").encode())
    process.stdin.close()
    contents: dict[Path, bytes] = {}
    for path, expected in zip(paths, hashed, strict=True):
        header = process.stdout.readline().decode().strip().split()
        if len(header) != 3 or header[:2] != [expected, "blob"]:
            raise RuntimeError(f"invalid git object for {path.relative_to(ROOT)}")
        data = process.stdout.read(int(header[2]))
        if process.stdout.read(1) != b"\n":
            raise RuntimeError(f"invalid git object boundary for {path.relative_to(ROOT)}")
        contents[path] = data
    if process.wait() != 0:
        raise RuntimeError("git cat-file failed")
    return contents


def main() -> None:
    old = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    listed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.decode().split("\0")
    root_files = {"Cargo.lock", "restack.toml", "uv.lock", "rust-toolchain.toml"}
    paths = {
        ROOT / name
        for name in listed
        if name and (name.partition("/")[0] in SOURCE_ROOTS or name in root_files)
    }
    paths = {
        path
        for path in paths
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FILES
        and path.name not in {"SOURCE-MANIFEST.json", "SHA256SUMS.txt"}
        and path.suffix not in {".pyc", ".so"}
    }

    ordered_paths = sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())
    contents = git_blob_contents(ordered_paths)
    files = []
    checksums = []
    for path in ordered_paths:
        relative = path.relative_to(ROOT).as_posix()
        data = contents[path]
        digest = hashlib.sha256(data).hexdigest()
        files.append({"path": relative, "bytes": len(data), "sha256": digest})
        checksums.append(f"{digest}  {relative}")

    (ROOT / "SOURCE-MANIFEST.json").write_text(
        json.dumps({"version": old["version"], "files": files}, indent=2) + "\n"
    )
    (ROOT / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(f"Recorded {len(files)} source files")


if __name__ == "__main__":
    main()
