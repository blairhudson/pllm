"""Regenerate the deterministic source inventory and checksum list."""
from __future__ import annotations

import hashlib
import json
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


def main() -> None:
    old = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    paths = {ROOT / item["path"] for item in old["files"]}
    for source_root in SOURCE_ROOTS:
        for path in (ROOT / source_root).rglob("*"):
            relative = path.relative_to(ROOT)
            if path.is_file() and not EXCLUDED_PARTS.intersection(relative.parts):
                paths.add(path)
    for name in ("Cargo.lock", "restack.toml", "uv.lock", "rust-toolchain.toml"):
        paths.add(ROOT / name)
    paths = {
        path
        for path in paths
        if path.is_file()
        and path.relative_to(ROOT).as_posix() not in EXCLUDED_FILES
        and path.name not in {"SOURCE-MANIFEST.json", "SHA256SUMS.txt"}
        and path.suffix not in {".pyc", ".so"}
    }

    files = []
    checksums = []
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix()):
        relative = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
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
