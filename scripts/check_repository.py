"""Dependency free release structure and provenance checks."""
from __future__ import annotations
import ast
import hashlib
import json
import re
import tomllib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    required = [
        "README.md", "LICENSE", "NOTICE", "SECURITY.md", "CONTRIBUTING.md", "RELEASING.md",
        "CITATION.cff", "pyproject.toml", "python/pllm/cli.py", "Cargo.toml", "crates/pllm-python/src/lib.rs", "crates/pllm-core/src/kernels.rs",
        "docs/package.json", "docs/next.config.mjs", "paper/manuscript.md", "paper/paper.lua", "scripts/build_paper.py",
        ".github/workflows/ci.yml", ".github/workflows/release.yml", ".github/workflows/pages.yml",
    ]
    for name in required:
        if not (ROOT / name).is_file():
            raise SystemExit(f"Missing {name}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["name"] == "pllm"
    assert project["project"]["scripts"]["pllm"] == "pllm.cli:main"
    assert project["build-system"]["build-backend"] == "maturin"
    assert project["tool"]["maturin"]["module-name"] == "pllm._native"
    assert project["tool"]["maturin"]["python-source"] == "python"
    assert project["tool"]["maturin"]["python-packages"] == ["pllm"]
    assert project["tool"]["maturin"]["manifest-path"] == "crates/pllm-python/Cargo.toml"
    assert not (ROOT / "he_openai").exists()
    assert not (ROOT / "python/he_openai").exists()
    assert not (ROOT / "paper/main.tex").exists()
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text())
    assert set(cargo["workspace"]["members"]) == {"crates/pllm-core", "crates/pllm-python"}
    core = tomllib.loads((ROOT / "crates/pllm-core/Cargo.toml").read_text())
    assert "pyo3" not in core["dependencies"]
    for base in ("python/pllm", "scripts", "tests", "benchmarks"):
        for path in (ROOT / base).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        assert "pull_request_target" not in text, path
        for action in re.findall(r"uses:\s*([^\s#]+)", text):
            if not action.startswith("./"):
                assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), (path, action)
    manifest = json.loads((ROOT / "SOURCE-MANIFEST.json").read_text())
    checksum_rows = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        assert not relative.is_absolute() and ".." not in relative.parts, relative
        data = (ROOT / relative).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert len(data) == item["bytes"], relative
        assert digest == item["sha256"], relative
        checksum_rows.append(f"{digest}  {relative.as_posix()}")
    assert (ROOT / "SHA256SUMS.txt").read_text().splitlines() == checksum_rows
    locks = ("uv.lock", "docs/package-lock.json", "Cargo.lock", "research/lifecycle/uv.lock")
    print(json.dumps({"structure": "passed", "python_syntax": "passed", "workflow_pins": "passed", "source_integrity": f"{len(checksum_rows)} files passed", "dependency_locks": {p: (ROOT/p).is_file() for p in locks}}, indent=2))

if __name__ == "__main__":
    main()
