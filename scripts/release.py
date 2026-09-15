"""Validate a release tag and build a clean source archive without importing PLLM."""
from __future__ import annotations
import argparse
import ast
import hashlib
import re
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def version() -> str:
    tree = ast.parse((ROOT / "python/pllm/_version.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            return str(ast.literal_eval(node.value))
    raise ValueError("No package version")

def validate(tag: str, require_locks: bool = False) -> str:
    expected = "v" + version()
    if not re.fullmatch(r"v\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", tag) or tag != expected:
        raise ValueError(f"Tag must match package version: {expected}; got {tag!r}")
    cargo_file = ROOT / "Cargo.toml"
    if cargo_file.exists():
        cv = tomllib.loads(cargo_file.read_text())["workspace"]["package"]["version"]
        normalized = re.sub(r"-(alpha|beta|rc)\.(\d+)$", lambda m: {"alpha":"a","beta":"b","rc":"rc"}[m[1]] + m[2], cv)
        if normalized != version():
            raise ValueError("Cargo and Python version differ")
    if require_locks:
        for relative in ("uv.lock", "docs/package-lock.json", "Cargo.lock"):
            if not (ROOT / relative).is_file():
                raise ValueError(f"Missing {relative}. Run scripts/lock_dependencies.py and commit its outputs before tagging.")
    return expected

def source_archive(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"pllm-{version()}-repository.tar.gz"
    # git archive excludes ignored and untracked files. It cannot include local keys.
    subprocess.run(["git", "archive", "--format=tar.gz", f"--prefix=pllm-{version()}/", f"--output={target.resolve()}", "HEAD"], cwd=ROOT, check=True)
    with tarfile.open(target) as archive:
        assert archive.getmembers()
    return target

def checksums(directory: Path) -> None:
    rows = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (directory / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n")

def set_version(value: str) -> None:
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", value):
        raise ValueError("Use a version such as 0.1.0a1 or 0.1.0")
    (ROOT / "python/pllm/_version.py").write_text(f'__version__ = "{value}"\n')
    cargo = ROOT / "Cargo.toml"
    cargo_version = re.sub(
        r"(a|b|rc)(\d+)$",
        lambda match: "-" + {"a":"alpha", "b":"beta", "rc":"rc"}[match[1]] + "." + match[2],
        value,
    )
    cargo.write_text(re.sub(r'^version = ".*"$', f'version = "{cargo_version}"', cargo.read_text(), count=1, flags=re.M))
    citation = ROOT / "CITATION.cff"
    citation.write_text(re.sub(r"^version: .+$", f"version: {value}", citation.read_text(), flags=re.M))

def prepare(value: str) -> None:
    set_version(value)
    subprocess.run([sys.executable, str(ROOT / "scripts/lock_dependencies.py")], cwd=ROOT, check=True)
    print("Prepared version and dependency locks. Update CHANGELOG.md and VALIDATION.md, then review and commit.")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("tag")
    check.add_argument("--require-locks", action="store_true")
    archive = sub.add_parser("archive")
    archive.add_argument("--output", type=Path, default=ROOT / "release-assets")
    sums = sub.add_parser("checksums")
    sums.add_argument("directory", type=Path)
    bump = sub.add_parser("set-version")
    bump.add_argument("version")
    prepared = sub.add_parser("prepare", help="Set all versions and regenerate dependency locks")
    prepared.add_argument("version")
    args = parser.parse_args()
    if args.command == "check":
        print(validate(args.tag, args.require_locks))
    elif args.command == "archive":
        print(source_archive(args.output))
    elif args.command == "checksums":
        checksums(args.directory)
    elif args.command == "set-version":
        try:
            set_version(args.version)
        except ValueError as error:
            parser.error(str(error))
        print("Updated version. Regenerate dependency locks, update CHANGELOG.md, and commit before tagging.")
    else:
        try:
            prepare(args.version)
        except ValueError as error:
            parser.error(str(error))

if __name__ == "__main__":
    main()
