"""Resolve real dependency locks. Requires UV, Cargo, npm and registry access."""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate existing locks instead of creating them")
    args = parser.parse_args()
    for tool in ("uv", "npm", "cargo"):
        if shutil.which(tool) is None:
            parser.error(f"{tool} is required")
    if args.check:
        for name in ("uv.lock", "docs/package-lock.json", "Cargo.lock"):
            if not (ROOT / name).is_file():
                parser.error(f"Missing {name}")
        subprocess.run(["cargo", "metadata", "--locked", "--format-version", "1"], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["uv", "lock", "--check"], cwd=ROOT, check=True)
        # npm ci checks package.json against package-lock.json; avoid lifecycle scripts here.
        subprocess.run(["npm", "ci", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=ROOT / "docs", check=True)
    else:
        subprocess.run(["cargo", "generate-lockfile"], cwd=ROOT, check=True)
        subprocess.run(["uv", "lock", "--python", "3.13"], cwd=ROOT, check=True)
        subprocess.run(["npm", "install", "--package-lock-only", "--ignore-scripts", "--no-audit", "--no-fund"], cwd=ROOT / "docs", check=True)
        print("Generated uv.lock, Cargo.lock and docs/package-lock.json. Review and commit all three.")

if __name__ == "__main__":
    main()
