from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DEVELOPER_REFERENCE = ROOT / "scripts/generate_developer_reference.py"


def run(*, check: bool) -> None:
    reference = [sys.executable, str(DEVELOPER_REFERENCE)]
    if check:
        reference.append("--check")
    subprocess.run(reference, cwd=ROOT, check=True)
    if check:
        subprocess.run(["npm", "test"], cwd=DOCS, check=True)
        subprocess.run(["npm", "run", "check:content"], cwd=DOCS, check=True)
    else:
        subprocess.run(["npm", "run", "generate"], cwd=DOCS, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    run(check=args.check)


if __name__ == "__main__":
    main()
