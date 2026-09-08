"""Copy the built paper and create source/evidence downloads for the static site."""
from __future__ import annotations
import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def zip_files(output: Path, root: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        checksums = []
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            archive.write(path, name)
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}")
        archive.writestr("SHA256SUMS.txt", "\n".join(checksums) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", type=Path, default=ROOT / "paper/main.pdf")
    args = parser.parse_args()
    if not args.paper.is_file():
        parser.error("Build the paper first with make paper (or pass --paper PATH)")
    downloads = ROOT / "docs/public/downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.paper, downloads / "paper.pdf")
    files = [
        ROOT / "paper" / name
        for name in ("manuscript.md", "paper.lua", "web.template.md", "README.md", "Makefile")
    ]
    files.append(ROOT / "scripts/build_paper.py")
    zip_files(downloads / "paper-source.zip", ROOT, files)
    evidence = ROOT / "research/evidence"
    files = [p for p in evidence.rglob("*") if p.is_file() and p.suffix in {".json", ".csv", ".md", ".patch"}]
    zip_files(downloads / "evidence.zip", evidence, files)
    shutil.copyfile(evidence / "SUMMARY.json", downloads / "lifecycle-summary.json")
    (ROOT / "docs/public/.nojekyll").touch()
    print(f"Prepared {len(files)} evidence files and manuscript downloads")

if __name__ == "__main__":
    main()
