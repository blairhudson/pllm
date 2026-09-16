"""Create evidence downloads for the static site."""

from __future__ import annotations
import hashlib
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def zip_files(output: Path, root: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        checksums = []
        for path in sorted(files):
            name = path.relative_to(root).as_posix()
            content = path.read_bytes()
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
            checksums.append(f"{hashlib.sha256(content).hexdigest()}  {name}")
        info = zipfile.ZipInfo("SHA256SUMS.txt", ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, "\n".join(checksums) + "\n")


def main() -> None:
    downloads = ROOT / "docs/public/downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    evidence = ROOT / "research/evidence"
    files = [
        p
        for p in evidence.rglob("*")
        if p.is_file() and p.suffix in {".json", ".csv", ".md", ".patch"}
    ]
    zip_files(downloads / "evidence.zip", evidence, files)
    shutil.copyfile(evidence / "SUMMARY.json", downloads / "lifecycle-summary.json")
    shutil.copyfile(
        evidence / "current-runtime-2026-09-11.json",
        downloads / "current-runtime-2026-09-11.json",
    )
    (ROOT / "docs/public/.nojekyll").touch()
    print(f"Prepared {len(files)} evidence files")


if __name__ == "__main__":
    main()
