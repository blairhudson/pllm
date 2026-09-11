"""Build and verify the standalone PLLM whitepaper artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "whitepaper.tex"
DEFAULT_PDF = ROOT / "paper" / "whitepaper.pdf"
DEFAULT_ARCHIVE = ROOT / "paper" / "whitepaper-source.zip"


def engine(requested: str | None) -> str:
    if requested:
        if shutil.which(requested) is None:
            raise SystemExit(f"PDF engine not found: {requested}")
        return requested
    for candidate in ("tectonic", "pdflatex"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit("Whitepaper build requires tectonic or pdflatex")


def build_pdf(selected: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if Path(selected).name == "tectonic":
        command = [selected, "--outdir", str(output.parent), str(SOURCE)]
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        command = [
            selected,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output.parent}",
            str(SOURCE),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(command, cwd=ROOT, check=True)
    built = output.parent / "whitepaper.pdf"
    if built != output:
        built.replace(output)


def page_count(pdf: Path) -> int:
    tool = shutil.which("pdfinfo")
    if tool is None:
        raise SystemExit("pdfinfo is required to enforce the two-page limit")
    result = subprocess.run(
        [tool, str(pdf)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    pages = None
    letter = False
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            pages = int(line.split(":", 1)[1].strip())
        if line.startswith("Page size:"):
            letter = "612 x 792 pts" in line
    if pages is None:
        raise SystemExit("pdfinfo did not report a page count")
    if not 1 <= pages <= 2:
        raise SystemExit(f"Whitepaper must be 1-2 pages; built {pages}")
    if not letter:
        raise SystemExit("Whitepaper must use US-letter pages (612 x 792 points)")
    return pages


def display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


def source_archive(output: Path) -> None:
    files = (SOURCE, Path(__file__).resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        checksums = []
        for source in files:
            name = source.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(name, (2026, 9, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            data = source.read_bytes()
            archive.writestr(info, data)
            checksums.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
        info = zipfile.ZipInfo("SHA256SUMS.txt", (2026, 9, 11, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, "\n".join(checksums) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--pdf-engine")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="copy generated artifacts into docs/public/downloads",
    )
    args = parser.parse_args()
    pdf = args.pdf.resolve()
    archive = args.archive.resolve()
    build_pdf(engine(args.pdf_engine), pdf)
    pages = page_count(pdf)
    source_archive(archive)
    if args.publish:
        downloads = ROOT / "docs" / "public" / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, downloads / "whitepaper.pdf")
        shutil.copyfile(SOURCE, downloads / "whitepaper.tex")
        shutil.copyfile(archive, downloads / "whitepaper-source.zip")
    print(
        f"Built {display_path(pdf)} ({pages} pages) and "
        f"{display_path(archive)}."
    )


if __name__ == "__main__":
    main()
