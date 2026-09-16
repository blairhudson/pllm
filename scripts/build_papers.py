"""Build the PLLM papers from Markdown with Pandoc."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
DOWNLOADS = ROOT / "docs" / "public" / "downloads"
WEB_DIR = ROOT / "docs" / "content" / "research"


@dataclass(frozen=True)
class Paper:
    source: str
    output: str
    web: str
    max_pages: int
    bibliography: bool = False


PAPERS = {
    "paper": Paper("manuscript.md", "paper.pdf", "paper.mdx", 4, True),
    "whitepaper": Paper("whitepaper.md", "whitepaper.pdf", "whitepaper.mdx", 2),
}


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required executable not found: {name}")
    return path


def pdf_engine(requested: str | None) -> str:
    if requested:
        return executable(requested)
    for name in ("tectonic", "pdflatex"):
        if path := shutil.which(name):
            return path
    raise SystemExit("Install Tectonic or pdfLaTeX to build the papers")


def pandoc_args(spec: Paper) -> list[str]:
    args = [executable("pandoc"), str(PAPER_DIR / spec.source), "--from=markdown+raw_tex"]
    if spec.bibliography:
        args.append("--citeproc")
    return args


def build_pdf(spec: Paper, engine: str) -> Path:
    output = PAPER_DIR / spec.output
    env = os.environ.copy()
    env.setdefault("SOURCE_DATE_EPOCH", "0")
    run(
        [
            *pandoc_args(spec),
            "--standalone",
            f"--pdf-engine={engine}",
            f"--include-in-header={PAPER_DIR / 'header.tex'}",
            f"--lua-filter={PAPER_DIR / 'pdf.lua'}",
            f"--output={output}",
        ],
        env=env,
    )

    info = subprocess.check_output([executable("pdfinfo"), str(output)], text=True)
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if pages_match is None:
        raise SystemExit(f"Could not read page count from {output}")
    pages = int(pages_match.group(1))
    if pages > spec.max_pages:
        raise SystemExit(f"{output} has {pages} pages; limit is {spec.max_pages}")
    if not re.search(r"^Page size:\s+612 x 792 pts", info, re.MULTILINE):
        raise SystemExit(f"{output} is not US Letter")
    return output


def build_web(spec: Paper) -> Path:
    output = WEB_DIR / spec.web
    run(
        [
            *pandoc_args(spec),
            "--standalone",
            "--to=gfm",
            "--wrap=none",
            f"--template={PAPER_DIR / 'web.template.md'}",
            f"--lua-filter={PAPER_DIR / 'web.lua'}",
            f"--output={output}",
        ]
    )
    return output


def source_archive(name: str, spec: Paper) -> Path:
    output = DOWNLOADS / f"{name}-source.zip"
    files = [
        Path("LICENSE"),
        Path("scripts/build_papers.py"),
        Path("paper") / spec.source,
        Path("paper/header.tex"),
        Path("paper/pdf.lua"),
    ]
    if spec.bibliography:
        files.append(Path("paper/references.bib"))
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative.as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *PAPERS], nargs="?", default="all")
    parser.add_argument("--pdf-engine")
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()

    names = PAPERS if args.target == "all" else (args.target,)
    engine = pdf_engine(args.pdf_engine)
    if not args.pdf_only:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        WEB_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = PAPERS[name]
        pdf = build_pdf(spec, engine)
        if not args.pdf_only:
            shutil.copyfile(pdf, DOWNLOADS / spec.output)
            build_web(spec)
            source_archive(name, spec)
        print(f"Built {name} from paper/{spec.source}")


if __name__ == "__main__":
    main()
