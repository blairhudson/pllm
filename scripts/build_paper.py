"""Build the paper PDF and web edition from Pandoc Markdown."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCE = PAPER / "manuscript.md"
FILTER = PAPER / "paper.lua"
WEB_TEMPLATE = PAPER / "web.template.md"
PDF = PAPER / "main.pdf"
WEB = ROOT / "docs/content/docs/research/paper.mdx"


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def pdf_engine(requested: str | None) -> str:
    if requested:
        if shutil.which(requested) is None:
            raise SystemExit(f"PDF engine not found: {requested}")
        return requested
    for candidate in ("tectonic", "pdflatex"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit("PDF build requires tectonic or pdflatex")


def build_pdf(engine: str | None) -> None:
    selected = pdf_engine(engine or os.environ.get("PANDOC_PDF_ENGINE"))
    command = [
        "pandoc",
        str(SOURCE),
        "--from=markdown+raw_tex+fenced_divs+link_attributes",
        "--standalone",
        "--number-sections",
        "--shift-heading-level-by=-1",
        f"--lua-filter={FILTER}",
        f"--pdf-engine={selected}",
        f"--output={PDF}",
    ]
    if Path(selected).name == "tectonic":
        command.append("--pdf-engine-opt=--chatter=minimal")
    run(command)
    print(f"Built {PDF.relative_to(ROOT)} with {selected}.")


def build_web() -> None:
    WEB.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "pandoc",
            str(SOURCE),
            "--from=markdown+raw_tex+fenced_divs+link_attributes",
            "--to=gfm",
            "--wrap=none",
            f"--lua-filter={FILTER}",
            f"--template={WEB_TEMPLATE}",
            f"--output={WEB}",
        ]
    )
    print(f"Built {WEB.relative_to(ROOT)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("all", "pdf", "web"), nargs="?", default="all")
    parser.add_argument("--pdf-engine")
    args = parser.parse_args()
    if args.target in ("all", "pdf"):
        build_pdf(args.pdf_engine)
    if args.target in ("all", "web"):
        build_web()


if __name__ == "__main__":
    main()
