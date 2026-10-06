"""Build the PLLM papers from Markdown with Pandoc."""

from __future__ import annotations

import argparse
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = ROOT / "paper"
BUILD_DIR = ROOT / "docs" / "build" / "papers"
FIGURES = BUILD_DIR / "figures"
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
    "paper": Paper("manuscript.md", "paper.pdf", "paper.mdx", 5, True),
    "whitepaper": Paper("whitepaper.md", "whitepaper.pdf", "whitepaper.mdx", 2),
}

WHITEPAPER_FIGURES = ("mechanics", "research-loop")


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


def browser_engine(requested: str | None) -> str:
    if requested:
        if path := shutil.which(requested):
            return path
        raise SystemExit(f"Browser executable not found: {requested}")
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        if path := shutil.which(name):
            return path
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if mac_chrome.is_file():
        return str(mac_chrome)
    raise SystemExit("Install Chrome or Chromium to build the whitepaper PDF")


def pandoc_args(spec: Paper) -> list[str]:
    args = [executable("pandoc"), str(PAPER_DIR / spec.source),
            "--from=markdown+raw_tex", "--fail-if-warnings"]
    if spec.bibliography:
        args.append("--citeproc")
    return args


def validate_pdf(output: Path, spec: Paper) -> None:
    info = subprocess.check_output([executable("pdfinfo"), str(output)], text=True)
    pages_match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if pages_match is None:
        raise SystemExit(f"Could not read page count from {output}")
    pages = int(pages_match.group(1))
    if pages > spec.max_pages:
        raise SystemExit(f"{output} has {pages} pages; limit is {spec.max_pages}")
    if not re.search(r"^Page size:\s+612 x 792 pts", info, re.MULTILINE):
        raise SystemExit(f"{output} is not US Letter")


def build_pdf(spec: Paper, engine: str) -> Path:
    output = BUILD_DIR / spec.output
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
    validate_pdf(output, spec)
    return output


def build_whitepaper_pdf(spec: Paper, browser: str) -> Path:
    output = BUILD_DIR / spec.output
    with tempfile.TemporaryDirectory(prefix="pllm-whitepaper-") as temporary:
        temp = Path(temporary)
        html = temp / "whitepaper.html"
        pdf = temp / spec.output
        run([
            *pandoc_args(spec),
            "--standalone",
            "--to=html5",
            "--embed-resources",
            f"--resource-path={BUILD_DIR}",
            f"--template={PAPER_DIR / 'whitepaper_print.html'}",
            f"--css={PAPER_DIR / 'whitepaper_print.css'}",
            f"--lua-filter={PAPER_DIR / 'whitepaper_print.lua'}",
            f"--output={html}",
        ])
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-extensions",
            "--disable-sync",
            "--no-default-browser-check",
            "--no-first-run",
            "--no-pdf-header-footer",
            f"--user-data-dir={temp / 'profile'}",
            f"--print-to-pdf={pdf}",
            html.as_uri(),
        ]
        # Chrome on macOS sometimes writes a complete PDF but leaves background
        # services running. Wait for the PDF trailer, then own process shutdown.
        with (temp / "browser.log").open("wb") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 40
                while time.monotonic() < deadline:
                    if pdf.is_file() and b"%%EOF" in pdf.read_bytes()[-1024:]:
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.2)
                else:
                    raise SystemExit(
                        "Browser timed out printing whitepaper:\n"
                        + (temp / "browser.log").read_text(encoding="utf-8", errors="replace")[-1200:]
                    )
                if not pdf.is_file() or b"%%EOF" not in pdf.read_bytes()[-1024:]:
                    raise SystemExit(
                        "Browser did not print whitepaper:\n"
                        + (temp / "browser.log").read_text(encoding="utf-8", errors="replace")[-1200:]
                    )
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
        if preview := os.environ.get("PLLM_WHITEPAPER_PREVIEW"):
            shutil.copyfile(pdf, preview)
        validate_pdf(pdf, spec)
        shutil.copyfile(pdf, output)
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
        Path("docs/scripts/build-papers.py"),
        Path("paper") / spec.source,
        Path("paper/web.lua"),
        Path("paper/web.template.md"),
    ]
    if spec.bibliography:
        files.extend((Path("paper/header.tex"), Path("paper/pdf.lua"), Path("paper/references.bib")))
    if name == "whitepaper":
        files.extend([
            Path("paper/whitepaper_print.lua"),
            Path("paper/whitepaper_print.html"),
            Path("paper/whitepaper_print.css"),
            Path("docs/scripts/render-paper-figures.py"),
            Path("docs/evidence/current-runtime-2026-09-11.json"),
        ])
        for figure in WHITEPAPER_FIGURES:
            files.extend([FIGURES.relative_to(ROOT) / f"{figure}.{extension}"
                          for extension in ("svg", "png")])
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative.as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    return output


def arxiv_archive(spec: Paper) -> Path:
    """Export standalone, citeproc-resolved TeX; arXiv need not run Pandoc."""
    output = DOWNLOADS / "paper-arxiv-source.zip"
    with tempfile.TemporaryDirectory(prefix="pllm-arxiv-") as temporary:
        source = Path(temporary) / "main.tex"
        run([
            *pandoc_args(spec), "--standalone", "--to=latex",
            f"--include-in-header={PAPER_DIR / 'header.tex'}",
            f"--lua-filter={PAPER_DIR / 'pdf.lua'}", f"--output={source}",
        ])
        files = {
            "main.tex": source.read_bytes(),
            "LICENSE": (ROOT / "LICENSE").read_bytes(),
            "README.txt": (
                "PLLM technical paper: standalone arXiv source\n\n"
                "Compile main.tex with pdfLaTeX twice. References are already\n"
                "resolved and embedded; no Pandoc, Python, BibTeX, external\n"
                "bibliography, or generated figures are required.\n\n"
                "Canonical authoring source: paper/manuscript.md in the PLLM\n"
                "repository, https://github.com/blairhudson/pllm\n"
            ).encode(),
        }
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(files.items()):
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["all", *PAPERS], nargs="?", default="all")
    parser.add_argument("--pdf-engine")
    parser.add_argument("--browser", help="Chrome or Chromium binary for whitepaper print layout")
    parser.add_argument("--pdf-only", action="store_true")
    args = parser.parse_args()

    names = PAPERS if args.target == "all" else (args.target,)
    engine = pdf_engine(args.pdf_engine) if "paper" in names else None
    browser = browser_engine(args.browser) if "whitepaper" in names else None
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if "whitepaper" in names:
        run([sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py")])
    if not args.pdf_only:
        DOWNLOADS.mkdir(parents=True, exist_ok=True)
        WEB_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = PAPERS[name]
        if name == "whitepaper":
            assert browser is not None
            pdf = build_whitepaper_pdf(spec, browser)
        else:
            assert engine is not None
            pdf = build_pdf(spec, engine)
        if not args.pdf_only:
            shutil.copyfile(pdf, DOWNLOADS / spec.output)
            if name == "whitepaper":
                figure_downloads = DOWNLOADS / "figures"
                figure_downloads.mkdir(parents=True, exist_ok=True)
                for figure in WHITEPAPER_FIGURES:
                    shutil.copyfile(FIGURES / f"{figure}.png",
                                    figure_downloads / f"{figure}.png")
            build_web(spec)
            source_archive(name, spec)
            if name == "paper":
                arxiv_archive(spec)
        print(f"Built {name} from paper/{spec.source}")


if __name__ == "__main__":
    main()
