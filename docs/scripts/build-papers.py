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
    figures: tuple[str, ...] = ()
    figure_inputs: tuple[str, ...] = ()


PAPERS = {
    # Total-document guards include references/appendices, not venue page limits.
    "paper": Paper("manuscript.md", "paper.pdf", "paper.mdx", 20, True),
    "low-memory": Paper("low-memory.md", "low-memory.pdf", "low-memory.mdx", 16, True,
                        figures=("low-memory-history",),
                        figure_inputs=("paper/artifacts/historical-results-v1.json",)),
    "whitepaper": Paper("whitepaper.md", "whitepaper.pdf", "whitepaper.mdx", 2,
                        figures=("mechanics", "research-loop"),
                        figure_inputs=("docs/evidence/current-runtime-2026-09-11.json",)),
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


def whitepaper_profile_processes(profile: Path) -> list[int]:
    """Read-only coverage for detached helpers retaining this unique profile."""
    result = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True,
                            text=True, check=True, timeout=1)
    pattern = re.compile(re.escape(str(profile)) + r"(?=$|[/\s\"'])")
    return [int(fields[0]) for line in result.stdout.splitlines()
            if len(fields := line.split(None, 1)) == 2
            and fields[0].isdigit() and pattern.search(fields[1])]


def cleanup_whitepaper_browser(process: subprocess.Popen, profile: Path) -> None:
    """Reap our child, drain its private group, and check visible profile helpers."""
    ownership = f"owned browser PID/group {process.pid}, profile {profile}"
    try:
        # The PDF trailer can arrive just before Chrome's ordinary exit. Let that
        # exit finish before signalling a sandboxed process group unnecessarily.
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            if process.poll() is not None:
                break
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                # The leader/group may exit between poll and killpg.
                pass
            except PermissionError:
                # Group denial does not authorize signalling discovered helpers.
                # Popen still owns this exact unreaped leader; ask it to retire
                # its children, then verify group drain rather than assuming it.
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    raise RuntimeError(f"Cannot signal {ownership}; cleanup unverified") from error
            try:
                process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired as error:
                if sig == signal.SIGKILL:
                    raise RuntimeError(f"Timed out reaping {ownership}; cleanup unverified") from error

    deadline = time.monotonic() + 5
    while True:
        denied = False
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            try:
                helpers = whitepaper_profile_processes(profile)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(f"Cannot inspect helpers for {ownership}; cleanup unverified") from error
            if not helpers:
                return
            detail = f"profile-associated helpers remain: {helpers}"
        except PermissionError:
            denied = True
        else:
            detail = "group members remain"
        if time.monotonic() >= deadline:
            if denied:
                detail = "group visibility denied"
            raise RuntimeError(f"Cannot confirm drain of {ownership}: {detail}; helper cleanup unverified")
        time.sleep(0.1)


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
                render_error = sys.exception()
                try:
                    cleanup_whitepaper_browser(process, temp / "profile")
                except Exception as cleanup_error:
                    if render_error is None:
                        raise
                    message = f"Browser cleanup also failed: {cleanup_error}"
                    render_error.add_note(message)
                    print(message, file=sys.stderr)
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
        ])
    if spec.figures:
        files.extend([Path("docs/scripts/render-paper-figures.py"),
                      *(Path(relative) for relative in spec.figure_inputs)])
        for figure in spec.figures:
            files.extend([FIGURES.relative_to(ROOT) / f"{figure}.{extension}"
                          for extension in ("svg", "png")])
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative.as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, (ROOT / relative).read_bytes())
    return output


def arxiv_archive(name: str, spec: Paper) -> Path:
    """Export standalone, citeproc-resolved TeX; arXiv need not run Pandoc."""
    output = DOWNLOADS / f"{name}-arxiv-source.zip"
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
                f"PLLM {name}: standalone arXiv source\n\n"
                "Compile main.tex with pdfLaTeX twice. References are already\n"
                "resolved and embedded; no Pandoc, Python, BibTeX, external\n"
                "bibliography, or figure generation are required. Referenced\n"
                "images are bundled at their TeX-relative paths.\n\n"
                f"Canonical authoring source: paper/{spec.source} in the PLLM\n"
                "repository, https://github.com/blairhudson/pllm\n\n"
                "A buildable archive does not establish submission readiness.\n"
                "Working-draft evidence placeholders remain in main.tex.\n"
            ).encode(),
        }
        for figure in spec.figures:
            relative = FIGURES.relative_to(ROOT) / f"{figure}.png"
            files[relative.as_posix()] = (ROOT / relative).read_bytes()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for relative, content in sorted(files.items()):
                info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
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
    engine = pdf_engine(args.pdf_engine) if any(PAPERS[name].bibliography for name in names) else None
    browser = browser_engine(args.browser) if "whitepaper" in names else None
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    figures = tuple(dict.fromkeys(figure for name in names for figure in PAPERS[name].figures))
    if figures:
        run([sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"), "--only", *figures])
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
            if spec.figures:
                figure_downloads = DOWNLOADS / "figures"
                figure_downloads.mkdir(parents=True, exist_ok=True)
                for figure in spec.figures:
                    shutil.copyfile(FIGURES / f"{figure}.png",
                                    figure_downloads / f"{figure}.png")
            build_web(spec)
            source_archive(name, spec)
            if spec.bibliography:
                arxiv_archive(name, spec)
        print(f"Built {name} from paper/{spec.source}")


if __name__ == "__main__":
    main()
