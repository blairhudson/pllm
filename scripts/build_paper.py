"""Build PDF, arXiv source, and web editions from canonical Pandoc Markdown."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCE = PAPER / "manuscript.md"
BIBLIOGRAPHY = PAPER / "references.bib"
FILTER = PAPER / "paper.lua"
WEB_TEMPLATE = PAPER / "web.template.md"
TEX = PAPER / "main.tex"
PDF = PAPER / "main.pdf"
METADATA = PAPER / "build-metadata.json"
ARXIV_ARCHIVE = PAPER / "arxiv-source.tar.gz"
WEB = ROOT / "docs/content/docs/research/paper.mdx"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def capture(command: list[str]) -> str | None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode:
        return None
    return result.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_engine(requested: str | None) -> str:
    if requested:
        if shutil.which(requested) is None:
            raise SystemExit(f"PDF engine not found: {requested}")
        return requested
    for candidate in ("tectonic", "pdflatex"):
        if shutil.which(candidate):
            return candidate
    raise SystemExit("PDF build requires tectonic or pdflatex")


def source_date_epoch() -> int:
    configured = os.environ.get("SOURCE_DATE_EPOCH")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as error:
            raise SystemExit("SOURCE_DATE_EPOCH must be an integer") from error
        if value < 0:
            raise SystemExit("SOURCE_DATE_EPOCH must not be negative")
        return value

    committed = capture(["git", "show", "-s", "--format=%ct", "HEAD"])
    if committed and committed.isdigit():
        return int(committed)
    return int(SOURCE.stat().st_mtime)


def build_environment(epoch: int) -> dict[str, str]:
    env = os.environ.copy()
    env["SOURCE_DATE_EPOCH"] = str(epoch)
    env["TZ"] = "UTC"
    return env


def pandoc_command() -> list[str]:
    if shutil.which("pandoc") is None:
        raise SystemExit("Paper build requires pandoc")
    return [
        "pandoc",
        str(SOURCE),
        "--from=markdown+raw_tex+fenced_divs+link_attributes",
        "--standalone",
        "--number-sections",
        "--shift-heading-level-by=-1",
        "--citeproc",
        "--fail-if-warnings",
        f"--lua-filter={FILTER}",
        f"--resource-path={ROOT}",
    ]


def build_tex(env: dict[str, str]) -> None:
    run([*pandoc_command(), "--to=latex", f"--output={TEX}"], env=env)
    print(f"Built {TEX.relative_to(ROOT)}.")


def compile_tex(selected: str, env: dict[str, str], source: Path, output_dir: Path) -> None:
    if Path(selected).name == "tectonic":
        run(
            [selected, "--chatter=minimal", "--outdir", str(output_dir), str(source)],
            env=env,
        )
    else:
        command = [
            selected,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output_dir}",
            str(source),
        ]
        run(command, env=env)
        run(command, env=env)
        for suffix in (".aux", ".log", ".out", ".toc"):
            source.with_suffix(suffix).unlink(missing_ok=True)


def compile_pdf(selected: str, env: dict[str, str]) -> None:
    compile_tex(selected, env, TEX, PAPER)
    print(f"Built {PDF.relative_to(ROOT)} from main.tex with {Path(selected).name}.")


def first_line(command: list[str]) -> str:
    output = capture(command)
    return output.splitlines()[0] if output else "unavailable"


def write_metadata(selected: str, epoch: int) -> None:
    source_paths = (SOURCE, BIBLIOGRAPHY, FILTER, Path(__file__).resolve())
    archive_members = [
        "main.tex",
        "README.txt",
        "build-metadata.json",
    ]
    record = {
        "artifact": "historical executable BFV public-weight masking study",
        "canonical_source": SOURCE.relative_to(ROOT).as_posix(),
        "normalized_timestamp_utc": datetime.fromtimestamp(epoch, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "source_date_epoch": epoch,
        "source_revision": capture(["git", "rev-parse", "HEAD"]) or "unavailable",
        "source_revision_scope": (
            "Git HEAD is the base revision; SHA-256 values identify built working-tree sources"
        ),
        "sources": {
            path.relative_to(ROOT).as_posix(): {"sha256": sha256(path)}
            for path in source_paths
        },
        "generated": {
            TEX.relative_to(ROOT).as_posix(): {"sha256": sha256(TEX)},
            PDF.relative_to(ROOT).as_posix(): {"sha256": sha256(PDF)},
        },
        "toolchain": {
            "python": sys.version.split()[0],
            "pandoc": first_line(["pandoc", "--version"]),
            "pdf_engine": first_line([selected, "--version"]),
        },
        "arxiv_archive_members": archive_members,
    }
    METADATA.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"Built {METADATA.relative_to(ROOT)}.")


def build_arxiv_archive(epoch: int) -> None:
    members = (
        ("main.tex", TEX.read_bytes()),
        (
            "README.txt",
            (
                "PLLM arXiv source bundle\n\n"
                "Compile main.tex directly with pdfLaTeX. The bibliography and figures "
                "are embedded; no shell escape, network access, or repository files are "
                "required. This paper reports only the historical BFV lifecycle study "
                "identified in main.tex, not the current PLLM runtime.\n"
            ).encode(),
        ),
        ("build-metadata.json", METADATA.read_bytes()),
    )
    with ARXIV_ARCHIVE.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for name, data in members:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    info.mode = 0o644
                    info.mtime = epoch
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    archive.addfile(info, io.BytesIO(data))

    with tarfile.open(ARXIV_ARCHIVE, "r:gz") as archive:
        actual = archive.getnames()
    expected = [name for name, _ in members]
    if actual != expected:
        raise SystemExit(f"Unexpected arXiv archive members: {actual}")
    print(f"Built {ARXIV_ARCHIVE.relative_to(ROOT)} ({len(actual)} files).")


def verify_arxiv_archive(selected: str, env: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="pllm-arxiv-check-") as temporary:
        root = Path(temporary)
        with tarfile.open(ARXIV_ARCHIVE, "r:gz") as archive:
            archive.extractall(root, filter="data")
        compile_tex(selected, env, root / "main.tex", root)
    print("Verified extracted arXiv source bundle.")


def build_paper(engine: str | None) -> None:
    selected = pdf_engine(engine or os.environ.get("PANDOC_PDF_ENGINE"))
    epoch = source_date_epoch()
    env = build_environment(epoch)
    build_tex(env)
    compile_pdf(selected, env)
    write_metadata(selected, epoch)
    build_arxiv_archive(epoch)
    verify_arxiv_archive(selected, env)


def build_web() -> None:
    WEB.parent.mkdir(parents=True, exist_ok=True)
    env = build_environment(source_date_epoch())
    run(
        [
            *pandoc_command(),
            "--to=gfm",
            "--wrap=none",
            f"--template={WEB_TEMPLATE}",
            f"--output={WEB}",
        ],
        env=env,
    )
    article = re.sub(
        r"<(https?://[^>]+)>",
        lambda match: f"[{match.group(1)}]({match.group(1)})",
        WEB.read_text(),
    )
    WEB.write_text(article)
    print(f"Built {WEB.relative_to(ROOT)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=("all", "pdf", "arxiv", "web"),
        nargs="?",
        default="all",
        help="pdf and arxiv both emit main.tex, main.pdf, metadata, and source archive",
    )
    parser.add_argument("--pdf-engine")
    args = parser.parse_args()
    if args.target in ("all", "pdf", "arxiv"):
        build_paper(args.pdf_engine)
    if args.target in ("all", "web"):
        build_web()


if __name__ == "__main__":
    main()
