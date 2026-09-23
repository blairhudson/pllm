from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "pllm_docs_regen", ROOT / "scripts/docs_regen.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regenerate_runs_reference_then_publication_generation(monkeypatch) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, *, cwd, check: calls.append((command, cwd, check)),
    )
    module.run(check=False)
    assert calls == [
        ([sys.executable, str(module.DEVELOPER_REFERENCE)], module.ROOT, True),
        (["npm", "run", "generate"], module.DOCS, True),
    ]


def test_check_mode_materializes_ignored_references_then_runs_freshness_gates(
    monkeypatch,
) -> None:
    module = _module()
    calls = []
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda command, *, cwd, check: calls.append((command, cwd, check)),
    )
    module.run(check=True)
    assert calls == [
        ([sys.executable, str(module.DEVELOPER_REFERENCE)], module.ROOT, True),
        (
            [sys.executable, str(module.DEVELOPER_REFERENCE), "--check"],
            module.ROOT,
            True,
        ),
        (["npm", "test"], module.DOCS, True),
        (["npm", "run", "check:content"], module.DOCS, True),
    ]


def test_whitepaper_archive_contains_its_figure_and_web_rebuild_inputs() -> None:
    with ZipFile(ROOT / "docs/public/downloads/whitepaper-source.zip") as archive:
        expected = {
            "paper/whitepaper.md",
            "paper/web.lua",
            "paper/web.template.md",
            "paper/whitepaper_print.lua",
            "paper/whitepaper_print.html",
            "paper/whitepaper_print.css",
            "scripts/build_papers.py",
            "scripts/render_paper_figures.py",
            "docs/evidence/current-runtime-2026-09-11.json",
        }
        for name in ("mechanics", "research-loop", "qwen-baseline"):
            expected |= {f"paper/figures/{name}.{extension}" for extension in ("svg", "png")}
        assert expected <= set(archive.namelist())
        for relative in expected:
            assert archive.read(relative) == (ROOT / relative).read_bytes()


def test_whitepaper_pdf_keeps_each_intentional_spread_and_final_caveat() -> None:
    assert (ROOT / "docs/public/downloads/whitepaper.pdf").read_bytes() == (
        ROOT / "paper/whitepaper.pdf"
    ).read_bytes()
    text = subprocess.check_output(
        ["pdftotext", "-layout", str(ROOT / "paper/whitepaper.pdf"), "-"], text=True
    )
    pages = [page for page in text.split("\f") if page.strip()]
    assert len(pages) == 3
    assert "Why PLLM" in pages[0] and "One private request" in pages[0]
    assert "not collude" in pages[0]
    assert "Research becomes a candidate capability" in pages[1]
    assert "Economic value needs an honest denominator" in pages[1]
    assert "Measured baseline; open comparison" in pages[2]
    assert "technical paper" in pages[2]
