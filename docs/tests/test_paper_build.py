"""Exercise generated paper outputs as part of the docs test suite."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "docs/build/papers"
DOWNLOADS = ROOT / "docs/public/downloads"


class PaperBuildTests(unittest.TestCase):
    def test_web_citations_remain_markdown_links(self) -> None:
        web = (ROOT / "docs/content/research/paper.mdx").read_text()
        self.assertNotIn('class="uri"', web)
        self.assertIn("[www.usenix.org/conference/", web)
        self.assertRegex(web, r"```(?:[ \t]*text)?\nSpecification\n")

    def test_whitepaper_archive_contains_its_rebuild_inputs(self) -> None:
        with ZipFile(DOWNLOADS / "whitepaper-source.zip") as archive:
            expected = {
                "paper/whitepaper.md",
                "paper/web.lua",
                "paper/web.template.md",
                "paper/whitepaper_print.lua",
                "paper/whitepaper_print.html",
                "paper/whitepaper_print.css",
                "docs/scripts/build-papers.py",
                "docs/scripts/render-paper-figures.py",
                "docs/evidence/current-runtime-2026-09-11.json",
            }
            for name in ("mechanics", "research-loop"):
                expected |= {
                    f"docs/build/papers/figures/{name}.{extension}"
                    for extension in ("svg", "png")
                }
            self.assertLessEqual(expected, set(archive.namelist()))
            for relative in expected:
                self.assertEqual(archive.read(relative), (ROOT / relative).read_bytes())

    def test_whitepaper_keeps_both_pages_and_final_claim_limits(self) -> None:
        self.assertEqual(
            (DOWNLOADS / "whitepaper.pdf").read_bytes(),
            (BUILD / "whitepaper.pdf").read_bytes(),
        )
        text = subprocess.check_output(
            ["pdftotext", "-layout", str(DOWNLOADS / "whitepaper.pdf"), "-"], text=True,
        )
        pages = [page for page in text.split("\f") if page.strip()]
        self.assertEqual(len(pages), 2)
        self.assertIn("Keep sensitive context", pages[0])
        self.assertIn("not collude", pages[0])
        self.assertIn("Progress is a scorecard", pages[1])
        self.assertIn("Client peak memory needs a fresh-process", pages[1])
        self.assertIn("266.91 MB", pages[1])
        self.assertIn("full physical wire", pages[1])
        layout = ET.fromstring(subprocess.check_output([
            "pdftotext", "-bbox-layout", str(DOWNLOADS / "whitepaper.pdf"), "-",
        ]))
        for index, page in enumerate(layout.findall(".//{*}page"), 1):
            body_bottom = []
            for line in page.findall(".//{*}line"):
                text = " ".join(word.text or "" for word in line.findall("{*}word"))
                if text not in {"PLLM / PRIVATE INFERENCE, MEASURED", f"{index:02d}"}:
                    body_bottom.append(float(line.attrib["yMax"]))
            self.assertLess(max(body_bottom), 738, f"Page {index} overlaps its footer")

    def test_arxiv_archive_compiles_without_repository_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pllm-arxiv-test-") as temporary:
            directory = Path(temporary)
            with ZipFile(DOWNLOADS / "paper-arxiv-source.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"main.tex", "README.txt", "LICENSE"})
                archive.extractall(directory)
            if engine := shutil.which("tectonic"):
                command = [engine, "main.tex"]
                passes = 1
            else:
                engine = shutil.which("pdflatex")
                self.assertIsNotNone(engine, "Install Tectonic or pdfLaTeX")
                command = [engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
                passes = 2
            for _ in range(passes):
                result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            info = subprocess.check_output(["pdfinfo", str(directory / "main.pdf")], text=True)
            pages = next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:"))
            self.assertLessEqual(int(pages), 5)

    def test_generated_outputs_are_untracked(self) -> None:
        output = subprocess.check_output([
            "git", "ls-files", "docs/build", "docs/public/downloads",
            "docs/content/research/paper.mdx", "docs/content/research/whitepaper.mdx",
            "paper/*.pdf", "paper/figures",
        ], cwd=ROOT, text=True)
        self.assertEqual(output, "", "Generated papers must not be committed")


if __name__ == "__main__":
    unittest.main()
