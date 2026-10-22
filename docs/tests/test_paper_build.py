"""Exercise generated paper outputs as part of the docs test suite."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "docs/build/papers"
DOWNLOADS = ROOT / "docs/public/downloads"


class PaperBuildTests(unittest.TestCase):
    def test_abstracts_use_full_width_and_body_typography(self) -> None:
        for name in ("paper", "low-memory"):
            with self.subTest(paper=name):
                layout = ET.fromstring(subprocess.check_output([
                    "pdftohtml", "-xml", "-stdout", "-zoom", "1", "-i",
                    "-fontfullname", "-f", "1", "-l", "1",
                    str(DOWNLOADS / f"{name}.pdf"),
                ]))
                page = layout.find("page")
                self.assertIsNotNone(page)
                fonts = {font.attrib["id"]: font.attrib for font in page.findall("fontspec")}
                rows = [(line, "".join(line.itertext())) for line in page.findall("text")]
                start = next(i for i, (_, text) in enumerate(rows) if text == "Abstract") + 1
                end = next(i for i, (_, text) in enumerate(rows)
                           if text.startswith("This manuscript is a draft."))
                abstract = [line for line, _ in rows[start:end]]
                introduction = next(i for i, (_, text) in enumerate(rows)
                                    if text == "1 Introduction")
                # Use the first body row after the heading, not mutable prose.
                body = rows[introduction + 1][0]
                self.assertGreater(float(body.attrib["top"]),
                                   float(rows[introduction][0].attrib["top"]))
                body_font = fonts[body.attrib["font"]]
                self.assertTrue(abstract)
                for line in abstract:
                    font = fonts[line.attrib["font"]]
                    self.assertEqual((font["family"], font["size"]),
                                     (body_font["family"], body_font["size"]))
                left = min(float(line.attrib["left"]) for line in abstract)
                right = max(float(line.attrib["left"]) + float(line.attrib["width"])
                            for line in abstract)
                body_right = max(float(line.attrib["left"]) + float(line.attrib["width"])
                                 for line, _ in rows[end:])
                self.assertAlmostEqual(left, float(body.attrib["left"]), delta=1.5)
                self.assertGreater(right - left, 0.95 * (body_right - left))
                self.assertLess(
                    max(float(line.attrib["top"]) + float(line.attrib["height"])
                        for line in abstract),
                    float(rows[end][0].attrib["top"]),
                    "Full-width abstract must not overlap either body column",
                )

    def test_latex_identifiers_wrap_without_changing_web_semantics(self) -> None:
        paths = (
            "crates/pllm-garble/src/private_pages.rs::Table::Vocabulary",
            "python/pllm/runtime/benchmark_memory.py::private_boundary_memory",
            "benchmarks/research/private_boundary_decode_confirmation.py::matched_suite",
            "docs/evidence/private-boundary-decode-confirmation-150x8.json",
            "Sampling.memory_bytes",
        )
        markdown = "\n\n".join(f"`{path}`" for path in paths) + "\n\n" + (
            "| Source |\n| --- |\n"
            "| [arXiv:2609.10264v1](https://arxiv.org/abs/2609.10264v1) |\n"
            "| DOI 10.14722/ndss.2025.230868 |\n"
        )

        def render(writer: str, filtered: bool = True) -> str:
            return subprocess.check_output(
                ["pandoc", "--from=markdown", f"--to={writer}",
                 *([f"--lua-filter={ROOT / 'paper/pdf.lua'}"] if filtered else [])],
                input=markdown, text=True, cwd=ROOT,
            )

        latex = render("latex")
        for path in paths:
            self.assertIn(r"\texttt{\nolinkurl{" + path + "}}", latex)
        self.assertIn(
            r"\href{https://arxiv.org/abs/2609.10264v1}{\nolinkurl{arXiv:2609.10264v1}}",
            latex,
        )
        self.assertIn(r"\nolinkurl{10.14722/ndss.2025.230868}", latex)
        for writer in ("html", "gfm"):
            with self.subTest(writer=writer):
                self.assertEqual(render(writer), render(writer, filtered=False))

    def test_web_citations_remain_markdown_links(self) -> None:
        for name in ("paper", "low-memory"):
            with self.subTest(paper=name):
                web = (ROOT / f"docs/content/research/{name}.mdx").read_text()
                self.assertNotIn('class="uri"', web)
                self.assertNotIn("[@", web, "Citations must be resolved")
                self.assertIn("[www.usenix.org/conference/", web)
                self.assertIn(f"/downloads/{name}-arxiv-source.zip", web)
        web = (ROOT / "docs/content/research/paper.mdx").read_text()
        self.assertRegex(web, r"```(?:[ \t]*text)?\nSpecification\n")

    def test_technical_archives_preserve_distinct_sources(self) -> None:
        for name, source in (("paper", "manuscript.md"), ("low-memory", "low-memory.md")):
            with self.subTest(paper=name), ZipFile(DOWNLOADS / f"{name}-source.zip") as archive:
                for relative in (f"paper/{source}", "paper/references.bib", "paper/header.tex",
                                 "paper/pdf.lua", "docs/scripts/build-papers.py"):
                    self.assertEqual(archive.read(relative), (ROOT / relative).read_bytes())
                other = "low-memory.md" if source == "manuscript.md" else "manuscript.md"
                self.assertNotIn(f"paper/{other}", archive.namelist())
            self.assertEqual((DOWNLOADS / f"{name}.pdf").read_bytes(),
                             (BUILD / f"{name}.pdf").read_bytes())

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
        page_text = [" ".join(page.split()) for page in pages]
        self.assertIn("Keep private context local", page_text[0])
        self.assertIn("Their private protocol information must stay separate", page_text[0])
        self.assertIn("Progress has several measures", page_text[1])
        self.assertIn("Comparisons of peak client memory with new processes are still necessary", page_text[1])
        self.assertIn("266.91 MB", pages[1])
        self.assertIn("not full physical-wire traffic or model downloads", page_text[1])
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
        for name, source, limit, marker in (
            ("paper", "manuscript.md", 20, "P1-D"),
            ("low-memory", "low-memory.md", 16,
             "A tenfold claim must use unrounded matched peak observations"),
        ):
            with self.subTest(paper=name), tempfile.TemporaryDirectory(
                prefix=f"pllm-{name}-arxiv-test-",
            ) as temporary:
                directory = Path(temporary)
                with ZipFile(DOWNLOADS / f"{name}-arxiv-source.zip") as archive:
                    expected = {"main.tex", "README.txt", "LICENSE"}
                    if name == "low-memory":
                        expected.add("docs/build/papers/figures/low-memory-history.png")
                    self.assertEqual(set(archive.namelist()), expected)
                    self.assertIn(f"paper/{source}", archive.read("README.txt").decode())
                    archive.extractall(directory)
                if engine := shutil.which("tectonic"):
                    command = [engine, "--only-cached", "main.tex"]
                    passes = 1
                else:
                    engine = shutil.which("pdflatex")
                    self.assertIsNotNone(engine, "Install Tectonic or pdfLaTeX")
                    command = [engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
                    passes = 2
                for _ in range(passes):
                    result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertNotRegex(
                        result.stdout + result.stderr,
                        r"(?:Overfull|Underfull) \\[hv]box|Invalid UTF-8 byte or sequence",
                        "Publication PDFs must compile without text overflow, spacing, or encoding warnings",
                    )
                info = subprocess.check_output(["pdfinfo", str(directory / "main.pdf")], text=True)
                pages = next(line.split(":")[1] for line in info.splitlines() if line.startswith("Pages:"))
                self.assertLessEqual(int(pages), limit)
                self.assertIn("612 x 792 pts", info)
                text = subprocess.check_output(["pdftotext", str(directory / "main.pdf"), "-"], text=True)
                self.assertIn(marker, " ".join(text.split()), "Draft evidence gates must survive publication")

    def test_generated_outputs_are_untracked(self) -> None:
        output = subprocess.check_output([
            "git", "ls-files", "docs/build", "docs/public/downloads",
            "docs/content/research/paper.mdx", "docs/content/research/whitepaper.mdx",
            "docs/content/research/low-memory.mdx",
            "paper/*.pdf", "paper/figures",
        ], cwd=ROOT, text=True)
        self.assertEqual(output, "", "Generated papers must not be committed")

    def test_historical_tables_keep_scope_captions_on_the_same_pdf_page(self) -> None:
        for name, row_marker, caption in (
            ("paper", "Serial digit8 issuance", "Original 150+8 tier cohort"),
            ("paper", "Offset seed-first", "Local link-emulation results (W1)"),
            ("low-memory", "230.56", "Retained source-present cohorts"),
        ):
            with self.subTest(paper=name, table=caption):
                text = subprocess.check_output(
                    ["pdftotext", "-layout", str(DOWNLOADS / f"{name}.pdf"), "-"],
                    text=True,
                )
                pages = [" ".join(page.split()) for page in text.split("\f")]
                self.assertTrue(
                    any(row_marker in page and caption in page for page in pages),
                    "A floating table must retain its evidence-scope caption",
                )


class PaperFigureIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("paper_builder", ROOT / "docs/scripts/build-papers.py")
        self.builder = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {spec.name: self.builder}):
            spec.loader.exec_module(self.builder)

    def test_low_memory_assets_reach_web_downloads_and_both_archives(self) -> None:
        (ROOT / "target").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "target", prefix="paper-figure-integration-") as temporary:
            directory = Path(temporary)
            downloads, web = directory / "downloads", directory / "web"
            pdf = directory / "fixture.pdf"
            pdf.write_bytes(b"PDF compilation is mocked in this asset-routing fixture")
            with (mock.patch.object(self.builder, "DOWNLOADS", downloads),
                  mock.patch.object(self.builder, "WEB_DIR", web),
                  mock.patch.object(self.builder, "pdf_engine", return_value="unused"),
                  mock.patch.object(self.builder, "browser_engine") as browser,
                  mock.patch.object(self.builder, "build_pdf", return_value=pdf),
                  mock.patch.object(self.builder, "run", wraps=self.builder.run) as run,
                  mock.patch.object(sys, "argv", ["build-papers.py", "low-memory"])):
                self.builder.main()
            browser.assert_not_called()
            self.assertEqual(run.call_args_list[0].args[0], [
                sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"),
                "--only", "low-memory-history",
            ])
            relative = "docs/build/papers/figures/low-memory-history.png"
            image = (ROOT / relative).read_bytes()
            self.assertEqual((downloads / "figures/low-memory-history.png").read_bytes(), image)
            self.assertIn('src="/downloads/figures/low-memory-history.png"',
                          (web / "low-memory.mdx").read_text())
            with ZipFile(downloads / "low-memory-source.zip") as archive:
                for path in (relative, relative.replace(".png", ".svg"),
                             "paper/artifacts/historical-results-v1.json",
                             "docs/scripts/render-paper-figures.py"):
                    self.assertEqual(archive.read(path), (ROOT / path).read_bytes())
                self.assertNotIn("paper/whitepaper.md", archive.namelist())
                self.assertNotIn("docs/evidence/current-runtime-2026-09-11.json", archive.namelist())
            with ZipFile(downloads / "low-memory-arxiv-source.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"main.tex", "README.txt", "LICENSE", relative})
                self.assertIn("{" + relative + "}", archive.read("main.tex").decode())
                self.assertEqual(archive.read(relative), image)

    def test_target_selection_keeps_other_paper_assets_and_page_guards(self) -> None:
        self.assertEqual({name: spec.max_pages for name, spec in self.builder.PAPERS.items()},
                         {"paper": 20, "low-memory": 16, "whitepaper": 2})
        for target, figures in (("paper", ()), ("low-memory", ("low-memory-history",)),
                                ("whitepaper", ("mechanics", "research-loop")),
                                ("all", ("low-memory-history", "mechanics", "research-loop"))):
            with (self.subTest(target=target),
                  mock.patch.object(self.builder, "pdf_engine", return_value="unused"),
                  mock.patch.object(self.builder, "browser_engine", return_value="unused"),
                  mock.patch.object(self.builder, "build_pdf"),
                  mock.patch.object(self.builder, "build_whitepaper_pdf"),
                  mock.patch.object(self.builder, "run") as run,
                  mock.patch.object(self.builder, "build_web") as web,
                  mock.patch.object(sys, "argv", ["build-papers.py", target, "--pdf-only"])):
                self.builder.main()
                if figures:
                    run.assert_called_once_with([
                        sys.executable, str(ROOT / "docs/scripts/render-paper-figures.py"),
                        "--only", *figures,
                    ])
                else:
                    run.assert_not_called()
                web.assert_not_called()


if __name__ == "__main__":
    unittest.main()
