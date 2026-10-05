# PLLM papers

`manuscript.md` and `whitepaper.md` are the canonical sources. Pandoc produces
both website articles and the technical PDF. The whitepaper uses Pandoc plus
its dedicated HTML/CSS print layout, rendered to PDF by headless Chrome or
Chromium. `header.tex` and `pdf.lua` remain technical-paper-only.

## Build

Install UV, Pandoc, Poppler (`pdfinfo`, `pdftotext`), ImageMagick, fontconfig,
Chrome or Chromium, and either Tectonic or pdfLaTeX. Run from the repository root:

```bash
npm --prefix docs run papers
```

Pass `-- paper` or `-- whitepaper` to build one document. Use `-- --browser PATH` to
select a browser binary, `--pdf-engine PATH` to select the technical PDF engine,
or `--pdf-only` when building an extracted source archive without the docs tree.
An extracted archive can run its included builder directly:
`python docs/scripts/build-papers.py whitepaper --pdf-only`.

The build enforces US Letter output and page limits of four pages for the
technical paper and two pages for the whitepaper. It writes PDFs under `docs/build/papers/`,
copies downloadable PDFs and deterministic source archives to
`docs/public/downloads/`, and renders the website pages under
`docs/content/research/`. All these outputs are ignored by Git. The ordinary
`npm --prefix docs run build` regenerates papers and evidence downloads before
building the site; no prebuilt paper assets are needed in a clean checkout.
Missing citations and other Pandoc warnings fail the build.

The technical build also exports `docs/public/downloads/paper-arxiv-source.zip`.
It contains standalone `main.tex` with resolved references and the complete
header embedded. Extract it and compile with pdfLaTeX twice; Pandoc, Python,
BibTeX, and repository-relative assets are unnecessary for that archive.
Submission metadata and arXiv category selection remain author responsibilities.

The whitepaper's two diagrams come from `docs/scripts/render-paper-figures.py`.
The paper build renders SVGs and PNGs into `docs/build/papers/figures/` and copies
the published PNGs into `docs/public/downloads/figures/`. The generator also
preserves the older baseline chart from pinned historical evidence; that chart
is not used in the current whitepaper. Source archives include rebuild inputs
and generated figure snapshots for distribution, without committing those outputs.

## Scope boundary

The technical paper introduces private inference, the prepared and two-worker
controls, and an evidence-bound workflow for external research agents. Its
distinct case studies cover emulated WAN scheduling, cold aggregate CPU, and
Qwen3-4B client/Preparation memory. The whitepaper explains the same motivation
and findings for a business/technical audience. Figures and reduction factors
retain their measurement scopes; these studies do not establish independent
providers, Internet performance, broad quality, autonomous discovery rates, or
tenfold whole-system improvement.
