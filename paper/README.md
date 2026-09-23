# PLLM papers

`manuscript.md` and `whitepaper.md` are the canonical sources. Pandoc produces
the PDFs and website MDX from those files; `header.tex` contains the small amount
of shared LaTeX needed by the two-column PDF layout.

## Build

Install Pandoc, `pdfinfo`, and either Tectonic or pdfLaTeX, then run from the
repository root:

```bash
uv run --no-project --python 3.13 python scripts/build_papers.py
```

Pass `paper` or `whitepaper` to build one document. Pass `--pdf-only` when
building an extracted source archive without the documentation tree.

The build enforces US Letter output and page limits of four pages for the
technical paper and three pages for the whitepaper. It writes PDFs under `paper/`,
copies downloadable PDFs and deterministic source archives to
`docs/public/downloads/`, and renders the website pages under
`docs/content/research/`.

The whitepaper includes three checked chart images under `paper/figures/`.
Rebuild those images after changing their data or design with
`uv run python scripts/render_paper_figures.py` (ImageMagick and fontconfig
required). Regular Pandoc and website builds use checked PNGs and need neither
graphics tool. Figure generator reads the pinned historical evidence JSON;
the source archive includes generator, SVGs, PNGs, and that evidence record.

## Scope boundary

The technical paper reports the implemented three-role offline-inventory path
and the retained Qwen2.5-0.5B CPU loopback study. It does not report WAN, GPU,
energy, price, concurrency, malicious-provider, or output-quality results.
