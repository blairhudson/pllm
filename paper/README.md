# Executable BFV mask preparation study

`manuscript.md` is the canonical source for this historical public-weight
masking study. Results describe `research/lifecycle` version
`0.14.0+inference.study1`, not the current PLLM runtime. Numerical evidence is
under `../research/evidence/`; citations are maintained in `references.bib`.

## Build

From repository root:

```bash
uv run --no-project --python 3.13 python scripts/build_paper.py pdf
```

The `pdf` target emits all submission-facing files together:

* `paper/main.tex`: standalone Pandoc-generated LaTeX; do not edit directly.
* `paper/main.pdf`: compiled from that exact `main.tex`.
* `paper/build-metadata.json`: source hashes, Git revision, normalized timestamp,
  and tool versions.
* `paper/arxiv-source.tar.gz`: deterministic submission archive with a
  self-contained root `main.tex`, build note, and metadata. The build extracts
  the archive and compiles that copy as its final portability check.

The website article remains a separate generated target:

```bash
uv run --no-project --python 3.13 python scripts/build_paper.py web
```

With no target, the script builds both paper artifacts and website article.
Install Pandoc and either Tectonic or pdfLaTeX. Set `SOURCE_DATE_EPOCH` to an
integer Unix timestamp for a caller-selected normalized build time; otherwise
the script uses the current Git commit time.

## Scope boundary

The historical study used Python/C++ code and TenSEAL 0.3.17. Current mixed
Python/Rust implementation status is summarized in
[IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md). Historical timings are
not current Rust or serving benchmarks.

## Submission metadata

The manuscript lists Blair Hudson as draft author. Before submission, user must
choose final author list/order, affiliations and ORCIDs, primary/cross-list arXiv
categories, arXiv distribution license, and whether to reserve or later add a
DOI. Build does not guess these fields.
