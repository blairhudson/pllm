# Current offline-inventory study

`manuscript.md` is the canonical source for the current public-weight inference
paper. The measured runtime evidence is retained in
`../research/evidence/current-runtime-2026-09-11.json`; citations are maintained
in `references.bib`.

## Build

From the repository root:

```bash
uv run --no-project --python 3.13 python scripts/build_paper.py pdf
```

The `pdf` target emits all submission-facing files together:

- `paper/main.tex`: standalone Pandoc-generated LaTeX; do not edit directly.
- `paper/main.pdf`: compiled from that exact `main.tex`, with a four-page limit.
- `paper/build-metadata.json`: source hashes, Git revision, normalized timestamp,
  and tool versions.
- `paper/arxiv-source.tar.gz`: deterministic submission archive with a
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

The paper reports the implemented three-role offline-inventory path and one
retained Qwen2.5-0.5B CPU loopback study. It does not report WAN, GPU, energy,
price, concurrency, malicious-provider, or output-quality results. Historical
BFV artifacts remain under `../research/lifecycle` and are not evidence for the
paper's current-runtime measurements.

## Submission metadata

The manuscript lists Blair Hudson as draft author. Before submission, choose the
final author list/order, affiliations and ORCIDs, primary/cross-list arXiv
categories, arXiv distribution license, and whether to reserve or later add a
DOI. Build tooling does not guess these fields.
