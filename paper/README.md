# Private LLM Inference

A two column research draft by Blair Hudson. `manuscript.md` is the canonical
Pandoc Markdown source. `paper.lua` preserves compact tables and numbered
equations in the PDF while keeping the web edition readable. Recorded numerical
evidence is under `../research/evidence/`.

```bash
uv run --no-project --python 3.13 python ../scripts/build_paper.py
```

From the repository root, `make paper` does the same. Then run
`uv run --no-project --python 3.13 python scripts/prepare_docs.py` to update site
downloads.

Install Pandoc and either Tectonic or a TeX Live distribution. The GitHub
workflow uses TeX Live; the build script prefers Tectonic when both are present.
The document is a research draft; no arXiv identifier or acceptance is implied.

The application runtime and lifecycle experiments are separate implementations.
The paper retains that distinction. No new inference benchmark is claimed by
this repository packaging revision.

## Native runtime migration

The manuscript records the earlier measured Python/C++ implementation. The current
Rust source and its validation status are described in [IMPLEMENTATION-STATUS.md](IMPLEMENTATION-STATUS.md).
Historical timings are not Rust benchmarks.
