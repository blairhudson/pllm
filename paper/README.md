# PLLM papers

The canonical sources and their roles are:

| Source | Role | Build target |
| --- | --- | --- |
| `manuscript.md` | Framework and benchmark paper | `paper` |
| `low-memory.md` | Low-tier coupled-boundary method paper | `low-memory` |
| `whitepaper.md` | Accessible two-page introduction | `whitepaper` |

The technical manuscripts are working drafts, not completed publications.
`PUBLICATION-PLAN.md` saves the complete strategy, scaffolds, critique and
milestones. `CLAIMS-AND-EVIDENCE.md` maps placeholders to existing evidence and
remaining work; `IMPLEMENTATION-STATUS.md` records progress.
`WRITING-GUIDE.md` defines the ASD-STE100 Issue 9 rules and technical terms for all three documents.

Publication evidence has separate, auditable owners:

- `COMPARISON-CONTRACT.md` and `comparison-contract.v1.json`: normative definitions,
  resource profiles, statistical design and explicit capability gaps.
- `PRIOR-ART-REVIEW.md` and `prior-art-sources.json`: dated primary-source review,
  source hashes and closest-system qualification queue.
- `COMPETITOR-QUALIFICATION.md`: pinned author implementation, exact construction
  mapping, adaptation tracks and outstanding complete-decoder comparison gates.
- `SECURITY-AND-CORRECTNESS.md`: exactness and a conditional static, untimed
  four-provider argument, with bounded witnesses in `artifacts/security/`.
- `IMPLEMENTATION-EVIDENCE.md` and `artifacts/conformance/`: frozen installed-wheel
  execution, independent oracles, failed attempts and local build provenance.
- `artifacts/release/`: separate empty-target build and fresh-environment generated
  replay, with a distinct native identity and retained failed setup attempts.
- `artifacts/clean-build/`: a separately traced empty-target build and fresh
  installation used by the full generated SDK and six-family conformance replays.
- `artifacts/listing1/`: exact published Qwen listing through SDK and ASGI gateway.
- `HISTORICAL-RESULTS.md` and `artifacts/historical-results-v1.json`: four original
  cohorts with raw replicate identities, separate windows and unknowns retained.

Read-only evidence checks, from the repository root:

```bash
node scripts/audit_publication_evidence.mjs --check
python3 paper/artifacts/conformance/verify.py
python3 paper/artifacts/listing1/verify.py
```

The table auditor's `--write` regenerates its JSON and historical Markdown;
it validates the manuscript table strings rather than rewriting manuscript prose.
Security witnesses run with `.venv/bin/python -m pytest -q
tests/test_paper_boundary_security.py`; the artifact README records their scope.

Pandoc produces website articles and both technical PDFs. The whitepaper uses Pandoc plus
its dedicated HTML/CSS print layout, rendered to PDF by headless Chrome or
Chromium. `header.tex` and `pdf.lua` remain technical-paper-only.

## Build

Install UV, Pandoc, Poppler (`pdfinfo`, `pdftotext`), ImageMagick, fontconfig,
Chrome or Chromium, and either Tectonic or pdfLaTeX. Run from the repository root:

```bash
npm --prefix docs run papers
```

Pass `-- paper`, `-- low-memory`, or `-- whitepaper` to build one document. Use `-- --browser PATH` to
select a browser binary, `--pdf-engine PATH` to select the technical PDF engine,
or `--pdf-only` when building an extracted source archive without the docs tree.
An extracted archive can run its included builder directly:
`python docs/scripts/build-papers.py whitepaper --pdf-only`.

The build enforces US Letter output with total-document guards of 20 pages for
the framework paper, 16 for the method paper, and two for the whitepaper. The
technical limits include references/appendices and are not arXiv requirements;
main-text writing targets are 10–12 and 8–10 pages respectively. It writes PDFs under `docs/build/papers/`,
copies downloadable PDFs and deterministic source archives to
`docs/public/downloads/`, and renders the website pages under
`docs/content/research/`. All these outputs are ignored by Git. The ordinary
`npm --prefix docs run build` regenerates papers and evidence downloads before
building the site; no prebuilt paper assets are needed in a clean checkout.
Missing citations and other Pandoc warnings fail the build.

Each technical build also exports `docs/public/downloads/<target>-arxiv-source.zip`,
where `<target>` is `paper` or `low-memory`. Each contains standalone `main.tex` with resolved references and the complete
header embedded. Extract it and compile with pdfLaTeX twice; Pandoc, Python,
BibTeX, and repository-relative assets are unnecessary for that archive.
Buildability does not establish submission readiness. Novelty, evidence,
submission metadata, license approval and arXiv category selection remain open
until their gates and author review complete. Paper 2 currently cites Paper 1 as
a working manuscript; no arXiv identifier has been invented.

The matching `<target>-source.zip` retains Markdown and rebuild inputs. To rebuild
an extracted technical source archive, run its included builder with its own
target, for example `python docs/scripts/build-papers.py low-memory --pdf-only`.

The whitepaper's two diagrams come from `docs/scripts/render-paper-figures.py`.
The paper build renders SVGs and PNGs into `docs/build/papers/figures/` and copies
the published PNGs into `docs/public/downloads/figures/`. The generator also
preserves the older baseline chart from pinned historical evidence; that chart
is not used in the current whitepaper. Source archives include rebuild inputs
and generated figure snapshots for distribution, without committing those outputs.

## Scope boundary

Paper 1 introduces private multi-party inference, the need for comparable
execution/evidence contracts, client-memory tiers, and independent mechanism
implementations. Its older WAN, CPU, paging and quality case studies remain in a
historical appendix with their original scope. New matched findings remain gated.

Paper 2 studies coupled private lookup and full-head sharing under Low's 256 MiB
ceiling. Communication is primary; complete-request latency is secondary. The
conditional abstract composition argument is written; concrete primitive review,
current-system measurements, provider-capacity normalization and independent
confirmation remain required. Historical cold-request gains do not establish a
warm/source-cold win, lowest memory, or a field-wide SOTA result.

The whitepaper remains a short introduction. No paper build supplies independent
operators, Internet performance, broad quality, physical wire or an unmeasured
tenfold gain. Generated PDFs, archives, figures and MDX remain untracked.
