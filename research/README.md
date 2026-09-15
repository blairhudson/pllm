# Research and evidence

Research follows the canonical [research and method standard](../design/research-standard.md).
Registry size and priority are curatorial metadata: methods advance independently through an
open-ended, evidence-gated lifecycle, and blocked or negative results remain part of the record.

`methods/` contains source-aware reproduction targets, source locks, cards, and citations;
`recipes/` contains planned workflows. `assurance/` contains finite formal specifications and public
negative-control fixtures; these are not production evidence. `evidence/` contains accepted or
historical scoped records. Clean-room references belong in `reference/<method-id>/` when present;
native implementations belong in semantic `../crates/` owners rather than paper-named runtime
packages. Upstream artifacts are external, pinned oracles and are never vendored here. Machine
record schemas live in `../schemas/`.

CLI and Python research surfaces described by the standard are target contracts unless their
implementation-status documentation says otherwise. Existing files do not imply generic lifecycle
commands, paper-specific runtime APIs, complete reproductions, native parity, or promotion.

`lifecycle/` contains the independent preparation and inference study. It has its own
`pyproject.toml`, test suite and executable entry points. Its recorded manuscript measurements are
historical data, not results newly produced by the repository packaging workflow.

The application server and the study do not share one fully integrated execution
backend. The study includes later coefficient preparation work that is not
advertised as the application provider's default engine.

Run the study separately:

```bash
cd research/lifecycle
uv sync --group test
uv run pytest
```

Tests containing real HE use the actual TenSEAL backend. Full stage benchmarks
can require substantial CPU, memory and preparation time. Read
`lifecycle/REPORT.md` before running model-sized experiments.

The measured large Qwen stages use synthetic matrices at actual model shapes.
The trained generation checkpoint is small. No complete 27B generation rate,
GPU performance, or general malicious security claim follows from these data.
