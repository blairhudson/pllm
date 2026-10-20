# Repository workflow

- Build wheels with standard uv/Maturin commands and normal tool options. See
  `CONTRIBUTING.md`; do not add bespoke build wrappers or attestation drivers.
- Experiment evidence is a reusable `.py` experiment plus the ordinary benchmark
  output. Use normal tests for correctness.
- Do not add receipt collections, copied source trees, environment dumps, nested
  evidence archives, or custom archive/replay verifiers as completion gates.
- Keep build products and temporary diagnostics in ignored `target/` or `dist/`.
  Use Git history for old source and change history.
- Simplify legacy evidence machinery when touching it. Preserve useful experiment
  definitions, benchmark results, numerical contracts, and regression tests.

## Reusable implementation

- Before implementing a paper, audit existing primitives, components, capabilities,
  source/import owners, scheduling, transport, and admission machinery.
- Reuse or extend shared model-/method-neutral contracts. Extract shared code when
  needed instead of adding parallel paper-specific implementations.
- Each paper should advance the reusable architecture. Keep its distinct numeric,
  trust, verification, and lifecycle contracts explicit and independently checked.
