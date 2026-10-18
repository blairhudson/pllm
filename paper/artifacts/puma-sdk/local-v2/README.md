# PUMA quality integration and bounded wide contractions

This slice extends the installed SDK and its ordinary quality benchmark:

- Private dot products wider than 1,024 values accumulate exact ring64 chunks
  before one Q18 truncation. Native checks detect per-chunk rounding and reject
  complete accumulators outside the public range.
- Generated Qwen2/Qwen3/Llama decoders exercise a 4,864-wide MLP. Public source
  admission prices the largest full-prefix compiler view before value ingestion.
- `benchmark quality` executes fresh three-worker, one-selection requests on
  common caller-owned prefixes. Final token agreement is measurable; hidden
  top-k recall and logit error remain null. This preserves D9's disclosure rule.
- Generated SDK/CLI checks cover teacher forcing, budgets, failures and retirement.
  These controls qualify the diagnostic, not representative model quality.

## Evidence

`summary.json` binds the final installed-wheel tests, native checks, input hashes,
package-file checks and common-Qwen preflight. Counts describe this suite and
must not be added to the overlapping `local-v1` suite.

The final build is a guarded **cached local build**, not a clean release build.
Its receipt records zero sampled build swap growth; that observation excludes
test and serving memory. Wheel SHA-256:

```text
6f3dd906728ca71d78b22e1ab3dcfcdd26aa317f057652612f8f4ea7b8f0a938
```

Original receipts, source/observer hashes, dependency versions, JUnit results and
native logs accompany the summary. `inputs.tar.gz` preserves the checked build
and validation inputs, including concurrent native modules present in this build.
The supplemental fixture archive retains its separate scope. Earlier quality
attempts remain historical diagnostics; the final suite supplies acceptance.

The pinned common Qwen2.5 checkpoint rejects native source admission even with
a four-input/two-output declaration. It has 24 layers and 151,936 vocabulary
rows; its main token table alone contains 136,134,656 values. Current admission
allows four layers and 1,048,576 source values, with a separate 256-row SDK
tokenizer ceiling. `common-qwen/` retains the ordinary CLI command and rejection.
No tensor execution, checkpoint-quality score, serving cost or full-model memory
qualification follows. Compact complete-model ownership and public range proofs
remain required.

## Replay

Check the retained bytes from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

From the repository root, with the matching wheel and quality dependencies
installed:

```sh
python -I -m pytest -q -o pythonpath=. -o junit_family=xunit1 \
  tests/test_replicated_*.py tests/test_delegated_source.py \
  tests/test_reference_agreement.py tests/test_reference_benchmark.py \
  tests/test_streamed_reference.py tests/test_benchmark_coverage.py
cargo test --locked --offline -p pllm-compiler --lib replicated_decoder
```

The root supplies test helpers and inventory scripts; installed `pllm` supplies
execution. Package-file receipts verify that boundary. Clippy retains the same
two recorded unrelated lint allowances as the preceding SDK slice.
