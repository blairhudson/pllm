# Historical executable BFV public-weight masking study

This directory is reproducible research artifact version
`0.14.0+inference.study1`, not current PLLM runtime or production release. It
completes BFV coefficient-matrix preparation through ciphertext reconstruction
and client decryption, measures synthetic public W4 matrices at selected
Qwen3.5-27B stage dimensions, and runs separate client/server processes for a
small trained decoder fixture.

Paper source is `../../paper/manuscript.md`; built PDF is
`../../paper/main.pdf`. Raw records are in `results/` and retained copies are in
`../evidence/`. Large-stage dimensions come from saved public configuration
analysis. No Qwen checkpoint was loaded or evaluated.

## Reproduce environment and tests

Run from repository root:

```bash
cd research/lifecycle
uv sync --frozen --group test
uv run --frozen python -m pytest -q
```

The retained `study-tests.xml` report records 45 tests, zero failures, zero
errors, and zero skips. This count covers this historical directory only, not
the current repository-wide suite.

The serialization bridge requires Zstandard. On macOS, install it with
`brew install zstd`; the bridge recognizes Apple Silicon and Intel Homebrew
locations.

## Lifecycle runs

Run commands from this directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python -m pllm_study.lifecycle --json results/reproduction-lifecycle.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --frozen python -m pllm_study.lifecycle --rtt-ms 10 --draft 4 --json results/reproduction-proposals.json
```

Each invocation trains deterministic four-block width-32 fixture for 160 steps,
starts separate server process, begins with empty preparation inventory,
prepares actual BFV correlations, processes 18-token prompt, and generates 96
token positions. It checks token equality against clear W4A4 graph. Ordinary
records also retain exact final-logit equality; proposal records do not retain
that check. Corpus is functional fixture, not language-quality dataset.

Six historical latency configurations were retained: ordinary and four-proposal
runs at 0, 1, and 10 ms injected delay. Each configuration has one complete run.
Delay is inserted once per online exchange inside application, not measured over
physical network. Therefore table cells are observations, not sample medians or
wide-area estimates.

## Cryptographic stage runs

Representative commands:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 uv run --frozen python -m pllm_study.benchmark --input 768 --output 256 --rounds 3 --json results/reproduction-768-256.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 uv run --frozen python -m pllm_study.benchmark --input 5120 --output 34816 --rounds 2 --json results/reproduction-expansion.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 uv run --frozen python -m pllm_study.online --json results/reproduction-online.json
uv run --frozen python -m pllm_study.head
uv run --frozen python -m pllm_study.compact --json results/reproduction-compact.json
uv run --frozen python -m pllm_study.recurrent --json results/reproduction-recurrent.json
uv run --frozen python -m pllm_study.summarize
```

Matched 768 by 256 control and coefficient paths retain three samples each.
Each of five full target-shaped preparation stages retains two samples. Online
non-head stage/batch combinations retain four timings, full online vocabulary
head combinations retain three, compact format retains three, W4A8 parameter
check retains one, and recurrent timing groups retain seven. Raw first-use
observations are not discarded.

Large matrices use synthetic signed four-bit values at target dimensions. Full
online head allocates about 1.27 GB int8 weights; allow at least 4 GiB system
memory and run large benchmarks sequentially. Full-head BFV preparation was not
executed: only 1,024 output rows were measured, then head preparation was
projected. Model planner outputs are configuration-derived or projected, not
measurements of loaded 27B model.

## Evidence boundary

Principal profile uses BFV polynomial degree 2,048, plaintext modulus 2,097,169,
and one 54-bit coefficient prime. SEAL accepts context under requested `tc128`
setting. This is library validation, not independent security analysis. No
rotations, bootstraps, relinearization, or ciphertext-ciphertext multiplication
occur in coordinate evaluator.

For fresh uniform mask, online masked value is uniform for fixed input. This
property alone is not end-to-end proof. Intended input-confidentiality argument
also relies on BFV, correct randomness, one-time use, and server following
protocol. Weights are public. Artifact does not provide model secrecy,
malicious-server correctness, hidden traffic metadata, or VM-snapshot rollback
resistance.

Frames use HMAC-SHA256 and sequence/epoch checks; tests reject alteration and
replay. Both processes burn correlations before reply. Remaining preparation is
discarded at demonstration session end. Transport authentication does not prove
that provider applied intended matrix.

## Layout

* `pllm_study/`: historical implementation and experiment entry points.
* `tests/`: arithmetic, serialization, lifecycle, replay, and transport tests.
* `results/`: raw timing, lifecycle, environment, and derived records.
* `reference/`: preserved prior coordinate implementation and provenance.
* `run-optimized.sh`: original sequential large-run order.
* `run-lifecycle.sh`: original lifecycle-run order.

Current public SDK, CLI, Rust kernels, and serving runtime live outside this
artifact and are not revalidated here.
