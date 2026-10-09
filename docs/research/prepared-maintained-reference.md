# Maintained prepared frontier and native digit preparation

This slice connects scoped targets to a repeatable serving study. Its reference
recipe is `examples/prepared_sdk.py`; exact case specifications, observed builds,
private cohort commitments, coverage gaps and all seven serving metrics travel
together. Historical evidence is never promoted to another build.

## Execution contracts

- `FrontierStudy.confirm` requires measured comparable finalists and creates a
  fresh control/finalist cohort with at least three process replicates. Its
  snapshot binds screen and confirmation, preserving both sets of tradeoffs.
- Runtime targets independently import Python, the SDK, NumPy, HTTPX, native
  kernels, the client, prepared execution and Preparation. Module roots explain
  entry-point loading; overlapping process footprints are not additive costs.
- Ordinary text clients now import JSON-schema validation only when strict tools
  request it. Strict tool schemas and generated arguments retain validation.
- `ModelAwareCorrections(kernel="digit8", min_rows=64)` binds an opt-in exact
  Preparation kernel and public dispatch threshold. Apple Silicon CPU Accelerate
  executes bounded SGEMM tiles. Weights and balanced base-256 digits have magnitude
  at most 128; 1,023-product blocks keep every product and possible partial sum
  below `2**24`. Rust reconstructs exact u16/u24/u32 residues using wrapping
  arithmetic. The native call releases the Python interpreter lock.
- Native digit tiles retain at most 64 MiB of explicit workspace. Binding copies,
  output/page overlap and platform-BLAS allowance bring conservative role
  admission to another 224 MiB per active Preparation stage. Paged weight files
  retain their authenticated immutable ownership. No persistent float weight
  snapshot is introduced. Small batches and unsupported shapes use existing
  integer kernels; the selected backend rejects unsupported platforms.
- Only CPU, unverified Preparation can select this kernel. Network offers lack
  its capability/workspace pricing, and reject it. This exact kernel changes
  neither one-use mask domains nor correction frames.

## Scoped measurements

The runtime, kernel and initial generated-serving screens use installed build
`9ac26c44087e4c4c483706d654625b4c377b019ef3f7ac9b70503d6bc34672d5`.
The runtime screen completes 30 fresh processes, the kernel screen 72, and the
generated-checkpoint serving screen ten. Each kernel/runtime case has three
process replicates; the tiny serving screen has one. These are separate cohorts.

### Client import floor

The fresh client-import child falls from **56.39 to 52.85 MB peak RSS (6.28%)**
and **0.21110 to 0.19109 CPU-s (9.47%)** against the previous installed build.
Preparation import remains essentially unchanged at **72.79 / 72.91 MB**.
Independent current-build entry points peak at 18.30 MB for Python, 18.32 MB
for the lazy SDK namespace, 30.33 MB for NumPy, 36.14 MB for HTTPX, 19.56 MB
for native kernels, and 39.27 MB for prepared execution. These footprints overlap;
adding them would double-count shared imports. Module-root inventories accompany
the measurements. This import-only improvement does not establish a complete
response memory reduction or satisfy the historical 22.585 MB tenfold budget.

Evidence: `docs/evidence/research-prepared-runtime.json`.

### Exact native digit kernel

Independent signed-integer oracles check **23,457,792 correction values**.
The table gives process medians for 150-row synthetic stages, with conversion,
mask expansion, correction encoding and file output included. CPU units are
seconds and memory uses decimal MB. Each paged candidate has its own paged
integer control.

| Semantic role | Storage | Integer CPU | Digit CPU | CPU reduction | Integer / digit peak RSS |
| --- | --- | ---: | ---: | ---: | ---: |
| QKV projection | Resident | 0.017296 | 0.007668 | 55.67% | 64.21 / 68.39 MB |
| QKV projection | Paged | 0.018250 | 0.007940 | 56.49% | 61.52 / 64.98 MB |
| Gate/up projection | Resident | 0.131149 | 0.040697 | 68.97% | 116.18 / 116.28 MB |
| Gate/up projection | Paged | 0.132176 | 0.041701 | 68.45% | 84.54 / 85.75 MB |
| Down projection | Resident | 0.074695 | 0.027142 | 63.66% | 93.85 / 97.86 MB |
| Down projection | Paged | 0.074548 | 0.030127 | 59.59% | 89.88 / 91.39 MB |

Sixteen-row results vary by shape: resident QKV CPU rises from 0.002906 to
0.003214 seconds, while resident gate/up falls from 0.014916 to 0.009064 seconds.
The isolated target explicitly forces the selected kernel; the serving component's
default 64-row threshold keeps these small batches on integer kernels. Synthetic
stage timing does not establish cold-first Preparation or complete-response gains.
The earlier NumPy digit screen retains its own implementation and measurements.

Evidence: `docs/evidence/research-prepared-native-kernels.json` and its
`.frontier.json` companion.

### Generated serving control

All ten ordinary SDK variants complete cold/repeat requests through separate
Preparation and Inference children. Comparable output commitments and all seven
serving metrics survive the shared runner. Separate installed-wheel tests check
correction-frame parity for resident and paged weights, plus complete tiny
logits/KV and gateway requests through paged digit roles. The tiny study validates
propagation and transport, not checkpoint performance.

Evidence: `docs/evidence/research-prepared-study-tiny.json` and its
`.frontier.json` companion.

### Real-checkpoint exploratory screen

After correcting the example's work window and atomic configuration updates,
all eleven pinned Qwen2.5-0.5B variants complete **150+8-token cold/repeat**
requests through ordinary separate-role SDK execution. These fresh clients use
shared native **100/40 Mbps** access limits and installed build
`15713efc57a7ff2949f6245923e28643935ed21691beb39d8f2f338a3c0f890d`.
Each candidate has one process sample. All output commitments match, ordinary
runtime/privacy checks pass, and every client observes zero host swap growth.

The table retains all seven serving axes. CPU and TTFT use seconds, RSS and
traffic use decimal MB. Client/Preparation CPU covers startup through the first
cold response; client RSS is the fresh process lifetime peak, and Preparation
RSS is sampled within request windows. TTFT and N−1 decode TPS are medians of
the two requests; stream traffic covers both requests plus measured startup.

| Variant | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prepared control | 239.16 | 20.515 | 16.029 | 215.81 | 21.755 | 3.761 | 496.046 |
| No reuse | 209.45 | 17.183 | 15.416 | 213.66 | 40.745 | 3.737 | 832.342 |
| Whole prefill | 233.01 | 20.699 | 15.514 | 210.53 | 27.455 | 3.716 | 495.352 |
| Serial issuance | 216.43 | 20.169 | 15.316 | 189.38 | 21.653 | 3.788 | 496.046 |
| Resident client | 497.71 | 20.697 | 15.236 | 210.86 | 22.092 | 4.105 | 496.046 |
| Resident Preparation | 234.42 | 20.346 | 14.177 | 579.01 | 21.798 | 3.771 | 496.046 |
| Uncached public encoding | 240.39 | 20.248 | 15.645 | 211.32 | 21.700 | 3.715 | 496.030 |
| Digit8 | 227.59 | 22.271 | 12.505 | 214.81 | 21.990 | 3.483 | 496.046 |
| Original tokenizer | 267.53 | 21.739 | 16.418 | 214.01 | 21.849 | 3.742 | 496.046 |
| No reuse + whole prefill | 217.63 | 18.810 | 15.885 | 213.88 | 52.477 | 3.755 | 830.954 |
| Serial + digit8 | 215.81 | 20.189 | 11.495 | 199.13 | 21.716 | 3.826 | 496.046 |

Digit8 lowers cold-first Preparation CPU **21.99%** against the windowed integer
control; serial+digit8 lowers it **24.95%** against serial integer and **28.28%**
against the windowed control. Complete cold requests take 107.049 seconds for
the control, 104.957 for digit8, 112.368 for serial integer and 108.571 for
serial+digit8. These smaller response effects cannot inherit the isolated
stage CPU ratios. Reuse cuts the repeat request from 73.882 to 10.402 seconds,
but the reusable prefix is exactly the preceding prompt.

First-use artifact publication belongs to the cold lifecycle; the second request
reuses the client's own object cache. Multi-fresh-client payback remains a distinct
workload. Paged weights and
the indexed tokenizer retain explicit RSS/compute tradeoffs. No tenfold result
or universal winning configuration follows from this screen.

Evidence: `docs/evidence/research-prepared-study-qwen25.json`, `.plan.json`
and `.frontier.json`. The corrected ten-case generated control is separately
retained as `docs/evidence/research-prepared-study-tiny-r2.json` with matching
plan/frontier files. Cached checkpoint distribution, public tokenizer publication
and delivery, physical wire, filesystem cache, representative quality and
independent operators remain outside these measurements.

### Independent finalist confirmation

`FrontierStudy.confirm` independently reruns the windowed control, serial
issuance, digit8 and serial+digit8 with **three fresh clients each**. All twelve
processes complete their cold/repeat sequence on the same installed build and
150+8-token workload as the screen. All output commitments match, all ordinary
checks pass, and every process observes zero swap growth. The table reports
medians across the three process records, preserving the seven scopes above.

| Variant | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prepared control | 224.80 | 20.533 | 15.834 | 211.75 | 21.755 | 3.722 | 496.046 |
| Serial issuance | 218.79 | 20.520 | 15.593 | 194.72 | 21.716 | 3.776 | 496.046 |
| Digit8 | 217.02 | 20.190 | 11.871 | 223.79 | 21.718 | 3.781 | 496.046 |
| Serial + digit8 | 217.17 | 20.336 | 11.614 | 192.99 | 21.816 | 3.773 | 496.046 |

Digit8 reduces cold-first Preparation CPU **25.03%** against the windowed
integer control, and **25.52%** within serial issuance. Its windowed cold request
median falls from **106.877 to 104.805 seconds (1.94%)**, while sampled Preparation
RSS rises **5.69%**. Aggregate cold-first CPU falls **7.27%**. Serial+digit8 lowers
Preparation CPU **26.66%** against the windowed control, but its **109.032-second**
cold request remains slower than that control; serial integer takes **112.616
seconds**. Repeat medians are 10.499, 10.872, 10.442 and 10.839 seconds in table
order. These results support the opt-in kernel's CPU benefit under this workload,
not a tenfold improvement or automatic selection across shapes and devices.

Covered application bodies are exactly **493,575,247 bytes** for every process.
TCP-stream counts differ by small control/framing amounts, so their rounded
agreement is not byte-identical wire evidence. The client RSS changes here are
kernel/concurrency ablations; they do not isolate lazy JSON-schema loading.
Three process replicates are not confidence intervals or representative quality.

Evidence: `docs/evidence/research-prepared-study-qwen25-confirm.json` and its
`.plan.json` and `.frontier.json` companions. The frontier retains both cohorts
and marks unconfirmed neighbors as missing in the confirmation cohort.

## Matched installed-build response memory

A separate six-client cohort alternates the prior and updated installed wheels,
with three fresh clients per build and one cold/repeat pair per client. Both use
the same maintained **Qwen2.5 150+8-token** recipe, native shared **100/40 Mbps**
caps, integer Preparation kernels and isolated client artifact caches. The public
checkpoint and indexed tokenizer are pre-positioned. This build ablation includes
lazy JSON-schema loading; it does not independently isolate that change from the
other code changes or combine its ratios with the digit8 cohort.

All twelve requests pass the runtime checks, with matching source, numeric, state,
workload and output commitments and **493,575,247 covered application-body bytes**
per cold/repeat pair. Every host guard reports zero observed swap growth.

| Priority metric, median across three fresh clients | Prior wheel | Updated wheel |
| --- | ---: | ---: |
| Client lifetime peak RSS, MB | 222.13 | 218.15 |
| Client cold-first CPU, s | 20.223 | 20.067 |
| Preparation cold-first CPU, s | 15.558 | 15.306 |
| Preparation sampled request-window peak RSS, MB | 217.74 | 210.68 |
| Cold / repeat online TTFT, s | 40.958 / 2.439 | 41.011 / 2.437 |
| Cold / repeat N−1 decode tokens/s | 3.770 / 3.783 | 3.769 / 3.813 |
| Cold+repeat all-link TCP streams, MB | 496.046394 | 496.046394 |

Client peak RSS falls **1.79%**. Complete cold requests remain nearly equal,
**106.658 → 106.731 seconds**, and repeat requests **10.367 → 10.414 seconds**.
Aggregate cold-first CPU is **59.972 → 59.224 seconds**. These three paired
samples establish a small scoped client-memory reduction, not a tenfold gain or
a causal speed benefit. Lifetime client RSS includes import and both requests;
Preparation RSS is sampled only inside requests. TCP-stream medians match, with
individual control-frame lengths differing by at most 17 bytes. Filesystem cache,
checkpoint/artifact distribution and physical wire remain outside the measurement.

Evidence: `docs/evidence/research-prepared-runtime-response.json`. Its prior build
is `21c6d6c9025bd80e748c8ae233e8ea09f3935ce3ad9b7714779ec6fd3598fecc`
and updated build is
`dc95903d89b6906fe927d6368fa51a1611b1e86faaeed7cedb2e56ff629a64a9`.
Cross-build comparison requires explicit `allow_build_changes=True`; all other
comparison commitments remain enforced. Replay with a retained prior installed
wheel in the control environment:

```sh
python -m benchmarks.research.prepared_kernel_targets --response \
  --control-python target/benchmark-targets-venv/bin/python \
  --artifacts target/prepared-frontier-artifacts \
  --output target/prepared-runtime-response.json
```

## SDK design feedback

- **Maintain a recipe, then measure neighbors.** The serving study imports the
  public SDK example instead of copying an older optimized configuration. Its
  explicit paging, reuse, chunking, issuance and kernel ablations retain rejected
  cases and interaction costs.
- **Serialized targets need public parameter aliases.** Reconstructing a generic
  Pipeline previously lost typed slot aliases, making valid ablations fail.
  Target updates now normalize those aliases, support explicit optional-component
  removal, and reject duplicate or overlapping paths.
- **Coupled changes must be atomic.** Reducing both issuance concurrency and its
  work budget previously validated an invalid intermediate component, depending
  on keyword order. Components, typed/generic Pipelines and nested Experiments
  now rebuild each owner once after sibling updates. Serialized ablations use
  that same configuration contract.
- **Interpreter identity includes virtualenv ownership.** Relative launch paths
  are made absolute before isolated workers change directory. Resolving symlinks
  would accidentally select the base interpreter instead of the installed wheel.
- **Confirmation is executable work.** A nominated measured case schedules fresh
  control/finalist processes; a screen's Pareto membership is not confirmation.
- **Whole-response targets bound isolated wins.** Confirmation separates repeatable
  Preparation CPU savings from small latency gains and higher workspace RSS. The
  installed-build comparison narrows a roughly 6% import-memory change to a 1.79%
  whole-client sample.
- **Import roots localize fixed costs.** Deferred schema validation trims one
  dependency branch, while the remaining client floor still blocks a tenfold
  memory result inside this Python runtime.
- **Tiny execution misses scale gates.** The first real-checkpoint screen's
  concurrent variants rejected the example's 16 MiB work window: a 157-row
  Qwen2.5 gate/up job requires 38,909,952 bytes. Only its two serial variants
  completed, so it has no comparable frontier. The example now explicitly
  declares the supported 64 MiB window; serial ablations restore the required
  16 MiB serial setting. The original record remains in
  `docs/evidence/research-prepared-study-window-rejected.json`. Generic exception
  class codes preserve privacy but poorly localize this failure; geometry-first
  capacity diagnostics would improve the target API.

## Validation

The final installed wheel passes **175 tests** with `uv.lock` dependencies; one
optional Metal test skips because MLX is absent. Coverage includes exact native
corrections, resident/paged dispatch, shape fallback, child-role SDK logits/KV,
gateway requests, configuration replay, frontier confirmation and memory pricing.
Rust arithmetic and compiler-admission tests, release Clippy, formatting, Python
lint, 90 documentation checks and the production documentation build also pass.

The original ten-second pressure-abort retirement regression remains unchanged.
It passes with locked dependencies but fails with newer dependencies on both the
prior and updated wheels. That existing shutdown issue remains open; successful
normal-response measurements do not establish pressure-shutdown correctness.

## Reproduction

Use an installed release wheel and a checkout containing the public study files.
Each output path must be fresh. All measured children run isolated from checkout
imports and share one parent-owned cohort salt per run.

```bash
python -m benchmarks.research.prepared_kernel_targets --runtime \
  --output target/evidence/prepared-runtime.json
python -m benchmarks.research.prepared_kernel_targets \
  --output target/evidence/prepared-native-kernels.json
python -m scripts.benchmark_prepared_frontier --tiny \
  --output target/evidence/prepared-study-tiny.json
```

The kernel study uses the pinned Qwen2.5 configuration already in the local Hub
cache, synthetic signed-i8 weights, three semantic projection roles and 16/150
rows. Resident/paged and integer/digit interactions share independent signed-i64
oracles. These targets execute mask expansion, matrix products and correction
file output; they do not execute provider push/ack or checkpoint numerics.

For checkpoint responses, `--artifacts target/prepared-frontier-artifacts` reads
the existing `indexed.json` source/tokenizer declaration and private `prompt.txt`.
Publication and pre-positioned public artifacts retain their separate costs.
The maintained study starts fresh role/client-cache owners, runs cold/repeat
requests under shared native 100/40 Mbps caps, and emits `.plan.json` and
`.frontier.json` companions. Inputs never appear in those reports.

`--confirm SCREEN --nominate CASE` independently measures finalists and their
control. Failed admission, unknown metrics, changed outputs and incomplete
cohorts remain explicit gaps rather than candidate wins.

```bash
python -m scripts.benchmark_prepared_frontier \
  --confirm target/evidence/prepared-study-qwen25.json \
  --nominate prepared/digit8 --nominate prepared/serial-issuance \
  --nominate prepared/serial-issuance+digit8 \
  --output target/evidence/prepared-study-qwen25-confirm.json
```

The same maintained response target can isolate a runtime-build change:

```bash
python -m benchmarks.research.prepared_kernel_targets --response \
  --control-python /path/to/previous-wheel/bin/python \
  --output target/evidence/prepared-runtime-response.json
```

This runs identical cold/repeat Experiments on both installed builds, with three
fresh clients each. `BenchmarkRun.compare(..., allow_build_changes=True)` makes
the build difference explicit while preserving source, numeric, workload, output
and measurement checks. It does not isolate individual changes inside a build.
