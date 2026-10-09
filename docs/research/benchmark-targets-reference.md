# Benchmark targets and frontier propagation

The installed SDK now uses **benchmark targets** for bounded work and the same
runner for complete response scenarios. `BenchmarkTarget`, `BenchmarkCase`,
`BenchmarkSuite`, `Sampling`, `BuildRef` and `TargetBudget` are public in
`pllm.metrics`; `FrontierStudy` and `Variant` are public in `pllm.search`.
[SDK usage](../content/docs/sdk/evaluate/benchmark-targets.mdx) and
[`examples/benchmark_targets.py`](../../examples/benchmark_targets.py) exercise
the public API. The broader [design](../plans/benchmark-frontier-and-probes.md)
and [five challenges](../plans/frontier-tenfold-challenges.md) retain their gates.

## Shared execution and evidence

The runner balances independent processes within one privately salted cohort.
Each worker has bounded inputs, resource admission, a deadline, temporary owners
and process-tree cleanup. Checkpoints preserve completed and failed trials without
resuming one-use state. Build inspection happens outside the measured target;
Python/native files, packaged configuration, numerical backends, dependency and
public CPU/memory identities are locked. Hostnames and device IDs are excluded.
An editable implementation changed during a run invalidates
its completed measurements.

Targets reuse native semantic scheduling, selected-stage import, paged snapshots,
mask expansion, residue codecs, ordinary local attention, operator benchmarks and
the ordinary response driver. Synthetic weights carry separate provenance from
checkpoint values. Independent integer checks run after timed stage work; the
reported target high-water mark is captured before those checks. Timings retain
conversion and reconstruction rather than pricing only matrix multiplication.

`BenchmarkResult` v2 extends the existing registry and leaves v1 interpretation
unchanged. Measured, deterministic, estimated, lower-bound and unavailable metrics
have distinct origins. CPU, memory, traffic, phase, owner, sample count and
denominator contracts must agree before comparison. Whole-response targets expose
client RSS/CPU, Preparation CPU/RSS, online TTFT, N−1 decode throughput and all-link
TCP-stream bodies; missing counters remain null. Covered application bodies have a
separate metric. Prompts, KV, keys and the private cohort salt are absent from
serialized configurations, records and exception strings.

## Frontier propagation

A study binds exact public configurations and builds. It expands declared
variants, interactions and leave-one-change-out ablations. Invalid compositions
remain visible. Coverage needs every process replicate under one matching cohort;
separate partial cohorts cannot be spliced into a measurement. Missing candidates
rerun their control in the new cohort, including when the old control was covered.
Trial limits count actual process executions.

Changed-path hints prioritize work; unknown paths affect every target. Hints never
authorize inheriting old performance. The measured snapshot preserves unknown
metrics, rejected cases and Pareto tradeoffs. A separate confirmation cohort can be
attached, with at least three independent process pairs. Response/operator inputs
and external source files require fresh revalidation: matching build/spec metadata
cannot certify unchanged private values or checkpoint contents. Self-contained
synthetic fixtures can close exact-build coverage. Snapshots are exploratory
evidence, not automatically selected application defaults or confidence intervals.

## What the five challenges test

The screen resolves only the pinned Qwen2.5 public configuration. Selected QKV,
gate/up and down stages use synthetic signed-i8 weights at 1, 16 and 150 rows.
Private mask material is fresh for each sample. State fixtures use native full-KV
geometry, deterministic synthetic contents and the ordinary full-layer attention
operation. They do not inherit real-checkpoint fidelity.

1. **Client arena:** a fresh client-import child tests the fixed runtime budget
   before an arena implementation. Its observed footprint is a veto for an
   unchanged Python envelope, not a lower bound on every redesigned client.
2. **Correction stripes:** the Preparation import floor and bounded mask-row
   chunks test resource feasibility. Row chunking does not implement the proposed
   output-stripe assembly, push/ack or transactional READY protocol.
3. **Digit GEMM:** four/eight-bit balanced decomposition executes exact CPU float32
   products and ring reconstruction. Conservative inner blocks prove every possible
   partial sum is below `2**24`, including signed-i8 −128. Full-precision CPU BLAS,
   conversion, correction encoding and output-file writing are charged. The
   `digit × row-chunk` cells test batching loss explicitly.
4. **Layer-swapped KV:** the bounded sealed-file reference authenticates source,
   state geometry and revision, loads complete layers, and preserves checked local
   attention bits. One/two-slab projections separately expose the runtime-floor
   interaction. This is not a compiler-bound transactional decoder owner.
5. **Persistent continuation:** a fresh child verifies synthetic sealed state
   under a separately retained trusted key/revision owner. Restart horizon
   arithmetic charges bootstrap, publication and resumed work. The reference has
   no SDK continuation credential, logits, pending-token provenance, verifier
   lineage or reusable preparation material.

## Measured screen

The corrected installed-wheel cohort completes **41 cases in 123 fresh processes**,
three processes per case. Independent integer oracles check **30,832,128 correction
values**; synthetic attention and restored-state checks cover another **2,377,728
values**. Runtime and geometry checks are counted separately. All cases complete,
and the study's exact-build coverage has no remaining cases to schedule. Completing
a case means its execution contract passed, not that its performance target passed.

Raw canonical evidence is retained in
`docs/evidence/research-benchmark-targets.json`; its `.frontier.json` and
`.coverage.json` companions derive from that same cohort. The installed build digest
is `21c6d6c9025bd80e748c8ae233e8ea09f3935ce3ad9b7714779ec6fd3598fecc`.
Results below are process medians in decimal MB. They are exploratory bounded
measurements, without a separate finalist confirmation or whole-response candidate.

| Challenge | Observation | Decision |
| --- | --- | --- |
| Client arena | Import-only client peak **56.51 MB**, against the historical **22.585 MB** tenfold budget. | An arena inside the unchanged Python envelope cannot meet this budget; a genuinely thinner runtime is prerequisite. |
| Correction stripes | Preparation import alone peaks at **72.86 MB**, above the **19.826 MB** budget. Four-row mask chunks reduce the 150-row gate/up stage peak **116.34 → 91.87 MB**, but raise CPU **0.18309 → 0.19120 s**. | Local chunking saves memory, but this is neither tenfold RAM nor acknowledged output-stripe transport. |
| Exact digit GEMM | Eight-bit digits at 150 rows reduce gate/up CPU **0.18309 → 0.06851 s (62.58%)**, with peak RSS **116.34 → 165.89 MB**. | Exact large-batch improvement; small batches regress, and no cold-first Preparation speedup is established. |
| Layer-swapped KV | At 2,048 synthetic tokens, peak RSS **108.15 → 72.43 MB (33.03%)**; attention CPU **0.01328 → 0.09386 s (7.07×)**. | Local memory/compute tradeoff; not tenfold client RAM or a transactional decoder owner. |
| Persistent continuation | A fresh reader verifies 3.15 MB of sealed synthetic state; restore costs **0.1414 CPU-s / 0.1743 s elapsed**, with publication separately **0.0413 CPU-s**. | Restart storage mechanics pass; response credentials, fidelity and TTFT/traffic improvement remain unimplemented. |

Digit-GEMM batching matters. At 150 rows, eight-bit digits reduce QKV/down CPU
**33.03% / 56.80%**; at one or sixteen rows, all tested digit choices use more CPU
than native integer kernels. Combining eight-bit digits with four-row chunks uses
**0.61305 CPU-s**, versus **0.19120 s** for native four-row chunks: **3.21× more CPU**.
The combined peak is also higher, **117.96 versus 91.87 MB**. The study therefore
retains separate Pareto tradeoffs rather than promoting the independently best
kernel and memory choices as a combined winner.

The state screen's resident/sealed controls own the same **50.33 MB** synthetic KV.
Sealed publication costs **0.1898 CPU-s** and **50.34 MB** of private file output,
against **0.1087 CPU-s** to construct resident state. These prerequisites are outside
the timed attention call and remain explicit. The restart reader's median **39.68
MB** peak and its parent's **60.74 MB** peak are separate lifetimes; they are never
added into a measured simultaneous peak. Filesystem cache is unmeasured.

The serialized assumptions also expose two conditional gates:

- For 32,760-token geometry, a fixed **62 MB** runtime gives optimistic one/two-layer
  slab memory ratios of **9.08× / 6.72×**. Replacing it with a hypothetical **8 MB**
  runtime gives **19.57× / 10.83×** before other owners. The latter runtime does not
  exist in this screen; these are estimates, not measured client savings.
- With a historical **116.84 MB** fresh-response body comparator, **6 MB** initial
  publication and 32 resumptions, an assumed **8 MB** resumed response gives
  **10.18×** lifecycle body savings; the first tenfold horizon is 31 resumptions.
  Assuming **20 MB** instead gives **5.05×** and can never reach tenfold. Both charge
  the initial request. No resumed decoder or measured network result follows.

Earlier development screens, including the u32 dispatch failure and an outer
120-second interruption, are excluded. The interrupted checkpoint cannot satisfy
complete cohort coverage. None of these five screens establishes a tenfold serving
improvement or changes an SDK serving default.

## Design feedback

- **Target is the useful noun.** Stage, artifact, state and response scopes share
  lifecycle machinery; they do not need independent ad hoc "probe" runners.
- **Binding is not execution.** Native geometry can cheaply reject a candidate,
  while synthetic values, checkpoint values, local protocol arithmetic and role
  transport remain distinct evidence scopes. Unsupported transport returns a
  coverage gap instead of silently using a local substitute.
- **Owners belong in metrics.** Restart timing includes child CPU explicitly;
  the import-floor child excludes its measurement harness. Parent-only high-water
  marks never become synchronized multi-process client peaks.
- **Interactions are necessary.** Smaller row chunks can erase digit-GEMM gains;
  separate kernel and memory ratios cannot predict the combined case.
- **Projection assumptions need names.** Runtime floors, live slab counts and
  resumed-byte costs stay serialized assumptions, not inferred measurements.
- **Failures improve the target contract.** The first screen exposed u32 native
  dispatch through the modular API; the corrected target uses `wrap32`. Failed
  trials remain failed evidence and cannot close current-build coverage.
- **Coverage and promotion are different.** All 41 cases have current-build
  measurements, while both runtime budgets fail and the complete serving candidates
  remain unimplemented. A filled coverage matrix cannot imply a successful advance.
- **Coverage needs observable identities.** Public hardware joins the software
  lock; private inputs and mutable external source files trigger revalidation.
  A matching case name and build cannot certify bytes the planner has not checked.

The next executable adapters are output-stripe transport and transactional decoder
state. Promotion requires complete source/numeric/ownership binding plus matched
response costs. This historical screen's state and NumPy digit kernels remain
bounded research executors. A later Rust-owned digit implementation and maintained
serving study have separate evidence in
[the maintained prepared frontier](prepared-maintained-reference.md); these
historical measurements do not establish their performance or a tenfold result.

## Reproduce

From an installed wheel and a checkout containing the public study:

```bash
python examples/benchmark_targets.py --output target/stage-study.json
python -m benchmarks.research.frontier_targets \
  --output target/evidence/frontier-targets-screen.json
```

The Qwen screen expects the configuration for the pinned source in the local Hub
cache; it loads no checkpoint tensor values. Output paths must be fresh. Companion
`.frontier.json` and `.coverage.json` files derive from the same run. Each rerun
gets new material, cohort commitments and observed build identity.
Allow at least ten minutes for the outer launcher; each target retains its separate
300-second deadline. Interrupted outputs are retained but never resumed or combined
with a later cohort.
