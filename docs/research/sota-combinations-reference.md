# Matched prepared-stack combinations

This screen tests interactions among installed serving components on one frozen
build. Independent historical savings are not added or multiplied. Exact cases
and build locks live in the saved `BenchmarkSuite`; the factories and driver are
`benchmarks/research/sota_combinations.py` and
`scripts/benchmark_sota_combinations.py`.

## Frozen five-way screen

The source is `Qwen/Qwen2.5-0.5B-Instruct` at
`7ae557604adf67be50417f59c2c2f167def9a775`, with separate 150- and 288-input-token
workloads and an eight-output-token allowance. Each fresh client executes a cold
request and its exact repeat. Natural EOS remains authoritative; an allowance is
not a measured output count.

| Choice | Difference from the maintained control |
| --- | --- |
| Control | `examples/prepared_sdk.py`, with the common 384-token input bound |
| Tiling | `MlpPrefillTiling(tile_rows=64)` |
| Digit8 | `ModelAwareCorrections(storage="paged", kernel="digit8", min_rows=64)` |
| Both | Tiling and digit8 |
| Both + placement | Both plus the frozen 38-group `ClientLinearRoles` selection |

All cases share W8A8 `prefix_f32`, paged client/Preparation weights, the indexed
tokenizer, compressed/batched artifact delivery, demand/on-demand inventory,
sixteen-row duplex prefill, exact packed requests and row-residue replies, and
bounded exact prefix reuse. All five 150-token variants use four-stage/64 MiB
preparation windows; all five 288-token variants use the existing serial issuance
contract. Whole-topology memory admission applies to both. Actual component
parameters, including provider-cache limits, are recorded in each saved target.
Pre-positioned tokenizer publication and checkpoint distribution retain their
separate costs.

Public fixture counts, source locks, exact component round trips, native
schedules, exact per-stage issuance-window costs and whole-topology admission are
checked before workers launch. Header-only and bound runtime row demand share
one implementation, as do execution-window admission and its verifier-aware job
pricing. Cold and repeat demand are checked independently.
The production schedules retain 23 initial-prefill tiled MLP regions. Placement
keeps at least 90% of body linear MACs remote independently in prefill and decode;
it is not a bound on the client's fraction of whole-request CPU. Exact-repeat
cache hits bypass initial-prefill tiling, and their smaller demand can fall below
the digit8 row threshold. Cold and repeat timing therefore remain separate.

The driver uses three serial, balanced process replicates per case: 30 fresh
clients and up to 60 responses. Each party shares a native 100/40 Mbps access
capacity, without added Internet RTT. The existing host reserve, memory admission,
pressure/swap guard, one-use inventory retirement and exclusive frontier-cohort
lock remain active. Per-case and rejected admission evidence is checkpointed;
failed or incomplete cells cannot establish a ranking. Provider cleanup is
enforced, but the ordinary target does not separately meter its CPU/time.

Generated three-layer Qwen2/Qwen3/Llama checks compare complete logits and active
KV across all five choices, cold/repeat requests and gateways. They also exercise
tile failure and reserved-material retirement. The real-checkpoint performance
screen compares output counts and salted output commitments; it is not a full
logit/KV or representative generation-quality oracle.

## Completed screen

The fresh-salt rerun completed **30/30 clients and 60/60 responses**, with exactly
eight outputs per response, matching output commitments within each workload and
zero observed swap growth. Every response passes the ordinary runtime/privacy
checks, every scenario conserves and retires prepared material, and all 60 request
traffic partitions reconcile to their directed application-body ledgers.

Original evidence is preserved byte-for-byte:

- [Complete report](../evidence/research-sota-combinations-qwen25.json), SHA-256
  `5fb73b6883a0e42ac42b65b33a84cdfe0c6646d1e459f0df72639c38756ad54d`.
- [Exact plan](../evidence/research-sota-combinations-qwen25.plan.json), SHA-256
  `3f8902f7f621adfd60dc301907ef144fbacc5f35c29c11bf02c689aaaf525204`.
- [Original target-local summary](../evidence/research-sota-combinations-qwen25.summary.json),
  SHA-256 `03a4cce5152caf837bb2920ad3271be0982cbbc0c77acfb806142e7edc3ba28e`.
- Installed build
  `ccc5bdd1c48b0fc5a691f1b6fef7a0bda2e6ae3fa3fa92cd8acd94ad1d4380eb`;
  exact measurement-driver and component-factory hashes are in the plan.

All figures below are medians of three fresh-client samples. Client RAM is the
**process-lifetime cold-plus-repeat peak**, including the in-process dashboard and
native WAN relay. Client CPU covers benchmark startup through the first cold
response; Preparation CPU covers process birth through that response. Preparation
RAM is sampled in the selected request window. TTFT is online only; complete
request latency includes client setup/preparation but excludes provider launch.
Cold TCP traffic includes startup, while repeat traffic is its own counter delta.
Decimal MB, seconds and generated tokens/second are used throughout.

### 150 input tokens, eight outputs

| Choice | Client RAM MB | Client CPU s | Prep CPU s | Prep RAM MB | Online TTFT s | N−1 decode tok/s | Cold TCP MB | Cold request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 222.00 | 20.18 | 15.09 | 215.60 | 41.06 | 3.80 | 478.65 | 106.80 |
| Tiling | 200.56 | 20.72 | 15.15 | 214.20 | 42.64 | 3.80 | 478.65 | 108.54 |
| Digit8 | 223.33 | 20.51 | 11.26 | 210.12 | 40.98 | 3.79 | 478.65 | 104.07 |
| Both | 202.49 | 20.65 | 11.34 | 224.02 | 42.60 | 3.84 | 478.65 | 106.19 |
| Both + placement | 202.18 | 21.30 | 10.75 | 212.65 | 39.08 | 4.46 | 458.78 | 108.71 |

### 288 input tokens, eight outputs

| Choice | Client RAM MB | Client CPU s | Prep CPU s | Prep RAM MB | Online TTFT s | N−1 decode tok/s | Cold TCP MB | Cold request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control | 280.25 | 27.61 | 20.21 | 260.26 | 76.53 | 3.76 | 787.88 | 178.59 |
| Tiling | 241.81 | 27.93 | 20.11 | 258.57 | 79.85 | 3.75 | 787.88 | 181.96 |
| Digit8 | 278.72 | 27.31 | 13.21 | 258.42 | 76.59 | 3.76 | 787.88 | 171.75 |
| Both | 242.30 | 28.04 | 13.18 | 258.46 | 79.75 | 3.74 | 787.88 | 174.89 |
| Both + placement | 242.16 | 27.97 | 12.30 | 259.15 | 73.27 | 4.37 | 723.89 | 168.60 |

### Exact repeats

Each repeat records an actual completed-prefill cache hit and zero newly issued
prefill rows. Initial-prefill tiling does not execute on this path. Repeat CPU
is unknown under this target's cold-first CPU window; the lifetime client RAM
above is not measured again as an independent repeat peak.

| Input | Choice | Prep RAM MB | Online TTFT s | N−1 decode tok/s | Repeat TCP MB | Repeat request s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 150 | Control | 156.32 | 2.46 | 3.84 | 17.41 | 10.43 |
| 150 | Tiling | 156.37 | 2.62 | 3.86 | 17.41 | 10.74 |
| 150 | Digit8 | 155.89 | 2.44 | 3.86 | 17.41 | 10.46 |
| 150 | Both | 175.39 | 2.63 | 3.88 | 17.41 | 10.73 |
| 150 | Both + placement | 169.21 | 2.63 | 4.49 | 14.91 | 10.32 |
| 288 | Control | 147.18 | 2.47 | 3.79 | 17.41 | 10.93 |
| 288 | Tiling | 141.59 | 2.64 | 3.79 | 17.41 | 11.24 |
| 288 | Digit8 | 144.52 | 2.48 | 3.80 | 17.41 | 10.92 |
| 288 | Both | 146.83 | 2.63 | 3.81 | 17.41 | 11.22 |
| 288 | Both + placement | 145.33 | 2.65 | 4.41 | 14.91 | 10.82 |

### Interpretation

- Tiling plus digit8 lowers client peak RSS **8.79% / 13.54%** and Preparation
  cold CPU **24.84% / 34.75%** at 150/288 inputs. Online cold TTFT increases
  **3.75% / 4.21%**; client cold CPU increases **2.36% / 1.54%**. Preparation RAM
  rises 3.91% in the shorter workload and falls 0.69% in the longer one.
- Adding the frozen placement lowers client peak RSS **8.93% / 13.59%** and
  Preparation cold CPU **28.72% / 39.14%** versus control. Online cold TTFT falls
  **4.84% / 4.26%**, and cold decode throughput rises **17.47% / 16.37%**.
  Complete cold requests become **1.79% slower / 5.59% faster**; client cold CPU
  rises **5.54% / 1.31%**. At least 90% of body linear MACs remain remote.
- Placement's median **complete cold-plus-repeat TCP streams** fall
  **496.06 → 473.69 MB (4.51%)** and **805.29 → 738.80 MB (8.26%)** after charging
  added client-weight delivery. Cold-only reductions are 4.15% / 8.12%; the exact
  repeat saves 14.33% at either length. These are distinct windows.
- Tiling alone has the lowest observed client RAM. Digit8 alone has the fastest
  short cold request. All five choices remain nondominated across the six
  measured resource/timing objectives plus total TCP traffic. Tiny control-byte
  differences and measurement noise prevent interpreting that as statistical
  superiority. Base and per-token headline objectives remain unknown.

This is the completed **five-way screen**, not independent finalist confirmation.
The per-model table labels its leaders **Screen best**. It does not pool earlier
attempts or historical component timings. Checkpoint/tokenizer distribution,
physical wire, Internet RTT, independent operators, whole-logit/KV parity on this
real checkpoint and representative generation quality remain outside the cohort.

## Earlier attempts

### Initial rejected long-workload cohort

The initial frozen build already selected four-stage/64 MiB windows. All fifteen
150-token cells completed, while all fifteen 288-token cells failed before online
execution. A separate private traceback diagnostic localized the rejection to
issuance-window admission: a gate/up job with 288 prefill plus seven prepaid
decode rows requires 73,107,456 declared work bytes (69.72 MiB), above the supported
64 MiB concurrent ceiling. Passing whole-topology memory admission alone did not
establish that this independent job bound was satisfied.

The [original report](../evidence/research-sota-combinations-qwen25-window-rejected.json)
and [frozen plan](../evidence/research-sota-combinations-qwen25-window-rejected.plan.json)
retain all 30 cells unchanged and cannot establish the five-way two-workload
ranking. The rerun uses fresh output/plan files, a fresh installed
build and private cohort salt, with serial issuance shared by all longer-workload
variants. No resource cap or pricing formula was relaxed. Diagnostic timings and
the original short-workload results are not pooled into the rerun.

All ten corrected production cases pass shared header-only issuance and topology
preflight. The installed-wheel generated smoke completes all five combinations;
the exact-state and failure-retirement regressions described above also pass.

### Interrupted corrected cohort

The first corrected [report](../evidence/research-sota-combinations-qwen25-interrupted.json)
and [frozen plan](../evidence/research-sota-combinations-qwen25-interrupted.plan.json)
retain 21 of 30 planned fresh clients: two complete balanced rounds and one cell
from the third. All 42 saved responses pass runtime checks, preserve the same
eight-output commitment within each workload, conserve material and covered
traffic, and observe zero swap growth. The runner was absent when inspected,
with no final summary or retained exit cause. The evidence remains incomplete
and excluded from the SOTA tables.

A complete rerun requires a fresh private salt and fresh evidence paths. Missing
replicates cannot be appended to this report or pooled across invocations.

## Eight objectives and traffic reconciliation

Client peak RSS, client CPU, Preparation CPU, Preparation peak RSS, online TTFT,
N−1 decode throughput, base traffic and output-dependent traffic remain distinct
objectives. Client lifetime RSS covers the complete cold/repeat scenario; it
cannot be presented as two independent per-request peak measurements.

The additive `pllm.benchmark_traffic_decomposition.v1` record conserves:

```text
covered all-link application bodies
  = known base + known output-dependent + unattributed
```

It separately retains startup, warmups, measured requests, cold-first and covered
lifecycle windows. Startup is charged once. Public delivery and measured prefill
payloads belong to base; measured decode exchanges and prepaid decode corrections
belong to output-dependent costs, even when reserved rows are burned unused.
Exact unused-material bytes remain unknown unless directly measured. The divisor
is actual generated outputs, not the allowance; executed decode steps are a
separate count because EOS probing can execute a further step.

Version-three directed accounting includes correction acknowledgements on the
Inference → Preparation edge. Serializer-certified row bits classify complete
payload bytes. Mixed envelopes, padding, shared packed phase-boundary bytes,
acknowledgements and lifecycle controls remain unattributed. Complete base MB per
request and output-dependent MB per generated token therefore stay unknown while
any residual remains. Historical reports retain their original coverage and
unattributed split. TCP-stream totals remain a separate diagnostic, with no
physical-wire claim.

The report makes no constant next-token marginal-cost claim: inventory thresholds,
cache hits, terminal pruning, early EOS and fresh-client setup can change the
workload-specific average. Epoch, in-flight sharing and public-capsule paths have
no admitted phase-attribution contract in this slice.

### Measured base and output-dependent subsets

The completed screen has partial attribution in every request. For a directly
reconcilable illustration, these are **replicate 0**, not medians; startup is
included only in the cold rows. Each total is exactly the sum of the preceding
three byte categories. All responses generate eight outputs.

| Input | Choice / window | Known base MB | Known output-dependent MB | Residual MB | Covered total MB |
| --- | --- | ---: | ---: | ---: | ---: |
| 150 | Control / cold | 460.593714 | 16.372533 | 0.138710 | 477.104957 |
| 150 | Placement stack / cold | 443.416338 | 14.106196 | 0.099932 | 457.622466 |
| 150 | Control / repeat | 0.002916 | 16.372570 | 0.137600 | 16.513086 |
| 150 | Placement stack / repeat | 0.001838 | 14.106220 | 0.099419 | 14.207477 |
| 288 | Control / cold | 769.326170 | 16.372567 | 0.138877 | 785.837614 |
| 288 | Placement stack / cold | 708.238743 | 14.106217 | 0.100013 | 722.444973 |
| 288 | Control / repeat | 0.002916 | 16.372570 | 0.137553 | 16.513039 |
| 288 | Placement stack / repeat | 0.001838 | 14.106220 | 0.099373 | 14.207431 |

The known output-dependent subset is about **2.04657 MB/generated token** for
control and **1.76328 MB/generated token** for the placement stack. These are
incomplete application-body averages, not complete per-token traffic or a
next-token marginal cost. Exact unused-output-material bytes remain null. The
table's disclosure retains all three replicates and all five combinations;
headline base/per-token cells stay unknown until the residual is attributed.

## Replay

Run from the repository root with the installed-wheel interpreter. The generated
transport smoke uses the same five public component combinations:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_sota_combinations.py \
  --tiny --processes 1 --output target/sota-combinations-smoke.json
```

The pinned checkpoint screen needs the maintained prepared example's indexed
tokenizer and public fixtures in `target/prepared-frontier-artifacts`, plus the
source-locked placement plan named by the driver. Admission can be checked before
allocating checkpoint tensors:

```sh
target/frontier-sdk-env/bin/python scripts/benchmark_sota_combinations.py \
  --preflight-only --output target/sota-combinations-admission.json

target/frontier-sdk-env/bin/python scripts/benchmark_sota_combinations.py \
  --processes 3 --output target/sota-combinations-qwen.json
```

Each output name must be fresh. Companion `.plan.json`, `.progress.json` and
`.summary.json` files preserve the exact targets, original build, fixture hashes,
admission and comparison outcome. Failed or interrupted runs retain their
completed cells; they do not resume one-use sessions or become a complete cohort.
`--replay-plan PATH` requires the original build, paths, configurations and public
fixture identities before execution. A changed wheel produces a new cohort.

The private comparison salt is fresh for each complete invocation and is never
published. Cold and repeat requests share one client within each case; different
cases and replicates use fresh clients and client artifact caches. Shared source
checkpoint and regenerable compiled-weight caches retain their explicit warm
lifecycle.
