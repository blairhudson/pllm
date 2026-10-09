# Compiler-bound MLP working sets

This research slice follows the rejected whole-client memory result for
[native stage ownership](client-stage-ownership.md). It changes the lifetime of
complete MLP intermediates: a small row tile passes through gate/up projections,
SiLU, multiplication and the down projection before the next tile starts.
Transport-only chunking still reconstructs a full stage output; this experiment
bounds the intervening MLP tensors as well.

## Contract and owners

The Rust compiler derives four-step regions from the admitted semantic schedule.
Eligibility requires grouped linear gate/up inputs, client-local float32 SiLU and
multiply, a matching down projection, common layer identity and exclusive interior
consumers. State roots, terminal-row pruning and externally consumed intermediates
cannot be silently folded into a region. Source-plan, composition, source-schedule,
row bound and exact operation identities bind the derived contract. Tile sizes
are public, from 1 to 256 rows, under the existing bounded matrix codecs.

The original linear and local numeric executors perform every operation. Tiling
does not split the feature reduction or change per-row quantization. Each tile's
intermediates retire at last use and before the next tile; a complete hidden-width
output remains for the residual consumer. The original one-use inventory supplies
disjoint row claims. Exceptions propagate through ordinary SDK cancellation and
burn unused reserved rows. Preparation remains offline and body weights remain
provider-owned in the measured prepared composition.

Only eligible initial-prefill regions are tiled. Decode, continuation, terminal
pruning, tokenizer, attention and full KV retain their ordinary execution. The
internal override has its own execution identity and is not a selectable Pipeline
component. Freivalds, protected nonlinear regions and retained-state compositions
reject. Whole-topology memory admission retains its conservative existing bound;
tiling cannot bypass full-stage residue-layout admission.

## Validation and measurement

Generated Qwen2, Qwen3 and Llama controls compare complete 33-row prefill logits
and three same-token decode steps, including every active KV bit. Tile sizes
1, 7 and 16 exercise partial final tiles. Weak-reference checks reject retaining
the previous tile's product into the next remote exchange. Separate prepared-role
children execute the same SDK requests and compare logits, state and output text.
Injected failure after a completed tile verifies reservation retirement and
rejected reuse.

The response benchmark's explicit `client_mlp={"tile_rows": ...}` selector uses
zero as its full-tensor control. Both arms compile eligibility and count the same
public metadata owners. Instrumentation records eligible regions, executed tiles
and maximum tile rows; these are ownership counts, not RSS. Unique Python-object
bytes for graph, schedule and binding metadata are reported separately from
process memory and native allocations.

The existing `FrontierStudy` runner supplies fresh clients, exact installed-build
locks, a shared private cohort salt, source/numeric/output comparison gates,
memory admission and native shared 100/40 Mbps pacing. Each workload retains all
seven serving metrics. A cold request and exact-repeat request share each client;
public checkpoint and tokenizer artifacts are pre-positioned. Added exchanges,
CPU, latency and traffic remain charged.

Replay with the public artifacts from the maintained prepared study:

```sh
python -m benchmarks.research.client_stage_ownership --working-set --processes 1 \
  --workload medium --workload long --input-bound 384 --serial-issuance \
  --output target/mlp-working-set-screen.json
```

The explicit 384-token bound fits existing residue admission for Qwen2.5's grouped
gate/up stage. It is not a change to the codec ceiling. Confirm a nominated cell
using the same study arguments plus `--confirm SCREEN --nominate medium/tile64`;
confirmation creates at least three fresh clients per arm. Physical wire,
checkpoint distribution and independent provider operation remain outside these
local measurements.

Confirmation also requires the screen's complete case identity, including its
recorded interpreter and installed-build digest. Run it with that same `BuildRef`;
a newer wheel needs a fresh matched screen. Changing the launcher environment
must not relabel historical measurements as current-build evidence.

## Initial checkpoint screen

Six fresh clients execute cold/repeat Qwen2.5-0.5B requests at **150+8** and
**288+8** tokens. The public input bound is 384 and Preparation is explicitly
serial. All twelve requests pass ordinary checks, preserve within-workload output
commitments and observe zero host swap growth. The installed build is
`b94df5b1343957e180d4470b45e406bcdaa3f5926ca998e9bd946d99491ebdff`.

CPU and TTFT use seconds; RSS and stream traffic use decimal MB. Client RSS is
process-lifetime high-water. Preparation RSS is sampled in request windows. CPU
covers cold-first execution, TTFT/decode summarize both requests, and traffic
covers setup plus cold/repeat. The seven priority axes remain separate.

| Workload / rows | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150+8 / full | 233.46 | 21.248 | 15.877 | 188.89 | 21.927 | 3.657 | 496.046 |
| 150+8 / 16 | 189.04 | 21.260 | 15.772 | 193.05 | 25.161 | 3.653 | 496.046 |
| 150+8 / 64 | 197.67 | 20.986 | 15.955 | 192.02 | 22.674 | 3.807 | 496.046 |
| 288+8 / full | 267.55 | 27.847 | 21.162 | 257.15 | 39.640 | 3.483 | 805.269 |
| 288+8 / 16 | 235.57 | 28.829 | 21.615 | 264.44 | 46.707 | 3.677 | 805.269 |
| 288+8 / 64 | 236.77 | 28.174 | 21.226 | 259.57 | 41.375 | 3.655 | 805.269 |

The 64-row candidate lowers client RSS **15.33% / 11.51%**, while median online
TTFT rises **3.41% / 4.38%**. Complete cold request time rises
**113.834 → 114.702 seconds** and **179.487 → 182.884 seconds**. Sixteen-row tiles
save slightly more memory but increase cold latency and TTFT further. Small
stream-count differences are framing/control bytes; there is no arithmetic-byte
saving. Both arms count **3,955,137 bytes** of unique public Python metadata per
runtime; this is not construction-peak attribution.

These single-process cells nominate 64-row tiles for a separate three-client-per-arm
confirmation. They do not establish repeatable savings or a tenfold improvement.
Evidence: `docs/evidence/research-mlp-tiling-qwen25-screen.json`, with its
`.plan.json` and `.frontier.json` companions.

## Independent confirmation

Twelve new clients use the same installed build, three per workload/arm, with an
independent private cohort salt and interleaved control/candidate order. All
24 cold/repeat requests pass runtime checks and preserve within-workload output
commitments, with zero observed swap growth. These confirmation medians do not
replace or pool the initial screen. Units and measurement windows remain as above.

| Workload / rows | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150+8 / full | 242.40 | 21.231 | 16.153 | 191.91 | 21.929 | 3.615 | 496.046 |
| 150+8 / 64 | 200.88 | 21.784 | 15.737 | 193.64 | 22.729 | 3.633 | 496.046 |
| 288+8 / full | 292.91 | 28.754 | 21.569 | 264.57 | 39.805 | 3.636 | 805.269 |
| 288+8 / 64 | 241.27 | 29.672 | 22.899 | 264.49 | 41.676 | 3.327 | 805.269 |

Median client peak RSS falls **17.13% / 17.63%**. Every matched replicate has
lower candidate RSS, despite visible process-to-process variation. Median online
TTFT rises **3.65% / 4.70%** and complete cold request latency rises
**113.502 → 115.520 seconds** and **180.371 → 184.235 seconds**. Repeat-request
medians are **11.037 → 11.037** and **11.798 → 12.296 seconds**. The longer
workload's decode median regresses **8.49%**; tiling itself runs only during the
initial prefill, so that observation does not establish a changed decode kernel.
Both topology CPU owners and tiny framing differences remain charged.

This confirms a bounded client-memory/latency tradeoff for these two Qwen2.5
workloads. It does not establish tenfold savings, broad quality, longer-context
admission or a Qwen3 result. The selector stays explicit research configuration;
there is no automatic device/scheduler choice. Evidence:
`docs/evidence/research-mlp-tiling-qwen25-confirmation.json`, with its `.plan.json`
and `.frontier.json` companions. The frontier export retains missing 16-row
confirmation cells instead of transferring their exploratory results.

### Separate confirmation B and host overlap

A second twelve-client confirmation has its own private cohort salt and the same
installed build. It also passes all 24 requests, preserves outputs and covered
application bodies, and observes zero swap growth. Its medians remain separate:

| Workload / rows | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150+8 / full | 234.11 | 21.902 | 16.936 | 193.17 | 22.075 | 3.459 | 496.046 |
| 150+8 / 64 | 198.34 | 21.941 | 16.443 | 193.07 | 22.735 | 3.204 | 496.046 |
| 288+8 / full | 268.35 | 28.846 | 22.433 | 263.11 | 39.750 | 3.576 | 805.269 |
| 288+8 / 64 | 242.50 | 29.036 | 21.836 | 264.08 | 41.520 | 3.577 | 805.269 |

Client RSS falls **15.28% / 9.63%**, while online TTFT rises **2.99% / 4.45%**.
Cold request medians are **114.304 → 116.103** and **180.658 → 184.223 seconds**;
repeat medians are **11.755 → 11.714** and **11.503 → 11.466 seconds**.

The two confirmation runs overlapped on the same host: A's response window was
20:50:11–21:26:53 UTC and B's was 20:57:36–21:34:26 UTC on 2026-10-09.
Each graph enforced its own shared party caps. These are separate process/cohort
observations, not exclusive-host performance measurements. Their memory direction
agrees, but process variation is substantial and timings cannot establish an
uncontended latency effect. Neither cohort is discarded or pooled. Evidence:
`docs/evidence/research-mlp-tiling-qwen25-confirmation-b.json`, with its `.plan.json`
and `.frontier.json` companions. Reproduce B with the same confirmation arguments;
the CLI always requires at least three clients per nominated arm.

### Non-overlapping confirmation C

A third twelve-client cohort ran after A and B completed, from
21:53:24 to 22:29:34 UTC on 2026-10-09. It uses the original screen's installed
build and another independent private cohort salt. All 24 cold/repeat requests
pass runtime checks, preserve source/numeric/state/privacy/workload/output
commitments and covered application bodies, and observe zero swap growth.
Every matched replicate has lower tiled client peak RSS.

| Workload / rows | Client RSS | Client CPU | Prep CPU | Prep RSS | TTFT | Decode TPS | Stream MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 150+8 / full | 216.84 | 21.910 | 16.218 | 190.86 | 21.984 | 3.492 | 496.046 |
| 150+8 / 64 | 196.28 | 22.065 | 16.089 | 192.23 | 22.797 | 3.451 | 496.046 |
| 288+8 / full | 283.95 | 27.714 | 20.089 | 257.26 | 39.905 | 3.368 | 805.269 |
| 288+8 / 64 | 237.65 | 26.281 | 20.101 | 256.97 | 41.061 | 3.771 | 805.269 |

Median client peak RSS falls **9.48% / 16.31%**, while online TTFT rises
**3.70% / 2.90%**. Complete cold-request medians are
**113.755 → 115.521** and **179.873 → 181.666 seconds**; repeat medians are
**11.368 → 11.353** and **11.610 → 10.857 seconds**. Decode medians change
**−1.16% / +11.95%** despite an unchanged decode executor. The seven-metric
frontier retains both the full and tiled cases as nondominated; there is no
unqualified winner. These samples strengthen the bounded memory/TTFT tradeoff,
without establishing a decode-kernel improvement or tenfold memory reduction.
Ordinary host/process variation remains; C is neither pooled with A/B nor used
to rewrite their measurements.

Evidence: `docs/evidence/research-mlp-tiling-qwen25-confirmation-c.json`, with
its `.plan.json` and `.frontier.json` companions. The confirmation plan binds
the original screen and both nominated 64-row cells; unmeasured 16-row cells
remain missing. The successful invocation was:

```sh
target/mlp-tiling-venv/bin/python -m benchmarks.research.client_stage_ownership \
  --working-set --processes 3 --workload medium --workload long \
  --input-bound 384 --serial-issuance --artifacts target/prepared-frontier-artifacts \
  --confirm target/evidence/mlp-tiling-qwen25-screen.json \
  --nominate medium/tile64 --nominate long/tile64 \
  --output target/evidence/mlp-tiling-qwen25-confirmation-c.json
```
