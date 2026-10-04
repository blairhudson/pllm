# Five additional network methods — 2026-10-03

Source: `Qwen/Qwen2.5-0.5B-Instruct`, revision
`7ae557604adf67be50417f59c2c2f167def9a775`. W8A8 and `prefix_f32` stay fixed.
Bounded research reports include source/model/weight identities and native-binary
and environment digests. Reports contain no prompt text, token IDs, logits, KV,
masks or credentials.

## 1. Generated-prefix qualification

All **24** executed generated-state checkpoints across three public contexts
match fresh prefill for every logit and KV value. The live benchmark compares
the existing lean prepared stack—canonical prefix reuse, on-demand inventory,
compressed artifacts—with the same stack plus generated-prefix qualification.
Three message-history requests generate eight tokens each; all six response
digests and token usages match.

| Covered application bodies | Completed-prefill control | Generated-prefix reuse |
| --- | ---: | ---: |
| Setup-inclusive total | 402.046 MB | 366.629 MB |
| Setup-inclusive MB/generated token (24 outputs) | 16.752 | 15.276 |
| Setup through first response | 241.723 MB | 241.723 MB |

Saving: **35.418 MB / 8.81%** over the optimized control. The unchanged first
response alone exceeds the 40.205 MB tenfold target for this cohort. These are
co-located application-body measurements, not full wire. No client body weights
are added; retained cache limits do not measure peak memory.

[`generated-prefix-qwen25-2026-10-03.json`](generated-prefix-qwen25-2026-10-03.json)
records the benchmark. Separate tiny SDK tests cover prepared, verified and
offset roles, pending-token exclusion, actual generated-prefix hits,
cancellation and `store=False`. Historical numeric mode still rejects promotion.

Reserved stage rows fall from **10,464 to 9,120**, with zero remaining rows at
close. Cold-first client CPU samples are 12.87 and 9.94 seconds respectively;
the first request has no generated-prefix hit, so this difference does not
establish a reuse CPU saving. Client CPU across the complete conversation and
peak client memory remain unmeasured.

## 2. Cross-configuration state equivalence

Delivery and attention-placement variants pass contract comparison, explicit
state transfer and exact suffix logits/KV checks. All arrays stay client-local;
verification strength and lineage remain bound. Changed numeric choices, source
commitments, invalid provenance and response-owned snapshots are negative gates.
The seal records trusted-client provenance; it does not authenticate arbitrary
caller mutations of client-owned arrays. Automatic planner/cache migration and
network switching savings are not measured here.

Evidence: [`next-five-state-qwen25-2026-10-03.json`](next-five-state-qwen25-2026-10-03.json).

## 3. Masked output aggregation

A real 896→9,728 weight kernel preserves exact integers for one- and 39-row
public numeric fixtures. B adds a fresh client-known AES mask before relaying
its output to A. Native handles burn before malformed argument conversion;
context/role checks reject cross-issuance frames. An exhaustive small-ring
control checks ideal view independence. This is neither an authenticated
distributed transport nor independent cryptographic review.

At 39 rows, direct client downloads are **2,178,038 bytes**; relay downloads are
**1,089,019 bytes**. The other **1,089,019 bytes** crosses B→A, with at least 64
fresh issuance bytes added. Client reconstruction/issuance takes **24.955 ms**
versus **4.764 ms** for direct reconstruction. Matrix work and HTTP are excluded.
The all-link gate fails; no topology is activated.

Evidence: [`masked-aggregation-qwen25-2026-10-03.json`](masked-aggregation-qwen25-2026-10-03.json).

## 4. Token-local projection memo

Semantic dependency analysis finds one token-local projection group. All checked
logits and KV values match across three contexts and generated trajectories.
The bounded native memo records **27 hits / 31 misses**, retaining **178,560
accounted bytes**. Selected-kernel CPU falls **6.209 → 3.709 ms**; this is not
whole-client CPU. Fixed ownership adds **1,032,192 bytes** each of client weights
and native snapshots. Its optimistic arithmetic-body reduction is **0.405%**,
with **99.71% of body-linear MACs** still remote.

Misses must also run locally: conditional remote calls would expose activation
equality. This remains an SDK probe, not a new selectable placement.

Evidence: [`token-local-qwen25-2026-10-03.json`](token-local-qwen25-2026-10-03.json).

## 5. Progressive exact head certificates

Rust uses conservative signed-i8 bitplane intervals and the source float32
dequantization order, including stable greedy ties. The oracle retains all weights.

| Initial bits | Exact agreements | Maximum observed residual rows | Median head CPU | Full-head CPU | Projected independent PIR worker CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 16/16 | 149,992 | 26.157 ms | 13.135 ms | 143,442 s |
| 5 | 16/16 | 4,766 | 15.075 ms | 14.340 ms | 4,543 s |
| 6 | 16/16 | 43 | 12.640 ms | 12.752 ms | 40.775 s |

Six-bit refinement projects **58,222 bytes** for 43 independent lookups from a
measured one-row PIR control. This excludes **102.709 MB** of initial prefix and
scales, and is not a measured 43-query protocol. Data-dependent counts lack a
public padding capacity. The oracle retains the full head and establishes no
client-memory saving. Private refinement and sampling remain unimplemented.

Evidence: [`progressive-head-qwen25-2026-10-03.json`](progressive-head-qwen25-2026-10-03.json).

## Validation

Non-slow/non-HE Python regression and integration shards cover the new probes,
SDK/gateway paths, cache lifecycle, planner and existing transports. The executed
documentation examples pass. Native checks pass 133 core tests, three continuation
tests and 16 decoder-schedule tests; Clippy and Ruff pass. Documentation checks
pass 64 tests and the 401-route production build/navigation gate.
The unfiltered compiler-library test command exceeded its 120-second limit;
the scoped continuation and schedule suites completed separately. An isolated
Python 3.13 wheel imports all five probe APIs and passes native mask/head smoke
checks plus generated-prefix compiler admission.

## Verified reproduction

Repository root; pinned checkpoint already cached:

```sh
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_generated_prefix.py --real --output docs/evidence/generated-prefix-qwen25-2026-10-03.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_next_five.py --real --method state --output docs/evidence/next-five-state-qwen25-2026-10-03.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_next_five.py --real --method aggregation --output docs/evidence/masked-aggregation-qwen25-2026-10-03.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_next_five.py --real --method token-local --output docs/evidence/token-local-qwen25-2026-10-03.json
HF_HUB_OFFLINE=1 .venv/bin/python scripts/probe_next_five.py --real --method head --output docs/evidence/progressive-head-qwen25-2026-10-03.json
```

Omit `--real` for generated tiny-weight smoke checks. The conversation script
uses message arrays through the ordinary benchmark context-sequence path and
checkpoints after each completed candidate.
