# Exact-ring network algebra and five whole-decoder frontiers

This round targets **aggregate application-body bytes per generated output**,
including preparation and cold client delivery. It prices client CPU and storage
alongside network cost. The hypotheses are project experiments, not claims of
priority in the literature. No general 10× or 100× result is established.

## Five algebraic candidates

The screen uses all 96 compiled W8A8 body stages of pinned
`Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, with synthetic private-value stand-ins.
The 39+8 geometry executes 46 rows per stage before terminal pruning. Its control
already uses exact per-output row residues. These are arithmetic-body projections;
the separate SDK cohort below measures actual covered protocol bodies.

| Hypothesis | Test and result | Decision |
| --- | --- | --- |
| Public-weight annihilator input packing | For output width `b_j`, input coordinate `i` needs only `max_j(b_j - v2(W_ji))` bits. Native correction/reconstruction agrees with independent clear integer products for all 96 stages. One width per stage reduces projected bodies from 111.07 to 106.84 MB (3.81%); coordinate-specific widths save only another 30.5 kB | Promote the simpler stage-wide codec; no client body weights or coordinate map |
| Exact odd-radix rings | Use fresh uniform masks in the smallest odd ring enclosing the signed output bound, then encode bounded 16-value radix blocks. Integer results agree, but projected bodies are 108.19 MB versus 106.84 MB with stage packing | Reject promotion: worse bytes, additional ring/issuance contract and measured codec cost |
| Public block-moment bypass | Refine the earlier integer-anchor screen with one, four or sixteen public block sums per stage. Both residual and coefficient matrices remain signed-i8; exact integer outputs agree | Only 5.8/13.4/55.3 kB beyond stage packing, versus 0.61/2.43/9.73 MB of client coefficients plus native snapshots and 22.3/64.3/232.2 million extra integer operations. Reject |
| Signed output-row dictionaries | Collapse only exact identical, opposite or zero public weight rows; verify bytes after hash matches | No rows removed across 96 stages; a decoder map would add 1.52 MB. Reject |
| Shift-recurrent correlations | Update masks by a cyclic shift plus sparse fresh coordinates; seek a low-displacement-rank weight recurrence | Every checked displacement minor has rank at least 253. Worse, refreshing one of 32 coordinates exposes 31 exact cross-input relations. Reject this recurrence on privacy before cost |

The block-moment candidate is explicitly a refinement of the earlier anchor
hypothesis, rather than an independent new breakthrough. The mask-recurrence
counterexample rejects that sparse refresh, not every possible correlation
generator or structured-weight construction.

### Why stage packing preserves the existing result

Let `b = max_j b_j`. The client sends `(x-r) mod 2^b` rather than its padded
16/24/32-bit representation. Every omitted contribution to `W(x-r)` is a
multiple of `2^b`, hence of every output ring `2^b_j`. Existing reduced
corrections and output-mask reconstruction therefore yield the same exact signed
integer outputs before dequantization. Projecting the uniform input mask into a
public quotient keeps it uniform; widths depend only on committed public weights
and numeric bounds, never on private activation values.

`MaskedLinear(output_encoding="row_residues", request_encoding="stage_packed")`
selects this codec in the ordinary SDK. Rust admits only the matching output
contract and binds the choice into the schedule digest. Client and provider
derive the stage width from the source-bound residue layout. A distinct frame
namespace, explicit session acknowledgement, exact dimensions, canonical padding,
bounded native decode and existing authenticated stage/ticket checks reject
forged widths or a codec downgrade. Reservation, cancellation and failure still
burn one-use material. Input masks, corrections and inventory issuance retain
their original domains and lifecycle.

The option includes compact single-row metadata and composes with terminal
pruning, paged/compressed artifacts, prefix reuse, duplex prefill and Freivalds
verification. Checked tiny compositions preserve full logits and KV, with
malformed-width, acknowledgement-downgrade and replay tests exercising failure
cleanup. It retains the baseline's trusted-Preparation and non-collusion contract;
these checks are not an independent cryptographic review.

## Matched ordinary SDK pair

One cold, capped 39-input/8-output response per candidate uses one cohort salt,
the same pinned checkpoint/body, greedy sampling and fresh client artifact caches.
Both use terminal pruning, row residues, paged/compressed batched artifacts and
request-sized four-stage preparation overlap. Only request encoding changes.

| Metric | Compact control | Stage-packed input |
| --- | ---: | ---: |
| Covered setup-inclusive bodies, MB | 234.068 | 229.983 |
| Covered MB/generated output | 29.259 | 28.748 |
| Online bodies, MB | 68.702 | 64.617 |
| Online MB/generated output | 8.588 | 8.077 |
| N−1 decode online MB/output | 1.554 | 1.462 |
| Cold client/dashboard CPU, s | 8.578 | 8.466 |
| Cold aggregate CPU, s | 36.333 | 36.322 |
| Request latency, s | 13.118 | 13.280 |

Packing saves **4,084,672 bytes**, or **1.75% covered** and **5.95% online**.
That is **0.511 MB less per generated output** on both scopes. All eight output
tokens' rendered text has the same captured digest; separate tiny tests check
complete numeric state. Both runs pass runtime/privacy checks, send zero
plaintext prompt/token-ID bytes, add zero client body weights and observe zero
new swap. Both issue/reserve 4,416 stage rows, claim 4,302 and burn 114; the
material ledger conserves all rows. No offline issuance saving is claimed.

Aggregate CPU is nearly equal; request latency increases 1.23% in this single
pair. Thus this is an optional byte reduction, not an automatic placement or
speed recommendation. Provider processes are separate but co-located. The
sequential client's lifetime RSS is not an independent candidate peak. Source
checkpoint distribution, full wire, representative WAN performance, independent
operators and a matched two-offset full-response compute comparison remain
unmeasured. Do not add percentages from the earlier four-candidate cohort.

## Five further 100× hypotheses

The historical 39+32 prepared control accounts for 178.97 MB, leaving **1.790 MB
per response** at 100×. The following are fresh whole-decoder directions with
bounded synthetic falsification tests. None is an executable private decoder.

1. **Correction-assisted side-information codec.** Decode a much shorter masked
   ingress using the provider's already-resident correction as side information.
   Exhaustively enumerate all 16 private inputs and 64 mask choices in a
   two-coordinate mod-4 example. Conditional input entropy remains four bits:
   the independent output mask removes the needed side information. The existing
   correction cannot provide a worst-case lossless saving in this toy. A new
   correlated encoding would need a different, reviewed privacy contract.
2. **Epoch-reusable hidden nonlinear encodings.** Amortize nonlinear material
   across many responses by refreshing opaque wire labels with a public epoch
   transform. A 12-wire witness recovers all 12 cross-request private equality
   relations by undoing public XOR refresh. Reuse with that refresh is rejected;
   this is not an attack on reviewed reusable garbling or function-sharing schemes.
3. **Whole-span polynomial function sharing.** Settle an entire layer span after
   one hidden input opening by composing its nonlinear polynomial before issuance.
   Even repeated squaring of one scalar through 24 layers has degree `2^24`.
   The dense translated-coefficient representation needs **268.44 MB** of two
   opaque 64-bit coefficient shares, already 150× the whole-response target.
   Four-layer evaluation matches an independent clear oracle. This prices one
   representation, not every possible function-sharing construction; it excludes
   attention, rounding and all other channels.
4. **Secret-index nonlinear tensor networks.** Factor the complete nonlinear
   function table, then privately contract only small hidden indices through a
   tensor train. Exact GF(257) unfolding ranks of an eight-input-bit gated table
   are `1,2,4,8,16,8,4,2,1`; dense cores need **1,360 bytes** versus the flat table's
   **512 bytes**, before private selection. This rejects that table's compact
   dense-core representation, not all tensor networks or Qwen function tables.
5. **Private observational-state automata.** Merge hidden decoder states with
   identical future observable outputs, keeping only a small private state index.
   In a 35-state affine toy, same-next-token equivalence has two classes, but
   eight-step traces require nine. An explicit pair agrees on the next token and
   differs on the following one. Greedy next-token equality is unsound for state
   merging; universal future/input equivalence, private transitions and a useful
   real-model quotient remain unimplemented.

These gates identify why straightforward implementations fail. They do not
prove 100× impossible, and do not establish trained-model quality, protected
selection, real-model performance or a compute-cap win.

## Reproduce

```bash
HF_HUB_OFFLINE=1 uv run --no-sync python scripts/probe_network_algebra.py \
  --output network-algebra.json

uv run --no-sync python scripts/probe_network_frontiers.py \
  --output network-frontier-gates.json

HF_HUB_OFFLINE=1 uv run --no-sync pllm benchmark run \
  --experiment scripts/aggregate_network_experiments.py:combined \
  --experiment scripts/aggregate_network_experiments.py:stage_packed \
  --factory --trust-python --backend native --warmups 0 --repetitions 1 \
  --max-output-tokens 8 --temperature 0 --capture-output-digest \
  --timeout 300 --output network-algebra-sdk.json
```

Both checkpoint commands require the pinned source in the shared Hub cache.
The first records source locks and public-weight geometry, not live client RSS.
The second needs no checkpoint and records the probe-code digest. The third
uses the ordinary memory guard and separately launched prepared-role children.

Evidence: [algebra screens](../evidence/network-algebra-qwen25-2026-10-06.json),
[ordinary SDK pair](../evidence/network-algebra-sdk-qwen25-2026-10-06.json),
[five frontier gates](../evidence/network-frontier-gates-2026-10-06.json).
Previous round: [aggregate-network research](aggregate-network.md).
