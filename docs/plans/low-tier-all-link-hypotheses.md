# Ten Low-tier all-link traffic hypotheses

Status: **proposed, unmeasured**. This round targets the **256 MiB whole-client
maximum**, exact existing-model execution and predominantly remote body work.
Novelty means mechanisms not yet screened in PLLM, including new extensions of
existing contracts; it is not a claim of priority in the literature.

## Objective and reference

Minimize total directed traffic across Client, Preparation and Inference,
including new public artifacts, setup, unused material and failed attempts.
Keep cold base, prompt-prefill, output-dependent and unattributed costs visible.
Report generated outputs and executed decode rows separately. A packet moved to
another link remains a cost.

Use the best **completed, Low-qualified** configuration for each workload as the
control, including selective placement where it already wins. Qualification uses
the largest fresh-client peak across every replicate, not median RSS. Native
allocation admission and live headroom remain additional requirements. Retain at
least 90% remote body MACs independently in prefill and decode; count local CPU,
page buffers, file cache/disk owners and every new artifact separately.

The [completed combinations screen](../research/sota-combinations-reference.md)
provides current control candidates. The separate
[six-recipe Low screen](../research/low-memory-combinations-reference.md) must
finish and pass its audit before its results can select a control.

For scale, the historical [reconciled 150+8 cold/exact-repeat pair](../research/all-link-frontier-reference.md)
contains 493.575 MB of application bodies:

| Category | MB | Share |
| --- | ---: | ---: |
| Preparation corrections | 139.061 | 28.17% |
| Returned arithmetic | 139.123 | 28.19% |
| Masked ingress | 90.444 | 18.32% |
| Public bundle | 124.905 | 25.31% |
| Other recorded control/seeds | 0.043 | 0.009% |

These are a **historical sensitivity reference**, not a new Low-tier measurement
or the directional breakdown of the latest screen. Source:
[`research-all-link-frontier.json`](../evidence/research-all-link-frontier.json),
`control.classes`. TCP streams and physical wire retain their separate scopes.

## Expected impact: conditional planning ranges

Ranges below are engineering hypotheses, not measured gains, confidence intervals
or guaranteed lower bounds. Each may save nothing or regress. Percentages mean
total all-link reduction within the stated matched workload; categories and
workloads differ, so the percentages cannot be added. Fresh-request estimates
require a comparison with the existing whole-group placement control.

| ID | New hypothesis | Conditional all-link impact | Workload / affected cost |
| --- | --- | --- | --- |
| H1 | Output-channel ownership instead of whole projection groups | **0–3% additional** | Fresh requests and repeats; corrections/results minus new weight delivery |
| H2 | Input-column ownership chosen for residue-bound crossings | **0–3% additional target; cold likely worse** | 4–16 distinct-prefix pairs sharing public weights; ingress and smaller residues |
| H3 | Oblivious public-prefix state atlas | **10–30%** in a sufficiently reused multi-template deployment | Public-catalog distribution, with private template choice; no gain on an existing local capsule hit |
| H4 | Context-conditioned lossless weight entropy model | **0–2%** | Cold public delivery; essentially zero on cache hits |
| H5 | Byte-value-aware public-artifact eviction | **5–15%** under material cache churn | Repeated model/placement switching at the same disk capacity; zero on the first cold pair |
| H6 | Vocabulary-certified first-projection residue bounds | **0–0.1%** | First Q/K/V stage only; fixed public bounds, no private-value adaptation |
| H7 | Cancellation-aware prefill preparation cuts | **3–10%** in a cancellation-heavy workload | Avoided unissued corrections; completed requests may become slower/larger |
| H8 | Authenticated chunk-resumable public artifacts | **1–8%** on disrupted downloads | Avoided repeated public bytes; clean-link cost may increase slightly |
| H9 | Receipt-based continuation of unreserved correction uploads | **1–8%** on disrupted preparation | Avoided discarded preparation uploads; no clean-link payload saving |
| H10 | Session-indexed control plane | **0.05–0.2%** per cold/repeat pair; **1–3%** on a small warm repeat | Public-manifest revalidation plus implicit lease row coordinates |

### H1. Output-channel ownership

Split a compiled linear group's **output rows**, placing a bounded subset at the
client and keeping the rest remote. Preserve the original input quantization;
stitch exact integer outputs before the original dequantization/nonlinear step.
Page only selected public weight rows. Re-optimize within the same local-MAC
budget instead of adding work on top of an already 10%-local placement.

Unlike existing `ClientLinearRoles`, ownership need not take an entire QKV or
gate/up group. This may spend the remaining MAC budget more efficiently and cross
fewer all-or-nothing weight-delivery thresholds. It removes selected correction
and reply lanes; **ingress remains while any output stays remote**.

Cost gate: saved correction/reply bytes minus actual compressed row artifacts,
scales, maps and framing. The 0–3% range is a 0–15 MB net target on the historical
scale, not a computed optimum. Reject if splitting loses the whole-group
control's ingress saving, violates tiling ownership, or requires excessive local
MACs/RSS. First experiment: a public-weight row-level cost frontier, then exact
generated-checkpoint logits/KV for one admitted split.

### H2. Input-column ownership and bound crossings

Split `W x` as `W_local x_local + W_remote x_remote`; add the exact integer
partials before any original scale/rounding boundary. Select public columns that
both remove upload coordinates and push many remote output bounds below a
power-of-two residue threshold. This targets **columns**, unlike H1's output rows
and earlier block-sum/anchor approximations.

For `k` removed columns, `n` output rows, `R` executions and `b` input bits,
ingress saves approximately `R*k*b/8` bytes against `n*k` raw weight bytes, before
compression, residue reductions and descriptors. At `n=896, b=24`, ingress alone
needs about 299 executed rows to repay raw weights. A 150+8 cold/exact-repeat pair
does not reach that threshold. Expansion stages have a much worse threshold.

The long-horizon target additionally requires beating whole-group/H1 ownership
under the **same** MAC and disk budgets. First experiment: price actual bound
crossings and establish whether any column choice is nondominated. Discard it
before runtime work if whole-group ownership wins everywhere.

### H3. Oblivious public-prefix state atlas

Publish a bounded catalog of exact, explicitly public prefix capsules at the two
non-colluding roles. A fresh two-server private-information-retrieval query returns
one capsule without revealing which public template the client selected. Perform
this before online decoder execution; the client then runs an arbitrary private
suffix through the existing numeric continuation contract.

This differs from a pre-positioned single `PublicPrefixCapsule` and from the old
finite-task **response** table: the returned object is exact prefix state, while
the private suffix and subsequent generation remain ordinary model execution.
Compare with downloading the complete public catalog when template choice must
stay private, with a locally cached capsule, and with ordinary private prefill.

For `N` equal-size capsules of `S` bytes and `K` cold clients, the catalog-download
control pays `K*N*S`. A simple two-replica XOR-share retrieval layout would pay
`2*N*S + K*(2*S + query_bytes)`, before additional framing, capsule compilation
and other decoder traffic. For `N=32, S=4 MB, K=16`, that is 2,048 MB versus
384 MB before query bytes: **catalog-only arithmetic**, not a serving result.
Smaller catalogs, few clients or repeated local hits can erase the benefit.

First experiment: bound complete DPF/query construction, both replies, trusted
publisher authentication and client import below Low's remaining headroom.
Charge atlas publication to both roles and the full cold-client cohort. The
retrieval protocol, role lifecycle and independent cryptographic review are new
gates; existing capsule validation supplies no PIR security claim.

### H4. Context-conditioned lossless public-weight coding

Fit a small, immutable entropy model to **public weights**, using bounded decoded
neighbors or public row classes to predict symbol probabilities. Stream original
integer bytes and scales through an independently checked decoder. Cap model
tables and codec scratch; retain the original decoded-object digest.

The new variable is conditional entropy modeling, rather than the previously
failed unconditional byte rANS, bitplane transforms, row deltas or token-parent
residuals. Freeze contexts and validation blocks before evaluating the whole
object; charge model tables, framing and cold client decode CPU.

A 2–8% reduction of the reference's 124.905 MB bundle would save 2.50–9.99 MB,
or 0.51–2.02% of total bodies **before added model bytes**. This is the target,
not an observed entropy gap. First experiment: held-out public-block coded lengths
against existing bounded zlib, followed only if positive by complete-object
lossless decoding and independent fresh-client peak measurement.

### H5. Network-value-aware public-artifact eviction

Replace LRU's recency-only decision with bounded object reuse/transfer-cost
accounting: protect expensive shared token/head objects when numerous cheaper
placement objects would otherwise evict them. Use public object identities and
past accesses; no prompt content or future-workload oracle.

Unlike existing artifact reuse, the hypothesis is **which objects survive at the
same capacity**. A fixed public switching trace must exercise actual capacity
pressure. Report disk occupancy, metadata RAM and local reads, and compare both
LRU and a simple frequency policy. Eliminating 20–60% of redelivery when public
artifacts occupy roughly a quarter of the churn trace gives a 5–15% sensitivity
range. An ample-cache or first-cold workload gets no such saving.

### H6. Exhaustive first-layer token-domain bounds

For an admitted text-only source whose first Q/K/V input is solely a token lookup
plus fixed local normalization/quantization, enumerate the entire **public
vocabulary** offline. Certify the actual maximum signed integer output per row,
rather than admitting every arbitrary signed-i8 input vector. Bind the certificate
to exact source tensors, operators and compiler provenance. Choose widths once,
publicly, and keep them fixed for all private tokens.

This is a reachable-input proof, not an empirical prompt range, private adaptive
width, changed quantizer or local projection memo. It excludes soft prompts and
any other unenumerated input path. The first QKV stage occupies only about 0.31%
of the historical pair's bodies, setting a small scope even before certificate
delivery. First experiment: complete vocabulary integer extrema and a compiler
domain proof; require saved payload to exceed the certificate and codec cost.

### H7. Prefill-stage preparation cuts for interrupted work

Extend lifecycle scheduling to prepare only a bounded, public prefix of the
**prefill graph** before permitting cancellation at an authenticated cut. Retire
the active online session before the next offline preparation segment. All
reserved material still burns; only material never issued can be saved.

This differs from output-token decode epochs: it addresses abandoned prefill
before the first generated token. Preparation stays idle during online execution.
Partial decoder tensors, extra admissions, replay protection and cut lifetimes
need a new compiler-bound contract.

A sensitivity example with 20–40% cancellations, 50–75% avoidable corrections in
those attempts and a roughly 28% correction fraction suggests about 3–8% of a
successful-request-equivalent traffic budget. Recompute the actual denominator
over **all attempts**; the table's 3–10% target allows different failure timing.
First experiment: fixed cancellation traces and complete lifecycle byte pricing.
Successful full requests may only gain overhead and latency, so retain them as
an explicit control.

### H8. Chunk-authenticated artifact download continuation

Add immutable chunk commitments and bounded resume receipts inside large public
weight objects. Retain already validated chunks across an interrupted transfer,
then verify the complete original object before execution. This extends existing
whole-object caching and bounded frame delivery with **partial-object** reuse.

Avoided bytes are the validated prefix that an interrupted object download would
otherwise resend, less chunk manifests, receipts and extra framing. Removing
5–30% of redundant delivery at a 25% artifact share gives a roughly 1–8% target
on the corresponding disruption workload. Record interruption position and both
attempts; clean-link traffic may increase. First experiment: bounded public-file
fault injection with exact bytes, corruption rejection and fresh-process RSS.

### H9. Unreserved correction-upload continuation

Keep a receiver-owned, expiring upload generation with monotonically committed
chunk receipts. After a transport disconnect, authenticate a continuation and
send only missing chunks **before inventory seal and before any reservation**.
One logical fresh inventory remains the sole owner. Explicit abort, expiry,
ambiguous ownership or any online consumption closes this path and burns it.

This is not reusable correction material or replay of a failed response. Current
failure/burn semantics cannot simply be bypassed: a new pre-reservation transfer
state machine must prove single publication and distinguish disconnection from
abort. Charge retained receiver storage, receipt durability/transport and erasure.

If avoidable reuploads comprise 5–30% of correction bytes and corrections occupy
roughly 28% of a disruption trace, eliminating them suggests 1–8% overall, with
no clean-link payload saving. These trace proportions need measurement. First
experiment: exhaustive small-state disconnect/lost-ACK/expiry
interleavings, then byte accounting. Reject if ambiguous completion requires
keeping material that the existing ownership contract must burn.

### H10. Session-indexed control plane

Authenticate an immutable public bundle/plan manifest once, then revalidate its
digest and provider epoch rather than redelivering it on each unchanged import.
Within each fresh authorized session, bind row ordinals to its one-use lease so
bulk frames need not repeat independently transmitted ticket identifiers.

Unlike existing compact stage frames, this removes repeated public-manifest
bodies and makes row selection implicit in an authenticated ordered range.
Offers, capacity and session authorization remain fresh; source/epoch changes
invalidate reuse. Never infer a row from unauthenticated packet order or a global
counter. Completion, cancellation and replay still consume the same ledger.

A historical public manifest is about 195 kB. Hundreds of kB of additional row
identifiers are an optimistic removable allowance at 150-token geometry; exact
current frame enumeration is required. That supports a small 0.05–0.2% pair
target, while 0.2–0.5 MB matters more against a roughly 16 MB warm repeat. Count
all replacement digests, counters and receipts. First experiment: enumerate
actual serialized redundancy, followed by source-change and replay controls.

## Investigation order and closure

1. **H1 and H4 first:** opportunities on ordinary fresh inputs, with cheap public
   cost screens before SDK work. H1 must beat current whole-group placement.
2. **H3 and H5 next:** potentially larger gains for explicitly specified
   multi-template or cache-churn workloads. Publish their reuse break-even points.
3. **H2, H6 and H10:** cheap cost ceilings; stop early if metadata, client work or
   stronger controls dominate the projected payload reduction.
4. **H7–H9:** investigate against a separately defined cancellation/disconnection
   workload, retaining successful clean-link requests as controls.

For any surviving implementation, freeze the recipe and use one private cohort
salt, fresh clients, alternating controls and at least three replicates under
the ordinary shared native 100/40 caps. Preserve complete logits/KV on generated
Qwen2/Qwen3/Llama controls, then the pinned real-checkpoint workload. Measure all
seven priority metrics, exact directed bodies and stream counters; include every
retry, added artifact and cleanup. Low requires every observed client peak at or
below 268,435,456 bytes, plus normal conservative admission and the swap guard.
Partial base/output attribution stays partial. A byte win earns promotion only
after exact execution and ownership checks; independent confirmation remains a
separate cohort.

Prior negative controls remain in [network algebra](../research/network-algebra.md),
[first-principles screens](../research/network-first-principles.md),
[predictive codecs](../research/predictive-weight-codec-reference.md) and
[the preceding five-way traffic round](all-link-next-five.md).
