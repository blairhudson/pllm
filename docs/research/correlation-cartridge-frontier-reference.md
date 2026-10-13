# Fixed-weight correlation cartridges across client RAM tiers

This follow-up examines three proposed uses of the
[reviewed exact-ring PCG sources](compact-setup-fulltext-review.md): a Medium
per-layer cartridge, a Low external-memory expansion schedule, and selective
deployment on expensive prepared stages. Low and Medium remain **256 MiB and
2 GiB whole-client maxima**, including the SDK, state and overlapping owners.

**Result: no reviewed candidate establishes a 20–50% serving saving or qualifies
for either RAM tier.** Selective bulk MLP gate/up prefill gives the most precise
next construction target. The Medium W-only alternative offers a conditional
traffic/computation tradeoff; the examined Low spill schedule has substantial
unmeasured expansion and I/O costs.

## Shared construction gate

The desired finite, source-bound generator has asymmetric outputs:

```text
Client:       fresh (r_j, s_j)
Inference:    q_j = W*r_j - s_j
Online:       Client sends x_j-r_j; Inference returns W*(x_j-r_j)+q_j
Client:       adds s_j and recovers the original exact-ring W*x_j
```

**None of the reviewed constructions currently supplies this functionality
with cheap client expansion for arbitrary checkpoint W.** Theorem 5.6 and
Appendix A.1 of 2025/1223 program randomly sampled sparse generator inputs.
They do not supply an inverse encoding for an arbitrary rectangular W. A public
random element of a structured algebra cannot simply be replaced by W under the
same assumption. Circuit-dependent preprocessing is a lead, but its full client
evaluation and setup still require accounting.

The concrete missing result is a **fixed-linear-map masked-image compiler**:
bounded public data and separate role keys, finite capacity, security for each
role's complete multi-index view, and client expansion without a fresh dense
`W*r` or `D*B_client` per row. Small seeds, extra RAM and fewer selected stages
do not supply that result.

## Medium: exact local generation is a compute tradeoff

The [Medium investigation](medium-correlation-cartridge-reference.md) specifies
a simple construction in which Client alone holds a private input-mask root,
both roles can derive an independent correction q, and Client computes
`s=W*r-q`. The algebra works, but Client must hold or page W and perform another
matrix product for every row.

On its separately identified historical 150-token cold/repeat geometry:

* New public signed-i8 W delivery is **357,826,560 bytes**.
* Additional client work is **56,615,845,888 MACs per pair**, before expansion,
  memory movement and ordinary client work.
* With original quotient codecs, the conditional 20% all-link saving starts at
  **six pairs**; full compiled-ring widths move that horizon to **ten pairs**.
  These are application-body budgets before unknown authentication, failure and
  other replacement costs, not measured gains.
* Original body computation remains remote, but the new client matrix work is
  additional. The remote-placement percentage must not hide that cost.

The bounded-column random-matrix control improves on the old fresh-per-decode
opening schedule. Within one triple it uses
`r=B_client`, `s=C_client+D*B_client`, and `q=C_inference-A*B_inference`, with
`D=W-A`. It preserves the required algebra while amortizing D over fresh
columns. Even optimistic packed-ring D deliveries cost **4.124 GB at 128
columns** or **2.103 GB at 256 columns** for the first historical pair.
Renewal at column exhaustion prevents a longer horizon from making D free.
All tested horizons reject the 20% target before PCG setup.

The published matrix-PCG construction remains over a field; the packed-ring
control is conditional algebra. The paper's natural 128-square expansion
workspace is **2–32 GiB per party**, already exhausting or exceeding Medium
before application owners. No candidate process peak was measured.

## Low: spilling preserves algebra, not cheap expansion

The [Low investigation](low-correlation-streaming-reference.md) checks direct,
in-memory and file-backed multivariate Galois-ring transforms, followed by the
trace reconstruction used in Construction 2. Moving those same operations to
local files preserves outputs. Secure FSS generation and authenticated spill
are prerequisites, not implemented by the plaintext fixture.

Its explicit unoptimized `c=5,t=27,m=2,k=64,N=3^15` schedule prices, **per party**:

* **688,747,536 bytes** of live secret disk payload, before authentication and
  renewal overlap;
* **918,330,048 bytes** of public transformed tables on local disk;
* **344,373,768,000 bytes** of logical transform-stage reads/writes alone.

Ingestion, transformed-term scans, trace accumulation, public-table reads,
output consumption, PRG work and authenticated-page management are additional.
These are analytical record counts, not measured device traffic or a lower
bound against a better algorithm.

An 8–16 MiB scratch window remains a hypothesis. It does not remove the global
transform, establish whole-client RAM, or turn one requested scalar correlation
into demand-only preprocessing. Dory's prefix-sum shortcut uses a different
correlated tree; it does not directly supply the Fourier-weighted sums needed
here. Low qualification remains open.

## Selective deployment: a narrow, workload-dependent allowance

The [selective investigation](selective-correlation-budget-reference.md) joins
source-locked stage shapes, ring/residue layouts, compiler semantic roles,
prefill/decode demand and recorded placement. It targets **bulk `mlp_gate_up`
prefill**, layers **0–22** in this model. The terminal one-row group, decode and
other stages retain their original prepared contract in this hypothetical split.
Original body MACs remain at least 90% remote separately in prefill and decode;
new client expansion is additional work.

| Control | Optimistic free-removal maximum, first pair | Warm, all original public delivery reused | 20% first-pair allowance crediting mandatory payload only |
| --- | ---: | ---: | ---: |
| 150 tokens, unplaced | 19.536% | 26.153% | Negative |
| 150 tokens, placement | 20.427% | 30.465% | 1.868 MB |
| 288 tokens, unplaced | 23.053% | 27.304% | 24.300 MB |
| 288 tokens, placement | 25.102% | 31.817% | 37.439 MB |

Each row retains its own cohort and median-total representative; rows are not
pooled. Free-removal maxima deliberately credit the selected subset with **all
unallocated correction residuals**, producing an optimistic upper bound. The
last column credits only mandatory exact payload. It is a conservative allowance,
not a hard upper bound if a later construction proves additional overhead
removable. The original partial attribution stays partial.

The 150-token unplaced case cannot reach 20% even under the optimistic bound.
Placement barely clears that threshold before implementation costs. **35% is
impossible for this subset in all four controls**, including the optimistic warm
endpoint. Broader subsets change the construction and ownership scope.

The first one-stage construction-cost target is semantic **layer 18 gate/up**,
shape **9,728 by 896**, with the original **24-bit ring**. Its exact output-width
histogram gives the highest payload per output lane among the bulk groups.
One stage cannot achieve the complete-pair target; it supplies a concrete shape
for the missing fixed-W algorithm and cost proof. Gate/up-prefill matrix controls
still cost **2.458/3.661 GB** in openings at 150/288 tokens, including the terminal
row, plus **30.080/57.745 billion** client adaptation MACs.

For construction screening, the **288-token placement workload** offers the
largest first-pair allowance among these controls: 37.439 MB when crediting only
known selected payload. Start there after the small-matrix proof gate. The
150-token placement case's 1.868 MB allowance is a subsequent, stricter target;
neither allowance is evidence that a generator can meet it.

## Next construction decision

The useful next gate is an explicit fixed-W construction, rather than another
whole-model run of a scalar primitive. Start with one arbitrary signed 3-by-5 W
over `Z/2^24`, retaining the proposed security parameters. Specify public data,
both keys, generation and expansion algorithms, finite index capacity and the
complete multi-index security statement. A small matrix tests the interface;
it does not justify reducing cryptographic security dimensions.

Reject a purported cheap-client specialization if it hides a fresh dense client
product or a Q-row output tape. A successful specification must also price the
generation and distribution of public data, both role deliveries, renewal,
discarded capacity and client work. Only then does a native stage prototype
have a construction to test. Fresh whole-client peaks and complete request
costs are subsequent gates.

The W-only alternative is already a concrete algebraic tradeoff for workloads
willing to pay the extra client computation. Measuring that tradeoff could
establish its practical cost, but cannot validate the missing cheap-expansion
construction.

## Evidence boundaries and audit

The Medium budget uses its historical **493,575,247-byte** application-body
pair, with **124,905,091 bytes** of common public delivery. Later Low-tier
control cohorts have their own measured totals; they cannot be pooled with
that historical geometry. Public reuse is a conditional horizon assumption
applied to both candidates. TCP streams, application bodies and physical wire
retain separate scopes. Partial base/output/unattributed decompositions do not
become complete per-token headline rates through this analysis.

The parent audit corrected three concrete issues before retaining results:

1. Published matrix-PCG setup and workspace estimates apply only to their
   **128-square** parameter point. The **256-square** dimension sensitivity now
   leaves those estimates unknown, preserving its exact opening geometry.
2. The Low ledger now separately charges **11,479,125,600 bytes** of reads of
   transformed cross-terms. Final output-consumption I/O stays explicitly
   mapping-dependent; 344.374 GB remains transform-stage-only.
3. Selective mandatory-payload allowances are conservative, not necessary upper
   bounds. The evidence separately retains optimistic all-residual ceilings at
   every horizon. A negative mandatory allowance alone cannot reject a candidate;
   the 150-token unplaced rejection survives the optimistic ceiling. Matrix
   controls explicitly include the terminal row, unlike the recommended subset.

All three probes pass **105 focused tests** and Ruff in the parent run. Fresh
CLI replays match all three frozen JSON reports; the selective full replay also
matches its published canonical hash and source/implementation locks. These tests
establish the stated algebra and ledgers; candidate cryptography, process peaks,
CPU times and serving savings remain unmeasured.

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q \
  tests/test_medium_correlation_cartridge_probe.py \
  tests/test_low_correlation_streaming_probe.py \
  tests/test_selective_correlation_budget_probe.py
```

The linked investigations give each probe's CLI and source-pinned record. Replay
outputs must use new paths. No implementation or security claim follows from
passing an ideal-dealer or plaintext external-storage control.
