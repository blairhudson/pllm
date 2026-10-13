# Low-tier exact-ring correlation expansion: streaming boundary

## Result

**Exact external-memory expansion is algebraically possible without full resident
output buffers. Cheap demand-driven expansion is not established. Low admission
is blocked.** The strongest opportunity is to traverse FSS trees in coefficient
order, spill locally, run the existing multivariate transform in external memory,
and stream the final coordinate-local trace/reconstruction. This preserves the
same outputs and hence their original distribution, conditional on using the
original secure keys, parameters, and authenticated storage correctly. It does
not create a new PCG, a fixed-weight specialization, or a measured Low client.

The required consumer remains Client `(r_j,s_j)` and Inference `W*r_j-s_j`, with
fresh finite indices. Construction 2 of 2025/1223 supplies scalar OLEs; its
programmability statement does not by itself specify this asymmetric arbitrary
fixed-`W` expansion. The fixed-weight construction is a separate gate. Everything
below is an expansion/storage investigation conditional on that missing gate,
not evidence that arbitrary fixed `W` is programmable or cheap.

[Evidence](../evidence/research-low-correlation-streaming.json) pins both PDFs
and their full-text extracts. References are **1-based PDF pages**. The source
map and broader contract are in [the full-text review](compact-setup-fulltext-review.md).

## What the sources actually permit

* **2025/1223 pp. 19–23:** the evaluation map `phi` is an isomorphism from
  `GR[X_1,...,X_n]/(X_i^d-1)` to `GR^(d^n)`. Frobenius and trace commute with
  this map. This makes the final arithmetic coordinate-local, not the transform
  itself. For `p=2,m=2,d=3`, use `GR(2^k,2)` with `z^2+z+1=0`. Do not substitute
  a length-`3^n` cyclic transform over the base ring: `3^n` does not divide
  `2^m-1`. There are `n` independent radix-three axes on `Z_3^n`.
* **2025/1223 pp. 28–29, Construction 2 / Eq.(3):** the FSS outputs are
  coefficient arrays. Each party's share is dense even though the combined
  cross-term is sparse. Applying public multipliers is convolution in coefficient
  space and pointwise multiplication only after evaluation. FullEval followed
  by multiplication/trace cannot simply become independent small coefficient
  tiles. Trace is local after transformation; it cannot remove global dependence.
* **2025/1223 pp. 34–36:** retain Table 3's regular-noise point
  `lambda=128,c=5,t=27,m=2,k=64,n=15,N=3^15`; `n=16` is also listed. The executable
  model refuses other `n` values rather than claiming their security. The separate
  Table 5 `109,000/s` point uses `(k,s)=(32,26)` and `c=5,t=27` on a machine with
  512 GB installed RAM. Neither it nor installed capacity is our RSS measurement.
  The roughly 115 MB/billion claim uses the **80-bit** `c=9,t=9` point. The
  `c=3,t=27,m=2` benchmark is broken and is never an admission option here.
* **2025/1660 pp. 11–15:** Dory uses correlated GGM children `H(v),v-H(v)`.
  A subtree's leaf sum equals its root, so prefix accumulation needs one path
  and sibling sums rather than all leaves. Its punctured receiver has a different
  accumulation algorithm; copying only the sender shortcut is not a protocol.
  Sparse expand-accumulate queries access these prefixes locally. Ordinary BGI
  FSS trees in the exact-ring paper do not have this subtree-sum invariant.
  In particular, arbitrary Fourier-weighted leaf sums are not supplied by Dory.
* **2025/1660 p. 21, Tables 2–3:** corrected LPN2 uses `N=51,478,528`, `t=1,571`,
  `log2(N/t)=15`, `ell=7`, capacity `10,295,705`, 128-bit security. The malicious
  eight-thread row has **20 MB maximum per-party memory**, `0.419 MB` base plus
  `0.389 MB` silent communication, both sides combined. This implementation is
  `F2` into `F_(2^128)` COT, not exact-ring matrix correlations. Malicious setup
  also recomputes trees for consistency checks (pp. 7, 15, 22). Neither 20 MB
  nor a proposed ring conversion qualifies a 256 MiB whole application.

2025/787 does not provide a missing exact-ring transform or storage schedule, so
it is not used to fill this gap.

## Concrete schedule and dependence

1. Bind the immutable public tables, key epoch, parameters, output layout, source
   digests, capacity, and role separation. For the straightforward `c=5`
   generalization on p. 34, write `b=sum_l a_l*e_l`, `a_0=1`, with five sparse
   error terms and four public random multiplier polynomials. Precompute their
   evaluation tables once; retain four tables on **local public disk**, not RAM.
   Retire original coefficient copies after validating the transform/digests if
   they are no longer needed. Keeping both representations adds disk costs.
   Replacing uniform public tables with small PRG seeds is not assumed here.
2. Traverse each original multipoint FSS tree depth-first, retaining keys,
   correction words, and a traversal frontier; emit coefficients in canonical
   base-three lexicographic order (least-significant axis first). Pad the binary
   FSS domain only as its original construction specifies; do not alter noise,
   security dimensions, or which domain indices are valid. A serial traversal
   can reduce frontier storage, but exact PRG scheduling/key serialization for
   the published optimized `c=5` implementation has not been audited. Its tree
   and key bytes therefore remain unknown, not zero.
3. Ingest one FSS array into encrypted local slot A. Transform one axis at a
   time, reading three coordinates separated by `3^axis`, applying a three-point
   Galois-ring butterfly, and writing slot B. Alternate A/B for all 15 axes.
   At the logical-record level this has two full disk vectors and six live GR
   elements, plus keys, I/O pages, AEAD state and scheduling buffers. On real
   authenticated pages, high-stride accesses require a page-cache/transposition
   schedule; six live algebra elements is **not** a native RAM claim.
4. Scan the transformed term and corresponding public multipliers, applying
   Frobenius to each public GR value locally. Multiply by the normal-basis
   coefficients in Eq.(3), take the trace, and add to the two base-ring `z` lane
   accumulators on local encrypted disk. They occupy one GR-vector-equivalent
   payload. Reuse the two transform slots for the next cross-term. The
   straightforward unoptimized generalization has `m*c^2=50` cross-term arrays;
   factoring, shared transforms and original optimized FSS evaluation may change
   this count. It is a concrete analytical schedule, not a claimed published
   implementation count or lower bound.
5. Once all terms have contributed, emit `(x,z)` coordinates in a declared order.
   Obtain `x` by evaluating the original sparse error polynomials against the
   public tables at that coordinate, then tracing. This avoids retaining the
   full `x` output array, but sparse evaluation and complete client expansion
   work remain unpriced. Basis-major order is simplest; coordinate-major order
   can emit both basis lanes together. A fixed-W consumer must map its vector
   indices/layout explicitly; scalar OLE capacity is not fixed-W vector capacity.
6. Reserve/burn the consumer issuance range before exposure, keep only its
   bounded output/reply window, and retire consumed outputs. Unconsumed `z`
   coordinates remain secret disk state until issuance or epoch destruction.
   Restarting a transform from immutable keys reproduces the same coordinates;
   it does not renew their one-use permission.

**Why this is exact:** each radix stage applies the same separable evaluation
map as full expansion. All intermediate operations are in the original Galois
ring. Frobenius is the ring automorphism `z -> z^2`, not squaring arbitrary
elements modulo `2^k`. Trace commutes with evaluation. Consequently reordering
independent operations and moving buffers to authenticated local storage changes
no output. Distribution preservation follows from equality for every original
key, not from a new randomness experiment. No proof or implementation of secure
FSS keys, authenticated storage, or the fixed-W specialization is provided.

**Alternative recomputation:** one demanded transform coordinate can be computed
by traversing every FSS leaf and accumulating its root-of-unity-weighted value.
For each of 50 terms, naive direct evaluation costs `N` GR multiplications and
`N-1` GR additions per requested coordinate, plus the complete FSS traversal
and weight generation. Across requests it repeats this work unless results or
frontiers are cached. It is correct and bounded-state but not cheap random
access. Computing multiple demand coordinates together adds accumulators and
can amortize traversal; no Dory-like shortcut or latency result is established.
This is an alternative to global preprocessing, not an impossibility theorem.

## Executable byte and work ledger

The model uses tightly packed 64-bit limbs, not Python object sizes. `N=14,348,907`
and one GR vector is `V=16*N=229,582,512` bytes (**218.95 MiB**).
The expansion ledger is **per party/client**, not combined two-party expansion.

| Owner / operation | Exact analytical amount |
| --- | ---: |
| Public transformed tables on client-local disk | `4V = 918,330,048` bytes |
| Secret alternating transform slots on client-local disk | `2V = 459,165,024` bytes |
| Secret two-lane trace accumulator on client-local disk | `V = 229,582,512` bytes |
| Secret payload disk peak, no renewal overlap | `3V = 688,747,536` bytes |
| Full resident scalar `(x,z)` outputs, if materialized | `2*m*N*8 = 459,165,024` bytes |
| 50 forward transforms, GR multiplications | `50*3*n*N = 32,285,040,750` |
| 50 forward transforms, GR additions | `50*2*n*N = 21,523,360,500` |
| 50 forward transforms, logical record I/O | `50*2*n*V = 344,373,768,000` bytes |
| FSS-array ingestion writes | `50V = 11,479,125,600` bytes |
| Transformed cross-term scans before trace accumulation | `50V = 11,479,125,600` bytes |
| Trace accumulator read/write envelope | `100V = 22,958,251,200` bytes |
| Public multiplier read envelope, two per cross-term | `100V = 22,958,251,200` bytes |
| Four public table transforms, logical record I/O | `4*2*n*V = 27,549,901,440` bytes |

The arithmetic counts deliberately use nine GR multiplies and six adds for each
three-point butterfly, including multiplies by one. Optimized butterflies may
use fewer. The two-public-read envelope charges even terms involving `a_0=1`;
identity factors can be elided. These are transparent schedule counts, not wall
time, disk-device traffic, or the complete operation count. FSS PRG work, public
sampling, page management, local trace arithmetic, sparse `x`, authentication,
and setup/renewal communication are additional owners.
The `344,373,768,000` bytes count is transform-stage-only I/O; it excludes the
separately listed ingestion, transformed-term scans, accumulator and public reads.
Final `z` accumulator reads and `(x,z)` output consumption are separately marked
unknown in the evidence: their access order, retention, vector mapping and
sparse/public `x` inputs depend on the eventual consumer. They are not free or
included in the transform-stage count. No total full-expansion or disk-device
traffic claim follows from this partial ledger.

The p. 34 approximate seed expression for OLE is
`m*c^2*t^2*(log2(N)-log2(t)+1)*(lambda+2)+lambda+k*m+c*t*(log2(N)+k*m)` bits.
Replacing `log2(N)` by its ceiling and `log2(t)` by its floor gives a conservative
integer envelope **of that approximate expression**, `12,441,160` bytes. It is
not a serialized-key upper bound. Table 3's separately rounded `log2(seed bits)`
is `26.60` (about 12.71 million bytes), underscoring that the formula envelope
cannot certify implementation allocation. The evidence reserves two formula
envelopes during renewal only to illustrate headroom, not to qualify ownership.

| Extra scratch hypothesis | Scalar `(x,z)` pairs per window | Optimistic remainder after two formula seed envelopes |
| --- | ---: | ---: |
| 8 MiB | 524,288 | 235,164,528 bytes |
| 16 MiB | 1,048,576 | 226,775,920 bytes |
| 64 MiB sensitivity only | 4,194,304 | 176,444,272 bytes |

Windows are total scratch hypotheses, not additional uncharged output buffers.
These pair counts assume the entire window held outputs; real I/O/frontier use
reduces them. A larger Medium-style window changes neither full-batch transform
work nor logical I/O in this schedule. No separate fixed-W Medium analysis is
performed here. A demand for one scalar pair still preprocesses all `2N`
`z` coordinates: **28,697,813 unrequested logical scalar OLE positions**. It
need not eagerly compute their `x` values. Partial demand therefore does not
turn global preprocessing into demand-only generation.

### Authenticated local spill is an owner, not free RAM

The executable storage control uses **plaintext synthetic fixtures only**. A
real design must encrypt and authenticate secret pages locally, bind epoch,
array identity, stage, coordinate range and write generation into authenticated
metadata, and use non-repeating nonces for every rewrite. Authentication alone
does not prevent replay: page versions and manifests need rollback protection.
Public tables need integrity binding too. No secret-dependent paging request
goes to Inference or Preparation; all reads, keys and caches stay client-local.

A hypothetical 4,096-byte payload page plus stored 12-byte nonce and 16-byte tag
gives `3*ceil(V/4096)*4124 = 693,462,972` secret disk bytes, before manifests,
filesystem overhead, retained prior generations and renewal. In a deliberately
uncached record-at-a-time schedule, each logical 16-byte read/write transfers a
4,124-byte page: transform traffic becomes **88,762,338,702,000 bytes**. This is
a sensitivity scenario, not observed traffic or a lower bound. A page-aware
transpose/tile cache is essential to approach the logical-record envelope;
its extra passes, temporary files, nonce/tag work and cache occupancy need a
new ledger. No such native encrypted schedule has been measured or built.

RAM must include SDK, existing state, both epoch keys, FSS frontiers/correction
words, AEAD keys and metadata, page caches, public-table windows, output/reply
buffers, allocator/runtime overhead, and renewal overlap. Disk must include
live old accumulators plus new transform slots, interrupted writes, manifests
and encrypted outputs. Full public residency alone plus other owners threatens
Low; full secret output residency already exceeds Low. Spill can remove those
resident arrays, not their work, device traffic or lifecycle requirements.

## Finite capacity and retirement prerequisites

Capacity is bounded by the original setup. The scalar model has `mN` positions;
an actual fixed-W vector consumer must charge its own packing and all positions
it consumes. Reserve ranges durably before either party exposes openings.
Cancellation, malformed replies, failures and expiry burn reservations. Renewal
uses genuinely fresh keys/correlations and both role deliveries; neither a new
label nor rerunning expansion renews old positions. Stop issuance on exhaustion
unless renewal has completed and authenticated checks have passed.

Restart, backup restoration and cloning require an external monotonic allocator
or equivalently audited anti-rollback ownership. A Python counter, encrypted
local file, or counter authenticated by a rollbackable key is insufficient.
Old/new epoch output windows, keys, transform files and dealer/proof costs overlap
unless renewal drains the old epoch first; draining instead introduces a stall.
On aborted renewal burn anything already issued or reserved and destroy its
partial secret files/keys. Cryptographic erasure must account for backups,
snapshots and filesystem remnants. These are prerequisites, **not an implemented
production owner** or a claim that deleting a toy file ensures erasure.

## Controls, reproduction and next gate

`scripts/probe_low_correlation_streaming.py` independently compares direct
multivariate evaluation, in-memory radix stages and packed local file stages at
`n=1,2,3`, base-ring widths `1,8,32,64`. Twelve cases verify inverse reconstruction,
schoolbook group convolution, and Frobenius commutation. Ideal additive cross-term
fixtures verify **312 scalar OLE reconstructions** from Construction 2 Eq.(3).
Exhaustive tiny domains verify transform and normal-basis trace bijections; they
are algebra controls, never smaller proposed security dimensions. No synthetic
generator is described as a secure PCG. Logical record counts are not Python
allocation tracing or whole-process peaks. No timing or performance benchmark
was performed.

The toy file helper retains its separate source, two stage files and a final
output copy, and counts that copy's I/O. The analytical schedule instead ingests
FSS directly into one alternating slot and consumes the final slot without a
copy. The toy verifies algebra/record mechanics, not the schedule's full disk
owner or its encrypted lifecycle.

```sh
target/frontier-wheel-env/bin/python scripts/probe_low_correlation_streaming.py \
  --output target/low-correlation-streaming-replay.json
target/frontier-wheel-env/bin/python -m pytest -q tests/test_low_correlation_streaming_probe.py
target/frontier-wheel-env/bin/python -m ruff check \
  scripts/probe_low_correlation_streaming.py tests/test_low_correlation_streaming_probe.py
```

The CLI refuses existing evidence outputs and changed source hashes. Tests
compare regenerated JSON content with the checked-in immutable evidence; JSON
whitespace is not evidence identity. All scratch uses the isolated ignored
`target/low-correlation-streaming*` prefix.

Validation: **27 focused tests passed**, Ruff passed, source SHA-256 locks
matched, and the separate ignored CLI replay matched the checked-in JSON content.
Canonical sorted/minified JSON SHA-256 is
`1da3417e68aba268b2417e969bdf6b495b8216f2d50d214c4cf8011fbfbc431b`.

**Next experiment:** after the fixed-W proof/algorithm gate, audit the exact
optimized `c=5` FSS key layout and design a native authenticated page-aware
radix schedule. First establish all key/frontier/page and renewal bytes plus
complete GR/PRG/I/O counts; only then authorize a bounded fresh-process memory
measurement, without shrinking security parameters. **Stop** if complete client
ownership cannot plausibly remain below 256 MiB, or if the transform/recomputation
ledger cannot meet the application work/I/O contract. Current result stops before
Low qualification: neither a secure native expansion owner nor a fixed-W
specialization nor an authorized peak measurement exists.
