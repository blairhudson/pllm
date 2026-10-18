# PUMA Q18 multiply → TruncAPr: bounded specification

Frozen specification/oracle slice, 17 October 2026. Implements public integer
fixtures for the already-approved [D6 profile](../../methods/DECISIONS.md#d6-approved-puma-source-derived-q18-profile)
and [next slice](../shared-profile-resolution.md#71-next-bounded-slices-after-the-native-freeze).
This is neither a private runtime nor a model execution/graph-range certificate.
All code here is independent, standard-library-only Python. No upstream runtime
or existing PLLM implementation is imported.

## Exact public range contract

Inputs are **already-encoded Q18 signed ring64 integers** X,Y; D6's finite
binary32/toward-zero external encoder remains a separate implementation task.
Admit 1–1024 equally shaped lanes with closed public integer boxes
`X∈[a,b], Y∈[c,d]`, endpoints in `[-2^63,2^63-1]`. Set

```text
L = min(a*c, a*d, b*c, b*d)
H = max(a*c, a*d, b*c, b*d)
admit iff -2^62 <= L and H < 2^62.
```

This is **necessary and sufficient for the whole Cartesian integer box**:
bilinearity puts extrema at the four corners, and those corners belong to the
box. Products are calculated as intended **unwrapped signed integers**, before
any ring reduction. The oracle uses bounded-width integer operands and wide
integer products; it rejects the complete batch's unsafe boxes before evaluating
any lane. It does not substitute a wrapped residue or silently clamp/saturate.

Examples: `(2^31)*(-2^31)=-2^62` is admitted; `(2^31)^2=+2^62` is not.
`(2^32)^2` wraps to zero but must reject. A zero operand permits the other operand's
full signed-word interval. The bound concerns X*Y, **not random share components**;
local shared arithmetic intentionally wraps in ring64. A tighter relational
domain (e.g. Y=-X) needs a separate relational proof; interval rejection is not
necessity over such an unstated domain. Individually safe products do not prove
a subsequent accumulator safe. No accumulation/coalesced truncation is included.

These are public compiler/caller obligations. A future party-local runtime must
not reconstruct/open/test private X,Y or X*Y to establish them. Clear membership
checks in this offline oracle validate public fixtures only. Actual graph-wide
membership and range propagation remain unproved.

## Exact rounding and multiplication equations

Let `P=X*Y`, `N=2^64`, `G=N/4`, `U=2^18`, fresh ideal uniform `R∈[0,N)` and
`C=(P+G+R) mod N`. The pinned TruncAPr result is

```text
T = floor((C mod (N/2))/U) - floor((R mod (N/2))/U)
    + (MSB(C) xor MSB(R))*(N/(2*U)) - G/U
  = floor((P + (R mod U))/U).
```

Proof: `P+G∈[0,N/2)`, so adding it to `R mod (N/2)` has at most one carry.
That carry equals the two MSBs' XOR, even when C wraps modulo N. Subtracting
the high-bit correction restores P+G; G and N/2 are divisible by U, leaving
exactly the low-residue carry. Thus upper rounding has probability `(P mod U)/U`,
otherwise floor; exact multiples are exact. No extra large-wrap error occurs
inside the admitted gap. The tight all-mask output interval for `[L,H]` is
`[floor(L/U), ceil(H/U)]`. In particular P=G-1 can round **up to +2^44**.
The deterministic `literal_signed` oracle is conditional on R, not a replacement
deterministic truncation. PRSS gives the source's computational analogue of ideal
fresh randomness; fixed example keys do not supply randomness/security evidence.

Party i owns only `(x_i,x_(i+1)), (y_i,y_(i+1))` and adjacent edge streams
`s_i,s_(i+1)`. It computes

```text
alpha_i = s_i - s_(i+1)
z_i = x_i*y_i + x_i*y_(i+1) + x_(i+1)*y_i + alpha_i   (mod N).
```

The zero terms telescope. Across three parties the three diagonal and six
cross terms equal `(sum x_i)*(sum y_i) mod N`. Send z_i to **PrevRank** and
receive z_(i+1) from **NextRank**, restoring `(z_i,z_(i+1))`, then apply TruncAPr(18).
Consistent replicated inputs and the source's semi-honest role assumptions are
premises, not malicious-consistency checks supplied by this oracle.

## Source pins and closed mixed-kernel schedule

[source-lock.json](source-lock.json) freezes nine exact file hashes and line spans:

- SPU: `a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4`.
- Its pinned YACL: `5418371c4335f4a64fbd0bdabb0efd94da2af808`.

`arithmetic.cc:425–459` fixes MulAA; `735–941` fixes opening/TruncAPr;
`communicator.h:134–146` fixes rotation. `prg_state.h:62–114` fixes pair controls.
For n lanes, `q=ceil(n/2)`, pair start s and public start p, the **per-role** source
order and successful consumption are:

| Stage | Pair start/control | Other stream/work |
| --- | --- | --- |
| MulAA zero term + multiplication | s / Both, all roles | 2q AES blocks per role; then rotate |
| TruncAPr public pivot | no pair fill | public p, one u64, cursor +1 |
| TruncAPr mask | s+q / A First, B Second, H Both | q, q, 2q AES blocks respectively |
| H correction | no pair fill | one private 2n-word fill: rb[0:n], then rc[0:n]; H cursor +n |
| TruncAPr output masks | s+2q / same controls | q, q, 2q AES blocks respectively |

Successful totals: pair cursor **+3q**, public **+1**, only H private **+n**;
AES work **4q+1** per evaluator and **6q+n+1** for H. There are **nine directed
frames / 11n ring words** before headers/setup/transport. No extra global barrier
is inferred. The offline fixture calculates public metadata ahead for convenience;
it is not a runtime execution trace authorizing early public draws.

MulAA's alpha already is a zero sharing from its **one** Both fill. Do not insert
a second fill. Source P2A's `ENABLE_MASK_DURING_ABY3_P2A` debug branch (`181–224`)
separately adds one Both fill and one rotation; ordinary P2A does neither.
Already-shared inputs exclude that initializer here. A graph using it must charge
it explicitly. Arbitrary other kernels, `None`, forks, concurrency/batch splitting
and compiler transformations have **no established composed schedule here**.

Each AES fill resets to LE128(cursor), increments counter blocks big-endian
within that fill, reads LE64 words and discards unused half-block bytes. Both
expands adjacent keys at the same start and advances the logical cursor only once.
The two overflowing u64 pivot words reject/burn without resampling, matching the
[bounded truncation contract](../../../docs/research/puma-truncation-reference.md).
The oracle keeps the existing 4096-operation / `(3*1024+3)*4096` per-role AES-work
ceilings; it does not raise them to fit longer compound schedules. Capacity
checks here are public arithmetic, not an allocator/RSS or live-retirement proof.

## Identifier and next native boundary

A **new execution/context namespace is required**:
`puma.spu-a42c261.mul-aa-truncapr18.ring64.quarter-gap.replicated3.v1`.
Keep numeric profile `puma.source-derived-q18.v1` and existing PRSS algorithm ID.
Do not relabel this as the frozen truncation-only owner: its pair consumption was
2q, it owned only one operand, and it had no multiplication/rotation phase.
This identifier distinguishes orchestration, not a changed numerical algorithm.

Future contexts must bind that ID, fresh setup/session, role, complete graph/
schedule digest, phase/ordinal, n/Q18, public interval envelopes, and distinct
`mul.zero`, `trunc.mask`, `trunc.output` cursor ranges. Metadata must not become
a new PRSS KDF or alter the source AES inputs. JSON canonicalization below is
an **offline fixture format**, not an authenticated native wire codec.

Remaining gates:
1. A source-ordered, non-cloneable compound owner: only two legitimate adjacent
   seeds per role; fresh OS setup; no seed reimport/reset; explicit phase tickets,
   replay rejection and sticky failure/cancellation burn. Success-path cursor
   algebra is proved here; native reservation/retirement behavior is not.
2. Joint live-owner/payload admission before issuance: X/Y input pairs, zero-share
   buffers, original rotation frames, product shares and all truncation payloads.
   Preserve the **256-KiB local ceiling** and leases until wipe/free; the old
   truncation-only scratch estimate is not a compound-owner bound.
3. Authenticated confidential transport, abort propagation, then ordinary
   [D5 SDK integration](../../methods/DECISIONS.md#d5-ordinary-sdk-integration-is-required):
   immutable component config, discovery/round trips, Pipeline/Experiment/compiler,
   actual client/gateway/CLI and benchmark/replay. No separate public runner.
4. Actual graph/checkpoint ranges, arbitrary mixed-kernel schedules, complete
   D9 protected feedback/decoder and independent quality/cost evidence. D6 approval
   remains closed; unavailable evaluated-build identity remains unavailable.

## Reproduce and frozen evidence

From repository root:

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts/puma -p 'test_oracle.py' -v
python3 -B paper/reproduction-contracts/puma/fixtures.py
```

**14 tests passed**, offline (about 1.4 s locally): 2025 interval boxes; 262144
input-share combinations over Z/8Z and 512 zero-share triples; 247552 small-ring/mask
equations; every Q18 low residue; all 1024 lane sizes/three pivots; wrap/endpoint,
shape/cursor/work rejection, FIPS AES known answer and within-fill carry at 256.

[vectors.json](vectors.json) freezes literal signed outputs, multiplication
components, replicated outputs, exact plans/cursors and full-transcript hashes.
[fixtures.py](fixtures.py) reproduces every underlying PRSS tape and all nine
frames for sequential n=3,1,4 fixtures; the large ring words are hexadecimal and
signed values are decimal strings. Canonical JSON uses sorted keys, compact
separators, ASCII, **no trailing newline** for hashing. Its complete digest is
`1e60a53b6049faa7f3c5f2cce9908d7545edbbbb9b55324b40ec9a66bfe4b167`.
All three full transcripts were independently cross-checked against system
OpenSSL AES-128-CTR during preparation. Reproduction/tests require only the
independent [AES arithmetic reference](aes_reference.py), not OpenSSL/upstream.
[FROZEN.sha256](FROZEN.sha256) seals this directory's seven specification files;
the manifest excludes itself. Existing native inputs were not edited.
