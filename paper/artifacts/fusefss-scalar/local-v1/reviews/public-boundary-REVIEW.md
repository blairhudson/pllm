# Independent public-boundary specialization review

**Verdict: no P1/P2 algebra, provenance, lifecycle or allocation-envelope blocker
found on the recorded snapshot.** Engineering/algebra review, not independent
cryptographic, side-channel or model certification. No production/helper/checker
files edited. All new artifacts/builds remain in this directory.

## Priorities and scope limitations

* **P3 — distinguish public-table specialization in documentation.**
  `crates/pllm-garble/src/fusefss_backend/mod.rs:3-10` and its
  `README.md:31-43,45-54` still describe only private-vector issuance: no table to
  parties, M independently issued vector keys, and existing-family lookup not
  used. PublicBoundary intentionally gives each evaluator **original immutable
  public** boundaries/coefficients and one universal scalar DCF key. Fix wording
  by backend; retain prohibition on publishing input-dependent rotated order,
  thresholds, correction constants, mask or selected clear coefficients. This
  is a documentation mismatch, not observed private-value disclosure.
* **Qualification boundary, not a code defect:** this is a specialization for the
  actual public frozen descriptor, not an implementation of arbitrary private
  `(B,v)` Gen. Internal public refinement may translate into a different explicit
  interval list than Protocol 3 would emit at the sampled mask; equivalence is
  functional over the whole ring, not byte/partition/transcript identity. Do not
  advertise generic private-LUT support or source optimized-key/cost identity.
* Public table payload, private material, live capacities, cumulative copying and
  DCF path work are different quantities. One key does **not** mean one DCF
  evaluation. This backend retains **512 evaluations / 25,600 levels per lane per
  party**, plus fresh product. No full protected fixture replay or cap increase
  was attempted here.

## Whole-ring compiler mapping / public duplicate proof

References: `fusefss_backend/compiled.rs:44-78,85-95,116-139`;
`fusefss/compile.rs:291-358`.

1. `Descriptor::gelu()` admits only the frozen canonical descriptor. PublicBoundary
   constructs its table from **that descriptor's actual `compile(0)` handoff**.
   The zero is public construction metadata, not an activation-dependent mask.
   The public table is immutable, privately owned by CompiledGelu, with no setter
   or externally supplied table argument on this public adapter.
2. The first eligible mask-zero duplicate splits `[0,64)` into `[0,1)` and
   `[1,64)`, both coefficients `(constant,linear)=(0,1)`. A partition refinement
   duplicating a payload preserves the selector function **for every ring word**.
   Other cuts remain exactly the independently solved signed-floor GELU cuts.
3. Our public-API check verifies complete actual 513-boundary geometry, duplicate
   payload equality and **1,024 interval endpoint values** against the original
   unfused GELU equation using hash-checked certified table bytes. Floor-bin
   inequalities ensure `(D,min(abs(floor(X/64)),255))` is constant within each
   original bin/tail; hence this is an affine proof on every interval, not a claim
   that endpoint sampling alone proves an arbitrary function's interior.
4. For any sampled input mask r, the public-table lookup computes `F((u-r) mod N)`.
   Protocol 3's shifted/padded lookup computes the same function. Rotation is a
   bijection; public duplicate padding therefore preserves the **whole A50 map**.
   This includes the negative clipped tail, signed half-ring transition and total
   ring extension. It does not prove signed38 frontend provenance.

No input-dependent table rotation/order was found in an evaluator owner.
`issue_public` passes fresh r only to private universal key/correction issuance;
party table copies remain the same original public table.

## Universal DCF / hidden correction derivation

References: `interval_fss.rs:70-112,132-148`;
`shared_affine_lookup_reference/ring.rs:183-205,288-325`.

Let `u=(x+r) mod N`, `gamma=(r-1) mod N`, p a public boundary, and write

```text
E_p(u) = [((u-1-p) mod N) < gamma] - [u > (p mod N)]
c_p    = [((r+(p mod N)) mod N) > (p mod N)]
kappa_p = c_p - [r>0] + [r+p >= N]
```

Then, in the output ring,

```text
[x < p] = E_p(u) - E_0(u) + kappa_p .
```

This is an **arithmetic difference**, not XOR-share multiplication. It handles
r=0, wrap, equality, p=0 and the wide p=N sentinel: corrections are respectively
0 and 1 at those two sentinels. No unsigned narrowing of N is used in the full
comparison identity.

For adjacent public coefficient vectors define `delta_j=v[j-1]-v[j]`. Dealer
forms and freshly additive-shares the corrected base

```text
v_last + sum_j delta_j*kappa_(boundary_j).
```

Each party adds its own shares of
`sum_j delta_j*(E_(boundary_j)(u)-E_0(u))`. Summing party outputs telescopes to
the selected original coefficient pair. Public deltas are safe here because
their order is **original public descriptor order**, not private rotated order.
Neither party receives clear kappa or corrected base: independently sampled 16
bytes provide two uniform A50 left-base shares; complementary right shares are
uniform even conditioned on that party's DCF key. The universal key hides r-1
under the existing DCF assumption; no new security proof is claimed.

Independent ideal-algebra script passes:

* 38,812 complete small `(r,u,p)` comparison cases, including sentinels;
* 392 ring50 endpoint cases;
* all 138 complete partitions of rings 2,4,8, with duplicate refinement where
  possible, every mask/input: **8,328 coefficient-selector cases**.

These counts are **ideal equations**, not native cryptographic tests. We did not
re-review/refactor the concurrently consolidated generic DCF. Exact underlying
source hash is recorded so primitive qualification can attach to this snapshot.

## Original-x product and consuming lifecycle

References: backend `mod.rs:331-347,440-535,582-660` (snapshot in receipt),
`shared_affine_lookup_reference/ring.rs:175-208`,
`shared_wide_product/ring.rs:54-83`.

* `Lookup::evaluate(self,...)` consumes either lookup owner. `advance` first takes
  Party state, replaces each lookup by `Consumed`, then evaluates it. On any
  error, already-consumed and remaining lane owners drop; no state is restored.
  RingParty's own `evaluate(self,opened)` consumes on invalid word or primitive
  error as well. Valid framed Party inputs cannot directly inject an out-of-ring
  helper argument because canonical peer shares are checked and the sum masked.
* Lookup returns shared `[constant,linear]`. `triple.open(c[1],x[i])` receives the
  original retained input share, never opened u. Consuming triple finish then
  adds constant. No implicit Q12 rescale is present.
* PrivateVector/PublicBoundary binding differs even with identical descriptor,
  source, context, lanes and controlled nonce: public binding adds
  `public-original-boundary.v1` plus all actual original table boundaries and
  coefficient words. Fixed shape gives unambiguous field lengths. Independent
  public-API test patches only the foreign nonce to match, then checks rejection
  **in both directions**; hence it tests backend binding, not just random nonce.
* Six malformed/reflected peer frames burn the owner. Sixteen real allocator
  failure injections cover eight issuance allocation sites (two outer Lane Vecs,
  four public-copy buffers, two DCF correction buffers) and all eight outer online
  allocation sites. Partial owners retire, Party stays consumed on online failure,
  and Budget returns to resident-descriptor charge after owner drop.
* Frames and Outputs preserve independent leases after both parties and source
  gate drop; frame-only leases release before final output leases. Public copies
  and private lookup allocations actually disappear during advance, while Party's
  conservative reservation remains until Party destruction.

Mask/key/triple freshness remains one instance per scalar lane. Tests use fresh
ordinary OS-random issuances, not fixed-mask fixture key reuse. Frame binding is
not a MAC; prior authenticated-transport qualification remains separate.

## Measured logical heap vs reservations

Actual public workspace API; custom GlobalAlloc counts requested live heap layout
bytes. Excludes allocator metadata/RSS, stack/register copies, caller input/output
copies and the external independent-oracle data. Default allocate-copy-deallocate
realloc conservatively includes overlap even if System could grow in place.
Reservations include named inline/scratch allowances in addition to heap. No
measured heap result is presented as a complete stack/security-erasure proof.

| Measured/charged quantity | 1 lane | 4 lanes |
| --- | ---: | ---: |
| Construction heap peak | 557,300 | 557,300 |
| Retained descriptor + original public table heap | 73,820 | 73,820 |
| Descriptor-owner reservation | 1,310,720 | 1,310,720 |
| Heap immediately after issuance | 101,980 | 186,460 |
| Issuance/online heap peak, including resident source | 102,236 | 187,004 |
| Actual held Budget charge after issuance (setup scratch refunded) | 1,350,372 | 1,435,764 |
| Combined construction/issuance admission report | 2,679,510 | 2,764,902 |
| Heap after lookup advance, all four frames held | 74,620 | 76,060 |
| Exported frames + outputs after gate/parties dropped | 384 | 576 |
| Final requested heap / Budget residual | 0 / 0 | 0 / 0 |

Public/private separation in `mod.rs:201-270`, RingResources and measured owners:

* Original public table: **12,296 payload/capacity bytes**, charged within separate
  CompiledGelu descriptor reservation. RingTable inline owner also exists.
* Each party/lane has its own **12,296-byte public table copy** (no Arc/Clone
  hidden sharing). Four lanes: 49,184 public-copy bytes per party, not private key
  material. Table capacities are counted separately from key capacities.
* Private payload per party/lane: **1,372 logical bytes** = 1,324 unit scalar DCF
  bytes +16 corrected-base shares +8 mask share +24 product triple shares.
  Actual DCF correction heap uses **1,600 bytes per party/lane**, not the encoded
  layout. Inline private owners, public table handles and triple metadata sit in
  separately charged `Lane` storage. Four-lane logical private payload: 5,488 B
  per party. Neither logical payload count is a codec or wire measurement.
* `RingTable::new`, copy construction and helper issuance verify capacities and
  derive a remaining DCF envelope before helper entropy. Outer shared Budget
  reserves both parties plus sequential issuance scratch before its first RNG.
  Reservation-minus-one rejects without creating party owners. Source table
  remains charged across concurrent issuances; copies belong to each Party.

No concrete unchecked arithmetic overflow, peer-slice panic or missing live table
copy charge found under admitted widths/counts. Coefficients/table ordering are
public; their buffers need not be treated as private erasure failures. DCF/base,
mask/triple, activation/opening/output share buffers retain zeroizing owners.

## Exact reproducible evidence

```sh
python3 -B target/fusefss-public-boundary-review-v1/run_review.py
```

Runner uses Cargo **one build job**, offline locked dependencies, 60-second native
timeout; ideal algebra has 10-second timeout. Final native command returned 0 in
**12.368154 seconds** including Cargo; seven public-API groups passed. Algebra
returned 0. All tracked production sources had **identical SHA-256 before/after**
that run. No concurrent-build blocker on final attempt.

Native numeric evidence: **45 PublicBoundary output checks** (40 direct-oracle
cases plus 1+4 heap/lifecycle cases), 8 PrivateVector controls; 1,024 actual
compile(0) endpoint checks. Full fixture/model/backend security qualification was
not attempted or inferred.

Exact hashes for every tracked source appear in `receipt.json`. Main identifiers:

| Source | SHA-256 |
| --- | --- |
| `fusefss_backend/compiled.rs` | `6ac2791b9141889c2fe6411b2b84ded5d08dd36734fee4716282f9ad35343d4a` |
| `fusefss_backend/mod.rs` | `9f8b51c58a1fbf789d6bd3c24771974c4fd03949e41cc8bc673a98b5f9da32f4` |
| `shared_affine_lookup_reference/ring.rs` | `60cc43e00d0c2f4d82ba7936967bd0f53ae7ac687e47ef5528b9d10d8c05d734` |
| `interval_fss.rs` | `a94762bfc2089c53c9e88faf8b509acbc2524b274861ef957619299539abd226` |

Paths in hash table are under `crates/pllm-garble/src/`. Source commitments:
canonical GELU descriptor `5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3`,
table-u64LE `b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b`.
Receipt records reviewed critical sources, not a full transitive build SBOM.
