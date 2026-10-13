# Matrix-triple PCGs: random-to-fixed prefill adaptation rejects

**Outcome:** the strongest priced fresh-random-triple consumer is functionally
correct, but dense `D=W-A` openings cost **31.55 GB (optimistically packed rings)**
or **41.60 GB (field31)** per historical pair even with 150-column prefill tiles.
This exceeds the entire 493.58 MB cold control before seeds, helpers or renewal.
The matrix-specific dimension-compression claim is real at abstract level; it
does not compress these chosen-matrix openings. No PCG is implemented or ranked.

## Primary verification and access limits

[2026/2298](https://eprint.iacr.org/2026/2298), **Pseudorandom Correlation
Generators for Matrix Triples**, names Maxime Bombar, Geoffroy Couteau, Nikolas
Melissaris and Anuja Modi. The primary landing page says received **2026-10-01**,
approved **2026-10-04**, preprint; HTML publication/modified timestamp is
`2026-10-01T16:31:24+00:00`. No retrieval-date assertion is recorded. A numbered version cannot be
verified: the actual linked versions endpoint returned 403. The PDF returned
403 through both direct download and web fetch; alternate author-text searches
returned no results. **Only primary HTML/abstract was accessible.** No theorem,
parameter table, benchmark constant, implementation or dimension-admissibility
claim is inferred from inaccessible full text.

The abstract explicitly describes random **square field** triples `(A,B,AB)`:

* First construction: QASD assumption, seed
  `poly(lambda, log n, log N)`, `Theta(n^3 log(n^3 N))` field operations per triple.
* Second: a new Weyl-Pauli twisted-QASD variant, the same seed asymptotics,
  expansion a small constant above creating triples in the clear.

Removing seed dependence linear in matrix dimension is distinct from existing
[Preparation row batching](preparation-factory-reference.md), which still
computes and delivers `Wr-s` for each fresh row. A small constant above a clear
matrix product is not lightweight client expansion. The source does not, at
accessible abstract level, establish rectangular triples, fixed-public-matrix
correlations, reusable `A`, exact integer-ring compatibility, finite setup size,
malicious security, or a concrete one-time seed capacity.

The primary HTML UTF-8 SHA-256 is
`9d38ef537bed564d1fdfddc3a6687086e1b3b477ecdd7ebff62472edbb800e00`
(18,583 bytes); **this is not a PDF hash**. Evidence also hashes a canonical
verified-metadata snapshot and all replay/source-geometry files. HTML hashing used
a SHA-256 routine checked against the standard `abc` vector in the fetch runtime.

[2025/1223](https://eprint.iacr.org/2025/1223), by Zhe Li, Chaoping Xing, Yizhou
Yao and Chen Yuan, is **Efficient Pseudorandom Correlation Generators over
Z/p^k Z** (received July 1, approved July 7, 2025). Its primary abstract does
claim Galois-ring OLE, authenticated triples, circuit-dependent preprocessing
and matrix multiplication applications. Thus field-only impossibility is not
claimed. It does not certify the new twisted construction over `2^k`, or provide
finite ring matrix-PCG costs in this screen. Its landing HTML SHA-256 is
`1b77ecdec75fac21664c7e6d5202713466d9c7dc37d26aa5c43e5fca78a0fa5a`.

## Complete preferred-topology consumer

Client holds private `X` of shape `n by t`. Public signed-i8 `W` is `m by n`.
Trusted, noncolluding, offline Preparation supplies an ideal random triple:
`A (m by n), B (n by t), C=AB (m by t)`, additively shared between Client and
the **one online Inference** role. Preparation never receives online openings.
The rectangular contract here is an optimistic functional oracle, not verified
support in the paper. Let subscripts `c,i` denote role shares.

```text
W_c=0, W_i=W
d_c=-A_c, d_i=W-A_i
D=d_c+d_i=W-A                     # exchange two dense opening shares
H_c=C_c+D B_c, H_i=C_i+D B_i       # H_c+H_i=WB
e_c=X-B_c                        # Client -> Inference only
E=e_c-B_i=X-B                    # computed only at Inference
Y_i=H_i+WE                       # Inference -> Client
Y=H_c+Y_i=WB+W(X-B)=WX
```

All arithmetic is in the selected domain. Assigning the whole public `WE` term
to Inference eliminates the reverse `E` opening of generic Beaver. The priced
consumer therefore does **not** charge an unnecessary second input message.
The public `D` requires two dense role-share messages; an alternative dealer
broadcast of `D` to both endpoints also costs two dense deliveries. These values
can be prepared offline, but remain fresh recurring material, not one bootstrap.
An opening only to Inference would leave Client unable to compute `D B_c`.

These byte floors apply to the **specified explicit dense-opening layout**, not
to every imaginable PCG consumer. A real construction might compartmentalize
short **A-only** generation keys that may safely be disclosed to both roles,
letting each reconstruct `D` locally without transmitting its entries. No such
key interface or security composition is established by the accessible abstract.
Disclosing the whole role's triple seed is invalid if it reveals that role's
`B`/`C` shares to the other role. The next full-text gate must inspect this exact
possibility; this report does not use an information-theoretic entropy argument
to dismiss computationally compressed `A` openings. Even safe A-only opening
keys would leave both dense `D B_role` adaptations and full expansion cost.

Because `W` is public, opening `D` reveals the **entire random A to both roles**.
This is not itself an activation leak under ideal independent additive sharing:
Inference knows `B_i,C_i`, not `B_c,C_c`; total `AB` remains hidden by the other
uniform `C` share. Client obtains only its authorized output. The toy independently
enumerates this scalar single-role view. Revealing total `C` as well can recover
`B` when `A` is invertible, and would make `E` disclose `X`. Giving Inference
Client's mask seed/share is likewise invalid. These arguments concern ideal
sharing; a real PCG needs a composed security proof after exposing `A`.

Each endpoint locally multiplies `D B_role`, dense work equal to **100% of the
original body MAC count again**, plus triple expansion, additions and storage.
Inference still performs all original `WE` body MACs: remote placement is 100%
for prefill and decode separately, so this does not dishonestly fail the 90%
*original-body* test by counting new crypto as old work. It instead fails to
supply a lightweight-client construction: **56,615,845,888 additional dense
client crypto MACs per pair**, and the same at Inference, before PCG expansion.
This arithmetic count is not a measured CPU-time ratio.

Moving `D B_c` to Preparation preserves the lightweight Client but requires
delivery of its matrix-dependent `H_c` (or `WB-s`) every use. This restores
output-sized correlation material rather than compressing it by independent
seeds. Supplying `(B,s,WB-s)` directly is the existing prepared-activation
contract, not a random-triple PCG consumer. Constructing compact correlated
keys for that *chosen* `W` function is the missing next gate.

A genuinely different topology has **two online providers** (Preparation and
Inference), sends two additive input shares from Client, exchanges two `E`
opening shares between providers, and returns two output shares. Its optimistic
all-link floor is `retained + D + 4*input_share + 2*output_share`, before new
controls, setup and verification. Client need not do dense adaptation, but
Preparation is now online and an activation-share computation party. This is
not silently treated as the preferred trusted-offline-Preparation topology.

## Freshness, static D, and failures

Fresh paper-style `A` means fresh dense `D` for **every independent triple**;
batching `t` independent columns amortizes one `D` only within that triple.
Every real `X` column needs independently fresh `B` (or `R` in the prepared
contract) and fresh output-sharing randomness. The exact repeat does not permit
correlation reuse. Reusing `B` exposes `E2-E1=X2-X1`, including across cloned,
cancelled or rolled-back attempts. Evidence contains a public scalar witness.

A separately specified ideal dealer can reuse `A` across fresh `B` and independent
`C` shares; the oracle tests three such uses and observes identical `D`. This
does **not** establish that the paper's PCG supports joint reusable-A security.
If it did, one-time two-endpoint `D` delivery would still be **2,773,155,840 field
bytes**, or optimistically **2,103,041,344 quotient-ring bytes**. Static `W`
broadcast does not remove `D`: random `A` changes it. Treating `A=W` is chosen
correlation generation, not random-triple adaptation. Evidence reports this
unsupported reusable-A sensitivity separately, not as achieved savings.

With hypothetical static `D`, field31 removes only **70,066,780 recurring bytes**
after larger input/output shares: **19.00%** of warm recurring traffic, below
even 20% before setup. Ideal exact-ring static `D` could remove the correction
category (139,060,689 bytes), leaving 37.72% warm savings before static delivery,
keys, CPU, memory, frames and renewal; 50% remains impossible. This is not an
admitted construction, and full client dense work remains.

The toy shared spent registry burns before validation, rejects replay/shallow
clone and cancellation, and burns malformed input. A deep-copy/rollback test
**demonstrates the missing durable ownership** by reproducing the input-difference
leak. Python is not non-cloneable storage or secure erasure. Production must bind
model/weight hash, field/ring, stage, roles, tile, column, epoch and use index;
durably reserve and burn material on disclosure, cancel, malformed delivery or
failed verification. Retries cannot reuse disclosed masks; unused padded columns
also retire with the use. Authentication, replay state, rollback protection,
malicious checks and failed/burned material delivery all have explicit unknown
costs. A correctness oracle is not malicious-result verification.

## Exact domains and output packing

The equations work exactly in `Z/2^k Z` for **k=16,24,32**. This is demonstrated
by the ideal dealer, not an implementation of a ring PCG. For the saved source
geometry the original rings are 24/32; public per-output residue widths vary.
Ring reduction `2^k -> 2^h`, `h<=k`, is a valid additive homomorphism. The ring
projection therefore grants **all existing output payload packing** and even
packs each `D` output row to its public residue width. This is an optimistic
matrix-opening lower bound with negligible padding, not an existing codec.

For a field, a bounded signed lift can be exact: `q=2^31-1` exceeds
`2*4864*128*128` for all saved input widths. After field-share reconstruction,
center the result, then reduce the signed integer modulo each original `2^k`.
This preserves exact integer results, but does not justify separately reducing
random field shares to a ring: odd-field wrap subtracts `q`, not `2^k`.
Field257 gives residue **133** instead of signed-integer/ring16 result **32,258**
for an explicit test. Source field-admissibility is unverified, so field31 is a
functional bounded-lift scenario, not a selected PCG parameter.

`Y_i` is a full-domain field share, not a bounded clear result; it needs 31 bits
per element in this layout. Existing 20–26-bit row-bound residue packing cannot
be credited without an additional secure conversion or proven cyclic-mask
consumer. Output bounds do not bound this random share/noise. Removing output
masks or sending a narrow function of them is not free: it needs a reviewed
privacy argument. Scales, biases, endpoint contract and float rounding are
unchanged; no decoder/float execution or quality claim is made.

## Complete source-shape ledger

All 96 stage identities, shapes and weight hashes cross-check
`research-all-link-frontier.json`, `research-narrow-ingress.json`, and
`research-bootstrap-ingress.json`, against the original
`network-algebra-qwen25-2026-10-06.json` geometry lock. Duplicate list stages and
duplicate JSON object keys reject before indexing; all four stage sets must
match exactly. Per-stage original rings, encoded input widths, row-residue
histogram counts/maxima, input/output payloads, executed rows and body-ledger
residuals must agree. Body, checkpoint, composition, source-lock, model and
revision identities cross-check without equating distinct workload plan digests.
The cold request executes 150 prefill rows
and seven decode operations for eight outputs. Its exact repeat executes no
prefill and another seven decode operations. Three terminal last-layer stages
execute only **one prefill row**. Thus 93 stages have 164 rows, three have 15,
total **15,297**, not 96 times 158 or two full prefills.

Control application bodies: first pair **493,575,247**, assumed common public
delivery **124,905,091**, recurring all-link **368,670,156**, corrections
**139,060,689**, original masked-input payload **90,104,112**, output payload
**138,864,059**. Recurring historical non-payload bodies excluding the entire
correction category are retained (**641,296 bytes**); new frames and delivery
work are additional unknowns, not zero. This assumes common public reuse on both
sides, not a demonstrated recurrence split of every public-bundle byte.

| Prefill tile | Field31 fresh D bytes | Optimistic row-packed ring fresh D bytes |
| ---: | ---: | ---: |
| 1 | 438,772,805,632 | 332,692,371,424 |
| 4 | 140,224,802,816 | 106,327,028,192 |
| 16 | 65,587,802,112 | 49,735,692,384 |
| 64 | 46,928,551,936 | 35,587,858,432 |
| 150 | 41,597,337,600 | 31,545,620,160 |

Every cell includes complete prefill remainder and **14 sequential decode
underfilled uses per stage**. No batching causally dependent decode tokens or
unrelated jobs is credited. For ordinary stages, tile 64 means 64+64+22, not
two tiles; tile 150 means one full prefill tile but each decode uses one active
column. Exact rectangular remainders are granted; fixed-width allocated slots
and corresponding extra work are also reported. Terminal prefill is one active
column. Tile150 ring recurring floor is **31,775,229,627 bytes**, field31 floor
**41,895,940,976**, before setup or additional helpers.

Only square triples are accessible-source-supported at abstract level. Evidence
also gives a **conditional** square-block padding map at sizes 1/4/16/64/**256**
(rounding requested150 up to a power of two). It counts all output/input spatial
blocks, full remainder/padded matrices, duplicated per-output-block input
openings, six generated share matrices across both endpoints, clear triple
`b^3` work, and each endpoint's `D B_role` work. This is not a claim that any
particular block size is source-admissible. The paper's expansion constant and
workspace remain unknown. The rectangular figures above are already favorable
lower bounds and reject; square padding cannot be promoted as a measured win.

Logical one-stage packed arrays project roughly 87–129 MB across these layouts;
they exclude PCG workspace, allocator/runtime, other model state and execution
scratch. They **do not admit or qualify a <=256 MiB whole-client process**.
Evidence reports both endpoint aggregate generated bytes separately from peak
storage and explicitly counts padding work. Neither resource estimate nor a
seed-only estimate qualifies Low.

For horizon `H`, security-reviewed epoch capacity `L`, bootstrap/renewal `B`:

```text
control   = P + H*R
candidate = Pnew + H*(R-S+helper+failure+frames+verify) + ceil(H/L)*B
headroom/pair at saving fraction p = S - p*(R+P/H)
```

Evidence derives exact rational **20/35/50%** headroom for 1/2/4/16/1024 pairs
and infinite warm reuse, for each complete cell and the hypothetical reusable-A
sensitivity. Every fresh-A cell has negative saving even at 0%, so all target
headrooms reject. Setup capacity, both role key generation/delivery, authenticated
helpers, checkpoint/provider distribution, retries, verification, renewal,
total-use security/failure probability and whole-wire bytes remain null.
Polylog seeds are not permission to reuse expanded masks or to assume unlimited
advantage budget. Both endpoints reuse the public artifact at the same horizon.

## Evidence and next gate

The standalone standard-library probe executes **3,024 independently checked
output values**: 3,000 across 48 fresh-A tiled cases in three rings and field31,
plus 24 across three separate ideal reusable-A cases. Each field output also
checks three final ring conversions, not counted as extra independent outputs.
Tests cover signed endpoints, full/remainder tiles, per-role adaptation,
single-role versus total-correlation leakage, seed/mask reuse, replay, clones,
cancellation, malformed burning, deliberate rollback limits, source geometry,
decode underfill and matched lifetime arithmetic. This is **test-only ideal
dealer functional/byte evidence**, not PCG throughput or cryptographic security.

Focused regression suite: **39 passed**. Existing environment's `ruff check`
reports **All checks passed** for the owned probe and test file.

Replay using existing environment, no installs/checkpoints/native imports:

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_matrix_triple_prefill_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_matrix_triple_prefill.py tests/test_matrix_triple_prefill_probe.py
target/frontier-sdk-env/bin/python scripts/probe_matrix_triple_prefill.py --output target/matrix-triple-prefill/replay.json
```

The output path must be fresh; overwrite is refused. Evidence:
`docs/evidence/research-matrix-triple-prefill.json`.

**Next gate:** obtain the full paper and numbered version; first inspect whether
A-only opening seeds can be disclosed without exposing B/C shares. Then specify a
reviewed *fixed-W* private consumer removing fresh dense `D` **and** client dense
adaptation. Show per-role shares/seeds, exact field/ring conversion, permitted
output packing, finite dimensions/security/renewal capacity, both endpoint CPU
and complete whole-client <=256 MiB admission, while retaining >=90% remote
original prefill and decode body MACs separately. Only then price complete
setup-inclusive matched all-link traffic. A new parameter certificate for
random triples alone cannot pass this consumer gate. This is distinct from
the generic streaming-PCF study, and does not assert universal impossibility.

Related boundaries: [bootstrap ingress](bootstrap-ingress-reference.md),
[bootstrap frontier](bootstrap-frontier-reference.md),
[correlation compiler](correlation-compiler-reference.md), and
[Maverick](maverick-reference.md) remain unchanged.
