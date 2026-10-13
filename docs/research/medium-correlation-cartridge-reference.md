# Medium fixed-weight correlation cartridges: construction gate

**Conclusion:** no reviewed construction supplies cheap asymmetric client expansion
for arbitrary signed-i8 checkpoint weights. Medium's 2 GiB whole-client limit does
not fix that cryptographic gap. A fully specified locally generated cartridge is
possible, but pays one additional dense client `W*r` per row. A bounded-column
random-matrix adapter is honest and improves the old fresh-per-decode opening
schedule, but still loses the hypothetical 20% all-link target on this saved
geometry. Nothing here qualifies Medium or measures candidate performance.

## Source-pinned reading and the exact missing ingredient

Page references are **1-based PDF pages**. PDF hashes come from
`docs/evidence/compact-setup-fulltext-review.json` and are checked on every replay.
The new evidence also hashes the actual text extractions and historical records.

* **2025/1223, Construction 1, pp. 24–25:** public random algebra element `a`;
  each party's key contains its sparse positions/coefficient vectors `(s,e)` and
  four SPFSS keys sharing `s0*s1`, `e0*s1`, `s0*e1`, `e0*e1`. Expansion computes
  `x_theta=phi(a*s_theta+e_theta)` and
  `z_theta=phi(a^2*u0+a*u1+a*u2+u3)` after four FullEval calls. It yields
  coordinatewise **random OLE**, not arbitrary matrix multiplication by W.
  QA-SD and secure SPFSS are assumptions; regular-noise optimization needs its
  additional assumption (p. 26). Generic `GR(2^k,1)` is not obtained by setting
  extension degree to one: §5.2 uses traces from an extension (pp. 26–29).
* **Theorem 5.6, pp. 29–30; Appendix A.1, pp. 41–42:** Gen accepts random
  `rho_theta=(positions_s,positions_e,coefficients_s,coefficients_e)`; public psi
  outputs the corresponding trace vector of `a*s+e`. Programmability preserves
  those random seed inputs. It neither proves psi is onto arbitrary checkpoint
  W nor gives an efficient inverse mapping W to a valid random rho. Its security
  quantifies random seeds, not adversarially chosen W hidden inside them.
  Choosing public a=W is also invalid: a is a random element of a structured
  commutative algebra, not a general rectangular matrix, and QA-SD distribution
  changes. Componentwise OLE's bilinear map cannot be relabeled `W*r`.
* **Circuit-dependent claim, p. 7:** seed size `O(|C| log N)` for N copies of
  circuit C; matrix extension claim follows on the same page. This is a real
  lead, not an explicit client-light fixed-W Gen/Expand. Compiling every MAC into
  a circuit charges its circuit size; the supplied claim does not eliminate the
  client-side local correlation/circuit evaluation. No special fixed-W proof or
  concrete table/expansion cost for the proposed functionality appears here.
* **Parameter separation, pp. 33–36:** 109k authenticated triples/s uses
  `c=5,t=27`, 128-bit target, `(k,s)=(32,26)`, `3^15`; the roughly 115 MB/billion
  point uses `c=9,t=9,m=2,n=15`, 80-bit target. Broken `c=3,t=27` excluded.
  512 GB benchmark machine capacity is not measured working-set size. None of
  these scalar-triple results measures fixed-W client expansion.
* **2026/2298, pp. 70–72:** rectangles use padded square products. Section 7.2
  permits at most n sequential fresh right-hand columns **within one triple**.
  Pages 74–75 explicitly do not prove operand-mask reuse across block products
  or instances; p. 42's public multiplier batch reuse is not activation reuse.
  Page 72 leaves exact `Z/2^64` extension to future work. Table 4, p. 68, natural
  128-square twisted field implementation needs estimated 2–32 GiB per party.
  Its 2 GiB workspace alone leaves no room under the whole-client Medium maximum.
  Table 7, p. 70, distributed setup is separate from amortized seeds.

**Concrete missing lemma, not merely “needs review”:** a fixed-linear-map masked
image compiler over `Z/2^k`, accepting arbitrary public W and Q, must provide
`Gen(W,k,layout,Q)->(T,K_C,K_I)` with joint role-view simulation of independent
fresh `(r_j,s_j)` and `q_j=W*r_j-s_j`, for all distinct `j<Q`. Public table T must
be bounded by `poly(m,n,lambda,log Q)`, not an output-sized Q-row tape. Client
expansion should use vector-linear/quasilinear work in m+n, no dense W multiplication
and no dense `D*B_C`; inference may do dense work. The security statement must
include public T, both expansion-key views separately, and every opening across
all Q uses. Supply a transformation of the **linear image of QA-SD sparse noise**
into role-separated compact keys, with a simulation proof and operation counts.
Construction 1's sparse cross-product FSS does not itself implement this linear
image compiler. Even precomputing `W*H` for a public noise-expansion map H must
charge the dense table, distribution and role-local application to private sparse
seeds/FSS shares; it does not prove either cheap expansion or privacy.

The direct/circuit-dependent candidate therefore has **no supplied keys, Gen,
Expand_C or Expand_I** meeting this lemma. These are explicitly absent rather
than invented from an ideal scalar OLE or the algebra controls below.

## Ranked constructions and exact role knowledge

### 1. Locally generated W-only public cartridge: concrete, but client-heavy

This is the most feasible **actual** construction for exact functionality, not
the desired cheap-client result. Offline trusted, noncolluding Preparation binds
signed-i8 W digest, original ring k, per-output widths, stage, roles, epoch, Q and
version in an authenticated context. Gen samples independent 256-bit roots R and
U. Client key is `(R,U,context)`; Inference key is `(U,context)`; public table is W
and its manifest. Preparation knows both roots; Client knows r,s,q; Inference
knows W,q,never R/r. U can safely be public; R cannot.

Exact specified expansion: SHAKE256 input is a length-prefixed domain string,
root, canonical context and little-endian u64 j (require `0<=j<Q<=2^64`). Separate
domain strings `medium/r/v1` and `medium/q/v1`. Read exactly `ceil(n*k/8)` and
`ceil(m*k/8)` bytes respectively; unpack least-significant-first k-bit lanes,
discard unused final bits. This samples full-ring lanes under the XOF-PRG
assumption; no rejection sampler or odd-field reduction is needed.

```
Expand_C(j): r=PRG(R,r-domain,j); q=PRG(U,q-domain,j)
             s_l=(sum_i W[l,i]*r_i - q_l) mod 2^h_l
Expand_I(j): q=PRG(U,q-domain,j); reduce q_l mod 2^h_l
online:      z=x-r; reply_l=(W*z+q)_l mod 2^h_l
client:      output_l=(reply_l+s_l) mod 2^h_l
```

This specification is **not implemented as cryptography in the probe**. Security
requires a multi-output PRG assumption for secret R (a seed known to Inference
would trivially break privacy); q is independent of r, and knowing q is harmless.
Given independent full-domain r and q, `(q,x-r)` is input-independent. A PRG hybrid
extends this to finite polynomial output length, not an unbounded lifetime or
concrete 128-bit certificate. At Q exhaustion use fresh R,U and a new bound
context; composing E epochs requires a reviewed aggregate advantage bound.
Default Q=65,536 in cost controls is a hypothesis, not a borrowed security policy.
Authentication/malicious response checking remain separate, unimplemented costs.

Public W may be delivered to Client once (357,826,560 bytes across 96 matrices)
or recreated from an already locally available authenticated checkpoint. Local
creation avoids this *new* transfer only if the checkpoint's earlier transfer is
already paid in the matched control. Repacking, reading and program memory cost
remain. There is no giant public `W*H` table. Each use pays n+m PRG lanes, one
client mn MAC product, n input subtractions and m output operations; Inference
still executes all original body MACs. Client does not get to call W*r “cheap
expansion.” Per historical pair this is **56,615,845,888 extra client MACs**.
Prefill and decode counts are separately reported; original remote fractions are
1.0 for each, with this additional work explicitly charged.

### 2. Honest bounded-column random-matrix adapter: construction conditional

Ideal-dealer equations below exactly specify the adaptation; they do not claim
2026/2298 implements ring triples. Gen for one batch samples A and B with b fresh
columns, computes AB, and independently additively shares A,B,C=AB. Role key is
that role's triple shares, or a **proven** PCG seed for those shares. Open
`D=W-A` in two directions (or two dealer deliveries). Public table is W and D;
both roles learn A from D, neither learns the other's B,C shares. At column j:

```
Expand_C: r=B_C[:,j]; s=C_C[:,j]+D*B_C[:,j]
Expand_I: q=C_I[:,j]-A*B_I[:,j]
```

Since `C_C+C_I=A*(B_C+B_I)`, `s+q=W*B_C`, giving **the requested cartridge
interface**, not a renamed whole-B mask requiring an extra online opening.
Online Client sends `x-B_C`; Inference replies `W*(x-B_C)+q`; Client adds s.
Fresh independent additive C shares in the ideal dealer give the required role
privacy. A real PCG requires composed security after A is exposed; revealing a
full other-party key may expose B_C and is forbidden. No safe A-only key interface
is assumed. Both roles still do an extra dense product per consumed column.

Columns consumed in order can span prefill, decode and request-pair boundaries.
An ordinary saved stage uses 164 columns per pair, terminal stages 15. At b=128,
one pair needs two and one triples respectively, **not 15 per stage**. Unconsumed
columns can remain private for later uses within that triple; on retirement they
are charged as unused. After b columns, new A/B/C and new D required. Gen's dense
shares are explicitly output-sized; using short PCG keys requires the paper's
assumptions/field/finite-batch proof, not the toy dealer.

The favorable exact-ring rectangular ledger packs D output rows to saved residue
widths. It is an **algebra-only lower-cost control**, not a reviewed ring PCG.
Published field64 square controls use 128-square blocks, independent block masks,
block-specific ingress and replies, both-role six generated share matrices,
padded CPU and finite PCG-batch retirement. No cross-block B/A reuse is credited.
Field64 shares cannot be truncated to quotient-ring widths: a proven conversion
is additional unknown. A bounded signed integer result can be centered only
**after field share reconstruction**, then mapped to the compiled ring.
256-square field cells are dimension sensitivities, not Table 4 parameter points.
Their seed/setup/batch-capacity/workspace estimates are null; only exact
algebra/opening/body geometry and the negative traffic gate are retained.

### 3. Locally regenerated large public tables: tradeoff, not missing PCG

The existing bootstrap cartridge locally reconstructs public `P=W*G` columns;
its checkpoint DAG is reusable, not reused secret masks. It uses BabyBear/sparse
noise with unresolved security/domain conditions, so it cannot be relabeled an
exact-ring proof. Rebuilding P locally removes a transfer but charges `16mn`
encoding lanes per complete stage batch plus fresh support applications. Choosing
W-only and computing each W*r reduces storage to signed W, but is candidate 1's
dense work again. An exact-ring generalization needs its own noise/security proof;
neither scalar OLE nor Definition A.1 supplies it. Local public preprocessing
amortizes across masks, whereas fresh private expansion and correction work repeat.

## Matched ledger, renewal and memory gates

Historical application bodies, not physical wire or latest qualified control:
`C=124905091`, `T=368670156`, removable correction bodies `S=139060689`.
The optimistic correction-only upper allowance is:

```
control(H)=C+H*T
candidate(H)=C+H*(T-S+extraIO+helpers)+newPublic+allIssuances
20% requires 5*candidate(H)<=4*control(H)
```

Every issuance includes first setup and both role deliveries. W-only root
delivery is 64 Client + 32 Inference bytes per stage epoch, plus a hypothetical
96-byte stage-epoch manifest delivered to **each** role (192 combined), charged
again on renewal. Stage-specific renewal is
`ceil(H*rows/Q)`; failures,
cancellations and expired reservations increase rows before rounding. The function
also accepts an explicit burned-attempt count per stage; its test crosses a renewal
boundary. Default reports use zero-failure conditional ledgers, **not** zero
measured failure cost; failed online bodies and expansion CPU remain unknown.
Bounded triples charge two D deliveries every `ceil(H*rows/b)` batches, unused
columns and both-party generated share bytes. For **128-square cells only**,
Table 4's rounded 5453 bytes per
party/triple at N=4096 yields a separately labeled dealer seed-delivery estimate
`2*ceil(triples/4096)*4096*5453`; Table 7's combined 200.3+49.5 MiB estimates
distributed setup instead. These alternative setup modes are not added together
or confused with exact serialized sizes. All omitted checks/helpers remain null.

With conditional saved ingress/output quotient codecs, W-only first reaches 20%
at six pairs before all unknown costs; roots are small, extra W delivery is not.
Full compiled-ring ingress and replies add **23,525,269 bytes/pair**, moving the
first conditional 20% horizon to ten pairs, not making it impossible. These exact
stage-ring totals supersede the older blanket 31/32-bit width sensitivity for
this candidate. All tested bounded-column adapters reject 20% before PCG setup, even
though D is amortized rather than repeated for every decode. Longer horizon does
not make fresh D free: it repeats at column exhaustion. Correction-only ceilings
remain historically ~26–32% first-pair / ~38% recurring, not seed-compression
ratios. A hypothetical **20% saving target** can survive for the dense
local candidate only conditionally; no positive serving saving is established.

Program memory: roots/manifests and signed W. Persistent storage: W or an
authenticated source checkpoint; no assumption that disk storage equals resident
RAM. CPU: count W*r or D*B_C separately from original remote MACs and PRG/FSS
work. Resident memory: vector buffers, layout/code tables, maximum live overlap,
SDK/model/KV/runtime, scratch, page cache effects and renewal overlap all matter.
512 MiB–1 GiB proposed workspace is a hypothesis; it leaves only 1–1.5 GiB for
everything else under Medium's **2 GiB whole-client maximum**. Published natural
2–32 GiB per-party matrix workspace threatens or exceeds Medium already. No
primitive estimate, packed table size or our Python probe RSS qualifies it.

Indices must reserve durably before exposure; cancel/malformed/timeout/failure
burn them. Restart, restore and clone share monotonic authority; epoch strings
alone do not stop rollback. Distinct stage/weight/layout/role bindings are mandatory.
Preparation must not collude with online Inference. No ownership/erasure proof,
malicious verification, complete decoder or float-boundary test is claimed.

## Evidence, focused replay and smallest next experiment

Owned probe performs 84 independently signed-oracle-checked bounded-column
outputs and 84 direct per-output quotient checks in ring3/16/24/32, plus exhaustive
512 scalar inference-view cases. Tests cover actual equations, capacity,
renewal/delivery/unused counts, strict dimensions/rings/residues, duplicate JSON
and stages, source-lock rejection, full-width gate reversal and block-specific
matrix accounting. Source geometry's four records are hash-checked against the
frozen matrix-triple record **before** running its validator; checkpoint, weight,
ring, row, width, body and composition locks then cross-check. No checkpoint runs.

```
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_medium_correlation_cartridge_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_medium_correlation_cartridge.py tests/test_medium_correlation_cartridge_probe.py
python3 scripts/probe_medium_correlation_cartridge.py --output target/medium-correlation-replay.json
```

Output must be NEW; overwrite refused. Frozen evidence is
`docs/evidence/research-medium-correlation-cartridge.json`. Unknown candidate
CPU/RSS/security/physical-wire fields are null, not inferred from algebra success.
Focused owned suite: **44 tests passed**, Ruff **All checks passed**. Replay
verifies all source hashes; its deterministic JSON must equal the frozen record.

**Next smallest meaningful experiment:** obtain or derive the masked-linear-map
compiler lemma for one arbitrary 3x5 signed W over `Z/2^24`, with explicit public
table and both keys, two-index full-role-view simulation, finite Q, and client
operation counts. Reject immediately if it requires mn client work or Q output
tape. Only then authorize a native one-stage, fresh-process whole-client peak
and both-role expansion measurement. The dense W-only construction already has
an algebra control; benchmarking it cannot repair the missing cheap-client lemma.
