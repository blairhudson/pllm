# Exact fixed-W correlation frontier: three construction gates

**Result: no Low-tier construction admitted.** A new sparse-noise masked-image
factorization is executable over the exact compiled rings, but requires a
specific, unproved multi-sample computational assumption. A transparent exact
key-homomorphic shortcut has an explicit cross-query attack. An exact
weight-circuit compiler is input-private with ordinary full-ring masks, but
requires an actual small circuit certificate for the checkpoint, not approximate
low rank. These are differentiated mechanisms, not three names for zero tapes.

The strongest conditional mathematical lead is **noise-repaired image
generation**: precompute `P=W*A`, then expand `r=A*u+e` and
`W*r=P*u+W*e`. The missing primitive is a securely parameterized, fixed-public-A,
multi-query sparse-noise ring generator whose image circuit is cheap enough.
No reviewed paper is claimed to supply that primitive. The present toy is
deliberately insecure and exposes why the missing security gate matters.

## Exact contract and boundaries

For every existing public signed-i8 matrix `W`, input `x`, compiled
`R=Z/(2^k)` (`k=24` or `32` in the locked geometry), require

```
r, image = Client.expand(index)       # image = W*r exactly in R
q = PRG(shared_q_root, binding, index)
s = image - q
Client -> Inference: z = x-r
Inference -> Client: y = W*z+q
Client: y+s = W*x in R
```

This is an outsourcing formulation, not automatically the original ideal
correlation/key distribution. Ideal independent `q` makes `s` uniform independent
of `r` algebraically; deriving `r` from a computational generator still needs a
complete-key security statement. The probe proves **only arithmetic identities
for supplied vectors**, not such a statement. Quotient input/output encodings
are the existing public, source-bound stage/residue projections. Signed decode,
bias, dequantization, rounding, float32 boundaries, nonlinearities, attention,
KV state, sampling and prompt remain unchanged. No full decoder executes here;
this is necessary linear-stage equivalence, not a new whole-model parity claim.

Preparation is offline, trusted and noncolluding with Inference. Client alone
owns the mask-generation root; both online parties own the independent q root.
Inference never receives `u`, `e`, input-mask roots or client expansion accesses.
Preparation knows both keys but receives no online openings. Collusion reveals
inputs. Every root/index must bind source/weight digest, dimensions, ring,
numeric layout, stage, roles, capacity and epoch. Burn indices before exposure,
including cancellation, malformed replies, expiry and failed attempts. Restart,
clone and rollback require an external durable authority; none is implemented.
Fresh q or a new label cannot repair reuse of r. New epochs renew private roots;
public source-specific artifacts may be reused only unchanged and authenticated.
All candidates here are unauthenticated arithmetic: malicious replies can add
undetected errors. Transport authentication is not a proof of W*x.

## H1: sparse-noise computational masks with precompiled images

**Construction.** Public `A in R^(n*d)`, sampled independently of W, and
`P=W*A in R^(m*d)`. Client gets authenticated W/P/A artifacts and two private
roots: one for fresh `(u_j,e_j)`, one shared q root. Inference gets only q root
plus public binding/A identifier. Each `u_j in R^d`; `e_j in R^n` has t distinct
secret coordinates with independently sampled ring values. Client computes
`r=A*u+e`, `image=P*u+sum_i e_i W[:,i]`; Inference expands q.
Preparation performs `mn*d` ring MACs once per W/A artifact. This supports
arbitrary chosen fixed W algebraically, including singular and rectangular W.
No inverse encoding of W into a structured random algebra is assumed.
The ledger delivers a 32-byte public XOF seed for A, not a uniform matrix tape:
the needed assumption must cover that seeded distribution specifically. A
uniform-A theorem alone does not certify public-seed substitution.

The executable takes supplied u/e, so it does **not** implement a secure sampler,
seed/key generation, PRG, party handles or durable renewal. For repeated public A,
the needed assumption is that polynomially many independent `A*u_j+e_j` are
indistinguishable from uniform **full ring vectors**, conditional on all public
artifacts and the independent full q key. Because P is a deterministic function
of public W/A, it is not an additional secret leakage channel under that
assumption. A chosen-input hybrid would then replace masks successively and
simulate uniform openings, including adaptive queries. This is a conditional
reduction, not a proof that the proposed noise distribution satisfies it.

This is not ordinary binary LPN asserted to work modulo 2^k. Reduction modulo 2
must already resist attacks; higher-bit conditioning, sparse support recovery,
multiple samples, all supported stages and total query count need their own
analysis. No security level follows from `d=128,t=27`; those are arbitrary cost
sensitivity coordinates, **not imported paper parameters or a 128-bit claim**.
Public low-rank r without noise is the previously rejected shortcut, not a
discovery. The noise is precisely the unproved computational repair.

**Falsification.** For `R=Z/4`, `A=(1,1,1)^T`, one noise position and uniform
ring u/error, chosen inputs `(0,0,0)` and `(0,1,2)` have exact opening-view
TV **1/2**. At x=0, at least two opening coordinates always agree. The counterexample
is reproduced by exhaustive enumeration, independent of any PRG. It rejects
these toy parameters, not computational expansion in general. Zero/even noise
also leaves public mod-2 relations; dense independent noise recovers uniformity
but restores dense W*e work. One cannot quietly substitute either endpoint.

**Costs/gates.** Client adds `d(n+m)+mt` ring MACs per row, generates public A
entries, streams `k*m*d/8` P bytes and reads `mt` signed-i8 W-column bytes.
Sparse access indices are secret: local trusted storage is required; an
Inference-observable object store requires unpriced oblivious access. W+P
delivery is `mn+kmd/8`, before bindings. Setup stores/writes P and scans W once
per d-column in the specified unoptimized schedule. Neither small roots nor
disk streaming removes these costs.

For horizon R, capacity Q, let `E_i=ceil((R*h_i+burn_i)/Q)` and `M_i` be packed
output bytes per stage-row. A necessary known-payload break-even condition is

```
sum_i [mn + kmd/8 + 64 + 32*E_i] < sum_i E_i*Q*M_i
```

The 64 includes delivery of public A's seed to both roles. All additional
unknown costs must fit the remaining margin. **Kill:** any efficient
fixed-A distinguisher at proposed parameters, insufficient composed security,
secret observable column accesses, or security-required d/t violating the
all-link/client-work budget. **Next gate:** specify a concrete multi-query
ring sampler/assumption, attack its parity and higher-bit syndromes, and derive
parameters before any timing, native code or serving cohort. Existing reviewed
QA-SD scalar OLE parameters do not discharge this gate.

## H2: exact key-homomorphic expansion and its transparent-module obstruction

Suppose a public evaluator `F(k,j)` were exactly R-linear in a compact key
`k in R^lambda`. Preparation samples coordinate keys `a_i,b_l`, sends these
to Client, and sends `c_l=sum_i W_li*a_i-b_l` to Inference. Client evaluates
`r_i=F(a_i,j), s_l=F(b_l,j)`; Inference evaluates `q_l=F(c_l,j)`.
The relation is exact for arbitrary W, and work no longer explicitly multiplies
W by each mask. This is a genuinely different correlated-key compiler.

**Attack under stated representation.** Because basis keys are publicly valid,
`F(k,j)=sum_h k_h*F(e_h,j)`. With a transparent free module, the public sequence
of evaluator rows has mod-2 rank at most lambda. More than lambda output indices
therefore have a public parity relation. Full-ring PRF security is already lost
modulo 2. This does not assume Gaussian elimination over a field inside the
2-adic ring. A stronger explicit full-ring witness for lambda=2 is
`F((a,b),j)=a+j*b`: at j=0,1,2,

```
z_0-2*z_1+z_2 = x_0-2*x_1+x_2 mod 2^k.
```

Tests exhaust every mod-8 key. All labels are fresh; no index reuse is needed.
The result is scoped to public evaluation, valid basis keys and exact linearity
on the entire free module. It is **not** a no-go theorem for hidden/constrained
key groups, nontransparent encodings, FSS, PCFs or computational PCGs generally.
Rounded/noisy key-homomorphic PRFs escape exact linearity only by introducing
an error; exact correlations then need another, priced correlated error-repair
primitive. Rounding that error away changes the model and is disallowed.

**Ledger/gate.** Transparent full-entropy fallback `lambda=Q, F(k,j)=k[j]`
is secure with independent ideal key words but is just tapes. Per stage/epoch,
Client delivery is `64+kQ(n+m)/8`, Inference delivery `64+kQm/8`;
setup performs `Qmn` MACs, Client reads `k(n+m)/8` bytes per row, worker reads
`km/8`. Online links are unchanged. Inference's ring tape is no smaller than
the baseline quotient tape, and Client tape is additional: **no break-even**
against the seeded prepared baseline. Both numerical workloads are in JSON.
**Kill:** a public cross-index relation, or lambda growing to Q with this
representation. **Next gate:** an explicit nontransparent correlated-key
functionality or exact noise-repair construction and full-key proof, not a
generic appeal to key homomorphism. Setup/delivery/expansion of such a missing
primitive remains null.

## H3: exact weight circuits, not low-dimensional input masks

Compile a certificate `W=U*V+E mod 2^k`, with inner dimension h and a sparse
exact residual E. Fresh r is an ordinary full-ring mask; Client computes
`image=U*(V*r)+E*r`, then subtracts shared q. Thus there is no mask-subspace
privacy loss. The executable includes a **full-rank** 3-by-3 W represented by
a rank-one matrix plus identity and checks all 64 masks with coordinates 0..3.
It also compiles an arbitrary unstructured W to an exact dense residual.
Dropping just one residual entry demonstrably changes output. Approximate
factorizations, learned low rank and quality proxies are not candidates.

The certificate supports arbitrary W through E=W as the universal fallback,
but that fallback is the already-known dense-client tradeoff, not an advance.
For a useful source-specific circuit the necessary arithmetic-count threshold
is `h(n+m)+nnz(E)<mn`; operations are not timing ratios. With full-ring factors,
public delivery is `kh(m+n)/8` plus exact sparse-residual values, indices and
metadata. Factors alone at h=128 across the locked stages cost **204,079,104
bytes**. The residual's nnz/representation is unknown, so delivery, complete
all-link total, streamed reads and client arithmetic remain null in JSON.
For each row, read all factor coefficients and residual records unless cached;
source parsing/certificate generation costs and paged writes also remain null.
Client roots are 128 bytes/epoch including binding, Inference 96; there is no
new public A seed. Setup delivers the certificate, not a fresh Q-row output tape.

**Break-even:** sum certificate/metadata bytes plus extra renewal/framing must
be below the removed correction material at the matched horizon, and the
above work threshold must hold independently. **Kill:** dense residual, failed
exact coefficient verification, or certificate/work exceeding the matched
budget. **Next gate:** parent-owned coefficient-only circuit certificate with
independent exact residual verification for one source-bound stage. No checkpoint
tensors were loaded here. Existing distinct-row results and high displacement
minors reject their particular dictionary/shift schemes; they do not establish
a lower bound on all exact circuits or on this factorization's residual.

## Complete conditional ledger: 150+8 and 150+64

The probe joins five SHA-256-locked metadata/evidence sources. Geometry has 96
stages, **357,826,560 signed-i8 W bytes**. Independent terminal-prefill metadata
identifies three one-row stages; all others use 150 prefill rows. Decode uses
outputs-1 rows per stage. Demand **14,625 / 20,001 stage rows** agrees with every
maintained replicate of the respective existing confirmation cohorts. The
cohorts' source/body identities agree; their measured bytes/RSS are **not**
substituted into a different analytical protocol or pooled across workloads.

Baseline here is explicitly specified: 64-byte binding plus two 32-byte client
roots per epoch; 64-byte worker binding plus Q packed correction rows. H1 sends
128-byte client and 96-byte worker key/binding records per epoch. Each online
direction pays digest32+index8 plus the locked stage-packed input or quotient
output. Per-row output packing rounds each row independently. Existing runtime
MessagePack/TLS/control records are not claimed equal to these modeled records.

At arbitrary **non-security-calibrated** d=128,t=27,Q=256:

| Known application payload / work | 150+8 | 150+64 |
| --- | ---: | ---: |
| Baseline Preparation→Client | 12,288 B | 12,288 B |
| Baseline Preparation→Inference | 224,525,824 B | 224,525,824 B |
| H1 Preparation→Client, including new public artifacts | 477,379,584 B | 477,379,584 B |
| H1 Preparation→Inference | 12,288 B | 12,288 B |
| Client→Inference, either | 86,709,192 B | 118,763,592 B |
| Inference→Client, either | 133,315,967 B | 182,644,687 B |
| Baseline known all-link total | 444,563,271 B | 525,946,391 B |
| H1 known all-link total | 697,417,031 B | 778,800,151 B |
| H1 additional client ring MACs | 10,650,317,056 | 14,589,333,760 |
| H1 logical P reads | 18,091,196,416 B | 24,785,305,600 B |
| H1 logical secret-index W reads | 1,242,853,632 B | 1,702,695,168 B |
| H1 public A expansion | 12,799,410,176 B | 17,533,730,816 B |

All 24,576 issued stage-columns are charged, including unused capacity. H1
setup also costs **45,801,799,680 ring MACs** and that many logical paged-i8
W reads under the specified d-scans schedule. W/P client disk writes cost
477,364,224 bytes. Worker online W reads/MACs and q PRG bytes, baseline tape
writes/reads, both endpoint mask expansion, index minimum bytes, and ordinary
client masking/reconstruction are separate JSON fields. Sparse support sampling
rejection work/bytes are unknown, not free. No wall-time or device-I/O inference
follows from logical counts.

Q=256 intentionally overprepares short requests. With unchanged public artifacts
reused and roots renewed at capacity, the first **known-payload** break-even
is request 4 / 3; first known 20% margin is request 5 / 5. These are optimistic
arithmetic sensitivities, **not savings or cryptographic admission**. JSON also
includes request-sized per-stage capacities, charging exactly the needed columns
per request, to prevent padding removal masquerading as a generator win. Renewals
are recomputed at every horizon; unused capacity is never carried across a
different root epoch. The original 26–32% cold/~38% warm correction-only bounds
remain separate matched-cohort facts. No candidate here removes online crossings;
50% does not follow from compact correlations.

Every directed known total excludes failed online messages. Burns add whole
setup epochs; failed openings/replies, durable authority RPCs, authentication,
oblivious storage, malicious verification, common checkpoint/original public
distribution, physical wire and setup/expansion time are explicit nulls. There
are no extra helper workers in these specified protocols; adding one requires
pricing its links and changing the non-collusion graph. Preparation→Inference
setup work and Inference→Client replies are not omitted as “server-side”.

The 125,329-byte H1 vector workspace is representation payload only. All public
matrices must be streamed for Low: W+P alone exceeds 256 MiB. Public disk
capacity, stream buffers, SDK, native/allocator state, transport, active-stage
overlap, renewal overlap and durable authority still contribute. **Complete
client including SDK must stay below 268,435,456 bytes**; no RSS or security
claim is made, and no candidate is Low-qualified. H2/H3 memory/unknown ledgers
likewise retain null complete-client peaks.

## Replay and evidence

```sh
python3 -m unittest discover -s tests -p test_exact_correlation_frontier.py -v
python3 scripts/probe_exact_correlation_frontier.py --output target/exact-correlation-frontier-replay.json
```

Output must be new. Frozen record:
`docs/evidence/exact-correlation-frontier-bounded-v1.json`. It includes source,
probe and test hashes, exact outputs, the TV counterexample, key-linear witness,
directed ledgers, horizons and explicit nulls. Tests challenge arbitrary
tall/wide/scalar signed matrices, full-ring/wrap inputs, wrong images/rings,
exact/dropped residuals, key-linear privacy, sparse support privacy, reuse,
hash drift, directed sums, renewal/burn accounting, padding and tape costs.
Only tiny stdlib arithmetic runs. No installation, upstream method import,
checkpoint tensors, crypto timing, serving cohort, build or commit occurs.
