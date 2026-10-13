# Persistent setup and indexed streaming matrix correlations

**Outcome:** the exact-ring consumer is simple and survives bounded independent
checks. No verified primary source here supplies a compact, client-light generator
for arbitrary checkpoint `W` with complete concrete parameters. Streaming COT is
not that generator. A fully specified ideal chosen-vector-OT adapter is rejected
by finite bytes and expansion work. No production integration or security claim.

## Primary provenance: abstract-only, not paper-parameter verification

The following primary ePrint landing-page abstracts were fetched and read.
Direct PDFs, download-query variants and the ePrint Asia mirror returned HTTP
403. The Dory author publication page independently links the same ePrint:
<https://wangxiao1254.github.io/publications>. Searches did not obtain alternate
full author PDFs. Thus sections, algorithms, implementation parameters and tables
remain **unverified**; abstract claims must not become measured PLLM results.
JSON locks selected normalized abstract excerpts by SHA-256, explicitly not PDF
or complete-page hashes. PDF SHA-256 is null. No source code imported, built,
linked or vendored; no external crypto implementation executed.

Bounded follow-up checked Boyle/Nof author publication links, Dutta and Scholl
publication lists, Kohl's author/CWI pages and CWI repository, Roy's author page,
and Dory arXiv search. Boyle/Nof and Dutta/Scholl link only the same ePrint copies;
other checked pages expose no alternate copy. Still abstract-only; no fabricated
full-paper parameter verification.

| Primary source | Actually verified claim | Does not establish |
| --- | --- | --- |
| [Preprocessing for Life, 2025/787](https://eprint.iacr.org/2025/787) | Feasible one-time setup, 2+1 structure with offline/sublinear party, trusted or untrusted dealer; FLIOP plus PCGs; storage/dealer communication improved by **more than three orders** against dealer-aided SPDZ for **Boolean circuits**. | 1000x all-link PLLM saving, exact ring matrix keys, negligible client circuit work, or unlimited concrete security lifetime. Online computation and peer openings remain. |
| [Dory, 2025/1660](https://eprint.iacr.org/2025/1660) | Malicious streaming subfield VOLE PCG, bounded output count; related PCF extension; implemented COT special case generates 10M COTs with 20 MB versus 337/624 MB. Attack gives old regular-noise expand-accumulate parameters only 14–22 bits; updated low-weight-noise parameters claimed 128-bit. | Whole-client RSS <=256 MiB, arbitrary chosen output-vector differences, exact `Z/2^k` matrix PCF, or parameter compatibility with an unrelated ring generator. |
| [DCR PCFs, 2026/2030](https://eprint.iacr.org/2026/2030) | COT and VOLE over `Z_M` for arbitrary modulus, solely DCR, two-round distributed setup; OLE and general degree-two correlations over **F2**. | Arbitrary-ring OLE/general quadratic correlations: F2 qualifier matters. Fixed public matrix specialization with independent coordinate masks, exact key sizes, client expansion costs and security total-use budget remain missing. |

The existing [ring-PCG/PCF review](bootstrap-ingress-reference.md) covers
2025/1223 and 2025/1637. Its R15 gate still requires Galois extension
`GR(2^k,m), m>=2`, QA-SD/Ring-LPN, SPFSS and distributed keys. Practical seeds
are megabits, not assumed 256-bit roots; old `c=3,t=27,m=2` set is broken.
No assertion that these generic constructions implement this relation or
client-light consumer. Ring-LWR/BFV arithmetic PCFs also carry FHE expansion
cost; neither ring conversion nor bootstrapping is free.

## Complete ideal matrix-PCF consumer (not implemented cryptography)

For public fixed `W in (Z/2^k)^(m*n)`, define correlation functionality at unique
index `j`:

```
r_j <- uniform (Z/2^k)^n; s_j <- independent uniform (Z/2^k)^m
Client gets (r_j,s_j); Inference gets q_j = W*r_j-s_j.
Client sends z_j = x_j-r_j.
Inference sends y_j = W*z_j+q_j.
Client reconstructs y_j+s_j = W*x_j, modulo 2^k.
```

This specifies all data openings: `z` and masked reply `y`; no `r`, `s`, seed or
private index choice is opened. Checkpoint/ring/output decoding/scales/rounding
stay original. A computational implementation must emulate **joint** multi-index
distribution against each role, not merely make individual outputs look random.
Independent full-ring `s` makes `q` uniform even conditioned on `r`, so provider's
`(q,z)` distribution is input independent. This is an ideal algebra/privacy
argument, not malicious-result verification; verification correlations, checks,
openings and malformed-reply policies must be added and priced.

Ideal setup would accept `W,k,model digest,roles,epoch,Q`, then issue separate
`K_C,K_I` enabling local indexed outputs above. Trusted offline noncolluding
Preparation may know both keys/roots, preserving baseline trust. Inference never
gets Client-readable root or a key permitting `r` reconstruction. There is one
online Inference and an offline Preparation; a construction requiring extra live
helper is a changed topology and its complete edges must enter the ledger.

Keys must bind checkpoint/weights, ring, stage, role, epoch, session and output
layout. Index allocation binds operation/row; independent domains alone do not
prevent rollback. Allocation must be authenticated, globally monotonic and
durable **before exposure**. Clone, restart or restore must consult same authority;
otherwise deterministic expansion repeats masks. Burn index on cancellation,
expiry, malformed input/reply, timeout and failure; never reclaim unused ticket
after possible exposure. Expiry alone cannot prevent cloning. Resetting local
state reproduces accepted index in the test counterexample. `TicketBook` models
an ideal shared authority only: Python objects are clonable, not production
ownership or secure erasure. Cryptographic capacity counts total generated/
exposed correlations across roles, sessions, failures and renewals, not successful
pairs alone. If one-index distinguishing/failure bound is epsilon, total-Q bound
needs composition (e.g. Q*epsilon where applicable), not amortization preference.

Client matrix work under this **missing primitive** is zero `W*r` MACs; vector
expansion/storage is `k*(n+m)/8` bytes per row plus key/scratch/state. Provider
stores `k*m/8` generated correction bytes per row and still computes `m*n` original
MACs. Streaming can bound resident output buffers, not eliminate total expanded
output, PRF/crypto work, checkpoint ownership or queue depth. Whole-client RSS,
client expansion CPU, setup key sizes/work and malicious checks remain unknown.
Original body MAC placement can stay 100% remote prefill/decode separately, but
extra crypto work must be reported independently; no 256 MiB admission follows
from buffer formulas.

## Implementable baseline versus explicit ideal adapter

**Baseline persistent roots:** Preparation and Client share two independent
secret roots for domain-separated indexed expansion of `r` and `s`. Preparation
computes dense `W*r`, sends `q` to Inference every time. SHAKE roots compress mask
delivery, not matrix corrections. No provider-readable mask root. This preserves
trust and permits distinct indices; saves no correction traffic. Existing
[subset compiler](correlation-compiler-reference.md) attempts faster Preparation
evaluation; its measured negative gate remains unchanged. Client receiving `W`
and evaluating `W*r` could remove that edge only by moving matrix work client-side.
It costs exactly `m*n` extra dense MACs per row, not cheap PRF expansion.

**Fully specified ideal chosen-vector OT, executed only as an algebra oracle:**
Client samples full-ring independent `r_i`. For every input coordinate `i` and
bit `ell=0..k-1`, sender samples independent vector
`T[i,ell] in (Z/2^(k-ell))^m`. Private ideal choice bit of `r_i` selects

```
M0 = -T; M1 = W[:,i]-T             modulo 2^(k-ell)
s = sum_i,ell 2^ell M_selected     modulo 2^k
q = sum_i,ell 2^ell T              modulo 2^k
```

Then `s+q=W*r` exactly. At `ell=0`, full-ring independent vector masks ensure
`s` is uniform independent of `r`; `q` also independent of `r`. No extra `q`
transmission is required: sender locally computes it. Client uses same two-frame
online consumer. Ideal OT hides choices from Sender, but public delta means
Client can recover `T=bit*W-M_selected` and the unselected message. Receiver
unselected-message secrecy is neither provided nor needed here. Authenticated
malicious implementation absent. Python `random.Random` is **toy fixture RNG**,
never cryptographic mask generator. Source algorithms are not implemented.

Per row selected messages cost
`n*sum_ell ceil(m*(k-ell)/8)` bytes; chosen-message adaptation from private random
OT sends **two ciphertext vectors** plus `k*n` private-choice switch bits, base
OT/key exchange, authentication, framing and failure checks. COT's single shared
delta cannot supply independently chosen public columns `W[:,i]` for free.
Streaming selected vectors bounds scratch but cannot remove message traffic.
Client performs `k*m*n` matrix-dependent lane accumulations, shifts and mask-bit
extracts, plus OT decryption. Direct dense Client `W*r` MACs = zero in this
layout; lane additions are **not timing-equivalent MACs**, nor free. Original
online MACs remain remote, but extra matrix-dependent work is 24/32 lane additions
per original MAC for this saved fixture. This rejects the layout, not all PCFs.
Other agent's matrix-triple-specific study owns Beaver/matrix-triple alternatives;
this study does not modify or infer its findings.

**Not a communication lower bound:** these bytes reject only this explicit
two-ciphertext adapter. Because `W[:,i]` is public, OT can be bypassed: deliver
`T` (or suitable independent public-to-both expansion seeds) plus public/cached
`W`; Client locally forms selected messages without sending mask bits. Sender
still learns no `r`, but Client now performs the same `k*m*n` matrix-dependent
regeneration. A simpler bypass samples independent `q`, gives both parties its
seed, and has Client compute `s=W*r-q`: exactly **m*n additional Client dense
MACs per row**, requiring Client access to `W` and complete artifact distribution,
storage/paging and runtime costs. Knowing `q` is safe; giving Inference the `r`
seed is not. Neither bypass is the missing client-light matrix-PCF. Original
**online body MAC placement stays 100% remote** by its narrow definition; this
does not make additional Client `W*r` work zero or establish lightweight CPU/RSS.
Report original prefill/decode placement and extra regeneration independently.

Saved-pair exact-ring adapter totals: **2,593,466,523,648 selected-vector bytes**,
**5,186,933,047,296 two-ciphertext payload bytes**, plus at least
**104,618,752 packed choice-switch bytes** and unpriced base/setup/framing.
Client matrix-dependent accumulations: **1,490,814,631,936**, against
**56,615,845,888** original dense body MACs. These are operation counts, not
benchmarked CPU ratios. Even choice switches alone exceed first-pair 20% headroom.

## Finite all-link headroom

Historical application bodies, not full wire/latest Low:
`C0=124905091`, `T0=368670156`, correction removal `S=139060689`; first pair
`493575247`. Source hashes bind saved 96 stage identities/weight hashes and
15,297 executed rows independently against narrow-ingress and row-demand evidence.
Each horizon pair is new 150-token prefix/eight outputs then its exact repeat;
seven decode executions per request, not eight. No full base/per-token rates
inferred. Exact-ring adapter uses original ring24/32, not saved 23/24/26 output
codec widths as permission to narrow full-ring pads.

For both endpoints, both control and candidate have same artifact lifetime:

```
control = C0 + H*T0                         # public delivery reused
candidate = control - H*S + B + H*F + ceil(H/L)*E
headroom/pair = S - p*(C0/H+T0) - B/H - ceil(H/L)*E/H
```

`F` includes **all** replacement traffic, openings, new helper, distribution,
failures, authentication and framing; `B` complete additional setup; `E` renewal
including first issuance; `L` reviewed capacity converted to pairs from total
correlations, not chosen for savings. If public delivery all recurs, replace
`C0/H+T0` by `C0+T0` for both sides. Full-wire overhead remains unknown.

| Total new cost allowance, no setup/renewal | 20% | 35% | 50% |
| --- | ---: | ---: | ---: |
| First pair, either endpoint | 40,345,639.60 | -33,690,647.45 | -107,726,934.50 |
| Infinite matched public reuse | 65,326,657.80 | 10,026,134.40 | -45,274,389.00 |

JSON gives exact rational headroom for 1/2/4/16/64/1024 pairs at both endpoints.
Free correction removal alone caps first pair at 28.17416%, recurring at
37.71954%. With zero new costs, 35% first occurs at five reused-public pairs;
50% impossible for this removal alone. At 16 pairs 35% allows only
7,293,835.534375 bytes/pair before setup/renewal. Additional byte ingress/reply
savings require independently specified consumers, never combined percentages.
Renewal causes discontinuities: test debits 100 MB issuance every four pairs and
shows headroom decrease immediately after boundary. Finite correction tape
`B=H*S` has no matched network win. Unknown costs are not assigned zero in any
implementation claim; zero-cost headroom is conditional upper allowance only.

## Privacy shortcuts and replay

Exhaustive `Z_8`, `W=[3]` provider-view test gives TV 0 for independent `(r,s)`.
Setting `s=2r` gives `q=r`, hence provider recovers input and TV 1. Reusing `r`
reveals input difference from two openings; fresh rank-one/shared-lane pads reveal
lane differences (TV 1). Provider-readable indexed seed reconstructs `r` directly.
These reproducible attacks disprove those shortcuts; finite checks do not prove
real PRG/PCF security. Fresh independent masks are a joint-distribution obligation.

```
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_streaming_matrix_correlation_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_streaming_matrix_correlation.py tests/test_streaming_matrix_correlation_probe.py
target/frontier-sdk-env/bin/python scripts/probe_streaming_matrix_correlation.py --output target/streaming-matrix-correlation/replay.json
```

Fresh output required. JSON hashes probe/tests/document and saved ledger sources;
runtime reports standard-library Python boundary, no native/SDK/crypto import.
96 ideal-OT outputs, 512 independent exhaustive scalar consumers and 512 quotient
mask uniformity enumerations are checked.
Public-delta enumeration additionally recovers all 224 unselected toy messages;
full Sender view including every quotient mask and `x-r` has TV zero between two
toy inputs. Stage checks reject duplicate/missing ingress identities and ring/
encoded-width mismatches against original saved geometry and narrow contract.
Ring16/24/32 signed coefficient and full-domain endpoint cases included. Tests
also check stage reconciliation, chosen/random adapter bytes, matched endpoints,
finite tape, cancellation/expiry/replay and rollback counterexample.

**Next gate:** obtain full source PDFs and identify exact fixed-W specialization,
role-separated key generation and Client expansion algorithm over `Z/2^k`.
Supply reviewed concrete security/total-Q parameters, setup/key size, every
conversion/opening and renewal cost. First small-stage experiment should compare
complete generation + both role expansions + online consumer against trusted
dense Preparation, while measuring client whole-process RSS and separately extra
crypto work. Without that specification, scaling Dory's COT memory or generic
DCR VOLE claims to PLLM would invent the decisive missing primitive.
