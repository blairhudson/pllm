# Fixed-W proof-first correlation gate

**Decision: no cheap-client generator admitted.** New executable boundary:
`q=0` is not inherently input-insecure. Fresh full-ring `r`, private `s=W*r`,
and opening `z=x-r` hide chosen inputs from Inference. However, eliminating
correction delivery does not eliminate computation or delivery of `W*r`.
Three concrete alternatives expose that bill. A zero-MAC lookup control also
shows why counting only client MACs can misclassify expensive expansion.

This is bounded correctness, leakage falsification and analytical accounting;
not a PCG security proof, serving benchmark, or Low/Medium qualification.
No checkpoint tensors, upstream runtime code or dependencies loaded.

## Contracts and distinct ideal functionalities

All products below use **arbitrary public signed-i8 W over Z/(2^32)**,
full-ring chosen inputs and exact ring outputs. Fixture is a nonstructured 3×5
matrix, including −128 and 127. Independent oracle uses signed Python integers
then masks once to 32 bits. Transcripts cover seven fresh rows per four main
variants; lookup control checks a separate exact operator, not a fifth role-key implementation.
Tests additionally cover arbitrary tall, wide and scalar matrices.

| Construction | Client-only view | Inference-only view, besides public W | Setup / expansion | Decision |
| --- | --- | --- | --- | --- |
| Prepared baseline | private roots for r and independent s | Q-row q tape, q=W*r−s | Preparation computes Q dense products; client expands two PRGs | Fully specified comparison; not compact generator |
| Zero-correction tape | private r root, Q-row W*r tape | no secret correlation key; q=0 | Preparation computes and delivers W*r to Client | Cheap online expansion, but output tape; only saves one 32-byte root/epoch |
| Zero-correction local | private r root, public W | no secret correlation key; q=0 | Client computes fresh W*r each row | Correct/input-private, not cheap |
| Direct-weight | private r root, q root, public W | q root, **never r root** | Client computes s=W*r−q; both expand q | Correct/input-private, adds dense client product |
| Binary subset-table control | private r root, public subset tables | no r root; q=0 | Exact W*r via table lookups, shifts and additions | Zero dense MACs, but large tables and more scalar work |

**Setup:** trusted offline Preparation receives W, source/layout/stage/role/epoch
binding and finite capacity Q, samples independent private roots, computes only
specified tapes, delivers each role its own key. It receives no online x, z or
reply. The replay fixture's deterministic RNG and disclosed transcripts are
public test data, not production secrets. SHAKE256 expansion is domain-separated
by full binding digest, index and r/s/q label. Security of seeded variants is
conditional on a private-root PRG; the program does not certify that assumption.

**Query:** externally reserve/burn index j, then Client derives `(r,s)` and sends
`(binding,index,z=x-r)`. **Evaluate:** Inference derives its q and returns
`(binding,index,W*z+q)`. **Reconstruct:** Client adds s; result is exact W*x.
Neither online object receives Preparation's combined key view. Client roots do
not enter InferenceKey. Independent s is used only by prepared baseline.

Original ideal correlation samples r and s independently and supplies q=W*r−s.
Zero-correction is **not** an implementation of that distribution: q=0 is a
perfect distinguisher. It instead realizes private outsourcing of a public
linear map. For independent uniform full-ring r per row, Inference's entire
online view is `(W,bindings,indices,z,W*z)` with z uniform independently of x;
a simulator samples z and computes W*z. This argument covers arbitrary W,
including singular W, and adaptive chosen inputs with fresh unused masks.
Preparation alone sees masks but not online transcript. Preparation/Inference
collusion reveals inputs, as does revealing r root. Direct-weight similarly
allows simulation given public q root independently sampled from r root.

These arguments establish **input privacy, not malicious correctness**. A
malicious Inference can add any output error undetected in every implemented
variant. A ciphertext MAC on transport would not prove W*x. No authenticated
correlation, computation proof, durable authority or full malicious protocol is
implemented/priced; all promotion gates remain closed.

## Falsification and lifetime controls

Exhaustive Z/8 full-view distributions give total variation 0 for independent
ideal pads **and** q=0/full-r; this avoids falsely rejecting q=0 merely because
s depends on r. Four bad shortcuts give TV=1:

* r restricted to ker(W) makes W*r=0 cheap but exposes W*x in W*z;
* r=2a retains rational rank but leaks parity over Z/(2^k);
* public low-rank r basis exposes a left-annihilator of x;
* cross-column relation r2=3r1 leaks x2−3x1 even with fresh column labels.

Any claimed public linear expansion r=B*a must be surjective on ring32, not
merely full-rank over Q: reduction B mod 2 must have row rank n. This condition
is necessary/sufficient for uniform a to give uniform r over this finite local
ring. It is a test for **this public linear shortcut**, not a lower bound for
computational PRGs or arbitrary fixed-W algorithms. Hidden structured generator
keys need their own complete multi-index security argument.

Authority burns index before validation/exposure. Cancellation, expiry, malformed
messages, mismatched roles/bindings and bad replies retire capacity; exhaustion
requires fresh epoch/setup. Client restart/clone retains the **same external
authority**. Tests also deliberately replace authority and show mask-reuse delta
leakage. Python set is an ideal external-owner model, not rollback-safe storage;
cloned authority snapshots and actual durable anti-rollback remain unresolved.
Bindings cover exact W digest, shape, ring, layout, stage, roles, epoch and Q.
Changing labels does not make restarting old material safe.

## Complete specified-stage ledger

Baseline uses the same W, ring32, rows, prepaid capacity, bindings, row framing,
failure retirement and online arithmetic as candidates. W is initially at
Preparation and Inference in all variants; new client W delivery is charged.
No baseline whole-client implementation/timing claim is imported from historical
measurements. Ledger is a fully specified **analytical stage comparison**, not a
replacement of existing variable-width SDK codecs.

Each epoch pays a 64-byte canonical binding on each Preparation→role link:
digest32 + epoch identifier16 + shape/ring/capacity four uint32s. Stage/layout/
source/role metadata are bound by digest and are common pre-agreed protocol
constants for this fixture. Client r root32 always; s root32 for baseline,
shared q root32 to each role for direct-weight. Every online direction pays
binding digest32 + column index8, plus ring32 vector. No acknowledgements,
signatures, TLS records, MAC/proof or authority RPC bytes are asserted zero;
those remain explicitly **unknown**, so none qualifies as complete wire savings.

Let E=ceil((successful+burned)/Q), I=E*Q, M=4m and N=4n:

| Item | Prepared | Zero tape | Zero local | Direct weight |
| --- | ---: | ---: | ---: | ---: |
| Preparation→Client | 128E | 96E+IM | 96E+mn | 128E+mn |
| Preparation→Inference | 64E+IM | 64E | 64E | 96E |
| Client→Inference | rows*(40+N) | same | same | same |
| Inference→Client | rows*(40+M) | same | same | same |
| Client dense MACs | 0 | 0 | rows*mn | rows*mn |
| Preparation dense MACs | I*mn | I*mn | 0 | 0 |
| Client tape writes / successful reads | 0 / 0 | IM / rows*M | 0 / 0 | 0 / 0 |
| Client paged-W reads | 0 | 0 | rows*mn | rows*mn |

All new public W bytes, PRG output bytes, client vector additions, live roots,
one-epoch secret tapes, streaming vector workspace and external logical I/O are
separate fields in JSON. Subset-table ledger separately charges table creation,
distribution, bit extraction, shifts and local table I/O. Reported workspace is representation-level vector
payload, **not Python peak/RSS**. Resident W/tables/tape, SDK, transport, scratch,
allocator and renewal overlap must be added. Inference reads W each successful
row; Preparation source scans for each generated product explicitly charge
`I*mn` i8 reads if W is paged. Cached or repeatedly scanned W does not
alter charged delivery/MACs. Client/Inference compute PRGs only for successful
rows in this optimistic ledger; failures after expansion add expansion costs.
Additional complete failed setup cohorts multiply epoch setup via `attempts`.
Outputs discarded before use still pay their full setup/material capacity.
**Directed totals and `all_link_bytes` exclude failed-attempt online traffic.**
Online links count successful rows only. Burned columns may fail after an
opening or partial/full reply was sent; prepaid material does not pay those
messages. `failed_attempt_online_bytes` leaves both online directions unknown,
and `directed_totals_exclude_failed_attempt_online_traffic` explicitly flags
this exclusion. These totals are not complete failure-inclusive traffic costs.
Even full failed setup cohorts charged via `attempts` leave their online traffic
unpriced. Comparing ledgers does not establish a failure-inclusive traffic win.

For analytical **9728×896 ring32**, 288 successful rows, Q=128, one burn:
384 columns issued. Zero tape relocates **14,942,208 bytes**, saving only 96
root bytes in total; client external successful reads add **11,206,656 bytes**.
Direct-weight/zero-local add **2,510,290,944 client MACs**, **8,716,288 W bytes**
and **2,510,290,944 logical paged-W read bytes** across successful rows. This is geometry
sensitivity, not original layer-18 24-bit compiled stage execution.

Subset tables group four input coordinates. Store 16 ring32 subset sums per
group and output lane. Expansion splits each r coordinate into 32 bitplanes,
fetches one vector per group per bitplane, shifts/adds it. Full-group tables
cost **16mn bytes**, expansion **8mn lane additions** and **32mn table-read
bytes per row**, despite zero dense MACs. At above shape: **139,460,608 bytes**
public table delivery, **69,730,304 additions/row**, **278,921,216 logical table
read bytes/row**. Already exceeds 14.94 MB removed correction material at this
horizon, before arithmetic, setup, process memory and renewals. This is a
specific rejected fast-linear-circuit control, not rejection of all such circuits.
Table indices depend on r. Local table storage must be trusted; revealing access
patterns to Inference would require an oblivious-access construction and extra
costs absent here. This control does not inherit secure external-memory access.

Baseline/candidate ledgers and admission tests are mandatory **before scaling**.
Zero-tape fails compactness; direct/local fail cheap expansion; tables fail
same-task traffic at tested horizon. No measured CPU/latency win follows from
any byte count, and no timing run was performed.

## Primary-source construction audit

Evidence JSON pins full PDF/text hashes, exact PDF-page text hashes and short
excerpts. Pages are **1-based PDF**, not document-internal page labels. Existing
available PDFs are reused without vendoring methods. On 2026-10-15 both 2025
landing pages were reachable, but fetching all four current PDF URLs returned
HTTP 403; **latest-revision equality cannot be certified**. Findings below refer
only to pinned available full texts, never abstract-derived interfaces.
Fresh `pdftotext -layout PDF -` extractions matched all four pinned text files
byte-for-byte; probe rejects either PDF or full-text hash drift.

| Source / exact location | Actual construction interface | Fixed-W / client / security consequence |
| --- | --- | --- |
| [2025/1223](https://eprint.iacr.org/2025/1223), Construction 2 p28, Thm5.4 p29 | Uniform public a in structured algebra; private sparse s,e; FSS FullEval then trace/isomorphism transforms; scalar OLE `x0*x1=z0+z1` | Exact ring32 supported, but neither arbitrary rectangular W input nor cheap asymmetric `W*r` expansion appears in this algorithm |
| Same, Thm5.6 proof p30, DefA.1 p42 | Programmability selects sparse-generator random coins ρ; outputs ψ(ρ) | Not an inverse encoder ψ⁻¹(W). Replacing random public a by checkpoint W is not licensed by stated QA-SD assumption |
| Same, Thm6.1/Remark6.2 p33 | Authenticated ring triples with wider ring and higher-bit masking at opening | Authentication exists for that triple protocol; blindly opening stronger wider-ring equality can leak. Not authenticated fixed-W zero-q interface |
| Same, §7.1/Table3 p35, Table5 p36 | Non-broken regular `(λ,c,t,m,n)=(128,5,27,2,15)` parameter point; expansion includes PRG calls and global transforms | Retain parameters if implementing scalar control; no reduced-security tiny crypto instance. No extrapolated scalar rate presented as fixed-W speed. Broken c=3,t=27 excluded |
| [2025/1637](https://eprint.iacr.org/2025/1637), Def6–7 p23, §7.1 p24 | Additive shares of C(random inputs); **each role** bootstraps pseudo-ciphertexts, evaluates C, decrypts shares | C(u)=(u,Wu) expresses a fixed map but yields shared input, not automatically client-owned r. One explicit share-opening adapter would send Inference's n-word input share to Client each row; complete-key/reveal security and both HE costs still need proof/pricing |
| Same, §6.6/Table3 p22, authentication p24 | Runtime estimates at t=2^64, not our ring32 stage; extra encrypted MAC multiplication | 1839s/encrypted element and ~9 triples/s are paper estimates, not ring32 or end-to-end fixed-W numbers. Authentication not free; negligible-noise conversion condition retained |
| [2026/2030](https://eprint.iacr.org/2026/2030), Cor3 p14, §4–4.1 p15, §5 p17 | Integer-modulus COT→VOLE uses about log2(M) COTs; generic OLE extension over F2 | Not a chosen arbitrary matrix generator. Cor3 invokes DCR+GRH; §4 assumes inner-product-membership weak PRF. Lifting/statistical failure compose across uses |
| [2026/2298](https://eprint.iacr.org/2026/2298), §7.2 p72, authentication gap p75 | Fresh sequential right columns of ONE field matrix triple; ring extension left future work | Not exact ring32 construction. Across-instance reuse/programming and MAC correlations not supplied/priced; bounded-column D*B_client dense adaptation remains |

New source-level distinction: **expressibility is not role ownership**. A generic
additive-circuit PCF for `(u,Wu)` gives `(uC,tC)` and `(uI,tI)` with
`tC+tI=W(uC+uI)`. Revealing uI to Client yields desired `r=uC+uI,s=tC,q=tI`
algebraically and costs another 4n bytes/row, but revealing outputs conditional
on full PCF keys needs its own compositional argument. It does not remove
client bootstrapping/evaluation. We implement neither this unproved adapter nor
a claimed impossible-to-exist theorem. Exact-ring programmable OLE remains a
valid primitive lead; the missing lightweight fixed-map compiler remains open.

## Replay and promotion

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_fixed_weight_correlation_gate.py
target/frontier-sdk-env/bin/python scripts/probe_fixed_weight_correlation_gate.py --output target/fixed-weight-correlation-gate-replay.json
```

Output path must be new. Frozen evidence:
`docs/evidence/fixed-weight-correlation-gate-bounded-v3.json`.
Original `fixed-weight-correlation-gate-bounded.json` is retained as first attempt;
v2 adds an explicit client-addition admission predicate, with unchanged ledgers.
Original/v2 evidence remain unchanged. Parent audit found zero-tape setup also
expanded unused s: Q4/m3/n5 actually generated 32 words versus ledger's 20.
V3 removes that expansion, retaining actual zero-tape construction's 20 words.
Independent interception now checks actual setup, Client and Inference PRG words
against literal per-label counts and ledger bytes for every implemented mode.
V3 additionally marks failed-attempt online traffic unknown as described above.
Worker validation: **35 focused tests** and Ruff pass; **84 ring32 transcript
outputs** checked, plus exact subset-table operator and exhaustive toy leakage
controls. No cryptographic security certification follows from these tests.
Promotion requires explicit role-local secure Gen/Expand for arbitrary W,
multi-index chosen-input/reveal statement, malicious-output decision, durable
one-use authority, fully priced setup/renewal/failure/I/O and same-task cheap
client win. Only then parent may admit larger serial experiments. Existing
26–32% cold/~38% warm correction-removal ceilings remain separate outer budgets;
none of these controls raises them or establishes serving savings.
