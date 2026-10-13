# Bootstrap plus exact byte ingress: bounded rejection and open PCF gate

**Outcome:** no priced correction-free cartridge passes. Direct vector chosen-OT
and client subset-table layouts fail. Silent OT/VOLE is a useful subprimitive,
not a complete matrix-carry consumer. Generic exact-ring circuit PCFs exist in
primary literature; dismissing them as mathematically impossible would be wrong.
Their concrete matrix-specific key generation, expansion, byte lift and complete
resource cost remain missing here. Nothing is integrated or benchmarked.

This screen owns only its probe, tests, this document and JSON evidence. It uses
standard-library integer algebra and saved source-bound checkpoint geometry. No
SDK/native/crypto import, new checkpoint, package install, protected decoder,
performance cohort, measured client-memory claim or upstream implementation.

## Exact signed-byte lift

Let `x` be signed i8, `r` independent fresh uniform in `[0,255]`, `u=x mod256`,
`s=[x<0]`, `c=floor((u+r)/256)`, and `z=(x+r) mod256`. Then exactly:

```text
x = z-r+256d,    d=c-s=floor((x+r)/256) ∈ {-1,0,1}.
zb=(x+128+r) mod256,
t=floor((x+128+r)/256)=d+[z>=128] ∈ {0,1},
x = zb-r-128+256t.
Wx = Wz-Wr+256Wd
   = Wzb-Wr-128 W1+256Wt.
```

Bias `128 W1` is public and precomputable, but signed carry is not dispensable.
For original ring `2^K`, `K∈{16,24,32}`, these equalities hold modulo that ring.
For public row-residue width `h`, only `Wd mod2^(h-8)` is needed when `h>8`;
rows `h<=8` need no carry contribution. Retain original compiler's signed bound,
residue decoding, scales, biases and float rounding. Endpoint `-128` is tested
algebraically; that does not widen an installed activation contract from -127.

Client knows every carry already. Challenge is privately **producing complete
`Wr-256Wd`**, or its additive output shares, without client doing unpriced dense
`W` work. Sending `d`, `Wd`, an Inference-owned pad seed, public private-value
lookup indices, or one shared scalar pad for a whole lookup table is invalid.
Tiny exact full-view enumeration gives byte-only TV zero, but `(z,Wd)` with legal
`W=[1]` TV `7/8` in signed3. Table differences identify all eight pads; reusing
one byte pad reveals cross-input differences. These are attacks on specified
shortcuts, not on PCFs or all private carry algorithms.

Probe exhausts 65,536 byte/pad pairs and independently checks 1,152 ring outputs.
Clear oracle directly multiplies Python signed integers; candidate separately
lifts bytes. All three original rings and signed endpoints are covered.

## Fully specified functional ideal-OT construction, priced rejection

One concrete correction-free functional construction avoids byte ingress entirely.
For each coordinate `i` and bit `k`, Sender/Inference samples independent fresh
uniform **whole output vector** `T[i,k]`. Coefficient is `2^k` for `k<7`, `-128`
for `k=7`. Ideal private chosen-OT returns Client one of:

```text
M0=-T[i,k], M1=coefficient*W[:,i]-T[i,k].
```

Client chooses corresponding two's-complement input bit and sums its received
vectors. Inference sends `sum T`; reconstruction is exact `Wx`. Probe independently
checks 252 outputs. In ideal-OT model, independent full-domain masks hide choices
from Sender and unselected messages from Receiver. This is a functional trusted
ideal-primitive specification, **not** a deployed OT security proof, ownership
implementation or malicious-verification protocol. Client gets authorized result;
sender's final aggregate mask is safe only under that private-choice primitive.

Already the selected vectors require **1,330,769,485,696 bytes** per historical
pair with exact public per-row widths. Ordinary two encrypted-message wrapper
requires **2,661,538,971,392 sender payload bytes**, plus Receiver's private-choice
switch flags when adapting random OT, base OT, frames and aggregate output.
Selected-vector floor is a bound on this explicit layout, **not** all vector OT.
Its `8*n*m` client additions also violate intended lightweight client placement.

Stronger byte layout publicly rotates `z` to `zb`, then uses eight private pad bits
and just **one** biased carry `t`. For pad bit `k`, sample vector masks modulo
`2^(h-k)`, choose between `-T` and `-W[:,i]-T`, and shift selected vector by `k`.
For carry, choose `-T` or `W[:,i]-T` modulo `2^(h-8)` and shift by eight. Sender
sums shifted masks into `Wzb-128W1`; receiver adds selected vectors. This fully
specified functional ideal-OT consumer independently checks another 252 outputs.
Fresh quotient masks and private ideal selections are essential. Sum per-row
selected widths is `sum(k=0..7)(h-k)+(h-8)=9h-36`, not nine full-width vectors.
Still needs **1,242,344,364,912 selected-vector bytes** for this fixture. Sending both
messages doubles this; both directions' flags/setup remain additional. Compact
ROT seeds do not create arbitrary pair differences `W[:,i]`: a chosen-message
adapter must send vectors, send public `W` and make Client expand them, or supply
a genuinely specified function-correlated generator. Relabeling that missing
generator “vector-correlated OT” does not price it.

## Literature: subprimitive versus missing algorithm

Only author primary pages were consulted; statements here use their abstracts,
not an imported implementation or assumed paper benchmark transfer.

* [Boyle et al., 2019/1159](https://eprint.iacr.org/2019/1159): two-round silent
  random OT under LPN variants; malicious variant in ROM. Supports low-communication
  **input-independent correlations**, not arbitrary chosen vector messages or
  free `W` multiplication. Parameters, OT conversion and both role directions
  must be priced separately.
* [Li et al., 2025/1223](https://eprint.iacr.org/2025/1223): PCGs over Galois rings,
  including `Z/2^K Z`, OLE, authenticated triples, matrix multiplication and
  circuit-dependent preprocessing. Abstract states sublinear communication with
  nonzero quasi-linear computation. Thus “only field PCGs exist” is false. No
  finite PLLM seed-size/renewal/CPU claim follows from asymptotics.
* [Hasler et al., 2025/1637](https://eprint.iacr.org/2025/1637): Ring-LWR/BFV PCF
  for arithmetic-circuit correlations; shared BFV setup, local pseudociphertext
  bootstrapping and evaluation. Plausible generic route to **matrix-specific
  correlations**, not symmetric seed expansion free of FHE work. Missing here:
  exact `W,r,s` role-separated circuit/keys, private carry-consumption circuit,
  concrete widths, parameter/security accounting and aggregate client expansion
  resources. This route remains open, not a negative cryptographic theorem.
* [Li et al., 2026/196](https://eprint.iacr.org/2026/196): prime-field OLE/VOLE via
  Walsh–Hadamard transform. One VOLE scalar across a vector is not independent
  carry scalars against arbitrary public matrix columns. Published prime-field
  throughput cannot be charged as exact ring lifting; explicit toy field257
  residue 133 differs from required ring16 residue 32,258. A larger field could
  certify bounded integer products, but randomized sharing, conversions and
  reconstruction then require their own exact ring contract.
* [Secret Replication, ITC 2026](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITC.2026.7):
  existing project operator has one scalar/vector consumer in fields 2/17/257.
  Neither independent lane masks nor full exact-ring correlation consumer is
  supplied by that installed contract. See
  [secret-replication reference](secret-replication-reference.md).

Even if an ideal matrix-PCF supplies fresh additive shares of `Wr`, that alone
does not supply input-dependent `Wd`. A fresh full-ring carry-share delegation
recodes `8+(k-8)=k` input bits and keeps original correction, as already rejected
by [narrow ingress](narrow-ingress-reference.md). A new PCF must specify **both**
material generation and its online private consumer, not count only compressed
keys. Standard Boolean/arithmetic conversions need their own correlations and
openings; this report grants no free conversion primitive.

### Bounded coefficient structure

Signed-i8 coefficient bounds do not certify useful low rank, sparse carry output
or common-column structure. Saved exact per-row `Wb` widths cannot beat quotient
widths; shifted unsigned affine bounds do not fix that. A future certified exact
factorization could help, but this screen has no such factorization of checkpoint
matrices and makes no universal full-rank claim.

Client-local subset tables can evaluate eight pad bit planes and one biased carry
plane without exposing indices. Exact full-model table bytes at block
widths 4/8/12 are **2,862,612,480 / 22,900,899,840 / 242,819,432,448**. Client lane
accumulations are respectively 2.25 / 1.125 / about 0.7527 times original dense MAC
count, plus expansion/indexing. Additions are not measured MAC time ratios. These
layouts fail client storage/placement gate; table delivery, setup and lifetime
ownership only worsen costs. Moving tables to Inference and sending indices
reveals private values; oblivious lookup is missing and cannot be free.
[Correlation compiler](correlation-compiler-reference.md) keeps tables at
Preparation and still transmits corrections; that is not a correction-free PCG.

## Source-derived rows and complete all-link budgets

Source-bound control is 493,575,247 application-body bytes: corrections
139,060,689; replies 139,123,150; inputs 90,443,518; public bundle 124,905,091;
controls/seeds 42,799. Recurring arithmetic subtotal is 368,627,357, whereas
recurring **all-link** subtotal is 368,670,156. No standalone percentages multiply.

Control's first request has 150 prefill rows and seven decode executions. Exact
repeat hits prefix cache and executes only seven decode rows; it is not another
150-row prefill. Terminal last-layer o/gate-up/down each execute one prefill row.
Thus 93 stages execute 164 rows each, three execute 15: **15,297 rows**, exactly
matching reserved/issued/claimed material ledger, zero burned/discarded/remaining
stage rows. Output-token count eight is not eight decode matrix executions.

Evidence verifies all 96 stage identities, shapes and checkpoint weight hashes
against independent saved narrow-ingress row histograms. Public stage widths are
23/24/26 bits, original rings 24/32 bits for this fixture. Exact ingress payload
is **90,104,112 bytes**, proposed byte payload **28,732,928**. Retained measured
input frame/envelope overhead is **339,406 bytes**. Probe independently enumerates
source `pack_encoded` array6/bin headers, batch and ticket IDs, dimensions, 8-byte
magic and u32 rows; saved outer-envelope residual remains charged. New byte schema
is hypothetical, not an accepted existing transport. Per-stage inner frame sizes
and residuals appear in JSON. Output payload is 138,864,059; original response and
correction metadata are visible rather than treated as payload. Optimistic bound
removes **all correction bodies including metadata**, retains other overhead.

Perfect correction removal plus byte narrowing saves **200,431,873 bytes** before
any helper/setup. Cold candidate floor is **293,143,374**, saving **40.6082%**.

| Matched horizon | Helper + amortized bootstrap budget/pair at 40% | At 50% |
| --- | ---: | ---: |
| One cold/repeat pair | 3,001,774.20 | **−46,355,750.50** |
| Two pairs, same artifact lifetime | 27,982,792.40 | −15,129,477.75 |
| Four pairs | 40,473,301.50 | **483,658.625** |
| Sixteen pairs | 49,841,183.325 | 12,193,510.90625 |
| Infinite matched warm lifetime | 52,963,810.60 | **16,096,795.00** |

Cold 50% is impossible for these savings alone: short **46,355,750.50 bytes**,
even granting free helper/bootstrap. Arithmetic-only 50% budget is
**16,118,194.50**, close to earlier ~15 MB hypothesis, but excludes controls and
public delivery. Warm control warms too; removing cold delivery only from candidate
would invent a win. Four-pair 50% headroom is under half MB, not 16 MB.

For byte `Wr`/carry vector-OT selected-vector floor, cold 50% target gap is
**1,242,390,720,662.50 bytes**, before both-message wrapping or setup. No measured
traffic saving is claimed. These are historical-body sensitivity projections;
changing artifact/request/cache/schedule lifetimes requires a new matched ledger.
Horizon projections repeat historical row demand (one cache-miss prefill plus one
exact-repeat cache hit per pair), not an endless identical-prompt hot-cache stream.
They assume same public artifact identity and category costs; future changed outer
metadata or cache behavior is not measured by this enumeration.

## Promotion gates, renewal and replay

For horizon `H` pairs and reviewed setup capacity `L`, charge:

```text
control   = P + H*R
candidate = Pnew + H*(R-saving+helper+fresh+failure+verify+frames)
            + ceil(H/L)*B.
```

`B` includes both OT directions where used, key exchange, authenticated role setup,
public compiled programs and distributed generation; unused/failed material burns.
`Pnew` includes all public artifacts, including extra client tables. Capacity `L`
cannot be invented from “one-time base seeds.” PCG/PCF can expand distinct fresh
use indices without fresh transmitted keys if its reviewed security contract says
so; reusing expanded masks, tables selected publicly by private values, labels or
correlations is not equivalent. Bind model/weights/rings/roles/stage/context/index,
prevent clones/replay, burn on malformed input/cancel/failure. Renewal follows
concrete cryptographic advantage and statistical-failure budget over total uses
`Q` (e.g. union-bound `Q*epsilon`), not amortization convenience. No parameters,
capacity, failure probability or trusted setup implementation selected here.

Selected verification must retain priced Freivalds/check correlations/openings and
failure handling; an algebra oracle is not malicious-result verification. Threat
model must explicitly distinguish trusted noncolluding offline Preparation,
semi-honest OT assumptions and malicious security. No provider gets Client pad
seeds. Client dense MAC placement, expansion CPU, table/cache storage, scratch,
lifetime RSS, latency and aggregate computation require separate admission and
measurement. Unknown costs remain JSON null/explicit unknown, never zero. To pass:
fully specify role-separated exact-ring PCF **and private byte consumer**, then
fit complete traffic within matching budget and retain >90% remote original body
MAC placement without hidden client `W*carry` work.

Replay, no conftest/SDK activation or checkpoint access:

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_bootstrap_ingress_probe.py
target/frontier-sdk-env/bin/python scripts/probe_bootstrap_ingress.py --output target/bootstrap-ingress-replay.json
```

Output must be fresh; runner refuses overwrite. JSON hashes probe/tests/document,
saved controls/geometry and actual checkout source files, plus Python executable.
It does not claim those checkout bytes were imported in historical cohorts: saved
evidence retains original runtime/source bindings. No live seeds, masks, prompts,
tokens or private values are archived; tiny public toy witnesses are explicit.
Evidence: `docs/evidence/research-bootstrap-ingress.json`.
