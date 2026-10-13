# Reusable public bootstrap cartridges: exact algebra, conditional network gates

**Result: storage tradeoff survives; no serving/Low/privacy promotion.** A public
two-accumulator checkpoint DAG reproduces every selected column of Maverick's
`P=W G`. It is reusable matrix-derived material, not a reused mask or a secret
correlation tape. Its reconstruction requires substantial paid client matrix work.
P-only is a distinct **unverified** contract matching only the prepared baseline's
lack of malicious-result checking; it does not retain Freivalds or Maverick's Q
checks. Matching that one property does not establish equivalent privacy.

## Construction and independently checked algebra

The probe transcribes the existing public SHA-256 rejection sampler, rate-1/8
repeat/permutation/scaling code and BabyBear arithmetic from
`crates/pllm-core/src/coded_linear.rs`. For each weight row `w`, let `L=8n`:

```
A[j] = A[j-1] + scale1[j] * w[perm1[j] // 8]
B[j] = B[j-1] + scale2[j] * A[perm2[j]]
P[row,j] = B[j]
```

Store signed-i8 W and both accumulator states immediately before each multiple
of stride `d`. To obtain column j, start at its preceding B checkpoint, walk at
most d steps, and regenerate each needed A value from its preceding A checkpoint
using at most d original weight entries. Cache repeated A dependencies within a
client-local selected-column batch. No supports or memory accesses may become
visible to Inference through an unreviewed remote paging mechanism.

For each fresh query, sample uniform support without replacement and independent
uniform nonzero field coefficients using OS randomness. Compute `r=G e` and
`c=P e` locally; send `x+r`; evaluator computes `W(x+r)`; client subtracts c and
centers the residue. This eliminates a linear stage's recurring Preparation
correction rather than uploading an equally large one-use tape at bootstrap.
Sparse supports/coefficients and corrections are never persisted in evidence.
The probe has no production handles, secure erasure, authenticated role transport,
or malicious-response rejection; Python copies are not an ownership proof.

Independent oracle builds dense G by encoding input basis vectors, rather than
using syndrome or checkpoint reconstruction, and computes Python-integer signed
clear products. Public synthetic shapes are 5x11, 7x17 and 8x64: small shapes
deliberately fall outside native delegation admission and prove algebra only.
Eight fresh queries per shape include all -128 and all 127 inputs. Four strides
(1/4/16/64) check every P column and independently sampled eight-query unions.

Observed checks: **24 fresh queries; 736 mask values; 160 corrections; 160 exact
signed outputs; 21,952 checkpoint column values; 640 batched corrections**.
These categories overlap deliberately; do not call their sum independent queries.
Instrumented first/second loop counts and cache hits appear in immutable evidence;
they are measured execution counts, not timings or CPU-equivalent MAC ratios.

## Exact storage and charged work

For one m-by-n stage:

| Representation | Public payload, excluding common source manifest |
| --- | ---: |
| Existing P+Q | 64mn bytes |
| P-only | 32mn bytes |
| Checkpoint cartridge | mn + 8m floor((8n-1)/d) bytes |
| W-only | mn bytes |

Proposed per-stage manifest adds **96 bytes**: public code seed 32, matrix digest
32, two u64 dimensions 16, format/domain identifier 16. Source/client/version
binding, authenticated delivery, whole-file commitments and physical framing are
additional unknowns, not zero. Seed generates a public code, **not arbitrary W
correlations**. Every coefficient vector still incurs matrix-dependent work.

Checkpoint generation costs `16mn` field multiply-add lanes; each fresh support
costs at most `m*t*(d^2+d)` reconstruction lanes plus `m*t` correction lanes.
Here t is the existing conditional sparsity heuristic, not certified security.
Syndrome adds `16n` multiplies and at most `24n` additions; input masking and
output subtraction add n+m. Public code sampling/rejection, random sparse support,
disk reads, cache overhead and failures also cost work. No compression assumed.

The 96 existing compiler-derived Qwen body matrices total **357,826,560 i8 bytes**.
Full P-only is **11,450,459,136 bytes** including proposed manifests, versus the
existing roughly 22.9 GB P+Q. Strides 16/64/256 reduce bootstrap to
**1,786,708,992 / 713,229,312 / 444,859,392 bytes**. W-only is **357,835,776 bytes**.
This is storage/network accounting, not peak RSS. Row tiles could bound explicit
buffers, but disk/page cache, interpreter, model state and maximum whole-client
peak remain unmeasured. Paging proves no Low qualification.

W-only has two honest regeneration choices:

1. Generate selected P columns by a complete `16mn` field-lane encoding once per
   stage batch, then apply fresh supports to retained corrections/masks. Future
   decode inputs cannot be assumed ready; pre-sampling fresh material still owns
   live masks/corrections and consumes attempts.
2. Skip P and calculate `W*(G e)` directly per use: one complete `mn` client
   matvec, plus syndrome/masking. This smallest public cartridge is algebraically
   sound but exchanges network traffic for **one extra dense client matrix pass
   per query**. It supplies no function-specific PCG speedup. A full-ring fresh
   mask variant would likewise require actually computing W*r; neither ordinary
   seeds nor generic scalar VOLE remove that cost.

Remote evaluator still executes all original W*x-equivalent body MACs in these
symbolic constructions. Added client correction work is separate, substantial,
and must be budgeted; this is not measured >=90%-remote qualification or a loophole
allowing original body projections to migrate unpriced to the client.

### Stage-batched sparse-column reuse

For h independent fresh supports of weight t, expected union is exactly
`U=L*(1-(1-t/L)^h)`. A public selected-column transfer costs `4mU` versus
`4mh` original raw corrections: ratio U/h. At h=1 it is t-fold worse; small native
batches remain bad. Only near/full code-domain saturation and very large h does
U/h drop below one. Such delivery recurs and may expose supports, so it is not an
admitted alternative bootstrap. Local-disk column reuse avoids that channel but
still pays reads/reconstruction, `h*m*t` correction work and `4h(m+n)` live mask/
correction bytes. Evidence quantifies h=1/8/64/158/512/4096/8192/65536, not execution
of those full-model batches. Complete streaming encoding costs `16mn/h` lanes per
query before sparse application. Neither this nor a dense mask matvec is free.

## Warm-control horizons: 20 / 35 / 50 percent

Historical source is `docs/evidence/research-all-link-frontier.json`, **application
bodies**, not latest Low baseline or physical wire. Set C0=124,905,091 shared
public bytes, T0=368,670,156 recurring bytes, S=139,060,689 correction bytes:

```
control   = C0 + R*T0
candidate = C0 + B + R*(T0-S+H)
reduction = (R*(S-H)-B)/(C0+R*T0)
```

Arithmetic subtotal is 368,627,357; the additional 42,799 historical control/seed
bytes recur. The optimistic S removes **every historical correction byte**;
actual linear-stage subset is unproven and may be smaller. Both sides receive
shared public delivery warm, never compare candidate to R*493.575 MB. Report also
shows 10%/100% bundle-redelivery sensitivities: full redelivery reduces ideal
steady-state ceiling to 28.174%, eliminating 35% even without new overhead.

| Cartridge | Ideal 20% R | Ideal 35% R | 31-bit ingress-only 20% R | 50% R |
| --- | ---: | ---: | ---: | --- |
| P-only | 176 | 1147 | 295 | impossible |
| Checkpoint d16 | 28 | 183 | 47 | impossible |
| Checkpoint d64 | 12 | 76 | 19 | impossible |
| Checkpoint d256 | 8 | 49 | 13 | impossible |
| W-only, paid regeneration | 6 | 41 | 10 | impossible |

Ideal means H=0, not a prediction. Warm correction-only ceiling is **37.720%**,
so 35% has a mathematical horizon only under exceptionally low expansion/framing.
31-bit ingress versus assumed historical 24-bit lanes adds **26,379,359.42 bytes**
per pair, cutting ceiling to **30.564%**: 35/50% impossible. This ingress-only
column depends on a separate **prime-field side-information output codec**;
no result from that worker is assumed here.

If both ingress and outputs expand from assumed 24 bits to packed 31 bits,
ceiling becomes **19.558%**; at 32 bits it becomes **16.963%**. Then **all 20/35/50%
horizons are impossible**, independent of bootstrap amortization. Applying class
bodies as lane counts is explicitly a sensitivity, not measured serialization:
actual widths, lane counts, framing and codec are unresolved. For any finite
20% horizon, total added recurring H must be below **65,326,657.8 bytes**; for
35%, below **10,026,134.4 bytes**. Fresh authentication/control bytes and any
retained verification challenges must fit inside H. P-only has no verification
challenges; verified Q/Protocol-2 extensions would add artifacts/work/traffic.

Native limits remain **8 live queries, 65,536 issuance attempts including invalid
attempts, dimensions 64..16384, 256 MiB P+Q payload preflight**. Existing code is
unchanged and rejects both MLP shapes before allocating. Gate/up's P-only matrix
alone is 278,921,216 bytes, also above Low. Cartridge accounting does not authorize
either stage. Assuming 158 or 316 executions per stage per historical pair gives
at most 414 or 207 pairs per native object; evidence flags horizons beyond those
limits. In particular P-only's ideal 35% horizon exceeds either limit. A fresh
object cannot silently reset a lifetime security budget; reviewed aggregate
issuance and lifecycle contracts remain gates.

## Real rejection/promotion gates

- BabyBear is below the cited `q>2^31` distance-bound condition. Actual seeded
  ensemble distance and dual-LPN parameters remain uncertified. Successful algebra
  checks certify neither privacy nor distance.
- P-only deliberately loses verification. Do not compare it as retaining a
  Freivalds guarantee. Exact whole-decoder/logits/KV/checkpoint identity remains
  untested; only bounded signed-i8 linear algebra was exercised.
- Public W/checkpoint/bootstrap source/client/version authentication, bounded
  buffers/cache ownership, support-private local disk access, fresh one-use
  handle burning and distributed replay protection need implementations/review.
- Full-width ingress/output serialization and new authentication can erase all
  correction savings. Remove only proved linear correction bytes, charge every
  use and failed attempt, and use a current Low-qualified matched control.
- Performance and maximum whole-client RSS require separately authorized serial
  confirmation; no native import, wheel rebuild, model benchmark or checkpoint
  download happened in this probe.

**Strongest surviving conclusion:** reusable public checkpoint DAG provides an
exact compact-storage tradeoff; W-only is smallest but pays dense correction work.
Neither currently clears security, lifecycle, Low or end-to-end performance gates.
20% traffic remains conditional on an output codec plus paid client work; 35%
survives only the unrealistic zero-expansion upper bound; 50% needs another cost
category. No presently admitted 20–50% serving construction follows.

## Exact replay and source binding

```
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_bootstrap_cartridge_probe.py
python3 scripts/probe_bootstrap_cartridge.py --output docs/evidence/research-bootstrap-cartridge.json
```

Evidence path is immutable: replay to a **new** output path if already present.
`source_sha256` binds probe, tests, this document, every required research/native
source and both historical geometry/traffic reports before and after execution.
Fresh OS samples mean instrumented union/cache counts may vary on replay; exact
algebra invariants, totals, public code, symbolic geometry and horizons do not.
Focused suite: **16 tests passed**. No wall-time benchmark claims.
