# Certificate-indexed local exact output head: bounded reference

## Result and scope

Research-only source revision: `e49a48b6082f8f7d17f744ec1190e84d13a653dc`.
No SDK, native kernel, compiler, cache admission, or placement changes. Evidence:
[`head-certificate-index-tiny.json`](../evidence/head-certificate-index-tiny.json).
The installed interpreter ran 15 tiny correctness tests, including its float32
dequantization helper. Neither native head execution nor complete-logit parity
was established. No timing, checkpoint values, build, install, or response cohort
was run.

For each of two queries, a deliberately separated 64-by-64 matrix tested 8 rows
and independently verified all 56 skipped rows. Two random-box queries tested
all 64 rows and skipped none. The latter are an early falsification signal for
high-dimensional coordinate boxes, not a universal impossibility proof.

| Fixture (each query) | Exact dot MACs | Bound endpoint products | Domain products | Full scan MACs | Oracle MACs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Separated | 512 | 896 | 64 | 4096 | 4096 |
| Random boxes | 4096 | 1920 | 64 | 4096 | 4096 |

Products are not equal-cost CPU measurements. Bounds also require selections,
integer additions, float32 conversions, 28 or 60 endpoint dequants, and traversal.
Each index is 2,392 serialized bytes; each original unbiased head is 4,352 bytes.
The Python immutable source adds a 256-byte zero-bias vector (4,608 total).
Index construction visits 16,384 weight coordinates per fixture, with min/max
reductions and scale/bias scans in addition. These payloads exclude Python object,
allocator, serialization-copy, and whole-client memory overhead.

## Difference from the previous progressive-head idea

`network-io-next-five.md` and `next-five-network-methods.md` describe public high
bitplanes and conservative residual bounds followed by private precision
refinement/PIR, with worker scans and a constant-shape private retrieval gate.
`crates/pllm-core/src/progressive_head.rs` retains all weights in its oracle and
generates residual records for that separate experiment.

This experiment never requests private rows from a provider. The full original
public W8 head, scales, optional bias, and public hierarchy must be delivered and
authenticated independently of queries. Queries, candidate IDs, local page
decisions, hidden activations, and skipped-row traces stay client-local. Only
redacted operation counts and public source/index hashes enter evidence. Local
page access must not be backed by a remote range API, observable fault service,
or provider-managed paging. The implementation is an in-memory screen, **not**
a tested paged-head engine.

The head is already client-owned. This changes neither its ownership nor remote
body MAC share. Any eventual cohort must separately establish at least 90% remote
body MACs from the actual placement; this probe has no body and cannot establish
that denominator. It establishes no network-body saving over the current local
head. Any throughput benefit would be local work reduction only.

## Arithmetic admission and certificate

The admitted domain is a signed-i8 `[rows, columns]` head and signed-i8 query,
positive finite float32 row and activation scales, and optional finite float32
row bias. Columns are at most 16,384; exact signed dots fit int32 because
`16384 * 128 * 128 = 268435456`. Thus wrap32 centering cannot change these dots.
The screen additionally bounds resident source to 16 MiB and 4,096 rows. The
optional raw scan admits at most 256 rows and 4 MiB of weights.

The arithmetic contract is:

```
dot = exact_integer_sum(w[j] * x[j])
scaled = f32(f32(f32(dot) * activation_scale) * row_scale)
score = f32(scaled + bias)  # omitted entirely when no bias exists
winner = first row with maximum finite score
```

Separate NumPy float32 ufuncs model separate operations, not fused multiply-add
or float64 intermediate arithmetic. The installed `dequantize_matmul` helper
uses the same unbiased order. Native `offset_transport.rs:348-399` validates
positive finite scales and finite optional bias, centers the integer result,
casts to float32, multiplies activation then weight scale, and optionally adds
bias. A conservative per-query all-row magnitude check rejects possible
nonfinite outputs before pruning. It may reject finite baseline cases; that is
an admission restriction, never permission to return a token from an unsafe
certificate. Nonfinite inputs/results are outside this finite-score policy.

The actual client-owned stage in `model_binding.py:202-220` quantizes its input,
calls the native matrix `clear` method (or eligible Metal matrix), invokes
`dequantize_matmul`, adds optional bias in-place, rejects nonfinite output, and
returns a complete contiguous float32 tensor. This is the head-stage contract
being modeled, not permission to replace its returned tensor with an index.

Each binary vocabulary-range node stores coordinate min/max i8 weights, min/max
positive scale, maximum bias, and child/range geometry. For a query, choose min
or max coordinates according to the sign of each query component and compute
exact integer lower/upper dots. Evaluate all four dot/scale endpoint combinations
through the exact float32 arithmetic, then take their maximum with maximum bias.
Integer-to-float conversion and rounded float32 arithmetic are monotone on this
admitted finite domain. Evaluating the native-order endpoints therefore includes
rounding without an empirical epsilon. Scale dependence changes sign on negative
dots; using four corners avoids the invalid assumption that maximum scale always
gives the largest score. Independent coordinate/scale/bias extrema deliberately
lose correlations; large dimensional boxes can be useless.

An exact tested row supplies incumbent `(score, row)`. Skip a node only if its
upper score is smaller, or equal and **all** its indices are later than incumbent.
Otherwise descend or evaluate every leaf row exactly. There is no approximate
candidate-only acceptance. A bound-node budget failure discards certificates
and recomputes the complete exact head, charging already tested rows and bounds
plus all full-head work. The no-pruning path likewise performs full exact dot
work plus its index overhead, but returns only the greedy result.

The source is copied into immutable byte-backed arrays. The canonical public
index serializer binds contract/source digest and geometry; its 52-byte header
and all node records are counted. There is deliberately no external index
deserializer or trust admission: a future implementation must authenticate
source/contract lineage and validate or rebuild every bound from original
weights before treating externally published metadata as certificates. Merely
hashing an untrusted producer's index does not prove conservative bounds.

## Counterexamples and tests

* Negative bound: a node with dot `-1` and scales `1, 10` has upper score `-1`,
  not `-10`. The latter can falsely eliminate a winner against incumbent `-5`.
* Float32 tie: two 2,048-dimensional integer dots differing by one above `2^24`
  can round to the same float32 score. Exact-real or float64 ordering then picks
  the later row incorrectly. The test constructs this and demands row zero.
* Equal upper bound alone is not sufficient to prune an earlier-index row.
  First-index policy is part of every skip condition, including signed-zero ties.

Tests also cover negative/scaled rows, positive and negative maximum i8 values
at 16,384 dimensions, bias changing winner, subnormal scales/underflow, neighboring
float32 scales and cast boundaries, nonfinite/overflow rejection, invalid query
types, immutable source, serialized index size, full-work budget fallback, and
60 random index/query configurations. An independent vectorized int64/f32 oracle
computes every row, checks winning score bits, and verifies **every skipped row**
against its recorded bound and incumbent tie condition. Those private traces
never enter public evidence. Native head/tie equivalence remains a later gate.

## Complete logits and state: current blocker

`crates/pllm-compiler/src/lib.rs:4558-4599` semantic greedy selection consumes a
complete wrap32 `[batch, vocabulary]` tensor and compares signed i32 values with
first-index ties. That selector is **not** the float32 selector certified here:
different row scales or bias can reverse integer ordering. No silent substitution
is valid. `decoder_continuation.rs:186-189` explicitly retains every logits row
for continuation/prefix checkpoints, including intermediate boundaries.

The float32 client paths also demand real logits: `conversation_checkpoint.py`
validates and copies a vocabulary-sized finite float32 `next_logits` vector;
`public_prefix.py` validates and publishes it; `tiered_execution.py` stores and
restores it with history and retained state. Greedy verification reference paths
retain score rows for accepted/rejected decode boundaries. Argmax-only parity is
not identical execution, snapshot interchangeability, or general sampling.

For an unmodified runtime, **every demanded head remains full**, so this idea is
currently research-only overhead. A hypothetical admitted mode must explicitly
restrict to deterministic greedy, no sampling, no logits/logprob disclosure or
score processors, no speculative score verification, and interior states not
being published/restored through today's complete-logit contracts. It must prove
token feedback and every KV/pending/history state unchanged. First/prefill,
final/stop/cancel publication, requested output logits, conversation/prefix/cache
boundaries, and cached-state export/import require full-head materialization.
If prefixes are published at every decode boundary, there are no admitted
argmax-only interior steps. Existing cached complete logits can supply a token
without a new head; a certificate must not invent a cache saving there.

Lazy logits are only a design possibility, not an implemented bypass. A future
typed deferred-logit state would retain exact final hidden/A8 query, activation
scale, immutable head/scale/bias lineage, arithmetic contract, generation/history
and pending-token identity. Resolving it must reproduce all score bits under
that contract, with no rerun under a changed source. Every consumer that currently
requires an ndarray must resolve before snapshot/publication/verification. An
unresolved thunk is not interchangeable with an existing complete-logit snapshot.
Charge retained hidden bytes (`D` for A8 plus scale and metadata, possibly `4D`
for a separate float32 representation), source leases, thunk bookkeeping and
lifetime, all later full-head work and logits allocation. If source/hidden cannot
be retained safely, re-execute the required body and charge that too. Deferral
may save nothing, or increase memory and CPU.

## Cost ledger and Low/Medium budgets

For `V` rows, `D` columns, `B` tested bound nodes, and `S` exact tested rows:

* Cold delivery/publication: full original head `VD + 4V` bytes, optional bias
  `4V`, all public metadata/index bytes, source authentication, public index
  construction/validation, and index publication/delivery. Count all provider
  links and storage copies; the full original artifact is never optional.
* Index payload: `52 + nodes * (2D + 28)` bytes in this reference. Each visited
  node reads `2D + 28` bytes of metadata, performs `2D` integer endpoint products
  and four dequants, plus traversal/selection work. Query magnitude checking
  visits `D` components; scale/bias domain checks scan all `V` rows. The current
  bias absolute-value scan creates a `4V` temporary and reads `12V` source bytes
  across scale/max/absolute-bias reductions. Per-node query int64 conversion and
  endpoint selection temporaries are additional allocations.
* Exact work: `SD` dot MACs and weight bytes, exact row scales/bias, conversion,
  dequantization, comparison, and local page faults. Real paging reads full
  storage pages, not only these row payloads: count unique and repeated pages,
  checksums, read buffers, alignment and physical I/O. Resident mode still charges
  all original weights and memory bandwidth. This probe reports row payloads,
  not measured page traffic.
* Failed budget: prior work plus `VD` full-head MACs and all full-head reads.
  Materializing demanded logits similarly costs all missing rows (or `VD` if
  recomputed), `4V` logits bytes, and vector assembly. Exact-tested rows can be
  reused only if their logits were retained and charged; this probe does not
  retain a complete partial-logit vector. No pruned logit is known exactly.
* Test oracle: another `VD` dot MACs/full weight scan and `4V` logits, plus `8VD`
  int64 weight expansion, dot scratch and comparisons. The evidence explicitly
  reports oracle work, not a free verifier. Native promotion must price whatever
  production verification its security contract requires separately.

Low's 256 MiB and Medium's 2 GiB are **whole-client** caps: interpreter/native
runtime, caches/KV/history, source snapshots, original weights or local backing
storage, index, pending query, page buffers, verification, temporary arrays,
serialization, allocator/page-cache behavior, and deferred logits all count.
Low could use a small hierarchy and genuinely local paged original weights;
Medium could afford deeper metadata or resident weights. Deeper boxes approach
`O(VD / leaf_rows)` metadata and may exceed any benefit. This resident Python
screen cannot qualify either tier, a storage policy, or SOTA.

## Replay and exact next gate

Bounded tiny replay (fresh worker, shared serial lock, reserve/headroom/swap and
installed-build checks from `frontier_probe_support.fresh_workers`; no timing):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath='' tests/test_head_certificate_index_probe.py
target/frontier-sdk-env/bin/python scripts/probe_head_certificate_index.py --output docs/evidence/head-certificate-index-tiny.json
```

Optional next **public weight-only** screen, not run here. Export an already
authenticated native W8 head as row-major raw signed-i8 plus little-endian f32
scales; provide known public file hashes and geometry. Do not substitute original
float checkpoint weights for native quantized rows. Bias is unsupported by this
raw CLI (but supported and tested by the in-memory certificate API).

```sh
target/frontier-sdk-env/bin/python scripts/probe_head_certificate_index.py --weights /absolute/public-head.i8 --scales /absolute/public-head-scales.f32 --weights-sha256 KNOWN_WEIGHT_SHA256 --scales-sha256 KNOWN_SCALE_SHA256 --rows 151936 --columns 896 --scan-rows 128 --output docs/evidence/head-certificate-index-public-prefix.json
```

Both complete raw files are streamed for hash admission and charged as public
delivery/verification. Only the bounded prefix is indexed, using synthetic
queries. Its winner and prune counts apply **only to that prefix**, never the
global vocabulary, real activations, full logits, or native engine. No model
downloads, checkpoint loader, hidden extraction, or response generation occurs.

Next gate belongs to the parent-run serial measurement: first reject this box
family if bounded authentic public row screens show no pruning after **all**
bound, scan, source, and index costs. Only if that survives, formulate explicit
token-only interior demand semantics and prove native arithmetic/tie and full
materialization/state equivalence. Then measure full-vocabulary admitted queries
with exact independent skip oracles, fallback rates, charged cold delivery,
actual local page traffic/CPU and whole-client peak memory under both budgets.
Keep first/final/cached-state full-head controls in the same cohort. There is no
current permission to run that measurement or promote a runtime path here.
