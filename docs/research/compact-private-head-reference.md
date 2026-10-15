# Compact private full head: source-certified scanner and selection gates

## Decision

**Implemented:** bounded two-pass scanner of actual public signed-i8 head rows,
scales and pinned metadata; independent Python-integer verification of **every
row**, optionally checked against installed native width implementation. This is
not another geometry-only width projection. Checkpoint mode invokes the ordinary
head-only importer after checking original checkpoint/config against a retained
independent lock. Raw mode requires an externally authenticated source pin.

**Pinned checkpoint gate completed:** two independent passes over all 151,936
Qwen2.5-0.5B head rows find mostly 23-bit replies, versus the installed uniform
25-bit layout. Both payloads fall 949,600 to 870,724 bytes (**8.31%**). Separate
installed-native reconstruction checks preserve 759,680 endpoint/random integers.
This is a source-bound codec gate, not an SDK implementation or measured speedup.

**Complete alternative implemented as a bounded numerical reference:** download
public head once, evaluate full head privately on client, then call unchanged
ordinary sampler. Preserves useful full-vocabulary generation, all float32 logits,
greedy ties, RNG state and arbitrary existing top-p settings. It trades roughly
136 MB extra client artifact and 136 million integer MACs/token for no head network
exchange at Qwen geometry. No new SDK integration, memory-tier or speed claim.

**Remote private selection gate fails.** The explicit two-boundary half-gates
composition already incurs about 180 MB/head of reconstruction tables at a
*hypothetical* 20-bit width, before OT, float32 dequantization, argmax or sampling.
No pinned NumPy-equivalent private full top-p circuit exists here. Greedy is a
separate failed implementation/cost gate, not silently substituted for sampling.

## Completed pinned source gate

The guarded fresh-process scan uses the retained checkpoint/source lock and
64-row chunks. It checks every row independently twice, including actual signed-i8
weight bytes and float32 scales. Widths are:

| Bits | Rows |
| --- | ---: |
| 21 | 1 |
| 22 | 11,794 |
| 23 | 139,982 |
| 24 | 159 |

There are no zero-bound rows. The selected public zlib layout occupies 9,951 bytes;
three recipients' complete modeled layout/metadata delivery totals 32,934 bytes.
After the modeled reply context and framing, both replies save 78,812 bytes/head,
or **597,562 net bytes over eight heads** with that setup charge. The corresponding
40 Mbps per-worker upload-floor difference is **7.88 ms/head**, not measured
request latency. Original requests and complete reconstructed logits remain
required. Native audit checks zero, both endpoints, alternating endpoints and
public-seed random values in every row using fresh arithmetic shares.

That favorable setup assumes independently trusted, source-bound public layout
metadata. Re-fetching the public weights and scales for client-side certificate
verification instead adds **136,742,400 bytes** and moves the estimated break-even
to **1,736 head calls**. The live thin-client verification, source/schedule binding,
codec admission, cancellation and full decoder/sampler parity gates remain open.
This cannot be promoted by trusting a worker's self-declared bound.

The explicit half-gates reconstruction bill rises to **208,319,168 bytes/head**
at these actual heterogeneous widths, before input labels, OT, dequantization or
selection. Thus neither this examined remote-selection construction nor the
client-local full-head alternative earns a cold-request or client-tier win.

Original evidence:

- [Pinned scan](../evidence/compact-private-head-qwen25-v1.json)
- [Native reconstruction audit](../evidence/compact-private-head-native-audit-v2.json)
- [Admitted scan Experiment](../evidence/compact-private-head-experiment-v1.json)

Read-only native replay, without loading checkpoint tensors:

```sh
target/frontier-sdk-env/bin/python scripts/audit_compact_private_head.py \
  docs/evidence/compact-private-head-qwen25-v1.json \
  --output target/compact-private-head-native-audit-replay.json
```

## Exact reconstruction floor: design-specific, not universal

For row `w`, signed-i8 weights including `-128`, and symmetric inputs
`x_j in [-127,127]`, define

```
B = 127 * sum_j abs(int(w_j))
z = sum_j int(w_j) * int(x_j)
b = max(1, bit_length(B) + 1)
```

The integer bound is tight: choose `x_j = 127*sign(w_j)` to attain `+B`, opposite
signs to attain `-B`. Adding both low-b-bit output shares modulo `2**b`, then
signed-centering, recovers every valid dot exactly iff `B < 2**(b-1)`.
One fewer bit necessarily fails on the attainable positive endpoint. This proves
minimal width for **this per-row low-residue/signed-center decoder**, including
rows whose attainable integer sets have gaps. It does not assert all integers
inside the interval are attainable.

With native one-bit minimum for zero rows, minimal fixed-layout reply payload is
`ceil(sum(b_i)/8)` **per worker**, twice that for both replies. Known zero rows
could be omitted by a changed codec; scanner explicitly counts them but does not
claim native support for zero-bit rows. The native signed-i8 result and scalar
oracle independently agree; scale values do not reduce the integer certificate.
No bias is admitted. Source activation range and quantizer are pinned, not inferred
from a handful of observed hidden states.

This is **not an information-theoretic lower bound** on full-head delivery or
private token selection. Rows depend on the same hidden vector; output shares can
have linear correlations. Alternate arithmetic, public bases, lattice-aware
decoders, public-source client computation and secure selection change the
construction. Client already owns the hidden state: with public weights resident,
its head-reply communication lower bound is zero. Costs of making that public
artifact usable cannot disappear from cold accounting.

## What the scanner verifies and prices

`scripts/probe_compact_private_head.py` owns this research implementation.

* Admission before row allocation: at most 262,144 rows, 65,532 columns,
  256 MiB signed weights, 256 chunk rows, 4 MiB row chunk; require no wrap32 overflow.
* Pass one: bounded vectorized int16 absolute values / uint64 row reductions.
  Pass two: Python integers and explicit signed-byte decoding, every coordinate.
  Hash actual weights **and scales on both passes** against pin; validate every
  scale finite, positive float32. Optional native checker runs on every chunk.
* Exact metadata binds source lock, checkpoint, config, descriptor, weight/scales
  hashes, quantizer, geometry, bounds and width hashes. Returned full little-endian
  bounds and widths permit read-only independent replay; hex report is storage,
  not proposed network serialization.
* Layout choices: tagged raw, packed five-bit widths, u32 run lengths,
  zlib-9, and scalar uniform when applicable. All roundtrip exactly. Decoder bounds
  decompression/expansion and rejects padding, trailing streams and invalid widths.
  Metadata costs compare tagged canonical JSON against tagged zlib-9 JSON.
* Setup model includes a 88-byte public envelope header, selected metadata and
  layout, delivered to **both workers and client**, with per-recipient transport
  allowance. Candidate replies charge existing 74-byte codec header **plus 32 bytes
  for layout binding**, not headerless optimistic output. Session/epoch/role/
  sequence binding, authenticated transport and original one-use retirement remain
  mandatory for any SDK implementation. No new serving protocol is installed here.
* Framing assumptions keep observed 142-byte outer body overhead and 137-byte HTTP
  stream overhead per reply. These reproduce incumbent 475,153 bytes/worker at
  Qwen 25 bits, but are **counterfactual estimates**, not measured new frames.
  MessagePack integer length changes, TLS, packets, source publication/authentication
  and different setup endpoints require fresh accounting. No analytical body
  estimate is called measured wire traffic.
* Report separate setup-only break-even and client-source-verification break-even.
  If client lacks trusted certificate issuer/provenance, it must fetch public weights
  and scales and recompute: another `V*(H+4)` bytes, explicitly priced. A digest
  from the same untrusted peer is not authentication. Existing workers may already
  own source weights; do not charge their already-present matrix twice.
* `fresh_workers` supplies fail-fast shared frontier lock, offline environment,
  post-reserve headroom, zero-swap/pressure watchdog, timeout, fresh subprocess and
  installed-build stability. Source locks cover driver, guard, native library,
  relevant installed loader/quantizer/client modules and NumPy/RNG binaries.
  Report output is exclusive; failed admission leaves an `incomplete` attempt.

Default reservation is **2 GiB**, timeout **240 s**, four compute threads from
shared runner. Checkpoint importer has explicit modeled allowance `4*V*H+512 MiB`
within reservation; raw mode prices mapped source plus `64*V+128 MiB`. Scanner
scratch allowance is `8*C*H+12*V` bytes, separate from importer/native owners,
Python objects, source hashing, report JSON and allocator. Two certificate passes
visit `2*V*H` coordinates; checkpoint source admission additionally streams original
checkpoint hash and importer performs its ordinary quantization/compilation.
These reservations are not measured peak RSS or a serving-tier qualification.

## Complete private local full-head alternative

Construction:

1. Derive canonical bias-free signed-i8 head and positive row scales using ordinary
   pinned source importer; publish/authenticate public head artifact once. This
   publication/export does not yet exist as a production path; preparation CPU,
   storage and distribution must be added to an SDK comparison.
2. Before request-dependent use, download **whole** public artifact (never
   query-dependent rows). Authenticate weight/scales hashes and source/quantizer.
   Store raw row-major weights locally; all reads below are private local reads.
3. Quantize final hidden row using original symmetric quantizer/scale. Reference
   `local_head` scans every public row in fixed chunks, computes exact int64 dot,
   casts to float32, multiplies by activation scale, then by row scale as separate
   float32 operations. Geometry proves integer result cannot wrap32. No fused
   multiply, float64 shortcut, scale cancellation or approximate head is used.
4. Feed complete float32 vector into unchanged
   `MaskedTransformerClientRuntime.sample`, on unchanged runtime/RNG object.
   Existing runtime stores ordinary logits and KV, records selected token and
   retires consumed preparation normally; only head RPC is removed. Reference
   tests directly invoke this sampler and compare full RNG states. Integration
   and complete state audit remain parent work, not accomplished by scalar tests.
5. Existing private lookup for selected token remains unchanged. Removing head
   exchange does not remove lookup traffic, other prepared stages or decoder KV.

For `V=151936,H=896`:

| Increment / removal | Exact payload or operation count |
| --- | ---: |
| Public weights | 136,134,656 bytes (129.83 MiB) |
| Public row scales | 607,744 bytes |
| Whole public head delivery if neither present | 136,742,400 bytes |
| Additional weights when client already owns scales | 136,134,656 bytes |
| Integer MACs per head | 136,134,656 |
| Float32 casts / multiplies per head | 151,936 / 303,872 |
| Complete float32 logits | 607,744 bytes |
| Conservative top-p numeric-array allowance | `128*V` = 19,447,808 bytes |
| 64-row int64 dot scratch allowance | `8*64*H+8*H+32*64` = 467,968 bytes |
| New protocol messages / head one-use material | 0 / 0 |
| Removed existing head request payload | `2*H*4` = 7,168 bytes/head |
| Removed incumbent framed head replies | 950,306 bytes/head, observed-overhead model |

Assuming scales already present and charging 279 extra artifact framing bytes,
weight delivery breaks even at **143 heads**, versus incumbent 25-bit replies
and unchanged request payload. This counts both replies, not just one. Artifact
download alone has **10.891 s** client-100-Mbps capacity floor; publisher upload
may be slower. Eight outputs save only about 7.66 MB of old head exchange and
do not amortize 136 MB cold delivery. Persistent artifact reuse changes that
calculation; each fresh client pays unless durable authenticated storage is
explicitly part of the contract. If weights/scales missing, add both. Publishing
quantized artifact absent today cannot be replaced by silently uncharged full
checkpoint download.

Conservatively adding these weights to incumbent ~229 MiB whole client would
exceed Low, not demonstrate Low admission. Bounded mmap/windowed private local
storage may avoid full residency, but I/O/page-cache/RSS then needs measurement;
no tier inherited. Medium construction is plausible, **not admitted**. CPU of
Python/int64 reference is not a competitive native-kernel measurement. Necessary
next gate: admitted local native head, full original sampler/logit/KV commitments,
cold source traffic, Preparation CPU, whole-client CPU/RSS and complete requests.

## Why remote exact argmax/top-p is not declared solved

For a concrete Yao/free-XOR/half-gates design between noncolluding boundary A/B:
privately input their b-bit residues; reconstruct modulo `2**b` inside circuit;
privately input client's activation scale; reproduce original int-to-float32,
two separately rounded multiplies; output only first-index float32 argmax to
client. Evaluator must not learn decoded token; output decoding belongs to client.
Even the modular ripple-adder stage uses `max(0,2*b-3)` AND gates per row:
one low carry AND, two ANDs for each interior carry, final sum XOR. Half-gates
need two 128-bit ciphertexts/AND, **32 bytes**. At hypothetical b=20:

* 5,621,632 reconstruction ANDs; **179,892,224** garbled-table bytes/head.
* 48,619,520 bytes garbler selected input labels, plus 3,038,720 evaluator-input
  OTs. OT extension/base setup, corrections and labels are additional, not zero.
* Fresh labels/tables/randomness per use; streaming lowers residency, not traffic.
* Float32 arithmetic, comparisons/ties, private client input/output, authentication,
  peer links, CPU and malicious-security strengthening remain extra.

Thus this particular composition fails the all-directed-head-traffic gate before
unfinished stages. It is **not a universal cryptographic lower bound**. A newer
construction can beat this composition; it must supply its complete executable
contract and costs. No claim that all MPC sampling is infeasible.

Full top-p has stricter oracle than greedy. Current source casts logits to float64,
divides by temperature, subtracts max, exponentiates, normalizes with NumPy sum;
then default `np.argsort(probabilities)[::-1]`, cumsum in that order,
`keep = cumulative <= max(0,p)`, forced `keep[0]=True`, zeros excluded entries,
renormalizes, and calls original `rng.choice` on full vocabulary. This **excludes**
the first crossing item unless it is first; do not replace with a conventional
"include crossing" implementation. Tied descending order is NumPy default sort
behavior, not greedy's first index. Sampling state includes exact bit generator
and how many random bits original choice consumes; matching distribution alone
does not match token/RNG commitments.

Private circuit would therefore need pinned float64 exp/reduction behavior,
NumPy sorting (including ties), normalization, choice's CDF/search and private
RNG-state update, plus float32 dequantization. IEEE arithmetic support alone
does not establish bit identity with pinned NumPy/libm implementations. No complete
circuit/cost is supplied: **fail exact full-sampler gate**. Fixed top-k, Gumbel-max,
approximate exp, reduced precision or different RNG are changed contracts requiring
independent quality and semantic gates; none inherits original generation.

Primary literature inspected: [Zahur–Rosulek–Evans, Two Halves Make a Whole,
EUROCRYPT 2015](https://eprint.iacr.org/2014/756), for two-ciphertext AND/free-XOR;
[Pullonen–Siim, Combining Secret Sharing and Garbled Circuits for Efficient
Private IEEE 754 Floating-Point Computations](https://eprint.iacr.org/2014/990),
for private IEEE arithmetic feasibility, not NumPy sampler equivalence.

## Unsafe shortcuts: concrete failures

* With attainable `z=+128`, eight-bit signed-center yields `-128`, even if all
  observed calibration logits were small. Width based on private values or
  observed maxima is not a source certificate and can disclose activation ranges.
* Divide shares before reconstruction: `(1//2)+(1//2)=0`, while `(1+1)//2=1`.
  Public even weights do not license dropping carry bits from arbitrary existing
  shares. A new exact lattice/ring construction needs its own proof.
* Arbitrary modulus is not a ring32 homomorphism: in ring8, shares 255 and 2
  reconstruct 1, but separately reducing mod251 yields `(4+2)%251=6`.
  Power-of-two low residues work because new modulus divides original ring.
* Even zero dot can have mask shares 127 and 129 in ring8. Do not range-compress
  individual shares as though each were a small signed dot. Entropy-compressing
  fresh mask-like bytes generally offers no promised gain; reuse of masks leaks
  differences. Output share correlations require explicit secure construction.
* Common positive activation scale cannot simply cancel before float comparison:
  float32 `(array([1,2])*1e-20)*1e-30` becomes two zeros; first-index greedy picks
  0, integer comparison picks 1. Different row scales can also reverse integer rank.
* Five equal logits assign positive probability to all five in full sampling;
  fixed top-2 destroys 3/5 of original mass. Exact ordinary top-p cannot be replaced
  by public/fetched candidate sets. Query-dependent source pages or shortlist IDs
  sent to provider leak private behavior; local-only access after full public
  download is different.
* Certificate checksum or self-declared source hash is not authentication against
  malicious peer. Source admission, distinct roles, fresh masks and replay/failure
  retirement remain required; no new malicious-security claim.

## Evidence, tests, and parent serial commands

Tiny evidence: `docs/evidence/compact-private-head-synthetic-v1.json` (17x32 public
fixture, no real checkpoint, no timing), SHA-256
`8e94ae195e3f1549d1d4301039670feb241c414ecb88472037d2ce71dbc27f27`.
Includes actual fixture certificates/native verification and separately labeled
Qwen **geometry-only** arithmetic; it contains no actual Qwen row histogram.

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. \
  tests/test_compact_private_head_probe.py
```

25 focused tests pass. Coverage includes -128, zero rows, tight endpoints, two-pass
mutation, invalid hashes/dtypes/scales/quantizer, native disagreement, all metadata
layout codecs, decompression/run/padding guards, both-role framing/cold accounting,
full local float32 logits, original greedy/top-p selections and RNG state, and
unsafe compression counterexamples. No sustained timing or checkpoint execution.

Parent checkpoint mode (run **serially**, fresh output; no separate lock around
this driver, which acquires shared frontier lock itself):

```sh
# Save actual admitted current Experiment.to_spec() as this input JSON.
# Preserve independent checkpoint lock below; do not generate pins from scanned data.
target/frontier-sdk-env/bin/python scripts/probe_compact_private_head.py \
  --experiment target/compact-private-head-experiment-v1.json \
  --checkpoint-lock docs/evidence/orthogonal-frontier-qwen25-source-lock.json \
  --chunk-rows 64 --head-calls 8 --admission-mib 2048 \
  --output target/compact-private-head-qwen25-v1.json
```

Experiment must explicitly select installed W8A8 `PrivateLookupAndFullHead`
configuration (parallel/uniform residues for incumbent identity), local children,
and original tied checkpoint. Use currently admitted Experiment recipe, not a
historical report's performance samples. Driver forces offline source access.
Current checkpoint mode supports retained single-file `model.safetensors` lock;
other checkpoint layouts fail explicitly rather than silently weakening admission.

Raw mode allows parent-exported actual public i8 rows and little-endian f32 scales:

```sh
target/frontier-sdk-env/bin/python scripts/probe_compact_private_head.py \
  --raw-manifest target/compact-private-head-pinned-raw-v1.json \
  --chunk-rows 64 --head-calls 256 --admission-mib 512 \
  --output target/compact-private-head-qwen25-raw-v1.json
```

Manifest exactly `{ "weights": "/absolute/head.i8", "scales": "/absolute/scales.f32",
"pin": {...} }`. Pin exactly: `source_lock_sha256`, `checkpoint_sha256`,
`model_config_sha256`, `descriptor_sha256`, `weight_sha256`, `scales_sha256`,
`vocabulary`, `hidden`, `qmax:127`, `weight_dtype:"i8"`,
`activation_quantizer:"symmetric-per-row-i8-qmax127-v1"`. Use independent admitted
descriptor/source commitments, not hashes newly asserted by untrusted artifact
supplier. Raw files must contain entire matrix/scales with exact sizes, no headers.

After parent scan: independently recompute histogram/bit sum and metadata
compression; compare two-worker application bodies against actual current codec;
choose promotion only if fresh setup/repeated-output cost justifies a new layout.
The preceding pacing reference's 25→20-bit hypothetical saves only **18.992 ms**
worker-upload capacity floor per head. Existing paired exchange 109.736 ms includes
92.684 ms overlapping pacing sleeps. These are not whole-request speed gains.
Then, and only then, consider parent-owned SDK/native variable-layout integration
and a fresh complete state/resource/traffic cohort.
