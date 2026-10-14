# Lossless quantization-origin KV storage

## Verdict

**Feasible as a restricted provenance codec, not yet a whole-client optimization or
SOTA candidate. No SDK/native component promotion.** Integer projection outputs
can be packed losslessly and replayed into exactly the existing FP32 cache bits
when the original scales, bias, normalization, rotary coefficients and numeric
operation order are retained. This is not approximate KV quantization. It does
not recover missing provenance from an existing FP32 snapshot.

The executable reference is `scripts/probe_projection_kv.py`; independent tests
are `tests/test_projection_kv_probe.py`. The bounded synthetic evidence is
[`projection-kv-synthetic.json`](../evidence/projection-kv-synthetic.json).
There were no checkpoint scans, timing cohorts, builds, full responses or installs.
The installed package was used, not the workspace Python package. Quantization,
semantic executor and rotary source hashes match the workspace files inspected.
The record contains resolved installed paths, native extension hash, NumPy binary
hash, Python/platform identity, probe hash and the captured `BuildRef`.

## Actual boundaries and exact arithmetic

* `python/pllm/runtime/quantization.py:179`: cast integer to FP32, multiply by the
  original activation-row scale, then multiply by the original output-row weight
  scale. Parenthesization is `(float32(integer) * activation_scale) * weight_scale`,
  not `integer * (activation_scale * weight_scale)`. A separate bias addition
  follows. Integer values above 2^24 are intentionally rounded by the original
  cast; retaining those integers still reconstructs the original float bits.
* `model_binding.py:185` shows the clear-native projection boundary;
  `transformer_client.py:1196` shows the ordinary remote unmask/dequantize boundary.
  Neither currently returns projection provenance to the semantic state owner.
* `PreparedRemoteLinear._decode_reply` uses `prepared_unpack_output` to expose
  centered i64 outputs, or `prepared_dequantize_output` for the fused unverified
  path. The latter discards the integers. In `transformer_client.py:1408`, the
  one-use verifier receives integer outputs **before** dequantization. Any future
  capture must occur after successful verification and correlation retirement,
  not by storing untrusted decoder results and calling them verified. The toy
  `verified=True` argument is a test precondition, not an implementation of a
  verifier or remote protocol.
* `crates/pllm-core/src/offset_transport.rs:59` computes public output widths from
  weight-row L1 norms and an absolute quantized-input bound. Native packing uses
  LSB-first two's-complement residues, canonical zero tail padding, rows 1..4096,
  at most 4,000,000 output elements and widths 1..32. Reconstruction is centered
  and returns i64 bytes. These are projection output-column bounds (weight rows),
  not activation-row-specific observed widths. The fused native dequantizer at
  line 319 preserves the same separate cast/multiply/multiply/bias steps.
* `semantic_executor.py:819` gives the FP32 RMS expression:
  `x / sqrt(mean(x*x, axis=-1, keepdims=True) + epsilon) * (weight + offset)`.
  For Qwen3, per-head Q/K normalization follows projection/reshape and precedes
  RoPE. The reference replays the original K norm operator; Q remains unchanged.
  Qwen2 commonly has Q/K/V projection biases and no Q/K norm; ordinary Llama has
  no Q/K norm and commonly no projection bias. These are fixture cases, not
  universal architecture defaults: inspect the compiled graph and bound config.
  Fused projection slicing, partial rotary width and actual bias/norm declarations
  must be bound explicitly for any admitted checkpoint.
* `semantic_executor.py:561` gives the original rotary dispatch. Ordinary FP32
  RoPE copies the input, concatenates `[-second_half, first_half]`, then evaluates
  `current*cos + rotated*sin` as separate FP32 operations. **No inversion.**
  `rotary_coefficients.py:27` explicitly warns that a different array shape can
  select a different NumPy SIMD/scalar libm path. The reference captures the
  original phase's cosine/sine bits and slices those tables, never regenerating
  coefficients on a smaller page. Storing positions and theta alone is insufficient
  without independently proving exact coefficient replay for the installed build.
* Semantic cache append stores post-RoPE K and post-projection V as FP32
  token/head/feature arrays. Original scores/values use per-query `np.einsum`,
  with the original scaling and softmax order. A codec must restore the original
  active sequence, head layout, strides/contiguity and numeric contract before
  those operations. Computing attention page by page or changing softmax/reduction
  order is a separate algorithm and is not justified by exact codec bits.

The reference admits only finite FP32 projection metadata, positive finite FP32
scales, finite bias/norm coefficients and monotonic nonnegative absolute int64
positions. It rejects integer outputs outside `[-qmax*L1, qmax*L1]`, signed wrap32
overflow, invalid lengths, noncanonical tails and restored nonfinite projections.
It keeps exact scales rather than recomputing them. Overflow or subnormal-scale
failures are not silently accepted. Normalization/rotary output must remain finite.
It does not admit BF16 rounding graphs, scaled/wavelength/per-frequency RoPE,
mRoPE, sliding/recurrent/retained/shared KV, protected nonlinear execution or
arbitrary state snapshots. Those need their own exact graph replays.

## What the executable proves (and does not)

Public reference caps: 64 rows, 4096 columns, 262144 elements per page/profile;
the weight matrix used to derive widths has the same element cap. Profile copies
are immutable bytes; coefficient arrays are read-only copies. Native parsing
checks canonical payload tails. Python validates declared sizes before hashing
or native decode. An unkeyed SHA-256 page commitment catches accidental changes;
it is **not authentication**, rollback protection or a persistent state manifest.
Trusted profiles must be bound externally to source/stage/layer/role. An attacker
can replace an unkeyed commitment; the tests deliberately do so to check independent
range and metadata validation. No archive parser, page store, LRU, remote transport,
transactional publish or cancellation owner is implemented here.

Sixteen independent tests passed against `target/frontier-sdk-env/bin/python`:
public-L1 endpoints/zero rows, an independently constructed bitstream, odd terminal
padding, malformed/truncated pages, unverified capture, out-of-range integers,
invalid dtype/scale/position/row count, restored invalid metadata, changed source
identity, rotary-domain binding and actual native fused dequantization above 2^24.
The tiny family fixtures use independently evaluated original projection/norm/
RoPE state, restore K in 3/3/1-row slabs, and compare uint32 bits for K, V and the
**actual installed original semantic attention score/scale/softmax/value output**.
These are synthetic architecture-style fixtures, not full compiled checkpoint
logits. One unmasked single-query attention consumes all seven prior keys; no
claim is made for causal prefill masks or complete decoder outputs.

For each fixture dense K/V is 896 bytes. Packed page payload plus duplicate
positions, scales and commitments is 760–762 bytes, but original rotary tables
add **448 bytes**, making 1208–1210 bytes before profiles, norms, Python objects,
page directory or authentication. This tiny example is a storage regression,
not a favorable compression demonstration. Larger layers can share identical
original tables by an exact phase/domain key, but that sharing must be real,
bound and priced rather than presumed.

## Storage and whole-client admission

For each public output column j, `B_j = qmax * sum(abs(weight_row_j))` and
`b_j = max(1, bit_length(B_j)+1)`. Require `B_j < 2^31` and verify every captured
integer. Per-role page payload is `ceil(rows * sum(b_j) / 8)`. Scales cost four
bytes per activation row **per distinct quantization event**. K and V scales may
share only when their original scale bits/events match. Absolute positions cost
eight bytes per row; pages can refer to one authenticated position vector rather
than duplicating it. Original coefficient tables cost `8*T*rotary_dim` bytes per
distinct original table domain (cos and sin FP32). Projection profiles cost at
least widths + original row scales + bounds + optional bias: 13 or 17 bytes per
column in a compact fixed-width representation; this Python tuple representation
has substantially larger object overhead. Norm weights, offsets, epsilon,
projection slices, phase IDs, layer/role, tensor layout, source/numeric digest,
page offsets/lengths and authentication tags/manifests are additional owners.

Illustrative worst-case uniform precision, **not checkpoint observations**:
W8A8 with input width 896 yields B=14,451,584 and 25 bits; width 2560 yields
B=41,290,240 and 27 bits. Projection payload alone is 78.1%/84.4% of FP32.
W4A4 at width 2560 yields B=125,440 and 18 bits (56.25%). Weight-L1 widths may
be smaller than these worst cases; no private observed width is needed. Widths
can reach 32, at which point scale/coefficient overhead makes storage worse.

**Low = 256 MiB and Medium = 2 GiB refer to the whole client, not this payload.**
Admission must cover:

1. Python/native imports, model/graph/tokenizer/head and all resident immutable
   weights, scales, bias, norms, prepared inventory and protocol buffers.
2. Entire packed backing at public worst-case size, index/manifest, position and
   coefficient backing, codec buffers and page authentication scratch. Disk/mmap
   backing still requires explicit resident-page/cache policy; filesystem cache
   and client RSS are different scopes.
3. Original floating state during conversion. Existing snapshots cannot be
   converted by inverse norm/RoPE or guessing integers. Safe migration requires
   retained origin records or re-execution; that original-state + packed-state +
   decoder-output + slab overlap is charged before any original array is retired.
4. Incoming raw/masked outputs, masks/corrections and centered i64 results;
   verifier inputs/returned copies; FP32 projections and bias temporary; original
   norm/RoPE input/output/scratch; original coefficient arrays and captured copies.
   Fused decode avoids i64 but currently removes the origin capture opportunity.
5. One **full active layer** FP32 K/V slab: `8*T*kv_heads*head_dim` bytes. Small
   codec pages do not make original attention consume a small slab. Existing
   tiered decoder uses at most a 128 MiB layer slab but additionally allocates
   attention scratch and caller copies; see the tiered reference. Restoring
   directly into a correctly laid-out destination could avoid a page-list/
   concatenate duplicate, but this tiny probe deliberately uses both and does
   not claim a production bound. Scores/probabilities and decoder live values
   remain charged at original sequence bounds.
6. Medium's decoded cache must have an explicit byte cap, root-aware alias
   accounting and eviction before another slab is admitted. Restore charges old
   cache + incoming packed bytes + authenticated plaintext + new slab + copy
   overlap. Medium cannot opportunistically retain whatever compressed data fits.

The Python page decoder allocates a zero mask (4 bytes/element), native i64 output
(8 bytes/element), FP32 cast/result/scale intermediates and bias output. Capture
allocates uint32 conversion bytes, packed bytes, metadata bytes and SHA concatenation
copies. Coefficient capture copies two tables. A byte upper bound on the logical
page payload is not a process-RSS ceiling or a native fused storage implementation.

Client-state evidence already reports Qwen2.5-0.5B complete-response paged-client
peak **306.43 decimal MB**, above Low even on a short prompt, and conservative
Qwen3-4B 64+8 estimate **1.07 GiB**; see
[`client-runtime-memory-2026-10-05.md`](../evidence/client-runtime-memory-2026-10-05.md).
The tiered-decoder reference demonstrates 24x slab ownership reduction but slightly
*higher* whole-process peak RSS. Neither source supports turning this codec's
payload fraction into a whole-client Low/Medium qualification.

Private-value local entropy/bit-width optimization could be added only behind a
fixed public worst-case admission and a client-local page format. No provider-visible
message length, width choice, cache policy feedback or data-dependent retry may
depend on private outputs. This reference uses only public weight-derived widths.
Provider protocol, stages, integer body MAC count and requests must remain unchanged.
Consequently storage itself need not move any body MAC local, but this toy proves
no remote-MAC ratio. Any complete gate must independently establish **>=90% remote
body MACs**. Dequantization, norm, RoPE, coefficient capture, packing, authentication,
disk reads and attention replay are local CPU work and must be measured/charged.

## Retirement and future checkpoint replay gates

Before integration: capture only verified, consumed projection results; retire
correlations on failure using the original protocol owner. Publish no partial KV
generation. Cancellation/exception must release the current slab, temporary
integers, coefficient copies and decoded cache and preserve only the previously
accepted authenticated generation. Failure after processing one role must not
leave an independently accepted K or V. Old manifests/orphan files, restoration
overlap and crash recovery need the same bounded transaction discipline as
`tiered-decoder-reference.md`. Immutable Python bytes cannot be reliably erased;
reference object release is not secure wiping. Ordinary snapshot APIs must reject
an incompatible compressed owner rather than bypassing its provenance contract.

Future checkpoint investigation must predeclare a source-locked fixture: exact
checkpoint revision and tensor/config digests, compiler/semantic-plan and numeric
configuration digests, installed `BuildRef`, NumPy/native implementation identity,
original stage weight/scales/bias/norm hashes, public positions/token fixture,
original rotary table bits and complete graph layout. Retain integer provenance
from the exact original projection route, including verification, not a separately
re-quantized approximation. Check each layer's reconstructed uint32 K/V and final
logits across prefill, append, release and fresh restore. Scans must reject invalid
ranges/tails before accepting records and use predetermined public resource caps.

Only after those correctness checks should the parent run serial fresh-process
cohorts with `scripts/frontier_probe_support.py:fresh_workers`, shared lock,
host-reserve/swap admission and `BuildRef` consistency checks. Separately measure
baseline, origin capture, packed execution, cache execution and fresh restoration;
include conversion peak, process-lifetime peak RSS, disk bytes, local CPU/elapsed
and exact remote-body-MAC accounting. No checkpoint scan CLI is supplied: the
current runtime lacks a sanctioned origin-capture interface, so a purported scan
command would misrepresent unsupported checkpoint replay. CPU and checkpoint
width distributions remain unknown. SOTA ranking and both tier gates remain open.

## Exact modest replay

Output must not already exist. This runs only the tiny correctness fixtures and
metadata capture; it is not a measurement cohort or checkpoint response.

```sh
target/frontier-sdk-env/bin/python scripts/probe_projection_kv.py --expected-source fd8a8226e697b54ff3422149968e9bf3d293f5d3ed3173922ffe5a946465b0fb --output target/projection-kv-replay.json
```

Tests actually run:

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath='' tests/test_projection_kv_probe.py
```
