# Exact exception-split Preparation screening

This research slice tests `W = B + E` for public signed-i8 checkpoint weights.
It adds no production kernel, runtime dispatch, role contract or crate change.
The executable is `scripts/probe_exception_corrections.py`; evidence is
`docs/evidence/exception-corrections-screen.json`. The current native comparison
is `crates/pllm-core/src/digit_gemm.rs`, not scalar Python or NumPy performance.

## Result and scope

**Do not promote this as an optimization yet.** Narrow weights do not reduce
dense products or total weight conversion elements. At the maintained Qwen
square/QKV/gate-up geometries, input width 896 already fits one native block:
there are no block-call savings even with zero exceptions. The down projection
has a plausible block-loop/reconstruction saving, contingent on actual tails,
native sparse execution, decoding, setup and total ownership. The initial
synthetic screen left real tails unknown. The separate parent scan below finds
68.61–75.65% b4 tails on four public Qwen down tensors, weakening that premise;
every tested packed-base/CSR layout exceeds original i8 storage. Native speed and
amortization remain unknown. The dense public counterexample additionally
falsifies a universal benefit, not every possible checkpoint-specific kernel.

The bounded screen checks 180 correction output values against an independent
Python arbitrary-integer oracle, across rings 16/24/32. The test file passes
35 tests, including extreme signed weights, wraparound, carries, partial blocks,
no exceptions, all exceptions, invalid indices/pointers/coefficients, packed
padding, reproducibility and the masked-correction identity. No timings are
reported; NumPy SGEMM is only a generic correctness reference, **not** a native
Accelerate candidate or speed baseline. Its Python sparse loop is not a credible
performance implementation.

## Construction and exact bounds

For width `b`, let `h=2**(b-1)`. Set `B=clip(W,-h,h-1)` and `E=W-B`.
Exception coefficients lie in `[-128+h,128-h]` and fit signed i8. They are stored
only where nonzero in canonical row-sorted CSR. This is clipping plus exact
integer restoration, not a new low-bit quantizer or an approximate weight model.
The source model's per-row scale and original W8 quantization remain unchanged.
All tuning and dispatch must depend solely on these public weights/geometry.

Use the current native balanced base-256 mask digits `d in [-128,127]`, including
the cumulative carry `sum(128 << earlier_shift)`. Products have asymmetric bounds:

| Base width | B range | d*B minimum | d*B maximum | Strict safe block | Worst partial absolute sum | E range |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 4 | -8..7 | -1,016 | 1,024 | 16,383 | 16,776,192 | -120..120 |
| 5 | -16..15 | -2,032 | 2,048 | 8,191 | 16,775,168 | -112..112 |
| 6 | -32..31 | -4,064 | 4,096 | 4,095 | 16,773,120 | -96..96 |
| 7 | -64..63 | -8,128 | 8,192 | 2,047 | 16,769,024 | -64..64 |
| Current 8 | -128..127 | -16,256 | 16,384 | 1,023 | 16,760,832 | 0 |

The minimum is `min(-128*(h-1), -127*h)`; assuming symmetric maxima or
forgetting the -128 digit is unsound. Every product and every reduction-tree
partial sum has integer magnitude strictly below `2**24`. Thus float32 multiply,
addition and FMA are exact for these blocks, independent of reduction order,
provided the implementation really uses float32 rather than reduced-precision
products. Partial K blocks use the same proof. Products are converted to signed
integers, shifted and summed with wrapping arithmetic, then masked to the ring.
The highest carry beyond the chosen ring is discarded modulo that ring.

A boundary counterexample is `16384*(-128)*(-8) + (-1)*(-1) = 2**24+1`.
Float32 addition rounds away the final unit. The probe therefore keeps the
current native strict `<2**24` convention, even though exactly `2**24` itself is
representable.

Sparse exceptions use integer arithmetic on the original mask residue, not
another float32 accumulation. For a row of `n<=16384` exceptions the full-ring32
absolute sum is at most `n*(128-h)*(2**32-1)`. At b=4 the bound is
8,444,249,299,353,600, below `2**63`; one coefficient product is at most
515,396,075,400. Ring24/16 are smaller. A proposed native implementation can use
signed i64 row accumulation with the geometry cap, or wrapping u32 multiply/add
without relying on signed overflow. The reference uses Python integers. CSR
validation rejects out-of-range, wrapped-negative, duplicated or unsorted indices,
invalid pointers, zero coefficients, excessive coefficients and corrections that
do not point at the appropriate clipped endpoint.

## Comparison with CURRENT native digit8

Native digit8 already converts weights tile-by-tile, calls Apple Silicon CPU
Accelerate with beta=0, reconstructs exact residues in Rust, releases the GIL,
uses at most 64 MiB explicit workspace and has no persistent float weight copy.
Unsupported platforms/shapes are rejected or dispatched by the existing serving
policy; its default public minimum batch is 64. A generic NumPy result cannot
inherit any of those properties.

Let `D=ring_bits/8`, `K` be the safe block clamped to input width, `T` the number
of output tiles, `R` output width, `C` input width and `Q` mask batch:

- Dense product count stays `D*Q*R*C` in both implementations.
- Float weight conversion stays `R*C`. A longer K does **not** eliminate these
  copies; an implementation decoding a packed base adds work per weight.
- Mask digit conversion stays `D*Q*C*T`, because the current native loop decodes
  the mask again per output tile. Longer K helps loop overhead, not this element
  count. Smaller output tiles can increase this count.
- SGEMM calls are `D*ceil(C/K)*T`; partial-output reconstruction is
  `D*Q*R*ceil(C/K)`. These are the actual possible savings.
- Exceptions add `Q*nnz(E)` integer MACs, indirect mask reads and coefficient/index
  traversal. These counts are not wall-time ratios or a sparse-density crossover.

For batch 150 and ring32, the zero-tail optimistic lower bounds are:

| Output x input | Current calls | Proposed calls b4 / b5 / b6 / b7 | Current explicit bytes | Proposed b4 explicit bytes |
| --- | ---: | --- | ---: | ---: |
| 896 x 896 | 4 | 4 / 4 / 4 / 4 | 4,824,064 | 4,824,064 |
| 1152 x 896 | 8 | 8 / 8 / 8 / 8 | 5,513,216 | 5,513,216 |
| 9728 x 896, gate/up | 40 | 40 / 40 / 40 / 40 | 10,658,816 | 10,658,816 |
| 896 x 4864, down | 20 | 4 / 4 / 8 / 12 | 5,355,432 | 21,426,176 |

The down stage's b6 workspace is 18,208,680 bytes; b7 is 9,639,848. These are
proposed native-buffer formulas, not measured RSS. At C=16384, b4 cannot keep a
1024-output tile within 64 MiB at this batch: the probe shrinks the tile, increasing
mask decoding. Calling the longer block a 16x arithmetic speedup is false.
For the 896-input stages, reject the **block-saving premise** immediately. Native
timing would be needed to establish some different benefit such as changed
SGEMM behavior; this slice has no evidence for it.

## Tail density, representation and setup gates

The scanner reports the complete signed-i8 histogram, exact exception density,
row minimum/maximum/p95 exception counts, quantized-weight SHA-256 and float32
scale SHA-256 for each explicitly selected public tensor. It reads every column
of complete 32-row chunks using the installed current native W8 per-row quantizer.
Chunking output rows does not change row scale or rounding. It records hashes of
the actual imported quantizer/support/native extension, not checkout substitutes.
Qwen public checkpoint tensors are output-by-input and require no transpose.
Separate gate/up or Q/K/V scans are not automatically fused compiled stages;
counts must be concatenated and source offsets verified before native promotion.
Explicit selected tensors are not all layers or complete topology coverage.

No real tail density is in the current evidence. As a reproducible adversarial
public distribution, uniform coverage of all 256 signed-i8 values gives densities
93.75%, 87.5%, 75%, 50% at widths 4,5,6,7. These are not Qwen measurements.

Unaligned CSR costs `4*(R+1)+5*nnz(E)` bytes with u32 column indices and i8
coefficients. Unpacked B adds `R*C` bytes and is always larger than original i8
weights once any CSR metadata is included. Packed B costs `ceil(b*R*C/8)`.
Ignoring row pointers, packed storage beats original i8 only below tail densities
10%, 7.5%, 5%, 2.5% for b4/5/6/7, respectively. Exact gates in evidence include
row pointers and packing round-up. These are storage crossovers, not performance
crossovers. An aligned 8-byte sparse entry makes these thresholds stricter. U16
indices could reduce overhead for these shapes but are not the supplied format.

Bit packing has an explicit executable canonical little-bitstream round trip.
The slow Python reference is not a native decoder. A real kernel must price
bit extraction/sign extension into float tile buffers on every issuance, or price
a retained decoded-i8/float cache rather than assuming its bytes and setup are
free. Clipping directly from authenticated original i8 can avoid retained B but
does not eliminate original weight ownership or conversion. The prototype
retains both W and its split when used as an oracle; this is not a production
memory optimization.

Setup scans O(R*C) values, clips, builds CSR, and optionally packs B. The Python
split implementation additionally allocates an i16 difference and `np.nonzero`
index arrays; its setup peak includes approximately W+B+2RC+16nnz, then CSR and
packing overlap. This is not a full-shape setup measurement. The optional histogram
scan does not build or time CSR at all. A future native setup can use bounded
row/tile construction but must report its actual peak and authenticated artifact
ownership. If measured setup delta is `S` and measured fresh-issuance saving is
`delta>0`, break-even is at least `ceil(S/delta)` queries, with source publication,
decode/cache startup and invalidation included. If delta<=0, no steady-state
amortization horizon exists. Neither S nor delta has been measured here.

For the maintained 24 occurrences of each of the four geometries above,
original body i8 matrices total 357,826,560 bytes. Hypothetical all-stage b4 packed
bases plus CSR row pointers already total 180,130,176 bytes at zero tails, before
exception entries; add `5*sum(nnz)` bytes. Retaining original authenticated
snapshots adds the original 357,826,560 bytes again. This projection excludes
embedding/head, scales, paging effects and artifacts, so it is not an admission
decision. Rebuild/discard or disk-streamed derivative ownership is not implemented.

## Privacy, outputs and peak owners

The only proposed change is trusted Preparation's computation of `W*r`.
`(B*r + E*r - s) mod 2**ring_bits` equals the original correction exactly.
Keep the original full-ring, fresh one-use masks, domain separation, issuance
binding, burn-on-cancel/replay behavior, transport and non-collusion assumptions.
Do not sparsify masks, share masks between candidates, expose sparse mask-index
access to Inference, or move exception evaluation to an untrusted role. Public
indices are not secret; the values gathered through them are Preparation masks.
The test checks algebra without implementing a replacement seed protocol or
claiming production secret erasure in Python. Existing production lifecycle
integration remains a required separate gate.

Per batch, both client and Preparation still expand `4*Q*(C+R)` bytes of canonical
ring arrays. Kernel output occupies `4*Q*R`; correction subtraction and encoding
retain further arrays. Encoded correction body is `Q*R*ring_bits/8` bytes before
unchanged headers/authentication. Outputs are not sparsified: sparse E does not
reduce correction traffic or client masking/reconstruction operations.

The native-style explicit workspace is
`4*(Q*R + Q*K + tile*K + Q*tile)` bytes, including its full u32 output.
The cost function solves for an output tile <=1024 under the existing 64 MiB
ceiling. Separate peak owners are input binding copies (`4QC`), output binding
copies (`4QR`), r/s expansion, correction subtraction arrays, encoded bytes and
file/page overlap, original paged immutable weights, B/CSR/packed artifact retention,
per-row scales, BLAS scratch allowance, interpreter/allocator and stage concurrency.
The report includes the current 64 MiB platform allowance and 224 MiB conservative
extra role admission from the maintained reference; neither is a measured candidate
peak, nor an authorization to silently reuse admission for extra retained weights.
The reference's full-matrix NumPy conversions, u64 digit/residue arrays and Python
objects are separate owners and cannot be represented as this proposed Rust peak.

Low client budget is 256 MiB and Medium is 2 GiB. No derivative weight or workspace
is charged to the client, but unchanged client expansion/output/encoding/import
peaks still need measurement. Admit **whole topology** independently under each
budget: resident/paged Inference weights and execution, Preparation snapshots and
derivatives, concurrent stages, masks, queues, transport, authenticated artifact
cache/publication and filesystem/page overlap must be accounted for on their actual
owners. The report deliberately leaves whole-topology admission null. Component
memory savings cannot establish client SOTA or complete-response performance.

The existing subset-correlation compiler negative result remains relevant:
its Python subset path is 39.62–70.86x slower than native digit8 at measured full
single-row shapes, and its retained 4-bit full-model tables exceed 2.86 GB. That
result motivates native controls and setup-inclusive accounting, not a numeric
prediction for this different exception representation. The maintained native
digit8 confirmation's 25.03% Preparation CPU saving but 1.94% cold-response saving
also prevents importing a kernel ratio as a serving ratio.

## Parent source-locked down-tensor scan

Evidence: [`exception-corrections-qwen-down-screen.json`](../evidence/exception-corrections-qwen-down-screen.json).
The parent scanned layers 0, 8, 16 and 23 of pinned public Qwen2.5-0.5B-Instruct,
17,432,576 weights in total. Each tensor is output-by-input `[896, 4864]`; bounded
row chunks use the installed original W8 per-row quantizer. Source hashes and
revision come from independently retained `research-residual-delegation.json`,
through [`orthogonal-frontier-qwen25-source-lock.json`](../evidence/orthogonal-frontier-qwen25-source-lock.json).
Actual files and implementation hashes are checked before and after the scan.

| Layer | b4 exceptions | b5 exceptions | b6 exceptions | b7 exceptions | b7 packed base + CSR, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 75.633% | 55.346% | 26.842% | 4.011% | 4.691 |
| 8 | 72.874% | 51.218% | 22.549% | 2.889% | 4.446 |
| 16 | 68.613% | 46.969% | 21.266% | 3.003% | 4.471 |
| 23 | 75.652% | 54.876% | 25.978% | 3.797% | 4.644 |

Original i8 weights cost 4.358 MB per tensor. All b4–b7 packed-base/CSR cases lose
on retained storage, even before keeping an original snapshot. At 150 rows/ring32,
b4/b5 would reduce modeled SGEMM calls 20 to 4 but add roughly 307–495 million
integer exception MACs and increase explicit workspace 5.36 to 21.43 MB. b7 has
a narrower 20-to-12-call hypothesis, 18.9–26.2 million exception MACs and 9.64 MB
workspace. Dense product counts are unchanged. These are modeled costs based on
observed public tails, not native candidate timings or real correction execution.

The scan reports zero observed swap growth under the existing shared lock and
pressure guard. Its conservative scan peak estimate is 1,127,296,288 bytes;
that is neither measured RSS nor client-tier qualification. No whole-model tail
claim follows from four tensors. Prefer attention/conveyor gates before investing
in this sparse-tail implementation.

## Replay and parent commands

Tiny correctness/cost screen, no checkpoint scan or timings:

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_exception_corrections_probe.py
target/frontier-sdk-env/bin/python scripts/probe_exception_corrections.py --output docs/evidence/exception-corrections-screen.json
```

The completed four-tensor scan, with no kernel timing or inference, replays as:

```sh
target/frontier-sdk-env/bin/python scripts/probe_exception_corrections.py \
  --scan-checkpoint "$CHECKPOINT" \
  --source-lock docs/evidence/orthogonal-frontier-qwen25-source-lock.json \
  --tensor model.layers.0.mlp.down_proj.weight \
  --tensor model.layers.8.mlp.down_proj.weight \
  --tensor model.layers.16.mlp.down_proj.weight \
  --tensor model.layers.23.mlp.down_proj.weight \
  --output target/exception-corrections-qwen-down-screen.json
```

`CHECKPOINT` is the existing offline cached public Qwen2.5-0.5B-Instruct directory
at revision `7ae557604adf67be50417f59c2c2f167def9a775`. A source lock must be an
independently retained JSON file with `revision` and `sha256` mapping relative
`config.json` and required `.safetensors` filenames to their exact hashes. Do not
generate this trusted lock from arbitrary changed local files and call that pinned
provenance. The scanner supports at most eight explicit keys and 32 Mi elements
selected elements, float32/float16/BF16 output-by-input tensors, no quantized
checkpoint formats or transposed architectures. It requires installed native
quantization, rejects oversized dimensions/mapped-page estimate before value
reads, uses 32-row chunks and verifies checkpoint/implementation hashes before
and after. File hashing itself reads public source bytes without quantizing them.
Hashing full shards twice costs real I/O and is not a free setup claim.

The shared `frontier_probe_support.fresh_workers` serial lock, 3 GiB post-reserve
headroom request, 240-second timeout, swap watchdog and installed BuildRef apply.
The conservative worker cap is 2 GiB including all locked mapped shard pages,
row-chunk scratch and 128 MiB runtime allowance. This is scan-role admission only,
not Low client or complete-topology admission.

Proceed past the tail gate only if real down tails and storage/ownership are
credible. Stop if exceptions are dense or there is no block-saving premise.
Remaining promotion gates: an authorized native Accelerate+sparse integer
implementation; source-locked full-shape oracle and correction-frame parity;
matched CURRENT native digit8 and integer dispatch controls, including batch 1,
16 and >=64; three fresh setup-inclusive process replicates; packed decoding and
mask expansion/subtraction/encoding/file costs; peak owner/concurrency admission
under both budgets; production one-use/privacy checks; and all seven matched
whole-response metrics with source/numeric/workload/output commitments. None of
those missing gates is inferred from the symbolic call reduction.
