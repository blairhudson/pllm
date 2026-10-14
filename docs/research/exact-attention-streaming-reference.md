# Reduction-preserving attention lifetime research

## Verdict

**Bounded numeric feasibility survives; production/SOTA qualification is open.**
The installed original operators and a query/KV-head-group lifetime candidate
match every output bit in 144 bounded synthetic cases. The initial independent
orchestration and ownership suite passes 65 tests. A separate parent-run screen
below adds 36 fresh Qwen-shaped attention processes. No checkpoint load, build,
installation, SDK response or native change was performed. This is not proof of
equivalence for every admissible shape/stride, or evidence that a whole client fits
Low (256 MiB) or Medium (2 GiB). Remote body MAC coverage is **unmeasured**, not 100%.

Files:

- `scripts/probe_exact_attention_streaming.py`
- `tests/test_exact_attention_streaming_probe.py`
- `docs/evidence/exact-attention-streaming-correctness.json`
- `docs/evidence/exact-attention-streaming-validation.json`
- `docs/evidence/exact-attention-streaming-qwen-shapes.json`

The reference imports the installed wheel, not `python/pllm`. Evidence records
interpreter, NumPy version, operator paths/hashes, probe hash and `BuildRef`.
The checked installed build digest is
`37b6546de0a4b221fb71fca33caf97094666551db8fb7c626888ce3c30dee86e`.
Source checkout was `e49a48b6082f8f7d17f744ec1190e84d13a653dc`.

## What changes, and what does not

For each public query tile and complete KV-head group, the candidate calls the
installed executor's original score, attention-scale, causal-mask, softmax and
weighted-value operators. It writes the tile output into a complete attention
output and retires the score/probability tile before processing another tile.
It never splits a feature or key reduction, truncates capacity, copies operands
to contiguous storage, transposes operands differently inside an operator,
reorders keys, recomputes normalization/RoPE or changes the softmax implementation.
GQA groups remain intact: a tile covers `kv_heads * group` adjacent query heads.
Partial final query and KV-head tiles are exercised.

All input Q/K/V owners and positions remain live and charged throughout the
stage. Slices are views, not evidence that their backing owner disappeared.
The output remains complete. The reference does not fetch, page, compress, evict
or relocate KV, and never moves private attention arithmetic to a provider.
Tests check that inputs remain bitwise unchanged and that the preceding output
tile is dead before the next tile starts. They do not constitute native allocation
tracing or secure erasure evidence.

Original operator selection matters:

| Probe profile | Installed score/value contract | Softmax contract |
| --- | --- | --- |
| `legacy` | Executor's per-query `kgd,tkd->kgt` / `kgt,tkd->kgd` einsums | Executor default, complete last axis |
| `dense` | Semantic attention `bkgqd,bktd->bkgqt` / `bkgqt,bktd->bkgqd`, `optimize=False` | Executor default, complete last axis |
| `bf16` | Same dense float32 score/value operators | Original float32 normalization with BF16 probability boundary |
| `prefix_f32` | Original canonical-prefix product materialization and `np.sum` | Original contiguous causal-prefix softmax |

`bf16` means the BF16-softmax branch only, **not** a complete stepwise-BF16 model.
The probe is not a generic schedule interpreter and does not implement arbitrary
declared numeric output boundaries, softcap, sliding windows or hybrid state.
The mask pipeline uses the ordinary absolute causal contract. A separate test
checks fixed-capacity `full_causal` attributes, valid lengths and sliced padding
masks. Invalid/mixed padding and other policies are not streaming admission.

`prefix_f32` already uses position-dependent prefixes internally, with its own
contiguous product tensors and numeric identity. The candidate preserves that
operator; it does not use it as a substitute for default full-domain reductions.
In particular, keeping the same mathematical sum is insufficient to preserve
default NumPy reduction order.

## Checked cases and real counterexamples

Correctness evidence covers four profiles, C/sequence-major/reversed-feature/
Fortran layouts, nine-row prefill, one-row decode at position eight and four
continuation rows starting at position five. Capacity is 23, feature width 33,
three KV heads and two query heads per KV head. Three tile geometries
`(query_rows, kv_heads) = (1,1), (3,2), (7,3)` produce 144 exact comparisons.
No tolerance comparison is used: float32 words are compared as uint32 bits.

The independent oracle in tests never invokes candidate `stages`, `stream` or
executor `_local`; it separately orchestrates the original primitives and
legacy einsums. Additional controls exercise large cancellation scales,
signed zeros, original Q/K RMSNorm and default RoPE before attention, fixed-capacity
mask partitioning and a token-dependent toy prefill/three-decode recurrence.
The toy recurrence compares logits, generated token IDs and complete K/V bits.
It is **not** a generated Qwen/Llama model, quantized decoder or SDK response.
The upstream norm/RoPE test keeps those operations outside the tiled region.

Recorded failures, all from deterministic seed zero on this installed environment:

| Incorrect transformation | Changed attention output words |
| --- | ---: |
| Copy reversed-feature Q/K/V to contiguous arrays before calling the same dense operators | 1,106 |
| Remove unused zero-padded slots: capacity 23 to active length 9 | 26 |
| Substitute `prefix_f32` for original dense arithmetic | 1,145 |

The contiguous-copy failure is a genuine stride/reduction-kernel counterexample,
not an input-value change. The truncation failure occurs despite future K/V slots
being zero and their probabilities masked: softmax's complete last-axis length
changes its sum tree. No failed transformation is accepted with relaxed tolerance.
These results also preclude assuming that a new Rust/SIMD reduction, GEMM,
FlashAttention, online softmax, key-block merging or `einsum(optimize=True)` is an
exact replacement. Successful sampled parity does not prove identical reduction
instruction order for all shapes, machines or NumPy builds.

## Charged owners and cost model

For query heads `H`, KV heads `K`, query rows `Q`, full key capacity `T`, feature
width `D`, group factor `G=H/K`, active KV heads `k` and active query rows `q`:

- Full Q owner: `4*H*Q*D` bytes; full K plus V: `8*K*T*D` bytes.
- Complete attention output: `4*H*Q*D` bytes; positions: `8*Q` bytes.
- One candidate score/probability slab: `S=4*(min(k,K)*G)*min(q,Q)*T` bytes.
- Full score slab: `4*H*Q*T` bytes. Prefill is quadratic when `Q~T`.
- Numeric accounting uses a deliberately conservative `8*S` transient allowance,
  plus mask/index/reduction metadata, full-operand finite-validation booleans and
  one output tile. `prefix_f32` additionally charges `8*T*D` for overlapping
  old/new product tensors. This envelope is an allocation estimate, not RSS.

The main dense slab scales with active queries times full context. Total scratch
is **not only that slab**: validation still reads full participating K/V and
canonical-prefix products retain their own context-by-feature workspace.
The tiny `(H,K,Q,T,D)=(6,3,9,23,33)` owner example records 25,416 input bytes,
7,128 output bytes and 4,413 conservative scratch bytes for `(q,k)=(1,1)`:
36,957 attention-only bytes. Its full score is 4,968 bytes versus a 184-byte
candidate slab. Neither slab ratio nor this envelope is whole-client savings.

For illustration only, `(H,K,Q,T,D)=(24,8,256,1024,128)` has a 24 MiB full
score slab, a 12 KiB `(q,k)=(1,1)` slab and a 384 KiB `(q,k)=(8,4)` slab.
Q/K/V/output together still cost 14 MiB, before other layers' KV or any weights.
These are algebraic dimensions, not an executed large fixture or measured cell.
Low might choose narrow groups; Medium might choose cache-tuned groups. Neither
selection is implemented as tier admission or justified by throughput evidence.

Per-output arithmetic and mathematical MAC counts remain unchanged. Full-domain
default attention still performs every original key reduction. Narrow tiles add
Python/operator dispatch, repeated finite validation, masking and cache scans;
small decode can gain little and head tiling may regress CPU. A native exact
lifetime region might avoid repeated validation only with a separately justified
immutable-input admission. This reference does not remove that work from costs.

Imports, allocator/native overhead, weights, all other decoder KV, residuals,
remote inputs/outputs, transport, filesystem cache and upstream Q/K normalization
temporaries remain separate whole-client owners. The fresh-process measurement
CLI charges fixture creation in process-lifetime RSS, times only attention, and
never retains a full oracle in a candidate child. It does not measure network,
provider CPU, complete TTFT, response throughput or whole-client peaks.

## Distinct from earlier work

- [Retained KV](mpcache-reference.md) changes physical retained rows and has a
  separate state/selection contract. This direction retains every original row.
- [Layer-paged decoder](tiered-decoder-reference.md) changes persistent layer
  ownership. Its 24x slab reduction did not reduce measured process peak RSS;
  that evidence cannot be transferred here.
- [Streaming state](streaming-state-reference.md) changes codec buffers while
  attention consumes complete records. This direction changes attention tensor
  lifetimes, not authentication, storage or disk traffic.
- [Native weight snapshots](native-snapshot-reference.md) change delivery/file
  ownership, not attention computation.
- [MLP working sets](mlp-working-set-reference.md) establish a compiler-bound
  row-lifetime precedent, with their own exactness and measured memory/latency
  tradeoff. They explicitly leave attention unchanged. Their cohort savings are
  not attention evidence and are not multiplied by this slab ratio.

## Parent-run isolated Qwen-shaped measurements

The parent ran six separate cells, each with three alternating fresh-process
control/candidate pairs: 36 processes in total. Synthetic inputs have 14 query
heads, two KV heads, feature width 64, sequence-major layout and key capacity
`input_rows + 7`. Sixteen-query-row candidates keep both KV-head groups; four-row
candidates process one complete KV-head group at a time. The control executes all
queries and groups. Each cell retains its own control. Every
complete output hash agrees within its cell and observed swap growth is zero.

| Numeric profile | Input rows | Query / KV-head tile | Control / candidate peak RSS, MB | RSS reduction | Control / candidate attention CPU, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `prefix_f32` | 150 | 4 / 1 | 54.89 / 51.43 | 6.30% | 50.83 / 55.25 |
| `prefix_f32` | 150 | 16 / 2 | 54.95 / 51.64 | 6.02% | 51.15 / 50.14 |
| `prefix_f32` | 288 | 4 / 1 | 65.16 / 54.33 | 16.62% | 129.80 / 137.36 |
| `prefix_f32` | 288 | 16 / 2 | 65.34 / 54.69 | 16.30% | 129.12 / 127.01 |
| `dense` | 150 | 16 / 2 | 56.31 / 51.30 | 8.90% | 9.98 / 9.76 |
| `dense` | 288 | 16 / 2 | 72.22 / 53.49 | 25.93% | 34.95 / 33.27 |

Values are separate three-sample medians, not confidence intervals. CPU covers
attention execution, while process-lifetime RSS includes imports and synthetic
input construction. It excludes all other decoder/client owners. The short CPU
windows do not establish a speedup; four-row `prefix_f32` tiles visibly cost more
CPU. This evidence supports trying sixteen-row lifetime tiles, not applying these
percentages to complete-client peaks or qualifying either memory tier.

Raw records retain every worker, public fixture, probe and installed implementation
hash, output hash, owner estimate and `BuildRef`. The parent additionally checked
complete control/candidate replicate coverage and identical implementation identity
inside each cell before archiving. A new replay produces new timings:

```sh
for profile in prefix_f32 dense; do
  for rows in 150 288; do
    target/frontier-sdk-env/bin/python scripts/probe_exact_attention_streaming.py \
      --measure --profile "$profile" --rows "$rows" --capacity "$((rows + 7))" \
      --width 64 --heads 2 --group 7 --layout sequence_major \
      --query-rows 16 --kv-heads 2 --replicates 3 \
      --output "target/attention-lifetime-${profile}-${rows}-tile16.json"
  done
done
```

The two four-row cells use `--profile prefix_f32 --query-rows 4 --kv-heads 1` with
the same other settings. Runs use the shared exclusive measurement lock and pressure/swap
guard. The source SHA-256 is pinned in each raw record; do not merge a changed
implementation or later machine into this cohort.

## Correctness replay and next gates

Small correctness replay:

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath='' \
  tests/test_exact_attention_streaming_probe.py -q
target/frontier-sdk-env/bin/python scripts/probe_exact_attention_streaming.py \
  --output docs/evidence/exact-attention-streaming-correctness.json
```

An additional, separately scoped small measurement configuration is:

```sh
target/frontier-sdk-env/bin/python scripts/probe_exact_attention_streaming.py \
  --measure --profile dense --rows 33 --capacity 128 --width 64 \
  --query-rows 4 --kv-heads 1 \
  --output target/exact-attention-streaming-dense-33.json
```

Repeat with `--profile legacy`, `--profile prefix_f32` or `--profile bf16` as
separate identities. The CLI uses `frontier_probe_support.fresh_workers`: shared
fail-fast measurement lock, post-reserve headroom, swap watchdog, fixed thread
settings, fresh serial control/candidate processes and a stable `BuildRef`.
Its admission includes a 256 MiB import allowance plus the larger numeric owner
envelope; this is host safety admission, **not** Low qualification. One pair is
an exploratory screen, not repeatable performance evidence.

Next gate requires compiler-bound discovery of an exclusive attention region,
original numeric/output-boundary identities, masks, complete KV ownership,
public tile geometry, downstream/residual lifetimes and failure/cancellation
cleanup. A native implementation must prove or exhaustively constrain the
actual reduction kernels/strides, not merely reuse an einsum expression name.
Then source-locked generated Qwen2/Qwen3/Llama controls must compare complete
prefill/decode logits and every active KV bit, including real RoPE/QK norms,
capacity growth and cache strides. Actual checkpoint checks remain parent-owned.

Only after that should serial whole-decoder/provider-role measurements evaluate
Low/Medium RSS, CPU, latency, traffic and **at least 90% remote body MACs** on the
unchanged prepared composition. Remote placement is conceptually unaffected by
local attention lifetimes, but this probe has no provider ledger and proves no
coverage fraction. Until those gates pass, this is a research survivor, not an
exact whole-client optimization or a SOTA result.
