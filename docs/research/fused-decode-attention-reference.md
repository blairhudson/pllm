# Single-query decode scratch fusion (investigation 8)

## Verdict

**Bounded exactness passes; gain remains unmeasured.** This is not initial-prefill
attention tiling: no query/head tiles, new einsum, key blocks, reordered reduction,
online softmax, approximate exponential, profile substitution or shortened capacity.
Single-query GQA keeps the original installed whole-head score/value kernels.
The candidate reuses their fresh exclusive score owner for scale, mask and dense
softmax. This removes intermediate allocation/dispatch opportunities, not MACs.
No SDK, compiler registration, checkpoint, build or shared file was changed.

Files: `scripts/probe_fused_decode_attention.py`,
`tests/test_fused_decode_attention_probe.py`, and
`docs/evidence/fused-decode-attention-synthetic.json`.
Source checkout inspected: `79ab200293044d8bb6829f44b0015a7c503e7513`.
Evidence pins installed operator/build, NumPy, interpreter, probe, test, note,
original streaming probe and checkout attention-tiling hashes. Evidence itself
can be hashed independently with `shasum -a 256`.

## Numeric contract

Dense scores/values call `semantic_attention_scores` / `semantic_attention_values`
with their existing `optimize=False` einsums. All query groups, original operand
strides and full logical KV capacity remain unchanged. Scale retains the runtime's
Python `float(1.0 / math.sqrt(width))` coefficient and float32 result boundary.
Single-query absolute causal masking writes negative infinity only into private
scores. Dense maximum and denominator retain the full last-axis domain and
original NumPy reduction. Subtraction, NumPy `exp`, division reuse score storage
via `out=`; operations remain separate, not FMA or real-arithmetic reassociation.
In-place alias/alignment paths are experimentally checked, **not universally proved**.

`prefix_f32` calls original `causal_reduction.scores`, `softmax`, and `values`.
Only scale/mask reuse storage. Original prefix product copies, `np.sum` domains,
softmax prefix copies and probability output remain charged. Its softmax is
already prefix-specialized; this reference does not pretend to accelerate it.
BF16, legacy einsum, factor-object scale, windowed/retained attention, padding,
softcap and arbitrary numeric boundaries are outside this candidate's admission.
Upstream normalization/RoPE are untouched.

Installed runtime baseline is independently invoked through `_local`; an oracle
separately orchestrates original primitives without `_local` or candidate calls.
120 evidence fixtures cover dense/prefix_f32, C/sequence-major/reversed-feature/
Fortran/oversized-backing-owner views, GQA groups 1/2/4, first/interior/final
positions, signed zeros, cancellation and saturated/underflowing exponentials.
Tests add 126 shape-boundary comparisons: widths 1/7/16/32/64/65/128, capacities
1/17/64, groups 3 and three strided layouts. Every comparison uses uint32 words;
inputs are checked unchanged. Weak references check retirement, dense score reuse,
prefix score retirement before weighted values, and nonaliasing final output.
Fixed-capacity mask tests check valid-length semantics and reject over-capacity.
Measurement-gate tests reject invalid iteration counts, per-child/cohort product
work and memory; inject host-admission denial before any fixture allocation;
reject child identity mismatch before allocation and at parent collection; check
charged operation counts and swap fields. Fake clocks/operators test separate
setup, cold and repeated totals/per-op fields without running timing cohorts.
Current suite: 273 tests pass; Ruff clean.

Historical streaming counterexamples rerun as negative controls: contiguous
operand copying, dropping zero future slots and substituting prefix_f32 for dense
all change output bits. Those controls use the historical multiquery fixture;
they are not claimed as single-query failures. Neither passing samples nor
mathematical equality authorizes a native replacement reduction.

## Owners and baseline limits

The existing runtime already retires values at last use (`semantic_executor.py`
execution loop), and existing attention tiling applies only to initial prefill.
Thus whole-stage last-use retirement is **not** a new decode gain. Candidate's
remaining mechanism is score-buffer reuse and removal of local stage dispatch.
If compiler-bound baseline already reuses these buffers, this mechanism yields
no further gain. Report no gain until matched timing and complete-response gates.

For H query heads, K KV heads, capacity T and width D, full Q/output cost 4HD
each, K/V cost 8KTD, positions 8 bytes and dense score scratch S=4HT. All backing
owners are charged once, not merely each view's `nbytes`. Candidate inputs remain
borrowed views; score/output are allocations. Canonical product arrays are copies.
Dense candidate has one reusable score slab plus reduction/validation metadata;
prefix candidate can overlap two slabs plus original product workspace. Ledger
conservatively charges baseline 8S, dense candidate S, prefix candidate 2S,
16H+16T metadata, largest operand/score finite-validation boolean array, and 8TD for
prefix product overlap. These are **allocation envelopes, not measured RSS**.
No full oracle is retained in timed candidate children.

Default child fixture (H=14,K=2,T=1024,D=64,position=287) owns 1,052,168 input
bytes. Prefix attention-only envelope is below 3 MiB per arm. Parent never
constructs a fixture: validated integer public geometry computes admission before
the helper's host check. Admission charges four input-owner equivalents for RNG
float64/transposition construction, baseline worst scratch, two complete outputs
for old/new repeated-call assignment overlap, and 256 MiB for parent/child
imports, allocator and thread-stack headroom. Default requested admission is
273,782,016 bytes for prefix_f32, 273,257,728 for dense. Actual process peak includes
fixture construction; interpreter and installed dependencies can dominate.
Other layers' complete KV, weights, residuals, remote buffers, transports,
Preparation and filesystem cache are excluded and must be added for whole client.
No Low/Medium qualification, >=90% prefill/decode remote-body-MAC claim, traffic,
TTFT, Preparation CPU/RSS or N-1 throughput result follows from this screen.

## Replay and serialized next gates

Small correctness only (no checkpoints):

```sh
target/frontier-sdk-env/bin/python -m pytest --noconftest -o pythonpath='' -q tests/test_fused_decode_attention_probe.py
target/frontier-sdk-env/bin/ruff check scripts/probe_fused_decode_attention.py tests/test_fused_decode_attention_probe.py
target/frontier-sdk-env/bin/python scripts/probe_fused_decode_attention.py --output docs/evidence/fused-decode-attention-synthetic.json
```

Parent-only bounded serial fresh-process timing, shared frontier lock and
host-pressure watchdog, six children total, 60-second timeout each:

```sh
target/frontier-sdk-env/bin/python scripts/probe_fused_decode_attention.py --measure --profile prefix_f32 --capacity 1024 --position 287 --width 64 --heads 2 --group 7 --iterations 256 --replicates 3 --output target/fused-decode-attention-prefix.json
```

Repeat separately with `--profile dense` and distinct output. Each child reports
fixture-setup CPU/wall, cold single-call CPU/wall/hash, then repeated same-fixture
CPU/wall totals and per-op values, final complete-output hash, complete-process
peak RSS and owner ledger. Cold/final hashes must match within each child and
across paired arms. Intermediate repeated outputs are not hashed; only cold and
final complete outputs are checked. Repeated-fixture work is **not serving
throughput**, TTFT or N-1 response throughput. Imports/identity checks are outside
setup/operator timers but remain inside process-lifetime RSS. No timing cohort
was run while implementing this refinement. `gain` stays null.

Limits: iterations 1–1024 (default 256), replicates 1–5; complete public geometry
T<=4096, D<=256, K<=16, G<=8 and 0<=position<T. All limits are joint, not permission
to execute every Cartesian-product combination. Each child charges 1+iterations
operations. Score/value float products per operation are 2HDT for dense or
2HD(position+1) for prefix_f32; metadata, scale and softmax work remain timed but
are not counted as these products. Reject more than 536,870,912 products per
child, 4,294,967,296 per paired cohort, or 536,870,912 admission bytes. Default
prefix child charges 132,636,672 products, cohort 795,820,032; dense child charges
471,597,056, cohort 2,829,582,336. Each default arm charges 771 operations.

Parent freezes full interpreter/NumPy, installed BuildRef/operator hashes,
probe/source/test/note hashes before admission. Each child checks frozen identity
before allocation and after execution; parent independently checks every returned
child identity and work count, plus its own final identity. Helper's final parent
BuildRef check remains additional protection, not a substitute. Result includes
host baseline/final swap, final-minus-baseline delta, watchdog maximum observed
swap growth, minimum available bytes and error, where supported. Unsupported
fields are null, never evidence of zero swap. Admission/work/identity/hash failure
raises without issuing a gain result.

Next compiler-binding gate: select only an exclusive contiguous decode five-op
region with one query and ordinary full_causal fixed capacity. Bind source plan,
composition, original schedule, exact profile/scale/output boundaries and an
execution-basis digest. Reject interior external consumers, state roots, aliases,
protected/verified nonlinear owners, retention/window/padding and unsupported
numeric contracts. Recheck position, GQA geometry, strides and cache owner/lineage;
preserve all original validation obligations and output lifetime. Failure burns
runtime/reservations and clears scratch before any output escapes. No synthetic
function here grants that authority. Then run generated compiled multi-step and
restored-prefix logit/KV bit comparisons, injected failures and complete matched
SDK responses under full-owner admission before pricing CPU or either RSS tier.
