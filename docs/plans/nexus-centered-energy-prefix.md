# N-N: bounded width-768 centered-energy prefix

**Status: D38 approves only the bounded clear/structural implementation below.**
Encrypted profiles, native HE builds/execution and ciphertext/key authority
remain excluded. Approval supplies no ambient full-trace metadata-work grant;
see [D38](../../paper/methods/DECISIONS.md#d38-three-independently-bounded-follow-ups-to-d37).
Historical proposal baseline: `7fa07d746f55ae7157f3a371a57def920fb9fbb0`.
No implementation, tests, native execution or measurements accompanied that
original packet; current implementation status is recorded below.

Historical Astra review revision against Main `a61f4bf6c8f52c31e6584de4aad5cb65dd23be28`:
§5 now specifies the shared construction quote and separates conditional
synthetic trace admission from the clear executor's physical buffer pool.
The original author's document and patch remain unchanged. Proposal/no-API wording
in the preserved design below records that stage, not the current source status.

Normative scope: [D37](../../paper/methods/DECISIONS.md#d37-specify-seven-separately-selectable-method-completion-contracts),
[N-N decision row](ten-method-execution-decisions.md#select-individual-rows-not-approve-everything),
and [Eq.(5) dependency proposal](../research/nexus-numeric-contract-audit.md#separate-unapproved-normalization-dependency-proposal).

## Implemented clear/structural slice — ea6057cf

[Implementation commit `ea6057cf`](https://github.com/blairhudson/pllm/commit/ea6057cf694a0d20a0144414376580abdd21fb63)
publishes the D38 clear-only slice under the original map, budgets and conditions.
Astra133 source review cleared all three admission repairs:

| Published module | Implemented binding and ordinary regression evidence |
| --- | --- |
| [`pllm_core::row_fold`](../../crates/pllm-core/src/row_fold.rs) | `Geometry`, `Invocation` and `WorkLedger`: caller-buffered ordered energy prefix; checked cumulative clear/oracle/control sum and control subcounter. [`row_fold_tests.rs`](../../crates/pllm-core/src/row_fold_tests.rs) covers independent dense parity, padding/row boundaries, exact/short work, poison and unwind. |
| [`pllm_compiler::row_fold_reference`](../../crates/pllm-compiler/src/row_fold_reference.rs) | `BorrowedClearBuffers` / `ChargedRowFold` retain original clear borrows and whole-pool charges; `row_fold_compiler_control_and_core_share_exact_aggregate_before_entry` checks compiler control and core work share the same checked ledger. This is not source-tensor authentication. |
| [`encrypted_level_schedule::construction`](../../crates/pllm-compiler/src/encrypted_trace_construction.rs) | `PublicCiphertextConstructionQuote` has private admission fields/read-only getters. Non-Clone `TraceAdmission` binds the exact immutable quote and original metadata ledger: `admit_trace` prepays once before construction; `validate_constructed` consumes it without another debit. [`encrypted_trace_construction_tests.rs`](../../crates/pllm-compiler/src/encrypted_trace_construction_tests.rs) checks exact-one-invocation admission, swap/abort/failure accounting and pre-factory refusal; compile-fail cases reject quote forgery/token replay. |

Final standard core/compiler package gate: **2,180 passed, nine existing ignored,
one explicitly skipped full-map test**, including integration/doc regressions.
Separately, `all_rows_dense_independent_parity_six_full_maps` already executed its
**six independent 32×1024 fixtures once**, checking every active/padding lane on
the same unchanged numerical kernel; that historical evidence is not added to
the 2,180 count. Strict `--lib --tests` Clippy passed. All-targets Clippy is not
claimed: the legacy `boundary_head_dotprod.rs:73` example warning remains.

The full-trace quote is **7,774 metadata records**, **1,991,168 validator payload
bytes** and **30,068,627,962,880 work quanta**: bounds/rejection ceilings, not an
ambient grant or successful full-trace verification. Independent finite metadata
admission remains required; missing/short admission returns
`MissingAmbientTraceAdmission` / `MetadataWorkRefused` before graph factories.
Clear-original borrow/charge checks confer no tensor provenance or source/security
adoption. Full LayerNorm, inverse/epsilon, HE execution, parameters and keys remain
excluded; the D37 unknowns and original normative conditions below remain intact.

## 1. Result and provenance boundary

For each row of **768** real entries, compute only

```text
s = sum(a_i)
z_i = 768*a_i - s
v = sum(z_i*z_i)
```

These equations are paper Eq.(5); the audit records its missing epsilon,
zero-variance, inverse-square-root domain/schedule and checkpoint mapping.
This prefix deliberately accepts `v=0`. It performs no inverse, square root,
gamma/beta, division, clipping, refresh, normalization or reveal.

**Proposed, not recovered:** the padded map, masked rotations, ordering, workspace
and budgets below. They realize QuickSum's row-sum semantics; they are NOT a
recovered instruction-for-instruction Algorithm 3 schedule. Paper slots32768
and degree65536 are reported in the audit's
[HE gaps](../research/nexus-numeric-contract-audit.md#missing-he-publication-and-complete-request-contracts),
not adopted as a usable parameter/security contract here. Artifact LayerNorm
`square/sum * 1/768` is not substituted for this centered-energy equation.

## 2. Reuse before implementation

| Existing owner/primitive | Reuse and explicit limit |
| --- | --- |
| `pllm_core::coefficient_compression::{CompressionGeometry,CoefficientCompression}` | Existing caller-buffered coefficient algebra remains unchanged. Do not reproduce it as a new NEXUS result. Its odd-limb coefficient ordering is NOT this CKKS slot layout; no zero-copy coefficient-to-slot conversion is asserted. |
| `pllm_core::slot_matrix_layout::{Layout,Limits,Error}` | Reuse checked geometry/work, padding and caller-buffer idioms where applicable, not THOR's diagonal/time/head address map. That map does not implement this row fold. Add a method-neutral row-fold contract, not a second paper-private matrix framework. |
| `pllm_core::ckks_layout::{parameters,softmax_parameters,padded_width}` | **Do not widen or use these as admission:** matrix/softmax width is bounded by32, softmax is power-of-two and its parameters are adaptations. Width768 is not admitted there. |
| `encrypted_level_schedule::{PublicCiphertextGraph,PublicCiphertextGraphOperation,PublicCiphertextValue,PublicCiphertextOwner,PublicCiphertextCapability,PublicCiphertextState,PublicCiphertextTraceStep,PublicCiphertextTraceCapacity,validate_public_ciphertext_trace}` | Reuse SSA, aligned/unary joins, exact capability matching, full backing and output-before-release accounting. `PublicPrimitive::Arithmetic` plus distinct kernel identities names the operations below. Names/digests do not prove source fidelity, keys, errors or capacity. |
| `encrypted_level_schedule::dag::{PublicEncryptedDagCapacity,quote_public_encrypted_dag_memo,validate_public_encrypted_dag_memo_solution}` | Existing branch/join machinery, if scheduling later becomes necessary. This packet freezes one schedule: **no DAG search/memo arena**. Serial `validate_public_encrypted_level_trace` cannot flatten this branching arithmetic graph. |
| Original input/caller workspace ownership | Keep exact immutable borrows; no input clone or digest-to-owner upgrade. Core clear algebra stays caller-buffered; do not introduce a core→garble dependency to obtain a Budget. At an owning integration boundary, use existing shared-allocation owners/counter continuity, not numeric byte quotes as grants. |

Shared trace validation is consistency-only. Its logical release is not native
buffer erasure/deallocation. Existing compression, trace and ownership APIs must
not have their legacy limits/accepted inputs silently changed.
The existing private `trace_preflight` prices validator work AFTER receiving a
caller-built graph. It does not admit that graph's construction. If a public
preconstruction resource seam is needed, extract/extend THAT shared counter with
bounded immutable construction counts; do not copy its traversal/formula into
a NEXUS-private validator. The numeric R calculation below is an audit check,
not a claim that an existing construction ticket already exists.

## 3. One frozen map, including the unused lanes

Logical cyclic extent **S=32768**, **32 rows**, block length **B=1024**.
Slot `j=1024*r+c`, `0<=r<32`, `0<=c<1024`:

* `c<768`: original row entry `a[r,c]`.
* `768<=c<1024`: padding, ignored on ingress and zero in the returned result.
* No partial rows, alternative packing, suffix, or second ciphertext in this slice.

`M[j]=1[c<768]`. Ingress is `A=M*a0`; finite nonzero padding cannot affect a row.
The clear reference rejects nonfinite input/result values without publishing
partial output; finite-input overflow is failure, not a new saturation policy.
This is not a certified domain for private model activations or CKKS encoding.

Define the global pull rotation exactly:
`Rot_d(x)[j]=x[(j+d) mod32768]` (Euclidean modulo, including negative d).
For `d in {1,2,4,8,16,32,64,128,256,512}`:

```text
L_d[j] = 1[c < 1024-d]
H_d[j] = 1[c >= 1024-d]
RowRot_d(x) = L_d*Rot_d(x) + H_d*Rot_(d-1024)(x)
```

Each selected source remains in the SAME row: it is `r*1024+(c+d) mod1024`.
Global rotation wrap at slot32767 cannot introduce row31 into row0. Masked-out
sources may cross rows, but do not contribute in exact algebra. CKKS mask leakage
is a separate error obligation, never presumed zero by this proof.

`Fold(x)` starts at `t=x`; for ascending d, replace `t` by `t+RowRot_d(t)`.
After ten stages every slot in a block contains that block's full1024 sum.
Ingress padding is zero, so this is precisely the 768-entry sum. No nonexistent
"log2(768)" stage count or normalization by1024 is allowed.

## 4. Exact ordered operations and consumption

Freeze this public instruction list, with no algebraic folding/fusion:

1. `A=M*a0`.
2. `s=Fold(A)`; each fold stage executes **in this order**, preserving old `t`:
   `u=Rot_d(t); l=L_d*u; w=Rot_(d-1024)(t); h=H_d*w; b=l+h; next=t+b`.
3. `x=C768*A`, where the public vector `C768[j]=768`.
4. `z_raw=x-s`; `z=M*z_raw` (centering must not populate padding).
5. `q=z*z` (one unary square, not a different polynomial).
6. `v_full=Fold(q)`, with the same stage ordering.
7. `v=M*v_full`: private energy repeated in active slots; padding zero.

Exactly **126 output-producing vector operations**:40 global rotations,
44 public products (40 fold masks, three M products, one C768 product),
41 add/subtract operations, one square. There are22 distinct public vectors:
M, C768, and ten ordered L/H pairs. Capture/borrow them immutably; no per-row
tables, secret-index lookup, adaptive schedule or fresh encryption of constants.

Source identities are fixed public ordinals, with ordered operands and exact
signed rotation offsets. Keep original input through completion. Retire u after l,
w after h, l/h after b, old t/b after next; retain A through step3. Retire x/s
after z_raw, z_raw after z, z after q, q through the first second-fold use, and
v_full after v. Return only the private v owner/view, never s/z/q debug output.
Failure/cancellation publishes no partial result. Any caller-retained view keeps
its FULL original backing and charge; no apparent slice-size credit.
Diagnostics contain public operation/shape/error codes only, never private
intermediate values. Clear tests may inspect only their own public literals.

### Levels and keys: known demand, unknown physical mapping

The abstract multiplication-dependency depth from ciphertext×ciphertext products
is **one**. This is NOT an adopted CKKS level consumption: encoding masks/768,
plaintext products, alignment, rescaling, relinearization and rotation switching
must have actual capability rows before encrypted lowering. In particular,
"mask is0/1" does not establish an exact scale-one CKKS selector.

Direct execution needs rotation offsets `{d,d-1024}` for those ten d values
(20 distinct offsets modulo32768), same captured key domain, and a legitimate
square/relinearization contract before subsequent rotations if the carrier
requires it. No conjugation or refresh is requested. Key decomposition,
scale/level/error at EACH output, actual Q/P primes, secret distribution and
key bytes remain **unknown**, not chosen here. Larger physical SIMD extents
need a new proven map; logical S must not silently become a backend slot count.

## 5. Proposed finite preconstruction envelope

These are proposed local ceilings, not current permission, measured peaks or a
raise to any existing cap. Caller/ambient limits must ALSO accept them; otherwise
stop. No materialize-to-discover-fit or retry with more capacity.

| Resource | Exact bound/ceiling and owner |
| --- | --- |
| Input | One immutable original `[f64;32768]` in clear fixtures:262144 payload bytes; parent/actual backing retained, not newly excused by borrowing. No source/model import. |
| Public vector owner |22×32768 f64 cells =5767168 payload bytes. One fixed allocation or caller-owned equivalent; actual capacities, headers and overlap additional. Never clone masks per operation. |
| Clear mutable workspace | Six disjoint full-S buffers =1572864 payload bytes. At most A/current plus three live stage temporaries; reuse only after last use. Whole pool remains charged through returned borrow. No per-operation Vecs or recursive executor. |
| Independent oracle | Two `[i128;768]` row buffers and two `[i128;32]` summaries:25600 payload bytes, before construction. Public small-integer fixtures only; not a new fixed-point NEXUS profile. |
| Core control | At most128 instruction records,128 value locations, six workspace-slot states,22 mask descriptors, ten shift records; one invocation cursor/work ledger, original input/mask refs, mutation/poison state and typed Result/iterator/argument slots. **65536-byte logical control ceiling**; quote actual concrete layouts before constructing arrays. No unpriced callbacks/closure captures. |
| Caller graph/trace metadata | At most128 operations/capabilities/steps, one input/output,149 owner declarations,22 persistent public owners, ID lengths≤64, input arity≤2, shape rank2. **8 MiB actual caller metadata envelope**, including nested String/Vec capacities/headers and construction copies; no parser or dynamic registration. |
| Aggregate clear/metadata owner cap | **64 MiB**, including original input, all above owners, constructor origin/destination/copy overlaps, validator workspace, diagnostics and cleanup. Quote both source and destination slots; private allocator/callframe/RSS terms are NOT zero or solved. Existing host/native caps remain unchanged. |
| Native factories/material | **Zero** in the requested implementation slice: no HE build, keygen, ciphertext allocation, RNG, checkpoint, model, backend installs or execution. |

Six buffers are a ceiling, not a free alias proof. At the largest first-fold
overlap A/current/l/w/h are live; inputs cannot be overwritten. The immutable
original a0 remains additional. The constructor must quote bootstrap/actual
control and owner capacities BEFORE input/mask/workspace/metadata factories;
use caller-preallocated storage or fallible direct heap initialization, not
large temporary stack arrays subsequently boxed/cloned outside that quote;
errors retain guards through payload/diagnostic cleanup. Source capture lifetime
must cover both quote and evaluation; no relabelled replacement arrays.
Pre-existing input/storage needs its original upstream admission; this helper
cannot retroactively admit its acquisition or shrink it to the visible slice.

### Clear owner phases and deterministic last-use policy

Allocate/admit the six-buffer pool once. Number its full-S buffers P0–P5;
`A` starts in P0. Every operation writes to the lowest-numbered free buffer,
distinct from all live inputs and retained values. The instruction cursor and
six slot states are the only location map; no per-operation allocations.
"Free" here means reusable within the still-charged pool, never allocator credit.

| Phase | Live values and exact release point |
| --- | --- |
| Ingress | Original a0 remains immutable/retained. Write A to P0; retain A through x. |
| First fold | Initially t aliases A without a copy. For each d: allocate u, then l; retire u after l; allocate w, then h; retire w after h; allocate b while t/l/h remain live; retire l/h after b; allocate next while t/b remain live; retire b and old t after next, except pinned A. Final t becomes s. |
| Centering | Allocate x with A/s live; retire A after x. Allocate z_raw with x/s live; retire x/s after z_raw. Allocate z with z_raw live; retire z_raw after z. Allocate q with z live; retire z after q. |
| Second fold | t initially aliases q. Apply the same stage order, with no pinned A; retain q through all its first-stage uses and retire it only after first next is complete. Final t becomes v_full. |
| Output | Allocate v with v_full live; retire v_full after masking. Return only a borrow of v, retaining the whole six-buffer pool and original-owner obligations until the borrow ends. |
| Failure | Publish no output view. Poison the invocation; retain pool/input/public-vector guards through cleanup. Cleanup failure yields no release credit or reusable invocation. |

At the first-fold peak A/t/l/w/h or A/t/l/h/b occupy at most five distinct
buffers; six is the fixed conservative pool ceiling. Reuse must follow these
last uses even if an optimizer notices aliases or algebraic shortcuts. Runtime
buffer-state assertions and independent lifetime cases establish this clear
executor property, not a ciphertext-owner certificate.

### Work accounting, before factories

Use separate named units, not a fictitious CPU budget:

* Array-access envelope per invocation: rotations80S, public products132S,
  add/sub123S, square2S =337S reads/writes. Add126S result checks, input scan S,
  mask initialize/validate/cleanup66S and workspace initialize/cleanup12S:
  **542S=17760256** deterministic units. Include failure cleanup, not success only.
* Per invocation ceiling **2^25** units, with actual separate control/geometry
  and dense-oracle counters (each≤2^20). At most six full-map acceptance fixtures;
  cumulative clear/oracle/control work **≤2^28**, never reset/refunded per row or
  after failed attempts. At most one process/thread,30 seconds,1 MiB output;
  timeout records incomplete failure, never an acceptance result.
* Shared SSA metadata is a DIFFERENT existing envelope. With one capability per
  operation,85 unary/41 binary, shape `[32,1024]`,149 owners/22 persistent owners,
 125 releases and retained a0/final v, `trace_preflight` counts **R=7774** records.
   Inventory:553 top-level +11 input-state +1931 operation +4902 capability
   +377 trace/release records. Existing formula requires
   `64*R^3+256*R =30068627962880` work quanta and
   `256*R+1024 =1991168` validator payload bytes. These are conservative metadata
  quanta, NOT that many executed instructions or ciphertext bytes. Quote them
  independently before graph/validator construction; do not substitute the slot
  counter, use usize::MAX or raise an ambient limit to get a pass. A changed
  record inventory requires a recomputed quote. No search/solver work is included.

The owning caller must reject an unknown capacity, overflow, short buffer,
incompatible lifetime, work exhaustion or incomplete manifest before respective
factories. Count original allocations and transition copies, not just the six
working arrays. Existing trace preflight does not itself price its caller's
graph construction or native dependencies; no full-admission claim follows.

### Shared preconstruction contract (proposed, not an existing ticket)

Extend/extract the existing `encrypted_level_schedule::trace_preflight` counting
seam in the compiler. Proposed names below describe one method-neutral contract;
do not implement another NEXUS-specific traversal, quote formula or budget owner.
Core row-fold algebra remains independent of the compiler and of HE manifests.

* `PublicCiphertextConstructionManifest<'a>`: an immutable borrowed count/length
  manifest, not a graph or ciphertext. Includes top-level lengths (149 owners,
  22 persistent,1 input,126 operations,126 capabilities,1 output,1 retained input,
  126 steps), every operation/capability arity, every state shape length (2),
  every release-list length (sum125), and selected execution variant. Counts are
  checked against §5 ceilings; no parser or caller-selected recursive type graph.
* Nested allocation descriptions enumerate each actual String and Vec backing,
  including state scheme/key/layout/scale/error identities, value/owner IDs,
  operation references, capability/error-assumption IDs, execution cost IDs,
  shapes, releases, outer arrays and trace storage. Each records initialized
  length, maximum original capacity, element layout/alignment and a bounded
  construction/live interval. Every string has length AND capacity at most64;
  Vec capacity is bounded by its declared record/arity/shape ceiling. Inline
  headers/enum payloads use the actual target's `size_of`/alignment; they are
  counted within their containing allocation, not charged twice as separate heaps.
* At most8192 backing descriptions and256 construction phases. Source/staging,
  destination, temporary copy and destructor/diagnostic storage get separate
  entries whenever simultaneously live. No `clone` or growth may create an
  unlisted allocation. The manifest's own original backing, fixed quote scratch
  and capture refs are retained and charged too. Its construction uses an already
  admitted caller-owned metadata region; it cannot allocate itself on an unissued
  quote. That upstream acquisition must fit the same8 MiB metadata/64 MiB total
  ceilings. Bootstrap control/quote scratch remains within65536 bytes.
* `quote_public_ciphertext_construction`: allocation-free checked pass over these
  bounded borrowed descriptions; reuse the shared trace record/work formula.
  Return `PublicCiphertextConstructionQuote` with record count, trace work,
  validator payload bound, caller metadata peak and aggregate simultaneous-owner
  peak, plus a borrow binding it to this immutable manifest. Quote construction
  and phase aggregation count against the separate ≤2^20 control counter; no
  allocation/search is permitted to discover the required quote.
* Quote is evidence only, not an execution/admission token. The owning caller
  reserves via its existing shared-allocation/phase owners BEFORE graph, masks,
  workspace or validator factories. Each phase checks unchanged captured lengths,
  capacities and lifetime bindings against that quote and ambient limits. Either
  use preallocated original owners, or a factory with a proven finite original
  allocation/temporary bound. Unknown allocator rounding/copies means refusal,
  not optimistic `Vec::with_capacity` followed by retrospective pricing.
* Proposed typed refusal variants: `UnknownOriginalCapacity`,
  `IncompleteConstructionManifest`, `ArithmeticOverflow`, `ShortStorage`,
  `MetadataWorkRefused`, `MetadataStorageRefused`, `ControlWorkExhausted`,
  `MissingAmbientTraceAdmission`, `LifetimeMismatch`, `Poisoned`.
  Return public codes without allocating an error string. Failure before admission
  invokes zero graph/storage factories. Failure after a permitted phase begins
  retains its guards through cleanup; it does not refund work or retry with more
  storage. Existing validator String errors need separately prepaid bounded
  diagnostics; this contract does not silently alter its legacy API.

There is no public construction ticket in current source. The future shared
implementation must check graph lengths against the same manifest and retain
ordinary `trace_preflight` as validation, rather than trusting the quote to skip
validation. Unknown fields and mutation between quote and use fail closed.

### Separate metadata admission: full trace remains conditional

For this fixed full manifest, set a **local rejection ceiling**, not a grant:
7774 records,30,068,627,962,880 metadata work quanta and1,991,168 validator payload
bytes; caller metadata≤8 MiB and aggregate owners≤64 MiB remain additional.
The effective ceiling is the minimum of these and an independently supplied,
already authorized finite `PublicCiphertextTraceCapacity`/owner admission. Never
initialize ambient admission from the computed requirement merely to make it fit.
Unrepresentable `usize` arithmetic rejects on that target.

Existing library `PublicCiphertextTraceCapacity` is caller-supplied; inspected
source supplies no universal production allowance for this N-N full trace.
Existing tests' `usize::MAX` is not permission and must not be copied. This packet
does not establish a pre-existing finite caller grant of30,068,627,962,880 quanta.
Therefore full synthetic graph construction/validation is **conditional**:
missing grant returns `MissingAmbientTraceAdmission`; a smaller existing grant
returns `MetadataWorkRefused`, before graph factories. Implement/test that refusal
using the borrowed manifest. Do not split the full graph into smaller passes and
claim whole-trace acceptance, relax the cubic formula, or raise any ambient cap.
A shared algorithmic improvement, if required, needs its own reviewed scope.

Keep a separate cumulative metadata-validation ledger with the same local
30,068,627,962,880-quanta ceiling and any tighter existing cumulative grant.
Debit each validator invocation's complete shared bound before entry; never
reset/refund on failure. This permits at most one full-manifest validation, not
one per clear fixture. If smaller validations already consumed this allowance,
the full trace refuses rather than topping it up. Preflight/count-only refusal
checks use the bounded control ledger and never execute an unadmitted validator.

The clear slot program and dense oracle can complete independently under their
own admitted counters. Smaller synthetic consistency cases may reuse existing
finite caller limits, but count as local lifetime/consistency evidence only. The
delivery must report full-trace admission/refusal separately from clear parity;
full-trace refusal does not authorize an unpriced substitute or HE execution.

### Logical synthetic owners are not the clear buffer pool

The149 owner declarations are127 distinct logical value backings (original a0
plus126 operation outputs) and22 persistent public-vector owners. The trace
releases125 intermediate handles, retaining a0 and final v. In a synthetic
manifest, assign explicit fresh logical output owners and honestly synthetic
state/cost/error assumptions; their conceptual overlap is checked by the existing
SSA validator. These declarations do not allocate127 clear arrays or establish
native CKKS capacities. Do not alias their IDs onto P0–P5, reuse released owner
IDs, or treat logical trace releases as physical pool refunds. The actual clear
executor keeps all six pool backings charged even when slots are reusable.

No real ciphertext manifest can currently be supplied: actual levels, keys,
errors and backing obligations remain unknown. Real lowering rejects rather
than borrowing synthetic `Some(0)` values or interpreting a public ID as authority.

## 6. Independent acceptance, not another compression oracle

Proposed tests (none run here):

1. Independently authored dense768 row sums, `z_i=768*a_i-s`, squared-energy,
   plus exact identity `v=768^2*sum(a_i^2)-768*s^2`, against EVERY active output.
   Public integer fixtures in[-8,8] keep these reference operations exact in f64;
   that fixture choice supplies no activation-domain/CKKS certificate.
2. At most six full-map cases: mixed zero/constant rows; dense signed rows;
   impulses at c0/c767; isolated adjacent rows with different data; finite
   garbage padding; explicit n768-versus-padded1024 and mask negative controls.
   Verify all32 rows, all256 padding slots/row and ordered operation/work counts.
3. **Centering-mask counterexample:** one active row is all1. Correct z/v=0.
   Without `M*z_raw`,256 padding values become−768, adding **150994944** false
   energy. Ingress masking alone is insufficient.
4. **Boundary counterexample:** row0 active entries1, row1 zeros. An unmasked
   global1024 fold at row0,c767 sums1 instead of768. Missing H branch/incorrect
   rotation sign must fail, not be hidden by checking row heads only.
5. Wrong shape/S/n, nonfinite/overflow, short/exact workspace, work one-short,
   failed construction/cleanup, reused poisoned state, borrowed-source mutation
   and escaped workspace views. Factory observers establish pre-entry refusal;
   no expected result calls the new fold/compression implementation.
6. Reuse shared SSA independent owner-interval tests for original/derived aliases,
   retained a0, output/scratch overlap and release order. Synthetic carrier
   fixtures must be visibly synthetic, never a fake CKKS manifest. Unknown real
   error/scratch/key/level rows reject; no dummy0 or fabricated `Some(...)` values.

## 7. Unresolved gates, not defaults

| Field | Blocking consequence |
| --- | --- |
| Exact native slot/ring mapping and owned input provenance | No encrypted input import, coefficient→slot alias or evaluated NEXUS identity. The map above is a proposed clear/public contract. |
| Ordered Q/P, scales, mask encodings, per-output error, rescale/relinearization and key owner capacities | No real `PublicCiphertextState`/capability can be filled truthfully; no encrypted execution/level claim. Preserve unknown values and reject real lowering. |
| Real transient native/backing capacity and construction/recovery overlaps | No HE admission. Logical public-owner trace is not an allocator, RSS or erasure proof. |
| Actual caller metadata/work quote fits current limits | Mandatory before that phase; failure does not authorize a cap increase or an unpriced fallback. |
| Eq.(5) inverse/zero-variance/epsilon, gamma/beta and source checkpoint contract | Full LayerNorm and trained-decoder completion remain blocked after this prefix. |

## 8. Separately selectable exact implementation request

> **N-N/PREFIX768-32x1024-v1 — implement the bounded CLEAR/STRUCTURAL slice only:**
> GPT-6.1 Sol adds or extracts a reusable caller-buffered row-fold schedule and centered-energy
> prefix specified in §§3–5; retain original immutable inputs, exact operand order,
> padding/row isolation and preconstruction work/owner gates. Extend the existing
> shared trace construction-resource seam only as needed, without duplicating
> its validator or changing legacy limits. Add the independent
> acceptance tests in §6 and connect to existing SSA/owner contracts only with
> complete supplied synthetic manifests and independently authorized finite
> metadata admission (synthetic fixtures remain synthetic). Implement explicit
> pre-factory refusal for the full trace when that grant is missing or too small;
> full-trace refusal is reported separately and does not block clear-only parity.
> Astra reviews contracts and results. Keep the
> 64 MiB local owner ceiling,2^28 cumulative clear-work ceiling and all existing
> tighter/global caps. Missing actual layouts/capacities or metadata admission
> stop the phase; never materialize to discover fit. No native HE build/execution,
> parameters, keys/RNG, model/source import, inverse/epsilon, SDK/default routing,
> reveal, cap increase, full LayerNorm or source-faithful reproduction claim is
> authorized. Return the unresolved encrypted fields for a separate decision.

**Exact user question:** Approve this bounded public-slot-program, clear-oracle
and conditional synthetic-trace implementation by GPT-6.1 Sol, with Astra review,
including ordinary clear/metadata correctness tests within §5's admitted limits;
or defer? This approves no encrypted profile, native HE build, ciphertext/key
authority or increase to existing admission limits. In particular, approval is
not the missing ambient full-trace work grant.

This wording is a **proposal awaiting user selection**, not an implementation
permission inferred from D37. Suggested decision-sheet link, not applied here:
`[N-N implementation packet](nexus-centered-energy-prefix.md#8-separately-selectable-exact-implementation-request)`.
