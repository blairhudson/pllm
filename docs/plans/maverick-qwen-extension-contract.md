# MAV-QWEN-CONTRACT-JOIN

## Status and source anchors

Contract proposal for independent review, based on Root `23d691c6`.
Common-decoder extension direction was selected previously; the continuation
below remains an unapproved contract proposal, not implementation permission.
This document implements no Rust, source acquisition, preprocessing, verification
or trained execution.
It is not a new FP8 interpretation, tolerance, security certificate or GKM route.

Source-backed premises (links identify existing definitions, not new authority):

1. [Model-neutral operations, graphs and DecoderPlan](../../crates/pllm-models/src/lib.rs#L250):
   operation IDs, producer inputs, shapes, attributes, prefill/decode and state edges.
2. [Existing declarative dense lowering](../../crates/pllm-models/src/lib.rs#L1530):
   Qwen2 has QKV bias and no Q/K norm; Qwen3 has the opposite declarations.
3. [D31/D32 exact numeric map](../research/maverick-numeric-source-recovery.md#32-exact-scalar-and-matrix-rules)
   and [original local schedule](../research/maverick-numeric-source-recovery.md#33-full-local-schedule).
4. [RAW retained source owner](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L37)
   and [allocation-free codec limitations](../../crates/pllm-compiler/src/verified_source/raw.rs#L1).
5. [Existing publication admission](../../crates/pllm-compiler/src/delegation_publication_inventory.rs#L440):
   actual AllocationPlan, ordered matrix/full-head coverage, explicit blocked status.
6. [Atomic request verification](../../crates/pllm-core/src/maverick_batch/request.rs#L21)
   and [D45 scope](../../paper/methods/DECISIONS.md#d45-maverick-atomic-two-batch-native-composition).
7. [Common checkpoint pin](../../examples/papers/common.py#L6):
   Qwen/Qwen2.5-0.5B-Instruct at `7ae557604adf67be50417f59c2c2f167def9a775`.

## Validated method contract over the complete graph

Proposed validation consumes the genuine immutable `DecoderPlan`, not a rebuilt
eight-operation template, model-name switch or caller-supplied semantic digest.
Both graphs must account for every operation, edge, state input/output and terminal
output. IDs are unique; each input resolves to an earlier producer or explicitly
declared graph/request/state input. Reject missing, foreign, cyclic or reordered
dependencies and inconsistent shape/layout/attribute/state declarations.

Derive roles from validated operator semantics, attributes and producer topology,
not names such as `layer.0.q_linear` or `model_family`. Record each role's source
tensor, ordered input/output coordinates, numeric rule and client/server boundary.
Unknown operators, unknown attributes or unsupported combinations REFUSE before
numeric work. An enum variant existing in `ModelOperator` is not method support.

- Delegated candidates: validated projection and full-output-head matrix products.
  Bind each complete logical dot, source orientation, row/token order and scale
  indices. Splitting publication/work must not introduce partial-dot BF16 rounding.
- Client-local candidates: token lookup, reshape, RMSNorm, RoPE, attention score/
  scale/mask/softmax/value operations, residuals, SiLU, multiply and bias addition.
  Each requires its own exact rule; local placement supplies no correctness proof.
- Last-token selection, greedy selection, feedback and KV append require explicit
  output/state contracts. No hidden delegated lookup, vocabulary truncation,
  attention substitution or silent host fallback for an unsupported operator.
- `CacheSuffix` is a separate functional role, not established by KV append alone.
  Bind ordered inputs `[append, input.positions, input.attention_mask,
  input.sequence_lengths]`, `axis=2`, `maximum_sequence`, and
  `semantics=visible_valid_prefix`; bind `absolute_write_positions_input`,
  `padding_mask_input` and `valid_lengths_input` to those exact roots, and preserve
  matching Key/Value `state_kind`. Visibility must include current valid entries
  without exposing future, padded or unwritten cells; the name permits no eviction.

QKV bias presence and Q/K norm absence follow actual declarations and edges.
Validate separate bias tensor dependencies inside linear attributes even when
they are not standalone operation nodes. No family-named Rust module is needed.
Retain full vocabulary, actual prefill length, decode position and grouped-query
head mapping. Prompt rendering and admission use actual source-rendered tokens;
neither eight tokens nor a measured common-checkpoint prompt count is established.

## Separate original and common numeric identities

Original D31+D32 identity remains the pinned FP8 Qwen3 candidate, not common BF16.
D31 uses `roundEven(256*x)` (eight fractional bits), canonical BabyBear residues,
complete-dot centered decoding and division by 65,536 directly rounded to BF16.
No intermediate F32 rounding, partial-sum conversion, saturation or tolerance.
Prove signed no-wrap and host reduction safety for every admitted complete dot;
integer range failure rejects, not a reason to change encoding or field.

D32 exactly promotes each stored BF16 block scale to F32 before the specified F32
multiply with decoded E4M3FN; preserve original 128x128 scale axes and orientation.
It recovers no earlier scale precision. These rules do NOT turn BF16 projection
weights into FP8 or authorize invented scales. Original BF16 head treatment does
not by itself authorize a BF16 rule for every common projection.

Common identity must separately bind revision, tensor extents/dtypes/aliases,
actual config, complete graph, numeric derivation and workload/state semantics.
Pin alone proves none of those tensor facts. Proposed common BF16-to-Q8 linear
derivation must name exact source-value promotion, rounding and complete-dot output
rule and receive numeric review; no re-quantization/import repair is implicit.

Original Norm uses epsilon F32 bits `0x358637bd` and cast-before-gamma order:
F32 square/sum/mean/epsilon/rsqrt, cast normalized value to BF16, then gamma multiply
with specified rounding. Original SiLU uses correctly rounded F32 of exact-real
SiLU then BF16; original attention has ordered, non-fused local reductions.
These are original-profile facts, not evidence stock common kernels match them.

Existing dense RoPE declares
`coefficient_profile=pllm.numeric.rope.q30.libm.v1`; preserve that explicit numeric
identity (and any declared `frequency_scaling` with its changed profile), rather
than silently treating it as original BF16 RoPE or approved common numerics.
Compatibility and coefficient/rounding derivation remain unresolved review gates.

Common derivation still owes independently checked source facts for epsilon value
and rounding, gamma dtype/order, biased-QKV cast/add order, SiLU and attention
rounding, RoPE frequencies/layout/positions, and KV stored dtype/append semantics.
Existing lowering supplies epsilon attributes, bias presence and producer topology,
not floating-point execution order. Do not invent numeric parameters or inherit
Qwen3 per-head norms when absent. Unsupported or unresolved rules block admission.

## Custody, admission and request-global acceptance

RAW22 join must retain the actual original immutable plan, independently provided
host pins, retained snapshot directory/reader, request-token owner, genuine producer
and original work ledger. Host pin provisioning is a trust premise, not publisher
authentication. Equal hashes, metadata, coordinates or cloned plans do not confer
ownership, authority or a replacement budget/root.

Before traversal/allocation, charge bounded graph visits, producer lookup/storage,
attribute parsing, contract backing, numeric/source passes and retained controls on
that original root. Price original buffers, directory/reader, contract, publication
windows, output/KV retention and construction/move/replay overlap together. Include
all-path cancellation/cleanup/retirement without refund; no implicit free traversal,
ledger reset, speculative allocation or inventory digest as an admission permit.
Logical counters do not close allocator/RSS, physical-byte, stack or backing holds.

Reuse `admit_maverick_batch_publication` for its actual geometry/coverage/allocation
checks, not as decoder/source/numeric authorization. It currently requires native
fixed geometry, one live window and complete matrix/head coverage; common shapes
may refuse. No tiling, cap increase, new owner or general decoder support follows.
Any future window plan must bind partitions, global source/scale indices, originals,
peak retained overlap and spent attempt state without resetting original limits.

Reuse `Client::begin_request_verification` and request-global finishing semantics:
fix ALL original replies/encodings before ANY primary challenges; fix ALL auxiliary
replies before ANY secondary challenges. Accept body/head only atomically after
every required check. No tentative output, public intermediate, cache commit or
token escape before global acceptance; failure/cancel retires private state.

D45 is two independent raw batches, not dependent decoder continuation. Publishing
an accepted intermediate to feed another batch cannot establish a request-global
join. Dependent layers require a separately reviewed private provisional lifecycle,
with all-original commitment timing reconciled before challenges. Existing API does
not supply that lifecycle; do not claim a complete decoder from an API-name join.

## Extension DAG and independent gates

### Reviewed original-request continuation (proposal only)

The two Astra reviews constrain the next proposed approval to the **first
projection**, not a complete decoder bridge. They confer no implementation,
source acquisition, preprocessing, native execution or checkpoint permission.
Common-model extension gates below remain separate from this original profile.

**Authentic input and source join.** Consume the whole original
`RawFirstNormActivation { words, norm }`, never detached words plus reconstructed
metadata. Retain gamma/file/extent/reservation → original charge/operation/producer
→ `lookup.buffer`/`source.input` → tokens/origin/graph/spec/charge/reader → original
root payload (files, directory, identity, header, reader_charge), together with the
original controller and ledger. Bind original norm to the selected `Linear`
operation by authentic graph traversal, not operation-name parsing. Bind original
tensor, scale, global axes/extents, FP8/BF16 scale mapping, publication and slot.
Equal values or a guard cannot replace any original authority edge.

**Fixed numeric boundary.** Original D31/D32 input is `[1,8,2560]`, positions
0–7; Q is `4096×2560`, K and V each `1024×2560`, for all eight positions. Preserve
`xq = RNE(256*x)` and
`Wq = RNE(256*R32(E4M3FN(w)*promoteBF16(global S[i/128,j/128])))`, without clipping;
reuse existing `field_float` helpers. No changed field, scale axes or tolerance.
Existing `window_plan` proposes `256×128` windows with genuine 128-row tails;
it is incompatible with current `Client`'s hardcoded `64×64`, B=2, noise 165 and
16 attempts. TX164 metadata is neither geometry nor noise authority.

For disjoint, exhaustive column partitions, demask partials in BabyBear and SUM
modulo the field to obtain each complete dot. Only then center and directly
BF16-round the complete integer divided by 65,536. Partial rounding is invalid:
`65,792/65,536` rounds to BF16 1 and `1/65,536` to BF16 `2^-16`.
Their exact sum is `1+2^-16`; BF16-rounded addition gives 1, whereas the complete
`65,793/65,536` rounds to BF16 `1.0078125`. A complete integer within
`±1,006,632,960` gives the correct centered result; partial no-wrap is unnecessary
for the modular identity. Honest complete-dot bounds, host-safe quantization/
field accumulators, and malicious arbitrary-canonical-reply totality through the
whole local graph are separate obligations. None follows from partition equality.

**Fixation and conditional security.** Make every original reply and encoding
immutable before ANY primary challenge, and every auxiliary reply immutable before
ANY secondary challenge. Preserve original request-global bindings and horizon;
never reset by window or reconstructed object. Existing four-raw-claim request
verifier supplies reusable barriers, not a full decoder proof. Early-primary
failure is concrete: choose nonzero `delta` in the kernel of the two
`(Gy*e)^T` rows (dimension at least 62 in dimension 64), then alter an unfixed
reply to `y+delta`. Both original checks and truthful auxiliary/secondary checks
can pass. Bind raw fields, not decoded BF16: residues 65,536 and 65,537 both
decode to 1.

Actual distance, certification failure, dual-LPN and adaptive-transcript premises
remain unresolved. With certified integer distance `d`, length `N`, field `p`,
set `b = min(1, C(N-d,31)/C(N,31) + 1/(p-1))` (impossible binomials are zero).
Only conditional on the required fixed-vector and independent-challenge premises
is the two-challenge bound at most `b²`. Union-bound over actual request original
and auxiliary claims, plus actual code-certification failure. Do not fabricate
quantitative whole-request security from defaults or this conditional formula.

**Prepaid admission.** Prepay ENTRY, refusal and cleanup before the first runtime
quote on the retained original ledger. Existing primitive quote 24 covers only
existing `raw_owner` scope, not a new projection. Component request quote currently
does 45 checked arithmetic operations (11 rows × 4 plus peak addition), besides
setup/loads/control/publication; it currently has no original-ledger debit there.
First claim with `n=64,L=512,k=165` has arithmetic-only 12,160 additions and
11,584 multiplications, excluding other work. `Sparse::sample` now reaches the
authored 32-fresh-u32 cap in `maverick_batch/arithmetic.rs::below`; this supplies
neither a funded `W_rng` quote nor a bound on entropy-provider time.

**Prospective funding proof; administrative sequencing released 7 November 2026.** An
already EXECUTED new payment is not required to author a prospective funding DESIGN
proof; requiring it would be circular. The current new-scope execution-paid witness
is still absent. The former mandatory DESIGN proof → explicit scope/rate/route/
authoring adoption → implementation → deployment witness sequence is historical:
the user's 7 November 2026 administrative hold release permits source/test authoring
and releases funding-first sequencing and Native holds. It does not establish
actual prepayment, scope/rate/issuer adoption, custody, security or qualification.

The existing five-element execution-work fold is
[`raw_first_norm.rs:53–54`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L53),
not the ten-element `norm_spec_demand` fold at
[`raw_clear_producer.rs:469–471`](../../crates/pllm-compiler/src/replicated_decoder/raw_clear_producer.rs#L469).
Replacing its existing operand with compile-time `4096+K` may avoid extra pre-debit
runtime actions ONLY after a whole-changed-path and failure-path audit. This is
neither descendant-scope authority nor an adopted price.

| Obligation | Existing fact | Prospective proof required |
| --- | --- | --- |
| Earlier combined original debit/order | Original admission precedes covered traversal; existing norm route combines demands. | Show the complete changed combined demand is debited before every newly covered action, including refusal. |
| Explicit descendant scope | `OriginalReservation` is TokenLookup-only (`raw_clear_producer.rs:268–275,312–320`). | Explicitly adopt each descendant operation; demand size or budget membership is not an operation license. |
| Canonical consuming retained-root issuer route | `RawFirstNormActivation` retains `norm` (`raw_first_norm.rs:11–14`); actual `MavClient` (`Client`, `protocol.rs:228–232`) has no `OriginalReservation` linkage. | Define and adopt a consuming issuer route retaining the genuine original root; no fabricated linkage or new root. |
| Complete K and resident bytes | Existing fixed operands are not a new-scope inventory. | Cover all entry/check/quote/refusal/publication/cleanup work in K, with separate resident-byte accounting. |
| Original custody through new failures | Existing producer/reservation custody is scope-limited. | Preserve original custody through every new failure path, attempt burn, no refund and no replay/reset. |

For a selected 64-bit target, `U < 2^40` and `K <= u64::MAX − U` are conditional
arithmetic premises for the WHOLE affected aggregate, not an adopted unit rate or
proof that budget outcomes are preserved. If U bounds only the local fold, bound
all subsequent `include` totals separately; a local bound cannot discharge them.
An increased fee may refuse an attempt that the old fee admitted. Prove overflow,
refusal and cumulative-exhaustion behavior rather than claiming outcome equality.

Complete funding proof and explicit adoption are NOT yet established. Nothing here
transfers bootstrap 24, creates a new root, adopts a fee/rate, or grants a permit or
source authority by itself. Current execution-paid witness remains absent;
funding, scope, issuer, custody and deployment evidence remain unproved, without
reinstating the administrative holds released on 7 November 2026.

#### Conditional native entry prefix: retained parent, no new byte reservation

This Astra-reconciled source/contract DESIGN candidate is not an adopted canonical
route or completed funding proof. The actual Root/CoreClient linkage remains absent.
It proposes a retained-parent entry prefix, not permission to execute native work.

- **Existing admission, changed constants only.**
  [`raw_first_norm.rs:43–56`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L43)
  has a controls fold ending in 2048 and five execution-work operands, including
  controls and 4096. Replace those compile-time operands by `2048+B` and `4096+K`.
  K and B are nonnegative, unpriced and unadopted. Add no fold items, layout types
  or reserve call. Controls already enter work: the exact delta is **work K+B,
  resident bytes B**, not K+2B. A larger demand may change admission failures.
  [`raw_clear_producer.rs:504–506`](../../crates/pllm-compiler/src/replicated_decoder/raw_clear_producer.rs#L504)
  combines the existing norm demand and extra demand into one original reservation;
  its [TokenLookup scope](../../crates/pllm-compiler/src/replicated_decoder/raw_clear_producer.rs#L268)
  does not thereby acquire descendant authority.

- **Whole affected aggregate, not a local-fold proof.** Write
  `Qw = Nw+Ew+Rw+Aw+K+B` and `Qb = Nb+Eb+Rb+Ab+B`, where N is norm demand,
  E is old execution demand, R is reservation overhead and A is primitive-admission
  overhead. Check every intermediate `usize` operation, conversion to `u64`,
  cumulative ledger sum, current 64 MiB limit and actual budget admission, including
  `chargeMetadata`; a local fold bound cannot discharge later totals. On a fixed
  64-bit target, a whole-old-aggregate `U < 2^40` lemma is arithmetic only, not an
  admission guarantee; each changed sum still needs its own representability and
  budget checks. The initial bootstrap fee 24 stays in its existing scope and is
  not transferred to this prefix or to a later stage.

- **Placement and fresh retained authority.** The activation constructor is at
  [`raw_first_norm.rs:147`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L147),
  before the norm loop. Insert exactly one new private borrowed-helper call after
  the successful `reader_failed=false` reset at lines 159–162 and before
  `Ok(output)` at line 163. Earlier `reader_failed=true` makes
  [`reader_payload`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L599)
  reject access. Preserve existing public functions, result/activation types and
  all pre-debit paths. The schematic below is not implementation or a new API:

  ```text
  execute consumes norm; construct protected output; finish existing norm loop
  finish existing success reset; release its root guard
  entry_result = private_borrowed_entry(&output)       // exactly once
  propagate entry_result only after every helper guard has ended
  Ok(output)

  private_borrowed_entry borrows protected output.norm/output:
    lock its genuinely retained root; handle poison without retaining its guard
    obtain fresh reader_payload; borrow SAME payload.header.original/controller
    result = original.prepay_primitive_work(&payload.header.controller,
                                           trusted_next_work)
    end payload borrows and root guard, including any PoisonError-contained guard
    return result                                     // no output ownership
  ```

- **Retained-entry trace v1 (conditional design, no helper implemented).**
  - **T1:** The existing norm success-reset guard has ended; output owns the
    activation.
  - **T2:** Borrow the retained root via `output.norm`; enter one root lock attempt.
  - **T3:** On poison, retire the contained guard inside the helper before
    guard-free failure.
  - **T4:** Obtain fresh `reader_payload`; success continues, invalid payload
    leads to guard-free failure after ending the root guard.
  - **T5:** Borrow the SAME payload original/controller and evaluate the
    **UNRESOLVED** trusted-next-work expression. Its derivation, temporaries and
    error paths remain unproved.
  - **T6:** Call existing `prepay_primitive_work` once, only if preceding steps
    succeed.
  - **T7:** Resolve the outcome; end payload borrows and the root guard before
    outward helper error normalization.
  - **T8:** Return a guard-free `Result`; outer Err destroys output, or Ok proceeds.

  Structural multiplicities apply only to this fixed skeleton, excluding
  unresolved T5; no whole-helper count or time bound follows from an arbitrary
  expression. Exact helper Rust syntax, intermediate `Result` storage, error
  conversions and layout remain unfrozen. Typed slot constraints are not B.

  The composition interface distinguishes actual-source, conditional-trace and
  unresolved contributions. Identify events by (site, branch, occurrence, owner);
  separate call-edge contributions from callee-body contributions to avoid double
  counting. Preserve mutually exclusive paths and their event sequences; derive
  storage from lifetime overlap, not a sum of all slot sizes. This trace supplies
   no count vectors, fees, new types, native code or completed proof; the approved
   design-only mutex definitions remain unchanged; administrative holds were
   released on 7 November 2026, not discharged by this trace.

- **One-shot custody, not a new seal.** No `reader_failed` mutation or new sticky
  seal belongs in this helper. The private consuming `execute`, its single caller,
  exactly one helper call, immediate Err destruction of the activation and the
  irreversible RawReader claim jointly supply one-shot behavior; the helper alone
  does not, and no global burn is asserted.
  No retry, fallback success, or panic recovery may retain/reuse the activation after
  helper failure. On unwind, helper locals and all guards—including poison-contained
  guards—must retire before outer output destruction.
  Fresh payload access is required;
  production has no post-seal payload replacement. The binding is genuine private
  parent custody, not budget membership alone. The helper performs no graph/header/
  file walk, new Core construction, FFI, entropy, client factory or byte lease.

- **Use the existing bounded debit, not another accounting mechanism.**
  [`original_source.rs:756–786`](../../crates/pllm-compiler/src/original_source.rs#L756)
  uses a logical ledger transition with a mutex, not `fetch_update` or
  `quote_on_root`; library-internal waiting/retries, physical work and progress are
  not bounded here. Its
  [metadata controls](../../crates/pllm-compiler/src/original_source.rs#L723) include
  the guard and six control-metadata items. Precondition failures occur before
  burn; cumulative exhaustion/overflow burns the ledger and siblings. Success is
  spent work with no refund. All helper-owned guards, including poison-contained
  guards, must retire before error propagation causes outer activation Drop. The
  existing inner debit `map_err(|_| Error::Burned)?` is allowed while the proposed
  outer root guard remains held: it drops the contained poisoned ledger guard,
  without requiring a code fix. The helper must release its root guard before
  output Drop, because RawReader Drop reacquires that root.
  Existing [activation Drop](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L15)
  wipes/frees full output capacity before norm destruction on propagated errors and
  ordinary unwind. Norm gamma retires before its reservation; reservation charge then
  drops before the independently charged producer. This preserves inherited cleanup
  ordering, not certification of opaque owner internals, physical erasure, or
  abort/double-panic cleanup.

- **Logical storage and open adoption.** B names only maximum simultaneously live
  incremental logical slots, expressed through their sizes, not a numeric price;
  existing parent/old slots are not duplicated. Physical native ABI stack and RSS
  remain excluded by the old contract. For this retained-entry design only, the
  **User-approved design-only typed logical boundary — 4 November 2026**:
  distinguish `Mutex<RawHostState>`
  from `Mutex<Ledger>`. Each dynamically entered lock invocation is one logical
  event, whether it succeeds, returns poison or never returns. Actual retirement
  of each `MutexGuard` is separately one event, including a guard contained in a
  `PoisonError`. Ownership moves, destruction of `Result`/`PoisonError` wrappers and
  the underlying unlock are not additional release events for that same guard;
  `into_inner` transfers ownership, not retirement. No release is presumed for a
  nonreturning lock invocation. Cleanup root reacquisition, when reached, is a
  distinct lock invocation, not the original guard's release. Attribution and
  already-paid exclusion still require separate proof.

  Visible project checks, metadata, arithmetic, error mapping and poison cleanup
  remain separate. This proposal excludes library/kernel waiting and retries and
  gives no CPU, RSS, termination or progress guarantee. Deriving and reviewing
  unpriced path vectors is now permitted. Partial source components appear below;
   the whole-prefix vector remains unestablished. Rates, K/B and funding remain
   unproved or unadopted; administrative authoring and Native holds were released
   on 7 November 2026. No completed proof is claimed. Other typed definitions remain current,
  but their composition and rates are open. K must cover the complete
  changed entry/check/refusal/cleanup path; no numeric K/B, next-stage work bound,
  canonical Maverick caller, scope/rate/lifecycle adoption or current paid witness
  is supplied here. Absence of an already executed payment does not block this
   prospective design. The user's administrative release, not this design,
   permits source/test authoring and releases Native holds. Standing Native
   fresh-worker, exclusive-slot, immutable-input and exact-command protocols and
   numeric caps remain in force; funding and security evidence remain unresolved.

#### Prospective fixed2batch integrated funding packet

**NONNORMATIVE prospective fixed native F only.** F is one matching import →
validate → split setup and at most one canonical fixed2batch request, with pinned
native callbacks and the now-authored selected 32-draw cap. This packet changes or adopts
no scope, rates, reservations, Root issuer, cap or helper body. It is not a funding
proof, full-Qwen result, current finite program, security claim or implementation
readiness finding. Funding-first sequencing and Native administrative holds were
released on 7 November 2026; the funding proof and Native qualification remain open.

**Once-owned composition.** Identify each reached event by site, branch, occurrence
and owner: caller, named helper, setup hash, sampler, control and retirement each
own their events once. Expanded bodies REPLACE opaque helper fees; payload
read/write and index projections are INSIDE Entries/Index/View, not additive
charges. For ordinary `Result` paths p and a prospective rate vector λ, define:

```text
Dstage(F, λ) = sup_p λ · [Vreached(p) + Vcleanup(p) + Rout(p)].
```

`Rout` includes each actual unretired owner's future authorized retirement ONCE;
it is not every historically transferred owner. Unwind, abort, nonreturn and
unknown retirement remain UNRESOLVED, not zero. Arbitrary public callbacks,
retries and outside schedules are not bounded by F. No finite checked-u64 quote
follows before composition, rates, ranges, authorization and cleanup proofs.
T5 must be an immutable constant specialized BEFORE execution, not a peek at
material, headers or balance. Entry K pays load/debit/helper outcomes;
`Dstage` is ADDITIONAL stage work, not K or B.

**Restricted normal-success arithmetic cross-check.** These are source field-helper
invocations, not whole work or fees. Expand each dot ONCE; no no-wrap or security
claim follows. Dot/SparseDot totals count reached fold-body add/mul operations;
fold iterations are not additional dot calls or fees.
Sources: [code.rs:116–169](../../crates/pllm-core/src/maverick_batch/code.rs#L116)
and [244–247](../../crates/pllm-core/src/maverick_batch/code.rs#L244);
[protocol.rs:342–356](../../crates/pllm-core/src/maverick_batch/protocol.rs#L342),
[618–663](../../crates/pllm-core/src/maverick_batch/protocol.rs#L618),
[696–715](../../crates/pllm-core/src/maverick_batch/protocol.rs#L696) and
[748–783](../../crates/pllm-core/src/maverick_batch/protocol.rs#L748).

| Reached field-helper scope | Adds | Multiplies | Subtracts |
| --- | ---: | ---: | ---: |
| Code: `4E512 + 4S512 + 4S1024` | 22,528 | 16,384 | 0 |
| Claims: `256 SparseDot512(weight=165)` plus masking | 42,496 | 42,240 | 0 |
| Server: `256 Dot64` | 16,384 | 16,384 | 0 |
| Projection: `8*31*64` | 15,872 | 15,872 | 0 |
| Equations outside syndrome S | 9,472 | 17,160 | 0 |
| Demask | 0 | 0 | 256 |
| **Total** | **106,752** | **108,040** | **256** |

The existing [successful acquisition schedule](maverick-numeric-completion.md#successful-fixed2batch-words-acquisitions-and-conditional-setup-composition)
and [payload extents and retirement](maverick-numeric-completion.md#successful-fixed2batch-buffer-payload-extents-and-retirement)
retain conditional counts: 92 acquisitions, 30,216 initialized words, 1,520 copied
words, and 74 internal plus 18 caller-transferred lifecycle owners. These are
references, not new fees; not all transferred owners remain live. Keep the existing
[32-draw policy](#appendix-partial-unpriced-retained-entry-composition) and
[setup/hash inventory](#source-linked-accounting-review-inventory-ingredients-not-a-complete-lu-tariff)
separate; do not duplicate or sum alternative preprocessing and import paths.

**Chronological outcomes and custody.** Citations in this table are relative to
[`maverick_batch/`](../../crates/pllm-core/src/maverick_batch/).

| Reached phase | Ordering, burn and retained-owner consequence |
| --- | --- |
| Import and setup | Import source validation precedes parts and meter (`publication.rs:134–136`): actual first-inspection funding/SourceAuth GAP. Ordered copies and binding precede consuming validation and one-pair Split. `Arc<State>`/`Arc<Control>` OOM abort is not ordinary Err. |
| Issue | `fetch_update` burns +2 before CAS: counter refusal burns 0, Busy burns 2. Four canonical input checks → one request-allocation quote → 4,096 logical reservation → four noise/claim constructions (`request.rs:77–88`); later Errors retain burn. |
| Begin | All original copies and encodes precede primary draws. Success has eight header/eight canonical FRAMING checks PLUS four transitive encode canonical checks. Borrowed caller replies are not retired by these views. |
| Server | Seen/mode burn precedes canonical checks and Z. Project sparse copies are outside the two-matrix loop; the first Sparse is already dead before second-copy failure. |
| Auxiliary and finish | Fix copies before secondary draws. First nonzero mismatch still reaches the second check; first Err stops. OR rejects before accept; second accept Err drops first outputs. |
| Trace | Two inline `Zeroizing<[[u32;2];4]>` arrays per batch: 16 zero initializations, 16 overwrites. Drop has 16 scalar volatile writes/barriers, conditional on zeroize 1.9; no Words, Vec or capacity pass. First trace is held during the second check. |
| Material retirement | Normal field Drop releases the Memory handle FIRST; Words reservations keep State alive. Caller-borrowed storage is not falsely attributed to this cleanup. |

**Stage Words storage shape.** For the matching imported shape, successful storage
is `2*4096 + 4*(3*512+1024) + 4*32768 + 512*1024 = 673792` u32 words.
Validation scratch has maximum stacked extent `128+1024+1024 = 2176` words
(stacked vector, first, result); it retires before Split. Fresh server storage is
`2*4096 + 2*32768 = 73728`, giving post-split persistent storage 747,520 words;
`673792+2176 < 747520`. A conservative SUCCESSFUL-request Words ceiling is
`747520+30216 = 777736` words, or 3,110,944 bytes. This is not an exact peak,
physical footprint or failed-acquisition bound. Sources:
[publication.rs:137–156](../../crates/pllm-core/src/maverick_batch/publication.rs#L137),
[189–226](../../crates/pllm-core/src/maverick_batch/publication.rs#L189),
[code.rs:90–98](../../crates/pllm-core/src/maverick_batch/code.rs#L90),
[117–144](../../crates/pllm-core/src/maverick_batch/code.rs#L117),
[protocol.rs:250–280](../../crates/pllm-core/src/maverick_batch/protocol.rs#L250) and
[memory.rs:265–302](../../crates/pllm-core/src/maverick_batch/memory.rs#L265).
Successful Words enforce capacity == length. The general live-backing coordinate is:

```text
B_words = 4 sup_p Σ_{a ∈ L(p) \ E0} capacity(a).
```

Here L contains distinct LIVE Words backings only, E0 is entry-existing backing,
and aliases are deduplicated. Existing Source, retained parent, metadata, inline
arrays, Arc, allocator excess, failed-prefix capacity and residency are SEPARATE;
extra Source copies cannot vanish. Native 4,096 logical reservation is not actual
physical allocation; traffic bytes are not capacity.

**Whole incremental bytes, not entry-only coverage.** Entry-helper B is NOT proof
of stage coverage, and scalar `Dstage` payment covers WORK ONLY. Whole incremental
logical-byte coverage must dominate stage plus retained-parent overlap BEFORE
allocation. The [entry candidate](#conditional-native-entry-prefix-retained-parent-no-new-byte-reservation)
constants `2048+B`, `4096+K` mean work delta K+B and byte delta B, NOT K+2B:

```text
Qw = Nw+Ew+Rw+Aw+K+B;  Qb = Nb+Eb+Rb+Ab+B.
```

All intermediates, conversions, cumulative totals, metadata and limits need checked
coverage. The relation between entry B and whole-stage B remains MISSING. This
introduces no extra reserve, new fee or adopted B value.

**Escape and authentic issuer gate.** Candidate 1 remains exactly one private
borrowed helper after `RawFirstNormActivation` success reset and guard release,
before `Ok(output)` at `raw_first_norm.rs:163`. Follow the same parent path
`output.norm.reservation.producer.source.input.reader` to fresh sealed live payload;
T5 uses the SAME Root's primitive prepay. End ALL guards before error cleanup
reacquires Root. Bootstrap 24 is nontransferable. A proposed whole-activation
consuming continuation would retain native Client/server/Root, but still needs
NO-ESCAPE, borrowed-lifetime or authentic WHOLE-OWNER-TRANSFER proof for ALL actual
backings: requests, replies, disclosures, auxiliary replies and AcceptedOutputs.
An adjacent Root is insufficient: native `Reservation` (`maverick_batch/memory.rs:246–249`) retains only
`Memory(Arc<State>)`, not Root. See
[protocol.rs:202–245](../../crates/pllm-core/src/maverick_batch/protocol.rs#L202),
[request.rs:39–47](../../crates/pllm-core/src/maverick_batch/request.rs#L39),
[memory.rs:196–204](../../crates/pllm-core/src/maverick_batch/memory.rs#L196) and
[246–262](../../crates/pllm-core/src/maverick_batch/memory.rs#L246).
Current Client lacks Root; `RawFirstNormActivation` is not an existing CustodyOwner
and cannot be cast into one. ControllerLoan accounting is NOT issuer authority;
TokenLookup licensing is NOT descendant scope. Actual expected tensor/code
SourceAuth issuer, numeric mapping and matching checkpoint file extent remain
independently unproved. Config provenance/digest equality is not checkpoint
authentication. No new Root or phantom permit closes these gaps.

**Unresolved closure.** The authored cap bounds `below` to 32 fresh-u32 calls,
not entropy-provider time. Independent static review passed; tests have not been
executed and the current Rust wheel is not qualified. Prospective counters,
bookkeeping, wipe accounting and qualified fresh uniform error-free RNG remain
necessary. Rates, whole-helper coverage,
abort/failure capacity, first inspection, SourceAuth, issuer byte coverage and
escaped-Root retirement proofs remain absent. This packet supplies no grant, fee,
security result, full-Qwen result or native qualification. Administrative Native
holds are released, not evidence that these obligations are closed.

#### Appendix: partial unpriced retained-entry composition

These are partial source-linked **UNPRICED** components and the conditional T1–T8
skeleton, not an established whole-prefix vector. Candidate K, B remain nonnegative.
`LINK` denotes a call dependency, not a generic fee; its callee owns its body once.
Key events by (site, branch, occurrence, owner); preserve exclusive branches and
lifetime overlap. `U` denotes unknown expansions/layouts, not zero cost or size.

**Actual debit components.** Source:
[`original_source.rs:756–786`](../../crates/pllm-compiler/src/original_source.rs#L756).
These symbols are source components, not numeric primitive coefficients.

| Component | Source-owned operations |
| --- | --- |
| A | Budget access, controller `is_for`, negation and branch. |
| B | `controller.payload_bytes()` before M, including M failure; forwarder LINK [639–641](../../crates/pllm-compiler/src/original_source.rs#L639) to M; caller `?` distinct; comparison only after M succeeds. |
| C | Root access (`Original`/self or `OriginalContext`/root configuration), `ledger.as_ref().ok_or(Error::Burned)?`, with eager error construction. |
| D | Typed Ledger lock, `map_err` and `?`; poison closure consumes its `PoisonError`, retiring the contained guard once. |
| E | Check `ledger.burned`. |
| F | Write `burned=true`; read used and `checked_add(visits)`; call `filter` even on None, but its predicate reads `used <= limit` only on Some; `ok_or(Error::Work)?` constructs its error eagerly even on success. |
| G | Write used, write `burned=false`, construct `Ok(())`. |
| R | Retire acquired guard once; wrapper destruction and underlying unlock add no release event. |

| Exclusive branch | Component sequence |
| --- | --- |
| Controller mismatch | A + ErrCounter |
| Metadata failure | A B(error) |
| Capacity failure | A B + ErrCapacity |
| No ledger | A B C(error) |
| Poison | A B C D(poison retirement) |
| Already burned | A B C D(ok) E + ErrBurned + R |
| Overflow | A B C D E F(None; predicate skipped) + R |
| Over limit | A B C D E F(Some; predicate false) + R |
| Success | A B C D E F(Some; predicate true) G R |

Writes describe effects of this call only, not globally unchanged concurrent state.

**Metadata M.** Body [723–737](../../crates/pllm-compiler/src/original_source.rs#L723),
`add` [943–945](../../crates/pllm-compiler/src/original_source.rs#L943).
Q sets up seven `size_of` expressions, one ordinary `3 * size_of::<u64>()`, six
array elements and their `IntoIter` Ia. Hs/Hf own checked-add Option, eager
`ok_or(Error::Overflow)` even on success, and semantic Result success/failure
construction; normalized bodies remain U. Sa is successful array advance, Ta
terminal advance, Da(0) exhausted iterator retirement, Ubytes assignment, KOk final Ok.

```text
Msuccess = Q + 6(Sa + Hs + DResult + Ubytes) + Ta + Da(0) + KOk
Mfail(j) = Q + (j−1)(Sa + Hs + DResult + Ubytes)
             + Sa + Hf + DResult + RetireArray(6−j), 1 <= j <= 6
```

On failure, DResult owns propagation and enclosing Err construction/conversion
where modeled, not merely a tag; do not add a duplicate Try fee. RetireArray names
remaining logical elements, not executable `usize` destructors; iterator-wrapper
retirement has one owner. The caller's `?` outside M is separate from the six loop
`?` sites; count the forwarder LINK once. Success has six checked additions and
seven iterator advances; failure has j additions/advances, no terminal advance.
`Error::Native(NativeError)` layout is not assumed zero. These sizes describe
inventory, not actual guard allocation or work funding.

**Conditional helper and unresolved T5.** Use T1–T8 above, excluding arbitrary T5
expression work from structural bounds: fixed skeleton has one root lock attempt
and at most one debit LINK. Each entered root lock invocation contributes an attempt
event. A returned normal or poison-contained guard contributes a retirement event
only when actually retired. Every normally returning helper path must retire its
guards before return; ordinary-unwind paths must retire them before outer activation
destruction. Neither lock completion, nonreturn nor unwind alone proves retirement.
Custody follows `output.norm.reservation.producer.source.input.reader.root.state`;
original/controller come from the same fresh payload. Reader gates are sealed,
not retiring, live, nonfailed and payload-present. No existing retained successor
work scalar, quote or balance is established: activation owns words/norm;
OriginalReservation owns charge/key/producer only; gamma extent is not demand.
Full composition needs immutable u64 tied to retained attempt and next-work extent,
formula/units/refusal-cleanup coverage, and prepaid derivation/overlap. No new API,
ticket or types are supplied; TokenLookup custody grants no Maverick issuer authority.

**Admission delta and coverage.** Through E (execution), S (norm), R (reservation)
and A (admission), checked quote addition gives `Q1 = Q0 + (K+B, B)` in
(work, resident), with no extra fold item or reserve call. Positive B changes the
old reserve request. Initial `REGISTRATION_READER_IO_BOOTSTRAP_BYTES` and ingress
provisioning stay unchanged. Earliest changed operations are execution controls/
work folds [raw_first_norm.rs:49–56](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L49),
before quote/admission. Pre-admission composition/refusal coverage remains unknown,
not impossible and not proof of no existing allowance. The 24-work primitive
bootstrap stays unchanged; subsequent full-demand debit precedes peak/byte charge
at [raw_owner.rs:590–596/648](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L590).
Existing `shared_allocation::reserve_bytes` uses `fetch_update`; project callback
retry multiplicity is unproved. Root mutex does not exclude all Budget aliases;
no Memory/ordinal atomic exception transfers here. DIRECT user44988's PUMA-specific
approval covers the whole standard atomic `fetch_update` in `Budget::reserve_bytes`,
including callback/retries, as one unit logical event; wrappers/caller/peak/Arc/payment
remain separate, with no CPU/termination/funding/implementation/execution grant.
A separate Maverick SELECTOR-only shared-budget exception exists. Neither is
primitive-wide or new retained-composition approval. Concrete caller/operation
attribution must determine whether an unchanged parent reservation belongs to
already-approved PUMA/selector work or a new out-of-scope delta: a descendant's
“Maverick” name alone does not make its PUMA parent uncovered, and Budget identity
alone does not prove applicability.

**Cleanup and remaining closure.** Helper guards retire before output full-capacity
wipe/free → norm gamma → reservation charge → independently charged producer/input/
reader. RawReader Drop separately reacquires root at
[raw_owner.rs:712–718](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L712).
Already-paid exclusions need authentic owner/operation/extent/lifetime evidence.
Opaque cleanup, workspace, charge-handle internals and last-owner behavior remain
unpriced/unproved. No CPU/RSS, progress, termination, caps or HOLD claim follows.
Closure still requires trusted T5, normalized bodies, storage overlap, complete
coverage, authority and rates; no funding, implementation or native-execution grant.

**Authored 32-draw rejection cap — 7 November 2026.**
`crates/pllm-core/src/maverick_batch/arithmetic.rs::below` consumes at most 32
fresh u32 draws, returns `Error::Randomness` after 32 rejections and propagates
an underlying error immediately. It allocates nothing, adds no retry/reset and
preserves `word_from_fill` zeroization. The accepted sampling law, current noise
165, code and 16 batch ordinals remain unchanged. D31/D32 did not authorize this
cap; explicit user policy approval and the 7 November administrative hold release
are separate from funding proof. Independent static review passed. Four unit and
two integration regressions are authored; the reference test binary includes
the four unit tests, giving six registered tests under `bounded_draw`. Tests
have not been executed; the current Rust wheel is not qualified. Funding/tariff
review is no longer a mandatory before-authoring administrative hold. Actual
prepayment and complete funding remain unproved; no cap increase or PRG adoption follows.

Rejection calculation: `R=2^32`, `T=R-(R mod b)`; reject draws `>=T`, with
exact rejection probability `r=(R mod b)/R`. For `L=512,k=165`, position bounds
are 512 through 348 and there are 165 coefficient bounds `FIELD-1=2,013,265,920`.
Coefficient threshold is 4,026,531,840, rejecting 268,435,456 values (`r=1/16`).
First claim uses 330 `below` calls, at most 10,560 draws. A fully exercised batch
uses 660 calls for two originals, 124 primary and 124 secondary: 908 total.
The horizon is 16 SHARED issuance ordinals: 16 batches OR eight two-batch requests,
not 16 requests. Fully exercising those ordinals uses 14,528 calls, at most 464,896 draws:
7,264 coefficient calls and 7,264 position calls. Under independent, uniform,
error-free entropy, the union bound is
`Pr[cap abort] <= 7264*2^-128 + 7264*(1023/2^32)^32 < 2^-115`.
This is not 16 two-batch requests or full-model security.

Conditional on sampler completion, uniform support/nonzero-coefficient law is
preserved: accepted residue is independent of rejection count with these public,
predetermined bounds. This is NOT conditioning on later protocol acceptance.
Entropy-provider errors remain separate and may invalidate the simulation; a draw
cap gives no latency bound.

Whole-horizon prepayment is not required. One consuming-attempt quote must cover
complete authorized sampling and cleanup before first entropy; later attempts
use the same cumulative owner ledger. A two-batch request-scoped quote must fund
BOTH ordinals. No refresh, reset or refund. Original-ledger debit for maximum
draws, deterministic bookkeeping and wipe before entropy remains unresolved.
Existing `raw_owner::prepay_primitive_work` bootstrap 24 is not transferable to
this scope; new numeric rates and control coverage remain unapproved. Invent no
fees or refunds.

**Logical external-call accounting selection; funding evidence remains open.** Provider, allocator and
atomic API invocations may be priced as logical external calls; all visible loops,
copies, hashes and wipes must be priced separately. This model excludes physical
CPU/instruction bounds, allocator-internal work and RSS; it proves no such bounds.
Only the accounting model is selected: numeric rates, logical sampling frontier,
E/Q and full cumulative fee remain unapproved, including any 45448 tariff or COMMON fees.
The 32-draw implementation is now authored under the 7 November 2026 administrative
release; no engine/profile, security or validation approval follows from this selection.
Draw law/targets/caps, original ledger, attempt burn/replay/refusals and trust/source ownership remain unchanged.

#### Source-linked accounting review: inventory ingredients, not a complete LU tariff

Only logical external-call accounting is approved. This numeric proposal is neither
implementation-ready nor offered for approval as a complete quote; the historical
45,448 proposal and approval boundaries above remain unchanged.
[Sampler source](../../crates/pllm-core/src/maverick_batch/code.rs) uses partial
Fisher–Yates (`code.rs:185`): `2k` successful `below` calls, hence 330 per first
claim, 908 per batch and 1,816 per two-batch request. The authored
32-draw cap per `below` call bounds draws by
10,560 / 29,056 / 58,112 respectively; it does not impose a global batch-draw cap.
Conditional on **unproved** `B(n)=512+4n`, per-draw 64, setup 16 and terminal 8,
`F=64+B(L)+B(2k)+4L+4208k = 1088+8L+4216k` gives sampling-only batch/request/
16-batch-horizon totals 1,953,360 / 3,906,720 / 31,253,760. These are conditional
identities, not approved fees or source-exact inventory: 46 spare/control units in
64 and sparse overhead remain unenumerated; fixed `B` overhead 512 is unproved.

Locked [zeroize 1.9.0 Vec implementation](https://docs.rs/crate/zeroize/1.9.0/source/src/lib.rs)
(`lib.rs:528–536`) wipes initialized elements, clears, then wipes full capacity
bytewise. For `u32`, successful `c=n` needs `n` initialization word writes, `n`
initialized volatile word writes and `4n` capacity byte writes, plus loops/barriers/
control. Length-zero pre-capacity-mismatch cleanup still needs `4c` byte writes.
Thus proposed success `<=4n` / failure `<=2c` are not established. Actual
`copy_from_slice` copies `4n` bytes and needs an explicit chosen copy rule.
Independent Rust 1.85.1 source proof: [RawVecInner::grow_exact](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/raw_vec.rs#L665)
sets `cap=len+additional=n`; fresh `current_memory=None` allocates;
[Global::alloc_impl](https://github.com/rust-lang/rust/blob/1.85.1/library/alloc/src/alloc.rs#L187-L197)
returns slice length `layout.size()=4n`. `set_ptr_and_cap` stores requested capacity,
read back for non-ZSTs: selected fresh `Vec<u32>`/`Global` advertises `c=n` (`n=0,c=0`).
No extra `c<=2n` restriction/backend is needed. Successful capacity mismatch is
unreachable for this selected implementation with a valid allocator; retain the
generic guard and hypothetical `4c`-byte cleanup. Actual compiler/library
qualification is UNVERIFIED; no RSS/allocator-internal bound follows. `B` and numeric
rates remain unproved/unapproved. USER71483 selects selector logical-reservation
accounting as clarified below; it does not select a 32-callback cap, bound physical
retries or establish a blanket backend tariff. Other scopes are not discharged.

Astra `ses_e79feb749ffem7rTcWj0fS6Ylz` audited native lifecycle ingredients in
[code](../../crates/pllm-core/src/maverick_batch/code.rs),
[protocol](../../crates/pllm-core/src/maverick_batch/protocol.rs) and
[publication](../../crates/pllm-core/src/maverick_batch/publication.rs): each
preprocess or validation phase has 768 encodes (256 at `L=512`, 512 at `L=1024`),
two full-length encode passes. Bind three `L=512` codes, one `L=1024` code, two
matrices and one publication; publication transcript is 2,621,770 bytes and 655,378
updates. `Code::digest`/copy also validate; hash calls are not opaque wholesale work.
Preprocess+validate+split has at most 3,105 buffer attempts; alternative import+
validate+split has 1,566. Never sum preprocess and import absent an actual callpath.
`B` covers acquisition through cleanup; remaining scalar control `K`, retained-root
retirement `T_root` and unbounded Busy retries remain unpriced. Publication has no
sampler calls; split's 32-byte entropy and later issuance `S_sample` stay separate.
`M=2^32` / `P=2^36` are arbitrary allowances, not proven bounds; work/bytes
inequalities with unknown actual `S`/`C` prove no fit. Audited authenticated
continuation stops at [RawFirstNormActivation](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs).
Missing: consuming whole owner → authentic original-graph `Linear` successor →
exact weight AND scale read owner → same original-ledger reservation → projection/
output owner. Existing weight lookup does not authenticate the norm→`Linear`
producer relation; standalone `projection_custody`/`window_linear` helpers confer
no authority. Retain full activation/gamma/lookup/tokens/reader/controller/root
through output retirement.
Retain [raw_owner prepay provenance](../../crates/pllm-compiler/src/verified_source/raw_owner.rs)
for validation/read/window cleanup without double charging, and original root/
controller/source-key custody. Native `64×64` versus proposed `256×128` remains
unresolved; no geometry/profile/cap, implementation, activation or route clearance.

**USER70904 approval scope — design revision only.** Authentic retained-original-graph
producer→`Linear` selection must bind full weight, scale and bias semantics. Authorization
commissions a corrected design, not draft acceptance, source implementation, rates,
execution, cap changes or geometry adaptation. First-projection implementation remains unapproved.

**Historical UNSELECTED revised selection proposal (design only).** The semantic
selector and its implementation remain unapproved. USER71554 now selects the
[lifetime direction below](#selected-selector-lifetime-direction), superseding
the historical lifetime gaps/candidates below, not approving their construction.
Private consuming `RawFirstNormActivation::select_linear(self, requested_index: usize)`
would return `RawLinearSelection`; this method/selection API does not exist today. The index is untrusted.
The whole private, non-Clone activation retains norm/gamma/lookup/tokens/reader and original
controller/ledger/attempt. A `TokenLookup` reservation never becomes a `Linear` capability;
selection grants no words/read API, arithmetic, geometry adaptation or result publication.

Resolve norm and requested indices in the original graph; authenticate producer edges and
capability shapes: norm `[B,L,K]`, selected `[B,L,M]`, weight `[M,K]`, bias `[M]`.
The IR has inputs, output shape and JSON attributes, not an input-shape field. Check unique
IDs in ONE actual `G` pass, with length-first target comparisons: `N=raw NAME=1024`,
at most `2G` comparisons / `2GN` bytes; unrelated long IDs need not refuse.
Proposed, unapproved attribute schema: object with at most two entries, required nonempty
string `weight`, and separately preserved absent/null/nonempty-string `bias`; refuse unknown,
nested or malformed attributes. One bounded iterator uses at most four key comparisons /
20 bytes; no arbitrary JSON-tree walk, copied strings or metadata Vec is proposed.
ONE actual `T` pass resolves weight and optional bias only, at most `2T` comparisons /
`2TN` bytes. Sealed-premise registration already proves global name uniqueness and
`scale_index` links: use DIRECT indices, not a four-name one-pass join. At most four
descriptor roles (weight/scale/bias/scale) use at most three file handles on current valid paths.
BF16/F32 weight/bias are unscaled; FP8 weight requires sealed exact positive divisibility by
128 and BF16 scales `[M/128,K/128]`, without ceiling. Explicitly refuse scaled/FP8 rank-one
bias rather than dropping it. No new numeric interpretation or parameter adoption follows.

The following source-reviewed semantic inventory is proposed, not a complete `C0` census or approved policy:

| Scope | Fixed predicates and bounded events still requiring explicit pricing |
| --- | --- |
| Graph selection | Two original-graph lookups; selected index later than norm, not necessarily adjacent (Q/K siblings share a producer); one `Linear` enum check; one input and sole edge equal to norm ID. |
| Target IDs / `G` loop | Two nonempty and two `N=1024` checks; two length tests and at most two bounded comparisons per entry; two match counters, each required to equal one. |
| Attributes | Existing four-key-comparison/20-byte bound; at most two string/nonempty/`N` checks plus required-weight presence. |
| `T` loop | Two length tests / at most two name comparisons per entry, at most `2TN` bytes; sealed global uniqueness and direct scale indices, without another duplicate-inventory walk. |
| Capability shapes | Norm and selected outputs each rank three; six u64→usize conversions; three capability-shape equalities, two B/L equalities and four positive B/L/K/M checks. |
| Weight | Rank two and two dimensions; BF16/F32 requires both scale fields absent, while FP8 requires both present. |
| FP8 scale | Two remainder checks and two divisions; one DIRECT scale-index lookup and bounded scale-name comparison, including its priced length guard; BF16/rank two/two dimensions and no scale-of-scale. |
| Bias | Rank one, dimension M, BF16 or F32, and both scale fields absent. |
| Selected records (at most three) | Stride 1/2/4 dispatch; rank-two element-product multiply (none for rank one), stride multiply, usize→u64 conversion, checked end−begin, byte equality and begin%stride. Each actual file binding needs lookup, Weak upgrade, Arc identity, actual/sealed size equality and end≤sealed size. |
| Loop/control inventory | Separate init/body/terminal and counter updates for G/T, attributes≤2 and records≤3; exact predicates, helpers, result/drop dispatch and `C0` remain UNFROZEN. |

**Concise event-vector proposal — not tariffs or final `C0`.** Let `G` be actual
graph length, `T` actual tensor-inventory length, `N=1024`, attribute count `A≤2`
and selected-record count `R≤3`. Proposed source-event coordinates are:

- Graph: `G` visits, `2G` length tests, at most `2G` comparisons / `2GN` bytes;
  tensor pass analogously `T`, `2T`, at most `2T` comparisons / `2TN` bytes.
  Each pass separately includes initialization, `n+1` iterator polls, `n` bodies,
  counter updates and terminal checks; attributes and records retain their own
  bounded loop/control events, not free helper calls.
- Attributes: at most four key comparisons / 20 bytes and two nonempty bounded
  strings; absent/null/string bias remains explicit. Refuse scaled or FP8 rank-one bias.
- Shapes: two rank-three checks, six conversions, three capability equalities,
  two B/L equalities and four positivity checks. FP8 scale: two remainders, two
  divisions, one direct lookup, at most `N` scale-name bytes plus its length guard;
  retain BF16/rank-two/dimension and no-scale-of-scale checks.
- Records: at most two rank-two products; `R` stride products, conversions,
  checked extent differences, byte equalities and alignment checks; `R` file
  bindings, Weak upgrades, identity checks, size equalities and bounds checks.
- One `check_header`: two Weak upgrades/drops, two Arc identity checks, two
  directory-budget checks and one fixed-premise check; no inventory scan.
- Ownership: at most three selected-file clones/drops, three temporary upgraded
  handle drops and one transient authentic-root clone/drop, with separately counted
  copy, destruction and cleanup events.
- Admission plus growing `try_build`: at most two logical reservation invocations,
  three peak atomic updates, one used load, one budget Arc clone/drop and one charge
  release. Internal retries/callbacks belong to their invoking logical reservation;
  separate invocations and surrounding owned work do not merge into it.
- Affine quote `C0+Cg*G+Ct*T`: two conversions, two checked multiplies and two
  checked adds, themselves prepaid at entry rather than funded by the resulting quote.

These are proposed source events, not frozen rates, a complete fixed-helper/control
census, final `C0` or a physical CPU/time bound. Complete fixed controls, helpers,
cleanup and typed simultaneous slots remain unproved because the selector is not
implemented. The former before-authoring quote-review hold was administratively
released on 7 November 2026; do not invent `C0` or
treat this vector as funding proof.

The proposed no-read selector retains ONLY relative `[begin,end)` extents using authenticated
sealed-inventory data-region validation. [`RawBacking`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L1091)
stores no data-start offset; [`data_start_into`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L1303)
reads eight bytes (lines 1303–1315). Do not call it or compute absolute offsets during selection.
A future actual-read owner requires separate admission; this extent choice grants no read permission.

Propose enlarging the original norm execution-demand combined quote through existing
`admit_input_norm_spec_with(extra)`→`reserve_original`, prepaying ONE future selector entry
and retained controls BEFORE activation issuance. This implicit private-owner invariant has
no detachable credit and requires every production issuer to establish it; CURRENT CODE
HAS NO CREDIT. Unused credit/work stays spent; pre-issuance failure needs separately prepaid
old-prefix cleanup. This does not solve upstream bootstrap globally.
Use one existing root lock. Check sealed/live/nonfailed/nonretiring `reader_payload` BEFORE
setting `reader_failed=true`; never call `reader_payload` again after latching. Inspect only
actual `G`/`T` lengths and index bounds before semantic traversal. Overflow, invalid index,
short work or short bytes must unlock before whole-owner Drop. Proposed public immutable
coefficients give `W_select=C0+Cg*G+Ct*T`: two checked usize→u64 conversions, two checked
multiplies and two checked adds; this is unpriced, not a complete `C0` event census.
Debit ORIGINAL work, then check actual capacity/peak and charge external storage BEFORE
authentication, scans or selected-file clones, except for the upstream issuer-prepaid transient
authentic-root clone needed before locking, as described below. Byte failure does not refund work; preserve exhaustion burn
and attempt continuity. Perform one actual `check_header` under the SAME guard, never nested
`check_source` or RawFirstNorm validation, which relock/scan/charge. Clear the latch only when
success construction is ready, then unlock/handoff; failure remains sticky. Dispose the guard
BEFORE activation Drop, which relocks the root; malformed/insufficient-work refusal remains.

Proposed `SelectionCustody` orders fixed Copy indices/descriptors/extents, four roles,
presence mask and absent/null/string bias tag → selected file handles → WHOLE activation;
`pllm_garble::shared_allocation::OwnedCharged<SelectionCustody>` places charge LAST
(private payload-before-charge fields, lines 552–555). Retain `RawGraph` by value (Copy),
not a graph Arc clone. The private constructor retains a NEW charge on the same original
counter, not a retagged `TokenLookup` charge; whole activation, not charge, supplies custody.
Four fixed handle roles need at most three selected handles plus one transient Weak upgrade
on current valid paths. A persistent Root Arc/custom retirement keeper is unsupported and
WITHDRAWN. Permit one issuer-prepaid TRANSIENT authentic Root Arc solely for independent
lock borrowing through transaction cleanup, not persistent custody or refund; quote its copy/drop
slots and work. `OwnedExternalCharge::drop` uses only its own Arc<Budget> (lines
429–432, 544–547), with no read/validation/source authentication/root access during refund.
Historical gap: `RawReader::drop` may clear `reader_live`, allowing parent retirement
before refund while the legitimate new charge counter's Arc<Budget> remains. Charge
refund itself needs no reader loan, but that fact does NOT prove collection-funded
frame-tail backing/lease retention. The selected direction below would prohibit
retirement during that tail with separate root-local `selection_live`; it is not
implemented and the existing gap is not claimed closed.
This ordering is CONDITIONAL design, not implemented, accepted or native-proved. `OwnedExternalCharge::try_build`
adds inline wrapper storage (lines 524–540): reconcile it with prepaid bootstrap/result/move
controls and new publication overlap; do not double-count the `OwnedExternalCharge` header
or old backing. Inherited root payload cleanup keeps original retirement funding; original
charge-before-producer ordering alone proves no new selector cleanup coverage. Whole S/F
typed `B`/`Q`/helper/result-move inventory, exact simultaneous slots and helper admission
internals remain UNFROZEN; no anonymous 512, RSS/physical-ABI proof or authoring grant follows.

Historical UNSELECTED scope-order candidate (refined by the selected lifetime
direction below): declare outer activation `Option`, then publication `Option`.
In a nested scope declare partial handles, then clone the authentic transient root through a
short activation borrow that ends before a borrowing, non-`move` closure; declare guard last
inside that closure. After ORIGINAL work debit, existing
[`OwnedExternalCharge::try_build`](../../crates/pllm-garble/src/shared_allocation.rs#L520-L540)
resizes before its builder performs header validation, scans or selected-file acquisition.
Only the final infallible, nonpanicking `activation.take()`/files move constructs custody
in field order fixed Copy data → files → activation; no ownership capture under the lock.
Retain status without propagation; success stores the wrapper in publication and clears
the latch with no subsequent fallible operations. After the closure statement drops guard
and captures, explicitly `drop(root)`, then `drop(handles)`, THEN propagate status or hand off.
Unwind order is closure/guard → root → handles → publication → remaining activation.
Together with existing payload-before-charge and original reservation charge-before-producer,
this removes necessity of a persistent root keeper/new shared primitive for transient-root
lifetime ONLY; it is an uncompiled sequence candidate, not a panic or funding proof.
Resize/builder Err may refund NEW charge early: authentic bootstrap coverage must still fund
partials, emptied outer slots, escaped `Result` and post-refund dispatch; norm credit is not blanket proof.
Fixed-slot overlap, panic/callback/ABI, `C0`, tariffs and issuer credit remain UNFROZEN.
Entry prepayment is proposed, not existing credit; the 7 November 2026 administrative
hold release permits source authoring but proves no funding, issuer or custody claim.

Astra24 independently finds collection-bootstrap funding plus a proposed non-Clone
`SelectorFrame` borrow conditionally feasible for the typed tail, NOT full retirement closure.
The sole explicit final payload/charge release is
[`release_final_charges(&self)`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L1080-L1087).
Requiring `&mut RawHost` there could exclude that release while a unique-holder borrow lives,
but only under audited private-constructor confinement: internal tests construct another
`RawHost` from the same root Arc
([line 630](../../crates/pllm-compiler/src/verified_source/raw_owner_tests.rs#L630),
[line 848](../../crates/pllm-compiler/src/verified_source/raw_owner_tests.rs#L848)).
Non-Clone holder/borrow alone is no universal Arc-uniqueness guarantee. This frame and its
bootstrap/initial prepayment are proposed, not current credit; retain issuer-work/payload
ownership without duplicate charging, a new root keeper, ledger, cap or fee.

More importantly, [`RawReader::drop`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L712-L718)
clears `reader_live`, while [`begin_retire`/`retire`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs#L1038-L1076)
accept `&self` and may clear files/remove the directory during the frame tail. Changing final
release alone preserves collection funding, NOT backing/lease. Any full fix needs exclusion
covering begin-retire, retire AND final release, audited through
[`CompilerPublisher`](../../crates/pllm-compiler/src/replicated_decoder/source.rs#L249-L285),
[detached native facades](../../crates/pllm-python/src/replicated_transport.rs#L204-L214) and
[Python retirement](../../python/pllm/runtime/replicated_source.py#L189-L235).
The bridge orders Ready → backing clear → retire → lease exit → final release: refusal must
precede Ready, not merely final release. Production reader/worker dispatch is
[absent](../../python/pllm/runtime/replicated_source.py#L244-L251); no existing borrow across
those detached calls is established.

#### Selected selector lifetime direction

**USER71649: DESIGN AMENDMENT ONLY / PROPOSED IMPLEMENTATION.** This Astra26
refinement preserves USER71554's selected lifetime direction and the historical
decision record; it supersedes neither history nor unresolved evidence obligations.
Administrative holds were released on 7 November 2026. Selection
authorizes this design tranche only: no selector, host-bridge or exclusive-retirement construction,
funding proof, frozen layout/rates, compiled API or execution approval follows.
[NEXUS §6.3](nexus-qwen-decoder-contract.md#63-minimum-shared-successor-selector--selected-lifetime-direction-proposed-implementation)
shares this model-/method-neutral norm → Linear lifetime direction; it grants
none of NEXUS's separate derived-Q/K execution authority.

**Verified baseline, not the proposed mechanism.**
[`RawHost`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs) owns a
private root Arc; the same original-root mutex protects `reader_live` and
retirement. Public [`Publisher::raw_*`](../../crates/pllm-compiler/src/replicated_decoder/source.rs)
lifecycle methods take shared references. [`NativePublisher`](../../crates/pllm-python/src/replicated_transport.rs)
detaches per individual call, not across the whole selector/retirement lifetime.
The gate in [`replicated_source.py`](../../python/pllm/runtime/replicated_source.py)
is not native-alias authentication. Existing `OwnedExternalCharge` retains an
independent Budget Arc and exposes only `&Budget`, not raw-root access; its
Drop's `fetch_sub` has no source lock. Existing `OwnedCharged` drops payload
before charge. None of those facts supplies full backing/outer-lease exclusion.
Authority remains the WHOLE `RawFirstNormActivation`, gamma, lookup, tokens and
reader; `OriginalReservation` remains TokenLookup-associated, not Linear authority.

**Authentic debit and proposed entry credit.** Existing
[`raw_owner::prepay_reader_bootstrap`](../../crates/pllm-compiler/src/verified_source/raw_owner.rs)
calls [`OriginalSourcePlan::prepay_primitive_work`](../../crates/pllm-compiler/src/original_source.rs)
with the retained original controller: the authentic original EXECUTION
`WorkLedger` is irreversibly debited, not replaced or copied. Successful debit
stays spent; cumulative exhaustion/overflow burns siblings under the existing
contract. Controller/precondition refusal is not a successful debit. This is an
existing seam, NOT evidence that current norm admission prepays selector entry.
Proposed `first_bf16_norm` includes selector-entry demand through
`admit_input_norm_spec_with`; only AFTER successful original debit may its private
issuer create a private, non-Clone, zero-sized `SelectorEntryCredit`, inseparable
from the whole activation. It authorizes only the prepaid entry transition, not
Linear semantics or later work. All authentic issuer/failure seams must enforce
that rule. Existing `RawFirstNormActivation` has no such credit; its existing
TokenLookup-associated reservation alone is insufficient. Provenance/root custody
stays in the retained reader: no new copied ledger/root or persistent root keeper.
The proposed credit is consumed exactly once at entry, never caller-provided,
independently transferable or reconstructible from equality of values. Its private
issuance follows the original predebit; authentic entry-prior access, native-entry
confinement and COLLECTION funding remain OPEN, not consequences of its type name.

**Proposed confinement and root exclusion.** Add private
`RawHostState.selection_live` under that SAME original-root mutex. Proposed
private non-Clone `SelectorFrame<'h> { host: &'h RawHost }` borrows
the authentic `RawHost` inside the genuine Publisher/native-owner scope;
authenticate exact frame/activation/host-root identity and original entry
prepayment BEFORE latching. No Python frame object, copied host, escaping
callback/task or selection alias. A higher-ranked scoped interface returns
owner-free status only, not a lifetime-erased owner. Exact API remains proposed.
Non-Clone does not prove Arc uniqueness: root-local exclusion must also cover
accidental second same-root facades. Proposed exclusive owner APIs span
`RawHost::begin_retire`, `retire`, `release_final_charges`, Publisher/native
lifecycle, replacement, ingestion, direct fallback and Drop paths. Native borrow
must span detach AND cleanup; retirement reservation must stay latched across
multicall phases. A local Python gate or exclusive final-release method alone
cannot establish this discipline. All these API changes remain uncompiled proposals.

**Required proposed sequence (not a current implementation):**

1. Original COLLECTION funding covers frame/control/refusal/post-charge cleanup
   slots BEFORE construction. Separately, EACH authentic activation issuer must
   include selector ENTRY demand in original EXECUTION admission and issue the
   inseparable credit only after successful debit, as proposed above. Consume it
   for entry; it is currently absent. Collection ledger and an expired norm
   reservation are not substitutes for execution-entry credit.
2. Retain the genuine Publisher/native owner AND actual outer lease owner. Where
   needed, take a scoped authentic transient root alias covered by collection
   funding; it is neither persistent custody nor a refund/root keeper.
3. Under the root mutex reject failed, unsealed, retiring or active state; verify
   exact original activation-root identity. Check `reader_payload` BEFORE the
   consuming failure latch, then set that latch and `selection_live`. Foreign
   input must never latch another root. Authentication/prepayment precede latching.
4. Entry credit funds checked quote computation from actual `G`/`T` counts.
   Debit original cumulative execution work BEFORE byte reservation or semantic
   construction. Later refusal never refunds work; attempts and exhaustion burns
   remain spent, without owner/ledger replacement.
5. Require bounded lock-aware `check_header` ONCE. Check exact graph successor,
   ordered inputs, attributes, shapes, backing/index and issuer relation BEFORE
   selection Ready. These validation obligations execute only inside the
   already-grown builder in step 6, not before its byte admission. Do not invoke
   relocking/scanning `check_source`, `validate`
   or `admit_primitive` wholesale. Any extracted checks need an explicit event
   census and remain unapproved implementation work.
6. `try_build` growth precedes validation/acquisition; transfer WHOLE activation
   plus selected handles, not a detached reservation. No source DATA reads,
   numeric execution or publication: the one bounded source-metadata header
   check is explicit, not a tensor/source-payload read. Clear consuming failure
   latch only after infallible completion; `selection_live` stays set for cleanup.
7. End ALL guard-containing lock Results and guard destinations before activation,
   reader or any source-relocking destructor, and before a fresh clearing lock.
   Consume both success and error variants, including `PoisonError<MutexGuard>`;
   dropping only the guard destination is insufficient. No unsafe formatting or
   deferred destruction of retained guard-bearing poison errors. Error/unwind
   order: lock containers/guards/closure → transient root alias → partial handles
   → completed selection, if any → remaining activation. Borrowing captures own
   references only. Independent Budget Arc charges may refund earlier under lock;
   this does not permit source-relocking payload/owner destruction there.
8. Actual retained ownership must keep authentic host, outer lease, exclusion and
   cleanup custody through cancellation, unwind and sticky failed-clear state.
   Stack borrow alone proves no post-unwind retention. The structural retained-owner
   candidate below is proposed only; enforcement and funding remain OPEN.
   Only after scoped aliases/payloads/refunds finish may a fresh short root lock
   clear `selection_live`; then release frame/borrow. Collection funds the tail,
   including partials surviving new-charge refund. Ambiguous clearing retains
   custody/fails closed, not successful flag clearing, recovery or release.
9. Exclusive retirement checks BOTH `reader_live` and `selection_live` BEFORE
   Ready. Preserve stop issuance → Ready reservation → backing-credit clear →
   native files/directory retire → successful actual lease exit → final charges.
   Mutation-free Busy preserves custody/backing/charges and permits explicit later
   retry (`raw_owner.rs:1050`, `replicated_source.py:230–233`), not automatic retry.
   Failed post-mutation or ambiguous retirement retains original custody/charge/
   lease with no inferred recovery; Busy retry does not reopen that failed branch.

**Reviewed selector control specification — DOC-ONLY control specification.** The following
expands the required sequence; source-authoring holds were administratively
released on 7 November 2026. Funding, exact schema, tariffs and issuer evidence
remain unresolved. Here
`G = original.operations.len()`, `T = sealed.tensors.len()`, `A≤2`, `R≤3`, and
`N=1024`; counts come from retained authentic owners, not caller estimates.

**Ordered stack states and failure recording (proposed).** Normal order is
Enter → Locked → Latched → Quoted → Debited → Growing → Validating → Assembled →
Draining → Clearing → Done. A fixed-size `FailureStage` records the failing stage
without allocating, formatting or propagating an error under the lock. Its exact
representation, helper scratch and error conversions remain unfrozen. Failure
branches retain status and enter unlocked Draining; they do not skip cleanup or
claim successful Clearing/Done when lock poison or unwind makes release ambiguous.
Pre-latch refusal never clears another invocation's selection flag; Clearing
applies only to the selection exclusion acquired by this invocation.

| State | Required control boundary |
| --- | --- |
| Enter | Retain authentic native owner and actual outer lease through cleanup. Check the outer activation `Option` and private activation-inseparable entry credit; consume that credit once. This is no caller capability or equality-based reconstruction. Entry-prior access and authentic native-entry enforcement remain OPEN. |
| Locked | Borrow the original-root mutex through retained `transient_root.as_ref()`; keep that root slot untouched through the ENTIRE transaction and poison-Result scope. Split mutable non-root fields before locking, as specified below. `reader_payload` is the FIRST payload check, before any consuming latch; authenticate activation/root identity. Poison, unsealed, failed or retiring state refuses. Never latch a foreign root. End ALL guard-containing Results/guard destinations before unlocked drain or fresh clearing lock; no poison recovery is claimed. |
| Latched | Reject an existing selection; set consuming failure latch and `selection_live` under the same guard. Access only counts and index bounds here, not graph/tensor semantics. Do not call `reader_payload` again after latching. |
| Quoted | Compute proposed `C0+Cg*G+Ct*T`: two checked count conversions, two checked multiplies, two checked adds. Entry prepayment must already cover these and refusal. Coefficients are NOT approved; this is not a complete fixed-cost quote. |
| Debited | Make genuine cumulative original EXECUTION debit. Successful debit remains spent through all later failures; preserve exhaustion/overflow burns and attempt identity. A controller/precondition refusal is not a successful debit. |
| Growing | Check resident-byte arithmetic, cap, actual capacity, peak and external reservation before `try_build`; its wrapper growth precedes builder execution. No semantic scan or selected-handle acquisition before admission. Builder captures ONLY references and borrows activation UNTIL final infallible assembly; returning `SelectionCustody` then moves ownership inside the builder. |
| Validating | Execute the single lock-aware header check and bounded validations below inside the already-grown builder. Keep partial handles in outer slots so failures destroy them only after unlocking. |
| Assembled | Finish graph/output authentication separately BEFORE final infallible whole-activation move. Success corridor: custody assembly → builder `Ok` → helper `payload` → wrapper assembly → wrapper Result → completed-wrapper destination → PREVIOUSLY EMPTY retained publication. EXHAUSTIVELY transfer success; never ignore success or replace occupied publication. After FIRST transfer: no `?`, `unwrap`, `expect`, allocation, validation or callback. Clear consuming failure latch, KEEP `selection_live`; explicit unlocked drain follows, with no escaping publication authority. |
| Draining | End ALL guard-bearing lock Results (success AND poison-error variants), guard destinations and borrowing closure FIRST. Drop transient root, partial handles, publication/completed selection, then remaining activation. Source-relocking owners wait for unlock; independent Budget Arc refunds may already occur under lock. Actual retained owner must preserve host/lease/exclusion/cleanup custody through cancellation, unwind and sticky failed-clear state; stack borrow alone proves no retention. |
| Clearing | Only after scoped aliases, payload destruction and refunds finish, and ALL earlier guard-containing Results/guards end, use fresh short root lock to clear `selection_live`. End clearing Result/guard variants too. Poison/ambiguous clear retains custody/fails closed; authenticated successful clearing remains a gate, not inferred recovery/release. |
| Done | Release frame/borrow only after authenticated successful Clearing and its entire lock-Result/guard tail. Actual retained-owner enforcement/funding through unwind or failed clearing remains OPEN. Retirement is separate, not selector completion. |

**Validation order inside the grown borrowing builder.** One lock-aware
`check_header` only; no relocking `check_source`, full activation validation or
unpriced helper traversal. Two indexed original-graph lookups bind norm and chosen
operation to authentic capabilities. Require `Linear`, selected index after norm
(not necessarily adjacent), exactly one ordered input and that sole edge equal to
norm ID. Guard target IDs as nonempty and length `≤N` BEFORE comparing bytes.
One `G` pass checks uniqueness of the two target IDs, not a second global walk;
unrelated long IDs need not refuse. Shape checks retain rank-three norm/selected
outputs, six conversions, capability dimensions and B/L/K/M constraints.

Check attribute object cardinality `A≤2` BEFORE iteration. Required `weight` is a
nonempty bounded string; preserve absent/null/nonempty-string `bias` unchanged.
Reject unknown keys, nested values and malformed attributes. This proposed exact
Linear schema is still OPEN, not approved by documenting the bounded iterator.
One length-first `T` pass resolves weight and optional bias. Use sealed direct
scale indices, not another name join; retain weight/bias/scale rank, shape, dtype,
scale-presence, positive exact 128 divisibility and no-scale-of-scale checks.
Scaled/FP8 rank-one bias refuses rather than disappearing.

For `R` selected records, check geometry, checked extent/byte equality, alignment,
indexed backing, Weak upgrade and Arc identity, actual/sealed size and bounds.
Acquire selected handles only into retained partial slots. All partial owners
survive until unlocked destruction. No hashes, tensor-data reads, absolute-offset
manufacture or numeric execution. Exact remaining predicates/helpers and native
API confinement/lock enforcement remain OPEN; Python gating is not their proof.

**Failure tail and separate retirement.** Growth charge may already refund on
builder failure, while partials, outer Options, Results and dispatch still live.
COLLECTION must fund that entire tail; successful work debit is never refunded.
End every guard-containing lock Result, including `PoisonError<MutexGuard>`, and
guard/closure before transient root → partials → publication → remaining activation.
Source-relocking payload/owners wait for unlock; independent Budget Arc charge
refunds may occur earlier under lock. Finish scoped alias/payload/refund work before
fresh clearing lock; do not clear early merely because `reader_live` became false.
Actual retained owner must preserve authentic host, outer lease, exclusion and
cleanup custody through cancellation/unwind and sticky failed clearing. Stack
borrow alone proves no post-unwind retention. Concrete owner/enforcement/funding
remain OPEN, not a new implemented field/API. Ambiguous clear retains custody/
fails closed, with no successful clearing, recovery or release claim.
Separately preserve Ready → backing clear → native Retire → leaseExit → final
charge, with exclusion before Ready. Mutation-free Busy permits explicit later
retry; failed post-mutation/ambiguous retirement retains original custody/charge/
lease, without inferred recovery. Selector completion is not retirement approval.
Existing `shared_allocation.rs:535–540` grows before builder; line 536 consumes
builder `Result<T, Error>` via `?` into helper `payload`, then infallibly assembles
the wrapper and returns its Result. This supports the proposed ownership sequence,
NOT verified construction: transfer success through every named boundary above
into previously empty publication, then drain scoped owner in planned cleanup,
never escape it.

**Fixed structural candidate — proposed containment, not construction/funding.**
`SelectionCustody` destruction order: `[SelectionDescriptor; 4]` →
`[Option<Arc<PublicFile>>; 3]` → WHOLE `RawFirstNormActivation`;
`OwnedChargedCustody = OwnedCharged<SelectionCustody>` remains payload → charge.
Each Copy descriptor retains original `RawGraph` enum, operation/tensor/file indices,
actual `shape: [usize; 3]` plus rank, `relative: [u64; 2]`,
`Weight/WeightScale/Bias/BiasScale` role, presence, `Missing/Null/Tensor` bias tag
and `RawScalarType`. Absent slots are inert placeholders, NEVER authority.
Rank-three storage is ONLY this restricted selector's representation: original
tensors remain rank 1..2, activation rank 3; independently validate original
operation-output `Vec<u64>` rank and conversions before copying dimensions.
Reject scaled/FP8 rank-one bias: `BiasScale` is ALWAYS absent; accepting it
invalidates the three-file bound. Relative extents grant no absolute-offset/read
authority; do not call payload-reading `data_start_into`. `RawGraph`, dtype and
reviewed error values are payload-free Copy; this proves no general allocator/error bound.

Put proposed `selector_retained: Option<SelectorRetained>` INSIDE existing `RawHost`,
alongside `root`, NOT inside `RootedRawHost.state`: retained activation's
`RawReader.root` would form a strong cycle there. `SelectorRetained` contains
`transient_root: Option<Arc<RootedRawHost>>`, three partial file Arcs,
`publication: Option<OwnedChargedCustody>`, `activation: Option<RawFirstNormActivation>`,
four descriptors and claim/failure/status tags. Existing admission-ledger chain
`CleanupSlot → _RawPublisherOwner → NativePublisher.owner → Publisher → RawHost`
roots actual Python CM/lease ownership; proposed selector enforcement DOES NOT exist.
No cleanup-performing `Drop`: explicit unlocked drain begins only after ALL
guard containers die. Sticky failure retains ORIGINAL owner/lease/exclusion,
not root self-retention alone; file-final-backing/FD/directory tail still needs proof.

**Lock anchor correction.** Locking through `retained.transient_root.as_ref()`
borrows that root SLOT until guard AND entire poison-Result scope end. Keep it
untouched throughout transaction; split mutable activation/files/descriptors/
publication/failure fields OUTSIDE locked state. Closure must not capture mutable
whole aggregate or `Option::take` root while any guard container lives. Do not
clone another Arc merely to appease borrowing without quoting it/new authority.
Error-held poison guard must Drop BEFORE unlocked drain or relocking
`RawReader::drop`, even on poison. Clearing lock follows same scope discipline.
Proposed Python in-flight/sticky fields arm BEFORE detach; `finally: _end_operation()`
is not evidence selection cleared. Sticky retirement is NOT Busy; only explicit
mutation-free Busy permits later retry. Native borrow/reader exclusion remains
unproved; the administrative Native hold was released on 7 November 2026.

Representation is frozen ONLY structurally. Exact containing layouts, entry credit,
original debit/call sites, temporary units and final-owner funder remain missing.
Let `H = size_of::<OwnedChargedCustody>()`, including padding; count guard inside
its containing Result without double counting. Measure whole `RawHost`,
`RawHostState`, `RootedRawHost`, Publisher/native variants and Python CM/error/
traceback capacities; no stack-reuse/move-elision credit or CAS retry bound follows.

**Named simultaneous containing slots; multiplicities, not compiled layout proof.**
Let `S<T> = size_of::<T>()`. The inventory names containing slots, not compiler
placement or a physical allocator bound. Distinct simultaneous slots count even
when empty; nested fields/errors count INSIDE their containers, not again as full
allocations. Reuse/disjointness must be proved, never assumed.

| Scope | Named multiplicities and overlap obligations |
| --- | --- |
| Enclosing owners | One existing `NativePublisher { owner: Option<Publisher>, pid: u32 }`, including Publisher's nested `Option<RawHost>` and other fields; no duplicate standalone host. Separately include actual Python holder, context-manager lease and cleanup owners/capacities. |
| Invocation | One `SelectorFrame` and retained aggregate inside existing RawHost: activation/publication Options and claim/failure/status fields count inside their WHOLE containing type, not duplicate outer owners. Layout remains unmeasured. |
| Transaction | Retained transient-root Option, three partial file slots and four descriptors already count inside retained aggregate; separately enumerate typed quote/count/counter/arithmetic scratch before freezing a total. |
| Lock acquisition | One `LockResult` containing its success/error variant AND one guard destination until reuse is proved. `PoisonError<MutexGuard>` retains a guard inside Result; consume/end ALL success/error containers and guard destinations before source-relocking destructors or fresh clearing lock. No unsafe poison-error formatting/deferred drop or successful poison-recovery claim. |
| Admission/helpers | External-charge Result AND charge destination, original-debit Result, and each typed helper scratch/result slot. Enumerate conversion/error scratch; no anonymous constant allowance. |
| Borrowing builder | Count ACTUAL closure/capture aggregate and branch multiplicities, with ONLY split-field references, never mutable whole retained aggregate. No captured owning activation/root/partial copy; exact capture layout remains OPEN. |
| Assembly/moves | Custody assembly slot → builder `Result<SelectionCustody, garble::Error>` → helper `payload: SelectionCustody` → `OwnedCharged<SelectionCustody>` wrapper assembly slot → wrapper `Result<OwnedCharged<SelectionCustody>, garble::Error>` → completed-wrapper move destination → PREVIOUSLY EMPTY outer publication `Option`. EXHAUSTIVELY transfer success; no ignored success, occupied-slot replacement or fallible/panicking work after transfer under lock. Captures remain references; custody owns activation only at final infallible assembly. These semantic move slots do not prove simultaneous initialized overlap, ABI size, stack bounds or move elision. Nested owners count inside containing slots, not duplicate payload/backing charges; no escaping authority. |
| Record binding | One transient Weak-upgrade `Option<Arc<PublicFile>>`/Arc destination, record reference and index scratch. Selected destination is already in partial array; not another selected-handle allocation. Source/result and destination coexist unless reuse is proved. |
| Clearing | Fresh clearing `LockResult` and guard destination are separate named slots from transaction lock slots unless their disjoint lifetimes/layout reuse are proved. Include final dispatch after unlock. |

Nested errors are included in Result/Option layouts. Error conversions, panic
runtime storage and Python exceptions are UNCENSUSED, not zero-sized/free. This
is no exact native/Python total or compiled stack-layout proof.
Builder-error conversion, moved binders, branch multiplicities and move-destination
lifetimes remain unresolved; retained-owner STRUCTURE above supplies no measured
layout, direct exhaustive-match implementation or storage-reuse credit.

For external backing `Bexternal`, successful new owned charge is
`Bexternal + S<OwnedCharged<SelectionCustody>>`. Existing `try_build` grows by
`S<OwnedCharged<SelectionCustody>> - S<OwnedExternalCharge>` before invoking
builder; do not add successful inline payload again to that charge. Original
COLLECTION requirement is the sum of distinct outer/control/helper/move slots,
separately owned capacities, and actual WHOLE RawHost/retained and root-state
containing-layout deltas (including mutex/root padding/alignment effects, not
an assumed one-byte flag increment). Separate new-wrapper coverage from existing
source-charged graph/files/upstream owners and old inline overlap; no new ledger,
cap or anonymous slot allowance. Exact overlap partition remains a gate.

Existing activation backing remains
`2 * (output.capacity() + gamma.capacity() + lookup.capacity()) + 8 * tokens.capacity()`
bytes under its old charges. Embedding activation inline overlaps the new wrapper's
containing layout; it does not create a second heap-backing charge. Existing
scratch/workspace and other inline fields remain in their containing-layout
inventory, not extra copies or an assertion that this backing expression is total.

**Audited current destruction and refund tail.** Actual declarations and Drop
bodies in [`raw_first_norm.rs`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs)
and [`raw_clear_producer.rs`](../../crates/pllm-compiler/src/replicated_decoder/raw_clear_producer.rs)
give: output full-capacity wipe/free → norm spec gamma wipe/free and gamma-file
release (with intervening fixed fields) → `OriginalReservation.charge` refund →
lookup buffer full-capacity wipe/free, scratch/workspace wipe and its charge →
tokens wipe/free, input charge → retained reader. Reader liveness clears last in
that chain, but may clear before selection's final outer tail. Proposed custody
drops selected handles before activation; `OwnedCharged` drops its entire payload
before its owned charge. Refund uses independent Budget Arc `fetch_sub`, not the
source lock. Final frame, emptied Options, escaped Results, status/move slots and
partials can outlive NEW refund. COLLECTION must cover them through final
alias/destructor/refund/dispatch tail; old activation wipe funding is not blanket
coverage for new dispatch. Keep host exclusion and `selection_live` through reader
clear and that final COLLECTION tail, then clear selection under fresh short lock.
Actual retained ownership must keep authentic host/outer lease/exclusion and
cleanup custody through cancellation, unwind and sticky failed-clear state.
Stack borrow alone proves no post-unwind retention; concrete owner, enforcement
and funding remain OPEN. Ambiguous clear retains custody/fails closed, not
successful flag clearing, recovery or release. No implemented retention API follows.

**Actual outer owner remains a separate gate.** Python `_RawPublisherOwner`
retains native owner, `_disk_cm`, `_lease`, cleanup slot/scope and gate, plus
busy/entered/closed/stopped/failure state. `DiskLease` retains capacity, token,
directory/bound/observation/backing and raw-holder state. Rust `S<T>` excludes
PyO3/Python object overhead, context-manager frames and capacity/admission-ledger
backing. Their capacities, containing layouts and callback cleanup bounds remain
unknown here; Python callback traversal is not bounded by `G`/`T`. Current native
calls detach individually: no actual authentic borrow across detach and cleanup
has yet been established. No Rust-only subtotal proves genuine outer-lease funding.
Retained-owner confinement through cancellation/unwind and failed clearing remains
an explicit lifetime/funding obligation; the 7 November 2026 administrative release
permits implementation but establishes no schema, issuer or custody proof.

**Accounting remains an EVENT VECTOR only.** Coordinates are entry/auth, quote,
G-pass, T-pass, attrs, shapes, records, header, ownership, admission, exit/refusal
and retirement tail. Preserve the preceding bounded event inventory: existing
G/T bodies AND `G+1`/`T+1` polls, initializations and terminal checks; names ≤`2G`/`2T`
comparisons and ≤`2048G`/`2048T` bytes (`N=1024`); attrs ≤2 entries, ≤4 key comparisons/
20 bytes; records ≤3 handles/4 descriptor roles; quote two conversions, two
checked multiplies and two checked adds. Growth path ≤2 logical reservations,
≤3 peak updates, with loads, Arc clone/drop and new-charge release separately
counted. All remaining fixed state/lock/dispatch/drop coefficients are UNFROZEN.
USER71483's one atomic logical invocation includes internal `fetch_update`
retries/callbacks, NOT separate calls, peak updates, `fetch_sub`, loops or
destruction. No runtime, physical-work or retry bound follows.

**USER71719 event table — finite coordinates, not approved coefficients.** These
coordinates refine, not add a second charge for, the earlier event inventory.

| Scope | Proposed bounded events; remaining predicates OPEN |
| --- | --- |
| Graph and tensors | `G`/`T` bodies and `G+1`/`T+1` iterator polls; at most `2G`/`2T` length comparisons and `2048G`/`2048T` compared bytes. Include initialization, counters, terminal tests and dispatch, not just comparison bytes. |
| Fixed graph/schema | Two graph lookups, ordered sole edge comparison `≤N` bytes after length guard; `A≤2` checked before iteration, at most four attribute-key comparisons / 20 bytes. Exact schema/predicates still OPEN. |
| Shapes/scale | Six shape conversions; scale-name comparison `≤N` bytes after length guard, two remainders and two divisions. Keep all rank/dtype/dimension/presence checks; no implicit free helper. |
| Records | `R≤3` geometry/stride/extent/alignment/indexed-backing arithmetic and bounds checks; at most `R` Weak upgrades, selected Arc clones and corresponding drops. No additional inventory walk. |
| Header | One `check_header`: two Weak upgrades/drops, two Arc identity checks, two budget checks and at most six fixed premises. This expands the earlier grouped fixed-premise item, not six new scans. Exact predicate census remains OPEN. |
| Root/admission | One transient root clone/drop. Preserve the earlier ≤2 logical reservations, ≤3 peak updates, used load, Budget Arc clone/drop and release inventory; count separate external invocations separately. |
| Quote | Two conversions, two checked multiplies, two checked adds, entry-prepaid before resulting debit. No approved `C0`, `Cg`, `Ct`. |

No hashing or data reads are included or authorized. All fixed helper, branch,
error-conversion, state, lock and destructor dispatch still needs an exact census.

**Cleanup event attribution, not new duplicate funding.** Let capacities be
`Co`, `Cgamma`, `Cl`, `Ct` and corresponding initialized lengths
`Lo`, `Lgamma`, `Ll`, `Lt`. Existing activation destruction performs
`Co+Cgamma+Cl` u16 wipe writes, plus
`(Co−Lo)+(Cgamma−Lgamma)+(Cl−Ll)` initialization writes for unused capacity,
`Lt` token wipe writes, `FRAME` scratch bytes, four workspace bytes and four Vec
frees. Preserve their old charge/work attribution; do not fund those same wipes
twice. Include loop/terminal/control events independently from write counts.
New selected-handle/wrapper destruction, refund and clear-selection dispatch
require selector prepayment plus COLLECTION tail coverage. Last-Arc retained-owner
attribution remains OPEN; Arc drop counts alone do not bound final-owner work.

Outer `CleanupSlot.drain` dispatches at most one callback; `_RawPublisherOwner.retire`
has at most three native calls and at most one context-manager exit. These are
outer dispatch counts only. Callback internals, exceptions and last-reference
destruction remain OPEN, not a bound inherited from `G`, `T` or `R`.
`DiskCapacity.observe` is NOT ordinary lease exit. If another callback reaches it,
callback execution occurs BEFORE its 4096-record check; that record cap therefore
does not bound the callback. Impose no invented callback restriction to close this gap.

**Funding boundaries — EXECUTION, COLLECTION and later retirement remain separate.**
Actual [`raw_first_norm.rs:112–114`](../../crates/pllm-compiler/src/replicated_decoder/raw_first_norm.rs#L112)
only passes `execution_demand` into `admit_input_norm_spec_with`; no selector credit
is implemented. Proposed issuer path is `first_bf16_norm` →
`admit_input_norm_spec_with(extra)` → `reserve_original`: it can extend original
EXECUTION demand only. Without separately approved changes it cannot establish
COLLECTION coverage, authentic native confinement or private entry-credit issuance.
Every authentic issuer and pre-issuance failure path remains an independent gate.

New external charge stays `Bexternal + S<OwnedCharged<SelectionCustody>>`, with
growth `Swrapper − S<OwnedExternalCharge>` where
`Swrapper = S<OwnedCharged<SelectionCustody>>`. No NEW numeric backing bound is
claimed. Old heap backing `2(Co+Cgamma+Cl)+8Ct` remains under old charges, not
double-counted; inline activation overlap with wrapper/move slots is real and
must be partitioned. COLLECTION needs distinct simultaneous slots/capacities and
the FULL root-mutex containing-layout delta, not one bool byte or a physical
allocator/RSS claim. New-charge refund does not terminate that requirement.

A future finite core quote may exclude later retirement ONLY with retained-owner
proof and separate prepayment for that retirement. Current complete TAIL funding
is NOT established. The logical atomic-retry convention prices one native logical
invocation, not arbitrary Python callbacks or physical work/retry/time bounds.
Next freeze helper scratch fields, error conversions, exact predicates and
retained-owner attribution, then obtain independent review of a finite core quote.
No complete `C0`, `Cg`, `Ct`, new tariff, framework or verifier follows here.

**Five open gates before a construction proposal can clear:**

1. ALL issuer/original execution-ledger entry and failure seams, including absent
   inseparable credit and its post-debit-only issuance/consumption.
2. Full control/helper/destructor event census → separately approved `C0`, `Cg`, `Ct`;
   none is frozen or approved by this amendment.
3. Finite exact simultaneous layout, closure multiplicities/overlap and actual
   root-state delta, including native/Python/outer-lease COLLECTION tail.
4. Lifecycle confinement and genuine lease binding across native aliases.
5. Exact Linear attribute/descriptor schema and lock-aware authentication.

No new security, cap, numerical, source-ownership or implementation approval is
created. Independent Astra review must evaluate this frozen design, not treat
the selected direction as funded or proven construction.

**USER71483 approved selector accounting ONLY.** One logical reservation invocation
includes its internal `fetch_update` retries and checked-add/filter callbacks.
Externally separate calls, owned loops, comparisons, copies, destruction and cleanup
are separately counted. This resolves only the selector's internal-callback
infinite-quote obstruction; it supplies no finite retry, time or physical-work bound,
numeric rates, blanket backend tariff, implementation, host-bridge or
exclusive-retirement approval. `fetch_sub` elsewhere and other externally separate
budget mutations do not become free or root-serialized. No callback cap or sampler
policy change follows.
Gates remain: fixed `C0` custody/rank/dtype/dimension/extent/loop/control/drop census, `B`/`Q`/layout,
complete helper/cleanup funding, numeric rates and independent selected build/native
proof. The source-authoring administrative hold was released on 7 November 2026.
Future ordinary tests only: bias/null/absent/unknown/nested attributes;
unrelated long IDs/duplicate producer; scale record before weight; foreign file/scale;
no semantic scan before funded debit (upstream entry work stays spent), byte-failure burn;
each partial acquisition, cancellation, unlock, destruction and retirement Busy.

Retain simultaneous source, windows, borrowed and owned publications, server,
masks, claims, requests, replies, encoded vectors, scratch, controls and native
backing in admission. Use capacities and actual containing layouts; no complete
total is established. Only LATER projection reuse of `prepay_primitive_work` and
`admit_reader`, within their existing scopes, follows authentic selection; proposed selector
admission above precedes semantic traversal. No C462, guard-as-authority, fresh ledger or
work refunds. Cleanup and retirement work stay prepaid.

**Terminal lifecycle and output.** Stop issuance → drain users → wipe/free
descendants → release reader liveness (and, in the proposed selector, finish its
scoped cleanup and clear `selection_live`) → `begin_retire` Ready → successful retire
→ outer lease exit → `release_final_charges`. Native transport drain, erasure and
physical backing remain symbolic obligations, not proved facilities. Existing
ordinal burns before Busy; all admitted failures remain spent. Sole eventual
accepted result is all 151,936 position-7 logits after every check. No intermediate
accepted output, token selection, persistent KV or feedback. Proposed next approval
stops at the first projection; even that approval has not been granted here.

Unapproved/unresolved: authentic first-projection join; compatible geometry/noise/
attempt profile; complete-dot bounds and whole-local-graph totality; dependent
global fixation and adaptive security; actual codes/certification and dual-LPN;
ENTRY/quote/cleanup debits, 32-draw-cap funding/qualification (implementation is
authored) and full retained-capacity totals;
native drain/erase/backing and terminal retirement; full bridge and common-model
numerics. No item is discharged by documenting it.

### Separate common-extension DAG (not an execution grant)

1. Review this graph/numeric/security contract; the former administrative freeze
   on source authoring was released on 7 November 2026.
2. Reuse shared existing `DecoderPlan`/`TrustedRawPremise` and existing Validator;
   extend shared contracts, not parallel custody. No new module route or validator
   wrapper is permitted by this document.
3. Validate original and common identities separately; derive common local/linear
   numeric rules from collected source facts. Unknown facts remain explicit gates.
4. Review dependent lifecycle, publication geometry/windows and complete resources.
5. Independently qualify numeric and security regressions with fresh lane grants.
6. Only then join the collected checkpoint through ordinary `Pipeline`/`Experiment`,
   genuine renderer, gateway and reusable benchmark experiment/output. Source
   collection, real execution and trained quality each need their own evidence and
   grants; recipe/common budgets are not measured fit or full-trained qualification.

Numeric review: check all producer edges/roles, dtype/alias facts, Q8 ties, complete
dots/no-wrap, exact scale promotion, epsilon/bias casts, SiLU, attention/RoPE and
prefill/decode/KV positions against independent exact references; no new tolerance.
Security review: keep dual-LPN assumptions, actual-code distance evidence, sparse
sampling, independence, request/query horizon, secondary checks and source-security
holds explicit and pending. Default attempt count is not whole-decoder security;
D31/D32 and native equations supply no missing certificate. No GKM substitution.

Finite future regressions: unknown operator/attribute; missing/duplicate producer;
foreign root/directory/token/plan; reordered dependency or reply; partitioned-dot
rounding or scale-index drift; wrong bias/epsilon/position; prefill/decode KV splice;
cancel/replay/tentative/cache escape; exact-budget and one-short refusal BEFORE work;
all-original/all-auxiliary barriers; no implicit model-family branch or fallback.

Open review questions: approve common BF16 linear and bias/cast derivations only
after source facts; select dependency-safe provisional lifecycle and geometry;
establish complete-root resource bounds and actual security horizon/evidence.
Until then, this is a source-supported join proposal, not an executable contract.
