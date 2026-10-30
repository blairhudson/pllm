# CacheMir: numerical and retained-state specification packet

**Status: DESIGN ONLY, not an adopted profile or an executable method.** This
packet implements the user's C-S specification request (D37), against
`7fa07d746f55ae7157f3a371a57def920fb9fbb0`. Numerical and state choices below
are separately selectable; neither selects a backend, changes caps, or resolves
output disclosure. No calibration, model, HE, build, or network test ran here.
The cached governing PDF was read and its hash checked; artifact findings below
are inherited from the linked immutable-source audit, not fresh artifact retrieval.

**26 October 2026, user62077 C-NP update:** specification approval selects base
Meta-Llama-3-8B / WikiText-2 raw-v1 calibration protocol below, preserving standard
Goldschmidt. Coefficients, certified domains, backend and execution remain
unapproved. See [source → capability integration contract](cachemir-source-capability-join.md)
for candidate public revision pins, HTTP-401 descriptor gaps, concrete operator
dependencies, GQA/physical-layout refusals, original-owner/admission requirements
and remaining decisions. This update does not adopt §4 state policy, numerical
evaluation conventions, disclosure or any private decoder completion.

**D38 update:** only the 66 N-Q synthetic numeric cases in §5 may be implemented
and checked within their stated limits. S-Q, calibration and numerical/state or
backend adoption remain unapproved; see the [decision register](../../paper/methods/DECISIONS.md#d38-three-independently-bounded-follow-ups-to-d37).

## 1. Governing evidence and the conflict this proposal preserves

The governing source is [CacheMir v1](https://arxiv.org/abs/2602.11470v1),
12 February 2026, [local PDF](../../papers/cachemir.pdf), SHA-256
`67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`.
Appendix E, p.16 explicitly replaces THOR's aSOR reciprocal with **standard
Goldschmidt**, using newly profiled ranges for larger models. Table 8's three
example layers and Table 6's RMSE precision are not all-node range/error proofs.

The [source-recovery audit](../research/paper-module-landing-map-2026-09-23.md#cachemir-source-recovery-and-implementation-decision-sheet)
binds three separate Zenodo releases, not a moving concept DOI:

| Release | Archive SHA-256 | Interpretation here |
| --- | --- | --- |
| `10.5281/zenodo.22844188`, v0.1-ae | `15afe42bca5a2f29baddc754c5f4d2489d6cae92571d66639f8a019da3d0b295` | Historical source; archived HEAD and annotated tag target differ. |
| `10.5281/zenodo.23189548`, v0.1.1-ae | `3457613fddd4b78a34f765f391a5a4d7b489741f5f99af1d396d978f309be006` | Separate immutable release, not evaluated-run identity. |
| `10.5281/zenodo.23377151`, v0.1.2-ae | `4462530bb49ba7b4cbdbf13d7d64f7304ccabf401e3f05b8f7f75cf92846d451` | Source comparison only; no recovered commit for this archive. |

All three implement adaptive reciprocal correction, not the paper's stated
standard-Goldschmidt schedule. The latest release also uses fixed THOR ranges,
centered normalization with generated gamma/beta, and one Chebyshev SiLU fit.
These do not establish the evaluated Llama graph or its calibration. Its random
weights/cache, fixed-position append, K-before-V mutation and hardcoded refresh
targets do not constitute encrypted prefill, transactional state, placement-search
integration or an autoregressive trained-model driver.

The independent author-associated `JerryFishY/rotation-sandbox` lead at
`7c953a13e12e0646ef935eae22477c833f21d2d5` is a **different tree**: dummy `0.01`
weights/KV, commented-out cache insertion/softmax, repeated layer execution, and
no recovered trained token driver. Do not transfer release file identities to it.
The anonymous endpoint's recorded HTTP 410 `repository_expired` does not establish
artifact absence or erase the recovered Zenodo releases. The exact evaluated
checkpoint, numeric files, calibration and multi-token driver remain **unknown**.
See also [the paper contract, §6](../../paper/methods/encrypted-computation-contracts.md#6-cachemir-contract).

## 2. Reuse boundaries at this source snapshot

| Existing owner/contract | Reuse and explicit limit |
| --- | --- |
| [`interleaved_kv_layout`](../../crates/pllm-core/src/interleaved_kv_layout.rs) | Reuse `Geometry`, distinct `KeyLayout`/`ValueLayout`, append descriptors, validity masks and immutable `DerivedView` oracle. These are bounded integer slots, not ciphertexts or a refresh policy. Current geometry requires power-of-two dimensions and equal Q/KV heads; real GQA and GPT2's 12 heads are not silently supported. |
| [`slot_matrix_layout`](../../crates/pllm-core/src/slot_matrix_layout.rs) | Preserve its distinct THOR diagonal/frame types and immutable operand/output contract. Do not relabel a THOR payload extent as CacheMir slots, substitute its layout, or create a second K/V oracle. Rectangular VMM, physical RoPE and the artifact's BSGS placement still need explicit source-equivalent transforms. |
| [`encrypted_level_schedule`](../../crates/pllm-compiler/src/encrypted_level_schedule.rs) | Reuse `PublicCiphertextState`, exact capability joins, error assumptions, persistent owners, `retained_inputs` and `validate_public_ciphertext_trace`. A view preserves the parent's full backing and physical state; level/scale changes need actual transforms and fresh outputs. Metadata identities establish consistency, not source/key/error authority. |
| [`encrypted_level_dag`](../../crates/pllm-compiler/src/encrypted_level_dag.rs) | The current exhaustive and memoized solvers already search supplied fixed typed SSA graphs. Preserve full ordered parents, fixed value states/backings, all legal capabilities and stable ties. Capacity exhaustion rejects the whole optimum, not just a losing path. They do not invent source lowering, levels, errors, kernels or measured costs. |
| [`encrypted_level_trace_bridge`](../../crates/pllm-compiler/src/encrypted_level_trace_bridge.rs) | The checked borrowed serial solution/trace binding remains **chain-only**. Use the existing DAG contract for reciprocal forks, residuals, K/V joins and shared live values; never flatten them through the unary bridge. Immutable borrows do not confer execution authority. |
| [`staged_storage`](../../benchmarks/native/poulpy_refresh/src/staged_storage.rs), [`staged_preparation`](../../benchmarks/native/poulpy_refresh/src/staged_preparation.rs) | Reuse original-owner borrows, prequoted overlap, consuming phase transitions and checked-release/fail-burn patterns. These are Poulpy qualification mechanics, not CacheMir adoption. `require_original_layout` still rejects opaque owners without original capacity/alignment; a matching span, logical shape or caller byte total cannot bypass it. |
| [`SnapshotStateBasis`](../../python/pllm/runtime/transformer_client.py), [response accept/rollback](../../python/pllm/runtime/client.py) | Reuse response-local source/execution/position custody and original-error-preserving cleanup patterns. Snapshot seals do not authenticate mutable payloads. Clear/AES-at-rest decode state is not homomorphic KV. |

The existing [CKKS reference](../../python/pllm/runtime/ckks_reference.py) separates
client decryption from public evaluation but supplies no real refresh. Pending
`EncryptedKvState` must remain pending. Implement shared numerical capability and
transaction contracts when authorized, not a parallel CacheMir runtime or a
registration-only component. Logical release in an SSA trace is not observed
allocator reclamation; a modeled residency check cannot mint native release.

The DAG solvers choose order/capabilities for **one supplied fixed-state graph**;
they do not synthesize all alternative node output levels or implement the
paper's hierarchical/residual optimizer. Source lowering must explicitly supply
legal level variants and joins before such a claim. Preserve the serial complete
feasible-DAG oracle and the known printed-pruning counterexample (depth1,
input1, output2); neither greedy refresh nor missing-cost-as-zero is a substitute.

## 3. Numerical attachment: proposed freeze, with blocking empty fields

Proposed identity: `cachemir.paper-v1.standard-goldschmidt.proposed.v1` (a document
label, not a registered profile). The proposed first source target is **base
Meta-Llama-3-8B**, not Instruct, Llama-3.1, Qwen, or the artifact's dummy network.
This selects a paper-listed family only; the following proposed acquisition and
calibration protocol does **not** identify the authors' evaluated run.

| Field | Proposed selection / mandatory pin before numerical adoption |
| --- | --- |
| Model/source | Selected base `meta-llama/Meta-Llama-3-8B`; public repository candidate revision `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`. Pinned config/tokenizer descriptors returned HTTP 401; exact bytes, tensor/index/shard identities and verified source mapping: **MISSING/BLOCKED**. No tensor/index body acquired. Bind actual norm/epsilon, activation, RoPE, head tying, Q/KV heads and precision from original source, not artifact or repository identity. See [pin/conflict table](cachemir-source-capability-join.md#1-immutable-identities-conflicts-and-missing-fields). |
| Corpus | Selected WikiText-2 raw-v1: train for calibration, validation for candidate selection, test held out for quality. Public candidate `Salesforce/wikitext@b08601e04326c79dfdd32d625aee71d232d685c3`, config `wikitext-2-raw-v1`; card lists CC-BY-SA-3.0/GFDL. Exact split bytes/hashes/order, loader identity and applicable data-use review **MISSING/BLOCKED**; no rows acquired. This is not asserted to be paper preprocessing. |
| Calibration procedure | Proposed batch one; strict UTF-8 text rows in pinned split order, joined with two LF characters, no shuffle, chat template or Unicode normalization. Tokenize with special-token insertion disabled; form the first 128 disjoint complete 511-ID stream segments and prepend the pinned model BOS ID to each, giving 512 IDs/window. Insert no EOS; retain any IDs naturally produced by the pinned tokenizer. Pin row/segment offsets and BOS identity; discard an incomplete final segment and reject fewer than 128 rather than change the protocol. Record every semantic-node input/output, not only three exemplar layers. No execution of this procedure is approved here. |
| Held-out separation | Never tune coefficients/ranges on the test split. Freeze calibration records, fitting program/version, input digest and full coefficient bytes before validation/quality. Actual observations and fitted coefficients: **UNKNOWN/BLOCKED**. |
| Per-node numerical record | Actual source operator ID, layer/phase/active geometry, approximation target, input interval, scaling/shift, positive denominator bounds `a_i,b_i`, integer `m_i`, polynomial stages/coefficient bytes/order, CKKS schedule and output error budget. **No node record is populated yet.** All must be complete before adoption. |

Calibration extrema are observations, not certified input bounds. Each record
needs an independently justified admitted domain covering source inputs **and**
preceding approximation/CKKS error, including decode/cache reuse. A static public
certificate must justify `0 < a_i <= x <= b_i` before issuing inversion. Do not
open an encrypted denominator, infer its range from samples, or silently clamp it.
If private runtime range detection is required, its secure mechanism and failure
leakage need a separate contract; no such mechanism is assumed here.
Raw calibration activations remain in the authorized source-side audit boundary,
not client-visible diagnostics. Public per-node ranges, coefficients, costs and
errors can themselves be model-dependent metadata; their allowed disclosure must
be reviewed rather than inferred from input encryption or a public model family.

### Standard Goldschmidt, not released adaptive correction

For one certified positive denominator, define `u=x/b`, `e_0=1-u`, `r_0=1`.
For exactly the public count `m`, using the **old** values at each iteration:

```text
t_k = 1 + e_k
r_(k+1) = r_k * t_k
e_(k+1) = e_k * e_k
y = r_m / b
1 - x*y = (1 - x/b)^(2^m)
```

**Completed conditional error closure, not numerical/backend/key/lifecycle adoption.**
Let source target be `D`, approximate denominator `x ∈ [a,b]`, `0<a<=b`,
`|x-D|<=A<a`, and decoded input `|xhat-x|<=C<a`; then `D>=a-A>0`.
E5 must supply `A,C` from all-active-group numerator bounds, reduction and mask
errors, including prior approximation/CKKS errors; calibration cannot supply proof.
For the simultaneous old-value reference above, define
`q_k=(1-a/b)^(2^k)`, `R_k=prod_(i<k)(1+q_i)<=b/a` (empty product `R_0=1`).
Then `|e_k|<=q_k`, `|r_k|<=R_k`; the open interval `(0,1)` alone gives no uniform `m`.
Use nonnegative bounds `d_e,d_r` against these reference values. With encoded
`|c_N-1/b|<=Q_N`, normalization multiply error `eta_N`, seed error `eta_seed`,
and encoded-one/subtraction error `eta_init`, initialize and propagate:
```text
d_e0 = C/b + (b+C)*Q_N + eta_N + eta_init; d_r0 = eta_seed
t_hat = 1 + e_hat + xi_t, |xi_t| <= eta_t; h_k = d_ek + eta_tk
d_r(k+1)^- = (1+q_k)*d_rk + R_k*h_k + d_rk*h_k + eta_rk
d_e(k+1)^- = 2*q_k*d_ek + d_ek^2 + eta_ek
```
Every actual operation needs legal error-expanded ranges, scales, levels and
no-wrap bounds; each `eta` includes its actual arithmetic/rescale/alignment errors.
For **every** actual refresh on reference value `v`, certify map `B` Lipschitz
gain `L` and identity bias `|B(v)-v|<=eta_B`; propagate `d^+=L*d^-+eta_B`.
Its certified domain must contain the complete error neighborhood, not only `v`.
No refresh resets error. With final encoded `|c_F-1/b|<=Q_F`, obtain
`d_y^- = d_rm/b + (R_m+d_rm)*Q_F + eta_F`; propagate every actual final
refresh/alignment to obtain `d_y`. Consequently:
```text
|yhat-1/x| <= B_x = d_y + q_m/a
|yhat-1/D| <= B_D = d_y + q_m/a + A/[a*(a-A)]
|1-x*yhat| <= S_x = q_m + b*d_y
|1-D*yhat| <= S_D = q_m + (b+A)*d_y + A/a
```
For source probability `z*/D`, `|zhat-z*|<=Z`, `|z*|<=M`, multiplication error
`eta_p` gives error `<= (M+Z)*B_D + Z/(a-A) + eta_p`; then propagate actual
output maps. Exact-real `q_m` alone is insufficient for approximate `D`.
Select public `m` only with complete per-node bounds and separately selected
`tau>0` (not Table 6 RMSE): strict residual requires `S<tau`, or outward `U>S`
with `U<=tau`, using `S_x` or `S_D` for the declared target. Equality suffices only
for a nonstrict claim. No feasible depth/refresh/work/owner envelope means rejection,
not aSOR, extra refresh or a smaller private domain. All actual constants remain unknown.
For externally specified positive error unit `u`, nonstrict conversion is
`N=ceil(B/u)`, giving `error<=N*u`; a strict upper bound uses `floor(B/u)+1`.
Check outward arithmetic and `u64` overflow; no automatic half-ULP assumption.
Existing [`PublicCiphertextError`](../../crates/pllm-compiler/src/encrypted_level_schedule.rs#L471)
has `model_identity` and `bound: Option<u64>`; capability [`error_assumption: Option<String>`](../../crates/pllm-compiler/src/encrypted_level_schedule.rs#L549)
adds no dedicated units/gain fields. External contracts must supply units, target and all backend bounds.
Cloneable metadata/string equality establish
consistency only, not authority: bind exact source operator/map, context parameters,
key epochs, coefficient bytes/order and refresh schedule to the original noncloneable owner.
Unknown backend, noise model, domain, encoding or refresh identity bias blocks
instantiation; invent neither deterministic bounds nor statistical confidence.
This conditional proof defines no new protocol/API and adopts no numerical profile.

Freeze coefficients as exact binary64 bit patterns plus reference targets, in
ascending monomial-degree order. Proposed per-stage evaluation is descending
Horner order with explicit multiply/add/rescale boundaries; composite stage
composition, masks, scale alignment and refresh sites must be listed, not inferred
from the coefficient list. This is an independent completion choice, **not** a
claim that the paper evaluated Horner. Coefficients/scalings remain blocked until
the source-derived candidate and its depth/error admission are supplied. Keep
normalization inverse-square-root and SiLU/GELU schedules separately specified;
the reciprocal formula does not complete them or authorize fixed THOR fits.

For softmax across multiple score ciphertexts, mask inactive exponentials before
one per-head reduction over **all active groups**. Exponentiating padded zero
scores and treating the result as masked is incorrect. Positive fitted
exponentials and denominator bounds require certification. Never normalize each
`S/H` block independently, omit a partial block, or silently renormalize outputs.

## 4. Separately selectable retained-state attachment

**Proposed policy:** request-local immutable encrypted parents, derived aligned
reads, and paired staged K/V publication. It is not adopted by choosing the
numerical target. Client-only secret-key custody and the paper's semi-honest
client/server setting remain; local ownership checks do not establish malicious
integrity or an authenticated distributed commit protocol.

Each committed layer snapshot must retain actual ciphertext owners plus request,
source/numeric/backend/key epochs, generation/parent identity, public position and
active extent, allocation capacity, distinct K/V address maps, validity masks,
and per-ciphertext level/scale/error history. The same source and request owner
must authorize inputs; strings/digests and shape agreement alone cannot create
producer custody. This proposal exposes geometry only if the selected privacy
contract already permits it; private-length hiding is not silently added.

1. **Begin and preadmit.** Exclusively borrow the expected generation/position;
   reject stale, replayed, cross-request/source/key/profile or already pending
   parents. Quote all phases before any copy, factory or growth. No mutable
   ciphertext view of a committed parent escapes.
2. **Read.** Preserve parent ciphertext bytes and level/scale metadata. Ordinary
   aliases retain full original backing. Level/scale alignment or refresh creates
   charged temporary outputs through exact typed capabilities, not a `View` or
   an in-place modulus switch of retained parents.
3. **Produce and stage.** Encrypted prefill/projection must create the genuine new
   K/V values. Keep both old parents and new replacements live. A partial K tile
   receives only the new token's admitted empty-slot contribution in a fresh
   ciphertext. A boundary creates a new tile. V uses its own group/mask layout;
   never update K visibly while V remains fallible. Unchanged tiles may share
   immutable owners, whose reference/backing charges persist.
4. **Commit.** After every required layer and operation succeeds, publish K, V,
   position, masks, error/level manifests and generation as one step transaction.
   Check the original parent again at that linearization point. Per-layer local
   commits must not expose a half-completed token. Delivery/output authorization
   remains a separate unresolved boundary, not permission to emit a token here.
5. **Abort and retire.** Cancellation, partial work, malformed replies, exceptions
   or failed checks poison the pending generation; it cannot be resumed/replayed.
   Leave committed parents unchanged and retire temporary/staged owners only
   after all borrowers finish. Preserve the original exception; cleanup failures
   add context and keep uncertain backing charged. A lost ACK is uncertain, not
   permission to append twice: query the same transaction outcome or fail closed.
   Never silently retry under a fresh generation or regenerate clear cache.

Commit is initially an in-process ownership requirement. Durable/distributed
recovery, crash persistence and authenticated receipts are **not supplied** by
this design. Do not advertise them using response IDs or generic JSON framing.

### Append levels, errors and capacity transitions

Propose fixed public per-layer retained targets `ell_K,ell_V` and explicit scale
identities, filled only after source/kernel feasibility is known. Fresh placement
must arrive at or legally reach those targets. Partial-tile addition preserves
the selected level only if its actual typed addition capability does; masks and
placement are not free. Whole untouched parents do not lose a level on each read.
Any replacement/refresh is staged, never applied retrospectively to the parent.

Track public error bounds per retained tile, not just one per model:

```text
epsilon_read <= L_align*epsilon_parent + eta_align
epsilon_new_group <= epsilon_projection_and_placement
epsilon_partial <= epsilon_old_aligned + epsilon_new_aligned + eta_add
epsilon_refreshed <= L_boot*epsilon_before + eta_boot
```

Bind each coefficient, domain, mask and recurrence to the actual capability;
no numeric values are known here. Refresh decisions may depend on the frozen
public index/error schedule, never a revealed secret residual. Charge the global
refresh count across layers, staged attempts and failures; no reset per tile or
reciprocal node. If the admitted error/level horizon ends, reject before mutation.

Use the existing oracle's `t=S/d`, `G=S/H`, distinct K/V counts and addresses.
Exercise transitions at `t-1,t,t+1` and `G-1,G,G+1`. Across `G`, allocate the
next score block and `head_dim` V ciphertexts per equal-head group; preserve all
earlier tokens and a single logical attention reduction. This is a proposed
encrypted completion beyond the artifact's `seqLen+1<=S/H` guard, **not** a claim
that released code supports it. GQA, rectangular layout or unsupported geometry
must be resolved explicitly. At capacity, fail before allocation/append: no
eviction, compression, reduced context, clear rebuild or cache-method substitution.

Peak admission must count, at each actual phase, keys/contexts and original
metadata, retained old K/V, staged replacements/new groups, aliases, aligned
reads, score/exponential/reduction and reciprocal accumulators, refresh/recovery
scratch, serialization, allocator capacity and retirement overlap. Count shared
backing once only when actual ownership establishes sharing. Do not charge only
the append delta or subtract old parents at logical commit/drop; failed or
unobserved release retains its charge. The existing SSA manifest envelope is not
a measured RSS or a whole-backend capacity certificate.

## 5. Finite qualification proposal, not authorization to execute

The next useful closure is ordinary method-neutral contract/oracle tests, not a
trained run or a new harness. Proposed bounded study: one process/thread, 30 s,
64 MiB working memory, 1 MiB output; no network, model, HE, key generation or
parameter changes. These are additional restrictive study limits, not permission
to replace any existing owner/host caps. Work/metadata/collection limits must be
preflighted, including serialization and oracle arithmetic; exhaustion fails the
entire cell, not an incomplete result reported as optimal.

| Finite cell | Exact proposed scope and acceptance |
| --- | --- |
| N-Q, 66 scalar/record cases | 60 exact rational cases with `b=1`, `a/b` in `{1,1/2,1/8,1/256}`, `m` in `{0,1,2,4,8}`, and `x` in `{a,(a+b)/2,b}`. Compare the stated loop with the independent residual/product identity and old-value update order. Six invalid records separately reject zero/negative `a`, zero/negative `b`, `a>b` and negative `m`. These synthetic domains/counts are **not model calibration or adopted tolerances**. |
| S-Q, eight boundary cases | Reuse `Geometry(S=64,H=2,head_dim=4,capacity=33)`: `t=8,G=32`. For active counts `0,1,7,8,9,31,32,33`, check partial/new groups, old-token invariance, masked padding and all-group dense QK and weighted-V algebra using supplied weights against the existing independent dense oracle. These cells do not qualify softmax, exponential masking or global softmax normalization. Capacity33 rejects a further append. No ciphertext allocation. |
| S-Q, 16 fault cases | One each: stale generation, foreign request, foreign source, wrong key epoch, missing K, missing V, cancel after K, cancel after V, callback exception, consumed replay, release with live borrower, unknown capacity, unknown cost, unknown error, wrong level and wrong scale. Every failure preserves the committed parent and original error; no clear/partial fallback. |
| S-Q, eight resource cases | Exact-fit/one-short pairs for owner, work and metadata limits (six); observed modeled success/failed-release poison (two). Include old+new+scratch overlap; all failures occur before the corresponding construction. Supplied release outcomes remain modeled, not native evidence. |
| S-Q, eight planner cases | Two tiny supplied graphs (fork/join and scalar broadcast), each with roomy, exact-fit, owner-one-short and search-exhaustion limits. Compare existing exhaustive DAG and memoized outcomes, complete feasible alternatives and canonical ties. Use finite prequoted work/metadata limits, never test-only `usize::MAX`. Unknown actual kernel costs remain unknown; this is no measured optimizer result. |

This fixes 66 N-Q and 40 S-Q cells, at most 106 in the combined proposal. A
manifest that cannot be prequoted inside the study envelope rejects before a
cell starts; test-only capacity is not an execution-budget exemption.

The 106-cell enumeration does not freeze executable S-Q work. Before requesting
S-Q implementation or execution approval, freeze the two planner graphs, their
capability/state/cost tables and concrete finite search/work/metadata limits,
plus exact resource fixtures and owner/work/metadata/overlap quotes for each
resource case. S-Q remains a proposal pending those attachments; N-Q's separate
66-case proposal and limits are unchanged. Neither may execute under D37;
implementation and execution require separate explicit user approval after review.

All native HE cells remain blocked until their own original-owner, source,
parameter, sampling and finite setup/error gates close. Preserve the existing
8,589,934,592-byte HE owner envelope **including** 67,108,864 guard bytes,
separate 4,294,967,296-byte physical host reserve, zero new swap, one execution
thread, 600-second bounded setup/execution windows and at most two refreshes per
attempt. No source-sized allocation is authorized to discover whether it fits.
Poulpy's distinct torus qualification is not CacheMir RNS parameter adoption;
the paper's degree65536/slots32768/13-level/15-bootstrap-depth descriptions do not
supply exact primes, security, CKKS precision or original native capacities.

## 6. Selectable decisions and implementation boundary

| Separate choice | Exact scope; no bundled adoption |
| --- | --- |
| C-N: retain evaluated-source requirement | Wait for the actual standard-Goldschmidt numeric/calibration manifest and trained driver. Do not substitute released aSOR. |
| C-NP: specify independent paper-aligned completion | User62077 selects base Meta-Llama-3-8B / WikiText-2 raw-v1 calibration protocol in §3, subject to every blocking source pin and certificate. This is a disclosed independent completion, not recovered evaluated numerics. Coefficient/evaluation choices, certified domains, backend and execution remain unapproved; do not bundle them with protocol approval. |
| C-ST: select retained-state specification | Select §4's request-local immutable-parent/paired-commit policy separately from numerics. Native ownership, exact levels/errors/refresh horizon and capacity feasibility must still be supplied before implementation/adoption. |
| N-Q and/or S-Q: request bounded contract implementation | Separately approve the corresponding non-HE qualification cells in §5. Neither approval activates the other attachment or the full method. |

**Suggested next implementation wording:** “Implement only the selected N-Q
and/or S-Q metadata/algebra contracts and ordinary tests, reusing the existing
slot oracle, typed SSA/DAG, source/owner admission and rollback contracts. Keep
all missing pins, per-node values and physical-owner facts blocking. No model
calibration, CKKS evaluator, ciphertext/key allocation, SDK activation, new
backend/profile, budget change or output disclosure is authorized.” Actual
encrypted operators require a later dependency-complete numerical/backend packet.

The following evidence cannot be approved into existence and remains outstanding:

- Exact model/tokenizer/tensor and corpus/preprocessing pins; actual Q/KV head
  mapping and source-module semantics, especially RMSNorm versus artifact LayerNorm.
- All-node calibration observations and independent certified domains, including
  decode history, positive denominators, polynomial coefficient/evaluation bytes,
  inversion/rsqrt counts, source approximation error and selected quality budgets.
- Complete ordered prime/scale/key/security manifest, real refresh legality/error,
  secure sampling/erasure and actual original-owner capacity/cleanup evidence.
- Encrypted prefill, VMM/physical RoPE and score/value kernels, source-preserving
  lowering, supplied cost/error tables and fixed-DAG versus paper-optimizer scope.
- Feasible staged append/read/refresh recurrence through partial/new groups and
  beyond `S/H`, with whole-attempt work/refresh and full old/new residency admission.
- Independent source-sized numerical, repeated-state and held-out quality results;
  later ordinary response/lifetime measurements rather than extrapolated module costs.
- Private lookup, head, selection/tie/error, EOS, re-embedding and token feedback;
  approved output leakage and circuit-privacy treatment. **No client logits,
  per-token client decryption/argmax, clear refresh or clear KV fallback.**

This packet supplies neither a full Attention/decoder implementation nor a
performance/security result. C-S design approval and successful metadata tests
cannot close those remaining gates.

Current integration handoff is [source → capability contract](cachemir-source-capability-join.md#6-calibration-attachment-and-earliest-executable-original-region):
earliest candidate original region is block0 RMSNorm → Q/K/V projections → RoPE
and fresh paired K/V staging, still blocked by original input/source ownership,
verified configuration/tensor mapping, numerical certificates, GQA/physical
kernels and real admission. **Current next native/Python command: NONE.** Existing
`22bbf91f` / `2ace0da2` synthetic append/control coverage is reused, not rerun or
reported as new method progress. Fresh metadata recovery used five requests and
23,784 response bytes; no model/calibration/build/encrypted execution occurred.
