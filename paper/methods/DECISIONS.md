# First implementation decisions

Status: author decisions recorded. CPU-first Apple-Silicon execution and the
common Qwen2.5-0.5B model are approved. Bounded Poulpy qualification is approved;
backend adoption for the source methods remains open. D6–D9 approve named PUMA
and FuseFSS source-derived profiles, retain BumbleBee's final-paper target, and
select a fixed-count private token-feedback contract for the four shared-computation
methods. These decisions do not qualify complete implementations or measurements.

The author has approved genuine independent reimplementations of all ten methods.
Their original protocols, arithmetic, role ownership and security requirements are
the default. Secure randomness, defined arithmetic and rejection of invalid
results are mandatory. They do not need a separate adaptation decision.

The canonical [contracts](README.md) and independent
[audits](../reproduction-contracts/README.md) separate unresolved source facts
from the following actual changes.

## D1. CPU-first execution

**Approved:** first implement and validate complete CPU configurations on Apple
M-series hardware. Use ARM64 SIMD and native CPU acceleration where exactness,
source security requirements and complete costs permit. Retain
the original CPU algorithm when a source supplies one. Label CPU ports of
GPU-evaluated methods explicitly. Implement the method's protected operations,
conversions and state; do not move them into clear client or provider kernels.

The first complete path is Gupta–Katz–Miers with two workers, dense private
float32-row lookup, original fixed-point linear delegation, secp256k1 checks,
and request-end verified acceptance. Native scalar/oracle work precedes compiler
and model integration. SIGMA's CPU branch keeps its own source schedule.

**Why needed:** the available development host is an Apple M5 with 32 GiB RAM.
It is not the CUDA hardware used by several source evaluations. CPU correctness
and measurements must not inherit the papers' GPU performance.

**Validation:** independent arithmetic and protocol oracles; original-model
correctness and quality; complete role/lifecycle checks; explicit CPU, memory,
network and storage accounting. GPU configurations require separate backend and
hardware qualification. The final fixed-provider-budget campaign is not selected
by this development decision.

## D2. Full-HE primitive dependency qualification

**Author direction:** prefer an existing native Rust backend. First examine its
actual primitives, source parameters and Apple-Silicon CPU support. A Rust wrapper
around a Go/C++ library is not a native Rust backend.

The earlier Lattigo v6.1.1 proposal was not approved. Its pinned candidate,
`37a34b4676e987795949a795f0ce9d0893763c96`, remains comparison material. PLLM
independently implements source-specific packing, lookup, scheduling, state and
role protocols regardless of primitive-library choice.

**Why needed:** the installed SEAL path has no real CKKS refresh or the required
cross-scheme conversion. Lattigo's lower-level encrypted-accumulator blind rotation
is a candidate primitive; its public-LUT wrapper is not a ROSETTA replacement.

**Validation before adoption:** actual source parameter/key graphs, client-only
secret-key custody, real refresh, required coefficient/slot and ring transforms,
encrypted-accumulator selection, target-level recovery, independent numerical
oracles, resource admission and complete primitive costs. A capability failure
requires another explicit proposal rather than a weaker protocol.

The [Rust backend review](rust-he-backend-qualification.md) identifies Poulpy's
existing CKKS refresh and Apple-M CPU primitives. D4 authorizes a bounded test of
that candidate. It does not establish that the library implements a paper.
Present specific missing capabilities and alternatives before adding a backend
in another language. NEXUS/THOR numerical differences remain separate
source-resolution decisions.

## D3. Common Qwen2.5-0.5B comparison track

**Approved:** retain original-model validation for each method, then add a named
common Qwen2.5-0.5B-Instruct track. Continue the existing research source pin:

- Repository: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Pin source: [research/common.py](../../benchmarks/research/common.py).
- The proposed common GPT-2 track is not selected. GPT-2 source-specific checks
  retain their own 1024-position bound where applicable.

Verify the existing complete checkpoint/tokenizer lock before import and freeze
full workload/quality rules before comparison. The common model choice does not
approve PLLM W8A8 as a substitute for each method's source numerical profile.

**Why needed:** the papers do not all evaluate the same model or functionality.
THOR evaluates BERT encoders; NEXUS and Maverick do not establish the complete
persistent generation lifecycle required by the proposed study. Their common
decoder configurations are extensions, not original evaluated workloads.

**Preserve:** each method's numerical rules, cryptography, roles, weight privacy,
output disclosure and permitted client operations. Compare exact correctness to
each method's own clear oracle, then apply a frozen quality criterion before cost
ranking. Do not force all methods into PLLM W8A8/ring32.

**Additional decisions remain necessary:** unresolved approximation profiles and
any new persistent KV policy. D6–D9 now settle the specified PUMA/FuseFSS numeric
choices and shared-computation token-feedback/output contract; other method
contracts remain separate.
Use full-prefix feedback where that is the evidenced original schedule; do not
silently replace it with a faster cache. Existing encrypted-cache methods retain
their own layouts and state rules. This model decision does not approve arbitrary
boundary or state substitutions.

## D4. Bounded Poulpy backend qualification

**Approved:** qualify native-Rust Poulpy v0.9.0 at commit
`ca3532570ff62b96599a74f7adb85de7726b0446` with its power-of-two torus
representation. Use a distinct backend-qualification profile on Apple-M CPU.
The approved limits are at most two encrypted refreshes, one CPU execution
thread, independent numerical checks, and an 8 GiB admission ceiling.

Freeze the dependency and toolchain, parameter/key graph, bounded fixture,
precision criteria, operations, and resource estimates before execution.
Price keys, contexts, setup, copies, and scratch together. Retain the host
reserve, monitor swap growth, and stop if admission fails. Build and setup have
separate bounded budgets. Keep this qualification outside the stable production
workspace until its toolchain and dependency requirements are qualified.

The evaluator must use public evaluation material without a secret key or
client refresh. Check every declared slot and report width, scale, layout,
error, and resource observations. A failed test does not authorize changed
thresholds or a weaker primitive.

**Approval boundary:** this decision permits the bounded torus qualification.
It does not approve replacement of NEXUS, THOR, CacheMir, or ROSETTA's source
parameters, representations, approximation schedules, or conversion protocols.
Each method still needs an explicit source mapping. TFHE-rs, OpenFHE, and
Lattigo adoption are separate decisions. No model-performance result follows
from a successful refresh test.

**First gate result:** the pinned eager `NTT4x30Portable` key bundle rejects
before allocation. Its 33 prepared automorphism keys alone need 29.78 GiB.
Borrowed preparation also keeps raw keys live, for a 37.22 GiB overlap before
other owners. No encrypted refresh or native build completed. The
[admission contract](../../benchmarks/native/poulpy_refresh/README.md) preserves
the rejected attempts and independent arithmetic. A bounded key-residency design
must retain the approved parameters and price every live owner before execution.
The 8 GiB cap remains unchanged.

The separate [staged owner review](poulpy-staged-owner-bound.md) also rejects
the proposed five-wide-ciphertext, seven-key policy when it uses the direct
public phase scratch query. Its charged subset is 8,613,846,336 bytes, already
23,911,744 bytes above the cap. The matrix-aware query has a smaller lower
bound, but no complete upper bound: allocator retention, private preparation
buffers, typed cross-role storage, and coefficient-solver setup remain unresolved.
These source calculations authorize no encrypted execution. Parameter security
and second-refresh numerical admission remain separate gates.

## D5. Ordinary SDK integration is required

**Author direction:** every method must fit the existing SDK capability contracts
so users can compose it through `Pipeline`, `Experiment`, the CLI and benchmarks.
Native or internal research execution is an implementation milestone, not a
completed method.

Define immutable component configuration and its numerical, role, verification,
state and lifecycle contracts before expanding a method-specific implementation.
Reuse component discovery, serialization, source resolution, native compilation,
role supervision, client/gateway execution and benchmark evidence. Extend a
shared contract when necessary; do not create an independent public runner,
configuration schema or method-named CLI branch.

**Required validation:** component discovery and exact round trips; Pipeline and
Experiment resolution; early rejection of unsupported compositions; generated
complete requests through the ordinary SDK and local gateway; the exact published
CLI invocation; and ordinary benchmark/replay with complete ownership accounting.
Request cancellation, partial output, malformed replies and final acceptance must
preserve each method's source contract across those entry points.

Operator-only availability and complete-response availability remain separate
capabilities. An installed operator does not qualify an unsupported decoder.
Internal references must retain their scope until their public execution path
passes these gates. The subsequent bounded GKM SDK slice now has genuine
party-local workers and verified request integration. Its generated-checkpoint
conformance does not qualify full-checkpoint execution or performance; the earlier
in-process reference retains its separate scope.
The [SDK integration contract](sdk-integration.md) records the shared extension
points and concrete end-to-end completion checks.

## D6. Approved PUMA source-derived Q18 profile

**Approved 17 October 2026 via the user question tool:** adopt the named profile
`puma.source-derived-q18.v1`, exactly as specified in
[source resolution §3.3](../reproduction-contracts/shared-profile-resolution.md#33-approved-p-n1-profile-source-derived-q18).
This selects the documented source-derived interpretation, not bit identity with
the missing evaluated graph.

The approved rules include ring64/Q18, finite binary32/toward-zero encoding,
pinned replicated arithmetic and stochastic `TruncAPr`; the stated GELU/SiLU
constants and evaluation order; shifted clipping before summation with five
softmax squares and encoded-zero exponential epsilon; and centered mean-square
LayerNorm with encoded epsilon two and the documented reciprocal/rsqrt schedules.
The explicit Llama RMSNorm/RoPE and checkpoint/frontend gaps in §3.3 remain gaps.

**Effect:** P-N1's adoption approval gate is closed for this named profile.
Complete operator/graph implementation, source-model/checkpoint locks, per-product
range proofs, mixed-kernel PRSS scheduling, protected transport, D5 SDK integration
and quality/cost qualification remain open. Original evaluated-table reproduction
still requires its missing graph/configuration evidence.

## D7. BumbleBee final-paper target

**Approved 17 October 2026 via the user question tool:** retain the NDSS final
paper as the governing target. Do **not** adopt the artifact's Q16/four-squaring profile or
combine artifact settings with final-paper settings by convenience.

**Effect:** target choice is settled; **B-N1 final-paper mapping remains open**.
Obtain the evaluated numeric/configuration/HE manifest enumerated in
[source resolution §4.4](../reproduction-contracts/shared-profile-resolution.md#44-b-n1-final-paper-mapping-remains-open):
rounding, per-node rings/scales, activation constants/endpoints, normalization,
truncation/error flags, ordered moduli/levels, noise/key switching and frontend
identity. Candidate artifact constants/primes remain evidence, not adopted
final-paper parameters. D9 selects the new transaction's output policy without
establishing the original evaluated frontend identity.
Independent HE/RLWE, noise, key-graph and circuit-privacy qualification also
remain open; candidate prime lists do not establish their security.

## D8. Approved FuseFSS source-derived descriptor profile

**Approved 17 October 2026 via the user question tool:** adopt the named profile
`fusefss.source-derived-ring50-q12.v1`: ring50/Q12 descriptors, nearest-even
**external input and weight** encoding, and SIGMA's floor-defined CPU tables,
ARS/rescales and internal evaluation order, as specified in
[source resolution §5.1](../reproduction-contracts/shared-profile-resolution.md#51-approved-f-n1-descriptor-profile).
External nearest-even encoding is not permission to change internal floor rules.

**Effect:** F-N1's adoption approval gate is closed for this named profile.
Freeze and implement every gate descriptor/LUT/coefficient/rescale/Phi byte,
secure backend and complete graph; establish range proofs, D9 boundaries,
transport, D5 SDK integration and quality/cost evidence. The missing evaluated
descriptor artifact still blocks claims of original FuseFSS numerical identity.

## D9. Fixed-count private token feedback

**Approved 17 October 2026 via the user question tool:** PUMA, BumbleBee, SIGMA
and FuseFSS use a public fixed generation count, protected full-prefix recomputation for every
selection, lowest-index protected argmax, private token-ID feedback/append,
and final **client-only token-ID reveal** after the complete loop. Equal encoded
maximum scores select the lowest vocabulary index. There is no intermediate
token/logit reveal or EOS-dependent early exit in this contract.

For SIGMA/FuseFSS, explicitly add and charge the missing protected token embedding,
final normalization and complete vocabulary head. The selected completion uses
protected equality→one-hot embedding and preserves each method's model ownership,
online roles and dealer assumptions. See the
[complete transaction](../reproduction-contracts/shared-profile-resolution.md#6-token-feedback-and-state-precise-evidenced-boundary).

**Effect:** T-N1's boundary/tie approval gate is closed. Full protected execution,
fresh per-step material, cancellation/retirement, prefill plus multiple actual
feedback selections, admission, D5 integration and quality/cost evidence remain
required. Persistent KV, streaming disclosure, stochastic sampling, full logits,
EOS-short-circuit and cross-request continuation are not approved by D9.

## D10. Approved positive-Q18 RMSNorm epsilon

**Approved via the user question tool:** use the smallest positive Q18 epsilon,
`2^-18 = 0.000003814697265625`, for common-Qwen PUMA integration. The source
epsilon `1e-6` encodes to zero under D6's binary32/toward-zero rule. This explicit
adaptation has its own identity, `puma.rmsnorm-epsilon1-q18.v1`.

RMSNorm uses uncentered, averaged squares, the approved source rsqrt and a public
per-column Q18 gamma, with each rescale retained in order. The epsilon is one
integer unit after the mean-square rescale. It is neither private-input clipping
nor permission to change LayerNorm, softmax or other coefficients. Zero rows
retain a positive denominator without opening the row.

**Effect:** the epsilon choice is settled. Native row execution, complete graph
range proofs, RoPE/source mapping, D5 integration and shared SDK quality
benchmarks retain their separate implementation and evidence gates. Historical
source-derived profiles and benchmark results retain their original identities.

## D11. Explicit public-weight PUMA adaptation

**Approved via the user question tool:** retain the private-weight method and
add a distinct public-weight mode for open checkpoints. Both retain private
prompts, activations and fixed-count token feedback. Prioritize common-compute
execution through reusable bounded storage and batched tensor operations.

The public mode may multiply public coefficients directly into each replicated
activation component. Source-certified file snapshots bound resident weight
storage; complete ring products still precede the original Q18 truncation.
Protected nonlinear operations and token selection retain fresh party-local
material. The public mode has distinct composition, execution and numeric
identities; it does not inherit private-weight confidentiality or historical
performance evidence. Whole-model ranges, quality and measured costs remain
required before practical full-checkpoint qualification.

## D12. Protected checked ranges for the public-weight PUMA completion

**Approved by the user's instruction to proceed with the complete completion
plan:** implement the [checked-range contract](puma-completion-plan.md#p1--resolve-the-entire-numeric-graph-once)
under a distinct public-weight execution and acceptance identity.

Public source/workload-bound ranges precede private execution. Membership checks
remain secret-shared. An invalid operand is privately replaced with a proved-safe
fallback before any operation whose domain would otherwise fail. A sticky shared
validity value reaches the final client-only acceptance boundary; an invalid
request releases no model tokens or reusable state. Workers execute the same
public schedule without learning the failing value or location. The client learns
the aggregate success/failure result, which belongs to this explicit contract.

Accepted requests retain the approved Q18 equations, stochastic rounding law and
piecewise tails. Fresh comparisons, products and transport are charged. Added
randomness changes execution identity and may change individual stochastic
realizations; neither historical bit parity nor a source-model quality claim is
implied. Primitive domains, complete-model correctness, bounded resources and
ordinary SDK/benchmark qualification remain required.

## D13. FuseFSS complete-decoder arithmetic

**Approved via the user question tool:** complete both the original GPT-2 track
and the separate common-Qwen track under explicit ring50/Q12 numeric identities.
This completes the remaining graph-order choices; it does not identify the
missing evaluated FuseFSS graph or qualify checkpoint quality.

`fusefss.gpt2-complete-ring50-q12.v1` retains D8 external nearest-even input and
weight encoding and the source-table nonlinear functions. Each complete linear
or attention contraction accumulates exact Q24 products, then applies one signed
floor to Q12. Linear bias is Q12 and is added after that floor. Attention scores
receive a separate multiplication by the pinned public Q12 inverse-square-root
scale and a separate floor before source softmax. No per-product truncation,
reordered attention scaling, or folded rescale is implied.

`fusefss.dense-rope-ring50-q12.v1` additionally uses uncentered source-table
RMSNorm: the complete untruncated squared sum plus integer one, the width-dependent
rsqrt lookup, and the original separate normalization/gamma floors. This differs
from the checkpoint's mean-square/epsilon rule and is not D10's PUMA adaptation.
Split-half RoPE uses pinned public finite F32 coefficient bytes encoded nearest-even
to Q12; each signed coefficient product is floored separately before addition.
For halves `l,r` and coefficients `c,s`, the results are
`floor(l*c/4096) + floor((-r)*s/4096)` and
`floor(r*c/4096) + floor(l*s/4096)`. The rotate-half negation precedes the
product's floor; replacing its term with `-floor(r*s/4096)` changes this contract.
Source positions, head layout, Q/K normalization and causal GQA remain bound by
semantic lowering. Unsupported rotary variants reject.

Both profiles preserve public weights, two online share holders, separate trusted
preprocessing, D9 full-prefix recomputation and final-only token-ID disclosure.
Checked source-derived ranges and complete live/work admission precede execution.
Independent integer parity, complete original/common-checkpoint quality and
ordinary SDK/role/benchmark qualification remain separate gates.

## D14. Opt-in checked ranges for FuseFSS Qwen completion

**Approved via the user question tool:** add an opt-in checked-range profile for
Qwen while retaining the D13 static profiles. Reuse existing range predicates,
arithmetic selection, consuming material and the shared quality-outcome path.

`fusefss.dense-rope-checked-ranges-ring50-q12.v1` retains D13 arithmetic for
accepted requests. Public source/workload-bound predicates bind the compiled
acceptance/execution identity; each private check precedes unsafe arithmetic. A failed predicate
selects a proved-safe zero before unsafe arithmetic and updates sticky shared
validity. All scheduled work still executes. Final token shares are privately
masked by aggregate validity; only the client learns final success or failure.
Invalid requests release no model tokens, reusable state or failing location.

The initial RMSNorm predicate bounds each input coordinate by the public integer
`floor_sqrt((2^49 - 2) / width)`, sufficient for the signed50 square sum plus one.
Failure replaces the complete row. Validity covers every padded row and selection.
Both normalization floors, source rsqrt tables, full-head selection and D9 private
feedback remain. Other arithmetic obligations still require proof or separately
bound checks; this approval supplies no exemption from them. Bounds cannot be
fitted to a private prompt. Every added comparison, selection, product, frame and
owner is charged, with fresh material and unchanged primitive live ceilings.

This is a changed numeric-acceptance contract, not a claim of the original paper's
evaluated implementation. Independent integer/negative controls, complete
checkpoint execution, quality (including rejected cases), SDK/gateway and matched
benchmarks remain required. Tighter public certificates may be investigated
separately; they do not retroactively qualify static-profile failures.

## D15. Explicit WAN-oriented FuseFSS online scheduling

**Approved by the user's continuation of the proposed execution rewrite:** add
public dependency-aware tensor batching and balanced private argmax. Preserve
Q12 equations, both normalization floors, per-term rotary floors, fresh per-lane
material, lowest-index ties, private feedback and final-only acceptance.

`online_schedule="wavefront-v1"` binds the execution order independently of
offline `material_delivery`. Public ready sets group the original consuming
primitives; bounded tensor tiles repeat immutable templates with distinct PCs and
material. Stable selection compares adjacent ordered subtrees, including odd
tails. Its logical tree depth does not imply one network exchange per level:
public frame, live-material and tile bounds may split each level into many batches.

The subsequent `online_schedule="tensor-v1"` implements compact public templates
and larger ready sets under the same byte ceilings. It retains separate private
lane registers and fresh material. Within the compiler's ordered tree, a strict
score comparison and left-on-tie movement replace redundant private index
comparisons: every left subtree precedes every right subtree in original leaf
order. This relies on compiler-owned tree provenance, not a claim about arbitrary
private index inputs. Numeric acceptance and final disclosure remain unchanged.

Schedule construction, live frontiers, framing, material and complete setup
remain charged under the existing ceilings. The original serial default keeps
its identity. This authorizes an execution adaptation, not changed method
arithmetic, preprocessing reuse or an automatic performance claim. Generated
integer/SDK parity and matched rate-plus-delay measurements precede practical
WAN claims; trained checkpoint acceptance, setup volume and quality remain gates.

## D16. Distinct CPU port of SIGMA's GPU-source DPF

**Explicitly approved by the user:** use a distinct CPU port of the GPU DPF
primitive at EzPC commit `aa0f2f860266fb79901405ef5e9965b8c9fe2e88`.
This narrowly supersedes D1's CPU-source choice for that primitive only. Preserve
the CPU numeric rules and existing default DPF profiles. The inspected leaking
CPU DPFET terminal remains prohibited; it is not repaired by dropping its low bit.
Source evidence and the attack's limited scope remain in
`paper/reproduction-contracts/sigma-dpf-terminal.md`.

The port retains randomized complementary root controls, the GPU source's AES
tags and byte mapping, and two complete AES-expanded terminal correction words.
For `d = n - 7`, its actual logical per-party key is `129*d + 255` bits, not the
paper's `129*d + 254`; framing, padding, live capacities and PRG work are additional
costs. CPU issuance may share identical public work across the two keys, but must
charge its actual AES expansions and never reuse secret material across queries.

This approves a source-profile adaptation, not byte-identical CPU-source fidelity,
GPU performance, an independent security proof or complete SIGMA availability.
Independent AES/point/comparison oracles, canonical malformed-input checks,
bounded allocation and erasure, native/compiler/SDK integration, trained-checkpoint
execution and quality remain separate gates. No trained-source or numeric-range
approval is implied.

## D17. SIGMA arithmetic-output mask width

**Explicitly approved by the user:** replace the pinned CPU output-mask choice
with fresh uniform masks over the complete arithmetic output ring. The pinned
CPU `SlothLRSfromWrap` and `SlothTRfromWrap`
sample literal one-bit output masks before reconstructing multi-bit arithmetic
outputs. Under the intended masked floor contract, results zero and two have
disjoint observable sets. Source trace and independent negative controls:
`paper/reproduction-contracts/sigma-arithmetic-masks.md`.

Do not adopt that output-mask choice. The approved disposition samples fresh
uniform masks over the complete output ring (`n-s` bits for TR, `n` bits for
same-ring shifts), retaining the original integer floor equations and charging
randomness, material, physical owners and delivery. Boolean masks remain one bit.
This source-view adaptation is separate from D16. Preserve existing default
profiles and bind the adaptation to a distinct source/numeric identity. Approval
does not establish implementation, composed security or SDK execution.

## D18. Bounded SIGMA owning activation prefix

**Explicitly approved by the user:** a fixed owning region, initially only
`TR6 → DReLU → Select`. Consume external capabilities once, retaining original
backing, masks and leases for four precommitted internal operand uses under a
private consuming ledger. Keep separate party owners, D16/D17 numeric contracts,
the matrix singleton and failure retirement. No capability cloning, exported
fanout, remasking or arbitrary graph execution is authorized.

The qualified local prefix and its tests are described in
`docs/research/sigma-activation-prefix-contract.md`. This decision does not
authorize full GELU/SiLU ownership, decoder integration or security claims.

## D19. Bounded SIGMA source lookup and reconstruction boundaries

**Explicitly approved by the user:** both reconstructed and deferred native
lookup boundaries in `docs/plans/sigma-source-lookup.md`, including consuming
deferred Select ownership, the specified u64-correction frame and aggregate
pre-reserved admission. Index widths are exactly 8 or 10, output widths 3–64,
lanes 1–256 and the existing logical live ceiling is 64 MiB.

Retain the audited source correction: open both the masked arithmetic result
`R=σT[x]+r` and masked sign `C=B XOR q`, then evaluate the original five-word
Select. Sample independent full-ring `r,u` and Boolean `q`; final mask is
`ρ=2u−r`, marginally uniform but parity-dependent on internal `r`. No independent
remask, division or widening is implied. D16 GPU127 mapping remains unchanged.

Reconstructed lookup opens Select's `S` and returns `2S−R`. Deferred lookup
retains distinct consuming party-share capabilities and returns `2z0−R, 2z1`;
later reconstruction remains separately admitted and charged. Frame bytes cannot
mint share owners, and reconstructed outputs cannot be relabeled as shares.
Reserve complete actual-owner overlap before issuance; retained frames/outputs
keep their leases through erasure/deallocation. Concrete accounting and numerical,
failure and independent-oracle checks remain implementation acceptance gates.

This approves bounded local protocol work, not full GELU/SiLU, typed lookup
composition, decoder/SDK activation, authenticated delivery, durable replay
protection or independent security qualification.

## D20. Bounded SIGMA full activation owning region

**Explicitly approved by the user:** the complete fixed GELU/SiLU region in
`docs/plans/sigma-full-activation.md`, including its private operand-use ledger,
38/50-bit range contract, affine/projection ownership, deferred combination,
final `PLLMAC01` reconstruction frame and continuous aggregate admission.

External input/output remain ring50/Q12; admitted encoded input is
`-2^37 <= X < 2^37`, declared as a local assumption rather than compiler or
checkpoint proof. Private ring38 TR6 produces ring32/Q6. Retain the fixed
absolute-value/clipping, D19 deferred lookup, deferred Select on original X,
role-local subtraction and final reconstruction, with no final shift or rounding.
The source-supported mode1 realization does not establish historical benchmark
settings. Original backing, masks, leases and guarded source owners survive their
last authorized uses; no exported fanout, capability cloning or remasking.

This approves bounded local implementation and its acceptance tests, not arbitrary
graphs, compiler/checkpoint ranges, decoder/SDK activation, authenticated transport,
durable replay protection or independent security qualification.

## D21: paper-derived BumbleBee packing-index corrections

The user explicitly approved these corrections for bounded pure OLT and
ParseMat algebra, after independent rectangular counterexamples exposed
inconsistent printed formulas in the locked final paper:

- Equation (1): encode a left row with stride `nw*mw`, not `kw*mw`.
  Figure 3's row starts and Proposition 1 support this interpretation.
- Algorithm 1: select the right block `V[β,γ]`, not `V[β,α]`.
- Figure 8: for group start `i` and lane `j`, use global block `t=i+j`,
  `(row_block,column_block)=divmod(t,n_prime)`, and skip padded blocks.
  Read coefficient `(r*nw+c)*mw+j`, without a group-dependent lane rotation.
- A final group is incomplete when `mw` does not divide `k_prime*n_prime`.

Document these as approved paper-derived corrections, not verified author errata
or verbatim reproduction of every printed expression. Reuse bounded negacyclic
coefficient primitives; check rectangular products, signs, partial windows and
padding against independent dense modular oracles. This decision authorizes no
HE backend, H2A/OT implementation, noise bound, final-paper numeric manifest,
decoder, SDK or security claim. B-N1 remains open.

## D22. Bounded SIGMA compiler-certified linear-to-activation handoff

**Explicitly approved by the user:** the bounded composition in
`docs/plans/sigma-compiler-linear-activation.md`: a private complete matrix dot,
exact GapARS12, then unchanged D20 GELU/SiLU, with source-bound interval admission
and continuous original-owner reservations.

Operands remain signed ring50/Q12. Certify the complete unwrapped Q24 sum inside
`[-2^48, 2^48)` before issuance; one signed floor by 4096 produces ring50/Q12
inside D20's existing input domain. No per-term floors, partial-dot substitution,
bias omission, remasking, masked sign extension, clipping or extra final shift.
Each input/weight/output matrix has at most 256 elements; existing work and
64 MiB logical allocation ceilings remain unchanged.

Bind immutable source, tensor/layout/coordinates, semantic operation, encoding,
numeric implementation and declared input domain. Use an immutable checkpoint
fixture; if only a submatrix fits, label it explicitly rather than claiming
original-layer execution. Observed extrema and public metadata are not range
proofs. Consume matching native owners without reconstructed-secret handoff,
retaining guarded backing and split original reservations through last use.

This approves the specified bounded certified handoff and acceptance tests, not
a full decoder, general numeric profile, authenticated transport, persistent KV,
checkpoint-wide range policy, SDK availability or security qualification.
D16–D20 and D9 remain unchanged.

## D23. Bounded PUMA mid-graph dispatch and deterministic test support

**Explicitly approved by the user:** stop ordinary execution immediately before
an admitted body interval, drain peer frames, and move the same original runtime
and PRSS owner through the body executor and back. Arithmetic, resource ceilings,
source ownership, sticky validity, original operation order and consuming failure
retirement remain unchanged. A local idle boundary does not establish that the
three-role router has drained; the handoff requires both.

The internal boundary must be explicit and fail closed on invalid targets,
pending-message misuse, foreign programs and protocol failure. Finish the whole
preceding original node, including all tensor-loop tiles, before yielding. No
runtime reset, second executor, automatic default selection or expanded semantic
interval is authorized.

Also approved: a disabled-by-default, explicitly insecure **test-support** feature
for fresh deterministic setup with public test entropy, solely to compare ordered
frames and local shares across control and body execution. Preserve the original
setup namespace binding. Ordinary production entropy is unchanged; live runtimes
cannot be cloned or reset. Test feature unification must not enable the helper in
ordinary production builds.

Admission, quiescence, ownership and exact transcript/share regressions precede
acceptance. Expanded oracle-capture mapping, full-checkpoint admission, SDK
promotion, security and performance qualification remain separate gates.

## D24. Bounded NEXUS GELU numerical decision experiment

**Explicitly approved by the user:** implement and run the bounded experiment in
`docs/plans/nexus-gelu-numeric-experiment.md`, using the full cited PUMA printed
coefficients as exact decimal-rational research inputs, the shorter BumbleBee
precision control, the published dyadic `f4,g_(4,1/4)` with two g then two f
compositions, threshold-difference normalization and adjacent-difference blending.
Preserve NEXUS's separate piecewise endpoint oracle. Study `[-8,8]` only as a
research domain; it is not a checkpoint/layer input certificate.

High-precision samples and bounded adaptive interval enclosures must remain
separate evidence. Retain uncertainty, inactive-branch amplification and failure;
sampled maxima do not become uniform bounds. Do not silently refit, clip, add
repetitions or move thresholds to repair the candidate.

This approves a mathematical decision experiment, not the authors' hidden
coefficient conversion, CKKS evaluation DAG/rounding/rescale/refresh parameters,
an error tolerance, an executable NEXUS profile, encrypted execution, or decoder
and SDK qualification. Existing backend and complete-request gates remain open.

## D25. Bounded THOR multi-block frame-routing correction

**Explicitly approved by the user:** retain Algorithm 2 line 7 and replace the
line 16–17 product-frame indices `j-1,j` with
`(j-floor(l/c)-1) mod M,(j-floor(l/c)) mod M`, where `M=m/c`, under the existing
bounded full-frame `c | m | n` contract. The derivation and independent printed
counterexamples are in `docs/research/thor-multiblock-indexing-audit.md`.

This is a documented paper-derived indexing correction, not verified author
errata. Preserve the single-frame case, directed rotations, interlaced heads,
delayed sums, checked work/buffers and atomic failure behavior. Retain the printed
algorithm's counterexamples and add independent dense-oracle regressions.

This approval adopts no physical CKKS slot-tail embedding, HE numeric/profile
parameters, backend, encrypted execution, encoder-to-decoder extension or SDK
method. Existing unresolved gates and resource ceilings remain unchanged.

## D26. Opt-in compact PUMA semantic outer representation

**Explicitly approved by the user:** implement an opt-in compact semantic outer
representation for complete-request admission, reusing the existing semantic body
and original consuming runtime/PRSS owners. This extends D23; it does not change
the default expanded representation or authorize a second PUMA execution engine.

- Count stored bounded semantic/loop descriptors rather than every expanded
  gather/product node, retaining the 65,536 stored-descriptor ceiling. This is an
  explicit change in the counting definition, not evidence that the old graph fit.
- Replace cumulative value charging with proven whole-program all-owner peak
  accounting, retaining the 64 MiB ceiling. Price immutable metadata, live values,
  slots, source/import owners, construction overlap, gathers, exports, frames and
  active scratch. No per-segment resets, uncharged constructors or refunded live
  owners. Borrowed external baseline and newly owned backing remain distinct.
- Permit automatic semantic-body dispatch beyond D23's binary intervals, using
  the same consuming runtime and PRSS owners, complete-node boundaries and a
  globally drained router.
- Preserve numeric equations, source bindings, complete-row/source-statistic/
  attention-energy certificates, sticky validity, original protected work and
  complete-request ordinal/AES/cursor ceilings and their definitions. Extract
  shared certificate/admission contracts rather than dropping proof obligations.
- Preserve canonical expanded identities where proved equivalent. Changed
  schedules or framing require a distinct explicitly approved identity; this
  approval does not authorize assigning such changes a frozen execution digest.

The current 4+2 expanded lowering charges 69,009,408 bytes for gate/up products
and both residual additions alone, already above 67,108,864 bytes. Therefore
metadata-only cleanup cannot rescue complete-program admission. Its historical
layer-six metadata failure is not proof of a scalar numeric-range failure.

Implementation acceptance requires complete-request ownership/work inventories,
early allocation admission and failure/retirement tests, deterministic frame,
both-share and PRSS/AES parity for representation-only changes, and positive and
negative certificate mappings before another installed publisher retry. Approval
does not establish source admission, complete-checkpoint execution, quality,
serving performance, security qualification or SDK availability.

## D27. Bounded SIGMA D22 consuming role split

**Explicitly approved by the user:** consume a fresh D22 run into two distinct
party-local handles, with native role owners and compiler wrappers. Preserve the
complete ring50/Q24 dot, exact GapARS12, unchanged ring50/Q12 D20 activation and
identical eleven-exchange transcript. The existing coupled API stays unchanged.

Move original roots, ordered-use ledger, native material, singleton backing and
leases rather than reconstructing them. Compiler wrappers retain the original
pinned-source owner and continuous external charge. Each evaluator receives only
its own role; the new coordinator must burn both siblings on failure. Any new
wrapper, coordination or transition owner must be admitted before allocation,
without releasing and reacquiring an existing reservation.

Acceptance requires transcript parity; wrong-role, session and replay burns;
original source/backing identity; exact-fit and one-short admission; retained-frame
and singleton lifetime; cancellation/drop ordering; compile-fail clone, reuse and
escape checks; and unchanged SDK-negative tests. No raw-native extraction,
serialization authority, new numeric composition, post-D22 Multiply,
authenticated transport, SDK availability or security qualification is approved.

## D28. Secure HE client sampling seam design and review

**Explicitly approved by the user:** Choice B of the
[HE client sampling decision](../../docs/plans/he-client-sampling.md).
Do not use the pinned disclaimer-bearing Poulpy `Source` for secret, mask or
error generation. Design and review a cryptographic RNG seam covering original
client generation and every internal seed expansion, preserving the frozen
distributions, numerical parameters and exact arithmetic.

This approval selects no RNG algorithm, revised dependency pin or backend wrapper.
Freeze and obtain approval for the implementation and its security/resource
contract before generation. Preserve D4 resource and role limits, original dense
and sparse secret-owner continuity, checked serial export and fail-burn lifecycle.
Complete resource, numerical and parameter-security gates remain required.

Choice A's qualification-only randomness exception is not approved. Design/review
permission establishes neither secure backend/profile adoption nor a completed
paper method, and authorizes no key generation or encrypted execution.

## D29. Conditional bounded SIGMA nExp implementation

**Explicitly approved by the user on 23 October 2026:** S-E's exact
[conditional local nExp packet](../../docs/plans/sigma-nexp.md#next-user-decision--bounded-nexp-only-after-d27).
Scope is 1–256 ring50/Q12 lanes with original-owner `0 <= X < 2^38`
provenance, Select16, private byte coordinates, two reconstructed D19 lookups,
fresh-masked consuming product and reconstructed GapARS50,12. Preserve the
specified equation, eighteen uses, eleven exchanges/twenty-two frames and
documented CPU/paper departures, 64 MiB ceiling and continuous original charges.

Reviewed canonical records, actual owner layout/quote census and exact table-floor
evidence are required before their respective construction/hash/issuance boundaries.
Conditional approval permits implementing those prerequisites, not bypassing them.
Require zero-draw invalid-root rejection, both-role burn/barrier and the fixed
`1688 + 208L` peer-byte contract. Joint-transcript privacy remains unqualified.
No new source/range authority, D22/D27 extraction, general post-D22 Multiply,
normalization, full softmax, transport, decoder or SDK promotion is approved.

## D30. Bounded THOR printed/artifact reconciliation study

**Explicitly approved by the user on 23 October 2026:** T-S's exact
[bounded study](../../docs/research/thor-multiblock-indexing-audit.md#proposed-next-bounded-experiment--approval-still-required).
Compare complete printed-literal and pinned artifact-call compositions, no hybrid
or repairs, at the declared 261 points/function with Decimal 80/160 checks and
separate binary64 diagnostics, retaining artifact Stockmeyer grouping.

One reusable Python experiment and ordinary output; one CPU process, at most
5,000 complete candidate-point evaluations, 30 seconds, 64 MiB working memory and
1 MiB output. No installs, network, checkpoints, upstream execution, native/HE or
builds. Preserve nonfinite values and precision sensitivity. Baselines are not
certified oracles. No profile/backend adoption, evaluated-run binding, quality,
encrypted execution or silent repair is approved by this study.

## D31. Maverick mathematical map and quality-screen choices

**Explicitly approved by the user on 23 October 2026:** M-N's exact
[numeric-only decision](../../docs/plans/maverick-numeric-completion.md#recommendation-and-exact-decision),
`M-N/Qwen3-4B-8prompt-candidate-1`, as a disclosed source-derived adaptation.
This selects the packet's pinned official FP8 artifact, multiplied 128×128
E4M3FN scales, nearest-even fixed-Q8 inputs/weights, direct centered-rational
division by 65,536 to BF16, specified local arithmetic/cast order and BF16-head
exception—not official dynamic-FP8 execution or recovered author numerics.

Scope remains eight IDs at positions 0–7 to all 151,936 final logits; no selection,
feedback or reusable KV. Honest no-wrap and all-canonical-reply totality proofs
remain required. Screening criteria: 1,024 pinned cases, at least 973 top-1 matches
against the specified BF16 oracle, mean NLL increase at most `ln(1.05)`, zero
failures and exact delegated/clear-candidate parity. Approval proves none of these.

Only the mathematical map and screening choices are selected. No implementation,
checkpoint acquisition or heavy execution, security/batch/storage/party/cap change
is authorized. M-B/M-C/M-S/M-D and remaining M-N evidence stay gated; no silent
numeric repairs are permitted.

## D32. Maverick pinned BF16 scale-storage amendment

**Explicitly approved by the user on 23 October 2026:** amend only D31's source
import rule for official `Qwen/Qwen3-4B-FP8` revision
`96b30dc13593a244a5e59e84687309f53c375cfa`. Its 252 block-scale tensors are stored
as BF16, contradicting the original F32-storage requirement; the
[header audit](../../docs/research/maverick-fp8-pinned-header-audit.md) preserves that
historical finding. Accept these pinned BF16 scales and promote each **exactly**
to F32 before the unchanged `R32(v*S)` multiplication and nearest-even Q8 rule.
For a finite BF16 word `b`, the F32 representation has bits `b<<16`; this preserves
the stored value, not hypothetical precision discarded before serialization.

This is the sole specified source-dtype exception, not permission to convert
other tensors, choose another pin/shape/scale, change rounding, graph, head,
workload or quality criteria. The document-only candidate name remains
`M-N/Qwen3-4B-8prompt-candidate-1`, selected by **D31 plus D32**; no runtime profile
or capability is adopted. See the audit's
[self-contained promotion proof](../../docs/research/maverick-fp8-pinned-header-audit.md#6-self-contained-exact-promotion-proof).

No implementation, checkpoint acquisition, payload read, heavy execution or
serving is authorized. Payload integrity, positive finite scale values, tied
head equality, totality/no-wrap proofs, quality and all other method gates remain
unproved. Header/index `total_size` drift is not repaired or waived. Official Qwen
metadata still does not identify the Maverick authors' evaluated artifact.

## Decision record

| ID | State | Blocks |
| --- | --- | --- |
| D1 | Approved: Apple-M CPU first | Source-faithful native development may proceed; source GPU timing claims remain separate. |
| D2 | Prefer an existing Rust backend | Poulpy is the selected qualification candidate under D4; source-method adoption remains open. |
| D3 | Approved: Qwen2.5-0.5B-Instruct | Common-model work preserves each method's numerics; substantial protocol/state resolutions still need approval. |
| D4 | Approved: bounded Poulpy torus qualification | At most two refreshes, one CPU execution thread, independent numerical checks, and an 8 GiB admission ceiling; no implied source-method mapping. |
| D5 | Required: ordinary SDK integration | Component, Pipeline, Experiment, compiler, SDK/gateway, CLI and benchmark checks gate complete-method status. |
| D6 | Approved: `puma.source-derived-q18.v1` | P-N1 adoption closed for §3.3's named interpretation; implementation/range/transport/SDK/quality and evaluated-build identity remain separate. |
| D7 | Approved: BumbleBee final-paper target | Artifact Q16 is not adopted; B-N1 evaluated-final-paper mapping remains open. |
| D8 | Approved: `fusefss.source-derived-ring50-q12.v1` | F-N1 adoption closed; nearest-even external encoding and SIGMA floor internals are explicit; full descriptors/backend/graph qualification remains open. |
| D9 | Approved: fixed-count private full-prefix feedback | T-N1 choice closed; lowest-index argmax, final client-only IDs, and charged SIGMA/FuseFSS boundary completion still need implementation and qualification. |
| D10 | Approved: `puma.rmsnorm-epsilon1-q18.v1` | Positive epsilon `2^-18`; separate numeric identity and shared SDK quality measurement required. |
| D11 | Approved: both private- and public-weight PUMA modes | Public-weight storage and batched execution have distinct contracts; full-checkpoint ranges, quality and resource measurements remain required. |
| D12 | Approved: public-weight protected checked ranges | Shared membership and safe fallback precede unsafe arithmetic; final client-only aggregate acceptance, fresh-work accounting and complete qualification remain required. |
| D13 | Approved: FuseFSS complete GPT-2 and common-Qwen arithmetic | Complete-dot floor, separate attention-scale floor, source-table RMSNorm and per-term Q12 RoPE floors bind distinct decoder numeric identities; exact execution and checkpoint quality remain unqualified. |
| D14 | Approved: opt-in FuseFSS checked ranges | Original Q12 arithmetic on accepted requests; secret safe fallback, sticky validity, final whole-request acceptance and complete work/quality qualification required. |
| D15 | Approved: explicit FuseFSS online wavefront and stable tree selection | Exact Q12 arithmetic and fresh material; bounded execution/resource identity, complete setup and measured bandwidth-plus-delay qualification required. |
| D16 | Approved: distinct CPU port of SIGMA GPU-source DPF | CPU numeric rules unchanged; randomized controls, full AES-expanded terminal, actual key/PRG costs and independent qualification required; leaking CPU terminal remains blocked. |
| D17 | Approved: full-ring SIGMA arithmetic-output masks | Literal one-bit CPU TR/LRS masks remain prohibited; fresh complete-output-ring masks preserve floor arithmetic and charge randomness, material, owners and delivery under a distinct identity. |
| D18 | Approved: bounded SIGMA owning activation prefix | TR6 → DReLU → Select only; original backing/leases and fixed internal-use ledger, without fanout, cloning or remasking. |
| D19 | Approved: both bounded SIGMA source lookup boundaries | Audited correction and dependent mask, consuming deferred Select, specified framing and aggregate admission; full activation and decoder remain separate gates. |
| D20 | Approved: bounded SIGMA full GELU/SiLU owning region | Fixed ring50/Q12 and local encoded range, private ring38/32 path, original backing/use ledger, deferred combination, PLLMAC01 final frame and continuous aggregate leases; no decoder/SDK qualification. |
| D21 | Approved: paper-derived BumbleBee OLT/ParseMat indexing corrections | Rectangular stride, right-block index and global unrotated parsing corrected explicitly; pure algebra only, with B-N1 and HE/decoder qualification open. |
| D22 | Approved: bounded SIGMA compiler-certified linear/rescale/activation handoff | Private ring50/Q12 complete dot, source-bound quarter-ring proof, exact GapARS12 and unchanged D20, bounded geometry and continuous original-owner admission; checkpoint submatrices labeled separately, no decoder/SDK qualification. |
| D23 | Approved: bounded PUMA mid-graph handoff and deterministic test support | Same original runtime/PRSS, drained router and complete-node boundary; unchanged numerics/ceilings/sticky validity, nondefault insecure test-only fresh setup, with exact parity and ownership validation still required. |
| D24 | Approved: bounded NEXUS GELU numerical decision experiment | Fixed source-derived polynomial/selector candidate, separate endpoint oracle, research-only domain and honest sampled/enclosed error; no executable CKKS/profile adoption or automatic repair. |
| D25 | Approved: bounded THOR multi-block frame-routing correction | Modular full-block routes, unchanged line-7 rotations and single-frame behavior; independent oracle and original counterexamples required, no HE/profile/decoder adoption. |
| D26 | Approved: opt-in compact PUMA semantic outer representation | Stored-descriptor counting and audited whole-program peak ownership retain 65,536/64 MiB ceilings; same runtime/PRSS and complete-request work, certificate mapping and automatic semantic dispatch need validation; changed schedules/framing need separately approved identity. |
| D27 | Approved: bounded SIGMA D22 consuming role split | Distinct native/compiler role owners move original material, source backing and continuous reservations; unchanged coupled API, equations and eleven-exchange transcript; coordinator burns both siblings on failure; no Multiply, transport or SDK promotion. |
| D28 | Approved: secure HE client sampling seam design/review | Cover original generation and all internal expansions; preserve distributions, arithmetic and D4 gates. RNG, implementation, revised pin and security/resource contract require separate approval before generation; no qualification-only randomness exception. |
| D29 | Approved conditionally: bounded SIGMA nExp | Exact S-E local implementation; records, owner quotes, floor proofs and privacy gates remain; no full softmax/normalization, decoder or SDK promotion. |
| D30 | Approved: bounded THOR reconciliation study | Exact printed/artifact candidates and budgets only; no hybrid, numerical/backend adoption or encrypted execution. |
| D31 | Approved: Maverick mathematical map/quality criteria | Exact named numeric-only candidate; no implementation/heavy execution, achieved proof/quality qualification, security/storage/batch or generation approval. |
| D32 | Approved: Maverick pinned BF16 scale-storage amendment | Exact BF16-to-F32 promotion before unchanged D31 multiplication only; source values/integrity/ties, proofs and quality remain unproved, with no implementation or heavy-execution authorization. |

D6/D8 settle only their explicitly named source-derived interpretations. BumbleBee
final-paper mapping, NEXUS numerical conflicts and any further adaptation still
need concrete source evidence or their own decision. This register grants no
blanket approval and changes no measured history or complete-method status.
