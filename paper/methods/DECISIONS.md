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

**User-approved resource-contract clarification, 24 October 2026:** construction
may use phased admission on the same original counter. This does not require an
upfront assurance that the whole request will fit, and does not establish that
phased construction is implemented or that a complete request fits 64 MiB.

- Before each allocation or factory, price and reserve **all simultaneously live
  owners**: retained native, source/import and history owners, metadata, slots,
  transition/move/result overlap and scratch, together with the proposed phase's
  new backing. Keep their original charges continuously live until actual
  retirement; no reset, refund/reacquisition or phase-local counter supplies
  credit for retained owners. The all-owner 64 MiB ceiling and global 65,536
  stored-descriptor ceiling remain unchanged.
- Debit cumulative native work before each factory. Include all construction and
  execution work required by the existing contract; never reset or refund work
  between phases. Phased storage admission does not narrow protected work,
  ordinal, AES or cursor accounting.
- Original source/import history still needs its all-owner historical-peak
  issuer. A current `used` sample or a zero external baseline cannot replace that
  provenance or erase earlier construction overlap. Logical peak accounting is
  not an RSS bound; private allocator and stack coverage remain outside this
  clarification's claim.
- A late refusal burns the unpublished invocation: no expanded fallback or
  partial success. Publish only **full owning sealed Storage**, after whole
  identity, work and retirement validation and before Worker execution. Earlier
  successful phases alone confer no execution or publication authority.

This clarification changes no arithmetic profile, source binding, certificate,
wire identity or cap. Expanded remains the default; production BODY/drain gates
remain closed. A first-instruction Worker storage path does not issue
`CoordinatorDrained` or establish a globally drained router. The original parity,
certificate and lifecycle acceptance gates above still apply.

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

## D33. HE compile-only host-memory admission

**Explicitly approved by the user on 23 October 2026:** for single-job HE
compilation only, replace the free-plus-speculative-only host gate with PLLM's
existing platform host-memory admission reading. Require at least **8 GiB
(8,589,934,592 bytes) available** and **normal memory pressure** before starting.
The threshold is unchanged; macOS uses the existing framework accounting, not a
new estimate or a memory-pressure percentage substituted for bytes.

This permits ordinary pinned, locked, offline compilation with one build job,
including test-binary construction via `--no-run`. It does **not** authorize test
execution, key generation, encryption, refresh, RNG changes, dependency/pin
updates or backend/profile adoption. D4 runtime ceilings and D28's design-only
sampling restrictions remain unchanged. Reject compilation if the host gate
fails; this decision establishes neither a physical-peak proof nor complete
encrypted-execution admission.

## D34. Implement the selected Maverick numeric map

**Explicitly approved by the user on 24 October 2026:** implement and unit-test
the already selected D31/D32 conversion/import primitives and a clear correctness
oracle for the fixed eight-ID request. Preserve the selected arithmetic, scale
storage amendment, checkpoint, full vocabulary and quality-screen criteria.

This adds implementation permission, not a new numeric map or role/protocol
adaptation. It does not authorize a trained-model run, generation, security or SDK
qualification. Source integrity/finiteness/tying, honest no-wrap, protocol,
dependency, storage and whole-method admission gates remain independent. Small
independent correctness fixtures do not establish a completed trained decoder.

## D35. Execute tiny HE storage tests

**Explicitly approved by the user on 24 October 2026:** execute only the 31
compiled `staged_storage` ownership tests, serially with one test thread, after
fresh D33 host admission. Preserve the existing allocator limits and resource
guards. These tests use tiny buffers, not HE keys, ciphertexts, the solver or
refresh. D33's compilation permission otherwise remains unchanged.

This permits neither encrypted execution, RNG/dependency/pin changes nor backend
adoption. Passing storage tests does not close trusted-origin, opaque-capacity,
sampling, cryptographic or complete setup/execution admission requirements. D28
remains design-only.

## D36. Conditional bounded SIGMA causal-softmax root

**Explicitly approved by the user on 24 October 2026:** conditional local
implementation of the [one-row composition](../../docs/research/sigma-normalization-softmax-audit.md)
with one to three public active positions. Preserve ring50/Q12, signed39 maximum
comparisons, the paper-derived unsigned14-to-unsigned8 denominator path, existing
D29 nExp, exact integer inverse/floors and public inactive zeros. Do not
renormalize quantized outputs or reveal clear intermediates. This explicitly
selects the packet's paper widths rather than the conflicting fixed CPU widths.

Genuine inherited owners, fixed internal uses/fanout, canonical records and
complete cumulative work/storage admission must precede issuance. Retain the
existing 64 MiB ceiling and fresh full-ring arithmetic masks. The proposed
20/22-exchange schedules for two/three active positions require independent
validation; they are not timing or execution evidence. One active position uses
the separately admitted public code 4096 path.

This does not authorize full-length rows, general normalization, SDK routing,
transport, checkpoint conformance or joint-privacy/security qualification. Local
clear-root range checks do not mint compiler attention/source certificates.
Allocator, private callframe and physical-memory obligations remain separate.
D29's nExp component checks do not establish this outer composition.

## D37. Specify seven separately selectable method-completion contracts

**Explicitly approved by the user on 24 October 2026:** prepare the N-N, B-S,
M-X, C-S, R-S, H-F and N-G design packets in the
[bounded execution decision sheet](../../docs/plans/ten-method-execution-decisions.md).
Each packet must return a separate implementation or adoption decision:

- **N-N:** width-768 QuickSum, centered-energy layout, padding, work and owner
  ceilings; no inverse, epsilon or HE adoption.
- **B-S:** paper-derived arithmetic and exact HE/OT/key/error attachments;
  no numerical-profile or backend adoption.
- **M-X:** one priced original-eight-token storage/window/batch candidate,
  preserving parties and verification; no generation or preprocessing-to-fit.
- **C-S:** paper-aligned reciprocal numerics and immutable cache/append/refresh
  lifecycle; no evaluated-source claim or state-policy adoption.
- **R-S:** four-segment reciprocal, exact conversions, key/error graph and bounded
  qualification proposal; no generic public-LUT substitute or backend adoption.
- **H-F:** fallible entropy/fork, shuffle, erasure and resource contract; D28
  remains design-only, with no sampler implementation or RNG/pin change.
- **N-G:** explicit GELU target, candidate, domain, error and boundary policies,
  acknowledging the proven error lower bound; no new sweep or numerical adoption.

These are specification commissions, not permission to fill missing evidence with
defaults. Missing author/source, security, numerical or capacity evidence must
remain explicitly unresolved. No model runs, changed caps, disclosure changes,
runtime authority, SDK promotion or completed-method claims are authorized.
D34/D35/D36 and their existing limits remain unchanged.

## D38. Three independently bounded follow-ups to D37

**Explicitly approved by the user on 24 October 2026:** implement the following
three actions separately, using GPT-6.1 Sol for implementation and Astra for
planning/review. None adopts a numerical/backend profile, enables the SDK, raises
caps or authorizes encrypted/model execution.

1. **NEXUS clear prefix:** implement the width-768 public slot schedule,
   caller-buffered clear oracle and ordinary correctness tests in
   `N-N/PREFIX768-32x1024-v1`. Retain at most 64 MiB of local owners, at most
   `2^28` cumulative clear-work units, and every tighter/global limit. Full
   synthetic-trace construction/validation must refuse without independently
   authorized finite metadata admission; this approval supplies no such grant.
   No inverse/epsilon choice, HE profile, ciphertext/key authority or source/model
   import follows.
2. **HE Layout patch preparation only:** author in isolation the two owning
   Layout accessors, existing raw/prepared rotation-consumer bindings and bounded
   test definitions described in the secure-seam packet; Astra reviews the
   patches. Do not apply them to Main, adopt a dependency/pin, build or execute
   tests, or implement sampler/client code. Return patches and proposed dependency
   identity for separate adoption/qualification decisions. D28 remains design-only.
3. **CacheMir N-Q only:** implement and check 60 exact-rational standard-
   Goldschmidt cases plus six invalid records. One process/thread, 30 seconds,
   64 MiB working memory and 1 MiB output, with preflighted arithmetic/storage
   limits. No S-Q retained-state/planner cells, model calibration, numerical/state
   adoption or HE/native execution is authorized.

These approvals do not activate any other D37 attachment or close unavailable
source, numerical, security, capacity or actual encrypted-execution evidence.
D36's separate bounded SIGMA composition and its existing limits are unchanged.

## D39. Bounded SIGMA private party-share terminal extraction

**Explicitly approved by the user on 24 October 2026:** implement the four-lane
ring50 private GapARS12 plus private-beta terminal extraction, its resource
contracts and custody tests. The terminal produces **party shares**, not a final
reconstructed result. Reuse the existing carry machinery and codec; this is no
authorization for a new arithmetic default or protocol-wide activation.

Retain the existing fresh full-ring masks `rF` and `rBeta`. Make **one fresh
independent split of their combined mask**: sample `h0`, then set
`h1 = (rF + rBeta) - h0` modulo `2^50`. The `h0` randomness is independent of
existing material. With `f0 + f1 = q + rF`, party `b` locally computes
`f_b + b * (Beta + rBeta) - h_b`; the two shares sum to `q + Beta`.
Delivery is one-use and genuinely owned, with no mask reconstruction or
substitution. Genuine consuming owners must preserve original custody. Price and
reserve the complete simultaneous owner, transition
and scratch inventory on the original counter before buffer allocations or
factories. Include fresh randomness/material and retained delivery owners, rather
than treating external backing or an already constructed payload as free.

Error, cancellation, drop and unwind must burn the sibling operation and retain
its custody/reservations until outstanding owners retire. Parent state and layout
must price these additional owners before activation; this bounded approval does
not establish that enclosing admission or an implementation already exists.

Existing caps, default arithmetic and D17 mask requirements remain unchanged.
No LayerNorm, LUT13, numerical/backend/compiler/SDK/transport activation, final
reconstruction, joint-privacy/security qualification or passing-test claim
follows. D26 and the other historical decisions are unchanged.

## D40. Server-bound scalar source policy for checked-public ingress

**Explicitly approved by the user on 24 October 2026:** implement and test a
server-bound scalar source policy for the explicit checked-public route, including
a metadata-only source quote. Authenticated server Experiment/config supplies the
finite source dimensions and workload; a client manifest cannot supply or enlarge
that authority. Retain toy defaults. Previous source-owned SDK tests used toy
vocabularies `V <= 128`; they did not qualify checkpoint-sized ingress or execution.

Derive geometry and phase bounds from the existing model mappings, rather than a
model-name shortcut. For the Qwen example, authenticated configuration yields 24
layers, MLP width 4,864, 14 attention heads and vocabulary 151,936. Let hidden width
be `D`, head width `d`, KV heads `G`, maximum input rows `I`, and requested output
bound `O`. Existing dense-decoder mapping uses `S = I + O - 1`: prefill has query
and maximum-key lengths `I`; decode has query length 1 and maximum-key length `S`;
both retain state capacity `S`. With batch 1 and phase lengths `Q, K`, shapes are
hidden `[1,Q,D]`, queries `[1,14,Q,d]`, KV state `[1,G,S,d]`, scores
`[1,14,Q,K]`, MLP intermediate `[1,Q,4864]`, and last-hidden/output-head
`[1,D]` / `[1,151936]`. Source/config validation and checked shape/phase arithmetic
remain required; these formulas confer no source identity or historical custody.

Unchanged limits include the 64 MiB all-owner ceiling, 65,536 descriptors, batch
1, `I + O <= 40`, `O <= 8`, 16 KiB config, 1 MiB raw input, existing JSON/work
ceilings and every downstream refusal. Policy is ingress permission, **not**
whole-checkpoint-fit assurance. Metadata-only quoting must not read tensors;
checkpoint arithmetic compilation, inference, fallback, budget resets and source
authority inferred from policy are not authorized. No passing-test or completed
implementation claim follows from this decision.

## D41. Private D19 scan-to-correction/Select ownership extraction

**Explicitly approved by the user on 24 October 2026:** implement the private
D19 scan-to-correction/Select ownership and exact-resource extraction, with
ordinary compatibility, resource and lifecycle tests. Preserve existing public
APIs, index-width gates 8/10, equations, RNG draw order, full-ring masks,
`rho = 2u - r`, frame bytes and default outputs. This is ownership/resource
refactoring, not a new numerical interpretation or profile adoption.

Split one original reservation before allocation and before RNG use; retain the
same original counter and source custody without refund/reacquisition. Price all
simultaneously retained owners and transitions before their factories; preserve
failure burn and retention through retirement. The extracted resource quote is
not whole-normalization-fit assurance, and existing caps remain unchanged.

Interval scan, a shared-key profile, LUT13, LayerNorm, compiler/SDK/transport
activation, joint privacy/security qualification and physical-fit claims are
excluded. This decision authorizes implementation and tests, not a claim that
either is complete or passing. D19's original boundaries remain unchanged.

## D42. Private original-parent lookup correction construction

**Explicitly approved by the user on 24 October 2026:** implement and qualify
the reviewed private original-parent lookup8/10 correction-build and resource
extraction. Accept only the original parent's sealed identity; do not create a
public independent issuer. Existing public APIs, profiles, equations, RNG draw
order, masks, framing and D41 retirement remain unchanged. This is not a new
numerical policy or profile adoption.

Split one original aggregate reservation before allocation/RNG, with no refund
and reacquisition or unsupported charge merge. The output-mask owner retains
**both residual dealer charges**, dropping payload before either final charge
release. Preserve original-counter/source custody and failure burn. Quote exact
concrete controls and transitions before their factories; existing caps remain
unchanged. Qualification requires deterministic old goldens, exact/one-short
admission, failure/unwind/cancellation and retained-output lifecycle tests; this
decision does not claim those tests have run or passed.

Interval lookup/scan, index50, shared-key profiles, LUT13, LayerNorm and SDK
activation are excluded. No full-row-fit, physical-fit or security qualification
follows from this private extraction or its logical resource quote.

## D43. Bounded paper SIGMA four-lane LayerNorm to party shares

**Explicitly approved by the user on 25 October 2026 (user59771):** implement
and qualify one native, process-local **LocalClearRoot** four-lane paper
LayerNorm component ending in separately held additive party shares. This
approves the reviewed `sigma-layernorm4-complete-proposed-packet.md` and its
enumerated private adapters, not merely prerequisite extraction. The
[paper/source audit](../../docs/research/sigma-normalization-softmax-audit.md#layernorm-explicit-paper-contract)
remains the source distinction; the exact bounded choices approved here are:

- Ring `2^50`, Q12 private `X`, `gamma` and `beta`; check all twelve input codes
  in `[-4096,4095]` before material factories. Use integer arithmetic:
  `mu=floor(sum(X)/4)`, `Z=X-mu`, `S=sum(Z²)+1`, with the `+1` an integer unit
  in Q24 energy. No CPU-artifact omission or alternative rounding placement.
- `e=floor(log2(S))`, `m=floor(128*S/2^e)-128`; certify all 8192 entries
  `table[64*m+e]=isqrt(floor(2^57/((128+m)*2^e)))` for `m=0..127`, `e=0..63`
  using checked integer arithmetic and both square inequalities before keys.
  With `T=table[64*m+e]`, preserve both signed floors:
  `Q_i=floor(Z_i*T/4096)` and `Y_i=floor(Q_i*gamma_i/4096)+beta_i`.

The approved composition includes the private 51-interval GPU50 scanner with
two independently masked correction/Select outputs, scoped LUT13 using unchanged
GPU127 terminal arithmetic, same-input/two-use square, signed and modular
products, and the unsigned structural coordinate adapter. In particular,
`u=2^(49-e)`, `t=S*u mod 2^50`, unsigned `TR50,42`, reduction modulo128 and
embedding by64 form the ring13 index with the independently masked exponent.
`S=1`, `u=t=2^49` must not be relabeled as signed-range authority. These are
consuming original-producer/counter/use ownership adapters, including the
native-to-typed joins; no caller tuple, hash or clear readback issues authority.
The existing [D39 terminal](#d39-bounded-sigma-private-party-share-terminal-extraction)
performs the final private floor/beta step and returns party shares, not a
public reconstruction or a matrix handoff.

Before any factory, complete the **same-original-counter** resource and
cumulative-work inventory: all roots, table generation/certification, native
children, scan/correction/Select, masks, entropy/material, adapters, controls,
coordinator, retained backing, frames and final shares. No double-counting credit,
work reset or premature refund. Retain original custody through the last owner;
failure/cancellation burns both roles and dependent uses and erases unpublished
component-owned clear copies/outputs. Borrowed caller storage has **no erasure
promise**. The unchanged **64 MiB cap** may refuse the fully instantiated quote;
approval is neither a fit result nor permission to raise a limit or fall back.

Sol implementation is limited to this bounded native composition and
failure-custody qualification, followed by independent Astra review. Require
independent integer/table and full four-lane final-share oracles, exact/one-short
resource/work admission and retained-owner/failure tests. Existing profiles,
public APIs/defaults, RNG ordering and goldens remain unchanged; new scoped
profiles cannot silently widen existing lookup8/10 entry points. This decision
does not claim implementation or passing tests.

No wider rows, matrix consumer, inherited source/compiler/checkpoint authority,
SDK or trained execution, joint privacy or whole-model security qualification
is approved. No HE/FSS permission follows; **FSS remains preparation-only**.

## D44. Maverick metadata-only partition, window and lifetime planner

**Explicitly approved by the user on Sunday, 25 October 2026 (user60134):**
implement and qualify **M-X-I only**, the shared metadata-only partition/window/
lifetime planner described in [the existing candidate's separate next decisions](../../docs/plans/maverick-window-batch-candidate.md#9-separate-next-decisions-and-stop-conditions).
Bound correctness tests to **4096 records, one process/thread, 30 seconds,
64 MiB memory and 1 MiB output**. Ordinary native planner builds and qualification
require a separately allocated, Parent-named existing validation slot; this
decision grants no immediate build or execution slot.

Preserve existing global caps and **both global challenge-fixation barriers**;
unknown admission refuses rather than becoming zero. D31/D32/D34 mathematics
and clear-oracle contracts remain unchanged. The M-X proposal's 449,460-batch
full-window figures are not adopted as whole-execution evidence. The planner is
metadata, not a live capability or source/security authority.

No masks, tentative-output publication, numerical model, protocol-owner bridge,
SDK or protected decoder activation is approved. This excludes M-X numerical
failure-timing/adversarial-continuation work, output generation, checkpoint
payload reads, native HE, key generation, model/FSS runs, and security/source
qualification. No cap increase, implementation result or passing-test claim
follows from this approval.

## D45. Maverick atomic two-batch native composition

**Explicitly approved on 25 October 2026 (user60486):** implement two batches
of two claims, each 64×64, under the unchanged **16 MiB** ceiling. Extend the
[existing Maverick contracts](../../docs/plans/maverick-window-batch-candidate.md),
not its unadopted full-decoder proposal. One exclusive lease consumes two
existing attempt ordinals with fresh per-claim noise. Fix **all four originals
before any primary challenge** and **all auxiliary replies before any secondary
challenge**; release all four outputs atomically only if every check passes.
Reuse existing equations, randomness contracts, default caps and `MAX_ATTEMPTS`.

Native correctness, tamper, replay, drop and admission tests require a separately
granted validation lane. Approval is not implementation or qualification evidence.
No BF16/Q8 bridge, fixed-eight decoder, generation, SDK, security qualification
or source-author proof is approved.

## D46. Four bounded clear/operator and compiler-join adaptations

**Explicitly approved on 25 October 2026 (user60494):** implement and qualify
the following **four independent adaptations**, not whole methods or SDK routes:

- **NEXUS:** [centered-energy prefix](../../docs/plans/nexus-centered-energy-prefix.md)
  at width768, 32×1024, binary64 Equation5, **no epsilon**. Explicit user
  clarification on 25 October 2026 (user60570) selects a row-head-rooted
  ascending-tree sum broadcast to every row lane, followed by a row-head-rooted
  ascending-tree centered-square energy broadcast, under the clear
  `row_head_ascending_tree...reference_v1` policy. This resolves O603's binary64
  cancellation mismatch (3.875 versus 4.0); it is **not a bitwise continuation**
  of the unchanged legacy per-lane cyclic-prefix execution. Reject `v<=0` and
  nonfinite values; preserve ordered, non-fused public gamma/beta float
  evaluation. Ordinary f32-plus-epsilon attributes remain incompatible and must
  be rejected. This user-selected clear reference does not adopt a source/HE
  profile or CKKS inverse.
- **THOR:** [pinned artifact arithmetic](../../docs/research/thor-multiblock-indexing-audit.md)
  for two rows of width4: LinearBias → exact pinned artifact `A[gelu]`, preserving
  coefficient order/reversal, Stockmeyer evaluation, `/64` halving and final
  multiplication → LinearBias. Stop **before residual/bootstrap**; no evaluated
  run or source-evaluated build identity is claimed.
- **CacheMir:** extend [existing paired append](../../docs/plans/cachemir-numeric-state-packet.md)
  to shared ciphertext SSA/DAG analysis using **explicit synthetic capabilities**
  and old-KV parent liveness. Qualification is bounded to 30 s, 64 MiB and 1 MiB
  output; no encrypted ownership or refresh qualification follows.
- **ROSETTA:** [proposed four-segment reference](../../docs/plans/rosetta-reciprocal-round-trip.md),
  8192 pairs, rational ties-to-even, half-open domain and explicit coefficient/
  index reference, bounded to 30 s, 64 MiB and 1 MiB output. No HomComp,
  Quant4096/65536 switch, recovery, public-LUT substitution or cryptographic
  error/backend adoption is approved.

Default caps, source distinctions and unsupported-input refusals remain. Native
tests/builds require new Parent lane grants; source implementation may proceed.
Completed operator-region/compiler-join qualification must be reported separately
from whole-method qualification: no SDK ranking or method-progress percentage
increase follows from approval or these bounded results alone.

## D47. BumbleBee clear whole-phase OLT/CRT composition

**Explicitly approved on 25 October 2026 (user60505):** implement the clear
whole-phase composition described by the [executable-profile proposal](../../docs/plans/bumblebee-executable-profile-d37.md):
OLT pack → **`U(m)=floor(q*m/t)` over the full whole encoding** → actual products,
D21 trace/merge → ParseMat → global CRT and exact Down, in **both cross
directions**. Neither `floor(q/t)*m` (Delta encoding) nor nearest-Up may replace
this U. Preserve the old Delta oracle and its counterexample.

Use the proposal's ordered limbs, `N=8192`, `t=2^64`, shape `(3,65,129)` and
window `(1,64,128)` as an **explicit separate clear-only geometry**; old public
caps remain unchanged. Before allocation/factories, admit **32 MiB logical
owners** and debit at most **2^28 cumulative arithmetic operations**, with refusal
on unknown or insufficient admission and no work reset/refund. Native tests/builds
require a separately granted lane; this approval supplies no passing evidence.

**Qualification-only amendment, explicitly approved on 25 October 2026
(user60726):** before launching the nine ordinary tests, check the **complete
aggregate declared arithmetic inventory is at most `2^30`**. It includes two
complete `Q=258503359` runs covering both cross directions and one invalid-mask
nonrefundable Q reservation: **`3Q=775510077` before all helper/control costs**.
That subtotal is not permission to omit helpers; the complete checked sum must
fit before any run. Two legitimate compile-fail docs are allowed, with compiler
resources separately admitted. Per-invocation `2^28` work and 32 MiB logical-owner
limits remain unchanged: no automatic counter reset, budget escape or cap raise.
All preconstruction checks, full-U numerics and closed backend/key/noise/model/
SDK/security gates remain. This records permission, not inventory proof or passing
tests.

**One postlanding regression grant, explicitly approved on 25 October 2026
(user60824):** after CacheMir lands, permit one new combined four-package
correctness/doctest regression and strict Clippy run, including the BumbleBee
nine-test campaign **once**, under an additional `2^30` aggregate qualification
allowance separate from initial qualification. The unchanged source supplies the
same checked **780444118** BumbleBee-subset inventory; admit the complete run
before execution rather than treating that subset as unpriced remaining work.
Per-invocation `2^28` work and 32 MiB logical owners remain unchanged. Explicitly
skip the historical legacy N-N six-map campaign. Additional full-region retries
are not automatic. No production cap increase, counter escape, model execution,
HE/backend adoption, benchmark or FSS inference is authorized; all prior D47
denials remain. This grants a run, not a latest-revision passing result.

The source-observed `fhe.rs` pin remains pinned but is not an adopted runtime.
Approximate backend Down equivalence is **not proved**. No backend adoption,
key/ciphertext/noise/security qualification or full B-N1 completion follows.

## D48. Isolated pinned Poulpy owning-Layout qualification

**Explicitly approved on 25 October 2026 (user61032):** apply in isolation the
two owning `Layout` accessors against frozen Poulpy base
`ca3532570ff62b96599a74f7adb85de7726b0446`, using reviewed patch SHA256
`c61b57b786657ab0bbbf3f5d4f7fd1958434b6923c8e5e2d360441d261aa530e`.
The [existing HE Layout/source proposal](../../docs/research/he-secure-seam-implementation-packet.md)
retains its other closed gates. Offline source application/compilation and only
two storage tests covering 12 tiny cases with backing at most 4096 bytes are
permitted. The test envelope is **60 seconds, 256 MiB, zero new swap and 1 MiB
output**; unavailable dependencies or guard admission refuse execution.
Single-job compilation uses existing D33 headroom, separately from test limits.

**Narrow amendment, explicitly approved on 25 October 2026 (user61165):** permit
one standard `cargo fetch --locked --target aarch64-apple-darwin` in the exact
frozen Poulpy workspace to populate the missing cache, followed by an offline
compilation retry. No source, lock or pin changes, Main application or consumer
adoption are permitted. The initial missing-`f128` refusal remains historical;
this amendment claims no tests ran. Original D33 admission (at least 8 GiB
available and normal pressure) and the 60 s/256 MiB/zero-new-swap/1 MiB test
guards remain unchanged. Permission to fill the cache is not qualification.

This qualifies only original **owning capacity and alignment**, not a borrowed
span, noise helpers, provenance or admission. No Main application, dependency
adoption, consumer activation, RNG, keys, transforms, encryption, security or
fit claim follows. This approval records permission, not completed compilation
or passing tests; other source/runtime gates remain closed.

## D49. Explicit compact PUMA SDK source integration

**Explicitly approved on 25 October 2026 (user61089):** implement immutable
`PumaThreePartyDecoder(storage='compact')`. Omitted, default and explicit
`expanded` canonical serialization/digests remain unchanged. Storage is distinct
from argmax selection and stays outside arithmetic `ExecutionPolicy` and wire
formats. Admitted client/server Experiment composition/config binds agreement;
workers select storage **only from their retained trusted server Experiment**,
never a client manifest, request or storage boolean. Support is bounded to public
weights, checked ranges, per-token lookup, expanded argmax and HTTP. Unknown or
unsupported choices refuse; compact failure never falls back to expanded.

Both publisher and all three workers must use **source-owned compilation before
Manifest/seal**. Reuse the [original source entry](../../crates/pllm-compiler/src/replicated_decoder/source_entry.rs)
and [owning sealed request](../../crates/pllm-compiler/src/replicated_decoder/compact_request_sealed.rs),
with original counter/work/binding/source owners and snapshot disk backing charged
through transitions. No legacy expanded Publisher transplant into a fresh counter.
D26's phased same-counter admission and safe late refusal retain the 64 MiB,
descriptor, work and global caps; complete identity/work/retirement checks precede
execution. Reuse the existing Worker/PRSS. Default SDK behavior is unchanged; no
BODY handoff or drain issuer is authorized. The private 31-file sealed path landed
`0d9dcaaa`, but does **not** establish ordinary SDK qualification.

Qualification requires bounded native/component tests, a matching wheel and the
ordinary gateway generated tiny four-to-five-token V3 example: accepted `[1,0]`
with validity 1; early/later sticky rejection releases no IDs. Cover cancellation
in both snapshot phases, seal/run/retire/role drop, config/source/composition
tampering, unknown policy and default compatibility. Public serving claims require
that exact normal gateway example to pass; approval itself supplies no result.
This is not permission for unbounded snapshots, caller-controlled admission,
trained-checkpoint/tensor-model jobs or downloads, numerical/wire/profile changes,
cap increases, HE keys, FSS inference, or joint-privacy/source-authentication upgrades.

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
| D26 | Approved: opt-in compact PUMA semantic outer representation | Phased same-counter construction permitted, not upfront whole-request-fit assurance; all-owner peak/global descriptors retain 64 MiB/65,536 ceilings, cumulative native work prepaid without resets; full owning sealed Storage and original provenance/parity/certificate/drain gates required. No implemented-fit or production BODY claim; changed schedules/framing need separately approved identity. |
| D27 | Approved: bounded SIGMA D22 consuming role split | Distinct native/compiler role owners move original material, source backing and continuous reservations; unchanged coupled API, equations and eleven-exchange transcript; coordinator burns both siblings on failure; no Multiply, transport or SDK promotion. |
| D28 | Approved: secure HE client sampling seam design/review | Cover original generation and all internal expansions; preserve distributions, arithmetic and D4 gates. RNG, implementation, revised pin and security/resource contract require separate approval before generation; no qualification-only randomness exception. |
| D29 | Approved conditionally: bounded SIGMA nExp | Exact S-E local implementation; records, owner quotes, floor proofs and privacy gates remain; no full softmax/normalization, decoder or SDK promotion. |
| D30 | Approved: bounded THOR reconciliation study | Exact printed/artifact candidates and budgets only; no hybrid, numerical/backend adoption or encrypted execution. |
| D31 | Approved: Maverick mathematical map/quality criteria | Exact named numeric-only candidate; no implementation/heavy execution, achieved proof/quality qualification, security/storage/batch or generation approval. |
| D32 | Approved: Maverick pinned BF16 scale-storage amendment | Exact BF16-to-F32 promotion before unchanged D31 multiplication only; source values/integrity/ties, proofs and quality remain unproved, with no implementation or heavy-execution authorization. |
| D33 | Approved: HE compile-only framework memory gate | At least 8 GiB available and normal pressure, single-job compilation only; no tests, keys, RNG changes, runtime-cap changes or adoption. |
| D34 | Approved: implement selected Maverick numeric map | D31/D32 primitives and fixed eight-ID clear correctness oracle; no new numeric/protocol adaptation, trained-model run, generation or SDK/security qualification. |
| D35 | Approved: 31 tiny HE storage tests | Only compiled `staged_storage`, one test thread and fresh D33 admission; no HE keys/ciphertexts, solver, refresh, RNG/pin changes or backend adoption. |
| D36 | Approved conditionally: one-row SIGMA causal softmax, active count 1–3 | Paper widths, inherited nExp and exact floors; genuine owners/internal uses and full work/storage admission before issuance; no full-row, normalization, SDK, transport or joint-privacy promotion. |
| D37 | Approved: seven design-only method-completion packets | Specify N-N/B-S/M-X/C-S/R-S/H-F/N-G separately; no numerical/backend/state/RNG adoption, implementation, model execution, cap increases or manufactured source evidence. |
| D38 | Approved: three separately bounded D37 follow-ups | NEXUS clear prefix and conditional synthetic metadata only; isolated HE Layout patch preparation without adoption/build/tests; CacheMir's 66 synthetic N-Q cases within fixed limits. No full methods, HE/models, SDK or cap increases. |
| D39 | Approved: four-lane SIGMA private party-share terminal extraction | Ring50 private GapARS12 plus private beta, existing carry/codec, one fresh independent split of combined rF+rBeta and preallocated original-counter custody; sibling burn/retention and enclosing state/layout pricing required. No final reconstruction, default/cap change, LayerNorm/LUT13 or compiler/SDK/transport/security qualification. |
| D40 | Approved: server-bound scalar source policy for explicit checked-public ingress | Authenticated server Experiment/config derives finite source/workload geometry; toy defaults and all caps/refusals unchanged. Tests and metadata-only quote permitted, not client-manifest authority, tensor reads, checkpoint arithmetic compilation, inference, fallback or whole-checkpoint fit. |
| D41 | Approved: private D19 scan-to-correction/Select ownership and exact-resource extraction | Original reservation split before allocation/RNG, continuous counter/source custody and failure burn; unchanged APIs, 8/10 gates, equations, draw order, masks and framing. Compatibility/resource/lifecycle tests only; no interval scan, shared-key profile, LUT13/LayerNorm, activation, security or physical/whole-Norm fit claim. |
| D42 | Approved: private original-parent lookup8/10 correction-build/resource extraction | Parent-created sealed identity only, one original aggregate reserve split without refund/reacquisition or unsupported merge; output-mask owner retains both residual dealer charges payload-first. Exact control/transition quotes and golden/admission/failure/lifecycle qualification required; APIs/profiles/numerics/RNG/framing/D41 unchanged. No independent public issuer, interval/index50/shared-key/LUT13/LayerNorm/SDK or full-row/physical/security qualification. |
| D43 | Approved: one LocalClearRoot paper LayerNorm4 to party shares | Ring50/Q12 bounded private X/gamma/beta, exact 8192 inverse table and both floors; enumerated interval/LUT13/product/unsigned-TR42 consuming adapters and D39 endpoint. Full same-counter storage/work admission before factories under unchanged 64 MiB, failure custody and independent review required; refusal valid. No matrix/wider-row/inherited-source/SDK/trained/joint-security or HE/FSS promotion. |
| D44 | Approved: M-X-I metadata-only shared partition/window/lifetime planner | At most 4096 records, one process/thread, 30 s, 64 MiB memory and 1 MiB output; native qualification only in a separately granted existing slot. Global caps, both challenge barriers and D31/D32/D34 unchanged; unknown admission refuses. No masks, tentative publication, protocol-owner/SDK/decoder activation, numerical failure-timing/continuation, payload/model/HE/FSS execution or source/security qualification. |
| D45 | Approved: Maverick atomic two batches × two claims, 64×64 | One exclusive lease/two existing ordinals, fresh per-claim noise, all-original/all-aux global barriers and atomic four-output release; unchanged 16 MiB/defaults/MAX_ATTEMPTS. Separately granted native tests only; no decoder/SDK/bridge/security or author-proof qualification. |
| D46 | Approved: four independent bounded clear/operator/compiler adaptations | Exact NEXUS binary64/no-epsilon prefix; pinned THOR two-row LinearBias/artifact-GELU/LinearBias; synthetic CacheMir shared-SSA append/liveness; proposed ROSETTA four-segment rational reference. Preserve caps/refusals/source distinctions; separate lane qualification, no whole-method/SDK/backend/security adoption or ranking. |
| D47 | Approved: BumbleBee clear whole-phase full-U OLT/product/trace/ParseMat/CRT/Down | Proposed N8192/t2^64 limbs, separate (3,65,129)/(1,64,128) geometry, 32 MiB logical owners and 2^28 cumulative arithmetic before factories; both cross directions, old Delta counterexample retained. No runtime/backend/key/cipher/noise/security or full B-N1 qualification; approximate Down equivalence unproved. |
| D48 | Approved: isolated pinned Poulpy owning-Layout qualification | Offline apply/compile and two storage tests, 12 tiny cases/backing ≤4096 B; 60 s/256 MiB/zero new swap/1 MiB output with refusal on unavailable dependencies/guards. Single-job D33 compile headroom; no Main/dependency adoption, consumer activation, RNG/keys/transforms/encryption, security or fit qualification. |
| D49 | Approved: explicit compact PUMA SDK source integration | Immutable opt-in storage, unchanged expanded defaults/digests; trusted-server Experiment policy and original-counter source-owned publisher/all-worker compilation before Manifest/seal. Bounded matching-wheel/gateway tests required, no fallback, cap change, trained run, BODY drain or automatic SDK/security qualification. |

D6/D8 settle only their explicitly named source-derived interpretations. BumbleBee
final-paper mapping, NEXUS numerical conflicts and any further adaptation still
need concrete source evidence or their own decision. This register grants no
blanket approval and changes no measured history or complete-method status.
