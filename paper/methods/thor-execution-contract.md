# THOR original-BERT whole-execution contract

Status: **D68 planning direction only; implementation and execution NOT granted**.
This document implements the documentation scope of user92529 after Astra
planning25. It requires independent Astra review; it does not assert that review
has passed. Only this file and D68 in [DECISIONS.md](DECISIONS.md) are amended.
No runtime, dependency, test or benchmark result is supplied.

## 1. Target and authority

Select a separately labelled **OpenFHE-generated RNS THOR original-BERT profile
adaptation** as the planning direction. The label describes an uninstantiated
design family, not a registered component, adopted numeric identity or executable
parameter set. Keep these tracks separate:

| Track | Authority and current limit |
| --- | --- |
| Original-paper BERT | Locked THOR PDF in `source-lock.v1.json`, SHA-256 `ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`; [method contract §5](encrypted-computation-contracts.md#5-thor-contract). Evaluated checkpoint/context binding and numerical reconciliation remain missing. |
| Generated-RNS original-BERT adaptation | Retain BERT functionality and THOR algorithm; independently qualify explicitly generated OpenFHE parameters and enumerate departures. D68 approves direction only. |
| Exact recovered basis alternative | Artifact cache has 30 Q + 4 P, 1759 total bits, degree 65536. Exact ordered basis, roots, drop order, rounding and HYBRID partitions require independent mapping; neither qualified nor selected as the generated route. |
| Common-model extension | D3/D60 common Qwen decoder/generation work has separate graph, numerics, state and acceptance. Approval of that comparison neither forces Qwen into this BERT contract nor qualifies original-paper execution. |

The generic candidate pin is **OpenFHE v1.6.1**,
`b59e1f291965f14fbee5172cd83a20f5b856204f`, as recorded in
[backend qualification §1.1](rust-he-backend-qualification.md#11-openfhe-candidate-direction-approval--4-november-2026).
That report supplies pinned source/API locators and restrictions, not a built
artifact. `CCParams`/`GenCryptoContext` generate Q and P; lower-level explicit-Q
representation does not prove arbitrary source-P import. Generated chains must
never be called the recovered basis or the authors' evaluated parameters.
Generic cryptographic-library candidacy is allowed; installation, adoption,
binding implementation and build still need explicit approval. Upstream THOR/
Liberate/PyArmor runtime is reference-only, never an implementation dependency.

The [multi-block and native-BERT audit](../../docs/research/thor-multiblock-indexing-audit.md)
pins author source at `b513c06aaabccc04b560721e35e9a98008428c25`.
Its completed logical routing audit, conditional source derivations and historical
D30 results remain intact. None becomes an encrypted or whole-BERT certificate.

## 2. Whole request: input to final client result

The paper target is BERT-base classification: 128 positions, width 768, twelve
64-wide heads, FFN width 3072, twelve encoder blocks, first-token pooler, Tanh
and task head. These are evidenced paper geometry, **not an admitted checkpoint
or an invented request**. MRPC/RTE/SST-2 are reported tasks; no particular task,
checkpoint, test split, prompt or head is selected here. Recovered MRPC config
and safetensors headers are metadata, not complete immutable weight authority.
If a fixed target cannot be recovered and approved, stop before model import.

Required complete path:

1. **Source/config:** trusted Experiment policy fixes task, immutable checkpoint,
   all tensor/dtype/shape mappings, tokenizer files/revision, config and head.
   Authenticate expected pins independently of supplied bytes. Bind source
   locations, semantic graph, numerical profile and workload to actual import
   owners; matching caller digests cannot mint source authority.
2. **Client frontend:** bind tokenization, paired-input construction, special IDs,
   truncation/padding sides, token types, position IDs and mask semantics. Bind
   the exact embedding tables and embedding normalization/dropout-evaluation
   behavior. Preserve the paper's client embedding boundary, accounting for its
   memory and disclosed model material. Artifact base-model client embeddings
   versus fine-tuned oracle embeddings remain an unresolved equality question.
3. **Client HE setup/import:** generate client secret/public/evaluation keys under
   an approved complete parameter/key graph. Import only admitted public material
   at the evaluator. Encrypt source-bound embeddings; bind request/key epoch,
   layout, physical slots, active support, scale and level to ciphertext owners.
4. **Every encoder layer:** Q/K/V affine projections → transpose/replication →
   QK scores and score scaling/mask → complete THOR softmax/inverse/update path
   with required real refresh → attention-value product → output projection,
   residual and attention LayerNorm → FF1/bias → GELU → FF2/bias → residual and
   FF LayerNorm. Preserve all twelve layer identities and layer-specific helpers;
   no omitted bias, normalization, projection, residual or private intermediate.
5. **Readout:** source-defined first-token gather, pooler affine, Tanh and complete
   task head. Return encrypted source-defined result to the client; client alone
   decrypts/decodes and applies the explicitly selected final-result policy.
   Head extent, label mapping, tie behavior and whether scores or labels are
   disclosed require an authoritative contract, not a generation default.
6. **Ordinary integration:** immutable components → `Pipeline`/`Experiment` →
   shared source admission/compiler/role execution → SDK/local gateway/CLI →
   ordinary benchmark and quality output, per [D5 integration](sdk-integration.md).
   Existing response infrastructure is decoder-oriented; BERT task/readout
   support must be extended neutrally where absent, not advertised as already
   supported or emulated with a method-specific public runner.

No isolated FFN, dense arithmetic, operator registration, refresh chain or one
layer establishes whole execution. Generated tiny fixtures test correctness but
do not qualify trained BERT or paper quality/performance.

## 3. Reuse map and finite missing implementation scope

The following are source-inspected existing owners, not HE capability grants.
Extend method-/model-neutral seams rather than copy a PUMA engine or add a
parallel THOR inventory. Concurrent PUMA repairs are outside this document's
ownership; none of these source files is edited here.

| Existing path / type | Reuse boundary; missing work |
| --- | --- |
| `crates/pllm-garble/src/shared_allocation.rs`: `Budget`, `OwnedCharged`, `OwnedExternalCharge`, admission diagnostics | Original-counter reservations, actual guarded backing and failure diagnostics. Missing: complete C++ allocation/caches/scratch/exception coverage and selected HE root binding; logical capacity is not RSS. |
| `crates/pllm-compiler/src/original_source.rs`: `OriginalSourcePlan`, private `PrimitiveAttempt` | Genuine factory provenance and cumulative consuming work, not digest adoption. Current factory stores `DecoderPlan`; shared encoder/source generalization requires review, not pretending it already admits BERT. Metadata entry is fail-closed/uninhabited, not an available HE issuer. |
| `crates/pllm-compiler/src/verified_source.rs`: `SelectedSourceWorkspace`, selected-byte verification | Independently expected source binding and admitted verification workspace. Missing full BERT config/tokenizer/tensor authority and complete import/lifecycle accounting; byte equality is not publisher authentication. |
| `crates/pllm-compiler/src/replicated_decoder/source_entry_ingress.rs`: `OwnedSourceIngress`, `SourceHeaderEntry`, `SourceCollector<T>`, private `CollectedSource<T>` | Existing consuming move-capsule pattern keeps payload, original source, controller and root together. Extract common custody only after review; these private replicated-decoder types are not callable HE ingress. Preserve their last-fallible-call and payload-before-lease semantics; do not copy fresh roots. |
| `crates/pllm-compiler/src/encrypted_level_schedule.rs`: `PublicCiphertextGraph`, `PublicCiphertextState`, `PublicCiphertextCapability`, owner/trace types; `encrypted_level_dag.rs`: `CheckedPublicEncryptedDag`, `PublicEncryptedDagCapacity` | Typed SSA joins, backing aliases, level/capacity transitions and finite planning. Missing actual backend-issued capability/error/owner mappings and full BERT lowering. Consistency checks and manifest owner limits are not cryptographic authority or `Budget` admission. |
| `crates/pllm-core/src/slot_matrix_layout.rs`: `Layout`, `lower_mm`, `optimized_lower_schedule`, `optimized_lower_mm`, `rectangular_block_pc_mm`, `BlockPcPhysicalPlan` | Reuse D25 logical routing, rectangular clear PC-MM and existing bounded physical/ideal-slot descriptions under their own scopes. `PhysicalScaleContract`/`PhysicalIdealWork` supply symbolic compatibility/logical events, not adopted scales, HE tariffs or executable capabilities. Missing backend codecs, qualified transpose/replication and paired CC-MM, complete gains, lazy tensor/rescale/relinearization and whole-graph error certificates. |
| `crates/pllm-compiler/src/execution_capsule.rs`: `ExecutionCapsule` | Public source-locked metadata only; current prefill/decode/greedy-selection assumptions do not carry consuming source custody and do not execute BERT. Never confuse this with private ownership move capsules. |
| `python/pllm/runtime/transport.py`: `PLLMTransport`; `private_boundary_transport.py`: `PrivateBoundaryTransport` | Reuse SDK interception, owned response and lifecycle patterns, not their existing decoder/two-share payload protocols as CKKS. Missing authenticated HE key/ciphertext framing, expanded-length admission, result ownership and cancellation/error integration. |

Finite future implementation work packages: (A) shared encoder/frontend/source
contracts, (B) qualified generic HE identity/custody and primitives, (C) THOR
packing/kernels and numerical composition, (D) complete encoder/readout lowering,
(E) transport/SDK lifecycle and ordinary benchmark integration. Their dependencies
are A+B before C/D, and admitted A–D before protected E execution. This is a scope
map, not permission to implement those packages now.

The existing bounded physical PC-MM legacy/scaled-support proposals are distinct
from the generated-RNS whole-BERT direction. Preserve their symbolic-prime,
mask-rounding, signed-key and ideal-slot limitations; neither imports an adopted
`Delta`, repairs legacy `P-U` by retagging, nor proves whole-model CKKS tails.

**Ownership split:** backend owns scheme/context and key identity, ordered bases,
levels, scales/noise degree, arithmetic/refresh semantics and foreign lifetimes.
THOR owns packing, masks, operation ordering and numeric composition. Compiler
binds both to genuine source/workload custody; transport cannot manufacture
either from serialized labels. Missing primitives must refuse, never route to
plaintext server computation.

## 4. Whole-graph packing, gains and numerical acceptance

Freeze one operation-level graph for the selected target, including every mask,
scalar, bias, rotation, conjugation, extended tensor, delayed sum, relinearization,
rescale, alignment, refresh and residual join. Record logical-to-physical slots,
zero holes/tails, directed rotations, interlaced heads and all ordered channel
meanings. D25 logical full-frame routing does not alone prove CKKS embedding.
The audit's twelve logical heads in sixteen physical lanes is conditional evidence,
not permission to admit arbitrary geometry or nonzero holes.

Required certificates, composed across **all layers and head**:

- **Packing/gain:** prove encode→operation→decode against independent dense
  coordinates. Distinguish zero-diagonal `scale_pt` from nonzero masked terms.
  Retain audit decomposition `Z=e*D+(e0-e)*P0+(K-e)*sum(T_l(P_l))`; derive actual
  complement-route norms/multiplicities, stored deviations, mask rounding and
  true removed-prime rescale effects. Neither nominal `2^41` nor a uniform gain
  factor certifies the routed product. Generated primes need a new derivation;
  old-source deviations/gain cancellation do not transfer automatically.
- **Paired channels:** for `z=x+i*y`, retain ordered channels and joint error on
  jointly perturbed domains, without independence assumptions. Preserve
  `z+conj(z)=2Re(z)` and `i*(conj(z)-z)=2Im(z)`. Pre-GELU merge has a preceding
  half; FF residual merge does not. Real/shared RHS is proved only under the
  audit's ideals; quantify imaginary leakage and resulting swap defect rather
  than assuming exact real ciphertexts. No implicit half/double normalization.
- **Nonlinear domains:** resolve complete coefficients/order, caller factors,
  GELU/Tanh conflict, softmax layer profiles/masking, actual reciprocal state,
  rounded root choice and finite inverse-square-root stopping behavior. Cover
  actual rounded positive denominator domains and whole-model reachable inputs,
  including perturbations. Helper intervals/sampled extrema are not certificates;
  uncovered, all-masked or nonfinite cases need explicit refusal/approved policy.
- **LayerNorm:** reconcile checkpoint epsilon with helper/caller effective epsilon,
  mask rounding, `p`/`q` product drift and sign of normalized inverse-root output.
  Initial interval shortfall remains open. The audit's deferred `13/500` example
  is **not an adopted tolerance**. No epsilon swap, clipping, fit or root sorting.
- **Refresh capacity:** specify entry domain/scale/noise degree, ordered paired
  payloads, output error, actual usable output levels and required next operations.
  Check consumption → refresh → downstream operations → later refresh within the
  complete graph. OpenFHE clone-return or more towers alone cannot prove restored
  usable capacity; `FIXEDMANUAL` still has internal adjustments. Pin `COMPLEX`,
  `HYBRID`, setup/precompute/packing mode and complete traces where applicable.
  Include the candidate's real/imaginary split and both internal bootstrap paths
  for complex slots, including sparse packing; never assume one real refresh
  proves an ordered complex pair.

Use **independent eager FP32 BERT** on exactly pinned tokenizer/config/tensors and
input semantics as the model reference. It must not call THOR kernels or use
their approximation output as its expected result. Separately use independent
dense/exact-rational or interval source oracles for packing and selected numeric
equations; clear approximation parity is not FP32 model accuracy.

Derive per-edge rounding/approximation/encryption/refresh bounds and propagate
them through linear operator norms, nonlinear sensitivity, reductions and
residual joins to final scores and task decisions. Define absolute/relative/RMSE
metrics, zero-reference treatment and decision-margin/quality criteria **before
execution**, justified by this propagation and source/task requirements. Missing
source tolerances or changed acceptance return to the user under D61; no guessed
numeric threshold, retrospective tuning or borrowed paper precision statistic.

## 5. Keys, security, disclosures and foreign ownership

The client alone owns secret keys/decryptors. The evaluator receives public key,
evaluation material and ciphertexts only; clear embeddings/activations, plaintext
server fallback and server secret-key copies are prohibited. If client-assisted
refresh is ever proposed, separately approve its roles, client views, frequency,
traffic and costs; it cannot masquerade as ciphertext-only server bootstrap.

Before setup, freeze the complete key graph: key epochs, secret/error laws,
public encryption, tensor/relinearization, every actual rotation and conjugation,
bootstrap precomputations/switching and any dense/sparse/encapsulation relationships.
Account for keys at each retained basis and for generated HYBRID QP, decomposition,
roots and security estimates. `HEStd_128` alone does not qualify the complete
graph; sigma `3.2` versus `3.19f`, sampling tails/rejection exhaustion and the
actual randomness expansion paths require explicit treatment. No generic attempt
count bounds unbounded internal RNG work.

Bind paper input privacy and client-only decrypted results without inventing
model confidentiality: client embedding tables are part of its view. Separately
resolve server-weight confidentiality, client-visible evaluated ciphertext/error
distributions, circuit privacy and any masking requirements. IND-CPA or experimental
noise flooding is not bootstrap-composition/circuit-privacy proof. Honest-but-curious
paper scope is not malicious-result verification.

Approve a complete disclosure table: task/geometry/physical length, key metadata,
token support/count, embeddings available at client, final scores/labels and
failure/timing traffic. Artifact mask reconstruction uses count-derived prefix
support; arbitrary holes/left padding are not preserved and evaluator-visible
plaintext masks expose support. Fixed 128 positions does not erase that disclosure.
Neither a public-weight adaptation nor new mask leakage is selected here.

Foreign-owner graph must include C++ contexts/static caches, keys, ciphertexts,
plaintexts, precompute tables, serialization buffers, scratch, exceptions and all
Rust/Python aliases. Retain original source/work payer and guarded owners through
foreign calls, moves, result construction and the **last alias/destructor**.
Exceptions must not cross Rust unsafely; failed/cancelled calls retain funded
cleanup custody. Unsupported cleanup leaves the request unavailable, never
refunds or retries uncertain work. Specify ABI, thread use, Apple-Silicon artifact,
transitive pins/licenses and erasure limits before implementation approval.

## 6. Admission: unresolved floor, all-role peaks and finite budgets

**UNRESOLVED:** backend review derives public encryption coefficient-payload floor
`N*w*(2*Lpk+3*ell)`. At `N=65536`, `w=8`, `Lpk=ell=30`, this is **75 MiB**,
before P or other owners, exceeding **one 64 MiB `Budget` counter**. Nonfit is
conditional on charging that full-basis route to one original counter; actual HE
root binding remains unselected. D68 permits **no counter splitting, cap waiver,
profile-driven fit claim or fabricated smaller-key route**. Generated parameters
may change geometry, but cannot by themselves resolve this funding conflict.

Keep distinct: per-counter allocation ceiling; simultaneous all-role logical
owners; foreign physical capacities/allocator retention; process-tree peak/RSS;
host reserve; source/import working/historical-peak ceilings; cumulative work;
transport/disk/time bounds. A current `used` sample does not establish historical
source peak. Old estimates retain their original scope and are not recomputed
without a reproducible static derivation from actual selected source geometry.

For each actual phase/factory, **before allocation/deserialization/key issuance**,
admit retained source/config/tokenizer/tensors, client and server keys/contexts,
live ciphertext aliases, encoded weights, construction/transition/result overlap,
scratch, foreign caches, frames and controller/error/cleanup storage together.
No refund while backing or aliases remain live; no phase-local resets of cumulative
work. Wire compressed lengths cannot authorize expanded Cereal allocations.
Authenticate request/epoch/role/sequence/schema/profile and bound expanded counts,
dimensions and bytes before any foreign parser/factory. Reject stale, duplicate,
wrong-key, truncated and oversized frames; cancellation/timeout/disconnect/partial
reply must burn uncertain invocation and preserve source/lease custody to retirement.

The future test/implementation budget must be finite and **fixed to the approved
target**, not an invented tariff or arbitrary memory allowance. Let frozen graph
have `L` layers, sequence `T`, width `d`, `H` heads, FF width `f`, head output `C`,
and an explicit finite request set `R`. Count all nodes/edges/physical frames from
that graph; enumerate matrices including `d×d`, `d×f`, `f×d`, and task head.
For any declared tensor shape `(a,b)`, logical full-frame geometry is
`M=a/c`, `s=c*b*H` only where divisibility/embedding premises hold; use admitted
padding proofs otherwise, not rounded-down frames. For the existing D25 clear
non-complexified schedule, counts `M*b`, `3*M*(b-1)`, `M*(b-1)` and
`M*[b>1]` respectively mean tensor products, plaintext products, input and delayed
output rotations. They are **not** whole-BERT or paired-HE costs.

Derive full HE work as sums of actual graph operation/key/refresh/serialization
counts, and storage as the maximum sum of distinct live full backings plus
scratch and construction overlap at each phase, separately covering every role.
Deduplicate only proved aliases, not equally named buffers. Bind finite positive
and negative fixture counts to each distinct operation/layout/transition and
role/failure boundary in this fixed graph. Exact request count, selected graph,
cryptographic capacities, full work totals, test time/disk/process limits and
resource upper bounds are **unknown and stop gates**, not zero. Existing tighter
caps/HOLDs remain; parent grants must name exact bounded commands after review.

## 7. Prompt → artifact → acceptance checklist

All entries below are **future gates**, not executed checks or a new receipt
system. Use normal tests and independently transcribed source oracles. One small
reusable `.py` Experiment factory plus ordinary benchmark report is sufficient
evidence structure; no new inventory framework, archive or replay verifier.

| Required prompt/decision | Artifact in ordinary project workflow | Acceptance |
| --- | --- | --- |
| Fix original task/checkpoint/frontend/request and authority | Shared source/semantic configuration and immutable expected pins | Exact tokenizer/config/tensor/head binding; unknown or mismatched source refuses before import; base/fine-tuned embedding question resolved. |
| Approve concrete generated profile and explicit source departures | Backend-owned parameter/key contract bound to THOR numerical graph | Independent security, paired-refresh/restored-capacity and whole-graph gain/error review; exact-basis alternative never conflated. |
| Resolve numeric contradictions and derive tolerances | Reviewed equations/constants/domain certificates and independent eager FP32 adapter | Dense/interval/source-oracle normal tests, all layers/head and final quality pass frozen justified criteria; invalid domains refuse, no clear fallback. |
| Resolve funding/root binding and source/foreign lifetimes | Shared admission/custody extensions, no parallel method inventory | Exact-fit/one-short and preallocation-refusal tests, all-role peaks, cumulative work, exception/cancellation/last-alias retirement; unresolved 75/64 conflict blocks grant. |
| Approve bounded implementation/build/test stages after independent review | Exact parent grants; ordinary tests and, if later approved, standard uv/Maturin builds | Existing caps and source/runtime HOLDs respected; negative role/key/level/scale/frame and foreign-failure cases retain custody. No build or test granted by this document. |
| Qualify complete original-BERT SDK request | Normal component round trips, Pipeline/Experiment/SDK/gateway/CLI tests; reusable paper `.py` and ordinary benchmark/quality report | Client frontend → every encoder layer → pooler/head → final client result; no intermediate private disclosure; full setup/key/import/traffic/cleanup costs and all-role observed peaks reported. |
| Compare adapted common model, if separately ready | Separately labelled D3/D60 configuration/report | Its own decoder/state/numeric/quality admission; never substitutes for original BERT or inherits its paper qualification. |

**Before separately authorized bounded backend-qualification execution:** approve
a concrete qualification profile and security/disclosure contract; establish
complete admission/custody bounds for that workload, including original funding,
source authority, all-role source/foreign/transport/cleanup owners and applicable
ABI/dependency/build contracts; fix finite commands/test budgets and obtain
independent review. The 75/64 conflict remains a blocker for any affected route;
no counter splitting, cap waiver or privacy relaxation is permitted. Measured
paired-refresh/restored-capacity evidence is an intended output of this separately
gated qualification, not a prerequisite for its execution grant.

**Before whole-BERT execution, additionally require:** measured paired refresh
and restored usable capacity for the selected profile; authoritative complete
source/task/frontend binding; adopted concrete generated parameters and complete
key/security/disclosure graph; physical packing and amplitude certificates;
reconciled complete numeric profile, domain coverage and derived acceptance;
selected original funding root resolving the 75/64 conflict; complete whole-model
all-role source/foreign/transport/cleanup upper bounds; reviewed ordinary SDK
encoder/readout implementation contracts; finite whole-model command/test budgets
and independent review. Bounded qualification alone supplies none of these
whole-model conclusions.

**Both execution stages remain explicitly UNAPPROVED.** This staging correction
grants neither execution nor backend implementation. Existing source, cap and
privacy conditions remain unchanged. Conflicting API/profile evidence must be
reported, never repaired by invented parameters or behavior.

After this bounded two-document amendment, freeze both files for independent
Astra review. No runtime readiness, execution, fit, security, original-paper
reproduction or complete-method result is claimed.

## 8. Static feasibility disposition

**NO ADMISSIBLE ROUTE ESTABLISHED** is the bounded static-study outcome and a
valid hard stop, not a proof of universal impossibility. The
[backend study §1.2](rust-he-backend-qualification.md#12-bounded-single-counter-feasibility-study)
uses one unchanged **64 MiB original counter**. It grants no execution, numeric,
security, capability, split, rerooting, new issuer or backend adoption; task and
checkpoint remain unselected. Existing WIP, historical results and the preceding
contract remain intact.

Authority is only OpenFHE **v1.6.1**, commit
`b59e1f291965f14fbee5172cd83a20f5b856204f`, at the backend report's existing
O8–O12 locators: O12 `rns-pke.cpp`, both `PKERNS::EncryptZeroCore` overloads;
O9 `ckksrns-fhe.cpp`, `EvalBootstrapSetup`, `EvalBootstrapKeyGen`,
`EvalBootstrapPrecompute`, `EvalBootstrap`; O8 context/API, O10 Q generation,
O11 P/CRT generation. These are source locators, not built, security or admission
qualification.

The study is limited to three candidates, with `B=N*w=0.5 MiB` for
`N=65536`, `w=8` bytes:

| Candidate | Static floor / stop |
| --- | --- |
| Full-basis public-key encryption | Retained two Q30 key polynomials plus Q30 `v/e0/e1`: `B*(2*Lpk+3*ell)=75 MiB`. Reject under the same cap before plaintext/P/context/other keys/scratch/output/expression overlap. |
| Reduced entry basis, retained Q30 public key | O12 clones `p0/p1` on basis difference alongside `v/e0/e1`: `B*(2*Lpk+5*ell)=30+2.5*ell MiB`; `ell>=14` already overflows. Source-supported lower entry does not shrink retained key. Lower entries lack complete BERT entry/refresh schedule and upper bound; numeric/level departure unapproved. |
| Client secret-key overload | Retained secret `Ls`, sampled `a/e`, and basis-difference `scopy` yield conservative `B*[Ls+2*ell+I(ell!=Ls)*ell]`, before outputs/expression overlap. Not admitted: separate key/disclosure/encryption-contract review required; public/evaluation keys cannot be presumed absent. |

No candidate establishes whole-BERT fit. Required owner/phase bounds remain:

| Phase / owner | Whole-request obligation |
| --- | --- |
| Context generation/import/retention | Q/P, roots and CRT caches/tables, physical capacities and construction/import overlap (O8/O10/O11). |
| Complete key graph | All rotations, conjugation `M-1`, relinearization/bootstrap keys and optional encapsulation `M-4`/`M-2`; client/evaluator retained material and generation/import overlap (O8/O9/O12). |
| Bootstrap setup/precompute | Retained O9 `m_bootPrecomMap` plus construction temporaries and table overlap, admitted before construction. |
| All arithmetic, transport and retirement | Distinct live backing, plaintext/ciphertext aliases, scratch, construction/expression/output and serialization/deserialization overlap, exceptions, cancellation and last-alias cleanup under original source/work custody. |

O9's **specific full-slot `levelBudget=(1,1)` branch** constructs two simultaneous
`s*s` complex matrices. At `s=32768`, **assuming ABI coefficient width 16 bytes
per complex coefficient**, coefficient payload alone is `2*s*s*16=32 GiB`.
This condition rejects that branch under the same cap; it is neither a measured
ABI claim nor a bound for all bootstrap branches. Other branches have **UNKNOWN
complete upper bounds**, not automatic admission.

For any hypothetical survivor, fix entry/output bases, scale and noise degree;
prove ordered paired-channel behavior through both complex approximation paths;
exclude clone-return and establish restored usable levels. Compose through all
**twelve BERT layers, every residual join, pooler Tanh and task head**. Toy
refresh evidence cannot supply this schedule. Peak is maximum distinct live
backing plus scratch/construction/serialization/deserialization/exception/
cancellation/last-alias overlap, not a sum of nominal ciphertext labels or a
current `used` sample. Existing `Budget`, source/attempt, DAG and transport seams
do not issue HE authority or establish complete admission/custody coverage.

**Finite stop:** only O8–O12 and directly necessary pinned callees for these
three candidates; stop at unavoidable overflow or missing whole-BERT/admission
proof. No parameter sweep, invented smaller profile, cap change or counter
split follows. Publish the non-admission outcome without universalizing it.

**Staging unchanged:** before stage A, approve its concrete qualification
profile, security/disclosure, full admission/custody and finite commands/budgets,
with independent review. Measured paired refresh/restored usable capacity is
an intended **output of A**, not an A execution prerequisite; it is a prerequisite
for whole-BERT stage B together with every additional §7 gate. **Both stages
remain UNAPPROVED.** This is documentation/static arithmetic only, not native,
build, test, installation, model/import, AST, probe or formatting evidence.
Freeze these two documents for independent Astra review; no DECISIONS or PUMA
change is part of this amendment.
