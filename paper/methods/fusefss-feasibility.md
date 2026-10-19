# FuseFSS reuse-first architecture and feasibility audit

Status: **source/architecture audit, not implementation or model admission**.
Reviewed 20 October 2026, before integration of the concurrent native compiler,
backend and conformance workers. Those new modules are not qualified here.
Scope remains the [completion plan](fusefss-completion-plan.md), D5/D8/D9 and
`fusefss.source-derived-ring50-q12.v1`: public model, private activations, client,
two non-colluding online share holders, separate trusted preprocessing dealer,
CPU execution, fixed-count private full-prefix feedback and final client-only IDs.
No checkpoint loading, native build, wheel installation, benchmark, commit or
PUMA edit was performed. Existing production/reference files were read only.

## 1. Verdict and integration decision

1. **PLLM already contains much of the required protected algebra.** Reuse its
   Boolean DPF comparisons, affine vector-selection algebra, one-use products,
   shared Boolean-circuit builder, context binding, codecs and retirement tests.
   Existing hard-coded numerical/role profiles are not FuseFSS implementations.
2. **A generic per-threshold arithmetic vector-DCF lookup is permitted by the
   paper's Protocol-3 backend contract.** It is not a reproduction of the paper's
   optimized backend or costs. See the page-pinned construction below. A new
   independent cryptosystem is unnecessary merely to satisfy the interface.
3. **A separate `fusefss_backend/` may be an adapter/composition layer, not a second
   incompatible primitive stack.** Prefer shared ring/group-aware DCF algebra,
   shared product ownership and shared framing/admission. A bounded new module
   can isolate the first scalar milestone, but duplicated PRGs, Beaver code,
   replay rules, codecs and cost definitions should not become permanent APIs.
   Any temporary duplication needs explicit consolidation ownership.
4. **Generic threshold keys are a reasonable bounded conformance control, not
   the default full-model plan.** Calculated activation-only material is terabytes
   even for one GPT-2-size selection. Existing universal-public-boundary lookup
   offers a much smaller-material extension, though still expensive local work.
5. **Frozen scalar semantics are settled. Protected composition and complete
   graph are missing.** Do not spend another tranche choosing GELU ties, fitting
   interpolation, changing coefficient scales, or implementing a second clear
   scalar compiler. The missing evaluated descriptor manifest still prevents
   claims of original evaluated-build numerical identity.

## 2. First-party source ruling: what a backend may implement

Primary evidence: pinned `papers/r14-fusefss.pdf`, arXiv:2606.09551v1, SHA-256
`6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
The [source lock](../reproduction-contracts/fusefss/source-lock.json) also pins
SIGMA and inert CPU corroboration. Author code was not imported or executed.

| PDF locator | Exact consequence |
| --- | --- |
| p.4, §3.6 | Two **primitive families**: public-view comparisons with XOR outputs; vector interval lookup with additive ring outputs. Public shapes may leak, private thresholds and selected interval may not. |
| p.17, Protocol 3, lines 1–22, 29–34 | Dealer translates/pads to `M=min(m+1,N)` and builds one lookup instance containing all vector payloads. A boundary-hit mask still receives padding; clear `(B,ord)` remains secret. |
| p.20, Appendix I.1, Definition I.1 and following paragraph | Explicitly permits one DCF/DPF comparison key per query, concatenated/batched. “Packed” does not require a new correlated-key cryptosystem or GPU. Outputs must be XOR shares. |
| p.20, Appendix I.2, Definition I.2 | Defines `(Gen,Eval0,Eval1)` by functionality: local evaluation on public `xhat`, additive shares of the selected vector. Does not mandate a particular interval-key representation. |
| pp.20–21, fixed-shape requirement/Proposition I.3 | `T,M,p`, widths and observable key lengths must be descriptor-determined, not sampled-mask-dependent. Zero payload/threshold shortcuts cannot shrink secret-dependent shape. |
| pp.26–27, Appendix N | Concrete `(a,b)` lookup, masked opening, original-x recovery and one Beaver product; sign output is separately XOR-shared. |
| p.27, Appendix O | Backend-agnostic accounting explicitly charges actual primitive preprocessing, Horner/Phi products, Boolean ANDs and conversions. “Two calls” is not two keys or two end-to-end rounds. |
| pp.24–25, Appendix M; pp.6–7, Theorem 4.7 | Security conditional on secure FSS and post-processing subprotocols; functional parity alone is insufficient. |

### Valid generic construction, with an important privacy constraint

For Protocol-3 dealer-only partition `B[0..M]` and payloads `v[0..M-1]`:

```text
LUT(u) = v[M-1] + sum(j=1..M-1, (v[j-1]-v[j]) * [u < B[j]])
```

Share the base vector randomly. Issue a **payload-hiding arithmetic DCF** for
each interior threshold, with payload `v[j-1]-v[j]` in the product group
`(Z/2^50)^p`; alternatively issue independent scalar payload DCFs per coordinate.
Each evaluator sums its own shares in a single fixed-shape local routine. Keep
all `M-1` positions, including duplicated-piece zero deltas. Never disclose
translated boundaries, payload order, clear deltas or interval IDs.

Issuing a unit-output DCF and multiplying by a **clear mask-permuted delta** is
not this construction: exposing that delta/order can reveal the mask. General
arithmetic coefficients also cannot be multiplied by XOR shares without an
explicit conversion. Product-group DCF needs an actual group-aware Convert/PRG
mapping and privacy rationale, not identical correlated scalar outputs. Here
`p=2,n=50` fits a 100-bit product group into a uniform 128-bit value seed;
general payload dimensions require domain-separated expansion as necessary.

Thus **per-threshold vector arithmetic DCF: permitted interface instantiation**,
subject to ordinary FSS security and fixed-shape obligations. **Same as original
optimized FuseFSS implementation/performance: not established.** Appendix I.1
explicitly blesses per-query packing; I.2 is a functional abstraction, not an
explicit claim that the above interval construction has the paper's costs.

The existing universal public-boundary algebra is another candidate for the
public descriptor case. It hides a common rotation through one DCF for `r-1`
and secret constant corrections. It does not directly implement arbitrary
private-boundary/private-payload `Gen(B,v)`: retain that distinction. A
descriptor-specialized implementation needs a proof/mapping that its result
equals the Protocol-3 translated/padded lookup and leaks only its public shape;
do not silently bypass compiler identity or advertise general I.2 support.

## 3. Reuse matrix: exact APIs and boundaries

Paths below are relative to repository root. “Direct” means reusable at its
stated component scope, not already production-qualified FuseFSS execution.

| Existing code/API | Reuse verdict | Exact extension or mismatch |
| --- | --- | --- |
| `crates/pllm-garble/src/compact_dpf.rs`: `Key::issue`, `issue_with`, `less`, `encode`, `decode` | **Direct protected Boolean comparison algebra** for widths 1–63; ring50 already fits. `less(p,u)` returns XOR share of `[u<point]`. | Dealer wrapper validates before assertion-based issuer; compute low/shifted public views locally. Keep each query key/threshold private, fixed query slots and one-use ownership. Source uses 128 seed bits plus separate controls, not SIGMA's 127-bit encoding. `all()` is restricted to at most 10 bits: never expand ring50. |
| `compact_dcf.rs`: `Comparison::{issue,eval,bytes}`; internal `issue_payload` | **Narrow shared group/width extension**, not from-scratch DCF. AES seeds, per-level correction algebra and erasure already exist. | Current input/threshold/payload/corrections are `u32`, width 1–32, output group `Z/2^32`; payload issue is private. Widen to checked `u64`/ring mask and product-group outputs without changing old identities. Current byte formula `20+22*n` cannot be reused for ring50/u64/vector corrections. No current export codec. |
| `interval_fss.rs`: `Comparison::issue(bits,r)`, `eval(p,opened,boundary)`, `correction` | **Narrow extension** of independently implemented universal public-boundary algebra. One key per width serves all public boundaries, not one key per boundary. | Width 1–32, `u32` group. `eval` alone is not a comparison bit: its dealer-only correction matters. Widen masks, subtraction, carry expressions and all constants. Boundary public here means original descriptor boundary, never translated private `B[j]`. Keys remain private to one lane/issuance. |
| `shared_affine_lookup_reference.rs`: `issue_reference(Arc<Profile>,Layout,Context,lanes)`, `Party::{start,finish,cancel}`, `resources` | **Closest existing vector coefficient selection.** Vector layout opens one mask and returns two coefficient shares; Separate is useful matched control. | Hard-wired `piecewise_gated_reference::Profile`, two `u32` coefficients, ring32, maximum 256 lanes; evaluates original public spline boundary differences with universal DCF. Generalize descriptor/table view and ring width, not interpolate a new profile. `finish` copies `constant.to_vec()` and allocates result/scratch. Key byte count excludes owners, allocator/metadata and public descriptor residency. |
| `shared_arithmetic_reference.rs`: `issue_reference(bits,lanes,Context)`, `Party::{start,finish,cancel}` | **Direct lifecycle/batched-product pattern**, width extension needed. | Bits 2–32, at most 1,024 lanes, triple `u32[3]`; cannot execute A50 by relabelling. Frames bind context/nonce but are not authentication or malicious checks. |
| `shared_wide_product.rs`: `issue(Context)`, `Party::{start,finish,cancel}` | **Direct ring64 Beaver algebra usable for ring50 Horner via reduction**. | Operates one scalar at a time, 24 triple bytes/party, 94-byte frame/direction. Lift local ring50 shares to u64, run ring64 product, locally reduce each result modulo `2^50`; quotient homomorphism preserves desired product even when lifts sum with an extra `2^50`. Reduce before any signed interpretation/rescale. Add a bounded batched/shared-ring adapter and codecs rather than duplicate Beaver equations. |
| `shared_circuit.rs`: `issue`, internal `issue_widths`, `Party::start/advance`, `Resources` | **Direct protected circuit/conversion substrate**, not ready-made FuseFSS Phi. Reconstructs only circuit wires, masks outputs, uses fresh dealer OT correlations. | Rings 2–64, 1–8 input words/outputs, at most 65,536 wires. `rounded()` is ties-even and is **wrong** for D8 internal floor. Build typed source-floor/TR/sign-extension circuits explicitly. Existing output interface is arithmetic sharing; XOR-output/AND/B2A/A2B interfaces need explicit types and costs. Dealer OT is not distributed OT; no transport authentication. |
| `shared_rescale_reference.rs`: `Descriptor::new`, `with_backend`, `resources`, `issue_reference`; `Rounding::Floor` | **Reuse exact floor/carry algebra and tests**, extend general width/types. | Bits 2–32, 1,024 lanes, 64 MiB combined estimate. Floor available; TiesEven is separate adaptation. Fused/unfused and PrefixDpf/CompactDcf/IntervalDcf remain explicit. Return type does not by itself model source narrowed TR ring versus same-ring ARS; introduce typed edge contracts. |
| `masked_wire_reference.rs`, `transport.rs`, `coins.rs`, `seeded_codec.rs` | **Reuse bounded GapARS/TR, DPF/sign-selection algebra, canonical party-local codec and burn patterns.** | Complete block is a different hard-coded ring40 residual-SiLU Q24→Q12 profile (effective width19, bounds48/128), 256 lanes, 64 MiB, 11/8 phases. Not D8 ring50 CPU GELU. Local dealer/client ownership and seeded-field compression are not separate-dealer support or source byte identity. Codec exports consume party but construct bit-buffer plus packet copies before old owner drops. Account overlap. |
| `shared_piecewise_gated_reference.rs`: `issue_reference`, `resources` | **Reuse composition/state-machine pattern and matched Separate/Vector experiment.** | Q16 input, Q8/Q9 working profile, slope Q14/intercept Q(f+14), ties-even shifts, six rounds, two products/four rescalings. None of those numeric constants transfers to GELU Q0 slope/Q12 intercept. |
| `mixed_gated.rs`: `Issuer`, `Step`, `Op::{Circuit,Product,Lookup}`, `Party` | **Reuse model-neutral composition ideas, child-context derivation and explicit cost parts/phases.** | Only Compact/Curl/LogRow profiles, 16 fixed registers, queued eagerly issued component owners, 64 MiB bound; not arbitrary typed Phi/graph interpreter. Conversion circuits use profiles' rounding. Count queue, scratch and unconsumed steps, not only current phase. |
| `shared_lut_reference.rs`: `Table::new`, `issue_reference`, `Party/Pending` | **Small-index control/oracle and consuming-owner pattern**, not general interval backend. | Public u32 table, power-of-two rows 2–4,096, u16 index, dense one-hot share vector and full scan. A ring50 full-domain table is impossible; source norm LUT has 8,192 entries and exceeds this ceiling. No interpolation may replace source lookup. |
| `point_fss.rs`: prefix-DPF `Comparison` | **Reference/control only**. Fixed zero-payload query shape useful negative control. | Arithmetic u32 output; quadratic-in-width construction, not optimized SIGMA DPF or ring50 implementation. |
| `python/pllm/runtime/scalar_fss_reference.py`: `AffineScalarSpec`, `ScalarGateIssuer`, `ScalarGateShare` | **Existing independent toy contract/oracle**, not timed production path. | Dense ring2–8, at most eight intervals/eight predicates, uint64 coefficient table, no native/source D8 descriptor. Never scale dense table to ring50 or call this installed FuseFSS. |

### Meaningful tests already present

Security boundary: these are independently implemented, unreviewed research
primitives. Hash-bound context/nonce headers are **not MACs**, and authenticating
transport does not establish malicious-output correctness. No active-adversary,
collusion-resistant, forward-secure or side-channel/constant-time claim follows.
Evaluate branches and indices may depend on public masked inputs; private dealer
generation and key-dependent local operations still need review. `Zeroizing` and
consuming handles are useful, but do not erase all stack copies, exported frames,
Python byte copies or allocator remnants automatically. Retire every surviving
owner on abort; never reuse masks/triples across lanes, requests or retries.

Inspected native tests, **not rerun** in this audit:

* `compact_dcf::tests::exhaustive_thresholds_payloads_and_fixed_key_shape` and
  `full_width_boundaries_and_fresh_roots`: preserve as ring32 regressions; extend
  with ring50 thresholds, vector payloads, endpoint carries and independent roots.
* `compact_dpf::tests::{exhaustive_small_points_and_parities_and_tree_boundaries,
  wide_comparisons_do_not_expand_the_domain,
  canonical_material_roundtrips_and_rejects_truncation_and_padding}`.
* `interval_fss::tests::{exhaustive_public_shifts_with_every_mask_including_zero,
  figure_14_all_small_intervals_and_full_width_endpoints}`: especially mask zero,
  full interval and all-zero/all-one endpoints; test ring50 arithmetic reduction.
* `shared_affine_lookup_reference::tests::{coefficient_selection_at_every_boundary_and_ring_endpoint,
  malformed_cross_profile_layout_and_cancel_burn}`: extend beyond current Q8/Q9
  profiles and test serialized fixed shape, not only `payload_bytes()` formulas.
* `shared_arithmetic_reference::tests::{exact_ring_products_and_one_use,
  malformed_domain_context_replay_and_cancel_burn}` and
  `shared_wide_product::tests::full_ring_signed_products_one_use_and_foreign_frames`.
* `shared_circuit::tests::{signed_rounding_random_share_wraps_and_ot_choices_are_exact,
  distinct_input_rings_preserve_carries_and_burn_invalid_shares,
  foreign_context_malformed_ot_and_cancellation_burn}`. Existing rounding success
  is not evidence for floor; add distinguishing negative half/low-bit cases.
* `tests/test_scalar_fss_reference.py`, `test_shared_fss_truncation.py`,
  `test_masked_wire_reference.py`, `test_masked_wire_fusion.py`,
  `test_masked_wire_runtime.py`, `test_masked_wire_sdk.py`: preserve their original
  scope. SDK tests include early unbounded-work rejection, fresh sessions,
  binding mismatches, complete-block accounting and seeded asymmetric payloads.

Fresh evidence: ran
`python3 -B -m unittest discover -s paper/reproduction-contracts/fusefss -p 'test_*.py' -v`:
**20 tests passed**, including local PDF source hashes; no skip. This verifies
existing scalar reference only, not newly concurrent native work.

## 4. Settled numerical target versus missing protected composition

The frozen [scalar reference](../reproduction-contracts/fusefss/README.md) already
settles canonical GELU descriptor/table identities, 511 original/512 padded
intervals, typed coefficients `(constant,linear)=(-T[J],D)`, total ring partition,
mask rewriting/padding, original-x Horner and identity Phi. It also settles
nearest-even **external** rational encoding, internal signed floor **before**
absolute value, Q12 constant/Q0 gain and **no extra ARS12 after Horner**.
`X=-1` yields -31. Full-ring mathematical extension is not source signed38
provenance: `X=2^37+64` distinguishes it from an incorrectly narrowed selector.

The separate [source resolution §§2,5](../reproduction-contracts/shared-profile-resolution.md)
already gives SiLU/nExp/inverse/norm scalar equations and table hashes/generation
rules. These are usable first-party-derived mathematical targets, not remaining
permission to invent constants. They are not all frozen typed native
descriptors/Phi programs. Remaining work is protection, composition, source-width
proofs and graph/frontend binding:

| Gate/family | Target already known | Missing protected obligations |
| --- | --- | --- |
| CPU GELU | Early TR/floor, clipped 256 table, Q0/Q12 affine descriptor | Correct A50 lookup, one fresh product, party-local outputs, fixed shapes, consuming owners. |
| CPU SiLU | Same early-floor pattern, 1,024-entry floor residual table | Frozen descriptor/circuit and capacity policy. Direct `(D,J)` partition has 2,047 original/2,048 padded intervals, **above current scalar compiler's 512/513 bounds**; either explicitly extend admission or compose smaller specified gates, never fit a 64-piece spline instead. |
| nExp/softmax | Clip magnitude to65,535; two 256-entry E0/E1 tables, product then floor shift; causal max/sum/inverse/output products | Reductions, comparisons, secret products, exact TR/ARS, masks/zeroed causal cells and complete Phi. Quantized `[0,-4096]` outputs `[3013,1107]`; renormalizing to sum4096 changes target. A monolithic ring50 interval map is not a free replacement for the two-table circuit. |
| Reciprocal | `Z=floor(sum(E)/64)`; `floor(2^18/Z)`; `q=6+ceil(log2(k+1))` index bits | Width depends on public active row length k; at k128 q14 means16,384 domain slots, already beyond dense LUT4096. Freeze unreachable-zero sentinel and range provenance, reject zero denominator outside contract. |
| LayerNorm/rsqrt | Width-specific8,192-entry integer table; approximate mean coefficient `floor(4096/k)`; untruncated centered squares and `+1` guard | Freeze k-specific identity, mean/squares/sum bounds, exponent/mantissa extraction, lookup and learned affine order. At width768 mean coefficient5, not exact division; constant-one row normalizes to ones, unlike width4. Powers-of-two-only helpers cannot silently alter k. |
| ARS/TR/conversions | Floor signed shifts; reduced-ring TR differs from same-ring sign-extended ARS; GapARS needs quarter-ring range | Prove every pre-wrap accumulator/product/gap; general-width typed circuits and actual one-use material. Local sharewise shifts, truncating before sum, or rescaling after wrong wrap are not source operations. |
| Common Qwen/RMSNorm/RoPE | D8 source profile and public ownership fixed, source Llama2 evidence available separately | Exact checkpoint/frontend, Qwen-specific graph/position/epsilon/coefficient choices and ranges remain explicit; do not import PUMA numeric choices or substitute float host operators. |

External `Fraction` encoder does not select binary32/binary64/bfloat16 checkpoint
bridge. Specify finite decoding, representability, ties-even and weight/bias/position
encoding with source/tokenizer lock. Existing graph `GeluTanh` is not automatically
the approved CPU GELU merely because both are named GELU.

## 5. Model-neutral compiler/SDK map: reuse public path, not public plaintext

D5 explicitly requires extending shared contracts, not a FuseFSS-specific runner.
Existing `FuseFssPredicateFusion` in `python/pllm/passes/__init__.py` and
`FuseFssNonlinearOperator` inventory entry are **pending placeholders**, not
capability evidence. An operator capability must stay distinct from full-response
support.

| Existing seam | Reuse and required change |
| --- | --- |
| `python/pllm/components/__init__.py`, `ComponentDescriptor`, `create_component`; `configuration.Pipeline/Experiment` | Immutable typed gate/compiler/protocol/topology choices, canonical round trips. Keep masks, keys, seeds, pending owners out of configuration. Register actual supported bounds; retain pending coverage elsewhere. |
| `python/pllm/modeling`, `model_loader`, `compiler.compile`/`plan_on_load` | Reuse model-neutral config lowering, source/tokenizer locks and native canonical request. Admit metadata before graph expansion/tensor import. Add source ring50/Q12 frontend and complete operation/range manifest, not model-name branches. |
| `crates/pllm-compiler/src/lib.rs`: typed regions and `DecoderCoverageReport`; `decoder_runtime_schedule.rs::lower_decoder_runtime_schedule` | Reuse operation IDs/shapes/dependency validation/digests and independent runtime checks. Current executor enum exposes ClientLocal/ClientLinear/RemoteStage/VerifiedRemoteStage/RemoteGatedBlock, not generic protected shared tensor execution. Add a typed shared-domain execution contract, including Boolean wires, Phi and reductions. Reject unsupported nodes rather than leave them client-local. |
| `masked_wire_schedule.rs::{valid,MaskedWireNumericContract,bind_regions}` | Reuse strict attribute/shape checking, overlap rejection and region binding. Current contract explicitly uses client float32→Q24 floor, client output decode, trusted-client dealer, SiLU×multiply only and max4,000,000 elements. Those are **numeric/trust mismatches**, not new backend settings. |
| `decoder_boundary.rs::PrivateBoundaryContract` | Reuse ownership/embedding/head admission concepts; current prepared-body/u32/quantized protocol IDs are not D9 ring50 equality→one-hot embedding/full head. Bind final norm and stable lowest-index protected argmax, with private append and no intermediate reveal. |
| `python/pllm/profiles/__init__.py::resolve_runtime_composition`, `roles.topology`, `runtime.servers`, `runtime.party` | Reuse composition resolution, supervision/authentication and provider admission. Declare client + two online holders + separate dealer, actual permitted views and all role owners. Existing worker names alone do not confer topology. |
| `deployment/admission.py::{ExecutionContract,RoleAdmission}`, native `network_placement`, `runtime/benchmark_memory.py` | Reuse fail-closed role admission. Add all dealer/key/frame/descriptor/scratch owners and full-prefix workload counts. Current contracts validate existing prefill/decode geometry; do not assume their incremental KV schedule implements D9. Unknown mandatory owner/work costs must reject. |
| `runtime/masked_wire.py::{WorkerRegions,ClientRegions}`, `crates/pllm-python/src/masked_wire.rs` | Reuse authenticated opaque relay, session/operation/ordinal binding, monotonic burn ledger, bounded native handles, cancellation/drain patterns. Existing client knows values and issues both packets; a separate dealer requires a real adapter and role admission. Existing frames cap1 MiB; generic DCF batches easily exceed it. |
| `runtime/model_binding.py`, `RuntimeClient`, gateway/sidecar | Reuse source validation, request cancellation/failure lifecycle and final-only response handling. Current `SemanticLocalOperators`/client float body are oracle/integration machinery, **not protected whole-model execution**. Keep model-derived tokens, logits, activations and selected interval inside permitted share owners. |
| `metrics/operator_benchmarks.py`, `metrics/masked_wire.py`, `metrics/shared_rescale.py`, ordinary `BenchmarkTarget`/CLI/benchmark suite | Reuse reports, replay identities, public fixture oracles and phase/resource accounting. Add actual native backend resource receipt, topology/source identity and capability gates. Existing masked-wire/rescale speedups do not transfer. Complete-response benchmark waits for accepted final IDs, not buffered streaming throughput. |

Do not reimplement source loading or benchmark plumbing alongside the ordinary
SDK. Do not reuse delegated-verification or replicated-three-party arithmetic as
FuseFSS numerical/trust semantics merely because those paths serve models.

## 6. Bounded full-workload feasibility, not blind implementation

Replay static calculations:

```sh
python3 -B paper/methods/fusefss-feasibility-costs.py
```

The accompanying [JSON](fusefss-feasibility-costs.json) is calculated design
evidence, not measured performance, implementation key sizes, security review or
full-model totals. Assumptions: ring50 held in u64, p2, current128-bit-seed/two
control-byte DCF style, padded lookup512, uncompressed keys, one ring64 triple,
8-byte input mask share and16-byte constant shares per party/scalar.

| Candidate design | Per-party payload/scalar including those48 bytes | Local DCF path levels/scalar/party |
| --- | ---: | ---: |
| Independent scalar payload DCF per threshold and coordinate | 1,353,176 B | 51,100 |
| Product-group vector DCF per threshold | 885,100 B | 25,550 |
| Universal public-descriptor boundary DCF extension | 1,372 B | 25,550 |

Derivation: scalar candidate `16+8+50*(16+8+2)=1324 B/key`; vector candidate
`16+16+50*(16+16+2)=1732 B/key`. Generic designs keep511 interior thresholds;
universal design evaluates510 original interior boundaries plus base `eval(0)`
with one key. These are **named design estimates**, not universal FSS lower
bounds. Ring50 bit packing, another backend or source-specific algebra changes
them and needs its own ledger. The universal variant radically reduces key bytes,
not traversal count. A scalar-style five-AES-block expansion would multiply path
levels by five encryptions; optimized tree-prefix work sharing has not been
implemented/measured and must not be assumed.

For fixed generation count T and initial prefix P:

```text
S = sum(j=0..T-1, P+j) = T*P + T*(T-1)/2
GELU scalars = layers * FFN_width * S
causal nExp values = layers * heads * sum(j, (P+j)*(P+j+1)/2)
equality→one-hot embedding equalities = vocabulary * S
embedding public-weight multiply-adds = vocabulary * hidden * S
last-row full-head public-weight multiply-adds = vocabulary * hidden * T
argmax pairwise comparisons = (vocabulary-1) * T
```

Public-weight linear arithmetic is local on shares, but rescaling/ranges are not
free. Attention QK/PV shared×shared products, selection, normalizations and
conversions need additional protected work. Full-prefix recomputation does not
permit copying an incremental KV cost. Public liveness may justify last-row-only
head; no narrowed vocabulary or unproved pruning.

**Illustrative original-target geometry only:** GPT-2-base12 layers, hidden768,
FFN3072, heads12, vocabulary50,257, P128. This is not an admitted checkpoint or
claim of exact evaluated descriptor identity.

| Activation-only scenario, both parties cumulative raw payload | T1:4,718,592 scalars | T8:38,780,928 scalars |
| --- | ---: | ---: |
| Scalar keys/threshold/coordinate | 12.770 TB | 104.955 TB |
| Vector keys/threshold | 8.353 TB | 68.650 TB |
| Universal public-boundary key | 12.948 GB | 106.415 GB |

Decimal TB/GB above; excludes every non-GELU gate, allocator, descriptor, codec,
transport, scratch and source owner. Even universal lookup needs about120.56
billion DCF path levels/party for T1; unknown CPU throughput is a hard feasibility
question, not permission to launch a blind model run. At T1 additionally count
1,188,864 causal nExp values,6,432,896 embedding equalities,4,940,464,128 local
embedding multiply-adds,38,597,376 head multiply-adds and50,256 argmax comparisons.
The JSON includes T8 equivalents and exact bytes.

With **hypothetical batched** u64 openings and product frames, peer traffic for
activation only is `48*scalars + 312*chunks` bytes across both directions, assuming
78-byte headers, one mask word and two Beaver-opening words per lane/direction,
two dependency rounds and256-lane chunks:232,243,200 B at T1. Existing scalar
`shared_wide_product` is not batched; reusing it lane-by-lane instead costs
`156+204*lanes` bytes per lookup batch. Relay legs, authentication, distribution
and network framing are extra. Key transfer is not online peer opening traffic.

### Admission ledger and stopping conditions

| Owner/work | Known now | Must be established before expansion |
| --- | --- | --- |
| Descriptor/compiler | Frozen GELU32,900 canonical bytes; clear instance11,846 bytes; bounded shape | Native object/table/expression overhead, scratch/sort/serialization copies and metadata owner; generalized SiLU/norm capacities. Clear compiler bytes are not FSS key bytes. |
| Dealer issuance | Two parties' keys/triples must coexist unless a proved streaming interface avoids overlap | Allocation preflight **before RNG**, compiler plan lifetime, root/correction clones, queues, packet staging and cancellation/partial-export retirement. |
| Evaluators | Own only role-local keys, shares, pending frames and outputs | Actual Rust layout/capacity/padding, key import/canonical rejection, scratch and output/frame reservations, zeroization and consumed-state semantics. Encoded22/26/34-byte corrections are not their `size_of`/allocator cost. |
| Live versus cumulative | Streaming can bound live window; never reduces total fresh material | With256 lanes, generic vector candidate alone is453,171,200 B for two parties, already over existing64 MiB estimate; scalar-coordinate candidate692,826,112 B. Universal candidate702,464 B before overhead. Choose bounds/chunks from actual owners, not inherited256. |
| Public source/tensors | Public model permits local share×public weight operations | Both online model residency, decoder tensor liveness, tokenizer/source import, weight conversion buffers, ranges and duplicated immutable descriptor caches. No checkpoint admitted here. |
| Remaining gates | Mathematical scalar targets available; vector reductions not compiled away | Full protected gate inventory, fresh triples/AND/conversions, norm/reciprocal table/domain costs, causal positions, private token append and final client reveal. |
| Time/network | Static work/byte counts only | Bounded scalar/chunk setup+issuance+online CPU and peak allocation, authenticated separate-role delivery, peer latency and abort accounting. No source speedup or old measurement can fill unknowns. |

**Must-fix before claiming F1:** compiler/backend descriptor binding; hidden
translated payloads; ring50 arithmetic; fixed shape under every mask; original-x
Horner without extra shift; one-use burn on all failures; actual bounded owners;
party-local views; independent native-output checks. Unsupported Boolean/Phi stays
explicit, not a clear fallback.

**Must-fix before F2/F3 expansion:** decide general versus public-descriptor lookup
adapter, qualify shared primitive reuse, complete exact gate manifests and typed
width/range proofs, resolve capacity mismatches, and price a complete generated
model **before** checkpoint load. If even the activation-only candidate exceeds
an approved cumulative-material/work budget, stop that candidate rather than
raise ceilings or mistake streaming for a solution.

**Must-fix before F4/F5 claims:** separate dealer role and authenticated transport,
full protected body/embedding/norm/head/reduction/feedback, D5 ordinary-path tests,
fixed-count final-only IDs, checkpoint/frontend and original/common-model locks,
independent whole-model integer/quality parity, complete fresh-process resources
and security review. No full-workload feasibility verdict follows from20 scalar
tests or a successful first lookup.
