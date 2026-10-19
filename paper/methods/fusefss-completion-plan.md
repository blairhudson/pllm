# FuseFSS implementation plan

Status: bounded native scalar, protected primitives and source-gate composition
implemented. Complete semantic execution and SDK/model qualification remain open.
PUMA is parked at the recorded
[local-v13 admission gate](../artifacts/puma-sdk/local-v13/README.md).

## Objective and governing contract

Implement FuseFSS independently in PLLM, through reusable native components and
the ordinary Pipeline, Experiment, SDK/gateway, CLI and benchmark paths. Follow
[D1, D3, D5, D8 and D9](DECISIONS.md), the
[shared-computation contract](shared-computation-contracts.md#4-fusefss), and the
[frozen scalar specification](../reproduction-contracts/fusefss/README.md).

The selected profile is `fusefss.source-derived-ring50-q12.v1`: nearest-even
external input/weight encoding, source-defined floor operations internally, two
online share holders and a separate trusted preprocessing dealer. Preserve public
model ownership, private activations and fixed-count full-prefix token feedback,
lowest-index argmax, and final client-only token-ID disclosure. CPU-first is the
approved execution profile. The missing evaluated descriptor manifest remains a
separate source-identity limitation.

## Ordered completion gates

| Gate | Deliverable | Evidence required before advancing |
| --- | --- | --- |
| F0: source and feasibility | Exact backend/source mapping, numeric frontend obligations, full gate inventory, owner and work budgets | Page-pinned sources, arithmetic counterexamples, complete known/unknown cost ledger; early rejection of unsupported capacities |
| F1: native scalar vertical slice | Typed Protocol-3 descriptor compiler, actual protected vector interval lookup, fresh shared Horner product and consuming scalar owners | Frozen descriptor/table parity; independent fused/unfused integer results; fixed public shape for every mask; role, replay, malformed-frame, cancellation and allocation controls |
| F2: complete protected gates | Source-defined SiLU, nExp, reciprocal/rsqrt, ARS/TR, comparisons, conversions and non-identity Phi | Exact descriptor/LUT bytes, independent exhaustive bounded oracles, explicit ranges and complete material/frame accounting |
| F3: complete semantic execution | Model-neutral source lowering, protected embedding, every body operation, full head, stable selection and private feedback | All generated-model tensors and state against an independent interpreter, multiple selections, zero unsupported fallbacks; complete metadata/live-owner/work admission |
| F4: ordinary SDK composition | Capability configuration, source and material roles, authenticated transport, lifecycle supervision, SDK/gateway/CLI/benchmark integration | Installed-wheel round trips, real separate-role requests, exact published invocations, cancellation/failure retirement and receipt-backed benchmark replay |
| F5: model qualification | Original-model track and separately identified pinned common-Qwen track | Source locks, full-checkpoint admission, complete integer parity and independent model quality; matched fresh-process CPU/memory/traffic/lifecycle measurements and confirmation |

Complete serving or method status requires F0–F5. Each earlier slice records its
actual scope. Full-model feasibility is examined during F0 and repriced after
F1/F2; a scalar success cannot supply a missing complete-model budget.

## First parallel implementation tranche

Reuse-first review precedes integration: extend existing exact primitives where
their numerical, secrecy and ownership contracts match. New module boundaries
below organize the work; they do not justify duplicating a compatible existing
backend or introducing separate SDK configuration, transport or benchmark stacks.

The coordinator owns shared module registration, integration, combined tests and
evidence. Sol 6.1 workers have non-overlapping ownership:

1. **Native compiler:** `crates/pllm-garble/src/fusefss/` only. Implement bounded
   typed descriptor validation and Protocol-3 mask compilation, preserving frozen
   canonical identities and Boolean/arithmetic distinctions. Clear mask-dependent
   plans are dealer-only; evaluators must never receive their thresholds or masks.
2. **Protected backend:** new `crates/pllm-garble/src/fusefss_backend/` only.
   Implement source-compatible bounded ring50 vector interval FSS and fresh
   shared degree-one Horner execution. Establish the backend mapping before
   implementation. Preserve party-local views, fixed shapes, fresh randomness,
   one-use failure semantics, erasure and allocation limits. Report any source
   uncertainty or required adaptation instead of silently substituting a protocol.
3. **Independent conformance:** new files under
   `paper/reproduction-contracts/fusefss/native-conformance/` only. Freeze
   independent numeric and compiler vectors, adversarial cases, and a replayable
   native-output checker. Retain original scalar evidence unchanged.
4. **Reuse/source/feasibility review:** `paper/methods/fusefss-feasibility.md` and its
   dedicated calculation script/artifact only. Audit backend legitimacy, remaining
   source/frontend decisions, complete-model work/material lower bounds and SDK
   extension points. Map existing primitives to direct reuse, bounded extensions
   and incompatible contracts before integrating new code. Clearly separate
   calculated costs from measurements.

Workers may read the entire repository but must not edit PUMA, overwrite another
worker's changes, run checkpoint loads, launch model benchmarks, rebuild/install
wheels, change resource ceilings, or create commits. Native validation uses one
Cargo build job and bounded serial tests; combined validation belongs to the
coordinator. Any shared-file change is proposed for coordinator integration.

### Reuse audit decision

The [reuse and feasibility audit](fusefss-feasibility.md) identifies existing
ring50 Boolean DPF comparisons, ring64 Beaver products, and public-boundary affine
lookup as the implementation base. A private payload vector-DCF implementation
is a valid bounded Appendix-I.2 control, but its projected activation material
fails the full-model feasibility gate. It is not the default model backend.

The first tranche consolidates common scalar/vector DCF and Beaver algebra and
extends the existing universal public-boundary comparison to ring50. The native
compiler preserves frozen identities; independent replay checks all 58,400 clear
compiler gates and 43,544 freshly issued protected GELU outputs. Both include the
38 exact-rational external-encoding cases. Private-vector control and original
public-boundary specialization have distinct bindings; only the latter is used
for the full protected replay. Its public duplicate refinement preserves the
whole-ring function without exposing sampled-mask-dependent table order.

Engineering reviews cover original-x multiplication, consuming lifecycle,
source binding and live reservations. They do not establish independent security
qualification. Complete results, source snapshots, costs and read-only replay:
[native scalar evidence](../artifacts/fusefss-scalar/local-v1/README.md).

### F2a: protected primitives and frozen source gates

The follow-up adds actual compiled protected Boolean execution, exact signed
floor/TR/ARS and extensions, B2A, consuming Phi selection and ordinary ring
products. Fixed 256-entry public chunks compose complete zero-extended lookup
tables up to 131,072 entries under the existing 513-interval primitive,
four-lane and 64 MiB limits. All chunks and lanes issue fresh material.
The independent [source gate contract](../reproduction-contracts/fusefss/gate-set/README.md)
freezes SiLU, exponential, reciprocal and width-specific normalization tables,
plus 275 complete/primitive clear cases. Selected-gate and complete-table replays
have different explicit coverage schemas; neither authorizes a complete-model
claim. Source-bound validation belongs to
[the gate-primitives evidence](../artifacts/fusefss-gates/local-v1/README.md).
Its accepted fresh attempt passes 239 native and 66 Python tests. Independent
checkers accept all 66 selected cases and 18,560 table outputs across 4,640 fresh
batches; the six external encodings remain local public frontend work. Archive
replay passes and rejects all 31 re-pinned negative mutations. Complete table
replay charges 1.428 GB of private material, 6.565 GB of public table copies and
9.215 MB of peer frames, with a 2,139,112-byte logical native peak and final zero
reservations. These cumulative costs are neither wire nor measured RSS.

### F2b: bounded complete source-gate composition

The [composed-gate slice](../artifacts/fusefss-composition/local-v1/README.md)
executes original-input SiLU/Phi, two-table nExp with its fresh product/floor,
protected max/causal reductions and reciprocal softmax, then private
exponent/mantissa extraction and both separate normalization affine floors.
Immutable public input intervals support checked-i128 signed38, nonwrapping
subtraction/product/sum and valid-index obligations. They remain caller promises,
not private membership checks or checkpoint range evidence. Original table bytes
and negative-floor boundaries are preserved; the existing width-two norm case
has a separately recorded exact rsqrt-table supplement.

Each role owns one register schedule and consumes externally issued fresh material
at bound public program counters. The runtime itself never issues or reconstructs
material. Existing leases and the 64 MiB native ceiling cover both roles, the
largest primitive and prior-frame overlap. Exported owners survive cancellation
and plan retirement. The complete 275-case protected/local-encoding export passes
the unchanged independent checker, including all source intermediates required
by its test-only diagnostic outputs. Cumulative private material, public copies,
native frames and logical ownership stay separate. Width-768 normalization must
execute; width 4,096 rejects the unchanged 32,768-instruction ceiling.

The earlier material/work estimates remain their own design evidence. Bounded
lookup capacity and scalar key reduction do not admit complete-model cumulative
work or select an SDK component. F3 must first bind full model-neutral lowering
and complete public range obligations, then reprice full-prefix D9 workloads
before checkpoint or role launch. Original-model and common-Qwen source/numeric
identities retain their separate qualification gates.

### Existing-path smoke checks

The coordinator checked two existing contracts before new native integration:

```sh
.venv/bin/python -m pytest tests/test_scalar_fss_reference.py -q
.venv/bin/python -m pytest tests/test_shared_rescale_sdk.py -q -k 'all_backends_and_layouts_preserve_signed_contract or replay_coverage_and_historical_material_contract'
```

These pass 8 and 11 checks respectively. The first covers bounded dense scalar
sharing and simulated authenticated peer exchanges; it does not use the new
ring50 backend. The second checks the existing three rescale backends, signed
rounding, fused/unfused layouts and replay accounting. This is source-checkout
Python 3.13.15 validation using the already-built
`python/pllm/_native.abi3.so` (SHA-256
`d8aa2635e2388eb38f5cec4a10314ea109901ccb6cb5baa3a119a0b628d6dc21`),
not an installed-wheel or new-FuseFSS validation receipt.

## Scalar integration boundary

The descriptor's public identity, types, geometry and source commitments precede
issuance. Protocol-3 shifted partitions and coefficient payloads remain at the
dealer. A general protected backend receives them only during key generation.
Each evaluator receives its own keys, mask shares and fresh correlations. The
public-boundary specialization additionally receives the unchanged original
public descriptor table, never its secret translated order or clear corrections.
Input and output values remain additive shares. A masked input may be opened;
an unmasked activation, selected interval or selected clear coefficient must not
be opened.
The Horner product uses the original shared input, not the opened masked input.

The initial GELU slice has 511 source/512 padded intervals, two arithmetic
coefficients, no Boolean outputs and identity Phi. Its slope is Q0 and constant
is Q12: no extra Q12 rescale follows the product. The signed38 effective-input
obligation belongs to source lowering; mathematical ring50 extension tests do
not discharge it. That frozen affine adapter still rejects Boolean/nonidentity
Phi. The separately implemented primitive owners do not implicitly expand its
descriptor contract.

## Resource and evidence rules

Preflight both parties' keys, dealer compilation and issuance overlap, triples,
frames, outputs, descriptor owners and scratch before randomness/allocation.
Charge live allocations separately from cumulative one-use material. Frames and
exported owners retain reservations until retirement. Keep setup, issuance,
online execution, local files and transport separate in reports. Historical
masked-wire, PUMA and scalar-reference costs do not transfer to this build.

Independent security review remains distinct from functional tests. Preserve
approved arithmetic, role/trust and output contracts; substantial substitutions
require an explicit decision. All unresolved model or SDK coverage remains
visible until its own gate passes.
