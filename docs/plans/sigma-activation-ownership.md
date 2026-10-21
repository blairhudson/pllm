# SIGMA bounded activation ownership proposal

Status: **NOT APPROVED**. Native/typed five-word Select validation passes:
471 unit, five external and twelve doc tests (eight opt-in skips), 280 leakage
controls, library Clippy and compiler check. Matrix handoff baseline: `920c48c1`.
This is a working plan, not
implementation evidence, source-schedule parity, or runnable SIGMA/SDK coverage.
Parent owns all integration and documentation outside this file.

## Smallest next executable gate

After approval and successful Select validation, implement exactly:

```text
Y = TR(X, 6)       # signed floor represented in the reduced ring
D = DReLU(Y)       # common XOR-masked [Y >= 0], including zero
R = Select(D, Y)   # common additive-masked D*Y
```

For Q12 input, Y/R are Q6; input width n becomes n-6. Admit only widths
supported by every constituent primitive (n >= 9, n <= 64), 1–256 lanes,
and the existing aggregate 64 MiB logical ceiling. This prefix is not GELU/SiLU.
Use existing D16 GPU127 and D17 fresh full-output-ring masks; no numeric variant.

## Ownership, uses, and leases

- Consume each external capability once into a fixed-template region. Separate
  dealer-mask owner from each role-local evaluator owner; never combine secrets.
- Precommit template digest, ordered use slots, operand positions, full source/
  plan/session/mask bindings, roles, widths and shapes before issuance.
- Prefix use counts: X once, Y twice (DReLU and Select), D once. Only R exits as
  an ordinary single-consumer capability. No internal capability escapes.
- Retain original native backings and leases; authorized internal helpers borrow
  them. No payload copying, replacement-capability minting, cloning or remasking.
- Bounded per-use ledger burns each attempted use. Wrong binding, retry, failure
  or cancellation terminates local region; coordinator explicitly cancels sibling.
- Reuse method-neutral consumer-use declarations beside `edge_contract`; public
  metadata remains declarations, never executable authorization or range proof.
- Charge retained inputs/masks, keys, ledger/metadata, workspace, frames and output
  overlap together before allocation/randomness. Erase/deallocate before refund.
- Preserve matrix singleton guards until original backing retires. No guard
  detachment, admission-credit gap, or authorization of another matrix operation.
- Existing public single-consumer APIs and `LocalAssumption` provenance remain;
  no arbitrary graph executor, exported fanout, transport or material codec.

## Source schedule and later gates

Pin: `mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`.
Paths below are under `GPU-MPC/ext/sytorch/ext/llama/`:

- `api.cpp:2707–2750` SlothGelu: TR6, DReLU(Y), Select(D,Y), A=2R-Y,
  clip to 255, LUT, Select(D,X), subtraction. `2784–2827` SlothSilu uses 1023.
  Full activation requires X twice, Y three times, D twice, plus clip/LUT uses.
- `api.cpp:2989–3036` SlothClip compares A-K, complements predicate, selects
  A-K+1, then adds K-1. Signed-minimum absolute value and comparison no-wrap
  bounds require explicit admission; masked-value checks cannot prove them.
- `api.cpp:2741/2743` and `2818/2820` defer LUT/final-Select reconstruction;
  subtraction precedes final reconstruction. Separately reconstructing those
  outputs changes exchange schedule and cannot silently claim source parity.
- `api.cpp:2048–2052` dispatches LUT_dpf widths >=8 to LUT_dfpet;
  `2121–2208` and `lut.cpp::{keyGenLUTDPFET,evalLUTDPFET_1}` specify masked
  result/sign correction, Select, and 2*Select-res. Legacy Lookup is not a drop-in.
- Numeric target remains `paper/reproduction-contracts/shared-profile-resolution.md`
  §2.1: early floor before absolute value, frozen paper-equation tables/hashes.

Lookup source audit can proceed independently before ownership approval. Bounded
candidate: 1–256 lanes, index widths exactly 8/10, public tables exactly 256/1024
canonical entries, output widths 3–64, common masked index to fresh full-ring
masked table value. Bind table/context/issuance; charge full scan and exchanges.
Before implementation freeze GPU127 sign-correction mapping and reconstruction
boundary. Changed correction protocol, terminals, exchanges or numerics need
explicit approval. FuseFSS additive-share lookup is not a common-wire substitute.

## Parallel Sol slices after approval

- Metadata Sol: `crates/pllm-core/src/edge_contract.rs`; bounded consumer-use
  declarations/digests and tests; preserve existing descriptor encoding.
- Ownership Sol: new `crates/pllm-garble/src/masked_wire_reference/typed/region.rs`
  and region-local tests; fixed prefix, private ledger, ownership and admission.
- Oracle Sol: bounded schedule/oracle files under `paper/reproduction-contracts/`;
  independent arithmetic, endpoints, use counts and table identities.
- Lookup Sol: dedicated lookup contract document; source audit first, no inferred
  implementation approval. Parent assigns exact new oracle/document filenames.
- Parent alone integrates `typed.rs`, `typed/selection_bridge.rs`, module exports
  and other docs. Freeze interfaces first; metadata precedes region integration.
  Oracle/lookup audits are independent; parent runs heavy validation serially.

## Independent acceptance checks (not executed by this plan)

- Independent signed-floor/Select equations; negative residues, -1/0/63/64,
  signed endpoints, both predicate masks, reduced widths/Q and absent provenance.
- Later activation: clip K-1/K/K+1, negative bin boundaries, signed-minimum
  handling; GELU -1/63/64 -> -31/63/33; GapARS quarter-ring endpoints unchanged.
- Duplicate/reordered/missing/wrong-slot uses; source/session/role/mask/template
  substitution; retry/cancel/failure burn; compile-fail cloning/capability escape.
- 1/256 lanes, checked overflow/over-budget admission, entropy/allocation failures,
  exact retained frame/output charges, erase-before-refund, matrix guard retention.

## Next approval question

Approve fixed-template owning regions with single external consumption, retained
original backings/leases and bounded precommitted internal-use ledgers, initially
only TR6 -> DReLU -> Select? External single-consumer APIs, D16/D17, numeric
semantics, role separation, matrix singleton and failure-burn contracts stay;
exported fanout, cloning, remasking and arbitrary graph execution remain excluded.
