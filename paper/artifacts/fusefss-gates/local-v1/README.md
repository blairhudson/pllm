# FuseFSS protected gate primitives — local v1

Status: **historical bounded F2a validation**. This is the
gate-primitives slice in the [completion plan](../../../methods/fusefss-completion-plan.md).
The [scalar GELU evidence](../../fusefss-scalar/local-v1/README.md) retains its
original build and scope.

## Implemented contract

The native owners in `crates/pllm-garble/src/fusefss_backend/` compose existing
DCF, universal public-boundary lookup, ring products, frame and allocation owners:

- `predicates.rs`: actual fresh-mask compiler handoff, XOR-shared comparison and
  Boolean-VM results, with fresh ring1 triples for AND/OR.
- `conversions.rs`: exact floor TR/ARS, zero/sign extension, B2A, arithmetic-bit
  Phi selection with public addition, and full modular multiplication. Widths
  remain 1–50 and at most four lanes; no sharewise shift substitutes for floor.
- `lookup.rs`: complete public tables composed from fixed 256-entry chunks.
  Every chunk/lane consumes independent masks and keys; every party evaluates
  every chunk and sums opaque output shares. The 513-interval primitive and
  64 MiB combined native reservation ceilings remain enforced. Index and payload
  rings remain distinct source obligations; normalization payloads require ring50.

Aggregate admission precedes entropy. Failure, malformed frames, replay and
cancellation consume party state. Exported frames and results retain leases
until deallocation. Fixed allocation paths reject excess reported capacity before
filling buffers with secrets. Nested allocation-failure injection is partial.

`SignedRange::source_contract` and Phi's bit commitment are caller obligations,
not proof verifiers. Zero-extension of a lookup does not establish source-index
validity. Public context headers are not authenticated transport. Each runtime
holds one party's share; local issuance and final reconstruction are test-only.

## Evidence coverage

The [independent gate fixture](../../../reproduction-contracts/fusefss/gate-set/README.md)
freezes 275 source/primitive cases and the original table bytes. Two separate
native producers and strict checkers qualify the implemented subset:

| Replay | Required coverage | Excluded claim |
| --- | --- | --- |
| Selected gates | 66 cases: shifts, B2A, comparisons, Phi-select and six local exact-rational external encodings; 62 fresh protected issuances | Complete nonlinear gates |
| Public tables | 18,560 outputs across eight complete tables and 4,640 fresh four-lane batches | SiLU, nExp, softmax or normalization composition |

The table replay covers SiLU, E0, E1, reciprocal widths 1/2/3 and normalization
widths 4/768. The full 131,072-entry reciprocal layout has a separate bounded
endpoint/capacity test; it does not inherit exhaustive replay from smaller tables.
Large canonical indices return zero; malformed noncanonical inputs consume state.

The results separate cumulative private material, duplicated public tables, actual
peer frames, peak logical native reservations and final retirement. None is
process RSS, physical wire, separate-role performance or a whole-model cost.
The fixture parser, caller buffers, descriptor parser, exports and filesystem
work remain outside the native resource counts.

### Recorded results

The original attempt was `target/fusefss-f2-final-v2`. Git retains its historical
records; source copies, receipts, logs and archive-only validators were retired.

- **239 native library tests pass**, with seven ignored in that invocation.
  The selected-gate and table exporters run separately and both pass.
- **66 Python checks pass**: 20 frozen-reference, 22 independent compiler-fixture
  and 24 gate-set controls.
- Clippy and Rust formatting pass. The exact Clippy command retains the three
  recorded lint allowances; this is not an unqualified all-lints claim.
- Both independent output checkers passed. Native reservations returned to zero
  in every replayed record.

| Both-party accumulated cost | Selected gates | Complete-table replay |
| --- | ---: | ---: |
| Private material bytes | 78,992 | 1,427,629,056 |
| Public table-copy bytes | 0 | 6,564,569,088 |
| Actual peer-frame bytes | 13,344 | 9,214,976 |
| Peer frames | 148 | 9,280 |
| Peak logical native reservation bytes | 28,426 | 2,139,112 |

Table replay performs **136,585,344 DCF evaluations per party**, including every
zero-support chunk. Its complete export test takes about 938 seconds, including
issuance, evaluation, checks and export; this is not an isolated online timing.
Bounded live ownership does not remove cumulative material, public-copy or
traversal costs. The earlier full-model feasibility estimates retain their own
scope until complete source-gate schedules supply a new ledger.

## Correctness tests

Current native tests live in `crates/pllm-garble/src/fusefss_backend/`.
The [independent fixtures](../../../reproduction-contracts/fusefss/gate-set/README.md)
retain source equations, expected values and direct native-export checkers.
Use ordinary Cargo and pytest commands; generated exports belong in ignored
`target/`. Experiments use ordinary benchmark outputs.

## Earlier attempts

The first full-library run found a flaky existing secret-replication negative
control: assigning 255 to a random header byte sometimes left that byte unchanged.
The regression now XORs header bytes and asserts that the malformed frame differs;
payload noncanonical-value controls keep their original mutation. Production
secret-replication code is unchanged.

Failed and interrupted attempts did not supply successful results.

## Next complete gate

F2 remains open: compose original-input SiLU/Phi; nExp's two table lookups, fresh
product and floor; protected max/causal reductions and reciprocal softmax; then
normalization exponent/mantissa extraction and both affine floors. Every edge
needs source scales and signed38, nonwrapping arithmetic and lookup-domain
provenance. The complete 275-case checker must accept actual protected execution
before advancing to complete semantic decoder, authenticated roles and ordinary
SDK/model qualification. Reprice cumulative D9 full-prefix work before model
launch. These finite controls do not constitute independent cryptographic review.
