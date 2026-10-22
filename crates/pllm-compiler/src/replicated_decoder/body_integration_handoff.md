# PUMA binary-body handoff

## Directed transcript test repair

Existing compiler run `target/parallel-lanes-compiler-current-r2.log` reports 367
passes and one failure: this test expected six post-rotation frames, observed four.
Source derivation: `puma/multiply.rs::rotate` starts truncation after receiving
NextRank's product; `puma/protocol.rs::begin_inner` emits A->B and B->A
`MaskedInput`, plus H->A and H->B `Correlation`. H completes locally. Evaluators
each consume one opening and correction, emit A->B/B->A `Reshare`, then complete
on peer reshare. Thus complete phase counts are 3/4/2, not 3/6/2.

Test now checks exact ordered directions/phases, three-word openings/rotations/
reshares and six-word corrections, shared truncation binding, and explicit open/
reshare barriers. Roles derive from public correction sender because fresh setup
uses OS entropy. Duplicate attack follows one accepted open inbox frame. Replay
and foreign-context attacks follow a fully completed first product and enter the
second using the same live runtime. Foreign-context attack uses the fixed-direction
Multiply phase: independent foreign setup may choose a different truncation pivot,
so matching foreign open/correction routes is not guaranteed. All attacks require
`Binding`, then permanent `Retired` for receive/advance/consuming finish.

Earlier coordinator validation (before the D23 additions below): ordinary compiler
library suite reported 374 passed, nine opt-in tests ignored. This is not validation
of the new D23 tests; this worker ran no Cargo, builds, or tests.
Replay:

```sh
cargo test --release --offline -j 1 -p pllm-compiler --lib
```

## Implemented

`body.rs` includes new `body_integration.rs`. `Compiled::new` lowers an exact
already-admitted `Program` interval of gather/gather/add-or-Q18-product triples
into the existing bounded binary body. `begin` consumes a quiescent `Worker`;
`Invocation` routes real protocol frames; `finish` returns that same worker with
live outputs and original retirement ledger restored. Original consuming runtime,
PRSS cursor, context, product widths/order/bounds and complete work are retained.
No new default, identity, cap, equations, source authority or acceptance policy.

This is deliberately narrow: public-weight opt-in; affine single-owner gathers;
original products at most 128 lanes; no matrix/norm-proof/checked-row replacement.
Public file owners cannot become secret operands through this adapter. Checked
validity, protected selections, feedback and final masking outside the interval
remain ordinary worker nodes. Original program payload plus body/bridge/storage
overlap and conservative lowering scratch must fit existing 64 MiB cap.

Allocation gate now precedes **all** lowering Vec/BTreeMap construction and
`Plan::new`. Its allocation-free checked upper bound counts duplicate external
owners, all site outputs simultaneously live, every interval node exported,
export-copy overlap, primitive ledger/gathers/bounds stack, fixed bridge/plan/
executor owners, and 4096 bytes per bounded constructor entry for vector growth,
tree entries, nested retirement vectors and simultaneous boxed copies. This
deliberately overrejects. Final exact plan/work/bounds/byte validation remains and
must also fit that early upper bound. Export dependency discovery now borrows
Code/Words directly instead of allocating `Node::inputs()` vectors.

This native validation does not include a trained-checkpoint publication,
installed SDK run or matched performance benchmark.

## Generated tests (coordinator already wired program.rs)

Tests need program-module privacy for generated fixtures. Cases: independent integer expectations,
original-vs-body execution, complete restored live-owner state, two stable private
selections with real first-ID feedback, ties, sticky rejection/final masking,
wrong program owner, unsupported mode/span, unchanged aggregate byte cap,
malformed-frame retirement and cancellation. Added: early rejection before entry
to the first allocating constructor region, with a successful-entry positive
control; strided/broadcast operands; exported gather owners and externally
retained inputs; multiple product widths (2/1, 3/2, 17/16, 128/127) followed by an
ordinary protected product; premature finish and pending drop release ownership.
Constructor-entry instrumentation is a control-flow seam, not a global allocator
or memory-erasure monitor. These passing synthetic graphs are not complete D9
full-prefix decoder or full-checkpoint evidence.

No production program.rs change is necessary for this bridge API. Caller compiles
`Compiled::new(program.clone(), WeightMode::Public, start, end)`, hands in an idle
worker with matching exact Program Arc and step, routes Invocation frames under
existing router limits, then resumes returned Worker normally. Initial import
boundary is supported when `start == program.inputs`. D23 adds opt-in
`Worker::advance_before(target)` and `Compiled::advance_to_start(&mut Worker)`:
exclusive original-node boundary, `Frames` or `Reached`, no target execution.
Route every frame and its generated responses across the three original runtimes
before advancing again. Begin only after all roles report `Reached` and the router
is empty; a local result cannot establish global quiescence. Ordinary `advance`
remains unchanged. No automatic/default dispatch or runtime reset is authorized.

## D23 boundary and lifecycle regressions

The non-public `#[cfg(test)]` `Worker::test_boundary_state` accessor in
`runtime.rs` adds read-only test observation, not production behavior.
Tests reuse existing Builder fixtures, ordinary workers, consuming protocol owners,
and public boundary methods; no private cursor mutation or cloned runtime.

- Globally drained three-role prefix routing; idempotent idle boundary calls.
- Local, ordinary-product, Tensor and 128+1 TensorLoop prefixes; target/body owners
  absent at entry, tail complete before entry, ordinary suffix products afterward.
- Adjacent and separated compiled intervals; independent exact integer outputs,
  complete retained-owner presence/control-state comparison with ordinary execution.
  All retained prefix lanes, exported gather lanes and first-body result lanes are
  final outputs, not merely a selected scalar. Random local shares are not compared.
- Negative source-read sentinel: a deliberately unavailable late file owner is not
  accessed before target; ordinary execution of the target read returns `Binding`.
  This is a boundary test, not valid source admission or source authority evidence.
- Prefix checked rejection, retained validity owner, later successful checked
  fallback and final sticky masking; admitted and rejected input controls.
- Backward/out-of-range, foreign exact Program Arc, active-at-target and partially
  buffered prefix inbox failures; permanent retirement and ownership release.
- Pending midgraph body cancellation/drop, prefix-to-body and body-to-ordinary-
  suffix replay retirement. Router bytes are copied, never protocol owners.
- Real Builder payload pressure: live prefix owners remain in unchanged aggregate
  cap; rejected control never enters allocating constructor region.

The source-read sentinel is intentionally invalid; it proves only that the target
is untouched. It is not a source-import or checkpoint-acceptance fixture.

## Remaining gates

Full semantic-body dispatch (rows, matrices, sources and checked sticky validity)
remains separate;
this bridge does not silently authorize it. Expanded test-only oracle capture is
explicitly rejected until its compact intermediate-boundary mapping exists.
No full-Qwen admission/execution, quality, security qualification or performance
claim. Existing full-Qwen range/source/resource unknowns and approval boundaries
remain unchanged.

## Deterministic exact parity

Factory is available:
`puma::prss::insecure_test_support::fresh_compound_parties(Context, PrssLimits)`.
Compiler's dev-dependency enables the nondefault `insecure-test-support` feature.
The new helper uses only its PUBLIC fixed keys and fresh consuming owners, with
the ordinary fixture's input splitting. Context/session/program bindings are
intentionally identical across the two graphs; routers are entirely separate.
No runtime cloning/reset, incoming-seed rewriting, or masked table oracle.

`compiled_midgraph_exact_deterministic_full_wire_and_returned_owner_share_parity`
compares default ordinary `Worker::advance` against bounded prefix, compiled body,
then ordinary suffix. Cases cover local/product/Tensor/128+1 TensorLoop prefixes,
both adjacent and separated intervals. The test verifies original tile and tail
widths and compares every complete frame, including outer tensor frames, bindings,
payload bytes, sender, recipient and global delivery order. It compares exact
transcript prefixes through each body return, then the whole execution including
the subsequent ordinary protected product. No framing normalization is allowed;
any mismatch fails and needs coordinator investigation, not byte stripping.

Final ordinary outputs retain every lane of each previously retained owner and
both body results. Exact final per-role first components are compared before any
reconstruction. The authorized test-only `Worker::test_boundary_state` now also
borrows every actual live `(first, second)` array and reads existing core
`Cursors` (`public_blocks`, `pair_blocks`, `private_blocks`, `aes_work`). It checks
liveness, complete imports, idle Ready runtime, empty inbox and zero loop offset;
completion is allowed. It cannot mutate/reset/clone runtime or accept seeds.
Only test snapshots copy borrowed arrays into zeroizing owners. Active/pending
and retired diagnostic access is rejected without changing worker state.

Exact state comparisons cover ALL live owners, including presence/absence and
every lane of both stored components:

- At every compiled entry and return, against a third fresh deterministic graph
  executing original ordinary nodes with `advance_before` only (no body).
- At final completion, directly against the graph driven exclusively by default
  `advance`, before consuming either worker set with `finish`.
- All four core counters, including consumed AES work, at every compared state.
- Sticky accepted/rejected prefixes use the same entry/return/final comparisons.
- Repeated idle boundary calls compare full node/share/cursor state before/after,
  and assert target plus every later owner remains absent. Combined with exact
  ordinary entry state this checks no target execution or extra source draw.

The independent default-advance graph remains the complete ordered-byte and
final-state baseline; the third boundary graph is only the intermediate-state
reference. Separate routers are used for all three. No stored shares are inferred
from neighboring components. No API gap remains for this bounded idle-state
contract. Dead, non-exported intermediates and mid-protocol/tile-offset states are
not snapshot targets; every TensorLoop tile/tail frame is still compared in the
full transcript, and the complete 129-lane prefix owner is compared at entry.

Exact regression names:

- `compiled_midgraph_exact_deterministic_full_wire_and_returned_owner_share_parity`
- `compiled_midgraph_sticky_prefix_rejection_survives_body_and_later_safe_check`

Accessor rejection checks also extend
`bounded_boundary_invalid_requests_cancel_and_release_original_owners` and
`bounded_boundary_pending_prefix_inbox_and_midgraph_drop_cancel`.

## Validation

The targeted release/offline compiler run passes all 18 body-integration tests,
including complete ordered-frame, both-component and PRSS/AES-counter parity.
Lifecycle checks explicitly drop unused peer workers before checking Program Arc
retirement. The live-prefix admission rejection uses actual Builder ownership at
the unchanged 64 MiB cap; the smaller accepted control remains separate.
The complete release/offline compiler library suite passes 391 checks, with nine
explicit opt-in checks ignored, under serial test execution.

The feature-enabled core suite passes 504 checks, including all three deterministic
factory tests. A normal/build/features dependency traversal rooted at
`pllm-python` does not enable `insecure-test-support`; resolver 2 excludes the
compiler dev-dependency from that production graph. Explicit all-feature or test
builds can enable it and must never use these public keys with production secrets.

Exact targeted invocation:

```sh
cargo test --release --offline -j 1 -p pllm-compiler --lib body_integration_tests -- --test-threads=1
```

These are native generated-graph checks, not installed SDK, trained-checkpoint,
quality, performance or composed-security qualification.
