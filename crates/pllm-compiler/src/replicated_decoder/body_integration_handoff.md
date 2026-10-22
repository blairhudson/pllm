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

Validated in the ordinary compiler library suite: 374 passed, nine opt-in tests
ignored. The repaired directed-transcript regression passes within that suite.
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
boundary is supported when `start == program.inputs`; arbitrary mid-graph entry
needs an existing quiescent idle seam. Current ordinary `Worker::advance` does not
stop at arbitrary requested nodes, so automatic mid-graph dispatch is **not**
claimed. Do not enable by default or reset runtime to create such a seam.

## Remaining gates

Full semantic-body dispatch (rows, matrices, sources and checked sticky validity)
remains separate;
this bridge does not silently authorize it. Expanded test-only oracle capture is
explicitly rejected until its compact intermediate-boundary mapping exists.
No claim of fixed-seed PRSS byte parity, full-Qwen admission/execution, quality,
security qualification or performance. Existing full-Qwen range/source/resource
unknowns and all approval boundaries remain unchanged.

**Requested fixed-seed transcript/share parity remains blocked, not implemented.**
`pllm-core/src/puma/prss.rs::SetupParty::with_entropy` is private to core; its
`SeedEntropy`/`SeedPurpose` injection is available only to core descendant tests.
Compiler's existing `workers()` uses OS seeds. Changing encoded incoming seed
messages cannot reproduce each role's private/own seeds. Neither cloning live
runtime nor resetting its cursor is permitted. None of the three owned files can
provide a safe fixed-seed runtime factory. Coordinator needs a separately scoped
core test-support facility (also accessible when core is a compiler dependency),
then compare ordered sender/recipient/frame bytes AND every returned local share
for the body interval and subsequent ordinary product under the same public test
keys/context/input shares. Current layout tests use exact products and prove only
integer/layout/ownership behavior, **not** PRSS transcript/share parity. No such
factory or production entropy change was added.
