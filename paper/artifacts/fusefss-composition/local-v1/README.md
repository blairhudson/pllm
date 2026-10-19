# FuseFSS bounded protected source gates — local v1

This slice composes the approved ring50/Q12 source gates over PLLM's existing
protected predicates, conversions, public-table lookup and ring products.
It follows [F2 in the completion plan](../../../methods/fusefss-completion-plan.md).
The earlier [scalar GELU](../../fusefss-scalar/local-v1/README.md) and
[gate-primitives](../../fusefss-gates/local-v1/README.md) evidence retain their
original source snapshots and scopes.

## Native execution contract

`crates/pllm-garble/src/fusefss_backend/composition/` contains:

- Original-input SiLU/Phi, signed floor-to-index conversion and the frozen table.
- Two-table nExp, a fresh shared product and its exact source floor.
- Protected active-row maximum, saturation, complete denominator reduction,
  reciprocal lookup and unrenormalized source softmax.
- Fixed-bin private exponent/mantissa extraction, the source rsqrt table,
  LayerNorm/RMSNorm and two separate affine floors.

Public intervals, scales, causal masks and affine coefficients are checked using
integer arithmetic and bound into typed schedules. These are sufficient range
proofs **conditional on the caller's public input promises**. Canonical individual
shares do not prove private membership; no checkpoint range is qualified here.
The table formula and a separate width-two rsqrt supplement preserve the original
275-case fixture, source lock and rounding identities.

Each runtime owns one role's registers. It requests material at a public program
counter; an external trusted dealer supplies fresh primitive owners. No evaluator
issues its own material or reconstructs an intermediate. Final fixture diagnostics
are test-only outputs; ordinary gate constructors return final shares. Plan,
context, role and step checks precede consuming execution. Malformed frames,
replay, wrong material, failed feeds, cancellation and allocation errors retire
the active owner; exported buffers retain leases until deallocation.

The plan ceiling is 32,768 instructions, 65,536 registers, 256 public tables,
16 MiB of plan ownership and `2^26` cumulative work units. Native live ownership
remains bounded by the existing combined 64 MiB ceiling. Complete admission
includes both runtime owners, the largest primitive and previous-frame overlap;
every later allocation rechecks shared headroom. Material streams between
primitives, while one lookup retains all its chunks. Table support through width
4,096 does not imply complete normalization support: both width-4,096 schedules
exceed the instruction cap and reject before protected issuance.

## Evidence boundary

The full native exporter reconstructs all **275 frozen cases**. Of those, 269
exercise protected gates/compositions and six remain local exact-rational external
encodings. All outputs must match the unchanged independent full-case checker;
an omitted, reordered or substituted case rejects. The width-768 norm must execute,
and its ordinary regression no longer accepts a silent resource skip.

The fresh source-bound run passed **278 native library tests** (eight opt-in
fixtures ignored by the ordinary suite, with the complete exporter run explicitly),
**66 Python fixture checks**, native-table replay, formatting and Clippy with its
recorded allowances. Read-only archive replay passes; **13 basic archive controls**
pass and **46 mutated evidence variants** reject, including re-pinned output,
source, accounting and coverage substitutions. Inputs matched before and after
native execution. `verification.json` binds the archived result to these replay
scripts and observations.

The 209 new composition fixtures cumulatively consume **81,551,384 private
material bytes**, **42,353,952 duplicated public-table bytes** and **10,082,032
native peer-frame bytes**. Their largest planned combined native envelope is
**11,882,256 bytes**; final reservations are zero. The 66 separately reported
primitive/local-encoding cases add 78,992 private bytes and 13,344 native frame
bytes. These sums describe this heterogeneous finite fixture set, not a model,
network measurement or serving comparison.

Receipts bind compiled source, original fixture/oracles, before/after identities,
native case coverage and final zero reservations. They separately record:

- cumulative private material and duplicated party-public-table payloads;
- retained original public-table/plan owners and conservative live envelopes;
- DCF keys/evaluations, triples and actual native frame counts/bytes;
- sampled live reservations, which can miss transient issuance scratch and are
  **not** a true high-water mark, allocator peak or process RSS.

Caller buffers, fixture parsing, reconstruction/export, source snapshots and
filesystem work remain outside the primitive allocation ledger. Cumulative public
copy bytes are local ownership work, not a measured network delivery. The trusted
dealer is local; context headers do not supply authenticated role transport.

## Reproduce

From repository root, with a new directory per fresh native attempt:

```sh
.venv/bin/python -B paper/artifacts/fusefss-composition/local-v1/run-native.py --output target/fusefss-composition-replay-new
.venv/bin/python -B paper/artifacts/fusefss-composition/local-v1/replay.py
.venv/bin/python -B paper/artifacts/fusefss-composition/local-v1/negative_controls.py --evidence paper/artifacts/fusefss-composition/local-v1
```

The native driver runs the full exporter/checker, library regressions, independent
Python fixtures, table generation, formatting and Clippy. It snapshots all Rust
and source-oracle inputs before execution and rejects drift. Clippy retains the
three explicit lint allowances recorded in its receipt. Read-only replay checks
the archived hashes and pinned independent oracle without issuing new material;
negative controls exercise re-pinned corruptions as well as broken hashes.

## Review resolutions and remaining gates

Independent engineering review found two actionable evidence gaps. Cumulative
public lookup copies are now charged separately, with independently counted
3,174,352/6,348,752-byte reciprocal controls at k768/k1024. The required k768
normalization regression now fails on any admission rejection. Dedicated tests
retain genuine width-4,096 capacity rejection. Earlier unsigned-shift,
original-input SiLU, subtraction-overflow and double-floor counterexamples are
covered by the exact source schedules and fixtures.

This qualifies finite source-gate execution, not the paper's optimized backend,
complete semantic decoder, original-model or Qwen ranges/quality, authenticated
roles, SDK availability, independent cryptographic review or serving performance.
F3 must compose the complete model-neutral graph and reprice full-prefix D9 work,
including matrix operations, lookup, selection, fresh material and delivery,
before checkpoint or worker launch.
