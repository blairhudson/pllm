# Assurance research

This directory contains specifications and public negative-control fixtures, not current
production evidence. Generated pass reports from handoff are deliberately excluded.

## Formal scope

`formal/manifest.json` defines ten finite SMT-LIB checks over small bitvector/algebraic or bounded
state models. Expected `unsat` can prove only stated finite property; expected `sat` asks solver
for counterexample. Some `unsat` checks prove leakage identities for deliberately weakened
schemes. None proves whole protocol, cryptographic hardness, deployed role separation, Rust
refinement, rollback resistance, or side-channel safety.

Run:

```bash
python3 research/assurance/run_formal.py
```

Runner loads system `libz3` through C API and writes result JSON to stdout. If library is absent,
it prints explicit unavailable diagnostic to stderr and exits nonzero. Unavailability, solver
`unknown`, malformed output, or expectation mismatch never becomes pass. Save output as new dated
evidence only after reviewing environment and scope.

## Attack fixtures

`attacks/fixtures.py` executes finite public examples against deliberately weakened constructions:
mask reuse, affine-label reuse, sparse point-and-permute, omitted truncation carry, early-exit
metadata, and low-rank masking. It also exhaustively checks one ideal finite-view control.

```bash
python3 research/assurance/attacks/fixtures.py
```

These fixtures do not attack current runtime. Production adversaries must receive only corrupted
party's permitted view. Failed attacks are `not_refuted`, never proof of privacy. See
`attacks/catalog.json` and `obligations.json` for integration targets and unresolved claims.
