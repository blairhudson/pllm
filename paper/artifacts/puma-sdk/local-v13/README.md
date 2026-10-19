# Trie lookup, certified attention and shared quality outcomes

This slice adds explicit public-weight trie lookup to the ordinary PUMA SDK,
source-bound attention admission, and opt-in shared quality-outcome reporting.
Full-Qwen execution remains blocked: both pinned 4+2 source-publication attempts
reject at layer-six RMSNorm before any worker launch.

## Implemented contracts

- `lookup="trie"` shares each token's Boolean conversion across a complete public
  vocabulary prefix trie. Compiler-proven initial IDs or complete stable-selection
  results establish the exact token domain. Source, layout, owner, work and
  fresh-material commitments remain mandatory. Eight-row ownership is bounded at
  29,773,248 bytes; 64+8 lookup alone costs 641,966 ordinals, not a full-program cost.
- Workers recompute per-head Q/K matrix and bias statistics. Complete-vector
  energy certificates cover the original projection and both RoPE roundings;
  exact query rows and physical key columns bind each dot. The independent
  RMSNorm energy ledger and original one-truncation product remain. Final native
  admission intersects energy and any admitted scalar hull. Source-norm metadata,
  copies and scan/output overlap share the existing 1 MiB allowance.
- Shared quality `outcomes="record"` retains typed numeric rejection and other
  failures at their original cohort positions. Accepted-only scores disclose
  their sample counts; incomplete cohorts cannot rank. Caller-held cohort keys,
  frozen continuations, bounded UTF-8 serialization, atomic writes and cancellation
  cleanup retain explicit ownership. V1/default abort behavior is unchanged.

Compact nonlinear/normalization constructors and the bounded body-layer executor
also have native operator, work, allocation and retirement controls. Neither is
selected by the outer decoder. Their local controls cannot close complete-model
capacity, source-certificate or serving gates. No cryptographic review is implied.

## Validation and build identities

| Scope | Result |
| --- | --- |
| Broad installed wheel | 656 passed, 8 skipped |
| Final bound-intersection wheel, targeted installed SDK checks | 59 passed |
| Original native core slice | 125 passed |
| Complete native decoder/compiler slice | 116 passed |
| Native allocation/multiply/norm/PRSS/truncation integration | 54 passed |
| Final norm-constructor follow-up | 10 passed |
| Final attention follow-up | 7 passed |

Follow-up suites overlap the earlier suites; these rows are not additive.
The compiler suite contains **70 generated oracle cases, 9,816 complete tensor
checkpoints and 962 negative controls**. These include two-layer GQA, complete
synthetic vocabulary, private feedback, acceptance failure and actual PRSS/AES
truncation tapes. They are not pinned-Qwen hidden-state conformance.

All **369 installed `pllm/` files** match each wheel before and after its suite.
The broad wheel is
`522386a1696e556d4de39d81d49f92d12bab730a2c6dcd431d25ecca10a2d47f`.
The final wheel is
`4e4541d526da6413282a8622407e605d8dd6faed36d13b56351883287206fc64`.
Their source inventories differ only in `multiply_norm.rs` and its tests; the
targeted follow-up covers the final constructor change. Build receipts report
zero swap growth. These are guarded cached-local builds, not release attestations.
Clippy passes with the existing `needless_range_loop` and `nonminimal_bool`
allowances. Rust formatting and reviewed benchmark surface checks pass.

## Pinned source gate

Both attempts use four fixed public zero IDs, two selections,
`max_abs_activation=256`, checked ranges, trie lookup, tiled selection and peer
streams. Fresh SDK memory admission and the ordinary pressure watchdog precede
source publication. Both reject at selection zero, step 190, layer-six RMSNorm.
The final input interval is `[-31289011493243,31289011484027]` in encoded Q18
units. Both observe zero swap growth. Source reading and publisher CPU/RSS remain
charged; no role workers, online response, checkpoint quality or client-memory
tier are established. The earlier local-v12 tiled-lookup layer-eight rejection
retains its different execution identity.

## Evidence and replay

`summary.json` derives validation counts and rejection records from original
receipts, JUnit and logs. `build-v1/` and `build-v2/` retain source inventories and
guarded build receipts. `validation-v1/` retains the broad suite;
`validation-v2/` retains the passing final-wheel retry. `attempts/` retains the
original wrapper failure (before pytest) and explicitly excluded redundant run.
`checkpoint-v1/` and `checkpoint-v2/` retain the separate publisher attempts.
No wheel, checkpoint, private prompt, secret tape or native capture is embedded.

`inputs.tar.gz` is a final-source/test overlay for the baseline commit recorded in
`summary.json`. `build-v1/inputs-overlay.tar.gz` restores the two original native
files over that overlay; both variants are hash-checked against their build
inventories. Rebuild/install the chosen variant before replaying its original
commands. Installed tests override checkout import settings with
`python -I -m pytest -o pythonpath=.`; `-I` alone is insufficient. The native norm
follow-up log is a labeled recovered session excerpt; remaining native logs are
retained command output. The collector records command scopes and exclusions.

Verify retained files from this directory:

```sh
shasum -a 256 -c SHA256SUMS
```

Checkpoint replay additionally requires the original pinned source and fresh
host admission. Complete model/resource gates remain in
[the feasibility account](../../../methods/puma-feasibility.md) and
[completion plan](../../../methods/puma-completion-plan.md).
