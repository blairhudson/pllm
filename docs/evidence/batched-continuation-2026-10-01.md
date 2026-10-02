# D/E1 decoder continuation — corrected lineage boundary, 2026-10-01

## Shipping vertical slice; generated fresh promotion vetoed

**Native batched prefill continuation and production provider admission work.
Generated decoder KV is not eligible for ordinary fresh-prompt reuse. Full D
generated-fresh-promotion work remains pending numerical compatibility.**

The ordinary SDK uses distinct inference/preparation child processes with no
reference admission adapter. Coordinator-owned provider admission reconstructs
the exact source/config/composition/schedule/numeric/bound contract and checks
installed ownership/stage artifacts before reserving rows. Client requires exact
`pllm.decoder_continuation_session.v1` acknowledgement; missing/forged acknowledgements
fail before stage calls. No schema downgrade or implicit sequential fallback.

Native compiler extension `pllm.decoder_continuation.v1` declares existing full-KV
inputs and append/valid-prefix operators while retaining original grouped prefill
linear stage identities. Absolute positions, RoPE and causal masking remain source
operations. Native checked token/memory bounds apply before append mutation.
Hybrid/recurrent/sliding/shared state remains unsupported. Invalid shape/nonfinite
execution burns the continuation session and one-use reservations.

## Execution-basis and geometry confinement

`RuntimeSnapshot.state_basis` carries process-local sealed source/numeric binding,
native phase digest, execution-chain digest, actual prefill reduction extent and
optional response owner. Native graph/schedule execution produces the qualifier;
it is not a remote callback label. Every decode makes the lineage incremental.
The seal authenticates provenance fields, **not mutable KV payloads**; existing
shape, finite-value and exclusive-owner validation remains mandatory.

- Every `ExactPrefillCache.put*`, including internal insertion, requires an unowned
  completed-prefill basis. Unknown/unqualified, decoded, owner-confined and relabeled
  state is rejected. There is no approximate opt-in class.
- Cache keys use a separate v2 domain and commit the original full attention
  reduction extent. Proper-prefix lookup requires that extent to equal the new
  full input width. Even prefill-only KV from another reduction width can differ
  numerically: the pinned follow-up exposed **1.8366047 logits** drift. Such a
  geometry mismatch now misses safely without changing the legacy executor.
- Immutable eight-row blocks remain bounded and deduplicated for qualified
  prefills. Hits return independent active-prefix snapshots with their sealed
  basis preserved through clipping.
- Generated state remains an independent private conversation snapshot. Only
  `restore_for_response(snapshot, matching_previous_response_id)` can activate it
  for native continuation, with source/numeric/execution basis and full-KV shape
  checks. Wrong owner/source/basis is rejected before stages. No generated block
  sharing or fresh-key promotion is implemented.
- Pending final sampled token is absent from evaluated state. EOS uses actual
  position/pending IDs; cancellation and `store=False` retain no completed private
  conversation snapshot. Retokenized text never substitutes for exact token IDs.

## Current measured production cohorts

JSON top-level `tiny` and `pinned_qwen` are the **post-veto** run. It records code,
source config, cohort and native contract digests plus real child PIDs. Independent
client/material horizons have separately charged warm-ups. One loopback repetition;
request latency includes request-sized preparation/authorization, excludes role
startup/loading. Covered bytes are online upload + download bodies, not full wire.

### Eight-layer Qwen2-shaped checkpoint

Source and branch both have 152 tokens; retained prefix 112; suffix 40. Payload
cap 1,048,576 bytes; retained after native reuse 788,496 bytes.

| Execution | Stage calls | Per-stage rows evaluated | Covered online bytes | Seconds | Full-logit difference vs fresh |
|---|---:|---:|---:|---:|---:|
| Fresh prefill | 32 | 4,864 | 6,151,048 | 1.005 | 0 |
| Explicit sequential control | 1,280 | 1,280 | 3,214,292 | 1.975 | 0 |
| Native batched suffix | 32 | 1,280 | 1,620,824 | 0.881 | **0; array-equal** |

Native suffix uses the same 40 complete one-use rows as sequential control, with
40x fewer stage calls. Selected token 183 agrees; every logit is array-equal.
Preparation during online execution, completed unused burns and idle unreserved
rows are zero for these request-sized horizons. Cancellation tests separately
check reserved-row burning while unreserved rows remain idle.

### Pinned Qwen2.5-0.5B-Instruct W8A8

Offline revision `7ae557604adf67be50417f59c2c2f167def9a775`; fixed input bound 128;
payload cap 67,108,864 bytes. First branch is token-count padded to the source's
63-token reduction geometry independently of numeric outcomes. Its retained
prefix is 40, suffix 23. Native execution consumes 23 complete rows / 2,208
per-stage rows, uses **96 calls** and **36,714,816 covered bytes** in **8.583s**.
Fresh execution uses 63 complete rows / 6,048 per-stage rows, 96 calls and
100,551,168 bytes in 16.627s. Native/fresh logits are **array-equal**.

Two changed-length branches contain 50 and 55 tokens. They correctly miss the
63-token source geometry and execute canonical fresh prefill, preserving every
logit exactly. Their `actual_execution_mode` is `fresh_prefill_geometry_miss`:
the requested benchmark mode is not misreported as an optimization. All branches
select expected token 16787. This is not a claim that arbitrary changed-length
proper prefixes are numerically interchangeable.

## Generated lineage: fresh numeric gate failed, explicit prior path valid

The initial pinned response has 38 input tokens and three evaluated generated
tokens, so private snapshot position is 41. Final sampled token remains pending.
**Zero generated tokens are promoted.** A 57-token ordinary follow-up matches
the generated token prefix exactly, but both generated lineage and original
38-token reduction geometry are ineligible for its fresh-cache lookup.

| Execution | Complete rows | Stage calls | Covered online bytes | Seconds | Every logit vs canonical fresh |
|---|---:|---:|---:|---:|---|
| Ordinary prompt, generated alias blocked | 60 | 384 | 96,123,332 | 16.671 | **Array-equal** |
| Canonical full-prefill control | 60 | 384 | 96,123,332 | 15.144 | Array-equal |
| Explicit prior owner + batched pending/suffix | 19 | 384 | 30,690,980 | 7.616 | **Different; fresh gate failed** |

The prior path differs from canonical fresh by **1.943387508392334 logits**.
Matching token 3966/text does not waive that failure. It is valid only against its
incremental execution reference: the opt-in test independently replays source
prefill, fixed evaluated teacher tokens and the admitted suffix through clear W8A8
kernels; every KV value, suffix logit and subsequent generated token is exact.
Actual singleton HTTP outputs are independently checked against those kernels.

The prior path's 19 complete rows / 1,824 per-stage rows remain valid incremental
measurements. They are **not selectable fresh-prompt optimization gains**. Current
generated cohort sets `fresh_generated_reuse_admitted: false` and its fresh numeric
gate false. The legacy executor remains unchanged.

## Historical evidence and checks

`historical_reference` preserves the original reference-adapter run.
`historical_pre_lineage_veto` preserves earlier production/research measurements,
including the 7.492s/30.69MB generated alias experiment. It explicitly records
`shipping_optimization_evidence: false` and failed generated-fresh numeric gate.
Those numbers are not shipping fresh-reuse evidence or current timings.

Checks after confinement: full focused run had 125 passes / 13 failures; two owned
missing-ACK fixtures were then corrected to same geometry, and those plus two new
qualified-block/all-logit tests passed. Final complete owned suite, including both
offline pinned-Qwen tests: **62 passed**. The remaining 11 failures are confined to
coordinator-owned `tests/test_prefill_cache.py`: raw snapshots lack explicit basis,
and cross-width hits are no longer legal. That test migration remains required;
no unqualified-default compatibility path was added to hide these failures.
Ruff and diff checks pass. Native compiler/build unchanged in this correction;
earlier two Rust tests and UV/maturin release installation remain verified.

Reproduce owned tests with `PLLM_RUN_CACHED_CONTINUATION=1`, and run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
uv run --no-sync python scripts/probe_batched_continuation.py \
  --cached-qwen --archive docs/evidence/batched-continuation-2026-10-01.json
```
