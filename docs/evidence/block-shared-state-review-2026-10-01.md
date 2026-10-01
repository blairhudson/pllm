# Block-shared exact state reuse — 2026-10-01

## Decision

Implemented immutable eight-row KV blocks in the existing trusted-client
`ExactPrefillCache`. Completed-prefill checkpoints share retained buffers;
retrieval creates independent mutable snapshots. Removing a checkpoint decrements
references, and the last reference zeroizes the block. Logits survive a checkpoint
replacement only when its KV state is bit-identical. Caller snapshots are never
mutated. Oversized prefills retain only a preflighted prefix; a separate 4,096-entry
ceiling bounds metadata growth.

No new component or topology is introduced. Existing `ClientPrefixReuse` selects
the same compiled prepared execution and one-use inventory semantics. Byte
admission still compares sequential suffix messages with ordinary batched prefill.

## Payload control

The cached pinned Qwen configuration supplies 24 layers, two KV heads, 64 features
and vocabulary size. Synthetic finite KV arrays isolate storage; these are not
actual Qwen states or sampled peak-memory results. At 512 rows and a 64 MiB payload
limit:

| Policy | Retained payload | Checkpoints | 384-token common-prefix hit |
| --- | ---: | ---: | --- |
| Old ascending full-copy/LRU payload model | 62,318,080 B | 5 | No |
| Executed block-shared cache | 13,190,656 B | 68 | Yes |

The unbounded full-copy demand would be 422,282,752 B. The new cache retains 64
physical KV blocks and one logits vector. Counts exclude Python metadata,
materialized runtime copies, candidate copies and peak RAM. The old control is an
independent payload-accounting model of the previous checkpoint policy, not an
old-process memory measurement.

## Canonical branching control

One generated eight-layer Qwen2-shaped W8A8 checkpoint runs through two role
children, ordinary SDK responses and the compiled schedule. A first 212-token
prefill populates a 1 MiB cache. A 151-token branch reuses 128 tokens. A third
request executes the same branch with `store=False`; both request greedy sampling.

- Reused online masked upload/download: **933,674 / 898,346 B**.
- Fresh online masked upload/download: **2,398,104 / 3,712,496 B**.
- Combined reduction: **70.0%**, from 6,110,600 to 1,832,020 B.
- Selected output text and authoritative usage agree.
- No plaintext prompt/token-ID bytes or online preparation are recorded.
- Reused suffix makes **736 stage calls**, versus **32 batched fresh calls**.

This validates transport and cache connectivity, not representative model quality,
real-checkpoint savings, full-wire bytes, CPU/latency or independent operators.
The call-count tradeoff is material; smaller bodies do not establish lower latency.

## Compiler gates and next work

General generated-prefix promotion is not activated. Explicit response
continuations already retain evaluated state and track the final sampled but
unevaluated token; their lifecycle remains separate. A general cache needs
transcript/token identity and numeric parity before admitting those states.

Batched suffixes remain unimplemented: the current compiled decode query bound is
one. `forward_ids` iterates those decode steps; it is not a batched continuation
phase. Next slice must declare a native cache-aware multi-query/state contract,
preserve causal masking and numeric boundaries, bind provider admission and burn
fresh rows on cancellation. It must compare complete logits/state and report
dependent calls as well as bytes before replacing sequential suffix execution.

## Reproduction

Requires the existing cached pinned Qwen configuration; no checkpoint download.
Generated transport weights are created locally.

```sh
uv run --no-sync python scripts/probe_state_reuse.py --verify-tiny --archive docs/evidence/block-shared-state-screen-2026-10-01.json
uv run --no-sync pytest -q tests/test_prefill_cache.py
```

Both commands were executed. Tests cover alias isolation, last-reference erasure,
branch deduplication, eviction correctness, oversized preflight, logits/state
consistency, malformed snapshots and existing SDK inventory/continuation paths.
