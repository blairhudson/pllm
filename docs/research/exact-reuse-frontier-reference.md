# Exact-reuse frontier: corrected installed-prefix comparator (v2)

**No substantial additional all-link opportunity survives this bounded screen.**
The former 1,024-byte sibling saving and 27.59–76.27% scaling claims are withdrawn.
They omitted installed intermediate-prefix publication from the control. Local
parent-qualified delta restoration and application-operation receipts remain
CPU/logical-I/O directions, not new provider-traffic savings or measured speedups.

## Evidence inventory and rejection of v1

* [`exact-reuse-frontier-reference.json`](../evidence/exact-reuse-frontier-reference.json)
  is rejected/superseded v1, retained **byte-identically** at its original path.
  SHA-256: `fb1e62c8c2b579658f355aa75aa06a32dd0ccfbf67153291f8a05827836e56c2`.
  Its incorrect baseline disqualifies its incremental branch conclusions.
* [`exact-reuse-frontier-reference-v2.json`](../evidence/exact-reuse-frontier-reference-v2.json)
  is the corrected replay target. It records v1's hash, rejection, and each
  recorded-source/current-v2-file hash mismatch. Script, tests and this reference
  have changed; v1 cannot be replayed with these current sources. The original
  source bundle was not archived, and its reproduction is **not asserted**.

At the start of this revision, the on-disk v1 above already routed corrections
Preparation-to-Inference and matched all its then-current source hashes. That
differs from the auditor's reported earlier stale snapshot (Preparation-to-Client
and mismatched hashes). That earlier snapshot has not been recovered; do not equate
it with the identified on-disk bytes or invent its hash. Keeping the identified v1
unchanged preserves the actual available artifact rather than rewriting history.

The probe imports no model runtime or checkpoint tensors. Bounded differential
tests load the actual Python prefix cache with synthetic arrays and process-local
synthetic basis seals; they execute no native decoder or real model values.
Existing decode-cache/recovery/inflight cohorts stay separate. Their published
savings and RSS are not pooled or multiplied into v2.

## Correct installed policy and admission

Source inspection, recorded in v2 source hashes:

* `python/pllm/runtime/prefill_cache.py:put_prefixes` publishes powers of two,
  every eight positions, and `total-1` after a completed prefill. If copying the
  full prefix exceeds the payload budget, it filters positions to the admitted
  row limit and adds that limit as a cut. References clip the final shared
  immutable eight-row block; **the entire backing block is charged**, not only
  its visible prefix. `_put` shares blocks, retains qualified logits when legal,
  and evicts entries in LRU order under byte and 4,096-entry ceilings.
* `runtime/client.py` invokes `put_prefixes` after completed initial prefill and
  proper-prefix continuation, then stores the full root with logits. Its earlier
  exact/longest-prefix probe already restores these entries before fresh issuance.
  `_profitable_cached_prefix` uses the unchanged stage-body admission estimate.
* `runtime/transformer_client.py:SnapshotStateBasis._prefix` already derives
  qualified shorter lineage from cache-eligible state, binding source, native
  phase, position, prefill extent, execution digest and verification lineage.
  Proper-prefix entries deliberately have optional logits. Existing suffix
  execution computes the needed final logits; there is no need to invent a new
  native cut primitive or execute an unused prefix head.

The installed contracts still require exact original tokens and causal positions,
compiled admitted `prefix_f32` full-KV state, eligible owner-free lineage, matching
verification strength and bounded ownership. `store=False`, response-owned,
retained/windowed or otherwise inadmissible states cannot be made reusable by
changing this reference. These are not grounds for disabling reuse in only the
control. Differential tests compare the modeled block/LRU retention to the actual
source cache at ample and constrained budgets, including clipped blocks, root
admission failure, entry pressure and logits-free prefix hits. They do not replace
native compiled admission or establish checkpoint behavior.

## Reassessment of the three directions

### 1. Sibling factoring: installed saving, not a new network lead

For original prompts `P+A` and `P+B`, a completed first prompt already publishes
eligible scheduled cuts. Boundaries 8/10 and 48/50 in v1 are **already scheduled**.
Both v2 arms use the same selection, shared-block payload budget (32,768 synthetic
bytes), LRU and entry bounds. They pay the same copy/publication counters and use
the same full-inventory or one-selection-cache policy alternatives.

In the three-sibling 8/10 fixture, **both arms** execute 23 synthetic causal rows.
The previous 39-versus-23 comparison was false. Incremental network saving is
**zero**. The 2/4/8/16-sibling 48/50 scaling cells also show **zero incremental
saving**, not 27.59–76.27%. First-unique, repeat, restart and unrelated cells likewise
show no extra provider-traffic gain. Restart retains the already-installed durable
root control instead of comparing recovery to replay.

Earlier publication during concurrent prefill may change latency, but an existing
serial prepared queue can wait for the first completed prefill/response and use
the same cuts. No latency limit or forced concurrency makes that feasible control
illegal here. Therefore v2 does **not** credit early sharing with a byte saving.
A fair future concurrent study must compare waiting/completed-cache scheduling,
inflight complete-prompt joins where applicable, queue/idle-only preparation,
cancellation and actual admission, not an artificially independent branch control.

For an **unscheduled** boundary, compare the nearest retained installed cut, not
zero. For 45 shared rows in a 47-row prompt, the nearest installed cut is 40:
gross remaining opportunity is five causal rows per later sibling. With an admitted
retained eight-row grid the gap is at most seven rows; eviction/admission can
invalidate that bound, so there is no universal saving. Existing `_prefix` can
already derive such lineage; exposing caller-selected positions would be a policy
extension, not new numeric execution. Candidate metadata/publication/retention
costs and eviction damage are additional, not free. Existing explicit prefix
prefill/state controls can trade a separate initial execution/publication and
terminal/logit work against that gap. Those exact costs are unknown here, so net
runtime gain is **null**, not the gross gap multiplied by a checkpoint byte rate.

Any caller-declared boundary is a public length/scheduling choice; compare both
arms under the same disclosure and original token equality checks. Tokens stay
local and are not reordered, normalized or semantically matched. No substantial
first-request application-boundary advance is established.

### 2. Parent-qualified delta child restoration: local work only

Installed exact child hits already issue zero provider rows but reconstruct and
authenticate full-prefix state. Reusing the exact current parent's immutable seal
could avoid rereading/copying its pages, authenticating only the child manifest,
new pages and exact child logits before a transactional append. A path hash alone
is insufficient: bind original source, numeric schedule, geometry, verification,
history, ordered page commitments and position; exclude mutable aliasing.

The simulator verifies parent identity independently and charges the complete
fixture manifest plus only appended page payload on qualified hits. This is a
logical-I/O construction, **not** an efficient native seal implementation. Root,
restart without a parent, changed/corrupt parent and unsupported state require
existing full restoration or ordinary fresh execution. Failed reads stay charged;
partial state is never published. In the repeat fixture, delta hits save 528
synthetic validation/page bytes but **zero network bytes**.

Qwen2.5 geometry from the identified existing reference gives 24,576 K/V bytes per
causal row and 607,744 logit bytes. Seven child hits after a 54-position root have
a 9,805,824-byte **parent-K/V payload reread ceiling**, not encrypted-I/O, CPU or
traffic measurements. Root restore, manifests, child logits and AEAD stay charged.
Physical disk amplification and allocation/RSS are unknown.

### 3. Durable application-operation receipts: retry semantics only

Returning the durable result of the **same completed logical operation** can avoid
new client/model setup, K/V restoration and repeated sampling/validation. It cannot
memoize a fresh generation by text or seed. A new invocation needs a new operation
identity and must advance its actual RNG normally; pending sampled tokens never
become executed K/V rows. Both state and independent oracle compare full logits,
active K/V, original history, outputs, pending selection and final RNG.

This can serve a restarted adapter's first delivery only if that logical operation
already completed. A genuinely new first operation misses. Compare the optimistic
installed ordinary-cold/full-hit-warm floor, including surviving public artifact
caches: provider-traffic gain remains **zero**. Receipts need authenticated bounded
storage, complete request/source/policy binding, caller-trusted freshest head,
durable publication/acceptance and explicit ambiguous-commit reconciliation.
External tool side effects require their own idempotency/transaction contract.
The bounded in-memory receipt fixture does not implement a durable receipt service.

## Ledger, thresholds and ownership

All byte totals are stipulated **fixture** sums, not complete runtime ledgers.
One synthetic causal macro-row costs 16 Client-to-Inference upload bytes, 24
Inference-to-Client reply bytes and 24 **Preparation-to-Inference** correction
bytes (`q = W*r - s`). No correction reaches the client. Four abstract 32-byte
fixture controls/session use Client-to-Preparation, Preparation-to-Client,
Client-to-Inference and Inference-to-Client links. Actual runtime framing/control
bytes, including Inference-to-Preparation acknowledgement, remain **null/unmodeled**.
The directed audit rejects same-total correction misrouting; issued = claimed +
burned, and burned material is never refunded. Initial cold public artifact costs
4,096 fixture bytes in both arms. Local storage contributes no network bytes;
logical reads/writes, validation, retained files/state and capture owners stay
separate. Full-inventory and fixed-one alternatives remain visible; choosing their
lower-byte envelope per workload is an optimistic comparator, not a predictor.

For unscheduled cuts only, potential break-even requires
`hit_rate * remaining_gap_link_bytes > extra_capture + lookup + publication + misses + eviction_damage`.
Subtract the best feasible explicit prefix/waiting control as well. Real rates and
costs are unknown, so no production threshold is reported. Scheduled-cut
incremental network ceiling is zero. Delta/receipt local-work thresholds require
measured validation, copying and publication costs, not assumed free storage.

Bounds enforce 128 token positions, 128 durable fixture nodes, 32 receipts and
16 KiB node manifests. Prefix entries additionally use the installed 4,096-entry
cap with a 32,768-byte fixture payload budget. Proper-prefix publication/capture
ownership is counted identically in both arms. Reported conservative logical
decode, receipt and prefix metadata/copy envelopes are separate owners, **not
measured RSS** or a largest whole-client admission proof. Interpreter, native
runtime, model/bundle/tokenizer, sampling, compiled-capacity K/V, allocator, OS
cache, old/new copies, AEAD and retained responses remain additional. New CPU/RSS,
physical disk/wire and whole-client 256 MiB qualification remain unknown/false.

All matching is caller-local. No cross-user private sharing or private remote
content-addressed index is authorized. Remote matches/access patterns disclose
equality and sizes; encryption alone does not hide them. A future protected lookup
must price padding, all replicas, misses, controls and access/key contracts.
Caller-owned encrypted external storage requires bounded buffers, trusted newest
head/key custody, explicit old-head forks, full bindings, atomic authenticated
publication, single writer and charged uploads/downloads, CPU, failed writes and
orphan/old-head retention. It is not a free remote cache.

## Next gate and reproduction

**Do not implement sibling factoring for network savings on this evidence.**
The best remaining narrow direction is parent-qualified delta lookup, contingent
on generated-state tests of immutable seals, atomic append/failure rollback and
full source/verification/RNG parity, followed by serial parent-authorized CPU/I/O
and whole-client accounting against installed full hits. Receipts belong at an
application's existing idempotency boundary. Neither justifies a network claim.

```sh
target/frontier-sdk-env/bin/python -I -m pytest -o pythonpath= -q tests/test_exact_reuse_frontier.py
target/frontier-sdk-env/bin/python -I -m ruff check scripts/probe_exact_reuse_frontier.py tests/test_exact_reuse_frontier.py
python3 scripts/probe_exact_reuse_frontier.py --output <fresh-path-v2.json>
```

The writer requires a new path. Compare the generated JSON to v2, not rejected v1.
No runtime edits, builds, installation, checkpoint values, live/timed cohort or
commit are required. Source/cache differential tests and exact evidence replay
are bounded correctness checks, not performance measurements.
