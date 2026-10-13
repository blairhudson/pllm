# Cancellation-aware prefill cuts: bounded screen

## Verdict

**Not expressible by current SDK contract. No promotion.** This is ranked screen
9's mechanism, named **H7** in `docs/plans/low-tier-all-link-hypotheses.md`, not that
file's correction-upload H9. Conditional body savings need a new compiler/runtime
and authenticated provider lifecycle contract. Cancellation frequency, CPU,
latency, process peak RSS, physical-wire bounds and source remote-MAC qualification
remain unknown. No checkpoint/model/provider loading or measurements ran.

Source evidence:

* `crates/pllm-compiler/src/decoder_runtime_schedule.rs`: complete prefill/decode
  phase schedules contain ordered dependencies, grouped outputs and state owners;
  no cut manifest, suspended program counter or partial-transaction continuation.
* `crates/pllm-compiler/src/decoder_continuation.rs`: continuation admits exclusively
  complete per-layer full key/value owner pairs; graph is derived from whole
  prefill. Fixed prefix/suffix token admission is not arbitrary graph suspension.
* `python/pllm/runtime/transformer_client.py`: `snapshot()` copies position,
  caches and shared KV, not graph-local activations or next-operation pointer.
  `prepare_ids()` resets and executes whole forward. Lease claims burn on use;
  close burns remaining reserved rows; unreserved cancellation discards issuance.
* `crates/pllm-core/src/prepared_stage.rs`: native stage claims retire input masks
  before returning, construction failure cancels masks; output-mask/scale owner
  bounds remain tied to one prepared stage, not resumable graph-local state.
* `python/pllm/runtime/client.py`: `_fresh_decode_epoch_inventory()` rejects
  active responses. `rotate_decode_epoch()` completes authenticated session,
  closes channel, retires inventory, then prepares anew and restores response
  snapshot at completed token boundary. Pending selected token is executed next.
* `prepared-frontier.md`: concurrent Preparation acknowledgements precede seal;
  stage window bounds offline work only. It cannot overlap an online response.

Counterexample: after first layer, two states with identical first-layer KV but
different residual activation yield different subsequent logits/KV. Partial KV
alone cannot distinguish them. Toy test demonstrates this information loss;
current snapshots also omit unfinished layer ownership and graph PC. Merely
changing inventory rows or reusing decode epochs does not supply missing state.
Complete-token chunk continuation is a different experiment: it must independently
prove same original full-prefill ordering/numerics and cannot justify mid-layer cuts.

## Required new contract (proposal, not implemented)

Freeze public cuts before private execution: monotonically increasing **compiled
operation-group boundaries**, final cut equals graph end. Never split fused linear
outputs, a quantizer/dequantizer boundary, active tensor tile, attention reduction
or unacknowledged operation. No private-value adaptive cuts. Compiler derives
exact live-in/live-out sets, row demands and remote ownership for every segment;
preserve original dependency order, W8A8 scales, rounding, prefix-f32 reduction,
terminal pruning, saved-activation bits, aliases and full-KV final state.

Each typed cut envelope must include:

1. Version; exact source tensor/config/plan/composition/numeric/schedule and cut-table
   digests; compiler provenance; request owner and immutable full input/output caps.
2. Next operation-group PC, segment ordinal and exact committed public cut; global
   token positions, masks/valid extents and original prompt/history commitment.
3. Every live tensor: producer/output ID, shape, dtype, layout/strides, endian,
   scale association, exact bit payload commitment, ownership and alias map.
   Includes residuals, normalized/quantized inputs, branch values, attention or
   MLP temporaries if live. Compiler bounds serialized length before allocation.
4. Every layer's K/V owner, active/allocated geometry, committed prefix extent,
   private staged append extent, original positions and transaction ID. Untouched
   future layers must be explicit, not interpreted as valid zero KV. No externally
   visible partial response/prefix-cache publication. Whole prefill commits only
   when all layers and final logits succeed; cancellation rolls back/discards the
   request transaction without corrupting prior committed prefix.
5. Original RNG state and pending token if present (normally neither consumed
   before prefill completes), application response/event counters, output budget,
   source-bound memory admission and maximum cut lifetime.
6. Retired session/provider epoch, drained channel sequence high-water marks,
   fresh continuation nonce/generation, expiry, inventory/lease IDs and exact
   issued/claimed/unused row commitments. No masks or tickets transferred as
   reusable state. Private tensor bytes stay client-local; providers receive only
   authenticated opaque commitments and public cut identity, never plaintext
   activations. Commitments need request-private blinding where necessary.

Protocol order:

* Admit full compiled-cap state plus peak live cut, temporary export/import copies,
  scratch, bundle/page caches, Python/native baseline, masks/corrections, metadata
  and transport owners against source-bound conservative native admission. No
  reduced full-cap budget because cancellation is expected. Process qualification
  still requires every fresh-client peak <=268,435,456 bytes. Check >=90% remote
  body MACs separately for original successful prefill/decode schedules.
* While globally idle, issue only next segment and any explicitly attached fixed
  output epoch. All issuance/acknowledgements priced; seal then reserve once.
  Admission binds fresh inventory to exact segment and prior cut commitment.
* Execute original schedule; cancellation may be observed during upload, after
  seal, on claimed online request, or at an acknowledged graph operation. Stop
  submitting operations, drain workers/transport; claimed rows already burned.
* For continuation, finish complete segment, validate exact live manifest and
  staged KV transaction, **end/authenticate retirement of online session**, close
  channel, confirm provider tombstone and sequence drain, burn all reserved unused
  rows and discard issued unreserved results. Retain global execution queue seat
  across pause so another request cannot bypass global idle-only exclusion.
* Only authenticated confirmed retirement enables new Preparation. Timeout,
  malformed ACK, missing retirement, ambiguous replay or expiry burns material,
  erases cut state and fails closed; no further preparation until retirement is
  established by explicit provider recovery. No reuse of failed reservations.
* Atomically consume single-use cut receipt/generation on next admission. Provider
  acknowledges exact continuation commitment before execution. Duplicate, stale,
  wrong source/cut/owner/geometry/expiry or altered payload rejects and burns new
  reservation; interrupted admission cannot fork. Preserve tombstones through
  replay horizon. Final segment commits full KV/logits then ordinary fixed output
  epochs continue. Cancellation at any pause cancels future unissued segments.

This requires new native resumable executor and compiler liveness/transaction
proof, session APIs and independently reviewed authentication/replay semantics.
Toy `authenticated=True` flag is only state-machine input, not cryptography.

## Bounded oracle and costs

`scripts/probe_prefill_cancellation_cuts.py` uses two toy layers, ten operations,
four rows, three remote i8-linear operations per layer and f32 additions. It checks
exact final toy output/persistent-state bytes after every one of nine internal
frontiers. Minimal live manifest retains future inputs plus persistent outputs;
largest payload is 64 bytes. It is not transformer attention or real-model exactness.
Failure tests cover dependency order, typed-state corruption/missing state,
generation replay, ownership conservation, in-flight claim burn and retirement
exclusion. No synthetic result establishes cryptographic security or native support.

Public candidate cuts are PC 5 and 10. Control prepares entire prefill plus same
first fixed output epoch at once; both prepare 36 stage rows on success. Candidate
issues 12 rows first, 24 only after authenticated first-cut retirement. Compare
cancellation **after acknowledged PC 5** in both: same completed arithmetic,
control has already issued all 36, candidate only 12. Other observable traces are
preparing (full segment conservatively issued but unacknowledged), sealed (reserved
but unclaimed) and online-inflight (claim burned, full reply charged as upper bound).
Evidence retains observations and terminal issued/reserved/claimed/burned/discarded
counters; correction ACK and in-flight counters are distinct from online ACKs.
Separate byte gauges show issued/ACKed/in-flight/reserved/claimed-burn/unused-burn/
discarded correction payload and acknowledged/in-flight reply bounds. Gauges
overlap; only directed-body totals are summed. Evidence includes twelve fixed
cancellation traces, not empirical cancellation prevalence.
Burn totals include claimed rows; `burned_unused` does not double-count claims.

Supplied toy prices: correction 96 bytes/row, ingress 64, reply 96;
cold public delivery 4096; session aggregate control 1024 and cut aggregate control
512 bytes. These are **assumed bounded body envelopes**, not existing serialized
SDK frame sizes. Control aggregates include both-direction admission, inventory,
completion/cancel, receipts/ACKs and cleanup within declared envelope; a real design
must enumerate every directed endpoint, seed, header/body and new cut artifact.
Unknown TCP/HTTP/WebSocket framing, retries and physical wire have no finite bound
here. Neither directed link relocation nor unacked issuance is treated as saving.

For issued rows I, claimed rows Q and sessions J, conditional all-directed-body
upper bound is `cold + 96*I + 64*Q + 96*Q + 1024*J + 512*nonfinal_cut_receipts`.
Unsealed failures charge all issued corrections; failed claimed calls charge full
possible replies. Real supplied finite frame envelopes can replace prices; without
them this formula is not a traffic upper bound for deployment. Source distribution,
new manifest delivery and source-native memory remain separately unmeasured.

| Toy trace | Control bodies | Candidate bodies | Candidate minus control |
| --- | ---: | ---: | ---: |
| Successful fixed epoch | 14,336 B | 15,872 B | +1,536 B |
| Cancel after acknowledged PC 5 | 10,496 B | 8,704 B | -1,792 B |

For cancellation fraction p **at this exact observable point**, mean body saving
is `p*1792 - (1-p)*1536`; strict break-even `p > 1536/3328 = 46.154%`.
Denominator for all attempts is `(1-p)*14336 + p*10496`, not successful-request
traffic alone. At p=1/2 saving is 128/12416 = 1.031%; synthetic, not deployment
frequency. For heterogeneous cancellation points use weighted actual trace totals,
including preparation failures and lost retirement; cannot reuse this threshold.
General gate: avoided never-issued correction/output-epoch bytes must exceed all
new controls/artifacts and changed arithmetic, priced over all attempts. Early
cancellation after whole first segment issuance may save less; late cancellation
after final segment issuance saves none. Success necessarily gains one pause and
control overhead here, no correction reduction. Pause includes retirement/drain,
fresh preparation/admission and restore; CPU/latency cost **unknown, not zero**.

Toy source-bound owner check prices baseline + scratch + correction owner + live
hidden/partial KV + export copy. It is an allocation illustration, **not <=256 MiB
whole-client qualification**; real source live manifests, native admission, page
owners and fresh-process peaks are absent. Toy remote-body fraction is 100%; real
prefill/decode fractions unproven. H7's 3–10% planning range is not established.

## Replay and next gate

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath=. \
  tests/test_prefill_cancellation_cuts_probe.py -q
target/frontier-sdk-env/bin/ruff check scripts/probe_prefill_cancellation_cuts.py \
  tests/test_prefill_cancellation_cuts_probe.py
target/frontier-sdk-env/bin/python scripts/probe_prefill_cancellation_cuts.py \
  --output docs/evidence/research-prefill-cancellation-cuts.json
```

Evidence hashes exact inspected sources and all three owned source/test/reference
files. Evidence does not hash itself. Replay deterministic for same file bytes;
changes to shared inspected sources legitimately change hashes. Only four owned
files changed. Retained checks: **44 tests pass; Ruff clean**. No heavy exact
command offered: current executor cannot run proposed
stage cuts. Parent-only next gate is implementation of contract above, then opt-in
serialized fresh-workers exact Qwen2/Qwen3/Llama saved-activation/logits/full-KV
controls, followed by matched pinned source cohort with real cancellation traces,
fresh-client Low admission/peaks and all-link body/stream/CPU/latency accounting.
Until that executable exists, inventing a `fresh_workers` command would imply
unsupported mid-layer SDK functionality.
