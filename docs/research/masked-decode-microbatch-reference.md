# Deadline-bounded masked decode microbatch reference (#10)

## Outcome and scope

Implemented `scripts/probe_masked_decode_microbatch.py` and small independent
controls in `tests/test_masked_decode_microbatch_probe.py`. This is an executable
online row owner, **not** `CorrectionBatch` Preparation, a live SDK integration,
or a complete model response benchmark. **No new serving optimization is
established: online batching already exists.** No checkpoint/model values, SDK/shared
source changes, builds, installs, commits, or child agents.

Important inventory finding: online matrix batching already exists. The shared
`StageBatchScheduler` collects stage payloads, and
`transformer_engine.py:1550–1640` validates arithmetic/route/weight metadata,
concatenates rows, executes one native matrix call and splits replies. Therefore
this reference does not establish a new batching primitive or an improvement
over the SDK's existing adaptive scheduler. It investigates bounded ownership
and admission around the same mechanism. Its serial baseline disables batching.

## Actual source inventory and invariants inspected

- `server.py:83–111`: execution/session API principal, terminal flags, inventory
  roots, reserved attempts, reservation rows/bytes, activity and request encoding.
- `server.py:331–378`: burn reserved material on termination; retire exhausted
  inventory only after its response owners finish; reclaim idle/terminal sessions.
- `server.py:1008–1057,1172–1182`: prepared session capacity, inventory/reservation
  admission and monotonically advancing reserved rows.
- `server.py:1252–1294`: active inventory, matching API principal/model/stage,
  response-owned reserved attempt and seeded arithmetic profile validation.
- `server.py:1399,1448–1458`: preloaded correction consumption before execution,
  followed by terminal-session check. Failure cannot restore consumed inventory.
- `correction_rendezvous.py:357–419`: locked preloaded validation/consume. Existing
  batch is **one session**, unique correlation IDs, and all-or-burn within that
  session. It must not be repurposed as one cross-client atomic ticket transaction.
- `correction_rendezvous.py:327–355,421–453`: timeout/cancel abort, burn reserved
  attempts, terminal cleanup. Per-session burned attempt history remains bounded
  by session admission/idle reclamation, not by reusing tickets.
- `masked_runtime.py:316–411`: queue is unbounded; max batch limits dispatch,
  not waiting ownership. Adaptive scheduler yields/drains runnable contention and
  waits only after multiple rows arrive. Cancellation suppresses delivery but
  does not filter cancelled payloads before computation. Batch failure fans out
  across that selected scheduler batch.
- `preprocessing_inventory.py:152–175`: separate sequential MPC material
  inventory validates operation/shape/weight before advancing consumption.
  This reference does not replace either inventory implementation.

These are checkout-source observations. Execution uses installed
`target/frontier-sdk-env` native matrix wrapper; no checkout shadowing. Evidence
records checkout source hashes and installed native extension identity separately.

## Owner and trust boundary

Each separately admitted session has its own 32-byte HMAC credential, bounded
ticket table and one pending row. Trusted fixture Preparation preloads a unique
ticket/correction before online submission. Online request HMAC binds session,
ticket, every public compatibility field and immutable masked input bytes.
Server-local row expiry is an owner policy, not an authenticated client deadline.
Reply data has independent session/ticket/key HMAC; results are caller-owned
handles, not a shared response cache. No cross-client prompt or cache reuse.

Public grouping includes model, stage, immutable weight digest, ring/modulus,
input/output width, codec, wire/weight/activation bits and body fingerprint.
Supported bounded fixture uses prime modular arithmetic, int8 weights and raw
uint32 rows. Unsupported ring/codec/dimension/weight contracts fail closed.
This is not wrapping-ring, residue-codec or all-stage coverage.

Fresh input mask `r` and output mask `s` are generated independently for every
fixture row: `z=(x+r) mod p`, `c=(s-Wr) mod p`, reply `Wz+c`, client subtracts
`s`. Owner retains neither clear input nor masks. Tests use a separate Python
integer sum oracle; benchmark uses native fixture corrections outside online
timing and independently checks complete last-wave masked replies afterward.
Seeded NumPy RNG is reproducible **synthetic test material**, not production
cryptographic mask generation. Credentials are synthetic too. Production requires
fresh globally unique session/attempt IDs, fresh credentials, existing approved
mask generation and inventory authorization, not this fixture API.

Constructor preflight accepts only existing int8 NumPy arrays with both public
dimensions in 1–1024; no array-like conversion occurs. It checks initial weight,
executor and batch-workspace reservation against budget before serializing/copying
weights or creating native executor/matrix. Integer policy limits require exact
Python `int` (not bool/float): batch, sessions, tickets/session and queue each
1–32, memory budget 1–134,217,728 bytes. Wait/idle accept finite Python int/float,
excluding bool, with wait 0–1 seconds and idle >0 through 3600 seconds. Adaptive
must be bool; clock and optional test kernel must be callable. Admission for live
sessions/material remains incremental and bounded, not a reservation for all
configured future sessions.

Normal execution requires installed native extension, `MaskedGEMM(threads=1,
build=False)`, and native Rust backend; Python fallback fails closed. Explicit
kernel injection is for lifecycle tests only, never benchmark execution.
Sentinel/mock tests prove insufficient initial budget rejects before weight
serialization/copy, native executor creation or compilation; policy/geometry and
missing native checks also precede allocations.

`dispatch()` is a single executable native compute owner. Its externally driven
timer interface is `next_wake()`; no background threads or SDK wiring. Eligible
public groups dispatch at batch size or oldest row's coalescing deadline; oldest
eligible group and FIFO rows first. One pending row/session prevents one session
from filling a batch. Admission failure does not consume a fresh ticket.
Authenticated malformed rows burn only their own ticket; invalid credentials
cannot burn a victim ticket. Duplicate submission races accept exactly one row.
Tickets atomically become consumed under owner lock before kernel invocation.
Failures and cancellation never roll them back. Cancellation can occur while
native execution is outside state lock; cancelled outputs are never delivered.
Session-generation identity prevents a recreated identifier from receiving an
old response or having its new pending state changed by old work.

Size/wait bounds are not hard real-time guarantees: external pump scheduling,
queued eligible groups, native service time and OS scheduling add latency. Expired
rows are purged before dispatch and checked again after native work. Idle expiry,
cancel and close burn unused material and remove queued ownership. Consumed/burned
tombstones persist only within bounded live ticket tables. No replenishment beyond
ticket limit: terminate/re-admit with fresh identity/material. Adaptive option
dispatches a lone queued row immediately; under contention the finite coalescing
window still applies. Principal-level fairness/rate limiting is not implemented;
one principal could own multiple independently admitted sessions.

`close()` only retires sessions and their unused material. It does not join native
work, release matrix/executor, or permanently close admission. Cancelled inflight
generations remain charged until dispatch unwinds. Matrix, executor and frozen
weight snapshot retire on final owner/reference release; allocator/native process
resources need not immediately return to OS. The post-close baseline charge still
includes those resources. Session cleanup is not full resource release.

## Memory and performance accounting

Executable `charge()` admission reserves:

```
2 * weight bytes + 1 MiB executor allowance
+ 4096 * live session generations
+ sum(1024 + encoded public key bytes + retained correction bytes) over all live tickets
+ queued/inflight rows * (2048 + 4*input width + 4*output width)
+ sum(encoded public key bytes) over queued/inflight rows
+ max batch size * (12*input width + 20*output width)
```

Detached cancelled session generations remain charged while a native batch owns
them. Capacity bounds sessions, lifetime tickets/session, rows and batch size.
Admission rolls back newly added state on memory failure. Corrections, payload
copies, concatenation/serialization, integer correction-add buffers and output
lifetimes have explicit reservations; fixture inputs/masks remain outside owner
charge and are reported separately. The 1 MiB executor allowance and Python
object reservations are **not an allocator/native-stack RSS proof**. Native worker
stack, extension/import base, allocator fragmentation and host headroom need actual
fresh-process measurement. No fixed client-RSS tier admission is asserted.

Synthetic benchmark executes serial wall-clock cohorts, not a simulated arrival
timeline: baseline batch=1 with simultaneous ready session waves; batched same
waves; then baseline/fixed-wait/adaptive lone-session waves. All sessions issue
exactly one row per wave and only advance after all wave replies complete. This
is not realistic continuous token arrival or independent token feedback latency.
Fixture Preparation/matrix setup CPU/wall time is separate from online admission,
HMAC, queueing, matrix execution, correction and reply HMAC CPU/wall time.
Report includes p50/p95 request latency, p95 queue delay, rows/sec, owner peak
charge, fixture retention upper estimate and process peak RSS. RSS is one process
high-water across serial cohorts, not per-cohort matched fresh-process qualification.

Initial tiny evidence: 16x16 weights, 4 sessions, 2 waves, batch=4, 0.25 ms wait,
one native executor thread. Loaded baseline used eight matrix calls; batched used
two. Fixed-wait lone rows visibly paid queue delay; adaptive dispatched immediately.
Only 8 loaded rows and 2 low-load rows/cohort: startup/cache/order/timer noise
dominates. **Do not use smoke timing ratios as measured serving improvement.**
Replay exact smoke:

Small correctness suite: **50 tests passed** in installed validation environment,
including original 23 numeric/lifecycle tests and constructor preflight/native-only
controls.

```sh
target/frontier-sdk-env/bin/python -m pytest -q tests/test_masked_decode_microbatch_probe.py
target/frontier-sdk-env/bin/python scripts/probe_masked_decode_microbatch.py --inputs 16 --outputs 16 --sessions 4 --batch 4 --rounds 2 --wait-ms 0.25 --seed 10 --output docs/evidence/masked-decode-microbatch-synthetic.json
```

Evidence JSON embeds probe/test/inspected-source hashes and installed native
extension hash. Exact times are intentionally not frozen into this note.

## Parent serialized gate

Runnable small next command (parent only, under frontier admission/lock):

```sh
target/frontier-sdk-env/bin/python scripts/probe_masked_decode_microbatch.py --inputs 128 --outputs 128 --sessions 8 --batch 8 --rounds 16 --wait-ms 0.25 --seed 10 --output docs/evidence/masked-decode-microbatch-parent-synthetic.json
```

Five serial cohorts; 128 loaded rows per loaded cohort, 16 lone rows per low-load
cohort. One native worker thread, matrix 16 KiB, explicit peak owner charge below
2 MiB for these dimensions, fixture-retention estimate 458,752 bytes. Installed
Python/native base is additional (tiny smoke process peak about 43 MiB); request
128 MiB fresh-process allowance plus host headroom. Expected duration seconds,
no checkpoint loading and no compilation. This command is supplied, not run by
worker. Repeat in matched fresh processes and reverse cohort order/warm kernels
before any stable synthetic performance inference. Compare existing adaptive
SDK scheduler separately before claiming incremental benefit; no live integration
is included here.

Missing gates: actual admitted complete-response workload, independent session
arrival/feedback and cancellation loads, full codec/ring coverage, existing SDK
baseline, fresh-process client and Preparation peak RSS/CPU, TTFT, N−1 decode
throughput, all-link traffic including authorization/corrections/metadata, public
headroom, and >=90% remote body MACs separately for prefill/decode. Native fixture
MAC work is remote but does not establish whole-model placement. No physical
operator independence, privacy upgrade, traffic/tier qualification or SOTA claim.
