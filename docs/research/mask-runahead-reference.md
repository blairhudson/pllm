# One-slot native mask runahead reference

Investigation 3 of `docs/plans/tier-objectives-ten.md`. Research only; no SDK,
native registration, configuration or capability changes.

## Result and scope

Existing native expansion is usable from a bounded worker. The reference expands
one next-stage chunk from an **already-reserved** lease while its caller reports
the current remote stage in flight. Installed native masks and row tickets match
the synchronous lazy path byte-for-byte at 16, 24 and 32 bits. Native masking and
unmasking match an independent signed-integer linear oracle. Existing fused
quantize/pack and dequantize boundaries also match byte-for-byte, including the
direct native stage-owner control.

22 small tests pass. Barrier-controlled fake remote waits prove scheduling
overlap with actual native expansion, **not network latency or response gain**.
No timing cohort, checkpoint, provider service, model generation or build ran.
There is no TTFT, decode-throughput, traffic, RSS-tier or SOTA qualification.

Files:

- `scripts/probe_mask_runahead.py`: exclusive-lease one-slot wrapper and bounded
  native-expansion timing CLI.
- `tests/test_mask_runahead_probe.py`: installed-native parity/lifecycle controls.
- `docs/evidence/mask-runahead-synthetic.json`: replay, hashes and missing gates.

## Inspected real ownership and client lifecycle

`crates/pllm-core/src/mask_stream.rs` owns two SHAKE256 readers and a per-row
one-use ledger. Domains include the existing request binding. `take` marks rows
used before expansion/allocation; invalid/replayed claims cancel the cursor.
Disjoint out-of-order claims reset/skip the stream without prefix allocation.
`burn` is idempotent; `cancel` drops readers and marks all rows used. Exported
buffers remain valid after cancellation: this is logical burn, not revocation.

`crates/pllm-python/src/mask_stream.rs` wraps that cursor in a mutex. `take`
releases the GIL around native work, then copies native Vec buffers into Python
bytes. `PreparedStageRows.take_masks` provides readonly NumPy views of those
bytes; it does not reimplement the PRG.

`PreparedInventory.reserve` burns reservation availability before online use.
`PreparedInventoryLease.take` advances stage offsets and derives existing row
attempt IDs under its lock, then calls the native cursor. It closes on exceptions.
`close` burns the remaining per-stage reservation without expanding it. One
closed lease does not burn another disjoint lease's rows.

`python/pllm/runtime/client.py` reserves inventory before starting online activity.
`_begin_preparation` rejects online activity. `_finish_prepared_response` closes
the lease before `_end_online`, then allows idle-refill scheduling. The reference
must be drained at that close boundary, never after declaring the response idle.
A barrier test invokes these real lifecycle methods against minimal client state
and a recorded refill submission: online guard remains held while expansion is
blocked; refill becomes schedulable only after drain/close. No real refill runs.

## Reference state machine

Constructor requires an untouched actual native, unverified
`PreparedInventoryLease`, public unique stage order and a 1–32-row chunk bound.
Eager fallback and verifier ownership are rejected, not silently changed. Caller
transfers exclusive ownership of the lease; aliases must not claim its stages.
Public fixture seeds are deterministic test data, **not production randomness**.

`remote_started(current)` -> `prefetch(next, count)` -> `remote_finished()` ->
`consume(handle)`. Only the immediate public next stage is allowed. There is one
worker, one future slot and at most one submission per reported remote window;
the executor queue cannot accumulate speculative chunks. There is no advance
across the last-stage/token boundary. Row claims and attempt IDs still come from
the existing lease, not a second seed reader or ticket generator.

Consumption clears the slot atomically and returns masks/tickets exactly once.
Replay or competing consumption failure closes the wrapper. Context exit handles
normal completion and exceptions; explicit `close` handles cancellation. Close
sets `closing`, drains queued/running expansion without cancel-and-retry, closes
the original lease, discards the future slot and sets `closed`. Unused expanded
material is counted as waste. Worker failure is propagated on consume and closes
ownership; early stop must use context exit/close even if no consume occurs.

Logical burn means no new claim/delivery or ticket reissue. It does **not** claim
secure erasure of immutable Python bytes, traceback-held objects or caller-owned
returned arrays. Production cancellation must await drain on a control thread;
calling close from an executor hook is outside the contract. Test hooks use
barriers, not sleeps, and are not transport/provider hooks.

Prefetched rows count as SDK stage claims even if never sent to Inference.
`waste_bytes` separately charges discarded speculation; SDK claimed/consumed
counters must not be reported as successful remote execution.

## Memory/resource charging

For future chunk geometry `(c, n, m)`, fresh input/output material is
`B = 4*c*(n+m)` bytes even for 16/24-bit wire rings (native masks expand to u32).
Constructor admission requires:

`base_peak_bytes + 3*B_max + 8,388,608 + 65,536 <= cap_bytes`.

`B_max` is the largest allowed future chunk across this lease. Three buffer
charges cover the retained fresh buffer, native Vec/PyBytes copy overlap and an
additional boundary/caller copy allowance. The 8 MiB thread-stack reserve and
64 KiB executor/bookkeeping allowance cannot be lowered. The original cursor
ledgers, main thread, current-stage masks/query, transport requests/responses,
activation/KV state, library threads and idle inventory remain in `base_peak_bytes`.
Delivered buffers still live in caller scope must be part of that base for the
next window. Terminal speculation wastes at most one B per window; cumulative
waste is output/stop-dependent and is reported separately.

These are conservative **reference admission inputs**, not measured whole-client
RSS guarantees. Parent must validate thread-stack mapping/residency, allocator
overlap, traceback retention and Python/executor overhead on the target host and
increase reserves if needed. Host headroom/admission remain separate. Fixed Low
and Medium caps are respectively 268,435,456 and 2,147,483,648 bytes, qualifying
only by maximum whole-client peaks across matched fresh-process replicates.

Default timing fixture: `c=4,n=2048,m=4096`, B=98,304 bytes; incremental reserve
8,749,056 bytes plus measured process base and three existing tiny cursor owners.
CLI maximum: `c=32,n=m=19456`, B=4,980,736 bytes; incremental reserve 23,396,352
bytes. One executor thread; no provider thread/network/checkpoint. At most five
samples per mode (ten total), one expansion per sample. Fixtures initialize
cursors only, not full matrices. Timing is single-process native expansion;
its process peak cannot qualify tiers or compare whole-response RSS controls.

## Exact replay and parent timing gate

From repository root, installed imports (no checkout shadowing):

```sh
target/frontier-sdk-env/bin/python -I -m pytest -q --noconftest -o pythonpath= tests/test_mask_runahead_probe.py
```

Latest observed: 22 tests passed; exact final run is recorded in evidence.
Tests include numeric/mask/ticket identity; fused
boundary control; cap/one-slot/next-stage enforcement; barriers before and after
expansion on completion/error/cancellation; expansion failure; inventory cancel;
replay; concurrent consume; disjoint leases; online-guard/idle-refill ordering.

Parent may run this command **only under its existing frontier lock and host
admission**, after checking the resource allowance above:

```sh
target/frontier-sdk-env/bin/python -I scripts/probe_mask_runahead.py --timing --width 2048 --output-width 4096 --rows 4 --repeats 3
```

CLI help was checked; timing command deliberately not run by worker. It reports
native wall/CPU times, matching mask/ticket hashes and process peak. It compares
synchronous expansion with executor submission/consumption overhead, with **no
artificial network delay and no response-gain inference**.

## Missing integration gates

1. Bind dispatch/completion markers and next-stage selection to the compiled
   public prefill/decode schedule, including branches, local work, cache skipping,
   row extents and end-of-token stop. The current marker is caller-reported.
2. Add a research-only consumption adapter for `PreparedRemoteLinear` and bounded
   prefill chunks. Its current call takes the whole stage's row extent; this
   wrapper is not an SDK drop-in and does not route full responses.
3. Preserve the **fused direct owner as a separate control**. `NativePreparedLinear`
   requires exact lease type and `_prepare_native` feeds the cursor to
   `PreparedStageLayout.prepare`. That API has no already-expanded-mask ownership
   entry point. Prefetched rows cause a second claim to fail, as tested. No
   existing native fused-owner integration is claimed or weakened.
4. Retain original session cancellation/acknowledgment and online-activity guards
   while draining; exercise real transport failure, replay and task cancellation.
   Prove zero online Preparation/refill calls with complete-client audit, not only
   the local lifecycle-method control.
5. Parent-run exact full logits/KV/generated-token parity versus synchronous lazy
   and fused controls, matched prepared attempts/material, prefill and N−1 decode.
   No extra client weights/trust; at least 90% remote body MACs in each phase.
6. Parent-run complete-response TTFT/decode cohorts under admitted topology with
   whole-client peak replicas, current/future copies, thread stacks and early-stop
   waste. Keep client and Preparation CPU/RSS, cold/base/output-dependent cost and
   all-link traffic visible. Executor contention can erase any latency gain.
7. Verify installed extension provenance against inspected native sources; hashes
   identify the tested binary and matching installed Python ownership sources but
   are not a build attestation. No build or registration occurred here.
