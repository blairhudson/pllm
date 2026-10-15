# Research-only full-head duplex/stripe probe

## Scope and finding

`scripts/probe_boundary_head_duplex.py` compares one complete head operation using:

1. Serial party A request/reply, then party B request/reply.
2. Concurrent complete replies from both parties.
3. Concurrent workers, each evaluating and delivering ordered public output stripes.
4. Distinct `pipelined` arm: while the current completed stripe uploads, compute
   at most one next stripe per worker. Existing `striped` arm remains the control.

Production `PrivateBoundaryTransport._exchange` currently loops over parties;
the worker evaluates `head.wrap32` and materializes one complete framed reply.
This probe does not modify either component. Production code supplies exact native
arithmetic, quantization/dequantization, strict head frame packing/parsing, and
canonical `SharedPacer`. No checkpoint, SDK response, lookup, or decoder runs.

Generated signed W8 weights, A8 activation quantization, fresh ring32 additive
shares, native full and native row-stripe matrices reconstruct **every** integer
output and float32 logit exactly. Dense int64 arithmetic (256-row scratch) and
complete native matvec provide oracles. No early argmax, top-k compression, changed
sampling, or incomplete output is accepted. Selection/RNG and KV parity remain
integration gaps, not inferred successes.

The arithmetic-only ideal for two 607,744-byte replies at independent 40 Mbps
worker uploads is about 243 ms serial versus 122 ms concurrent, before overhead,
compute, bursts, and shared-client limits. These are hypotheses, not attributed
SDK timings. Client download remains one shared 100 Mbps bucket; concurrency
cannot turn that into two access links.

## Transport, boundedness and cost coverage

Actual loopback TCP transports carry requests and replies. Every application
stream chunk is reserved through the existing native `SharedPacer` before its
socket write; no invented sleep-based bandwidth emulator or analytic timing is
reported as measurement. This uses the canonical primitive directly, **not**
the HTTP `NativePartyNetwork` relay. Its scope label is overridden accordingly.
The primitive's generic 32-connection field is not this fixture's concurrency:
the fixture initiates exactly two connections and two handlers, one per worker.

Both directions share per-party upload/download buckets. All directed bodies
and framing are charged once: a four-byte request length, original production
MessagePack head frame, and each reply's 24-byte research stripe envelope plus
production reply frame. Public envelope fields bind magic, stripe index, row
start/count, vocabulary and body size. Inner frame checks role, session, epoch,
operation sequence and full row width. Production protocol does **not** implement
this stripe extension; no compatibility or authentication claim follows.

The `striped` control computes one native output, then finishes its send before
computing the next stripe. The `pipelined` arm submits one next native call before
awaiting the current stripe upload. It consumes that task before submitting any
further call. There is no per-stripe task queue: one pending native task per worker,
at most two active native calls globally, and two client tasks. Row geometry
and repeats are bounded; each exchange has a configurable deadline (default
30 seconds). Native cancellation cannot stop a running Rust native call: cleanup waits for an
already running Rust native call before retiring its owner. A wedged native call
still needs process isolation; the deadline is not hard Rust execution preemption.
Failure/cancellation during either current computation or current upload drains
the one pending task and closes the retained/current output's transport before
retiring the owner. The consumed current native task already finished; it cannot
remain orphaned. Tests hold next native execution during current upload and
inject both cancellation and upload failure to exercise that state.

Sockets use 16 KiB StreamReader limits, 16 KiB writer high-water marks and 8 KiB
paced writes. Asyncio read limits are flow-control thresholds, not strict process
allocation bounds; kernel socket buffers remain unmeasured. Worker output,
serialized body and envelope copies are charged to the reported process RSS;
the pipeline also reserves a second native output slot. Per-worker native output
storage is at most `min(2 * stripe_rows, vocabulary) * 4` bytes in `pipelined`,
versus `stripe_rows * 4` for the stripe control. Current serialization has a
conservative separate bound of `12 * stripe_rows + 2048 + 24` bytes for payload,
frame and envelope copies. These bounds exclude allocator metadata, compiled
matrix state and Rust/internal scratch; RSS includes them. JSON reports both
configured bounds and observed peak reserved output slots/bytes per owner.
Pending results may finish during upload but cannot induce further submissions.
complete delivery retains full client reply arrays in every arm. Reconstruction
and test oracles add scratch arrays. This is not a memory-minimal production
stripe design. The generated source and both full/striped compiled matrices stay
resident, so RSS cannot qualify the serving client.

JSON separates generated-source/oracle/compile setup, per-case share/listener
setup, and online connect/request/native/reply/reconstruction/drain costs. CPU
uses whole-process CPU, including worker threads. RSS is the process lifetime
high-water mark, with platform units stated; it is not a phase-specific peak.
Imports/process birth and checkpoint distribution are excluded. Setup network
bytes are zero because all source generation is local, **not** because a deployed
source is free. Body ledgers record attempted complete frames; canonical pacer
ledgers record admitted bytes. Failed runs never claim complete traffic coverage
or return a full-logit digest.

## Executed correctness and bounded smoke

Installed-wheel tests (avoid repository pytest `pythonpath` shadowing):

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_boundary_head_duplex_probe.py
```

Latest result after pipeline refinements: **38 passed in 0.90 seconds** against
the installed wheel. Only small correctness runs executed for this refinement.

Tests cover exact native integer/full-logit parity, final short stripe, serial and
parallel byte conservation, aggregate shared-client download accounting, real
paced sockets, wrong stripe order/context, malformed/oversized/truncated frames,
one-worker EOF, stalled-worker timeout, and cancellation during a held native
call. Cancellation tests prove owner retirement waits for native completion in
all four arms, including pipeline lookahead during a held current upload. Tests
also establish that the stripe control does not start next computation during
upload, pipeline/control directed byte counts match, output/task bounds hold,
and CLI execution order and native extension binary hash are recorded correctly.

Small CLI smoke, run once while other agents were active:

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_duplex.py \
  --vocabulary 4097 --hidden 8 --stripe-rows 512 \
  --download-mbps 1 --upload-mbps 2
```

Historical three-arm smoke before pipeline refinement (not rerun as performance
evidence), including reconstruction/checks and cleanup:

| Mode | Wall s | Process CPU s | All directed stream bytes |
| --- | ---: | ---: | ---: |
| serial | 0.266141 | 0.003777 | 33,454 |
| parallel | 0.265331 | 0.005766 | 33,454 |
| striped | 0.286141 | 0.015706 | 36,076 |

All three complete float32 logit digests were
`f70305dba75330befb5e0b59903fdfe84c43577840b27520878d54533ce50d26`.
This intentionally client-download-limited smoke validates transport/accounting;
it is not a statistically powered performance cohort. Small traffic overhead and
per-stripe thread scheduling can dominate. No stripe advantage is established.

## Deferred bounded measurement command

Run only after concurrent agents/benchmarks finish. This is generated Qwen-sized
geometry, **not** real Qwen source or an SDK request. It executes sixteen exchanges
serially (four repetitions of four arms), never overlapping case execution.
Each exchange has a 30-second network deadline, plus bounded native drain.
The four-arm Williams order balances positions and all directed predecessors
within four repetitions: `serial,parallel,pipelined,striped`, then
`parallel,striped,serial,pipelined`, then `striped,pipelined,parallel,serial`, then
`pipelined,serial,striped,parallel`. Incomplete four-repetition blocks are not
fully balanced. `--order-offset` selects the starting design row; parent may vary
it across serially launched fresh processes. JSON records full orders,
repetition/position per result, installed module paths, native extension **binary
SHA-256**, and helper/protocol/pacer hashes. Use a fresh output filename.

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_duplex.py \
  --vocabulary 151936 --hidden 896 --stripe-rows 8192 --repeats 4 --order-offset 0 \
  --download-mbps 100 --upload-mbps 40 --timeout-seconds 30 \
  > target/boundary-head-duplex-generated-151936x896-r4-offset0.json
```

Do not interpret within-process exchanges as independent fresh client benchmark
replicates. Parent can orchestrate fresh-process copies later with offsets 0..3,
but source generation remains a fixture rather than model distribution.
Full/stripe matrix duplication and source/oracles
mean this command is not a <=256 MiB client admission test.

## Gates before production conclusions

- Bind real locked source, compiled public geometry, authenticated role/session
  admission, one-use contexts, replay protection and consumed operation budgets.
- Specify a production stripe protocol and native owner lifetime; extend failure,
  cancellation, duplicate/reordered frames and teardown accounting tests there.
- Integrate coupled private lookup and ordinary decoder body. Audit every full
  logit, active KV, position, selected token and RNG commitment against incumbents.
- Measure matched fresh clients with canonical whole-process ownership, independent
  worker CPU/buffers, startup/setup/teardown, HTTP/TLS and complete directed streams.
- Independently remeasure full cold traffic and maximum complete-client <=256 MiB
  eligibility. Fixture RSS, warm head timings and hypothetical bandwidth floors
  establish none of these gates; no SOTA or new privacy claim is made.
