# Cross-stage MLP conveyor: executable TCP ownership gate

## Decision

**Advance transport/ownership construction; do not promote to serving.**
`scripts/probe_mlp_conveyor_transport.py` now executes real framed loopback TCP,
not a scheduling simulation. Its dedicated client sender and receiver share one
connection. A serialized provider executes each complete request, then replies.
The conveyor can issue the next gate/up request before the preceding gate/up reply
retires. This permits request prepositioning; direct adjacent-gate write overlap
was **zero in both parent cohorts**, not a demonstrated mechanism benefit.
It establishes neither useful SiLU overlap nor a speedup. No SDK channel is called concurrently.

Production remains blocked on masked/native arithmetic integration, real
correlation/session owners, timing leakage, whole-client admission, and matched
complete-response evidence. No shared SDK/Rust/compiler code changed. No build,
install, checkpoint tensor load, sustained timing cohort or commit was performed
by this worker. Parent's original measured gate is reported separately below.

## Parent original cohort and v2 audit repairs

Parent ran original six guarded fresh workers at
`target/mlp-conveyor-transport-qwen150-288-v1.json`; file remains untouched
(SHA-256 `bd7c15d35f0653c7a6a9c7069737da8a50f373d7faf07b3ddca83ea6558a34ba`).
Read-only audit confirms all complete output bits and directed bodies match.

| Public rows | Serial region s | Conveyor region s | Full region s |
| --- | ---: | ---: | ---: |
| 150 | 2.714968 | 2.512251 | 2.540732 |
| 288 | 5.041836 | 4.781963 | 4.836080 |

This is **one fixed-order sample, not a speedup claim**. Original overlap field is
zero in all six cases. It sums only next-tile G upload-write intersections with
previous-tile G reply-write intervals, excluding G/D and other frame pairs and all
provider compute intervals. Zero does not exclude request prepositioning or
upload/compute overlap; neither zero nor differing times establishes causality.
Original JSON lacks guard/swap observations; existing guard use does not supply
missing measurements retroactively. Do not pool it with v2 instrumentation.

V2 renames this metric precisely to
`next_gate_upload_vs_previous_gate_reply_write_overlap_seconds`, with explicit
`overlap_scope`. Tests independently reconstruct its interval sum from events.
No new broad overlap category or packet-level overlap claim is introduced.

Audit also exposed a real cleanup bug: re-awaiting an already-failed kernel in
provider `finally` could skip lease/socket/handler retirement. V2 gathers kernel
exceptions and nests mandatory cleanup. It clears consumed exception tracebacks,
which otherwise can retain failed down-input product views through cycles. Actual
threaded kernel exceptions at G1, D0 and final D now verify zero terminal handlers,
writers, native jobs, tile slots and queued requests, closed leases and no
unhandled `client_connected_cb` exception. Pre-submission provider faults alone did
not test this path.

Measured v2 driver exclusively reserves an **incomplete output before admission**;
records bounded plan, source identity and captured build before execution; then
atomically checkpoints each accepted arm with fsync/rename. A later worker,
validation, admission, source/build or guard failure retains earlier accepted arms
and explicit failed status (bounded 4,096-character failure detail). Abrupt process
death leaves last durable incomplete checkpoint, not an absent all-or-nothing
report. CLI exits nonzero for a failed report. Existing outputs and foreign
reservations are rejected, never overwritten. This is fresh-attempt evidence,
not a retry/resume facility for consumed rows.

Acceptance now checks worker/launcher source and installed build identity, public
geometry, native compiler contract hash, wire binding, full oracle/output hashes,
cross-control output hash, measured per-direction header/body bytes, frame counts,
strict public send/ordered-reply trace, one-use conservation, owner bounds,
terminal retirement and shared100/40 pacing accounting. Guard snapshots record
initial/current host reserve/availability/swap, cumulative minimum availability,
maximum swap growth and guard error before/after arms and at close/failure. These
remain host observations, not per-party RSS or a replacement for admission.

`docs/evidence/mlp-conveyor-transport-tiny-screen-v2.json` contains **17 tiny real
TCP attempts**, including executing G1/D0 kernel failures and native-bound control.
New plus prior correctness probes: **79 passed**. Driver failure tests use real
tiny TCP arithmetic but mocked launch/admission/build metadata; they establish
checkpoint durability and rejection, not measured Qwen performance or real host
qualification. They cover later-arm preservation, unchanged existing/foreign
outputs, admission failure and rejection of altered source/build/hash/geometry/
wire/frame/compiler/pacing/order evidence. No new heavy cohort was run here.

## Completed six-worker v2 source-locked screen

The repaired driver completed six serial fresh processes under the shared frontier
lock and native 100/40 caps. All outputs agree with their complete synthetic oracle;
matched controls transmit identical bodies and envelopes. Every terminal owner
retires, and all recorded host guards observe **zero swap growth**.

| Public rows | Serial region s | Conveyor region s | Full region s | Serial / conveyor / full fixture peak MiB |
| --- | ---: | ---: | ---: | ---: |
| 150 | 2.653455 | 2.510891 | 2.550488 | 206.81 / 209.09 / 206.89 |
| 288 | 5.042897 | 4.779714 | 4.829978 | 241.34 / 245.03 / 237.66 |

Conveyor region times are 5.37%/5.22% below serial, but only 1.55%/1.04% below
full. Each is one fixed-order sample; **these differences establish no causal
speedup**. The precisely scoped adjacent-gate write-overlap metric remains zero.
The retained gate/product and frame owners satisfy their bounds, but total fixture
RSS rises versus both controls. These peaks contain provider weights and the
oracle, omit the real decoder and cannot qualify whole-client memory tiers.

All six cases conserve **86,138,832 application TCP bytes**. Reports remain separate:
[original v1](../evidence/mlp-conveyor-transport-qwen150-288-v1.json) and
[audited v2](../evidence/mlp-conveyor-transport-qwen150-288-v2.json).
The original's missing guard observations are not filled from v2. Next useful
step is a more informative public-order transport/compute schedule and an
order-balanced fixture gate before paying for masked SDK integration.

## Original v1 tiny evidence (retained)

`docs/evidence/mlp-conveyor-transport-tiny-screen.json` records 15 tiny real-TCP
attempts: three successful controls, three cancellation cuts, eight wire/provider
faults and one successful native-eligibility-bound conveyor. Shapes are 17 rows,
hidden8/intermediate13, tile7 (untiled control uses all 17 rows), chunk16. These
are generated public inputs and weights, not private activations or checkpoints.

* Serial and conveyor each deliver six request/reply pairs: 1,866 request bytes
  and 2,750 reply bytes, including each 73-byte fixed envelope. Untiled control
  uses four pairs and 1,720/2,604 bytes. Tiny tile7 differs from full control in
  frame overhead; this is disclosed, not a matched-frame timing comparison.
* All four successful attempts reproduce complete float32 output bits. Projection
  imports the original research `project` unchanged: row quantization, full-feature
  int64 dot products, then original float32 scaling. Tiny reference uses independent
  scalar Python integer sums. SiLU/product imports the same original float32
  expression in both arms, so transcendental implementation is not independently
  verified. No installed W8A8/prefix_f32 or checkpoint-output parity is implied.
* Serial/full peak one tile owner; conveyor peaks two. Dedicated wire regression
  observes G1 submission before G0 reply completion. All attempts remain within
  two outstanding frame credits and one queued request descriptor.
* Cancellation drains every submitted frame before connection close. Cuts consume
  14/21/24 stage-row tickets and burn 20/13/10, respectively, conserving all 34.
  Both peers maintain independent stage cursors. All canceled wire bodies reconcile.
* Wrong reply session/sequence/row, truncated reply, provider failure after claim,
  and wrong request session/sequence/row terminate without a partial output or retry.
  Failure reports separately charge bytes submitted, bytes read, and unread bytes
  discarded on close; failed attempts need not deliver all submitted bodies.
* Weak references prove actual gate/product arrays retire after terminal cleanup.
  Focused tests additionally cover singleton/partial tiles, five cancellation cuts,
  every header field, external task cancellation, native public 150/288 geometry,
  and a tiny shared100/40 pacer check. New plus prior probe tests: **61 passed**.

Tiny diagnostic wall/CPU/RSS fields are **not a timing cohort**: several cases run
in one process and `ru_maxrss` is cumulative. Do not compare their latency or RSS.
The original screen is retained, never overwritten by a replay.

## Protocol and owner boundaries

Public order is `G0,G1,D0,G2,D1,...,Dlast`, versus `G0,D0,G1,D1,...` for serial.
Untiled control runs one complete G/local/D with identical chunk16 framing. Every
header binds magic/version, request/reply kind, random attempt session, public
geometry/numeric contract digest, sequence, stage, tile, absolute row start/count,
width and exact byte length. Headers validate **before body allocation**; maximum
body is 16 MiB. Worker-mode frames additionally bind the native compiler contract
digest. Payload dtype/endianness is fixed little-endian float32.

Each peer monotonically claims ascending rows separately for G and D. Sequence
order is strict, not multiplexed/reordered. A terminal lease rejects replay and
burns every unclaimed reserved stage-row. Claimed work is consumed even if a reply
is lost. These are real fixture owner ledgers, **not cryptographic one-use material
or SDK prepared-inventory retirement**. No accepted attempt is retried.

One `asyncio.Queue(maxsize=1)` holds request views, not serialized stage lists.
One sender can hold another view while waiting on two frame credits; it constructs
its body only after obtaining a credit. Only receiver releases frame credits after
validated complete replies. At most two tile slots exist, including pending gates;
only D retirement frees a slot. Provider holds one executing kernel and waits for
its reply send before the next receive. Kernel work runs in a worker thread, not
another transport sender/receiver. Cancellation joins in-flight kernel work, clears
queued views/slots, closes sockets and burns both leases. A hard fault closes rather
than draining an invalid stream. External cancellation joins handlers and kernels;
public-cut cancellation deliberately drains accepted work first. Socket readers
and write buffers have bounded application thresholds; kernel socket ownership
remains separately unknown.

Both write-submitted and read-completed header/body bytes are counted for each
direction. Successful/drained attempts reconcile exactly against an independent
closed-form byte oracle. On faults, unread differences are reported, not silently
removed. Submitted TCP bytes are not packet capture: bytes actually emitted before
a reset, IP headers, ACKs and retransmissions remain unknown. Measured frame counts
mean provider-validated request frames and client-validated reply frames.

## Compiler, arithmetic and privacy scope

Installed `_native.decoder_mlp_tiling` admits generated one-layer Qwen-shaped
gate/up→SiLU→multiply→down regions at hidden896/intermediate4864 for public150/288,
tile64, and a tiny8/13 region. Exact returned source-plan/schedule/composition
digests and five operation IDs are retained. This establishes **eligible geometry
and descriptor binding only**. It does not turn NumPy research projections into
compiled SDK numeric execution or verify real checkpoint bindings. Untiled288
has no fictitious tile288 production selection: its eligibility witness uses legal
tile256, while its separately labeled transport control is one 288-row region.

All three body projections execute at the logical provider: successful fixture
body-MAC fraction is 1.0. The >=90% real SDK prefill/decode contract remains
**unknown**, not inherited. Original research integer/float32 arithmetic is unchanged;
feature reductions are never split. Full-size fixture reference uses complete
vectorized integer projections instead of the prohibitively slow Python scalar
oracle. It remains a synthetic arithmetic contract, not source logits/KV/sampler.

Descriptor order, geometry, pacing and cancellation cuts depend only on public
parameters, never activation values, sparsity or measured private work. **This
fixture is plaintext and unpadded; it must not receive real private activations.**
Public order alone does not hide readiness/CPU/backpressure timing. A deployment
needs public release slots with bounded failure behavior or an explicit approved
leakage contract, plus masking/corrections and native owner integration. No privacy
claim is made for this executable mechanics check.

## Accounting and parent-only measured gate

Each fresh worker retains original full input, full output and residual plus full
oracle output. Generated provider weights, original int64 cast/dot/quantization
scratch, both parties' wire/decoded buffers, two tile owners, Python/native state
and construction all affect measured **whole fixture process** high-water RSS.
`peak_retained_gate_product_bytes` prices only persistent gate/product arrays, not
nonlinear scratch or a total-memory bound. `whole_fixture_peak_rss_bytes` is process
high-water through cleanup/checksum, not separate client/provider RSS or post-JSON
process-exit telemetry. Two credits can still lose Low admission.

Region wall/process CPU include both logical parties and transport setup/cleanup,
but exclude fixture construction and output oracle. Provider kernel thread CPU is
also reported. Process CPU minus kernel thread CPU is a useful nonprovider residual,
**not certified client CPU**; foreign/native threads, transport/provider parsing
and scheduling attribution are not isolated. RSS includes construction/oracle;
region CPU does not. No Memory Tier qualification follows from mixed scopes.

SDK KV/state, attention/other outputs, real checkpoint artifacts, correlations,
prepared inventory, setup/renewal, preparation peaks, whole response/decode,
dashboard/relay and independent-party topology remain explicitly unknown. Original
full state is not pruned or credited as saved. No Low/Medium/Pareto result exists.

Replay tiny evidence or run correctness tests (no checkpoint loading):

```sh
target/frontier-sdk-env/bin/python -m pytest -o pythonpath='' \
  tests/test_mlp_conveyor_transport.py tests/test_mlp_conveyor_probe.py -q

target/frontier-sdk-env/bin/python scripts/probe_mlp_conveyor_transport.py \
  --source-lock docs/evidence/mlp-conveyor-transport-tiny-screen-v2.json \
  --output target/mlp-conveyor-transport-tiny-replay-v2.json
```

**Parent only; serial, fresh-process measured gate, no checkpoint weights:**

```sh
target/frontier-sdk-env/bin/python scripts/probe_mlp_conveyor_transport.py \
  --measure --rows 150 288 \
  --source-lock docs/evidence/mlp-conveyor-transport-tiny-screen-v2.json \
  --output target/mlp-conveyor-transport-qwen150-288-v2.json
```

Launcher uses `frontier_probe_support.fresh_workers`: exclusive frontier lock,
post-reserve **2 GiB** headroom allowance, continuous pressure/swap guard, stable
installed BuildRef and serial subprocesses. Sources must remain fixed throughout.
Original v1 screen/measurements keep their old source locks; updated code intentionally
rejects those locks. Preserve them and use fresh v2 names, never overwrite/relabel v1.
Do not invoke `--worker` directly. Existing `SharedPacer` applies shared per-party
100 Mbps download/40 Mbps upload budgets: both direct directions are constrained
by sender40, not independent100 pipes per tile. No WAN RTT/TLS is modeled.

Cost: **six processes**, one region/control each, no repetitions; 240-second worker
deadline each (24-minute maximum wait, excluding launcher overhead), 180-second
transport deadline. Each control evaluates its full oracle and candidate once:
about **34.36 billion generated integer MACs total**, not checkpoint inference.
Public150 uses 20 frame pairs and 3,457,460/6,375,860 directed bytes per control;
public288 uses 36 and 6,638,148/12,241,476. Three controls across both shapes send
**86,138,832 application TCP bytes** total. Both directions charge all envelopes.
No admission or wall-time prediction is inferred from these arithmetic costs.

At these shapes tile64/full share frame counts and body bytes. Report exact output
hash agreement, wire reconciliation, process CPU, kernel CPU, whole-fixture peaks,
tile/frame owner peaks, overlap intervals and all three controls. Observed async
write-call overlap is not packet-level concurrency or a speedup proof. A single
fixed-order observation is only a mechanism screen; replicated order-balanced
cohorts and separate-process attribution come later if worthwhile.

Next gates: real masked native stage owners and compiler numeric identity; public
timing/privacy contract; exact complete logits/KV/sampler/cancellation parity;
whole-client/topology admission including two slots and original lifecycle;
then matched source-locked 150/288+8 checkpoint cohorts preserving >=90% remote
MACs independently for prefill/decode. No whole-SDK parity/performance claim before
those gates pass.
