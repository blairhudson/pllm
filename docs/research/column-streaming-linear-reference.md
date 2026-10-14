# Authenticated within-stage column streaming: investigation 9

## Outcome and scope

Bounded executable client/provider reference exists in
`scripts/probe_column_streaming_linear.py`; independent controls in
`tests/test_column_streaming_linear_probe.py`. Installed Rust `MaskedGEMM.wrap32`
and native packed-residue codecs execute the numeric path. No SDK registration,
production protocol, native build, checkpoint scan, or deployment change.

This demonstrates **correct within-projection partial products before later
input blocks arrive**, not a latency, network, tier, or SOTA qualification.
Direct execution measures local complete projection work. Optional socketpair
mode measures actual bounded AF_UNIX stream transport, not TCP/TLS/WAN. WAN
overlap numbers are labeled models and omit important costs.

Small controls: 11 unittest methods passed, including a two-row local socketpair
correctness check (not a timing cohort). Direct synthetic 3x17 -> 7 projection
uses 5 native launches for block size 4 versus 1 for full-stage. Complete upload
frames total **789 vs 425 bytes**; final download is **264 bytes** for either.
Thus this tiny case adds 364 online bytes, with no traffic win; modeled latency
is upload/RTT dominated. Single warm/cold-order local durations are not evidence
of speedup. At three rows the ROW comparator degenerates to full-stage; admitted
parent command uses 32 rows, comparing two 16-row groups.

## Existing consumption and namespaces inspected

- `python/pllm/runtime/server.py::execute_prepared_payloads` validates admitted
  request encoding, stage root, wire width and reserved per-row correlations.
  It calls `consume_preloaded_batch` **before** running the native stage; adds
  correction once and emits a prepared row-residue response.
- `python/pllm/runtime/correction_rendezvous.py::consume_preloaded_batch` uses
  `(inventory session, 16-byte attempt ID)` keys, checks active authorization,
  exact correction metadata and duplicate IDs under lock, then burns entries
  before returning correction references. Failed batch metadata burns its keys.
  Session terminal/cancellation burns still-reserved material. A streaming
  implementation cannot call this once per column: that would either replay a
  burned ticket or incorrectly apply a complete-stage correction repeatedly.
- `python/pllm/runtime/residue_codec.py` permits only `offset` and `prepared`
  namespaces, bounds rows to 4096 and values to 4,000,000, and binds final
  response to ticket/stage/rows/wire bits/layout digest. Prototype keeps existing
  **prepared output codec**, with a separate research-only outer frame magic.
- `python/pllm/metrics/prepared_duplex.py` is existing ROW baseline: native
  kernels/codecs are measured; its full-duplex timeline is explicitly
  hypothetical. Production row chunks own distinct correlations. Column blocks
  instead share a single claimed projection bundle of distinct per-row tickets.
  This reference's row-group comparator uses the same research envelope, and
  serial group round trips. It is **not** production ROW WebSocket timing or a
  fair proof that columns beat the existing concurrent ROW baseline.

## Numeric and trust contract

For signed-i8 `W` and admitted activations in `[-127,127]`, Preparation supplies
client independent full-ring `r,s`; provider receives only packed `Wr-s`.
Authentication key is independent of masks and is not mask-expansion material.
No provider input mask, output mask, seed, or client weights are introduced.

Each column block carries `(x-r) mod 2^b`, with `b` in `{16,24,32}`. Provider
executes signed-weight `wrap32` partial products and accumulates in unsigned
32-bit output storage. Lower power-of-two rings are exact homomorphic
reductions of this sum. Final native `prepared_pack_output` adds preloaded
correction **once**, reducing each output to its committed public row-residue
width. Client `prepared_unpack_output` adds `s`, centers signed output, and
returns exact integers. Weight-derived absolute-sum widths must fit the wire
ring; no floating-point partial sums or unchecked signed int32 accumulation.

Oracle uses independent Python arbitrary-precision scalar sums, not a native
kernel or NumPy dot. Controls include uneven blocks, rings 16/24/32, signed
weight extrema, explicit maximal ring masks causing modular overflow, too-small
output rings, invalid activation `-128`, and mismatched geometry/layout.

## Lifecycle and authentication

Public contract hashes session, stage, distinct per-row 128-bit tickets,
geometry, block size, bits, weight digest and exact widths. Every full frame
contains 4-byte stream length, 8-byte domain magic, bounded MessagePack record
and 32-byte HMAC-SHA256. HMAC covers domain, contract digest, kind, sequence,
column start/end, and payload; received lengths are bounded before allocation.
Start and finish are separately authenticated. Contiguous fixed-public blocks
and monotonically increasing sequence prohibit reorder, overlap and duplicates.

START atomically claims the whole preloaded correction bundle in this single
owner reference. No ACK, accumulator or prefix result is returned. Only FINISH
after all expected columns can return one authenticated final prepared frame.
Wrong order, tamper, truncated payload, wrong session/stage/ticket/ring/layout,
early finish, EOF or explicit cancellation burns the entire bundle, clears
partial accumulator and drops correction. Client accepts one final frame once;
failed final authentication also closes it. Replay after final/cancel fails.
Client stream issuance is one-use; transport failure cancels both owners and
zeros client masks. Callers abandoning a direct stream must call cancel/EOF;
the socket reference enforces a ten-second I/O timeout.

TicketBook is deliberately bounded, volatile and single-owner, **not** durable,
rollback-safe, concurrent or an authorization substitute. Production would need
existing session reservation, timeout/cancellation, persistent/restart-safe
ownership where required, and its admitted transport authentication. Tests prove
local state-machine behavior, not security against a malicious provider; existing
verification/trust assumptions must remain unchanged. Public schedules leak no
new input-dependent geometry. All body MACs remain remote in this isolated stage;
whole-decoder >=90% prefill/decode placement gates are not evaluated.

## Resource and traffic accounting

Hard bounds: rows <=32, input/output width <=1024, repetitions <=8, frame <=8
MiB, block count <=1024, per-book tickets <=128 (actual <=32), one native compute
lane. CLI admission reserves 128 MiB runtime/imports, 64 KiB per packed matrix,
8 MiB thread stacks, payload/copy workspaces and all co-resident role arrays;
rejects estimate or observed macOS process peak above 256 MiB. These are
conservative research reserves, not certified allocator/stack bounds. Parent
must enforce available host headroom (recommend >=512 MiB) and frontier lock.

Client retains complete signed input plus full `r` and `s`, but creates masked
input and encoded request one column block at a time. Private provider
accumulator is `4*rows*outputs` bytes; transient partial result another equal
buffer; correction <= that size. Native packed column matrices total one W
copy, plus Python source W and per-matrix overhead. Creation repacks column
slices; native requests/outputs and Python codecs add copies. Repeated runs can
temporarily overlap old/new ownership, covered by runtime/headroom reserve, not
claimed eliminated. Socketpair adds bounded kernel socket queues and second
thread stack. Preparation native compile/correction CPU is separate; mask
generation/delivery, setup CPU and independent oracle are outside online timing.

Evidence records exact **complete online stream frame bytes**, including START,
all columns, FINISH, final result and outer authentication. There are zero
application ACK frames. Socketpair has no TCP ACKs; TCP/IP/TLS packet and ACK
costs therefore remain a missing transport gate, not silently zero WAN traffic.
Offline ledger separately charges actual packed correction and this reference's
raw `r,s` payload sizes; no offline transport frames are executed. Cold weights,
session authorization/reservation, offline framing and delivery must be charged
before any all-link claim. Synthetic raw masks are not SDK seeded-issuance
bandwidth results.

Report separates provider receive thread CPU, residual client/transport CPU,
Preparation correction CPU, combined-process peak RSS and resource estimates.
Client-only and Preparation fresh-process RSS, whole decoder TTFT and N-1
throughput are explicitly missing. Neither low (268,435,456 bytes) nor medium
(2,147,483,648 bytes) tier qualification follows from co-resident synthetic RSS.

## Timing interpretation and serial parent gate

Native partial-and-accumulate time is measured. WAN model serializes complete
uploaded frames on one access link, starts each compute after that chunk's
arrival, uses one compute lane, waits for authenticated final input, and permits
only a final download. `--contention` multiplies compute time by 1..16; this is a
sensitivity model, **not measured loaded-serving contention**. It excludes client
packing/authentication CPU, final correction/codec time, queue scheduling,
ACK/TLS, and shared socket/server contention. Thus modeled speedup cannot establish
measured transport or complete-response improvement. Small blocks increase
launches/auth frames/copies, can lose native efficiency and contend with concurrent
stages; column geometry is not automatically preferable to ROW duplex.

Evidence records whether first partial starts before **local sender** finishes
last input frame. Socket buffering can make that false for tiny inputs even
though provider processes earlier chunks before receiving later chunks. This
flag distinguishes actual local scheduling from the mechanism's permitted
overlap; sendall return is not a WAN remote-delivery timestamp.

Small local controls (allowed for worker):

```sh
target/frontier-sdk-env/bin/python tests/test_column_streaming_linear_probe.py
target/frontier-sdk-env/bin/python scripts/probe_column_streaming_linear.py --rows 3 --columns 17 --outputs 7 --block 4 --transport direct --output docs/evidence/column-streaming-linear-synthetic.json
```

**Parent only**, serialize under existing admission/frontier lock. No checkpoint
or sustained-network benchmark; this bounded local transport command has <=32
rows, 1024x1024 signed-i8 weights, one native thread, one provider I/O thread and
at most 4 repeats per each of three modes. Estimated co-resident reserve about
149 MiB, hard admission 256 MiB, recommended free headroom >=512 MiB:

```sh
target/frontier-sdk-env/bin/python scripts/probe_column_streaming_linear.py --rows 32 --columns 1024 --outputs 1024 --block 128 --bits 32 --repetitions 4 --transport socketpair --contention 1 --output docs/evidence/column-streaming-linear-parent-transport.json
```

Run fresh processes for matched RSS cohorts only when parent admits them. Keep
this AF_UNIX result distinct from modeled WAN sensitivity, production duplex
baseline, complete checkpoint response and loaded-serving tests. Evidence stores
source/test/note/inspected numeric-path hashes, installed native artifact hash,
environment, parameters and replay argv. Secrets and mask values are deliberately
not persisted; exact numeric fixture is deterministic, masks/tickets are fresh.
Evidence file's own SHA256 must be computed separately to avoid self-reference.

No commits, subagents, builds, sustained cohorts or shared source edits performed.
