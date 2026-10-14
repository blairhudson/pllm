# Bounded cross-stage MLP conveyor: research rejection and next gate

## Verdict

The hypothesis remains plausible as a **native transport/scheduling experiment**,
but is not an implemented serving improvement. The current compiler-bound tiling
executor cannot run this conveyor: it synchronously completes each tile's gate/up,
local float32 SiLU/product and down projection before advancing. Existing duplex
prefill overlaps upload and download inside a stage, not between MLP stages.
No production SDK, compiler, runtime or admission contract is changed here.

The executable probe is `scripts/probe_mlp_conveyor.py`; its regressions are
`tests/test_mlp_conveyor_probe.py`. Evidence is
`docs/evidence/mlp-conveyor-synthetic-screen.json`. It contains deterministic
synthetic numeric checks and a non-calibrated scheduling simulation, **not measured
latency, installed SDK execution, checkpoint logits, KV parity or privacy proof**.
Exact existing-model results, at least 90% remote body MACs, whole-client Low
(256 MiB) and Medium (2 GiB) qualification, and a Pareto gain all remain unproved.
No literature-first novelty claim is made.

## Actual path and blockers

The source hashes in the evidence freeze these observations at the checkout:

* `python/pllm/runtime/semantic_executor.py:1282` selects the initial-prefill
  region; `runtime/mlp_tiling.py:33-69` allocates a complete hidden output and
  completes `_tile` synchronously. Its function scope deliberately retires each
  tile's intermediate before the next remote exchange. A two-slot conveyor breaks
  that particular lifetime guarantee and must replace, not bypass, its admission.
* `runtime/transformer_client.py:1324-1397` claims all stage rows, prepares all
  request chunks, waits for all inference payloads, decodes parts and concatenates
  the complete stage output. Sixteen-row duplex does not expose gate/up rows to
  local SiLU as replies arrive. Full tile outputs, request/reply lists, corrections,
  scales, and concatenation copies still have owners.
* `runtime/client.py:499-531` uses one sender thread and one ordered receiver,
  with at most 128 request frames and 16 MiB aggregate request/reply bodies per
  exchange. `client.py:4246-4267` assigns sequences for a single stage and validates
  ordered replies. Concurrent calls on that channel are not a multiplexing API;
  two arbitrary executor threads would race the receiver and shared sequence.
* `runtime/server.py:2235-2244` receives one frame, awaits its complete execution,
  then sends its reply before receiving another. `server.py:1399-1425` consumes
  prepared corrections before execution and constructs the reply. The ordinary
  `transformer_engine.py:1614` delegates a kernel to a thread, but this does not
  create concurrent kernels on that WebSocket. Global provider serialization
  across independent sessions is not inferred from this code.
* `transformer_client.py:154-167` advances each stage's row cursor under a lock;
  `:112-129` burns unused rows on close. Gate/up and down can have different
  cursors, but each must remain ascending. `server.py:1038-1057` also requires
  monotonically claimed response reservations. The proposed fixed order does not
  relax either rule. `runtime/protocol.py:95-113` permits bounded envelope sequence
  reordering, unlike strict peer sequencing in `shared_session.py`; neither is
  permission to retry consumed correlations. Normal channel retries after accepted
  frames are explicitly forbidden (`client.py:486`).

Compiler eligibility remains the existing four-step MLP contract in
`crates/pllm-compiler/src/mlp_tiling.rs`: matching grouped gate/up and down,
exclusive interiors, exact operation identities and common layer, no state roots
or external intermediate consumers, and unchanged semantic/numeric binding.
Decode, continuation, prefix-hit paths, terminal pruning, attention and retained
state are not pipelined. Unsupported verifier/Metal/nonlinear compositions cannot
be assumed eligible. Original full outputs and state are not removed by a
cross-stage scheduling change.

## Frozen candidate protocol and privacy gate

The probe freezes depth two, tile64 (tile16 as a separate control), chunk16,
384 public input-row ceiling, and 16 MiB frame and per-stage body ceilings. The
small correctness fixtures additionally exercise tile7. Its deterministic order is
`G0, G1, D0, G2, D1, ..., Dlast`, where `G` is grouped gate/up. Local product for
tile A can overlap remote gate/up for B; after D(A) retires, B's successor may
claim the freed slot. Down and gate/up kernels never overlap in this single-channel
model. Each stage claims disjoint ascending one-use rows, with fresh frame sequence
numbers. A stage identifier plus row, not the row alone, identifies a ticket.

Backpressure permits two live tile owners, one outgoing frame, one incoming frame,
and at most one complete request queued before provider execution. Inferred sender
progress never authorizes an extra tile. Failure/cancellation clears products,
burns every unused reserved stage-row, and closes the lease. Accepted work cannot
be replayed on another channel. A native version needs real attempt/session
retirement, not this toy lease's substitutes.

Scheduling order and geometry may depend only on admitted public shapes/contracts.
No activation, sparsity, output value, private checkpoint scan, mask, or private
measured execution time may select depth, tile size, route or next frame. **An
unpadded implementation that changes externally visible sending/readiness decisions
based on private local work is rejected here.** Fixed descriptor order alone is
not a timing-privacy proof. The model assumes public constant service durations;
it does not supply public wall-clock padding/deadlines. A native gate must establish
either public fixed release slots with safe bounded completion/failure behavior or
an explicit unchanged leakage contract before any serving claim. Ordinary transport
backpressure cannot be repurposed into a private adaptive scheduler.

## Numeric, scheduling and ownership oracles

Synthetic fixtures use 8 hidden and 13 intermediate features, signed integer
weights, independent scalar Python integer dot products for the complete reference,
and vector integer dot products for candidate tiles. Per-row quantization and
float32 SiLU/product evaluation stay fixed; complete output float32 bits compare
exactly. Rows 1, 7, 17, 150 and 288, tiles 7/64, and depths 1/2 cover singleton,
divisible and partial final tiles. These are not existing-model weights or the
installed W8A8/prefix_f32 implementation. Nonlinearity uses the same NumPy expression
in both arms, so this is not an independent transcendental/kernel validation.

Independent tests inspect event dependencies and resource non-overlap, reconstruct
every stage's contiguous row coverage, and check byte totals with a closed-form
oracle separate from frame construction. Slower public links and a five-second
local stall keep descriptors unchanged and owner credits bounded. Cancellation at
four internal cuts tests claimed-plus-burned conservation, zero retained products
and rejection of reuse. Live replay, skipped rows and reused sequence numbers are
also rejected. Weak references verify actual synthetic product retirement after
down and on cancellation. These tests do not certify transport cancellation or
retained KV in the SDK.

The ledger prices each slot's grouped gate/up float32 output, five intermediate
widths of nonlinear/product scratch, quantized activation, scales, mask, output
correction, decoded parts plus concatenation, down output, request/reply lists and
envelope copies. Shared owners include the original complete input, complete output,
residual and transport frame copies. Separate provider request queue, provider
kernel buffers/correction copies and reply are modeled resource owners, not client
RAM. A separate numeric provider ledger prices the queued and executing request
frames, decoded input/combined array, kernel output, correction parts/concatenation,
int64 modulo/cast scratch, prepared reply and reply/transport copies. Original
provider weights, inventory and runtime stay unknown; topology admission remains
required in a native implementation.
The client ledger intentionally sums several noncoincident peaks; it is an
allocation bound, **not RSS or a measured peak**. Conservative 32-bit request/reply
payloads and 1,024 bytes per frame are assumptions, not residue-codec measurements.

All remaining original owners must be supplied as a nonregion whole-client
baseline: KV/state, attention, other full outputs, prepared inventory and paged
corrections, imported weights/metadata, Python/native/allocator, dashboard, relay
and socket queues. They are unknown here, never zero. The probe's budget predicates
return null without that baseline. Historical RSS cannot be added to this ledger
without removing overlapping owners and respecting construction/request/lifetime
peaks. Prepaid +8 demand, natural EOS, full state, setup and decode remain charged
and unchanged; the scheduling simulation only prices an eligible initial-prefill
MLP region. It is not a 150/288+8 whole-response model.

## Modeled cost screen (not measurements)

Each party has shared 100 Mbps download / 40 Mbps upload capacity. On a direct
client/provider path both directions are limited by the sending party's 40 Mbps;
the probe does not assign a new independent 100 Mbps pipe to every tile. Other
party traffic would consume the same capacity. Internet RTT is zero in this model.
Provider compute is assumed 1e-11 seconds/MAC, local work 2e-5 seconds/row. Neither
coefficient is measured. Qwen2.5-0.5B geometry is represented by hidden896 /
intermediate4864; no checkpoint is loaded.

Provider receive/compute/send is serialized, with store-and-forward whole-frame
transmission. Existing duplex inside each tiled stage is included in both arms.
This is a deliberately simple illustrative event model: conservative absolute
durations do **not** make its relative gain a conservative bound. TCP streaming,
native float32 vectorization, provider queuing, serialization, padding and real
CPU contention can invalidate the predicted savings.

| Input / tile | Serial region s | Conveyor region s | Difference s | Serial priced MiB | Conveyor priced MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 150 / 16 | 1.996884 | 1.422705 | 0.574179 | 9.02 | 15.61 |
| 150 / 64 | 1.835740 | 1.573881 | 0.261859 | 28.79 | 55.15 |
| 288 / 16 | 3.833034 | 2.741661 | 1.091373 | 10.44 | 17.03 |
| 288 / 64 | 3.520455 | 2.990939 | 0.529517 | 30.21 | 56.57 |

Frame counts and arithmetic bytes are identical between the arms. The evidence
also screens zero and 1 ms/row local work. At zero local time, tile64 still predicts
0.258859/0.523757 seconds saved per region: most of this model's gain is hiding
cross-stage upload behind replies, **not** hiding SiLU. This immediately weakens
the original local-compute explanation. Multiplying by 23 regions is not a measured
TTFT estimate. No benefit is predicted for decode or exact prefix-cache hits.

The extra conservative tile64 slot costs 27,640,064 bytes (26.36 MiB). Neither
Low nor Medium is admitted on these subtotal figures. A two-slot implementation
could lose Low admission even while improving latency; Medium fit alone would
not establish a Pareto gain in CPU, traffic, Preparation memory or latency.

Historical measurements are context, not calibration: the matched combinations
reference reports tiled TTFT 42.64 versus 41.06 seconds at 150+8 and 79.85 versus
76.53 at 288+8. Earlier working-set confirmations have different controls/builds
and cannot be pooled. Existing sixteen-row duplex already delivered a separate
within-stage responsiveness improvement. Neither historical result establishes
the proposed conveyor's performance or exact existing-model parity.

## Replay and required native gate

Run from the repository root; this modest replay performs only bounded synthetic
checks and simulation. It does not import `pllm`, launch workers or claim installed
build identity. Evidence binds HEAD and exact relevant source SHA-256 hashes;
`--source-lock` refuses a changed source set. Use a fresh output name:

```sh
target/frontier-sdk-env/bin/python scripts/probe_mlp_conveyor.py \
  --source-lock docs/evidence/mlp-conveyor-synthetic-screen.json \
  --output target/mlp-conveyor-replay.json

target/frontier-sdk-env/bin/python -m pytest -o pythonpath='' \
  tests/test_mlp_conveyor_probe.py -q
```

There is no honest heavier conveyor serving command yet. The next gate requires
a private research native owner/channel implementation with one sender/receiver,
compiler-bound eligible region descriptors, complete source/numeric/execution
identity, two tile credits, bounded frame queues, fixed public leakage contract,
unchanged one-use claim/retirement and complete whole-client/topology admission.
It must preserve existing kernels and float32 evaluation, rather than splitting
feature reductions or replacing them with this NumPy model.

Then the parent may serialize installed-build generated Qwen2/Qwen3/Llama full
logit/KV, cancellation/replay and real transport controls, followed by matched
source-locked 150/288+8 checkpoint cohorts against current tile64 + duplex16.
Keep original full-output/state owners, prepaid rows and >=90% remote body MAC
checks independently for prefill/decode. Full vectorized controls are necessary:
smaller per-tile SiLU/quantization calls can lose CPU throughput. Include source
revision `7ae557604adf67be50417f59c2c2f167def9a775` for the existing Qwen2.5 artifact,
cold/repeat, all seven serving metrics and original complete lifecycle costs.
Fresh-process work must use `scripts/frontier_probe_support.py:fresh_workers`,
which supplies the exclusive cohort lock, post-reserve pressure/swap guard and
stable installed BuildRef; never bypass it. No checkpoint scans, timing cohorts,
builds, package installs or response benchmarks were run for this investigation.
