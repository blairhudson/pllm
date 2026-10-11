# Bounded multi-inventory Preparation factory

Research probe only. No production serving activation, independent-operator
authentication, GPU-free compute or whole-response improvement claim.
CUDA is unavailable on this Apple Silicon host.

`scripts/probe_preparation_factory.py` combines fresh independently bound
Preparation requests against one immutable public stage snapshot. Native mask
cursors retain existing input/output domains. Each attempt has independent random
seed, session and attempt identity. Batches admit at most eight inventories and
stream at most 128 rows per matrix invocation, with a conservative 64 MiB row
workspace bound. Long inventories span windows; no full mask inventory is expanded.
Native integer, exact native digit8, existing exact Metal and adaptive 64-row
CPU/Metal modes have separate matched separate-job/batched-job controls.

Public deterministic synthetic i8 weights use offline pinned Qwen2.5-0.5B
configuration and compiler-derived QKV, gate/up and down stage geometry. This is
not checkpoint tensor execution or model quality evidence. CPU immutable weights
remain resident alongside GPU copies. Existing matrix APIs own arithmetic; Python
does bounded orchestration, subtraction and encoding, not a NumPy hot-loop GEMM.

HMAC-bound frames attribute each correction to original request, row offset and
row count. Private temporary correction files stay incomplete until all rows in
that admitted group finish. Replays, public-stage substitution, over-capacity,
kernel/receiver failures and cancellation burn every ticket owned by that factory.
Sealing cancels native mask cursor; factory closure retires private files. Python
burn means logical retirement, not cryptographic zeroization of all allocator
copies. In-process paired masks and receiver are explicitly test-local, not a
production party-separation protocol.

Independent signed-i64 oracle reconstructs test-local masks after captured service
and memory windows and checks every output. Reconstruction is correctness-only,
never another execution attempt. No seeds, masks, IDs, private output commitments
or correction contents are written to evidence.

## Costs and admission

Construction includes offline configuration/compiler work, synthetic weight
creation, immutable CPU/GPU copies, native cursor binding and private file creation.
First Metal invocation may compile shaders; that cost remains in service, not
silently warmed away. Service splits mask/copy, transfer-inclusive matrix and
subtraction, authenticated encoding/receiver decode/private file I/O. File flush
is not durable fsync. Output disk payload and encoded bytes remain separate from
provider or physical-wire traffic (unknown). Native digit8's pre-existing 224 MiB
conservative CPU workspace allowance fits inside 768 MiB process admission.

RSS is captured before oracle; oracle elapsed and lifetime CPU are separate.
Logical GPU weights and conservative row workspace, plus MLX allocator
active/peak/cache counters, are not physical device memory. GPU compute time is
unknown; process CPU excludes GPU work. No overlap, concurrent provider delivery
or pipelined GPU execution is inferred: existing Metal returns materialized output.

Public arrival schedules use 0, 1 or 10 ms spacing and a 5 ms batch deadline.
Fill waits and service-backlog waits are **modeled**, separately reported per
inventory; service itself is measured. The process does not sleep to simulate
live arrivals. Completion includes all group windows. Sparse deadline batches can
retain CPU fallback; threshold effects must not masquerade as batching gains.

## Two-stage execution

Initial implementation runs only small CPU functional/oracle tests:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python -m pytest -o pythonpath= -q tests/test_preparation_factory_probe.py
```

After explicit coordinator resume, run serial fresh-process cohorts under existing
exclusive measurement lock, host reserve admission and swap watchdog:

```sh
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_preparation_factory.py --processes 3 --output docs/evidence/research-preparation-factory.json
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_preparation_factory.py --processes 1 --rows 16 --arrival-interval-ms 1 --output docs/evidence/research-preparation-factory-arrival-1ms.json
/Users/blairhudson/workspace/pllm/target/frontier-wheel-env/bin/python scripts/probe_preparation_factory.py --processes 1 --rows 16 --arrival-interval-ms 10 --output docs/evidence/research-preparation-factory-arrival-10ms.json
```

No Cargo/maturin build required: installed wheel already exposes immutable native
matrices, digit8 and native row cursors; MLX is installed. GPU availability is not
tested in initial phase. Unsupported runtime cells preserve failures separately.
Whole-response seven-metric gate remains open.

`-o pythonpath=` prevents checkout Python sources shadowing installed extension.
Initial CPU checks cover all three rings, cross-inventory exact parity, row
streaming, replay, substituted bindings, capacity/workspace rejection, partial
receiver/kernel failure, cancellation and small-batch CPU fallback. Offline
compiler check resolves shapes `[out,in]` 1,152×896, 9,728×896 and 896×4,864;
configuration SHA-256 is
`18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45`.
Largest logical GPU weight is 8,716,288 bytes; largest conservative GPU row
workspace is 43,515,904 bytes. Main default cohort has 216 fresh children;
each arrival follow-up has 24. Estimated main duration 5–20 minutes, follow-ups
1–3 minutes each; estimates, not measurements. Per-child admission is 768 MiB.

Before entering cohort, freeze SHA-256 of probe, shared fresh-worker helper,
actual imported installed arithmetic/mask/codec and pressure/shutdown modules,
native extension, NumPy arithmetic extension and msgpack codec extension. Record
installed NumPy/msgpack/MLX versions and native capabilities beside BuildRef.
Both `mlx` core extension and split `mlx-metal` dylibs/shader library are hashed
without importing MLX core or initializing GPU. GPU workers additionally verify
the actual loaded MLX core path/content against this identity. Parent checks
identity before each cell and at cohort end; workers verify before/after execution.
Hashing uses bounded 1 MiB reads and occurs outside measured service/oracle.
Lifetime CPU includes identity validation; service and construction do not.

Pressure/watchdog or identity failure stops the flattened whole cohort immediately:
completed samples and failed cells remain separate, remaining cells are explicitly
unattempted, and report status is `interrupted`. No retry or remaining-cell launch
follows lost headroom. Modeled fill and busy-queue waits retain their own fields;
neither is folded into measured service time.

## Serial measured cohort

All **264 fresh children** finish: 216 main cells (three replicates per
shape/row/backend/layout), plus 24 cells each for 1 ms and 10 ms arrivals (one
replicate per cell). Independent oracles check **200,851,456 exact corrections**:
188,792,832 main and 6,029,312 per follow-up. Every completed inventory has exact
row attribution; no failures, blocked cells or unattempted cells occur. All three
cohorts observe zero swap growth and unchanged frozen runtime identities.

Installed build is
`d4c5bc12436c5354c1974b2731c5ccd45ef6e68f35c22023c5dd057e78b0a04e`;
measured probe SHA-256 is
`ce694ed66b95da5fb67da40d94fe17032b3eda7a4df58d0e5b5e9d9d20750008`.
NumPy 2.5.3, msgpack 1.2.2, MLX/MLX-Metal 0.32.2 and native API 2 / crate
0.1.0-alpha.2 accompany 19 frozen file identities in each report. Source,
compiler schedule, synthetic public weights and geometry match across controls.
Private materials are independently fresh per child, not identical reused seeds.
Three replicates are sample medians, not confidence intervals.

### Same-backend service comparisons

Times below are summed service for **four inventories**, each with indicated row
count, excluding construction, modeled waits and independent oracle. Full service
includes native mask expansion/copy, transfer-inclusive kernel/subtraction,
authenticated encoding, in-process receiver decode and private correction-file
write/flush. Positive change means less service time. Each median uses three
fresh processes; never pool distinct shapes into one performance ratio.

| Stage | Rows/inventory | Backend | Separate, s | Batched, s | Service reduction |
| --- | ---: | --- | ---: | ---: | ---: |
| QKV | 16 | Integer | 0.007779 | 0.008542 | −9.80% |
| QKV | 16 | Digit8 | 0.014453 | 0.012107 | 16.23% |
| QKV | 16 | Metal | 0.018900 | 0.017902 | 5.28% |
| Gate/up | 16 | Integer | 0.043066 | 0.062093 | −44.18% |
| Gate/up | 16 | Digit8 | 0.056789 | 0.050530 | 11.02% |
| Gate/up | 16 | Metal | 0.052714 | 0.056409 | −7.01% |
| Down | 16 | Integer | 0.027815 | 0.023755 | 14.60% |
| Down | 16 | Digit8 | 0.042467 | 0.025020 | 41.08% |
| Down | 16 | Metal | 0.027942 | 0.023748 | 15.01% |
| QKV | 150 | Integer | 0.079914 | 0.069407 | 13.15% |
| QKV | 150 | Digit8 | 0.075419 | 0.050198 | 33.44% |
| QKV | 150 | Metal | 0.069440 | 0.067534 | 2.74% |
| Gate/up | 150 | Integer | 0.577412 | 0.551024 | 4.57% |
| Gate/up | 150 | Digit8 | 0.439184 | 0.441597 | −0.55% |
| Gate/up | 150 | Metal | 0.435414 | 0.446808 | −2.62% |
| Down | 150 | Integer | 0.233833 | 0.234446 | −0.26% |
| Down | 150 | Digit8 | 0.204732 | 0.212930 | −4.00% |
| Down | 150 | Metal | 0.166902 | 0.157267 | **5.77%** |

**Strongest bounded GPU result:** down/150 Metal batching reduces median service
5.77%, process service CPU 0.095452 → 0.082742 seconds (13.32%), and eight matrix
calls to five. Each of three matched repeat pairs improves service:
0.159078/0.155490, 0.166938/0.157267 and 0.166902/0.157721 seconds.
Its batched service is also below native integer and digit8 controls. However,
kernel/subtraction phase itself rises 0.082205 → 0.085000 seconds; mask/copy
phase falls 0.067573 → 0.055048 seconds, while encoding falls
0.016775 → 0.015232 seconds. Phase medians are separately computed and need
not sum to service median. This is an orchestration/streaming benefit, **not
evidence of faster GPU GEMM throughput**.

Down/150 Metal construction medians are 0.391553 / 0.377777 seconds; paired
construction-plus-service medians are 0.550631 / 0.535499 seconds (2.75% lower).
These exclude fresh interpreter/import/identity-validation and oracle time, so
are not whole-process or response latency. Peak RSS rises 118.55 → 120.78 MB
(decimal). Logical GPU weights are 4,358,144 bytes in both layouts; conservative
GPU row workspace is 23,592,960 bytes. MLX peak allocator medians rise
7,307,264 → 7,766,016 bytes. These counters do not measure physical GPU memory.

Large CPU ratios require strongest-neighbor checks: down/16 digit8 gains 41.08%
against separate digit8, but batched integer is still faster (0.023755 versus
0.025020 seconds). QKV/150 digit8's 33.44% median reduction has substantial
variation: separate samples 0.044409/0.075419/0.077139 versus batch
0.047461/0.058888/0.050198; first pair regresses. Gate/up/16 integer batch also
varies widely (0.043888–0.083729 seconds) and regresses at median. No universal
winning backend or stable large factor follows.

At one row per inventory, batched integer remains substantially faster than
batched Metal: QKV 0.002629/0.010094, gate/up 0.005520/0.014635 and down
0.003813/0.013568 seconds. Adaptive dispatch correctly keeps these on CPU,
but still pays GPU snapshot construction. Combining four 16-row inventories
crosses the adaptive 64-row threshold and selects Metal; adaptive batch service
regresses versus its separate CPU policy on all three shapes (84.99%, 31.00%,
21.01% more time). A threshold change must not inherit same-backend batching gains.

### Admission delay and per-inventory completion

Batch capacity four seals a group only after all its streamed windows finish.
Even simultaneous arrivals therefore trade individual completion for aggregate
service. Down/150 Metal mean per-inventory completion rises from median
0.108198 to 0.157267 seconds despite lower total service. No TTFT benefit follows.

The 1 ms follow-up fills a four-inventory group at 3 ms: mean fill wait is
1.5 ms, reported separately from service. Down/16 Metal service is
0.028619 / 0.027728 seconds (3.11% lower), but modeled mean arrival-to-completion
is 0.020371 / 0.029228 seconds (43.48% higher). Separate mean busy-queue wait
is 0.013216 seconds; batch busy-queue wait is zero, distinct from fill wait.
Gate/up/16 Metal service improves 0.059657 → 0.041966 seconds in this one-sample
follow-up, but mean completion still rises 0.039730 → 0.043466 seconds.
Do not promote this follow-up over the three-replicate main gate/up regression.

At 10 ms spacing with 5 ms deadline, every batch is a singleton; all inventories
incur 5 ms fill wait. There is **no cross-inventory batching**. Down/16 Metal
modeled completion rises 0.010555 → 0.012965 seconds; adaptive remains on CPU.
Service differences here are independent-process variability, not batching gains.
All waiting/backlog/completion numbers are virtual schedule models built from
measured service, not live-provider arrival or latency measurements.

### Memory, encoding and remaining gates

Maximum captured pre-oracle process lifetime peak RSS across main cells is
180.57 MB. Full GPU gate/up snapshot remains 8,716,288 bytes with conservative
43,515,904-byte row workspace. At gate/up/150 Metal, MLX allocator peak rises
14,647,296 → 19,136,512 bytes (first replicate); end cache rises
6,389,760 → 14,172,160 bytes. CPU immutable snapshots remain resident. Oracle
memory is excluded from those captured process peaks but remains subject to host
guard while executing.

Gate/up/150 always writes 23,347,200 private correction-payload bytes. Separate
and batched layouts encode 23,348,052 / 23,349,320 frame bytes respectively:
more per-ticket windows add headers despite fewer matrix calls. Kernel binding,
mask generation, public synthetic weight construction, copies, authenticated
encoding and file writes are charged within their recorded scopes. This is
local disk/in-process delivery, not TCP/TLS/WAN or reusable inventory transport.
Durable filesystem sync, checkpoint weight loading/distribution, separate peer
delivery, physical wire, physical GPU memory and GPU execution time remain
unknown. Whole-response seven metrics are explicitly null. Shared first-use
shader caches are not evicted; cold filesystem/compiler-cache cost is not isolated.
Total cohort wall duration was not separately instrumented; initial time estimates
must not be mistaken for measured execution totals.

**Decision:** bounded factory preserves exact fresh-ticket corrections and has
a modest repeat-consistent down/150 Metal service benefit, with higher RSS and
worse group-completion latency. Small inventories prefer CPU; adaptive threshold
batching does not beat its separate CPU controls. No production activation,
100× claim, GPU-free-compute claim or whole-response benefit is justified.

Evidence:
- `docs/evidence/research-preparation-factory.json`
- `docs/evidence/research-preparation-factory-arrival-1ms.json`
- `docs/evidence/research-preparation-factory-arrival-10ms.json`
