# Exact residue stripes over paired HTTP: feasibility gate

This is the frozen fixture contract. Completed parent-owned measurements and
their separate workload/cohort limits are recorded in
`docs/research/private-boundary-decode-followup.md`; they do not alter this
source-locked initial gate.

## Decision

**No-go for production promotion.** New research fixture exercises real loopback
HTTP/1.1 and installed native arithmetic/codec against the **installed parallel,
uniform-residue** transport and authenticated worker. No production files or
frozen evidence changed. Only tiny generated correctness runs were performed;
there is no new full-geometry speed claim.

Installed native `Matrix` exposes `shape` and `weight_bytes`; installed `Executor`
exposes full-matrix `wrap32`, not a borrowed row-range evaluator. Compiling stripe
views creates independent immutable snapshots. A sliced full native result is
not row-streamed compute. These are separate experimental arms:

| Arm | Execution | First possible body | Extra persistent weights |
| --- | --- | --- | --- |
| `parallel` | Unmodified `PrivateBoundaryTransport.linear`, parallel dispatch, installed authenticated worker, uniform codec | After complete native head and full encoding | None |
| `sliced` | Full installed `CompiledMatrix.wrap32`, then per-stripe installed residue encoding | After complete native head; potentially before full encoding | None |
| `snapshots` | Separate compiled stripe matrices, at most one next native call while delivering current stripe | After first stripe compute/encode | `V*H` bytes in this shared-process fixture |

Old ~149 ms parallel / ~129 ms stripe results used another TCP/u32 fixture and
cannot establish gain for this combination. Full-geometry timings remain parent
work, serialized against other expensive jobs.

## Inspected installed interfaces

- `python/pllm/runtime/private_boundary_transport.py`: distinct origins and
  credentials, explicit paired admission, current-step two-future executor for
  parallel heads. `_post` accumulates HTTP chunks and joins entire reply before
  validating outer frame. Cancellation signals closed state, then drains active
  exchange under operation lock before session/client retirement.
- `python/pllm/runtime/private_boundary_worker.py`: actual immutable weight,
  scale, tied-lookup and signed-bound validation; trusted admission callback;
  credentials bound as a pair. Sequence/head quota consumed before native work.
  Head response is a full `Response`, not `StreamingResponse`. Native cancellation
  retains busy capacity until task completion and burns slot.
- `python/pllm/runtime/native.py`: `CompiledMatrix` compiles immutable snapshots;
  `weight_view` is read-only alias, but `compile(weight_view[start:end])` still
  copies into a new native Matrix. `wrap32` serializes input and returns complete
  byte-backed ndarray. `crates/pllm-core/src/kernels.rs` confirms `Matrix::new`
  copies weights and `wrap32` allocates complete output before returning. Source
  inspection is not substituted for installed API discovery: report records both
  installed method lists, binary filename and SHA-256 (not private absolute paths).
- `python/pllm/runtime/private_boundary_head_codec.py`: same installed codec is
  used for every stripe, retaining original signed-i8/symmetric A4/A8 quantizer,
  full ring32 input shares, packing, reconstruction, padding and signed-bound
  checks. Public conservative width is `(2*H*128*127).bit_length()`; Qwen H=896
  yields 25 bits even under A4. No L1 certificate or client weight scan added.

## Research admission and binding

Fixture generates deterministic signed-i8 source, positive float32 row scales
and finite float32 activation, using original production quantizer. Installed
worker factory independently checks actual full native matrix/table identity.
Trusted fixture callback compares exact locally constructed descriptor and
one-lookup/one-head admission. This is generated-source admission, **not** a
checkpoint/source-lock or compiled decoder admission claim.

For all arms, real installed `transport.open` opens both actual worker sessions
with distinct bearer keys, peer credential digests, common session/epoch and
admission. Candidate route then uses a **separate research one-shot owner**, not
the inaccessible installed `_Session`. Full worker slots are cancelled via
installed transport during cleanup. The route has no public server entrypoint
outside the bounded local fixture.

Geometry is fixed before opening: full descriptor digest, V, public stripe rows,
ascending complete order, installed codec identity and one lookahead. Its hash is
checked against locally admitted geometry, not accepted as client-selected layout.
For each fixed range, codec-only derived descriptor keeps original source,
quantizer, ring and bounds; its composition commitment additionally binds original
geometry hash, index, start and row count. Derived admission binds that descriptor
and original plan digest. Thus installed codec frame binding includes complete
range identity plus paired context, role, epoch and operation sequence. Outer
installed head frames independently validate role/session/epoch/sequence/order.
Length prefix is bounded by exact expected framed size; incomplete, duplicate,
reordered or trailing stripes fail. Last stripe covers actual tail, without
vocabulary truncation. Final success requires complete integer oracle parity and
all float32 logits equal, not top-k or approximate equality.

These derived descriptors are codec research adapters, **not valid independent
source/table descriptors for production worker admission**. A production version
must add an explicitly versioned stripe codec binding, rather than reinterpret
descriptor composition fields this way.

Each role reserves one head/sequence-zero attempt before parsing its request;
malformed or oversized attempts burn that research owner. Successful and failed
attempts cannot replay. Credential failures do not reserve an unauthenticated
owner. On any pair failure, no partial logits return; both HTTP streams close,
all issued native calls drain, then both research and installed owners retire.
The streaming generator shields both AnyIO cancellation scope and native task:
an asyncio shield alone does not suffice under Starlette disconnect cancellation.
Cleanup also waits all tracked native tasks before returning. This demonstrates
local paired lifecycle, not an installed worker slot retrofit or malicious-worker
correctness proof. Hashes/checksums do not authenticate an adversarial worker;
remote deployment still requires authenticated HTTPS and trusted source admission.

## Ownership and accounting

Snapshot arm owns current output and at most one next reserved native output per
role, never one task per stripe. A native task may own result bytes before Python
resumes; reserved counts include that next result. Sliced arm retains complete
4V output until streaming ends. Counts are logical raw-output reservations, **not
RSS or all copies**: native temporary output/transpose/Python conversion, raw-u32
serialization, packed payload, codec envelope, outer msgpack, HTTP chunk framing,
socket/TLS buffers and allocator overhead are additional owners. At most current
and next native calls/results are admitted by application scheduling; no claim
that two buffers describe all framework/network allocations.

One fixture process shares full/stripe matrices and native executor between both
logical workers; it is not two independent worker processes or thin-client RSS.
Setup includes installed source verification, tied-lookup root/context binding,
HTTP servers/proxies and session admission, excluded from exchange time. Candidate
clients are constructed before timed exchange. Candidate quantization, random
sharing, complete reconstruction/dequantization and extra integer-oracle check
are timed. Common float32 assertion is timed in every arm. Installed baseline
uses synchronous httpx plus its paired thread executor; candidate uses async
httpx and incremental parsing. This is a path-level comparison, not a pure native
kernel experiment. Six repetition permutations balance three-arm order; longer
parent cohorts should use fresh processes and check CPU/RSS separately.

Transparent TCP relays pace actual HTTP payload using installed `SharedPacer`:
independent worker upload limits share client download capacity. Every directed
link counter includes actual HTTP headers and chunk framing, counted once, not
estimated payload sizes. Setup/online/all-phase totals remain distinguishable.
Per-role bodies separately report attempted request bytes, generated candidate
reply bytes and client-observed reply bytes; baseline exposes installed online
body counters and head operation counters. Replays are outside online snapshot,
inside all-phase network totals. Partial/failing online coverage is explicitly
incomplete; generated, delivered and observed bodies are not conflated. A closed
client can leave already admitted relay bytes unobserved. TCP/IP headers, ACKs,
retransmission, TLS, Internet latency and checkpoint distribution are absent.
This userspace token-bucket fixture starts pacing during setup and permits its
declared burst; it is not a physical line-rate/RTT emulator.

Timelines distinguish aggregate native/encoding time, first ASGI body, first
candidate client body and first jointly decoded stripe. `sliced` versus baseline
screens serializer/framing/read-ownership effects without earlier compute;
`snapshots` versus `sliced` can screen earlier native availability, but additional
calls/snapshots/cache effects and changed framing remain charged. ASGI send is
not physical delivery. Candidate `aiter_bytes(chunk_size=...)` may wait for the
next stripe to fill read buffer: run read-size ablations before attributing gain
to earlier start. No arm promises universal improvement.

## Verified tiny evidence

Initial suite: **22 passed**; Ruff checks passed. Tiny 17×8 / 6-row CLI smoke
completed all three arms with identical complete float32 logit SHA-256
`752f42d78a27f1e1723dae9bb5734c20a1b62b2b19828b21dc44b577e2ba219e`.
Earlier smoke used `.venv`, not the parent's SDK environment. Its native binary SHA-256:
`6b9370fba12ddd3df25979631d2ae9d37097d60bf0ae9bba1cd1bc2e8389a41e`.
Per role: baseline 171 request / 252 response body bytes; both stripe arms
171 request / 691 generated and observed response body bytes. Three native
calls per snapshot role, one per sliced role; peak logical raw output 48 bytes
for current+next snapshots versus full 68-byte sliced owner. All owners retired,
zero native tasks remained. These tiny numbers verify accounting and correctness,
not performance or full-geometry memory. Temporary smoke report path appears below;
no frozen evidence artifact was edited.

CLI hardening suite: **32 passed** using `target/frontier-sdk-env/bin/python`;
Ruff passed. Added rejection-before-allocation, exclusive output/race protection,
shared-lock contention, one-byte swap rejection, full-reserve protection,
mocked sampled-plan freezing and process-retirement-before-unlock checks.
Only tiny/model-free tests ran; no guarded full-geometry worker was launched.

## Commands (parent owns heavy runs)

Tiny correctness smoke, safe while other agents work:

```sh
target/frontier-sdk-env/bin/python -m pytest -q tests/test_boundary_head_residue_stripes.py
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py \
  --vocabulary 17 --hidden 8 --stripe-rows 6 --repeats 1 \
  --output target/residue-stripes-sdk-guarded-tiny.json
```

### Frozen sampled Qwen plan: six fresh workers

**Not executed here. Parent must wait until existing independent SDK cohorts
finish, then run all six workers serially.** Each CLI invocation holds shared
`~/.cache/pllm/frontier-probe.lock` fail-fast and launches exactly one fresh
measurement subprocess with the same interpreter. Three fresh workers per read
choice, not six repetitions sharing a worker across read choices. Fixed source
seed=17, generated Qwen V=151936/H=896, stripe rows=8192, A8/W8/ring32,
100/40 Mbps, installed parallel/uniform-residue baseline and both bounded stripe
arms. `--sampled-qwen` freezes these settings and six permutations per worker;
`--sampled-worker` labels independent workers 1..3. Client reads are 8192 or 1024.

Each worker uses this exact per-repeat order:

1. parallel, sliced, snapshots
2. parallel, snapshots, sliced
3. sliced, parallel, snapshots
4. sliced, snapshots, parallel
5. snapshots, parallel, sliced
6. snapshots, sliced, parallel

Each arm occurs twice in each position; each directed predecessor occurs twice
within repeats. Exact repeat and position accompany every retained run. Report
freezes pre-allocation public geometry hash, source-file hashes, installed native
hash and `BuildRef` lock before launching; verifies provenance again afterward.
Generated source/descriptor and codec geometry hashes are emitted after admitted
construction and retained in JSONL before timings. SDK interpreter build may
differ from earlier `.venv` smoke; do not transfer its build hash or timings.

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 1 --read-chunk-bytes 8192 --output target/residue-stripes-read8192-worker1.json
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 2 --read-chunk-bytes 8192 --output target/residue-stripes-read8192-worker2.json
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 3 --read-chunk-bytes 8192 --output target/residue-stripes-read8192-worker3.json
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 1 --read-chunk-bytes 1024 --output target/residue-stripes-read1024-worker1.json
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 2 --read-chunk-bytes 1024 --output target/residue-stripes-read1024-worker2.json
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_residue_stripes.py --sampled-qwen --sampled-worker 3 --read-chunk-bytes 1024 --output target/residue-stripes-read1024-worker3.json
```

All output paths must be fresh, including `.progress.json` and `.runs.jsonl`
sidecars. CLI exclusively reserves all three before acquiring shared lock; it
never overwrites prior evidence. Failures retain sanitized progress/failure code,
frozen plan when available, and already flushed per-arm JSONL. No private
exception text, credentials or activation values appear in progress/failure logs.
Running progress updates elapsed seconds; a rejected run remains rejected even
when some arms completed. Pure synthetic `run_case` test API needs no CLI lock.

Before metadata subprocesses or Fixture allocation, CLI samples existing native
host telemetry and requires allocation allowance plus OS reserve. Allowance is
`max(1 GiB, 6*V*H + 512 MiB)`; reserve is `max(4 GiB, host.reserve)` with existing
`host.reserve=max(1 GiB,total/8)`. Disk free must cover allowance. Headroom is
rechecked after metadata inspection, immediately before worker launch. Supervisor
samples every 100 ms while child owns work: retain full reserve, reject **any
observed new swap byte** relative to running low-water mark, not the generic
runner's 256 MiB grace. Check again after child and provenance verification.
Pressure/timeout/interruption terminates only owned worker, waits five seconds,
then kills if necessary, **before releasing shared lock**. This emergency
process retirement is distinct from normal `run_case` cancellation drainage;
no still-running native owner is reused. Sampling cannot prove absence of
sub-interval pressure spikes. Parent should still monitor RSS externally.

Keep build stable and record report hashes. Generated geometry is not real-model
confirmation. Outside frozen cohort, snapshot-free `--modes parallel sliced`
remains available and compiles no stripe snapshots.

### Resource estimates, not measured peaks

For V=151936, H=896: one int8 matrix is 136,134,656 bytes (~129.83 MiB).
All-three-arm fixture retains generated source + full native + stripe native:
408,403,968 bytes (~389.48 MiB), before oracles, scales and framework. Source/full
only retains ~259.66 MiB. Full native compile additionally serializes a matrix;
chunked i64 oracle uses up to `256*H*8` (~1.75 MiB) temporary weights. Budget at
least ~0.6 GiB actual process headroom and monitor actual peaks. CLI deliberately
requires larger conservative allowance (~1.26 GiB at Qwen geometry) **plus full
host reserve**, not merely 1 GiB free;
these are not Low/Medium client or worker tier certifications.

19 stripes: eighteen 8192-row stripes plus 4480-row tail. Each role sends 474800
packed residue payload bytes, identical to full 25-bit control; 19 codec headers
and outer frames/prefixes replace one header/frame. Header is 74 bytes. Exact
framed body/network totals come from report, not payload arithmetic. Per role,
current+next raw native outputs reserve at most 65536 bytes versus 607744 for
sliced/full native output. Full result remains client-owned: 1,215,488-byte i64
integer array plus 607,744-byte float32 logits, excluding codec/conversion copies.
With 8192-byte reads, client application reader retains one bounded frame plus
one read chunk per link. Fixture also retains full independent oracle. Installed
packing/decode owner formulas in `HeadOutputCodec` still apply per stripe.

## Production integration design and go/no-go contract

Model-neutral design, **not implemented**:

1. Version public delivery contract: source/descriptor/admission commitment,
   codec version/width, full V/H, fixed rows, exact tail and order. Bind range
   index/start/count/full V, paired context, role, epoch and operation sequence
   inside each codec frame/checksum. Client never chooses secret-dependent rows.
2. Add borrowed native `Executor.wrap32_rows(matrix, input, batch, start, rows)`
   (or equivalent immutable range handle), validating overflow/range/batch and
   returning only exact row range. Hold original immutable Matrix owner across
   jobs; do not recompile or copy weights. Reuse scalar/SIMD wrapping arithmetic
   and installed original quantizer. No persistent model-specific snapshots.
3. Installed worker reserves existing single paired head attempt before streaming,
   consumes sequence/budget once, keeps busy until stream and any next native job
   finish/drain. Schedule current plus one next maximum, account all packing and
   transport buffers separately. Cancellation cannot recycle native slot early.
4. Client owns two bounded streaming readers and current paired stripe, fills
   complete logits in ascending order, validates exact lengths/padding/bounds and
   final EOF. On malformed/replay/order/context/partial failure, burn both roles,
   drain issued work and never deliver partial logits. Preserve lookup/head shared
   sequence and teardown behavior. Charge each actual framed body/link separately.
5. **Go only after** installed-baseline parity under A4/A8 and signed-i8 extremes;
   full-tail logits exact; sequence/replay/auth/admission tests; cancellation while
   native and HTTP writes are blocked; two-slot ownership/RSS accounting; actual
   HTTP/TLS counters; stable-build generated and checkpoint paired measurements.
   Attribute gain with full-sliced/read-buffer controls, then real checkpoint
   logits, active KV, selection/history/RNG and complete-response CPU/RSS/traffic
   parity. Set explicit CPU/memory/latency acceptance thresholds before timings.

Actual sampled benefit against installed baseline may justify adding and
evaluating borrowed-row-range primitive; it does not by itself justify promoting
delivery protocol. No borrowed-row-range implementation is requested here.

**No-go** if native cannot borrow immutable row ranges without extra weight
owners, stripe/context framing is not admitted, owners recycle before native
drain, any tail/logit differs, partial failure produces output, or benefit exists
only against obsolete serial/u32 control. Snapshot-only timing may justify
further research, never silent production promotion.
