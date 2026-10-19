# PUMA P0: public-source delivery and lifecycle audit

Read-only source audit, 2026-10-19. Repository HEAD
`5dd110042aae58a886afb7ef1b8126257644479b`, with concurrent uncommitted PUMA work.
Line references describe inspected working files, not an immutable release.
No builds, model loads, checkpoint inspection, capacity/default changes, or
tracked feasibility-document edits performed for this slice.

## Decision

**Current public mode independently rebuilds public Q18 files at every worker on
every admitted request. It does not send all 494,032,768 weights over client-worker
links. It does not reuse a previously certified Q18 owner between requests.**

Source-present means original HF/Safetensors files are available locally. It does
not mean decoded Q18 artifacts have been delivered, certified, or cached. OS/HF
caches can reuse raw bytes, not native certification or request-owned Q18 files.
The path below is conditional on successful numerical/resource admission; this
audit does not claim any full-model request currently admits.

The revised draft's 641,966 lookup ordinals leave only 406,610 of a 1,048,576
ordinal allowance for everything else. That is not a complete-body admission or
deadline proof. Body nonlinear gates, online frames, import, and local I/O remain.
Do not substitute draft lookup timings into the identity-stable baseline report.

## 1. Actual owners and delivery path

1. `python/pllm/runtime/replicated_source.py:102-150::admit_source` resolves a
   directory or HF snapshot, admits native full-envelope geometry before tensor
   access, verifies source identities/metadata and source locks, and returns a
   **metadata-only** `ReplicatedSource`. Its base owner is documented in
   `delegated_source.py:519-536`. It contains no verified Q18 matrix cache.
2. `replicated_transport.py:24-25::execute_request` creates `_Peers`, then calls
   `source.publisher`. `replicated_source.py:83-95` creates a new native publisher
   and traverses every source shape. `_append_source:44-69` hashes the locked
   source before and after import; between those checks it reads original tensor
   ranges, converts F16/BF16/F32 to F32, and invokes native append in <=4,096-value
   chunks. Source F32 conversion is `delegated_source.py:723-730`.
3. `crates/pllm-compiler/src/replicated_decoder/source.rs:153-167,215-295` quantizes
   and hashes every value. Public mode retains full values only for RMSNorm gamma;
   other weights contribute commitments, extrema, and row certificates, not a
   resident matrix. `seal:300-350` compiles and binds a fresh session/program.
4. Client sends manifest/epoch/request admission envelopes **serially** to the
   three workers (`replicated_transport.py:29-45`). Manifest includes source
   descriptors and public gamma arrays, not the entire public weight payload.
5. Worker startup separately runs `admit_source` (`replicated_worker.py:168`).
   Each admission constructs a fresh native worker, compiles manifest, constructs
   `PublicSourceImport`, then calls `source.populate_public_worker`
   (`replicated_worker.py:460-508`; native binding
   `crates/pllm-python/src/replicated_transport.rs:280-365`). This independently
   repeats local checkpoint hashing, reads, F32 conversion and Q18 conversion.
6. `public_source.rs:76-183::PublicSourceImport` verifies ordered shapes, value
   bounds, public coefficients, row certificates and Q18 digests. Non-gamma
   tensors receive new `PublicFileWriter`s. Only completed verified files survive.
   Public gamma is already in manifest; it is checked but not written to files.
7. `source.rs:342-349::Publisher::seal` retains only input IDs for secret sharing
   in public mode. `next_chunk:362-414` makes three fresh replicated input-share
   frames, each 16 bytes per input token. `Program::inputs` filters out
   `PublicInput` (`program.rs:2221-2228`). **Weight sharing is private-mode behavior,
   not public-mode request traffic.**
8. `replicated_transport.py:94-110` closes publisher per request. Worker retirement
   closes native owner (`replicated_worker.py:109-132`). No worker source cache or
   public artifact registration/reuse endpoint appears on this path.

Local supervisor does not copy a checkpoint to three remote machines:
`servers.py:536-540,808-825,883-913` spawns local processes serially with same
Experiment and optional HF cache directory (`:654-668`). Raw files may therefore
be physically shared on that host. Independently deployed remote workers must
already have local files or independently resolve/download HF snapshots. HF
resolution uses `runtime/hf_hub.py::resolve_huggingface_source`; this is raw
checkpoint acquisition, not Q18 artifact publication. Network origin, cache-hit
state, sharding, exact dtype/metadata sizes and deployment distribution topology
remain unknown without the actual frozen deployment/checkpoint inventory.

## 2. Exact dimension-derived bytes; no invented Q18 wire protocol

Dimensions: 24 layers, hidden 896, intermediate 4,864, KV width 128, vocabulary
151,936, tied embedding/head. Common source pin:
`benchmarks/research/common.py:10-11`, Qwen/Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`.

| Category | Values | Source tensors |
|---|---:|---:|
| Embedding plus seven matrices per layer | 493,961,216 | 169 |
| Q/K/V biases | 27,648 | 72 |
| Two RMSNorm gammas per layer plus final gamma | 43,904 | 49 |
| Total | **494,032,768** | **290** |

Tied head is not another stored copy. Worker private Q18 files exclude gamma:
241 files, **3,951,910,912 bytes/worker**, **11,855,732,736 bytes/three workers**.
Full Q18 semantic equivalent including gamma is 3,952,262,144 bytes.
SDK's disk admission estimate conservatively includes gamma:
`sdk.rs:114-130` reports 2 * source_float32_bytes. Worker requires that estimate
plus 64 MiB free before construction (`replicated_worker.py:491-494`):
**4,019,371,008 free bytes**, independently checked at each serial admission.

Decimal Mbps; ideal payload-only delivery floors, excluding headers, ACKs, TLS,
TCP/IP, retries and verification. These are **conditional distribution scenarios,
not measured or charged current client-worker bulk traffic**:

| Hypothetical bytes actually distributed | Raw bytes | 100 Mbps | 40 Mbps |
|---|---:|---:|---:|
| Selected F16/BF16 checkpoint values, one destination | 988,065,536 | 79.045 s | 197.613 s |
| Selected F32 checkpoint values, one destination | 1,976,131,072 | 158.090 s | 395.226 s |
| All Q18 values, one destination | 3,952,262,144 | 316.181 s | 790.452 s |
| All Q18 values, three copies over shared sender uplink | 11,856,786,432 | 948.543 s | 2,371.357 s |
| Non-gamma private-file payload, one destination | 3,951,910,912 | 316.153 s | 790.382 s |

All supported checkpoint dtypes use >=2 bytes/value; actual checkpoint files also
include headers/metadata and possibly unused accepted tensors. Thus first row is
a known minimum payload, not an assertion that the pinned files are all F16/BF16.
Three independent parallel 100-Mbps destination links do not imply the shared
sender can deliver three copies in one-copy time; specify bottleneck explicitly.
Conversely, do not multiply per-link floors by three when destinations really
have independent sources/uplinks and concurrent acquisition.

**No existing Q18 network delivery frame exists here.** `append_public_source`
accepts <=16,384 bytes of **F32**, via local PyO3 calls, not a public HTTP upload
endpoint. Exact framed Q18 artifact size cannot be inferred from this API. Any
future artifact framing costs >=raw payload above; framing bytes remain unknown
until a real format exists. Calling these local F32 frames delivered Q18 bytes
would mislabel both encoding and transport.

### Actual request publication: exact known binary-body formulas

Let `B` be length of actual serialized JSON manifest. Its actual numeric gamma
values/certificates are unknown until source certification. The 49 arrays of 896
integers alone require >=87,857 JSON bytes, hence MessagePack manifest uses bin32.
`delegated_protocol.py:42-46::pack` uses MessagePack with binary types.

* `admit` upload = **B + 161 bytes per worker**, exactly: three-key map, request
  and epoch as 64-byte strings, manifest as bin32. Worker response
  `{"accepted": true}` = **11 bytes**. All three admit uploads+ACKs = **3B + 516**.
* Public mode's only `input` uses key `input.tokens`, zero offset, and 16r-byte
  binary share. MessagePack upload size is **16r + 32 + bin-prefix-bytes**.
  Prefix is 2 bytes for r=4, 3 for r=16/64. Native header reconstructed at receiver
  (`replicated_protocol.py:60-71`) is local and must not be counted again on wire.

| Input rows | Raw share/worker | Framed input upload/worker | Three input uploads+ACKs | Three admit+input uploads+ACKs |
|---|---:|---:|---:|---|
| 4 | 64 | 98 | 327 | 3B + 843 |
| 16 | 256 | 291 | 906 | 3B + 1,422 |
| 64 | 1,024 | 1,059 | 3,210 | 3B + 3,726 |

These are exact component totals, lower bounds on whole setup. Manifest GET
responses, setup controls/ACKs, public agreement/seed traffic and optional stream
handshakes are additional. Exact B and whole framed setup remain unknown; neither
is zero. Using only gamma-array minimum for B gives the following conservative
numeric floors on those framed components, not estimates of actual full setup:

| Workload | Known framed component floor | 100 Mbps | 40 Mbps |
|---|---:|---:|---:|
| 4+2 | 264,414 bytes | 0.02115312 s | 0.0528828 s |
| 16+2 | 264,993 bytes | 0.02119944 s | 0.0529986 s |
| 64+8 | 267,297 bytes | 0.02138376 s | 0.0534594 s |

Actual descriptor JSON and omitted controls make totals larger. These are shared
client-uplink byte floors; per-destination independent links need separate pricing.
For any known body count X, ideal floor = X/12,500,000 seconds at 100 Mbps
or X/5,000,000 at 40 Mbps. Bulk Q18 upload contribution is exactly zero in this
request path; gamma-containing manifests are nonzero public source publication.
HTTP/TLS/TCP bytes are outside these application-body formulas.

Peer online bodies are separate directed worker-worker traffic. Source delivery
bytes do not replace, reduce, or amortize nonlinear peer traffic. Existing
`cost::admitted` reports logical execution public-file reads/MACs, not import,
checkpoint distribution, or publication costs (`cost.rs:739-748`). Its peer
unknowns remain unknown. No report identity or timing fields changed here.

## 3. Local I/O and memory are not free

Each complete `_append_source` makes **120,717 append calls**: 120,596 matrix
chunks, 72 bias chunks, 49 gamma chunks. Client plus three workers make
**482,868 local native append calls per request** after metadata admission.
Each import bridges 1,976,131,072 F32 bytes and generates/hashes 3,952,262,144 Q18
bytes. Across four imports, F32 bridge volume is **7,904,524,288 bytes**.
Those are cumulative byte/call volumes, not resident matrices or network rounds.

Let S be complete locked checkpoint bytes, P selected tensor payload bytes, W
worker Q18-file payload. `_locked` invokes real content hashing, not an identity
memo (`delegated_source.py:170-177`; `model_loader.py:177-221::_source_lock`).
Each import reads **at least 2S + P** logically: pre-hash, selected conversion
read, post-hash. Worker also writes W. Minimum half-precision selected-only case:

* Worker import: **6,916,107,520 logical read+write bytes**.
* Client plus three worker imports: **23,712,519,168 logical read+write bytes**.
* Of that, selected/raw reads alone are >=11,856,786,432 bytes.

Initial `admit_source` adds at least two full source-lock passes per owner. Usual
local supervisor admits then discards a source before spawning; each worker and
fresh client also admit independently. Supplied/reused metadata changes that
initial count, not per-request `_append_source` behavior. Page-cache hits can
avoid physical disk reads; do not turn these logical byte counts into disk-time
claims without measurements. Filesystem bandwidth, SHA throughput, quantization,
PyO3 overhead, fsync/durability requirements and page-cache/RSS effects are unmeasured.

`crates/pllm-core/src/public_file.rs:58-114` creates anonymous private temp files,
writes sequentially and verifies complete SHA-256/size. `PublicFile` owns an
`Arc<File>` with bounded positional reads; no writable handle/path escapes. This
shares lifetime within native owners/readers, **not a cross-request source cache**.
No secure erasure is promised; closing references releases file ownership.
Execution repeatedly reads these files (`public_source.rs::multiply`,
`accumulate_rows`); those later reads are additional. Disk free-space check is
not an atomic multi-worker reservation; disk quotas, shared-host pressure and
storage contention remain deployment admission requirements.

## 4. Which clock starts where?

| Clock | Start / coverage | Code and limitations |
|---|---|---|
| Source resolution/import admission | No enclosing wall timeout shown | Client `_ensure_source`/tokenization precede `_execute` (`delegated_client.py:483-495`); worker `admit_source` runs before app is ready; supervisor pre-admission precedes process spawn. HF network library has its own behavior, not a demonstrated end-to-end source deadline. |
| Supervisor readiness, default 300 s | Each `_wait` begins after that role is spawned; roles spawn/wait serially | `servers.py:779-800,901-913,1144`. Not a graph-wide 300-s deadline; excludes supervisor's initial admission. Includes child's imports, source resolution and source admission before health. |
| Client absolute request timeout: ordinary SDK default **300 s**; internal direct-client default **60 s** | `_Peers.__init__` sets one absolute monotonic deadline **before publisher rescan**, then all role setup, online execute and finish calls share remaining time | `client.py:5603-5605::OpenAI` explicitly supplies 300 s if timeout omitted; `servers.py:100-109::role_client` uses this route. `ReplicatedClient` inherits `delegated_client.py:289,319` default 60 s only when constructed directly without override. Clock implementation: `delegated_transport.py:185,206-248`; `replicated_transport.py:24-25`. Neither is a per-stage allowance. |
| Admission body read, 60 s | Worker begins reading bounded admit HTTP body after admission reservation | `replicated_worker.py:472-474`. Only body read, not a total compile/import deadline. |
| Worker attempt, 1,200 s | `_Attempt` constructed **after native manifest.compile**, **before public source population** | `replicated_worker.py:81,495-504`; native constructor `replicated_transport.rs:288-296`. Setup/import/online share same deadline; no reset at execute. |
| Worker expiry watcher | Checks every 0.25 s after ASGI startup | `replicated_worker.py:348-352`. Schedules/drains retirement; not proof of hard process termination within 0.25 s of expiry. |
| HTTP peer RPC, 60 s | Each outgoing peer request's httpx operation timeout | `replicated_worker.py:343-347`. Separate from attempt lifetime; online coordinator has remaining-lifetime timeout (`:313`). |
| Peer streams | Call/receive constrained by remaining attempt lifetime; open timeout 60 s, close timeout 5 s | `replicated_peer.py:102-110`; receiver `replicated_worker.py:416`. Streams do not restart worker TTL or remove barrier/wave ceilings. |
| Client cancellation retirement | Separate shared 5-s cancel budget, then I/O draining | `delegated_transport.py:34-35,368-400`; I/O drain allowance 1 s at multiple drain/join sites `:127-169`. Not a single guaranteed 6-s complete cleanup deadline. |
| Native/worker cleanup | Must drain in-flight native work, streams and response leases before refund | `replicated_worker.py:37-58,93-132`. Cancelled native call is shielded/joined. No independent hard bounded CPU-drain timeout is shown. |
| Supervisor shutdown | Graph-wide SIGTERM grace 5 s, then SIGKILL wait 2 s | `servers.py:142-169`. Per escalation stage, not per role. Forced death differs from cooperative native cleanup. |

Important deadline gaps:

* Client publisher is synchronous. It receives cancellation callback, not
  `peers.check`; source hash passes do not poll it. Deadline is consumed, but
  exceeding configured 60/300 s during hashing/compile is detected only when
  transport next checks. Neither timeout is a hard upper bound on publisher
  CPU/I/O wall time.
* Native worker compile occurs before `_Attempt` exists: no worker TTL guards
  that compile. Client transport can time out, but server drains native work.
* Worker import polls cancellation/TTL between source chunks, not throughout
  pre/post full-source hash passes. Disconnect is checked after population
  (`replicated_worker.py:505`), not by a continuously running import disconnect
  watcher. Client timeout does not prove immediate stopped hashing or refunded
  native admission.
* Since admissions are serial, earliest worker's TTL includes its own population,
  later workers' compile/population, remaining setup/input, and full online body.
  A later worker's fresh TTL cannot extend earliest worker's lifetime.

Therefore neither a source-present deadline pass nor failure time is numerically
established without measured imports. Required same-source publisher plus three
worker population wall times must fit **one configured client budget (ordinary
SDK: 300 s; direct-client default or explicit timeout=60: 60 s)** along with
compile/setup/online, unless a separately approved lifecycle replaces it.
Network floors cannot be applied to local I/O as if disk ran at 40/100 Mbps.

Cold external checkpoint transfer, when needed, is >=79.045/197.613 s per
destination at 100/40. It already exceeds 60 s if placed inside that request
budget, but current worker startup/client pre-admission often place it outside
that particular clock. This 60-s conclusion does not apply to ordinary SDK's
300-s default. Delivering a complete Q18 copy within a 300-s request would fail
at both rates on raw payload alone (316.181/790.452 s), but current path does not
make that WAN transfer. A cold F32 payload at 40 Mbps exceeds default 300-s child
readiness on payload alone if downloaded during that child's readiness window.
Supervisor pre-admission downloads are outside its per-role readiness clock.

## 5. Smallest missing reusable-publication contract (proposal only)

Prefer immutable **verified source inventory**, not private-session reuse:

1. Cold explicit prepublication: admit exact pinned raw source and full envelope;
   run existing quantization, hashes, row-bound and public-gamma verification once;
   retain immutable public files plus certified descriptors at each worker.
   Give publication its own explicit bounded import/cleanup deadline and atomic
   memory/disk/source-count reservation. Record acquisition, verification and
   publication costs rather than hiding them as source-present initialization.
2. Cache identity must bind raw source lock, ordered source shapes/keys and Q18
   digests, quantization/numeric schema, certified ranges/row certificates and
   gamma. Existing `program::source_binding` is a useful descriptor binding, not
   by itself proof those descriptors came from pinned raw checkpoint bytes.
   Local cache must only mint entries from verified importer output; no arbitrary
   pathname, caller-declared digest, or unchecked serialized object can mint one.
3. Client needs its own reusable certified **source descriptor** so publisher
   can bind a fresh request/session without rereading original tensors. Receiving
   a worker's unaudited descriptor is a new trust assumption, not equivalent to
   today's trusted publisher computing source metadata. Preserve explicit trusted
   source authority and deterministic raw-source-to-Q18 certification.
4. Warm admit references source-inventory identity and worker epoch; check exact
   manifest descriptor binding before borrowing an `Arc`-owned immutable source.
   Continue fresh session IDs, input shares, PRSS/setup and result/program bindings.
   No reuse of mutable secret worker state or private streams/counters.
5. Worker cache holds separate ownership/reservation until eviction; per-request
   retirement drops only its borrow. Drain requests before final file release.
   Specify restart/epoch, invalidation, LRU/size limits, concurrent borrowers and
   failure behavior. `PublicFile` already supplies bounded immutable file ownership;
   current `NativeWorker::new` and Python admission always create a new importer.
6. If artifacts are transmitted instead of locally derived, define an actual
   bounded artifact frame format and authority chain first. Verify content and
   source/quantization provenance before inventory publication. No network
   compression ratio or trusted publisher signature infrastructure exists here
   to assume free. Shared-uplink three-copy Q18 floor is 948.543/2,371.357 s before
   framing; prepublication can move it outside worker attempt TTL, not erase cold
   end-to-end latency. Checkpoint-derived local prepublication avoids that Q18 WAN
   transfer only when original checkpoint was already distributed.

Small integration surface: `source.rs::Publisher` certified-descriptor constructor;
`public_source.rs::PublicSource` shared verified owner;
`crates/pllm-python/src/replicated_transport.rs::NativeWorker` verified-source
borrow; `replicated_source.py` certified descriptor owner;
`replicated_worker.py` source inventory and admission borrow;
`replicated_transport.py` inventory identity binding; supervisor/benchmark source
phase accounting. Existing SDK/benchmark extension hooks should report separate
cold acquisition, cold certification/publication, warm admission, setup and online
times/bytes/ownership. Do not create an alternate disposable benchmark runtime.

## 6. Evidence required before go

* Original defaults remain; no admission bypass. Source-present and cold/warm
  cases explicit; frozen wheel/full-model source admission only after authorization.
* Same ordinary SDK request under actual execution policy; certified source
  descriptors and source/codec/profile/epoch mismatches fail closed.
* Warm second request performs zero checkpoint value reads and zero Q18 rewrites
  **only after cache implementation proves that property**. Today's warm request
  does not. Separate process-restart cold behavior and raw HF-cache warmth.
* Import fault/cancel/timeout/disk-full/corruption tests prove files/leases drain
  before refund; slow pre/post hashing and pre-attempt compile covered explicitly.
* Exact application bytes measured separately from HTTP/WebSocket/TLS overhead;
  concurrent directed-link/shared-uplink assumptions documented; no ordinal-to-round
  conversion. Record acquisition traffic even if outside request's online window.
* Full body plus source lifecycle fits client, earliest worker and external
  benchmark deadlines. Draft lookup work alone cannot establish this. Existing
  independently priced legacy SiLU traffic remains a body blocker until its actual
  protocol is optimized and re-audited; caching cannot remove peer gates.

Reproduce numeric arithmetic without importing project or opening checkpoint:
`python3 -B target/puma-source-delivery-next/calculations.py`.
