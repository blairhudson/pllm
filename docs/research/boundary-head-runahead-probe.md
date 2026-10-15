# Input-independent random-head-share runahead

Status: bounded research evidence only. No runtime/compiler/manifest changes,
checkpoint execution, reviewed privacy claim, or measured serving-speedup claim.

## Result and scope

`scripts/probe_boundary_head_runahead.py` isolates one ahead B result. **Primary
schedule is admitted-current-step:** finish that token's embedding lookup, commit
to executing its transformer body, reserve its paired head sequence, and issue B
during that body. A joins the same reserved head sequence once q is ready. Do not
issue the next head immediately after the previous head. Normal EOS/cap termination
therefore needs no unused B call. For fresh
uniform ring32 `r`, B computes `W r` without knowing body output. After body output
is quantized to `q`, A computes `W(q-r)`. Recombine modulo 2^32, center once, then
use the existing float32 dequantization function/order on full vocabulary.
Never dequantize shares independently; never truncate vocabulary.

Installed-wheel native oracle agrees with int64 clear multiplication and bitwise
float32 logits. Tests cover zero/tied logits, negative/extreme codes, ring wrap,
greedy and temperature/top-p sampling, and identical sampler RNG states.
No transformer/KV executes here: full decoder state equivalence remains a gate.
This is head-result runahead, not previous mask-expansion runahead: retained object
is full B projection, not only input mask.

**Key negative result:** with independent equal A/B service times, one-ahead's
`max(T+a,b)` equals ready-parallel's `T+max(a,b)` for every nonnegative body
window. There is **zero modeled gain over ready-parallel**. Equal native heads do
not acquire a new critical-path saving merely because B starts during body.

Revised admitted-step smoke (257 x 32, three consumed heads, one repetition):

| Synthetic measured quantity | Ready serial | Ready parallel | One ahead |
|---|---:|---:|---:|
| Wall time including correctness oracle | 0.254 ms | 0.422 ms | 0.337 ms |
| Process CPU including oracle | 0.259 ms | 0.466 ms | 0.375 ms |
| B calls | 3 | 3 | 3 |
| Unused B calls | 0 | 0 | 0 |
| Research-envelope upload bytes | 1,776 | 1,776 | 1,776 |
| Research-envelope download bytes | 7,176 | 7,176 | 7,176 |

These are bounded smoke observations, not statistically meaningful benchmark
results. No body window sleeps or simulated windows contaminate measured timing.
One-ahead measurement starts B before current-row quantization, overlaps A where
possible, and drains owner on normal completion without issuing another head.
Ready-parallel uses an actual native owner with identical one-future cleanup and
arithmetic, but quantizes first and dispatches both worker paths only once q is
ready. Neither control executes embedding lookup or body; supplied synthetic row
stands for admitted current body's output. Tiny timing differences do not establish
benefit versus ready-parallel. Worker matrices are tiny synthetic
fixtures and co-located logical roles, not deployed independent services. Native
call times are reported separately; total wall includes oracle overhead. Thread
startup, Python scheduling and memory contention can dominate this scale.

## Authority and sequence inspection

Current `python/pllm/runtime/private_boundary_transport.py`:

* `linear`: quantizes first, generates fresh `right=r`, constructs `left=q-r`.
* `_exchange`: reserves one sequence and sends A then B serially; validates role,
  paired session, epoch, sequence, operation and row count before delivery.
* `_finish`: signals cancellation before acquiring operation lock/draining.

Current `python/pllm/runtime/private_boundary_worker.py`:

* Factory validates actual native head digest/scales, descriptor, role credentials,
  trusted admission callback and role-local lookup root.
* `execute` requires exact per-role next sequence and non-busy slot, charges budget
  and advances sequence **before** native work; exceptions burn session.
* Lookup and head share that sequence. A/B head calls consume matching counters.
  Native lookup children are context-bound and cannot be repurposed as B head state.

Candidate does not jump pending lookup frames. Prefill lookup finishes before first
body-window head; decode lookup finishes before each admitted body/head step.
**Paired sequence reservation can suffice:** reserve current head sequence once,
B consumes it during body, then A consumes that same sequence after body. The next
paired lookup waits until both complete. Independent stage lanes are an alternative,
not a proven requirement. Existing `_exchange` allocates and dispatches both roles
together, so reservation/join ownership still needs an explicit production gate;
merely wrapping unmodified `_exchange` in a thread is not this design.

Research `Binding` includes context (must represent session/epoch/admission), role,
source commitment, head stage, and head step. `NativeRole` checks exact next-step
binding before native work; used steps cannot replay. Its context and source are
trusted research inputs, **not authenticated wire authority**. Production must
derive them from validated descriptor/admission and authenticated session, not
accept caller strings as evidence. Research frames are length-prefixed canonical
JSON ownership envelopes plus payloads; byte ledger is not current HTTP traffic.

## Ownership, failure and boundedness

One owner has at most one submitted future, one mask, one retained B result.
Only B gets background task; caller invokes A. No unbounded queue or reusable
mask pool. `issue` while occupied rejects without replacing existing work.
Each issue calls OS entropy; no deterministic timing masks or secret seed shared
across roles. Compilation occurs before online timing; owner has no Preparation
API and calls none online. B work is caller-local online ownership, not offline
preparation credit. Logical role roots are separate compiled objects; harness
possesses fixture weights solely for native arithmetic oracle.

Consumption mismatch, substitution, malformed reply, either-worker failure, or
cancellation makes owner terminal, drains outstanding B task, wipes mutable mask,
A input and owned replies. No retry after partial success, no mask reassignment,
no refund. Replay cannot consume delivered head again. Cancellation signals before
drain; a late successful B completion cannot deliver logits. Normal EOS/cap follows
delivered current head and closes without new issue. Cancellation after body-step
admission but before head consumption charges unused B. Explicit `--unused-tail`
ablation admits an extra step then cancels it; this is not unavoidable EOS overhead.
Close is idempotent for resources.
All issued B work executes even if queued; this avoids hiding cancellation costs.

Drain is blocking and relies on finite callbacks. Python cannot forcibly stop
running native callbacks; production requires bounded native/HTTP deadlines and
request-level capacity. An unlimited hung callback is outside this research
contract. No zeroization guarantee for OS/Python immutable random bytes, installed
native immutable reply backing, intermediate frame copies, allocator pages, or
exception traceback retention. Owned mutable buffers are deterministically wiped;
secure native-buffer destruction remains a gate, not a claimed property.

Tests include cancel before consumption, clean normal termination, delayed stale completion,
cancellation during A, either-worker failure after peer issuance, substituted
context/source/stage/role/step, bad reply geometry, replay and destruction.

## Cost and scheduling model

At V=151936, H=896, each B input is 3,584 bytes; each result 607,744 bytes.
Persistent speculative state adds mask plus result: 611,328 payload bytes/request.
Owner array peak bound is 2,441,728 bytes across split/combine; excludes quantization,
public scales, frame copies, native scratch, oracle, allocator and RSS. Both worker
int8 matrices total 272,269,312 raw bytes if independently retained (not new client
weights). Native mutable-copy policy adds transient original immutable reply.
One unused B adds full matvec CPU, input/result frames, result download and drain.
Ledger charges successful result frames and attempted invocation input frames;
failed remote partial-body coverage is not modeled. Process CPU includes entropy,
oracle and scheduling; per-worker thread CPU and mask-generation CPU also reported
for one ahead. These overlapping CPU categories must not be summed.

Pure model uses explicit nonnegative body window T, A service a, B service b,
N consumed heads, U in {0,1} unused B (primary U=0):

* Ready serial: `N*(T+a+b)`.
* Ready parallel head transfers: `N*(T+max(a,b))`.
* Admitted current-step one ahead with A after body and B at body start:
  `N*max(T+a,b) + U*b` (conservative unhidden unused tail).
* Charged worker service: `N*a + (N+U)*b`.

Primary JSON models use explicit hypothetical service windows (a,b)=(10,10),
(5,15), (15,5) ms, crossed with chosen body windows. Repeat `--service-window-ms A B`
to override. A separate `modeled_native_equal_service` uses tiny mean measured
native-call wall for illustration; neither is a Qwen service estimate.
For U=0 the gain over ready-parallel is `N*min(T,max(b-a,0))`. Thus equal service
or A slower than B yields zero gain; only slower B admits modeled benefit. Example
T=10,a=5,b=15,N=8 gives ready-parallel 200 ms vs one-ahead 120 ms; equal a=b=10
gives 160 ms for both. Explicit unused-tail ablations are separate JSON arrays and
charge extra B CPU/frames/result/drain, never primary candidate overhead.
Windows are explicitly **modeled**, not measured
SDK benefits. Model assumes no cross-step pipelining, resource contention or
lookup/window startup costs; actual implementations must measure them.
Parallel-transfer gains and runahead gains overlap: compare against ready-parallel
baseline; do not add savings relative to same serial baseline. If `a >= b`,
parallel head already hides B behind A and runahead may have no critical-path
benefit, while unused work remains real cost.

## Reproduction and serial follow-up

Run from repository root using installed wheel; avoid pytest source shadowing:

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath= tests/test_boundary_head_runahead_probe.py
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_runahead.py --vocabulary 257 --hidden 32 --steps 3 --repeats 1 --body-window-ms 0 10 100
```

Later bounded timing, **serial after other workers stop benchmarks**:

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_runahead.py --vocabulary 4096 --hidden 896 --steps 8 --repeats 10 --body-window-ms 0 25 50 100 250 > boundary-head-runahead-4096.json
```

CLI deliberately caps synthetic V<=4096, H<=1024, steps<=16, repeats<=10;
full Qwen dimensions are resource arithmetic only. No checkpoint downloads.
JSON includes SHA256/path/size for probe source, installed native binary, installed
native wrapper and quantization source, plus fixture weight hash and NumPy version.
Hashing occurs outside measured online sections. These identify artifacts, not a
reviewed native build or privacy certification.

Optional tiny cancellation/unused-tail and asymmetric-model smoke:

```sh
target/frontier-sdk-env/bin/python scripts/probe_boundary_head_runahead.py --vocabulary 17 --hidden 8 --steps 2 --unused-tail --service-window-ms 10 10 --service-window-ms 5 15 --service-window-ms 15 5
```

Integration gates/sites: split `PrivateBoundaryTransport.linear` quantize/join
from issue; replace `_exchange` serial ownership only after worker sequence design;
extend worker `execute` admission/budget/cancel rules; trigger first reservation
after token lookup before transformer body in `MaskedTransformerClientRuntime`
prepare/decode paths; bind request response ownership and teardown at SDK boundary.
Require independent-role authentication/context tests, native one-use buffer API,
EOS/maximum-length and failure ledgers, actual frame/HTTP traffic/RSS/CPU, parallel
transfer baseline, unchanged full logits/KV/tokens/sampler RNG, and serial full-SDK
source-present/source-missing evidence. No new privacy or serving-speedup approval
is implied by passing this probe.
