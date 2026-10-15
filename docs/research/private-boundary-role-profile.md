# Private-boundary role-cost profile (opt-in research)

Status: six-client installed-role smoke and a separate six-client pinned Qwen2.5
150+8 cohort pass exact-state, admission, zero-swap and read-only replay checks.
Per-role native dispatch occupies about 13.3 ms; paired client exchange about
109 ms. This is instrumented attribution with charged overhead, not a serving
speed improvement or an isolated network measurement.

## Cohort and bootstrap

Identity: `pllm.private_boundary_role_profile.parallel_residues.v1`.
Exactly six fresh clients: current native coupled parallel/uniform-residue recipe,
matched control/probe, three balanced replicates, one thread, no warmup, one cold
150+8 response each (generated Qwen2 smoke is 128+2). No additional kernel arm.
Both arms use the research launcher; only probe enables timing wrappers. Launch,
import and wrapper overhead remain charged, never subtracted or repaired.
Probe-minus-control deltas include run noise; they do not isolate causal overhead.

The driver redirects only the ordinary isolated benchmark worker command. Inside
that fresh worker, an allowlisted exact standalone loopback boundary-role command
is redirected to the same script's private child entry point. Original role CLI,
credentials, experiment, supervisor, WAN relay, watchdog and descendant cleanup
remain responsible for execution. Inference/preparation commands are untouched.
The child wraps installed Python owners, then calls the original role CLI. No
production hook, copied numerical algorithm, response field, body/header buffering,
extra task, executor, queue or synchronization of the original native job is added.
Timers, bounded record locks and synchronous sidecar writes still perturb work.

Original arithmetic, returned objects/bytes, codec checks, sessions, replay burns,
cancellation shielding, randomness and dispatch order are not replaced. Native
dispatch wraps `CompiledMatrix._execute`, not a particular kernel operation name:
optional byte-dot implementations using that seam remain observable. Default suite
stays native control. A future kernel bypassing the seam fails owner-coverage audit;
it is not silently classified as measured native work.

The unchanged client profiler/auditor is loaded into a private module namespace,
specialized to this identity, one recipe and admission function. Its original module
globals and files are never edited. The driver freezes both new research sources
and all reused research helpers, captures installed build, and checks sources again
after execution. Raw results retain original source/numeric/privacy/state/workload
and output commitments. New measurement digest additionally binds complete client
and role telemetry; replay verifies binding, then reconstructs ordinary commitments
through the existing strict auditor. This unkeyed integrity binding is not a signature
or a claim of protection against an adversary rewriting all evidence and digests.

## Directly measured owners

Each role's head record has request-local monotonic intervals and CPU counters:

| Owner | Exact seam | Limits |
| --- | --- | --- |
| Native head dispatch | `CompiledMatrix._execute` | Includes native dispatch/call/return; excludes matrix input validation/serialization and output array conversion outside this seam. Not a hardware-kernel-only timer. |
| Reply codec | `HeadOutputCodec.encode` | Original residue encoding and checks; caller's `astype(...).tobytes()` precedes this owner. |
| Reply frame | worker's `pack_frame` alias | Original final response frame encoding only. |
| Server handling | pure ASGI wrapper around execute call | Begins after ASGI routing into app, includes request receive, validation, native wait, encoding and original ASGI send; ends when app returns. Not TCP socket-drain completion. |
| Handling unclassified | handling elapsed minus three serial owner intervals | Includes input/output conversion, validation, await/scheduling gaps, response construction/send and other work. No inferred cause. |

Exactly one native, codec and frame owner, in original order, is required per head.
Intervals must be contained in role-local handling and nonoverlapping. Local
exclusive bins reconcile exactly. Lookup frames are not emitted as head records.
Failures/cancellation yield incomplete or missing records and cannot pass replay.
A cancelled native job is neither joined nor recycled by instrumentation; original
worker owns retirement. Sidecars are collected only after original target cleanup.

Every owner has calling-thread CPU. Native-thread CPU excludes other native pool
threads. Handling separately records whole-role `process_time_ns` delta (all role
threads) and ASGI calling-thread CPU. These CPU counters overlap and are **not
summed**; role process CPU is not client CPU. CPU attributions cover handling, not
startup, teardown or full role lifetime; original report retains lifetime accounting.

## Pairing, clocks and honest residuals

No shared cross-process monotonic epoch is asserted or needed. Role timestamps
are relative to each individual handling call; absolute epochs, PIDs and session
identifiers are not retained. A/B sidecars have separate trusted owners, paired to
client head HTTP spans by chronological head ordinal. Current one-request, sequential
per-role session and fixed original scheduling make this public ordering sufficient.
Replay requires N calls for each role and exactly one nested A/B HTTP span in each
client paired-head exchange. No role timestamp is overlaid on a client timeline.

Client request and first-to-last-output windows retain original exact monotonic
anchors, clipping, exclusive partitions, required metric metadata and **N−1**
denominator. Role decode diagnostics identify whole head ordinals 1 through N−1,
excluding first-output head; they are not a clipped, aligned role-CPU timeline.
No role timing is averaged over N and relabelled decode cost.

`unaligned_http_minus_handling_ns` is a signed **difference of durations**, not
an interval partition or network measurement. HTTP and ASGI owners have different
endpoints, and their tails may cross: negative values are retained, not clamped.
This diagnostic can include transport, shared pacing, queueing, client work and
scope/scheduling effects. It is never called pure network time. Request-body
reception is already inside handling. No one-way delay or physical wire latency
is inferred. A/B run in parallel: never sum their spans or add role owners/CPU to
client pair elapsed. Client pair wall and HTTP unions remain authoritative, separate
nonadditive diagnostics. Residual share generation also remains unknown.

## Safety and admission

Both control and probe pay an additional 16 MiB conservative topology allowance:
8 MiB client/collection plus 4 MiB in each boundary provider peak. Aggregate native
peak increases by exactly 16 MiB. Labels alone are insufficient: replay requires
concrete positive integer `client_peak_bytes`, all four `provider_peak_bytes`, and
`native_total_peak_bytes` in the retained ordinary baseline. Baseline native total
must equal client plus all provider peaks. Final estimate must equal a complete,
independently reconstructed copy of that baseline with exactly +8 MiB client,
+4 MiB boundary A, +4 MiB boundary B, and +16 MiB native total for research, plus
the ordinary state audit's separate +32 MiB client and native total. Inference and
preparation peaks, Docker estimates and all other ordinary fields remain unchanged.
Missing numerical fields, decreased/extra charges, wrong individual providers,
changed aggregate, already-charged baselines and changed baseline metadata reject.

Baseline ownership and order are explicit:

* **Preflight:** this research module calls ordinary `metrics.benchmark_memory`
  once per case and retains its full estimate before any allowance. Existing
  `state_audit_admission` adds 32 MiB, then research pricing adds 16 MiB. Research
  pricing removes the known audit increment only from its retained baseline copy;
  it never adds the audit twice. The resulting baseline is compared exactly to
  the original pre-audit estimate captured independently by preflight.
* **Runtime:** inherited research bootstrap patches only
  `pllm.metrics.benchmark_memory`. Research pricing retains the original ordinary
  estimate and adds 16 MiB. Ordinary response target subsequently invokes
  `state_audit_admission`, adding 32 MiB exactly once. Estimate retains full
  `private_boundary_role_profile_ordinary_estimate` and explicit
  `private_boundary_role_profile_admission_order` through that deep-copy step.
* **Raw report cross-check:** dashboard separately calls the unpatched runtime
  `benchmark_memory` function. Its `report.memory_preflight.estimate` has neither
  research nor state-audit charges. Runtime completion and read-only replay require
  exact equality of this full raw estimate and retained ordinary baseline, not
  equality of another research component label. Both recorded native admission
  candidates must also require their respective native estimate plus exactly
  32 MiB for the existing WAN relay; this relay is outside the allocation estimate,
  distinct from the state audit. Candidates must be admitted, without reasons,
  within recorded positive admission budget capped at the suite's 8 GiB.

Full topology admission (including retained baseline, numerical estimates, order,
and candidate arithmetic) and raw ordinary dashboard admission are bound into the
research telemetry digest. Validation precedes digest normalization for the frozen
auditor. Tests also refresh the unkeyed binding on invalid numerical evidence to
verify that arithmetic/raw-baseline checks, not merely hash mismatch, reject it.
Read-only replay performs dictionary/integer comparisons and commitment hashing;
it calls no memory estimator, model/native planner, host probe or admission API.
It cannot independently prove a completely rewritten raw report authentic; the
integrity binding is not an authenticated external measurement signature.

Sidecars live in a private temporary directory, precreated exclusive mode 0600.
Open/collection rejects symlinks, nonregular files, other owners, hard links,
incorrect permissions and oversized files. Maximum 32 records per role, 4096 bytes
per record, 128 KiB per sidecar; each request retains at most eight spans. Collection
checks line/byte bounds before parsing, rejects duplicate fields, validates exact
allowlisted schema and requires complete N-owner coverage. Normal completed head
has three spans. Files are deleted after collection or failure, while validated
public timing records become part of ordinary raw benchmark details.

New telemetry contains only static owner labels, relative times, ordinal pairing,
CPU counts, completion booleans and public research metadata. No prompt, token ID,
share, mask, raw frame, credential, session ID or argument/return value is stored.
Exceptions are propagated, not serialized into role telemetry.

All original fresh-process build checks, full-logit/active-KV/position/selection/RNG
audits, output equality, shared 100/40 WAN caps and strict zero-swap rules remain.
Exclusive durable output-prefix reservation rejects existing/interrupted prefixes.
`~/.cache/pllm/frontier-probe.lock` remains fail-fast and nonblocking. No checkpoint
fixture is created before lock and prefix ownership. No driver build/install occurs.

## Commands and validation

Use installed frontier interpreter; `-o pythonpath=.` avoids stale checkout native
imports. Model-free tests, CLI help, installed-role smoke, native admission and
the complete pinned checkpoint cohort passed.

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. \
  tests/test_private_boundary_role_profile.py tests/test_private_boundary_decode_profile.py
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_roles.py --help

# Parent-only admission; fresh output prefix each invocation.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_roles.py \
  --artifacts target/prepared-frontier-artifacts \
  --output target/private-boundary-roles-preflight.json --preflight-only

# Parent-only generated real-role 128+2 smoke, six clients.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_roles.py \
  --output target/private-boundary-roles-tiny.json --tiny

# Parent-only real 150+8 matched cohort, six clients; no other model jobs.
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_roles.py \
  --artifacts target/prepared-frontier-artifacts \
  --output target/private-boundary-roles-qwen25.json
```

After each measured cohort, replay saved raw JSON and compare exact summary:

```sh
target/frontier-sdk-env/bin/python - <<'PY'
import json
from pathlib import Path
from benchmarks.research.private_boundary_role_profile import audit
p = Path('docs/evidence/private-boundary-roles-qwen25.json')
plan = json.loads(p.with_suffix('.plan.json').read_bytes())
actual = audit(json.loads(p.read_bytes()),
    source_identity=plan['research_source_identity'], tiny=plan['tiny'])
assert actual == json.loads(p.with_suffix('.summary.json').read_bytes())
print('Role profile summary replays exactly.')
PY
```

Actual child command compatibility, complete sidecar collection after supervisor
cleanup, state/body parity, numerical admission and all six completed clients pass.
The current wheel's target and raw dashboard ordinary estimates match exactly;
no fields were exempted and replay performs no planning. Missing owner/record,
changed arithmetic/build or component-only admission evidence still rejects.

## Pinned Qwen2.5 measurements

All **52,561,920** checked logit/KV values and complete state/output commitments
match across three control/probe pairs, with zero observed swap growth. Every
request reaches eight outputs; each probe has eight complete head records per
role. Complete client lifetime peaks range **190.45–220.67 MiB**, including relay,
supervisor and audit. Providers retain separate scope. These instrumented clients
do not establish a faster serving recipe or inherit another cohort's timings.

Values below are medians across three clients of each client's mean over whole
decode head ordinals 1–7. This avoids pooling heads or summing concurrent roles.

| Role-local owner / diagnostic, ms | Boundary A | Boundary B |
| --- | ---: | ---: |
| Native dispatch wall | 13.307 | 13.298 |
| Reply codec wall | 1.005 | 0.985 |
| Reply frame wall | 0.051 | 0.051 |
| Unclassified handling wall | 0.748 | 0.691 |
| Complete handler process CPU | 15.069 | 15.010 |
| Client HTTP wall | 108.075 | 108.369 |
| Unaligned HTTP minus handling duration | 92.951 | 93.322 |

Medians of separate columns need not sum exactly. On the client clock, exclusive
decode intervals average **365.037 ms** at the three-client median: body execution
excluding lookup/head **235.894 ms**, paired exchange **108.986 ms**, and token
lookup **12.548 ms**. Remaining admitted owners and residuals stay in raw summaries.
The role CPU and wall durations are not added to those client intervals.

The roughly 93 ms duration difference remains mixed transport, pacing, queueing,
client work and endpoint effects. Native compute is a small part of exchange wall;
these measurements do not establish that eliminating compute would save the same
duration, or rank a different transport/kernel under a causal model. Probe-minus-
control decode differences are −0.0118/−0.0232/−0.0049 tokens/s; overhead and run
noise remain charged rather than subtracted. No historical timing is revised.

Original bytes: `docs/evidence/private-boundary-roles-qwen25.json`, `.plan.json`,
`.summary.json`, `.progress.json`. The replay above reproduces the exact saved
summary, including numerical ordinary/research/audit/relay charges. Source/tokenizer
distribution, physical wire, Internet RTT and independent operators remain outside
the source-present local cohort.
