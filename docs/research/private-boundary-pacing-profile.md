# Private-boundary HTTP and shared-pacer attribution

This bounded research measurement follows the remaining ~109 ms paired head
exchange in `private-boundary-role-profile.md`. It measures the installed
parallel/uniform-residue path under shared native 100/40 Mbps access caps.
Instrumentation changes the measurement identity; its times do not replace
untraced serving cohorts.

## Hypothesis and arithmetic

Qwen2.5's 151,936 vocabulary rows at 25 bits produce 474,800 packed bytes **per
boundary worker**, before its authenticated codec envelope and HTTP framing.
Each worker's 40 Mbps upload takes 94.96 ms for that payload without burst
credit; the 16 KiB burst can cover at most 3.2768 ms. The client's shared 100 Mbps
download has a separate 75.968 ms floor for both payloads before burst credit.
These are capacity calculations, not measured network attribution. The prior
~13 ms native dispatch per worker and ~109 ms paired exchange motivate directly
observing the original pacing decisions.

## Instrumentation and ownership

`benchmarks/research/private_boundary_pacing_profile.py` wraps, rather than
reimplements, installed `SharedPacer.reserve`. A module-local asyncio proxy
observes only sleeps inside owned head reservations. The original token buckets,
read size, burst, socket routing and admission code still execute. Each sleep
retains its requested delay, actual elapsed interval and both public token/rate
observations. Replay recomputes the limiting upload/download deficit. Actual sleep
elapsed time includes event-loop wakeup and scheduling; it is not a pure
bandwidth-delay measurement.

HTTPX's ordinary trace extension records only allowlisted phase names and elapsed
intervals: connection setup, request headers/body, response headers/body and
response closure. Callback information, headers, URLs, credentials, payloads,
tokens, seeds and sockets are never retained. Relay observations contain byte
counts and clocks only. Per-head application bodies reconcile against the SDK's
original audit counters; framed bytes, non-head boundary traffic and any
post-scenario retirement bytes stay explicit. All boundary reserve bytes must
reconcile with the original relay's final counters.

Client HTTP and relay events share the same process monotonic clock. Parallel
A/B intervals use unions, not additive latency. Request and decode windows are
inherited from the bounded client profiler; decode includes heads 1 through N−1.
The role profiler retains its own independent clocks and CPU scopes. HTTP time
outside observed reservations still mixes role execution, relay reads/writes,
client handling and scheduling. Subtracting a role duration cannot turn that
residual into measured network delay.

The trace admits at most eight heads per role, 2,048 chunks, 4,096 sleeps and
sixteen HTTP intervals per request. In addition to the existing 8 MiB client and
4 MiB per-boundary profiling owners, both control and probe price 32 MiB of client
trace/collection storage. Total profiling allowance is 48 MiB, separate from the
ordinary 32 MiB state audit and WAN relay allowance. Numerical charges must
reconcile against the raw dashboard's ordinary estimate. Overflow, cancellation,
failed requests or incomplete owners reject the measurement and restore wrappers.

## Cohort and replay

Six serial fresh clients use one private cohort: control/probe × three balanced
replicates of the identical ordinary SDK recipe. Every response retains complete
logit/KV/sampler commitments, material/body/TCP reconciliation, setup/cleanup,
all five roles' CPU, whole-worker RSS, native host admission and zero-swap guards.
The shared fail-fast frontier lock prevents concurrent managed heavy jobs.
Reports and sidecars use exclusive fresh paths; rejected attempts remain intact.

```sh
target/frontier-sdk-env/bin/python -m pytest -q -o pythonpath=. \
  tests/test_private_boundary_pacing_profile.py \
  tests/test_private_boundary_role_profile.py tests/test_native_wan.py

target/frontier-sdk-env/bin/python scripts/profile_private_boundary_pacing.py \
  --tiny --preflight-only --output target/private-boundary-pacing/preflight-tiny-v1.json
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_pacing.py \
  --tiny --output target/private-boundary-pacing/tiny-v2.json
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_pacing.py \
  --preflight-only --output target/private-boundary-pacing/preflight-150x8-v1.json
target/frontier-sdk-env/bin/python scripts/profile_private_boundary_pacing.py \
  --output target/private-boundary-pacing/qwen-150x8-v1.json

target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_pacing.py \
  --report target/private-boundary-pacing/qwen-150x8-v1.json \
  --plan target/private-boundary-pacing/qwen-150x8-v1.plan.json \
  --summary target/private-boundary-pacing/qwen-150x8-v1.summary.json
```

Paths shown identify the initial attempts; subsequent execution needs fresh paths.
The first tiny attempt completed all clients but its initial reader inverted the
nested `_post` timing relationship. Pacing's outer hook encloses the earlier
client-profiler hook; both are bounded by the same paired exchange. Corrected
read-only replay accepts those original intervals; the original rejection stays
retained and a fresh v2 smoke uses the corrected source.
Source locks additionally cover installed WAN, HTTPX and HTTPCore trace code.
`audit(document, source_identity=..., tiny=...)` is read-only and reconstructs the
original cohort gates, exact role and tracing admission, integrity bindings,
interval ownership, limiting deficits and counter conservation. No replay loads
a checkpoint or substitutes newly measured data.

Control/probe costs describe instrumentation effects in their own cohort.
Analytical bandwidth floors, isolated kernel costs, role-local CPU and traced
HTTP elapsed time remain distinct. This profiler selects no new serving method.

## Completed Qwen2.5 150+8 cohort

All six fresh clients complete eight outputs with **52,561,920 checked logit/KV
values** and identical complete state, selection, RNG and output commitments.
Observed host swap growth is zero and all six guards remain untripped. The largest
complete-worker peak is **224.78 MiB**, including SDK, supervisor, dashboard,
relay, audit and tracing; the idle launcher and provider peaks remain separate.
This instrumented resource observation does not replace serving-tier evidence.

The following are medians of three probe-client means over seven decode heads
per client. Role durations have independent clocks. Sleep categories identify
the limiting bucket at each original reservation decision; they include actual
timer wakeup, not only requested delay.

| Decode-head observation | Boundary A | Boundary B |
| --- | ---: | ---: |
| Reply application body | 475,016 bytes | 475,016 bytes |
| Reply TCP stream including HTTP framing | 475,153 bytes | 475,153 bytes |
| Native head dispatch, role-local wall | 13.564 ms | 13.606 ms |
| Reply codec, role-local wall | 0.972 ms | 0.989 ms |
| Client HTTP interval | 108.789 ms | 108.974 ms |
| HTTP response-header wait | 16.052 ms | 16.443 ms |
| HTTP response-body receive | 91.755 ms | 91.828 ms |
| Actual sleep, worker upload limiting | 90.194 ms | 89.759 ms |
| Actual sleep, client download limiting | 0.457 ms | 0.967 ms |
| HTTP outside relay reservations | 17.646 ms | 17.707 ms |
| Burst-adjusted worker-upload floor | 91.754 ms | 91.754 ms |

The paired head exchange is **109.736 ms**; the union of its overlapping A/B
pacing sleeps is **92.684 ms**, about **84.46%** of the exchange at the ratio of
those medians. Parallel sleeps must not be added. Every recorded head upload
fits burst credit without sleeping. Per-worker response transfer from first
reservation to final reservation takes **92.087/92.299 ms**, close to the
**91.754 ms** floor after granting the full 16 KiB burst. Individual column
medians need not add or subtract exactly.

The evidence identifies each boundary worker's **40 Mbps upload** as the dominant
observed limiting bucket. The client's 100 Mbps download limit contributes much
less sleep in this cohort. Requested sleep sums are about 79.6 ms per worker,
but actual sleeps are about 90.7 ms: wakeup delay replenishes credits before the
next reservation, so their difference is not independently removable overhead.
HTTP outside reservations still includes role work, socket read/write/drain,
client work and scheduling. This is local shared-rate emulation, not Internet
RTT or physical-wire attribution.

Instrumentation remains charged. Probe-minus-control decode differences are
−0.0094/−0.0174/−0.0047 tokens/s. Median cold-client CPU is 20.237/20.356 seconds
for control/probe and request latency is 82.160/82.172 seconds. These costs are
retained rather than used to correct other cohorts. No new serving-speed or
traffic improvement is established. Smaller exact reply payloads or overlap
would need fresh end-to-end evidence; kernel-only timings cannot remove this
observed upload requirement.

### Immutable evidence replay

`docs/evidence/private-boundary-pacing-qwen25.manifest.json` and the separate
`private-boundary-pacing-tiny.manifest.json` bind original report, plan and summary
archives by SHA-256. Gzip extraction reproduces each original file byte for byte.
The read-only reader checks those hashes, frozen source/suite identity, every
inherited state/resource/body/metric gate and reconstructed pacing summary. Its
additional raw-trace checks reject cross-head cumulative-counter reuse, counters
outside the original request window and sleeps shorter than their requested timer.
No saved summary can substitute for replay.

```sh
target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_pacing.py \
  --archive docs/evidence/private-boundary-pacing-qwen25.manifest.json
target/frontier-sdk-env/bin/python scripts/summarize_private_boundary_pacing.py \
  --archive docs/evidence/private-boundary-pacing-tiny.manifest.json
```

The plan retains all eighteen source hashes, including the installed native-WAN,
HTTPX and HTTPCore files. All matched the measured files at archival time. The
benchmark's ordinary installed-build locks remain additional evidence; this
replay does not require a checkpoint, native workload or host resource admission.
