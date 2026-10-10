# Fast client planning from prior measurements

The client can compile a passed, output-matched network benchmark comparison into
`pllm.search.PlanningPrior`. The bounded table retains at most 32 assignment
samples and 32 KiB of canonical metadata. Ordinary configured SDK and gateway
requests reuse it through the `performance_prior` setting or
`PLLM_PERFORMANCE_PRIOR`; explicit planning accepts the same object.

Prior rows bind source, Pipeline, compiled workload envelope, runtime build,
client hardware/thread settings, installed host epochs and directed link
conditions. Matching complete coverage ranks median admission-to-first-output
time, then median decode milliseconds per token. Policy can reverse or disable
these preferences. Changed, expired, missing or incomplete evidence falls back
to geometry ranking. Forecasts describe the recorded token counts and public-cache
warm state within that envelope; they do not time a new private prompt.

The source comparison must preserve numeric, workload and output identity across
controls, complete its responses and retire its leases. Historical reports
without the performance bindings cannot populate a table. Evidence and forecasts
remain client-local; providers receive the existing exact execution contract.

## Immutable planning work

A frozen Rust `NetworkPlacementAdmission` validates the source/composition once
and retains only bounded role requirements and public digests. Its per-assignment
method still validates the fresh placement document. Python caches that owner and
serialized allocation/geometry facts under the model plan, Pipeline, workload,
allocation policy, thread cap and applicable state-evidence identities.

The process cache has a 64-entry and 2 MiB accounting bound. Each caller receives
fresh mutable geometry, then applies current process RSS, concurrent reservations,
headroom, resource policy, offers and link observations. No selected plan, capacity
reservation, lease, mask or model tensor enters this cache. First use still pays
compilation; a new process has an empty cache.

## Installed-wheel overhead screen

The public Qwen2.5-shaped 64-input/8-output fixture has 24 layers and 32 combinations
of client storage, codecs, full/tiled scheduling, CPU workers and inventory demand.
Its prior values are synthetic ranking fixtures, not serving measurements.

Six cache-empty decisions each precede eight warm decisions in one initialized
process. All 54 decisions evaluate all 32 candidates/assignments and select the
same Pipeline. The screen runs after the test/build jobs complete.

| Planning state | Decisions | Median wall time | Median process CPU |
| --- | ---: | ---: | ---: |
| Cache empty | 6 | 7,687.32 ms | 7,646.86 ms |
| Cache hit | 48 | 70.78 ms | 70.67 ms |

The warm decision is about **109× faster than the cache-empty decision** in this
isolated screen. The 32 retained entries charge 609,136 bytes. Request refresh,
native planning and decision construction are timed; source resolution, initial
lowering, discovery, leasing and model execution are excluded. These timings do
not establish cold-interpreter, end-to-end negotiation or response latency.

Evidence: `docs/evidence/research-planning-latency.json`, including samples,
coverage, environment and implementation commitments. Replay against the installed
wheel, using a new output path:

```sh
python scripts/probe_planning_latency.py --output planning-latency.json --pairs 6 --warm-decisions 8
```

`tests/test_planning_prior.py` checks cache isolation, fresh native validation,
resource rejection and stale/mismatched prior fallback. The network benchmark
test compiles actual generated-checkpoint measurements, loads the saved prior
through ordinary client configuration, completes an SDK response and checks lease
retirement and client-local evidence. The 13-module installed-wheel planning,
network, resource and gateway selection passes **180 tests**.
