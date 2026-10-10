# Measured stage/phase kernel selection

This bounded native compiler reference selects CPU or Metal independently for
prefill and decode of existing semantic linear groups, under one joint GPU
weight/workspace budget. Source weights, group ownership, semantic dependencies,
numeric boundaries and local nonlinear/state operations remain fixed.

Profiles bind the model, compiled schedule, complete checkpoint binding, installed
build and exact public workload. Every group needs CPU/Metal prefill/decode costs;
missing, zero, duplicate, changed or overlarge evidence rejects. A Pareto dynamic
program chooses the minimum recorded full-workload kernel cost under the shared
budget, charging a GPU snapshot once when either phase selects it. CPU snapshots
remain resident. This optimum concerns supplied estimates, not future latency.

The compiler admits at most 512 groups and 4,096 nondominated states, with compact
Boolean decision histories. Plan/profile inputs and device allocations have
separate bounds. GPU pricing includes immutable i8 weights and conservative
input/output/scratch storage, summed across selected owners. Profiles exceeding
one bounded stage or the 2 GiB aggregate budget cannot allocate GPU snapshots.
The Python owner independently rechecks actual stage commitments and numeric
boundaries before allocation, admits only profiled row counts and retires on
malformed execution or failure.

This first vertical slice executes a complete local-clear compiled decoder. It
does not yet negotiate per-stage provider kernels, weight paging, tiling or
ownership choices through an ordinary Pipeline. Those interactions require their
own complete profiles, resource contracts and matched protected-response evidence.

## Pinned checkpoint gate

[`research-stage-kernels-r2.json`](../evidence/research-stage-kernels-r2.json)
records one profiler followed by twelve fresh decoder processes. Three alternating
groups compare CPU-only, a 64 MiB GPU budget, a 2 GiB GPU budget and the existing
fixed 64-row CPU/Metal dispatch rule. Each executes pinned Qwen2.5-0.5B-Instruct
at 150 input rows and eight selected outputs. All complete logit/KV digests
agree, with zero observed swap growth. Profiling independently checks 367,386,624
integer outputs and costs 20.543 elapsed / 40.710 CPU seconds.

| Median per decoder process | CPU only | 64 MiB selection | 2 GiB selection | Fixed 64-row rule |
| --- | ---: | ---: | ---: | ---: |
| Decoder elapsed, seconds | 4.073 | 4.025 | 3.631 | 3.502 |
| Prefill elapsed, seconds | 3.618 | 3.583 | 3.119 | 3.024 |
| Decoder CPU, seconds | 9.403 | 8.888 | 2.745 | 2.634 |
| Construction CPU, seconds | 2.546 | 2.673 | 3.013 | 2.227 |
| Process lifetime CPU, seconds | 21.658 | 21.275 | 15.317 | 14.688 |
| Process lifetime peak RSS, decimal MB | 1,110.29 | 1,156.15 | 1,485.88 | 1,487.09 |
| Additional GPU weight payload, decimal MB | 0 | 23.17 | 347.28 | 357.83 |
| GPU snapshot/workspace admission, decimal MB | 0 | 67.10 | 912.37 | 940.28 |

The larger learned selection cuts decoder elapsed time 10.86% versus CPU but
is 3.67% slower than the fixed rule. It selects Metal for 93 of 96 prefill groups
and eleven decode groups; the fixed rule selects all prefill groups and keeps
decode on CPU. CPU time excludes GPU compute. CPU snapshots remain resident,
and GPU payload/admission counts are not physical device-memory measurements.
The limited-budget choice trades 4.13% more RSS for only 1.19% lower sampled
elapsed time. These three-replicate samples establish exact dispatch and bounded
resource selection, not a speed improvement over the existing Metal policy or a
protected-response benefit. Automatic provider selection remains evidence-gated.

The earlier [`research-stage-kernels.json`](../evidence/research-stage-kernels.json)
contains the initial CPU/limited/full screen without the fixed-rule control.
Its 14.5% full-versus-CPU timing result remains separate; it cannot establish an
advantage over the pre-existing Metal policy. The second cohort reprofiles and
reruns every control rather than combining the two timing groups.

## Replay

```sh
uv run --no-sync pytest -q tests/test_stage_kernels.py
uv run --no-sync python scripts/probe_stage_kernels.py \
  --output target/stage-kernels.json --processes 3
```

A separate profiler measures transfer-inclusive integer kernels on public
synthetic inputs with pinned Qwen2.5 weights. Three alternating fresh-process
groups compare zero, 64 MiB and 2 GiB GPU budgets plus fixed-threshold Metal at
150 input rows plus eight selected outputs. All target logits and active KV must
match. Profile cost,
construction, whole-decoder execution, CPU, RSS and logical GPU weights remain
separate. GPU compute, physical device memory, provider transport and WAN latency
are not inferred from CPU process samples.
