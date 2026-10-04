# Metal placement, communication per token and load-time selection

## Scope

Pinned Qwen2.5-0.5B-Instruct, W8A8 `prefix_f32`, greedy sampling. One sample per
configuration of fresh 39+32, repeated 39+32 and extended 45+32 requests. Each
configuration uses one client and two co-located provider children. Checkpoint
files are already cached. Application bodies include setup once; full wire,
checkpoint download, client peak memory and GPU compute remain unmeasured.
Request times exclude role startup. All nine response output digests match their
same-workload peers; this is numeric-mode parity, not independent task quality.

## Measurements

| Across 96 generated outputs | Default prepared CPU | Combined CPU | Combined Metal |
| --- | ---: | ---: | ---: |
| Online application MB/output token | 3.648026 | 2.020272 | 2.020272 |
| Setup-inclusive application MB/output token | 9.939232 | 4.920438 | 4.920438 |
| Decode-only online MB/output token | 1.654720 | 1.348040 | 1.348040 |
| Cold first response MB/output token | 15.938047 | 9.768788 | 9.768785 |
| Total request seconds | 88.65 | 60.62 | 60.33 |
| Cold-first client CPU seconds | 15.61 | 15.99 | 18.18 |
| Cold-first aggregate CPU seconds | 85.61 | 61.17 | 56.98 |

Combined means client attention projections, completed-prefill reuse,
request-sized/on-demand inventory and zlib bundle delivery. Fresh online rates
are 3.548282 MB/output for baseline and 2.913158 for either combined option.
Combined repeat/extension rates are 1.305914/1.841745. Decode-only uses measured
post-first-output traffic and 31 outputs per request, not the response's total
32-output denominator. The combined sequence saves 44.6% online and 50.5%
setup-inclusive bodies against this control.

Metal now composes with client attention and prefix placement. Client GPU
weights are plan-bound, reused across revalidation, and exclude provider-owned
stages; provider GPU allocations exclude client-owned body stages. Small rows
and decode retain CPU execution. Tiny Qwen2/Qwen3 tests compare every prefill and
decode logit and KV value and observe actual client GPU dispatch.

Attention placement adds 44,040,192 bytes of client i8 weights, 196,608 bytes of
scales and 44,040,192 bytes of native CPU snapshots. Metal adds a further
44,040,192-byte GPU snapshot. These are storage categories, not peak RAM.
87.69% of body-linear MACs remain remote; this is not whole-response compute.

The CPU/Metal request-time difference is under 0.5% in this single sample. Their
online bytes agree; an 84-byte cold difference comes from serialized bundle
representation. No automatic Metal speedup follows. The canonical comparator
retains `matched_kernel_backend=false` and does not rank CPU and GPU together.

## Automatic selection

`search.optimization_space` proposes a bounded existing `SearchSpace`.
`compiler.plan_on_load` resolves once, checks the source-locked configuration,
lowers the public semantic graph and invokes ordinary native-admitted planning.
It preserves numeric/privacy contracts and returns the existing `PlanningResult`.
Weight ownership and cache expansion require explicit proposal permissions.
Resources must admit a miss; cache capacity is never priced as free.

The retained load probe exhaustively evaluates 48 choices in 5.90 seconds,
including source resolution, without tensor execution or role startup. It selects
CPU attention ownership under a 2 GiB declared client capacity and an 80%
body-linear remote-MAC floor. Its weight estimate is a conservative six bytes per
owned element. Unpriced delivery/inventory/cache changes do not displace the
incumbent's choices on a tie. A per-candidate graph index and validated schedule
are reused; independent native admission remains intact.

This is a cost-model selection, not a learned tuning table. Persist benchmark
evidence by source, numeric contract, input/decode shape, reuse horizon, hardware,
runtime implementation and network conditions. Do not enable every legal option
or benchmark every option on each load. Measured GPU/compression cost lookup is
still a follow-on; hard requirements with unknown costs fail closed.

## Compatibility audit

| Combination | Status / next contract |
| --- | --- |
| Metal + client attention/prefix | Implemented and measured here; separate CPU/GPU snapshots, CPU small-row fallback. |
| Artifact locality + zlib | Current delivery alternatives. Next useful composition: bounded compressed object delivery with raw content hashes/cache, measured CPU and cold bytes. |
| Prefix reuse + two-offset workers | Needs the same compiler-bound continuation schedule admitted by both authenticated workers, with replay/cancellation budgets. A whitelist edit is insufficient. |
| Prefix reuse + Freivalds | Needs verifier-aware cached-state lineage/admission; cannot silently skip integrity work. |
| Client prefix + selected linear roles | Current mutually exclusive ownership declarations. A reusable union selector must deduplicate stage ownership, storage and resource budgets. |
| Remote head + tied embedding | Moving the head does not remove the client token table. Private lookup is a separate cost-gated protocol. |
| Prepared + seeded two-offset transport | Alternative role/protocol graphs. Their reductions cannot be added as independent flags. |
| Metal + Linux Docker roles | The shipped container runtime is CPU-only. Metal requires supported Apple hosts. |

## Reproduction and records

From a repository checkout with the pinned checkpoint cached and the Metal extra
installed on Apple Silicon:

```sh
python scripts/probe_combined_runtime.py --candidate baseline --output docs/evidence/metal-placement-baseline-qwen25-2026-10-03.json
python scripts/probe_combined_runtime.py --candidate compressed --output docs/evidence/metal-placement-cpu-qwen25-2026-10-03.json
python scripts/probe_combined_runtime.py --candidate compressed_metal --output docs/evidence/metal-placement-metal-qwen25-2026-10-03.json
python scripts/probe_combined_runtime.py --summarise docs/evidence/metal-placement-baseline-qwen25-2026-10-03.json docs/evidence/metal-placement-cpu-qwen25-2026-10-03.json docs/evidence/metal-placement-metal-qwen25-2026-10-03.json --output docs/evidence/metal-placement-summary-qwen25-2026-10-03.json
python examples/benchmarks/automatic_planning.py --output /tmp/pllm-automatic-selection.json --evidence docs/evidence/load-time-optimization-qwen25-2026-10-03.json
```

Runtime configurations: `examples/benchmarks/combined_runtime.py`.
Source reports retain configuration/environment/source commitments and sanitized
cohort/output digests. `metal-placement-summary-qwen25-2026-10-03.json` contains
the matched comparison and exact numerator/denominator measurements.
