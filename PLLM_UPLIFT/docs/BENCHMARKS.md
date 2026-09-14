# Benchmark, search and evidence specification

## 1. Two tools, one record format

Production `pllm bench` runs complete experiments; `pllm assure` runs claim-driven adversaries and formal/instrumented checks against the same locked plan. Search accepts only candidates permitted by a frozen privacy/numeric policy. This staging package implements a small native kernel benchmark source and executable specification/negative-control checks; it does not launch a private LLM fleet yet.

One public configuration is resolved into role-specific plans and credentials. Users do not edit three copies. Launchers: `local_processes` first, then preinstalled remote agents over SSH control, followed by container/orchestrator adapters. Runtime data plane and clocks remain in Rust. Separate processes on one host establish only local role separation, not independent administrators.

## 2. Execution phases

`resolve → import → compile → public_setup → one_time_prepare → install → ready → request_queue → prefill → decode → verify/decode_output → retire`.

Record every phase, including failures and capacity rejection. Kernel compilation/weight caches are reusable public setup; one-time material is not. No warmup may authorize reuse on different inputs. Benchmark repetitions use fresh material for distinct semantic executions; exact-same-input kernel timing is separately labeled a microbenchmark.

Frozen-inventory mode disconnects Preparation after readiness and fails on any attempted online contact. Sustainable-supply mode measures independent replenishment and its capacity/cost; any stalls still affect request latency. Finite-horizon strict offline deployments report the token/request budget and depletion behavior rather than pretending to run forever.

## 3. Workload ladder and matching

1. Exact operators and their representation conversions.
2. Complete MLP, attention, normalization and vocabulary-feedback regions.
3. A real pinned Qwen2.5-0.5B-Instruct checkpoint and tokenizer, first CPU/local.
4. Multi-process across physical hosts and controlled network conditions.
5. Larger dense and newly supported architectures after independent conformance.

The original 30/63/255-token cases and at most16 generated tokens remain archival cases only when the original numeric manifest is recovered. New fixed-work tests explicitly ignore/pad EOS according to a workload definition and count useful versus padded work separately. They must actually advance the full model/KV state each token, not repeatedly run one projection.

Report three references: upstream floating/checkpoint runtime, frozen plaintext numeric graph, and protected execution of that graph. Changing W4/W8/activation rounding or approximations creates a different accuracy cohort. Same tensor shape does not imply same workload or security model. Compare external author-reported numbers in a separate table; never divide them by a local microbenchmark and call that speedup.

## 4. Measurements

| Metric | Definition |
|---|---|
| request TTFT | Trusted gateway accepts request → first verified/releasable token; client monotonic clock |
| ready TTFT | Inventory-ready release → first verified/releasable token; separately labeled |
| decode TPS | `(G-1)/(last_token_time-first_token_time)` for G≥2; undefined otherwise |
| request output TPS | Useful generated tokens / full request wall time |
| aggregate output TPS | Useful emitted tokens / workload wall time; not sum of per-request ratios |
| inter-token latency | Individual intervals with per-request clustering |
| transfer bytes | Sender-counted directed transfer by phase; two endpoint observations do not double it |
| material | Generated, installed, bound, consumed, retired, wasted, peak stored bytes and production rate |
| compute | Per-role CPU-seconds, device active time and allocated device-seconds |
| memory | Client model/secret footprint, weights, RAM/VRAM, pinned memory, preparation disk/HBM |
| integrity/correctness | Reference mismatches, rejected tampering, unspecified/unsupported checks and failures |
| quality | Loss/perplexity and task deltas, long-context, greedy agreement and sampling-distribution tests |

Use device events and synchronize the measured completion, not just kernel enqueue. Do not subtract timestamps from different machines without a supported clock model. Unknown/unsupported counters are null with a reason, not zero. Payload, framing, TLS, retransmission and physical link counters are distinct layers. Directed pair counters count both directions once each.

For each material j with generation rate r_j and per-useful-token consumption u_j including waste, `TPS_sustainable ≤ min(TPS_online, min_j r_j/u_j)`. Also include preparation-transfer bandwidth, storage/refill limits and all parties' compute. This necessary capacity bound is not a queueing-theory proof. Amortize public compilation over a stated horizon; never amortize one-time garbling across unlimited unrelated requests.

## 5. Statistical reporting

Retain raw request-level and kernel-level samples, order, warmups, retries, failures, timeouts, environment and resource limits. Prefer interleaved matched candidate/control runs. Report medians, quantiles and request-clustered confidence intervals. Three runs are a smoke test; they do not support p99 or universal speed claims. Tail estimates require sufficient independent requests, with uncertainty shown.

Tune on public calibration data and use a held-out public corpus for final comparisons. Reproducible input seeds are distinct from cryptographic randomness. Remote LLM judges are disabled for private data; metric plugins declare whether they invoke external services.

## 6. Compiler search

Search a grammar of legal region implementations, not a Cartesian product of arbitrary encodings. Dimensions include public bounds, permitted numeric alternatives, kernel tile/ISA, label layout, region fusion, LUT ordering, conversion boundaries and preparation batching. Numeric alternatives require an explicitly approved quality cohort. Threat model, corruption threshold, non-collusion, security level and allowed leakage cannot be traded for performance.

Static admission → low-cost arithmetic/assurance checks → operator/region measurements → full-block cost correction → full-model parity/quality → deployment/steady-supply evaluation → held-out confirmation.

Keep Pareto frontiers over online latency/TPS, all-party cost, total traffic and peak memory. Predictions have provenance and uncertainty; top whole plans must be executed because region costs are not simply additive under overlap/contention. Cache public compilation, not consumed secret material. Failed or pruned candidates retain reasons. Any successful in-contract attack immediately removes that method/version from default eligibility and triggers dependent-plan invalidation.

## 7. Run bundle

`experiment.resolved.json`, `model.lock.json`, `numeric.lock.json`, `plan.lock.json`, `environment.json`, `sources.json`, `compile_report.json`, `requests.jsonl`, `events.jsonl`, `traffic.jsonl`, `resources.jsonl`, `inventory.json`, `correctness.json`, `quality.json`, `assurance.json`, `summary.json`, `failures.jsonl`.

Public hashes identify code, algorithms, sources and fixtures. Raw private prompts/outputs, masks, label offsets, active material and credentials are excluded. Deliberate public-fixture attack traces live under a separate threat-evidence policy. Publication bundles use the same manifest and generated tables as the CLI.
