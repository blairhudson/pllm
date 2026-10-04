# Qwen3-4B benchmark memory failure

The attempted four-configuration, two-response WAN cohort produced **no completed
benchmark result**. The CLI timed out during initial prepared inventory startup;
the user subsequently reported host memory exhaustion and a host crash.

## Evidence

- Host: Apple Silicon, 32 GiB unified memory.
- Docker Desktop was temporarily raised from approximately 8 GiB to 20 GiB.
- macOS recorded `JetsamEvent-2026-10-04-220406.ips` at
  `2026-10-04T22:04:06+11:00`, with `vm-compressor-space-shortage` reasons.
- Its largest process was `com.apple.Virtualization.Virtual`. The recorded
  `rpages` was 1,312,165 at 16,384 bytes/page (approximately 20.02 GiB of process
  footprint; not a sampled per-role resident-memory measurement).
- Post-failure inspection found 17.5 GiB of host swap in use. Docker initially
  returned metadata I/O errors during cleanup.
- Docker Desktop was stopped, its temporary memory override removed, and
  restarted with 8,321,515,520 bytes reported by `docker info`. The surviving
  benchmark-owned containers and network were removed. Other containers and
  model caches were preserved.

## Inadequate admission

`qwen3-4b-preflight-2026-10-04.json` measures checkpoint identity, schedule
geometry and selected retained storage only. Its explicit scope excludes peak
memory. It was incorrectly used to justify increasing Docker's budget.

Each provider owns quantized weights and a native snapshot, including both
boundary-stage orientations even when the client bundle deduplicates a tied
token/head table. Startup also allocates floating-point quantization work,
validation temporaries, bundle serialization/import copies, client snapshots,
inventory masks/corrections and decoder state. Docker has a separate guest page
cache and competes with the host client, filesystem cache and other applications.
The incident log establishes pressure, not which individual allocation peaked.

## Implemented admission and native fallback

`pllm.metrics.benchmark_memory` and `benchmark run --preflight-only` now price
the complete local role graph and client startup/runtime phases before loading
model values or creating providers. Allocation estimates have 25% slack; host
admission retains 25% of physical RAM (at least 2 GiB), counts existing available
RAM, and excludes swap. Docker also checks its existing capacity and other
containers, gives providers hard no-additional-swap limits, and never raises the
Desktop memory budget. A 250 ms runtime monitor aborts owned work on reserve loss
or more than 256 MiB of new swap. These are conservative estimates and monitored
admission, not a proof against every concurrent allocation or an observed peak.

The cached Qwen3-4B lean prepared composition estimates **34.78 GiB** for native
startup/operation and rejects on this 32 GiB host. Docker rejects its provider
peaks against the restored ~8 GiB guest. No 4B inference was relaunched after the
incident. Missing crypto allocation models also reject rather than assigning a
zero cost.

The canonical rejection is retained in
`qwen3-4b-memory-admission-2026-10-04.json`. Compiled cache-input bounds are priced
separately from admitted request lengths, including cache capacity on a miss.

`--backend native` selects existing local provider processes, including explicitly
selected Metal. `--backend auto` can choose native when Docker is unsafe. GPU
snapshots still count against unified RAM and retain the 2 GiB per-role bound.
Enforced Docker WAN/link shaping never silently becomes unthrottled native work;
native `--wan-estimate` remains analytical.

## Bounded native control

The ordinary multi-Experiment CLI completed CPU and Metal **22+2-token** tiny
prepared requests under automatic backend selection:

- Both selected native; Docker's whole VM did not fit the then-available host
  budget, and Metal additionally requires native roles.
- Both report every individual runtime/privacy check passing, identical output
  digests, zero plaintext prompt/token-ID traffic, and no online preparation.
- The memory guard observed **zero swap growth**, no abort, and minimum available
  host RAM of 13,932,904,448 / 13,902,954,496 bytes (CPU / Metal).
- No client body weights were added. Provider cleanup completed.
- A separate pressure-injection regression starts both real native provider
  children, trips the host reserve, and verifies both PIDs and the guard thread
  retire. It injects telemetry, not actual memory exhaustion.
- Comparison `all_candidates_passed` and `matched_workload` are true;
  `matched_kernel_backend` is false. The combined comparison is intentionally
  **unranked** across CPU/GPU scopes, rather than claiming a performance winner.

Report: `native-memory-safety-2026-10-04.json`. Generated tiny weights establish
functionality and guarded backend selection, not Qwen3-4B capacity or quality.

```sh
uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/memory_safety.py:cpu \
  --experiment examples/benchmarks/memory_safety.py:metal \
  --trust-python --backend auto --prompt Hi --max-output-tokens 2 \
  --warmups 0 --repetitions 1 --temperature 0 --capture-output-digest \
  --timeout 90 --output docs/evidence/native-memory-safety-2026-10-04.json

HF_HUB_OFFLINE=1 uv run --no-sync python -m pllm benchmark run \
  --experiment examples/benchmarks/qwen3_4b.py:lean --trust-python \
  --backend auto --preflight-only --max-output-tokens 8
```

## Follow-up: reduce allocations

The [single-snapshot loader follow-up](weight-snapshot-memory-2026-10-04.md)
replaces duplicate provider i8 storage with read-only views of Rust-owned weights
and bounds float quantization/validation chunks. The same lean Qwen3-4B native
estimate falls from **34.78 to 23.60 GiB**. It still rejects against current host
headroom; this is not a successful 4B rerun. Older Docker images retain the
legacy allocation upper bound. The original rejection record above is preserved.
