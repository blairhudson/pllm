# Complete-response client memory

Three fresh client processes ran the existing benchmark driver with pinned
Qwen2.5-0.5B W8A8, 37 input tokens, eight greedy outputs, no warmup, one cold
request and two separate native prepared-provider children. Source, numeric
configuration and token/output cohort matched under one private cohort salt.
All body linear stages remained remote.

| Client allocation path | Process peak RSS, decimal MB | Cold client CPU, s | Request time, s |
| --- | ---: | ---: | ---: |
| Eager mask control + resident weights | 665.49 | 14.23 | 24.54 |
| Lazy native masks + resident weights | 582.84 | 12.42 | 20.07 |
| Lazy masks + streamed paged weights | 306.43 | 10.97 | 19.33 |

Peak RSS falls **54.0%, or 2.17×**. All runtime/privacy checks and outputs match;
observed host swap growth is zero. These are single samples. CPU/latency are not
a general performance ranking: the resident head uses its existing native default
pool, while the paged executor follows the selected CPU thread setting.

The scope includes client import, tokenization, full decoder prefill/decode,
inventory, KV and the in-process dashboard. It excludes provider process RSS,
filesystem cache and total machine memory. Ordinary reports now expose
`client_process_memory`: current RSS and **process-lifetime** peak RSS. Later
candidates in the same process cannot inherit that peak as their own measurement.
The eager control reinstates the independent mask expander only inside its client;
all providers and protocol checks use the ordinary runtime.

## Target audit

**Tenfold whole-client memory is not demonstrated.** The separate Qwen3-4B
import/mask/token-head probe reached 9.44×, but excludes body and KV execution.

The conservative Qwen3-4B 64+8 paged-client estimate is **1.07 GiB**, against the
historical 8.25 GiB estimate and initial 0.83 GiB target. Part is accounting
correction: the decoder already released last-use tensors. New pricing tracks
allocation roots through views/grouped outputs, bounded numeric scratch and
geometric KV/snapshot overlap. Unknown operators/state keep the unreleased-output
price. Estimate correction is not an allocation or RSS reduction.

Benchmarks now retire completed response-owned KV/history after each result.
Separately sealed prefix reuse still works: a three-request regression checks
zero retained response snapshots, cache hits and matching outputs. Application
response-history behavior is unchanged.

A separate phase diagnostic measured 307.74 MB peak. It observed four tokenizer
constructions, up to 63.73 MB of new high-water memory during construction, and
26.69 MB during prefill. Nested samples are not additive. Tokenizer lifetime and
runtime/graph overhead are the next measured targets; no further saving follows
from this diagnostic alone.

Whole-topology admission still blocks a complete 4B response on available host
headroom. A later broad regression attempt passed 97 checks; 11 benchmark cases
were rejected before role launch by low-headroom admission. Earlier isolated
tiny and real-checkpoint runs passed. Rerunning the 11 affected checks in fresh,
smaller test processes passed all 11. Limits were not lowered to force a run.

## Reproduction

From the repository root, with the checkpoint cached and adequate headroom:

```bash
uv run --no-sync python scripts/probe_client_runtime_memory.py \
  --output /tmp/pllm-client-runtime-memory.json
uv run --no-sync python scripts/probe_client_runtime_memory.py --profile-only \
  --output /tmp/pllm-client-runtime-memory-profile.json
```

Outputs must not exist. The parent prices the topology plus coordinator and eager
control; ordinary admission and pressure monitoring remain active. Each candidate
gets a fresh process and artifact cache. Completed reports are checkpointed.
`--tiny` checks transport; `--four-b` must pass admission before provider launch.

Data: `client-runtime-memory-qwen25-2026-10-05.json` and
`client-runtime-memory-profile-qwen25-2026-10-05.json`.
