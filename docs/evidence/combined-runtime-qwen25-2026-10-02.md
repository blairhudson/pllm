# Combined current runtime optimisations: Qwen2.5-0.5B

Measured 2026-10-03 UTC against source revision
`7ae557604adf67be50417f59c2c2f167def9a775` and runtime commit
`6b4d515ebcf72cddb53acbf62f2a765276aac54d`. The filename is the cohort identifier.
Experiment specifications and native composition identities are retained in each
JSON record. Benchmark configurations live in
`examples/benchmarks/combined_runtime.py`; execution and cohort verification use
`scripts/probe_combined_runtime.py` and the ordinary loopback benchmark driver.

## Cohort

- W8A8 with the same explicit `causal_reduction="prefix_f32"` numeric choice.
- Greedy decoding, 32 output tokens per request.
- Three ordered requests: cold 39-token prompt, identical 39-token prompt, then a
  45-token extension. All seven candidates produced matching output-text digests
  and token counts for every request: 21 completed responses, 672 output tokens.
- One measurement per position and configuration; timings are diagnostic samples.
- Apple M5, 10 CPU cores, 32 GiB unified memory, macOS 26.5.2. Separate co-located
  provider processes; four-thread CPU kernels
  except the explicit Metal candidate (`min_rows=32`). CPU and GPU results are
  not ranked together by the canonical comparator.
- Fresh isolated client bundle/object cache per candidate; checkpoint and compiled
  weights already cached locally. Model downloading is excluded.
- Bytes are decimal MB of recorded application bodies, not full wire traffic.
  Setup-inclusive columns include recorded initial bundle and inventory costs.
  Request wall times exclude provider startup. Cold CPU includes client/dashboard
  startup and role CPU from process birth through the first response.

The prepared default control explicitly uses 64-row prewarming, idle refill, raw
bundle transport and no prefill cache. It uses the same canonical numeric mode as
the candidates, rather than comparing different attention arithmetic.

## Combined results

| Configuration | Three-request setup-inclusive bodies | Online bodies | Sum of request wall time | Cold aggregate CPU | Cold client CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prepared default control | 954.17 MB | 350.21 MB | 79.63 s | 72.95 s | 12.30 s |
| Reuse + on-demand inventory + artifacts | 517.94 MB | 236.89 MB | 54.14 s | 53.41 s | 13.95 s |
| Reuse + on-demand inventory + zlib | 497.37 MB | 236.89 MB | 60.94 s | 54.61 s | 13.50 s |
| Above + client attention projections | 497.61 MB | 193.95 MB | 48.80 s | 49.21 s | 15.29 s |
| Client attention + reuse + on-demand + zlib | **472.36 MB** | **193.95 MB** | 49.24 s | 50.79 s | 14.10 s |
| Reuse + on-demand + artifacts + Metal | 517.94 MB | 236.89 MB | 54.82 s | 53.24 s* | 15.23 s |
| Two-offset, seeded inputs + row residues | Unknown | 532.67 MB | 68.96 s | 47.87 s | 14.92 s |

\* CPU only; GPU work is not measured. The offset canonical report retains unknown
setup-inclusive accounting; its three recorded request windows contain 678.96 MB.
That narrower quantity is not substituted for the missing setup-inclusive total.

The best measured network combination reduces setup-inclusive bodies **50.49%
(2.02x)** and online bodies **44.62%** across this three-request workload. Summed
request time is 38.17% lower in these samples. Exact-repeat and shared-prefix
savings do not apply to unrelated fresh prompts.

## Fresh response versus reuse

| Request | Control online bodies | Combined online bodies | Control request wall time | Combined request wall time |
| --- | ---: | ---: | ---: | ---: |
| Fresh 39+32 | 113.55 MB | **93.22 MB** | 24.39 s | 21.52 s |
| Identical 39+32 | 113.55 MB | **41.79 MB** | 24.66 s | 13.08 s |
| Extended 45+32 | 123.12 MB | **58.94 MB** | 30.58 s | 14.63 s |

Combined here means the client-attention/zlib candidate. Its fresh cold
setup-through-response bodies are **312.60 MB**, versus **510.02 MB** for the
control (38.71% lower). Its fresh online saving is **17.90%**, not the 50.49%
three-request saving. Fresh online execution takes 10.67 s after preparation;
the complete request takes 21.52 s, excluding role startup. Decode rates are
5.58, 5.20 and 4.86 tokens/s across the three positions, not whole-request rates.

The exact repeat reuses completed-prefill logits/KV and still executes 31 decode
rows. The extension reuses a sealed 32-token checkpoint and executes the 13-token
suffix plus decode. The reports record zero batched-continuation hits; this cohort
does not establish a batched-suffix acceleration.

## Client and protocol tradeoffs

The client-attention option adds 44.04 MB i8 weights, 0.20 MB scales, and a 44.04 MB
native weight snapshot: **88.28 MB of additional owned payloads**, not peak RSS.
The private cache has a 64 MiB cap, not a measured allocation. Body-linear MACs
remain 87.69% remote; this is not a whole-response remote-compute percentage.
Cold client CPU rises from 12.30 to 14.10 CPU-s for the zlib combination even as
aggregate cold CPU falls from 72.95 to 50.79 CPU-s.

The low-client-memory configurations keep all body-linear weights remote and
avoid those 88.28 MB of extra payloads. The artifact option uses 517.94 MB overall
(45.72% below control). Zlib reduces that to **497.37 MB (47.87% below control)**,
with 303.87 MB through its first cold response, but summed request time increases
from 54.14 to 60.94 s in the checked samples. Client-attention/zlib saves another
5.03% versus low-client-memory/zlib across this cohort. That incremental byte
reduction should be weighed against the additional client storage and arithmetic.

On-demand inventory issues exactly the rows claimed in these cohorts and discards
none. Control issues 47,040 stage rows, claims 20,736, and discards 26,304 at close.
The low-client-memory candidate issues/claims 13,920; client attention issues/claims
6,960. Material accounting is conserved; all remaining rows are zero after close.

Artifact delivery and zlib are alternative encodings. The cold fixed-placement
cohort gives zlib 25.25 MB fewer bodies than artifact delivery; artifact locality's
cross-placement object reuse is not exercised here. Client-attention placement
and Metal are currently rejected as a combined compiler composition, so the Metal
candidate retains remote body projections. It shows no latency advantage over the
otherwise matched CPU artifact candidate in this one sample.

Seeded input and row-residue encodings belong to the two-offset protocol, not the
prepared path. They cannot be added to its savings. Private pages, approximate head
retrieval and standalone file-backed kernels are not admitted whole-decoder
components and do not enter these numbers.

## Validation and reproduction

Every candidate passed the canonical completion, source/body, sampling and
no-plaintext-prompt/token-ID checks. The summary verifies matched ordered workloads,
exact Experiment specifications and identical output-text digests. CPU-only
comparison checks pass. The combined CPU/GPU comparison intentionally does not
pass backend matching; no mixed-backend canonical ranking is produced. Output
agreement is with this W8A8 control, not independent broad generation quality.
Client peak memory, complete wire accounting and operator independence remain
unmeasured.

Example measured invocation from the repository root (shared pinned checkpoint
must already be cached):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/probe_combined_runtime.py \
  --candidate compressed \
  --output docs/evidence/combined-runtime-zlib-qwen25-2026-10-02.json
```

Other candidate names: `baseline`, `lean`, `lean_compressed`, `combined`, `metal`, `offset`.
`--summarise` accepts their JSON files and validates the cohort without rerunning
models. Detailed data: `combined-runtime-summary-qwen25-2026-10-02.json` and its
seven SHA-256-bound candidate records.
