# Shared native client-weight snapshots

The existing `ClientBundleTransport("artifacts", storage="paged")` selection now
streams each raw integer-weight object into one authenticated native file owner.
Bundle validation reads its actual bytes in bounded chunks, and paged execution
retains the same immutable file instead of copying it into a second snapshot.
Source, object, bundle, weight and numeric identities remain checked.

## Ownership and admission

Rust owns the bounded writer, incremental SHA-256, signed-i8 magnitude bound,
private temporary file and positional reads. Only a completed, digest-matching
writer can produce a snapshot; incomplete, oversized and mismatched streams
retire their pending file. The file exposes no writable path or byte alias.
Paged binding checks both geometry and original content digest. Python reads fill
one bounded unpublished bytes object directly, releasing the interpreter lock
without an intermediate full-chunk Rust allocation.

Each matrix retains an independent reference to the same file. Closing the
delivery owner or another matrix cannot invalidate a live kernel; the last owner
releases the file. Mutation of source buffers, cache corruption and eviction do
not change an already admitted snapshot. Tokenizer source files retain their
separate existing contract. The direct `PagedGEMM.from_raw` API still snapshots
arbitrary caller paths before use.

Whole-topology admission requires the shared-snapshot backend and prices one
private client-weight file payload. Artifact cache, tokenizer source/transient
files, indexed-tokenizer snapshot, provider caches and compiled source storage
remain separate disk owners. RAM estimates retain their bounded I/O and kernel
workspace allowances. No persistent cache or private token-indexed download is
introduced.

Generated prepared, Freivalds-verified and two-offset SDK controls cover exact
prefill/decode logits and KV, client-owned body stages, tied and untied boundaries,
cache reuse, gateway execution and failure cleanup. Native and Python controls
also check malformed writers, digest/shape substitution, source mutation and
independent owner retirement.

## Isolated fresh-process import screen

Twelve fresh processes alternate the previous Python-file-plus-native-copy
control and shared native snapshot. Both stream identical public synthetic i8
weights, re-hash the actual snapshot for bundle validation and evaluate one head
row against an independent bounded NumPy int64 oracle. Three processes per
choice and shape preserve all **1,823,232 checked outputs**, with zero observed
swap growth.

| Synthetic head shape | Copied import CPU | Shared import CPU | Copied peak private-file payload | Shared peak private-file payload |
| --- | ---: | ---: | ---: | ---: |
| 151,936 × 896 | 0.782006 s | 0.638230 s | 272.27 MB | 136.13 MB |
| 151,936 × 2,560 | 2.195098 s | 1.821806 s | 777.91 MB | 388.96 MB |

Median import CPU falls **18.39% / 17.01%**. Private-file payload counts are
logical live-owner accounting, not physical disk occupancy or filesystem-cache
measurements. Both choices retain one file after delivery retirement. Median
process peak RSS is **45.07 → 43.70 MB** and **44.76 → 44.71 MB**; this screen
does not establish a material whole-client RAM reduction. Execution CPU remains
roughly 0.03 / 0.10 seconds. Public sources are pre-positioned, OS cache is
uncontrolled, and publication, checkpoint values, provider transport and full
response execution are outside these timings.

```sh
python scripts/probe_public_snapshot.py --samples 3 --output public-snapshot.json
```

Evidence: `docs/evidence/research-native-snapshot-import.json`. Initial development
screens with an extra read-buffer copy are not included in this cohort.

## Matched whole-response screen

A separate installed-wheel pair runs the same pinned Qwen2.5-0.5B W8A8
Experiment at **150 input + eight output tokens**, with native shared **100/40
Mbps** caps and one fresh client per choice. The allocation control selects the
previous Python file plus second native copy and prices both disk owners; the
candidate uses shared native files. Both use the same native library, source,
Pipeline, numeric body and one parent-owned comparison salt. Each client has an
isolated artifact cache; the checkpoint and public tokenizer index are
pre-positioned. The control is an allocation ablation, not an older complete
wheel.

Both responses and the canonical comparison pass, with identical output text
digests and **476,952,886 covered application-body bytes** each. All runtime audit
checks pass, with zero observed swap growth. Evidence:
`docs/evidence/research-native-snapshot-qwen25.json`.

| Priority metric | Copied files | Shared native file |
| --- | ---: | ---: |
| Client lifetime peak RSS, MB | 222.54 | 237.13 |
| Client cold-first CPU, s | 25.795 | 26.371 |
| Preparation cold-first CPU, s | 19.633 | 20.120 |
| Preparation sampled request-window peak RSS, MB | 198.36 | 202.08 |
| Online TTFT, s | 51.044 | 51.881 |
| N−1 decode tokens/s | 2.906 | 3.532 |
| Cold-through-response all-link TCP streams, MB | 477.933111 | 477.933120 |

Complete request time rises **115.041 → 116.738 seconds (1.48%)**, client peak RSS
**6.55%**, and aggregate cold-first CPU **75.230 → 77.994 seconds (3.67%)**.
Sampled simultaneous topology RSS is nearly unchanged at **1,549.43 / 1,551.84
MB**. Preparation elapsed time rises **34.309 → 36.147 seconds**. The higher
decode sample does not establish a causal decode speedup from an import change.
This single pair establishes no whole-response latency or client-memory benefit;
the isolated import CPU and logical file-payload savings remain separate results.

Client RSS covers each fresh process lifetime; client CPU starts at benchmark
startup and provider CPU at process birth. Preparation RSS is the largest sample
inside the request window, so it can miss startup and shorter peaks. Online TTFT
excludes setup and demand preparation. TCP-stream counts include application
framing, with a nine-byte control-length difference, but exclude kernel headers,
ACKs, retransmissions, telemetry and checkpoint distribution. The sampler excludes
filesystem cache and uncharged GPU allocations. No independent-operator or
generation-quality result follows.

Build the public artifacts with the [prepared-frontier instructions](prepared-frontier.md#reproduction),
then run using an installed wheel from this revision. Allow at least ten minutes
for the parent process and use a fresh output path:

```sh
PLLM_RESEARCH_ARTIFACTS=target/prepared-frontier-artifacts HF_HUB_OFFLINE=1 \
python -m scripts.probe_client_response_memory \
  --allocation-control public-snapshot \
  --experiment benchmarks.research.sota:experiment \
  --prompt-file target/prepared-frontier-artifacts/prompt.txt \
  --max-output-tokens 8 --output public-snapshot-response.json
```

An earlier attempt was interrupted by an outer 120-second timeout and contributes
no completed measurement to this pair.

## Qwen3 whole-response admission

The fresh-client Qwen3-4B 16+8-token pair rejected before provider launch: its
estimated host peak was **7.61 GiB**, above **6.12 GiB** safe headroom. No Qwen3
whole-response result follows from the synthetic import measurements.
The rejection and original empty progress record are retained separately in
`docs/evidence/research-native-snapshot-qwen3-admission.json`.

This is a native ownership and load-ready artifact slice. It does not implement
a Python-free decoder, replace source verification with trusted metadata, or
establish a 10× client-memory or serving improvement.
