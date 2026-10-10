# Native-owned execution instructions

This internal load-ready artifact moves validated public graph instructions and
last-use metadata into a Rust owner. Each phase exposes one bounded instruction
frame to the existing semantic executor; Python does not retain the complete
parsed operation and schedule lists in that executor. Tensor arithmetic, transport,
tokenization, dependencies, source binding and Python dispatch remain unchanged.

The native compiler derives instructions, dependencies, logit roots and retirement
points from the original model-neutral semantic plan. An explicit trusted
publisher digest, complete checkpoint binding, model/composition identity and
native/Python implementation digest precede import. The artifact is bounded at
8 MiB and 16,384 steps per phase, with 64 KiB loaded instruction frames. It stores
public metadata only. A digest obtained from an untrusted artifact is not a
publisher authorization.

The reference admits ordinary dense full-KV prefill/decode. Continuation,
transformed state, protected nonlinear, verifier and tiled execution remain
separately gated. Generated Qwen2/Qwen3/Llama controls compare logits and active
KV exactly, exercise publisher-digest rejection and retain failure cleanup.
The immutable source plan and compiled binding still exist; this is not a whole
native decoder or elimination of compiler startup allocation.

## Pinned checkpoint screen

[`research-execution-capsule.json`](../evidence/research-execution-capsule.json)
records one publisher and three alternating fresh-client pairs, using the same
installed wheel. Every compared logit and active KV digest matches, with zero
observed swap growth.

| Median per client | Python instructions | Native instructions |
| --- | ---: | ---: |
| Unique Python graph/schedule bytes | 3,214,260 | 138,557 |
| Process lifetime peak RSS, decimal MB | 1,108.49 | 1,108.90 |
| Runtime construction CPU, seconds | 1.251 | 1.170 |
| Complete decoder execution CPU, seconds | 9.391 | 9.435 |
| Complete decoder execution elapsed, seconds | 4.139 | 4.020 |
| Cold process CPU, seconds | 20.904 | 21.460 |

The 23.20-fold reduction concerns only the counted Python graph/schedule objects.
Native instructions, immutable source and checkpoint weights remain additional
owners. Peak RSS is effectively flat and cold process CPU rises 2.66%; this
screen does not establish a thin-client memory improvement. The public capsule
is 994,972 bytes. Its separately timed publication costs 1.104 CPU seconds after
checkpoint import; publisher startup/loading and artifact distribution are not
included in that publication window. These three-replicate local-clear samples
do not establish prepared SDK response gains or independent performance confirmation.

## Replay

```sh
uv run --no-sync pytest -q tests/test_execution_capsule.py
uv run --no-sync python scripts/probe_execution_capsule.py \
  --output target/execution-capsule.json --processes 3
```

A separate publisher plus six alternating fresh clients execute the pinned
Qwen2.5-0.5B checkpoint at 150 input rows and eight selected outputs. Reports
separate publication, construction, execution, process CPU/RSS and unique Python
graph/schedule ownership. Native instructions and immutable source bytes are
additional owners, not absent memory. Complete local-clear execution charges all
checkpoint kernels; it does not measure prepared SDK client RSS or WAN serving.
