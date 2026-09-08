# Tune the right bottleneck

Measure the complete lifecycle before increasing a batch.


## Start with serial latency

If a token requires `R` dependent exchanges and each exchange adds `L` seconds, network dependency alone contributes `R × L`. A persistent WebSocket removes connection setup but not those dependencies.

The Qwen3.5-27B study plan has 257 exchanges. At 20 ms per exchange, the dependency floor is 5.14 seconds per token before arithmetic or transfer.

## Batch compatible stages

```bash
pllm serve "$PLLM_MODEL_SOURCE" \
  --weights public \
  --local-files-only \
  --max-batch-size 16 \
  --max-wait-ms 0.5 \
  --engine-threads 4
```

These are tuning inputs, not a recommended universal optimum. Measure single conversation latency separately from aggregate throughput. Other users can fill a matrix batch without making one user's dependent stages parallel.

## Bound client preparation

The reference SDK defaults to four prefetched correlations and a token cache of 512 entries. Public-weight preparation packs those four masks into guarded BFV slot segments and tiles matrices that exceed one segment. Increasing the horizon can help long conversations but wastes work for short ones. The separate coordinate preparation study packs 2,048 future masks under one key; it is not the reference server's default.

## Count both links

Report input and output bytes separately, and include the preparation stream. State whether a bandwidth assumption is shared or available independently in both directions. The paper's 1.34 tokens/s figure is a bandwidth ceiling for a model plan, not measured Qwen throughput.

## Keep scales private

Activation scales stay local. Do not send them to simplify a server dequantization path. The integer provider operation does not need them, and a scale can fingerprint a private activation.

See [the benchmark record](/docs/research/benchmarks) for comparable scopes and downloadable results.
