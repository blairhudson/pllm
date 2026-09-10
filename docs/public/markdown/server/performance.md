# Tune the right bottleneck

Measure the complete lifecycle before increasing a batch.


## Start with serial latency

If a token requires `R` dependent client-to-inference exchanges and each exchange adds `L` seconds, online network dependency alone contributes `R × L`. The preparation WebSocket preloads inventory offline and is not an additional online round.

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

## Measure offline inventory preparation

Before chat, the client creates an inventory and sends one root seed per stage for
a batch of rows. The default is 64 rows per stage and
`PLLM_PREPARED_INVENTORY_ROWS` configures it. Preparation sends each `W·r-s` batch
to inference over a persistent binary WebSocket and waits for durable
acknowledgement. Inference seals the complete inventory `READY`. Online stages then
carry only ticket plus `x-r` from client to inference and `W·x-s` back; preparation
is idle. Refill occurs only between executions.

Measure authorization and root-seed upload, durable acknowledgements,
preparation-to-inference correction bytes, seal latency, warm client-to-inference
traffic, and consumed or burned rows separately. Preparation metrics expose
correction payload bytes, channel upload bytes, compute nanoseconds, and send
nanoseconds. Inference metrics expose channel connections, frames, accepted bytes,
failures, and processing nanoseconds. Byte counters cover application payloads and
PLLM's MessagePack envelopes, not WebSocket, TLS, or IP framing. Restart or idle
expiry discards inventory; cancellation, early end of stream, and failure burn
unused reserved rows, so requested batch size is not useful throughput.

## Count both links

Report input and output bytes separately, and include the preparation stream. State whether a bandwidth assumption is shared or available independently in both directions. The paper's 1.34 tokens/s figure is a bandwidth ceiling for a model plan, not measured Qwen throughput.

## Keep scales private

Activation scales stay local. Do not send them to simplify a server dequantization path. The integer provider operation does not need them, and a scale can fingerprint a private activation.

See [the benchmark record](/docs/research/benchmarks) for comparable scopes and downloadable results.
