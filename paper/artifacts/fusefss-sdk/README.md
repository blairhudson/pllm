# FuseFSS SDK smoke benchmarks

Ordinary CLI outputs for the generated experiments in
[`examples/papers/fusefss_smoke.py`](../../../examples/papers/fusefss_smoke.py).
These exercise two private full-prefix selections through a separate dealer and
two online workers. The generated weights have no model-quality claim.

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:experiment --factory --trust-python \
  --backend native --prompt hello --max-output-tokens 2 --timeout 300 \
  --capture-output-digest --output target/fusefss-gpt2.json
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_experiment --factory --trust-python \
  --backend native --prompt hello --max-output-tokens 2 --timeout 300 \
  --capture-output-digest --output target/fusefss-dense.json
```

The two numeric profiles are separate cases, not a speed comparison. Buffered
final disclosure has no streaming TTFT or decode-throughput measurement. Dealer
and worker costs, directed application bodies, and encrypted tape I/O retain
their ordinary report fields. Local children do not establish independent
operators. Full trained checkpoints remain outside current admission limits.

Correctness checks: 392 installed Python tests and 590 native library tests pass;
the final focused decoder run passes 51 tests. Production-library Clippy and all
43 docs checks pass. All-target Clippy still reports pre-existing PUMA test lints.
