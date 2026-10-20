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
operators. Trained request acceptance and quality remain unmeasured.

Historical smoke-build checks: 392 installed Python tests and 590 native library tests pass;
the final focused decoder run passes 51 tests. Production-library Clippy and all
43 docs checks pass. All-target Clippy still reports pre-existing PUMA test lints.

## Serial delivery phase sample

`dense-phase-single.json` is a later ordinary run of the same dense generated
factory with phase counters. Its native quote has 3,875 material records and
4,513 peer frames per holder. Preparation takes 5.396 seconds, of which 5.279
seconds (97.82%) is delivery callbacks. Maximum holder-native run duration is
3.683 seconds; it is not the client online interval. This single sample is
separate from the original smoke reports and any future batching comparison.

The subsequent Decoder/checked-range integration passes 1,025 native tests,
816 installed Python checks and two three-child Decoder controls. Its pinned
Qwen 4+1 source/numeric/work admission is distinct from these generated serving
samples; no successful trained serving result is recorded here.
