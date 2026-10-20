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

## Matched material-delivery pair

`material-batch-pair.json` retains one fresh client per choice under a shared
private cohort. Both use the same generated 1+2 source and numeric contract.

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss_smoke.py:dense_single_smoke \
  --experiment examples/papers/fusefss_smoke.py:dense_batch_smoke \
  --factory --trust-python --backend native --prompt hello \
  --max-output-tokens 2 --timeout 300 --capture-output-digest \
  --isolate-candidates --output target/fusefss-material-batch-pair.json
```

| Measured owner/window | Single | Batch v1 |
| --- | ---: | ---: |
| Dealer preparation, seconds | 5.393 | 0.390 |
| Delivery callbacks, seconds | 5.277 | 0.288 |
| Maximum holder-native run, seconds | 3.565 | 3.656 |
| Full request, seconds | 9.680 | 4.796 |
| Aggregate cold-first CPU, seconds | 11.022 | 6.956 |
| Covered all-link bodies, bytes | 21,618,568 | 19,909,435 |
| Online bodies, bytes | 7,681,488 | 7,681,488 |
| Each holder's tape reads/writes, bytes each | 6,189,173 | 6,189,173 |

Output digests agree and both memory guards observe zero swap growth. Fresh
material remains distinct per request. Full request time excludes topology startup;
cold-first CPU includes it. The unchanged online protocol retains its own cost.
This generated single pair establishes neither trained-checkpoint acceptance nor
an independently confirmed performance effect. Installed batching checks pass 282
tests; the factory-name follow-up passes 24 checks. Native decoder
tests pass 107 cases; both batched three-process parity/rejection tests pass.

## Pressure-stopped trained-Qwen attempt

`qwen-checked-batch-pressure.json` preserves the ordinary CLI progress checkpoint
from the pinned checked-range/batched factory, with a four-input-token bound and
one output selection:

```sh
uv run --no-sync pllm benchmark run \
  --experiment examples/papers/fusefss.py:qwen_checked_batch_smoke \
  --factory --trust-python --backend native --prompt hello \
  --max-output-tokens 1 --timeout 14400 --capture-output-digest \
  --output target/fusefss-qwen-checked-batch-response.json
```

Source and topology admission passed, and the holders reached online execution.
After approximately 66 minutes in the measured request window, the benchmark
stopped with `benchmark stopped: macOS memory pressure is warning`. The four-hour
deadline had not expired. All role processes retired; temporary disk ownership
was released. The host-pressure observation does not identify which allocation
or process caused it.

The checkpoint retains `status="failed"`, `complete=false` and `report=null`.
There is no accepted token, final numeric verdict, completed resource report or
quality/performance result. A three-second live stack sample was also taken
during this attempt; it is a narrow diagnostic, not whole-request attribution.
Online peer exchanges remain unbatched. Calibrate dependency-safe online batching
and callback/transport costs before another long trained-model attempt.
