# Historical conformance results

This directory retains original benchmark reports and generated numerical
results from the 16 October 2026 checks. Each run keeps its original directory,
build, workload and scope; runs are not pooled or relabeled as current results.

- `ordinary-benchmark.json`: ordinary `pllm.benchmark_run.v1` output.
- Operator JSON reports: ordinary `pllm.operator_benchmark_report.v1` output.
- `operator-replay.json`: historical collections of those operator reports.
- `generated-conformance.json`: generated Qwen2/Qwen3/Llama comparison results.

The installed-v4 and clean-local-v1 generated runs each checked 29,592 tiled versus
untiled W8A8 logit/KV values, separately from their upstream FP32 comparisons.
The earlier run's provider-child import identities were unknown. These generated
results establish neither checkpoint quality nor matched serving performance.

Use experiment definitions in `examples/` and normal tests for new work. See
[the implementation summary](../../IMPLEMENTATION-EVIDENCE.md) and
[contributing workflow](../../../CONTRIBUTING.md).

Build receipts, environment dumps, copied sources, generated checkpoints and
archive verifiers were removed during repository cleanup. Earlier versions are
available in Git history. New evidence needs only reusable experiment files and
standard benchmark outputs.
