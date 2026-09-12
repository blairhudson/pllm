# Run your own benchmark

Measure private inference on your hardware and extend PLLM's map of eligible compute.


PLLM's market mission depends on knowing where private inference can run and what
the complete lifecycle costs. Benchmark the current public-weight offline-inventory
runtime on your hardware, then publish a comparable, privacy-safe record. The
retained study is available as
[current-runtime-2026-09-11.json](/downloads/current-runtime-2026-09-11.json).

## Start the dashboard

```bash
pllm benchmark dashboard --model Qwen/Qwen2.5-0.5B-Instruct --no-open
```

For an exact comparison, match PLLM revision `277d19f`, Python 3.13.15,
Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`, macOS 26.5.2, an Apple M5, and 32
GiB memory. Client, Preparation, and Inference were separate loopback processes.

For your own cohort, choose explicit public prompts and record the complete
environment. For an exact reproduction, run one excluded warmup, then submit the
three public prompts in the evidence file three times each with
`max_output_tokens=16`. Preserve the exact input-token count reported by the
tokenizer rather than estimating it from words.

## Validate the privacy boundary

For every accepted run, verify:

- schema version 3 and status `completed`;
- identical model and immutable body fingerprints;
- zero online Preparation protocol requests and operations;
- zero plaintext prompt and token-byte audit counters;
- one explicit request ID and one authoritative completion;
- clean one-signal shutdown and no orphan role processes.

Export sanitized records, host information, Git revision, model revision, public
workload, cache state, topology, and command flags together. Dashboard history
deliberately excludes prompt and output text, so publish only an explicitly
non-sensitive workload manifest beside it.

## Compare like with like

Hold model/body fingerprint, source revision, cold/warm mode, exact input tokens,
output allowance, host, and topology constant within a cohort. Report preparation,
transition, online latency, client traffic, correction traffic, inventory rows,
and process effort separately. Offline work remains real cost.

Do not merge CPU loopback, WAN, GPU, or multi-host records into one performance
claim. Separate cohorts make the emerging supply map useful: they show which
hardware and deployment boundaries can deliver compatible private model work, at
what complete cost.
