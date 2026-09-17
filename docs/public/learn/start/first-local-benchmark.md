# Run the local development benchmark

Run the real PLLM roles on one machine and understand what the result does and does not measure.

[View canonical HTML](https://pllm.run/learn/start/first-local-benchmark/)

Document ID: `pllm.docs.start.first-local-benchmark`  
Release: `0.1.0`  
Build: `sha256:06781cb06588c2919922b5252154662ef566cf6cbec9ad8049f32aa70887b75e`  
Source hash: `sha256:c277b81019938f48fc33ad027df967873ac7f89044b32d030597da2200c2f1e6`

Run the real local client, preparation, and inference roles:

```bash
pllm benchmark run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt "Explain why private inference matters." \
  --max-output-tokens 8 \
  --output benchmark.json
```

Review the exact [`benchmark run` options](/cli/reference/benchmark/run/) before
changing the model, workload, repetitions, or output behavior.

The command is headless by default, reports startup and execution progress on the
terminal, writes a sanitized JSON report, and stops all three roles. Add
`--show-dashboard` only when you want the interactive local view. Use `--tiny`
only to test transport with generated weights. The report is not a canonical
benchmark evidence record. A local run does not measure a multi-host or wide-area
deployment, model quality, energy use, price, adversarial behavior, or production
non-collusion.

Continue with [research evidence requirements](/research/evidence/) and
[metrics](/research/records/metrics/) before comparing results.
