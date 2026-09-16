# Benchmark and inspect runs

Run the headless transport benchmark or open the diagnostic dashboard.

[View canonical HTML](https://pllm.run/cli/benchmarking/)

Document ID: `pllm.docs.cli.benchmarking`  
Release: `0.1.0`  
Build: `sha256:50bdd8d40f8b1f362adbabe08e845d37acfffd39aacc3351d7706456b26c4b1a`  
Source hash: `sha256:30078dca8a0fc8994d2a45c1d93b5fa1e12415c8e45e020cb62cf4caa480df6c`

Use the headless runner for repeatable local measurements:

```bash
pllm benchmark run --prompt-file prompt.txt --max-output-tokens 24 --repetitions 3 --output benchmark.json
```

The report contains sanitized diagnostic measurements. It is not a canonical
research `EvidenceReport`, and a loopback run does not establish WAN, multi-host,
energy, price, quality, or production non-collusion claims.

Use the dashboard when you need a visual transport trace:

```bash
pllm dev dashboard --tiny
```

`--tiny` uses random generated weights and validates transport behavior only.
Interactive runs without `--tiny` use the configured real model.

See [`benchmark run`](/cli/reference/benchmark/run/),
[`dev dashboard`](/cli/reference/dev/dashboard/), the
[benchmark SDK guide](/sdk/research/benchmarks/), and
[research evidence records](/research/records/).
