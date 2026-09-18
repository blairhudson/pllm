# Compare

Compare runs with matching inputs and report every relevant cost and uncertainty.

[View canonical HTML](https://pllm.run/research/recipes/compare/)

Document ID: `pllm.docs.measure.compare`  
Release: `0.1.0`  
Build: `sha256:2e8eacabaf27e4c40f6b41814c3af934ab5186546825b569c6c9c1cb0a7b4db4`  
Source hash: `sha256:a602c74ba70adde296d62e32165b2c058f5a3a472ad847124318fcb7a50a1d52`

Matched comparison freezes model/tokenizer, workload, numeric semantics, quality acceptance, privacy contract, roles/topology, hardware, software, preparation/freshness, cache, warmups, repetitions, failure accounting, and metrics.

Report TPS, latency, peak memory, CPU, network, disk, preparation, quality, and privacy as vector. Different cohorts remain separate. A Pareto improvement is no worse on every comparable objective and strictly better on at least one under predeclared uncertainty treatment.

The loopback diagnostic runner compares multiple Experiments directly:

```bash
pllm benchmark run \
  --experiment examples/benchmarks/qwen-prepared-cpu-1.yaml \
  --experiment examples/benchmarks/qwen-prepared-cpu-4.yaml \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 16 \
  --warmups 1 \
  --repetitions 3 \
  --output comparison.json
```

It emits no rankings unless model fingerprint, input and output token counts,
token cap, and warm state match exactly. This is a diagnostic comparison, not a
canonical evidence-producing search facade. Quality acceptance, uncertainty,
multi-host topology, energy, price, and Pareto-front selection remain future work.
