# Benchmark and inspect runs

Run the headless transport benchmark or open the diagnostic dashboard.

[View canonical HTML](https://pllm.run/cli/benchmarking/)

Document ID: `pllm.docs.cli.benchmarking`  
Release: `0.1.0`  
Build: `sha256:d19e46409d656eb3dd08fdadbb8ef6a9e8ca4c33fb48893359235fe855b4a7c4`  
Source hash: `sha256:e6a9254b5ee1cc3476bbb88678124ba2eedbdd93afb87fb6375602216410b86a`

Use the headless runner for repeatable local measurements:

```bash
pllm benchmark run --prompt-file prompt.txt --max-output-tokens 24 --repetitions 3 --output benchmark.json
```

The report contains sanitized diagnostic measurements. It is not a canonical
research `EvidenceReport`, and a loopback run does not establish WAN, multi-host,
energy, price, quality, or production non-collusion claims.

Repeat `--experiment` to compare complete experiment configurations. Each
Experiment contains one immutable Pipeline:

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

The comparison ranks full latency, online latency, time to first token, and
throughput only when model fingerprint, input and output token counts, token cap,
and warm state match. Add `--save-best winner.json` to export the lowest-median
full-latency Experiment for a later rerun. Checked candidate and winner files live
under `examples/benchmarks/`.

An Experiment can instead be built with the SDK in a Python module:

```bash
pllm benchmark run \
  --experiment examples/benchmarks/qwen_prepared.py:cpu_4 \
  --trust-python \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 16 \
  --warmups 1 \
  --repetitions 3 \
  --output benchmark.json
```

The `path.py:object` target names one public `Experiment`. `--trust-python` is
required in noninteractive use because resolving the target imports and executes
the local module. Repeat `--experiment` with other objects to compare them.

The same target can configure a provider role:

```bash
pllm serve inference \
  --experiment examples/benchmarks/qwen_prepared.py:cpu_4 \
  --trust-python \
  --host 127.0.0.1 \
  --port 8000
```

The Experiment supplies the model and CPU thread count. Role addresses and
credentials remain `serve` arguments. The example's `prepared_cpu(model,
threads=...)` SDK function accepts another supported model ID, so component
selection does not need to be duplicated for each model.

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
