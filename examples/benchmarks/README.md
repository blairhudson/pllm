# Pipeline benchmark examples

Each YAML file defines one PLLM `Experiment`, containing one immutable `Pipeline`.
Compare candidates under the same workload with:

```bash
pllm benchmark run \
  --experiment examples/benchmarks/qwen-prepared-cpu-1.yaml \
  --experiment examples/benchmarks/qwen-prepared-cpu-4.yaml \
  --prompt "Explain private inference in one sentence." \
  --max-output-tokens 16 \
  --warmups 1 \
  --repetitions 3 \
  --output comparison.json \
  --save-best examples/benchmarks/top-performers/qwen-prepared.json \
  --force
```

The runner ranks candidates only when every measured run has the same model
fingerprint, input-token count, output-token count, token cap, and warm state.
`--save-best` writes the lowest-median-full-latency Experiment in canonical JSON.

The same candidate can be defined with the public Python SDK. Python targets are
explicit objects in `path.py:object` form and require opt-in because importing the
module executes local code:

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

Compare both SDK-defined objects by repeating `--experiment` with
`qwen_prepared.py:cpu_1` and `qwen_prepared.py:cpu_4`.

Its `prepared_cpu(model, threads=...)` function is reusable across model IDs.
The named `Experiment` objects can also configure either provider role:

```bash
pllm serve inference \
  --experiment examples/benchmarks/qwen_prepared.py:cpu_4 \
  --trust-python \
  --host 127.0.0.1 \
  --port 8000
```

Use the same target with `pllm serve preparation` and that role's endpoint and
credentials. The Experiment binds the model and CPU threads; serve arguments
continue to own role networking and authentication.

The checked-in Qwen winner is the four-thread prepared pipeline. In its matched
single-host loopback run it reached median full latency of 5.229 seconds versus
7.886 seconds for one thread. This is local diagnostic evidence, not a portable
performance or security claim; rerun the candidates on each target system.
