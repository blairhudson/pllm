# Benchmarks

Measure PLLM systems with reproducible records and compare only equivalent runs.

[View canonical HTML](https://pllm.run/sdk/research/benchmarks/)

Document ID: `pllm.docs.benchmarks`  
Release: `0.1.0`  
Build: `sha256:a3db51eca723684328314c5428e7e991d7b5bb20ff973c6d46dfe9a79aec3a08`  
Source hash: `sha256:d27ea7f28d2654287fed98258151eed265ddc263ce1d076a7821857edfc7776a`

`pllm.benchmark(...)` measures supported native compiled regions and returns an
immutable `EvidenceReport`. `pllm.deployment_benchmark(request)` validates
observations supplied by the caller, but cannot prove where they came from. The
CLI also exposes the existing real-role loopback runtime as a non-interactive
diagnostic through [`pllm benchmark run`](/cli/reference/benchmark/run/):

```bash
pllm benchmark run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file prompt.txt \
  --max-output-tokens 24 \
  --repetitions 3 \
  --output benchmark.json
```

The command chooses an ephemeral loopback port, starts the client, preparation,
and inference roles, waits for readiness, runs each request, reads the
authoritative archive record, and stops the process group. `--format json`
returns a machine-readable CLI envelope and `--dry-run` starts no services.

Every record identifies whether it covers a compiled region, protocol, deployment,
complete model, privacy test, or quality test. Compare runs only when the model,
body fingerprint, plan, workload, numeric settings, environment, threads, SIMD
settings, warmup, repetitions, and cold or warm mode match.

The CLI report and local dashboard help diagnose a loopback deployment, but they
do not create a canonical [research evidence](/research/evidence/) record. See
[metrics](/research/records/metrics/) and
[research experiments](/research/recipes/experiments/).

This SDK example requires a PLLM source checkout because it reads the checked-in
`schemas/fixtures/compile-request.valid.json` compile request.

## Python SDK example

```python
from pathlib import Path
import pllm

fixture = Path("schemas/fixtures/compile-request.valid.json")
plan = pllm.compile(fixture.read_text(encoding="utf-8"))
report = pllm.benchmark(
    plan,
    weights=bytes([1, 2, 3, 4, 5, 6]),
    input=b"".join(value.to_bytes(4, "little") for value in (7, 8, 9, 10, 11, 12)),
    id="docs-region",
    privacy_cohort="masked-linear",
    numeric_cohort="wrap32",
    environment={"fixture": fixture.name},
    warmups=1,
    repetitions=3,
)
document = report.to_dict()
assert document["scope"] == "region"
assert document["plan_lock_digest"] == plan.plan_lock_digest
assert len(document["samples"]) == 3
```

API: [`pllm.compile`, `pllm.benchmark`](/sdk/reference/python/pllm/#objects-and-signatures)

This checked `EvidenceReport` is distinct from the loopback diagnostic JSON.
