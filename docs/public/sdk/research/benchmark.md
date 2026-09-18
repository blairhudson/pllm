# Benchmark

Run scoped Python benchmarks and understand what the development dashboard does not measure.

[View canonical HTML](https://pllm.run/sdk/research/benchmark/)

Document ID: `pllm.docs.measure.benchmark`  
Release: `0.1.0`  
Build: `sha256:27f84427615190d0e8c08970a963d10d2ec7d1c4bb9a2db6446fba32419c8ada`  
Source hash: `sha256:662d0c074f84cef4751f69f07f5050d18a4b2ac2ea1c67c4d034704dc0327120`

`pllm.benchmark(...)` measures supported native compiled regions and returns immutable `EvidenceReport`. It requires explicit plan, immutable weights/input bytes, IDs, privacy/numeric cohorts, and environment. Warmups, repetitions, threads, SIMD, failures, and oracle comparison remain report data.

`pllm.deployment_benchmark(request)` validates supplied observations; it does not
prove their origin. The separate
[`pllm benchmark run`](/cli/reference/benchmark/run/) command exercises the real
client, preparation, and inference roles on loopback:

```bash
pllm benchmark run \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompt-file prompt.txt \
  --output benchmark.json
```

This headless command writes a sanitized diagnostic report and shuts down all
roles. It does not create a canonical benchmark evidence record. Use
[`pllm dev dashboard`](/cli/reference/dev/dashboard/) only when you want the
interactive development view. See [research evidence](/research/evidence/) for
the records and scope required to support a claim.

Repeat `--experiment PATH` to run and compare immutable Pipeline selections from
multiple Experiment configurations. The runner records each configuration and
Pipeline digest, suppresses rankings for unmatched workloads, and can export the
lowest-median-full-latency configuration with `--save-best PATH`. Reusable
candidates and measured local winners are kept in `examples/benchmarks/`.
These targets may be declarative JSON/YAML files or explicit SDK objects such as
`examples/benchmarks/qwen_prepared.py:cpu_4`; Python targets require
`--trust-python` because their module is executed during resolution.

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

This runs the native compiled region against its scalar oracle. It is separate
from the loopback diagnostic JSON described above.
