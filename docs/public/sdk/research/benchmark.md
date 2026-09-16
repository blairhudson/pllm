# Benchmark

Run scoped Python benchmarks and understand what the development dashboard does not measure.

[View canonical HTML](https://pllm.run/sdk/research/benchmark/)

Document ID: `pllm.docs.measure.benchmark`  
Release: `0.1.0`  
Build: `sha256:88685675b040f8400eed3855030abc8a7816f2d3688fae13cf86f2739853697d`  
Source hash: `sha256:d361426233eb578fec74fd0bcff69e32a3f1a555f63aaae0e76dda8964759916`

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
