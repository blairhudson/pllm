# Frontier findings in the SDK

Historical probe cohorts remain in [the five-frontier report](../plans/hundredfold-frontier-probes.md).
Their timing and memory ratios do not become SDK measurements when an implementation is integrated.

## Exact recurrent ownership

The compiled semantic executor pairs exclusive `gated_delta_rule` and
`gated_delta_state_update` operations with identical inputs and attributes. It
computes both results once, updates the existing recurrent-state array, and
shares one bounded scratch slab across layers. Additional state consumers retain
the original copying implementation. Pairing uses semantic declarations, never
model-family names. Admission charges the scratch alongside the existing
conservative hybrid-decoder allocation bound.

Float32 normalization retains the original complete tensor layout: slicing
strided projections into rows before reduction can change the last bits. Generated
hybrid checkpoint controls compare every checked logit, convolution/recurrent
state and active KV bit against the original executor. The upstream Torch oracle,
ordinary prepared SDK and local gateway paths also remain covered. Cancellation
burns mutated state and scratch; close/reset zero the owned arrays.

`pllm.state.OwnedGatedDeltaRecurrence` exposes the same bounded local arithmetic
for caller-owned projected inputs, with independent output/snapshot arrays,
explicit token limits, context-manager retirement and no copyable live owner.
`BenchmarkTarget.recurrence(experiment, config=..., owner="owned")` compares it
with `owner="copy_reference"` through `benchmark_suite`. The target derives
geometry from the source plan, uses synthetic projected inputs, checks every
output and final state bit, and separates retained state, scratch and process RSS.
Its timed scope includes construction, transitions, output hashing and retirement.
Checkpoint values, full-response costs and role transport require separate
measurements.

```python
from pllm.metrics import BenchmarkCase, BenchmarkSuite, BenchmarkTarget, Sampling, benchmark_suite

target = BenchmarkTarget.recurrence(experiment, tokens=64)
result = benchmark_suite(BenchmarkSuite((
    BenchmarkCase("copy", target.with_params(owner="copy_reference")),
    BenchmarkCase("owned", target),
), Sampling(processes=3)))
```

The experiment must select an admitted gated-delta source and a workload bound
covering the requested transitions. No constant-state model replacement or quality
equivalence follows from this ownership change.
