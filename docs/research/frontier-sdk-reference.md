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

## Same-client encrypted SIMD

`PackedEncryptedLinear(batch_size=8)` interleaves independent vectors from the
same client/key context in the existing CKKS operator. Geometry, public weights,
input range and the float64 oracle are unchanged. Rotation strides and key
requests bind the batch layout; a partial final batch pays for one complete
zero-padded ciphertext. Default configuration and numeric identity remain stable.

`benchmark_operator` and `BenchmarkTarget.operator` report actual query counts,
setup delivery, allocation estimates and separate issuance/evaluation costs.
Tests cover rectangular and scalar matrices, both rotation-key policies, partial
batches, replay and component serialization. This exposes the useful SIMD
primitive; encrypted residual decoding, multi-client keys and arrival waiting
retain their separate contracts and evidence.

## Explicit checkpoint sessions

`pllm.state.CheckpointSession` owns an already-bound compiled full-KV decoder,
caller-supplied linear executor, AES-GCM key, and private directory. Select
`storage="layer_pages"` or `"context_graph"`. `append()` executes explicit token
IDs; `advance()` executes only the currently pending greedy token. The caller
retains the immutable `CheckpointHead` and key, and supplies both on reopening.
The latest fully committed head stays accessible after cancellation/failure.
Live owners cannot be copied or pickled; context exit retires runtime plaintext.

This low-level state API preserves the native continuation and numeric admission
of the underlying owner. It is not an `OpenAI.conversation` persistence mechanism
and does not restore provider credentials, leases or correction inventories.
Disk pages are immutable and shared; abandoned branches and generations remain
charged rather than being silently deleted.

`BenchmarkTarget.checkpoint(experiment, storage="context_graph")` exercises the
installed owner, closes it, restores in a fresh interpreter, and checks all logits,
active KV and pending-token provenance against uninterrupted execution. The
bounded target requires the W8A8 `prefix_f32` baseline, at most 16 MiB of source
float32 weights, and a 768 MiB admission budget. SDK sessions retain their own
underlying geometry limits. Timings include original publication, child import,
restoration, continuation and complete KV audit; original source loading and
issuer/child lifetime peaks remain separate. Reports persist neither tokens,
keys, heads nor plaintext frames.

## Explicit local Preparation batches

`pllm.preparation.CorrectionBatch` accepts one immutable native `CompiledMatrix`
and independent `PreparationRequest` records. Requests must bind identical source,
stage, matrix digest, quantization and ring geometry. The owner admits at most
eight inventories, 128 working rows, 64 MiB of correction files and eight billion
integer products before issuing material. CPU weights, row buffers, digit8
workspace and any GPU snapshot are separately counted.

`add()` returns a non-copyable `CorrectionTicket`. `execute()` carries bounded
row windows across fresh native mask cursors, preserves original ring-packed
corrections, authenticates local row attribution, and seals only after all rows
arrive. `take_rows()` consumes the resulting ordered stream once. Malformed
requests, substituted/replayed rows, cancellation or kernel errors close every
owned cursor and file. SDK ownership retains no root seeds; callers still own
their input requests. Temporary Python mask buffers have ordinary array lifetimes.

Kernels are explicit: `native`, `digit8`, `metal` or threshold-based `adaptive`.
There is no autonomous batching policy, arrival queue, HTTP inventory protocol or
change to live provider defaults. `BenchmarkTarget.preparation_batch()` compares
group sizes and row windows through the installed SDK owner; the native digit8
target spelling remains `kernel="digit8_native"`. It charges factory/GPU setup,
fresh masks, local attribution, sealing, export and retirement. Native matrix
import and independent signed-integer audit costs are reported separately.
Fresh benchmark samples use fresh roots and never execute their oracle copies.

## Replay

`examples/benchmarks/frontier_sdk.py` constructs ordinary immutable targets for
recurrence, checkpoint storage and Preparation batches. It runs fresh processes
through `benchmark_suite`, preserves shared-cohort comparison rules, and writes
the canonical report requested on the command line. The same target factories
work with `FrontierStudy` variants and confirmation.

Installed-wheel regression coverage totals 217 passing checks across the broad
run and focused component-registry rerun; two optional real-Gemma checks are
skipped without their checkpoint path. The public example completes all six
fresh-process cases with this exact invocation:

```sh
target/frontier-sdk-env/bin/python examples/benchmarks/frontier_sdk.py \
  --output target/frontier-sdk-example-final.json
```

Use a fresh output path for another run. Documentation regeneration and the
production docs build pass. These checks establish installed functionality and
cleanup, not a new performance cohort.

These installed APIs do not inherit the historical probes' speed/RSS ratios.
Whole-decoder real-checkpoint recurrence measurements, independent live inventory
delivery and complete-response benefits remain separate measurement gates.
