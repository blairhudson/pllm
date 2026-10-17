# Explicit GKM decoder capacity

The full-share GKM SDK composition has a separately selected resource policy:
`OpenWeightVerifiableOutsourcing(capacity="decoder")`. The default `"capsule"`
policy retains its existing limits and serialized identity. Capacity is a local,
compiler-bound choice. A worker cannot select it from an incoming frame.

## Contract

Both policies use the original signed-i16/Q11 source mapping, checked ring64
products, original float32 lookup rows, client-local float32 operators, full head
and full KV. Each worker receives only its own shares. The client withholds all
model-derived output until the final native verification accepts the request.
The larger policy permits only full shares; seeded transport remains unavailable.

Native limits precede source value reads and allocations:

| Owner or work | Decoder ceiling |
| --- | ---: |
| Matrix rows / vocabulary | 262,144 |
| Matrix columns / lookup width | 16,384 |
| Matrix elements | 268,435,456 |
| Matrix stages | 256 |
| Complete window operations | 65,536 |
| Encoded sources, including bounds | 2 GiB |
| Native window payload | 128 MiB |
| Publication reservation pool | 4 GiB per process |
| Additional wire reservation pool | 4 GiB per process |
| Compiler source-import reservation pool | 4 GiB per process |
| Encoded publication | 32 MiB |
| Operation frame | 4 MiB |
| Source import frame | 64 KiB |
| Compiler input rows / decode steps | 512 / 64 |
| Complete-request linear MACs per worker | 137,438,953,472 |

These are intersecting limits, not a promise that every combination fits. The
compiler also bounds source, tensor and execution ownership. Input roots,
grouped projection backing arrays, retained views, numeric scratch, KV growth
and final state copies are charged. Metadata/tokenizer and outer frame copies
have separate allowances. Ordinary host admission adds process costs and its
25% allocation margin before starting the role graph.

Decoder admission also derives `accepted_result_bytes` from every returned logit
vector, the complete active KV state and a 16 KiB acceptance-receipt allowance.
The client checks this native bound before source loading or remote execution,
then validates the actual result geometry and byte count before delivery. The
Capsule policy keeps its original 16 MiB result ceiling. Pinned Qwen's 150/288-row,
31-decode-step results need 23,896,064/27,287,552 bytes before the receipt; both
therefore require the explicit Decoder policy.

Failed client execution clears tentative arrays and nested exception frames before
request admission is released. Cause, context and exception-group chains retain
their diagnostic structure, without retaining the failed tensor owners. Cleanup
failures receive the same treatment.

Worker replies retain a share of their request's host reservation. Cancellation,
expiry and completion release the native root; the final reply releases the charge
only after its body is dropped. Successful sends, interrupted sends, construction
failures and unsent response destruction follow the same order. Retained replies
therefore continue to count against admission of later requests. Server concurrency
bounds the live reply count; transport and caller-created copies remain separate.

The native publication and additional-wire pools are independent, with a combined
logical ceiling of 8 GiB. The compiler's transient source-import pool is another
independent 4 GiB allowance. These limits do not establish a 4 GiB whole-process
memory guarantee. Capsule publication/wire reservations also consume their
corresponding shared Decoder pools. Caller buffers, allocator overhead and process
RSS require outer accounting. Fixed-size native `Arc` allocation retains its
documented abort-on-allocation-failure exception.

## Pinned Qwen source geometry

Metadata-only checks use Qwen2.5-0.5B-Instruct revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Compiler grouping yields 97 matrices,
290 source tensors and 121 local tensors. The tied lookup/full head remains two
different numeric owners: original float32 lookup rows and signed-i16 head
weights.

| Ownership | Bytes |
| --- | ---: |
| Native weights and original lookup table | 1,532,461,056 |
| Sources including bounds | 1,537,566,720 |
| Imported sources including local tensors | 1,537,852,928 |
| Window accumulators | 20,422,656 |
| Maximum window envelope | 37,008,608 |
| Encoded-publication upper bound | 21,334,232 |

One complete publication reservation is 2,161,008,640 bytes under the inspected
64-bit release layout, or 2,169,026,560 bytes under debug. Two complete publication
reservations exceed the 4 GiB publication pool. Importing an exported publication
while retaining another complete publication therefore rejects. The SDK client
publishes once and retains that owner; each separate worker imports its own
authenticated publication and transfers its streamed source owners without a
second complete source copy. An in-process reference that constructs both
workers has a different ownership overlap.

## Verification status

Compiler tests admit the pinned source geometry at `(input rows, decode steps)`
of `(3,1)`, `(150,31)` and `(288,31)`. They independently reconcile inventory and
source owners, reject implicit capacity changes and malformed envelopes, and
check a streamed 257-row head against exact integer equations. Substituted
capacity, shortened schedules and corrupted final results reject.

A guarded cached wheel passes 640 installed Python and 134 native checks plus
Clippy. All 362 installed files match before and after testing. Six generated
Qwen2/Qwen3/Llama cases with 1,100 vocabulary entries preserve 20,376 complete
logit/KV values through two real worker processes, including tied and untied heads.
SDK, gateway, CLI, benchmark, capacity reconstruction and failure-lifetime checks
pass on that installed wheel. The build observes zero new swap growth.

A separate source-present Qwen metadata preflight at a 64-input/two-output bound
requires 10.83 GiB of host allocation against 8.11 GiB of safe headroom. It rejects
before tensor-value decoding or worker launch. This historical attempt establishes
no execution or resource measurement.
Original receipts, negative controls and read-only replay:
[Decoder-capacity archive](../../paper/artifacts/gkm-sdk/decoder-capacity.md).

A subsequent source-present retry passes fresh admission and completes four input
tokens plus two verified outputs through separate local workers. The original
full-width source, publication and runtime numeric checks accept this workload.
One request takes 143.87 seconds and 87.23 MB of covered all-link bodies. Client
peak RSS is 1.78 GiB; sampled simultaneous client/worker RSS is 3.79 GiB, with zero
observed swap growth. All 362 installed files and seven checkpoint files match
their before/after receipts. Independent complete-logit/KV comparison, source-model
quality, broader workloads and matched performance remain open. Buffered final
acceptance supplies no decode-throughput measurement. Exact costs, scope and replay:
[First Qwen response](../../paper/artifacts/gkm-sdk/qwen-response.md).

Native APIs and focused replay commands:
[`wire-api.md`](../../crates/pllm-core/src/gkm/wire-api.md).
Previous installed SDK and streaming evidence retains its original Capsule scope:
[`gkm-sdk-reference.md`](gkm-sdk-reference.md) and
[the streaming-import archive](../../paper/artifacts/gkm-sdk/source-stream.md).
