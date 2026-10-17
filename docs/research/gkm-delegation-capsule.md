# Source-faithful delegation capsule

The Rust `pllm_core::gkm` module implements a bounded, in-process slice of
Gupta–Katz–Miers private and verifiable linear delegation. It preserves the
selected source arithmetic, dense byte lookup and final verification barrier.
It is not a compiled decoder, distributed service or performance result.

The governing source and page locators are in the
[delegation contract](../../paper/methods/delegation-contracts.md). The source
PDF has SHA-256
`e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d`.
The [decision record](../../paper/methods/DECISIONS.md) approves Apple-M CPU-first
development and a separately qualified Qwen2.5-0.5B common-model track.

## Arithmetic

The encoder multiplies a finite real value by 2048 and rounds to the nearest
integer. Halfway cases use ties-to-even. This completes the paper's unspecified
tie rule; it does not reproduce the author's truncating integer casts.
Unrepresentable inputs reject instead of clipping.

Immutable weights use signed i16. Private inputs use signed i64 representatives,
with two fresh additive shares in the ring modulo `2^64`. Each worker's products
wrap in that ring. The client reconstructs the signed integer result before
conversion to float32 and division by `2048^2`.

For each output row, public input bounds must satisfy

```text
sum_j abs(weight[i, j]) * input_bound[j] < 2^63.
```

The inequality permits wrapped share products while excluding overflow of the
true signed result. It is not evidence that an imported model's activations obey
the declared bounds. The existing W8A8/ring32 model path has a different numeric
identity.

## Dense private lookup

The client splits a dense one-hot selector into two uniform XOR shares. Each
worker XORs selected rows of the original little-endian float32 byte table.
The client reconstructs the exact row bytes and checks the owner's pinned
SHA-256 digest before use. Signed-zero bits remain distinct. Non-finite table
entries reject.

This is the paper's dense construction. It does not select the installed packed
DPF lookup, a quantized token table or a clear token-indexed provider request.

## Verification and acceptance

The honest source owner publishes secp256k1 matrix commitments. For row scalars
`r`, the public values encode `R = g^r` and `S = g^(W^T r)`.
The client verifies the reconstructed signed relation, with negative integers
mapped consistently into the prime-order scalar field.

Fresh unbiased scalar challenges follow frozen responses. They combine the
scheduled claims within the acceptance window. Context, stage identity and
claim counts bind the transaction. A malformed, incomplete or incorrect window
cannot produce an accepted result.

The trusted client may inspect tentative values for dependent local computation.
That access does not authorize accepted text, reusable decoder state or committed
sampling state before the final check. Model state and output buffering are
future integration responsibilities. Smaller verification windows would change
the source batching contract and require a separate decision.

The primitive tests check arithmetic, byte recovery, group relations and consuming
ownership. They do not certify the complete cryptographic implementation or its
network deployment. Privacy requires two non-colluding workers and confidential,
authenticated transport. The current tests place both worker computations in one
process.

## Resource audit

The initial capsule admitted up to 256 rows or columns per matrix, 4,096 table
rows, 16 stages and 128 steps per window. These are bounded conformance limits,
not model-serving capacity. Source import, publication, clients, windows,
responses and scratch have distinct ownership.

Independent review found four defects in the initial ownership implementation:

| Defect | Required invariant |
| --- | --- |
| Independent client budgets exceeded one publication's live allowance. | All clients must share publication admission, with separate aggregate admission for live publications. |
| Response handles escaped pending-window budgets and survived cancellation uncharged. | Each pending role must have a bounded response slot whose lease survives until its buffer retires. |
| Fallible iterator collection grew an encoded temporary beyond its estimate. | Reserve the encoded length before filling and charge actual capacity. |
| Payload allocation could abort inside APIs that return `Result`. | Use fallible payload reservation and state the separate fixed-handle allocation contract. |

The first review found no arithmetic or verification bypass. Its independent
rerun passed 19 native tests and four compile-fail ownership checks. Those tests
did not detect the four resource defects.

The repair shares eight live window leases across all clients of a publication.
At most 64 client handles share that owner. Complete publication envelopes reserve
against one 16 MiB process-local payload allowance. Response constructors require
a pending role slot; duplicate or malformed responses poison the transaction even
if the caller ignores the error. Response and accepted-output buffers keep their
reservations until they retire.

Dynamic payloads use fallible exact reservation. The allocator regression checks
a 66,048-byte import temporary for a `256 × 129` matrix, replacing the earlier
131,072-byte temporary. Independent review repeated all 26 unit tests, one
allocation regression and five compile-fail tests. It closed all four findings
within the stated payload scope.

Separate allocator probes measured 2,104,840 bytes for eight live windows against
a 2,621,584-byte declaration. A ninth window rejected across distinct clients.
One 2,048-byte response remained charged while 600 duplicate attempts rejected.
Retained responses, including `Result` wrappers, and accepted outputs kept their
leases. Each tested publication reserved 5,632,180 bytes: two fit the shared
allowance, and a third rejected before allocation. Failure released reservations.
Three independent borrow checks rejected shares that outlived their owners.

The replay commands are:

```sh
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --lib gkm:: -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --test gkm_allocation -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline --locked -p pllm-core --doc gkm -- --test-threads=1
```

Fixed-size `Arc::new` handle allocation retains Rust's abort-on-OOM behavior.
Caller-owned standalone buffers, caller copies and allocator overhead are outside
the payload envelope. Logical admission is not a process-RSS measurement or
universal OOM recovery guarantee. These counters do not persist across restart.

## Next integration gates

1. Bind checkpoint, graph, matrix, table-byte and public-verification identities.
2. Establish public input bounds for each source numerical operation.
3. Separate client verification material from worker weight and table ownership.
4. Admit real model geometry with every live copy, scratch allocation and response.
5. Add the source's AES share compression and declared prefill batching.
6. Bind complete logits, KV, pending tokens and sampler state to atomic acceptance.
7. Add authenticated confidential transport, independent worker processes and
   failure retirement through the ordinary SDK and gateway.
8. Compare against the method's own clear numeric oracle before measuring costs.

Qwen import, original-model validation, SIMD/GPU acceleration, wire codecs,
restart-persistent budgets, whole-request quality and resource measurements are
not implemented by this capsule. No installed Pipeline or benchmark ranking is
enabled.
