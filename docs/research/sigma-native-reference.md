# Bounded SIGMA native references

Source arithmetic: [SIGMA](https://petsymposium.org/popets/2024/popets-2024-0107.php),
§3.2.1 masked multiplication and §4.1.1 matrix multiplication. The reference
independently implements these equations; upstream code is not a dependency.

The matrix reference is a **single-process, coupled two-share harness**, not a
SIGMA decoder, independently hosted party implementation, SDK component or
security qualification. The approved GPU-derived DPF port is a separate primitive
under decision D16; it is not used by this matrix product.

## Arithmetic and reuse

`pllm_garble::masked_wire_reference::matrix` extends the scalar negative-mask
product algebra without scalar difference openings. For caller-supplied masks
`RX`, `RW`, `RY`, issuance shares:

```text
A = -RX
B = -RW
C = RX·RW + RY
Y_party = A_party·W_hat + X_hat·B_party + C_party
          + [party == 1] X_hat·W_hat
```

With `X_hat = X + RX` and `W_hat = W + RW`, reconstruction gives the complete
`XW + RY mod 2^n`. Widths 1–64 and input/output or output/input coefficient
layouts are explicit. No intermediate floor, rescaling, clipping, bias or
float conversion is supplied. This is not a complete private linear layer.

Scalar products retain their existing expression, random draw order and encoding.
Resident private contractions reuse the allocation-free ring64 traversal in
`pllm_core::streamed_matrix`; private coefficients never enter a public-file
owner. Existing streamed public products keep their arithmetic order and layout.

## Ownership and admission

Each input, weight and output matrix is limited to 256 elements. Complete work
is priced as six contractions and rejects above `2^20` MACs. Per-party logical
key size is `n*(a+b+c)` bits, where `a`, `b`, `c` are the three matrix sizes;
physical storage uses u64 elements rather than packed keys.

The shared logical budget reserves `8*(6a+6b+7c)` payload bytes plus typed owner
metadata, 16 KiB stack allowance and per-allocation allowance under the existing
64 MiB ceiling. It includes admitted caller masks, masked operands and retained
clear inputs as well as native overlap. Excess caller capacities, aliases and
unrelated allocations remain additional owners. These allowances are not process
RSS, allocator or stack guarantees.

Only one operation is live process-wide. Borrowed non-Send handles retain the
complete reservation until the operation owner drops, after output buffers.
Duplicate imports, malformed bindings, cancellation and dropped unfinished
handles retire the coupled operation. Fallible exact-capacity allocations and
zeroizing buffers preserve erase/deallocate-before-refund ordering.

Issuance freshly samples correlation splits and an operation nonce. It accepts
the masks from its caller: **it does not sample them or certify that the caller
has not reused them**. There is no authenticated transport, replay persistence or
malicious-output verification. Both materials reside in one operation owner;
the evaluator API is not an isolated worker security boundary. Production
integration needs separately owned party material, context-bound import/export,
fresh wire-mask ownership and complete transport/failure lifecycle.

## Separate GPU-derived DPF primitive

Decision D16's private `compact_dpf::gpu127` module implements the approved
CPU port under `pllm.compact-dpf.source-gpu127.v1`. Widths 8–63 retain 127-bit
secret seeds, randomized complementary root controls and two complete
AES-expanded terminal correction words. AES plaintext tags are 0 and 2 with
the pinned GPU word/byte mapping; existing compact-DPF defaults stay unchanged.

For `d = n-7`, each logical key costs `129*d+255` bits. The distinct PLLM codec
adds an 18-byte header and canonical zero tail padding: 66–953 encoded bytes.
Paired issuance uses `4*d` AES blocks; point and unsigned strict-less evaluation
each use `d`. Full expansion admits only widths 8–10, at most 1,024 output bytes
and 14 AES blocks per key. Fixed inline key owners and stack/AES workspace remain
additional to encoded bytes; allocation limits do not bound whole-process memory.

Issuance uses one paired OS-random draw. Encoding consumes its owner; internal
evaluation borrows it for table scans. This private algebra does not enforce
authenticated session binding or one-use transport: those remain outer-runtime
duties. Same-profile payload substitution is not detected by its canonical codec.
There is no upstream wire-byte equivalence, GPU performance or security-proof
claim. D17 separately approves complete-output-ring arithmetic masks; its source
audit remains in `paper/reproduction-contracts/sigma-arithmetic-masks.md`.

## Bounded TR / GapARS / DReLU adaptation

`masked_wire_reference::arithmetic` uses the distinct
`pllm.sigma.gpu127-full-ring-masks.v1` identity and D16 GPU127 keys. It is not
the pinned CPU implementation or the existing Q12 gated-block profile; defaults
remain unchanged. `rescale.rs` factors shared masked-floor equations, not a new
rounding rule. Native tests below qualify bounded arithmetic and ownership,
not a composed protocol or decoder.

Admission covers 1–256 lanes and input widths `3 <= n <= 64`. TR admits
`0 < s < n`, returning exact unsigned floor modulo `2^(n-s)` (also the reduced-ring
representation of signed floor). GapARS admits `0 < s < n-1`, returning signed
floor in the original ring **conditional on the unwrapped private input promise
`-2^(n-2) <= x < 2^(n-2)`**. That promise is not enforced or proved by this owner.
DReLU returns the XOR-masked bit `[signed x >= 0]`, including zero.

Comparison width is `c=max(s,8)` for TR/GapARS and `c=max(n-1,8)` for DReLU.
Comparisons below eight bits publicly zero-extend threshold and query into the
GPU127 domain; they do not truncate or change the comparison. Issuance samples
fresh uniform masks over all `n-s` TR output bits, all `n` GapARS output bits,
or one DReLU bit. Caller-supplied input-mask freshness remains a caller obligation.

For `L` lanes, `d=c-7` and output width `m`, checked costs are:

| Cost | TR | GapARS | DReLU |
| --- | --- | --- | --- |
| Logical key bits per party | `L*(129*d+255+2*m)` | `L*(129*d+255+3*m)` | `L*(129*d+256)` |
| RNG byte requests (including nonce, roots, masks and splits) | `32+64*L` | `32+72*L` | `32+48*L` |
| Frames per party | 2 | 2 | 1 |

Each operation requests `m*L` output-mask bits; RNG request bytes are not an
entropy/security claim. Paired issuance costs `4*d*L` AES blocks; evaluation costs
`d*L` per party. Each frame is exactly `76+8*L` bytes, totaling
`2*frames_per_party*(76+8*L)` across both directions. Party material physically
stores `L*sizeof(Lane)` bytes, including inline keys and three u64 constants per
lane even where unused; logical key bits are not encoded delivery bytes.

`resources()` separately prices dealer, each party and combined logical live
reservations before randomness/allocation, under the shared 64 MiB ceiling. With
`F=76+8*L+sizeof(Frame)+64`, `O=8*L+sizeof(MaskedOutput)+64`, party reservation is
`P=sizeof(Party)+L*sizeof(Lane)+24*L+3*64+frames_per_party*F+O+8192`;
dealer reservation is `D=sizeof(OutputMasks)+16*L+64`, combined `2*P+D`.
Borrowed input masks (`8*L`) and both masked-input views (`16*L`) are included
allowances, not transferred ownership; excess backing capacities and unrelated
owners remain additional. These are logical allowances, not RSS/allocator limits.

A local trusted research dealer sees input masks and returns a separate output-mask
owner plus two separate party-local owners, each holding only its own key and
constants. This differs from the coupled matrix owner. Frames bind public
plan/session/operation/wires/index, nonce, input digest, sender and phase, but are
not authenticated. Successful completion or a failed call consumes local execution
state; cancellation burns it. Orchestration must also retire the other party on
failure. Frames and outputs retain split leases independently; cancelled/consumed
party owners retain their remaining reservation until drop. Zeroizing owners
erase/deallocate before lease refund. No material codec, authenticated material
delivery or peer transport, persistent replay protection, malicious-output
verification, SDK integration, runnable SIGMA decoder or composed security proof
is supplied by this source slice.

## Correctness checks

Normal tests cover rectangular independent u128 equations, signed/wrapping
values, both layouts, tiny exhaustive scalar mask/share cases, legacy adapter
agreement, resource counters, malformed bindings and inputs, cancellation,
allocation/randomness failures and reservation retention. Replay with:

```sh
cargo test --release --offline -j 1 -p pllm-core --lib streamed_matrix::tests:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --lib masked_wire_reference:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --lib fusefss_backend:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --lib compact_dpf:: -- --test-threads=1
cargo test --release --offline -j 1 -p pllm-garble --test sigma_masked_floor_chain
```

The FuseFSS run checks the extracted shared logical allocator's existing users.
The DPF run adds independent AES wordwise/NIST/frozen OpenSSL vectors, complete
width-8/9 point and strict-less tables under both root phases, wide endpoints,
bounded expansion/work counters, canonical parsing and erasure/failure controls.
The complete `cargo test --release --offline -j 1 -p pllm-garble` run passes
413 unit tests and two external integration tests; eight opt-in exporters or
measurements remain ignored. Its fourteen arithmetic tests cover every legal
small-ring shift/input/mask, exact RNG requests and measured AES block counts,
64-bit boundaries, all-gate entropy/allocation failures and retained-owner leases.
The external chain passes common masked outputs and dealer-only masks through
TR, GapARS and DReLU without opening intermediates to either evaluator, against
independent signed floor equations. It does not enforce general graph provenance
or establish matrix-chain execution. Shared-core traversal's eight tests and
280 independent source-terminal/arithmetic-mask negative controls also pass.
Filtered native reruns overlap the complete suite, not additional distinct checks.
Primitive parity supplies no trained-checkpoint, quality, WAN, performance or
whole-client memory result. Complete source-specific truncation, nonlinear
programs, compiler scheduling and ordinary SDK/gateway/benchmark execution remain
required for runnable SIGMA.
