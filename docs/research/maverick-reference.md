# Maverick: full-width RAA reference and admission gates

PLLM independently implements a bounded, in-process version of Maverick's
Protocol 3 algebra. The public code construction follows Appendix B; fresh
private sparse vectors follow the exact fixed-weight distribution in Appendix A.
This is a **component reference**. Its code-distance and dual-LPN assumptions
remain uncertified, so it cannot be selected by an SDK Experiment or ranked as
private Qwen inference.

Source: [Maverick v1](https://arxiv.org/abs/2609.10264v1), Sections 5.2, 6, 7.1
and Appendices A/B. The pinned PDF hash is
`c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`.
No upstream implementation is built, linked or vendored.

## Implemented contract

For a public signed-i8 matrix `M` with output/input dimensions `m,n`:

1. Two separately seeded, public rate-1/8 RAA codes produce `P = M Gx` and
   `Q = Gy^T M` over BabyBear, `q = 2,013,265,921`.
2. The client samples a fresh sparse `e` with uniformly chosen distinct positions
   and independent uniform nonzero field coefficients using operating-system
   randomness. It forms `x + Gx e` and retains `P e` locally.
3. The evaluator receives only the masked input and uses the existing native
   modular matrix kernel. The reference has no provider transport.
4. After the complete claimed response is fixed, the client samples two secret
   weight-41 verification challenges. Both checks run before a combined decision.
   A passing reference check permits subtracting the correction and recovering
   the exact centered signed integer result.

The native query handle is non-cloneable and consumed on every finish path,
including malformed responses and wrong matrix/code bindings. Cancellation drops
and zeroizes retained correction material. Private sparse vectors and transpose
scratch are also zeroized. At most eight queries can remain live; at most 65,536
issuance attempts belong to one reference object. This is Rust ownership, not an
authenticated distributed replay protocol or a process-wide security budget.

Dimensions are bounded at 16,384. Preprocessing admits the combined `P`, `Q`,
public-code payload and maximum live-query payload under 256 MiB before allocating
either code or matrix. Caller weights, temporary work, allocator overhead and OS
cache are separate. The reproduction driver additionally requires 1 GiB safe
headroom beyond the ordinary host reserve and monitors pressure and swap growth.

## Source/security findings

- **Full code distance remains unknown.** Appendix B's quoted finite-field
  failure estimate specifies `q > 2^31`; the paper's Section 7.1 and this reference
  use BabyBear, which is smaller. That particular statement cannot directly
  certify this instantiation. This is a missing applicability argument, not a
  demonstrated attack on Maverick.
- **Privacy parameters are conditional.** Equation (1) gives a linear-test
  heuristic using a target relative distance of one half. It is not a complete
  concrete-security analysis against every attack or a measured privacy result.
- **Code generation differs in representation.** Public SHA-256 seeds expand
  permutations/scales, with separate privacy and verification domains. No
  certificate connects this seeded ensemble to the cited uniform-code bound.
- **Verification parameters differ conservatively.** The paper's evaluation uses
  sparsity 31 with two checks. This reference retains sparsity 41 and two checks.
  Neither choice removes the missing distance/failure bound. Successful corruption
  tests are evidence of rejection behavior, not a soundness proof.
- **The integer boundary is narrower than a decoder.** Signed-i8 matrix products
  fit the centered field at every admitted width. There is no imported checkpoint
  value, bias, activation-scale, logit, KV or generation-quality comparison.
- **Batch verification is separate.** Protocol 2 adds another challenge/response
  round and an auxiliary encoded matrix. Neither its state nor transport is
  included in this direct Protocol 3 reference.

## Replay and evidence

```sh
uv run python -m benchmarks.research.maverick_reference \
  --output docs/evidence/research-maverick-reference-reproduction.json
cargo test -p pllm-core
```

The Python driver resolves the pinned Qwen2.5-0.5B source configuration, lowers
the ordinary compiled schedule, and derives every distinct remote stage shape.
It runs each admissible shape in a fresh native process using synthetic public
weights and eight signed-i8 queries. Shape failures are retained before any
weight allocation. The evidence records source hashes, full observations,
preprocessing payload, mask/check time, evaluator time and a one-thread clear
SIMD control. Times are isolated elapsed time, not full-response aggregate CPU
or 100/40 WAN measurements. Raw field input/output sizes are counts, not measured
network traffic. No reference point is inserted into the decoder frontier.

Evidence: `docs/evidence/research-maverick-reference-qwen25.json`.
Native implementation: `crates/pllm-core/src/coded_delegation_reference.rs`.
Replay configuration: `benchmarks/research/maverick_reference.py`.

### Recorded Qwen2.5-shaped screen

Each shape occurs in 24 compiler stages. The first two admitted shapes cover
48/96 stages; both MLP shapes reject before allocation. These are public synthetic
matrices at checkpoint-derived dimensions, not checkpoint numeric measurements.

| Output × input | P+Q MB | Reference outcome | Client ms/query | Evaluator ms/query | Clear SIMD ms/query |
| --- | ---: | --- | ---: | ---: | ---: |
| 896 × 896 | 51.38 | Exact outputs; changed replies reject | 1.188 | 0.064 | 0.031 |
| 1152 × 896 | 66.06 | Exact outputs; changed replies reject | 1.398 | 0.083 | 0.038 |
| 896 × 4864 | 278.92 | Over 256 MiB combined-payload bound | — | — | — |
| 9728 × 896 | 557.84 | Over 256 MiB combined-payload bound | — | — | — |

The admitted cases check 16,384 output values and six changed replies in total.
Their public code payload brings persistent client storage to 51.72/66.45 MB;
initial preprocessing takes 0.140/0.132 seconds. Client issuance plus checking
costs 38.7×/36.5× the corresponding clear matvec in these eight-query medians.
Each field reply/request pair counts 7,168/8,192 raw bytes, before framing.
Observed new swap is zero. These measurements expose costs of this independent,
untuned reference; they do not reproduce the paper's hardware or implementation.

## Next promotion gate

Establish a concrete code-distance/failure certificate and reviewed dual-LPN
parameters for the actual field and seeded code ensemble. A storage design must
also overcome the existing 22.90 GB `P+Q` lower bound for the 96 Qwen2.5 remote
stages. That bound already excludes batch auxiliaries and distribution.
Only then can authenticated role transport, full checkpoint parity and matched
private/verifiable response measurements justify an executable method.
