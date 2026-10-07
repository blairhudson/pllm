# Complete LogRow product reference

`pllm.nonlinear.LogRowGatedMultiplyQ9` evaluates the complete bounded
`SiLU(gate) × up` operator through an embedded LogRow lookup and a consuming
half-gates circuit. Intermediate lookup labels feed arithmetic directly; only
the trusted client decodes the final Q9 result. The installed operator and
replay configuration are separate from compiler or provider execution.

## Sources and adaptation

- LogRow, <https://eprint.iacr.org/2024/369>: masked one-hot row selection and
  complete masked table bits, using the earlier two-row setup already documented
  in [the nonlinear frontier](nonlinear-frontier.md).
- Two Halves Make a Whole, <https://eprint.iacr.org/2014/756>: free-XOR half-gates
  arithmetic. Source PDF SHA-256:
  `789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083`.

The Boolean builder admits an embedded lookup on its own wire encodings and
delta. The lookup keeps a fresh hash-domain issuance identifier; its output
encodings become ordinary circuit wires. A fixed public instruction schedule
preserves hidden input-dependent row addresses. Cross-circuit inputs and
decoders reject, and consuming Rust APIs prevent a second evaluation. Circuit
material is not serialized or authenticated between distributed roles.

Index reduction, signed multiplication and exact ties-even shifts are PLLM
composition work, not LogRow's optimized complete system. This CPU reference
does not implement OT for independently supplied evaluator inputs; the trusted
client supplies both private operands and garbles their evaluation.

## Exact common numeric identity

The implementation uses the **unchanged uncompressed Curl Q18/Q9 profile**:
gate/up ranges ±48/±128, a 4,096-entry SiLU table on `[-64,64)` at spacing
1/32, Q9 table values and output. The index is the floor-reduced, public-offset
gate; protected wire selection implements that exact reduction. Up and final
product use exact ties-even division by 512. Public bounds prove all chosen
signed widths; invalid inputs reject rather than clip.

The independent Python integer oracle, native clear profile and protected
operator agree. That numeric identity allows referencing the existing
[twelve-prompt Qwen diagnostic](../evidence/research-wavelet-gated-quality-qwen25.json)
without relabelling it as a protected decoder run: 10/12 prefill selections,
35/36 same-token decode selections and 10/12 eight-token trajectories match;
no complete logits or KV states do. Worst absolute logit error is 2.9478.
This is the existing cohort, not a new quality measurement. Float conversion
and enforcement of the declared ranges remain client-local.

## Complete resource accounting

Each lane receives a fresh circuit. The runner retires it before issuing the
next lane and admits 1–256 lanes with 1–16 samples. Its 32,768-wire ceiling plus
bounded lookup workspace preflights a conservative 16 MiB serial-live allocation
bound. This is neither peak RSS nor the material total across all lanes.

Reports charge separately:

- garbler-client input and output encodings;
- evaluator one-hot labels, row-lookup ciphertexts and all masked table bits;
- half-gates ciphertexts for up rescaling, multiplication and output rescaling;
- constant labels, public instruction metadata and client-boundary label estimates.

There are no evaluator-to-evaluator rounds in this layout. Zero peer-frame
bytes therefore cannot be compared to Curl's four rounds as if LogRow had zero
communication. Evaluator material and input/output labels require delivery in
any future transport; retained garbler encodings are client storage, not wire.
Public instruction metadata may be reusable in a future transport, but that
transport and its serialization do not exist. Complete physical wire remains
unknown. All fresh secret material is charged per lane.

## Reproduction

[Matched raw report](../evidence/research-logrow-product-cost.json): six fresh
processes check **1,926 exact outputs**. The 256-lane, three-sample medians are:

| Method | Retained material by role | Issuance | Local online | Peer frames |
| --- | ---: | ---: | ---: | ---: |
| Uncompressed Curl | 5,079,040 B per worker | 134.60 ms | 202.03 ms | 46,304 B |
| LogRow complete | 614,400 B client; 15,560,704 B evaluator | 6,297.79 ms | 4,423.94 ms | 0 B |

These local timings include sharing/encoding and reconstruction/decoding in
their respective execution boundaries. Role graphs differ, so the zero-peer
column is not a complete communication comparison.

At the hypothetical historical 39+8 geometry, LogRow requires **12.89 GB** of
fresh client encodings and **326.40 GB** of evaluator material. The latter
includes 62.46 GB for lookup, **242.29 GB for multiplication**, 19.59 GB for
up/output rescaling and 2.06 GB of constant labels. It additionally estimates
6.79 GB of client-boundary labels/framing and 378.01 GB of public instruction
metadata across all issued circuits. This does not require retaining every
lane at once, nor establish that a future transport must retransmit every
public instruction. The current complete block fails the material-cost screen.

The matched uncompressed Curl control reproduces 213.08 GB of both-worker keys
and 971.27 MB of peer frames under its different two-worker contract. The
comparison demonstrates why lookup-only ciphertext counts are insufficient;
it does not reject the papers' other constructions or optimizations.

```sh
uv run python -m benchmarks.research.logrow_product \
  --output docs/evidence/research-logrow-product-reproduction.json
uv run python examples/benchmarks/logrow_product.py --lanes 64 --samples 3 \
  --output docs/evidence/research-logrow-product-sdk-reproduction.json
cargo test --locked --release -p pllm-garble logrow_product
uv run pytest tests/test_logrow_product.py
```

The matched driver creates separate processes for LogRow and uncompressed Curl
at one, 64 and 256 lanes, with three one-use samples each. It verifies the common
numeric profile and complete quality record before projecting the historical
39+8-token, 24-layer, 4,864-width geometry. The quality reference is bound by
its file hash. Neither the projection nor the in-process timing is a protected
checkpoint response, WAN measurement or client-memory improvement claim.

## Admission

This operator exposes complete negative or positive measurements for research.
The common profile already fails the held-out checkpoint fidelity screen.
Complete material cost, independent review, tensor resource admission and role
transport remain necessary before any decoder component can execute it.
