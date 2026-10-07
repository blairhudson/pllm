# Complete Compact polynomial reference

`pllm.nonlinear.CompactGatedMultiplyQ12` evaluates a complete bounded
`SiLU(gate) × up` through private piece selection, polynomial arithmetic and
signed multiplication. Only the trusted garbler-client decodes the final result.
The immutable component uses the ordinary `benchmark_operator` and
`replay_operator_benchmark` contracts.

## Primary source and explicit adaptations

Compact, <https://arxiv.org/abs/2309.04664>, Sections 4.2.1–4.2.2 and 4.2.6,
motivates density-weighted error and Chebyshev interpolation. The pinned PDF
SHA-256 is `e4497a81055e60cca1c462dac5ae04b4504c5f3fc76e392e170cc32bf8fee86c`.
The half-gates construction comes from <https://eprint.iacr.org/2014/756>, PDF
SHA-256 `789ecaf8651b6dc8341d6033f951d76c0e6f4b04427cdde26cf91ecfb0af3083`.
No upstream implementation is built, linked or vendored.

The bounded fitter accepts 1,024 **public offline** histogram counts over
`[-8,8)`, adds one pseudocount per bin and refines the interval with the greatest
reduction in integrated weighted absolute error at fixed bin centers. Each split
is dyadic; coefficients interpolate the continuous SiLU at Chebyshev nodes.
Degree two or three and at most sixteen pieces are allowed. The coordinate
power-basis conversion precedes coefficient quantization.

This replaces the paper's batch-normalization-based normal density, threshold
search and annealed configuration search. Q12 arithmetic, explicit ±8 tails,
fixed profiles and the Boolean backend are also adaptations. In particular,
the reference does **not** reproduce Compact's complete MPC system or its costs.
Qwen's RMSNorm does not justify importing a batch-normalization distribution.
Negative costs of this Boolean implementation do not reject other paper layouts.

## Exact arithmetic and private execution

The gate/up inputs are signed Q18 in fixed public ranges ±48/±128. Values outside
those ranges reject rather than saturate. Each input is rescaled to Q12 using
exact ties-to-even division by 64. Gates at or below −8 use zero; gates at or
above +8 use the identity. Between the tails:

1. Private comparisons select one piece and its public coefficient words.
2. An exact Q20 normalized coordinate follows from that dyadic interval.
3. Horner multiplication/addition rounds each Q20 product ties-to-even.
4. The resulting SiLU is rounded to Q12, multiplied by the rescaled up value,
   then rounded once more to Q12.

Every selected coordinate has magnitude at most one. A public coefficient-sum
bound includes intermediate rounding and the final half-unit, proving the
40-bit Horner accumulator, signed 19-bit activation and signed 26-bit result.
The implementation retains the full rounding carry before narrowing under those
bounds. Unselected coordinates never enter polynomial evaluation. No plaintext
piece index, intermediate activation or product is returned to the evaluator.

The immutable profile digest binds source, degree, requested/actual pieces,
public calibration and exact coefficients. The independent Python scalar and
vector integer oracles check the full boundary and Horner computation. A clear
native mapping supports a separately scoped checkpoint diagnostic; it does not
execute the protected circuit across the model.

Each lane owns fresh half-gates randomness and a unique issuance identifier.
Inputs, program and decoder are consuming, non-cloneable Rust handles. Foreign
issuance and invalid inputs reject while consuming their handles. Only aggregate
reports and immutable public profiles cross the installed Python boundary.
Input range enforcement and float32 conversion remain client-local; there is
no independently supplied evaluator input or OT protocol.

## Complete ownership and admission

One circuit is retired before issuing the next lane. The fixed 131,072-wire
ceiling precedes secret allocation and prices a conservative 64 MiB live bound,
including growing instruction/encoding owners and scratch. It is an allocation
estimate, not measured RSS. At most 256 serial lanes and sixteen fresh samples
belong to one benchmark invocation.

Reports separately charge garbler-client input/output encodings, evaluator
half-gate ciphertexts, active constant labels, public profile/instruction data
and client-boundary labels/framing estimates. The ciphertext breakdown includes
input rescaling, private piece selection, polynomial arithmetic and final
tails/product/output rounding. The role subtotals conserve complete material.
There are no evaluator-peer frames; material and client-boundary delivery would
still be required by any future transport. Public instruction schedules might
be reusable, but no transport serialization or physical-wire claim exists.

Public fitting is timed once in the first sample. Native-call process CPU
excludes that public fitting, while issuance and local online timings remain
separate. A hypothetical whole-MLP projection prices every element and keeps
client-retained keys distinct from material delivered to an evaluator. It does
not assume all lanes must be resident simultaneously.

## Complete cost measurement

`docs/evidence/research-compact-product-cost.json` contains nine fresh processes:
three frozen profiles, each at 1/64/256 lanes with three fresh-material samples.
All 2,889 protected outputs match the independent integer oracle. The complete
MLP projection uses 24 layers × 4,864 channels × (39 prefill + 7 decode rows),
or 5,369,856 elements for eight generated outputs.

| Profile | Evaluator material | Polynomial arithmetic subtotal | Private piece-selection subtotal |
| --- | ---: | ---: | ---: |
| 4-piece quadratic | 1.971 TB | 1.342 TB | 155.17 GB |
| 8-piece cubic | 2.925 TB | 2.005 TB | 366.87 GB |
| 16-piece cubic | 3.423 TB | 2.005 TB | 737.35 GB |

Each additionally retains 13.40 GB of garbler-client encodings over the complete
hypothetical execution. The 390.07 GB tails/product/output-rounding subtotal and
25.78 GB input-rescaling subtotal are already included in evaluator material.
Active constant labels are also included. Estimated client-boundary labels and
framing add 7.05 GB; the uncompressed public instruction/profile estimate is
2.74–4.66 TB. All quantities use decimal units and conserve their role totals.

These costs fail admission for recommended complete-model execution. They count
our Boolean layout, including the expensive private fixed-point multiplications,
and are **not Compact paper-system costs**. Serial lifetime bounds limit live
allocations but do not reduce cumulative one-use material. Zero evaluator-peer
frames omit material/boundary delivery. No full wire, peak RSS or complete Qwen
protected response was measured.

## Fresh pinned-checkpoint quality

The canonical clear numeric cohort is
`docs/evidence/research-compact-product-quality-qwen25.json`. All profiles were
frozen before the twelve new public prompts; none rejected an input range.

| Profile | Prefill selections | Same-token decode selections | Exact eight-token trajectories | Worst absolute logit error |
| --- | ---: | ---: | ---: | ---: |
| 4-piece quadratic | 9/12 | 31/36 | 3/12 | 7.2688 |
| 8-piece cubic | 9/12 | 35/36 | 9/12 | 3.0785 |
| 16-piece cubic | 9/12 | 35/36 | 8/12 | 3.0831 |

No candidate has an exact logit vector or KV state at any of the 48 checked
teacher-forced positions. The fresh-runtime W8A8 control preserves all 48 logit
vectors and KV states and all twelve generation trajectories. Each candidate
checks 131,444,736 encoded outputs against the independent integer oracle across
teacher-forced and generation execution. Observed swap growth is zero.

These profiles fail the fidelity screen. Increasing the number of polynomial
pieces does not remove input/output quantization, tail and intermediate rounding
changes. This cohort does not isolate their individual effects, establish
representative quality or compare against upstream floating point. Historical
Q7 activation-only evidence retains its separate scope. An earlier shell attempt
hit a two-minute harness timeout before writing a report; its record is
`docs/evidence/research-compact-product-quality-timeout.json`, not a quality result.

## Reproduction

```sh
uv run python -m benchmarks.research.compact_product --mode quality \
  --output docs/evidence/research-compact-product-quality-reproduction.json
uv run python -m benchmarks.research.compact_product --mode cost \
  --output docs/evidence/research-compact-product-cost-reproduction.json
uv run python examples/benchmarks/compact_product.py --lanes 16 --samples 3 \
  --output docs/evidence/research-compact-product-sdk-reproduction.json
cargo test --locked --release -p pllm-core -p pllm-garble compact_product
uv run pytest tests/test_compact_product.py
```

The quality driver freezes uniform 4-piece quadratic, 8-piece cubic and
16-piece cubic profiles before twelve new public prompts. It checks an exact
fresh-runtime control, prefill and same-token decode logits/KV, and eight-token
greedy trajectories under a memory/swap guard. The locked W8A8 checkpoint is
the comparator, not upstream floating point. Complete private float conversion,
compiler tensor scheduling, role transport and independent cryptographic review
remain separate requirements for decoder execution.
