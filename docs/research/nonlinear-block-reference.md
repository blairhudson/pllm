# Complete shared Q7 gated block

The signed-rescale references now compose with two fresh Beaver multiplications
into the complete `pllm.numeric.gated_multiply.q7.v1` block. This is a bounded
in-process component reference inspired by SIGMA/FuseFSS, not either paper's
complete system or spline/coefficient-lookup implementation.

## Execution and ownership

Inputs are signed Q14 gate/up values with a caller-enforced public `[-1,1]`
bound. The 24-bit ring holds all intermediate integer products without signed
overflow on that domain. The block performs:

1. Two Q14-to-Q7 ties-to-even rescalings, in parallel.
2. A shared square using a fresh one-use Beaver triple.
3. The quadratic SiLU numerator `g² + 256g`, rounded by 512.
4. A second fresh shared multiplication by the Q7 up input.
5. Final ties-to-even division by 128.

Each evaluator owns one share of every triple and FSS key. The test-only dealer
issues both parties' material. No intermediate activation is reconstructed;
the probe reconstructs only the final output for its independent oracle.
Contexts bind the plan, session, operation, lane range and sub-operation.
Malformed frames, out-of-order use, replay and cancellation retire the entire
remaining block. Secret owners zeroize on retirement. Aggregate allocation is
preflighted under 64 MiB and 512 outputs before any issuance.

Compact DCF and universal interval keys run the same integer operation schedule.
The fused schedule needs five sequential peer rounds; the unfused needs eight.
Peer accounting includes all helper and multiplication frames. Key accounting
includes all four rescalings and both triples. Input/output share payloads are
reported separately; offline key transport and authenticated network framing
are not implemented. Input-range proofs, distributed issuance, malicious
verification and independent cryptographic review remain absent.

## Measured complete-block costs

Replay:

```sh
uv run python -m benchmarks.research.nonlinear_block_reference \
  --output docs/evidence/research-nonlinear-block-reference-reproduction.json
cargo test -p pllm-garble shared_gated_block_reference
```

Twelve fresh native processes cover both backends/layouts and 1, 64 and 512
lanes, with five fresh issuances each. **11,540 outputs** match independent
integer oracles. For 64 lanes:

| Backend / layout | Key bytes per party | Peer bytes | Issuance ms | Online ms |
| --- | ---: | ---: | ---: | ---: |
| Compact / unfused | 465,152 | 7,392 | 55.35 | 48.37 |
| Interval / unfused | 208,128 | 7,392 | 26.07 | 46.73 |
| Compact / fused | 545,536 | 4,876 | 63.42 | 58.15 |
| Interval / fused | 246,272 | 4,876 | 29.42 | 55.96 |

Times include the in-process dealer or both evaluators as appropriate; they are
not distributed timings. Zero new swap was observed. Native source/binary
build, checkpoint loading and the separate domain diagnostic are outside the
isolated timings. Full native process CPU and peak RSS remain in the report.

## Checkpoint domain and whole-width rejection

Eight fixed public prompts execute the pinned Qwen2.5-0.5B W8A8 baseline through
prefill and two decode steps. The diagnostic observes semantic SiLU/up tensors,
without saving activations, logits or token IDs. Of **40,974,336** pairs,
**14,778,202 (36.07%)** put at least one input outside `[-1,1]`. Even widening
the gate profile to the separate public maximum `[-16,16]` leaves **418**
out-of-range gate values. No clipping, private-prompt calibration or silent
profile change is performed. In-domain rounded inputs only match **811** of
the baseline float outputs bit-for-bit; encoded correctness is not W8A8 parity.

Using compiler dimensions alone, a hypothetical 39+8-token full-MLP placement
with interval keys and 64-lane batches projects:

| Layout | Both parties' one-use keys | Peer bodies | Input/output share payloads |
| --- | ---: | ---: | ---: |
| Unfused | 34.93 GB | 620.22 MB | 85.92 / 42.96 MB |
| Fused | 41.33 GB | 409.12 MB | 85.92 / 42.96 MB |

The older **8.14 GB** figure covered one helper per SiLU element. It was not
the cost of this complete block. These larger projections exclude attention,
normalization, body linear work, private feedback, model distribution and
complete wire costs. No protected checkpoint or candidate decoder was run:
the numeric domain and resource gate both fail. The method remains ineligible
for SDK search or a whole-response score.

Code: `crates/pllm-garble/src/shared_arithmetic_reference.rs` and
`shared_gated_block_reference.rs`. Replay and raw report:
`benchmarks/research/nonlinear_block_reference.py`,
`docs/evidence/research-nonlinear-block-reference-qwen25.json`.

## Wider-domain follow-up

The separate [public piecewise-gated reference](piecewise-gated-reference.md)
tests Q8/Q9 affine SiLU with explicit tails and complete scalar/vector coefficient
lookup costs. All new held-out inputs fit its range, but its best profile matches
only 10/12 prefill and 35/36 decode selections. Vector lookup reduces matching
complete-block material by 11.51%; hypothetical 39+8 still needs 60.10 GB raw
material and 465.16 MB peer bodies. Its numeric identity differs from this Q7
reference, and neither profile is promoted.
