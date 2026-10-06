# Compact DCF backend for shared rescaling

PLLM independently implements the linear-size distributed comparison function
in **FSS for Mixed-Mode Secure Computation**, Figure 1. The existing SIGMA/FuseFSS
signed-rescale helpers can select it under the same numeric, one-use and
allocation contracts as the quadratic prefix-DPF control. This is a native
component reference with measured helper costs, not an executable decoder.

## Source and construction

Specification: [Boyle et al., ePrint 2020/1392](https://eprint.iacr.org/2020/1392),
Section 3, Figure 1 and Theorem 2. Pinned PDF SHA-256:
`dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4`.
No upstream code is built, linked or vendored. The
[existing helper note](shared-rescale-reference.md) records the SIGMA/FuseFSS
source contracts, masked equations and the separate PLLM ties-to-even extension.

The private comparison returns additive shares of `[x < threshold]` in
`Z/(2^32)`. It admits 1–32 input bits, samples two fresh 128-bit roots, and
issues one correction per input bit. Figure 1's running group correction makes
the shares sum to one below the threshold and zero at/above it. Evaluation
traverses exactly one path; no threshold bit changes the public key shape.

The PRG uses AES-128 under the current seed with five domain-separated public
counter blocks: two child seeds, two value seeds and separate control bits.
The group conversion takes 32 bits, as permitted for this power-of-two group.
AES uses the workspace's zeroizing implementation. Root, correction and working
seed owners erase their retained storage; evaluator keys have no public clone,
export or standalone evaluation interface. This is not an independent
cryptographic or side-channel review.

Controls occupy separate bytes. A `k`-bit comparison's per-party payload is:

```text
16-byte root + 4-byte final correction + k * (16-byte seed + 4-byte value + 2 controls)
  = 20 + 22k bytes
```

The quadratic control uses `20k + 9k(k+1)` bytes. At 32 bits the comparator
therefore drops from **10,144 to 724 bytes**. The published bit-packed format is
smaller still; these measurements use PLLM's explicit byte controls. There is
no offline transport codec: payload counts exclude its future framing and
allocator overhead. Each helper also counts its mask and constant shares.

Backend identity joins the numeric/layout context commitment. A mismatched
backend is rejected even if a test forces the issuance nonces to match. Both
backends pass exhaustive 2–6-bit signed inputs at every shift and rounding mode,
full-width boundaries and negative ties, and malformed/replayed/cancelled
one-use lifecycle checks. Standalone compact-DCF tests additionally cover every
1–6-bit threshold with zero, one and wraparound group payloads. The same 64 MiB
combined allocation estimate and 1,024-lane bound precede issuance.

## Fresh-process measurements

Reproduce with a new output path:

```sh
uv run python -m benchmarks.research.compact_rescale_reference \
  --output docs/evidence/research-compact-rescale-reference-reproduction.json
cargo test -p pllm-garble
```

One pinned configuration launches **24 fresh native processes**: three
width/shift pairs, floor/ties-to-even, fused/unfused and both backends. Adjacent
backend pairs alternate execution order. Five fresh issuances of 64 public
synthetic inputs per case check **7,680 helper output pairs** against independent
native and Python integer oracles. Observed new swap is zero.

These are matched controls from the new cohort; the previous 12-case timing
cohort stays retained separately. Source resolution and native compilation are
outside the timings. The table shows ties-to-even, per-party/lane key payload
and median elapsed time for a complete 64-lane batch. Online time includes both
local evaluators, frames, reconstruction and retirement, excluding issuance
and input sharing.

| Width / shift | Layout | Prefix → compact key bytes | Prefix → compact issuance ms | Prefix → compact online ms | Peer bodies, either backend |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 / 7 | Unfused | 5,888 → 1,426 | 69.17 → 10.88 | 58.54 → 8.94 | 1,336 B |
| 16 / 7 | Fused | 9,428 → 1,718 | 109.13 → 12.70 | 96.32 → 11.13 | 668 B |
| 24 / 12 | Unfused | 12,556 → 2,108 | 144.90 → 15.76 | 128.50 → 13.78 | 1,336 B |
| 24 / 12 | Fused | 21,012 → 2,620 | 240.02 → 19.15 | 219.76 → 17.56 | 668 B |
| 32 / 12 | Unfused | 17,628 → 2,460 | 202.01 → 18.25 | 183.02 → 16.25 | 1,336 B |
| 32 / 12 | Fused | 29,540 → 2,972 | 335.38 → 21.69 | 312.59 → 20.06 | 668 B |

For 16/7 fused ties-even, material falls **5.49×** and local online time
**8.65×** in this cohort. Full process CPU for its five samples falls
**1.032 to 0.123 s**; fresh process peak RSS falls **3.72 to 2.05 MB**. Those
process measurements include the local dealer and both evaluators, not a client
or independent provider deployment. Every raw sample, floor case, phase timer,
CPU/RSS measurement and implementation digest is retained in the JSON.

Fusion still exchanges one peer opening instead of two. With compact DCF,
16/7 ties-even fusion changes **1,336 to 668 bytes**, but increases keys
**1,426 to 1,718 bytes per party/lane** and local online time
**8.94 to 11.13 ms**. Lower key growth improves that tradeoff; it does not make
fusion free or establish either paper's system-level speedup.

## Whole-width gate

The same compiler-derived hypothetical placement uses one helper per semantic
SiLU element. It has no mapping to Qwen's current W8A8 floating-point nonlinear
execution, checkpoint values, logits, KV or generation quality.

| Workload | Compact layout | Both parties' one-use key payload | Peer bodies at 64 lanes |
| --- | --- | ---: | ---: |
| 39+8 | Unfused | 15.31 GB | 112.10 MB |
| 39+8 | Fused | 18.45 GB | 56.05 MB |
| 39+32 | Unfused | 23.31 GB | 170.58 MB |
| 39+32 | Fused | 28.08 GB | 85.29 MB |

The 39+8 fused rescale/predicate helper alone projects about **1,066 s issuance**
and **934 s local online work** by linear extrapolation. These are projections,
not measured decoder CPU. They omit polynomial evaluation, other rescaling,
attention, token feedback, distribution and complete response work. A compact
backend alone does not pass the tensor/whole-response cost gate.

Next work needs independent cryptographic review and substantially cheaper
complete operators. Reusing one comparison key across a published
multiple-interval construction is a concrete next source-review candidate;
the current helpers still issue independent comparison keys. Tensor scheduling,
distributed dealer, authenticated independent roles and Qwen numeric admission
remain unimplemented. Context framing does not verify honest peer values.

Code: `crates/pllm-garble/src/compact_dcf.rs` and `shared_rescale_reference.rs`.
Replay: `benchmarks/research/compact_rescale_reference.py`.
Evidence: `docs/evidence/research-compact-rescale-reference-qwen25.json`.
