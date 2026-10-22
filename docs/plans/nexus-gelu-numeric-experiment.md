# NEXUS GELU numerical decision experiment

Status: **APPROVED for the bounded numerical experiment only** (D24 in
`paper/methods/DECISIONS.md`). The user explicitly approved this scope. No
executable numerical profile, encrypted backend, decoder, or acceptance tolerance
is adopted by this document.

## Purpose and source boundary

Test whether the fixed source-derived comparison composition can support the
paper's GELU selector construction before implementing CKKS arithmetic around it.
Source pins, printed coefficients, and paper/artifact conflicts remain in
`docs/research/nexus-numeric-contract-audit.md`. Do not substitute artifact
constants, refit coefficients, add repetitions, clip inputs, or change thresholds.

The final NEXUS piecewise oracle is:

- zero for `x <= -4`;
- `P(x)` for `-4 < x <= -1.95`;
- `Q(x)` for `-1.95 < x <= 3`;
- identity for `x > 3`.

Its endpoint assignments differ from the cited PUMA construction. Preserve them.

## Approved choices for this experiment only

1. Interpret the full PUMA printed P/Q literals recorded in the audit as exact
   decimal rationals in ordinary powers of unshifted x. Retain the shorter
   BumbleBee literals as a separate sensitivity control. Neither choice is
   asserted to reproduce NEXUS's actual encoded coefficients.
2. Select the published exact dyadic `f4` and `g_(4,1/4)` from the audit, with
   `S = f4(f4(g4(g4(z))))`. Do not include the artifact's intermediate factors.
3. For a public research domain `[L,U]`, define, for each threshold t,
   `Delta_t = max(abs(L-t), abs(U-t))` and
   `H_t(x) = (1 + S((x-t)/Delta_t))/2`. Reject nonpositive normalizers.
4. Use adjacent-difference selectors:
   `(b0,b1,b2,b3) = (1-H_-4, H_-4-H_-1.95, H_-1.95-H_3, H_3)`.
   Evaluate `Y = b1*P(x) + b2*Q(x) + b3*x`.
5. Treat Y as a blending candidate, not an exact implementation of the piecewise
   endpoint oracle. At its own threshold an exact comparator is 1/2; selectors
   need not be Boolean, nonnegative, or free of approximation leakage.

For the fixed composition, the exact derivative at zero is
`S'(0) = (315/128)^2 * (5850/1024)^2`, approximately 198. This motivates
checking claimed near-threshold behavior; it does not itself certify an error
bound or prove a complete paper-system defect.

## Bounded reusable experiment

Use one ordinary `.py` experiment and its ordinary numerical output. Do not add
receipt archives or custom replay/completion machinery. No checkpoint, source
weights, private prompts, encrypted material, or provider roles are needed.

- Study `[-8,8]` as the paper-observed research domain only, not as an admitted
  checkpoint/layer range. Its three difference bounds are `12`, `9.95`, `11`.
- Compare Y separately against the exact piecewise oracle, tanh GELU, and
  `x*Phi(x)`. Do not silently select one target definition.
- Include domain endpoints, exact thresholds and one-sided neighborhoods, plus
  normalized offsets `+/-2^-k` for `k=0..24` wherever inside the domain.
  Explicitly include `+/-2^-20`. Keep normalized and original-unit bands distinct.
- Use high-precision evaluation and bounded adaptive interval enclosure for
  whole-domain checks. Bound work and refinement depth; unresolved enclosures
  remain unresolved, never replaced by sampled maxima presented as certificates.
- Report sign error outside the nominal uncertainty band, selector leakage and
  overshoot, boundary blending error, complete-domain GELU error, and coefficient
  precision sensitivity. Separate sampled observations from proven enclosures.
- Account for inactive-branch amplification. With comparator errors e_t,
  `abs(Y-F) <= abs(e_-4*P) + abs(e_-1.95*(Q-P)) + abs(e_3*(x-Q))`.
  Small comparator error alone does not establish small final GELU error.

## What this approval does not settle

This experiment cannot recover hidden author implementation choices or choose
user tolerances. It does not approve decimal-to-CKKS conversion, a BSGS evaluation
DAG, rounding, ordered Q/P primes, rescale/alignment, refresh, public model-range
certificates, domain-failure handling, or exact-versus-blended endpoint acceptance.
The existing PLLM CKKS reference's scale and drift policy cannot be inherited.

Failure returns a concrete numerical gate for review. It does not authorize
automatic polynomial changes, another backend, or a complete NEXUS SDK profile.
