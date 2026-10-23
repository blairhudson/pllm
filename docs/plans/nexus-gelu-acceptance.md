# N-G: separately selectable GELU acceptance contract

**D37 design only; NOT ADOPTABLE yet.** This packet selects nothing. The user
commissioned an acceptance specification, not another numerical study, a runtime,
or an executable NEXUS profile. No candidate has the useful certified uniform
upper bound required below. Proposed budgets and policies are choices for a
subsequent explicit decision, not source facts or approved tolerances.

Prepared against `7fa07d74`; authority is D37 in the
[decision register](../../paper/methods/DECISIONS.md). Preserve the completed
[D24/four-schedule study](nexus-gelu-numeric-experiment.md#approved-follow-up-study--executed-no-adoption)
and [source audit](../research/nexus-numeric-contract-audit.md). Neither needs
renewed approval merely to recognize its existing results.

## 1. Immutable interpretation and source limits

| Input | Pinned identity / role |
| --- | --- |
| NEXUS final, NDSS 2025, DOI `10.14722/ndss.2025.230868` | Paper SHA-256 `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`; Eq.(3), p.8 governs segment endpoints. |
| NEXUS artifact | Commit `e15058c7f8f58fcf037c4709211109cb1be7d1e0`; inspected `src/gelu.cpp` SHA-256 `37d6afb3db39db65c6cd255cb809b865f723ae1deca8d606fcf148477a5aef20`. Its ±3.5/degree-12 construction is **not** the candidate below. |
| PUMA v3, §4.2 Eq.(3) | PDF SHA-256 `84c82810a16be95b41200ba8e0071c39a8cadb15a45008ff8e75e295e0960973`; full printed P/Q decimals, interpreted as exact rationals in ordinary powers of unshifted x. |
| Cheon–Kim–Kim, ASIACRYPT 2020 | DOI `10.1007/978-3-030-64834-3_8`; publisher PDF SHA-256 `3cebc1b089066707780fd54a4b16c71fec56ec56e6b7be72824a0a4570519146`; exact printed `f4`, `g_(4,1/4)`, repeated-g then repeated-f. |
| BumbleBee final | PDF SHA-256 `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`; shorter P/Q literals remain a sensitivity control only. |

These pins identify inspected inputs, not NEXUS's evaluated coefficient precision,
encoded CKKS constants, checkpoint, secret keys, or executed author build. PUMA's
and BumbleBee's endpoint, fixed-point and backend rules do not transfer to NEXUS.
The [audit's recovered literals](../research/nexus-numeric-contract-audit.md#recovered-primary-source-polynomials-not-an-adopted-profile)
and existing `P`, `Q`, `F4`, `G4` tuples in
[`nexus_gelu_numerics.py`](../../examples/benchmarks/nexus_gelu_numerics.py)
are the exact candidate definitions; do not refit, shorten or re-encode them here.

## 2. Candidate and domain choices

All offered candidates are the **already studied** scalar mathematical maps

```text
S_(r,s)(z) = f4 composed s times after g_(4,1/4) composed r times
t = (-4, -39/20, 3); Delta = (12, 199/20, 11)
H_t(x) = (1 + S_(r,s)((x-t)/Delta_t))/2
(b0,b1,b2,b3) = (1-H_-4, H_-4-H_-39/20, H_-39/20-H_3, H_3)
Y_(r,s)(x) = b1*P(x) + b2*Q(x) + b3*x
```

Selectable schedule labels: `C22=(2,2)`, `C62=(6,2)`, `C102=(10,2)` or
`C103=(10,3)`. The last three are research adaptations of the cited comparison
construction, not schedules attributed to NEXUS. **Recommendation: retain C22 as
historical control only; adopt none.** No schedule is a fallback for another.
The artifact's intermediate factors of 0.5, stage clipping, expanded composition
coefficients and coefficient refitting are excluded.

The offered public research domain is exactly **D=[-8,8]**, closed, including
all thresholds and both endpoints. Each normalizer is the difference bound
`Delta_t=max(|-8-t|,|8-t|)`, not an input-magnitude bound. This does not admit any
actual layer/checkpoint input. A different domain requires a separately frozen
normalization, candidate identity and certificate; it is not an automatic retry.
No separated subset or empirical input distribution replaces D.

## 3. Select exactly one target and mathematical budget

Each target has its **own** statistic

`E_target(C)=sup_{x in D} |Y_C(x)-F_target(x)|`.

This is full-domain absolute error in original output units, not relative,
average, median, sign accuracy or distributional error. Selecting a target does
not select either of the others. No quality target follows from these options.

| Target choice | Exact reference | Proposed mathematical budget options, NOT approved |
| --- | --- | --- |
| `N-G-P` | Exact source-derived piecewise oracle: `F=0` for `x<=-4`; `P(x)` for `-4<x<=-39/20`; `Q(x)` for `-39/20<x<=3`; `x` for `x>3`. P/Q interpretation is §1, not recovered author encodings. | `P010: E_P<=1/100`; or `P020: E_P<=1/50`. Necessary jump condition holds, but sufficiency is **unknown** for every candidate. |
| `N-G-T` | `F=x*(1+tanh(sqrt(2/pi)*(x+(8943/200000)*x^3)))/2`, with exact mathematical constants; `8943/200000=0.044715`. | `T015: E_T<=3/200`; or `T025: E_T<=1/40`. All candidate upper certificates **unknown**. |
| `N-G-X` | `F=x*Phi(x)=x*(1+erf(x/sqrt(2)))/2`, exact normal CDF, not the tanh approximation. | `X015: E_X<=3/200`; or `X025: E_X<=1/40`. All candidate upper certificates **unknown**. |

The displayed budget values are deliberate **proposals**, not values supplied by
the paper, achieved accuracy, source-model quality criteria or HE error allowances.
A user may defer all choices or request a different explicit packet; omitted
budgets never receive defaults.

For `N-G-P`, the exact jump at 3 is
`Q(3)-3=0.0158635394011536003`. For every continuous Y on D, the two one-sided
limits and triangle inequality give

`sup_D |Y-F_P| >= |Q(3)-3|/2 = 0.00793176970057680015`.

All four blends are continuous polynomials. Therefore zero error, exact hard-
branch equality and any budget below this floor (for example `1/128`) are
impossible for them. A budget above the floor is **not** a certificate. This
lower bound neither proves an achieved maximum nor transfers to `N-G-T` or
`N-G-X`. Removing a finite threshold set does not remove the one-sided uniform
lower bound. A different hard-branch implementation needs a separate proposal;
it cannot be silently substituted for Y.

## 4. Boundary, blending and negative-weight choices

The target's endpoint ownership stays **zero at -4, P at -39/20, Q at 3**.
For every candidate, `H_t(t)=1/2`; other comparators can still leak, so the
complete output is not automatically an exact average of just two branches.
`N-G-T` and `N-G-X` are smooth targets; the candidate's same blend remains in
place. No exact-tie bypass, epsilon shift, boundary exemption, zeroing or clamp.

Choose one explicit negative-weight policy:

- **`RAW-SIGNED` (compatible mathematical proposal):** retain every signed
  adjacent-difference weight and its overshoot in Y. There is no convex-mixture
  promise. Certify intermediate/weight ranges and the complete target error,
  including inactive-branch amplification; do not normalize weights.
- **`CONVEX-REQUIRED` (additional proof obligation):** require `0<=b_i<=1` over
  all D as well as the chosen output-error bound. C22 already fails this policy:
  retained outward point certificates prove `b1(-85/64)<0` and `b2(83/16)<0`.
  Other candidates have no whole-domain convexity certificate. Failure rejects
  the candidate, not permission to clip, renormalize or change coefficients.

The exact telescoping identity `sum b_i=1` alone proves neither nonnegativity nor
accuracy. Tiny sampled negatives, or values near the 80-digit noise floor, must
not be converted into global sign claims. Comparator/sign error alone also
cannot discharge end-to-end error: the existing branch-amplification expression
`|e_-4*P| + |e_-39/20*(Q-P)| + |e_3*(x-Q)|` must include all relevant branches.

## 5. Retained observations versus the adoption gate

The completed follow-up's ordinary output is
`target/nexus-gelu-follow-up.json`; no copy/archive is added. It reports
`completed_bounded_study_not_certified`, 1,936 attempted interval-cell
evaluations and 3.374153833836317 seconds. Each candidate used 291 normalized
and 381 original-unit sample locations. Existing evidence, not a new run:

| Candidate | Sampled max vs piecewise | vs tanh GELU | vs x*Phi(x) |
| --- | ---: | ---: | ---: |
| C22 | 0.00793176970058 | 0.01180187834 | 0.01221801982 |
| C62 | 0.00793176970058 | 0.01946455556 | 0.01987691824 |
| C102 | 0.00793176970158 | 0.01950079097 | 0.01991309323 |
| C103 | 0.00793176970058 | 0.01950079098 | 0.01991309323 |

Displays are rounded; retained JSON values remain authoritative observations.
These sampled maxima are **not certified upper bounds** and are not by
themselves rigorous point refutations of a proposed budget. Sharper sampled sign
does not imply better GELU. All 16 passes reached their 64-leaf caps and all
1,024 leaf stage-domain checks remained unresolved. This proves neither a useful
upper bound nor actual escape from the stage domain. No tolerance is approved
by completing bounded work. Earlier
[`docs/evidence/nexus-gelu-numerics.json`](../evidence/nexus-gelu-numerics.json)
is the distinct initial study, not the four-schedule result.

Before **mathematical candidate adoption**, require a useful rigorous certificate
`U_target(C,D)<=chosen epsilon` for the exact chosen map and target. It must:

1. Cover all D, with exact rational endpoints and thresholds, both one-sided
   limits and the assigned values; no gaps or substitution of separated sets.
2. Use outward bounds including rational coefficient conversion and, for smooth
   targets, certified transcendental enclosures. Track intermediate compositions,
   signed selectors and inactive-branch amplification without assuming convexity.
3. Discharge every proof premise. A remaining unresolved cell/premise, cap hit,
   unbounded intermediate or nonfinite enclosure means **NOT ADOPTABLE**, not
   evidence of failure or permission to substitute samples.
4. State the finite work/workspace/refinement ceiling and precision for any
   proposed proof execution before requesting its separate grant. Existing
   study caps are unchanged; this packet authorizes no retry, larger cap, new
   sweep, study or proof execution, and does not promise those caps suffice.

A direct certificate may bound actual intermediate ranges without proving they
all lie in `[-1,1]`: the exact polynomials are defined outside that interval too.
It must not assume the comparison source's domain lemma when its premises are
unproved. The earlier unresolved stage checks do not prove adoption impossible.

A vacuous upper enclosure larger than epsilon is not useful acceptance evidence.
After mathematical certification, encrypted error remains separate: a future
executable contract needs outward encoding/evaluation/decryption-error bounds
and an explicit total error budget (triangle inequality), not zero residual
allowances inferred from the mathematical one. Candidate/tolerance adoption still
requires a separate explicit decision; a proof result cannot adopt itself.

## 6. Public-range failure and reusable implementation boundary

Proposed **`PUBLIC-RANGE-FAIL-CLOSED`** policy: require source/invocation-bound
public evidence that the complete admitted input lies in D before issuing
numerically dependent material. Missing, mismatched or insufficient range
evidence rejects admission with no GELU result. A legitimately public clear
input outside D also rejects; do not clip it, widen D, change normalizers or
switch schedules/targets. Encrypted inputs do not become public to run this
test. No client decrypt/reveal fallback or per-element range disclosure is
authorized. A private checked-range mechanism, fallback and its leakage/error
semantics would require their own approved packet. CKKS input uncertainty must
fit inside the certified domain; a nominal endpoint with unaccounted error is
not range evidence.

Reuse, rather than clone, the current pieces:

- [Existing experiment/oracles](../../examples/benchmarks/nexus_gelu_numerics.py):
  `polynomial`, `sign_stages`, `piecewise`, `blend`, `Arithmetic.targets`,
  `cell_metrics`, `arithmetic_cost`, `oracle_jumps`; keep existing exact literals,
  point enclosures, DAG counts and incomplete-status behavior.
- [Normal correctness tests](../../tests/test_nexus_gelu_numerics.py): independent
  rational powers, endpoints, interval/cap behavior and actual arithmetic counts.
  The retained study records 48 passing tests; none were rerun for this packet.
- [Public ciphertext trace machinery](../../crates/pllm-compiler/src/encrypted_level_schedule.rs):
  `validate_public_ciphertext_trace` can check an explicitly specified later
  public schedule. Trace consistency grants no numerical error, native capacity,
  source origin, key-security or release proof. No HE DAG is selected here.

CKKS coefficient encoding/rounding, DAG, ordered Q/P primes, rescale/alignment,
refresh error, source-bound ranges, original allocation/owner quotes and actual
key domains remain independent. The existing `ckks_reference.py` scale/drift
defaults and other methods' profiles cannot supply them. Uniform-ring masking,
dual-key composition, Argmax/ties, whole-decoder quality, lifecycle and SDK
qualification also remain open. No public gamma, fixed inverse, new runtime or
source-authentication/cryptographic claim is introduced.

## 7. Current review result and optional objective wording

**Current review result — no renewed approval required:**

> N-G adoption is deferred. Retain C22=(2,2) only as historical control. None of C22,
> C62, C102 or C103 is currently adoptable: useful certified full-domain upper
> errors are unknown. Do not run another study or change numerical/runtime/HE
> policy under D37.

There is no execution or adoption question until a concrete bounded certification
plan exists. Preferences are optional; none is inferred selected.

**Optional, separately selectable acceptance objective, not adoption or execution:**

> N-G specify objective [N-G-P / N-G-T / N-G-X], candidate [C22 / C62 / C102 /
> C103], D=[-8,8], full-domain supremum absolute error, proposed budget
> [P010 / P020 / T015 / T025 / X015 / X025, matching the target], original
> endpoint/blending rules, [RAW-SIGNED / CONVEX-REQUIRED], and
> PUBLIC-RANGE-FAIL-CLOSED exactly as this packet defines. Commission only a
> bounded certification proposal using existing primitives. Return a plan stating
> the exact mathematical quantity, proof method, precision, work, workspace and
> refinement caps, feasibility, and unresolved premises. Do not
> adopt the candidate/tolerance as an executable numeric profile, execute a
> study/proof/backend/model, raise caps, or infer evaluated-author identity,
> privacy, quality or decoder completion. A useful upper certificate would require
> later separately authorized proof execution; it is not a D37 deliverable.

The bracketed fields must each be resolved explicitly; an incomplete reply is
not approval. A request for exact hard-branch equality or a sub-jump piecewise
budget is rejected as incompatible with these continuous candidates, not repaired.
