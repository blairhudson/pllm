# NEXUS GELU numerical decision experiment

Status: **APPROVED for the bounded numerical experiment only** (D24 in
`paper/methods/DECISIONS.md`). The user explicitly approved this scope. No
executable numerical profile, encrypted backend, decoder, or acceptance tolerance
is adopted by this document.

The user separately approved the four-schedule follow-up below with "Approve
bounded study". That approval covers clear numerical research only, not a change
to the frozen D24 candidate or any runtime numerical contract.

## Purpose and source boundary

Test whether the fixed source-derived comparison composition can support the
paper's GELU selector construction before implementing CKKS arithmetic around it.
Source pins, printed coefficients, and paper/artifact conflicts remain in
`docs/research/nexus-numeric-contract-audit.md`. Do not substitute artifact
constants, refit coefficients, add repetitions, clip inputs, or change thresholds
within the frozen D24 experiment. The separately approved follow-up compares only
the four schedules explicitly listed below.

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

## Recorded D24 outcome

The fixed experiment ran at 80 decimal digits with **34 tests** at that stage
(the current extended suite has 48 tests). Reproduction commands and the separation of sampled maxima,
rigorous point enclosures and unresolved whole-domain cells are recorded in
`docs/research/nexus-numeric-contract-audit.md`, under "Approved bounded GELU
numerical experiment". Current ordinary output is
`target/nexus-gelu-numerics.json`; the earlier checked-in evidence JSON is unchanged.

The fixed selector remains **unqualified**, not an adopted numeric contract.
Strictly outside the nominal band, at normalized `+/-2^-19`, sign error is about
`0.999623`, established by outward point enclosures. Moreover, the exact oracle's
jump at 3 imposes the exact uniform-error lower bound
`0.00793176970057680015` on any continuous approximation to that oracle. Sharper
comparisons cannot remove this jump floor. Neither fact supplies a uniform
upper-error certificate against any target.

## Approved follow-up study — executed, no adoption

The user explicitly approved this separate bounded clear-arithmetic experiment.
It does not revise D24's fixed candidate or permit an executable numerical profile.
The original composition and its result remain the baseline. All four schedules
below have now been evaluated in the existing reusable Python experiment.

### Sources and four frozen candidates

Use only the printed exact dyadic `f4` and `g_(4,1/4)` pinned in the audit to
Cheon–Kim–Kim, ASIACRYPT 2020, DOI `10.1007/978-3-030-64834-3_8`, pp.232,241–242,
§§3.1,3.5, Algorithm 3 and Corollary 3. Algorithm 3 supplies the repeated-g then
repeated-f construction and `(S+1)/2` comparator conversion. Corollary 3 relates
repetition counts to input separation and output accuracy; it is not an error
certificate for any schedule chosen here.

Compare exactly four compositions `S_(r,s)(z)=f4^s(g4^r(z))`, where superscripts
mean composition, not scalar powers:

| Candidate | r | s | Formal degree | Status |
| --- | ---: | ---: | --- | --- |
| Existing baseline | 2 | 2 | `9^4` | D24 candidate, unqualified |
| Research candidate A | 6 | 2 | `9^8` | Studied, not adopted |
| Research candidate B | 10 | 2 | `9^12` | Studied, not adopted |
| Research candidate C | 10 | 3 | `9^13` | Studied, not adopted |

The three new repetition counts are explicit research adaptations within the
source-cited construction, **not schedules claimed by NEXUS or guaranteed by the
comparison source**. Do not fit or import coefficients, change tau, insert the
artifact's intermediate factors, clip stages, change comparison normalization or
expand these high-degree compositions into coefficient arrays. If a candidate
cannot be enclosed or is numerically unstable within the budget, report that
result; do not repair it or add candidates automatically.

### Explicit domains, targets and diagnostics

- Standalone sign/comparator domains are the three closed separated sets
  `D_epsilon=[-1,-epsilon] union [epsilon,1]`, for exactly
  `epsilon in {2^-4,2^-10,2^-20}`. Report each separately. Also retain strict
  outside-band witnesses at `+/-2^-19`; do not infer accuracy arbitrarily close
  to zero from a separated-domain result.
- Original-unit GELU research domain remains `[-8,8]`. Keep the same normalizers
  `12,199/20,11`, adjacent-difference selectors, exact PUMA P/Q coefficients,
  shorter-print sensitivity control, and authoritative endpoint assignments:
  zero at -4, P at -1.95, Q at 3. No model/layer range is admitted.
- For each epsilon, label the original-unit separated subset explicitly as
  `E_epsilon={x in [-8,8]: abs((x-t)/Delta_t)>=epsilon for every threshold t}`.
  Report it separately from the full domain and from within-band neighborhoods.
  Do not replace full-domain results by results on E_epsilon.
- Measure absolute sign and comparator error; every intermediate stage range;
  selector leakage, negative weights and overshoot; inactive-branch amplification;
  exact-threshold and one-sided blending error; and end-to-end error separately
  against the unchanged piecewise oracle, tanh GELU and `x*Phi(x)`. Retain the
  oracle-jump lower bound and coefficient-precision sensitivity.
- Report formal degree, number of composed polynomial stages, and arithmetic
  multiply/add counts and dependency depth for an explicitly stated ordinary
  polynomial-evaluation DAG. Distinguish constant products from variable products.
  These are clear-arithmetic cost diagnostics, **not CKKS level, latency, memory,
  rescale, security or executable-profile claims**.
- Keep sampled estimates, exact-rational statements, outward point enclosures and
  interval domain bounds in separate fields. A cap hit stays unresolved; samples
  cannot promote an unresolved interval to a certificate. No error tolerance or
  candidate is automatically accepted.

### Exact study budget and ordinary output

- Four candidates only, 80 decimal digits for MP/IV evaluation, existing optional
  `mpmath==1.3.0`; no installs, native workers, checkpoints or heavy builds.
- Per candidate, at most 307 distinct normalized sign samples: the 257-point
  uniform grid on `[-1,1]` plus `+/-2^-k` for `k=0..24`, deduplicated. At most
  381 original-unit samples: the existing `sample_points(257)`. Classify these
  same samples into the separated domains; no extra grid sweeps or search.
- Four interval passes per candidate: one for each D_epsilon and one for the
  full original-unit GELU domain, with explicit cuts at thresholds and band
  boundaries. Each pass is capped at 64 leaf cells, 128 cell evaluations and
  refinement depth 12. Across the entire study: at most 16 passes and **2,048
  cell evaluations**. Restrict full-domain cells to label E_epsilon results where
  possible; a cell crossing its boundary is unresolved for that restricted
  result, not silently treated as wholly separated. Point diagnostics use only
  the listed samples, thresholds and strict-outside witnesses, not a second sweep.
- Wall-time cap: 120 seconds per candidate, 480 seconds total experiment time;
  a timed-out candidate remains incomplete/unresolved, with no automatic retry
  or budget increase. Ordinary small regression tests are separate from that
  experiment budget.
- Extend only the existing reusable `.py` experiment and dedicated
  tests. Produce one ordinary JSON comparison at
  `target/nexus-gelu-follow-up.json`, plus a concise result update in the existing
  audit. No receipt collections, evidence archives, replay verifiers or additional
  experiment framework. The actual invocation and results are recorded below.

### Executed result and reproduction

```sh
uv run --no-sync python examples/benchmarks/nexus_gelu_numerics.py --follow-up --output target/nexus-gelu-follow-up.json
uv run --no-sync python -m pytest tests/test_nexus_gelu_numerics.py -q
uv run --no-sync ruff check examples/benchmarks/nexus_gelu_numerics.py tests/test_nexus_gelu_numerics.py
```

The study completed in **3.374 seconds**, using **1,936 cell evaluations** total
(484 per candidate). Each candidate used **291 normalized and 381 original-unit
sample locations**, within both caps. Each of the 16 interval passes retained
64 leaves; all **1,024 leaf stage-domain checks remained unresolved**, maximum
depth 7. Completion of bounded work is not numerical qualification. The current
correctness suite passes **48 tests**; Ruff passes.

The audit records the numeric comparison tables. In brief: `(10,2)` and `(10,3)`
sharpen the sampled sign on the smallest separated domain, but their sampled
tanh GELU error grows to about **0.01950**, versus the baseline's **0.01180**.
The exact piecewise-oracle jump floor persists, and selector negativity was not
assumed away. No candidate has a useful certified uniform upper-error bound.

Hard limits are enforced with nested POSIX wall-time alarms and a shared attempted-
cell counter. The CLI's outer alarm includes setup and JSON output. Candidate
timeouts/failures are marked incomplete; total timeout stops study work and saves
only a compact incomplete-status JSON, with nonzero exit status. No extra grid
sweep, candidate search or interval-budget increase was run.

### Remaining adoption decision

These results still require a separate user decision on the
accepted target (piecewise oracle versus tanh GELU versus `x*Phi(x)`), numerical
error budget and statistic (uniform versus distributional), boundary/blending and
negative-weight acceptance, and public-range/failure policy. Any uniform
piecewise-oracle budget below the exact jump floor is impossible for a continuous
candidate. Decimal-to-CKKS conversion, evaluation/rounding, Q/P/rescale/refresh,
resource admission and a full executable profile remain outside this study
and require independent approval and qualification.
