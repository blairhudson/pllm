# Hundredfold network research: conditions before implementation

## Objective and accounting

Fresh prompts and an unchanged pretrained checkpoint remain the hard target.
Plaintext inputs/state stay within the trusted client or secret-shared execution;
workers must not receive both shares or a recoverable complete mask. Most model
compute remains remote. Full-response aggregate compute must be measured against
the existing two-offset comparator; lower network does not excuse unbounded
dealer or client computation.

Count client links, worker links, fresh correlations, control and amortized public
distribution. Keep warm application-body counts separate from cold distribution
and full wire. Mask/ticket reuse is not a saving. Average bytes per generated
token include prefill; a marginal decode claim needs separately identified evidence.

Pinned Qwen2.5 W8A8 prepared controls give these **covered-body** targets:

| Workload | 100× response online | 100× response all-link | All-link per generated token, including prefill |
| --- | ---: | ---: | ---: |
| 39+8 | 738,317 B | 1,168,439 B | 146,055 B |
| 39+32 | 1,135,450 B | 1,789,705 B | 55,928 B |

The finite difference between those matched source/body cohorts is 1,654,720
online and 2,588,608 all-link bytes per extra output token. Its 100× targets are
**16,547 online / 25,886 all-link bytes**. This is a difference of response
cohorts, not an isolated marginal-decode measurement. The separate 39+32 offset
cohort has 227.08 MB online; its complete cold all-link cost remains unknown.
Rank against both controls only when scopes and workloads match.

## What must become true

For the tested 24-bit resident-share layout at 39+32:

- There are 70 executed rows and 48 declared independent attention/MLP sources
  per row. Their peer-opening arithmetic alone is **18.06 MB**, 15.91× the
  entire 100× online budget.
- At most **three** full-width openings per executed row fit if everything else
  were free. Full-width private ingress/client-local feedback consumes 548,352 B,
  leaving room for only **one** such opening per row, before any other work.
- The entire all-link budget is only **0.219 B per gated element**, before
  attention, normalization, rescaling, feedback or distribution.
- Spending the online budget leaves **654,255 B** for fresh material and other
  all-link-only costs. The new quadratic core alone needs 53.69 MB of keys;
  generalized quartic coefficients project 151.75 MB.

These constrain the sampled layout; they are not impossibility theorems for all
cryptographic protocols. Narrower rings require an exact carry/range contract.
Compressed feedback must actually implement hidden lookup and selection.

## Hypotheses and bounded tests

| Hypothesis | What would need to be true | First falsifiable test | Current outcome |
| --- | --- | --- | --- |
| **Opaque cross-layer state** | Most attention/MLP transitions avoid another full hidden-vector opening; exact rescale and KV contracts survive composition | Two complete consecutive layers under one opaque-state contract; count every reshare/refresh and check exact outputs | Unimplemented; budget requires order-of-magnitude fewer source openings |
| **Programmed correlation expansion** | Party-local short keys generate nonlinear mask correlations without reconstructing the peer mask; fresh material fits the sub-MB remainder | Generate square/product/contracted-constant shares with independent party state; test leakage, replay, expansion CPU and actual key bodies | Open; ordinary PRG seeds do not supply this capability |
| **Polynomial core plus exact sparse correction** | Residuals can be discovered/scattered privately, with certified fixed traffic and exact output codes | Lock public fit; count exact support on independent trajectories, then price padded positions, values, core material and private selection | Current construction vetoed; mean sparsity hides large tails and material |
| **Masked-program handoff** | Public code plus opaque state carries a nonlinear result into the next region without exposing the mask | Expose shifted polynomial coefficients; construct an adversary from the permitted view | Simple public-coefficient construction leaks; coefficients must stay opaque |
| **Compact private token boundary** | Lookup and selection/feedback cost near token-sized messages, with tolerable scan/selection CPU | Compose actual private lookup and selection with one state update, not an ideal token oracle | Prior lookup is partial; whole private feedback remains unimplemented |

The research combination worth pursuing is **opaque state + programmed
correlations + exact correction + compact feedback**. None may be replaced by
a zero-cost box. A narrower experiment can falsify a required property without
claiming it executes that combination.

## New experiments executed

### Sparse integer correction

The independent float32 quartic screen left **2.39%** of quantized MLP codes
different on reference trajectories, but **700/4,864 (14.39%)** in the worst
checked row. True dynamic scales were supplied for free. Polynomial-only rollout
still failed (4/8 prefill and 3/8 decode selections).

Even an ideal hidden-scatter interface with a 13-bit position and 9-bit signed
difference delivered to each party projects **1.082 MB** at rounded-up observed
mean rate, **6.468 MB** at observed worst-row padding rate, and **151.75 MB**
additional fresh quartic correlation material.

Those are conditional projections across different public prompt cohorts, not
measured 39+32 sparse execution. Observed maximum is not a capacity certificate.
Positions, counts, signs and overflow decisions cannot simply become public.
Private discovery, scatter, signed lifting, scaling and framing remain unpriced.
This core fails 100× before those operations. Retain residual structure as a
question for a different core, not a selectable optimization.

### Public shifted-polynomial handoff

For `F(G,U)=((G-a)^2 + L(G-a))(U-b)`, exposed coefficients contain
`c_GU=L-2a` and `c_G²=-b`. Thus `U+c_G²` reveals the exact up activation;
`G-(L-c_GU)/2` reveals the gate modulo `2^(k-1)`.

The new native assurance control executes this witness: exhaustive 8-bit checks
and 320 public fixtures across 8/16/24/32/64-bit rings recover the declared
values. Making these coefficients public to eliminate a share is invalid.
This does **not** attack opaque party-local correlations or prove their privacy.

## Next experimental sequence

1. Specify a **programmed correlation generator** for precisely the required
   mask products. Enumerate each party's complete view. Price seed/key delivery,
   corrections and expansion rather than seed length alone. Compare against the
   Rust full-key ablation before tensor-scale work.
2. Keep exact sparse correction as a separate gate: independent confirmation
   data, secret support, signed lifting, public fixed-capacity handling and exact
   code equality. Do not tune against the inspected eight-prompt cohort.
3. Require a two-layer opaque-state proof before expanding to a decoder. A
   per-layer protocol retaining 24/48 wide openings cannot meet this target
   merely through smaller coefficient keys.
4. Only a surviving complete contract enters native compiler admission, role
   transport and ordinary Experiment benchmarks. Compare selected outputs, all
   links, cold/warm horizons, CPU and memory on matched topologies.

Rust owns numeric/cryptographic execution and one-use state. Python owns SDK
records, search/accounting, orchestration and independent float oracles.

## Reproduce and SDK

```sh
.venv/bin/python scripts/probe_token_network_budget.py --output docs/evidence/token-network-hypotheses-qwen25-2026-10-02.json
```

This uses cached public configuration and retained quality evidence, without
loading weights. APIs: `pllm.metrics.TokenNetworkBudgetProbe` and
`pllm.assurance.PublicPolynomialShiftRegression`. Their
[SDK guide](docs/content/docs/sdk/evaluate/network-hypotheses.mdx) links canonical
API documentation. Neither is an inference Pipeline component.
