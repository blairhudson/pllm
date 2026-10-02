# Projected polynomial correlation gates

Three combined design changes are executable in a bounded Rust reference:
derive linear shifted coefficients from the party's own mask, contract constants
through the public output matrix, and seed one party's random coefficient shares.
Independent integer oracles check every ablation at 24/32/64 bits. This is a
modular numerator without rounding, rather than a protected Qwen MLP.

The native 8-row 32→128→32 fixture reduces key bodies **4.56×**, or **3.90×**
including peer openings. It does not reduce those opening bodies. The same
native size formulas are bound to complete Qwen2.5 prefill/decode schedules by
`ProjectedPolynomialCostProbe.project`; no full-width keys are allocated.

## Further combination: client-opened masked source

At a client-owned nonlinear boundary, the trusted client could receive the two
fresh mask seeds from the dealer, then send `x+r` to both workers directly.
Each worker retains only its own mask and correlation shares. This removes
the redundant sequence of client input-share delivery followed by a peer opening.
We price the additional dealer→client seed channel. This is a **hypothesis**:
the direct-input contract and independent role transport are unimplemented.

## Compiler-bound results

Known **covered-body floors**, decimal MB, including both parties' fresh material:

| Cohort | Assumed ring | Direct client-masked online | Known all-link | Known body-matrix MACs / two-offset |
| --- | ---: | ---: | ---: | ---: |
| 39+8 | 24 bits | 11.91 | 47.16 | 1.962× |
| 39+8 | 32 bits | 15.87 | 62.85 | 1.962× |
| 39+8 | 64 bits | 31.69 | 125.59 | 1.962× |
| 39+32 | 24 bits | 18.20 | 71.97 | 1.962× |
| 39+32 | 32 bits | 24.22 | 95.84 | 1.962× |
| 39+32 | 64 bits | 48.30 | 191.32 | 1.962× |

Ring width is an assumption, not a checkpoint range certificate. Native
whole-ring parity does not imply signed overflow safety. Contraction cannot
cross a per-channel truncation that the original model requires.

The 39+32 prepared control has 113.55 MB online and 178.97 MB warm all-link
covered bodies. Its tenfold budgets are 11.35/17.90 MB. Every tested layout
fails that target on known bodies. The 24-bit hypothesis remains **inconclusive**
at a 50% reduction; 32-bit remains inconclusive at 25%. Neither establishes
full-model numeric parity or resource admission.

A separate same-body/count two-offset cohort records 227.08 MB online, under a
different 64-token plan bound. Its cold all-link total is unknown. A superficial
comparison with the hypothetical 18.20 MB online floor suggests 12.5×, but drops
53.77 MB of fresh correlations/seed delivery and every missing protected numeric
operation. **That is not a tenfold inference-network result.**

Coefficient derivation trades bytes for extra private-mask matrix projections;
constant contraction adds dealer matrix work. The known client-attention candidate
already uses 1.962× the two-offset body's matrix MACs, before nonlinear/crypto
work. MACs are not interchangeable with measured CPU, and no compute cap passes.

## Reproduce

```sh
.venv/bin/python examples/benchmarks/projected_polynomial.py
.venv/bin/python scripts/probe_region_contract_cost.py --max-output-tokens 8 --projected-polynomial
.venv/bin/python scripts/probe_region_contract_cost.py --max-output-tokens 32 --projected-polynomial
```

Only the last two commands need the already-cached pinned public config. Raw
reports: `projected-polynomial-ablation-2026-10-02.json` and
`projected-polynomial-qwen25-{8,32}-2026-10-02.json`. Code hashes, plan,
composition, schedule and baseline body commitments are retained. Material,
prompts, token IDs, logits and private tensors are excluded.

## Held-out numeric gate

Eight new public calibration prompts and eight disjoint evaluation prompts were
locked before evaluation. All 24 SiLU activations were replaced in the unchanged
pinned float32 checkpoint. Each prompt checks prefill and one teacher-forced
decode; candidate caches evolve under their own numeric path. Fits minimize
public per-channel gated-output error using bounded offline least squares.

| Float oracle | Prefill top-1 | Same-token decode top-1 | Mean / worst active quantized residual |
| --- | ---: | ---: | ---: |
| Fixed Taylor quadratic | 0/8 | 0/8 | 24.75% / 61.57% |
| Public per-channel affine | 0/8 | 0/8 | 56.69% / 90.32% |
| Public per-channel quadratic | 6/8 | 5/8 | 18.66% / 71.57% |
| Public per-channel quartic | 4/8 | 3/8 | 2.39% / 14.39% |

No candidate passes. The latter three are more flexible than the current native
numerator, yet still do not preserve checkpoint decisions. Higher polynomial
degree improves local residual sparsity but worsens checked rollout drift; local
fit error is not a whole-decoder certificate.

Residual counts compare quantized gated outputs on *unmodified reference*
trajectories, with the true per-row scale supplied for free. This is optimistic
input to a new sparse-correction hypothesis, not private selection, protected
quantization or a sparse execution result. The observed maximum is not a public
worst-case capacity certificate. No input-dependent coordinate lists may simply
be exposed to a worker.

```sh
.venv/bin/python scripts/probe_projected_polynomial_quality.py --output docs/evidence/projected-polynomial-quality-qwen25-2026-10-02.json
```

The bounded command uses cached weights and Torch as an independent float oracle;
it does not time Python as a serving kernel. Native execution remains in Rust.
Report binds checkpoint bytes, public dataset, token cohort, profile and code
digests, and archives aggregates only. **Stop this polynomial-only decoder
construction.** Retain the tested coefficient compression as a primitive;
further investigation must supply exact residual correction or another numeric
contract, then satisfy the complete byte and compute gates above.
