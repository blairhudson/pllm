# One-evaluator protected-region cost gate

The preferred serving graph keeps Preparation offline and has one online
Inference provider. Existing complete LogRow and Compact half-gates operators
have a trusted client garbler and one evaluator, making their ownership relevant
to this graph. Their current implementations are bounded in-process operators;
an offline Preparation garbler is a separate, unimplemented composition.

This screen executes those installed SDK operators and binds their scalar costs
to the compiler's semantic gated-MLP geometry. It checks complete schedule coverage,
the pinned plan/composition/schedule identities and the actual response bound.
Decode contributes **N−1** executed rows. No checkpoint tensor values or protected
decoder are loaded, and the changed Q18/Q9 or Q18/Q12 arithmetic does not inherit
the control's W8A8 numeric identity.

## Fresh scalar reproduction and projected ownership

Three fresh processes each execute three one-lane samples, checking **nine outputs**
against the installed independent integer oracles. These small reproductions check
the cost reports; broader arithmetic and checkpoint-quality evidence remains in
[LogRow](logrow-product-reference.md) and [Compact](compact-product-reference.md).
Compact uses its default public calibration: one quadratic piece or eight cubic
pieces. These are distinct profiles from historical density-calibrated cohorts.

The pinned Qwen2.5-0.5B plan has 24 layers and 4,864 gated elements per layer row.
Its 39+8 and 39+32 geometries contain **5,369,856 / 8,171,520** gated elements.
Each projected element needs fresh one-use material; serial retirement bounds
live allocation but does not reduce cumulative material.

| Operator | Evaluator material, 39+8 | Evaluator material, 39+32 | Client encodings, 39+8 / 39+32 |
| --- | ---: | ---: | ---: |
| LogRow Q9 | 326.40 GB | 496.70 GB | 12.89 / 19.61 GB |
| Compact Q12, one quadratic piece | 1.818 TB | 2.766 TB | 13.40 / 20.40 GB |
| Compact Q12, eight cubic pieces | 2.925 TB | 4.451 TB | 13.40 / 20.40 GB |

All units are decimal. Evaluator material includes lookup/arithmetic bodies and
constant labels exactly once. Client encodings are a separate local owner when
the client issues material; they are **not** added to that layout's network floor.

Two hypothetical delivery layouts make the distinction explicit:

- **Client-issued:** client → Inference carries evaluator material. Input/output
  labels remain an additional online boundary.
- **Offline Preparation-issued:** Preparation → Inference carries evaluator
  material, and Preparation → client carries the client's encoding material.
  Authenticated distribution, garbler separation, decoding ownership and erasure
  would need their own protocol before this layout could execute.

Online entry/exit estimates remain separate: **6.79 / 10.33 GB** for LogRow and
**7.05 / 10.72 GB** for either Compact profile. Public circuit metadata and the
once-per-profile fitting metadata also remain separate estimates. The screen
does not count estimates as measured frames or reinterpret zero evaluator-peer
frames as zero network traffic. Whole-response CPU/RSS, physical wire, TTFT and
decode throughput are unknown; local scalar timings are not extrapolated.

## Decision

The historical 39+8 / 39+32 prepared controls account for **116.84 / 178.97 MB**
of warm all-link application bodies. Their tenfold budgets are **11.68 / 17.90 MB**.
Every measured operator layout exceeds even the full control budget on evaluator
material alone, before input/output labels, private quantization, protected float
conversion, linear/label conversion, attention, normalization, KV or transport.
Moving issuance to Preparation changes ownership but does not remove these bytes.

Even a sensitivity scenario charging only one fresh uncompressed half-gate AND
per gated element needs **171.84 / 261.49 MB** of ciphertexts. This is a conditional
floor for that particular layout, **not** a lower bound on all protected protocols
or compressed/correlated constructions. No new general impossibility claim follows.

The control supplies a cost budget, not a matched-quality ranking. Existing Qwen
numeric diagnostics remain imperfect, and the protected composition cannot
reuse their clear-checkpoint runs as a live decoder result. Mixed and masked-wire
two-worker methods retain their separate online topology; they cannot silently
replace the one-provider requirement.

## Replay

Use an installed wheel with the pinned official config already in the shared
Hugging Face cache. The exact command was run against the installed SDK:

```sh
python scripts/probe_prepared_region_gate.py --samples 3 --output prepared-region-gate.json
```

Evidence: `docs/evidence/research-prepared-region-gate.json`. It retains full
replayable operator configurations, independent-oracle counts, native/SDK/code
digests, config/control hashes, plan/schedule/composition identities, role-specific
costs and every unpriced obligation. Tests in `tests/test_prepared_region_gate.py`
check model-neutral Qwen2/Qwen3 geometry, final-token exclusion, both delivery
graphs, material conservation, rejected role substitutions and the rule that
generous budgets never admit missing work.
