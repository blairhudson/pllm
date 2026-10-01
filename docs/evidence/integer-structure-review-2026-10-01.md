# Exact modular/integer structure screen — 2026-10-01

## Decision

**No-go: sub-minimum-dimension exact factors on all nine sampled matrices.**
Every matrix has an independently verified odd 896-by-896 minor. Therefore any
exact `W = A B mod 2^w`, for any `w >= 1`, requires factor width at least 896.
For qkv and gate-up this equals the entire original input width: no narrower
linear ingress. Down projections can reduce 4,864 input coordinates to 896;
the trivial exact choice `B=W, A=I` performs the entire original down projection
on the client. Dense minimum-width factors also cost 100% of original down MACs
on the client. This does not prove all alternative addition circuits expensive.

**No-go: screened cheap-structure-plus-small-sparse-residual routes.** Best
screened residuals remain **96.5223%–97.2030% nonzero**, with every original input
column and output row active. Sparse storage and full-width residual ingress
erase the proposed benefit. Arbitrary low-rank-plus-sparse decompositions and
larger exact circuits remain open; these finite screens do not prove their
impossibility.

**No-go: tenfold network via these factors while retaining existing per-stage
full-output openings/corrections.** Even granting zero ingress, free projection,
and unattainably favorable uniform 16-bit outputs misses both control budgets.
Bounded adjacent-pair common subexpressions save **0.8012%–1.1807%** of dense
remote products, with **zero input/output network reduction**.

## Source and scope

- Cached public checkpoint: `Qwen/Qwen2.5-0.5B-Instruct`.
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Checkpoint digest: `f69322057253c4fb853c10e73da5d84111262e0fb1edba33933b9e5b8caf59fd`.
- Source-lock digest: `383f5ed6947ba56ceed6083a9503d6302f09559e1e57ee614409900b36084c29`.
- Plan digest: `62381dbd11470bd150723cfb0d7bab096c52f331ec8e9f5dd23e178df28a0a4d`.
- Schedule digest: `bf3bc2e91b6e95a8d0b9ac7eebfb2866a3b3b8c2fe4c2ff401dc625f130a0d5c`.
- Layers **0, 12, 23**, each qkv, fused gate-up, down: nine real matrices.
- Existing `SymmetricPerRow` W8A8 quantizer and existing stage/source loaders;
  matrices oriented **output-by-input**. No approximate fitting or SVD.
- `Model.hf(..., local_files_only=True)` plus offline environment flags. No
  download, training, upstream runtime import, or added dependency.
- One stage resident at a time, one native/BLAS thread. Largest W8 stage is
  **8,716,288 bytes**; measured process peak RSS **521,404,416 bytes
  (497.25 MiB)** includes source-loading and diagnostic scratch memory. Wide
  reconstruction checks run in 64-row chunks. The whole decoder body is never
  loaded. Recorded run completed in **7.413 seconds**; timing/RSS vary by run.
- 39 input + 32 output tokens corresponds to **70 executed stage rows**. This
  algebra probe does not execute or measure a private response.

Semantic `StageSpec` objects initially carry default W4A4 fields. The probe
explicitly replaces those fields with W8A8, matching `engine.load`. Independent
chunk re-quantization checks each fused source's row offsets and its original
per-output-row scales. Per-matrix W8 and scale hashes are recorded. This source
lock matches earlier controls; the full-body fingerprint is not recomputed by
this sampled probe.

## Exact certificates and measured structure

| Layer | Stage | Shape `m × n` | Odd-minor size | Best residual nonzero | Residual exact-lift bits | Adjacent-pair product saving |
|---|---|---:|---:|---:|---:|---:|
| 0 | qkv | 1,152 × 896 | 896 | 96.8308% | 24 | 1.0134% |
| 0 | gate-up | 9,728 × 896 | 896 | 97.2030% | 24 | 0.8123% |
| 0 | down | 896 × 4,864 | 896 | 96.7898% | 26 | 1.0544% |
| 12 | qkv | 1,152 × 896 | 896 | 97.0076% | 24 | 0.9101% |
| 12 | gate-up | 9,728 × 896 | 896 | 97.0432% | 24 | 0.8992% |
| 12 | down | 896 × 4,864 | 896 | 96.5223% | 26 | 1.1807% |
| 23 | qkv | 1,152 × 896 | 896 | 96.7346% | 23 | 0.9564% |
| 23 | gate-up | 9,728 × 896 | 896 | 97.1945% | 24 | 0.8012% |
| 23 | down | 896 × 4,864 | 896 | 96.7727% | 26 | 1.0316% |

Best structure is 16-by-16 block-constant in every sample. Other checked exact
decompositions: zero structure, row-modal rank-one, column-modal rank-one, and
anchored additive rank-two (`W[i,0] + W[0,j] - W[0,0]`). All five per matrix
reconstruct **every coefficient** exactly as `structured + residual` using
independent wide-integer addition. Anchored additive structure zeros the first
residual row and column by construction; its support still spans `m-1` outputs
and `n-1` inputs. Other candidates span all inputs/outputs. The best candidate
spans all inputs/outputs in every sample.

Across both axes of all nine matrices: **zero zero-vectors, duplicate vectors,
rational-proportional vectors, or unit-proportional mod256 collisions**. Every
nonzero vector has an odd entry. Unit-proportional equality at any wider ring
would imply a mod256 collision, so its absence excludes those wider unit
collisions too. Nonunit proportionality is not inferred from this normalization.

### Why the modular conclusion is sound

Generator uses packed Python-integer GF(2) row elimination; witness records
**original** row indices and pivot column indices. Checker selects that minor
from original W and uses separate Boolean **column** elimination. Successful
checker establishes determinant odd, hence a unit modulo every `2^w`.

If `W=AB mod2^w`, reducing modulo2 gives `rank_GF2(W) <= k`, with `k` the factor
width. Odd minor supplies lower bound 896. Since 896 equals `min(m,n)`, trivial
identity-oriented factors attain it, so the exact minimum width is 896 for these
samples. Tall matrices have full **column** rank; wide down matrices have full
**row** rank, never full column rank. Witness size never exceeds rectangular
minimum dimension.

In Smith-normal-form language, all minimum-dimension invariant factors are odd
units over these rings. A full integer SNF calculation adds no stronger
dimension certificate here. A deficient GF(2) rank would not establish an exact
factorization over `2^w`; tests include `2I mod256`, whose parity rank is zero
but whose nonzero image needs two generators.

For any rank-`r` structured part, rank subadditivity gives residual GF(2) rank
at least `896-r`. This is only a weak residual-support lower bound, not proof
that every possible residual must be dense.

## Arithmetic, precision, and privacy prices

For dense `A[m,k]`, `B[k,n]`, client projection costs `n*k` MACs per row, remote
expansion `m*k`; dense client fraction of original stage MACs is `k/m`.

| Stage | Minimum `k` | Dense client MACs, 70 rows | Client/original MAC fraction | Dense remote MACs, 70 rows |
|---|---:|---:|---:|---:|
| qkv | 896 | 56,197,120 | 77.7778% | 72,253,440 |
| gate-up | 896 | 56,197,120 | 9.2105% | 610,140,160 |
| down | 896 | 305,070,080 | 100% | 56,197,120 |

These are **dense-factor prices**, not required MAC lower bounds. Tall stages
can instead use `B=I, A=W`, zero client arithmetic and unchanged ingress/remote
work. Down can use `B=W, A=I`, full client arithmetic and zero remote expansion.
Down's public W8 client weights plus original float32 output scales cost
**4,361,728 bytes per selected stage**, before bias, packaging, or distribution.

Generic 32-bit dense B storage would be 3,211,264 bytes for qkv/gate-up, or
17,432,576 bytes for down, excluding A, scales, and indices. No claim that
32-bit factor coefficients are necessary: identity-oriented alternatives retain
original W8 values. Generic centered 32-bit coefficients require **49–52 signed
bits** for unreduced Bx under W8 input bounds, and **73 signed bits** for naive
unreduced expansion after centered u32 projection. Arithmetic must reduce in
the ring safely; blindly applying signed int32/int64 matmul is insufficient.
The small reference uses Python integers and supports rings through `2^64`.

The original activation quantizer runs **before Bx**. Projected coordinates are
not automatically A8 activations and must not be re-quantized. Reconstruct the
original integer accumulator, then apply original activation scale and original
output-row weight scale, then original bias exactly once. Factorizing integer
W modulo256 only guarantees accumulator residues; it does not authorize
mod256 inference with signed integer lifting. Original prepared samples use
24-bit rings for qkv/gate-up and 32-bit rings for down. Best residual bounds need
23–26 signed bits, not a two-/eight-bit residual path.

For a bounded W8 input domain, residual certificate is
`127 * max_i sum_j abs(R[i,j])`; exact centered lifting requires this strictly
below `2^(w-1)`. Residual coefficients can exceed W8, so the report prices
u16 values and u16 column indices plus u32 row pointers. Best residual CSR
storage is approximately 3.86–3.89 times original dense W8 bytes, before
structured coefficients.

The explicit route estimate freshly masks projected ingress, residual ingress,
and final outputs, with one full-output correction. Best residual has all
original input columns active: it retains full-width input masking and
full-width output/correction bodies. Additional structured projection does not
remove that path. Rank-one projection may use fewer unmasked coordinates but
does not make residual ingress cheap. Public common expressions such as
`x[2j] ± x[2j+1]` can be computed remotely from fully masked inputs; they do not
reduce independent input dimensions or output width. Their public chain index
bytes are only chain descriptors, not a complete encoded evaluation program.

No new cryptographic protocol or one-use mask implementation is claimed.
Proof of privacy, mask refresh, bias/scale transport, feedback, and broader
topology interactions remain separate requirements. Injectivity/rank arguments
concern unrestricted ring modules; they do not claim W8 activations contain
`n*w` bits of entropy.

## Network gates against both controls

Existing stage route arithmetic bodies are priced optimistically as
`rows * (input_width + output_width) * bytes_per_element` online, plus another
full-output correction all-link. Framing, tickets, model distribution, scales,
bias, control, retries, and client CPU are omitted.

| 39+32 control | Baseline all-link | Baseline online | Tenfold all-link budget | Tenfold online budget |
|---|---:|---:|---:|---:|
| Original, 96 remote stages | 178,970,558 | 113,545,024 | 17,897,055 | 11,354,502 |
| Attention-owned, 48 MLP stages | 148,297,986 | 93,221,048 | 14,829,798 | 9,322,104 |

| Favorable shape-only thought experiment | Original all-link / online | Attention-owned all-link / online |
|---|---:|---:|
| **Zero ingress**, uniform u16 output + correction | **85,155,840 / 42,577,920** | **71,393,280 / 35,696,640** |
| Zero ingress, uniform u32 output + correction | 170,311,680 / 85,155,840 | 142,786,560 / 71,393,280 |
| Every stage full-minimum-rank ingress, uniform u32 | 194,396,160 / 109,240,320 | 154,828,800 / 83,435,520 |

Uniform u16 is unattained for these real W8A8 accumulators and deliberately
favors the proposal. Even that output-only floor is above tenfold budgets. Thus
removing every ingress element cannot yield tenfold reduction under retained
per-stage output/correction semantics. These are schedule-shape calculations,
**not** measured candidate wire traffic or lower bounds for redesigned
protocols. The full-minimum-rank row assumes corresponding rank for unselected
layers only as a hypothetical projection; measurements certify nine matrices.

## Verification and reproduction

From repository root:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=python .venv/bin/python scripts/probe_integer_structure.py --output docs/evidence/integer-structure-screen-2026-10-01.json
.venv/bin/python -m pytest tests/test_integer_structure.py -q
.venv/bin/python -m ruff check scripts/probe_integer_structure.py python/pllm/runtime/integer_structure_reference.py tests/test_integer_structure.py
```

**27 tests passed.** Tests cover independent scalar factor/residual products,
nonuniform output scales and bias, modulo-versus-integer distinctions, 64-bit
overflow, invalid shapes/types, sound rectangular witnesses, exhaustive 3-by-3
Boolean determinant parity, deficient parity rank, planted duplicate/structured
cases, client compute/mask/residual cost gates, and measured artifact consistency.
Real-source run independently validates every odd minor and every residual
reconstruction. Runtime and process RSS are recorded, not treated as stable
reproducibility fields.

Files:

- `scripts/probe_integer_structure.py`
- `python/pllm/runtime/integer_structure_reference.py`
- `tests/test_integer_structure.py`
- `docs/evidence/integer-structure-screen-2026-10-01.json`
- `docs/evidence/integer-structure-review-2026-10-01.md`

Remaining open: exact larger addition chains, different structured families,
bounded integer/common-subexpression searches beyond adjacent pairs, and
protocol redesign eliminating full-output openings. None receives a runtime
choice or promotion from this evidence.
