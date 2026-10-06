# Audit: the 60.10 GB piecewise-block projection versus the papers

**The arithmetic is correct for PLLM's current reference. It is not a
paper-faithful SIGMA/FuseFSS cost estimate or a lower bound on those methods.**
The earlier rejection applies to the measured PLLM construction. It cannot
justify rejecting the papers' optimized constructions before implementing them.

## Reconcile the PLLM number

Evidence: `docs/evidence/research-piecewise-gated-reference-qwen25.json`, Q9,
64 pieces, vector lookup, 64 lanes. The compiler counts

```text
E = 24 layers × 4,864 gated outputs × (39 prefill + 7 decode rows)
  = 5,369,856 elements.
```

The reported **358,144 bytes are per party per 64 elements**. They are raw
key/mask/triple payloads, counted from the material owners, not allocator size
or an expanded DCF evaluation tree. In `compact_dcf.rs` a key contains a
16-byte root, one correction per input bit, and a 4-byte final word:
`20 + 22 × bits` bytes. A 32-bit comparison is therefore 724 bytes per party.
`interval_fss.rs` shares that key across public thresholds within the same
one-use lane and width. It does not allocate one key per SiLU interval.

| Material | Bytes/party/element | Both parties × E, decimal GB |
| --- | ---: | ---: |
| Four exact rescalings | 4,836 | 51.937247232 |
| Vector coefficient lookup | 736 | 7.904428032 |
| Two Beaver triples | 24 | 0.257753088 |
| **Total** | **5,596** | **60.099428352** |

Thus `2 × (358144 / 64) × 5369856 = 60,099,428,352` bytes: **60.10 GB**,
or **55.97 GiB**, combined; **30.05 GB per party**. There is no extra doubling
of an already combined count. Rescalings account for **86.42%**.

The 39+8 projection executes 46 rows, not 47: the final selected token has not
yet been fed into the decoder. It is a geometry projection from bounded native
cases, not a 60 GB allocation or a measured whole-decoder run.

The separate **465.16 MB peer-body** count covers the six-round block layout.
Raw material, its eventual distribution traffic, online bodies, and peak
resident memory are different quantities. Streaming can reduce resident memory
without eliminating one-use material. No compressed distribution or full-wire
measurement was performed. Adding raw material to bodies is an explicit
uncompressed-delivery cost screen, not a measured network total.

## What the primary sources actually report

These are the papers' reported values and units, not normalized PLLM results:

| Source | Workload | Reported offline key material |
| --- | --- | ---: |
| [SIGMA, Appendix L/Table 9](https://eprint.iacr.org/2023/1269.pdf) | GPT-2 | 14.29 GB |
| Same table | Llama2-7B | 255.41 GB |
| Same table | Llama2-13B | 419.01 GB |
| [FuseFSS, Table 1](https://arxiv.org/pdf/2606.09551) | GPT-2, 128 tokens | 11.920 GB |
| FuseFSS, Appendix B.2/Table 8 | LLaMA-7B, 16 tokens | 68.95 GB |
| Same table | LLaMA-7B, 64 tokens | 134.19 GB |

SIGMA Appendix L explicitly says dealer-to-server key transfer dominates
preprocessing. FuseFSS Table 8 explicitly describes total offline material,
host-side key buffers and streaming. Large offline volumes are therefore real
reported costs; small online communication does not imply small offline keys.
However, these full-model workloads, numeric policies and accounting scopes
cannot validate or replace a Qwen MLP-only estimate. SIGMA's primitive key-size
definition in Sections 2.3/2.4 is **one party's key**; keep table totals in their
published scope rather than silently doubling or halving them. FuseFSS Appendix
B.2 also explicitly calls its online communication figures per-party.

Pinned local PDF SHA-256 values used for this audit:

- SIGMA: `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
- FuseFSS: `6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
- FSS for Mixed-Mode Secure Computation:
  `dc94c63a77a5cb52cfd12b186a327d045de4a1fc662060c1de6412b334f4fcf4`.

## Material differences from the published protocols

1. **Rescaling semantics and backend.** PLLM uses four generic 32-bit
   ties-to-even rescale-plus-sign helpers, with arithmetic-output DCF keys at
   several widths. SIGMA Section 4.2 uses one-bit-output, early-terminated DPF
   comparisons, faithful truncation/arithmetic shifts, and a cheaper
   guaranteed-gap protocol where a public bound proves the gap. These are
   different algorithms and rounding contracts. For a concrete scale check,
   SIGMA Theorems 1/3 give `DPF(7,1) + 3×32 = 128+96 = 224` bits, **28 bytes per
   party**, for 32-bit shift-by-seven GapARS. Our corresponding ties-even/sign
   helper uses `(724+174+196)+16 = 1,110` bytes. The paper's already-masked-wire
   input and gap precondition are required; this is not a drop-in replacement
   or a complete-block estimate.
2. **SiLU approximation.** SIGMA Appendix F computes
   `ReLU(x) − delta(x)`, using an even residual and a **1,024-entry LUT**, with
   precision 12 and tails outside `(-16,16)`. PLLM's rejected reference uses
   Q8/Q9, 16/64 affine pieces on `[-8,8]`, two coefficient shares and an extra
   secret slope multiplication. It is not that SIGMA activation protocol.
3. **Wire representation.** SIGMA maintains masked wires between operators;
   its Section 4.1 multiplication operates on already-masked inputs. PLLM's
   standalone additive-share helpers separately open newly masked operands.
   The latter's two-Beaver-opening floor is specific to that layout.
4. **Effective widths and packing.** SIGMA Section 5.4.1 derives narrower
   comparison widths from preceding truncation; Section 6 packs nonstandard
   widths. PLLM's bounded block deliberately keeps a 32-bit share interface.
   Key/control packing and operator-specific widths have not been reproduced.
5. **FuseFSS is a compiler contract, not a free nonlinear oracle.** Appendix
   I.1 explicitly permits one comparison key per query in a batched call;
   I.2 describes vector lookup. Appendices N/O retain Beaver products,
   Boolean operations and conversions. Two backend calls do not mean two
   fixed-size keys per whole tensor. But our choice of backend, rounding and
   helper boundaries does not reproduce the paper's optimized gate costs.

## Corrected implementation priority

First reproduce the **SIGMA SiLU residual lookup, DPF-based shift/truncate
primitives and masked-wire composition** in bounded native references. Prove
any gap and effective-width assumptions from public numeric contracts. Preserve
faithful-shift and ties-to-even identities separately; test each on fresh
held-out trajectories before using it in a decoder. Then measure complete
SiLU × up costs, including conversions and the final product/rescale, and apply
FuseFSS-style fusion to that baseline.

The current piecewise and dense joint-table/half-gate rejections remain valid
for their specified layouts. **A paper-faithful Qwen cost remains unknown.**
Those rejections do not establish that a new cryptographic construction is
necessary before investigating the existing papers' cheaper primitives.

## Reproduce the accounting

```bash
uv run python - <<'PY'
import json
from pathlib import Path
r = json.loads(Path('docs/evidence/research-piecewise-gated-reference-qwen25.json').read_text())
c = next(c for c in r['cases'] if (c['fraction'], c['pieces'], c['layout'], c['lanes']) == (9, 64, 'vector', 64))
p = next(p for p in r['geometry_projections'] if p['output_tokens'] == 8)
parts = [c[k] for k in ('rescale_key_payload_bytes', 'lookup_key_payload_bytes', 'triple_key_payload_bytes')]
assert sum(parts) == c['party_key_payload_bytes'] == 358144
assert p['gated_elements'] == 24 * 4864 * (39 + 8 - 1)
total = 2 * sum(parts) * p['gated_elements'] // c['lanes']
assert total == 60099428352
print({'both_party_raw_bytes': total, 'rescale_fraction': parts[0] / sum(parts)})
PY
```
