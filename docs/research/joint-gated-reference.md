# Joint SiLU × up: numeric mapping and pre-issuance cost gate

This is a **clear numeric reference and layout-specific rejection**, not an
executable SDK component, FuseFSS reproduction or protected Qwen benchmark.
It follows the complete-block rejection in
[`piecewise-gated-reference.md`](piecewise-gated-reference.md). The aim is to
replace separate secret products and rescalings with a single output-share
function, while counting its actual representation rather than granting free
nonlinear evaluation.

Implementation: `crates/pllm-core/src/joint_gated_reference.rs` and
`crates/pllm-core/examples/joint_gated_probe.rs`.
Frozen configuration and reproduction:
`benchmarks/research/joint_gated_reference.py`.
Evidence: `docs/evidence/research-joint-gated-reference-qwen25.json`.

## Frozen public numeric contract

Four profiles combine **Q12/Q16 inputs and outputs** with **512/2,048 uniform
SiLU secant intervals** on `[-16,16]`. Slopes have 28 fractional bits;
intercepts have `28+f`. The tails are zero below −16 and identity at or above
+16. Gate and up values must lie in `[-48,48]` and `[-128,128]`; non-finite or
out-of-domain inputs reject. There is no clipping or prompt-dependent range.
Coefficients come only from the mathematical function, not calibration traces.

For encoded inputs `g,u`, the chosen interval supplies integer coefficients
`a,b`. The complete output is

```text
q = ties_even(((a*g + b)*u) / 2^(28+f)).
```

There is **one final rounding**, after the gated product. This changes numeric
identity relative to separately rounding SiLU and then multiplying. The source
float32 inputs first round directly to Qf, and the result returns through a
float32 boundary. Neither this conversion nor protected range enforcement has
a share-executable decoder binding.

The widest admitted numerator exceeds signed 64 bits. Rust uses checked i128;
an independent Python oracle decomposes the affine value into quotient and
remainder before multiplying, keeping each NumPy intermediate within i64.
Unbounded Python integers additionally check every interval edge, signed
extrema and tails. The immutable profile digest binds every coefficient, bound,
precision and tail. Public corner extrema bound all interior numerators because
the function is bilinear within each interval.

Twelve new public prompts and four profiles were fixed before evaluating the
pinned Qwen2.5 W8A8 checkpoint. Each candidate replaces every semantic MLP gated
block and follows the control's same-token path through one prefill and three
decode positions. Complete logits and full KV are checked independently of
selected-token agreement. This is a narrow approximation screen against PLLM's
W8A8 control, not upstream float32 accuracy or representative generation quality.

| Frozen profile | Prefill selections | Decode selections | Worst absolute logit error | Mean top-5 recall |
| --- | --- | --- | --- | --- |
| Unmodified control | 12/12 | 36/36 | 0 | 1.0000 |
| Q12, 512 pieces | 11/12 | 35/36 | 3.1944 | 0.9167 |
| Q12, 2,048 pieces | **12/12** | **36/36** | 2.7799 | 0.8875 |
| Q16, 512 pieces | 11/12 | 35/36 | 2.8293 | 0.9125 |
| Q16, 2,048 pieces | 11/12 | 35/36 | 4.5401 | 0.9083 |

All profiles stay within their declared domain, but **none** preserves exact
logits or KV at any checked position. Q12/2,048 passes only the predeclared
selection screen. Each candidate checks 63,387,648 gated outputs against the
independent integer oracle, for **253,550,592** total. Zero new swap was observed.
Higher precision is not a monotonic quality improvement through the surrounding
W8A8 quantization. No coefficients or thresholds were retuned after these results.

## Fully specified direct-output candidate

The candidate is deliberately simple enough that its complete raw storage can
be calculated before allocating any material.

Let the input rings have `kg=f+7` and `ku=f+9` bits. The inclusive positive
up endpoint `+128` needs the extra sign bit; an `f+8`-bit ring cannot encode it.
Output shares use the 32-bit ring. Define the total public function `F(g,u)` by
signed decoding and the numeric contract above, with dummy zero outside its
public valid domain. Honest inputs must satisfy the source-bound range contract;
the dummy entries are not clipping permission or a malicious-input proof.

For **each single use**, a trusted research dealer would:

1. Sample independent uniform masks `r,s` in the input rings and split each
   into two additive shares.
2. Sample a uniform 32-bit value `A[z,w]` independently at **every** cell of the
   `2^(kg+ku)`-entry table.
3. Set `B[z,w] = F(center(z-r), center(w-s)) - A[z,w] mod 2^32`.
4. Give party 0 only `A,r0,s0` and party 1 only `B,r1,s1`.

Party `i` opens its contributions `gi+ri` and `ui+si`. Both reconstruct the
masked indices `z,w`; party 0 retains `A[z,w]` and party 1 retains `B[z,w]` as
their opaque output shares. Their sum is the correctly rounded joint result.
The table itself incorporates piece selection, signed wraps and final rounding.
It does not expose the selected piece to either party.

Each party's table is marginally uniform, as are the missing mask shares. A
single party cannot simply reconstruct the input from the masked indices and
its own key. This elementary distribution argument assumes a trusted dealer,
honest execution, fresh masks and no collusion; it is not an independently
reviewed protocol or a malicious-security claim. A real implementation would
also need authenticated session/profile framing and non-cloneable one-use state,
burned on use, cancellation or failure. Reusing masks exposes input differences.
**No such protected material or transport is implemented or issued here.**

## Exact cost vetoes

The compiler supplies **5,369,856** gated elements for 39 input plus eight
generated tokens: `24 × 4,864 × (39+8−1)`. Historical prepared control covers
**116.84 MB** across the whole response. This permits about 21.76 bytes per
MLP element if every other decoder operation were free. It is a permissive
resource screen, not a matched runtime ranking or full-wire comparison.

| Concrete layout | Raw requirement | Decision |
| --- | --- | --- |
| Q12 dense joint shares | `2^40` cells, **8.80 TB of table shares per element** across both parties | Reject before allocation |
| Q16 dense joint shares | `2^48` cells, **2.25 PB per element** across both parties | Reject before allocation |
| Per-lane uncompressed half-gates | Even **one** AND per element costs **171.84 MB** of ciphertext bodies at 39+8 | Reject before circuit construction |
| Public shifted-polynomial coefficients | Reveal masks in the native bilinear toy | Reject privacy shortcut |

The half-gates floor uses the existing 16-byte-label representation: two
ciphertexts per AND, 32 bytes total. For every profile, a native-checked truth
table witness has `F(0,0)=F(1,0)=F(0,1)=0` and `F(1,1) != 0`, using encoded
real one. Thus XOR/NOT alone cannot implement this function. The floor omits
all remaining gates, input labels/transfer, sharing conversions and output
handling. It rejects this uncompressed per-lane layout, not every garbling,
FSS, vector correlation or secure-computation construction.

Even an **unimplemented compressed joint key** still needs the chosen layout's
two masked input openings. Byte-packed narrow rings cost **64.44 MB** for Q12
or **75.18 MB** for Q16 at 39+8. Only about **9.76/7.76 bytes per element** remain
for *all* key material under the permissive whole-response budget. Framing,
conversions, other operators and independent issuance are still extra. A
tenfold 11.68 MB target is already smaller than those openings. A compressed
key with the required properties and costs has not been supplied; its cost
and admission remain **unknown**, never zero.

The evidence also prices 39+32 geometry, both parties' table/mask owners, peer
openings and extra input/output share delivery if state is not already resident.
CPU, peak memory, framing and full wire remain unknown because the raw lower
bounds reject issuance before any protected prototype is warranted.

## Carry, rounding and privacy counterexamples

A bounded native audit exhausts **4,096** combinations of signed toy inputs and
masks. Write `z=(g+r) mod R` and `g=z-r+R*c`, likewise
`u=w-s+R*d`. For an affine numerator, the missing terms are

```text
(a*g+b)*u = (a*z+b-a*r)*(w-s)
           + R*(a*c*(w-s) + d*(a*z+b-a*r)) + R²*a*c*d.
```

The audit checks the full identity before and after ties-to-even rounding and
finds **1,664** wrong outputs when the wrap terms are dropped. Correcting only the
coefficient lookup does not remove these secret terms.

Rounding shares independently also fails: `RTE(3/4)+RTE(−1/4)=1`, while
`RTE((3−1)/4)=0`. Arithmetic rounding cannot be removed by distributing it over
additive shares.

Finally, the bilinear toy's shifted numerator is
`z*w − s*z − r*w + r*s`. Publishing its coefficients publishes `−s` and `−r`;
the native witness recovers both original inputs from the masked openings.
Keeping coefficients secret moves work into an actual FSS/MPC construction;
it does not justify free public evaluation. These are exact counterexamples,
not attacks on the cited FuseFSS implementation.

## Reproduction

```bash
uv run python -m benchmarks.research.joint_gated_reference \
  --output .artifacts/joint-gated-reproduction.json
cargo test --locked -p pllm-core joint_gated_reference --lib
uv run pytest -q tests/test_joint_gated_reference.py
```

The runner refuses to overwrite evidence, checks source/body identity, requires
4 GiB available beyond the host reserve, monitors memory pressure/swap and owns
its bounded numeric subprocesses. Protected table sizes are integers in a
preflight calculation, never allocations. Native code owns numeric execution;
Python owns orchestration, independent oracles and the report.

Next admission requires a fresh held-out numeric pass and a concrete protected
construction with complete costs below the budget. The subsequent
[primary-paper audit](fss-paper-cost-audit.md) prioritizes reproducing SIGMA's
DPF-based shifts, residual SiLU lookup and masked-wire composition before
assuming a new joint construction is necessary. Higher precision alone cannot
authorize tensor, provider or SDK integration.
