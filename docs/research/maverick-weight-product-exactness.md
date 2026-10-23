# Maverick D32: conditional exactness of the FP8–BF16 weight product

**Conditional mathematics only.** This is an increment to the
[D31+D32 selected map](maverick-numeric-source-recovery.md#32-exact-scalar-and-matrix-rules),
not a new importer, numerical choice, threshold or implementation authorization.
The existing sufficient premise is that the stored scale is finite positive BF16
with `0<S<=2^45`, promoted exactly to F32 as approved in
[D32](../../paper/methods/DECISIONS.md#d32-maverick-pinned-bf16-scale-storage-amendment).
No actual source scale/value, checkpoint or whole-graph/no-wrap condition is
certified. The index-size discrepancy and all source/quality gates remain.

## 1. Theorem and exact format cases

Let `v` be any finite selected E4M3FN value, including either signed zero. Let `S`
satisfy the premise above. Then `R32(v*S)` is **exactly** the real product; it has
no F32 rounding error, underflow loss or overflow. If nonzero,

```text
2^-142 <= |v*S| <= 448*2^45 = 7*2^51 < 2^54.
```

For nonzero positive magnitudes write `v=m*2^t`, `S=n*2^u`:

| Format case | Integer significand | Exponent |
| --- | --- | --- |
| E4M3FN subnormal | `1<=m<=7` | `t=-9` |
| E4M3FN normal, exponent field `e=1..14` | `8<=m<=15` | `t=e-10` |
| E4M3FN exponent field `e=15`, finite | `8<=m<=14` | `t=5`; `m=15` is NaN |
| BF16 subnormal, positive | `1<=n<=127` | `u=-133` |
| BF16 normal, exponent field `a=1..171` | `128<=n<=255` | `u=a-134` |
| BF16 upper endpoint allowed by the existing premise | `n=128` | `u=38`, exactly `S=2^45` |

These are mathematical classes, not a serialized-byte decoder. They partition
126 positive nonzero E4M3FN values and 22,016 positive BF16 scales through `2^45`.
There are 16 weight classes and 173 scale classes: **2,768 class pairs**, rather
than a sweep over millions of products or any model values.

## 2. Proof, including F32 subnormals

The positive product is `q*2^k`, where `q=m*n<=15*255=3825<2^12` and
`k=t+u>=-142`. Exact promotion changes neither operand's rational value; each
finite FP8 operand also has an exact F32 representation.

For a **normal** result, removing trailing zeros from `q` cannot increase its
at-most-12-bit significand. F32 provides 24 bits, and the maximum result is below
`2^54`, far below its finite upper limit. Thus every normal product is representable.
For a **subnormal** result, F32's quantum is `2^-149`. Since `k>=-142`, the product
is an integer multiple of that quantum. Its subnormal magnitude makes the integer
less than `2^23`, so it fits the subnormal fraction exactly. There is no value in
the gap below the smallest nonzero F32 value. The smallest product, attained at
`v=2^-9,S=2^-133`, is `2^-142=128*2^-149`, not zero. The greatest is attained at
`v=448,S=2^45`. Both signs follow by symmetry. Correct rounding therefore leaves
every nonzero product unchanged, including all subnormal cases.

For **signed zero**, finite positive `S` has sign zero. The selected IEEE multiply
sign rule is `sign(v) XOR sign(S)=sign(v)`: both zero signs survive the multiply.
The rational checker forgets zero signs, so its separate two-case sign truth table
is a symbolic check of this rule, not a hardware bit-parity experiment. The later
integer Q8 zero has no sign, as before.

The two E4M3FN NaN encodings (`e=15,f=7`, either sign) are excluded, not assigned a
finite product. E4M3FN has no infinity encoding. BF16 nonfinite values (`a=255`),
either zero and negative scales fail the unchanged positive-finite premise;
exact promotion does not validate or repair them. No payload was inspected.

## 3. What tightens, and what does not

For these projection weights the unchanged formula simplifies mathematically to
`Wq=I(256*v*S)`, without a product-rounding error term. **Integer Q8 rounding is
not removed.** For example, finite FP8 `v=1` or `3` with BF16 `S=2^-9` gives Q8
ties `1/2` or `3/2`, rounding respectively to `0` and `2`; negative ties are
symmetric. The maximum is attained without a tie:

```text
|Wq| <= 256*448*2^45 = 7*2^59 < 2^63-1.
```

The negative bound is also strictly above `-2^63`. For a supplied row of length
`d`, this yields the conditional exact error inequality
`sum_j (Wq[j]-256*v[j]*S[j])^2 <= d/4`. It can remove a generic F32-multiplication
margin from future source-derived row certificates, **not** their Q8 error,
actual row/spectral norms, client casts, input quantization or linear partial-sum
requirements. It establishes neither honest no-wrap nor the source premises.
The head remains governed by its separate BF16-table rule; it is not an FP8 block
projection. Field encoding, centered decoding and all runtime gates are unchanged.

The hypotheses matter. Finite BF16 `S=2^120` times `v=448` overflows F32; finite
`S=2^47` gives an exact finite product but Q8 exceeds i64. Neither satisfies the
existing bound. Nor may arbitrary F32 scales be substituted: `S=1+2^-23` is not
exact BF16 and multiplying by finite FP8 `15/8` requires F32 rounding. With F32
`S=2^-149`, the product with FP8 `2^-9` rounds to zero. These are excluded
counterexamples, not proposed changes to the selected source or scale limit.

## 4. Reusable scalar checks

[`maverick_weight_product_bounds.py`](../../benchmarks/research/maverick_weight_product_bounds.py)
reuses the frozen totality helper's exact powers, nearest-even integer rule and
F32/BF16 rational rounding oracle. It checks precision/grid/range conditions for
all class pairs, independently regresses signed endpoint rounding, and keeps
source/whole-graph status unresolved. It accepts mathematical classes only, not
bytes, paths, tensors or checkpoint values. The normal
[tests](../../tests/test_maverick_weight_product_bounds.py) cover the format
partition, subnormal grid, tight endpoints, Q8 ties, special cases and exclusions.

From the isolated repository root, ordinary scalar execution is:

```sh
python -B -m benchmarks.research.maverick_weight_product_bounds
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest --noconftest -q tests/test_maverick_weight_product_bounds.py
```

No model or PLLM native extension, tensor payload, network request, importer or
transcendental implementation is involved. This proof does not amend D31/D32 or
certify an actual checkpoint, source integrity, serving admission or security.
