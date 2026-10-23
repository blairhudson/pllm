# Maverick: source-conditional honest no-wrap inequalities

**D31 mathematical analysis only; selected specification at `41f555d4`.** No
checkpoint values have been acquired or inspected. This note gives sufficient
inequalities, not an achieved no-wrap proof, source certificate, security result,
or serving/implementation authorization. The input domain remains every eight-ID
vector in `[0,151936)^8`, positions 0–7; no corpus restriction or numeric repair.

The controlling rules are [§3.2, exact scalar/matrix arithmetic](maverick-numeric-source-recovery.md#32-exact-scalar-and-matrix-rules),
[§3.3, full local schedule](maverick-numeric-source-recovery.md#33-full-local-schedule),
and [§4, integer equivalence](maverick-numeric-source-recovery.md#integer-equivalence-and-finite-local-arithmetic).
The [conditional totality theorem](maverick-canonical-reply-totality.md#5-conditional-totality-of-the-complete-client-numeric-graph)
is separate. Let `H=1,006,632,960`, `p=2H+1`, and `D=65,536`.

## 1. Premises and exact integer row tests

**Finiteness premise F:** the selected source and graph have an independent proof
of finite local F32/BF16 arithmetic, positive denominators and signed-i64-safe
weight/input quantization. The earlier totality theorem supplies sufficient
source conditions for F, but their truth for the pinned checkpoint is unproved.
Finite source tensors alone do not establish F. Every rounding-error inequality
below is conditional on F; no nonfinite result is silently continued or repaired.

Every matrix `W` below means the **exact signed integer Wq** of §3.2, including
FP8 decode, multiplied block scale, F32 rounding and nearest-even fixed-Q8. The
head uses the specified tied BF16 source table. Norms of stock FP8 weights, an
unrounded real matrix or another checkpoint cannot replace these quantities.

For an honest BF16 input vector `x` of length `d`, put `z=I(256*x)` coordinatewise.
For row `w_i`, define the public exact integers

```text
A_i = sum_j w_i[j]^2,       L_i = sum_j abs(w_i[j]).
```

If a public rational `Q` bounds `norm2(z)`, the sufficient row test is

```text
A_i * Q^2 <= H^2.                                      (R1)
```

It follows by Cauchy–Schwarz that the mathematical integer dot is in `[-H,H]`.
This is not a bound on an unreduced host accumulator. The field evaluator still
reduces as selected. Alternatively, if `norm2(x)<=Z`, exact integer rounding gives
`z=256*x+delta`, `abs(delta_j)<=1/2`, hence

```text
256*sqrt(A_i)*Z + L_i/2 <= H.                           (R2)
```

No irrational calculation is necessary: require `T_i=H-L_i/2>=0` and check
`65,536*A_i*Z^2<=T_i^2`. A component certificate
`sum_j abs(w_i[j])*B_j<=H`, where `abs(z_j)<=B_j`, remains another valid option.
These are sufficient alternatives, not necessary tests.

To propagate useful output bounds, record an integer `0<=K_i<=H` bounding that
row's absolute integer dot, for example with `A_i*Q^2<=K_i^2`. Then

```text
B_i = R16(exact_rational(K_i,D))
```

bounds the honest decoded output. This uses the selected **direct** rational-to-
BF16 conversion, not an extra F32 cast. Before the row test is proved, a canonical
reply bound cannot be treated as evidence of the intended integer dot.

## 2. RMSNorm L2 bound that includes its rounded variance

Use the exact rational constants

```text
u=2^-24, eta=2^-150,       v=2^-8, xi=2^-134,
b=1+u, r=1+v,            k=b*r, e=r*eta+xi.
```

For finite rounded results, IEEE nearest-even with gradual underflow satisfies
`abs(R32(y)-y)<=u*abs(y)+eta` and
`abs(R16(y)-y)<=v*abs(y)+xi`. Thus a selected `R16(R32(y))` is bounded in magnitude
by `k*abs(y)+e`. The additive terms retain underflow cases; they are not dropped
as negligible. Signed-zero bit parity is not inferred from magnitude inequalities.

Let `S=sum_j x_j^2` for a BF16 norm input, and `a=1-u`. In the **specified order**,
rounded nonnegative squares followed by the `d` rounded additions give

```text
z_d >= a^(d+1)*S - 2*d*eta.
R32(z_d/d) >= a^(d+2)*S/d - 3*eta.
A=R32(R32(z_d/d)+eps) >= a^(d+3)*S/d + a*eps - 4*eta,
eps=R32(1/1,000,000).
```

For the first line, each nonnegative square has reduction weight at least `a^d`;
the product-rounding losses and sum-rounding losses are each at most `d*eta`.
The next lines include their own division and addition rounding. In particular
this does **not** replace the computed variance by `S/d`.

By Bernoulli's inequality, with `c_d=1-(d+3)*u`,

```text
A >= c_d*S/d + eps0,       eps0=(1-u)*eps-4*eta > 0.
```

For `d=2560` and `128`, `c_d>0`; the selected `2^-20<=eps<=2^-19` also makes
`eps0>0`. Consequently
`S/A<=d/c_d`, including `S=0` by the positive epsilon. Rational choices
`t_2560=51` and `t_128=12` satisfy `t_d^2*c_d>=d` and also `t_d>=sqrt(d)`.

Suppose `norm2(x)<=X` and `Gamma=max_j abs(gamma_j)` for this actual learned
norm. The correctly rounded reciprocal square root obeys

```text
R32(1/sqrt(A))*sqrt(S) <= (1+u)*t_d + eta*X.
```

Following both selected cast stages, **including cast-before-gamma**, yields

```text
norm2(Norm(x,gamma,d)) <= N_d(Gamma,X),
N_d(Gamma,X) = k^2*Gamma*((1+u)*t_d + eta*X)
               + (k*Gamma+1)*e*t_d.                   (N)
```

Indeed the first double rounding bounds each coordinate by `k*beta*abs(x_j)+e`;
gamma multiplication and its second double rounding bound it by
`k^2*Gamma*beta*abs(x_j)+(k*Gamma+1)*e`. Apply the vector triangle inequality.
No ideal-real RMSNorm normalization identity or fused gamma operation is used.
For a delegated norm output, the resulting rational quantized-input norm bound is

```text
Q_norm = 256*N_d(Gamma,X) + t_d/2.                     (Q)
```

The last term is the L2 norm of the **integer** rounding-error bound. It is not an
activation F32 cast. F independently establishes i64 representability.

## 3. Source bounds for honest projection outputs

After proving the relevant row tests, every nonzero exact integer dot divided by
`D` has magnitude at least `2^-16`. Its BF16 rounding cannot be subnormal, so the
output's relative error bound needs no `xi` term. If a nonnegative rational `T_W` satisfies
`norm2(W*z)<=T_W*norm2(z)` for every real vector `z`, then

```text
norm2(Lin_W(x)) <= r*T_W*Q/D.                          (V)
```

Such a source premise can be stated exactly as
`T_W^2*I-W^T*W` positive semidefinite. A rational Frobenius bound with
`T_W^2>=sum_ij W[i,j]^2` is sufficient but may be much looser. No approximate
eigensolver output, sampled cancellation, or unrounded-weight spectral estimate
is a certificate. This review constructs or inspects none of those matrices.

## 4. Attention-to-o without assuming rounded probabilities sum to one

First prove the q/k/v integer row tests using their common input-layer norm
bound (Q). Premise F covers Q/K norms, the selected split-half RoPE, all score
casts (including BF16 **before** alpha), and finite score-minus-maximum. This
section needs no claim that rounded RoPE is exactly orthogonal.

For a causal row let `n=t+1<=8`. Its correctly rounded exponentials are
nonnegative, at most one, and include an exact one. Let their exact real sum be
`Eexp>=1` and their increasing-order F32 sum be `z`. The rounded-sum lower bound is

```text
z >= (1-n*u)*Eexp - n*eta >= c_n*Eexp,
c_n=1-n*(u+eta)>0.
```

Thus the **actual**, separately rounded probabilities satisfy

```text
sum_j prob_j <= M_n = k/c_n + n*e.                    (P)
```

This includes the F32 division and BF16 probability cast, allowing exponential
underflow to zero. It does not assume exact normalization or add an unselected
renormalization step.

Let `V_c` bound an honest V coordinate uniformly over positions, obtained from
the v-row `K_i` above. Product rounding, all ordered D32 sum roundings and the
final **single BF16 cast** give the context-coordinate bound

```text
C_n(V_c) = r*(b^(n+1)*V_c*M_n + 2*n*eta*b^n) + xi.    (C)
```

For every `1<=n<=8`, exact rational comparison gives `M_n<101/100` and
`r*b^(n+1)*M_n<101/100`. Thus this is a useful source-V bound rather than the
speculative `8*L` bound. The additive `r*2*n*eta*b^n+xi` is still retained;
these are analytic inequalities, not observed attention statistics.

For example, before the final cast, triangle inequalities give
`b^n*sum_j abs(R32(prob_j*V_jc)) + eta*sum_{j=0}^{n-1} b^j`;
the displayed bound follows using (P). All constants are positive, so `C_8`
covers every position. GQA repeats each KV-coordinate **bound** four times;
it does not make the four query heads' probabilities or values identical. A
rational `Z_o` satisfying

```text
Z_o^2 >= 4*sum_{c=0}^{1023} C_8(V_c)^2
```

bounds the 4096-coordinate context norm. Use `Q_o=256*Z_o+32` in (R1) for every
o row, or `Z_o` in (R2). The 32 is exactly `sqrt(4096)/2`.

## 5. MLP down: preserve the common honest gate/up input

Let `z=I(256*Norm(h1,post_gamma,2560))`, with `norm2(z)<=Q` from (Q).
Write `G` and `U` for the selected `9728 x 2560` integer gate/up matrices.
Equality of their clear mathematical inputs does not share one-use masks or
corrections; protocol material remains separate.
**First prove all their row no-wrap tests.** Their honest decoded outputs are

```text
g_j=R16((G*z)_j/D),       up_j=R16((U*z)_j/D).
```

Arbitrary canonical server replies cannot replace these relations. In
particular, the earlier `15,360^2` envelope is also a valid coarse bound on honest
ff, but it neither identifies honestly reachable values nor proves any down-row
no-wrap test. Its endpoint is not substituted for an actual honest activation.

### 5.1 Global product-norm option

For each actual BF16 gate coordinate, correctly rounded exact-real SiLU followed
by BF16 rounding satisfies `abs(sg_j)<=abs(g_j)`: exact SiLU has that bound and
both rounding operations fix the representable endpoints `+/-abs(g_j)`. This
does not evaluate a naive F32 `exp(-g)`.

Let `Ginf=max_j B_gate,j`, `Uinf=max_j B_up,j`, and use (V) to obtain
`G2=r*T_G*Q/D`, `U2=r*T_U*Q/D`. Then

```text
Z_ff = k*min(Ginf*U2, Uinf*G2) + 99*e                (F)
```

bounds the norm of the selected `ff=R16(R32(sg*up))`, since
`sqrt(9728)<99`. The factor `k` and additive error account for **both** product
roundings. Apply (R2) to each down row using `Z_ff`; or use
`Q_ff=256*Z_ff+99/2` with (R1). Neither option substitutes independent canonical
gate/up endpoints for the common honest input.

### 5.2 Row-weighted joint quadratic option

A stronger row-specific certificate can retain gate/up correlation. For down
row `d_i`, put `L_i=sum_j abs(d_i[j])`. Choose positive **public rational proof
weights** `tau_j`; these are inequality witnesses, not quantization scales.
Define the positive-semidefinite matrix

```text
K_i(tau) = (1/2)*sum_j abs(d_i[j]) *
           (tau_j*G_j^T*G_j + tau_j^-1*U_j^T*U_j).
```

If a rational `kappa_i` satisfies `kappa_i*I-K_i(tau)` positive semidefinite,
Young's inequality gives, for the **same** input vector z,

```text
sum_j abs(d_i[j])*abs((G*z)_j)*abs((U*z)_j)
    <= z^T*K_i(tau)*z <= kappa_i*Q^2.
```

An exact rational absolute-row-sum bound on the symmetric `K_i` supplies one
sufficient `kappa_i`; its trace supplies another, potentially loose, bound.
Alternatively Cauchy gives a valid bound with
`kappa_i=min(T_(D_i*G)*T_U, T_G*T_(D_i*U))`, where
`D_i=diag(abs(d_i))` and each T is a sound source operator-norm upper bound.

The direct projection BF16 casts contribute `r^2`, SiLU cannot increase each
gate magnitude, the product contributes `k,e`, and exact nearest-integer Q8
contributes at most `1/2` per coordinate. Consequently the sufficient down test is

```text
(256*k*r^2/D^2)*kappa_i*Q^2 + (256*e+1/2)*L_i <= H.    (DWN)
```

All factors and rounding margins are explicit rational quantities. To see their
units, `G*z` and `U*z` are integer dots; **each** divides by 65,536 before BF16.
The coefficient `256/D^2=2^-24` accounts for both divisions and the later Q8
encoding. Omitting either division, the product roundings or the final integer
rounding would certify a different expression. Signed cancellation was not
assumed. These are source-dependent sufficient tests, not claims that any actual
row passes. They can improve on a row-L1 test based on uniform `L^2` endpoints.

## 6. Exact remaining inductive certificate

The following conditional induction covers the graph; none of its source
premises have been discharged for the pinned checkpoint in this review:

1. Start with per-coordinate bounds on **all** source embedding rows and a
   rational norm bound `X_l` for each residual. Obtain input-norm Q from (N,Q).
   Prove all q/k/v row tests, retaining honest V component bounds.
2. With F, apply (P,C) to get `Z_o,Q_o`; prove every o row. Propagate each residual
   bound using the exact positive bound operation `R16(R32(R_l+B_o))`. Any
   nonfinite bound is inconclusive, not permission to continue through infinity;
   an independently proved finite totality bound may instead bound that residual.
3. Obtain the post-attention norm Q from its actual gamma and residual norm
   bound. Prove gate/up rows before using their honest joint projection relations.
   For every down row prove (F)+(R1/R2), (DWN), or another sound public inequality.
   Choose an integer `K_down<=H` above the proved dot bound, decode it directly to
   BF16, and propagate the second residual addition in the same stated order.
4. Repeat for all 36 layers, uniformly over the eight positions. Normalize final
   position 7 with its own learned gamma; prove all 151,936 head row tests using
   the selected BF16-derived integer head. No token selection or KV step is added.

Public rational norm bounds can be checked by squaring; PSD/source-norm premises
must themselves be sound, not inferred from floating estimates. Source-derived
component intervals or tighter relational invariants may replace conservative
bounds. If an inequality fails, that is **not** an observed model counterexample:
the remaining obligation is a stronger public invariant covering the same full
token domain and every intervening rounded operation. A sampled quality screen
or cancellation observed on prompts cannot discharge it.

**Conditional conclusion:** F plus these source inequalities imply honest integer
no-wrap for every selected linear claim by induction. Actual checkpoint validity,
spectral/row quantities, inequalities and any tighter inductive invariants remain
unproved. The conditional totality theorem alone implies none of these row tests.
This note changes no arithmetic, accepted input domain, protocol, roles, storage,
batching, security claim, profile or execution authorization. No private-value
range abort, clipping, tuning or fallback is supplied as a repair.
