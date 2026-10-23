# Maverick: canonical-reply totality at MLP requantization

**Documentation/math only; reviewed at `41f555d4` (D31), 23 October 2026.**
This note analyzes the selected `M-N/Qwen3-4B-8prompt-candidate-1` map; it
does not amend it. [D31](../../paper/methods/DECISIONS.md#d31-maverick-mathematical-map-and-quality-screen-choices)
approves mathematical choices and screening criteria, not implementation,
checkpoint acquisition, heavy execution, or achieved numerical/security proofs.

**Finding:** the large MLP input is already covered by the specified modular
encoding. It is not an undefined/rejecting transition. This resolves that proposed
counterexample, **not** the whole graph's all-canonical-reply totality obligation
or its separate honest no-wrap obligation.

## 1. Controlling clauses

References below are to the selected [numerical specification](maverick-numeric-source-recovery.md)
as committed at `41f555d4`, not to an assumed implementation:

| Clause | Relevant rule |
| --- | --- |
| §1, “What the paper actually fixes” | BabyBear modulus `p=2,013,265,921`; centered field output divided by 65,536. |
| [§3.2, scalar rules](maverick-numeric-source-recovery.md#32-exact-scalar-and-matrix-rules), first paragraph | Exact nearest-even integer `I`; finite local intermediates; `Wq` and `xq` fit signed i64, proved at admission, not repaired online. |
| §3.2, `Lin_W` equations | `xq[j]=I(256*exact_value(x[j]))`, **`xF[j]=xq[j] mod p`**, and `Mq[i,j]=Wq[i,j] mod p`; exact field dot followed by direct centered-rational-to-BF16 conversion. No requirement `abs(xq)<=H` is stated. |
| [§3.3, step 6](maverick-numeric-source-recovery.md#33-full-local-schedule) | `sg=R16(correctly_rounded_F32(exact_real_silu(g)))`, evaluated without an overflowing exponential intermediate; `ff=mul16(sg,u)`. Step 7 feeds `ff` to `Lin_down`. |
| [§4, integer equivalence](maverick-numeric-source-recovery.md#integer-equivalence-and-finite-local-arithmetic) | Honest integer row dots must satisfy `abs(sum Wq*xq)<=H`; all admitted eight-ID inputs, not sampled cases. |
| [§4, speculative totality](maverick-numeric-source-recovery.md#total-speculative-arithmetic-a-concrete-public-certificate-obligation) | Arbitrary canonical replies must not induce private-dependent numeric failure; **“On speculative paths, an integer dot may wrap”**. Source-dependent finite/i64 certificates remain required before serving. |

Here `H=(p-1)/2=1,006,632,960`. The centered bound is an **output decode**
bound, not a signed input-encoding restriction. Applying it to `xq` would change
the selected map. The selected [acceptance dependencies](../plans/maverick-numeric-completion.md#hard-acceptance-and-security-dependencies)
likewise distinguish the large input from an honest no-wrap certificate.

## 2. Exact endpoint calculation

Every canonical demasked residue centers into `[-H,H]`. Since
`H/65,536=L=15,360` is exactly BF16-representable, monotonic nearest-even rounding
also keeps every decoded linear reply within `[-L,L]`.

Consider canonical gate and up replies decoding to `g=u=L`. The endpoint is
actually attained by the specified SiLU rounding, not merely a loose upper bound.
For exact SiLU at positive `L`, its distance below `L` is

```text
delta = L/(1+exp(L)) < 2/L = 1/7,680 < 1/2,048.
```

The first inequality follows from `exp(L)>L^2/2`. The F32 spacing at `L` is
`2^-10`; its half-ULP is `1/2,048`. Thus correctly rounded F32 SiLU is exactly
`L`, and the following BF16 cast retains it. Both `L=15*2^10` and
`L^2=225*2^20` are exactly BF16/F32-representable. Consequently:

```text
sg       = 15,360
ff       = mul16(sg,u) = 235,929,600
xq       = I(256*ff)   = 60,397,977,600
         = 30*(p-1)    = 29*p + (p-30)
xF       = xq mod p   = p-30 = 2,013,265,891
center(xF)             = -30
```

The quantized integer is below `2^63`; its canonical residue is below `p` and
fits the specified u32 wire encoding. There is no i64 overflow, undefined
conversion, or specified online rejection here. Reducing it is the **already
selected** rule, not a proposed overflow repair.

This covers more than the positive endpoint: for arbitrary decoded
`g,u in [-L,L]`, `abs(silu(g))<=L`. Monotonic F32/BF16 rounding fixes the
representable endpoints, so `abs(ff)<=L^2` and
`abs(I(256*ff))<=60,397,977,600<2^63`. The MLP-to-down input conversion is total
throughout this canonical-reply range.

The pair `g=u=L` is a permissible continuation for this universal arithmetic
obligation. Additive demasking translates the field bijectively for any fixed
correction. This does **not** assert that an adversary knows that correction or
can make the replies verify, nor that honest execution of the pinned checkpoint
reaches these values.

## 3. What wraps, and what has not been disproved

For illustration only, a scalar row with `Wq=1` would decode this input's field
product to `R16(-30/65,536)`, not the positive unwrapped value
`xq/65,536=921,600` (the real fixed-Q8 weight is `1/256`). This is a counterexample
to the implication **“finite, i64-safe
quantization implies honest no-wrap”**. It is not a counterexample involving an
actual pinned-checkpoint row, whose coefficients have not been inspected here.
An input exceeding `H` alone does not prove a particular row dot exceeds `H`.

On dishonest speculative paths, modular wrap is expressly allowed. On honest
paths, the separate integer-dot bound must prove the decoded field result equals
the intended unwrapped result. The verifier's required raw-field equations and
whole-request release barrier must still protect acceptance; BF16 collisions
cannot excuse a false field claim. Their soundness is an independent, unproved
obligation here, not a consequence of this modular arithmetic calculation.

Nor must a host multiply the unreduced signed `Wq*xq` in i64. §3.2 specifies
reduced operands and permits reduction after each term:

```text
(p-1)^2+(p-1) = 4,053,239,666,646,712,320 < 2^63.
```

That is a defined field evaluation strategy already in the specification, not
permission to use an overflowing unreduced dot accumulator.

## 4. Remaining totality and adoption boundary

| Statement | Status of this review |
| --- | --- |
| Every decoded linear reply is within ±15,360 | Proved from canonical centering and direct rounding. |
| Canonical gate/up replies cannot overflow MLP-to-down i64 quantization | Proved above, including the attained positive endpoint. |
| Attention context uses the selected `8*L=122,880` bound, hence fixed-Q8 bound 31,457,280 | Conditional on the specified finite score/softmax operations; not a proof of those operations. |
| Every actual embedding/residual/norm/attention path stays finite, and every other `xq` and public `Wq` fits i64 | Unknown; actual source values and certificates unexamined. |
| All honest row dots avoid centered-field wrap over every admitted eight-ID input | Unknown; the malicious-reply example neither proves nor disproves it. |
| Full candidate totality, quality, cryptographic acceptance, or implementation admission | Not established or authorized by this note. |

The necessary numeric gate is therefore the **existing** public certificate, not
a new encoding decision: validate the selected source tensors/scales and derived
`Wq`; propagate §4's residual bounds from source embedding maximum `E`, use each
source norm maximum `G` in the stated `N(H,G)` bound, and cover every required
F32/BF16 operation, denominator and signed-i64 quantization on canonical-reply
paths. The score subtraction/softmax finiteness checks are included. Finite BF16
values alone do not imply i64-safe quantization. A tighter sound public proof is
allowed; an unchecked sufficient bound is not a completed certificate.

Separately discharge honest no-wrap for the entire declared token domain. The
1,024-case quality screen cannot replace either universal proof. A failed public
certificate blocks admission; it does not authorize a secret-dependent online
range abort, clipping, scale change, fallback, narrower input domain, or earlier
verification window. Do not implement those as a response to the large MLP
integer. D31's implementation/acquisition/heavy-execution restrictions and the
separate protocol/lifecycle/security gates remain in force.

Only exact integer identities and the displayed symbolic inequalities were
checked here. No checkpoint, tokenizer, model/native code, floating-point
experiment, build, or trained execution was used; no evidence report was changed.

## 5. Conditional totality of the complete client numeric graph

**Analytic continuation of the review; the preceding note is unchanged.** This
section supplies sufficient source conditions for the previously open finite/i64
argument. It does not assert that the selected checkpoint satisfies them. The
rules remain exactly §§3.2–3.3: each displayed F32 operation and reduction rounds
at its stated point, BF16 casts are not postponed, and functions denote the stated
correctly rounded exact-real functions. The claim is numerical totality, not
verification soundness, timing independence, resource admission, or execution
permission. Arbitrary canonical demasked replies need not be correct.

### 5.1 A sufficient public source envelope

Let `E` be the maximum absolute value of the **entire** BF16 embedding table and
`G` the maximum across **all** learned BF16 norm coefficients, including input,
post-attention, final and both shared Q/K norms. Require finite, conforming source
values and some public integers `h,g` satisfying

```text
h >= 22,  g >= 0,  h+g <= 44,
E <= R = 2^h,     G <= 2^g.
```

For public projections, additionally require every stored block scale to satisfy
`0<S<=2^45` with valid finite E4M3FN payloads. This is a sufficient way to prove
the selected source-weight quantization is finite/i64-safe: `abs(v)<=448<2^9`,
so `abs(R32(v*S))<=2^54` and `abs(I(256*Weff))<=2^62`. The BF16 head uses the
unchanged tied table, hence its integer magnitude is at most `2^(h+8)`.

These are **certificate conditions**, not newly adopted clipping thresholds or
input restrictions. Failure of this convenient envelope does not prove the map
fails; a tighter sound public proof may succeed. No payload, gamma, scale, domain
or arithmetic rule may be changed to make the inequalities hold. Actual source
conformance, tying and these bounds remain unexamined.

The selected epsilon is not adjustable. Put `eps=R32(1/1,000,000)`. Since
`2^19<1,000,000<2^20` and both reciprocal powers are exactly representable,
monotonic rounding gives

```text
2^-20 <= eps <= 2^-19.
```

We use monotonic round-to-nearest and exactly representable bounding endpoints,
not an ideal-real RMSNorm estimate or an assumption of negligible roundoff. Within
this envelope, all power-of-two bounding endpoints used in the proof are normal
and finite in both F32 and BF16.

### 5.2 Residual induction, including all 36 layers

Every attention/down linear reply has magnitude at most `L=15,360`. There are
two `add16` residual updates per layer, 72 in total. Let
`M=R+2^(h-8)`, the BF16 midpoint immediately above the power of two `R`.
It is exactly F32-representable. Since `L<2^14<=2^(h-8)`,

```text
R16(R32(R+L)) <= R16(R32(M)) = R.
```

The last equality uses ties-to-even: the power-of-two lower endpoint has even
significand. Rounding is sign-symmetric as well as monotone, so adding any value
in `[-L,L]` to a residual in `[-R,R]` stays in `[-R,R]`. Initialization is covered
by `E<=R`. Induction therefore bounds every saved residual, `h1`, final-layer
state and final-norm input across all eight positions and 36 layers. Large
`ff` values are **not** added to residuals: only the decoded `Lin_down` replies
are. This argument includes each prescribed F32 addition and BF16 cast.

### 5.3 RMSNorm: rounded reductions, positive denominators, and integer inputs

Consider a norm input bounded by `2^s`, with `s=h` for residual/final norms and
`s=14` for Q/K norms (`L<2^14`). Its dimension is exactly 2560 or 128. F32
promotion is exact; each rounded square is nonnegative and at most `2^(2s)`.
For the prescribed increasing-index reduction, induction gives

```text
0 <= z_k <= k*2^(2s),    k <= d <= 2560.
```

Every positive bounding sum is exactly F32-representable: `k` needs at most 12
significant bits. Thus the induction accounts for reduction rounding, rather
than replacing the reduction by a real sum. Its largest bound is below
`2^(2s+12)<=2^100`. No product or partial sum can overflow. The division by the
positive, exactly represented integer `d` yields
`0<=R32(z_d/d)<=2^(2s)`.

For the subsequent epsilon addition and reciprocal square root:

```text
eps <= a=R32(R32(z_d/d)+eps) <= 2^(2s+1),
2^(-s-1) <= correctly_rounded_F32(1/sqrt(a)) <= 2^10.
```

The lower bound on `a` follows from monotonic rounding at the exactly represented
positive `eps`. The upper bound is a finite power-of-two envelope. Reciprocal
square root is therefore defined, finite and strictly positive, including for an
all-zero norm input. Its displayed endpoints also bound the correctly rounded
value, not only the exact real reciprocal square root.

Consequently the exact selected cast-before-gamma order satisfies

```text
abs(R16(R32(u*b))) <= 2^(s+10),
abs(mul16(gamma, R16(R32(u*b)))) <= 2^(s+g+10).
```

Both intermediate casts and the gamma multiply are finite; none is replaced by
a fused or reordered expression. Residual/final norm outputs are at most
`2^(h+g+10)<=2^54`, so their exact fixed-Q8 integer magnitudes are at most
`2^(h+g+18)<=2^62`, strictly inside signed i64. These cover inputs to q/k/v,
gate/up and the final head. Q/K norm outputs, used locally in attention, have the
sharper bound `Nq=2^(g+24)`.

### 5.4 Positions, RoPE and stable causal attention

For the selected positions `t=0..7` and indices `i=0..63`, the precomputed
`e=R32((2*i)/128)=i/64` is exact and in `[0,1)`. The specified correctly rounded
power `1,000,000**e` is finite and in `[1,1,000,000]`; its rounded reciprocal is
in `[eps,1]`. Thus the rounded angle is in `[0,7]`. Correctly rounded F32 sine
and cosine and their BF16 casts stay in `[-1,1]`. No angle, reciprocal, power or
cast can overflow or divide by zero.

Each split-half RoPE product is bounded by `Nq`; its separately rounded addition
is bounded by `B=2*Nq=2^(g+25)`. Sign changes and reshapes introduce no arithmetic
growth. For each selected 128-term F32 score reduction, rounded products are at
most `B^2`; induction as above bounds its partial sums by `k*B^2`. Those
endpoints are exactly F32-representable for `k<=128`. Therefore

```text
abs(D32(q,k)) <= 128*B^2 = A = 2^(2g+57) <= 2^101.
```

Importantly, the **BF16 cast before scaling** is finite and bounded by `A`.
The correctly rounded `alpha=1/sqrt(128)` is in `[2^-4,1]`, so the
following F32 multiply and second BF16 cast preserve this bound. Finite rounded
scores lie in `[-A,A]`. Subtraction of their maximum, with the specified F32
rounding, lies in `[-2A,0]`, hence within `[-2^102,0]` and cannot overflow.

For each causal row, there are `n=t+1<=8` scores. Correctly rounded exponential
returns values in `[0,1]`, including a value exactly one for an argmax. Very
negative inputs may underflow to zero: the specification explicitly permits
underflow, and zero is finite. The ordered F32 sum remains between one and `n`:
each integer upper endpoint is exactly representable, all terms are nonnegative,
and adding the term one cannot round below one. Therefore division and the BF16
probability casts are defined and remain in `[0,1]`. No post-round normalization
is assumed.

V was a decoded linear reply, not a norm/RoPE output. Each context product is
bounded by `L`; every ordered F32 partial sum is bounded by `k*L`. For `k<=8`
these endpoints and `8*L=122,880` are exactly F32/BF16-representable. The final
BF16 context is bounded by `8*L`, giving fixed-Q8 magnitude at most 31,457,280.
GQA's fixed `floor(query_head/4)` mapping does not change the bound.

### 5.5 MLP, field evaluation and final output

Section 2 proves, for arbitrary canonical gate/up replies, finite correctly
rounded SiLU, finite `mul16`, and `abs(xq_down)<=60,397,977,600<2^63`. It uses the
selected **exact-real SiLU primitive**, which expressly excludes an overflowing
exponential intermediate. It does not justify implementing a naive F32
`exp(-g)` at negative `g=-15,360`, nor assume system-libm conformance.

Together the bounds cover every delegated input role:

| Roles | Exact fixed-Q8 magnitude bound |
| --- | --- |
| q/k/v, gate/up, head | `2^(h+g+18)<=2^62` |
| o | `31,457,280` |
| down | `60,397,977,600` |

Nearest-even integer rounding is monotone and fixes each integer endpoint, so
these are bounds on the rounded integers themselves. For negative inputs they
are also strictly above `-2^63`. Reducing each signed input/weight modulo `p`
is defined even outside the centered field range. Reduced multiply-adds fit
the §3.2 bound in section 3 when reduced after every term, independently of row
length or cancellation. No unreduced signed dot accumulator is assumed. Demasking
and centering return to the same bounded canonical domain, closing the induction
through all 2,017 linear claims. Finally every BF16 head reply is bounded by `L`
and has an exact finite F32 promotion for the 151,936-value output.

### 5.6 Why finiteness alone is insufficient; what this proof does not claim

Two hypothetical source examples show why the quantitative premises matter.
They are **not observations about the pinned checkpoint**:

- A finite BF16 embedding coordinate `2^64` has exact square `2^128`. Its first
  F32 RMSNorm square rounds to infinity. The selected finite-intermediate rule
  disallows that path; one cannot silently continue with a zero reciprocal norm.
- For a norm input of all ones and finite gamma `2^55`, the ordered variance is
  exactly one. Here `a<=1+2^-19`, so its rounded reciprocal square root lies in
  `[1-2^-19,1]` and its BF16 cast is exactly one. Gamma multiplication is finite
  and returns `2^55`; the next fixed-Q8 integer is `2^63`, **outside** positive
  signed-i64 range. This is an actual rejecting/undefined-as-i64 transition if
  such a source is served without the required admission proof, not a modular
  wrap problem. It violates this theorem's source envelope.

The selected map specifies finite intermediates and admission-time i64 proofs;
neither counterexample is permission for an online private-value abort or numeric
repair. Under the sufficient conditions above, no F32/BF16 nonfinite or out-of-i64
case is reached, so this proof does not need to invent behavior for such a case.
No cast/reduction nonfinite-policy ambiguity is used as a proof premise. Actual
host implementations of correctly rounded functions, rounding modes and the
specified stable SiLU remain unvalidated, not presumed conformant.

**Conditional theorem:** conforming sources satisfying §5.1 yield a finite,
defined client numeric graph with i64-safe Q8 inputs for every declared eight-ID
input and every sequence of canonical demasked replies. The source inequalities
and full payload conformance have **not** been established for the selected
checkpoint. Even if they were, this theorem imposes no honest integer-dot no-wrap
bound: speculative modular wrap remains defined, while honest equivalence needs
its entirely separate source/graph proof. The quality screen cannot replace it.
There is no new runtime candidate, profile, source acquisition, implementation
authorization, or claim of accepted adversarial output.
