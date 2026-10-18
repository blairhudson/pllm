# Shared-computation source profiles: implementable arithmetic and decisions

Review: 17 October 2026. Governing contract:
[shared-computation-contracts.md](../methods/shared-computation-contracts.md).
Decision update: **17 October 2026**, explicit user approvals via the question tool.
This companion supplies concrete mathematical specifications and records their
approval status under [D6–D10](../methods/DECISIONS.md#d6-approved-puma-source-derived-q18-profile).
Approval does not qualify a complete implementation.
The earlier [independent audit](shared-computation.md), including its negative
evidence and historical snapshot hash, is unchanged.

**Result:** SIGMA CPU activation/softmax/norm equations, FuseFSS's integer compiler
algebra, PUMA's candidate replicated truncation, and BumbleBee's candidate
conversion/error semantics now have executable independent scalar oracles.
P-N1/F-N1 adoption is now approved only for the named source-derived profiles
below; original evaluated-build numerical identity remains unestablished.
BumbleBee retains the **final-paper target**; B-N1 mapping remains open. T-N1's
fixed-count private feedback/tie/boundary choices are approved, but unimplemented
complete-model coverage is not promoted. The pinned examples are not evaluated-build locks.
Apple-M CPU-first and common-Qwen work are already authorized by D1/D3 in
[DECISIONS.md](../methods/DECISIONS.md); no new device/model approval is requested.

## 1. Evidence identity and scope

PDF identities are the four entries in
[source-lock.v1.json](../methods/source-lock.v1.json). References below use
one-based PDF pages. New source inspections retain the previous immutable pins:

| Prefix | Repository and commit |
| --- | --- |
| P | `secretflow/spu@a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4` |
| B | `AntCPLab/OpenBumbleBee@47c5560d069543591f3b0176eb530ede28e33fc3` |
| S | `mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88` |

An immutable file URL is `https://raw.githubusercontent.com/{repository}/{commit}/{path}`.
The previous audit hashes P's `encoding.cc`, `fxp_base.cc`, `fxp_approx.cc`,
`aby3/arithmetic.cc`, config/proto and generation files, plus B's activation,
extension, configuration and generation files. Additional inspected bytes:

| Prefix/path | SHA-256 |
| --- | --- |
| S `GPU-MPC/ext/sytorch/ext/llama/api.cpp` | `a8233617b21952396e3bb4bd0c27d024b982eb8a9c5d27910cc6ba3a063c1650` |
| P `examples/python/ml/flax_gpt2/3pc.json` | `f7fe8db54192be1fff27533e7a04521ba21f85c07fb921fa570a0cb3b9027e9b` |
| P `requirements.txt` | `c68c8cabc8e471d0c139c264396808f589e94b4109eb95ced37db7e44dce2cf5` |
| P `libspu/compiler/passes/reduce_truncation.cc` | `d4a4dcafdde5a1758feed2db8ae45c06a3a646f0471e2f708a163d0f320b55fa` |
| P `libspu/compiler/passes/optimize_denominator_with_broadcast.cc` | `20df494a6a1268873fb0450d6a903201d1fc92d9d54466958c59f7db386dab7a` |
| B `libspu/core/encoding.cc` | `2067140861153771c35d6356cdc70bc254c0c97a7b54ba45d9658dbc2951013e` |
| B `libspu/core/config.cc` | `0a57daadbe2a91d17d0c602312a4ab05f9c9a78958b84918aaaf2294286fee15` |
| B `libspu/kernel/hal/fxp_base.cc` | `4c748fe95119a1934fe95034016a602fbd0d1ac788169c14be9fe7cb29e330c2` |
| B `libspu/kernel/hal/fxp_approx.cc` | `fb0b0d792eb3e3bf28aaceb9ed80a3d9b12e24ccef57a9de09d9ee48d32e7889` |
| B `libspu/mpc/cheetah/alg.cc` | `400f963150a9353a304d0c950d0db8b591b831c96a84cac85b864f18485164e0` |
| B `libspu/mpc/cheetah/conversion.cc` | `f1b42bd6b94a32282af6fa596d71b20ae6d8d7d8c458a1be7942ed80b9737d94` |
| B `libspu/mpc/cheetah/nonlinear/ring_ext_prot.h` | `9e48b35d277742983f71848cf3a4999630425915e0fc029520c1b9b76026fb03` |
| B `libspu/mpc/cheetah/nonlinear/truncate_prot.h` | `474a9f37f1143cbb05caad513191c90f99f33fd0f1e05bb4fd57dfa9900de805` |
| B `libspu/mpc/cheetah/nonlinear/truncate_prot.cc` | `b1fb05ec3dda9fcadff77b7e848e4059f01b8d702e1039493a2f5078ef3bcc5f` |
| B `libspu/mpc/cheetah/arith/cheetah_dot.cc` | `4d660d5c2e376035b9abfaebace05b9575053fc8c11ac75e6fb2704de5e0c12f` |
| B `libspu/mpc/cheetah/arith/cheetah_mul.cc` | `6a78e7afdf62881707c56d9b711776e0ba8b49151c4945b3450b40439d0d12c1` |
| B `libspu/mpc/cheetah/arith/simd_mul_prot.cc` | `6fe0d21ebae66adda0c5cc818bb2a4029ba5f8ecad455492503ee7e6843c6ec5` |
| B `libspu/mpc/cheetah/arith/simd_mul_prot.h` | `c4f76ebd3d3cdc8034cae12553130f1d2927fed4e0fb6f7d85f6670cbc98e189` |
| B `libspu/mpc/cheetah/rlwe/modswitch_helper.cc` | `1503c231a9db527f82510fd0dda712ac02667152db573886b4587521455e538e` |
| B `libspu/mpc/cheetah/rlwe/utils.cc` | `79a3272360aefae730e88d630a14ec601c681422f49972aeda35b0bd44054032` |
| B `bazel/repositories.bzl` | `73457044e1a2e5189034e66d8aa20ee16eb7b5e5da325e1c2514ab310e458873` |
| B `bazel/patches/seal.patch` | `9bb83ab4e2bf84f9f3061b85b9236a566cbe1e98edf1377e576a05dcfb5c4875` |

B's dependency factory is SEAL v4.1.2. Static inspection of
`microsoft/SEAL/v4.1.2/native/src/seal/modulus.cpp` (SHA-256
`e8426f6a905432421b8ef4e578b956f96680dc473d7082fc1dc7f8551276d76a`)
and `native/src/seal/util/numth.cpp` (SHA-256
`acbf81bfc6dca10760409f133dfff827c90ae29e8bd4b0cc2244cc7c35d82522`)
settles its prime-selection rule. These dependency files are specification
evidence, not an adopted runtime dependency.

Sources were downloaded as inert bytes into approved temporary storage. No author
file was executed, imported, built, linked or copied into PLLM implementation.
At the scalar-review stage only the new standard-library oracle/tests were
executed; that review performed no native build, model, HE or performance run.
The later bounded [PUMA native truncation/PRSS reference](../../docs/research/puma-truncation-reference.md)
has its own recorded native evidence. The original approval update ran only the
existing scalar checks. D10 and its native RMSNorm implementation retain separate
evidence and change no historical measurements.

## 2. SIGMA: paper-defined CPU numeric profile is implementable

Use signed two's-complement ring50/Q12 for original GPT-2, effective post-linear
width 38. Encode real inputs/weights with floor. Apply exact signed floor shift
after protected products; retain reduced-ring TR versus same-ring ARS types.
GapARS inputs need a proved quarter-ring gap, not a test on a revealed value.

### 2.1 CPU activation and softmax constants

S Fig.6, p.8 and App.F define the following **integer** CPU function. For input
code `X`, set `Y=floor(X/64)`, `D=[Y>=0]`, and
`J=min(abs(Y), K−1)`. Return `D*X−T[J]`. Use `K=256` for GELU and `1024` for
SiLU. The tables are:

```text
GELU: T[i] = floor(4096 * (i/64) * (1-erf((i/64)/sqrt(2))) / 2)
SiLU: T[i] = floor(4096 * (i/64) / (1+exp(i/64)))
```

This is early TR **before** absolute value. At Q12, GELU codes `−1,63,64` map
to `−31,63,33`. Computing `abs(X)//64` would map `−1` differently. It is not a
rounding improvement. S `api.cpp:2707–2827` independently confirms this CPU
schedule and the coefficient-free tables.

S Fig.8/§5.2, pp.9–10 specify nExp and inverse completely:

```text
C = min(maximum-X, 65535)       # maximum over unmasked causal entries only
E0[i] = floor(4096*exp(-i/4096)),  i=0..255
E1[i] = floor(4096*exp(-i/16)),    i=0..255
E = floor(E0[C mod 256]*E1[C//256]/4096)
Z = floor(sum(E)/64)            # Q6 denominator; Z >= 64
R = floor(2^18/Z)               # q=6+ceil(log2(k+1)) LUT index bits
output_i = floor(E_i*R/4096)
```

The inverse table entry zero is unreachable; a native generator must assign a
fixed sentinel payload without pretending reciprocal zero is defined. Causal
masked entries have output zero and do not enter maximum or nExp. One active
element returns 4096. **Do not renormalize quantized outputs:** `[0,−4096]`
produces `[3013,1107]`, whose sum is 4120, not 4096. Equal two/three-entry rows
produce `[2048,2048]` / `[1365,1365,1365]`.

The oracle generates all 1,792 table entries from the equations. SHA-256 is over
successive unsigned 64-bit little-endian entries, independent of native packing:

| Table | SHA-256 |
| --- | --- |
| GELU | `b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b` |
| SiLU | `ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f` |
| E0 | `e094819aa436afcf59d46bfd2937b91860045f5c905387d98e271731259a963b` |
| E1 | `21a20b930ba1cd6dd57601ec54b4b6b85f206e553304b7c58da5dff2f45c5ed9` |

80- and 96-digit independent Decimal generations agree on every floored entry.
These are **paper-equation tables**, not a claim of byte identity with a particular
author compiler/libm output. A native implementation can freeze these bytes as
the paper-defined mathematical target.

### 2.2 Norm: remove the non-power-of-two ambiguity exactly

S Figs.11–12, p.18 permit real-valued `f_in` because it is used only to construct
the public rsqrt table. No approximate `log2(k)` is necessary. At `f=12`:

```text
mu = floor(floor(4096/k)*sum(X_i)/4096)
Z_i = X_i-mu
S = sum(Z_i^2)+1                # untruncated Q24 sum, then one integer guard
e = floor(log2(S))
m = floor(128*S/2^e)-128        # 0 <= m < 128
index = 64*m+e
T[index] = floor(sqrt(2^48*k*128 / ((128+m)*2^e)))
Q_i = floor(Z_i*T[index]/4096)
Y_i = floor(Q_i*gamma_i/4096)+beta_i
```

The table floor is exactly `isqrt(numerator//denominator)`. All 8,192 entries
can therefore be generated with integer arithmetic, even at hidden width 768.
The native ring implementation must also bind no-wrap/gap bounds before each
shift and the modulo-ring payload representation. This scalar oracle admits
only bounded, unwrapped values; it is not a model-range proof.

**Important source behavior:** `floor(4096/768)=5`. For 768 copies of real 1,
`mu=3840`, `Z=256`, `S=50331649`, `T=65536`, and all normalized outputs are 4096
before learned affine terms. Four copies instead normalize to zero. Replacing
this mean with an exact divide changes Fig.12. `api.cpp:3377–3416` confirms
the reciprocal-integer mean for non-power-of-two widths. Conversely,
`SumOfSquare:3269–3342` and `SlothLayerNorm:3429–3432` do not show the paper's
`+1` guard; paper-defined PLLM keeps that guard. Affine bias also needs its
recorded scale; this paper profile uses Q12 beta **after** the second shift.
No author-source shortcut overrides these equations.

RMSNorm removes centering and retains the table width factor and zero guard.
RoPE remains a separate Llama2 profile; these GPT-2 module specifications do not
settle its coefficient bytes or authorize common-Qwen numerical substitutions.

## 3. PUMA: exact candidate truncation and scalar schedules

### 3.1 TruncAPr is stochastic, with a derivable exact law

P `aby3/arithmetic.cc:750–941` chooses a public rotating pivot and relabels three
online roles `A,B,H`. `H` has the two pairwise PRSS pieces of uniform ring mask
`R`, but does **not** receive the opening `C`. `A,B` open the masked two-share
sum to one another. `H` distributes additive shares of `MSB(R)` and the middle
bits of `R`; fresh PRSS output shares restore replicated triples. This is not a
new trusted dealer. The outer replicated orientation remains `(x_i,x_(i+1))`.

Let `N=2^ell`, `U=2^b`, `1<=b<=ell−2`, and
`−N/4 <= X < N/4`. With `C=(X+N/4+R) mod N`, the reconstructed source result is

```text
Y = floor((C mod (N/2))/U) - floor((R mod (N/2))/U)
    + (MSB(C) xor MSB(R))*2^(ell-1-b) - 2^(ell-2-b)
  = floor(X/U) + [(X mod U)+(R mod U) >= U].
```

Thus `Pr[Y=floor(X/U)+1]=(X mod U)/U`; otherwise it is floor. Exact multiples
are exact. There is no additional large wrap error **inside the stated gap**.
For `(ell,b,X)=(8,3,−9)`, 32 of 256 masks return −2 and 224 return −1. This
derivation closes the distribution question for the pinned candidate; it does
not prove that every PUMA experiment used this revision or a model-wide gap.
The scalar oracle checks every mask, every admitted X and every valid shift for
ell=4 through 8 against the independent low-carry identity.

### 3.2 Constants and reciprocal/rsqrt order are now explicit

P's encoder scales then casts toward zero (`encoding.cc:63–98`), unlike SIGMA
floor. Nonfinite/clamped inputs are an author-example behavior, not a requirement
to accept invalid PLLM tensors. Use finite, range-admitted inputs for the proposed
PLLM profile. Every secret rescale below uses independent fresh `TruncAPr`.
Public scalar constants use the pinned literal's binary32 value where the HAL
writes an `F` suffix, then toward-zero Q18 encoding.

For positive Q18 integer X, put `j=floor(log2(X))`, `F=2^(35−j)`,
`mul(a,b)=TruncAPr(a*b,18)`, and `ONE=262144`. P
`fxp_base.cc:157–194` gives the **reciprocal**, not the different direct-division
schedule:

```text
C=mul(X,F)
W=mul(763940-2*C,F)
E=ONE-mul(X,W)
R=mul(W,ONE+E)
E=mul(E,E)
result=mul(R,ONE+E)
```

Positive input `0<X<2^36` is needed for this bit-reversal normalization; every
raw product still needs its quarter-ring bound. Do not move F multiplication
past refinement to copy BumbleBee's later HAL. The broadcast-denominator
optimization is explicitly enabled in P's Llama example (`flax_llama7b.py:44–48`).

For rsqrt, P `fxp_approx.cc:462–570` and `fxp_base.cc:28–62` specify:

```text
J=bit_length(X); U=mul(X,2^(35-J))             # 0<X<2^35; U approx Q18*[.25,.5)
U2=mul(U,U); U3=mul(U2,U); U4=mul(U3,U)       # NOT U4=mul(U2,U2)
P=TruncAPr(-4057974*U +10085068*U2 -13072088*U3 +6823457*U4,18)+1086023
COMP=(724 if J even else 512)*2^(17-floor(J/2))
result=mul(P,COMP)
```

`724=floor(512*sqrt(2))`; it is an integer coefficient, not Q18 encoding of
sqrt(2). The coefficient for `38.4714796F` is **10085068**, whereas converting
the exact printed decimal directly yields 10085067. That one-unit difference
is tested. Zero/negative rsqrt and out-of-domain normalization need rejection or
a separately justified range policy, never a clear floating fallback.

Reproducible **zero-mask test tapes** (not production randomness):

| Input Q18 | 65536 | 131072 | 262144 | 524288 | 1048576 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reciprocal output Q18 | 1048570 | 524285 | 262142 | 131070 | 65535 |
| Rsqrt output Q18 | 524249 | 370660 | 262124 | 185330 | 131062 |

### 3.3 Approved P-N1 profile: source-derived Q18

**Approved by D6:** `puma.source-derived-q18.v1`, the following exact
**paper/source-derived Q18 interpretation**. P-N1's adoption gate is closed;
implementation, graph/range proof, transport, SDK and quality gates remain open.

1. Ring64/Q18; finite binary32 inputs/weights and constants cast toward zero;
   pinned replicated multiplication and the TruncAPr law above. No truncation
   coalescing without a graph-equivalence check using this stochastic contract.
2. GELU paper Eq.(2)–(3), including `x6=Square(x3)`. Binary32/toward-zero
   coefficient tuples, ordered `(constant,x,x2,x3)` and `(constant,x,x2,x4,x6)`,
   are `(-132488,-110694,-30952,-2892)` and
   `(2235,131072,94458,-9879,473)`. Pivots are Q18
   `−1048576,−511180,786432`; intervals are `<−4`, `<−1.95`, `<=3`, `>3`.
   Each coefficient product is rescaled before addition in this approved profile;
   evaluating an accumulated Q36 polynomial once is another numeric profile.
3. Softmax: protected max; shifted input `Z=X−max(X)`; no epsilon subtraction
   (encoded `10^-6` is zero); keep iff `Z>−14*262144`; compute
   `V=ONE+TruncAPr(Z,5)`, then five Q18 protected squares; zero clipped terms
   **before** the sum; one reciprocal using §3.2, then broadcast products.
   All eagerly evaluated powers, including clipped inputs, need range bounds.
4. LayerNorm: centered mean-square variance, not unaveraged squared sum:
   `mu=mul(E(1/k),sum(X)); D=X−mu; V=mul(E(1/k),sum(mul(D,D)))+2`;
   `R=rsqrt(V); Y=mul(mul(D,R),gamma)+beta`. Here `E` is the same binary32/
   toward-zero Q18 encoder, `2=E(1e-5)`, and all operations/order are explicit.
   This corrects the printed divisor/epsilon omission; it is not literal Alg.3.
5. Llama SiLU is separately the located P2 polynomial, zero/identity tails,
   pivots `−8,−4,+4`, and `x6=mul(x2,x4)`. Q18 tuples in the same respective
   power order are `(-80413,-21489,-1453)` for `(1,x,x2)` and
   `(2229,131072,59806,-2913,71)` for `(1,x,x2,x4,x6)`. Llama RMSNorm/RoPE and
   checkpoint/frontend identity still need their own graph, before that family.

The user selected this explicit interpreted profile. The unselected alternative
was to wait for the evaluated config/IR and use its exact graph; those bytes
remain necessary only for an evaluated-build identity claim. Literal Alg.2
remains a **nonrecommended** mathematical control: equal logits `[-20,-20]` are both
masked to zero because its test is unshifted. A variance-sum implementation
normalizes `[-1,-1,1,1]` to magnitudes 1/2 instead of 1. Consequences are real
quality and arithmetic changes; neither can be concealed as implementation detail.

**Additional negative source evidence:** P's GPT-2 example has no GELU/softmax
replacement context and its `3pc.json` does not set exp iterations. Together
with `config.cc` it defaults to eight, not the paper's five. Only the inspected
Llama example explicitly selects five and shifted clipping. Requirements give
`jax[cpu]>=0.3.15`, not an evaluated dependency lock. The generic GPT-2 example
cannot certify PUMA's evaluated nonlinear graph or tie handling.

### 3.4 Approved positive-epsilon RMSNorm adaptation

D10 selects `puma.rmsnorm-epsilon1-q18.v1` for common-Qwen integration. Qwen's
`1e-6` epsilon becomes zero under D6's encoder. The approved replacement is
exactly `2^-18`, encoded as integer one. The bounded row graph is:

```text
S_i = mul(X_i, X_i)
V = mul(E(1/k), sum(S_i)) + 1
R = rsqrt(V)                  # section 3.2, once per row
Y_i = mul(mul(X_i, R), gamma_i)
```

There is no centering or bias. Public gamma uses D6's binary32/toward-zero
encoding. Every `mul` retains its own fresh source-order truncation; the sum is
exact and unwrapped. All possible stochastic results need public range bounds
before execution. An all-zero row has `V=1` and returns zero without a clear
test or data-dependent fallback. Native programs bind this name separately from
`puma.source-derived-q18.v1`; LayerNorm's epsilon two and softmax's encoded-zero
epsilon are unchanged. RoPE, whole-checkpoint range admission, transport and
ordinary SDK quality/cost evidence remain separate gates.

## 4. BumbleBee: conversion choices are concrete; final-paper identity is not

### 4.1 New resolved facts and error contracts

* **Encoding conflict:** B `encoding.cc:62–95` also casts toward zero. Final
  paper §II specifies floor. Q16 vs Q18 was not the only encoding discrepancy.
* **Extension M:** `ring_ext_prot.h:30` fixes `kHeuristicBound=2`;
  `ring_ext_prot.cc:213–245` adds `M=2^(source_width−2)` to rank 0, performs
  known-positive signed extension, then subtracts M. `conversion.cc:143–155`
  actually selects this heuristic. Its sufficient exact interval is
  `−M<=X<M`, since `X+M` then has sign bit zero. General sign extension is a
  different cost profile. For ring32 this is raw integer ±2^30; after restoring
  Q18 scale, real range is `[-4096,4096)`, not ±2^18. Test all shares at both
  admitted endpoints and reject a descriptor without a public range proof.
* **Mixed conversion order:** activation code casts ring64 to ring32 **before**
  downscaling to Q12. The ring32 interpretation of the original-scale input must
  therefore already be valid. At Q18, source input must lie in `[-8192,8192)`
  even before checking intermediate-product and extension bounds.
* **Approximate comparisons:** `alg.cc:72–93,102–139` subtracts the public encoded
  threshold in one share, locally drops `d=f−approx_less_precision` bits from
  each share, then evaluates MSB in the shorter ring. Configuration retains
  four fractional bits: d=14 at Q18 or 8 at Q12. This is **not** exact comparison
  of rounded reconstructed X. With difference shares a,b, the compared signed
  integer is `floor((X−T)/2^d)−carry_low`, away from signed-wrap boundaries.
  At exact equality, it reports less-than for `1−2^-d` of uniform splits.
  There is no deterministic half-open interval endpoint oracle without share
  inputs. The test's ring8/d4 equality gives 240 less-than results out of 256.
* **Truncation:** `truncate_prot.cc:249–292` reconstructs
  `floor(X/2^b)−carry_low+parity0`, where parity0 is the low bit of party 0's
  already shifted/corrected share. It is **not** PUMA stochastic rounding.
  For `b<=ell−2`, uniform source shares and `a=X mod 2^b`, the exact marginal
  probabilities for errors `−1,0,+1` are
  `(2^b−1−a)/2^(b+1), 1/2, (a+1)/2^(b+1)`.
  The source comment's 25/50/25 statement is not a value-conditioned law.
  Bounds `{-1,0,+1}` are valid here; independence across later gates is not
  implied. Test with actual share splits and correction shares, not ad hoc noise.
* **Lift tie:** `modswitch_helper.cc:105–115` implements
  `floor((t*x+2^(ell−1))/2^ell)` on canonical unsigned x: nearest/**ties-up**.
  With ell=4,t=257,x=8, output is 129, not ties-even 128. Down's denominator is
  odd, so exact half ties cannot occur: `2*2^ell*y=(2a+1)*t` has even left and
  odd right. The independent toy bOLE oracle exhausts x,y in ring16 and all
  masks in Z257 and obtains exactly error support `{-1,0,+1}`. This is arithmetic
  support, not a proof of the implementation's RLWE noise or circuit privacy.

### 4.2 Pinned candidate nonlinear functions

B `activation.cc:99–205` defines ring32/Q12 mixed GELU on **±3**, with zero and
identity tails; it does not implement wide Eq.(2)'s epsilon-offset tails.
It computes the approximate branch predicates at `−3,0,+3`; uses those shared
bits to form absolute value and branch masks; computes x2, x4=Square(x2),
x3=Mul(x,x2); and truncates the accumulated polynomial once. In Q12 that sum is

```text
2048*X + 27192 -155*ABS +2216*X2 -751*X3 +85*X4
```

The constant is **27192 in Q24**, not `6*4096`: it is encoded after multiplying
the binary32 c0 by 4096. The approximate predicate at zero can choose a sign for
zero, but tails/pivots generally remain share-dependent as above. Restore the
outer scale by left shift before signed extension. Replacing ±3 with ±2.7 or
forcing deterministic endpoints changes this artifact profile.

`activation.cc:61–96,213–287` also supplies a **different SiLU** from final
Appendix D: it approximates sigmoid(abs(x)) on `|x|<8` with degree-four coefficients
`(0.49352908920523014,0.3028510171823098,−0.06808619212463336,
0.006747908074344234,−0.0002472776978734074)`, uses symmetry and multiplies by x.
The quartic term uses working precision `max(f,21)`. It is not the paper's
quadratic/sixth-degree `−8,−4,+4` SiLU. Llama artifact and paper variants must
therefore remain separate.

Negative exp is `Z>=−14` through the approximate less-than complement, followed
by `1+Trunc(Z,t)` and t protected squares, then Boolean–arithmetic masking
(`activation.cc:290–318`). Author example sets t=4/Q16; final paper t=6/Q18.
No source evidence selects a final-paper evaluated combination of those files.

B's reciprocal is also not the PUMA HAL schedule. `fxp_base.cc:90–125,203–219`
normalizes with F, starts `W=E(2.9142F)−2*C`, performs two refinements of W using C,
and only **then** multiplies by F. Rsqrt uses the same polynomial coefficient
family and exponent compensation as P but B's own truncation/multiplication
errors; its polynomial routine is `fxp_base.cc:28–70`, rsqrt
`fxp_approx.cc:540–652`. Neither operation permits dropping the error-bearing
protected conversions or importing the PUMA oracle's zero-mask answers.

### 4.3 Exact author-candidate HE primes; paper discrepancy persists

B `cheetah_dot.cc:118–151` uses N=8192 and key-context prime widths
`[59,55,49,55]` for ring64. The last 55-bit prime is special; first data context
has widths `[59,55,49]`. These are nominal **218/163** bits, not paper q≈147.
SEAL's patched context behavior is in `seal.patch:20–29`. The factory scans
`p=1 mod 16384` downward from the largest integer of each width, then pops
same-width primes in reverse discovery order. Independent integer derivation:

```text
OLT key-context order:
576460752303210497, 36028797017571329, 562949952847873, 36028797018652673

bOLE example allow_high_prob_one_bit_error=true:
ciphertext widths [60,59,40], N=8192, no special prime
1152921504606830593, 576460752303210497, 1099511480321

bOLE plaintext CRT primes, descending, 47 bits each:
140737488273409, 140737488125953, 140737488044033
```

`cheetah_mul.cc:64–96,124–129,274–282` computes three plaintext primes for
full ring64 operands from `ceil((64+64+4)/47)=3`. Its ciphertext nominal width
is **159**, not the unused `kCipherModulusBits=145` constant or paper q≈148;
its plaintext product has nominal width 141 rather than three ≈45-bit primes.
The error-disabled branch instead selects `[60,32,52]` and **four** 45-bit
plaintext primes (`ceil((64+64+32)/45)=4`). It is not the paper's three-prime
profile either. Reading the named constant alone would choose the wrong context.

SIMD noise selection is pinned, too: `simd_mul_prot.h:29` uses 40 flooding bits;
`.cc:295–307` sets
`noise_bits=40+(plaintext_prime_bits−1)+ceil(log2(2*sqrt(8192)*6*3.19))`,
which is 98 for 47-bit primes, then passes **97** to the signed noise sampler.
`rlwe/utils.cc:446–515` increments that argument, draws a uniform 98-bit word
and centers it at 2^97: the exact coefficient interval is **[-2^97,2^97)**.
This closes the sampler endpoint question for the artifact; it must not be
silently equated to a paper parameter interpreted as an amplitude B'=2^98.
`sec_level_type::none` bypasses SEAL's built-in check; it proves neither security
nor insecurity. A source profile still needs independent RLWE/noise/key-graph
qualification. Exact primes alone do not admit encrypted execution.

### 4.4 B-N1: final-paper mapping remains open

**D7 selects the final-paper target:** retain Q18/t=6/±2.7 and obtain its
evaluated config/graph. The artifact alternative (Q16, cast-toward-zero, t=4,
±3 mixed GELU and the HE contexts above) is **not adopted**. B-N1 remains an open
final-paper mapping gate. The required manifest must include input/constant rounding,
wide-GELU epsilon subtraction/endpoints, per-node rings, SiLU variant, norm
variance/epsilon/order, truncation/error flags, ordered primes and modulus levels,
noise sampler/width, key switching and frontend argmax patch status. These are
numerical/security inputs, not details supplied by choosing the final-paper label.
A hybrid chosen for convenience is not approved.

The unselected artifact option also cannot settle imported Flax normalization
or patched argmax identity without its frontend lock. Preserve the negative
evidence that `E(1e-5)` is zero at Q16 and two at Q18; no Q16 zero-variance policy
or changed normalization has been approved. D9 independently selects lowest-index
protected argmax for the named transaction, not original frontend equivalence.
OLT/interleaving and conversion algebra can proceed while final-paper mapping
remains blocked.

## 5. FuseFSS: integer compiler can start without guessing float ties

F Protocol 3, p.17, Lemma 4.3 and App.E specify a complete mask-dependent
preprocessing transformation whose public shape depends only on the descriptor.
This module does not need a floating-point encoder or an evaluated model manifest.

Input: n-bit ring; complete strictly increasing boundaries
`0=alpha[0]<...<alpha[m]=2^n`; fixed-degree padded arithmetic coefficient tuples;
typed Boolean formulas/thresholds/low-bit views; fixed post-circuit Phi. An
independent masked instance receives fresh mask r and shares of r.

1. Sort `((alpha[i]+r) mod N,i)` for i<m. If first translated start is nonzero,
   split the wrapping interval at 0/N and duplicate its payload. If it is zero
   and m<N, split the **first** interval with length at least two at start+1 and
   duplicate that payload. If m=N, keep N intervals. Always
   `M=min(m+1,N)` and exactly M payloads.
2. For each threshold beta, preserve
   `[x<beta]=[xhat<(r+beta) mod N] xor [xhat<r] xor [r+beta>=N]`.
   For low-bit width b use `r mod 2^b`, `xhat mod 2^b`. Descriptor sentinels
   beta=0/N may be simplified before masking. A mask-generated zero threshold
   must **not** remove a query, shrink keys or change the public trace.
3. Payload is the selected polynomial in **x**, not in xhat. Evaluators derive
   shares of x=xhat−r and use Horner with charged Beaver products, then typed
   Boolean normalization/Phi. No binomial coefficient translation is needed.
   Signed comparisons first apply the descriptor's public sign shift.
4. At native n=64, represent sentinel N and boundaries in a wider public integer
   type; truncating N to a ring u64 creates a full-domain bug.

Example n=3, boundaries `(0,3,8)`: masks 0,2,5 yield respectively
`B=(0,1,3,8),ord=(0,0,1)`;
`B=(0,2,5,8),ord=(1,0,1)`;
`B=(0,1,5,8),ord=(1,1,0)`.
The oracle tests all 128 partitions of ring8, all masks and all x, plus threshold
sentinels and low-bit views through ring32. It tests algebra, **not FSS key
privacy, serialized lengths, communication or a protected backend**.

### 5.1 Approved F-N1 descriptor profile

**Approved by D8:** `fusefss.source-derived-ring50-q12.v1`, a named source-derived
GPT-2 profile: ring50/Q12,
nearest/ties-even **external input and weight** encoding; S's floor-defined CPU
LUT/ARS equations in §2; explicit descriptors for each gate and no changed
internal rescale order. This makes the external rounding difference explicit;
it does not prove original FuseFSS numeric identity. **F-N1's adoption gate is
closed** for this profile. Full descriptor/LUT/coefficient bytes, per-edge
widths/scales, rescale/Phi graph, secure backend, range proofs and complete graph
still must be implemented/frozen/validated. Neither primary PDF nor its versioned
arXiv record supplies the evaluated descriptor artifact; that historical gap
still blocks original table reproduction. Whole-model quality/costs remain unqualified.

## 6. Token feedback and state: precise evidenced boundary

P `flax_gpt2.py:43–54,72–77`, `flax_llama7b.py:134–144,164–167` and B
`flax_gpt2.py:119–135` implement the following fixed-count transaction:

```text
private prefix I_0; public generation count T
for j=0..T-1:
    logits_j = protected complete model(I_j, method-owned weights)
    token_j = protected argmax(logits_j[last position,:])
    I_(j+1) = protected concatenate(I_j, token_j)
reconstruct final token-ID array only to client after all T selections
```

Positions are those of the entire growing prefix on every call. The loop has no
KV input/output, no pending unconsumed token, no sampler RNG, no EOS-dependent
break and no intermediate reveal. It returns after exactly T selections even
if a selected ID is EOS; final text decoding may skip special IDs. Reusing weights
is distinct from reusing multiplication/OT/FSS consumables. Examples specify no
cross-request continuation or cancellation API. A fresh full-prefix request is
the evidenced starting state contract, not incremental serving coverage.

**Historical tie conflict:** P's unpinned JAX default intends
lowest-index argmax; B README **lines 116–135** proposes deleting that tie breaker,
making ties follow reduction operand order. This corrects the old audit's
README locator without altering its historical text. For equal maxima at indices
0/1, lowest-index gives 0; a strict-greater reducer keeps its accumulator, which
can be 1 in the opposite operand order. A fixed tree/order or original lower-index
rule must be frozen; “greedy” alone is insufficient. D9 now selects the
lowest-index rule for the named transaction; it does not establish which
frontend/reduction tree produced the source measurements.

**T-N1 approved by D9:** fixed-T full-prefix feedback, private IDs throughout,
final client-only ID reveal, and lowest-index protected argmax for all four
shared-computation methods. PUMA uses D6's named numeric profile; BumbleBee's
final-paper mapping remains open under D7. For SIGMA/FuseFSS, D9 approves **explicit completion**
with protected equality→one-hot embedding, complete final norm/head, same
client-only ID policy and fixed-T full-prefix recomputation. Charge every added
operation, including the missing embedding, final normalization and full head.
SIGMA retains private weights/two online owners/dealer; FuseFSS retains
its public-weight two-server/client/dealer setting. Streaming token disclosure,
stochastic sampling, full-logit disclosure, EOS-short-circuit and persistent KV
are not supplied by that completion and would need separate contracts. T-N1's
approval gate is closed; full protected implementation, actual multi-selection
feedback, lifecycle/range admission, SDK integration and quality remain open.

## 7. Ready next modules and acceptance vectors

| Native module to start | Frozen source target | Remaining gate |
| --- | --- | --- |
| Masked-ring CPU activation/softmax/norm scalar interpreter and gate schedules | S paper equations and table hashes, ring50/Q12, §2; D9 boundary completion | Source DPF/dealer, complete charged boundaries/graph and range proofs; transport/SDK/quality |
| Typed mask-rewrite/interval-padding compiler | F Protocol 3/§5; D8 named ring50/Q12 descriptors | Complete frozen descriptors, secure backends/Phi, fixed serialized shapes, D9 boundaries and range/SDK/quality |
| Replicated3 ring multiply, Boolean equality/embedding and nonlinear graph | D6 named Q18 profile; §3.1 TruncAPr; D9 feedback | Bounded truncation/PRSS exists; mixed-kernel scheduling, remaining operators, full graph/range/transport/SDK/quality |
| OLT/InterLeave/ParseMat algebra and approximate conversion oracle | D7 final-paper target; artifact §4 facts remain separate evidence | B-N1 evaluated-final-paper numeric/HE mapping, HE/OT backend and complete protected graph |

D6/D8 approve the named profiles, not original evaluated identity. D7 selects
the final-paper target without filling its missing manifest; D9 approves the
specified completion without implementing it. The bounded PUMA native milestone
is documented separately. All modules still require D5 ordinary SDK integration
and their own complete-model correctness, quality and cost evidence.

### 7.1 Next bounded slices after the native freeze

These are implementation contracts for the next work, **not work executed by this
documentation update**. Existing PUMA Rust inputs remain frozen.

**PUMA — one replicated Q18 multiply followed by TruncAPr(18).** Use pinned P
`aby3/arithmetic.cc:425–459` (`MulAA`): party i, owning only its two adjacent input
components and PRSS streams, computes in ring64

```text
z_i = x_i*y_i + x_i*y_(i+1) + x_(i+1)*y_i + s_i - s_(i+1).
```

Send z_i to the previous party and receive z_(i+1) from the next to restore
`(z_i,z_(i+1))`; then apply the existing stochastic TruncAPr with b=18. Admit only
public range envelopes proving the reconstructed raw product lies in
`[-2^62,2^62)` (e.g. `|X|,|Y| <= 2^30-1` for bounded fixtures); never test bounds
on individual random shares or reveal an out-of-range secret. An independent
wide-integer product followed by §3.1's conditional-on-mask oracle is the result
contract, not deterministic exact rounding.

For n lanes and q=ceil(n/2), source consumption is one Both PRSS fill for MulAA,
then two fills for TruncAPr: shared pair cursor +3q, public pivot cursor +1, and
only H's private cursor +n. AES work is 4q+1 per evaluator and 6q+n+1 for H.
Extend the typed mixed-kernel scheduler with explicit phase/purpose/cursor binds;
do not reset/reimport seeds between stages. Retain the current 1024-lane and
256-KiB local ceilings, deriving complete joint-owner/work admission before
issuance rather than assuming the existing truncation scratch reserve covers it.
Require replicated-pair consistency, all multiplication cross terms, low-residue
error frequencies, mask independence and sticky replay/cancellation/failure burn.

**FuseFSS — bounded typed Protocol-3 compiler and independent integer interpreter.**
Freeze D8 ring50/Q12 metadata, nearest-even external input/weight encoding and
§2's floor-defined CPU internals. A first concrete gate is CPU GELU:
`Y=floor(X/64)`, `D=[Y>=0]`, `J=min(abs(Y),255)`, output `D*X-T[J]`, using §2's
exact 256-entry floor table/hash. Represent it as a complete signed-to-ring50
piecewise-affine descriptor with integer slope D and Q12 intercept -T[J]; no new
polynomial fit or changed internal rescale is authorized.

Initial compiler limits: at most 512 source intervals, at most 513 after public
padding, degree at most one, and wide public sentinel N=2^50. Apply §5's exact
mask rewrite, first-eligible padding, fixed public query shape and original-x
Horner rule. Bound descriptors, payloads and post-circuit work before allocation;
every protected Horner/Boolean/conversion operation must remain explicit/charged.
Check exhaustive small-ring partitions/masks/sentinels and ring50 wrap/rounding
vectors against the unfused integer descriptor interpreter. This slice supplies
compiler/interpreter conformance, not an online FSS backend or complete model;
secure lookup/predicate/Phi execution and D9's charged boundaries remain gates.

Reproduce the bounded arithmetic checks from repository root:

```sh
python3 -B -m unittest discover -s paper/reproduction-contracts -p 'test_shared_profile_oracle.py' -v
python3 -B paper/reproduction-contracts/shared_profile_oracle.py
```

The second command emits deterministic vectors, table hashes, exact candidate
primes and diagnostic zero-mask schedules. Neither command starts a model or
uses author code. Tests cover small-ring exhaustive laws and explicit conflicting
source cases, not a cryptographic security claim or source performance result.
**Validation: 15 tests passed.** This includes 247,552 PUMA mask/input/shift
combinations, 8,192 FuseFSS partition/mask/input combinations, 65,792 toy bOLE
input/mask combinations, and integer floor inequalities for 16,384 norm entries.
One initial test assertion used the wrong decimal coefficient as a binary32
counterexample; it was corrected from `−49.86605845` to `38.4714796` after exact
rational checking. No source behavior or acceptance threshold was changed.

**Remaining queue:** B-N1's evaluated-final-paper numeric/HE manifest and mapping;
full PUMA/FuseFSS graph/descriptors and range proofs; protected operators/backends
and charged D9 embedding/final-norm/head/feedback; mixed-kernel PRSS scheduling,
transport/lifecycle, D5 SDK and original/common-model quality/cost qualification.
P-N1/F-N1/T-N1 approval gates are closed only within D6/D8/D9's explicit scope.
Evaluated-build identity remains unestablished, and no complete-method or measured
performance status is created by these decisions.
