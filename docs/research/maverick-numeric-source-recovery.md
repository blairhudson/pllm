# Maverick M-N: source recovery and selected mathematical map

**M-N remains blocked. M-D remains separate.** This is a
source-recovery result and D31-approved mathematical map, not an adopted execution profile or
evidence of source-faithful model execution. See the
[scoped reviewer decision](../plans/maverick-numeric-completion.md).
An independently implemented, unambiguous paper construction needs no new fidelity
permission. The user selected the exact map and screening criteria on 23 October
2026; that decision authorizes neither implementation nor heavy execution.
No runtime/compiler file, capability, security parameter, or allocation cap changes.

## 1. Result and bounded search

No immutable **Maverick-author** repository, evaluated checkpoint revision, or
numerical manifest was found in the sources inspected below. This is not a claim
that no such artifact exists. An official, immutable **Qwen** FP8 checkpoint and
Qwen implementation were found; neither establishes Maverick's evaluated choices.

### Governing paper and ancillary source

- [arXiv 2609.10264v1](https://arxiv.org/abs/2609.10264v1), cached PDF
  `papers/r24-maverick.pdf`, SHA-256
  `c9570fdeb51a26f9d707bcc7bf485f2001e7d3e15ebdf34815123935a1e8e2a0`.
- [v1 source archive](https://arxiv.org/src/2609.10264v1): 411,816 bytes, SHA-256
  `12b75f1d847ee2050c962f7d5c7d945c82e56db2df3e9f05b626ae79beb3ec6d`.
  `/e-print/2609.10264v1` redirects to those same bytes. The archive was inspected
  in memory, including TeX comments, bibliography, and `00README.json`. It contains
  TeX/style/bibliography files, figures, and TeX build metadata; no ancillary code,
  checkpoint identifier, model manifest, or numerical configuration. `00README.json`
  selects `main.tex`, pdflatex, and TeX Live 2025, not an inference implementation.
- Extracted `src/7_evaluations.tex` SHA-256:
  `541e9267e8822b28826884fe7f8e80a5a226bec25c28806e18e671ea215de014`.
  Lines **384–394** specify locally evaluated nonlinearities, dependent linear
  claims, optimistic verification at execution end, and acceptance only if every
  check accepts. Lines **430–440** give FP8 weights, nearest-integer weight
  quantization with `s=8`, centered field decoding, division by `2^16`, and BF16.
  Lines **444–467** give Qwen3-4B, eight input tokens, next-token logits, and the
  seven projection shapes. Neither comments nor other extracted text supplies
  `xq`, FP8 subtype/scales, tie rules, a checkpoint revision, or a local schedule.
- The [v1 HTML](https://arxiv.org/html/2609.10264v1) GitHub links concern arXiv
  feedback/LaTeXML infrastructure, not Maverick code. The source bibliography's
  EZKL repository and Kimi discussion are references, not implementation links.

### Author-linked discovery, not immutable implementation evidence

- [Ben Merbaum's publication page](https://www.benmerbaum.com/publications) lists
  this title and links both arXiv and [ePrint 2026/1955](https://eprint.iacr.org/2026/1955),
  but no code/model artifact. ePrint's page, PDF, and version-history endpoints
  returned HTTP 403 in this audit; their contents remain unverified.
- [Wenhao Wang's page](https://wenhao-wang0.github.io/) links the paper and his
  GitHub identity. Listing that linked user's public repositories yielded no
  Maverick implementation. [Fan Zhang's page](https://www.fanzhang.me/) and
  [publication list](https://www.fanzhang.me/publications/), plus public repositories
  in the linked `hackingdecentralized` group, supplied no matching artifact.
- [Yale Applied Cryptography Laboratory](https://yacl.cs.yale.edu/) and
  [Katerina Sotiraki's page](https://sotiraki.com/) supplied no numerical manifest.
  Exact-title/arXiv-ID searches did not recover an author repository. Private,
  unpublished, unlinked, or inaccessible artifacts are not inferred or guessed.

### What the paper actually fixes

```text
p       = 2,013,265,921
Wq      = nearest_integer(256 * Wreal)          # tie rule unspecified
r       = Wq * xq mod p                        # xq not specified
center(r) = r if r <= (p-1)/2 else r-p
y       = BF16(center(r) / 65,536)             # cast path/tie rule unspecified
```

The source says no modular overflow was **observed**, not that every admitted
input has a proved no-wrap bound. Its sentence "all arithmetic operations" being
in BabyBear must be read alongside its explicit local BF16 nonlinearities; it
does not specify field-valued RMSNorm, softmax, RoPE, or SiLU.

Two lightweight exact-rational checks show why those omissions matter:

1. E4M3FN's smallest positive subnormal is `1/512`. At scale 256 it is exactly
   `1/2`: nearest-even gives 0; nearest-away gives 1.
2. Let `r=16,842,753`, well inside the centered BabyBear range. Direct nearest-even
   BF16 conversion of the exact rational `r/65,536` gives **258**. First rounding
   `r` to float32, dividing by `65,536`, then converting to BF16 gives **256**.
   "Convert to BF16" alone therefore does not fix an executable oracle.

   Exact proof: `r/65,536 = 257 + 2^-16`. Adjacent BF16 numbers are 256
   (`0x4380`) and 258 (`0x4381`), midpoint 257. The exact value is above the
   midpoint, hence 258. F32 conversion of `r` rounds the halfway integer to
   **16,842,752** (`0x4b808000`); division is then exactly 257. BF16's tie goes
   to even significand 256. The negative example gives -258 versus -256.
   This demonstrates ambiguity, **not** which path the Maverick authors used.

## 2. Recovered Qwen artifacts and conflicts

These pins are real, official artifacts, **not recovered Maverick provenance**:

| Artifact | Immutable pin / observation |
| --- | --- |
| [Qwen/Qwen3-4B-FP8](https://huggingface.co/Qwen/Qwen3-4B-FP8/tree/96b30dc13593a244a5e59e84687309f53c375cfa) | Revision `96b30dc13593a244a5e59e84687309f53c375cfa` |
| `config.json` | SHA-256 `68fff9378aacf280608bbe1fbf594432a7e1eaeee12d738d53ed55024da1fc29` |
| `model.safetensors.index.json` | SHA-256 `be5d78efd19058e26d5363a322f148b16ade6dcd85a6f335139595df0e4eb64b`; 651 tensor names, 252 `weight_scale_inv` names, no bias names |
| Shard 1 | Advertised LFS SHA-256 `30b1bebf7b81f39f280b09c74f90feed00835a73145955a8d2fb69543b8371fb`, 4,412,140,424 bytes |
| Shard 2 | Advertised LFS SHA-256 `79f82dbc594a0237e24982800665eb0ddbe8c7bbdf3ac49c2a0005682aca15dc`, 777,912,448 bytes |
| `tokenizer.json` | Advertised LFS SHA-256 `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| [Qwen/Qwen3-4B BF16 comparison source](https://huggingface.co/Qwen/Qwen3-4B/tree/1cfa9a7208912126459214e8b04321603b3df60c) | Revision `1cfa9a7208912126459214e8b04321603b3df60c`; same advertised tokenizer bytes |
| BF16 `model.safetensors.index.json` | SHA-256 `6dc0981b8829fead746441f68f38f24c5ca4a3a66351f652c26c6df0efc43ab2`; **398** names; no `lm_head.weight`; tied head must use embedding |
| BF16 `config.json` | SHA-256 `8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`; `tie_word_embeddings=true`, BF16 default, original dimensions |
| [Transformers v4.51.0](https://github.com/huggingface/transformers/tree/0720e206c6ba28887e4d60ef60a6a089f6c1cc76) | Annotated tag dereferenced to commit `0720e206c6ba28887e4d60ef60a6a089f6c1cc76`; checkpoint config names this version |
| [Tokenizers v0.21.1](https://github.com/huggingface/tokenizers/tree/133db48d4f272070076fbc7a649c9e8aa8cd0646) | Tag resolves to commit `133db48d4f272070076fbc7a649c9e8aa8cd0646`; proposed quality-tokenization engine, within pinned Transformers' `>=0.21,<0.22` requirement |

Only metadata and implementation text were fetched. No checkpoint shard, tensor
header, tokenizer payload, or quality dataset was downloaded. Advertised LFS
hashes identify future imports; they are not a claim that payloads were verified.
Config dtype and index names alone do not prove tensor dtypes, values, or ties.

Relevant pinned Transformers source:

- [`modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/models/qwen3/modeling_qwen3.py),
  SHA-256 `704c914530530a1acb0b443add1f520404e3ac2c28c0ab7e16f80f86cfe8ccb2`.
  RMSNorm casts back **before** multiplying its learned weight (lines 71–76);
  attention normalizes Q/K per head before RoPE (217–222); MLP is
  `down(silu(gate(x))*up(x))` (93–95); head has no bias (771).
- [`finegrained_fp8.py`](https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/integrations/finegrained_fp8.py),
  SHA-256 `97fb7b1d2b8a0c372c0cf198d1c572971acd7daed1d43d7b08d87a2cafaea31f`.
  Activations use dynamic max-absolute-value scaling into `float8_e4m3fn`
  (39–57). The block matmul **multiplies** by `weight_scale_inv` despite its name
  (123–129). Loading this checkpoint with the default FP8 linear path would
  introduce dynamic activation FP8 quantization (331–353), **not** the paper's
  fixed-scale integer-field product. That path is not proposed here.
- [`modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/0720e206c6ba28887e4d60ef60a6a089f6c1cc76/src/transformers/modeling_rope_utils.py),
  SHA-256 `c28b3e88edca8fdb5497e5c36091bf753db49bd94ace33a84e9f9c61cbf66032`.
  Default inverse frequencies are `1 / theta**(2*i/head_dim)` (129–131).

Specific conflicts/limits, rather than silent substitutions:

- Official config is scaled `e4m3`, **128×128 weight blocks**, dynamic activations,
  and a BF16 model default. Paper does not identify this serialization or scales.
- The index gives scales for all `36*7` transformer projections, **not** the
  embedding or head. Config says `tie_word_embeddings=true`, but index contains
  both tensors. Their equality and serialized dtype require payload validation.
  A blanket "every weight is FP8" import cannot be inferred.
- Official graph resolves local order and bias presence for this particular
  Qwen revision, not Maverick's implementation, floating kernels, or reductions.
- Qwen's card recommends sampling for generation and warns against greedy
  thinking-mode generation. Neither that policy nor default `use_cache=true`
  defines Maverick's one-shot next-logits workload. No token is generated here.

## 3. Complete numerical candidate — proposal only, not a recovered FP8 model

Document-only name: **M-N/Qwen3-4B-8prompt-candidate-1**. This would be a
**source-derived completion**, not recovery of the evaluated author program.
Every choice below not fixed by the paper is proposed for explicit review.
It does not adopt a new numeric, security, state, transport, or storage profile.

**Classification:** no new training or synthesized checkpoint is proposed, but
the inference function is materially different from the official checkpoint's
dynamic-FP8 execution. It uses an official-Qwen source anchor with **unverified
identity to the authors' model**, followed by fixed-Q8 integer encoding (eight
fractional bits, not eight-bit coefficients), BF16 local operations and a
BF16-derived head. The paper itself requests an FP8-to-field conversion, so
different stock-FP8 behavior is **not by itself** a deviation from the paper.
Neither unavailable author code nor selection of an original-family source pin
alone proves an actual model substitution. The claimed completion still needs
explicit review of its chosen boundaries, especially the BF16-head exception;
the provenance gap remains an evidence gap, not something permission can erase.
"Source-derived" here describes the paper's architectural/equation ancestry,
**not** numerical equivalence to an identified author implementation or to stock
Qwen FP8 inference. It must not be labeled recovered Maverick numerics, native
FP8 parity, or the paper's measured model solely because shapes match.

| Boundary | Recovered fact | Proposed completion/adaptation |
| --- | --- | --- |
| Model | Paper says Qwen3-4B/FP8, no artifact revision | Select official FP8 source anchor in §2; no claim it changes or matches the authors' unknown revision |
| Weight value | Official serialized projections have block scales; named implementation multiplies scales | Decode E4M3FN, multiply scale in F32, then paper's fixed-Q8 weight rounding |
| Input value | Paper omits `xq`; official FP8 path dynamically quantizes activations | Use exact nearest-even `256*x`, **not** official dynamic FP8 |
| Decode | Paper says centered `/2^16` then BF16 | Choose direct rational-to-BF16, not F32 intermediate |
| Local arithmetic | Qwen graph/order recoverable; original kernels unpinned | Fixed reductions, cast sites and correctly rounded functions below |
| Embedding/head | Selected FP8 index has both without scale names; tying declared | Require equal BF16 copies; local embedding; Q8 head derived directly from BF16 |
| Workload | Paper says eight tokens to next logits | One eight-ID request, positions 0–7, full final vocabulary, no persistent state |

Unknown artifact values and unproved admission certificates are evidence blockers,
not unspecified rounding choices. M-B/M-C/M-S remain deliberately outside this
**numeric-map** proposal; it is not a complete delegated execution profile.

### 3.1 Import and workload

Use exactly the FP8 checkpoint and Transformers graph pins in §2. Before any
execution, validate all shard hashes, tensor names/dtypes/shapes, finiteness,
strictly positive finite F32 block scales, and config agreement. Required types:
E4M3FN projection payloads, F32 scales, BF16 embeddings, head and norm weights.
Reject any mismatch; no dtype conversion, replacement checkpoint, repair, tensor
omission, vocabulary resize, or fallback importer.

Exact FP8 tensor set: for `l=0..35`, the seven weights
`model.layers.l.self_attn.{q,k,v,o}_proj.weight` and
`model.layers.l.mlp.{gate,up,down}_proj.weight`, each with the corresponding
`.weight_scale_inv` tensor replacing `.weight` in its name. Tensor axes are
`[output,input]`; scale axes are `[output_block,input_block]`. Scale shapes are
q `32×20`, k/v `8×20`, o `20×32`, gate/up `76×20`, down `20×76`.
All dimensions divide 128; no edge-block padding rule is needed.
Per layer, `input_layernorm.weight` and `post_attention_layernorm.weight` have
2560 coordinates; `self_attn.q_norm.weight` and `self_attn.k_norm.weight` have
128 coordinates each, shared across heads, not separate parameters per head.
Global tensors are `model.embed_tokens.weight` and `lm_head.weight`, each
`151936×2560`, and `model.norm.weight`, length 2560. These are exactly 651
named tensors. Reject duplicates, missing or extra names, conflicting shard
ownership, unexpected biases or overlapping/invalid on-disk tensor extents.
Logical tensor values are taken from validated little-endian safetensors fields;
no reinterpretation of F8 bytes as signed integers is permitted.

Require the separately serialized head and embedding BF16 arrays to agree
byte-for-byte, enforcing the source's tying declaration. Missing or unequal arrays
reject this candidate rather than selecting an unspecified loader overwrite rule.
Use the BF16 embedding unchanged for client-local lookup. **Proposed exception to
the paper's generic FP8 statement:** derive head integers directly from the tied
BF16 table; do not invent head FP8 scales or requantize it into FP8.

Input: exactly one vector of **eight private token IDs**, each in `[0,151936)`.
No padding, automatic BOS/EOS, chat template, text tokenization, prompt truncation,
or early stopping. Positions are `0..7`; all eight tokens participate, including
special-token IDs. Output: every one of the **151,936 next-token logits** for
position 7, not argmax/top-k, and no token feedback. `use_cache=false`; K/V and all
scratch are request-local and never become persistent accepted state. This is not
an M-D decision.

Frozen config: 36 layers; hidden size 2560; intermediate size 9728; 32 query heads;
8 KV heads; head width 128; epsilon `1e-6`; RoPE theta `1,000,000`; no RoPE scaling,
sliding window, attention dropout, or learned biases. Matrix shapes below are
**output × input**, not transposed checkpoint conventions:

| Role | Shape | Uses per layer |
| --- | --- | --- |
| q | 4096 × 2560 | 8 |
| k, v | 1024 × 2560 each | 8 each |
| o | 2560 × 4096 | 8 |
| gate, up | 9728 × 2560 each | 8 each |
| down | 2560 × 9728 | 8 |
| head | 151936 × 2560 | Once, after final norm, position 7 only |

No projection fusion or omitted output rows. This gives `36*7*8+1 = 2,017`
semantic linear claims. Head is one full-vocabulary matvec, not eight head calls.

### 3.2 Exact scalar and matrix rules

`I(z)` is exact nearest integer, ties to even, with no clipping. `R32` and `R16`
are IEEE binary32 and BF16 round-to-nearest-ties-to-even, gradual underflow,
no flush-to-zero. Stored finite numbers denote exact binary rationals. Integer
zero maps to positive floating zero. Other local signed-zero behavior follows
IEEE arithmetic. All non-mask intermediates must remain finite; failure is not a
permission to saturate, change scale, or accept an out-of-domain request.
`Wq` and `xq` must fit signed i64; prove this at model/domain admission rather
than changing the encoding when an online value does not fit. The reference
integer/rational oracle uses arbitrary precision; field payloads remain u32.

Every elementary F32 `+,-,*,/` rounds once; no reassociation or FMA. Define
`add16(a,b)=R16(R32(a+b))`, `mul16(a,b)=R16(R32(a*b))`.
For local, nondelegated dot products:

```text
D32(a,b):
    z = +0
    for j in increasing index order:
        z = R32(z + R32(a[j]*b[j]))
    return z
```

Transcendentals and reciprocal square root below mean correctly rounded F32
values of the stated exact-real functions, not an unspecified system `libm` or
GPU approximation. This is a portable numerical oracle contract, **not** a claim
that stock Transformers kernels meet it or that it is already efficient.
In particular, this fixes approximation error to correct rounding; approving
this oracle would not authorize substituting an untested fast approximation.

For completeness, an FP8 byte has sign `s=b>>7`, exponent `e=(b>>3)&15`,
fraction `f=b&7`. For `e=0`, its value is `(-1)^s*f*2^-9`, including signed
zero. Otherwise it is `(-1)^s*(1+f/8)*2^(e-7)`, except `e=15,f=7`, which
is NaN and rejects. There are no infinity encodings; maximum finite magnitude
is 448. This is E4M3FN, **not** E4M3FNUZ or E5M2.

For projection row `i`, column `j`, decode FP8 value `v[i,j]` using E4M3FN and
its corresponding stored F32 block scale `S[floor(i/128),floor(j/128)]`:

```text
Weff[i,j] = R32(v[i,j] * S[floor(i/128),floor(j/128)])
Wq[i,j]   = I(256 * Weff[i,j])
```

Multiplication by 256 and integer rounding are evaluated exactly on the F32
rational; there is no extra F32 intermediate. No inverse-scale interpretation,
dynamic activation FP8 conversion, i8 clamp, zero-point, or integer W8A8 path.
For head only, `Weff` is the original tied BF16 value, exactly promoted to F32.
Require every computed Weff to be finite before integer rounding; finite input
payload/scale values alone do not prevent their F32 product overflowing.

Every delegated `Lin_W(x)` uses the same explicit input rule and output rule:

```text
xq[j]    = I(256 * exact_value(x[j]))
Mq[i,j]  = Wq[i,j] mod p
xF[j]    = xq[j] mod p
r[i]     = sum_j Mq[i,j]*xF[j] mod p
c[i]     = r[i] if r[i] <= 1,006,632,960 else r[i]-p
Lin_W(x)[i] = R16(exact_rational(c[i],65,536))
```

The output conversion is **direct**, not through F32. All field operations are
exact; a machine implementation must reduce often enough to avoid host-integer
overflow, independently of the signed-integer no-wrap proof. The same numeric
matrix underlies clear execution, delegation, publication, P/Q, and verification.
Canonical residues use four-byte little-endian u32 encoding, never signed casts.
For example, u64 multiply-add followed by reduction after **each** term is safe:
`(p-1)^2+(p-1) = 4,053,239,666,646,712,320 < 2^63`. This does not permit an
unreduced u64 dot-product accumulator.

### 3.3 Full local schedule

All activation arrays below are BF16 unless a rule explicitly says F32. Learned
norm weights are the imported BF16 tensors; they are not integer-field matrices.
Layer/token/head/dimension loops use increasing indices. Independent field claims
are ordered by layer, role `q,k,v,o,gate,up,down`, then token `0..7`, followed by
head. This is a semantic ordering, not approval of a batch partition.

Define `Norm(x,gamma,d)` as:

```text
u[j] = exact F32 promotion of x[j]
z = +0
for j=0..d-1: z = R32(z + R32(u[j]*u[j]))
v = R32(z/d)
a = R32(v + R32(1/1,000,000))
b = correctly_rounded_F32(1/sqrt(a))
n[j] = mul16(gamma[j], R16(R32(u[j]*b)))
```

This deliberately preserves the Qwen cast-before-gamma order. `d=2560` for
input/post-attention/final norms; `d=128` for each Q/K head norm.

Precompute once per request, for `i=0..63`, `t=0..7`:

```text
e[i] = R32((2*i)/128)
f[i] = R32(1 / correctly_rounded_F32(1,000,000**e[i]))
a[t,i] = R32(t*f[i])
cos[t,i] = R16(correctly_rounded_F32(cos(a[t,i])))
sin[t,i] = R16(correctly_rounded_F32(sin(a[t,i])))
```

Duplicate each 64-coordinate cosine/sine half to length 128. For a head vector
`z`, define `rot(z)=concat(-z[64:128],z[0:64])` and
`RoPE(z)[k]=add16(mul16(z[k],cos[k]),mul16(rot(z)[k],sin[k]))`.
No interleaved-pair rotation, fused expression, or scaled/YaRN RoPE.

Initialize `h[t]=embedding[token[t]]`. For every layer `l=0..35`:

1. Save residual `h`; compute `a[t]=Norm(h[t],input_norm_l,2560)`.
2. `q=Lin_q(a)`, `k=Lin_k(a)`, `v=Lin_v(a)` for every token. Reshape to
   `[8,32,128]`, `[8,8,128]`, `[8,8,128]`. Apply learned per-head Q/K norms,
   then RoPE to Q/K. Do not normalize or rotate V.
3. Query head `hidx` uses KV head `floor(hidx/4)`. For each query position `t`
   and key position `j<=t`, compute
   `score[j]=R16(R32(R16(D32(q[t,hidx],k[j,kv])) * alpha))`, where
   `alpha=correctly_rounded_F32(1/sqrt(128))`. Future positions are excluded
   (probability exactly positive zero), not represented by a field mask.
4. Softmax over `j=0..t`: promote scores to F32; let `m` be their maximum;
   `e[j]=correctly_rounded_F32(exp(R32(score[j]-m)))`;
   sum `e[j]` into F32 `z` in increasing `j`; set `prob[j]=R16(R32(e[j]/z))`.
   No dropout or post-round renormalization. Context coordinate is
   `R16(D32(prob[0..t],v[0..t,kv,coordinate]))`. Concatenate query heads in order.
5. `attn[t]=Lin_o(context[t])`; `h1[t]=add16(h[t],attn[t])` coordinatewise.
6. `b[t]=Norm(h1[t],post_attention_norm_l,2560)`;
   `g[t]=Lin_gate(b[t])`; `u[t]=Lin_up(b[t])`.
   `sg=R16(correctly_rounded_F32(exact_real_silu(g)))`, with
   `silu(z)=z/(1+exp(-z))`, evaluated without an overflowing exponential
   intermediate. Set `ff[t]=mul16(sg[t],u[t])` coordinatewise.
7. `d[t]=Lin_down(ff[t])`; `h[t]=add16(h1[t],d[t])` coordinatewise.

Finally normalize `h[7]` with the model's final norm, then apply the full head
`Lin_head`. Return its BF16 values exactly lifted to F32, serialized as 151,936
little-endian F32 values (607,744 bytes). No logit scaling, softmax, sampling,
token choice, or reusable KV/session state is part of this request.
All softmax sums start at positive F32 zero. Tensor reshape maps projection
coordinate `128*head+coordinate` to that head; Q/K norm gamma is indexed by
`coordinate`. SiLU at signed zero retains that zero's sign. Position/frequency
arithmetic, unary negation and all other zero signs follow the stated IEEE rules.

## 4. Proof and acceptance obligations, not evidence already obtained

### Integer equivalence and finite local arithmetic

Let `H=(p-1)/2=1,006,632,960`. For **each** semantic linear operation, its source
integer matrix and every admitted **honest** graph input must satisfy

```text
abs(sum_j Wq[i,j]*xq[j]) <= H                 for every row i
```

A sufficient independently checkable certificate is public component bounds
`abs(xq[j])<=B[j]` and exact integer row sums
`sum_j abs(Wq[i,j])*B[j] <= H`. Bounds must propagate through the complete BF16
graph, including source embeddings/norm weights, Q/K normalization, softmax,
residual additions, SiLU products, and the head. Include all rounding errors;
RMSNorm bounds for exact reals alone are insufficient. Source coefficient and
input encodings must also be proved within the declared machine integer ranges.

The declared domain is **all** eight-ID vectors above, not only a public test
corpus. A tighter sound certificate may replace conservative component bounds,
but cancellation seen in sampled dots is not a proof. If a certificate cannot
cover the declared domain, this candidate is inadmissible; no hidden narrowing,
clipping, field substitution, per-request scale adjustment, or private
range-dependent retry/abort policy is approved. Any narrower public domain or
different encoding requires a new explicit proposal.

Separately prove no local F32 overflow, invalid norm/softmax denominator, NaN,
or unexpected infinity for all admitted honest inputs, and bound speculative
local execution on arbitrary canonical but incorrect server replies. The latter
is a resource/failure obligation, not proof those replies should be accepted.
The centered-field conversion alone bounds a decoded linear value by 15,360;
it does **not** bound a residual, norm-weight product, or SiLU×up product enough
to establish the next dot's no-wrap property.

### Total speculative arithmetic: a concrete public certificate obligation

The honest no-wrap proof and malicious-reply totality proof are **different**.
The former preserves intended integer inference when prior claims are correct.
The latter must prevent a malicious canonical reply from probing a private value
through a later numeric failure, early abort, fallback or changed message shape.
Do not add an online private-value range test as a substitute for either proof.

The following conservative certificate is available without choosing numeric
thresholds after seeing private inputs. Evaluate it on authenticated public source
values before serving anything. A positive `R32`/`R16` bound operation yielding
infinity means **admission fails**; a tighter sound public certificate is allowed,
but cannot change the numeric map. Underflow to zero is defined, not failure.

Let `L=15,360`, `E=max(abs(source_embedding))`, and let each norm's `G` be the
maximum absolute source gamma. Let `A(a,b)=R16(R32(a+b))` for positive bounds.
Initialize `H_0=E`. For every layer:

```text
H_attention = A(H_l,L)
H_(l+1)    = A(H_attention,L)
```

This covers **arbitrary canonical demasked linear replies**, including dishonest
ones: each decoded reply is bounded by L independently of the secret correction.
For every norm input bound H (residual H, or L for Q/K), require the fixed `d`
F32 square/sum/mean/epsilon operations from §3.3 to remain finite when evaluated
on the positive upper bound H. Put `eps=R32(1/1,000,000)` and
`beta=correctly_rounded_F32(1/sqrt(eps))`. A conservative norm-output bound is

```text
N(H,G) = R16(R32(G * R16(R32(H*beta))))
```

It is valid because the computed variance is nonnegative and the denominator
is at least eps. Require all bound intermediates and both casts to remain finite.
Input/post-attention/final norms use their respective residual H and source G.
Q/K norms use `N(L,Gq)` and `N(L,Gk)`; rotation bounds are `A(N,N)` because
rounded sine/cosine magnitudes do not exceed one. Bound each score's 128-term
`D32` with those positive Q/K bounds, then its BF16 cast/scale/cast and the F32
`score-m` subtraction; all must remain finite.

Softmax then has `0<=e<=1`, one e exactly 1, `1<=sum(e)<=8`, and each rounded
probability in `[0,1]`. Consequently context magnitude is at most **122,880**
(`8*L`). For gate values in `[-L,L]`, exact SiLU magnitude is at most L; the
rounded SiLU stays within L. Thus the pointwise MLP input to down is bounded by
**235,929,600** (`L^2`, exactly BF16-representable). Its fixed-Q8 integer bound
is **60,397,977,600**, which fits i64 but is far above H; finiteness is not a
no-wrap certificate. Context's fixed-Q8 bound is **31,457,280**. For all norm
outputs used by q/k/v/gate/up/head, also require `I(256*N)` to fit the stated
signed integer range. Source matrix Wq values must independently fit i64.

These are mechanically checkable **sufficient** totality conditions, not checked
facts about the downloaded weights (none were downloaded). They propagate every
honest token choice and every canonical adversarial reply. No-wrap still requires
the stronger source/graph integer proof above for honest traces. On speculative
paths, an integer dot may wrap; a first wrong field claim must still be rejected
before ordinary acceptance. The verifier must check raw field relations, not
rounded BF16 outputs, so two different residues collapsing to the same BF16 value
cannot excuse a false claim.

The fixed shape/loop bounds here do not certify constant-time implementations of
transcendentals, secret-index publication accesses, allocator behavior or channels.
Interactive-view/timing/abort composition remains a separate security obligation.
Numeric totality is not a whole-protocol worst-case time bound: the unchanged
exact unbiased sparse sampler uses rejection draws without a fixed retry bound.
This packet neither introduces biased/truncated sampling nor promises bounded
entropy-retry latency; runtime admission/failure analysis must account for it.

### Numerical and ordinary response acceptance

- Independent clear oracle must compare every encoded matrix/input, integer dot,
  field residue, direct BF16 output and local intermediate, then **all logits**,
  bit-for-bit. Tests must include signed ties, FP8 subnormals/scales, the §1
  double-rounding counterexample, centered endpoints and deliberate wrap cases.
  Approximate logits or a matching argmax are not arithmetic acceptance.
- Import must recompute or soundly validate all intended-source P/Q/auxiliary
  entries and bind matrix role, position, source bytes, codes and numeric rules.
  A source digest is a binding declaration, not independent checkpoint authenticity.
- Every original response/input pair used by speculative local computation must
  be immutable before relevant challenges are sampled. Auxiliary replies need
  the second fixation barrier. Exact partitioning and code/security parameters
  remain the separate M-B/M-C gates; this numerical proposal does not select them.
- Dependent outputs remain tentative inside one request-owned graph. On any
  malformed/replayed/missing/failed claim or cancellation, retire all its private
  owners and tentative descendants; release no logits, token, cache, or success
  receipt. Ordinary acceptance must bind the exact complete request, graph and
  final logits and require **every** claimed linear result to have passed.
  Existing two-input-upfront native batch owners do not implement this graph.
- Code distance, actual-field failure probability, dual-LPN applicability and
  total-window/query security accounting remain independently required. Neither
  oracle parity nor the following quality test qualifies cryptographic security.

The one-use dependency is the paper's, not a new role/state profile:

```text
publicly admitted source + code + fixed ordered claim IDs
  -> fresh one-use mask/correction for each of 2,017 claims
  -> frozen masked input and complete response; tentative demasking/local graph
  -> all request originals frozen before any primary reveal
  -> complete ordered auxiliary responses frozen before secondary sampling
  -> all auxiliary + original equations checked
  -> exactly one accepted full-logits owner OR terminal retirement of whole request
```

IDs bind `(request,layer,role,token,matrix,publication,numeric-map)`; the head binds
position 7 and the full vocabulary. Neither a worker-supplied ID nor a borrowed
mutable response buffer can choose the verified graph. Fresh independent privacy
noise is required even for equal clear inputs or repeated matrices. Once reserved,
mask inventory cannot be reused after cancellation, failure or retry. Challenges
must be sampled after fixation, not merely kept secret after earlier sampling.
Secrets/corrections and secondary challenges remain client-owned. An untrusted
publisher's correct P/Q entries do not establish trustworthy code selection.
Fetching P columns remotely at secret sparse-mask indices is not acceptable
storage: it leaks the mask support. Public state must have a client-owned,
aggregate-admitted access plan before an online claim.

Security accounting is not reset by rebuilding a client. For a fixed collection
of actual codes C and batch attempts W, require a reviewed bound of the form
`sum_C p_bad_code + sum_W(eps_primary^ell + eps_secondary^ell_secondary)` and a
dual-LPN argument covering at least all `2,017*requests` fresh privacy samples
plus cancellations/retries. These are not automatic 128/40-bit deployment claims.
For context only, source `31×2` with actual distance 1/2 has per-relation
conditional error `961/1037629354146162278400`; multiplying by 2,017 gives about
`2^-48.927` before code failures if checked individually. The source system's
Protocol-2 bound instead counts its frozen batches, without an extra B factor.
This example selects neither an individual-check replacement nor model batches.

Malformed/publicly invalid frames may retire a request; canonical but false
responses must not trigger a secret-dependent early numeric abort. The public
totality certificate is required before any such request. Changing to early
verified windows, numeric fallback, dummy continuation, remote attention/lookup,
or persistent state is **not** authorized by this numeric proposal. Availability,
machine faults and complete side-channel security are not established by it.

### Proposed quality gate, explicitly not a paper-recovered threshold

To avoid selecting tolerances after observing results, propose this fixed,
one-shot, non-generative screen as part of the approval decision:

1. Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
   revision `b08601e04326c79dfdd32d625aee71d232d685c3`, file
   `wikitext-2-raw-v1/test-00000-of-00001.parquet`, advertised LFS SHA-256
   `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
   Validate the file hash; read the `text` column in physical row/row-group order,
   require string values, retain empty strings, and concatenate text with exactly
   two LF characters between rows (no additional leading/trailing separator or
   Unicode/newline normalization). Use the pinned Tokenizers v0.21.1 source in §2:
   load the validated tokenizer JSON as strict UTF-8 with `Tokenizer.from_str`,
   call `no_padding()` and `no_truncation()`, then `encode(text,
   add_special_tokens=False).ids`. Require its BPE dropout to be absent/null/zero;
   reject stochastic configuration. Apply the tokenizer's serialized normalizer
   and pretokenizer, not a Transformers chat wrapper or external normalization.
   Use the first 1,024 disjoint nine-token
   blocks: first eight IDs are input, ninth is held-out target. No shuffling,
   filtering, retry, or discarded failing example; reject if corpus is too short
   or any selected ID is outside the model vocabulary.
2. BF16 comparison uses the pinned **BF16** source above and the same scalar/local
   graph schedule, replacing each `Lin_W` by `R16(D32(W_BF16,x))`, row by row,
   without field encoding or FP8. Require compatible config/tokenizer and validate
   its source tensors. Its pinned index contains **398** tensors: the same
   projection/norm names without FP8 scales, plus embedding and final norm,
   **without a head tensor**. Require BF16 types and exact source shapes; derive
   its head from that source's embedding as dictated by its tying declaration.
   Do not use the FP8 candidate's embedding or require a nonexistent second copy.
   This is an explicitly defined **original-BF16-checkpoint comparison oracle**,
   not an assertion of bitwise parity with an unspecified GPU framework.
3. Require no failed/overflowed example; candidate and clear-field oracle must
   agree exactly for all 1,024 requests. Require at least **973/1,024** next-token
   top-1 matches against that BF16 comparator (ties choose lowest token ID) and
   mean target NLL increase at most **ln(1.05)** nats. Report every comparison,
   max logit error and rejection count, not only a passing aggregate.
4. For a deterministic metric, use round-to-nearest-even binary64 after each
   operation, correctly rounded binary64 exp/log, stable log-sum-exp, vocabulary
   sums in increasing ID order and request sums in block order. All sums start
   at positive binary64 zero; promote BF16 logits exactly. Let m be the maximum
   logit and accumulate `exp64(R64(logit[i]-m))` with R64 after each addition.
   Define each NLL as `R64(R64(m + log64(sum)) - logit[target])`.
   Accumulate the 1,024 separately rounded candidate-minus-comparator NLLs,
   then divide by 1,024 with one R64 rounding;
   threshold is correctly rounded binary64 `ln(21/20)`.

The dataset and thresholds are **proposed engineering acceptance**, not author
evaluation settings or a general language-quality guarantee. They may fail; no
checkpoint substitution, scale tuning or threshold change follows automatically.
Persistent generation quality remains outside M-N and requires M-D separately.

There are **three distinct comparisons**, not one interchangeable "clear model":

| Comparison | Required result / permitted claim |
| --- | --- |
| Delegated candidate versus independent **clear candidate** | Every field claim, local intermediate and full logit bit agrees; numerical/protocol conformance only |
| Clear candidate versus **original BF16 checkpoint** under the defined graph | Proposed 973 matches and NLL threshold; quantization/model-port quality screen |
| Candidate versus **author's quantized clear baseline** or official dynamic-FP8 engine | Not established: author's executable manifest unavailable; stock FP8 activation/kernel semantics differ |

The source paper's timing comparator is its quantized field model, not an
unquantized BF16 oracle. Paired performance must therefore use the exact clear
candidate with all import/conversion/local costs disclosed, while the original
BF16 checkpoint remains the quality reference. Passing this short-context screen
does not reproduce original-runtime quality. Its loss ratio is a fixed-eight-token
conditional cross-entropy proxy, not conventional full-context Wikitext perplexity;
95% agreement is not accuracy or an IID statistical guarantee. A requirement for
author-numeric or official-runtime parity cannot be waived by passing this screen.

## 5. Exact next decision and remaining gates

**Preferred decision: request the author numerical manifest**, naming arXiv
2609.10264v1 and asking for (a) repository commit, (b) checkpoint/tensor hashes,
FP8 subtype and scale interpretation, (c) exact activation encoding and ties,
(d) field-to-BF16 cast path, (e) local kernel/order/precision and Q/K norm/RoPE,
(f) embedding/head/bias treatment, and (g) the actual eight-token validation
inputs, complete next-logits oracle and quality/no-wrap evidence. Do not ask only
for "the FP8 model": that leaves executable numerics ambiguous.

If author artifacts remain unavailable, the precise alternative decision is
the [limited numeric adaptation request](../plans/maverick-numeric-completion.md):
the §3 equations/source pins plus the §4 acceptance screen, not blanket permission
to pick a profile. Source-faithful independent implementation of recovered facts
needs no new permission. Actual numerical changes here do; changing roles, batch
windows, state, storage access or the privacy/security contract needs its own
decision/evidence. Nothing is adopted by this document, and approval cannot stand
in for the missing public admission proofs or checkpoint authentication.

Even a resolved M-N leaves dependent optimistic-claim ownership, role transport,
M-B batch partition/lifetime, M-C code/security evidence, and complete storage
admission unresolved. For perspective, these 253 distinct delegated matrices
contain **4,022,272,000** coefficients. Materializing ordinary source rate-1/8
P and Q for all of them uses at least `64*sum(m*n)` = **257,425,408,000 bytes**
(239.746 GiB), before matrices, codes, auxiliary preprocessing, scratch, replicas,
transport or accepted-state owners. This is an arithmetic lower bound for that
materialization, not a proposed allocation or permission to raise caps. Streaming,
recomputation and publication scheduling need explicit aggregate admission.

Further exact counts for this proposed schedule: **29,455,482,880** field
multiply-add terms in honest server linear evaluation; **16,089,088,000 bytes**
for one u32 copy of all matrices; **66,678,272 bytes** for all original masked
input/output vectors if retained as u32 arrays, before IDs/framing/auxiliaries/
copies; and **5,190,052,872 bytes** in advertised FP8 checkpoint shards alone.
These are resource estimates, not measured peaks, throughput, a runtime admission
plan, permission to download, or permission to bypass existing caps. The latter
remain unchanged even if these estimates make the proposed model inadmissible.

M-D remains an independent approval for any token feedback, KV ownership,
cancellation/rollback across requests, persistent session state, EOS, or generation.
No isolated operator result or approved numerical proposal can claim those.

## Checks performed in this audit

Read-only source retrieval and metadata inspection; in-memory source archive
inspection/hash checks; exact-rational scalar checks of four signed ties, the
E4M3FN unit-subnormal tie, both BF16 cast paths, centered range, claim count and
P+Q lower-bound arithmetic. No model-quality claim or tensor-value check was run.
No builds, Cargo, checkpoint downloads, native launches, or commits.

Decision-readiness follow-up inspected only the small pinned BF16 tensor index
and tokenizer/version source metadata. It found and fixed the absent BF16 head,
unfixed quality-tokenizer engine, implicit sum/reshape conventions and missing
distinction between honest no-wrap and all-reply totality. The remaining unknowns
are authenticated payload values, actual admission certificates, and the separate
M-B/M-C/M-S implementation/security evidence; they are not silently filled by a
quality threshold. No new search for guessed artifacts or tensor download occurred.
