# Decoder arithmetic: D13 approved completion

This directory is an independent **clear bounded oracle**, not protected execution,
native replay, checkpoint reproduction, or quality evidence. No native code or
author implementation is imported. Generated tensors are not checkpoint weights.

## Settled by D1/D3/D8/D9

* Public weights, ring50/Q12, finite external F32/F16/BF16 values decoded exactly
  from their IEEE bits, then scaled by 4096 and rounded nearest/ties-even. No F32
  intermediate conversion of F16/BF16 or double rounding. Reject nonfinite and
  out-of-range codes; negative zero maps to zero.
* Every internal rescale is signed mathematical floor, never nearest-even or
  truncation toward zero. Reject before signed50 product/accumulator overflow;
  this is conservative clear fixture admission, not a protected range protocol.
* GPT-2 structure: token plus learned absolute position embeddings; pre-attention
  LayerNorm; causal MHA; attention residual; pre-MLP LayerNorm; GELU MLP;
  second residual; final LayerNorm and complete generated vocabulary head.
  Source GPT-2 uses a tanh GELU approximation; D8 deliberately selects SIGMA's
  table GELU instead. Do not call this original floating-point GPT-2 parity.
* D8 LayerNorm is exactly shared-profile-resolution section 2.2, including
  `floor(4096/k)` mean coefficient and **untruncated squared sum plus integer 1**.
  This guard is not external encoding of GPT-2 epsilon `1e-5`. Do not replace it
  with source checkpoint epsilon. Gamma/beta are learned Q12 public coefficients.
* D9: private equality-to-one-hot vocabulary lookup; public fixed T; recompute
  every position of growing prefix each selection; lowest-index maximum;
  append privately; only final token-ID array released to client. EOS does not
  stop execution. No persistent KV state, streaming, or logits release.

## Whole-graph choices now approved by D13

User approved both exact graph proposals. Identities:
`fusefss.gpt2-complete-ring50-q12.v1` and
`fusefss.dense-rope-ring50-q12.v1`. Following arithmetic is no longer pending:

1. Linear contractions: oracle sums Q24 products in input-feature order,
   admits every product and partial sum, floors once to Q12, then adds Q12 bias.
   It does not floor every product independently, fold bias into Q24, or commute
   contractions/rescales. GPT-2 Conv1D source storage must transpose to canonical
   `[out,in]`; tied head is transpose of embedding lookup interpretation, using
   exactly the same `[vocab,hidden]` bytes. Untied head must be explicit.
2. Attention: forms Q24 QK contraction, floors to Q12, multiplies by
   explicitly supplied Q12 `1/sqrt(head_dim)`, floors again, then D8 softmax.
   Scaling Q before the dot, combining scale with one Q36 shift, and source
   floating division can differ. No inverse-layer-index attention scaling or
   reordered/upcast GPT-2 attention is inferred. Generated head_dim=4 has exact
   scale 2048, avoiding coefficient-generation ambiguity but not ordering choice.
3. Common-Qwen RMSNorm: D13 explicitly authorizes this numerical substitution.
   Source declarations use mean square plus positive epsilon (typically `1e-6`).
   D13 instead uses uncentered `sum(X²)+1` and width-dependent
   rsqrt table. Those are **different functions**, even though both avoid zero.
   D10's epsilon-one decision is PUMA Q18 only; cannot transfer it to FuseFSS.
4. Q12 RoPE: existing semantic contracts establish split-half pairing, absolute
   positions, `[batch,sequence,heads,feature]` input and causal GQA grouping. They
   alone did not approve Q12 coefficient generation or truncation order. D13 now
   selects pinned finite F32 coefficient bytes, nearest-even encoding to Q12,
   and separate **signed** term products/floors before addition. For halves
   `l,r` and Q12 coefficients `c,s`, outputs are exactly
   `floor(l*c/4096) + floor((-r)*s/4096)` and
   `floor(r*c/4096) + floor(l*s/4096)`. Rotate-half negation occurs **before**
   multiplication/floor. `-floor(r*s/4096)` is not interchangeable with
   `floor((-r)*s/4096)`; the original oracle already implements the latter.
   Parent clarified D13 wording without changing these equations or fixtures. Public
   cos/sin are explicit finite F32 bit payloads before nearest-even Q12 encoding.
   Source F32/BF16 stepwise semantics are not this integer graph. No scaling,
   partial RoPE, sliding windows, multimodal RoPE, or cache adaptation is inferred.
5. Qwen2: dense SiLU gated MLP, Q/K/V biases when explicitly supplied. Qwen3:
   additionally per-head learned Q/K RMSNorm **before RoPE**; Llama: explicitly
   no Q/K normalization and no projection biases in generated control. Existing
   runtime semantic contracts support these structures, not D8 numeric adoption.

**D13 status:** all generated full-decoder fixtures carry
`d13-approved-arithmetic`, with explicit per-family profile. Original finite
F32/F16/BF16 weight payloads are decoded exactly, then nearest-even Q12 encoded;
no intervening BF16/F16 narrowing is inferred. Approval settles arithmetic, not
native parity, evaluated-build identity, full-checkpoint quality or ordinary SDK
qualification. Qwen3/Llama generated controls exercise supported semantic
structures under the dense profile; D3's selected common checkpoint remains
Qwen2.5-0.5B-Instruct. Generated RoPE F32 payloads are frozen fixture coefficients,
not proof of any checkpoint/framework coefficient generator. A complete checkpoint
importer must freeze its own F32 coefficients.

## Evidence inspected, not executed

`paper/methods/DECISIONS.md` D1/D3/D8/D9/D13;
`paper/reproduction-contracts/shared-profile-resolution.md` sections 2, 5.1, 6;
`paper/reproduction-contracts/fusefss/gate-set/reference.py`;
`python/pllm/runtime/semantic_attention.py` physical causal GQA axes;
`python/pllm/runtime/semantic_numeric.py` (separate BF16 semantics);
`python/pllm/runtime/tiny_llama.py` generated Qwen2/Qwen3/Llama declarations.
Frozen file hashes live in `source-lock.json` after `freeze.py` runs.
