# Shared-computation reimplementation contracts

Review: 17 October 2026. Scope: SIGMA, FuseFSS, BumbleBee, and PUMA.

**None currently has a qualified, complete independent PLLM implementation.**
This document specifies the work and decisions required. It authorizes no substantial
protocol, numerical, state, security, or execution adaptation. Named missing details
are qualification blockers, not permission to choose convenient replacements.

Evidence: primary PDFs, [coverage audit](../TEN-METHOD-COVERAGE.md), component cards,
and static inspection of the local paths below. No implementation, build, benchmark,
model execution, or upstream software execution/linking was performed.

The independent [source-fidelity audit](../reproduction-contracts/shared-computation.md)
adds immutable author-source inspections and concrete paper/code discrepancies.
It preserves its original canonical snapshot by hash. The refinements below
incorporate those findings; recommendations remain distinct from approval.

## 1. Source identity and reading convention

Page references are **1-based PDF pages**, including appendices. SIGMA's printed
page number is PDF page + 60. Hashes identify reviewed bytes, not every revision
reachable through a mutable URL.

| ID | Governing source | Reviewed bytes and principal locators |
| --- | --- | --- |
| S | *Sigma: Secure GPT Inference with Function Secret Sharing*, PoPETs 2024(4), 61–79 | [PDF](../../papers/r13-sigma.pdf), SHA-256 `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`. Representation/security pp.2–4; protocols pp.5–10; implementation/evaluation pp.10–13; RoPE/SiLU/norm pp.16–18; preprocessing p.19. |
| F | *FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing*, arXiv:2606.09551v1, 8 June 2026 | [PDF](../../papers/r14-fusefss.pdf), SHA-256 `6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`. Setting pp.3–4; compiler/theorems pp.4–7; evaluation pp.7–8,12–15; full compiler p.17; backend pp.20–21; proof/accounting pp.23–27. |
| B | *BumbleBee: Secure Two-party Inference Framework for Large Transformers*, NDSS 2025 final | [Publisher PDF](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf), SHA-256 `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`. Setting pp.3–4; OLT/InterLeave pp.5–7; multiplication p.8; activations pp.8–10; evaluation pp.11–13; security/conversion pp.15–16; constants/parameters pp.17–18. |
| P | *PUMA: Secure Inference of LLaMA-7B in Five Minutes*, arXiv:2307.12533v3 | [PDF](../../papers/puma.pdf), SHA-256 `84c82810a16be95b41200ba8e0071c39a8cadb15a45008ff8e75e295e0960973`. Stamp: 26 September 2023; title-page date: 27 September. Sharing/security pp.3–4; algorithms pp.5–7; evaluation pp.7–9; architectures/SiLU qualification pp.12–13. |

All five relevant PDF hashes, including BumbleBee's older cache, were recomputed.
B's final snapshot is `bumblebee-final.pdf` under `fetched_snapshot_base` in
[prior-art-sources.json](../prior-art-sources.json). A verified identical copy now
lives at [r22-bumblebee-final.pdf](../../papers/r22-bumblebee-final.pdf), recorded
in the [ten-source lock](source-lock.v1.json). The historical record is unchanged.
The local [older BumbleBee PDF](../../papers/r22-bumblebee.pdf), SHA-256
`15583a86593998079d42b8b134dc4f4f5be9ca72a7ced8f31bba05f68893516a`, differs from
the final, including the LLaMA evaluation. In **both reviewed byte sets**, OLT and
Figure 6 are in §III, pp.5–7. The component card's §IV locator is stale. Keep its
historical source identity separate from the final-paper contract.

## 2. Common acceptance contract

These are proposed implementation gates, not additional guarantees attributed to
the papers. The [comparison contract](../COMPARISON-CONTRACT.md) remains separate.

1. **Freeze identity.** Bind PDF, checkpoint/tokenizer revision and hashes, complete
   graph, operator descriptors, widths/scales, coefficient/LUT bytes, truncation
   schedule, roles, device, randomness setup, output policy, and public capacities.
   A paper model name alone does not pin its evaluated checkpoint.
2. **Implement complete computation.** Token input/embedding, positions, every
   attention/MLP/normalization/residual operation, final normalization, complete
   head, and authorized result handling must execute under that method's protection.
   No clear client attention or nonlinear fallback counts as protected coverage.
3. **Separate correctness from quality.** Use independently written scalar/ring
   oracles and a clear interpreter of the frozen candidate graph. Require exact
   parity for deterministic semantics; explicitly model randomized truncation or
   approximate conversion errors. Separately qualify against original floating-point
   model/task quality with a predeclared acceptance margin. W8A8 parity is not the
   default oracle for these papers. Never transfer author-reported scores or costs.
4. **Make state explicit.** A complete original fixed-length model evaluation is
   the first vertical slice, not a complete serving comparator. Before serving,
   bind protected state ownership, cache/recomputation policy, token feedback,
   positions, lengths, pending token, EOS, sampler randomness, continuation,
   cancellation, and retirement. Test at least prefill plus two selections with
   actual feedback, including equal-score tie behavior and capacity boundaries.
5. **Keep views separate.** Record each role's inputs, shares/keys, disclosures,
   permitted collusion, and output recipient. Test-only reconstruction of hidden
   tensors must not become a production disclosure. None of these source contracts
   supplies malicious-result verification. Authenticated transport does not add it.
6. **Fail closed.** Reject unsupported graph nodes, numeric profiles, HE parameters,
   device paths, or resource shapes before execution. Test wrong role/session,
   malformed/truncated messages, replay, disconnect, timeout, cancellation, and
   partial setup. Retire consumed randomness and avoid partially committed state.
   A secret-dependent range failure must not reveal the offending value or location.
7. **Admit all owners.** Before material issuance or weight allocation, account for
   checkpoint encoding, every model/share copy, dealer/setup material, temporary
   serialization, communication buffers, activations, state, and device transfers.
   Bound live tiles separately from total consumables. Later measure each role's
   full lifecycle, client helpers, device memory, simultaneous host totals, and
   directional/all-link traffic. Estimates, key bytes, and RSS are different quantities.

## 3. SIGMA

### Original functionality and trust

- **Models:** BERT-tiny/base/large encoders; GPT-2, GPT-Neo-1.3B, Llama2-7B/13B
  causal models. S Table 1, p.11 gives `(blocks, heads, hidden)` as `(2,2,128)`,
  `(12,12,768)`, `(24,16,1024)`, `(12,12,768)`, `(24,16,2048)`,
  `(32,32,4096)`, `(40,40,5120)`, respectively. S §7.2.3, p.13 also evaluates
  a constructed GPT-like 2.7B scaling model; it is not a separately identified
  pretrained checkpoint. BERT/GPT use GeLU and LayerNorm; Llama2 uses SiLU,
  RMSNorm, and RoPE (S pp.4–5,16).
- **Coverage:** classification and LAMBADA next-word evaluation, normally 128 input
  tokens; GPT-2 length sweep 64–1,024 (S pp.11–12,19). This establishes
  autoregressive *architectures*, not a specified persistent-KV serving lifecycle.
- **Roles/security:** model owner and input owner are the two online parties.
  A trusted, input-independent preprocessing dealer creates masks and correlated
  keys. Static semi-honest corruption of either online party is covered; dealer
  compromise/collusion is outside that assumption (S §2.2, p.3). Model and input
  are private. Only the designated owner obtains each input/output wire's mask.
  Public circuit/shape and masked wires are distinct from unmasked activations.
- **Wire invariant:** for each wire, evaluators know `xhat = x + r mod 2^n` and
  their own gate keys; they reconstruct masked gate outputs. The dealer connects
  adjacent gates using the same logical wire mask, with independent masks for
  distinct wires. Public-weight two-worker outsourcing with a client dealer is
  a changed topology, not this source contract (S §§2.4,3.2.3, pp.3,5).

### Numerical and protocol contract

| Item | Required source behavior |
| --- | --- |
| Encoding/rings | Two's-complement `floor(real·2^f) mod 2^n`, not nearest rounding. All evaluated models use `f=12`. Table 4 fixes `n=37` for BERT-tiny, `50` for BERT-base/large and GPT-2, `51` for GPT-Neo, `48` for Llama2-7B/13B (S pp.2,11–12). |
| Linear arithmetic | Masked multiplication/matmul with dealer correlations, followed by protected precision reduction. A scalar masked multiplication uses the paper's `3n`-bit per-party key construction; it is not a new standalone Beaver opening at every existing masked-wire boundary (S §§3.2,4.1, p.5). |
| Truncation/range | Exact arithmetic right shift (floor on signed integers). `TR(n,f)` drops low bits and returns ring `2^(n-f)`; general ARS is TR plus protected sign extension. GapARS requires signed `v ∈ [-2^(n-2),2^(n-2))`, adds the public quarter-ring offset, then corrects it. Low carry and high wrap must both be handled. Comparison through subtraction also requires the stated quarter-ring operand bounds (S §§4.2–4.3, pp.6–7). |
| FSS backend | Boolean-output, early-terminated DPF and DPF-derived comparisons; `λ=127`, AES-128 counter PRG, half-PRG evaluation. Theorem 1 gives per-party DPF key bits `2^n` for `n≤7`, otherwise `(n−7)(λ+2)+2λ` (S §§2.5–2.6, pp.3–4). Preserve the source seed/control encoding for a cost-faithful backend. |
| GeLU/SiLU | `ReLU(x) − δ(x)` with an even residual. At Q12, GeLU uses 256 LUT entries over the clipped `(-4,4)` domain; SiLU uses 1,024 over `(-16,16)`. Table entries are floored Q12 values indexed at six fractional bits. Implement the selected CPU or GPU evaluation order, including its early TR and sign/interval handling; do not replace it with an arbitrary polynomial (S §5.1/Figs.5–7, pp.8–9; App.F, pp.16–17). |
| Softmax | Protected maximum, subtract maximum, negative exponential, sum, one reciprocal, products and protected shifts. nExp clips its nonnegative magnitude to a 16-bit Q12 interval and uses two 256-entry LUTs, multiplication and GapARS. Reciprocal input lies in `[1,k]`; reduce from Q12 to Q6, then use a `q=6+ceil(log2(k+1))`-bit LUT. Masked causal entries contribute zero and skip unnecessary max/nExp work (S §§5.2,5.4.2, pp.9–10). |
| Norm | Reciprocal-square-root LUT has 13 index bits: seven mantissa and six exponent bits, produced through interval lookup, multiplication, and TR. Preserve Fig.12's mean, untruncated sum of squares, `s+1` zero guard, adjusted input precision `2f+log2(k)`, learned affine terms, and justified GapARS operations. RMSNorm removes centering as specified; do not substitute host floats (S §5.3, p.10; App.H/Figs.11–12, pp.17–18). |
| RoPE | Source Llama2 convention, public coefficients, reduced coefficient precision `f−3`, and the resulting proved-gap shift. Freeze position and coefficient rounding, including endpoints (S App.C, p.16). Modern RoPE variants are not implied. |

**Qualification gaps:** S §3.1.1/Fig.1, p.4 describes embedding and the final
linear/softmax path but does not give a token-ID lookup protocol or a complete
token-output/sampling API. Identify those boundary semantics before claiming a
complete token-to-token implementation. The paper does not specify persistent
KV append/read, cache ownership across calls, or cancellation/restart behavior.
Do not infer a cache from the word “GPT.”

### Required PLLM work and honest reuse

- **Native:** general-width masked tensor arithmetic, private-model matmul,
  source DPF/selection/LUT/truncation, complete softmax and norms, protected
  embedding/head composition, and independent dealer generation/distribution.
  GPU mode additionally needs source-style AES, LUT traversal, matrix, bit-packing,
  and host/device streaming kernels. Source CPU mode uses a different activation
  schedule and omits the GPU network-packing optimization (S §§5.1,6, pp.8–11).
- **Compiler/runtime:** original-family graph import, a fixed-point interpreter,
  explicit truncation insertion, effective-width/gap proofs, causal masking,
  per-wire masks, one-use key inventory, two online owners, and separate dealer
  lifecycle. S §6.0.3, p.11 describes this compiler responsibility.
- **Reusable basis:** [compact_dpf.rs](../../crates/pllm-garble/src/compact_dpf.rs)
  provides independently written Boolean DPF algebra; [masked_wire_reference.rs](../../crates/pllm-garble/src/masked_wire_reference.rs)
  provides bounded GapARS/TR, selection, residual lookup, and one-use framing
  patterns. Preserve their oracles while porting, rather than claiming byte identity.
- **Must replace/generalize:** current DPF uses 128 seed bits plus separate control
  bits; current block uses a 40-bit Q24→Q12 profile with gate/up bounds 48/128.
  [masked_wire_schedule.rs](../../crates/pllm-compiler/src/masked_wire_schedule.rs)
  explicitly binds client float conversion and SiLU×up regions only.
  [masked_wire.py](../../python/pllm/runtime/masked_wire.py) issues material at the
  client and relays workers. Reuse framing/retirement patterns, not those role or
  numerical choices. Existing Qwen response evidence is not SIGMA evidence.
  See [component card](../../docs/research/masked-wire-reference.md) and
  [decoder card](../../docs/research/nonlinear-decoders.md).

**Minimum complete-model slice:** full original GPT-2, 12 blocks/768 hidden,
128-token input, Q12/ring50, source CPU schedule or source GPU schedule explicitly
selected. Include embedding, positions, all layers, full head, and source-qualified
result disclosure. Boundary qualification above remains a prerequisite. The
published CPU implementation is a faithful device choice; a CPU execution of
today's Qwen gate is not its reproduction.

**Oracles/admission:** exhaust small rings over all masks for ARS/TR/comparison;
test gap endpoints and an outside-gap rejection; enumerate LUT inputs and negative
rounding boundaries; compare every layer and final output to the independent
integer interpreter. Evaluate source tasks separately. Budget dealer generation,
both parties' keys, host buffers, GPU staging, and online traffic. S App.L, p.19
explicitly separates key generation/transfer from online time; streaming does not
eliminate that material. No existing local component limit admits the full model.

## 4. FuseFSS

### Original functionality and trust

- **Models:** BERT-tiny/base/large and GPT-2/GPT-Neo at sequence length 128,
  batch one; BERT-base/GPT-2 length sweeps; LLaMA-7B and **LLaMA-3.1-8B** at
  lengths 16/32/64 (F Table 1, p.8; Tables 7–8, p.13). Thus encoder and causal
  model coverage both exist. Llama-3.1 is already in the source evaluation;
  choosing it is not inherently an architecture invention. Precise checkpoint,
  graph, and numerical profiles still need qualification.
- **Roles/security:** two non-colluding online servers; client shares input;
  conceptual trusted preprocessing dealer; semi-honest corruption of at most one
  server. Architecture and parameters are public unless stated otherwise, while
  client inputs/intermediates are secret. The compiler can compose with public-
  or secret-weight linears (F §3.1, p.3). Do not silently inherit SIGMA model privacy.
- **State:** masked arithmetic wires and XOR-shared helper bits. Fresh independent
  masks/keys/triples per scalar wire and inference. Reusable *compiled programs*
  are not reusable private material (F pp.3,7,14–15). The reported “cache” ablation
  is a program cache, not KV-cache evidence. Persistent decode state is unspecified.

### Exact compiler/numerical contract

- Arithmetic type `A_n = Z/(2^n)` uses additive shares; `B` uses XOR shares.
  Real input encoding is **nearest integer with a fixed tie rule**, whose identity
  is not supplied; signed ARS is two's-complement floor shift (F §§3.2–3.5, p.3).
  F App.B.3, p.14 says activation benchmarks compile SIGMA's same fixed-point
  operator specifications. It does not publish a complete per-model width/scale,
  rounding, LUT/coefficient, or rescaling manifest. Qualify that manifest; do not
  silently equate nearest encoding with SIGMA's floor encoding.
- A public typed descriptor gives a full canonical interval partition,
  degree-bounded arithmetic pieces, Boolean predicates, and fixed-point metadata.
  A fixed share-based post-processing circuit `Φ` supplies truncation, corrections,
  multiplications, ANDs, and B2A/A2B conversions. Normalized Boolean formulas
  and interval selection must have no secret-dependent control flow (F pp.4–6,18–19).
- For `xhat=x+r mod N`, rewrite `[x<β]` as
  `[xhat<(r+β) mod N] XOR [xhat<r] XOR [r+β≥N]`.
  Keep the carry secret-shared; apply the analogous rule to low-bit predicates.
  Handle signed shifts and sentinel thresholds explicitly (F pp.5,16,22).
- Translate the lookup partition and pad to `M=min(m+1,2^n)` intervals, including
  wrap-at-boundary and full-domain cases. Predicate count/widths, payload width,
  key lengths, and post-processing counts depend only on the public descriptor.
  Never remove queries because a sampled secret threshold happens to be zero
  (F Protocol 3, p.17; App.I, pp.20–21).
- “Two calls” means at most two **batched backend interface calls per scalar
  gate**, not two keys per tensor, two network rounds per model, or free secret
  arithmetic. App.I.1 permits one comparison key per query. Charge Horner/`Φ`
  Beaver products, Boolean AND, conversions, and masked openings. Vector maximum,
  sum, attention, and top-k remain separate circuit operations (F pp.6,18,20,27).
- Theorem 4.7 is conditional gate-level semi-honest security with public shape
  leakage and secure backend/subprotocol assumptions. Its composition argument
  does not automatically certify a new PLLM backend or output/state extension
  (F pp.6–7,24–25).

### Required PLLM work and honest reuse

- **Native/compiler:** general typed descriptor IR, Protocol 3 compiler,
  multi-view predicate and vector-payload interval lookup backends, conversions,
  fixed post-processing interpreter, secure reductions, and complete protected
  tensor execution. Bind widths, ranges, coefficients, and exact arithmetic
  order in the program identity. Numerical overflow changes are not optimizations.
- **Runtime:** client sharing, two share holders, independent preprocessing,
  fixed-shape batching, one-use material streaming, and GPU program reuse.
  Source device: one RTX PRO 6000 Blackwell GPU per party, EPYC 9654 CPUs,
  CUDA 13.0 (F §6.1, p.7). A CPU executor is a new execution profile.
- **Reuse:** source-qualified SIGMA kernels can support the same mathematical
  operations. Existing local DPF, interval algebra, serialization, and cancellation
  patterns are starting points. [interval_fss.rs](../../crates/pllm-garble/src/interval_fss.rs)
  has a bounded ring32 arithmetic-output interface; it is not the typed general
  compiler/backend. [Shared-rescale card](../../docs/research/shared-rescale-reference.md)
  distinguishes its floor helper from PLLM's ties-even extension.
- **Replace:** `MaskedWireGatedMultiplyQ12(layout="fused")` is one hard-coded
  eight-phase Qwen block, not operator-specification compilation. Its local
  dealer, client float bridge, wider fixed keys, and CPU scheduling cannot supply
  the source's whole-model contract. `FuseFssPredicateFusion` remains pending.

**Minimum complete-model slice:** original GPT-2 at 128 tokens, using a qualified
full numerical manifest and the source two-server/dealer topology. Compile all
applicable activation, nExp, reciprocal/rsqrt, and rescaling gates through the
general compiler; execute embedding, complete body/head, and vector reductions.
Pair it with an independent unfused interpreter/protocol using exactly the same
descriptor semantics. Boundary/output and recurrent-state qualification remain
required; one fused activation is not this slice.

**Oracles/admission:** exhaustive small-domain descriptor interpretation versus
compiled results across every mask; sentinel/full-partition/wrap cases; XOR versus
arithmetic type errors; coefficient order and all truncation placements. Across
masks, assert identical public shapes and serialized lengths. Verify fused/unfused
whole-model parity before quality assessment. Price every query key, lookup payload,
triple/conversion, GPU program, host key buffer, and transfer. F Table 8, p.13
explicitly uses host buffers and streaming. Its projected LAN/WAN numbers are
not new PLLM network measurements.

## 5. BumbleBee

### Original functionality and trust

- **Models:** BERT-base/large encoders; ViT-base image classifier; GPT2-base and
  original LLaMA-7B causal models (B §VII, pp.11–13). GPT2 is `(12,768,12)` and
  LLaMA is `(32,4096,32)` in `(blocks,hidden,heads)` order; ViT-base is
  `(12,768,12)`, 224×224 RGB input (B Tables IV/VI, pp.13,17).
- **Coverage:** complete protected inference, with secure token-ID equality,
  embedding, body, and top-1 output selection in the reported language-model
  breakdown. GPT2 uses 128 input tokens and LLaMA eight for the one-token
  breakdown. Source does not specify persistent KV maintenance (B pp.12–13).
- **Roles/security:** client plus private-model server, no trusted preprocessing
  dealer. Static semi-honest corruption of one party. Client learns public
  architecture and authorized inference result; server may learn the result only
  if the application permits it (B §II.D, p.4; App.B, pp.15–16). Both parties
  hold additive arithmetic shares and XOR Boolean shares. Distributed OT and HE
  key setup are required. Local random-OT issuance changes this contract.

### Numerical and cryptographic contract

| Item | Required source behavior |
| --- | --- |
| Base representation | Two's-complement ring64, floor encoding, 18 fractional bits. Multiplication/square implicitly include protected truncation. Full truncation construction/error policy comes from cited subprotocols, not an assumed sharewise local shift (B pp.3–4,9,11). |
| Mixed precision | Evaluated mixed profile uses ring32/Q12 for selected nonlinear blocks, with secure downscale and ring extension. §VII, p.11 applies it to GeLU and exp for BERT-base/GPT2/ViT, and activation only for larger models. App.E/Table VI, p.17 instead says ViT exp is ring64: qualify this discrepancy before a ViT claim. Algorithm 5's positive-offset ring extension has an unspecified heuristic `M`; prove the admitted range or qualify its original setting (B pp.10–11,17–18). |
| Matrix multiplication | OLT on private matrices using coefficient packing; one OLT for shared×private-held matrix, two for shared×shared cross terms. Block windows satisfy `m_w` power-of-two and `k_w·m_w·n_w≤N`. Optimized InterLeave divides in the **odd ciphertext modulus**, then performs the FFT-style trace/negacyclic merge; ParseMat restores indices. Preserve batched attention packing and the source's small-output strategy, including PackLWEs when appropriate (B §III/Alg.1/Fig.6, pp.5–7; Fig.8, p.17). |
| HE parameters | App.E, p.17: OLT `N=8192, q≈2^147, t=2^64`, automorphism modulus `q'≈2^49`; bOLE `N=8192, q≈2^148`, CRT plaintext product of three approximately 45-bit primes. SIMD noise-flooding range parameter is `B'=2^98`. Exact primes, noise bounds, key-switch decomposition, and failure/security parameters need a frozen validated profile. Approximate bit counts alone are not that profile. |
| bOLE with error | `Lift(x)=round(t·x/2^ℓ) mod t`, `Down(y)=round(2^ℓ·y/t) mod 2^ℓ`; prime `t≡1 mod 2N`, `t>2^(2ℓ)` in the abstract construction. Uniform masking and Down introduce `e∈{−1,0,1}` in reconstructed products. Preserve this approximate functionality and its CRT realization (B §IV/Alg.2, p.8). |
| Circuit privacy | HE→share has **two** paths. SIMD uses noise flooding before decryption. Coefficient form uses ciphertext-ring masking plus fresh encryption of zero and approximate resharing; App.C allows a ±1 coefficient error. Positive decryption noise budget alone does not establish circuit privacy. The ideal exact OLT notation must be reconciled with this concrete error contract (B Remark 1, p.4; App.C/Alg.4, p.16). |
| Activations | Wide GeLU has tails `−ε`, `x−ε`, cubic/sixth-degree middle pieces at −5, −1.97, 3. Mixed GeLU may use the degree-four `0.5x+Σc_i|x|^i` profile on `[-2.7,2.7]`. SiLU uses tails and quadratic/sixth-degree pieces at −8, −4, 4. Bind the actual decimal constants in App.D, p.17, not a new fit. Retain square/cubic reuse and batched approximate branch selection; source uses Ferret OT, 16-way messages, and a 50-bit MSB input for the stated wide branch selection (B pp.8–11,17–18). |
| Softmax/norm | Protected maximum, shifted negative exp, sum, one reciprocal then broadcast/products. Evaluated exp uses `Texp=−14`, six squarings of `1+x/64`; the illustrative §V.B example uses −13. Approximate branch selection, truncations, reciprocal and rsqrt are part of the contract. Freeze mixed-profile thresholds/precision separately (B pp.9–11; App.A, p.15). |

**Qualification gaps:** exact tie rules for Lift/Down; concrete truncation,
reciprocal/rsqrt schedules; approximate-comparison treatment at interval endpoints;
positive-extension bound; and normalization divisor/epsilon. Eq.(2), p.8 and
Alg.3, p.9 also need reconciliation: Alg.3 adds `ε` to polynomial pieces but
does not show the final subtraction needed for Eq.(2)'s tails. Its inequality
tests and interval annotations disagree at endpoints. B p.4 writes the
normalization denominator using a sum of squared deviations, not an explicitly
averaged variance. Checkpoint-equivalent treatment must be resolved, not silently
corrected. Source's approximate HE conversion also precludes promising exact
ring equality for every complete execution without specifying its error model.

### Required PLLM work and honest reuse

- **Native:** full-width OLT, shared matrix cross terms, ring64-compatible RLWE
  coefficient arithmetic, InterLeave/PackLWEs/ParseMat, CRT SIMD bOLE, both
  circuit-private H2A paths, distributed base OT/Ferret extension, truncation,
  mixed-ring conversion, equality/comparison/mux, activations, reciprocal/rsqrt.
  These are independent implementations, not a link to the source framework.
- **Compiler/runtime:** complete original model graph; constant versus secret
  operand classification; packing-window selection; explicit precision/conversion
  graph; source square/cubic ciphertext reuse; two-party key custody and channels;
  complete embedding/head and private top-1. Pin model weights at the server.
- **Reuse:** [coefficient_repacking.rs](../../crates/pllm-core/src/coefficient_repacking.rs)
  and [coefficient_repacking.py](../../python/pllm/runtime/coefficient_repacking.py)
  contain independent trace, RNS division, and negacyclic interleaving algebra.
  Reuse their algebra/oracles after generalizing hard-coded degree 4,096, at most
  two RNS limbs, size-two serialization, and bounded group counts.
- **Replace:** [packed-embedding card](../../docs/research/packed-embedding-reference.md)
  describes a FastQuery-derived bounded one-hot lookup, plaintext modulus 8,192,
  low-bit packed fields, local-dealer half-gates conversion, and ring32 output.
  It is not BumbleBee OLT or a private-model transformer. Its HE wrapper is not
  evidence that the original ring64/CRT/noise-flooding profile is supported.
  `BumblebeeTwoPartyTransformer` remains pending.

**Minimum complete-model slice:** original GPT2-base, 128 input tokens, one top-1
output, source ring64/Q18 with its declared mixed nonlinear profile. Include
secure ID→one-hot, private embedding, all 12 blocks, full head, and protected
argmax. The source's measured top-1 output does not authorize releasing complete
logits for arbitrary client sampling. Resolve numerical gaps before acceptance.

**Oracles/admission:** compare packing/matmul/ParseMat to independent modular
matrix products at full and partial block boundaries; test both OLT directions,
negative monomial signs, and all-coefficient masking. Separately validate H2A and
bOLE error envelopes, noise after the largest admitted sum, and mixed-ring
conversion bounds. Test activation breakpoints and near-tied argmax. Whole-model
quality needs the actual approximate path, not clear requantization of W8A8.
Price both HE contexts, public/Galois keys, CRT copies, ciphertext scratch, base
OT/extension state, and share tensors. Source evaluations are CPU-based: two
64-vCPU/256-GB hosts; source table costs exclude disk model loading, and primitive
Table II excludes key generation/base OT (B pp.11–12). Include those in a new
complete-lifecycle contract without rebranding source timings.

## 6. PUMA

### Original functionality and trust

- **Models:** BERT-base, RoBERTa-base, BERT-large encoders; GPT2-base/medium/large
  and original LLaMA-7B causal models. Appendix A gives GPT2 configurations
  `(12,768,12)`, `(24,1024,16)`, `(36,1280,16)` and encoder configurations
  `(12,768,12)`, `(12,768,12)`, `(24,1024,16)` in `(layers,hidden,heads)` order
  (P pp.7,12).
- **Coverage:** GLUE tasks, GPT2 Wikitext-103 perplexity, GPT2 generation with
  varying output lengths, and LLaMA examples. GPT2's standard cost case is
  32 input tokens/one output; LLaMA Table 6 includes `(4,1)`, `(8,1)`, `(8,2)`.
  Appendix B shows fixed-randomness multi-token text. Neither the title nor that
  example supplies a sustained-decoding or persistent-KV contract (P pp.7–9,12–13).
- **Roles/security:** model owner and client provide shares to three computing
  parties. At most one computing party may be semi-honestly corrupted. Owners
  may coincide with computing parties; the actual placement must be declared.
  All three compute online. PUMA's third party is not an offline-only SIGMA dealer
  (P §§3.3,4.1, pp.3–4). Model, input, and intermediates remain shared; results
  return to the client. Inference-output attacks are outside the paper's guarantee.
- **Sharing/setup:** `x=x0+x1+x2 mod 2^ℓ`; `P_i` holds `(x_i,x_(i+1))`, indices
  modulo three. Boolean shares use the same replicated structure over bits.
  For multiplication, `P_i` forms `z_i=x_i y_i+x_(i+1)y_i+x_i y_(i+1)`, adds
  zero-sum setup randomness `α_i`, and sends to `P_(i−1)` to restore replication.
  Implement the referenced setup for correlated randomness; do not introduce a
  dealer who knows all live private state (P §3.3, p.3).

### Numerical contract and unresolved paper details

- **Base arithmetic:** ring64/Q18 (P §5, p.6). Protected `Trunc_f` follows
  fixed-point multiplication/square and is implicit in their notation (P p.4).
  The PDF does not fix input rounding, the exact truncation distribution/failure
  bound, or reciprocal/rsqrt algorithms: these are black-box dependencies citing
  ABY3, Lu et al., and MP-SPDZ (P pp.3–4). Resolve the underlying constructions
  and bounds before claiming exact numerical identity.
- **GeLU:** P Eq.(2)–(3)/Alg.1, p.5 specifies `0` below −4, a cubic on
  `[-4,−1.95)`, a sixth-degree polynomial on `[-1.95,3]`, and identity above 3.
  Freeze the printed coefficients verbatim, coefficient encoding, powers
  `x²,x³,x⁴,x⁶`, protected shifts, and Boolean–arithmetic selection. This is
  different from BumbleBee's shifted boundaries/tails and SIGMA's LUT.
- **Softmax intent:** subtract protected maximum and positive epsilon; use clipped
  negative exp `0` below −14, otherwise `(1+x/32)^32`, implemented with five
  protected squares; one reciprocal of the sum and vector products (P §4.3,
  pp.5–6). **Alg.2 conflicts with that description:** its comparison is on the
  original `x` before max subtraction, and its mask is applied after normalization.
  The printed algorithm therefore does not simply implement clipped exp on the
  shifted input. This requires source clarification or an approved correction.
- **Epsilon/range:** suggested `10^-6` is below one Q18 unit (`2^-18`), so it
  need not survive encoding. Do not claim strictly negative exponent inputs until
  the encoded epsilon is fixed. Preserve secure max, clipping, and normalization;
  do not substitute a public shift or narrow checkpoint-dependent score range.
- **Embedding:** compute Boolean one-hot shares via `Eq(i,id)` for every vocabulary
  entry, then Boolean–arithmetic multiplication with shared embeddings. No
  fixed-point truncation is needed for the Boolean selection. Client-generated
  one-hot queries or DPF lookup change this protocol (P §4.4, p.6).
- **Normalization:** Alg.3, p.7 uses shared mean, sum of squared deviations,
  one rsqrt, broadcast multiplication, and learned affine terms. Printed
  `σ=Σ(x−μ)²` lacks the usual variance division by width and explicit epsilon.
  Resolve the evaluated semantics instead of silently inserting either.
- **LLaMA:** App.B, p.12 directs readers outside the PDF for the SiLU polynomial
  coefficients. The independent audit locates them in the cited example at a
  contemporaneous pinned SPU revision. Its zero/identity tails and multiplication
  order are explicit. Association with the evaluated build, encoding, compiler
  lowering, RoPE/RMSNorm and sampling still require qualification. Borrowing
  BumbleBee's different tails or SIGMA LUTs would create an adaptation. GPT2
  remains the better first slice.

### Additional author-source qualification

The independent audit records exact commits, paths and hashes for these findings:

- **SIGMA:** original CPU and GPU schedules are both valid source targets.
  The inspected GPU timing path starts from embedded tensors and omits final
  norm/head. It cannot supply a complete token-boundary specification by itself.
- **FuseFSS:** nearest encoding remains distinct from SIGMA floor encoding.
  The evaluated tie rule, per-edge widths/scales and descriptor bytes still need
  discovery. A named development descriptor does not establish evaluated identity.
- **BumbleBee:** the final paper's ring64/Q18, six-squaring profile differs from
  the pinned example's Q16/four-squaring configuration. The mixed GELU interval
  is ±2.7 in the paper and ±3 in the inspected code. Recover evaluated settings
  or obtain approval for a complete paper-derived profile before those kernels.
- **PUMA:** the inspected source supplies candidate cast, `TruncAPr`, Goldschmidt
  and inverse-square-root rules. It clips shifted exponentials before summation,
  unlike literal Algorithm 2. The normalization variance/epsilon discrepancy
  remains substantive. These source-derived candidates need explicit resolution;
  coefficient discovery alone does not qualify the full numerical graph.
- **Feedback:** the inspected PUMA/BumbleBee generation examples recompute the
  growing prefix. Persistent protected KV is a separate proposed extension,
  not a feature established by those loops.

### Required PLLM work and honest reuse

- **Native:** replicated ring64/Boolean tensor types, three-way share transport,
  setup randomness, multiplication/square/truncation, Boolean–arithmetic product,
  equality/order/max, reciprocal/rsqrt, source GeLU/softmax, embedding and head.
- **Compiler/runtime:** preserve the replicated-share invariant at every layer;
  source-family graph import and Q18 interpreter; explicit scale/bound tracking;
  owner→three-party input/model sharing; client output reconstruction; bounded
  serialization of large tensors. P §5.4, p.9 specifically requires chunking
  structures beyond the serializers' 2-GB limit. No checkpoint/KV share may be
  reconstructed at an unauthorized worker to reuse a clear kernel.
- **Reuse:** generic source identities, graph metadata, framing, lifecycle
  accounting, and numerical-oracle infrastructure may be reused after supporting
  three parties and ring64. There is no PUMA-specific executable core in the
  inspected inventory. `PumaThreePartyDecoder` and `PumaSharedKvState` are
  pending [registry declarations](../../python/pllm/components/planned_methods.json). The similarly named
  [secret_replication.rs](../../crates/pllm-garble/src/secret_replication.rs) is a
  bounded **two-party PCF/VOLE** reference over small fields, not PUMA's replicated
  sharing. Existing W8A8/ring32 linears and clear KV are not substitutes.

**Minimum complete-model slice:** original GPT2-base, all 12 blocks, 32-token
input, complete shared embedding/body/head, client-authorized output. Then a
two-output feedback case, consistent with P's generation coverage. Gate this on
the truncation, softmax, normalization, and output-selection resolutions above.
Require protected intermediate tensors and all three actual online roles.

**Oracles/admission:** validate reconstruction and replicated-pair consistency,
all multiplication cross terms, signed truncation/carry cases, zero-sum randomness,
comparison boundaries, and no-truncation Boolean embedding. Compare the complete
candidate to its independently frozen interpreter; evaluate GLUE/perplexity and
generation separately from exact tensor checks. Source CPU testbed uses three
32-vCPU/128-GB hosts; LLaMA uses three 128-thread/1-TB hosts (P pp.6–7,9).
Uncompressed ring64 model storage alone is `2·8·P` bytes **per computing party**
for `P` parameters, before copies, state, scratch, or serialization. This is an
admission formula, not measured RSS or proof that all such storage must be resident.

## 7. Approval questions and faithful defaults

Request approval **before** implementing any departure below. A source ambiguity
can instead be resolved by additional pinned documentary/static source evidence;
never execute or link upstream implementations as the resolution method.

| ID / scope | Concrete question for the user | Recommended faithful default |
| --- | --- | --- |
| A1 — common model | May we add a separately labeled Qwen/Llama adaptation, replacing original checkpoints/tokenizers, dimensions, embedding vocabulary, positional encoding, biases, norm/MLP definitions, attention grouping, and possibly Q/K normalization? Which exact checkpoint/revision? | First complete each original GPT2 workload. For later Llama work, retain each source generation: SIGMA Llama2; FuseFSS includes Llama-3.1; BumbleBee/PUMA original LLaMA. Do not assume one family name means one graph. |
| A2 — precision | May we replace source fixed-point arithmetic with PLLM W8A8/ring32, change fractional bits, rounding, overflow, LUT/polynomial, or truncation order? If needed for modern-model ranges, may we widen rings and rederive all gap proofs? | Preserve each paper's separate numeric identity. No silent Q24 client bridge, ties-even substitution, calibration from private prompts, or rescaled approximation tails. Widening is an approved new profile. |
| A3 — unresolved numerics | For PUMA, will you authorize a documented correction/choice for Alg.2 versus Eq.(5), encoded epsilon, normalization, and underlying truncation/reciprocal? For BumbleBee, which activation endpoint/offset semantics, mixed profile, ring-extension bound, H2A error policy, and normalization? For FuseFSS, which tie rule and complete per-model descriptor manifest? | Resolve against pinned source evidence first. Otherwise keep the complete method unqualified; any authorized correction gets a new explicit identity. Candidate SiLU coefficients are pinned; evaluated-build association, encoding, lowering, and LLaMA graph remain unqualified. |
| A4 — SIGMA/FuseFSS boundaries | May we complete under-specified token input/output boundaries with protected full-vocabulary lookup/head composition using their existing arithmetic protocols? What result may the client receive: task result, token, probabilities, or full logits? | Qualify original boundary semantics first. Charge every added operation and preserve private-model SIGMA versus public-model FuseFSS assumptions. |
| A5 — decode/state | May we extend the fixed-length workloads with persistent protected KV, incremental attention, protected append/read and position handling, and cancellation/restart semantics? May the client receive state? | Keep state in each method's protected representation. Start with source workload; declare a complete-request extension before claiming serving coverage. Never reuse consumed masks/triples after restart. |
| A6 — sampling | May we replace BumbleBee's top-1 output with full-logit disclosure or stochastic sampling? For other methods, which fully specified selection/tie/RNG/EOS policy resolves missing details? | BumbleBee protected top-1. Freeze an authorized policy for the others; a client sampler must not receive additional outputs merely because PLLM expects them. |
| A7 — device | For FuseFSS, may we build a CPU backend preserving compiler, precision, protocols, and material accounting instead of the evaluated Blackwell GPU path? For SIGMA, which published CPU/GPU schedule should govern? | SIGMA's published CPU branch is faithful; BumbleBee and PUMA already evaluate on CPUs. FuseFSS defaults to source GPU execution. CPU availability does not justify replacing protected computation with clear kernels or claiming GPU-equivalent performance. |
| A8 — roles/setup | May we change private models to public weights, move client computation to two servers, colocate the trusted dealer with an online party, replace distributed OT with local issuance, or reduce PUMA to two online roles? | No topology/security substitutions. Keep SIGMA dealer + two owners, FuseFSS client + two servers + dealer, BumbleBee client/server 2PC, and PUMA three online share holders. |
| A9 — cryptographic backend | May we retain PLLM's different DPF seed format, bounded HE parameters, or replacement OT/conversion backend rather than reproduce the source construction/parameters? | Source construction first, independently implemented. Existing bounded algebra can seed development, but backend substitutions need explicit security, numerical, and cost qualification. |
| A10 — admission | If the original model/device/material budget does not fit, may we use different public tile sizes or a separate reduced-workload/profile for development? | Reject before allocation and report unresolved admission. A smaller model, one layer, lower precision, omitted setup, or client-local fallback cannot fill a complete source-system result. |

**Recommended order:** qualify missing numeric/boundary details; implement the
complete original GPT2 slices; validate source-specific semantics and privacy
boundaries; then seek approval for common-model and complete-serving extensions.
All adaptation and performance claims remain absent until their own evidence exists.
