# Shared-computation reimplementation contracts

Review: 17 October 2026. Scope: SIGMA, FuseFSS, BumbleBee, and PUMA.

**None currently has a qualified, complete independent PLLM implementation.**
This document specifies remaining work and the scoped author approvals in
[D6–D9](DECISIONS.md#d6-approved-puma-source-derived-q18-profile), dated
17 October 2026. Approval of named profiles does not authorize other protocol,
numeric, state, security or execution substitutions. Missing evaluated source
details remain unknown; no method is promoted to complete status.

Evidence: primary PDFs, [coverage audit](../TEN-METHOD-COVERAGE.md), component cards,
and static inspection of the local paths below. The initial contract review was
document-only. The later companion's scalar review added bounded independent
tests without a native/model/benchmark run or upstream execution/linking.
The subsequent [bounded PUMA truncation/PRSS reference](../../docs/research/puma-truncation-reference.md)
records its separate native evidence. The present decision update changes
documentation only and preserves all historical measurements.

The independent [source-fidelity audit](../reproduction-contracts/shared-computation.md)
adds immutable author-source inspections and concrete paper/code discrepancies.
It preserves its original canonical snapshot by hash. The refinements below
incorporate those findings; recommendations remain distinct from approval.

The subsequent [scalar source-profile resolution](../reproduction-contracts/shared-profile-resolution.md)
supplies explicit equations, source pins, table hashes, exact candidate HE primes,
and independently written bounded arithmetic tests. Those scalar tests were
executed; that review did not execute a native model or protected protocol.
It closes several source-detail questions without qualifying an evaluated build.
Its decision items now have explicit outcomes: D6 closes P-N1 adoption for
`puma.source-derived-q18.v1`; D8 closes F-N1 adoption for
`fusefss.source-derived-ring50-q12.v1`; D9 closes T-N1's feedback/tie/boundary
choice. D7 retains BumbleBee's final-paper target, leaving B-N1 evaluated-manifest
mapping and HE security open. Implementation, range, transport, SDK and quality
gates remain. D1/D3 already approve Apple-M CPU-first and the separate common-Qwen track.

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
   implement D9's approved fixed-count full-prefix feedback for these four
   methods, and bind protected state ownership, token feedback,
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

**Closed scalar specifications:** the companion §2 gives full Q12 CPU
GELU/SiLU/nExp table equations and hashes, inverse indexing, and integer-only
generation of all 8,192 norm entries for arbitrary public width. Fig.12's
`2f+log2(k)` becomes an exact rational factor k in that table; it requires no
rounded logarithm. Preserve its approximate mean coefficient `floor(4096/k)`:
at width 768 it is 5. A constant real-one row then normalizes to real ones,
whereas width four normalizes to zero. This source-defined counterexample is
tested; substituting exact division is a numeric change. The pinned CPU source
confirms that mean coefficient but omits the paper's displayed `s+1` guard in
the inspected norm path. The paper-defined profile retains the guard. These
details permit native scalar/gate-schedule work before token-boundary completion.

**Source gap and approved completion:** S §3.1.1/Fig.1, p.4 describes embedding and the final
linear/softmax path but does not give a token-ID lookup protocol or a complete
token-output/sampling API. D9 approves explicit protected equality→one-hot
embedding, final normalization and complete head, charged in full, with fixed-count
full-prefix feedback, lowest-index protected argmax and final client-only IDs.
Their implementation/qualification is still required. Persistent KV and
cross-request continuation remain unspecified and unapproved by this completion.

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
result disclosure under D9. Complete boundary implementation/qualification remains
a prerequisite. The
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
   rounding, LUT/coefficient, or rescaling manifest. D8 selects nearest-even
   external encoding for the named source-derived ring50/Q12 profile, while
   preserving SIGMA's floor-defined internal tables/rescales. This closes no
   evaluated-manifest gap and does not equate the two encoding rules.
- **Compiler algebra is independently implementable now:** the companion §5
  freezes Protocol 3's first-eligible-interval padding, full-domain/sentinel
  handling, mask rewrites, and original-x Horner evaluation. Its all-partition,
   all-mask small-ring oracle needs no guessed float tie policy. D8 closes F-N1's
   choice for `fusefss.source-derived-ring50-q12.v1`; exact per-gate descriptors,
   secure backends/Phi and full graph still require implementation and validation.
   The named interpretation does not establish evaluated-source identity.
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

The bounded [D8 GELU descriptor/compiler reference](../reproduction-contracts/fusefss/README.md)
freezes typed scalar semantics, source hashes and independent conformance vectors.
It is a non-production reference; protected backends and complete-method gates remain open.

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

**Minimum complete-model slice:** original GPT-2 at 128 tokens, using D8's approved
named numerical profile and the source two-server/dealer topology. Freeze and validate
its full descriptor manifest. Compile all
applicable activation, nExp, reciprocal/rsqrt, and rescaling gates through the
general compiler; execute embedding, complete body/head, and vector reductions.
Pair it with an independent unfused interpreter/protocol using exactly the same
descriptor semantics. Implement and charge D9's embedding/final norm/head and
fixed-count private feedback; boundary/output and lifecycle qualification remain
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

**Final-paper association gaps:** the pinned candidate now specifies Lift ties,
truncation, reciprocal/rsqrt, approximate endpoints and the extension offset/range
below. Their association with the evaluated final-paper profile remains missing,
as does its normalization divisor/epsilon. Eq.(2), p.8 and
Alg.3, p.9 also need reconciliation: Alg.3 adds `ε` to polynomial pieces but
does not show the final subtraction needed for Eq.(2)'s tails. Its inequality
tests and interval annotations disagree at endpoints. B p.4 writes the
normalization denominator using a sum of squared deviations, not an explicitly
averaged variance. Checkpoint-equivalent treatment must be resolved, not silently
corrected. Source's approximate HE conversion also precludes promising exact
ring equality for every complete execution without specifying its error model.

**New static resolutions and conflicts:** companion §4 derives the actual pinned
extension offset `M=2^(width−2)` and sufficient signed range `[-M,M)`, plus the
share-dependent dropped-bit comparison and truncation laws. At a comparison
pivot, uniform share splits can yield less-than with probability `1−2^-d` when
d low bits are locally dropped; deterministic endpoint tests are not faithful.
Pinned modulus-up is nearest/ties-up on unsigned representatives; modulus-down
by an odd modulus cannot encounter exact half ties. The candidate encoder casts
toward zero, conflicting with the paper's floor encoding. Candidate SiLU is also
a degree-four sigmoid-based construction, not the final-paper polynomial.
Ordered candidate HE primes are now derived independently from the source
factory: OLT uses nominal 218-bit key/163-bit data contexts, and enabled-error
bOLE uses a 159-bit ciphertext modulus and three 47-bit plaintext primes.
These do not resolve the final-paper 147/148-bit profiles by equivalence. D7
retains the final-paper target and does not adopt the Q16 artifact. B-N1's
enumerated evaluated-final-paper numeric/HE mapping and HE security remain open.
Pure packing/conversion
algebra can proceed; protected whole-model numeric identity remains unqualified.

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
argmax. D9 selects lowest-index ties, fixed-count full-prefix feedback and final
client-only token IDs; full logits are not approved. Resolve B-N1's numeric/HE
mapping and implement/validate the complete transaction before acceptance.

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

### Numerical contract, approved profile and unresolved source identity

**D6 selects `puma.source-derived-q18.v1`**, exactly the companion's §3.3
interpretation. P-N1's approval gate is closed for that profile, including its
fixed constants/order and corrected formulas. This is not a bit-identical claim
about the unavailable evaluated graph. Operator/graph implementation, range
proofs, transport, SDK and quality remain open.

- **Base arithmetic:** ring64/Q18 (P §5, p.6). Protected `Trunc_f` follows
  fixed-point multiplication/square and is implicit in their notation (P p.4).
  The PDF does not fix input rounding, the exact truncation distribution/failure
  bound, or reciprocal/rsqrt algorithms: these are black-box dependencies citing
  ABY3, Lu et al., and MP-SPDZ (P pp.3–4). Resolve the underlying constructions
   and bounds before claiming original evaluated numerical identity; D6's named
   profile uses the resolved pinned construction below.
- **Candidate truncation law resolved:** companion §3 derives pinned `TruncAPr`
  exactly as `floor(X/2^b)+[(X mod 2^b)+(R mod 2^b)>=2^b]` for a fresh uniform
  mask R and `X in [-2^(ell−2),2^(ell−2))`. Its probability of rounding up is
  `(X mod 2^b)/2^b`; exact multiples are exact. It is not deterministic floor.
  Exhaustive small-ring tests establish the arithmetic identity. The companion
  also specifies two-iteration reciprocal, rsqrt power/compensation order,
  binary32-derived Q18 constants and reproducible test tapes. Evaluated-build
   association remains unestablished. D6 fixes the stated scalar placement/order;
   complete graph lowering and per-product range proofs remain gates.
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
   shifted input. D6 approves §3.3's shifted clipping before summation and five
   protected squares for the named profile; the printed conflict remains evidence.
- **Epsilon/range:** suggested `10^-6` is below one Q18 unit (`2^-18`), so it
  need not survive encoding. Do not claim strictly negative exponent inputs until
   the encoded epsilon is fixed. D6 explicitly selects encoded-zero exponential
   epsilon. Preserve secure max, clipping, and normalization;
  do not substitute a public shift or narrow checkpoint-dependent score range.
- **Embedding:** compute Boolean one-hot shares via `Eq(i,id)` for every vocabulary
  entry, then Boolean–arithmetic multiplication with shared embeddings. No
  fixed-point truncation is needed for the Boolean selection. Client-generated
  one-hot queries or DPF lookup change this protocol (P §4.4, p.6).
- **Normalization:** Alg.3, p.7 uses shared mean, sum of squared deviations,
  one rsqrt, broadcast multiplication, and learned affine terms. Printed
  `σ=Σ(x−μ)²` lacks the usual variance division by width and explicit epsilon.
   D6 explicitly selects §3.3's centered mean-square variance and encoded epsilon
   two. This approved interpretation does not resolve the evaluated semantics.
- **LLaMA:** App.B, p.12 directs readers outside the PDF for the SiLU polynomial
  coefficients. The independent audit locates them in the cited example at a
  contemporaneous pinned SPU revision. Its zero/identity tails and multiplication
  order are explicit. Association with the evaluated build, encoding, compiler
   lowering and RoPE/RMSNorm still require qualification; D9 now fixes selection
   and feedback for the named transaction. Borrowing
  BumbleBee's different tails or SIGMA LUTs would create an adaptation. GPT2
  remains the better first slice.

### Additional author-source qualification

The independent audit records exact commits, paths and hashes for these findings:

- **SIGMA:** original CPU and GPU schedules are both valid source targets.
  The inspected GPU timing path starts from embedded tensors and omits final
  norm/head. It cannot supply a complete token-boundary specification by itself.
- **FuseFSS:** nearest encoding remains distinct from SIGMA floor encoding.
  The evaluated tie rule, per-edge widths/scales and descriptor bytes still need
  discovery for original-artifact fidelity. D8's named nearest-even external /
  floor-internal profile is approved; its descriptors still need implementation.
- **BumbleBee:** the final paper's ring64/Q18, six-squaring profile differs from
  the pinned example's Q16/four-squaring configuration. The mixed GELU interval
  is ±2.7 in the paper and ±3 in the inspected code. D7 retains the final-paper
  target; evaluated settings, source mapping and HE security remain blockers.
- **PUMA:** the inspected source supplies candidate cast, `TruncAPr`, Goldschmidt
  and inverse-square-root rules. It clips shifted exponentials before summation,
  unlike literal Algorithm 2. The normalization variance/epsilon discrepancy
  remains substantive evidence. D6 now resolves the named interpretation;
  coefficient discovery/approval alone does not qualify the full numerical graph. More
  precisely, shifted clipping/five squares is evidenced by the Llama example;
  the pinned generic GPT-2 example has no replacement hooks and defaults to
  eight exp iterations. It cannot certify PUMA's evaluated GPT-2 graph. P-N1 in
  the companion now gives the approved named Q18 scalar graph, corrected
  centered mean-square norm with encoded epsilon two, and exact consequences
  versus literal Algorithms 2–3. The correction is approved for D6's profile,
  not certified as the unavailable evaluated graph.
- **Feedback:** the inspected PUMA/BumbleBee generation examples recompute the
  growing prefix. Persistent protected KV is a separate proposed extension,
  not a feature established by those loops. They execute a public fixed number
  of private greedy selections, append IDs privately, and reveal the final ID
  array only after the loop, without EOS-dependent termination. D9 closes T-N1
  with lowest-index protected argmax, final client-only IDs, and explicitly
  added/charged SIGMA/FuseFSS embedding, final normalization and complete head.
  Protected execution and lifecycle/quality/cost qualification remain open.

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
  accounting, and numerical-oracle infrastructure support the bounded
  [PUMA SDK composition](../../docs/research/puma-truncation-reference.md#bounded-sdk-composition).
  `PumaThreePartyDecoder`, `ThreePartyReplicatedRoles` and `ReplicatedPrivatePrefix`
  select three authenticated workers and D9 full-prefix recomputation through
  ordinary Experiments. Generated small decoders exercise private lookup, body,
  head, argmax and feedback; the shared full-size checkpoint exceeds admission.
  `PumaSharedKvState` remains a pending persistent-state declaration. The similarly named
  [secret_replication.rs](../../crates/pllm-garble/src/secret_replication.rs) is a
  bounded **two-party PCF/VOLE** reference over small fields, not PUMA's replicated
  sharing. Existing W8A8/ring32 linears and clear KV are not substitutes.

**Minimum complete-model slice:** original GPT2-base, all 12 blocks, 32-token
input, complete shared embedding/body/head under D6 and D9. Then a two-output
fixed-count full-prefix feedback case with final client-only IDs. Choice gates
are resolved; require implementation of the approved scalar graph, per-product
range proofs, complete boundary/lifecycle admission and independent quality checks.
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

[D11](DECISIONS.md#d11-explicit-public-weight-puma-adaptation) additionally approves
an explicit public-weight adaptation. Its file-backed public coefficients and
batched protected token boundaries retain a separate composition/numeric identity;
the private-weight source method above remains distinct. The
[installed bounded contract](../../docs/research/puma-truncation-reference.md#explicit-public-weight-adaptation)
records current source, compiler, range and full-checkpoint gates.

## 7. Resolved decisions and remaining adaptation questions

D6–D9 close only the choices explicitly marked resolved below. Request approval
**before** implementing any additional departure. A source ambiguity
can instead be resolved by additional pinned documentary/static source evidence;
never execute or link upstream implementations as the resolution method.

| ID / scope | Concrete question for the user | Recommended faithful default |
| --- | --- | --- |
| A1 — common model | D3 already approves the pinned separate Qwen2.5-0.5B-Instruct track. Its numerical/state mappings still need qualification. | Preserve original-model validation and each source generation: SIGMA Llama2; FuseFSS includes Llama-3.1; BumbleBee/PUMA original LLaMA. Do not assume one family name means one graph. |
| A2 — precision | May we replace source fixed-point arithmetic with PLLM W8A8/ring32, change fractional bits, rounding, overflow, LUT/polynomial, or truncation order? If needed for modern-model ranges, may we widen rings and rederive all gap proofs? | Preserve each paper's separate numeric identity and D6/D8's explicit choices. No silent Q24 client bridge, ties-even substitution, calibration from private prompts, or rescaled approximation tails. Any widening requires a separately approved profile. |
| A3 — numeric profiles | D6/D8 close P-N1/F-N1 adoption for their named profiles. D7 retains BumbleBee final-paper target; B-N1 evaluated numeric/HE mapping remains open. | Preserve exact approved rounding/constants/order; source unknowns remain unknown. Named interpretations do not establish evaluated-artifact fidelity or complete-method quality. |
| A4 — SIGMA/FuseFSS boundaries | D9 approves protected equality→one-hot embedding, final normalization and full vocabulary head, with final client-only token IDs. | Implement and charge every added operation; preserve private-model SIGMA versus public-model FuseFSS roles and weight privacy. Boundary approval is not execution evidence. |
| A5 — decode/state | May we extend the fixed-length workloads with persistent protected KV, incremental attention, protected append/read and position handling, and cancellation/restart semantics? May the client receive state? | Keep state in each method's protected representation. Start with source workload; declare a complete-request extension before claiming serving coverage. Never reuse consumed masks/triples after restart. |
| A6 — sampling/feedback | D9 closes T-N1: fixed public count, full-prefix recomputation, lowest-index protected argmax, private IDs throughout, final client-only ID reveal. | No streaming/logits/stochastic-sampling/EOS-short-circuit approval. Preserve method roles and fresh per-step material; implement and test actual feedback/cancellation. |
| A7 — device | D1 already approves Apple-M CPU-first execution, including explicitly labeled CPU ports. | Keep SIGMA's source CPU schedule; preserve FuseFSS compiler, precision, protocols and material accounting in its CPU port. GPU source timings remain separate. BumbleBee and PUMA already evaluate on CPUs. |
| A8 — roles/setup | D11 approves a separate public-weight PUMA mode alongside the private-weight method. Moving client computation to two servers, dealer colocation, local OT issuance or fewer PUMA parties still requires approval. | Keep the two PUMA weight contracts distinct and retain three online share holders. SIGMA dealer + two owners, FuseFSS client + two servers + dealer and BumbleBee client/server 2PC retain their source contracts. |
| A9 — cryptographic backend | May we retain PLLM's different DPF seed format, bounded HE parameters, or replacement OT/conversion backend rather than reproduce the source construction/parameters? | Source construction first, independently implemented. Existing bounded algebra can seed development, but backend substitutions need explicit security, numerical, and cost qualification. |
| A10 — admission | If the original model/device/material budget does not fit, may we use different public tile sizes or a separate reduced-workload/profile for development? | Reject before allocation and report unresolved admission. A smaller model, one layer, lower precision, omitted setup, or client-local fallback cannot fill a complete source-system result. |

**Recommended order:** implement the approved named numeric graphs and D9
boundaries with range/admission proofs; resolve BumbleBee's final-paper mapping;
qualify complete original GPT2 slices through D5, then the already approved D3
common-model track. Further state/protocol changes still need their own approval.
Correctness, quality and performance claims require their own evidence.
