# Independent source-fidelity audit: encrypted inference

Audit date: 17 October 2026. Scope: NEXUS, THOR, CacheMir, ROSETTA.

**Verdict: the canonical draft preserves the important source distinctions, but none of the four methods is implementation-ready.** Its existing gates are warranted. Independent inspection adds concrete paper/code conflicts, an invalid boundary case in CacheMir’s printed pruning rule, a GPT-2 context-length qualification, and a more precise primitive-backend finding for ROSETTA.

**Canonical document:** [encrypted-computation-contracts.md](../methods/encrypted-computation-contracts.md), reviewed SHA-256 `f9e4b5d4490eeb2d6891f191650bead306a84efe966d4259ee3acac9c6178ecd`, 541 lines. Section and line references below refer to those bytes. This file is an independent audit and decision report, not another implementation contract. The canonical file was not edited. No implementation, model, build, paper-code execution, or benchmarking job was run.

## 1. Evidence identity and audit boundary

Page numbers are one-based **PDF pages**. These local hashes were recomputed and agree with [prior-art-sources.json](../prior-art-sources.json) and [TEN-METHOD-COVERAGE.md](../TEN-METHOD-COVERAGE.md).

| ID | Pinned primary source | SHA-256 / version boundary |
| --- | --- | --- |
| N | [NEXUS, NDSS 2025](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf); [local PDF](../../papers/r21-nexus.pdf), 17 pages | `c809182345f27f76e21d983d97dffa0d086dbcd17dabc265f0df9fb6ca9b70f2`; final NDSS paper |
| T | [THOR, ePrint 2024/1881](https://eprint.iacr.org/2024/1881); [local PDF](../../papers/thor.pdf), 15 pages | `ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`; cached September 2025 bytes, not independently proved identical to the current ePrint revision |
| C | [CacheMir, arXiv:2602.11470v1](https://arxiv.org/abs/2602.11470v1); [local PDF](../../papers/cachemir.pdf), 16 pages | `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`; v1, 12 February 2026 |
| R | [ROSETTA, arXiv:2609.16915v1](https://arxiv.org/abs/2609.16915v1); [local PDF](../../papers/rosetta.pdf), 15 pages | `b4c7195d32808881b2da8b3a6876b571d8215bbc02aaafaa78ef327aa27fb4e8`; v1, 15 September 2026; embedded CCS ’26 matter does not prove proceedings-byte identity |

Static author-source and primitive-library inspections supplement the PDFs; immutable commits and selected-file hashes appear in §8. A repository version can clarify or contradict a paper but does not silently supersede it. Inspected files were fetched as text only, never imported or executed. Source-dependent findings are limited to the inspected functions, not artifact-wide correctness claims.

C Appendix A, p.15 links [Cachemir-B254](https://anonymous.4open.science/r/Cachemir-B254); R Appendix A, p.15 links [ROSETTA_artifact-B77F](https://anonymous.4open.science/r/ROSETTA_artifact-B77F/). Direct README-page retrieval returned the anonymous-site application shell, not source bytes. Their exact evaluated commits, numeric constants and backend customizations remain unverified here. This is a review gap, not evidence that artifacts do not exist.

## 2. Findings and release-blocking decisions

Severity denotes what is blocked: **method** = claiming the intended complete reproduction; **profile** = selecting/implementing the disputed numerical, protocol or execution variant; **evidence** = transferring a published result. “Confirmed gate” means the canonical draft already identifies the problem. No approval is granted by this audit.

### F1 — NEXUS paper and pinned author implementation disagree materially

**Severity:** profile. **Canonical:** §§4.2–4.3, G0; new static evidence.

`zju-abclab/NEXUS` commit `e15058c7f8f58fcf037c4709211109cb1be7d1e0` is dated 20 September 2024, before the final NDSS paper. Therefore the following differences need an explicit source-version resolution:

| Paper requirement | Inspected author-source observation | Required action |
| --- | --- | --- |
| N p.8, unnumbered formula after Eq.(4): `EXP(x)≈(1+x/2^r)^(2^r)`, `r=8` | `src/ckks_evaluator.cpp:534–547`: initial `x/128`, then seven squarings, yielding `(1+x/128)^128` | Preserve power 256 as the pinned-paper default. A 128-power code-compatible profile is a numerical deviation needing approval, not a silent correction. |
| N pp.9–10: ring degree `2^16`, 1763-bit modulus, 128-bit target; p.12 scale `2^50` | `src/bootstrapping.cpp:33–75`: degree `2^15`, scale `2^46`, 16 remaining levels plus 14 bootstrap levels, Hamming weight 192, `sec_level_type::none`; main operator harness also has several profiles and disables the library check | Recover exact paper-context primes/key graph independently. Do not adopt the test harness’s context as a certified published profile. Disabled checking is not by itself proof of insecurity. |
| N Algorithm 4/Eq.(5), p.9: subtract mean, normalize, apply gamma/beta | `src/layer_norm.cpp:6–32` squares/sums supplied input, divides by 768, applies inverse square root; this function does not center or apply gamma/beta | Trace intended caller-side preprocessing before treating this helper as the complete LayerNorm specification. |
| Goldschmidt reciprocal needs an admitted denominator domain | `src/softmax.cpp:23–34` scales denominator by `0.01` and restores the factor | Freeze the actual denominator bounds. `0.01` is not a universal range proof for an arbitrary model/context. |

The README requires modified `thirdparty/SEAL-4.1-bs`. The canonical draft correctly says stock SEAL is insufficient; an apparent “SEAL backend” match must not conceal reuse of the paper’s custom implementation.

**Exit evidence:** a source-resolution record selecting the mathematical identity, exact context and coefficient/schedule tables; separate named identity for any approved code-compatible profile. Static source reading may resolve missing constants, but copying or linking its inference implementation remains excluded.

### F2 — NEXUS dual-key preprocessing is essential; sustained autoregression is unproved

**Severity:** method. **Canonical:** §§4.1, 4.3–4.4; confirmed gate with quantitative checks.

N §III.C, p.6 and Appendix B, p.16 require two secret-key domains. Server initially sends compressed weights under its own key; client decompresses, forms encrypted `UW`, and adds its key’s RLWE term. In the paper’s convention:

```text
(A, A*s_server + m + e)
  → (A, A*s_server + A*s_client + m + e + e')
  → (A, A*s_client + m + e + e')
```

Only the server’s layer is removed. The client’s key never goes to the server. This is neither ordinary double encryption nor a free generic key switch. Online `A'-U` plus `Enc_client(UW)` yields the desired product. N Appendix D, p.16 specifically relies on **uniform plaintext-ring one-time pads**; finite bounded real masks do not inherit that proof. Mask freshness, scale/ring encoding, client decompression, encrypted-weight delivery and preprocessing retirement are mandatory.

The construction assumes client-known `A'`; its application to subsequent encrypted hidden layers requires a separate derivation. Revealing hidden activations to the client would change the non-interactive protocol. The canonical draft correctly retains the unresolved column/diagonal/coefficient layout chain and attention-value product.

N Table IV, p.12 establishes one-word BERT and Llama-3-8B workloads, **not** an implemented private multi-token loop or persistent KV service. Llama rows use GELU/LayerNorm labels and full-width projections; genuine Llama RMSNorm/SwiGLU/RoPE/GQA mapping is not established by that table. Initial embedding custody, output-to-input lookup, recurrent levels, capacity and stopping need source clarification or an approved extension.

Algorithm 2, p.7 returns encrypted **selection vectors**, nominally multi-hot on exact ties. Its `2m<S` precondition fails for both source vocabularies with `S=32768`: BERT `m=30522`; Llama `m=128256`. Full-vocabulary cross-ciphertext folding/selection is therefore mandatory; no reduced-vocabulary proxy qualifies. Returning full logits for arbitrary client sampling changes output disclosure.

**Exit evidence:** dual-key ownership/mask proof; complete encrypted layout/head mapping; full-vocabulary selection/tie specification; explicit decision on any private token-feedback extension. Charge two-key setup and all masks even when amortized.

### F3 — THOR’s source bootstrap is opaque and its source numeric profiles refine the PDF

**Severity:** profile and method. **Canonical:** §§5.1–5.4, B3/A5; new static evidence; encoder-only conclusion confirmed.

`crypto-starlab/THOR` commit `b513c06aaabccc04b560721e35e9a98008428c25` (18 September 2025): README lines 1–9 put datasets, encoded/fine-tuned models and keys outside the repository; line 23 says **bootstrapping is PyArmor-protected and requires Python 3.10**. A bundled binary/obfuscated bootstrap is not an independently inspectable primitive specification or permitted shortcut to an independent PLLM implementation.

Notebook `forward.ipynb`, source cell 7, gives `logN=16`, `scale_bits=41`, four special primes and `quantum="pre_quantum"`. It does not bind the complete evaluated prime chain. T §7.1, pp.11–12 gives 13 usable levels and a 128-bit target but not that full chain or bootstrap precision. Copying CacheMir’s parameters because it cites THOR is unjustified.

`src/thor/nonlinear/softmax.py:7–10,122–178,180–231` uses two profiles (`[-27.2493,21.72692]`, `n=2,l=2`; `[-70,70]`, `n=2,l=4`), folds range/scale transformations into exponential coefficients and computes aSOR factors from public error bounds. `src/thor/bert.py:112` selects the second profile for layer index 2. `layernorm.py:5–12` supplies variance profiles `[0.15,10]`, `[0.2,150]`, `[0.75,2500]`, with scaling/adaptive inverse-square-root logic at lines 21–28,64–75,104–129.

Those profiles need reconciliation with the PDF’s MRPC score range `[-44.19,50.25]`, `delta0=11.805,delta1=2,delta2=4` (T §7.2.2, p.12), and rounded Appendix C coefficients/factor lists (p.15). They may be task/layer specializations or algebraically equivalent scaled fits; the evidence does not justify labeling every difference a bug. Equally, the printed repeated reciprocal factors must not be assumed equivalent to a bound-driven source recurrence without checking.

THOR natively evaluates **BERT encoder classification**: client embedding, 12 encoder blocks, pooler/Tanh/classifier, final client decryption; no evaluated KV/token-feedback loop. T §8, p.13 says the techniques may apply to GPT; this does not instantiate GPT-2. For GPT-2, required changes include causal attention, learned positions, pre-norm/final norm, its activation variant, head, recurrent state and feedback. RoPE/RMSNorm/SwiGLU/GQA are relevant to other decoder checkpoints, not inherent GPT-2 requirements.

**Exit evidence:** auditable approved primitive bootstrap, complete parameter graph, reconciled task/layer numeric registry, native BERT graph. A CPU port and a GPT-2 extension need separate execution/functionality approval.

### F4 — CacheMir’s printed pruning rule has a concrete infeasible-predecessor case

**Severity:** compiler profile. **Canonical:** §6.3, G2; new blocking algebraic finding.

C §6/Fig.7/Eq.(1), p.9 declares input feasibility `x≥ell(i)`, computation before bootstrap, and edge cost:

```text
w[i,x,y] = t_i(x) + 1[x-ell(i)<y] * t_boot(y)
```

The same page says to prune any edge with `x*(x-y-ell(i)) != 0`, comparing it to input level `x-1`. Take an internal layer with `ell(i)=1`, `x=1`, `y=2`. The original edge is feasible: compute to level zero, then refresh to two. The printed prune expression is nonzero, but the proposed replacement uses input zero, **below the layer’s required depth**. Therefore the printed rule cannot be transliterated across this boundary. The intended “bootstrap only when level reaches zero” prose supports keeping this compute-to-zero case.

There is also a cost-assumption qualification: C Table 9, p.16 reports plaintext-ciphertext multiplication **158.22 μs at level 7 versus 140.82 μs at level 10**, a nonmonotone measured entry. Monotone `t_i(x)` is a stated pruning assumption; do not assume raw empirical module/primitive data automatically satisfy it. This observation does not prove the authors’ complete layer cost table violates monotonicity.

**Action:** use the complete feasible DAG as the mathematical reference; derive a guarded/corrected dominance rule rather than guessing a replacement factor. Check residual subgraphs and periodic-block composition against exhaustive small DAGs under both monotone and nonmonotone costs. Keeping an unpruned graph is a useful correctness milestone, but reproducing the paper’s optimized compiler requires resolving the pruning algorithm.

C §7.2’s 234.24 versus 463.31 **milliseconds** is placement-algorithm runtime (1.98×), not whole-inference bootstrap speedup. The canonical draft does not make that mistake; retain the distinction.

### F5 — CacheMir’s K and V layouts are distinct; persistent refresh remains a method gate

**Severity:** method. **Canonical:** §§6.2,6.4; confirmed, with explicit address/boundary checks.

C §§4–5, pp.4–8 and Appendices C–D, p.15 support the canonical reconstruction:

* Interleaved replicated VMM uses spacing `t=S/d`, generalized diagonals and BSGS, with extraction fused into a subsequent real operation. Invalid slots must remain unable to enter valid reductions; nonlinear constant terms make this a substantive invariant.
* For one head, K ciphertext `j` stores `t` tokens; the logical address is `slot[f*t+r]=K[j*t+r,f]`, with `ceil(n'/t)` ciphertexts. Append changes the projection’s output alignment to fill an empty slot or allocate a tile, not decrypt/repack the cache.
* V uses the distributed layout compatible with compact scores and interleaved outputs: one-head `d` ciphertexts per `S` tokens; multi-head `d/H` per `S/H` tokens. A K layout relabeled V is wrong. Full token-to-slot mapping and wraparound must be derived and checked independently.
* With `S=32768,H=32`, score/V-group capacity is 1024 tokens. Beyond it, allocate/merge blocks without context truncation or separate per-block softmax normalization. Key-tile and score/V-group boundaries differ.
* Source prefill uses compatible encrypted layouts and is explicitly suboptimal (C §5.4, p.8). A clear supplied cache would change the protocol and cost boundary. Appendix D’s RoPE `±1` rotations still need reconciliation with physical interleaving stride.

The paper does not settle retained-ciphertext versus temporary modulus-drop semantics over many decodes. Destroying persistent levels on every read exhausts state; repeated refresh can accumulate error. Include key epoch, layout, live length, physical capacity, scale/level and pending token in the state identity; append K/V atomically. Session failure/replay/cancellation is a PLLM integration layer requiring validation, not a claimed original-paper feature.

C Appendix E, p.16 explicitly replaces THOR aSOR with **standard Goldschmidt** and recalibrates larger-model ranges. R §5.1, p.11 says CacheMir inherits THOR nonlinearities “without modification.” The canonical draft correctly flags this conflict. Bind the actual CacheMir and ROSETTA baseline variants independently.

**Exit evidence:** exact K/V address maps and model GQA mapping; persistent-level/error policy; encrypted prefill/append/head/feedback specification; independent comparisons at `t-1/t/t+1` and `S/H-1/S/H/S/H+1`, including old-token invariance. CacheMir’s tabled runtime is not evidence that this complete lifecycle ran.

### F6 — ROSETTA requires blind rotation of encrypted accumulators, not generic public-LUT PBS

**Severity:** backend and method. **Canonical:** §§7.2–7.4, B2/B5; confirmed requirement plus new API-level evidence.

R Algorithm 1, p.8, especially lines 11–23, constructs coefficients depending on encrypted `x`, selects a local interval via encrypted segment index, then performs **two stages of blind rotation over ciphertext accumulators**. Four segments yield 3 HomComp calls, 5 plaintext-ciphertext multiplications, 6 blind rotations and 5 traces, before conversion and HomComp’s internal PBS. A generic function `PBS(public_table, encrypted_index)` is not the same operation.

At Lattigo v6.1.1 commit `37a34b4676e987795949a795f0ce9d0893763c96`:

* `core/rgsw/blindrot/evaluator.go:49` exposes `Evaluate(ct, testPolyWithSlotIndex map[int]*ring.Poly, BRK)`: **plaintext** test polynomials.
* Lines 104–111 initialize a trivial accumulator from that public polynomial and set its second component to zero. Feeding ROSETTA’s encrypted candidate through this interface would discard its ciphertext semantics.
* **Useful lower-level primitive:** `BlindRotateCore(a []uint64, acc *rlwe.Ciphertext, BRK)`, line 133, accepts a ciphertext accumulator and performs automorphisms/external products (lines 165–179,202–210). This is concrete support for investigating a source-aligned independent composition, not proof of a complete drop-in ROSETTA bridge.
* The helper modulus switch rounds to `2N` and adjusts nonzero even coefficients to odd (lines 274–296). ROSETTA’s interval `floor` must not be equated with this helper’s rounding by assumption. Normalization, pre/post rotations, encrypted-accumulator noise, key domains and sign conventions need explicit derivation.

R §3.3, pp.7–8 requires StC → full/small ring switch → RLWE constant-term plus LWE extraction → lookup → reverse ring switch → **ModRaise + CtS + EvalMod**. Raw modulus raising is not recovery. The source full ring is 65536, blind-rotation ring 2048, LWE dimension 1024 (R §5.1, pp.10–11). The 2048→1024 secret/key relationship is not explained by coefficient extraction alone. All corresponding switching/trace/RGSW keys must remain client-generated with only evaluation material at the server.

**Exit evidence:** derive the encrypted-accumulator wrapper and exact key/encoding graph over qualified generic primitives; legal StC entry and recovery target levels; high-precision HomComp; interval floor/endpoints and complete error budget; multi-value sharing without head permutation. Test `tau_end`, where the naive formula yields index `N_br`, and both sides of every segment/interval boundary.

OpenFHE advertises CKKS bootstrap and CKKS↔FHEW/TFHE switching; that is not proof that its high-level bridge implements this protocol. Replacing ROSETTA with standard fixed public LUT evaluation requires approval as an algorithmic deviation and would not be the original method.

### F7 — Common GPT-2 is partly native, but the source cost table is not a native 8k model

**Severity:** functionality profile and evidence. **Canonical:** §§5.4,6.1,7.5,A3; new qualification.

| Method | GPT-2 status from pinned source | What remains before a common complete request |
| --- | --- | --- |
| NEXUS | GPT-2 mentioned in examples/vocabulary discussion; no complete evaluated GPT-2 decoder | New model mapping, causal attention, full head/selection, private feedback and state; original BERT/Llama one-word evidence retained |
| THOR | Encoder-only evaluated BERT; applicability discussion is not decoder execution | Approved GPT-2 functionality extension after native BERT reproduction |
| CacheMir | GPT2-base is native to the evaluated architecture/module-cost scope, C Table 3/p.10 | Exact checkpoint/tokenizer, source-compatible numeric fits, embeddings/head/output/feedback and real multi-step lifecycle |
| ROSETTA | GPT2-base native to decode/module-cost scope, R Table 7/p.12 | Same full-lifecycle gaps; qualify context geometry separately from model semantics |

R Table 7 lists 512, 2048 and 8192 contexts for GPT2-base. The pinned supplemental [GPT-2 configuration](https://huggingface.co/openai-community/gpt2/raw/607a30d783dfa663caf39e06633721c8d4cfcd7e/config.json), SHA-256 `0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4`, has `n_ctx=n_positions=1024`, `activation_function="gelu_new"`, epsilon `1e-5`, vocabulary 50257. This is an independent reference config, **not proof it was the authors’ checkpoint**. Thus the 2k/8k shape estimates cannot establish unchanged standard GPT-2 inference. Pin the authors’ actual configuration or treat extended positions as an explicit model adaptation.

R §5.4, p.13 gives **ciphertext-aligned plaintext** perplexity; C Table 7, p.12 evaluates an approximated model. Neither establishes encrypted full-model multi-token quality. R’s public-prefix calibration is also unpinned, and its normalization variance bound 8192 covers only approximately **72%** of profiled operations (§5.2, p.13). Adopting it as a universal domain is unsupported.

**Decision:** recommend a ≤1024-total-position standard GPT-2 common profile for C/R once source details are pinned; seek approval before N/T adaptations or any positional extension. Preserve broader-context cost-shape experiments as such. Do not change ranges, prefix, precision, clipping, or checkpoint without resolving the numerical identity.

### F8 — NEXUS’s sign error argument cannot be inherited as a certified bound

**Severity:** numeric qualification. **Canonical:** §4.3/G2; new primary-source inconsistency.

N §II.D, p.4 composes four degree-9 polynomials: `f(f(g(g(x))))`. Its formal polynomial degree is `9^4=6561`, absent degenerate cancellation, **not 36**. N §VI.E, p.12 explicitly calls it degree 36 and uses that in its relative-error argument. The canonical draft correctly says composition; it should not inherit the source’s resulting bound as a proven implementation tolerance.

**Action:** independently propagate approximation, CKKS, bootstrap and selector errors through the actual composition, including normalization and near-tie uncertainty. This is a verification/source-resolution issue, not approval to change the sign approximation. Empirical average error does not certify full-vocabulary Argmax or model privacy.

### F9 — Source hardware and amortization cannot stand in for complete measured requests

**Severity:** evidence. **Canonical:** §§4.5,5.4,6.4,7.5,G6; confirmed.

| Source | Original hardware/workload boundary verified | Required qualification |
| --- | --- | --- |
| N §§VI.A–D, pp.9–12 | Two 3.70 GHz Intel Xeon instances, 128 GB RAM, 32 threads; four A100-40GB; 10 runs. LAN 3 Gbps/0.8 ms; WAN 100/200 Mbps and 40/80 ms. BERT 128 tokens; Llama 8 tokens/one-word output. Figure 9 amortizes 128 inputs; Table IV batches 32. | Keep cohorts distinct. The 37.34-second BERT number is amortized source GPU latency, not unbatched admission-to-first-token. Include encrypted-weight/mask preparation and its reuse denominator. |
| T §7.1/Table 6, pp.11–12 | Single A100/Liberate.FHE; BERT-base 128 tokens; 602.26 seconds including 337.86 bootstrap seconds; MRPC/RTE/SST-2, batch-16/lr `5e-5`/five-epoch fine-tuning | GPU capacity and full host profile unpinned. This is not an autoregressive token rate. A CPU port is an execution adaptation. |
| C §7.1/Table 5, pp.10–11 | Xeon Platinum 8558P, max 4 GHz, 192 threads; one A100-80GB; Lattigo CPU/customized Phantom GPU; batch one. GPT2-base, TinyLlama-1.1B, Llama-3-8B; `(n0,n')` includes `(64,512),(128,256),(16,1024)` | End-to-end estimates aggregate measured modules; baselines use operation-count estimates. Table 5’s 3.02 GPU seconds is one Llama decoder block with amortized prefill, not whole-model token latency. Host RAM/exact backend commits unpinned. |
| R §5.1/Table 7, pp.10–12 | Xeon Gold 6240R 2.40 GHz, 48 threads, 384 GB; Go 1.24, Lattigo v6; customized Phantom/CUDA 12.4/A100-80GB; same three runtime model families | One decode iteration estimated from modules **excludes prefill**. Authors’ reimplemented baselines use identical crypto parameters, not necessarily original-system backends/profiles. Separate from encrypted whole-model quality and complete request cost. |

None supplies malicious-result proofs. Report proof generation/verification as **not part of the specified protocol**, rather than zero-cost verified computation. Include key generation, evaluation-key delivery, contexts, weight encoding, polynomial/LUT preparation, optimizer cost data, private preprocessing, prefill, decode, selection/feedback and cleanup. Account for host/device copies, runtime/GC and helpers. C’s ~17 GB encrypted Llama KV and ~694 GB encoded weights (§7.5/Fig.10, p.12) do not establish total RSS or simultaneous fit on an 80 GB GPU.

## 3. Numerical/security cross-check against canonical draft

These checks agree with the canonical reconstruction; they do **not** certify missing executable parameters.

| Profile | Verified pinned requirements | Still blocking qualification |
| --- | --- | --- |
| NEXUS | N pp.9–12: degree `65536`, slots `32768`, stated 1763-bit modulus/128-bit target; total depth 35, bootstrap 14, nominal 21 computation levels; `q0≈2^60`, others `≈2^50`, scale `≈2^50`, Gaussian `σ=3.2`, secret weight 192 | Exact `Q/P`, sparse-secret evaluation-key security, bootstrap algorithm and error profile; source-code conflicts F1; malformed `20→21` multiplication level row in Table IV |
| THOR | T §7.1: degree `65536`, slots `32768`, 13 usable levels, 128-bit target; diagonal/BSGS/lazy-rescale/lazy-relinearization/complexification kernels; source-range ASoftmax+aSOR, centered LayerNorm, composed GELU/Tanh | Exact primes/keys/bootstrap; task/layer-scaled coefficients F3; masks, epsilon, coefficient order and fit domains; client embedding and final-only decryption |
| CacheMir | C §7.1, p.10: degree `65536`, slots `32768`, stated 1763-bit modulus; **13 computation levels plus 15 bootstrap depth**; CtS/StC/EvalMod depths 4/3/8; `q0≈2^53`, others `≈2^41`; weight-192 sparse secret **with encapsulation** | Exact scale and prime/key graph; complete per-layer fits and standard Goldschmidt schedules; retained-cache error/level policy; pruning F4 |
| ROSETTA | R §5.1, pp.10–11: same reported CKKS profile; LWE dimension 1024, blind-rotation ring 2048, modulus 53 bits; four segments, logarithmic inverse/invsqrt boundaries and uniform SiLU/GELU boundaries | Error distributions, gadget bases, failure probabilities, all domain switches/keys, signed encoding, interval endpoints and recovery precision; encrypted accumulators F6 |

The canonical document appropriately distinguishes N/C/R semi-honest model/input privacy from T §6.1’s input-privacy statement. T requires decrypted results to stay with the key owner; debug output or feedback must not defeat that restriction. Semantic security does not independently establish circuit privacy of model-dependent returned ciphertexts; retain the canonical simulator/view and leakage qualification gate. None of these requirements is satisfied by PLLM’s current exact-W8A8/ring32 or narrow-domain no-bootstrap operators.

## 4. Backend decision: mature primitives permitted, paper implementation reuse excluded

**Concur:** full methods need a mature capability beyond stock SEAL. An independent implementation may use approved general-purpose crypto primitives. It must independently implement the papers’ packing, kernels, polynomials’ method-specific composition, cache updates, segmented lookup, compiler and orchestration. Linking, wrapping, invoking, copying or porting selected paper implementations does not satisfy the user’s independence requirement.

| Candidate | Evidence inspected | Audit recommendation |
| --- | --- | --- |
| Current SEAL/TenSEAL | Static PLLM CKKS wrapper: leveled arithmetic only; eager rescale/relinearize; no CKKS bootstrap/TFHE/conversion | Reuse ownership/framing/admission patterns, not as the complete backend. Increasing depth is not bootstrap. |
| Lattigo v6.1.1 / `37a34b4676e987795949a795f0ce9d0893763c96` | README documents RLWE/RGSW, blind rotation, DFT/polynomial/inverse/CKKS bootstrap/repacking; F6 inspects actual blind-rotation API | Best source-aligned CPU candidate for C/R. Qualify sparse encapsulation, key/ring conversion, encrypted accumulator wrapper and recovery before approval. This tag is not asserted to be the authors’ evaluated build. |
| OpenFHE v1.4.2 / `aa391988d354d4360f390f223a90e0d1b98839d7` | README documents CKKS bootstrap, DM/FHEW, CGGI/TFHE, LMKCDEY and CKKS↔FHEW/TFHE switching | Mature C++ alternative; qualify low-level encrypted-accumulator/ring/trace APIs and exact precision. Approval needed for source-backend/bootstrap substitutions; advertising a switch does not settle F6. |
| Generic Liberate.FHE / `b263100c830fda96be86776ff1b4fd6394e6a071` | General Python/CUDA RNS-CKKS upstream; THOR bundle differs and protects bootstrap | Generic upstream primitive reuse is conceptually separate from importing THOR. No auditable equivalence to its bundled bootstrap established. |
| Generic Phantom / other GPU backend | N/C/R name Phantom; C/R explicitly use customized versions | Required refresh/conversion/kernel availability remains unqualified. Source GPU hardware and custom costs cannot transfer to a CPU profile or generic unmodified package. |

Lattigo behind a native adapter/provider process or OpenFHE behind C++/Rust are substantial architecture/backend choices already covered by canonical B1–B5. Include cross-language copies, runtime/GC, serialization, device transfer, evaluation-key residency and key lifetime in qualification. Source-specific adapters may be necessary; one common backend is not intrinsically more faithful.

## 5. Actionable approval set

Keep user approval focused on substantial deviations. Missing exact constants or source ambiguities should first be researched; choosing source-faithful constants is not itself a new user-facing protocol decision.

| Decision | Source-supported options | Recommended disposition |
| --- | --- | --- |
| **D1 — Primitive backend/bootstrap** | Source-aligned generic backend plus independently implemented composition; or explicitly approved OpenFHE/Lattigo ports with different refresh/key/precision profiles | Approve a qualified mature primitive dependency, not paper packages. C/R Lattigo first; N/T bootstrap ports separately. Resolve F1/F3/F6 before implementing disputed profiles. |
| **D2 — CPU/GPU** | Reproduce source hardware arms; or CPU functional profiles followed by independently qualified GPU work | CPU first is reasonable, especially C/R. Explicitly classify THOR CPU as a port; preserve source GPU workload/parallelism/setup costs without inherited latency claims. |
| **D3 — Common GPT-2** | Native BERT/NEXUS forward reproductions plus named GPT-2 adaptations; or approved decoder-first adaptations while retaining native completion obligations | Native first. C/R GPT-2 is architecture/cost-native, not turnkey private generation. Use a pinned ≤1024-position standard profile unless positional extension is separately approved (F7). |
| **D4 — Numerical identity** | Pinned-paper expressions with reconciled constants; named code-compatible profiles or new fits/ranges/precision/public-prefix calibration | Resolve N 256-versus-128, THOR scaled profiles, C Goldschmidt and R baseline inheritance. Approve changed behavior before code; do not silently clamp or use current PLLM narrow-range approximations. |
| **D5 — Output and feedback** | Source-supported encrypted selection/head/embedding loop; or approved client-decrypted per-token result with encrypted next input | Preserve N selection-only disclosure, T embedding custody and C/R privacy assumptions. Sending IDs clear, returning full logits, intermediate decryption or clear KV are material changes. Essential source gaps remain unresolved. |
| **D6 — Whole-model/state/security additions** | Source clarification; or independently specified, approved completion for missing layout/head/cache/GQA/prefill details | Preserve actual checkpoint graph. CacheMir cache inside N/T is a named hybrid. Public-weight-only custody or added verification changes the comparison/security class; do not imply original equivalence. |

F4’s pruning derivation and F8’s corrected error analysis can be resolved without changing mathematical functionality. They require documented independent validation; an altered optimizer/cost claim or altered sign approximation would require profile approval.

## 6. Dependency order, reusable work, and independent exit gates

Concur with canonical §12, with **backend capability qualification before implementation of disputed profiles**:

1. Source/approval lock → exact parameter and key graph → public evaluator/client-secret ownership → real bootstrap and representation primitives.
2. In parallel only as a dependency diagram, not a request to spawn workers: NEXUS dual-key compression branch; THOR diagonal/lazy/numeric branch.
3. THOR-like primitive arithmetic supports CacheMir VMM/KV/numerics; CacheMir’s independent level DAG then supports ROSETTA’s joint scheme/level compiler. Preserve distinct approximation profiles.
4. ROSETTA raw ring/LWE/encrypted-accumulator/conversion branch joins qualified CacheMir-compatible state and the joint compiler.
5. Native full-model requests → approved common-workload adaptations → later quality/resource campaign. First slices do not replace any requested full reproduction.

**Reusable PLLM patterns, not already completed paper components:** [CKKS key custody](../../python/pllm/runtime/ckks_reference.py), [bounded framing](../../python/pllm/runtime/seal_frames.py), [layout admission](../../crates/pllm-core/src/ckks_layout.rs), [native binding](../../crates/pllm-python/src/ckks.rs), [execution-capsule provenance](../../crates/pllm-compiler/src/execution_capsule.rs), [runtime schedules](../../crates/pllm-compiler/src/decoder_runtime_schedule.rs), [state provenance](../../crates/pllm-compiler/src/provenance_primitives.rs) and [semantic-state checks](../../python/pllm/runtime/semantic_state.py). Current capsule admission expects an untransformed schedule; local framing is unauthenticated; clear integer state is not encrypted KV. Shared new work should be backend/key/layout/scale/level/error typed IR, deterministic layout transforms, counters, lifetime accounting, authenticated role transport and atomic encrypted-cache transactions. NEXUS coefficient packing, THOR multi-diagonal and CacheMir interleaving must remain different physical types.

| Method | Bounded faithful first slice | Independent exit gate |
| --- | --- | --- |
| NEXUS | Source coefficient compression/decompression and dual-key linear product, then source folding/nonlinear/real-refresh path | Uniform-ring mask ownership/retirement; independent dense algebra; exact layout chain; eight-squaring paper profile; full-vocabulary selection/tie tests before one complete BERT block/request |
| THOR | Native-shape BERT attention path: PC-MM → transpose/replication → CC-MM → ASoftmax/bootstrap → values/output/residual/norm | Basis-vector and garbage-slot tests; 12-head order/padding; lazy scale/relinearization trace; independent source-profile numerics; then FFN, 12 blocks, pooler/classifier |
| CacheMir | Encrypted prefill plus attention/cache transaction spanning partial/new K tiles and score/V capacity boundary, with real selected bootstrap | Both K and V decoded against independent source-layout oracle; old-token invariance; all levels/errors/key epochs; complete feasible DAG versus pruned result; then complete source-model block and approved private token loop |
| ROSETTA | Four-segment reciprocal with complete CKKS→RLWE/LWE→CKKS round trip, then shared multi-head conversion and hybrid softmax | Actual encrypted-accumulator selection; every segment/interval endpoint; floor/sign/key-domain checks; no plaintext scalar exposure; trace conversion counts and joint DAG choice; then full block and approved lifecycle |

Use source cryptographic parameters with bounded tensors, not toy-security contexts claimed as faithful slices. Maintain three independent comparisons: exact checkpoint computation, the specified plaintext approximation, and ciphertext computation. Test actual boundary/failure behavior, not an oracle imported from the candidate implementation. Numerical tests do not establish 128-bit security; a few successful decryptions do not establish bootstrap/PBS failure probability.

The canonical future gates should additionally name F1/F3 source-conflict resolution, F4’s minimal-input refresh edge and nonmonotone costs, F6’s plaintext-versus-encrypted accumulator distinction, F7’s GPT-2 positional boundary, and F8’s composition error analysis. A successful layer or externally supplied-token cache test is not a full private generation lifecycle.

## 7. Canonical-document dispositions

* **Retain:** dual-key NEXUS and single-output limitation; THOR encoder-only distinction; CacheMir separate K/V layouts/persistent-level gate; ROSETTA encrypted accumulator/complete recovery; original hardware/amortization; unknown numerical/key fields; native/backend/transport separation.
* **Add evidence:** F1/F3 author-code pins/conflicts; F4 counterexample; F6 actual Lattigo API boundary; F7 GPT-2 reference config; F8 sign-degree issue. These remain blocking until resolved, rather than downgrading the entire draft to unsupported.
* **Small locator fix:** canonical line 409 says ROSETTA needs “all primitives in §6.3”; the intended conversion reference is **§7.3**. This is editorial, not an approval issue.
* **Status:** “source-qualified implementation specification” at canonical line 3 should continue to be read alongside its explicit unresolved gates. It is source-qualified research planning, not an executable/admitted full-method contract. No backend, full numerical profile, whole-model extension or performance claim is approved by either document.

## 8. Supplemental immutable source locks and receipt

Raw author/library URL pattern: `https://raw.githubusercontent.com/<repository>/<commit>/<path>`. Hashes identify selected text inspections, not complete artifacts or the evaluated builds. Model configuration was read as JSON only; no model weights were downloaded or loaded.

| Repository / immutable commit | Path | SHA-256 |
| --- | --- | --- |
| `zju-abclab/NEXUS` / `e15058c7f8f58fcf037c4709211109cb1be7d1e0` | `README.md` | `b131427bbe4c41dae659d2f815b81e66118cba21f4f2e468c131f60a78d816d3` |
| same | `src/softmax.cpp` | `3dc8a77978ae9c047effe5ddb62785695e87c7f237ca17ee03a5127a53c74c70` |
| same | `src/layer_norm.cpp` | `7532fc8fdb5bd7c349b9f76944d3c311f759375c4b417b7c8e4c0848b28d6b46` |
| same | `src/bootstrapping.cpp` | `ced66d2a727bcf33cefa8fd76e5618ee8bd6f8597e4c5316c1ffc3774473395c` |
| same | `src/main.cpp` | `62803e05d66cdf24b9acafb542c0f4515c996eaed335225c1e89821486d558ca` |
| same | `src/ckks_evaluator.cpp` | `5350565afab4664717b12c2dc671e86f0b9b51ac4c466a071d3788b923a5d7ab` |
| `crypto-starlab/THOR` / `b513c06aaabccc04b560721e35e9a98008428c25` | `README.md` | `58d4ae6ba48ce6d52f47063a3df411d27bca7af470fd71339dd694fb53f0ec62` |
| same | `src/thor/ckks.py` | `8eec8fb7ac00a8b064b4f3c2b6938267b7cad88b079a40354948af65a6c79718` |
| same | `src/thor/nonlinear/softmax.py` | `69e377c27f23a5409cd0a11c92572041d8faa11dd958755313a7abcbbd540ecc` |
| same | `src/thor/nonlinear/layernorm.py` | `3435caeedba0218881c396bef3ca900cf8727eb2e873d6d26e4d49ae856540fb` |
| same | `src/thor/bert.py` | `21e04a41c1c769eec70cd3910bfbd25d63f21875f35dc61df83b2c7c690389c3` |
| same | `forward.ipynb` (JSON/source cells only) | `14eef46e671d8c2a192db2e374a7643fa8ae9e4acdcb9df9ae4a9bd47c7b2466` |
| `tuneinsight/lattigo` / `37a34b4676e987795949a795f0ce9d0893763c96` | `README.md` | `05e71c7fea53a9512c45cf2c98e083d95cf75752732d443c0962950d60f6b1f7` |
| same | `core/rgsw/blindrot/evaluator.go` | `4bdf3fe5918b70ff14caad5650dd53f33887e276861484cb910f5be4f6601260` |
| same | `core/rgsw/blindrot/blindrot.go` | `8c02080399c448adf0f9e8829d4aee7d424044926e1e61a4a85a406ab029eb86` |
| same | `core/rgsw/blindrot/keys.go` | `cdb88b610e5a596a0edc995df15ad21a13b8d1e610eb5c7bcc5fce98ee172edb` |
| `openfheorg/openfhe-development` / `aa391988d354d4360f390f223a90e0d1b98839d7` | `README.md` | `c9950b76a53cebea9e3cc08fddd85582cbc70eea24be347d7316167a5006ed75` |
| `Desilo/liberate-fhe` / `b263100c830fda96be86776ff1b4fd6394e6a071` | `README.md` | `103be782fb27c82403f38b1932a940554ff452c37560f6a65c6415c76576351f` |

Primary text came from existing extracts under the manifest’s `fetched_snapshot_base`; exact PDFs remain authoritative. Extraction artifacts include damaged mathematical layout in R Figure 6; associated prose and Algorithm 1 were cross-read. Unresolved details were kept unresolved rather than filled from undocumented defaults.

Performed: primary-text reading and local PDF hashing; selected immutable author/primitive-library text inspection; canonical-draft comparison; static PLLM reuse inspection; document-only link/hash/format validation. Validation passed: all 15 local links resolve; four PDF hashes and the canonical-document hash match; all 18 supplemental author/library file hashes match retrieved immutable bytes; fences are balanced. The canonical file remains byte-identical to the reviewed snapshot. No cryptographic, model or performance execution evidence was generated.
