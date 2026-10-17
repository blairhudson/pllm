# Independent source-fidelity audit: shared computation

Review date: 17 October 2026. Papers: SIGMA, FuseFSS, BumbleBee, PUMA.

**Canonical draft:** [shared-computation-contracts.md](../methods/shared-computation-contracts.md). This file is an independent audit of that draft, not a second implementation contract. The reviewed canonical snapshot has 462 lines and SHA-256 `fae5702d44e0ce27940deaf2e1203500e656e683c9271fe6197d7cccff6237c1`; line references below identify that snapshot. The canonical file was not edited.

**Verdict:** the draft correctly separates original methods from PLLM's bounded components and identifies the important security, numerical, and serving gaps. No complete method is qualified. This audit adds static author-source evidence, resolves PUMA's version and locates its external SiLU coefficients, and narrows the highest-priority decisions. It also finds author-snapshot/paper mismatches that must not silently become implementation defaults.

Substantial numerical, protocol, security, state, execution, and architectural choices still require user approval **before implementation**. Recommendations here are not approvals. Activities were document inspection, hashing and static source retrieval only: no implementations, builds, model jobs, benchmarks, or upstream-code execution.

## 1. Findings requiring action, in priority order

| Priority / finding | Canonical location | Independent result and concrete recommendation | What it blocks |
| --- | --- | --- | --- |
| **P0-N1: BumbleBee evaluated numerical identity** | Lines 279–297; A3 | Final paper says ring64/Q18, exp cutoff −14 with six squarings; pinned author `2pc.json` instead says Q16/four iterations. Mixed GELU paper interval is ±2.7; author activation file uses ±3 with the same degree-four coefficient family. Obtain a source-linked evaluated configuration; otherwise ask approval for a fully enumerated **final-paper-derived** profile. Keep endpoint/epsilon, mixed-width, H2A error and extension-heuristic choices explicit. | Affected nonlinear kernels, full-model correctness/quality and source-table reproduction. Pure OLT algebra can be specified independently. |
| **P0-N2: PUMA evaluated arithmetic and pseudocode resolution** | Lines 369–402; A3 | v3 is verified. A contemporaneous SPU pin resolves candidate SiLU coefficients, input cast policy, selected `TruncAPr`, Goldschmidt iteration default and rsqrt polynomial. Its softmax clips **shifted exponentials before summation**, unlike literal Algorithm 2. Recommend that pinned source-derived interpretation with explicit approval/author confirmation; preserve architecture-correct mean variance/epsilon only as an approved resolution of Algorithm 3. | Protected numerical kernels depending on these choices and full-model admission. Version identification and coefficient discovery themselves are no longer blockers. |
| **P0-N3: FuseFSS lacks an evaluated descriptor manifest** | Lines 186–192; A3 | Nearest encoding with unspecified tie-breaking is genuinely distinct from SIGMA's floor encoding. No exact per-model n/f, table/coefficient bytes, rescale graph or evaluated backend configuration was established. Seek the evaluated manifest. If unavailable, seek approval for explicit source-derived descriptors; **do not adopt floor or ties-even by implication**. | Claims of original numeric identity and reproduced model tables. Generic compiler algebra can be developed only under an explicitly chosen development specification. |
| **P0-T: roles and device topology** | Lines 91–101,174–182,268–273,354–365; A7–A9 | SIGMA has original **CPU and GPU schedules**, both with two online owners plus a separate trusted dealer. Its GPU online client needs GPU execution. FuseFSS already permits an external CPU client and two GPU servers, and defaults to public weights. BumbleBee requires distributed OT/HE, not a local OT dealer. PUMA requires three online replicated-share parties. Recommend preserving these distinct topologies. | Any deployment/backend design that moves an owner to a server, colocates a private-model dealer, changes weight privacy, substitutes local OT or reduces PUMA's online roles. |
| **P0-B: complete token boundaries and lifecycle** | Lines 116–121,239–245,322–326,424–428; A4–A6 | Inspected SIGMA timing files operate on zeroed embedded tensors and omit final norm/head. PUMA/BumbleBee example loops recompute the full prefix; neither passes persistent KV. Recommend source full-functionality/output qualification first, full-prefix feedback where evidenced, then a separately approved persistent-KV/streaming extension. | Complete token-to-token and serving claims. A complete body benchmark or registry state name cannot fill this gap. |
| **P1-M: exact original/common model identity** | Lines 81–90,168–173,260–267,344–353; A1 | GPT-2 is a defensible common first family for all four, but common checkpoint/tokenizer identity is not established by its name. Follow canonical source workloads first: GPT-2 length 128 for SIGMA/FuseFSS/BumbleBee, 32 for PUMA. Add a separately identified common-workload cohort once approved. Preserve each paper's BERT/RoBERTa/Llama/ViT coverage separately. | Full original-workload coverage and matched benchmarking, rather than primitive implementation. |

### What is already resolved, rather than a user choice

* The reviewed PUMA is **v3**, not v1. Its GELU constants can be transcribed exactly. SiLU constants are recoverable from the externally referenced author file; an evaluated-build association is still needed.
* SIGMA is not intrinsically GPU-only. The original CPU branch is faithful if its own schedule is used; FuseFSS's evaluated implementation is GPU-based.
* BumbleBee's **Ferret OT and SiRNN-derived ring extension are real protocol dependencies**, not optional labels. The task is to independently realize them or explicitly approve a different primitive boundary.
* PUMA's third party computes online; FuseFSS's default public weights and SIGMA/BumbleBee/PUMA's private-model functionality are different comparison classes.
* Existing measured negative adaptation results are not failed original implementations. Missing methods remain missing, with unknown cost and eligibility.

## 2. Evidence and source pins

### 2.1 Primary documents

All manifest PDF hashes below were independently recomputed and matched. PDF page numbers start at one; SIGMA printed pages are PDF page + 60. See [prior-art-sources.json](../prior-art-sources.json) for cache locations. The final BumbleBee cache needs durable preservation before publication.

| ID | Source / reviewed bytes | SHA-256 | Audit locators |
| --- | --- | --- | --- |
| S | [SIGMA, PoPETs 2024(4):61–79](https://petsymposium.org/popets/2024/popets-2024-0107.php), `papers/r13-sigma.pdf` | `f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf` | §§2–6, pp.2–11; Tables 1–5, pp.11–12; Appendices C/F/H, pp.16–18; Tables 7–9, p.19. |
| F | [FuseFSS, arXiv:2606.09551v1](https://arxiv.org/abs/2606.09551v1), `papers/r14-fusefss.pdf` | `6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0` | §§3–6, pp.3–8; Appendix B, pp.12–15; Protocol 3, p.17; backend/proof/accounting appendices. |
| B | [BumbleBee, NDSS 2025 final](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf), `bumblebee-final.pdf` under manifest `fetched_snapshot_base` | `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b` | §II, pp.3–4; §III, pp.5–7; §IV/Algorithm 2, p.8; §V/Algorithm 3, pp.8–10; §VII, pp.11–13; Appendices C–E, pp.16–18. |
| B-old | Older BumbleBee bytes behind PLLM's existing interleaving component, `papers/r22-bumblebee.pdf` | `15583a86593998079d42b8b134dc4f4f5be9ca72a7ced8f31bba05f68893516a` | OLT/interleaving is **§III**, pp.5–7, in these bytes too. The old component card's §IV locator is stale; canonical lines 30–34 correctly identify this. |
| P | [PUMA, arXiv:2307.12533v3](https://arxiv.org/abs/2307.12533v3), `papers/puma.pdf` | `84c82810a16be95b41200ba8e0071c39a8cadb15a45008ff8e75e295e0960973` | §§3–4/Algorithms 1–3, pp.3–7; §5, pp.6–9; Appendix B, pp.12–13. |
| P-v1 | [PUMA v1 PDF](https://arxiv.org/pdf/2307.12533v1), fetched for this audit into approved temporary storage | `f13108cfca6335783f23227ca2106aeea7e7451b9b5e8f084c43a70096ffcfe1` | Version stamp/title page, p.1; GELU Eq.(2)–(3), p.5. A comparison source, **not the selected contract**. |

**Version check:** P's PDF stamp is `26 Sep 2023`, title-page date 27 September; the primary arXiv history identifies it as v3. P-v1 is stamped `24 Jul 2023`, with title-page date 25 July. Direct inspection finds the same GELU Eq.(3) decimal coefficient strings in these two PDFs, but section order changes: v1 §4.2 is embedding and §4.3 is GELU; v3 §4.2 is GELU and §4.4 is embedding. Equal coefficients do not make the versions interchangeable. FuseFSS's primary arXiv page independently confirms v1, submitted 8 June 2026.

### 2.2 Selected author files: static evidence only

Files below were fetched as inert bytes and SHA-256 hashed. No file was imported, built or run. These are selective source inspections, not complete artifact audits.

* **S-code:** [mpc-msri/EzPC, `aa0f2f860266fb79901405ef5e9965b8c9fe2e88`](https://github.com/mpc-msri/EzPC/tree/aa0f2f860266fb79901405ef5e9965b8c9fe2e88), latest commit returned for the SIGMA experiment path, 31 May 2024.
* **B-code:** [AntCPLab/OpenBumbleBee, `47c5560d069543591f3b0176eb530ede28e33fc3`](https://github.com/AntCPLab/OpenBumbleBee/tree/47c5560d069543591f3b0176eb530ede28e33fc3), `ndss` branch snapshot, 3 February 2025. README names SPU base `724d3d1c3c37e891fcd0493a6cff0bf4310eda70`.
* **P-code:** [secretflow/spu, `a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4`](https://github.com/secretflow/spu/tree/a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4), latest commit returned before `2023-09-27T23:59:59Z`. This is a **contemporaneous source candidate**, not an author-certified evaluated revision. P's Appendix B explicitly points to `flax_llama7b.py`.
* **F-code:** no evaluated author artifact/configuration pin was established from the inspected primary sources. This does not assert that no code exists.

Paths are relative to the corresponding repository pin. Source code line references below use these exact bytes.

| ID | File | SHA-256 |
| --- | --- | --- |
| S1 | `GPU-MPC/experiments/sigma/README.md` | `0dd7674319ab3109b1bf7b1db5ba947a36eb38cab480312d5cd761de6a48b503` |
| S2 | `GPU-MPC/experiments/sigma/sigma.cu` | `f7fd8ce3c3db8ec6218c39d804daf063f024beb20babc45d1addf67f07798c76` |
| S3 | `GPU-MPC/experiments/sigma/gpt2.h` | `8b9692c25385bd10730e89a1f1f0380232554448accd5f229254773dc9a0a779` |
| S4 | `GPU-MPC/experiments/sigma/llama2.h` | `43105bb5d6a424e5255e2ff0b919523385a8e6334876b46d8751dc0314385e04` |
| B1 | `README.md` | `3ac32f770ef3ef7defc7c85a04b112fd6acea9078d82f38b3b4880636f99060b` |
| B2 | `examples/python/conf/2pc.json` | `8a79dda79e22759c0eddc35efb40c3640d6135c9c5d5afb67d8f848dba931d19` |
| B3 | `examples/python/ml/flax_gpt2/flax_gpt2.py` | `d8b5131b660ba55163c69943139a87a1341516dbea7e4c15fdf4038122bc71bd` |
| B4 | `libspu/kernel/hal/intrinsic/nn/activation.cc` | `107109fd12e6b803b63a0328a065ddad30bcd160ee35b4757ed35690337ecd0f` |
| B5 | `libspu/mpc/cheetah/nonlinear/ring_ext_prot.cc` | `fbfda2a15c647420467fb9de7641bb5ac3ed21032f8dd5a025ec0cc040da9a65` |
| B6 | `libspu/mpc/cheetah/ot/yacl/ferret.cc` | `791cf53e63aa7e0af82493fcfba3cdb7a6d706005b1d3e0a8ead89c95aa948e6` |
| B7 | `libspu/mpc/cheetah/nonlinear/README.md` | `945eca3dc7c31bad36486806addd4f81aec2b8b9e9a7c04931222c42af38a2ae` |
| P1 | `examples/python/ml/flax_llama7b/3pc.json` | `b65bc73070db894c8ebc64b90199e6fd11d0d1f3483955ef9ccb46de22ec5d5d` |
| P2 | `examples/python/ml/flax_llama7b/flax_llama7b.py` | `0c9c178c06d32151981009f94e60410809a344fad0745f3dab7f655bb67c267c` |
| P3 | `examples/python/ml/flax_gpt2/flax_gpt2.py` | `65af704c3ee9b025e393d739f3ac18c78a5ba04f2635ee4845ce86c8dbb5ea7a` |
| P4 | `libspu/core/encoding.cc` | `43e9df235c6938e0e4ad1509cf4d98a92d78fd45f310d2a4c69be0400b43fc80` |
| P5 | `libspu/kernel/hal/fxp_approx.cc` | `aa6003fe2bc3fc5154ab9d5a49dfbe133fa514a0ce2e0cbfa63bb311b9bdb5a9` |
| P6 | `libspu/kernel/hal/fxp_base.cc` | `2d6a32362c5996fb181128b3560315e8f38753287648bf30fa7ae11b49dbddb7` |
| P7 | `libspu/mpc/aby3/arithmetic.cc` | `e06d00adb3a292f4016cb8de590700a2baa5293e518fc1b90cd3eb72c945554e` |
| P8 | `libspu/core/config.cc` | `4cef3b0ce2154106d5ac40816336c2494d339405e0c2a1d44831a1506f9bfa59` |
| P9 | `libspu/mpc/aby3/protocol.cc` | `079e1fd685ccbb4a34209a6b8a963ce7e918e699be5d8f7c4eb5f1f4dd75a73b` |
| P10 | `libspu/spu.proto` | `b986a842f1132775c86a5f11692a8c6d9b14f7e8ba715424c39084b27375115c` |

Repository date and path relevance do not bind an evaluated binary. Any adoption of these candidates must also qualify frontend/dependency versions, compiler rewrites, configuration and graph identity.

## 3. SIGMA audit

### Confirmed canonical claims

* **Numerics:** S §2.1.1 specifies floor encoding; Table 4 fixes f=12 and ring widths 37/50/50/50/51/48/48 for BERT-tiny/base/large, GPT-2, GPT-Neo and Llama2-7B/13B. Exact signed ARS, reduced-ring TR and quarter-ring `GapARS` precondition agree with canonical lines 107–109. Local share shifts do not replace these protocols.
* **Operators:** CPU/GPU GELU schedules in Figs.6–7 are genuinely different. Q6-indexed residual LUTs have 256 GELU and 1,024 SiLU entries; the source clipping codes are respectively `[-2^(f+2)+1,2^(f+2)-1]` and `[-2^(f+4)+1,2^(f+4)-1]`. Moving truncation across absolute value can change low bits. Softmax's protected max, two 8-bit nExp tables, reciprocal and norm's 13-bit RecSqrt table are substantive requirements, not existing generic interval coverage.
* **Norm/RoPE:** Fig.12's `s+1` guard and precision `2f+log2(k)` must be retained as specified; `log2(k)` must not silently become an integer shift for arbitrary hidden widths. Appendix C's RoPE coefficient reduction by three fractional bits is part of Llama2's numeric identity. Canonical lines 113–114 correctly distinguish these from ordinary floating norm/RoPE.
* **Security:** two online owners, independent trusted dealer, input-independent fresh masking, private model and input, one static semi-honest online corruption. A trusted client that knows all private-model masks and sees their masked wires would recover the model. Canonical dealer questions are true security blockers, not merely accounting choices.
* **Backend:** λ=127 seed/control packing and AES-128/early termination are source-specific. Existing PLLM 128-bit seed plus separate controls is reusable algebra, not byte/cost equivalence.

### New static artifact evidence

S1 lines 10–14 require NVIDIA GPUs/CUDA for **that artifact** and identify tested Ubuntu 20.04, CUDA 11.7 and CUTLASS 2.11. It retains preprocessing in host RAM and advertises power-of-two sequence lengths. This does not invalidate the separately published CPU algorithm.

S2 lines 138–168/176–212 zero-initialize a model and input and benchmark a tensor `(sequence,hidden)`. S3 lines 94–124 and S4 lines 92–120 build transformer stacks with final norm commented out and no token embedding or full vocabulary head. Thus these particular timing paths do not independently settle canonical lines 116–121's token-boundary gap. This is **not** a claim that every author quality path is missing; model quality is evaluated separately in the paper.

### Hardware/lifecycle check

S §7.0.3 uses two machines, each AMD EPYC 7742, 1 TB RAM, NVIDIA A6000 reported as 46 GB memory, LAN 9.4 Gbps/0.05ms ping. Source CPU measurements use four threads. Appendix L/Table 9 separates dealer key generation/transfer from online latency. Model loading, material delivery, host/device buffers and streaming must be costed in a new whole-lifecycle cohort; they are not included merely because a headline is called “inference.”

**Default recommendation:** retain original CPU and GPU profiles with distinct schedules and costs. With a CPU-only **client**, choose original CPU execution unless the user approves changing its online role. An external CPU client with two GPU servers is a topology adaptation for SIGMA, even though it fits FuseFSS's source setting.

## 4. FuseFSS audit

Canonical lines 174–215 are supported: public model by default, two non-colluding online servers, trusted preprocessing, one semi-honest server corruption; typed arithmetic/XOR outputs; canonical interval descriptors; fixed mask-independent instance shapes; at most **two batched backend interface calls per scalar gate**, not two network rounds for a model. Horner/Boolean products, conversions, rescaling and openings are real costs. Vector max/sum/attention/norm remain composed graph operations.

### Rounding is a genuine unresolved field

F §3.2, p.3 specifies nearest-integer encoding with a fixed tie rule; it does not identify the rule. S §2.1.1 specifies floor. F Appendix B.3's “same fixed-point specifications” does not erase this distinction. For a real input with scaled value `−1.5`, floor gives −2; different nearest tie policies can give −2 or −1. The difference matters at activation pivots and later protected products even if most benchmark inputs never hit an exact tie.

**Default recommendation:** do not choose a production tie policy yet. Recover evaluated descriptor bytes, input encoding and all per-edge n/f/rescale settings. If unavailable, ask the user to approve a named paper-derived interpretation; nearest/ties-even can be proposed for determinism, but it is not established as original and must not overwrite SIGMA floor semantics. The independent oracle must evaluate that exact descriptor rather than reuse the compiler's output evaluator.

### Hardware and state are correctly separated in the draft

* F §6 uses one RTX PRO 6000 Blackwell Workstation Edition GPU per online party, EPYC 9654 CPUs and CUDA 13.0. CPU-only runtime is a portability profile, not reproduced source GPU execution.
* Network projections use **1 GB/s, 0.5ms** and **400 MB/s, 4ms**, unlike BumbleBee's Gbps/Mbps network settings. Preserve printed units and the distinction between projected and observed latency.
* Main models are BERT-tiny/base/large, GPT-2/GPT-Neo, batch 1 and length 128. Appendix B.2/Table 8 includes **LLaMA-7B and LLaMA-3.1-8B**, lengths 16/32/64, using host buffers/key streaming. Canonical lines 168–173 correctly include these; Llama-3.1 is not automatically a new architecture claim here.
* Appendix B.4's reusable **program** cache is not persistent KV. Fresh masks, FSS keys, triples and conversion material remain per-use. The scalar compiler theorem does not settle token lookup/head/output feedback.
* Appendix B.5 explicitly treats attempted BOLT/BumbleBee/SHAFT comparisons as non-comparable. They cannot become negative complete-method results in PLLM.

**Oracle implication:** test complete partition/wrap/sentinel cases under all small-ring masks, including mask-independent key lengths and padding. One hard-coded fused SiLU block, or agreement only with an unfused implementation sharing the same bug, is insufficient.

## 5. BumbleBee audit

### Additional conflicts in the author snapshot

The canonical draft already catches the wide GELU offset/endpoint conflict, approximate H2A error, missing heuristic `M`, norm variance wording and ViT mixed-width discrepancy. Static evidence adds these blockers:

| Source | Observed setting | Consequence |
| --- | --- | --- |
| B §VII, p.11 | ring64/Q18; evaluated exp cutoff −14, n=6; selected ring32/Q12 activation variants | Intended final-paper target, distinct from §V.B's illustrative cutoff −13. |
| B2 lines 23–36 | `fxp_fraction_bits=16`, `fxp_exp_iters=4`, `YACL_Ferret`, multiplication LSB error enabled, `approx_less_precision=4` | This configuration is not interchangeable with the paper's evaluation profile. Its association with the final tables needs evidence. |
| B §V.C/Appendix D, pp.10,17 | Degree-four absolute-value GELU near `[-2.7,2.7]` | Paper-derived mixed GELU range. |
| B4 lines 99–155 | Ring32/Q12 path, `apprx_range=3.0`, degree-four coefficients, one final truncation of accumulated terms | A concrete ±3 variant. Changing to ±2.7 alters branch behavior; copying it silently would not resolve the paper contract. |
| B4 lines 161–205 | Explicit `CastRing` before/after mixed GELU, extension with unknown sign | Demonstrates a real conversion protocol boundary. Width reduction is not evidence that extension can be local. |
| B4 lines 290–318 | Cutoff −14, repeated squaring count read from configuration | Source supports the cutoff but does not resolve four-versus-six iterations. |
| B5 lines 201–328 | General and positive/heuristic extension paths | Must bind selected path and admitted range. General extension can change cost even if it improves range coverage. |

### Coefficients and the actual ambiguous function

B Appendix D's wide GELU coefficients closely match PUMA's GELU family, but **different pivots, tail offsets, approximate comparisons and decimal spellings prevent substitution**. B Eq.(2) uses tails `−epsilon` and `x−epsilon`, with epsilon `10^-5` and pivots −5/−1.97/3. Algorithm 3 adds epsilon to middle polynomials and then aggregates without showing the final subtraction needed for those tails. Its comparison tests and interval annotations also disagree on endpoints. Canonical lines 285–297 correctly flag this.

The mixed polynomial is

```text
P4(x) = 0.5*x + c0 + c1*|x| + c2*|x|^2 + c3*|x|^3 + c4*|x|^4
c0 =  0.001620808531841547
c1 = -0.03798164612714154
c2 =  0.5410550166368381
c3 = -0.18352506127082727
c4 =  0.020848611754127593
```

These coefficient strings agree between B Appendix D and B4; the **active interval differs**. Source quality cannot be inferred from constants alone. Freeze floor coefficient encoding, exact accumulation/truncation order, approximate-comparison bit dropping, tails and all pivot-neighbor behavior in the selected profile.

**Recommended resolution:** request final evaluated config/graph evidence first. Failing that, propose final-paper Q18/n=6 with documented model-specific Q12 activation paths, and explicitly choose Eq.(2)'s intended tail function versus literal Algorithm 3 under user approval. Do not call an arbitrarily repaired combination “the original implementation.”

### Distributed OT, HE and security check

* B §II.D: private client input/private server model, one static semi-honest corruption, no trusted dealer. B6 lines 124–167/878–926 creates sender/receiver Ferret adapters on a communication context and invokes one-time setup; B2 selects that implementation. A local dealer handing both sides correlated OT material is not this setup.
* B §V.C explicitly builds ring extension from **SiRNN [61] plus a heuristic optimization**. B7 and B5 identify the relevant nonlinear implementation family. This dependency includes signedness, overflow and two-way communication, not only interface names.
* Appendix E.2: OLT `N=8192,q≈2^147,t=2^64,q'=2^49`; bOLE `N=8192,q≈2^148`, CRT plaintext product of three approximately 45-bit coprimes, SIMD flooding bound `B'=2^98`. Concrete primes/noise/key-switch parameters remain unpinned. PLLM's bounded BFV lookup cannot obtain this functionality merely by changing its plaintext-modulus field.
* B §IV/Appendix C permits reconstructed ±1 errors in the relevant concrete conversions. SIMD noise-flooding H2A and coefficient-form approximate resharing are different constructions. Exact polynomial matmul or positive noise budget alone does not prove circuit privacy or complete-path exactness.

### Workload, output and lifecycle check

Main CPU hosts are two Alibaba `ecs.g7.16xlarge` instances, each 64 vCPUs at 2.70GHz/256 GB RAM. LAN is 1 Gbps/0.5ms ping; WAN is 400 Mbps/4ms. Table II uses 25 threads and excludes HE key generation/base OT; §VII excludes disk loading. Table IV uses GPT2 128-input/one-top-1 and LLaMA 8-input/one-top-1; Table V's GPT2 comparison uses **64** input tokens. Paper GB means `2^30` bytes. These distinctions matter when deriving a new lifecycle cohort.

B3 lines 119–135 computes the whole growing prefix, selects shared argmax, appends its ID and finally reveals IDs. No KV argument is passed. Its eight-token example is not Table IV's one-token cost case. B1 lines 175–203 proposes removing JAX's explicit smaller-index tie handling from argmax; therefore freeze ties rather than inherit arbitrary framework behavior. Extra logit disclosure or stochastic sampling is a functionality change, as canonical A6 already states.

## 6. PUMA audit: version, constants and source-derived defaults

### 6.1 Exact v3 GELU constants

Canonical lines 375–379 are supported. P v3 Eq.(2)–(3), p.5 gives:

```text
F0(x) = -0.011034134030615728*x^3
        -0.11807612951181953*x^2
        -0.42226581151983866*x
        -0.5054031199708174

F1(x) =  0.0018067462606141187*x^6
        -0.037688200365904236*x^4
        +0.3603292692789629*x^2
        +0.5*x
        +0.008526321541038084

GELU(x) = 0          for x < -4
          F0(x)      for -4 <= x < -1.95
          F1(x)      for -1.95 <= x <= 3
          x          for x > 3
```

Algorithm 1 forms `x2=Square(x)`, `x3=Mul(x,x2)`, `x4=Square(x2)`, **`x6=Square(x3)`**. Every square/multiply includes the selected secure fixed-point truncation. Horner evaluation or `x6=x2*x4` changes this rounded computation. The decimal constants match v1 Eq.(3); that limited check does not certify full v1/v3 identity. Quantized constant bytes still depend on the chosen input/constant encoding and intermediate representation.

### 6.2 SiLU is externally recoverable, not an invented polynomial

P v3 Appendix B, p.12 directs readers to `flax_llama7b.py`. P2 lines 86–118 provides this contemporaneous candidate:

```text
A(x) = -0.0055465625580307*x^2
       -0.0819767021525476*x
       -0.3067541139982155

B(x) =  0.0002743776353465*x^6
       -0.011113046708173*x^4
       +0.2281430841728270*x^2
       +0.5*x
       +0.0085064025895951

SiLU(x) = 0          for x < -8
          A(x)       for -8 <= x < -4
          B(x)       for -4 <= x <= 4
          x          for x > 4
```

The intervals above follow the **Boolean expressions**, not the file's inconsistent interval comments. P2 computes `x2=Square(x)`, `x4=Square(x2)`, `x6=x2*x4`; this is its SiLU schedule, not P's GELU schedule. The coefficient strings also appear in BumbleBee Appendix D, but P2 has **zero/identity tails**, whereas BumbleBee states epsilon-offset tails. Same coefficients do not imply the same function or protocol.

**Recommended canonical update:** qualify lines 399–402/A3 from “coefficients missing” to “coefficient source located at P2; evaluated-build association, encoding, lowering and LLaMA graph still unqualified.” Do not copy modern `main`, borrow BumbleBee's tails or use SIGMA LUTs. No canonical edit was made by this audit.

### 6.3 Candidate implementation defaults are now concrete

These observations resolve what the inspected source does; they do not prove what every v3 experiment executed.

| Area | P-code evidence | Proposed default, conditional on approval/qualification |
| --- | --- | --- |
| Ring/fraction | P1 lines 24–30 selects ABY3/FM64; P8 lines 22–29 defaults FM64 to **18** fractional bits | Ring64/Q18, matching the paper. |
| Input encoding | P4 lines 63–98 uses a signed integer target and scaled cast **toward zero**. Clamps to encoded `[-2^62,2^62-1]`; NaN maps to zero | Explicit source-candidate encoding, distinct from SIGMA floor. The quarter-ring gap corresponds to about `[-2^44,2^44)` real Q18 inputs; it is not a proved intermediate bound. Input validity/clamping policy requires qualification. |
| Truncation dispatch | P9 lines 57–62 defines `ENABLE_PRECISE_ABY3_TRUNCPR` and registers **`TruncAPr`**, not `TruncA` | Adopt the pinned protocol only if source-derived profile is approved. P7 lines 750–887 cites *Secure Evaluation of Quantized Neural Networks* §3.2.2; its name “precise” is not proof of deterministic floor. Derive error law and range/security assumptions from the full protocol. |
| Reciprocal | P6 lines 228–269 uses normalized Goldschmidt with start `2.9142−2c`, `c in [0.5,1)`; P8 lines 51–52 defaults to **two iterations** | Two-iteration source candidate, with all internal rescale operations and constant precision preserved. |
| Exp | P1 sets five iterations; P8 defaults exp mode to Taylor; P5 lines 138–149 implements truncated division by `2^t` and repeated protected squares | `t=5` for the inspected LLaMA example, matching P's intended softmax approximation; do not use global default eight iterations. |
| Rsqrt | P5 lines 462–570 normalizes to `[0.25,0.5)` and uses degree-four `26.02942339*u^4−49.86605845*u^3+38.4714796*u^2−15.47994394*u+4.14285016`, then exponent/parity compensation. P10's lower-accuracy flag defaults false | Source-candidate full polynomial/compensation path; preserve the source constants' floating representation before ring encoding. Not a Newton-loop placeholder or local float rsqrt. |
| Model graph/compiler | P10 includes switches for sqrt/divide rewrites and truncation-reduction optimization | A protocol pin alone is insufficient. Bind actual compiler passes, graph/truncation placement and external Flax/EasyLM revisions. |

### 6.4 Pseudocode conflicts remain true blockers

**Softmax:** P Algorithm 2 computes its mask on unshifted `x` and applies it after normalization. P2 lines 64–70 instead computes `x−max(x)`, applies `x>−14` to exponential values **before their sum**, then divides; it subtracts no epsilon. P1/P5 map exp to the five-square candidate above. These are different functions: for a row `[-20,-20]`, the literal Algorithm 2 masks both outputs to zero, while shifted clipping yields equal normalized outputs. This is a mathematical counterexample, not an executed workload.

**Recommendation:** propose P2's shifted-and-clipped-before-sum semantics, with integer epsilon zero if that source-derived behavior is approved. Do not claim the strictly-negative-input property from P's suggested `10^-6`, which is below one Q18 unit (`2^-18≈3.81e-6`). Alternatively recover a certified evaluated graph; no silent repair is justified.

**LayerNorm:** Algorithm 3 and surrounding text omit division of squared deviations by width and do not fix epsilon. For `[-1,-1,1,1]`, mean variance gives normalized magnitudes one, while unaveraged sum gives one-half before affine parameters. This is a substantive model change, not an insignificant rounding typo.

**Recommendation:** seek the evaluated graph, then ask approval for checkpoint-equivalent mean variance and epsilon if interpreting the prose intent. Keep the optimized single-rsqrt/broadcast construction. Do not claim literal pseudocode fidelity for that correction.

### 6.5 Security, hardware and state check

Canonical lines 354–365 correctly preserve three online replicated-share holders, one semi-honest corruption, private weights/input, per-layer share invariant, all multiplication cross terms and zero-sum setup randomness. The word “dealer” in one P §4.1 owner-placement example does not turn P2 into an offline-only party.

P §5 uses three CPU `ecs.g7.8xlarge` hosts, each 32 vCPUs/128 GB, Xeon Platinum 8369B at 2.70GHz, about 5 Gbps/1ms RTT. LLaMA §5.4 uses three `ecs.r7.32xlarge` hosts, each **128 threads/1 TB**. Its text prints “20GB bandwidth, 0.1ms round-trip-time”; bandwidth units are ambiguous and must not be silently normalized to Gbps. Table 6 cases are `(input,output)=(4,1),(8,1),(8,2)`, not steady-state decoding. The 2-GB serializer/chunking requirement is real.

P2 lines 134–144 and P3's generator call the model on the full growing prefix, perform secure greedy argmax and append token IDs; no KV argument is passed. P Appendix B's fixed-randomness text example does not define a general stochastic sampler. This supports **full-prefix feedback as the initial source-derived default**, not a claim that all original executions used the same sampler or that persistent KV already exists.

## 7. Implications for implementation planning

This section audits dependencies in the canonical draft; it does not replace its plan or authorize work.

### Reusable architecture versus false method coverage

Static local checks support the canonical gaps:

* [masked_wire_schedule.rs](../../crates/pllm-compiler/src/masked_wire_schedule.rs) binds client float conversion/SiLU×up regions and rejects `GeluTanh`; [masked_wire.py](../../python/pllm/runtime/masked_wire.py) uses client material issuance and relay. Neither is original private-model SIGMA or a general FuseFSS compiler.
* [compact_dpf.rs](../../crates/pllm-garble/src/compact_dpf.rs) has a different seed/control layout. [interval_fss.rs](../../crates/pllm-garble/src/interval_fss.rs) and shared-rescale helpers are bounded primitives, not FuseFSS descriptor compilation or complete shared attention.
* [coefficient_repacking.rs](../../crates/pllm-core/src/coefficient_repacking.rs) hard-limits degree 4,096 and at most two RNS limbs. Existing packed lookup's plaintext modulus 8,192/local-dealer conversion differs from BumbleBee OLT/bOLE, distributed Ferret and protected extension.
* [model operator definitions](../../crates/pllm-models/src/lib.rs) have RMSNorm/RoPE/SiLU/`GeluTanh` but no dedicated LayerNorm or learned-position embedding operator. GPT-2 requires genuine generic frontend/operator work, including final norm, biases and tied head. A Qwen graph with different weights is not GPT-2.
* PUMA's pending protocol/state names in [planned_methods.json](../../python/pllm/components/planned_methods.json) do not instantiate three-party replicated tensor operations. Local projected resharing and two-party PCF/VOLE secret replication have different semantics.

Recommended abstractions remain capability-based: typed ring/scale/error/range/share representations, explicit conversions/truncation, model-neutral operator graphs, generic CPU/GPU and additive2/replicated3/masked/ciphertext backends, and independently owned role transport/material inventories. Paper names belong in frozen composition manifests, not runtime branches. These interface changes require architectural approval.

### Source-specific versus common GPT-2

The canonical first-complete-model choices are reasonable. Use each source's original GPT-2 task geometry before claiming source-cost reproduction; common checkpoint and common prompt/output lengths form another approved cohort. Preserve original-model quality metrics alongside complete response metrics:

* SIGMA/FuseFSS: BERT classification and GPT/LAMBADA; Llama2 versus LLaMA/Llama-3.1 distinctions remain explicit.
* BumbleBee: BERT/ViT quality results do not by themselves establish GPT2/LLaMA language-model quality. BERT is a bidirectional encoder, not a source of autoregressive KV claims.
* PUMA: BERT/RoBERTa GLUE and GPT2 WikiText-103 v1 perplexity; original LLaMA-7B is distinct from SIGMA's Llama2 and FuseFSS's Llama-3.1.

Modern common-model substitution, retraining, checkpoint conversion, client one-hot generation, extra logit disclosure and persistent KV each have separate numerical/functionality/security consequences. None is justified just to fit existing operators.

### Dependency order and meaningful gates

1. Resolve the prioritized profile/topology decisions in §1; preserve primary bytes and pin checkpoint/frontend/backend identities.
2. Write independent integer/rational reference semantics and constant/table generation from approved sources. Bind deterministic versus probabilistic error contracts before native code.
3. Supply generic numeric types and real protocol foundations: FSS/dealer, distributed OT plus circuit-private RLWE conversions, or replicated3/PRSS/truncation. Select GPU/CPU schedule explicitly.
4. Use **component integration slices as milestones only**: SIGMA private affine→rescale→source GELU; FuseFSS a genuinely compiled multi-output gate with conversions/products; BumbleBee OLT→H2A→mixed GELU→extension; PUMA private token ID→equality/embedding→private affine→GELU. These do not replace canonical complete-model slices.
5. Complete attention/norm/activation/conversion coverage, full embedding/body/final norm/head and authorized output. Exercise actual separate online roles without clear test-oracle reconstruction in the execution path.
6. Qualify source full-prefix feedback and any separately approved protected-KV/sampler extension. Gate on prefill plus repeated selections, complete state, replay/failure retirement and fresh consumed material.
7. Only then admit complete lifecycle measurements, including loading, dealer/OT/HE setup/delivery, all parties' memory/device staging, directional/all-link bytes, output and cleanup. Keep source online/projection figures separate from newly measured complete-request cohorts.

Highest-value additional oracle cases are PUMA's shift/clipping and sum-versus-mean counterexamples; BumbleBee epsilon tails, pivot neighbors, dropped-bit comparisons and approximate H2A support; SIGMA negative truncation/order and gap boundaries; FuseFSS all-mask public-shape invariance. Matching final tokens alone cannot validate these systems. No oracle or workload was executed in this audit.

## 8. Evidence boundaries and audit disposition

The [coverage audit](../TEN-METHOD-COVERAGE.md) remains the implementation inventory. Its measured negative results stay separate from missing original methods:

| Existing evidence | Permitted conclusion |
| --- | --- |
| Clear Qwen SIGMA-derived Q24→Q12 gate: zero exact logit/KV positions out of 48 | That frozen adaptation failed its exact W8A8-control screen; original SIGMA was not rejected. |
| Two-worker Qwen responses with unfused gates | Genuine local complete responses with trusted-client conversion/issuance and clear remaining operations, not private-model SIGMA or complete FuseFSS. |
| Fused interval/gate cohorts | Bounded helper correctness/cost, not a general compiler or source GPU transformer. |
| BumbleBee-derived interleaving lookup | Bounded older-source packing reuse, not final-paper OLT/bOLE/nonlinear/distributed-OT inference. |
| Clear packed-embedding requantization failure | Numeric failure of that adaptation; encrypted complete BumbleBee did not run. |
| PUMA pending names | No implementation, measurements, failure result or serving rank. |
| CKKS bounded softmax results/rejections | A different approximate-HE component; not these papers' shared-computation softmax. |

**Disposition:** retain the canonical draft's overall contract and its no-complete-implementation status. Incorporate the new source discrepancies and candidate defaults above before implementation approval. Most importantly, narrow “unknown PUMA coefficients/defaults” to the remaining evaluated-profile association, and expand BumbleBee's numeric blocker to include the newly pinned Q16/n=4 and ±3 conflicts. Source fidelity is not complete until those choices, full token boundaries and actual execution roles are bound; measured adaptation failures must not fill the missing-system cells.
