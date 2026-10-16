# Primary-source prior-art review

**Review cutoff: 2026-10-16.** Baseline supplied for this review: `99e10e2c`.
Scope: Paper 1 mechanism selection and Paper 2's coupled private lookup/full-head
composition. This review informs P1-C/P2-A; it does not close either publication
gate. Source byte hashes and retrieval outcomes are in `prior-art-sources.json`.
The dated source/code follow-up is [COMPETITOR-QUALIFICATION.md](COMPETITOR-QUALIFICATION.md);
its quarantined records supplement, rather than replace, the frozen primary records.

## 1. Main findings

1. **The generic Paper 2 architecture already has very close prior art.** Gupta,
   Katz, and Miers, *Private and Verifiable Outsourcing of Open-Weight LLM
   Inference*, ePrint 2026/1849, explicitly combines two-server private embedding
   lookup, additive sharing for public matrix–vector products, client-local
   nonlinear computation, and a lightweight client. Its embedding and linear
   protocols are composed in one inference protocol. It must be the first
   substantive comparator, not a peripheral verification citation.
2. **DPF lookup, full-domain evaluation, packed/early-terminated DPF evaluation,
   and additive public-matrix evaluation are not new primitives.** Tying the
   input/output weights does not change that algebra. Reusing an immutable weight
   owner can be a useful implementation contribution, but is not by itself a new
   cryptographic construction.
3. **A narrower systems contribution remains plausible, not established:**
   exact execution of a pinned quantized decoder, coupled vocabulary ownership
   alongside the prepared body, complete logits and original client sampler/full
   KV, and a measured all-link communication advantage under a whole-client
   256 MiB ceiling. The source-present/cold-client versus warm-client crossover,
   extra provider capacity, and preparation costs are central to that claim.
4. **Paper 1 also has close current evaluation/framework prior art.** Chen et al.
   explicitly report reproducibility analysis and standardized re-benchmarking;
   Das, Azad, and Saxena explicitly introduce evaluation contract cards and
   comparability gates. Al Badawi et al. propose a reporting card and reference
   implementation. Paper 1 should demonstrate its executable, resource/lifecycle
   accounting and independently implemented mechanisms, not claim invention of
   fair comparisons, contracts, reporting cards, or private-inference frameworks.
5. **No external system is defeated by this review.** Published speedups are
   author-reported results in their own contracts. Missing local implementations,
   failed bounded adaptations, and unavailable measurements are unresolved cells.

## 2. Review method and source identity

Read `PUBLICATION-PLAN.md`, `CLAIMS-AND-EVIDENCE.md`, `low-memory.md`,
`docs/data/research/paper-library.json`, `docs/data/research/implementations.json`,
and `paper/references.bib`. The local library supplied discovery/provenance;
technical assertions below use the papers themselves or explicitly labeled
author abstracts. The implementation inventory describes PLLM's coverage, not
the original papers' achieved performance.

Current discovery used arXiv searches for private/secure LLM inference, private
inference, and embedding/DPF; IACR searches for LLM, embedding, and distributed
point functions; backward references; and official PoPETs, NDSS, USENIX, ACL,
and PMLR records. Search pages were discovery only. Direct primary records were
opened for included new findings. This is a targeted review, not an exhaustive
systematic search or proof of absence. The initial literature pass did not acquire
upstream code. The 16 October follow-up retrieved selected revision-pinned author
source files as documentary text and retained hashes and bounded extracts. Upstream
code remains quarantined specification/oracle input: never imported, built,
executed or linked. No checkpoints or benchmarks were run by either review pass.

**Locators:** `PDF p.` means the PDF page, independent of proceedings pagination.
FastQuery has six PDF pages; unusual extracted control characters must not be
counted as additional pages. Section/figure identifiers further identify claims.
Unversioned cached PDFs are pinned by SHA-256, not silently assigned a revision.
Fresh HTML hashes identify retrieval snapshots, not immutable publisher content.

Important version distinctions:

| Source | Version used / primary record checked |
| --- | --- |
| BGI FSS | EUROCRYPT 2015 chapter; cached full text. |
| BGI improvements | CCS 2016 work, expanded ePrint 2018/707 text; cached full text. Cite the construction's publication and the text actually read separately. |
| Open-weight outsourcing | ePrint 2026/1849, received 2026-08-31, approved 2026-09-03. Current abstract checked; PDF re-fetch returned 403, so technical review uses the hash-pinned existing PDF. Byte identity with a fresh PDF is unverified. |
| Open-weight outsourcing code, follow-up | Public `kanav99/dvllm`, pinned commit `271637be7196d26a5433ca1e62b1a5357e769a5b` (2026-10-01). GitHub repository metadata links ePrint 2026/1849; selected source bytes, provenance and extracts are locked in the qualification archive. Identity with the authors' evaluated build is unresolved. |
| SIGMA | Final PoPETs 2024(4):61–79. Fresh PDF equals local cache byte-for-byte. |
| BumbleBee | Both cached ePrint 2023/1678 and freshly fetched NDSS 2025 final PDF; hashes differ. Use final-paper locators below. |
| FastQuery | arXiv:2405.16241v1, 2024-05-25; PDF identifies DAC 2024. |
| Compact | arXiv:2309.04664v2, 2024-03-17; official PoPETs 2024(3):25–41 metadata checked. |
| NEXUS | NDSS 2025 official PDF/record; paper title is *Secure Transformer Inference Made Non-interactive*. |
| THOR | ePrint 2024/1881; record received 2024-11-19, revised 2025-09-12; record reports CCS 2025. Cached PDF pinned independently. |
| FuseFSS / Maverick | arXiv:2606.09551v1, 2026-06-08 / arXiv:2609.10264v1, 2026-09-09. Preprints; no invented venue. |
| CacheMir / ROSETTA | arXiv:2602.11470v1 / arXiv:2609.16915v1; ROSETTA v1 dated 2026-09-15. |
| CrypTen / MPCache | arXiv:2109.00984v2, 2022-09-15 / arXiv:2501.06807v2, 2025-10-18. Original publication years remain distinct. |
| Das–Azad–Saxena | arXiv:2610.19607v1, primary history and PDF explicitly dated **2026-10-14**; included before cutoff. Do not infer the date from the identifier's sequence number. |
| Chen et al. / Kei–Chow | ePrint 2026/491, last revision 2026-07-20 / ePrint 2026/2005, last revision 2026-09-14. First reviewed here at abstract level; second also in cached full text. |

## 3. Paper 2: closest-prior-art table

“Not established” below means the inspected evidence does not answer that
specific question. It does not mean the system cannot support the feature.

| Priority / primary source and precise evidence | Actual overlap | Material differences / unanswered questions | Comparator disposition |
| --- | --- | --- | --- |
| **1. Gupta–Katz–Miers, [open-weight outsourcing](https://eprint.iacr.org/2026/1849).** Abstract; §4.1/Fig.1 public matvec, PDF pp.6–7; §4.3/Figs.3–4 embedding/verification, pp.9–10; §4.4/Fig.5 composition, pp.10–11; §5 pp.12–14. [Source qualification and code locators](COMPETITOR-QUALIFICATION.md). | Same two non-colluding public-weight servers handle private embedding and all linear layers; client reconstructs and computes remaining operations. **Dense XOR-PIR**, not an explicit DPF substitution or garbled lookup, is specified in §4.3. Selected author source confirms remote full-vocabulary head, client full KV and greedy/stochastic sampling. | Model owner publishes reusable commitments; client performs **online/batched verification**, including final rejection checks. Paper specifies nearest-integer fixed-point encoding at scale 2048, int16 weights/int64 vectors; inspected code uses truncating casts. Neither is unchanged PLLM W8A8/ring32. Public source is available; evaluated-build identity, exact Qwen state/sampler parity, whole-client memory and matched lifecycle costs remain unresolved. Narrow inspected-path RNG/abort concerns require follow-up, not an artifact-wide security verdict. **179 MB storage** and the **4 GB client VM** are not whole-client RSS measurements. | **Mandatory closest-system comparison.** Distinguish original numerical contract, proposed exact-W8A8 GKM-derived dense-PIR adaptation and stronger packed-DPF architecture control. Colocating lookup/head with each of two offset workers follows its party structure; existing PLLM components do not yet admit that complete composition. Missing/mismatched evidence is unresolved, not exclusion or defeat. |
| **2. Boyle–Gilboa–Ishai, [FSS improvements](https://eprint.iacr.org/2018/707).** Introduction pp.1–3; §3.2.3 pp.16–17, Fig.4, Theorem 3.12; original FSS chapter §1. | DPF evaluation shares sum/difference to a point function; two-server PIR privately selects a database entry. Full-domain traversal reduces repeated tree work; early termination packs several short output-group elements. | Generic primitive, not complete LLM inference, quantized embedding scales, tied-owner lifetime, or a 256 MiB deployment. Boolean/XOR lookup reconstruction is not automatically interchangeable with arithmetic head shares. | Foundational attribution for lookup and packed evaluation. Explain PLLM's row/scale encoding, PRG, output group, tail handling and fresh keys as implementation bindings, not invented DPF functionality. |
| **3. [FastQuery](https://arxiv.org/abs/2405.16241v1).** §2.1–2.4/Fig.2 p.2; §4.1–4.2/Figs.4–6 p.3; §4.3 and §5 pp.4–5. | Treats embedding lookup as a major private-inference cost; one-hot query optimization, low-bit mixed-width packing, HE output consolidation, bit-width alignment. | Client/server 2PC protects private model weights as well as input. Network/protocol co-optimization includes embedding quantization and data-free fine-tuning. Does not establish PLLM's exact tied full-head/client full-KV composition. Requantization can alter the model; looking up an embedding alone does not remove a client-resident tied head. | Closest boundary-specific mechanism. Compare equal numerical functions where possible; otherwise a quality-matched track. PLLM's bounded lookup adaptation is not FastQuery end-to-end performance. |
| **4. [BumbleBee](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf).** Final abstract/p.1; §II.D p.4; §III/Fig.6 pp.5–7; §V nonlinear protocols; §VII evaluation. | Complete two-party transformer system; explicitly motivates large embedding-table matrix multiplication. RLWE matrix products, ciphertext interleaving and nonlinear protocols; final paper reports real protocol execution, not simulation. | Client/server private-model 2PC rather than two public-weight boundary workers plus prepared body. Full system uses its own fixed-point/nonlinear conventions and hardware. Neither exact PLLM logit/KV identity nor whole-client 256 MiB eligibility is established here. | Strong complete-system qualification candidate and packing mechanism reference. No inference from PLLM's bounded interleaving result to BumbleBee's full model cost. |
| **5. [SIGMA](https://petsymposium.org/popets/2024/popets-2024-0107.php).** §2.2 p.3; FSS/DPF and truncation/nonlinear constructions §§3–5; implementation/evaluation §§6–7, pp.10–13. | GPU-accelerated FSS secure transformer inference; two online parties in preprocessing model. Relevant masked-wire, DPF, truncation and nonlinear techniques. | Preprocessing dealer and both online parties remain substantive roles. Secure nonlinear evaluation, model privacy and per-operation material differ from client-local nonlinear execution. Whole-client memory, decoder-state identity and a matched current serving run remain unmeasured here. | Required system context. Selected CPU helpers are not SIGMA reproduction. Do not import a third paper's assertion about SIGMA's decoder support as independently verified fact. |
| **6. [FuseFSS](https://arxiv.org/abs/2606.09551v1).** §3.1 p.3; compiled gates §§4–5, especially §4.6/Theorem 4.7 p.6; §6 and Appendix B.5 p.15. | Explicit two-party preprocessing setting, public architecture/model parameters, ring/Boolean types, compiled nonlinear gates and shape leakage. Particularly relevant to execution contracts and public-weight comparison. | Dealer-generated material, masked-wire conversions, exact declared fixed-point semantics and GPU regime differ. Its own Appendix B.5 labels BOLT/BumbleBee attempts as non-comparable best-effort outcomes. | Contemporary mandatory FSS reference. CPU region fusion is a mechanism adaptation; no whole-system ranking. Preserve public-shape leakage and preprocessing rather than treating them as free compiler details. |
| **7. [Maverick](https://arxiv.org/abs/2609.10264v1).** §1.3 p.2; code/delegation preliminaries §§3–4; outsourcing/evaluation later sections; abstract and p.4 related-work comparison. | Public-weight matrix–vector delegation with client-local nonlinear work and privacy/verifiability; explicitly compares with Gupta et al. | Single-server coding/LPN-based delegation, reusable preprocessing and verifiability differ from fresh additive sharing/non-colluding boundary pair. Client preprocessing matrices and field/code parameters matter. Local uncertified RAA controls do not establish the original scheme's insecurity or infeasibility. | Strong alternative systems point, separate security class. Retain as unresolved full-system comparator; current PLLM component has no qualifying decoder score. |
| **8. [Slalom](https://arxiv.org/abs/1806.03287v2).** §3/Fig.1 pp.4–6, privacy/integrity and preprocessing; evaluation §4. | Outsources linear work, uses masks/precomputation, verifies products with Freivalds, keeps nonlinear work in trusted domain. Direct ancestry of prepared masked-linear execution. | Original trusted hardware/SGX boundary and CNN-oriented evaluation are substantive. A software trusted-client adaptation changes deployment and storage. Integrity and privacy are separate guarantees. | Required ancestry and verified reference. Label current PLLM as an SDK adaptation; no claim to reproduce enclave/attestation or full Slalom performance. |
| **9. [NEXUS](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf).** §II.A p.2 and Appendix D p.17; §III matrix products; §IV/Algorithm 2 p.7; softmax p.8. | Non-interactive CKKS transformer inference, packing, nonlinear evaluation and secure Argmax. A current complete-system alternative, not just a softmax paper. | Protects a private model in a client/server HE setting. Secure Argmax is not the same output functionality as delivery of all logits for arbitrary local sampling. CKKS approximation and constant-shift softmax need a numerical/quality contract. Full-state decoder semantics and total client lifecycle cost are not established by PLLM's component. | Qualify separately; adapt output only with explicit attribution and changed costs. Failed bounded softmax admission does not defeat NEXUS. |
| **10. [THOR](https://eprint.iacr.org/2024/1881).** §§3–4 matrix multiplication; §5.1 and Algorithm 3, pp.9–10; evaluation §6. | HE transformer computation, packing and iterative approximate softmax; relevant alternative to FSS/hybrid pipelines. | CKKS ciphertext lifetimes, key material, bootstrapping and approximation are not represented by a bounded BFV/ring helper. Task/model and output semantics need alignment. | Contemporary encrypted-system reference; current local softmax adaptation cannot supply a complete-system score. |
| **11. [CacheMir](https://arxiv.org/abs/2602.11470v1) and [ROSETTA](https://arxiv.org/abs/2609.16915v1).** CacheMir §§2.2–2.3 pp.3–4, decode/bootstrapping sections and evaluation; ROSETTA §2.4 p.5, adaptive LUT/hybrid selection sections and pp.10–12 evaluation. | Specifically target autoregressive encrypted decoding with KV-cache work. ROSETTA combines CKKS/TFHE and chooses schemes for nonlinear workloads, extending the recent decode frontier beyond NEXUS/THOR. | Private model and client/server HE assumptions; approximate numerical functions. Neither complete-logit disclosure, exact fullKV parity with PLLM, nor 256 MiB whole-client eligibility is established here. These systems cannot be dismissed merely because older HE baselines are encoder-heavy. | Add to current-system qualification queue. No claimed win without matched quality, prefill/decode workload, client costs and hardware. |
| **12. [CipherGPT](https://eprint.iacr.org/2023/1147) and [PUMA](https://arxiv.org/abs/2307.12533).** CipherGPT overview and Algorithms 2–3/top-k/sampling/embedding, PDF pp.7–10; PUMA threat model p.4, §§4.3–4.4 pp.5–6 and §5.4 p.9. | Prior private generation and embedding handling. CipherGPT includes secure top-k and sampling; PUMA explicitly implements secure embedding from shared token indices. | CipherGPT is 2PC; PUMA uses three-party replicated sharing with an honest-majority assumption. Secure top-k/Argmax/sampling are different outputs from all-logit client reconstruction. Check exact rounding/distribution and active KV semantics before comparison. | Required generation context. “Five minutes” in PUMA's title is not a sustained-chat throughput measurement. Full original systems remain unmeasured locally. |

### 3.1 Additional current work and evidence limits

| Primary source / evidence actually read | Relevance and disposition |
| --- | --- |
| [Nimbus](https://arxiv.org/abs/2411.15707), abstract and §2.3/p.3; [Breaking the Layer Barrier](https://arxiv.org/abs/2508.19525), cached abstract/overview | HE/MPC computation placement and cross-layer optimization are prior systems ideas. Nimbus explicitly uses a two-party semi-honest client/server contract. Include when describing placement/fusion ancestry; exact boundary composition and low-client-memory eligibility remain unestablished. |
| [EncFormer](https://arxiv.org/abs/2604.09975v1), cached abstract and §2.3/p.2; [MOSAIC](https://arxiv.org/abs/2607.29221v1), primary abstract and cached overview | Further recent hybrid HE/MPC/fusion systems. EncFormer explicitly uses semi-honest client/server private-model inference. Retain as qualification candidates; no numerical or memory exclusion follows merely from a different implementation backend. |
| [Hyperion](https://eprint.iacr.org/2025/2318), **primary abstract only**, last revision 2026-07-03 | Private inverse-transform token sampling under HE. This prevents treating private non-Argmax token generation as an unaddressed problem. Paper 2's trusted-client sampler is a different placement/functionality choice; full sampling-distribution compatibility and head/lookup coverage were not audited. |
| [Tricycle](https://eprint.iacr.org/2025/1200), **primary abstract only**, revised 2026-09-14 | Tricyclic matrix encodings, GPU CKKS and statistical max estimation; abstract's full-system example is BERT. Relevant current packing/system work, not established full-KV decoder competitor in this review. Venue/DOI claims are as reported by the primary record; no invented proceedings details. |
| [Error Propagation-Aware Scale Design](https://eprint.iacr.org/2026/2242), **primary abstract only**, received 2026-09-28 | Numerical-error-aware CKKS scale/modulus/bootstrapping allocation. Modern precision tradeoffs need review before any blanket conclusion that HE cannot preserve useful model quality. No exact W8A8 or full-head inference established. |
| [Secure Speculative Decoding for Large Language Models](https://arxiv.org/abs/2610.08678v2), **primary abstract only**, v2 dated 2026-10-13 | **Excluded from cryptographic inference comparator set:** studies jailbreak/prompt-injection security of speculative decoding, not input-private MPC/HE inference. A “secure” title is insufficient evidence of cryptographic relevance. |
| [TruncFormer](https://arxiv.org/abs/2412.01042v1), abstract, §§1–3 PDF pp.1–4 | Explicitly describes **plaintext emulation** of private inference and static truncation placement. Relevant numerical and compiler mechanism; do not relabel emulated costs as a measured cryptographic deployment. |
| [MPCache](https://arxiv.org/abs/2501.06807v2), abstract and cached opening sections | Static eviction/dynamic cache selection intentionally changes attention/cache behavior. It is a quality-matched alternative, not an exact fullKV baseline. Its absence from the exact track is a functionality distinction, not a performance loss. |

Discovery also surfaced generic programmable DPFs, TEE partitioning, private
retrieval/vector search, and further HE/MPC work. These are not asserted as
complete matched competitors from search snippets. The manifest labels discovery
and mistaken URL resolution separately; absence of detailed review is retained.

### 3.2 Dated closest-comparator source qualification — 16 October 2026

[COMPETITOR-QUALIFICATION.md](COMPETITOR-QUALIFICATION.md) supplies the complete
matrix, source-to-construction mapping, inspected-path limitations, model-neutral
integration seams and measurement gates. Its [quarantined source-review archive](artifacts/source-review/2026-10-16-gkm-qualification-v1/README.md)
pins the author revision, exact code-file hashes, retrieval receipts, PDF-page
extracts and local file/line locators, with a read-only verifier. The source
discovery supplements the original primary manifest; an earlier missing link or
failed PDF request was not evidence that code was unavailable.

The paper's verified protocol and a disclosed semi-honest restriction both remain
substantive comparators. A model-neutral W8A8 adaptation must preserve their
complete lookup/body/head functionality and disclose changed representation,
rounding, PRG, batching and verification contracts. Replacing dense XOR-PIR with
packed DPF additionally needs BGI attribution; it is initially an architecture
control, not measured reproduction of Gupta–Katz–Miers. This documentary slice
closes source discovery and description corrections only, not P2-A or a performance
comparison.

## 4. Cryptographic novelty versus systems composition

### 4.1 Established algebra and techniques

- For a replicated public table `T`, DPF keys encode a private one-hot selector.
  Each server evaluates a share against the table; the client combines replies.
  BGI already supplies compact keys, full-domain traversal and packed early
  termination. Upload size, full-table server work, row/scale encoding and reply
  size must be counted separately.
- For public `W` and ring shares `h = a + b`, linearity gives
  `W a + W b = W h`. This applies to an output projection just as to any other
  public linear layer; the generic protocol already returns a vector, not merely
  an Argmax. Name the full head as a functional boundary, not a new primitive.
- A tied embedding/output matrix offers shared public storage. A single native
  owner, zero-copy views, parallel independent requests and residue serialization
  are engineering choices. Their value must be demonstrated through lifetime,
  communication and capacity measurements.
- Gupta–Katz–Miers already puts the lookup and linear protocols into the same
  inference algorithm (Fig.5). Their term “lightweight client” cannot be treated
  as a novelty gap just because PLLM defines a different, more explicit RSS tier.

### 4.2 What the candidate could contribute

The strongest supportable direction is **a constrained, exact systems design and
measurement result**: use two additional boundary workers to remove a tied
vocabulary matrix from the client while preserving the prepared body's contract,
then locate the measured communication/cold-latency tradeoff for complete
requests under a complete-client memory limit. This requires a contribution
beyond “we combined known operations”: demonstrably useful ownership/lifecycle
design, a sound multi-role composition, and a matched frontier improvement.

The following are unresolved research claims, not conclusions of the literature
review:

1. Whether that specific prepared-body/boundary composition has not already been
   implemented elsewhere. No reviewed paper was verified to implement every one
   of PLLM's pinned exact-state, lifecycle and measurement requirements; this is
   an observation about reviewed evidence, **not a firstness claim**.
2. Privacy against each allowed coalition across Preparation, Inference, Boundary
   A and Boundary B. Two separate pairwise non-collusion statements do not prove
   the joint execution. Include setup/publisher custody and erasure, correlated
   masks, reconstruction/scale messages, repeated token queries, cancellation,
   and declared shape/timing/access leakage.
3. Exact execution versus original-model quality. Integer reconstruction modulo
   `2^32` needs a signed output bound and the specified rounding/scaling order.
   Matching W8A8 logits/state is not matching BF16, FP32, a CKKS approximation,
   or a retrained/quantized alternative's quality.
4. The crossover between removing initial vocabulary delivery and adding recurring
   full-vocabulary responses. One cold short response does not establish a warm
   session win. Public source certificates, fresh queries, worker table storage
   and setup all belong to the relevant lifecycle cell.

## 5. Paper 1: representative independent mechanism set

Recommended main-text set: **six families**, with paper-specific variants nested
inside them. This covers trusted delegation, numerical adaptation, three distinct
nonlinear representations, encrypted boundary packing, FSS gates and HE
nonlinear execution without implying that every cited system was reproduced.
Status here comes from the current implementation inventory and remains subject
to exact-build replay/independent audit.

| Family and original evidence | PLLM implementation to describe | Preserved versus changed scope / publication treatment |
| --- | --- | --- |
| **1. Slalom-derived verified linear delegation.** Slalom §3/Fig.1 and Theorem 3.2. | `pllm.verification`, `runtime/client.py`; `benchmarks/research/slalom_qwen.py`. | SDK adaptation with trusted-client Freivalds and one-use masks; no SGX/attestation reproduction. An end-to-end adaptation only where the frozen workload completes, and a separate verified comparison class. |
| **2. SmoothQuant-derived public equalization.** [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html), scaling equations and Figs.4–5, PDF pp.3–4. | `runtime/public_equalization.py`; `benchmarks/research/smoothquant_qwen.py`. | Fixed `alpha=3/4`, bounded public scales and trusted-client scaling. Model/numeric adaptation, not privacy protocol. Calibration data, publication/delivery, original versus equalized identity and quality must remain explicit. |
| **3. Nonlinear representation alternatives:** [Compact](https://petsymposium.org/popets/2024/popets-2024-0065.php), §§3.1/4.2 density-aware fitting; [Curl](https://eprint.iacr.org/2024/1127), Figs.4/6 and shared-lookup discussion; [LogRow](https://eprint.iacr.org/2024/369), abstract and garbled-lookup construction. | `CompactGatedMultiplyQ12`, `CurlWaveletGatedMultiplyQ9`, `LogRowGatedMultiplyQ9`; `benchmarks/research/mixed_gated.py`. | Complete bounded gated operators, not full original systems. Compact's dyadic Chebyshev fit/backend adapts the simulated-annealing/MPC method. Curl uses public Haar compression with exact replacement truncation, not all original wavelet/float machinery. LogRow reference uses the earlier two-row setup, not the paper's optimized headline logarithmic-ciphertext construction. Charge table bits, conversion, selection and multiplication; the three numeric profiles are not automatically quality-equivalent. |
| **4. FSS masked-wire/gate compilation:** SIGMA §§3–5; FuseFSS §§3–5/Theorem 4.7. | `MaskedWireGatedMultiplyQ12`, compact DPF, GapARS/residual-SiLU and region-fusion references. | CPU/128-bit-seed helpers, public gap/width bounds, local issuance and trusted-client conversions differ from the full GPU systems. An installed complete PLLM response containing selected gates is still not full SIGMA/FuseFSS reproduction. Include fresh material and exact floor/rounding behavior. |
| **5. HE private embedding and output repacking:** FastQuery §§2.2/4.1–4.2; BumbleBee final §III/Fig.6. | `PrivateEmbeddingLookup`; `packed_embedding.rs`, `coefficient_repacking.rs`; `benchmarks/research/repacked_embedding.py`. | Bounded mixed-width one-hot lookup, extracted-LWE versus direct interleaving, signed conversion. Local random-OT/half-gates replace distributed setup/SiRNN. No trained FastQuery checkpoint or complete BumbleBee decoder. The tied client head can retain the entire vocabulary matrix even when input lookup is remote. |
| **6. Encrypted nonlinear controls:** NEXUS softmax/Argmax; THOR §5.1/Algorithm 3. | Bounded encrypted softmax references in the inventory. | Explicitly mechanism adaptations, not faithful full CKKS/bootstrap/decoder systems. Include key/context setup, conversions, approximation range and blocked geometries. A local exact-integer HE implementation does not inherit CKKS error/performance results. |

Appendix/context: Maverick coded delegation (uncertified local code parameters),
MPCache/CipherPrune changed-state alternatives, CrypTen/Cheetah/THE-X, and generic
garbling/correlation mechanisms. Adding more names is less useful than a complete
reimplementation card for each selected family.

**Card minimum:** original paper/version/hash and exact equations/figures;
functionality and protected assets; actual party/dealer/client roles; permitted
corruptions and leakage; ring/field/scales/rounding; changed backend/approximation;
implemented versus omitted steps; independent oracle; admitted geometry; full
setup/online/conversion/delivery/material costs; current code/build/configuration;
raw evidence and replay. Keep three labels: executable end-to-end adaptation,
executable component adaptation, analytical/blocked reference.

### 5.1 Closest Paper 1 publications

| Primary source | Evidence and consequence |
| --- | --- |
| Chen et al., [*SoK: Private Transformer-Based Model Inference*](https://eprint.iacr.org/2026/491), primary abstract and record, revision 2026-07-20 | Explicitly evaluates reproducibility and re-benchmarks representative solutions under standardized configurations. Full implementation inventory was not audited here. Paper 1 cannot claim the first reproducibility study or standardized comparison; assess concrete differences before submission. |
| Das, Azad, Saxena, [*A Graph-Based Analysis of Comparability, Scale, and Attribution in Private Transformer Inference*](https://arxiv.org/abs/2610.19607v1), §§1/4/5, PDF pp.1–7, Tables 3–4 | Audits 62 systems, defines a Private Inference Contract card and eight comparability gates, distinguishes fail/unknown/pass. Its “1 of 1891 pairs” figure is its own corpus result, not independently validated here. Contracts and missing-evidence treatment are direct overlap; PLLM's distinction must be executable measurements and specific resource/lifecycle results. |
| Al Badawi et al., [*SoK: Private LLM Inference using Approximate Homomorphic Encryption*](https://eprint.iacr.org/2026/935), primary abstract, revision 2026-06-03 | Proposes Private LLM Card System and POLARIS reference implementation, with current proof-of-concept BERT-Tiny/Mini. Further prior art for reporting standardization and executable references. Full system/proof not audited. |
| Kei–Chow, [*SoK: Private Transformer Inference Across Systems, Models, and Cryptography*](https://eprint.iacr.org/2026/2005), §§1.1–1.2 pp.2–3; §3.3 p.5; §§8.1–8.3 | Systematizes applicability/composition, phase availability, private execution structure, numerical adaptation and security boundaries. Use for cross-checking coverage; verify individual system assertions against those original papers before importing them. |

### 5.2 Older systems that must remain accurately characterized

- **CrypTen:** [primary v2](https://arxiv.org/abs/2109.00984v2), abstract and
  design/security discussion (§§3–5, especially §5.2). A flexible tensor/autograd secure
  MPC framework with GPU support and semi-honest security. Not a specific
  two-boundary-worker tied-head protocol, and not independently reproduced by
  merely using its approximation formulas. Its existence rules out a generic
  “first reusable private tensor framework” claim.
- **Cheetah:** [USENIX 2022 paper](https://www.usenix.org/system/files/sec22-huang-zhicong.pdf),
  abstract and §2.2 p.3. Two-party semi-honest DNN inference using efficient HE
  linear algebra and nonlinear protocols; original image-model evaluation does
  not establish autoregressive full-state LLM coverage. Its packing techniques
  are genuine ancestry, not a complete defeated LLM baseline.
- **THE-X:** [ACL 2022 paper](https://aclanthology.org/2022.findings-acl.277/),
  §3.1 pp.2–3 and §3.5/Fig.5 pp.4–5. Approximates/replaces transformer operations
  and applies a fine-tuning procedure. It belongs to an explicitly quality-matched
  model-adaptation track; it is not an exact unchanged-W8A8 comparator.

## 6. Bibliography audit and exact corrections for the parent

**`paper/references.bib` was not edited.** New entries at baseline lines 203–252
were checked against primary records and/or full-text title pages. This is not
an audit of every older citation or the external URLs in PLLM's own artifacts.

### 6.1 Required corrections

1. **`zhang2025nexus`, line 247:** replace
   `title = {{NEXUS}: Secure Transformer Inference Made Non-interactive}` with
   `title = {Secure Transformer Inference Made Non-interactive}`.
   The official NDSS title page/record does not prefix the title with NEXUS.
   NEXUS remains the system name in prose. Listed authors and 2025 venue match
   the primary title page. DOI: `10.14722/ndss.2025.230868`.
2. **`zeng2025mpcache`, lines 190–193:** bind the citation to v2 if using the
   current cached paper. Exact primary v2 fields:

   ```bibtex
   author = {Wenxuan Zeng and Ye Dong and Jinjin Zhou and Jin Tan and Lei Wang and Tao Wei and Runsheng Wang and Meng Li},
   title = {{MPCache}: {MPC}-Friendly {KV} Cache Eviction for Efficient Private {LLM} Inference},
   year = {2025},
   url = {https://arxiv.org/abs/2501.06807v2}
   ```

   Current entry adds Junming Ma, omits Lei Wang/Tao Wei, changes author order,
   and spells out “Large Language Model” where the primary title uses “LLM”.
3. **BumbleBee mechanism locator in the inventory:** the freshly checked final
   NDSS paper's InterLeave/Fig.6 is in **§III, PDF pp.5–7**, not §IV. §IV begins
   *Faster Batch OLE from RLWE* on p.8. Bind the local adaptation card to one
   version; do not transplant section numbers between ePrint and proceedings.
   This is a source-map correction, not a shared-file edit performed here.

### 6.2 Accurate new entries; optional publication normalization

| Key | Finding / exact metadata recommendation |
| --- | --- |
| `knott2021crypten` | Title, all six authors (including **Shobha Venkataraman**) and NeurIPS 2021 identity agree with the primary paper. The v2 record is dated 2022-09-15; keep original venue year 2021 and pin v2 separately when using that text. |
| `heath2024logrow` | Title/authors/year and ePrint 2024/369 agree. “LogRow” is the local mechanism label, not part of the paper title. Do not add an unverified proceedings venue. |
| `gupta2024sigma` | Title/authors correct. Prefer `@article`, journal `Proceedings on Privacy Enhancing Technologies`, volume `2024`, number `4`, pages `61--79`, DOI `10.56553/popets-2024-0107`, year `2024`; official URL already correct. |
| `ma2026fusefss` | Title, Yuhan Ma/Yong Li/Stefan Schmid, year and identifier agree with arXiv v1. Pin `https://arxiv.org/abs/2606.09551v1`; retain preprint status. |
| `santos2024curl` | Title and seven-author order agree with ePrint 2024/1127; received 2024-07-10, revised 2024-09-18. No invented later venue/year. |
| `islam2024compact` | Title/authors correct; 2024 is valid proceedings year but the unversioned arXiv record starts in 2023. Prefer `@article`, journal `Proceedings on Privacy Enhancing Technologies`, volume `2024`, number `3`, pages `25--41`, DOI `10.56553/popets-2024-0065`, official URL `https://petsymposium.org/popets/2024/popets-2024-0065.php`; separately pin arXiv v2 if that is the text used. |
| `huang2022cheetah`, `chen2022thex` | Primary records confirm the older cited author lists. In particular **Li Dong** in `paper/references.bib` is correct for THE-X; the research library's **Dong Li** is not the ACL record's ordering of that person's name. Do not copy that library metadata back into the bibliography. |
| `hudson2026pllmframework` | Keep `@unpublished`, working-manuscript date and explicit absence of an assigned arXiv identifier. No fabricated identifier, DOI, venue or acceptance. This is an internal status assertion, not an externally published primary-paper record. |

### 6.3 Missing citations needed for the argument

Add only those used in the final manuscript, with these verified identities:

- **Kanav Gupta, Jonathan Katz, Ian Miers.** *Private and Verifiable Outsourcing
  of Open-Weight LLM Inference*. ePrint **2026/1849**, 2026.
- **Elette Boyle, Niv Gilboa, Yuval Ishai.** *Function Secret Sharing*.
  EUROCRYPT 2015, DOI **10.1007/978-3-662-46803-6_12**; and *Function Secret
  Sharing: Improvements and Extensions*, CCS 2016, DOI
  **10.1145/2976749.2978429**, expanded text ePrint **2018/707**. The original
  Gilboa–Ishai DPF work is cited by these sources; this review did not verify
  its full primary text and does not supply a guessed ePrint identifier.
- **Chenqi Lin, Tianshi Xu, Zebin Yang, Runsheng Wang, Ru Huang, Meng Li.**
  *FastQuery: Communication-efficient Embedding Table Query for Private LLM
  Inference*. DAC 2024; arXiv **2405.16241v1**. DOI
  **10.1145/3649329.3657374** appears in the primary PDF.
- **Wen-jie Lu, Zhicong Huang, Zhen Gu, Jingyu Li, Jian Liu, Cheng Hong,
  Kui Ren, Tao Wei, WenGuang Chen.** *BumbleBee: Secure Two-party Inference
  Framework for Large Transformers*. NDSS 2025, DOI
  **10.14722/ndss.2025.230057**. Do not use the 2023 ePrint year as the final
  venue year; retain both when needed.
- **Jungho Moon, Dongwoo Yoo, Xiaoqian Jiang, Miran Kim.** *THOR: Secure
  Transformer Inference with Homomorphic Encryption*. ePrint **2024/1881**;
  current record reports CCS 2025, DOI **10.1145/3719027.3765150**.
- **Souvik Das, Ariful Azad, Nitesh Saxena.** *A Graph-Based Analysis of
  Comparability, Scale, and Attribution in Private Transformer Inference*.
  arXiv **2610.19607v1**, 2026-10-14.
- **Yuntian Chen, Tianpei Lu, Zhanyong Tang, Bingsheng Zhang, Zhiying Shi,
  Yuxiang Luan, Zhuzhu Wang.** *SoK: Private Transformer-Based Model Inference*.
  ePrint **2026/491**. The primary record reports USENIX Security; use the
  checked ePrint version unless the final venue record is independently pinned.
- **Ahmad Al Badawi, Andreea Alexandru, Yuriy Polyakov, Vinod Vaikuntanathan.**
  *SoK: Private LLM Inference using Approximate Homomorphic Encryption*.
  ePrint **2026/935**, revision 2026-06-03.
- **Andes Y. L. Kei, Sherman S. M. Chow.** *SoK: Private Transformer Inference
  Across Systems, Models, and Cryptography*. ePrint **2026/2005**.

## 7. Defensible contribution wording and remaining decisions

Suggested **Paper 1** wording, conditional on completed evidence:

> PLLM provides executable contracts and independently implemented mechanism
> adaptations for resource-constrained private inference. It measures complete
> client/provider ownership and setup-through-cleanup costs under pinned numerical,
> security and serving contracts, and demonstrates [measured findings].

Cite the current SoKs/frameworks; state the six-family coverage and deviations.
No “first framework”, “first fair benchmark” or claim that the field ignores
client memory is supported by this review.

Suggested **Paper 2** wording, conditional on completed evidence:

> Building on DPF-based lookup and shared public-weight delegation, we implement
> coupled vocabulary-boundary execution alongside a prepared decoder. For [pinned
> model/workload/security/lifecycle], it preserves complete quantized logits,
> client sampling and full decoder state while reducing [measured complete-link
> metric] against [named matched baselines], within a measured 256 MiB whole-client
> ceiling, with [decode/first-token/provider-resource tradeoffs].

Until external coverage is sufficient, use **“best among evaluated compatible
configurations”**. Do not claim new cryptographic primitives, first two-server
private LLM inference, first lookup/linear composition, universal low-memory
superiority, or state of the art. Numerical incompatibility is not a reason to
discard quality-matched competitors from broader practical claims.

Outstanding decisions for manuscript integration:

1. Use the new Gupta–Katz–Miers qualification card and pinned source evidence.
   Source supports full logits/full KV; exact Qwen parity, tied-owner lifetime,
   whole-client memory and lifecycle still need independently implemented,
   authorized conformance and measurement. Upstream code remains specification/oracle
   input only. Verification differences stay visible rather than silently removed.
2. Resolve Paper 1's overlap with Chen et al.'s standardized re-benchmarking and
   the current contract-card SoKs beyond abstract-level comparison.
3. Freeze functionality classes: exact W8A8/full-state versus task-quality-matched;
   full logits versus token-only; fullKV versus eviction; verified versus
   semi-honest; private-model versus public-weight. Pin genuine security
   implications, not only paper labels or online party counts.
4. Qualify current complete systems (especially recent decoding systems), or mark
   them **not reproduced / unknown**. No bounded-component result closes this.
5. Complete the multi-role proof and implementation binding. Fresh randomness,
   ring bounds, scale payloads, owner reuse and erasure need joint treatment.
6. Run the parent-owned matched campaign and confirmation: whole-client maxima,
   full directed traffic, preparation, cold/warm/source-cold costs, output-length
   crossover, same total provider/access capacity, representative prompts and
   task quality. This review supplies no new performance measurements.

Historical 23.70% traffic / 24.18% request-time reductions remain the particular
archived cohort's findings, with its 32.68% slower decode and 8.00% later first
text. The literature review neither promotes those results to a current global
comparison nor invalidates the cohort's scoped measurements.
