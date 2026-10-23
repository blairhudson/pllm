# PLLM paper-to-module landing map

Original map: 23 September 2026. **85 paper placements**, including the subsequently audited SmoothQuant adaptation.

These are architecture proposals based on the supplied module descriptions, not existing API names, implemented capabilities, security endorsements or a repository audit. All papers retain their locked `pllm.research` records. Source evidence levels and citation edges are preserved from the original catalog; this mapping does not upgrade any evidence.

## Design recommendation

Keep the existing public module boundaries. Map reusable methods to component owners; map complete, implemented configurations to reviewed profiles. A paper can supply several methods and a method can have several sources.

Promote explicit representation conversions and decoder state transitions to typed contracts inside the existing modules before adding new top-level packages. Keep matrix-only `pllm.kernels` and current integer-native interfaces narrow; other cryptographic backends require separate reviewed capabilities.

Separate approximation, protected evaluation and scheduling. Separate correlation semantics, offline production and runtime material consumption. Protocols with different online parties, assumptions, ownership or numeric domains are not interchangeable because shapes match.

## Immediate work order

1. Specify representation, material-lifetime, operation-coverage and output-release contracts.
2. Prototype Maverick/EMVP/trapdoored matrix delegation independently against the prepared-masked baseline.
3. Add explicit FSS primitive/conversion boundaries and one FuseFSS-style scalar compilation experiment in its supported topology.
4. Exercise a complete decoder lifecycle, private selection and persistent KV behavior before optimising token throughput.
5. Keep HE, multiparty and TEE systems as separately gated provider/profile candidates.

## Contract ownership

| Concern | Owner |
|---|---|
| Research provenance, attacks and source status | `pllm.research` |
| Locked semantic, numeric, privacy and representation contracts | `pllm.plan` |
| Protected representation conversions and party exchanges | `pllm.protocols` |
| Mathematical approximation / supported protected nonlinear method | `pllm.nonlinear` |
| Correlation distribution and share-ownership specification | `pllm.correlation` |
| Offline production and material inventories | `pllm.preparation` |
| Mutable sessions, secure material leases and consumption | `pllm.runtime` |
| Decoder-state placement/retention descriptors | `pllm.state` |
| Semantic transformations and scheme assignment | `pllm.passes` |
| Ordering independent compatible work | `pllm.schedulers` |
| Validation and approved native lowering | `pllm.compiler` |
| Required integrity statement, soundness and release requirements | `pllm.verification` with `pllm.plan` |
| Provider manifests and approved factory loading | `pllm.providers` |
| Concrete component registration and construction | `pllm.components` |
| Reviewed combinations, not monolithic paper implementations | `pllm.profiles` and `pllm.pipeline` |
| Numeric units/directions and comparable measured outcomes | `pllm.metrics`, `pllm.evidence`, `pllm.search` |

## Full paper mapping

### `pllm.quantization`

#### [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html) (2023)

**Paper ID:** `smoothquant` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

- **Suggested contribution:** Public offline channel equalization before W8A8 quantization.
- **First experiment:** Source-locked public calibration and matched quantized-model quality and cost.
- **Promotion gate:** Separate adapted numeric identity and calibration from original SmoothQuant hardware results.
- **Original evidence boundary:** PLLM fixes alpha=3/4, normalizes and clips scales, and applies activation scaling at the client; this is a bounded adaptation, not the original fused GPU implementation.

### `pllm.correlation`

#### [Compressing Correlations via Secret Replication: PCFs from Symmetric Cryptography](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITC.2026.7) (2026)

**Paper ID:** `pcf-secret-replication` · **Track:** `preprocessing_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** The construction covers specified correlations and domains; it is not a drop-in secure LLM evaluator.

#### [Efficient Pseudorandom Correlation Generators for Any Finite Field](https://eprint.iacr.org/2025/169) (2025)

**Paper ID:** `finite-field-pcg` · **Track:** `preprocessing_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Choose by arithmetic domain and downstream protocol rather than treating field and ring encodings as interchangeable.

#### [Efficient Pseudorandom Correlation Generators over Z/p^k Z](https://link.springer.com/chapter/10.1007/978-3-032-01884-7_7) (2025)

**Paper ID:** `ring-pcg` · **Track:** `preprocessing_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** A PCG supplies correlated material, not an entire delegated nonlinear inference protocol.

#### [Compressing Unit-Vector Correlations via Sparse Pseudorandom Generators](https://doi.org/10.1007/978-3-031-68397-8_11) (2024)

**Paper ID:** `sparse-unit-correlations` · **Track:** `preprocessing_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Reference-discovered; evaluate the exact correlation domain before proposing an implementation.

#### [MoZZarella: Efficient Vector-OLE and Zero-Knowledge Proofs over Z2^k](https://doi.org/10.1007/978-3-031-15985-5_12) (2022)

**Paper ID:** `mozzarella` · **Track:** `preprocessing_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Title/publication verified through author metadata and citations; full proof not reviewed.

#### [Efficient Pseudorandom Correlation Generators from Ring-LPN](https://doi.org/10.1007/978-3-030-56880-1_14) (2020)

**Paper ID:** `ring-lpn-pcg` · **Track:** `preprocessing_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Ring-LPN assumptions must match the implemented construction and parameters.

#### [Efficient Pseudorandom Correlation Generators: Silent OT Extension and More](https://eprint.iacr.org/2019/448) (2019)

**Paper ID:** `silent-ot` · **Track:** `preprocessing_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Quantify local expansion, setup and security assumptions rather than counting only seed bytes.

#### [Compressing Vector OLE](https://eprint.iacr.org/2019/273) (2018)

**Paper ID:** `compressing-vole` · **Track:** `preprocessing_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Typed correlation sources: correlation kind, domain, recipient shares, distribution, authentication and valid reuse policy.

**First experiment:** Expand material and consume it in one explicitly compatible protocol; count setup, local expansion, storage and communication.

**Promotion gate:** Neither seed compression nor a matching tensor shape proves compatibility; finite fields and integer rings require separate contracts.

**Original evidence boundary:** Correlated secret seeds are not generic public mask compression.

### `pllm.deployment`

#### [Bifrost: Hybrid TEE–FHE Inference for Privacy-Preserving Transformer and LLM Serving](https://arxiv.org/html/2606.17421v1) (2026)

**Paper ID:** `bifrost` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.roles`, `pllm.protocols`, `pllm.state`, `pllm.runtime`, `pllm.providers`, `pllm.profiles`.

**Suggested contribution:** TEE/FHE deployment profile with declared trusted enclave execution and sensitive state ownership.

**First experiment:** Check attestation policy and key provisioning before evaluating performance.

**Promotion gate:** Declaring an enclave role is not attesting it; do not silently substitute hardware trust for cryptographic trust.

**Original evidence boundary:** A useful product-design branch, but it explicitly changes the hardware-trust assumption.

### `pllm.nonlinear`

#### [SHAFT: Secure, Handy, Accurate and Fast Transformer Inference](https://www.ndss-symposium.org/ndss-paper/shaft-secure-handy-accurate-and-fast-transformer-inference/) (2025)

**Paper ID:** `shaft` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.preparation`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Secure numerical operator candidates with explicit approximation, fixed-point and sharing-domain contracts.

**First experiment:** Check scalar/vector accuracy and secure evaluation at signed boundaries, then full-model quality.

**Promotion gate:** A mathematical approximation alone is not a private evaluation protocol.

**Original evidence boundary:** Approximation quality, clipping assumptions and the chosen MPC backend matter.

#### [Compact: Approximating Complex Activation Functions for Secure Computation](https://arxiv.org/html/2309.04664) (2024)

**Paper ID:** `compact` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.passes`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Piecewise approximation method with a locked input domain and numeric error contract.

**First experiment:** Plaintext approximation error first, then private-evaluator cost and generation-quality impact.

**Promotion gate:** A faster approximation is not evidence that its interval selection is private.

**Original evidence boundary:** Scalar approximation error does not establish long-form generation quality.

#### [Curl: Private LLMs through Wavelet-Encoded Look-Up Tables](https://nillion.pub/curl-private-llms-through-dwt-lut.pdf) (2024)

**Paper ID:** `curl` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.preparation`, `pllm.passes`, `pllm.schedulers`, `pllm.evidence`.

**Suggested contribution:** Wavelet-encoded nonlinear representation, separately composed with private lookup and reconstruction.

**First experiment:** Account for represented table bytes, private index selection, reconstruction and numeric error.

**Promotion gate:** Function compression and private table access are distinct obligations.

**Original evidence boundary:** Function compression does not by itself hide lookup indices or pay for private reconstruction.

#### [LLAMA: A Low Latency Math Library for Secure Inference](https://eprint.iacr.org/2022/793) (2022)

**Paper ID:** `llama-math` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.preparation`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Secure numerical operator candidates with explicit approximation, fixed-point and sharing-domain contracts.

**First experiment:** Check scalar/vector accuracy and secure evaluation at signed boundaries, then full-model quality.

**Promotion gate:** A mathematical approximation alone is not a private evaluation protocol.

**Original evidence boundary:** LLAMA here is the cryptographic math library, not Meta’s Llama model.

#### [SIRNN: A Math Library for Secure RNN Inference](https://eprint.iacr.org/2021/459) (2021)

**Paper ID:** `sirnn` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.preparation`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Secure numerical operator candidates with explicit approximation, fixed-point and sharing-domain contracts.

**First experiment:** Check scalar/vector accuracy and secure evaluation at signed boundaries, then full-model quality.

**Promotion gate:** A mathematical approximation alone is not a private evaluation protocol.

**Original evidence boundary:** RNN-era workloads and two-party assumptions should remain explicit.

### `pllm.passes`

#### [ATLAS: Automated Approximation of Transformers for Efficient Homomorphic Inference in One Hour](https://arxiv.org/html/2607.23478v2) (2026)

**Paper ID:** `atlas` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.models`, `pllm.sources`, `pllm.plan`, `pllm.nonlinear`, `pllm.evidence`.

**Suggested contribution:** Explicit model/approximation transformation records and separately identified adapted checkpoints.

**First experiment:** Compare adapted cleartext and protected execution, plus the original model quality baseline.

**Promotion gate:** Training/distillation is an external artifact-producing workflow, not hidden I/O during semantic lowering.

**Original evidence boundary:** Discovered and title/version checked; not substantively evaluated in this search. A follow-up reading item.

#### [EncFormer: Secure and Efficient Transformer Inference over Encrypted Data](https://arxiv.org/html/2604.09975v1) (2026)

**Paper ID:** `encformer` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.compiler`, `pllm.plan`, `pllm.protocols`, `pllm.metrics`, `pllm.search`.

**Suggested contribution:** Representation/placement assignment and conversion-aware region transformations.

**First experiment:** Compare full multi-operator paths including boundary costs, not isolated operator minima.

**Promotion gate:** Only pre-reviewed conversion edges and compatible trust/precision contracts are eligible.

**Original evidence boundary:** A cost-model or packing improvement must be checked across complete layer boundaries and workloads.

#### [FuseFSS: Efficient Secure LLM Inference with Function Secret Sharing](https://arxiv.org/html/2606.09551v1) (2026)

**Paper ID:** `fusefss` · **Track:** `priority_prototype` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.protocols`, `pllm.preparation`, `pllm.correlation`, `pllm.compiler`, `pllm.plan`.

**Suggested contribution:** Typed scalar-operator lowering and mask-aware predicate/coefficient fusion, paired with a compatible FSS backend.

**First experiment:** Compile one fixed-point nonlinear operator; check wraparound, signedness, mask-independent public shapes and complete material consumption.

**Promotion gate:** The rewrite requires the exact sharing domains and online party topology; fusion must preserve transcript and freshness obligations.

**Original evidence boundary:** Operator fusion does not remove the preprocessing role or cooperating online computation parties.

#### [ROSETTA: Efficient and Accurate Privacy-Preserving LLM Decoding via Hybrid CKKS/TFHE Evaluation](https://arxiv.org/html/2609.16915v1) (2026)

**Paper ID:** `rosetta` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.protocols`, `pllm.plan`, `pllm.compiler`, `pllm.state`, `pllm.evidence`.

**Suggested contribution:** Joint scheme-assignment optimization plus segmented nonlinear LUT evaluation and explicit CKKS/TFHE conversions.

**First experiment:** A complete decode path with conversion/level costs; evaluate the planner idea separately from implementing HE.

**Promotion gate:** Do not treat scheme assignment as ordinary scheduling or assume existing integer-native APIs execute ciphertexts.

**Original evidence boundary:** September 2026 preprint; reported 1.5–2.1x end-to-end gains are against Cachemir, not plaintext inference. Not independently reproduced.

##### ROSETTA evaluated-source recovery (23 October 2026)

The locked `arXiv:2609.16915v1` PDF (SHA-256
`b4c7195d32808881b2da8b3a6876b571d8215bbc02aaafaa78ef327aa27fb4e8`)
names [ROSETTA_artifact-B77F](https://anonymous.4open.science/r/ROSETTA_artifact-B77F/).
Fresh retrieval of its [public file-tree API](https://anonymous.4open.science/api/repo/ROSETTA_artifact-B77F/files/)
returned **HTTP 410**, exact body `{"error":"repository_expired"}`; the landing
page's HTTP 200 was only an application shell. Author-linked public listings and
scoped archive queries yielded no replacement evaluated release. These are
retrieval findings, not proof of universal artifact absence. Direct response
bodies totaled 1,281,081 bytes, plus search-result metadata; no upstream code ran.

The author-associated `rotation-sandbox` lead is Cachemir-labelled, not an
identified ROSETTA evaluated revision; see the
[CacheMir audit](#cachemir-source-recovery-and-implementation-decision-sheet).
Still needed: immutable evaluated source/config binding exact LUT boundaries,
coefficients/scales and endpoint/floor behavior; HomComp encoding; the actual
65536-RLWE → 2048-RLWE → 1024-LWE secret/key-switch graph, decomposition/noise and
recovery; and version-bound nonlinear/calibration and conversion/level costs.
The [ROSETTA contract](../../paper/methods/encrypted-computation-contracts.md#7-rosetta-contract)
is not completed by surrogate public-LUT PBS or by inheriting a different
CacheMir release's numerics. No profile/backend adoption follows.

#### [Breaking the Layer Barrier: Remodeling Private Transformer Inference with Hybrid CKKS and MPC](https://www.usenix.org/conference/usenixsecurity25/presentation/xu-tianshi) (2025)

**Paper ID:** `breaking-layer-barrier` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.compiler`, `pllm.plan`, `pllm.protocols`, `pllm.metrics`, `pllm.search`.

**Suggested contribution:** Representation/placement assignment and conversion-aware region transformations.

**First experiment:** Compare full multi-operator paths including boundary costs, not isolated operator minima.

**Promotion gate:** Only pre-reviewed conversion edges and compatible trust/precision contracts are eligible.

**Original evidence boundary:** Its benchmark speedups are relative to specific competing secure systems, not plaintext inference.

#### [CipherPrune: Efficient and Scalable Private Transformer Inference](https://arxiv.org/html/2502.16782v2) (2025)

**Paper ID:** `cipherprune` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.models`, `pllm.protocols`, `pllm.runtime`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Explicit token-pruning and degree-selection transformations with protected dynamic decisions where required.

**First experiment:** Check quality and observable execution shape after pruning.

**Promotion gate:** Lossy semantic changes and token-dependent leakage must be explicit.

**Original evidence boundary:** The basic pruning mask is opened; optional oblivious compaction hides original locations but discloses retained counts, and polynomial reduction opens a compacted-order mask. Learned thresholds/model quality and this leakage need explicit contracts.

#### [ReDASH: Fast and efficient Scaling in Arithmetic Garbled Circuits for Secure Outsourced Inference](https://arxiv.org/html/2506.14489v1) (2025)

**Paper ID:** `redash` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.nonlinear`, `pllm.preparation`, `pllm.plan`, `pllm.compiler`.

**Suggested contribution:** Scaling policy plus explicit residue/scaling conversion operations in the garbled computation.

**First experiment:** Signed rescaling at modulus boundaries, followed by a complete supported model fragment.

**Promotion gate:** Keep policy decisions separate from executable scaling/conversion semantics.

**Original evidence boundary:** Not a complete decoder-LLM evaluation. Follow its references to Garbled Neural Networks Are Practical.

#### [Ditto: Quantization-Aware Secure Inference of Transformers upon MPC](https://arxiv.org/html/2405.05525v1) (2024)

**Paper ID:** `ditto` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.models`, `pllm.sources`, `pllm.plan`, `pllm.nonlinear`, `pllm.evidence`.

**Suggested contribution:** Explicit model/approximation transformation records and separately identified adapted checkpoints.

**First experiment:** Compare adapted cleartext and protected execution, plus the original model quality baseline.

**Promotion gate:** Training/distillation is an external artifact-producing workflow, not hidden I/O during semantic lowering.

**Original evidence boundary:** Three-party honest-majority setting; adapted model evidence is not unchanged-model fidelity.

#### [MPC-Minimized Secure LLM Inference](https://arxiv.org/html/2408.03561v1) (2024)

**Paper ID:** `mpc-minimized` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.plan`, `pllm.runtime`.

**Suggested contribution:** Client/server model partitioning with explicit model-visibility and client-compute contracts.

**First experiment:** Measure the complete trusted client workload as well as reduced MPC execution.

**Promotion gate:** Do not hide extra client-side model work inside a scheduler or kernel.

**Original evidence boundary:** Changes the model-visibility and client-computation contract; not interchangeable with fully remote inference.

#### [MPCFormer: Fast, Performant and Private Transformer Inference With MPC](https://arxiv.org/abs/2211.01452) (2023)

**Paper ID:** `mpcformer` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.models`, `pllm.sources`, `pllm.plan`, `pllm.nonlinear`, `pllm.evidence`.

**Suggested contribution:** Explicit model/approximation transformation records and separately identified adapted checkpoints.

**First experiment:** Compare adapted cleartext and protected execution, plus the original model quality baseline.

**Promotion gate:** Training/distillation is an external artifact-producing workflow, not hidden I/O during semantic lowering.

**Original evidence boundary:** Reference-discovered; original construction not substantively reviewed in this pass.

#### [HyCC: Compilation of Hybrid Protocols for Practical Secure Computation](https://encrypto.de/papers/BDKKS18.pdf) (2018)

**Paper ID:** `hycc` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.compiler`, `pllm.plan`, `pllm.protocols`, `pllm.metrics`, `pllm.search`.

**Suggested contribution:** Representation/placement assignment and conversion-aware region transformations.

**First experiment:** Compare full multi-operator paths including boundary costs, not isolated operator minima.

**Promotion gate:** Only pre-reviewed conversion edges and compatible trust/precision contracts are eligible.

**Original evidence boundary:** Conversions, protocol security and online party requirements must compose.

#### [Factored Edge-Valued Binary Decision Diagrams](https://link.springer.com/article/10.1023/A:1008691605584) (1997)

**Paper ID:** `fevbdd` · **Track:** `representation_only` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.plan`, `pllm.research`.

**Suggested contribution:** Optional public function/decision-diagram factoring transformation.

**First experiment:** Check semantic equivalence and representation size, then measure any downstream secure evaluation.

**Promotion gate:** Factoring has no standalone privacy claim.

**Original evidence boundary:** Not a privacy primitive. Deeper traversal of generic decision-diagram literature was relevance-pruned.

### `pllm.protocols`

#### [Duty-Free Bits: Projectivizing Garbling Schemes](https://eprint.iacr.org/2026/476) (2026)

**Paper ID:** `duty-free-bits` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.plan`, `pllm.compiler`, `pllm.preparation`, `pllm.correlation`.

**Suggested contribution:** Typed garbling-label conversion capability between the specific source and target encodings.

**First experiment:** One supported conversion followed by its consuming garbled operation.

**Promotion gate:** Encoding, key ownership and protocol assumptions must match; this is not a byte-format conversion.

**Original evidence boundary:** Abstract available; full PDF was not accessible in this search. Bibliography traversal therefore incomplete.

#### [MOSAIC: Masked Outsourcing of Secure AI Computations](https://arxiv.org/html/2607.29221v1) (2026)

**Paper ID:** `mosaic` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Masked-outsourcing protocol and a separately identified trusted-accelerator deployment.

**First experiment:** Measure trusted compute/memory and communication under the intended remote network, not only local accelerator links.

**Promotion gate:** Keep substantial trusted hardware requirements out of the minimal-client default.

**Original evidence boundary:** Measured setup uses one trusted GPU and three untrusted GPUs over NVLink. A low trusted-compute fraction does not imply a phone-sized client or low-WAN-traffic service.

#### [Maverick: Private and Verifiable LLM Inference Made Practical via Matrix-Vector Multiplication Delegation](https://arxiv.org/abs/2609.10264) (2026)

**Paper ID:** `maverick` · **Track:** `priority_prototype` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.verification`, `pllm.correlation`, `pllm.preparation`, `pllm.kernels`, `pllm.runtime`.

**Suggested contribution:** Separate matrix-delegation privacy and coding-based verification capabilities, with an explicitly reviewed composition.

**First experiment:** One matrix-vector product against prepared masked-linear: client memory and work, bytes, rounds, preparation, and malicious-result rejection.

**Promotion gate:** Review the exact masking and verification constructions, numeric domain, repeated-query security and composition before accepting a profile.

**Original evidence boundary:** The Qwen3-4B benchmark takes eight prompt tokens and returns next-token logits. Its 15.46 tokens/s boost figure excludes communication and is not sustained generated-token throughput. Client storage and trusted local work remain.

#### [Oblivious Ciphertext Compression via Linear Codes](https://eprint.iacr.org/2026/329) (2026)

**Paper ID:** `oblivious-compression` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.plan`, `pllm.compiler`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Protocol-specific encrypted sparse-vector representation transform, not generic transport compression.

**First experiment:** Round-trip the supported encrypted sparse domain and reject configurations without the required bound.

**Promotion gate:** The sparsity guarantee concerns the protected plaintext domain; do not assume a masked dense activation satisfies it.

**Original evidence boundary:** The sparse/input-domain condition is essential; not arbitrary dense ciphertext compression. Full PDF not available in this search.

#### [Private and Verifiable Outsourcing of Open-Weight LLM Inference](https://eprint.iacr.org/2026/1849) (2026)

**Paper ID:** `open-weight-verifiable` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.verification`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** Two-non-colluding-server private/verifiable outsourcing profile candidate.

**First experiment:** Read the full protocol before defining executable component boundaries.

**Promotion gate:** Abstract-only catalog evidence; do not map two remote online parties into one evaluator.

**Original evidence boundary:** Abstract-only review: two malicious non-colluding servers. Not evidence for a one-online-server implementation.

#### [Sort, Sweep, Mirror: Batch Private Interval Lookup with Logarithmic Cost](https://staff.ie.cuhk.edu.hk/~smchow/scholar/sp26-sort-sweep-mirror.html) (2026)

**Paper ID:** `sort-sweep-mirror` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.schedulers`, `pllm.nonlinear`, `pllm.preparation`, `pllm.evidence`.

**Suggested contribution:** Batch private interval-lookup protocol; scheduler groups only compatible independent queries.

**First experiment:** Single-query and batched costs with identical table/query arrangements.

**Promotion gate:** Do not transfer amortized batch bounds to unrelated single-query workloads.

**Original evidence boundary:** Benefits depend on batching and table/query arrangement; not a blanket logarithmic-byte bound for arbitrary single-query tables.

#### [BumbleBee: Secure Two-party Inference Framework for Large Transformers](https://www.ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf) (2025)

**Paper ID:** `bumblebee` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Protocol-family references and selectively extracted secure operators; whole-system reproduction belongs in a separately reviewed profile.

**First experiment:** One bounded operator and full communication/material accounting before whole-model claims.

**Promotion gate:** Retain each paper-specific role, numeric and security contract; these papers are not mutually interchangeable.

**Original evidence boundary:** Both parties participate online; evaluated workloads and client cost must be retained.

#### [Dash: Accelerating Distributed Private Convolutional Neural Network Inference with Arithmetic Garbled Circuits](https://arxiv.org/html/2302.06361v2) (2025)

**Paper ID:** `dash` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`, `pllm.runtime`, `pllm.providers`.

**Suggested contribution:** Boolean/arithmetic garbling primitives and prepared-circuit evaluator variants.

**First experiment:** One prepared circuit with evaluator bytes, key/label lifetimes and end-to-end numeric checks.

**Promotion gate:** Primitive or CNN evidence does not imply transformer coverage; circuit reuse requires an explicit security argument.

**Original evidence boundary:** CNN evidence, not full LLM evidence. Keep first-online and proceedings dates separate.

#### [Encrypted Matrix-Vector Products from Secret Dual Codes](https://eprint.iacr.org/2025/858) (2025)

**Paper ID:** `emvp` · **Track:** `priority_prototype` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.correlation`, `pllm.kernels`, `pllm.runtime`, `pllm.plan`.

**Suggested contribution:** Alternative delegated-linear protocols with explicit prepared-matrix artifacts and trusted client key/hint handling; not merely replacement matrix kernels.

**First experiment:** Identical matrix shapes and precision; measure preparation, client retained bytes, local operations and per-query communication.

**Promotion gate:** Review the selected construction and arithmetic/security assumptions; reusable key/hint lifetimes must not be confused with reusable masks.

**Original evidence boundary:** Semi-honest linear-algebra construction; sub-2x figures concern specific large-matrix settings, not an entire LLM. The ePrint record was revised in August 2026.

#### [MOAI: Module-Optimizing Architecture for Non-Interactive Secure Transformer Inference](https://openreview.net/forum?id=qJn4HtTzhH) (2025)

**Paper ID:** `moai` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Publication/search record verified, but the full OpenReview page encountered a browser challenge; not a full-text review.

#### [NEXUS: Secure Transformer Inference Made Non-interactive](https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf) (2025)

**Paper ID:** `nexus` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Forward inference is not equivalent to an autoregressive service with persistent private KV state and private sampling.

#### [Practical Secure Delegated Linear Algebra with Trapdoored Matrices](https://arxiv.org/html/2502.13060v3) (2025)

**Paper ID:** `trapdoored-matrices` · **Track:** `priority_prototype` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.correlation`, `pllm.kernels`, `pllm.runtime`, `pllm.plan`.

**Suggested contribution:** Alternative delegated-linear protocols with explicit prepared-matrix artifacts and trusted client key/hint handling; not merely replacement matrix kernels.

**First experiment:** Identical matrix shapes and precision; measure preparation, client retained bytes, local operations and per-query communication.

**Promotion gate:** Review the selected construction and arithmetic/security assumptions; reusable key/hint lifetimes must not be confused with reusable masks.

**Original evidence boundary:** Linear algebra primitive; nonlinear composition, precision and server integrity need separate treatment.

#### [SHARK: Actively Secure Inference Using Function Secret Sharing](https://eprint.iacr.org/2025/716) (2025)

**Paper ID:** `shark` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.correlation`, `pllm.nonlinear`, `pllm.verification`, `pllm.roles`, `pllm.deployment`.

**Suggested contribution:** Active-secure FSS protocol family with its required preprocessing and transcript checks, not an optional verification wrapper.

**First experiment:** Test malicious deviations and abort behavior in a minimal composed nonlinear computation.

**Promotion gate:** All protocol and preprocessing obligations must hold; disabling required checks removes the active-security claim.

**Original evidence boundary:** Preprocessing and two-party online computation remain; not a free conversion of SIGMA into one-server inference.

#### [THOR: Secure Transformer Inference with Homomorphic Encryption](https://eprint.iacr.org/2024/1881) (2025)

**Paper ID:** `thor` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Primarily transformer-forward evidence, not by itself a sustained private autoregressive service.

#### [BOLT: Privacy-Preserving, Accurate and Efficient Inference for Transformers](https://eprint.iacr.org/2023/1893) (2024)

**Paper ID:** `bolt` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Record any model changes and whether results measure BERT-like forward passes rather than generation.

#### [FastQuery: Communication-efficient Embedding Table Query for Private LLM Inference](https://arxiv.org/abs/2405.16241) (2024)

**Paper ID:** `fastquery` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.models`, `pllm.passes`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Protected embedding-table selection capability; model lowering identifies the gather operation.

**First experiment:** Compare remote private lookup with trusted-local embedding lookup under the same workload.

**Promotion gate:** Protect the lookup index and retain quantization assumptions; embedding-only support is not whole-model support.

**Original evidence boundary:** Embedding lookup only; does not establish privacy or performance of the transformer body.

#### [Garbled Circuit Lookup Tables with Logarithmic Number of Ciphertexts](https://eprint.iacr.org/2024/369) (2024)

**Paper ID:** `logrow` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Private lookup execution primitive used by nonlinear methods and other protected-selection workloads.

**First experiment:** Complete private lookup, including the table payload and output reconstruction.

**Promotion gate:** Ciphertext count alone is not total communication.

**Original evidence boundary:** Ciphertext count is not total represented table bytes. Full PDF not accessible in this search.

#### [Nimbus: Secure and Efficient Two-Party Inference for Transformers](https://arxiv.org/abs/2411.15707) (2024)

**Paper ID:** `nimbus` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** BERT-oriented results are not a decoder-serving benchmark.

#### [Orca: FSS-based Secure Training and Inference with GPUs](https://eprint.iacr.org/2023/206) (2024)

**Paper ID:** `orca` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Protocol-family references and selectively extracted secure operators; whole-system reproduction belongs in a separately reviewed profile.

**First experiment:** One bounded operator and full communication/material accounting before whole-model claims.

**Promotion gate:** Retain each paper-specific role, numeric and security contract; these papers are not mutually interchangeable.

**Original evidence boundary:** Preprocessing, bandwidth and workloads must remain visible in comparisons.

#### [SIGMA: Secure GPT Inference with Function Secret Sharing](https://petsymposium.org/popets/2024/popets-2024-0107.pdf) (2024)

**Paper ID:** `sigma` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Protocol-family references and selectively extracted secure operators; whole-system reproduction belongs in a separately reviewed profile.

**First experiment:** One bounded operator and full communication/material accounting before whole-model claims.

**Promotion gate:** Retain each paper-specific role, numeric and security contract; these papers are not mutually interchangeable.

**Original evidence boundary:** Online role count, preprocessing volume and network bandwidth remain part of the cost.

#### [CipherGPT: Secure Two-Party GPT Inference](https://eprint.iacr.org/2023/1147) (2023)

**Paper ID:** `ciphergpt` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.runtime`, `pllm.plan`, `pllm.state`.

**Suggested contribution:** Protected top-k/sampling capability alongside the paper-specific two-party transformer operators.

**First experiment:** Check output-token privacy and the sampling distribution under supported numerical semantics.

**Promotion gate:** Logits, sampled token IDs and index-dependent accesses must not become plaintext at an untrusted provider.

**Original evidence boundary:** Two online parties; record the version because the 2023 preprint and 2026 journal record are distinct.

#### [FLUTE: Fast and Secure Lookup Table Evaluations](https://eprint.iacr.org/2023/499) (2023)

**Paper ID:** `flute` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Private lookup execution primitive used by nonlinear methods and other protected-selection workloads.

**First experiment:** Complete private lookup, including the table payload and output reconstruction.

**Promotion gate:** Ciphertext count alone is not total communication.

**Original evidence boundary:** Benchmark the private query and reconstruction, not only the compressed table size.

#### [Grotto: Screaming fast (2 + 1)-PC for Z2^n via (2, 2)-DPFs](https://eprint.iacr.org/2023/108) (2023)

**Paper ID:** `grotto` · **Track:** `alternate_deployment` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.correlation`, `pllm.roles`, `pllm.deployment`.

**Suggested contribution:** FSS/DPF-based nonlinear evaluation variants with explicit party and preprocessing contracts.

**First experiment:** One supported nonlinear function under the exact paper-specific topology.

**Promotion gate:** Do not relocate an online evaluator into the offline preparation role.

**Original evidence boundary:** Reference and publication verified; full protocol not reviewed. The 2+1 role structure must be retained.

#### [PUMA: Secure Inference of LLaMA-7B in Five Minutes](https://arxiv.org/abs/2307.12533) (2023)

**Paper ID:** `puma` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Protocol-family references and selectively extracted secure operators; whole-system reproduction belongs in a separately reviewed profile.

**First experiment:** One bounded operator and full communication/material accounting before whole-model claims.

**Promotion gate:** Retain each paper-specific role, numeric and security contract; these papers are not mutually interchangeable.

**Original evidence boundary:** Do not read the title’s five-minute figure as a sustained chat decoding rate. Keep three-party assumptions explicit.

#### [Primer: Fast Private Transformer Inference on Encrypted Data](https://arxiv.org/abs/2303.13679) (2023)

**Paper ID:** `primer` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Compare exact workload, network assumptions and numerical behavior rather than headline speedups.

#### [Cheetah: Lean and Fast Secure Two-Party Deep Neural Network Inference](https://www.usenix.org/conference/usenixsecurity22/presentation/huang-zhicong) (2022)

**Paper ID:** `cheetah` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Original evaluation is neural-network inference, not modern autoregressive LLM serving.

#### [Iron: Private Inference on Transformers](https://proceedings.neurips.cc/paper_files/paper/2022/hash/64e2449d74f84e5b1a5c96ba7b3d308e-Abstract-Conference.html) (2022)

**Paper ID:** `iron` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Foundational baseline rather than a modern low-latency chat implementation.

#### [Pika: Secure Computation using Function Secret Sharing over Rings](https://eprint.iacr.org/2022/826) (2022)

**Paper ID:** `pika` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.correlation`, `pllm.roles`, `pllm.deployment`.

**Suggested contribution:** FSS/DPF-based nonlinear evaluation variants with explicit party and preprocessing contracts.

**First experiment:** One supported nonlinear function under the exact paper-specific topology.

**Promotion gate:** Do not relocate an online evaluator into the offline preparation role.

**Original evidence boundary:** Three-party/preprocessing assumptions are substantive; large-table costs still matter.

#### [THE-X: Privacy-Preserving Transformer Inference with Homomorphic Encryption](https://aclanthology.org/2022.findings-acl.277/) (2022)

**Paper ID:** `the-x` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Primary abstract reviewed; original implementation not audited.

#### [ABY2.0: Improved Mixed-Protocol Secure Two-Party Computation](https://www.usenix.org/conference/usenixsecurity21/presentation/patra) (2021)

**Paper ID:** `aby2` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.plan`, `pllm.passes`.

**Suggested contribution:** Arithmetic/Boolean conversion primitives and compatible mixed-domain material.

**First experiment:** Round-trip conversions plus a composed computation and signed/truncation corner cases.

**Promotion gate:** Matching numeric widths are insufficient; share authentication, topology and input distributions must also match.

**Original evidence boundary:** Semi-honest two-party computation; conversions and client participation remain.

#### [Function Secret Sharing for Mixed-Mode and Fixed-Point Secure Computation](https://eprint.iacr.org/2020/1392) (2021)

**Paper ID:** `fss-mixed` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`.

**Suggested contribution:** FSS primitive interfaces, fixed-point conversions and preprocessing contracts used by higher-level nonlinear components.

**First experiment:** Known-answer tests of one primitive and the consuming protocol, including its conversion edges.

**Promotion gate:** Review source theorems and exact domains before selecting or composing a concrete construction.

**Original evidence boundary:** Correlated preprocessing and representation conversions are part of the protocol contract.

#### [CrypTFlow2: Practical 2-Party Secure Inference](https://eprint.iacr.org/2020/1002) (2020)

**Paper ID:** `cryptflow2` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.preparation`, `pllm.roles`, `pllm.deployment`, `pllm.kernels`, `pllm.runtime`, `pllm.profiles`.

**Suggested contribution:** Protocol-family references and selectively extracted secure operators; whole-system reproduction belongs in a separately reviewed profile.

**First experiment:** One bounded operator and full communication/material accounting before whole-model claims.

**Promotion gate:** Retain each paper-specific role, numeric and security contract; these papers are not mutually interchangeable.

**Original evidence boundary:** Cleartext fixed-point equivalence is not equivalence to an arbitrary native floating-point model; both parties remain online.

#### [Improved Primitives for MPC over Mixed Arithmetic-Binary Circuits](https://link.springer.com/chapter/10.1007/978-3-030-56880-1_29) (2020)

**Paper ID:** `mixed-arithmetic-primitives` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.plan`, `pllm.passes`.

**Suggested contribution:** Arithmetic/Boolean conversion primitives and compatible mixed-domain material.

**First experiment:** Round-trip conversions plus a composed computation and signed/truncation corner cases.

**Promotion gate:** Matching numeric widths are insufficient; share authentication, topology and input distributions must also match.

**Original evidence boundary:** Publication and edaBit contribution verified; full construction and proof not audited.

#### [Efficient Two-Round OT Extension and Silent Non-Interactive Secure Computation](https://eprint.iacr.org/2019/1159) (2019)

**Paper ID:** `silent-nisc` · **Track:** `preprocessing_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.roles`, `pllm.plan`.

**Suggested contribution:** Non-interactive protocol construction and OT-extension setup as separate contributions.

**First experiment:** Review the concrete construction, then one supported circuit with complete setup accounting.

**Promotion gate:** Non-interactive online execution does not mean setup-free or arbitrary reusable material.

**Original evidence boundary:** Reference-discovered; no full-text construction review in this pass.

#### [Garbled Neural Networks Are Practical](https://eprint.iacr.org/2019/338) (2019)

**Paper ID:** `garbled-nn` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`, `pllm.runtime`, `pllm.providers`.

**Suggested contribution:** Boolean/arithmetic garbling primitives and prepared-circuit evaluator variants.

**First experiment:** One prepared circuit with evaluator bytes, key/label lifetimes and end-to-end numeric checks.

**Promotion gate:** Primitive or CNN evidence does not imply transformer coverage; circuit reuse requires an explicit security argument.

**Original evidence boundary:** CNN-era architecture and public-weight assumptions must not be silently generalized to full LLMs.

#### [Secure Computation with Preprocessing via Function Secret Sharing](https://doi.org/10.1007/978-3-030-36030-6_14) (2019)

**Paper ID:** `fss-preprocessing` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`.

**Suggested contribution:** FSS primitive interfaces, fixed-point conversions and preprocessing contracts used by higher-level nonlinear components.

**First experiment:** Known-answer tests of one primitive and the consuming protocol, including its conversion edges.

**Promotion gate:** Review source theorems and exact domains before selecting or composing a concrete construction.

**Original evidence boundary:** Preprocessing does not make the required online parties optional. Original proof not reviewed.

#### [GAZELLE: A Low Latency Framework for Secure Neural Network Inference](https://www.usenix.org/conference/usenixsecurity18/presentation/juvekar) (2018)

**Paper ID:** `gazelle` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** Original image-classification setting; do not transfer its latency directly to decoder LLMs.

#### [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/) (2016)

**Paper ID:** `cryptonets` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.nonlinear`, `pllm.passes`, `pllm.models`, `pllm.roles`, `pllm.deployment`, `pllm.profiles`.

**Suggested contribution:** HE/hybrid inference references with selectively extracted packing, linear and nonlinear techniques; separate whole-system profiles only after implementation.

**First experiment:** One supported operator and one complete forward/decode workload, clearly distinguished.

**Promotion gate:** Retain each paper-specific model adaptation, online topology, numerical contract and state coverage.

**Original evidence boundary:** High batch throughput on a small adapted network is not low single-query latency or LLM generation speed.

#### [Function Secret Sharing: Improvements and Extensions](https://doi.org/10.1145/2976749.2978429) (2016)

**Paper ID:** `fss-extensions` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`.

**Suggested contribution:** FSS primitive interfaces, fixed-point conversions and preprocessing contracts used by higher-level nonlinear components.

**First experiment:** Known-answer tests of one primitive and the consuming protocol, including its conversion edges.

**Promotion gate:** Review source theorems and exact domains before selecting or composing a concrete construction.

**Original evidence boundary:** Reference-discovered; not a full-text proof review.

#### [Garbling Gadgets for Boolean and Arithmetic Circuits](https://eprint.iacr.org/2016/969) (2016)

**Paper ID:** `garbling-gadgets` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`, `pllm.runtime`, `pllm.providers`.

**Suggested contribution:** Boolean/arithmetic garbling primitives and prepared-circuit evaluator variants.

**First experiment:** One prepared circuit with evaluator bytes, key/label lifetimes and end-to-end numeric checks.

**Promotion gate:** Primitive or CNN evidence does not imply transformer coverage; circuit reuse requires an explicit security argument.

**Original evidence boundary:** Abstract/publication record reviewed; full PDF unavailable in this search.

#### [Function Secret Sharing](https://doi.org/10.1007/978-3-662-46803-6_12) (2015)

**Paper ID:** `fss-original` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.correlation`, `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`.

**Suggested contribution:** FSS primitive interfaces, fixed-point conversions and preprocessing contracts used by higher-level nonlinear components.

**First experiment:** Known-answer tests of one primitive and the consuming protocol, including its conversion edges.

**Promotion gate:** Review source theorems and exact domains before selecting or composing a concrete construction.

**Original evidence boundary:** Found in reviewed bibliographies; original proof not reviewed.

#### [Two Halves Make a Whole: Reducing Data Transfer in Garbled Circuits using Half Gates](https://eprint.iacr.org/2014/756) (2015)

**Paper ID:** `half-gates` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.nonlinear`, `pllm.plan`, `pllm.runtime`, `pllm.providers`.

**Suggested contribution:** Boolean/arithmetic garbling primitives and prepared-circuit evaluator variants.

**First experiment:** One prepared circuit with evaluator bytes, key/label lifetimes and end-to-end numeric checks.

**Promotion gate:** Primitive or CNN evidence does not imply transformer coverage; circuit reuse requires an explicit security argument.

**Original evidence boundary:** Primitive-level efficiency, not full private LLM efficiency; full PDF unavailable.

#### [Oblivious decision program evaluation](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ifs.2012.0032) (2013)

**Paper ID:** `oblivious-decision-programs` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.passes`, `pllm.nonlinear`, `pllm.preparation`.

**Suggested contribution:** Protected traversal/evaluation of a fixed decision program.

**First experiment:** One small decision program with observed-access and communication accounting.

**Promotion gate:** Program representation and path privacy require separate contracts.

**Original evidence boundary:** Fixed decision-program evaluation is not a complete transformer protocol.

### `pllm.protocols.masked_linear`

#### [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](https://arxiv.org/html/1806.03287v2) (2019)

**Paper ID:** `slalom` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.preparation`, `pllm.verification`, `pllm.roles`, `pllm.deployment`.

**Suggested contribution:** Reference for prepared masked-linear execution and separate trusted-client verification; preserve the original trusted-hardware boundary in paper comparisons.

**First experiment:** Exact masked-linear result and rejection of altered results across supported numeric domains.

**Promotion gate:** A local-client adaptation is not a reproduction of the original TEE deployment.

**Original evidence boundary:** Trusted boundary is material; the original setting is not hardware-free remote private inference.

### `pllm.research`

#### [Breaking Euston: Recovering Private Inputs from Secure Inference by Exploiting Subspace Leakage](https://arxiv.org/html/2604.17238v1) (2026)

**Paper ID:** `breaking-euston` · **Track:** `security_review` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.evidence`, `pllm.compiler`, `pllm.plan`.

**Suggested contribution:** Version- and construction-specific security analyses, regression fixtures and promotion restrictions.

**First experiment:** Reproduce the relevant attack or check the criticized security reduction with the exact target variant.

**Promotion gate:** Distinguish a concrete recovery attack from a challenged reduction; scope restrictions to affected constructions.

**Original evidence boundary:** Attack authors’ results were reviewed, not reproduced; distinguish the analyzed construction/version from any subsequent repair.

#### [Euston: Efficient and User-Friendly Secure Transformer Inference with Non-Interactivity](https://eprint.iacr.org/2026/046) (2026)

**Paper ID:** `euston` · **Track:** `security_review` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.evidence`, `pllm.protocols`, `pllm.passes`.

**Suggested contribution:** Version-specific protocol record linked to the Breaking Euston analysis.

**First experiment:** Review the exact analyzed transmission construction and proposed repairs before an executable candidate.

**Promotion gate:** Do not promote the analyzed vulnerable SVD transmission path.

**Original evidence boundary:** Do not adopt its SVD-transmission optimization without resolving Breaking Euston’s findings.

#### [SoK: Private Transformer Inference Across Systems, Models, and Cryptography](https://eprint.iacr.org/2026/2005) (2026)

**Paper ID:** `sok-transformers` · **Track:** `research_only` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.evidence`, `pllm.metrics`.

**Suggested contribution:** Taxonomy, missing-coverage checks and benchmark/evidence schema improvements.

**First experiment:** Review full text before claiming coverage of every surveyed framework.

**Promotion gate:** Survey coverage is not executable capability or reproduced evidence.

**Original evidence boundary:** Abstract-only access. Its complete 58-framework inventory was not inspected; this is not a claim to have read every framework it surveys.

#### [Game of Arrows: On the (In-)Security of Weight Obfuscation for On-Device TEE-Shielded LLM Partition Algorithms](https://www.usenix.org/conference/usenixsecurity25/presentation/wang-pengli) (2025)

**Paper ID:** `game-of-arrows` · **Track:** `security_review` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.evidence`, `pllm.compiler`, `pllm.plan`.

**Suggested contribution:** Version- and construction-specific security analyses, regression fixtures and promotion restrictions.

**First experiment:** Reproduce the relevant attack or check the criticized security reduction with the exact target variant.

**Promotion gate:** Distinguish a concrete recovery attack from a challenged reduction; scope restrictions to affected constructions.

**Original evidence boundary:** Distinguish attacks on earlier schemes from the later critique of ArrowCloak’s proof; do not conflate a failed reduction with demonstrated full model recovery.

#### [Ripple: Enhancing Homomorphic Lookup Tables with Wavelets](https://nillion.pub/curl-private-llms-through-dwt-lut.pdf) (2024)

**Paper ID:** `ripple` · **Track:** `discovery_only` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.nonlinear`, `pllm.protocols`, `pllm.preparation`.

**Suggested contribution:** Discovery-only wavelet/HE lookup lead; nonlinear and lookup components are possible eventual owners.

**First experiment:** Locate and inspect a standalone primary publication.

**Promotion gate:** The original catalog did not verify a standalone source; no executable capability should be inferred.

**Original evidence boundary:** A standalone primary publication URL was not verified. URL points to the citing primary paper; title-only follow-up, not a reviewed independent source.

#### [Slalom at the Carnival: Privacy-preserving Inference with Masks from Public Knowledge](https://cic.iacr.org/p/1/3/40/pdf) (2024)

**Paper ID:** `carnival` · **Track:** `security_review` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.evidence`, `pllm.protocols`, `pllm.correlation`, `pllm.preparation`.

**Suggested contribution:** Versioned construction records and subspace-leakage regression cases; only separately reviewed unaffected constructions become protocol candidates.

**First experiment:** Test the analyzed public-subspace masking construction using projection attacks.

**Promotion gate:** Do not promote the attacked instantiation; do not generalize the attack to every variant.

**Original evidence boundary:** Maverick attacks a vector-space instantiation; record the attacked construction and parameters, not a blanket assertion that every masking scheme is broken.

### `pllm.runtime`

#### [An Efficient Private GPT Never Autoregressively Decodes](https://proceedings.mlr.press/v267/li25q.html) (2025)

**Paper ID:** `private-gpt-no-autoregression` · **Track:** `decoding_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.plan`, `pllm.pipeline`, `pllm.models`, `pllm.sources`, `pllm.state`, `pllm.protocols`, `pllm.schedulers`.

**Suggested contribution:** Explicit speculative-decoding state machine with draft/target models, token acceptance and KV commit/rollback.

**First experiment:** Compare private target invocations, client draft work and bytes per accepted token; test rejection, cancellation and retries.

**Promotion gate:** Token acceptance is not cryptographic execution verification; rolling back speculative KV never permits reusing consumed material.

**Original evidence boundary:** Adds local draft-model computation. Speculative verification means token acceptance, not a cryptographic proof against a malicious server.

#### [Piranha: A GPU Platform for Secure Computation](https://www.usenix.org/conference/usenixsecurity22/presentation/watson) (2022)

**Paper ID:** `piranha` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.kernels`, `pllm.native`, `pllm.protocols`, `pllm.providers`, `pllm.evidence`.

**Suggested contribution:** GPU execution/backend integration; only compatible bounded matrix routines enter the existing kernel contract.

**First experiment:** Separate matrix arithmetic, protocol communication, device transfers and total participating-party resources.

**Promotion gate:** Generic secure GPU operators need reviewed backend extensions, not silent widening of the matrix-only public kernel API.

**Original evidence boundary:** Publication/abstract reviewed; bibliography not fully traversed. Party count and communication depend on the chosen protocol.

### `pllm.state`

#### [Cachemir: Fully Homomorphic Encrypted Inference of Generative Large Language Model with KV Cache](https://arxiv.org/html/2602.11470v1) (2026)

**Paper ID:** `cachemir` · **Track:** `alternate_deployment` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.passes`, `pllm.runtime`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Encrypted incremental KV-state descriptors and decode-specific execution/refresh planning.

**First experiment:** Several decode steps with persistent state; include memory growth, bootstrap costs and per-token communication.

**Promotion gate:** Runtime owns mutable ciphertext state; immutable state configuration owns descriptors only.

**Original evidence boundary:** Reported Llama3-8B GPU next-token latency is about 1.61 minutes, not interactive near-native speed. Encrypted memory is substantial.

##### CacheMir source recovery and implementation decision sheet

This subsection is the CacheMir research audit, not a changed method approval or
runtime-availability claim. Public release metadata below predates the
21 October 2026 cutoff. Static source/archive inspection and bounded public-scalar
arithmetic only: no upstream execution, builds, checkpoint/tensor downloads or
encrypted measurements. Archives were inspected in memory, not copied into PLLM.

**Immutable sources.** Governing source remains
[`papers/cachemir.pdf`](../../papers/cachemir.pdf), arXiv:2602.11470v1,
SHA-256 `67db77545b62a4ec0f6906706b947492e6d1bcd9a37b84e5afc32445b41c98e7`.
The [author publication page](https://mengli.me/publication/sp-2027-cachemir/)
identifies the paper/authors/venue. The anonymous Appendix A endpoint did not
yield source; its API returned 401 and was not retried. That is not artifact
absence: the following public Zenodo versions contain `Cachemir_ae.zip`.

A separate 23 October check of the [public file-tree API](https://anonymous.4open.science/api/repo/Cachemir-B254/files/)
returned **HTTP 410**, exact body `{"error":"repository_expired"}`. This does not
invalidate the previously recovered immutable Zenodo sources below.

| Immutable release DOI | Version / publication date | Bytes | Archive SHA-256 |
| --- | --- | ---: | --- |
| [10.5281/zenodo.22844188](https://zenodo.org/records/22844188) | `v0.1-ae`, 2026-09-19 | 3,261,037 | `15afe42bca5a2f29baddc754c5f4d2489d6cae92571d66639f8a019da3d0b295` |
| [10.5281/zenodo.23189548](https://zenodo.org/records/23189548) | `v0.1.1-ae`, 2026-10-06 | 67,458 | `3457613fddd4b78a34f765f391a5a4d7b489741f5f99af1d396d978f309be006` |
| [10.5281/zenodo.23377151](https://zenodo.org/records/23377151) | `v0.1.2-ae`, 2026-10-14 | 63,073 | `4462530bb49ba7b4cbdbf13d7d64f7304ccabf401e3f05b8f7f75cf92846d451` |

Downloaded MD5s match each record's public file metadata. The concept DOI
`10.5281/zenodo.22844187` follows versions; it is not the immutable release pin.
The first archive's HEAD is commit
`0d5695869730427c1a5dc4320d3bd375e727d2b5`, tree
`f646b8e4197d5babbc292b2320dcf4206f5b9388`; its tracked archived files match that
tree and the inspected Git objects hash correctly. Annotated tag `v0.1.0-ae`
is object `57f56adbc318f8c2c552778e338a8747b19b8c60`, targeting the **different**
commit `dcbb3645b15f8ffde3d08ab5a1fa09dd53386779`. The latest archive has no Git
metadata; no latest commit or public author-linked Git remote was established.

Zenodo describes acceptance at S&P 2027 and dates the future meeting
17–20 May 2027. The author page also names S&P. These are publication/venue
metadata, **not** an already held conference, evaluated runtime, or independent
performance result. No proceedings-final byte identity was verified. The first
archive includes a different 19-page manuscript, `FHE_transformer_CCS (3).pdf`,
SHA-256 `67e7ebbf4c4a65d4e24ca53247274757c75ebc83ce189284329229b6a3183d18`;
it does not supersede the pinned PDF. The
[arXiv v1 source](https://arxiv.org/src/2602.11470v1) contains TeX/figures, not
an additional evaluated numeric/configuration manifest.

**Recovered numeric graph; not an adopted profile.** Paths/lines below are
inside `Cachemir_ae/` in `v0.1.2-ae`, unless explicitly qualified.

| Operation | Exact recovered source | Fidelity boundary |
| --- | --- | --- |
| Exponential | `nonlinear_thor.go:40–81`: complete 16-entry binary64 monomial arrays `thorExp1Coeffs` / `thorExp2Coeffs`, degree 15, declared ranges `[-27.2493,21.72692]` / `[-70,70]`. Small fit subtracts midpoint `(-27.2493+21.72692)/64` at `:160–165`. | Fixed THOR-labelled fits, not recovered per-layer CacheMir calibration. |
| Softmax | `SoftmaxThor:1125–1206`: pre-scale raw QK scores by `1/(8*sqrt(head_dim))`; call `cExp8xSmallOn(x,0,tokenSlots)`; reduce per head over valid positions; bootstrap denominator to 13; invert; multiply exponential by inverse. | Zero exponential squarings in this call. Valid-slot masking is substantive: exponentiating padded zero scores would corrupt normalization. No full-model denominator-domain proof supplied. |
| Reciprocal | `Inv:620–806`: adaptive `k=2/(e+1)`, bound recurrence `e <- k*e*(2-k*e)`; softmax supplies `e0=2^-11`, `alpha=.001`; cap 15. Initial `a=1,b=u` update gives `a'=k^2*(2/k-b)`, `b'=b*a'`; later updates multiply both accumulators by the same correction. | Public-scalar recurrence yields **8 total iterations**, not a measured HE result or standard-Goldschmidt schedule. Below two remaining levels, refresh both accumulators; `:707–724,765–770` scales the reciprocal accumulator by epsilon when a level remains and restores gain later. |
| Inverse square root | `InvSqrt:337–592`: solve `(1-e^3)k^3+(6e^2-6)k^2+(9-9e)k=0`; select smaller real root `>1+1e-6`, fallback 3; bound update `e <- k*e*(3-k*e)^2/4`; cap 10. Accumulator updates use factors `(k^(3/2)/2)*(3/k-a)` and `(k^3/4)*(3/k-a)^2`. | `GetDefaultLayerNormConfigForLlama:871–899` supplies variance ranges `.15–10`, `.2–150`, `.75–2500`, epsilon `1e-5`. With `e0=min/max`, `alpha=.001`, scalar equations yield **5/6/7 iterations**. No CKKS error/convergence certificate follows. |
| Normalization | `normThor:920–957,1068–1093`: generated gamma=1/beta=0; centered variance; input multiplier `1/sqrt((maxVar*1.05+1e-5)*N^2)`; scaled epsilon added before inverse square root. | Three fixed THOR ranges, not checkpoint/layer fits or Llama RMSNorm. `DecoderThor:68,78` uses post-residual normalization with type argument 0 at both sites; 5/6/7 above describe configuration variants, not three evaluated model layers. |
| SiLU | `nonlinear_moai.go:13–35`: one Chebyshev approximation of `x/(1+exp(-x))` on `[-20,20]`, interval precision 128, literal `Nodes:1023`, mapped only to expanded active slots. | Not the paper's specified composite-polynomial/per-layer calibrated schedule. Exact generated coefficient/evaluation bytes still require freezing before independent execution. |

**Paper/artifact contradiction, confirmed across all three releases.** Pinned
PDF Appendix E p.16 and its TeX `docs/Appendix.tex:169–173` explicitly choose
**standard Goldschmidt rather than aSOR** and recalibrate larger-model ranges.
The bundled later manuscript p.18 repeats that choice. All three archives instead
implement adaptive reciprocal inversion: first/middle `nonlinear_thor.go:619–639`,
latest `:624–644`. First/middle file SHA-256 is
`2ec321a6f62f37a87d44b28a2541bd64ee4be0b412bf953d88f8d76453c337db`;
latest is `3e73def49eb51a84a1ec52c50c24b55b900b95a87579fa3604df87a9c27698f8`.
The latest source's normalization and SiLU choices also cannot establish
the evaluated source-model graph. Neither release recency nor a successful module
test would resolve those differences.

**HE literals/defaults recovered.** Latest `main.go:30–47` sets
`LogQ=[53,41 x 13]`, `LogP=[61 x 4]`, default scale `2^41`, ternary secret
weight 192, and bootstrap `LogMessageRatio=11`. This last override is version
specific, not evidence of a frozen configuration shared by all releases.
`LogN` is a CLI input: `main.go:15–22` defaults to `logN=8`, hidden dimension 32,
two heads and synthetic sequence length 29; README `:43–50` separately labels
`logN=16,hidDim=4096` the paper-sized configuration. Neither the tiny defaults
nor the example command establishes the evaluated full configuration or security.
`go.mod:3–5` pins Go 1.24.2 and Lattigo v6.1.1; the dependency tag resolves to
commit `37a34b4676e987795949a795f0ce9d0893763c96`. Static inspection of its
[`circuits/ckks/bootstrapping/parameters_literal.go`](https://github.com/tuneinsight/lattigo/blob/37a34b4676e987795949a795f0ce9d0893763c96/circuits/ckks/bootstrapping/parameters_literal.go)
recovers inherited defaults: CtS/StC depths 4/3, EvalMod depth 8 / log-scale 60,
ephemeral encapsulation weight 32, modular polynomial degree 30, three double
angles, one bootstrap iteration. File SHA-256:
`5d47229908c264afb4d8395f407e9a8b60d766cf3b48d51b92c9e27739821027`.
These are source-library facts, not a new PLLM parameter selection or security
approval. Instantiated ordered primes, complete key/error manifest and evaluated
precision were not obtained by executing the library. `util.go:238–271` places
decryptor and evaluator in one test harness, not separated service roles.

**State and evaluated-model boundary.** `util.go:521–541` generates random
weights; `:646–658` generates random cache entries and explicitly says actual
cache should come from prefill. `readCache:661–754` supplies real encrypted
packing and BSGS-oriented V placement, not encrypted prefill. `PrepareCache:627–629`
rejects `seqLen+1 > S/H`. `linear.go:237–278` appends at fixed `size.seqLen`,
mutates K before V, and has no position increment, regenerated masks or rollback.
`QK_T:289–392` / `AttnV:404–496` form new product ciphertexts from retained
parents; they do not specify a retained-cache refresh/error policy. `Model` in
`llama.go:113–122` repeats blocks on the same owner, not a token-feedback loop.
`DecoderThor:49–78` hardcodes activation refresh targets 9/7/13/7; it does not
consume placement-search output. README `:145–151` excludes real-model weights
and datasets. The bundled manuscript p.19/Table 12 claims three-token error
propagation, but matching trained-model driver/calibration files were not found
in the released tree. No checkpoint/tokenizer revision, GQA mapping, calibrated
real-model identity or complete private head/lookup/token loop was recovered.

**Independent sandbox cross-check (23 October 2026).** The ROSETTA author-linked
[`JerryFishY/rotation-sandbox`](https://github.com/JerryFishY/rotation-sandbox/tree/7c953a13e12e0646ef935eae22477c833f21d2d5)
at immutable `7c953a13e12e0646ef935eae22477c833f21d2d5` identifies Cachemir in
`README.md:1–3`, but is not an additional trained/evaluated driver. In this
**different tree**, `main.go:42–50` uses synthetic input;
[`util.go:420–438,543–555`](https://github.com/JerryFishY/rotation-sandbox/blob/7c953a13e12e0646ef935eae22477c833f21d2d5/util.go#L420-L438)
fills weights/KV with constant `0.01`, explicitly as dummy data, not checkpoint
loading or prefill. `llama.go:93–104,145–156` comments out cache insertion and
Softmax in both default decoder and comparison. `Model:186–197` repeats the same
layer/maps 32 times, not a generated-token loop. `main.go:163` also supplies
`"mask"` to `PrepareCache`, whose `util.go:526–538` switch accepts only `k`/`v`.

`archive_old_materials/run_test.sh:12` prints paper-sized flags, but `:14`
executes only `-test`/`-level`, retaining tiny `main.go:13–20` defaults. Its
printed command therefore cannot bind table rows to the advertised configuration.
The sole branch has parentless root
`36fb362118f6a218d643d5fb2c7045aef7866a48`, followed by two TODO-only edits
(`5f52bf44d3c0f66c761dfa97acf60560e4757b40`, pinned head); tags/releases were
empty. No earlier evaluated implementation pointer was recovered from that
history. This bounded follow-up used 25 public text/metadata requests totaling
208,572 response bytes, with no upstream execution or dependency/model downloads.
It neither replaces the Zenodo audit nor resolves the source/model/calibration,
per-layer retained-state and token-loop questions below.

**Minimal decision sheet; no approval recorded here.**

| Work / decision | Sufficient existing contract; exact remaining boundary |
| --- | --- |
| Reuse now, unchanged | `crates/pllm-core/src/interleaved_kv_layout.rs`: distinct K/V addresses, allocation deltas, masks, append and immutable-read integer oracle. Do not add another layout oracle. Rectangular VMM, physical RoPE, GQA/checkpoint mapping and real ciphertext kernels remain separate. Artifact BSGS V placement needs an explicit equivalent transform, not relabeling the oracle. |
| Reuse compiler contracts | `crates/pllm-compiler/src/encrypted_level_schedule.rs`: serial complete-feasible-DAG solver plus `PublicCiphertextState`, `PublicCiphertextGraph`, `PublicCiphertextCapability`, `validate_public_ciphertext_trace`. Composite trace checks immutable retained parents, typed joins, error-assumption identities and full backing/scratch/recovery ownership. It neither optimizes arbitrary graphs nor proves kernels, source lowering or error bounds. Supply source-bound capabilities; do not duplicate this IR. |
| Reuse ownership/lifecycle, not arithmetic | `python/pllm/runtime/ckks_reference.py` separates client decryptor/public evaluator but uses bounded SEAL arithmetic without real refresh. `transformer_client.py`'s `SnapshotStateBasis` and response-owner checks, plus `client.py` accept/rollback/abort, provide lifecycle patterns. `DecodeCache` uses AES-GCM at rest and restores clear arrays; it is not homomorphic KV. `EncryptedKvState` remains pending. |
| Implement source-compatible encrypted block after admission | Use source projection placement, immutable derived read copies and staged paired K/V updates; charge old/new backing overlap through the shared trace contract. Missing: source-mapped real-refresh backend, encrypted prefill, actual projection/attention/nonlinear kernels, graph lowering and response-owned ciphertext transaction. Backend worker owns backend qualification; D4 torus qualification does not adopt CacheMir parameters or Lattigo. No registration-only or clear-cache completion. |
| Numerical choice requiring clarification/approval | Preserve pinned-paper **standard-Goldschmidt** target. Do not adopt released aSOR, fixed THOR ranges or single Chebyshev SiLU by convenience. Obtain evaluated standard-Goldschmidt/configuration source; otherwise an independently calibrated source-derived numerical completion requires explicit approval, including corpus/model revision, ranges, coefficients, evaluation order, precision/error and admission policy. No constants selected here. |
| Retained state requiring explicit completion | Propose request-local immutable retained ciphertext parents; derive level-aligned temporary copies; stage K/V together and publish only after successful step. Approval/source clarification must bind retained append levels, refresh/error recurrence and capacity transitions beyond `S/H`. Cancellation poisons staged work; no eviction, compression or cross-request continuation is implied. |
| Token loop requiring separate approval | Recover private lookup/head/selection/re-embedding and approximate-score tie/error contract. Paper §§2.1–2.3 pp.3–4 gives boundary-only client operations but no executable completion. D9 names four shared-computation methods, not CacheMir. Do not substitute client logits/argmax, per-token decryption or another disclosure boundary. |

**Bounded standard-Goldschmidt proposal, not chosen defaults.** For an externally
certified denominator interval `0 < a <= x <= b`, let `u=x/b`, `e0=1-u`, `r0=1`;
perform exactly `m` public iterations `r <- r*(1+e); e <- e*e`, return `r/b`.
In exact arithmetic, relative residual is `(1-u)^(2^m)` and is at most
`(1-a/b)^(2^m)`. This supplies the standard, unrelaxed algebra; **not** a
paper-specified seed/count or CKKS error certificate. Select no `a,b,m` until
source clarification or approval supplies per-node denominator bounds and error
budget, with CKKS/refresh propagation included. The paper's open interval `(0,1)`
alone gives no finite uniform iteration count; Table 6's RMSE bits do not supply
the missing uniform bound. Regenerated fits or bounds are an explicit completion,
not recovery of evaluated coefficients.

**Exact next source question / decision.** Ask authors for the immutable source
and numeric/configuration files producing pinned Table 7 and Appendix E's standard
Goldschmidt results, and the later Table 12 trained-model multi-token driver:
checkpoint/tokenizer revisions, calibration corpus/ranges, polynomial bytes,
ordered reciprocal/rsqrt schedule, retained-versus-temporary level/refresh policy,
per-layer ownership and position/mask advancement. Identify which paper statement
or release is authoritative where they conflict. If unavailable, the next user
decision is specifically whether to approve an **independently calibrated,
paper-aligned standard-Goldschmidt completion** with the above explicit bounds
and request-local retained-state contract. Released aSOR is not the default
fallback. Output/feedback disclosure and backend adoption remain separate
decisions; this sheet grants neither.

#### [MPCache: MPC-Friendly KV Cache Eviction for Efficient Private Large Language Model Inference](https://arxiv.org/html/2501.06807v2) (2025)

**Paper ID:** `mpcache` · **Track:** `component_candidate` · **Source evidence:** `key_sections_reviewed`

**Supporting owners:** `pllm.passes`, `pllm.protocols`, `pllm.runtime`, `pllm.plan`, `pllm.evidence`.

**Suggested contribution:** Explicit cache-retention policy, static eviction transform and protected dynamic selection.

**First experiment:** Check next-token behavior and retained-state size; inspect whether selected positions or work sizes become public.

**Promotion gate:** Any lossy retention is an explicit semantic change, and dynamic access patterns require a privacy contract.

**Original evidence boundary:** Separate paper protocols, artifact backend and complete serving support. The linked v2 author list differs from the pasted registry.

### `pllm.verification`

#### [DeepProve: Verifiable End-to-End Large Language Model Inference](https://eprint.iacr.org/2026/1112) (2026)

**Paper ID:** `deepprove` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.plan`, `pllm.preparation`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Distinct verification schemes bound to the exact proved computation and numeric semantics.

**First experiment:** Altered-result rejection plus proof-generation, verification, communication and retained-state costs.

**Promotion gate:** Correctness verification alone does not establish input privacy or a safe composition with private execution.

**Original evidence boundary:** Verifiability alone is not client-input privacy. Full proof and implementation were not audited.

#### [LAMP: Linear Verification of Matrix Multiplication via Proximity Testing](https://eprint.iacr.org/2026/1571) (2026)

**Paper ID:** `lamp` · **Track:** `component_candidate` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.plan`, `pllm.preparation`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Distinct verification schemes bound to the exact proved computation and numeric semantics.

**First experiment:** Altered-result rejection plus proof-generation, verification, communication and retained-state costs.

**Promotion gate:** Correctness verification alone does not establish input privacy or a safe composition with private execution.

**Original evidence boundary:** Proof/verification complexity is not automatically private-inference cost.

#### [Laminate: Succinct SIMD-Friendly Verifiable FHE](https://eprint.iacr.org/2025/2285) (2025)

**Paper ID:** `laminate` · **Track:** `alternate_deployment` · **Source evidence:** `abstract_or_publication_record_reviewed`

**Supporting owners:** `pllm.protocols`, `pllm.plan`, `pllm.runtime`, `pllm.providers`, `pllm.evidence`.

**Suggested contribution:** Verifiable-FHE backend candidate as a jointly reviewed encryption/verification composition.

**First experiment:** One encrypted computation with its proof generation and verification included.

**Promotion gate:** Requires its actual FHE/proof backend; not a generic verifier that wraps arbitrary protocols.

**Original evidence boundary:** Not evidence of near-native whole-LLM throughput; examine proof generation as well as verification.

#### [Matrix Multiplication Verification Using Coding Theory](https://doi.org/10.4230/LIPIcs.APPROX/RANDOM.2024.42) (2024)

**Paper ID:** `matrix-coding-verification` · **Track:** `component_candidate` · **Source evidence:** `discovered_in_reviewed_references`

**Supporting owners:** `pllm.protocols`, `pllm.plan`, `pllm.preparation`, `pllm.runtime`, `pllm.evidence`.

**Suggested contribution:** Distinct verification schemes bound to the exact proved computation and numeric semantics.

**First experiment:** Altered-result rejection plus proof-generation, verification, communication and retained-state costs.

**Promotion gate:** Correctness verification alone does not establish input privacy or a safe composition with private execution.

**Original evidence boundary:** Reference-discovered; no stand-alone theorem audit.

## Integration cautions

Required integrity checks must complete before releasing tokens covered by a verified-output claim. A whole-run proof produced after streaming cannot retroactively make earlier tokens verified at release.

Cryptographic one-use material is consumed independently of decoder rollback. Once exposure is possible, cancellation, retry or rejected speculation must not return that material to the available pool.

A compiled capability contract can reject a known-invalid combination; it does not establish a new composition theorem. A declared topology also cannot establish that providers are operationally non-colluding.

Do not attach crypto implementations to client, server or SDK integration facades. Expose plan identity, bounded errors and approved output-release behavior there; keep execution in the runtime.

The JSON companion preserves the original 84 paper records and 83 observed citation edges and adds structured landing suggestions. It does not add unverified citation links.
