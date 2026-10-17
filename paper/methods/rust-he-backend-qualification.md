# Rust HE backend qualification

Review date: **17 October 2026**. Status: **static capability qualification; no backend adopted**.

## 1. Decision

**Poulpy v0.9.0 is the strongest native-Rust CKKS candidate found in this review. It already contains CKKS bootstrapping, including public component stages, sparse-secret encapsulation, and Apple-silicon CPU support.** A claim that Rust has no CKKS bootstrap would be incorrect for the inspected current sources. However, Poulpy uses **bivariate power-of-two torus arithmetic, not the papers' RNS prime chains**. It is a candidate for an explicitly approved backend adaptation, not an already qualified source-profile replacement.

**TFHE-rs v1.8.1 supplies a useful mature native-Rust primitive family, including blind rotation of an existing encrypted GLWE accumulator.** Its low-level capabilities exceed a public-LUT-only interface. It does not supply the CKKS half or establish ROSETTA's complete conversion protocol.

No inspected candidate is admitted here as a complete, mature, source-matching native-Rust backend for all four methods. The remaining gaps are specific below; this is not a claim about every Rust library or unpublished implementation.

**Recommended next bounded slice:** qualify one ciphertext-only CKKS refresh chain using Poulpy's existing degree-65536 preset, with independent PLLM ownership, layout, precision and capacity checks. First obtain approval for this particular dependency/maturity decision and its torus representation. Keep full source-system reproduction blocked until the source-profile mapping is resolved. Do not respond to a failed qualification by silently adopting Lattigo.

Already approved by the user: **Apple-M CPU development and Qwen2.5-0.5B as the common model**. These do not need to be requested again. They do not approve Lattigo, a particular Rust dependency, changed cryptographic parameters, or an unexamined nonlinear/conversion algorithm. Qwen is the common-workload adaptation; the original BERT and source-decoder obligations remain identified in the existing contracts.

This review supplements [encrypted-computation-contracts.md](encrypted-computation-contracts.md) and [encrypted-inference.md](../reproduction-contracts/encrypted-inference.md). Their Lattigo recommendations were proposals, not adoption. Their older library pins remain historical evidence, not current release claims. Only this report and [its source manifest](rust-he-backend-sources.json) are owned by this work.

## 2. Requirements used for qualification

The contracts' §§3, 8.2 and G1 require explicit scheme/key/layout/scale/level identity, ciphertext-only server evaluation, client-held secret keys, and a complete parameter/evaluation-key graph. A library name or a successful plaintext approximation cannot satisfy those gates.

| Required capability | Exact qualification boundary |
| --- | --- |
| CKKS refresh | Actual encrypted refresh with legal entry scale/range and restored usable capacity; public evaluation material only. Modulus raising alone is insufficient. |
| Representation and kernels | Explicit physical slots/coefficients, rotations/conjugation, scale and modulus control; NEXUS coefficient/dual-key operations; THOR deferred rescale/relinearization and complex packing. |
| Source parameter class | Degree 65536 and 32768 standard complex slots. NEXUS: roughly 1763 modulus bits, scale 50 bits, total depth 35/bootstrap depth 14. CacheMir/ROSETTA: 13 usable levels, bootstrap depth 15, approximate 53/41-bit prime sizes, weight-192 sparse secret with encapsulation. THOR's complete chain remains unresolved. Rounded sizes are not exact primes or security evidence. |
| ROSETTA conversion | StC → 65536-to-2048 ring conversion → RLWE plus LWE extraction/key switching → encrypted segmented lookup → reverse ring conversion → ModRaise + CtS + EvalMod. LWE dimension 1024 does not follow from extracting a degree-2048 rank-one GLWE. |
| ROSETTA selection | Blind rotation of coefficients containing encrypted `a*x+b`, not merely a fixed public LUT. Four segments require 3 HomComp calls, 5 plaintext–ciphertext multiplications, 6 blind rotations and 5 traces, excluding conversion and HomComp's internal PBS. |
| CPU and provenance | Native ARM64/macOS path, pinned code/license/toolchain, explicit key/scratch residency. Published upstream performance is not a PLLM measurement. |
| Independence and maturity | Reuse general-purpose cryptographic primitives only. PLLM independently implements method packing, kernels, numerical composition, lookup, placement and state. A new library's feature list does not by itself establish the required maturity. |

## 3. Current candidate matrix

“Present” means inspected source/API implements the primitive; it does **not** mean a build, security audit, parameter validation or local execution passed. Full immutable commits and selected file hashes are in the manifest.

| Candidate and inspected identity | Native Rust? / license | CKKS and refresh | Raw conversion / encrypted accumulator | Apple-M evidence and disposition |
| --- | --- | --- | --- | --- |
| **Poulpy v0.9.0**, release 8 Oct 2026; `ca3532570ff62b96599a74f7adb85de7726b0446` | Yes, scheme code and portable/ARM backends; Apache-2.0 | Present: standard CKKS, CtS/StC, EvalMod, C2S-first/S2C-first bootstrap, sparse encapsulation. **Torus widths, not selectable RNS chains.** | GLWE/LWE extraction and switching; keyed ring fold/unfold. Encrypted GLWE rotation exists with **GGSW-encrypted index bits**; ordinary LWE/LUT entry point initializes a public accumulator. Complete ROSETTA bridge unqualified. | Explicit AArch64 NEON and portable Rust backends; macOS ARM CI and upstream Apple-M release measurements. Best native CKKS candidate, conditional on maturity and representation approval. [P1–P12] |
| **TFHE-rs v1.8.1**, release 17 Sep 2026; `187fc0b95ab35352422deec6c027d0bd8743b6db` | Yes for CPU TFHE; BSD-3-Clause-Clear, with separate patent-policy statement | TFHE Boolean/integer/PBS implementation; no CKKS scheme in the inspected public module inventory | Present: GLWE/LWE entities, sample extraction, LWE key switching, and `blind_rotate_assign` preserving an encrypted accumulator. No qualified CKKS/RNS bridge. | Native `m1mac` CI builds/tests core crypto. Strong auxiliary primitive candidate, not a complete CKKS backend. [T1–T6] |
| **fhe.rs**, default-branch snapshot 16 Oct 2026; `98a087f1bf5f7fa95196ebe16b0e1b0eeb632442` | Native Rust; MIT | Current checkout declares `fhe` **0.2.0**, but registry latest observed is **0.1.1**. README and exported modules implement RNS-BFV/experimental multiparty BFV, not CKKS bootstrap. | BFV/RNS substrate does not establish the requested TFHE/CKKS bridge | Inspected CI is Ubuntu; no Apple-M qualification established here. Experimental/unaudited per its README. Not the requested CKKS dependency. [F1–F3] |
| **Fheanor 0.11.9**, `2ba00539204cc03816dd258b57fad94904f54d41` | Rust toolkit; MIT declared in README/Cargo; optional foreign Intel HEXL acceleration | BFV, BGV, CLPX; bootstrap for **BFV/BGV**, not evidence of CKKS bootstrap | Useful ring/gadget/RNS building blocks; no qualified requested hybrid pipeline | Research/alpha, single-threaded per README; Apple-M support not established by inspected evidence. Does not meet this full-backend/maturity gate. [H1–H2] |
| **FairMath OpenFHE-rs 0.3.2**, `a60b38a101e9cccbada7629245c6a79a4c65741e` | **Rust wrapper over C++**, BSD-2-Clause | Wrapper exposes `EvalBootstrap`, `EvalCKKStoFHEW`, `EvalFHEWtoCKKS` | Its inspected bindings do not expose the raw accumulator API discussed below | WIP wrapper; C++/CXX/OpenMP integration and exact upstream-version compatibility require qualification. Not native Rust. [O1–O3] |
| **OpenFHE v1.6.1**, release 9 Oct 2026; `b59e1f291965f14fbee5172cd83a20f5b856204f` | **C++**, BSD-2-Clause | Current upstream has RNS-CKKS bootstrap, CKKS↔FHEW/TFHE and additional vectorized functional-bootstrap paths | Public CGGI `EvalAcc(..., RLWECiphertext& acc, ...)` is concrete encrypted-accumulator evidence. Still no demonstrated ROSETTA round trip at its source parameters. | Upstream macOS build documentation; reviewed page cites Mojave, not an Apple-M validation receipt. Mature foreign-library fallback candidate only after explicit approval. [O4–O7] |
| **Sunscreen `seal_fhe` 0.8.1**, `f8ee4a0dc32418f2acd9e176c6f0af73cdf68d40` | **Rust SEAL wrapper**, AGPL-3.0-only for this crate | Inspected crate explicitly says CKKS is unsupported and exports BFV evaluator/encoders | No requested refresh/conversion evidence | A Rust API does not make SEAL native Rust. Not suitable for this slice. Its pinned SEAL submodule is `8426f35e35a67d382fa76451e2d8ff2940098bed`, not current SEAL. [S1–S2] |
| **LIT Protocol `openfhe-rs` 0.1.0**, `b0b901d0a1f6afb59387e778cbbc6c2fbb483fc9` | Separately named Rust port; Cargo declares Apache-2.0 OR MIT | No complete CKKS capability qualified here | Not confused with FairMath's wrapper | Current README says **archived, no longer maintained, educational/reference only**. Reject as the mature dependency candidate without asserting it contains no useful code. [L1–L2] |

For an additional current check, Microsoft **SEAL v4.7.0**, released 14 Oct 2026, resolves to `79ad62fb50fb59880c7b717275c82fe497634639` (MIT, C++). Its inspected public `Evaluator` header exposes multiply/relinearize/mod-switch/rescale operations, with no bootstrap entry point. This review did not find a bootstrap route in that public API; no claim about every extension/fork follows. Wrapping SEAL does not supply missing primitives automatically. [S3–S4]

## 4. Poulpy: substantial current capability, material representation difference

### 4.1 Real encrypted refresh is implemented

The pinned [`CKKSBootstrappingOps` API][P2], lines 85–150, provides `ckks_mod_up_into`, `ckks_bootstrap_mod_up`, and `ckks_bootstrap(ct_out, ct_in, ctx, keys, scratch)`. The evaluator takes a compiled context and evaluation keys, not a decryptor/secret key. Its [`reference/bootstrapping.rs` body][P3], lines 429–523:

* validates rank-one ciphertexts and consistency between the sparse-encapsulation recipe and key bundle;
* dispatches S2C-first separately;
* calls encapsulated modulus raising, encrypted coefficient-to-slot conversion, EvalMod on the real/imaginary halves, then encrypted slot-to-coefficient conversion for C2S-first.

The [bootstrap documentation][P4], lines 10–28 and 53–65, explicitly publishes both stage orders and dense→sparse→dense switching. Lines 177–180 and 229–231 distinguish the plan, prepared context and validation. This is implemented bootstrap composition, not just a roadmap bullet or “increase ciphertext modulus” placeholder. Static inspection does not establish its error/security behavior for PLLM's profiles.

The [evaluation-key types][P-key-bundle], lines 53–180, expose rotation, tensor and optional encapsulation keys as an unprepared `BootstrappingKeySet`, plus a secret-free `prepare` step and `BootstrappingKeys` evaluator view. This supports a public-evaluation-material boundary. It does not establish a bounded, authenticated PLLM import format or demonstrate that an evaluator can load the entire required bundle within the proposed memory cap.

The default standard-ring degree-65536 presets are particularly useful qualification fixtures:

| Preset | Input scale | Input / output / bootstrap width | Net usable capacity | Published precision description |
| --- | --- | --- | --- | --- |
| `n16_d35_k600_p19_c2s` | `2^35` | 40 / 600 / 1427 bits | 560 bits, described as sixteen 35-bit levels | 19-bit minimum measured on the reference vector |
| `n16_d35_k720_p19_s2c` | `2^35` | 160 / 720 / 1382 bits | 560 bits; initial StC consumes part of the input width | Same qualification |

These are **library profiles**, not NEXUS/THOR/CacheMir/ROSETTA parameter matches. Both use dense-secret weight 1024 and sparse-encapsulation weight 32. The documentation explicitly limits the advertised precision to a measured reference vector and says the associated precision test is registered as ignored; do not turn “19 bits” into a universal bound or assume routine CI executed that test. See [P4 lines 248–269][P4]. All source prime/key/security fields remain to be resolved independently.

### 4.2 Level, scale, layout and deferred arithmetic

[P1 lines 176–194][P1] and [P5 lines 79–120][P5] explicitly distinguish Poulpy from RNS. Ciphertexts carry `log_delta`, `log_sparsity`, slot kind, and a GLWE width `k`; remaining `log_budget` is derived from `k - log_delta`. ModUp raises `2^k_small` to `2^k_large`. Bit shifts and selected output widths control capacity; there is no equivalent promise to import an arbitrary ordered list of the papers' RNS primes.

The `NTT4x30Neon` backend uses four arithmetic CRT primes internally. **Those are convolution-backend primes, not the ciphertext's source RNS modulus chain.** Choosing this backend does not remove the torus/RNS difference. [P10 lines 67–91][P10]

Explicit homomorphic DFT, rotation, conjugation, polynomial and linear-transformation APIs are present. Standard CKKS supports complex slots; the newer conjugate-invariant ring does not directly implement DFT/EvalMod/bootstrap and uses explicit standard-ring maps instead. Preserve the source standard ring and its `R/2` slots; `N` real CI slots are not a free source-compatible capacity increase. [P5 lines 343–370, 422–447][P5]

At the high level, multiplication performs tensor-key switching and preserves its documented precision while consuming budget. [P6 lines 18–90, 128–158][P6] Core separately exposes `glwe_tensor_apply(..., cnv_offset, ...)` and `glwe_tensor_relinearize(...)`. [P7 lines 152–208][P7] Thus deferred composition has lower-level building blocks, but THOR's lazy-rescale/lazy-relinearization operation trace is **not** established by calling the fused CKKS multiply. PLLM would need its own typed wrapper, intermediate-width accounting and proof of the approved schedule.

### 4.3 Raw ring/LWE operations and the accumulator distinction

Core exposes `LWESampleExtract`, `LWEFromGLWE`, `GLWEFromLWE`, and LWE/GLWE key switching. [P8] `CKKSFoldOps` and `RingSwitchKeysLayout::generate` implement keyed embedding/packing and inverse splitting across related ring degrees, including explicit input/bootstrap secret relationships. [P9] This is useful conversion infrastructure; it does not prove that a degree-65536 coefficient ciphertext can be fed through the exact degree-2048/1024-dimensional ROSETTA graph and recovered with the required scales and noise.

Two different blind-rotation interfaces must be distinguished:

1. **`BlindRotationExecute::blind_rotation_execute(res, lwe, lut, brk, scratch)` takes `LookupTable`.** In the standard reference path, lines 845–861 copy the destination and then zero that temporary and initialize it from `lut.data[0]`. Passing an encrypted accumulator in `res` does not preserve it. [P11]
2. **`GLWEBlindRotation::glwe_blind_rotation(res, a, encrypted_bits, ...)` really accepts encrypted `a`.** The declared operation is `a * X^(±k)` using GGSW-encrypted bits of `k`. Its reference implementation copies the encrypted source, rotates both ciphertext components and performs successive CMux operations. [P12] This positively establishes encrypted-accumulator selection capability, with a **different index interface**. It does not establish the LWE-phase route or make conversion of an extracted scalar into those bits free. The reference also clears noise metadata with `res.set_noise(None)`; an application error bound cannot be inferred from retained metadata after this operation.

For ROSETTA, independently derive and charge either an LWE-driven encrypted-accumulator composition using generic external products/rotations or a justified bit-index construction. Switching to a different construction requires recording the algorithm/cost deviation. Generic functional bootstrap, PaCo, SHIP or public-LUT evaluation cannot silently replace the paper's segmented encrypted-affine algorithm.

An additional algebraic blocker follows from the representation: **division by `R=2^r` is not multiplication by a modular inverse in `Z/(2^k)`**. A torus trace/recovery port needs a justified shift/lift/scaling construction and its capacity/noise budget; it cannot simply reuse the contracts' odd-prime modular-division reasoning. Likewise, “53-bit modulus” does not identify a modulus or a cross-library encoding.

The README still lists higher-level scheme switching as work above the implemented primitives. [P5 lines 404–420][P5] The common torus plaintext space and available ring maps are encouraging, but no end-to-end source-compatible bridge has been verified here.

### 4.4 Apple silicon and maturity

Poulpy's ARM crate targets AArch64/Apple Silicon with `FFT64Neon`, `NTT4x30Neon` and optional Rayon variants; portable scalar Rust is also available. [P10] The pinned workflow includes Apple Silicon Quad/encoding tests and a separate native AArch64 NEON job. [P13] The [v0.9.0 release][P-release] reports CKKS bootstrap measurements on an Apple M5 Pro. Those are upstream claims, not results from this Mac and not evidence for a source-model workload.

The CKKS crate requires **nightly Rust**, pinned to **`nightly-2026-05-14`**; its portable `Quad` uses unstable `f128`. Optional `libquadmath` changes routing on supported platforms; macOS uses portable setup math. [P5 lines 22–49][P5], [P14] This is native Rust cryptography rather than SEAL/OpenFHE FFI, but its toolchain and resolved math dependencies must be pinned for reproducible setup. The root README's deterministic-encoding claim and the CKKS README's warning about last-bit backend differences are not perfectly aligned; use the actual pinned tests/source and independent cross-backend checks before asserting byte-identical encoding.

Maturity evidence includes released general-purpose crates, named maintainers, explicit primitive contracts, an independent scalar oracle, backend parity suites and an ARM path. [P1] These are stronger than an unsupported proof-of-concept label. They still do not establish an independent security audit or reliability at the requested parameter/key graph. The review did not establish such an audit and does not assert none exists. Its [security policy][P15] claims IND-CPA with appropriate parameters and explicitly does not guarantee circuit privacy. **The user's mature-dependency condition remains an admission gate, not something waived by this recommendation.**

## 5. TFHE-rs: encrypted accumulators are available at core level

At the current stable pin, [`fft64_pbs.rs` lines 186–248][T2] exposes:

```rust
pub fn blind_rotate_assign<OutputScalar, OutputCont, KeyCont>(
    msed_input: &impl ModulusSwitchedLweCiphertext<usize>,
    lut: &mut GlweCiphertext<OutputCont>,
    fourier_bsk: &FourierLweBootstrapKey<KeyCont>,
)
```

The parameter name `lut` does not mean plaintext. The [implementation][T3], lines 294–379, initially rotates **every polynomial** of the supplied GLWE, retains it as `ct0`, and applies encrypted CMux/external-product updates. It does not zero the mask or replace the accumulator with a public polynomial. This is direct evidence for the primitive ROSETTA needs, with the following limitations:

* This inspected FFT64 entry point requires a **power-of-two ciphertext modulus**, matching GLWE size/ring degree and modulus-switched LWE dimension. For non-native power-of-two moduli, lines 367–378 explicitly round back into the MSB-aligned representation. A rounded “53-bit” source modulus is not automatically this encoding.
* A ciphertext accumulator contributes its own error. Public-table PBS parameter/failure claims cannot be inherited unchanged; include accumulator error, affine coefficients, external products, modulus switching and comparison/index margins.
* [Sample extraction][T4], lines 89–117, enforces `output_lwe_dimension = glwe_dimension * polynomial_size` and equal moduli. A rank-one degree-2048 input therefore extracts dimension **2048**, not 1024. The available LWE key-switch primitive must be part of the independently specified client-generated key graph.
* TFHE parameters and high-level integer comparisons do not instantiate ROSETTA's referenced high-precision HomComp, exact endpoint/floor behavior, CKKS StC/CtS, ring switch or EvalMod recovery. Two serialized ciphertext families do not constitute a conversion protocol.

[T1] describes Boolean/integer TFHE and stabilizes the **high-level x86 CPU API** starting with v1.0. Do not extend that statement to stability of every low-level interface. The [M1 workflow][T5], lines 34–39 and 101–135, specifically runs core builds/tests on `m1mac`; this supports Apple-M CPU use without implying local validation.

License: [T6] is **BSD-3-Clause-Clear**, whose line 19 grants no patent license. The [README license section][T1] separately states Zama's commercial-patent policy while allowing research/development/prototyping use. Record both; do not mislabel it ordinary BSD-3-Clause or presume this paper research decision grants later commercial rights.

## 6. Alternatives and source-fidelity implications

**fhe.rs and Fheanor were checked at current source snapshots, not dismissed from historical memory.** fhe.rs's actual `lib.rs` exports `bfv` and feature-gated `mbfv`; Fheanor's README explicitly names BFV/BGV bootstrapping and research/alpha status. Their RNS machinery could support new cryptographic development, but writing a CKKS bootstrap on it is a substantially different proposal from selecting an existing mature CKKS backend. [F1–F3, H1–H2]

**OpenFHE deserves a current, stronger description than “advertises scheme switching.”** v1.6.1 has public CKKS bootstrap setup/keygen/evaluation and raw homomorphic CtS/StC declarations. Its CGGI accumulator's public `EvalAcc` accepts a mutable `RLWECiphertext`, and the implementation updates its existing elements. [O5–O6] FairMath's Rust wrapper exposes high-level bootstrap and both switch directions through `cxx::bridge`, but its inspected bindings do not expose `EvalAcc`; `build.rs` links `OPENFHEpke`, `OPENFHEbinfhe` and `OPENFHEcore`. [O2–O3] The wrapper snapshot predates v1.6.1, so compatibility is unproved. A Rust wrapper around this mature C++ primitive remains a foreign-backend choice and would need explicit approval and low-level binding qualification.

Per-method blockers remain:

| Method | Backend-specific blocker after this review |
| --- | --- |
| NEXUS | Qualify raw two-key commutable encryption and uniform-ring masks under the selected representation; preserve coefficient layouts, bootstrap/scale schedule and source output disclosure. A torus representation changes the concrete masking domain and needs a security/numerical mapping. |
| THOR | Reconstruct the deferred tensor/rescale/relinearization/complexification schedule using suitable low-level primitives. High-level fused arithmetic does not certify the source kernel trace. Exact bootstrap/numerical profiles remain unresolved. |
| CacheMir | Map the level DAG to an approved, explicit cryptographic capacity profile. Relabeling arbitrary bit budgets as the source's prime levels changes the optimization problem. Freeze new primitive costs later; preserve independent VMM/K/V algorithms and retained-state error policy. |
| ROSETTA | Derive the complete ring/LWE key and encoding graph, encrypted-accumulator route, HomComp, trace division, index rounding/endpoints and recovery. Neither Poulpy's shared torus nor TFHE-rs's correct blind-rotation primitive settles that composition. |

Poulpy's generic approximation fitting and functional-bootstrap facilities must not silently regenerate the papers' missing coefficient tables or replace their numerical identity. Those remain source-resolution work. A backend adaptation does not authorize a different inference algorithm.

## 7. First bounded slice and approval boundary

### Proposed slice: `rust-ckks-refresh-qualification-v1`

Use **Poulpy v0.9.0**, standard-ring degree **65536**, and the existing **`n16_d35_k600_p19_c2s`** fixture. Begin with the portable exact NTT backend; qualify NEON against it before using accelerated results. This chooses an existing real refresh implementation and source-size ring while keeping a deliberately small tensor. It is a **library qualification profile**, not a claimed source cryptographic reproduction or a Qwen inference result.

Before implementation, freeze dependency/toolchain identity, the parameter/key graph, the admitted plaintext interval, expected scale/width transitions, numerical thresholds and an allocation upper bound. Review the preset's security bounds including evaluation-key auxiliary widths. If the mature-primitive requirement cannot be supported, stop this candidate rather than lower that requirement.

Proposed subsequent execution, **only after approval**:

1. Client creates one ownership/key epoch, including the dense secret and client-generated sparse-encapsulation/switching material. A separate evaluator state contains no secret key/decryptor; import the explicit evaluation-key package through PLLM's own boundary.
2. Encrypt one deterministic, bounded dense vector with known valid/padding slots. Independently check rotations, conjugation and a small plaintext–ciphertext affine map against a clear oracle.
3. Consume capacity to the preset's legal entry, perform one real bootstrap, evaluate another small operation, and return to the legal entry width before the second refresh. Verify restored `k`, `log_delta`, layout, nonmutation of retained inputs and client-only final decryption. Compare all slots, not only one decoded value.
4. Report maximum absolute error, RMSE, error growth, actual operations, key/ciphertext/scratch bytes and observed peak memory. Treat upstream 19-bit reference precision as a fixture claim; freeze a justified local acceptance bound before execution. Test rejected entry widths/key-layout mismatches without falling back to client refresh.

Suggested initial cap for approval: **one worker/thread, one live input vector, at most two sequential refreshes, 8 GiB process-memory ceiling and ten-minute execution ceiling after build/setup**. First estimate the complete keys/context/setup/scratch footprint; reject before allocation if it cannot fit. Setup/build also need an explicit separate budget. These limits are a proposed bounded experiment, not a claim that it will fit or finish on the user's Mac. No timing benchmark campaign is part of this slice.

A successful result would establish an owned Rust evaluator with real refresh and a tested capacity/precision contract. It would not complete NEXUS, THOR, CacheMir, ROSETTA, Qwen inference or the mature/security qualification for other parameter sets. The next crypto-specific gate for ROSETTA is an encrypted-affine accumulator test and the fully specified CKKS→small-ring/LWE→CKKS round trip; model execution follows those gates.

### Further approval actually needed

| Decision | Requested scope |
| --- | --- |
| **Dependency and maturity** | Accept Poulpy's pinned generic primitives for a bounded qualification after reviewing the maturity evidence; pin nightly/toolchain and dependency resolution. No dependency addition is authorized by this report. TFHE-rs is a separate optional primitive dependency, not implicitly selected. |
| **Representation/profile adaptation** | Permit torus widths/bit-level capacity in a separately named profile, with fresh key/security/error analysis and explicit mapping to source operations. If exact source RNS chains are mandatory, Poulpy cannot satisfy that field; a different qualified backend or separately approved cryptographic development is needed. |
| **Execution** | Approve the bounded build/setup/runtime budgets and admission limits above. Apple-M platform approval is already given. |
| **Changed method behavior, if necessary** | Obtain approval before alternative index-bit conversion, altered trace/recovery arithmetic, different bootstrap/approximation schedules or changed compiler cost semantics. Faithful resolution of missing source constants should be researched first. |
| **Foreign-backend fallback, only if chosen** | Separately approve C++/OpenFHE with a qualified Rust binding, or Go/Lattigo. Neither is inferred from failure to qualify a native candidate. |

The existing source numerical, key-ownership, circuit-privacy, output/feedback and persistent-cache gates remain. Those implementation areas belong to their respective owners; this review neither implements nor changes them.

## 8. Evidence and validation receipt

Primary sources were discovered with current web searches, GitHub release/commit/tree APIs and the crates.io `fhe` registry response, then inspected as text at immutable commits. For released candidates the release pin was used; for libraries without a suitable release pin the inspected default-branch snapshot is explicitly identified. No upstream source was executed, imported, built, benchmarked, installed as a dependency or vendored into PLLM. No model weights were fetched. Temporary text copies were used only for static inspection and SHA-256 receipts.

The manifest pins nine repositories, source-input snapshots, release/version distinctions, licenses and selected-file SHA-256 values. Hashes identify inspected bytes, not full-artifact audits or authors' evaluated builds. Its two contract hashes identify the read snapshots; concurrent work on those files is not owned by this report. Static validation passed for JSON, 21 selected-file hashes, two input hashes, 43 immutable-link targets, three local links and balanced fences. **No cryptographic execution or performance evidence was produced.**

### Primary-source locators

All code/document locators below resolve under the commits in the manifest. Release pages and registry metadata are dated observations; they are not substitutes for immutable code pins.

* **P1:** [Poulpy README][P1] — representation, crates, backend/parity architecture, license.
* **P2–P4:** [bootstrap API][P2], [implementation][P3], [bootstrap specification/presets][P4].
* **P5–P7:** [CKKS README][P5], [CKKS multiply contract][P6], [separate core tensor/relinearization API][P7].
* **P8–P9:** [GLWE/LWE conversions][P8], [ring fold/unfold API][P9]; [ring-switch key generation][P9-keys].
* **P10–P12:** [ARM backend][P10], [public-LUT blind rotation implementation][P11], [encrypted-GLWE rotation API][P12] and [implementation][P12-impl].
* **P13–P15:** [CI][P13], [toolchain][P14], [security policy][P15]; [v0.9.0 release][P-release].
* **T1–T6:** [TFHE-rs README][T1], [blind-rotation API][T2], [implementation][T3], [sample extraction][T4], [M1 CI][T5], [license][T6]; [v1.8.1 release][T-release].
* **F1–F3:** [fhe.rs README][F1], [exported modules][F2], [crate version][F3]; [registry observation][F-registry].
* **H1–H2:** [Fheanor README][H1], [Cargo identity/features][H2].
* **O1–O7:** [FairMath README][O1], [bindings][O2], [build script][O3]; [OpenFHE README][O4], [CKKS bootstrap/stage declarations][O5], [CGGI accumulator API][O6] and [implementation][O6-impl], [macOS instructions][O7]; [v1.6.1 release][O-release].
* **S1–S4:** [`seal_fhe` API scope][S1], [crate identity/license][S2]; [SEAL v4.7.0 README][S3], [public Evaluator][S4].
* **L1–L2:** [LIT archive notice][L1], [Cargo identity/license][L2].

[P1]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/README.md
[P2]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/api/bootstrapping.rs#L85-L150
[P3]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/reference/bootstrapping.rs#L429-L523
[P-key-bundle]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/layouts/bootstrapping_keys.rs#L53-L180
[P4]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/docs/bootstrapping.md
[P5]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/README.md
[P6]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/api/mul.rs#L18-L158
[P7]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/api/operations.rs#L152-L208
[P8]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/api/conversion.rs#L9-L51
[P9]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/api/fold.rs
[P9-keys]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/layouts/bootstrapping_keys.rs#L323-L418
[P10]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-cpu-arm/README.md
[P11]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/reference/blind_rotation/algorithm.rs#L799-L890
[P12]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/api/bdd/glwe_blind_rotation.rs
[P12-impl]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-bin-fhe/src/reference/bdd/glwe_blind_rotation.rs#L40-L108
[P13]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/.github/workflows/ci.yml
[P14]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/rust-toolchain.toml
[P15]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/SECURITY.md
[P-release]: https://github.com/poulpy-fhe/poulpy/releases/tag/v0.9.0
[T1]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/README.md
[T2]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/tfhe/src/core_crypto/algorithms/lwe_programmable_bootstrapping/fft64_pbs.rs#L186-L248
[T3]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/tfhe/src/core_crypto/fft_impl/fft64/crypto/bootstrap.rs#L294-L379
[T4]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/tfhe/src/core_crypto/algorithms/glwe_sample_extraction.rs#L89-L117
[T5]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/.github/workflows/m1_tests.yml
[T6]: https://github.com/zama-ai/tfhe-rs/blob/187fc0b95ab35352422deec6c027d0bd8743b6db/LICENSE
[T-release]: https://github.com/zama-ai/tfhe-rs/releases/tag/tfhe-rs-1.8.1
[F1]: https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/README.md
[F2]: https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/lib.rs
[F3]: https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/Cargo.toml
[F-registry]: https://crates.io/api/v1/crates/fhe
[H1]: https://github.com/FeanorTheElf/fheanor/blob/2ba00539204cc03816dd258b57fad94904f54d41/Readme.md
[H2]: https://github.com/FeanorTheElf/fheanor/blob/2ba00539204cc03816dd258b57fad94904f54d41/Cargo.toml
[O1]: https://github.com/fairmath/openfhe-rs/blob/a60b38a101e9cccbada7629245c6a79a4c65741e/README.md
[O2]: https://github.com/fairmath/openfhe-rs/blob/a60b38a101e9cccbada7629245c6a79a4c65741e/src/lib.rs#L320-L363
[O3]: https://github.com/fairmath/openfhe-rs/blob/a60b38a101e9cccbada7629245c6a79a4c65741e/build.rs
[O4]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/README.md
[O5]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/pke/include/scheme/ckksrns/ckksrns-fhe.h
[O6]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/binfhe/include/rgsw-acc-cggi.h#L79-L99
[O6-impl]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/src/binfhe/lib/rgsw-acc-cggi.cpp#L208-L217
[O7]: https://github.com/openfheorg/openfhe-development/blob/b59e1f291965f14fbee5172cd83a20f5b856204f/docs/sphinx_rsts/intro/installation/macos.rst
[O-release]: https://github.com/openfheorg/openfhe-development/releases/tag/v1.6.1
[S1]: https://github.com/Sunscreen-tech/Sunscreen/blob/f8ee4a0dc32418f2acd9e176c6f0af73cdf68d40/seal_fhe/src/lib.rs#L4-L14
[S2]: https://github.com/Sunscreen-tech/Sunscreen/blob/f8ee4a0dc32418f2acd9e176c6f0af73cdf68d40/seal_fhe/Cargo.toml
[S3]: https://github.com/microsoft/SEAL/blob/79ad62fb50fb59880c7b717275c82fe497634639/README.md
[S4]: https://github.com/microsoft/SEAL/blob/79ad62fb50fb59880c7b717275c82fe497634639/native/src/seal/evaluator.h
[L1]: https://github.com/LIT-Protocol/openfhe-rs/blob/b0b901d0a1f6afb59387e778cbbc6c2fbb483fc9/README.md
[L2]: https://github.com/LIT-Protocol/openfhe-rs/blob/b0b901d0a1f6afb59387e778cbbc6c2fbb483fc9/Cargo.toml
