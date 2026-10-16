# Closest-comparator qualification: Gupta–Katz–Miers

**Documentary review: 16 October 2026. Status: mandatory comparator; complete
matched execution and measurements unresolved.** This card qualifies ePrint
2026/1849 against PLLM's exact W8A8 Qwen2.5 compiled decoder and four-provider
`PrivateLookupAndFullHead` plus prepared-body candidate. It supplies neither a
new runtime nor a security/performance certificate. Missing code paths, differing
contracts and failed/missing evidence remain unresolved cells, not exclusions.

## 1. Identity, source availability and evidence boundary

- **Paper:** Kanav Gupta, Jonathan Katz, Ian Miers, *Private and Verifiable
  Outsourcing of Open-Weight LLM Inference*, ePrint 2026/1849. Existing cached
  PDF `papers/r19-open-weight-verifiable.pdf`, SHA-256
  `e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d`.
  Preserve the original `prior-art-sources.json` entry and failed fresh-PDF
  retrieval. No new revision or fresh-PDF byte identity is inferred.
- **Author repository:** [kanav99/dvllm](https://github.com/kanav99/dvllm), commit
  [`271637be7196d26a5433ca1e62b1a5357e769a5b`](https://github.com/kanav99/dvllm/tree/271637be7196d26a5433ca1e62b1a5357e769a5b),
  dated 2026-10-01; tree `e742e67ed278aba39a440a3d78c782134fe8732c`.
  Repository metadata links the ePrint as its homepage; owner metadata identifies
  Kanav Gupta. These bounded mutable discovery fields are retained in
  [repository-record.json][discovery]. The repository's MIT license text is pinned;
  dependency/license completeness and identity with the paper's evaluated build
  are not established. No absence claim follows from an earlier missing link.
- **Quarantine:** [archive README][archive], [code hashes/extracts][code],
  [primary locators][primary], [local locators][local]. Twelve selected upstream
  files were read as bytes/text. Only bounded extracts are retained, not a vendored
  build tree. Upstream code is specification/oracle input only: never built,
  executed, imported or linked. The checker reads bytes and optionally retrieves
  pinned HTTP content; it does not invoke an inference/operator implementation.
- **Citation convention:** `U path:start-end` identifies inclusive lines at the
  pinned upstream commit; `L path:start-end` identifies the separately hash-pinned
  local working-tree file. Both resolve in the linked locator records. `P6` means
  PDF page 6, selected with `pdftotext -f/-l`, not a count of extracted form feeds.
  The README's introductory single-server/local-embedding description
  (`U README.md:1-32`) is not used as a complete protocol specification.

## 2. Construction and numerical qualification

**Complete semi-honest restriction.** Section 4.1/Fig.1 ([P6], [P7]) shares each
linear input as `x = x0 + x1 mod N`; workers holding public `M` return `M*x0`
and `M*x1`; client reconstructs the complete vector. Section 4.3/Fig.3 ([P9],
[P10]) shares a **dense Boolean one-hot selector by XOR**, evaluates it over
bit-string table rows, and XORs replies. The explanatory prose on P10 supplies
`x1 = x XOR x0`; Fig.3's step 2 has an apparent self-reference typo. Neither an
explicit DPF substitution nor garbled lookup is specified there. Section 4.4/
Fig.5 ([P11]) composes these with client-local remaining layers, using the same
two servers. Omitting verification is a disclosed semi-honest restriction of this
complete construction, not an invented embedding-only comparator.

**Paper numerics versus inspected code.** Section 3 defines nearest-integer
rounding, without a tie rule; §4 uses `Q(x) = round(2048*x)`, `N = 2^64` and
output rescaling by `2048^-2` ([P6]). Section 5 uses int16 weights/int64 vector
entries ([P12]); the code declares those types/scales in
`U include/constants.hpp:1-8`. In contrast, `U src/quantize.cpp:123-145` and
`U include/client.hpp:211-279` use floating-to-integer casts, which truncate
toward zero for representable values. Out-of-range behavior and wrapping must be
qualified separately; the paper's no-final-overflow condition is not a per-model
bound established by this review. The same quantizer writes float embeddings
and a separate quantized classifier even when the original weights are tied.
XOR-PIR returns float32 embedding bits, not quantized-row/scales bytes.

PLLM instead uses per-row float32 scales, ties-to-even rounding and clipping to
`[-127,127]` (`L crates/pllm-core/src/codec.rs:142-174`), with a specified ordered
float32 dequantization (`L python/pllm/runtime/quantization.py:179-202`). Its
private lookup reconstructs i8 rows and original scale bytes; head reconstructs
ring32 outputs (`L python/pllm/runtime/private_boundary_transport.py:307-383`).
Fig.1 is algebraically usable for this numerical computation, but changing to
W8A8/ring32 is an adaptation, not unchanged source numerical execution.

**Nonlinear, output, state and sampler support.** Fig.5 permits arbitrary
client-local remaining computation. The inspected client provides RMSNorm and
softmax (`U include/client.hpp:142-175`), RoPE/attention/SwiGLU/residual execution
and a remote full-vocabulary classifier (`U include/client.hpp:403-538`). Batched
prefill returns the last position's complete vocabulary logits
(`U include/client.hpp:682-689`), not all prefill-position logits. Client buffers
include full float32 key/value caches and vocabulary logits
(`U include/client.hpp:56-76`). Thus full-output/full-KV source support is
established for these paths; parity with Qwen's active state, continuation and
failure behavior is not. Model dimensions are configurable, but that alone is
not Qwen import/operator support (`U include/common.hpp:1-35`).

Greedy, temperature, multinomial/top-p and xorshift sampling occur in
`U src/llama3.2q.cpp:313-436`. Compare against
`L python/pllm/runtime/transformer_client.py:1828-1846`: PLLM uses float64
probabilities and its own RNG/choice and top-p cutoff. The source includes the
threshold-crossing token; its stochastic sampler mutates logits in place.
Capture logits **before** sampling when comparing. The source RMSNorm epsilon
is hardcoded `1e-5`; Qwen's compiled coefficients, epsilon, RoPE, attention,
biases and state layout must be bound independently. Matching labels such as
"RMSNorm", "top-p" or "W8A8" is not equivalence.

## 3. Privacy, verification and narrow source concerns

The paper assumes an honest model owner, public weights and non-colluding servers
for privacy; either/both servers may be malicious and colluding for its
verification claim (§3, [P6]). There is no model-confidentiality guarantee to
preserve. Pure Fig.1/Fig.3 sharing uses uniform shares; §5 ([P12]) compresses one
share using a PRG seed. The inspected client streams PRG shares and sends a seed
to secondary workers (`U include/client.hpp:211-279`,
`U include/client.hpp:288-340`, `U src/llama3.2q.cpp:662-695`). A fair adaptation
must retain or separately price that optimization and batching.

Prompt length, step bound, stage schedule, geometry, sizes and timing are visible
in the inspected protocol. Lookup caching may additionally expose query omission/
cache-hit information; do not silently adopt its cache when aligning leakage.
The inspected networking helpers use sockets (`U include/networking.hpp:117-218`);
this is not evidence of an authenticated confidential deployment channel, operator
independence, side-channel protection or a formal bounded-leakage theorem.

**Offline publication is not offline verification.** Section 4.2/Fig.2
([P7], [P8]) publishes `R_i = g^r_i`, `S_j = g^(M^T*r)_j` under a discrete-log
assumption in a group of prime order `p >> N`. The client checks
`product(S_j^x_j) = product(R_i^y_i)`, individually or using randomized batches
across calls/layers. Section 4.3/Fig.4 ([P10]) publishes row hashes, then checks
recovered embeddings online. Fig.5 ([P11]) requires rejection/abort before final
accepted output. Section 5 names secp256k1 and SHA-256 ([P12]). Distinguish:

1. Original verified construction and its delayed/batched acceptance semantics.
2. Explicit semi-honest restriction omitting verification, with no result-integrity
   claim. Current source calls the verifier unconditionally on the inspected
   path; this review does not establish a ready-made verification-off executable.
3. PLLM's prepared-body verification options or row-width certificate, neither of
   which establishes equivalence to this complete verification construction.

The inspected CLI drains its verification queue and loads commitments before the
timed `Verify()` block (`U src/llama3.2q.cpp:662-695`). That printed timing is not
the full verification lifecycle cost; account loading, queue ownership and final
acceptance separately when collecting a comparable verified cell.

The following are **inspected-path concerns requiring resolution**, not a complete
artifact audit, demonstrated exploit, or verdict about the abstract protocol:

| Concern | Exact evidence | Qualification consequence |
| --- | --- | --- |
| Privacy RNG / verifier randomness | `U src/llama3.2q.cpp:627-640` seeds PRNGs from `rand()`; `U include/verification.hpp:59-64` does likewise for batching. `U ext/simdcrypt/src/PRNG.cpp:43-75` directly keys AES from the supplied seed and expands counters. | Secure entropy acquisition, seed secrecy/independence, stream separation and restarts are not established by these paths. Audit the complete path; independently implemented adapters need a reviewed CSPRNG contract. Do not infer every entrypoint's seeding or an observed attack. |
| Abort propagation | Embedding mismatch prints an error and the single-token path caches the result (`U include/client.hpp:288-340`, `U include/client.hpp:377-399`). Matrix failures print diagnostics in `U include/verification.hpp:147-165`. Generation precedes the final verification call in `U src/llama3.2q.cpp:662-695`. | These inspected paths do not establish Fig.5's end-to-end rejection/accepted-output contract. Resolve return/exception/streaming semantics before any verified qualification; do not transfer this observation to all wrappers or the paper proof. |
| Numeric/build correspondence | Truncating casts above differ from the paper's nearest-integer definition; signed int64 share arithmetic is present in `U include/client.hpp:211-279`. | Freeze the intended rounding, finite/range checks, signed reconstruction and wrapping semantics in an independent implementation. No compiled behavior was tested; signed overflow/build semantics remain unresolved. |

Whole-client ownership includes full KV, temporaries, logits, tokenizer, local
embedding cache, commitments and asynchronous verifier copies/queue
(`U include/async_verify.hpp:20-81`). The abstract's **179 MB local storage**
([primary PDF, p.1][pdf]) and the **4 GB client VM** ([P12]) establish neither
whole-client peak RSS nor eligibility/ineligibility under 256 MiB.

## 4. Comparator qualification matrix

Labels below are distinct artifacts/claims, not interchangeable baseline names:

- **O — original numerical construction/source reference:** fixed-point/ring64,
  float embedding, dense XOR-PIR. Keep verified and explicitly restricted
  semi-honest subvariants distinct; resolve paper/source rounding disagreement.
  Upstream remains documentary input to independent implementations.
- **A — proposed exact-W8A8 GKM-derived adaptation:** same two-worker complete
  lookup/body/head structure and dense XOR-PIR, but PLLM's exact quantized
  row/scales, ring32, local nonlinear/state/sampler contract; no verification claim.
- **C — stronger packed-DPF architecture control:** A's complete functionality
  with BGI-attributed compact/packed DPF, optional native owner/kernel/serialization
  optimizations. "Stronger" means a less handicapped performance control, **not**
  stronger security. It is not the original GKM source implementation.
- **P — current PLLM candidate:** prepared body plus two extra private-boundary
  providers, four roles overall; exact contract and costs still need the common
  cross-topology gates in [COMPARISON-CONTRACT.md](COMPARISON-CONTRACT.md).

| Qualification axis | O: original | A: exact derived | C: packed-DPF control | P: prepared-body candidate / unresolved comparison |
| --- | --- | --- | --- | --- |
| Complete functionality | Fig.5; inspected remote lookup/full head and local full KV | Required complete compiled decoder, not body-only | Same requirement; isolated DPF/head timing insufficient | Coupled path exists; candidate-only execution does not measure comparator |
| Model/checkpoint | Author evaluation models; evaluated-build identity unresolved | Pin same Qwen2.5 configuration, source/checkpoint/tokenizer artifacts | Same pins as A | Same pins, source publication and exact workload required |
| Numeric/rounding | Scale 2048, int16/int64; nearest vs cast discrepancy unresolved | Per-row W8A8/ring32, exact scales/rounding/overflow | Same as A | Same as A; identity to original FP/BF16 is a separate quality claim |
| Lookup/output | Dense XOR-PIR over float rows; full head | Dense XOR-PIR over specified quantized row/scale encoding; all logits | Packed DPF over same rows; all logits | Packed private lookup/full head; row certificate is not output verification |
| Client nonlinears | RMSNorm/RoPE/attention/softmax/SwiGLU etc. on inspected paths | Exact compiled coefficients, operations and ordering | Same as A | Same exact numerical computation required |
| State/sampler | Full KV and local greedy/top-p exist; Qwen equality unresolved | Active KV, positions, history, pending token, RNG, EOS, continuation, cancellation | Same as A | Existing state witnesses are not automatically complete across every contract field |
| Parties/setup | Two replicated public-weight servers, honest owner; reusable public commitments for verified subvariant | Two workers; fresh secret shares/queries; explicit owner/source provisioning | Same two workers with additional DPF/PRG assumptions | Preparation, Inference, A, B; fresh prepared material/custody/erasure and source publication |
| Privacy/collusion | Non-collusion for privacy; verified integrity allows collusion in paper | Joint view of one worker across lookup/body/head must be reviewed | Same plus DPF proof/encoding/stream binding | Common singleton class and stronger cross-pair class require distinct composition evidence |
| Leakage/channel | Length/schedule/size/timing; source cache policy and channel unresolved | Predeclare equal leakage, cache/termination and channel contract | Same as A | Charge/list certificates, stage/access metadata and cache/prefix policy |
| Integrity/abort | Owner commitments plus online/batched client checks; source abort concerns above | Semi-honest only, explicit omission | Semi-honest only unless separately implemented/reviewed | Current coupled admission rejects verification; no transferred malicious guarantee |
| Randomness/reuse | Source seeding concern unresolved; §5 PRG compression | Fresh reviewed entropy, domain/role/session/step separation, no share reuse | Plus DPF key lifecycle/cancellation retirement | Prepared inventory plus boundary keys need joint lifetime analysis |
| Tied-owner lifetime | Quantizer writes float embedding and int16 head separately; runtime allocation audit incomplete | Quantized representation can share owner; demonstrate actual allocation lifetime | Same; single native owner is optimization to measure | Existing shared-owner mechanisms do not prove complete process peaks |
| Native/geometry support | CUDA/int16/int64 author implementation; no upstream execution here | Model-neutral i8 kernels/compiled schedule seams; composition not admitted today | Same; packed DPF bounds and tail handling additional | Current CPU/W8A8 bounds, request/state-audit caps and placement restrictions visible |
| Client memory | Storage and VM numbers are not RSS; full measured peak unknown | Fresh-process complete-client lifetime maximum required | Same as A | Include SDK/tokenizer/audit/helpers through finalization; Low = 268435456 bytes |
| Communication/resources | Paper Table 2 is author-reported own setup | All directed TCP-stream bytes; raw bodies separately; same aggregate provider/access budget | Same; retain seeded sharing/batching and price DPF/server work | Sum all four providers and preparation; role count/capacity cannot be free |
| Lifecycle/performance | Per-token costs amortized over 100 tokens; verification column covers all 100 ([P13]) | Source-present cold, warm-public/fresh-private, source-cold; prefill/decode/first-text/full request | Same workload, lifecycle, controls and repeats | Candidate-only listing and archived different-scope cohorts cannot fill O/A/C cells |
| Quality/status | Original numeric track needs own clear oracle and predeclared quality comparison | Proposed; exact conformance and security binding unresolved | Proposed architecture control; source mapping/deviations must be published | Current candidate, not a closest-comparator win; P2-A remains open |

## 5. Faithful adaptation versus architecture control

Colocating each worker's body share with its lookup/head share is faithful to
**the paper's two-server party structure**. It does not require four workers.
Worker A may own body-A plus boundary-A and B own body-B plus boundary-B, with
fresh/domain-separated randomness and a proof for each worker's combined view.
Two processes on one host do not demonstrate independent operators.

Existing two-offset plus `PrivateLookupAndFullHead` is initially **C**, not O or
an already qualifying A. W8A8/ring32, quantized embedding/scales, packed DPF,
SHAKE/AES choices, wire compression, kernels, batch schedule and client runtime
are explicit changes. A construction-by-construction map can justify the name
"GKM-derived semi-honest W8A8 adaptation with BGI-style DPF substitution" for C;
it cannot justify "original GKM reproduced". Keep A's dense-PIR control if
attributing improvements specifically to compact DPF. Omitting verification must
not silently change the common privacy/leakage contract or subtract measured
verified costs to invent an unverified result.

## 6. Model-neutral integration seams and hard admission barriers

This is an implementation specification for a later authorized independent pass.
No source/runtime change is made by this review.

| Seam | Existing support and exact locator | Gate before integration can qualify |
| --- | --- | --- |
| Compiler composition | `L crates/pllm-compiler/src/decoder_boundary.rs:160-195` requires explicit prepared boundary roles and CPU W8A8; `L crates/pllm-compiler/src/lib.rs:2123-2206` restricts offset slots; `L python/pllm/profiles/__init__.py:690-712` separately restricts composition. | Add an explicit model-neutral offset-plus-boundary topology/admission rule, validated cost floors and complete source/plan binding. Do not bypass guards or alter privacy digests to force ranking. |
| Runtime ownership/session | `L python/pllm/runtime/model_binding.py:368-409` requires `PreparedRemoteLinear` plus preparation lease; `L python/pllm/runtime/client.py:1109-1159` validates four-role boundary endpoints/credentials. | Admit bound offset transport and paired boundary context with explicit role-to-worker mapping; preserve replay, failure, cancellation and cleanup ownership across both services. |
| Semantic call hooks | `L python/pllm/runtime/semantic_executor.py:1131-1155` provides full-head delegation; existing boundary lookup/head reconstruct exact client values. | Route generic semantic operations through A or C; preserve full logits, biases/scales and state/sampler semantics, rather than special-casing Qwen tensor names. |
| Vocabulary bundle | `L python/pllm/runtime/offset_worker.py:157-163` exports offset client bundle; `L python/pllm/runtime/model_binding.py:1928-1960` requires local head weights unless boundary admitted. | Remove lookup/head matrix payloads under the new explicit contract; retain source identities, small coefficients and public scales. Demonstrate absence of duplicate client vocabulary allocations. |
| Shared worker owner | `L crates/pllm-python/src/private_pages.rs:26-48` exposes `PrivatePageServer.from_matrix`; `L crates/pllm-garble/src/private_pages.rs:413-485` supports shared immutable ownership. | Serve body and boundary roles in each authenticated worker with shared loaded owner; prove allocator/lifetime behavior and account public table/scale/certificate storage. |
| Numeric primitives | `L python/pllm/runtime/native.py:72-178` stores i8 weights and provides both `wrap32` **and `wrap64`**. | A/C can reuse i8 primitives. O's int16 matrices cannot be plugged in unchanged; implement independently or leave unresolved. Do not misreport absence of ring64 support. |
| Envelope/deployment | `L python/pllm/runtime/offset_worker.py:31-37` caps outputs at 32 and tensors at 4M elements; `L crates/pllm-compiler/src/network_placement.rs:233-241` rejects current boundary live-network placement. | Admit only checked shapes; expand limits only with separate conformance. An unadmitted long decode or network cell is unresolved, not a performance loss. |

## 7. Concrete integration and measurement gates

1. **Documentary gate (this slice):** preserve primary records; pin source commit,
   bytes, provenance, exact locators and bounded extracts; validate JSON/citations
   and rerun the read-only hash check. This establishes source access and the
   stated static observations only. Scientific interpretation still needs review.
2. **Independent source-to-adapter gate:** specify O/A/C before implementation;
   map every Fig.1/Fig.3/Fig.5 step and §5 optimization to independent code, with
   deviations and optional verification explicitly recorded. Resolve rounding,
   signed bounds/wrapping, randomness, cache leakage and accepted-output/abort
   contracts. Upstream stays specification/oracle text, never a runtime dependency.
3. **Complete exact/security gate:** same checkpoint/config/tokenizer and compiler
   schedule; compare complete logits before sampler mutation, active KV,
   positions/lengths, executed history, pending tokens and RNG against the exact
   clear decoder plus an independent numerical oracle. Include greedy and declared
   stochastic trajectories, prefill/decode, EOS, continuation, cancellation,
   replay and failure retirement. Review combined per-worker views and the
   candidate's four-role composition under named collusion/leakage classes.
   Original numerics require a separate predeclared quality-matched track if
   exact equivalence is not established; a mismatch is not exclusion.
4. **Parent-owned serial measurement gate:** only after qualification, freeze
   workload/envelope and repeated fresh-process cells. Measure complete-client
   lifetime/process-group peak through finalization, provider peaks/CPU/device
   resources, all directed TCP-stream traffic and application bodies separately.
   Charge setup, source/certificate/commitment publication and delivery, fresh
   private material, verification/queues where enabled, and cleanup. Compare
   source-present cold, warm-public/fresh-private and source-cold cases; publish
   prefill, first text, full request, decode and output-length crossover. Match
   aggregate provider and client-access capacity across two versus four workers,
   retain failed/pressure attempts and unrounded maxima, and apply the common
   contract's quality/statistical rules. More candidate-only Qwen runs or isolated
   operator costs do not close this gate.

**Next concrete gate:** independently implement and admit the reviewed two-worker
complete-decoder A/C contract, then pass exact/security conformance before the
parent serially collects matched cells. Until then: **closest-comparator gap
unresolved**, with source discovery and documentary corrections completed.

[archive]: artifacts/source-review/2026-10-16-gkm-qualification-v1/README.md
[code]: artifacts/source-review/2026-10-16-gkm-qualification-v1/code-lock.json
[primary]: artifacts/source-review/2026-10-16-gkm-qualification-v1/primary-locators.json
[local]: artifacts/source-review/2026-10-16-gkm-qualification-v1/local-locators.json
[discovery]: artifacts/source-review/2026-10-16-gkm-qualification-v1/repository-record.json
[pdf]: ../papers/r19-open-weight-verifiable.pdf
[P6]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-6.txt
[P7]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-7.txt
[P8]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-8.txt
[P9]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-9.txt
[P10]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-10.txt
[P11]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-11.txt
[P12]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-12.txt
[P13]: artifacts/source-review/2026-10-16-gkm-qualification-v1/gkm-pdf-page-13.txt
