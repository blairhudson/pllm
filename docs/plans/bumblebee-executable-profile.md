# BumbleBee B-N1: executable-profile decision packet

Status: **proposal, not approved or executable**. Source-only review, 22 October
2026; no builds, tests, model execution or author-reproduction claim.

## Selected candidate, and the actual decision

Recommend `bumblebee.final-paper-derived-gpt2-mixed.v1` as the **candidate**, with
native-Rust **fhe.rs** at `98a087f1bf5f7fa95196ebe16b0e1b0eeb632442` selected
for **odd-RNS primitive qualification**, not production backend adoption.
Do not choose the pinned OpenBumbleBee configuration or Poulpy's torus preset.

The smallest adoption decision is one named profile with **two complete
attachments**: (1) node-level numerical semantics; (2) exact HE/OT parameters,
key graph and conversion/error contracts. A final-paper label or dependency name
alone supplies neither. No complete pair is established by the audited evidence.

[D7](../../paper/methods/DECISIONS.md#d7-bumblebee-final-paper-target) already
selects the final paper, but requests its evaluated manifest. The irreducible
user choice is: **continue requiring that author-linked manifest, or explicitly
authorize a separately named, fully enumerated final-paper-derived interpretation
without claiming evaluated-build identity**. The latter must resolve the open
fields below; it is not permission to fill them from artifact defaults.
Backend qualification/adoption is a separate scope of that decision. D2's Rust
preference and D4's bounded Poulpy experiment do not adopt this candidate.

## Candidate facts that are actually sourced

Governing bytes: [NDSS final](../../papers/r22-bumblebee-final.pdf), SHA-256
`184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`,
locked in [source-lock.v1.json](../../paper/methods/source-lock.v1.json).

| Boundary | Candidate source contract; not an invented default |
| --- | --- |
| Input representation | Signed two's-complement ring64; `floor(real*2^18)` before sharing (§II.B, p.3; §VII, p.11). Exact imported-float/decimal interpretation still needs binding. |
| GPT2 mixed nodes | GeLU **and exp** use ring32/Q12; remaining base arithmetic ring64/Q18 (§VII, p.11). Conversion order, rescale sites and public range certificates remain explicit inputs. |
| Mixed GeLU | `0.5*x + sum(c[i]*abs(x)^i, i=0..4)` on `[-2.7,2.7]` (§V.C, p.10; App.D, p.17). Constants below. Tail/endpoint lowering is not supplied by that interval alone. |
| Shifted exp | Protected row maximum; cutoff `-14`; six protected squares of `1+x/64`, including protected division/truncation by 64 (§V.B; evaluated settings p.11). Do not use illustrative cutoff `-13` or artifact four-square schedule. |
| OLT ring and scale | `Z_q[X]/(X^8192+1)`, message modulus `t=2^64`; App.E gives only `q≈2^147` and automorphism modulus `q'≈2^49`. These are **not ordered primes, a full chain or a decomposition**. Message/HE scaling is distinct from Q18. |
| Coefficient H2A | Uniform polynomial mask over **all coefficients of R_q**, plus fresh public-key encryption of zero before decryption. Masking sender outputs `-round(t*r/q) mod t`; key holder rounds masked phase (App.C/Alg.4, p.16). No replacement by plaintext-ring masking or universal exact reconstruction. |
| SIMD bOLE/H2A | `N=8192`; `q≈2^148`; three approximately 45-bit plaintext CRT primes (App.E). Abstract `Lift`/`Down` are nearest-scaled conversion; uniform plaintext-ring share masking plus noise flooding is a different H2A path (Alg.2/App.C). `B'=2^98` is printed, but does not freeze sampler endpoints or implementation width. |
| Keys and OT | Each OLT encryption direction retains its own decrypting owner's secret key; peer receives public/evaluation material, not that secret. Shared×shared products require two cross terms. Ferret distributed OT, no trusted/local OT dealer; 16-way OT messages and the paper's stated wide 50-bit MSB optimization remain distinct from mixed-node precision (pp.3,5,9,11,18). Exact evaluation-key inventory remains open. |

App.D's mixed-GeLU coefficient strings, without new fitting or implicit binary32
conversion:

```text
c0 =  0.001620808531841547
c1 = -0.03798164612714154
c2 =  0.5410550166368381
c3 = -0.18352506127082727
c4 =  0.020848611754127593
```

## Fields that still prevent adoption

These require author evidence or an explicit, documented user-selected
interpretation; this packet assigns no guessed values.

| Required manifest field | Exact unresolved choice |
| --- | --- |
| Constants and nonlinear branches | Decimal versus frontend floating representation; coefficient/bias scales and accumulation/truncation order; mixed-GeLU tails/endpoints and retained comparison precision. If wide GeLU is used, reconcile Eq.(2)'s epsilon-offset tails with Alg.3's added epsilon and missing final subtraction. Neither ±3 artifact GELU nor a deterministic pivot oracle follows from final-paper mixed GeLU. |
| Protected rescaling/comparisons | Bind truncation protocol/error flags and share-dependent comparison law at every node. Pinned candidate truncation has errors `{-1,0,+1}` with a value-conditioned law, **not** PUMA's `TruncAPr`; candidate dropped-bit comparisons are share-dependent even at equality. Their final-paper association is unproved. |
| Mixed extension | Select actual cast/downscale order, extension protocol/offset and certified intermediate ranges. Artifact `M=2^(source_width-2)` requires `[-M,M)` at extension; it is evidence, not the paper's evaluated M. No silent general-extension substitution or clear clipping. |
| Normalization | Sum versus mean-square denominator, epsilon value/encoding, placement, reciprocal/rsqrt iteration and truncation order. Checkpoint-compatible LayerNorm cannot be inferred from the model name. |
| HE and error model | Ordered OLT/bOLE primes; data/key/decryption levels; secret/error distributions; key-switch decomposition, automorphism indices and both owners' key epochs; exact flooding sampler; all-coefficient mask freshness; worst admitted accumulated noise and approximate H2A/CRT error. No security bits or failure tolerance selected here. |
| Frontend/source | Exact GPT2 checkpoint/tokenizer/frontend and lowered operations, including norm and protected head/argmax. D9's lowest-index rule defines the new transaction, not historical frontend equivalence. |

The incompatible artifact is immutable
[`OpenBumbleBee@47c5560d069543591f3b0176eb530ede28e33fc3`](https://github.com/AntCPLab/OpenBumbleBee/tree/47c5560d069543591f3b0176eb530ede28e33fc3).
[Source-resolution §4](../../paper/reproduction-contracts/shared-profile-resolution.md#4-bumblebee-conversion-choices-are-concrete-final-paper-identity-is-not)
already derives its toward-zero encoder, Q16/four-square example, ±3 mixed GELU,
different SiLU, OLT nominal 218-bit key/163-bit data context and enabled-error
bOLE 159-bit ciphertext/three 47-bit plaintext contexts. Its unsigned Lift is
nearest/ties-up; Down by odd modulus has no exact half ties. These are **artifact
facts only**, not adopted final-paper rounding or cryptographic parameters.

## Backend candidate and exact reuse boundary

New targeted inspection of the already pinned native-Rust candidate gives a
concrete reason to qualify it for this method, independent of CKKS refresh:

- [`bfv/parameters.rs`](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/parameters.rs):
  `plaintext_modulus(Into<BigUint>)`, explicit ordered `ciphertext_moduli`, and
  degree 8192 are expressible. Thus `t=2^64` is not excluded by a u64-only
  parameter API. This is static expressibility, not a successful construction.
- [`bfv/ciphertext.rs`](https://github.com/tlepoint/fhe.rs/blob/98a087f1bf5f7fa95196ebe16b0e1b0eeb632442/crates/fhe/src/bfv/ciphertext.rs):
  public `components`, `into_components`, `from_components` expose the raw
  polynomial boundary needed to investigate coefficient masking/trace adapters.
  `trivial_zero` is explicitly not fresh randomized encryption of zero.
- Its configurable noise is **centered binomial** (default variance 10), not
  evidence for source error/flooding distributions. Do not inherit that default,
  automatic prime generation, security claims, scaling, or key-switch behavior.
  Raw Galois application, both H2A paths, exact rounding and complete owner bounds
  still need qualification. Experimental/maturity and Apple-M gates in the
  [backend review](../../paper/methods/rust-he-backend-qualification.md) remain.

Choose generic primitives only; independently implement BumbleBee protocols.
No bootstrap is required by the paper's OLT/bOLE primitive set. A CKKS-refresh
candidate's success/failure therefore does not decide this backend. Poulpy's
power-of-two torus cannot silently implement odd-modulus inverse trace division.
Existing bounded SEAL/BFV adapters also do not admit the source profile; foreign
backend adoption would require a separate explicit decision.

Reuse, extending shared contracts rather than creating a BumbleBee execution stack:

| Existing owner | Precise next boundary, not current capability |
| --- | --- |
| `crates/pllm-core/src/coefficient_layout.rs`, `coefficient_repacking.rs` | D21-corrected packing/InterLeave/ParseMat and odd-limb division. Layout degree currently stops at **4096**, below source 8192. Generalize with backend/resource qualification; do not change source N to fit. |
| `crates/pllm-core/src/coefficient_conversion.rs` | Clear `Delta=floor(q/t)` phase algebra and shared nearest `down`, not encryption/H2A. **bits=64 admits no q** under its `2*t*(q-1)+q<=u128::MAX` gate; source q also exceeds u128. Extend a shared wide/CRT conversion owner, not independent per-limb Down or smaller source parameters. |
| `crates/pllm-compiler/src/encrypted_level_schedule.rs` | Reuse SSA identities, immutable parents, full backing liveness and persistent/scratch owner charging. Current execution variants are `Ckks`, CKKS-oriented `Hybrid`, and `View`: add a method-neutral reviewed RLWE/share-conversion contract where needed, not fake CKKS depth or a free conversion. Trace consistency does not prove noise, privacy or key custody. |
| Ordinary component/source/runtime path | D5's `ComponentRef` → Pipeline/Experiment → native source admission → party-local execution → SDK/gateway/CLI/benchmarks. Reuse composition, source binding, channels and cancellation; not PUMA's three-party arithmetic or a local OT dealer. `BumblebeeTwoPartyTransformer` remains pending. |

**Unlock condition:** supply the author-linked numerical/HE manifest, or approve
the named source-derived interpretation with every open field above explicitly
resolved, plus the precise scope of native-Rust primitive qualification. Only a
qualified backend/parameter/error/key contract permits protected integration;
qualification permission alone does not promote SDK availability. Keep D9's
fixed-count full-prefix private feedback and final client-only token IDs, D21's
index corrections, and original GPT2 versus D3 common-Qwen identities unchanged.
No new runtime, numeric, decision-record or capability change is made here.
