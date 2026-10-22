# HE client sampling: decision required

**Status: Choice B design/review approved under
[D28](../../paper/methods/DECISIONS.md#d28-secure-he-client-sampling-seam-design-and-review).**
Choice A is not approved. No RNG, implementation, revised dependency pin, key
generation or backend adoption is authorized by this decision.
[D4](../../paper/methods/DECISIONS.md#d4-bounded-poulpy-backend-qualification)
permits bounded functional qualification, not method/security-profile adoption;
the decisions' opening rules also require secure randomness. Resolve this
specific source-contract conflict before implementing original client generation.

## Pinned facts

All source locations below refer to **Poulpy v0.9.0,
[`ca3532570ff62b96599a74f7adb85de7726b0446`](https://github.com/poulpy-fhe/poulpy/tree/ca3532570ff62b96599a74f7adb85de7726b0446)**.

| Concrete interface | Source location |
| --- | --- |
| `Source` wraps `ChaCha8Rng`; explicitly **“Not suitable for cryptographic key generation.”** | `poulpy-hal/src/source.rs:9–29` |
| `GLWESecretSampling::glwe_secret_fill_ternary_hw` requires `&mut Source`. | `poulpy-core/src/layouts/glwe_secret.rs:194–203,237–242` |
| `GLWEAutomorphismKeyEncryptSk`, `GLWETensorKeyEncryptSk`, `GLWESwitchingKeyEncryptSk` require concrete mask/error `Source`s. | `poulpy-core/src/api/encryption.rs:345–378,438–453` |
| `GLWEMaskFill` and `GLWEEncryptSk` also require `Source`; seed-based mask convenience explicitly expands through `Source::new`. | Same file, `24–39,75–102` |
| `CKKSEncryptOps::ckks_encrypt_sk` forwards concrete mask/error `Source`s. | `poulpy-ckks/src/delegates/encryption.rs:24–39` |

Secret/noise delegates derive `source.new_seed()`:
`poulpy-core/src/delegates/sampling.rs:16–23,31–40,48–57`.
Portable kernels then instantiate **another `Source(seed)`** for actual secret
coefficients/noise: `poulpy-cpu-portable/src/sampling.rs:25–36,55–94`.
Fresh OS entropy changes the seed, **not the expansion algorithm or its declared
contract**. This is not evidence of an RNG exploit or a claim that ChaCha8 is broken.

## Choice A — explicitly limited functional qualification

**Proposed approval wording:**

> Permit the pinned Poulpy `Source` only for a non-adversarial D4 functional,
> precision and resource experiment. This is an explicit qualification-only
> exception to the secure-randomness requirement, not evidence of cryptographic
> secrecy or secure backend/profile adoption. Require fresh private 32-byte
> OS-random seeds per existing client domain; prohibit fixed/public secret seeds.
> Preserve the frozen distributions, parameters, key graph and arithmetic, at
> most two refreshes, one CPU thread, 8-GiB cap and existing deadlines/reserves.
> Resource/numerical admission must pass before execution. No production backend
> or paper/method adoption follows.

**Prerequisites:** original client ownership, checked serial export, fail-closed
entropy handling, full-slot numerical checks and complete resource bounds below.
OS-seeding `Source` is not secure-RNG qualification.

## Choice B — require a genuinely secure sampling/encryption seam

**Proposed approval wording:**

> Do not use the disclaimer-bearing `Source` for secret, mask or error generation.
> Permit design/review of a cryptographic RNG seam covering original client
> generation and every internal seed expansion, while preserving the frozen
> distributions, numerical parameters and exact arithmetic. No RNG algorithm,
> revised dependency pin or backend wrapper is selected by this approval.
> Retain D4 resource/role limits. Freeze and approve the implementation and its
> security/resource contract before generation. Secure backend/profile adoption
> remains separate; this permission qualifies neither security nor a method.

**Narrow implementation options, requiring selection/review:**

- **Upstream seam:** reviewed secure `Source` or cryptographic RNG injection
  through secret, noise, mask/encryption APIs **and** portable expansion. Requires
  approved revised pin; no silent stream change under current frozen source.
- **Sampling trait:** `poulpy-core/src/oep/sampling.rs:31–58` exposes unsafe
  `SamplingImpl` with private child seeds and distribution/precision obligations.
  Cannot reimplement the existing foreign backend implementation. A reviewed
  newtype/backend is possible, but this trait alone leaves outer `Source` and
  mask filling unresolved; it is not a complete secure seam or approved backend.
- **Standalone original client:** serial ownership with existing public keygen
  primitives needs no upstream observer API, but still uses `Source`. A secure
  alternative needs audited lower-level sampling/encryption integration, not
  just a client wrapper; no qualified implementation exists.

Before implementing B, specify the chosen CSPRNG/entropy interface, domain
separation, failure/reseed and secret-erasure contracts; verify existing secret,
mask and noise laws/precision through ordinary independent regressions. No
default generator or replacement distribution is proposed here.

## Shared origin and resource prerequisites

- One private client samples/prepares dense secret, encrypts input, and generates
  every evaluation key from that **same dense owner**. Sample sparse secret once
  for both switches. Derive immutable expectations inside original generation/
  export calls—not from imports, tags, hashes or “valid/dense” flags. Digests prove
  byte identity against trusted expectations, not publisher authenticity/role
  authorization. Evaluator receives no secret or seed.
- Existing original calls are visible in
  `poulpy-ckks/src/layouts/bootstrapping_keys.rs:279–308`; ciphertext continuity in
  `src/test_suite/presets.rs:112–165` (same crate). The latter is a **fixed-seed
  reference fixture**, not an approved client. Generate/export/checked-retire one
  full raw key before the next; no eager bundle or regeneration on reload.
  Preserve original capacities and fail-burn lifecycle.
- Known keygen leaf subset is **419,430,400 payload / 419,921,920 charged bytes**,
  including reserved scratch and private tensor heap, **not a complete bound**.
  Retaining an additional original raw export adds **242,270,208 charged bytes**.
  The [1-GiB setup target](../../paper/methods/poulpy-lifetime-admission.md) remains
  a proof target, not an allowance. Full context/solver/codec/ciphertext/source,
  export/kernel/I/O and runtime bounds remain open under either choice.
  Parameter-security and second-refresh entry-conversion gates also remain open.

**Recorded decision: B, design/review only.** Freeze the proposed implementation,
RNG/entropy interface, dependency identity and security/resource contract for
separate approval before generation. Literal client generation, preparation
promotion and production availability remain closed.
