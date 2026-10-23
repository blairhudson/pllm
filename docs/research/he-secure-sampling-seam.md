# Secure HE sampling/encryption seam — review candidate

**Design/review only, under [D28](../../paper/methods/DECISIONS.md#d28-secure-he-client-sampling-seam-design-and-review).**
Choice B is selected; the reference-RNG exception is not. No RNG algorithm,
implementation route, revised source pin, byte budget or backend adoption is
selected here. No implementation or execution evidence is claimed.

The later [design packet](../plans/he-secure-sampling-design.md) distinguishes
declared laws from the frozen dependency's realized shuffle, audits compressed
codec expansion and compares concrete engines. It remains unapproved. In
particular, the declared uniform-support law below is not a qualification of the
default `rand` chooser's documented tiny-bias path; choosing an exact-uniform
correction requires an explicit source-contract disposition.

## 1. Reuse audit before introducing a primitive

| Existing native owner | Reuse | Do not inherit |
| --- | --- | --- |
| `crates/pllm-core/src/gkm/mod.rs:119–156` | Private fallible entropy interface, purpose-separated `getrandom::fill`, zeroizing buffers, unbiased rejection and terminal error. | GKM field law or its 128-attempt limit as an HE sampling bound. |
| `crates/pllm-core/src/gkm/seeded.rs:22–91`, `seeded_window.rs:43–60` | Non-cloneable private stream, checked byte/counter preflight, zeroized state, fresh independent purpose keys and no reset/import. | GKM's AES-128 duplicated-LE64 counter format, sharing assumptions or seed-delivery roles as an approved HE RNG/profile. |
| `crates/pllm-garble/src/shared_arithmetic_reference.rs:27–52` | Fallible OS entropy, secret buffer erasure and public context/domain binding. | Public SHA-256 binding as secret randomness, a KDF or authentication. |
| `crates/pllm-core/src/codec.rs:176–193` | Rejection instead of biased modular reduction; bounded input chunks. | Its `u32` field API, allocating output shape or unbounded rejection loop as the HE implementation/resource contract. |

Root dependencies already declare `getrandom = "=0.3.3"`, zeroizing AES support
and `zeroize` (`Cargo.toml:33,44,53`). This establishes reusable native patterns,
not permission to change the separate HE consumer's dependencies/lock. Prefer a
small shared entropy/lifecycle primitive over importing GKM's protocol owner or
copying a PRG implementation. Independent deterministic tapes belong in private
test support only; no production seed/tape injection or fallback.

## 2. Pinned call graph: every random branch must be covered

Paths below are relative to **Poulpy v0.9.0
[`ca3532570ff62b96599a74f7adb85de7726b0446`](https://github.com/poulpy-fhe/poulpy/tree/ca3532570ff62b96599a74f7adb85de7726b0446)**.
Prefixes: **C** = `poulpy-core/src/`, **P** = `poulpy-cpu-portable/src/`,
**K** = `poulpy-ckks/src/`, **H** = `poulpy-hal/src/`.

| Original operation | Actual path and random leaves |
| --- | --- |
| Dense and once-sampled sparse secrets | `C/layouts/glwe_secret.rs:194–203,237–242,282–295`: `glwe_secret_fill_ternary_hw` → scalar delegate → `P/sampling.rs:25–36` → `P/scalar_znx_fill.rs:68–76` signs and shuffle. |
| Rotation/conjugation evaluation keys | `K/layouts/bootstrapping_keys.rs:279–283` → `C/api/encryption.rs:438–453` → `C/reference/encryption/glwe_automorphism_key.rs:89–107`: transform the **same dense secret**, then GGLWE encryption. |
| Tensor evaluation key | `K/layouts/bootstrapping_keys.rs:286–289` → `C/api/encryption.rs:364–378` → `C/oep/derived/encryption.rs:340–380`: prepare original secret and its tensor, then GGLWE encryption. Tensor preparation (`C/layouts/glwe_secret_tensor.rs:235–321`) changes products/representation, not secret epoch. |
| Both directional switches | `K/layouts/bootstrapping_keys.rs:292–308`: sample sparse secret once and tag `ENCAPSULATED`, then `(dense,sparse)` and `(sparse,dense)`. `C/reference/encryption/glwe_switching_key.rs:86–108` prepares the respective output secret and calls GGLWE encryption. |
| Evaluation-key mask/error leaves | `C/reference/encryption/gglwe.rs:134–155`: every row/input column calls `fill_glwe_mask_from_source` then `glwe_encrypt_sk_internal`; `C/reference/encryption/glwe.rs:487,622` adds `Noise::ENCRYPTION`. |
| Input ciphertext | `K/delegates/encryption.rs:24–39` → `K/reference/encryption.rs:46–65`: `glwe_encrypt_zero_sk`, add encoded plaintext, normalize, retain metadata. Must use the prepared form of the **original dense owner**, not an imported/asserted identity. |
| Uniform masks | `C/api/encryption.rs:24–39`; `C/reference/encryption/glwe.rs:43–62` → uniform-source HAL operation → `P/kernels/vec_znx/sampling.rs:8–34` → `P/kernels/znx/sampling.rs:3–8`. |
| Error expansion | `C/delegates/sampling.rs:31–57` → `P/sampling.rs:55–94` → `P/kernels/noise.rs:558–612` and its table/sign or rejection draws. |

Outer APIs take concrete `&mut Source`, not an injectable secure RNG.
`H/source.rs:9–29` wraps ChaCha8 and explicitly disclaims cryptographic keygen;
`32–46` derives child seeds/branches through that same stream. Secret/noise
delegates (`C/delegates/sampling.rs:16–57`) call `source.new_seed()`; portable
implementations **construct another `Source(seed)`**. Seed-based mask convenience
also specifies `Source::new` (`C/api/encryption.rs:31`). Thus an outer secure RNG,
fresh OS seeds, a new checksum, or authenticated imports alone cannot close this
seam. No claim that ChaCha8 is broken follows from its unqualified source contract.

Compressed-key/seed-reexpansion paths are not selected for this raw-key store.
They must remain unreachable; enabling them later requires the same end-to-end
review, not a retained legacy `Source::new` shortcut.

## 3. Laws and arithmetic that cannot change

- **Secrets:** existing `TernaryFixed(1024)` dense and `TernaryFixed(32)` sparse,
  degree 65,536/rank 1. Declared law: uniform positions, independent unbiased signs;
  realized shuffle law remains unresolved as described above. Keep the
  source's zero/sign/shuffle semantics (`P/scalar_znx_fill.rs:27–33,68–76`). No
  binary substitution, independent-Bernoulli shortcut or resampling on reload.
- **Masks:** centered radix-52 uniform digits at the requested torus precision;
  preserve partial-last-limb masking and zero padding beyond logical precision
  (`P/kernels/vec_znx/sampling.rs:18–34`). No modulo-biased bounded draws.
- **Errors:** `C/noise_distribution.rs:3–26,50–72` defines discrete Gaussian
  conditioned on `|z| <= floor(6 sigma)`, with the exact binary64 value of `3.2`
  for encryption, hence support `[-19,19]`. Preserve the original 128-bit CDT and
  sign handling (`P/kernels/noise.rs:14–37,333–340,558–612`), not a continuous
  Gaussian, decimal-rational replacement or different cutoff.
- **Sampler extension contract:** `C/oep/sampling.rs:5–58` requires distribution,
  placement and precision guarantees; small Gaussian tables permit statistical
  distance at most `B*2^-128`, while larger Gaussians/uniform noise require exact
  sampling. Fixed-child-tape results must remain independent of destination
  column, allocated limb count and preexisting values. Preserve actual source
  thresholds and tables, not merely a weaker approximation meeting that ceiling.
- Backend, decompositions, key graph, operation order, scale/width transitions,
  normalization and independent numerical thresholds remain frozen. Different
  secure random bytes are not permission to change their laws or arithmetic.

## 4. Recommended bounded API shape — not an approved implementation

Recommend an **additive, fallible private-tape path** through the existing
samplers and encryption primitives, rather than a separate encryption algorithm.
Illustrative interfaces, not existing Poulpy APIs:

```text
PrivateEntropy.try_fill(purpose, &mut bytes) -> Result
ClientRandomOwner.open(domain, admitted_draw_budget) -> Result<DomainTape>
DomainTape.try_fill(&mut bytes) -> Result
sample_secret / fill_mask / add_error(..., &mut DomainTape) -> Result
generate_evaluation_key / encrypt_input(..., mask_tape, error_tape, scratch)
    -> Result<original-owner-bound result>
```

Reuse native fallible OS entropy and zeroizing/non-cloneable owner semantics.
Every nested sampler must receive the reviewed tape, or use an explicitly
reviewed child derivation/expander; **never convert back to legacy `Source`**.
Current keygen/secret APIs often return `()`, and shuffle helpers consume an
infallible RNG. Secure variants must propagate failures through every layer;
zero-fill, ignored errors, panic-and-retry or silently continuing partial keys
are not adapters. Reject invalid public geometry before entropy or mutation.

**Unapproved engine choices:** direct OS-backed tape is a reviewable baseline
reusing existing entropy, but its entropy volume/latency needs a bound. A seeded
cryptographic expander needs an explicitly selected algorithm/version, seed and
counter layout, domain derivation, exhaustion/reseed and erasure contract. No
algorithm is selected here; neither GKM's AES format nor a `CryptoRng` marker is
by itself the security review. No public seed-derived entropy or new ad hoc KDF.

**Domain/lifecycle proposal:** independent fresh OS domain material, following
existing purpose separation, or a separately reviewed keyed derivation. Bind
client/session and private epoch to secret-vs-mask-vs-error purpose, key role and
Galois element/direction, operation ordinal and child-draw ordinal. Public labels
are binding metadata, not entropy or authentication. Child placement must not
violate the fixed-tape independence contract above. No clone, seed export,
rewind, counter reuse or restart/resume. Check byte/counter exhaustion before
output; entropy/limit/cancellation/serialization failure burns the entire client
epoch and all pending artifacts, with no in-epoch replacement sample.
Draw budgets must come from the proved shared admission schedule, not an
unchecked caller-provided integer; unknown budgets cannot mint a usable tape.

**Implementation route needing approval:** a reviewed Poulpy patch/new immutable
pin could add this path from public APIs through HAL/portable leaves for the
separate isolated D4 qualification experiment only. It does not authorize linking
an upstream paper runtime into PLLM's independent method path. That path requires
independently implemented shared samplers and HE arithmetic; the design packet
keeps these routes distinct. `SamplingImpl` is an unsafe seed-based
extension, not a complete solution: it leaves outer seed derivation/masks and
fallibility unresolved; the foreign portable implementation cannot be replaced
by a conflicting impl. A backend newtype or independent lower-level client is
a separately reviewed implementation choice, not authorized by D28. Existing
public primitives suffice for serial ownership; they do not supply this secure tape.

## 5. Origin, export and resource contract

The private client must **perform**, not assert, dense sampling and preparation,
all original evaluation-key generation, and original input encryption. Retain
one dense epoch throughout; sample sparse once for both switches and retire/wipe
it only after both exports. Evaluator gets no secret, seed, tape or private
reconstructible state. No process isolation, constant-time implementation,
adaptive erasure or production security is established by this design.

Mint origin-bound immutable records only after successful original generation and
bounded streaming export; bind frozen profile/type/layout/role/epoch/Galois
element, capacities, exact wire length and digest. An import can compare these
trusted expectations but cannot mint them. Checked-retire each full raw key
before the next allocation; keep only immutable records, not all original raw
owners. Atomic complete publication must not authorize partial stores or release
reservations early. Preserve original backing capacities and live-view lifetimes.

Keep D4: **8 GiB**, separate **4 GiB host reserve**, existing guard/deadlines,
one CPU thread, no extra swap, at most two refreshes. Budget RNG state, secret
copies, buffers, rejection work, all caller scratch and private heaps, source/
module/context/codec ownership, export and kernel I/O together. Known keygen
leaf subset **419,430,400 payload / 419,921,920 charged bytes** is not a complete
setup bound; another retained original raw export adds **242,270,208** charged
bytes. The 1-GiB setup target is not an allowance. Byte/rejection limits require
source-backed bounds and abort semantics, not biased fallback; a deadline is a
stop rule, not a completion guarantee. Failed unmap remains charged and invalid.

## 6. Exact next approval and evidence gate

Before implementation, approve a named candidate containing **(1)** concrete
engine/entropy and child-domain contract, **(2)** patch/new-pin versus reviewed
backend-wrapper route, **(3)** complete fallible API/secret-erasure contract,
and **(4)** explicit resource/rejection/draw accounting. D28 approves none of
these choices. Current consumer and production availability remain closed.

Required later evidence, parent-serialized through ordinary tests/builds:

1. Source/import call-graph audit proving zero legacy `Source` fallthrough on
   every selected path; secure-engine/entropy provenance and independent known-
   answer/counter/domain tests, or OS-interface failure tests for direct entropy.
2. Independent fixed-tape parity of unchanged arithmetic, secret weights/signs,
   unbiased shuffle/bounded draws, masks/padding and exact original CDT endpoints
   and sign/precision logic. Histograms alone do not establish these laws.
3. Fault injection at every outer/inner draw and export boundary: no partial
   acceptance, duplicate domain, replay/resume, replacement sample, secret/seed
   delivery or release credit after failed unmap. Demonstrate original dense
   continuity and same sparse owner for both switches, not matching metadata alone.
4. Complete allocator-inclusive setup/online/I/O bounds, independent all-slot
   numerics and re-entry qualification; estimator/parameter security and method
   adoption remain separate. No copied trees, evidence archives or custom verifier.

This review executed no upstream code, RNG, test, build or key generation.

## 7. Newer upstream source check (23 October 2026)

Latest published release/tag remained v0.9.0 at the frozen commit above. Newer
main was inspected at immutable
[`4246d324c57d97d73bf3209d7b832a57454512ec`](https://github.com/poulpy-fhe/poulpy/tree/4246d324c57d97d73bf3209d7b832a57454512ec).
Paths/lines in this section refer only to that revision; earlier frozen-source
anchors and historical qualification results remain unchanged.

- **Secure source still missing:** `poulpy-hal/src/source.rs:1–29,37–46,102–118`
  still uses ChaCha8, disclaims keygen suitability and has `TryRng::Error =
  Infallible`; its queried post-release history was empty. Concrete Source/child
  seed forwarding remains in `poulpy-core/src/delegates/sampling.rs:16–57`, with
  reconstruction in `poulpy-cpu-portable/src/sampling.rs:34,65,86` and uniform-mask
  `hal_defaults/vec_znx.rs:373–374`. Fresh outer entropy does not replace these.
- **Export epoch still separate:** `poulpy-ckks/src/layouts/bootstrapping_keys.rs:322–440`
  generates a key set from a supplied dense secret and three concrete Sources;
  plain key serialization is not the trusted client-origin authority required by
  PLLM's `benchmarks/native/poulpy_refresh/src/staged_identity.rs:440–443`.
- **Opaque capacity still unresolved:** raw/prepared automorphism-key fields stay
  `pub(crate)` (`poulpy-core/src/layouts/glwe_automorphism_key.rs:57–59`,
  `layouts/prepared/glwe_automorphism_key.rs:12–14`). Backend borrows do not retain
  public access to original owning capacity/alignment. `AlignedBuf` already has
  those methods; that does not close the specialized-owner gate in
  `benchmarks/native/poulpy_refresh/src/staged_preparation.rs:37–58`. No inspected
  change unblocks that adapter. The smallest missing seam is read-only access to
  the specialized owner's original typed buffer/Layout, not another shape quote.

A genuine new **leaf** improvement is
[`e97e2ce069545d151936cc81ad5ba84a216424e7`](https://github.com/poulpy-fhe/poulpy/commit/e97e2ce069545d151936cc81ad5ba84a216424e7)
plus alignment/reuse follow-up
[`a8e472ac6d91ac15ef3f0eaf4490160b64a51198`](https://github.com/poulpy-fhe/poulpy/commit/a8e472ac6d91ac15ef3f0eaf4490160b64a51198):
secret-tensor preparation moves formerly private heap temporaries into caller
scratch, counts aligned buffers and wipes before normal return
(`poulpy-core/src/layouts/glwe_secret_tensor.rs:232–249,327`). Key encryption
reuses disjoint preparation/encryption scratch via their maximum. This is not
failure-path erasure qualification, secure sampling, trusted export or a complete
allocator-inclusive setup/CPU/RSS bound. Private/helper ownership remains explicit.

Current main is not a drop-in pin: its `CHANGELOG.md:20–21,124–129,153–158`
records changed tie rounding and gadget/offset key wire formats. No change was
adopted or backported. Remaining authority/evidence is still the approved concrete
secure source route, original client epoch/export, actual owning layouts and
complete admission/qualification in §6; **D28 stays design-only**. This source
check used 20 public requests / 506,356 response bytes, with no dependencies,
builds, native operations, RNG/keygen or models executed.
