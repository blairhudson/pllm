# Secure HE sampling: approval packet, not implementation

**Status: proposed design for review under D28; nothing below is adopted.**
[D28](../../paper/methods/DECISIONS.md#d28-secure-he-client-sampling-seam-design-and-review)
and [Choice B](he-client-sampling.md) authorize design/review only. This document
selects a **recommendation**, not an approved algorithm, dependency, implementation
or execution. Original client generation, preparation promotion and encrypted
execution remain closed. Choice A, including an OS-seeded ChaCha8 exception, is
not approved.

Only this new document is owned by this review. Existing staged HE files and
decision records are unchanged. Evidence here is read-only source inspection;
no dependency resolution, installation, RNG implementation, generation, build,
test, benchmark, encrypted operation or commit was performed.

## 1. Requested decision and architecture boundary

Recommend the following indivisible design contract for a later implementation
proposal:

1. A small **model-/method-neutral, independently implemented** private entropy,
   byte-expansion and integer-sampling boundary, with checked consuming ownership.
   Preserve the laws and arithmetic in sections 3–4, subject to the explicit
   shuffle law/implementation conflict below. Do not create a paper-specific
   RNG, client runtime or second admission system.
2. Fallible OS entropy and **20-round ChaCha**, 256-bit keys, for both parent seed
   streams and every coefficient/mask/error child. No fallback to `Source`,
   ChaCha8, public fixtures or an unreviewed platform default.
3. Uncompressed original evaluation-key exports only. No seed-compressed codec,
   stream-state serialization, root/child-seed export or regeneration on reload.
4. Original dense/sparse ownership, checked serial export, zeroization and
   epoch-wide fail-burn; complete preconstruction resource admission.
5. Approve the exact implementation/dependency identities and security/resource
   proof separately **before** generation. Review approval alone permits no code
   or dependency changes under the present assignment.

**Upstream is specification/oracle input for the paper runtime.** Do not vendor,
copy an upstream runtime, or link Poulpy/another upstream HE implementation into
the independently implemented paper path. A cryptographic building-block crate
is not a paper runtime. Existing isolated Poulpy D4 qualification remains its
own bounded primitive-library experiment; it grants neither a revised pin nor
paper-runtime linkage, method adoption, or a production security profile.

### Reuse audit

Existing PLLM contracts already supply useful patterns:

- `Cargo.toml:32–53` and `crates/pllm-core/Cargo.toml:15–25`: existing `aes`
  with `zeroize`, `getrandom =0.3.3`, `sha2`/`sha3`, and `zeroize` dependencies.
- `crates/pllm-core/src/gkm/mod.rs:119–150`: private purpose-typed entropy,
  fallible `getrandom::fill`, guarded temporary bytes and rejection, without a
  production seed-import interface.
- `crates/pllm-core/src/gkm/seeded.rs:22–91`: non-cloneable consuming stream,
  checked byte limit before writes, redaction and cipher/buffer erasure. Its
  AES-128 duplicated-LE64 counter and GKM sharing profile are **not** a generic HE
  sampler and must not be silently relabeled or changed.
- `crates/pllm-core/src/maverick_batch/memory.rs:164–220`: reserve before buffer
  construction; original backing is erased/dropped before reservation release.
- Existing D4 allocator/storage/identity work is qualification-only. Reuse its
  original-capacity accounting and fail-burn semantics, not its byte digest as
  authority to invent original sampling provenance.

Extract shared contracts where suitable; retain existing method identities,
PRSS/AES transcripts and private source owners. No existing inspected interface
already supplies the complete secure HE sampling/encryption seam.

## 2. Frozen-source audit: complete relevant seed graph

All Poulpy paths/line numbers in this document refer to **v0.9.0, commit
`ca3532570ff62b96599a74f7adb85de7726b0446`**, not moving upstream HEAD.
Read-only local checkout:
`/Users/blairhudson/.cargo/git/checkouts/poulpy-66ed1d416543fd59/ca35325/`.
[Frozen source tree](https://github.com/poulpy-fhe/poulpy/tree/ca3532570ff62b96599a74f7adb85de7726b0446).
This review did not independently rehash that checkout; identity revalidation
belongs to the eventual source/dependency freeze.

### 2.1 Concrete outer boundary and every internal construction on this path

| Boundary / symbol | Frozen source | Required disposition |
| --- | --- | --- |
| `Source::{new,branch,new_seed}` | `poulpy-hal/src/source.rs:18–47` | `ChaCha8Rng::from_seed([u8;32])`; `branch` returns both the raw child seed and another `Source`. Replace the entire seed/expansion contract, not merely the entropy feeding it. |
| `GLWESecretSampling::glwe_secret_fill_ternary_hw` / `glwe_secret_fill` | `poulpy-core/src/layouts/glwe_secret.rs:186–203,237–242,281` onward | Concrete `&mut Source`, one distribution call per secret column. Both dense and sparse owners enter here. |
| `ScalarZnxFillDistribution`, `VecZnxAddNoise`, `VecZnxBigAddNoise` delegates | `poulpy-core/src/delegates/sampling.rs:16–23,31–40,48–57` | Each consumes a private **32-byte** `source.new_seed()`, including calls whose eventual implementation is portable. |
| `impl_sampling_host!`: scalar, small-noise, big-noise methods | `poulpy-cpu-portable/src/sampling.rs:25–52,55–74,76–95` | **Three separate internal `Source::new(seed)` sites**, at 34/65/86. Changing only the outer source or only scalar sampling leaves insecure-contract descendants. |
| `GLWEMaskFillReference::fill_glwe_mask_from_source_reference` | `poulpy-core/src/reference/encryption/glwe.rs:43–63` | One HAL uniform call per mask column; full-GLWE random filling also covers the body. |
| `VecZnxFillUniformSource` delegate | `poulpy-hal/src/delegates/vec_znx.rs:470–480` | Another **32-byte child seed per column**. This mask branch is separate from `SamplingImpl`. |
| `vec_znx_fill_uniform_source_all_derived` | `poulpy-hal/src/oep/derived.rs:421–438` | All-column convenience derives the same per-column children in increasing order. It must not retain a legacy expansion fallback. |
| `HalVecZnxDefault::vec_znx_fill_uniform_default` | `poulpy-cpu-portable/src/hal_defaults/vec_znx.rs:410–421` | **Fourth portable `Source::new(seed)` site**, for actual uniform mask coefficients. |
| `GLWEEncryptSkReference::{glwe_encrypt_sk_reference,glwe_encrypt_zero_sk_reference}` / `glwe_encrypt_sk_internal` | `poulpy-core/src/reference/encryption/glwe.rs:141–182,188–226,551–631` | Fill mask, perform unchanged ring operations, add `Noise::ENCRYPTION` through the noise delegate at 622. Pre-sampled-mask convenience does not remove error expansion. |
| `GGLWEEncryptSkReference::gglwe_encrypt_sk_reference` | `poulpy-core/src/reference/encryption/gglwe.rs:116–160` | Input-column then gadget-row loop: mask at 147 and error-producing internal encryption at 148–155 for each row. |
| Automorphism and tensor construction | `poulpy-core/src/reference/encryption/glwe_automorphism_key.rs:89–109`; `.../gglwe_to_ggsw_key.rs:84–116` | Automorphism prepares an inverse-Galois-transformed original secret; tensor prepares original-secret products. Both forward the concrete mask/error sources into GGLWE. These derived secrets/temporaries also require erasure. |
| CKKS input encryption | `poulpy-ckks/src/delegates/encryption.rs:24–39` | `CKKSEncryptOps::ckks_encrypt_sk` forwards both concrete sources; a secure client wrapper alone cannot override the descendants. |
| Eager original key generator | `poulpy-ckks/src/layouts/bootstrapping_keys.rs:238–319` | Three outer concrete sources. Rotations 279–283, tensor 286–289, **one** sparse sample 298–300, both switches 305–308. Use the same original graph, not this eager ownership schedule. |

Thus the active uncompressed graph is:

```text
private client entropy -> private purpose/operation parent streams
  secret parent -> 32-byte child -> scalar secret expansion
  mask parent   -> 32-byte child per mask column -> HAL uniform expansion
  error parent  -> 32-byte child per error call -> small OR big noise expansion
original key/ciphertext export -> checked reload/preparation -> deterministic refresh
```

`Source` explicitly says **“Not suitable for cryptographic key generation”**
(`poulpy-hal/src/source.rs:9–17`). Its helper functions are part of the numeric
contract too: masked rejection at 56–61, low-53-bit `next_f64` at 69–71, and
`next_u128 = (first_u64 << 64) | second_u64` at 94–95. The branch documentation's
“statistically independent” wording is not a literal information-theoretic
guarantee for streams determined by a finite seed. Proposed expansion relies
on computational pseudorandomness, not entropy multiplication.

### 2.2 Seed-compressed / adjacent paths: audited, excluded rather than forgotten

These are not needed for the frozen full-raw-key schedule. They cannot be
silently enabled later or bypassed when claiming a generally secure backend.

| Expansion / codec dependency | Frozen source |
| --- | --- |
| Seeded GLWE and LWE mask defaults create `Source::new(seed_xa)` before the same source-driven/HAL path. | `poulpy-core/src/oep/derived/encryption.rs:29–44` (`fill_glwe_mask_from_seed_derived`, `fill_lwe_mask_from_seed_derived`). LWE source mask forwarding: `poulpy-core/src/reference/encryption/lwe.rs:28–34`. |
| Compressed public-key generator constructs a parent `Source::new(seed)`, collects 32-byte entry children, encrypts with each child and stores them. | `poulpy-core/src/oep/derived/encryption.rs:107–144`, especially 134–142. |
| Compressed GGLWE constructs `Source::new(seed)`, calls `branch` per input-column/gadget-row and stores row-major seed entries before seeded-mask encryption. | `poulpy-core/src/reference/encryption/compressed/gglwe.rs:125–179`, especially 130/155/162. |
| Compressed GGSW constructs `Source::new(seed_xa)`, branches per row and ciphertext column, then expands each seeded mask. | `poulpy-core/src/reference/encryption/compressed/ggsw.rs:93–140`, especially 102/120/128. |
| Compressed GGLWE-to-GGSW/tensor key constructs `Source::new(seed_xa)`, branches per tensor input and calls compressed GGLWE, adding another expansion level. | `poulpy-core/src/reference/encryption/compressed/gglwe_to_ggsw_key.rs:90–113`, especially 97/109. |
| GLWE wire format contains a 32-byte seed, **no expansion algorithm/version**; decompression invokes seeded-mask filling. | `poulpy-core/src/layouts/compressed/glwe.rs:287–317,330–349`. |
| LWE wire format similarly stores a 32-byte seed and reconstructs its mask. | `poulpy-core/src/layouts/compressed/lwe.rs:112–134,143–157`. |
| GGLWE stores a vector of 32-byte seeds; seed count and matrix shape are checked but expansion identity is absent. GGSW has the analogous seed vector. | `poulpy-core/src/layouts/compressed/gglwe.rs:334–377`; `.../ggsw.rs:310,339–365,381–395`. |

`branch` also constructs a child object even where compressed callers discard
the returned object and retain only its seed. That construction must be covered
by any revised upstream contract. Compressed automorphism/switch wrappers inherit
these GGLWE paths; there is no independent cryptographic expander hidden in them.
Stored **mask** seeds may be public by that codec's contract, unlike secret/error
seeds; neither fact authorizes exposing a parent capable of deriving other domains.

Changing `Source` globally changes decoded masks for unchanged seed bytes. This
is a **semantic codec break**, even when byte length and Rust signatures match.
Existing compressed artifacts cannot be reinterpreted. A future compressed profile
needs an explicit expander/version identity and separate admission/compatibility
review. The smallest present design rejects all compressed imports/constructors.

### 2.3 Refresh boundary and excluded fixtures

The frozen CKKS refresh reference has no `Source`, seed or entropy argument:
`poulpy-ckks/src/reference/bootstrapping.rs::ckks_bootstrap_reference`
(429 onward), encapsulated ModUp (787 onward), C2S (854 onward), serial EvalMod
halves (897 onward), and S2C (939 onward). It uses ciphertext and prepared public
keys. Reload **full uncompressed** bytes and prepare them; do not regenerate
keys or introduce evaluator sampling on either refresh. A future compressed
loader would reintroduce the seed graph above and is excluded.

`poulpy-ckks/src/test_suite/presets.rs:112–165` illustrates dense-key continuity
but uses fixed/public seeds at 113/119/156. It is an oracle fixture, not a secure
client. Test-suite, benchmark/example, CPU-oracle and controlled-sampling adapters
are not production entropy providers. Binary FHE, MHE, alternate ring/SHIP/PACO,
AVX/ARM/Rayon backends are not qualified by this audit. “All sites” here means
all seed expansion in the frozen Standard-ring portable CKKS client-to-refresh
graph, including its optional compressed descendants, not every library protocol.

## 3. Distribution and exact-arithmetic contract

Changing the expander necessarily changes individual coefficients, ciphertexts,
digests and deterministic vectors. Preserve the **mapping/law under ideal uniform
bits**, then state the CSPRNG computational assumption separately; a finite-seeded
generator is not literally an IID entropy source. Do not claim ChaCha8 bit parity.

### 3.1 Secrets

`poulpy-cpu-portable/src/scalar_znx_fill.rs:68–76` zeros the column, assigns `hw`
independent signs using `2*(next_u32() & 1)-1`, then shuffles the whole column.
The **declared** contract is a uniformly chosen support of exactly `hw` positions
and independent fair signs on that support: each signed support would have
probability `1 / (binomial(N, hw) * 2^hw)` in the ideal-bit model. Dense `hw=1024`,
sparse `hw=32`, `N=65536`, rank one. No Bernoulli approximation, balanced-sign
constraint, weight drift or substitution of a sampled tag for original sampling.

The frozen shuffle uses `rand 0.10.3::seq::SliceRandom`; its implementation is
Durstenfeld/Fisher–Yates using `IncreasingUniform` below `u32::MAX`
(`rand-0.10.3/src/seq/slice.rs:470–507`). **Dependency-level conflict:**
`rand-0.10.3/src/seq/increasing_uniform.rs:40–75` samples a product-range chunk
using `random_range(..bound)`, then extracts mixed-radix indices. That call reaches
`UniformInt<u32>::sample_single_inclusive` in
`rand-0.10.3/src/distr/uniform_int.rs:168–224,285`. Without `unbiased`, this is
explicitly the **biased** Canon variant. The dependency documents a small bias
(`uniform_int.rs:59–64`); `rand/Cargo.toml:58–84` does not enable `unbiased` by
default, and no inspected Poulpy/isolated-consumer manifest requests it. A lock
alone does not record final feature unification; freeze that graph before
claiming an actual compiled choice. This inspection claims neither an exploit
nor a quantified secret-support bias for this particular N/weight.

**Recommended minimal-change candidate P:** preserve the frozen realized integer
map, including this dependency behavior, while replacing only its random-bit
provider. Independently specify/implement the map, not the upstream runtime:
ascending swaps use the chooser's largest fitting consecutive-u32-product chunks
(`increasing_uniform.rs:79–108`); for bound `b`, split `u32*b` into high/low
halves, retain the high half, and if low `> (-b mod 2^32)`, add the carry of low
plus the high half of a second independent `u32*b`. Extract chunk indices by
remainder/division in the original order. No rejection loop occurs in that
default branch. Preserve exact integer overflow/carry comparisons. This keeps
the realized distribution under ideal random words, **not a proof of its declared
perfect-uniform law**. It is not permission to accept the contract discrepancy.

**Alternate U:** independent full in-place Fisher–Yates with masked exact rejection
would supply the declared uniform law, without an index heap. But it changes the
realized frozen map/law if the default biased branch is governing. Enabling
`rand/unbiased` likewise changes the map. Neither is silently authorized as an
RNG-only fix. Obtain an explicit source-contract disposition/correction decision
before choosing U or claiming P satisfies the ideal law. This unresolved evidence
gate blocks implementation freeze/generation under either interpretation.

Other exposed laws are audited but **not requested**: probability ternary uses
weights `[p/2,1-p,p/2]`, probability binary `[1-p,p]`, through
`rand_distr::weighted::WeightedIndex<f64>`; binary fixed shuffles exactly `hw`
ones; binary block chooses one of `block_size+1` equally likely positions, the
extra position meaning all zero (`scalar_znx_fill.rs:59–108`). Their floating
weight implementation/rounding must not be claimed exact Bernoulli arithmetic
without separate qualification. `ZERO` is testing-only; `NONE`/`ENCAPSULATED` are
not sampleable (`poulpy-cpu-portable/src/sampling.rs:43–50`). Reject unsupported
laws before drawing or writing; do not silently implement a different sampler.

### 3.2 Masks

`poulpy-cpu-portable/src/kernels/znx/sampling.rs:3–8` samples each radix digit
uniformly in `[-2^(base2k-1), 2^(base2k-1)-1]`. At the frozen radix 52 this is
`(u64 & (2^52-1)) - 2^51`, **not** `% q` with a rounded floating modulus.
`.../kernels/vec_znx/sampling.rs:18–34` uses `ceil(k/base2k)` limbs, clears the
low `base2k-(k mod base2k)` bits of the last limb when needed, and zeros allocated
limbs beyond precision. This yields the uniform `k`-bit torus grid in the source
representation. Preserve coefficient/limb order, padding, signed digits and ring
normalization; no change to NTT4x30 integer operations or gadget decomposition.

### 3.3 Errors, including the easily missed numeric qualification

`poulpy-core/src/noise_distribution.rs:3–26,50–72` specifies a **discrete Gaussian
conditioned on** `|z| <= floor(6*sigma)`, with probability proportional to
`exp(-z^2/(2*sigma^2))` on that support. `Noise::ENCRYPTION` has binary64
`sigma=3.2`, exactly `3602879701896397 / 2^50`, hence support `[-19,19]`.
It is neither an unbounded Gaussian nor rounded Box–Muller/normal samples.

Frozen small-noise realization:

- `poulpy-cpu-portable/src/kernels/noise.rs:14–37` fixes 19 unsigned 128-bit
  cumulative thresholds for the magnitude. Keep these exact integer values.
- `sample_table`, `table_entropy`, `scan_table_batch`, `add_table_vectorizable`
  at 331–394 use full threshold scans, fair sign bits and batches of 64.
  Per full batch: 512 bytes high words, 512 bytes low words, then eight sign bytes;
  each word is little-endian. Scalar magnitude construction treats the first
  `next_u64` as the high half of `next_u128`, matching `Source`.
- This is a dyadic CDT approximation to the conditioned Gaussian, **not exact
  irrational PMF sampling**. The unsafe `SamplingImpl` contract permits distance
  at most `B*2^-128` for `B<=64`
  (`poulpy-core/src/oep/sampling.rs:5–29`). For this profile retain the same CDT,
  hence the same permitted `19*2^-128` bound per sample. Statistical histograms
  alone cannot establish that bound.
- Add integer `z * 2^-k` in the torus: `size=ceil(k/base2k)`,
  `shift=(base2k-k%base2k)%base2k`; preserve balanced digits, modular wrapping,
  final normalization and zero padding. Source placement/validation is at
  `kernels/noise.rs:438–613`. At radix 52, `k=1680/120/40` gives shifts
  `36/36/12`, respectively; `19 << shift` fits the one-last-limb path.

The big-noise portable entry uses the same sampling law and placement, writing
`i128` rather than `i64` coefficients. It is not a separate weaker RNG license.
For wider exposed laws, `Noise::Uniform{bits}` is the exact signed interval
`[-2^(bits-1),2^(bits-1)-1]`; larger Gaussians use exact rational rejection
(`kernels/noise.rs:147–329,517–610`). These paths include big-integer temporaries,
variable rejection work and, for dynamic small Gaussians, precision-doubling CDT
construction/cache (`95–145`). They are **excluded** from the smallest profile.
They require separate arithmetic, side-channel and capacity proofs before exposure.
The rare Bernoulli-index arithmetic assumption noted at 284–287 is not a proved
worst-case overflow bound and must not be inherited as one.

Backend sampler invariants remain: fixed seed/degree/law/radix/precision must not
depend on selected column, number of columns, allocated extra limbs or prior
destination values; write only the selected column and declared precision;
invalid placement rejects before writing (`SamplingImpl:15–29`). Domain allocation
chooses a seed before these backend-local invariants apply.

## 4. Algorithm and dependency comparison

| Candidate | Benefit | Cost / disposition |
| --- | --- | --- |
| OS-seed frozen `Source` / outer secure wrapper, inner ChaCha8 | No stream-source patch | **Rejected by D28.** Does not replace any internal expander or its declared contract. This is not a claimed attack on ChaCha8. |
| ChaCha12 | Conservative improvement over eight rounds; fast | Still a reduced-round selection without a measured need. Prefer full 20-round margin; no inference of insecurity from rejection. |
| `rand_chacha::ChaCha20Rng` 0.10.0 | Existing RNG-family API and 32-byte child seeds; small conceptual change | Inspected `rand_chacha-0.10.0/src/chacha.rs:98–132,193–200` exposes seed retrieval and a cloneable state. Its manifest `Cargo.toml:51–71` has no zeroize feature. A private wrapper can hide cloning but **cannot prove erasure of inaccessible core/buffer state**. Not sufficient as-is. |
| RustCrypto `chacha20::ChaCha20Rng` with `rng,zeroize` | Full 20 rounds, 256-bit seed; inspected core and buffered-output drop erasure | **Recommended candidate**, subject to approved exact lock/features and erasure review below. This RNG uses the original 64-bit counter/64-bit stream layout, not RFC8439's 32/96 layout. |
| AES-256 counter expansion using existing `aes` + `zeroize` | Avoids new cipher family; existing private GKM stream gives lifecycle pattern | Viable alternate design, not drop-in reuse of AES-128 duplicate-counter GKM profile. Needs a new domain/counter mapping and PRP-to-PRG concrete bound. At `q` blocks per key the generic switching term is order `q^2/2^128`, regardless of 256-bit key; multi-MiB streams must not be advertised as 256-bit statistical security. No automatic algorithm selection merely because dependency exists. |
| SHAKE256/XOF using existing `sha3` | Natural arbitrary-length domain-separated expansion | Also viable; exact absorb framing and erasure of finalized/squeezing state need inspection. No demonstrated smaller reviewed state-erasure path here. No improvised hash concatenation or XOF substitution. |
| Direct OS bytes for all coefficients | No user-space PRG seed hierarchy | Billions of bytes/OS-call work instead of small entropy requests; requires broader fallible sampling plumbing and resource proof. Not the smallest proposed seam. |

### Candidate identities, not a dependency update

The inspected upstream workspace requirements are `rand 0.10.3`,
`rand_chacha 0.10.0`, `rand_core 0.10.1`, `rand_distr 0.6.0`
(`Poulpy Cargo.toml:36–39`). Upstream `Cargo.lock:817–851` supplies their
registry identities/checksums; these are also present in the existing isolated
consumer `benchmarks/native/poulpy_refresh/Cargo.lock:310–344`.
`rand` appearing in a lock does not mean its default RNG replaces `Source`.

Important closure difference: upstream lock has `chacha20 0.10.0`
(`Cargo.lock:104–112`), whereas the inspected standalone consumer has
`chacha20 0.10.2`, checksum
`65c35e4b699c7e15ccbe7ee35c005e4fc0a278d22238a2857e6ce2dadeda1b06`
(`benchmarks/native/poulpy_refresh/Cargo.lock:46–54`). The latter currently lists
no zeroize dependency for this package. Neither presence nor a semver-compatible
resolution enables the required feature or qualifies its behavior.

Design candidate: exact `chacha20 0.10.2`, default features disabled,
`rng` and `zeroize` enabled, plus reviewed `rand_core 0.10.1` trait plumbing;
reuse existing PLLM `getrandom =0.3.3` and `zeroize` contract rather than silently
upgrading entropy dependencies. This is **not** a Cargo instruction or approved
pin. A later proposal must freeze actual consumer lock, features, checksums and
full dependency closure, including the resolved zeroize package. No revised
Poulpy commit exists in this packet.

Read-only cached-source evidence:

- `chacha20-0.10.2/Cargo.toml:51–89`: RNG feature, optional zeroize dependency;
  `src/rng.rs:21–40,94–99,108–171`: 32-byte seed, 64/64 layout, four-block
  buffered generator, buffer-erasing `Generator::drop`, `ZeroizeOnDrop` marker.
- `chacha20-0.10.2/src/lib.rs:306–314`: core `Drop` wipes its state;
  `rand_core-0.10.1/src/block.rs:92–110,132–135`: buffer drop hook, and the
  otherwise important partial-word discard behavior.
- `getrandom-0.3.3/src/backends/getentropy.rs:18–25`: macOS entropy backend
  fills at most 256 bytes per `getentropy` call and reports failure. The separate
  cached 0.4.3 backend is not permission to upgrade PLLM's entropy dependency.

These paths are below the local crates.io source directory
`/Users/blairhudson/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/`.
Inspection does not establish compiler spill erasure, SIMD-register clearing,
successful feature unification, constant-time generated machine code or OS entropy
quality. Those claims require separate evidence and explicit threat scope.
In particular, `chacha20-0.10.2/src/rng.rs:138–147` accepts the seed by value and
does not explicitly wipe that parameter. Its later core/buffer drop hooks do not
resolve this constructor-copy question. The cached package is **not acceptable
unchanged merely on its ZeroizeOnDrop marker**: either prove the approved target's
constructor/temporary erasure path, use a reviewed reference-taking construction
path with complete cleanup, or propose a separately approved primitive revision.
Do not vendor a local patched dependency or weaken the erasure contract silently.

## 5. Smallest complete seam: interfaces and stream schedule

The following are conceptual contracts, **not implemented APIs**:

- `Entropy::try_fill(purpose, &mut private_bytes) -> Result`: private provider;
  production constructs only the OS provider. No public seed/provider injection.
  Tests may use an isolated nonproduction provider, with no feature-unification
  route into production. A partial/error fill wipes the whole destination and
  burns the epoch; no clock/PID/counter, repeated seed or deterministic fallback.
- `PrivateSeed`: unique guarded 32-byte owner, not `Copy`/`Clone`/`Debug`/serde;
  consumed by construction, wiped after transferring into the expander. Audit
  constructor-by-value copies and spills; `Zeroizing` around the caller's array
  alone does not erase a callee's copied seed.
- `ByteStream::try_fill`, `try_u32_le`, `try_u64_le`, private child issuance:
  checked finite byte/child counters, no clone, reset, seek, reseed-in-place,
  seed getter, export or checkpoint. Underlying library controls are inaccessible.
- `sample_fixed_ternary`, `fill_uniform_torus`, `add_encryption_error`: independently
  implemented integer samplers over that fallible byte contract, rejecting all
  unsupported laws. Secret sampling's governing P/U disposition must be explicit
  in the execution identity; P's name cannot imply perfect uniformity. Ring/encryption
  code borrows these streams; no hidden default source may be constructed by a
  backend, mask helper or codec.

### Domain separation without a new KDF

Use a fresh independent OS 32-byte seed for each `(epoch, original object,
purpose)` parent; do not retain one master seed for the run. There are two secret
objects (dense, sparse), and 37 encryption objects (36 evaluation keys, one input
ciphertext), each encryption having distinct mask and error parents. Thus **76
parent entropy fills = 2,432 private bytes** over setup; an additional private
32-byte random epoch identifier, if used, is charged separately and is never a
sampling seed. The OS security assumption is explicit; IDs/timestamps are not
entropy. Roots are associated with immutable descriptors, never selected by an
untrusted caller after construction.

Within this new profile, assign reviewed fixed `u64` stream IDs: parent-seed
expansion `1`, secret coefficients `2`, uniform masks `3`, encryption errors `4`.
Set once at construction, position zero; hide subsequent stream/cursor mutation.
Independent root keys separate original objects/purposes; typed stream IDs
separate parent versus leaf use. No KDF, public-label hash or secret fingerprint
is required for this bounded fresh-root design. Descriptor binding is enforced
by the private owner, not a claim that public metadata supplies entropy.

Parents output only 32-byte child seeds, with checked monotonically consuming
ordinals; leaves output only sampler bytes and cannot branch. Preserve one secret
child per column, one mask child per mask column, one error child per error call.
The approved graph contains **496 children**: two secrets plus 247 masks plus 247
errors. Parent/child seeds stay private even though masks become ciphertext
material. No parent state persists into evaluation or key reload.

For a reproducible new byte contract, refill a guarded **64-byte** buffer only
with complete 64-byte reads from the chosen ChaCha20 RNG; serve all caller reads
and LE word draws from that buffer without discarding partial bytes. Include the
underlying four-block buffer in both erasure and resource accounting. This avoids
accidentally inheriting `rand_core`'s discarded partial-word semantics across
different caller partitions. `next_u128` for the retained CDT is explicitly two
LE64 draws with the first in the high half. Freeze scalar/batched consumption
order; do not silently change distributions, table entries or numeric placement.

### Failure and work limits

Candidate limits for approval, not established execution bounds:

- Checked stream output ceiling `2^32` bytes per root/leaf; total setup delivered
  PRG bytes ceiling `2^33`; count library-generated/refill bytes separately,
  rounding each stream to its four-block generation boundary. Reject before
  overflow, counter wrap, output mutation or a disallowed child issuance.
- Candidate P's frozen chooser uses at most two `u32` draws per new chunk and
  at most `N-1` new chunks per shuffle: a finite conservative bound without a
  rejection cap. Preserve that actual map pending the section 3.1 disposition;
  do not describe it as perfect-uniform merely because work is bounded.
- If alternate U is separately approved, a possible exact-rejection contract is
  at most **256 masked u64 draws per index**. Exhaust means failure, never modulo
  reduction or substitution of a biased index. Tight-mask exhaustion is below
  `2^-256` per index and below `2^-239` over both secrets in the ideal model.
  Accepted indices are independent of rejection counts, so this cap does not
  bias successful outputs in that model. This is a separate candidate, not the
  frozen source map. No arbitrary Gaussian retry cap follows by analogy.
- No automatic mid-operation reseed. Obtain fresh roots for each new authorized
  original object; any entropy, allocation, write, counter, deadline, validation,
  cancellation or erasure/retirement failure burns the whole generation epoch.
  Restart requires new admitted owners and fresh entropy, never replay of a
  partially consumed stream or a successful partial bundle.
- A wall-clock timeout reports failure, never a numerical result. No claim that
  conditioning on every real-world timing/OS failure preserves a sampler law;
  state failure probability and successful-output qualification separately.

## 6. Revised dependency / integration options

| Seam option | Completeness and authority |
| --- | --- |
| Independent shared samplers plus independent exact encryption/ring implementation | **Recommended paper-path architecture.** Upstream supplies laws, parameters and independent comparison inputs only. Implementing this later still requires a complete HE arithmetic/backend plan; this document does not claim that such an implementation exists. A sampler is not a complete HE backend. |
| Reviewed revised upstream `Source` contract | Smallest conceptual patch for **isolated D4 only** if globally routed through all constructors in section 2. Requires a newly approved immutable upstream identity, explicit codec break/versioning, zeroization and fail-limit plumbing. A silent alias to a different RNG under the frozen commit is prohibited. It does not authorize upstream linkage in the paper path. |
| Upstream cryptographic source/expansion-policy injection | Can preserve legacy test streams while adding a distinct secure profile, but the policy must cross secret/noise APIs, HAL masks, all seed defaults, portable kernels and compressed boundaries. No legacy default may satisfy a secure type. More review surface than the restricted independent seam; revised dependency and D4 approval still required. |
| `SamplingImpl` newtype/backend alone | **Incomplete.** It changes scalar/noise leaves only; outer concrete `Source`, HAL mask seeds, seeded defaults and codecs remain. Foreign-trait coherence forbids replacing an existing foreign backend implementation. A new backend introduces substantial forwarding/unsafe-contract/resource qualification, not a free wrapper. |
| Standalone serial client using frozen public keygen APIs | Solves ownership/eager-bundle issues but **not randomness**. APIs still reach the frozen concrete sources. No observer callback is required for serial generation, but no secure generation permission follows. |

An infallible upstream `Source` signature also cannot transparently express the
proposed finite-stream failure contract. A revised upstream proposal must either
carry errors throughout or prove fixed work and enforce preflight at every
admitted operation; unwinding through unsafe/backend boundaries is not a design
substitute. Reject reflection, transmute, source-tree copying, local monkeypatches
or a parallel foreign runtime as a way around the revision/approval gate.

## 7. Origin continuity, erasure and fail-burn

One private original client owns the raw/prepared dense secret, sources and
generation epoch. Sample dense once, prepare that owner, generate every
automorphism/tensor and both switches through its real secret lineage, and
encrypt input under the same prepared dense owner. Sample sparse once, retain
the `ENCAPSULATED("sparse-encapsulation")` restriction, and use it for both
dense→sparse and sparse→dense before erasure. Never serialize a secret or
resample sparse per switch. Inverse-Galois/tensor preparations are derived
secret owners, not independently sampled replacement keys.

For each of the 36 evaluation keys: admit full original shape → generate one
raw key → stream its checked export → derive immutable expected content and
opaque **local** origin binding inside that same original generation/export
operation → checked-retire raw key before allocating the next. Publish the
usable epoch only after the whole inventory and ciphertext pass their gates.
On partial-export failure, quarantine/invalidate every sibling export and
ciphertext; a readable file or matching digest cannot survive as reusable
authority. No eager bundle, hidden additional original retained for comparison,
or regenerated reload.

Bindings include actual execution/sampler/codec identity, epoch, key role and
direction, Galois element, degree/rank, full original capacities, exact parameters
and serialized byte identity. They are minted by private generation, not accepted
as supplied `dense_valid` flags, hashes, tags or imported expectations. No secret
hashing. A digest proves equality to trusted original bytes, **not** publisher
authenticity, remote origin or role authorization. Existing quarantined imports
cannot be promoted by attaching the new algorithm name.

Wipe live seed buffers, root/leaf keys and buffered PRG output, random shuffle
words/signs, sampled secret coefficients, prepared/derived secret data, all
secret-bearing scratch and failed partial plaintext/ciphertext/export buffers.
CDT high/low/sign/magnitude temporaries contain error draws and are sensitive;
ordinary return/drop is not an erasure proof. The public fixed CDT need not be
secret, but its allocations still count. Wipe retained capacity, not only logical
length. End all borrows, erase, then checked-retire backing; keep reservations
until actual deallocation/release succeeds. Failed release retains charge and
poisons the run. No use-after-wipe or refunded-live owner.

Evaluator receives public context, original encrypted evaluation material and
ciphertext only; no roots, child seeds, sparse/dense secrets or client refresh.
Client dense secret survives only for authorized validation/final decryption and
is then erased. Setup parents/leaf state retire before evaluator refresh.
This private-module design does not claim process isolation or malicious-OS
resistance. Zeroization is best-effort for ordinary controlled lifetimes; abort,
SIGKILL, crash dumps, swap, registers and compiler-created copies need explicit
threat/operational treatment. A kill guard prevents continuation, not guaranteed
physical erasure. Preserve the zero-additional-swap policy; do not overclaim it
as proof that no private byte ever reached persistent storage.

## 8. Preconstruction numeric and resource admission

### Frozen numbers, not adjustable RNG tuning knobs

`poulpy-ckks/src/presets/bootstrapping.rs::n16_d35_k600_p19_c2s`
(265 onward; derived key-shape checks at 620–670) and the existing
`benchmarks/native/poulpy_refresh/contract.json:23–101` define:

- `N=65536`, rank one, 32,768 complex slots, no sparsity; radix 52;
  entry `k=40`, scale `2^35`; bootstrap `k=1427`; output `k=600`, scale `2^35`.
- Dense/sparse weights 1024/32; existing modulus caps 1714/120 are source facts,
  **not independent lattice-security qualification**.
- High keys: `dnum=7,dsize=4,k_aux=224,total_k=1680`, 33 physical limbs.
  Low dense→sparse: `dnum=1,dsize=1,k_aux=68,total_k=120`, three limbs.
- 33 automorphism keys (32 rotations plus conjugation), one tensor, two switches.
  Exact Galois inventory, DFT schedules, degree-30 CosHKEven/interval-16 EvalMod,
  coefficient precision and operation order remain frozen. No altered key graph.
- Full-slot checks, including transformed padding, retain maximum absolute errors
  `2^-20` fresh/pre-refresh-1, `2^-14` refresh-1, `2^-12` pre-refresh-2/final;
  finite values required. Second-refresh entry reduction `600→40`, scale 35,
  needs a real numerical proof; changed RNG does not close it.

### Finite sampling work inventory

For the restricted uncompressed rank-one graph, 35 high keys (33 automorphisms,
tensor, sparse→dense) each contain seven encrypted GLWE rows; low dense→sparse
has one row, and input encryption one row: **247 mask/error pairs**. This follows
from the original GGLWE loop and tensor construction in section 2, not a request
to generate them.

| Delivered sampler bytes | Source-derived count |
| --- | ---: |
| High-key masks: `35*7*N*33*8` | 4,238,868,480 |
| Low-key and input masks: `N*(3+1)*8` | 2,097,152 |
| All masks | 4,240,965,632 |
| Errors: `247*(N/64)*(512+512+8)` | 261,021,696 |
| Secret signs: `(1024+32)*4` | 4,224 |
| Candidate P shuffle upper bound: `2*(N-1)*2*4` | 1,048,560 |
| Parent-issued child seed bytes: `496*32` | 15,872 |
| Candidate P combined worst delivered PRG bytes | 4,503,055,984 |

Alternate U's proposed 256-draw cap instead gives 268,431,360 shuffle bytes and
4,770,438,784 total delivered bytes. Neither count establishes uniformity of P
or authorizes U; the distribution disposition must precede the implementation
freeze. Both totals are source/design arithmetic, not executed measurements.

Add finite per-stream refill rounding and entropy calls separately; none of this
is retained as a giant randomness array. The `2^33` aggregate candidate cap has
room for four-block refill rounding but is not a timing proof. Maximum parent
output is seven 32-byte seeds for a high-key mask or error parent. At most two
current encryption parents and one child need coexist; dense/sparse secret
backing remains separately charged. Parents for later objects are not prebuilt.

The 247 error calls draw `247*N = 16,187,392` coefficients. A union bound for the
retained small-CDT approximation is `307,560,448 / 2^128` across this generation
graph, before CSPRNG advantage or other protocol errors. This is not a proof of
RLWE security, six-sigma adequacy, decoding success, circuit privacy or complete
refresh failure probability. It must not be confused with rejection-exhaustion
probability or measured precision.

### Admission must precede entropy and every constructor

Use an immutable public plan binding the stream schedule, full key layouts,
numeric profile, source/sampler/codec identities, operation inventory, capacities,
scratch, exports, limits and cleanup. Reserve the whole current live-owner overlap
**before** obtaining roots or allocating secret/key/context buffers. Caller-side
`size_of` guesses, later RSS observations and post-construction checks do not
admit an allocation. No release/reacquire gap for a continuing original owner.

Preserve D4: at most two refreshes, one CPU execution thread, owner cap
**8,589,934,592 bytes**, existing **67,108,864-byte guard**, separate
**4,294,967,296-byte host reserve**, zero extra swap, 600-second setup and
600-second execution deadlines. Build/toolchain budgets remain separate and
unchanged; this review authorizes neither. The 1-GiB setup target is a **proof
target**, not an additional allowance or an established complete upper bound.

Known keygen subset from the existing lifetime/source audit:

| Owners | Bytes |
| --- | ---: |
| One full high raw key | 242,221,056 payload |
| Both raw/prepared dense and sparse secret pairs | 5,242,880 payload |
| Reserved keygen scratch | 163,577,856 payload |
| Four tensor-preparation private heap leaves | 8,388,608 payload |
| Subset total | **419,430,400 payload / 419,921,920 charged** |
| Keeping an additional original raw export | **+242,270,208 charged** |

These are partial source calculations, not complete upper bounds or measurements.
See [lifetime admission](../../paper/methods/poulpy-lifetime-admission.md), notably
sections 5–6 and the original-generation audit in
`benchmarks/native/poulpy_refresh/src/staged_identity.rs:68–79`.

The secure seam must add and prove: concrete expander/core/output-buffer sizes,
seed-constructor copy overlap, guarded 64-byte refill, public descriptors and
child ledger, failure/control stack, sign/shuffle temporaries, CDT batch arrays,
all library/SIMD stack frames, ordinary allocation rounding, and erasure work.
Include original raw/prepared/derived secrets and full scratch capacities even
when actual sampling touches one limb. Price context/solver/codec/plaintext/
ciphertext owners, bounded streaming exports and hash state, retained/quarantined
files, kernel I/O/cache, runtime and allocator metadata together. Additional
entropy/codec dependencies cannot consume an unproved remainder by declaration.

No complete setup or whole-process bound is supplied here. Existing conditional
DFT margins/operational reserves are not proved allocations for this new seam.
Library state that is stack-resident still consumes admitted owner/runtime bytes;
dropping a handle alone proves neither erasure nor physical reclamation.

## 9. Later acceptance evidence — not run or authorized here

Use ordinary correctness tests and an ordinary reusable experiment/benchmark
when separately approved. No copied trees, receipt collections, custom replay
verifiers or archive machinery are a completion gate.

Required independent checks:

1. Primitive KATs for the **selected 64/64 ChaCha20 layout**, including stream IDs,
   zero position, word endianness and block boundaries. RFC8439 vectors require
   an explicitly correct layout mapping; do not assume RNG layouts coincide.
   Partitioned byte reads must match single reads; root/leaf/profile separation,
   byte-limit and child-count failures reject before writes. Check scalar/NEON
   agreement without treating upstream RNG output as the only oracle.
2. Scripted-byte integer oracles: exact signed supports/hamming weights, independent
   sign assignment, chunk/product bounds and every Canon carry comparison for P;
   final feature graph and source-map parity, with declared versus realized laws
   separated. If U is separately approved, add unbiased exhaustive small shuffle
   cases, tight-mask rejection and all 256 failures. Check untouched columns and
   invalid-law/precision preflight. No production public-test-material entry.
3. Independent high-precision interval/rational reconstruction of all 19 CDT
   thresholds for exact dyadic sigma; boundary words immediately below/equal/above
   each threshold, both signs/zero, scalar/batched parity, and retained placement
   for `k=40,120,1680` in small/big word families. Histograms are secondary checks.
4. Exact integer torus/gadget/encryption parity under supplied test draws, including
   the automorphism inverse-secret and tensor-product lineage. Different PRGs do
   not require matching ciphertext bytes; identical injected draws must preserve
   arithmetic. Upstream may supply comparison inputs, not the paper runtime.
5. Fault injection at each entropy, child creation, allocation, export/read/write,
   validation, cancellation, erase and release boundary; epoch-wide burns,
   one-shot root/child consumption, no sibling promotion, no seed/secret logging,
   no compressed-codec fallback and no RNG use during evaluator refresh.
6. Origin-negative tests: imported bytes/tags/digests cannot mint original client
   authority; wrong dense epoch/sparse direction/Galois/shape/profile burns;
   checked serial retirement and full raw capacity survive through last borrow.
7. Exact-fit/one-short admission before construction, all seed/RNG/batch/stack
   and constructor-copy overlaps, retained failed-release charges, and proof
   limits separated from observed RSS. Audit actual locked feature graph and
   erasure drops/temporary copies on the approved target/compiler.
8. Only after distinct parameter-security, complete resource, backend and numeric
   entry gates close: the unchanged full-slot independent D4 precision checks,
   at most two real evaluator refreshes. This is not permission to run them now.

## 10. Unresolved evidence, separate from the proposed decision

| Open evidence / approval | Consequence |
| --- | --- |
| Algorithm/stream/entropy/failure contract and exact implementation not approved | No RNG code, original generation or dependency change. |
| Secret shuffle declares uniform support but default `rand` chooser reaches a documented tiny-bias integer path; final features and concrete induced bias are unqualified | Freeze governing feature graph and obtain source-contract disposition. P preserves the realized map only; U/unbiased correction needs explicit approval. Neither ideal-law compliance nor a security failure is established here. |
| Candidate cryptographic dependency closure/features, source identities, target code and constructor/temporary erasure not frozen/verified | Cached source/drop hooks are evidence inputs, not a qualified secure build. No claim of complete zeroization or side-channel resistance. |
| Independent paper-path ring/encryption integration absent; upstream revision for any separately requested D4 seam absent | No secure full HE backend exists by virtue of this design. No source/runtime vendoring/linkage shortcut. |
| Complete setup/live-owner, solver/codec/context/stack/kernel/runtime/I/O and cleanup upper bounds absent | Partial 419,921,920-byte subset and 1-GiB target cannot authorize construction. Existing conditional allocator budgets remain conditional. |
| Independent RLWE/parameter-security and whole-circuit error/failure analysis open | Fresh entropy/full-round expansion does not qualify modulus caps, sparse-secret technique or the composed key graph. |
| Independent CDT reconstruction, integer arithmetic/origin/negative/failure tests not executed here | Distribution-preserving design is a specification, not implementation correctness evidence. |
| Second-refresh `600→40` entry conversion and complete full-slot numerical admission open | No second refresh; no threshold changes to repair failure. |
| Threat scope for private-client side channels, crashes, swap/core dumps and residual copies not accepted | No malicious-host isolation or guaranteed physical-erasure claim. |
| Method-specific source mapping, protected lifecycle and SDK/role integration remain separate | D4/D28 imply neither paper adoption nor production availability. |

**Approval wording to consider:** approve the restricted design in sections 1,
3–8 as the basis for a later immutable implementation proposal, with ChaCha20 as
the proposed expander and no compressed material. Resolve section 3.1's declared
versus realized shuffle law explicitly; do not silently approve a correction or
call the realized biased path perfectly uniform. Require explicit approval of
that implementation, exact dependency identity/features, security scope and full
preconstruction resource contract before any generation. Keep every unresolved
gate above open. This wording does not itself authorize implementation, dependency
updates, building, execution, adoption or commits.
