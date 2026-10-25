# HE source, original export and backing seams: implementation decision packet

**D28 design/review only. No implementation approval is inferred.**
**D38 update:** only §6's isolated Layout patch preparation is explicitly
approved. Main application, dependency adoption, builds/tests and the sampler
remain excluded; see the [decision register](../../paper/methods/DECISIONS.md#d38-three-independently-bounded-follow-ups-to-d37).
This packet gives proposed interfaces, not existing or compiled APIs. It selects
no RNG, sampling-law correction, new dependency pin, backend or method profile.
D35's 31 tiny storage tests do not qualify sampling, generation, preparation or
encrypted execution. No code, dependency, native operation or model ran for this
packet; inspection used existing local source only.

Read with [the decision request](../plans/he-client-sampling.md), [the detailed
design/options](../plans/he-secure-sampling-design.md), [the source audit](he-secure-sampling-seam.md)
and [D28](../../paper/methods/DECISIONS.md#d28-secure-he-client-sampling-seam-design-and-review).
Those documents retain their historical findings and unapproved alternatives.
Here **UNKNOWN** is a blocking value/proof, never zero or an implementation default.

**Astra review revision:** the only selectable next action is §6's isolated
Layout patch preparation. Its later qualification envelope is specified, not
authorized to execute. The source/sampler tranche is **NOT SELECTABLE** (§2.1).
The two accessors address one D4 rotation-capacity seam; they do not qualify five
HE methods. This document preserves the original H proposal's factual boundaries.

## 1. Boundary and decisions

The exact inspected upstream is Poulpy v0.9.0,
`ca3532570ff62b96599a74f7adb85de7726b0446`. Paths prefixed H/C/P/K below mean
`poulpy-hal/src`, `poulpy-core/src`, `poulpy-cpu-portable/src`,
`poulpy-ckks/src` at that commit, respectively. The consumer's existing four
Poulpy Git resolutions, lock, sampler and numerical profile must not change to
implement this document. A future upstream delta requires a separately frozen
and approved revision; it is not byte-identical v0.9.0 qualification.

**Observed consumer identity at Main `a61f4bf6c8f52c31e6584de4aad5cb65dd23be28`:**
`benchmarks/native/poulpy_refresh/Cargo.toml:12–15` pins `poulpy-ckks`,
`poulpy-core`, `poulpy-hal`, and `poulpy-cpu-portable` individually to `=0.9.0`
and that revision; portable explicitly enables `enable-ckks`. Their four
`Cargo.lock` package records each resolve to
`git+https://github.com/poulpy-fhe/poulpy?rev=ca3532570ff62b96599a74f7adb85de7726b0446#ca3532570ff62b96599a74f7adb85de7726b0446`.
This is a read-only manifest/lock observation, not evidence of compiled feature
unification or permission to redirect any package. No revised upstream commit
exists in this proposal. A patch against this base is a proposed delta, never
the identity of an adopted dependency.

Keep two routes distinct:

* The separate D4 consumer can propose an upstream API revision plus a private
  serial client. That is the concrete call graph described here.
* Independently implemented paper paths may reuse the contracts, not link/copy
  the upstream runtime as their implementation. Source and arithmetic owners
  remain independently implemented and separately qualified.

| Blocked method | Reusable prerequisite, not adoption of the D4 profile |
| --- | --- |
| NEXUS | Private source and original-key/export ownership; its dual-key/source-specific graph still needs its own binding. |
| THOR | Same ownership/fallibility contracts; its Liberate.FHE parameters, laws and evaluated arithmetic are not Poulpy's. |
| CacheMir | Serial original keys and retained KV-domain identity; evaluated Lattigo configuration/calibration remains independent. |
| ROSETTA | Scheme/domain-specific original generation and conversion-key lineage; this CKKS scaffold supplies neither TFHE nor the missing conversion graph. |
| BumbleBee | Party-specific source/key/export custody; no substitution of this single-client CKKS experiment for its two-party protocol. |

**Decisions still required:** engine/entropy/child-stream contract and erasure;
realized versus declared fixed-weight shuffle law; exact revision/features;
complete finite admission. The earlier ChaCha20/P/U recommendations are options,
not approvals. The interfaces below expose where those decisions enter; they
cannot supply their missing security or resource proofs.

## 2. Exact source cut: additive fallible path, no legacy coercion

The existing `Source` stays unchanged for compatibility. It is concrete ChaCha8,
infallible and disclaimer-bearing (H/`source.rs:18–46,102–118`). It must not
implement or coerce into the proposed secure path. A `CryptoRng` marker or fresh
seed does not close that distinction.

Proposed public upstream mechanism, for the **seeded-child option only**:

```rust,ignore
// Proposed new types. Engine implementation/layout/version still UNKNOWN.
pub struct SecureSource { /* engine, immutable scope, counters, failed bit */ }
#[derive(Clone, Copy)]
pub struct SourceVersion { /* immutable compiled implementation identity */ }
#[derive(Clone, Copy)]
pub enum SourcePurpose { Secret, Mask, Error }
#[derive(Clone, Copy)]
pub struct SourceScope {
    pub epoch_label: [u8; 32], pub object_ordinal: u32,
    pub purpose: SourcePurpose,
}
pub struct SourceLimits { pub bytes: u64, pub children: u32 }
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceError { Entropy, Limit, Scope, Unsupported, Failed }

impl SecureSource {
    pub fn try_from_entropy<E: rand_core::TryRng>(
        entropy: &mut E, scope: SourceScope, limits: SourceLimits,
    ) -> Result<Self, SourceError>;
    pub fn try_fork(
        &mut self, limits: SourceLimits,
    ) -> Result<Self, SourceError>;
    pub fn try_fill_bytes(&mut self, out: &mut [u8]) -> Result<(), SourceError>;
    pub fn try_next_u32(&mut self) -> Result<u32, SourceError>;
    pub fn try_next_u64(&mut self) -> Result<u64, SourceError>;
    pub fn try_next_u128(&mut self) -> Result<u128, SourceError>;
    pub fn version(&self) -> SourceVersion;
    pub fn scope(&self) -> SourceScope;
}
```

These signatures are a proposed plumbing contract, not evidence an arbitrary
`E` is secure. The public upstream mechanism may be supplied test entropy; only
the private PLLM client installs the reviewed production OS adapter. Neither
this constructor nor `SourceVersion` mints a trusted export epoch. No public
seed getter, raw-child-seed result, clone, serde, seek, reset or reseed API.
Scope fields are public association labels, never entropy or authority; the
private client assigns and checks them against its original-object schedule.
`SecureSource` stores an immutable copy. The metadata return types may be
`Copy`; source/seed/state owners may not. No caller-provided version selects an
engine. Exact canonical scope/version encoding and concrete engine remain **UNKNOWN** pending
approval; no numeric version tag is allocated here.

`SourceLimits` enforces mechanical finite counters, not admission authority.
The private client must derive it from its checked, prepaid whole-operation
quote. Child limits must be checked against the aggregate ledger as well as
the parent seed budget. Leaf children cannot fork. The offending source becomes
terminal on failure; the private client must propagate every failure into its
shared epoch burn before another root/child operation. Merely constructing the
public upstream mechanism does not supply that client lifecycle. No counter
refund, replacement child or in-epoch retry.
Geometry/limit rejection precedes destination mutation. If a later draw fails,
the owner becomes unusable and sensitive partial output is erased; partial
entropy fills and seed temporaries must be wiped on every return/unwind path.

These are obligations, not an erasure proof. Before ownership is admitted,
invalid geometry returns without touching a caller's destination or drawing
entropy. Within an admitted epoch, any validation/draw failure terminally poisons
the source and epoch even if validation preceded writes. Once mutation begins,
the private client owns the destination's entire sensitive backing and all
temporaries; a borrowed destination therefore requires an enclosing cleanup
owner installed before the call. No partially written result may escape as valid.
Failed cleanup retains reservations and failed-state obligations; neither a
logical invalidation nor a dropped handle refunds them. Unwind guards must cover
callbacks and every fallible descendant; process abort/crash erasure is not
claimed. An implementation lacking this ownership path must reject before entry.

The fork checks and consumes the next child ordinal internally, inherits the
original scope/purpose, consumes one private 32-byte child seed, never returns
it, constructs the child through the same reviewed engine, and retires all
temporary copies. A child draw does not accept a caller-chosen replacement
ordinal. Delegates check the required purpose before invoking it.
This contract is **conditional on selecting seeded expansion**. Direct OS bytes
would be a different reviewed resource/stream contract, not an interchangeable
implementation. Preserve explicit LE word consumption and the original
`next_u128 = (first_u64 << 64) | second_u64`; freeze partition/refill semantics.

### 2.1 Source tranche: concrete recommendations, NOT SELECTABLE

The following recommendations restate the existing D28 design (§§3–5), not new
approvals: independent 32-byte OS roots; RustCrypto `chacha20 =0.10.2` with default
features disabled and `rng,zeroize`; reviewed `rand_core =0.10.1` plumbing;
original 64-bit counter/64-bit stream layout. Candidate registry checksum is
`65c35e4b699c7e15ccbe7ee35c005e4fc0a278d22238a2857e6ce2dadeda1b06`.
The standalone consumer's actual entropy/zeroize dependency closure remains to
be proposed; PLLM's `getrandom =0.3.3` is a reuse target, not a resolved consumer
dependency. No manifest, feature or lock change is selected here.

Recommended stream IDs are parent 1, secret leaf 2, mask leaf 3, error leaf 4,
set once at position zero. Serve partition-independent bytes through a guarded
64-byte buffer filled by complete reads; charge and erase the underlying
four-block generator buffer too. LE draws and the high-first `u128` rule remain
as above. Constructor-by-value copies remain unresolved; these settings do not
make the cached crate acceptable unchanged.

**Proposed private schedule, conditional on the frozen D4 graph:**

| Original object | Independent parents | Children per parent |
| --- | --- | --- |
| Dense secret, then sparse secret when needed | One Secret root each | One leaf each |
| 33 automorphisms in sorted Galois order, tensor, high switch | One Mask and one Error root per object (35 objects) | Seven leaves each, original column/row order |
| Low switch, then input ciphertext | One Mask and one Error root per object (2 objects) | One leaf each |

Thus 2 + 2×37 = 76 root fills and 2 + 2×(35×7+2) = 496
children. High/low switch direction must bind the actual source graph before
construction; a caller cannot relabel or supply ordinals. One-child roots reserve
32 delivered parent bytes; seven-child roots reserve 224. Each child ordinal is
consumed once before construction. Failed construction does not return that
ordinal, root bytes or reservation. All child descriptors and their byte/work
quotes must be admitted in the epoch ledger before the first root; checking only
a parent's seed counter is insufficient. Leaves cannot fork. New epochs require
fresh admission and fresh roots, not replacement children.

Candidate **P** remains the D28 minimal-change recommendation: preserve the
realized finite default shuffle map, without claiming the declared perfectly
uniform support law. **U** remains a distinct correction proposal; it cannot
replace P silently. For P, two secrets require 4,224 sign bytes and at most
1,048,560 shuffle bytes; masks/CDT retain §5's conditional totals. Per-leaf
coefficient/limb/batch work, rounded generator work, erasure, export and cleanup
must be added to the epoch ledger. D28's `2^32`/`2^33` candidate ceilings are not
adopted allowances; aggregate delivered-byte arithmetic is not a complete work
or memory bound.

**Precise author proofs still blocking a sampler decision:** freeze the actual
dependency/features closure and canonical execution/scope encoding; reconcile
P's realized law versus declared uniformity (or separately specify U's finite
failure law); prove seed-constructor/temporary/full-capacity erasure on all
returns and unwinds; enumerate each leaf's finite byte/work/backing quote plus
root/refill rounding and simultaneous-owner peaks; demonstrate the full secure
secret/mask/noise delegate cut and epoch-wide terminal propagation. Resolve
client construction/commit, trusted export, layout and setup bounds independently.
Neither this table nor any later Layout approval authorizes sampler code.

### Reuse boundary

PLLM already has private `Entropy::fill` plus terminal error, guarded bytes and
OS-only production construction in `crates/pllm-core/src/gkm/mod.rs:119–156`;
PUMA has the same fallibility pattern in `puma/protocol.rs:365–377`.
Those traits are protocol-private, **not an existing public shared HE API**.
If cross-crate sharing is needed, extract their byte-fill/OS-error boundary
once, retain protocol-local purpose/error adapters, and preserve all current
transcripts. Adapt that narrow provider to `rand_core::TryRng` for the upstream
boundary above; do not import GKM's field sampler, AES format or 128-attempt cap.
The standalone consumer's crate/dependency placement for this extraction is
**UNKNOWN** and requires review; adding the entire core/protocol stack is not
the proposed dependency closure. Existing root `getrandom =0.3.3` is not license
to alter the separate consumer's entropy dependencies/features.

### Required backend entry points (new, no defaults forwarding to old traits)

Add a sibling of unsafe `SamplingImpl`, retaining its placement/precision
obligations and full existing backend word types:

```rust,ignore
pub unsafe trait SecureSamplingImpl: Backend {
    fn try_scalar_znx_fill_distribution(
        module: &Module<Self>, res: &mut ScalarZnxBackendMut<'_, Self>,
        res_col: usize, dist: Distribution, source: &mut SecureSource,
    ) -> Result<(), SourceError>;
    fn try_vec_znx_add_noise(
        module: &Module<Self>, base2k: usize, k: usize,
        res: &mut VecZnxBackendMut<'_, Self>, res_col: usize,
        noise: Noise, source: &mut SecureSource,
    ) -> Result<(), SourceError>;
    fn try_vec_znx_big_add_noise(
        module: &Module<Self>, base2k: usize, k: usize,
        res: &mut VecZnxBigBackendMut<'_, Self>, res_col: usize,
        noise: Noise, source: &mut SecureSource,
    ) -> Result<(), SourceError>;
}

// Additive HAL extension, rather than change the legacy seed ABI in place.
pub unsafe trait SecureUniformImpl: Backend {
    fn try_vec_znx_fill_uniform(
        module: &Module<Self>, base2k: usize, k: usize,
        res: &mut VecZnxBackendMut<'_, Self>, res_col: usize,
        source: &mut SecureSource,
    ) -> Result<(), SourceError>;
}
```

Each core/HAL **delegate** forks the next scoped child, passes it by mutable
borrow, and drops it after the leaf. A portable leaf never receives bare seed
bytes and never calls legacy `Source::new`. No blanket implementation using
`SamplingImpl` or the old HAL uniform method is permitted.

Use companion public traits `GLWESecretSamplingSecure`, `GLWEMaskFillSecure`,
`GLWEEncryptSkSecure`, `GGLWEEncryptSkSecure`,
`GLWEAutomorphismKeyEncryptSkSecure`, `GLWETensorKeyEncryptSkSecure`,
`GLWESwitchingKeyEncryptSkSecure` and `CKKSEncryptOpsSecure`, with matching
secure OEP/reference delegation. Do not add required methods to existing
traits or require unqualified backends to implement a secure marker. Their
new `try_` methods have the exact old operands/bounds plus the secure backend
requirements, the source substitutions and result changes specified below.

| Exact frozen cut | Change on selected secure path |
| --- | --- |
| C/`layouts/glwe_secret.rs:194–203,237–242`; C/`delegates/sampling.rs:16–57` | Add `try_glwe_secret_fill_ternary_hw<S>(..., hw: usize, source: &mut SecureSource) -> Result<(), SourceError>` with existing `S: GLWESecretToBackendMut<BE> + GetDistributionMut + GLWEInfos`; add fallible delegate counterparts, fork once per original column/call. |
| P/`sampling.rs:25–52,55–94` | Implement the three secure sampling methods, using fallible integer draws in the existing scalar/noise algorithms. Eliminate legacy re-expansion at lines 34, 65, 86 on this path. `SliceRandom` cannot absorb fallible errors: the chosen, reviewed realized-map implementation must be explicitly ported, not wrapped with panic/retry. |
| H/`delegates/vec_znx.rs:470–490`; H/`oep/hal_impl.rs:318–334` | Add `try_vec_znx_fill_uniform_source` and `..._all` counterparts. Same operands/generics, `SecureSource` instead of `Source`, `Result<(), SourceError>` instead of `()`. Fork columns in increasing order. |
| P/`hal_defaults/vec_znx.rs:410–421`; P/`kernels/{vec_znx,znx}/sampling.rs` | Fallible secure uniform leaf and integer-word reads; no fourth legacy reconstruction at line 420. Preserve precision masking and zero padding. |
| C/`api/encryption.rs:24–39,75–102,345–378,438–453` and their reference/derived/OEP delegates | Add `try_` counterparts for mask/full fill, GLWE encrypt/zero-encrypt, GGLWE encryption, automorphism/tensor/switch generation. Keep every non-source operand, generic bound, scratch and arithmetic order; change both `source_xe`/`source_xa` to `&mut SecureSource`, return fallible result. Propagate through all internal calls, not just the public trait. |
| K/`delegates/encryption.rs:24–39` and matching API/OEP/reference methods | Add `try_ckks_encrypt_sk` with unchanged CT/PT/secret constraints and scratch; both secure sources; preserve CKKS validation error plus source-error cause without swallowing either. |

The selected core source-taking entry points currently return `()`; their new
counterparts use `Result<(), SourceError>`. Do not change unrelated `CoreError`
APIs. CKKS encryption already returns `CKKSResult<()>`: add
`CKKSError::Source(SourceError)` with `From<SourceError>` and display/source
handling, preserving its existing non-source errors. The explicit variant
avoids allocating an `anyhow` wrapper merely to report an entropy fault;
diagnostic/control backing still needs accounting. `SourceError` must implement
`std::error::Error` with no secret data in its diagnostic representation.
Invalid unsupported laws reject before draws/writes. The bounded candidate
supports only the selected fixed-weight ternary, uniform mask and original
encryption-noise paths; it must reject rather than enter unbounded Gaussian
table/rejection machinery or implement another distribution.

### Version and codec boundary

Leave H/`source.rs` and every old seeded/codec entry point byte-compatible.
The secure route has **no** `From<Source>`, raw `[u8;32]` mask convenience,
compressed constructor/import or implicit default backend fallback. Explicit
old seeded helpers remain legacy-only and unreachable from the private client.
GLWE/LWE/GGLWE compressed formats do not encode the expander version; a secure
source must never reinterpret those old bytes. Future compressed support needs
separate versioned codec approval. The uncompressed store binds the compiled
source/sampling-map identity in its existing execution/content binding; a label
does not confer security, client origin or adoption.

## 3. Original `generate_keys_into`: private client, not an observer requirement

K/`layouts/bootstrapping_keys.rs:238–319` is an **eager** generator. Existing public
single-key primitives already permit the same serial graph. No upstream export
observer or second key-generation algorithm is necessary. Proposed local API:

```rust,ignore
// D4 consumer types only; all fields/constructors private to original client.
// No construction from an imported secret, raw key, epoch bytes or expected hash.
pub struct ClientEpoch<BE: Backend, F> { /* original owners and failure state */ }
pub struct TrustedExportEpoch { /* bounded immutable records + origin/liveness */ }
#[derive(Clone, Copy)]
pub enum ExportObject {
    Automorphism(i64), Tensor, DenseToSparse, SparseToDense, InputCiphertext,
}
pub trait KeyExportSink {
    fn write_chunk(&mut self, object: ExportObject, bytes: &[u8])
        -> std::io::Result<()>;
    fn finish_object(&mut self, object: ExportObject, wire_bytes: u64)
        -> std::io::Result<()>;
}
impl<BE: Backend, F> ClientEpoch<BE, F> {
    pub fn generate_keys_into<S: KeyExportSink>(
        &mut self, sink: &mut S,
    ) -> Result<(), ClientError>;
}
```

`ClientError` must preserve source/core/CKKS/storage/I/O failure categories.
No public constructor or general `adopt_dense` API is proposed. Its original
client-only construction requires a complete admitted plan **before** module,
context, secret or scratch construction. That complete quote and concrete
context-construction signature remain **UNKNOWN**, so the constructor must stay
closed; a type name cannot fill the missing budget. `generate_keys_into` returns
no trusted capability by itself. Private epoch commit is possible only after
all 36 exports **and input ciphertext** have completed and checked retirement
has succeeded. The final publication/validation-secret custody signature also
needs review; do not invent an evaluator handle containing secret material.

The client owns raw/prepared dense secret sampled once through the secure
path. It iterates the original sorted Galois set, generates one raw automorphism
key, serializes that same owner with `WriterTo`, hashes/binds the stream inside
the original call, and checked-retires it before the next allocation. Then tensor;
then sample sparse **once**, preserve `ENCAPSULATED`, generate dense→sparse and
sparse→dense, retire sparse after both; finally encrypt the input with the same
prepared dense owner. Never use eager `BootstrappingKeySet`, resample on reload,
or substitute an imported secret plus a “valid/dense” flag.

Per-object sources are opened by that private client from its approved schedule:
two secret roots and distinct mask/error roots for each encryption object, if
the earlier independent-root design is selected. Do not silently replace that
proposal's 76 roots with the eager generator's three reused outer sources.
The exact root schedule is a decision input, not adopted by these signatures.

The sink receives bounded borrowed byte chunks, not a raw key or source. Adapt
existing `WriterTo`/`Write` and original-content hashing; no export archive,
attestation driver or user-provided expected digest. A public sink implementation
can allocate/fail/panic: production must use the closed, admitted bounded store;
generic `S` is not a resource proof. Callbacks run without owner/control mutexes.
Set failure-burn guard before callbacks; unwind/failure invalidates all sibling
records/publication and erases partial private material before releasing charges.
I/O completion is not proof of durable storage, kernel-cache bounds or successful
deletion; failed cleanup retains its obligations and prevents promotion.

Local trust means private construction plus continuing same-owner lineage,
not a secret fingerprint or publisher signature. Later integration at the existing
`RotationTicket::require_client_epoch` / `QuarantinedRotation::into_preparation`
seams must borrow the live, internally committed epoch and match the
actual role/Galois/layout/content record, including its shared failed state.
No imported bytes/tag/hash may construct `TrustedExportEpoch`; no detached
receipt may survive an epoch burn as authorization. Byte equality and trusted
origin are still not complete preparation/execution admission.

## 4. Minimal original-capacity API: two read-only methods

Raw C/`layouts/glwe_automorphism_key.rs:38` hides `GGLWE<D,W>`. Prepared
C/`layouts/prepared/glwe_automorphism_key.rs:12–14` hides
`GGLWEPrepared<D,B>`. These own `MatZnx`/`VmpPMat` backing, respectively.
H/`aligned.rs:22–26,97–104,118–124` retains the allocation Layout independently
of truncated length. A backend slice is not that Layout.

Proposed minimal public upstream additions, **owning `AlignedBuf` only**:

```rust,ignore
impl<W: ZnxWord> GLWEAutomorphismKey<AlignedBuf, W> {
    pub fn original_allocation_layout(&self) -> std::alloc::Layout;
}
impl<B: Backend> GLWEAutomorphismKeyPrepared<AlignedBuf, B> {
    pub fn original_allocation_layout(&self) -> std::alloc::Layout;
}
```

Bodies read the nested original `AlignedBuf`, using its original byte capacity
and alignment; never recompute from key shape, logical span or pointer divisibility.
Both wrappers can read `self.key.data` through the existing `DataView::data`:
raw data is `MatZnx<AlignedBuf, W>`, prepared data is
`VmpPMat<AlignedBuf, B::DftWord, B>`. Construct the returned Layout from that
buffer's `capacity()` and `align()`; `AlignedBuf` stores an already-valid byte
Layout. This requires no new HAL allocator or mutable backing API.
No guessed backend-independent owning Layout for generic `D` or backend borrows.
The proposed tranche chooses these two Layout-returning methods, not the
alternative `original_buffer()` API. It exposes no mutable backing or generic
capacity authority.

Wire into `staged_preparation::raw_layout` / `CheckedOwner::raw_rotation` and
the prepared rotation adoption checks in `staged_storage`. Continue checking
geometry/Galois and native alignment/span requirements. Save and charge the
actual original Layout, including spare capacity, before any parse/move; use
that saved Layout after metadata overwrite and checked release. If the selected
policy rejects oversized owners, reject explicitly rather than replace their
Layout with the smaller theoretical one. End all original borrowed views before
refill/preparation/movement. Existing `RefillGuard` failure-burn is unchanged.

This only closes **one backing's capacity evidence**. Prepared VMP helper Vecs,
noise Vec→Arc parsing, allocator metadata/mappings, scratch, context and kernel
owners remain additional. Tensor/switch owners need the same existing-ownership
audit when actually integrated; these two methods do not certify them. Full
secret-capacity erasure needs a reviewed mutable owning-storage path: a read-only
eval-key accessor or `as_mut_slice()` of truncated length cannot prove it.

## 5. Finite work, overlap and proof failures

Reuse the existing checked ownership/phase quote and original allocator charges;
do not create a second budget or call this logical meter CPU time. Admit one
raw key + ongoing secrets + scratch + exports/hash buffers + stream state and
all retained views **before** roots or construction. Disjoint phases use maxima;
simultaneously live owners add. Failed release/unmap gives no credit.

Known source arithmetic for the **restricted D4 graph**, not a method profile:

| Quantity | Known bound / qualification |
| --- | --- |
| Key inventory | 33 automorphisms, tensor, two switches; 35 high keys ×7 rows + low switch + input =247 mask/error pairs. |
| Seeded-child schedule | 2 secret +247 mask +247 error children =496; 15,872 child-seed bytes. Independent-root option: 76 fills/2,432 root entropy bytes; any random epoch identifier extra. |
| Delivered uniform-mask / CDT bytes | 4,240,965,632 / 261,021,696. Not resident arrays, CPU cycles or entropy calls. |
| Fixed-weight signs / default-map shuffle ceiling | 4,224 /1,048,560 bytes conditional on the earlier realized-map P analysis and frozen final features. Neither resolves its uniform-support discrepancy. |
| Combined delivered PRG bytes under that option | 4,503,055,984; refill/core generation, root entropy, zeroization and metadata are additional. Not an approved cap. |
| Known generation leaf subset | 419,430,400 payload /419,921,920 charged bytes; another retained original raw export adds 242,270,208 charged. Neither is a complete setup bound. |
| One high raw / prepared coefficient payload | 242,221,056 /968,884,224 bytes. Use actual Layout for ownership; these are logical frozen-shape facts. |

Known integer loops can be prepaid from public degree/limbs/rows/columns and
the chosen map. Checked multiplication/addition/ordinal overflow, short capacity,
unknown source version/law, excessive child/draw count or unsupported noise
must reject before entropy and constructors. No imported GKM rejection cap,
candidate `2^32`/`2^33` ceiling or 1-GiB proof target becomes an allowance here.
If a selected exact sampler needs rejection, its finite failure policy requires
separate approval; exhaustion yields failure, never modulo/clamp/default output.

Preserve original mask digits/padding; dyadic binary64 encryption sigma and
19-entry CDT, high/low/sign consumption, scalar/batch placement and original
ring arithmetic. The fixed-weight shuffle's realized-versus-declared law and
final `rand` feature graph remain blocking, not silently repaired by `unbiased`.

**UNKNOWN before generation:** concrete source/core/buffer Layouts and copies,
refill/seed-constructor overlap; secure shuffle/error temporary backing/stack;
complete module/context/solver/codec, metadata, hash/I/O/kernel and runtime peaks;
aggregate work including erasure/export/cleanup; complete setup bound. Logical
payload facts or observed RSS cannot substitute. D4's runtime ceilings/deadlines
and D33 compile-only guard are different contracts; this packet changes neither.

Zeroizing the visible seed does not erase by-value copies, spills, vector registers,
swapped/crashed pages or inaccessible library buffers. Inspect constructor copies,
all error/unwind paths and full retained capacity. Newer upstream tensor-scratch
normal-return wiping is not part of this frozen source and supplies no failure-path
proof here. No constant-time, malicious-host, parameter-security, circuit-privacy
or composed-error guarantee follows from these interfaces.

## 6. Selectable next action: Layout-only patch preparation

**D38-approved patch preparation only:** GPT-6.1 Sol prepares reviewable source
and test patches in isolation; Astra reviews the resulting delta. This permits
authoring a candidate implementation, not applying it to Main, changing the
consumer's active dependencies, compiling it or executing tests. D33/D35 supply
none of those additional permissions. No upstream commit is invented or adopted.

### 6.1 Exact delta and consumer binding

Upstream base is the full commit in §1. Allow only the two owning accessor
implementations and their local test modules in
`poulpy-core/src/layouts/glwe_automorphism_key.rs` and
`poulpy-core/src/layouts/prepared/glwe_automorphism_key.rs`. Necessary imports are
included. Keep HAL, allocators, samplers, encryption, codecs and dependencies
unchanged. Accessors return the stored backing's capacity/alignment, including
after truncation; they do not allocate, mutate, traverse coefficients or claim
that metadata allocations are covered.

The isolated consumer patch may change only
`benchmarks/native/poulpy_refresh/src/staged_preparation.rs` and
`benchmarks/native/poulpy_refresh/src/staged_storage.rs`, including local tests:

* `raw_layout` retains `high_key_layout`, raw sizing and backend visible-span
  checks, then validates the original Layout accessor. `CheckedOwner::raw_rotation`
  saves that actual Layout before `RotationImport::read_from` can mutate metadata.
  Existing `RotationImport::adopt` and `RotationPreparation::adopt_local_raw` consume
  this path; no new import-to-preparation authority is created.
* `CheckedOwner<Rotation>::rotation` retains its geometry/backend checks, uses
  the prepared accessor and saves the actual Layout. Existing phase adoption
  remains its consumer. `CheckedOwner<Tensor>::tensor` continues to refuse
  unproved backing; do not globally weaken `require_original_layout`.
* Reuse `StorageQuote::allocation`, `CheckedOwner::known_storage`, `drop_checked`
  and `RefillGuard`. A private scalar Layout/span validator may be extracted in
  these modules for shared raw/prepared checks; it cannot mint a trusted key or
  construct an owner from caller-supplied capacity.

Return two ordinary patches plus a concise proposed dependency-binding note:
the observed four-package identities from §1, exact patch paths/digests and
proposed future immutable revision **pending**. Do not generate a replacement
lockfile or claim patched core is byte-identical to the frozen revision. No
mixed-version resolution is presumed valid. A later adoption decision must name
the actual revised revision and coherent four-package resolution; neither an
upstream merge nor an API name supplies that approval.

### 6.2 Proposed exact acceptance rule

Let `E` be required backend span from independently checked production geometry,
`V` the real backend visible span, and `(C,A)` the accessor's original Layout.
For both raw and prepared rotations require:

1. Existing exact geometry and role/Galois checks pass at their current boundaries.
   Production `E` remains 242,221,056 raw or 968,884,224 prepared bytes.
2. `V == E`, `C >= V`, nonzero `C`, valid Layout, `A >= 64` and the existing native
   alignment requirements. Reject short/excess visible span; do not infer original
   alignment from pointer divisibility. Require the original owning accessor,
   never a caller-supplied Layout or borrowed generic backend as authority.
3. **Spare original capacity is accepted**, not truncated away: charge `C,A`
   through checked allocator geometry, including alignment/page/bookkeeping
   overhead, against existing admitted limits. Extra alignment is accepted only
   when allocator geometry and native requirements support it. Overflow, unknown
   capacity, unhealthy allocator or insufficient capacity rejects.
4. The original allocation is already charged by the bounded allocator; adoption
   does not charge it twice. Retained-owner quotes use the actual saved Layout;
   new simultaneous owners require preconstruction admission. Accessors cannot
   retroactively admit an oversized allocation made outside the existing allocator.
5. Never refresh the saved Layout from parsed shape or shorter metadata. A future
   replacing/reallocating operation needs fresh authority/admission; this tranche
   authorizes neither. Existing consuming release observes actual allocator
   deltas and failure counters; failed release provides no credit.

### 6.3 Bounded later qualification specification (preparation-era scope)

Execution was not granted during this preparation. The later D48 isolated
storage-only result below does not qualify this entire proposed suite.

All limits below are **proposed test limits**, not measured bounds or increases
to D4 runtime allowances. Ordinary targeted tests only; no benchmark harness,
receipt collection, key generation, encryption, preparation, refresh, module,
context or solver construction. If these fixtures cannot be expressed using
existing internal storage constructors, stop and return the exact missing type
or constructor; do not add another public API or allocate full-size keys.

| Evidence | Bounded construction and acceptance |
| --- | --- |
| Actual upstream raw/prepared owning accessors | In-module tests construct zero-filled inert wrappers around real tiny `AlignedBuf` owners using existing internal layout constructors; no cryptographic factory. Maximum 4,096 bytes per backing, 65,536 simultaneously live fixture bytes, 16 live backings; test-only metadata and no noise owner. Check original capacity and 64/128-byte alignment, truncation and shortened metadata, borrow end and move continuity. Never call native arithmetic on these fixtures. |
| Production arithmetic without production allocations | At most 64 table-driven scalar cases for the private validator and existing checked quote arithmetic. Include exact raw/prepared `E`, `V=E-1/E/E+1`, `C<V/C=V/C>V`, zero/invalid/unsupported alignment, overflow, exact-fit/one-short budget. Production-sized numbers are scalars only, never vector lengths or fake owner certificates. |
| Consumer binding | Compile/type-check the real accessor calls and inspect that every raw/prepared path saves their returned Layout. Tiny wrappers establish accessor behavior; scalar cases establish policy. This does **not** claim successful production-size `CheckedOwner` adoption or encrypted execution. Production shape checks remain intact; no test bypass in production. |
| Retirement failures | Reuse tiny synthetic `CheckedOwner`/`RefillGuard` fault fixtures and allocator mapper failures: successful release, denied/unobserved release, failed refill, poisoned phase and no reuse/credit. No HE object needed. Retained metadata/noise costs remain outside the one-leaf claim. |

Proposed suite maximum: 96 named/table cases total, 1 process and 1 test thread,
60 seconds execution, 256 MiB process working-set ceiling, zero new swap and
1 MiB output. These bounds exclude compilation, which requires a separate
priced compile grant; no D33 inheritance. Any unsupported bound measurement,
fixture overflow or timeout is a failed qualification, not permission to expand.
Sol must return exact test names and runnable targeted Cargo commands in its
patch-preparation result; the reviewer checks those against this envelope before
seeking execution permission. No test is run during patch preparation.

**Completion of patch preparation:** the two normal patches implement precisely
the specified accessors/bindings and encode the bounded checks above; Astra's
source review confirms no extra authority/dependency/runtime change. Compilability
and passing qualification were unverified at preparation completion; the later
D48 result below covers only the isolated accessor storage tests.
This closes a selectable patch-authoring decision, not the backing proof itself.

### 6.4 Approved preparation wording, retained as the scope boundary

> Approve **Layout-only patch preparation**: GPT-6.1 Sol may author, in isolation,
> the two owning Layout accessors, their existing raw/prepared rotation consumer
> bindings and the bounded test definitions in §§6.1–6.3 against the frozen
> Poulpy base; Astra reviews the patches. No Main application, dependency/pin
> adoption, build/test execution, HE objects at runtime, sampler or client code.
> Return patches and proposed dependency identity for separate adoption and
> qualification decisions. Approve this preparation only, or defer?

This is wired work: `RotationImport::adopt` / local-raw preparation →
`CheckedOwner::raw_rotation` → `raw_layout` currently stops at unprovable capacity;
`CheckedOwner<Rotation>::rotation` has the matching prepared refusal. The proposed
delta addresses that one coefficient-backing seam only. It does not make
`admit_encrypted_execution`, trusted export or secure sampling succeed and does
not qualify any of the five independent paper methods.

For the subsequent source tranche, the approval must name the complete §2
vertical (secret **and** all mask/noise descendants through encryption), selected
engine/law/version, dependency closure and bounded ownership proof. A source
class without these real delegates is not an acceptable substitute. No original
generation should be enabled while the UNKNOWN gates below remain.

### 6.5 D48 isolated storage evidence, 25 October 2026

HE1290's user-approved observational run, independently reviewed by W632, passed
**two test methods covering 12 sequential tiny cases**, exit 0, wall 0.438 s and
1148 bytes output. Owners supplied capacities 128/512/4096 bytes and alignments
64/128; cases covered truncation to 64 bytes, metadata reduction and moves.
**There was no empty-backing test:** zero metadata does not mean an empty
allocation. Original owning capacity/alignment, not a borrowed span, is the
tested property.

The 60 s and 1 MiB output bounds were enforced. Observed peak RSS was
**2,342,912 bytes <256 MiB**; cumulative `Swapouts` was unchanged at
**104224387 → 104224387**, with 16384-byte pages. These readings are **not hard
memory/swap isolation**. Earlier hard-guard refusal, offline missing-`f128`
refusal, explicitly approved locked cache fill and compilation history remain
historical; no earlier refusal is recast as a passing run.

The worker record identifies upstream `ca3532570ff62b96599a74f7adb85de7726b0446`,
patch `c61b57b786657ab0bbbf3f5d4f7fd1958434b6923c8e5e2d360441d261aa530e`
and unchanged lock hash prefix `4f841f88`; the review did not independently
rehash those identities. Ordinary local logs/results remain under
`he-layout-d38-upstream/target/d48-61208-*` (including `d48-61208-tests.log` and
the worker result), not a new evidence archive.

This does not adopt a dependency/pin or Main/consumer path, or qualify crypto,
keys, Module/NTT/RNG, transforms, solver/refresh, noise, source provenance,
admission, resource completeness, security or a complete HE method. See the
[D48 record](../../paper/methods/DECISIONS.md#d48-isolated-pinned-poulpy-owning-layout-qualification)
for the distinct permissions and retained exclusions.

| Contract | Present status / required disposition |
| --- | --- |
| Frozen original calls and four portable re-expansion sites | KNOWN source; existing legacy path unqualified for secure generation. |
| Minimal owning Layout seam and exact adapter consumers | D48 isolated patched-accessor storage tests passed under observational guards; Main/dependency adoption and actual consumer activation remain unapproved/unqualified. |
| Secure source/fork, full fallible delegate cut and codec separation | SPECIFIED proposed signatures; engine/version/constructor erasure and implementation UNKNOWN. |
| Governing secret shuffle law/features | UNRESOLVED decision; no correction adopted. |
| Original same-dense / once-sparse serial sink and fail-burn epoch | SPECIFIED lifecycle; client/constructor/commit admission not implemented or qualified. |
| Finite leaf counts and partial backing subset | KNOWN conditional source arithmetic; no full admission or CPU/RSS claim. |
| Full work/backing, failures, source/role/security/numeric proof gates | UNKNOWN/unqualified; fail closed. |
| Five paper profiles, TFHE/conversion paths, trained runs and SDK availability | NOT supplied or promoted by this packet. |

Approval of any implementation must state which tranche and revision, separately
from execution and security/profile adoption. **Current outcome: D28 remains
design-only; no new implementation, runtime, source pin or method is approved.**
