# GKM AES seeded shares: source-derived bounded profile

Status: **implemented; source profile reused by explicit seeded SDK transport**.
Sections 1–5 retain the original native capsule contract and evidence. Section 6
describes the later one-use wire binding; the [SDK reference](gkm-sdk-reference.md)
owns complete decoder qualification. The full-share profile remains distinct.

## 1. Documentary qualification

Paper: *Private and Verifiable Outsourcing of Open-Weight LLM Inference*, cached
ePrint 2026/1849, PDF SHA-256
`e1863980621e0bbd2d446b43f2d9415318ebeec400b0e6c818685752f6448e1d`, §5, PDF p.12.
The paper describes replacing a uniform share with its AES PRG seed. It does not
specify counter packing or a restart protocol.

Pinned author source: `kanav99/dvllm`, commit
`271637be7196d26a5433ca1e62b1a5357e769a5b`, tree
`e742e67ed278aba39a440a3d78c782134fe8732c`. Immutable raw files were read and hashed;
upstream was never built, linked, vendored, or executed. The existing archive is
`paper/artifacts/source-review/2026-10-16-gkm-qualification-v1/`.

| File at the pinned commit | SHA-256 | Relevant evidence |
| --- | --- | --- |
| `ext/simdcrypt/src/PRNG.cpp` | `74c8c7a8ca74712639f55f77e64525f6ec5c12e5a78b21c6a5c593c8e8910489` | Lines 43–55 key AES directly from the seed and start at counter zero; 68–75 expand successive counter blocks. Matches the archived whole-file hash. |
| `ext/simdcrypt/src/AES.cpp` | `36aa91bfb3dad739a50550c9f1cca013e1daf249a3c344abf890536f15c4fc08` | Lines 37–77 implement AES-128; 87–99 duplicate the **same** uint64 counter into both 64-bit lanes before encryption. |
| `ext/simdcrypt/include/simdcrypt/AES.hpp` | `cac11f6ce607dcb230be8f24090947b9adf23ea447f59178d3ab6e8038d2b503` | 16-byte blocks, direct seed bytes and eleven AES-128 round keys. |
| `ext/simdcrypt/include/simdcrypt/PRNG.hpp` | `c595c81c43f0032f422c15ec02bbcc5068855c3c8ecbba6d4b8d1e4c415d2e2e` | Lines 21/35 default to 256 buffered blocks; 73–92 consume bytes continuously, including across call and block boundaries. |
| `include/client.hpp` | `f4eb29a483692d3ff6bf99ea19f5e9b5625b94a7d34ee19ffdfc504941ff80d2` | Lines 232, 305, 360 consume int64 linear words or packed selector bytes. The inspected source interleaves both purposes in one worker PRG. |
| `src/serverq.cpp` | `1c814a371600ae6359ebd48b8eaf57e425f04da338aa346f1a7c5b7c64265a38` | Lines 338–349 receive a seed only in secondary/PRG mode; 488–532 expand lookup/linear inputs locally. |
| `src/llama3.2q.cpp` | `f383f90c15faecc5138b7a4533d20f86cc800b826698ac949ddfdb50ac39cb1d` | Lines 627–631 use weak `rand()` seeds; 662–673 send the seed to secondary workers only, once before generation. |

Raw URL prefix for every table entry:
`https://raw.githubusercontent.com/kanav99/dvllm/271637be7196d26a5433ca1e62b1a5357e769a5b/`.

No unresolved AES primitive or byte-consumption conflict remains for this bounded
profile. It is not an exact replica of the author's weak seeding, shared-purpose
stream, cache omissions, or unchecked counter wrap.

## 2. Frozen profile before code

Profile identifier: `gkm.seeded.aes128.duplicate-le64.two-purpose.window.v1`.

1. **Primitive:** AES-128 encryption, 16-byte key taken directly from the seed.
   Block `c` is `AES_K(LE64(c) || LE64(c))`, beginning at `c=0`. This deliberately
   preserves the source's duplicated lanes; it is not a conventional BE128 or
   zero-high-lane CTR encoding. Ciphertext bytes are consumed in increasing byte
   offset. Linear ring words are little-endian uint64 bit patterns.
2. **Freshness and purposes:** sample two independent 16-byte keys directly from
   the OS CSPRNG once per complete admitted window: linear and lookup. Both are
   bound to that window, even when one purpose has zero calls. This separation
   and secure entropy are the explicitly requested correction to the inspected
   shared-purpose `rand()` path. No derivation from public context, user seed,
   verifier state, or stochastic sampler is permitted.
3. **Roles:** worker **One** is the designated seeded worker, following the
   inspected secondary-worker path. Worker **Zero** receives only complementary
   explicit shares. The client and worker One each own their stream state. No
   seed or expanded worker-One input is delivered through worker Zero's view.
   Seed exposure to worker One is intrinsic; no adaptive-erasure guarantee is
   claimed.
4. **Consumption:** each linear step consumes exactly `8 * columns` bytes from
   the linear stream; each lookup consumes exactly `ceil(V/8)` bytes from the
   lookup stream. Calls use the frozen public schedule. Tails remain available
   to the next call of the same purpose, with no per-stage reseeding, alignment,
   padding, or skipped bytes. Selector bits are LSB-first. Unused high bits in the
   final selector byte are zeroed after consuming that byte, preserving the
   capsule's canonical dense-selector contract.
5. **Window and exhaustion:** each stream is capped at the exact byte total
   implied by the complete declared schedule. No counter may wrap or repeat;
   the final counter is `2^64-1`, giving at most `2^68` stream bytes. A one-block
   lazy buffer preserves the source byte sequence without its 256-block eager
   prefetch. This is a buffering choice, not a source performance claim. The
   actual admitted window caps are much smaller: at most 262,144 linear bytes
   or 65,536 lookup bytes under the 128-step/256-dimension/4096-row limits.
   The theoretical counter-domain endpoint is not an enlarged usage admission.
6. **Restart:** the original capsule offers no seed/cursor import, clone, rewind, retry,
   or resumption API. A new window draws fresh keys. Replay or any failed
   expansion retires the existing window; it never resamples a replacement key.
   Process restarts need fresh OS entropy; durable deployment replay accounting
   remains outside this capsule.
7. **Arithmetic and acceptance:** original dense float32 byte XOR, i16/ring64,
   nearest-even/2048 encoding, signed reconstruction and no-overflow bounds are
   unchanged. Verifier challenges remain independent unbiased OS field draws,
   after response fixation. Only the complete original window can return an
   accepted result. The new sharing profile explicitly claims computational
   privacy under the AES PRG assumption and non-collusion, not information-
   theoretic sharing privacy.

## 3. Required native integration and evidence

Reuse the existing window, pending-role response slots, sticky error state,
authenticated publication and shared publication/process admission. Worker
expansion must be bound to the exact request, profile, public schedule, ordinal,
role and artifact; it may happen at most once per step. Native acceptance must
also establish complete seeded-worker schedule consumption.

Charge client and worker AES state, keys, cursors, seed-copy temporaries, shared
full-window schedule/control handles, and expanded worker scratch in the
publication's reserved window envelope. Keep reservations while workers,
responses or accepted outputs retain payloads. Variable payloads remain fallible;
the documented fixed-size `Arc::new` OOM exception remains explicit.

Required evidence: independent AES known-answer vectors, continuous split-call
and tail tests, counter exhaustion, canonical selector padding, exact numeric and
byte oracles, matching full/seeded worker transcripts under matched test-only
random tapes, all role/context/replay/substitution failures, cancellation,
fresh-window/restart behavior, and allocation/capacity rollback. Seeds and
cursors are opaque, non-cloneable and zeroizing. Tests are bounded native tests,
not model, benchmark, or upstream execution.

## 4. Implemented ownership and resource contracts

- `Client::begin` retains the full-share default and its previous plan digest.
  `Client::begin_seeded` returns `(Window, SeededWorker)`; the latter belongs only
  to worker One. `SharingProfile::Aes128WindowV1` is explicit in bindings and the
  acceptance receipt, and uses the frozen profile identifier as its plan-hash
  domain. Neither profile changes `NUMERIC_PROFILE` or owner matrix/table hashes.
- `LinearShare::values()` and `LookupShare::selector()` now return `Option`:
  `Some` means an explicit borrowed payload; `None` means worker One must expand
  through its opaque `SeededWorker`. The seeded pending owner retains no expanded
  worker-One share. This is a native API change, not an installed SDK migration.
- Both evaluators spend the existing pending-role response slot before expansion
  or allocation. Full binding comparison includes publication, session, sequence,
  context, profile, complete plan/count, ordinal, step, artifact and role. The
  exact shared lease must match too. Duplicates, substitutions and ignored errors
  poison the pending transaction and the supplied worker's original transaction.
- Successful native expansions record schedule progress in the existing shared
  lease. Final acceptance requires all declared client steps, all worker steps,
  both exact client stream lengths, and both worker stream lengths. Merely
  submitting a correctly shaped untrusted response cannot bypass expansion.
  Response fixation and field challenges retain the original verification path.
- Dropping a window closes/poisons its lease. Dropping an incomplete worker
  retires its window. Worker, response and accepted-output owners retain the
  lease until their owned buffers/key state/full-window schedule actually drop.
  The schedule is one fallibly allocated `Vec<Step>`, shared by fixed-size Arc
  handles rather than copied into an independent worker queue.
- `PublicationResources::seeded_window_extra_bytes` reserves
  `2 * size_of::<Streams>() + 256*8 + ceil(4096/8) + 32 + 512` per live window:
  four endpoint/purpose cipher states with cursors and block buffers, expanded
  worker linear/selector scratch, two temporary seed arrays, and fixed control/
  block/key/Arc allowance. Existing charges cover the full schedule, client
  masks, responses, accumulators, reconstruction and final checks. Expanded-key
  size follows the selected RustCrypto build, not a hardcoded wire-key length.
  This conservative extra is reserved for both admission profiles, including
  publications serving mixed clients. Hard caps remain **eight live windows per
  publication**, **64 client handles per publication**, and **16 MiB aggregate
  complete publication envelopes**. No cap was increased.
- All new variable-size payloads use the existing fallible exact-capacity helper.
  Real allocator-null failures propagate `Error::Allocation`. Fixed-size
  Budget/Lease/plan `Arc::new` allocations still have stable Rust's abort-on-OOM
  semantics. Caller-owned standalone buffers, allocator overhead and incidental
  compiler/primitive stack copies are outside a total RSS guarantee. Zeroizing
  owned state is not proof of adaptive erasure or of clearing every machine copy.

RustCrypto `aes` 0.8.4 with its existing workspace `zeroize` feature is the primitive;
no dependency/lock change or upstream inference dependency was needed.

## 5. Independent fixtures and bounded native evidence

`seeded_tests.rs` uses the FIPS 197 Appendix C.1 AES-128 known-answer vector and
separate public OpenSSL fixtures. The latter were generated from documentary
counter packing, not from the native stream or author code:

```python
import hashlib, subprocess
key = bytes(range(16))
blocks = b''.join(c.to_bytes(8, 'little') * 2 for c in range(513))
stream = subprocess.check_output(
    ['openssl', 'enc', '-aes-128-ecb', '-K', key.hex(), '-nopad', '-nosalt'],
    input=blocks)
print(stream[:96].hex())
print(hashlib.sha256(stream[:8197]).hexdigest())
```

The 8197-byte prefix hash is
`862d1180811883a8684e67feb5e4d31ba33bd549be2b484492ea37fe474e51ca`.
The last-counter fixture encrypts `ff` repeated 16 times under the same key and
produces `3c441f32ce07822364d7a2990e50bb13`. The independent lookup fixture instead
uses `bytes(range(16, 32))` as its key. Fixtures are public test data, never runtime
seed injection.

Checks preserve all prior bigint/affine-group/byte oracles and allocation/lifetime
regressions. New checks cover every short split/tail, the source 4096-byte prefetch
boundary, final counter and declared-byte exhaustion, selector padding, exact
matched-tape full/seeded payload equality across repeated matrices/different
widths/interleaved lookups, signed/float outputs, all binding fields, profile and
artifact substitution, 600 replay attempts, ignored constructor/expansion errors,
cancellation, incomplete expansion, corrupted replies, seed and verifier entropy
failures, fresh process keys/identities, allocation sweeps and real null-payload
allocations. Profile/framing identities intentionally differ; payload equality is
under matched test-only random tapes, not an assertion about independently drawn
production transcripts.

The native allocator test checks a full 128-step window and includes inline
`Window`/`SeededWorker` state in its conservative peak comparison. Shared
publication admission still has a concurrent multi-client regression, and the
aggregate-publication contention seam reserves exactly two computed envelopes
so it accounts for target-specific AES state size. Production's 16 MiB cap stays
fixed.

Executed sequentially on the Apple-Silicon host after memory-reserve checks
(32 GiB total, 70–72% reported free); cached/offline, `CARGO_BUILD_JOBS=2`:

```sh
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core gkm --lib -- --test-threads=2
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --test gkm_allocation -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --test gkm_seeded_restart -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --doc gkm
CARGO_BUILD_JOBS=2 cargo clippy --offline -p pllm-core --lib --tests -- -D warnings -A clippy::needless_range_loop
```

Initial slice results: 37 unit tests; two allocator tests; restart test binary passes and checks
two fresh child processes; six compile-fail ownership checks. This is conformance
evidence, not a throughput, latency, acceleration, or resource-eligibility result.
Clippy passes with the pre-existing `byte_dot.rs` range-loop lint suppressed;
targeted rustfmt and diff-whitespace checks pass.

### 5.1 Independent-audit conformance closure — 17 October 2026

The independent read-only audit reported no production blocker and requested
four additional conformance checks. This follow-up changes tests and the two
payload-getter compile-fail examples only; the source profile, arithmetic,
commitments and production admission caps retain their previous definitions.

**Maximum-shape allocation evidence.** `tests/gkm_allocation.rs` now traces owned
matrix/table imports, publication/client construction and eight simultaneously
admitted seeded windows. All eight pending owners and both response buffers per
window coexist before freezing. Accepted outputs continue holding all eight
leases after worker owners are dropped. The comparison includes conservative
inline sizes for source/client/publication return owners, key-bearing workers,
pending/partial/window/accepted holders, response holders and borrowed views.
Caller-owned raw import/oracle fixtures are outside the trace; allocator headers
and incidental compiler stack copies remain outside the payload-envelope claim.
The existing 128-tiny-lookup case and real null-allocation regressions remain.

Measured requested-allocation bytes on this Apple-Silicon build:

| Fixture; eight live windows | Heap peak | Inline allowance | Combined | Admitted complete envelope |
| --- | ---: | ---: | ---: | ---: |
| Linear `256×256` | 414,492 | 39,544 | 454,036 | 1,069,748 |
| Lookup `4096×256`, 512-byte selectors, 1024-byte rows | 4,350,522 | 39,256 | 4,389,778 | 4,888,164 |

Each combined value fits the existing full publication envelope. Both cases
retain exact output checks: an independently widened signed-integer oracle for
linear results and original per-row byte comparisons, including negative zero,
for lookups. These are bounded allocation conformance checks, not RSS or
performance measurements.

**Lookup failure paths.** `seeded_window_tests.rs` mirrors all eleven binding
mutations through `SeededWorker::evaluate_lookup`, plus the actual complementary
worker view, a same-shape substituted table, cross-window worker/response
substitution, 600 duplicate expansion attempts, retained response leases and
ignored untrusted-constructor errors. The ordinary `FloatTable::evaluate` rejects
a seed-only `None` selector and poisons retries. A valid raw response constructed
from the independent OpenSSL selector fixture and an explicit byte-XOR oracle
passes row authentication, but still fails final acceptance with `Incomplete`
when the designated worker never expanded its step.

**Independent restart fingerprints.** `tests/gkm_seeded_restart.rs` now executes
both lookup and linear steps in each fresh child process. Its lookup fingerprint
is 32 selector bytes (two complete AES blocks), with the known one-hot token
removed. Lookup, linear and request fingerprints are compared separately. Fresh
linear/session/publication entropy therefore cannot hide lookup-key reuse. This
remains a fresh-initialization test, not a durable restart/resume guarantee.

**Borrow proofs.** The `LinearShare::values` and `LookupShare::selector` rustdoc
examples retain their respective `Option<&[u64]>`/`Option<&[u8]>` payload, attempt
to drop the pending owner, then use the payload. Both fail compilation with the
owner still borrowed (`E0505`).

Follow-up checks ran sequentially, offline, with `CARGO_BUILD_JOBS=2`, after a
fresh headroom check reported 72% free on the 32 GiB host:

```sh
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core gkm::window::seeded_window --lib -- --test-threads=2
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --test gkm_allocation -- --test-threads=1 --nocapture
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --test gkm_seeded_restart -- --test-threads=1
CARGO_BUILD_JOBS=2 cargo test --offline -p pllm-core --doc gkm
CARGO_BUILD_JOBS=2 cargo clippy --offline -p pllm-core --lib --tests -- -D warnings -A clippy::needless_range_loop
```

Results: **11 focused seeded-window tests, four allocator tests, two restart
binary tests (including two fresh child processes), and eight compile-fail
checks passed**. Clippy passed with the existing range-loop lint suppression.
No unexpected production bug was found; no cap adjustment or commit was made.

## 6. Consuming seeded transport

The explicit `OpenWeightVerifiableOutsourcing(sharing="seeded")` composes this
profile with the existing authenticated SDK transport, for both Capsule and
Decoder capacities. `Client::begin_seeded_wire` transfers the two fresh keys
only in Worker One's consuming start frame. Worker Zero's start contains the
complete public schedule without keys; its operations retain complementary
explicit shares. Worker One's subsequent operations are header-only. This is a
bound request-start operation, not a general cursor import or restart API.

The existing frame namespace binds source publication, sharing profile, complete
plan, request/session, role and monotonic ordinal. The worker verifies exact
stream exhaustion before completion; malformed starts, substitution, replay and
cancellation retire its owner. Confidentiality and authentication are both
required for key delivery. Trusted loopback and verified HTTPS have their existing
deployment contracts; co-located children do not establish operator independence.

The in-process progress rule in section 4 cannot prove a remote worker's actual
AES computation. The wire client records a bound reply acknowledgement only as
transport progress. Original response fixation and complete-window numeric
verification remain authoritative; acknowledgements cannot accept outputs on
their own. Worker-local stream checks do not become a malicious-worker proof.

Native key/stream, scratch, frame and schedule owners retain payload leases until
retirement. Wire admission charges seeded-only expansion scratch to Worker One;
all roles still price their actual inline owners. Decoder publication estimates
include its larger maximum-width scratch and remain separate from the historical
Capsule bounds above. Outer SDK admission covers Python frame copies; native
zeroization cannot erase caller-owned immutable bytes. Durable replay state remains
external. Arithmetic and source matrix/table commitments are unchanged; seeded
sharing adds an explicit computational-privacy assumption and distinct binding.

## 7. Remaining gates

The [bounded decoder reference](gkm-decoder-reference.md) and later
[SDK transport](gkm-sdk-reference.md) compose this profile with generated semantic
prefill/decode and final-window acceptance. Decoder capacity admits larger source
geometry separately; the original Capsule evidence cannot supply its checkpoint
or performance results.

- Bounded native source/ownership audits passed. Independent cryptographic review
  and complete composed-protocol security remain open.
- Seeded wire, authenticated role admission and separate-worker execution are now
  implemented. Independent deployment, transport retry/reconnect and durable
  crash-resumption/replay accounting remain open. Fresh-process tests establish
  fresh initialization, not restart persistence.
- Batched prefill, seeded full-checkpoint state/quality qualification and stochastic
  sampler integration remain open. Broader source-model quality is separate from
  the historical full-share Qwen quality smoke.
- SIMD/GPU and complete source-performance qualification remain open; generated
  tests do not supply measured full-checkpoint communication savings.
- Any different seed scope, counter packing, purpose schedule, numeric graph or
  acceptance window remains a separate versioned review/approval decision.
