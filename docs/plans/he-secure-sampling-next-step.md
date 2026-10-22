# Shared HE: next decision, not secure-generation approval

**D28 design/review only.** This packet consolidates existing findings; it selects
no RNG, shuffle correction, dependency, upstream revision or runtime adoption.
It supplies no native HE compilation, generation or execution evidence.
Existing findings remain authoritative:
[sampling design](he-secure-sampling-design.md),
[Choice B](he-client-sampling.md),
[allocator/lifetime review](../../paper/methods/poulpy-lifetime-admission.md).

The `staged_*` Rust source cited below was uncommitted and unbuilt at the
2026-10-23 review. Those references report prototype inspection, not landed or
native-validated enforcing interfaces.

## One executable next decision

**Recommend a bounded, public owner-residency trace primitive, not RNG code.**
Extend the existing shared trace machinery; do not create another paper runner,
admission system or evidence format. The approved allocator/lifetime harness is
the reuse lane; D28 itself grants no implementation permission. To remove scope
ambiguity, the minimal explicit implementation approval is:

> Approve only a method-neutral, non-executing owner-residency extension beside
> `validate_public_ciphertext_trace` in
> `crates/pllm-compiler/src/encrypted_level_schedule.rs`, reusing its identities,
> finite preflight and immutable-parent rules. Separate handle retirement from
> retained backing charge and modeled checked-release outcomes. Permit ordinary
> synthetic correctness regressions in a later implementation assignment. No
> RNG/sampler code, dependency/pin change, upstream build/link/runtime adoption,
> HE-sized allocation, original generation, import promotion or encrypted execution.

This is an implementation-scope decision, **not** approval of the ChaCha20 design.
This review implements nothing beyond the packet.

### Smallest useful contract

- Existing trace keeps full alias backing, persistent owners and simultaneous
  scratch/recovery overlap. Its last-handle release subtracts logical backing
  (`encrypted_level_schedule.rs:1049–1058`); this is not an allocator-release
  claim or a bug in that logical contract. Preserve existing semantics.
- Add a separate charged-residency view: last handle may retire while backing
  stays charged. Only a distinct modeled successful checked-release event removes
  that charge, after all modeled borrows end. Failure retains charge and poisons
  the run; no retry, resurrection or release/reacquire credit for continuing owners.
  Reallocation charges old/new overlap before release. Count live charge, not
  cumulative allocation traffic. Bound records, scans and metadata before work.
- Reuse `StorageQuote`, `PhaseOwnerQuote`, original `Layout` accounting and sticky
  allocator failure semantics in `benchmarks/native/poulpy_refresh/src/` as contract
  inputs, not a dependency from the paper compiler onto Poulpy. Public supplied
  outcomes demonstrate consistency only: they cannot mint native release,
  capacity, client-origin or execution authority. Actual allocator observations
  remain separate, serial qualification evidence.
- Unknown original capacity/alignment remains unknown. A logical shape, backend
  span or caller number cannot create authoritative capacity. No opaque-key
  adoption bypass. Planned ordinary regressions: final alias versus release,
  retained parent, old/new overlap, failed-release poison, exact-fit/one-short,
  unknown capacity and finite-work refusal. None ran here.

**Unlock:** one shared way to reject schedules that spend retired-but-unreleased
storage twice, before any crypto implementation. **Does not unlock:** secure
sampling, native owner adoption, trusted export, complete memory admission or HE.

## Exact blockers; no inferred permission

| Gate | Existing finding and smallest remaining disposition |
| --- | --- |
| Secure expansion identity | Frozen Poulpy `ca3532570ff62b96599a74f7adb85de7726b0446` uses disclaimer-bearing concrete `Source`; replacing outer entropy misses scalar/small-noise/big-noise and separate HAL mask expansion. Approve a complete fallible entropy/parent/child/byte contract and exact implementation/features/closure separately. Proposed full-round ChaCha20, 76 parent fills and 496 children remain unapproved. Reject compressed imports: legacy seed codecs omit expander identity. No OS-seeded ChaCha8 exception. |
| Frozen shuffle law conflict | Declared uniform fixed-weight signed support conflicts with the inspected default `rand 0.10.3` chooser's documented biased integer path; final feature unification still needs freezing. P preserves realized map, not perfect uniformity; U/exact rejection or `unbiased` changes it. Explicitly dispose of the source-law conflict before implementation freeze. Neither a quantified induced bias nor an exploit is established. |
| Constructor-copy erasure | Proposed `chacha20 0.10.2` core/output-buffer drop erasure does not wipe/prove absence of the by-value seed parameter copy at `src/rng.rs:138–147`. Require approved target/compiler evidence for constructor/temporary cleanup, reviewed reference-taking construction, or separately approved primitive revision. Caller `Zeroizing`, hidden clone APIs and `ZeroizeOnDrop` are insufficient. No silent vendored patch. |
| Trusted export epoch | `staged_identity.rs` checks byte identity only and explicitly returns `MissingTrustedClientExportEpoch`. Its retained original costs another 242,270,208 charged bytes. Only a future approved original client can bind all 36 serial key exports and input ciphertext to the same dense owner, one sparse owner for both switches, exact execution identity and whole-epoch fail-burn. Mint records inside original generation/export; erase and checked-retire raw backing before next key; publish only complete epoch. Imports, digests or public tags cannot mint this authority. |
| Opaque original capacities | `staged_storage.rs::require_original_layout` rejects even a matching backend span with `UnprovenOwnerCapacity`; raw/prepared specialized adoption cannot recover original backing capacity/alignment from shape. Need authoritative construction-time ownership evidence or a separately reviewed API/revision. Generic factories can retain excess capacity. Noise `Vec`→`Arc` overlap and other metadata remain additional. Do not weaken this refusal to advance identity/preparation. |
| Finite setup / Han–Ki coefficient solver | Existing coefficient-setup audit bounds topology: 31 initial nodes, at most 32 centered nodes, 1,089 matrix entries, at most 101 iterations per local Taylor loop and 256 epsilon steps. It does **not** bound Dashu parse/base conversion, clone/add/subtract/multiply/divide, bigint capacities, scalar conversion or realloc overlap. Consumer lock/source acquisition and recorded features now exist (consumer README); older “missing lock/source” statements are historical, not today's blocker. Prove these arithmetic capacities under actual closure/allocator and frozen nightly collection implementation. The separate finite public level solver/trace budget does not bound this numerical solver or complete context setup. |

Sampling-only finite arithmetic is already recorded: candidate P delivers at most
4,503,055,984 PRG bytes; candidate U's proposed capped rejection gives
4,770,438,784. Both exclude separately charged entropy/refill details and prove
neither completion time nor memory admission. Do not repeat source research or
treat these design counts as executed evidence.

## Runtime boundary and downstream value

**Independent PLLM paper path:** reuse method-neutral contracts and independently
implement paper protocols, arithmetic and scheduling. Upstream remains
specification/oracle input, not a copied/linked paper runtime.
**Separate Poulpy D4 path:** bounded generic-building-block qualification only;
its existing allocator/storage work does not approve a revised pin or method
adoption. A secure upstream seam would need its own immutable revision approval.

| Method | Residency primitive helps represent/reject | Still blocked |
| --- | --- | --- |
| BumbleBee | OLT/bOLE original backing, masks, conversion/output overlap and checked retirement, without pretending BFV is CKKS depth. No new execution variant is approved here. | Final-paper B-N1 numeric/HE identity, source-sized wide/CRT conversion, error/flooding and two-party key contracts; qualified primitive backend. CKKS refresh is not a prerequisite for its OLT/bOLE. |
| NEXUS | Dual-key/preprocessing owners, coefficient/slot copies and refresh scratch lifetimes. | Source numerical/layout/level and uniform-ring-mask contracts, real qualified HE, Argmax disclosure; persistent decoder extension separate. |
| THOR | Deferred tensor/relinearization/rescale owners and multi-diagonal routing overlap. | Evaluated-run binding of recovered ordered chain, checkpoint/key provenance and nonlinear identity/domains; real refresh and full BERT path; autoregressive adaptation separate. |
| CacheMir | Retained KV parents versus append outputs, key/context residency and refresh overlap. | Source capacity/cost/calibration identity, encrypted prefill/cache lifecycle and full output/feedback contract. Logical trace is not its graph optimizer. |
| ROSETTA | CKKS/small-ring/LWE keys, encrypted accumulator and entry/evaluation/recovery overlap. | Complete dimension/key graph, HomComp, encrypted-affine selection, rounding/endpoints and recovery. Torus division by a power of two is not odd-modulus inverse trace division. |

## Admission stays shut

Pinned `nightly-2026-05-14` and consumer lock/source are available. Native HE
compilation/tests remain unrun: unchanged compilation guard requires 8 GiB
free+speculative memory. Last reported availability was approximately 2.5 GB;
normal memory pressure does not satisfy that guard. This packet authorizes no
build, RNG, key generation or HE execution.

Known keygen subset **419,921,920 charged bytes** is not a complete bound; **1 GiB
is a setup proof target, not an allowance**. Conditional DFT headroom of
37,666,816 bytes relies on unproved operational budgets. Preserve six staged
unknowns: setup/solver, metadata/remaining owners, remaining private heaps,
runtime, kernel and I/O. Capacity-provenance and trusted-origin gates are separate.

Retain 8,589,934,592-byte owner cap including 67,108,864-byte guard, separate
4,294,967,296-byte host reserve, zero additional swap, one execution thread,
600-second setup/execution deadlines and at most two refreshes. Parameter/RLWE
and composed error/security analysis, complete owner/process bounds, erasure
threat scope and second-refresh `600→40`/scale-35 full-slot numerics remain open.
No security, production/SDK availability, 1-GiB admission or completed method is
claimed by this proposal.
