# Maverick explicit-code native batch capsule

Source-derived **algebra conformance**, not an admitted private/verifiable model.
Implements Maverick v1 Protocol 3 and the complete two-level Protocol 2 path.
Specification and independent vectors:
[paper companion](../../paper/reproduction-contracts/maverick/README.md).

## Fixed scope

| Parameter | Value |
| --- | --- |
| Field | BabyBear, `q=2,013,265,921`; canonical u32 residues |
| Matrices | Ordered pair of 64×64 public field matrices, including coefficients outside i8 |
| Codes | Explicit public RAA8 descriptions; two privacy codes, shared primary code, secondary code |
| Privacy | `Nx=512`, exact weight 165; fresh OS randomness for every issuance |
| Primary verification | `Ny=512`, exact weight 31, two independent checks |
| Secondary verification | Dimension 128, `N'=1024`, exact weight 31, two independent checks |
| Limits | One live query/window, 16 issuance attempts per client, scalar serial execution |
| Admission | Checked 16 MiB **per-capsule** allocation ceiling, including both role owners; not process-global |

The historical seeded/i8/41 RAA and BabyBear APIs are unchanged. The proposed
larger field is rejected. No FP8/BF16 conversion, integer no-wrap claim, clipping,
truncation, model import, persistent decode, transport or SDK registration is added.

## Files and use

New implementation lives entirely under `crates/pllm-core/src/maverick_batch/`:
`arithmetic.rs`, `code.rs`, `memory.rs`, `publication.rs`, `protocol.rs`, `mod.rs`;
`testing.rs` is compiled only under `cfg(test)`.

Parent integration now exports `pllm_core::maverick_batch` from core `lib.rs`.
Deterministic injection tests retain a `#[path]` test instance. A separate
production-export test imports public input arrays and runs both complete
protocols through the real exported types, without deterministic RNG injection.
Six exported-API compile-fail doctests reject cloning and reuse of Query, Batch
and ChallengedBatch.

```text
AllocationPlan::bounded()
  -> preprocess(plan, intended matrices + explicit code descriptions)
  -> UnvalidatedPublication::validate(intended source)
  -> ValidatedPublication::split()
  -> (Client, Server)
```

`UnvalidatedPublication::import` admits externally produced P/Q/Qaux only through
the same complete validation. Every entry is recomputed, including the last
entry of Qaux. Matching hashes or a few random spot checks do not qualify it.
Intended matrices and code descriptions are compared in full, not merely by hash.
Public-source dimensions, canonical residues and permutations are checked before
allocation. Native code descriptors use the companion's fixed little-endian
encoding and SHA-256 identity. Publication identity binds all matrices, code
descriptions, derived matrices and fixed parameter choices.

**Protocol 3:** `issue_query` returns a private non-cloneable `Query` and a full
masked `Request`. Server evaluates the complete field vector. `finish_query`
consumes the query on success **and every error**, copies the claimed response,
samples two fresh exact-weight challenges, checks both equations, then demasks.
Only `AcceptedOutput::values()` releases the canonical result.

**Protocol 2:** `issue_batch` creates both private claims. After both replies,
`begin_verification` consumes the batch, validates ordered IDs, copies **all**
original responses, then samples/discloses two primary challenges. Server returns
both stacked `Qstack*e` projections. `finish_batch` consumes `ChallengedBatch`,
copies the **complete** auxiliary reply, then samples secret secondary challenges.
It checks all four secondary and all four original equations before returning
either accepted result. No secondary challenge or privacy noise goes to Server.

Server owns separate public matrix/Q buffers and replay state; it has no P,
privacy code, mask correction or client controller. Only a resource-counter
meter is shared between the in-process owners. Replies bind publication, random
session identity, attempt, mode and ordered matrix slot. Server rejects replay
and auxiliary projection before both originals have been evaluated. Dropping any
private owner cancels it, erases owned buffers and releases the live-window lease.
Failed or busy issuance still consumes an attempt; the counter is an operational
bound, **not** a deployment cryptographic lifetime.

Only OS randomness is available through normal APIs. Unbiased rejection sampling
and partial Fisher-Yates produce exact support cardinality without duplicates;
nonzero coefficients are independently sampled. Geometry and allocation are
bounded; the rejection loop has **no fixed retry bound**. Its distribution is
unchanged. Entropy failure returns an error and retires private ownership.
The OS fill scratch is `Zeroizing<[u8;4]>`, including partial-fill failure paths.
Challenge-derived diagnostic arrays stay in zeroizing owners on success and every
error path; only deterministic `cfg(test)` helpers copy their values out.
All owned word buffers zeroize before deallocation.
This does not claim erasure of CPU registers, allocator metadata or caller copies.

## Allocation receipt

`AllocationPlan` is immutable outside the module. Unsupported dimensions,
checked-size overflow, undersized ceilings and ceilings above 16 MiB reject before
allocation or hashing. Every variable-sized module buffer also reserves its
capacity plus 128 bytes before calling the allocator.

| Simultaneous charge | Bytes |
| --- | ---: |
| Client public matrices/codes/P/Q/Qaux | 2,695,168 |
| Borrowed source plus incoming publication overlap | 2,695,168 |
| Separate server matrix/Q copies | 294,912 |
| Preparation/validation/online/message transient allowance | 524,288 |
| Control, stack and allocator allowance | 262,144 |
| All 16 successful two-result outputs retained | 8,192 |
| **Planned peak** | **6,479,872** |
| **Hard allocation ceiling** | **16,777,216** |

Qaux alone is 2,097,152 bytes. Validation streams rows without another full Qaux
or dense G allocation. Runtime meters include the external/control reservation;
the dedicated allocator test independently measures requested allocation layouts,
checks release after the complete 16-window lifetime, and observes private-buffer
zeroization immediately before free. This is a bounded allocation receipt with
explicit allowances, not whole-process RSS, allocator-residency or model admission.
Arbitrary caller-created copies and multiple independent capsules require their
own aggregate owner budget.

Dynamic word-buffer exhaustion uses `try_reserve_exact` and returns
`Error::Allocation` with consuming-owner retirement. **Fixed-size `Arc::new`
allocations for the meter and client control remain an exception:** they follow
Rust's abort-on-OOM policy, even though their sizes are covered by the control
allowance. This capsule does not promise universal recoverable OOM behavior.

The allocation-failure binary returns null from the actual `GlobalAlloc` call
at each dynamic allocation index across import, full P/Q/Qaux validation, split,
query/batch issuance, evaluation, both fixation boundaries, projection and final
verification. Fixed Arc allocations are deliberately excluded. A nonallocating
ledger observes real buffer pointers and live reservation counters: admission
must precede allocation, bytes must be zero before physical free, and refunds
must follow free. It also checks exact retained-message charges after private
owners retire, monotonically consumed attempts, new-owner recovery and complete
cleanup. Observer hooks exist only under `cfg(test)` and never simulate failure.

## Targeted checks

```sh
CARGO_BUILD_JOBS=2 CARGO_PROFILE_TEST_OPT_LEVEL=2 cargo test -p pllm-core \
  --test maverick_batch_reference --test maverick_batch_allocation \
  --test maverick_batch_export --test maverick_batch_failures \
  -- --test-threads=1 --nocapture
CARGO_BUILD_JOBS=2 cargo test -p pllm-core --doc maverick_batch -- --test-threads=1
```

Coverage includes frozen Python-vector parity for complete 64×64 Protocols 3/2;
literal-matrix RAA basis/transpose checks; full P/Q/Qaux corruption and intended
identity rejection; fresh-noise round trips; exact support draw enumeration;
forged original and auxiliary replies; an auxiliary lie orthogonal to the original
input; immutable response copies; replay, malformed/omitted replies, cancellation,
attempt exhaustion and entropy failures at all private sampling phases, including
partial OS-fill rejection. The failure sweep is finite but exhaustive over the
observed dynamic allocation indices; full validation includes its final Qaux row.
Use the test-only optimization setting above for this sweep: repeating complete
validation at every allocation index exceeded a 600-second debug-run limit after
all 1,539 validation failure indices had passed. It changes no source parameters.
Deterministic noise/challenge injection exists only in `cfg(test)` helpers.

Audited 17 October 2026 targeted run: **15 native tests and six ownership
compile-fail doctests passed**. The optimized failure sweep completed in 35.69 s:

| Failure stage | Actual allocation indices exhausted |
| --- | ---: |
| Import | 23 |
| Complete P/Q/Qaux validation | 1,539 |
| Party split | 4 |
| Query / batch issuance | 8 / 16 |
| Server evaluation | 1 |
| Protocol 3 verification | 10 |
| Original fixation / primary reveal | 12 |
| Server auxiliary response | 3 |
| Auxiliary fixation / Protocol 2 verification | 13 |
| **Total injected allocation failures** | **1,629** |

The ledger checked **1,341,622 buffer destructions** for erasure-before-free and
refund-after-free, plus exact retained-message charges, private-owner retirement
and recovery under a new attempt ID. Real production-export Protocols 3/2 round
trips and the unchanged frozen vectors passed. Observed tracked peak including reservations was
**5,984,208 bytes**; independent allocator-layout peak plus external/control
reserves was **5,974,040 bytes**. Both fit the 6,479,872-byte receipt and 16 MiB
ceiling. All component allocations were released at teardown; the private-buffer
erasure probe passed. These are conformance observations, not model benchmarks.

## Unchanged qualification gates

Explicit code arrays do not prove uniform code selection, distance, or dual-LPN
security. No tiny-code certificate is extrapolated. Finite-distance evidence,
actual-code/noise security and complete query-horizon accounting remain required.
The secondary parameters are the approved bounded source-compatible capsule
instantiation, not recovered settings from the paper's evaluated model system.

Exact model FP8/activation/BF16 semantics, original eight-token Qwen3-4B validation,
accepted-state decoder ownership, complete resource measurements and ordinary
Pipeline/Experiment/SDK/gateway integration remain separate gates. Next integration
step is bounded role transport using the now-exported consuming owners;
no complete-method capability or performance result follows from this capsule.
