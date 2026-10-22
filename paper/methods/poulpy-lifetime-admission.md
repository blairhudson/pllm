# Poulpy lifetime admission: bounded allocator candidate, completion still blocked

**Update: the author authorized a qualification-only allocator, retaining the
exact Poulpy source and arithmetic.** A dependency-free `GlobalAlloc` adapter now
maps each allocation separately and releases its charged mapping only after
successful `munmap`. Source arithmetic gives a **conditional** uniform-DFT
envelope of **8,552,267,776 bytes**, leaving **37,666,816 bytes**. This includes
explicit operational overhead budgets; their sufficiency remains unproved.
**Successful completion and the full owner upper bound remain unqualified.**

**Independent-lane update:** candidate DFT phases now enumerate every
known leaf allocation's original capacity and alignment (340 evaluation mappings,
342 loading mappings). An integer-only mapping ledger checks allocation overlap,
retained capacity, explicit unmap outcomes and poison. It is a simulation, not
evidence that an OS mapping was released. The Rust allocator now also rejects
same-size realloc after poison, preserving the original allocation on rejection.
The standalone allocator's ten Rust unit tests pass; no backend dependency was
compiled and no encrypted execution occurred.

Sections 1–4 preserve the source/default-allocator certificates. Sections 5–8
describe the approved allocator route, exact remaining proof obligations, and
the executable isolated-test gate. No upstream API/storage patch is selected.

Scope remains Poulpy v0.9.0 at
`ca3532570ff62b96599a74f7adb85de7726b0446`, the original torus preset,
`NTT4x30Portable<Standard>`, one execution thread, at most two refreshes,
8,589,934,592 owner bytes including the 67,108,864-byte guard, and the separate
4-GiB host reserve. This follow-up adds a Rust allocator and small isolated
fixtures. Executed checks include Python tests and standalone Rust allocator
unit tests. The earlier guarded Rust-test attempt was rejected before launch.
No HE, native dependency build, dependency resolution/download, model or estimator
execution. No source-method implementation is called.

The [separate helper](../../benchmarks/native/poulpy_refresh/staged_admission.py)
does not allocate HE-sized buffers or enable execution. Its
[source manifest](../../benchmarks/native/poulpy_refresh/staged_admission.sources.json)
inherits the original S01–S81 identifiers and adds A-series immutable hashes.
Original manifests, eager receipts, parameters and parent-owned build guard are
preserved. Resource totals below are **source calculations/declared budgets, not
memory measurements**. The independent OS probe is explicitly identified below.

## 1. The matrix-aware query is now closed

For the five-wide-ciphertext policy, both arguments to `ckks_dft_tmp_bytes`
become 28-limb physical envelopes. Actual prepared diagonals each have one limb.
`Z(c,s)=65536*c*s*8`, `D(c,s)=65536*c*s*32`, and
`G(c,s)=65536*c*s*16` as in the preceding review.

The prepared core query's dominant lazy branch is **729,314,880 bytes**.
To prove dominance, the helper conservatively bounds every other branch,
using the source's gadget output cap of 33 limbs. Smaller actual gadget windows
can only decrease these competing terms:

| Competing scratch branch | Upper bound, bytes |
|---|---:|
| Automorphism, including accumulating fallback | 270,533,184 |
| Plain multiplication | 257,949,696 |
| Right preparation | 4,194,304 |
| Baby preparation, serial executor | 329,253,440 |
| Lazy automorphism from SMALL | 241,173,056 |
| Non-lazy giant fallback | 422,576,704 |
| Copy, shifts, add/subtract and imaginary rotation | 16,252,928 |

All are below the lazy branch. The DFT helper also sizes **streamed** evaluation,
even when execution uses already prepared diagonals. Its exact additional
reservation is one degree-65536 right operand plus right-preparation scratch:

```text
prepared core query                    729,314,880
28-limb assign copy                     29,360,128
one-limb streamed right operand          2,097,152
right preparation, 8*N*sizeof(u64)       4,194,304
full matrix-aware query                764,966,464 bytes
```

The earlier **758,675,008** was correctly labelled a branch lower bound. It
omitted the final two terms, totaling **6,291,456 bytes**. This closes a scratch
query, not the private heaps or total process envelope. Arithmetic is unchanged:
three baby keys then up to seven giant keys, the same lazy accumulation and one
final normalization. The CKKS split-phase wrappers do not force allocation of
their looser ciphertext-proxy query (`S10:197–265`).

Primary sizing paths: `S07:313–377`, `S22:75–117,257–291`,
`S19:79–115`, `S21:27–74`, `S26:1521–1541`, `S51:437–494`,
`S52:38–52,646–648,737–739`, `S71:268–303`, `S73:183–216,551–555`.
The NTT automorphism-add plan uses zero scratch (`A13:155–157`), so it does
not exceed the gadget staging term inside the lazy branch.
The manifest identifies wrapper/normalization leaf sources as well.

Updated factor-2 subset:

| Quantity | Bytes |
|---|---:|
| Seven keys, current baby cache, both matrices, five wide ciphertexts plus entry, both NTT families, full query, guard | 8,324,411,712 |
| Maximum remaining allowance before all omitted owners | 265,522,880 |
| Same subset with one raw high-key owner retained | 8,566,632,768 |
| Remaining allowance after raw owner | 23,301,824 |

Encoding tables, client secrets/codec, private heaps, container capacity, allocator
metadata, runtime and attributable I/O remain additional. No allowance above is
a budget assigned to those unbounded terms.

## 2. Public lifetimes exclude the obvious no-reclamation workaround

### The baby cache must change owners

`LinearTransformationBabySteps` owns a **private** `BTreeMap` of prepared left
operands. Public mutable access changes coefficients, neither keys nor operand
shapes (`S18:125–183`). Preparation requires `prepared.size() == a.size()`
(`S19:271–272`). Factor 1 uses rotations `{0,2048,4096,6144}` at 28 limbs;
factor 2 uses `{0,128,256,384}` at 27 limbs. Its cache cannot be recycled through
these public accessors by overwriting coefficients alone. A union cache would
also request additional keys and hit the same width check.

Keeping the first cache owns **469,762,048 extra bytes**. Added to the updated
subset, it yields **8,794,173,760 bytes**, exceeding the cap by **204,239,168**,
even without a raw-key owner. Removing the entire proposed outer ciphertext
pool would save only 147,849,216 bytes; it cannot rescue this particular
retain-both-caches policy. A drop removes the logical owner, not a proved charge
for allocator-retained pages. This newly identified cache term is separate from
retired evaluation keys.

### Tensor pooling is a real typed boundary; switching is more flexible

Same-shaped automorphism owners can be refilled once all phase-map views end.
`glwe_automorphism_key_prepare` updates the Galois element (`S70:147–164`).
The integer ledger rejects refill or retirement while any modelled view exists.

The previous broad cross-role warning can be narrowed:

* **Sparse→dense switching can share high automorphism storage at the trait
  level.** `BootstrappingKeys::SwitchingKey` requires generic
  `GGLWEPreparedToBackendRef + GGLWEInfos`, not a specialized switching view
  (`S03:61–79`). ModUp passes generic views (`S05:820–830`). Public
  `gglwe_prepare` accepts an automorphism owner through its generic mutable
  implementation (`S25:207–229`; `S70:213–245`). Both high shapes agree. A phase
  provider can own the resulting generic borrow, alongside a generic borrow of
  the distinct low key. This route uses the real serialized switching key and
  validates its dense/sparse identity, direction, degrees and shape before
  preparation. The stale automorphism tag must never be exposed as a rotation;
  refill the correct automorphism material before recreating a rotation map.
  This is a source-level design, not a compiled Rust adapter.
* **Tensor lookup still requires `GLWETensorKeyPreparedBackendRef`.**
  `GetTensorKey` fixes that return type; its wrapped GGLWE field is crate-private
  (`S16:48–50`; `S39:15,154–164`). Allocating a specialized tensor owner is public;
  creating one from an automorphism owner or generic scratch view is not supplied
  by these APIs. A generic GGLWE borrow is insufficient, and `transmute` is not a
  storage/lifetime proof.

Keeping a separate tensor owner alongside seven reusable automorphism slots
adds **968,884,224 bytes** to the factor-2 subset: **703,361,344 over cap**.
This matters at S2C and later refreshes after tensor evaluation, even when its
logical work has ended: S2C factor 2 has a smaller 234,881,024-byte baby cache,
but its corresponding retained-tensor subset is still **485,257,536 over cap**.
Keeping all three specialized key-role pools instead
uses `9*968884224 + 12582912 + guard = 8,799,649,792` bytes **before** ciphertexts,
matrices or scratch. The generic switching exception saves one high owner but
does not solve tensor/cache reclamation.

## 3. Private preparation requires reuse inside a call

The pinned VMP kernel's private `vec![0u32;8*N]` remains outside caller scratch.
It requests **2,097,152 bytes** on each of **462** sequential row/column
iterations per high key (`S49:98–127`). Thus one high-key preparation has a
2-MiB maximum live vector payload but **968,884,224 cumulative requested bytes**.
No caller-owned scratch argument replaces this vector in the pinned function.

At the factor-2 evaluation subset with the raw slot retained, the residual could
hold at most **eleven** such blocks even if every other omitted owner cost zero.
Twelve retained blocks already exceed that residual. Counting all 462 as live
would be false; treating all retired blocks as certainly returned would also be
false. The helper tracks both live payload and a conservative no-reclamation
charge, and never grants release credit on `retire`.

`AlignedBuf` likewise calls `std::alloc::alloc/dealloc`; truncation preserves
allocation capacity (`S58:46–83,118–140`). Poulpy pins neither macOS `libmalloc`
behavior nor a global allocator that bounds retained pages. A process boundary
after a factor cannot establish reuse of those 462 private vectors **inside**
preparation. RSS sampling is observational and cannot prove a missed peak or
successful completion under a hard allocation budget.

## 4. The coefficient-solver unknown ends at identified dependencies

The loop topology is bounded in Poulpy: the degree-30/interval-16 request starts
with 31 nodes; permitted mirroring gives at most 32 for the folded degree-15
plan, and at most **1,089 matrix entries**. Both local Taylor loops have explicit
`n > 200` exits (at most 101 iterations); `epsilon` has 256 steps.
This is not an unbounded polynomial-degree or convergence loop.

The missing bound is the storage of **arithmetic calls**, including decimal
parsing/base conversion in `pi_big`, Context add/subtract/multiply/divide,
cloning, and the ordinary subtraction/conversion calls in `fbig_to_scalar`
(`S81:12–64,110–158,296–409`). A 256-bit rounded result does not specify
intermediate integer capacities, realloc overlap, allocator size classes or
retained pages. Counting every matrix entry as 32 bytes is invalid.

The pinned **upstream** Cargo.lock supplies exact starting identities (`A01`):

| Dependency | Version | Registry checksum |
|---|---|---|
| dashu-float | 0.6.2 | `4be5a53987dee3a7e9e874aad24cd42cf074f0aa1e6b4d1230189837ea672a49` |
| dashu-int | 0.6.2 | `64a0490b40a6a3f6427970035f17f76fd59c01d3cd9f9140809672863a0d64c8` |
| dashu-base | 0.6.1 | `861a614e79388a4c72aa66b0bdabc91e3a4f30d53e0c8d962953e207d75884d0` |

These dependency sources were absent from the inspected source snapshot and
local Cargo cache; none was downloaded. More importantly, the standalone probe
still has **no Cargo.lock**, and upstream uses semver requirements (`A02:30–31`).
An upstream workspace lock is not automatically the lock of a consumer crate.
Therefore there is no exact executed arithmetic dependency closure to certify.
The finite next proof is to freeze that closure and bound the listed operations
under its allocator, not to invent a solver allowance or replace its coefficients.

## 5. Approved allocator implementation and exact limits

Implementation:
[`src/bounded_allocator.rs`](../../benchmarks/native/poulpy_refresh/src/bounded_allocator.rs).
Frozen design:
[`bounded-allocator-contract.json`](../../benchmarks/native/poulpy_refresh/bounded-allocator-contract.json).
The existing eager executable remains separate. A future staged executable
installs `#[global_allocator] static A: BoundedAllocator =
BoundedAllocator::qualification();`. The independent smoke fixture demonstrates
that installation syntax without importing Poulpy.

Both `AlignedBuf` (`S58`) and ordinary Rust `Vec` use the global allocator. The
inspected Poulpy source supplies no competing global allocator or macOS malloc
bypass. Its optional huge-page advisory is Linux-only (`poulpy-hal/src/lib.rs`).
Other dependency/runtime paths still require the locked-closure audit.
`AlignedBuf::zeroed` checks element-size multiplication and Layout construction,
bypasses allocation for zero bytes, and routes a null result to
`std::alloc::handle_alloc_error` (`S58:46–78`). Its Drop uses the stored original
Layout even after truncation (`S58:118–140`), matching this adapter's deallocation
contract. Exhaustion is a failed qualification, not a smaller-key fallback.

| Case | Adapter handling |
|---|---|
| Page and target | Only reviewed for `aarch64-apple-darwin`; require runtime page size 16,384 before mapping |
| Header | 64 bytes within the charged mapping, immediately before the aligned user pointer; no heap registry |
| Alignment | Effective alignment is `max(Layout.align(),16)`. For alignment ≤ page, offset is `align_up(64,alignment)`; otherwise worst offset is alignment. Base is checked page-aligned; returned range stays inside mapping |
| Size arithmetic | Checked addition/rounding; reject mapped extent above `isize::MAX`, charge overflow, invalid runtime page, budget or mapping-count exhaustion |
| Zero size | Defensive `alloc`/`alloc_zeroed` returns null. `dealloc` retains ordinary GlobalAlloc preconditions: original nonzero Layout, valid live allocation of this allocator. Realloc-to-zero returns null with old allocation intact |
| Zero filling | Anonymous `MAP_PRIVATE|MAP_ANON`, read/write, fd −1, offset 0. Header writes precede user range; all user bytes retain anonymous zero initialization |
| Realloc | Same size returns original pointer. Otherwise allocate/copy/min(old,new)/unmap, including shrinking. Both complete charges coexist. Failure leaves old pointer/data valid; no inferred shrink credit |
| mmap failure | Null return after reservation rollback; saturating diagnostic counters, no heap allocation or formatting |
| munmap failure | Keep mapping charge/count, set poison, reject further allocations and qualification; no unwinding or fictitious release credit. A successful realloc followed by failed old unmap keeps both charges and poisons the run |
| Pointer misuse | Caller still owes GlobalAlloc safety contract. Layout/header mismatch aborts without unwinding; header checks do not make arbitrary invalid pointers safe |
| Concurrency | Atomic reservation checks include in-flight maps/realloc overlap. Snapshot fields are individual observations, not a coherent concurrent RSS snapshot; qualification execution stays serial |

After poison, even a same-size realloc returns null; its original pointer and
Layout remain valid for eventual deallocation. A failed old unmap during a
successful realloc retains both charges and returns the copied new allocation;
poison prevents any subsequent allocation/realloc. Deallocating another valid
owner remains permitted, but cannot erase the failed-unmap charge. There is no
production API to retry or forgive that failed deallocation.

The allocator object itself is 88 bytes on this target, charged to the external
runtime allowance; that size is asserted by the standalone Rust unit tests.

```text
owner ceiling                                  8,589,934,592
  reserved guard                                  67,108,864
  non-Rust runtime operational allowance           67,108,864
  page-table operational allowance                 16,777,216
  allocator mapping-plus-ticket ceiling        8,438,939,648
```

Each live/in-flight mapping is charged its **whole page-rounded extent** plus a
**32,768-byte kernel-bookkeeping ticket**. Maximum live/in-flight mappings: 4,096.
The large aligned-buffer charge is payload + 49,152 bytes when payload is a page
multiple and alignment ≤64; a private VMP vector therefore charges **2,146,304
bytes**, not its entire allocation history. Actual deallocation invokes munmap
before subtracting the extent and ticket. Truncating a still-owned upstream
buffer does not invoke deallocation and earns no release credit.

**Bookkeeping qualification remains open.** The tickets and 64-/16-MiB reserves
are explicit operational allowances, not source-proved XNU/runtime maxima.
VM objects, map-entry zones/caches, page-table allocation/retention, dyld/libSystem,
thread stack and attributable file cache do not become bounded merely because
Rust requested bytes are bounded. Kernel bookkeeping can outlive the user
mapping. The adapter proves a ceiling on its reserved mapping charges; it does
not prove whole-process/attributable physical memory ≤8 GiB. The unchanged outer
RSS/swap guard remains required, and sampling is still not a missed-peak proof.
The observed host is build **25F84**, Darwin **25.5.0**,
`xnu-12377.121.10~1/RELEASE_ARM64_T8142`; SDK headers freeze API constants only,
not that kernel's allocation internals. No kernel dependency source was fetched.

This route removes the need for cross-role typed constructors. End references,
drop the original specialized owner, unmap its extent, then allocate the next
specialized owner through its real public constructor. No transmute, fake raw
row count, upstream storage patch, or arithmetic substitution is involved.

## 6. Finite candidate allocation plan

[`allocator_plan.py`](../../benchmarks/native/poulpy_refresh/allocator_plan.py)
rounds **each** leaf buffer separately. Rounding total diagonal bytes once would
miss substantial small-map overhead. Its common charges include:

| Source-sized leaf owners | Mapping + ticket bytes |
|---|---:|
| Five wide ciphertexts and one entry copy | 148,144,128 |
| Both full NTT families | 36,831,232 |
| Both encoding-twiddle families | 7,274,496 |
| Both prepared DFT matrix sets | 84,377,600 |
| Both raw/prepared client secret pairs | 5,439,488 |

With seven high prepared keys, four current 27-limb baby operands and the full
matrix-aware scratch allocation, factor 2 charges **8,282,783,744 bytes** for
these source-sized leaves. It uses 340 known mappings, before the omitted
metadata/context owners. Its loading-phase counterpart is **7,764,344,832 bytes**:
one full raw key and both 2-MiB preparation buffers replace DFT scratch. Drop
raw import and preparation scratch before allocating evaluation scratch.

For a uniform DFT envelope, conservatively retain all seven key owners even in
factor 1 and charge its larger four **28-limb** baby operands. All DFT operand
widths/degrees and one-limb diagonals stay below these envelopes. The known
charge becomes **8,299,560,960 bytes**, leaving **139,378,688 bytes** of the
allocator's ceiling. This deliberately avoids assuming a smaller first-factor
pool. Only the current baby cache is retained.

The frozen target budgets add **96 MiB charged** for all remaining Rust contexts,
noise/container/plan metadata, codec, fixture and oracle owners, and **1 MiB
charged** for streaming buffers. Adding the 144-MiB fixed external reserves
gives **8,552,267,776 bytes**, or **37,666,816 bytes** below 8 GiB. The tighter
factor-2 total is **8,535,490,560**, with **54,444,032 bytes** left.

These positive margins establish **conditional compatibility**, not successful
execution. The 96-/1-MiB budgets must be proved sufficient, including per-map
rounding and realloc overlap. They are not substitutes for bounds on requests.
No complete owner or setup upper bound is populated.

The plan's `candidate_dft_phases` now includes separately named leaf Layouts for
27-/28-limb loading and evaluation, with partial charge/count totals and an
explicit `null` complete bound. The 28-limb loading charge is **7,781,122,048
bytes**; evaluation remains **8,299,560,960 bytes**. Loading and evaluation scratch
cannot coexist at the allocator ceiling, even before the unknown owners. Retiring
raw import and preparation owners without successful unmapping therefore cannot
authorize evaluation scratch. Likewise, while a retained 28-limb baby cache leaves
room for one new 27-limb leaf, the second already exceeds the charge ceiling;
the complete replacement cache cannot coexist. The tested transition releases
old mappings before creating the complete next cache. Neither test establishes
that actual typed upstream references have ended: the owned adapter is absent.

`MappingLedger` takes original Layout capacity, not logical length. Shrinking
realloc reserves a second mapping exactly as growth does; refusal leaves the old
model owner unchanged. A supplied failed-unmap outcome keeps its mapping/count
and poisons all further allocation, including same-size realloc. Python never
constructs HE-sized buffers in these checks. The old no-reclamation payload ledger
remains separate evidence for default-allocator policies, not the selected
allocator's release accounting.

The **1-MiB stream allowance covers charged Rust buffers only**. It does not also
prove a bound for attributable file cache, kernel I/O, runtime I/O internals or
their retention. Those costs remain unbounded obligations, alongside the
96-MiB container/context capacity allowance and the kernel/runtime reserves.
No solver dependency source or actual consumer lock has become available in this
lane; its arithmetic-allocation bound remains absent. No budget was relabelled a
source-proved bound, and no non-DFT phase gained a completion certificate.

Frozen phase/lifetime sequence:

1. **Public context/coefficients:** 1-GiB operational owner target, completion
   bound pending. Use the real public free function
   `poulpy_ckks::layouts::compile_eval_mod` and exact `EvalModPlan`. The private
   cosine module is not a callable external shortcut. Build prepared matrices,
   then drop raw/setup-only matrices and solver temporaries before online work.
2. **Key generation/export:** one real full-shape raw key at a time. Preserve
   one dense key identity and the same sparse secret for both switches. The
   existing **419,430,400-byte** source payload/scratch combination remains a
   partial bound; target is 1 GiB including context/runtime and bounded export.
   Never invoke the two-row generic raw-scratch shortcut for the seven-row key.
3. **Affine/ModUp and DFT:** only current operation's typed owners. For each DFT
   factor: prepare baby map → end borrows → refill up to seven giant slots →
   release raw/setup scratch → evaluate with unchanged matrix query, accumulation
   width/order and normalization. Drop old baby cache before creating the next
   differently indexed/width cache. End views before refill or retirement.
4. **Split/EvalMod:** drop DFT keys and scratch, load the specialized tensor key,
   evaluate halves serially, then drop tensor and temporaries before S2C. The
   existing **528,482,304-byte** coefficient-buffer envelope remains partial:
   public query maxima, metadata and all private allocations still need closure.
5. **S2C/re-entry/second refresh:** repeat typed staging; five outer ciphertext
   owners and entry copy remain charged. Keep each input immutable through its
   refresh and checkpoint validation before reuse. Exactly compose the existing
   CKKS wrappers; the owned Rust adapter and full liveness/type check remain
   pending. Re-entry still requires its real numerical 600→40/scale-35 proof.

The 1-GiB setup targets are proof targets, not newly implemented dynamic allocator
ceilings. The hard allocator charge ceiling stays the fixed value in section 5.
The original 600-second setup/execution deadlines and at-most-two-refresh rule
also remain separate completion obligations.

## 7. Remaining dependency and proof gates

The machine-readable contract freezes the upstream Dashu closure's **seven**
registry package identities/checksums: the three Dashu packages above,
`cfg-if 1.0.4`, `num-modular 0.6.5`, `num-order 1.2.0`, and
`static_assertions 1.1.0`. It binds upstream Cargo.lock hash
`4f841f88c4b03c5ecea115f3277c52757bb874ab2536c5d85ca046445c902a39`.
This is reference closure evidence; no standalone consumer lock or downloaded
dependency source has been manufactured.

After dependency acquisition/build is separately permitted, freeze the complete
actual consumer lock and feature graph, require those exact solver identities,
and inspect allocation capacities of decimal/base conversion, Context operations,
IBig/UBig temporaries, cloning and scalar conversion. Use the bounded 1,089-entry,
101-local-iteration, 256-bit topology from section 4; a rounded 256-bit result
still does not bound temporary limbs. A solver substituted with precomputed or
different-precision coefficients would change arithmetic and remains blocked.

Then close the source upper bounds for the declared heap/runtime/kernel/I/O
budgets, every non-DFT phase and context setup; compile the actual owned adapter
under the frozen nightly; complete independent parameter and numeric re-entry
qualification. A successful small allocator test or an allocation refusal alone
does not unlock HE. No further arithmetic/backend/protocol authorization is needed
to continue the approved allocator/lifetime harness work itself.
The container-capacity proof must include `Vec`, `BTreeMap` and `HashMap` growth
from the actual nightly's rustc/libstd source identity. Stable-test compiler
identity or abstract collection API guarantees cannot supply that upper bound.

## 8. Replay and observed evidence

```sh
python3 -m unittest discover -s benchmarks/native/poulpy_refresh/tests -p test_staged_admission.py -v
python3 benchmarks/native/poulpy_refresh/staged_admission.py --source-dir /path/to/existing/pinned/poulpy
python3 -m unittest discover -s benchmarks/native/poulpy_refresh/tests -p 'test_*.py' -v
python3 benchmarks/native/poulpy_refresh/allocator_plan.py
python3 benchmarks/native/poulpy_refresh/allocator_probe.py
```

The second command reports source-derived certificates and exits **78** even
when hashes pass. **94 source hashes verified** (81 inherited, 13 supplementary);
supplementary Git blobs also matched the exact commit's metadata. Five dedicated
tests cover scratch dominance/streamed terms,
negative retention policies, private-vector cumulative versus live payload,
borrow-safe same-role refill, and invalid lifetime/size operations. They execute
no upstream primitive. `complete_owner_upper_bound_bytes`,
`complete_setup_upper_bound_bytes` and measured peak remain `null`.

**26 Python tests passed** on this follow-up, including eleven
allocator-plan/OS tests. The OS
test maps only 32,768 bytes, verifies zero initialization/alignment and successful
munmap return. This is an API observation, not a Rust-adapter execution or an RSS
measurement. An initial test wrongly expected Linux-style ENOMEM from macOS
mincore on an unmapped range; the installed macOS man page promises no such
result, so that invalid assertion was removed. Residency bits are not used as a
release certificate.

The native fixture contains **ten Rust unit tests** plus a separate global-Vec/
aligned-allocation smoke executable. Cases include alignment/page residues,
overflow/zero, exact-cap boundaries, mapping-count limits, failed mmap rollback,
failed munmap poisoning, grow/shrink realloc overlap/failure preservation and 462
small repeated allocations. **All ten allocator unit tests pass** using an
ordinary standalone `rustc --test` build. The separate global-Vec smoke
executable remains unrun.
The two added fixtures cover retained original capacity and successful realloc
with failed old unmap (both charges retained, copied bytes preserved, subsequent
same-size/growing-or-shrinking allocation refused). These checks establish only
bounded allocator behavior, not dependency closure, a complete owner bound,
guarded resource qualification, or encrypted-refresh admission.
Fixtures live under `tests/allocator/` without `main.rs`, so Cargo does not
auto-discover standalone module tests as backend integration-test crates.

Ordinary allocator unit-test replay (no backend dependencies):

```sh
rustc --edition=2021 --test benchmarks/native/poulpy_refresh/src/bounded_allocator.rs -o target/poulpy-allocator-tests
target/poulpy-allocator-tests
```

The separately guarded qualification gate remains `allocator_probe.py`: Rust 1.85.1
only, no Cargo/dependencies/downloads, codegen units 1, serial unit tests. It reuses
the unchanged aggregate process-group guard/lock, 4-GiB host reserve, zero extra
swap, 512-MiB/120-second compile ceiling and 128-MiB/30-second run ceiling. A later
repeat under `nightly-2026-05-14` is required before qualification integration.

Recorded guarded attempt:
[`20261017T212955-allocator-f7348f37/result.json`](../../benchmarks/native/poulpy_refresh/receipts/20261017T212955-allocator-f7348f37/result.json):
**114,262,016 bytes** free+speculative versus required **4,831,838,208**; exit 78,
`child_launched=false`, no measured native peak. Original and earlier attempt
receipts are preserved. Header/API hashes, source hashes and test commands are
frozen in the dedicated contract/receipts.
