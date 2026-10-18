# Poulpy staged owner-bound gate: negative certificates and lifetime blocker

**Follow-up:** the [lifetime admission appendix](poulpy-lifetime-admission.md)
now implements the authorized qualification-only mmap/munmap allocator candidate.
Its conservative uniform-DFT plan has **37,666,816 bytes conditional headroom**
after per-allocation page rounding and explicit operational overhead budgets.
Those budgets remain unproved; isolated Rust compilation was blocked by host
headroom. **No full owner bound or encrypted execution is admitted.**

The preceding default-allocator review in that appendix
now closes the matrix-aware scratch query at **764,966,464 bytes**. Its charged
subset is **8,324,411,712 bytes**, leaving at most **265,522,880 bytes**, or
**23,301,824 bytes with the raw high-key owner retained**. Keeping the preceding
baby cache exceeds the cap by **204,239,168 bytes**; keeping a separate tensor
key exceeds it by **703,361,344 bytes**. These reject concrete retention policies,
not every possible same-profile schedule. The appendix narrows cross-role API
restrictions and identifies the exact solver-dependency boundary. A complete
allocator-inclusive bound remains unavailable; no encrypted execution is admitted.

**Not admitted.** For the previously proposed five-wide-ciphertext pool and seven-key phase maps, allocating the public CKKS phase scratch query gives a **live allocation subset plus the mandatory guard charge of at least 8,613,846,336 bytes**. This exceeds 8 GiB by **23,911,744 bytes**, before encoding tables, client secrets, heap temporaries, allocator overhead or disk cache.

This rejects that concrete allocation policy, **not every seven-key schedule**. The existing matrix-aware public DFT query avoids its oversized plaintext proxy. That route still has a specific unclosed owner-envelope gate: allocation/reclamation of private VMP preparation buffers and retired key buffers under the frozen default allocator. A Rust `drop` is not a source-level physical-page reclamation guarantee. No invented overhead allowance closes that gap.

Source review, 2026-10-17. The original review added this file and [its manifest](poulpy-staged-owner-bound.sources.json). The appendix adds separate integer/lifetime helpers, an uncompiled qualification allocator and dedicated tests under `benchmarks/native/poulpy_refresh/`; the original manifests, [residency design](poulpy-key-residency.md), eager contract/executable, receipts and parent-owned guard remain unchanged. No dependency resolution, compilation, upstream execution, cryptographic allocation, HE/estimator/model execution or commit. Resource totals are source deductions and explicitly labelled candidate budgets, **not measured RSS or performance results**. The appendix separately records a tiny macOS mapping API observation.

## 1. Fixed scope and meaning of the result

Poulpy v0.9.0, Apache-2.0, commit `ca3532570ff62b96599a74f7adb85de7726b0446`; `n16_d35_k600_p19_c2s`; `NTT4x30Portable<Standard>` and its serial executor; degree 65,536, rank 1, radix 52, full complex packing. Entry/internal/output widths, scales, dense/sparse secrets, all key identities and `(dnum,dsize,k_aux)` remain those in the frozen contract. No smaller preset, FFT ciphertext backend, changed decomposition or one-key arithmetic loop is considered.

Owner cap is **8,589,934,592 bytes including 67,108,864 guard bytes**; host reserve remains separately **4,294,967,296 bytes**, zero additional swap, one thread, at most two sequential refreshes, original setup/execution deadlines. The five-wide pool means five **allocated 28-limb** rank-1 ciphertexts, plus one one-limb entry copy. It is the pool proposed in the frozen residency design, not a claim that five wide owners are mathematically indispensable.

There are three different quantities:

1. **Allocation-subset lower bound:** sum of buffers actually reserved by a specified allocation policy. Other owners can only increase it. Enough to reject that policy without pretending to know a complete upper bound.
2. **Logical payload envelope:** source-derived coefficient/table/vector storage; excludes allocator capacity, headers, retained freed pages and OS storage effects unless explicitly included.
3. **Admitted total envelope:** all owners plus the frozen guard, runtime/allocator retention and attributable I/O effects. This remains `null` for the tighter schedule. No smaller payload number is substituted for it.

The manifest includes an independently executable integer replay, immutable hashes and source locators. Source IDs S01–S49 inherit the separately hash-frozen prior manifest; S50–S81 extend it. All **81** local source files were matched to Git blobs in the untruncated tree at the exact commit, then SHA-256 checked. Links use that commit, not a moving branch.

## 2. Decisive certificate: direct public phase scratch exceeds the cap

The phase arithmetic and key borrow order remain **three baby keys → retire map → up to seven giant keys**, with the prepared baby cache retained. No key view is invalidated. Factors and diagonal order, original lazy accumulation, decomposition and single final normalization are preserved.

The public [`ckks_eval_linear_transformation_tmp_bytes`](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/reference/linear_transformation.rs#L106-L117) deliberately passes the ciphertext as **both ciphertext and plaintext sizing operands**, then adds the assign-wrapper working copy. The underlying [physical-width query](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/reference/linear_transformation/eval.rs#L75-L117) uses `max_size()`, not the reduced stage `k`.

Define, for `N=65536`:

```text
Z(c,s) = N*c*s*8       raw coefficients
D(c,s) = N*c*s*32      NTT/prepared coefficients
G(c,s) = N*c*s*16      BIG coefficients
P = N*7*1*2*33*32 = 968,884,224
R = P/4              = 242,221,056
```

For the direct CKKS query at factor 2, `a_size=pt_size=28`, `prod=56`, `lazy=max(33,56)=56`, and `mask=min(56,33)=33`. The wrapper copy is different: `glwe_bytes_of_from_infos` explicitly uses `infos.k()`, so its query contribution is 27 limbs at current `k=1377` (`S79:35–36,70–71`; `S80:408–413`). This does **not** reduce the outer allocated ciphertexts or the query's physical-width convolution operands. Its lazy branch alone, including that smaller wrapper copy, is:

```text
digit = min(ceil(mask/4),7) = 7
gadget = D(1,7) + (16+8*7)*8
inner = 8*16*prod*8
lazy_rotation = D(1,mask) + G(1,prod) + Z(1,mask)
              + D(1,1) + D(2,33) + gadget
S_proxy_lower = Z(2,27) + D(2,prod) + D(2,lazy)
              + inner + D(2,33) + lazy_rotation + G(2,lazy)
              = 1,054,401,088 bytes
```

This is a **lower bound on the public query's returned reservation**, since that query takes a maximum over this and other paths. It is not a claim that the arithmetic touches every reserved byte simultaneously. Once a caller allocates the returned scratch buffer, that whole reservation is a live owner.

Transitive sizing evidence: `S21:45–74` gives lazy-rotation scratch; `S51:114–138,254–274` gives the stride-1, four-limb gadget path, selected for this backend by `S78:9` and `S56:65–80`; `S49:135–144` gives its 576-byte VMP staging; `S50:76–89` → `S52:416–417` gives fused-product staging with `CNV_ACC_GROUP=16`; `S31:294–296` gives IDFT scratch. `S29:73` selects one worker. Every omitted allocation is nonnegative; no omitted heap is treated as zero in an upper bound.

Both degree-65,536 modules eagerly construct the whole NTT family `n=1,2,...,65536` (`S29:190–196`; `S31:71–90`). Each forward and inverse table allocates `8*n` u64 twiddles (`S53:242,386`):

```text
two_modules_NTT_twiddles = 2*2*64*(2^17-1) = 33,554,176 bytes
```

This counts neither NTT step metadata nor any encoding cache. The explicit buffers below are already sufficient:

| Simultaneously reserved at C2S factor 2 | Bytes |
|---|---:|
| Seven prepared automorphism keys | 6,782,189,568 |
| Four prepared baby operands at logical input width 1377, physically 27 limbs each | 452,984,832 |
| Both prepared DFT matrices | 75,758,592 |
| Five 28-limb ciphertexts plus one entry copy | 147,849,216 |
| Both modules' NTT twiddle payloads | 33,554,176 |
| Direct public phase-query scratch, lazy-branch lower bound | 1,054,401,088 |
| Frozen guard charge | 67,108,864 |
| **Subset total** | **8,613,846,336** |
| **Excess above the unchanged cap** | **23,911,744** |

No last raw key, tensor/switch key, allocator margin or filesystem cache was added to obtain this rejection. If the phase arena is sized once at C2S entry, its wrapper copy is 28 limbs and the same subset is 8,614,894,912 bytes; re-querying at factor 2 saves one MiB but still fails. C2S factor 2 occurs in each refresh; one occurrence rejects the complete proposed run. Setup and later EvalMod cannot subtract this peak. Consequently a complete positive admission bound is unnecessary for this **negative certificate**.

## 3. Tighter public sizing: useful, but not an admission certificate

[`ckks_dft_tmp_bytes`](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/reference/dft/eval.rs#L313-L377) builds physical `max_k()` ciphertext envelopes and uses the actual matrix diagonals. This public query is applicable to the same split-phase operations. It does not change their arithmetic. Here every prepared diagonal has one limb; the corresponding lazy branch uses `prod=29`, `lazy=33`, `mask=29`:

```text
S_matrix_branch_lower = 758,675,008
same_subset_with_this_branch = 8,318,120,256
maximum_unassigned_headroom = 271,814,336
with_one_retained_raw_slot   = 8,560,341,312
maximum_headroom_after_raw   = 29,593,280
```

These are **lower bounds and upper limits on remaining headroom**, not full query values or memory upper bounds. The public DFT helper additionally covers streamed-right preparation and other operations. Encoding caches, client state, private heaps and runtime consume more. The raw slot must be charged if allocated through evaluation; dropping it is not evidence that its pages have been returned.

The follow-up independently bounds every competing scratch branch and finds the
streamed-right allowance is **6,291,456 additional bytes**. Thus the full query
is the branch above plus that allowance; the historical lower bounds remain
valid. See the appendix's executable dominance certificate and updated headroom.

One previously retired high-key allocation is **968,884,224 bytes**, greater than the entire 271,814,336-byte residual before those extra owners. Thus an unproved assumption that even **one** such allocation was reclaimed/reused can change admission by at least **697,069,888 bytes**. This is a concrete unresolved lifetime term, not a proposed overhead margin or an assertion that macOS actually retains that allocation.

### Private VMP allocation prevents a caller scratch pool from proving the total

The [portable preparation kernel](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-cpu-portable/src/kernels/ntt4x30/vmp.rs#L50-L127) uses both:

- caller scratch: `4*N*8 = 2,097,152` bytes;
- private `vec![0u32;8*N]`: **another 2,097,152 bytes**, allocated anew inside each row/column iteration.

For one high key there are `7*(2*33)=462` such allocations. Their **cumulative requested payload** is 968,884,224 bytes, although only one is logically live at a time. Loading seven giant keys issues 3,234 of them, cumulatively **6,782,189,568 bytes**, independently of the seven output-key allocations.

Across the existing two-pass schedule (138 high loads and two low loads), those private preparation vectors alone request **133,731,188,736 bytes cumulatively**. This is neither simultaneous live memory nor measured retention. It establishes why summing only maximal live buffers is insufficient without a bounded reuse/reclamation argument. The existing public VMP API provides no caller-owned replacement for this vector.

`NTT4x30Portable::OwnedBuf` is `AlignedBuf` (`S29:78–88`). Its [allocation/drop implementation](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-hal/src/aligned.rs#L46-L83) calls `std::alloc::alloc`, zero-fills, then [calls `std::alloc::dealloc` on drop](https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-hal/src/aligned.rs#L134-L140). It supplies no platform page-release or retained-capacity bound. `truncate` explicitly retains the original allocation layout (`S58:118–124`). Neither a drop nor a logical-width reduction permits subtracting retained pages from the admitted envelope.

### What public pooling actually supports

Same-role prepared automorphism owners can be refilled after every old map/view borrow ends: `glwe_automorphism_key_prepare` accepts a mutable prepared owner and updates its Galois element (`S70:147–164`). Immutable phase maps can then borrow those owners safely. No fabricated `Borrow` or cache lifetime is required.

That does **not** prove one shared allocation for automorphism, tensor, switching and scratch roles. The specialized prepared types contain crate-private GGLWE fields (`S70:12–15`; `S39:15`); the public accessors return role-specific borrows, not general owned-buffer extraction/reconstruction. `take_gglwe_prepared_scratch` constructs a generic GGLWE view, not those role-specific key-provider return types (`S69:203–230`). No safe cross-role pooling proof is established here.

The follow-up narrows this restriction: the ModUp switching associated type
accepts **generic GGLWE views**, so same-shaped sparse→dense material can use an
automorphism owner's storage after all rotation borrows end, via public
`gglwe_prepare`. Tensor lookup still requires the specialized tensor view. This
generic switching exception neither supplies a tensor constructor nor bounds
allocator retention; see the appendix for the required role/identity checks.

There is also a concrete raw-arena shortcut mismatch: `take_gglwe_scratch` allocates `ceil(infos.dnum()/infos.dsize())` rows (`S69:178–196`). Supplying this preset's `(7,4)` allocates **two rows, 69,206,016 bytes**, whereas its actual high raw key has **seven rows, 242,221,056 bytes**. The VMP preparation kernel checks row equality (`S49:83–87`). That direct helper invocation cannot justify sharing the full raw-key owner with scratch. This source observation is not a runtime test, and does not establish that every other public storage construction is impossible. Forged layout information is not an acceptable workaround.

## 4. Other allocation terms traced; none rescue the negative certificate

The following deductions prevent previously hidden allocations from being silently absorbed into `scratch_bytes`. They are **payload/operation bounds**, not a claimed complete allocator-inclusive peak. They are not needed to make the certificate in section 2 exceed the cap.

### Modules, encoding and context setup

- Both NTT families are charged above. Each f64 encoding family also owns separate forward/inverse twiddles for every slot count `1..32768`: `2*sum(round_up_64(16*2^i), i=0..15) = 2,097,280` bytes per module, **4,194,560 bytes for two** (`S32:74–88,118–123,155–177`; `S60:65–67`; `S61:72`; `S59:285–298`). These are codec tables, not a changed HE backend.
- Encoding permutations own `Vec<(usize,usize)>` swap lists; a list has at most `slots-1` entries. Across both modules, element payload is at most `2*16*(65535-16) = 2,096,608` bytes; vector capacities/containers are additional. Construction also holds `slot_map` and `seen`, at most `32768*(8+1)=294,912` bytes for the largest family member (`S62:43–72`). Cache entries persist until module destruction (`S33:78–95`).
- Matrix setup holds roots `4*32768*16 = 2,097,152` bytes and the pow5 table `(2*32768+1)*8 = 524,296` bytes. It also holds completed factors, old/new merged factors, original and cloned butterfly triples, `cd_get`, `rotate_and_mul`, and new real/imaginary diagonal vectors (`S08:81–101,138–157,185–215,232–255,273–302,394–455`). Thus its scalar matrices are additional owners even before encoded plaintexts. Both raw matrix payloads total 18,939,648 bytes; existing preparation's raw/prepared matrix overlap is 85,228,416 bytes (`S02:440–477`; prior integer period replay).
- Fixed EvalMod setup uses the existing CosHKEven **degree request 30**, interval 16, no inverse. The fitted degree is derived from its node count, not assumed equal to the request. `gen_degrees` starts with 31 nodes and immediately reaches its 31-node bound; each cluster degree stays one. Two mirrored clusters would give folded degree 16, rejected by the plan's strict depth-improvement test; either remaining choice gives folded degree 15 and MinDepth split 2 (`S81:187–215,434–439`; `S14:318–380`; `S65:87–95`).
- The coefficient solver is **256-bit `dashu_float::FBig` even for f64 output**. It owns raw/centered node vectors, sampled values, interpolation vectors and a `(nodes+1)^2` matrix, plus temporary cloned big numbers (`S81:12–20,256–393,449–464`). Here at most 32 centered nodes imply at most 1,089 matrix elements. A 256-bit precision setting is not a 32-byte allocation bound: bigint capacity and arithmetic temporaries require the arithmetic dependency's allocation contract. No full setup bound is asserted by counting only final f64/encoded coefficients. This is an additional specific setup gate, independent of the online negative certificate.

### EvalMod heap and physical scratch

The degree-15/base-4 replay gives powers `{1,2,3,4,8}` and baby degrees **`[3,3,3,1,1]`**: the MinDepth leading-split rule splits the final degree-three block again (`S65:205–270`). In addition to five stored power ciphertexts and five evaluated baby ciphertexts, private paths allocate the T2 square/double pair, the Chebyshev product/double pair and a prepared-right convolution operand (`S64:46–53`; `S15:153–188`; `S63:435–450`; `S66:102–159`; `S72:152–164`). The last is **heap-owned**, despite the EvalMod scratch query also allowing for hoisted-right work. It must not disappear from the heap ledger.

A deliberately inclusive envelope for these **coefficient buffers only** is `(5+5+2+2)*Z(2,28) + D(2,28) = 528,482,304` bytes. It charges all listed phases together at the unchanged outer physical capacity, even though their live scopes and effective widths are smaller. Index/term vectors, noise metadata and allocator retention are additional. Real and imaginary EvalMod calls are serial, so their *live* heap envelopes need not be doubled; absent a reclamation bound their retired allocations cannot simply be erased from physical accounting.

Scratch remains a separate maximum of the actual public operation queries. In particular `S13:75–105,131–145` covers the work layout and **destination allocated capacity**; it reaches multiplication/squaring, tensor/relinearization and consume-IDFT sizing (`S72:24–57,240–269`; `S71:725–849,1115–1161`; `S73:551–555`). The NTT BIG normalization itself needs `3*N*16 = 3,145,728` bytes (`S74:952–954`). A destination's reduced logical `k` is not grounds to size its copy at that smaller width. None of these owners is required to overlap the seven DFT keys logically; default-allocator retained pages remain a separate issue.

### Serial client key generation and export

One high raw key at a time replaces eager setup. Dense and sparse raw secret payloads are each 524,288 bytes, prepared payloads each 2,097,152 bytes: retaining both raw/prepared pairs costs **5,242,880 bytes** before keygen-local copies. Secret ownership remains client-only.

For the 33-limb high key, source queries give `GLWE encrypt = 135 MiB`, `GGLWE encrypt = 151.5 MiB`, automorphism keygen `154 MiB`, switching keygen `154.5 MiB`, and tensor keygen `156 MiB` (`S37:122–137`; `S36:52–62`; `S34:44–60`; `S35:46–56`; `S68:320–337`). These are query-sized scratch, including local secret copies, not whole setup bounds.

Tensor-secret preparation has **8,388,608 further heap bytes outside its advertised scratch**: two one-limb NTT vectors, one BIG vector and its separately allocated 3-MiB normalization scratch (`S67:231–233,268–280`). The already reserved keygen scratch is still live while these are allocated. Adding one high raw key, all four retained secret payloads, largest keygen scratch and this private heap gives **419,430,400 bytes** as a concrete charged keygen payload/scratch combination; module/context, source-generator/container/runtime and export owners remain extra. It is not a full setup admission bound. Serial generation must reuse the same sparse secret for both switching keys and preserve dense-key identity.

### Ciphertexts, checkpoints, parsing and disk

The certificate explicitly charges the old five-wide-plus-entry pool. Scratch-carved assign temporaries are additionally in the public query. Other private DFT allocations include the baby/key-view vectors (`S19`, `S20`), a 262,144-byte NTT automorphism permutation (`S31:719–721`), per-giant `CnvDftAccTerm` lists (`S50:55–72`), and the convolution kernel's nested schedule/term-column/row-view vectors (`S52:437–539`). These remain positive additions to the certificate. A checkpoint cannot alias a ciphertext subsequently mutated: either serialize it before mutation or charge the separate copy. Client decryption plaintext, encoding/decoding float buffers, whole-vector comparison state, affine constants and rejection/error records are positive owners outside that pool. This review does not invent buffer caps for an unimplemented transport/codec.

Raw key readers fill existing storage; matrix byte lengths are checked before reading coefficients (`S26:1583–1603`; `S28:518–550`). Noise parsing checks its component count against the already trusted output-column count and `stored <= count` before its loop (`S57:365–408`). For rank-1 keys that count is two, not an attacker-selected unbounded vector length. Import must still validate every shape, Galois element, identity/digest and trailing-byte condition before preparing a key.

Maximum valid raw framing is **117 bytes per generic GGLWE/tensor key**, **125 per automorphism/switch key**: noise `4+8+9+4+8+8+2*8=57`, gadget fields 12, matrix header 48, specialized fields at most 8 (`S57:339–361`; `S75:140–156`; `S24:818–825`; `S76:282–290`; `S77:295–305`). Thus the 36-key store has a **wire-byte upper bound of 8,480,887,180**, excluding the adapter manifest, trace, filesystem allocation and delivery copies. Two passes read at most **33,432,814,668 wire bytes** under the existing 69-high/one-low loads per pass, excluding additional validation/rejection reads. These are artifact/I/O counts, not RAM charges or runtime predictions.

Filesystem page cache is not an explicit Rust owner and is not proved absent by `Read::read_exact`. A single raw owner plus streaming hashing avoids a whole-file user-space copy; it does not prove a bound on kernel buffering or retained allocator pages. No disk cache is assigned zero in a complete bound.

## 5. Result and concrete route to reopening the bound

**Direct phase-query policy: rejected by the byte certificate. Matrix-aware policy: blocked by missing bounded reuse/reclamation.** The original branch-only residual was at most 271,814,336 bytes; the completed query tightens it to **265,522,880 bytes**, before other owners. Complete owner/setup/process/disk-allocation upper bounds remain null; the raw-key **wire** bound above is finite and distinct.

The authorized qualification allocator now supplies a source-level mapping/release mechanism for phase keys, private per-column VMP vectors, baby caches and phase scratch. Separate correctly typed public owners can be dropped and recreated without a cross-role constructor or upstream patch. The [appendix](poulpy-lifetime-admission.md) freezes this design and its remaining gates: native allocator validation, exact solver dependency allocation bounds, full staged phase liveness, and sufficiency of runtime/kernel/I/O budgets. Correct seven-row raw owners remain required. Default-allocator negative certificates above remain valid; neither a full source upper bound nor successful HE completion has been established. Arithmetic/backend/protocol changes and encrypted execution remain blocked.

Estimator association and numeric re-entry conversion remain **independent open gates**. The source's modulus-cap checks and `p19` reference-vector precision claim do not supply a preset-specific lattice-estimator receipt, a failure bound or the second-refresh 600→40 entry-conversion qualification (`S01:139–145,545–569`; `S44`). This remains torus-only bounded-qualification research, not paper representation mapping, production-security qualification, Qwen inference or a serving benchmark.
