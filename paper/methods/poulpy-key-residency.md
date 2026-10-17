# Poulpy v0.9.0: bounded key-residency admission design

**Verdict: public phase boundaries permit fewer than 33 resident automorphism keys, but admission remains blocked.** An arithmetic-preserving public-API schedule can hold three baby-rotation keys, retire them, then hold up to seven giant-rotation keys. The existing phase implementations do **not** support a one/two-key evicting cache: they retain simultaneous borrowed key views. The complete conservative owner peak is **`null`**, not 8 GiB or the size of one key.

Source-only review, 2026-10-17. No resolution, compilation, key allocation, upstream execution, HE execution or timing measurement. This document and its [source-derived JSON](poulpy-key-residency.sources.json) are the only new artifacts. The frozen eager contract, receipts and parent-owned harness remain authoritative and unchanged by this review.

## 1. Fixed profile and evidence

- Poulpy `0.9.0`, Apache-2.0, commit `ca3532570ff62b96599a74f7adb85de7726b0446`.
- `n16_d35_k600_p19_c2s`; standard ring, degree 65,536, rank 1, 32,768 complex slots, no packing sparsity; radix exponent 52.
- Ciphertext widths 40 → 1427 → 1227 → 763 → 623 → 600; entry/output scale exponent 35, internal scale exponent 58.
- Dense secret weight 1024; ephemeral encapsulation secret weight 32. Existing high-key shape `(dnum=7,dsize=4,k_aux=224,total_k=1680,size=33)` and low dense-to-sparse shape `(1,1,68,120,3)`.
- `poulpy_cpu_portable::NTT4x30Portable<Standard>`, `SerialTaskExecutor`, unchanged portable exact-NTT arithmetic. CKKS scalar encoding/DFT-coefficient setup still uses its existing floating-point codec; this is not replacement of the ciphertext backend with FFT.
- Original maximum two sequential refreshes, one CPU thread, owner ceiling 8,589,934,592 bytes, separate host reserve 4,294,967,296 bytes, guard charge 67,108,864 bytes, zero additional swap/swapout, 600-second execution deadline and separately bounded setup. No concurrent build or HE workload is admissible.
- The existing whole-public fixture, independent binary-rational oracle, valid/padding-slot comparisons, rejection requirements and torus-only scope remain fixed. No source-method RNS mapping, production-security claim, paper-numerics qualification or Qwen inference claim follows.

The JSON records SHA-256 hashes for **49 immutable source files**. Each local file's Git blob hash was independently checked against the pinned GitHub tree response before using its bytes. Source IDs below resolve through that manifest; line ranges refer to that exact commit. The frozen `benchmarks/native/poulpy_refresh/contract.json` hash is recorded separately.

## 2. What the public interfaces actually permit

### Safe seven-key phase adapter: dependency API changes unnecessary

1. [`BootstrappingKeys`][S03] exposes associated rotation/tensor/switching providers; its documentation explicitly permits custom key managers (`S03:15–28,53–80`). Individual raw-key generation/preparation primitives are public. The eager `generate_keys` and `prepare` convenience methods are not mandatory.
2. [`GetAutomorphismKey`][S16] returns a backend view borrowed from `&self`, checks requested versus returned Galois element, and leaves physical key/decomposition selection to the provider (`S16:24–45`). A phase-local immutable map of fully prepared, fingerprint-checked keys is safe. No lazy mutation while returned views remain live is needed.
3. Context accessors expose both prepared DFT matrices and EvalMod. DFT factor operands and their baby/giant schedules are public (`S02:374–402`; `S06`; `S17:27–69`).
4. The stock per-factor wrapper allocates a `LinearTransformationBabySteps`, calls `ckks_prepare_linear_transformation_baby_steps`, then calls `ckks_eval_linear_transformation_assign` (`S07:268–289`). An adapter can make those same calls separately, dropping the first key map after preparation returns and before loading the giant-key map.
5. Crucially, the second CKKS call checks the **prepared baby cache**, not the old baby keys; it requests only nonidentity giant keys (`S10:315–348`). The cache owns convolution-domain ciphertext operands, not borrowed evaluation keys (`S18:125–183`). Thus baby keys may really be retired.
6. Both calls retain the selected backend, factor/diagonal order, full physical key shapes, convolution offset, lazy accumulator and final normalization. The CKKS wrapper performs its usual budget/scale updates and copy (`S10:197–265`). Preserve the DFT wrapper's final noise-metadata clearing as well (`S07:245–265`).

This is a **user-space orchestration adapter**, including a streaming client key-export path and a disk-backed public-key provider. It is not an implemented or memory-admitted adapter. Rust type-checking and equivalence/rejection tests remain future gates.

### Why one/two-key caching is not supplied by those calls

The [baby phase][S19] collects all its nonidentity key views into `key_refs` before computing rotations (`S19:216–250`). The [giant phase][S20] similarly collects `giant_keys`, checks all bases, derives its maximum work width, then keeps those views throughout the giant loop (`S20:192–294`). For the middle factors those counts are three and seven respectively.

Returning a view into a replaceable single cache slot would invalidate earlier borrows. A `RefCell` guard cannot be discarded while returning an unguarded view. Leaked allocations, fabricated references, overwriting still-borrowed key memory and an unbounded retaining cache do not solve residency. Mapping all prepared keys and relying on OS paging also does not establish the required physical owner bound or zero-swap guarantee.

The underlying arithmetic consumes one giant key at a time, so seven is **an existing-call lifetime requirement, not a cryptographic lower bound**. Two narrower possibilities require separate review:

- **Dependency API change:** expose scoped per-key access or begin/step/finalize phase APIs. Preflight immutable key metadata to retain the original all-key maximum width and original nonzero-giant term count; acquire one real key view for one operation, release the view, then retire its owner. Preserve the single final IDFT/normalization. This would require a reviewed patch/new immutable dependency pin.
- **Lower-level user-space implementation:** public HAL arithmetic, `DiagonalProd::accumulate_giant_prod` and mutable baby-cache views provide building blocks (`S18:164–183`; `S20:46–91`). However, the exact lazy-rotation/finalization helpers and phase drivers are private (`S21:27–155`; `S46:10–38`). A one-key adapter would need independently reviewed reconstruction of that control flow and sizing rules. It is substantially more than key storage and is not qualified here.

Calling the complete transform on one-giant submatrices and adding normalized outputs is **not** the same schedule: it changes accumulation bounds, normalization boundaries and potentially rounding/noise. Switching BSGS strategy, changing decomposition, truncating physical keys or regenerating keys per load is outside this proposal.

## 3. Exact source-derived schedule

Integer-only replay of `DFTPlan::replay_diagonals` (`S06:415–519`) and the BSGS layout (`S17`) gives the following. Indexes are slot rotations modulo 32,768; identity zero needs no key. This replay matched all 32 rotations in the frozen contract exactly. It neither generated plaintext coefficients nor executed upstream code.

| Factor | Nonidentity baby rotations | Nonidentity giant rotations | Diagonals | Compact plaintext degree |
|---|---|---|---:|---:|
| C2S 1 / S2C 4 | 2048, 4096, 6144 | 8192, 16384, 24576 | 16 | 65536 |
| C2S 2 / S2C 3 | 128, 256, 384 | 512, 1024, 1536, 30720, 31232, 31744, 32256 | 31 | 4096 |
| C2S 3 / S2C 2 | 8, 16, 24 | 32, 64, 96, 32640, 32672, 32704, 32736 | 31 | 256 |
| C2S 4 / S2C 1 | 1, 2, 3 | 4, 32760, 32764 | 15 | 16 |

C2S runs top-to-bottom; S2C bottom-to-top. Each transform has 12 nonidentity baby and 20 nonidentity giant operations. The union across both transforms is 32 rotation keys; conjugation adds the 33rd automorphism key. The affine rotations 1 and 8 are already in that union.

### One complete refresh; repeat once using the same immutable keys

| Phase | Public operation and required order | Public evaluation keys live |
|---|---|---|
| Affine checkpoint | Existing rotation, conjugation, multiply-by-two and complex offset; independent client numeric check | Rotation then conjugation, one at a time |
| ModUp | `ckks_bootstrap_mod_up`: dense→sparse switch, existing raise/scale work, sparse→dense switch | Both encapsulation keys; retain the pair for the whole call |
| C2S | Copy raised ciphertext into real destination; four factors, each baby phase then giant phase | Three baby keys, then three/seven giant keys, never both sets |
| Split | Conjugate transformed ciphertext; subtraction/addition; divide imaginary result by `i`; set both halves `Real` | Conjugation key; split temporary ciphertext also live |
| EvalMod | Real half then imaginary half; preserve source destination-width reset before overwriting second output | One tensor key reused across both halves; power-basis owners and scratch charged |
| S2C | Form `real + i*imag`; four factors in reverse schedule | Same three-then-three/seven residency pattern |
| Finish | Restore entry scale and slot-kind metadata exactly as stock pipeline; client checks all slots | No evaluation key required for scale restoration/checking |

Source call graph: `S05:388–425,488–574` → `S11:56–123` → `S07:245–289` → `S10:165–265` → `S19` / `S20`; EvalMod follows `S12:148–255` → `S15` / polynomial evaluation → tensor multiplication/relinearization. ModUp retains both real switching-key borrows. A stage-local `BootstrappingKeys` provider may explicitly reject unused tensor/rotation roles; it must never fabricate a key or silently disable encapsulation.

The adapter must reproduce every rank/degree/entry/scale/sparsity/key-shape check from the skipped outer wrappers, in addition to remaining checks inside public primitives. Missing keys, wrong identities, wrong fingerprints, unsupported shapes and insufficient scratch invalidate the whole run. No third refresh, warmup refresh or secret-bearing evaluator path is permitted.

## 4. Payload arithmetic: exact subsets, not complete process bounds

`S26:key_size`, `S24`, `S25:126–201` and `S27:308–355` determine storage. `S29:70–89` selects 8-byte raw coefficients and `S30:23–35` selects 32-byte prepared words.

```text
high limbs = ceil((52*7*4 + 224)/52) = 33
R = 65536 * 7 * 1 * 2 * 33 * 8  =   242,221,056 bytes
P = 65536 * 7 * 1 * 2 * 33 * 32 =   968,884,224 bytes
low limbs = ceil((52*1*1 + 68)/52) = 3
r = 65536 * 1 * 1 * 2 * 3 * 8   =     3,145,728 bytes
p = 65536 * 1 * 1 * 2 * 3 * 32  =    12,582,912 bytes
```

At rank 1 the tensor and sparse→dense keys have the same high-key payload `R/P`; dense→sparse uses `r/p` (`S01`, `S39`, `S40`). These are exact data-buffer payloads and therefore lower bounds on their full owners, not allocator-inclusive upper bounds.

| Simultaneously charged subset | Bytes |
|---|---:|
| Seven prepared high keys | 6,782,189,568 |
| Seven prepared high keys plus last raw key during preparation | 7,024,410,624 |
| Complete raw key set: 33 automorphism + tensor + sparse→dense + dense→sparse | 8,480,882,688 |
| Complete prepared key set | 33,923,530,752 |
| Both prepared DFT matrices, payload only | 75,758,592 |
| Both unprepared DFT matrices, payload only | 18,939,648 |
| Peak raw/prepared matrix overlap in existing context `prepare` | 85,228,416 |

Compact matrix sizes follow the period replay above, compact encoding (`S09:140–155`) and per-diagonal prepared allocation (`S23:45–77`). Each factor has one plaintext limb: C2S coefficient width `50+2=52`, S2C `35+2=37`. Prepared bytes per factor, in C2S order, are **33,554,432; 4,063,232; 253,952; 7,680**; S2C reverses them. Each matrix is 37,879,296 bytes. Context preparation releases each unprepared matrix after preparing it (`S02:440–477`), so its matrix-only peak is `prepared_C2S + raw_S2C + prepared_S2C`. EvalMod coefficients, scalar matrices during generation, encoding caches, vectors/maps and scratch are extra.

Each prepared baby cache owns **four ciphertext convolution operands**, including identity:

```text
B(k) = 4 * 65536 * 2 * ceil(k/52) * 32 bytes.
```

For C2S input widths `1427,1377,1327,1277`, these payloads are `469,762,048;452,984,832;436,207,616;419,430,400`. For S2C widths `763,728,693,658`, they are `251,658,240;234,881,024;234,881,024;218,103,808` (`S19:44–76`). Dropping baby **keys** does not drop this cache.

The largest known seven-key phase is C2S factor 2:

```text
evaluation subset = 7P + B(1377) + both prepared matrices
                  = 7,310,932,992 bytes
preparation subset = evaluation subset + R
                   = 7,553,154,048 bytes
```

Neither is an admitted peak. For illustration, an explicit adapter pool of five full-width ciphertexts (input, output, raised, real, imaginary) plus one entry-width copy adds `5*(65536*2*28*8) + 65536*2*1*8 = 147,849,216` bytes. Adding the frozen 67,108,864-byte guard charge leaves only **821,822,464 bytes** during giant-key loading, or **1,064,043,520 bytes** during evaluation, for **all** remaining owners. This pool is a proposed allocation constraint, not proof that every implicit/caller checkpoint copy fits it. Its limit must be enforced and all additional allocations separately charged.

One concrete omitted preparation owner illustrates why scratch queries alone are insufficient: the portable VMP kernel advertises `4*N*8 = 2,097,152` scratch bytes, but also creates a simultaneous `Vec<u32>` of `8*N` entries, another **2,097,152 bytes**, for each sequential polynomial column (`S49:50–55,98–111`). Both belong in preparation work, along with wrapper/alignment/allocator costs; neither was included in the subset/headroom figures above.

### Setup is independently blocked on the eager path

`BootstrappingContext::generate_keys` retains its compiled context while accumulating all raw keys (`S03:266–319`). Its raw-key payload plus prepared DFT payload plus mandatory guard charge already totals:

```text
8,480,882,688 + 75,758,592 + 67,108,864 = 8,623,750,144 > 8,589,934,592.
```

This is an admission lower bound including a required reserve, before client secrets, key-generation scratch or context metadata. An evaluator-only residency fix cannot make that setup admissible.

Required replacement is client-side serial generation/export through the individual generic primitives, one raw key owner at a time. Keep the same dense secret, sample the same required sparse-secret distribution once, derive both switches from that same sparse secret, retain no secret in delivered artifacts, then retire/wipe sparse secret material. Raw/prepared dense secrets, transformed secret copies, source state and encryption scratch remain charged (`S34:44–109`, `S35:46–113`, `S36–S42`). A rank-1 raw secret payload is 524,288 bytes; its prepared polynomial payload is 2,097,152 bytes. These small payloads do not stand in for the whole client owner.

## 5. Immutable delivery, disk and two-pass I/O

Proposed delivery is a write-once raw-key store, **not** a resident raw-key bundle. Every record binds source/profile/backend/ring/key type/Galois element/dense-key identity, exact shape, byte length and cryptographic digest. Generate in a fixed domain-separated order; load the same serialized key on every use. Do not resample on reload. Publish a complete manifest only after bounded streaming writes finish; use temporary-file-to-final atomic publication on the same filesystem. Key files contain encrypted evaluation material only.

Public raw serialization writes directly to `Write`; import can fill preallocated bounded storage from `Read` (`S24:800–825`, `S28:518–575`). The matrix reader bounds byte length against destination storage, but imported metadata can change: validate **every** expected field after decoding and reject trailing bytes before preparation. Hash while streaming into the one raw owner, then verify before use. Do not introduce a whole-file `Vec`, `to_host_bytes` copy or simultaneous raw-file read buffers. Bound headers/noise metadata and parser allocations as well.

| Quantity | Source-derived payload, excluding framing/filesystem overhead |
|---|---:|
| Immutable raw-key artifact set / setup writes | 8,480,882,688 bytes |
| Same-machine shared read-only store | One artifact set; page-cache effects still charged/controlled |
| Separate client and evaluator file copies, if required | 16,961,765,376 bytes plus transfer staging |
| Two-refresh public evaluation reads, retiring every phase's key set | 33,432,797,184 bytes |
| Prepared key payload produced over those two refreshes | 133,731,188,736 bytes cumulatively; not simultaneous residency |

Read count: per refresh, 64 DFT rotation loads + one split conjugation + one tensor load + one sparse→dense load + two affine automorphism loads = **69 high-key loads**, plus one low dense→sparse load. Two passes therefore read `138R + 2r`. Both real/imag EvalMod calls share one tensor load. Extra validation passes, key-rejection fixtures, retransmission, retained key sets or rereading for hashing must add their actual costs. No prepared-key disk cache is assumed.

Disk bytes are not RAM bytes. Conversely, filesystem buffering, compression, copy-on-write effects, resident file mappings and OS page-cache growth cannot be claimed free. A total disk-space bound, bounded delivery/parser implementation and zero-extra-swap-compatible I/O policy are **unresolved**. The 600-second timer must include online file reading, digest checks, key preparation, both refreshes and required checkpoints. Published resident-key benchmarks do not bound this I/O-heavy serial schedule. Setup/build estimates and execution-time estimates remain unqualified; a deadline is a stop rule, not a completion-time prediction.

## 6. Complete ledger and remaining admission blockers

The complete conservative peak must be a maximum over **setup and both full passes**, summing every simultaneously live owner. The following expression specifies required terms; no unknown may be replaced with zero.

```text
common = prepared_DFT_payload + EvalMod_context_and_container_owners
       + evaluator_module_all_plans + client_module_all_plans
       + retained_client_secrets_and_prepared_secrets
       + outer_ciphertext_pool + fixture/encoding/decoding/oracle_owners
       + bounded_IO_buffers + allocator/runtime/stack/retained_pages
       + guard_charge

key_prepare_work = key_prepare_scratch + key_prepare_heap_buffers
                 + preparation_wrapper_and_alignment_costs

phase_A(i) = B_i + max(3P + R + key_prepare_work,
                       3P + baby_phase_scratch_i)
phase_B(i) = B_i + max(g_i*P + R + key_prepare_work,
                       g_i*P + giant_phase_scratch_i)   [g_i = 3 or 7]
mod_up     = max(P+p+R+key_prepare_work,
                 P+p+mod_up_scratch_and_copies)
split      = max(P+R+key_prepare_work,
                 P+conjugation_scratch+owned_split_temporary)
eval_mod   = max(P+R+key_prepare_work,
                 P+power_basis_and_polynomial_heap_owners+eval_mod_scratch)
affine     = max(P+R+key_prepare_work,
                 P+affine_plaintexts_copies_and_scratch)

full_peak = max(full_context_setup_peak,
                full_serial_client_keygen_export_peak,
                common + max(all online phases across both passes))
```

Release key-preparation scratch before allocating phase-evaluation scratch; otherwise charge their sum. Keep any arena allocated throughout a phase charged even while idle. Retired Rust objects do not prove allocator-retained pages disappeared. Both passes must charge all live caches, including any retained after pass one. Lower-level scratch scopes are not new processes or zero-cost memory.

| Missing bound/proof | Exact place to close it |
|---|---|
| Setup scalar matrices, EvalMod approximation/encoded constants, maps and overlapping allocations | `S08:gen_dft_matrices`; `S09:encode_linear_transformation_from_diagonals`; `S14:compile_eval_mod`; `S02:compile_unprepared/prepare` |
| Keygen and preparation scratch, temporary secret owners, alignment/copy overhead | `S34–S42`; `S25:gglwe_prepare_tmp_bytes`; portable VMP preparation `S49` |
| All module-owned NTT/encoding/automorphism tables, for both client and evaluator | `S29`; `S31:NttPlanSet` builds the whole degree family; `S32:EncodingFFTTable`; `S33:ModulePlanCache` retains entries until module destruction |
| Conservative physical-width baby/giant scratch | `S22:45–117`; `S19:79–115`; `S21:45–74`; matrix-aware `S07:313–377`. Queries use allocated `max_size`, not just post-factor logical width. The broader CKKS query uses ciphertext as plaintext proxy (`S10:106–129`); a smaller hand estimate cannot replace it without proving the selected tighter public query |
| EvalMod power-basis and intermediate heap owners outside scratch | `S12:183–221,260–278`; `S15:119–175`; `S13:53–146` sizes scratch, not the entire power-basis heap |
| Every ciphertext, split temporary, affine plaintext, immutable checkpoint copy and client codec/oracle allocation | `S05:488–574`; `S11:79–93`; actual future adapter's complete ownership graph. `S01:229–234` explicitly retains physical bootstrap capacity after logical narrowing |
| Allocator/process overhead, retained freed pages, OS I/O effects, complete disk/walltime bounds | No source-derived complete bound established here; must be reviewed before setup admission, with parent-owned guarded execution still fail-closed |
| Parameter qualification and second-refresh entry conversion | Existing frozen contract still has null estimator/failure receipts and unimplemented numeric entry reduction; residency does not close either gate |

Thus `complete_owner_peak_upper_bound_bytes`, `complete_setup_peak_upper_bound_bytes` and `complete_disk_peak_upper_bound_bytes` remain **null**. Source establishes a useful smaller public working set, not a safe launch. It also does not establish that every same-profile schedule is impossible.

## 7. Advertised preset security and estimator association

The pinned preset publishes secret weights and modulus caps **1714 / 120**, then checks rounded gadget width, auxiliary width and total key width against those configured caps (`S01:545–569`). High-key total width 1680 and low-key total width 120 satisfy the published cap checks. Those checks do not compute lattice attack cost.

Its `p19` claim is explicitly **minimum slot-wise precision measured on a nominal reference vector with f64 matrices, rounded down**, linked to `bootstrapping_presets_meet_precision` (`S01:139–145`; `S45`). It is neither 19 security bits nor a whole-circuit failure bound. `Noise::ENCRYPTION` specifies Gaussian sigma 3.2 (`S43:26`); any estimator input must match the actual sampler convention and scaling, secret distributions, ring/dimension, key exposures and modulus interpretation.

[`SECURITY.md`][S44] advertises IND-CPA security **with appropriate parameters** and recommends `malb/lattice-estimator`. Searches of the pinned tree for estimator references, cap values and security-bit claims found this recommendation, but no preset-specific estimator commit, input script, attack-cost output, target security level or receipt tying the 1714/120 caps to this preset. This is a scoped evidence gap, not a claim that no external analysis exists. No estimator was run. Sparse encapsulation's linked paper supplies technique provenance, not automatic security qualification of this concrete combined key set.

Keeping exactly the existing parameters preserves assumptions; it does not turn the missing estimator/decryption-failure evidence into a qualification. No inference from smaller presets is used.

## 8. Recommendation and narrow next approval

Prefer finishing the **seven-key public-phase adapter's allocation and parameter proof** before considering a new one-key arithmetic loop. It preserves existing generic crypto calls and avoids a dependency API change. Required deliverable is a finite, fully sourced peak ≤8 GiB for setup and online phases, complete immutable transport/storage bounds, and remaining parameter/entry-conversion evidence. If any component remains unknown, retain blocked status.

Minimal approval question for later implementation: **Approve a new staged-orchestration/streaming-key contract, with identical cryptographic profile and original limits, only after the complete bound and parameter gates close?** Existing eager receipts should remain unchanged. If seven-key residency cannot be bounded below the cap, a separately reviewed scoped-key upstream API patch or exact lower-level user-space orchestration is the narrow alternative; neither is authorized or implemented by this review.

[S03]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-ckks/src/layouts/bootstrapping_keys.rs
[S16]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/layouts/key_helpers.rs
[S19]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/reference/linear_transformation/baby_steps.rs#L216-L250
[S20]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/poulpy-core/src/reference/linear_transformation/prepared_giants.rs#L192-L294
[S44]: https://github.com/poulpy-fhe/poulpy/blob/ca3532570ff62b96599a74f7adb85de7726b0446/SECURITY.md
