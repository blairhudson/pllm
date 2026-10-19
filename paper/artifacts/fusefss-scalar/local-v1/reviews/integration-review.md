# FuseFSS scalar integration: independent read-only review v2

## Verdict

No blocking scalar arithmetic, private-table disclosure, reuse, lifecycle, or
logical-budget defect found in the inspected implementation. This is a source
review, not independent cryptographic certification or a fresh test pass.

**Protected replay qualification remains pending.** At inspection,
`target/fusefss-native-protected-v1/` contained only `protected.jsonl`; neither
success receipt was present. The running replay was not disturbed. A partial
stream is not completion evidence. No builds, tests, benchmarks, model loads,
commits, or existing-artifact modifications were performed by this review.

## Findings

### P3 — early export rejection can retain an older adapter success receipt

Location: `crates/pllm-garble/src/fusefss/conformance.rs:519-574`;
`crates/pllm-garble/src/fusefss_backend/conformance.rs:73-78,174-209`.

The adapter deletes `protected-receipt.json` only after opt-in, identity, fixture,
context, and cumulative-work preflight. An earlier successful receipt therefore
survives a later invocation rejected by one of those guards. The native wrapper
removes `runtime-receipt.json` earlier, so such a rerun can leave an old adapter
receipt without a runtime receipt. Likewise, adapter success is published before
the wrapper performs final runtime reconciliation. This is not a false claim by
the adapter about its own successful prior stream, nor a backend privacy defect.
It is an artifact-consumer hazard if the mere presence of the adapter receipt is
treated as success of the latest complete native run.

Recommendation: consumers require both receipts, matching output identity and
reviewed source hashes, successful process exit and independent checker result.
If latest-attempt semantics are desired, use an explicit attempt identifier and
transactional publication/invalidation. No stale receipt was observed in the
currently running attempt.

### P3 — module privacy summary needs the public specialization qualifier

Location: `crates/pllm-garble/src/fusefss_backend/mod.rs:8-10`.

The opening summary says neither party receives boundaries or clear coefficients.
That is true for `PrivateVector`, not `PublicBoundary`: each public party owns a
copy of original public boundaries and coefficients. Detailed comments and types
correctly distinguish this (`mod.rs:189-195`, `compiled.rs:41-43`, shared ring
module). No shifted/private partition is disclosed. Qualify the summary when
documentation is next edited; this does not block bounded scalar execution.

## Source proofs / checked obligations

### Public-original-boundary mapping

* `compiled.rs:45-65` constructs only the frozen descriptor and compiles **zero**
  rotation into `RingTable`. `compile.rs:307-335` preserves original ordering at
  zero rotation, inserting the first eligible public split with duplicate payload
  to obtain the fixed padded shape. This changes no table value.
* Actual fresh mask goes only into universal DCF issuance and dealer correction;
  no mask-shifted table is passed to a party. Shared `ring.rs:175-180` contains
  original public table, role, private comparison and randomized base only.
* Let `D_p` be reconstruction of universal eval at boundary p, `c_p` its shift
  correction, N the ring modulus, u=x+r mod N. `interval_fss.rs:99-148` gives
  `D_p-D_0 + c_p-c_0+[r+p>=N] = [x<p]` for original boundaries. Sentinels p=0,N
  produce 0,1. Shared ring `:189-202,288-311` folds exactly that last correction
  into secret base and applies deltas `v[j]-v[j+1]`. Thus reconstruction is
  `v_last + sum delta_j*[x<boundary_j]`, including wrap and r=0.
* Horner uses original input shares (`mod.rs:626`), not opened u. Shared Beaver
  `shared_wide_product/ring.rs:71-83` computes exact c+d*b+e*a+[party=0]*d*e;
  constant addition occurs at `mod.rs:652`. No post-product Q12 rescale.

### Shared vector DCF / provenance / erasure

* Backend `dcf.rs:23-49` delegates issuance, resource sizes and evaluation to
  `compact_dcf::VectorComparison`; no second PRG/traversal implementation remains.
* `compact_dcf.rs:76-94` implements coordinatewise product-ring operations;
  Convert uses disjoint 64-bit halves. At widths <=50 this retains <=100 bits
  from a 128-bit value seed, not two identical scalar coordinates.
* Generation `:274-314` and evaluation `:337-355` retain scalar sign conventions:
  generation corrections use current right control; evaluation negates party 1.
  Private lookup deltas and final base at `mod.rs:502-520` have matching signs.
* Roots/final words are Zeroizing; each correction Drop erases seed, payload and
  controls (`compact_dcf.rs:20-39`). Children, evolving seeds/controls, path value,
  threshold and payload use Zeroizing. AES output blocks are erased (`:119-121`)
  and workspace AES dependency enables `zeroize` (`Cargo.toml:33`). This is owned
  material erasure, not a claim about every compiler-generated stack/register copy.
* Private path issues every padded key, including zero threshold/delta; public
  path evaluates every public boundary even for zero delta. No sampled-value
  pruning or data-dependent shape was found.

### Freshness and pre-randomness admission

* `mod.rs:452-481` validates shape, context-index arithmetic and reserves the
  complete combined envelope before nonce or mask randomness. The frozen adapter
  also reserves descriptor ownership before construction (`compiled.rs:44-46`).
* Each lane samples a fresh independent input mask, FSS roots, base shares,
  mask-share split and Beaver correlation (`mod.rs:485-534`). Each issuance has
  fresh nonce. Context is public binding, never RNG seed.
* Shared ring issuance allocates/checks both public copies before DCF issuance
  (`ring.rs:241-281`); compact generation reserves/checks both correction owners
  before root entropy (`compact_dcf.rs:244-265`). No dynamic allocation follows
  first entropy request inside that shared scalar lookup helper.
* Do not conflate logical reservation with allocation of all future buffers:
  enclosing issuance samples nonce/mask before some fallible allocations, and
  online phases allocate within already-reserved capacity. Failure safely drops
  material; code does not promise allocation success after randomness begins.

### Lifetimes, failure burning, substitution

* `start`, `advance`, `finish` take state before validation/allocation
  (`mod.rs:597-641`). Wrong order, invalid domain, malformed frame, replay and
  allocation error burn the local owner. Cancel drops state. Partial issuance
  drops prior lanes/keys; the top-level lease refunds on error.
* Frame checks bind descriptor/source, widths/counts, plan/session/operation/index,
  role, phase and nonce (`mod.rs:458-478,582-595`). Public binding additionally
  hashes all original public table words and backend domain separator. Controlled
  cross-backend nonce test exists at `compiled.rs:259-275`.
* Independent Frame/Output leases survive Party and CompiledGelu destruction
  (`mod.rs:374-390,577-580,654-657`). Buffer fields precede lease fields, so refund
  occurs after buffer destruction. Consumed/cancelled parties conservatively
  retain their remaining lease until dropped.
* Local burning does not cancel the remote party or erase caller-made byte/share
  copies. Supervisor must cancel/drop peer and retained exports on transaction
  abort. Frames are context-bound but unauthenticated, as documented.

### Logical budget accounting

* Public path charges both independent table copies for every lane and actual
  private-key owner/correction capacity (`mod.rs:219-232`); `size_of::<Lane>()`
  contains embedded public RingParty owners. Shared helper accounts source table,
  both public copies, private capacities and issuance/simultaneous-evaluation
  scratch (`shared .../ring.rs:134-168,261-286`).
* Frozen owner keeps separate 1,310,720-byte descriptor lease; each concurrent
  issuance reserves its own compiler allowance, copied partition, all party keys,
  state/scratch, frames and outputs (`mod.rs:247-269`, `compiled.rs:97-119`).
  Retained source public table fits that descriptor lease: recorded frozen
  constructor upper is 574,108 bytes; compile upper 86,764, and additional public
  table payload is 12,296 plus its small owner. Constructor scratch is gone before
  this table copy. No need to raise allowance.
* Private key count is lanes*M; public private-key count is lanes, with lanes*M
  DCF evaluations per party. Public ring50 private logical bytes are 1,372/lane:
  1,324 DCF + 16 randomized base + 8 mask share + 24 triple. Public table copies
  are 12,296 bytes/lane/party and reported separately, not mislabeled peer traffic.
* Four-lane and 64 MiB caps remain (`mod.rs:35-36,74-76,156-167,253-255`). Claims
  remain logical layout/owner bounds, not allocator overhead, process RSS, global
  unrelated memory or generic parser admission. Exact-capacity assumptions are
  target/toolchain-qualified; shared scalar owners recheck actual capacities.
* Ordinary CompiledGelu accepts no caller descriptor. Generic JSON is not part of
  admitted serving path; its separate logical-owner limitation is not a new
  blocker for the frozen adapter.

## Complete protected-export gate

Source implementation correctly separates controlled public compiler-fixture
masks from fresh protected issuance. Callback reconstructs actual native results
and retires secret owners (`backend/conformance.rs:31-65`); it does not read
expected outputs or use the fixture mask. Helper validates eight GELU runs and
43,544 words, all contexts and cumulative private-work admission before callbacks
(`fusefss/conformance.rs:534-564`). Public-copy work is separately preflighted;
final material/copy/frame/time/zero-residency reconciliation is present in native
wrapper (`backend/conformance.rs:90-103,161-209`). Runtime success is intentionally
distinct from independent checker success.

Before claiming completed protected replay, require:

1. Successful current release test process, both complete receipts, 43,544 native
   words, eight runs, and matching JSONL byte count/SHA-256 across receipts.
2. Backend and compiler embedded-source hashes match the intended reviewed tree.
   Shared worktree edits after compilation must not be silently attributed to the
   binary that actually ran.
3. Work limits and zero final native Budget pass. Independently reconcile lane
   counts, frame bytes, private material and public-copy totals.
4. Separate `check_native.py --suite gelu` succeeds against completed output.

This review does not assert any of these runtime/checker gates passed. Source
hashes below match prior public-boundary review for the overlapping backend/shared
primitive files, but previous test evidence is not a new execution by this review.

## Reviewed source identity

SHA-256, captured during this review:

```text
9f8b51c58a1fbf789d6bd3c24771974c4fd03949e41cc8bc673a98b5f9da32f4  fusefss_backend/mod.rs
6ac2791b9141889c2fe6411b2b84ded5d08dd36734fee4716282f9ad35343d4a  fusefss_backend/compiled.rs
7c2621fab3cc7715fcd057399f248a9fc909db0d86e7f523bec0a6c2d0db159f  fusefss_backend/dcf.rs
b3a79dd9d04d86b98a549a5377d290b5e07fbfde81b526431298762879a0bb8a  fusefss_backend/conformance.rs
f4d2122be2e5d8be4a86cdb91f14defd1a297894b2b7c846bdc1d904731b8f99  compact_dcf.rs
a94762bfc2089c53c9e88faf8b509acbc2524b274861ef957619299539abd226  interval_fss.rs
60cc43e00d0c2f4d82ba7936967bd0f53ae7ac687e47ef5528b9d10d8c05d734  shared_affine_lookup_reference/ring.rs
83642c294839baea0fddda0b9213bfa8456945c8729295d03b74bd1321f21949  shared_wide_product/ring.rs
6529b05903cccba7e354315260d649ab10cd7c171a7d0dc6f1d67d5f17b0d673  fusefss/accounting.rs
0c6708f7e85987bafa3d3079288da005b545ee16d971114afbbdca87a253cf57  fusefss/fixture.rs
b498ed87ffc6ad4bfa1ea49b7905895c7cbb8fe3c608688c7dc4e0d8cee5631b  fusefss/compile.rs
9b2f0c12bb0b68dd83fb570fec0dfa84ac7688e0af74c5085b6730c9aa2bde08  fusefss/conformance.rs
```

Paths in identity block are relative to `crates/pllm-garble/src/`.
Adapted public-boundary backend remains distinct from the original paper's
optimized implementation. No whole-model, quality, transport or security claim.
