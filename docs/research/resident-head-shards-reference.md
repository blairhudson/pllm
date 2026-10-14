# Resident output-head shards: bounded synthetic reference

Investigation 7, `resident_head_shards`. Executable reference and installed-native
correctness controls only; no SDK registration, production edits, checkpoint
loads, builds, timing collection, tier qualification or speedup claim.

## Outcome

Immutable resident signed-i8 shards and native PagedGEMM fallback shards produce
the complete vocabulary output exactly. All 20 tests pass in the installed
`target/frontier-sdk-env` environment. Three tiny synthetic choices (paged,
partial resident, full resident) preserve identical integer and float32 logits
hashes, including original scale multiplication order and bias addition.
This is not certified argmax pruning: every output row is computed, even rows
that cannot win. Remote body placement is unchanged in this reference, but no
remote MAC fraction has been measured.

Native inspection **narrows the proposed benefit**: `paged.rs::decode()` reads
and optionally decompresses private file bytes; it does not cryptographically
authenticate each page on every execution. Actual-byte SHA-256 authentication
occurs at snapshot import, with decoded weight identity checked for exported
artifacts. Resident execution can avoid repeated page reads, decompression and
ephemeral Matrix construction. It cannot claim elimination of a repeated page
MAC/hash absent from the current implementation.

## Existing ownership and identity contracts

- `PagedGEMM.from_raw` authenticates actual caller-file bytes into an independent
  private snapshot. Subsequent caller-file mutations cannot change execution.
- `PagedGEMM._from_payload_executor` retains an already verified native PublicFile
  through `PagedMatrix.from_public_file`. Geometry and original weight digest
  must match. Independent handles share the native file owner; closing delivery
  or another matrix does not revoke remaining references. Legacy file payloads
  instead create a second snapshot. FilePayload validation hashes actual bytes,
  not a digest field alone.
- Exported PagedGEMM artifacts are authenticated as complete stored objects and
  also validated against their decoded signed-i8 weight hash during import.
  Nonresident execution retains these admitted, non-writable private snapshots;
  this probe adds no metadata-only authentication shortcut.
- `TransformerClient` resolves version-2 weight references once. Tied token/head
  stages may share exactly one matrix/scales pair. Resident production import
  already transfers tied metadata aliases together to immutable native views.
  Transposed token layouts use column gathering and original scale orientation.
- CompiledMatrix owns copied immutable native i8 bytes. Its `weight_view()` is a
  read-only alias that extends native lifetime. There is no explicit close API;
  retirement requires dropping every owner/view. Probe public lookup methods
  return independent copies and never expose native views. Private-attribute
  introspection is not a security boundary.

Inspected paths and their hashes, installed native-module hash and exact
synthetic identities appear in
[`../evidence/resident-head-shards-synthetic.json`](../evidence/resident-head-shards-synthetic.json).
Checkout-source hashes identify inspection inputs, not proof that an installed
wheel was built from those exact sources. Native execution provenance is the
separate installed `.so` hash.

## Reference mechanism and limits

`scripts/probe_resident_head_shards.py::ResidentHead` borrows an authenticated
original paged matrix while constructing a contiguous, gap-free partition. It
gathers each bounded shard, exports a separate raw/adaptive-zlib artifact, and
imports a native fallback. The concatenated actual gathered bytes must reproduce
the **original** weight digest. Per-shard object/weight hashes are additional
derived identities, not replacements for original object, weight or numeric
contract identity. The evidence's synthetic numeric metadata hash covers scales
and bias only; it is not a real manifest numeric-contract digest.

Selected shards are promoted deterministically in selection order if they fit
the resident payload cap. Every promotion re-hashes actual gathered bytes before
native compilation. Re-promotion is idempotent; over-cap requests reject before
gather/compile. Selection rejects duplicates and invalid indices. Weight bytes
are immutable; residency alone changes. Eviction drops the compiled owner and
falls back to the already-admitted paged matrix. Close is idempotent, retires all
resident/fallback owners, and rejects subsequent work. Construction failure
retires all completed native snapshots while leaving the borrowed source live.
Both close and construction failure clear resident matrices, drop the shared
resident executor and clear closed shard references, even if the ResidentHead
wrapper remains reachable. External private views/owners still extend their
own lifetime and must be charged independently.

Clear i8, wrap32 and prime modular operations dispatch each shard to its resident
or paged native matrix. Outputs are copied into original vocabulary positions.
Full integers are recombined before existing public `dequantize_matmul` and bias
operations. Row and column token gathers preserve repeated IDs and request order.
Tied synthetic head/token aliases reference one ResidentHead; untied tokens use
an independent authenticated paged matrix. Production stage wiring is not tested.

The native API has **no selective row-range product**. Consequently this probe
uses derived shard snapshots rather than pretending the original PagedGEMM can
skip selected pages. This creates explicit duplication:

| Owner/allocation | Tiny default footprint |
| --- | ---: |
| Original synthetic i8 head | 2,211 bytes |
| Derived stored artifacts, including five headers/indexes | 2,651 bytes |
| Derived private native fallback snapshots | 2,651 bytes |
| Resident i8 payload, paged / partial / full | 0 / 1,056 / 2,211 bytes |
| Reserved promotion weight workspace | 2,112 bytes |
| Fallback native metadata | 160 bytes |
| Largest fallback transient weight allowance | 1,056 bytes |

Original source and original native snapshot coexist at construction, then the
caller explicitly closes the borrowed original snapshot. Stored shard artifacts
and their private snapshots remain separate disk owners. Resident shards retain
fallback disk ownership for eviction. The untied lookup control temporarily adds
another 2,211-byte public file and private snapshot. Five fallback matrices each
own an executor; resident matrices share one additional executor. Object headers,
allocator retention, thread stacks, inputs, output/recombination buffers and OS
cache are not contained in the i8 payload budget. Temporary files survive owner
close until their enclosing fixture directory is cleaned. Reference is serial,
not concurrency-safe. Synthetic head is hard-limited to 8 MiB, shards to 4 MiB
and 4,096 rows, and complete output geometry to the existing bounded policy.
An independent **eight-fallback-owner maximum** is checked before source digest
access, native executor construction, export or gathering. Byte bounds alone
cannot bound file/executor counts when `shard_rows=1`. This count cap allows at
most eight fallback file/matrix/executor owners plus one shared resident executor;
the borrowed original and independent untied lookup remain separate owners.
Fake source geometries of 9×1 and 8,388,608×1 refuse without any native/source
allocation or exported files. The owner-count cap is a conservative tiny-reference
policy, not a whole-process memory admission model.

The four-tile promotion workspace allowance is an allocation estimate, not an
RSS limit or measured allocator bound. Whole-client admission must price all
owners and overlap separately. A Low few-MiB resident allowance is viable only
within **268,435,456-byte** whole-client peak. Medium full-head retention remains
conditional on **2,147,483,648-byte** peak, including token aliases, original and
derived snapshots, scales, outputs and startup. The tiny choices are not Low or
Medium qualifying clients.

## Evidence and interpretation

20 tests cover raw/zlib choices, clear/wrap32/modular full outputs, byte-exact
float32 scale/bias application, tied rows/columns, independent untied rows,
source mutation, immutable views, copied lookup outputs, cap/range enforcement,
eviction/re-promotion, closed owners, corrupted actual artifact bytes (including
attempted substitution of a newly computed object hash), failed-construction
cleanup, executor/shard-reference retirement, owner-count refusal before
allocation and rejection of the removed unguarded fresh-process option.
Each recorded choice checks 201 clear output elements, plus additional
full ring products and lookup controls.

Evidence was generated in **one process**, not a matched fresh-process cohort.
Process peak RSS is cumulative; recorded values cannot establish marginal
resident RAM cost. No CPU/elapsed timings are collected in the revised probe.
Original paging versus derived-shard overhead has not been benchmarked.
The `paged` choice is itself **sharded all-paged execution**, not ordinary
unsharded PagedGEMM. It is a correctness/residency ablation, not an installed
baseline performance control.

Logical execution payload counters cover products only, excluding token gathers,
import, metadata, source publication and OS effects. For five checked products,
paged/partial/full read-path payload counts are 11,055 / 6,831 / 2,211 bytes; the
last product deliberately runs after complete eviction. These counts model
requested stored payload, not physical disk reads, network bytes, authentication
work, RSS, host cache or all-link traffic. Full-head products still perform every
MAC. No client CPU, TTFT, N−1 throughput, Preparation cost or end-to-end benefit
follows from these arithmetic counts.

## Exact replay and parent-only guarded fresh processes

Small correctness replay; no checkout `PYTHONPATH`, native comes from installed
environment:

```sh
target/frontier-sdk-env/bin/python -m pytest -q tests/test_resident_head_shards_probe.py
target/frontier-sdk-env/bin/python scripts/probe_resident_head_shards.py \
  --output target/resident-head-shards-replay.json
```

Output path must be new. `--fresh-process` and its child-worker path were removed:
this probe contains no subprocess launcher, frontier lock, host admission,
watchdog or installed-build/source equivalence check. Source/native hashes are
provenance observations, not that equivalence check. It must not be used as an
unguarded process/timing cohort driver.

Optional fresh-process replay is **parent-only**. Before each serial launch,
the parent must acquire the existing frontier lock, pass existing conservative
host admission, activate the existing memory watchdog, and establish installed
build/source identity with the parent-controlled identity checks. Only then may
the parent launch this ordinary tiny correctness command in a new process:

```sh
target/frontier-sdk-env/bin/python scripts/probe_resident_head_shards.py \
  --compression zlib \
  --output target/resident-head-shards-fresh.json
```

One process, fixed 67×33 geometry, three correctness arms; no checkpoints,
timings or sustained loops. Parent should enforce a 60-second timeout and reserve
conservatively 256 MiB RAM for parent plus child and 1 MiB scratch disk; the
existing admission/watchdog may impose stricter limits. This estimate is not an
admission bypass. Each process peak remains cumulative across all three arms,
not an independent per-arm measurement. Future matched per-arm benchmarking
requires a separate reviewed parent-controlled harness; none is supplied here.
Each correctness arm compares complete outputs against independent NumPy int64
arithmetic and the process compares output/logit hashes across arms.

## Integration gates

1. Add reviewed native shared-snapshot row-range execution or shard descriptors
   in a separate parent-owned production change. Avoid derived full-head copies
   and per-shard executor proliferation. Authenticate original bytes before use;
   preserve source, original object/weight, bundle and numeric contract hashes.
2. Transfer tied token/head/auxiliary aliases atomically to one explicit owner;
   preserve row/column scale orientation and independent untied matrices. Audit
   outstanding metadata, buffer views, token owners, cache eviction and shutdown.
3. Admit full topology with startup/promotion overlap, every output, scales,
   resident/fallback owners and stacks; keep host headroom separate. Require
   maximum whole-client peak across matched fresh processes for each fixed tier.
4. Parent-only admitted real-artifact correctness and complete-response controls:
   identical full logits, tokens, KV and response contracts, original identities,
   unchanged remote body placement and independently ≥90% remote body MACs in
   prefill/decode. Synthetic algebra is insufficient.
5. Measure repeated decode CPU, TTFT, N−1 throughput, client/Preparation CPU/RSS,
   cold import/promotion cost and all-link traffic. Include ordinary installed
   **unsharded PagedGEMM** control; the current sharded all-paged arm cannot
   substitute for it. Charge full source import, derived shard export/import,
   native executor setup, resident promotion and duplicated fallback snapshots,
   including peak ownership overlap. Include OS-cache conditions,
   transfer/eviction ablations and matched fresh-process replicates.
   Publication/download bytes do not disappear merely
   because weights become resident. Promote only if measured benefit survives
   the fixed-tier and trust gates.
