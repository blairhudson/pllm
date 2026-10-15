# Boundary scan fusion: actual test-only fused reference

## Result and limits

**Actual test-only fused traversal implemented and checked.** Parent authorized
a narrowly scoped `cfg(test)` child hook after initial public-API blocker audit.
Production public API/runtime unchanged; fused path not SDK-selectable.
No physical memory traffic reduction, production/security suitability, SDK
timing, or whole-response improvement established. New reference performs one
**source-level traversal**; no compiler assembly single-load assertion.

Artifacts:

- `scripts/probe_boundary_scan_fusion.py`: bounded baseline/test-fusion launcher,
  prebuild/binary-reuse modes and explicitly analytical material budget.
- `crates/pllm-garble/src/boundary_scan_fusion_probe.rs`: private test child using
  original DPF primitives, admission and owner, with fused DFS, joint ticket
  retirement tests and ignored JSON fixture.
- `crates/pllm-garble/src/private_pages.rs`: four-line `cfg(test)` child hook only.
- `crates/pllm-garble/examples/boundary_scan_fusion.rs`: existing packed128 DPF
  plus existing exact ring32 kernel, **separate calls**. No copied DPF/runtime,
  layout conversion, selection reconstruction, or substituted crypto domain.
- `tests/test_boundary_scan_fusion_probe.py`: accounting/CLI tests and opt-in
  native tail/role/replay smoke matrix.

Research reference only. Random packets/timings vary; dimensions, byte-level
oracles and assertions replay. No full checkpoint or SDK runs needed.

Validation: default Python suite 15 passed (native cases opt-in);
with `PLLM_BOUNDARY_SCAN_FUSION_NATIVE=1`, **23 passed**. Targeted Rust tests:
**3 passed, 1 ignored** (explicit fixture), including bounded ledger exhaustion
and rejection of non-matrix tables before work. Seven fused geometries cover
1x1 through 257x33; additional single 129x896 fused smoke passed. No heavy
benchmark run. Two isolated one-repetition smoke samples reported fused/separate
elapsed ratios 0.874 (129x19) and 0.842 (129x896). Neither sample was slower,
but shared-host microsecond samples are **not performance evidence** and do not
establish speedup. Parent must run serial larger bounded fixture before deciding.

## Original public-API blocker, resolved for tests only

`crates/pllm-garble/src/private_pages.rs` owns private `Key`, `Layout`,
`domain`, `leaf`, `stretch`, `fill_bits`, `Table::xor_record`, and
`Server::{admit,visit}`. Public `Server::evaluate` returns only final XOR reply;
`evaluate_batch` first fills V*batch private masks then traverses table. Neither
exports admitted packed terminal masks, scan callback, or bounded tile iterator.
`cancel` burns key but cannot yield its admitted state to another scan. These
remain public-API restrictions, not blockers for new private test child.

`Matrix::as_signed_slice` is public in `crates/pllm-core/src/kernels.rs`, but
that does not expose DPF masks. `Matrix::wrap32` performs its own traversal;
calling it from beside `evaluate` or `evaluate_batch` is still two traversals.
Re-creating keys against a different identity/basis table changes digest and DPF
domain. Replacing packed DPF with revealed token or random selectors would not
be faithful. No whole-module copy or mask exposure used: child uses `super::*`,
original `admit`, `domain`, `stretch`, `leaf`, `Key`, and `Arc<Table>`.

Binding owner is `crates/pllm-python/src/private_pages.rs`:
`Server::from_matrix` clones `Arc<Matrix>` from `crate::Matrix::inner`.
Native Python matrix implementation lives in `crates/pllm-python/src/lib.rs`.
`Server::from_matrix` preserves signed-i8 bytes and original little-endian scale
bits; descriptor equals explicit `[H weight bytes | 4 scale bytes]` records.
Bound contexts share `Arc<Table>` without another matrix or scale copy.

## What baseline and actual fusion check

One immutable `Arc<Matrix>` shared by two role-specific native table roots in
this **single-process test**. Deployment has separate processes/owners, not one
cross-worker Arc. Independent raw interleaved servers provide byte-exact XOR
reply oracle for **each party**, using identical production packed128 packets
and context digest. Complete recovered H+4 bytes compared to original record,
including every scale bit. This old example oracle intentionally adds storage
and scans. New fused fixture instead shares exact `Arc<Table>` with independent
original-evaluator ledgers (`Arc::ptr_eq` asserted), with no interleaved copy.
It compares each complete fused reply against both `evaluate_batch` and
`evaluate`, then recovers complete selected H+4 bytes.

For both role cases, fresh OS-random full-width u32 input compared at every
output coordinate between native SIMD-enabled `wrap32` and independent scalar
signed-i8 wrapping arithmetic. Also checks `Wr + W(q-r) == Wq` for every output.
These are two independent role tests, **not** proposal to send same random share
to both workers. Random head vectors never derive from DPF query IDs, roots or
correction words. Original issue API supplies separate DPF entropy.

Checks: packed leaves 127/128, incomplete leaves, odd H and SIMD tails, first and
last rows, negative i8 and full-ring wrap, wrong party/context, malformed packet,
query replay, explicit cancellation replay, duplicate context, response-binding
failure with consuming Rust decoder and burned server IDs. Native malformed
packet rejection precedes admission; it does **not** burn a valid query ID.
HTTP layer instead retires whole session on execution failure. New joint native
test admission validates packed namespace, exact input/matrix geometry, source,
party, query and independent head-ticket capacity/replay before output allocation.
It consumes original query ledger plus test-private head ticket synchronously
before arithmetic. No replay with changed head ticket, no fresh query with old
head ticket, and no replay after cancellation. Bad shape/malformed/wrong source
or party does not burn otherwise valid transaction. Injected failure after seven
rows charges 7H+28 source values/scale bytes and burns both identities; failure
after all V rows charges complete work, drops/zeroizes outputs and burns both.
This is synchronous fault injection, **not** live ASGI cancellation testing.
Returned successful buffers remain caller-owned; transport lifetime/wiping still
requires SDK integration. Native test ticket binds one admitted call, not an
authenticated session/token schedule contract.

Old example JSON reports setup and separate baseline lookup/head elapsed samples;
samples exclude independent oracle and extra retirement checks. Setup includes
test-only interleaved copies and reference owners; it is not production setup
cost. New fixture alternates fused-first/separate-first across iterations/roles,
reports summed paired elapsed samples, and excludes independent scalar oracle,
original-evaluate cross-check and additive recombination. Setup is separately
reported and includes source generation, matrix allocation, hashing, roots and
context setup. No fused query/mask preparation is precomputed off-clock.
Entropy/query issuance occurs before both samples. Small smoke times are not
benchmark evidence; results include test ticket preflight overhead on fused path.

## Candidate traversal, public tiles and scheduling

At token t, client knows token ID and can independently sample r_t before body.
Role receiving r_t can return its packed-XOR lookup share and W*r_t in one
operation; other role returns normal lookup share. After body quantizes q_t,
other role computes W*(q_t-r_t). Keep XOR lookup and additive head domains
separate. Original packet descriptor, packed128 PRG domain, scales and signed
weights must remain identical. No share conversion required or justified.

Implemented private test child admits query plus exact H-u32 random vector,
burns joint ticket, walks original tree in public increasing-row order,
produces one packed terminal word, and for each of its <=128 rows binds each
weight to one source local feeding masked XOR and wrapping dot in same iteration.
Scales feed XOR only. No secret-dependent skipped rows, mask branches, tile
boundaries, or work scheduling. Tile 128 or public multiples thereof; final row
count public. DFS state O(log(ceil(V/128))), terminal selection 16 bytes or at
most tile_rows unpacked bytes, lookup accumulator H+4, output V u32, input H u32.
No interleaved V*(H+4) copy needed. Actual test reference keeps 16-byte packed
terminal mask, not V-byte expanded masks. It does not materialize whole tree.
Counters record rows, source weight values and scale-byte visits; these are loop
accounting, not hardware load counters. Recursive seed/children/terminal word and
unfinished outputs use Zeroizing even on injected error. Compiler may still
spill or reload source values, so physical load count remains unverified.

Existing `Matrix::wrap32` has optimized private `dot_wrap32`; a scalar fused
reference may lose more SIMD throughput than it saves bandwidth. Merely calling
dot then XOR on same row is not proven one weight load: compiler may emit two
streams. Real implementation needs fused inner-loop inspection or hardware
counters, plus unchanged native kernel as exact oracle. ARM AES DPF expansion,
integer NEON dot/XOR, register pressure, and threading can contend. Existing ARM
AES/packed128 ~92x lookup gain is prior work, not evidence for this proposal.

Decode is best pairing opportunity. Prefill batches up to 64 lookup records per
scan but one final head: cannot charge one head saving per input record. Earliest
lookup batch may carry final head's r, but retaining W*r until final prefill head
lengthens ownership. Runahead can generate r and dispatch early, yet unknown
future token IDs prevent genuine future-token lookup. Paired lookup must wait
for W*r completion before body if API returns one joint response: could worsen
critical-path TTFT despite fewer logical reads. Streaming two results changes
failure semantics. Alternate roles only under explicit same-token identity;
never recycle r or a lookup ticket to make queues appear full.

## Material and work accounting (analytical, not instrumented)

Per role matrix VH bytes, immutable native scale copy 4V, plus retained caller
scales 4V if still live. Matrix constructor owns new allocation while source VH
bytes still exists; setup source+matrix = 2VH, excluding loader/checkpoint maps.
Descriptor hashing scans VH+4V and uses H-byte row scratch. Repeated owner setup,
source verification hashes and max-weight scan are real work, not amortized away.
Old single-process baseline adds original weights, interleaved oracle records,
two raw table copies and extra reference/recovery buffers; do not use its RSS as
deployment estimate. New fixture has one matrix, original source weights, caller
scales, two native scale copies (one per role), roots/contexts, reference ledgers,
and test expected/output/oracle arrays. It is still not a deployment RSS model.

For paired role alone: separate lookup+random head logical scan = 2VH+4V;
ideal fused lower bound VH+4V. **Across both roles per token** baseline 4VH+8V,
candidate 3VH+8V: complementary post-body head remains. At Qwen V151936,H896,
VH=136,134,656 bytes; ideal total logical scan saving ~24.94%, not 50%, and not
DRAM measurements. Network payload remains two lookup shares and two V-u32 head
shares; fusion may save framing/one interaction, not head payload.

Per active paired job: query `97+17*depth` bytes, decoded root/final/corrections
`32+17*depth` seed/control bytes plus descriptor/id metadata, H+4 accumulator,
`65+H+4` reply, 4H input, 4V output. Baseline single-query `evaluate_batch` allocates
V-byte mask; actual test fusion uses 16-byte terminal mask. Test preflight and
original `admit` both decode key, temporarily doubling seed/correction payload;
JSON charges that separately. Simultaneously live
accumulator/reply/output and serialization buffers must all be charged, not
counted as interchangeable. Client retains 4H r and 4V W*r per pending token;
J pending jobs cost J*(4H+4V), before arrays/transport duplication. This is
611,328 bytes per pending Qwen job (4,890,624 for eight).

Context commitment 32 bytes, query ID 16 bytes, bounded native ledgers up to
4096 IDs and 4096 contexts **plus** BTreeSet nodes/headers. Test fusion adds
independent 16-byte head-ticket ledger (up to 4096 entries), not query-derived r.
Arc, Vec, Server,
decoded-key metadata, recursive stack, executor, Python/HTTP objects, TLS and
allocator overhead remain explicit unknowns in JSON, not silently zero. Bounds
are payload accounting, not complete RAM or tier certification.

On cancellation after native dispatch, worker currently shields native thread
and holds busy slot until completion (`private_boundary_worker.py::native_job`).
Worst case discard entire fused VH+4V scan, dot arithmetic, DPF expansion and
reply+4V output; burn query, r ticket, result and session even if response lost.
Do not count saved complementary work as completed useful tokens. Cancellation
before dispatch still consumes reserved context/sequence/budgets per admission
policy, but may avoid scan; any partial work must remain charged.

## Candidate integration points (not implemented)

1. Promote validated private-pages-owned primitive only after further review;
   present implementation is `cfg(test)` only, not public native functionality.
   Do not expose raw masks to Python; preserve original domain/one-use identity.
   Native wrapper in `crates/pllm-python/src/private_pages.rs` detaches GIL.
2. `python/pllm/runtime/private_boundary_protocol.py::pack_frame/unpack_frame`
   currently accept only lookup/head; new operation needs exact two-input,
   two-output lengths, role rule and token ticket. Bind source/plan/session/epoch,
   token step, query ID and head operation; transport r bytes authenticated with
   same frame. Reserve both lookup and head quotas atomically before work; reject
   duplicate/mismatched ticket or changed-r retry without refund.
3. `private_boundary_worker.py::execute` currently dispatches either
   `child.evaluate_batch` or `head.wrap32`. Replace only paired branch after
   production native hook exists; preserve busy/cancel retirement and sequence.
4. Actual client seam is `private_boundary_transport.py::{token_lookup,linear}`,
   not generic `transport.py` relay. Mint fresh r during lookup, retain consumed
   token-scoped pending result, then linear sends only q-r to complementary role.
   Aborts/exception/failure wipe pending r/results; failed lookup cannot leave
   reusable speculative head state. Keep scale dequantization ordering unchanged.
5. Loader/compiler admission currently commits lookup/head calls and scratch;
   fused schedule/tickets and pending memory require independently derived contract
   before any SDK experiment. Only production-source edit is disabled-outside-test
   child declaration; runtime/public API and manifests unchanged.

## Replay and serial timing commands

Small correctness only (safe alongside workers):

```sh
target/frontier-sdk-env/bin/python -m pytest -q tests/test_boundary_scan_fusion_probe.py
PLLM_BOUNDARY_SCAN_FUSION_NATIVE=1 target/frontier-sdk-env/bin/python -m pytest -q tests/test_boundary_scan_fusion_probe.py
cargo test --offline --release -p pllm-garble --lib boundary_scan_fusion_probe -- --nocapture
target/frontier-sdk-env/bin/python scripts/probe_boundary_scan_fusion.py --fused
target/frontier-sdk-env/bin/python scripts/probe_boundary_scan_fusion.py --rows 151936 --columns 896 --pending 8
```

Parent: build once, retain exact discovered libtest binary. Build excluded from
subsequent direct fixture runs. No full-checkpoint or SDK timing supported:

```sh
FUSION_TEST_BINARY="$(target/frontier-sdk-env/bin/python scripts/probe_boundary_scan_fusion.py --build-fused | target/frontier-sdk-env/bin/python -c 'import json,sys; print(json.load(sys.stdin)["test_binary"])')"
```

After other workers idle, run **serially** (not run during this exploration):

```sh
PLLM_FUSION_ROWS=1025 PLLM_FUSION_COLUMNS=896 PLLM_FUSION_INDEX=1024 PLLM_FUSION_REPETITIONS=32 "$FUSION_TEST_BINARY" private_pages::boundary_scan_fusion_probe::fused_fixture --exact --ignored --nocapture --test-threads=1
PLLM_FUSION_ROWS=4096 PLLM_FUSION_COLUMNS=896 PLLM_FUSION_INDEX=4095 PLLM_FUSION_REPETITIONS=32 "$FUSION_TEST_BINARY" private_pages::boundary_scan_fusion_probe::fused_fixture --exact --ignored --nocapture --test-threads=1
```

These include verification work outside reported samples and timing interference
from oracle/cache warmup. Geometry capped at 4096 per dimension, 4 MiB weights;
not full-checkpoint evidence. Script equivalent can reuse binary via
`--fused --test-binary "$FUSION_TEST_BINARY"` and matching geometry flags.
JSON `fused_over_separate_elapsed_ratio > 1` explicitly means fused reference
slower; report negative results unchanged. No speed threshold in correctness
tests. Assembly inspection would be needed to upgrade source-traversal claim;
SDK timing still deferred until authenticated joint protocol integration.
Existing context:
`private-lookup-acceleration-reference.md` and
`private-boundary-tiers-reference.md`; their cold gains coexist with coupled
decode 1.924 TPS versus maintained 3.794 TPS. This probe does not improve those
results.
