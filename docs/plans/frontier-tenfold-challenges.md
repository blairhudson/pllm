# Five tenfold challenges for the frontier/probe design

Status: **research proposals and a design exercise**, 6 October 2026. None of
these constructions has a measured tenfold result or an installed SDK option.
Novelty means new concrete PLLM experiments, not literature-first algorithms.
The [frontier/probe proposal](benchmark-frontier-and-probes.md) supplies the
measurement and propagation machinery; these cases test its requirements.

## Fix the denominator first

Keep the current model, W8A8 numeric contract, provider-owned body, offline
Preparation and one online Inference provider. Exact state, masks, tickets,
source validation and cancellation semantics remain obligations of each candidate.

The [repeated prepared-prefill cohort](../research/prepared-prefill-reference.md)
is a historical anchor, not proof that its composition is today's best SDK stack.
Its Qwen2.5 150+8 cold/repeat results give these concrete stretch budgets:

| Same-scope metric | Historical anchor | Tenfold target |
| --- | ---: | ---: |
| Client lifetime peak RSS | 225.85 MB | at most 22.585 MB |
| Client cold-first CPU | 22.560 s | at most 2.256 s |
| Preparation cold-first CPU | 16.754 s | at most 1.6754 s |
| Preparation sampled cold-request peak RSS | 198.26 MB | at most 19.826 MB |
| Online cold TTFT | 41.306 s | at most 4.1306 s |
| Cold N−1 decode throughput | 3.686 tokens/s | at least 36.86 tokens/s |
| Both requests' all-link TCP streams | 832.326 MB | at most 83.2326 MB |

Every promotion reruns the incumbent and candidate together. These anchors do
not supply current-build controls. Each hypothesis names its own primary metric;
five component improvements cannot be multiplied into a whole-response result.
Long-context and restart workloads receive separate study cells and baselines.

## A. Frame-to-state arena compiler

**Target:** tenfold whole-client RSS, with client CPU a secondary stretch target.

Compile the client portion of a validated semantic schedule into a compact native
instruction tape plus a checked allocation plan. Colour tensor lifetimes into
reusable arena slots, including wire frames, mask expansion, residuals, numeric
scratch and KV. Receive a bounded frame into its assigned storage, validate it,
then consume it through unmask/dequantize and the existing local operation order.
Recycle the slot only after every consumer and verifier has finished.

The concrete advance over a generic Rust port is a compiler-checked **memory and
transport schedule**: no Python tensor round trips, duplicate graph dictionaries,
or independent receive/decoded/converted owners where their lifetimes can safely
share storage. The lightweight SDK binding owns one native execution handle.
The image binds source, numeric schedule, kernel ABI and resource bounds; an
unchecked serialized program cannot bypass ordinary model admission.

- **First probes:** process initialization and image import/binding, then one
  grouped gate/up stage through the arena. Price tokenizer, TLS, runtime mappings,
  paging and response state before extrapolating. Test malformed frames and
  cancellation while a buffer is in flight.
- **Falsifier:** the measured client floor plus unavoidable live owners exceeds
  22.585 MB. A smaller Rust worker does not pass if its Python parent still owns
  the old client footprint. Numeric reassociation also fails the exact contract.
- **Propagation:** every client-binding, stage-codec and decoder-phase case;
  prepared and verified paths need separate checks. Then cold/repeat whole-client
  confirmation, including startup allocations.

This specializes the earlier thin-client direction into an allocation algorithm
and an executable admission contract. It is not a claim that rewriting code in
Rust alone produces tenfold savings.

## B. Seed-to-correction stripe engine

**Target:** tenfold whole-Preparation RSS, including the serving runtime.

Make Preparation a bounded native transducer. For an admitted public stage it
reads a small authenticated weight stripe, expands the existing fresh mask stream
for a bounded row window, computes `W*r-s`, packs exact output residues and pushes
the completed stripe. It retires scratch after acknowledgement and advances the
stripe cursor. The full model, expanded mask inventory and complete correction
batch need not coexist in RAM.

Use a proposed chunk commitment/assembly contract at Inference: ordered stripes
bind inventory, stage, row range, output range and layout. Partial rows are never
reservable; READY still requires the complete acknowledged inventory. Failure
burns the attempt. Public weights may be reused; masks and correction rows may
not. Original seed domains and emitted arithmetic values must remain exact.

This extends existing paged Preparation beyond weight storage to **bounded live
state across generation, encoding, transport and acknowledgement**, with a native
role envelope small enough to test the approximately 20 MB target.

- **First probes:** empty-role floor, one wide stage with actual push/ack and
  backpressure, then increasing stage counts at a fixed stripe budget. Check exact
  corrections and incomplete/replayed/cancelled inventory rejection.
- **Falsifier:** runtime floor already exhausts the budget, or staging at either
  endpoint recreates an unbounded batch. Repeated weight reads and mask expansion
  may make its memory saving lose on Preparation CPU or setup time.
- **Propagation:** Preparation storage, inventory windows, framing and cancellation
  cases. Measure both endpoints and disk/I/O. The strongest interaction is B × C;
  fine stripes may destroy the matrix batching that C needs.

## C. Exact digit-lifted matrix multiplication

**Target:** tenfold Preparation CPU. A kernel-only win is an intermediate result.

Rewrite an unsigned ring input using balanced digits:

```text
r = sum_j 2^(d*j) * r_j                 modulo 2^w
W*r = sum_j 2^(d*j) * (W*r_j)           modulo 2^w
```

The signed-i8 weights and small signed digits are exactly representable in
float32. Select each inner-dimension block so that

```text
max_output_row sum_k abs(W[row,k]) * 2^(d-1) < 2^24.
```

Every product and every possible partial sum is then an exactly representable
integer under full IEEE float32 multiply/accumulate. This permits an optimized
dense float32 GEMM, followed by exact integer shifting and modular recombination.
For `abs(W) <= 127`, four-bit balanced digits permit an inner block of 16,513;
eight-bit digits permit 1,032. Actual row bounds may admit larger blocks.
Drop only multiples of the declared output ring. Uniform masks, seed domains,
corrections and numeric outputs stay unchanged.

The experiment asks whether a CPU matrix engine, such as an admitted full-float32
Accelerate path, beats hand-vectorized modular products despite multiple GEMMs.
Backend arithmetic must be explicit; approximate accumulators cannot inherit this
argument. Converted weight panels, digit extraction and recombination are charged.

- **First probes:** independent integer parity at worst-case signed bounds for
  16/24/32-bit rings; then compiler-derived QKV, gate/up and down shapes at
  1/16/150 rows. Time conversion, issuance, encoding and push separately.
- **Falsifier:** extra passes, conversions or small-row dispatch erase the benefit.
  If unaffected cold-start work exceeds 1.6754 CPU seconds, this construction alone
  cannot deliver tenfold cold-first Preparation CPU, even with free GEMM.
- **Propagation:** native kernel, paged/resident Preparation and public linear-stage
  cases. Confirm full cold-first CPU, elapsed time and RSS; label hardware engines
  explicitly. Do not infer an online TTFT improvement from offline CPU savings.

## D. Transactional layer-swapped KV

**Target:** tenfold client RSS in an admitted long-context cell, with exact full KV.

Introduce a client-local, encrypted, append-only KV store with one transaction
per completed decoder step. Before a layer executes, load only that layer's full
K/V slab into the original numeric layout. After use, commit its new rows to a
pending generation and retire the slab. Atomically publish the generation after
all layers succeed; failure retains the previous generation. Immutable old rows
are shared on disk, while pending rows, metadata and temporary owners are bounded.

Unlike eviction, every original row and absolute position remains present. Unlike
the earlier remote encrypted-KV screen, the store stays on the client device:
there is no provider KV fetch on every decode. Keeping the original full-layer
layout avoids introducing a new attention reduction tree as part of this slice.
Local plaintext buffers still retire; file pages use client-owned authenticated
encryption and an explicit key/owner lifetime.

Let `F` be all other peak-live memory, `L` the layer count, and `K` a layer's KV
payload. In the optimistic one-live-slab case, the ratio is
`(F + L*K) / (F + K)`. Tenfold requires `(L-10)*K >= 9*F`.
Actual attention scratch, crypto, transaction overlap and load-time peaks make the
gate stricter. Two live slabs instead require `(L-20)*K >= 9*F`.

- **First probes:** native geometry veto, synthetic exact state round-trip,
  one-layer attention, then teacher-forced decode with real prefill provenance.
  Inject failure between layer writes and at generation publication.
- **Falsifier:** the fixed floor or attention scratch prevents tenfold RSS; local
  read/decrypt cost regresses decode throughput beyond the predeclared margin.
  The short Qwen3 16+8 workload has only about 6.78 MB of KV and cannot earn a
  tenfold whole-client result from this method alone.
- **Propagation:** state admission, continuation, rollback, paging and decoder
  cases. A × D deserves an explicit interaction cell: reducing the client floor
  can make KV paging relevant, but separate ratios cannot be multiplied.

## E. Restart-persistent private continuation capsules

**Target:** tenfold request-start TTFT and all-link traffic for an explicitly
restart-heavy conversation lifecycle, including checkpoint writes and bootstrap.

At a successful turn boundary, encode a bounded exact continuation capsule and
seal it with a client-held key. Persist it locally or in an opaque blob store.
After a client restart, authenticate and restore it, rebind the compiled source
and numeric/state contracts, and prepare only the new suffix/decode work.
Use a different blob policy as a different candidate, with all transfer legs
charged. A public-prefix capsule cannot contain this private state.

The capsule contains eligible executed KV, positions, continuation metadata and
the pending-token distinction. It never contains reusable tickets, mask seeds
or reserved inventory. Each resumed execution receives fresh one-use material.
Client-held revision records bind the committed chain; an older remote blob
cannot silently replace it. Verifier provenance must survive restoration before
a verified composition can select this storage policy. Remote size/access
disclosure and padding require an explicit storage privacy contract.

The advance over in-memory prefix reuse is **authenticated state continuity
across process lifetimes**, not a cache hit on a still-running client. Compared
with per-decode remote KV paging, transfer happens at a chosen turn boundary and
restore, rather than once per layer on every generated token.

For an equal-cost illustrative sequence, let `B0` be common bootstrap bodies,
`E` additional initial capsule publication, `C` the control's bytes per resumed
turn, `R` all candidate bytes per resumed turn including restore, fresh execution
and checkpoint updates, and `H` the number of resumptions. Tenfold requires:

```text
(B0 + H*C) / (B0 + E + H*R) >= 10
H*(C - 10*R) >= 9*B0 + 10*E.
```

If `C <= 10*R`, no horizon rescues this layout. Real growing histories use the
sum of actual per-turn costs, not this constant-cost approximation.

- **First probes:** sealed state export/import, bounded parsing, exact continuation
  against uninterrupted execution, stale revision/substitution failure, then a
  real process restart. No full model download is needed to falsify the round-trip
  or transfer budget.
- **Falsifier:** restore and fresh-suffix costs fail the horizon inequality, source
  changes invalidate state, or required verifier/storage ownership is unsupported.
  The first bootstrap remains charged. This does not improve unrelated new prompts.
- **Propagation:** persistent state, key ownership, artifact transport and restarted
  conversation cases. Compare online TTFT and request-start TTFT separately; only
  the latter includes restore and fresh preparation.

## What this design exercise exposes

The runner/probe proposal needs the following explicit acceptance obligations:

1. **Predeclared budgets and cheap vetoes.** A hypothesis records metric, baseline,
   lifecycle, absolute/relative target and its necessary conditions. Geometry,
   runtime floors and bandwidth floors can produce `target_blocked` without a
   decoder run. A useful incremental result remains separate from meeting 10×.
2. **Backend and owner identity.** Build locks must cover native executables and
   linked numeric engines as well as the wheel. Client parents, native children,
   relays and setup belong to declared owners. For multiple client processes,
   measure a synchronized boundary RSS sum separately from each lifetime peak;
   never add unrelated peaks or hide a parent to obtain a win.
3. **State lifecycle probes.** Extend bounded decoder/state fixtures to initialize,
   export, retire, restart and restore. Allow authenticated caller-owned private
   state, never serialized benchmark prompts, keys or KV. Track key/nonce/revision
   ownership independently of benchmark cohort salts.
4. **Prerequisite and failure costs.** Record image compilation, panel conversion,
   file writes, restore, aborted issuance and eventual cleanup. Keep them separate
   from online windows but included in the lifecycle that incurred them.
5. **Prerequisite bundles and negative interactions.** A × D and B × C must remain
   selectable even if their isolated effects are inconclusive. New evidence creates
   retest requirements; it never installs a speedup on dependent compositions.

### Bounded first study

Run no-model floor/geometry gates for all five. Then prioritize C's exact stage
probe and B's pushed stripe probe, while A's client floor determines whether its
larger executor investment can meet the memory budget. D needs a separately
admitted long-context cell. E needs an explicit restart lifecycle and horizon.

For each surviving implementation, generate control/single-change cases plus
the A × D and B × C interactions. Confirm finalists on fresh, balanced matched
processes with all seven serving metrics. Keep rejected, unsupported and untested
cells visible. Test that a faster C kernel with unchanged cold CPU, a smaller D
KV owner with unchanged process RSS, and an E restore excluding its initial write
all fail whole-frontier promotion.

This is a requirements test of the SDK design. Executable probe adapters,
candidate implementations and measured confirmation remain future work.
