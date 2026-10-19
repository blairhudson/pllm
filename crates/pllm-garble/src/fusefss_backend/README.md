# Bounded arithmetic FuseFSS backend

Research implementation; not independently security/side-channel reviewed and
not a claim to reproduce the authors' evaluated backend or complete FuseFSS.
The frozen native descriptor is connected through `CompiledGelu`. The module
composes shared PLLM primitives; the additional bounded source-gate schedules
below retain separate conformance from complete decoder and SDK integration.

## Governing source and algorithm

Reviewed `papers/r14-fusefss.pdf`, arXiv:2606.09551v1, SHA-256
`6109399761ab0cc1ca6eb617fec1a811cbe59d474455ce3549467e5a5b77cbf0`.
Hash was checked against the source lock. Appendix I.2, PDF pp.20–21, specifies
private vector-payload interval lookup and mask-independent padded shape. It does
not mandate a particular lookup-key construction. Appendix N, pp.26–27, specifies
coefficient lookup followed by a Beaver product on the **original** shared input.
Appendix O explicitly makes concrete preprocessing costs backend-dependent.

For dealer-only translated boundaries `alpha` and payloads `v=[constant,linear]`:

```text
LUT(u) = v[M-1] + sum(i=1..M-1) (v[i-1]-v[i]) * [u < alpha[i]]
```

`dcf.rs` delegates to PLLM's shared `compact_dcf::VectorComparison` implementation
of the compact DCF equations (Boyle et al., ePrint 2020/1392, Figure 1). The common
group-aware core supports vector group `(Z/2^n)^2`, for `1 <= n <= 50`, while
preserving the legacy scalar recurrence and PRG. Both private delta components
enter DCF generation. Each seed
expansion produces separate child seeds, two vector values and control bits;
conversion maps 128 seed bits to two ring elements (at most 100 output bits).
PRG uses domain-separated AES-128 with the workspace's zeroizing AES feature.

With `LookupBackend::PrivateVector`, each evaluator receives randomized DCF root keys, correction words,
a uniform additive base share, a mask share and fresh independent Beaver triple
shares. It receives no table, boundaries, mask, coefficient deltas or selected
interval. Independent-key FSS privacy and additive base sharing give the I.2
lookup interface under the DCF/PRG assumptions; this is not a new security audit.
In particular, it is **not** the insecure construction that reveals private
coefficient deltas and multiplies them by ordinary comparison shares.

That control generates and evaluates exactly **M vector DCF keys per lane per party**:
the M-1 internal thresholds plus a zero-payload key at threshold zero. Padding,
duplicate coefficients and zero payloads never remove keys. This conservatively
prices one key for every padded interval, including zero thresholds. These keys
form one local lookup-interface call, not one comparison key for the whole table.

`LookupBackend::PublicBoundary` instead specializes the **original public** frozen
descriptor. Its immutable public table comes from the compiler's mask-zero
handoff, including the first eligible duplicate split. Duplicate payloads preserve
the whole-ring function. Actual secret input masks never select a public table.
One shared universal DCF per lane serves every original boundary; secret rotation
corrections are folded into freshly shared bases. Parties receive original public
boundaries and coefficients, their private key/base/mask shares and a fresh triple.
They never receive the sampled-mask-dependent partition/order, clear corrections,
input mask or selected clear coefficients. This is functionally equivalent to the
frozen Protocol-3 lookup, not identical translated-partition bytes or a general
private-payload lookup implementation.

Both choices reuse existing native families:

* `compact_dcf.rs`: common scalar/vector correction, AES and traversal machinery,
  with fallible allocation and explicit input/output rings.
* `interval_fss.rs`: ring-aware universal comparison and dealer-only correction.
* `shared_affine_lookup_reference/ring.rs`: consuming public-original-table lookup
  and explicit public-copy/private-key ownership.
* `shared_wide_product/ring.rs`: consuming ring1–64 Beaver shares. The existing
  standalone wide-product API delegates to the same algebra.

The frozen affine adapter supports no Boolean outputs and identity Phi only.
Separate predicate, conversion and lookup owners are described below; they do
not change that adapter's descriptor or automatically compose a complete gate.
Ring multiplication is exact modulo `2^n`, using low u64 product bits;
`n<=50` makes u64 wrapping followed by masking equivalent. Horner uses slope
times original input plus constant. No implicit scale shift is performed.

## Integration API

```text
Budget::new(limit <= 64 MiB)
CompiledGelu::new(&budget) -> frozen descriptor owner, private-vector control
CompiledGelu::with_backend(&budget, LookupBackend::PublicBoundary)
gate.resources(lanes) -> descriptor plus complete issuance admission
gate.issue(lanes, Context) -> [Party; 2]

// Lower-level trusted compiler interface, private-vector control only:
resources(Spec) -> public cost/admission report
issue(Spec, Context, &Budget, |lane, fresh_mask| -> Result<DealerPartition, Error>)
    -> [Party; 2]
DealerPartition::from_slices(fresh_mask, shifted_boundaries, [constant,linear] payloads)

party.start(own_input_shares) -> Frame
party.advance(peer_masked_frame.as_bytes()) -> Frame
party.finish(peer_product_frame.as_bytes()) -> Output
output.shares() -> borrowed additive shares
party.cancel()
```

`Spec` carries ring width, **source** interval count, lane count, descriptor/source
digests, Boolean output count and identity-Phi declaration. Padding is derived as
`M=min(source_intervals+1,2^n)`, with at most 512 source/513 padded intervals.
Callback is trusted dealer code, invoked once per lane. It must use the provided
fresh mask, preserve compiler/source identity, and erase its own artifacts. The
backend validates the complete strict partition and canonical payloads, copies
the bounded handoff into zeroizing dealer-only storage, and destroys it before
returning party owners. It does not independently prove compiler semantic parity.
Ordinary frozen-GELU integration uses `CompiledGelu`, which admits no caller JSON,
partition or self-asserted GELU label. Generic descriptor parsing has separate
logical-owner bounds and is outside this frozen adapter's admitted path.

`Context` is `crate::shared_rescale_reference::Context`. Public binding hashes
descriptor/source identity, ring, source count, lanes, plan/session/operation and
first tensor index. An independent 256-bit OS-random nonce identifies issuance.
Public-boundary binding also commits the backend namespace and complete public
table. A cross-backend frame rejects even when its nonce is deliberately matched.
Both phases use exact-length frames with ring, phase, peer role and lane count.
Noncanonical ring words, reflected roles, wrong context/nonce/order and replay
reject. **Context binding is not network authentication**; authenticated private
transport remains a separate integration gate. Security model remains trusted
dealer and at most one semi-honest, noncolluding online corruption.

## Lifecycle and logical allocation envelope

Parties, secret keys, frames and outputs are noncloneable and omit Debug. Secret
heap contents and retained masks/correlations use zeroizing owners or Drop.
Every online method takes state before validation/allocation, so malformed input,
bad frames, wrong phase, replay and allocation errors burn remaining material.
Cancellation drops state. Dealer callback/allocation failures drop partial keys.

`Budget` is shared across issuances and borrowed by all owners. Atomic preflight
reserves the **combined** envelope before randomness/compilation. Fallible exact
Vec reservations cover dynamically sized backend storage. Separate leases move
from parties into exported frames/outputs and survive party destruction; they
release only when the corresponding allocations drop. Party reservations remain
conservatively live until party destruction, even after completion/cancellation.
No hidden Arc/control-block heap allocation is needed.

The envelope includes both parties' actual Rust key/correction element sizes,
all online state buffers, both exported frames, output shares, stack/owner
allowances, the copied dealer partition, and **1.25 MiB** for compiler/descriptors
(frozen 1 MiB clear-instance plus 256 KiB canonical descriptor ceilings).
The trusted callback must fit **all** its retained descriptors, plans and scratch
within that overlap allowance; the backend cannot police arbitrary callback
allocations. Callers must separately admit transport copies or any copies they
make from borrowed output/frame slices. This is a logical allocation bound, not
allocator-overhead, process-RSS, OS-entropy-service or measured network accounting.
`CompiledGelu` retains a separate 1.25 MiB descriptor reservation, including its
original public table when selected. Each lane's two public-table copies belong
to the parties and add to issuance admission. Private logical key bytes, public
table bytes, actual native capacities and framed bodies are separate quantities.

Hard limit is four lanes, no automatic full-tensor execution. On the tested
64-bit target, maximum four-lane/513-padded envelope is below 10 MiB, under the
unchanged 64 MiB combined ceiling. A shared Budget prevents concurrent issuances
from independently claiming the same live allowance.

## Protected predicate and conversion substrate

The additional public native modules reuse the same `Budget`, `Lease`, `Frame`
and arithmetic `Output` owners, shared DCF, universal comparison and ring-product
implementations. They do not introduce a second cryptographic backend or transport.

* `predicates::{resources, issue}` takes an immutable typed descriptor and executes
  its actual fresh-mask compiler handoff. Comparison queries use one-bit DCF;
  shared `Bit` operands and fresh ring1 triples preserve the complete Boolean VM.
  `Party::{start, advance, cancel}` returns `Step::{Frame, Output}`. Final
  `BooleanOutput::shares()` are XOR shares. The descriptor and generic parser
  remain separately admitted caller owners. Public instruction shapes, query
  widths and view shifts are independent of the secret mask.
* `conversions::{Spec, resources, issue}` admits exact reduced-ring TR, same-ring
  ARS, zero/sign extension, B2A, arithmetic-bit Phi selection plus a public add,
  and an ordinary modular `Multiply`. `start`, `start_phi` and `start_product`
  enforce distinct method contracts, followed by consuming `finish` or `cancel`.
  Widths are 1–50, with at most four lanes. A product uses one fresh existing
  ring triple per lane. ARS/TR use private carry comparisons, preserving signed
  floor rather than sharewise shifts or nearest rounding. Phi consumes an
  arithmetic bit; an XOR output first needs a separate fresh B2A issuance.
* `lookup::Table::new_zero_extended` binds the original public table, ring,
  source commitment and fixed 256-entry chunk geometry. Each chunk is zero
  outside its public slice. Every party evaluates **all** chunks, with independent
  fresh masks and DCF owners for every chunk/lane, then sums opaque output shares.
  The existing 513-interval primitive ceiling stays unchanged. The composition
  admits at most 131,072 entries under the same four-lane/64 MiB ceiling; larger
  public compositions reject. Each chunk contains at most 2 KiB of original u64
  values, but boundary/coefficient copies and all other owners cost more and are
  separately charged. Index and payload both remain ring50 for the source norm
  and reciprocal tables, whose payloads exceed their index bit widths.

`SignedRange::source_contract` and Phi's `bit_contract` are caller provenance
commitments, not proof verifiers or private membership checks. Lookup returns zero
outside its table and does not prove source-index validity. A model compiler must
establish these obligations before selecting any of these owners. The bounded
source-gate schedules below provide conditional per-edge scale and no-wrap checks;
lookup and primitive parity alone do not supply those schedules.

Aggregate reservation precedes entropy. Every online call takes state before
validation. Exported buffers retain leases after party retirement; fixed typed
allocations reject allocator capacity over-grants before receiving secrets.
Nested allocation-failure injection remains partial, although the allocation
paths themselves are fallible. Counts distinguish DCF-only keys, complete private
material, original public tables, duplicated public tables, native reservation
peaks and actual peer-frame lengths. Cumulative material is not live memory.

Frozen source contracts and independent checkers are under
`paper/reproduction-contracts/fusefss/gate-set/`. The source fixture contains 275
cases; explicit selected-gate replay covers 66 (60 protected outputs and six
local exact-rational external encodings). A separate public-table replay covers
18,560 entries across eight tables. Neither accepts a complete-gate claim from
partial output. Evidence belongs to
`paper/artifacts/fusefss-gates/local-v1/`, separate from the historical GELU slice.
Its accepted fresh attempt passes 239 native library tests, 66 Python checks,
both actual native exporters and both independent output checkers. The archived
selected cases and table outputs replay independently, and 31 archive mutations
reject. The eight table replays use 4,640 fresh batches and return their final
reservations to zero; cumulative costs and logical peaks stay separately reported.

## Bounded source-gate schedules

`composition` lowers fixed public contracts into immutable typed instruction
plans. `gates::{silu, nexp, reciprocal, softmax}` return a `Plan` and output layout;
`norm::{Spec, Kind, compile}` selects LayerNorm or RMSNorm with public affine
parameters. Native arithmetic retains the approved ring50/Q12 profile:

* SiLU narrows only under the caller's signed38 promise, computes the signed
  floor/index privately, and applies Phi to the **original** shared input.
* nExp evaluates both original public tables, multiplies fresh shares and floors
  the complete product. Softmax privately selects the active-row maximum, checks
  public subtraction bounds, saturates before lookup and floors the complete
  denominator once. Public inactive entries stay zero.
* Normalization uses 50 fixed public exponent bins, private comparison and
  selection, unsigned TR with **zero-extension**, and the original rsqrt table.
  Centering, square sums, both affine products and their **separate** floors have
  checked-i128 range obligations before material issuance. The source's Q12
  reciprocal mean is retained, including its nonzero constant-row result at 768.

Table constants replay from the frozen independent fixture. A separate exact
integer `rsqrt-k2` supplement covers its already-existing width-two norm case;
the original fixture and source locks are unchanged. Comparison descriptors use
a bounded typed constructor and stack canonical encoder, avoiding generic JSON
parser allocation in the composed path.

Input ranges are fixed public assumptions; canonical party shares cannot prove
membership of their reconstructed secret. A model compiler must establish those
assumptions. Plan identities bind ranges, mask, shape, source, actual table bytes,
affine parameters, register types and instruction order. They do not authenticate
a publisher or establish whole-model ranges.

`plan.runtime(role, context, inputs)` owns exactly one party's registers. `poll`
returns `NeedMaterial { pc }`, `Frame` or `Output`. The external trusted dealer
calls `plan.issue_step(pc, context)`; each role consumes its own material through
`feed` and peer bytes through `advance`. No evaluator internally invokes issuance
or reconstruction. Wrong role/plan/context/step, replay, failed feed, malformed
frames, allocation failure and cancellation burn that runtime. Native frame and
output leases survive runtime and plan retirement.

Plans retain at most 32,768 instructions, 65,536 registers, 256 tables, 16 MiB of
plan ownership and `2^26` cumulative work units. The combined native ceiling stays
64 MiB, including two runtime owners, the largest primitive and previous-frame
overlap. Construction probes current headroom; runtime construction and every
issuance recheck reservations. The probe is not a future concurrency guarantee.
Lookup owns all chunks of one invocation; only successive primitives stream.
The table formula permits width 4,096, but complete LayerNorm and RMSNorm at that
width exceed the unchanged instruction cap and reject before protected issuance.

Resources distinguish original table owners, cumulative duplicated public tables,
private material, keys/evaluations, triples, actual native frames and planned live
envelopes. Sampled reservations can miss issuance scratch and are not a true
high-water mark or process RSS. The opt-in full exporter runs all 275 frozen cases
through native owners, with six external encodings remaining local public work.
Receipt-backed evidence and complete replay commands live in
`paper/artifacts/fusefss-composition/local-v1/`. This is bounded protected-gate
conformance, with model lowering, authenticated roles and SDK qualification
retaining their own gates.

## Historical scalar evidence

Command, repository root:

```sh
CARGO_BUILD_JOBS=1 cargo test -p pllm-garble fusefss_backend -- --test-threads=1 --nocapture
```

The initial scalar backend's focused suite passed 14 tests. Logical layout counts
below are checked by native controls; neither key codec nor wire measurement is
implied. Both frozen-GELU choices use 511 source / 512 padded intervals.

| Ring50, one lane | Private vector | Public boundary |
| --- | ---: | ---: |
| DCF keys per party | 512 | 1 |
| Private material logical bytes per party | 886,832 | 1,372 |
| Public table copy bytes per party | 0 | 12,296 |
| DCF evaluations per party | 512 | 512 |
| DCF path levels per party | 25,600 | 25,600 |
| Masked frame bytes, each direction | 88 | 88 |
| Horner-product frame bytes, each direction | 96 | 96 |
| Total peer frame bytes, both phases/directions | 368 | 368 |
| Exported output bytes, each party | 8 | 8 |

One vector DCF key is `32+34*n` logical bytes, hence 1,732 bytes at ring50.
Private payload count adds 8 mask-share bytes, 16 base-share bytes and 24 triple
bytes per lane. Each party performs `M*n` DCF levels, five AES blocks per level;
dealer generation performs two expansions per level. One fresh Beaver product
per lane, two peer-exchange phases, no Boolean gates/conversions or rescale.
These are operation counts, not measured setup/online performance.

The independent public-boundary engineering review additionally checks actual
allocation failures and requested heap through the public API. Its one/four-lane
issuance/online peaks are 102,236/187,004 bytes against combined reservations of
2,679,510/2,764,902 bytes, including the separately reserved descriptor. Constructor
requested-heap peak is 557,300 bytes. These counts exclude allocator overhead,
stack and process RSS. All reviewed heap and Budget owners retire. They do not
establish whole-client memory or independent cryptographic review.

Tests cover independent exhaustive rings1–4 over every mask/input, full-domain
partitions, vector DCF thresholds/payloads exhaustively through ring5 and endpoint
cases at every width through ring50, ring50 endpoints/wrap, constant/slope privacy
structure, original-x/no-extra-rescale controls, 512/513 key shapes, malformed
frames/roles/context/nonce/order, replay/cancel burns, allocation failure at all
online allocation sites and issuance failure sites, preflight rejection before
callback, and leases surviving exported frame/output lifetimes.

Remaining gates: independent cryptographic/side-channel review; real role separation and network
authentication; complete source nonlinear/Phi composition; signed38 provenance
and full-model feasibility; SDK and model qualification. Scalar backend success
does not discharge any complete-method claim.
