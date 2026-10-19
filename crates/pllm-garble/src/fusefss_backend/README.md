# Bounded arithmetic FuseFSS backend

Research implementation; not independently security/side-channel reviewed and
not a claim to reproduce the authors' evaluated backend or complete FuseFSS.
The frozen native descriptor is connected through `CompiledGelu`. The module
composes shared PLLM primitives; complete gate, model and SDK integration remain
separate completion gates.

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

The initial backend supports affine arithmetic, no Boolean outputs, identity Phi
only. Ring multiplication is exact modulo `2^n`, using low u64 product bits;
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

## Observed bounded evidence

Command, repository root:

```sh
CARGO_BUILD_JOBS=1 cargo test -p pllm-garble fusefss_backend -- --test-threads=1 --nocapture
```

The integrated backend's focused suite passes 14 tests. Logical layout counts
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
authentication; complete source gate/Boolean/Phi coverage; signed38 provenance
and full-model feasibility; SDK and model qualification. Scalar backend success
does not discharge any complete-method claim.
