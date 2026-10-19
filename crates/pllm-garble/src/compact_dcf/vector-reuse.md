# Shared vector DCF reuse

## Scope and construction

The shared `compact_dcf` implementation now uses a single coordinatewise group
interface for u32, u64 and `[u64;2]`. All three use the same `Key`, `Correction`,
AES expansion, Figure-1 generation and traversal. The legacy scalar PRG domain
`PLLM.DCF.PRG.v1\0` and scalar low-word conversion remain unchanged. Existing
`Comparison` and `RingComparison` APIs and resource formulas remain unchanged.

Vector conversion retains each half of the 128-bit value seed, masking each to
the requested output ring (at most 100 total payload bits). A vector key has one
root, one correction path, two value coordinates and one control path. It is
**one vector DCF key, not two scalar keys**. Private coefficients enter the
payload corrections; neither clear payloads nor thresholds remain in key owners.
No second backend-specific AES/DCF algorithm remains in `fusefss_backend/dcf.rs`.

The generic control's PRG identity changes from `PLLM.FUSE.DCF.1\0` to the shared
domain. No descriptor, interval map, numerical value or old shared scalar key
identity is intentionally changed. Fresh independent roots remain required for
every issuance. This does not certify an independently reviewed cryptographic
implementation or restore full-model feasibility; it is bounded generic control.

## Exact shared API

```rust,ignore
crate::compact_dcf::VectorComparison::resources(input_bits, output_bits)
    -> Result<Resources, shared_rescale_reference::Error>

VectorComparison::issue_with_reservation(
    input_bits: u8, output_bits: u8,
    threshold: u64, payload: [u64;2], allocation_limit: usize,
    before_reserve: impl FnMut(usize) -> Result<(), shared_rescale_reference::Error>,
) -> Result<[VectorComparison;2], shared_rescale_reference::Error>

key.eval(party: u8, input: u64)
    -> Result<[u64;2], shared_rescale_reference::Error>
key.allocation() -> Resources
```

Widths1..50, canonical threshold/payload and total resource limit are checked
before allocation hooks or RNG. Both correction vectors reserve fallibly, with
one callback per requested vector capacity; actual capacities are rechecked
against the combined limit before either root is sampled. Callback/allocator/RNG
failure destroys partial keys. Evaluation checks party0/1 and canonical input.
Keys/corrections are noncloneable, omit Debug, and erase secret contents on Drop.
No raw key codec or deterministic-entropy API is exported.

The callback does not create/hold a lifetime lease. The enclosing existing
backend Party must retain its combined allocation reservation and its existing
frame/output leases. Keys contain no borrow of the callback, Budget or dealer
partition. This is shared FSS algebra, not a new online lifecycle or transport.

## Thin backend adapter and parent integration

`fusefss_backend/dcf.rs::Key` is a transparent one-field adapter. Existing issue
signature remains `issue(bits,threshold,[constant_delta,linear_delta],&Budget)`.
It obtains shared costs, supplies the exact primitive allocation limit and calls
`Budget::allocation` once before each shared correction-vector reservation.
Injected allocation errors preserve backend `Error::Allocation`; other shared
errors pass through parent `primitive_error`. Shared allocator/capacity errors
remain the shared primitive's `Resource` error.

Parent-owned `backend/mod.rs` needs these replacements (not applied here):

```rust,ignore
// Inside the private-vector resource branch:
let cost = dcf::Key::resources(spec.bits)?;
let party_keys = lanes * (
    m * (cost.key_owner_bytes + cost.key_capacity_bytes)
    + size_of::<Lane>()
);
// Keep existing private payload count and public frame sizes.
// Existing resource tuple's issue_scratch entry can use:
let issue_scratch = cost.issuance_scratch_bytes;
// Existing 4096-byte per-party fixed allowance still covers evaluation scratch.

// In Lookup::Private local evaluation:
let value = Zeroizing::new(key.eval(party, opened)?);
// Old key.eval(party, opened, mask) no longer applies: widths are bound in key.
```

There is intentionally no exported `dcf::Correction` layout. The shared
`Resources` record supplies encoded-key, owner, actual/requested capacity and
issuance-scratch counts; callers must not reconstruct shared private layouts.

## Static counts

For input width n, one vector key per party:

* Logical private payload: `32 + 34*n` bytes (root16, final vector16, then
  seed16 + vector16 + controls2 per bit).
* On this 64-bit target: owner64 bytes including public width metadata;
  correction capacity `40*n` bytes. Ring50: **1,732 logical bytes**, **64 owner
  bytes**, **2,000 correction-capacity bytes**, **2,064 retained owner+capacity**.
* Compared with previous backend-local vector owner56, retained storage increases
  by eight bytes per key per party. At512 keys: +4,096 bytes per party.
* Dealer pair: two independent 128-bit roots, two expansions/bit, five AES blocks
  per expansion. Ring50: **500 AES block encryptions per key pair**.
* Each evaluator: one expansion/bit, five AES blocks/expansion. Ring50:
  **250 AES block encryptions per vector key**, not twice that count.

512-padded generic lookup still issues/evaluates512 vector keys per lane/party,
including its zero-threshold dummy, and retains886,832 logical private bytes per
party/lane including base/mask/triple payloads. Scalar shared ring50 remains
`24+26*50 = 1,324` logical bytes/key. No new online round, frame word or rescale.

## Validation status

Before refactor, a single focused native test captured SHA-256 identities over
deterministic public fixture roots, every correction word/control and final word
for legacy u32 and ring50 u64 scalar keys. That baseline test passed; its exact
fingerprints are now assertions in `group_tests::frozen_scalar_key_identities`.
These deliberately insecure fixture roots are test-only.

Added tests cover exhaustive small vector-input domains with independently
varying output widths, zero/private payloads, first-coordinate equality to
existing scalar keys/shares under identical fixture roots, ring50 endpoints,
static/actual resource counts, invalid widths/payloads/limits before callbacks or
entropy, both allocation-hook failures before entropy, both entropy failures,
and checked evaluation. Adapter tests exercise both existing Budget failpoints.

Post-refactor command attempted:

```sh
CARGO_BUILD_JOBS=1 cargo test -p pllm-garble compact_dcf -- --test-threads=1 --nocapture
```

It was blocked by parent-owned stale correction sizing and old eval signature,
plus separate `backend/compiled.rs` compiler-versus-primitive error adaptation.
No diagnostics were emitted for owned implementation/test files. Parent must
apply integration above and fix its compiled adapter, then run focused shared
primitive and backend tests. No passing post-refactor run is claimed here.
