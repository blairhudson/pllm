# Bounded native FuseFSS scalar compiler

This is a scalar cryptographic primitive in `pllm-garble`, not a decoder
activation, FSS backend, production-security claim, or complete method.

## Integration API

```rust,ignore
let descriptor = Descriptor::from_canonical_bytes(bytes)?;
let digest: [u8; 32] = descriptor.digest();
let shape: Shape = descriptor.shape();
let dealer_plan = descriptor.compile(mask)?;
dealer_plan.with_dealer_handoff(|input| {
    // Trusted dealer key generation ONLY:
    // input.shifted_boundaries: &[u64], including wide 0/N sentinels
    // input.coefficients: &[[u64; 2]], [constant, linear]
    // input.predicate_queries: &[PredicateQuery]
    // input.boolean_program: &[Instruction], split by shape.program_lengths
    // Return only protected party-local keys/shares from this boundary.
    Ok(())
})?;
```

`Descriptor::gelu()` constructs the frozen 32,900-byte descriptor from explicit
public table data. Its domain-separated digest is exactly
`5f1e888e7bdabe3cbad4b6bbf1abd6749c8dff0918c6f243916db071398e79d3`.
The original source-lock hash and every source-schedule field remain unchanged.

The D8 profile admits **only this exact frozen descriptor**: other gates or
modified metadata/pieces return an explicit identity/profile/Phi error. The
synthetic profile admits the complete bounded reference affine/Boolean language
(const, unsigned/low-bit/signed comparison, shifted MSB, NOT/XOR/AND/OR).
Only identity Phi and two affine coefficient slots are supported. No implicit
mixed-domain conversion, rescale, external tensor encoding, or guessed gate is
provided.

## Dealer boundary

`DealerPlan`, `DealerHandoff`, queries and instructions deliberately have no
`Debug`; plans have no `Clone` or serialization API. Mask-dependent stored
buffers and construction-error paths use `Zeroizing`. Production plans do not
retain the mask or source interval ordinals. Clear interpretation and the frozen
diagnostic-record encoder exist **only under `cfg(test)`**.

Handoff is a documented trusted-role boundary, not a runtime role-authentication
mechanism. Never disclose its shifted partition, coefficients, query thresholds,
or carry-bearing instruction operands to evaluators. Public descriptor
coefficients are public model data; their shifted ordering is mask dependent.
Backends must secret-share VM bit operands (including carries), issue each
comparison query even when its sampled threshold is zero, and protect interval
selection. Public query widths/shifts and instruction topology remain fixed.

Horner must use the original shared `x`, not opened `xhat`. The GELU linear
coefficient is Q0 and constant is Q12; there is **no post-product ARS12**.

## Admission and evidence

Canonical parsing rejects alternate key order, duplicate keys, whitespace,
floating/nonfinite values, integer aliases, noncanonical ASCII escapes, unknown
fields and wrong types. Parsing has bounded depth/value count and fallible vector
allocations. Source/Boolean/work/reference-byte caps match the frozen contract.
Compilation dry-counts normalized work before private allocation and preallocates
fixed descriptor-derived capacities; mask values never change public shape.

`Shape::reference_bytes` is the exact frozen **diagnostic** byte layout, not a
wire/key size, native heap/RSS cap, backend reservation, or traffic estimate.
The JSON DOM, native typed vectors, compile scratch, backend copies/keys and
issuance overlap still need complete issuer live-owner accounting. This module
does not modify or claim to discharge backend/runtime resource ceilings.

Tests cover frozen table/source/descriptor identities, all seven frozen GELU
record hashes, all 8,192 ring8 partition/mask/input cases, exhaustive small typed
predicate views, every GELU boundary neighbor and central signed codes, first
eligible split/full-domain partitions, maximum source count, exact AST cap,
malformed encodings/types, work limits and full-width ring products.

Run from repository root (module registration owned by coordinator):

```sh
CARGO_BUILD_JOBS=1 cargo test -p pllm-garble fusefss:: -- --test-threads=1
```

The missing evaluated descriptor manifest, signed38 graph provenance, other gate
descriptors/Phi, complete protected graph, source-model identity and production
security remain outside this compiler slice.

## Explicit independent ALL-suite export

Default tests never export diagnostics. Opt in with both the ignored test name
and environment option, from repository root:

```sh
PLLM_FUSEFSS_EXPORT_COMPILER_CONFORMANCE=1 CARGO_BUILD_JOBS=1 cargo test -p pllm-garble fusefss::conformance::export_compiler_conformance_all -- --ignored --exact --test-threads=1
python3 -B paper/reproduction-contracts/fusefss/native-conformance/check_native.py target/fusefss-native-conformance-v1/compiler.jsonl > target/fusefss-native-conformance-v1/compiler-check.json
```

`conformance.rs` is `cfg(test)` only. It identity-checks the frozen fixture and
every parsed native descriptor, streams actual compiled diagnostics and native
clear-interpreter results, and implements checked i128 finite-rational Q12
nearest-even encoding. No expected result field is read. JSONL has a 32 MiB
ceiling; the largest run streams individual outputs rather than building a whole
record. This is explicitly `compiler-clear`, not protected execution.

Target outputs are `compiler.jsonl`, `compiler-receipt.json` (source hashes,
counts and output identity), `compiler-owner-receipt.json` (capacity audit), and
the separately invoked read-only checker's `compiler-check.json`.

Observed independent ALL replay: **184 descriptors, 1,552 runs, 58,400 gates,
38 exact-rational encodings; checker pass**. Original frozen hashes still pass.

## Exact frozen GELU integration allowance

Use `Descriptor::gelu()` for locally constructed fixed GELU. For externally
supplied canonical bytes, use `Descriptor::from_frozen_gelu_bytes(bytes)`, which
checks exact length and frozen SHA-256 **before** allocating a JSON DOM. Generic
`from_canonical_bytes` cannot inherit this exact-input accounting claim.

On the audited aarch64 layout, the conservative logical-owner bounds are:

| Phase | Bytes |
| --- | ---: |
| Frozen GELU constructor, including canonical scratch, source bounds, JSON DOM/reallocation overlap and typed descriptor | 574,108 |
| Descriptor plus dealer compilation and all private scratch | 86,764 |
| Descriptor plus retained production plan and borrowed handoff | 74,516 |
| Retained descriptor heap alone | 61,524 |

Constructor accounting includes 262,144 bytes of canonical scratch, 4,096 source
boundary scratch, 229,536 DOM capacity bytes, one additional 16,384-byte maximum
DOM block for in-progress vector reallocation, and all typed descriptor owners.
DOM capacities only grow on this exact successful parse; final capacities plus
one largest block conservatively bound reallocation overlap. The audit overlaps
typed owners with that parser peak even though typed allocation starts later.

Thus the existing **1,310,720-byte** backend compiler allowance covers this exact
single-construction/single-plan compiler path, leaving **736,612 bytes**. It does
not cover unrelated retained descriptors/plans, backend-issued keys or callback
copies. `with_dealer_handoff` itself allocates nothing. The receipt conservatively
uses test-build plan metadata, which includes fields absent in production.

These are logical vector-layout/owner bytes, **not** allocator metadata, RSS or
thread-stack bounds. Tests assert observed capacities and record type layouts;
re-run the audit for another target/toolchain/layout. Generic descriptor parsing
has bounded bytes/depth/nodes and fallible allocations, but still lacks general
simultaneous owner reservation/accounting. No existing resource cap is increased.
