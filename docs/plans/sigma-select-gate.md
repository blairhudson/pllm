# SIGMA direct Select gate

Status: implementation gate after consuming matrix handoff `920c48c1`.
This is bounded native/typed composition, not a decoder or remote protocol.

## Frozen source contract

Use the five-ring-word construction in
`mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`,
`GPU-MPC/ext/sytorch/ext/llama/select.cpp`, independently reimplemented in PLLM.
Upstream is a specification/oracle, never a runtime dependency or copied source.

For common masked inputs `x_hat = x + r_x` and `b_hat = b XOR beta`, issue
independent additive splits modulo `2^n` of:

```text
a = beta
B = r_x
c = beta*r_x + r_z
d1 = 2*beta
d2 = -2*beta*r_x
```

Sample `r_z` freshly across all output-ring bits. Holder `p` computes:

```text
z_p = -a_p*x_hat - B_p*b_hat + c_p
      + [b_hat=0]*(d1_p*x_hat + d2_p)
      + [p=1]*b_hat*x_hat
```

One consuming exchange reconstructs only `b*x+r_z`. This is direct
common-XOR-predicate selection, not an arithmetic-bit cast, FuseFSS B2A/Phi,
or the legacy four-word affine Select. Existing profiles and defaults stay put.

## Native implementation boundary

Add `masked_wire_reference/selection.rs` with separate party-local material,
dealer-only output masks, non-cloneable frames and masked outputs. Reuse shared
logical allocation/zeroization and the established masked-wire frame-binding
patterns. No transport, material codec or new decoder engine.

Admit 1–256 lanes, widths 3–64, and the existing aggregate 64 MiB logical ceiling.
Checked geometry, lengths, canonical predicate/ring residues, context and resource
admission precede allocation/randomness. Input-mask freshness is caller-owned;
the later typed adapter supplies its existing root provenance. Each invocation
has a fresh nonce and independent splits; exact RNG requests are `32+48*L` bytes.
No DPF or AES work is needed by this primitive. Logical key bits are `5*n*L`
per party, not serialized delivery bytes. One frame per party uses the existing
76-byte header pattern and `8*L` output-share bytes, with a separate namespace.

Price physical constants, dealer masks, borrowed operands/masks, all actual
working vectors, owner metadata/allocator allowances, retained frame/output
overlap and bounded workspace. Erase and deallocate before refund. Failure
consumes local state; explicitly cancel the sibling. No malicious-output or
persistent replay guarantee. Do not broaden the matrix singleton contract.

## Typed integration boundary

Add consuming typed selection over an existing `DealerMask<Predicate>` and
`DealerMask<Ring>` and their original role-bound capabilities. Keep separate
predicate and ring types; output fractional width equals the ring input's.
Derive only the conditional `0`/input interval hull, not a compiler proof.
Retain intact native backings and all their leases while borrowed; include typed
metadata/output overlap in aggregate admission. Do not implement fanout, cloning,
remasking, lookup, public roots or predicate reinterpretation in this gate.

Complete GELU/SiLU needs multiple internal uses of original inputs/predicates.
An activation owner or explicit fanout/remask design needs separate approval
before implementation. This gate does not authorize it.

## Normal validation

- Independent exhaustive small-ring masks, both predicate values/masks, signed
  values and output masks; wrapping ring64 endpoints and 256-lane geometry.
- Five-word construction/counters, exact RNG and full-ring output-mask coverage.
- Distinguish XOR-masked predicates from arithmetic-bit interpretations.
- Both input bindings, role, phase, malformed/noncanonical peer frames, retry,
  cancellation, entropy/allocation failures and exact retained-owner charges.
- Typed wrong source/mask/consumer/type controls and a consuming DReLU-to-Select
  chain using independent single-consumer ring roots; no repeated capability.
- Existing masked-wire/FuseFSS tests and library Clippy/compiler check unchanged.

Heavy builds run serially in the coordinating session. Commit a validated slice;
primitive checks alone do not qualify source activations or runnable SIGMA.
