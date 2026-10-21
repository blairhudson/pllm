# SIGMA fixed activation prefix: ownership and numerical contract

## Scope and source order

This note specifies the explicitly approved bounded prefix, not full GELU/SiLU,
an SDK activation, or a new numerical profile. D16/D17 remain unchanged.

The CPU source schedule records `Y=floor(X/64)` **before** DReLU, absolute value,
or clamping. See [shared-profile-resolution.md §2.1](../../paper/reproduction-contracts/shared-profile-resolution.md#21-cpu-activation-and-softmax-constants)
for the pinned source and full activation equations. The source's complete
activation later selects **X** and subtracts a table correction. This prefix
instead selects **Y** and ends there; it does not implement that full equation.

| Edge | Contract |
| --- | --- |
| X | Signed n-bit Q12, `12 <= n <= 64`, `1 <= lanes <= 256`; common additive mask |
| Y | `floor(signed(X)/64)`, signed (n-6)-bit Q6; common additive mask |
| D | `[Y >= 0]`, including zero; common XOR mask, not an arithmetic bit |
| R | `D * Y = max(floor(signed(X)/64), 0)`; signed (n-6)-bit Q6 |

TR is reduced-ring truncation, not same-ring GapARS and not division toward zero.
Unsigned TR residues interpreted in the reduced signed ring give the specified
signed floor, including signed minimum and ring64 endpoints.

The exact Q12 template rejects n=9..11: existing `Numeric::new` requires
`fractional_bits <= ring_bits`. Do not relax that metadata rule to admit a
narrower input. At n=12, Y and R are signed six-bit Q6, which meets the same rule.

The native prefix has exactly four peer exchanges: TR phase one, TR phase two,
DReLU, then Select. Each exchange sends one frame in each direction, for eight
directed frames total. Each frame is `76 + 8*lanes` bytes; total directed prefix
traffic is `8*(76 + 8*lanes)` bytes, excluding any downstream consumer.

## Approved ownership only

The fixed owning region consumes each external X capability exactly once. It
retains the original Y backing for its predetermined DReLU read and subsequent
Select consumption, and consumes the internally produced D predicate. No Y
capability escapes, and no caller-visible split, payload clone, fanout, remasking,
or independent reconstruction from edge metadata is authorized. Existing generic
single-consumer arithmetic and Select contracts are not silently relaxed.

Each party has its own owner, material, frames, and role-bound input. The local
dealer retains masks; evaluators must never access them. The caller exchanges
peer frames and receives only an opaque final common-masked ring capability.
Cancellation, rejected frames, repeat calls, completion, and retained-frame
accounting remain obligations of the implementation's tests.

## Independent numerical tests versus public structural fixture

`crates/pllm-garble/tests/sigma_activation_prefix.rs` preserves seven independent
scalar specification tests. They use i128 modulus and signed-residue
arithmetic, frozen boundary expectations, literal ring64 endpoints, exhaustive
small rings, and every admitted width/lane count. These are oracle checks, **not
native numerical parity**.

The same file now includes three external public-API tests:

- `public_prefix_consumes_original_inputs_and_hands_outputs_to_drelu` covers
  `(input bits, lanes)` = `(12, 1)`, `(16, 12)`, `(32, 15)`, `(64, 256)`.
  It moves the original role-bound X capabilities into the region, requires
  three successive frame results followed by output on the fourth exchange,
  asserts all eight directional frame lengths and total prefix traffic against
  fixed constants, checks R descriptors and terminal owners, and passes the
  actual opaque R capabilities to the existing consuming DReLU API. It checks
  downstream predicate descriptors, not their hidden values. All eight prefix
  frames remain retained through completion and downstream consumption; budget
  usage must remain positive until those frames are dropped, then reach zero.
- `public_prefix_rejects_wrong_role_or_context_and_cancels_sibling` checks a
  wrong-role original input, an independently issued root with identical public
  metadata, and same-role frame reflection. Each rejection must return
  `Error::Context` and retire the failing owner; the caller explicitly cancels
  its sibling and checks terminal `Error::Consumed` and complete budget release.
- `public_prefix_cancel_keeps_retained_frames_charged_until_drop` cancels both
  parties after begin, repeats cancellation, checks terminal rejection, and
  checks that retained frames keep reservations live until dropped.

All ten external/oracle checks compile and pass in the ordinary native suite.
The public-API checks establish structure and lifecycle behavior, not hidden-value
parity. Region-private tests separately compare final masked outputs against
independent signed i128 floor/selection expectations. Ordinary validation commands
are:

```sh
cargo test --release --offline -j 1 -p pllm-core --lib edge_contract::
cargo test --release --offline -j 1 -p pllm-garble --test sigma_activation_prefix
cargo test --release --offline -j 1 -p pllm-garble
```

`ConsumerUse` declarations cannot authorize runtime reuse; the owning region's
private ledger must enforce every use. The fixture follows the typed-root and
peer-exchange patterns in `sigma_material_handoff.rs`, using the region's public
API:

```text
budget -> LocalDealer(source, plan, session)
prepare_root(X descriptor, independently declared BoundRange, region consumer)
    .encode(clear fixture) -> (dealer X mask, [party-zero X, party-one X])
PrefixSpec::new(operation, X descriptor, Y descriptor, D descriptor, R descriptor,
                next consumer)
dealer.activation_prefix(spec, X mask)
    -> (dealer R mask, [region-zero, region-one])
region-zero.begin(original party-zero X) -> local frame
region-one.begin(original party-one X) -> local frame
region.advance(peer_frame.as_bytes()) -> PrefixStep::{Frame, Output}
exchange through TR1 -> TR2 -> DReLU -> Select (four exchanges)
    -> [opaque party-zero R, opaque party-one R]
existing consuming dealer.drelu(next consumer, R mask, predicate descriptor, ...)
existing consuming ArithmeticParty::begin(original R), exchange -> predicates
drop all owners and frames; budget.used_bytes() == 0
```

The fixture declares the full signed n-bit input interval
`[-2^(n-1), 2^(n-1))`, independently of private samples. It does not decode final
values, inspect mask values, add getters, counterfeit capabilities, or substitute
independently encoded roots for internally retained Y. Expected phase counts
and frame lengths are constants, not derived from observed execution. The
external schedule checks frame/output timing; private tests must establish the
identity and numerical behavior of the internal stages.

Region-private unit tests compare final R against independently calculated i128
expectations without widening the public capability surface.
`native_drelu_predicate_is_true_at_zero_and_false_at_negative_floor_boundary`
separately checks actual native D before final selection, for X=-1 and every
X=0..63, both Boolean masks, both parties and widths 12/16/64. R=0 alone cannot
distinguish D=0 from D=1. Likewise, sign(X) equals sign(floor(X/64)); tests inspect
the fixed phase sequence rather than inferring TR-before-DReLU from final R.

Executed compile-fail cases live on the public region and shared capability
Rust types, rather than inert snippets in this Markdown note:

- Reusing X after `begin(X)` is a moved-value error.
- Cloning a region owner or `CommonMasked<Ring>` is unavailable.
- Constructing an intermediate Y or region permit from public metadata is
  unavailable; no intermediate capability accessor exists.
- Feeding a ring capability where a predicate is required fails type checking.
- Reading final masked payloads or dealer-mask values is unavailable publicly.

The complete garbling suite passes 490 unit tests, fifteen external/oracle checks
and seventeen doctests; eight opt-in exporters/measurements remain ignored.
The serial core library suite passes all 458 tests, including fifteen metadata
checks. Separate leakage controls pass 280 checks. Library-only core/garble Clippy
with warnings denied and the compiler library check pass in release/offline mode.
Filtered reruns overlap these totals; they are not additional distinct evidence.

## Explicitly not authorized by this prefix

Full activation reconstruction requires a later approved ownership design for
original X, absolute value/clamp, lookup, and correction subtraction. In
particular, retaining Y here does not authorize retaining or reconstructing X.
Signed-minimum absolute value cannot be represented as a positive value in the
same signed ring; this prefix needs no absolute value and grants no rule for
handling that later operation. No table lookup, widening rule, alternate floor,
numeric mapping, full-model schedule, or security qualification follows from
this bounded prefix.
