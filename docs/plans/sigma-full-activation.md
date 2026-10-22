# SIGMA bounded full activation owning region

Status: **APPROVED FOR BOUNDED LOCAL IMPLEMENTATION (D20)**. User explicitly
approved this complete region contract after D19 validation and commit
`0b489df3`. The bounded local region is implemented, with native, public integer-
oracle, lifetime and fault tests passing. This is not decoder/SDK qualification.
[D16–D19](../../paper/methods/DECISIONS.md#d16-distinct-cpu-port-of-sigmas-gpu-source-dpf)
authorize constituent work; D20 separately authorizes this composition. D18 covers only
[TR6 → DReLU → Select](sigma-activation-ownership.md); D19 covers the two
[native lookup boundaries](sigma-source-lookup.md).

## Fixed admission and numeric identity

- One fixed template, selected publicly as GELU or SiLU; 1–256 lanes and the
  existing aggregate 64 MiB logical live ceiling, not a process-RSS bound.
- External input: original ring50/Q12 owner, with encoded integer range
  `-2^37 <= X < 2^37` (Q12 real range `[-2^25, 2^25)`). Final output: ring50/Q12.
- Internal effective input: ring38/Q12; TR38 by six produces ring32/Q6.
  Broader width combinations are outside this first gate.
- Root clear encoder checks the encoded integer against the declared bound before
  entropy. Bind this as `LocalAssumption`, not compiler/checkpoint range proof.
  Imported protected inputs are unsupported absent matching bound provenance;
  masked payloads cannot establish a clear-value range.
- Retain D16 GPU127, D17 full-ring arithmetic masks and D19 correction algebra.
  No remasking, changed rounding, masked sign-extension or numerical substitution.

## Source schedule and evidence limit

Pin: `mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`.
Under `GPU-MPC/ext/sytorch/ext/llama/`, `api.cpp:2707–2758` and
`2784–2835` give GELU/SiLU; `2989–3036` gives clipping; `2121–2208` gives
lookup correction and deferred subtraction. TR/DReLU/absolute value/clip use
`bin`; lookup, final Select and reconstruction use global `bitlength`.

`GPU-MPC/ext/sytorch/include/sytorch/backend/llama_transformer.h:55–57,69–71`
passes `bitlength-scale` in mode1: 50/12 therefore implies bin38. Mode0 instead
uses full width and is the default. This is a source-supported mode1 schedule,
not evidence of an evaluated 50/12/mode1 invocation: inspected benchmark examples
include width51 GPT-2 and width48 Llama configurations. Preserve that distinction.
No upstream runtime is adopted or executed.

## Exact integer and masked schedule

Let `b=8` for GELU or `b=10` for SiLU, and `K=2^b`.

```text
Xhat38 = Xhat50 mod 2^38; r38 = r50 mod 2^38
Y = TR38(X,6)                         # floor(X/64), ring32/Q6
D = DReLU32(Y)                        # [Y >= 0], XOR-masked
P = reconstructed Select32(D,Y)
A = 2*P-Y mod 2^32                    # unsigned absolute-value residue
H = DReLU32(A-K)
E = NOT H
V = reconstructed Select_b(E,(A-K+1) mod 2^b)
J = V+K-1 mod 2^b                    # min(abs(Y),K-1)
L_p = deferred LUT_b_to_50(J)         # own share of T[J]+rho
F_p = deferred Select50(D,X_original) # own share of D*X+rF
O_p = F_p-L_p mod 2^50
Ohat = reconstruct(O_0,O_1)          # D*X-T[J]+rF-rho
```

Projection reduces both common masked values and corresponding dealer masks;
it does not recover or sign-extend a masked secret. Source clipping computes its
predicate in ring32 before projecting the Select arithmetic operand/mask to
ring b. Public offsets affect common masked values, not additive masks;
predicate complement flips the common masked bit while retaining its XOR mask.
Affine masks are derived exactly, including `rA=2*rP-rY mod 2^32`.

The admitted X range yields `Y in [-2^31,2^31-1]`. Its absolute-value residue A
has **unsigned** domain `[0,2^31]`, including Y's signed minimum. Consequently
`A-K in [-K,2^31-K]` fits signed32 for both K values. Do not label A signed-positive
or exclude that endpoint silently. This arithmetic argument depends on the
declared input bound; it does not prove a checkpoint satisfies the bound.

D19 lookup returns `L_0=2*z_0-R`, `L_1=2*z_1`: only role zero subtracts common R.
Final `F_p-L_p` subtracts each role's own lookup share on **both** roles. There is
no separate opening of L or F, and no final shift or rounding. Lookup mask
`rho=2*u-r` is marginally uniform but parity-dependent on internal r; r and u are
independent full-ring draws. Final Select supplies independent full-ring rF;
output mask is exactly `rF-rho`, without an added remask.

## Immutable tables

Use [shared-profile-resolution §2.1](../../paper/reproduction-contracts/shared-profile-resolution.md#21-cpu-activation-and-softmax-constants):

```text
GELU T[i] = floor(4096*(i/64)*(1-erf((i/64)/sqrt(2)))/2), i=0..255
SiLU T[i] = floor(4096*(i/64)/(1+exp(i/64))),             i=0..1023
```

SHA-256 over successive u64 little-endian entries:

- GELU: `b6a45a82dc8f660724943c1eb76654704c2e49563ffc0b1a0bc5388b3425a01b`
- SiLU: `ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f`

These are frozen paper-equation tables, not author compiler/libm byte identity.
Extract reusable method-neutral immutable table data/provenance from existing
owners where appropriate; do not copy or relabel FuseFSS execution/capabilities.

## Approved ownership authorization

- Consume each external capability once into separate dealer and role-local
  owners. Retain original wide X, mask backing, leases and matrix singleton guard
  through their respective last uses; never detach a guard or mint replacement X.
- Precommit a private ordered operand-use ledger: X twice (projection/TR path and
  final Select), Y three times (DReLU, Select, affine A), D twice, plus every
  projection, affine, clip, lookup and final-combination use. Burn attempted uses.
- Extract shared private ledger mechanics beside generic `ConsumerUse`/binding
  contracts. The existing prefix cannot be called then extended: its final
  transition retires X/Y/D. No arbitrary graph executor or exported internal view.
- Keep narrow projections private and bound to their original source/mask identity.
  Charge projection workspace; no cloning, exported fanout or payload-copy escape.
- Consume distinct deferred LUT/Select owners only with matching role, ring, lanes
  and full invocation bindings. Retain their backing/leases through transformation;
  final reconstruction creates the sole ordinary output capability.
- Bind source, numeric/table identity, template, every wire/use slot, session,
  issuance, shape and first tensor index. No cross-row secret inventory reuse.
  Any failure retires the local region; coordinator explicitly cancels its sibling.

## Exchanges and continuous admission

Eight exchanges: TR twice, DReLU, prefix Select, clip DReLU, clip Select, lookup
R/C, final reconstruction. Deferred Select evaluations add no exchanges.
Proposed final-frame domain: `pllm.sigma.activation-reconstruction.v1`, distinct
from constituent Select and lookup. Its 76-byte header contains eight-byte magic
`PLLMAC01`, 32-byte complete region binding, 32-byte common external-input digest,
one-byte sender role, one-byte final phase (8), and u16 little-endian lane count.
The common-input digest covers the original canonical ring50 masked input, never
private shares. Region binding includes source/template/tables, numeric/range
contract, complete wire/use identities, context, geometry and fresh issuance.
Payload contains L canonical u64 little-endian ring50 shares. Validate the complete
peer frame (exact length, opposite role, phase, identities and every residue)
before mutation; any error burns remaining local state. This codec is proposed,
not authenticated. With this codec, 16 directed frames total
`14*(76+8L)+2*(80+16L) = 1224+144L` bytes. This is proposed PLLM framing, not
upstream wire parity. Dealer issuance/material costs are separately charged.

Before entropy/allocation, validate the complete template/ranges/tables and price
all concurrent native and wrapper owners: original roots, masks, table sources and
two lookup copies, keys, ledger, projections, workspace, retained sends, borrowed
receives, deferred outputs, final frame/output and transition overlap. Reserve one
aggregate and split original leases into every native/nested owner. Extend reserved
entry points where needed; do not inherit prefix reserve/drop/reacquire behavior.
Backings erase/deallocate before lease refunds, including partial failures.

## Acceptance and scope boundary

- Independent wide-integer oracle: `D*X-T[min(abs(floor(X/64)),K-1)]`.
- Signed endpoints including minimum Y; negative bin boundaries; K-1/K/K+1;
  GELU X `[-1,63,64]` gives `[-31,63,33]`; forced mask 0/1/half/max and both bits.
- Frozen table hashes, wrong-table/profile rejection, exact eight-phase transcript
  and complete frame/material accounting; no hidden deferred-output opening.
- Wrong/reordered/repeated slots, roles, sessions and bindings; replay/failure burn;
  compile-fail cloning, projection escape and deferred/common capability confusion.
- L=1/256, exact-budget success and one-byte-short rejection, zero-draw preflight,
  allocation/entropy fault sweeps, retained-owner drop orders and guard retention.

User approval explicitly covers the full use ledger, fixed 38/50 range contract,
affine/projection ownership, deferred combination, final reconstruction codec and
continuous preadmitted leases. Approval is not implementation or validation evidence.
This gate supplies bounded local activation only: not runnable decoder/SDK support,
authenticated delivery, durable replay protection, compiler/checkpoint range proof
or composed security qualification.
