# SIGMA protected normalization and causal softmax audit

Current status, 24 October 2026: **D29 CONDITIONALLY APPROVED; BOUNDED nExp
IMPLEMENTATION COMMITTED; BOUNDED SOFTMAX ROOT CONDITIONALLY APPROVED IN D36**.
The [nExp packet](../plans/sigma-nexp.md) received conditional local approval in
[D29](../../paper/methods/DECISIONS.md#d29-conditional-bounded-sigma-nexp-implementation).
Its private owning region landed in `ec68ad19`; Parent reported **755 / 43 / 35
component checks passed**. The guarded public clear-root facade landed in
`d6b6e910` with **95 checks passed**. Those results are not full softmax,
normalization, checkpoint, decoder, SDK or security qualification. The inherited
nExp entry needed below is still closed; clear-root component tests do not supply it.

The original read-only audit, recorded 22 October, is retained below, including
its then-unapproved recommendations and paper/CPU conflicts. Its statements
about missing implementations describe that date, not the current D29 status.
No implementation code was changed; no tests, benchmarks or native/model imports
or execution accompanied this update. D17/D19/D20/D22/D27/D29 do not authorize
the proposed outer composition,
broader numeric domains or decoder availability.

## Source identity and locations

Paper: [SIGMA, PoPETs 2024(4)](https://petsymposium.org/popets/2024/popets-2024-0107.php),
cached at `papers/r13-sigma.pdf`; repository-recorded SHA-256
`f911b715b0e0523fa4d77f21c28c6a0440c5be781e67b5b30419f20e505a6ebf`.
Page references below are one-based PDF pages, not printed page numbers.

All author paths below are relative to
[`mpc-msri/EzPC@aa0f2f860266fb79901405ef5e9965b8c9fe2e88`](https://github.com/mpc-msri/EzPC/tree/aa0f2f860266fb79901405ef5e9965b8c9fe2e88).
An immutable source URL is
`https://raw.githubusercontent.com/mpc-msri/EzPC/aa0f2f860266fb79901405ef5e9965b8c9fe2e88/{path}`.
These are specification sources, not adopted runtime dependencies or an
evaluated-build lock.

| Contract | Exact source location |
| --- | --- |
| nExp, reciprocal, causal exclusion and effective widths | Paper pp.9–10, Fig.8, §§5.2 and 5.4 |
| RecSqrt, interval lookup and optimized LayerNorm | Paper pp.17–18, Appendix H, Figs.11–12 |
| Executed CPU nExp branch | `GPU-MPC/ext/sytorch/ext/llama/api.cpp:2222–2266` |
| CPU inverse and ordinary softmax | Same `api.cpp:2458–2531` |
| CPU triangular softmax | Same `api.cpp:3671–3753` |
| CPU max/reduction | Same `api.cpp:3045–3267` |
| CPU fixed-to-float index and rsqrt | Same `api.cpp:2539–2659`; `GPU-MPC/ext/sytorch/ext/llama/fixtobfloat16.cpp:26–134` |
| CPU energy, mean and norm | Same `api.cpp:3269–3542` |
| CPU elementwise multiplication and lookup reconstruction | Same `api.cpp:1185–1236,2121–2208` |
| Backend width selection | `GPU-MPC/ext/sytorch/include/sytorch/backend/llama_transformer.h:75–101,134–140` |
| LayerNorm bias scale and RMSNorm outer floor | `GPU-MPC/ext/sytorch/include/sytorch/layers/layers.h:1088–1157,165–170` |

Earlier independent interpretations are in
`paper/reproduction-contracts/shared-profile-resolution.md:82–175` and
`paper/reproduction-contracts/shared_profile_oracle.py:282–321`.
The recorded SHA-256 for the pinned `api.cpp` is
`a8233617b21952396e3bb4bd0c27d024b982eb8a9c5d27910cc6ba3a063c1650`
(`shared-profile-resolution.md:40–42`). These recorded identities are not new
hash-verification or evaluated-build claims from this audit.

## LayerNorm: explicit paper contract

LayerNorm has the explicit optimized protocol in Fig.12. Paper §5.3 also says
the techniques yield RMSNorm; removing centering does not authorize a checkpoint
epsilon substitution or a new common-Qwen profile. The ring50/Q12 interpretation
below follows the existing paper-derived GPT-2 contract, not the paper's distinct
ring48 Llama2 evaluation settings.

For public row width `k`, inputs, private affine parameters and output are signed
two's-complement ring50/Q12, encoded with floor. The existing independent scalar
reference bounds `k` to 1–4096; this is not admission of a protected norm region:

```text
c     = floor(4096/k)
mu    = floor(c * sum(X_i) / 4096)
Z_i   = X_i - mu
S     = sum(Z_i^2) + 1
e     = floor(log2(S))
m     = floor(128*S / 2^e) - 128
index = 64*m + e
T[index] = isqrt(floor((2^55*k) / ((128+m)*2^e)))
Q_i   = floor(Z_i*T[index] / 4096)
Y_i   = floor(Q_i*gamma_i / 4096) + beta_i
```

- `S` is the complete **untruncated Q24** energy plus **one integer code**,
  not Q12 epsilon. There is no per-square floor, variance division, or energy
  floor before RecSqrt.
- RecSqrt's conceptual input precision is `24+log2(k)`. The public table absorbs
  the width factor exactly; rounding `log2(k)` into a shift changes the equation.
- `S` is **unsigned ring50**, with required domain `1 <= S < 2^50`.
  `e` is a six-bit integer, `m` a seven-bit integer in `[0,127]`, and `index`
  is a 13-bit/Q0 coordinate. The table has 8,192 ring50/Q12 payloads.
- Keep both signed floors: normalize first, multiply by gamma second. A single
  floor of the triple product is not equivalent.
- For non-power-of-two `k`, retain the floored reciprocal mean. At `k=768`,
  `c=5`; 768 copies of code 4096 yield `mu=3840`, `Z=256`, `S=50331649`,
  `T=65536` and pre-affine output 4096, not zero.

Required proofs precede issuance: mean accumulator/product and centered values
must satisfy their signed no-wrap contracts; squares and the complete energy
must satisfy the selected unsigned domain. Each normalization product
`Z_i*T[index]` and affine product `Q_i*gamma_i` must lie in
`[-2^48,2^48)` if using ring50 GapARS12. Final beta addition needs its declared
output bound. Observed extrema, metadata and masked-value tests are not proofs.
Correlated energy/table bounds may be needed where independent intervals are
too loose. Choosing a narrower energy domain is explicit admission, not evidence
that a trained checkpoint satisfies it.

### Protected exponent/mantissa construction

Paper Fig.11 derives protected interval-lookup outputs `e` and `u`, then:

```text
u     = 2^(49-e)
t     = S*u mod 2^50
m     = TR50,42(t) mod 128
index = 64*m + e mod 8192
```

The selected `t` lies in unsigned `[2^49,2^50)`, so signed no-wrap matrix
provenance is insufficient. TR42 here is structural integer extraction, not
ordinary Q24 rescaling: current typed TR subtracts the shift from fractional
bits and cannot simply be called with Q24/shift42.

Fig.11 permits a narrow local-extension exception: extending seven-bit mantissa
contributions then multiplying by 64 annihilates wrap multiples modulo `2^13`.
This is not general masked zero/sign extension. The exponent contribution must
already have correct ring13 mask/share semantics, and the combined index needs
its full mask/provenance proof.

Figs.11–12 reconstruct masked energy, normalization multiplier, product/index
and the indicated masked intermediate results, never their clear secrets.
Fig.12 finally returns party shares. A common-masked region output instead needs
an explicitly admitted reconstruction boundary, owner and frame cost. Dealer
masks must never enter evaluator ownership. Gamma/beta remain private model
operands; public-affine FuseFSS execution is not a replacement.

### Paper versus CPU artifact conflicts

| Difference | Consequence |
| --- | --- |
| `api.cpp:3429–3432,3494–3497` lacks the paper's `S+1` | Do not silently drop the guard or claim literal CPU equivalence when retaining the paper equation. |
| LayerNorm adds bias before final floor (`3456–3467`); `layers.h:1091–1092` declares bias Q24 | Bind bias encoding. The paper-derived contract above uses Q12 beta after the second floor. |
| RMSNorm helper comments out its final floor (`api.cpp:3523–3532`) | The caller restores it: `layers.h:1134–1139` sets forward truncation, executed at `165–170`. Two floors remain; helper-only inspection is misleading. |
| `fixtobfloat16.cpp:84–103` loops only over `i < bin`, without a terminal highest-bin contribution | Static inspection does not establish coverage of the paper's full unsigned domain. Resolve before allowing `S >= 2^49`; an initial narrower bound must be explicit. |
| The CPU converter uses arithmetic-output DCFs (`fixtobfloat16.cpp:31,41,79–90,123`) | Current GPU127 Boolean comparisons are not a drop-in replacement. Interval payloads/conversions and their security/cost contracts need review. |

## Causal softmax: explicit paper contract

Let `A` be the public active causal set and `1 <= k=|A| <= 1024`, matching the
bounded source workload/reference under discussion. This is not protected-row
admission. Scores and output use signed ring50/Q12; differences are certified
nonnegative integers:

```text
M     = max(X_i : i in A)
C_i   = min(M-X_i, 65535)
E0[j] = floor(4096*exp(-j/4096))       # j=0..255
E1[j] = floor(4096*exp(-j/16))         # j=0..255
E_i   = floor(E0[C_i mod 256] * E1[floor(C_i/256)] / 4096)
D     = sum(E_i : i in A)
Z     = floor(D/64)
R     = floor(2^18/Z)
P_i   = floor(E_i*R/4096)              # active entries
P_i   = 0                             # inactive entries
```

- If scores are certified in `[-2^37,2^37)`, their pair differences require
  signed39, not signed38. The paper's general effective-width schedule uses
  max width 39 and nExp width 40 at ring50/Q12.
- `C` is unsigned16/Q12; its two eight-bit coordinates select ring50/Q12
  payloads. Their complete Q24 product lies in `[0,2^24]`; one GapARS12 follows.
- `4096 <= D <= 4096*k` and `64 <= Z <= 64*k` because a maximum contributes 4096.
  Set `p=12+ceil(log2(k+1))`, `q=6+ceil(log2(k+1))`. Project the denominator
  into ring `p`, then TR6 to ring `q`/Q6. The inverse LUT has `2^q` ring50/Q12
  payloads. Index zero is unreachable and needs a fixed sentinel, not a claim
  that reciprocal zero is defined. At `k=1024`, `q=17`.
- Final multiplication is Q24 followed by its own signed floor. Do not
  renormalize quantized outputs: `[0,-4096]` gives `[3013,1107]`, sum 4120.
- Inactive entries do not enter maximum, nExp or denominator. Do not invent a
  finite negative-infinity substitute. One active entry gives code 4096.

Public positions, prefix lengths and selection counts require the declared
leakage policy; private lengths need separate treatment. Never reveal the clear
maximum, winning position, denominator, reciprocal, or secret-dependent validity
control flow. Reconstructed intermediates remain freshly masked. Public causal
zeros and the deterministic first-row one are distinct from secret output
disclosure (`api.cpp:3730–3741`).

### Paper versus CPU artifact conflicts

`api.cpp:2466–2474` fixes the inverse lookup to 16 bits. Ordinary and triangular
softmax supply `bw=scale+10` (`2516,3707`), hence 22 at Q12. This differs from the
paper's `k+1` widths. A static endpoint counterexample is 1,024 equal scores:
`D=2^22`, which is lost on projection into ring22, whereas the paper retains
`Z=65536` in ring17. This is a derived source counterexample, not a runtime test.

Triangular mode1 passes width 38 (`llama_transformer.h:134–140`); general mode1
passes 39 (`75–81`). Width 38 is insufficient for all pair differences from the
full signed38 score domain. A tighter source-bound certificate or an explicit
paper-width choice is required. D22's MLP certificate does not certify attention
scores. Neither width choice is adopted here.

## Historical reuse audit and implementation contracts (22 October)

| Existing shared owner/path | Reusable part and remaining boundary |
| --- | --- |
| `crates/pllm-garble/src/masked_wire_reference/arithmetic.rs`, `rescale.rs`, `typed.rs:957–1208` | D17 full-ring masks, TR, GapARS and XOR-masked DReLU. General same-ring faithful ARS and unsigned structural extraction remain separate work. |
| Same directory, `selection.rs` | Five-word reconstructed/deferred Select. New row-max, clip and broadcast ownership must be explicit. |
| Same directory, `source_lookup.rs:26–39` | D19 correction and reconstructed/deferred outputs; indices currently exactly 8/10 bits. General typed lookup, rsqrt13 and inverse widths 7–17 are absent. |
| `crates/pllm-garble/src/compact_dpf/gpu127.rs:79–112` | D16 point/less profiles 8–63. Full expansion is capped at 10; wider lookup needs bounded accounting and validation, not a constant-only change. |
| `masked_wire_reference/product.rs:1–20`; legacy `masked_wire_reference.rs:430–447` | Shared negative-mask product equations. A reusable consuming party-local elementwise-product/square owner is missing. |
| `masked_wire_reference/typed/internal_use.rs` | Fixed consuming ledger, not exported fanout; current slot counter supports at most 255 uses. |
| `crates/pllm-garble/src/fixed_activation_tables.rs`; `fusefss_backend/composition/source_tables.rs:31–87` | Precedent for method-neutral immutable tables; existing exp literals and exact inverse/rsqrt formulas. Extract public data/generation, not FuseFSS execution, capabilities or identity. |
| `crates/pllm-core/src/edge_contract.rs` | Signed/unsigned/predicate metadata, source/consumer bindings and resource quotes; declarations are neither capabilities nor range proofs. |
| `crates/pllm-compiler/src/sigma_linear_activation.rs` and its source/execution modules | Source-byte ownership, floor encoding, checked interval and continuous handoff patterns; norm/attention certificates and source inventories remain missing. |
| `crates/pllm-garble/src/shared_allocation.rs` | Existing 64 MiB aggregate logical admission and split original leases; not process RSS. |

The private product owner must remain distinct from the matrix singleton.
Guarded input backing can block issuance of another matrix operation; replacing
an elementwise product with fresh 1x1 matrices or detaching the guard is not an
authorized workaround. Preserve original source/backing/mask ownership through
last use, including frames retained after cancellation.

The old `masked_wire_reference/transport.rs` codec targets a different bounded
layout; it does not supply transport for new typed regions. Existing hashes,
checksums and local consuming permits do not authenticate peers or dealer
delivery, prevent restart replay, establish independent hosting, or elevate
security. Material needs confidential authenticated delivery in a future runtime.

## Next slice and unresolved decisions

This subsection retains the 22 October recommendations. D29 and D36 now govern
the subsequently approved bounded nExp and one-row softmax scopes; the historical
recommendation below does not revoke or expand those decisions.

The recommendation then was only the **unapproved bounded nExp owning region** in
[`../plans/sigma-nexp.md`](../plans/sigma-nexp.md). It uses existing 8-bit lookup
geometry and advances reusable private-product ownership without pretending to
complete row softmax or normalization.

Preapproval review tightened the proposal; these are **proposed PLLM
adaptations**, not a verbatim source schedule. Paper Fig.8 uses Select16;
pinned `api.cpp:2224,3026` uses global-width Select50 under ring50. Pinned CPU
TR reconstructs in one exchange (`api.cpp:2850–2913`); existing PLLM GPU127 TR
uses two. Paper nExp returns masked-result party shares; the proposal instead
uses reconstructed common-masked GapARS output. Thus its eleven exchanges are
a proposed PLLM composition count, not observed timing or source-wire parity.

The [proposal](../plans/sigma-nexp.md) now defines the unapproved private
coordinate mappings, 18 ordered owner uses, product/wrapper domains, both-input
ordered digest, phase-specific byte contract and canonical validation. These
reuse native-frame/activation-wrapper conventions, not authentication or legacy
transport. A clear root must pass every `0 <= X < 2^38` check before operation
entropy/secret allocation; inherited inputs must retain original-owner range,
mask and source provenance. D22 supplies no attention/nExp admission authority.

Product coefficients require independently uniform fresh splits per coefficient,
lane and invocation, plus an independent fresh full-ring output mask. D19 rho's
parity dependency is not a demonstrated attack; joint two-lookups → product →
GapARS privacy remains an independent composition-review obligation. Integer
correctness and normal tests cannot prove that privacy contract.

Reuse/extract the exact rational exp-floor certificates already in
`paper/reproduction-contracts/fusefss/gate-set/reference.py:138–174`, not a new
proof framework. Every selected literal still needs comparison against those
certificates. Metadata/hash/workspace quotes must be owner-defined and reserved
before construction or hashing, with continuous handoff leases; exact new owner
layouts and quote formulas remain an explicit preapproval blocker. Canonical new
spec/coordinate-provenance record encodings also require approval; declarative
specs must remain separate from actual issued bindings to avoid digest cycles.
None of these documentation details constitutes implementation or validation.

Outstanding decisions/proofs:

1. Approve that fixed region, its consuming product owner and reconstruction
   boundaries separately; no numeric-profile or decoder adoption is implied.
2. Prove the selected table literals exactly equal the floor formulas, and bind
   their source, table, mask and index domains. Existing fixture/hash history is
   not completion evidence for a new region.
3. Resolve paper-sized inverse versus fixed-16 CPU artifact; bind admitted row
   widths and causal/public-length policy.
4. Approve any wider GPU127 expansion and source lookup, especially 13/17 bits.
5. Choose paper interval lookup or a separately audited CPU DCF converter port;
   do not substitute FuseFSS's index network.
6. Bind norm kind, energy domain, guard, affine scales, floors, private parameter
   ownership, compiler proofs and final share/common reconstruction boundary.
7. Establish independent integer oracles, negative controls, normal runtime and
   lifetime/fault tests before any implementation-completion claim.

Checkpoint epsilon mapping, authenticated transport, private embedding/head,
stable selection/private feedback, full-decoder scheduling and SDK/quality gates
remain separate. No scaffold or local primitive supplies those qualifications.

## Conditionally approved next slice: one public-causal softmax root, active count 1–3

**D36 authorizes conditional local implementation, not achieved qualification.** This is the
next bounded decoder operator, not another nExp adoption or a claim of complete
paper reproduction. One row has public geometry of at most three positions and
an ordered, nonempty public causal active set `A`, `k=|A|` in `1..=3`. Positions
and lengths are public leakage; inactive positions are public zeros and never
enter any private maximum, exponent or denominator. No finite negative infinity.
Arbitrary private gamma/normalization, SDK routing and the full 1,024-entry row
remain unapproved.

### Numeric and owner contract

Retain the paper equation above, unchanged:

```text
M=max(X_i:i in A); E_i=nExp(M-X_i)
D=sum(E_i:i in A); Z=floor(D/64); R=floor(2^18/Z)
P_i=floor(E_i*R/4096)                   # inactive P_i=0
```

The external score root retains the original immutable charged input/source,
same counter and plan/session/operation identities. Check every declared score
in signed ring50/Q12 `[-2^37,2^37)` and the entire public geometry before operation
entropy or private allocation. This is a local trusted-dealer clear-input
premise, not checkpoint authentication or a compiler attention certificate.
Already-masked scores cannot enter by copying them into a new clear root.
For `k=1`, use existing public-value ownership for code **4096** and inactive
zeros; there is no private comparison/nExp/inverse/product schedule. Bind that
public output's actual custody/quote rather than inventing a private input.

For `k=2,3`, freeze this traversal in public position order:

1. Start with `M=X_a0`. Each of the `k-1` updates forms full-ring `d=M-X_aj`,
   privately projects it to signed39 for DReLU, obtains masked `b=[d>=0]`, and
   uses existing reconstructed **Select50** to form `M=X_aj+Select50(b,d)`.
   Keep the full ring50 operand; do not sign-extend a masked ring39 view. Equal
   values retain the earlier maximum convention; no winner/tie bit is opened.
2. Form the `k` private differences `M-X_i` in `[0,2^38)` and transfer them to
   **one batched inherited D29 nExp child**, with its original eighteen uses and
   eleven exchanges. Transfer original dealer/evaluator Bindings, masks, source,
   lanes and leases; never invoke clear-root nExp on recovered or guessed values.
3. Consume that child's original E owner into the outer root. Its private fixed
   ledger authorizes each E lane twice: denominator reduction and final product.
   Reduce without opening D. The maximum relation proves some difference is zero,
   hence some E is exactly 4096: `4096<=D<=4096*k<=12288`.
4. Privately project D and its mask modulo `2^14`; D is **unsigned14/Q12**, not
   signed14. Use native **unsigned TR14,6**, then a private numeric-coordinate
   handoff: unsigned8/Q6 output becomes the **same raw-code unsigned8/Q0 lookup
   coordinate** Z, with original backing/mask and `64<=Z<=64*k<=192` provenance.
   No public typed-TR relabel, ring widening, Q12 index or signed8 interpretation
   (which misreads 128–192). `p=14,q=8` follows the paper for k=2/3; k=1 bypasses it.
5. One reconstructed **D19 LUT8-to-50** supplies R. Authorize its scalar owner for
   exactly k internal product reads; broadcast is a private lane mapping, not
   cloned/exported capabilities. Reuse the consuming native elementwise product
   for E and that R, followed by reconstructed GapARS50,12. `0<=E,R<=4096` gives
   `E*R<=2^24<2^48`; preserve this complete Q24 product and its one final floor.
   Do not renormalize the quantized row or substitute matrix singletons.

This requires new **closed composition/provenance work**, not new numerical
recipes. At `ec68ad19`, `typed/nexp_region.rs::Input/construct` accepts charged
clear storage and prepares a root; it is not an inherited-input API. Its private
product is restricted to the D29 lookup pair. Required producers are the outer
comparison/difference owners; consumers are the nested nExp ingress/completion,
unsigned denominator-coordinate bridge and closed E/R product. Matching public
descriptors/digests, successful D27 outputs or a fresh `RootKnown` declaration
cannot mint those private inputs. Comparison, reduction, repeated-use/fanout and
their range/mask relations need proof before those boundaries may issue.

### Exact table and ordered-use declaration

Use the existing verified E0/E1 bytes/floors in `fixed_exp_tables.rs`; no new exp
or floating-point table generator. The sole new public inverse table is exact
integer arithmetic: `T[0]=0` (unreachable sentinel), `T[z]=(1<<18)/z` floored for
`z=1..255`. Zero is not a defined reciprocal or a private validity branch.
Retain the proof that only coordinates 64–192 are reachable. All 256 entries,
including the sentinel, must be checked before owning/hashing/issuing the table.
The full table has `T[1]=262144`: its global maximum is **not** 4096. The closed
lookup-result binding needs the genuine reachable-coordinate proof to establish
`1365<=R<=4096`; do not infer that interval from table identity or clip other
entries to make the bound appear true.

For `B=concat(z=0..255, u64LE(T[z]))`, record `H_payload=SHA256(B)`. Reuse D19's
**existing** table identity, not a digest-only capability:

```text
H_lookup = SHA256(b"pllm.sigma.source-lookup.canonical-table.v1"
                  || u8(8) || u8(50) || numeric_contract[32] || B)
```

Here `numeric_contract` must bind these equations/floors, unsigned coordinate
mapping, source/paper pins, E0/E1 identities and ring/scale constants. Canonical
record field order must be reviewed before hashing; no new record is approved by
this paragraph. Source/operation, k/A order, both product operands, role, issuance,
every edge/use and lane mapping belong in the invocation/ledger bindings, not
the table payload hash. Follow `source_lookup.rs::table_identity` and its distinct
`invocation_binding`; preserve D19's `rho=2u-r` dependency.

Declare outer uses in this order, separately for dealer role2/evaluator roles0/1:

| Block | Ordered reads/transfers |
| --- | --- |
| Entry | One original score-root entry. |
| Each maximum update | Eight: M/X subtraction operands; d50 projection; d39 DReLU; b/d50 Select operands; X/Select-result affine operands. |
| Differences | Two per active lane: final M and original X. |
| Nested child | One difference-vector transfer; one original E-vector receipt. Child retains its separate eighteen-use ledger. |
| Denominator | k E reads, then D projection, projected D TR, original TR-output coordinate mapping, coordinate LUT: k+4. |
| Output | k ordered E/R product pairs, one product-to-GapARS transfer, one final output handoff. |

Thus the proposed outer template has **27 uses at k=2, 40 at k=3** (`13k+1`),
plus the unchanged nested eighteen-use template and native gate ledgers. Reuse
`typed/internal_use.rs::Ledger`, burning before checking; no public fanout API,
capability Clone, raw-native extraction or extra retry. Slot, operand ordinal and
actual owner/coordinate identity must match. This is a proposed template, not an
already-issued use permit.

### Exchange, admission and failure phases

For k=2/3, the public online order is: `2*(k-1)` exchanges for maximum
(DReLU39 then Select50 per step); **11** for the single batched nExp child;
**2** for TR14,6; **2** for reconstructed inverse lookup; **1** for product;
**2** for final GapARS. Proposed totals: **20/22 exchanges, 40/44 peer frames**.
These are logical composition counts, not timing, bytes, offline-material
delivery costs or current execution evidence. Local affine/reduction/projection
work is not zero merely because it sends no frame.

- Reuse D27's shared `typed/role_coordination.rs` custody, two sibling slots and
  both-Ready barrier. Bind outer/child operation, phase, role and replay context;
  no returned root output before both siblings finish. Caller polls remain
  outside locks. Wrong frame/role/phase, cancel/drop or unwind burns both siblings.
- Validate whole root/source/range/geometry, tables, counters, work and aggregate
  caps **before** entropy, masks, keys or private factories. Checked-add all actual
  prior/request work and pending comparison + inherited-child + TR + lookup +
  product + GapARS work, including metadata/hash, local lane visits, initialization,
  repeated reads, failure erasure and cleanup. Reuse native owner work categories;
  do not treat descriptor counts as a CPU budget or reset per child/phase.
- Quote concrete roots/source backing and actual capacities, all role/ledger/
  Arc/header/transition/Result/workspace/table slots, child construction overlap,
  mailboxes, retained frames and returned outputs before their construction under
  the **same existing 64 MiB logical ceiling**. No released/reacquired guard,
  phantom spare-capacity credit or unpriced clear-input/child/frame allocation.
  Current clear-root nExp quotes do not prove inherited/outer quotes. Exact new
  layouts and total byte/work formulas remain required before factory entry;
  this packet supplies no fabricated full-root byte total or Budget authority.
- Consume/burn issuance and use slots before partial material can escape. Any
  construction error/unwind retires the root and produced child material; no
  nonce/entropy/work refund or retry on that owner. Retained output/frame leases
  survive cancellation until actual erasure/drop. Keep full-output-ring arithmetic
  masks, independent Boolean masks and fresh independent product splits/output
  masks; no small-mask shortcut or decoder-visible intermediate.

### D36 approval and next acceptance gate

[D36](../../paper/methods/DECISIONS.md#d36-conditional-bounded-sigma-causal-softmax-root)
approves **only conditional local implementation of this fixed one-row k<=3
composition**, including its inherited private bridges, ordered uses and explicit
reconstructed boundaries. Freeze canonical records, genuine owner handoffs, full
owner/work quotes and numerical premises before their respective
hash/construction/issuance boundaries. D29 alone is not that permission.

First acceptance: independent literal integer oracle for k=1/2/3, ties, score
endpoints, clip65535/65536, equal-score Z=128/192, inactive positions and
`[0,-4096] -> [3013,1107]` without renormalization; all256 inverse entries and
independent table-hash bytes; both-role transcript/owner parity. Then real-factory
zero-draw invalid/exhausted-work rejection, exact/one-short capacity/work limits,
original-counter/source identity and repeated-read misuse, partial entropy/factory
failures, retained frames, cancellation/replay/unwind and both-Ready negatives.
Those tests are **not run or added here**. Independent joint-view review must
cover max/Select, correlated E reduction and R broadcast, D19 corrections and
product/GapARS; functional tests do not prove that composition private.

No transport authentication, checkpoint conformance, general normalization,
arbitrary gamma, wider 11–17-bit expansion, full-row/decoder/SDK, request history
or security promotion follows. Allocator behavior, native/private callframes,
CPython and process RSS remain outside logical owner/work accounting.
