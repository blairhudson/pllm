# PUMA candidate: bounded replicated arithmetic

`pllm_core::puma` implements three-party scalar `TruncAPr`; `puma::prss`
implements its pinned pairwise setup and bounded sequential stream owner.
It is independent Rust, with no upstream runtime dependency. Specification:
`secretflow/spu@a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4`,
`libspu/mpc/aby3/arithmetic.cc:750–941`, as resolved in
[source contract §3.1](../../paper/reproduction-contracts/shared-profile-resolution.md#31-truncapr-is-stochastic-with-a-derivable-exact-law).
This is the pinned **candidate**, not proof of the evaluated PUMA build.

**Decision update, 17 October 2026:** [D6](../../paper/methods/DECISIONS.md#d6-approved-puma-source-derived-q18-profile)
approves `puma.source-derived-q18.v1`, exactly
[source resolution §3.3](../../paper/reproduction-contracts/shared-profile-resolution.md#33-approved-p-n1-profile-source-derived-q18),
including its fixed rounding/constants/order and corrected softmax/normalization
formulas. P-N1's choice is resolved for that named interpretation, not the missing
evaluated graph. [D9](../../paper/methods/DECISIONS.md#d9-fixed-count-private-token-feedback)
approves fixed-count private full-prefix feedback, lowest-index protected argmax
and final client-only IDs. Those approvals did not themselves change native inputs or
previous test/measurement record.

## Bounded SDK composition

`pllm.profiles.PumaThreePartyCpu` composes `PumaThreePartyDecoder`,
`ThreePartyReplicatedRoles`, `ReplicatedPrivatePrefix` and CPU execution through
ordinary Pipeline/Experiment resolution. `examples/papers/puma.py` shares the
paper study's model and budget. Native source and schedule admission precedes
source-value loading and one-use request publication.

The model-neutral compiler lowers the existing semantic decoder into public
instructions for private complete-table lookup, matrix products, RMSNorm, rotary,
causal attention, SiLU/gated products, full head and lowest-index argmax. D6's
source-derived Q18 arithmetic and D10's integer-one RMSNorm epsilon remain distinct
from W8A8. A public conditional range proof relates each RMSNorm input to its
own row's squared sum before inverse multiplication. It covers floor/ceil
stochastic rounding without inspecting an activation or selecting a private range.

D9 selects full-prefix recomputation. Every output selection executes the complete
original prefix plus all earlier shared selections, with fresh arithmetic work.
The compiled program retires intermediate full-KV arrays between executions;
it does not reuse a decode cache. Worker-local argmax outputs feed the next
private lookup directly. Only the final fixed-count token-ID vector is returned
to the trusted client. Natural-EOS stopping, cross-request continuation and
client-visible logits are outside this contract.

Each of three worker processes owns only its adjacent replicated components and
two pairwise PRSS keys. Directed peer channels have distinct credentials; secret
seed and arithmetic frames travel directly between their owners. Worker A
coordinates public barriers without relaying the other pair's payloads.
Authenticated phase/session/source checks, bounded queues, expiry, replay burns,
drained cancellation and response-held admission reservations govern lifetime.
The existing transport admits loopback HTTP and certificate-checked IP-literal
HTTPS. The source threat model still permits at most one static semi-honest
corruption; these controls provide no malicious-result verification or evidence
of independently operated workers.

The default private-weight source path admits complete tokenizer vocabularies at most 256,
at most 32 input tokens and eight outputs, four layers, feature dimensions at
most 8,192 and 1,048,576 source values. The primitive-specific row/rotary limits,
compiler-view budget and eager-range gates can further reject a
configuration. Primitive payload ceilings remain 256 KiB per party; compiled
shares, public program metadata and transport copies have separately admitted
owners. Source weights remain client-owned during publication. These bounds
do not admit the study's full-size Qwen checkpoint or qualify a memory tier.
The pinned Qwen2.5 source declares 24 layers and 151,936 vocabulary rows;
its main token table alone contains 136,134,656 values. The ordinary quality
command rejects even the 4-input/two-output declaration at native source
admission. Enlarging a dot tile does not supply compact full-model ownership
or complete-decoder public range proofs.

Long private contractions split their inner dimension into at most 1,024-value
chunks. Each chunk uses fresh exact ring64 multiplication, the client never
reconstructs a partial result, and the worker adds the partial shares locally.
Only the complete accumulated dot is truncated to Q18. The complete public
accumulator range must pass admission, even when each chunk fits separately.
This adds product rotations but does not change the numeric rounding boundary
or the primitive payload ceiling. Small contractions retain the existing path.

The compiler now retains bounded strided source descriptors instead of one
reference pair per weight and per source tile. Execution copies only the admitted
primitive tile from its immutable party-local source owner. Source descriptors
check endpoint arithmetic, owner identity and tile size before adding a node.
Identical immutable tensor programs share one allocation; every invocation still
creates fresh party-local protocol state and consumes its complete work allowance.
Construction views retire at their last semantic consumer. Six frozen generated
Qwen2/Qwen3/Llama execution identities match the preceding installed wheel, and
mixed compact/expanded-gather workers preserve exact integer matrix outputs.
Resident source shares, expanded operation schedules and primitive work retain
their existing admission bounds. This construction change supplies no measured
whole-client memory or full-checkpoint capacity result. Build and installed SDK
checks: [compact-compiler evidence](../../paper/artifacts/puma-sdk/local-v3/README.md).

### Explicit public-weight adaptation

[D11](../../paper/methods/DECISIONS.md#d11-explicit-public-weight-puma-adaptation)
adds `PumaThreePartyDecoder(weights="public")` as the `linear` slot of
`PumaThreePartyCpu`. Omission retains the private-weight composition and its frozen
execution identities. The public mode has a separate numeric profile:
`puma.public-q18.rmsnorm-epsilon1.full-prefix.batched-final-greedy-ids.v1`.
It retains private prompts, activations, argmax and feedback under the same
three-party corruption assumption; it makes no model-weight confidentiality claim.

The client streams original checkpoint values through bounded native publication
frames, retaining tensor commitments and public coefficient ranges instead of
resident weight shares. Each worker independently imports the locked local source
into immutable Q18 files, checking every encoded value, tensor hash and public
affine coefficient against the publication. A 64 KiB read buffer feeds complete
ring64 products into both replicated activation components. Both traverse every
weight in public order. One Q18 truncation follows the complete sum, never a file
chunk. Lookup uses private one-hot selection; no private token selects a disk
address. Bounded equality batches and a stable argmax tournament keep selections
shared, including odd widths and equal-logit lowest-index ties.

Public-source geometry can admit up to 64 layers, 262,144 vocabulary rows and
536,870,912 source values. The existing compiler-view, live-value, primitive and
work limits still apply. Snapshot disk admission is separate from RAM; incomplete
imports, malformed frames and cancellation retire their owners. Source-present
workers read their own checkpoint; missing-worker distribution is unmeasured.
No resident client weight matrix or weight-share delivery is introduced in this
mode. Public file reads, nonlinear exchanges, full-prefix recomputation and all
three worker costs remain charged.

The pinned Qwen2.5 4-input/two-output metadata check now rejects the compiler-view
payload bound rather than the private-mode source-expansion bound. It reads no
tensor values and launches no workers. Compact activation views and loop-based
execution, complete-model range admission, quality and measured response costs
remain necessary for practical full-checkpoint use. Generated SDK controls and
complete integer matrix oracles establish this bounded adaptation only. Evidence:
[public-weight slice](../../paper/artifacts/puma-sdk/local-v4/README.md).

### Shared SDK interfaces

Ordinary `OpenAI`/`AsyncOpenAI`, gateway Responses/Chat Completions,
`pllm benchmark run` and `BenchmarkTarget.response` use this composition.
The benchmark charges client publication, all directed peer requests and ACKs,
all three provider CPU/peak-memory owners, and native shared access caps.
Buffered final output cannot provide a streaming prefill or N−1 decode rate;
those metrics and their body-window partitions remain unknown.

`pllm benchmark quality` also accepts this composition. It scores final token
IDs against the locked FP32 reference through ordinary three-worker requests.
Each measured position completes one fixed-count, one-selection request on the
common caller-owned prefix. Subsequent diagnostic prefixes use the first
candidate's greedy trajectory. Every position repeats source publication,
sharing and setup; no hidden logit or intermediate feedback export is added.
The Experiment must budget every diagnostic request and the longest extended
input. Top-1 agreement is observable; top-k recall and logit error are null.
These repeated requests measure numeric agreement rather than a single serving
response's costs. Full-checkpoint admission still precedes any quality execution.
The separate [quality/wide-contraction evidence](../../paper/artifacts/puma-sdk/local-v2/README.md)
retains generated controls, the installed build and the common-Qwen rejection.

Generated Qwen2/Qwen3/Llama controls include nonzero attention/MLP paths whose
independent sign oracle distinguishes private prefixes, private feedback, protocol
failure retirement, ordinary SDK/gateway execution and benchmark accounting.
Complete hidden-tensor conditional-on-mask equality, full-checkpoint ranges,
source-model quality and matched publication measurements remain separate gates.
The [frozen SDK conformance package](../../paper/artifacts/puma-sdk/README.md)
retains 692 broad installed-wheel checks (one optional Metal skip), 138 final-build
checks, 165 native checks and five ownership doctests. All 367 installed package
files match before and after each suite. The final rebuild only sorts native
module declarations; overlapping suite counts remain separate. The ordinary
common-Qwen factory preflight rejects during resolution, contributing no serving
measurement.
Replay uses the ordinary installed-wheel tests, not a paper-specific executor:

```sh
python -I -m pytest -q -o pythonpath= tests/test_replicated_sdk.py \
  tests/test_replicated_worker.py tests/test_replicated_interfaces.py
cargo test --locked --offline -p pllm-compiler --lib replicated_decoder
```

## Numeric contract

Unsigned wrapping ring64 represents signed two's-complement X. Require
`-2^62 <= X < 2^62`, `1 <= b <= 62`, and at most 1024 lanes. These bounds
apply to the reconstructed value, never individual random share components.
The compiler/caller must establish this public range obligation; the runtime
does not open, reconstruct or test private X to enforce it. For products and
accumulators, the obligation concerns the **intended unwrapped integer before
ring reduction**. An overflowing integer that wraps into the quarter-gap does
not satisfy it. Q18 multiplication uses b=18 only after proving that raw-product
bound; accumulation requires its own bound before any modulo reduction.

For fresh uniform R in ring64, C=(X+2^62+R) mod 2^64 and

```text
Y = ((C mod 2^63) >> b) - ((R mod 2^63) >> b)
    + (MSB(C) xor MSB(R))*2^(63-b) - 2^(62-b)
  = floor(X/2^b) + [(X mod 2^b)+(R mod 2^b) >= 2^b].
```

Thus upper rounding has probability `(X mod 2^b)/2^b`; exact multiples are
exact. There is no additional large-wrap error within the admitted quarter-ring
gap. `oracle::{distribution,truncate}` is a separate wide-integer, deterministic
**conditional-on-mask** oracle, not a replacement exact truncation protocol.
For X=-9,b=3, R=0 returns -2; low R=1..7 returns -1 (7/8 probability).

## Party-local protocol and material boundary

Party i owns only `(x_i,x_(i+1))`. A public pivot relabels A,B,H cyclically.
The low-level API consumes a public pivot; `prss::SourceParty::plan` now derives
it from the pinned public PRG and binds its cursor into the operation.
There is no global runtime/reconstruction function in production. H is the
third online party, not a dealer. H's authorized view includes the full random
mask R; H never receives C or an evaluator opening.
The source threat model permits at most one static semi-honest corruption.
Consistent replicated inputs, correct setup and authenticated private channels
are assumptions; this primitive provides no malicious-party verification.

The low-level `LocalMaterial` adapter imports **one party's** PRSS outputs.
The higher-level `SourceParty` issues these internally and never detaches them:

| Party | Authorized fresh components |
| --- | --- |
| A | rA, yA shared only with H |
| B | rB, yB shared only with H |
| H | rA,rB,yA,yB from its two pairwise streams |

R=rA+rB. The numeric law above assumes ideal fresh uniform words; source PRSS
implements their computational pseudorandom analogue. Low-level raw imports do
not certify freshness. `SourceParty` enforces source cursor issuance and retires
keys with the online owner. Test-only deterministic fixtures know all shares;
production has no aggregate dealer, deterministic seed constructor or key registry.

1. `begin`: A/B send their masked input contributions to each other. H draws
   private splitting words, adds MSB(R)/middle(R) to A's
   pieces and negates the original words for B, exactly the source equations.
   H sends two correlation frames and retains output `(yB,yA)`.
2. `open`: each evaluator supplies the peer masked contribution and its H
   correlation frame at one round barrier, forms C, computes its additive
   high-bit correction and subtracts yA/yB. It sends a `Reshare` frame.
3. `reshare`: A/B combine those pieces. Outputs are `(yA,z)`, `(z,yB)`,
   `(yB,yA)` in pivot order. `take_output` moves one local pair once.

Six directed online frames carry 8 ring words per lane in total, excluding
headers, setup and transport. The source owner fills H's complete `2*n` array
with one private-PRG call (all rb words, then all rc words). The low-level raw
material adapter retains its separate fresh-OS private-splitting path.

## Pinned PRSS semantics and custody

Additional immutable specification: SPU `libspu/mpc/common/prg_state.{h,cc}` and
`bazel/repositories.bzl`; the latter pins
`secretflow/yacl@5418371c4335f4a64fbd0bdabb0efd94da2af808`.
YACL `yacl/crypto/tools/prg.{h,cc}`, `crypto/base/symmetric_crypto.cc`, and
`crypto/utils/rand.{h,cc}` fix the following **64-bit little-endian** profile:

- Each role draws independent 128-bit public contribution, private key and own
  pairwise key. Native implementation uses separate OS entropy calls; no key is
  derived from public metadata or a central master seed. Public seed is the
  three contributions' sum modulo 2^128, not XOR.
- Actual source sends `self_seed` to **PrevRank** and receives **NextRank**'s
  seed; its prose comment is reversed. Each owner holds exactly its own and next
  edge keys, plus its private key. The third edge key is absent.
- Each fill starts AES-128-CTR at `LE128(cursor)` with raw little-endian key
  bytes. Inside that call, counter blocks increment **big-endian**, matching
  EVP CTR. Return cursor is `cursor+ceil(bytes/16)`; unused partial-block bytes
  are discarded. A buffered conventional CTR stream is not equivalent.
- Per truncation: public cursor +1 for one u64 pivot; pair cursor
  `+2*ceil(n/2)` for separate n-word mask and output fills. A uses First, B
  Second, H Both; selecting one stream still advances the shared logical cursor.
  H's private cursor advances n blocks; other roles' private cursors do not.
- Source computes `(raw_pivot+1)%3` and `(raw_pivot+2)%3`. The two u64 values
  that overflow alias roles in the source. This bounded owner rejects and burns
  them, without repair/resampling (public abort probability 2/2^64 under ideal
  uniform pivot words). The admitted numeric truncation law is unchanged.

`SetupParty` performs public contribution gathering, then one directed secret
seed exchange. Canonical role-ordered contributions, profile and original
Context bind a fresh setup digest. Operations additionally bind exact public,
mask and output cursor starts with distinct purpose labels. Those labels bind
metadata; they do **not** alter the source AES inputs or invent a KDF.

`SourceParty` owns PRSS and `PartyRuntime` together. `plan` burns a public draw;
`begin` reserves pair/private ranges and AES work **before** material allocation.
All online failures/cancellation retire both owners. There is no Clone, seed
export/import, persistence, cursor reset, detachable material or source fork API.
New setup always draws fresh OS seeds; its fresh contribution transcript rejects
old seed messages even if caller Context is repeated. Process-ID checks retire
forked owners. Arbitrary process-memory snapshot rollback is not a persistence
feature and cannot be made safe by these in-memory types.

Admission remains 1024 lanes/4096 operations. Additional cumulative work ceiling:
`(3*1024+3)*4096` AES blocks, locally reducible. Each evaluator uses `2*ceil(n/2)`
pairwise AES blocks; H uses `4*ceil(n/2)+n`, plus each role's one public block.
Fixed-size setup allocates no payload buffers. The direct `SourceParty` API admits
sequential TruncAPr-only schedules. Its compound owner below additionally composes
elementwise/matrix multiplication, typed Boolean AND and arithmetic-to-Boolean
carry circuits on that same stream.
Other kernels still need their source-ordered consumption contracts.

### Native Q18 multiply → truncate

`puma::multiply` now composes the existing source PRSS and truncation owners
with replicated multiplication. It implements the frozen
[multiply/truncate specification](../../paper/reproduction-contracts/puma/README.md)
inside `pllm-core`, for subsequent decoder integration.

`InputBounds` checks all four unwrapped signed-i128 product corners for every
lane before randomness. The whole product interval must lie in `[-2^62,2^62)`.
It also derives the all-mask rounded output interval. These public bounds remain
caller/compiler obligations; no private operands are opened to check membership.

Each role owns two adjacent shares of each operand. Its fresh pairwise zero share
masks the three local cross-products before one rotation to the previous role.
The received next-role component enters the existing truncation owner without
reconstruction or a second product snapshot. The execution ID is
`puma.spu-a42c261.mul-aa-truncapr18.ring64.quarter-gap.replicated3.v1`.
Setup, interval envelopes, phase/ordinal and separate zero/mask/output ranges
bind that identity; the approved Q18 numeric profile and AES algorithm remain.

`plan` admits the complete operation without consuming PRSS. `begin` consumes
the product's Both fill and emits the rotation. Only `rotate` draws the public
pivot and starts TruncAPr. Helper private splitting precedes output-mask fills;
its prepared buffer becomes the existing correction buffer. This retains exact
source ordering without an additional allocation or fresh-entropy fallback.
For `q=ceil(n/2)`, successful pair/public/private consumption is `3q/1/n` for the
helper and `3q/1/0` for evaluators. AES work is `6q+n+1` or `4q+1`, respectively.
Nine directed frames carry `11n` ring words before headers/setup/transport.

One non-cloneable compound owner couples all phases, keys, counters and failures.
Its unchanged 256 KiB ledger retains original rotations, truncation frames and
outputs through erasure and deallocation. Admission uses the joint conservative
`4096 + 128n` bound plus earlier live payloads. The original truncation-only
role estimate is insufficient: a helper retaining its outgoing rotation reaches
11 words/lane, versus the truncation helper's 10. The 1024-lane regression measures
this overlap. Retaining eight complete output pairs blocks the next full-size
operation before PRSS consumption. Every cancellation phase retires the chain.

`PartyRuntime.plan_matmul` also binds row-major matrix products to this same
owner and stream. Pinned `arithmetic.cc:628–660` and `fxp_base.cc:264–269`
specify one output-shaped zero fill and rotation, followed by truncation of
each **complete** dot product. Operand position, all three dimensions and the
public unwrapped accumulator bound enter a distinct matrix execution identity.
Each input/output has at most 1024 entries. Joint admission additionally prices
both replicated input tensors; `Plan.admitted_bytes()` reports that bound.
Matrix tests compare raw reconstructed dot products with independent signed-i128
sums, then check the stochastic output law, mixed elementwise/matrix cursors,
operand substitution and allocator-failure retirement through the exported API.

Native tests read the existing frozen independent vectors directly, checking
product components, complete output shares and all stream cursors across odd/even
chained shapes. Fresh OS-key exported-API tests cover up to 1024 lanes, the signed
rounding law and frame/work reconciliation. Allocation-index sweeps inspect
erasure and still-held reservations inside deallocation; compile-fail checks
reject cloning setup, runtime and Boolean-share owners. The current suite passes
**87 native checks** (50 unit, 8 truncation, 3 PRSS, 2 compound exported-API,
21 allocation, 3 compile-fail). The frozen multiplication oracle's 14 checks and
four independent Boolean/carry/OT/PPA transcript checks also pass.
Focused formatting passes; Clippy passes with the existing unrelated
`needless_range_loop` warnings allowed. Replay uses the commands below.

Graph-wide range derivation, other mixed kernels, confidential authenticated
transport, protected feedback and ordinary SDK/Experiment execution remain open.
This native integration supplies no checkpoint quality or serving measurement.

### Typed Boolean products on the same stream

`PartyRuntime.plan_and`, `begin_and`, `receive_and` and `take_boolean_output`
implement the same source's
[`AndBB`](https://github.com/secretflow/spu/blob/a42c261e50fe0602ccdec4b188b8dc2cbcfbb7b4/libspu/mpc/aby3/boolean.cc).
Each party computes three local AND cross-products, XORs its adjacent fresh
edge words, then rotates the masked component. No party reconstructs a value.
Boolean shares have their own non-cloneable type and frame namespace; operand
position, nominal widths, setup, phase, ordinal and pair cursor bind admission.

The source chooses an 8/16/32/64-bit backing type from the larger public operand
width. Individual shares fill that type even when the reconstructed value has
fewer significant bits. One Both fill advances the shared pair cursor by
`ceil(n*backing_bytes/16)` and uses twice that many AES blocks per party. Public
and private cursors do not advance. Tail bytes retire after each fill, so replacing
these typed draws with ring64 draws would change subsequent truncation material.
Native payload storage remains fully charged u64 words, including narrow shares;
typed AES consumption is not a wire-byte or allocation-saving claim.

Independent scalar AES/XOR fixtures check complete rotation hashes across all four
backing types and partial-block transitions. Native checks cover every nominal
width 0–64 and up to 1024 lanes. Fresh OS-key exported-API execution interleaves
Boolean and Q18 matrix/elementwise operations on one source owner. Allocation
failure, substitution, replay, wrong output extraction and cancellation retire the
coupled keys; original frames remain charged until erasure and deallocation.
The existing 256 KiB ceiling is unchanged.

### Arithmetic-to-Boolean carry circuits

`PartyRuntime.plan_conversion`, `begin_conversion` and `advance_conversion`
implement the pinned source's full ring64 `A2B` and `MsbA2B`. A masked split
rotation precedes the source's Kogge–Stone addition or halving carry circuit.
Seven AND rotations use the existing Boolean primitive; their public vmap lane
order, backing widths and partial AES fills preserve the shared cursor schedule.
All intermediates remain party-local adjacent shares. The final Boolean owner
comes from `take_boolean_output` without reconstruction. These are arithmetic
conversion and sign primitives, not arbitrary signed comparisons or nonlinear
operators.

Admission checks the complete eight-round work envelope before randomness,
limits inputs to 512 lanes and prices `4096 + 256n` bytes plus previous live
payloads. This includes all original outgoing frames retained through completion.
Compound gate metadata stays inline within the fixed reserve. Round-specific
bindings reject replay/substitution; malformed frames, allocation failure and
cancellation retire the common PRSS owner. The per-runtime 256 KiB limit remains.

Independent OpenSSL/scalar fixtures check complete rotation transcripts, including
random padding bits in narrow carry stages. For three lanes, full A2B consumes
22 pair blocks and MSB consumes 11; neither consumes public/private PRSS. Native
checks cover signed endpoints, wraparound, odd lanes and maximum tiles. Exported
OS-seeded tests interleave both circuits with Boolean AND and Q18 matrix products;
allocation-index sweeps check every role and every circuit phase. Native frames
still store/charge u64 words (42n words across all parties), independently of the
source's narrower Boolean storage types. Transport/wire metrics and whole-model
range/fidelity claims require the subsequent SDK composition.

### Boolean-to-arithmetic selection

`plan_boolean_to_arithmetic`, `begin_boolean_to_arithmetic` and
`receive_boolean_to_arithmetic` now implement the pinned `B2AByOT` selector for
public widths 0–8. Planning binds an existing opaque Boolean owner; sign outputs
therefore enter this conversion directly. Helper, receiver and sender retain
separate views, with pairwise PRSS supplying the source three-party OT masks.
The receiver returns one arithmetic component to the helper. No participant
reconstructs the predicate; `take_output` returns an ordinary replicated ring64
owner. Zero-width public values produce zero shares without PRSS or peer frames.
Widths 9–64 select the source's distinct PPA implementation described below.

For `L=n*bits` and `q=ceil(L/2)`, nonzero conversion draws one public pivot and
advances pair cursors by `3q`. Helper/sender use `4q` pair AES blocks; the receiver
uses `2q`, with two logical `None` fills advancing its cursor. Private PRSS stays
unchanged. Four directed bodies contain `3L+n` ring64 words. These remain native
payload counts before authenticated transport, framing and setup.

Admission checks the full expanded geometry and worst-role work before the public
pivot, with `L<=1024` and `4096+8*(16L+8n)` bytes plus existing payloads. Planning
consumes the nonzero public draw; abandoning its plan retires the common owner.
Source overflow aliases reject as for truncation. All private payload allocations
are fallible, and original outgoing messages keep their reservations through
erasure/deallocation. Independent AES/scalar vectors check masks, both OT messages,
return components, full output pairs and cursors. Maximum-width/tile, opaque
MSB-to-arithmetic, malformed phase, replay and allocation-failure controls pass.
General nonlinear graphs and ordinary SDK/Experiment execution remain open.

The full-width `B2AByPPA` path reuses the existing carry owner. Two source Both
fills form independent arithmetic masks and a Boolean sharing of their sum;
the original input adds to that sharing through the same seven-gate carry
schedule. Only role zero receives the masked `x+r`, while roles one and two
retain negative mask components. A final rotation forms adjacent additive
pairs. `advance_boolean_to_arithmetic` consumes one bounded frame and may emit
the reveal and rotation together; all unmasked values remain shared.

For `q=ceil(n/2)`, pair cursors advance by `3q+6n`; every role performs twice
that many AES blocks, without public/private draws. The 28 directed native
frames contain `46n` ring64 words. At most 512 lanes enter the unchanged
256 KiB ledger under `4096+320n` byte admission plus earlier owners. Independent
full-width transcripts, maximum tiles, opaque A2B→B2A round trips and allocation
sweeps cover split, every carry round, masked reveal and final rotation. This
completes the source's bounded conversion selector, not the remaining nonlinear
decoder, authenticated transport or ordinary SDK method.

### Composable arithmetic owners

The same runtime now executes bounded public affine combinations and standalone
`TruncAPr`. `affine` binds input lineage, exact public coefficients and interval
bounds, charges both retained inputs and a separate output pair, and proves the
declared unwrapped result fits signed ring64. It consumes a schedule ordinal but
no randomness. `begin_rescale` admits the full helper/work envelope and quarter-ring
input interval before the pivot draw. Public coefficient products therefore use
local `MulAP` followed by the original truncation rather than a private `MulAA`.

`plan_ring_product` separately exposes the exact, unrescaled source `MulAA` for
integer selection products. It uses one Both fill and rotation, with no public
pivot, helper mask or private draw. Its distinct identity leaves existing Q18
elementwise and matrix schedules unchanged. `bind_operand` and
`bind_conversion_input` consume existing local share owners into admitted slots
without exporting or copying their private components. The caller/compiler still
supplies the public graph and range obligations.

A composed exact ReLU control runs affine ownership, private sign extraction,
Boolean-to-arithmetic conversion, exact integer multiplication and subtraction.
It never reconstructs an intermediate in a role. Independent integer checks
cover complete ring64 endpoint selections, maximum tiles and stochastic rescale
hulls; allocation sweeps cover local outputs and every rescale role. This is an
arithmetic-composition control, not the PUMA GELU/SiLU graph or a decoder result.

### Shared tensor programs and approved activations

`pllm_core::puma::multiply::tensor` now composes those owners into bounded public
programs. `Builder` supports affine arithmetic, Q18 or exact integer products,
standalone rescaling, shared sign extraction and Boolean-to-arithmetic conversion.
`Program::activation` constructs the approved D6 SiLU/GELU graphs, with the frozen
binary32/toward-zero constants, coefficient-by-coefficient rescaling, original
power order and exact branch endpoints. GELU computes `x6=Square(x3)`; SiLU uses
`x6=mul(x2,x4)`. Every lane executes both polynomials and all comparisons. Exact
integer selection products keep predicates and selected outputs shared.

Activation programs admit 1–128 lanes and at most 192 nodes. Public interval propagation
checks **every eagerly evaluated intermediate**, including polynomial values on
tail inputs; it rejects unsafe quarter-ring truncations before an executor draws
randomness. These proofs assume the caller/compiler's public input interval.
They neither inspect private inputs nor establish whole-checkpoint ranges.
Program digests bind instructions, coefficients, shifts, geometry, numeric profile
and all intervals. The runtime preflights the complete work/ordinal envelope and
conservative register, frame and primitive overlap under its existing 256 KiB cap.

`Executor` consumes one party's existing runtime and input pair. Its public
`start_step`/`receive`/`finish_step` schedule carries opaque, non-cloneable frames;
program, invocation, instruction and primitive bindings reject substitutions and
replay. Successful completion returns that same runtime with advanced PRSS cursors
and one output share. Last-use registers retire during execution. Cancellation,
incomplete completion and any transition/allocation error burn the runtime;
original exported frames retain payload and metadata charges until destruction.
The caller must still propagate failure to peers. Authenticated transport and
compiler/SDK decoder execution remain the next integration gates.

Independent stochastic-integer recipes check all branch endpoints, odd lanes and
128-lane tiles. Production OS-keyed controls check replicated-pair consistency,
shared zero/identity tails and sequential runtime reuse. Every instruction and
protocol round has cancellation coverage; allocation-index sweeps and allocator
callbacks verify erasure before charge refund. The activation-slice native suite passed
97 checks (60 unit, 8 truncation, 3 PRSS, 3 exported multiplication/tensor and
23 allocation), and 33 source-profile/oracle checks pass. Targeted Clippy passes
with the existing `needless_range_loop` warning allowance; unrestricted core-crate
Clippy still reports unrelated byte-dot/Maverick warnings. These checks establish
the bounded activation executor, not model quality or serving performance.

### Protected normalization and row programs

The same `Program` and party-local `Executor` now implement the approved positive
Q18 reciprocal, reciprocal square root, shifted softmax and centered LayerNorm.
Public row sums, rotations, broadcasts and per-lane affine coefficients share the
existing owner. Row geometry and coefficients enter the program digest. LayerNorm
currently accepts public gamma/beta; private parameter binding remains a separate
decoder integration requirement.

Reciprocal admits encoded inputs `1..2^36-1`; rsqrt admits `1..2^35-1`. A fixed
Boolean prefix circuit extracts the highest set bit without opening it. Public
XOR-linear maps derive normalization factors and rsqrt parity/compensation; the
existing narrow-OT or full-width PPA conversion returns arithmetic shares.
The original exponent, factors and every intermediate remain shared. Rsqrt keeps
sequential `U2`, `U3`, `U4`, one rescale of its accumulated polynomial and the
source's integer 724/512 compensation. Reciprocal keeps both source refinements.

Independent operand intervals alone would reject these graphs because the
normalization factor is inversely related to its input. Native construction
therefore partitions the **public input domain** into at most 36 exponent ranges
and propagates complete stochastic-rounding intervals within each range. Their
union certifies each unwrapped product before execution. The partition is proof
metadata, not a private runtime branch or disclosed exponent. Only these native
constructors can supply a correlated product envelope; ordinary public product
APIs retain Cartesian-box checks. Caller/compiler input-range obligations and
the quarter-ring truncation gate remain unchanged.

Softmax finds a shared row maximum, shifts before testing the strict `Z>-14`
boundary, evaluates all five squarings, masks clipped terms before reduction,
and broadcasts one reciprocal per row. Its maximum lane proves a positive
denominator. LayerNorm uses the encoded public reciprocal width, centered
mean-square variance, `E(1e-5)=2`, one rsqrt per row, and the approved two-step
gamma product. The initial slice uses a 128-total-lane tile and 256 KiB per-runtime
allowance; the wider-row extension below retains that byte ceiling. Unsafe eager
powers or oversized row geometry reject before PRSS.

The executor additionally accepts the independent PPA rotation arriving before
its masked reveal, retaining its charged buffer until the receive barrier.
The earlier executor retained registers until completion despite its last-use
description. This slice implements instruction-level retirement, including newly
produced unused registers, and prices the live-register peak plus primitive
overlap. Output ownership, failed transitions and cancellation retain the same
erasure-before-refund rule. These logical bounds are not process-RSS measurements.
Independent integer recipes check every supported normalization exponent,
clipping neighbors, non-power-of-two rows, constant rows and 128-lane geometry.
Cancellation checks cover every row-program instruction/protocol round; separate
allocation-index sweeps cover row owners and both Boolean conversion paths.
The normalization slice passed **105 native checks** (65 unit, 8
truncation, 3 PRSS, 3 exported multiplication/tensor and 26 allocation), five
compile-fail doctests and 33 independent Python profile/transcript checks.
Focused formatting and Clippy pass with the existing `needless_range_loop`
allowance. Unrestricted strict Clippy reports three pre-existing warnings in
`byte_dot.rs` and the separate Maverick batch implementation.

This is bounded native tensor execution. Compiler-derived whole-model ranges,
authenticated three-role transport and an ordinary SDK
Experiment remain open. Qwen's pinned RMSNorm epsilon is `1e-6`, which the
approved Q18 encoder maps to zero. D10 subsequently approves the explicit
positive-epsilon adaptation below; complete-model integration remains open.

### Explicit positive-epsilon RMSNorm

`Program::rms_norm` implements the separately approved
`puma.rmsnorm-epsilon1-q18.v1` profile. Its exact epsilon is `2^-18`, integer one
after mean-square rescaling. The graph squares each uncentered input, sums each
row, rescales its encoded reciprocal width, adds one, applies the existing shared
rsqrt once per row, then separately rescales normalization and public per-column
gamma. It has no centering, bias, clear zero test or private range selection.
Zero rows have a positive denominator and return exact zero shares. The public
input interval still must prove every intermediate's unwrapped range.

The profile name enters the native program digest and is available through
`numeric_profile()`. Source-derived activation, LayerNorm and softmax identities
retain their original values; an independently serialized pre-D10 digest checks
this compatibility. Geometry, gamma and the epsilon instruction are bound as
before. The initial slice uses the same 128-total-lane/256 KiB ownership limits;
the wider-row extension below retains the byte ceiling and numeric identity.

Independent stochastic-integer tests cover zero, constant, signed and small-code
rows, non-power-of-two widths and maximum tiles. The existing cancellation,
allocation-index and erasure-before-refund controls also exercise RMSNorm.
The scalar source oracle separately verifies the approved epsilon and zero-row
behavior. These bounded controls do not qualify checkpoint execution, private
gamma binding, whole-model range proofs or shared SDK quality measurements.

The D10 slice passes **107 native checks** (67 unit, 8 truncation, 3 PRSS, 3
multiply and 26 allocation), **5 compile-fail doctests** and **34 scalar-oracle
checks**. The focused Clippy command above and Rust formatting checks pass.
The combined integration command reached its 120-second outer timeout after the
allocation, multiply and PRSS suites passed; the eight truncation tests then
passed in a separate invocation. No performance measurements were taken.

### Public-coefficient half-split rotary program

`Program::rotary_half` composes the existing public row permutation, lane-affine
and standalone truncation instructions. Each public row supplies one cosine and
sine per pair of half-split coordinates, already encoded in Q18. For a pair
`(a,b)` and coefficients `(c,s)`, execution is:

```text
left  = TruncAPr(a*c, 18) + TruncAPr(-b*s, 18)
right = TruncAPr(b*c, 18) + TruncAPr( a*s, 18)
```

Each product retains its own fresh stochastic truncation. Coalescing them changes
low-code results and is not selected. The literal pair layout agrees with the
existing semantic executor's half-split rotary operator; this native entry point
does not generate or authenticate checkpoint/position coefficient tables. All
parties retain only their local replicated pair, with unchanged PRSS ownership.
Even widths, total lanes at most 128, coefficient magnitudes at most Q18 one,
complete input/product/sum bounds and both coefficient arrays precede execution.
The instruction digest binds row geometry, coefficients and the public range.

The PUMA library suite passes **69 tests**. Independent pairwise integer oracles
cover signed rotations, two 64-coordinate heads, maximum tiles and exact
identity-position results. Cancellation at each instruction/round and a separate
allocation-index/zeroization test pass; focused Clippy and formatting pass.
No serving benchmark or complete checkpoint RoPE mapping is claimed.

### Full-width bounded row execution

Row constructors now admit up to 1,024 total lanes before their per-primitive and
live-payload checks. The unchanged **256 KiB per-party cap** admits tested
896/1,024-value RMSNorm rows, 896-value centered LayerNorm rows and 150/288/512-value
softmax rows. Geometry alone does not guarantee admission: a 1,024-value softmax
rejects its expanded Boolean conversion before register allocation or randomness.
Activation and rotary tiles retain their 128-lane bound.

The public program maps non-overlapping lifetimes into reusable register slots.
Public coefficient tables use fallible, actual-length allocations; failed table
construction leaves the builder unchanged. Executor admission charges those
tables and the register table, preserving the charge through deallocation on
success, cancellation and constructor failure. Caller-owned program construction
and copies remain separate from an admitted runtime's ownership.

Scratch pricing distinguishes arithmetic, Boolean conversion and expanded OT
shapes. Consuming an already charged input moves its charge into the graph's
live-register envelope. Other retained owners, including an original OT frame
held by another party's output, retain their complete charge and can reject the
next program. No ledger reset or reservation refund substitutes for destruction.
Instruction order, stochastic rescales, public coefficients and existing program
digests remain unchanged.

Independent stochastic-integer oracles cover the wider rows. Exact-allowance
runtime tests cover consecutive normalization/softmax execution, maximum-width
OT and retained peer-output rejection. Register-reuse controls retain needed
inputs across unused instructions; allocator callbacks check wide-row erasure
and public-table charging through normal and failed retirement. The combined
verification passes **114 native checks** (71 unit, 29 allocation, 8 truncation,
3 PRSS and 3 multiplication), **5 compile-fail doctests** and **34 Python oracle
checks**. The three long allocation sweeps passed separately from the remaining
integration checks; focused Clippy retains the documented warning allowance.
These are bounded arithmetic and ownership checks, not process-RSS, checkpoint
quality, compiler-bound decoder execution or an SDK Experiment result.

### Independent primitive and transcript vectors

For a zero AES key, eight bytes at cursor 0 are `66e94bd4ef8a2c3b`; a **new**
eight-byte fill at cursor 1 is `47711816e91d6ff0`. By contrast, a single 16-byte
fill at cursor 0 continues with `884cfa59ca342b2e`. Tests also cover starts
2,255,256 and odd lengths. A separate 4128-byte fill at cursor zero checks the
**within-fill** big-endian carry: block 255 is
`f70ddef93ba62588242a0e67d0d645e0`; block 256 is
`fb56cc09b680b1d07c5a52149e29f07c`. These differ from new fills starting at
little-endian cursors 255/256. SHA-256 of all 4128 ciphertext bytes:
`b844e6a17d006cd080e7adcdf6d82be099aead7190912154e5358900dc3bdc04`.
These vectors came from standard OpenSSL encryption
of zero bytes with `-aes-128-ctr -K <key> -iv <LE128(cursor)>`, not upstream code.

Full fixed fixture: public contributions `(1,2,3)`, private keys `(101,102,103)`,
edge keys `(11,22,33)` as LE128. First public word is `5284552621516802438`,
so `(A,B,H)=(2,0,1)`. At n=3,b=3, input `(-9,1,2^62-1)` yields
`(-1,0,576460752303423488)`. `prss_tests.rs` freezes all helper correction and
output-share words from independent OpenSSL plus integer arithmetic, not merely
the reconstructed result. Odd/odd/even chained fills check exact cursors.

## Binding, retirement and bounds

`Operation` binds profile, session, complete schedule digest, request phase,
monotonic ordinal, pivot, shift and lanes. Each runtime starts at ordinal zero;
one operation is live at a time. Every frame checks binding, sender, recipient,
message phase and exact size before arithmetic. Typed frames are not an
authenticated wire codec. Future transport must authenticate roles, keep
payloads confidential, bound receive allocation and cancel on decode failure.

Any runtime transition failure permanently retires that runtime; the ordinal is
burned before payload allocation/material use. Cancellation/drop wipes retained
heap buffers. Entropy/allocation failure has no retry or fallback. The caller
must propagate cancellation to the other two runtimes, burn producer tickets,
discard already returned/in-flight frames and avoid releasing a partial request.
Local failure cannot remotely erase another process. Caller-owned input copies
and stack/register copies are outside the heap-wiping guarantee.

### Live payload accounting repair — 17 October 2026

The earlier per-operation estimate was **not a retained-payload bound**. The
independent audit found that a fixed-helper P2 operation at 1024 lanes can leave
32 KiB of original outgoing frames after output extraction. Nine retained
operations exceed 256 KiB; 4096 could retain 128 MiB. The earlier 31-test result
below did not establish a lifetime bound. This repair preserves the arithmetic,
source AES consumption, role custody and one-use retirement behavior.

Limits remain **1024 lanes, 4096 operations and 256 KiB per runtime**. One
runtime ledger enforces `live charged payload bytes <= max_bytes - 4096`:
**258048 bytes** at the unchanged default ceiling. The 4096-byte fixed reserve
covers owner/ledger metadata. This is a per-runtime payload contract, **not a
process-global or RSS cap**.

- Each runtime/source-PRSS dynamic word buffer acquires a reservation before
  its fallible allocation. The exact requested capacity is priced; an unexpected
  capacity is rejected before secret copying. Allocation failure refunds the
  unused reservation and retires the owner, with no cursor rollback or retry.
- Reservations live inside payload ownership, following original `Message` and
  `ReplicatedShare` buffers through return, output extraction, cancellation and
  runtime destruction. Erasure and real vector deallocation occur **before**
  refund. Dropping a runtime cannot forgive its exported original buffers.
- Outer imports (`from_local`, raw material constructors and `from_received`)
  are fallibly bounded per object. Their caller allocations need a separate
  outer policy until admission. Accepted original input buffers are charged
  before any PRSS expansion; imported material and incoming frames are charged
  before arithmetic or further owned allocations. Receiving an original frame
  keeps its issuer reservation and adds a receiver reservation until free.
  That deliberate overlapping accounting is not a process-wide memory sum.
- `payload_usage()` is a read-only, clonable ledger handle; it exposes charged
  live/peak/limit bytes, not secret data or reset/refund authority. Payloads,
  shares, seed messages and material remain non-cloneable.
- **Fixed-allocation exception:** one fixed-size `Arc` ledger allocation at
  runtime construction uses standard Rust's abort-on-OOM behavior. Stable Rust
  does not make that constructor fallible here. It fits the fixed reserve;
  cloning handles/reservations adds no allocation. Variable payload allocations
  use fallible exact reservations. Allocator overhead, caller containers/header
  storage, independent caller/transport copies and other runtime instances are
  outside this per-runtime payload cap and require their own outer bounds.

The existing shape-only conservative estimate `4096 + 128*lanes` is unchanged.
`Limits::role_working_bytes` additionally reports an operation's working set
including its current original outgoing frames: H uses at most **10 words/lane**
(2 input + 4 material + 4 outgoing); A/B use at most **8 words/lane** at opening
(2 retained state + 1 earlier outgoing + 3 incoming + 2 new reshare buffers).
These estimates do not replace lifetime accounting for earlier payloads.

At 1024 lanes, the fixed-helper regression retains six original frame pairs;
the seventh operation rejects and retires within the ceiling. Dropping retained
frames before exhaustion permits later fresh ordinals. Eleven retained output
pairs alone similarly block the next operation. Refund after failure never
revives the retired owner. Tests observe zeroed bytes and still-held charges
inside the allocator's deallocation callback, including after its system free.

Context/session freshness across restarts is enforced by fresh setup without
seed persistence; authentication and abort propagation belong to transport.

## Verification and remaining gates

```sh
cargo test --locked --offline -p pllm-core --lib puma::
cargo test --locked --offline -p pllm-core --test puma_truncation --test puma_prss --test puma_multiply --test puma_allocation
cargo test --locked --offline -p pllm-core --doc puma
python3 -m unittest discover -s paper/reproduction-contracts -p 'test_shared_profile_oracle.py'
python3 -m unittest discover -s paper/reproduction-contracts/puma -p 'test_*.py'
```

Historical pre-repair result: **31 tests passed** (13 unit, 8 truncation,
3 PRSS, 7 allocation). Its nine-file freeze, superseded by this audit repair:
`bafabbdd285eb4d9f5bd6f544bb17f870ebc9b50941af5c47760cefb2ceafaac`.

Repair verification: **37 tests passed** (14 unit, 8 truncation, 3 PRSS,
12 allocation); focused formatting checks passed. Allocation-index sweeps cover
all 14 role-specific `begin` allocation failures and four `open` failures after
earlier successful expansions, plus five success endpoints. They verify rollback,
erasure, transferred-input/incoming-frame pricing and sticky cursor retirement.
Retained-original-frame/output exhaustion, refund and owner-drop cases are covered.

**Repaired Rust inputs frozen for parent integration:** ten PUMA Rust files,
all `crates/pllm-core/src/puma/*.rs` and `crates/pllm-core/tests/puma_*.rs`.
SHA-256 of sorted `relative_path SP file_sha256 LF` rows:
`1b40c646243d300f217958e1db59a64c131e4d3ea277de5e1f345967d099ede2`.

Checks include 247,552 exhaustive small-ring equation/mask cases; actual ring64
party transcripts for all pivots, signed gap endpoints and wrapped masks;
exhaustive low-residue frequencies; replay/role/phase/context rejection;
cancellation, entropy/allocator failure and admission-before-allocation; source
AES/transcript vectors, seed direction/separation, stale setup rejection and
logical/AES-work cursor exhaustion.

P-N1/T-N1 adoption choices are closed for D6/D9's named contracts. Complete numeric
graph implementation, model-wide range proofs, authenticated wire transport,
mixed-kernel cursor scheduling, protected decoder/token loop, ordinary SDK
integration and independent original/common-model quality remain explicit gates. This
primitive supplies no composed-Qwen accuracy, security or performance certification.
