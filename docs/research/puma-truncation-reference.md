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
whole-client memory or full-checkpoint capacity result. Compiler regressions live
in `crates/pllm-compiler/src/replicated_decoder/`.

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

The original pinned Qwen2.5 4-input/two-output metadata check rejected the
compiler-view payload bound rather than the private-mode source-expansion bound.
It read no tensor values and launched no workers. Generated SDK controls and
complete integer matrix oracles establish this bounded adaptation only.

### Compact activation views and tensor loops

Activation references now use bounded owner/offset/stride spans. Slices,
broadcasts, attention permutations and partial tiles preserve their logical word
order. Gather commitments still hash that order, so the six frozen private-weight
execution identities remain exact. Construction admission prices the largest
live semantic frontier, retained state views, descriptor growth and temporary
copies; emitted programs keep their separate 64 MiB allowance.

The public-weight executor additionally uses bounded 128-lane loops for private
lookup predicates, complete-product truncation and SiLU. A loop retains one
main program, an optional tail program and its complete output owner. Every
iteration creates a fresh consuming executor on the advancing PRSS runtime.
Source owners remain live through the final iteration; replay, cancellation or
failure burns partial outputs and the runtime. All iterations count toward the
existing whole-program work limit. Public execution commitments bind the loop
shape, operands, programs and tail. This changes the public execution schedule,
while retaining the approved numeric operations and final-only disclosure.

The original loop/view Qwen metadata check cleared the view-memory gate, but
private lookup alone exceeded its declared-work allowance. Native source admission
checked that necessary lower bound before checkpoint value reads or worker launch.
At four inputs and two outputs, lookup alone charged 21,878,784 units against
1,048,576 allowed. This is a lower bound on the compiler's conservative work
charge, not a timing or cryptographic-operation measurement. Whole-request work
qualification, complete-model range admission, quality and matched resource costs
remain gates. These storage/scheduling changes establish no whole-client memory
or full-checkpoint serving result. Installed-wheel validation passes 245 Python
and 20 native checks, with seven optional skips and all 367 package files matching
before and after.

### Schedule-derived work admission

The native tensor program now derives separate sequence and PRSS budgets from
its immutable instruction schedule. Sequence counts measure consumed top-level
ordinals, including local arithmetic. They do not count messages, Boolean rounds
or CPU work. Public/pair cursors are exact; private draws and AES expansions use
per-party upper bounds across every public pivot. Each AES fill rounds its own
tail. Signed conversion backing widths, OT and full-width Boolean conversion
retain their actual nested schedules.

The compiler sums every node, full loop iteration and tail, including every D9
prefix. Tensor admission checks the remaining sequence and AES allowance, plus
cursor overflow, before allocation and randomness. The original 1,048,576-ordinal
and 3,224,371,200-AES-block complete-program ceilings remain independently enforced;
a wide conversion can use more AES blocks than the former per-ordinal heuristic.
Primitive payload ceilings, source ranges, fresh state and execution digests stay
bound to their existing contracts.

One 128-lane public lookup predicate consumes 28 ordinals, with at most 2,817 AES
blocks per party. At Qwen's 151,936 vocabulary rows and four inputs/two outputs,
the lookup-only floor is 299,124 ordinals and 30,094,011 AES blocks per party.
That metadata gate admits. At sixteen inputs/two outputs, its 1,096,788 lookup
ordinals reject before tensor reads. The private-weight source-expansion gate
remains separate. Metadata admission does not qualify complete-model ranges,
whole-request work, checkpoint execution, quality or serving resources.
The work-budget slice's ordinary `benchmark_memory` retry reached the shared
Safetensors reader's private-weight header limits. That rejection remains in
the historical work-budget attempt.
The installed-wheel checks pass 245 unique Python cases and 144 native checks,
including five ownership doctests, with seven optional skips.

### Native-derived source-header bounds

The shared source reader now derives maximum dimensions and element counts from
the admitted native source shapes, after checking those shapes against the
semantic plan. It then validates exact checkpoint shapes, floating-point dtypes,
contiguous byte ranges, tensor cardinality and shard ownership before value reads.
File identity, source locks, header/metadata ceilings and bounded conversions
retain their existing contracts. GKM and private/public PUMA use the same reader.

A sparse generated 8,193-by-128 token table crosses both old private-header
limits. F32, F16 and BF16 checks preserve all converted float32 bits, signed zero,
frame boundaries and the last partial frame. Changed shapes, dimensions, dtypes,
ranges and missing/extra tensors reject before source hashing or value decoding.
The original failing installed-wheel regression remains recorded.

The separate pinned Qwen 4+2 `benchmark_memory` retry now admits its complete
source-present topology. Its 2,348,810,240-byte allocation estimate includes a
25% margin; required disk is 14,004,270,080 bytes. Source hashing and headers are
checked, but this preflight does not decode tensor values or launch workers.
Complete-model numeric/program admission and checkpoint execution remain open;
these estimates are not measured RSS, quality or serving costs. The original
[standard memory-preflight output](../../paper/artifacts/puma-sdk/local-v7/documented-preflight-replay.json)
retains this result.

### Public range certificates and semantic admission diagnostics

The source publisher now reports the selection, semantic step/operator/layer and
public input intervals when native program construction rejects. These intervals
come from the locked checkpoint and public geometry, not sampled activations or
private token values. Publication failure retires its incomplete owner.

Public matrix descriptors additionally retain the maximum encoded absolute row
sum and maximum encoded squared row norm. Publisher and each importing worker
compute both with checked integers across the original bounded source frames.
The worker compares the complete statistics before accepting a tensor. An
immutable source-descriptor commitment binds cached file owners to their ranges
and certificates as well as the original coefficient digest; changing metadata
cannot reuse an owner validated under another declaration. Private-weight
descriptors omit these fields.

The public-mode compiler uses a finer RMSNorm proof over the same Q18 operations.
It partitions the normalized rsqrt polynomial's public integer domain into
256-value cells and unions their floor/ceil envelopes. Every normalized integer
and all sixteen power/final-rescale rounding choices are checked by an independent
integer oracle with different cell boundaries. Partitions select no runtime
branch and add no protocol messages or randomness.

For each public denominator exponent interval, the proof also bounds the complete
input row. With `Q = 2^18`, `S = sum(TR(x_i^2))`, width `n` and positive public
`c = E(1/n)`, the original mean and epsilon give

```text
d = TR(S*c) + 1
S <= floor((d_max*Q - 1) / c)
sum(x_i^2) <= Q*S + n*(Q - 1)
```

The certified reciprocal-square-root interval bounds the row product. Each
normalization and public-gamma truncation adds less than one integer unit per
lane; upward-rounded Euclidean bounds charge those errors separately. This
produces a squared-norm certificate for each complete RMSNorm output row.
Zero, near-zero, constant, signed, sparse, wide and multiple-row integer controls
check the certificate and the unchanged stochastic arithmetic, work and payload.

Public matrix products intersect their ordinary interval with
`max_abs(input) * max_row_l1(weight)`. For an intact certified RMSNorm row, they
also intersect with the upward-rounded Cauchy–Schwarz bound from the two squared
norms. Duplicated, incomplete, mixed-owner or differently sized views cannot
inherit that certificate. Complete ring64 accumulation and its single Q18
truncation retain their original limits and arithmetic. Private execution keeps
its six frozen generated identities; public proof metadata changes its checked
execution identity.

The pinned Qwen2.5 4+2 source publication initially rejects layer-zero attention
scores. Scalar/row-sum tightening alone still rejects there. Complete row norms
admit that product and reach layer-zero `Softmax`, which rejects its eagerly
evaluated exponential schedule under the public input interval. No role workers
or online decoder are launched. All retained publication attempts observe zero
swap growth; their process peaks and CPU are admission diagnostics, not serving
metrics. Whole-model range/work admission, quality and matched response costs
remain open.

### Guarded shared softmax tails

The public-weight schedule uses `Program::softmax_guarded` arithmetic. It computes
the original shared row maximum and shifted score `z`, then the same strict
predicate `b = [z > -14*2^18]`. One exact, untruncated shared product sets
`w = z*b` before the five exponential squarings. The native proof bounds `w`
between `max(public_lower(z), -14*2^18+1)` and zero. It derives this bound from
the predicate equation; no private score, bit or exponent is opened.

Retained lanes have `w=z` and keep the original five-bit rescale, constants and
five sequential Q18 square/truncation operations. Clipped lanes instead evaluate
the bounded zero input and retain the original exact zero output after the final
`b` mask. Row reduction, reciprocal and final normalization remain unchanged.
The shared predicate and unwrapped product still require public range admission.
The original eager constructor and private-weight execution retain their identities.

The changed public schedule and extra product bind the program digest and add to
sequence/AES pricing. Every invocation consumes fresh party-local state. Tests
replay each schedule's own truncation tape against independent source equations;
the changed PRSS cursors retain distinct transcripts. Integer controls cover
strict-cutoff neighbors, constant and
wide-range rows, non-power-of-two geometry and 512-value rows. Cancellation and
allocation-index controls include the guarded schedule under the original 256 KiB
party ceiling. Generated Qwen2/Qwen3/Llama sources with nonzero wide Q/K execute
private feedback; the Qwen2 control also traverses three authenticated SDK workers.
Their V/O/MLP weights stay zero, so a separate signed-head oracle checks feedback
while the primitive oracles check complete softmax values. Each generated
regression rejects softmax on the previous installed wheel.

The guarded-softmax slice's pinned Qwen2.5 4+2 publication clears the first
softmax. It rejects layer-zero post-attention `RmsNorm` at semantic step 22,
with public encoded input
interval `[-1571860067, 1571850851]`, before role-worker launch. This guarded source
publication observes zero swap growth. Whole-model ranges, complete-program work,
hidden-state parity, quality and response costs remain open.

### Complete attention-row and residual-energy proofs

Public-weight compilation now selects `Program::softmax_certified`, retaining
the guarded instructions, truncations, PRSS work and party payload bound. The
additional proof partitions each public denominator exponent into at most 256
integer cells. It propagates every floor/ceil outcome through the original reciprocal
recipe. For nonnegative exponentials `e_i`, denominator `d = sum(e_i)`, shared
reciprocal `r`, and `Q = 2^18`, it certifies

```text
p_i = TR(e_i*r)
sum(p_i) <= floor((upper(d*r) + n*(Q-1)) / Q) = M
```

Partitions do not inspect a score or select a runtime branch. The compiler binds
`M` to the complete ordered softmax row. An attention product with value interval
`[L,U]` then lies in `[M*min(L,0), M*max(U,0)]` before its one final rescale.
Ordinary product and quarter-ring admission still run first. Partial, duplicated,
reordered or mixed-owner views cannot inherit the row certificate.

Public source descriptors also bind `row_l1_squared_sum`, the sum of squared
absolute row sums over a complete weight matrix. The publisher and each worker
derive it from the original streamed Q18 coefficients with checked integers.
Worker import rejects both understatements and overstatements before accepting
the tensor; source-owner commitments reject metadata substitution. Private-weight
descriptors omit the statistic.

For a non-transposed public product, let `A` bound the input's absolute encoded
values and `E = sum_j[(sum_k(abs(W_jk)))^2]`. The complete output row after its
original Q18 rescale has Euclidean radius at most

```text
R = ceil(A*ceil_sqrt(E) / Q) + ceil_sqrt(output_width)
```

The second term charges every stochastic truncation error. Residual addition
combines complete-row certificates by the triangle inequality; uncertified rows
use their scalar bounds. Certificates retain exact ordered word provenance across
tensor chunks. Metadata owners count against the existing program-byte ceiling.

`Program::rms_norm_with_input_norm` accepts that public squared-energy obligation
`B` for an intact row. Its unchanged square/reduction schedule uses

```text
sum(TR(x_i^2)) <= floor((B + n*(Q-1)) / Q)
```

before the original reciprocal-width rescale and epsilon. This avoids assigning
every coordinate the largest coordinate's magnitude. The native primitive, like
its scalar interval, requires a caller-proved bound; it does not reconstruct
private values to check the obligation. The compiler obtains it only from bound
source statistics and admitted operations. No limits, polynomial constants,
rounding rules or protected work budgets change.

Independent stochastic-integer controls check complete softmax and RMSNorm values,
row certificates, 896-value sparse normalization, cancellation and allocation
failure. Generated Qwen2/Qwen3/Llama controls exercise nonzero V/O products,
concentrated output energy and private feedback; Qwen2 also uses three SDK workers.
Their zero Q/K and MLP provide an independent signed-head oracle rather than a
checkpoint-quality result.

Mass alone reduces the pinned Qwen post-attention scalar interval to
`[-145907592, 145898376]`, but RMSNorm still rejects. The complete residual-energy
proof admits it. Pinned 4+2 source publication next rejects layer-zero `Silu`,
step 25 of selection zero, at encoded interval `[-28618714, 28618714]`. No role
workers or online decoder run; retained source-publication attempts observe zero swap
growth. Whole-model ranges, total work, hidden-state correctness, quality and
serving measurements remain open.

### Guarded piecewise activations

Native `Program::activation_guarded` implements SiLU/GeLU; public-weight decoder
compilation selects its SiLU program.
The existing strict lower/middle and inclusive upper comparisons remain shared.
For their original encoded cuts `a0 < a1 < a2`, it computes

```text
c0 = [x < a0], c1 = [x < a1], c2 = [x < a2+1]
b = c2-c0
t = x*b
```

The fresh ring product `t` has no truncation: `b` is exactly zero or one.
Both original polynomials evaluate on `t`; their coefficients, powers and Q18
stochastic rescalings stay unchanged. The original final selections retain
zero below `a0`, the appropriate polynomial within `[a0,a2]`, and the original
`x` above `a2`. At retained points `t=x`. Tail polynomials instead receive zero,
and their discarded values cannot affect the final zero/identity output.
This is private branch evaluation, not clipping an accepted activation or checking
a caller-supplied range at runtime.

The public output proof partitions each polynomial's integer domain into at most
1,025 cells and propagates all possible floor/ceil outcomes. These cells add no
runtime selection, PRSS use or private-data inspection. Public schedules bind a
changed instruction digest and charge one additional fresh product. Eager
activation primitives and private-weight execution remain available with their
original identities. The existing 128-lane primitive and 256 KiB party limits
continue to apply.

Independent stochastic-integer controls cover both activations, each endpoint
and adjacent encoded values, wide signed tails, multiple PRSS tapes, exact output
intervals, cancellation and work cursors. Allocation-index sweeps and zeroization
audits include both guarded programs. Generated Qwen2/Qwen3/Llama controls exercise
wide gate values and private feedback; the Qwen2 case also runs three authenticated
SDK workers. Their signed-head oracle is independent of the polynomial evaluator.

Pinned Qwen 4+2 source publication now clears layer-zero SiLU and MLP. It rejects
selection-zero, step-29, layer-one `RmsNorm`, at encoded residual interval
`[-343639410012,343639400796]`. The source-publication attempt observes zero swap
growth and launches no role workers. Source-only whole-model range proofs,
complete-program work, full-checkpoint hidden-state correctness, source-model
quality and matched response costs remain open. Protected runtime range-check
acceptance would require a separate contract; this slice does not select it.

### Conditional FFN row-energy theorem—inactive

This is a conditional numerical theorem, not an activated certificate or an
admission result. Astra's theorem was independently reviewed by
`ses_e73031d70ffeuG3zuFUrz6tEOw`: the conditional numerical proof is
**SOURCE-CLEAR**, with no formula mismatch. Actual numerical admission,
certificate activation, publication, funding and a 24-layer fit remain
unestablished. No numerical caps or profile changes are adopted here; the
historical results above and Native **HOLD** remain unchanged.

**Inputs and source boundary.** Set `Q = 262144`, let `E(x) = sum_i x_i²`
for encoded integers, and write `c(t) = ceil(sqrt(t))`. A seed `(B,M)` means
`E(x) <= B` and `|x_i| <= M` for one complete, correctly ordered row. The
actual normalized `post_norm` row has seed `(Bx,Mx)`, while the original
`attention_residual` has the **distinct** seed `(Br,Mr)`. The
[model graph](../../crates/pllm-models/src/lib.rs#L2027-L2092) feeds the former
to gate/up projections and adds the latter after the down projection; they must
not be substituted for one another. For a width-`n` row, `n*M²` is a valid
coordinate-derived energy seed; a sharper seed is optional but needs its own
authorized provenance.

#### Conditional seed provenance from existing owners

Existing owners already supply conditional seed authority; no new statistics
map or witness is needed. [Normalization ownership](../../crates/pllm-compiler/src/replicated_decoder/compact_norm.rs)
retains original gamma, checks complete ordered rows and publishes actual native
output bounds. [Compact lowering](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs)
requires native `GeneralOutputRowSquared`, and
[retained validity](../../crates/pllm-compiler/src/replicated_decoder/compact_validity.rs)
exposes authentic `output_energy`. Thus set `Bx = Bnorm` from that retained
factory bound and `Mx = max(unsigned_abs(lo), unsigned_abs(hi))` from its actual
published output interval—not the input guard cap. No optional integer square
root, new tightening or multiplication is needed to form this normalized seed.

This authority concerns the [exact rounded RMS recipe](../../crates/pllm-core/src/puma/tensor_numeric.rs):
float32 mean coefficient, integer-one epsilon and the actual selected polynomial
inverse, not ideal inverse square root, unit energy or guessed model gamma.
Preceding arithmetic must already be admitted before energy refinement. Validity
checks common input bounds and energy across every guarded row; invalid lanes
become zero, while valid lanes continue unchanged after sticky validity is false.
General `Bnorm` may conservatively cover fallback paths without sticky success.

The distinct original `Op::ResidualAdd` owner is
`PublishedKind::Binary { product: false }`: it retains the original output
interval, equal logical shape and `Products` mapping with ordered 128-lane tiles. Retained validity
also exposes its authentic binary `row_energy`. Take `Mr` from that **original**
published interval and `Br = n*Mr²`, checked, or an already genuine general-row
bound. Later guarded/normalized output cannot replace this residual seed.
Here `n` is semantic last-axis width from the complete actual ordered row binding,
not a caller label. Squared magnitudes from `i64` endpoints fit `u128`;
`n*M²` and every later evaluated theorem operation must still be checked to fit.

Both seeds are available only after successful outer `lower_next`, including
`retire_phase` and `EmissionAttempt` commit, while the original context is open.
Inner emit/push, `validate_view` or `!failed` alone is insufficient.
These are authority preconditions, not checks enforced by `row_energy_for`:
its RMSNorm/Binary arms return retained fields without independently authenticating
context, owner, ordered-row binding or completed outer publication. This closes
seed provenance conditionally, not actual trained-value or factory admission,
downstream FFN publication, work funding, activation or native proof. Original
source `R`/`S` import authentication remains a separate closed boundary.

#### Projection and propagation bounds

For each genuine original, non-transposed matrix `w` with output rows `j`, use
`R = max_j sum_i w_ji²`, `S = sum_j (sum_i |w_ji|)²`, and post-rescale bias
absolute maximum `J` (zero when absent). These are the existing source statistics,
not a new map: the [source scan](../../crates/pllm-compiler/src/replicated_decoder/source.rs#L526-L589)
derives `R` and `S` from encoded original rows, and the existing importer
authenticates them. Transposition, incomplete rows, reordering, duplication or
mixed owners cannot inherit this proof without a separate justification.

Write `T(z)` for either `floor(z/Q)` or `ceil(z/Q)`, including signed `z`;
it is exact when `Q` divides `z`. Every raw interval sent to a rescale must
independently satisfy `-2^62 <= lo <= hi < 2^62`, as required by
[`rescaled_bounds`](../../crates/pllm-core/src/puma/arithmetic.rs#L135-L145).
The complete dot is accumulated before its **one** rescale, never rounded per
chunk: see the [public product](../../crates/pllm-compiler/src/replicated_decoder/program.rs#L1666-L1705)
and [complete-dot chunk path](../../crates/pllm-compiler/src/replicated_decoder/program.rs#L1816-L1848).
Bias is added only afterward, as the [projection driver](../../crates/pllm-compiler/src/replicated_decoder/graph_driver.rs#L350-L365)
and its [bias implementation](../../crates/pllm-compiler/src/replicated_decoder/graph_driver.rs#L625-L633)
confirm.

**Projection lemma.** For `m` outputs and an independently valid postbias
coordinate magnitude `Y`, define

```text
F = min(m*R, S)
D = min(M²*S, B*F)
P_m(B,M,R,S,J,Y) = min(m*Y²,
    [ceil(c(D)/Q) + c(m) + c(m*J²)]²).
```

Indeed, the raw dot vector `q` satisfies `sum_j q_j² <= M²*S` by rowwise
absolute sums. Cauchy–Schwarz gives `sum_j q_j² <= B*sum_ji w_ji²`;
the squared Frobenius norm is at most both `m*R` and `S`. Thus `E(q) <= D`.
After division by `Q`, each stochastic rounding error has magnitude less than
one, so its row radius is at most `c(m)`. The post-rescale bias radius is at
most `c(m*J²)`. The triangle inequality, upward integer roots and the independent
coordinate cap give `P_m`. The checked symbolic implementation is
[`RowEnergyWorkspace::projection`](../../crates/pllm-compiler/src/replicated_decoder/row_energy.rs#L53-L123).
Apply it separately to obtain

```text
Eg = P_m(Bx,Mx,Rg,Sg,Jg,Yg)
Eu = P_m(Bx,Mx,Ru,Su,Ju,Yu).
```

**Guards and selected SiLU.** A lane guard `G_L` keeps `x_i` exactly when
`|x_i| <= L` and replaces only invalid lanes with zero. Valid lanes remain
unchanged even if prior sticky validity is false. Hence `E(G_L(x)) <= E(x)`;
the [range lowering](../../crates/pllm-compiler/src/replicated_decoder/ranges.rs#L65-L121)
reduces validity separately into the sticky bit. Arithmetic failure instead
refuses or cancels execution; it is not a zero-valued continuation.

The [guarded activation recipe](../../crates/pllm-core/src/puma/activation_work.rs#L119-L210)
selects zero below `-8Q`, the low polynomial on `[-8Q,-4Q)`, the high polynomial
on `[-4Q,4Q]`, and the original gate `g` above `4Q`. It evaluates both
polynomials on the guarded input and powers

```text
z = g * indicator[-8Q,4Q](g)
p2 = T(z²), p4 = T(p2²), p6 = T(p2*p4)
low  = -80413 + T(-21489*z) + T(-1453*p2)
high =   2229 + T(131072*z) + T(59806*p2)
              + T(-2913*p4) + T(71*p6).
```

Each coefficient product has its own rescale. On the selected low domain,
`|z| <= 8Q`, `p2 <= 64Q`, and the absolute output is at most
`80413 + 21489*8 + 1453*64 = 345317`. On the selected high domain,
`|z| <= 4Q`, `p2 <= 16Q`, `p4 <= 256Q`, `p6 <= 4096Q`, giving
`2229 + 131072*4 + 59806*16 + 2913*256 + 71*4096 = 2519957 = C`.
These integer upper bounds cover every rounding choice because the corresponding
extremal products are divisible by `Q`. Branchwise, the selected result `a_i`
therefore satisfies `a_i² <= g_i² + C²`, including the identity and zero tails.
For an independently valid selected-output cap `A`,

```text
Ea = min(Eg + m*C², m*A²).
```

In particular, the cap must include `SiLU(0) = 2229`. The selected bound `C`
is **not** an eager-intermediate bound: the eager input can reach `|z| = 8Q`,
so `p2 <= 64Q`, `p4 <= 4096Q`, `p6 <= 262144Q`, and the raw `p2*p4`
can reach `2^54`. Every eagerly evaluated polynomial, selection and affine
intermediate still needs its own interval admission, even when later discarded.

**Paired product, down projection and residual.** For the actual ordered lane
pairing of selected SiLU `a` and up projection `u`, each of
`A²*Eu`, `Yu²*Ea` and `Ea*Eu` bounds `sum_i a_i²*u_i²`: the first two use
coordinate caps, and the third follows by retaining the diagonal terms in the
product of the two nonnegative sums. No independence or energy-spreading
assumption, and no division by width, is permitted. With an independently valid
post-product magnitude `H`, the same rounding-radius argument gives

```text
K = min(A²*Eu, Yu²*Ea, Ea*Eu)
Eh = min(m*H², [ceil(c(K)/Q) + c(m)]²)
Ed = P_n(Eh,H,Rd,Sd,Jd,Yd)
Ev = min(n*V², [c(Br) + c(Ed)]²)
Ein = min(Ev, n*L²).
```

Here `n` is the down/residual width, `V` is an independently valid magnitude
of the actual residual sum, and `L` is the next RMSNorm guard cap. The residual
triangle inequality uses the original `Br`, not `Bx`; the final guard can only
reduce energy. These are the activation, product and residual inequalities in
[`RowEnergyWorkspace`](../../crates/pllm-compiler/src/replicated_decoder/row_energy.rs#L89-L123),
not authority to publish a certificate.

**Unclosed arithmetic obligations.** Every evaluated checked-`u128`
intermediate must be at most `U = 2^128-1`, including alternatives discarded
by `min`. This includes, for every projection, `m*R`, `M²`, `M²*S`, `B*F`,
`J²`, `m*J²`, `Y²`, `m*Y²`, both radius additions and the radius square;
for activation, `C²`, `m*C²`, `Eg+m*C²`, `A²`, `m*A²`; for the paired product,
`A²`, `A²*Eu`, `Yu²`, `Yu²*Ea`, `Ea*Eu`, `H²`, `m*H²`, its radius addition
and square; and for the residual/guard, `c(Br)+c(Ed)`, its square, `V²`,
`n*V²`, `L²` and `n*L²`. Root/ceiling helpers and all other evaluated proof
arithmetic must likewise remain representable; a small final minimum does not
excuse overflow along another evaluated path.

RMSNorm timing is source-confirmed: the
[shared normalization recipe](../../crates/pllm-core/src/puma/tensor_numeric.rs#L839-L871)
first intersects the coordinate interval using energy, **before** squaring,
then constructs the ordinary square and row sum **before** refining their energy
bound. Using the conservative symmetric envelope `[-t,t]`, where
`t = min(floor(sqrt(Ein)), L)`, sufficient bounds for these early admission
checks are

```text
t² < 2^62
n*ceil(t²/Q) <= 2^63-1
Smax = min(n*ceil(t²/Q), floor((Ein + n*(Q-1))/Q))
cn = trunc0((1.0_f32 / (n as f32)) * (Q as f32)) > 0
cn*Smax < 2^62
Sq = sum_i T(x_i²)
d = T(cn*Sq) + epsilon
0 < dmin <= d <= dmax < 2^35.
```

The energy numerator `Ein+n*(Q-1)` and its constituent products/addition must
be checked, as must the other evaluated bound arithmetic. The
[mean coefficient](../../crates/pllm-core/src/puma/tensor_numeric.rs#L945-L950)
uses the source's float32 reciprocal/multiply conversion, not an exact rational
replacement; `epsilon` remains the existing integer-one RMSNorm epsilon.
Every remaining rsqrt, normalized-product and gamma-product interval must
independently satisfy the signed raw rescale domain, and every affine/selection
intermediate must fit signed `i64`. The displayed inequalities are unclosed
obligations for actual authorized seeds, not premises supplied by an assumed
successful request. No trained constants or trained certificates are claimed.

This theorem establishes only conditional energy arithmetic. It supplies no
authenticated publication of postbias `Products`, quote-entry funding,
production wiring, native qualification or trained-model usefulness. Integration
still lacks actual model-specific numerical seed success, not reusable seed
authority. Postbias `Products` consumption and complete publication remain open;
neither relabeling `Products` as `Dense` nor removing bias's intentional row-energy
clearing establishes provenance or funding. Existing bias row-energy clearing
must remain intact pending approved mapping-aware publication. Integration
still requires separate approval and funding, with the existing source-retirement
boundary preserved. It does not establish complete-model admission or change
Native **HOLD**.

### Prospective compact quote-entry boundary

This is shared compiler design, not PUMA-specific runtime architecture or an
implemented type change. Proposed private `Hold<'ctx,'src>` contains only
`context: &'ctx EmissionContext<'src>` and `controls: usize`;
`Frame` contains `hold`, `error: Option<Error>` and `ready: bool`. Neither owns
a reservation nor binds graph/op/input. Proposed private `Entry<'e,'g,'s>` is
`{ graph: &'e CompactGraph<'g,'s>, op: &'e ModelOperation,
input: &'e PublishedView, logical: ViewShape }`. Context is a copied external
`EmissionContext` reference; Hold/Frame confer no reservation authority.
Frame/Entry live only in the admitted contraction branch at **both**
`lower_next_inner` call sites (`4655–4662`, `4718–4719`), through consumer return,
not publication; no outer `Option<Frame>`. `lower_contraction` takes `&self`, so
shared Entry borrows may survive that call but end before outer graph mutation
or publication. If the consumer becomes `&mut self`, those borrows must end first.
The existing `None` path, pre-arm validation/admission, outer `lower_next` wrapper
attempt and nested admission attempts remain unchanged; one proposed baseline
call applies only to this new stage, not the whole route.

Private straight-line initialize/check/admit/consume runs once with unchanged
arguments, without callback/reentry: initialize `ready=false`, `controls=0`,
`error=None`; check identity and zero control cell; obtain type-only selected
layout plus existing extras/conditional bias; compute baseline and call existing
`admit_local(baseline + controls)`; immediately recheck zero and install once;
consume, constructing `ContractionView::new(input.clone(), ...)` without requoting;
reset the existing `ContractionControls` guard, then end the arm. Refusal keeps
`ready=false`. The original caller `input.clone()` remains baseline. Baseline
includes installed controls exactly once; the conservative sum is **not
admission-neutral** and may refuse formerly valid requests. Exact selected static
layouts are not adopted. Current guard reset at consumer return does not extend
through outer publication, successful `retire_phase` or cleanup; those remain
separate funding/publication obligations. See
[contraction entry](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L2703-L2709)
and [outer publication](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L4866-L4897).

Production `contraction_control` still constructs `S::tokens`, inferring its
actual type for shared `size_of::<I>()` arithmetic. Its anonymous-closure compact
RPIT cannot support this proposed prequote. Approved source/test prerequisites
preserve that iterator; named `CompactTokens`, factory and prospective helper stay
`#[cfg(test)]`, not authority. No closure/function-pointer shortcut is assumed.
Layout compatibility requires authorized execution/review and is target/compiler-
specific, not portable proof; production iterator adoption remains unauthorized.

Provisional **entry-only** schema: let `R = size_of::<Entry>() + size_of::<Frame>()`;
the branch envelope is `caller_live + R + max(A, B, C + layout_body,
D + admission_body, E + consumer_body, refusal + cleanup_body)`. R is present
in every branch. A–E represent only incremental, actually simultaneous staging,
parameters and results, not independent maxima or already embedded fields.
A is initialization, B checks, C quotation, D admission, E installation/consumption.
Separate construction moves need an actual source census. Existing baseline
already includes four `PublishedOp` slots and installed controls: reconcile those
before claiming a disjoint baseline. This is not funded whole-entry/request proof.
For that envelope, maximize **whole simultaneous branch sums**, including retained
baseline and disjoint issuer/caller/entry/callee/publication/cleanup terms; do not
combine independent maxima or take duplicate credits.

The root guard chain is `Pending.controller -> consume_compact -> emit ->
EmissionScope::new -> EmissionContext.guard -> OwnedEmission.guard`. Only after
authority/census approval could the existing quote be enlarged before factories;
no new guard/ledger/ticket/map is proposed. Current pending quote is exactly
`3 * size_of::<Pending>() + size_of::<SourceRequestImport>() +
size_of::<Result<SourceRequestImport, Error>>() +
size_of::<Result<SourceRequestProgram, Error>>()`. `consume_compact`'s
`max(old, context_bytes + metadata_controls)` capacity is not additive pending/
context overlap proof; direct-storage ingress also needs coverage. Preserve the
controller, retained floor and payload-before-charge-drop order. Field order alone
does not prove the complete envelope. See
[source entry](../../crates/pllm-compiler/src/replicated_decoder/source_entry.rs#L828-L838).

Metadata work must debit the retained original ledger through
[`OriginalSourcePlan::prepay_compiler_metadata`](../../crates/pllm-compiler/src/original_source.rs#L739-L786),
not `EmissionContext.work` protocol/AES accounting. The `shared_allocation` counter
root is created at `source_entry.rs:833` and reserved at lines 836–838, distinct from
`OriginalSourcePlan` becoming available at line 938. Its first explicit metadata DTO
debit at lines 964–969 follows JSON/config/policy/quote work. Neither setup is funded
by that later debit, nor is either an established unpaid exception.

The ten source audits establish the following boundaries, not broader proof.
Source-level counts describe operations or visits, not adopted tariffs, CPU cost,
physical-stack bounds or whole-entry funding.

| Boundary | Source-established | Still open |
| --- | --- | --- |
| Driver quote | Both computed layouts use 14 checked projection operations plus 22 lookup operations: 36 on success; refusal follows an ordered prefix `j`. Production constructs `S::tokens` once and calls `Iterator::next` zero times. | Factory/body cost; retained `I` plus a separate bridge-reference parameter versus embedded `2 * size_of::<I>()`. This is not a type-only production path. |
| Metadata debit | Direct primitive prepayment has six capacity-helper additions, with prefix refusal. Early failure leaves state unchanged; overflow/limit failure after burning is sticky. Successful debits, including zero, reset only this call's temporary burn; previously burned ledgers reject. | Bootstrap and repeated zero/burned prefixes; no enforced entry fee, attempt count or one-shot invocation. |
| Baseline/attributes | Import/norm/op/nested-resource scans repeat; native prefix-resource visits can be quadratic. The bias consumer scans the source prefix, `L` indices and `ceil(L/128)` chunks. | Complete visits, helpers and refusal work; the factory bounds and pre-JSON root-prefix limits below do not establish funding. |
| Ingress/handoff | A moved-from, empty `SourceRequestImport` still embeds `Pending` capacity. History overlaps a second graph; the returned result overlaps registry/catalog/origin. Direct ingress bypasses `InputReceiver`; the original payload lease transfers. | Assembly/move/catalog coverage and factory/feed/drop costs; a maximum alone is not additive overlap coverage. |
| Context/consumer | The actual `EmissionContext` is involved. Phase admission creates an attempt; local admission only checks capacity. Phase retirement has no metadata debit; the consumer guard resets on return. | Initial descriptor/origin/registry scans versus the declared `N` debit; nested/conversion/native/refusal work and attempt drops. |
| Caller/publication | Contraction arms retain input through consumption. Publication moves input/kind/prepared and clones two outputs. The outer result lives through retirement; retirement errors bypass diagnostics/commit. | Live-slot/transfer/drop sums and diagnostic/publication/cleanup funding; count embedded fields and the existing four-`PublishedOp` allowance once. |

Here `Value` attributes use `BTreeMap`; the four-byte bias query bounds comparison
prefix length, not comparison count. Arbitrary unused keys are a semantic-validation
counterexample, not proof that the authorized factory route is unbounded:
`OriginalSourcePlan` invokes the registered factory, not a caller-supplied plan.
No silent attribute limit is introduced.

Next gates are evidence-backed accounting for finite events, tariffs, invocation and bootstrap, plus
complete whole-branch sums reconciled with baseline and controls. Remaining owners
include caller/callee, refusal, publication and drop. The logical tariff remains
**UNADOPTED**; selected totals do not prove workspace/funding equivalence. No new
wrapper/root funding, production iterator adoption, energy integration, authoring
or execution is authorized. Propagation stays unchanged. The layout prerequisite
tests remain **UNRUN**; Native **HOLD** remains unchanged.

#### Current authenticated, untransformed factory: source bounds

These bounds cover only the current authenticated, untransformed factory route:
[`source_entry` 929–938](../../crates/pllm-compiler/src/replicated_decoder/source_entry.rs#L929-L938)
through [`original_source` 1225–1258](../../crates/pllm-compiler/src/original_source.rs#L1225-L1258),
[`source_work`](../../crates/pllm-models/src/source_work.rs) and `ConfigLowererQuote`.
[`source_mappings`](../../crates/pllm-models/src/source_mappings.rs#L383-L457) admits
qwen2/qwen3/llama. For `L` layers, the shared graph has `28L + 6` operations per
phase, plus `2L` Qwen3 QK-norm operations: both phases are bounded by `O = 2(30L + 6)`.
Per operation, attribute roots have at most 9 entries, total keys at most 14,
values including the root at most 15, depth at most 2 edges, and no arrays.
Literal keys have at most 32 bytes; non-epsilon strings have at most 64 bytes.
There are at most `8L + 2` epsilon copies; let `E = max(original raw config bytes, 1024)`.
Conservative totals are `15O` values, `14O` keys, `448O` key bytes,
`896O + (8L + 2)E` string bytes and `29O` complete key/value visits.
These are payload/event counts, not helper tariffs.

On refusal, `source_descriptor` copies four subtrees before validation. From a
source with `N` values, the new descriptor alone has at most `N + 2` values;
copied text is bounded against the original config plus fixed literals. This
excludes live config/source, serialization and error overlap. The arbitrary
semantic-attributes counterexample above is not an authenticated-route escape.
Future mapping changes require revisiting these bounds. Separately authored
`original_source_tests` regression cases remain **UNRUN**; they do not establish
a universal bound or a passing result.

#### Compact lowering: parameterized collection-visit bounds

This metric counts source-level **reached collection positions** in the named
loops/iterators, including positions rejected by predicates. Capacity reads,
key comparisons, numeric/scalar/index reads and library callbacks are separate;
this is not full work, CPU/runtime/security cost, full-fit proof or funding.
Let independent `P = ops.len()`, `I = imports.nodes.len()` and
`U = imports.public_norms.len()`. On the current authenticated, **untransformed**
genuine-graph route, `P <= n_prefill <= 30L + 6 <= O = 2(30L + 6)` (`L`: layers).
Count descriptors, not successfully returned publications: postappend failure
can leave `P = cursor + 1`. Construction
[explicitly rejects capacity mismatch](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3388-L3391).
Never substitute this `P` bound for imports, sources, nested rows, history,
physical nodes or native capacities.

Leaf composition, conditional on reaching each complete successful helper:

| Leaf | Reached positions and separate queries |
| --- | --- |
| [Native resources](../../crates/pllm-core/src/puma/tensor.rs#L579-L670) | `C_native(n) = 2 + n + n(n+1)/2 <= 18722` when `n <= 192`: two affine-table descriptors, `n` outer steps, triangular predicate positions. No affine-element traversal; table capacities remain independent. |
| [Imports](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3213-L3251) | `I + 4 + U`: nodes, four fixed reserve entries, public norms. Capacity reads and norm metadata queries remain separate. |
| [CheckedRows](../../crates/pllm-compiler/src/replicated_decoder/compact_validity.rs#L311-L330) | `R` runs plus `2R` option positions; `p <= 2R` owned-program resource calls. Repeated Arc identities are not memoized. |
| [Prepared map](../../crates/pllm-compiler/src/replicated_decoder/prepared_tensor_map.rs#L613-L622) | `m = 1 + tail_present <= 2` program visits and `m` resource queries; baseline adds only a fixed scalar size to that template sum. |

For **one** [`CompactGraph::baseline`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3436-L3511),
order is imports first, cached frontier/control bytes and graph capacities next,
then `P` descriptor positions, with these per-descriptor branches:

| Branch | Actual nested traversal/query |
| --- | --- |
| AttentionValues / AttentionScores | Two capacity reads each; no element traversal. |
| AttentionSoftmax | Cached retained bytes only; no row traversal or resource query. |
| AttentionScale | Every chunk: resource query; one capacity read plus layout arithmetic. |
| Rotary | Every native row: resource query; one capacity read. |
| Prepared map | Only `MapStorage::Owned`: `m` program visits/queries. |
| RMSNorm | Owned main program resource query; checked rows independently traversed, regardless of main-program ownership. |
| Argmax | Every program entry inspected, resource query only for owners; one capacity read. |
| TokenLookup | Full program and optional tail both inspected, including nonowners; resource query only for owners. |

Last comes a fixed four-`PublishedOp` allowance. No attribute, frontier, replay
or predecessor scan occurs inside this baseline. With `s_i` scale chunks,
`r_i` rotary rows, `a_i` argmax entries, `t_i` lookup tail presence, and sums over
the corresponding reached owners/checked rows, successful visit count is

```text
B = I + 4 + U + P
    + sum_scale s_i + sum_rotary r_i + sum_argmax a_i
    + sum_lookup (1 + t_i) + sum_owned_maps m_i
    + sum_checked_rows 3R_i + sum_(q in Q) C_native(n_q).
```

`Q` is a **multiset of actual native resource calls**: scale chunks, rotary rows,
owned map programs, owned RMS programs, owned checked programs, owned argmax
programs and owned lookup programs. Thus native part is at most `18722|Q|`
**only if every queried program satisfies `n_q <= 192`**. Sharing gives no credit.
Native callbacks, scalar/index reads and numeric operations remain separate,
not implicitly funded omissions. Capacity reads require independent bounds,
not inference from visit counts. Refusal follows the ordered reached prefix,
including work within the failing helper; no later scans are charged as reached.

Call-site composition below counts baseline invocations, **not** complete
one-contraction cost. Products of call/visit counts here mean addition of actual
repeated visits, not multiplication of independent byte maxima.

| Reached contraction route | Baseline composition and additional visits |
| --- | --- |
| Outer / shared prelude | [`lower_next_inner:4434`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L4434) calls once; [`lower_contraction:2732`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L2704-L2732) is already inside the inner leaf. Never add that shared prelude twice. |
| Unbiased contraction | Whole route: two baselines with no request/reused map; three with no request/fresh map, or request/either map. Extra preparation at [`3733`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3733) or [`compact_request_rescale:17`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering/compact_request_rescale.rs#L17). |
| Bias, if reached | Adds one baseline at [`2591`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L2546-L2609), plus source prefix (including public-filtered positions), map-owner chain and output/chunk visits described below. |
| Public `PerToken` lookup | Whole route has three baselines: outer `4434`, shared `2732`, first lookup quote [`1789`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L1718-L1789). Additional lookup visits below are not a full lookup tariff. |

For bias, actual owner-chain indices strictly decrease: at most `P + 1` inspected
positions, not an extra unbounded predecessor walk. Use `B_out` for output count
(not layer `L`) and `K = ceil(B_out/128)`: `B_out` output-index positions and
`K` chunks; `2K` gather, `K` combine and `K` append are **helper-call counts only**.
Conditional postbias-energy branches repeat source/block queries; do not merge
those queries into the earlier prefix or treat helpers as constant-work proofs.
See [bias continuation](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L2592-L2702).

For public `PerToken` lookup, `T` compiler iterations are **not**
`T * ceil(V/128)` vocabulary tiles. Source-prefix scan repeats `T + 2` times; reached positions total `(T + 2)*s`, where `s` counts every source entry through the matching private source, including public-filtered entries.
full/tail program positions are at most two. Prior searches repeat `2 + a` times
per width (`a = request_present`), with their actual reached descriptor/program
prefixes. Eight old native-carrier intern passes are distinct nested traversals,
not a raw `9T` formula. Current contraction path refuses private first chunk for nonzero admitted token/vocabulary geometry and
public `Tiled`/`Trie` after shape checks; this says nothing about all runtime profiles.
See [lookup continuation](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L1790-L2542).

Selection uses [`Enumerate<slice::Iter>`](../../crates/pllm-compiler/src/replicated_decoder/graph.rs#L93-L99):
specialized `.nth(cursor)` indexes, rather than scanning a cursor-length prefix.
Frontier finish instead traverses actual `op.inputs`. Optional prebound binding
prefix and shape visits are outside baseline. Inner-failure diagnostics may repeat
source prefixes; retirement failure skips diagnostics/commit while the original
uncommitted attempt burns. Collection drop is not priced. See
[outer lifecycle](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L4387-L4897).

**Scoped ROW/COMPLETE/alias collection terms.** Genuine imports have `slot=None`;
construction starts with empty `ops`, and lowering validates input before assigning
`slot=ops.len()` and pushing: every parent slot `i < j`. See
[construction](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3387-L3420)
and [lowering/publication](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L4387-L4897).
Here locally `L` is initial lanes (not model layers), `w > 0` is row width, and
`L*` is maximum **reached** ancestor lanes, at most `MAX_VALUES`. Genuine
CausalMask/CacheSuffix/KvInitialize preserve shape; Feedback preserves lanes and
input shape. Thus ROW's only `alias_range`, if reached, has count `L`, although
`L*` can be larger when Reshape bypasses LastToken.

| Query | Reached collection/validation bounds |
| --- | --- |
| [ROW (`row_norm`)](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3963-L4003) | At most `P + 1` calls including initial; fallback row iterations at most `floor(L*/w)`; validations at most `P + 1 + floor(L*/w)`; alias descriptors at most `P`, comparison entries at most `L`. Direct import makes exactly one COMPLETE attempt then returns `None`, even with multiple rows. |
| [Standalone COMPLETE (`complete_row`)](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L3911-L3961) | At most two validations, `P` alias descriptors and `w` comparison entries. Alias stripping includes CausalMask; COMPLETE does not directly recurse on CausalMask. |
| [Alias `A(v,s,n)`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering.rs#L4007-L4069) | Visited descriptors `a <= P`, initial word-call indicator `u <= 1`, comparison entries `m <= n`; word calls at most `u + 2m`. A failing word call can leave fewer completed comparisons. |

These are actual reached prefixes: failed checks/word calls stop later positions.
Delegated History comparison and Score arithmetic (`nodes_prefix`/`tile_sum`)
remain separate bounded noncollection costs. For
[`ValueLayout::word`](../../crates/pllm-compiler/src/replicated_decoder/compact_lowering/compact_attention_values.rs#L236),
the `sequence + row % sequence` loop occurs only in the initial word call under
that mapping; unsupported inverse mapping then returns `None`. Add actual reached
`z <= 2L* - 1`, not `m*z`. Scalar/numeric/helper/shape-fold costs are not priced.
Finite ancestry proves neither stack safety nor failed-helper request gating.
The [alias-order regression](../../crates/pllm-compiler/src/replicated_decoder/compact_norm_tests.rs#L345)
is **UNRUN**, ordering/certificate retention only, not integration/funding/native proof.

Collisions, norm-block queries, helper bodies, library refusals and cleanup
remain explicit unclosed costs. These bounds adopt no tariff, limit, guard, map,
witness or authority. Tests **UNRUN**; Native **HOLD** unchanged. This is bounded
source-query composition only, not complete contraction or request accounting.

#### Pre-JSON root prefix: source accounting limits

This covers only `source_entry` 828–847: pre-JSON success, not successful `new_inner` return (JSON starts at 848). Let `Q` be the bootstrap bytes quoted by
`shared_allocation::OwnedBudget::resources()`: Arc backing quote
`A`, seven named logical transition slots and `size_of::<OwnedBudget>()`, as calculated by
[`OwnedBudget::resources`](../../crates/pllm-garble/src/shared_allocation.rs#L327-L358).
Let `G = size_of::<OwnedExternalCharge>()`, `P = pending_controller_bytes`,
`S = server_controller_bytes`, and `s` be 0 or 1. On success at line 847,
`usage = peak = Q + P + G + s(S + G) <= MAX_ALLOCATION_BYTES`, with `2 + s`
strong links and one counter heap allocation. The quote at 804–807 instead uses
`Q + S + G`; direct-header admission at 190–208 uses
`Q + P + S + 3 * size_of::<SourceHeaderEntry>() + G`. This named inventory is not
complete call-frame, `Option`, result or refusal-overlap coverage. Named slots
are not proof of simultaneous physical storage; this endpoint equation is not
a whole-branch maximum. Complete sums for root refusal, pending refusal, server
refusal, success with/without server, and unwind retain symbolic source weights for caller,
move, result, closure, scalar and unwind terms. Under the **DESIGN ONLY**
source-occurrence convention (2026-10-31), distinct callee-return, caller-call-result
and move-source IDs each count once; embedded fields and backing each count once.
Expanding an envelope replaces, not adds to, its counted slots: a conservative reached-slot
union, not a physical-stack claim or adopted runtime/funding contract. Injective funding remains open.
Under the non-adopted retain-all-reached **DESIGN** convention, completed-root envelope `D_root` strictly exceeds `Q` (comparison below).

- Canonical ID = function + source expression/parameter/local/return endpoint + role.
  An argument/receiver expression counts once, without extra staging; callee parameters,
  callee return endpoints, caller call-results, extractions and named locals remain distinct.
- Nested expressions, value-context reads, operator operands and results count separately;
  place-only paths are not values. Embedded fields/captures count once, not again as slots.
- A `?` operand aliases its existing result; count one reached success/error extraction
  and the reached function-return boundary. Compiler residuals, identity conversions and
  private `Err` construction remain unknown `U`, not invented source IDs.
- Implicit capture construction remains `U`; an explicit capture-initializer expression
  has its own source ID. The closure value includes captures once; embedded storage
  does not add separate field/capture slots.
- Direct Rust function-item types are ZSTs: optional zero-weight IDs, no owner header.
  Function pointers are not function items; actual first-class parameters retain their
  actual types. This does not erase dynamic callbacks or method receivers.
- Count return endpoints only when reached: no whole `new_inner` `Result<Self, Error>`
  reserved at entry, and no success return reached at 847. Loop/CAS callbacks reuse
  occurrence IDs for storage, without bounded-work inference. Smallest current design
  removes extra argument staging; any retained logical padding needs distinct padding IDs,
  never source-occurrence IDs. These rules remain **DESIGN ONLY**, not runtime/funding adoption.

At `shared_allocation` 372–375, reservation is not allocation; `allocation()` at 373 is a test-failure hook (production `Ok(())`), not allocator-resource proof.
Actual root backing allocation occurs inside `Arc::new` at 375; hook success proves no allocator resources.
**Reference-version qualification (2026-10-31):** the closed census, source IDs and
citations below describe the **pre-normalization reference**, not current bodies.
Behavior-preserving private-helper/function-item, server-match and subtotal changes
leave old `Q` and legacy resource fields unchanged; normalization is source-reviewed,
not compiled, with tests **UNRUN**. The full modified-prefix census and new logical
quote remain unclosed; concrete `E`/API is not yet authored, and source accounting
is not adopted. Native **HOLD** remains unchanged.

Final bounded reconciliation closes only the reference **non-test SOURCE enumeration** under
the **DESIGN ONLY** convention, not complete-resource accounting or funding.
Scope: `shared_allocation` (SA) 35–51/101–115/121–132/327–377/398–407 and
`source_entry` (SE) 549–583/828–847. Remove CHARGE IDs `O402.self_read`,
`O404.self_read`, `R102.self_read`, `R105.self_read`, `R110.self_read`:
these are place bases beneath projections/borrows, not value copies. Implicit
autoderef stays `U`; retain actual receiver/reference IDs, `R105.limit_read`
and explicit `*next` operand. No fictitious ROOT reads or padding are needed.
Namespace each callee ID by callsite: root reserve, pending charge, optional server
charge. Replace the root-reserve opaque envelope with its `Budget::reserve_bytes`
instance, never both. Caller argument/callee parameter and callee return/caller
call-result remain distinct; duplicate alias roles merge. Count helper returns
only when reached, not a return endpoint for every successful `?`;
`new_inner` success return is not reached at 847.
`C_src = max_{b in Bprod} sum_{i in ReachedCanonicalIDs(b)} sizeof(type(i))`.
`Bprod` includes raw-empty, raw-oversize, root/pending/server failure and prefix
success without/with server, using actual short-circuit/failure prefixes. Fresh
used-check refusal is unreachable under intact construction. Weights stay symbolic,
not guessed layouts. Non-test accounting excludes SA113 test accounting;
no test-configuration census is closed. Callback ID reuse bounds source storage
only, not CAS retries or work. Count backing once when reached inside `Arc::new`,
not at reservation or the hook.
Full obligation remains `max_branch(source sum + once-counted backing + U)`, with
transitive/library/compiler/ABI/atomic/allocator/unwind/cleanup terms explicit in
`U`, never set to zero. Bootstrap funding remains **PARTIAL**: no source-only sum
proves a finite total, reproducible numeric target or actual paid credits. Cleanup,
early parameter/raw-validation coverage, injective credits, full-resource reserve adequacy,
runtime adoption and all funding remain open; tests **UNRUN**, Native **HOLD** unchanged.

Under the non-adopted retain-all-reached **DESIGN** convention, define
`D_root = A + sum(reached ROOT SOURCE weights)`; backing `A` counts once, while `C_src` excludes backing.
`Q = A + sum(Z1..Z7) + H`: SA337–343 quote the seven types below; SA348 quotes `H`.

| Quote type | Distinct reached ROOT ID (function, SA line, role) |
| --- | --- |
| Z1: `Budget` | `from_fresh_budget:365:budget` parameter |
| Z2: `Result<Budget, Error>` | `new:362:Budget::new` call-result |
| Z3: `OwnedBudgetResources` | `from_fresh_budget:366:quote` local |
| Z4: `Result<OwnedBudgetResources, Error>` | `from_fresh_budget:366:resources` call-result |
| Z5: `Result<OwnedBudget, Error>` | `new:361:return` endpoint |
| Z6: `Arc<Budget>` | `from_fresh_budget:375:Arc::new` call-result |
| Z7: `Result<(), Error>` | `from_fresh_budget:372:reserve_bytes` call-result |
| H: `OwnedBudget` | `from_fresh_budget:374:Self` origin |

All eight IDs are reached at completed `OwnedBudget::new` return, not `new_inner` success at 847.
Extra disjoint `new:361:limit` parameter has positive `sizeof(usize)`; all remaining weights are nonnegative.
Therefore `D_root >= Q + sizeof(usize) > Q`, without native queries, guessed layouts, `U` or caller receipts.
This logical-envelope comparison is not a runtime admission bug, physical undersizing, replacement reserve cost,
injective funding or current-policy failure. The unpaid bootstrap DESIGN exception is independent; no `Q` or cap changes.

Caller parameters and raw validation at 830–831 precede root construction; later successful server reservation cannot retroactively fund them or receiver copies.
The approved root-establishment exception alone does not settle these obligations.

Server-guard refusal drops controller then root. Subsequent unwinding drops later
locals first, then unmoved server guard/controller/root. Root drop subtracts
`size_of::<OwnedBudget>()`; the last Arc destroys the counter with the remaining
`Q - size_of::<OwnedBudget>()` numeric baseline, not a surviving reservation leak.
Source API event counts are finite, but `AtomicUsize::fetch_update` weak-CAS retries
have no finite bound even for a fresh root. Allocator/abort/OOM behavior supplies
no resource, time or unwind proof; repeated calls repeat setup/refusals without
a global attempt cap. The established-root metadata/setup exception in the
[prepayment contract](../plans/raw-check-prepayment-contract.md#L1045-L1058)
excludes root construction and cleanup. No actual work owner exists before
`source_entry:833`. An explicit unpaid root-bootstrap exception **once per fresh
ingress invocation, ending before JSON**, is approved for **DESIGN ONLY**
(historical approval, 2026-10-31), not implementation or execution. All cleanup, including refusal/unwind, stays open; aggregate calls are
unbounded. No new authority, ledger, ticket, map, guard or caps are introduced; the open funding,
publication/cleanup obligations and Native **HOLD** above remain unchanged.

**Scoped authoring permission (2026-10-31):** source/test authoring for the logical
pre-JSON bootstrap is authorized after a concrete Astra plan and independent review,
not execution, runtime activation or adoption. Public `OwnedBudget::reserve_bootstrap`
and five `bootstrap_extension*` regression tests are authored and independently
source-reviewed only (Astra46); tests are **UNRUN**, with no compilation or rustfmt.
The numeric-baseline API delegates to existing `Budget::reserve_bytes`; successful
calls accumulate, with no API one-shot guarantee, new guard, ledger or refund ticket,
and no `Q`, cap, layout or atomic changes. Ordinary owned/native guard drops and
cloned root owners cannot refund `E`; final strong-owner retirement destroys the
counter, not a reservation surviving externally. Tests cover atomic refusal,
overflow/exact fit, repeated increments, clone/split/native/guard lifetimes, failed
builders and synchronous scope drops, not actual async cancellation or native execution.

The corrected zero-padding quote is authored as typed checked constants and
`source_union_quote`, with `NC = 16`, `NR = 2` and the 61-occurrence resources group.
The resources group preserves `Layout::extend`'s `(Layout, usize)` return order.
Independent source review caught two omitted operand reads:
the `closure_bytes` operand in `FIXED.checked_add(closure_bytes)` and the
`reference_bytes` operand in `subtotal.checked_add(reference_bytes)` are distinct
canonical value-context reads, not their bindings. The earlier 29 claim is
superseded by HFIXED's 31 `usize` SOURCE occurrences, not padding; its other groups
remain 6 `(usize, usize)`, 8 `Option<usize>`, 10 `Result<usize, NativeError>` and
4 `NativeError`; independent recount found 6 measurement and 25 compiler occurrences.
Independent source review cleared bounded authoring of the corrected quote and
two tests, not full proof, funding or runtime activation. The quote uses actual
reservation-update closure/reference widths from the public
`OwnedBudgetResources::reserve_update_layout`, without invoking the callback or
querying resources again. Four checked runtime operations combine the widths with
the const-evaluated fixed groups; overflow of constant arithmetic fails const evaluation.
One unconditional private fresh-ingress call reserves additional `E` on the original root
after construction, before pending/server charges and JSON. This is source authoring
only, not runtime activation or executed proof. The conservative self-inclusive
non-test typed SOURCE union adds no padding, subtracts no `Q` and repeats no Arc
backing; it is neither a physical peak, exact minimum nor full-funding proof.
The existing `source_union_quote_matches_typed_manifest_and_adds_to_bootstrap` test
checks the literal typed manifest and arithmetic/API composition, not census
exhaustiveness. The existing `source_prefix_length_refusals` test covers guards
at source level, without JSON-order instrumentation. Existing custody assertions
now require live usage `> Q + E` and retired usage `== Q + E`.
Scope remains the unpaid pre-JSON logical reservation only; all builds/tests remain
**UNRUN**, with no compilation, MSRV check, rustfmt or native-execution permission.
Compiler/ABI/allocator and opaque `U`,
work/CAS retries and cleanup are not declared paid or bounded by this logical metric.
Post-JSON controller custody, cleanup and full-resource funding remain **BLOCKING**;
Native **HOLD** is unchanged, runtime activation is unauthorized, and no cap increase
or full-funding claim is authorized.

User-approved bounded SOURCE authoring adds private `cfg(test)` retirement
observers, paired adapters and one shared `pending_transition` body; two independent
SOURCE reviews cleared this implementation, not runtime activation or executed proof.
Ordinary production identity adapters contain no trace storage; all observers are
`cfg(test)`. Ingress remains intact through fallible preparation, committing afterward.
One genuine same-counter changed-workload `check_source` Binding refusal regression
asserts five events after ordinary nonpanicking actual-slot retirement, with
manifest/directory/original/raw before controller. Restoring early destructuring
would fail by SOURCE reasoning only; mutation remains **UNRUN**.
Expanded `None` directory observes its Option slot only, not a pointee or compact proof.
Existing 31-scalar quote, `E`, pre-JSON reservation, root/ledger/API/caps/charges
remain unchanged. New post-JSON helper/adapter boundary coverage is **UNPROVED**.
Test wrappers are outside logical/physical qualification; no full test-build
resource-funding claim follows. Three inline envelopes remain logical assignments
for ingress staging, pending assembly/transfer and destination Pending, not complete
SOURCE, storage, capacity, transitive payload or physical stack-lifetime proof.
Tests/builds/MSRV/rustfmt/mutation remain **UNRUN**; Native **HOLD** is unchanged.
Approval covers SOURCE authoring only, not execution or runtime activation.
Broader compact, pre-scope, escaped-Arc, unwind, cleanup and full funding remain
**BLOCKING**; unproved storage and opaque `U` remain open.

User76201 explicitly approved two additional bounded local SOURCE/test repairs:
`consume_compact` retains the remaining Pending through fallible pre-emission
preparation after moving importer/import charge into ImportStage; EmissionScope
first stages actual created inputs before its actual controller in PreEmissionScope.
Both production and private observed entrances share their respective transition
bodies. Genuine importer(4) prebox Bounds and deliberately contrived undersized
incoming-controller constructor regressions are authored **UNRUN**. Prebox trace
marks exactly four payload slots: directory Option(None), original, manifest and
raw, plus original controller; not five payloads or exact pointee destruction. Importer is
NOT in this combined trace: ImportStage retains its existing separate drop
instrumentation. Constructor trace marks the actual CreatedInputs aggregate and
the same incoming guard, deliberately shrunk to zero, not a natural production
constructor failure. Fixed per-instance atomic markers observe nonpanicking normal drop
completion only, not unwind completion; budget counters establish pressure/identity,
not order. Under existing User41221/76201 SOURCE/test-authoring authority, exact-count
strengthening is authored **UNRUN**: initial trace counter 1; prebox ordinals are
a permutation of 1..=5, controller exactly 5, next counter 6; constructor created
exactly 1, controller exactly 2, next counter 3. Existing explicit order assertions
remain. Old-order sensitivity is SOURCE reasoning only, not executed mutation.
Astra21 SOURCE-cleared ownership; Astra30 re-review closed the exact-once finding:
exact-count strengthening has final bounded **SOURCE-CLEAR** status, **UNRUN**.
Prior `into_pending` clearance above is unchanged.
The consume boundary ends before
`emit`: on emit error, outer owners can still outlive the moved controller.
Post-handoff, escaped-Arc, source/helper/wrapper STORAGE coverage, work, cleanup,
opaque U and full funding remain **OPEN/BLOCKING**. No quote/cap/allowance widening,
bootstrap-31/Q+E, root/controller/ledger/charge/resize/check ordering, error ordering
or attempt semantics changes are authorized. No execution/build/test/rustfmt,
syntax validation, native activation or HOLD release is authorized; Native **HOLD**
is unchanged. Tests/builds/parser/rustfmt/mutation remain **UNRUN**. These bounded
source edits are frozen; SOURCE clearance grants no execution or broader closure.

### Explicit checked-range and compact-execution policies

The public-weight component additionally accepts independent `range_policy`,
`lookup`, `selection` and `peer_transport` choices. Their defaults preserve the
existing component, source and execution identities. Nondefault choices bind the
complete execution policy at native source admission and every worker. Private
weights reject these public-only choices.

`range_policy="checked"` freezes `max_abs_activation` in public integer units
(default 256, bounded to 1–1,024). Where public certificates do not already prove
the complete RMSNorm input interval, full-word protected comparisons test the
inclusive Q18 endpoints. A Boolean mux selects either the unchanged input or zero
**before** arithmetic that depends on this interval. These guards do not assume
the bound that they are checking. Tile validity reduces through fresh exact
products into one sticky shared request bit. At the end of all fixed-count
selections, every token is multiplied by this bit before final shares leave the
workers. The client accepts only an aggregate bit of exactly one; a later feedback
failure therefore suppresses earlier tokens as well. No failing value or location
is opened. The client does learn aggregate acceptance, under the separate D12
contract and checked-range numeric identity. Accepted arithmetic retains the
approved equations; additional randomness changes the execution identity and
can change a particular stochastic realization.

`lookup="tiled"` traverses the full public vocabulary with at most eight private
token rows and 128 vocabulary entries per tile. It reuses each streamed public
weight range across those rows and both replicated components. It retains fresh
predicates for every token/entry tile, exact ring64 accumulation, source commitments
and the original complete table result. A public interval proof excludes nonzero
multiples of the selected power of two before choosing a narrower equality
domain. This equality is not a range check. Distinct compact tensor/Boolean frame
namespaces validate the expected width and canonical padding before allocation;
arithmetic OT words remain full-width. Immutable full/tail predicate templates
are shared and counted once, while active executors and accumulators remain
separate owners.

`selection="tiled"` replaces the expanded tournament metadata with one bounded
owner. It compares every required pair and preserves the lowest original index
on ties. A dedicated last-use gather supplies the complete logits; the original
logit owner and this copy are separately charged. Fresh protected comparisons
and untruncated score/index products execute in public 64-comparison tiles.
Metadata reduction does not imply fewer cryptographic operations.

`peer_transport="streams"` uses bounded persistent authenticated peer WebSockets.
An exact request/source/configuration/worker-epoch handshake precedes setup;
ordered sequences, one-use connection admission and completion sealing reject
substitution and reconnect replay. Public quiescence barriers and freeze-before-send
waves remain. Queue/backpressure bounds, cancellation drains and directed control
and data counters apply to each link. Admission adds 8 MiB per worker once, beyond
the original request owners. Neither this transport nor the new native owners
increase the existing session, work, node or payload ceilings.

The generated native integer oracle captures complete semantic tensors, full
prefix K/V, vocabulary logits, feedback and final acceptance under test-only
hooks. Independent AES reconstruction derives the actual truncation tapes; every
draw must be consumed exactly once. No trace or private-state export is installed
in the SDK. These bounded controls and shape diagnostics do not qualify a full
checkpoint, the shared 64+8 workload, source-model quality or response costs.
The [completion plan](../../paper/methods/puma-completion-plan.md) and
[feasibility account](../../paper/methods/puma-feasibility.md) retain those gates.
The checked-execution slice passed 462 Python checks and 218 native checks;
64 generated oracle cases rejected 824 negative controls. Its separate pinned
4+2 source publication rejected layer-eight attention before worker launch.

### Trie lookup and complete attention-vector certificates

The public-only `lookup="trie"` choice converts each private token to Boolean
shares once, then builds an MSB-first prefix trie over the complete public
vocabulary. Bounded bit projection and fresh protected products retain distinct
selectors for each token. A branch is omitted only when the public vocabulary
contains no such prefix. The compiler authenticates the original admitted token
IDs or the complete stable-selection output as provenance for `[0,V−1]`; an
interval annotation alone cannot authorize the shorter conversion. The immutable
source, vocabulary, logical row order, dedicated last-use input gather, primitive
profile and complete work budget bind the owner. Public table scans still cover
all rows and both replicated components.

Eight-row lookup ownership is bounded at 29,773,248 bytes in the native fixture,
including selector overlap and primitive scratch. Schedule-derived lookup work
is 10,707 ordinals for 4+2 and 641,966 for 64+8 at the pinned vocabulary. These are
lookup-only contracts and counts; complete-decoder work, admission and response
costs require separate evidence. Existing complete-program ceilings still apply.

Checked public source publication additionally streams Q/K projection and bias
statistics per semantic head block: maximum absolute row/column sums and squared
Frobenius norm. Each worker recomputes every statistic from its immutable Q18
source before execution. Shape, head width, source identity and the complete
record set bind publication and native planning. Ordered scans and bounded
metadata include partial import-frame tails and retire on malformed input.
The existing 1 MiB source-norm allowance now preflights retained metadata copies,
request-map capacity and overlapping scan/output scratch before cloning or reads.

An attention-only compiler ledger propagates complete normalized-row energy
through the original projection, bias and half-split RoPE. Each RoPE output
coordinate retains both separately truncated coefficient products and both
rounding errors. Exact word-reference provenance binds complete query rows and
the physical right-matrix **columns** representing each causal key. Native
integer Cauchy–Schwarz admission then bounds the full contraction before its
single Q18 truncation, intersecting any already admitted scalar-product hull.
A loose energy certificate cannot widen that scalar bound. The general RMSNorm
energy ledger remains separate;
these sidecar proofs do not select a different normalization recipe. Scalar
operator domains, ring64 quarter-gap admission and fresh-material budgets still
apply.

Three compiler/source controls and six generated native/Python oracle cases
cover this integration. The latter compare 1,150 complete tensor checkpoints
across two-layer Qwen2/Qwen3/Llama GQA, including later-feedback rejection,
actual truncation tapes and wrong-KV-routing controls. Worker source imports
recompute the tested statistics. The complete compiler run covers 70 generated
oracle cases, 9,816 tensor checkpoints and 962 negative controls. Installed-wheel
validation passes 656 Python tests with eight skips; the subsequent two-file
bound-intersection change passes 59 targeted installed SDK checks. All 369
installed package files match each tested wheel before and after its suite.

Two pinned Qwen 4+2 source-publication attempts select trie lookup, checked ranges,
tiled selection and streams. Both reject at layer-six RMSNorm before worker
launch, with zero observed swap growth. They retain separate identities from the
earlier tiled-lookup layer-eight attention rejection. No whole-checkpoint response
or quality result follows. Historical source versions remain in Git.

Checked RMSNorm lowering now preserves a complete ordered row's existing energy
certificate across the unchanged private zero-fallback guard. Its bound is
`min(B, width * cap²)`; only strict improvements over the scalar bound retain
additional, ordinarily charged metadata. Missing, reordered, duplicated or
mixed-owner rows cannot inherit another row's proof. Static alias paths and
their frozen execution identities remain unchanged. Native controls cover the
129-lane boundary/tail, stochastic integer parity, all 40 ordered truncation
events and final sticky rejection. This does not prove a Qwen numeric defect or
clear its complete-program limits.

Admission failures now include a public construction stage and resource ledger
alongside the existing operator context. These diagnostics do not enter the
execution digest or expose private values. The opt-in ordinary test
`test_pinned_public_qwen_checked_trie_source_publication` in
`tests/test_replicated_public_weights.py` retries the pinned 4+2 publisher through
the installed SDK, retaining whole-topology memory admission and pressure/swap
monitoring. It does not launch workers or establish accepted decoder execution;
the historical attempts above remain separate from any rebuilt retry.

The separate native compact-sign constructors cover guarded SiLU/GELU,
reciprocal/rsqrt, complete-row normalization and softmax. Signed interval proofs
select public 8/16/32/64-bit conversion buckets, with checked subtraction and
full-word membership guards where required. Distinct compact frame identities
retain full-width arithmetic OT words and rounding equations. These constructors
are not yet selected by a decoder execution policy.

The bounded binary-body adapter consumes an already-admitted original worker,
executes only an exact gather/gather/add-or-Q18-product interval, then returns the
same runtime and PRSS owner. D23 adds explicit `Worker::advance_before` and
`Compiled::advance_to_start` boundaries: every preceding original node, including
all tensor-loop tiles, finishes before the target can execute. Local readiness
does not establish global quiescence; the caller must drain all three roles and
their router before moving ownership. Invalid targets, foreign programs, replay
and cancellation retain consuming retirement. Original program ownership plus
bridge/lowering overlap remains inside the unchanged 64 MiB cap.

The nondefault `insecure-test-support` core feature creates fresh deterministic
test graphs with publicly known keys. Production setup keeps OS randomness; no
live runtime can be cloned or reset. The feature belongs only to isolated test
dependency graphs, not production secrets. Exact parity controls and the bounded
handoff contract live in
`crates/pllm-compiler/src/replicated_decoder/body_integration_handoff.md`.
Full semantic-body dispatch, compact-oracle boundary mapping, trained-checkpoint
admission and SDK promotion remain separate gates. No automatic policy, numeric,
security, quality or performance qualification follows from this native adapter.

### D26 compact-execution foundations

Expanded compilation uses the extracted ordered graph traversal. Bounded
registry/expanded controls compare ordered frames, both replicated components
and PRSS/AES cursors. The descriptor view borrows validated source/norm maps;
this is not zero-cost request ownership. The committed
[`SourceRequestImport`](../../crates/pllm-compiler/src/replicated_decoder/source_entry.rs)
retains charged manifest/raw owners and registry/import/controller guards,
alongside the original source plan and importer. Construction also allocates
validation temporaries; no whole-request fit follows from borrowing descriptors.

Semantic-body construction admits container/proof growth and overlap before
allocation, and burns a partially failed builder. Native tensor factories use
shared preconstruction quotes, with caller-owned buffers charged separately.
RoPE fills exact caller-owned coefficient buffers without allocation; the legacy
generated-Vec quote remains unsupported. Shared tensor execution borrows the
original immutable Arc template without deep-copying public coefficient tables;
standalone admission still charges full backing for each invocation.

The closed [prepared-map capability](../../crates/pllm-compiler/src/replicated_decoder/prepared_tensor_map.rs)
covers only rescale and guarded
SiLU, retaining at most the original full and tail templates. Its local consuming
tests cover full/tail parity, late failure, cancellation, retained frames and
returned-output ownership. No arbitrary Program adoption, external numeric-proof
authority or reconstructed template substitution is provided.

A separate, still-gated one-loop rescale18 adapter has historical compiler-owned
[fixture controls](../../crates/pllm-compiler/src/replicated_decoder/prepared_tensor_map_tests.rs)
at 1, 128, 129 and 257 lanes. These bounded controls compare complete ordered
frames, both stored share components, PRSS/AES cursors and work against the original
expanded Worker, including a subsequent protected product, retained/last-use
inputs, accepted/rejected sticky validity and final masking. Active-tail malformed
receives and cancellation consume original owners while retained peer frames stay
valid. The candidate entry is private and test-only; the committed
[production parity gate](../../crates/pllm-compiler/src/replicated_decoder/body_tensor_integration.rs)
still rejects explicitly. Historical fixture results do not validate current
uncommitted prototypes, trained layer-six execution or a complete compact decoder.

Staged finalization quotes the actual populated Builder and exact outputs before
construction, adding the retained original owner baseline. Named adapter control
and result slots use explicit already-priced destinations; native Plan allowance
is not borrowed. Independent owner-capacity equality and exact/one-short gates
cover the baseline and finalization boundary. This is a scoped logical-owner
census, not recursive helper, ABI-stack, allocator or RSS certification.

These are bounded logical owner/buffer envelopes, not measured RSS or a complete
request-wide peak proof. Later landed source-owned checked-public SDK setup and
creation-bound RoPE capture do not open production BODY dispatch or establish
globally drained role handoff, complete-owner admission or trained-checkpoint
execution. The [D26 phased-admission clarification](../../paper/methods/DECISIONS.md#d26-opt-in-compact-puma-semantic-outer-representation)
permits same-counter phased construction; it does not say that construction is
implemented or a whole compact request fits 64 MiB. The neutral admitted-peak
counter alone is not compiler-instrumented phased-constructor evidence. Native
Q18 probabilistic truncation, security/range/source-proof obligations, numeric
identities, workload and protocol limits remain unchanged; this note adopts no
new contract.

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
Generated quality and wide-contraction controls retain the common-Qwen admission
rejection as a distinct outcome.

The explicit quality `outcomes="record"` mode uses a separate version-two report.
A typed native numeric rejection retains its original logical cohort position;
transport/admission failures remain distinct. Accepted-only agreement reports its
own sample count, and incomplete candidates cannot enter ordinary rankings. A
caller-owned private cohort key binds comparison identity. Decode diagnostics
require caller-supplied frozen continuations rather than deriving a teacher from
a possibly rejected first candidate. Bounded serialization, atomic publication
and cancellation cleanup preserve sanitized partial reports. Default version-one
execution continues to abort on rejection. Generated installed-wheel gates cover
typed numeric rejection, accepted/rejected sample accounting, stopped candidates,
partial-cohort ranking rejection and cancellation. UTF-8 report byte limits and
private traceback-reference cleanup are checked; cleanup is not a secure-erasure
claim. The CLI exposes the same explicit record mode and frozen cohort inputs.

Generated Qwen2/Qwen3/Llama controls include nonzero attention/MLP paths whose
independent sign oracle distinguishes private prefixes, private feedback, protocol
failure retirement, ordinary SDK/gateway execution and benchmark accounting.
Full-checkpoint hidden-tensor conditional-on-mask equality, full-checkpoint ranges,
source-model quality and matched publication measurements remain separate gates.
The initial SDK slice passed 692 broad installed-wheel checks (one optional
Metal skip), 138 final-build
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
