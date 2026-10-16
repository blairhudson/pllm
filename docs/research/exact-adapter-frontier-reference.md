# Lossless source-bound adapters: three construction gates

**Decision: no whole-model exchange-removing adapter admitted.** Three bounded
constructions preserve their declared originals exactly, but their useful source
conditions or complete protected costs do not survive promotion. This is not a
universal impossibility result. There is no prompt compression, distillation,
learned replacement, plaintext activation disclosure, public mask basis, TEE, or
free nonlinear oracle in an admitted construction.

Implementation: `scripts/probe_exact_adapter_frontier.py` (standard library only).
Tests: `tests/test_exact_adapter_frontier.py`.
Frozen report: `docs/evidence/exact-adapter-frontier-2026-10-16.json`.

## Original contract and evidence boundary

An adapter must reproduce the **same original inputs, every logit, active KV,
state, selections and sampler/RNG evolution**, or provide a precise invertible
state map. The original's quantization, rounding nodes, reduction ordering,
backend math implementation, signed zeros and exceptional-value policy belong
to its identity. A new integer profile is not the upstream float original.
The source-bound W8A8 analysis below concerns that existing W8A8 original only.
Nothing here establishes exactness against unquantized upstream Qwen.

Earlier projected polynomial arithmetic omitted intermediate rounding;
joint SiLU/up changed it explicitly. Rank/affine extraction changed full outputs
even when token choices agreed; selecting all channels still had a reported
0.000142 maximum discrepancy. See the hashed source snapshots in the report:
`ARCHITECTURE.md` (projected polynomial, joint gated, extraction and share-linear
sections), `network-first-principles.md`, `network-algebra.md`,
`joint-gated-reference.md`, `prepared-region-gate.md`,
`fixed-weight-correlation-gate.md`, `sota-breakthrough-reference.md`,
`residual-delegation-reference.md`, and the rank/affine extraction implementations
and frozen evidence. These references motivate gates; they do not supply a new
private backend or transfer approximate quality evidence into exactness.

Model facts come from frozen `network-algebra-qwen25-2026-10-06.json` and
`research-prepared-region-gate.json`, whose body fingerprints must agree.
The report retains their separate source-lock identities rather than treating
different experiments as one cohort. Metadata gives 96 body stages, 24 layers,
hidden width 896 and MLP width 4,864 for pinned Qwen2.5-0.5B revision
`7ae557604adf67be50417f59c2c2f167def9a775`. No new tensors, activations, ranks,
entropy estimates or checkpoint duplicate measurements are loaded or inferred.
All executable numeric fixtures are explicitly synthetic.

## 1. Ordered nonlinear DAG quotient, not low-rank fusion

**Map.** A compiler identifies channels with exactly the same ordered gate/up
subgraphs: row bytes, scale/bias bits, input identity, quantizer, rounding nodes,
backend, and position/state dependencies. For a certified equivalence class
representative map `r` and channel map `c`, compute only `z_a=h_{r(a)}`.
The inverse on the promised image is `h_j=z_{c(j)}`. The compact encoding is
injective on vectors satisfying the certified equalities, not on arbitrary
vectors. No observed-activation equality may establish a permanent class.

**Equivalence proof.** Identical deterministic ordered subgraphs have identical
output bits. Replace their uses by exact bit copies. Retain every original
down-projection multiply and add in its original order; do **not** replace
`sum_j RN(w_j*h_j)` by `sum_a RN((sum_class w_j)*z_a)`. Induction over the DAG
then preserves all outputs and state. A backend GEMM's shape-dependent reduction
algorithm must not change; its reduction tree/kernel is part of the certificate.
Register reuse is not algebraic reassociation or permission to fuse an FMA.

**Counterexamples.** Even rank-one matrices `W1=[2^24,1,-2^24]^T`, `W2=[1,1,1]`
at `x=1` produce 0 with the original ordered float32 sum, versus 1 after exact
real-algebra fusion. Equal numerical zeros do not establish equal bit patterns.
Position-dependent inputs and unequal scale bits defeat row-only signatures.
The positive toy uses declared `RN(RN(g*g)*u)`, not an approximation to SiLU;
it compares every result to a separate rational nearest-even oracle. A bounded
source compiler groups identical gate/up coefficient rows **before seeing any
inputs**, binds the complete original program with SHA-256, and checks twelve
input cases with a 4-to-2 channel quotient. Source changes and forged maps reject.
Separate literal-value tests exercise the inverse only; runtime coincidences do
not authorize compiler grouping. Real scale/bias/backend certificates remain
unimplemented; the toy source has only ordered linear nodes and the declared gate.

**Roles and traffic.** Both workers keep the body. A protected backend must
produce one shared wire per certified representative **before** material issuance,
then fan out the shared wire inside each worker. Independently masked duplicate
values cannot simply be discarded. Workers never receive an activation or the
other worker's share. Client sees only the cut values it already owns in the
baseline trust contract. The reference executes clear algebra only.

For a conditional two-worker ring-word cut, each lane sends two gate/up shares
from each worker to client (four words) and one output share back to each worker
(two words). With width `M`, compact width `K`, executed rows `R`, word bytes `w`,
and total one-use issuance `mu` bytes per lane across both workers, component
cost over `T` requests is

```text
baseline cut/material = T R M (6w + mu)
adapted cut/material  = (8M + 4) + T R K (6w + mu)
net component saving  = T R (M-K)(6w+mu) - (8M+4).
```

The setup charges a uint32 map to **each** worker and a width to client. Original
body delivery to both workers, certificates, source creation, other stages,
framing, protected float conversions, failures and complete client RSS remain
additional, unknown costs. `mu=0` is only a specified local-cut layout, never a
claim that a remote nonlinear backend has no material. Fresh client masks consume
`T R K` words, with traffic already included in the output shares. Changes to
linear input/output shapes or their corrections must be priced separately.

**Source veto.** Historical algebra evidence reports zero omitted exact signed
rows across all 96 W8A8 matrices, including combined gate/up matrices. Therefore
this particular syntactic duplicate-row construction has no supported Qwen
channel saving; it adds maps if installed. This does not exclude reachable-domain
functional equality between different rows, or a different checkpoint.

## 2. Full-state one-hot lift: exact and private, but potentially much wider

**Map and inverse.** Enumerate a finite complete state set `S`, including KV,
position, original logits' dependencies and sampler/RNG state. Encode `E(s)=e_s`.
For deterministic complete transition `f`, define `A_f e_s=e_f(s)`. Thus
`A_f E(s)=E(f(s))` exactly, even when `f` is not injective. A public observable
matrix returns the original state fields, logits and selection. State decoding
is injective on one-hot codewords; arbitrary worker shares need not be one-hot.
Rows with colliding transition targets require **addition**, not overwriting.

The synthetic original has 128 complete `(hidden, KV, position, RNG)` states,
two complete integer logits and a declared deterministic finite sampler. Tests
check all 128 initial states over six steps, including every state component,
both logits and selections. The independently written direct recurrence is the
oracle. Dropping position or RNG is rejected. This is not a Qwen state quotient.

**Roles.** Client creates fresh independent uniform additive shares of `e_s` in
GF(257). Each worker holds one share and the public full transition/observable
tables, then updates locally with no inter-worker activation exchange. Its entire
view is independent of initial state by uniform sharing and deterministic
postprocessing; a small exhaustive test checks the *full vector* distribution.
The deterministic test RNG is not production randomness. No cryptographic
security, independent operators, malicious correctness or transport is claimed.

**Cost.** With `n=|S|`, field-word width `w`, `d` output fields, `b` bytes per
transition index and `H` returned steps, this specified layout costs

```text
client -> workers initial shares: 2 n w
source -> both workers tables:   2 n b + 2 n d w
workers -> client observables:   2 H d w
fresh client mask material:      n field words (local creation)
```

The toy has `n=128,w=b=2,d=7`: 512 initial-share bytes, 512 transition-table
bytes, 3,584 output-table bytes, and 28 return bytes per requested step. Worker
updates scan all `n` entries with public access independent of state. Setup can
amortize over requests; fresh initial shares and returned fields cannot. Private
input ingestion/state indexing, all source artifacts, authentication, failure
retirement and complete client memory are extra. A streaming encoder might avoid
resident `n`-word buffers; it cannot remove their bytes or generation work.

A raw unary float32-state lift already sends `2*4*2^32 = 34.36 GB` of initial
shares. A gate/up raw-bit-pair lift sends `2*4*2^64 = 147.57 EB`. Invalid patterns
may map to a declared rejecting state; these are costs of this exhaustive raw-bit
encoding, not a claim that every pattern is reachable. They exceed 256 MiB even
before body/state dimensions. A smaller reachable exact automaton needs a
**proof**, not empirical low entropy. Finite field words do not automatically
represent arbitrary float output bits; output bits/limbs must be added explicitly.

For private varying controls `u`, `sum_u u_u A_u E(s)` is bilinear in secret
control and secret state. Granting public control matrices would disclose private
tokens. The autonomous toy avoids this by including sampling and all controls
in its complete transition; for a real decoder this expands state further.
New private inputs require protected injection or a new entry encoding. One-hot
existence does not give a cheap controlled nonlinear oracle.

## 3. Exact rounding-cell prediction with exact fallback

**Map and proof.** At an original boundary computing `RN32(exact_integer_dot)`,
evaluate prefix `P` and bound the omitted integer tail by public `[-B,B]`.
If `RN32(P-B)` and `RN32(P+B)` have identical bits, monotonic nearest-even
rounding proves the original result equals that word for every admissible tail.
Otherwise calculate the entire original integer dot. Original scale, bias,
activation and all subsequent rounding nodes remain unchanged. The state map is
identity. No approximation, token margin or sampled error bound is accepted.
Signed zeros and non-finite policies need explicit treatment; the bounded integer
fixture has positive exact zero and excludes overflow. An arbitrary ordered
float dot is **not** this source boundary: integer reassociation can change it.

The independent rational oracle exhausts signed tails around ties and binade
boundaries. A synthetic prefix `2^25+4` with nonzero tail bounded by one certifies
success; a `2^24` boundary forces full fallback. The positive example's giant
coefficient is explicitly not a signed-i8 checkpoint claim.

**Sharp source veto.** For symmetric full-domain W8A8 inputs `|x_i|<=127` and
`|w_i|<=128`, an input-width-896 gate/up integer dot has magnitude at most
14,565,376, below `2^24`; integer-to-float32 conversion is injective there.
Across all body input widths (maximum 4,864), magnitude is at most 79,069,184
and float32 rounding cells are at most eight integer units wide. Any omitted
nonzero integer coefficient has public L1 uncertainty `B>=127`, hence interval
width at least 254. Endpoint equality is impossible. The report derives this
from hashed stage geometry; no activation sampling is needed. Tighter correlated
public domains or certificates after the **entire original** scale/nonlinearity
sequence are different constructions, not counterexamples to this veto.

**Protected protocol and fallback cost.** Neither worker can test a secret
prefix's cell in plaintext. A protected comparison/selection must produce opaque
shares of the predicted word or route the exact fallback without revealing the
hit bitmap. This functionality is unimplemented and unpriced, not free. If `A`
is speculative traffic, `V` protected validation/selection, `F` complete original
fallback traffic and `h` certified hit fraction, optimistic component expectation
is `A+V+(1-h)F`, beating `F` only if `A+V<hF`. Worst case is `A+V+F`.
Preprocessing, both workers, artifacts, failed requests and all other stages add
to both expressions. Public variable fallback volume leaks hits/access patterns;
fixed-capacity padding to worst case gives zero worst-case traffic saving.
Speculation must not advance real KV/position or consume sampler RNG before
commit. Tests have no speculative state mutation and do not claim rollback code.

## Qwen 150+8 / 150+64 geometry-only gate

Decode executes `N-1` rows, not `N`. For final-layer terminal prefill pruning,
charged MLP elements are `4864*[23*(150+N-1)+N]`; unpruned alternatives remain
in the report. These are exact compiler-geometry projections, not measured
transport or a new 150+64 response budget.

| Component, decimal MB | 150+8 | 150+64 |
| --- | ---: | ---: |
| Terminal-pruned gated elements | 17,602,816 | 24,140,032 |
| Conditional six ring32-word cuts | 422.47 | 579.36 |
| One fresh half-gate AND/lane, material alone | 563.29 | 772.48 |
| Two ring32 masked input openings, both workers | 281.65 | 386.24 |

The half-gate row prices the existing uncompressed 16-byte-label layout at 32
bytes/AND, **not** every secure implementation or an AND count for original SiLU.
The cuts/openings are distinct conditional layouts, not additive totals. Original
float support, quantization, attention, state, token feedback and full logits are
still required. No new all-link win, complete-client <=256 MiB qualification,
timing, quality substitution or crypto benchmark is asserted.

## Narrow remaining hypothesis and next experiment

Best remaining lead is **source-proved equality of complete rounded nonlinear
subgraphs over a certified reachable domain**, followed by ordered fanout. It can
combine exact circuit specialization with a stronger rounding-cell certificate,
but none is supplied for this checkpoint. Syntactic row deduplication and the
full-domain pre-cast tail certificate are already vetoed here.

Next gate: parent may inspect frozen source operator metadata first, selecting
one original gate/up/output quantization chain with every scale and rounding node
bound. Derive a public interval/finite-domain invariant without private traces;
then prove two distinct source subgraphs equivalent, or prove a nonzero omitted
tail stays in one **final original rounding cell**. Include a counterexample
search at adjacent representable values, quantizer thresholds, position changes,
and signed zero. A proof must cover the original accepted domain, or use a
protected, priced range check with full exact fallback. Only after a useful
certificate exists should parent serialize tensor scanning or protected cost
work. The subsequent generated decoder gate must compare full logits, active KV
and sampled RNG state, not greedy tokens alone. Stop if certificate checks plus
padding/material exceed removed boundary bytes.

## Replay

```sh
python3 -m unittest discover -s tests -p test_exact_adapter_frontier.py -v
python3 scripts/probe_exact_adapter_frontier.py \
  --output docs/evidence/exact-adapter-frontier-2026-10-16.json
```

The report includes SHA-256 and byte lengths of the probe, tests, this reference,
and the original evidence/implementation snapshots. Regeneration hashes current
inputs; changed hashes must not be silently described as the earlier frozen run.
No dependencies, builds, checkpoint tensor loads, live cohorts or costly timing/
crypto work are required. Twenty-two bounded tests run independently of native SDK.
