# FuseFSS implementation plan

Status: bounded source decoder and ordinary SDK integration implemented and
tested on generated checkpoints. Full trained-model qualification remains open.
Separate PUMA compact-execution work has not cleared its admission gate.

## Contract

Follow [D1, D3, D5, D8, D9, D13, D14 and D15](DECISIONS.md), the
[shared computation contract](shared-computation-contracts.md#4-fusefss) and
[frozen scalar specification](../reproduction-contracts/fusefss/README.md).
The implementation is independent PLLM Rust/Python code.

D13 selects two explicit ring50/Q12 profiles:

- `fusefss.gpt2-complete-ring50-q12.v1`: learned positions, centered LayerNorm,
  affine GELU and Conv1D-oriented public projections.
- `fusefss.dense-rope-ring50-q12.v1`: source-table RMSNorm, gated SiLU and
  split-half public RoPE. Negative rotary coefficients are applied before each
  product floor; floors cannot move across subtraction.

Both preserve nearest-even external encoding, source-defined internal floors,
two online share holders, a separate trusted offline dealer, complete-vocabulary
lowest-index argmax, fixed-count full-prefix feedback and final client-only IDs.
The dense-RoPE profile is a named adaptation. The paper's missing evaluated
descriptor manifest still prevents claiming its exact evaluated numeric identity.

## Gates

| Gate | Current evidence | Remaining work |
| --- | --- | --- |
| F0: source and feasibility | Pinned equations, public model metadata and explicit cost estimates | Reprice complete trained-model execution after each scaling change |
| F1: scalar | Typed descriptors, shared DCF, affine lookup and consuming products; independent frozen GELU checks | Independent security qualification |
| F2: protected gates | Complete SiLU, softmax, LayerNorm/RMSNorm and floor/conversion gates; 275 frozen cases | Full-model domains and efficient complete widths |
| F3: semantic decoder | Five generated GPT-2/Qwen2/Qwen3/Llama controls; 918 complete semantic observations, private feedback and final IDs | Trained-checkpoint integer/domain admission; original 5,125 internal-tensor interface is not claimed |
| F4: ordinary SDK | Immutable components, source import, three actual role children, encrypted one-use tapes, SDK/gateway/CLI, failure/cancellation and standard benchmark paths | Larger admitted workloads, independent deployment and security review |
| F5: model qualification | No passing full trained-model result | Original-model and pinned common-Qwen capacity, exact integer parity, quality and matched resource/performance cohorts |

The complete method remains unqualified until F5 passes. Earlier gates supply
only their stated scopes; generated benchmarks cannot fill a trained-model cell.

## Implemented paths

- Shared absolute-position model lowering lives in
  `crates/pllm-models/src/absolute_position_decoder.rs`; checkpoint names and
  orientations remain source-mapping data.
- `crates/pllm-compiler/src/fss_decoder/` binds public source, numeric profile,
  semantic operations, fixed-count execution, resource bounds and one-use roles.
- `crates/pllm-garble/src/fusefss_backend/` supplies the common typed gates,
  exact kernels, compact private one-hot lookup and bounded material codecs.
  Immutable source-table sharing removes repeated table construction; fresh
  masks, keys and request state are never cached.
- `pllm_core::{public_file,fixed_file,streamed_matrix,integer_bounds}` supplies
  shared immutable storage, streamed public contractions and integer certificates.
  PUMA and FuseFSS retain their separate source encodings, rounding and proofs.
  Shared tokenizer validation and process-local resource reservations remain
  outside method-specific arithmetic.
- `FuseFssSourceDecoder`, `TrustedDealerTwoShareRoles` and
  `FixedCountPrivatePrefix` compose through `FuseFssSourceCpu`.
- `python/pllm/runtime/fss_*` owns bounded source import, authenticated channels,
  one-use encrypted preprocessing tapes, offline-dealer retirement and draining
  cancellation. Local children establish role separation, not non-collusion.

## Reproduce

Build/install with the standard uv/Maturin commands in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md). Use ordinary tests for correctness:

```sh
cargo test --release --locked --offline -p pllm-compiler --lib fss_decoder:: -- --test-threads=1
uv run --no-sync python -m pytest -o pythonpath=. tests/test_fss_sdk.py tests/test_fss_native_transport.py
```

[`examples/papers/fusefss_smoke.py`](../../examples/papers/fusefss_smoke.py)
constructs reproducible generated sources for both profiles. Its ordinary CLI
outputs are in [`paper/artifacts/fusefss-sdk/`](../artifacts/fusefss-sdk/README.md).
Use [`examples/papers/fusefss.py`](../../examples/papers/fusefss.py) for the pinned
common-Qwen configuration; selection does not waive its admission failures.

Benchmark reports retain dealer and holder costs, directed bodies, separate tape
I/O and setup-inclusive final disclosure. Buffered output supplies neither streaming
TTFT nor model decode throughput. No receipt archive or copied build/source tree
is required.

## Current blocking slice: joint exact signed-SiLU FSS specification only

Source-derived specification and independent numerical review are consistent:
a **SPEC STUB**, not a construction, security proof, implemented layout or achieved
cost. For the proposed new joint signed-SiLU construction, only specification and
independent review are authorized; implementation, keys and activation remain held.
D64 fully costed capability pursuit remains authorized, not new crypto, cap or
sequence increases, trained execution or adoption.
Corrected ordinary ChatML workload is **30 prefill + 2 output tokens**: it still
requires prefixes **30 and 31**, both over source cap 16, and remains over the
unchanged full-cost material budget. No silent sequence-32 change or cap rise.
All 1+1, 3+1 and 4+1 prompt/admission results below are **HISTORICAL**, unchanged;
`user: hi` = 3 there is not current ordinary corrected-30 ChatML fit or proof.

### Frozen numeric and composition contract (D14)

- Interpret canonical ring50 input `X` as signed Q12; accepted interval is
  `[-2^37, 2^37-1]`. Set `Y = floor(X/64)`; sign-extend before `abs(Y)`,
  especially at `Y = -2^31`. Define `D = [Y >= 0]`, `A = abs(Y)`,
  `B = [A < 1024]`, `J = min(A, 1023)`, `Z = D*X - T[J] mod 2^50`.
  Multiplicand is **original X**, never Y. Preserve negative-floor corners
  `(-65, -64, -1, 0, 63, 64)` and clamp changes at X `-65473/-65472`
  and `65535/65536`, although adjacent A=1023/1024 select the same index.
- Existing owners: [composition/gates.rs](../../crates/pllm-garble/src/fusefss_backend/composition/gates.rs),
  [composition/source_tables.rs](../../crates/pllm-garble/src/fusefss_backend/composition/source_tables.rs),
  [fixed_activation_tables.rs](../../crates/pllm-garble/src/fixed_activation_tables.rs).
  `fixed_activation_tables::silu_q12` is the rational-certified frozen oracle;
  no floating regeneration or new table. Reuse existing Descriptor/Plan/Party,
  tape/import, conversions, predicates and lookup owners, not parallel machinery.
- Public declared interval `InputRange` is **not secret membership proof**. Existing
  certificate or protected check must precede unsafe narrowing/arithmetic. D14 requires safe
  substitution, sticky validity and full schedule; client-only aggregate acceptance,
  no invalid tokens, reusable state or failing-location disclosure. Optional helpers
  `n,b` require fixed public descriptor and internal shares, never adaptive or clear disclosure.
- Shared masked opening already exists but queries fresh keys. `VectorComparison`
  is one-threshold/multi-payload, **not multipoint compression**. No compact key
  layout is asserted and no speculative compression is claimed.

### Astra35 refinement: exact signed partition and shared two-channel identity (SPEC ONLY)

This refines the existing [JointBasis reference](../../crates/pllm-garble/src/shared_affine_lookup_reference/joint_basis.rs),
not another implementation or approval. Its two channels share **one universal
comparison key per party**, evaluated at zero and each public interior boundary;
this does not mean one DCF evaluation for the whole table. The bounded reference
already has the consuming affine continuation. No duplicate comparison/product
API, regenerated approximation, model-specific branch or capability change is proposed.

#### Signed cells, helper semantics and constructive descriptor mapping

Write `N = 2^50`, `L = 2^37`, and decode the canonical input word `w` as
signed50 integer `x`. Only `-L <= x < L` is accepted. The actual lowering in
`composition/gates.rs::silu_inner` narrows to signed38, truncates six bits into
ring32, **sign-extends** to ring50, then negates/selects. Thus
`y = floor(x/64)` in `[-2^31, 2^31-1]`, not truncation toward zero and not
`floor(abs(x)/64)`. At `x=-L`, the ring32 word `2^31` denotes `-2^31`;
after sign extension its ring50 negation is the representable integer `2^31`.
It is not a ring32 absolute value. The frozen `D,A,B,J,Z` equations above follow.

Here is a complete half-open signed partition retaining the clamp helper `B`:

| Signed cell | Parameters | `(D,B,J)` | Channel row modulo N |
| --- | --- | --- | --- |
| `[-L,-64*1023)` | negative tail | `(0,0,1023)` | `(-T[1023],0)` |
| `[-64*k,-64*(k-1))` | `k=1023,1022,...,1` | `(0,1,k)` | `(-T[k],0)` |
| `[64*k,64*(k+1))` | `k=0,1,...,1023` | `(1,1,k)` | `(-T[k],1)` |
| `[65536,L)` | positive tail | `(1,0,1023)` | `(-T[1023],1)` |

On each negative central cell `y=-k`; on each positive central cell `y=k`.
In the tails `y` still changes at every multiple of 64, but `D,B,J` and channels
are constant. `B` changes at **-65472** (from zero to one) and **65536**
(from one to zero). In particular `x=-65473` has `y=-1024`, whereas
`x=-65472` has `y=-1023`; `x=65535/65536` has `y=1023/1024`.
All four select `J=1023`. The cells are disjoint, consecutive, and cover exactly
`[-L,L)`, including both signed38 extremes `-L,L-1`.
Their count is **1 + 1023 + 1024 + 1 = 2049**, independently derived from these
ranges, not inferred from the reference's comment. This count retains `B`
transitions, not every `Y` transition. A two-channel-only descriptor may merge
the two adjacent tail/index-1023 pairs, giving 2047 cells before any further
equal-`T` merging. Other equal frozen values may allow more merging; no minimal
count is asserted. Merging channel rows does **not** preserve optional helper
outputs automatically. In particular `Y,A` are not constant on the tails.

For an explicit canonical ring descriptor with the 2049 retained cells, use
these boundaries, in exactly the displayed order:

```
0;
64*k                    for k=1,...,1024;
L;
N-64*k                  for k=1023,1022,...,1;
N.
```

There are 2050 strictly increasing boundaries. Rows in order are positive
central `k=0,...,1023`, positive tail, negative tail, then negative central
`k=1023,...,1`, with the channel rows from the table. The negative-tail row
starts at `L`, deliberately extending its constants across the invalid gap
`[L,N-L)` and the accepted negative tail `[N-L,N-65472)`. This is an explicit
total-function extension, **not** acceptance of that gap or a membership check.
Alternatively a separate gap row would require its own boundary/count; it must
not be silently added to the 2049 claim. Valid negative cells map by `w=N+x`,
nonnegative cells by `w=x`; `N` is an exclusive sentinel, not a canonical word.
The accepted-domain endpoint `N-L` lies within the extended row. Across the
cyclic seam `N/0`, negative index 1 changes to positive index 0; no negative
threshold or translated secret boundary is passed to the evaluator. The
`D,B,J` labels outside the accepted domain have no promised source semantics.
These formulas and frozen `T` entries specify every boundary and row without
a generated table dump. Protected safe substitution/sticky validity remains
required before any use dependent on signed38 membership.

#### Telescoping and reconciliation with actual dealer/base signs

For any total ring table let `0=alpha_0 < alpha_1 < ... < alpha_m=N`,
cell `j=[alpha_j,alpha_{j+1})`, `j=0,...,m-1`, row
`v_j=(-T[J_j],D_j)` on accepted cells, and `delta_j=v_{j-1}-v_j`
for **interior** indices `j=1,...,m-1`. All vector arithmetic is modulo N.
For canonical `w` in cell `k`, strict comparison `[w<alpha_j]` is zero for
`j<=k`, one for `j>k`. Hence, including equality at every left endpoint,

```
v_last + sum(j=1..m-1, delta_j * [w < alpha_j])
  = v_(m-1) + sum(j=k+1..m-1, v_(j-1)-v_j) = v_k.
```

This is an unsigned comparison on canonical `w`, not a signed comparison on
`x`. The signed-to-cyclic mapping above makes its accepted outputs the desired
signed function. Neither endpoint 0 nor N is an interior summand.

Now reconcile [ring.rs:315–337,462–494](../../crates/pllm-garble/src/shared_affine_lookup_reference/ring.rs)
with [interval_fss::RingComparison](../../crates/pllm-garble/src/interval_fss.rs).
Let fresh full-ring mask be `r`, opening `u=(w+r) mod N`,
`t_p=(r+p) mod N`, `c_p=[t_p>p]`, and `h_p` the **sum** of both
`eval(p)` shares. From the DCF threshold `(r-1) mod N` and actual evaluator's
subtraction of `[u>p]`, `h_p=[u<t_p]-c_p` for `0<=p<N`.
Subtracting the zero-boundary evaluation gives

```
k_p = c_p - [r>0] + [r+p >= N]                 (lt_correction)
h_p - h_0 + k_p = [w<p],                       0 <= p < N.
```

The second equality follows by cutting the rotated arc at zero:
`[u<t_p]-[u<r]+[r+p>=N]=[w<p]`. This includes `r=0`, wrapped
rotations and strict equality `w=p`. Indeed for every interior `p`,
`c_p=[r>0]-[r+p>=N]`, so `k_p=0`; the formula is not a missing
negative correction. At sentinel `p=N`, `eval(N)=eval(0)` and
`lt_correction(N)=1`; at `p=0` both the difference and correction are zero.

The actual dealer initializes `corrected=v_last` and **adds**
`sum delta_j*k_(alpha_j)`, then draws two independent uniform ring words for
the left base and sets right base to `corrected-left`. Evaluators each **add**
`sum delta_j*(eval(alpha_j)-eval(0))`. Summing parties therefore gives
exactly `v_last + sum delta_j*[w<alpha_j] = v_k` above, with the actual
positive base correction and the actual zero subtraction, not their negations.
This proves ring reconstruction algebra conditional on the DCF's functional
comparison identity; it is not a correlated-key simulation/security proof.

The existing `JointParty::advance_affine/finish_affine` then consumes one fresh
existing ring Beaver triple for channel 1 times the retained **original ring50
input share**, adding channel 0 afterward. For triple `(a,b,c=ab)` and openings
`e=D-a`, `f=w-b`, reconstruction is `c+e*b+f*a+e*f=D*w`, hence
`Z=D*w-T[J] mod N`. `D` is Q0; `w,T` are Q12. No multiplication by
truncated `Y`, no extra truncation/rescale, and no reusable triple is permitted.

#### Independent oracle definition and unresolved protected refinement

Acceptance definition (not executed here): independently construct these public
cells/rows; compare selected `(D,B,J)` and reconstructed `Z` against
[`fusefss_silu_integer_campaign.py::oracle`](../../examples/papers/fusefss_silu_integer_campaign.py)
on every accepted `64*q + d`, `q=-2^31,...,2^31`, `d=-1,0,1`,
discarding values outside `[-L,L)`. This quantifies **every floor boundary**,
including tail boundaries omitted by the constant-channel descriptor. Also
check each descriptor/clamp boundary and signed endpoint at offsets -1,0,+1;
out-of-domain neighbors must follow protected rejection/safe substitution, not
the arbitrary gap extension. Include `-L,L-1`, the negative-floor corners
listed above and the four clamp neighbors explicitly. Check tail formulas for
all remaining residues, not only selected samples. Compare all 1024 entries
through `source_tables::silu()` / `fixed_activation_tables::silu_q12()` to the
independent frozen fixture, retaining u64LE SHA-256
`ff696cf63579aa28571af2c56895c06b1ed2e64ce6fab942fcbbe9454744f63f`.
No regeneration, approximate fit, oracle execution or new passing test is claimed.

The cell mapping, count, telescope, dealer/base reconstruction and affine
product identity are **symbolically proved here**. Conversions-to-selectors
remain **candidate/algebra ONLY**: replacing narrow/TR/sign-extension,
low-carry and high-wrap protected comparisons in
[`conversions.rs`](../../crates/pllm-garble/src/fusefss_backend/conversions.rs)
requires a concrete joint construction and transcript refinement. Equality of
final integer functions does not show those protected transcripts disappear.
Retaining optional helpers requires explicit share interfaces and correlated
simulation, not disclosure of helper values.

**Full-size construction/security remain UNPROVED.** `RingTable::MAX_INTERVALS`
is 513; `JointBasis::new` enforces it and excludes this full SiLU partition,
even after the two obvious merges. The bounded JointBasis candidate has no full
SiLU codec/import/provider integration. Ring material helpers are not such an
integration. Its context-bound scalar framing is explicitly unauthenticated;
do not adopt it as production transport. Reuse existing `fusefss::Descriptor`,
composition `Plan`/`Party`, and the consuming authenticated tape/transport
contract referenced above. A joint one-holder simulator must cover correlated
two-channel bases and shared DCF, mask shares, masked-input and Beaver openings,
upstream/downstream conversion views, output shares, and failure/access metadata.
No simulator may assume secret helpers or final client-only acceptance as leakage.
SIGMA common-mask work remains D63-held; leaky CPU terminals and one-bit output
masks are prohibited. This specification closes neither the construction blocker
nor FuseFSS, and licenses no geometry, memory, key-capacity or native execution.
Parent assigns independent Astra review before any further work.

### Construction, security and lifecycle obligations (missing)

- Supply concrete generation/evaluation equations and key format, plus joint
  simulator for one corrupted holder's keys, input/output shares, openings,
  prior/downstream conversions, and access/failure metadata. Simulator uses only
  prescribed holder input/output shares and public descriptor/leakage consistent
  with D14, not actual secret helper values or client-only aggregate acceptance.
- Trust: independent trusted dealer; at most one semi-honest online holder.
  Context binding is not authenticated transport. Output-ring masks must be
  full-ring uniform; joint masking of Booleans must be proved, not presumed.
- Fresh material per invocation/context/lane/epoch. Begun instances burn on
  failure/cancellation/replay; public preadmission without material invents no burn.
  Secret-invalid inputs still execute full schedule before aggregate acceptance.
- Independent numerical review found the target consistent. **NEW CONSTRUCTION
  SECURITY REVIEW NOT PERFORMABLE**: construction/layout/simulator absent; security has not passed.
  The reusable [public integer experiment](../../examples/papers/fusefss_silu_integer_campaign.py)
  and [ordinary regression tests](../reproduction-contracts/test_fusefss_silu_integer_campaign.py)
  cover finite floor/clamp/endpoints, frozen-table equality and public models of
  invalid safe substitution, sticky validity and full-schedule rejection.
  Neither was run in this source-only finishing pass; no new passing result is claimed.
  Public arithmetic/control checks are not protected membership or lifecycle evidence.
  The protected lifecycle/security campaign remains **UNRUN**: secret-invalid
  safe substitution and full-schedule rejection, failure/cancellation/replay burn,
  fresh-request isolation and joint-transcript security.

### Conditional cost only; no achieved bound

- Proposed projection **25,601,495,040 B/holder** is **UNIMPLEMENTED/UNAPPROVED**,
  not verified achieved bound. It leaves **8,758,243,328 B** of 32 GiB for
  **7,120,896 logical SiLU lanes plus all other operations**; **1229.936 B/lane**
  is only the historical 61-row optimistic ceiling, not achieved gate cost,
  lower bound or complete padded-construction/pricing bound.
- This historical logical count uses `30+31=61` rows. Actual
  `Schedule::execute` (runtime.rs:868–938) visits fixed node shapes for both
  epochs; `Schedule::run` calls `Quote::begin/tables/finish` for each visit
  (runtime.rs:492–563). The existing padded-scan regression at
  runtime_accounting_tests.rs:117–143 corroborates that pricing convention.
  D64 corrected-30 maximum extent P=31 therefore means **62 padded rows**;
  the full I=32/O=2 envelope has P=33 and **66 padded rows**, not 65.
  No revised byte tariff, complete repricing, fit or new approval follows;
  historical projection figures and tariff rates remain historical/conditional.
- Require complete ledger: projection/embedding/norm/attention/FFN products/RoPE,
  head/selection/checks/output masks, tape/frame/envelope, work and live owners.
  Retirement never refunds cumulative one-use bytes. Preserve **32 GiB material
  per holder**, existing **64 MiB native-resident budget**, separate host RSS guards,
  descriptors and work caps unchanged. No cost/admission substitution between them.

## D64 actual/configured request-envelope contract (proposed; activation held)

[D64](DECISIONS.md#d64-fully-costed-fusefss-prefix-support-for-actual-request-envelopes)
authorizes pursuit of fully costed support, not activation, cap increases or execution.
Existing facts below identify reuse points; metadata and regressions are proposals.

### Envelope identity and phase semantics

| Envelope | Input I / outputs O | Logical phase prefixes | Status |
| --- | --- | --- | --- |
| Corrected ordinary request | 30 / 2 | 30, 31 | Both exceed native sequence 16 |
| D59 bounded campaign ceiling | 32 / 2 | 32, 33 | Requires fresh grant and admission; exceeds 16 |
| Common configured workload | 64 / 8 | 64 through 71 | Exceeds 16 and selection ceiling 4 |

The configured workload is defined in [common.py:11](../../examples/papers/common.py#L11).
D59's 32/2 campaign does not authorize 64/8; configured capability is not permission.
Selection ceiling 4 is independent of sequence extent and separately rejects O=8.
Propose one shared model-/method-neutral validated phase-metadata contract on an
existing admitted owner: positive I/O, checked `P = I + (O - 1)`, `0 <= s < O`,
logical prefix `I + s`, padded execution extent P. SDK public maximum-extent shape
validation is distinct from runtime active prefixes: P must not replace `I + s`
in masks or final-row selection (runtime.rs:1595, 1617, 1674). Derive metadata from
admitted configuration; no caller boolean, copied authority record or separate planner.
Reuse `SourcePreflight.maximum_sequence`, Program initial/selections/sequence,
channel Geometry and SDK contract identity rather than creating parallel owners.
Preserve shared template/BOS rendering and actual-token admission; a shorter
request never substitutes for full saved-envelope pricing.

### Existing owners validation and quotation

Native anchors below are relative to
[`fss_decoder/`](../../crates/pllm-compiler/src/fss_decoder/):

| Existing owner | Source-backed constraint / reuse obligation |
| --- | --- |
| [program.rs:457–498](../../crates/pllm-compiler/src/fss_decoder/program.rs#L457) | Checked initial-plus-feedback extent; separate hard-16 graph/state guards |
| [channel.rs:66–81](../../crates/pllm-compiler/src/fss_decoder/channel.rs#L66) | Geometry validates against default limits, including checked combined extent |
| [sdk.rs:384–456](../../crates/pllm-compiler/src/fss_decoder/sdk.rs#L384) | Composition validation and checked sequence admission |
| [benchmark_delegated.py:113–141](../../python/pllm/runtime/benchmark_delegated.py#L113) | Benchmark options retain saved envelope, native 16/4 and full-prefix checks |
| [fss_source.py:144–159](../../python/pllm/runtime/fss_source.py#L144) | Bind configuration, composition, source, tokenizer, counts and numeric profile |
| [benchmark_memory.py:1144–1187](../../python/pllm/runtime/benchmark_memory.py#L1144) | Imported full-envelope pricing requires live LoadedFssSource and matching native Program; not a lower prompt price |

Existing [runtime.rs:868–938](../../crates/pllm-compiler/src/fss_decoder/runtime.rs#L868)
runs every Program node for every selection at fixed `p.sequence` shapes, with
private feedback. [sdk.rs:929–977](../../crates/pllm-compiler/src/fss_decoder/sdk.rs#L929)
expands sequence and attention axes to P. Logical growth is not dynamic work:
no triangular-sum discount. Existing [runtime_accounting_tests.rs:117–143](../../crates/pllm-compiler/src/fss_decoder/runtime_accounting_tests.rs#L117)
prices three selections at four rows each, not 2+3+4. A cheaper unpadded schedule
would require separate review.
Retain checked shape products in [sdk.rs:650–661](../../crates/pllm-compiler/src/fss_decoder/sdk.rs#L650)
and vocabulary/graph/storage bounds in [capacity.rs:25–69](../../crates/pllm-compiler/src/fss_decoder/capacity.rs#L25);
price actual shapes and products, never invented lane counts.

### Complete admission and lifecycle obligations

Reuse existing `preflight -> Schedule::execute` and `Schedule::run ->
resource_quote::Quote::{begin,tables,finish}` paths: runtime.rs:254–286, 492–595.
[resource_quote.rs:75–189](../../crates/pllm-compiler/src/fss_decoder/resource_quote.rs#L75)
accumulates every visited protected embedding, projection, norm, attention,
activation, product, head, selection and checked-validity kernel's work, material,
frames and peak owners. Its lines 1–4, 41–70 admit only an initial bootstrap floor,
not complete cost. Preserve checked arithmetic, cache-delta error ordering,
invocation contexts and obligation hashes; do not introduce another quote/tariff.

[program.rs:50–87](../../crates/pllm-compiler/src/fss_decoder/program.rs#L50) retains
sequence 16 / selections 4 even for Decoder; Decoder work, material, frames and
public-copy limits remain distinct. Preserve combined native admission at 64 MiB
and documented 32 GiB per-holder material scope. Neither substitutes for native
aggregate material, source, tape, RSS, descriptor or kernel caps. Aggregate versus
per-holder reconciliation remains OPEN; no fit claim follows from either number.
Reuse `SourcePhaseAdmission` / `phase_admission` in sdk.rs:459–496, 552–623 for
serialization, ingestion and compile-preparation overlap, not merely payload size.
Proposed D64 requirement: bind and charge initialized source owners before cache
or schedule construction, including all failure-live overlap. runtime.rs:385–409
and resource_quote.rs:41–70 provide bootstrap reuse points, not proof this complete
overlap is admitted; the initial bootstrap floor is insufficient.

[fss_transport.py:101–178](../../python/pllm/runtime/fss_transport.py#L101) routes
native packet/record pricing into `MaterialTape.price` for tape/replay; receive and
batch reservations remain separate `TransportResources` fields (lines 169–178).
Admit combined concurrent overlap without collapsing caps. Retain all live source,
cache, frame, tape and request owners through drain and actual retirement;
cancellation is not an early refund.
Keep runtime.rs:906–918 sticky-invalid full schedule and final aggregate mask,
and consuming `Worker::execute` obligations/cancellation at runtime.rs:207–249.
[fss_transport.py:497–535](../../python/pllm/runtime/fss_transport.py#L497)
reconstructs only after holder finish. Require fresh material/context; no reuse,
invalid output or intermediate-token disclosure.

### Approved design-only: opt-in failed-parent custody

The user approved design-only integration of partial source proofs, NOT source-
authoring readiness, funding adoption, implementation or test authoring, execution,
or any native HOLD change/release.
The shared, method-neutral policy is OPT-IN, initially only for Python worker
`_Attempt`. Preserve legacy `_CleanupSlot` callback retry/default behavior and
`drain_failed` owner retry; uncertain backend exit remains one-shot
under the existing uncertain-backend-exit contract. No FSS registry is proposed. The broader native custody endpoint in
[the SDK reference](../../docs/research/fusefss-sdk-reference.md#L468) remains
UNSELECTED. Source-import-local rewrites are excluded; broader D64, caps, PUMA and
HE scope remains unchanged.

The embedding caller supplies an authentic ALREADY FUNDED entry region and retained
return destination shared by FuseFSS, ROSETTA and source binding. Establishing that
external owner is an explicit trust boundary, not a recursively self-funded
constructor. This responsibility is selected; actual finite coverage, tariffs,
concrete API/owner types and implementation execution are NOT adopted. Original
ledgers and payload/backing remain charged through actual retirement. Narrow
checked-publication exclusions are NOT selected.

The following are conceptual states, not existing code or API. OWNED retains the
original parent, owners, token, manager and charges. CLOSING latches before
cancel/drain; reentry observes the existing attempt. OWNER_RETIRING permits
dependent-owner retirement only after genuine drain. FAILED_OR_UNKNOWN never
retries the failed or unknown operation and retains unresolved original custody
and charges; qualified positive child-retirement witnesses still matter.
RESOURCE_RETIRED, BACKEND_RETIRED, COMMIT_PENDING and COMMITTED remain distinct:
parent closed is not resource retirement, manager exit, or ledger commit.
One-shot accounting follows each actual operation and extent, including
cancel/close aliasing; it does not ban a normal protocol pair. Retain original
references before wrapper detachment. Bookkeeping follows only authenticated
original-ledger release; token absence alone does not authorize recovery.

Reuse `resources.py`'s `_ClientCapacity._cleanup`/`_owners` and module-rooted
`_CleanupSlot`, subject to durable-binding proof. Its quote, 64 KiB + 5 * (dictionary
backings) + 1024, describes logical control bytes, not lifecycle work or payload-
backing proof. The 1024 cap bounds `_cleanup`, NOT `_owners` N; dictionary backing
depends on prior peak. `hold(select)` (216–245) accepts a zero-cost callback without
local positivity validation: arbitrary finite N interfacewise, not a canonical-
planner claim. `retain` lacks owners-membership/amount validation; `owns` checks
identity/membership, not amount. The proposed prearmed slot is absent in the worker.

The actual first path is [fss_worker.py:1086–1109](../../python/pllm/runtime/fss_worker.py#L1086)
→ `_Reservation(reserve_incremental_memory(role_memory))`; delegated_worker.py:65–72
enters the manager before wrapper installation. resources.py:289–302 / 191–204
observes BEFORE `_owners[token]` installation. Pre-observe helper calls are LINKs,
not fees. Linux resources.py:60–124: accepted text limits are 65536/131072/64
characters; each read requests one additional detection character. With M usable
matching mounts, M ≤ 131072, bounded-text calls ≤ 2 + 512M. These source bounds
prove no private cost, CPU/RSS, complete funding or runtime qualification.

The actual first Darwin observation can initialize a process-lifetime source
(benchmark_memory.py:85–91 → host_memory.rs:167–205); old unqualified initialization
cannot be retrofunded. The [shared bounded observer lifecycle](../../docs/plans/client-resource-negotiation.md#bounded-observer-lifecycle-design-only)
is selected prospectively, NOT implemented or authoring-ready; no cap or HOLD change follows.

The exact release audit is COMPLETE, not pending.
[`_CleanupSlot.drain`, resources.py:585–589](../../python/pllm/deployment/resources.py#L585)
clears `_payload`/`_retire` and marks closed BEFORE `release_cleanup`.
`release_cleanup` (161–164) checks SLOT IDENTITY ONLY, removes the `_cleanup` root
FIRST, then the `_owners` charge SECOND. The callback-only exception handler does
not cover release. Failure can leave a closed, empty, registered slot with an
unchanged charge, or remove the root before charge pop. Normal-path local
references are not durable post-failure custody. A separately proved durable
binding and control-ledger commit are required; `_MemoryReservation` transaction
or recovery guarantees CANNOT transfer by analogy.
Python GIL/RLock supplies neither commit atomicity nor recovery. The bounded audit
found no existing native control-ledger primitive; a proposed native journal boundary
remains UNSELECTED and needs concrete callback/signal/failure-corridor proof. The
same external owner may prospectively cover a bounded tail; actual funding or a
new keeper does not follow.

Source obstacles remain concrete. Worker admission orders `_Reservation` →
`_Attempt` → active → request state BEFORE source/native/tape. Manager entry
precedes wrapper installation; `_Attempt` construction is fallible; disk entry
precedes append. Unreturned constructors need separate handoff proof.

Actual worker paths expose a SOURCE GAP against the prospective requirement that retirement/release remain possible after F:
Under the shared native-authority contract linked above, controlled cancel must resolve the already-funded AUTHENTIC native owner ↔ actual Attempt/retirement custody ↔ authorized target ↔ original capacity token/outcome BEFORE Python active/seen/closed selects temporary admission (fss_worker.py:1067–1076); capacity identity alone grants neither FSS authorization nor cleanup permission. For an original live, not-yet-attempted owner, only authorized funded original cleanup is permitted; join in-progress cleanup, retain failed/unknown originals with NO RETRY, and fail closed on unresolved state with its retained funded outcome.
Python map absence/closed/counters prove neither native Retired nor NoOwner; native-proven absence requires authenticated lookup-scope coverage. Fresh reservation after F cannot recover old custody; process-local HTTP-response funding remains separately required, not supplied automatically by a client token/original borrow. Existing best-effort cancellation/ACK semantics remain unchanged; actual native binding remains unproved, with no deadlock claim or implemented repair.

Under the SELECTED native whole-publication design, the authentic
original-owner/authorized-target/terminal-outcome handoff must be PREARMED under
an already-funded original entry BEFORE fallible reservation-manager entry
(`fss_worker.py:1091`; `delegated_worker.py:65–72`, entry at 69) and BEFORE
`_Attempt` construction/publication (`fss_worker.py:783–785`). The original
authority must retain the actual process/service-generation identity, authorized
target scope and original terminal destination, with gap-free custody of the
actual Attempt, reservation manager, token and ledger as acquired; prearming does
not claim that not-yet-created resources exist. Partial failures, including
unreturned constructors and request-ID assignment at `793–797` before epoch
validation at `807–810`, remain attached to that original authority/outcome
without treating an assigned ID as an authorized target. Request IDs, epoch labels
and Python maps confer no authority. This original binding must remain resolvable
before cancellation’s projection-dependent temporary-reservation selection at
`1067–1076`; unknown outcomes retain genuine original custody and charges without
retry or inferred refund. Worker-local lookup, refusal and response funding remain
separately required from old-attempt cleanup funding; client tokens or
original-owner borrowing do not supply them automatically. Concrete native binding
remains UNADOPTED and unproved: no types/API, registry, receipt, implementation or
authoring grant follows. Existing observer-F admission boundary and best-effort ACK
semantics remain unchanged; joining retirement does not prove discharge.

| Path and source (`fss_worker.py` unless noted) | Actual custody and outcome | Prospective F consequence / limit |
| --- | --- | --- |
| Live matching worker: 1097–1109; operation 863–868 | Uses original worker reservation, signals and constructs `retired: true`. Finally 1132–1150 joins ONLY when no existing `close_task`. | Response is not universal cleanup completion or a destruction receipt. |
| Matching retiring phase != closed with reservation: 1072–1084 | Reuses original root; handler/reply leases precede first await on the same loop only, not a global interruption proof. `await retire` joins original task; failure rethrows without backend retry. | Joining is not a whole-physical-retirement certificate. |
| Closed with recorded cleanup errors: 282–295; 368–375 | Root restored, but native/tape/loaded/client owners cleared. Closed branch 1066–1084 needs NEW temporary `role_memory`; without F, old failed task rethrows. | Exhaustion would refuse before leases/body/join. Retained charge does not restore cleared-owner custody or prove retirement. |
| Seen ID, no matching candidate: normal close OR reservation-close failure at 289–301, 371–375 | New temporary reservation, no join. Failure detaches root without restoration and clears active. Absence, `retired_sessions` and body cannot distinguish failed manager exit; normal response leases may still hold charge. | F could refuse ACK even after genuine original retirement. `_Reservation` (delegated_worker.py:79–88) flags closed/detaches manager before exit; failed release remains unknown. Not every closed path proves retirement. |

FSS fss_transport.py:15, 444, 562 uses shared `_Peers`; delegated_transport.py:687–711
explicitly makes remote close best effort, NOT acknowledged destruction, swallowing ordinary
cancel errors and relying on expiry. Conditional idempotency does not guarantee a funded ACK.
tests/test_fss_worker.py:524–554 covers nominal repeated 200 responses/two joiners only,
not exhaustion or whole physical retirement. Ordinary temporary-admission errors can yield 400;
outer-finally retirement errors can prevent delivery. Neither implies universal deadlock.

Temporary `_Reservation` immediately enters its manager (delegated_worker.py:65–72 →
resources.py:192–204 observation). Original worker leases and client failed-owner retention
are distinct PROCESS-LOCAL charges: a client token cannot fund remote HTTP or prove complete
backing-byte/lifecycle-work coverage. Even refusal needs an authentic ALREADY FUNDED external entry and retained
return destination under the selected trust boundary; actual canonical binding is unproved.

This source gap selects no repair/product change, weakens no idempotency contract and promises
no stronger ACK. Qualified resource-specific RAII may prove retirement without positive ACK;
all source/test authoring and execution remain held.

Tape (299–310) clears its file before close and swallows `Exception`. Its actual
`TemporaryFile(mode="w+b", buffering=0)` (fss_tape.py:264) is FileIO, NOT BufferedRandom,
on canonical stock non-Cygwin POSIX CPython. Under UNADOPTED CPython 3.13.0 plus
Linux-style close-result semantics, actual complete close and no escaped fds,
mappings or in-flight use, error exits CAN prove fd retirement; generic POSIX/EINTR
alone cannot. Resource-specific ordinary RAII witnesses are allowed, not blanket
positive acknowledgements. Source (505–508) marks closed early but retains `native_program`. Native
Request close/cancel can remove an owner before error: error does not prove it
still live. NativeProgram.close (crates/pllm-python/src/fss_source.rs:359–361) drops
ONE Arc; completed Rust dealer error return (fss_transport.rs:668–760) proves its
local owner retired, not all aliases. `None` fields, flags or cancel return do not
prove drain. Require resource-specific witnesses, not a blanket leak claim.
`native_call` cancellation hooks, inner-task cancellation and callback publication
leave executor completion and quiescence unproved.

BEFORE ANY source or test authoring, independently prove durable binding/release;
finite prearm, installation and validation plus lifecycle funding; constructor
handoff; producer drain, child witnesses and storage overlap. Ordinary secondary
errors must not replace an established primary error. Policy for process-control
`BaseException`s, direct cleanup cancellation, exception groups, and original
versus sanitized terminal reporting remains OPEN; no user adoption is inferred.
Frame clearing leaves `JSONDecodeError.doc` / `UnicodeDecodeError.object`; metadata
input is capped at 4096 bytes (fss_protocol.py:14, 43–56), not universally unbounded.

Bounded failure oracles remain UNAUTHORED prose: parent-reference loss,
interrupted installation, partial child retirement, cancellation before executor
completion, unknown backend exit, interrupted ledger release, repeated entry and
unchanged legacy behavior. They must establish original identities, correct
charges, no retry of selected failed/unknown operations, and the selected error.
No CPU, RSS, termination or interruption-immunity claim follows. Remaining independent
proofs/review come next; no source execution is authorized.

### Proposed ordinary regressions—unrun

These are regression definitions, not implementation or new passing claims:

- Existing `sdk_tests`, `channel_tests`, `capacity_tests`: zero/overflow refusal,
  unchanged 16/4 boundaries, and actual-versus-configured identities for all three
  envelopes above; proposed metadata must not turn an over-cap request into admission.
- `runtime_accounting_tests`: public preflight versus driven walks agree on padded
  shapes, resources and obligations, not triangular-prefix pricing. Explicit unrun
  gate: validate SDK maximum extent P separately from runtime `I + s` masks and
  final-row selection; reject a future all-P active-prefix substitution.
- `checked_range_tests`: invalid inputs still run full schedule; final-only rejection.
- `transport_tests`, [test_fss_native_transport.py](../../tests/test_fss_native_transport.py)
  and [test_fss_sdk.py](../../tests/test_fss_sdk.py): fresh contexts, replay refusal,
  cancellation/drain and actual owner retirement.
- [test_fss_checkpoint_admission_guards.py](../../tests/test_fss_checkpoint_admission_guards.py)
  and [test_fss_topology_admission.py](../../tests/test_fss_topology_admission.py):
  full saved-envelope source-owner checks and complete-topology refusals.

Ordinary cargo/pytest commands in [Reproduce](#reproduce) (lines 66–72 above) are
**PROPOSED / UNRUN for this slice** and require a fresh parent grant, not launch here.
Preserve existing typed refusals; do not invent an `Error::Bounds` variant.

### Independent review handoff and activation blockers

Review existing facts separately from proposed phase metadata and unrun regressions.
Complete funding and live-owner admission remain OPEN. Signed-SiLU construction,
implemented layout and joint simulator remain MISSING; trained numeric/topology
admission, final acceptance and quality remain OPEN. Independent review and complete
admission precede activation. No silent cap rise, numeric/security/protocol change,
clear fallback or execution clearance; native 64/8 capability would not grant use.
Historical cohorts below retain their original scope and results unchanged.

## Next blocking slice (historical record; not corrected-30 ChatML evidence)

Prioritize pinned Qwen benchmarking. Exact-key indexed replay owners and
framing-derived tape admission now support the larger generated Qwen SDK control,
including file-backed Decoder execution. Native tests preserve all 186 independent
Qwen observations; three-role tests preserve three selections and failure cleanup.

The static profile's earlier 1+1 checkpoint import rejects numeric admission
before workers launch. An offline Gram/Hadamard-lift diagnostic certifies the
layer-zero residual, but has neither full-layer coverage nor a bounded production
owner. Later conditional propagation does not certify earlier obligations.

The approved [D14](DECISIONS.md#d14-opt-in-checked-ranges-for-fusefss-qwen-completion)
profile now passes actual pinned-Qwen source, numeric and work admission at 4+1.
It preserves D13 arithmetic on accepted requests and rejects invalid requests only
at final aggregate acceptance. Shared private predicates/selection and their full
schedule are charged. Generated three-role tests cover late rejection without
token disclosure and fresh-request recovery. Actual checkpoint request acceptance
is still unmeasured; static proof improvements remain a separate follow-up.

Next: calibrate and reduce online execution overhead, complete the ordinary short
Qwen request, check independent complete integer results, then quality and matched
benchmarks.
The initial inline-metadata tape quote needs 71.53 GB including reserve; the host
rejects before child launch. External-AAD tape storage reduces the quote to
47.63 GB without removing authenticated metadata, material or replay claims.
Generated Decoder tests cover this format through actual role processes.
A separate small serial-HTTP benchmark attributes 97.82% of dealer preparation
to delivery callbacks. Explicit bounded delivery batching now preserves original
one-use records, complete acknowledgements and numeric identity through ordinary
SDK execution. A fresh-client generated pair lowers preparation from 5.393 to
0.390 seconds with matching outputs, tape bytes and online bodies. This remains
separate from trained-Qwen execution and independent performance confirmation.
A subsequent checked-range/batched Qwen attempt passes topology admission and
reaches online execution, but the host memory-pressure guard stops it after about
66 minutes, before final acceptance and before its four-hour deadline. The
[ordinary failed checkpoint](../artifacts/fusefss-sdk/qwen-checked-batch-pressure.json)
retains no completed report. All role processes retire. The cause of host pressure
is unassigned. Scalar online gates still incur millions of Python/HTTP callbacks;
measure a representative bounded fixture and reuse shared transport/scheduling
owners for dependency-safe batching before another long checkpoint run.
Launch uses bounded temporary source qualification, fresh exact
topology pricing and independently checked worker imports. Qualification cost,
all source/tape/replay disk, cumulative material and execution work stay charged.
The common 64+8 workload still exceeds sequence/selection bounds. Preserve the
approved arithmetic and role graph; obtain approval before substantial protocol,
numeric, state or trust adaptations.

### WAN requirement

Loopback completion does not close F5. Use the ordinary three-role benchmark
under shared 100/40 Mbps party caps and a declared inter-party delay, starting
with the 20-ms-per-direction profile in `examples/benchmarks/wan_20ms.json`.
The shared native relay now routes dealer delivery and both holder-peer paths;
its bounded propagation queues and report validation are transport qualification.

The scalar schedule has 13,424,643 peer frames per Qwen holder. Merely replacing
HTTP or batching offline delivery does not remove those dependent waits. D15's
explicit `online_schedule="wavefront-v1"` now groups independent tensor lanes and
uses balanced stable selection. Every lane still consumes fresh, context-bound
material and exact replay claims; frontier storage is priced before issuance
under the existing live limits. Public group and tile bounds still split logical
tree levels into multiple exchanges.

The core passes 105 composition checks; 115 decoder checks include all five
frozen semantic outputs and the checked-profile control. Four installed
three-process checks cover generated Decoder feedback, late rejection/recovery,
Capsule execution and buffered gateway output. A fresh-client matched generated
pair at 100/40 Mbps plus 40 ms added RTT lowers full requests from 222.505 to
34.583 seconds, with identical output commitments and zero observed swap growth.
The [ordinary report](../../docs/evidence/fusefss-wavefront-wan-pair.json) retains
that tiny source and separate cohort.

Pinned-source 4+1 wavefront qualification still rejects: repeated public schedule
construction pushes the kernel walk past the existing 32 GiB public-copy ceiling
at layer 23 SiLU. A separate 1+1 control admits but retains 29,334 sequential peer
exchanges per holder, roughly 19.6 minutes of waiting at 40 ms RTT before compute.
Its packet allowance is 5.18 GB per holder, an upper-bound price rather than a
measured transfer. No private request runs in either qualification control.
That result motivated compact public schedule ownership and wider bounded tensor
waves; increasing capacity ceilings is not the solution.

The subsequent explicit `online_schedule="tensor-v1"` shares instruction,
register-description and dependency-span templates while retaining separate
private registers and fresh material. Its compiler-owned ordered tree removes
redundant private index comparisons without changing lowest-original-index ties.
The composition and decoder suites pass 110 and 118 checks; the installed wheel
passes 408 focused checks plus four three-process SDK/gateway checks. The
[new generated WAN pair](../../docs/evidence/fusefss-tensor-wan-pair.json) observes
35.749/34.732-second wavefront/tensor requests, equal output commitments and zero
swap growth under the same caps and delay. Its separate cohort is not a trained
performance result.

Historical pinned tensor 4+1 qualification rejected at the output head, with
35,178,175,330 public-copy bytes against the unchanged 34,359,738,368-byte ceiling.
The separate 1+1 control admits 19,229 peer exchanges per holder and 7.45 GB of
complete native material. About 12.8 minutes of RTT waiting at 40 ms remains
before compute; packet allowances still reach 4.17 GB per holder before HTTP
metadata. No workers launch in these source-only controls. More dependent-round
and material reduction remains necessary before another long trained request.

With shared immutable public chunks, fresh installed-wheel source-admission tests
now pass at both 1+1 and 4+1 under the unchanged ceilings. The earlier cumulative
copy rejection is no longer the current source gate. These tests perform the
complete native numeric/resource walk but create no tapes or workers. A separate
opt-in test now prices native transport from the live qualification owner, retires
it, and requires fresh whole-topology memory/disk admission and reservations.
That gate, followed by an accepted private SDK request, remains distinct from
source admission. Shared tables do not establish fewer dependent rounds or fresh
private bytes; those still require repricing and measurement.

The separate complete-topology admission passes at 1+1, including ordinary
logical memory/disk reservations after source retirement. Before the one-hot
staging fold, the 4+1 case rejects during native transport pricing, before final
host/disk admission. A rebuilt-wheel diagnostic retry identifies
`aggregate_reservation`: 68,913,145
bytes against the unchanged 67,108,864-byte cap, exceeding it by 1,804,281 bytes.
The SDK reference records the complete public owner terms. Reduce actual
ownership/copy overlap rather than discounts or ceilings. No material, workers
or private requests run in either case. Trained serving and quality remain
unqualified.

The one-hot staging fold (`607813fd`) removes actual full-domain backing while
preserving individual-party output bytes and source work. Fresh installed-wheel
retries pass the 1+1 complete-topology control and clear 4+1 native transport
admission at 66,784,028 bytes, 324,836 bytes below the unchanged cap. The complete
pricing walk reduces each workload's combined reservation by 2,129,117 bytes;
per-party allocation savings alone did not establish that result.

The 4+1 retry now reaches final host/disk admission, where disk rejects:
45,517,613,936 required bytes, including the unchanged 2 GiB reserve, exceed
41,405,534,208 available bytes. Host memory admits at 4,068,474,880 required bytes
with normal pressure. Do not remove live owners, weaken reservations or raise
caps to force acceptance. Complete-topology admission must pass before worker
launch; no material, workers or private request ran in this retry. Logs remain
ordinary pytest output under ignored `target/fusefss-one-hot-fold-topology-{1,4}.log`.

The admitted 1+1 control does not fit ordinary SDK rendering of short prompts:
the pinned tokenizer encodes `user: hi` as three tokens. Preserve the Responses
role-prefix contract; raw-token or prompt-format bypass is not SDK acceptance.

The separate `qwen_checked_tensor_minimal_smoke` 3+1 factory fits that rendering.
Its guarded installed-wheel complete-topology check passes without changing
guards: native combined reservation 64,155,060 bytes, required host memory
4,068,474,880 bytes with the normal 25% margin, and required disk
37,476,300,632 bytes including the 2 GiB reserve. Qualification owners retire
before graph reservation. No workers, tapes, material or private request run
in this admission test (`target/fusefss-one-hot-fold-topology-3.log`). Actual
ordinary SDK execution remains a separate acceptance gate; 4+1 remains
disk-blocked on this host.

The subsequent ordinary 3+1 SDK benchmark stopped with exit 6 at
`macOS memory pressure is warning`. Its normal progress report marks the
candidate failed, with no completed result (`target/fusefss-qwen-3-plus-1-sdk.json`);
CLI output is the adjacent `.log`. Do not count the attempt as private-request
acceptance, quality or measured throughput. Keep pressure guards enabled and
resample live capacity before retrying.

Round reduction alone does not solve cold-request bytes. Reprice and measure the
complete fresh-material delivery, source/setup scope, online traffic and aggregate
resources before another long trained run. Require generated integer/feedback
parity and failure retirement first, then a bounded delay-injected response, then
pinned-Qwen acceptance, quality and matched WAN measurements. Keep warm/offline
pre-positioning separate from a setup-inclusive result.

See the [SDK contract](../../docs/research/fusefss-sdk-reference.md) and the
[historical scalar cost inventory](fusefss-decoder-feasibility.md) for separate
current and earlier scopes.
