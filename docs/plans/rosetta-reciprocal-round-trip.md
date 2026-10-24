# ROSETTA: proposed four-segment reciprocal round trip

**D37 / R-S design only, 24 October 2026; source baseline `7fa07d74`.**
No backend adoption, numeric deployment, key generation, build, encrypted test or
method-completion claim. Memory-campaign metrics/claims remain unchanged.
**Executable specification incomplete:** HomComp internals, phase-faithful index
quantization, exact cryptographic manifest, joint security/error analysis and
complete owner bounds are missing.

## 1. Evidence, choices and reusable owners

| Class | Binding |
| --- | --- |
| Known paper/source | [Contracts §§7.2–7.4](../../paper/methods/encrypted-computation-contracts.md#72-segmented-lut-algorithm-preserve-the-encrypted-accumulator), [independent F6](../../paper/reproduction-contracts/encrypted-inference.md#f6--rosetta-requires-blind-rotation-of-encrypted-accumulators-not-generic-public-lut-pbs): four segments, 2048 local entries each; encrypted affine accumulators; ring degrees 65536/2048, final LWE dimension 1024; source rounded 53/41-bit moduli, 13 usable CKKS levels, StC/CtS/EvalMod depths 3/4/8. These are not exact primes or qualified parameter sets. |
| Proposed here, not evaluated-source numerics | The finite input grid, half-open range, coefficients, phase/scale conventions, independent small/LWE keys, level-4 endpoints, rounding/error targets and ceilings below. No inherited paper precision, calibration or performance. |
| Missing evidence | Author-evaluated manifest/authority and exact HomComp algorithm attachment; RNS/special primes, decomposition, error/secret distributions and encapsulation graph; DFT/EvalMod coefficients/schedules; circular/related-key and circuit-privacy analysis; primitive error/failure and allocator-inclusive bounds. Approval supplies none of these. |

Reuse [coefficient layout](../../crates/pllm-core/src/coefficient_layout.rs) for
negacyclic monomials, extraction and odd-modulus inverse/trace algebra (already
bounded to degrees ≤4096). [Repacking](../../crates/pllm-core/src/coefficient_repacking.rs)
frames are BFV/degree4096, not an interchangeable CKKS or LWE transport. The
[coefficient converter](../../crates/pllm-core/src/coefficient_conversion.rs) is
clear bounded arithmetic, not encrypted conversion or an error proof.
Reuse [level/scheme planning and shared SSA](../../crates/pllm-compiler/src/encrypted_level_schedule.rs),
its [trace bridge](../../crates/pllm-compiler/src/encrypted_level_trace_bridge.rs)
and graph/owner validation. No parallel ROSETTA planner, sampler or authority.

Recovery means **ModRaise + CtS + EvalMod + actual scale restoration**, never a
metadata level increase. [Existing refresh contract](../../benchmarks/native/poulpy_refresh/contract.json)
and [Rust backend review §4](../../paper/methods/rust-he-backend-qualification.md#4-poulpy-substantial-current-capability-material-representation-difference)
describe a distinct, blocked torus qualification: its widths, errors and keys do
not qualify this RNS round trip. No torus inverse-trace or Lattigo adoption.

## 2. One exact proposed scalar function

Define `RNE` as mathematical nearest integer, ties to even, including negatives;
generate everything with exact rational/integer arithmetic, never binary64 fits.
The qualification domain is **only** `x=X/2^20`, integer `2^20≤X<16·2^20`.
This constrains public test fixtures; it does **not** quantize an arbitrary model
intermediate or establish a private-input range. Such input lacks admission.

For `s=0,1,2,3`, set `tau_s=2^s`, `tau_4=tau_end=16`, `Nbr=2048`,
`h_s=2^(s-11)`, `l_si=tau_s+i·h_s`, `u_si=l_si+h_s`, `0≤i<2048`.
Use reciprocal secants, not four total affine pieces:

```text
a_si = -1/(l_si*u_si)                 b_si = 1/l_si + 1/u_si
aHat_si = RNE(2^24*a_si)/2^24         bHat_si = RNE(2^44*b_si)/2^44
S = sum(j=1..3) [X >= 2^20*tau_j]
I = floor((X - 2^20*tau_S) / 2^(9+S))
phi(x) = aHat_SI*x + bHat_SI
```

This generator uniquely fixes all **8192 coefficient pairs**; q-dependent
encoded operands below become concrete only when the exact primes are supplied.
At a segment boundary, equality chooses the **right** segment, `I=0`. At every
local boundary equality chooses the right interval; immediately preceding grid
point chooses the left interval. `x=1` is `(S,I)=(0,0)`; largest valid grid point
is `(3,2047)`. **`x=16`, below 1, off-grid and above 16 are outside this proposed
contract, not clamped, wrapped, extrapolated or silently mapped to entry2047.**
Reject known invalid fixtures before encryption. This is an explicit half-open
adaptation of the paper's displayed closed range; no encrypted runtime range
test/decryption is inferred. General-model endpoint handling remains blocked.

For exact secants, relative error is `(x-l)*(u-x)/(l*u)`, at most
`1/(4*2048*2049)`; absolute error is no larger on this domain. Quantizing the
slope contributes `<2^-21` absolute error (`x<16`); intercept contributes
`≤2^-45`. These algebraic bounds do not cover encryption or conversions.

## 3. Representations, HomComp and floor without a rounding/clamp shortcut

Use `R_H=Z_Q[T]/(T^65536+1)` and `R_S=Z_Q[Y]/(Y^2048+1)`, `Y=T^32`,
rank one after switching. Coefficient ciphertext convention is `b+a*s=m+e`;
sign changes for another primitive API require a proved adapter, not relabeling.
The proposed exact prime manifest must have distinct odd primes `q0` (53-bit),
`qj` (41-bit, `j≥1`), each `1 mod 131072`; `Q_l=product(q0..ql)`.
Exact prime values, auxiliary moduli and roots are **unknown**. Full largest
simultaneous cryptographic modulus including auxiliary material must respect the
source 1763-bit envelope; that bound is not a composite security proof.

Input/output CKKS scale is `Delta=2^36`. Proposed StC layout is
`coefficient[2j]=Delta*Re(slot[j])`, `coefficient[2j+1]=Delta*Im(slot[j])`.
Only logical slot0 contains `x`; other 32767 complex slots are zero. Inverse
recovery returns `phi(x)` to slot0 and zero to all padding slots. These are
transform semantics, not an implemented/certified DFT matrix or cost-free map.
The small-ring `C_x` retains scale `Delta`, so its constant phase is `2^16*X`.
Produce companion LWE `C_X` by **modulus restriction** `Q1→q0` preserving scale,
coefficient extraction and explicit key switch. Do not confuse restriction with
rescaling by q1 or torus rounding.

**HomComp:** require the paper's high-precision comparison, three calls, with
input phase `2*C_X - 2*2^16*K_j + 2^16`, `K_j=2^20*tau_j`; its semantic bit
is `[X≥K_j]`, including equality. Output is LWE under `s_L`, phase
`beta*bit`, `beta=RNE(q0/4096)`. Sum the three ciphertexts for encrypted `S`.
Proposed input-error requirement is `|e_X|≤2^16/64`; threshold separation is at
least one integer X unit before errors. HomComp must preserve this margin and
the output rotation cell; a low-resolution sign/PBS promise is insufficient.
**Internal source algorithm, PBS/decomposition schedule and its precision proof
are not recovered.** Interface semantics are frozen, implementation is blocked;
no generic public-LUT comparison substitute is approved.

For floor, let `B_s=2^(9+s)`, `L_s=4096/B_s` and `K_s=2^20*tau_s`.
The identity `I=RNE((2*(X-K_s)-B_s+1)/(2*B_s))` is exact on the declared grid:
both sides of every integer boundary have strict margin `1/(2*B_s)` from a tie.
At `Q1`, build one encrypted index polynomial with four occupied coefficients:

```text
A_s = RNE(q1*q0*L_s/2^40)
Bidx_s = RNE(q1*q0*L_s*(-2*K_s-B_s+1)/2^25)
C_idx = Rescale_q1(C_x * sum_s A_s*Y^s + sum_s Bidx_s*Y^s)
```

Plain operands are residues modulo Q1. `Rescale_q1` is componentwise centered
nearest division by q1 into q0; odd q1 excludes integer half ties. Its error is
not zero. Blind-rotate `C_idx` by encrypted S; **extract constant without trace**,
key-switch 2048→1024, then require an encrypted phase-faithful `Quant4096`:
rotation must equal `RNE(4096*whole_LWE_phase/q0)`. The selected ideal rotation is
the floor above. Require total normalized rounding/extraction/key-switch error
`<1/(2*B_s)`, uniformly `<1/8192`. No decryption computes that phase.

**Concrete missing implementation, not a zero-error assumption:** simply RNE
each LWE component can change the resulting phase by up to
`(1+||s_L||_1)/2` rotation cells (≤512.5 for binary dimension1024), vastly larger
than the interval margin. The same issue applies to selecting S. Neither raw
component switching nor Lattigo's odd-adjusting helper qualifies `Quant4096`.
A phase-correction/compatible encoding attachment, including any extra keys,
operations and noise, is required for **both** index routes. It is not supplied
here. Extra correction work must be charged as conversion, with any algorithmic
deviation returned for approval; it cannot be concealed inside the six lookup
rotations. No real floor of a noisy guess, clamp or public reciprocal LUT.
4096 is the rotation group size, not an adopted torus coefficient modulus;
encrypted accumulators and normalized traces remain over odd q0.

For each segment construct, online and still encrypted,

```text
P_s = Rescale_q1(C_x * sum_i RNE(q1*2^8*aHat_si)*Y^i
                           + sum_i RNE(q1*2^44*bHat_si)*Y^i)
```

This gives nominal output phase `2^44*(aHat_si*x+bHat_si)` at q0; coefficient
encoding/rescale errors remain in the ledger. For `BR(C,J)=Y^(-J)*C`, rotate
each encrypted `P_s` by encrypted I and normalize-trace its constant to `c_s`.
Gather `sum_s Y^s*c_s`, rotate by encrypted S and normalize-trace once more.
All coefficients include encrypted x; none becomes a public affine LUT.
Trace is `2048^-1*sum_{g in Gal(R_S)} sigma_g` modulo each **odd** limb. Reuse
the existing coefficient-layout automorphism/inverse algebra for this exact
proposed doubling schedule, where `sigma_g(Y)=Y^g` modulo `Y^2048+1`:

```text
z = 2048^-1 * C                         # BOTH ciphertext components, FIRST
z = z + KeySwitch_to_s_S(sigma_-1(z))
for k = 0..9:
    g = 5^(2^k) mod 4096
    z = z + KeySwitch_to_s_S(sigma_g(z))
```

Each right-hand side uses the preceding z. These 11 stages enumerate all 2048
odd automorphisms; consecutive exponents `5^1,5^2,...` are not this schedule.
Client owns s_S and creates the named Galois switching-key family
`TraceKeys_S={sigma_g(s_S)->s_S: g=-1,5^(2^k), k=0..9}`; server retains its
evaluation-key backing through every dependent trace. Entry trace needs the
Q1-compatible family; lookup traces need q0-compatible material. Derivation or
restriction between those key owners, auxiliary moduli, decomposition and their
raw/prepared coexistence remain UNKNOWN, not an implicit shared-key credit.
Do not divide fresh key-switch errors by2048 after the sum: modular division need
not be small real division. Derive bounds in the displayed order, including
propagation of each switch error through later additions/automorphisms. Charge
inverse word multiplies, every automorphism/key switch and live scratch inside
trace. This is an algebraic specification, not an adopted cryptographic circuit
or qualified implementation. Existing torus shifts or bare integer division are
not inverse trace.

## 4. Complete key/conversion and proposed level graph

All secrets belong exclusively to client. Server has only evaluation material,
contexts and ciphertexts; no intermediate decryption/re-encryption or numeric
debug export. Client decrypts **only final returned CKKS output**. Fixed public
fixture oracles do not supply clear online activations to evaluator.

| Edge / material | Required relationship; never implicit |
| --- | --- |
| CKKS encryption, DFT/conjugation/relinearization and refresh | Propose independent binary full-ring H and uniform ternary weight192 full-ring E, with both encapsulation switch directions. This assignment is **not** recovered paper key custody/distributions; source reports weight192 with encapsulation, not this complete graph. Error distributions and exact manifests remain unresolved. Do not borrow Poulpy's dense1024/sparse32 choices. |
| `H→S` degree65536→2048 | Decompose plaintext as `sum_{r=0..31} T^r*m_r(Y)`. Secret splitting creates rank32 small-ring relations; client-generated switching material must convert those **complete** relations to proposed independent binary rank-one `s_S∈{0,1}^2048`. Raw coefficient splitting is not a ring switch. Retain or stream all necessary cosets/backings explicitly. |
| Scalar extraction | Constant sample has vector `(a0,-a2047,...,-a1)` under `vec(s_S)`, dimension2048, modulus q0. Nonconstant message terms require the entry shift/trace; extraction alone selects only one phase. |
| `vec(s_S)→s_L` | Client supplies decomposed LWE switching key to proposed independent binary `s_L∈{0,1}^1024`; same q0. Not truncation, zero padding, first1024 coefficients or an unexplained related secret. Used for companion X and selected index, plus internal HomComp outputs when required. |
| Blind rotation and HomComp | RGSW evaluation material under s_S for all 1024 s_L bits; complete gadget/auxiliary/noise and index-output keys. Encrypted accumulator must survive, including its original mask/error. HomComp's additional domains/keys are unknown and must be attached, not assumed covered by BRK. |
| Phase-faithful rotation indices | Both S and I require `Quant4096` under the named index key domain. Any correcting/decomposition keys and intermediate domains remain unspecified; raw 2048→1024 switching does not supply them. Missing attachment rejects the round trip. |
| Trace / return `S→H` | Small-ring Galois keys; embed `Y=T^32`, then switch embedded `s_S(T^32)` to H with client material. Include all automorphism and ring-switch errors and encoded scale44. |
| Recovery | H→E at q0, ModRaise under E, E→H at Q17, then full CtS/EvalMod/relinearization/rotation keys. Raw ModRaise does not remove q0 multiples. Joint analysis must include **cycles** `H→S→L→S→H`, H↔E and every auxiliary modulus; separate per-key security estimates are insufficient. |

**Complete-domain mask lifecycle, unresolved.** Each edge must name its input
and output mask domain (all coefficients and active RNS limbs of the full ring,
or the complete LWE vector), its joint distribution and retained key/randomness
owners. Uniformity is not a per-valid-slot property. A uniform element projects
uniformly only through a proved surjective map onto the stated target domain;
the proof must also cover any joint distribution subsequently required. Coset
splitting of a uniform full-ring mask is a coefficient bijection to the complete
32-coset tuple, not a proof that the resulting rank32 relation has a uniform
rank-one mask after secret switching. Likewise, canonical modulus restriction
has a surjective projection argument, whereas rounded rescaling/modulus
switching needs its own distribution analysis. Embedding `Y=T^32` occupies a
proper subspace and does NOT itself yield a uniform complete full-ring mask.
Neither embedding nor key switching may silently claim to restore uniformity.

Required mask-distribution proofs for H→S, extraction/2048→1024 switching,
HomComp/index correction/BR, S→H and H↔E/recovery remain UNKNOWN. For each
edge requiring restoration, an attachment must name the proposed
rerandomization/sanitization operation, client/evaluator randomness custody and
freshness/retirement rule, source and destination key owners, full-ring/modulus
sampling law, extra operations and error/failure bounds. It must price retained
inputs, new outputs, randomness, evaluation keys and temporary/prepared backing
BEFORE releasing prior owners. No particular restoration algorithm, secret-key
implementation or new executable node is adopted here. Uniform masks are NOT
circuit privacy: returned ciphertext distributions, model-dependent error and
client-visible evaluation material need a separate circuit-privacy argument.
Missing either attachment blocks encrypted execution, including any claim that
the clear coefficient converter or public graph/trace validator supplies it.

Only proposed external edge is **CKKS level4/scale36 → level4/scale36**:

| Stage | Proposed typed transition; not presently a legal capability |
| --- | --- |
| Entry | StC `4→1`, scale36; ring-down, constant alignment and normalized entry trace at Q1. Companion modulus restriction/extraction/KS to q0/level0. |
| Evaluation | Five affine ciphertext–plaintext products/rescales `1→0`; encrypted selection/traces at q0. Result scale44. |
| Return | Reverse ring switch at level0/scale44; H→E; ModRaise `0→17`; E→H; CtS `17→13` depth4; EvalMod `13→5` depth8; one **additional charged** plaintext restoration/rescale `5→4` producing scale36: `Rescale_q5(C_EvalMod * RNE(q5/256))`. Its nominal encoded phase is multiplied by `2^-8` before declaring derived scale36; encoding error is charged, never a metadata-only fix. |

These scales/depth budgets are required interface targets; exact DFT, EvalMod,
rescale and normalization schedules remain missing, so **currently legal edge
set is empty**. Public planner maximum stays13; intermediate recovery level17 is
internal to an atomic hybrid contract (or separately validated internal SSA with
maximum17), not a level17 ordinary-compute option. Different required depth or
entry scale demands a revised proposal, never an implicit extra prime/refresh.
Recovery period is exactly `lambda=q0/2^44`. CtS must expose values
`phi+lambda*K`; require joint wrap bound `|K|≤256`. Ideal EvalMod returns
`lambda*(t-RNE(t))`, `t=K+phi/lambda`, including padding phi=0. The actual
depth-eight approximation, normalization/rescales and joint error/wrap proof
are missing. This functional target does not invent an EvalMod coefficient set.

## 5. Error, work and complete owner accounting

Required error graph is: input encoding → StC → ring-down/alignment/entry trace
→ X restriction/extract/KS → HomComp×3 → S; and retained C_x → index affine/
rescale → BR(S) → extract/KS/MS → I; retained C_x → four encrypted affine
polynomials/rescales → BR(I)×4/trace×4 → gather → BR(S)/trace → ring-up →
ModRaise/CtS/EvalMod/scale restoration → final decoding. Each arrow needs a
specific absolute/phase bound, failure probability, parameter/key identity and
owner overlap; correlated errors cannot be assigned independent probabilities.
Entry must bound the **whole error polynomial**, not only extracted X: affine
polynomial multiplication convolves its coefficients and amplifies their norm.
Include H→E and E→H errors and every `Quant4096` correction edge.

Proposed final maximum absolute error against **1/x** is `2^-18`, not paper RMSE.
Entry requires X-unit error ≤1/64, segment rotation error <1/4 cell and interval
rotation error strictly below the margin above. Reserve ≤`2^-21` each for lookup
accumulator error, return ring-switch error and complete recovery/decoding;
encoding beyond the frozen numeric coefficients gets ≤`2^-24`. Together with
secant/coefficient/input perturbation bounds these fit `2^-18` **conditionally**.
They are proof obligations, not estimates or successful decryptions. At most256
failure events/round trip, each ≤`2^-93`, imply ≤`2^-80` for ≤32 round trips by
union bound, without independence. Neither these tails nor 128-bit security is
proved. Each event must bound the entire vector, including all coefficients,
padding slots and key rows, not silently reuse a per-slot tail. Unknown bounds
stay unknown; no approximate selector repair/fallback.

Lookup count is exactly **3 HomComp / 5 plaintext–ciphertext products / 6 blind
rotations / 5 traces**, excluding HomComp internals and conversion. Index
selection uses extraction, not a sixth lookup trace. Add **five lookup rescales**,
one entry trace, all ring/secret switches, two external extract/KS routes,
modulus rounding, StC and complete recovery including the additional restoration
product. Count internal HomComp PBS/BR/KS/traces separately; do not bill recovery
as ModRaise alone, or equate a round trip with one generic public-LUT PBS.

Reuse `PublicHybridRoundTrip(entry,evaluate,recover)` only after all three costs
are known. Bind it to `PublicCiphertextCapability`/shared SSA, physical states,
immutable parents and owner lifetimes. The validator checks consistency, **not**
cryptographic authority. Unknown cost/error/owner bytes reject; do not put zero
or an invented `error_assumption` identifier into the registry.

Named proposed ceilings for the **next qualification, not execution permission**:

| Limit | Finite scope |
| --- | --- |
| Geometry | One active scalar, standard complex slots32768, degrees65536/2048, LWE2048→1024; four segments ×2048 entries. No toy-security dimension reduction. |
| Key/parameter records | ≤32 named key families, ≤64 RNS limbs, ≤64 gadget digits/row and ≤131072 key rows, with full serialized/prepared backing sizes. These are rejection ceilings, not proposed secure decomposition parameters; missing actual geometry rejects. |
| Public oracle work | ≤32768 cases covering every segment/local boundary and grid neighbors, including rejection of tau_end; exact arithmetic ≤256-bit scalar intermediates; all8192 coefficient pairs, ≤1 MiB table metadata. Exceeding a limit rejects, never truncates. |
| Encrypted work, if later separately admitted | ≤32 fresh single-scalar round trips, one key epoch/thread, no warmups; ≤32 internal PBS per HomComp and ≤256 failure events per round trip. Missing HomComp schedule cannot be set to zero or shortened to fit. |
| Shared records | Per scalar circuit: ≤4096 operations/values, ≤65536 nested SSA records, ≤100000000 validation-work units and 64 MiB metadata. ≤64 live logical ciphertext values; key ciphertext records are separately charged, not hidden by this cap. No unbounded per-case history. |
| Complete owners | ≤8 GiB aggregate ceiling, host reserve≥4 GiB, zero added swap/swapout; ≤4 GiB temporary disk plus ≥8 GiB disk reserve. Future separate build ceiling900 s/4 GiB, setup600 s within8 GiB, encrypted execution600 s, one CPU thread. No build or cap increase authority inferred from D4. |

Owner bound must include client secrets, randomness/encoders, every raw/prepared
key and their coexistence, imported copies, all full-ring/coset/small/LWE
backings, all four affine accumulators, HomComp scratch, DFT/NTT/EvalMod contexts,
recovery's expanded-RNS buffers, scale restoration, transport/serialization,
allocator/runtime arenas and final decoding. Charge new output/scratch/recovery
**before** releasing inputs; shared views retain entire backing. Do not shrink
keys because only one slot is valid. Total/key residency bounds remain unknown;
current 8 GiB admission is **not established**.

## 6. Separate next decision and acceptance

Proposed selectable wording: **“Approve R-Q1 independent bounded public
arithmetic/oracle tests for this named half-open/grid/secant proposal and
attachment review under §5 ceilings. Do not adopt these numerics in a model,
select a backend, generate keys or run cryptography. Return exact HomComp,
prime/key/security/error/recovery and full-owner attachments before requesting
R-Q2 encrypted primitive qualification, including phase-faithful index conversion.”**
Deferring R-Q1 leaves D37 design only.

R-Q1 is only a proposed later decision, not current approval; D38 does not
authorize it. Before it is eligible, freeze a public-only task manifest within
§5's existing case/table/256-bit ceilings: exact case enumeration, finite total
arithmetic-operation and trace-validation work, aggregate live owner bytes,
wall/CPU time limits, and output record/byte limits. These additional limits are
currently UNKNOWN, so no oracle implementation, sweep or calibration run is
authorized. Its scope may cover exact rational grid/boundary/index identities
and the displayed trace algebra only; results confer no encrypted authority.

R-Q2, if separately approved after those missing attachments, must exercise real
encrypted affine accumulators; every equality/neighbor and rejected tau_end in
independent public oracles; both extraction dimensions/key domains; wrong-key,
scale/level/layout/rounding failures; all padding slots after client-only final
decryption; complete operation/owner ledger and immutable retained input. No
status substitution, public LUT, client refresh or threshold weakening.
Even a successful R-Q2 would not qualify source-evaluated constants, universal
range/security, automatic joint scheme/level placement, multi-head sharing,
private KV/output/feedback, full-model quality or complete ROSETTA performance.
