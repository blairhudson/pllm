# Ten first-principles network hypotheses

This round asks whether a **different representation of computation**, rather
than another packet codec, can remove the repeated wide private boundaries.
Five hypotheses target 10× and five target 100× aggregate application-body
reduction per generated token. Those are research targets, not achieved results.
Their older mathematical ingredients are attributed below; novelty here means
new PLLM constructions and tests, not a claim to have invented those ingredients.

The budget reference is the preceding stage-packed, cold 39+8 Qwen2.5 SDK
response: about **229.98 MB** covered setup-inclusive bodies. Its 10× and 100×
budgets are about **23.00 MB** and **2.30 MB**. The JSON binds the exact historical
artifact and totals. New screens use the same pinned W8A8 checkpoint, but are
local algebra/numeric diagnostics, not matched transport or compute measurements.
Historical cold CPU is only a conditional construction budget, not a matched
two-offset compute comparison. In decimal MB per generated output, these body
budgets are **28.748 / 2.875 / 0.2875** for control / 10× / 100×. Bytes per token
are distinct from bandwidth in Mbps or throughput in tokens per second.

## Recorded outcome

The retained [machine-readable evidence](../evidence/network-first-principles-qwen25-2026-10-06.json)
prices all **96 public body stages**, checks a **39+8** trajectory and four fresh
public **prefill plus two same-token decode** cases. It completes with zero
observed new swap. No construction passes whole-response promotion.

| Target | Construction | Concrete result and immediate gate |
| --- | --- | --- |
| 10× | Output coset code | Exact 16× symbol saving on the promised toy; **0/437,184** checked Qwen blocks meet its one-sparse domain. A wrong-domain word can alias a valid codeword. |
| 10× | Exact span cache | **4,403** independent rows among **4,416** checked stage rows. At most **0.124%** arithmetic savings, even before basis solving and **64.22 MB** of lower-rank input/output storage. |
| 10× | Saturation cones | Only **126/5,369,856** SiLU elements equal the identity/signed-zero limit bit-for-bit. Almost every element remains exceptional. |
| 10× | Rational SiLU / inverses | Explicit-tail profile matches **12/12** checked token choices, but **0/12** full logits and **0/12** KV states, with **2.3902** worst absolute logit error. The smaller concrete inversion layout alone projects **128.88 MB** of peer openings. |
| 10× | Packed ramp shares | One gated product per element projects **128.88 MB–1.42 GB** for the checked resharing layouts; ideal public linear work remains **1.03125–1.5×** two offsets. |
| 100× | Permutation telescopes | AND/telescope algebra passes. The specified depth-12 construction already needs **29.36 MB** of ideal branch permutations or **536.87 MB** of two 16-byte branch objects per instruction. |
| 100× | Koopman observable lift | Exact **256-step** shared nonlinear toys with local linear updates. GF(127) needs 20 features; two controlled maps expand the GF(61) case to all 61 dimensions. Qwen feature/encoding costs unknown. |
| 100× | Hermite jets | One worker's value and derivative recover a degree-one masked secret. Restoring one-worker privacy still needs **16,777,217 workers** for 24 quadratic layers under generic no-refresh interpolation. |
| 100× | Zech-log arithmetic | Exact small-field addition/multiplication, but public body sums become **16.20 billion private lookups**, each allowed just **0.000142 bytes** under the 100× target if everything else were free. |
| 100× | Encrypted carry-save | Specified public-weight circuit needs **2.84 trillion ANDs**. Its half-gate body projects **90.93 TB**; an FHE implementation would have **0.0128 aggregate CPU ns/AND** under the historical CPU budget. No FHE backend measured. |

The two most useful follow-up directions are **cheaper protected rational
arithmetic** and **tractable shared-state observable closures**. Current oracles
establish neither a privacy-preserving Qwen implementation nor an all-link win.
The positive rational token result is a small W8A8-to-W8A8 numeric diagnostic,
not upstream-reference agreement or generation-quality validation.

## Five 10× hypotheses

### T1. Output-mask side information as a coset decoder

**Hypothesis.** Stop attempting to compress an apparently random masked response
in isolation. The client already knows its output mask. Send a short linear
syndrome of the response and subtract the mask's syndrome locally. If the true
output belongs to a small, publicly certified domain, recover it without
receiving every coordinate. This is a Slepian–Wolf/error-correcting-code idea
applied to the *decoder's* side information.

For `y = z-s` over a field, `Hy + Hs = Hz`. A two-row matrix whose columns are
`(1,i)` decodes any at-most-one-nonzero vector: its amplitude is the first
syndrome, its position the ratio of the second to the first. A 32-coordinate
word therefore takes two symbols, a **16× component reduction**. The independent
oracle checks every one of its 8,192 nonzero-domain words over GF(257).

**Gate.** The domain promise is essential. `e_0 + e_2` and `2 e_1` have identical
syndromes. Neither the syndrome nor a successful sparse decode certifies which
word was sent. The real-checkpoint screen counts one-sparse 32-value blocks of
the original integer outputs. A data-dependent fallback would need its own
privacy/access-pattern contract, padding, authentication and cost accounting.
The small-field oracle is not the deployed power-of-two ring codec.

This differs from the previous unsuccessful ingress-side-information screen:
Inference's correction does not reveal information about a fresh uniform input
mask, whereas the *client* explicitly holds `s`. It also differs from low-rank
weight factorization: reconstruction uses a domain code rather than client body
weights. Offline corrections, ingress, mask expansion and public delivery still
need full accounting; a 16× response component cannot be called a 16× request.

### T2. Exact span caches instead of identical-token memoization

**Hypothesis.** Cache a few client-owned pairs `(x_i, W x_i)` acquired through
ordinary protected execution. If a new quantized activation is exactly
`x = sum a_i x_i`, reconstruct `W x = sum a_i W x_i` locally. A small invariant
input span could eliminate both new online work and new correction issuance
across many stages, without downloading `W`.

The finite-field oracle reconstructs 32 new rows from three prior pairs. The
checkpoint test instead asks whether the required low-dimensional span exists:
an exact GF(2) minor rank lower-bounds integer and 2-adic independence. It retains
at most 64 public rows and 2,048 columns per stage, records the bound, and prices
client storage for full input/output pairs at that lower rank. The projected
removable arithmetic fraction even grants perfect basis solving and removal of
offline issuance for every potentially dependent row.

**Gate.** A high minor rank rules out a tiny span for this construction. It does
not rule out a different output quotient whose kernel the client could certify.
Basis solving, scales, state identity, full-width domain coverage, private miss
handling, storage lifetime and cold acquisition remain necessary. Variable
miss/hit traffic must not silently acquire a new leakage contract.

### T3. Certified saturation cones and exact exceptional gates

**Hypothesis.** Float32 SiLU has regions where it rounds exactly to the identity
or signed zero. Compile those regions as simple arithmetic, and use expensive
protected nonlinear work only for a small, fixed-capacity exceptional set.
This borrows saturation/abstract-interpretation reasoning rather than fitting a
different activation or assuming small observed residuals are a certificate.

The screen compares original SiLU bits with its signed-zero/ReLU limit over a
public grid and every SiLU element in the checked 39+8 trajectory. It counts
exactly equal elements, not merely small absolute errors.

**Gate.** Exceptional locations are private. A low observed count would still
need a public capacity certificate and protected routing. Even a perfect SiLU
simplification leaves the gated multiplication, quantization, normalization,
attention and token feedback. The screen grants those other operations for free
when asking whether the proposed sparse exception interface is plausible.

### T4. Rational SiLU plus simultaneous inversion

**Hypothesis.** Polynomial fits struggled with source fidelity. Use the classical
continued fraction

```text
tanh(z) = z / (1 + z² / (3 + z² / (5 + ...)))
SiLU(x) = x * (1 + tanh(x/2)) / 2.
```

Continuants turn each bounded approximation into a numerator and denominator
with **one final division**. Montgomery's trick batches `n` nonzero field
denominators using one inverse and `3(n-1)` multiplications. A resident shared
decoder might trade many interactive divisions and client nonlinear cuts for
batched arithmetic.

The oracle checks simultaneous inverses against independent scalar inversion.
The numeric probe fixes `[-32,32]` publicly, rejects out-of-range values, checks
16/24-term candidates on four new public prefill plus same-token decode cases,
and uses distinct execution identities. It compares full logits and active KV
bits as well as selected tokens. There is no fitting to these prompts.
A separate, explicitly identified 24-term candidate uses identity/signed-zero
tails outside that fixed interval. Tail selection would itself need protection;
it is not a fallback that inherits the bounded candidate's numeric identity.
Both strict-range candidates reject on all four new cases. On the 65,537-point
public grid, the 24-term approximation's worst error against the declared
float32 oracle is 0.000001908; a grid is not a global error certificate. The
ordinary 39+8 trace contains three values outside the fixed core range.

**Gate.** A real-valued rational approximation, IEEE division and field inversion
are three different contracts. A small error or matching top token does not
restore the original float32 execution identity. The concrete two-party Beaver
layout also charges both parties' openings for every batch-inverse product.
Montgomery batching is not assumed optimal for MPC: an alternative multiplies
each nonzero denominator by a fresh shared nonzero mask, opens the product and
scales the mask by the public inverse. The independent oracle checks 4,096
inverses and exhausts the nonzero-mask view for three denominators. With ordinary
Beaver multiplication and four-byte words, this alternative charges 16 bytes for
the product's two masked openings and eight for its output opening per element.
Secure nonzero-mask issuance, denominator evaluation, numeric conversion,
verification and offline material are additional costs. A secret zero cannot
silently enter this nonzero-domain contract. One cheap inverse is not a cheap
complete rational layer.
Over 5,369,856 hypothetical denominators, ordinary Beaver evaluation of the
batch-inverse products alone projects 257.75 MB. Fresh-mask inversion halves
that to 128.88 MB in the specified four-byte-field layout, still above the
23.00 MB target before evaluating any numerator or other decoder operation.
These are costed constructions, not a lower bound on all secure inversions.

### T5. Packed ramp-sharing amortization

**Hypothesis.** Use many independent workers to pack `k` private activations in
one polynomial sharing. Public body matrices act locally on shares; amortize
nonlinear degree reduction across the packed values. This changes topology and
trust explicitly: the bounded example tolerates one passive corrupt worker.

For `k` slots and one random mask slot, degree is `k`. A product needs `2k+1`
workers for generic reconstruction. The oracle performs packed multiplication,
actual all-to-all degree reduction, and exhaustive one-worker view checks on
the small field. Its concrete resharing sends `n(n-1)` field words per batch.

**Gate.** Public linear work per packed secret is `n/k = 2 + 1/k`, already above
two offset workers. The peer traffic grows with packing width in this layout.
This is a gate on this simple degree-reduction construction, not a lower bound
on every honest-majority MPC protocol. Local simulated parties do not establish
operator independence or a protected decoder.
The projection grants perfect packing across scalar gates. Real slot alignment,
single-row decode, attention, checkpoint distribution to all workers and
resharing masks can only add work to this specified layout.

## Five 100× hypotheses

### H1. Permutation telescopes

**Hypothesis.** Compile nonlinear computation into constant-width permutation
programs. Blind adjacent instructions with cancelling group elements; evaluate
their product without opening intermediate tensors. Barrington's construction
makes this algebraically possible for bounded-depth Boolean formulas.

The oracle checks a five-cycle commutator implementing AND and exact telescoping
of separately blinded instructions. The specified recursive compiler expands
to `4^d` instructions at depth `d`. At depth 12, even two ideal seven-bit branch
permutations occupy 29.36 MB; two 16-byte objects per instruction occupy
536.87 MB. These omit oblivious branch selection, cryptographic encoding,
consistency checks, model delivery and output decoding.

**Gate.** Width is not program size. These are costs of this recursive compiler,
not a minimum size theorem for Qwen or all branching programs. The group test
does not implement a secure randomized encoding.

### H2. Exact Koopman state lifting

**Hypothesis.** Encode a nonlinear state once into a larger feature vector whose
evolution is linear. Two workers could then update their own additive feature
shares without communication, returning only final output shares. This differs
from merging states that happen to select the same next token: the lift must
intertwine the *entire* transition exactly.

The oracle constructs an exact finite-field Krylov closure of the observable
`x` under `x -> x²+1`. It checks every initial state for 256 transitions in three
fields, with each worker holding only its own simulated share during updates.
The companion update is linear and uses no peer messages. The report charges
initial share width and the public client encoding table separately.
Over GF(127), the lift has 20 coordinates and a 5,080-byte client lookup table;
ingress plus final scalar shares use 84 bytes in the specified two-byte encoding.
The table already contains future-trajectory observables: constructing or
evaluating an analogous Qwen feature map cannot be assumed free. Requiring
closure under two externally controlled maps expands the GF(61) example from
17 to all 61 dimensions. Private operator selection remains unimplemented.

**Gate.** This is the most constructive representation result of the round, but
the executable system is an autonomous scalar toy. A decoder needs tractable
features, cheap private input encoding, all operators, growing KV, token
selection and private feedback. A finite-state lookup expansion always exists;
a *small* useful Qwen lift has not been found or priced. No response-wide ratio
is transferred from the toy to a language model.

### H3. Hermite jets instead of repeated share refresh

**Hypothesis.** Evaluate an entire polynomial decoder under a coded mask and
reconstruct once. Give each worker several derivatives (Hasse jets), hoping to
replace a huge worker count with a short high-multiplicity response.

The oracle reconstructs an eighth-degree composition from Hermite observations.
It also exposes a critical privacy error: for `f(t)=x+rt`, one worker receiving
`f(a)` and `f'(a)` computes `x=f(a)-a f'(a)`.

To hide one worker's `s` jets in this univariate layout, use at least `s` random
mask coefficients. Even granting one quadratic per layer, composition then has
degree `s * 2^L`. Generic interpolation needs more than that many scalar
observations. Dividing into `s` observations per worker still requires
`2^L+1` workers. At 24 layers this is 16,777,217 workers, before introducing real
SiLU, rounding or attention.

**Gate.** Derivatives cannot be treated as free extra private observations.
This is a concrete leakage witness and a cost gate for no-refresh generic
interpolation, not an impossibility result for coded computation.

### H4. Zech-log hidden arithmetic

**Hypothesis.** Store private field values as hidden exponents. Multiplication
then becomes exponent addition; perhaps gated nonlinear products stop requiring
communication. Zech logarithms supply the dual addition rule:

```text
g^a + g^b = g^(a + Z(b-a)),  where g^Z(t) = 1 + g^t.
```

The oracle exhausts all 66,049 input pairs over GF(257), including zero and
cancellation. The checkpoint screen counts the additions in every public dense
projection. In a fully log-domain engine, those previously cheap additions
become private Zech lookups. It reports how few amortized bytes each lookup
could consume if every other part of the response were free.

**Gate.** This representation moves the expensive operation into the dominant
public linear work. A provider must not see a secret exponent or lookup index;
zero tests and conversion between representations also need protection. The
field oracle does not reproduce W8A8 float scales or establish a dual-domain
protocol.

### H5. Carry-save encrypted public linear circuits

**Hypothesis.** Keep the whole decoder encrypted, but avoid a generic multiplier
circuit for every public coefficient. Compile W8A8 projections into public
bit heaps, delay carries with Wallace 3:2 counters, and spend encrypted nonlinear
operations only on carries and actual nonlinear boundaries. Resident ciphertexts
could remove most wide stage exchanges if setup and aggregate compute fit.

The independent bit-level oracle handles signed extremes and exact 8/16/24/32-bit
wraparound. The checkpoint compiler counts a specified unsigned-biased i8
bitheap for all 96 stages and the actual 46 executed rows. It prices a simple
two-AND full adder and an unsimplified final ripple adder. The report gives both
a two-ciphertext half-gate material projection and a conditional CPU allowance
per AND for an encrypted implementation.

**Gate.** This is not a minimum circuit-size bound: constant folding, common
subexpressions, Booth recoding and better synthesis may help. No FHE backend is
run, so its keys, ciphertext distribution and CPU remain unknown. Actual float
nonlinear work and token feedback are absent from this already-large circuit.
Classical clear-circuit size alone cannot authorize a private runtime.

## Reproduction and evidence

```bash
uv run --no-sync pytest -q tests/test_network_first_principles.py
uv run --no-sync python scripts/probe_network_first_principles.py \
  --output docs/evidence/network-first-principles-qwen25-2026-10-06.json
```

`--algebra-only` skips checkpoint loading. The full run uses the public pinned
Qwen2.5-0.5B checkpoint already used by earlier screens, checks physical memory
headroom, stops on pressure, and records observed swap growth. Its process RSS
includes the full local research engine and is not a client-memory measurement.
Source/body/plan bindings, source-file digests, historical evidence hashes,
domain failures and unknown costs remain explicit. It archives counts and
digests, not prompt text, token IDs, logits, activations or secret material.

All ten constructions remain non-selectable research. A promotion requires a
complete numeric and privacy contract, full-response accounting, client CPU/RAM
admission, one-use state where applicable, and the ordinary compiled execution
path. No general 10× or 100× result follows from this screen.

## Older ideas used as specifications

- Slepian and Wolf (1973), [Noiseless coding of correlated information
  sources](https://doi.org/10.1109/TIT.1973.1055037).
- [NIST DLMF §4.39.1](https://dlmf.nist.gov/4.39#E1), the classical tanh
  continued fraction; Montgomery (1987), [Speeding the Pollard and elliptic
  curve methods of factorization](https://doi.org/10.1090/S0025-5718-1987-0866113-7),
  simultaneous inversion.
- Shamir (1979), [How to share a secret](https://doi.org/10.1145/359168.359176),
  polynomial privacy and interpolation.
- Barrington (1989), [Bounded-width polynomial-size branching programs recognize
  exactly those languages in NC1](https://doi.org/10.1016/0022-0000(89)90037-8).
- Korda and Mezić, [Linear predictors for nonlinear dynamical systems: Koopman
  operator meets model predictive control](https://arxiv.org/abs/1611.03537),
  an accessible modern account of the older lifting idea. Our finite-field
  oracle is exact and independently constructed; it does not use their learned
  approximation or code.
- Yu et al., [Lagrange Coded Computing](https://arxiv.org/abs/1806.00939),
  polynomial computation under coded privacy.
- [Zech logarithms](https://en.wikipedia.org/wiki/Zech%27s_logarithm), classical
  finite-field addition in exponent coordinates.
- Wallace (1964), [A Suggestion for a Fast
  Multiplier](https://doi.org/10.1109/PGEC.1964.263830); Chillotti et al. (2016),
  [Faster Fully Homomorphic Encryption](https://eprint.iacr.org/2016/870),
  encrypted execution/bootstrapping context, not a measured backend here.

These references provide mathematical context. No upstream implementation was
built, linked, vendored or substituted for PLLM execution.
