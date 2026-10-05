# Threshold / collective CKKS refresh: adjacent private-transformer review

Date: 2026-10-01. Status: **real protocol capability, no admitted PLLM backend or
tenfold decoder**. This parallel track owns this report and the two new bounded
oracle files listed in §10. Existing runtime and root research plan are preserved.

## 1. Decision and technical correction

Collective refresh is a genuine alternative to non-interactive FHE bootstrapping:
key-share holders jointly extend the ciphertext modulus without revealing the
activation. Lattigo and OpenFHE implement concrete local protocol operations.
Neither makes protected attention, RMSNorm, SiLU, quantization, matrix packing or
resident KV free. Most dense computation can remain at an evaluating worker;
the second worker performs refresh/key-share operations rather than repeating
the entire transformer. No training or TEE is required by this placement.

**Correction to any proposal that says “two workers refresh every four products
using the measured contexts”:** measured *exhaustion depth* is not *usable depth
between secure refreshes*. Mask/correctness headroom must remain before
exhaustion. At scale `2^40`, Lattigo's nominal 128-bit helper needs ~169 live
modulus bits for two parties. The degree-8,192 sample has only 140 active-Q bits;
the degree-16,384 sample has 220 and must retain its first four primes (~180
bits). Only **one** 40-bit level can be spent before that masked refresh, not
the four products measured when exhausting the chain. Stronger range/session
guards can consume the last usable level too.

OpenFHE's **two-party distributed-rounding** method is materially different:
one extra RNS limb for FIXED modes, two for FLEXIBLE modes, rather than the
general multiparty mask reserve. It deserves its own gate. It still sends an
input polynomial and a *full-level ciphertext* back for every refreshed object.
Replacing that reply by a small seed or one scalar has no support in the
inspected implementation. Table §6 already vetoes these sampled layouts on
known worker bodies, even ignoring complete tensor packing and client bodies.

Also distinguish CKKS from BFV: Mouchet et al.'s original Protocol 5 is a BFV
decode/re-encode bootstrap. The CKKS variants here restore modulus/scale while
retaining prior approximation error and adding fresh error. They do **not**
recover the exact pretrained W8A8 result or erase accumulated numerical error.

## 2. Locked baseline and evidence classes

Read before this review:

- `docs/research/network-io-10x.md`, especially “Token-boundary encrypted execution”,
  “Gate 3 result”, and the batching/seeded-serialization screens.
- `python/pllm/runtime/he_layer_feasibility.py`, compiler-dependent four-product
  per-layer path and explicitly unpriced complete operators.
- `docs/evidence/he-token-boundary-feasibility-qwen25-2026-09-30.json` (SHA-256
  `0864e83e58a78dc54d966f46d361bcd3f605a49418781d7a8bd7b7966284f944`).
- `docs/evidence/ckks-boundary-cost-qwen25-2026-09-29.json`, the earlier per-gate
  screen; `compiler-region-contract-qwen25-2026-09-30.json` for matched budgets
  and tracked body MACs; `resident-quadratic-layer-gate-2026-09-28.json` for
  model-shaped intermediate dimensions.

Pinned model: `Qwen/Qwen2.5-0.5B-Instruct` revision
`7ae557604adf67be50417f59c2c2f167def9a775`, fixed weights, W8A8 prepared control,
body fingerprint `5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
Pinned config [S12] confirms 24 layers, hidden 896, intermediate 4,864, 14 query
heads, 2 KV heads, head dimension 64, KV width 128, vocabulary 151,936.

| Cohort | Covered all-link control | Covered online control | 10× all-link budget | 10× online budget | Executed rows / forward invocations |
|---|---:|---:|---:|---:|---:|
| 39+8 | 116,843,966 | 73,831,744 | 11,684,396 | 7,383,174 | 46 / 8 |
| 39+32 | 178,970,558 | 113,545,024 | 17,897,055 | 11,354,502 | 70 / 32 |

Executed rows are 39 prefill rows plus `G-1` feedback rows; first output comes
from prefill. Each of the `G` forwards traverses all 24 layers. The existing
dependency gate gives **at least 96 serial ciphertext products per forward**
for its direct polynomial CKKS construction, *before* norm/softmax/public
projection rescaling. This is not a universal lower bound on every HE circuit.

Evidence categories remain distinct:

1. **Measured locally earlier:** TenSEAL 0.3.17 ciphertext/context bytes and
   synthetic depth 2/4; not a full layer.
2. **Inspected upstream oracle:** versioned source/specification capability;
   never imported, built, installed or run here.
3. **New bounded arithmetic:** scalar rounding, nominal bit-headroom and
   directed raw-RNS accounting; not secure refresh or checkpoint fidelity.
4. **Unknown:** real Lattigo/OpenFHE context generation, serialized refresh
   messages, complete CKKS layer, latency, aggregate memory and full wire.

## 3. Privacy contract, 2-of-2 DKG, and ownership

### Worker-owned collective key: concrete candidate

Let online workers A and B hold independently sampled secret shares `s_A` and
`s_B`; ideal secret key is `s=s_A+s_B`. A also evaluates dense operations and
coordinates refresh. B is an independently operated key-share/refresh worker.
Client C receives only the collective public key and encrypts embeddings under
it. A keeps activations and KV encrypted across the whole body. No provider
receives the prompt, embeddings, intermediate plaintext, masks of its peer, or
the combined secret key. Public checkpoint weights are available to A.

**2-of-2 DKG:** agree on authenticated party identities, CKKS parameters,
distribution, public CRS and session/key identifiers. Each worker generates
its share locally. They jointly generate the public key using common uniform
polynomial `a` and public encryption-key contributions; no trusted dealer
ever samples or reconstructs `s`. Generate collective relinearization and
specified rotation keys separately. At 2-of-2, Lattigo's optional Shamir
thresholdization for `t<N` is unnecessary [S3]. Key validation, distribution
and all transmissions still require accounting. Sum-of-share secret/noise
distribution must be included in the security estimator.

Security scope supported by the cited constructions: **passive/semi-honest,
at most one corrupt worker**, RLWE and the specified evaluation-key/security
assumptions, with authenticated transport and honest protocol execution.
A and B must not collude, including exchanging stored key shares later.
Both shares compromise all ciphertexts retained under that key. One honest
worker must also keep fresh private masks/private randomness secret. An
external coordinator may collude with one worker without obtaining the other
share under the supported cloud-assisted passive model. This does not provide
malicious correctness, robustness, adaptive/mobile corruption security or
availability if either worker disappears. Lattigo explicitly lacks the share
proofs needed for its suggested active-security extension [S3].

This is a **different contract from client-key-only HE**: the client alone
cannot decrypt collective-key activations; both online workers collectively
can. Reusing a worker-pair key across clients is possible in principle but
changes isolation and compromise scope. It requires per-client output
authorization and noise/query accounting; it is not a free per-model key.

### Output boundary: only the trusted client learns the result

Return final **pre-final-RMSNorm hidden state** only. C performs final RMSNorm,
public output head and token selection, then encrypts the next embedding.
There are two concrete output placements:

- **Direct threshold decryption to C:** A sends its lead partial decryption,
  B sends its main partial decryption over confidential authenticated links
  to C; C alone fuses. A must never collect both plaintext decryption shares.
  With A holding the ciphertext, charge `A→B: c1`, `A→C: one polynomial`,
  `B→C: one polynomial`, i.e. three polynomial bodies per returned ciphertext,
  plus control. Flooding/remaining-level requirements still apply. No secret
  key is sent to C or assembled at a provider.
- **Collective public-key switch to C's independent key:** give both workers
  `pk_C`; generate switches on the authorized final ciphertext and aggregate
  publicly. Lattigo `PublicKeySwitchProtocol` implements this external-receiver
  boundary [S7]. Charge `A→B: c1`, `B→A: two-polynomial switch share`,
  `A→C: two-polynomial switched ciphertext` = five polynomial bodies, plus
  key delivery and controls. A's own switch share is local. Ordinary
  secret-to-secret `KeySwitchProtocol` does not supply this boundary when
  workers know only `pk_C`.

The old sampled download bytes assumed a ciphertext directly decryptable by
C. They **undercharge** either worker-collective output placement. Public-key
switching only at final output is important: a reusable, unrestricted
collective→client switch key changes which intermediate states that receiver
can decrypt. Output authorization and fresh noise must bind to the circuit,
cohort, tensor, client key, step and request. No provider plaintext fallback.

### Client share versus client-only key

Using C and A as the two key holders makes C participate in every refresh
with its private share. It need not reconstruct intermediate activations, but
refresh work and messages now cross C's link and C is needed throughout
execution. It is not the two-online-worker placement or client-key-only HE.
With *only* `sk_C` and no secret share at any worker, the inspected collective
protocols cannot execute worker-only refresh. Splitting an existing client key
or importing ciphertexts into a collective key needs its own secure setup,
conversion, distribution and communication gate.

## 4. Concrete protocols and backend support

### Mouchet et al. and Lattigo masked refresh

Mouchet et al. [S1] §IV-H, Protocol 5 combines encryption-to-sharing and
sharing-to-encryption in a single contribution round. The paper instantiates
BFV, including a decode/round step. Its §IV-E and Appendix A address smudging;
these cannot be transplanted as CKKS precision guarantees without analysis.
Lattigo v6.1.1 implements the CKKS version directly [S4–S6]:

1. Each party computes a masked decryption contribution at the input level
   and re-encryption contribution at the output level. Private mask remains
   party-local. A common random output polynomial comes from public CRS.
2. Aggregate both kinds of contributions. Coordinator sees only the masked
   centered coefficient polynomial, not the activation.
3. Apply identity recoding/scale change, cancel masks in encrypted form and
   output a degree-one ciphertext at the restored level/default scale.

`RefreshShare` contains **two polynomials**, one at each level, and metadata.
`GenShare`, `AggregateShares`, `Finalize` and serialization exist. Upstream
provides **local operations, not networking** [S3]. `MaskedLinearTransformation`
can fuse a public linear operation/packing transform with refresh. It cannot
apply SiLU/softmax/RMSNorm in the clear: nonlinear `f(x−ΣM_i)+Σf(M_i)` does not
equal `f(x)`. A user-defined callback does not relax that algebraic restriction.

The source helper computes

```text
logBound = lambda + ceil(log2(input_scale))
required log2(Q_input) >= ceil(logBound + log2(number_of_parties))
```

and returns unsupported when no prefix of Q reaches that bound [S5]. This is
a necessary code-level parameter gate, **not a complete privacy proof for
unbounded Qwen activations**. Source comments claiming 128-bit security do not
certify any scale/range, flooding choice, number of decryptions or transcript.

### OpenFHE: two distinct interactive methods

OpenFHE v1.4.2 [S8] identifies:

- **`IntMPBoot*`, general n-party method:** optimized masked protocol from
  ePrint 2023/1203 Appendix E; one low-Q masked-decryption polynomial and one
  full-Q re-encryption polynomial per key holder, plus common random full-Q
  polynomial. `IntMPBootAdjustScale`, `RandomElementGen`, `Decrypt`, `Add`,
  `Encrypt` are implemented [S9]. SLACK/COMPACT are different mask reserves,
  not interchangeable security labels. Appendix E and actual mask products
  must determine admissibility; example prose alone is insufficient.
- **`IntBoot*`, specialized two-party method:** Appendix D's distributed
  rounding; `IntBootDecrypt`, `IntBootEncrypt`, `IntBootAdd` are implemented in
  `rns-multiparty.cpp` [S10]. A sends only input `c1`; B partially decrypts,
  rounds its share, extends basis, **encrypts** that share under the joint
  public key and sends a full-Q ciphertext. A extends/rounds its local share
  and adds it. No additive mask/decryption share is sent in plaintext.

The two-party example uses a genuine sequential public-key generation API
(`KeyGen`, then `MultipartyKeyGen(publicKey)`), separate eval-key generation,
and 1/2 reserved extra levels for FIXED/FLEXIBLE [S10]. The **n-party demo**
collects all private keys in one vector for centralized demonstration setup
[S9]. That convenience overload is **not a distributed DKG deployment**.

Appendix D's formal Protocol 2 rerandomizes input with `Enc(0)` and derives
uniform `rho=H(pk,a)`. The inspected optimized C++ helper does neither itself;
it implements rounding, and the example calls it on an existing ciphertext.
Appendix D's “Practical considerations” explains the computational-uniformity
optimization and a server-local additive-output variant, contingent on later
secure flooding and intermediate outputs not being exposed. A future gate
must choose that model explicitly or implement the full protocol wrapper.
Do not assert the stand-alone random-oracle correctness theorem follows
automatically from calling these helpers on arbitrary evaluated ciphertexts.

### Installed capability check

Only introspection was necessary. Local `.venv` contains TenSEAL **0.3.17**;
`dir(Context)` and `dir(CKKSVector)` contain no method with `bootstrap`,
`refresh` or `multiparty`. `importlib.util.find_spec('openfhe')` is false.
No Lattigo runtime was imported or built. Therefore collective refresh is
**unsupported in the tested local backend**, while upstream C++/Go capability
is concrete. No new context/evaluation keys were generated in this review.

## 5. Correctness, privacy, noise and extra precision

### Mask magnitude is in encoded coefficients, not activation bit width

For a centered uniform integer mask of interval length P, shifted by a
coefficient difference `d`, exact total variation is `min(1, |d|/P)`. A bounded
scalar probe verifies this expression, not transcript security. If each
coefficient has magnitude at most B, two admissible messages can differ by
`2B`; a simple sufficient product/session union bound for N coefficients and
R exposed masked transcripts is `2*N*B*R/P`. This conservative guard is not
claimed to be Lattigo's tight theorem or a necessary bound for every protocol.

Thus masks need real bounds on **encoded coefficient magnitude, existing
noise, transform norm and scale**, not “W8A8 therefore eight mask bits”. At
scale `2^40`, API nominal mask width is 168 bits, assuming normalized scale
adequately describes magnitude. A slot range of 128 rather than 1 can require
another seven range bits in a sufficient analysis. A full-response statistical
target may also require dimension/query slack. For the degree-16,384 hidden-size
proxy (§6), B=`2^40`, R=3,420, the simple sufficient guard requires a **195-bit
mask**, then aggregate/correctness room beyond it. Only the full 220-bit Q
prefix survives; no 40-bit compute level remains. This is a conditional
conservative veto, not a new universal CKKS lower bound.

Appendix E [S2], Theorem 3 gives concrete general-protocol correctness condition
`q >= k*|m|*2^lambda + 2*|m|`, with the existing encoded-message/error magnitude
accounted for. Its 64-bit RNS discussion needs mask limbs plus a decryption
limb: up to three extra limbs for ~50-bit plaintexts at lambda=128. Lattigo's
integer masks and OpenFHE's mask-modulus basis representation differ; do not
mix one backend's limb reserve with the other's share sizes.

### Two-party rounding correctness is probabilistic and response-wide

Appendix D [S2], Theorem 1 gives failure probability at most `2*N*beta/q`
per ciphertext for encoded coefficient bound beta and uniform random-oracle
shift. For R refreshes, a sufficient response target `<=2^-kappa` is

```text
q >= 2*N*beta*R*2^kappa
```

At beta=`2^40` (an **assumption, not certified Qwen range**), N=16,384 and
R=1,116, a response target `2^-50` needs at least **116 nominal q bits**; one
refresh needs only 105. The transferred 60+40-bit input product supplies
approximately 100 bits, giving nominal response bound **3.1719e-11** (~`2^-34.88`),
not `2^-50`. Actual primes are below powers of two, so exact prime arithmetic
must be used for admission. Increasing first/scale primes or adding a reserve
limb costs ciphertext/key size and usable depth. This correctness target and
128-bit computational RLWE security are different parameters.

The scalar oracle exhausts 10,914 share pairs at q=64/256/65/257, messages
[-8,8], reproducing the source interval rule. For tiny even q, endpoint
conventions require guard `2*(|m|+1)/q`, not an exact `2*|m|/q` count. For odd
q, using `floor(q/2)` adds coefficient error -1 or 0 in otherwise successful
lifts; large lift errors still occur near thresholds. Real RNS products are
odd. These synthetic facts call for a rounding/encoding error budget, **not**
an assertion of a backend security vulnerability. This reference generates
no key, ciphertext, secure mask or actual refresh.

### Flooding and accumulated CKKS error

Lattigo accepts a caller-selected discrete-Gaussian flooding distribution;
`KeySwitchProtocol` combines its variance with fresh encryption noise [S6].
The refresh helper does not select that distribution or derive a complete
transformer error bound. Mouchet §IV-E smudging depends on ciphertext-noise
variance. OpenFHE's noise-flooding document [S11] specifies

```text
sigma_flood = sqrt(12*tau) * 2^(s_stat/2) * t_error
```

where tau counts adversarial output queries, `s_stat` is statistical security,
and `t_error` is the computation error estimate. With tau=32, this alone
multiplies error scale by roughly `2^19.29` at `s_stat=30`, or `2^68.29` at
`s_stat=128`. Those factors are not Qwen parameters and must not be confused
with 128-bit RLWE security. They illustrate additional precision/scale cost
before Gaussian tails and encoding amplification. OpenFHE's empirical
imaginary-slot noise estimate is not a worst-case bound on every private
input. Its multiparty decryption has separate flooding modes; one mode
requires at least three remaining towers [S10]. Hence the smallest measured
last-level TenSEAL output is especially optimistic for this boundary.

Require a public admitted input domain, coefficient/noise bounds and chosen
per-response/key-lifetime leakage/query target. Budget polynomial approximation,
scale change, division truncation, odd-modulus rounding, rotations/key-switch
noise, norm/softmax sensitivity and final flooding together. A refresh that
merely restores Q carries `m+existing_error` forward and adds new error.
It is neither exact W8A8 quantization nor arbitrary-precision repair.

## 6. Directed messages, depth and measured-context cost gates

### Roles and link ledger

Use A as evaluator, coordinator and one key holder; B as the other key holder.
This avoids charging a fictitious third server yet counts every actual body.

| Phase | Messages and participants | Round/flight accounting |
|---|---|---|
| Parameters/CRS | Agree A↔B; deliver public context/collective pk to C; deliver `pk_C` to both if public-key switching | Setup; public CRS may use agreed seed and unique invocation IDs |
| Lattigo public DKG | Local contribution at A; B→A one contribution; A→B A's contribution if B needs pk; A→C collective pk | One contribution round after CRS; 3 bodies in this explicit placement |
| Relinearization key | B→A round-1 share, A→B aggregate, B→A round-2 share; A retains final key | Two contribution rounds; 3 bodies, separately sized gadget shares |
| Each Galois key | B→A contribution; A combines locally and retains final key | One contribution round per chosen automorphism; may batch |
| Input/feedback | C→A encrypted packed embeddings | `ceil(39*896/S)` initial objects + `G-1` feedback objects |
| Masked refresh | A→B input c1+metadata+CRP identifier; B→A low-Q and full-Q share pair; A's pair/aggregation local | One contribution round once input/CRP known; operationally **2 sequential one-way flights** |
| OpenFHE two-party refresh | A→B adjusted c1; B→A full-Q encrypted rounded share | **2 sequential one-way flights**; A alone retains refreshed output |
| Token output | Direct partials or public-key switch from §3; C performs final norm/head/choice | G authorized outputs; peer/output/flooding costs beyond old boundary estimate |

If a keyless external coordinator owns/evaluates the ciphertext, masked
refresh instead sends c1 to **both** A and B, receives **both** share pairs,
and retains the output: four directed bodies, twice the two-worker payload
below, plus coordinator/CRS traffic. Sending refreshed output back to both
workers adds two ciphertext bodies if replication is required. If a client
relays A↔B messages, every hop counts; it does not turn worker bytes into zero.
OpenFHE's specialized 2-party protocol needs a designated key-holder/evaluator;
a keyless aggregator cannot perform A's local rounding unmodified.

Fresh private masks/randomness are generated per ciphertext refresh. Public
CRP seeds may be agreed once and indexed; that does not make private masks
public or safely reusable. Bind invocation metadata and burn fresh state on
cancellation/replay. Setup/evaluation keys are reusable public material,
not per-refresh one-use correlations. Count their generation, serialization,
delivery and memory separately; do not charge them as zero or again for every
refresh. Existing single-client TenSEAL eval contexts of ~35.29/179.50 MB are
comparators, not measured collective setup keys or universal key-size floors.

### How the concrete body floors are derived

For **this uint64 uncompressed RNS representation**, one polynomial with N
coefficients and l live limbs contains `8*N*l` coefficient bytes. Lattigo
`ring.Poly`/`structs.Matrix` serialization actually stores those uint64 arrays
[S6]; metadata/header bytes add cost. OpenFHE source confirms polynomial
counts and live/full bases [S9–S10], but its serialized wire format is not
measured here. These are representation-specific coefficient-payload floors,
not claims that no compression/custom transport/other HE parameters can beat
them. In particular, the measured smallest TenSEAL ciphertext is **not** a
universal lower bound and is not used as a fabricated refresh share.

For A+B with A evaluating/coordinating:

```text
masked:   A→B = 8*N*l_in; B→A = 8*N*(l_in+l_out)
rounding: A→B = 8*N*2;    B→A = 8*N*2*l_out
```

Transfer the *already measured* chain shapes, not their API implementations:
SEAL's last 60-bit special key-switch prime is excluded from active Q. Hence
active prefixes are `[60,40,40]` and `[60,40,40,40,40]`. This transfer is an
arithmetic comparison, **not evidence those contexts are portable or that
either backend accepts them with collective security**.

| Degree / transferred protocol | Usable product levels before refresh | Payload per refreshed ciphertext A→B / B→A | Sum |
|---|---:|---:|---:|
| 8,192, Lattigo masked | **Unsupported**, Q=140 <169 | — | — |
| 16,384, Lattigo masked, nominal helper only | 1 | 524,288 / 1,179,648 | 1,703,936 |
| 8,192, OpenFHE-style rounding FIXED | 1 | 131,072 / 393,216 | 524,288 |
| 8,192, rounding FLEXIBLE | 0, no compute span | — | — |
| 16,384, rounding FIXED | 3 | 262,144 / 1,310,720 | 1,572,864 |
| 16,384, rounding FLEXIBLE | 2 | 262,144 / 1,310,720 | 1,572,864 |

These rounding rows additionally need §5's correctness/rerandomization and
output-flooding gates; favourable payload does not admit their parameters.
FLEXIBLE starts with three towers before adjustment drops to two; the request
can be sent after A's adjustment, so two input towers are charged.

If k products fit between refreshes, optimistic refresh epochs per forward
are `ceil(96/k)-1 = floor(95/k)` (no refresh after final product). Thus 1/2/3/4
products imply **95/47/31/23** epochs. Norms, softmax and projections can only
add to this direct circuit. At least one ciphertext on a serial dependency
path must be refreshed at every epoch. Even granting **only that one object**
per forward gives the following conditional known-body floor:

| Transferred protocol | Epochs per forward | Calls at +8 / +32 | Worker raw bodies at +8 / +32 |
|---|---:|---:|---:|
| N=16,384 masked, k=1 | 95 | 760 / 3,040 | 1,294,991,360 / 5,179,965,440 |
| N=8,192 rounding FIXED, k=1 | 95 | 760 / 3,040 | 398,458,880 / 1,593,835,520 |
| N=16,384 rounding FIXED, k=3 | 31 | 248 / 992 | **390,070,272 / 1,560,281,088** |
| N=16,384 rounding FLEXIBLE, k=2 | 47 | 376 / 1,504 | 591,396,864 / 2,365,587,456 |

Every listed worker-only floor exceeds **both** matched budgets. It excludes
client embeddings/outputs, setup, framing and wider tensors. Minimum forward
sequential worker flights at +32 are respectively 6,080 / 6,080 / 1,984 / 3,008
(two flights per epoch). If RTT were 1 ms, even the most favourable row has
~0.992 seconds of serial refresh RTT before HE compute; at 10 ms, ~9.92 s.
These are illustrative latency components, not measured end-to-end times.

For intuition about actual rows, hidden-state-only packing gives initial
9/5 ciphertexts at S=4,096/8,192, then one per feedback row: 40/36 live chunks
at +32, 16/12 at +8. Applying the same epochs to these chunks gives
**proxies, not proven schedules or floors**: masked N=16,384 +32 costs
5,827,461,120 worker coefficient bytes (3,420 calls); fixed rounding costs
1,755,316,224 (1,116 calls). Refresh within attention or the 4,864-wide MLP
cannot be assumed to operate on a compact 896-wide hidden vector. Frontier
packing must be compiled to replace these proxies.

Existing measured ideal client-boundary bytes, still undercharging collective
output, are 6,347,992 / 17,444,584 at N=8,192 and
14,736,868 / 46,308,940 at N=16,384 (+8 / +32). The larger sample already fails
either cohort before refresh. The smaller +32 sample also fails; +8 leaves
only **1,035,182** online bytes for refresh and all missing output/control work.

## 7. Encrypted resident KV and complete-layer requirements

KV must remain encrypted under the collective key for all 24 layers and all
feedback steps. At +32 it holds `2*24*70*128 = 430,080` scalar K/V values;
at +8, 282,624. RoPE applies to K; V remains encrypted. Old KV can be
mod-switched for a query, but repeated switching/use cannot restore its
level, and its origin already contains earlier-layer error. Track ciphertext
level/scale/error age per KV block and charge refresh or retain suitable
copies before new Q·K and probabilities·V products. No client/provider
reconstruction of the cache is allowed to conceal that cost.

The bounded oracle compares raw full-level storage in the transferred bases:

| Cohort / degree | Ideal combined K+V chunks per layer | Total ideal raw KV bytes | Naive one-ciphertext per K or V row |
|---|---:|---:|---:|
| +8 / 8,192 | 3 | 28,311,552 | 868,220,928 |
| +32 / 8,192 | 5 | 47,185,920 | **1,321,205,760** |
| +8 / 16,384 | 2 | 62,914,560 | **2,894,069,760** |
| +32 / 16,384 | 3 | 94,371,840 | **4,404,019,200** |

Ideal combined packing is storage arithmetic, not an attention layout. It
needs bounded rotations, masks and append/read scheduling; retaining
duplicated levels/layouts adds memory. Naive per-row layouts exceed the 1 GiB
cap in three rows before keys/workspace. Ideal KV storage leaves potential
room, but no total-memory admission follows. Sequentially refresh blocks to
avoid materializing all masks/replies simultaneously.

A complete region must specify and price **all** of the following:

- Two RMSNorms per layer: protected square/reduction, inverse square root,
  multiply, epsilon and scale/domain management.
- Seven public dense projections, bias, quantization/scales, GQA and head-safe
  rotations, RoPE, both encrypted K/V appends and cache lifetime.
- Q·K, scale, public causal mask, stable private softmax exponent/normalization,
  probabilities·V and attention residual; no plaintext normalizer.
- SiLU approximation and its range, gate×up, down projection and residual.
- Every ciphertext/constant product, rescale, level alignment, error-sensitive
  edge and packing conversion; no unpriced HE↔MPC boundary.
- Refresh frontier: every live branch/object, message, coefficient bound,
  scale, restored Q and fresh randomness, including resident KV refreshes.
- Authorized output-only partial decrypt/key-switch, flooding and client final
  norm/head/token feedback; final-only output never exposes body activations.

For example, prefill scores have `14*39*39=21,294` values per layer (6/3
ideal ciphertexts); SiLU gate has `39*4,864=189,696` (47/24). Decode SiLU
alone occupies 2/1 objects at those slot capacities. Head-preserving layouts
can need more. Public linear “zero ciphertext depth” in the existing gate
is deliberately optimistic, not zero rotations, time, levels or evaluation
keys. Existing +32 tracked body projection work is ~25.05 billion integer
MACs; translate the actual fixed weights into packed HE operations and charge
CPU rather than equating this with encrypted throughput.

## 8. Could alternative parameters fit?

**Ordinary interactive CKKS with these layouts: no plausible fit yet.**
The margins are not a small serialization tweak. Once-per-layer hidden refresh
would already involve 920/828 objects at +32 (23 boundaries ×40/36 chunks).
At the hypothetical 128,000-byte *average client ciphertext* from the old
discussion, the smaller-slot +32 boundary is 9,216,000 bytes, leaving
2,138,502: only **2,324 bytes per refresh**, before output conversion/KV/control.
At +8 the allowance is **11,715 bytes** over 368 such objects. No inspected
CKKS share format approaches those whole-exchange sizes. Increasing N for
slot utilization does not pack future autoregressive feedback steps together.

Larger compute spans decrease rounds but enlarge full-Q replies and keys.
For the uncompressed specialized fixed-rounding representation, assume
`l_out=k+2`, input two limbs, degree fixed at 8,192 and only one live
ciphertext per forward. Worker bodies are

```text
16*N*G*floor(95/k)*(k+3),  1 <= k <= 95.
```

An exact bounded sweep finds its minimum at **k=48**: 53,477,376 bytes for
+8 and 213,909,504 for +32, before client bodies. This exceptionally generous
chain needs far more modulus than the sampled 128-bit-security degree can
support; an actual security-preserving larger N worsens these raw bodies.
It demonstrates why “longer chain fixes refresh bandwidth” is not enough in
this representation, not a universal lower bound on compressed/other HE.
Taking k≥96 removes these refreshes but transfers depth/security/size cost
to the very long initial chain, with norms/softmax still absent.

Potential reopening requirements are therefore concrete and simultaneous:

1. A specified low-level serialization/compression or fundamentally different
   refresh/execution placement that eliminates most full-Q peer replies, with
   privacy proof and **both-link** byte measurements. Seeded encryption may
   reduce one encryption component, but arbitrary evaluated polynomials and
   masked-decryption shares are not automatically seed-compressible. Halving
   these GB-scale examples cannot close the gap.
2. A smaller-scale/modulus circuit with certified ranges and held-out
   pretrained-model fidelity. Reducing scale also reduces precision; mask
   privacy and rounding-failure bounds still require extra headroom. Fixed
   W8A8 baseline cannot silently become a trained polynomial model.
3. Few refreshes for the whole private region, with real packing/KV/key memory
   under 1 GiB and affordable dense HE compute; report first-response DKG and
   key delivery separately from warmed requests. All-link slack beyond online
   is only 4,301,222 / 6,542,553 bytes (+8 / +32) if online target is saturated.
4. A non-interactive client-key FHE backend could remove worker refresh bodies,
   but that is a separate bootstrap/compute/key gate, not evidence collective
   refresh meets this budget. Same-client batching can amortize traffic across
   concurrent independent requests; it changes the workload denominator and
   does not admit this single-sequence control.

No alternate parameters have demonstrated online budget **and** complete-layer
fidelity **and** compute feasibility. Efficient interactive refresh in the
literature's small statistical circuits does not establish transformer cost.

## 9. Primary sources fetched and inspected as oracle

All accessed 2026-10-01; upstream code version-pinned. No upstream code becomes
a runtime/build dependency. PDF text was extracted with installed `pdftotext`
into the approved temporary directory solely to inspect primary specifications.

| ID | Primary source / exact section or symbols used |
|---|---|
| S1 | [Mouchet et al., ePrint 2020/304](https://eprint.iacr.org/2020/304), PDF §IV-E smudging and Protocols 3–4, §IV-H Protocol 5, Appendix A/B security. BFV instantiation, single contribution round, external receiver distinction. |
| S2 | [Geva et al., ePrint 2023/1203](https://eprint.iacr.org/2023/1203), Appendix D pp.25–30, Protocol 2, Theorems 1–2 and Practical considerations; Appendix E pp.31–32, Protocol 3, Theorems 3–4 and RNS reserve discussion. |
| S3 | [Lattigo v6.1.1 multiparty README](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/README.md), passive adversaries, local-only networking scope, setup 1.i–iv, external output 2.iii. |
| S4 | [mpckks/refresh.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/mpckks/refresh.go) and [transform.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/mpckks/transform.go): `GenShare`, `AllocateShare`, `Finalize`, `Transform`, `applyTransformAndScale`, linear transform constraint. |
| S5 | [mpckks/utils.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/mpckks/utils.go): `GetMinimumLevelForRefresh`; [sharing.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/mpckks/sharing.go): private centered coefficient masks, unsupported-level check, E2S/S2E. |
| S6 | [multiparty/refresh.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/refresh.go): share pair / `BinarySize`; [keyswitch_sk.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/keyswitch_sk.go): Gaussian noise combination and single-polynomial share; [ring/poly.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/ring/poly.go), [utils/structs/matrix.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/utils/structs/matrix.go): uint64 coefficient representation/serialization. |
| S7 | [keyswitch_pk.go](https://github.com/tuneinsight/lattigo/blob/v6.1.1/multiparty/keyswitch_pk.go): two-polynomial external-receiver switch share, fresh encryption/noise. [Threshold evaluation-key example](https://github.com/tuneinsight/lattigo/blob/v6.1.1/examples/multiparty/thresh_eval_key_gen/main.go): optional `t<N` thresholdization and centralized test-only ideal-key sum. |
| S8 | [OpenFHE v1.4.2 INTERACTIVE_BOOTSTRAPPING.md](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/examples/INTERACTIVE_BOOTSTRAPPING.md): distinguishes Appendix D 2-party from Appendix E n-party and extra RNS limbs. |
| S9 | [tckks-interactive-mp-bootstrapping.cpp](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/examples/tckks-interactive-mp-bootstrapping.cpp), [ckksrns-multiparty.cpp](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/lib/scheme/ckksrns/ckksrns-multiparty.cpp): `IntMPBoot*`, mask basis, share pair, COMPACT/SLACK, demo collecting all private keys. |
| S10 | [interactive-bootstrapping.cpp](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/examples/interactive-bootstrapping.cpp), [rns-multiparty.cpp](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/lib/schemerns/rns-multiparty.cpp): `PolynomialRound`, `ExtendBasis`, `IntBootDecrypt/Encrypt/Add`, two-party reserve, decryption flooding modes. [base-multiparty.cpp](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/lib/schemebase/base-multiparty.cpp): public-key generation versus centralized private-key-vector overload. |
| S11 | [CKKS_NOISE_FLOODING.md](https://github.com/openfheorg/openfhe-development/blob/v1.4.2/src/pke/examples/CKKS_NOISE_FLOODING.md), “Solution: Noise Flooding”, “Static Noise Estimation”, query/statistical/error formula and 64/128-bit precision workflow. |
| S12 | [Pinned Qwen config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/7ae557604adf67be50417f59c2c2f167def9a775/config.json): exact dimensions. |

Also inspected [POSEIDON, arXiv:2009.00349v3](https://arxiv.org/abs/2009.00349v3)
abstract, cited by OpenFHE for CKKS masked/linear-transform lineage. Its
federated-training benchmarks are not used as a fixed-transformer inference
result. One guessed OpenFHE filename returned 404; actual examples were
resolved through the versioned GitHub directory listing. No capability was
inferred from that failed fetch.

Fetched PDF SHA-256:

- S1: `e9447abeeb3897fb4156f7c88c7e33ba23bde60378333365f1d00bee89c55a97`
- S2: `a441a1d58bf40aa6903f260398c0e47d103f56f5caf830b3535205f683da0392`

## 10. Exact checks, artifacts and next gate

New owned files:

- `docs/evidence/adjacent-refresh-review-2026-10-01.md`
- `scripts/probe_collective_refresh_feasibility.py`
- `tests/test_collective_refresh_feasibility.py`

Executed checks:

```sh
# Bounded standard-library oracle; reads only the locked local JSON evidence.
/usr/bin/time -l .venv/bin/python -B scripts/probe_collective_refresh_feasibility.py

# New protocol-specific algebra, range/session and directed-link regressions.
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_collective_refresh_feasibility.py

.venv/bin/python -B -m ruff check \
  scripts/probe_collective_refresh_feasibility.py tests/test_collective_refresh_feasibility.py
.venv/bin/python -B -m ruff format --check \
  scripts/probe_collective_refresh_feasibility.py tests/test_collective_refresh_feasibility.py
```

Final result: **13 passed in 0.06 s**; Ruff checks passed, two files already
formatted. `git diff --no-index --check /dev/null <path>` also passed for each
of the three owned files. Probe: **0.03 s real, 0.03 s user**,
maximum resident set **20,594,688 bytes**, peak memory footprint 10,109,360
bytes, comfortably below 1 GiB. Initial tests exposed tiny-even-q endpoint
effects; the final discrete guard explicitly includes them rather than
asserting an inapplicable exact continuous/asymptotic failure count. No model
weights, large keys or upstream runtime objects were allocated.

Backend introspection command executed:

```sh
.venv/bin/python -c 'import importlib.util; print({m: importlib.util.find_spec(m) is not None for m in ("tenseal", "openfhe", "pytest")}); import tenseal as ts; print({"tenseal_version": ts.__version__, "refresh_or_bootstrap_methods": {t.__name__: [n for n in dir(t) if any(k in n.lower() for k in ("bootstrap", "refresh", "multiparty"))] for t in (ts.Context, ts.CKKSVector)}})'
```

Result: TenSEAL present/version 0.3.17; OpenFHE absent; pytest present; no
matching Context/CKKSVector methods. This is introspection only, not a repeat
of the earlier ciphertext-depth experiment.

Additional exact bounded calculation executed (standard-library `python -B -c`):
enumerate k=1..95 in §8's expression for G=8/32; compute the hypothetical
128,000-byte boundary allowances. Results: `(53,477,376, k=48)` /
`(213,909,504, k=48)` and 11,715 / 2,324 bytes respectively:

```sh
.venv/bin/python -B -c 'print({"fixed_rounding_uncompressed_degree8192_depth_sweep": {g: min((16*8192*g*((96-1)//k)*(k+3), k) for k in range(1,96)) for g in (8,32)}, "hypothetical_128000_boundary_once_per_layer_allowance": {g: (budget-(9+g-1+g)*128000)//(23*(9+g-1)) for g,budget in ((8,7383174),(32,11354502))}})'
```

`shasum -a 256`
verified the two oracle PDFs and locked HE JSON. Primary PDFs were downloaded
with `curl --fail --silent --show-error --max-time 60` into approved temp paths
and converted with `pdftotext -layout`; other primary specs/code/config were
fetched read-only. No install, build dependency, runtime edit or commit.

**Next gate:** select one concrete protocol *and privacy placement*, then
produce a bounded, compiler-bound **complete-layer refresh-frontier contract**
before any full-model cryptographic build. It must supply actual active prime
values, security estimate for collective shares/QP, chosen masking or rounding
theorem/wrapper, coefficient and lifetime-query bounds, every live tensor/KV
layout, per-link serialized input/share/output bytes, exact rotation-key set,
setup/eval-key delivery, complete numeric circuit and a <1 GiB peak plan.
First reject on known full-response online/all-link cost. Reopen a real
backend probe only if this contract beats the GB-scale interactive bodies
with a concrete mechanism; then evaluate one complete real-weight layer,
with held-out prefill/same-token decode and final-output fidelity. The current
local backend's missing API and the current sampled layouts are explicit
blockers, not permission for a plaintext or toy-refresh substitute.
