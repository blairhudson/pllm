# Adjacent representation review: homomorphic secret sharing (HSS)

Date: 2026-10-01. Scope: HSS only, primary-source review and bounded independent
math. **No feasible fixed-checkpoint private decoder instantiation established.**
HSS is materially different from ordinary additive sharing: selected nonlinear
programs can be evaluated locally, with no per-multiplication peer openings.
The strongest direct constructions do not provide a practical universal
transformer evaluator.

## Decision and ranked candidates

1. **Roy–Singh DCR HSS, with token-bit inputs: strongest explicit state/adaptive
   contract to investigate on paper.** Its typed restricted-multiplication (RM)
   model explicitly supports online inputs, retained state and adaptive circuit
   choice [S5, §3]. Large messages and negligible error avoid the original DDH
   conversion barrier. An 18-bit-per-token input interface has a conditional
   2.54/3.87 MB upload projection at +8/+32 using the additive-decoding variant.
   This is a possible boundary, not a decoder: private embedding lookup, exact
   full-layer RM/branching-program compilation and compact terminal conversion
   remain unspecified. No CPU or complete-response byte admission.
2. **BKS direct RLWE HSS with newer SIMD: strongest lattice arithmetic candidate.**
   Cheap distributed-decryption multiplication, bounded-integer semantics,
   negligible error and packing are concrete. The 2026 tensor construction removes
   the interval scheme's prescribed-zero-secret restriction, retaining circular
   RLWE assumptions. Its reported instances and the screened BKS embedding
   interface miss both budgets before keys. Treat it as a compiler/packing
   research candidate; reopening requires a different input/output interface.
3. **Polynomial-modulus LWE HSS (ACK): promising share-size improvement, blocked
   resident composition.** Perfect correctness uses branch-list outputs and a
   non-additive decoder, expected rather than bounded work, and unresolved
   private reconstruction/timing handling [S4]. Trusted-client terminal decoding
   is more natural than using those lists as internal KV/activation state.
4. **DCR Vector-HSS: useful complete matrix–vector region, not a decoder.**
   Supports submitted encrypted/auxiliary vectors and private matrices; its
   comparison protocol is interactive and releases the comparison result.
   Its “inference” experiment computes `W x`, not SiLU/RMSNorm/attention or
   repeated decoding [S9, Fig. 3]. Public fixed weights make ordinary local
   linear evaluation on additive shares a much cheaper comparator.
5. **Original DDH HSS and low-degree HE-to-HSS compilers: deprioritize.** DDH
   has inverse-polynomial conversion error and input/key-dependent failure
   concerns. Constant-degree compilers do not cover composed decoder semantics.
   Generic FHE-backed HSS is functionally stronger but brings back FHE compute,
   depth/refresh/key costs; it is not evidence of a cheap direct HSS solution.

No complete **useful fixed-checkpoint nonlinear layer** is admitted. Complete
bounded inner products, matrix–vector products and quadratic reductions are
supported classes. Their applicability to a share-resident decoder depends on
how inputs enter those classes and how outputs become legal inputs to subsequent
nonlinear computation. Another scalar gate simulator would not resolve this.

## Locked problem and comparators

Pretrained checkpoint:
`Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`.
W8A8 body fingerprint:
`5d631be30158b3ea2a72cae355ce09a3b304bbfddde48b5758c339c346a34974`.
Dimensions: 24 layers, hidden width 896, vocabulary 151,936. The compiler-locked
cohorts execute 46 and 70 body rows: 39 prefill plus 7/31 decode rows.
Token selection occurs after each forward pass; the last selected token has no
subsequent body evaluation within its response.

| Cohort | Covered all-link control | Covered online control | 10× all-link budget | 10× online budget |
| --- | ---: | ---: | ---: | ---: |
| 39+8 | 116,843,966 | 73,831,744 | 11,684,396 | 7,383,174 |
| 39+32 | 178,970,558 | 113,545,024 | 17,897,055 | 11,354,502 |

All numbers are serialized-body bytes, not full wire. Cold model distribution
is reported separately and cannot disappear through amortization. Co-located
offset cold CPU median **38.01 s**, prepared **41.03 s**: two samples each,
cached source checkpoints, incomplete offset all-link reconciliation. These
are comparators, not a full CPU-cap or independent-provider privacy proof.

Read before this screen:

- `NETWORK_IO_10X_RESEARCH.md`, including the failed Q7 quadratic, 12-bit lift,
  whole-layer correlation and HE depth gates.
- `docs/evidence/compiler-region-contract-qwen25-2026-09-30.json`.
- `docs/evidence/prepared-cold-compare-qwen-smol-2026-09-29.json`.
- `python/pllm/runtime/shared_gate_polynomial.py`: bounded numerator
  `(256 g + g²)u`, not checkpoint SiLU or complete MLP.
- `python/pllm/runtime/he_layer_feasibility.py`: complete semantic operator list.
- `python/pllm/runtime/semantic_executor.py`: RMSNorm, stable softmax, branching
  float SiLU, attention and persistent KV semantics.

## Trust and function visibility

Proposed topology changes to **two online, non-colluding, semi-honest evaluators**.
Client generates/retains the full secret key and distributes distinct evaluation
key shares. Neither worker receives the full key, plaintext prompts, activations
or KV. A separate trusted dealer could perform setup, but then its transmissions
and trust are explicit; dealer collusion that exposes the full secret key is
outside this contract. Cancellation/replay, authenticated links, key ownership
and operation-ID binding still need protocol definitions. No TEE or training.

Public-function HSS gives both workers the program and its public weights. This
matches fixed publicly available checkpoints, but **does not hide model weights,
program topology, sequence lengths or timing**. Function-private evaluation is
a separate requirement: BGI's FSS application [S1, §4.3] is not automatically
obtained by invoking public-function HSS. Sharing weights as additional secret
inputs or using function-private HE changes input size and supported program
class and must be costed. Vector-HSS's private-matrix region is a specific
exception, not universal function privacy.

Ordinary additive shares support public linear maps locally. They cannot locally
multiply two secrets just because their shares are named “HSS.” Direct HSS
maintains structured key-dependent shares, together with suitable encrypted
input encodings. Having an additive output share is not possession of another
fresh encrypted HSS input.

## Verified construction restrictions

### DDH and optimized group HSS

BGI [S1, §§2–3] provides two-server HSS for branching programs. Sharing is
polynomial in input size/security parameter; evaluation is polynomial in program
size, magnitude bound and `1/δ`. The efficient ElGamal presentation assumes
circular security (Theorem 3.8); the DDH-only version removes that assumption
using a different encryption construction (§3, Theorems 3.17–3.18), with an
efficiency penalty. Do not label every optimized implementation plain DDH.

RMS instructions permit load, addition, **input × register**, and output.
They exclude register × register. Branching-program state can be updated using
input-bit selectors. This covers polynomial-size branching programs and
polynomial-size formulas/NC1 representations, not arbitrary polynomial-size
circuits at their native gate count. A representation's *program length* matters.

Original distributed-log conversion costs
`O(M log(1/δ)/δ)` group operations [S1, §3.1]. Later optimizations are much
better, but inverse-polynomial error remains. BKS discusses a group-conversion
barrier of `Ω(sqrt(B/δ))` absent improved interval discrete-log techniques and
an `S^(3/2)` evaluation dependence for fixed aggregate error [S3, §1.1].
These are barriers for that conversion approach, not impossibility results for
DCR groups or all HSS.

Optimized Las Vegas variants [S2, §§2.1, 4.1, 4.5] produce local potential-failure
flags. Failure is input/key dependent; repetition and failure sanitization have
communication/computation/privacy costs. Outputs are small additive/subtractive
shares modulo the requested output modulus, but sanitizing error cannot be
silently omitted. Reusable setup does not authorize unsafe pad/randomness reuse.

### Direct LWE/RLWE HSS: BKS

[S3, Theorem 1, §3, Fig. 4] maintains shares of `s y`, where `s=(1, secret)`;
input encodings encrypt the components of `s x`. Restricted multiplication
distributively decrypts to obtain shares of `s x y`, then locally rounds and
lifts to maintain the invariant. This is a **cryptographic share conversion**,
not exact fixed-point division or W8A8 requantization.

- Input encoding: `d` ciphertexts, each `d` ring elements. RLWE `d=2` gives
  **four Rq elements per general public-key input ring element**.
- Nonterminal multiplication/load: dominant cost **four Rq multiplications per
  worker**; terminal multiplication can use two. Additions are local.
- Bounds: all **unreduced integer-ring** intermediate coefficients bounded by
  `B`, with `B << p << q/B`; parameters are not controlled solely by circuit
  depth. Fresh randomization/PRF evaluation and unique instruction IDs remain.
- Negligible asymptotic error requires superpolynomial modulus gaps. Concrete
  Table 5 uses per-ring-operation error `2^-40`, not whole-response `2^-40`.
- Table 5 row `B=2^32`: `N=8192`, `log2(q)=203`, estimated 142-bit computational
  security. Other rows were only targeted at least 80 bits. These historical
  estimates are not a contemporary parameter audit for this application.
- CRT SIMD [S3, Appendix C.4, Theorem 6 and Remark 6] does not preserve small
  integer coefficient bounds. A packed finite-field program needs a bound for
  the actual lifted coefficient computation. Degree/depth can make that bound
  enormous even for Boolean-valued slots.

### Polynomial-modulus LWE: ACK

[S4, §1.1, §§3–5] detects ambiguous rounding/lifting locally. P0 remembers
flags; P1 computes multiple branches; terminal reconstruction chooses the
consistent branch. This yields perfect correctness **with non-additive
reconstruction** and expected-size lists, not fixed-size plain additive outputs.

For multiplicative program length `L`, dimension `N`, bound `B`, expected list
bound `γ>1`, Lemma 3 requires `p >= 8 B N L / ln(γ)` and
`q >= 8 p N L / ln(γ)` before additional concrete constraints. Tables with
`L=2^20, γ=2` give, for `B=2^32`, `N=4096, log(q)=104`, versus BKS's
8192/203. A fourfold input-size advantage in this setting is real; arbitrary
transformer program length is not `2^20` by assumption.

Worst-case branch work can be exponential. The paper expressly leaves
response-time leakage/private reconstruction handling open for secure 2PC.
Its non-adaptive expected-time formulation is not a proof for arbitrarily
adaptive repeated decoding. A client can receive branch lists at a final
boundary; an internal decoder region cannot expose flags/lists to a peer and
claim their selection/refresh is free. Fixed caps, padding, private selection,
tail bounds and continuation after reconstruction require a separate design.

### DCR large-message HSS and vector specialization

Roy–Singh [S5, §§3–4] directly captures adaptive online algorithms with typed
retained state. RM instructions allow a linear expression in **original input
wires** times a linear expression in registers. This improves linear-form
handling; it still excludes arbitrary register × register multiplication.

Damgård–Jurik inputs lie in `Z/(N^s)` and ciphertexts in `Z/(N^(s+1))`.
Theorem 20 gives per-operation conversion error at most `M N^(1-s)` for
integer magnitude bound `M`. Their example uses a 3072-bit RSA modulus and
`s=2`; this is not “all error is zero.” The initial decoder divides out a
secret factor `φ`; **additive decoding is a separate construction**, with a
second key, extra shares/exponentiations and roughly one extra level `s`
[§4.3, Fig. 6, Theorem 22]. Its error is approximately `2 M N^(2-s)`;
choose parameters for all operations and repeated outputs. Terminal shares
are cryptographic-size integers until a specified output conversion, not
automatically 32-bit hidden-state shares.

Orlandi–Scholl–Yakoubov [S6, §§3–4] has a perfectly correct Paillier distributed
discrete log, but the HSS construction still has negligible-error integer
share lifting and KDM-secure input/key encodings. “Perfect DDLog” does not mean
perfect full decoder arithmetic.

Vector-HSS [S9, Fig. 1, Protocols 3–4] removes ConvertInput for a specific
vector interface. Client sends additive vector shares plus an auxiliary
group element for one input type, and a vector ciphertext plus evaluation-key
shares for the other. Those extra encodings/key shares are indispensable.
Products of two newly computed nonlinear activation vectors do not acquire
that interface for free. Its cosine module normalizes the query at the user
and uses an available squared norm/approximate public scale; it is not private
RMSNorm. Comparison exchanges two Paillier ciphertexts one way and one back
and reveals the result to a server. That disclosed branch bit is incompatible
with hidden provider activation comparisons without an added protected-bit
protocol. Experimental primes were 512-bit each (roughly 1026-bit N), not a
128-bit-security 3072-bit-RSA benchmark. Its quoted speedups cannot price Qwen.

### Recent SIMD does not remove the program-class restriction

Interval SIMD [S7, §§4–5, Theorem 1, Construction 8] packs approximately
`sqrt(N)` bounded integer values and supports repeated **restricted** products
using `O(log N)` ring operations. Assumption: circular RLWE with entropic
secrets whose specified coefficient subset is zero. Extra automorphism,
ring-conversion and evaluation-key material must be distributed. The published
HSS example is `N=8192`, 32 slots, 175-bit q (or 16384/64 slots); the CPU
60 μs/slot figure is estimated from primitive microbenchmarks for bounded
Boolean RMS workloads, not measured transformer evaluation [§7.1, Table 2].

Tensor SIMD [S8, §§2–5] admits ordinary bounded/ternary secrets under **circular
RLWE for the required automorphisms**. Construction 1 maintains paired
encodings or a type-consistent alternating fast path. Construction 2 packs
`N^(1-o(1))` asymptotically, with extraction cost growing through automorphisms
and constant multiplications. Actual §5 instances:

| Instance | Ring dimension | Slots | q bits | Per-evaluator CT–share products / roundings / component constant products |
| --- | ---: | ---: | ---: | --- |
| C1 paired | 41,472 | 162 | 540 | 42 / 21 / 4 |
| C1 typed A→B | 41,472 | 162 | 540 | 20 / 10 / 2 |
| C2 | 45,360 | 495 | 540 | 68 / 34 / 232 |

The paper validates small `[-3,3]` inputs, outputs modulo 257, exact call counts
and a per-execution correctness-failure bound below `2^-180`. It does not
publish a fixed-Qwen secure runtime, contemporary computational-security
estimate for our workload, wire/key sizes or whole-response CPU. These claims
remain attributed source results, not independently reproduced cryptography.

The interval paper's aHMAC path has different, stronger power-RLWE assumptions
and uses plaintext data in tag evaluation; it is not an activation-private
HSS replacement. Its universal Boolean garbling application also uses an FHE
offline phase and per-gate material [S7, §6]. Count these costs before using it
as a bridge to arbitrary rounded arithmetic.

## Mapping a complete decoder, including numeric boundaries

| Required step | Direct HSS support / unresolved work |
| --- | --- |
| Public W8 dense projections, bias, residuals | Local linear operations on structured memory shares. Real integer accumulators and public scales require certified magnitude bounds. Producing a suitable encrypted **input-type** linear form is construction-specific. |
| Q·K and probability·V | Bilinear if Q/K or probabilities/V are supplied original encodings. Within a decoder both operands are computed state; native RMS instruction is unavailable. Expand into original inputs, use a BP representation, or pay for a secure re-encoding/hybrid bridge. |
| SiLU(gate)·up | A chosen polynomial SiLU makes a polynomial MLP, but gate/up are computed values. Low-degree expansion can handle one isolated region; actual stable float SiLU and all rounded edges remain different semantics. |
| RMSNorm | Sum of squares is degree two; inverse square root, epsilon, mean, gamma scaling, float operations and rounding are not native arithmetic HSS instructions. A complete exact finite-domain circuit/BP or a validated approximation is needed. |
| Causal softmax | Public position mask can be public; stable max/comparison, exp and reciprocal normalization are not polynomial-ring primitives. Finite-machine semantics can be compiled to Boolean circuits/BPs in principle, with unknown representation size. |
| Dynamic W8A8 quantization/rescale | Cryptographic share rounding removes encryption noise; it does not implement dynamic max/scale, signed conversion, accumulator requantization, ties-to-even or floating point. Private comparison/carry protocols must preserve result shares. |
| Rotary and public layout changes | Public linear/layout operations may be local. Rounded trigonometric constants, coefficient extraction, automorphisms and layout alignment need a cost/numeric contract. |
| KV append/reuse | Public-address appends can retain legal party-local memory shares. Future query/key products still need a supported wire type; updated computed KV is not another input ciphertext. Norm/noise bounds, randomization, identifiers, rollback and repeated token schedules remain required. |
| Final boundary and token choice | Client may reconstruct final hidden state, apply final norm/head and select a token. Output conversion and client head weights/MACs are charged; providers never reconstruct body hidden states. |

SiLU, inverse square root and softmax are not finite-degree real polynomials on
an interval. On a finite machine domain, exact lookup/interpolation or Boolean
representations exist, but their degree/program size and floating-point operation
order cannot be ignored. Approximations require held-out prefill, same-token
decode and free-generation checks on fixed weights without retraining. They
are not automatically pretrained W8A8 parity.

**Independent degree probe:** on 256 consecutive points over F257, the unique
degree-at-most-255 interpolation polynomials for the signed-i8 sign threshold
and signed-i8 division by two with ties-to-even both have degree **255**;
exhaustive 256-point reconstruction passes. This is a certificate for those
particular encodings, not a lower bound for Boolean comparison/rounding or Qwen.
Changing to bit inputs can yield short comparisons but changes the whole program.

**Degree versus depth:** `x -> x³` composed 24 times has degree
`3^24 = 282,429,536,481`, despite requiring only 48 unrestricted arithmetic
multiplications. An input-times-state RMS chain for that exact integer monomial
needs at least `degree−1` multiplications because each such step increases
degree by at most one. Twelve repeated squarings similarly have degree 4096
but need at least 4095 RMS-chain products. These are witness programs, not
universal decoder lower bounds. A deliberately simplified attention/MLP proxy
with linear probability polynomial and quadratic gate has degree `9^24`,
before norms/rounding. Neither “24 low-degree layers” nor “polynomial degree”
alone proves small RMS program length. Boolean BPs can avoid huge integer
magnitudes, but circuit→formula/BP expansion and state width are then the costs.

## Quantitative boundary screens and directed-link ledger

These are **unseeded, tightly bit-packed projections for specified encodings**,
not measured ciphertext serialization and not universal HSS lower bounds.
Input encodings are identical at both workers, but delivering them costs two
links. Sending once to A and forwarding A→B still transmits two copies overall.
Seeded serialization, coefficient selection or token-only input protocols could
change these figures only with a concrete security/encoding contract.

### Public-key embedding interface

For BKS's cited 8192/203 parameters, one ring element is 207,872 bytes and one
four-element encoding is 831,488 bytes **per worker**. Granting an unvalidated
8192-value packing gives five prefill blocks, then **one fresh block for each
of 7/31 adaptive decode inputs**. Packing all future embeddings together would
give only 6/8 blocks, but those embeddings are unavailable before token choice.

| Encoding / cohort | Client→A | Client→B | Input total | Conditional 32-bit A→client + B→client |
| --- | ---: | ---: | ---: | ---: |
| Optimistic BKS full-ring / +8 | 9,977,856 | 9,977,856 | **19,955,712** | 57,344 |
| Optimistic BKS full-ring / +32 | 29,933,568 | 29,933,568 | **59,867,136** | 229,376 |
| Tensor C2 reported instance / +8 | 1,041,012,000 | 1,041,012,000 | **2,082,024,000** | 57,344 |
| Tensor C2 reported instance / +32 | 1,628,877,600 | 1,628,877,600 | **3,257,755,200** | 229,376 |

All four fail both corresponding budgets on input alone. Tensor C2 has
3,061,800-byte ring elements; a four-element input delivered to both workers
is **24,494,400 bytes for 495 values**, already above the +32 all-link budget.
One 896-wide decode embedding needs two blocks, 48,988,800 bytes. The multi-GB
figures are integer projections only: no such allocations or keys were created.

The 32-bit output column assumes an exact bounded terminal conversion that has
**not** been implemented. It is an illustrative optimistic interface, not a
price for native `Rq`, integer REG shares or ACK branch lists. Cryptographic
output words can be much larger. Intermediate outputs cannot be sent to the
client and reshared for free under a resident-layer proposal.

### DCR boundaries: scalar embedding veto, token-bit possibility

At 3072-bit N and `s=2`, each IN ciphertext modulo `N³` costs 1152 bit-packed
bytes per worker; the additive-decoding variant with `s=3` costs 1536. This
prices only ciphertext delivery, not all setup or output shares.

| DCR input boundary | +8 both-worker bytes | +32 both-worker bytes |
| --- | ---: | ---: |
| One ciphertext per embedding scalar, s=2 | 94,961,664 | 144,506,880 |
| One ciphertext per embedding scalar, s=3 | 126,615,552 | 192,675,840 |
| 18 input bits per token, s=2 | **1,907,712** | **2,903,040** |
| 18 input bits per token, s=3 | **2,543,616** | **3,870,720** |

Token-bit input is possible at the abstraction level: privately evaluate public
token lookup and the remaining program on encrypted bits. It saves input
traffic but may require expensive scanning/branching and bit-level rounded
arithmetic. No bound for its actual RMS/BP length, computational work, output
conversion or KV state is known here. This sub-budget ingress is **not a pass**.
It also demonstrates why embedding-upload failures cannot veto all HSS layouts.

### Unpriced work is not zero

- Client/dealer→A and →B: public key and distinct secret evaluation shares;
  tensor automorphism keys, PRF seeds/public setup, optional FHE/garbling material.
  Client setup avoids a third-party dealer only if its full work is counted.
- A↔B: zero per-gate online interaction for a pure supported direct HSS program;
  **unknown**, not zero, for any re-encoding, protected rounding, comparison,
  ACK private list selection or hybrid arithmetic needed by this decoder.
- A/B→client: terminal structured shares, exact combination/decoding and final
  hidden representation. Raw `φ y` or `Rq` words cannot be assigned i32 size.
- Client→A/B on feedback: fresh ciphertexts/encodings for each chosen token.
  Setup can be reusable; adaptive inputs cannot be batched across unseen tokens.
- Worker state: persistent KV, input encodings, structured register components,
  transformations, fresh PRF IDs and whole-response error accounting. Ordinary
  KV storage is not automatically compatible with nonlinear HSS continuation.
- Cold distribution: compiler evidence lists 44,236,800 attention-side plus
  314,806,272 MLP weight/scale bytes. Replicating these uncompressed snapshots
  to two cold workers projects **718,086,144 bytes** for tracked body weights
  alone; compressed/full-wire delivery remains unknown. Embeddings, head,
  source checkpoint encoding and client head distribution add costs. Preinstalled
  public weights may amortize these later but do not erase cold delivery.

For BKS's example per-operation `2^-40`, a union upper bound across `2^20`
operations is `2^-20 ≈ 9.54e-7`; across `2^35` it is `1/32`. These are illustrative
union bounds, **not measured failure rates or Qwen operation counts**. Actual
global correctness target, conversions, both phases and all KV reuse determine
parameters. Malicious robustness/authentication is distinct from input privacy.

Most compute can reside remotely if a legal program exists, but duplicated
cryptographic evaluation by both workers, client setup/encryption, output
conversion and head MACs must be measured. Source microbenchmarks and the
co-located 38.01/41.03-second medians do not establish that aggregate cap.

## Complete regions and the next falsifiable gate

**Supported, complete abstract regions:**

- BKS terminal degree-two reduction `sum_i x_i²` or inner product, where operands
  are admitted input encodings and magnitudes are bounded. Uses actual HSS, not
  a dealer's five-coefficient polynomial correlation. It returns a reduction,
  not RMSNorm, and its input/output conversion has to be included.
- DCR Vector-HSS full submitted-input `W x` with its matrix/vector setup,
  key shares, auxiliary rows and final output shares [S9, Protocol 4/Fig. 3].
  Useful for private-weight one-shot linear inference; for public Qwen weights
  it adds crypto to a linear operation already cheap under ordinary sharing.
- BGI/BKS/RS public branching-program evaluation with encrypted input bits.
  A polynomial-size BP is a supported target. This review has no such complete
  BP for the rounded fixed decoder.

**Unimplemented:** secure instantiation/parameter audit, compiled whole layer,
SiLU/RMSNorm/softmax/quantization semantics, computed-state product bridges,
private terminal conversion, compatible KV continuation, independent workers,
full directed-link serialization, held-out checkpoint parity and full CPU cap.

**Next gate — representation certificate before cryptographic implementation:**

Choose RS's typed adaptive RM construction or BKS plus a specific SIMD encoding.
Submit one exact **complete semantic layer** (both norms, seven projections,
rotary, QK, mask/softmax, PV, residuals, SiLU×up, all W8A8 rounded edges and
KV append), plus the final-hidden conversion and token-feedback interface.
Each multiplication must identify an admissible original-input encoding or an
explicit charged bridge. If a Boolean/BP lowering is used, report its actual
instructions, width/state, source-bit counts and maximum integer/coefficient
magnitudes; do not call the original circuit its BP. Include two successive
decode steps using retained KV to expose type-conversion/state issues.

Pass condition: a source-backed security/correctness contract, legal instruction
typing, exact numeric parity on a bounded oracle, and concrete parameterized
directed-link **upper** estimates below both matched cohort budgets after
counting setup, all conversion/material and feedback. For the s=3 token-bit
boundary, only **4,839,558 / 7,483,782 online bytes** remain at +8/+32 for
every other online operation and output. Full all-link budgets leave
9,140,780 / 14,026,335 bytes before setup/material/wire. Output i32 parity,
global failure budget and most-dense-work-remote must also be established.
Fail immediately on an untyped register product, omitted rounded operator,
incompatible KV continuation or known byte floor over budget. If the certificate
passes, measure one protected complete layer under strict resource caps; until
then, no runtime topology or decoder build is justified.

## Primary sources verified

All landing pages below were retrieved with **webfetch on 2026-10-01**. For
S1–S9, PDFs were fetched separately into the approved
temporary directory and converted with existing `pdftotext -layout` for section
verification. This is source verification, not implementation/security auditing.

| ID | Primary source / URL | Sections verified and scope |
| --- | --- | --- |
| S1 | Boyle, Gilboa, Ishai, *Breaking the Circuit Size Barrier for Secure Computation Under DDH* (CRYPTO 2016), https://eprint.iacr.org/2016/585 ; https://eprint.iacr.org/2016/585.pdf | §§2–3 definitions, RMS, conversion, circular versus DDH-only variants; §4.3 FSS; abstract BP/NC1 claim |
| S2 | Boyle et al., *Homomorphic Secret Sharing: Optimizations and Applications* (CCS 2017/full version 2018), https://eprint.iacr.org/2018/419 ; https://eprint.iacr.org/2018/419.pdf | §§2.1–2.2, 4.1, 4.5–4.6: Las Vegas flags/leakage, terminal multiplication, BP/formula support |
| S3 | Boyle, Kohl, Scholl, *Homomorphic Secret Sharing from Lattices Without FHE* (EUROCRYPT 2019), https://eprint.iacr.org/2019/129 ; https://eprint.iacr.org/2019/129.pdf | Theorem 1, §§1.1–1.2, 3–4, Tables 1–2/5, Fig. 4; Appendix C.4 Theorem 6/Remark 6 |
| S4 | Attema, Capitão, Kohl, *On Homomorphic Secret Sharing from Polynomial-Modulus LWE* (PKC 2023), https://eprint.iacr.org/2023/382 ; https://eprint.iacr.org/2023/382.pdf | §1.1, Lemmas 1–3, Tables 1–2; §§3–5: corrected lists, expected work and private reconstruction caveats |
| S5 | Roy, Singh, *Large Message Homomorphic Secret Sharing from DCR and Applications* (CRYPTO 2021), https://eprint.iacr.org/2021/274 ; https://eprint.iacr.org/2021/274.pdf | §§3–4, Definition 4, Theorems 20/22, Figs. 5–6: adaptive state, typed RM, magnitude/error and additive decoding |
| S6 | Orlandi, Scholl, Yakoubov, *The Rise of Paillier: Homomorphic Secret Sharing and Public-Key Silent OT* (EUROCRYPT 2021), https://eprint.iacr.org/2021/262 ; https://eprint.iacr.org/2021/262.pdf | §§3–4, Lemma 3.3, Definition 4.2/Theorem 4.4/Construction 4.5: perfect DDLog versus negligible-error HSS |
| S7 | Kim, Li, Lin, Liu, *SIMD HSS and aHMAC from Interval Encoding with Application to One-Bit-Per-Gate Garbling* (2026 preprint), https://eprint.iacr.org/2026/485 ; https://eprint.iacr.org/2026/485.pdf | §1.1, §§4–5, Construction 8, §§6–7.1, Tables 1–2: RMS restriction, entropic/circular assumptions, garbling offline FHE, microbenchmark estimates |
| S8 | Kim, *Tensor Encodings for SIMD HSS* (2026 preprint), https://eprint.iacr.org/2026/1569 ; https://eprint.iacr.org/2026/1569.pdf | §§1–2 and 5, Table 2: circular automorphism assumptions, packing/extraction, concrete dimensions/modulus and restricted-product validation |
| S9 | Yang et al., *ConvertInput-Free Vector Homomorphic Secret Sharing and Its Applications* (2026 preprint), https://eprint.iacr.org/2026/1517 ; https://eprint.iacr.org/2026/1517.pdf | Fig. 1, Protocols 2–4, Fig. 3, §7.1: encodings, revealed comparison, MVM-only inference, experimental modulus |
| S10 | Lai, Malavolta, Schröder, *Homomorphic Secret Sharing for Low Degree Polynomials* (ASIACRYPT 2018), https://eprint.iacr.org/2018/1065 | **Landing abstract only:** degree-`k` (multi-key) HE → degree-`((k+1)m−1)` public polynomials; polynomial inputs and sub-logarithmic server count. Not evidence of arbitrarily composable decoder support. |
| S11 | Chillotti et al., *Scooby: Improved Multi-Party Homomorphic Secret Sharing Based on FHE* (SCN 2022/revised 2024), https://eprint.iacr.org/2022/862 | **Landing abstract only:** general-circuit HSS requires FHE; Scooby superpolynomial modulus/noise ratio, Scrappy generic FHE+NC1 HSS; Shaggy is limited constant-degree without FHE. |

## Verification and saved artifacts

New files only:

- `docs/evidence/adjacent-hss-review-2026-10-01.md` — this review and locked results.
- `scripts/probe_hss_feasibility.py` — standard-library math/cost audit, no model,
  network, dependency install, cryptographic key generation or runtime imports.

Exact executed probe command:

```sh
/usr/bin/time -l python3 -B scripts/probe_hss_feasibility.py
```

Final execution: **exit 0**, 0.04 s wall, 0.03 s user, 0.00 s system,
**20,234,240 bytes maximum RSS**, no swaps. Checks: both 256-point interpolation
certificates exhaustively reconstruct; signed negative/positive half ties pass;
checkpoint/cohort row locks and adaptive packing pass. Printed JSON contains
the traffic/degree/error-bound numbers tabulated above. No secure HSS scheme,
trained/modified checkpoint or full layer was executed. The script uses O(256)
working entries and O(256²) small-field operations; projections allocate no
tensor/ciphertext-sized arrays.

PDF verification used existing tooling (nine PDFs, no build/install), after
checking the temporary parent directory:

```sh
for id in 2016/585 2018/419 2019/129 2023/382 2026/485 2026/1569; do name=${id//\//-}; rtk curl --fail --silent --show-error --max-time 60 "https://eprint.iacr.org/${id}.pdf" -o "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.pdf" && pdftotext -layout "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.pdf" "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.txt"; done
for id in 2026/1517 2021/262 2021/274; do name=${id//\//-}; rtk curl --fail --silent --show-error --max-time 60 "https://eprint.iacr.org/${id}.pdf" -o "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.pdf" && pdftotext -layout "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.pdf" "/var/folders/3d/16hqtng12ng_bfz1n99jg04w0000gn/T/opencode/pllm-hss-${name}.txt"; done
```

Both commands exited 0; dedicated Read/Grep tools verified the cited sections.
These temporary source files are not deliverables or runtime resources.

Whitespace verification for both new, untracked files:

```sh
rtk git diff --no-index --check /dev/null docs/evidence/adjacent-hss-review-2026-10-01.md
rtk git diff --no-index --check /dev/null scripts/probe_hss_feasibility.py
```

Both exited 0 with no output. Existing untracked research work, including
`NETWORK_IO_10X_RESEARCH.md`, was preserved.
