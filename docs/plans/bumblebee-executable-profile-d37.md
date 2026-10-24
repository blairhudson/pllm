# D37 B-S: a concrete BumbleBee profile and a separate backend experiment

**Status: design proposal only, 24 October 2026.** Neither numerical adoption,
backend adoption, implementation, dependency changes, key generation, encrypted
execution nor SDK availability is authorized by this document. D37 authorizes
this specification, not its choices. Base for this document: `7fa07d74`.

Candidate name: **`bumblebee.final-paper-derived-gpt2-mixed.v1`**. This is a
proposed, independently specified interpretation, **not** an identified author
configuration or an assertion of evaluated-build fidelity. It is not the Q16
OpenBumbleBee artifact, PUMA arithmetic, or the common-Qwen adaptation.

This packet contains both requested attachments:

1. **Attachment A:** complete proposed arithmetic/branch/frontend contract.
2. **Attachment B:** concrete proposed HE/OT/key/error graph.

Section C retains a future qualification proposal for pinned native-Rust
`fhe.rs`, **NOT SELECTABLE** until its missing fields close. None of these
sections silently selects another.
Blocking evidence can remain missing even when a proposed value is explicit.

**Astra review revision, Main `a61f4bf6c8f52c31e6584de4aad5cb65dd23be28`:**
Attachment A is unchanged. B2/B3 now freeze proposed phase/sign/rounding equations;
these are independently specified choices, not recovered backend behavior.
Section C and B-S/Q remain **NOT SELECTABLE**. Only D's bounded source-only
analysis tranche is proposed as the next action; no whole-profile adoption is
requested. Original C8 document and artifact are preserved separately.

## Evidence vocabulary and audited reuse

- **R — recovered:** stated by the locked final paper or existing reviewed
  source; the locator states which. Recovered artifact facts are not final-paper
  choices.
- **P — proposed:** an explicit new interpretation for a later decision. A
  calculable constant or tuple is not author evidence or security evidence.
- **U — UNKNOWN / blocking:** evidence or a qualified implementation is absent.
  No default, model name, library name or successful toy test closes it.

Governing final paper: `papers/r22-bumblebee-final.pdf`, NDSS 2025,
SHA-256 `184bbe5104f183d4df66db2413f89c96c6c814e0b0ec6297f19745a1fa6e203b`.
Its bytes were checked locally for this preparation. See the existing
[source lock](../../paper/methods/source-lock.v1.json),
[B-N1 packet](bumblebee-executable-profile.md),
[source-resolution section 4](../../paper/reproduction-contracts/shared-profile-resolution.md#4-bumblebee-conversion-choices-are-concrete-final-paper-identity-is-not),
and [D21 indexing corrections](bumblebee-indexing-corrections.md).

| Existing contract inspected | Reuse, and actual limit |
| --- | --- |
| `crates/pllm-core/src/coefficient_layout.rs` | Checked rectangular packing, negative-exponent signs, global ParseMat indexing and trace/interleave workspace. `MAX_DEGREE=4096`, at most 256 interleave inputs; **not source N=8192**, encryption or key switching. |
| `crates/pllm-core/src/coefficient_repacking.rs` | Existing odd-limb inverse-power-of-two algebra and public ciphertext extraction. Existing serialized degree-4096/two-limb API must not be relabelled source-sized. |
| `crates/pllm-core/src/coefficient_conversion.rs` | `Delta=floor(q/t)`, shared nearest `down`, and explicit caller-supplied error/zero-error/mask phases. Its exact u128 capacity admits **no ring64 q**, not even a convenient smaller source q. It promises neither freshness nor universal one-bit reconstruction. |
| `paper/reproduction-contracts/shared_profile_oracle.py` | Exact bounded share-level truncation/comparison, positive extension, Lift/Down and coefficient-phase references. Clear oracles do not supply HE/OT, circuit privacy, secure randomness or source authority. |
| `crates/pllm-compiler/src/encrypted_level_schedule.rs` | Reuse immutable SSA/source owners, full backing liveness and priced persistent/scratch transitions. Do not encode BFV/OLT as fictitious CKKS depth. |
| Ordinary component/source/admission/runtime owners | Reuse actual source pins, original allocation counters, immutable input custody, bounded channels, cancellation and final output policy. No new BumbleBee runtime framework or local trusted OT dealer. |

External code facts below are inherited from the existing pinned static review;
this preparation did not fetch or execute upstream code. Relevant source pins:

- `AntCPLab/OpenBumbleBee@47c5560d069543591f3b0176eb530ede28e33fc3`:
  reviewed `encoding.cc`, `activation.cc`, `truncate_prot.cc`, `ring_ext_prot.cc`,
  `cheetah_dot.cc`, `cheetah_mul.cc`, `modswitch_helper.cc`, `simd_mul_prot.cc`
  and `rlwe/utils.cc`; exact locators in source-resolution section 4. Its
  Q16/four-square/±3 example and 218/163-bit OLT, 159-bit bOLE contexts are
  **negative mapping evidence**, not parameters selected here.
- `tlepoint/fhe.rs@98a087f1bf5f7fa95196ebe16b0e1b0eeb632442`:
  [existing backend review](../../paper/methods/rust-he-backend-qualification.md)
  and [manifest](../../paper/methods/rust-he-backend-sources.json). Reviewed
  checkout declares `fhe` 0.2.0; the registry observation 0.1.1 is not this pin.
  BFV, BigUint plaintext parameter expressibility and raw ciphertext components
  are prerequisites only. Experimental status, Apple-M support, ring64
  plaintext behavior and complete selected-parameter support remain unqualified.

## Attachment A — proposed per-node arithmetic

### A1. Word, encoding and source/frontend identity

**R:** final §II.B uses floor encoding and two-party additive shares; §VII uses
ring64/Q18 except mixed GeLU and exp in ring32/Q12. **P:** freeze the following
precise interpretation instead of importing artifact rounding:

| Field | Proposed exact value |
| --- | --- |
| Wide / mixed word | `Z/(2^64)` / `Z/(2^32)`, centred two's-complement interpretation; canonical wire residues unsigned. |
| Encoding `E_f` | `floor(r * 2^f)` of the **exact finite source value**, then residue conversion. Reject nonfinite values and absent signed-range evidence; never saturate. IEEE source words denote their exact dyadic rational; decimal strings denote exact base-10 rationals. No implicit f32 intermediate, fitting, calibration or runtime float conversion. |
| Public constants | Exact decimal/rational strings below, separately floor-encoded at the named scale. Constants, biases and weights bind original source words and tensor orientation. |
| Arithmetic | Ring add/subtract; integer sums in increasing logical reduction index. Public arithmetic constants injected into P0's share, except explicitly designated protocol offsets (Alg.5's M goes to P1). Mathematical interpretation requires separately admitted ranges/errors; modular arithmetic alone supplies no no-wrap certificate. |
| Fixed sizes proposed for the GPT2-base graph | 12 pre-LN blocks; hidden 768; 12 heads of width 64; MLP 3072; vocabulary 50,257; learned absolute positions; no RoPE, RMSNorm, SiLU, dropout or KV cache. These specify the candidate graph, not an identified checkpoint. |
| Transaction | Batch one; initial prefix `1 <= P`, exactly `G >= 1` private decode steps, `P+G <= 128`; recompute the full current prefix. Last row only enters the full-vocabulary output head. Fixed-count feedback, lowest-index protected argmax; final client-only token IDs per D9. EOS does not shorten protected work. |
| Source owners | P0 owns private input/tokenizer output; P1 owns private model parameters. A parameter held only by P1 has structural P0 share zero, not a falsely public weight. Generic shared operands use two additive shares. Tied embedding/head retains the same original tensor owner, not a digest-matched copy. |

**U, blocking full-model use:** exact checkpoint/revision and tensor hashes;
tokenizer revision/files/normalizer/BOS/EOS/padding; original frontend lock;
learned-position bounds; weight orientation and alias map; calibration/workload
identity; author linkage and evaluated dependency/configuration manifest.
Model dimensions do not recover any of these. Proposed resolution: bind an
immutable author-linked manifest if obtainable; otherwise seek a **separate
explicit checkpoint/frontend selection** labelled derived, not author-evaluated.
Do not fabricate an author, repository revision, calibration corpus or source
certificate. Source import and full constructor/CPython admission remain gates.

### A2. Primitive semantics; every use is a graph node

1. **`T(ell,b)` — P:** use the reviewed two-party truncation law, not PUMA
   `TruncAPr`. `b=0` is identity; otherwise `1 <= b <= ell-2`. For input
   shares `x0,x1`, let `N=2^ell`, `a=(x0+N/2) mod N`, `c=x1`,
   `w=floor((a+c)/N)`. An exact protected wrap protocol gives additive shares
   `w0,w1` of w. Define
   `z0=(floor(a/2^b)-w0*2^(ell-b)-2^(ell-b-1)) mod N`,
   `z1=(floor(c/2^b)-w1*2^(ell-b)) mod N`, then output
   `(z0+(z0&1), z1) mod N`. Wrap shares are fresh and not opened.
   The reviewed uniform-split marginal errors `-1,0,+1` have probabilities
   `(2^b-1-r)/2^(b+1), 1/2, (r+1)/2^(b+1)`, `r=X mod 2^b`.
   This is not a distribution-independent or across-gate independence theorem.
   The actual joint share/OT schedule is normative; no injected ±1-noise stand-in.
2. **`M_f(A,B)` — P:** protected product in the same ring, then exactly one
   `T(ell,f)`; `S_f(A)=M_f(A,A)` uses the square cross-term optimization.
   A dot product accumulates its complete raw product reduction **before** its
   single T, not per MAC or per HE inner block. HE reconstruction error enters
   before T, not after it and not assumed to disappear.
3. **`LessApprox_f(X,Y)` — P:** form additive shares of `X-Y`, drop the low
   `delta=f-4` bits **locally on each share**, then exact protected MSB of their
   sum in `ell-delta` bits. Thresholds are E_f encoded. Wide Q18 uses delta 14
   and a 50-bit MSB; mixed Q12 uses delta 8 and a 24-bit MSB. The paper's
   reported 50-bit optimization must not be claimed for a 32-bit call.
   Near a pivot, even equality is share-dependent, not a deterministic rounded
   comparison. Public admitted difference ranges must exclude signed wrap.
4. **`LessExact` / sign / Mux — P:** exact protected signed comparison on
   admitted differences; exact Boolean-to-arithmetic selection with fresh OT
   masks. Used for absolute value, secret exponent selection, norm floor and
   final argmax. Never reveal a branch bit, exponent, validity or intermediate.
5. **Mixed entry — P:** reduce each wide share modulo 2^32 **first**, then
   `T(32,6)` from Q18 to Q12. The wide input must already have a valid signed
   ring32 Q18 interpretation: `-2^31 <= X18 < 2^31`.
6. **Mixed exit — P:** multiply each Q12 share by 64 in ring32; then positive
   extension to ring64 with `M=2^30`. Following final Alg.5, P1 adds M;
   obtain protected wrap shares for the updated unsigned ring32 sum; subtract
   `2^32*w_l` and subtract M from P1 after lifting. Requires the **restored Q18
   reconstructed integer** in `[-2^30,2^30)`. No clear clipping or unpriced
   general-extension fallback. The artifact's heuristic is selected only as a
   proposed interpretation; final paper does not supply this exact M.

**R/P distinction:** the paper describes these functionality families and mixed
precision; the concrete truncation/cast/extension laws above are **P**, informed
by reviewed code, not recovered evaluated-final-paper associations. Every
protected bit operation still needs the qualified distributed OT implementation
in Attachment B. No functionality name makes protocol work or storage free.

### A3. Mixed GeLU: exact constants, tails and ordered powers

**R:** final §V.C/App.D give the quartic on `[-2.7,2.7]` and these exact strings:

```text
c0 =  0.001620808531841547
c1 = -0.03798164612714154
c2 =  0.5410550166368381
c3 = -0.18352506127082727
c4 =  0.020848611754127593
```

**P:** after mixed entry let `X` be Q12,
`Aminus=E12(-2.7)=-11060`, `Aplus=E12(2.7)=11059`.
Floor encoding is asymmetric: never replace Aminus by -Aplus.
Compute `L=LessApprox12(X,Aminus)`, `U=LessApprox12(Aplus,X)`,
`I=(!L)&(!U)`, then `Y=Mux(I,X,0)` **before powers**. This safe-argument
selection is a declared proposal, not a recovered author lowering or a clear
branch. It avoids evaluating discarded tails at arbitrary large magnitude.
Use exact sign for `R=abs(Y)`; do not substitute the artifact's approximate
zero predicate. In order:

```text
R2 = S12(R)
R4 = S12(R2)
R3 = M12(R, R2)       # retain/reuse the same immutable operand encryptions
Q24 = 2048*Y + 27192 - 156*R + 2216*R2 - 752*R3 + 85*R4
Q12 = T(32,12)(Q24)  # one accumulated-polynomial truncation
Z12 = Mux(L, 0, Mux(U, X, Q12))
result = mixed_exit(Z12)
```

The negative floor coefficients **-156 and -752** deliberately differ from
artifact toward-zero -155/-751. `27192` is the Q24 constant; no extra epsilon
or offset subtraction. Left tail zero, right tail identity, and approximate
endpoint behavior are **P**; the interval/coefficient list alone does not define
them. The wide four-segment GeLU and its Eq.(2)/Alg.3 epsilon discrepancy are
excluded. No ±3 or artifact SiLU path is implicitly enabled.

Required range records cover mixed entry/exit, pivot differences, the safe core
including quantization/pivot slack, every error-bearing power, each signed
coefficient product and the **partial Q24 accumulation**, not only the selected
output. A caller's assertion or sampled calibration is not a source proof.

### A4. Stable softmax: six squares, explicit clipping and normalization

**R:** final §VII selects cutoff -14 and six squares; GPT2 exp uses mixed Q12.
**P:** for each causal row, in logical key order:

1. Compute exact protected maximum of the **active causal entries only** in
   Q18; subtract it to obtain Z. Public padding/future positions are excluded
   structurally, not represented by an unbounded negative floating sentinel.
2. `K = !LessApprox18(Z, E18(-14))`; `Zsafe=Mux(K,Z,0)` in ring64.
   Wide-before-mixed comparison/safe-argument order is explicitly P. It prices
   the 50-bit comparison and prevents a clipped large negative value from
   overflowing the eager mixed path. No clear predicate or host fallback.
3. `X=mixed_entry(Zsafe)`; `V=4096 + T(32,6)(X)`; perform exactly six
   ordered `V=S12(V)` calls, including division/truncation by 64 first.
4. `E12=Mux(K,V,0)`; restore/extend each element to Q18 using A2. Sum all
   active E18 in increasing index. Call the reciprocal below **once per row**;
   output each `M18(E18_i, reciprocal(sum))`. No renormalization, clipping to
   unit sum or forced probability one. Empty active rows are invalid.

Do not equate this to an artifact four-square schedule, illustrative cutoff -13,
or deterministic pivot rule. All possible internal errors, denominator bounds
and extension conditions require admission; the mathematical maximum's exp=1
is not by itself a proof about the error-bearing implementation.

### A5. Reciprocal and rsqrt: closed proposed schedules, not recovered identities

The paper refers to existing functionality; its exact evaluated schedules are
**U**. The following **P** gives a reviewable complete alternative, rather than
borrowing PUMA masks or leaving iteration counts implicit.

Define `Scale(v,j)` to multiply a Q18 integer by 2^j: ring left shift for j>=0;
`T(64,-j)` otherwise. For a **secret** j, build all candidates over the public
finite exponent set and select using protected one-hot bits. Charge every
candidate T and selection even if discarded; no secret loop count or disclosure.
Unselected ring-left-shift candidates may reduce modulo 2^64; they are not
claimed unwrapped model intermediates. Selected results require their own range
proof. Secret exponent determination uses a fixed exact bit/priority circuit,
not a host `log`, floating `sqrt` or clear opening.

- **Reciprocal:** require positive Q18 raw D `<2^61`. Let
  `k=1+floor(log2(D))`, k in 1..61, selected privately. Set
  `C=Scale(D,18-k)`, `W=763940-2*C` (`E18(2.9142)=763940`). Repeat
  `W=M18(W, 524288-M18(C,W))` exactly twice. Return
  `Scale(W,18-k)`. Thus each normalization/compensation selects among 43
  right shifts, 17 left shifts and identity; each iteration's two M calls
  has its own truncation. This makes the reviewed initialization family
  explicit without claiming final-paper evaluated identity.
- **Rsqrt:** require positive Q18 raw D `<2^61`. Privately choose
  `m=ceil((k-18)/2)`, m in -8..22; set `C=Scale(D,-2*m)`, `Y=262144`.
  Exactly eight iterations, ordered as
  `Y2=S18(Y); A=786432-M18(C,Y2); Y=T(64,1)(M18(Y,A))`.
  Return `Scale(Y,-m)`. Ideal normalization is `C in [1/4,1)`; admission
  must include actual truncation/product errors. This is an explicit Newton
  proposal, **not** the artifact's initial rsqrt polynomial, fitted coefficients,
  a source error bound or permission to replace a paper algorithm silently.

There is no fallback on zero/negative denominator, convergence failure or absent
range proof. A5 needs separate numerical acceptance and independent range/error
and whole-checkpoint quality evidence. It is intentionally not described as
already matching source or having a proved precision.
Positivity/convergence are public pre-execution obligations over admitted states,
not permission to abort or branch observably on a secret denominator. No
secret-dependent failure channel is adopted here.

### A6. LayerNorm, attention, residuals and complete block schedule

**P:** LayerNorm on d=768 uses this exact graph, with Q18 gamma/beta:

```text
mu = T(64,18)(sum_in_index_order(X) * E18(1/768))  # E18(1/768)=341
D_i = X_i - mu
S_i = S18(D_i)
V0 = T(64,18)(sum_in_index_order(S) * 341) + 2    # E18(1e-5)=2
V = Mux(LessExact(V0,2), 2, V0)
R = rsqrt(V)
Y_i = M18(M18(D_i,R), gamma_i) + beta_i
```

The mean-square convention, epsilon placement/encoding and multiplication
order are proposed values, not inferred from a model name. The **secret floor
of V0 at 2 is an explicit additional proposal**: approximate products/truncation
can make even a zero-variance row nonpositive; epsilon two alone is not a proof
otherwise. It neither opens V0 nor silently changes a source checkpoint's
epsilon. Rejecting this proposal leaves A6 blocked until another fully specified,
safe normalization contract is separately accepted. Do not claim checkpoint
equivalence or use a clear zero-row shortcut.

The candidate per-layer lowering is fixed:

| Order / node | Ring, scaling and operand order |
| --- | --- |
| Input lookup | Private one-hot token row times P1's embedding table via OLT, Q18 table values; add learned position row on P1. Original table retained for tied head. |
| Pre-attention norm | A6; all rows, gamma then beta in the order above. |
| Q/K/V projection | For each output cell accumulate complete raw Q36 dot, then `T(64,18)`, then add Q18 bias. Q, K, V logical concatenation/order bound by frontend manifest. |
| Score dot | Per head, complete width-64 Q36 dot then `T(64,18)`; apply **another** `T(64,3)` for scale 1/sqrt(64)=1/8. This separate truncation is P, not inferred equivalent to a different multiply/rescale schedule. |
| Causal softmax | A4 per row, exactly active positions; padded lanes zero/structurally excluded. |
| Probability × V | Complete active-key reduction at Q36, single `T(64,18)`. Preserve head ordering when concatenating. |
| Attention output | Complete Q36 projection, `T(64,18)`, Q18 bias, then residual add to original block input. |
| Pre-MLP norm | A6 on attention residual. |
| MLP expansion | 768→3072 complete dots, `T(64,18)`, bias, A3 mixed GeLU. |
| MLP contraction | 3072→768 complete dots, `T(64,18)`, bias; add attention residual. |
| After 12 blocks | Final A6; last row times original tied embedding transpose, all 50,257 logits, complete dot then `T(64,18)`; no invented output bias. |
| Private feedback | Exact stable maximum tree, lower token index wins equal logits. Full vocabulary, public padding if required; never use approximate activation comparison for argmax. Retain IDs privately for next full prefix; release only D9's final token-ID result. |

**U:** actual frontend may differ in norm, bias, transpose, output head, attention
mask or tokenization. It must match this proposed graph explicitly or receive
another named profile; no model/source heuristic chooses a different graph.

### A7. Numerical acceptance, still not granted

Before A can become executable, freeze a public range/error record for every
node and eager transition, including HE errors, truncation outcomes, comparisons,
extensions, secret exponent candidates and positive norm/softmax inputs. Prove
the selected intermediate bounds; include all complete-dot and repeated-prefix
work. A clear-root fixture cannot issue an inherited model-range certificate.

Independent tests must include exact source-word encoding, coefficient floors,
both shares at all pivots, negative truncation, both extension endpoints, zero
and constant norm rows, active counts 1/3/128, causal padding, all full logits
and lowest-index ties. Reference quality must use the **same pinned source and
workload**, streamed FP32 plus accepted complete output positions; source drift,
missing/failed positions and nonfinite results are invalid, not ranked.
Checkpoint, workload, quality thresholds and actual accuracy are **U**; no
manufactured source or calibration identity is supplied by this specification.

## Attachment B — exact proposed HE/OT/key/error graph

### B1. Source-size ring and explicit ordered tuples

**R:** final App.E says OLT N=8192, t=2^64, q≈2^147 and q'≈2^49;
bOLE N=8192, q≈2^148, three approximately 45-bit plaintext primes. Those bit
counts do not identify primes. **P:** use these exact candidate tuples, ordered
as printed, each prime congruent to 1 mod 16384:

```text
OLT data q limbs:
  [562949952847873, 562949952798721, 562949952700417]
OLT special key-switch prime q':
   562949952274433
OLT key limbs = data limbs followed by q'

bOLE ciphertext q limbs, same list in every CRT context:
  [1125899906826241, 562949952847873, 562949952798721]
bOLE plaintext CRT limbs:
  [35184371613697, 35184371417089, 35184371138561]
```

Products are respectively:

```text
q_OLT = 178405960980747204108026560376635421240770561       # 147 bits
q_KEY = 100433627219585994566577152443730065352832756315331459366913
                                                               # 196 bits
q_bOLE = 356811922413223917036199361012173016050532353      # 148 bits
T_CRT = 43556140369735457451271451648667273920513           # 135 bits
```

Derivation: descending distinct primes p=1 mod 16384 below 2^49 (four), below
2^50 (one), and below 2^45 (three); fixed deterministic u64 primality witnesses
`[2,325,9375,28178,450775,9780504,1795265022]`. These are **new proposed
public integers**, not recovered SEAL outputs or author-selected parameters.
No auto-generated library chain, torus substitution or artifact tuple is allowed.
This calculation supplies neither security nor BFV correctness.

### B2. Owners, levels, evaluation keys and direction graph

**P:** one request/key epoch, two online parties P0/P1; no trusted dealer.

- Each party independently owns one OLT secret and one secret per each of the
  three bOLE plaintext contexts: **eight distinct secret keys total**. Secrets
  never travel. Public keys/evaluation keys travel only to the peer with an
  immutable parameter, source/request, owner, purpose and epoch binding.
- OLT public/evaluation material is generated over q_KEY; application
  ciphertexts are size two over q_OLT. Preserve the special-prime transition in
  key switching; q' is **not** a fourth free data level. bOLE has size-two
  ciphertexts over q_bOLE and no special prime, Galois key or relinearization.
  No ciphertext×ciphertext multiplication or bootstrap is needed in this graph.
  An OLT data public key is the **CRT restriction** of the raw key-level
  public key, not a BFV message-rescaling/mod-switch of that public key.
  Simultaneously retained data/key copies are charged separately.
- Fix OLT matrix windows `(kw,mw,nw)=(1,64,128)` with edge padding/cropping.
  `kw*mw*nw=8192`. Use D21's lhs row stride `nw*mw`, `V[beta,gamma]`,
  global output block `t=alpha*n_prime+gamma`, group/lane `divmod(t,mw)` and
  unrotated lane j. Inner block products accumulate **before** interleaving/H2A.
- Required automorphism indices for stride 64:
  **`[257,513,1025,2049,4097,8193]`** modulo 16384, i.e. N/h+1 for
  h=32,16,8,4,2,1. Both OLT owners issue keys returning `sigma_g(s)` to s;
  this is **12 logical Galois keys**, not rotations under a shared secret.
- Proposed special-prime gadget contract: radix 2^20, eight digits for the
  147-bit data coefficients; each key row is the **raw RLWE phase** pair
  `(-a*s + e + q'*(2^20)^j*sigma_g(s), a)` at q_KEY, j=0..7, with fresh
  uniform a and CBD(20) e. This is not BFV plaintext encryption with another
   implicit Delta multiplication. The exact proposed transition is B2.1 below.
   A backend's incompatible native decomposition does not silently substitute
   for it. Exact feasibility/API support and full noise/security qualification
   remain **U**.

### B2.1 Proposed special-prime switch: complete phase convention

Everything in this subsection is **P**, not an assertion about fhe.rs or the
evaluated paper. Write `q=q_OLT`, `p=q'`, `Q=q*p`, `B=2^20` and
`R_Z=Z[X]/(X^8192+1)`. Ciphertext order is `(c0,c1)` with phase
`c0+c1*s`; component zero is the `b` component. All products below are
negacyclic polynomial products; congruences and representative maps are
coefficientwise. No additional BFV Delta multiplies a switching-key row.

For an automorphism g from the frozen list:

1. Compute `u0=can_q(sigma_g(c0))`, `u1=can_q(sigma_g(c1))`, where
   `can_q` returns residues in `[0,q)`. The input's decrypting secret after this
   automorphism is `sigma_g(s)`. Do not automorph s again in the row product.
2. Decompose each canonical coefficient of u1 into exactly eight base-B digits:
   `d_j=floor(u1/B^j) mod B`, j=0..7. Thus `u1=sum_j B^j*d_j` over integers;
   j=0..6 digits are at most B-1, and j=7 at most127. Lift these digits and u0
   to Q by their same nonnegative integers, not a BFV modulus switch, centred
   extension or separately scaled plaintext encoding.
3. For row `(b_j,a_j)` satisfying
   `b_j+a_j*s = p*B^j*sigma_g(s)+e_j (mod Q)`, initialize
   `U0=p*u0 mod Q`, `U1=0`. In increasing j, accumulate
   `U0=(U0+d_j*b_j) mod Q`, `U1=(U1+d_j*a_j) mod Q`.
   No intermediate special-prime reduction or digit fusion. Original rows use
   independent fresh noise; later switches reuse those immutable key rows, so
   their induced errors are correlated and must not be counted as fresh draws.
4. Apply `DropSpecialPrime` independently to U0 then U1. Let
   `Z_i=center_Q(U_i)` in `[-(Q-1)/2,(Q-1)/2]`, and
   `D_i=floor((2*Z_i+p)/(2*p))`. Return `c'_i=can_q(D_i)`.
   Integer Z_i and odd p exclude exact half ties, including for negative inputs.
   This is nearest integer division by p on centred Q representatives, followed
   by reduction modulo q; it is not dropping the last residue or applying BFV
   plaintext decryption. Define `rho_i=p*D_i-Z_i`, so
   `|rho_i| <= (p-1)/2` coefficientwise. Reconstruct/qualify this exact mapping
   across all RNS limbs; a backend-specific different rounding rule is unselected.

For the integer lifted input phase `P_in=u0+u1*sigma_g(s)` and
`E_KS=sum_j d_j*e_j`, the key equations give
`Z_0+Z_1*s = p*P_in+E_KS+Q*L` for some integer polynomial L.
Consequently the exact output phase is

```text
c'_0+c'_1*s = P_in + E_drop (mod q)
E_drop = (E_KS + rho_0 + rho_1*s) / p.
```

The numerator is coefficientwise divisible by p by the preceding identity;
individual summands need not be. Thus E_drop is an integer polynomial, not a
rounded independent error sample. If incoming phase is `Delta*M+E_in (mod q)`,
the switched phase is `sigma_g(Delta*M+E_in)+E_drop (mod q)`, before any
subsequent message-representative normalization. Such normalization and the
trace/merge inverse-power-of-two maps require their own carry accounting.

A conditional deterministic **single-switch** bound, using only the proposed
CBD support and ternary-secret support, is

```text
||E_KS||_inf <= 8192*20*(7*(2^20-1)+127)
||E_drop||_inf <=
  [8192*20*(7*(2^20-1)+127) + (p-1)/2*(1+||s||_1)] / p,
||s||_1 <= 8192.
```

This triangle bound is neither a complete OLT noise bound nor a circuit-privacy
or security proof. It does not assume independence between digits, rounded
components and s. Q-level row generation/expansion, raw/prepared owners, canonical
CRT extension, full-Q centred reconstruction and both rounded outputs must be
qualified/priced before factories. None is supplied by naming this transition.

For `Q=Q0+Q1`, `V=V0+V1`, the graph is:

| Term | Owner/direction | Output handling |
| --- | --- | --- |
| Q0 V0, Q1 V1 | Local on original party, ring arithmetic | Retain original source/order; no wire or fictitious HE charge. |
| Q0 V1 | Q0 encrypted under P0's OLT key; P1 evaluates V1 | Fresh coefficient-H2A to P0; both parties retain resulting shares. |
| Q1 V0 | Q1 encrypted under P1's OLT key; P0 evaluates V0 | Independent coefficient-H2A to P1. |
| Final product | Add the four terms in listed order | Then A's complete-dot truncation; do not truncate inner HE blocks. |

Structurally absent terms for model-only P1 operands can be skipped only when
the source/operand type proves zero P0 backing, not by inspecting secrets.
General Hadamard M uses two opposite bOLE cross terms plus local terms. Square
uses one cross term doubled; **its error is doubled too**. Cubic/power reuse may
borrow the same already encrypted immutable operand, but each returned result
gets fresh output masking. No reconstruction of an operand by a coordinator.

**P SIMD ordering:** for each plaintext tj, choose the first integer a>=2 for
which `zeta = a^((tj-1)/16384) mod tj` satisfies `zeta^8192=-1 mod tj`.
Lane i, i=0..8191, is evaluation at `zeta^(2*i+1)`; encoding is the inverse
transform in that exact order. Backend row/bit-reversal permutations require an
explicit qualified adapter, never a silent slot-order change. Internal NTT
choices may differ only while preserving these coefficient and lane meanings.

### B3. Two H2A paths, with different domains and rounding

**Coefficient OLT H2A — R algorithm family / P concrete contract:**

1. Encode message coefficients using `Delta=floor(q_OLT/2^64)`, not `Lift`.
2. Evaluate the exact admitted packing/ct-plaintext/add/automorphism graph under
   its original owner. Preserve all coefficients until conversion is complete.
3. Evaluating peer samples **every coefficient** of r uniformly in R_q_OLT,
   not just useful ParseMat lanes, and independently generates a fresh public-key
   encryption of zero. Send `ct + Enc_pk(0) + (r,0)` to the key owner.
   A backend's trivial-zero ciphertext is suitable only for structural padding,
   never this fresh rerandomization step.
4. Key owner outputs `Down_q(phase)`; masking peer outputs `-Down_q(r)` mod 2^64,
   with `Down_q(a)=floor((2*2^64*(a mod q)+q)/(2*q)) mod 2^64`.
   Odd q excludes exact half ties. Only then perform D21 ParseMat/cropping.

Retain the complete phase equation
`phase = Delta*M + E_total + E_fresh_zero + r (mod q)`.
`E_total` includes encoding/scaling discrepancies and all evaluation/key-switch
errors, not only encryption noise. Shared conversion must reconstruct the full
147-bit coefficient: **never independently Down each RNS limb**. Existing u128
native conversion refuses this domain. A preadmitted generic fixed-wide/CRT
owner is required; data-phase Down needs a numerator up to 212 bits. No smaller
source q/t or truncation of high limbs is a substitute.

**Exact scaling-carry obligation (P):** put `t=2^64`, `Delta=floor(q/t)` and
`r_q=q-t*Delta`. For canonical input-message polynomials x,y, form their integer
negacyclic product `H=x*y=M+t*K`, where M has coefficient residues in `[0,t)`
and K is the resulting signed integer quotient polynomial. Then

```text
Delta*H = Delta*M + q*K - r_q*K
Delta*H mod q = Delta*M - r_q*K mod q.
```

If the input ciphertext phase is `Delta*x+E_enc`, the ct-plaintext product phase
is therefore `Delta*M + E_enc*y - r_q*K (mod q)`, before additional operations.
Additions and inner-block accumulation produce further quotient/carry terms
when their message sums are reduced modulo t; retain them rather than resetting
the error at block boundaries. For a scalar check `x=y=t-1`, `M=1` and `K=t-2`:
the scaling discrepancy is exactly `-(t-2)*r_q`, not zero merely because the
desired ring product is1. This identity is not a new executed experiment.

Full-width uniformly split shares make these H/K terms potentially large.
Neither small reconstructed activations nor the single-switch bound constrains
them. Each actual odd-modulus inverse trace/interleave operation must propagate
the entire modular phase, including its representative carries; do not divide
an error bound by the trace factor without proving that relation. A complete
rounded-H2A error guarantee under this exact tuple is **UNKNOWN** and may fail.
No ±1 theorem, larger modulus or revised mask follows from this proposal.

**SIMD bOLE H2A — R algorithm family / P concrete contract:**

1. Compute **global** `Lift_T(x)=floor((T*(x mod 2^64)+2^63)/2^64) mod T`.
   T is T_CRT, greater than 2^128; ties are upward. Reduce this value into
   each plaintext CRT prime; pack each via its own SIMD encoder. Per-prime
   independent Lift is wrong. The peer packs canonical y reduced in each prime.
2. Each context performs ct×plaintext to obtain `ct_product`. Evaluating peer
   draws a uniform full-coefficient plaintext polynomial r_j in that context,
   independently across CRT limbs (equivalent to uniform R_T), and returns
   **`ct_return = ct_product + Enc_pk((-r_j) mod t_j) + (F_j,0)`** modulo q_bOLE.
   Ciphertext order is `(c0,c1)` and decryption phase is `c0+c1*s`; F_j goes
   into component zero only, coefficientwise. `Enc_pk` is independently fresh
   ordinary public-key encryption under the same context and decrypting owner,
   never trivial zero or a replacement for ct_product. The key owner decrypts
   the returned ciphertext; evaluating peer retains r_j as its plaintext share.
3. Each party SIMD-decodes **each context first** in the declared lane order;
   then CRT-reconstructs its **canonical full T share per lane** and applies
   `Down_T` once. Never apply
   per-limb Down or a per-limb arithmetic truncation. Full numerator is at most
    200 bits. Resulting ring64 shares enter A's product/truncation graph.

**Returned-phase equation (P):** for context j, let `q_b=q_bOLE`,
`Delta_j=floor(q_b/t_j)`, `n_j=can_tj(-r_j)`, and let `Psi_product,j` be the
actual evaluated phase, including its accumulated errors and scaling carries.
For public key `(b_pk,a_pk)` with `b_pk+a_pk*s=e_pk (mod q_b)`, fresh encryption
uses `(b_pk*u+e0+Delta_j*n_j, a_pk*u+e1)` and has noise
`E_fresh,j=e_pk*u+e0+e1*s`. Hence

```text
Psi_return,j = Psi_product,j + Delta_j*n_j + E_fresh,j + F_j (mod q_b).
```

The actual BFV decryption/rounding contract must prove that this returns the
intended plaintext product minus r_j modulo t_j. It cannot omit the evaluated
product, its scale/carry error, fresh encryption noise or F_j. Ordinary BFV
plaintext decoding occurs within each context before SIMD decoding. Both the
key-owner decoded share and the evaluating peer's retained mask polynomial are
SIMD-decoded in B2's exact lane order; each party then CRT-reconstructs its own
canonical T share per lane and applies Down_T once. A decryption error is not
an allowed additional ±1 conversion outcome.

The ideal noiseless Lift/multiply/uniform-mask/Down arithmetic has error support
`{-1,0,+1}` per cross term. This does **not** prove backend decryption succeeds,
coefficient-H2A has the same universal bound, or composed error independence.

### B4. Samplers, freshness, noise propagation and security targets

All entries here are **P**, not recovered final-paper samplers:

| Purpose | Exact proposed law |
| --- | --- |
| Secret and public-encryption ephemeral u | Independent iid uniform coefficients in `{-1,0,1}`. Two-bit rejection mapping 0→-1, 1→0, 2→1, 3→retry. No fixed Hamming weight or all-zero qualification secret. |
| Ordinary encryption/key-switch errors | Independent coefficient `CBD(20) = sum_(i=1..20)(B_i-C_i)`, fair independent bits; support [-20,20], variance 10. This explicitly proposed law is not source SEAL sigma 3.19 and is not inherited from a backend default. |
| Coefficient masks | Uniform canonical coefficients of the complete q_OLT CRT ring. Rejection per limb, exact unbiased conditional acceptance, including discarded/padded lanes. |
| SIMD plaintext masks | Uniform all-coefficient R_tj masks in each CRT context; no decoded-lane-only or small ring64 mask. |
| SIMD flooding | Ordinary fresh encryption of negative plaintext mask, plus independent F in its b component only: F is a uniform 98-bit word minus 2^97, interval **[-2^97,2^97)**. Every coefficient, every return, every context. This makes an endpoint choice explicit; printed `B'=2^98` does not identify this interval. |
| Rejection/entropy failure | At most 128 rejection draws per coefficient; then typed whole-attempt failure, burn and erase. Never biased modulo reduction or reused masks. Bound all RNG draws/work before the sampler factory. |

Freshness domains include protocol/profile, request/key epoch, gate/use, owner,
direction, CRT context, batch, coefficient and purpose. Original encryption,
rerandomizing zero, coefficient mask, plaintext mask, flood, OT extension and
OT application masks have distinct domains. An immutable input-ciphertext alias
for cubic reuse does **not** license reused return masks or one-time correlations.
Secure fallible entropy and fork/thread handling must cover backend-internal
expansions too; D28 has not adopted a replacement RNG seam for this backend.

**Proposed security targets:** 128-bit classical computational security for the
complete RLWE/key-switch/OT graph, and total transcript statistical distance
at most 2^-40 for the explicitly bounded qualification/request being claimed.
Quantum estimates must be separately reported; no quantum-bit claim is selected.
Semi-honest two-party privacy is the target, not malicious correctness or a
verifiable-inference proof. Authenticated channels/replay protection do not
upgrade this threat model.

**Blocking analysis, with a concrete calculation obligation:**

- Propagate `E_enc=e_pk*u+e0+e1*s`, ct-plaintext convolution and inner-block
  accumulation; then each actual trace/merge/key-switch/rounding operation.
  Include Delta-vs-nearest encoding and modular representative/carry effects.
  Do not assume inverse trace division divides every newly introduced error.
- For each bOLE CRT component prove decryption under the actual
  `floor(q_bOLE/tj)` gap after fresh-zero error **and** F. For coefficient H2A
  prove its exact rounded-share error under actual admitted phases/masks.
- Full-width uniformly random **shares** are HE plaintext operands. A bound on
  the clear Q18 value does not bound their coefficient magnitude. Derive a
  worst-case or explicitly justified probabilistic bound for these actual
  samplers/graphs, including retained-input reuse and both directions.
- For uniform flooding of width W=2^98, a shift bound E yields per-coordinate
  variation bounded by `min(1,abs(E)/W)`. Sum over **all released coefficients,
  CRT contexts, gates and repeated requests in the claimed transcript**, adding
  exceptional-event probabilities. A per-coordinate 40-bit claim is not a
  40-bit transcript claim. Conditioning on visible abort also needs analysis.
- Qualify computational security at the **196-bit OLT key modulus** as well as
  each data modulus, with these exact secret/error/key distributions, public
  evaluation-key relations and key reuse. Publish estimator identity/inputs and
  independent review; a disabled library check or nominal prime width is not it.

**U:** no such combined proof/estimator result or mature-dependency/security
qualification is supplied. The proposed q/flood pair may fail these targets.
The concrete next resolution is this finite propagation/transcript analysis,
not a toy all-zero test or a blanket one-bit theorem. If it fails, stop: no
automatic extra primes, wider flooding, error tolerance, altered masks or
weaker security target. Any revised tuple/graph is a separately named decision.

### B5. OT graph and transcript ownership

**R:** final paper uses distributed Ferret COT, 1-of-16 OT on two- and six-bit
messages; the wide segment MSB optimization is 50 bits. **P:** separate COT
streams in both directions, exact protected wrap/MSB/Boolean operations feeding
A2's explicitly selected circuits; batch threshold messages only when their
operand/source/use binding matches. Fresh application correlations never recur.
Do not turn 1-of-16 into 16-bit messages or import a trusted/local OT dealer.

**U:** exact Ferret/base-OT implementation commit, algorithm/parameter set,
dependency/license, malicious-versus-semi-honest mode, seed/expansion layout,
setup transcript and finite setup/storage/work quote. `fhe.rs` does not supply
that completed OT stack. Proposed resolution: pin and independently qualify a
generic distributed Ferret/base-OT implementation and fixed A2 circuits before
any method integration; no unreviewed FFI/native-Rust substitute is adopted.

Every frame binds profile, original parameter order, key epoch, source/request,
purpose/direction, logical gate, sequence and exact payload length. Admission
and epoch-use consumption precede entropy/encoding. Both parties retain their
original keys/source and continuous reservations through late frame/output
retirement. No partially accepted token, coefficient, branch result or logit
escapes. Any entropy, transport, backend, decryption or validation failure burns
the whole attempt and cancels/drains sibling work before final refund; callback
panics/errors cannot replace the original failure or run under the owner lock.
These are requirements to qualify, not claims that this two-party route exists.

## C. Future native-Rust primitive qualification — NOT SELECTABLE

**NOT SELECTABLE at this revision.** The following scope is a retained future
proposal, not a bounded implementation grant. Frontend/source binding, exact
OT circuits/messages/setup, work units, backend constructors, full error/security
and phase-owner admission remain incomplete. The source-only next action in D
cannot open this gate or establish any of those unknowns by approval.

### C1. Proposed scope and fixed fixtures

Name: **`bumblebee.odd-rns-primitives.fhers-98a087f.v1`**. Dependency pin is the
full fhe.rs commit above, generic primitives only, with PLLM's own packing,
conversion, role/key custody and admission. **No adoption is proposed by merely
selecting this experiment. CKKS refresh/bootstrap is not an OLT/bOLE prerequisite.**

1. Static/API and parameter-security gate first. Verify N=8192, exact ordered
   tuples, ring64 plaintext, raw components, chosen Galois/decomposition and
   secure fallible samplers are actually supported. Reuse/extend shared wide/CRT
   and coefficient owners; do not fork a second paper-specific implementation.
2. After a **separate implementation/execution approval**, test OLT in both
   directions on shape **(3,65,129)** with fixed window **(1,64,128)**: six
   useful output blocks, last inner block and columns partial; one interleave
   group with eight padded inputs traced to stride 64. Six input ciphertexts,
   twelve ct-plaintext products per direction; charge all padded trace work
   (31 automorphisms per direction for this straightforward schedule), not only
   useful cells. Compare every parsed cell against independent exact integer
   matrix composition, including signed inputs, both share components and tails.
3. Test bOLE length **8195 = N+3**, both cross directions and one square path;
   three CRT contexts per direction/batch. Check global Lift/CRT/Down, final
   partial batch, mask freshness and observed algebraic error separately from
   decryption/circuit-privacy claims. All full-ring unused coefficients still
   participate in the masking contract. No model or tokenizer needed.
4. Independently exercise A's scalar contracts, softmax rows of lengths 1 and 3
   and its 768-lane norm on synthetic data, but only after numerical/protocol
   adoption: pivots ±2.7,
   cutoff -14, extension boundaries, zero/constant norm and exact stable ties.
   This is not GPT2 quality, checkpoint provenance or full-prefix qualification.
5. Wrong parameter order, owner/key/context/epoch, noncanonical payload, replay,
   short tail, failed entropy, callback error/panic and sibling cancellation
   must fail closed. Retain a previous successful frame/output while failing a
   later one; prove erasure before final original-counter refund.

### C2. Proposed admission ceilings and named live owners

These are **new proposal limits**, not measured costs, spare allowance, a cap
increase or inherited D4 approval. Freeze one CPU execution thread, one request
epoch, no model, maximum 512 MiB combined logical-owner envelope; setup deadline
300 s and evaluation deadline 300 s; at most 2^30 modular arithmetic work units,
2^20 COT choices and 2^18 protected scalar applications. Resolve the cost-unit
definition/circuit counts before acceptance; exhaustion rejects, never renews a
lease or changes parameters. Require at least 4 GiB additional host headroom and
ordinary disk admission; neither is a physical-peak proof. Build/dependency
resolution needs its own later bounded approval; it is not authorized here.

| Named live owner/transition | Concrete accounting requirement |
| --- | --- |
| Keys | Eight original secrets plus canonical/NTT copies actually retained; public keys for both owners/all contexts; data/key-level public copies if both exist. Six Galois indices × two owners × eight gadget rows × two polynomials × 8192 × four key limbs × eight bytes = **50,331,648 raw Galois-key payload bytes**, before wrappers/preparation. |
| Prepared keys / setup | Charge raw+prepared overlap, decompositions, NTT roots/tables, generation seeds/entropy, samplers and all scratch. Unknown prepared/dependency costs are blocking, not zero or credits against the raw-key number. |
| OLT | Both party-local input matrices, original encoded polynomials and immutable cached ciphertexts; ct-plaintext products, accumulated group, eight-input/padded trace scratch, output group and H2A fresh-zero/mask/decryption/rounding overlap. One data ciphertext's plain u64-limb payload is `2*8192*3*8 = 393216` bytes; actual backend representation may be larger. |
| bOLE | Original ring64 shares; full T Lift/CRT canonical values, three SIMD polynomial buffers, all live ciphertext contexts, masks, flood and decoded/rounded output shares. Global wide conversion workspace remains live beside source/CRT limbs; no per-limb Down or guard-after-copy. |
| Transport | Proposed 128-byte application-header allowance and uncompressed payload; one ciphertext/context per data frame, at most four live frames of `393216+128` bytes. Setup key transfer uses payload-bounded chunks, retaining the complete original key owner. The exact authenticated generic header/commitment schema remains to freeze and price, including endpoint/parser/task/result/queue controls, socket/authentication buffers and cancellation overlap. This allowance is not a wire-adoption decision or evidence all controls fit it. No unpriced transcript history. Cached input ciphertexts are additional owners, not this queue's capacity. |
| Work/retirement | Count both directions, all CRT components, all inner blocks, partial/padded groups, input reuse lifetimes, entropy expansion, encode/decode/CRT/rounding, OT setup/application, all discarded arithmetic candidates, frame validation/hash and erasure. No one-kernel or one-map-only quote. |
| Shared counter | Quote complete phase maxima before context/key/buffer factories on each originating party's original Budget; retain immutable original source/input guards through output/frame drop. The combined ceiling sums both parties, not a fabricated cross-host shared pointer. Include each actual capacity once and every simultaneously live alias/backing/transition, not a detached counter or refund/reacquire. |

**U:** full constructor/phase inventory, backend allocation contracts and private
dependency callframes, allocator/RSS terms. Known native/consumer wrapper,
argument/Result, callback, iterator, cursor and packaging slots must be named and
priced before execution; the table is not a complete static proof. Record
unpriced physical/dependency terms explicitly and keep their execution gate
closed. Neither 512 MiB nor an observed ceiling-only check proves a peak bound.

## D. Candidate decisions and bounded next action

Approval of D37 or this document does **not** select any wording below.
The A/H alternatives are retained design choices, not the current adoption
request. The next question is only B-S/ANALYZE-1; Q remains closed.

- **B-S/A — numerical specification only:** “Select Attachment A's named
  derived interpretation, including floor source-word encoding, share-level T,
  mixed cast/restore order, ±2.7 approximate tails with safe argument, six-square
  exp, explicit reciprocal/two and rsqrt/eight iterations, mean-square epsilon
  two with secret minimum-two floor, complete pre-LN schedule and exact final
  argmax. This does not assert author/evaluated identity, qualify error/ranges,
  select source/checkpoint/calibration, authorize implementation/execution or
  promote SDK availability.” Rejecting any named choice keeps A unadopted;
  specify a separately reviewed alternative rather than inheriting defaults.
- **B-S/H — cryptographic candidate for analysis only:** “Select Attachment B's
  N=8192 ordered tuples, two-owner/eight-secret graph, special-prime/gadget
  transition, complete-coefficient H2A, global-CRT bOLE, exact proposed samplers,
  distributed Ferret requirement and 128-classical/40-statistical targets as the
  candidate to analyze. No security level, failure bound, OT dependency or
  executable backend is certified or adopted by this selection.”
- **B-S/Q — NOT SELECTABLE; incomplete future qualification packet:** Section C
  records candidate fixtures and ceilings only. Do not request conditional or
  blanket implementation permission while its frontend, OT/circuit/message/setup,
  work-unit and complete owner/API/error/security fields remain unresolved.
  A completed later packet must separately select implementation, dependency/build
  and actual key/encrypted execution scopes. No CKKS-refresh prerequisite,
  automatic fallback, full GPT2 run or backend adoption is implied.
- **B-S/author — retain the existing author-manifest gate:** “Do not select the
  derived arithmetic/HE pair. Continue seeking the exact final-paper evaluated
  numerical/frontend/HE manifest; preserve D21's independent algebra scope.”

### Proposed next action: B-S/ANALYZE-1, source-only and finite

**Narrow approval — October 24, 2026:** The user approved only
B-S/ANALYZE-1 as bounded below: Astra source-only analysis of twelve fixed
symbolic cases, at most sixteen backend API rows and eight OT rows, within
sixty minutes and all existing limits. This does not select B-S/A, B-S/H or
B-S/Q, adopt a backend, or authorize implementation, source-code edits,
execution, network access or keys. All other proposal and adoption gates remain
unchanged.

**Analysis complete:** the [source-only design delta](../research/bumblebee-coefficient-phase-analysis.md)
records the twelve-case result: the proposed canonical Delta/full-share product
and H2A graph's universal one-bit claim fails at a noiseless endpoint. This is
not a claim against the paper or centered BFV. B-S/Q remains **NOT SELECTABLE**;
no encoding, tuple or profile change, execution or adoption is authorized.

This action commissions **Astra planning/analysis only**, not Attachment A/B
adoption or B-S/Q. Existing numbers remain proposed; no tuple, sampler, profile,
checkpoint or dependency may change to make an inequality pass. No implementation
worker, source edits, executable experiment, sweep, test, build, dependency
resolution, network, entropy, keygen, native HE call or model execution.

**Fixed deliverable:** one concise design delta with the following twelve
symbolic cases; each ends in an exact identity, a justified bound with its stated
assumptions, or a named blocking unknown/counterexample. Do not keep expanding the
case set to obtain a favorable result.

| Case | Required source-only result |
| --- | --- |
| 1 | Raw public-key encryption phase and support-based convolution error, including retained key/ephemeral correlations. |
| 2 | Coefficient ct-plaintext full-share product and `-r_q*K` carry, including x=y=t-1. |
| 3 | Complete inner-block accumulation and further message-reduction carries, without per-block reset. |
| 4 | B2.1 automorphism/digit/row phase identity and single-switch rounding/divisibility bound; identify any unavailable API. |
| 5 | D21 eight-input interleave/trace to stride64: propagate full phases through the31 automorphisms, odd inverse factors and representative carries; no assumed error division or independence. |
| 6 | Final coefficient-H2A mask/fresh-zero/Down relation and a conditional reconstruction bound or explicit unresolved/failing condition. |
| 7 | Global T Lift and canonical y multiplication, preserving the full-T conversion error support. |
| 8 | bOLE return phase from B3, negative-mask encoding/carries and actual BFV decryption-gap inequality for each of three contexts. |
| 9 | Per-context plaintext/SIMD decoding, both parties' full CRT reconstruction and single Down; partial batch padding remains masked. |
| 10 | Uniform98-bit flooding shift bound with whole fixed-fixture coefficient/context/direction counts, reused-input/key correlations and exceptional-event terms; distinguish an upper bound too loose to prove security from a demonstrated failure. |
| 11 | Proposed bounded-rejection samplers: number of original/fresh/key-row draws and finite exhaustion-union obligation; unknown backend expansion remains blocking. |
| 12 | Security/circuit-privacy boundary: exact public key relations and full196-bit key modulus, known assumptions versus missing estimator/OT proof inputs. No estimator execution or claimed128/40-bit result. |

Case10 uses only C1's one OLT fixture in two directions and one bOLE fixture
of8195 lanes with two cross directions plus one separately specified square
cross term. All three bOLE paths have two batches and three contexts:18 returned
full rings,147,456 flooded coefficients. The square doubles its error after
share conversion; it does not silently add a second cross-term return. Charge
padding coefficients too. Repeated requests/full-model counts are not inferred
from this one-fixture tally. OLT H2A is a different masking mechanism, not an
unlisted bOLE flooding event. If source analysis shows additional returns are
necessary, report the mismatch and stop that bound; do not reuse this tally.

**API/circuit attachment:** at most16 static API-edge rows against the pinned
fhe.rs source and existing shared contracts: parameter construction; ring64
plaintext representation; canonical/centred CRT access; exact component order;
raw phase keys; gadget digits; automorphism direction; p-scaled c0 accumulation;
DropSpecialPrime; fresh encryption; explicit samplers/flood injection; SIMD lane
adapter; complete H2A conversions; original backing/temporary access; error/burn
propagation; shared transport/admission binding. Record exact existing symbol and
source locator, or UNKNOWN. A similarly bounded eight-function OT inventory
covers wrap, MSB, exact comparison, Boolean-to-arithmetic, mux, one-hot exponent
selection, truncation and positive extension: known functional law versus missing
concrete circuit/base-OT/Ferret pin, message count, setup and work quote. Do not
invent implementation counts from functionality names. This inventory may close
with explicit unknowns; it does not claim a completed OT attachment.

**Budgets and stop rules (proposed):** one Astra reviewer, one pass, at most60
minutes; existing local documents/locked paper (at most18 pages) and at most32
existing source/manifest files, at most16 MiB source text read. At most12 symbolic
cases,16 API rows and8 OT rows; one new/revised Markdown design delta at most
128 KiB. No copied source tree, evidence archive or executable driver. These are
analysis/output limits, not HE memory/work grants. Missing local source, ambiguous
sign/rounding, exhausted budget or unavailable allocation/API evidence produces
a bounded unresolved finding; no fetch, invented proof or extension of scope.

**Completion:** deliver case outcomes, exact unchanged parameter/key identity,
fixed-fixture count ledger and ranked missing proofs. Keep B-S/Q closed unless
a separately reviewed later packet actually supplies all required fields.
The present single-switch triangle bound, candidate prime list and proposed
flood width establish neither a safe whole-graph noise envelope nor transcript
security. An infeasible candidate is an acceptable analysis result.

**Exact bounded question:** Approve **B-S/ANALYZE-1**, the source-only Astra
propagation/API analysis above, or defer? This selects an analysis task only;
it adopts neither the whole arithmetic/HE profile nor a backend, and authorizes
no implementation, build, test, native/key execution or security claim.

### Remaining adoption gates

| Blocking gate | Strongest concrete next resolution; not a default |
| --- | --- |
| Author/source/frontend identity | Obtain a hash-bound evaluated manifest, or explicitly select a different pinned derived checkpoint/frontend with the A1 mapping. Keep the original identity claim unresolved. |
| A's proposed numeric departures and quality | Review/accept each named departure; independently bound all stage errors/ranges and run an approved same-source complete-output quality workload later. Do not infer acceptance from the final-paper label. |
| Exact HE API and source-size arithmetic | Prove the selected pinned library can construct ring64/plaintext and the exact Galois/special-prime graph; extend shared N=8192/wide-CRT owners under their own reviewed admission. No q/N reduction to fit existing code. |
| Full error/circuit privacy | Complete B4's actual-share noise, rounding, flooding and transcript-union analysis. If q/flood/error target conflicts, return a named revised decision, not silent parameter growth. |
| Security and OT | Independent full-modulus estimator/review, secure fallible/fork-safe sampler seam, exact distributed base-OT/Ferret pin and costs; no local dealer or inherited library security label. |
| Logical and physical owner admission | Complete phase-by-phase keys/setup/copies/frames/consumers/cleanup quote on original owners; separately establish or exclude dependency/physical terms without declaring them zero. |
| Complete method | Original source-bound decoder, fixed-count private feedback, cancellation/transport, ordinary Component/Pipeline/Experiment/SDK and same-workload evidence. No current component, scalar oracle or qualification fixture supplies this route. |

This document changes no implementation, numeric default, RNG, dependency,
capability, model/source pin, runtime cap, state policy or disclosure rule.
