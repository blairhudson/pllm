# THOR multi-block Algorithm 2 indexing audit

Status: **paper-derived indexing correction approved and validated as bounded clear
algebra** (D25 in `paper/methods/DECISIONS.md`). Original read-only source audit,
22 October 2026; implementation and validation follow below. No author erratum claimed.

## Pinned source and existing gate

`papers/thor.pdf` SHA-256 independently verified as
`ddb57004b56ee8e00beffc4f563fd0921242753fda6c704065d4900b792439cd`,
matching `paper/methods/source-lock.v1.json`. Scope follows
`paper/methods/encrypted-computation-contracts.md` §5.2.

Existing owners: `crates/pllm-core/src/slot_matrix_layout.rs` (`Layout`,
`lower_mm`, `optimized_lower_schedule`, `optimized_lower_mm`) and
`slot_matrix_layout_tests.rs::printed_multiblock_rotation_has_explicit_counterexample`.
The basic `lower_mm` already implements the required diagonal contraction.
At the time of the audit, the optimized implementation admitted only `m=c`; its
multi-block gate was sound for the printed algorithm. The audit itself changed
no implementation files; the later approved correction is described below.

## Printed facts

- §2.1, p.3: `Rot(v;k)[x] = v[(x+k) mod s]` (left rotation).
- Definition 3.1, p.3: `L_d(A)[t] = A[(d+t) mod m,t]` for `A` of shape
  `m*n`, `m<=n`. Corollary 3.8, p.4, requires `m | n`.
- §4.1/Eq. (7), p.4: interlace heads, then concatenate consecutive diagonals;
  `s=c*n*H`, `c | m`. Figure 2, p.5, illustrates `m=n=4,H=2,c=2`.
- §4.3.1/Eq. (10), p.7, requires output diagonal `c*j+r` to sum
  `Rot(L_(c*j+r-l mod m)(A);l) * L_l(B)` over `0<=l<n`.
- Algorithm 2, p.8, lines 7–8 form products from
  `Rot(A_j;(-n*(l mod c)+l)*H)` and replicated `B_l`.
  Lines 16–17 select only product frames `j-1,j`; no `floor(l/c)` appears.
  Line 18 rotates the second partial sum by `-n*H`.
- §4.3.2, pp.7–8, derives that delayed rotation using replication's invariance
  under rotation by `n*H`. Figure 5 uses `m=c=2,n=4,H=1`: only one frame,
  so it cannot distinguish any missing frame offset.

The p.7 mask prose describes selected source diagonal ranges. Destination-lane
indicator formulas below are derived from Eq. (10), not quoted mask formulas.
Likewise, the modular frame correction below is a proposal, not printed text.

## Exact coordinate map and proof

Declare positive integers `c | m | n`, `H>=1`, `M=m/c`, `s=c*n*H`.
All arithmetic, extents and work must fit existing checked caller limits.
These are logical full frames, not physical CKKS slots with an unused tail.

Frame `j`, local diagonal `r`, time `t`, head `h` have local address
`x=(r*n+t)*H+h`, global address `j*s+x`, and matrix coordinate
`(h,(c*j+r+t) mod m,t)`. For `B_l`, every diagonal-sized subslot contains
`B[h,(l+t) mod n,t]`. This recovers the existing lower-layout map directly.

Write `l=q*c+b`, `0<=b<c`, and `e=1 if r<b else 0`. Eq. (10) requires

```
d = (c*j+r-l) mod m
source frame = (j-q-e) mod M
source local diagonal = (r-b) mod c
source time = (t+l) mod n
source head = h
```

Proof: `c*j+r-l = c*(j-q-e) + (r-b+e*c)`, with the latter term in
`[0,c)`. Taking modulo `m=c*M` gives exactly that frame and local diagonal.
Substituting source time into Definition 3.1 gives matrix row
`(c*j+r+t) mod m`, since `m | n`, and column `(t+l) mod n`.
Multiplying by `B[h,(t+l) mod n,t]` and summing over `l` is therefore the
ordinary row-major contraction, with the inner index visited exactly once.

For the optimized schedule, keep the printed rotation
`delta=(-n*b+l)*H`. If `t<n-l`, read product lane `y=x`. Otherwise read
`y=(x-n*H) mod s`, later moved to `x` by the printed final rotation.
In both cases `(y+delta) mod s` has local diagonal `(r-b) mod c`, time
`(t+l) mod n`, and head `h`: in the second case subtracting `n*H` cancels
the time-wrap carry. `B_l[y]=B_l[x]` because `B_l` repeats every `n*H`.

Thus the same products and masks suffice; only frame routing changes.
Define destination masks at `(r,t,h)`:

```
mu0 = [r<b]  * [t<n-l]     mu1 = [r>=b] * [t<n-l]
mu2 = [r<b]  * [t>=n-l]    mu3 = [r>=b] * [t>=n-l]
```

The second pair is applied in product coordinates as `Rot(mu2;n*H)` and
`Rot(mu3;n*H)`. These four applied masks partition the frame, so Algorithm 2
line 14's subtraction remains valid. Heads never mix: every shift is a
multiple of `H`.

## Smallest proposed change

Retain line 7 and product labels indexed by original input frame. In lines
16–17, replace `j-1` by `(j-floor(l/c)-1) mod M` and `j` by
`(j-floor(l/c)) mod M`, separately for each summand `l`. Line 18 and `l=0`
remain unchanged. Equivalently, relabel line 7's input as
`A_((j-floor(l/c)) mod M)` and keep lines 16–17; **do not apply both fixes**.

For `M=1` the correction vanishes, preserving the admitted single-frame case.
For `q` larger than `M`, modular reduction is essential (rectangular `n>m`).
This is index-only, with no extra rotations, products, or new primitive.
Exact non-complexified clear schedule counts become `M*n` tensor products,
`3*M*(n-1)` plaintext products, `M*(n-1)` input rotations, and
`M*[n>1]` delayed output rotations. Existing declared lazy boundaries scale
per frame; these counts do not qualify an HE scale/level DAG.

## Numeric counterexamples

Smallest positive multi-block dimensions: `m=n=2,c=H=1`. Let
`A=[[0,1],[2,3]]`, `B=[[0,0],[1,0]]`. At output `(0,0)`, only `l=1`
contributes. Printed selection uses frame 0 and reads `L_0(A)[1]=3`;
Eq. (10) requires frame 1 and `L_1(A)[1]=1`. Printed result is 3, dense
result is 1.

Existing regression independently confirmed: `m=n=4,c=2,H=1`,
`A[i,k]=4*i+k`, `B[2,0]=1`, all other `B` entries zero. At output `(0,0)`,
`l=2,q=1,b=0`: printed frame 0 yields `L_0(A)[2]=10`, but frame 1 yields
`L_2(A)[2]=2`, the dense answer. This also demonstrates failure with more
than one diagonal in each frame.

## Independently executable oracle

Python standard library only; exact integers, no PLLM imports or dependencies.
The candidate executes rotations, four masks, frame routing and delayed sums.
The reference uses only row-major multiplication. Exhaustive coordinate checks
use uniquely labeled entries, proving each selected operand address separately
within the enumerated geometry; numeric tests check the complete accumulation.
Run this fenced block with Python 3.

```python
def rot(v, k):
    return [v[(x + k) % len(v)] for x in range(len(v))]

def candidate(A, B, m, n, c, H, corrected=True):
    M, s = m // c, c * n * H
    frames = [[A[h][(c*j+r+t) % m][t]
               for r in range(c) for t in range(n) for h in range(H)]
              for j in range(M)]
    products = []
    for j in range(M):
        row = []
        for l in range(n):
            av = rot(frames[j], (-n*(l % c)+l)*H)
            bv = [B[h][(t+l) % n][t]
                  for r in range(c) for t in range(n) for h in range(H)]
            row.append([a*b for a, b in zip(av, bv)])
        products.append(row)
    out = [[[0]*n for _ in range(m)] for _ in range(H)]
    for j in range(M):
        first, second = products[j][0][:], [0]*s
        for l in range(1, n):
            q, b = divmod(l, c)
            if not corrected:
                q = 0
            masks = [[int((r < b) == (v % 2 == 0) and
                          (t >= n-l) == (v >= 2))
                      for r in range(c) for t in range(n) for h in range(H)]
                     for v in range(4)]
            for v in range(4):
                mask = rot(masks[v], n*H) if v >= 2 else masks[v]
                p = products[(j-q-int(v % 2 == 0)) % M][l]
                acc = second if v >= 2 else first
                for x in range(s):
                    acc[x] += p[x]*mask[x]
        result = [a+b for a, b in zip(first, rot(second, -n*H))]
        for r in range(c):
            for t in range(n):
                for h in range(H):
                    out[h][(c*j+r+t) % m][t] = result[(r*n+t)*H+h]
    return out

def dense(A, B, m, n, H):
    return [[[sum(A[h][i][k]*B[h][k][t] for k in range(n))
              for t in range(n)] for i in range(m)] for h in range(H)]

geometries = lanes = 0
for m in range(1, 9):
    for n in range(m, 17, m):
        for c in range(1, m+1):
            if m % c:
                continue
            for H in (1, 2, 3):
                M, s = m//c, c*n*H
                for j in range(M):
                    for r in range(c):
                        for t in range(n):
                            for h in range(H):
                                for l in range(n):
                                    q, b = divmod(l, c)
                                    x = (r*n+t)*H+h
                                    y = (x-n*H) % s if t >= n-l else x
                                    src = (y+(-n*b+l)*H) % s
                                    f = (j-q-int(r<b)) % M
                                    sd, rem = divmod(src, n*H)
                                    st, sh = divmod(rem, H)
                                    actual = (sh, (c*f+sd+st) % m, st)
                                    expected = (h, (c*j+r+t) % m, (t+l) % n)
                                    assert actual == expected
                                    lanes += 1
                A = [[[((h*m+i)*n+k)*7 % 19-9 for k in range(n)]
                      for i in range(m)] for h in range(H)]
                B = [[[((h*n+k)*n+t)*11 % 23-11 for t in range(n)]
                      for k in range(n)] for h in range(H)]
                assert candidate(A, B, m, n, c, H) == dense(A, B, m, n, H)
                geometries += 1
for m, c, expected, printed in ((2, 1, 1, 3), (4, 2, 2, 10)):
    A = [[[m*i+k for k in range(m)] for i in range(m)]]
    B = [[[int(k == c and t == 0) for t in range(m)] for k in range(m)]]
    assert candidate(A, B, m, m, c, 1, False)[0][0][0] == printed
    assert candidate(A, B, m, m, c, 1)[0][0][0] == expected
    assert dense(A, B, m, m, 1)[0][0][0] == expected
print(geometries, 'geometries;', lanes, 'operand coordinates; counterexamples passed')
```

Executed directly from this document's Python block; observed output:

```text
240 geometries; 209880 operand coordinates; counterexamples passed
```

## Approval packet and remaining boundaries

The user approved only the modular full-block routing correction above in the existing
clear-slot optimized owner and its tests. Keep source-derived correction
explicit; retain the literal printed counterexample as a regression. Reuse
existing bounds, layout ownership, borrowed buffers, validation-before-mutation,
padding and suffix contracts. Test multi-frame wrap, `l>=m`, `b=0`, both time
branches, all heads, single-frame equivalence, and denied work/extent handling.

The proof extends to existing zero-extended valid rectangles by linearity if
source padding is ignored before arithmetic and output padding is zeroed.
Poisoned padding and buffer/error contracts still require implementation tests.
Partial final physical frames (`c` not dividing `m`) are outside this proposal.

No further source fact is missing for this **logical full-frame indexing**
resolution. Physical CKKS tail/rotation embedding (24576 logical lanes versus
32768 slots for the cited BERT geometry), Appendix B replication/transpose
mask schedules, packed lower-lower complexification, HE parameters and numerical
qualification remain separate gates. This approval does not resolve those gates
or authorize any encoder-to-decoder extension, upstream runtime or dependency.

## Implemented correction and validation

The existing `optimized_lower_schedule` and `optimized_lower_mm` owners now admit
bounded full logical frames with the approved modular routes. Original rotations,
scratch geometry and work ceilings are unchanged. Native regressions independently
check all 240 geometries and 209,880 operand coordinates, dense products, both
printed counterexamples, operation counts, exact/short work budgets, single-frame
compatibility, poisoned padding, untouched suffixes and failure atomicity.

```sh
cargo test --release --offline -j 1 -p pllm-core --lib
cargo clippy --release --offline -j 1 -p pllm-core --lib -- -D warnings
```

The full core suite passes **507 tests**; library Clippy passes. This validates
clear full-frame algebra only. The physical-slot, HE, numerical and decoder gates
above remain open; no encrypted execution or performance benefit is established.

## Native BERT source/numeric audit

This section records a separate, read-only source investigation. It does not
extend D25, change the approved routing, adopt a numerical profile, or authorize
an experiment. No upstream code, tensor payload, encrypted evaluator or bootstrap
was executed. Source inspection and bounded HTTP metadata reads are not model
evaluation. Native BERT remains the gate; no decoder extension is proposed here.

### Provenance and citation convention

The governing paper is the locked PDF/hash above: §5.2 p.10, §6.2 pp.10–11,
§7 pp.11–13 and Appendix C p.15. The author artifact is
[`crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25`](https://github.com/crypto-starlab/THOR/tree/b513c06aaabccc04b560721e35e9a98008428c25).
All artifact paths and line numbers below refer to that immutable Git commit;
not to a moving branch or an installed runtime. Existing artifact provenance
also lives in `paper/reproduction-contracts/encrypted-inference.md`. Source is an
oracle/provenance input, never a runtime dependency.

### Polynomial order and complete caller formulas

`src/thor/nonlinear/polynomial.py:4–64`,
`evaluate_polynomial_stockmeyer`, accepts increasing-order coefficients:
`P(p,u) = sum(p[i] * u**i)`. Lines 17–31 form consecutive baby blocks;
lines 33–64 combine them with `u**4`, `u**8`, `u**16`. There is no internal
coefficient reversal or input renormalization. For the actual list lengths
16, 28 and 32, these statements give the ordinary polynomial algebra, before
CKKS rounding and level/scale effects.

Both `gelu.py:48–53` and `tanh.py:29–36` reverse their literal lists first.
For a literal tuple `L=(L[0],...,L[d])`, their effective polynomial is therefore
`D_L(u) = sum(L[j] * u**(d-j))`. Paper Appendix C instead labels its tuples
through `f1_i*x**i`, `f2_i*x**i`, `g1_i*x**i`, `g2_i*x**i`. Paper §5.2 describes
GELU as direct `f2(f1(x))`. Do not silently equate these descriptions.

Let `F1,F2` be the descending-list polynomials from **all** original literal
values in `src/thor/nonlinear/gelu.py:32–47`, before line 51's factor `0.5`.
Statements in lines 18–25 and 48–55 establish:

```text
gelu.he_tanh_single(u) = F2(F1(u)) / 2
he_gelu(u) = 64*u * (1/2 + F2(F1(u))/2)
           = 32*u * [1 + F2(F1(u))]
```

This helper is not the identically named pooler helper in `nonlinear/tanh.py`.
The GELU docstring at lines 9–10 ("Input is 2*x" / "Output is gelu(x)/2") is
not a substitute for the explicit caller/encoder derivation:

| Boundary | Source statements and ideal payload consequence |
| --- | --- |
| FF1 weights | `model_encoder.py:66–85,178–226`: `gelu_scale=1/64`; `_encode_w_ff` stores complex lanes `scale*(W_a-i*W_b)/2`, i.e. `(W_a-i*W_b)/128`. |
| FF1 bias | `_encode_b:290–325`, especially line 320, stores `scale*b/2`, i.e. `b/128` for FF1. |
| FF1 extraction | `bert.py:696–713` adds encoded bias, then `ct+conj(ct)`. Under the intended real-lane matrix mapping, the complexification half cancels, giving `u=Z/64` for affine result `Z=W*h+b`. |
| Pre-GELU bootstrap | `bert.py:667–676` merges `u0+i*u1`, multiplies by `1/2`, bootstraps, then extracts with `temp+conj(temp)` and `i*(conj(temp)-temp)`. With an identity-on-payload bootstrap, the result is `u0,u1`, not another halving. |
| GELU result | Substitution gives `Z/2 * [1 + F2(F1(Z/64))]`. This is an algebraic artifact-call formula, not a proof of GELU accuracy or source-checkpoint equivalence. |
| FF2 and next residual | Default FF2 encoder scale is 1; complex weight/bias halves cancel at `bert.py:715–741`'s final conjugate sum. The subsequent residual merge/bootstrap at lines 680–688 has **no** preceding half, so ideal real extraction doubles that residual payload. |

`forward.ipynb`, zero-based cell 28, repeats this FF path. Cell 32's
`plot_variables` multiplies displayed GELU input by 64 and divides displayed
`ln2_in` by 2, independently documenting these caller conventions. There is no
`encode_matrix` symbol in this pinned encoder; the actual owners are
`ThorModelEncoder._encode_w_ff`, `_encode_b`, `_encode_w_pooler`, `_encode_b_pooler`.

All formulas here assume the stated real-lane mapping, linear operations and
an identity-on-payload bootstrap. They do not reconstruct actual encoded
coefficient rounding, bootstrap gain/error, physical-slot qualification or a
complete numerical DAG. In particular, do not infer that changing only a
polynomial order or constant preserves the implementation's accuracy.

For pooler Tanh, define `G1,G2` from the complete descending literal tuples in
`nonlinear/tanh.py:21–28`. `_encode_w_pooler` at `model_encoder.py:229–254`
stores `(W_a-i*W_b)/2`; `_encode_b_pooler:328–359` stores `b/2`.
`bert.py:772–807` adds bias and the conjugate sum, recovering intended affine
payload `A`; `forward:761–769` explicitly divides it by 40. Thus the ideal
artifact call is `G2(G1(A/40))`. `tanh.py:34–37` bootstraps before and after
the composition. Its `min_x`, `max_x`, `scale` parameters are passed through but
unused in helper arithmetic; the caller, not those parameters, supplies `/40`.

Two literal discrepancies must remain visible:

- Tanh second tuple, zero-based position 10: paper `2.7497e-1`; artifact
  `tanh.py:27` `2.74974898e+0`. With artifact reversal this is the coefficient of
  the inner result's **fifth power**. Caller `/40` does not compensate one
  isolated coefficient. In printed increasing order that tuple position would
  instead label the tenth power; order and literal value are separate conflicts.
- GELU first tuple, position 23: paper `1.2738e+3`; artifact `gelu.py:37`
  `1.27394360e+3`. These do not agree at the displayed paper rounding precision.

Neither discrepancy is corrected here. Nor are ordinary rounded paper tuples
silently replaced with the artifact's longer decimals.

### Recovered public MRPC metadata, not an evaluated-run identity

The README-linked [public Drive folder](https://drive.google.com/drive/folders/1mWBkNdsu3JCQPrSuedyeN_3WJD7h-6RO)
exposes `finetuned_models.tar`, Drive file ID
`1gC6-C2bCf-bEALhQtzk2s3MGSuHpcGAh`. Retrieved HTTP `206` byte-range
responses reported archive length **437,964,800 bytes**. Only tar headers,
configuration and safetensors metadata were inspected, not weight payloads:

| Object | Range/identity |
| --- | --- |
| `finetuned_models/mrpc/config.json` | Tar header at offset 512; 710-byte payload at inclusive archive bytes `1024–1733`; SHA-256 `29e736907be556ae889df44abd765ca3999ed72314b8d79dbcb30139e346fa5a`. |
| `finetuned_models/mrpc/model.safetensors` metadata | Tar header at 2048; file starts 2560; little-endian header-length bytes `2560–2567` specify 23,544 bytes; JSON header at `2568–26111`; header-only SHA-256 `3a4906d2ff5a91e61a30572c646eaa9ca9d43c6599d75cc1e2c26964e29fef4b`. This is **not** a checkpoint/tensor hash. |

Drive IDs/URLs are locators, not immutable versions; the above hashes bind only
the retrieved metadata bytes. They do not pin the rest of the mutable archive
or connect it to the paper's evaluated checkpoint. No copied archive is needed.

Recovered config: `BertForNextSentencePrediction`, `hidden_act="gelu"`,
`layer_norm_eps=1e-12`, 12 layers, 12 heads, width 768, FFN width 3072,
absolute positions/512 maximum positions, vocabulary 30,522, F32 and
`transformers_version="4.43.3"`. Safetensors metadata describes 201 tensors,
including embeddings `[30522,768]`, positions `[512,768]`, first intermediate
weight `[3072,768]` and `cls.seq_relationship.weight` `[2,768]`.

Remaining identity issues are specific, not an absent-config claim:

1. HE `nonlinear/layernorm.py:5–12` defaults to `var_e=1e-5`; recovered config
   says `1e-12`. Native HE callers (`bert.py:113,643–647`) use these helpers
   without a `var_e` override.
   This does not alone establish effective epsilon after all numerator/variance
   scaling; that needs the full LayerNorm algebra. It is not permission to
   replace either literal.
2. `utils/model.py:4–12` does **not** consume archived config. It constructs an
   unrevisioned `bert-base-uncased` model and loads the fine-tuned state dict.
   The recovered config therefore cannot establish that loader's exact runtime.
3. `forward.ipynb`, cell 15, obtains client embeddings independently from
   unrevisioned `bert-base-uncased`; cell 21's plaintext oracle loads the
   fine-tuned state. Equality of base and fine-tuned embedding tensors is
   **unverified**. No assumption of freezing/equality is licensed.
4. Notebook cells 13/18 choose MRPC sample 0 and encrypted-input level 20;
   its source has null execution counts and no saved outputs. It is not proof
   of which checkpoint, cache, key set or profile generated Tables 6–7.

The readable artifact does recover a layer-wise **declared helper profile**:
`bert.py:112–113` chooses `he_softmax2` only at zero-based layer 2 and
`he_layernorm1` for attention normalization; `bert.py:643–647` chooses
`he_layernorm3` for FF normalization at zero-based layers 9 and 10, otherwise
`he_layernorm2`. Their literal variance intervals are `[0.15,10]`, `[0.2,150]`,
`[0.75,2500]` (`layernorm.py:5–12`). `nonlinear/softmax.py:7–10` declares
`he_softmax1` bounds `[-27.2493,21.72692]`, `(n,l)=(2,2)`, inverse epsilon
`2^-11`, and `he_softmax2` bounds `[-70,70]`, `(n,l)=(2,4)`, inverse epsilon
`2^-18`; both use `output_alpha=0.01`. These are source function arguments,
not evidence of actual checkpoint coverage. They are not silently identified
with paper §7.2.2 p.12's MRPC microbenchmark range `[-44.19,50.25]` and
`(delta0,delta1,delta2)=(11.805,2,4)`. Source `he_inv` computes relaxation from
its bound state (`softmax.py:122–160`); it does not merely execute Appendix C's
printed literal factor sequence. Full softmax scale/denominator reconciliation
is outside the small polynomial experiment below.

### Ordered HE-prime source: recovered artifact cache, evaluated binding missing

Notebook cell 7 declares `logN=16`, `scale_bits=41`, four special primes and
`quantum="pre_quantum"`. Readable
`liberate/src/liberate/fhe/context/ckks_context.py:225–266` constructs, in order:

```text
q = B59[:15] ++ S41[:r] ++ [q0,p1,p2,p3,p4]
```

Here `B59` and `S41` are ordered cache lists indexed by `(59,65536)` and
`(41,65536)`; `r` is the source's `num_scales` selected by lines 248–263, not an
approved local parameter. Comments at lines 240–242 specify rescaling drops
from the left and special/key-switch primes from the right. Static pickle-opcode
inspection, without unpickling/executing upstream data, recovered the final
block from the immutable committed
`liberate/src/liberate/fhe/cache/resources/message_special_primes.pkl`:

```text
q0 = 1152921504606584833
p1 = 1152921504598720513
p2 = 1152921504597016577
p3 = 1152921504595968001
p4 = 1152921504592822273
```

**Initial recovery boundary:** the pinned tree lacks `scale_primes.pkl`, and
`context/generate_primes.py:252–279`'s default generation enumerates scale bits
`20,25,30,35,40,45,50,59`, **not 41**. A cache hit at
`ckks_context.py:183–194` can instead restore the entire context and return
before this construction. The README-mentioned external `resources.tar.gz`
was not among the four visible top-level Drive archives at inspection time;
absence from that listing did not establish that the data was unavailable.
The follow-up below recovers the complete linked artifact-cache chain, not
proof of the context used for the paper's evaluated run. No replacement primes
were generated.

#### Bounded follow-up source recovery, 22 October 2026

Immutable source links establish the intended resource location and consumer:

- [`README.md:17–18`](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/README.md#L17-L18)
  links external resources in the same Drive folder and names the cache directory.
- [`encode.py:12–19`](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/encode.py#L12-L19)
  declares `logN=16`, `scale_bits=41`, four special primes and
  `quantum="pre_quantum"`, then selects the MRPC safetensors path.
- [`context/ckks_context.py:175–194`](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/context/ckks_context.py#L175-L194)
  derives the cache filename from the generation string and restores cached
  fields before returning. Its
  [lines 248–266](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/context/ckks_context.py#L248-L266)
  distinguish `r` ordinary scale primes from the stored `num_scales=r+15`.
- [`ckks_engine.py:61–63`](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/ckks_engine.py#L61-L63)
  defines the context hash as SHA-256 of UTF-8
  `generation_string + "_" + ",".join(str(qi) for qi in q)`.

The follow-up Drive listing exposed **`resources.tar`**, not the README's
`resources.tar.gz`, with file ID `1LPQex129MuFclJp5F4sJN9fIrMVYLgXH`:
[resource download locator](https://drive.usercontent.google.com/download?id=1LPQex129MuFclJp5F4sJN9fIrMVYLgXH&export=download&confirm=t).
**This Drive URL/file ID is mutable, not an immutable version or content pin.**
The immutable source links establish author linkage, not the identity of future
Drive responses. Retrieved HTTP `206` responses reported archive length
**14,766,817,280 bytes**. All offsets below are zero-based; ranges are inclusive.

| Inspected object | Archive position and scope |
| --- | --- |
| Matching tar member header | Bytes `9414646272–9414646783` (512 bytes); names `resources/62_41_16_None_4_128_pre_quantum_uniform.pkl`; declares member payload length **301,993,736 bytes**. |
| Member payload | Starts at `9414646784`; full member was **not** downloaded or hashed. |
| Initial bounded payload read | Bytes `9414646784–9414679551` (32,768 bytes); inspected as static pickle opcodes only. |
| Independently reread metadata prefix | Bytes `9414646784–9414647682` (899 bytes); contains the generation string, scalar declarations, full ordered `q` list and `total_qbits`. |

SHA-256 identities bind only the indicated retrieved bytes:

```text
tar header (512 bytes):
ed03d690c3eb33845ed5fa7a6fd3cbb50ce33842f6afa4e0fae47dc4d519d97f
metadata prefix (899 bytes):
d12a221ecfc22a220639ad1f2b960b88c3c1a6d6cb0be9d6ffcb17d7bd65cfa7
```

Neither digest is a full archive, member, checkpoint or evaluated-run hash.
Inspection used standard-library `pickletools.genops`, **not** `pickle.load`
or `pickle.loads`. No upstream module was imported or executed; global/reduce
opcodes were not invoked. Parsing stopped at an array-byte opcode whose declared
payload exceeded the bounded prefix; no array reconstruction was attempted.
No checkpoint/tensor payload was read for this follow-up, and no source archive
collection or copied source tree was created. The inspection existed only as
inline commands; no saved inspection experiment was found to extend safely, so
this update adds documentation only, not a new experiment or test.

The inspected cache declares the following values; these are **artifact
configuration fields**, not independently verified security or runtime claims:

```text
generation_string = 62_41_16_None_4_128_pre_quantum_uniform
buffer_bit_length = 62; N = 65536; logN = 16; scale_bits = 41
message_bits = 60; num_special_primes = 4
security_bits = 128; quantum = pre_quantum; distribution = uniform
sigma = 3.2; uniform_tenary_secret = True
num_scales = 29; max_qbits = 1760; total_qbits = 1759
```

Its complete stored `q` list is ordered below. Group breaks identify the first
15 bootstrap primes, the next **14** ordinary scale primes (`r=14`), and the
message/four-special-prime block; breaks are not entries or reorderings.

```text
576460752308273153, 576460752289923073, 576460752315482113,
576460752289529857, 576460752319021057, 576460752289005569,
576460752319414273, 576460752279306241, 576460752321642497,
576460752273801217, 576460752328327169, 576460752272228353,
576460752364240897, 576460752267509761, 576460752371187713,

2199028891649, 2199010148353, 2199030071297, 2199007789057,
2199043571713, 2199006216193, 2199053008897, 2199006085121,
2199056416769, 2198996647937, 2199056547841, 2198958899201,
2199057203201, 2198955753473,

1152921504606584833, 1152921504598720513, 1152921504597016577,
1152921504595968001, 1152921504592822273
```

Applying the cited source hash formula to this generation string and all 34
ordered integers, independently of upstream execution, gives:

```text
bfdd988a99a05a594adcef5e82685e24cfc6dcccf778149a4bdeb590eb61298d
```

This is a **source-defined context identifier calculated from recovered metadata**,
not a hash of the complete serialized cache or an observed evaluated-run ID.
The final five integers agree with the committed message/special block above.
Recovery removes the missing-public-artifact-chain obstacle, but supplies no
author record binding this cache to Tables 6–7, loaded keys, an exact MRPC
checkpoint or base-model revision. No numerical profile is adopted; CKKS
security, bootstrap correctness/precision and encrypted execution readiness
remain unqualified. No decoder or full-logit boundary is approved.

Readable non-bootstrap key topology is not wholly unspecified:
`liberate/src/liberate/fhe/ckks_engine.py:2114–2131` gives `evk: s^2 -> s`,
lines 2258–2279 give rotation-key switching, and lines 2893–2915 give
conjugation-key switching. The Galois bundle uses powers-of-two rotations
(lines 81, 2326); notebook cells 9–10 contain the literal additional rotation
lists and load `pk`, `evk`, `gk`, `conjk`, then supply bootstrap rotation keys.
This recovers source topology, not actual loaded-key provenance, distributions,
security or protected-bootstrap precision. Shared backend qualification remains
owned elsewhere; no backend implementation is proposed here.

### Minimal native-BERT decision packet

**Faithful derivations already available:** evaluator coefficient semantics;
artifact's caller formulas and half/double cancellation under the ideal-payload
caveat; metadata byte identities; complete linked artifact-cache prime ordering
and readable key graph. Evaluated-paper context/checkpoint binding remains missing.
These may inform implementation contracts without claiming an adopted profile.

**Targeted author/source questions:**

1. Confirm intended coefficient order and full GELU variable/output scaling;
   issue the complete authoritative tuples, resolving both literal discrepancies
   above. Is the paper's direct-composition description or the artifact caller
   the intended numerical function? An order-only explanation does not resolve
   the additional GELU product/caller envelope.
2. Identify the exact evaluated MRPC checkpoint/base-model revision, embedding
   custody/freeze policy, effective LayerNorm epsilon and layer-wise nonlinear
   profiles. Tie them to the evaluated run rather than the output-free notebook.
3. Did Tables 6–7 use context hash
   `bfdd988a99a05a594adcef5e82685e24cfc6dcccf778149a4bdeb590eb61298d`?
   Supply an immutable evaluated MRPC checkpoint SHA-256, base-model revision
   and run record binding that checkpoint to this context; if different, publish
   the actual evaluated ordered chain/cache. The recovered linked cache is not
   that binding. Do not regenerate primes from defaults that omit scale 41.

**User choice after source resolution:** approve one explicitly named native
BERT numerical identity, with its source departures enumerated, or authorize
only the bounded reconciliation study below. No request here adopts a profile.
Reversing printed tuples, adding artifact caller envelopes to them, swapping
epsilon, refitting, widening domains, clipping, replacing primes/bootstrap or
combining selected constants are **unadopted reinterpretations/adaptations**,
not silent source reproduction. Governing paper remains authority; contradictory
artifact paths remain separately labeled oracle candidates.

### Proposed next bounded experiment — approval still required

**Approval update, 23 October 2026:** the user approved this exact bounded study
as D30. The historical heading remains for stable links. Execution/results and
profile adoption are not established by approval.

Smallest useful request: one reusable, source-independent Python reconciliation
experiment and ordinary output; no copied source tree, evidence archive,
tensor/checkpoint download, upstream import/execution, network, HE/native call,
build, installation or profile change. At the time of this proposal, no code or
numerical study had been run; the subsequent D30 execution is recorded below.

- **P (printed-literal candidate):** transcribe the complete Appendix C tuples
  at displayed precision, use their printed increasing order and the literal
  direct compositions `f2(f1(Z))`, `g2(g1(A))`. Missing fit domains/scaling remain
  explicit; this is a literal reading, not a claimed valid approximation.
- **A (artifact-call candidate):** use all original pinned literal decimals,
  descending order and exactly `Z/2*[1+F2(F1(Z/64))]`, `G2(G1(A/40))` above.
  No constant replacement, rounding-to-paper or omitted caller factors.
- **No hybrid candidate.** Explain the Tanh discrepancy symbolically at tuple
  position 10; do not splice its printed coefficient into A and label the
  result source reproduction. A later order-only or literal-only experiment
  would require separately named diagnostic variants/approval.
- **Inputs:** GELU `Z=k/16`, Tanh `A=5*k/16`, integers `-128<=k<=128`, plus
  `+/-2^-10` and `+/-2^-20` for each: 261 points/function. GELU `[-8,8]` is
  a declared diagnostic window, **not** a recovered fit/activation domain;
  Tanh `[-40,40]` is the helper's declared parameter interval, not proof of
  checkpoint coverage. No data-dependent range search.
- **Arithmetic/output:** local scalar evaluation with Decimal 80/160-digit
  checks and a separate binary64 diagnostic. For A, retain the readable
  Stockmeyer grouping in the binary64 diagnostic; do not equate real algebra
  with CKKS execution. Report nonfinite values and precision sensitivity rather
  than clipping or suppressing them. `math.erf`-based exact-GELU formula and
  `math.tanh` are binary64 comparison baselines only, not certified oracles.
- **Hard budget:** one CPU process, at most 5,000 complete candidate-point
  evaluations, 30 seconds wall time, 64 MiB working memory and 1 MiB ordinary
  output; stop on any exceeded bound. No dependency installation. If this
  budget cannot support the checks, report the limit, not a larger run.
- **Cannot infer:** intended author variant, full-model accuracy, actual
  activation domains, fine-tuned/base embedding equality, effective epsilon,
  CKKS error/security, bootstrap correctness, exact primes, encrypted latency
  or admission. Observing one better curve does not authorize adopting it.

### Observed D30 literal reconciliation, 23 October 2026

The approved clear scalar study completed once. Its experiment and nine normal
regression tests are committed in **`560522f6`**. The retained ordinary output
is `target/thor-literal-reconciliation-r1.json` (ignored, not an archived source
tree). This documentation update reads that output; it does not rerun the study.
All 92 printed coefficients were checked against the pinned Appendix C PDF,
and all 92 artifact coefficients against the previously retrieved, locally
retained source at `b513c06aaabccc04b560721e35e9a98008428c25`. No new upstream
fetch/import or checkpoint, HE/native, installation or build workload was used.

**Source-to-result mapping.** In
[`benchmarks/research/thor_literal_reconciliation.py`](../../benchmarks/research/thor_literal_reconciliation.py),
`P` and `A` retain the two complete literal sets; `points` fixes the grids above;
`evaluate` applies the two complete caller formulas and three arithmetic modes.
P uses increasing-order Horner evaluation. A reverses the pinned lists and
retains Stockmeyer grouping, including GELU's outer-coefficient halving before
evaluation. `study` produces `points.<function>` and
`summary.<function>.<candidate>` in the JSON. The numerical tables below read
that summary's `classifications`, `baseline_binary64_errors.decimal160`,
`decimal80_vs_160` and `binary64_vs_decimal160` fields; display values are rounded,
while the JSON retains full recorded decimal strings and round-trip binary64
strings. No hybrid, coefficient repair, clipping or additional variant was run.

There were **3,132 complete candidate-point evaluations**:
`2 functions × 261 points × 2 candidates × 3 arithmetic modes`. Recorded resource
windows were:

| Recorded field | Observed value and scope |
| --- | --- |
| `evaluation_wall_seconds` | `0.04957262519747019` seconds, from budget initialization through evaluation/summary construction. |
| `wall_seconds_before_output` | `0.051475292071700096` seconds, through the initial output-encoding phase, before final serialization/write. |
| `process_peak_rss_bytes_through_evaluation` | `21,200,896` bytes (`20.21875 MiB`), process high-water RSS observed at the evaluation endpoint. |
| `process_peak_rss_bytes_before_output` | `22,249,472` bytes (`21.21875 MiB`), process high-water RSS observed before final serialization/write. |
| `output_bytes` | **370,762 bytes**, matching the retained JSON's actual length. |

The run exited successfully; recorded counts/readings were below the fixed
5,000-evaluation, 30-second, 64-MiB working-RSS and 1-MiB output ceilings. The
script also armed a wall timer and checked time/RSS after final serialization;
that last check's readings were not stored. These times
exclude interpreter startup, imports and literal construction before budget
initialization, and do not measure final writing/exit. RSS is the process's
observed lifetime high-water mark through each named endpoint, not a separate
allocation total or a measured final-exit peak. There were no worker processes.

**Observed values, not accuracy certificates.** Every Decimal80 and Decimal160
result was finite (261 per row in each mode). All nonfinite binary64 results
were **negative infinity**; there were no positive infinities or NaNs.
The baseline column is maximum absolute difference between Decimal160 and the
binary64 `math.erf`-based GELU or `math.tanh` diagnostic, not a certified function
error or a full-model quality measurement.

| Function / candidate | Binary64 finite / negative infinity | Maximum absolute diagnostic-baseline difference (input) |
| --- | --- | --- |
| GELU / P | 79 / 182 | `1.358872781611939e754` (`Z=-8`) |
| GELU / A | 261 / 0 | `1.038682580615624e-3` (`Z=-3.125`) |
| Tanh / P | 161 / 100 | `2.158494128906946e353` (`A=40`) |
| Tanh / A | 261 / 0 | `0.9418092907000301` (`A=40`) |

**Precision sensitivity.** Decimal80-versus-Decimal160 differences used all 261
finite pairs per row. Relative differences divide by the absolute Decimal160
value only when it is nonzero; zero references are counted separately, without
an invented denominator floor. Agreement between two finite precisions is not
a certified bound on either result.

| Function / candidate | Maximum absolute Decimal80–160 difference (input) | Maximum relative Decimal80–160 difference (input) |
| --- | --- | --- |
| GELU / P | `1.277920120655198e675` (`-8`) | `1.087235957555707e-74` (`1`) |
| GELU / A | `8.841653844605608e-75` (`-8`) | `1.257286801769220e-70` (`-7.6875`) |
| Tanh / P | `1.020736062623355e274` (`40`) | `8.325845448902007e-79` (`23.4375`) |
| Tanh / A | `1.495099157975856e-76` (`39.6875`) | `1.603145608392157e-76` (`39.6875`) |

The binary64-versus-Decimal160 summary reports the following maxima only over
finite pairs; P's excluded nonfinite pairs remain explicitly counted above.

| Function / candidate | Finite pairs | Maximum absolute binary64–Decimal160 difference (input) | Maximum relative difference (input) |
| --- | --- | --- | --- |
| GELU / P | 79 | `1.627223666010971e289` (`2.3125`) | `1.551844557347051e-11` (`-1`) |
| GELU / A | 261 | `2.813225336728333e-11` (`-7.9375`) | `2.189652998037005e-6` (`-7.6875`) |
| Tanh / P | 161 | `2.383296712530603e293` (`-24.375`) | `5.503361656932969e-13` (`-1.5625`) |
| Tanh / A | 261 | `5.618416073768156e-12` (`40`) | `9.655177160335911e-11` (`40`) |

A's finite outputs therefore do **not** establish accuracy: its Tanh diagnostic
discrepancy near `0.9418` at `A=40` persists at both Decimal precisions, whereas
the binary64–Decimal160 difference there is about `5.62e-12`. Conversely, P's
literal results on these declared windows do not identify the intended author
function or recover missing fit domains. Neither comparison selects a winner,
adopts a numerical profile/backend, binds source to the evaluated paper run, or
qualifies CKKS/bootstrap error, security, actual activation coverage or full-model
quality.

Reusable commands from the repository root (not rerun for this update):

```sh
mkdir -p target
python3 benchmarks/research/thor_literal_reconciliation.py > target/thor-literal-reconciliation.json
python3 -m unittest discover -s tests -p 'test_thor_literal_reconciliation.py' -v
```

The observed nine-test parent run passed; its ordinary log is
`target/thor-study-parent-tests-r1.log`. Tests cover exact rational composition
references, the declared grids/literals, binary64 Stockmeyer grouping,
nonfinite/zero-reference handling and evaluation/time/memory/output limits.
They do not certify either diagnostic baseline or qualify encrypted execution.
