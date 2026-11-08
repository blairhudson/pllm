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

### Reviewed conditional physical embedding and literal source routing

Independent Astra51 algebra review and Astra56 read-only source recovery supply
this conditional attachment, pinned to commit
`b513c06aaabccc04b560721e35e9a98008428c25`:
[`src/thor/linear.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/linear.py),
[`src/thor/bert.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/bert.py).
Astra56 fetched source text; neither source recovery nor the parent session
executed it. This is current derivation only, not an extension of D25 approval.

**Zero-filled embedding.** For 12 logical heads in 16 physical lanes define
`E(v)[16*u+h]=v[12*u+h]` for `0<=h<12`, with lanes `12..15` zero and
`u mod 2048`. Write `R_d(v)[x]=v[(x+d) mod len(v)]`. For every integer `k`,
`R_(16*k) E = E R_(12*k)`: both sides read logical group `u+k`, head `h`;
holes remain holes. Source-aligned rotations, pointwise multiplication,
addition/subtraction, conjugation and multiplication by `i` preserve this image
when operands are E-images. Residue-constant plaintext masks need NOT be zero
at holes: multiplying a zero-filled operand still preserves its zero holes.
This excludes arbitrary nonzero holes; CKKS error is independently OPEN.

**Literal routing under ideal slots.** Use `c=16,n=128`, physical address
`2048*r+16*t+h`, and `M=m/16`. General algebra assumes `16 | m | 128`;
source methods instantiate fixed cases, not arbitrary admitted `M`. For diagonal
`l`, put `b=l%16`, `q=floor(l/16)`, `rrot=-16*(l-128*b)`. The literal input
rotation is by `-rrot`; mask time threshold is `2048-16*l`. Delayed mask
columns 1 and 3 use `R_(-2048)`, reading local diagonal `r-1` modulo 16.
The four-mask selector is `f=(j-q-[r<b]) mod M`, with source coordinates
`((r-b) mod 16,(t+l) mod 128,h)`, matching the logical coordinate proof above.
For `b=0` only two masks are needed; otherwise mask 3 is `1-M0-M1-M2`.
The zero-diagonal product `P_(j,0)` is separate: score loops use `l=1..63`,
context loops `l=1..127`, both excluding zero.

**Packed literal cases.** Scores pack eight real frames into four ciphertexts
(`p=4`); context packs four real frames into two (`p=2`). Set
`k=q+[r<b]`, `f=(j-k) mod (2*p)`: select source ciphertext `f mod p`, swapping
components iff `f>=p`. Context's `(l//16)%4` correctly wraps for `l>=64`.
Columns 2 and 3 collect swapped contributions. For REAL component frames,
`i*conj(a+i*b)=b+i*a`. The RHS must be REAL and SHARED by both paired
components; otherwise product swapping conjugates the RHS and this argument
fails. Actual codec, RHS replication, zero holes and source-entry premises
remain required. `he_softmax1/2`, the actual context-RHS producers, are outside
these two retrieved files; this attachment does not establish their output
representation or the complete source-to-codec chain.

**Scaling boundary — zero diagonal only.** Literal `scale_pt` applies ONLY to
`l=0`, not the nonzero-diagonal masked terms. Score halves that zero-diagonal
scale when `bootstrap=True`; context halves it when `complex=False`, but still
returns complex ciphertexts. After rescale and optional bootstrap, score
extraction is exactly `z+conj(z)=2*Re(z)` and
`i*(conj(z)-z)=2*Im(z)`; context returns rescaled complex `z`.
Supplied scale/flags therefore cannot be described as uniformly scaling the
WHOLE product. Actual call-site encoder factors and arithmetic-helper contracts
are required before asserting a dense-output identity. This is distinct from,
and does not rewrite or contradict, the FFN scaling audit below.

Remaining helper obligations include `CkksEngine.encode`,
`pt_ct_mult_extended`, `mult_int_scalar_triplet`, rescale semantics for the
hardcoded `2**41` complement alignment, and `cc_add` with `None` accumulators.
No encrypted-qualified rotations/program, encoder issuer, CKKS error bound,
security or adoption follows. No Qwen/decoder or D25 extension, code, tests or
execution is authorized. Native HOLD and cleanup UNKNOWN remain unchanged.

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
   The conditional ideal-payload derivation below gives effective epsilon
   `1e-5/768^2` relative to the undoubled residual, not the helper literal alone.
   This differs from the config, but establishes neither a numerical bug nor
   reconciliation, and is not permission to replace either literal.
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
was a source gap in this earlier audit and remains outside the small polynomial
experiment below; the recovered subsection below narrows that gap, not HE qualification.

#### Recovered LayerNorm payload algebra and inverse-square-root proof gaps

Two independent Astra recoveries agree on this documentation-only source
attachment at `crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25`:
[`nonlinear/layernorm.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/layernorm.py),
[`bert.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/bert.py),
[`model_encoder.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/model_encoder.py),
[`ckks.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/ckks.py).
These are existing artifact sources, not added dependencies. Source text was
inspected, not executed; this derivation is not an authorized reconciliation.

**Conditional reduction geometry.** Assume 32,768 physical slots, the intended
real-lane support and left-rotation semantics, and ideal payload arithmetic.
For one token, the eight input ciphertexts each contribute sixteen stride-2048
positions with six live residues per group of sixteen: `n=8*16*6=768`.
`CkksEngine.rotsum(...,2048)` then sums the sixteen positions. Successive
additions of rotations `+1,+2,+4` sum **eight**, not six, residues; the
residue-zero result is the intended six-value sum only because residues 6 and 7
are zero after the first mask. The residue-zero mask retains that sum, and
rotations `-1,-2,-4` broadcast it over residues 0 through 7. The same reduction
is used for squared inputs, and the inverse-square-root result is broadcast
likewise. This is a conditional source layout argument, not qualification of
physical rotations, support preservation or the earlier matrix indexing gates.

Let `r` be the **undoubled residual** for that token, with population statistics
`mu=sum(r)/n` and `v=sum((r-mu)^2)/n`. Write helper input as `g*r`, its live
mask coefficient as `s/sqrt(C)`, and `t=s*g`, where

```text
e = 1e-5
C = (1.05*max_var + e)*n^2
N = n*t*(r-mu)/sqrt(C)
D = (n^2*t^2*v + e)/C
```

The numerator is `n` times the masked input minus its reduced sum. The
denominator input is `n*sum(masked_input^2)-sum(masked_input)^2+e/C`, not
population variance plus the unscaled literal `e`. Let `R(D)` denote the
returned inverse-square-root payload, including its actual recurrence factors,
and define `rho(D)=sqrt(D)*R(D)`. `_encode_b` stores gamma and beta payloads
with a half; `he_layernorm`'s final affine doubling cancels **both** halves.
Consequently, for `t>0` and `D>0`, exact ideal-payload algebra gives

```text
y = gamma*N*R(D) + beta
  = gamma*(r-mu)*rho(D)/sqrt(v + e/(n^2*t^2)) + beta
```

Here gamma and beta mean the intended unhalved affine parameters. No assertion
that `rho(D)=1` is made.

| Caller boundary | Conditional gain and effective epsilon |
| --- | --- |
| Attention (`bert.py`, residual addition before `self.layernorm`) | `g=1`, `s=1`, hence `t=1`. |
| FF (`bert.py:680–688`, both FF profiles) | Residual is added **before** bootstrap; merge/extraction has no preceding half. Identity-on-payload bootstrap and conjugate extraction double **both residual branches**, so `g=2`; LayerNorm's half mask gives `s=1/2`, hence `t=1`. |

Thus both paths have ideal effective epsilon `e/n^2=1e-5/768^2`, approximately
`1.69542e-11`, relative to their respective **undoubled** residual. Relative
to the FF helper input itself (`2*r`), it is instead `4*e/n^2`. The recovered
config's `1e-12` is different; this does not identify the intended evaluated
function, establish a numerical bug or authorize epsilon replacement.

This simplification is not bit-exact source floating-point algebra. If `b` is
the actual rounded live mask coefficient and `d` the source-computed additive
scalar `e/C`, the corresponding ideal-operation ratio is
`epsilon_ratio=d/(n^2*b^2*g^2)`. Independent rounding of these scalars need not
preserve the symbolic cancellation above; CKKS encoding and arithmetic add
further unqualified effects.

**Pinned normalization attachment.** The FF profiles are `(a,M)=(0.2,150)`
and `(0.75,2500)`, with `e=1e-5`, `n=768`
([lines 8–12](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/layernorm.py#L8-L12)).
Source sets `eta=a/M`, `C=(1.05*M+e)*n**2` and live mask coefficient
`1/C**(1/2)`, then divides the FF mask by two; this is exponentiation syntax,
not `np.sqrt`
([lines 21–27](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/layernorm.py#L21-L27)).
The denominator is `D=n*sum(masked_x^2)-sum(masked_x)^2+e/C`.
LayerNorm passes `enc_one` and `alpha=0.001`
([lines 65–75](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/layernorm.py#L65-L75));
`he_invsqrt` requires `alpha`, with no default.

**Inverse-square-root caveat 1 — constants and invariant.** On a retained real
lane, the general helper starts `A=D=denominator`, `B=N=numerator`, and bound
state `eta`; only LayerNorm's `enc_one` specializes `N=1`. The pinned
[helper, lines 104–130](https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/layernorm.py#L104-L130)
selects the **unsorted second returned root**
`k=np.roots([1-eta**3,6*eta**2-6,9-9*eta])[1]` and separately computes
`p=k**(3/2)/2`, `q=k**3/4`, `h=3/k`. Its optional bootstrap scales `A` by
64, bootstraps and divides by 64; it also bootstraps the already-scaled `p*B`.
Under intended reduction support and payload-preserving bootstrap, rescale and
alignment, the ideal lane updates are

```text
B' = (p*B)*(h-A)
A' = (q*A)*(h-A)^2                 (both use OLD A)
eta' = k*eta*(3-k*eta)^2/4
```

To avoid assuming a numerator of one or dividing by a possibly zero payload,
define `R_0=1`, `R'=p*R*(h-A)`. Then literal helper output is `B=N*R`,
including `N=0`. With `E=A-D*R^2`, exact algebra gives

```text
E' = (h-A)^2 * [q*E + (q-p^2)*D*R^2]
A_m = D*R_m^2 * product_{j=0}^{m-1}(q_j/p_j^2)
```

The product identity assumes finite, defined constants and every `p_j!=0`;
it divides by no `D`, `N` or `R`. Exact positive-`k` constants satisfy `q=p^2`
and yield the familiar invariant `A=D*R^2`, not general-helper `A=D*B^2`.
Separate floating evaluations permit, but do not prove, `q!=p^2` or an actual
nonzero propagated defect. Root ordering/choice, positivity and finite floating
constant instantiation have not been measured or qualified. The bound-state
update and stopping condition `eta>=0.999` for this caller are not recurrence
accuracy certificates for `R(D)` or `rho(D)`.

**Conditional finite ideal-payload residual proof.** Astra9's independent review
conditionally verifies the algebra, but did not find a local pinned
`layernorm.py`; source alignment here relies on the existing audit transcription,
not fresh upstream verification.

Subsequent read-only review directly inspected the pinned LayerNorm helpers,
attention/FF caller gains and affine half encoding at the same revision, finding
no scalar-transcription discrepancy. That closes source attachment for these
symbols only; rounded-value premises and runtime qualification remain
unestablished.

Fix a finite iteration count `m`, `D>0`,
`A_0=D`, `R_0=1`, and iteration-indexed finite real constants `q_j>0`,
`p_j!=0`, `h_j`, `k_j`. Assume finite scalar trace `e_j` (the helper's bound
state `eta`) and defined ideal-payload updates using old `A_j`:

```text
A_{j+1} = q_j*A_j*(h_j-A_j)^2
R_{j+1} = p_j*R_j*(h_j-A_j)
G_j(x) = k_j*x*(3-k_j*x)^2/4
r_j = e_{j+1}-G_j(e_j)
F_j(x) = q_j*x*(h_j-x)^2
s_j = F_j(e_j)-e_{j+1}
    = e_j*[(q_j*h_j^2-9*k_j/4)
           +(-2*q_j*h_j+3*k_j^2/2)*e_j
           +(q_j-k_j^3/4)*e_j^2]-r_j
Delta_j = A_j-e_j
H_j = q_j*[h_j^2-2*h_j*(A_j+e_j)+A_j^2+A_j*e_j+e_j^2]
Delta_{j+1} = H_j*Delta_j+s_j
Delta_m = (D-e_0)*prod_{j=0}^{m-1} H_j
          + sum_{i=0}^{m-1} s_i*prod_{j=i+1}^{m-1} H_j
```

Here `r_j` includes scalar evaluation rounding, not merely rounded constants;
the payload updates themselves remain ideal real operations. The difference
factorization `F_j(A_j)-F_j(e_j)=H_j*(A_j-e_j)` proves the recurrence and finite
variation expression, with empty products equal to 1 and empty sums to 0.
Define `P_m=prod_{j=0}^{m-1}(q_j/p_j^2)>0`. Induction gives
`A_m=D*R_m^2*P_m`, hence `rho_m=sqrt(D)*R_m` satisfies
`rho_m^2=(e_m+Delta_m)/P_m`, even when `R_m=0`. To bound distance from **+1**,
also require `rho_m>=0`: every `p_j*(h_j-A_j)>=0` suffices; nonnegative terminal
product of those factors is a weaker sufficient condition.

The stop threshold must be the **actual finite floating comparison operand**
`t`, not exact-real `1-alpha`. If the finite comparison really exits with
`e_m>=t`, put `alpha_stop=(1-t)_+`, where `x_+=max(x,0)`. Then

```text
|rho_m-1| = |rho_m^2-1|/(rho_m+1)
          <= [alpha_stop+(e_m-1)_+ + |Delta_m|+|P_m-1|]/P_m.
```

This uses `|e_m-1|<=(1-t)_+ +(e_m-1)_+`; it assumes no `e_j<=1` invariant.
NaN comparison failure is not convergence. Initial shortfall `D-e_0`, terminal
overshoot and product drift remain explicit. For general output `B_m=N*R_m`
and additive encrypted output error `xi`, the numerator-relative normalized
error obeys
`|sqrt(D)*(B_m+xi)-N| <= |N|*|rho_m-1|+sqrt(D)*|xi|`.
Only live `enc_one` lanes specialize `N=1`, giving the normalized live bound
above plus `sqrt(D)*|xi|`. No bound on `xi` is supplied here.

**Smaller direct interval route, still unfinished.** Rather than bounding every
defect term, ordinary future proof/tests could certify actual root selection,
rounded constants, finite scalar trace and stop comparisons; establish the
positive actual rounded normalization domain
`D(v)=768^2*b^2*g^2*v+d` from rounded mask `b`, caller gain `g` and additive
scalar `d`, without symbolic ideal cancellation; and propagate an enclosing
interval `I_j` through `F_j`. Its extrema require endpoints **and any contained
stationary points `h_j/3`, `h_j`**, since
`F_j'(x)=q_j*(h_j-x)*(h_j-3*x)`. Sign control and positive bounds for `P_m`
remain necessary. For a desired `0<=tau<1`, the resulting enclosure must satisfy
`A_m/P_m subset [(1-tau)^2,(1+tau)^2]` together with `rho_m>=0`.
This is a smaller proof route than the defect route, not an executed
implementation or benchmark. No tolerance is adopted or rounded BERT certificate
completed; no source-model variance coverage, CKKS, security or runtime accuracy
qualification follows. The symbolic initial-coverage shortfall below remains
relevant and is not erased by this conditional proof.

**Premise owners and failure boundaries (source versus mathematics).**
Read-only code inspection identifies existing producer **S** =
[`scripts/thor_scalar_certificate_experiment.py`](../../scripts/thor_scalar_certificate_experiment.py)
and independent test transcription/oracles **T** =
[`tests/test_thor_scalar_certificate_experiment.py`](../../tests/test_thor_scalar_certificate_experiment.py).
Sites below describe code ownership, not executed checks; runtime **UNRUN**.
S transcribes the pinned source above; T independently transcribes/checks those
specific scalar expressions, not arbitrary `Trace` provenance.

| Required premise / boundary | Existing owner and inspection site | Failure meaning |
| --- | --- | --- |
| Pinned profiles, literal types, `n=768`, evaluation grouping and rounded mask/addend | S: `PIN`, `PROFILES`, `N`, `source_trace` (24–26, 115–130); T: `test_fixed_profile_values_literal_types_and_dimension`, source oracle (41–51, 220–232) | Changed source attachment needs rechecking; exact decimal cancellation cannot substitute. |
| Actual finite floating states; NaN refused **before comparison** | S: `float_ratio`, `source_trace` (43–52, 133–138, 155–162); T: nonfinite/root/arithmetic/post-state refusal tests (132–140, 280–332) | Nonfinite state is refusal, never convergence. |
| `roots[1]`: **UNSORTED second returned root** | S: `source_trace` (144–150); T: independent `np.roots` call and returned-pair/index checks (236–244) | Stored roots/index alone do not prove source root selection. |
| Scalar state chain and literal `en` update; both payload updates use old `A` | S: `source_trace` (143–167), `exact_step` (229–235), `direct_enclosure` (250–258); T: source oracle (245–254), old-`A` oracle (73–81) | A disconnected chain, regrouped scalar update or sequential new-`A` update loses the stated correspondence. |
| Actual `1-.001` stop operand and finite termination | S: `source_trace` (126–142), `analyze` (344–346); T: stop/terminal oracle (231–257), safety and overshoot tests (180–189, 270–278) | `completed` alone is not proof; unfinished/safety-limited execution cannot certify convergence. |
| Independently authenticated source `Trace` provenance | S: mutable `Step`/`Trace` (77–101); T: independent source oracle (212–268), separate synthetic fixtures (22–37) | Arbitrary mutable traces are **not authenticated** by stored values, `root_index`, `completed`, `check_sufficient` or `analyze`; mathematical checks cannot establish provenance. |
| Positive actual rounded domain `D`, not ideal cancellation | S: `domain`, `direct_enclosure` (176–180, 245–247); T: independent rounded-domain oracle (258–263), nonpositive-domain tests (142–150) | Nonpositive `D` invalidates this inverse-root route; assumed variance endpoints are not actual-model domain coverage. |
| Separately rounded `p`, `q`; positive `P=product(q/p^2)`, **not assumed 1** | S: coefficient capture (151–167), `exact_step`/`direct_enclosure` (229–258); T: independent coefficient ratios (246–249), `q!=p^2` oracle (73–81) | Dropping product drift or admitting undefined/nonpositive coefficients breaks the square-identity bound. |
| Sign of `rho`; zero is nonnegative, not strictly positive | S: `direct_enclosure` (253–270); T: signed fallback and zero-factor tests (152–167) | Square enclosure alone cannot select the positive branch; signed-`R` fallback retains sign uncertainty. |
| Optional sufficient theorem versus direct fallback | S: `check_sufficient`, `analyze` (273–360); T: failed-maximum and all-premises tests (83–123) | Failed sufficient premise leaves that theorem unproved; direct enclosure remains mathematically valid under its own premises, but cannot repair source correspondence. |
| Conditional tolerance criterion and result scope | S: `analyze`/status sites (259–270, 344–359); T: enclosure/status oracles (83–123, 152–167) | `CERTIFIED_BOUND` denotes an ideal-scalar enclosure only, not application acceptance, actual-model domain, encrypted-error or BERT qualification. |

**Adoption DEFERRED.** Only **if ever adopted**, `tau=13/500` would require
`A_m/P_m subset [237169/250000,263169/250000]` together with `rho_m>=0`
for the square-identity route. Failure to establish this criterion means
**unproved**, not an actual violation. No execution, recurrence reproof, new
evidence or tolerance adoption is claimed here. D61 remains source-only;
existing HOLD and caps remain unchanged. Independent Astra review: PASS for this
premise-owner/failure-boundary table only; source inspection remains read-only,
runtime UNRUN, with no broader qualification.

**Inverse-square-root caveat 2 — initial interval coverage.** Under the preceding
FF gain-two/half-mask cancellation (`t=1`), treating source decimals as exact
symbolic rationals gives

```text
delta = e/n^2 = 1/58982400000
D(v) = (v+delta)/(21*M/20+1/100000)
L = D(a), U = D(M), eta = a/M
```

Assuming **undoubled residual** variance `v` lies in `[a,M]`, the two FF profiles
satisfy the independently checked positive-denominator cross-multiplication bounds:

| Exact symbolic profile | Lower endpoint | Upper endpoint |
| --- | --- | --- |
| `a=1/5`, `M=150` | `1/788 < L < 1/787 < eta=1/750` | `19/20 < U < 20/21` |
| `a=3/4`, `M=2500` | `1/3501 < L < 1/3500 < eta=3/10000` | `19/20 < U < 20/21` |

For example, the second profile's `L<1/3500` reduces to
`3500*delta<1/100000`. Thus the assumed symbolic variance interval is not fully
covered by the helper's initial `[eta,1]`. The earlier attention profile
`(a,M)=(0.15,10)` also has `D(a)<eta`: in general this is equivalent to
`e*M/n^2 < 0.05*a*M+a*e`, and `e/n^2<0.05*a` suffices for all three profiles.
These are ideal equation/proof-gap observations only, not actual model bugs or
rounded CKKS-profile failures. Rounded masks and decimals are not certified by
these exact rational bounds. Recurrence accuracy and finite-constant/root-choice
instantiation remain unproved; no epsilon, bounds or profile change follows.

All conclusions remain conditional on bootstrap identity/gain, payload
rescale and level semantics, gamma/beta level compatibility, and physical
rotation/support mapping. In particular, `ckks.py::pc_add` warns that different
plaintext/ciphertext levels give incorrect results; source affine half
cancellation does not prove compatible levels. No CKKS qualification, protocol
activation, security claim, numerical-profile adoption or new decision follows.
Existing mask findings and logical indexing scopes remain unchanged.

#### Recovered tokenizer and attention-mask dataflow (Astra29)

Provenance: `crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25`;
the following immutable raw sources supply line citations and **zero-based**
notebook cell citations, not evidence of an evaluated run:
[`src/thor/data.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/data.py),
[`forward.ipynb`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/forward.ipynb),
[`src/thor/bert.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/bert.py),
[`src/thor/nonlinear/softmax.py`](https://raw.githubusercontent.com/crypto-starlab/THOR/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/softmax.py).

- **Tokenizer/input boundary:** `data.py:34,146–150` uses
  `AutoTokenizer.from_pretrained(self.checkpoint, model_max_length=128)` and
  paired MRPC tokenization with `max_length=128`, `padding='max_length'`,
  `truncation=True`. Neither `return_attention_mask`, padding side nor truncation
  side is explicitly set. Lines 138–143 set Torch format; lines 41–44 use
  `DataCollatorWithPadding(tokenizer=self.tokenizer)` and validation batch size 1.
  Notebook cells 13/15/18 select MRPC sample 0 and pass only `input_ids` and
  `token_type_ids` to `embed_data` (`data.py:129–132`); encrypted embeddings must
  have shape `(128,768)` (`data.py:46–51`). Fixed physical extent alone does not
  establish tokenizer-side defaults or logical token support.
- **Actual mask producer:** cell 18 passes
  `attention_mask.cpu().numpy().squeeze().T` to
  `data_encryptor.encode_attention_mask(..., level=15)`, not the standalone
  helper defined in cell 17. Cells 3/15/18 import `ThorDataEncryptor`, instantiate
  the receiver and call its instance method. Independent Astra review
  (`ses_e7a018385ffePGPIGe5noYsw8E`) verifies the pinned class method
  `ThorDataEncryptor.encode_attention_mask` (`data.py:69–88`): it validates shape
  `(128,)`, takes `k=np.count_nonzero(attention_mask)` and constructs eight
  pre-encoding messages. For `0<=i<8`, `0<=j<16`, `0<=t<128`, `0<=h<12`,
  it writes
  `M_i[2048*j+16*t+h] = 1[((16*i+j+t)%128)<k]`; the other four head lanes
  remain zero, then it calls `self.engine.encode(msg, level)`. Support is
  reconstructed from the count, not copied from the original positions. It
  preserves original support only when valid positions are exactly prefix
  `[0,k)`; arbitrary holes or left padding are not preserved. This attaches the
  mapping to the pinned class source and notebook instance target/source import,
  not independently inspected runtime package re-export resolution or evaluated
  execution. The distinct `encrypt_attention_mask` method has no modulo and is
  not this attachment.
- **Notebook consumer:** uncommented layer cells 33 onward call cell 28's
  `forward_layer`: Q/K scores, score bootstrap, then
  `thor_attention.softmax(x=sftmx_in, attention_mask=thor_attention_mask, ...)`.
  `bert.py:112` selects `he_softmax2` only at zero-based layer 2, otherwise
  `he_softmax1`; wrappers are `nonlinear/softmax.py:7–12`. Lines 22–34 first
  approximate the exponential, then apply
  `rescale(pt_ct_mult(attention_mask[i], exp_u[i]))`, before denominator
  accumulation and `rotsum`. `update_inv_D`, line 84, reapplies the mask to
  the inverse. Static `[1]*12+[0]*4` masks select head lanes, **not** token
  support. The separate top-level `ThorBert.forward` also forwards a mask
  (`bert.py:76,163`), but is not this notebook's layer runner.
- **Distinct plaintext comparison:** cell 21's helper calls
  `get_extended_attention_mask(attention_mask,768)`, adds its result to scores
  **before** Torch softmax. The exact additive sentinel and upstream defaults
  remain unbound; this is not the HE post-exponential multiplication path.
  No equivalence or new reference identity is adopted. Null execution counts
  and absent saved outputs still do not prove evaluation of either runner.

**Unresolved gates specific to this boundary:** bind tokenizer revision/defaults
and establish the valid-prefix condition before claiming support fidelity;
reconcile the two masking paths without silently adopting numerical equivalence.
The verified class-method mapping makes this source plaintext-mask boundary
communicate count-derived support to the evaluator.
An explicit leakage/view contract must be approved
before protected execution; physical length 128 alone does not erase `k`.
This is a source dataflow observation, not a deployed privacy-attack claim or
evidence that the project declares actual `k` public. Existing coefficient,
epsilon, helper-profile and evaluated-context qualifications remain unchanged.

#### Recovered softmax and inverse contracts; source attachment limits

This documentation-only recovery records facts independently reviewed by Astra
(`ses_e7a173877ffedxSMdXVBItS2A`), pinned to
`crypto-starlab/THOR@b513c06aaabccc04b560721e35e9a98008428c25`:
`src/thor/nonlinear/softmax.py`, `src/thor/nonlinear/polynomial.py`,
`src/thor/bert.py` and `forward.ipynb`. No execution or source adoption occurred
in this update; prior missing-source claims below are historical, not new gaps.

The softmax helper reverses each 16-entry literal coefficient list before the
increasing-power polynomial evaluator. With that effective polynomial `P_a`,
the initial exponential surrogate is `E_a(u)=P_a(u-m/d)^2` (`n=2`). Profiles:

| Helper | `m` | `d` | `l` | Initial inverse epsilon | Updates |
| --- | --- | --- | --- | --- | --- |
| `he_softmax1` | `(-27.2493+21.72692)/2` | 32 | 2 | `2^-11` | One final update |
| `he_softmax2` | 0 | 64 | 4 | `2^-18` | One ordinary, then one final update |

BERT selects profile 2 only at zero-based layer 2. Notebook plotting uses nominal
`u=s/d`; this is not proof of upstream score scaling or encoded payload identity.

For inverse input `D`, initialize `A=1`, `B=D`, `delta=1`, `e=eps`.
Each iteration sets `k=2/(1+e)` and uses the **old** `B` in both products:
`A'=A*(2*delta/k-B)`, `B'=B*(2*delta/k-B)`,
`delta'=delta^2/k^2`, `e'=k*e*(2-k*e)`.
The helper returns raw `A`, not normalized `A/delta`. Initial and ordinary
inversions use `alpha=0.01`; final inversion uses `alpha=0.001` (correcting the
earlier review's initial-`0.001` statement). Subsequent inversions receive
`eps=e_previous/256`; their state is not the initial epsilon reused unchanged.

Only under **explicit ideal-real row-scaling assumptions**, let `p` be the token
mask and start `v0=p*E`. With `A,delta` from the preceding inversion, each update
is `v <- [2*max(floor(1/(2*delta)),1)*v*p*A]^2`, followed by reinversion of the
updated row sum. Final factor is `2*(floor(1/(2*delta))+1)*v*A`, using that final
inverse state. Thus active-lane `E^2` / `E^4` ratios describe the respective
profiles only where shared factors are defined and nonzero; preceding row
moments still affect the state. The earlier single-pass model `p*E*R(sum)` is
not a valid attachment to this source path. Source floating division and integer
conversion are not identical to exact-real floor arithmetic. Branch-specific
levels, conjugation, bootstrap behavior and `1/1.2` scaling cannot be erased by
this conditional algebra; no float/engine equivalence is established.

The reviewed two-active-lane nominal-score case `{0,1}` yields initial sums
`D<9.68e-6<2^-11` for profile 1 and `D<2e-8<2^-18` for profile 2, **outside**
the inverse's claimed `[eps,1]` domain. All-zero support gives `D=0`, also outside.
These are domain constraints, not in-domain approximation counterexamples or
mapped floating-point/engine failures. No out-of-domain behavior is adopted.

The sparse-support collision `{0}` versus `{127}` is attached to the pinned
class method `ThorDataEncryptor.encode_attention_mask` (`data.py:69–88`):
distinct same-count masks produce identical pre-encoding messages. Notebook
cells 3/15/18 identify the import, receiver and instance call, not cell 17's
standalone helper. Runtime package re-export resolution and evaluated execution
remain independently uninspected; this is not encoded-object equality or HE
behavior. Plaintext Torch dtype, additive sentinel, revision and all-masked
convention remain unbound: finite sentinels, infinity and underflow need not
have identical semantics.

Approval remains required before any new oracle, scale/refresh change,
out-of-domain policy, numerical profile, clear-softmax replacement or engine
activation. Recovery does not close evaluated-run identity or HE numeric gates.

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
regression tests are committed in **`560522f6`**. The retained
[ordinary output](../evidence/thor-literal-reconciliation.json) is
`docs/evidence/thor-literal-reconciliation.json`, unchanged from the original
`target/thor-literal-reconciliation-r1.json`. This documentation update reads
that output; it does not rerun the study or add an evidence archive.
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

## Source-to-codec RHS attachment and remaining gain obligation

**Conditional ideal-slot derivation, not CKKS qualification.** This D61
source-contract addendum narrows historical source gaps above without changing
their approved scope. All upstream citations pin `crypto-starlab/THOR` commit
`b513c06aaabccc04b560721e35e9a98008428c25`; it remains a specification/oracle
input only, not an adopted numerical profile, backend or runtime dependency.

### Producer, support selection and real shared RHS

[Softmax wrappers and producer][rhs-softmax] select `he_softmax1` for eleven
default layers: `(n,l)=(2,2)`, bounds `[-27.2493,21.72692]`, epsilon `2^-11`,
`output_alpha=0.01`. Only zero-based layer 2 selects `he_softmax2`, with
`(n,l)=(2,4)`, bounds `[-70,70]`, epsilon `2^-18`, `output_alpha=0.01`.
These are alternatives, not consecutive stages. Each `he_exp1/2` evaluates its
reversed real 16-coefficient polynomial and then squares once (`n=2`).
The `l=2` path has zero nonfinal inverse updates; `l=4` has one, despite the
source's fixed zero-iteration comment. Each has one final update. Metadata and
levels control bootstrap branches; no fixed branch schedule is established.

Both final updates multiply the inverse by the flat 2,048-entry numeric mask
`([1]*12+[0]*4)*128`; the initial inverse mask has 32,768 entries.
The [THOR subclass][rhs-subclass] overrides neither `cm_mult` nor `mc_mult`.
Inherited [`cm_mult`, lines 3505–3509][rhs-message] delegates to `mc_mult`;
lines 3455–3461 compute `np.array(m)*sqrt(deviations[ct.level_calc+1])`
and call `self.encode(m,15)`. [`encode`, lines 957–968][rhs-padding] defaults
to `padding=True`; `padding`, lines 945–954, uses `np.pad` with zero suffix
to `num_slots`. Thus, for the flat numeric mask and 32,768-slot premise,
only the leading 2,048-slot block is selected: **no message-slot tiling**.
The subsequent `tile_unsigned` at line 3463 acts after encoding, expanding
polynomial coefficients across RNS modulus channels, not repeating message slots.
Before rounding and rescale, its ideal plaintext coefficient gain is
`Delta*d_15*sqrt(d_(t+1))` for `t=ct.level_calc`; this does not certify amplitude.
Nested-list inputs are outside this padding argument.

For `B=2048`, let `N=32768` denote message slots here (not ring degree),
`R=R_B`, and `T=sum_(a=0)^15 R^a`. Since `R^16=I`, `R*T=T` by reindexing.
Let `w_ij` be the final pre-copy product for `0<=i<4`, `0<=j<16`.
The producer's rotsum and conjugate extraction give exactly, under ideal slots,

```text
P_(16*i+j)    = T*w_ij + conj(T*w_ij)     = 2*Re(T*w_ij)
P_(64+16*i+j) = i*(conj(T*w_ij)-T*w_ij)   = 2*Im(T*w_ij).
```

Consequently all 128 returned ideal messages are real and 2,048-periodic,
even when `w_ij` is complex. The padding proof attaches structural block
selection to this copying argument; neither establishes correct amplitudes.
The actual attention mask has zero head holes `12..15`. Under intended support,
ideal masking, block rotations and extraction preserve those holes; neither
stored CKKS plaintexts nor encrypted operations are asserted exactly error-free.

In [context multiplication][rhs-bert], the same `P_n` multiplies both components
of each packed value pair `v_i+i*v_(i+2)`. For real `a,b,P_n`,
`i*conj((a+i*b)*P_n)=(b+i*a)*P_n`; different `P_n` need **not** be equal.
This closes the earlier REAL/SHARED RHS premise in the conditional ideal model.
For RHS `r+i*epsilon` and packed `z`, actual minus intended swapped product is
`i*conj(z*(r+i*epsilon))-i*conj(z)*(r+i*epsilon)=2*epsilon*conj(z)`.
That is an algebraic swap defect, not a privacy-attack claim or CKKS error bound.

### Actual caller gains are still conditional

`ThorBertAttention.forward` calls scores with `bootstrap=False`, `scale=1`,
`rescale=False`. The score helper extracts `2*Re`/`2*Im`; its caller repacks,
bootstraps and performs a **second unhalved extraction**. Context uses defaults
`scale=1`, `complex=True`, `rescale=False`, returns complex ciphertexts, and
the caller bootstraps those outputs. A payload-identity bootstrap is a premise,
not a recovered guarantee; the reached bootstrap body is PyArmor-opaque.
No reverse engineering or execution is authorized to fill that boundary.

[`ThorModelEncoder.encode_att`][rhs-encoder] explicitly sets Q/V weight and bias
scale to 1 and K weight and bias scale to `kappa=1/1024` at layer 2, otherwise
`1/512`; `_encode_w_qkv`'s unused default `1/256` is not the caller contract.
Only **if all projection, helper, mask-gain and normalization contracts hold**
does the nominal score input reduce to `4*kappa*Q*K^T`. This is not an
unconditional score identity or a whole-model probability claim.

Literal `scale_pt` acts on **zero diagonal only**. Scores halve it only when
`bootstrap=True`; context halves it only when `complex=False` and still returns
complex outputs. Neither flag uniformly scales the complete routed product.
Mask flags equal to 1 alone cannot prove exact dense equality; any earlier
nominal `(4,4)` / `(1,1)` gain pairs remain conditional, not qualified gains.

### First missing quantitative contract: mask/complement gain

[`ThorLinearEvaluator.pre_encode_masks`][rhs-linear] encodes shared masks at
level 15. Zero-diagonal `scale_pt` is encoded at the current operand level
in each score/context helper. Runtime level alignment does not re-encode masks.
For nominal `scale_bits=41`, put `Delta=K=2^41` and `d_L=deviations[L]`.
Writing `F` for the source forward-normalized encoding transform, the ideal
coefficient expression is `Delta*d_L*F(m)`, plus encoding/rounding residuals.
Thus unit-mask gain is `e=K*d_15`, while zero-diagonal gains are separately
`e0,s=K*d_(L_s)` and `e0,c=K*d_(L_c)` at actual score/context levels.
By contrast, `mult_int_scalar_triplet` multiplies by integer `K`, not an
encoded one. [Rescale][rhs-rescale] divides by the actual removed odd prime
`q_L`, with rounding, not nominal `K`; channel alignment remains a premise.

The engine's [`make_adjustments_and_corrections`][rhs-deviations] computes
`d_0=1`, `d_(r+1)=d_r^2*(B_r/q_r)^2`, `B_r=2^round(log2(q_r))`.
Actual NumPy float64 evaluation, calibration and cached/loaded context matter.
In particular `d_15` depends on fifteen preceding 59-bit bootstrap primes or
the loaded context. Neither exact `e=K` nor closeness is established here.
Exact-rational non-equality would not prove a stored-float mismatch.

For sum `S` of explicit masks, complement algebra is
`K-e*S=e*(1-S)+(K-e)`. The last term is a **routing residual** which can remain
where the intended complement is zero, not a uniform whole-product scale or
confidentiality leak. Conditional on ideal encoding and arithmetic before the
common final rescale, define `D` as the intended routed product and `P_0` as its
zero-diagonal contribution. For each nonzero diagonal `l`, `P_l` is the complete
per-diagonal product tuple across all source ciphertexts; `T_l` sums the actual
complement-column routes, including delayed rotation and component swap.
For the 63 score or 127 context nonzero diagonals, with the appropriate separate
`e0`, the decomposition is

```text
Z = e*D + (e0-e)*P_0 + (K-e)*sum_(l>0) T_l(P_l).
```

No isometry, operator norm or routing multiplicity bound for `T_l` is assumed.
If `|log(d_15)|<=A`, `A>=0`, then `|K-e|<=K*expm1(A)`; zero-diagonal mismatch
needs its own bound, as do encoding residuals. Required next proof inputs are
actual primes/stored deviations, current levels, channel alignment, rounding,
noise, true-`q_L` rescale, downstream normalization and bootstrap gain/error.

**CLOSED under stated ideals:** real/periodic RHS, shared paired operand and
flat-mask leading-block support selection. **OPEN:** amplitude/gain and all
error contracts, evaluated keys/context, authentication, CKKS/security,
funding/admission and execution. No new numerics, backend adoption, D25 extension,
Native causal claim or implementation change follows; existing HOLD remains.

[rhs-softmax]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/nonlinear/softmax.py#L7-L119
[rhs-subclass]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/ckks.py#L1-L320
[rhs-message]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/ckks_engine.py#L3455-L3509
[rhs-padding]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/ckks_engine.py#L945-L968
[rhs-bert]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/bert.py#L112-L610
[rhs-encoder]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/model_encoder.py#L33-L45
[rhs-linear]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/src/thor/linear.py#L225-L320
[rhs-rescale]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/ckks_engine.py#L1957-L2024
[rhs-deviations]: https://github.com/crypto-starlab/THOR/blob/b513c06aaabccc04b560721e35e9a98008428c25/liberate/src/liberate/fhe/ckks_engine.py#L363-L375
