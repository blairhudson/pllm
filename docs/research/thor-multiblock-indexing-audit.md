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
